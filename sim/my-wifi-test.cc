#include "ns3/abort.h"
#include "ns3/ampdu-subframe-header.h"
#include "ns3/boolean.h"
#include "ns3/command-line.h"
#include "ns3/double.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/mac48-address.h"
#include "ns3/mobility-helper.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/socket.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-phy-state-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/spectrum-wifi-helper.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

using namespace ns3;

namespace
{

// ---------------------------------------------------------------------------
// Simulates one Wi-Fi deployment and reports what a newly arriving client
// would observe before joining, and the throughput it gets after joining a
// given AP.
//
// Command-line inputs:
//   nAPs, nSTAs        number of APs and of background stations
//   hotspotAPs         which APs background stations crowd around
//   targetAP           the AP the candidate joins
//   candidateSeed      seed for the candidate's draws listed below (or
//                      candidateX/Y to place the candidate by hand)
//   topologySeed       seed for the deployment draws listed below
//   rngSeed            seed for ns-3's radio randomness: fading, backoff, rate
//                      control, packet timing
//   outDir, runTag     where the outputs go
//
// Drawn inside from the seeds:
//   AP positions        not drawn: a fixed lattice set by nAPs
//   AP channels         topologySeed
//   station positions   topologySeed, denser around hotspotAPs
//   station uplink load topologySeed, one log-normal draw per station
//   candidate position  candidateSeed, uniform over the area stations occupy
//   link shadowing      topologySeed; the candidate's own links candidateSeed
//
// What happens:
//   0.5 s        background stations start sending uplink traffic, each at
//                its own drawn rate
//   2.0-8.0 s    the candidate listens passively on every occupied channel
//                and records every frame it hears
//   8.0 s        the candidate starts joining targetAP, and from then until
//                20 s offers more uplink traffic than its link can carry
//
// Outputs, in outDir/runTag/:
//   observation.csv   every frame from a transmission wholly inside the
//                     listening window
//   chanbusy.csv      per-channel busy fraction, one row per channel per 1 ms
//   metadata.json     parameters, the candidate's throughput (the label),
//                     association outcome, and simulator ground truth
//
// Nothing before 8 s depends on targetAP, so running the same inputs once per
// AP gives the true throughput of every AP from one shared observation.
// ---------------------------------------------------------------------------
constexpr uint16_t kBasePort = 8000;
constexpr uint32_t kPacketSize = 1250;
constexpr double kApSpacingM = 20.0; // distance between neighbouring APs

// Background traffic needs ~1.5 s to reach steady state: association
// completes quickly but Minstrel-HT converges only after heavy retransmission.
constexpr double kBackgroundStartS = 0.5;
constexpr double kObservationStartS = 2.0;
constexpr double kCandidateStartS = 8.0; // listening ends and joining starts here
constexpr double kSimulationStopS = 20.0;

constexpr double kCandidateOfferedMbps = 100.0; // saturates the candidate's link
// Each background station's uplink load is one log-normal draw, clamped at the cap.
constexpr double kBgLoadMedianMbps = 2.25;
constexpr double kBgLoadSigmaLog = 0.9;
constexpr double kBgLoadCapMbps = 25.0;

constexpr double kHotspotDensity = 4.0; // A station is kHotspotDensity times more 
// likely to land in a hotspot disc than anywhere else of equal area. 
constexpr double kHotspotRadiusM = kApSpacingM / 2.0;

// Time resolution of chanbusy.csv. The listening window is cut into buckets of
// this length, and each row reports the fraction of one bucket during which the
// candidate's radio on that channel sensed the medium busy. 1 ms is the
// resolution at which real clients export channel busy time.
constexpr double kBusyBucketS = 0.001;
constexpr double kPathLossExponent = 3.0;
// Log-normal shadowing sigma. One draw per unordered node pair, fixed for the
// whole run, applied both by the association model and by the simulated
// channel, so the power a station used to pick its AP is the power the channel
// then delivers.
constexpr double kLinkShadowingDb = 5.0;
constexpr double kPi = 3.14159265358979323846;

// Four non-overlapping 20 MHz channels in the 5 GHz band; each AP draws one
// uniformly at random. The propagation reference loss below is free-space loss
// at 1 m at 5.15 GHz, within 0.2 dB of all four, so it must change if these
// channels do.
const std::vector<uint8_t> kApChannels = {36, 40, 44, 48};

// A channel no AP occupies. The candidate's association radio parks here until
// 8 s so it hears nothing and consumes no fading RNG draws, which is what keeps
// everything before 8 s independent of targetAP. Scanner radios move here when
// the listening window ends.
constexpr uint8_t kParkChannel = 149;

struct RunConfig
{
    uint32_t nStas{12};
    uint32_t nAps{3};
    uint32_t targetAp{0};
    // NaN when not given, in which case the position is drawn from candidateSeed.
    double candidateX{std::numeric_limits<double>::quiet_NaN()};
    double candidateY{std::numeric_limits<double>::quiet_NaN()};
    uint32_t candidateSeed{0};
    std::string hotspotApsText{"none"};
    std::vector<uint32_t> hotspotAps;
    uint32_t topologySeed{1};
    uint32_t rngSeed{0};
    std::string outDir{"runs"};
    std::string runTag;
    bool captureObservation{true};
};

struct Topology
{
    std::vector<Vector> apPositions;
    std::vector<Vector> staPositions;
    std::vector<uint32_t> staAssociatedAp; // What AP is each STA associated with
    Vector candidatePosition;
    std::vector<double> candidateApDistance;
    double hotspotProbability{0.0};
    // Shadowing in dB between every pair of nodes; positive means a stronger
    // signal. shadowingDb[i][j] == shadowingDb[j][i]. Nodes are numbered APs
    // first, then background stations, then the candidate: the same order
    // main() adds them to allNodes, which is how the channel looks them up.
    std::vector<std::vector<double>> shadowingDb;
};

std::vector<uint32_t>
ParseHotspotAps(const std::string& text, uint32_t nAps)
{
    if (text.empty() || text == "none")
    {
        return {};
    }

    std::vector<uint32_t> hotspotAps;
    std::istringstream list(text);
    std::string token;
    while (std::getline(list, token, ','))
    {
        uint32_t ap;
        std::string trailing;
        std::istringstream value(token);
        NS_ABORT_MSG_IF(!(value >> ap) || (value >> trailing),
                        "hotspotAPs must be 'none' or comma-separated AP indices");
        NS_ABORT_MSG_IF(ap >= nAps, "every hotspotAPs entry must be less than nAPs");
        NS_ABORT_MSG_IF(std::find(hotspotAps.begin(), hotspotAps.end(), ap) !=
                            hotspotAps.end(),
                        "hotspotAPs cannot contain duplicate AP indices");
        hotspotAps.push_back(ap);
    }

    return hotspotAps;
}

// ---------------------------------------------------------------------------
// Command-line options set the deployment, the candidate, the seeds and the
// output location.
// ---------------------------------------------------------------------------
RunConfig
ParseRunConfig(int argc, char* argv[])
{
    RunConfig config;
    CommandLine cmd(__FILE__);

    cmd.AddValue("nSTAs", "total number of background stations", config.nStas);
    cmd.AddValue("nAPs", "number of APs in the fixed two-dimensional layout", config.nAps);
    cmd.AddValue("targetAP", "index of the AP the candidate joins", config.targetAp);
    cmd.AddValue("candidateX",
                 "candidate x coordinate in metres; omit both coordinates to draw "
                 "the position from candidateSeed",
                 config.candidateX);
    cmd.AddValue("candidateY", "candidate y coordinate in metres", config.candidateY);
    cmd.AddValue("candidateSeed",
                 "seeds the candidate's shadowing links, and its position when "
                 "candidateX/candidateY are omitted",
                 config.candidateSeed);
    cmd.AddValue("hotspotAPs",
                 "comma-separated APs around which background stations gather; 'none' disables",
                 config.hotspotApsText);
    cmd.AddValue("topologySeed",
                 "seeds the whole deployment: station placement and offered "
                 "loads, AP channel assignment, and the shadowing between "
                 "every pair of non-candidate nodes",
                 config.topologySeed);
    cmd.AddValue("rngSeed",
                 "ns-3 PHY/MAC seed; 0 chooses and records a random seed",
                 config.rngSeed);
    cmd.AddValue("outDir", "directory under which the run directory is written", config.outDir);
    cmd.AddValue("runTag", "run directory name; empty derives one from the seeds and target AP", config.runTag);
    cmd.AddValue("captureObs",
                 "write observation.csv and chanbusy.csv; 0 skips them without "
                 "changing the simulation",
                 config.captureObservation);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(config.nAps == 0, "nAPs must be at least 1");
    NS_ABORT_MSG_IF(config.nAps > 200, "nAPs must be at most 200");
    NS_ABORT_MSG_IF(config.nStas == 0, "nSTAs must be at least 1");
    NS_ABORT_MSG_IF(config.targetAp >= config.nAps, "targetAP must be less than nAPs");
    NS_ABORT_MSG_IF(config.topologySeed == 0, "topologySeed must be positive");
    NS_ABORT_MSG_IF(std::isnan(config.candidateX) != std::isnan(config.candidateY),
                    "give both candidateX and candidateY, or neither");
    NS_ABORT_MSG_IF(std::isnan(config.candidateX) && config.candidateSeed == 0,
                    "candidateSeed must be positive when candidateX/candidateY are omitted");
    config.hotspotAps = ParseHotspotAps(config.hotspotApsText, config.nAps);

    if (config.rngSeed == 0)
    {
        std::random_device randomDevice;
        config.rngSeed = 1 + randomDevice() % 2147483646u;
    }

    if (config.runTag.empty())
    {
        // candidateSeed belongs here: without it two candidate positions drawn
        // from one deployment derive the same tag and overwrite each other.
        config.runTag = "topology_" + std::to_string(config.topologySeed) + "_candidate_" +
                        std::to_string(config.candidateSeed) + "_seed_" +
                        std::to_string(config.rngSeed) + "_ap_" +
                        std::to_string(config.targetAp);
    }

    return config;
}

// ---------------------------------------------------------------------------
// APs sit on a fixed triangular lattice in at most two rows, kApSpacingM
// between neighbours, with AP 0 at the origin. The bottom row holds half the
// APs, rounded up but never fewer than two, filled left to right; the rest
// fill the row above. So 2 APs form one row, 3 a triangle of 2 below and 1
// above, and 4, 6 and 8 two rows of 2, 3 and 4.
//
//     3 APs:     2           6 APs:     3   4   5
//              0   1                  0   1   2
// ---------------------------------------------------------------------------
std::vector<Vector>
BuildApPositions(uint32_t nAps)
{
    const uint32_t perRow = std::max<uint32_t>(2, (nAps + 1) / 2);
    const double rowPitch = kApSpacingM * std::sqrt(3.0) / 2.0;

    std::vector<Vector> positions;
    positions.reserve(nAps);
    for (uint32_t index = 0; index < nAps; ++index)
    {
        uint32_t row = index / perRow;
        uint32_t column = index % perRow;
        positions.emplace_back((column + 0.5 * row) * kApSpacingM, row * rowPitch, 0.0);
    }

    return positions;
}

// A background station joins the AP with the strongest received power:
// log-distance path loss plus that link's shadowing, taken from the same
// shadowingDb the channel uses. Transmit power and reference loss are equal
// for every AP, so they are left out of the comparison.
uint32_t
StrongestAp(double x, double y, const std::vector<Vector>& apPositions,
            const std::vector<double>& shadowingToApsDb)
{
    uint32_t best = 0;
    double bestPowerDb = -std::numeric_limits<double>::infinity();

    for (uint32_t ap = 0; ap < apPositions.size(); ++ap)
    {
        double distance =
            std::max(1.0, std::hypot(x - apPositions[ap].x, y - apPositions[ap].y));
        double powerDb =
            -10.0 * kPathLossExponent * std::log10(distance) + shadowingToApsDb[ap];
        if (powerDb > bestPowerDb)
        {
            bestPowerDb = powerDb;
            best = ap;
        }
    }

    return best;
}

// ---------------------------------------------------------------------------
// Builds the parts of the deployment that are fixed before simulation starts:
//   1. AP positions on the lattice.
//   2. The placement area: the AP footprint plus a border.
//   3. Shadowing for every pair of nodes.
//   4. The candidate's position, uniform over the area unless given.
//   5. Background station positions, uniform over the area except that a
//      point inside a hotspot disc is kHotspotDensity times as likely as one
//      outside. Each station joins its StrongestAp.
// ---------------------------------------------------------------------------
Topology
BuildTopology(const RunConfig& config)
{
    Topology topology;
    topology.apPositions = BuildApPositions(config.nAps);

    double minX = topology.apPositions[0].x;
    double maxX = topology.apPositions[0].x;
    double minY = topology.apPositions[0].y;
    double maxY = topology.apPositions[0].y;
    for (const auto& position : topology.apPositions)
    {
        minX = std::min(minX, position.x);
        maxX = std::max(maxX, position.x);
        minY = std::min(minY, position.y);
        maxY = std::max(maxY, position.y);
    }

    // Equal to the hotspot radius, so every hotspot disc lies inside the box.
    double border = kHotspotRadiusM;
    std::mt19937 placementRng(config.topologySeed);

    // Shadowing has its own random streams, so it never shifts placement draws.
    const uint32_t nNodes = config.nAps + config.nStas + 1;
    const uint32_t candidateIndex = nNodes - 1;
    std::mt19937 shadowRng(config.topologySeed * 2654435761u + 4u);
    // The candidate's links use candidateSeed, so moving the candidate leaves
    // every background pair unchanged.
    std::mt19937 candidateShadowRng(
        (config.candidateSeed ? config.candidateSeed : config.topologySeed) *
            2654435761u +
        7u);
    topology.shadowingDb.assign(nNodes, std::vector<double>(nNodes, 0.0));
    // Each stream needs its own distribution object. This is a C++ standard
    // library pitfall, not an ns-3 one: std::normal_distribution generates
    // values in pairs, keeps the spare inside the distribution object, and
    // returns it on the next call whichever engine that call passes. With
    // one shared object, whenever the number of background pairs was odd,
    // the candidate's link to AP 0 received the background stream's
    // leftover value, identical for every candidateSeed. Nothing crashes
    // and every value is a valid draw, so it only shows up as candidate
    // positions that fail to vary. (The uniform distributions below are
    // shared safely: libstdc++ keeps no state in them.)
    std::normal_distribution<double> shadowDraw(0.0, kLinkShadowingDb);
    std::normal_distribution<double> candidateShadowDraw(0.0, kLinkShadowingDb);
    for (uint32_t i = 0; i < candidateIndex; ++i)
    {
        for (uint32_t j = i + 1; j < candidateIndex; ++j)
        {
            double draw = shadowDraw(shadowRng);
            topology.shadowingDb[i][j] = draw;
            topology.shadowingDb[j][i] = draw;
        }
    }
    for (uint32_t i = 0; i < candidateIndex; ++i)
    {
        double draw = candidateShadowDraw(candidateShadowRng);
        topology.shadowingDb[i][candidateIndex] = draw;
        topology.shadowingDb[candidateIndex][i] = draw;
    }
    std::uniform_real_distribution<double> uniformX(minX - border, maxX + border);
    std::uniform_real_distribution<double> uniformY(minY - border, maxY + border);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // The candidate lands uniformly anywhere in the same box as the stations,
    // on its own stream so that moving it never shifts a background draw.
    if (std::isnan(config.candidateX))
    {
        std::mt19937 candidateRng(config.candidateSeed);
        double x = uniformX(candidateRng);
        double y = uniformY(candidateRng);
        topology.candidatePosition = Vector(x, y, 0.0);
    }
    else
    {
        topology.candidatePosition = Vector(config.candidateX, config.candidateY, 0.0);
    }

    // The chance a station lands in some hotspot disc, set so that density
    // inside a disc is kHotspotDensity times the density outside. Discs never
    // overlap, so their areas simply add.
    const double boxArea = (maxX - minX + 2.0 * border) * (maxY - minY + 2.0 * border);
    const double discArea =
        config.hotspotAps.size() * kPi * kHotspotRadiusM * kHotspotRadiusM;
    topology.hotspotProbability =
        config.hotspotAps.empty()
            ? 0.0
            : kHotspotDensity * discArea /
                  (boxArea + (kHotspotDensity - 1.0) * discArea);
    std::uniform_int_distribution<std::size_t> hotspotChoice(
        0,
        config.hotspotAps.empty() ? 0 : config.hotspotAps.size() - 1);

    topology.staPositions.reserve(config.nStas);
    topology.staAssociatedAp.reserve(config.nStas);
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        double x;
        double y;
        if (unit(placementRng) < topology.hotspotProbability)
        {
            double radius = kHotspotRadiusM * std::sqrt(unit(placementRng));
            double angle = 2.0 * kPi * unit(placementRng);
            uint32_t hotspotAp = config.hotspotAps[hotspotChoice(placementRng)];
            const Vector& centre = topology.apPositions[hotspotAp];
            x = centre.x + radius * std::cos(angle);
            y = centre.y + radius * std::sin(angle);
        }
        else
        {
            // Discs got their share in the branch above, so reject points
            // that land inside one.
            bool insideDisc;
            do
            {
                x = uniformX(placementRng);
                y = uniformY(placementRng);
                insideDisc = false;
                for (uint32_t hotspotAp : config.hotspotAps)
                {
                    const Vector& centre = topology.apPositions[hotspotAp];
                    if (std::hypot(x - centre.x, y - centre.y) <= kHotspotRadiusM)
                    {
                        insideDisc = true;
                        break;
                    }
                }
            } while (insideDisc);
        }

        topology.staPositions.emplace_back(x, y, 0.0);
        topology.staAssociatedAp.push_back(StrongestAp(
            x, y, topology.apPositions, topology.shadowingDb[config.nAps + sta]));
    }

    topology.candidateApDistance.reserve(config.nAps);
    for (const auto& apPosition : topology.apPositions)
    {
        topology.candidateApDistance.push_back(
            std::hypot(topology.candidatePosition.x - apPosition.x,
                       topology.candidatePosition.y - apPosition.y));
    }

    return topology;
}

// ns-3's ChannelSettings attribute string for a 20 MHz 5 GHz channel:
// {channel number, width in MHz, band, index of the primary 20 MHz channel}.
std::string
ChannelSettings(uint8_t number)
{
    return "{" + std::to_string(number) + ", 20, BAND_5GHZ, 0}";
}

std::string
MacToString(Mac48Address address)
{
    std::ostringstream stream;
    stream << address;
    return stream.str();
}

// ---------------------------------------------------------------------------
// Recording what the candidate observes while listening.
// ---------------------------------------------------------------------------

// Busy time on one channel, summed into kBusyBucketS buckets. UpdateCCARecord()
// takes one busy interval, in seconds from the window start, and splits it
// across the buckets it overlaps; CCABusyFraction() gives the share of one
// bucket that was busy.
struct CCABusyRecord
{
    std::vector<double> busySeconds;

    void UpdateCCARecord(double start, double end)
    {
        if (end <= start)
        {
            return;
        }

        std::size_t first = static_cast<std::size_t>(start / kBusyBucketS);
        std::size_t last = static_cast<std::size_t>(std::ceil(end / kBusyBucketS)) - 1;
        if (busySeconds.size() <= last)
        {
            busySeconds.resize(last + 1, 0.0);
        }

        for (std::size_t bucket = first; bucket <= last; ++bucket)
        {
            double overlapStart = std::max(start, bucket * kBusyBucketS);
            double overlapEnd = std::min(end, (bucket + 1) * kBusyBucketS);
            if (overlapEnd > overlapStart)
            {
                busySeconds[bucket] += overlapEnd - overlapStart;
            }
        }
    }

    double CCABusyFraction(std::size_t bucket, double bucketDuration) const
    {
        // busySeconds stops at the last bucket that saw any busy time.
        if (bucket >= busySeconds.size())
        {
            return 0.0;
        }
        return busySeconds[bucket] / bucketDuration;
    }
};

// Records what the candidate hears while listening. On each listening radio,
// three ns-3 traces are tapped:
//   PhyRxPayloadBegin  when a decodable transmission started and will end
//   MonitorSnifferRx   every frame the radio decodes        -> observation.csv
//   PHY state          when carrier sense reports busy       -> chanbusy.csv
// Both files keep only what happens inside the listening window, with times in
// seconds from its start. Radios are numbered by their position in the list
// given to the constructor.
class ObservationRecorder
{
  public:
    // Opens both files in runDir and taps the traces of every listening radio.
    // channels[i] is the channel devices[i] listens on.
    ObservationRecorder(const std::string& runDir,
                        const std::vector<Ptr<WifiNetDevice>>& devices,
                        const std::vector<uint32_t>& channels)
    {
        frames.open(runDir + "/observation.csv");
        channelBusy.open(runDir + "/chanbusy.csv");
        NS_ABORT_MSG_IF(!frames.is_open(), "could not open observation.csv");
        NS_ABORT_MSG_IF(!channelBusy.is_open(), "could not open chanbusy.csv");
        // 9 significant digits write times in the 6 s window to 10 ns. The
        // default of 6 would round them to 10 us.
        frames << std::setprecision(9);
        channelBusy << std::setprecision(9);
        frames << "tx_start,tx_end,freq_mhz,bssid,ta,cat,is_beacon,retry,len,signal_dbm,"
                  "duration_us,rate_mbps\n";
        channelBusy << "start,end,channel,freq_mhz,busy_frac\n";

        radios.resize(devices.size());
        for (std::size_t radio = 0; radio < devices.size(); ++radio)
        {
            radios[radio].channel = channels[radio];
            Ptr<WifiPhy> phy = devices[radio]->GetPhy();
            phy->TraceConnectWithoutContext(
                "PhyRxPayloadBegin",
                MakeCallback(&ObservationRecorder::OnPayloadBegin, this, radio));
            phy->TraceConnectWithoutContext(
                "MonitorSnifferRx",
                MakeCallback(&ObservationRecorder::OnFrame, this, radio));
            phy->GetState()->TraceConnectWithoutContext(
                "State",
                MakeCallback(&ObservationRecorder::OnPhyState, this, radio));
        }
    }

    // The traces hold a pointer to this object, so it must not be copied.
    ObservationRecorder(const ObservationRecorder&) = delete;
    ObservationRecorder& operator=(const ObservationRecorder&) = delete;

    // Writes chanbusy.csv and closes both files. Call after Simulator::Run().
    void Close()
    {
        frames.close();

        std::size_t nBuckets = static_cast<std::size_t>(std::ceil(kWindowS / kBusyBucketS));
        for (const Radio& radio : radios)
        {
            for (std::size_t bucket = 0; bucket < nBuckets; ++bucket)
            {
                double start = bucket * kBusyBucketS;
                double end = std::min(kWindowS, (bucket + 1) * kBusyBucketS);
                channelBusy << start << ',' << end << ',' << radio.channel << ','
                            << (5000 + 5 * radio.channel) << ','
                            << radio.busy.CCABusyFraction(bucket, end - start) << '\n';
            }
        }
        channelBusy.close();
    }

  private:
    static constexpr double kWindowS = kCandidateStartS - kObservationStartS;

    struct Radio
    {
        uint32_t channel{0};
        CCABusyRecord busy;
        // Start and end of the transmission the radio is receiving.
        Time txStart;
        Time txEnd;
    };

    std::ofstream frames;
    std::ofstream channelBusy;
    std::vector<Radio> radios;

    // A listening radio has decoded a transmission's preamble and header and
    // begins its payload. The transmission started one preamble-and-header
    // earlier and ends when the payload does.
    void OnPayloadBegin(std::size_t radio, WifiTxVector txVector, Time payloadDuration)
    {
        Time now = Simulator::Now();
        radios[radio].txStart = now - WifiPhy::CalculatePhyPreambleAndHeaderDuration(txVector);
        radios[radio].txEnd = now + payloadDuration;
    }

    void OnFrame(std::size_t radio,
                 Ptr<const Packet> packet,
                 uint16_t channelFreqMhz,
                 WifiTxVector txVector,
                 MpduInfo aMpdu,
                 SignalNoiseDbm signalNoise,
                 uint16_t /*staId*/)
    {
        // Only frames whose whole transmission lies inside the window.
        Time txStart = radios[radio].txStart;
        Time txEnd = radios[radio].txEnd;
        if (txStart < Seconds(kObservationStartS) || txEnd > Seconds(kCandidateStartS))
        {
            return;
        }

        // Each frame of an aggregate arrives with a 4-byte delimiter in front
        // and, on every frame but the last, padding behind to a 4-byte
        // boundary. Strip both, so the packet starts at the MAC header and its
        // size is the frame length the delimiter records.
        Ptr<Packet> copy = packet->Copy();
        if (aMpdu.type != NORMAL_MPDU)
        {
            AmpduSubframeHeader delimiter;
            copy->RemoveHeader(delimiter);
            copy->RemoveAtEnd(copy->GetSize() - delimiter.GetLength());
        }

        WifiMacHeader header;
        copy->PeekHeader(header);

        // BSSID: address 3 in management frames. In data frames, address 1
        // when sent to the AP, address 2 when sent by it, otherwise address 3.
        // Control frames get none.
        std::string bssid;
        if (header.IsMgt())
        {
            bssid = MacToString(header.GetAddr3());
        }
        else if (header.IsData())
        {
            if (header.IsToDs() && !header.IsFromDs())
            {
                bssid = MacToString(header.GetAddr1());
            }
            else if (!header.IsToDs() && header.IsFromDs())
            {
                bssid = MacToString(header.GetAddr2());
            }
            else
            {
                bssid = MacToString(header.GetAddr3());
            }
        }

        // CTS and ACK carry no transmitter address; every other frame has it
        // in address 2.
        std::string transmitter =
            (header.IsCts() || header.IsAck()) ? "" : MacToString(header.GetAddr2());

        // Airtime of this frame. An aggregate goes out as one transmission
        // behind one preamble, so only its first frame is charged for the
        // preamble.
        Time duration =
            WifiPhy::CalculateTxDuration(copy->GetSize(), txVector, WIFI_PHY_BAND_5GHZ);
        if (aMpdu.type == MIDDLE_MPDU_IN_AGGREGATE || aMpdu.type == LAST_MPDU_IN_AGGREGATE)
        {
            duration -= WifiPhy::CalculatePhyPreambleAndHeaderDuration(txVector);
        }

        // 0 management, 1 control, 2 data.
        int category = header.IsMgt() ? 0 : (header.IsCtl() ? 1 : 2);

        // Times are in seconds from the window start. Every frame of an
        // aggregate carries the start and end of the whole transmission;
        // duration_us is its own share of it.
        frames << (txStart.GetSeconds() - kObservationStartS) << ','
               << (txEnd.GetSeconds() - kObservationStartS) << ','
               << channelFreqMhz << ',' << bssid << ','
               << transmitter << ',' << category << ','
               << (header.IsBeacon() ? 1 : 0) << ','
               << (header.IsRetry() ? 1 : 0) << ',' << copy->GetSize() << ','
               << signalNoise.signal << ','
               << duration.GetMicroSeconds() << ','
               << txVector.GetMode().GetDataRate(txVector) / 1e6 << '\n';
    }

    // ns-3 reports each PHY state interval once it ends, as (start, duration,
    // state). On one radio, receiving and carrier-sense-busy intervals never
    // overlap, so adding them up gives the channel's busy time. The interval
    // is clipped to the listening window; one wholly outside clips to zero
    // length, which UpdateCCARecord ignores.
    void OnPhyState(std::size_t radio, Time start, Time duration, WifiPhyState state)
    {
        if (state != WifiPhyState::RX && state != WifiPhyState::CCA_BUSY)
        {
            return;
        }

        double intervalStart = start.GetSeconds() - kObservationStartS;
        double intervalEnd = std::min(kWindowS, intervalStart + duration.GetSeconds());
        intervalStart = std::max(0.0, intervalStart);
        radios[radio].busy.UpdateCCARecord(intervalStart, intervalEnd);
    }
};

// ---------------------------------------------------------------------------
// Uplink traffic, and the bytes the candidate delivers.
//
// gCandidateBytes adds up every packet that reaches the candidate's socket on
// its AP. Divided by the time from association to the end of the run, it is
// the candidate's throughput.
// ---------------------------------------------------------------------------
uint64_t gCandidateBytes{0};

void
ReceiveCandidatePacket(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    while ((packet = socket->Recv()))
    {
        gCandidateBytes += packet->GetSize();
    }
}

Ptr<Socket>
InstallSink(Ptr<Node> node, const Address& localAddress, TypeId socketType)
{
    Ptr<Socket> sink = Socket::CreateSocket(node, socketType);
    sink->Bind(localAddress);
    return sink;
}

// Background packets leave at exponentially distributed gaps, so each station
// sends a Poisson stream at its mean rate.
//
// Each station draws its gaps from its own generator (seeded in main, see
// trafficBase), so its packet times do not depend on the order in which the
// simulation interleaves different stations' events.
std::vector<std::mt19937> gTrafficRngs;

void
GenerateTraffic(Ptr<Socket> socket, uint32_t packetSize, double meanInterval,
                std::size_t stream)
{
    socket->Send(Create<Packet>(packetSize));
    std::exponential_distribution<double> gap(1.0 / meanInterval);
    Simulator::Schedule(Seconds(gap(gTrafficRngs[stream])),
                        &GenerateTraffic,
                        socket,
                        packetSize,
                        meanInterval,
                        stream);
}

void
ScheduleBackgroundTraffic(Ptr<Node> sender,
                          const InetSocketAddress& destination,
                          TypeId socketType,
                          Time start,
                          Time meanPacketInterval,
                          std::size_t stream)
{
    Ptr<Socket> source = Socket::CreateSocket(sender, socketType);
    source->Connect(destination);
    double meanInterval = meanPacketInterval.GetSeconds();
    // The first packet leaves at a uniformly random offset within one mean
    // interval after start, so stations do not all begin at the same instant.
    std::uniform_real_distribution<double> phase(0.0, meanInterval);
    Simulator::ScheduleWithContext(sender->GetId(),
                                   start + Seconds(phase(gTrafficRngs[stream])),
                                   &GenerateTraffic,
                                   source,
                                   kPacketSize,
                                   meanInterval,
                                   stream);
}

// The candidate sends one packet every packetInterval, a constant rate above
// what its link can carry, so it always has a packet waiting to send.
void
GenerateCandidateTraffic(Ptr<Socket> socket, uint32_t packetSize, Time packetInterval)
{
    socket->Send(Create<Packet>(packetSize));
    Simulator::Schedule(packetInterval, &GenerateCandidateTraffic, socket, packetSize,
                        packetInterval);
}

void
ScheduleCandidateTraffic(Ptr<Node> sender,
                         const InetSocketAddress& destination,
                         TypeId socketType,
                         Time start,
                         Time packetInterval)
{
    Ptr<Socket> source = Socket::CreateSocket(sender, socketType);
    source->Connect(destination);
    Simulator::ScheduleWithContext(sender->GetId(), start, &GenerateCandidateTraffic,
                                   source, kPacketSize, packetInterval);
}

// When, and with which AP, the candidate first associated. time stays -1 if it
// never does.
struct AssociationResult
{
    double time{-1.0};
    Mac48Address ap;
};

AssociationResult gAssociation;

void
CandidateAssociated(Mac48Address apAddress)
{
    if (gAssociation.time < 0.0)
    {
        gAssociation.time = Simulator::Now().GetSeconds();
        gAssociation.ap = apAddress;
    }
}

} // namespace

int
main(int argc, char* argv[])
{
    RunConfig config = ParseRunConfig(argc, argv);
    RngSeedManager::SetSeed(config.rngSeed);

    Time candidateInterval =
        Seconds(kPacketSize * 8.0 / (kCandidateOfferedMbps * 1e6));

    // -----------------------------------------------------------------------
    // Propagation.
    //
    // Every radio shares one MultiModelSpectrumChannel. Channel numbers do not
    // isolate radios from one another: a transmission on an adjacent channel
    // still puts power into a receiver's band.
    // -----------------------------------------------------------------------
    Ptr<MultiModelSpectrumChannel> sharedChannel =
        CreateObject<MultiModelSpectrumChannel>();

    // AddPropagationLossModel prepends (SpectrumChannel::AddPropagationLossModel
    // does loss->SetNext(head); head = loss), so these go in reverse to give a
    // log-distance -> shadowing -> Nakagami chain.
    sharedChannel->AddPropagationLossModel(
        CreateObject<NakagamiPropagationLossModel>());

    // Per-link shadowing, fixed for the whole run: the matrix model applies one
    // stored loss per pair of nodes, filled in once positions exist (below).
    // Pairs never set get 0 dB rather than the model's default of infinite loss.
    Ptr<MatrixPropagationLossModel> shadowingModel =
        CreateObject<MatrixPropagationLossModel>();
    shadowingModel->SetDefaultLoss(0.0);
    sharedChannel->AddPropagationLossModel(shadowingModel);

    Ptr<LogDistancePropagationLossModel> pathLoss =
        CreateObject<LogDistancePropagationLossModel>();
    pathLoss->SetAttribute("Exponent", DoubleValue(kPathLossExponent));
    pathLoss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    pathLoss->SetAttribute("ReferenceLoss", DoubleValue(46.6777)); // free-space loss at 1 m, 5.15 GHz
    sharedChannel->AddPropagationLossModel(pathLoss);

    sharedChannel->SetPropagationDelayModel(
        CreateObject<ConstantSpeedPropagationDelayModel>());

    // -----------------------------------------------------------------------
    // The deployment: AP channels and SSIDs, station loads, packet timing and
    // positions.
    // -----------------------------------------------------------------------

    // Each AP's channel is drawn uniformly from kApChannels. AP channels and
    // station loads come from topologySeed alone, each through its own
    // generator, so neither shifts the placement draws in BuildTopology.
    // (2654435761 is a multiplicative hash constant that spreads nearby seeds
    // apart.)
    std::mt19937 channelRng(config.topologySeed * 2654435761u + 1u);
    std::mt19937 rateRng(config.topologySeed * 2654435761u + 2u);
    std::uniform_int_distribution<std::size_t> channelPick(0, kApChannels.size() - 1);
    std::vector<uint8_t> apChannel(config.nAps);
    std::vector<Ssid> apSsid;
    apSsid.reserve(config.nAps);
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        apChannel[ap] = kApChannels[channelPick(channelRng)];
        apSsid.emplace_back("wifi-ap" + std::to_string(ap));
    }

    // The distinct channels in use, in ascending order. APs can share a
    // channel, so there may be fewer of these than APs.
    std::set<uint32_t> distinctChannels(apChannel.begin(), apChannel.end());
    std::vector<uint32_t> occupiedChannels(distinctChannels.begin(), distinctChannels.end());
    uint32_t nOccupiedChannels = static_cast<uint32_t>(occupiedChannels.size());

    // Each background station's offered load: one log-normal draw per station,
    // capped at kBgLoadCapMbps.
    std::lognormal_distribution<double> loadDraw(std::log(kBgLoadMedianMbps), kBgLoadSigmaLog);
    std::vector<double> staOfferedMbps(config.nStas);
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        staOfferedMbps[sta] = std::min(kBgLoadCapMbps, loadDraw(rateRng));
    }

    // One packet-timing generator per background station, seeded from both
    // topologySeed and rngSeed. A station's mean rate comes from topologySeed
    // alone (above); the moments its packets leave also change with rngSeed.
    const uint32_t trafficBase =
        (config.topologySeed * 2654435761u) ^ (config.rngSeed * 2246822519u);
    gTrafficRngs.reserve(config.nStas);
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        gTrafficRngs.emplace_back(trafficBase + 101u + sta);
    }

    Topology topology = BuildTopology(config);

    // -----------------------------------------------------------------------
    // Nodes and Wi-Fi devices.
    //
    // Every AP has its own SSID. Each background station is given the SSID of
    // the AP chosen for it in BuildTopology (strongest received power), so it
    // can only associate with that AP.
    // -----------------------------------------------------------------------
    NodeContainer apNodes;
    NodeContainer backgroundNodes;
    NodeContainer candidateNode;
    apNodes.Create(config.nAps);
    backgroundNodes.Create(config.nStas);
    candidateNode.Create(1);

    NodeContainer allNodes;
    allNodes.Add(apNodes);
    allNodes.Add(backgroundNodes);
    allNodes.Add(candidateNode);

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");

    // SpectrumWifiPhy spreads each transmission's power across frequency with
    // the 802.11 transmit spectral mask, so a transmitter on an adjacent channel
    // adds power in a receiver's band and can register in its carrier sense.
    SpectrumWifiPhyHelper wifiPhy;
    wifiPhy.SetChannel(sharedChannel);

    WifiMacHelper wifiMac;
    std::vector<NetDeviceContainer> apGroupDevices(config.nAps);
    std::vector<std::vector<uint32_t>> apStaGlobalIndex(config.nAps);

    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(apChannel[ap])));
        wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, apNodes.Get(ap)));
    }

    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        uint32_t ap = topology.staAssociatedAp[sta];
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(apChannel[ap])));
        wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, backgroundNodes.Get(sta)));
        apStaGlobalIndex[ap].push_back(sta);
    }

    // One listening radio per occupied channel, all on the candidate node. They
    // do not probe and their SSID matches no AP, so they never associate or
    // transmit.
    std::vector<Ptr<WifiNetDevice>> scannerDevices;
    scannerDevices.reserve(nOccupiedChannels);
    for (uint32_t channelIndex = 0; channelIndex < nOccupiedChannels; ++channelIndex)
    {
        wifiPhy.Set("ChannelSettings",
                    StringValue(ChannelSettings(occupiedChannels[channelIndex])));
        wifiMac.SetType("ns3::StaWifiMac",
                        "Ssid",
                        SsidValue(Ssid("scanner-never-associates")),
                        "ActiveProbing",
                        BooleanValue(false));
        NetDeviceContainer installed =
            wifi.Install(wifiPhy, wifiMac, candidateNode.Get(0));
        scannerDevices.push_back(DynamicCast<WifiNetDevice>(installed.Get(0)));
    }

    // The candidate's association radio waits on kParkChannel with an SSID no
    // AP uses. At kCandidateStartS it moves to targetAP's channel and takes its
    // SSID (scheduled below), then associates through the normal procedure.
    wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(kParkChannel)));
    wifiMac.SetType("ns3::StaWifiMac",
                    "Ssid",
                    SsidValue(Ssid("pending-join")),
                    "ActiveProbing",
                    BooleanValue(false));
    NetDeviceContainer candidateDeviceContainer =
        wifi.Install(wifiPhy, wifiMac, candidateNode.Get(0));
    Ptr<WifiNetDevice> candidateDevice =
        DynamicCast<WifiNetDevice>(candidateDeviceContainer.Get(0));
    apGroupDevices[config.targetAp].Add(candidateDeviceContainer);

    // -----------------------------------------------------------------------
    // Position and network-layer configuration.
    //
    // ListPositionAllocator hands out positions in the order allNodes was
    // assembled: APs, background stations, then the candidate. Each AP and its
    // stations get their own IPv4 subnet; the candidate's association radio is
    // addressed in targetAP's subnet.
    // -----------------------------------------------------------------------
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAllocator = CreateObject<ListPositionAllocator>();
    for (const auto& position : topology.apPositions)
    {
        positionAllocator->Add(position);
    }
    for (const auto& position : topology.staPositions)
    {
        positionAllocator->Add(position);
    }
    positionAllocator->Add(topology.candidatePosition);

    mobility.SetPositionAllocator(positionAllocator);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    // Now that every node has a mobility model, give the channel the shadowing
    // drawn in BuildTopology, the same values StrongestAp used. SetLoss takes
    // attenuation, so the sign flips: a positive draw means extra received power.
    for (uint32_t i = 0; i < allNodes.GetN(); ++i)
    {
        for (uint32_t j = i + 1; j < allNodes.GetN(); ++j)
        {
            shadowingModel->SetLoss(allNodes.Get(i)->GetObject<MobilityModel>(),
                                    allNodes.Get(j)->GetObject<MobilityModel>(),
                                    -topology.shadowingDb[i][j],
                                    true);
        }
    }

    InternetStackHelper internet;
    internet.Install(allNodes);

    Ipv4AddressHelper ipv4;
    std::vector<Ipv4Address> apAddress(config.nAps);
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        std::string subnet = "10.1." + std::to_string(ap + 1) + ".0";
        ipv4.SetBase(subnet.c_str(), "255.255.255.0");
        Ipv4InterfaceContainer interfaces = ipv4.Assign(apGroupDevices[ap]);
        apAddress[ap] = interfaces.GetAddress(0);
    }

    // -----------------------------------------------------------------------
    // Uplink traffic.
    //
    // Every background station sends to its own socket on its AP from
    // kBackgroundStartS. The candidate sends to its own socket on targetAP from
    // kCandidateStartS.
    // -----------------------------------------------------------------------
    TypeId udpSocketType = TypeId::LookupByName("ns3::UdpSocketFactory");

    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        for (uint32_t localSta = 0; localSta < apStaGlobalIndex[ap].size(); ++localSta)
        {
            Address sinkAddress = InetSocketAddress(apAddress[ap], kBasePort + localSta);
            InstallSink(apNodes.Get(ap), sinkAddress, udpSocketType);

            uint32_t globalSta = apStaGlobalIndex[ap][localSta];
            ScheduleBackgroundTraffic(backgroundNodes.Get(globalSta),
                                      InetSocketAddress(apAddress[ap], kBasePort + localSta),
                                      udpSocketType,
                                      Seconds(kBackgroundStartS),
                                      Seconds(kPacketSize * 8.0 /
                                              (staOfferedMbps[globalSta] * 1e6)),
                                      globalSta);
        }
    }

    Address candidateSinkAddress =
        InetSocketAddress(apAddress[config.targetAp], kBasePort + config.nStas);
    InstallSink(apNodes.Get(config.targetAp), candidateSinkAddress, udpSocketType)
        ->SetRecvCallback(MakeCallback(&ReceiveCandidatePacket));
    ScheduleCandidateTraffic(candidateNode.Get(0),
                             InetSocketAddress(apAddress[config.targetAp],
                                               kBasePort + config.nStas),
                             udpSocketType,
                             Seconds(kCandidateStartS),
                             candidateInterval);

    // Retuning is scheduled before changing the SSID at the same timestamp.
    // ns-3 executes equal-time events in insertion order, so the MAC begins
    // searching for the selected SSID only after its PHY reaches that channel.
    auto setOperatingChannel =
        static_cast<void (WifiPhy::*)(const WifiPhy::ChannelTuple&)>(
            &WifiPhy::SetOperatingChannel);
    Simulator::Schedule(Seconds(kCandidateStartS),
                        setOperatingChannel,
                        candidateDevice->GetPhy(),
                        WifiPhy::ChannelTuple{
                            apChannel[config.targetAp], 20, WIFI_PHY_BAND_5GHZ, 0});
    Simulator::Schedule(Seconds(kCandidateStartS),
                        &WifiMac::SetSsid,
                        candidateDevice->GetMac(),
                        apSsid[config.targetAp]);
    candidateDevice->GetMac()->TraceConnectWithoutContext(
        "Assoc",
        MakeCallback(&CandidateAssociated));

    // -----------------------------------------------------------------------
    // Per-run files and observation traces.
    //
    // The listening radios are installed whether or not captureObs writes
    // their traces, so writing them never changes the simulation.
    // -----------------------------------------------------------------------
    std::string runDir = config.outDir + "/" + config.runTag;
    std::filesystem::create_directories(runDir);

    std::optional<ObservationRecorder> recorder;
    if (config.captureObservation)
    {
        recorder.emplace(runDir, scannerDevices, occupiedChannels);
    }

    // When the listening window closes, the scanners move to the park channel:
    // nothing they hear afterwards is recorded.
    for (const auto& scanner : scannerDevices)
    {
        Simulator::Schedule(Seconds(kCandidateStartS),
                            setOperatingChannel,
                            scanner->GetPhy(),
                            WifiPhy::ChannelTuple{
                                kParkChannel, 20, WIFI_PHY_BAND_5GHZ, 0});
    }

    std::cout << "run=" << config.runTag << " topologySeed=" << config.topologySeed
              << " rngSeed=" << config.rngSeed << " nAPs=" << config.nAps
              << " nSTAs=" << config.nStas << " targetAP=" << config.targetAp
              << std::endl;

    Simulator::Stop(Seconds(kSimulationStopS));
    Simulator::Run();

    // -----------------------------------------------------------------------
    // Throughput label and structured run metadata.
    //
    // throughput_mbps is the candidate's delivered bytes averaged from the
    // moment it associated to kSimulationStopS; nothing it sends before
    // association is delivered. metadata.json also records the run's
    // parameters and the deployment.
    // -----------------------------------------------------------------------
    bool candidateAssociated = gAssociation.time >= 0.0;
    double candidateObservedSeconds =
        candidateAssociated ? kSimulationStopS - gAssociation.time : 0.0;
    double candidateMbps =
        candidateAssociated && candidateObservedSeconds > 0.0
            ? gCandidateBytes * 8.0 / 1e6 / candidateObservedSeconds
            : 0.0;

    std::string associatedAp =
        candidateAssociated ? MacToString(gAssociation.ap) : std::string{};

    std::ofstream metadata(runDir + "/metadata.json");
    NS_ABORT_MSG_IF(!metadata.is_open(), "could not open metadata.json");
    metadata << "{\n";
    metadata << "  \"run_id\": \"" << config.runTag << "\",\n";
    metadata << "  \"rng_seed\": " << config.rngSeed << ",\n";
    metadata << "  \"params\": {\n";
    metadata << "    \"topology_seed\": " << config.topologySeed << ",\n";
    metadata << "    \"packet_size\": " << kPacketSize << ",\n";
    metadata << "    \"n_stas\": " << config.nStas << ",\n";
    metadata << "    \"n_aps\": " << config.nAps << ",\n";
    metadata << "    \"target_ap\": " << config.targetAp << ",\n";
    metadata << "    \"ap_spacing\": " << kApSpacingM << ",\n";
    metadata << "    \"candidate_start_time\": " << kCandidateStartS << ",\n";
    metadata << "    \"observation_start_time\": " << kObservationStartS << ",\n";
    // The end of the listening window on the clock observation.csv and
    // chanbusy.csv use, which reads 0 when listening begins.
    metadata << "    \"feature_window_end\": " << (kCandidateStartS - kObservationStartS)
             << ",\n";
    metadata << "    \"sim_stop_time\": " << kSimulationStopS << ",\n";
    metadata << "    \"background_start_time\": " << kBackgroundStartS << ",\n";
    metadata << "    \"candidate_interval_s\": " << candidateInterval.GetSeconds()
             << ",\n";
    double bgTotalOffered = 0.0;
    for (double rate : staOfferedMbps)
    {
        bgTotalOffered += rate;
    }
    metadata << "    \"link_shadowing_db\": " << kLinkShadowingDb << ",\n";
    metadata << "    \"bg_load_median_mbps\": " << kBgLoadMedianMbps << ",\n";
    metadata << "    \"bg_load_sigma_log\": " << kBgLoadSigmaLog << ",\n";
    metadata << "    \"bg_load_cap_mbps\": " << kBgLoadCapMbps << ",\n";
    metadata << "    \"bg_mean_per_sta_mbps\": "
             << (config.nStas ? bgTotalOffered / config.nStas : 0.0) << ",\n";
    metadata << "    \"bg_total_offered_mbps\": " << bgTotalOffered << ",\n";
    std::vector<double> apOfferedMbps(config.nAps, 0.0);
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        apOfferedMbps[topology.staAssociatedAp[sta]] += staOfferedMbps[sta];
    }
    metadata << "    \"candidate_offered_mbps\": " << kCandidateOfferedMbps
             << ",\n";
    metadata << "    \"hotspot_aps\": [";
    for (std::size_t index = 0; index < config.hotspotAps.size(); ++index)
    {
        metadata << config.hotspotAps[index]
                 << (index + 1 < config.hotspotAps.size() ? ", " : "");
    }
    metadata << "],\n";
    metadata << "    \"n_hotspots\": " << config.hotspotAps.size() << ",\n";
    metadata << "    \"hotspot_radius\": " << kHotspotRadiusM << ",\n";
    metadata << "    \"hotspot_density\": " << kHotspotDensity << ",\n";
    metadata << "    \"hotspot_probability\": " << topology.hotspotProbability
             << ",\n";
    metadata << "    \"n_channels\": " << nOccupiedChannels << "\n";
    metadata << "  },\n";
    metadata << "  \"candidate_seed\": " << config.candidateSeed << ",\n";
    metadata << "  \"candidate_position\": {\"x\": " << topology.candidatePosition.x
             << ", \"y\": " << topology.candidatePosition.y << "},\n";
    metadata << "  \"aps\": [\n";

    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        std::string mac = MacToString(
            Mac48Address::ConvertFrom(apGroupDevices[ap].Get(0)->GetAddress()));

        metadata << "    {\"index\": " << ap << ", \"ssid\": \""
                 << apSsid[ap].PeekString() << "\", \"mac\": \"" << mac
                 << "\", \"channel\": " << +apChannel[ap]
                 << ", \"position\": {\"x\": " << topology.apPositions[ap].x
                 << ", \"y\": " << topology.apPositions[ap].y
                 << "}, \"sta_count\": " << apStaGlobalIndex[ap].size()
                 << ", \"offered_mbps\": " << apOfferedMbps[ap]
                 << ", \"candidate_distance\": " << topology.candidateApDistance[ap]
                 << "}" << (ap + 1 < config.nAps ? ",\n" : "\n");
    }

    metadata << "  ],\n";
    metadata << "  \"background_stations\": [\n";
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        metadata << "    {\"index\": " << sta << ", \"ap\": "
                 << topology.staAssociatedAp[sta] << ", \"position\": {\"x\": "
                 << topology.staPositions[sta].x << ", \"y\": "
                 << topology.staPositions[sta].y << "}, \"offered_mbps\": "
                 << staOfferedMbps[sta] << "}"
                 << (sta + 1 < config.nStas ? ",\n" : "\n");
    }
    metadata << "  ],\n";
    metadata << "  \"candidate\": {\n";
    metadata << "    \"target_ap\": " << config.targetAp << ",\n";
    metadata << "    \"associated\": " << (candidateAssociated ? "true" : "false")
             << ",\n";
    metadata << "    \"assoc_time\": " << gAssociation.time << ",\n";
    metadata << "    \"assoc_delay\": "
             << (candidateAssociated ? gAssociation.time - kCandidateStartS : -1.0)
             << ",\n";
    metadata << "    \"assoc_ap_mac\": \"" << associatedAp << "\",\n";
    metadata << "    \"observed_seconds\": " << candidateObservedSeconds << ",\n";
    metadata << "    \"bytes_total\": " << gCandidateBytes << ",\n";
    metadata << "    \"throughput_mbps\": " << candidateMbps << "\n";
    metadata << "  }\n";
    metadata << "}\n";
    metadata.close();

    if (recorder)
    {
        recorder->Close();
    }

    Simulator::Destroy();
    return 0;
}
