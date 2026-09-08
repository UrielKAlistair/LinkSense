#include "ns3/abort.h"
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
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace ns3;

namespace
{

// ---------------------------------------------------------------------------
// Simulator 1 models one stationary client choosing between several nearby
// access points. Every matched run uses the same physical deployment and
// pre-association observation; only the AP that the candidate eventually
// joins changes. The resulting throughput is the label for that AP option.
//
// These constants are assumptions of this experiment, not sweep dimensions.
// Keeping them together makes the simulated system visible without requiring
// the reader to reconstruct it from command-line defaults spread through
// main().
// ---------------------------------------------------------------------------
constexpr uint16_t kBasePort = 8000;
constexpr uint32_t kPacketSize = 1250;
// TGax Simulation Scenarios (IEEE 802.11-14/0980r16) states "~10-20m inter AP
// distance" for both dense indoor scenarios (2 Enterprise, 3 Indoor Small BSS
// Hotspot). 20 m is the top of that range, chosen as a round declared value
// rather than inheriting the exact hexagon arithmetic of a layout we do not
// otherwise follow.
constexpr double kApSpacingM = 20.0;
constexpr double kBackgroundStartS = 0.5;
// Background traffic needs ~1.5 s to reach steady state: association
// completes quickly but Minstrel-HT converges only after heavy early
// retransmission. Recording from 2.0 s means the client joins a network
// already in flight rather than watching the same rising edge every run.
constexpr double kObservationStartS = 2.0;
constexpr double kCandidateStartS = 8.0;
constexpr double kFeatureGuardS = 0.5;
constexpr double kSimulationStopS = 20.0;
constexpr double kCandidateOfferedMbps = 100.0;
// A station is kHotspotDensity times more likely to land in a hotspot disc
// than anywhere else of equal area. This is a density, not a per-AP count: the
// disc is 314 m^2 against a 346 m^2 cell, so a 4x density gives a hotspot AP
// roughly 2.2x the clients of an ordinary one, not 4x (measured 2.17-2.19
// over 4000 sampled scenarios, flat across AP count).
constexpr double kHotspotDensity = 4.0;
// Half the AP spacing, so the rim of the disc is equidistant from the
// neighbouring AP. Stations seeded near the rim therefore associate away from
// the intended hotspot a good fraction of the time, which caps how far a
// hotspot can concentrate. An earlier spacing/3 kept every seeded station on
// the intended AP, which is exactly the concentration we want to bleed off.
constexpr double kHotspotRadiusM = kApSpacingM / 2.0;
constexpr double kBusyBucketS = 0.1024;
constexpr double kPathLossExponent = 3.0;
// TGax uses 5 dB log-normal shadowing, iid per link, in Residential,
// Enterprise and Indoor Small BSS alike. One draw per unordered node pair,
// fixed for the whole run, applied BOTH by the association model and by the
// simulated channel -- so the power a station used to pick its AP is the power
// the channel then delivers. A per-transmission draw (as Nakagami is) would
// average away over a window of beacons; real shadowing does not.
constexpr double kLinkShadowingDb = 5.0;
constexpr double kPi = 3.14159265358979323846;

// Four non-overlapping 20 MHz channels in the 5 GHz band. Each AP draws one
// uniformly at random, so co-channel separation is not a function of AP count.
// The propagation reference loss below is free-space loss at 1 m near 5.2 GHz,
// so the channel plan and propagation model describe the same frequency band.
const std::vector<uint8_t> kApChannels = {36, 40, 44, 48};

// A channel no AP occupies. The candidate's association radio parks here
// until the decision time. This prevents that radio from receiving frames and
// consuming fading RNG draws before the matched variants are allowed to
// differ. Scanner radios are moved here when their observation window ends.
constexpr uint8_t kParkChannel = 149;

struct RunConfig
{
    uint32_t nStas{12};
    uint32_t nAps{3};
    uint32_t targetAp{0};
    double candidateX{15.0};
    double candidateY{8.66};
    std::string hotspotApsText{"none"};
    std::vector<uint32_t> hotspotAps;
    double backgroundMbpsPerSta{0.0};   // >0 pins every station to one rate
    double backgroundMedianMbps{2.25};
    double backgroundSigmaLog{0.9};
    double backgroundCapMbps{25.0};
    double linkShadowingDb{kLinkShadowingDb};
    double backgroundMeanMbps{0.0};
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
    std::vector<uint32_t> staServingAp;
    Vector candidatePosition;
    std::vector<double> candidateApDistance;
    double hotspotProbability{0.0};
    // [i][j] in dB, symmetric, zero diagonal. Node order matches allNodes:
    // APs, then background stations, then the candidate.
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

    NS_ABORT_MSG_IF(hotspotAps.empty(),
                    "hotspotAPs must be 'none' or contain at least one AP index");
    return hotspotAps;
}

// ---------------------------------------------------------------------------
// Command-line options describe either the physical scenario or the identity
// of one matched run. Radio standard, timing, propagation, channel reuse, and
// packet-generation assumptions are fixed above and deliberately unavailable
// as incidental command-line variations.
// ---------------------------------------------------------------------------
RunConfig
ParseRunConfig(int argc, char* argv[])
{
    RunConfig config;
    CommandLine cmd(__FILE__);

    cmd.AddValue("nSTAs", "total number of background stations", config.nStas);
    cmd.AddValue("nAPs", "number of APs in the fixed two-dimensional layout", config.nAps);
    cmd.AddValue("targetAP", "index of the AP the candidate joins", config.targetAp);
    cmd.AddValue("candidateX", "candidate x coordinate in metres", config.candidateX);
    cmd.AddValue("candidateY", "candidate y coordinate in metres", config.candidateY);
    cmd.AddValue("hotspotAPs",
                 "comma-separated APs around which background stations gather; 'none' disables",
                 config.hotspotApsText);
    cmd.AddValue("bgLoadMeanMbps",
                 "target mean per-station offered load; overrides bgLoadMedianMbps "
                 "so sigma varies spread without moving the mean",
                 config.backgroundMeanMbps);
    cmd.AddValue("linkShadowingDb",
                 "log-normal shadowing sigma in dB, one draw per node pair fixed "
                 "for the run, used by both association and the channel; 0 gives "
                 "nearest-AP assignment and an unshadowed channel",
                 config.linkShadowingDb);
    cmd.AddValue("bgLoadMedianMbps",
                 "median of the log-normal per-station offered load",
                 config.backgroundMedianMbps);
    cmd.AddValue("bgLoadSigmaLog",
                 "shape (sigma of log) of the per-station offered load",
                 config.backgroundSigmaLog);
    cmd.AddValue("bgLoadCapMbps",
                 "upper clamp on a drawn per-station offered load",
                 config.backgroundCapMbps);
    cmd.AddValue("bgPerStaMbps",
                 "pin every background station to this load instead of drawing; "
                 "0 draws each station from the log-normal",
                 config.backgroundMbpsPerSta);
    cmd.AddValue("topologySeed",
                 "seed used only for background-station placement",
                 config.topologySeed);
    cmd.AddValue("rngSeed",
                 "ns-3 PHY/MAC seed; 0 chooses and records a random seed",
                 config.rngSeed);
    cmd.AddValue("outDir", "directory under which the run directory is written", config.outDir);
    cmd.AddValue("runTag", "run directory name; empty derives one from the seeds and target AP", config.runTag);
    cmd.AddValue("captureObs",
                 "write the shared pre-association observation for this matched variant",
                 config.captureObservation);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(config.nAps == 0, "nAPs must be at least 1");
    NS_ABORT_MSG_IF(config.nAps > 200, "nAPs must be at most 200");
    NS_ABORT_MSG_IF(config.nStas == 0, "nSTAs must be at least 1");
    NS_ABORT_MSG_IF(config.targetAp >= config.nAps, "targetAP must be less than nAPs");
    // Mean of a log-normal is median * exp(sigma^2 / 2), so pinning the mean
    // lets sigma control spread alone rather than dragging the mean with it.
    if (config.backgroundMeanMbps > 0.0)
    {
        config.backgroundMedianMbps =
            config.backgroundMeanMbps /
            std::exp(config.backgroundSigmaLog * config.backgroundSigmaLog / 2.0);
    }
    NS_ABORT_MSG_IF(config.backgroundMedianMbps <= 0.0,
                    "bgLoadMedianMbps must be positive");
    NS_ABORT_MSG_IF(config.backgroundSigmaLog < 0.0,
                    "bgLoadSigmaLog must not be negative");
    NS_ABORT_MSG_IF(config.backgroundCapMbps <= config.backgroundMedianMbps,
                    "bgLoadCapMbps must exceed bgLoadMedianMbps");
    NS_ABORT_MSG_IF(config.backgroundMbpsPerSta < 0.0,
                    "bgPerStaMbps must be zero (draw per station) or positive");
    NS_ABORT_MSG_IF(config.topologySeed == 0, "topologySeed must be positive");
    config.hotspotAps = ParseHotspotAps(config.hotspotApsText, config.nAps);

    if (config.rngSeed == 0)
    {
        std::random_device randomDevice;
        config.rngSeed = 1 + randomDevice() % 2147483646u;
    }

    if (config.runTag.empty())
    {
        config.runTag = "topology_" + std::to_string(config.topologySeed) + "_seed_" +
                        std::to_string(config.rngSeed) + "_ap_" +
                        std::to_string(config.targetAp);
    }

    return config;
}

// ---------------------------------------------------------------------------
// AP geometry is deterministic: a triangular (hexagonal) lattice at
// kApSpacingM, filled row-major with alternate rows offset by half a spacing.
// Two APs form a pair, three an equilateral triangle, and the four-, six- and
// eight-AP cases use two rows of two, three and four. AP 0 remains at the
// origin, which keeps controlled distance tests simple.
// ---------------------------------------------------------------------------
std::vector<Vector>
BuildApPositions(uint32_t nAps)
{
    // Triangular (hexagonal) lattice: every AP sits at the centre of one
    // hexagonal cell, adjacent centres are exactly kApSpacingM apart, and
    // alternate rows are offset by half a spacing. For nAps = 3 this is the
    // equilateral triangle the previous special case built by hand. The
    // square grid it replaces contained diagonals at 1.414x and 2.236x the
    // spacing, which made choice difficulty vary with AP count.
    uint32_t columns = nAps >= 4 && nAps <= 8 && nAps % 2 == 0
                           ? nAps / 2
                           : static_cast<uint32_t>(
                                 std::ceil(std::sqrt(static_cast<double>(nAps))));
    std::vector<Vector> positions;
    positions.reserve(nAps);

    const double rowPitch = kApSpacingM * std::sqrt(3.0) / 2.0;
    for (uint32_t index = 0; index < nAps; ++index)
    {
        uint32_t row = index / columns;
        uint32_t column = index % columns;
        double rowOffset = (row % 2) * kApSpacingM / 2.0;
        positions.emplace_back(rowOffset + column * kApSpacingM,
                               row * rowPitch,
                               0.0);
    }

    return positions;
}

// A station joins the AP it hears most strongly, which is what a real client
// does. Received power is the same log-distance path loss the channel uses,
// plus a log-normal shadowing term: transmit power and reference loss are
// identical for every AP, so they cancel in the comparison and only distance
// and shadowing decide it.
//
// Shadowing is why association is not a Voronoi partition in practice. Without
// it, an AP's client count would be a deterministic function of the geometry,
// and counting transmitters would substitute for measuring load. Sigma is in
// dB; 0 reproduces nearest-AP assignment exactly and leaves the channel
// unshadowed. At 5 dB roughly one station in six does not join its nearest AP.
//
// Shadowing is drawn once per node pair before any station is placed, so this
// reads the station's row rather than drawing: the same value is handed to the
// channel below, and the stream cannot depend on which AP wins.
uint32_t
ChooseServingAp(double x, double y, const std::vector<Vector>& apPositions,
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
// The topology seed controls physical placement and is separate from the
// ns-3 seed used for fading, contention, and rate-control randomness. Thus,
// three PHY/MAC seeds for one topology really do preserve AP positions,
// background-station positions, loads, and the candidate's location.
//
// Placement is one inhomogeneous point process either way. Without hotspots it
// is uniform over the AP footprint plus a border. With hotspots the density is
// kHotspotDensity times higher inside the discs, so the share landing in one is
// k*A_D / (A_B + (k-1)*A_D) -- a consequence of the geometry, not a quota. The
// disc radius equals half the AP spacing, so its rim is equidistant from the
// neighbouring AP and a fraction of those stations associate away.
// ---------------------------------------------------------------------------
Topology
BuildTopology(const RunConfig& config)
{
    Topology topology;
    topology.apPositions = BuildApPositions(config.nAps);
    topology.candidatePosition = Vector(config.candidateX, config.candidateY, 0.0);

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

    // The border equals the hotspot radius, so every disc lies wholly inside the
    // box and its area needs no clipping. It is also the more defensible rule:
    // coverage reaches half a cell past the outermost APs. The border does not
    // depend on whether the scenario has hotspots, so uniform and clustered
    // scenarios share one box and one baseline density.
    double border = kHotspotRadiusM;
    std::mt19937 placementRng(config.topologySeed);

    // One shadowing draw per unordered node pair, on its own stream so it can
    // neither shift nor be shifted by placement. Node indices match allNodes in
    // main(): APs, background stations, candidate. Fixed for the run, so a
    // window of beacons cannot average it away the way it averages Nakagami.
    const uint32_t nNodes = config.nAps + config.nStas + 1;
    std::mt19937 shadowRng(config.topologySeed * 2654435761u + 4u);
    std::normal_distribution<double> shadowDraw(
        0.0, std::max(0.0, config.linkShadowingDb));
    topology.shadowingDb.assign(nNodes, std::vector<double>(nNodes, 0.0));
    if (config.linkShadowingDb > 0.0)
    {
        for (uint32_t i = 0; i < nNodes; ++i)
        {
            for (uint32_t j = i + 1; j < nNodes; ++j)
            {
                double draw = shadowDraw(shadowRng);
                topology.shadowingDb[i][j] = draw;
                topology.shadowingDb[j][i] = draw;
            }
        }
    }
    std::uniform_real_distribution<double> uniformX(minX - border, maxX + border);
    std::uniform_real_distribution<double> uniformY(minY - border, maxY + border);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    // Placement is an inhomogeneous process: density is kHotspotDensity inside
    // the discs and 1 outside. Normalising f = k*c on D and c on B\D over the
    // box gives c = 1 / (A_B + (k-1)*A_D), so a station lands in a disc with
    // probability k*A_D / (A_B + (k-1)*A_D). Discs never overlap -- the radius
    // is half the lattice spacing, so neighbouring discs are at worst tangent --
    // hence A_D is just their summed area.
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
    topology.staServingAp.reserve(config.nStas);
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
            // The discs carry their own density, so the baseline draw rejects
            // anything landing inside one. Discs cover at most a third of the
            // box, so this accepts on the first or second try.
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
        topology.staServingAp.push_back(ChooseServingAp(
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

/** Build the ChannelSettings attribute string for a 20 MHz 5 GHz channel. */
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
// Pre-association observation recording.
//
// Each occupied channel has one passive scanner PHY on the candidate node.
// MonitorSnifferRx records frames the scanner successfully decodes, while the
// PHY-state trace separately records how long carrier sense reports the medium
// busy. Both stop 0.5 s before association begins, so every target-AP variant
// is built from one shared observation that predates the choice.
// ---------------------------------------------------------------------------
struct BusyMeter
{
    std::vector<double> busySeconds;

    void Add(double start, double end)
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

    double Fraction(std::size_t bucket, double bucketDuration) const
    {
        if (bucket >= busySeconds.size() || bucketDuration <= 0.0)
        {
            return 0.0;
        }
        return std::min(1.0, busySeconds[bucket] / bucketDuration);
    }
};

struct ObservationRecorder
{
    std::ofstream frames;
    std::ofstream channelBusy;
    double windowStart{0.0};
    double windowEnd{0.0};

    double Duration() const { return windowEnd - windowStart; }
    std::vector<BusyMeter> busyByChannel;
    std::vector<uint32_t> busyChannelNumbers;
};

ObservationRecorder gObservation;

void
MonitorSniffRx(Ptr<const Packet> packet,
               uint16_t channelFreqMhz,
               WifiTxVector txVector,
               MpduInfo /*aMpdu*/,
               SignalNoiseDbm signalNoise,
               uint16_t /*staId*/)
{
    double now = Simulator::Now().GetSeconds();
    if (!gObservation.frames.is_open() || now < gObservation.windowStart ||
        now >= gObservation.windowEnd)
    {
        return;
    }

    Ptr<Packet> copy = packet->Copy();
    WifiMacHeader header;
    if (copy->PeekHeader(header) == 0)
    {
        return;
    }

    // Which address carries the BSSID depends on the distribution-system
    // bits. Control frames such as ACK and CTS have no BSSID at all.
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

    std::string transmitter =
        header.IsCtl() && !header.IsRts() ? "" : MacToString(header.GetAddr2());
    Time duration =
        WifiPhy::CalculateTxDuration(packet->GetSize(), txVector, WIFI_PHY_BAND_5GHZ);

    // Frame category 0/1/2 follows the 802.11 management/control/data type
    // field rather than ns-3's finer-grained internal WifiMacType values.
    int category = header.IsMgt() ? 0 : (header.IsCtl() ? 1 : 2);

    // Timestamps are emitted relative to the window start, so everything
    // downstream still sees a window that begins at t = 0.
    gObservation.frames << (now - gObservation.windowStart) << ',' << channelFreqMhz << ',' << bssid << ','
                        << transmitter << ',' << category << ','
                        << (header.IsBeacon() ? 1 : 0) << ','
                        << (header.IsRetry() ? 1 : 0) << ',' << packet->GetSize() << ','
                        << signalNoise.signal << ',' << signalNoise.noise << ','
                        << duration.GetMicroSeconds() << ','
                        << txVector.GetMode().GetDataRate(txVector) / 1e6 << '\n';
}

void
ScannerPhyState(uint32_t channelNumber, Time start, Time duration, WifiPhyState state)
{
    if (duration <= Time(0))
    {
        return;
    }

    double intervalStart = start.GetSeconds() - gObservation.windowStart;
    double intervalEnd = intervalStart + duration.GetSeconds();
    if (intervalStart >= gObservation.Duration() || intervalEnd <= 0.0)
    {
        return;
    }

    intervalStart = std::max(0.0, intervalStart);
    intervalEnd = std::min(gObservation.Duration(), intervalEnd);
    bool busy = state == WifiPhyState::CCA_BUSY || state == WifiPhyState::TX ||
                state == WifiPhyState::RX;
    if (!busy || intervalEnd <= intervalStart)
    {
        return;
    }

    auto channel = std::find(gObservation.busyChannelNumbers.begin(),
                             gObservation.busyChannelNumbers.end(),
                             channelNumber);
    if (channel != gObservation.busyChannelNumbers.end())
    {
        std::size_t index =
            static_cast<std::size_t>(channel - gObservation.busyChannelNumbers.begin());
        if (index < gObservation.busyByChannel.size())
        {
            gObservation.busyByChannel[index].Add(intervalStart, intervalEnd);
        }
    }
}

void
WriteChannelBusyCsv()
{
    if (!gObservation.channelBusy.is_open())
    {
        return;
    }

    std::size_t nBuckets =
        static_cast<std::size_t>(std::ceil(gObservation.Duration() / kBusyBucketS));
    for (std::size_t channelIndex = 0;
         channelIndex < gObservation.busyByChannel.size();
         ++channelIndex)
    {
        uint32_t channel = gObservation.busyChannelNumbers[channelIndex];
        for (std::size_t bucket = 0; bucket < nBuckets; ++bucket)
        {
            double start = bucket * kBusyBucketS;
            double end = std::min(gObservation.Duration(), (bucket + 1) * kBusyBucketS);
            double duration = end - start;
            if (duration <= 0.0)
            {
                continue;
            }

            gObservation.channelBusy
                << start << ',' << end << ',' << channel << ',' << (5000 + 5 * channel)
                << ',' << gObservation.busyByChannel[channelIndex].Fraction(bucket, duration)
                << '\n';
        }
    }
}

// ---------------------------------------------------------------------------
// Traffic accounting records bytes delivered by each uplink flow. Background
// flows are later aggregated per AP; the candidate flow becomes the throughput
// label. Binding the flow index directly into the receive callback avoids a
// second address-to-flow lookup structure.
// ---------------------------------------------------------------------------
struct FlowStats
{
    uint64_t bytesTotal{0};
};

std::vector<FlowStats> gFlowStats;

void
ReceivePacket(std::size_t flowIndex, Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    while ((packet = socket->Recv()))
    {
        gFlowStats.at(flowIndex).bytesTotal += packet->GetSize();
    }
}

std::size_t
InstallSink(Ptr<Node> node, const Address& localAddress, TypeId socketType)
{
    std::size_t flowIndex = gFlowStats.size();
    gFlowStats.emplace_back();

    Ptr<Socket> sink = Socket::CreateSocket(node, socketType);
    sink->Bind(localAddress);
    sink->SetRecvCallback(MakeBoundCallback(&ReceivePacket, flowIndex));
    return flowIndex;
}

// Background arrivals are a Poisson process at the station's mean rate, not a
// metronome. Deterministic same-phase CBR gives every bin the same expected
// occupancy once the transient passes, so the binned sequence carries little
// beyond its own mean - which would bias this study against the temporal
// hypothesis it exists to test. Exponential gaps preserve E[offered load], so
// the label is unchanged. Every non-full-buffer model in the TGax set
// randomises phase; its own calibration config specifies "Random start time
// during a 10 ms interval". Note MEASURED: this removed the artefact but did
// not add temporal structure -- busy-fraction autocorrelation sits inside the
// noise band, so the series is white noise about a constant.
//
// Each station owns one mt19937 seeded off BOTH seeds (see trafficBase in
// main). Its mean rate is a property of the deployment and hangs off
// topologySeed; when its packets actually leave does not, so the arrival
// realisation varies across radio seeds while staying fixed within a matched
// target-AP replay.
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
    // A uniform phase over one mean interval breaks the lock-step that made
    // same-rate stations on one AP transmit in formation for the whole run.
    std::uniform_real_distribution<double> phase(0.0, meanInterval);
    Simulator::ScheduleWithContext(sender->GetId(),
                                   start + Seconds(phase(gTrafficRngs[stream])),
                                   &GenerateTraffic,
                                   source,
                                   kPacketSize,
                                   meanInterval,
                                   stream);
}

// The candidate stays deliberately deterministic: it is offered 100 Mbit/s to
// remain backlogged so its delivered rate measures capacity. Jittering it would
// add variance to the label for no gain.
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
    // Radio and propagation model.
    //
    // Every radio shares one MultiModelSpectrumChannel, so scanner radios retune
    // across the same physical environment. Channel numbers do not partition the
    // medium: under the spectrum model an adjacent-channel neighbour still
    // deposits power through the transmit mask (see below).
    // -----------------------------------------------------------------------
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");

    // SpectrumWifiPhy rather than Yans. YansWifiChannel::Send skips outright
    // any receiver whose channel number differs from the sender's -- "For now
    // don't account for inter channel interference nor channel bonding" -- so an
    // AP alone on its channel is perfectly isolated. The spectrum model carries
    // an 802.11 transmit spectral mask instead, so an adjacent-channel neighbour
    // deposits real power in the receiver's band and appears in its carrier
    // sense, which is what a client measuring occupancy would actually see.
    // Measured on matched scenarios, a channel with no offered load reads a
    // busy fraction of 0.072 here against 0.002 under Yans.
    SpectrumWifiPhyHelper wifiPhy;
    wifiPhy.Set("RxGain", DoubleValue(0.0));

    Ptr<MultiModelSpectrumChannel> sharedChannel =
        CreateObject<MultiModelSpectrumChannel>();

    // AddPropagationLossModel prepends (SpectrumChannel::AddPropagationLossModel
    // does loss->SetNext(head); head = loss), so these go in reverse to give a
    // log-distance -> shadowing -> Nakagami chain.
    sharedChannel->AddPropagationLossModel(
        CreateObject<NakagamiPropagationLossModel>());

    // Per-link shadowing, held constant for the run. A matrix model is the only
    // stock loss model whose value is a property of the pair rather than of the
    // call: RandomPropagationLossModel redraws every transmission, so it would
    // average out over a window of beacons exactly as Nakagami does. Unset pairs
    // must default to 0 dB, not the model's default of infinite loss.
    Ptr<MatrixPropagationLossModel> shadowingModel =
        CreateObject<MatrixPropagationLossModel>();
    shadowingModel->SetDefaultLoss(0.0);
    sharedChannel->AddPropagationLossModel(shadowingModel);

    Ptr<LogDistancePropagationLossModel> pathLoss =
        CreateObject<LogDistancePropagationLossModel>();
    pathLoss->SetAttribute("Exponent", DoubleValue(kPathLossExponent));
    pathLoss->SetAttribute("ReferenceDistance", DoubleValue(1.0));
    pathLoss->SetAttribute("ReferenceLoss", DoubleValue(46.6777));
    sharedChannel->AddPropagationLossModel(pathLoss);

    sharedChannel->SetPropagationDelayModel(
        CreateObject<ConstantSpeedPropagationDelayModel>());

    wifiPhy.SetChannel(sharedChannel);

    // Channel per AP is drawn independently, so co-channel separation and the
    // number of channels in use stop being functions of the AP count and the
    // fixed layout. Both streams below hang off topologySeed alone: they must
    // be identical across the matched target-AP replays and across radio
    // seeds, and they must not consume from the placement stream.
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

    // Whatever the draw produced. Two APs may share a channel and four may
    // occupy one; the scanner set follows the channels actually in use rather
    // than a prefix of kApChannels.
    std::vector<uint32_t> occupiedChannels;
    for (uint8_t channel : apChannel)
    {
        occupiedChannels.push_back(channel);
    }
    std::sort(occupiedChannels.begin(), occupiedChannels.end());
    occupiedChannels.erase(std::unique(occupiedChannels.begin(), occupiedChannels.end()),
                           occupiedChannels.end());
    uint32_t nOccupiedChannels = static_cast<uint32_t>(occupiedChannels.size());

    // Offered load is drawn per station, so an AP's load is no longer its
    // station count times a scenario-wide constant. Two APs with equal station
    // counts can now differ in load, which is what stops a transmitter count
    // from standing in for the measurement.
    std::vector<double> staOfferedMbps(config.nStas, config.backgroundMbpsPerSta);
    if (config.backgroundMbpsPerSta <= 0.0)
    {
        // Per-user traffic volume is log-normal, not Gaussian: an 18-year
        // longitudinal study finds log-normal beats both Gaussian and Weibull
        // at every timescale tested, and WLAN flow sizes fit log-normal
        // regardless of device type. The right skew is what puts most stations
        // on a light load and a few on a heavy one without needing a mixture.
        std::lognormal_distribution<double> loadDraw(
            std::log(config.backgroundMedianMbps), config.backgroundSigmaLog);
        for (uint32_t sta = 0; sta < config.nStas; ++sta)
        {
            staOfferedMbps[sta] =
                std::min(config.backgroundCapMbps, loadDraw(rateRng));
        }
    }

    // One traffic stream per background station. Each station's mean rate is a
    // property of the deployment and hangs off topologySeed above; when its
    // packets actually leave is not, so the arrival realisation is drawn from
    // both seeds. A radio seed therefore means the same deployment observed on a
    // different occasion, rather than the identical packet sequence replayed
    // with different fading. Both seeds are fixed within a matched target-AP
    // replay, so the observation stays byte-identical across it.
    const uint32_t trafficBase =
        (config.topologySeed * 2654435761u) ^ (config.rngSeed * 2246822519u);
    gTrafficRngs.clear();
    gTrafficRngs.reserve(config.nStas);
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        gTrafficRngs.emplace_back(trafficBase + 101u + sta);
    }

    Topology topology = BuildTopology(config);

    // -----------------------------------------------------------------------
    // Nodes and Wi-Fi devices.
    //
    // Background stations associate by strongest shadowed received power, as
    // computed in BuildTopology and applied by the channel. Each AP
    // uses a distinct SSID so a matched variant can force the candidate onto
    // exactly one option without changing any other part of the deployment.
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
        uint32_t ap = topology.staServingAp[sta];
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(apChannel[ap])));
        wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, backgroundNodes.Get(sta)));
        apStaGlobalIndex[ap].push_back(sta);
    }

    // Scanner radios are passive and never associate. One scanner per occupied
    // channel gives the raw observation for the whole guarded window. The
    // dataset builder subsequently applies the dwell schedule of one sweeping
    // radio, using these parallel captures as the complete source material.
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

    // The association radio remains on an unused channel with an unmatched
    // SSID until the decision time. It then retunes to the selected AP and
    // adopts that AP's SSID. Association is allowed to complete normally, so
    // the throughput window begins at the traced completion time rather than
    // at an assumed fixed delay.
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
    // ListPositionAllocator consumes positions in exactly the order in which
    // allNodes was assembled: APs, background stations, then the candidate.
    // Each AP group receives its own IPv4 subnet; the candidate receives an
    // address only in the subnet of the AP selected for this variant.
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

    // Now that every node has a mobility model, hand the channel the same
    // shadowing draws the association model used. SetLoss takes attenuation, so
    // the sign flips: a positive draw was extra received power there.
    if (config.linkShadowingDb > 0.0)
    {
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
    // Uplink traffic and byte accounting.
    //
    // Every background station sends to a dedicated socket on its AP. The
    // candidate is offered 100 Mbps so it remains backlogged; its delivered
    // rate therefore measures capacity won from the selected BSS rather than
    // merely reproducing a low application sending rate.
    // -----------------------------------------------------------------------
    TypeId udpSocketType = TypeId::LookupByName("ns3::UdpSocketFactory");
    std::vector<std::vector<std::size_t>> apFlowStats(config.nAps);

    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        for (uint32_t localSta = 0; localSta < apStaGlobalIndex[ap].size(); ++localSta)
        {
            Address sinkAddress = InetSocketAddress(apAddress[ap], kBasePort + localSta);
            apFlowStats[ap].push_back(InstallSink(apNodes.Get(ap), sinkAddress, udpSocketType));

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
    std::size_t candidateFlowIndex =
        InstallSink(apNodes.Get(config.targetAp), candidateSinkAddress, udpSocketType);
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
    // Only one variant in a matched set needs to write the shared observation;
    // every variant still installs identical scanner devices so trace capture
    // cannot alter the simulated radio configuration or RNG sequence.
    // -----------------------------------------------------------------------
    std::string runDir = config.outDir + "/" + config.runTag;
    std::filesystem::create_directories(runDir);

    if (config.captureObservation)
    {
        gObservation.windowStart = kObservationStartS;
        gObservation.windowEnd = kCandidateStartS - kFeatureGuardS;
        gObservation.frames.open(runDir + "/observation.csv");
        gObservation.channelBusy.open(runDir + "/chanbusy.csv");
        NS_ABORT_MSG_IF(!gObservation.frames.is_open(), "could not open observation.csv");
        NS_ABORT_MSG_IF(!gObservation.channelBusy.is_open(), "could not open chanbusy.csv");

        gObservation.frames
            << "t,freq_mhz,bssid,ta,cat,is_beacon,retry,len,signal_dbm,noise_dbm,"
               "duration_us,rate_mbps\n";
        gObservation.channelBusy << "start,end,channel,freq_mhz,busy_frac\n";
        gObservation.busyByChannel.assign(nOccupiedChannels, BusyMeter{});
        gObservation.busyChannelNumbers = occupiedChannels;

        for (uint32_t channelIndex = 0; channelIndex < nOccupiedChannels; ++channelIndex)
        {
            scannerDevices[channelIndex]->GetPhy()->TraceConnectWithoutContext(
                "MonitorSnifferRx",
                MakeCallback(&MonitorSniffRx));
            scannerDevices[channelIndex]->GetPhy()->GetState()->TraceConnectWithoutContext(
                "State",
                MakeBoundCallback(&ScannerPhyState, occupiedChannels[channelIndex]));
        }
    }

    // The 0.5 s guard excludes ns-3's target-dependent pre-join polling near
    // association time. Once that guarded window closes, scanners move to the
    // parking channel because no later frame contributes to the observation.
    for (const auto& scanner : scannerDevices)
    {
        Simulator::Schedule(Seconds(kCandidateStartS - kFeatureGuardS),
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
    // Candidate throughput is averaged from actual association completion to
    // simulation stop. Bytes sent during scanning are not delivered and do not
    // enter the numerator. Metadata records both observable identities and
    // simulator-only ground truth needed to validate the generated dataset.
    // -----------------------------------------------------------------------
    bool candidateAssociated = gAssociation.time >= 0.0;
    double candidateObservedSeconds =
        candidateAssociated ? kSimulationStopS - gAssociation.time : 0.0;
    double candidateMbps =
        candidateAssociated && candidateObservedSeconds > 0.0
            ? gFlowStats[candidateFlowIndex].bytesTotal * 8.0 / 1e6 /
                  candidateObservedSeconds
            : 0.0;

    std::string associatedAp =
        candidateAssociated ? MacToString(gAssociation.ap) : std::string{};
    double backgroundObservedSeconds = kSimulationStopS - kBackgroundStartS;

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
    metadata << "    \"feature_guard\": " << kFeatureGuardS << ",\n";
    metadata << "    \"observation_start_time\": " << kObservationStartS << ",\n";
    // Emitted as the window DURATION: frame and bucket timestamps are relative
    // to the window start, so downstream still sees a window beginning at 0.
    metadata << "    \"feature_window_end\": "
             << (kCandidateStartS - kFeatureGuardS - kObservationStartS) << ",\n";
    metadata << "    \"sim_stop_time\": " << kSimulationStopS << ",\n";
    metadata << "    \"background_start_time\": " << kBackgroundStartS << ",\n";
    metadata << "    \"candidate_interval_s\": " << candidateInterval.GetSeconds()
             << ",\n";
    double bgTotalOffered = 0.0;
    for (double rate : staOfferedMbps)
    {
        bgTotalOffered += rate;
    }
    metadata << "    \"link_shadowing_db\": " << config.linkShadowingDb << ",\n";
    metadata << "    \"bg_load_median_mbps\": " << config.backgroundMedianMbps
             << ",\n";
    metadata << "    \"bg_load_sigma_log\": " << config.backgroundSigmaLog << ",\n";
    metadata << "    \"bg_load_cap_mbps\": " << config.backgroundCapMbps << ",\n";
    metadata << "    \"bg_per_sta_fixed_mbps\": " << config.backgroundMbpsPerSta << ",\n";
    metadata << "    \"bg_mean_per_sta_mbps\": "
             << (config.nStas ? bgTotalOffered / config.nStas : 0.0) << ",\n";
    metadata << "    \"bg_total_offered_mbps\": " << bgTotalOffered << ",\n";
    std::vector<double> apOfferedMbps(config.nAps, 0.0);
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        apOfferedMbps[topology.staServingAp[sta]] += staOfferedMbps[sta];
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
    metadata << "  \"candidate_position\": {\"x\": " << topology.candidatePosition.x
             << ", \"y\": " << topology.candidatePosition.y << "},\n";
    metadata << "  \"aps\": [\n";

    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        uint64_t backgroundBytes = 0;
        for (std::size_t flowIndex : apFlowStats[ap])
        {
            backgroundBytes += gFlowStats[flowIndex].bytesTotal;
        }
        double backgroundMbps =
            backgroundBytes * 8.0 / 1e6 / backgroundObservedSeconds;
        std::string mac = MacToString(
            Mac48Address::ConvertFrom(apGroupDevices[ap].Get(0)->GetAddress()));

        metadata << "    {\"index\": " << ap << ", \"ssid\": \""
                 << apSsid[ap].PeekString() << "\", \"mac\": \"" << mac
                 << "\", \"channel\": " << +apChannel[ap]
                 << ", \"position\": {\"x\": " << topology.apPositions[ap].x
                 << ", \"y\": " << topology.apPositions[ap].y
                 << "}, \"sta_count\": " << apStaGlobalIndex[ap].size()
                 << ", \"background_mbps\": " << backgroundMbps
                 << ", \"offered_mbps\": " << apOfferedMbps[ap]
                 << ", \"candidate_distance\": " << topology.candidateApDistance[ap]
                 << "}" << (ap + 1 < config.nAps ? ",\n" : "\n");
    }

    metadata << "  ],\n";
    metadata << "  \"background_stations\": [\n";
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        metadata << "    {\"index\": " << sta << ", \"ap\": "
                 << topology.staServingAp[sta] << ", \"position\": {\"x\": "
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
    metadata << "    \"bytes_total\": " << gFlowStats[candidateFlowIndex].bytesTotal
             << ",\n";
    metadata << "    \"throughput_mbps\": " << candidateMbps << "\n";
    metadata << "  }\n";
    metadata << "}\n";
    metadata.close();

    if (gObservation.frames.is_open())
    {
        gObservation.frames.close();
    }
    if (gObservation.channelBusy.is_open())
    {
        WriteChannelBusyCsv();
        gObservation.channelBusy.close();
    }

    Simulator::Destroy();
    return 0;
}
