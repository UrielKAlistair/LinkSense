#include "ns3/abort.h"
#include "ns3/ampdu-subframe-header.h"
#include "ns3/boolean.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/mac48-address.h"
#include "ns3/mobility-helper.h"
#include "ns3/multi-model-spectrum-channel.h"
#include "ns3/packet-sink-helper.h"
#include "ns3/propagation-loss-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/socket.h"
#include "ns3/spectrum-wifi-helper.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/uinteger.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mpdu.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy-state-helper.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-psdu.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
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
//   nCols              rooms per row of the floor, which has kRoomRows rows
//   nAPs               number of APs, at most one per room
//   meanOccupancy      mean background stations per room
//   occupancyShape     how unevenly rooms are occupied: small crowds a few
//                      rooms, large spreads stations evenly
//   loadMedianMbps     median of the load a background station offers
//   downlinkFraction   chance a station's flow runs from its AP to it
//   tcpFraction        chance a station's flow uses TCP rather than UDP
//   targetAP           the AP the candidate joins; -1 draws one it can hear
//   candidateSeed      seed for the candidate's draws listed below (or
//                      candidateX/Y to place the candidate by hand)
//   topologySeed       seed for the deployment draws listed below
//   rngSeed            seed for ns-3's radio randomness and for background
//                      traffic up to the join
//   futureSeed         seed for background traffic after the join
//   redrawSessions     1 also redraws, at the join, how long each station
//                      stays in the session or pause it is in
//   printLossMatrix    1 prints the loss between every pair of nodes (APs,
//                      then stations, then the candidate) and exits
//   printSessionsAtJoin  1 simulates up to the join, prints 1 or 0 per
//                      station for whether it is in a session then, and exits
//   outDir, runTag     where the outputs go
//
// Drawn inside from the seeds:
//   AP rooms, positions   topologySeed
//   AP channels           topologySeed
//   stations per room     topologySeed, a Poisson count with a gamma mean
//   station positions     topologySeed, uniform in the station's room
//   station traffic       topologySeed: offered load, direction, transport,
//                         mean session and pause lengths
//   candidate position    candidateSeed, uniform over the floor
//   link shadowing        topologySeed; the candidate's own links candidateSeed
//   drawn target AP       candidateSeed
//
// What happens:
//   0.5 s        background flows start, each alternating sessions and pauses
//   2.0-8.0 s    the candidate listens passively on every channel and records
//                every frame it hears
//   8.0 s        the candidate starts joining targetAP, and from then until
//                20 s offers more uplink traffic than its link can carry
//
// Outputs, in outDir/runTag/:
//   observation.csv   every frame from a transmission wholly inside the
//                     listening window
//   chanbusy.csv      per-channel busy fraction, one row per channel per 1 ms
//   metadata.json     parameters, the candidate's throughput (the label) with
//                     what makes it up, association outcome, and simulator
//                     ground truth
//
// Nothing before 8 s depends on targetAP or futureSeed. Running the same
// inputs once per AP gives the true throughput of every AP from one shared
// observation; running them once per futureSeed gives the spread of the
// throughput that one observation can be followed by. With redrawSessions
// that spread includes how long the traffic heard during the observation
// goes on, which the observation cannot tell.
// ---------------------------------------------------------------------------
constexpr uint16_t kBasePort = 8000;
constexpr uint32_t kPacketSize = 1250;
constexpr uint32_t kTcpSegmentBytes = 1448;

// The floor is kRoomRows rows of square rooms. Radio loss between two points
// follows the TGax residential model at 5 GHz: free-space-like up to the
// breakpoint, a steeper slope beyond it, and a fixed loss per wall crossed.
constexpr uint32_t kRoomRows = 2;
constexpr double kRoomSizeM = 10.0;
constexpr double kWallLossDb = 5.0;
constexpr double kBreakpointM = 5.0;
constexpr double kLossAtOneMetreDb = 46.425; // 40.05 + 20 log10(5 / 2.4)
constexpr double kSlopeBeyondBreakpoint = 35.0;
// Log-normal shadowing sigma. One draw per unordered node pair, fixed for the
// whole run, applied both by the association model and by the simulated
// channel, so the power a station used to pick its AP is the power the channel
// then delivers.
constexpr double kLinkShadowingDb = 5.0;
constexpr double kTxPowerDbm = 16.0;

// A drawn target AP is one whose mean received power at the candidate is at
// least this: 6 dB under the level at which a radio detects a preamble, so
// that an AP heard only on fading peaks can still be drawn.
constexpr double kDrawnTargetFloorDbm = -88.0;

// Background traffic needs ~1.5 s to reach steady state: association
// completes quickly but Minstrel-HT converges only after heavy retransmission.
constexpr double kBackgroundStartS = 0.5;
constexpr double kObservationStartS = 2.0;
constexpr double kCandidateStartS = 8.0; // listening ends and joining starts here
constexpr double kSimulationStopS = 20.0;

constexpr double kCandidateOfferedMbps = 100.0; // saturates the candidate's link
// A background station's offered load is one log-normal draw around the
// deployment's median. The cap sits above the fastest rate the radios have, so
// a station drawn at or near it always has a packet waiting.
constexpr double kBgLoadSigmaLog = 0.9;
constexpr double kBgLoadCapMbps = 80.0;
// A flow alternates sessions, in which it sends, and pauses. Each station
// draws its mean session length and mean pause length log-uniformly between
// these bounds.
constexpr double kSessionMinS = 0.3;
constexpr double kSessionMaxS = 30.0;

// Time resolution of chanbusy.csv. The listening window is cut into buckets of
// this length, and each row reports the fraction of one bucket during which the
// candidate's radio on that channel sensed the medium busy. 1 ms is the
// resolution at which real clients export channel busy time.
constexpr double kBusyBucketS = 0.001;

// Four non-overlapping 20 MHz channels in the 5 GHz band; each AP draws one
// uniformly at random, and the candidate listens on all four.
const std::vector<uint8_t> kApChannels = {36, 40, 44, 48};

// A channel no AP occupies. The candidate's association radio parks here until
// 8 s so it hears nothing and consumes no fading RNG draws, which is what keeps
// everything before 8 s independent of targetAP. Scanner radios move here when
// the listening window ends.
constexpr uint8_t kParkChannel = 149;

// Spreads nearby seeds apart before an offset picks one random stream.
constexpr uint32_t kSeedSpread = 2654435761u;

struct RunConfig
{
    uint32_t nCols{3};
    uint32_t nAps{3};
    double meanOccupancy{2.0};
    double occupancyShape{1.0};
    double loadMedianMbps{2.25};
    double downlinkFraction{0.5};
    double tcpFraction{0.5};
    int32_t targetAp{-1};
    // NaN when not given, in which case the position is drawn from candidateSeed.
    double candidateX{std::numeric_limits<double>::quiet_NaN()};
    double candidateY{std::numeric_limits<double>::quiet_NaN()};
    uint32_t candidateSeed{1};
    uint32_t topologySeed{1};
    uint32_t rngSeed{1};
    uint32_t futureSeed{1};
    bool redrawSessions{false};
    bool printLossMatrix{false};
    bool printSessionsAtJoin{false};
    std::string outDir{"runs"};
    std::string runTag;
    bool captureObservation{true};
};

// What one background station's flow carries.
struct StationTraffic
{
    double offeredMbps;
    bool isDownlink;
    bool isTcp;
    double meanSessionS;
    double meanPauseS;
};

struct Topology
{
    std::vector<Vector> apPositions;
    std::vector<Vector> staPositions;
    std::vector<uint32_t> staAssociatedAp; // What AP is each STA associated with
    std::vector<StationTraffic> staTraffic;
    Vector candidatePosition;
    std::vector<double> candidateApRxDbm; // mean power of each AP at the candidate
    uint32_t targetAp;
    // Loss in dB between every pair of nodes: distance, walls and shadowing.
    // lossDb[i][j] == lossDb[j][i]. Nodes are numbered APs first, then
    // background stations, then the candidate: the same order main() adds
    // them to allNodes, which is how the channel looks them up.
    std::vector<std::vector<double>> lossDb;
};

// ---------------------------------------------------------------------------
// Command-line options set the deployment, the candidate, the seeds and the
// output location.
// ---------------------------------------------------------------------------
RunConfig
ParseRunConfig(int argc, char* argv[])
{
    RunConfig config;
    CommandLine cmd(__FILE__);

    cmd.AddValue("nCols", "rooms per row of the floor", config.nCols);
    cmd.AddValue("nAPs", "number of APs, at most one per room", config.nAps);
    cmd.AddValue("meanOccupancy", "mean background stations per room", config.meanOccupancy);
    cmd.AddValue("occupancyShape",
                 "gamma shape of a room's mean occupancy; small crowds a few rooms",
                 config.occupancyShape);
    cmd.AddValue("loadMedianMbps",
                 "median offered load of a background station",
                 config.loadMedianMbps);
    cmd.AddValue("downlinkFraction",
                 "chance a station's flow runs from its AP to it",
                 config.downlinkFraction);
    cmd.AddValue("tcpFraction", "chance a station's flow uses TCP", config.tcpFraction);
    cmd.AddValue("targetAP",
                 "index of the AP the candidate joins; -1 draws one it can hear",
                 config.targetAp);
    cmd.AddValue("candidateX",
                 "candidate x coordinate in metres; omit both coordinates to draw "
                 "the position from candidateSeed",
                 config.candidateX);
    cmd.AddValue("candidateY", "candidate y coordinate in metres", config.candidateY);
    cmd.AddValue("candidateSeed",
                 "seeds the candidate's shadowing links, its drawn target AP, and "
                 "its position when candidateX/candidateY are omitted",
                 config.candidateSeed);
    cmd.AddValue("topologySeed",
                 "seeds the whole deployment: AP and station placement, channel "
                 "assignment, station traffic, and the shadowing between every "
                 "pair of non-candidate nodes",
                 config.topologySeed);
    cmd.AddValue("rngSeed",
                 "seeds ns-3's radio randomness, and background traffic until "
                 "the candidate joins",
                 config.rngSeed);
    cmd.AddValue("futureSeed",
                 "seeds background traffic after the candidate joins",
                 config.futureSeed);
    cmd.AddValue("redrawSessions",
                 "1 redraws the length of every session and pause in force at the join",
                 config.redrawSessions);
    cmd.AddValue("printLossMatrix",
                 "1 prints the loss in dB between every pair of nodes and exits",
                 config.printLossMatrix);
    cmd.AddValue("printSessionsAtJoin",
                 "1 simulates up to the join, prints which stations are in a session, and exits",
                 config.printSessionsAtJoin);
    cmd.AddValue("outDir", "directory under which the run directory is written", config.outDir);
    cmd.AddValue("runTag", "run directory name", config.runTag);
    cmd.AddValue("captureObs",
                 "write observation.csv and chanbusy.csv; 0 skips them without "
                 "changing the simulation",
                 config.captureObservation);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(config.nCols == 0, "nCols must be at least 1");
    NS_ABORT_MSG_IF(config.nAps == 0 || config.nAps > kRoomRows * config.nCols,
                    "nAPs must be between 1 and the number of rooms");
    NS_ABORT_MSG_IF(config.targetAp >= static_cast<int32_t>(config.nAps),
                    "targetAP must be less than nAPs");
    NS_ABORT_MSG_IF(config.runTag.empty(), "runTag must be given");
    NS_ABORT_MSG_IF(std::isnan(config.candidateX) != std::isnan(config.candidateY),
                    "give both candidateX and candidateY, or neither");
    return config;
}

// ---------------------------------------------------------------------------
// The floor and the loss across it.
//
// Rooms are kRoomSizeM squares in kRoomRows rows, room (0, 0) with its corner
// at the origin. A link crosses one wall for every room boundary between its
// ends, counted along x and along y separately.
// ---------------------------------------------------------------------------
uint32_t
WallsBetween(const Vector& a, const Vector& b)
{
    auto cell = [](double coordinate) {
        return static_cast<int>(std::floor(coordinate / kRoomSizeM));
    };
    return std::abs(cell(a.x) - cell(b.x)) + std::abs(cell(a.y) - cell(b.y));
}

// Loss between two points before shadowing, the distance never taken below 1 m.
double
PathLossDb(const Vector& a, const Vector& b)
{
    double distance = std::max(1.0, std::hypot(a.x - b.x, a.y - b.y));
    double loss = kLossAtOneMetreDb + 20.0 * std::log10(std::min(distance, kBreakpointM));
    if (distance > kBreakpointM)
    {
        loss += kSlopeBeyondBreakpoint * std::log10(distance / kBreakpointM);
    }
    return loss + kWallLossDb * WallsBetween(a, b);
}

// A uniformly random point inside one room.
Vector
PointInRoom(uint32_t room, uint32_t nCols, std::mt19937& rng)
{
    std::uniform_real_distribution<double> along(0.0, kRoomSizeM);
    double x = (room % nCols) * kRoomSizeM + along(rng);
    double y = (room / nCols) * kRoomSizeM + along(rng);
    return Vector(x, y, 0.0);
}

double
LogUniform(double low, double high, std::mt19937& rng)
{
    std::uniform_real_distribution<double> exponent(std::log(low), std::log(high));
    return std::exp(exponent(rng));
}

// ---------------------------------------------------------------------------
// Builds the parts of the deployment that are fixed before simulation starts:
//   1. APs: nAPs distinct rooms, each AP uniform inside its room.
//   2. Background stations: each room's count is Poisson around its own mean,
//      the means gamma-distributed around meanOccupancy; each station uniform
//      inside its room.
//   3. Each station's traffic.
//   4. The candidate's position, uniform over the floor unless given.
//   5. The loss between every pair of nodes: distance, walls, and a shadowing
//      draw per pair.
//   6. Each station joins the AP it receives most strongly.
//   7. The target AP, drawn among those the candidate can hear unless given.
// ---------------------------------------------------------------------------
Topology
BuildTopology(const RunConfig& config)
{
    Topology topology;
    const uint32_t nRooms = kRoomRows * config.nCols;
    std::mt19937 placementRng(config.topologySeed);

    // 1. APs.
    std::vector<uint32_t> rooms(nRooms);
    std::iota(rooms.begin(), rooms.end(), 0);
    std::shuffle(rooms.begin(), rooms.end(), placementRng);
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        topology.apPositions.push_back(PointInRoom(rooms[ap], config.nCols, placementRng));
    }

    // 2. Background stations.
    std::gamma_distribution<double> roomMean(config.occupancyShape,
                                             config.meanOccupancy / config.occupancyShape);
    for (uint32_t room = 0; room < nRooms; ++room)
    {
        std::poisson_distribution<uint32_t> count(
            std::max(roomMean(placementRng), 1e-9));
        uint32_t nInRoom = count(placementRng);
        for (uint32_t sta = 0; sta < nInRoom; ++sta)
        {
            topology.staPositions.push_back(PointInRoom(room, config.nCols, placementRng));
        }
    }
    const uint32_t nStas = topology.staPositions.size();

    // 3. Station traffic, on its own stream so it never shifts a placement draw.
    std::mt19937 trafficRng(config.topologySeed * kSeedSpread + 2u);
    std::lognormal_distribution<double> loadDraw(std::log(config.loadMedianMbps),
                                                 kBgLoadSigmaLog);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    for (uint32_t sta = 0; sta < nStas; ++sta)
    {
        StationTraffic traffic;
        traffic.offeredMbps = std::min(kBgLoadCapMbps, loadDraw(trafficRng));
        traffic.isDownlink = unit(trafficRng) < config.downlinkFraction;
        traffic.isTcp = unit(trafficRng) < config.tcpFraction;
        traffic.meanSessionS = LogUniform(kSessionMinS, kSessionMaxS, trafficRng);
        traffic.meanPauseS = LogUniform(kSessionMinS, kSessionMaxS, trafficRng);
        topology.staTraffic.push_back(traffic);
    }

    // 4. The candidate, on its own stream so that moving it never shifts a
    // background draw.
    if (std::isnan(config.candidateX))
    {
        std::mt19937 candidateRng(config.candidateSeed);
        std::uniform_real_distribution<double> alongX(0.0, config.nCols * kRoomSizeM);
        std::uniform_real_distribution<double> alongY(0.0, kRoomRows * kRoomSizeM);
        double x = alongX(candidateRng);
        double y = alongY(candidateRng);
        topology.candidatePosition = Vector(x, y, 0.0);
    }
    else
    {
        topology.candidatePosition = Vector(config.candidateX, config.candidateY, 0.0);
    }

    // 5. Loss between every pair of nodes.
    std::vector<Vector> positions = topology.apPositions;
    positions.insert(positions.end(), topology.staPositions.begin(), topology.staPositions.end());
    positions.push_back(topology.candidatePosition);
    const uint32_t nNodes = positions.size();
    const uint32_t candidateIndex = nNodes - 1;

    // The candidate's links use candidateSeed, so moving the candidate leaves
    // every background pair unchanged. Each stream needs its own distribution
    // object: std::normal_distribution generates values in pairs and keeps the
    // spare inside the object, whichever engine the next call passes.
    std::mt19937 shadowRng(config.topologySeed * kSeedSpread + 4u);
    std::mt19937 candidateShadowRng(config.candidateSeed * kSeedSpread + 7u);
    std::normal_distribution<double> shadowDraw(0.0, kLinkShadowingDb);
    std::normal_distribution<double> candidateShadowDraw(0.0, kLinkShadowingDb);
    topology.lossDb.assign(nNodes, std::vector<double>(nNodes, 0.0));
    for (uint32_t i = 0; i < nNodes; ++i)
    {
        for (uint32_t j = i + 1; j < nNodes; ++j)
        {
            double shadowing = j == candidateIndex ? candidateShadowDraw(candidateShadowRng)
                                                   : shadowDraw(shadowRng);
            double loss = PathLossDb(positions[i], positions[j]) + shadowing;
            topology.lossDb[i][j] = loss;
            topology.lossDb[j][i] = loss;
        }
    }

    // 6. Each station joins the AP with the least loss to it.
    for (uint32_t sta = 0; sta < nStas; ++sta)
    {
        const std::vector<double>& loss = topology.lossDb[config.nAps + sta];
        topology.staAssociatedAp.push_back(
            std::min_element(loss.begin(), loss.begin() + config.nAps) - loss.begin());
    }

    // 7. The target AP.
    std::vector<uint32_t> audibleAps;
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        topology.candidateApRxDbm.push_back(kTxPowerDbm - topology.lossDb[candidateIndex][ap]);
        if (topology.candidateApRxDbm[ap] >= kDrawnTargetFloorDbm)
        {
            audibleAps.push_back(ap);
        }
    }
    if (config.targetAp >= 0)
    {
        topology.targetAp = config.targetAp;
    }
    else if (audibleAps.empty())
    {
        topology.targetAp =
            std::max_element(topology.candidateApRxDbm.begin(), topology.candidateApRxDbm.end()) -
            topology.candidateApRxDbm.begin();
    }
    else
    {
        std::mt19937 targetRng(config.candidateSeed * kSeedSpread + 11u);
        std::uniform_int_distribution<std::size_t> pick(0, audibleAps.size() - 1);
        topology.targetAp = audibleAps[pick(targetRng)];
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
// Background traffic.
//
// Every background station owns one flow, uplink or downlink, over UDP or TCP.
// The flow alternates sessions and pauses of exponentially distributed length.
// Inside a session its sender offers packets at exponentially distributed
// gaps, a Poisson stream at the station's offered load. A TCP flow offers the
// same stream to a TCP socket, which then paces what reaches the air and drops
// what its buffer cannot hold.
//
// Each station draws from two generators of its own: one for every draw made
// before the candidate joins, seeded from topologySeed and rngSeed, and one
// for every draw made afterwards, which futureSeed also seeds.
// ---------------------------------------------------------------------------
struct TrafficStreams
{
    std::mt19937 beforeJoin;
    std::mt19937 afterJoin;
};

struct BackgroundFlow
{
    Ptr<Socket> source;
    InetSocketAddress destination;
    double meanPacketGapS;
    double meanSessionS;
    double meanPauseS;
    double sessionEndS;
    bool isInSession;
    EventId nextSession;
};

std::vector<TrafficStreams> gTrafficStreams;
std::vector<BackgroundFlow> gFlows;

std::mt19937&
TrafficRng(std::size_t sta)
{
    return Simulator::Now().GetSeconds() < kCandidateStartS ? gTrafficStreams[sta].beforeJoin
                                                            : gTrafficStreams[sta].afterJoin;
}

double
ExponentialDraw(double mean, std::size_t sta)
{
    return std::exponential_distribution<double>(1.0 / mean)(TrafficRng(sta));
}

void StartSession(std::size_t sta);

// Sends one packet and schedules the next, or starts a pause once the session
// has run out.
void
SendInSession(std::size_t sta)
{
    BackgroundFlow& flow = gFlows[sta];
    if (Simulator::Now().GetSeconds() >= flow.sessionEndS)
    {
        flow.isInSession = false;
        flow.nextSession =
            Simulator::Schedule(Seconds(ExponentialDraw(flow.meanPauseS, sta)), &StartSession, sta);
        return;
    }
    flow.source->Send(Create<Packet>(kPacketSize));
    Simulator::Schedule(Seconds(ExponentialDraw(flow.meanPacketGapS, sta)), &SendInSession, sta);
}

void
StartSession(std::size_t sta)
{
    gFlows[sta].isInSession = true;
    gFlows[sta].sessionEndS =
        Simulator::Now().GetSeconds() + ExponentialDraw(gFlows[sta].meanSessionS, sta);
    SendInSession(sta);
}

// Redraws how long the flow stays in the session or pause it is in. Both
// lengths are exponential, so what remains of one is distributed like a new
// one, and the redrawn future is as likely as the one it replaces.
void
RedrawSession(std::size_t sta)
{
    BackgroundFlow& flow = gFlows[sta];
    if (flow.isInSession)
    {
        flow.sessionEndS = Simulator::Now().GetSeconds() + ExponentialDraw(flow.meanSessionS, sta);
    }
    else
    {
        flow.nextSession.Cancel();
        flow.nextSession =
            Simulator::Schedule(Seconds(ExponentialDraw(flow.meanPauseS, sta)), &StartSession, sta);
    }
}

// Connects the flow and puts it in a session or a pause, each with the chance
// the flow spends in that state in the long run.
void
StartFlow(std::size_t sta)
{
    BackgroundFlow& flow = gFlows[sta];
    flow.source->Connect(flow.destination);
    double sessionShare = flow.meanSessionS / (flow.meanSessionS + flow.meanPauseS);
    if (std::uniform_real_distribution<double>(0.0, 1.0)(TrafficRng(sta)) < sessionShare)
    {
        StartSession(sta);
    }
    else
    {
        flow.nextSession =
            Simulator::Schedule(Seconds(ExponentialDraw(flow.meanPauseS, sta)), &StartSession, sta);
    }
}

void
DiscardReceived(Ptr<Socket> socket)
{
    while (socket->Recv())
    {
    }
}

void
AcceptedConnection(Ptr<Socket> socket, const Address&)
{
    socket->SetRecvCallback(MakeCallback(&DiscardReceived));
}

// A socket that receives one flow and reads everything that arrives, so a TCP
// sender is never held back by its receiver.
Ptr<Socket>
InstallSink(Ptr<Node> node, uint16_t port, bool isTcp)
{
    Ptr<Socket> sink = Socket::CreateSocket(
        node,
        TypeId::LookupByName(isTcp ? "ns3::TcpSocketFactory" : "ns3::UdpSocketFactory"));
    sink->Bind(InetSocketAddress(Ipv4Address::GetAny(), port));
    if (isTcp)
    {
        sink->Listen();
        sink->SetAcceptCallback(MakeNullCallback<bool, Ptr<Socket>, const Address&>(),
                                MakeCallback(&AcceptedConnection));
    }
    else
    {
        sink->SetRecvCallback(MakeCallback(&DiscardReceived));
    }
    return sink;
}

// ---------------------------------------------------------------------------
// The candidate: its traffic, and what it achieved.
//
// The candidate sends one packet every packetInterval, a constant rate above
// what its link can carry, so it always has a packet waiting to send. Its
// throughput is the bytes that reach its socket on the AP. Beside that we
// count what the throughput is made of: how long the candidate's radio spent
// transmitting, at what rates, and how many of the data frames it sent were
// acknowledged.
// ---------------------------------------------------------------------------
struct AssociationResult
{
    double time{-1.0}; // stays -1 if the candidate never associates
    Mac48Address ap;
};

struct CandidateOutcome
{
    AssociationResult association;
    // Delivered bytes in each whole second after kCandidateStartS.
    std::vector<uint64_t> bytesPerSecond =
        std::vector<uint64_t>(static_cast<std::size_t>(kSimulationStopS - kCandidateStartS), 0);
    double transmitSeconds{0.0};
    uint64_t dataFramesSent{0};
    uint64_t dataFramesAcked{0};
    double sentRateSumMbps{0.0}; // PHY rate summed over the data frames sent
};

CandidateOutcome gCandidate;

void
GenerateCandidateTraffic(Ptr<Socket> socket, Time packetInterval)
{
    socket->Send(Create<Packet>(kPacketSize));
    Simulator::Schedule(packetInterval, &GenerateCandidateTraffic, socket, packetInterval);
}

void
ReceiveCandidatePacket(Ptr<Socket> socket)
{
    std::size_t second = std::min<std::size_t>(Simulator::Now().GetSeconds() - kCandidateStartS,
                                               gCandidate.bytesPerSecond.size() - 1);
    Ptr<Packet> packet;
    while ((packet = socket->Recv()))
    {
        gCandidate.bytesPerSecond[second] += packet->GetSize();
    }
}

void
CandidateAssociated(Mac48Address apAddress)
{
    if (gCandidate.association.time < 0.0)
    {
        gCandidate.association.time = Simulator::Now().GetSeconds();
        gCandidate.association.ap = apAddress;
    }
}

void
CandidateTransmits(WifiConstPsduMap psduMap, WifiTxVector txVector, double)
{
    double rateMbps = txVector.GetMode().GetDataRate(txVector) / 1e6;
    for (const auto& [staId, psdu] : psduMap)
    {
        for (std::size_t i = 0; i < psdu->GetNMpdus(); ++i)
        {
            if (psdu->GetHeader(i).IsQosData())
            {
                gCandidate.dataFramesSent += 1;
                gCandidate.sentRateSumMbps += rateMbps;
            }
        }
    }
}

void
CandidateAcked(Ptr<const WifiMpdu> mpdu)
{
    if (mpdu->GetHeader().IsQosData())
    {
        gCandidate.dataFramesAcked += 1;
    }
}

void
CandidatePhyState(Time, Time duration, WifiPhyState state)
{
    if (state == WifiPhyState::TX)
    {
        gCandidate.transmitSeconds += duration.GetSeconds();
    }
}

} // namespace

int
main(int argc, char* argv[])
{
    RunConfig config = ParseRunConfig(argc, argv);
    RngSeedManager::SetSeed(config.rngSeed);
    Config::SetDefault("ns3::TcpSocket::SegmentSize", UintegerValue(kTcpSegmentBytes));

    Topology topology = BuildTopology(config);
    if (config.printLossMatrix)
    {
        for (const std::vector<double>& row : topology.lossDb)
        {
            for (double loss : row)
            {
                std::cout << loss << " ";
            }
            std::cout << "\n";
        }
        return 0;
    }
    const uint32_t nStas = topology.staPositions.size();
    const uint32_t targetAp = topology.targetAp;

    // -----------------------------------------------------------------------
    // Propagation.
    //
    // Every radio shares one MultiModelSpectrumChannel. Channel numbers do not
    // isolate radios from one another: a transmission on an adjacent channel
    // still puts power into a receiver's band. The loss between two nodes is
    // the fixed value BuildTopology drew for the pair, followed by Nakagami
    // fading drawn anew for every transmission.
    // -----------------------------------------------------------------------
    Ptr<MultiModelSpectrumChannel> sharedChannel = CreateObject<MultiModelSpectrumChannel>();

    // AddPropagationLossModel prepends, so the model added last is applied first.
    sharedChannel->AddPropagationLossModel(CreateObject<NakagamiPropagationLossModel>());
    Ptr<MatrixPropagationLossModel> pairLoss = CreateObject<MatrixPropagationLossModel>();
    sharedChannel->AddPropagationLossModel(pairLoss);
    sharedChannel->SetPropagationDelayModel(CreateObject<ConstantSpeedPropagationDelayModel>());

    // -----------------------------------------------------------------------
    // AP channels and SSIDs, and the stations' packet-timing generators.
    // -----------------------------------------------------------------------
    std::mt19937 channelRng(config.topologySeed * kSeedSpread + 1u);
    std::uniform_int_distribution<std::size_t> channelPick(0, kApChannels.size() - 1);
    std::vector<uint8_t> apChannel(config.nAps);
    std::vector<Ssid> apSsid;
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        apChannel[ap] = kApChannels[channelPick(channelRng)];
        apSsid.emplace_back("wifi-ap" + std::to_string(ap));
    }

    // The distinct channels in use, in ascending order. APs can share a
    // channel, so there may be fewer of these than APs.
    std::set<uint32_t> distinctChannels(apChannel.begin(), apChannel.end());
    std::vector<uint32_t> occupiedChannels(distinctChannels.begin(), distinctChannels.end());

    const uint32_t trafficBase =
        (config.topologySeed * kSeedSpread) ^ (config.rngSeed * 2246822519u);
    for (uint32_t sta = 0; sta < nStas; ++sta)
    {
        gTrafficStreams.push_back(
            {std::mt19937(trafficBase + 101u + sta),
             std::mt19937((trafficBase ^ (config.futureSeed * 3266489917u)) + 50101u + sta)});
    }

    // -----------------------------------------------------------------------
    // Nodes and Wi-Fi devices.
    //
    // Every AP has its own SSID. Each background station is given the SSID of
    // the AP chosen for it in BuildTopology, so it can only associate with
    // that AP.
    // -----------------------------------------------------------------------
    NodeContainer apNodes;
    NodeContainer backgroundNodes;
    NodeContainer candidateNode;
    apNodes.Create(config.nAps);
    backgroundNodes.Create(nStas);
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
    wifiPhy.Set("TxPowerStart", DoubleValue(kTxPowerDbm));
    wifiPhy.Set("TxPowerEnd", DoubleValue(kTxPowerDbm));

    WifiMacHelper wifiMac;
    std::vector<NetDeviceContainer> apGroupDevices(config.nAps);
    std::vector<uint32_t> apStaCount(config.nAps, 0);

    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(apChannel[ap])));
        wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, apNodes.Get(ap)));
    }

    // Where each station's device sits in its AP's group, which is where its
    // address will sit among the group's addresses.
    std::vector<uint32_t> staSlotInGroup(nStas);
    for (uint32_t sta = 0; sta < nStas; ++sta)
    {
        uint32_t ap = topology.staAssociatedAp[sta];
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(apChannel[ap])));
        wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(apSsid[ap]));
        staSlotInGroup[sta] = apGroupDevices[ap].GetN();
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, backgroundNodes.Get(sta)));
        apStaCount[ap] += 1;
    }

    // One listening radio per occupied channel, all on the candidate node. They
    // do not probe and their SSID matches no AP, so they never associate or
    // transmit.
    std::vector<Ptr<WifiNetDevice>> scannerDevices;
    for (uint32_t channel : occupiedChannels)
    {
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(channel)));
        wifiMac.SetType("ns3::StaWifiMac",
                        "Ssid",
                        SsidValue(Ssid("scanner-never-associates")),
                        "ActiveProbing",
                        BooleanValue(false));
        NetDeviceContainer installed = wifi.Install(wifiPhy, wifiMac, candidateNode.Get(0));
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
    apGroupDevices[targetAp].Add(candidateDeviceContainer);

    // -----------------------------------------------------------------------
    // Positions, pair losses and addresses.
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

    for (uint32_t i = 0; i < allNodes.GetN(); ++i)
    {
        for (uint32_t j = i + 1; j < allNodes.GetN(); ++j)
        {
            pairLoss->SetLoss(allNodes.Get(i)->GetObject<MobilityModel>(),
                              allNodes.Get(j)->GetObject<MobilityModel>(),
                              topology.lossDb[i][j],
                              true);
        }
    }

    InternetStackHelper internet;
    internet.Install(allNodes);

    Ipv4AddressHelper ipv4;
    std::vector<Ipv4InterfaceContainer> apGroupAddresses(config.nAps);
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        std::string subnet = "10.1." + std::to_string(ap + 1) + ".0";
        ipv4.SetBase(subnet.c_str(), "255.255.255.0");
        apGroupAddresses[ap] = ipv4.Assign(apGroupDevices[ap]);
    }

    // -----------------------------------------------------------------------
    // Traffic.
    //
    // Each background station's flow starts at kBackgroundStartS, offset by up
    // to one packet gap so stations do not all begin at the same instant. The
    // candidate sends to its own socket on targetAP from kCandidateStartS.
    // -----------------------------------------------------------------------
    for (uint32_t sta = 0; sta < nStas; ++sta)
    {
        const StationTraffic& traffic = topology.staTraffic[sta];
        uint32_t ap = topology.staAssociatedAp[sta];
        Ptr<Node> staNode = backgroundNodes.Get(sta);
        Ptr<Node> sender = traffic.isDownlink ? apNodes.Get(ap) : staNode;
        Ptr<Node> receiver = traffic.isDownlink ? staNode : apNodes.Get(ap);
        Ipv4Address receiverAddress =
            apGroupAddresses[ap].GetAddress(traffic.isDownlink ? staSlotInGroup[sta] : 0);
        uint16_t port = kBasePort + sta;

        InstallSink(receiver, port, traffic.isTcp);
        Ptr<Socket> source = Socket::CreateSocket(
            sender,
            TypeId::LookupByName(traffic.isTcp ? "ns3::TcpSocketFactory"
                                               : "ns3::UdpSocketFactory"));
        double meanPacketGapS = kPacketSize * 8.0 / (traffic.offeredMbps * 1e6);
        gFlows.push_back({source,
                          InetSocketAddress(receiverAddress, port),
                          meanPacketGapS,
                          traffic.meanSessionS,
                          traffic.meanPauseS,
                          0.0,
                          false,
                          EventId()});
        std::uniform_real_distribution<double> phase(0.0, meanPacketGapS);
        Simulator::ScheduleWithContext(
            sender->GetId(),
            Seconds(kBackgroundStartS + phase(gTrafficStreams[sta].beforeJoin)),
            &StartFlow,
            static_cast<std::size_t>(sta));
    }

    uint16_t candidatePort = kBasePort + nStas;
    Ptr<Socket> candidateSink = Socket::CreateSocket(
        apNodes.Get(targetAp),
        TypeId::LookupByName("ns3::UdpSocketFactory"));
    candidateSink->Bind(InetSocketAddress(Ipv4Address::GetAny(), candidatePort));
    candidateSink->SetRecvCallback(MakeCallback(&ReceiveCandidatePacket));

    Ptr<Socket> candidateSource = Socket::CreateSocket(
        candidateNode.Get(0),
        TypeId::LookupByName("ns3::UdpSocketFactory"));
    candidateSource->Connect(
        InetSocketAddress(apGroupAddresses[targetAp].GetAddress(0), candidatePort));
    Time candidateInterval = Seconds(kPacketSize * 8.0 / (kCandidateOfferedMbps * 1e6));
    Simulator::ScheduleWithContext(candidateNode.Get(0)->GetId(),
                                   Seconds(kCandidateStartS),
                                   &GenerateCandidateTraffic,
                                   candidateSource,
                                   candidateInterval);

    if (config.redrawSessions)
    {
        for (std::size_t sta = 0; sta < gFlows.size(); ++sta)
        {
            Simulator::ScheduleWithContext(gFlows[sta].source->GetNode()->GetId(),
                                           Seconds(kCandidateStartS),
                                           &RedrawSession,
                                           sta);
        }
    }

    // Retuning is scheduled before changing the SSID at the same timestamp.
    // ns-3 executes equal-time events in insertion order, so the MAC begins
    // searching for the selected SSID only after its PHY reaches that channel.
    auto setOperatingChannel =
        static_cast<void (WifiPhy::*)(const WifiPhy::ChannelTuple&)>(
            &WifiPhy::SetOperatingChannel);
    Simulator::Schedule(Seconds(kCandidateStartS),
                        setOperatingChannel,
                        candidateDevice->GetPhy(),
                        WifiPhy::ChannelTuple{apChannel[targetAp], 20, WIFI_PHY_BAND_5GHZ, 0});
    Simulator::Schedule(Seconds(kCandidateStartS),
                        &WifiMac::SetSsid,
                        candidateDevice->GetMac(),
                        apSsid[targetAp]);

    candidateDevice->GetMac()->TraceConnectWithoutContext("Assoc",
                                                          MakeCallback(&CandidateAssociated));
    candidateDevice->GetMac()->TraceConnectWithoutContext("AckedMpdu",
                                                          MakeCallback(&CandidateAcked));
    candidateDevice->GetPhy()->TraceConnectWithoutContext("PhyTxPsduBegin",
                                                          MakeCallback(&CandidateTransmits));
    candidateDevice->GetPhy()->GetState()->TraceConnectWithoutContext(
        "State",
        MakeCallback(&CandidatePhyState));

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
                            WifiPhy::ChannelTuple{kParkChannel, 20, WIFI_PHY_BAND_5GHZ, 0});
    }

    std::cout << "run=" << config.runTag << " topologySeed=" << config.topologySeed
              << " rngSeed=" << config.rngSeed << " nAPs=" << config.nAps
              << " nSTAs=" << nStas << " targetAP=" << targetAp << std::endl;

    if (config.printSessionsAtJoin)
    {
        Simulator::Stop(Seconds(kCandidateStartS));
        Simulator::Run();
        for (const BackgroundFlow& flow : gFlows)
        {
            std::cout << "session " << (flow.isInSession ? 1 : 0) << "\n";
        }
        Simulator::Destroy();
        return 0;
    }

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
    const AssociationResult& association = gCandidate.association;
    bool isAssociated = association.time >= 0.0;
    double associatedSeconds = isAssociated ? kSimulationStopS - association.time : 0.0;
    uint64_t candidateBytes = std::accumulate(gCandidate.bytesPerSecond.begin(),
                                              gCandidate.bytesPerSecond.end(),
                                              uint64_t{0});
    double candidateMbps =
        associatedSeconds > 0.0 ? candidateBytes * 8.0 / 1e6 / associatedSeconds : 0.0;

    std::vector<double> apOfferedMbps(config.nAps, 0.0);
    double totalOfferedMbps = 0.0;
    for (uint32_t sta = 0; sta < nStas; ++sta)
    {
        const StationTraffic& traffic = topology.staTraffic[sta];
        double meanMbps = traffic.offeredMbps * traffic.meanSessionS /
                          (traffic.meanSessionS + traffic.meanPauseS);
        apOfferedMbps[topology.staAssociatedAp[sta]] += meanMbps;
        totalOfferedMbps += meanMbps;
    }

    std::ofstream metadata(runDir + "/metadata.json");
    NS_ABORT_MSG_IF(!metadata.is_open(), "could not open metadata.json");
    metadata << "{\n";
    metadata << "  \"run_id\": \"" << config.runTag << "\",\n";
    metadata << "  \"params\": {\n";
    metadata << "    \"topology_seed\": " << config.topologySeed << ",\n";
    metadata << "    \"candidate_seed\": " << config.candidateSeed << ",\n";
    metadata << "    \"rng_seed\": " << config.rngSeed << ",\n";
    metadata << "    \"future_seed\": " << config.futureSeed << ",\n";
    metadata << "    \"redraw_sessions\": " << (config.redrawSessions ? 1 : 0) << ",\n";
    metadata << "    \"n_cols\": " << config.nCols << ",\n";
    metadata << "    \"n_rooms\": " << kRoomRows * config.nCols << ",\n";
    metadata << "    \"room_size_m\": " << kRoomSizeM << ",\n";
    metadata << "    \"n_aps\": " << config.nAps << ",\n";
    metadata << "    \"n_stas\": " << nStas << ",\n";
    metadata << "    \"n_channels\": " << occupiedChannels.size() << ",\n";
    metadata << "    \"mean_occupancy\": " << config.meanOccupancy << ",\n";
    metadata << "    \"occupancy_shape\": " << config.occupancyShape << ",\n";
    metadata << "    \"load_median_mbps\": " << config.loadMedianMbps << ",\n";
    metadata << "    \"downlink_fraction\": " << config.downlinkFraction << ",\n";
    metadata << "    \"tcp_fraction\": " << config.tcpFraction << ",\n";
    metadata << "    \"bg_total_offered_mbps\": " << totalOfferedMbps << ",\n";
    metadata << "    \"wall_loss_db\": " << kWallLossDb << ",\n";
    metadata << "    \"link_shadowing_db\": " << kLinkShadowingDb << ",\n";
    metadata << "    \"tx_power_dbm\": " << kTxPowerDbm << ",\n";
    metadata << "    \"packet_size\": " << kPacketSize << ",\n";
    metadata << "    \"candidate_offered_mbps\": " << kCandidateOfferedMbps << ",\n";
    metadata << "    \"background_start_time\": " << kBackgroundStartS << ",\n";
    metadata << "    \"observation_start_time\": " << kObservationStartS << ",\n";
    metadata << "    \"candidate_start_time\": " << kCandidateStartS << ",\n";
    // The end of the listening window on the clock observation.csv and
    // chanbusy.csv use, which reads 0 when listening begins.
    metadata << "    \"feature_window_end\": " << (kCandidateStartS - kObservationStartS)
             << ",\n";
    metadata << "    \"sim_stop_time\": " << kSimulationStopS << "\n";
    metadata << "  },\n";
    metadata << "  \"candidate_position\": {\"x\": " << topology.candidatePosition.x
             << ", \"y\": " << topology.candidatePosition.y << "},\n";

    metadata << "  \"aps\": [\n";
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        std::string mac =
            MacToString(Mac48Address::ConvertFrom(apGroupDevices[ap].Get(0)->GetAddress()));
        metadata << "    {\"index\": " << ap << ", \"ssid\": \"" << apSsid[ap].PeekString()
                 << "\", \"mac\": \"" << mac << "\", \"channel\": " << +apChannel[ap]
                 << ", \"position\": {\"x\": " << topology.apPositions[ap].x
                 << ", \"y\": " << topology.apPositions[ap].y
                 << "}, \"sta_count\": " << apStaCount[ap]
                 << ", \"offered_mbps\": " << apOfferedMbps[ap]
                 << ", \"candidate_rx_dbm\": " << topology.candidateApRxDbm[ap] << "}"
                 << (ap + 1 < config.nAps ? ",\n" : "\n");
    }
    metadata << "  ],\n";

    metadata << "  \"background_stations\": [\n";
    for (uint32_t sta = 0; sta < nStas; ++sta)
    {
        const StationTraffic& traffic = topology.staTraffic[sta];
        metadata << "    {\"index\": " << sta << ", \"ap\": " << topology.staAssociatedAp[sta]
                 << ", \"position\": {\"x\": " << topology.staPositions[sta].x
                 << ", \"y\": " << topology.staPositions[sta].y
                 << "}, \"offered_mbps\": " << traffic.offeredMbps
                 << ", \"is_downlink\": " << (traffic.isDownlink ? "true" : "false")
                 << ", \"is_tcp\": " << (traffic.isTcp ? "true" : "false")
                 << ", \"mean_session_s\": " << traffic.meanSessionS
                 << ", \"mean_pause_s\": " << traffic.meanPauseS << "}"
                 << (sta + 1 < nStas ? ",\n" : "\n");
    }
    metadata << "  ],\n";

    metadata << "  \"candidate\": {\n";
    metadata << "    \"target_ap\": " << targetAp << ",\n";
    metadata << "    \"associated\": " << (isAssociated ? "true" : "false") << ",\n";
    metadata << "    \"assoc_time\": " << association.time << ",\n";
    metadata << "    \"assoc_delay\": "
             << (isAssociated ? association.time - kCandidateStartS : -1.0) << ",\n";
    metadata << "    \"assoc_ap_mac\": \""
             << (isAssociated ? MacToString(association.ap) : std::string{}) << "\",\n";
    metadata << "    \"observed_seconds\": " << associatedSeconds << ",\n";
    metadata << "    \"bytes_total\": " << candidateBytes << ",\n";
    metadata << "    \"bytes_per_second\": [";
    for (std::size_t second = 0; second < gCandidate.bytesPerSecond.size(); ++second)
    {
        metadata << (second ? ", " : "") << gCandidate.bytesPerSecond[second];
    }
    metadata << "],\n";
    metadata << "    \"transmit_seconds\": " << gCandidate.transmitSeconds << ",\n";
    metadata << "    \"data_frames_sent\": " << gCandidate.dataFramesSent << ",\n";
    metadata << "    \"data_frames_acked\": " << gCandidate.dataFramesAcked << ",\n";
    metadata << "    \"mean_phy_rate_mbps\": "
             << (gCandidate.dataFramesSent
                     ? gCandidate.sentRateSumMbps / gCandidate.dataFramesSent
                     : 0.0)
             << ",\n";
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
