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
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-helper.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
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
constexpr double kApSpacingM = 30.0;
constexpr double kBackgroundStartS = 1.0;
constexpr double kCandidateStartS = 6.0;
constexpr double kFeatureGuardS = 0.5;
constexpr double kSimulationStopS = 18.0;
constexpr double kCandidateOfferedMbps = 100.0;
constexpr double kHotspotFraction = 0.70;
constexpr double kHotspotRadiusM = 10.0;
constexpr double kBusyBucketS = 0.1024;
constexpr double kPi = 3.14159265358979323846;

// Four non-overlapping 20 MHz channels in the 5 GHz band. APs use these in
// index order and reuse them only when a topology contains more than four APs.
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
    double backgroundMbpsPerSta{4.0};
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
    std::vector<uint32_t> staNearestAp;
    Vector candidatePosition;
    std::vector<double> candidateApDistance;
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
    cmd.AddValue("bgPerStaMbps",
                 "offered UDP load per background station in Mbps",
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
    NS_ABORT_MSG_IF(config.backgroundMbpsPerSta <= 0.0,
                    "bgPerStaMbps must be positive");
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
// AP geometry is deterministic. Two APs form one pair and three form an
// equilateral triangle. The four-, six-, and eight-AP cases use regular
// two-row grids with two, three, and four columns. Other accepted AP counts
// fall back to a compact row-major grid, with an incomplete final row centred
// under the row above. AP 0 remains at the origin, which keeps controlled
// distance tests simple.
// ---------------------------------------------------------------------------
std::vector<Vector>
BuildApPositions(uint32_t nAps)
{
    if (nAps == 3)
    {
        return {
            Vector(0.0, 0.0, 0.0),
            Vector(kApSpacingM, 0.0, 0.0),
            Vector(kApSpacingM / 2.0, kApSpacingM * std::sqrt(3.0) / 2.0, 0.0),
        };
    }

    uint32_t columns = nAps >= 4 && nAps <= 8 && nAps % 2 == 0
                           ? nAps / 2
                           : static_cast<uint32_t>(
                                 std::ceil(std::sqrt(static_cast<double>(nAps))));
    std::vector<Vector> positions;
    positions.reserve(nAps);

    for (uint32_t index = 0; index < nAps; ++index)
    {
        uint32_t row = index / columns;
        uint32_t column = index % columns;
        uint32_t firstInRow = row * columns;
        uint32_t countInRow = std::min(columns, nAps - firstInRow);
        double rowOffset = (columns - countInRow) * kApSpacingM / 2.0;
        positions.emplace_back(rowOffset + column * kApSpacingM,
                               row * kApSpacingM,
                               0.0);
    }

    return positions;
}

uint32_t
FindNearestAp(double x, double y, const std::vector<Vector>& apPositions)
{
    uint32_t nearest = 0;
    double bestDistance = std::hypot(x - apPositions[0].x, y - apPositions[0].y);

    for (uint32_t ap = 1; ap < apPositions.size(); ++ap)
    {
        double distance = std::hypot(x - apPositions[ap].x, y - apPositions[ap].y);
        if (distance < bestDistance)
        {
            nearest = ap;
            bestDistance = distance;
        }
    }

    return nearest;
}

// ---------------------------------------------------------------------------
// The topology seed controls physical placement and is separate from the
// ns-3 seed used for fading, contention, and rate-control randomness. Thus,
// five PHY/MAC seeds for one topology really do preserve AP positions,
// background-station positions, loads, and the candidate's location.
//
// Without hotspots, background stations are uniform over the AP footprint
// plus a small border. With hotspots, 70% are divided randomly among 10 m
// discs around the selected APs; that radius is smaller than half the AP
// spacing, so these stations remain clients of the intended AP. The remaining
// stations still cover the deployment and prevent every scenario from being
// a deliberately clustered test.
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

    double border = 0.35 * kApSpacingM;
    std::mt19937 placementRng(config.topologySeed);
    std::uniform_real_distribution<double> uniformX(minX - border, maxX + border);
    std::uniform_real_distribution<double> uniformY(minY - border, maxY + border);
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    uint32_t nHotspotStations = config.hotspotAps.empty()
                                    ? 0
                                    : static_cast<uint32_t>(
                                          std::lround(kHotspotFraction * config.nStas));
    std::uniform_int_distribution<std::size_t> hotspotChoice(
        0,
        config.hotspotAps.empty() ? 0 : config.hotspotAps.size() - 1);

    topology.staPositions.reserve(config.nStas);
    topology.staNearestAp.reserve(config.nStas);
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        double x;
        double y;
        if (sta < nHotspotStations)
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
            x = uniformX(placementRng);
            y = uniformY(placementRng);
        }

        topology.staPositions.emplace_back(x, y, 0.0);
        topology.staNearestAp.push_back(FindNearestAp(x, y, topology.apPositions));
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
    double windowEnd{0.0};
    std::vector<BusyMeter> busyByChannel;
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
    if (!gObservation.frames.is_open() || now >= gObservation.windowEnd)
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

    gObservation.frames << now << ',' << channelFreqMhz << ',' << bssid << ','
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

    double intervalStart = start.GetSeconds();
    double intervalEnd = intervalStart + duration.GetSeconds();
    if (intervalStart >= gObservation.windowEnd || intervalEnd <= 0.0)
    {
        return;
    }

    intervalStart = std::max(0.0, intervalStart);
    intervalEnd = std::min(gObservation.windowEnd, intervalEnd);
    bool busy = state == WifiPhyState::CCA_BUSY || state == WifiPhyState::TX ||
                state == WifiPhyState::RX;
    if (!busy || intervalEnd <= intervalStart)
    {
        return;
    }

    auto channel = std::find(kApChannels.begin(), kApChannels.end(), channelNumber);
    if (channel != kApChannels.end())
    {
        std::size_t index = static_cast<std::size_t>(channel - kApChannels.begin());
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
        static_cast<std::size_t>(std::ceil(gObservation.windowEnd / kBusyBucketS));
    for (std::size_t channelIndex = 0;
         channelIndex < gObservation.busyByChannel.size();
         ++channelIndex)
    {
        uint32_t channel = kApChannels[channelIndex];
        for (std::size_t bucket = 0; bucket < nBuckets; ++bucket)
        {
            double start = bucket * kBusyBucketS;
            double end = std::min(gObservation.windowEnd, (bucket + 1) * kBusyBucketS);
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

void
GenerateTraffic(Ptr<Socket> socket, uint32_t packetSize, Time packetInterval)
{
    socket->Send(Create<Packet>(packetSize));
    Simulator::Schedule(packetInterval,
                        &GenerateTraffic,
                        socket,
                        packetSize,
                        packetInterval);
}

void
ScheduleTraffic(Ptr<Node> sender,
                const InetSocketAddress& destination,
                TypeId socketType,
                Time start,
                Time packetInterval)
{
    Ptr<Socket> source = Socket::CreateSocket(sender, socketType);
    source->Connect(destination);
    Simulator::ScheduleWithContext(sender->GetId(),
                                   start,
                                   &GenerateTraffic,
                                   source,
                                   kPacketSize,
                                   packetInterval);
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

    Time backgroundInterval =
        Seconds(kPacketSize * 8.0 / (config.backgroundMbpsPerSta * 1e6));
    Time candidateInterval =
        Seconds(kPacketSize * 8.0 / (kCandidateOfferedMbps * 1e6));

    // -----------------------------------------------------------------------
    // Radio and propagation model.
    //
    // Every radio shares one YansWifiChannel object. Channel numbers, rather
    // than separate propagation objects, define which APs contend. This lets
    // scanner radios retune across the same physical environment while APs on
    // different channel numbers remain separate interference domains.
    // -----------------------------------------------------------------------
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");

    YansWifiPhyHelper wifiPhy;
    wifiPhy.Set("RxGain", DoubleValue(0.0));

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                                   "Exponent",
                                   DoubleValue(3.0),
                                   "ReferenceDistance",
                                   DoubleValue(1.0),
                                   "ReferenceLoss",
                                   DoubleValue(46.6777));
    wifiChannel.AddPropagationLoss("ns3::NakagamiPropagationLossModel");

    Ptr<YansWifiChannel> sharedChannel = wifiChannel.Create();
    wifiPhy.SetChannel(sharedChannel);

    uint32_t nOccupiedChannels = std::min<uint32_t>(config.nAps, kApChannels.size());
    std::vector<uint8_t> apChannel(config.nAps);
    std::vector<Ssid> apSsid;
    apSsid.reserve(config.nAps);
    for (uint32_t ap = 0; ap < config.nAps; ++ap)
    {
        apChannel[ap] = kApChannels[ap % nOccupiedChannels];
        apSsid.emplace_back("wifi-ap" + std::to_string(ap));
    }

    Topology topology = BuildTopology(config);

    // -----------------------------------------------------------------------
    // Nodes and Wi-Fi devices.
    //
    // Background stations associate with the physically nearest AP. Each AP
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
        uint32_t ap = topology.staNearestAp[sta];
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
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(kApChannels[channelIndex])));
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
            ScheduleTraffic(backgroundNodes.Get(globalSta),
                            InetSocketAddress(apAddress[ap], kBasePort + localSta),
                            udpSocketType,
                            Seconds(kBackgroundStartS),
                            backgroundInterval);
        }
    }

    Address candidateSinkAddress =
        InetSocketAddress(apAddress[config.targetAp], kBasePort + config.nStas);
    std::size_t candidateFlowIndex =
        InstallSink(apNodes.Get(config.targetAp), candidateSinkAddress, udpSocketType);
    ScheduleTraffic(candidateNode.Get(0),
                    InetSocketAddress(apAddress[config.targetAp], kBasePort + config.nStas),
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

        for (uint32_t channelIndex = 0; channelIndex < nOccupiedChannels; ++channelIndex)
        {
            scannerDevices[channelIndex]->GetPhy()->TraceConnectWithoutContext(
                "MonitorSnifferRx",
                MakeCallback(&MonitorSniffRx));
            scannerDevices[channelIndex]->GetPhy()->GetState()->TraceConnectWithoutContext(
                "State",
                MakeBoundCallback(&ScannerPhyState,
                                  static_cast<uint32_t>(kApChannels[channelIndex])));
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
    metadata << "    \"feature_window_end\": "
             << (kCandidateStartS - kFeatureGuardS) << ",\n";
    metadata << "    \"sim_stop_time\": " << kSimulationStopS << ",\n";
    metadata << "    \"background_start_time\": " << kBackgroundStartS << ",\n";
    metadata << "    \"bg_interval_s\": " << backgroundInterval.GetSeconds() << ",\n";
    metadata << "    \"candidate_interval_s\": " << candidateInterval.GetSeconds()
             << ",\n";
    metadata << "    \"bg_per_sta_mbps\": " << config.backgroundMbpsPerSta << ",\n";
    metadata << "    \"bg_total_offered_mbps\": "
             << config.backgroundMbpsPerSta * config.nStas << ",\n";
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
    metadata << "    \"hotspot_fraction\": "
             << (config.hotspotAps.empty() ? 0.0 : kHotspotFraction) << ",\n";
    metadata << "    \"hotspot_radius\": " << kHotspotRadiusM << ",\n";
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
                 << ", \"offered_mbps\": "
                 << apStaGlobalIndex[ap].size() * config.backgroundMbpsPerSta
                 << ", \"candidate_distance\": " << topology.candidateApDistance[ap]
                 << "}" << (ap + 1 < config.nAps ? ",\n" : "\n");
    }

    metadata << "  ],\n";
    metadata << "  \"background_stations\": [\n";
    for (uint32_t sta = 0; sta < config.nStas; ++sta)
    {
        metadata << "    {\"index\": " << sta << ", \"ap\": "
                 << topology.staNearestAp[sta] << ", \"position\": {\"x\": "
                 << topology.staPositions[sta].x << ", \"y\": "
                 << topology.staPositions[sta].y << "}}"
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
