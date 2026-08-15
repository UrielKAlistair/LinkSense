#include "ns3/abort.h"
#include "ns3/boolean.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/log.h"
#include "ns3/mac48-address.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-helper.h"
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiSimpleInfra");
uint64_t g_totalBytes = 0;
static const uint16_t BASE_PORT = 8000;

// Simulated time at which the candidate finished associating, and the AP it
// actually landed on. -1 means "never associated" (too far, or the target
// was unreachable), which is itself a meaningful outcome to record.
double g_candidateAssocTime = -1.0;
Mac48Address g_candidateAssocAp;

// ---------------------------------------------------------------------------
// Pre-association observation recording.
//
// Earlier revisions captured pcap on the scanner radios. That was both
// wasteful and imprecise: pcap runs for the whole simulation (including the
// saturating post-association traffic nothing reads, which reached ~84 MB per
// run and exhausted the disk quota), and radiotap only carries a rounded
// signal value. Tapping MonitorSnifferRx instead yields exact signal and
// noise, the real TX vector, and lets recording stop at the end of the
// feature window - a few hundred KB per run, and no tshark dependency.
// ---------------------------------------------------------------------------
std::ofstream g_obsFile;
double g_obsWindowEnd = 0.0;

/** Mac48Address -> "xx:xx:xx:xx:xx:xx" (its stream operator's format). */
static std::string
MacToString(Mac48Address addr)
{
    std::ostringstream os;
    os << addr;
    return os.str();
}

void
MonitorSniffRx(Ptr<const Packet> packet,
            uint16_t channelFreqMhz,
            WifiTxVector txVector,
            MpduInfo /*aMpdu*/,
            SignalNoiseDbm signalNoise,
            uint16_t /*staId*/)
{
    double now = Simulator::Now().GetSeconds();
    if (!g_obsFile.is_open() || now >= g_obsWindowEnd)
    {
        return;
    }

    Ptr<Packet> copy = packet->Copy();
    WifiMacHeader hdr;
    if (copy->PeekHeader(hdr) == 0)
    {
        return;
    }

    // Which address carries the BSSID depends on the DS bits; control frames
    // (ACK, CTS) carry only a receiver address and no BSSID at all.
    std::string bssid;
    if (hdr.IsMgt())
    {
        bssid = MacToString(hdr.GetAddr3());
    }
    else if (hdr.IsData())
    {
        if (hdr.IsToDs() && !hdr.IsFromDs())
        {
            bssid = MacToString(hdr.GetAddr1());
        }
        else if (!hdr.IsToDs() && hdr.IsFromDs())
        {
            bssid = MacToString(hdr.GetAddr2());
        }
        else
        {
            bssid = MacToString(hdr.GetAddr3());
        }
    }

    std::string ta = hdr.IsCtl() && !hdr.IsRts() ? "" : MacToString(hdr.GetAddr2());

    Time duration = WifiPhy::CalculateTxDuration(packet->GetSize(), txVector, WIFI_PHY_BAND_5GHZ);

    // 0/1/2 = management/control/data, matching the 802.11 frame-type field,
    // rather than ns-3's internal WifiMacType enum which mixes all three
    const int cat = hdr.IsMgt() ? 0 : (hdr.IsCtl() ? 1 : 2);

    g_obsFile << now << ',' << channelFreqMhz << ',' << bssid << ',' << ta << ','
              << cat << ',' << (hdr.IsBeacon() ? 1 : 0) << ','
              << (hdr.IsRetry() ? 1 : 0) << ',' << packet->GetSize() << ','
              << signalNoise.signal << ',' << signalNoise.noise << ','
              << duration.GetMicroSeconds() << ','
              << txVector.GetMode().GetDataRate(txVector) / 1e6 << '\n';
}

/**
 * Fired by StaWifiMac's "Assoc" trace when the candidate completes
 * association. Only the first association is recorded: the label is defined
 * over the window starting at the candidate's initial join.
 */
void
CandidateAssociated(Mac48Address apAddr)
{
    if (g_candidateAssocTime < 0.0)
    {
        g_candidateAssocTime = Simulator::Now().GetSeconds();
        g_candidateAssocAp = apAddr;
    }
}

/**
 * Bytes received since the last PrintStats tick for one tracked socket
 * (a background STA, or the candidate), plus a running total for the whole
 * run (bytesSinceLastTick alone isn't enough to compute the candidate's
 * final throughput label, since it gets reset every tick).
 */
struct StaTraffic
{
    std::string label;
    uint64_t bytesSinceLastTick = 0;
    uint64_t bytesTotal = 0;
};

// One entry per listening socket, populated in main() when each socket is
// created. g_staIndexByAddress maps a socket's bound local address to its
// slot in g_staTraffic, since ReceivePacket only sees that address.
std::vector<StaTraffic> g_staTraffic;
std::map<Address, std::size_t> g_staIndexByAddress;

/**
 * Function called when a packet is received.
 *
 * @param socket The receiving socket.
 */
void
ReceivePacket(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;

    Address localAddress;
    socket->GetSockName(localAddress);

    while ((packet = socket->RecvFrom(from)))
    {
        uint32_t packetSize = packet->GetSize();

        g_totalBytes += packetSize;
        StaTraffic& sta = g_staTraffic[g_staIndexByAddress.at(localAddress)];
        sta.bytesSinceLastTick += packetSize;
        sta.bytesTotal += packetSize;
    }
}

/**
 * Print aggregate and per-station throughput since the last tick.
 */
void
PrintStats()
{
    static uint64_t lastTotalBytes = 0;

    uint64_t totalDeltaBytes = g_totalBytes - lastTotalBytes;
    double totalMbps = (totalDeltaBytes * 8.0) / 1e6;

    std::cout << "t=" << Simulator::Now().GetSeconds()
              << "s " << totalMbps << " Mbps";

    for (auto& sta : g_staTraffic)
    {
        std::cout << " " << sta.label << "=" << (sta.bytesSinceLastTick * 8.0) / 1e6;
        sta.bytesSinceLastTick = 0;
    }

    std::cout << std::endl;

    lastTotalBytes = g_totalBytes;

    Simulator::Schedule(Seconds(1), &PrintStats);
}

/**
 * Generate traffic.
 *
 * @param socket The sending socket.
 * @param pktSize The packet size.
 * @param pktCount The packet count.
 * @param pktInterval The interval between two packets.
 */
static void
GenerateTraffic(Ptr<Socket> socket, uint32_t pktSize, Time pktInterval)
{
    NS_LOG_INFO("Generating one packet of size " << pktSize);
    socket->Send(Create<Packet>(pktSize));
    Simulator::Schedule(pktInterval,
                        &GenerateTraffic,
                        socket,
                        pktSize,
                        pktInterval);
}

static constexpr double kPi = 3.14159265358979323846;

// Non-overlapping 20 MHz channels in the 5 GHz band. YansWifiChannel drops
// any frame whose channel number differs from the receiver's, so APs placed
// on different numbers neither hear nor interfere with each other - which is
// what makes AP choice a load decision rather than purely a signal decision.
// (The 46.68 dB reference loss used above is free-space loss at 1 m for
// ~5.2 GHz, so 5 GHz operation is the self-consistent choice here.)
static const std::vector<uint8_t> kApChannels = {36, 40, 44, 48};

// A channel no AP ever occupies. The candidate's association radio parks
// here until it joins. This is not cosmetic: propagation loss (including
// the Nakagami fading draw) is only evaluated for receivers that share the
// transmitter's channel, so a radio sitting on a live channel consumes RNG
// draws and shifts every subsequent random value. Parking keeps the
// pre-association capture bit-identical across the variants of a matched
// set, so the options in a choice set differ only by the choice itself.
static constexpr uint8_t kParkChannel = 149;

/** Build the ChannelSettings attribute string for a 20 MHz 5 GHz channel. */
static std::string
ChannelSettings(uint8_t number)
{
    return "{" + std::to_string(number) + ", 20, BAND_5GHZ, 0}";
}

int
main(int argc, char* argv[])
{
    uint32_t packetSize{1250}; // bytes => 10 Kb
    uint32_t nSTAs{9};         // total background STAs, split across APs by nearest distance
    uint32_t nAPs{2};
    uint32_t targetAP{0};
    double apSpacing{40.0}; // metres between neighbouring APs
    std::string intervalArg;
    double candidateStartTime{5.0};
    double simStopTime{30.0};
    uint32_t rngSeed{0}; // 0 (default) means "pick a random seed"
    std::string outDir{"runs"};
    std::string runTag; // empty => auto-generate a unique run id
    bool verbose{false};
    // candidate's position is an explicit, independently controllable
    // offset from its target AP (polar coordinates), not a derived
    // function of topology - this is the single most important axis to
    // sweep for a throughput-vs-signal-quality dataset
    double candidateDistance{10.0}; // metres from targetAP
    double candidateAngleDeg{0.0};  // degrees from the +x axis
    // absolute placement, overriding the polar spec above when set. Needed
    // for matched-set runs: to compare "candidate joins AP0" against
    // "candidate joins AP1" fairly, the candidate has to stay in one fixed
    // spot while only targetAP changes - which a targetAP-relative offset
    // can't express, since it moves the candidate along with the target.
    double candidateX{0.0};
    double candidateY{0.0};
    bool candidateAbsolute{false};
    // stddev (metres) of Gaussian jitter applied to background STA
    // placement (x and y); 0 keeps the old fully-deterministic 1D line
    double jitterStd{2.0};
    // Load hotspots. Spreading background STAs evenly gives every AP a
    // similar client count, so the only thing separating two APs is signal -
    // and AP selection collapses back into "pick the strongest". Real
    // deployments are lumpy (a full meeting room next to an empty corridor),
    // and it is exactly that lumpiness that makes the choice interesting.
    int32_t staClusterAp{-1};       // -1 disables clustering
    double staClusterFrac{0.0};     // share of STAs drawn into the hotspot
    double staClusterRadius{10.0};  // metres
    // number of distinct wifi channels to spread APs across, round-robin
    // by AP index; 1 (default) reproduces the old co-channel-everywhere
    // behaviour, >1 gives each AP group its own interference domain
    uint32_t nChannels{1};
    // Offered load per background STA. Total background load is therefore
    // nSTAs * bgPerStaMbps, so adding clients genuinely makes an AP busier -
    // the previous scheme pinned TOTAL offered load to a constant and split
    // it among however many STAs existed, which meant "more clients" changed
    // nothing about congestion, only per-client granularity.
    double bgPerStaMbps{6.0};
    // Offered load for the candidate. Deliberately far above what an 802.11n
    // 20MHz link can carry (~40-50 Mbps of UDP goodput), so the candidate is
    // always backlogged and its measured throughput reports the capacity it
    // could actually win. If this is set at or below achievable capacity the
    // label degenerates into "did it get its offered rate", which measures
    // the traffic generator rather than the network.
    double candidateOfferedMbps{100.0};
    // Features are read from [0, candidateStartTime - featureGuard). The
    // guard exists because the candidate's own join machinery starts
    // perturbing the medium slightly before candidateStartTime (observed:
    // the last ~0.11 s of the window stops matching across the variants of
    // a matched set when APs sit on different channels). Cutting the window
    // short restores exact cross-variant identity, and is also the honest
    // model: a client decides which AP to join a moment before it acts.
    double featureGuard{0.5};
    // Every variant of a matched set observes exactly the same thing, so the
    // orchestrator records it for one variant only.
    bool captureObs{true};

    CommandLine cmd(__FILE__);
    cmd.AddValue("packetSize", "size of application packet sent", packetSize);
    cmd.AddValue("interval", "optional interval between packets (for example, 2ms)", intervalArg);
    cmd.AddValue("verbose", "turn on all WifiNetDevice log components", verbose);
    cmd.AddValue("nSTAs", "total number of background stations, across all APs", nSTAs);
    cmd.AddValue("nAPs", "number of APs", nAPs);
    cmd.AddValue("targetAP", "index of the AP the candidate joins", targetAP);
    cmd.AddValue("apSpacing", "distance in metres between neighbouring APs", apSpacing);
    cmd.AddValue("candidateStartTime", "Time when candidate STA starts sending", candidateStartTime);
    cmd.AddValue("simStopTime", "total simulated seconds to run", simStopTime);
    cmd.AddValue("rngSeed", "RNG seed to use; 0 (default) picks a random seed", rngSeed);
    cmd.AddValue("outDir", "directory to write per-run output (metadata + pcap) under", outDir);
    cmd.AddValue("runTag", "explicit run directory name; empty (default) auto-generates a unique one", runTag);
    cmd.AddValue("candidateDistance", "metres from targetAP the candidate sits", candidateDistance);
    cmd.AddValue("candidateAngleDeg", "angle (degrees, from +x axis) from targetAP to candidate", candidateAngleDeg);
    cmd.AddValue("candidateX", "absolute x of candidate; requires candidateAbsolute=1", candidateX);
    cmd.AddValue("candidateY", "absolute y of candidate; requires candidateAbsolute=1", candidateY);
    cmd.AddValue("candidateAbsolute", "use candidateX/candidateY instead of candidateDistance/Angle", candidateAbsolute);
    cmd.AddValue("jitterStd", "stddev (metres) of Gaussian jitter on background STA placement", jitterStd);
    cmd.AddValue("staClusterAp", "AP index to cluster background STAs around; -1 disables", staClusterAp);
    cmd.AddValue("staClusterFrac", "fraction of background STAs placed in the cluster", staClusterFrac);
    cmd.AddValue("staClusterRadius", "radius (metres) of the STA cluster", staClusterRadius);
    cmd.AddValue("nChannels", "number of distinct wifi channels APs round-robin across", nChannels);
    cmd.AddValue("featureGuard", "seconds before candidateStartTime at which the feature window ends", featureGuard);
    cmd.AddValue("captureObs", "write observation.csv for this run (one variant per matched set is enough)", captureObs);
    cmd.AddValue("bgPerStaMbps", "offered load per background STA in Mbps", bgPerStaMbps);
    cmd.AddValue("candidateOfferedMbps",
                "offered load for the candidate in Mbps; keep well above link capacity so the "
                "measured throughput reflects available capacity rather than the generator",
                candidateOfferedMbps);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(targetAP >= nAPs, "targetAP must be less than nAPs");
    NS_ABORT_MSG_IF(simStopTime - candidateStartTime < 1.0,
                    "candidateStartTime leaves less than 1s to measure throughput before simStopTime");
    NS_ABORT_MSG_IF(nChannels == 0, "nChannels must be at least 1");
    NS_ABORT_MSG_IF(nChannels > kApChannels.size(),
                    "nChannels exceeds the number of non-overlapping channels available");
    NS_ABORT_MSG_IF(!candidateAbsolute && candidateDistance <= 0.0,
                    "candidateDistance must be positive");
    NS_ABORT_MSG_IF(featureGuard < 0.0, "featureGuard must be non-negative");
    NS_ABORT_MSG_IF(candidateStartTime - featureGuard <= 1.0,
                    "feature window must be longer than 1s (raise candidateStartTime or lower featureGuard)");
    NS_ABORT_MSG_IF(bgPerStaMbps <= 0.0, "bgPerStaMbps must be positive");
    NS_ABORT_MSG_IF(candidateOfferedMbps <= 0.0, "candidateOfferedMbps must be positive");

    // if the caller didn't pin a seed, draw a real one so repeated runs
    // aren't silently identical; either way, the seed actually used gets
    // written into this run's metadata below so runs stay reproducible
    if (rngSeed == 0)
    {
        std::random_device rd;
        rngSeed = rd();
        if (rngSeed == 0)
        {
            rngSeed = 1; // 0 is reserved to mean "not provided"
        }
    }
    RngSeedManager::SetSeed(rngSeed);

    // separate PRNG (not ns-3's own event-scheduling RNG stream) used only
    // for the one-time placement jitter computed below, before
    // Simulator::Run() - seeded off the same rngSeed so layouts stay
    // reproducible for a given seed without perturbing ns-3's packet-level
    // randomness (backoff draws, fading, etc.)
    std::mt19937 placementRng(rngSeed);
    std::normal_distribution<double> jitter(0.0, jitterStd);

    // per-packet spacing that realises the requested offered rates; the
    // background and the candidate get their own, since the candidate is
    // deliberately driven far harder (see candidateOfferedMbps above)
    Time interval;
    if (intervalArg.empty())
    {
        interval = Seconds(packetSize * 8.0 / (bgPerStaMbps * 1e6));
    }
    else
    {
        interval = Time(intervalArg);
    }
    Time candidateInterval = Seconds(packetSize * 8.0 / (candidateOfferedMbps * 1e6));

    if (verbose)
    {
        WifiHelper::EnableLogComponents(); // Turn on all Wifi logging
    }

    // Wifi Standard
    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager"); // adaptive rate control
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);

    // Antenna Gain on Node
    wifiPhy.Set("RxGain", DoubleValue(0));

    // Loss and Delay Model
    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                               "Exponent", DoubleValue(3.0),
                               "ReferenceDistance", DoubleValue(1.0),
                               "ReferenceLoss", DoubleValue(46.6777));
    wifiChannel.AddPropagationLoss("ns3::NakagamiPropagationLossModel"); // fast-fading, stacked on the mean loss above

    // A single shared medium: separation between APs comes from channel
    // NUMBERS, which YansWifiChannel already enforces, rather than from
    // separate channel objects. Using one object keeps every radio in the
    // same propagation world, so a scanning radio can be tuned to any
    // channel and hear exactly what a real one would.
    Ptr<YansWifiChannel> sharedChannel = wifiChannel.Create();
    wifiPhy.SetChannel(sharedChannel);

    // which channel each AP operates on, round-robin by index
    std::vector<uint8_t> apChannel(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apChannel[ap] = kApChannels[ap % nChannels];
    }

    // Set up MAC
    WifiMacHelper wifiMac;

    // one SSID per AP, so the candidate (and, below, each background STA)
    // can target a specific one
    std::vector<Ssid> apSsid;
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apSsid.push_back(Ssid("wifi-ap" + std::to_string(ap)));
    }

    // AP positions, decided up front so we can assign each background STA
    // to its nearest AP below, before any devices get installed
    std::vector<Vector> apPosition(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apPosition[ap] = Vector(ap * apSpacing, 0.0, 0.0);
    }

    // spread the STAs evenly across the AP span, then bucket each one under
    // whichever AP is physically closest to it - background load per AP is
    // now a consequence of topology, not a fixed count handed out per AP.
    // Small Gaussian jitter on both axes breaks the perfectly deterministic
    // 1D line (same seed -> same jitter, different seed -> a different but
    // still reproducible layout).
    double staSpan = (nAPs - 1) * apSpacing;
    std::vector<Vector> staPosition(nSTAs);
    std::vector<uint32_t> staNearestAp(nSTAs);
    // STAs assigned to the hotspot, if one was requested; the remainder keep
    // the even spread across the AP span
    uint32_t nClustered = (staClusterAp >= 0 && staClusterAp < static_cast<int32_t>(nAPs))
                            ? static_cast<uint32_t>(std::lround(staClusterFrac * nSTAs))
                            : 0;
    std::uniform_real_distribution<double> unit(0.0, 1.0);

    for (uint32_t i = 0; i < nSTAs; ++i)
    {
        double x;
        double y;
        if (i < nClustered)
        {
            // uniform over the disc around the hotspot AP (sqrt keeps it
            // area-uniform rather than bunched at the centre)
            double r = staClusterRadius * std::sqrt(unit(placementRng));
            double th = 2.0 * kPi * unit(placementRng);
            x = apPosition[staClusterAp].x + r * std::cos(th);
            y = apPosition[staClusterAp].y + r * std::sin(th);
        }
        else
        {
            uint32_t spreadIdx = i - nClustered;
            uint32_t nSpread = nSTAs - nClustered;
            double baseX = (nSpread > 1) ? (spreadIdx * staSpan / (nSpread - 1))
                                         : (staSpan / 2.0);
            x = baseX + jitter(placementRng);
            y = jitter(placementRng);
        }
        staPosition[i] = Vector(x, y, 0.0);

        uint32_t nearest = 0;
        double bestDist = std::hypot(x - apPosition[0].x, y - apPosition[0].y);
        for (uint32_t ap = 1; ap < nAPs; ++ap)
        {
            double dist = std::hypot(x - apPosition[ap].x, y - apPosition[ap].y);
            if (dist < bestDist)
            {
                bestDist = dist;
                nearest = ap;
            }
        }
        staNearestAp[i] = nearest;
    }

    // candidate sits at an explicit polar offset from its target AP -
    // independent of apSpacing/nAPs, so "close to target" and "far from
    // target" are controlled, comparable scenarios rather than a side
    // effect of topology
    double angleRad = candidateAngleDeg * kPi / 180.0;
    Vector candidatePosition =
        candidateAbsolute
            ? Vector(candidateX, candidateY, 0.0)
            : Vector(apPosition[targetAP].x + candidateDistance * std::cos(angleRad),
                    apPosition[targetAP].y + candidateDistance * std::sin(angleRad),
                    0.0);

    // true distance from the candidate to every AP, recorded as ground
    // truth so the Python side can check its RSSI-derived proxies against
    // the real geometry (and so matched-set runs can be reasoned about)
    std::vector<double> candidateApDistance(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        candidateApDistance[ap] = std::hypot(candidatePosition.x - apPosition[ap].x,
                                            candidatePosition.y - apPosition[ap].y);
    }

    NodeContainer APNodes;
    NodeContainer staNodes; // all background STAs, flat - staNearestAp says who they belong to
    NodeContainer newSTA;

    NodeContainer allNodes;

    APNodes.Create(nAPs);
    staNodes.Create(nSTAs);
    newSTA.Create(1);

    allNodes.Add(APNodes);
    allNodes.Add(staNodes);
    allNodes.Add(newSTA);

    // devices grouped by AP (AP + whichever STAs are nearest to it), since
    // each AP gets its own IP subnet below - the candidate's device joins
    // whichever group matches targetAP once it associates
    std::vector<NetDeviceContainer> apGroupDevices(nAPs);
    // global STA index (into staNodes/staPosition/staNearestAp) for each
    // slot within an AP's group, in the order devices were added there -
    // needed later to pair sockets/labels back up with the right node
    std::vector<std::vector<uint32_t>> apStaGlobalIndex(nAPs);

    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(apChannel[ap])));
        wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, APNodes.Get(ap)));
    }

    for (uint32_t i = 0; i < nSTAs; ++i)
    {
        uint32_t ap = staNearestAp[i];
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(apChannel[ap])));
        wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, staNodes.Get(i)));
        apStaGlobalIndex[ap].push_back(i);
    }

    // Scanner radios: one per occupied channel, living on the candidate node,
    // listening promiscuously and never associating (a bogus SSID plus
    // passive scanning, so they transmit nothing and perturb nothing). Their
    // captures ARE the pre-association observation. A real client sweeps one
    // radio across channels sequentially; modelling that as parallel radios
    // gives the full window on every channel instead of a ~100 ms slice, so
    // these features are somewhat cleaner than a real scan would produce.
    std::vector<NetDeviceContainer> scannerDevices(nChannels);
    for (uint32_t c = 0; c < nChannels; ++c)
    {
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(kApChannels[c])));
        wifiMac.SetType("ns3::StaWifiMac",
                        "Ssid", SsidValue(Ssid("scanner-never-associates")),
                        "ActiveProbing", BooleanValue(false));
        scannerDevices[c].Add(wifi.Install(wifiPhy, wifiMac, newSTA.Get(0)));
    }

    // The candidate's association radio. It parks on an empty channel so it
    // hears nothing (and therefore disturbs nothing) until join time, then
    // retunes to the target's channel and adopts its SSID - which is also a
    // fair description of what a real client does once it has picked an AP.
    wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(kParkChannel)));
    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("pending-join")),
                    "ActiveProbing", BooleanValue(false));
    NetDeviceContainer candidateDevice = wifi.Install(wifiPhy, wifiMac, newSTA.Get(0));
    apGroupDevices[targetAP].Add(candidateDevice);

    // Set Positions for everything - order must match allNodes (APs, then
    // STAs in creation-index order, then the candidate)
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        positionAlloc->Add(apPosition[ap]);
    }
    for (uint32_t i = 0; i < nSTAs; ++i)
    {
        positionAlloc->Add(staPosition[i]);
    }
    positionAlloc->Add(candidatePosition);

    mobility.SetPositionAllocator(positionAlloc);
    mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    mobility.Install(allNodes);

    InternetStackHelper internet;
    internet.Install(allNodes);

    // each AP is its own subnet: 10.1.<ap+1>.0/24
    Ipv4AddressHelper ipv4;
    std::vector<Ipv4Address> apAddress(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        ipv4.SetBase(("10.1." + std::to_string(ap + 1) + ".0").c_str(), "255.255.255.0");
        Ipv4InterfaceContainer interfaces = ipv4.Assign(apGroupDevices[ap]);
        apAddress[ap] = interfaces.GetAddress(0); // AP is always first in its group
    }

    // Open Listening Sockets
    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");

    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        for (uint32_t localI = 0; localI < apStaGlobalIndex[ap].size(); ++localI)
        {
            Ptr<Socket> recvSink = Socket::CreateSocket(APNodes.Get(ap), tid);
            Address local = InetSocketAddress(apAddress[ap], BASE_PORT + localI);
            recvSink->Bind(local);
            recvSink->SetRecvCallback(MakeCallback(&ReceivePacket));

            g_staIndexByAddress[local] = g_staTraffic.size();
            g_staTraffic.push_back({"ap" + std::to_string(ap) + "-sta" + std::to_string(localI)});
        }
    }

    Ptr<Socket> candidateSink = Socket::CreateSocket(APNodes.Get(targetAP), tid);
    Address candidateLocal = InetSocketAddress(apAddress[targetAP], BASE_PORT + nSTAs);
    candidateSink->Bind(candidateLocal);
    candidateSink->SetRecvCallback(MakeCallback(&ReceivePacket));
    std::size_t candidateStatsIndex = g_staTraffic.size();
    g_staIndexByAddress[candidateLocal] = candidateStatsIndex;
    g_staTraffic.push_back({"candidate"});

    Simulator::Schedule(Seconds(2), &PrintStats); // Schedule Logger
    // Open Sending Sockets and schedule the requiste sends
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        for (uint32_t localI = 0; localI < apStaGlobalIndex[ap].size(); ++localI)
        {
            uint32_t staIndex = apStaGlobalIndex[ap][localI];
            Ptr<Socket> source = Socket::CreateSocket(staNodes.Get(staIndex), tid);
            InetSocketAddress remote = InetSocketAddress(apAddress[ap], BASE_PORT + localI);
            source->Connect(remote);
            Simulator::ScheduleWithContext(source->GetNode()->GetId(),
                                        Seconds(1),
                                        &GenerateTraffic,
                                        source,
                                        packetSize,
                                        interval);
        }
    }

    // new STA joins the network with some delay:
    Ptr<Socket> source = Socket::CreateSocket(newSTA.Get(0), tid);
    InetSocketAddress remote = InetSocketAddress(apAddress[targetAP], BASE_PORT + nSTAs);
    source->Connect(remote);

    // actually associate the candidate with its target AP at join time, by
    // switching it off the bogus SSID it was installed with. StaWifiMac
    // re-reads GetSsid() every time it restarts scanning (and it keeps
    // rescanning while nothing matches), so swapping the SSID here is what
    // triggers the join - association completes a scan interval later, not
    // instantly, which is why the actual time is traced rather than assumed.
    Ptr<WifiNetDevice> candidateWifiDev = DynamicCast<WifiNetDevice>(candidateDevice.Get(0));
    // retune off the parking channel onto the target's channel, then adopt
    // its SSID; the retune must land first, or the MAC would start scanning
    // for the target while still listening to an empty channel
    auto setChannel =
        static_cast<void (WifiPhy::*)(const WifiPhy::ChannelTuple&)>(&WifiPhy::SetOperatingChannel);
    Simulator::Schedule(Seconds(candidateStartTime),
                        setChannel,
                        candidateWifiDev->GetPhy(),
                        WifiPhy::ChannelTuple{apChannel[targetAP], 20, WIFI_PHY_BAND_5GHZ, 0});
    Simulator::Schedule(Seconds(candidateStartTime), &WifiMac::SetSsid, candidateWifiDev->GetMac(), apSsid[targetAP]);
    candidateWifiDev->GetMac()->TraceConnectWithoutContext("Assoc",
                                                        MakeCallback(&CandidateAssociated));

    Simulator::ScheduleWithContext(source->GetNode()->GetId(),
                                Seconds(candidateStartTime),
                                &GenerateTraffic,
                                source,
                                packetSize,
                                candidateInterval);

    // Every run gets its own output directory. Nanosecond timestamp AND pid
    // AND seed: second-resolution alone collides whenever two runs share a
    // seed within the same second, which is exactly what matched-set runs do
    // on purpose (same seed, one variant per targetAP) - that silently made
    // later variants overwrite earlier ones. runTag lets an orchestrator
    // pin a name instead, so it can group related runs itself.
    auto epochNanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    std::string runId = runTag.empty()
                            ? ("run_" + std::to_string(epochNanos) + "_" +
                            std::to_string(static_cast<long>(getpid())) + "_" +
                            std::to_string(rngSeed))
                            : runTag;
    std::string runDir = outDir + "/" + runId;
    std::filesystem::create_directories(runDir);

    // Record the pre-association observation from the scanner radios. Within
    // a matched set every variant would record exactly the same thing, so
    // the orchestrator only asks for it once per group (--captureObs).
    if (captureObs)
    {
        g_obsWindowEnd = candidateStartTime - featureGuard;
        g_obsFile.open(runDir + "/observation.csv");
        g_obsFile << "t,freq_mhz,bssid,ta,cat,is_beacon,retry,len,signal_dbm,noise_dbm,"
                     "duration_us,rate_mbps\n";
        for (uint32_t c = 0; c < nChannels; ++c)
        {
            Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(scannerDevices[c].Get(0));
            dev->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx",
                                                    MakeCallback(&MonitorSniffRx));
        }
    }

    // Once the observation window closes the scanners have no further job, so
    // park them on an empty channel. This is not only tidiness: leaving them
    // decoding the candidate's saturating post-association traffic costs real
    // simulation time for data nothing consumes.
    for (uint32_t c = 0; c < nChannels; ++c)
    {
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(scannerDevices[c].Get(0));
        Simulator::Schedule(Seconds(candidateStartTime - featureGuard),
                            static_cast<void (WifiPhy::*)(const WifiPhy::ChannelTuple&)>(
                                &WifiPhy::SetOperatingChannel),
                            dev->GetPhy(),
                            WifiPhy::ChannelTuple{kParkChannel, 20, WIFI_PHY_BAND_5GHZ, 0});
    }
    // Output what we are doing
    std::cout << nSTAs << " background STAs across " << nAPs << " APs at "
              << bgPerStaMbps << " Mbps each (" << (bgPerStaMbps * nSTAs)
              << " Mbps offered total), candidate targeting AP " << targetAP << " at "
              << candidateOfferedMbps << " Mbps (saturating), rngSeed=" << rngSeed << std::endl;

    Simulator::Stop(Seconds(simStopTime));
    Simulator::Run();

    // candidate's throughput label: uplink bytes it delivered to its AP,
    // averaged over the window from ACTUAL association to the end of the run.
    // Measuring from candidateStartTime instead would fold the scan/assoc
    // delay into the average and understate throughput by however long the
    // join took. No bytes can arrive before association, so this is exactly
    // the achieved rate while connected.
    bool candidateAssociated = g_candidateAssocTime >= 0.0;
    double observedSeconds =
        candidateAssociated ? (simStopTime - g_candidateAssocTime) : 0.0;
    double candidateMbps =
        (candidateAssociated && observedSeconds > 0.0)
            ? (g_staTraffic[candidateStatsIndex].bytesTotal * 8.0) / 1e6 / observedSeconds
            : 0.0;

    // did it land on the AP we aimed it at? Distinct SSIDs per AP should
    // guarantee this, but recording it means a silent mis-association shows
    // up in the data instead of quietly corrupting the label.
    std::ostringstream assocApStream;
    if (candidateAssociated)
    {
        assocApStream << g_candidateAssocAp;
    }

    // background STAs start sending at t=1s (see the scheduling loop above);
    // used for their mean-throughput figures below, which are ground-truth
    // load numbers for validating (not for feeding directly into) the
    // pre-association features the Python side derives from the pcap alone
    double backgroundObservedSeconds = simStopTime - 1.0;

    // structured run metadata: settings used (including the RNG seed), the
    // AP index/SSID/MAC mapping so the pcap's source addresses can be
    // resolved back to a specific AP, and the resulting throughput label
    std::ofstream meta(runDir + "/metadata.json");
    meta << "{\n";
    meta << "  \"run_id\": \"" << runId << "\",\n";
    meta << "  \"rng_seed\": " << rngSeed << ",\n";
    meta << "  \"params\": {\n";
    meta << "    \"packet_size\": " << packetSize << ",\n";
    meta << "    \"n_stas\": " << nSTAs << ",\n";
    meta << "    \"n_aps\": " << nAPs << ",\n";
    meta << "    \"target_ap\": " << targetAP << ",\n";
    meta << "    \"ap_spacing\": " << apSpacing << ",\n";
    meta << "    \"candidate_start_time\": " << candidateStartTime << ",\n";
    meta << "    \"feature_guard\": " << featureGuard << ",\n";
    meta << "    \"feature_window_end\": " << (candidateStartTime - featureGuard) << ",\n";
    meta << "    \"sim_stop_time\": " << simStopTime << ",\n";
    // seconds, not truncated milliseconds: sub-ms spacings are normal here
    // (a 100 Mbps offered rate at 1250B is 100us) and GetMilliSeconds()
    // silently floors those to 0
    meta << "    \"bg_interval_s\": " << interval.GetSeconds() << ",\n";
    meta << "    \"candidate_interval_s\": " << candidateInterval.GetSeconds() << ",\n";
    meta << "    \"bg_per_sta_mbps\": " << bgPerStaMbps << ",\n";
    meta << "    \"bg_total_offered_mbps\": " << (bgPerStaMbps * nSTAs) << ",\n";
    meta << "    \"candidate_offered_mbps\": " << candidateOfferedMbps << ",\n";
    meta << "    \"candidate_distance\": " << candidateDistance << ",\n";
    meta << "    \"candidate_angle_deg\": " << candidateAngleDeg << ",\n";
    meta << "    \"candidate_absolute\": " << (candidateAbsolute ? "true" : "false") << ",\n";
    meta << "    \"jitter_std\": " << jitterStd << ",\n";
    meta << "    \"sta_cluster_ap\": " << staClusterAp << ",\n";
    meta << "    \"sta_cluster_frac\": " << staClusterFrac << ",\n";
    meta << "    \"sta_cluster_radius\": " << staClusterRadius << ",\n";
    meta << "    \"n_channels\": " << nChannels << "\n";
    meta << "  },\n";
    meta << "  \"candidate_position\": {\"x\": " << candidatePosition.x << ", \"y\": "
        << candidatePosition.y << "},\n";
    meta << "  \"aps\": [\n";
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        // GetAddress() returns a generic ns3::Address, which streams as
        // "<type>-<len>-<bytes>" (eg "05-06-00:00:00:00:00:01"); converting
        // to Mac48Address first gives a plain "xx:xx:xx:xx:xx:xx" string
        // that matches what tshark reports for wlan.bssid, which the
        // Python feature extractor matches against
        std::ostringstream macStream;
        macStream << Mac48Address::ConvertFrom(apGroupDevices[ap].Get(0)->GetAddress());

        uint64_t apBytes = 0;
        for (uint32_t localI = 0; localI < apStaGlobalIndex[ap].size(); ++localI)
        {
            std::size_t idx = g_staIndexByAddress.at(
                InetSocketAddress(apAddress[ap], BASE_PORT + localI));
            apBytes += g_staTraffic[idx].bytesTotal;
        }
        double apBackgroundMbps = (apBytes * 8.0) / 1e6 / backgroundObservedSeconds;

        meta << "    {\"index\": " << ap << ", \"ssid\": \"" << apSsid[ap].PeekString()
             << "\", \"mac\": \"" << macStream.str() << "\", \"channel\": " << +apChannel[ap]
             << ", \"position\": {\"x\": " << apPosition[ap].x << ", \"y\": " << apPosition[ap].y
             << "}, \"sta_count\": " << apStaGlobalIndex[ap].size()
             << ", \"background_mbps\": " << apBackgroundMbps
             << ", \"offered_mbps\": " << (apStaGlobalIndex[ap].size() * bgPerStaMbps)
             << ", \"candidate_distance\": " << candidateApDistance[ap] << "}"
             << (ap + 1 < nAPs ? ",\n" : "\n");
    }
    meta << "  ],\n";
    meta << "  \"candidate\": {\n";
    meta << "    \"target_ap\": " << targetAP << ",\n";
    meta << "    \"associated\": " << (candidateAssociated ? "true" : "false") << ",\n";
    meta << "    \"assoc_time\": " << g_candidateAssocTime << ",\n";
    meta << "    \"assoc_delay\": "
        << (candidateAssociated ? (g_candidateAssocTime - candidateStartTime) : -1.0) << ",\n";
    meta << "    \"assoc_ap_mac\": \"" << assocApStream.str() << "\",\n";
    meta << "    \"observed_seconds\": " << observedSeconds << ",\n";
    meta << "    \"bytes_total\": " << g_staTraffic[candidateStatsIndex].bytesTotal << ",\n";
    meta << "    \"throughput_mbps\": " << candidateMbps << "\n";
    meta << "  }\n";
    meta << "}\n";
    meta.close();

    if (g_obsFile.is_open())
    {
        g_obsFile.close();
    }

    Simulator::Destroy();

    return 0;
}

// Note: we are not currently logging CCA_BUSY state (channel occupancy as seen
// by the candidate's own radio - the real signal a WiFi chip uses for carrier
// sense, distinct from anything recoverable from the pcap). Training on pcap
// alone for now; wire up WifiPhyStateHelper's "State" trace source on
// candidateWifiDev->GetPhy()->GetState() later for a richer feature set.
