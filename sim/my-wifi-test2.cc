// AP selection, second paradigm: a network that changes while you watch it.
//
// sim/my-wifi-test.cc holds the first paradigm - every background station
// offers a constant saturated uplink stream, forever. That was the scenario
// this project was asked to model, and it produced a clean result: with
// stationary load there is nothing in a time series that is not already in
// its mean, a summary statistic is a sufficient statistic, and an MLP over
// per-scan aggregates is the right model. Nothing a recurrent or attention
// model can express is worth expressing.
//
// This file exists to break every assumption that made that true, so that
// the question "does the trajectory matter" has a chance of a real answer:
//
//   * TRAFFIC FLOWS BOTH WAYS. In paradigm one the load was pure uplink, so
//     an AP transmitted almost nothing but beacons - which meant a passive
//     scan either caught a beacon or learned nothing about that BSS, and
//     26% of APs went undetected. Real load is downlink-dominated. Here each
//     station has an uplink and a downlink flow, --dlUlRatio apart.
//
//   * LOAD VARIES, AND VARIES SLOWLY ENOUGH TO EXTRAPOLATE. Each AP carries
//     a demand level that random-walks (AR(1), --demandSigma per --demandStep),
//     and each station switches between ON and OFF sessions with exponential
//     holding times. The fast component is noise a single scan cannot see
//     through; the slow component is a trend that several scans can. That
//     gap is the whole point: the label is throughput AFTER association, so
//     what matters is where the load is GOING, not where it was.
//
//   * EVERYTHING MOVES. Background stations random-walk. The candidate walks
//     a straight line at --candSpeed, so its path loss to each AP trends over
//     the observation window - "I am walking towards AP2" is a fact only a
//     sequence of scans contains, and it changes which AP is the right answer.
//
//   * THE HORIZON IS LONG ENOUGH TO HOLD A SEQUENCE. --obsWindow defaults to
//     60 s, against 4.5 s in paradigm one. Real clients re-scan every 20-180 s
//     (Android, mobility-dependent) or every 30 s (wpa_supplicant bgscan), so
//     a minute of pre-association history is one to a few real scan cycles.
//
//   * THE CLIENT MEASURES LOAD THE WAY A CLIENT ACTUALLY CAN. Paradigm one
//     derived channel occupancy by sniffing every frame on the medium, which
//     no real station does - it decodes beacons and probe responses, not other
//     people's data. Two honest channels replace it, both written here:
//       bssload.csv   what each AP advertises about ITSELF in the BSS Load
//                     element of every beacon - station count and the
//                     fraction of time it sensed the channel busy. This is
//                     the 802.11 QBSS Load element, and it is readable from
//                     a single received beacon, so it survives a 30 ms dwell.
//       chanbusy.csv  what the CLIENT's own radio senses on each channel,
//                     from its PHY carrier-sense state. This is the CCA-busy
//                     counter every chipset exposes.
//     observation.csv therefore only needs to carry management frames, which
//     is also what keeps 60 s of capture down to a few hundred KB.
//
// What is deliberately UNCHANGED, because the choice-set design rests on it:
// matched sets (one scenario, one run per candidate AP, one shared seed),
// absolute candidate placement, per-AP channels on one shared medium,
// parallel scanner radios recording a superset that scripts/build_dataset.py
// carves a realistic single-radio scan out of, and an association radio
// parked on an empty channel so the choice cannot influence the observation.

#include "ns3/abort.h"
#include "ns3/boolean.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/constant-velocity-mobility-model.h"
#include "ns3/double.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/log.h"
#include "ns3/mac48-address.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/rectangle.h"
#include "ns3/rng-seed-manager.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/ap-wifi-mac.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-mac-header.h"
#include "ns3/wifi-net-device.h"
#include "ns3/wifi-phy.h"
#include "ns3/wifi-phy-state-helper.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-helper.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiApSelectionDynamic");

static const uint16_t BASE_PORT = 8000;
static constexpr double kPi = 3.14159265358979323846;

// Non-overlapping 20 MHz channels in the 5 GHz band, all UNII-1 and therefore
// DFS-free - which is why an active scan is legal on every one of them, and
// why scripts/build_dataset.py models one by default.
static const std::vector<uint8_t> kApChannels = {36, 40, 44, 48};

// A channel no AP ever occupies; the candidate's association radio parks here
// until it joins. Not cosmetic: propagation loss (including the Nakagami
// draw) is only evaluated for receivers sharing the transmitter's channel, so
// a radio on a live channel consumes RNG draws and shifts every subsequent
// random value, which would break byte-identity across a matched set.
static constexpr uint8_t kParkChannel = 149;

// 802.11 default beacon interval. Also the natural period for the BSS Load
// element, which is re-advertised in every beacon.
static constexpr double kBeaconIntervalS = 0.1024;

static std::string
MacToString(Mac48Address addr)
{
    std::ostringstream os;
    os << addr;
    return os.str();
}

static std::string
ChannelSettings(uint8_t number)
{
    return "{" + std::to_string(number) + ", 20, BAND_5GHZ, 0}";
}

// ---------------------------------------------------------------------------
// Carrier-sense accounting.
//
// The PHY state trace reports completed state periods as (start, duration,
// state), and a period can straddle any number of reporting buckets. Rather
// than attribute a whole period to whichever bucket its start fell in, clip it
// across buckets - occupancy is the headline observable here and a 102.4 ms
// bucket is short enough for the difference to matter.
// ---------------------------------------------------------------------------
struct BusyMeter
{
    double bucketS{kBeaconIntervalS};
    std::vector<double> busy;  // seconds of non-idle time, per bucket

    void Add(Time start, Time duration, bool isBusy)
    {
        if (!isBusy || duration <= Time(0))
        {
            return;
        }
        double a = start.GetSeconds();
        double b = a + duration.GetSeconds();
        auto first = static_cast<std::size_t>(a / bucketS);
        auto last = static_cast<std::size_t>(b / bucketS);
        if (busy.size() <= last)
        {
            busy.resize(last + 1, 0.0);
        }
        for (std::size_t i = first; i <= last; ++i)
        {
            double lo = std::max(a, i * bucketS);
            double hi = std::min(b, (i + 1) * bucketS);
            if (hi > lo)
            {
                busy[i] += hi - lo;
            }
        }
    }

    /** Busy fraction of the bucket that ends at `now`, clamped to [0,1]. */
    double FractionAt(double now) const
    {
        auto i = static_cast<std::size_t>(now / bucketS);
        if (i == 0 || i - 1 >= busy.size())
        {
            return 0.0;
        }
        return std::min(1.0, busy[i - 1] / bucketS);
    }
};

// One meter per AP radio, one per scanner radio.
std::vector<BusyMeter> g_apBusy;
std::vector<BusyMeter> g_scannerBusy;

static void
OnApPhyState(uint32_t apIndex, Time start, Time duration, WifiPhyState state)
{
    // "Busy" is what the QBSS Load element means by it: time the radio could
    // not have transmitted, whether because it was sending, receiving, or
    // carrier sense said the medium was occupied.
    bool busy = (state == WifiPhyState::CCA_BUSY || state == WifiPhyState::TX ||
                 state == WifiPhyState::RX);
    g_apBusy[apIndex].Add(start, duration, busy);
}

static void
OnScannerPhyState(uint32_t chanIndex, Time start, Time duration, WifiPhyState state)
{
    bool busy = (state == WifiPhyState::CCA_BUSY || state == WifiPhyState::TX ||
                 state == WifiPhyState::RX);
    g_scannerBusy[chanIndex].Add(start, duration, busy);
}

// ---------------------------------------------------------------------------
// Pre-association observation (management frames by default; see --obsFrames).
// ---------------------------------------------------------------------------
std::ofstream g_obsFile;
double g_obsWindowEnd = 0.0;
bool g_obsMgmtOnly = true;

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
    // A real station in scan mode decodes beacons and probe responses; it does
    // not process other people's data frames. Recording only management frames
    // is therefore both more honest and what keeps a 60 s capture small.
    if (g_obsMgmtOnly && !hdr.IsMgt())
    {
        return;
    }

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
    const int cat = hdr.IsMgt() ? 0 : (hdr.IsCtl() ? 1 : 2);

    g_obsFile << now << ',' << channelFreqMhz << ',' << bssid << ',' << ta << ','
              << cat << ',' << (hdr.IsBeacon() ? 1 : 0) << ','
              << (hdr.IsRetry() ? 1 : 0) << ',' << packet->GetSize() << ','
              << signalNoise.signal << ',' << signalNoise.noise << ','
              << duration.GetMicroSeconds() << ','
              << txVector.GetMode().GetDataRate(txVector) / 1e6 << '\n';
}

// ---------------------------------------------------------------------------
// Association tracking.
// ---------------------------------------------------------------------------
double g_candidateAssocTime = -1.0;
Mac48Address g_candidateAssocAp;

void
CandidateAssociated(Mac48Address apAddr)
{
    if (g_candidateAssocTime < 0.0)
    {
        g_candidateAssocTime = Simulator::Now().GetSeconds();
        g_candidateAssocAp = apAddr;
    }
}

// ---------------------------------------------------------------------------
// Traffic: per-flow ON/OFF sessions modulated by a per-AP demand level.
// ---------------------------------------------------------------------------
struct Sink
{
    std::string label;
    uint32_t apIndex{0};
    bool isCandidate{false};
    bool downlink{false};
    uint64_t bytesTotal{0};
    uint64_t bytesAtAssoc{0};  // snapshot when the candidate joined
};

std::vector<Sink> g_sinks;
std::map<Address, std::size_t> g_sinkByAddress;

void
ReceivePacket(Ptr<Socket> socket)
{
    Ptr<Packet> packet;
    Address from;
    Address localAddress;
    socket->GetSockName(localAddress);
    while ((packet = socket->RecvFrom(from)))
    {
        g_sinks[g_sinkByAddress.at(localAddress)].bytesTotal += packet->GetSize();
    }
}

struct Flow
{
    Ptr<Socket> sock;
    uint32_t apIndex{0};
    uint32_t pktSize{1250};
    double baseRateMbps{1.0};  // offered rate while ON, before demand scaling
    bool on{false};
    bool saturating{false};    // the candidate: always on, never demand-scaled
};

std::vector<Flow> g_flows;
std::vector<double> g_apDemand;  // AR(1) multiplier, one per AP

static void
FlowTick(uint32_t id)
{
    Flow& f = g_flows[id];
    double rate = f.saturating ? f.baseRateMbps : f.baseRateMbps * g_apDemand[f.apIndex];
    if ((f.on || f.saturating) && rate > 1e-6)
    {
        f.sock->Send(Create<Packet>(f.pktSize));
        Simulator::Schedule(Seconds(f.pktSize * 8.0 / (rate * 1e6)), &FlowTick, id);
    }
    else
    {
        // Idle flows still need a heartbeat to notice they have been switched
        // back on. 10 ms is coarse enough to cost nothing and fine enough that
        // a session boundary is not visibly quantised.
        Simulator::Schedule(MilliSeconds(10), &FlowTick, id);
    }
}

// Session and demand randomness lives in its own generator, seeded from the
// run seed. Keeping it off ns-3's event RNG means the traffic pattern is
// identical across the variants of a matched set no matter how the radios
// behave - which is what makes those variants the same scenario.
std::mt19937 g_loadRng;
double g_demandSigma = 0.15;
double g_demandMin = 0.25;
double g_demandMax = 1.75;
double g_demandStep = 1.0;
double g_sessionOnMean = 3.0;
double g_sessionOffMean = 4.0;
std::vector<double> g_flowToggleAt;

static void
UpdateLoad()
{
    double now = Simulator::Now().GetSeconds();
    std::normal_distribution<double> step(0.0, g_demandSigma);
    for (auto& d : g_apDemand)
    {
        // A random walk rather than independent draws: consecutive levels are
        // correlated, so a few scans back genuinely predict a few seconds
        // forward. Independent noise would leave the trajectory worthless.
        d = std::clamp(d + step(g_loadRng), g_demandMin, g_demandMax);
    }
    for (std::size_t i = 0; i < g_flows.size(); ++i)
    {
        if (g_flows[i].saturating)
        {
            continue;
        }
        if (now >= g_flowToggleAt[i])
        {
            g_flows[i].on = !g_flows[i].on;
            double mean = g_flows[i].on ? g_sessionOnMean : g_sessionOffMean;
            std::exponential_distribution<double> hold(1.0 / mean);
            g_flowToggleAt[i] = now + hold(g_loadRng);
        }
    }
    Simulator::Schedule(Seconds(g_demandStep), &UpdateLoad);
}

// ---------------------------------------------------------------------------
// Periodic telemetry: what each AP would advertise, and what the client senses.
// ---------------------------------------------------------------------------
std::ofstream g_bssLoadFile;
std::ofstream g_chanBusyFile;
std::vector<Ptr<ApWifiMac>> g_apMac;
std::vector<std::string> g_apMacStr;
std::vector<uint8_t> g_apChannelNum;
double g_telemetryEnd = 0.0;

static void
WriteTelemetry()
{
    double now = Simulator::Now().GetSeconds();
    if (now >= g_telemetryEnd)
    {
        return;
    }
    for (std::size_t ap = 0; ap < g_apMac.size(); ++ap)
    {
        // Exactly the two numbers the 802.11 BSS Load element carries, taken
        // from the AP's own point of view: how many stations are associated,
        // and what fraction of the last beacon interval its carrier sense
        // called busy. A client reads both out of one received beacon.
        std::size_t nSta = g_apMac[ap]->GetStaList(SINGLE_LINK_OP_ID).size();
        g_bssLoadFile << now << ',' << ap << ',' << g_apMacStr[ap] << ','
                      << +g_apChannelNum[ap] << ',' << nSta << ','
                      << g_apBusy[ap].FractionAt(now) << '\n';
    }
    for (std::size_t c = 0; c < g_scannerBusy.size(); ++c)
    {
        g_chanBusyFile << now << ',' << +kApChannels[c] << ','
                       << (5000 + 5 * kApChannels[c]) << ','
                       << g_scannerBusy[c].FractionAt(now) << '\n';
    }
    Simulator::Schedule(Seconds(kBeaconIntervalS), &WriteTelemetry);
}

int
main(int argc, char* argv[])
{
    // ---- topology (unchanged in meaning from paradigm one) ----
    uint32_t packetSize{1250};
    uint32_t nSTAs{9};
    uint32_t nAPs{2};
    uint32_t targetAP{0};
    // TODO (Simulator 2): sample the channel assignment instead of fixing it.
    // Simulator 1 used apChannel[i] = kApChannels[i % nChannels] over a fixed
    // grid, so co-channel separation had exactly one value per deployment size
    // (30 m at eight APs, 42 m at six, no reuse at or below four). Reuse was
    // therefore tied to AP count and could not be varied independently. Drawing
    // the assignment at random, or sampling the reuse distance directly, would
    // decouple them and produce a wider range of contention environments.
    //
    // Settled, for the record: nothing in 802.11 makes an AP vacate a busy
    // channel. Selection is configuration - vendor ACS at boot, or controller
    // RRM - not protocol. 802.11h DFS forces a move only for radar, and only
    // on DFS channels; 36/40/44/48 are UNII-1 and non-DFS, so it never applies
    // here. Co-channel APs contending under CSMA/CA is the normal dense-
    // deployment case, so keeping co-channel pairs is right; only the fixed
    // regularity needs changing.
    //
    // TODO (Simulator 2): lower this. Simulator 1 used 30 m and this defaults
    // to 40 m, which is wide. At exponent 3.0 with no walls the two disagree:
    // 3.0 is an obstructed-indoor value while the geometry is open-field, so
    // either the exponent drops toward 2.0-2.2 or walls get modelled. Measured
    // consequence at 30 m: co-channel APs sit only +2.5 dB (6 AP) to +7.0 dB
    // (8 AP) above the -82 dBm CCA threshold, and the 6-AP case is inside
    // Nakagami's fading swing - so those pairs drift in and out of hidden-node
    // behaviour depending on the radio seed. Prefer sweeping this as a
    // parameter over silently picking another fixed number.
    double apSpacing{40.0};
    uint32_t nChannels{1};
    double candidateX{0.0};
    double candidateY{0.0};
    double jitterStd{2.0};
    int32_t staClusterAp{-1};
    double staClusterFrac{0.0};
    double staClusterRadius{10.0};

    // ---- the long horizon ----
    // 60 s of observation is one to a few real scan cycles (Android re-scans
    // every 20-180 s; wpa_supplicant bgscan every 30 s), and long enough for
    // a demand random walk to show a trend worth extrapolating.
    double obsWindow{60.0};
    double labelWindow{20.0};
    double featureGuard{0.5};

    // ---- time-varying, bidirectional load ----
    double bgPerStaMbps{6.0};
    // Real traffic is downlink-heavy. This is not a detail: with pure uplink
    // an AP transmits nothing but beacons, and whether you ever detect it
    // becomes a coin flip on a single beacon.
    double dlUlRatio{4.0};
    double sessionOnMean{3.0};
    double sessionOffMean{4.0};
    double demandStep{1.0};
    double demandSigma{0.15};
    double demandMin{0.25};
    double demandMax{1.75};

    // ---- mobility ----
    bool staMobility{true};
    double staSpeed{1.0};        // m/s, walking pace
    double candSpeed{0.8};       // candidate drifts, so its RSSI trends
    double candHeadingDeg{0.0};

    // ---- candidate ----
    double candidateOfferedMbps{100.0};
    std::string candidateDirection{"dl"};  // dl | ul | both

    uint32_t rngSeed{0};
    std::string outDir{"runs"};
    std::string runTag;
    std::string obsFrames{"mgmt"};  // mgmt | all
    bool captureObs{true};
    bool verbose{false};
    bool progress{false};

    CommandLine cmd(__FILE__);
    cmd.AddValue("packetSize", "size of application packet sent", packetSize);
    cmd.AddValue("nSTAs", "total number of background stations across all APs", nSTAs);
    cmd.AddValue("nAPs", "number of APs", nAPs);
    cmd.AddValue("targetAP", "index of the AP the candidate joins", targetAP);
    cmd.AddValue("apSpacing", "metres between neighbouring APs", apSpacing);
    cmd.AddValue("nChannels", "distinct channels the APs round-robin across", nChannels);
    cmd.AddValue("candidateX", "absolute x of the candidate at t=0", candidateX);
    cmd.AddValue("candidateY", "absolute y of the candidate at t=0", candidateY);
    cmd.AddValue("jitterStd", "stddev (m) of Gaussian jitter on STA placement", jitterStd);
    cmd.AddValue("staClusterAp", "AP index to cluster STAs around; -1 disables", staClusterAp);
    cmd.AddValue("staClusterFrac", "fraction of STAs placed in the cluster", staClusterFrac);
    cmd.AddValue("staClusterRadius", "radius (m) of the STA cluster", staClusterRadius);
    cmd.AddValue("obsWindow", "seconds of pre-association observation", obsWindow);
    cmd.AddValue("labelWindow", "seconds of post-association throughput measurement", labelWindow);
    cmd.AddValue("featureGuard", "seconds before join at which the observation stops", featureGuard);
    cmd.AddValue("bgPerStaMbps", "peak offered load per background STA (both directions summed)",
                 bgPerStaMbps);
    cmd.AddValue("dlUlRatio", "downlink:uplink ratio of background load", dlUlRatio);
    cmd.AddValue("sessionOnMean", "mean ON session length (s) per background flow", sessionOnMean);
    cmd.AddValue("sessionOffMean", "mean OFF gap (s) per background flow", sessionOffMean);
    cmd.AddValue("demandStep", "seconds between per-AP demand updates", demandStep);
    cmd.AddValue("demandSigma", "per-step stddev of the AP demand random walk", demandSigma);
    cmd.AddValue("demandMin", "lower clamp on the AP demand multiplier", demandMin);
    cmd.AddValue("demandMax", "upper clamp on the AP demand multiplier", demandMax);
    cmd.AddValue("staMobility", "let background STAs random-walk", staMobility);
    cmd.AddValue("staSpeed", "background STA walking speed (m/s)", staSpeed);
    cmd.AddValue("candSpeed", "candidate walking speed (m/s); 0 keeps it still", candSpeed);
    cmd.AddValue("candHeadingDeg", "candidate heading in degrees from the +x axis", candHeadingDeg);
    cmd.AddValue("candidateOfferedMbps", "saturating offered load for the candidate",
                 candidateOfferedMbps);
    cmd.AddValue("candidateDirection", "candidate traffic direction: dl | ul | both",
                 candidateDirection);
    cmd.AddValue("rngSeed", "RNG seed; 0 picks one at random", rngSeed);
    cmd.AddValue("outDir", "directory to write per-run output under", outDir);
    cmd.AddValue("runTag", "explicit run directory name", runTag);
    cmd.AddValue("obsFrames", "which frames observation.csv records: mgmt | all", obsFrames);
    cmd.AddValue("captureObs", "write observation.csv and telemetry for this run", captureObs);
    cmd.AddValue("verbose", "turn on all WifiNetDevice log components", verbose);
    cmd.AddValue("progress", "print a per-second throughput line", progress);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(targetAP >= nAPs, "targetAP must be less than nAPs");
    NS_ABORT_MSG_IF(nChannels == 0 || nChannels > kApChannels.size(),
                    "nChannels must be between 1 and the number of available channels");
    NS_ABORT_MSG_IF(obsWindow - featureGuard <= 1.0,
                    "observation window must exceed 1s (raise obsWindow or lower featureGuard)");
    NS_ABORT_MSG_IF(labelWindow < 1.0, "labelWindow must be at least 1s");
    NS_ABORT_MSG_IF(bgPerStaMbps <= 0.0, "bgPerStaMbps must be positive");
    NS_ABORT_MSG_IF(dlUlRatio < 0.0, "dlUlRatio must be non-negative");
    NS_ABORT_MSG_IF(demandMin <= 0.0 || demandMax < demandMin, "invalid demand clamp range");
    NS_ABORT_MSG_IF(candidateDirection != "dl" && candidateDirection != "ul" &&
                        candidateDirection != "both",
                    "candidateDirection must be dl, ul or both");
    NS_ABORT_MSG_IF(obsFrames != "mgmt" && obsFrames != "all",
                    "obsFrames must be mgmt or all");
    NS_ABORT_MSG_IF(staClusterFrac < 0.0 || staClusterFrac > 1.0,
                    "staClusterFrac must lie in [0,1]");

    if (rngSeed == 0)
    {
        std::random_device rd;
        rngSeed = rd();
        if (rngSeed == 0)
        {
            rngSeed = 1;
        }
    }
    RngSeedManager::SetSeed(rngSeed);

    std::mt19937 placementRng(rngSeed);
    // Offset so placement and load draw from different parts of the sequence;
    // both are still a pure function of rngSeed, so a matched set is exact.
    g_loadRng.seed(rngSeed ^ 0x9e3779b9u);

    g_demandSigma = demandSigma;
    g_demandMin = demandMin;
    g_demandMax = demandMax;
    g_demandStep = demandStep;
    g_sessionOnMean = sessionOnMean;
    g_sessionOffMean = sessionOffMean;
    g_obsMgmtOnly = (obsFrames == "mgmt");

    const double candidateStartTime = obsWindow;
    const double simStopTime = obsWindow + labelWindow;

    if (verbose)
    {
        WifiHelper::EnableLogComponents();
    }

    WifiHelper wifi;
    wifi.SetStandard(WIFI_STANDARD_80211n);
    wifi.SetRemoteStationManager("ns3::MinstrelHtWifiManager");
    YansWifiPhyHelper wifiPhy;
    wifiPhy.SetPcapDataLinkType(WifiPhyHelper::DLT_IEEE802_11_RADIO);
    wifiPhy.Set("RxGain", DoubleValue(0));

    YansWifiChannelHelper wifiChannel;
    wifiChannel.SetPropagationDelay("ns3::ConstantSpeedPropagationDelayModel");
    wifiChannel.AddPropagationLoss("ns3::LogDistancePropagationLossModel",
                                   "Exponent", DoubleValue(3.0),
                                   "ReferenceDistance", DoubleValue(1.0),
                                   "ReferenceLoss", DoubleValue(46.6777));
    wifiChannel.AddPropagationLoss("ns3::NakagamiPropagationLossModel");
    Ptr<YansWifiChannel> sharedChannel = wifiChannel.Create();
    wifiPhy.SetChannel(sharedChannel);

    std::vector<uint8_t> apChannel(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apChannel[ap] = kApChannels[ap % nChannels];
    }

    WifiMacHelper wifiMac;
    std::vector<Ssid> apSsid;
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apSsid.push_back(Ssid("wifi-ap" + std::to_string(ap)));
    }

    std::vector<Vector> apPosition(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apPosition[ap] = Vector(ap * apSpacing, 0.0, 0.0);
    }

    // Background STA placement: even spread with jitter, optionally with a
    // hotspot drawn around one AP (real deployments are lumpy, and that
    // lumpiness is what stops AP choice collapsing into "pick the strongest").
    double staSpan = (nAPs - 1) * apSpacing;
    std::vector<Vector> staPosition(nSTAs);
    std::vector<uint32_t> staNearestAp(nSTAs);
    uint32_t nClustered = (staClusterAp >= 0 && staClusterAp < static_cast<int32_t>(nAPs))
                              ? static_cast<uint32_t>(std::lround(staClusterFrac * nSTAs))
                              : 0;
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> jitter(0.0, jitterStd);

    for (uint32_t i = 0; i < nSTAs; ++i)
    {
        double x;
        double y;
        if (i < nClustered)
        {
            double r = staClusterRadius * std::sqrt(unit(placementRng));
            double th = 2.0 * kPi * unit(placementRng);
            x = apPosition[staClusterAp].x + r * std::cos(th);
            y = apPosition[staClusterAp].y + r * std::sin(th);
        }
        else
        {
            uint32_t spreadIdx = i - nClustered;
            uint32_t nSpread = nSTAs - nClustered;
            double baseX =
                (nSpread > 1) ? (spreadIdx * staSpan / (nSpread - 1)) : (staSpan / 2.0);
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

    Vector candidateStart(candidateX, candidateY, 0.0);
    double candHeadingRad = candHeadingDeg * kPi / 180.0;
    Vector candidateVelocity(candSpeed * std::cos(candHeadingRad),
                             candSpeed * std::sin(candHeadingRad),
                             0.0);
    // Where the candidate will be when it actually associates. With a moving
    // client the distance that decides the label is the one at join time, not
    // the one at t=0, so both are recorded as ground truth.
    Vector candidateAtJoin(candidateStart.x + candidateVelocity.x * candidateStartTime,
                           candidateStart.y + candidateVelocity.y * candidateStartTime,
                           0.0);

    std::vector<double> distanceAtStart(nAPs);
    std::vector<double> distanceAtJoin(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        distanceAtStart[ap] = std::hypot(candidateStart.x - apPosition[ap].x,
                                         candidateStart.y - apPosition[ap].y);
        distanceAtJoin[ap] = std::hypot(candidateAtJoin.x - apPosition[ap].x,
                                        candidateAtJoin.y - apPosition[ap].y);
    }

    NodeContainer APNodes;
    NodeContainer staNodes;
    NodeContainer newSTA;
    APNodes.Create(nAPs);
    staNodes.Create(nSTAs);
    newSTA.Create(1);

    std::vector<NetDeviceContainer> apGroupDevices(nAPs);
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

    // Scanner radios: one per occupied channel, on the candidate node,
    // listening and never associating. They record a SUPERSET of what one
    // sweeping radio could have heard; scripts/build_dataset.py carves the
    // realistic scan out of it, so the scan model can be changed without
    // re-running any of this.
    std::vector<NetDeviceContainer> scannerDevices(nChannels);
    for (uint32_t c = 0; c < nChannels; ++c)
    {
        wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(kApChannels[c])));
        wifiMac.SetType("ns3::StaWifiMac",
                        "Ssid", SsidValue(Ssid("scanner-never-associates")),
                        "ActiveProbing", BooleanValue(false));
        scannerDevices[c].Add(wifi.Install(wifiPhy, wifiMac, newSTA.Get(0)));
    }

    wifiPhy.Set("ChannelSettings", StringValue(ChannelSettings(kParkChannel)));
    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("pending-join")),
                    "ActiveProbing", BooleanValue(false));
    NetDeviceContainer candidateDevice = wifi.Install(wifiPhy, wifiMac, newSTA.Get(0));
    apGroupDevices[targetAP].Add(candidateDevice);

    // ---- mobility: three populations, three models ----
    MobilityHelper apMobility;
    Ptr<ListPositionAllocator> apAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apAlloc->Add(apPosition[ap]);
    }
    apMobility.SetPositionAllocator(apAlloc);
    apMobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    apMobility.Install(APNodes);

    MobilityHelper staMob;
    Ptr<ListPositionAllocator> staAlloc = CreateObject<ListPositionAllocator>();
    for (uint32_t i = 0; i < nSTAs; ++i)
    {
        staAlloc->Add(staPosition[i]);
    }
    staMob.SetPositionAllocator(staAlloc);
    if (staMobility && staSpeed > 0.0)
    {
        // A bounded random walk keeps stations in the deployment while letting
        // each AP's client population - and therefore its load - drift.
        double pad = 30.0;
        staMob.SetMobilityModel(
            "ns3::RandomWalk2dMobilityModel",
            "Bounds", RectangleValue(Rectangle(-pad, staSpan + pad, -pad, pad)),
            "Time", TimeValue(Seconds(5.0)),
            "Speed", StringValue("ns3::ConstantRandomVariable[Constant=" +
                                 std::to_string(staSpeed) + "]"));
    }
    else
    {
        staMob.SetMobilityModel("ns3::ConstantPositionMobilityModel");
    }
    staMob.Install(staNodes);

    // The candidate moves on a straight line at a fixed velocity. Deliberately
    // NOT a random walk: its path is then a pure function of the scenario
    // parameters, identical across the variants of a matched set, and the
    // sweep can control "walking towards AP1" as a scenario axis rather than
    // hoping the RNG produces one.
    MobilityHelper candMob;
    Ptr<ListPositionAllocator> candAlloc = CreateObject<ListPositionAllocator>();
    candAlloc->Add(candidateStart);
    candMob.SetPositionAllocator(candAlloc);
    candMob.SetMobilityModel("ns3::ConstantVelocityMobilityModel");
    candMob.Install(newSTA);
    DynamicCast<ConstantVelocityMobilityModel>(newSTA.Get(0)->GetObject<MobilityModel>())
        ->SetVelocity(candidateVelocity);

    NodeContainer allNodes;
    allNodes.Add(APNodes);
    allNodes.Add(staNodes);
    allNodes.Add(newSTA);

    InternetStackHelper internet;
    internet.Install(allNodes);

    Ipv4AddressHelper ipv4;
    std::vector<Ipv4Address> apAddress(nAPs);
    std::vector<Ipv4InterfaceContainer> apInterfaces(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        ipv4.SetBase(("10.1." + std::to_string(ap + 1) + ".0").c_str(), "255.255.255.0");
        apInterfaces[ap] = ipv4.Assign(apGroupDevices[ap]);
        apAddress[ap] = apInterfaces[ap].GetAddress(0);
    }

    TypeId tid = TypeId::LookupByName("ns3::UdpSocketFactory");
    g_apDemand.assign(nAPs, 1.0);

    // ---- background flows, both directions ----
    // bgPerStaMbps is the peak total per station; the ratio splits it, so
    // raising the ratio moves load onto the AP without changing the total.
    double dlShare = dlUlRatio / (1.0 + dlUlRatio);
    uint16_t port = BASE_PORT;
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        for (uint32_t localI = 0; localI < apStaGlobalIndex[ap].size(); ++localI)
        {
            uint32_t staIndex = apStaGlobalIndex[ap][localI];
            // interface 0 of an AP group is the AP itself; STA localI is at
            // interface localI + 1 in creation order
            Ipv4Address staAddr = apInterfaces[ap].GetAddress(localI + 1);

            // uplink: STA -> AP, sink on the AP
            {
                Ptr<Socket> sink = Socket::CreateSocket(APNodes.Get(ap), tid);
                Address local = InetSocketAddress(apAddress[ap], port);
                sink->Bind(local);
                sink->SetRecvCallback(MakeCallback(&ReceivePacket));
                g_sinkByAddress[local] = g_sinks.size();
                g_sinks.push_back({"ap" + std::to_string(ap) + "-sta" +
                                       std::to_string(localI) + "-ul",
                                   ap, false, false, 0, 0});

                Ptr<Socket> src = Socket::CreateSocket(staNodes.Get(staIndex), tid);
                src->Connect(InetSocketAddress(apAddress[ap], port));
                g_flows.push_back({src, ap, packetSize,
                                   bgPerStaMbps * (1.0 - dlShare), false, false});
                ++port;
            }
            // downlink: AP -> STA, sink on the STA. This is the direction that
            // makes the AP transmit, and therefore the direction that makes it
            // discoverable and its load visible.
            {
                Ptr<Socket> sink = Socket::CreateSocket(staNodes.Get(staIndex), tid);
                Address local = InetSocketAddress(staAddr, port);
                sink->Bind(local);
                sink->SetRecvCallback(MakeCallback(&ReceivePacket));
                g_sinkByAddress[local] = g_sinks.size();
                g_sinks.push_back({"ap" + std::to_string(ap) + "-sta" +
                                       std::to_string(localI) + "-dl",
                                   ap, false, true, 0, 0});

                Ptr<Socket> src = Socket::CreateSocket(APNodes.Get(ap), tid);
                src->Connect(InetSocketAddress(staAddr, port));
                g_flows.push_back({src, ap, packetSize, bgPerStaMbps * dlShare, false, false});
                ++port;
            }
        }
    }

    // ---- candidate flows ----
    // The candidate is saturating and exempt from demand scaling: its measured
    // rate has to report the capacity it could win, not what a generator chose
    // to offer. Downlink by default, because that is what a user experiences
    // and the direction the AP has to contend for.
    Ipv4Address candAddr = apInterfaces[targetAP].GetAddress(
        apInterfaces[targetAP].GetN() - 1);
    std::size_t candidateDlSink = SIZE_MAX;
    std::size_t candidateUlSink = SIZE_MAX;

    if (candidateDirection == "dl" || candidateDirection == "both")
    {
        Ptr<Socket> sink = Socket::CreateSocket(newSTA.Get(0), tid);
        Address local = InetSocketAddress(candAddr, port);
        sink->Bind(local);
        sink->SetRecvCallback(MakeCallback(&ReceivePacket));
        candidateDlSink = g_sinks.size();
        g_sinkByAddress[local] = candidateDlSink;
        g_sinks.push_back({"candidate-dl", targetAP, true, true, 0, 0});

        Ptr<Socket> src = Socket::CreateSocket(APNodes.Get(targetAP), tid);
        src->Connect(InetSocketAddress(candAddr, port));
        g_flows.push_back({src, targetAP, packetSize, candidateOfferedMbps, true, true});
        ++port;
    }
    if (candidateDirection == "ul" || candidateDirection == "both")
    {
        Ptr<Socket> sink = Socket::CreateSocket(APNodes.Get(targetAP), tid);
        Address local = InetSocketAddress(apAddress[targetAP], port);
        sink->Bind(local);
        sink->SetRecvCallback(MakeCallback(&ReceivePacket));
        candidateUlSink = g_sinks.size();
        g_sinkByAddress[local] = candidateUlSink;
        g_sinks.push_back({"candidate-ul", targetAP, true, false, 0, 0});

        Ptr<Socket> src = Socket::CreateSocket(newSTA.Get(0), tid);
        src->Connect(InetSocketAddress(apAddress[targetAP], port));
        g_flows.push_back({src, targetAP, packetSize, candidateOfferedMbps, true, true});
        ++port;
    }

    // Background flows start at t=1s and toggle from there. Candidate flows do
    // not start until join time, so nothing it does can colour the observation.
    g_flowToggleAt.assign(g_flows.size(), 0.0);
    std::exponential_distribution<double> firstOn(1.0 / sessionOnMean);
    for (std::size_t i = 0; i < g_flows.size(); ++i)
    {
        if (g_flows[i].saturating)
        {
            continue;
        }
        g_flows[i].on = true;
        g_flowToggleAt[i] = 1.0 + firstOn(g_loadRng);
        Simulator::ScheduleWithContext(g_flows[i].sock->GetNode()->GetId(), Seconds(1.0),
                                       &FlowTick, static_cast<uint32_t>(i));
    }
    for (std::size_t i = 0; i < g_flows.size(); ++i)
    {
        if (g_flows[i].saturating)
        {
            Simulator::ScheduleWithContext(g_flows[i].sock->GetNode()->GetId(),
                                           Seconds(candidateStartTime), &FlowTick,
                                           static_cast<uint32_t>(i));
        }
    }
    Simulator::Schedule(Seconds(1.0), &UpdateLoad);

    // ---- association ----
    Ptr<WifiNetDevice> candidateWifiDev = DynamicCast<WifiNetDevice>(candidateDevice.Get(0));
    auto setChannel =
        static_cast<void (WifiPhy::*)(const WifiPhy::ChannelTuple&)>(&WifiPhy::SetOperatingChannel);
    Simulator::Schedule(Seconds(candidateStartTime), setChannel, candidateWifiDev->GetPhy(),
                        WifiPhy::ChannelTuple{apChannel[targetAP], 20, WIFI_PHY_BAND_5GHZ, 0});
    Simulator::Schedule(Seconds(candidateStartTime), &WifiMac::SetSsid,
                        candidateWifiDev->GetMac(), apSsid[targetAP]);
    candidateWifiDev->GetMac()->TraceConnectWithoutContext("Assoc",
                                                           MakeCallback(&CandidateAssociated));

    // ---- output ----
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

    // Carrier-sense meters. The AP ones back the BSS Load element; the scanner
    // ones are what the client's own radio would report while dwelling.
    g_apBusy.assign(nAPs, BusyMeter{});
    g_scannerBusy.assign(nChannels, BusyMeter{});
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(apGroupDevices[ap].Get(0));
        dev->GetPhy()->GetState()->TraceConnectWithoutContext(
            "State", MakeBoundCallback(&OnApPhyState, ap));
        g_apMac.push_back(DynamicCast<ApWifiMac>(dev->GetMac()));
        g_apMacStr.push_back(MacToString(Mac48Address::ConvertFrom(dev->GetAddress())));
        g_apChannelNum.push_back(apChannel[ap]);
    }
    for (uint32_t c = 0; c < nChannels; ++c)
    {
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(scannerDevices[c].Get(0));
        dev->GetPhy()->GetState()->TraceConnectWithoutContext(
            "State", MakeBoundCallback(&OnScannerPhyState, c));
    }

    const double obsEnd = candidateStartTime - featureGuard;
    if (captureObs)
    {
        g_obsWindowEnd = obsEnd;
        g_obsFile.open(runDir + "/observation.csv");
        g_obsFile << "t,freq_mhz,bssid,ta,cat,is_beacon,retry,len,signal_dbm,noise_dbm,"
                     "duration_us,rate_mbps\n";
        for (uint32_t c = 0; c < nChannels; ++c)
        {
            Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(scannerDevices[c].Get(0));
            dev->GetPhy()->TraceConnectWithoutContext("MonitorSnifferRx",
                                                      MakeCallback(&MonitorSniffRx));
        }

        g_bssLoadFile.open(runDir + "/bssload.csv");
        g_bssLoadFile << "t,ap_index,bssid,channel,sta_count,channel_util\n";
        g_chanBusyFile.open(runDir + "/chanbusy.csv");
        g_chanBusyFile << "t,channel,freq_mhz,busy_frac\n";
        g_telemetryEnd = obsEnd;
        Simulator::Schedule(Seconds(kBeaconIntervalS), &WriteTelemetry);
    }

    // Once the observation closes the scanners have no further job; parking
    // them stops them decoding the candidate's saturating traffic, which costs
    // real wall-clock time for data nothing reads.
    for (uint32_t c = 0; c < nChannels; ++c)
    {
        Ptr<WifiNetDevice> dev = DynamicCast<WifiNetDevice>(scannerDevices[c].Get(0));
        Simulator::Schedule(Seconds(obsEnd), setChannel, dev->GetPhy(),
                            WifiPhy::ChannelTuple{kParkChannel, 20, WIFI_PHY_BAND_5GHZ, 0});
    }

    if (progress)
    {
        std::cout << nSTAs << " STAs across " << nAPs << " APs, " << bgPerStaMbps
                  << " Mbps peak each at " << dlUlRatio << ":1 DL:UL, candidate -> AP"
                  << targetAP << " (" << candidateDirection << "), obs " << obsWindow
                  << "s + label " << labelWindow << "s, seed=" << rngSeed << std::endl;
    }

    Simulator::Stop(Seconds(simStopTime));
    Simulator::Run();

    // ---- label ----
    bool associated = g_candidateAssocTime >= 0.0;
    double observedSeconds = associated ? (simStopTime - g_candidateAssocTime) : 0.0;
    auto rate = [&](std::size_t idx) {
        if (!associated || observedSeconds <= 0.0 || idx == SIZE_MAX)
        {
            return 0.0;
        }
        return (g_sinks[idx].bytesTotal * 8.0) / 1e6 / observedSeconds;
    };
    double dlMbps = rate(candidateDlSink);
    double ulMbps = rate(candidateUlSink);

    std::ostringstream assocApStream;
    if (associated)
    {
        assocApStream << g_candidateAssocAp;
    }

    double backgroundSeconds = simStopTime - 1.0;

    std::ofstream meta(runDir + "/metadata.json");
    meta << "{\n";
    meta << "  \"run_id\": \"" << runId << "\",\n";
    meta << "  \"paradigm\": 2,\n";
    meta << "  \"rng_seed\": " << rngSeed << ",\n";
    meta << "  \"params\": {\n";
    meta << "    \"packet_size\": " << packetSize << ",\n";
    meta << "    \"n_stas\": " << nSTAs << ",\n";
    meta << "    \"n_aps\": " << nAPs << ",\n";
    meta << "    \"target_ap\": " << targetAP << ",\n";
    meta << "    \"ap_spacing\": " << apSpacing << ",\n";
    meta << "    \"n_channels\": " << nChannels << ",\n";
    meta << "    \"obs_window\": " << obsWindow << ",\n";
    meta << "    \"label_window\": " << labelWindow << ",\n";
    meta << "    \"feature_guard\": " << featureGuard << ",\n";
    meta << "    \"feature_window_end\": " << obsEnd << ",\n";
    meta << "    \"candidate_start_time\": " << candidateStartTime << ",\n";
    meta << "    \"sim_stop_time\": " << simStopTime << ",\n";
    meta << "    \"bg_per_sta_mbps\": " << bgPerStaMbps << ",\n";
    meta << "    \"bg_peak_total_mbps\": " << (bgPerStaMbps * nSTAs) << ",\n";
    meta << "    \"dl_ul_ratio\": " << dlUlRatio << ",\n";
    meta << "    \"session_on_mean\": " << sessionOnMean << ",\n";
    meta << "    \"session_off_mean\": " << sessionOffMean << ",\n";
    meta << "    \"demand_step\": " << demandStep << ",\n";
    meta << "    \"demand_sigma\": " << demandSigma << ",\n";
    meta << "    \"demand_min\": " << demandMin << ",\n";
    meta << "    \"demand_max\": " << demandMax << ",\n";
    meta << "    \"sta_mobility\": " << (staMobility ? "true" : "false") << ",\n";
    meta << "    \"sta_speed\": " << staSpeed << ",\n";
    meta << "    \"cand_speed\": " << candSpeed << ",\n";
    meta << "    \"cand_heading_deg\": " << candHeadingDeg << ",\n";
    meta << "    \"candidate_offered_mbps\": " << candidateOfferedMbps << ",\n";
    meta << "    \"candidate_direction\": \"" << candidateDirection << "\",\n";
    meta << "    \"obs_frames\": \"" << obsFrames << "\",\n";
    meta << "    \"jitter_std\": " << jitterStd << ",\n";
    meta << "    \"sta_cluster_ap\": " << staClusterAp << ",\n";
    meta << "    \"sta_cluster_frac\": " << staClusterFrac << ",\n";
    meta << "    \"sta_cluster_radius\": " << staClusterRadius << "\n";
    meta << "  },\n";
    meta << "  \"candidate_position\": {\"x\": " << candidateStart.x << ", \"y\": "
         << candidateStart.y << "},\n";
    meta << "  \"candidate_position_at_join\": {\"x\": " << candidateAtJoin.x << ", \"y\": "
         << candidateAtJoin.y << "},\n";
    meta << "  \"candidate_velocity\": {\"x\": " << candidateVelocity.x << ", \"y\": "
         << candidateVelocity.y << "},\n";
    meta << "  \"aps\": [\n";
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        uint64_t apBytes = 0;
        for (const auto& s : g_sinks)
        {
            if (s.apIndex == ap && !s.isCandidate)
            {
                apBytes += s.bytesTotal;
            }
        }
        meta << "    {\"index\": " << ap << ", \"ssid\": \"" << apSsid[ap].PeekString()
             << "\", \"mac\": \"" << g_apMacStr[ap] << "\", \"channel\": " << +apChannel[ap]
             << ", \"position\": {\"x\": " << apPosition[ap].x << ", \"y\": " << apPosition[ap].y
             << "}, \"sta_count\": " << apStaGlobalIndex[ap].size()
             << ", \"background_mbps\": " << ((apBytes * 8.0) / 1e6 / backgroundSeconds)
             << ", \"peak_offered_mbps\": " << (apStaGlobalIndex[ap].size() * bgPerStaMbps)
             << ", \"candidate_distance\": " << distanceAtStart[ap]
             << ", \"candidate_distance_at_join\": " << distanceAtJoin[ap] << "}"
             << (ap + 1 < nAPs ? ",\n" : "\n");
    }
    meta << "  ],\n";
    meta << "  \"candidate\": {\n";
    meta << "    \"target_ap\": " << targetAP << ",\n";
    meta << "    \"associated\": " << (associated ? "true" : "false") << ",\n";
    meta << "    \"assoc_time\": " << g_candidateAssocTime << ",\n";
    meta << "    \"assoc_delay\": "
         << (associated ? (g_candidateAssocTime - candidateStartTime) : -1.0) << ",\n";
    meta << "    \"assoc_ap_mac\": \"" << assocApStream.str() << "\",\n";
    meta << "    \"observed_seconds\": " << observedSeconds << ",\n";
    meta << "    \"throughput_dl_mbps\": " << dlMbps << ",\n";
    meta << "    \"throughput_ul_mbps\": " << ulMbps << ",\n";
    meta << "    \"throughput_mbps\": " << (dlMbps + ulMbps) << "\n";
    meta << "  }\n";
    meta << "}\n";
    meta.close();

    if (g_obsFile.is_open())
    {
        g_obsFile.close();
    }
    if (g_bssLoadFile.is_open())
    {
        g_bssLoadFile.close();
    }
    if (g_chanBusyFile.is_open())
    {
        g_chanBusyFile.close();
    }

    Simulator::Destroy();
    return 0;
}
