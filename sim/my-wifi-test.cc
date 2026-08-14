#include "ns3/abort.h"
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
#include "ns3/wifi-net-device.h"
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

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiSimpleInfra");
uint64_t g_totalBytes = 0;
static const uint16_t BASE_PORT = 8000;
static const uint16_t SAT_CAP = 55; // Mbps

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
    bool verbose{false};
    // candidate's position is an explicit, independently controllable
    // offset from its target AP (polar coordinates), not a derived
    // function of topology - this is the single most important axis to
    // sweep for a throughput-vs-signal-quality dataset
    double candidateDistance{10.0}; // metres from targetAP
    double candidateAngleDeg{0.0};  // degrees from the +x axis
    // stddev (metres) of Gaussian jitter applied to background STA
    // placement (x and y); 0 keeps the old fully-deterministic 1D line
    double jitterStd{2.0};
    // number of distinct wifi channels to spread APs across, round-robin
    // by AP index; 1 (default) reproduces the old co-channel-everywhere
    // behaviour, >1 gives each AP group its own interference domain
    uint32_t nChannels{1};

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
    cmd.AddValue("candidateDistance", "metres from targetAP the candidate sits", candidateDistance);
    cmd.AddValue("candidateAngleDeg", "angle (degrees, from +x axis) from targetAP to candidate", candidateAngleDeg);
    cmd.AddValue("jitterStd", "stddev (metres) of Gaussian jitter on background STA placement", jitterStd);
    cmd.AddValue("nChannels", "number of distinct wifi channels APs round-robin across", nChannels);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(targetAP >= nAPs, "targetAP must be less than nAPs");
    NS_ABORT_MSG_IF(simStopTime - candidateStartTime < 1.0,
                    "candidateStartTime leaves less than 1s to measure throughput before simStopTime");
    NS_ABORT_MSG_IF(nChannels == 0, "nChannels must be at least 1");
    NS_ABORT_MSG_IF(candidateDistance <= 0.0, "candidateDistance must be positive");

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

    Time interval;
    if (intervalArg.empty())
    {
        interval = Seconds(packetSize * 8.0 * nSTAs / (SAT_CAP * 1e6));
    }
    else
    {
        interval = Time(intervalArg);
    }

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

    // one independent YansWifiChannel per wifi channel slot; APs (and their
    // bucketed STAs) round-robin across these by AP index, so with
    // nChannels>1 APs on different slots don't interfere with each other at
    // all, matching real non-overlapping-channel deployment planning
    std::vector<Ptr<YansWifiChannel>> channels(nChannels);
    for (uint32_t c = 0; c < nChannels; ++c)
    {
        channels[c] = wifiChannel.Create();
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
    for (uint32_t i = 0; i < nSTAs; ++i)
    {
        double baseX = (nSTAs > 1) ? (i * staSpan / (nSTAs - 1)) : (staSpan / 2.0);
        double x = baseX + jitter(placementRng);
        double y = jitter(placementRng);
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
    Vector candidatePosition(apPosition[targetAP].x + candidateDistance * std::cos(angleRad),
                            apPosition[targetAP].y + candidateDistance * std::sin(angleRad),
                            0.0);

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
        wifiPhy.SetChannel(channels[ap % nChannels]);
        wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, APNodes.Get(ap)));
    }

    for (uint32_t i = 0; i < nSTAs; ++i)
    {
        uint32_t ap = staNearestAp[i];
        wifiPhy.SetChannel(channels[ap % nChannels]);
        wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, staNodes.Get(i)));
        apStaGlobalIndex[ap].push_back(i);
    }

    // candidate starts on a bogus SSID so it can't associate with any AP yet -
    // its target AP's real SSID gets swapped in below, scheduled at
    // candidateStartTime. It shares targetAP's channel slot, since it has to
    // actually be able to hear/associate with that AP.
    wifiPhy.SetChannel(channels[targetAP % nChannels]);
    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("pending-join")));
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
    // switching it off the bogus SSID it was installed with
    Ptr<WifiNetDevice> candidateWifiDev = DynamicCast<WifiNetDevice>(candidateDevice.Get(0));
    Simulator::Schedule(Seconds(candidateStartTime), &WifiMac::SetSsid, candidateWifiDev->GetMac(), apSsid[targetAP]);

    Simulator::ScheduleWithContext(source->GetNode()->GetId(),
                                Seconds(candidateStartTime),
                                &GenerateTraffic,
                                source,
                                packetSize,
                                interval);

    // every run gets its own output directory, named from when it started
    // and the seed it used, so a batch of runs never collides on disk
    auto epochSeconds = std::chrono::duration_cast<std::chrono::seconds>(
                            std::chrono::system_clock::now().time_since_epoch())
                            .count();
    std::string runId = "run_" + std::to_string(epochSeconds) + "_" + std::to_string(rngSeed);
    std::string runDir = outDir + "/" + runId;
    std::filesystem::create_directories(runDir);

    // Tracing
    wifiPhy.EnablePcap(runDir + "/candidate-trace", candidateDevice.Get(0), true);
    // Output what we are doing
    std::cout << "Saturated Transmission from " << nSTAs << " total STAs across " << nAPs
              << " APs, candidate targeting AP " << targetAP << ", intervals of "
              << interval.As(Time::MS) << ", rngSeed=" << rngSeed << std::endl;

    Simulator::Stop(Seconds(simStopTime));
    Simulator::Run();

    // candidate's throughput label: uplink bytes it delivered to its AP,
    // averaged over the window from association to the end of the run
    double observedSeconds = simStopTime - candidateStartTime;
    double candidateMbps =
        (g_staTraffic[candidateStatsIndex].bytesTotal * 8.0) / 1e6 / observedSeconds;

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
    meta << "    \"sim_stop_time\": " << simStopTime << ",\n";
    meta << "    \"interval_ms\": " << interval.GetMilliSeconds() << ",\n";
    meta << "    \"candidate_distance\": " << candidateDistance << ",\n";
    meta << "    \"candidate_angle_deg\": " << candidateAngleDeg << ",\n";
    meta << "    \"jitter_std\": " << jitterStd << ",\n";
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
             << "\", \"mac\": \"" << macStream.str() << "\", \"channel\": " << (ap % nChannels)
             << ", \"position\": {\"x\": " << apPosition[ap].x << ", \"y\": " << apPosition[ap].y
             << "}, \"sta_count\": " << apStaGlobalIndex[ap].size()
             << ", \"background_mbps\": " << apBackgroundMbps << "}"
             << (ap + 1 < nAPs ? ",\n" : "\n");
    }
    meta << "  ],\n";
    meta << "  \"candidate\": {\n";
    meta << "    \"target_ap\": " << targetAP << ",\n";
    meta << "    \"throughput_mbps\": " << candidateMbps << "\n";
    meta << "  }\n";
    meta << "}\n";
    meta.close();

    Simulator::Destroy();

    return 0;
}

// Note: we are not currently logging CCA_BUSY state (channel occupancy as seen
// by the candidate's own radio - the real signal a WiFi chip uses for carrier
// sense, distinct from anything recoverable from the pcap). Training on pcap
// alone for now; wire up WifiPhyStateHelper's "State" trace source on
// candidateWifiDev->GetPhy()->GetState() later for a richer feature set.
