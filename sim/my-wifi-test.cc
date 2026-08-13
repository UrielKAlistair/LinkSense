#include "ns3/abort.h"
#include "ns3/command-line.h"
#include "ns3/config.h"
#include "ns3/double.h"
#include "ns3/internet-stack-helper.h"
#include "ns3/ipv4-address-helper.h"
#include "ns3/log.h"
#include "ns3/mobility-helper.h"
#include "ns3/mobility-model.h"
#include "ns3/ssid.h"
#include "ns3/string.h"
#include "ns3/wifi-mac.h"
#include "ns3/wifi-net-device.h"
#include "ns3/yans-wifi-channel.h"
#include "ns3/yans-wifi-helper.h"
#include <map>
#include <string>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("WifiSimpleInfra");
uint64_t g_totalBytes = 0;
static const uint16_t BASE_PORT = 8000;
static const uint16_t SAT_CAP = 55; // Mb

/**
 * Bytes received since the last PrintStats tick for one tracked socket
 * (a background STA, or the candidate).
 */
struct StaTraffic
{
    std::string label;
    uint64_t bytesSinceLastTick = 0;
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
        g_staTraffic[g_staIndexByAddress.at(localAddress)].bytesSinceLastTick += packetSize;
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

int
main(int argc, char* argv[])
{
    uint32_t packetSize{1250}; // bytes => 10 Kb
    uint32_t nSTAs{9};
    uint32_t nAPs{2};
    uint32_t targetAP{0};
    double apSpacing{40.0}; // metres between neighbouring APs
    std::string intervalArg;
    double newSTAStartTime{5.0};
    bool verbose{false};

    CommandLine cmd(__FILE__);
    cmd.AddValue("packetSize", "size of application packet sent", packetSize);
    cmd.AddValue("interval", "optional interval between packets (for example, 2ms)", intervalArg);
    cmd.AddValue("verbose", "turn on all WifiNetDevice log components", verbose);
    cmd.AddValue("nSTAs", "number of background stations per AP", nSTAs);
    cmd.AddValue("nAPs", "number of APs", nAPs);
    cmd.AddValue("targetAP", "index of the AP the candidate joins", targetAP);
    cmd.AddValue("apSpacing", "distance in metres between neighbouring APs", apSpacing);
    cmd.AddValue("candidateStartTime", "Time when candidate STA starts sending", newSTAStartTime);
    cmd.Parse(argc, argv);

    NS_ABORT_MSG_IF(targetAP >= nAPs, "targetAP must be less than nAPs");

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
    wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager",
                                "DataMode", StringValue("HtMcs7"),
                                "ControlMode", StringValue("HtMcs0"));
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
    wifiPhy.SetChannel(wifiChannel.Create());

    // Set up MAC
    WifiMacHelper wifiMac;

    // one SSID per AP, so the candidate can target a specific one
    std::vector<Ssid> apSsid;
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        apSsid.push_back(Ssid("wifi-ap" + std::to_string(ap)));
    }

    NodeContainer APNodes;
    std::vector<NodeContainer> staGroups(nAPs); // background STAs, per AP
    NodeContainer newSTA;

    NodeContainer allNodes;

    APNodes.Create(nAPs);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        staGroups[ap].Create(nSTAs);
    }
    newSTA.Create(1);

    allNodes.Add(APNodes);
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        allNodes.Add(staGroups[ap]);
    }
    allNodes.Add(newSTA);

    // devices grouped by AP (AP + its own STAs), since each AP gets its own
    // IP subnet below - the candidate's device joins whichever group matches
    // targetAP once it associates
    std::vector<NetDeviceContainer> apGroupDevices(nAPs);

    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        // setup AP
        wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(apSsid[ap]));
        apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, APNodes.Get(ap)));

        // setup this AP's STAs
        wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(apSsid[ap]));
        for (uint32_t i = 0; i < nSTAs; ++i)
        {
            apGroupDevices[ap].Add(wifi.Install(wifiPhy, wifiMac, staGroups[ap].Get(i)));
        }
    }

    // candidate starts on a bogus SSID so it can't associate with any AP yet -
    // its target AP's real SSID gets swapped in below, scheduled at newSTAStartTime
    wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(Ssid("pending-join")));
    NetDeviceContainer candidateDevice = wifi.Install(wifiPhy, wifiMac, newSTA.Get(0));
    apGroupDevices[targetAP].Add(candidateDevice);

    // Set Positions for everything
    MobilityHelper mobility;
    Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();

    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        positionAlloc->Add(Vector(ap * apSpacing, 0.0, 0.0)); // AP ap
    }
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        for (uint32_t i = 0; i < nSTAs; ++i)
        {
            positionAlloc->Add(Vector(ap * apSpacing + 5.0 + i, 0.0, 0.0));
        }
    }
    // candidate sits at the midpoint of the AP span, independent of targetAP,
    // so its physical position stays fixed across runs that vary targetAP
    positionAlloc->Add(Vector((nAPs - 1) * apSpacing / 2.0, 5.0, 0.0));

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
        for (uint32_t i = 0; i < nSTAs; ++i)
        {
            Ptr<Socket> recvSink = Socket::CreateSocket(APNodes.Get(ap), tid);
            Address local = InetSocketAddress(apAddress[ap], BASE_PORT + i);
            recvSink->Bind(local);
            recvSink->SetRecvCallback(MakeCallback(&ReceivePacket));

            g_staIndexByAddress[local] = g_staTraffic.size();
            g_staTraffic.push_back({"ap" + std::to_string(ap) + "-sta" + std::to_string(i)});
        }
    }

    Ptr<Socket> candidateSink = Socket::CreateSocket(APNodes.Get(targetAP), tid);
    Address candidateLocal = InetSocketAddress(apAddress[targetAP], BASE_PORT + nSTAs);
    candidateSink->Bind(candidateLocal);
    candidateSink->SetRecvCallback(MakeCallback(&ReceivePacket));
    g_staIndexByAddress[candidateLocal] = g_staTraffic.size();
    g_staTraffic.push_back({"candidate"});

    Simulator::Schedule(Seconds(2), &PrintStats); // Schedule Logger
    // Open Sending Sockets and schedule the requiste sends
    for (uint32_t ap = 0; ap < nAPs; ++ap)
    {
        for (uint32_t i = 0; i < nSTAs; ++i)
        {
            Ptr<Socket> source = Socket::CreateSocket(staGroups[ap].Get(i), tid);
            InetSocketAddress remote = InetSocketAddress(apAddress[ap], BASE_PORT + i);
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
    Simulator::Schedule(Seconds(newSTAStartTime), &WifiMac::SetSsid, candidateWifiDev->GetMac(), apSsid[targetAP]);

    Simulator::ScheduleWithContext(source->GetNode()->GetId(),
                                Seconds(newSTAStartTime),
                                &GenerateTraffic,
                                source,
                                packetSize,
                                interval);


    // Tracing
    wifiPhy.EnablePcap("traces/new-device-trace", candidateDevice.Get(0), true);
    // Output what we are doing
    std::cout << "Saturated Transmission from " << nSTAs << " STAs per AP across " << nAPs
              << " APs, candidate targeting AP " << targetAP << ", intervals of "
              << interval.As(Time::MS) << std::endl;

    Simulator::Stop(Seconds(30));
    Simulator::Run();
    Simulator::Destroy();

    return 0;
}

// 2. Drop in multiple APs to reach the real problem
// 3. Make a rudimentary ML solution

// Note: we are not currently logging CCA_BUSY state (channel occupancy as seen
// by the candidate's own radio - the real signal a WiFi chip uses for carrier
// sense, distinct from anything recoverable from the pcap). Training on pcap
// alone for now; wire up WifiPhyStateHelper's "State" trace source on
// candidateWifiDev->GetPhy()->GetState() later for a richer feature set.
