# wifi-ap-selection

A Wi-Fi client in range of several access points commonly joins the one with
the strongest signal. The throughput it then gets depends on channel
contention, competing stations, transmission rates, retries and the airtime of
neighbouring networks, which signal strength reflects poorly. This project asks
whether a client can choose the AP that will give it the most throughput using
only what its radio passively hears before it associates.

The data come from ns-3 simulations of multi-AP deployments. Each scenario is
simulated once per AP the client could join, changing only that choice, so the
runs share one pre-association observation and each yields the throughput of
joining one AP.

The problem, the simulator and the results are described in
[the report](report/simulator_v1_ieee.pdf).
