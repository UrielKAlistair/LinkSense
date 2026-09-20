# Design choices V3

- **Simulator:** `SpectrumWifiPhy` on one `MultiModelSpectrumChannel`, 802.11n
  at 20 MHz, Minstrel-HT, log-distance loss (n = 3) then per-link shadowing then
  Nakagami fading. Yans, which was used in the first simulator was dropped because it zeroes adjacent-channel interference, which heavily affects CCA Busy signals.

- **AP layout:** 2, 3, 4, 6 or 8 APs on a 20 m triangular lattice. Initially,
  other configurations were simulated, but it was changed to a triangular lattice
  following TGax, which points out that this layout keeps the nearest neighbour
  distance constant, so AP count can vary without completely changing the overlap
  and spacing around them.

- **Channels:** each AP draws uniformly from 36/40/44/48, as TGax's residential
  and enterprise scenarios do. Initially, the APs were deliberately put in distinct channels, and then overlapped in AP number order, but this fixed the co channel contention pattern in every topology, both restricting the space of topologies, and potentially teaching the model unrealistic biases.

- **Station placement:** a density field four times denser inside 10 m hotspot
  discs. Initially, a fixed share of stations was placed around the hotspot APs,
  which left many ordinary APs with no stations at all: on the old corpus half
  of all scans contained an empty AP, and in more than half of those it was the
  best choice, so about 29% of scans reduced to picking the empty AP and the
  dataset was biased toward easy decisions. Essentially, the hotspot STA density is a dangerous parameter that can bias the entire dataset, and having it discretely set to a value was a disaster; A probability density smooths out the scenarios, and is much more forgiving and consequently, easier to find a reasonably value for.
  
  The station total was also once scaled up with the number of hotspots, on the
  idea that a hotspot draws more people.(This was a bad call by Claude) This changed whole scenario instead of redistributing stations: hotspot scenarios carried 62% more traffic than uniform ones, and 10.3% of APs were overloaded. 3GPP TR 36.814 and TGax both keep the population fixed and redistribute it instead.
  
- **Stations per AP:** 3, 4 or 5 drawn per AP and summed into the deployment's
  station count; how many each AP actually serves follows from placement and
  association. 
  
  Initially, 2, 3 or 4 were drawn. This left 6.7% of APs with no
  stations, an easy "pick the empty AP" answer, and only 2.0% of APs overloaded
  (over 35 Mbit/s offered, past the point where delivered throughput stops
  rising). A Monte Carlo over 4000 placements showed that using 3–5 STAs cuts idle APs to 3.2% and raises overloaded APs to 4.9%, keeping overload rare without removing it.
  
- **Hotspot count:** none with probability 0.25, otherwise favouring fewer,
  capped at ⌈N_AP/3⌉; the station total is not scaled by it.

- **Per-station load:** log-normal, median 2.25 Mbit/s, σ 0.9, capped at
  25 Mbit/s. The family is cited; the parameters are assumed. 
  
  Initally a fixed value was used for all background stations as an inherited default, but this lead to the count of STAs associated with an AP being a perfect proxy for its recieved bandwidth, allowing a simple linear model to ace the problem. Further, STAs with the same data rate always broadcasted lockstep in the same phase, showing an unrealisitc periodicity. This was fixed with randomised Poisson arrivals.

- **Background traffic:** Poisson with a uniform start phase, replacing
  phase-locked constant-rate sources.

- **Timing:** background traffic from 0.5 s, listening 2–8 s, joining at 8 s,
  stop at 20 s. 
  The first 2 datasets were using a 0.5 s pre-join guard, because of an ns-3 internal bug in YansWifi, where some internal polling would start for the about to associate canidate STA, which changed the observation data based on the target AP. The guard was removed after switching to SpectrumPhy, and Observation was measuered to be bit identical until 8.000 s.

- **Candidate placement:** uniform over the station area.
  Initially, a 70/30 boundary/near-AP mix was used so that the candidate had difficuly choices more often. I decided in hindsight that this was a knob used to tune the dataset difficulty and potentially biasing results, and dropped it in favour of just stratifying the dataset for metrics if the need arose. 

- **Association and shadowing:** strongest received power plus shadowing (which acts as a random perturbation). Initally, I associated STAs to their closest AP, causing strictly voronoi clusters, which was unrealistic. 

- **Candidate traffic:** 100 Mbit/s of 1250-byte packets, so the label is
  capacity under saturation.

- **Seeds:** `topologySeed`, `candidateSeed` and `rngSeed` are independent.
  Three candidate positions and one radio seed per topology, because radio-seed
  replicas were near-duplicates (ICC 0.989, 1.1% of label variance).

- **Matched replay:** a scan's runs share a byte-identical observation, checked
  on sampled scans by `scripts/tests/check_observation_identity.py`. The other
  physics checks in `validate_sim.py` were dropped.

- **Frame times:** each row carries its transmission's start and end, and the
  single-radio sweep keeps a frame only if the whole transmission falls inside a
  dwell. 
  
  Initially, I did not record both start and end, instead opting to rely on end and transmission duration. This caused a bug in an edge case interaction with bursts of packets: If a collection of packets share a preamble and get a group ACK, then, a radio sweep that stops in the middle of the set of frames, or enters into the channel in the middle of this transmission should decode nothing. This behaviour cannot be implemented without both start and end, so I kept track them along with each frame's airtime. In the case of bursty transmissions, the start and end of the whole set is used, so that sweeps that completely contain the set only decipher the frames. 

- **Scan dwell:** 110 ms, what Linux (HZ/9, about 111 ms) and Qualcomm's
  Android configuration (110 ms) ship.

- **Sweep passes:** as many as the 6 s window fits. It might be worth examining in the future, how the models perform with just <1 second of data. Claude decided for some reason that this should be the default and allowed only one single pass over all the channels, which silently broke my V0 results.

- **BSS Load:** not simulated and not used; still deferred.

- **Splits:** the topology, not the scan, is the unit kept whole, because the
  scans of one topology hear the same deployment and splitting them would
  reward memorising it. Five folds dealt under a fixed seed, each AP count
  dealt separately so every fold carries the same mix of deployment sizes; a
  run tests on fold f and validates on f+1, giving 60/20/20 with every topology
  tested exactly once over the rotation. The seed lives in `splits.py` rather
  than being passed in, so every model trained from this repo is tested on the
  same folds.

- **Android as a reference rule:** AOSP's `ThroughputScorer` and
  `ThroughputPredictor` are reproduced in `baselines/android_throughput.py`, but
  nothing imports that file and no Android number is reported. The rule needs the
  AP's own channel utilization from the BSS Load element, which the corpus does
  not simulate. Substituting the client's measured busy fraction was tried and
  rejected: it is the medium as the client hears it rather than as the AP hears
  it, and it is already an input to our own models, so it would compare a feature
  rather than compare Android. The file waits on BSS Load.
