As of 6 October 2026: Claude has read 1,454,837,704 input tokens (most of which is reloaded context) and written 7,528,828 output tokens for this project, which comes to $1,207 at the global per-million-token API rates. Yikes!

---

# Simulator Design

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

- **Scanner PHYs only on occupied channels:** the candidate carries a scanner PHY
  only for channels that hold an AP, so a dwell on an empty channel records no
  frames and exactly zero CCA. A real radio would register some adjacent-channel
  energy there, which works against the reason `SpectrumWifiPhy` was chosen over
  Yans. Four scanner PHYs on every candidate would record that leakage and make
  every scan uniform, at the cost of a larger cache that is mostly zeros. The
  model gives a silent cell a learned vector in place of its pooled frames. To
  revisit in later simulators.

- **BSS Load:** not simulated and not used; still deferred.

# ML Design

- **Splits:** the topology, not the scan, is the unit kept whole, because the
  scans of one topology hear the same deployment and splitting them would
  reward memorising it. Five folds dealt under a fixed seed, each AP count
  dealt separately so every fold carries the same mix of deployment sizes; a
  run tests on fold f and validates on f+1, giving 60/20/20 with every topology
  tested exactly once over the rotation. The seed lives in `splits.py` rather
  than being passed in, so every model trained from this repo is tested on the
  same folds.

  Initially, Claude decided to draw a 60/20/20 train/val/test split at random five times and pool the results, to get a better estimate of performance by training the
  models more than once. 
  Random draws are bad, I replaced it with K-fold cross-validation, a much more established methodology for the this purpose. Random draws mean that not every topology is used as a test datapoint, it also means test sets in the different runs are
  not disjoint so how much that averaging is worth is questionable.

- **Evaluation:** the reference rules, the tabular predictors and the
  custom-backbone models are scored over all five folds, with hyperparameters and
  checkpoints chosen on each fold's validation part, and reported as mean and
  sample standard deviation. RSSI minus busy is the one rule with a parameter, and
  its weight is fitted on the training scans. The LoRA backbone has only been run
  on fold 0: one fold takes 2.5 h (joint) to 9 h (target) on the local 8 GB GPU,
  so its five folds wait for the cluster. Its numbers compare against fold 0 of
  the custom runs, never against their five-fold means.

- **Identity in the token, and the two attention biases:** every cell token
  carries learned vectors for what kind of cell it is, which dwell it covers,
  which channel it was heard on and which AP it belongs to. They are summed onto
  a normalised cell vector and drawn large enough at initialisation to be heard
  against it. We have the opportunity to add biases to the logits in the encoder
  and decoder layers based on the channel distance and AP identity. There is much 
  precedent for this idea. Question: should we?

  Q1: How should we initialize identity tags? 
  
  The cell vector out of the MLP is not Normalized after being embedded, so its magnitude can grow unconstrianed. The tags were initialized as small guassian noise and added, the tags could never become of comparable magnitude and thus attention could not learn to use these tags at all. (Cell vector ~21 magnitude, Identity tags ~0.7 magnitude)

  `Normalize_MLP`: We thus considered normalise the cell vector's embedding.

  `Loud Init`: We considered initalizing tags at a "loud" value, so that they find optima with strong identity components.

  Q2: Should we add biases in the encoder?

  The rule has precedent (T5, Swin and Graphormer all add a learned per-pair
  scalar to the logits) but precedent is not a reason. Writing a token as content
  plus tag and expanding an attention logit, `q·k = cᵀMc + cᵀMt + tᵀMc + tᵀMt`
  with `M = W_QᵀW_K`, the last term is already a table over pairs of tag values —
  the same object the bias adds. 

  `Decoder Bias`: Maybe?

  `Encoder Bias`: Maybe?

  Now we are left with 4 variables to consider, and they are not independent by any 
  means. The Encoder/Decoder Biases compensate for poor itag dentites, and under 
  their presence, the tags will not need to worry so much about finding optima.

  The presence of these tags could just be a stabilizing force for the model to find optimal use of the relationships, but it could just as likely happen that they prevent 
  tag gradients from taking the tags to their optima by virtue of scalar biases being the quick and easy optimizing mechanism as opposed to the quadratic attention relationship.

  |norm  |loud | dec |  enc |   arm      | regret|
  |------|-----|-----|------|------------|-------|
  |yes   |no   | no  |  no  |   joint_N  |  0.640|
  |yes   |yes  | no  |  no  |   joint_M  |  0.395|
  |------|-----|-----|------|------------|-------|
  |no    |no   | no  |  yes |   joint_C  |  0.517|
  |yes   |no   | no  |  yes |   joint_K  |  0.521|
  |no    |yes  | no  |  yes |   joint_L  |  0.417|
  |yes   |yes  | no  |  yes |   joint_E  |  0.340|
  |yes   |yes  | yes |  yes |   joint_J  |  0.358|
  |no    |no   | yes |  yes |   joint_A  |  0.359|
  |------|-----|-----|------|------------|-------|

  From the table: normalising alone buys nothing, loud tags alone get most of the
  way, and the two together (0.340) beat the original model with its biases
  (0.359). With both, the decoder bias was redundant (0.358 with it, 0.340
  without), while removing the encoder bias as well cost 0.395, so the encoder
  bias stayed. These arms predate the fixed codes of the next entry, which
  replaced the normalisation and loud-init knobs; the current model keeps both biases.

  The identity tag is not how a query row finds its own AP's cells. Removing AP
  identity from the cells and the query rows moved the own-AP attention preference
  by under 0.2 of a logit, against preferences of +0.6 to +13.7: a row finds its
  own cells by content, since they and its descriptor carry the same beacons and
  RSSI. Dropping the tag still cost 0.072 Mbit/s, most likely because it separates
  APs that look alike.

- **Cell tags, fixed orthogonal codes on raw content:** cell type, channel and
  joint-AP identity take fixed, mutually orthogonal unit directions, and the dwell
  a sinusoidal code; each codebook has one trainable length, starting at 0.5. Over
  five folds the joint model scored 0.422 Mbit/s mean regret against 0.442 with
  learned Gaussian tags. The gap changes sign across folds, so this is a design
  choice rather than a measured win. The target model's three roles stay learned
  vectors.

  The content the codes are added onto is not normalised. On fold 0, normalising
  it to unit length gave 0.537 Mbit/s and a LayerNorm gave 0.519, against 0.447
  raw. The AP-identity length finished at 0.0056, 0.031 and 0.068 respectively:
  the more the content was constrained, the further the model turned the identity
  code down.

- **Tags are added, not concatenated:** giving them 16 dimensions of their own and
  leaving 48 for content was worse at every setting: 0.446 against 0.340 without
  the decoder bias, 0.433 against 0.358 with it. A quarter of the cell width costs
  more than a tag subspace free of content buys.

- **Frame content through an MLP, frame time concatenated:** content features and
  category go through Linear, GELU, Linear. A single linear layer makes every
  column a weighted sum of single features, so the mean pool only recovers
  averages the cell aggregates already hold; the MLP lets a column hold a
  conjunction, such as a retry on a discovered AP. Time is concatenated: over five
  folds, adding it gave 0.440 Mbit/s against 0.422, worse on four of five folds,
  and addition had also lost in earlier runs, so only concatenation remains in the
  code. Concatenation lost for the cell tags but wins here, likely because a
  sinusoid has a fixed magnitude and keeps its share of the frame vector without
  having to learn it.

- **Carrier sense keeps its time axis:** the encoder averages its filters down to
  four stretches of the dwell rather than over the whole trace. Where a busy
  stretch fell is the only coarse link between the trace and the frames decoded
  in the same dwell, and busy time with no decoded frame under it points to a
  transmitter the radio could not decode.

- **Two grid models on one base:** `CellGridModel` holds the cell encoder, the
  backbone, the decoder blocks and the head; `JointAPModel` and `TargetAPModel`
  differ only in the identity tag and in how many copies of the grid the backbone
  reads. The joint identity does not depend on which AP is scored, so one grid
  serves every AP; the target role does, so the grid is copied once per discovered
  AP, which is why the target model costs about four times as much. The backbone
  is a component, so the LoRA backbone works with either model from one flag.

- **Batching:** the scans' frame tables are laid end to end and each cell's
  member rows offset by where its scan starts, since frames are reached only
  through cells. Padding per scan left 38% of the table as dummy frames at batch
  size 16. Cells are pooled in eight groups of similar frame count, each padded to
  its own largest, which is 17% of the attention work of padding every cell to
  the batch's largest.

- **LoRA backbone, how it attaches:** it replaces the custom backbone and nothing
  else. A linear layer either side crosses between width 64 and Llama's 2048, at
  264,256 parameters for the pair. Only the frozen weights are bfloat16, so the
  cast happens at the backbone's boundary. The token embedding table is replaced
  by a stub, since nothing is tokenised, which frees 525 MB. A cell may always
  attend to itself, padding included: without that, a padded cell's row can be
  masked in full, and the softmax over nothing is NaN, which reaches the AP rows
  even at zero attention weight because zero times NaN is NaN.

- **LoRA rank 8, not 32:** rank 32 would reproduce NetLLM's 0.31% adapter ratio,
  but that ratio is their choice for a 7B model. Rank 8 gives 851,968 adapter
  parameters and 1.363 million trainable in the joint model, four times the
  custom joint model rather than twelve.

- **Two optimiser defaults, resolved after parsing:** a model trained from scratch
  and a frozen backbone under adapters want different learning rates. Requiring
  `--lr 1e-4` on the command line would leave a forgotten flag training the
  adapters at ten times the intended rate, which would not fail, only be wrong.

- **Checkpoints carry only what trains, and record what they cannot carry:**
  frozen parameters are dropped, which saves about 2 GB a fold for the LLM. With
  the backbone absent, a load cannot tell which backbone trained the adapters, so
  the checkpoint records the backbone name, the random-weight flag, the rank and
  the grid reader, and a load refuses a run whose settings differ.

- **A run resumes from its last epoch:** a target-AP LoRA fold takes about 9.5 h
  on the laptop, longer than it stays on, so `train.py` rewrites
  `progress.pt` after every epoch with the finished folds and the fold in training
  (trainable weights, optimiser, schedule, random streams, best epoch so far). The
  file is written aside and renamed, so a stop mid-write leaves the previous copy.
  A restart with other settings is refused. Tested on CPU: a run killed mid-fold
  and restarted matched an uninterrupted one in every selection metric.

- **Android as a reference rule:** AOSP's `ThroughputScorer` and
  `ThroughputPredictor` are reproduced in `baselines/android_throughput.py`, but
  nothing imports that file and no Android number is reported. The rule needs the
  AP's own channel utilization from the BSS Load element, which the corpus does
  not simulate. Substituting the client's measured busy fraction was tried and
  rejected: it is the medium as the client hears it rather than as the AP hears
  it, and it is already an input to our own models, so it would compare a feature
  rather than compare Android. The file waits on BSS Load.

# Simulator v2: throughput prediction

The task became throughput prediction on 2026-11-13. The entries above describe
the v1 simulator, which was tuned to make AP selection non-trivial; these describe
`sim/my-wifi-test.cc` and `scripts/simulate/run_sweep.py` as they are now.

- **Regimes are sampled, not tuned:** every nuisance parameter is a per-topology
  draw recorded in `metadata.json`. A regressor is judged on coverage of the
  regime space, so no single value has to be defended; the ranges do.
- **Path loss:** TGax residential model at 5 GHz (IEEE 802.11-14/0980r16), with
  5 dB per wall and 5 dB iid shadowing per link. It replaces a log-distance
  exponent of 3 that had no source.
- **Floor:** two rows of 10 m rooms, as in the TGax residential scenario; 2 to 6
  columns. The upper bound is a choice, made to bound run time.
- **AP placement:** distinct rooms at random, uniform inside the room (TGax
  residential). 2 to 8 APs; 8 is the v1 maximum and within the models' 16 identities.
- **Stations:** a room's count is Poisson with a gamma-distributed mean. Mean
  occupancy is log-uniform on 0.3 to 4 per room and the gamma shape log-uniform on
  0.3 to 10; both ranges are choices. Small shapes give crowded rooms, which
  replaces the hotspot discs.
- **Background traffic:** each station has one flow; uplink or downlink and UDP or
  TCP by per-topology probabilities uniform on 0 to 1. Sessions and pauses are
  exponential, their means log-uniform on 0.3 to 30 s per station: from under one
  sweep pass to longer than the run. That range is a choice. Without on/off
  traffic the listening window is close to a sufficient statistic of the future.
- **Offered load:** log-normal per station, sigma 0.9 as before, the median
  log-uniform on 0.3 to 10 Mbit/s per topology (a choice). The cap is 80 Mbit/s,
  above the top PHY rate, so capped stations are full-buffer as in TGax.
- **TCP flows:** the same Poisson offer written to a TCP socket (Cubic,
  1448-byte segments), ending at the AP. A bulk transfer and a wired segment
  behind the AP are not modelled; this is a shortcut.
- **Target AP:** drawn among APs whose mean power at the candidate is at least
  -88 dBm, 6 dB under preamble detection (a choice). One label per scan buys more
  distinct observations per simulated second than one label per AP.
- **Matched and replicated topologies:** one topology in ten is simulated once
  per AP, to score selection as a secondary metric; another one in ten repeats the
  drawn AP under 5 futures, which measures the label's spread given the
  observation. Both fractions are choices.
- **Extra labels:** per-second delivered bytes, transmit time, mean PHY rate,
  data frames sent and acknowledged, association delay. Delay and loss were not
  recorded: for a saturated UDP sender they are functions of throughput.
- **Transmit power:** 16 dBm set explicitly (ns-3's default is 16.02), so the
  association model and the channel use one number.
- **Noise floor with sessions redrawn (`redrawSessions`):** `futureSeed` alone keeps
  every session and pause that was in force at the join, so its replicates share
  how long the traffic heard during the observation goes on. The observation
  cannot tell that length, since session and pause lengths are exponential, so
  the replicates that measure the floor redraw them at the join. The corpus runs
  (`__draw`, `__ap`, `__f`) do not use the flag and are unchanged.
- **Time-scrambled control (`tf/train.py --scramble-time`):** the frame model is
  trained and scored on scans whose frame offsets, pass order and carrier-sense
  sample order are drawn at random, which leaves each dwell's frames and
  aggregates intact. It measures what the model gains from time order. Per-frame
  idle gaps are kept: they are a property of each frame, not of the order.

# Pilot: heavy-tailed sessions

- **Session and pause lengths (`sessionShape`):** Pareto of shape 1.5 at the same
  means as the exponential ones, with every flow started in the long-run state
  of its on/off process. It changes one property of the corpus, whether a
  period's age tells how long it will go on, and nothing else. The shape is a
  choice: 1.5 is what Crovella and Bestavros (1997) report for user think times
  on a wired network in 1995; no measurement of Wi-Fi session persistence at the
  scale of seconds was found.
- **Replicated futures under heavy tails** redraw each period in force at the
  join given its age, so the label spread includes what the observation cannot
  tell about how long the traffic it heard goes on.
- **Pilot size:** the first 300 topologies of the v2 plan, so each pilot scan
  has a v2 twin with the same deployment, client and regime.
- **Per-station check (`scratch_v2/station_features.py`):** airtime is attributed
  by transmitter address; frames without one count as overhead nobody owns. The
  five largest transmitters are kept as columns; five is a choice.
