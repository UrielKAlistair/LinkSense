# Throughput prediction: redesign status and handoff

Written 2026-11-13 07:05 IST by the session that started the redesign, for whoever
continues it (a delegate agent, then the user at 14:00 IST). Branch
`throughput-prediction`, pushed to `origin`. Commit and push at every milestone,
with messages in the user's format (imperative subject, bulleted body of
"Added/Updated/Removed ...", no history).

## 1. Where things stand

Done and pushed:

- `sim/my-wifi-test.cc` is simulator v2 (commit `bf105d0`). It builds and six pilot
  runs gave sane output (8-48 s wall time each; labels 0, 0.2, 2.7, 22, 27, 31 Mbit/s).
- The v1 binary is kept at `~/workspace/ns-3.45/build/scratch/ns3.45-my-wifi-test-v1`.
  The v1 corpus in `data/runs`, `data/cache` is untouched. Do not delete either.

- Observation identity holds on v2: for one topology, `observation.csv` and
  `chanbusy.csv` were byte-identical across three target APs and across two
  `futureSeed` values, and a repeated run reproduced its label exactly (manual check;
  the scripted test is not yet adapted).
- `scripts/simulate/run_sweep.py` is rewritten for v2 (section 4.1 describes it).
  Each scan runs `__draw` first, then its matched (`__apK`) or replicated (`__fN`) runs.
- A 30-topology pilot was launched at 07:03 IST into `data/runs_v2_pilot`, log in
  `logs/sweep_v2_pilot.log`. The heavy-compute check passed before launch, but with
  14 workers the CPU (`Tctl`) read 95 C at 07:06, above the user's 90 C alarm line.
  The user's rule is that an agent reports this and never pauses or stops the job
  itself. **Do not launch the full sweep at 14 workers.** Start it at 8, read
  `sensors` after five minutes, and only raise the count if `Tctl` stays under 90 C.
- Run time is long-tailed: light runs take 10-20 s, heavy ones over 2 minutes. The
  sweep prints progress in scan order, so one slow scan holds the counter back.

Broken by the rewrite, to fix: `scripts/tests/check_observation_identity.py` imports
`sample_candidate`, and `scripts/tests/test_models.py` imports `Run` and
`already_done`, none of which exist any more.

Pilot result (30 topologies, 150 scans, 245 runs, 6.0 min at 14 workers, 213 MB):

- 6% of drawn-target runs never associated, 10% have a label of exactly 0.
- Label quartiles 15.6 / 36.1 / 52.6 Mbit/s, mean 33.9, sd 20.7. 52 of 150 labels lie
  in 50-60 Mbit/s: the near-empty-channel case. v1 had median 5.3. The regime draw
  puts a third of the mass where prediction is easy; whether to reweight it (for
  example a higher lower bound on occupancy) is a choice for the user, not made here.
- Median 5 stations per topology, 90th percentile 25, maximum 40.
- Noise floor, from 15 replicated scans of 5 futures each: mean absolute deviation
  from the scan median 0.76 Mbit/s, mean within-scan sd 1.1. Small sample; the full
  sweep gives 500 such scans.

**The full sweep is running** since 07:10 IST: 1,000 topologies, 8 workers, into
`data/runs_v2`, log `logs/sweep_v2.log`, started by the first session as a background
job. Expect about 8,200 runs at roughly 1,400 an hour, so it ends near 13:00. It
resumes if restarted with the same command (in `run_sweep.py`'s docstring, plus
`LD_LIBRARY_PATH`, `--workers 8`). Topologies finish roughly in order, so a cache of
what exists at any time is a valid smaller corpus. Check `sensors` now and then; the
rule on temperature above applies.

Status at 07:16 IST:

- **Temperature, for the user to decide:** with only the 8-worker sweep running,
  `Tctl` still reads 95.1 C, steady, CPU fan at 7,700 RPM. That is over the user's
  90 C line. The first session did not pause or stop the sweep, because the user's
  rule forbids an agent doing so; the options are `kill -STOP 18098` /
  `kill -CONT 18098` to pause and resume `run_sweep.py` (its children finish their
  current run), or stopping it. Nobody but the user takes either.
- `scripts/tests/check_observation_identity.py` is adapted and passes on 4 scans
  (all target APs and a second `futureSeed`). `scripts/tests/test_models.py` still
  imports `Run` and `already_done` and fails at import.
- `baselines/train.py` runs on the pilot datatable. On 145 pilot scans (about 100
  for training, AP-count stratum flattened by hand in a scratch copy): random forest
  MAE 6.3 Mbit/s, gradient boosting 6.4, ridge 8.4. These are smoke-test numbers on
  a corpus far too small to report; all they suggest is that v2 has headroom above
  its noise floor of roughly 0.8.

Not done, in priority order (section 4 gives the detail):

1. While the sweep runs: cache what exists (`scripts/tf/cache_dataset.py data/runs_v2
   --out data/cache_v2`), and get `baselines/train.py` and `tf/train.py` running on it.
2. `parse_scans.py`, `cache_dataset.py`, `build_datatable.py`, `tf/train.py`,
   `common/evaluate.py` and `baselines/train.py` are edited for v2 (unlabelled APs are
   NaN and masked out of the loss and the scores; checkpoints and grids are selected
   on validation MAE; regime parameters travel in `_index.npz` and as `gt_*` columns).
   Smoke-tested end to end on the first 65 pilot scans: cache (63 cached, 2 scans did
   not discover the AP they joined), datatable (71 labelled rows) and 8 epochs of the
   joint model all ran. `baselines/train.py` is not yet run on v2. The splitter needs
   at least 5 topologies per AP count, so it only works on a full sweep. Known loose ends:
   the ranking objective (`--objective ranking`) has no meaning under MAE selection;
   selection metrics (regret, top-1) are only meaningful on matched topologies
   (index % 10 == 0) and should be reported on those alone; `scripts/tests/test_cells.py`
   still builds v1 metadata; the splits stratify on AP count only.
3. Train the bespoke joint model and the tabular baselines; measure the noise floor.
4. (done) Choices are logged in `report/design_choices.md`, section "Simulator v2".
   (they are only here and in `run_sweep.py` so far).
5. Write the direction report for the user (section 5 is its skeleton).

Build and run:

```
cd ~/workspace/ns-3.45 && cmake --build cmake-cache-release --target scratch_my-wifi-test -j 14
export LD_LIBRARY_PATH=~/workspace/ns-3.45/build/lib64
~/workspace/ns-3.45/build/scratch/ns3.45-my-wifi-test --nCols=3 --nAPs=3 --meanOccupancy=2 \
    --topologySeed=1 --candidateSeed=1 --outDir=/tmp/x --runTag=p1
```

The source is symlinked into `scratch/`. Only the scratch target should need
rebuilding now (the first release build of the day rebuilt all of ns-3).

## 2. What the user decided (do not reopen)

- The task is throughput prediction. AP selection is a secondary metric, scored on a
  small matched subset of the test set only.
- The candidate stays saturated (uplink UDP, queue never empty). Extra labels are
  recorded per run. Unsaturated demand is deferred.
- Radio stays 802.11n, 5 GHz, 20 MHz, Minstrel-HT.
- Variety to add: TCP traffic, walls/rooms, irregular AP layout, and background
  traffic variety (downlink, on/off sessions, some saturated stations). Candidate
  mobility is out of scope.
- The v1 dataset restrictions that existed to keep the *choice* problem hard are lifted.
- Design choices must be principled, coherent and logged; unexamined defaults were
  the failure of the last iteration.
- Not wanted: retraining every model (the LLM especially) on the v1 corpus.

## 3. Simulator v2 as implemented, and why

One principle replaces v1's tuning: **sample the regimes, do not tune them.** v1
picked single values (load median, hotspot density, station count range) so that
the best AP was neither obvious nor random. A regressor has no such requirement.
What it needs is coverage, so every nuisance parameter is now a per-topology draw
over a wide range, recorded in `metadata.json`, and results are reported per regime
and on held-out regimes.

Paragraph by paragraph against `report/simulator_v1_ieee.tex`, section "Simulator
Design and Assumptions":

| v1 paragraph | v1 choice | Critique under prediction | v2 |
|---|---|---|---|
| Environment | Open floor, log-distance exponent 3 ("assumed"), 5 dB shadowing | The exponent had no source. No walls, so received power is a near-monotone function of distance and there are few hidden nodes | TGax residential loss at 5 GHz (IEEE 802.11-14/0980r16): 40.05 + 20 log10(fc/2.4) + 20 log10(min(d,5)) + 35 log10(d/5) for d > 5, + 5 dB per wall crossed, walls counted along x and y; 5 dB iid shadowing per link. Floor is 2 rows of 10 m rooms (TGax uses 2 x 10 apartments) |
| AP count, topology | 2-8 APs on a two-row triangular lattice, 20 m spacing | The lattice was a planned-enterprise assumption; it fixes inter-AP distance, so co-channel coupling barely varies | APs in distinct rooms chosen at random, each uniform inside its room (TGax residential placement). `nAPs` and `nCols` are drawn per topology |
| Channels | uniform over 36/40/44/48 | Has precedent (TGax: random assignment). Keep | unchanged |
| Background STA count and hotspots | 3-5 stations per AP, plus hotspot discs of density 4 around chosen APs | Station count was tied to AP count, and hotspots were a device to make the strongest AP a bad choice. Both are choice-problem tuning | Stations belong to rooms, not APs: a room's count is Poisson with a gamma-distributed mean (a negative binomial). `meanOccupancy` sets the level, `occupancyShape` the unevenness; small shape reproduces crowded rooms without a special case. APs with no stations now occur |
| Association | strongest received power | Standard (TGax enterprise). Keep | unchanged, computed from the same loss matrix the channel uses |
| Background load | uplink UDP Poisson, lognormal median 2.25 Mbit/s, sigma 0.9, cap 25 | One direction, one transport, stationary: the observation window is then a sufficient statistic of the future, which makes prediction artificially easy. The median was tuned | Per flow: uplink or downlink (`downlinkFraction`), UDP or TCP (`tcpFraction`), on/off with exponential session and pause lengths whose means are log-uniform on 0.3-30 s per station. Load median is a per-topology draw. Cap raised to 80 Mbit/s, above the top PHY rate, so capped stations are full-buffer (the TGax traffic model) |
| Candidate placement | uniform in the lattice's bounding box with margin | fine | uniform over the floor |
| Candidate traffic and label | 100 Mbit/s uplink UDP, mean delivered rate from association to 20 s | Keep (user decision). But one scalar hides why a prediction fails | Same label, plus `bytes_per_second` (12 values), `transmit_seconds`, `mean_phy_rate_mbps`, `data_frames_sent`, `data_frames_acked`, `assoc_delay` |
| Matched replays | every candidate run once per AP | Needed only to score selection. It spends 2-8 runs per observation; for regression, more distinct observations are worth more than more labels per observation (to be confirmed, see 4.4) | `targetAP=-1` draws one target among APs whose mean power at the candidate is at least -88 dBm. Matched replays only for a test subset |
| Seeds | topology, candidate, rng | The label's irreducible spread given the observation could not be measured directly (v1 estimate from an ICC: MAE floor about 1.15 Mbit/s) | `futureSeed` reseeds background traffic from 8 s on. Runs that differ only in it share the observation exactly, so their label spread is the noise floor, measured |

Flag for the user: the option they picked said "delay, loss and airtime share" as
extra labels. For a saturated UDP sender, queueing delay and loss are functions of
throughput (the queue is always full), so they carry nothing new. v2 records instead
the pieces throughput is made of: throughput is roughly airtime share x PHY rate x
delivery ratio. Airtime share = `transmit_seconds / observed_seconds`, delivery ratio
= `data_frames_acked / data_frames_sent`.

Choices made without precedent (declare them as choices in `design_choices.md`):
room rows = 2 and at most 6 columns (runtime bound); session/pause range 0.3-30 s
(from shorter than one sweep pass, 0.44 s, to longer than the run); -88 dBm floor for
a drawn target (6 dB under preamble detection); 80 Mbit/s cap; a TCP flow is the same
Poisson offer written to a TCP socket (Cubic, 1448-byte segments) rather than a bulk
transfer; TCP is to the AP itself, with no wired segment behind it.

Known limitations to state, not fix: no candidate mobility; one flow per station;
no rate limiting or backhaul bottleneck; 11n only; `transmit_seconds` counts
management and ACK transmissions too.

## 4. Remaining work, in order

### 4.1 `run_sweep.py`

New command line per run: `--nCols --nAPs --meanOccupancy --occupancyShape
--loadMedianMbps --downlinkFraction --tcpFraction --targetAP --candidateSeed
--topologySeed --rngSeed --futureSeed --outDir --runTag --captureObs`. The binary no
longer takes `nSTAs`, `hotspotAPs`, `candidateX/Y` are optional.

Per-topology draws (ranges are choices; log them):

- `nCols` uniform on 2..6; `nAPs` uniform on 2..min(8, 2*nCols). Check `--identities`
  in `scripts/tf/train.py` covers 8.
- `meanOccupancy` log-uniform 0.3-4; `occupancyShape` log-uniform 0.3-10.
- `loadMedianMbps` log-uniform 0.3-10.
- `downlinkFraction`, `tcpFraction` each uniform on 0-1.

Run plan per topology, 5 candidates each:

- every candidate: one run, `targetAP=-1`, `captureObs=1` (tag `tNNNNN__cNN__draw`).
  Read the drawn target from `metadata["candidate"]["target_ap"]`.
- matched subset, 10% of topologies: additionally one run per remaining AP with
  `captureObs=0`. These topologies go to the test side only when scoring selection.
- noise-floor subset, 10% of topologies (disjoint): 4 more runs of the drawn target
  with `futureSeed` 2..5, `captureObs=0`.

The drawn target is an output of the first run, so the extra runs of a scan must be
planned after it finishes (two passes are simplest).

Budget: pilots took 8-48 s; assume 25 s mean, 14 workers, about 2,000 runs an hour.
Aim for about 1,000 topologies (5,000 training scans, roughly 7,500 runs, under 4 h).
Pilot 30 topologies first and check: fraction never associated, label histogram,
runtime tail, share of scans whose drawn target is not discovered by the projection.
The user's rule: pilot before any long job, run it in the background, run the
`heavy-compute-check` skill first. The user approved simulation and training jobs
until 14:00. Disk has 18 GB free; a v1 run directory averaged about 0.8 MB with
observation, so 7,500 runs fit.

### 4.2 Observation identity

`scripts/tests/check_observation_identity.py` checks that `observation.csv` before
8 s is identical across target APs. Adapt it to v2 flags and also check identity
across `futureSeed`. If it fails, the suspect is anything created per target before
8 s (the candidate's sockets and its place in `apGroupDevices`).

### 4.3 Cache and training

- `parse_scans.find_scans` raises when an AP's run is missing: a scan is now one
  observation run plus labels for any subset of APs.
- `cache_dataset.build_scan`: labels become NaN for unlabelled discovered APs; drop
  `n_hotspots` from the index; add the regime parameters and the extra labels.
- `train.py`: `gaussian_nll` already takes a mask; AND it with "label is present".
  `Scale` must take target statistics over present labels only. Select checkpoints
  on validation MAE, not regret (regret needs matched scans).
- A scan whose drawn target was not discovered has no labelled row; count and report
  them, do not silently drop.

### 4.4 Experiments worth the time (bespoke model and tabular only)

State each hypothesis and its rejection test before running.

1. Noise floor: MAE of predicting a scan's mean label over its 5 futures. Every
   model MAE is read against this.
2. Baselines on v2: predict-median, gradient boosting on aggregates, the joint
   bespoke model. Plus a physics baseline with no learning: PHY rate implied by the
   target's RSSI times the idle fraction of its channel.
3. Per-regime error: MAE binned by TCP share, downlink share, load, occupancy, walls
   between candidate and target.
4. Held-out regime: train on low `tcpFraction`, test on high, and the same for load.
5. If time: train on one label per scan versus all labels per scan at equal run
   count, on the matched subset, to test the "distinct observations beat matched
   labels" claim of section 3.

### 4.5 Rules that bind

User instructions in `CLAUDE.md` and `~/.claude/rules/` apply: documentation style,
choices logged in `report/design_choices.md`, no credential files, never kill
processes by pattern, never commit `linux_restore`, long jobs in the background,
report failures as failures. Untracked `AGENTS.md` and `presentation/` are the
user's; leave them. Do not touch any NetFlow/IPFIX data on this machine.

## 5. The direction report: what it should argue

Measured so far, on the v1 corpus (1,000 topologies, 21,235 labelled rows):
five-fold MAE in Mbit/s: bespoke joint 2.03 +- 0.07, LoRA Llama-3.2-1B 2.17 +- 0.05,
gradient boosting 2.26 +- 0.08, aggregates-only 2.37 +- 0.20, predict-median 10.33.
All checkpoints were selected on regret, not MAE, so these MAE comparisons are not
yet fair. Estimated (not measured) noise floor about 1.15 Mbit/s.

Inferences, to be tested on v2:

- On v1 everything learned sits within 0.35 Mbit/s of everything else, about 0.9
  above the estimated floor. The corpus is too easy to separate architectures: its
  traffic was stationary, so window averages are nearly sufficient. v2's on/off and
  TCP traffic is where the frame-level branch should matter if it ever does.
- The 1B LLM did not beat a 0.35M-parameter model. A 10B model on the same corpus
  will most likely tie it; the size ladder (Qwen3 0.6B to 14B, each with a
  random-initialised twin) is informative only on a corpus with headroom.

Candidate contributions, in my order of preference:

1. A benchmark for passive pre-association throughput prediction: the v2 simulator,
   regime metadata, a measured noise floor, and held-out-regime splits. Nothing
   comparable is public as far as I checked earlier in the session; verify before
   claiming it.
2. A negative-or-positive result on LLM backbones for this task, with the size ladder
   and random-init twins as the control NetLLM-style papers lack: does pretraining
   help at all, and does size?
3. The decomposed label (airtime x rate x delivery) as an explanation of *where*
   predictors fail: is the error in predicting contention or in predicting the link?

Open decisions for the user: whether unsaturated candidate demand comes next;
which cluster GPUs exist (sets the largest LLM); whether the report keeps selection
at all. Also: the report's LoRA row says "fold-0 provisional" but
`results/final/llm_joint_5fold_20261023` holds five folds.
