# Throughput prediction: what was rebuilt, what it shows, where to take it

13 November 2026. Branch `throughput-prediction`. Numbers marked *measured* come
from the code and data named beside them; everything else is labelled as an
estimate or an inference. Result files are under `results/v2/`, which git ignores,
so the numbers are repeated here.

## 1. The situation

The v1 simulator was built to make *choosing* an AP hard. Its single-valued
settings (load median, hotspot density, 3 to 5 stations per AP, a fixed AP lattice)
were tuned so the strongest AP was often the wrong one. On that corpus every learned
model landed between 2.03 and 2.37 Mbit/s mean absolute error (MAE), against a noise
floor I estimated at 1.15. There was about 0.9 Mbit/s of room, and no architecture
question could be settled inside it.

Once the task is prediction, none of that tuning is forced. One rule replaces it:
**sample the regimes instead of tuning them.** Each quantity the predictor should be
robust to becomes a random draw per deployment, over a wide range, written to the
run's metadata. The ranges have to be defended; single values no longer do.

## 2. What simulator v2 is

A *topology* is one deployment; a *scan* is one client standing in it and listening
for 6 s; the *label* is the client's mean uplink throughput from association to 20 s
when it sends more than the link can carry.

- **Floor:** two rows of 10 m rooms, 2 to 6 columns. Path loss is the TGax
  residential model (IEEE 802.11-14/0980r16) with 5 dB per wall crossed and 5 dB
  shadowing per link. v1's exponent of 3 had no source.
- **APs:** 2 to 8, in distinct rooms chosen at random, placed uniformly in the room.
  Channel uniform over 36/40/44/48, as before.
- **Stations:** per room, a Poisson count whose mean is gamma-distributed. Mean
  occupancy 0.3 to 4 per room; the gamma shape (0.3 to 10) sets how uneven rooms
  are. This replaces hotspot discs. Stations join the AP they receive most strongly.
- **Background traffic:** one flow per station. Uplink or downlink, UDP or TCP, by
  per-topology probabilities uniform on 0 to 1. Each flow alternates sessions and
  pauses of exponential length, with means between 0.3 s and 30 s per station.
  Offered load is log-normal per station with a per-topology median of 0.3 to 10
  Mbit/s, capped at 80, so capped stations always have a packet waiting.
- **Candidate:** unchanged, a saturated UDP uplink. The run also records the
  per-second delivered bytes and the three factors of the label (section 4.6).
- **Target AP:** the simulator draws one AP among those the client can hear. One
  topology in ten is run once per AP (*matched*, to score selection), and another one
  in ten is repeated five times with different traffic after the join (*replicated*,
  to measure the noise floor). The observation is byte-identical across all of these
  (checked by `scripts/tests/check_observation_identity.py` on 4 scans).

Every choice is logged in `report/design_choices.md`, section "Simulator v2", with
the ones that have no precedent marked as choices.

## 3. The corpus

*Measured*, from `data/runs_v2` (1,000 topologies, 5,000 scans, 8,790 runs, 232
minutes on 8 cores, no failed run):

- 4,797 scans were cached, giving 6,212 labelled rows. The other 203 scans (4%)
  joined an AP whose beacon the sweeping radio never decoded, so the models have no
  row for it. These are dropped, which removes some of the hardest cases.
- 5.4% of clients never associated; 11.2% of labels are exactly 0.
- Label quartiles 12.5 / 35.5 / 51.2 Mbit/s, mean 31.8, standard deviation 20.7.
  27% of labels lie between 50 and 60: the near-empty channel. v1's median was 5.3.

## 4. What was measured on it

All errors are MAE in Mbit/s on topology-held-out folds, mean and standard deviation
over five folds unless stated. The tabular numbers of 4.2 to 4.6 come from
`scripts/baselines/regime_analysis.py`: one gradient-boosted model at one untuned
setting, fitted to absolute error, on the window-aggregate features.

### 4.1 Noise floor

The 500 replicated scans have several labels that share one observation and differ
only in what the background stations did after the join. No predictor that sees the
observation can separate them, so their spread is a lower bound on the error.

Two kinds of replicate were run. The first (five labels per scan) changes only the
random draws made after the join, so a session or pause in force at the join keeps
its length. The second (four labels per scan, `redrawSessions`) also redraws those
lengths. That is the correct one: session and pause lengths are exponential, so how
long the overheard traffic goes on cannot be told from the observation.

| | Sessions kept | Sessions redrawn |
|---|---|---|
| Mean absolute deviation from the scan's median (optimistic) | 0.98 | **1.16** |
| Each label predicted by the mean of the others (pessimistic) | 1.40 | **1.66** |

Both columns use the same 500 scans and four labels per scan (futures 2 to 5), so
they compare like with like. Redrawing the sessions raises the floor by about 18%.
The floor is not uniform over the label range (sessions redrawn, pessimistic
version): 0.15 where the scan's mean label is under 1 Mbit/s, 2.4 from 1 to 10, 2.5
from 10 to 30, 2.1 from 30 to 50, 0.74 above 50. Within-scan variance is 1.5% of the
total label variance. The redrawn replicates are in `data/runs_v2_redraw`, produced
and summarised by scratch scripts, not by `run_sweep.py`.

The floor is the error of a predictor that knows the entire state of the simulated
world at the join. It is not the error reachable from the client's observation,
which sees a fraction of that state. The reachable error lies somewhere between
the floor and the best model, and section 4.9 is about where.

### 4.2 Predictors

| Predictor | MAE |
|---|---|
| Training median | 17.39 ± 0.44 |
| PHY rate implied by RSSI × idle share × 0.9, nothing fitted | 7.85 ± 0.21 |
| Ridge regression (tuned on validation) | 6.64 ± 0.17 |
| Random forest (tuned on validation) | 3.71 ± 0.18 |
| Gradient boosting (tuned on validation) | 3.69 ± 0.15 |
| Gradient boosting, untuned, absolute-error loss, train + validation | 3.51 ± 0.18 |
| Bespoke joint model (cell transformer), five folds | 3.70 ± 0.12 |
| Noise floor (sessions redrawn) | 1.16 to 1.66 |

**The frame-level model ties the aggregate model.** Fold by fold, the joint model
against tuned gradient boosting on the same test topologies: 3.75 / 3.67 / 3.84 / 3.77 / 3.48 against 3.71 / 3.89 / 3.73 / 3.69 / 3.41. The
differences change sign and are smaller than the fold-to-fold spread. Its
validation error in fold 0 stopped improving around epoch 45 of 60. Its 50% and
90% predictive intervals cover 50% and 85% of the test labels.

The 0.9 in the second row is the ratio of the largest label (59 Mbit/s) to the top
PHY rate (65): it is read off the corpus, so that rule is not strictly parameter-free.

### 4.3 Which inputs carry the prediction

| Features given to the model | MAE |
|---|---|
| The target AP's RSSI statistics only | 8.86 ± 0.50 |
| Its channel's statistics only | 8.76 ± 0.33 |
| Both | 3.56 ± 0.17 |
| Everything, including the AP's own traffic and its rank among APs | 3.51 ± 0.18 |

Link quality and channel load are each worth about half the problem and almost all
of what aggregates can deliver. The per-AP traffic features add 0.05, which is
inside the fold spread.

### 4.4 Where the error is

- **By label:** MAE is 2.6 for labels under 1, **5.6 for 1 to 10, 5.1 for 10 to 30**,
  4.0 for 30 to 50 and 1.7 above 50. The middle of the range, where the channel is
  shared, is where both the error and the noise floor are largest.
- **By load:** MAE rises from 2.0 in the least-loaded third of topologies to 5.2 in
  the most-loaded third.
- **By TCP share and downlink share:** flat. 3.45 / 3.54 / 3.56 across thirds of TCP
  share and 3.60 / 3.51 / 3.45 across thirds of downlink share.

### 4.5 Held-out regimes

Fit on the half of the topologies below the median of a regime parameter, test on
the half above, and compare with a fit on equally many topologies from the whole
range, on the same test rows.

| Held-out half | Fitted on the other half | Fitted on the whole range |
|---|---|---|
| High TCP share | 3.59 ± 0.30 | 3.61 ± 0.29 |
| Low TCP share | 3.68 ± 0.22 | 3.70 ± 0.25 |
| High downlink share | 3.69 ± 0.05 | 3.61 ± 0.12 |
| Low downlink share | 3.81 ± 0.38 | 3.66 ± 0.38 |
| High offered load | **9.62 ± 0.52** | 5.02 ± 0.33 |
| Low offered load | 3.34 ± 0.23 | 2.48 ± 0.20 |

I expected transport and direction to matter and they do not: a model that has never
seen TCP-heavy or downlink-heavy deployments predicts them as well as one that has.
What does not transfer is load. A model fitted on light deployments nearly doubles
its error on heavy ones.

### 4.6 The factors of the label

For a saturated sender, throughput is close to (share of time transmitting) × (PHY
rate) × (share of frames acknowledged). On v2 the product tracks the label with a
log-log correlation of 0.997 and a median ratio of 0.93.

| Factor | Quartiles | Variance of its log | Variance explained by the tabular model |
|---|---|---|---|
| Airtime share | 0.65 / 0.87 / 0.95 | 2.36 | 0.77 ± 0.04 |
| PHY rate (Mbit/s) | 55.7 / 63.3 / 64.5 | 0.21 | 0.80 ± 0.02 |
| Delivery ratio | 0.60 / 0.84 / 0.95 | 0.91 | 0.83 ± 0.01 |

Airtime share varies most and is predicted worst. The open part of the problem is
contention, the share of the medium the client will win, and not the link.

### 4.7 Selection, as a secondary metric

On the 457 matched scans with at least two discovered APs, mean regret (throughput
of the best AP minus that of the chosen one) is 2.26 Mbit/s for the strongest-RSSI
rule, 0.72 for gradient boosting and 0.70 for the random forest. The strongest AP is
the best one in 69% of scans. Selection is not trivial on v2 even though nothing was
tuned to make it so.

### 4.8 Does more data help

Tabular MAE against the share of training topologies: 4.58 at 5%, 4.16 at 10%, 3.83
at 25%, 3.65 at 50%, 3.51 at 100% (800 topologies). Each doubling still buys 0.15 to
0.2, so the curve has not flattened, but another 1,000 topologies would close only
a small part of the 2 Mbit/s gap to the floor.

### 4.9 What does not explain the gap

Four checks on why every model stops near 3.5, all with the same tabular model.

- **True deployment facts.** Adding the true mean link power, and the station count
  and offered load of the AP and of its channel, to the features: 3.51 → 3.36 ± 0.18.
  A richer version (stations split by distance from the client, by direction, by
  transport, by session length, each with duty-cycle-weighted load; a scratch
  calculation) gives 3.34 ± 0.17. Both are inside the fold spread.
- **True link powers.** For every transmitter on the AP's channel, its true mean
  power at the client and at the AP, binned at -82 and -92 dBm, with the
  duty-cycle-weighted load and count per bin. The bin "strong at the AP, unheard by
  the client" is the hidden-terminal load. Adding these: 3.51 → 3.29 ± 0.16 (a
  scratch calculation; the simulator's new `printLossMatrix` flag supplies the powers).
- **True session state at the join.** For each of those bins, the number of stations
  in a session at the join and their offered load (the simulator's new
  `printSessionsAtJoin` mode). Run on the first 400 topologies only (1,926 scans),
  where the features alone give 3.89 ± 0.43 and features plus link powers 3.64 ± 0.43.
  Adding the session state: 3.64 ± 0.41. No change.
- **Those facts alone**, without the observation: 7.5 to 7.9 in every version. The static description of
  the deployment is a much worse predictor than six seconds of listening.

These checks are weaker than they look, and I only saw why after the last one. The
privileged features are offered loads in Mbit/s. What competes with the client is
airtime, which is load divided by each station's PHY rate, and for TCP and for
saturated stations the load carried is not the load offered. The measured busy
fraction is airtime directly; my privileged features are a poor proxy for it, and a
tree model cannot build the ratio from binned sums. So the result is "these
hand-made descriptions of the hidden state do not help", not "hidden state does not
matter".

## 5. What this means

1. **The gap is real but nothing I tried closes it.** Tabular aggregates, a
   frame-level transformer, and aggregates plus the true deployment all land between
   3.3 and 3.8, against a floor of about 1.2 to 1.7. More data of
   the same kind buys about 0.15 per doubling (4.8).
2. **I do not know where the missing 2 Mbit/s is.** Three explanations remain, and
   they are inferences, not measurements:
   (a) it is in state the client cannot observe: which stations will be on the air
   and for what share of it, including what starts or stops while the sweeping
   radio is on another channel. Section 4.9 tried to test this and could not
   settle it;
   (b) it is in the observation and both models fail to extract it, for lack of
   data or of the right structure;
   (c) the floor is still understated in some way I have not found.
   The error is far above the floor at the ends of the label range as well: 2.6
   against a floor of 0.2 for labels under 1 Mbit/s, which are failed or dead links.
3. **The hard part is contention under load**, in the 1 to 30 Mbit/s band (4.4),
   through the airtime-share factor (4.6), and it is the one regime that does not
   extrapolate (4.5).
4. **TCP and downlink traffic changed nothing measurable** for an aggregate model.
   Either they do not matter to a saturated uplink client, or the way they were
   modelled (a rate-limited offer through a TCP socket, no bulk transfers) is too
   mild. I cannot tell which from this run.

The experiment that separates (a) from (b) has to give a model the hidden state in
the units that matter. The clean version: have the simulator record every station's
airtime share over the listening window and over the label window (the per-station
transmit time is already traced for the candidate). A model given the true airtime
shares of the listening window tests the sweep (what a client that heard everything
would know); one given those of the label window tests the future. My estimate is a
day of work and one re-simulation of the corpus, about four hours on this machine.

## 6. Directions, and the one I recommend

**A. A benchmark for passive pre-association throughput prediction.** The simulator,
the regime metadata, the measured noise floor and the held-out-load split, with
tabular, rule-based and neural reference points. The contribution is that a
predictor can be placed between a floor and a ceiling and tested out of regime.

What exists, from three web searches today whose result summaries I read but whose
papers I did not, so treat every line as a lead to verify:

- The ITU AI/ML challenge of 2020 and its follow-ups (Wilhelmi et al., arXiv
  2105.14219; dataset on Zenodo, record 4059189) predict throughput in overlapping
  WLANs with MLPs and graph networks. Their input is the deployment itself (node
  positions, channels, powers) from the Komondor simulator: the planner's view, and
  not what one client can hear before it joins.
- Passive client-side estimation is older and model-based: Vasudevan et al. (IMC
  2005) estimate "potential bandwidth" from beacon timing, and papers of 2011 to
  2014 add CSMA/CA models over sniffed traffic. None in the results was learned.
- One search for an LLM predicting WLAN throughput returned none; the LLM papers
  it found tune contention windows or make roaming decisions.

If that holds up, the gap in the literature is specific: a learned predictor on the
client's passive view, with a public corpus. Section 4.9 adds a contrast with the
ITU setting that I did not expect: here the planner's view alone (7.5) predicts
much worse than the client's view (3.5), though with the weak featurisation
that 4.9 admits to.

**B. Does a pretrained LLM backbone help, and does its size?** The Qwen3 ladder with
random-initialised twins. This is the question your advisor asked. On v1 it could
not be answered because everything tied. On v2 the bespoke frame model again ties
gradient boosting, so the same outcome is the likeliest one for the ladder: every
backbone at about 3.7. That is a publishable negative only if the reason is known,
which is what the experiment at the end of section 5 provides.

**C. What limits prediction: the information or the model.** Locate the gap between
3.5 and the floor: the privileged-information ladder (observation, plus true link
table, plus true activity), the three label factors as separate targets, and the
held-out-load split. This turns "our model reaches X" into "a passive client can
know this much, and here is what it cannot know".

My recommendation: **A as the frame and C as the central result, with B run only
after the airtime-oracle experiment.** If the limit turns out to be information, B
becomes one confirming row (a 14B backbone cannot beat it either) and costs one
cluster run, not a ladder. If the limit is the model, the ladder is justified and
has a target to reach. Running B first risks a table of ties with no explanation,
which is where v1 ended.

## 7. Decisions that are yours

- **The easy mass.** 27% of labels are the near-empty channel. Raising the lower
  bound of occupancy or load would move data to where the error is. I did not,
  because it would be tuning the corpus to a result; reporting by label band (4.4)
  is the alternative.
- **Extra labels.** You chose "delay, loss and airtime share". For a saturated UDP
  sender delay and loss are functions of throughput, so I recorded the three factors
  and the per-second series instead.
- **Dropped scans.** 4% of scans join an AP the scan never discovered. Dropping them
  flatters every model equally. The alternative is to draw the target among
  discovered APs, which needs the projection inside the sweep loop.
- **TCP realism.** Whether to add bulk TCP transfers and a wired segment behind the AP.
- **Unsaturated candidate demand**, deferred by your decision.
- **CPU temperature.** Both sweeps held `Tctl` at 95 °C with the fan near 7,000 RPM,
  above your 90 °C line, at 8 workers and again at 4 workers beside the GPU job. I
  did not pause them, per your rule, and could not ask you.

## 8. Shortcuts and loose ends

- The redrawn-session floor, the three privileged-fact checks beyond the one in
  `regime_analysis.py`, and the selection numbers of 4.7 come from scratch scripts,
  copied unpolished into `scripts/scratch_v2/` so the numbers can be traced; they
  do not follow the repository's conventions. The simulator flags they use are (`redrawSessions`,
  `printLossMatrix`, `printSessionsAtJoin`). If you want to keep these analyses they
  should become proper scripts.
- The simulator binary was rebuilt three times today. Without the new flags it
  reproduces corpus runs exactly (checked on three runs after the first rebuild).
- The tabular analyses use one untuned model setting. The tuned grid of
  `baselines/train.py` gives 3.69, so the conclusions do not hinge on it.
- The unit tests pass again (the test of the sweep's removed resume check is
  deleted). The ranking objective of `tf/train.py` has no meaning
  under MAE selection. The folds stratify on AP count only.
- The v1 report's LoRA row says "fold-0 provisional", but
  `results/final/llm_joint_5fold_20261023` holds five folds.
- `data/runs_v2` (7.1 GB) and `data/runs_v2_pilot` leave 9 GB free on the disk.
