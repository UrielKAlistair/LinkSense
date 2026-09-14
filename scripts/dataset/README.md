# Dataset builders

These scripts turn the simulator's run directories into the corpora the models
train on.

## What a client has

A scan is a client standing still for the 6 s before it joins, listening. The
simulator records each scan in its run directories:

| File | Contents |
|---|---|
| `observation.csv` | every frame decoded on channels 36, 40, 44 and 48: transmission start and end, channel, BSSID, transmitter, frame type, beacon and retry flags, length, RSSI, airtime, rate. No payload and no receiver address. |
| `chanbusy.csv` | for every 1 ms of the window, the share of it each occupied channel's carrier sense reported busy |
| `metadata.json` | the deployment the simulator built, and the label |

After the projection below, the first two files are everything the client
knows, and every model input is computed from them alone. Model inputs are
therefore estimates of what the simulator knows exactly. An AP's station count,
for instance, can only be the number of stations heard talking to it, and a
station never heard is never counted.

`metadata.json` is simulator ground truth and never a model input. The builders
read it to join each run to its row (which AP the run joined, and that AP's
BSSID), for the labels, to check that a scan has all of its runs, and for the
`gt_*` columns used to slice results.

## What is predicted

The throughput the client would get from each AP it could join. The simulator
replays a scan once per AP, joining a different AP each time. The replays are
identical up to the join, so one observation stands for all of a scan's runs,
and each run adds only its label. A model scores every option in a scan and is
judged by the one it ranks first.

## Step 1: project onto one radio

The simulator gives every occupied channel its own listening radio for the
whole window, which is more than any client receives. A client has one radio,
which hops between the four channels. [`projection.py`](projection.py) keeps
only what that radio could have received:

1. Round every frame's RSSI to a whole dBm, the resolution a device reports.
2. Schedule the hops: the four channels in a random order, 110 ms per visit,
   for as many whole passes as the window holds (13 in 6 s), the last visit
   ending at the join.
3. Keep a frame, and a `chanbusy.csv` millisecond, only if the radio was on
   its channel, and done retuning, from its start to its end. About a quarter
   of the frames survive whatever the number of occupied channels, since the
   radio also spends its dwells on empty ones.

`project()` returns a `ProjectedScan`, held in memory and never written:

| Field | Contents | Used for |
|---|---|---|
| `frames` | the kept `observation.csv` rows | every feature of every builder |
| `busy` | the kept `chanbusy.csv` milliseconds | the busy fraction of each binned step and of each table channel |
| `dwell_schedule` | the channel of each visit | the channel of each binned step |
| `sweep_start` | when the first visit began | where the first binned step begins |
| `window` | the length of the window | the frame corpus's `time_fraction`; the table's `meta_window_s`; the number of binned steps under `--all-channels` |
| `listen_s` | the time spent receiving on each channel, computed from `dwell_schedule` | dividing the table's counts into rates |

`frames` and `busy` record what the radio heard, and the schedule records when
it listened. The rows alone cannot tell the two apart: a visit that decodes
nothing leaves no frame, and a channel with no transmitter has no
`chanbusy.csv` rows at all. The visit order is seeded from the scan ID, so
every builder gets the same `ProjectedScan` for a scan.

`--all-channels` skips steps 2 and 3 and keeps every channel for the whole
window. No client can observe that; it measures what the hopping costs.

## Step 2: present the frames to a model

The projection leaves each scan as a variable-length list of frames. There are
three ways to hand that list to a model:

| Representation | Builder | Output | Trained by |
|---|---|---|---|
| every frame as a token | `build_frame_corpus.py` | `frames.npz` | `train_frames.py` |
| frames summarised in short time bins | `build_binned_corpus.py` | `binned.npz` | `train_temporal.py`, `aggregate_baseline.py` |
| frames summarised over the whole window, one row per option | `build_aggregate_table.py` | `aggregate.csv` | `train_eval.py`, `learning_curve.py` |

The frame and binned corpora are what the project studies. The aggregate table
gives linear and tree models a fixed-length input, as a reference point.

Every builder takes the same projection flags and gets its scans from
[`scans.py`](scans.py). It groups the run directories into scans, checks that
each scan has a run for every AP, and keeps as options the APs with a beacon in
the projected recording, each with its label, since a client cannot join a BSS
it never heard. A scan left with fewer than two options is skipped. No builder
reads another's output, and no corpus carries another's features.

### Frame corpus

One token per frame, in time order, carrying when in the window its
transmission started, RSSI, airtime, length, rate, frame type, beacon and retry
flags, and the time since the previous frame. BSSIDs are not model input.
Instead, every (option, frame) pair gets a relation code: whether the frame was
on the option's channel, from its BSS, and sent by the AP itself. A scan over
16,384 frames is thinned evenly across the window, which happens to most scans
under `--all-channels`.

### Binned corpus

Each 110 ms dwell is cut into whole bins of about 10 ms, and each bin is one
step of the sequence, on the channel the radio was tuned to. For every option,
a step gives:
- where it lies in the window and in its dwell;
- whether the tuned channel is the option's own, and the share of options on
  that channel;
- the channel's busy fraction;
- a summary of every frame in the bin;
- a summary of the frames from the option's own BSS.

A signal level with no frame behind it is the noise floor if the radio listened
and heard nothing. Otherwise the last reading is carried forward, with its age.

Under `--all-channels` there are no dwells. Each 100 ms bin becomes four steps,
one per channel, laid out exactly like a single-radio step.

### Aggregate table

One row per (scan, option). The rows of a scan share one observation but
describe different APs, so their features differ, and two APs on one channel
share their `feat_chan_*` values.

| Prefix | Meaning |
|---|---|
| `topology_id`, `scan_id`, `ap_index`, `run_id` | identifiers |
| `feat_chan_*` | the option's channel: airtime and busy fractions, frame and byte rates, frame type mix, retries, data rate, RSSI, BSSIDs and transmitters |
| `feat_ap_*` | the option's own BSS: beacon RSSI, beacon count and spacing, frame and byte rates, airtime, data frames and rate, transmitters (the station-count estimate), retries |
| `feat_rel_*` | the option against the other options of its scan: option count, RSSI rank and margins, share of airtime and of transmitters |
| `label_*` | throughput and association after joining the option |
| `gt_*` | simulator parameters, for slicing results; never model input |
| `meta_*` | the window, whether it was projected onto one radio, and how long the option's channel was listened to |

## Other scripts

- `scan_stats.py` prints, per deployment size, the random regret (how much
  throughput a random pick among the heard APs loses against the best one) and
  the share of APs, and of their stations, that the radio heard.

## Running

```
python scripts/dataset/build_aggregate_table.py data/runs --out data/aggregate.csv
python scripts/dataset/build_binned_corpus.py data/runs --out data/binned.npz
python scripts/dataset/build_frame_corpus.py data/runs --out data/frames.npz
```

Add `--all-channels` to any of them for the unprojected version.
