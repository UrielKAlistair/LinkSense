# wifi-ap-selection

Learning which WiFi access point a client should join, from what its radio
can observe *before* it associates.

An ns-3 simulation generates realistic multi-AP scenarios; a client scans,
picks an AP, and the throughput it then achieves becomes the label. The
learning task is a **choice set**: one scenario, one pre-association
observation, and one label per AP the client could have joined.

The full write-up is `report/report.tex` (compile with `pdflatex`). Read that
first — it explains the design decisions and, importantly, several ways the
earlier pipeline was silently broken.

## Layout

| Path | Role |
|---|---|
| `sim/my-wifi-test.cc` | The ns-3 simulation. Symlinked into `ns-3.45/scratch/`. |
| `scripts/validate_sim.py` | Seven physics/behaviour checks (V1–V7). Run this after touching the sim. |
| `scripts/run_sweep.py` | Samples scenarios, runs each once per AP as a matched set. |
| `scripts/build_dataset.py` | Observations → choice-set feature table. |
| `scripts/inspect_dataset.py` | Describes a dataset and fails on structural problems. |
| `scripts/train_eval.py` | Trains all models, selects on validation, reports on test. |
| `scripts/ablation_context.py` | Whether the set context beats hand-built relative features. |
| `models/data.py` | Loading, imputation, group-wise splits, leakage guard. |
| `models/evaluate.py` | Regret / top-1 / Spearman, plus the heuristic baselines. |
| `models/baseline.py` | Tree ensembles with permutation importance. |
| `models/ranker.py` | DeepSets-style listwise ranker (and its pointwise ablation). |

## Setup

```bash
# ns-3.45 lives alongside this repo. NS3_WARNINGS_AS_ERRORS=OFF is required:
# an unrelated warning in ns-3's own ipv4-l3-protocol.cc otherwise fails the build.
cmake -S ../ns-3.45 -B ../ns-3.45/cmake-cache-release -G Ninja \
      -DCMAKE_BUILD_TYPE=Release -DNS3_EXAMPLES=OFF -DNS3_TESTS=OFF \
      -DNS3_VISUALIZER=OFF -DNS3_WARNINGS_AS_ERRORS=OFF
ninja -C ../ns-3.45/cmake-cache-release ns3.45-my-wifi-test

python3 -m venv .venv
.venv/bin/pip install numpy pandas scikit-learn matplotlib
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
```

The bundled `./ns3` wrapper does not run under Python 3.14 (an argparse
incompatibility unrelated to this project), so CMake and Ninja are driven
directly. Use the release build for anything at scale — debug is ~20x slower.

## Pipeline

```bash
BIN=../ns-3.45/build/scratch/ns3.45-my-wifi-test

.venv/bin/python scripts/validate_sim.py --binary $BIN --workers 10
.venv/bin/python scripts/run_sweep.py --binary $BIN --out-dir data/runs \
    --n-groups 800 --workers 12                       # ~46 min, ~750 MB
.venv/bin/python scripts/build_dataset.py data/runs --out data/dataset.csv
.venv/bin/python scripts/inspect_dataset.py data/dataset.csv
.venv/bin/python scripts/train_eval.py data/dataset.csv --out-dir report/
```

## Two things to keep in mind

**Only `feat_*` columns may be model inputs.** `gt_*` columns are simulator
ground truth (true distances, true offered load, seeds) that a real client
cannot observe before associating. `models/data.py` enforces this; don't
route around it.

**Split by group, never by row.** Rows within a group share one observation,
so a row-wise split leaks near-copies of test observations into training.
Use `split_by_group`.
