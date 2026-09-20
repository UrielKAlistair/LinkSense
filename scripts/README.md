# Pipeline

## 1. Simulation

`simulate/run_sweep.py` sweeps topological arrangements of APs and candidate
positions, and gives back the raw observation data everything downstream is
built from. A topology is one deployment of between two and eight APs and a
random number of background stations; a candidate is the client standing
somewhere else in that same deployment. The output is one directory per run:

    t00007            a topology: one deployment of APs and background stations
    t00007__c01       a candidate: that same client standing somewhere else
    t00007__c01__ap1  a run: that candidate joining AP 1

Under every frozen observation configuration the simulation is rerun once per
AP, the candidate associating with a different one each time. That is what makes
each AP's true throughput known, and those true throughputs are what we train
on to predict what an AP will deliver before the client has associated with it.

    .venv/bin/python3 scripts/simulate/run_sweep.py \
        --binary <path to the built simulator in your ns-3 tree> \
        --out-dir data/runs --n-topologies 800 \
        --candidates-per-topology 5 --workers 14

`--binary` is the executable ns-3 builds, not the source. The simulator itself
lives in `sim/my-wifi-test.cc`, one level up from here: symlink that file into
ns-3's `scratch/` directory and build it there, so it stays version-controlled in
this repository while ns-3 compiles it in place.

Everything random is drawn from `--sweep-seed`, left at its default here, and
keyed per topology, so raising `--n-topologies` extends the corpus instead of
redrawing it. A rerun resumes: runs already on disk are left alone.

## 2. Dataset generation

The simulator gives each of its scanner radios a channel of its own, all four
listening for the whole window at once. `common/projection.py` projects that
recording down onto what one real sweeping radio would have seen: a single radio
dwelling on channels 36, 40, 44 and 48 in turn, keeping a frame only where it
was tuned to that channel, and done retuning, for the whole transmission.

`tf/cache_dataset.py` consumes the projection. It reads through every CSV the
simulation wrote, adds summary statistics, and stores the result as NumPy arrays
for quick access down the line.

    .venv/bin/python3 scripts/tf/cache_dataset.py data/runs --out data/cache

That cache is then read a second time to build one aggregate CSV, one row per
candidate AP, for the baseline models that work only from tabular data.

    .venv/bin/python3 scripts/baselines/build_datatable.py data/cache \
        --out data/aggregate.csv

## 3. Models

`tf/` holds three models over the cached cells. Their architectures are in the
report; what separates them is who reads the grid:

  `JointAPModel`   scores every AP in one pass, the AP rows attending to each
                   other, so each prediction sees its competitors.
  `TargetAPModel`  scores one AP at a time, every cell tagged by its role
                   relative to that AP.
  `BinnedModel`    reads per-dwell aggregates alone, with no frames and no
                   carrier-sense trace.

`baselines/` holds three scikit-learn estimators over the aggregate CSV: ridge
regression, a random forest and a histogram gradient boosting ensemble. Each is
a few lines in `baselines/train.py`.

Each model's training file sits in its own folder, but the train, validation and
test split is done once, in `common/splits.py`.

    .venv/bin/python3 scripts/tf/train.py data/cache \
        --model joint_ap --out results/joint_ap

    .venv/bin/python3 scripts/baselines/train.py data/aggregate.csv \
        --out-dir results/table

The three `--model` choices are invoked the same way.
