#!/bin/bash
# Build the corpus on one whole CPU node and fit the tabular models on it.
#
# INPUT   N_TOPOLOGIES, and the ns-3 binary built by setup.sh
# OUTPUT  data/runs, data/cache, data/aggregate.csv, results/table
#
# Run:
#   sbatch --export=ALL,N_TOPOLOGIES=1000 scripts/cluster/corpus.sbatch
#SBATCH -J linksense-corpus
#SBATCH -p cpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH -t 1-00:00:00
#SBATCH -o logs/%x-%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster/environment.sh

python scripts/simulate/run_sweep.py --binary "$NS3_DIR/build/scratch/ns3.45-my-wifi-test" \
    --out-dir data/runs --n-topologies "$N_TOPOLOGIES" --workers "$SLURM_CPUS_PER_TASK"
python scripts/tf/cache_dataset.py data/runs --out data/cache --workers "$SLURM_CPUS_PER_TASK"
python scripts/baselines/build_datatable.py data/cache --out data/aggregate.csv
python scripts/baselines/train.py data/aggregate.csv --out-dir results/table
