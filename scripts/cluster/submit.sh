#!/bin/bash
# Queue the whole run on PARAM Rudra: one CPU job builds the corpus, and once
# it has succeeded one GPU job per model trains on it.
#
# PROCESS
#   1. Submit corpus.sbatch, which simulates, caches and fits the tabular models.
#   2. Submit train.sbatch as an array held until step 1 exits cleanly.
#
# Run, from the repository root on a login node:
#   N_TOPOLOGIES=1000 scripts/cluster/submit.sh
set -euo pipefail

mkdir -p logs
corpus_job=$(sbatch --parsable --export=ALL,N_TOPOLOGIES="${N_TOPOLOGIES:?}" \
    scripts/cluster/corpus.sbatch)
sbatch --dependency=afterok:"$corpus_job" scripts/cluster/train.sbatch
