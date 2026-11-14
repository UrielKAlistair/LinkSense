#!/bin/bash
# Train one model of the backbone ladder per array task, each on one A100.
#
# The array runs every backbone twice, with its pretrained weights and with the
# same architecture initialised at random, and ends with the bespoke model that
# reads the grid with its own encoder blocks. A task that hits its time limit
# carries on from progress.pt when submitted again.
#
# INPUT   data/cache, and the backbones fetch_backbones.sh put under HF_HOME
# OUTPUT  results/ladder/<model>
#
# Run:
#   sbatch scripts/cluster/train.sbatch
#SBATCH -J linksense-train
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 2-00:00:00
#SBATCH --array=0-10
#SBATCH -o logs/%x-%A_%a.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/cluster/environment.sh

mapfile -t backbones < scripts/cluster/backbones.txt
task=$SLURM_ARRAY_TASK_ID

if (( task == 2 * ${#backbones[@]} )); then
    python scripts/tf/train.py data/cache --model joint_ap --out results/ladder/bespoke
    exit
fi

backbone=${backbones[task / 2]}
name=${backbone##*/}
if (( task % 2 )); then
    python scripts/tf/train.py data/cache --model joint_ap --grid-reader llm \
        --backbone "$backbone" --random-backbone --out "results/ladder/${name}_random"
else
    python scripts/tf/train.py data/cache --model joint_ap --grid-reader llm \
        --backbone "$backbone" --out "results/ladder/$name"
fi
