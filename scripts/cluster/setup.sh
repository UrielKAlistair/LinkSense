#!/bin/bash
# Prepare PARAM Rudra once, on a login node: the Python environment, the ns-3
# binary and the backbones.
#
# INPUT   the ns-3.45 source tree copied to NS3_DIR, this repository on /scratch
#
# PROCESS
#   1. Create the conda environment the jobs activate.
#   2. Build the simulator in ns-3's scratch directory with GCC 14.
#   3. Download every backbone of backbones.txt into HF_HOME.
#
# Run, from the repository root:
#   scripts/cluster/setup.sh
set -euo pipefail

module load miniconda
conda create -y --name linksense python=3.12
source scripts/cluster/environment.sh
pip install torch transformers peft numpy pandas scikit-learn

module load spack
. /home/apps/spack/share/spack/setup-env.sh
spack load gcc@14.2.0
ln -sf "$PWD/sim/my-wifi-test.cc" "$NS3_DIR/scratch/my-wifi-test.cc"
(cd "$NS3_DIR" && ./ns3 configure --build-profile=release && ./ns3 build scratch_my-wifi-test)

while read -r backbone; do
    HF_HUB_OFFLINE=0 hf download "$backbone"
done < scripts/cluster/backbones.txt
