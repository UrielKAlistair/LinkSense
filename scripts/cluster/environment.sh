# Sourced by every job and by setup.sh: where things live on PARAM Rudra, and
# the Python environment. Compute jobs read backbones from HF_HOME alone.
export NS3_DIR=/scratch/$USER/ns-3.45
export HF_HOME=/scratch/$USER/huggingface
export HF_HUB_OFFLINE=1
export LD_LIBRARY_PATH=$NS3_DIR/build/lib64:$NS3_DIR/build/lib:${LD_LIBRARY_PATH:-}

module load miniconda
conda activate linksense
