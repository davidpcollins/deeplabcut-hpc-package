#!/bin/bash
#SBATCH --job-name=dlc-train
#SBATCH --partition=pod                     # ← change to your cluster's GPU partition
#SBATCH --gres=gpu:1                        # GPUs to request
#SBATCH --cpus-per-task=8                   # CPU cores for data loading
#SBATCH --mem=64G                           # RAM
#SBATCH --time=06:00:00                     # wall-time limit (HH:MM:SS)
#SBATCH --output=logs/dlc-train_%j.out      # save output text to logs/dlc-train_JOBID.out
#SBATCH --error=logs/dlc-train_%j.err       # save error text to logs/dlc-train_JOBID.err

# =====================================================================
#  DLC 3 — Train network on a single GPU node
# =====================================================================
#
#  Submit with:
#      sbatch slurm_train.sh
#
#  Before first use:
#      1.  Build the container:  apptainer build dlc3.sif dlc3.def
#      2.  Create your DLC project + label frames on your workstation
#      3.  Copy the project directory to the cluster
#      4.  Edit the paths below
# =====================================================================

# ── Paths (EDIT THESE) ──────────────────────────────────────────────
DLC_SIF="/cluster/path/to/apptainer/file.sif"           # Apptainer image
CONFIG="/cluster/path/to/deeplabcut/config.yaml"        # DLC config  
SCRIPTS_DIR="/cluster/path/to/dlc_hpc_package"          # dir with dlc_analyze.py
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "GPUs        : $CUDA_VISIBLE_DEVICES"
echo "Start time  : $(date)"

# Load Apptainer if it's a module on your cluster
module load apptainer 2>/dev/null || true

# --nv  enables NVIDIA GPU passthrough
# --bind mounts the filesystem so the container can see your data
# Check dlc_train.py for additional arguments (e.g. --shuffle, --maxiters)
apptainer exec --nv \
    --bind /path/to/your/data:/path/to/your/data \
    "$DLC_SIF" \
    python "${SCRIPTS_DIR}/dlc_train.py" "$CONFIG" \
        --shuffle 1

echo "End time    : $(date)"
