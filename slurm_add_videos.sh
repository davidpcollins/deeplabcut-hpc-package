#!/bin/bash
#SBATCH --job-name=add-videos
#SBATCH --partition=cpu                     # which cluster partition to use
#SBATCH --cpus-per-task=8                   # CPU cores for data loading
#SBATCH --mem=64G                           # RAM
#SBATCH --time=05:00:00                     # wall-time limit (HH:MM:SS)
#SBATCH --output=logs/add-videos_%j.out     # save output text to logs/add-videos_JOBID.out
#SBATCH --error=logs/add-videos_%j.err      # save error text to logs/add-videos_JOBID.err

# =====================================================================
#  DLC 3 — Add new videos to existing project on HPC cluster with Apptainer
# =====================================================================
#
#  Submit with:
#      sbatch slurm_add_videos.sh
#
#  Before first use:
#      1.  Build the container: apptainer build dlc3.sif dlc3.def
#      2.  Make sure your project directory (with a config.yaml) is hosted on HPC
#      3.  Add new videos to a directory on the HPC (can be the same as your project dir or a different one)
#      4.  Edit the paths below
# =====================================================================

# ── Paths (EDIT THESE) ──────────────────────────────────────────────
DLC_SIF="/cluster/path/to/apptainer/file.sif"           # Apptainer image
SCRIPTS_DIR="/cluster/path/to/dlc_hpc_package"          # dir with dlc_add_videos.py
CONFIG="/cluster/path/to/deeplabcut/config.yaml"        # DLC config
TO_ADD="/cluster/path/to/new_videos"                    # dir with new videos to add
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"

# Load Apptainer if it's a module on your cluster
module load apptainer 2>/dev/null || true

# --nv  enables NVIDIA GPU passthrough
# --bind mounts the filesystem so the container can see your data
apptainer exec --bind /scratch/user/davcollins:/scratch/user/davcollins \
    "$DLC_SIF" \
        python "${SCRIPTS_DIR}/dlc_add_videos.py" \
        --config_path "$CONFIG" \
        --folder_to_add "$TO_ADD"

echo "End time    : $(date)"
