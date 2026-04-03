#!/bin/bash
#SBATCH --job-name=dlc-array
#SBATCH --partition=pod                     # ← change to your cluster's GPU partition
#SBATCH --gres=gpu:1                        # Number of GPUs to request (per array task)
#SBATCH --cpus-per-task=4                   # CPU cores for data loading (per array task)
#SBATCH --mem=16G                           # RAM (per array task)
#SBATCH --time=02:00:00                     # per-video time limit
#SBATCH --output=logs/dlc-array_%A_%a.out   # save output text to logs/dlc-array_ARRAYID_TASKID.out
#SBATCH --error=logs/dlc-array_%A_%a.err    # save error text to logs/dlc-array_ARRAYID_TASKID.err
#SBATCH --array=0-39%8                      # ← set upper bound = (number of videos - 1)

# =====================================================================
#  DLC 3 — Analyse videos in PARALLEL (one video per array task)
# =====================================================================
#
#  Each SLURM array task gets its own GPU and processes one video.
#  This is the fastest way to analyse many videos.
#
#  BEFORE SUBMITTING: count your videos and set --array=0-<N-1>.
#  You can also throttle concurrency:  --array=0-49%10  (max 10 at once)
#
#  Submit with:
#      sbatch slurm_analyze_array.sh
# =====================================================================

# ── Paths (EDIT THESE) ──────────────────────────────────────────────
DLC_SIF="/cluster/path/to/apptainer/file.sif"           # Apptainer image
CONFIG="/cluster/path/to/deeplabcut/config.yaml"        # DLC config
VIDEO_DIR="/cluster/path/to/all/videos"                 # dir with videos to analyze (can be same as project dir or different)    
SCRIPTS_DIR="/cluster/path/to/dlc_hpc_package"          # dir with dlc_analyze.py
VIDEOTYPE=".mp4"                                        # string of comma-separated video extensions, e.g. ".mp4,.avi,.mov"
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

echo "Array Job ID : $SLURM_ARRAY_JOB_ID"
echo "Task ID      : $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
# ───────────────────────────────────────────────────────────────────

module load apptainer 2>/dev/null || true

# --nv  enables NVIDIA GPU passthrough
# --bind mounts the filesystem so the container can see your data. If HPC has multiple partitions (e.g. /scratch and /home), 
#   you may need multiple --bind statements to give the container access to all necessary paths.
# Check dlc_analyze.py for additional arguments. Must include --array_mode to run in parralel
apptainer exec --nv \
    --bind /path/to/your/data:/path/to/your/data \
    "$DLC_SIF" \
    python "${SCRIPTS_DIR}/dlc_analyze.py" "$CONFIG" \
        --video_dir "$VIDEO_DIR" \
        --videotype "$VIDEOTYPE" \
        --shuffle 1 \
        --create_video \
        --no-filter \
        --array_mode \ 
        

echo "End time     : $(date)"