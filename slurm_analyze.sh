#!/bin/bash
#SBATCH --job-name=analyze-videos
#SBATCH --partition=gpu                         # ← change to your cluster's GPU partition
#SBATCH --gres=gpu:1                            # Number of GPUs to request
#SBATCH --cpus-per-task=8                       # CPU cores for data loading
#SBATCH --mem=32G                               # RAM
#SBATCH --time=12:00:00                         # wall-time limit (HH:MM:SS)
#SBATCH --output=logs/analyze-videos_%j.out     # save output text to logs/analyze-videos_JOBID.out
#SBATCH --error=logs/analyze-videos_%j.err      # save error text to logs/analyze-videos_JOBID.err

# =====================================================================
#  DLC 3 — Analyse ALL videos sequentially on one GPU
# =====================================================================
#
#  Good when you have a moderate number of videos (< ~50).
#  For large batches, use slurm_analyze_array.sh instead.
#
#  Submit with:
#      sbatch slurm_analyze.sh
# =====================================================================

# ── Paths (EDIT THESE) ──────────────────────────────────────────────
DLC_SIF="/cluster/path/to/apptainer/file.sif"           # Apptainer image
CONFIG="/cluster/path/to/deeplabcut/config.yaml"        # DLC config
VIDEO_DIR="/cluster/path/to/all/videos"                 # dir with videos to analyze (can be same as project dir or different)    
SCRIPTS_DIR="/cluster/path/to/dlc_hpc_package"          # dir with dlc_analyze.py
VIDEOTYPE=".mp4"                                        # video file extension (e.g. .mp4, .avi)
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

echo "Job ID      : $SLURM_JOB_ID"
echo "Node        : $(hostname)"
echo "Start time  : $(date)"

module load apptainer 2>/dev/null || true

# --nv  enables NVIDIA GPU passthrough
# --bind mounts the filesystem so the container can see your data
# Check dlc_analyze.py for additional arguments (e.g. --filter)
apptainer exec --nv \
    --bind /path/to/your/data:/path/to/your/data \
    "$DLC_SIF" \
    python "${SCRIPTS_DIR}/dlc_analyze.py" "$CONFIG" \
        --video_dir "$VIDEO_DIR" \
        --videotype "$VIDEOTYPE" \
        --filter

echo "End time    : $(date)"
