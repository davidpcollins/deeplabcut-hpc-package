#!/bin/bash
#SBATCH --job-name=dlc-headdir
#SBATCH --partition=cpu               # ← no GPU needed for head direction
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/dlc-headdir_%j.out
#SBATCH --error=logs/dlc-headdir_%j.err

# =====================================================================
#  Head-direction computation from DLC .h5 output (CPU-only)
# =====================================================================
#
#  This does NOT need a GPU — it just reads .h5 files and computes
#  angles.  Run it after the video analysis jobs complete.
#
#  Submit with:
#      sbatch slurm_head_direction.sh
# =====================================================================

# ── Paths (EDIT THESE) ──────────────────────────────────────────────
DLC_SIF="/path/to/dlc3.sif"
SCRIPTS_DIR="/path/to/hpc"
H5_DIR="/path/to/all/videos"          # where DLC saved the .h5 files
OUTPUT_DIR="/path/to/head_direction_results"
FPS=100.0
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs "$OUTPUT_DIR"

echo "Job ID      : $SLURM_JOB_ID"
echo "Start time  : $(date)"

module load apptainer 2>/dev/null || true

apptainer exec \
    --bind /path/to/your/data:/path/to/your/data \
    "$DLC_SIF" \
    python "${SCRIPTS_DIR}/dlc_head_direction_batch.py" \
        --h5_dir "$H5_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --fps "$FPS"

echo "End time    : $(date)"
