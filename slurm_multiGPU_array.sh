#!/bin/bash
#SBATCH --job-name=multigpu-array
#SBATCH --partition=gpu                        # ← change to your cluster's GPU partition
#SBATCH --gres=gpu:2                           # Number of GPUs to request (per array task)
#SBATCH --cpus-per-task=8                      # CPU cores for data loading (per array task)
#SBATCH --mem=64G                              # RAM (per array task)
#SBATCH --time=04:00:00                        # per-video time limit
#SBATCH --output=logs/multigpu-array_%A_%a.out # save output text to logs/iter2_analyze_array_ARRAYID_TASKID.out
#SBATCH --error=logs/multigpu-array_%A_%a.err  # save error text to logs/iter2_analyze_array_ARRAYID_TASKID.err
#SBATCH --array=0-13                           # set upper bound to ceil(num_videos/2)-1

# =====================================================================
#  DLC 3 — Analyse videos in PARALLEL (two videos per array task)
# =====================================================================
#
#  Each SLURM array task gets multiple GPUs and processes one videos
#  on each. Useful if the cluster has restrictions on the number
#  of simultaneous jobs which can throttle the --array_mode arg.
#  This is the fastest way to analyse many videos.
#
#  BEFORE SUBMITTING: 
#     1. count your videos and set --array=0-<ceil(N/2)-1>.
#     You can also throttle concurrency:  --array=0-49%10  (max 10 at once)
#     2. edit paths below
#     3. edit --bind commands below if needed
#     4. edit args passed to dlc_analyze.py (e.g. --shuffle 1 or --dynamic_cropping)
#        Do NOT pass --array_mode arg as this script explicitely passes a single
#        video path
#
#  Submit on cluster with:
#      sbatch slurm_multiGPU_array.sh
# =====================================================================

# ── Paths (EDIT THESE) ──────────────────────────────────────────────
DLC_SIF="/scratch/containers/dlc3_cached.sif"             # Apptainer image
CONFIG="/scratch/path/to/your/config.yaml"                # DLC config
VIDEO_DIR="/scratch/path/to/your/videos"                  # dir with videos to analyze (can be same as project dir or different)
SCRIPTS_DIR="/home/path/to/repository"                    # dir with dlc_analyze.py
VIDEOTYPE=".mp4"                                          # string of comma-separated video extensions, e.g. ".mp4,.avi,.mov"
MANIFEST="/scratch/temp/location/for/video_manifest.txt"  # list of videos to create prior to run
# ────────────────────────────────────────────────────────────────────

set -euo pipefail
mkdir -p logs

echo "Array Job ID : $SLURM_ARRAY_JOB_ID"
echo "Task ID      : $SLURM_ARRAY_TASK_ID"
echo "Node         : $(hostname)"
echo "Start time   : $(date)"
# ───────────────────────────────────────────────────────────────────

module load apptainer 2>/dev/null || true

# Ensure creation of only one list of videos (MANIFEST) to be used by all processes
LOCKFILE="${MANIFEST}.lock"

# Open a dedicated file descriptor for the lock file
exec 9>"$LOCKFILE"

# Block until holding the lock
flock -x 9

# Only the first task that gets the lock will create MANIFEST
if [[ ! -f "$MANIFEST" ]]; then
    echo "Creating manifest..."
    tmp_manifest="${MANIFEST}.tmp.$$"
    find "$VIDEO_DIR" -type f -name "*.mp4" \
        ! -name "*_labeled.mp4" \
        ! -name "*_filtered.mp4" \
        | sort > "$tmp_manifest"
    mv "$tmp_manifest" "$MANIFEST"
fi

# Release the lock
flock -u 9
exec 9>&-

mapfile -t VIDEOS < "$MANIFEST"

i1=$((SLURM_ARRAY_TASK_ID * 2))
i2=$((i1 + 1))

run_one () {
    local video="$1"
    local gpu="$2"

    CUDA_VISIBLE_DEVICES="$gpu" apptainer exec --nv \
        --bind /scratch/user/davcollins:/scratch/user/davcollins \
        --bind /home/remote/davcollins:/home/remote/davcollins \
        "$DLC_SIF" \
        python3 "${SCRIPTS_DIR}/dlc_analyze.py" "$CONFIG" \
            --video_dir "$VIDEO_DIR" \
            --video_path "$video" \
            --shuffle 3 \
            --show_gpu \
	    --filter \
	    --create_video
}

run_one "${VIDEOS[$i1]}" 0 &
if [[ -n "${VIDEOS[$i2]:-}" ]]; then
    run_one "${VIDEOS[$i2]}" 1 &
fi
wait

echo "End time     : $(date)"
