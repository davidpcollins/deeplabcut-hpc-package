#!/usr/bin/env bash
#
# slurmtail - Lightweight, live, color-coded monitor for SLURM job log files
# Designed for HPC cases where .sh script prints to .out and calls other scripts 
# that print to .err with progress bars and errors. If array jobs are detected,
# it automatically splits tmux panes per sub-job. It also watches for new log files
#
# Usage: slurmtail <JOB_ID> <LOGS_DIR>
#
# Notes:
#   - May need to run chmod +x ./slurmtail.sh to make it executable
#   - If uploading from windows machine, run dos2unix on the script to fix line endings

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────
POLL_INTERVAL=2          # seconds between checks for new files
ERR_POLL_INTERVAL=1      # seconds between .err file progress updates
TRIGGER_WORDS='error|fatal|traceback|exception|failed|abort|segfault|killed|oom|panic'

# ANSI color codes (Dracula theme because I like it)
BLUE='\033[38;2;189;147;249m'   # Dracula purple #BD93F9
RED='\033[38;2;255;85;85m'      # Dracula red    #FF5555
GREEN='\033[38;2;80;250;123m'   # Dracula green  #50FA7B
DIM='\033[2m'
RESET='\033[0m'

# ── Usage / argument parsing ─────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $(basename "$0") <JOB_ID> <LOGS_DIR>

Monitor SLURM job logs in real time with color-coded output.

Arguments:
  JOB_ID     The numeric SLURM job ID
  LOGS_DIR   Path to the directory containing .out/.err log files

Colors:
  Blue   = lines from .out files (stdout)
  Red    = lines from .err files matching trigger words
           (${TRIGGER_WORDS//|/, })
  Green  = all other lines from .err files

Array jobs are auto-detected and displayed in tmux split panes.
Press Ctrl-C to stop monitoring.
EOF
    exit 1
}

[[ $# -ne 2 ]] && usage

JOB_ID="$1"
LOGS_DIR="$2"

if [[ ! -d "$LOGS_DIR" ]]; then
    echo "Error: directory '$LOGS_DIR' does not exist." >&2
    exit 1
fi

# Normalise to absolute path
LOGS_DIR="$(cd "$LOGS_DIR" && pwd)"

# ── Helpers ──────────────────────────────────────────────────────────────────

# Find all log files (out + err) whose name contains the JOB_ID
find_log_files() {
    find "$LOGS_DIR" -maxdepth 1 -type f \( -name "*_${JOB_ID}.out" -o -name "*_${JOB_ID}.err" \
        -o -name "*_${JOB_ID}_*.out" -o -name "*_${JOB_ID}_*.err" \) 2>/dev/null | sort
}

# Extract unique sub-IDs from array-job filenames.
# Array files look like  JOBNAME_<ID>_<subID>.{out,err}
get_sub_ids() {
    find "$LOGS_DIR" -maxdepth 1 -type f \( -name "*_${JOB_ID}_*.out" -o -name "*_${JOB_ID}_*.err" \) \
        2>/dev/null \
        | sed -E "s/.*_${JOB_ID}_([^.]+)\.(out|err)$/\1/" \
        | sort -n -u
}

# ── Single-stream tail monitor ───────────────────────────────────────────────
# Monitors all files for one (sub-)job, color-coding each line.
# Arguments:
#   $1 = JOB_ID
#   $2 = LOGS_DIR
#   $3 = optional sub-ID (empty string for non-array jobs)
monitor_job() {
    local job_id="$1"
    local logs_dir="$2"
    local sub_id="${3:-}"
    local label

    if [[ -n "$sub_id" ]]; then
        label="[${job_id}_${sub_id}]"
    else
        label="[${job_id}]"
    fi

    # Build the file-matching pattern
    local out_pattern err_pattern
    if [[ -n "$sub_id" ]]; then
        out_pattern="*_${job_id}_${sub_id}.out"
        err_pattern="*_${job_id}_${sub_id}.err"
    else
        out_pattern="*_${job_id}.out"
        err_pattern="*_${job_id}.err"
    fi

    echo -e "${DIM}${label} Watching for log files in ${logs_dir} ...${RESET}"

    # Track which files we're already tailing (PIDs keyed by filename)
    declare -A tail_pids

    cleanup() {
        for pid in "${tail_pids[@]}"; do
            kill "$pid" 2>/dev/null || true
        done
        wait 2>/dev/null
    }
    trap cleanup EXIT INT TERM

    # Tail a single file with appropriate coloring
    start_tail() {
        local filepath="$1"
        local fname
        fname="$(basename "$filepath")"

        if [[ "$fname" == *.out ]]; then
            # Blue for stdout — awk handles coloring with no per-line subshell
            stdbuf -oL tail -n +1 -F "$filepath" 2>/dev/null \
                | stdbuf -oL awk -v lbl="$label" \
                    -v blue="$BLUE" -v reset="$RESET" \
                    '{print blue lbl " [out] " $0 reset; fflush()}' &
        else
            # Stderr files often contain \r-delimited progress bars (tqdm)
            # that produce thousands of records per second. We poll the
            # last 4KB of the file on a fixed interval using a single
            # tail -c | tr | tail pipeline per tick.
            (
                local prev_line=""

                while [[ ! -f "$filepath" ]]; do sleep "$ERR_POLL_INTERVAL"; done

                while true; do
                    last_line=$(tail -c 4096 "$filepath" 2>/dev/null | tr '\r' '\n' | grep -av '^[[:space:]]*$' | tail -1)

                    if [[ -n "$last_line" && "$last_line" != "$prev_line" ]]; then
                        if echo "$last_line" | grep -qaiE "$TRIGGER_WORDS"; then
                            printf "\r\033[38;2;255;85;85m%s [err] %s\033[0m\n" "$label" "$last_line"
                        else
                            printf "\r\033[38;2;80;250;123m%s [err] %s\033[0m" "$label" "$last_line"
                        fi
                        prev_line="$last_line"
                    fi

                    sleep "$ERR_POLL_INTERVAL"
                done
            ) &
        fi
        tail_pids["$filepath"]=$!
    }

    # Main watch loop — pick up new files as they appear
    while true; do
        for f in "$logs_dir"/$out_pattern "$logs_dir"/$err_pattern; do
            # skip if glob didn't expand
            [[ -e "$f" ]] || continue
            if [[ -z "${tail_pids[$f]+_}" ]]; then
                echo -e "${DIM}${label} Attaching to $(basename "$f")${RESET}"
                start_tail "$f"
            fi
        done
        sleep "$POLL_INTERVAL"
    done
}

# ── Detect job type ──────────────────────────────────────────────────────────

# Wait briefly for at least one file to appear
echo "Looking for log files matching job ${JOB_ID} in ${LOGS_DIR} ..."

attempts=0
max_attempts=30
while [[ -z "$(find_log_files)" ]]; do
    (( attempts++ ))
    if (( attempts >= max_attempts )); then
        echo "Error: no log files matching job ${JOB_ID} found after ${max_attempts}s." >&2
        exit 1
    fi
    sleep 1
done

SUB_IDS="$(get_sub_ids)"

# ── Single job path ──────────────────────────────────────────────────────────
if [[ -z "$SUB_IDS" ]]; then
    echo "Detected single job ${JOB_ID}. Monitoring logs ..."
    echo "Press Ctrl-C to stop."
    echo ""
    monitor_job "$JOB_ID" "$LOGS_DIR" ""
    exit 0
fi

# ── Array job path — use tmux ────────────────────────────────────────────────
SUB_ARRAY=()
while IFS= read -r sid; do
    SUB_ARRAY+=("$sid")
done <<< "$SUB_IDS"
NUM_SUBS=${#SUB_ARRAY[@]}

echo "Detected array job ${JOB_ID} with ${NUM_SUBS} sub-job(s): ${SUB_ARRAY[*]}"

# Require tmux
if ! command -v tmux &>/dev/null; then
    echo "Warning: tmux not found. Falling back to interleaved single-stream mode." >&2
    echo "Press Ctrl-C to stop."
    echo ""
    for sid in "${SUB_ARRAY[@]}"; do
        monitor_job "$JOB_ID" "$LOGS_DIR" "$sid" &
    done
    # Also watch for new sub-IDs
    (
        declare -A known
        for sid in "${SUB_ARRAY[@]}"; do known["$sid"]=1; done
        while true; do
            sleep "$POLL_INTERVAL"
            while IFS= read -r sid; do
                if [[ -z "${known[$sid]+_}" ]]; then
                    echo -e "${DIM}[watcher] New sub-job detected: ${sid}${RESET}"
                    known["$sid"]=1
                    monitor_job "$JOB_ID" "$LOGS_DIR" "$sid" &
                fi
            done <<< "$(get_sub_ids)"
        done
    ) &
    WATCHER_PID=$!
    trap "kill 0 2>/dev/null" EXIT INT TERM
    wait
    exit 0
fi

SESSION_NAME="slurmtail_${JOB_ID}"
SCRIPT_PATH="$(realpath "$0")"

# If we're already inside the tmux session, just monitor the assigned sub-job
if [[ "${SLURMTAIL_SUBID:-}" != "" ]]; then
    monitor_job "$JOB_ID" "$LOGS_DIR" "$SLURMTAIL_SUBID"
    exit 0
fi

# Kill any existing session with this name
tmux kill-session -t "$SESSION_NAME" 2>/dev/null || true

# Create the session with the first sub-job
echo "Launching tmux session '${SESSION_NAME}' ..."
tmux new-session -d -s "$SESSION_NAME" \
    -e "SLURMTAIL_SUBID=${SUB_ARRAY[0]}" \
    "bash '$SCRIPT_PATH' '$JOB_ID' '$LOGS_DIR'"

# Split panes for remaining sub-jobs
for (( i=1; i<NUM_SUBS; i++ )); do
    tmux split-window -t "$SESSION_NAME" \
        -e "SLURMTAIL_SUBID=${SUB_ARRAY[$i]}" \
        "bash '$SCRIPT_PATH' '$JOB_ID' '$LOGS_DIR'"
    # Re-tile after each split to keep things even
    tmux select-layout -t "$SESSION_NAME" tiled
done

# Launch a hidden watcher pane that detects new sub-IDs and spawns panes
tmux split-window -t "$SESSION_NAME" -e "SLURMTAIL_WATCHER=1" \
    "bash -c '
    KNOWN=\"${SUB_ARRAY[*]}\"
    is_known() { for k in \$KNOWN; do [[ \"\$k\" == \"\$1\" ]] && return 0; done; return 1; }
    while true; do
        sleep ${POLL_INTERVAL}
        for sid in \$(find \"${LOGS_DIR}\" -maxdepth 1 -type f \\( -name \"*_${JOB_ID}_*.out\" -o -name \"*_${JOB_ID}_*.err\" \\) \
            2>/dev/null | sed -E \"s/.*_${JOB_ID}_([^.]+)\\.(out|err)\$/\\1/\" | sort -n -u); do
            if ! is_known \"\$sid\"; then
                KNOWN=\"\$KNOWN \$sid\"
                tmux split-window -t \"${SESSION_NAME}\" \
                    -e \"SLURMTAIL_SUBID=\$sid\" \
                    \"bash \\\"${SCRIPT_PATH}\\\" \\\"${JOB_ID}\\\" \\\"${LOGS_DIR}\\\"\"
                tmux select-layout -t \"${SESSION_NAME}\" tiled
            fi
        done
    done
    '"
# Make the watcher pane tiny
tmux select-layout -t "$SESSION_NAME" tiled
tmux resize-pane -t "$SESSION_NAME:.$(tmux list-panes -t "$SESSION_NAME" -F '#{pane_index}' | tail -1)" -y 1

# Attach
tmux attach-session -t "$SESSION_NAME"