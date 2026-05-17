#!/bin/bash
# E9 — Sequential SRNet training queue for J-UNI reference detectors.
#
# Waits for the current SRNet training to finish, then launches the next one
# in priority order. Each entry: "quality payload". Trainings run one at a
# time on MPS (sequential, not parallel).
#
# Priority: §6.2 headline is on BOSSbase QF75, so QF75 detectors first
# (ascending payload — low payload = hardest detection task). QF95 detectors
# second (for E11 cross-QF).

set -uo pipefail
cd "$(dirname "$0")/.."

LOG_FILE="runs/2026-05-14-e9-srnet-queue/queue.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[queue $(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

# Wait for any currently-running SRNet training to finish.
wait_for_srnet_idle() {
    while pgrep -f 'train_srnet.py' > /dev/null; do
        sleep 30
    done
}

# Wait for stegos to finish generating for a given (quality, payload).
wait_for_stegos() {
    local q="$1"
    local p="$2"
    local p_pad
    p_pad=$(printf "%03d" "$(echo "$p * 100" | bc -l | awk '{print int($1+0.5)}')")
    local d="data/stego/juniward/qf${q}/payload_${p_pad}"
    log "waiting for $d (target 10000 files)..."
    while true; do
        local n
        n=$(ls "$d" 2>/dev/null | wc -l | awk '{print $1}')
        if [ "$n" -ge 9500 ]; then  # accept 95%+ — some covers fail capacity
            log "  $d has $n files, proceeding"
            return 0
        fi
        sleep 60
    done
}

# Train one SRNet detector on cover vs J-UNI at (quality, payload).
train_one() {
    local q="$1"
    local p="$2"
    local p_pad
    p_pad=$(printf "%03d" "$(echo "$p * 100" | bc -l | awk '{print int($1+0.5)}')")
    local run_name="e9-srnet-juniward-qf${q}-pf${p_pad}-s042"
    local run_dir="runs/${run_name}"
    mkdir -p "$run_dir"
    log "launching SRNet train: qf${q} payload ${p}"
    .venv/bin/python detectors/train_srnet.py \
        --payload "${p}" \
        --quality "${q}" \
        --seed 42 \
        --run-name "${run_name}" \
        > "${run_dir}/train.log" 2>&1
    log "SRNet train done: qf${q} payload ${p} (exit $?)"
}

# Priority queue. Each line: "quality payload"
QUEUE=(
    "75 0.05"
    "75 0.20"
    "75 0.30"
    "75 0.50"
    "95 0.20"
    "95 0.40"
)

log "=== queue START — ${#QUEUE[@]} trainings ahead ==="
log "current SRNet PID(s): $(pgrep -f 'train_srnet.py' | tr '\n' ' ')"

# Wait for the currently-running SRNet (J-UNI@QF95@0.10) to finish first.
log "waiting for current SRNet training to finish..."
wait_for_srnet_idle
log "GPU idle — starting queue"

for entry in "${QUEUE[@]}"; do
    read -r q p <<< "$entry"
    wait_for_stegos "$q" "$p"
    train_one "$q" "$p"
done

log "=== queue DONE ==="
