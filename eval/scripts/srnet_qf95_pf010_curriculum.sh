#!/bin/bash
# Curriculum retraining of SRNet J-UNI@QF95@0.10.
#
# The from-scratch run collapsed to chance (PE 0.4999). Use the QF95@0.40
# from-scratch checkpoint (when ready) to curriculum-init a fine-tune at
# payload 0.10. --init-from halves the LR for fine adjustments.
#
# Strategy:
# 1. Wait for the main queue to fully drain (the QF95@0.40 SRNet training
#    is the LAST entry in the main queue at scripts/srnet_juniward_queue.sh).
# 2. Once that finishes, locate the best.pt checkpoint by glob.
# 3. Launch a curriculum retraining at QF95@0.10 from that checkpoint.

set -uo pipefail
cd "$(dirname "$0")/.."

LOG_FILE="runs/2026-05-14-e9-srnet-queue/qf95_pf010_curriculum.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[curric $(date +%H:%M:%S)] $*" | tee -a "$LOG_FILE"
}

wait_for_main_queue_done() {
    local marker='=== queue DONE ==='
    local queue_log="runs/2026-05-14-e9-srnet-queue/queue.log"
    log "waiting for main queue to drain..."
    while ! grep -qF "$marker" "$queue_log" 2>/dev/null; do
        sleep 60
    done
    log "main queue drained"
}

find_qf95_pf040_checkpoint() {
    # Look for the most recent run dir matching the QF95@0.40 run name.
    local glob='runs/*e9-srnet-juniward-qf95-pf040-s042*/checkpoints/best.pt'
    local found
    found=$(ls -t $glob 2>/dev/null | head -1)
    if [ -z "$found" ] || [ ! -f "$found" ]; then
        return 1
    fi
    echo "$found"
}

log "=== QF95@0.10 curriculum retraining script START ==="
wait_for_main_queue_done

log "searching for QF95@0.40 best.pt..."
INIT_PATH=""
for i in $(seq 1 30); do
    INIT_PATH=$(find_qf95_pf040_checkpoint || true)
    if [ -n "$INIT_PATH" ]; then
        log "found init checkpoint: $INIT_PATH"
        break
    fi
    sleep 60
done

if [ -z "$INIT_PATH" ]; then
    log "ERROR: no QF95@0.40 checkpoint found after 30 min — aborting"
    exit 1
fi

# Also wait for any train_srnet.py still running (sanity).
while pgrep -f 'train_srnet.py' > /dev/null; do
    sleep 30
done
log "GPU confirmed idle, launching curriculum retraining"

RUN_NAME="e9-srnet-juniward-qf95-pf010-curriculum-s042"
RUN_DIR="runs/${RUN_NAME}"
mkdir -p "$RUN_DIR"

.venv/bin/python detectors/train_srnet.py \
    --payload 0.10 \
    --quality 95 \
    --seed 42 \
    --run-name "${RUN_NAME}" \
    --init-from "${INIT_PATH}" \
    > "${RUN_DIR}/train.log" 2>&1
log "curriculum retraining done (exit $?)"
log "=== DONE ==="
