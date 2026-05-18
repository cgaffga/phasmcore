#!/bin/bash
# Chain: E16 (ALASKA-finetune reverse cross-source) -> E17 (100-epoch EffNet).
# Total expected wall-clock: ~3h (E16) + ~5h (E17) ≈ 8h.
#
# Launch:
#   nohup bash scripts/run_e16_then_e17.sh > runs/$(date +%Y-%m-%d)-e16-e17-chain.master.log 2>&1 &
#
# Each sub-pipeline is fault-tolerant: if E16 fails, E17 still runs (independent
# GPU consumer). The wrapper does not short-circuit on E16 failure so that the
# longer EffNet result isn't blocked by an ALASKA prep issue.

set -uo pipefail
cd "$(dirname "$0")/.."

DATE=$(date +%Y-%m-%d)
CHAIN_LOG="runs/${DATE}-e16-e17-chain.master.log"

log() {
    echo "[chain $(date +%H:%M:%S)] $*" | tee -a "$CHAIN_LOG"
}

log "=== E16 -> E17 CHAIN START ==="

log "Stage 1: E16 ALASKA reverse cross-source pipeline"
bash scripts/run_e16_alaska_reverse_pipeline.sh
E16_RC=$?
log "E16 exited rc=$E16_RC"

log "Stage 2: E17 100-epoch EffNet pipeline (runs regardless of E16 outcome)"
bash scripts/run_e17_effnet_longer.sh
E17_RC=$?
log "E17 exited rc=$E17_RC"

log "=== CHAIN END: E16 rc=$E16_RC, E17 rc=$E17_RC ==="
exit 0
