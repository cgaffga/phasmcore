#!/bin/bash
# E16 — ALASKA-trained → BOSSbase reverse cross-source.
#
# E12 was BOSSbase J-UNI@0.20 detector → ALASKA Phasm Ghost (forward direction).
# E16 is the reverse: ALASKA J-UNI@0.20 detector → BOSSbase Phasm Ghost.
#
# Uses 10k ALASKA-2 covers (downloaded fresh via Kaggle API), re-encoded to QF75
# grayscale to match the §6.2 BOSSbase protocol, with from-scratch SRNet
# training at E9-comparable scale (4000 train / 1000 val / 5000 test, 80 epochs).
#
# Pipeline:
#   1. Prep ALASKA: re-encode 10k covers @ QF75 grayscale, generate J-UNI@0.20
#   2. Train: SRNet from-scratch, 80 epochs on ALASKA
#   3. Cross-eval cells:
#      A. Self-test: ALASKA-trained → ALASKA J-UNI test split (sanity / native PE)
#      B. Cross-source J-UNI control: ALASKA-trained → BOSSbase J-UNI@0.20
#      C. Cross-source Phasm headline: ALASKA-trained → BOSSbase Phasm Ghost@~0.19
#
# Compare to E12 (BOSSbase-trained → ALASKA) for symmetric-vs-asymmetric story.
#
# Output: runs/<date>-e16-alaska-reverse/

set -uo pipefail
cd "$(dirname "$0")/.."

DATE=$(date -u +%Y-%m-%d)  # UTC to match train_srnet.py's run_dir naming
RUN_NAME="e16-srnet-alaska-finetune-qf75-pf020-s042"
TRAIN_DIR="runs/${DATE}-${RUN_NAME}"
EVAL_DIR="runs/${DATE}-e16-alaska-reverse"
MASTER_LOG="${EVAL_DIR}/master.log"

mkdir -p "$EVAL_DIR"

log() { echo "[e16 $(date +%H:%M:%S)] $*" | tee -a "$MASTER_LOG"; }

log "=== E16 ALASKA-finetune cross-source START ==="
log "Train dir: $TRAIN_DIR"
log "Eval dir: $EVAL_DIR"

# ─── Stage 1: Prep ALASKA covers + J-UNI stegos ───────────────────────────
log "Stage 1: Prep ALASKA (re-encode QF75 + J-UNI@0.20 gen)"
log "  source: data/alaska2_full/Cover (downloaded fresh via prep/download_alaska_covers.py)"
.venv/bin/python prep/prepare_alaska_qf75.py \
    --src-covers data/alaska2_full/Cover \
    --cover-out data/alaska2/jpeg_qf75_full \
    --stego-out data/stego/juniward_alaska_full/qf75/payload_020 \
    --limit 12000 \
    2>&1 | tee -a "$MASTER_LOG"
if [ ! -d data/alaska2/jpeg_qf75_full ] || [ ! -d data/stego/juniward_alaska_full/qf75/payload_020 ]; then
    log "ERROR: Stage 1 outputs missing — abort"
    exit 1
fi
N_COVERS=$(ls data/alaska2/jpeg_qf75_full/ | wc -l | tr -d ' ')
N_STEGOS=$(ls data/stego/juniward_alaska_full/qf75/payload_020/ | wc -l | tr -d ' ')
log "  Prepared $N_COVERS ALASKA covers, $N_STEGOS J-UNI@0.20 stegos"

# Sanity: need enough covers for the planned 4k/1k/5k split. Fail loudly if not.
MIN_COVERS=9500
if [ "$N_COVERS" -lt "$MIN_COVERS" ]; then
    log "ERROR: only $N_COVERS covers (< $MIN_COVERS required for 4k/1k/5k split) — abort"
    log "  Re-run prep/download_alaska_covers.py to top up, then re-launch this pipeline"
    exit 1
fi

# ─── Stage 2: Train SRNet on ALASKA from-scratch ──────────────────────────
log "Stage 2: Train SRNet from-scratch on ALASKA J-UNI@0.20"
.venv/bin/python detectors/train_srnet.py \
    --payload 0.20 \
    --quality 75 \
    --seed 42 \
    --epochs 80 \
    --batch-size 8 \
    --n-train 4000 \
    --n-val 1000 \
    --n-test 5000 \
    --cover-dir data/alaska2/jpeg_qf75_full \
    --stego-dir data/stego/juniward_alaska_full/qf75/payload_020 \
    --run-name "$RUN_NAME" \
    2>&1 | tee -a "$MASTER_LOG"

TRAINED_CKPT="${TRAIN_DIR}/checkpoints/best.pt"
if [ ! -f "$TRAINED_CKPT" ]; then
    log "ERROR: Training produced no best.pt — abort"
    exit 1
fi
log "  Trained ALASKA detector: $TRAINED_CKPT"

# ─── Stage 3: Cross-eval ──────────────────────────────────────────────────
DEVICE=$(.venv/bin/python -c "import torch; print('mps' if torch.backends.mps.is_available() else 'cpu')")
log "Stage 3: Cross-eval on device=$DEVICE"

# Cell A: self-test (sanity — should reproduce final test PE from training)
log "Cell A: ALASKA-finetuned → ALASKA J-UNI@0.20 (self-test)"
.venv/bin/python detectors/cross_eval_pe.py \
    --checkpoint "$TRAINED_CKPT" \
    --cover-dir data/alaska2/jpeg_qf75 \
    --stego-dir data/stego/juniward_alaska/qf75/payload_020 \
    --out-dir "${EVAL_DIR}/A_self_test_alaska_juni" \
    --label "ALASKA-finetuned -> ALASKA J-UNI@0.20 (self-test)" \
    --device "$DEVICE" --limit 100 2>&1 | tee -a "$MASTER_LOG"

# Cell B: cross-source J-UNI control
log "Cell B: ALASKA-finetuned → BOSSbase J-UNI@0.20 (cross-source control)"
.venv/bin/python detectors/cross_eval_pe.py \
    --checkpoint "$TRAINED_CKPT" \
    --cover-dir data/bossbase/jpeg_qf75 \
    --stego-dir data/stego/juniward/qf75/payload_020 \
    --out-dir "${EVAL_DIR}/B_alaska_to_bossbase_juni" \
    --label "ALASKA-finetuned -> BOSSbase J-UNI@0.20 (cross-source control)" \
    --device "$DEVICE" --limit 1000 2>&1 | tee -a "$MASTER_LOG"

# Cell C: cross-source Phasm headline (the §6.7 reverse-direction test)
log "Cell C: ALASKA-finetuned → BOSSbase Phasm Ghost@~0.19 (REVERSE HEADLINE)"
.venv/bin/python detectors/cross_eval_pe.py \
    --checkpoint "$TRAINED_CKPT" \
    --cover-dir data/bossbase/jpeg_qf75 \
    --stego-dir data/stego/phasm_ghost/qf75/payload_020 \
    --out-dir "${EVAL_DIR}/C_alaska_to_bossbase_phasm" \
    --label "ALASKA-finetuned -> BOSSbase Phasm Ghost@~0.19 (reverse cross-source)" \
    --device "$DEVICE" --limit 1000 2>&1 | tee -a "$MASTER_LOG"

log "=== E16 DONE — summary ==="
.venv/bin/python <<PY 2>&1 | tee -a "$MASTER_LOG"
import json
from pathlib import Path
out = Path("$EVAL_DIR")
print("=== E16 ALASKA-finetuned reverse cross-source summary ===")
for d in sorted(out.glob("*/results.json")):
    r = json.loads(d.read_text())
    print(f"  {r['label']}")
    print(f"     PE={r['PE']:.4f}  AUC={r['AUC']:.4f}  n_cover={r['n_cover']} n_stego={r['n_stego']}")
PY

log "=== E16 PIPELINE END ==="
