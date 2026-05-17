#!/bin/bash
# E12 — Cross-source generalization driver.
#
# BOSSbase QF75 J-UNI-trained SRNet evaluated on ALASKA-2 stegos
# (color → Y via run_trained_srnet's Pillow convert("L") path).
#
# Tests two payload-matched cells:
#   1. detector @ pf=0.20 → ALASKA Phasm Ghost @ 100% capacity (~0.19 bpnzAC)
#   2. detector @ pf=0.40 → ALASKA J-UNIWARD @ 0.40 bpnzAC
#
# Cell 2 is the cross-source control: same detector, same nominal payload,
# but ALASKA covers instead of BOSSbase. Comparing the two gives the
# "cross-source degradation" number reviewers expect.
#
# Reverse direction (ALASKA-trained → BOSSbase) is NOT in this script
# because we do not yet have an ALASKA-2-native SRNet checkpoint;
# training one would queue behind the ongoing E9 run.
#
# Output: runs/<date>-e12-cross-source/<cell>/results.json

set -uo pipefail
cd "$(dirname "$0")/.."

DATE=$(date +%Y-%m-%d)
OUT_BASE="runs/${DATE}-e12-cross-source"
mkdir -p "$OUT_BASE"

CKPT_pf020="runs/2026-05-15-e9-srnet-juniward-qf75-pf020-s042/checkpoints/best.pt"
CKPT_pf040="runs/2026-05-10-srnet-juniward-pf040-s042/checkpoints/runs/2026-05-10-srnet-juniward-pf040-s042/checkpoints/best.pt"

ALASKA_COVER="data/alaska2/cover"
ALASKA_PHASM_100="data/path2_alaska2_eval/phasm_ghost_100"
ALASKA_JUNI_040="data/path2_alaska2_eval/juniward_040"

log() { echo "[e12 $(date +%H:%M:%S)] $*" | tee -a "$OUT_BASE/master.log"; }

log "=== E12 Cross-source generalization START ==="

if [ ! -f "$CKPT_pf020" ]; then
    log "ERROR: pf020 checkpoint missing — decompress runs first"
    exit 1
fi
if [ ! -f "$CKPT_pf040" ]; then
    log "ERROR: pf040 checkpoint missing — decompress runs first"
    exit 1
fi

# Cell 1: BOSSbase J-UNI@QF75@0.20 detector → ALASKA Phasm Ghost @ ~0.19 bpnzAC
log "Cell 1: BOSSbase J-UNI@QF75@0.20 -> ALASKA Phasm Ghost@100% capacity"
.venv/bin/python detectors/cross_eval_pe.py \
    --checkpoint "$CKPT_pf020" \
    --cover-dir "$ALASKA_COVER" \
    --stego-dir "$ALASKA_PHASM_100" \
    --out-dir "$OUT_BASE/bossbase_pf020_to_alaska_phasm" \
    --label "BOSSbase J-UNI@QF75@0.20 -> ALASKA Phasm Ghost @ 100% capacity (~0.19 bpnzAC)" \
    --device cpu --limit 484

# Cell 2: BOSSbase J-UNI@QF75@0.40 detector → ALASKA J-UNI @ 0.40 bpnzAC
log "Cell 2: BOSSbase J-UNI@QF75@0.40 -> ALASKA J-UNI@0.40 (control)"
.venv/bin/python detectors/cross_eval_pe.py \
    --checkpoint "$CKPT_pf040" \
    --cover-dir "$ALASKA_COVER" \
    --stego-dir "$ALASKA_JUNI_040" \
    --out-dir "$OUT_BASE/bossbase_pf040_to_alaska_juni" \
    --label "BOSSbase J-UNI@QF75@0.40 -> ALASKA J-UNI @ 0.40 bpnzAC (cross-source control)" \
    --device cpu --limit 484

log "=== E12 DONE ==="

# Summary
.venv/bin/python -c "
import json
from pathlib import Path
out = Path('$OUT_BASE')
print('=== E12 cross-source summary ===')
for d in sorted(out.glob('*/results.json')):
    r = json.loads(d.read_text())
    print(f\"  {r['label']}\")
    print(f\"     PE={r['PE']:.4f}  AUC={r['AUC']:.4f}  n_cover={r['n_cover']} n_stego={r['n_stego']}\")
"
