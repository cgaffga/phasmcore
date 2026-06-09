#!/usr/bin/env bash
# ============================================================================
# Re-launch of the matched-detector TRAINING ONLY                2026-06-09
# ============================================================================
# The first full run (run_matched_detector_experiment.sh, 2026-06-08) generated
# the stego fine but BOTH trainings crashed at startup:
#   ValueError: split sum 10000 > 9926   (primary: 74 covers failed capacity)
#   ValueError: split sum  6500 > 5934   (shadow-N2: ~41% failed capacity)
# i.e. the hardcoded test split exceeded the actual stego yield. Data is good;
# this just re-runs training with the test split shrunk to fit (primary 4800,
# n2 800). No re-gen. Appends to the same run.log. ~15-16h on MPS, nohup-safe.
#
# Monitor:  tail -f eval/runs/2026-06-08-matched-detector-exp/run.log
# ============================================================================
set -u
R=/Users/cgaffga/Development/phasm/eval
PY="$R/.venv/bin/python"
LOG="$R/runs/2026-06-08-matched-detector-exp/run.log"
log()   { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$1"/*.jpg 2>/dev/null | wc -l | tr -d ' '; }

log "=================== RELAUNCH (training only, split fixed) ==================="
log "available stego: primary@0.20=$(count "$R/data/stego/phasm_ghost/qf75/payload_020")  shadow-N2=$(count "$R/data/stego/phasm_shadows_bossbase/n2")"

# ---- Exp 1 — matched primary detector (~7-8h) ----
log "Exp 1: train matched SRNet on Phasm-primary@0.20 (4000/1000/4800, 80ep, s42)"
"$PY" "$R/detectors/train_srnet.py" \
    --cover-dir "$R/data/bossbase/jpeg_qf75" \
    --stego-dir "$R/data/stego/phasm_ghost/qf75/payload_020" \
    --payload 0.20 --seed 42 --epochs 80 --batch-size 8 \
    --n-train 4000 --n-val 1000 --n-test 4800 \
    --run-name matched-srnet-phasm-primary-pf020-s042 >>"$LOG" 2>&1
log "  Exp 1 exit=$?"

# ---- Exp 2 — matched shadow detector, N=2 (~7-8h) ----
log "Exp 2: train matched SRNet on Phasm-shadow-N2 (4000/1000/800, 80ep, s42)"
"$PY" "$R/detectors/train_srnet.py" \
    --cover-dir "$R/data/bossbase/jpeg_qf75" \
    --stego-dir "$R/data/stego/phasm_shadows_bossbase/n2" \
    --payload 0.10 --seed 42 --epochs 80 --batch-size 8 \
    --n-train 4000 --n-val 1000 --n-test 800 \
    --run-name matched-srnet-phasm-shadowN2-s042 >>"$LOG" 2>&1
log "  Exp 2 exit=$?"

# ---- summary ----
log "=================== RELAUNCH SUMMARY ==================="
for rn in matched-srnet-phasm-primary-pf020-s042 matched-srnet-phasm-shadowN2-s042; do
    rd=$(ls -d "$R"/runs/*"$rn" 2>/dev/null | tail -1)
    pe=$("$PY" -c "import json;print(round(json.load(open('$rd/results.json'))['test']['pe'],4))" 2>/dev/null || echo ERR)
    log "  $rn -> test PE = $pe   ($rd)"
done
log "Reference: J-UNIWARD native@0.20 PE=0.293 | cross-stego J-UNI->Phasm@0.20 PE=0.245"
log "=================== RELAUNCH DONE ==================="
