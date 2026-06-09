#!/usr/bin/env bash
# ============================================================================
# Matched ("informed white-box") detector experiment            2026-06-08
# ============================================================================
# Answers the reviewer question (Bas): "Is Phasm detectable by a detector
# trained directly ON Phasm?" — fills the gap that §6.2/§6.3 used only
# CROSS-STEGO detectors (trained on J-UNIWARD, transferred to Phasm).
#
# Threat model: Phasm is open-source (GPL), so an attacker can generate
# unlimited Phasm cover/stego training data. A detector trained on Phasm is
# the strongest realistic steganalyst and is INSIDE the paper's own white-box
# threat model ("the white-box adversary can run Phasm's source as an oracle").
#
#   Exp 1  matched SRNet on  cover vs Phasm-primary @0.20
#          -> compare to J-UNI native@0.20 (PE 0.293) and cross-stego (0.245).
#          Expected ~0.29: "the primary is exactly as detectable as J-UNIWARD,
#          even for a matched attacker" (the primary IS J-UNIWARD+STC).
#   Exp 2  matched SRNet on  cover vs Phasm-(primary + 2 shadows)
#          -> "are shadow-bearing Phasm images detectable by a matched CNN?"
#          Honestly bounds the §6.3 wash-out (a cross-stego artifact).
#
# Identical SRNet config to the J-UNIWARD@0.20 baseline (80 epochs, batch 8,
# lr 2e-4, seed 42, 512x512 grayscale) so the ONLY variable is the training
# stego type. ~15-16h on MPS. Unattended / nohup-safe.
#
# Monitor:  tail -f eval/runs/2026-06-08-matched-detector-exp/run.log
# ============================================================================
set -u
R=/Users/cgaffga/Development/phasm/eval
PY="$R/.venv/bin/python"
EXP="$R/runs/2026-06-08-matched-detector-exp"
mkdir -p "$EXP"
LOG="$EXP/run.log"
log()   { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
count() { ls "$1"/*.jpg 2>/dev/null | wc -l | tr -d ' '; }

log "================ MATCHED-DETECTOR EXPERIMENT START ================"
log "phasm bin: $(ls "$R"/bin/phasm-* 2>/dev/null | head -1)"

# ---- STEP 1: generate Phasm stego (idempotent; measured yield primary ~99%
#      = 9926/10000, shadow-N2 ~59% = 5934/10000). Test split sized to fit. ----
log "STEP 1a: generate Phasm primary @0.20 (full BOSSbase)"
"$PY" "$R/prep/generate_phasm_ghost.py" --payload 0.20 --workers 6 >>"$LOG" 2>&1
log "  primary@0.20 stego now: $(count "$R/data/stego/phasm_ghost/qf75/payload_020")"

log "STEP 1b: generate Phasm shadow N=2 (full BOSSbase)"
"$PY" "$R/prep/generate_phasm_shadows.py" \
    --cover-dir "$R/data/bossbase/jpeg_qf75" \
    --out-dir "$R/data/stego/phasm_shadows_bossbase" \
    --n-list 2 --workers 6 >>"$LOG" 2>&1
log "  shadow-N2 stego now: $(count "$R/data/stego/phasm_shadows_bossbase/n2")"

# ---- STEP 2: Exp 1 — matched primary detector (~7-8h) ----
log "STEP 2: train matched SRNet on Phasm-primary@0.20 (4000/1000/4800, 80ep, s42)"
"$PY" "$R/detectors/train_srnet.py" \
    --cover-dir "$R/data/bossbase/jpeg_qf75" \
    --stego-dir "$R/data/stego/phasm_ghost/qf75/payload_020" \
    --payload 0.20 --seed 42 --epochs 80 --batch-size 8 \
    --n-train 4000 --n-val 1000 --n-test 4800 \
    --run-name matched-srnet-phasm-primary-pf020-s042 >>"$LOG" 2>&1
log "  Exp 1 train exit=$?"

# ---- STEP 3: Exp 2 — matched shadow detector, N=2 (~7-8h) ----
log "STEP 3: train matched SRNet on Phasm-shadow-N2 (4000/1000/800, 80ep, s42)"
"$PY" "$R/detectors/train_srnet.py" \
    --cover-dir "$R/data/bossbase/jpeg_qf75" \
    --stego-dir "$R/data/stego/phasm_shadows_bossbase/n2" \
    --payload 0.10 --seed 42 --epochs 80 --batch-size 8 \
    --n-train 4000 --n-val 1000 --n-test 800 \
    --run-name matched-srnet-phasm-shadowN2-s042 >>"$LOG" 2>&1
log "  Exp 2 train exit=$?"

# ---- STEP 4: summary ----
log "================ SUMMARY ================"
for rn in matched-srnet-phasm-primary-pf020-s042 matched-srnet-phasm-shadowN2-s042; do
    rd=$(ls -d "$R"/runs/*"$rn" 2>/dev/null | tail -1)
    pe=$("$PY" -c "import json;print(round(json.load(open('$rd/results.json'))['test']['pe'],4))" 2>/dev/null || echo ERR)
    log "  $rn -> test PE = $pe   ($rd)"
done
log "Reference: J-UNIWARD native@0.20 PE=0.293 | cross-stego J-UNI->Phasm@0.20 PE=0.245"
log "================ MATCHED-DETECTOR EXPERIMENT DONE ================"
