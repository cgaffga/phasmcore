#!/bin/bash
# Deferred-completion handler for the J-UNI@QF75@0.50 SRNet training.
#
# WHAT IT DOES (autonomous, but no commits / no LaTeX edits):
# 1. Polls the train.log for the "FINAL test PE" line (every 60 s).
# 2. Runs cross-stego inference: detector @ 0.50 -> Phasm Ghost @ 0.50.
# 3. Writes a `pending_paper_updates.md` summary with proposed LaTeX +
#    REPRO + figure-script patches the user can review and apply.
#
# WHAT IT DOES NOT DO (intentionally):
# - No git commits. The user reviews the numbers and the proposed cell
#   updates, then applies + commits manually.
# - No LaTeX or REPRO.md edits. Same reason.
# - No figure re-render. The build_paper_figures.py script auto-picks
#   up the new JSONs once they exist; just re-run it manually.
#
# This separation is deliberate: a training collapse (PE ~ 0.5) would
# otherwise auto-publish a misleading paper update.
#
# User explicitly approved this script's creation 2026-05-16.

set -uo pipefail
cd "$(dirname "$0")/.."

EVAL_ROOT="$(pwd)"
TRAIN_LOG="${EVAL_ROOT}/runs/e9-srnet-juniward-qf75-pf050-s042/train.log"
PHASM_STEGO="data/stego/phasm_ghost/qf75/payload_050"
COVER_DIR="data/bossbase/jpeg_qf75"
OUT_DIR="${EVAL_ROOT}/runs/2026-05-16-e9-eval-vs-phasm-pf050"

mkdir -p "$OUT_DIR"
log_file="${OUT_DIR}/finalize.log"
log() { echo "[finalize $(date +%H:%M:%S)] $*" | tee -a "$log_file"; }

log "=== pf050 deferred handler START (compute-only mode) ==="

# === 1. Wait for training to finish ===
log "polling $TRAIN_LOG for 'FINAL test PE'..."
until grep -q "FINAL test PE" "$TRAIN_LOG" 2>/dev/null; do
    sleep 60
done
log "training complete"

# === 2. Read native PE from train.log ===
NATIVE_PE=$(tr '\r' '\n' < "$TRAIN_LOG" | grep "FINAL test PE" | tail -1 | sed 's/.*PE = \([0-9.]*\).*/\1/')
log "QF75@0.50 native PE = $NATIVE_PE"

# === 3. Locate the checkpoint (script auto-creates dated dir) ===
CKPT=$(ls "${EVAL_ROOT}"/runs/*-e9-srnet-juniward-qf75-pf050-s042/checkpoints/best.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    log "ERROR: could not locate pf050 best.pt — aborting"
    cat > "${OUT_DIR}/pending_paper_updates.md" <<EOF
# pf050 deferred handler — PARTIAL (checkpoint missing)

Training reported FINAL test PE = ${NATIVE_PE}, but the checkpoint
\`best.pt\` could not be located on disk. Cross-stego inference was
NOT run.

Manual next step: locate the .pt file under \`runs/*pf050*/checkpoints/\`,
then run:

    .venv/bin/python detectors/cross_eval_pe.py \\
      --checkpoint <path-to-best.pt> \\
      --cover-dir data/bossbase/jpeg_qf75 \\
      --stego-dir data/stego/phasm_ghost/qf75/payload_050 \\
      --out-dir runs/2026-05-16-e9-eval-vs-phasm-pf050 \\
      --device cpu --limit 1000
EOF
    exit 1
fi
log "checkpoint: $CKPT"

# === 4. Cross-stego inference ===
log "running cross-stego inference (cover @ QF75 + Phasm Ghost @ QF75 @ 0.50)..."
.venv/bin/python detectors/cross_eval_pe.py \
    --checkpoint "$CKPT" \
    --cover-dir  "$COVER_DIR" \
    --stego-dir  "$PHASM_STEGO" \
    --out-dir    "$OUT_DIR" \
    --label      "BOSSbase J-UNI@QF75@0.50 -> Phasm Ghost @ 0.50 (cross-stego)" \
    --device cpu --limit 1000 \
    > "${OUT_DIR}/cross_eval.log" 2>&1
PHASM_PE=$(.venv/bin/python -c "import json; print(json.load(open('${OUT_DIR}/results.json'))['PE'])")
log "Phasm @ 0.50 PE = $PHASM_PE"

# === 5. Sanity check on PE (warn if either looks like training collapse) ===
collapse_warn=""
NATIVE_FMT=$(printf "%.3f" "$NATIVE_PE")
PHASM_FMT=$(printf "%.3f" "$PHASM_PE")
if .venv/bin/python -c "import sys; sys.exit(0 if abs(${NATIVE_PE} - 0.5) < 0.02 else 1)"; then
    collapse_warn="${collapse_warn}
- ⚠️  J-UNI native PE = ${NATIVE_FMT} is within 0.02 of chance — possible training collapse"
fi
if .venv/bin/python -c "import sys; sys.exit(0 if abs(${PHASM_PE} - 0.5) < 0.02 else 1)"; then
    collapse_warn="${collapse_warn}
- ⚠️  vs Phasm PE = ${PHASM_FMT} is within 0.02 of chance — possible training collapse"
fi

# === 6. Copy JSONs to figure-script-readable location ===
mkdir -p "${EVAL_ROOT}/runs/2026-05-16-e9-eval-vs-phasm"
cp "${OUT_DIR}/stego_scores.json" "${EVAL_ROOT}/runs/2026-05-16-e9-eval-vs-phasm/phasm_qf75_pf050.json"
cp "${OUT_DIR}/cover_scores.json" "${EVAL_ROOT}/runs/2026-05-16-e9-eval-vs-phasm/cover_qf75_byDet050.json"

# === 7. Write proposed paper updates summary ===
cat > "${OUT_DIR}/pending_paper_updates.md" <<EOF
# pf050 deferred handler — RESULTS READY FOR REVIEW

Generated: $(date)
Handler: \`eval/scripts/wait_pf050_then_eval.sh\`

## Numbers

| Cell | PE |
|---|---|
| §6.2 Table 1 — SRNet J-UNI native @ 0.50 | **${NATIVE_FMT}** |
| §6.2 Table 1 — SRNet J-UNI vs Phasm @ 0.50 | **${PHASM_FMT}** |${collapse_warn}

## Proposed paper updates (manual apply)

### 1. \`marketing/research/paper/sections/6_evaluation.tex\`

In the §6.2 Table 1, replace the two \\TODO placeholders at the 0.50 column:

\`\`\`diff
- SRNet J-UNI native       & 0.449 & 0.410\$^*\$ & 0.293 & 0.204 & 0.148 & \\TODO{queued} \\\\
- SRNet J-UNI vs.\\ Phasm   & \\TODO{TBD} & 0.410\$^*\$ & 0.297 & \\TODO{TBD}    & 0.160 & \\TODO{TBD} \\\\
+ SRNet J-UNI native       & 0.449 & 0.410\$^*\$ & 0.293 & 0.204 & 0.148 & ${NATIVE_FMT} \\\\
+ SRNet J-UNI vs.\\ Phasm   & \\TODO{TBD} & 0.410\$^*\$ & 0.297 & \\TODO{TBD}    & 0.160 & ${PHASM_FMT} \\\\
\`\`\`

### 2. \`eval/scripts/build_paper_figures.py\` (in the monorepo parent repo)

Add the 0.50 data point to the \`juni_native\` dict and extend the cross-stego loop:

\`\`\`python
juni_native = {
    0.05: 0.449,
    0.10: 0.410,
    0.20: 0.293,
    0.30: 0.204,
    0.40: 0.148,
+   0.50: ${NATIVE_FMT},
}
…
- for p in (0.05, 0.20, 0.30):
+ for p in (0.05, 0.20, 0.30, 0.50):
\`\`\`

Then re-render:
\`\`\`bash
cd /Users/cgaffga/Development/phasm/eval
.venv/bin/python scripts/build_paper_figures.py
\`\`\`

### 3. \`marketing/research/paper/anc/REPRO.md\`

Replace the §6.2 0.50 placeholder line:

\`\`\`diff
- | QF75 J-UNI native @ 0.50 | SRNet from-scratch (queued) | 0.50 | \`runs/2026-05-16-e9-srnet-juniward-qf75-pf050-s042/\` | (training, ETA Sat 06:00) |
+ | QF75 J-UNI native @ 0.50 | SRNet from-scratch | 0.50 | \`runs/*-e9-srnet-juniward-qf75-pf050-s042/\` | ${NATIVE_FMT} |
\`\`\`

And append to the §6.2 cross-stego section:

\`\`\`
| QF75 vs Phasm @ 0.50 | \`runs/2026-05-16-e9-eval-vs-phasm-pf050/\` | ${PHASM_FMT} |
\`\`\`

### 4. Commit (review then run)

\`\`\`bash
cd /Users/cgaffga/Development/phasm/marketing
git add research/paper/sections/6_evaluation.tex \\
        research/paper/anc/REPRO.md \\
        research/figures/fig_eval_singlemsg_pe.pdf \\
        research/figures/fig_eval_singlemsg_pe.png
git commit -m "research: §6.2 0.50 cells filled (deferred handler)"

# If build_paper_figures.py itself was edited, commit it in the parent repo:
#   cd /Users/cgaffga/Development/phasm
#   git add eval/scripts/build_paper_figures.py
#   git commit -m "eval: figure script — add 0.50 data point"

cd /Users/cgaffga/Development/phasm
git add eval/runs/2026-05-16-e9-eval-vs-phasm-pf050 \\
        eval/runs/2026-05-16-e9-eval-vs-phasm/phasm_qf75_pf050.json \\
        eval/runs/2026-05-16-e9-eval-vs-phasm/cover_qf75_byDet050.json \\
        eval/runs/e9-srnet-juniward-qf75-pf050-s042/train.log
git commit -m "eval: pf050 + cross-stego complete (deferred handler)"
\`\`\`
EOF

log "wrote ${OUT_DIR}/pending_paper_updates.md"
log "=== pf050 deferred handler DONE ==="
log "REVIEW the numbers above before applying the proposed updates."
