#!/bin/bash
# Generic deferred-completion handler — one per (QF, payload) cell.
#
# Args:
#   --qf {75|95}
#   --payload {0.05|0.10|0.20|0.30|0.40|0.50}
#   --run-name <run-dir name without date prefix>   (e.g. e9-srnet-juniward-qf95-pf010-curriculum-s042)
#   --label    <human description for the summary>
#
# Polls runs/<run-name>/train.log every 60 s for "FINAL test PE".
# When found:
#   1. Reads the native PE from train.log
#   2. Locates the auto-dated checkpoint dir
#   3. Runs cross_eval_pe.py against cover@QF + Phasm Ghost@QF@payload
#   4. Sanity-checks both PEs against chance (warns if abs(PE - 0.5) < 0.02)
#   5. Writes pending_paper_updates.md with the numbers + suggested integration
#
# Does NOT auto-edit LaTeX, REPRO.md, or commit. Same review boundary as
# wait_pf050_then_eval.sh (user explicitly approved this pattern 2026-05-16).

set -uo pipefail
cd "$(dirname "$0")/.."

EVAL_ROOT="$(pwd)"

# --- arg parsing ---
QF=""
PAYLOAD=""
RUN_NAME=""
LABEL=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --qf)        QF="$2"; shift 2 ;;
        --payload)   PAYLOAD="$2"; shift 2 ;;
        --run-name)  RUN_NAME="$2"; shift 2 ;;
        --label)     LABEL="$2"; shift 2 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [ -z "$QF" ] || [ -z "$PAYLOAD" ] || [ -z "$RUN_NAME" ]; then
    echo "usage: $0 --qf {75|95} --payload <decimal> --run-name <run-dir>" >&2
    exit 1
fi

# Derive paths
PAYLOAD_PAD=$(printf "%03d" "$(.venv/bin/python -c "print(int(round($PAYLOAD * 100)))")")
TRAIN_LOG="${EVAL_ROOT}/runs/${RUN_NAME}/train.log"
PHASM_STEGO="data/stego/phasm_ghost/qf${QF}/payload_${PAYLOAD_PAD}"
COVER_DIR="data/bossbase/jpeg_qf${QF}"
OUT_DIR="${EVAL_ROOT}/runs/2026-05-16-e9-eval-vs-phasm-qf${QF}-pf${PAYLOAD_PAD}"

mkdir -p "$OUT_DIR"
log_file="${OUT_DIR}/finalize.log"
log() { echo "[finalize qf${QF}@${PAYLOAD} $(date +%H:%M:%S)] $*" | tee -a "$log_file"; }

log "=== deferred handler START ==="
log "  qf=${QF}  payload=${PAYLOAD}  run-name=${RUN_NAME}"
log "  cover_dir=${COVER_DIR}"
log "  stego_dir=${PHASM_STEGO}"

# Verify cover/stego dirs exist
if [ ! -d "${EVAL_ROOT}/${COVER_DIR}" ]; then
    log "ERROR: cover dir missing: ${COVER_DIR} — aborting"
    exit 1
fi
if [ ! -d "${EVAL_ROOT}/${PHASM_STEGO}" ]; then
    log "ERROR: stego dir missing: ${PHASM_STEGO} — aborting"
    exit 1
fi

# === 1. Wait for training to finish ===
log "polling $TRAIN_LOG for 'FINAL test PE'..."
until grep -q "FINAL test PE" "$TRAIN_LOG" 2>/dev/null; do
    sleep 60
done
log "training complete"

# === 2. Read native PE ===
NATIVE_PE=$(tr '\r' '\n' < "$TRAIN_LOG" | grep "FINAL test PE" | tail -1 | sed 's/.*PE = \([0-9.]*\).*/\1/')
log "native PE = $NATIVE_PE"

# === 3. Locate the checkpoint ===
CKPT=$(ls "${EVAL_ROOT}"/runs/*-${RUN_NAME}/checkpoints/best.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ] || [ ! -f "$CKPT" ]; then
    log "ERROR: could not locate ${RUN_NAME} best.pt — partial summary"
    cat > "${OUT_DIR}/pending_paper_updates.md" <<EOF
# qf${QF}@${PAYLOAD} deferred handler — PARTIAL (checkpoint missing)

Training reported FINAL test PE = ${NATIVE_PE}, but the checkpoint
\`best.pt\` could not be located on disk.

Manual next step: locate the .pt file under \`runs/*${RUN_NAME}/checkpoints/\`,
then run cross_eval_pe.py manually.
EOF
    exit 1
fi
log "checkpoint: $CKPT"

# === 4. Cross-stego inference ===
log "running cross-stego inference (cover @ QF${QF} + Phasm Ghost @ QF${QF} @ ${PAYLOAD})..."
.venv/bin/python detectors/cross_eval_pe.py \
    --checkpoint "$CKPT" \
    --cover-dir  "$COVER_DIR" \
    --stego-dir  "$PHASM_STEGO" \
    --out-dir    "$OUT_DIR" \
    --label      "${LABEL:-J-UNI@QF${QF}@${PAYLOAD} -> Phasm Ghost @ QF${QF} @ ${PAYLOAD} (cross-stego)}" \
    --device cpu --limit 1000 \
    > "${OUT_DIR}/cross_eval.log" 2>&1
PHASM_PE=$(.venv/bin/python -c "import json; print(json.load(open('${OUT_DIR}/results.json'))['PE'])")
log "Phasm PE = $PHASM_PE"

# === 5. Sanity check ===
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

# === 6. Write summary ===
cat > "${OUT_DIR}/pending_paper_updates.md" <<EOF
# qf${QF}@${PAYLOAD} deferred handler — RESULTS READY FOR REVIEW

Generated: $(date)
Handler: \`eval/scripts/wait_then_eval.sh --qf ${QF} --payload ${PAYLOAD} --run-name ${RUN_NAME}\`
Run dir: $(ls -d "${EVAL_ROOT}"/runs/*-${RUN_NAME} 2>/dev/null | head -1)
Cross-stego JSON: \`${OUT_DIR}/results.json\`

## Numbers

| Cell | PE |
|---|---|
| QF${QF} J-UNI native @ ${PAYLOAD} | **${NATIVE_FMT}** |
| QF${QF} J-UNI vs Phasm @ ${PAYLOAD} (cross-stego) | **${PHASM_FMT}** |${collapse_warn}

## Suggested integration

These QF${QF} cells should land in either:
  - **Option A**: Extend §6.2 Table 1 with two new QF${QF} rows (mirror the QF75 rows)
  - **Option B**: Move all QF${QF} cells to §6.7 (cross-QF generalisation), since they are the native-QF complement to the E11 cross-QF transfer test
  - **Option C**: Add a new subsection §6.2.X "QF95 reference" with a small table of the QF${QF} payload sweep

The Phasm-eval cell (${PHASM_FMT}) is the more interesting number for
the deniability narrative — it directly compares Phasm Ghost detection
to the matched J-UNI baseline at the corresponding QF.

## Cover image-set used

  cover_dir: ${COVER_DIR}  ($(ls "${EVAL_ROOT}/${COVER_DIR}" | wc -l) JPEGs)
  stego_dir: ${PHASM_STEGO}  ($(ls "${EVAL_ROOT}/${PHASM_STEGO}" | wc -l) JPEGs)

## Manual commit (after applying the integration)

\`\`\`bash
cd /Users/cgaffga/Development/phasm
git add eval/runs/2026-05-16-e9-eval-vs-phasm-qf${QF}-pf${PAYLOAD_PAD} \\
        eval/runs/${RUN_NAME}/train.log
git commit -m "eval: qf${QF}@${PAYLOAD} training + cross-stego complete"

# Then update LaTeX/REPRO/figures in the marketing repo as appropriate.
\`\`\`
EOF

log "wrote ${OUT_DIR}/pending_paper_updates.md"
log "=== handler DONE — REVIEW numbers above before applying ==="
