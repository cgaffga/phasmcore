#!/bin/bash
# E17 end-to-end pipeline: EfficientNet-B0 pretrained, 100 epochs (vs E15's 40),
# then re-run shadow-N inference to check whether the flat-PE result holds with
# stronger training.
#
# E15 caveats:
#   "EffNet trained for 40 epochs only (vs Yousfi 2020 typically 100+). Native
#    test PE 0.366 vs SRNet's 0.293 — EffNet is weaker overall in this
#    fine-tuning regime, which may partially explain the flat shadow-N curve."
#
# E17 addresses the first half. If 100-epoch EffNet (a) reaches stronger native
# PE AND (b) still shows flat shadow-N, the deniability story strengthens.
#
# Designed to run unattended in background. Estimated wall-clock: ~5h train + 5min eval.

set -uo pipefail
cd "$(dirname "$0")/.."

DATE=$(date -u +%Y-%m-%d)  # UTC to match train_efficientnet.py's run_dir naming
TRAIN_RUN="${DATE}-e17-effnet-juniward-qf75-pf020-s042-ep100"
EVAL_RUN="${DATE}-e17-effnet-shadow-pe-curve-ep100"
LOG="runs/${DATE}-e17-effnet-pipeline.master.log"

mkdir -p runs/"$TRAIN_RUN"

log() {
    echo "[e17 $(date +%H:%M:%S)] $*" | tee -a "$LOG"
}

log "=== E17 EfficientNet-pretrained 100-epoch validation — START ==="
log "training: 100 epochs, batch 8, MPS, seed 42, J-UNI@QF75@0.20"
log "run dir: runs/${TRAIN_RUN}"

.venv/bin/python detectors/train_efficientnet.py \
    --payload 0.2 --quality 75 --seed 42 \
    --epochs 100 --batch-size 8 \
    --run-name "${TRAIN_RUN#${DATE}-}" \
    > "runs/${TRAIN_RUN}/train.log" 2>&1
TRAIN_RC=$?
log "training exited rc=${TRAIN_RC}"

if [ "$TRAIN_RC" -ne 0 ]; then
    log "training FAILED — tail of train.log:"
    tail -20 "runs/${TRAIN_RUN}/train.log" | tee -a "$LOG"
    exit 1
fi

TEST_PE=$(.venv/bin/python -c "
import json
d = json.loads(open('runs/${TRAIN_RUN}/results.json').read())
print(f\"{d['test']['pe']:.4f}\")")
log "training done: native test PE = ${TEST_PE} (100-epoch EffNet on J-UNI@QF75@0.20)"

log "running shadow-N inference..."
.venv/bin/python scripts/run_e15_effnet_shadow_pe.py \
    --checkpoint "runs/${TRAIN_RUN}/checkpoints/best.pt" \
    --out-dir "runs/${EVAL_RUN}" \
    --device mps \
    >> "$LOG" 2>&1
EVAL_RC=$?
log "shadow eval exited rc=${EVAL_RC}"

if [ "$EVAL_RC" -ne 0 ]; then
    log "shadow eval FAILED — see log above"
    exit 1
fi

# Three-way comparison: E10 SRNet vs E15 EffNet-40 vs E17 EffNet-100
log "synthesising 3-way comparison..."
.venv/bin/python <<PY 2>&1 | tee -a "$LOG"
import json
from pathlib import Path

srn = json.loads(Path('runs/2026-05-17-e10-shadow-pe-curve/per_n_pe.json').read_text())
eff40 = json.loads(Path('runs/2026-05-17-e15-effnet-shadow-pe-curve/per_n_pe.json').read_text())
eff100 = json.loads(Path('runs/${EVAL_RUN}/per_n_pe.json').read_text())

print()
print('=== Shadow-N PE: SRNet vs EffNet-40ep vs EffNet-100ep ===')
print(f"  {'N':>2}  {'SRNet':>7}  {'Eff40':>7}  {'Eff100':>7}  {'Δ(100-40)':>10}")
for n in range(5):
    s = srn['per_N'][str(n)]['per_n_max']['PE']
    e40 = eff40['per_N'][str(n)]['per_n_max']['PE']
    e100 = eff100['per_N'][str(n)]['per_n_max']['PE']
    print(f"  {n:>2}  {s:>7.3f}  {e40:>7.3f}  {e100:>7.3f}  {e100-e40:>+10.3f}")

e100 = [eff100['per_N'][str(n)]['per_n_max']['PE'] for n in range(5)]
print()
print(f"EffNet-100 trend: PE_N=0 = {e100[0]:.3f}, PE_N=4 = {e100[4]:.3f}")
if e100[4] > e100[0] + 0.05:
    verdict = "RISING (shadows wash out 100-ep EffNet too — like SRNet)"
elif e100[4] < e100[0] - 0.05:
    verdict = "FALLING (EffNet sees through shadows more clearly at high N)"
else:
    verdict = "FLAT (deniability holds for 100-ep EffNet, like E15)"
print(f"Verdict: {verdict}")

# Write pending_paper_updates.md
import datetime
pending = Path('runs/${EVAL_RUN}/pending_paper_updates.md')
with pending.open('w') as f:
    f.write(f'''# E17 100-epoch EfficientNet validation — RESULTS READY FOR REVIEW

Generated: {datetime.datetime.now().isoformat()}
Pipeline: \`eval/scripts/run_e17_effnet_longer.sh\`

## Headline

100-epoch EffNet (vs E15 40-epoch): shadow-N trend is **{verdict}**.

## Three-way comparison
''')
    f.write(f"| N | SRNet | EffNet-40 | EffNet-100 | Delta(100-40) |\\n|---|---|---|---|---|\\n")
    for n in range(5):
        s = srn['per_N'][str(n)]['per_n_max']['PE']
        e40v = eff40['per_N'][str(n)]['per_n_max']['PE']
        e100v = eff100['per_N'][str(n)]['per_n_max']['PE']
        f.write(f"| {n} | {s:.3f} | {e40v:.3f} | {e100v:.3f} | {e100v-e40v:+.3f} |\\n")
    f.write(f'''
## Native test PE
- SRNet from-scratch: 0.293 (E9)
- EffNet 40-epoch:    0.366 (E15)
- EffNet 100-epoch:   ${TEST_PE} (E17)

## §6.3 paper update
If verdict = FLAT: the flat-PE finding is robust to longer training; the
"shadows are neutral to pretrained-EffNet" claim strengthens.
If verdict = RISING: the wash-out is detector-strength-dependent, not
architecture-dependent; rewrite §6.3 to note the convergence at adequate
training.

## Run artefacts
- training: \`runs/${TRAIN_RUN}/\`
- shadow eval: \`runs/${EVAL_RUN}/\`
- pipeline log: \`runs/${DATE}-e17-effnet-pipeline.master.log\`
''')
print()
print(f"Wrote: {pending}")
PY

log "=== E17 PIPELINE END ==="
