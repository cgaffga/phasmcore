#!/bin/bash
# E15 end-to-end pipeline: train EfficientNet-B0 pretrained on QF75 J-UNI@0.20,
# then run the shadow-N inference, then write a pending_paper_updates.md.
#
# Designed to run unattended in background:
#   nohup bash scripts/run_e15_full_pipeline.sh > runs/2026-05-17-e15-effnet-pipeline.master.log 2>&1 &
#
# Reads no env vars. Estimated wall-clock: ~2 hours train + ~5 min eval.

set -uo pipefail
cd "$(dirname "$0")/.."

DATE=2026-05-17
TRAIN_RUN="${DATE}-e15-effnet-juniward-qf75-pf020-s042"
EVAL_RUN="${DATE}-e15-effnet-shadow-pe-curve"
LOG="runs/${DATE}-e15-effnet-pipeline.master.log"

mkdir -p runs/"$TRAIN_RUN"

log() {
    echo "[e15 $(date +%H:%M:%S)] $*" | tee -a "$LOG"
}

log "=== E15 EfficientNet-pretrained shadow validation — START ==="
log "training: 40 epochs, batch 8, MPS, seed 42, J-UNI@QF75@0.20"
log "run dir: runs/${TRAIN_RUN}"

.venv/bin/python detectors/train_efficientnet.py \
    --payload 0.2 --quality 75 --seed 42 \
    --epochs 40 --batch-size 8 \
    --run-name "${TRAIN_RUN#${DATE}-}" \
    > "runs/${TRAIN_RUN}/train.log" 2>&1
TRAIN_RC=$?
log "training exited rc=${TRAIN_RC}"

if [ "$TRAIN_RC" -ne 0 ]; then
    log "training FAILED — tail of train.log:"
    tail -20 "runs/${TRAIN_RUN}/train.log" | tee -a "$LOG"
    exit 1
fi

# Pull the headline test PE out of results.json
TEST_PE=$(.venv/bin/python -c "
import json
d = json.loads(open('runs/${TRAIN_RUN}/results.json').read())
print(f\"{d['test']['pe']:.4f}\")")
log "training done: native test PE = ${TEST_PE} (on J-UNI@QF75@0.20)"

log "running shadow-N inference (5 N values, MPS)..."
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

# Synthesise headline comparison vs SRNet baseline (E10 per_n_pe.json)
log "synthesising paper-update brief..."

.venv/bin/python <<PY | tee -a "$LOG"
import json
from pathlib import Path

eff = json.loads(Path('runs/${EVAL_RUN}/per_n_pe.json').read_text())
srn = json.loads(Path('runs/2026-05-17-e10-shadow-pe-curve/per_n_pe.json').read_text())

print()
print('=== EffNet (E15) vs SRNet (E10) per-N max comparison ===')
print(f"  {'N':>2}  {'SRN cover':>9}  {'SRN stego':>9}  {'SRN PE':>7}  |  {'EFF cover':>9}  {'EFF stego':>9}  {'EFF PE':>7}  {'delta PE':>9}")
for n in range(5):
    s = srn['per_N'][str(n)]['per_n_max']
    e = eff['per_N'][str(n)]['per_n_max']
    print(f"  {n:>2}  {s['cover_mean']:>9.3f}  {s['stego_mean']:>9.3f}  {s['PE']:>7.3f}  |  "
          f"{e['cover_mean']:>9.3f}  {e['stego_mean']:>9.3f}  {e['PE']:>7.3f}  "
          f"{e['PE']-s['PE']:>+9.3f}")

# Determine washing-out verdict
e_pe = [eff['per_N'][str(n)]['per_n_max']['PE'] for n in range(5)]
trend = 'rising (shadows wash out EffNet too)' if e_pe[4] > e_pe[0] else 'falling (EffNet sees through shadows)'
print()
print(f"EffNet trend: PE_N=0 = {e_pe[0]:.3f}, PE_N=4 = {e_pe[4]:.3f} -> {trend}")

# Write the pending_paper_updates.md
pending = Path('runs/${EVAL_RUN}/pending_paper_updates.md')
with pending.open('w') as f:
    import datetime
    f.write(f'''# E15 EfficientNet-pretrained validation — RESULTS READY FOR REVIEW

Generated: {datetime.datetime.now().isoformat()}
Pipeline: \`eval/scripts/run_e15_full_pipeline.sh\`

## Headline

EfficientNet-B0 (ImageNet pretrained, fine-tuned on QF75 J-UNI@0.20 for
40 epochs) evaluated on BOSSbase QF75 cover + Phasm Ghost shadow stegos
\`data/path3_shadow_eval/shadow_n{{0..4}}/\`.

Native test PE on its own J-UNI@0.20 test set: **${TEST_PE}**.

## Per-N max comparison vs E10 SRNet baseline

| N | SRNet cover_mean | SRNet stego_mean | SRNet PE | EffNet cover_mean | EffNet stego_mean | EffNet PE | delta PE (Eff - SRN) |
|---|---|---|---|---|---|---|---|
''')
    for n in range(5):
        s = srn['per_N'][str(n)]['per_n_max']
        e = eff['per_N'][str(n)]['per_n_max']
        f.write(f"| {n} | {s['cover_mean']:.3f} | {s['stego_mean']:.3f} | {s['PE']:.3f} | {e['cover_mean']:.3f} | {e['stego_mean']:.3f} | {e['PE']:.3f} | {e['PE']-s['PE']:+.3f} |\n")
    f.write(f'''
## Universal-subset (n=40) comparison

| N | SRNet PE | EffNet PE |
|---|---|---|
''')
    for n in range(5):
        sp = srn['per_N'][str(n)]['universal_subset']['PE']
        ep = eff['per_N'][str(n)]['universal_subset']['PE']
        f.write(f"| {n} | {sp:.3f} | {ep:.3f} |\n")
    f.write(f'''
## Trend verdict

EffNet PE N=0 = {e_pe[0]:.3f}, PE N=4 = {e_pe[4]:.3f} -> **{trend}**.

## Files

- training checkpoint: \`runs/${TRAIN_RUN}/checkpoints/best.pt\`
- training history:    \`runs/${TRAIN_RUN}/results.json\`
- shadow eval:         \`runs/${EVAL_RUN}/per_n_pe.json\`
- per-image scores:    \`runs/${EVAL_RUN}/{{cover,shadow_n0..4}}_scores.json\`

## Suggested paper updates

Apply if EffNet trend matches SRNet (washing out): F7 validation = YES;
update §6.3 with a second curve, add column to Table IV, downgrade the
v2-deferral note in §2/§6 to a paragraph saying "we also verify against
the post-SRNet detector lineage (Yousfi 2020) and the trend holds".

If EffNet trend reverses (sees through shadows): scope the §6.3 claim to
"matched-payload from-scratch SRNet" and elevate the limitation note in
§7 — the v2-deferral note now points at a measured *negative* result.
''')

print(f'\\nwrote {pending}')
PY

log "=== E15 pipeline DONE ==="
log "pending paper updates: runs/${EVAL_RUN}/pending_paper_updates.md"
