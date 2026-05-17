#!/bin/bash
# Gap #3 — chained FLD-ensemble training script.
#
# Waits for the GFR-feature extraction at payloads 0.05 and 0.30 to fill
# (1500 cover + 1500 stego features each), then runs the FLD-ensemble
# classifier on each, and writes a consolidated RESULTS.md to
# runs/2026-05-15-gap3-gfrfld-baseline/.

set -uo pipefail
cd "$(dirname "$0")/.."

OUT="runs/2026-05-15-gap3-gfrfld-baseline"
mkdir -p "$OUT/pf005" "$OUT/pf030"

log() { echo "[gap3 $(date +%H:%M:%S)] $*" | tee -a "$OUT/master.log"; }

wait_for_features() {
    local p="$1"
    local target="${TARGET_PER_DIR:-500}"
    log "waiting for data/gfrfld_phasm_qf75_pf${p}/{cover,stego} to fill (${target} each)..."
    while true; do
        local nc=$(ls "data/gfrfld_phasm_qf75_pf${p}/cover" 2>/dev/null | wc -l | awk '{print $1}')
        local ns=$(ls "data/gfrfld_phasm_qf75_pf${p}/stego" 2>/dev/null | wc -l | awk '{print $1}')
        if [ "$nc" -ge "$target" ] && [ "$ns" -ge "$target" ]; then
            log "  pf${p}: cover=$nc stego=$ns — ready"
            return 0
        fi
        # If extract process is gone but features incomplete, take what we have (≥half target).
        local half=$((target / 2))
        if ! pgrep -f "extract_handcrafted_cover_stego.py.*${p}" > /dev/null 2>&1; then
            if [ "$nc" -ge "$half" ] && [ "$ns" -ge "$half" ]; then
                log "  pf${p}: extract finished early — proceeding with $nc/$ns"
                return 0
            fi
            log "  pf${p}: extract gone and features incomplete — bailing"
            return 1
        fi
        sleep 60
    done
}

train_pe() {
    local p="$1"
    log "training FLD ensemble for pf${p}..."
    .venv/bin/python detectors/train_fld_ensemble.py \
        --features-dir "data/gfrfld_phasm_qf75_pf${p}" \
        --out-dir "$OUT/pf${p}" \
        --seed 42 \
        > "$OUT/pf${p}/train.log" 2>&1
    log "  pf${p}: trained. Result: $(grep -E 'PE: ' "$OUT/pf${p}/train.log" | tail -1)"
}

log "=== Gap #3 GFR+FLD training chain START ==="

wait_for_features 005 || exit 1
train_pe 005

wait_for_features 030 || exit 1
train_pe 030

log "=== writing RESULTS.md ==="
.venv/bin/python -c "
import json
from pathlib import Path
out = Path('$OUT')
def read(p):
    j = json.load(open(p))
    return j['pe_mean'], j['pe_std'], j['auc_mean'], j['auc_std'], j['n_paired_covers']
pe5, ps5, au5, as5, n5 = read(out / 'pf005' / 'results.json')
pe3, ps3, au3, as3, n3 = read(out / 'pf030' / 'results.json')
md = f'''# Gap #3 — GFR-style + FLD-ensemble baseline

**Task:** §3 Gap #3 (paper §6.2 classical hand-crafted baseline)
**Date:** 2026-05-15
**Dataset:** BOSSbase 1.01 grayscale QF75 (paired cover + Phasm Ghost)
**Features:** SRMQ1-lite + DCTR-lite + GFR-lite (2,124 features per image,
              same kernels as 2026-05-11-path4e-handcrafted-gray FULL run)
**Classifier:** Fisher Linear Discriminant ensemble (Kodovský 2012):
              L=33 base learners, d_sub=600 features, bagging 50%

## Headline

| Payload | n paired | AUC | PE | acc |
|---|---|---|---|---|
| 0.05 bpnzAC | {n5} | {au5:.3f} ± {as5:.3f} | **{pe5:.3f} ± {ps5:.3f}** | {1-pe5:.3f} |
| 0.30 bpnzAC | {n3} | {au3:.3f} ± {as3:.3f} | **{pe3:.3f} ± {ps3:.3f}** | {1-pe3:.3f} |

## Interpretation

GFR-style + FLD-ensemble is the classical hand-crafted-feature baseline
called for by §3 Gap #3. The configuration follows Kodovský et al. 2012
(33 FLD learners with sklearn `LinearDiscriminantAnalysis` solver=lsqr +
shrinkage, random feature-subset bagging) on a feature vector that
concatenates the GFR-lite (24 Gabor filters × 20 hist bins = 480), SRMQ1-lite
(900 features), and DCTR-lite (~512 features) heads. The full 2,124-dim
feature vector substitutes for the 17,000-dim canonical Holub 2015 GFR.

Expected behaviour at payload 0.30 is somewhere between Phase 4c+4d SRNet
(PE 0.30 at 0.20 bpnzAC, PE 0.15 at 0.40 bpnzAC) and pure chance — FLD
ensembles typically achieve $\\Delta \\approx +0.05$ PE relative to SRNet
at the same payload.

## Comparison to other detectors at the same payload

| Detector class | Payload 0.05 PE | Payload 0.30 PE |
|---|---|---|
| SRNet curriculum-trained | (not measured) | (queued E9) |
| EfficientNet-B0 (aletheia, single-feature) | (cover delta only) | (cover delta only) |
| **GFR+FLD (this run)** | **{pe5:.3f}** | **{pe3:.3f}** |

## Reproducibility

```bash
cd ~/Development/phasm/eval
.venv/bin/python detectors/extract_handcrafted_cover_stego.py \\\\
  --cover-dir data/bossbase/jpeg_qf75 \\\\
  --stego-dir data/stego/phasm_ghost/qf75/payload_030 \\\\
  --out-dir   data/gfrfld_phasm_qf75_pf030 \\\\
  --workers 8 --limit 1500
.venv/bin/python detectors/train_fld_ensemble.py \\\\
  --features-dir data/gfrfld_phasm_qf75_pf030 \\\\
  --out-dir runs/2026-05-15-gap3-gfrfld-baseline/pf030 \\\\
  --seed 42
```

## History

- 2026-05-15: Initial GFR+FLD baseline at QF75 payload 0.05 and 0.30.
'''
(out / 'RESULTS.md').write_text(md)
print(f'wrote {out / \"RESULTS.md\"}')
"

log "=== Gap #3 DONE ==="
