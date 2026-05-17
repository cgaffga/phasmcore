#!/bin/bash
# Wait for J-UNIWARD reference generation (10k covers × 2 payloads) to
# complete, then kick off SRNet from-scratch training at 0.4 bpnzAC.
#
# Why 0.4 first: that's the regime where the detector has clear signal
# (J-UNIWARD@0.4 → 86% per-image classification under aletheia EffNet
# at n=100). Training a SRNet detector here produces our first
# canonical "trained-from-scratch on BOSSbase QF75" detector.
# We can re-train at 0.1 afterward for the lower-payload regime.
#
# Output: runs/<datestamp>-srnet-juniward-pf040-s042/
# - config.json
# - results.json
# - checkpoints/best.pt
# - LAUNCH.log (this script's stdout)

set -e
cd "$(dirname "$0")/.."

PAYLOAD_010_DIR="data/stego/juniward/qf75/payload_010"
PAYLOAD_040_DIR="data/stego/juniward/qf75/payload_040"
EXPECTED_COUNT=10000

echo "[launcher] waiting for J-UNIWARD generation to complete..."
while true; do
    n10=$(ls "$PAYLOAD_010_DIR" 2>/dev/null | wc -l | tr -d ' ')
    n40=$(ls "$PAYLOAD_040_DIR" 2>/dev/null | wc -l | tr -d ' ')
    echo "[launcher] $(date +%H:%M:%S) payload_010=$n10  payload_040=$n40 / $EXPECTED_COUNT"
    if [ "$n10" -ge "$EXPECTED_COUNT" ] && [ "$n40" -ge "$EXPECTED_COUNT" ]; then
        echo "[launcher] J-UNIWARD generation complete — kicking off training"
        break
    fi
    sleep 120
done

echo "[launcher] starting SRNet training @ 0.4 bpnzAC, 40 epochs, seed=42"
uv run python detectors/train_srnet.py \
    --payload 0.4 \
    --quality 75 \
    --seed 42 \
    --epochs 40 \
    --batch-size 8 \
    --n-train 4000 \
    --n-val 1000 \
    --n-test 5000 \
    --image-size 512 \
    --run-name srnet-juniward-pf040-s042

echo "[launcher] training complete"
