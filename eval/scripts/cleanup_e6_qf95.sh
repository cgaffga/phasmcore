#!/bin/bash
# Cleanup script — run AFTER all QF95-dependent experiments (E9, E11) complete.
# Deletes the QF95 generated dataset (covers + stegos + J-UNI baseline),
# leaving only the canonical 7z archive that they were regenerated from.
#
# Regeneration recipe (if needed later):
#   bash eval/runs/2026-05-14-e6-qf95-pipeline/run_e6.sh

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== E6 cleanup START $(date) ==="
echo "Before:"
du -sh data/bossbase/jpeg_qf95 data/bossbase/raw data/bossbase/phasm_ghost data/bossbase/juniward 2>/dev/null || true

# Keep: data/bossbase/BOSSbase_1.01.7z
# Delete: everything we generated from it for the QF95 experiments
rm -rf data/bossbase/raw
rm -rf data/bossbase/jpeg_qf95
# Phasm Ghost stegos are organized as phasm_ghost/qf95/payload_*/
if [ -d data/bossbase/phasm_ghost/qf95 ]; then
    rm -rf data/bossbase/phasm_ghost/qf95
fi
# J-UNIWARD baseline likewise
if [ -d data/bossbase/juniward/qf95 ]; then
    rm -rf data/bossbase/juniward/qf95
fi

echo "After:"
du -sh data/bossbase 2>/dev/null
df -h /Users/cgaffga | tail -1
echo "=== E6 cleanup DONE $(date) ==="
echo "7z preserved: $(ls -lh data/bossbase/BOSSbase_1.01.7z)"
