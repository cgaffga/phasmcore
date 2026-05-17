#!/bin/bash
# Re-hydrate compressed checkpoints from runs/*/checkpoints.7z back into
# runs/*/checkpoints/. Run before any experiment that needs to load .pt
# weights (Phase 1c eval, Phase 3 cross-detector replication, etc.).
#
# Idempotent: skips runs that already have a checkpoints/ directory.
# Leaves the .7z in place so the next cleanup pass doesn't need to
# re-compress.
#
# To re-compress after use: see scripts/compress_checkpoints.sh.

set -euo pipefail

cd "$(dirname "$0")/.."
runs_dir="$(pwd)/runs"

found=0
for archive in "$runs_dir"/*/checkpoints.7z; do
    [ -f "$archive" ] || continue
    parent="$(dirname "$archive")"
    target="$parent/checkpoints"
    if [ -d "$target" ] && [ "$(ls -A "$target" 2>/dev/null)" ]; then
        echo "  skip $(basename "$parent")/checkpoints (already populated)"
        continue
    fi
    mkdir -p "$target"
    echo "  expand $(basename "$parent")/checkpoints.7z..."
    7z x -o"$target" "$archive" >/dev/null
    found=$((found + 1))
done

if [ "$found" -eq 0 ]; then
    echo "Nothing to decompress (all checkpoints already populated or no .7z files)."
fi
