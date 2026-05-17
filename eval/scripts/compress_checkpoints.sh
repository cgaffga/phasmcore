#!/bin/bash
# Compress runs/*/checkpoints/ back into runs/*/checkpoints.7z, then
# remove the originals. Counterpart to decompress_checkpoints.sh.
#
# Uses 7z -mx=9 (highest setting). SRNet .pt weights are float32 tensors
# so the compression ratio is modest (~8%), but the .7z is a clean
# single-file archive per run that's easier to ship/move than a dir.

set -euo pipefail

cd "$(dirname "$0")/.."
runs_dir="$(pwd)/runs"

for d in "$runs_dir"/*/checkpoints; do
    [ -d "$d" ] || continue
    [ "$(ls -A "$d" 2>/dev/null)" ] || continue
    parent="$(dirname "$d")"
    archive="$parent/checkpoints.7z"
    if [ -f "$archive" ]; then
        echo "  skip $(basename "$parent")/checkpoints.7z (exists; delete it first to recompress)"
        continue
    fi
    echo "  compress $(basename "$parent")/checkpoints/..."
    7z a -t7z -mx=9 -mmt=on "$archive" "$d"/* >/dev/null
    rm -rf "$d"
done
