#!/bin/bash
# Regenerate /tmp/img*_1080p_f10.yuv calibration clips from the
# gitignored source .MOV files in this directory. Idempotent; safe to
# re-run. Delete the /tmp outputs by hand when no longer needed, or
# let a reboot wipe them.

set -e
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

regen_one() {
    local name=$1   # e.g. "IMG_4138"
    local out_tag=$2  # e.g. "4138"
    local mov="$SRC_DIR/$name.MOV"
    local yuv="/tmp/img${out_tag}_1080p_f10.yuv"

    if [ ! -f "$mov" ]; then
        echo "ERROR: $mov missing."
        echo "       Put a copy of $name.MOV in:"
        echo "       $SRC_DIR"
        echo "       (folder is gitignored; keep your personal source"
        echo "        copy here so the calibration sweeps can find it)."
        return 1
    fi

    if [ -f "$yuv" ] && [ "$mov" -ot "$yuv" ]; then
        echo "$yuv already up-to-date (source older than derived) — skipping."
        return 0
    fi

    echo "Regenerating $yuv  (from $mov)..."
    # Pad/crop to 1920x1072 (the dimensions the harness expects —
    # 1080 → 1072 to get MB-alignment, matches the original extraction).
    ffmpeg -y -loglevel error -i "$mov" \
        -vf "scale=1920:1072,format=yuv420p" \
        -frames:v 10 -f rawvideo "$yuv"
    ls -la "$yuv"
}

regen_one IMG_4138 4138
regen_one IMG_4273 4273
echo ""
echo "Done. Calibration harnesses can now read /tmp/img4138_1080p_f10.yuv"
echo "and /tmp/img4273_1080p_f10.yuv."
echo ""
echo "To clean up when finished:"
echo "  rm /tmp/img4138_1080p_f10.yuv /tmp/img4273_1080p_f10.yuv"
