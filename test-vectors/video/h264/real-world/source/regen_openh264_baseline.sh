#!/bin/bash
# Regenerate /tmp/openh264_baseline_*.h264 — Annex-B reference clips
# encoded with the OpenH264 fork (cgaffga/phasm-openh264) at constant
# QP, no stego callbacks. Then run the histogram extractor and commit
# the resulting tables to core/tests/data/openh264_baseline_*.txt.
#
# Used by Phase C.4-pre.2 (task #390) to produce the cover-story
# comparison reference for OpenH264-backend output. Compared against
# x264-medium (regen_x264_reference.sh) and iPhone VideoToolbox
# (regen_iphone_vt_reference.sh) to pick the closest real-encoder
# cover story in C.4-pre.3 (#391).
#
# Idempotent; safe to re-run. The /tmp Annex-B outputs are gitignored;
# the committed artefact is the per-fixture histogram .txt in
# core/tests/data/.
#
# OpenH264 mainline (and this fork's default config) does NOT emit
# B-frames — gop_size=30 here produces IPPPP not IBPBP. That divergence
# from x264-medium IBPBP shape is itself a key data point for the
# cover-story decision: OpenH264's natural fingerprint is close to
# iPhone VideoToolbox (both default IPPPP), not to x264-medium.

set -e
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
# source/ -> real-world/ -> h264/ -> video/ -> test-vectors/ -> core/ -> repo root
REPO_ROOT="$(cd "$SRC_DIR/../../../../../.." && pwd)"

N_FRAMES=10
QP=26
GOP=30

# Per-fixture target dims: preserve native orientation, fit longest
# side into 1920, 16-MB-align both. CarPlane (1080x1920 portrait)
# becomes 1072x1920; landscape 1920x1080 becomes 1920x1072; DJI
# 2720x1530 becomes 1920x1072.
get_target_dims() {
    local src=$1
    local iw ih
    IFS=',' read iw ih _ < <(ffprobe -v error -select_streams v:0 \
        -show_entries stream=width,height -of csv=p=0 "$src")
    local tw th
    if [ "$iw" -ge "$ih" ]; then
        tw=1920
        th=$(( (1920 * ih / iw) / 16 * 16 ))
    else
        th=1920
        tw=$(( (1920 * iw / ih) / 16 * 16 ))
    fi
    echo "$tw $th"
}

# fixture descriptor: SOURCE_FILE|OUTPUT_TAG
fixtures=(
    "IMG_4138.MOV|img4138"
    "IMG_4273.MOV|img4273"
    "lumix_g9_1080p_30fps_h264_high.mp4|lumix_g9"
    "dji_mini2_2_7k_24fps_h264_high.mp4|dji_mini2"
    "Artlist_CarPlane.mp4|carplane"
)

build_example() {
    echo "Building example openh264_clean_encode --features openh264-backend ..."
    (cd "$REPO_ROOT/core" && cargo build --release --example openh264_clean_encode \
        --features "openh264-backend video h264-encoder cabac-stego" >/dev/null 2>&1) \
        || { echo "ERROR: cargo build failed"; exit 1; }
}

regen_one() {
    local src=$1
    local tag=$2
    local mov="$SRC_DIR/$src"

    if [ ! -f "$mov" ]; then
        echo "SKIP $tag: $mov missing — keep your personal source copy here (gitignored)."
        return 0
    fi

    local width height
    read width height < <(get_target_dims "$mov")
    local yuv="/tmp/openh264_baseline_${tag}_${width}x${height}_f${N_FRAMES}.yuv"
    local h264="/tmp/openh264_baseline_${tag}.h264"
    local hist="$REPO_ROOT/core/tests/data/openh264_baseline_${tag}_reference_mb_histogram.txt"

    if [ ! -f "$yuv" ] || [ "$mov" -nt "$yuv" ]; then
        echo "Extracting YUV for $tag (${width}x${height}, ${N_FRAMES} frames) ..."
        ffmpeg -y -loglevel error -i "$mov" \
            -vf "scale=${width}:${height},format=yuv420p" \
            -frames:v $N_FRAMES -f rawvideo "$yuv"
    fi

    echo "Encoding $tag via OpenH264 fork (${width}x${height}, qp=${QP}, gop=${GOP}) ..."
    "$REPO_ROOT/core/target/release/examples/openh264_clean_encode" \
        "$yuv" "$h264" $width $height $N_FRAMES $QP $GOP

    echo "Extracting histogram for $tag -> $hist ..."
    python3 "$REPO_ROOT/scripts/h264_mb_partition_histogram.py" "$h264" "$hist"
    echo ""
}

build_example
for f in "${fixtures[@]}"; do
    IFS='|' read -r src tag <<< "$f"
    regen_one "$src" "$tag"
done

echo "Done. Committed reference histograms:"
ls -la "$REPO_ROOT"/core/tests/data/openh264_baseline_*_reference_mb_histogram.txt 2>/dev/null || \
    echo "  (none yet — check that the source MOVs are present in $SRC_DIR)"
