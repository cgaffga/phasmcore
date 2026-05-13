#!/bin/bash
# Visual A/B demo render for OpenH264 backend.
#
# For each of 3 representative real-world fixtures, render TWO mp4s
# to ~/Desktop:
#   ~/Desktop/openh264_<tag>_clean.mp4   — clean encode, no stego
#   ~/Desktop/openh264_<tag>_stego.mp4   — same encode with N CoeffSign
#                                          override on first N positions
#
# Purpose: visual sanity-check that the OpenH264 backend + stego hook
# pipeline produces viewable output on real-world content. NOT
# cascade-safe — stego mp4 won't round-trip the payload but should be
# visually intact.
#
# Run after regen_openh264_baseline.sh has built the example.

set -e
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../../../../../.." && pwd)"
DESK="$HOME/Desktop"

N_FRAMES=60      # 2 seconds at 30 fps
QP=26
GOP=30
PAYLOAD="phasm-stego-visual-demo-2026-05-12"  # ~250 bits

fixtures=(
    "IMG_4138.MOV|img4138"
    "Artlist_CarPlane.mp4|carplane"
    "lumix_g9_1080p_30fps_h264_high.mp4|lumix_g9"
)

# Per-fixture target dims: preserve native orientation, fit longest
# side into 1920, 16-MB-align both. CarPlane (1080x1920 portrait)
# becomes 1072x1920; landscape 1920x1080 becomes 1920x1072; DJI 2720x1530
# becomes 1920x1072.
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

build_example() {
    echo "Building examples (clean_encode + visual_demo) ..."
    (cd "$REPO_ROOT/core" && cargo build --release \
        --example openh264_clean_encode \
        --example openh264_visual_demo \
        --features "openh264-backend video h264-encoder cabac-stego" >/dev/null 2>&1)
}

render_one() {
    local src=$1
    local tag=$2
    local mov="$SRC_DIR/$src"

    if [ ! -f "$mov" ]; then
        echo "SKIP $tag: $mov missing"
        return 0
    fi

    local width height
    read width height < <(get_target_dims "$mov")
    local yuv="/tmp/openh264_demo_${tag}_${width}x${height}_f${N_FRAMES}.yuv"
    local clean_h264="/tmp/openh264_demo_${tag}_clean.h264"
    local stego_h264="/tmp/openh264_demo_${tag}_stego.h264"
    local clean_mp4="$DESK/openh264_${tag}_clean.mp4"
    local stego_mp4="$DESK/openh264_${tag}_stego.mp4"

    if [ ! -f "$yuv" ] || [ "$mov" -nt "$yuv" ]; then
        echo "Extracting YUV from $src ($width x $height × $N_FRAMES) ..."
        ffmpeg -y -loglevel error -i "$mov" \
            -vf "scale=${width}:${height},format=yuv420p" \
            -frames:v $N_FRAMES -f rawvideo "$yuv"
    fi

    echo "Encoding $tag (clean + stego) via OpenH264 backend ..."
    "$REPO_ROOT/core/target/release/examples/openh264_visual_demo" \
        "$yuv" "$clean_h264" "$stego_h264" \
        $width $height $N_FRAMES $QP $GOP "$PAYLOAD"

    echo "Muxing $tag clean -> $clean_mp4 ..."
    ffmpeg -y -loglevel error -framerate 30 -i "$clean_h264" -c:v copy "$clean_mp4"

    echo "Muxing $tag stego -> $stego_mp4 ..."
    ffmpeg -y -loglevel error -framerate 30 -i "$stego_h264" -c:v copy "$stego_mp4"
    echo ""
}

build_example
for f in "${fixtures[@]}"; do
    IFS='|' read -r src tag <<< "$f"
    render_one "$src" "$tag"
done

echo "Done. Visual demos on Desktop:"
ls -la "$DESK"/openh264_*.mp4 2>/dev/null || echo "  (no demos rendered — check sources)"
