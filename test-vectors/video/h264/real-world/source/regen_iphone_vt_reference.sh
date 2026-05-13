#!/bin/bash
# Regenerate iPhone VideoToolbox reference histograms used by Phase
# C.4-pre.3 (task #391) to inform the OpenH264 cover-story decision.
#
# Per the C.4-pre design (user-decided 2026-05-12), the raw .h264 +
# .txt outputs are NOT committed — the memo embeds the per-fixture
# marginal numbers directly. This script is the regen path for
# reproducing or refreshing those numbers locally.
#
# Two flavours of "iPhone VT" reference are captured:
#
#   1. **iPhone-device output** — the iphone{5,7}_1080p_30fps_h264_high.mov
#      source files are recorded straight from iPhone hardware via
#      iPhone VideoToolbox on-device. The committed marginals from
#      these ARE the canonical "look like an iPhone video" target.
#
#   2. **macOS VT re-encode** — ffmpeg -c:v h264_videotoolbox on the
#      same 1920x1072 × 10f YUV extract used for the OpenH264 baseline
#      in regen_openh264_baseline.sh. Useful for apples-to-apples
#      comparison on the same content. Slightly different than (1)
#      because macOS VT defaults to higher quality + different param
#      defaults than iPhone hardware.
#
# Output dir: $HOME/.cache/phasm/iphone-vt-refs/
# Outputs are .h264 (intermediate, kept for hand-inspection) + .txt
# (histogram). The directory is gitignored by convention; rerun this
# script to refresh.

set -e
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
# source/ -> real-world/ -> h264/ -> video/ -> test-vectors/ -> core/ -> repo root
REPO_ROOT="$(cd "$SRC_DIR/../../../../../.." && pwd)"

OUT_DIR="$HOME/.cache/phasm/iphone-vt-refs"
mkdir -p "$OUT_DIR"

WIDTH=1920
HEIGHT=1072
N_FRAMES=10

histogram_one() {
    local h264=$1
    local hist=$2
    echo "  Histogram -> $hist"
    python3 "$REPO_ROOT/scripts/h264_mb_partition_histogram.py" "$h264" "$hist"
}

# Flavour 1: histogram iPhone-device source files directly. These are
# the canonical references (real iPhone hardware output).
flavour1_one() {
    local src=$1
    local tag=$2
    local mov="$SRC_DIR/$src"
    local h264="$OUT_DIR/iphone_device_${tag}.h264"
    local hist="$OUT_DIR/iphone_device_${tag}_reference_mb_histogram.txt"

    if [ ! -f "$mov" ]; then
        echo "SKIP $tag (device): $mov missing"
        return 0
    fi

    if [ ! -f "$h264" ] || [ "$mov" -nt "$h264" ]; then
        echo "Extracting Annex-B from $src (iPhone hardware-encoded, no re-encode) ..."
        # `-bsf:v h264_mp4toannexb` converts from .mov MP4 NALU format to
        # Annex-B start-code stream the histogram script expects.
        ffmpeg -y -loglevel error -i "$mov" -c:v copy -bsf:v h264_mp4toannexb \
            -f h264 "$h264"
    fi
    histogram_one "$h264" "$hist"
}

# Flavour 2: macOS VideoToolbox re-encode on a 10-frame extract from
# the same iPhone source MOVs, for same-content comparison with the
# OpenH264 baseline.
flavour2_one() {
    local src=$1
    local tag=$2
    local mov="$SRC_DIR/$src"
    local yuv="/tmp/iphone_vt_${tag}_1080p_f${N_FRAMES}.yuv"
    local h264="$OUT_DIR/macos_vt_${tag}.h264"
    local hist="$OUT_DIR/macos_vt_${tag}_reference_mb_histogram.txt"

    if [ ! -f "$mov" ]; then
        echo "SKIP $tag (macOS VT): $mov missing"
        return 0
    fi

    if [ ! -f "$yuv" ] || [ "$mov" -nt "$yuv" ]; then
        echo "Extracting YUV from $src for macOS VT re-encode ..."
        ffmpeg -y -loglevel error -i "$mov" \
            -vf "scale=${WIDTH}:${HEIGHT},format=yuv420p" \
            -frames:v $N_FRAMES -f rawvideo "$yuv"
    fi

    echo "macOS VT re-encoding $tag (h264_videotoolbox) ..."
    ffmpeg -y -loglevel error -f rawvideo -pix_fmt yuv420p \
        -s "${WIDTH}x${HEIGHT}" -r 30 -i "$yuv" \
        -c:v h264_videotoolbox -profile:v high -g 30 \
        -an "$h264"
    histogram_one "$h264" "$hist"
}

echo "=== Flavour 1: iPhone-device hardware output ==="
flavour1_one iphone5_1080p_30fps_h264_high.mov iphone5
flavour1_one iphone7_1080p_30fps_h264_high.mov iphone7

echo ""
echo "=== Flavour 2: macOS VideoToolbox re-encode on same-content extract ==="
flavour2_one IMG_4138.MOV img4138
flavour2_one IMG_4273.MOV img4273
flavour2_one iphone7_1080p_30fps_h264_high.mov iphone7

echo ""
echo "Done. Local reference histograms in $OUT_DIR:"
ls -la "$OUT_DIR"/*_reference_mb_histogram.txt 2>/dev/null || \
    echo "  (none — check that source files exist in $SRC_DIR)"
