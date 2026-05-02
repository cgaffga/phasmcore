#!/bin/bash
# Regenerate `core/tests/data/{lumix_g9,dji_mini2}_*_reference_mb_histogram.txt`
# from the gitignored Lumix G9 + DJI Mini2 source clips. Used by
# §6E-A6.5 (#126) stealth-distribution gate as alternative reference
# encoders alongside x264-medium-CRF23 (see regen_x264_reference.sh).
#
# Idempotent. Source MP4s are gitignored; histograms are committed.
# Re-run after dropping in a new sample camera capture, or to
# refresh after histogram-extractor improvements.

set -e
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SRC_DIR/../../../../../.." && pwd)"
TESTS_DATA="$REPO_ROOT/core/tests/data"
HISTOGRAM_TOOL="$REPO_ROOT/scripts/h264_mb_partition_histogram.py"

regen_one() {
    local mp4=$1
    local label=$2
    local hist_out=$3
    local h264_temp=$4

    if [ ! -f "$mp4" ]; then
        echo "ERROR: $mp4 missing — drop the original capture in $SRC_DIR."
        return 1
    fi

    if [ -f "$hist_out" ] && [ "$mp4" -ot "$hist_out" ]; then
        echo "$hist_out already up-to-date — skipping."
        return 0
    fi

    echo "Demuxing $label → $h264_temp..."
    ffmpeg -y -hide_banner -loglevel error \
        -i "$mp4" -c copy -f h264 -bsf:v h264_mp4toannexb "$h264_temp"

    echo "Histogram → $hist_out..."
    python3 "$HISTOGRAM_TOOL" "$h264_temp" "$hist_out"
}

regen_one \
    "$SRC_DIR/lumix_g9_1080p_30fps_h264_high.mp4" \
    "Lumix G9 1080p" \
    "$TESTS_DATA/lumix_g9_1080p_reference_mb_histogram.txt" \
    /tmp/lumix_g9_demux.h264

regen_one \
    "$SRC_DIR/dji_mini2_2_7k_24fps_h264_high.mp4" \
    "DJI Mini2 2.7K" \
    "$TESTS_DATA/dji_mini2_2_7k_reference_mb_histogram.txt" \
    /tmp/dji_mini2_demux.h264

echo ""
echo "Done. Committed histograms refreshed at:"
echo "  $TESTS_DATA/lumix_g9_1080p_reference_mb_histogram.txt"
echo "  $TESTS_DATA/dji_mini2_2_7k_reference_mb_histogram.txt"
