#!/bin/bash
# Regenerate /tmp/img*_x264_reference.h264 — Annex-B reference clips
# encoded with x264 (HandBrake "Fast 1080p30"-equivalent settings).
#
# Used by Phase 6E-A6 stealth-comparison work
# (`docs/design/h264-encoder-algorithms/6E-A6-bslice-partitions.md`).
# x264 is the dominant in-the-wild H.264 producer (HandBrake, ffmpeg
# default, OBS, video converters), so matching its output distribution
# is the canonical "look like a typical converted H.264 video" target.
#
# Idempotent; safe to re-run. The output Annex-B files are gitignored;
# the static derived histograms in `core/tests/data/` are the
# committed artefact.

set -e
SRC_DIR="$(cd "$(dirname "$0")" && pwd)"

regen_one() {
    local name=$1     # e.g. "IMG_4138"
    local out_tag=$2  # e.g. "4138"
    local mov="$SRC_DIR/$name.MOV"
    local h264="/tmp/img${out_tag}_x264_reference.h264"

    if [ ! -f "$mov" ]; then
        echo "ERROR: $mov missing."
        echo "       Put a copy of $name.MOV in:"
        echo "       $SRC_DIR"
        return 1
    fi

    if [ -f "$h264" ] && [ "$mov" -ot "$h264" ]; then
        echo "$h264 already up-to-date — skipping."
        return 0
    fi

    echo "Encoding $h264 (x264 medium-CRF23 IBPBP from $mov)..."
    # x264 medium-CRF23 with IBPBP: this is roughly HandBrake's
    # "Fast 1080p30" preset shape — the most common consumer
    # convert-to-H.264 configuration.
    #   -bf 2 -g 30 -keyint_min 30 -sc_threshold 0 :
    #     IBPBP M=2 with 30-frame closed GOP, no scene-cut IDRs
    #     (so the GOP shape is regular and easy to compare against).
    ffmpeg -y -loglevel error -i "$mov" \
        -c:v libx264 -profile:v high -preset medium -crf 23 \
        -bf 2 -g 30 -keyint_min 30 -sc_threshold 0 \
        -an "$h264"
    ls -la "$h264"
}

regen_one IMG_4138 4138
regen_one IMG_4273 4273
echo ""
echo "Done. Run the histogram extractor against either reference:"
echo "  python3 scripts/h264_mb_partition_histogram.py /tmp/img4138_x264_reference.h264"
echo ""
echo "The committed reference histogram lives at:"
echo "  core/tests/data/x264_reference_mb_histogram.txt"
