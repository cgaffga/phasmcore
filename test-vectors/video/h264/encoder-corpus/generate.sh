#!/usr/bin/env bash
# Regenerate the H.264 encoder-corpus test vectors.
#
# Each clip is produced from a deterministic ffmpeg lavfi source
# (testsrc, mandelbrot, smptebars) — no external downloads needed,
# and byte-exact across machines. All clips are short (1s, 15 fps)
# and small (< 100 KB) so the corpus stays under ~1 MB total.
#
# Usage:
#   ./generate.sh              # regenerate all clips
#   ./generate.sh libx264      # regenerate only libx264 outputs
#   ./generate.sh videotoolbox # regenerate only VideoToolbox outputs
#
# Phase 6.0c — see:
#   docs/design/video/h264/encoder-algorithms/fingerprint-regression.md
#   docs/design/video/h264/encoder-algorithms/oracle-harness.md
#
# Add new clips: extend the MATRIX table below. Use deterministic
# lavfi sources, include pix_fmt yuv420p for Baseline compatibility,
# and keep duration at 1s to bound total corpus size.

set -euo pipefail

cd "$(dirname "$0")"

FILTER="${1:-all}"

# Fail loudly if ffmpeg is missing. Corpus regeneration is a
# development-time task, not a build-time one — CI-less environments
# consume the committed outputs directly.
command -v ffmpeg >/dev/null 2>&1 || {
    echo "ffmpeg not found — install via 'brew install ffmpeg' (macOS) or" >&2
    echo "package manager. Corpus regeneration requires ffmpeg 6+."     >&2
    exit 1
}

# Baseline-only encoding: all outputs restricted to Baseline + CAVLC.
# Phase 6A targets Baseline; Phase 6C broadens coverage to Main + CABAC.
GLOBAL_OPTS="-hide_banner -loglevel error"
OUTPUT_OPTS="-pix_fmt yuv420p -profile:v baseline -level 3.0"

# MATRIX: source | encoder | extra_args | output_name
# Kept small on purpose. Each line regenerates one clip.
run() {
    local src="$1" enc="$2" extra="$3" out="$4"
    if [[ "$FILTER" != "all" && "$FILTER" != "$enc" ]]; then
        return 0
    fi
    echo "  → $out"
    # shellcheck disable=SC2086
    ffmpeg $GLOBAL_OPTS -f lavfi -i "$src" $OUTPUT_OPTS -c:v "$enc" $extra -y "$out"
}

echo "Regenerating H.264 encoder-corpus ($FILTER)..."

# testsrc — deterministic RGB pattern. Exercises intra prediction on
# hard edges + flat regions.
run "testsrc=duration=1:size=320x240:rate=15" \
    libx264 "-preset fast -crf 23" \
    testsrc_x264_crf23_fast.mp4

run "testsrc=duration=1:size=320x240:rate=15" \
    libx264 "-preset slow -crf 18" \
    testsrc_x264_crf18_slow.mp4

run "testsrc=duration=1:size=320x240:rate=15" \
    h264_videotoolbox "-q:v 60" \
    testsrc_vt_q60.mp4

# mandelbrot — fractal with fine-grained texture. Exercises inter
# prediction since successive frames are similar.
run "mandelbrot=size=320x240:rate=15" \
    libx264 "-preset fast -crf 23 -t 1" \
    mandelbrot_x264_crf23_fast.mp4

run "mandelbrot=size=320x240:rate=15" \
    h264_videotoolbox "-q:v 60 -t 1" \
    mandelbrot_vt_q60.mp4

# smptebars — colorbar pattern. Very low entropy; exercises skip-MB
# decisions and chroma intra modes.
run "smptebars=duration=1:size=320x240:rate=15" \
    libx264 "-preset fast -crf 23" \
    smptebars_x264_crf23_fast.mp4

echo "Done. Corpus contents:"
ls -la *.mp4 2>/dev/null || echo "  (no .mp4 outputs found)"
