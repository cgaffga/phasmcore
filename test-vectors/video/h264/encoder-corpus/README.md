# H.264 Encoder Corpus

Test vectors for Phase 6 encoder fingerprint regression + decode
oracle harness. Used by:

- `docs/design/h264-encoder-algorithms/fingerprint-regression.md`
- `docs/design/h264-encoder-algorithms/oracle-harness.md`

## Regenerating

```sh
./generate.sh               # all clips
./generate.sh libx264       # only libx264 outputs
./generate.sh h264_videotoolbox  # only VideoToolbox outputs
```

Requires ffmpeg 6+ with libx264 + VideoToolbox (macOS only for the
latter). All sources are deterministic lavfi generators — no external
downloads; byte-exact across machines with the same ffmpeg version.

## Current clips

| File | Source | Encoder | Size |
|------|--------|---------|------|
| `testsrc_x264_crf23_fast.mp4` | testsrc (color pattern) | libx264, preset=fast, crf=23 | ~9 KB |
| `testsrc_x264_crf18_slow.mp4` | testsrc | libx264, preset=slow, crf=18 | ~12 KB |
| `testsrc_vt_q60.mp4` | testsrc | VideoToolbox, q=60 | ~13 KB |
| `mandelbrot_x264_crf23_fast.mp4` | mandelbrot (fractal) | libx264, preset=fast, crf=23 | ~46 KB |
| `mandelbrot_vt_q60.mp4` | mandelbrot | VideoToolbox, q=60 | ~46 KB |
| `smptebars_x264_crf23_fast.mp4` | smptebars (SMPTE color bars) | libx264, preset=fast, crf=23 | ~3 KB |

Total ~130 KB — well under the 1 MB budget for the corpus.

All clips: 320×240, 15 fps, 1s duration, yuv420p, Baseline / Constrained
Baseline profile, level 3.0. CAVLC entropy coding (Baseline-mandated).

## Why these sources

- **testsrc** — hard edges, flat regions, color sweeps. Exercises
  intra prediction under simple pattern content.
- **mandelbrot** — fractal, non-trivial motion frame-to-frame.
  Exercises inter prediction + ME quality differentiation.
- **smptebars** — near-pathological low-entropy content. Forces
  encoders to pick skip-MB modes and chroma intra modes they rarely
  visit in real footage.

## Adding new clips

1. Append a `run ...` line to `generate.sh`.
2. Run `./generate.sh`.
3. Commit both the script change and the new .mp4.
4. Add a row to the table above.

Clips should stay < 100 KB each. Use lavfi sources for determinism;
if you must use an external clip, download it into `/tmp/` at
generation time and document the URL + license in a comment.

## Budget

If this directory grows past 1 MB, split real-world clips out into
a gitignored `external/` subdirectory and fetch them lazily in
`generate.sh`.
