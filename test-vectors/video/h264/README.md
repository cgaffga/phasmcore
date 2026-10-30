# H.264/AVC Test Vectors

Test vectors for H.264 video steganography.

| Path | What it is |
|------|------------|
| `encoder-corpus/` | Small external-encoder clips (libx264, VideoToolbox) for the ffmpeg decode-oracle smoke test (`core/tests/h264_oracle_test.rs`). |
| `real-world/` | Raw YUV fixtures from the retired pure-Rust encoder's test suite, plus `source/` (gitignored camera clips and the regen scripts for the calibration / reference-histogram work). |
| `test_baseline_320x240.mp4`, `test_tiny.mp4` | Legacy tiny clips. No current test references them. |

The production encoder is OpenH264 (feature `h264-encoder`); the decoder is
the pure-Rust walker (`h264-decoder`). Large source videos are never
committed; see `real-world/source/README.md`.
