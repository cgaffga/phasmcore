# Calibration-sweep source videos

Source `.MOV` / `.MP4` files used by the H.264 calibration sweep
harnesses (tasks #52 / #53 / #54 — Phase D.2-stealth) AND the §6E-A6
B-frame stealth-distribution work (#123 / #126 / #127 — Phase
6E-A6). **The videos themselves are gitignored** (see repo root
`.gitignore`); keep a personal copy here locally so the regen
scripts below can find them.

The derived `.yuv` files go to `/tmp/` on demand — never committed,
wiped on reboot, regenerate cheaply.

## Contents (gitignored)

### iPhone 13 Pro HEVC (calibration sweep — pixel content only)

| File | Size | Resolution | FPS | Codec | Notes |
|---|-----:|---|---:|---|---|
| `IMG_4138.MOV` | ~20 MB | 1920×1080 | 30 | HEVC Main | Photo-content calibration baseline. Christoph 2024. |
| `IMG_4273.MOV` | ~25 MB | 1920×1080 | 30 | HEVC Main | Photo-content, motion-heavy variant. Christoph 2024. |

The HEVC bitstreams are NOT used by phasm directly (HEVC stego is
archived). Only the decoded YUV pixels matter — they're real
camera-capture content with natural noise + motion patterns useful
as encoder-input fixtures.

### Real-world H.264 reference encoders (§6E-A6 stealth-distribution)

These are full-bitstream H.264 captures from real consumer/prosumer
cameras. They're the source of the committed reference histograms in
`core/tests/data/REFERENCE_DISTRIBUTIONS.md` — phasm's encoder
output is compared against their mb_type / partition / direction
distributions to verify the L3 fingerprint matches "in-the-wild" H.264.

| File | Size | Resolution | FPS | Codec | GOP | Notes |
|---|-----:|---|---:|---|---|---|
| `lumix_g9_1080p_30fps_h264_high.mp4` | ~24 MB | 1920×1080 | 29.97 | H.264 High | **IBBP M=3** | Panasonic Lumix G9. Original camera output, no re-encode. 67% B-frames — first non-x264 H.264 IBBP reference in the project. |
| `dji_mini2_2_7k_24fps_h264_high.mp4` | ~38 MB | 2720×1530 | 23.976 | H.264 High | IPPPP | DJI Mini2 drone. Original camera output. 0 B-frames (drone encoder restricts to I+P only). Distinct partition fingerprint: only 16x16 + 8x8, no 16x8 / 8x16. |
| `iphone5_1080p_30fps_h264_high.mov` | ~26 MB | 1920×1080 | 29.97 | H.264 High | IPPPP | Apple iPhone 5 (real-world video from the user's old device). Original camera output. 0 B-frames. Closest partition mix to x264-medium of any iPhone-era reference (heavy 8x8 use, 25%). |
| `iphone7_1080p_30fps_h264_high.mov` | ~31 MB | 1920×1080 | 30 | H.264 High | IPPPP | Apple iPhone 7 (real-world video from the user's old device). Original camera output. 0 B-frames. Distinct generation fingerprint: heavy 16x8/8x16 use (~34% combined), low 8x8 (~5%) — Apple's H.264 encoder evolved noticeably between iPhone 5 and 7. |

**Provenance**: all four clips are personal / unmodified original
camera outputs from the user's devices (Christoph, captures spanning
multiple years). The two iPhone clips are real-world casual videos
from the user's iPhone 5 and iPhone 7 — kept as period-representative
H.264 fingerprints for Apple's hardware encoder across generations.
None of these clips are redistributed; all stay gitignored.

### ytnews fallback (legacy)

For the ytnews clip (1280×720 news broadcast), regen pulls the
already-compressed `.h264` elementary stream in `/tmp/ytnews.h264`
(also gitignored). If that's missing you need to re-save from the
upstream source; see `docs/design/h264-encoder-quality-plan.md`
§ D.2-stealth for the original download path. Single-vendor and
not in active use post-§6E-A6 reference-distribution work.

## Regen

Run from repo root:

```bash
core/test-vectors/video/h264/real-world/source/regen.sh
```

Produces `/tmp/img4138_1080p_f10.yuv` and `/tmp/img4273_1080p_f10.yuv`
(30 MB each, 10 frames @ 1920×1072 yuv420p) — the dimensions expected
by the calibration harnesses. Re-run after a reboot or if you've
cleared `/tmp/`.

Exits non-zero with a clear message if a source `.MOV` is missing —
put the file back here, or update the script's source path.

## Cleanup

When a calibration sweep is done for a given phase, delete the
`/tmp/img*.yuv` files manually; they compress poorly and eat /tmp
space fast. A reboot will clean them up automatically.
