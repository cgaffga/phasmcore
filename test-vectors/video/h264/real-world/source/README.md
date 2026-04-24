# Calibration-sweep source videos

Large source `.MOV` / `.MP4` files used by the H.264 calibration sweep
harness (tasks #52 / #53 / #54 — Phase D.2-stealth). **The videos
themselves are gitignored** (see repo root `.gitignore`); keep a copy
here locally so the regen script below can find them.

The derived `.yuv` files go to `/tmp/` on demand — never committed,
wiped on reboot, regenerate cheaply.

## Contents (gitignored)

| File | Size | Resolution | Purpose |
|------|-----:|------------|---------|
| `IMG_4138.MOV` | ~20 MB | 1920×1080 @ 30 fps | Photo-content calibration baseline |
| `IMG_4273.MOV` | ~25 MB | 1920×1080 @ 30 fps | Photo-content, motion-heavy variant |

For the ytnews clip (1280×720 news broadcast), regen pulls the
already-compressed `.h264` elementary stream in `/tmp/ytnews.h264`
(also gitignored). If that's missing you need to re-save from the
upstream source; see `docs/design/h264-encoder-quality-plan.md`
§ D.2-stealth for the original download path.

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
