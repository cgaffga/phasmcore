# Real-world YUV test vectors

Raw planar yuv420p frames extracted from `IMG_4138.MOV` (a real iPhone
camera capture on the user's Desktop — NOT committed here). Used by
the encoder integration suite to catch spec-conformance bugs that
synthetic patterns (flat frames, `deterministic_yuv420p`) don't
trigger.

| File | Size | Frames | Purpose |
|------|------|--------|---------|
| `img4138_32x32_f1.yuv`   | 32×32   | 1  | Minimum repro for the "bitstream unparseable on real content" bug |
| `img4138_64x48_f5.yuv`   | 64×48   | 5  | 2×2 MBs wide, tests I+P sequence |
| `img4138_128x80_f10.yuv` | 128×80  | 10 | Larger repro with many MBs + GOPs |

Regenerate with:
```
ffmpeg -i ~/Desktop/IMG_4138.MOV -vf "scale=W:H,format=yuv420p" -frames:v N -f rawvideo out.yuv
```
