# Image test vectors

Small JPEG covers for the codec and stego tests in `core/tests/`. The
file name gives size, JPEG quality and chroma subsampling
(`photo_640x480_q75_420.jpg` = 640×480, QF 75, 4:2:0); the set covers
baseline 4:4:4 / 4:2:2 / 4:2:0, grayscale, non-MCU-aligned and 8×8 images.

- `ghost-cost-v1-compat/` — Ghost v1 stego images (primary + shadows)
  from the released 0.4.3 CLI; they must keep decoding.
- `wire-format/` — Ghost stego images encoded on a known platform that
  must decode on every platform (`core/tests/wire_format_cross_platform_decode.rs`).

The encoder salts randomly, so stego output is not byte-reproducible;
tests round-trip the message rather than compare bytes. Larger
real-world fixtures are gitignored and skipped when absent.
