# AV1 Stego — Real-World Corpus

This directory hosts the manifest for the AV1 stego corpus. The actual
**source media** lives one tree over, under
`core/test-vectors/video/h264/real-world/source/` — we share the same
gitignored MOV/MP4 corpus with the H.264 stego tests so cross-codec
behavior can be compared on apples-to-apples content.

## Files

- `manifest.toml` — per-fixture baseline metrics (encode timing, natural
  packet bytes, AC_COEFF_SIGN cover-bit capacity). Recorded on
  2026-05-21 against phasm-rav1e `6254b700` + phasm-dav1d `619908ef`.
  Consumed by the W5 visual-gate CI workflow.

## Tests

`core/tests/av1_corpus_validation.rs` drives this corpus. Per
fixture: ffmpeg extracts a single frame as YUV4:2:0 → rav1e encodes
as an AV1 key frame via `encode_frame_with_phasm_tee` → orchestrator
embeds an encrypted message → dav1d decodes + extracts → plaintext
round-trip asserted.

Run:

```bash
cargo test --features av1-encoder,av1-backend --test av1_corpus_validation --release
```

## Source media

The MOV/MP4 source files for each fixture in `manifest.toml` are
gitignored. To run the tests, ensure the corresponding source file
exists at:

```
core/test-vectors/video/h264/real-world/source/<source>
```

Each entry in `manifest.toml` declares its `source` filename. The
H.264 corpus directory's `README.md` documents provenance and
includes `regen_*.sh` scripts that explain where the files came
from.

## Adding fixtures

1. Drop the new source MOV/MP4 into
   `core/test-vectors/video/h264/real-world/source/`.
2. Add a `Fixture { ... }` literal + `#[test]` function to
   `core/tests/av1_corpus_validation.rs`.
3. Run the test once, capture the eprintln baseline values.
4. Add an `[[fixture]]` block to `manifest.toml` with those values.

## v0.3 scope reminder

- Single-tile, single key frame.
- AC_COEFF_SIGN channel only (Tier 1 from `channel-design.md` § 6).
- Uniform STC costs (J-UNIWARD + cascade-safety land in v0.5+).
