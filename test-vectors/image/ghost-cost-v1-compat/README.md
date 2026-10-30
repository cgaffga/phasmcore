# Ghost cost-v1 compatibility fixtures

Ghost stego images (primary + 1–4 shadows) encoded by the **released
phasm-core 0.4.3 CLI** (Ghost v1: binary STC on the non-zero AC set,
v1 J-UNIWARD costs, shadow layout v1). The current decoder (Ghost v3,
with the v1 decoder as fallback) must decode every primary and every
shadow: `core/tests/ghost_cost_v1_compat.rs`.

- Covers: `photo_320x240_q75_420.jpg`, `photo_640x480_q75_420.jpg` (JPEG
  pass-through) and a PNG of the 640×480 photo (SI / Deep Cover path).
- `*_tight_*`: shadows sized to nearly fill the cheapest 5 % pool.
- Messages and passphrases: `regen.sh` (and mirrored in the test).
- Larger fixtures (12 MP photo, WhatsApp 1200×1600, two real PNG photos)
  are written to the gitignored `core/tests/real_photos/ghost-cost-v1-compat/`
  by `regen.sh <bin> --large`; the test skips them when absent.

Regenerate only with a v1-cost binary (see `regen.sh`). The encoder salts
randomly, so regenerated files differ byte-wise; the test only needs the
messages to match.
