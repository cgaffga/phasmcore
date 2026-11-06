# phasm-core

[![crates.io](https://img.shields.io/crates/v/phasm-core.svg)](https://crates.io/crates/phasm-core)
[![License: GPL-3.0](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.93%2B-orange.svg)](https://www.rust-lang.org)
[![Platforms](https://img.shields.io/badge/platforms-iOS%20%7C%20Android%20%7C%20Web%20%7C%20CLI-green.svg)](https://phasm.app)

Pure-Rust steganography engine for hiding encrypted text messages in JPEG photos.

This is the core library behind [Phasm](https://phasm.app) — available on [iOS](https://apps.apple.com/app/phasm-steganography/id6759446274), Android, and the [web](https://phasm.app).

> **Build from source, including H.264 and AV1 video stego: see [BUILDING.md](BUILDING.md).** Clone with `git clone --recursive` (the video codecs are git submodules), then `make check-deps && make cli-video`.

> **Name**: original etymology = *phasmid* (stick-insect family — hide-in-plain-sight, hence the green mascot). Technical backronym: **PHASM = Plausibly-deniable Hidden Adaptive Shadow Messaging** — each word maps to a property of the engine in this crate.

## Install

Add the library to your Cargo.toml with default features (no H.264 encoder, per [Via LA AVC](https://www.via-la.com/licensing-2/avc-h-264/) patent obligations):

```bash
cargo add phasm-core
```

Or install the CLI (pre-built binaries: see [CLI](#cli) below), or build it from source:

```bash
cargo install --git https://github.com/cgaffga/phasmcore phasm-cli
```

**Building from source, with or without video: [BUILDING.md](BUILDING.md)** covers the prerequisites per OS (meson, ninja, nasm for the video codecs), every feature combination, the `make` targets, and using phasm-core as a git dependency.

### Optional Cargo features

| Feature | Description |
|---------|-------------|
| `parallel` | [Rayon](https://github.com/rayon-rs/rayon) parallelism for J-UNIWARD costs, STC embedding, Armor decode sweeps. Recommended for native builds. |
| `h264-decoder` | H.264 video stego **decode** (pure-Rust CABAC walker, no C dependency). Implies `video`. |
| `video` | MP4 demux/mux and H.264 bitstream parsing; the base of `h264-decoder`. |
| `simd` | A NEON kernel for the H.264 stego walker (aarch64). The image-side SIMD kernels (NEON / AVX2 / WASM SIMD128) are always built. |

```bash
cargo add phasm-core --features parallel
cargo install --git https://github.com/cgaffga/phasmcore phasm-cli --features h264-decoder   # adds H.264 video decode
```

The H.264 encoder (`h264-encoder`) and the AV1 features (`av1-encoder`, `av1-decoder`) are not part of the crates.io package. They build from a recursive clone of this repository or as a git dependency; see [BUILDING.md](BUILDING.md) and "Video steganography" below.

## Two Embedding Modes

### Ghost (Stealth)

Optimizes for **undetectability**. Uses the [J-UNIWARD](https://phasm.app/blog/uerd-vs-juniwird-detection-benchmarks) cost function to assign distortion costs per DCT coefficient, then embeds via a ternary (double-layered) [Syndrome-Trellis Code (STC)](https://phasm.app/blog/syndrome-trellis-codes-practical-guide) over **every** luminance AC coefficient, zeros included, to minimize total distortion (Ghost v3). Large photos are split into independent lanes that are embedded in parallel. Measured with SRNet (BOSSbase 1.01, QF75, 0.20 bpnzAC, trained from scratch), the detection error is 0.272, against 0.190 for the earlier design (0.5 = undetectable); the capacity bar's comfort band keeps typical messages far below that rate.

Images written by core ≤ 0.4.3 (Ghost v1) still decode.

Use Ghost when the stego image will be stored or transmitted without recompression — the embedding does not survive JPEG re-encoding.

Ghost mode also supports **file attachments** (Brotli-compressed, multi-file).

#### Shadow Messages (Plausible Deniability)

Ghost mode supports **[shadow messages](https://phasm.app/blog/shadow-messages-plausible-deniability-steganography)** — multiple messages hidden in a single image, each with a different passphrase. If coerced into revealing a passphrase, you can give the shadow passphrase instead of the primary one. The adversary decodes a decoy message and has no way to detect whether additional messages exist.

Each shadow is a short **keyed-region STC** on the same luminance AC set as the primary. Key design properties:

- **Keyed regions, no costs at the decoder** — each passphrase defines a keyed ChaCha20 order over every Y AC coefficient (zeros included). A shadow of `n` RS-coded bits owns the first `n·w` positions of its order and embeds into the coefficient parity with an STC of width `w`. The AC set never changes under embedding, so the decoder rebuilds the order exactly
- **Cost-adaptive at the encoder** — the trellis minimizes the J-UNIWARD (or SI-UNIWARD) cost of the shadow's changes, so they land in textured areas like the primary's
- **Wet regions, no collisions** — each shadow treats every earlier shadow's region as wet (infinite cost), and the primary treats all regions as wet. No layer writes into another's region, so every layer arrives error-free; RS parity 4 is only an integrity gate
- **Headerless decode** — no magic bytes or agreed-upon parameters. For each candidate width the decoder reads a 48-bit syndrome header, requires the first RS block to decode with zero corrections, and only then extracts the full region. AES-256-GCM-SIV authentication is the only validator
- **Verify = decode** — after the primary is embedded, every shadow is extracted exactly as the decoder reads it

A single shadow may hold half the Ghost maximum, and the capacity bands bound the total of primary and shadows. Shadows written by core ≤ 0.4.3 (shadow layout v1) still decode. `smart_decode` automatically tries shadow decode as a fallback when the primary Ghost decode finds nothing.

#### SI-UNIWARD (Deep Cover)

When the input is not a JPEG (PNG, HEIC, a camera photo), **SI-UNIWARD** (Side-Informed UNIWARD) uses the JPEG quantization rounding errors of the cover as side information. Core writes the cover JPEG itself from the raw pixels (`deep_cover_jpeg`, byte-identical to libjpeg-turbo `cjpeg`; the apps use IJG quality 92), so every rounding error `e` is known exactly. A change toward the pre-quantization value costs the J-UNIWARD cost scaled by `1 − 2|e|`; a change away from it costs the full J-UNIWARD cost.

The result: **~25% more capacity** than standard Ghost (max nzAC/4 instead of nzAC/5 frame bits), with even better stealth — measured with SRNet at QF92 (BOSSbase 1.01, 4800 test pairs), the detection error is 0.450 ± 0.003 at the larger payload versus 0.366 ± 0.003 for standard Ghost at its maximum (0.5 = undetectable). The bonus is decided by core from the pixels: a cover that is not the writer's output for those pixels falls back to approximate side information (cost factor floored at 0.3) and standard Ghost capacity, and measures no better than standard Ghost. The decoder is completely unchanged — `ghost_decode` works identically on standard and SI-UNIWARD stego images.

Use `deep_cover_jpeg` to write the cover, then pass the same pixels to `ghost_encode_si` / `ghost_bands_si` / `ghost_capacity_si` (or let `ghost_encode_deep_cover` do both in one call).

### Armor (Robust)

Optimizes for **survivability**. Uses Spread Transform Dither Modulation (STDM) to embed bits into stable low-frequency DCT coefficients, protected by [adaptive Reed-Solomon ECC](https://phasm.app/blog/surviving-jpeg-recompression) with [soft-decision decoding via log-likelihood ratios](https://phasm.app/blog/soft-majority-voting-llr-concatenated-codes). A [DFT magnitude template](https://phasm.app/blog/dft-template-geometric-resilience-steganography) provides geometric synchronization against rotation, scaling, and cropping.

Armor is the **default mode** on all platforms.

Armor's robustness depends on the cover: it survives recompression and resizing only when the cover's longest edge is at most `ARMOR_TARGET_DIMENSION` (1600 px) and it has been re-encoded at QF65 first. The apps, the website and the `phasm` CLI prepare that cover automatically (the CLI's `--full-res` opts out, with a warning). If you call `armor_encode` from the library, prepare it yourself: a full-resolution Armor image still decodes, but does not survive a messenger's resize.

#### Fortress Sub-Mode

For short messages, Armor automatically activates **Fortress** — a [BA-QIM (Block-Average Quantization Index Modulation)](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) scheme that embeds one bit per 8x8 block into the block-average brightness. Fortress exploits [three invariants of JPEG recompression](https://phasm.app/blog/jpeg-recompression-invariants) — block averages, brightness ordering, and coefficient signs — that persist even through aggressive re-encoding.

Fortress survives WhatsApp recompression (QF ~62, resolution resize to 1600px). See [how 15 messaging platforms process your photos](https://phasm.app/blog/how-15-platforms-process-your-photos) for a comprehensive platform analysis.

[Watson perceptual masking](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) adapts the embedding strength per-block based on local texture energy, keeping modifications invisible in textured regions while protecting smooth areas.

### Decode Auto-Detection

`smart_decode` tries all modes automatically — Ghost primary (v3, then v1) → Ghost shadow → Armor → Fortress — no mode selector needed on the decode side.

### Which Mode Should I Use?

| | Ghost | Armor | Fortress |
|---|---|---|---|
| **Goal** | Undetectable | Survives recompression | Survives WhatsApp |
| **Algorithm** | J-UNIWARD + ternary STC | STDM + Reed-Solomon | BA-QIM + Watson masking |
| **Best for** | Direct sharing, AirDrop, email, "send as file" | Social media, cloud storage, cross-platform | WhatsApp, messaging apps with aggressive compression |
| **Capacity** | High (12 MP photo: ~2.6 KB comfort band, ~96 KB max) | Medium | Low (short messages only) |
| **Resists detection** | Yes (steganalysis-resistant) | Moderate | Moderate |
| **Resists recompression** | No | Yes (>95% at mild QF drop) | Yes (survives QF ~62) |
| **Plausible deniability** | Yes (shadow messages) | No | No |
| **File attachments** | Yes (up to 2 MB) | No | No |
| **Activation** | `--mode ghost` | Default | Auto (short messages in Armor) |

**Rule of thumb:** If the image stays intact → Ghost. If it gets recompressed → Armor. If it goes through WhatsApp → Fortress auto-activates.

## Quick Start

```rust
use phasm_core::{ghost_encode, ghost_decode};

let cover = std::fs::read("photo.jpg").unwrap();
let stego = ghost_encode(&cover, "secret message", "passphrase").unwrap();
let decoded = ghost_decode(&stego, "passphrase").unwrap();
assert_eq!(decoded.text, "secret message");
```

```rust
use phasm_core::{armor_encode, armor_decode};

let cover = std::fs::read("photo.jpg").unwrap();
let stego = armor_encode(&cover, "secret message", "passphrase").unwrap();
let (decoded, quality) = armor_decode(&stego, "passphrase").unwrap();
assert_eq!(decoded.text, "secret message");
println!("Integrity: {:.0}%", quality.integrity_percent);
```

```rust
use phasm_core::{
    deep_cover_jpeg, ghost_bands_si, ghost_encode_si, ghost_decode, JpegImage,
    DEEP_COVER_DEFAULT_QUALITY,
};

// Deep Cover (SI-UNIWARD): raw RGB pixels (row-major, 3 bytes per pixel)
// decoded from a PNG / HEIC / camera photo by your image library
let (raw_rgb, width, height): (Vec<u8>, u32, u32) = load_rgb("photo.png");

// Core writes the cover JPEG, so the side information is exact
let cover = deep_cover_jpeg(&raw_rgb, width, height, DEEP_COVER_DEFAULT_QUALITY).unwrap();

// Capacity bar: the same pixels unlock the Deep Cover bands (max nzAC/4)
let img = JpegImage::from_bytes(&cover).unwrap();
let bands = ghost_bands_si(&img, &raw_rgb, width, height, 0, 0).unwrap();
println!("comfort {} B, max {} B", bands.comfort_bytes, bands.max_bytes);

let stego = ghost_encode_si(
    &cover, &raw_rgb, width, height, "secret message", "passphrase"
).unwrap();
// Decode is identical — no special decoder needed
let decoded = ghost_decode(&stego, "passphrase").unwrap();
assert_eq!(decoded.text, "secret message");
```

```rust
use phasm_core::{ghost_encode_with_shadows, ghost_decode, ghost_shadow_decode, ShadowLayer};

let cover = std::fs::read("photo.jpg").unwrap();
let shadows = vec![
    ShadowLayer {
        message: "decoy message".into(),
        passphrase: "decoy_pass".into(),
        files: vec![],
    },
];
let stego = ghost_encode_with_shadows(
    &cover, "real secret", &[], "real_pass", &shadows, None
).unwrap();

// Primary message — with the real passphrase
let primary = ghost_decode(&stego, "real_pass").unwrap();
assert_eq!(primary.text, "real secret");

// Shadow message — with the decoy passphrase
let shadow = ghost_shadow_decode(&stego, "decoy_pass").unwrap();
assert_eq!(shadow.text, "decoy message");
```

## API

```rust
// Ghost mode (stealth)
ghost_encode(jpeg_bytes, message, passphrase) -> Result<Vec<u8>>
ghost_encode_with_quality(jpeg_bytes, message, passphrase) -> Result<(Vec<u8>, EncodeQuality)>
ghost_decode(jpeg_bytes, passphrase) -> Result<PayloadData>
ghost_encode_with_files(jpeg_bytes, message, files, passphrase) -> Result<Vec<u8>>
ghost_encode_with_files_quality(jpeg_bytes, message, files, passphrase) -> Result<(Vec<u8>, EncodeQuality)>
ghost_capacity(jpeg_image) -> Result<usize>
ghost_bands(jpeg_image, shadow_count, shadow_total_bytes) -> Result<GhostBands>

// Ghost shadow messages (plausible deniability)
ghost_encode_with_shadows(jpeg_bytes, message, files, passphrase, shadows, si) -> Result<Vec<u8>>
ghost_encode_with_shadows_quality(jpeg_bytes, message, files, passphrase, shadows, si) -> Result<(Vec<u8>, EncodeQuality)>
ghost_encode_si_with_shadows(jpeg_bytes, raw_rgb, width, height, message, files, passphrase, shadows) -> Result<Vec<u8>>
ghost_encode_si_with_shadows_quality(jpeg_bytes, raw_rgb, width, height, message, files, passphrase, shadows) -> Result<(Vec<u8>, EncodeQuality)>
ghost_shadow_decode(stego_bytes, passphrase) -> Result<PayloadData>
estimate_shadow_capacity(jpeg_image) -> Result<usize>
estimate_shadow_capacity_si(jpeg_image, raw_rgb, width, height) -> Result<usize>
shadow_capacity(y_nzac) -> usize
ghost_capacity_with_shadows(jpeg_image, shadow_count, shadow_total_bytes, is_si) -> Result<usize>  // is_si is ignored

// Ghost SI-UNIWARD "Deep Cover" (stealth + side-informed, ~25% more capacity with exact SI)
deep_cover_jpeg(raw_rgb, width, height, quality) -> Result<Vec<u8>>  // the cover JPEG (= libjpeg-turbo cjpeg)
ghost_encode_si(jpeg_bytes, raw_rgb, width, height, message, passphrase) -> Result<Vec<u8>>
ghost_encode_si_with_quality(jpeg_bytes, raw_rgb, width, height, message, passphrase) -> Result<(Vec<u8>, EncodeQuality)>
ghost_encode_si_with_files(jpeg_bytes, raw_rgb, width, height, message, files, passphrase) -> Result<Vec<u8>>
ghost_encode_si_with_files_quality(jpeg_bytes, raw_rgb, width, height, message, files, passphrase) -> Result<(Vec<u8>, EncodeQuality)>
ghost_bands_si(jpeg_image, raw_rgb, width, height, shadow_count, shadow_total_bytes) -> Result<GhostBands>
ghost_capacity_si(jpeg_image, raw_rgb, width, height) -> Result<usize>
ghost_encode_deep_cover(raw_rgb, width, height, quality, message, files, passphrase) -> Result<(Vec<u8>, EncodeQuality)>
ghost_encode_deep_cover_with_shadows(raw_rgb, width, height, quality, message, files, passphrase, shadows) -> Result<(Vec<u8>, EncodeQuality)>
ghost_capacity_deep_cover(raw_rgb, width, height, quality) -> Result<usize>
DEEP_COVER_DEFAULT_QUALITY = 92

// Armor mode (robust)
armor_encode(jpeg_bytes, message, passphrase) -> Result<Vec<u8>>
armor_encode_with_quality(jpeg_bytes, message, passphrase) -> Result<(Vec<u8>, EncodeQuality)>
armor_decode(jpeg_bytes, passphrase) -> Result<(PayloadData, DecodeQuality)>
armor_capacity(jpeg_image) -> Result<usize>
armor_capacity_info(jpeg_bytes) -> Result<ArmorCapacityInfo>

// Unified decode (auto-detects mode)
smart_decode(jpeg_bytes, passphrase) -> Result<(PayloadData, DecodeQuality)>

// Capacity estimation with Brotli compression
compressed_payload_size(text, files) -> usize

// Image dimension validation
validate_encode_dimensions(width, height) -> Result<()>

// Real-time progress tracking
progress::init(total)      // reset and set total steps
progress::advance()        // increment step (capped at total)
progress::finish()         // mark complete (step = total)
progress::get() -> (u32, u32) // read (step, total)
progress::cancel()         // request cancellation
progress::check_cancelled() -> Result<()> // returns Err(Cancelled) if set
```

### Types

- **`PayloadData`** — decoded message text + optional file attachments
- **`FileEntry`** — file attachment (filename + content bytes)
- **`ShadowLayer`** — shadow message for plausible deniability (message + passphrase + optional files)
- **`EncodeQuality`** — encode-time score (0-100), contextual hint key, mode (stealth for Ghost, robustness for Armor)
- **`DecodeQuality`** — signal integrity percentage, RS error count/capacity, fortress flag
- **`GhostBands`** — Ghost capacity-bar bands in plaintext bytes (`max_bytes`, `comfort_bytes`, `caution_bytes`)
- **`ArmorCapacityInfo`** — capacity breakdown by encoding tier (Phase 1/2/3, Fortress)

The SI-UNIWARD functions (`ghost_encode_si*`, `ghost_bands_si`, `ghost_capacity_si`, `estimate_shadow_capacity_si`) accept additional parameters for the raw uncompressed pixels (`raw_rgb: &[u8]`, `width: u32`, `height: u32`). The side information is exact, and the Deep Cover capacity applies, only when the cover JPEG is `deep_cover_jpeg`'s output for those same pixels; any other cover gets approximate side information and standard Ghost capacity. The decoder does not need side information — `ghost_decode` and `smart_decode` work for both standard and SI-UNIWARD encoded images.

The capacity bands are policy for the apps' capacity bar (comfort ≤ max/4 with a square-root cap on large photos, caution ≤ max/2, max = nzAC/5 frame bits, nzAC/4 with exact side information). The encoder itself enforces only technical feasibility.

## Video steganography (H.264)

H.264 video stego has one encoder and one decoder:

- **Encode** (`h264-encoder`): the [phasm-openh264](https://github.com/cgaffga/phasm-openh264)
  fork of Cisco's OpenH264 codec (BSD-2-Clause), statically linked via
  the `core-openh264-sys` workspace member (`openh264-sys/`, built with
  meson + ninja from the `vendor/phasm-openh264` submodule, pinned by SHA
  in `openh264-sys/build.rs`). A 1080p × 30-frame stego encode takes
  about 1.3 s. The stego bits reach the bitstream as overrides at CABAC
  emit time; the encoder's own state (residuals, motion vectors) is not
  modified.
- **Decode** (`h264-decoder`): a pure-Rust CABAC walker with no C
  dependency, which reads any Annex-B input. Decode-only builds use it
  without linking OpenH264.

Both sides share one wire format through the streaming sessions
(`StreamingEncodeSession` / `StreamingDecodeSession`, see
`src/codec/h264/streaming_session.rs`): the same 4-domain canonical
cover ordering (CS → CSL → MVDs → MVDsl), per-GOP `chunk_frame`
protocol, STC seed derivation and `CostWeights`. Shadow messages work
in video too; encode, verify and decode run one GOP at a time, so memory
is bounded by the GOP, not by the clip length. The CLI's `phasm decode`
(`cli/src/decode.rs::decode_h264_cabac`) decodes video through
`StreamingDecodeSession`; `phasm video-encode` and `phasm
video-capacity` encode and measure. These two subcommands exist only in
builds with an encoder (`h264-encoder` or `av1-encoder`); a decode-only
build (`h264-decoder`) lists just `encode`, `decode` and `capacity`, and
`--codec` offers only the compiled-in codecs.

`phasm video-capacity` prints the same capacity bands as the apps (comfort,
and max = the red end) plus the technical STC ceiling, which is not a
recommendation: above the max the video is visibly damaged. `phasm
video-encode` does not enforce the bands (like image Ghost) but warns on
stderr when the payload exceeds the max.

`phasm video-encode` writes its H.264 MP4 with the in-tree muxer in the
shape of a libavformat transcode with the libopenh264 encoder
(`MuxerProfile::LavfOpenH264`): a container that agrees with the OpenH264
bitstream inside it, carrying the source's audio interleaved with the
video and none of the source file's timestamps or names. (Releases up to
0.4.3 wrote a HandBrake/x264-style container with a synthesised x264 SEI
banner, which the OpenH264 bitstream contradicted.)

An earlier pure-Rust H.264 encoder and its CAVLC pipeline have been
retired. The feature name `h264-encoder` now denotes the OpenH264
encoder (it was called `openh264-backend` before).

In the Phasm apps, video encoding is experimental and off by default
(Settings → "Enable video (experimental)"); decoding a stego video
always works. Video is not available in the web tool (WASM).

If `vendor/phasm-openh264` is absent, `openh264-sys` builds against
fail-fast stubs: the build prints a `cargo:warning=...`, and the encoder
calls return errors at runtime.

### Distribution (patent-pool posture)

H.264 encoders fall under the
[Via LA AVC patent pool](https://www.via-la.com/licensing-2/avc-h-264/).
Source-only distribution is royalty-free; binary distribution above the
pool's free tier is not. Phasm therefore ships no H.264 encoder outside
the mobile apps:

- the crates.io package does not declare the `h264-encoder` feature;
- the pre-built `phasm` CLI binaries on GitHub Releases (installers in
  the CLI section below) contain no video code at all;
- this repository carries the encoder as **source**: a recursive clone
  (or a git dependency) builds it, with the OpenH264 fork as a pinned
  submodule ([BUILDING.md](BUILDING.md));
- video **decode** alone builds from source without any C code:

```bash
cargo install --git https://github.com/cgaffga/phasmcore phasm-cli --features h264-decoder
phasm decode stego.mp4 -p "passphrase"
```

Mobile-app builds include the encoder and are subject to the patent-pool
terms applicable to those distribution channels, not to source consumers
of this crate. Whoever distributes a binary built with `h264-encoder`
takes on the AVC patent-pool obligations for it; review the Via LA AVC
license terms before you publish one.

## Video steganography (AV1 — Phase A, EXPERIMENTAL)

Phase A landed 2026-05-21 (v0.3-AV1 architectural ship: W2 dav1d hook
foundation, W3 rav1e WriterStego + cursor parity, W4 real-content
corpus round-trip, W5 regression-bound drift gate, W6 self-steganalyzer
v0.3, W7 cross-arch determinism via Rosetta 2). End-to-end stego
round-trip via the **dav1d** decoder (`av1-decoder`) and **rav1e**
encoder (`av1-encoder`) through the
[phasm-dav1d](https://github.com/cgaffga/phasm-dav1d) and
[phasm-rav1e](https://github.com/cgaffga/phasm-rav1e) forks. Both
codecs are royalty-free per the
[AOM AV1 patent license](https://aomedia.org/license/patent-license/) —
no Via LA-style patent-pool constraint, unlike H.264.

The AV1 code lives under `src/codec/av1/` + `dav1d-sys/`. phasm-dav1d is
a pinned submodule (`vendor/phasm-dav1d`); phasm-rav1e is consumed from
GitHub source only (never published to crates.io) and patched in at a
pinned commit, so a recursive clone builds AV1 with `make build-video`
(see [BUILDING.md](BUILDING.md), "The rav1e fork"). The features are **not
in the crates.io package**: dav1d needs the submodule's sources, which a
registry package cannot carry, and the rav1e fork needs a `[patch]` that a
registry package cannot give its users. Stealth calibration vs the H.264
production path is still maturing.

## Cargo Features

| Feature | Description |
|---------|-------------|
| `parallel` | Enables [Rayon](https://github.com/rayon-rs/rayon) parallelism for J-UNIWARD cost computation, STC embedding, and Armor decode sweeps. Recommended for native builds. |
| `simd` | A NEON kernel for the H.264 stego walker (aarch64). The image-side SIMD kernels are always built. |
| `video` | MP4 demux/mux and H.264 bitstream parsing; the base of the H.264 features. See "Video steganography (H.264)" above. |
| `h264-decoder` | H.264 video stego decode: pure-Rust CABAC walker + streaming decode session, no C dependency. Implies `video`. |
| `h264-encoder` | H.264 video stego encode via the `phasm-openh264` fork (BSD-2-Clause, statically linked, needs meson + ninja). Implies `h264-decoder`, `parallel` and `simd`. Builds against fail-fast stubs if `vendor/phasm-openh264` is absent (warns at build time, returns errors at runtime). Builds from a recursive clone; not in the crates.io package (AVC patent pool). |
| `av1-decoder` | dav1d decoder backend via the [`phasm-dav1d`](https://github.com/cgaffga/phasm-dav1d) fork (BSD-2-Clause, royalty-free per AOM AV1). EXPERIMENTAL. Builds from a recursive clone (`vendor/phasm-dav1d`; meson + ninja); not in the crates.io package. |
| `av1-encoder` | rav1e encoder via the [`phasm-rav1e`](https://github.com/cgaffga/phasm-rav1e) fork (BSD-2-Clause, royalty-free per AOM AV1), patched in from GitHub (`make build-video`, BUILDING.md). Implies `av1-decoder` and `parallel`. EXPERIMENTAL; same posture as `av1-decoder`. |
| `wasm` | WASM bridge support via `wasm-bindgen` + `js-sys`. |

## CLI

A command-line interface is available in the `cli/` directory.

### Install

```bash
# macOS / Linux
curl --proto '=https' --tlsv1.2 -LsSf https://github.com/cgaffga/phasmcore/releases/latest/download/phasm-cli-installer.sh | sh

# Windows PowerShell
powershell -c "irm https://github.com/cgaffga/phasmcore/releases/latest/download/phasm-cli-installer.ps1 | iex"

# Or build from source (in a recursive clone; see BUILDING.md)
cargo install --path cli                                        # image stego
cargo install --path cli --features h264-encoder,av1-encoder    # + H.264 and AV1 video
```

### Usage

```bash

# Encode (Armor mode, default)
phasm encode photo.jpg -m "secret message" -p "passphrase"

# Encode (Ghost mode)
phasm encode photo.jpg --mode ghost -m "secret" -p "pass" -o secret.jpg

# PNG input → auto Deep Cover (SI-UNIWARD)
phasm encode photo.png --mode ghost -m "hello" -p "pass"

# Strip the metadata (GPS, camera, serial numbers…), like the apps' magic wand.
# Without --optimize the output keeps it, and encode lists what is kept.
phasm encode photo.jpg --mode ghost -m "hello" -p "pass" --optimize

# Shadow messages (plausible deniability)
phasm encode photo.jpg --mode ghost -m "real" -p "main" --shadow-message-2 "decoy" --shadow-passphrase-2 "shadow"

# File attachments
phasm encode photo.jpg --mode ghost -m "see attached" -p "pass" --attach plans.pdf

# Decode (auto-detects mode)
phasm decode stego.jpg -p "passphrase"

# Extract file attachments
phasm decode stego.jpg -p "pass" --extract ./output/

# Show capacity (Ghost: the apps' comfort / caution / max bands; display only)
phasm capacity photo.jpg
# Ghost:   comfort 2563 bytes, caution 47863 bytes, max 95773 bytes
# Armor:   5210 bytes (Fortress: 135 bytes) on the 1200x1600 cover
# Shadows: 47115 bytes per layer

# Pipe message from stdin
echo "hello" | phasm encode photo.jpg -p "pass"

# Machine-readable output
phasm encode photo.jpg -m "hello" -p "pass" --json
phasm decode stego.jpg -p "pass" --quiet
```

See `phasm encode --help`, `phasm decode --help`, `phasm capacity --help` for all options.

## Building & Testing

Prerequisites, the full feature matrix, `make` targets and troubleshooting: [BUILDING.md](BUILDING.md).

```bash
cargo test -p phasm-core                    # default (single-threaded)
cargo test -p phasm-core --features parallel # with Rayon parallelism
```

### Examples

```bash
# Encode and decode a message
cargo run -p phasm-core --example test_encode -- photo.jpg "Hello" "passphrase"
cargo run -p phasm-core --example test_encode -- --decode stego.jpg "passphrase"

# Timing benchmark
cargo run -p phasm-core --example test_timing -- stego.jpg

# Quick decode test
cargo run -p phasm-core --example test_link -- stego.jpg
```

## Architecture

### Design Principles

- **Zero C FFI** — pure Rust from JPEG parsing to AES encryption, compiles to native and WASM
- **Deterministic** — identical output across x86, ARM, and WASM using a [pure-IEEE 754 polynomial math module](https://phasm.app/blog/deterministic-cross-platform-math-wasm) (no `f64::sin`/`cos` — they compile to non-deterministic `Math.*` in WASM)
- **Memory-efficient** — a Ghost encode needs ≈ 7.25 bytes per pixel of fixed buffers (parsed JPEG, keyed order, output) plus ≈ 74 bytes per position of each lane in flight (64 of them the h = 9 trellis back pointers). Costs are computed strip-wise and gathered per lane group, never held for the whole image. Given a memory budget, the encoder walks a ladder: as many in-memory lanes at once as fit, then one lane at a time, then checkpointed lanes (O(√n) memory, twice the trellis work); the output is byte-identical on every rung. Peak stays under 1 GB up to ~100 MP and is ~1.4 GB at 200 MP; measured on an iPhone 16 Pro, a 12 MP encode peaks at 629 MB and a 48 MP encode at 1.03 GB.
- **Any payload size** — Ghost carries anything from a few characters to hundreds of KB, bounded by the cover's capacity; Armor and Fortress target short messages
- **Stego output is raw** — the JPEG bytes after encoding must be saved/shared without re-encoding

### JPEG Codec

The [`jpeg`](src/codec/jpeg/) module is a [from-scratch JPEG coefficient codec](https://phasm.app/blog/pure-rust-jpeg-coefficient-codec) — zero dependencies beyond `std`. It parses JPEG files into DCT coefficient grids, allows modification, and writes them back with **byte-for-byte round-trip fidelity**. Supports both baseline (SOF0) and progressive (SOF2) JPEG input, always outputs baseline.

Original Huffman tables are preserved from the cover image (with fallback to rebuild if needed).

The Deep Cover cover writer (`cover_writer.rs`) encodes RGB pixels to a baseline JPEG that is **byte-identical to libjpeg-turbo `cjpeg -quality Q`** (default options), integer arithmetic only, with identical output on aarch64, x86_64 and wasm32. It also reports every luma coefficient before quantization, which is where Deep Cover's exact side information comes from. The cover carries the most common JPEG signature there is (IJG tables, standard Huffman tables, JFIF) and no phasm-specific marker.

### Cryptography

All payloads are encrypted before embedding:

- **Key derivation**: Argon2id (RFC 9106) — two tiers:
  - *Structural key* (deterministic from passphrase + fixed salt): drives coefficient permutation and STC matrix generation
  - *Encryption key* (passphrase + random salt): AES-256-GCM-SIV (RFC 8452) with nonce-misuse resistance
- **PRNG**: ChaCha20 for all key-derived randomness (permutations, spreading vectors, template peaks)
- **Payload compression**: Brotli (RFC 7932) for compact payloads; flags byte indicates compression

### Ghost Pipeline

1. Parse JPEG into DCT coefficients
2. Derive structural key (Argon2id, overlapped with step 3 on multi-core) → seeds for the keyed coefficient order and the STC matrices (domain-separated from the v1 seeds, so a v3 image never parses as v1 or vice versa)
3. Compute [J-UNIWARD costs](https://phasm.app/blog/uerd-vs-juniwird-detection-benchmarks) for every Y AC coefficient **strip-by-strip** (Daubechies-8 wavelet, 3 subbands, parallel row+column filtering) — costs are gathered per lane group, never held for the whole image
4. *(SI-UNIWARD only)* Modulate the costs with the rounding errors (`i8` codes): with exact side information a change toward the pre-quantization value costs `cost × (1 − 2|e|)`, a change away the full cost, and exact ties stay unmodulated; approximate side information floors the factor at 0.3
5. Order every Y AC coefficient, zeros included, with a keyed Fisher-Yates shuffle (ChaCha20)
6. Encrypt payload (AES-256-GCM-SIV) and frame (length + CRC)
7. **Lanes**: split the order and the frame bits into `L = clamp(min(⌊n / 2²⁰⌋, ⌊m / 4096⌋), 1, 64)` contiguous lanes (`n` AC coefficients, `m` frame bits), planned in parallel
8. **Ternary embedding per lane** (double-layered STC, Filler–Judas–Fridrich 2011): half of a lane's bits go into the second bit plane `⌊c/2⌋ mod 2` (layer 2, with costs from the marginal flip probabilities of the optimal ternary embedding), the rest into the parity `c mod 2` (layer 1). Each layer is a binary [STC](https://phasm.app/blog/syndrome-trellis-codes-practical-guide) with constraint height **h = 9** and the widest width of a fixed ladder (1, 2, …, 16, then ×1.2 steps) that fits the lane. The ±1 direction of a change carries message bits, so zeros may become ±1 and ±1 may become 0
9. *(Shadow mode)* Every shadow region is wet (infinite cost) for the primary
10. Write modified coefficients back to JPEG (with progress reporting per MCU row)

The decoder needs no costs. For each lane count and width the cover admits, it reads lane 0's 48-bit frame header and accepts it only if the header is consistent with that lane count and width; everything else is then determined, and the CRC and the AES-GCM-SIV tag validate. After v3 it tries the v1 decoder (binary STC on the non-zero AC set, h = 7, widths 1–10), so images written by core ≤ 0.4.3 still decode.

#### STC Viterbi Optimizer

The STC encoder uses a Viterbi-style dynamic programming algorithm to find the minimum-cost modification sequence across a trellis with 2^h = 512 states (h = 9; the trellis supports any height up to 12). Each target state has exactly 2 candidate predecessors (stego bit 0 or 1), so one bit per state records the choice: a trellis step's back pointers fit in 2^h / 64 = 8 `u64` words, **64 bytes per position**.

```rust
for s in 0..num_states {
    let a = prev[s] + c0;          // stego bit 0, predecessor s
    let b = prev[s ^ col] + c1;    // stego bit 1, predecessor s ^ col
    let take = b < a;
    cur[s] = if take { b } else { a };
    word |= (take as u64) << (s & 63);   // 1 bit records the choice
}
```

Traceback reconstructs predecessors from the single bit; at message-block boundaries the pre-shift state follows from the known message bit, so no storage is needed:

```rust
state = ((state << 1) | message_bit) & mask;   // invertible shift
let bit = ((rows[k + (state >> 6)] >> (state & 63)) & 1) as u8;
if bit == 1 { state ^= col; }                  // undo XOR to get predecessor
```

Two memory schedules produce **bit-for-bit identical plans**:

- **Inline**: one forward pass that keeps every step's back pointers. A lane of ~2²⁰–2²¹ positions needs 64–128 MB; this is the default on every platform.
- **Segmented checkpoint**: a forward pass that keeps only the state costs at every K-th message block, then segment by segment from the end: recompute that segment's back pointers from its checkpoint and trace it back. O(√n) memory, twice the trellis work — the lowest rung of the memory ladder, used only when a memory budget requires it.

The ladder (`memory::ghost_lane_plan`) picks how many lanes run at once and which schedule they use; the output bytes are the same on every rung. Measured on an iPhone 16 Pro: 12 MP encode 2.4 s (peak 629 MB), 48 MP encode 10.3 s (peak 1.03 GB), 12 MP decode 0.42 s.

#### Strip-Based UNIWARD

The J-UNIWARD cost function requires wavelet decomposition of the entire decompressed image — three directional subbands (LH, HL, HH) via Daubechies-8 filters. For a 200 MP image, storing the full pixel array + 3 wavelet subbands would need ~3.2 GB.

The **strip-based streaming** approach eliminates this:

1. Process the image in horizontal strips of block rows
2. Each strip decompresses its pixel rows (with padding for the 16-tap wavelet filter boundary), computes row/column-filtered wavelet subbands, and evaluates per-coefficient costs
3. The costs of the lanes being planned are gathered from the strips — no full cost map is ever materialized
4. Strip memory is freed before the next strip begins

Capacity estimation is instantaneous: the capacity policy is stated in frame bits per non-zero AC coefficient (max = nzAC/5, nzAC/4 with exact side information), so it counts non-zero AC coefficients directly from the DctGrid — no wavelet computation needed.

### Armor Pipeline

1. Parse JPEG, derive structural keys
2. Compute stability map (select low-frequency AC coefficients)
3. Encrypt and frame payload with adaptive RS parity
4. Embed via STDM with spreading vectors (ChaCha20-derived)
5. For short messages: activate Fortress ([BA-QIM](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) on block averages with [Watson masking](https://phasm.app/blog/watson-perceptual-masking-qim-steganography))
6. Embed [DFT magnitude template](https://phasm.app/blog/dft-template-geometric-resilience-steganography) for geometric resilience
7. Decode: three-phase parallel sweep with [soft-decision concatenated codes](https://phasm.app/blog/soft-majority-voting-llr-concatenated-codes)

### Progress Reporting

Both encode and decode pipelines report real-time progress via global atomics (`progress::advance()`). This enables responsive UI feedback on all platforms:

- **Ghost encode**: 166 steps (`GHOST_ENCODE_STEPS`) — 5 (parse) + 100 (UNIWARD cost pass) + 1 (key + order) + 40 (the trellises of all lanes, by trellis work, from any thread) + 20 (JPEG write MCU rows)
- **Ghost encode with shadows**: 167 steps (`GHOST_ENCODE_WITH_SHADOWS_STEPS`) plus 5 per shadow trellis; a retried shadow layout raises the total at runtime
- **Ghost decode**: 9 steps (`GHOST_DECODE_STEPS`) — 5 (parse) + 2 (v3 order + search) + 2 (v1 permutation + extraction). No cost computation.
- **Armor encode**: 6 steps (STDM path) or 3 steps (Fortress path, auto-adjusted at runtime)
- **Armor decode**: dynamic total based on candidate count (~50 steps). Phase 1 delta sweep parallelized across ~21 candidates.

Cancellation is cooperative: `progress::cancel()` sets a flag checked at natural loop boundaries via `progress::check_cancelled()`; every Ghost trellis checks it every 64 k positions.

### FFT

In-house Cooley-Tukey + Bluestein FFT implementation using deterministic twiddle factors (`det_sincos()`). No external FFT crate — guarantees [bit-identical results across all platforms](https://phasm.app/blog/deterministic-cross-platform-math-wasm).

## Project Structure

```
src/
  lib.rs            Public API re-exports
  det_math.rs       Deterministic math (sin/cos/atan2/hypot/exp/powi)
  codec/
    jpeg/
      mod.rs          JpegImage: parse, modify, serialize
      bitio.rs        Bit-level reader/writer with JPEG byte stuffing
      cover_writer.rs Deep Cover cover writer (byte-identical to libjpeg-turbo cjpeg)
      dct.rs          DCT coefficient grids and quantization tables
      error.rs        JPEG parsing errors
      frame.rs        SOF frame info (dimensions, components, subsampling)
      huffman.rs      Huffman coding tables (two-level decode, encode)
      marker.rs       JPEG marker iterator
      pixels.rs       IDCT/DCT for pixel-domain operations (forward DCT, luma extraction)
      scan.rs         Entropy-coded scan reader/writer
      tables.rs       DQT/DHT table parsing
      zigzag.rs       Zigzag scan order mapping
    h264/           H.264 parsing, CABAC walker, streaming stego sessions, OpenH264 glue (video features)
    mp4/            MP4/MOV demuxer and muxer (video features)
    av1/            AV1 stego, experimental (AV1 features)
  stego/
    mod.rs          Ghost/Armor encode/decode entry points, smart_decode
    crypto.rs       AES-256-GCM-SIV + Argon2id key derivation
    frame.rs        Payload framing (length, CRC, mode byte, salt, nonce)
    payload.rs      Payload serialization (Brotli, file attachments)
    permute.rs      Fisher-Yates coefficient permutation
    memory.rs       Memory budget + Ghost lane ladder
    progress.rs     Real-time progress tracking (atomics + WASM callback)
    error.rs        StegoError enum
    ghost/
      pipeline.rs   Ghost orchestration (encode, decode order, v1 decoder)
      v3.rs         Ghost v3: ternary STC over all Y AC coefficients, lanes
      deep_cover.rs Deep Cover: pixels in, writer cover + exact side information
      side_info.rs  SI-UNIWARD: rounding errors (exact / approximate), cost modulation
      shadow.rs     Shadow messages (keyed-region STC; layout v1 decode)
      capacity.rs   Ghost capacity and capacity-bar bands
      quality.rs    Encode quality scoring (stealth for Ghost, robustness for Armor)
    cost/
      mod.rs        Cost function trait
      uniward.rs    J-UNIWARD (Daubechies-8 wavelet, 3 subbands)
      uniward_legacy_v1.rs  v1 J-UNIWARD costs (layout v1 shadow fallback only)
      uerd.rs       UERD cost function (legacy)
    stc/
      mod.rs        Syndrome-Trellis Codes
      trellis.rs    Viterbi for any constraint height (inline + segmented schedules)
      embed.rs      Binary STC Viterbi embedding (h <= 7)
      extract.rs    STC extraction (syndrome computation)
      hhat.rs       H-hat submatrix generation (ChaCha20-derived)
    armor/
      mod.rs        Armor mode re-exports
      pipeline.rs   Armor encode/decode (STDM + Fortress + DFT template)
      embedding.rs  STDM embed/extract with adaptive delta
      selection.rs  Coefficient stability map
      spreading.rs  ChaCha20-derived spreading vectors
      ecc.rs        Reed-Solomon GF(2^8) encoder/decoder
      repetition.rs Repetition coding with soft majority voting
      capacity.rs   Armor capacity estimation
      fortress.rs   BA-QIM block-average embedding + Watson masking
      template.rs   DFT magnitude template (geometric resilience)
      dft_payload.rs DFT ring-based payload embedding
      fft2d.rs      2D FFT (Cooley-Tukey + Bluestein)
      resample.rs   Bilinear resampling for geometric correction
cli/                Command-line interface (phasm binary)
openh264-sys/       OpenH264 fork build + FFI (h264-encoder)
dav1d-sys/          dav1d fork build + FFI (AV1)
test-vectors/       Synthetic JPEG test images
tests/              Integration tests (round-trip, cross-platform, geometry)
examples/           CLI tools (encode/decode, timing, diagnostics)
```

## Research & Publications

The algorithms in phasm-core are documented in detail on the [Phasm blog](https://phasm.app/blog):

### Steganography Fundamentals
- [Adaptive Steganography at Low Embedding Rates: UERD vs J-UNIWARD Detection Benchmarks](https://phasm.app/blog/uerd-vs-juniwird-detection-benchmarks)
- [Syndrome-Trellis Codes for Practical JPEG Steganography](https://phasm.app/blog/syndrome-trellis-codes-practical-guide)
- [Surviving JPEG Recompression: A Quantitative Analysis of DCT-Domain Robust Steganography](https://phasm.app/blog/surviving-jpeg-recompression)

### Robustness & Error Correction
- [Three Invariants of JPEG Recompression: Block Averages, Brightness Ordering, and Coefficient Signs](https://phasm.app/blog/jpeg-recompression-invariants)
- [Soft Decoding for Steganography: How LLRs and Concatenated Codes Turn 19% BER Into Zero Errors](https://phasm.app/blog/soft-majority-voting-llr-concatenated-codes)
- [Perceptual Masking Meets QIM: Adaptive Embedding Strength for Invisible Robust Steganography](https://phasm.app/blog/watson-perceptual-masking-qim-steganography)

### Geometric Resilience
- [Fourier-Domain Template Embedding: How Phasm Survives Rotation, Scaling, and Cropping](https://phasm.app/blog/dft-template-geometric-resilience-steganography)

### Plausible Deniability
- [Shadow Messages: How Phasm Hides Multiple Secrets in a Single Photo](https://phasm.app/blog/shadow-messages-plausible-deniability-steganography)

### Implementation
- [Building a Pure-Rust JPEG Coefficient Codec](https://phasm.app/blog/pure-rust-jpeg-coefficient-codec)
- [When f64::sin() Breaks Your Crypto: Building Deterministic Math for WASM Steganography](https://phasm.app/blog/deterministic-cross-platform-math-wasm)
- [From 600 MB to 3 MB: Fitting Viterbi Steganography on a Phone](https://phasm.app/blog/segmented-viterbi-memory-efficient-stc)

### Platform Analysis
- [How 15 Messaging Platforms Process Your Photos](https://phasm.app/blog/how-15-platforms-process-your-photos)

## How Does Phasm Compare?

| Feature | Phasm | OpenStego | Steghide | F5 |
|---------|-------|-----------|----------|----|
| **Status** | Active (2025–present) | Unmaintained | Unmaintained (2003) | Unmaintained |
| **Language** | Rust (native + WASM) | Java | C++ | Java |
| **Encryption** | AES-256-GCM-SIV + Argon2id | DES | Blowfish / AES | None |
| **Embedding** | J-UNIWARD + ternary STC (h=9) | LSB | LSB-adjacent | DCT histogram shift |
| **Steganalysis resistance** | High (measured against SRNet) | Low | Low | Moderate |
| **Survives recompression** | Yes (Armor / Fortress) | No | No | No |
| **Survives WhatsApp** | Yes (Fortress, short messages) | No | No | No |
| **Plausible deniability** | Yes (shadow messages) | No | No | No |
| **Side-informed mode** | Yes (SI-UNIWARD with exact side information, +25% capacity) | No | No | No |
| **File attachments** | Yes (Ghost, up to 2 MB) | No | Yes | No |
| **Geometric resilience** | Yes (DFT template) | No | No | No |
| **Platforms** | iOS, Android, Web (WASM), CLI | Desktop (Java) | Linux / Windows CLI | Java |
| **Web (no install)** | Yes (WebAssembly) | No | No | No |
| **Image format** | JPEG (any input auto-converted) | PNG, BMP | JPEG, BMP, WAV, AU | JPEG |
| **Open source** | GPL-3.0 | GPL-2.0 | GPL-2.0 | MIT |

## FAQ

**Does Phasm work with WhatsApp?**
Yes. Fortress sub-mode (auto-activates for short messages in Armor mode) survives WhatsApp's default recompression pipeline. For longer messages, send the image as a document/file to bypass compression entirely.

**How detectable is Ghost mode?**
It depends on how much you hide. The capacity bar's comfort band (max/4) stays close to chance for every detector measured (CNN detection error about 0.40, 0.5 = random guessing). At the maximum (0.20 bits per non-zero AC coefficient), a deep-learning detector trained on Ghost does better than chance: SRNet's detection error is 0.272 at QF75 (BOSSbase 1.01, trained from scratch; the earlier design scored 0.190) and 0.366 at QF92. Deep Cover (SI-UNIWARD with exact side information, auto-activated for non-JPEG inputs) is harder to detect even at its larger maximum: 0.450 at QF92.

**What image sizes are supported?**
Minimum 200 pixels per side, maximum 16384 pixels per side and 200 megapixels (`validate_encode_dimensions`); the apps downsample larger images first. A Ghost encode peaks under 1 GB up to ~100 MP and at ~1.4 GB at 200 MP, and a memory budget lowers the lane part of that without changing the output.

**Is the output JPEG or PNG?**
Always JPEG. Input can be any common format (JPEG, PNG, WebP, HEIC) — non-JPEG inputs are auto-converted and enable SI-UNIWARD Deep Cover in Ghost mode.

**Can I hide files, not just text?**
Yes, in Ghost mode. File attachments up to 2 MB per file are supported alongside text messages. Files are Brotli-compressed before embedding.

**What happens if I lose my passphrase?**
The message cannot be recovered. Phasm uses AES-256-GCM-SIV with Argon2id key derivation — no backdoor, no recovery mechanism. Only the correct passphrase can decrypt the message.

**Can someone prove a message exists without the passphrase?**
Ghost mode is designed to resist this. Shadow messages add plausible deniability — you can reveal a decoy passphrase that decodes a harmless message, with no way to prove additional messages exist.

**Does Phasm work offline?**
Yes. All processing happens on-device. The web version runs entirely in WebAssembly — no server communication, no account required.

## Acknowledgements

Phasm stands on other people's work. Thank you to:

- **Holub, Fridrich and Denemark** for [J-UNIWARD and SI-UNIWARD](https://doi.org/10.1186/1687-417X-2014-1), the distortion functions behind Ghost mode.
- **Filler, Judas and Fridrich** for [Syndrome-Trellis Codes](https://doi.org/10.1109/TIFS.2011.2134094), which embed with minimal distortion.
- **The libjpeg-turbo and Independent JPEG Group authors** for the [JPEG reference work](https://libjpeg-turbo.org) our DCT kernels and cover writer follow.
- **The [RustCrypto](https://github.com/RustCrypto), [Rayon](https://github.com/rayon-rs/rayon) and [rust-brotli](https://github.com/dropbox/rust-brotli) authors** for AES-GCM-SIV, Argon2, parallelism and Brotli compression.
- **The VideoLAN [dav1d](https://code.videolan.org/videolan/dav1d) team**, whose AV1 decoder is used for the experimental video feature.
- **Xiph.Org and the [rav1e](https://github.com/xiph/rav1e) contributors**, whose AV1 encoder is used for the experimental video feature.
- **Cisco's [OpenH264](https://github.com/cisco/openh264)**, the H.264 codec used for the experimental video feature.

Their copyright notices and license texts are in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES); the apps link to the same list at [phasm.app/licenses](https://phasm.app/licenses).

## License

GPL-3.0-only. See [LICENSE](LICENSE).

Third-party dependency licenses (including the AV1 and H.264 video codecs) are listed in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
