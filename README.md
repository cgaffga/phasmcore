# phasm-core

Pure-Rust steganography engine for hiding encrypted text messages in JPEG photos.

This is the core library behind [Phasm](https://phasm.app) — available on [iOS](https://apps.apple.com/app/phasm/id6742044613), Android, and the [web](https://phasm.app).

## Two Embedding Modes

### Ghost (Stealth)

Optimizes for **undetectability**. Uses the [J-UNIWARD](https://phasm.app/blog/uerd-vs-juniwird-detection-benchmarks) cost function to assign distortion costs per DCT coefficient, then embeds via [Syndrome-Trellis Codes (STC)](https://phasm.app/blog/syndrome-trellis-codes-practical-guide) to minimize total distortion. The result resists state-of-the-art steganalysis detectors (SRNet, XedroudjNet) at typical embedding rates.

Use Ghost when the stego image will be stored or transmitted without recompression — the embedding does not survive JPEG re-encoding.

Ghost mode also supports **file attachments** (Brotli-compressed, multi-file).

### Armor (Robust)

Optimizes for **survivability**. Uses Spread Transform Dither Modulation (STDM) to embed bits into stable low-frequency DCT coefficients, protected by [adaptive Reed-Solomon ECC](https://phasm.app/blog/surviving-jpeg-recompression) with [soft-decision decoding via log-likelihood ratios](https://phasm.app/blog/soft-majority-voting-llr-concatenated-codes). A [DFT magnitude template](https://phasm.app/blog/dft-template-geometric-resilience-steganography) provides geometric synchronization against rotation, scaling, and cropping.

Armor is the **default mode** on all platforms.

#### Fortress Sub-Mode

For short messages, Armor automatically activates **Fortress** — a [BA-QIM (Block-Average Quantization Index Modulation)](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) scheme that embeds one bit per 8x8 block into the block-average brightness. Fortress exploits [three invariants of JPEG recompression](https://phasm.app/blog/jpeg-recompression-invariants) — block averages, brightness ordering, and coefficient signs — that persist even through aggressive re-encoding.

Fortress survives WhatsApp recompression (QF ~62, resolution resize to 1600px). See [how 15 messaging platforms process your photos](https://phasm.app/blog/how-15-platforms-process-your-photos) for a comprehensive platform analysis.

[Watson perceptual masking](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) adapts the embedding strength per-block based on local texture energy, keeping modifications invisible in textured regions while protecting smooth areas.

### Decode Auto-Detection

`smart_decode` tries all modes automatically — no mode selector needed on the decode side.

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

## API

```rust
// Ghost mode (stealth)
ghost_encode(jpeg_bytes, message, passphrase) -> Result<Vec<u8>>
ghost_decode(jpeg_bytes, passphrase) -> Result<PayloadData>
ghost_encode_with_files(jpeg_bytes, message, files, passphrase) -> Result<Vec<u8>>
ghost_capacity(jpeg_image) -> Result<usize>

// Armor mode (robust)
armor_encode(jpeg_bytes, message, passphrase) -> Result<Vec<u8>>
armor_decode(jpeg_bytes, passphrase) -> Result<(PayloadData, DecodeQuality)>
armor_capacity(jpeg_image) -> Result<usize>

// Unified decode (auto-detects mode)
smart_decode(jpeg_bytes, passphrase) -> Result<(PayloadData, DecodeQuality)>

// Capacity estimation with Brotli compression
compressed_payload_size(text, mode) -> usize
```

### Types

- **`PayloadData`** — decoded message text + optional file attachments
- **`DecodeQuality`** — signal integrity percentage, RS error count/capacity, fortress flag
- **`ArmorCapacityInfo`** — capacity breakdown by encoding tier (Phase 1/2/3, Fortress)

## Cargo Features

| Feature | Description |
|---------|-------------|
| `parallel` | Enables [Rayon](https://github.com/rayon-rs/rayon) parallelism for J-UNIWARD cost computation, STC embedding, and Armor decode sweeps. Recommended for native builds. |
| `wasm` | WASM bridge support via `wasm-bindgen` + `js-sys`. |

## Building & Testing

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
- **Deterministic** — identical output across x86, ARM, and WASM using [FDLIBM-based math](https://phasm.app/blog/deterministic-cross-platform-math-wasm) (no `f64::sin`/`cos` — they compile to non-deterministic `Math.*` in WASM)
- **Short messages** — optimized for text payloads under 1 KB
- **Stego output is raw** — the JPEG bytes after encoding must be saved/shared without re-encoding

### JPEG Codec

The [`jpeg`](src/jpeg/) module is a [from-scratch JPEG coefficient codec](https://phasm.app/blog/pure-rust-jpeg-coefficient-codec) — zero dependencies beyond `std`. It parses JPEG files into DCT coefficient grids, allows modification, and writes them back with **byte-for-byte round-trip fidelity**. Supports both baseline (SOF0) and progressive (SOF2) JPEG input, always outputs baseline.

Original Huffman tables are preserved from the cover image (with fallback to rebuild if needed).

### Cryptography

All payloads are encrypted before embedding:

- **Key derivation**: Argon2id (RFC 9106) — two tiers:
  - *Structural key* (deterministic from passphrase + fixed salt): drives coefficient permutation and STC matrix generation
  - *Encryption key* (passphrase + random salt): AES-256-GCM-SIV (RFC 8452) with nonce-misuse resistance
- **PRNG**: ChaCha20 for all key-derived randomness (permutations, spreading vectors, template peaks)
- **Payload compression**: Brotli (RFC 7932) for compact payloads; flags byte indicates compression

### Ghost Pipeline

1. Parse JPEG into DCT coefficients
2. Derive structural key (Argon2id) -> permutation seed + STC seed
3. Compute [J-UNIWARD cost map](https://phasm.app/blog/uerd-vs-juniwird-detection-benchmarks) (Daubechies-8 wavelet, 3 subbands)
4. Permute coefficient order (Fisher-Yates with ChaCha20)
5. Encrypt payload (AES-256-GCM-SIV) and frame (length + CRC)
6. Embed via [STC](https://phasm.app/blog/syndrome-trellis-codes-practical-guide) (h=7) minimizing weighted distortion
7. Write modified coefficients back to JPEG

### Armor Pipeline

1. Parse JPEG, derive structural keys
2. Compute stability map (select low-frequency AC coefficients)
3. Encrypt and frame payload with adaptive RS parity
4. Embed via STDM with spreading vectors (ChaCha20-derived)
5. For short messages: activate Fortress ([BA-QIM](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) on block averages with [Watson masking](https://phasm.app/blog/watson-perceptual-masking-qim-steganography))
6. Embed [DFT magnitude template](https://phasm.app/blog/dft-template-geometric-resilience-steganography) for geometric resilience
7. Decode: three-phase parallel sweep with [soft-decision concatenated codes](https://phasm.app/blog/soft-majority-voting-llr-concatenated-codes)

### FFT

In-house Cooley-Tukey + Bluestein FFT implementation using deterministic twiddle factors (`det_sincos()`). No external FFT crate — guarantees [bit-identical results across all platforms](https://phasm.app/blog/deterministic-cross-platform-math-wasm).

## Project Structure

```
src/
  lib.rs            Public API re-exports
  det_math.rs       Deterministic math (FDLIBM sin/cos/atan2/hypot)
  jpeg/
    mod.rs          JpegImage: parse, modify, serialize
    bitio.rs        Bit-level reader/writer with JPEG byte stuffing
    dct.rs          DCT coefficient grids and quantization tables
    frame.rs        SOF frame info (dimensions, components, subsampling)
    huffman.rs      Huffman coding tables (two-level decode, encode)
    marker.rs       JPEG marker iterator
    pixels.rs       IDCT/DCT for pixel-domain operations
    scan.rs         Entropy-coded scan reader/writer
    tables.rs       DQT/DHT table parsing
    zigzag.rs       Zigzag scan order mapping
  stego/
    mod.rs          Ghost/Armor encode/decode entry points
    pipeline.rs     Ghost mode pipeline
    crypto.rs       AES-256-GCM-SIV + Argon2id key derivation
    frame.rs        Payload framing (length, CRC, mode byte, salt, nonce)
    payload.rs      Payload serialization (Brotli, file attachments)
    permute.rs      Fisher-Yates coefficient permutation
    capacity.rs     Ghost capacity estimation
    progress.rs     Real-time decode progress (atomics + WASM callback)
    error.rs        StegoError enum
    cost/
      mod.rs        Cost function trait
      uniward.rs    J-UNIWARD (Daubechies-8 wavelet, 3 subbands)
      uerd.rs       UERD cost function (legacy)
    stc/
      mod.rs        Syndrome-Trellis Codes
      embed.rs      STC embedding (Viterbi forward pass)
      extract.rs    STC extraction (syndrome computation)
      hhat.rs       H-hat submatrix generation
    armor/
      mod.rs        Armor mode re-exports
      pipeline.rs   Armor encode/decode (three-phase parallel sweep)
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

### Implementation
- [Building a Pure-Rust JPEG Coefficient Codec](https://phasm.app/blog/pure-rust-jpeg-coefficient-codec)
- [When f64::sin() Breaks Your Crypto: Building Deterministic Math for WASM Steganography](https://phasm.app/blog/deterministic-cross-platform-math-wasm)

### Platform Analysis
- [How 15 Messaging Platforms Process Your Photos](https://phasm.app/blog/how-15-platforms-process-your-photos)

## License

GPL-3.0-only. See [LICENSE](LICENSE).

Third-party dependency licenses are listed in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
