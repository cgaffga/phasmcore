# phasm-core

Pure-Rust steganography engine for hiding encrypted text messages in JPEG photos.

This is the core library behind [Phasm](https://phasm.app) — available on [iOS](https://apps.apple.com/app/phasm-steganography/id6759446274), Android, and the [web](https://phasm.app).

## Two Embedding Modes

### Ghost (Stealth)

Optimizes for **undetectability**. Uses the [J-UNIWARD](https://phasm.app/blog/uerd-vs-juniwird-detection-benchmarks) cost function to assign distortion costs per DCT coefficient, then embeds via [Syndrome-Trellis Codes (STC)](https://phasm.app/blog/syndrome-trellis-codes-practical-guide) to minimize total distortion. The result resists state-of-the-art steganalysis detectors (SRNet, XedroudjNet) at typical embedding rates.

Use Ghost when the stego image will be stored or transmitted without recompression — the embedding does not survive JPEG re-encoding.

Ghost mode also supports **file attachments** (Brotli-compressed, multi-file).

#### Shadow Messages (Plausible Deniability)

Ghost mode supports **[shadow messages](https://phasm.app/blog/shadow-messages-plausible-deniability-steganography)** — multiple messages hidden in a single image, each with a different passphrase. If coerced into revealing a passphrase, you can give the shadow passphrase instead of the primary one. The adversary decodes a decoy message and has no way to detect whether additional messages exist.

Shadows use **Y-channel direct LSB embedding** with Reed-Solomon error correction. Key design properties:

- **Cost-pool position selection** — shadow positions are drawn from the cheapest UNIWARD-cost regions via two-tier filtering (top-N% cost pool + keyed ChaCha20 permutation), ensuring modifications land in textured areas for maximum stealth
- **∞-cost protection** — when the primary message uses dynamic w ≥ 2, shadow positions receive `f32::INFINITY` cost in the STC Viterbi trellis, routing the primary encoder around them with **BER ≈ 0%**
- **Headerless brute-force decode** — no magic bytes or agreed-upon parameters. A "first-block peek" decodes just the first RS block for each (fraction, parity) combination to read `plaintext_len` and derive the exact frame data length — **~30 RS block decodes** instead of scanning thousands of FDL values. A small-FDL fallback handles tiny messages (single partial RS block). AES-256-GCM-SIV authentication is the only validator
- **Stego-cost verification** — after STC embedding, the encoder re-runs UNIWARD on the stego image to verify shadow BER. If verification fails, an **escalation cascade** automatically increases RS parity (4 → 8 → 16 → … → 128) until the shadow survives or capacity is exhausted

`smart_decode` automatically tries shadow decode as a fallback after primary Ghost decode.

#### SI-UNIWARD (Deep Cover)

When the encoder has access to the original uncompressed pixels (e.g. PNG, HEIC, or RAW input converted to JPEG), **SI-UNIWARD** (Side-Informed UNIWARD) exploits JPEG quantization rounding errors to dramatically reduce embedding costs. Coefficients that were "close to the boundary" between two quantization levels can be flipped at near-zero cost, and the modification direction is chosen to move *toward* the pre-quantization value rather than the default nsF5 direction.

The result: **~43% higher capacity** at the same detection risk, or equivalently the same capacity with significantly lower distortion. The decoder is completely unchanged — `ghost_decode` works identically on standard and SI-UNIWARD stego images.

Use `ghost_encode_si` / `ghost_capacity_si` when raw pixels are available alongside the JPEG cover.

### Armor (Robust)

Optimizes for **survivability**. Uses Spread Transform Dither Modulation (STDM) to embed bits into stable low-frequency DCT coefficients, protected by [adaptive Reed-Solomon ECC](https://phasm.app/blog/surviving-jpeg-recompression) with [soft-decision decoding via log-likelihood ratios](https://phasm.app/blog/soft-majority-voting-llr-concatenated-codes). A [DFT magnitude template](https://phasm.app/blog/dft-template-geometric-resilience-steganography) provides geometric synchronization against rotation, scaling, and cropping.

Armor is the **default mode** on all platforms.

#### Fortress Sub-Mode

For short messages, Armor automatically activates **Fortress** — a [BA-QIM (Block-Average Quantization Index Modulation)](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) scheme that embeds one bit per 8x8 block into the block-average brightness. Fortress exploits [three invariants of JPEG recompression](https://phasm.app/blog/jpeg-recompression-invariants) — block averages, brightness ordering, and coefficient signs — that persist even through aggressive re-encoding.

Fortress survives WhatsApp recompression (QF ~62, resolution resize to 1600px). See [how 15 messaging platforms process your photos](https://phasm.app/blog/how-15-platforms-process-your-photos) for a comprehensive platform analysis.

[Watson perceptual masking](https://phasm.app/blog/watson-perceptual-masking-qim-steganography) adapts the embedding strength per-block based on local texture energy, keeping modifications invisible in textured regions while protecting smooth areas.

### Cover Image Optimizer

`optimize_cover` preprocesses raw pixels before JPEG compression to improve embedding quality and capacity. Each mode has a different pipeline:

- **Ghost**: Texture-adaptive 4-stage pipeline (noise injection, micro-contrast, unsharp mask, smooth-region dithering). Per-pixel 5×5 variance map adapts strength to existing texture — avoids degrading pre-optimized images.
- **Armor**: Light pipeline (block-boundary smoothing, DC stabilization) for STDM robustness.
- **Fortress**: Minimal (block-boundary smoothing only) to stabilize DC averages for BA-QIM.

**"Do no harm" guarantee**: if optimization reduces average gradient energy (a proxy for stego capacity), the original pixels are returned unchanged. Modifications are imperceptible (PSNR > 44 dB, SSIM > 0.993).

```rust
use phasm_core::{optimize_cover, OptimizerConfig, OptimizerMode};

let optimized = optimize_cover(&raw_pixels_rgb, width, height, &OptimizerConfig {
    strength: 0.85,
    seed: [0u8; 32], // ChaCha20 seed for deterministic noise
    mode: OptimizerMode::Ghost,
});
```

### Decode Auto-Detection

`smart_decode` tries all modes automatically — Ghost primary → Ghost shadow → Armor → Fortress — no mode selector needed on the decode side.

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
use phasm_core::{ghost_encode_si, ghost_decode};

// SI-UNIWARD: when you have original uncompressed pixels + the JPEG cover
let raw_pixels_rgb = /* RGB pixel data from PNG/HEIC/RAW */;
let cover = std::fs::read("photo.jpg").unwrap();
let stego = ghost_encode_si(
    &cover, raw_pixels_rgb, width, height, "secret message", "passphrase"
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
ghost_decode(jpeg_bytes, passphrase) -> Result<PayloadData>
ghost_encode_with_files(jpeg_bytes, message, files, passphrase) -> Result<Vec<u8>>
ghost_capacity(jpeg_image) -> Result<usize>

// Ghost shadow messages (plausible deniability)
ghost_encode_with_shadows(jpeg_bytes, message, files, passphrase, shadows, si) -> Result<Vec<u8>>
ghost_encode_si_with_shadows(jpeg_bytes, raw_rgb, width, height, message, files, passphrase, shadows) -> Result<Vec<u8>>
ghost_shadow_decode(stego_bytes, passphrase) -> Result<PayloadData>
estimate_shadow_capacity(jpeg_image) -> Result<usize>

// Ghost SI-UNIWARD (stealth + side-informed, ~43% more capacity)
ghost_encode_si(jpeg_bytes, raw_rgb, width, height, message, passphrase) -> Result<Vec<u8>>
ghost_encode_si_with_files(jpeg_bytes, raw_rgb, width, height, message, files, passphrase) -> Result<Vec<u8>>
ghost_capacity_si(jpeg_image) -> Result<usize>

// Armor mode (robust)
armor_encode(jpeg_bytes, message, passphrase) -> Result<Vec<u8>>
armor_decode(jpeg_bytes, passphrase) -> Result<(PayloadData, DecodeQuality)>
armor_capacity(jpeg_image) -> Result<usize>

// Unified decode (auto-detects mode)
smart_decode(jpeg_bytes, passphrase) -> Result<(PayloadData, DecodeQuality)>

// Cover image optimizer (preprocessing before JPEG compression)
optimize_cover(pixels_rgb, width, height, config) -> Vec<u8>

// Capacity estimation with Brotli compression
compressed_payload_size(text, mode) -> usize

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
- **`DecodeQuality`** — signal integrity percentage, RS error count/capacity, fortress flag
- **`ArmorCapacityInfo`** — capacity breakdown by encoding tier (Phase 1/2/3, Fortress)
- **`OptimizerConfig`** — optimizer settings (strength, seed, mode)
- **`OptimizerMode`** — pipeline variant (Ghost, Armor, Fortress)

The SI-UNIWARD functions (`ghost_encode_si`, `ghost_encode_si_with_files`) accept additional parameters for the raw uncompressed pixels (`raw_rgb: &[u8]`, `pixel_width: u32`, `pixel_height: u32`). The decoder does not need side information — `ghost_decode` and `smart_decode` work for both standard and SI-UNIWARD encoded images.

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
- **Memory-efficient** — strip-based wavelet computation, compact positions (8 bytes each), 1-bit packed Viterbi back pointers (32× reduction), segmented checkpoint for large images (O(√n) memory), `i8`-quantized SI-UNIWARD rounding errors (8× smaller than `f64`), strip-by-strip luma block computation, strategic `drop()` calls before JPEG output. Supports **200 MP** images under ~1 GB peak memory.
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
2. Derive structural key (Argon2id, overlapped with step 3 on multi-core) → permutation seed + STC seed
3. Compute [J-UNIWARD costs](https://phasm.app/blog/uerd-vs-juniwird-detection-benchmarks) **strip-by-strip** (Daubechies-8 wavelet, 3 subbands, parallel row+column filtering) — positions collected inline, no full CostMap materialized
4. *(SI-UNIWARD only)* Modulate costs inline using quantized rounding errors (`i8`, <2% precision loss): `cost *= (1 - 2|error|)`. Luma blocks computed in strips of 50 block-rows to minimize peak memory.
5. Permute coefficient order (Fisher-Yates with ChaCha20)
6. Encrypt payload (AES-256-GCM-SIV) and frame (length + CRC)
7. **Short STC with dynamic w**: embed only the actual `m` message bits (not zero-padded to `m_max`). The encoder picks `w = min(⌊n/m⌋, 10)` — for small messages this means **2,500× fewer coefficient modifications** compared to fixed `w`. The decoder brute-forces all `w` values 1–10 in parallel (`rayon::find_map_first`); CRC32 in the frame format provides instant validation (~0 false positive rate at 2⁻³²).
8. *(Shadow mode)* Assign `f32::INFINITY` cost to shadow positions → Viterbi routes around them
9. Embed via [STC](https://phasm.app/blog/syndrome-trellis-codes-practical-guide) (h=7) minimizing weighted distortion; SI-UNIWARD uses informed modification direction (toward pre-quantization value)
10. Write modified coefficients back to JPEG (with progress reporting per MCU row)

#### STC Viterbi Optimizer

The STC encoder uses a Viterbi-style dynamic programming algorithm to find the minimum-cost modification sequence across a trellis with 2^h = 128 states. Two key memory optimizations make it practical for large images (32MP+ photos, 48MP camera sensors):

**1-bit packed back pointers** — The standard Viterbi stores one predecessor state (u32) per trellis state per cover step. Since there are exactly 2 candidate predecessors per target state (stego_bit = 0 or 1), only 1 bit is needed. One `u128` (16 bytes) stores all 128 states' decisions per step, replacing a `Vec<u32>` of 128 entries (512 bytes). **32× memory reduction.**

The forward pass iterates over *target states* instead of source states:

```rust
for s in 0..num_states {
    let cost_0 = prev_cost[s]       + rho_0;  // stego=0, pred=s
    let cost_1 = prev_cost[s ^ col] + rho_1;  // stego=1, pred=s^col
    if cost_1 < cost_0 {
        curr_cost[s] = cost_1;
        packed_bp |= 1u128 << s;  // 1 bit records the choice
    } else {
        curr_cost[s] = cost_0;
    }
}
```

Traceback reconstructs predecessors from the single bit:

```rust
let bit = ((back_ptr[j] >> s) & 1) as u8;
if bit == 1 { s ^= col; }  // undo XOR to get predecessor
```

At message-block boundaries (shift steps), the pre-shift state is fully determined from the known message bit — no storage needed:

```rust
s = ((s << 1) | message_bit) & (num_states - 1);  // invertible shift
```

**Segmented checkpoint Viterbi** — For images with >1M usable positions, even 16 bytes/step can exceed mobile memory (e.g., 100M positions × 16 bytes = 1.6 GB). The segmented approach reduces memory to O(√n):

- **Phase A (forward scan)**: Run the full Viterbi without storing back pointers. Save cost array checkpoints every K = ⌈√m⌉ message blocks (~1 KB each). Total checkpoint memory: ~1.5 MB for maximum payload.
- **Phase B (segment recomputation)**: Process segments in reverse order. For each segment, re-run the forward pass from its checkpoint storing back pointers for that segment only, then traceback and free. Peak memory: ~200 KB per segment.

The dispatcher auto-selects the optimal path:

```rust
const SEGMENTED_THRESHOLD: usize = 1_000_000;

if n_used <= SEGMENTED_THRESHOLD {
    stc_embed_inline(...)   // single pass, all back_ptr in memory
} else {
    stc_embed_segmented(...) // checkpoint/recompute, O(√n) memory
}
```

Both paths produce **bit-for-bit identical output** (verified by equivalence tests). The segmented path trades 2× compute time for O(√n) memory — typically ~3 MB total for any image size.

| Image | n_used | Inline memory | Segmented memory |
|-------|--------|--------------|-----------------|
| 12 MP phone | 2M | 32 MB | 3 MB |
| 32 MB PNG | 5M | 80 MB | 3 MB |
| 48 MP camera | 10M | 160 MB | 3 MB |
| 100 MP medium format | 30M | 480 MB | 3 MB |
| 200 MP flagship | 59M | 944 MB | 3 MB |

#### Strip-Based UNIWARD & Compact Positions

The J-UNIWARD cost function requires wavelet decomposition of the entire decompressed image — three directional subbands (LH, HL, HH) via Daubechies-8 filters. For a 200 MP image, storing the full pixel array + 3 wavelet subbands would need ~3.2 GB.

The **strip-based streaming** approach eliminates this:

1. Process the image in horizontal strips of 50 block rows (~400 pixels)
2. Each strip decompresses its pixel rows (with ±22px padding for the 16-tap wavelet filter boundary), computes row/column-filtered wavelet subbands, and evaluates per-coefficient costs
3. Embeddable positions are collected inline into a compact `Vec<CoeffPos>` — no full CostMap is ever materialized
4. Strip memory is freed before the next strip begins

The `CoeffPos` type uses **compact representation**: `u32` flat index + `f32` cost = 8 bytes per position (down from 16 bytes with `usize` + `f64`). The STC accepts `f32` costs directly, promoting to `f64` only for internal Viterbi accumulation.

| Image | DctGrid | Positions | Strip buffers | **Total peak** |
|-------|---------|-----------|---------------|---------------|
| 12 MP phone | 24 MB | 48 MB | 12 MB | **84 MB** |
| 48 MP camera | 96 MB | 190 MB | 30 MB | **316 MB** |
| 100 MP medium format | 200 MB | 400 MB | 50 MB | **650 MB** |
| 200 MP flagship | 400 MB | 800 MB | 170 MB | **~1 GB** |

Capacity estimation is instantaneous: it counts non-zero AC coefficients directly from the DctGrid (no wavelet computation needed), since J-UNIWARD assigns finite costs to all non-zero coefficients.

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

- **Ghost encode**: 177 steps — 5 (parse) + 100 (UNIWARD sub-steps) + 50 (STC Viterbi sub-steps) + 2 (permute + key derivation) + 20 (JPEG write MCU rows)
- **Ghost encode with shadows**: 277 steps — adds 100 (verification UNIWARD pass) for stego-cost check
- **Ghost decode**: 107 steps — 5 (parse) + 100 (UNIWARD) + 2 (STC extraction + decrypt). Dynamic w brute-force runs in parallel.
- **Armor encode**: 6 steps (STDM path) or 3 steps (Fortress path, auto-adjusted at runtime)
- **Armor decode**: dynamic total based on candidate count (~50 steps). Phase 1 delta sweep parallelized across ~21 candidates.

Cancellation is cooperative: `progress::cancel()` sets a flag checked at natural loop boundaries via `progress::check_cancelled()`.

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
    error.rs        JPEG parsing errors
    frame.rs        SOF frame info (dimensions, components, subsampling)
    huffman.rs      Huffman coding tables (two-level decode, encode)
    marker.rs       JPEG marker iterator
    pixels.rs       IDCT/DCT for pixel-domain operations (forward DCT, luma extraction)
    scan.rs         Entropy-coded scan reader/writer
    tables.rs       DQT/DHT table parsing
    zigzag.rs       Zigzag scan order mapping
  stego/
    mod.rs          Ghost/Armor encode/decode entry points
    pipeline.rs     Ghost mode pipeline (J-UNIWARD + STC)
    crypto.rs       AES-256-GCM-SIV + Argon2id key derivation
    frame.rs        Payload framing (length, CRC, mode byte, salt, nonce)
    payload.rs      Payload serialization (Brotli, file attachments)
    permute.rs      Fisher-Yates coefficient permutation
    side_info.rs    SI-UNIWARD: rounding errors, cost modulation, direction selection
    shadow.rs       Shadow messages (Y-channel direct LSB + RS ECC, plausible deniability)
    optimizer.rs    Cover image optimizer (texture-adaptive preprocessing, do-no-harm)
    capacity.rs     Ghost capacity estimation
    progress.rs     Real-time progress tracking (atomics + WASM callback)
    error.rs        StegoError enum
    cost/
      mod.rs        Cost function trait
      uniward.rs    J-UNIWARD (Daubechies-8 wavelet, 3 subbands)
      uerd.rs       UERD cost function (legacy)
    stc/
      mod.rs        Syndrome-Trellis Codes
      embed.rs      STC Viterbi embedding (1-bit packed + segmented checkpoint)
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
- [From 480 MB to 3 MB: Fitting Viterbi Steganography on a Phone](https://phasm.app/blog/segmented-viterbi-memory-efficient-stc)

### Platform Analysis
- [How 15 Messaging Platforms Process Your Photos](https://phasm.app/blog/how-15-platforms-process-your-photos)

## License

GPL-3.0-only. See [LICENSE](LICENSE).

Third-party dependency licenses are listed in [THIRD_PARTY_LICENSES](THIRD_PARTY_LICENSES).
