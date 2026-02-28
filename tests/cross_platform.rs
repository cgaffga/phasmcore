// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Cross-platform determinism tests.
//!
//! Two classes of invariants are pinned here:
//!
//! 1. **PRNG permutation**: `select_and_permute` must produce identical output
//!    on native (64-bit `usize`) and WASM (32-bit `usize`). Uses `u32` for
//!    `gen_range()` so the PRNG consumes identical entropy on both platforms.
//!
//! 2. **Deterministic math (sin/cos/atan2/FFT)**: All trig and FFT operations
//!    use `det_math` functions that compile to WASM intrinsics (f64.add/mul/
//!    div/sqrt/floor) — never JavaScript `Math.*`. This guarantees identical
//!    results on macOS Safari, iOS Safari, Android Chrome, etc.
//!
//! If any pinned value changes, cross-platform encode/decode is broken.

use phasm_core::det_math::{det_sin, det_cos, det_sincos, det_atan2, det_hypot};
use phasm_core::stego::permute::select_and_permute;
use phasm_core::stego::cost::CostMap;
use phasm_core::stego::armor::fft2d;
use phasm_core::stego::armor::template::generate_template_peaks;
use phasm_core::{ghost_encode, ghost_decode, armor_encode, armor_decode, smart_decode};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/{name}")).unwrap()
}

/// Create a cost map where every AC position has finite cost (1.0).
/// DC positions remain WET (infinity).
fn all_finite_map(bw: usize, bt: usize) -> CostMap {
    let mut map = CostMap::new(bw, bt);
    for br in 0..bt {
        for bc in 0..bw {
            for i in 0..8 {
                for j in 0..8 {
                    if i == 0 && j == 0 {
                        continue; // skip DC
                    }
                    map.set(br, bc, i, j, 1.0);
                }
            }
        }
    }
    map
}

// ---------------------------------------------------------------------------
// 1. Pin known permutation output (the most important test)
// ---------------------------------------------------------------------------

/// Pin the first 20 flat_idx values from `select_and_permute` for a 4x4-block
/// cost map with seed [42; 32]. These values are the ground truth for the u32
/// Fisher-Yates shuffle. If this test fails, the shuffle algorithm has changed
/// and WASM/native decode compatibility is broken.
#[test]
fn pin_known_values_4x4_seed42() {
    // 4x4 blocks = 16 blocks x 63 AC = 1008 positions
    let map = all_finite_map(4, 4);
    let seed = [42u8; 32];
    let positions = select_and_permute(&map, &seed);

    assert_eq!(
        positions.len(),
        1008,
        "4x4 blocks should have 1008 AC positions"
    );

    let first_20: Vec<usize> = positions.iter().take(20).map(|p| p.flat_idx).collect();

    // Pinned from the u32 Fisher-Yates shuffle on 2026-02-23.
    // These values MUST be identical on native (64-bit) and WASM (32-bit).
    let expected: Vec<usize> = vec![
        258, 980, 673, 988, 76, 41, 725, 301, 438, 872, 667, 574, 867, 881, 46, 240, 965, 56,
        339, 941,
    ];

    assert_eq!(
        first_20, expected,
        "Permutation output changed! This breaks WASM/native compatibility.\n\
         If you intentionally changed the shuffle algorithm, update these \
         pinned values AND verify that existing stego images can still be decoded."
    );
}

// ---------------------------------------------------------------------------
// 2. Deterministic: same input => same output
// ---------------------------------------------------------------------------

#[test]
fn permutation_is_deterministic() {
    let map = all_finite_map(4, 4);
    let seed = [42u8; 32];

    let a = select_and_permute(&map, &seed);
    let b = select_and_permute(&map, &seed);

    let a_idx: Vec<usize> = a.iter().map(|p| p.flat_idx).collect();
    let b_idx: Vec<usize> = b.iter().map(|p| p.flat_idx).collect();
    assert_eq!(a_idx, b_idx, "Same seed must produce identical permutation");
}

// ---------------------------------------------------------------------------
// 3. Different seeds produce different permutations
// ---------------------------------------------------------------------------

#[test]
fn different_seeds_produce_different_permutations() {
    let map = all_finite_map(4, 4);
    let seed_a = [1u8; 32];
    let seed_b = [2u8; 32];

    let a = select_and_permute(&map, &seed_a);
    let b = select_and_permute(&map, &seed_b);

    let a_idx: Vec<usize> = a.iter().map(|p| p.flat_idx).collect();
    let b_idx: Vec<usize> = b.iter().map(|p| p.flat_idx).collect();
    assert_ne!(
        a_idx, b_idx,
        "Different seeds must produce different permutations"
    );
}

// ---------------------------------------------------------------------------
// 4. Verify u32 range invariant with large cost map
// ---------------------------------------------------------------------------

/// The Fisher-Yates shuffle casts `i` to `u32` for `gen_range`. This test
/// verifies that a cost map with 258,048 positions (well within u32::MAX)
/// shuffles correctly with no index corruption.
///
/// Note: Creating a cost map with >= 2^32 positions is unrealistic for JPEG
/// steganography (would require ~68 billion DCT blocks). The maximum real-world
/// case is 8192x8192 pixels = ~1M blocks x 63 AC = ~66M positions, which fits
/// comfortably in u32.
#[test]
fn u32_range_invariant_large_map() {
    let map = all_finite_map(64, 64); // 4096 blocks x 63 AC = 258,048 positions
    let seed = [99u8; 32];
    let positions = select_and_permute(&map, &seed);
    assert_eq!(positions.len(), 4096 * 63);

    // Verify all indices are unique (no corruption from u32 cast).
    let mut indices: Vec<usize> = positions.iter().map(|p| p.flat_idx).collect();
    indices.sort();
    indices.dedup();
    assert_eq!(
        indices.len(),
        4096 * 63,
        "All positions must be unique after shuffle — u32 cast may be corrupting indices"
    );
}

// ---------------------------------------------------------------------------
// 5. Ghost roundtrip (cross-platform confirmation)
// ---------------------------------------------------------------------------

#[test]
fn ghost_roundtrip_cross_platform() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Cross-platform Ghost test";
    let passphrase = "cross-platform-key-123";

    let stego = ghost_encode(&cover, message, passphrase).unwrap();
    let decoded = ghost_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
}

// ---------------------------------------------------------------------------
// 6. Armor roundtrip (cross-platform confirmation)
// ---------------------------------------------------------------------------

#[test]
fn armor_roundtrip_cross_platform() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Cross-platform Armor test";
    let passphrase = "cross-platform-key-456";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = armor_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x02);
    assert!(quality.integrity_percent >= 85,
        "Pristine Armor integrity should be high: {}%", quality.integrity_percent);
}

// ---------------------------------------------------------------------------
// 7. smart_decode roundtrip for both modes
// ---------------------------------------------------------------------------

#[test]
fn smart_decode_ghost_cross_platform() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Ghost via smart_decode";
    let passphrase = "smart-ghost-xplat";

    let stego = ghost_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x01, "smart_decode should detect Ghost mode");
}

#[test]
fn smart_decode_armor_cross_platform() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Armor via smart_decode";
    let passphrase = "smart-armor-xplat";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded.text, message);
    assert_eq!(quality.mode, 0x02, "smart_decode should detect Armor mode");
    assert!(quality.integrity_percent >= 85,
        "Pristine Armor integrity should be high: {}%", quality.integrity_percent);
}

// ===========================================================================
// 8. Deterministic math — pinned bit-exact values
// ===========================================================================

/// Pin det_sin/det_cos at specific inputs. These exact bit patterns must be
/// identical on native and WASM. Any change breaks cross-platform decode.
#[test]
fn pin_det_sincos_bits() {
    let cases: &[(f64, u64, u64)] = &[
        //  x,               det_sin(x) bits,     det_cos(x) bits
        (0.1,  0x3fb98eaecb8bcb2c, 0x3fefd712f9a817c0),
        (0.5,  0x3fdeaee8744b05f0, 0x3fec1528065b7d50),
        (1.0,  0x3feaed548f090cee, 0x3fe14a280fb5068c),
        (2.5,  0x3fe326af0dcfcab0, 0xbfe9a2f7ef858b7d),
        (-1.5, 0xbfefeb7a9b2c6d8b, 0x3fb21bd54fc5f9a7),
        (5.0,  0xbfeeaf81f5e09933, 0x3fd22785706b4ada),
    ];
    for &(x, expected_sin, expected_cos) in cases {
        assert_eq!(
            det_sin(x).to_bits(), expected_sin,
            "det_sin({x}) bit mismatch: got {:#018x}, expected {:#018x}",
            det_sin(x).to_bits(), expected_sin
        );
        assert_eq!(
            det_cos(x).to_bits(), expected_cos,
            "det_cos({x}) bit mismatch: got {:#018x}, expected {:#018x}",
            det_cos(x).to_bits(), expected_cos
        );
    }
}

/// Verify det_sincos returns identical bits to separate det_sin/det_cos calls.
#[test]
fn pin_det_sincos_consistency() {
    for i in 0..50 {
        let x = (i as f64 - 25.0) * 0.37;
        let (s, c) = det_sincos(x);
        assert_eq!(s.to_bits(), det_sin(x).to_bits(), "sincos sin mismatch at x={x}");
        assert_eq!(c.to_bits(), det_cos(x).to_bits(), "sincos cos mismatch at x={x}");
    }
}

/// Pin det_atan2 at specific inputs.
#[test]
fn pin_det_atan2_bits() {
    let cases: &[(f64, f64, u64)] = &[
        (1.0,  1.0,  0x3fe921fb54442d18),
        (1.0, -1.0,  0x4002d97c7f3321d2),
        (-3.0, 4.0,  0xbfe4978fa3269ee1),
        (0.5,  2.0,  0x3fcf5b75f92c80dd),
    ];
    for &(y, x, expected) in cases {
        assert_eq!(
            det_atan2(y, x).to_bits(), expected,
            "det_atan2({y},{x}) bit mismatch: got {:#018x}", det_atan2(y, x).to_bits()
        );
    }
}

/// Pin det_hypot at specific inputs.
#[test]
fn pin_det_hypot_bits() {
    let cases: &[(f64, f64, u64)] = &[
        (3.0, 4.0, 0x4014000000000000), // 5.0 exact
        (1.0, 1.0, 0x3ff6a09e667f3bcd), // √2
        (0.1, 0.2, 0x3fcc9f25c5bfedda),
    ];
    for &(x, y, expected) in cases {
        assert_eq!(
            det_hypot(x, y).to_bits(), expected,
            "det_hypot({x},{y}) bit mismatch: got {:#018x}", det_hypot(x, y).to_bits()
        );
    }
}

// ===========================================================================
// 9. IDCT cosine table — pinned entries
// ===========================================================================

/// Pin selected IDCT cosine table entries. These feed into pixel↔coefficient
/// transforms used by the geometry-resilient DFT template.
#[test]
fn pin_idct_cosine_table() {
    let entries: &[(usize, usize, u64)] = &[
        (0, 0, 0x3ff0000000000000), // cos(0) = 1.0
        (0, 4, 0x3ff0000000000000), // cos(0) = 1.0
        (1, 0, 0x3fef6297cff75cb0),
        (1, 1, 0x3fea9b66290ea1a3),
        (2, 3, 0xbfed906bcf328d46),
        (3, 7, 0xbfea9b66290ea1a2),
        (5, 2, 0x3fc8f8b83c69a60b),
        (7, 7, 0xbfc8f8b83c69a5d6),
    ];
    for &(u, x, expected) in entries {
        let val = det_cos((2 * x + 1) as f64 * u as f64 * std::f64::consts::PI / 16.0);
        assert_eq!(
            val.to_bits(), expected,
            "cosine[{u}][{x}] bit mismatch: got {:#018x}", val.to_bits()
        );
    }
}

// ===========================================================================
// 10. FFT output — pinned for known input
// ===========================================================================

/// Pin 2x2 FFT output. Tests the full FFT pipeline (radix-2 path).
/// Spectrum uses f32 (Complex32) after Phase 3 memory optimization.
#[test]
fn pin_fft_2x2_output() {
    let pixels = vec![1.0, 2.0, 3.0, 4.0];
    let spectrum = fft2d::fft2d(&pixels, 2, 2);

    // f32 bit patterns: 10.0=0x41200000, -2.0=0xc0000000, -4.0=0xc0800000, 0.0=0x00000000
    let expected_re = [0x41200000u32, 0xc0000000, 0xc0800000, 0x00000000];
    let expected_im = [0x00000000u32, 0x00000000, 0x00000000, 0x00000000];

    for (i, c) in spectrum.data.iter().enumerate() {
        assert_eq!(
            c.re.to_bits(), expected_re[i],
            "FFT[{i}].re bit mismatch: got {:#010x}", c.re.to_bits()
        );
        assert_eq!(
            c.im.to_bits(), expected_im[i],
            "FFT[{i}].im bit mismatch: got {:#010x}", c.im.to_bits()
        );
    }
}

/// FFT roundtrip must preserve values within f32 precision.
/// Internal computation uses Complex32 after Phase 3 memory optimization.
#[test]
fn fft_roundtrip_deterministic() {
    let w = 16;
    let h = 12; // non-power-of-2 height triggers Bluestein path
    let pixels: Vec<f64> = (0..w * h).map(|i| 100.0 + (i as f64) * 0.73).collect();

    let spectrum = fft2d::fft2d(&pixels, w, h);
    let recovered = fft2d::ifft2d(&spectrum);

    for i in 0..pixels.len() {
        assert!(
            (pixels[i] - recovered[i]).abs() < 0.1,
            "FFT roundtrip mismatch at [{i}]: {:.6} vs {:.6}",
            pixels[i], recovered[i]
        );
    }
}

// ===========================================================================
// 11. Template peak positions — pinned for fixed passphrase
// ===========================================================================

/// Pin the first 5 template peak positions for a known passphrase.
/// These peaks are used for DFT template embedding in Armor mode.
#[test]
fn pin_template_peaks() {
    let peaks = generate_template_peaks("determinism-test", 256, 256);
    assert_eq!(peaks.len(), 32);

    let expected: &[(u64, u64)] = &[
        (0x4030685798345b94, 0xc02cf16958a18bcc),
        (0xc020723de9ad4fd4, 0xc023ddccdc1eb324),
        (0x403504dc96d68438, 0x402bab49b7c2b4d2),
        (0x403b7eb6a3415fe5, 0x4033d431bc2293d7),
        (0x40366bc957e159d9, 0x403a9705f70d67ef),
    ];

    for (i, &(eu, ev)) in expected.iter().enumerate() {
        assert_eq!(
            peaks[i].u.to_bits(), eu,
            "peak[{i}].u bit mismatch: got {:#018x}", peaks[i].u.to_bits()
        );
        assert_eq!(
            peaks[i].v.to_bits(), ev,
            "peak[{i}].v bit mismatch: got {:#018x}", peaks[i].v.to_bits()
        );
    }
}

/// Same passphrase + dimensions must produce identical peaks every time.
#[test]
fn template_peaks_deterministic() {
    let a = generate_template_peaks("det-test-2", 512, 384);
    let b = generate_template_peaks("det-test-2", 512, 384);
    for i in 0..a.len() {
        assert_eq!(a[i].u.to_bits(), b[i].u.to_bits(), "peak[{i}].u differs");
        assert_eq!(a[i].v.to_bits(), b[i].v.to_bits(), "peak[{i}].v differs");
    }
}

// ===========================================================================
// 12. sin²+cos² identity holds across the full range
// ===========================================================================

/// Pythagorean identity must hold for all inputs used in practice (angles
/// from PRNG, FFT twiddle factors, etc.).
#[test]
fn sincos_pythagorean_identity_full_range() {
    for i in 0..1000 {
        let x = (i as f64 - 500.0) * 0.013;
        let (s, c) = det_sincos(x);
        let err = (s * s + c * c - 1.0).abs();
        assert!(err < 1e-14, "sin²+cos² = {} at x={x} (err={err})", s * s + c * c);
    }
}
