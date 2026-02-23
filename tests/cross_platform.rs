//! Cross-platform determinism tests for the Fisher-Yates shuffle.
//!
//! The key invariant: `select_and_permute` must produce identical output on
//! native (64-bit `usize`) and WASM (32-bit `usize`) targets. This is
//! achieved by using `u32` for `rng.gen_range()` in the Fisher-Yates shuffle,
//! so the PRNG consumes the same entropy regardless of platform word size.
//!
//! These tests pin known permutation outputs so any future change to the
//! shuffle algorithm is immediately caught as a regression.

use phasm_core::stego::permute::select_and_permute;
use phasm_core::stego::cost::CostMap;
use phasm_core::{ghost_encode, ghost_decode, armor_encode, armor_decode, smart_decode};

fn load_test_image(name: &str) -> Vec<u8> {
    std::fs::read(format!("../test-vectors/{name}")).unwrap()
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
    assert_eq!(decoded, message);
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
    assert_eq!(decoded, message);
    assert_eq!(quality.mode, 0x02);
    assert_eq!(quality.integrity_percent, 100);
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
    assert_eq!(decoded, message);
    assert_eq!(quality.mode, 0x01, "smart_decode should detect Ghost mode");
}

#[test]
fn smart_decode_armor_cross_platform() {
    let cover = load_test_image("photo_320x240_q75_420.jpg");
    let message = "Armor via smart_decode";
    let passphrase = "smart-armor-xplat";

    let stego = armor_encode(&cover, message, passphrase).unwrap();
    let (decoded, quality) = smart_decode(&stego, passphrase).unwrap();
    assert_eq!(decoded, message);
    assert_eq!(quality.mode, 0x02, "smart_decode should detect Armor mode");
    assert_eq!(quality.integrity_percent, 100);
}
