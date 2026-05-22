// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! M1 phase gate: Ghost shadow encode produces bit-identical output
//! regardless of rayon fanout — **when** crypto salt+nonce are forced
//! deterministic via `PHASM_DETERMINISTIC_SEED`.
//!
//! Without that env var, encode is intentionally nondeterministic
//! (per-run random AES-GCM-SIV salt + nonce embedded in the stego
//! bytes via `core/src/stego/crypto.rs:285`). The cascade selection
//! inside encode IS deterministic though; this test pins that
//! property by zeroing out the only known source of intentional
//! nondeterminism.
//!
//! The M4 ladder's rung 1 (`CappedParallel(1)`) wraps the existing
//! parallel cascade in a single-threaded `ThreadPoolBuilder`. This
//! test proves the wrapper is safe — that the cascade algorithm
//! (`collect-all-per-parity, max_by_key` at
//! `core/src/stego/ghost/pipeline.rs:704-797`, including the
//! work-skipping `best_fraction` AtomicUsize) is bit-deterministic
//! across worker counts.
//!
//! Run before/after every M2-M6 change. If this test fails, the
//! ladder is unsafe to ship.
//!
//! NOTE: tests in this file all set the same env-var value
//! (`PHASM_DETERMINISTIC_SEED=1`) before encoding and clear it after.
//! They are safe to run concurrently within this binary but unsafe
//! to interleave with other env-mutating tests.

#![cfg(feature = "parallel")]

use phasm_core::{ghost_encode_with_shadows, ShadowLayer};
use rayon::ThreadPoolBuilder;

const DETERMINISTIC_SEED: &str = "1";

fn load_shadow_test_image() -> Vec<u8> {
    std::fs::read("test-vectors/image/photo_640x480_q75_420.jpg").unwrap()
}

fn encode_with_shadows_in_pool(threads: usize, cover: &[u8]) -> Vec<u8> {
    // Force the salt+nonce RNG into deterministic ChaCha8 mode for the
    // duration of this encode. SAFETY: set_var/remove_var are
    // process-global; see file-level note.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", DETERMINISTIC_SEED); }
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .expect("rayon pool");
    let out = pool.install(|| {
        ghost_encode_with_shadows(
            cover,
            "primary message that is longer than each shadow",
            &[],
            "primary-pass-determinism",
            &[
                ShadowLayer {
                    message: "shadow one".to_string(),
                    passphrase: "shadow-pass-1".to_string(),
                    files: vec![],
                },
                ShadowLayer {
                    message: "shadow two".to_string(),
                    passphrase: "shadow-pass-2".to_string(),
                    files: vec![],
                },
            ],
            None,
        )
        .expect("ghost_encode_with_shadows")
    });
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }
    out
}

#[test]
fn fanout_1_matches_fanout_default() {
    let cover = load_shadow_test_image();
    let stego_serial = encode_with_shadows_in_pool(1, &cover);
    let stego_parallel = encode_with_shadows_in_pool(num_threads_default(), &cover);
    assert_eq!(
        stego_serial.len(),
        stego_parallel.len(),
        "byte length must match across fanouts",
    );
    assert!(
        stego_serial == stego_parallel,
        "stego bytes must match across rayon fanouts (single-threaded vs N-threaded). \
         first-difference offset = {:?}",
        first_diff(&stego_serial, &stego_parallel),
    );
}

#[test]
fn fanout_1_matches_fanout_2() {
    let cover = load_shadow_test_image();
    let s1 = encode_with_shadows_in_pool(1, &cover);
    let s2 = encode_with_shadows_in_pool(2, &cover);
    assert!(
        s1 == s2,
        "fanout 1 vs 2 differ at offset {:?}",
        first_diff(&s1, &s2),
    );
}

#[test]
fn fanout_default_matches_repeat() {
    let cover = load_shadow_test_image();
    let n = num_threads_default();
    let a = encode_with_shadows_in_pool(n, &cover);
    let b = encode_with_shadows_in_pool(n, &cover);
    assert!(
        a == b,
        "repeat encodes at the same fanout must be byte-identical. \
         first-difference offset = {:?}",
        first_diff(&a, &b),
    );
}

/// Sanity gate: without the env var, encode IS nondeterministic. If
/// this ever starts passing, something has removed the per-run
/// salt/nonce randomness and stego output is no longer as private
/// as the design promises.
#[test]
fn without_seed_encode_is_intentionally_nondeterministic() {
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }
    let cover = load_shadow_test_image();
    let pool = ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("rayon pool");
    let a = pool.install(|| {
        ghost_encode_with_shadows(
            &cover,
            "msg",
            &[],
            "p",
            &[ShadowLayer {
                message: "s".into(),
                passphrase: "sp".into(),
                files: vec![],
            }],
            None,
        )
        .unwrap()
    });
    let b = pool.install(|| {
        ghost_encode_with_shadows(
            &cover,
            "msg",
            &[],
            "p",
            &[ShadowLayer {
                message: "s".into(),
                passphrase: "sp".into(),
                files: vec![],
            }],
            None,
        )
        .unwrap()
    });
    assert!(
        a != b,
        "without PHASM_DETERMINISTIC_SEED, two encodes should differ \
         (per-run random salt + nonce embedded in stego). If they match, \
         crypto randomness regressed.",
    );
}

/// M4 rung-equivalence gate: every rung 0/1/2 must produce the
/// SAME stego bytes for the same input. Without this, the ladder
/// can't ship — the user would get different output depending on
/// available memory.
///
/// At 640×480 (~0.3 MP) the predictions are:
///   rung 0 (6 workers): ~58 MB
///   rung 1: ~20 MB
///   rung 2: ~13 MB
///   rung 3: ~8 MB
///
/// We pick budgets that cross each threshold.
#[test]
fn rung_2_matches_rung_0_byte_exact() {
    let cover = load_shadow_test_image();
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", DETERMINISTIC_SEED); }

    phasm_core::set_memory_budget(None);
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads_default())
        .build()
        .expect("pool");
    let stego_rung0 = pool.install(|| encode_with_shadows_in_pool_inner(&cover));

    // ~15 MB budget → predict_peak_memory drops rung 0 (~58), rung 1 (~20),
    // settles at rung 2 (~13). Verifies the StreamingNoCache path.
    phasm_core::set_memory_budget(Some(15 * 1024 * 1024));
    let stego_rung2 = encode_with_shadows_in_pool_inner(&cover);

    phasm_core::set_memory_budget(None);
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }

    assert_eq!(stego_rung0.len(), stego_rung2.len(), "rung 2 byte length must match rung 0");
    assert!(
        stego_rung0 == stego_rung2,
        "rung 0 vs rung 2 differ at offset {:?}",
        first_diff(&stego_rung0, &stego_rung2),
    );
}

#[test]
fn rung_3_matches_rung_0_byte_exact() {
    let cover = load_shadow_test_image();
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", DETERMINISTIC_SEED); }

    phasm_core::set_memory_budget(None);
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_threads_default())
        .build()
        .expect("pool");
    let stego_rung0 = pool.install(|| encode_with_shadows_in_pool_inner(&cover));

    // ~5 MB budget → settles at MinimalClones.
    phasm_core::set_memory_budget(Some(5 * 1024 * 1024));
    let stego_rung3 = encode_with_shadows_in_pool_inner(&cover);

    phasm_core::set_memory_budget(None);
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }

    assert_eq!(stego_rung0.len(), stego_rung3.len(), "rung 3 byte length must match rung 0");
    assert!(
        stego_rung0 == stego_rung3,
        "rung 0 vs rung 3 differ at offset {:?}",
        first_diff(&stego_rung0, &stego_rung3),
    );
}

#[test]
fn rung_1_matches_rung_0_byte_exact() {
    let cover = load_shadow_test_image();
    let n = num_threads_default();

    // Baseline: no budget → rung 0 (FullParallel).
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", DETERMINISTIC_SEED); }
    phasm_core::set_memory_budget(None);
    let pool0 = ThreadPoolBuilder::new()
        .num_threads(n)
        .build()
        .expect("pool");
    let stego_rung0 = pool0.install(|| encode_with_shadows_in_pool_inner(&cover));

    // Tight budget → rung 1 (CappedParallel single-thread).
    phasm_core::set_memory_budget(Some(40 * 1024 * 1024));
    let stego_rung1 = encode_with_shadows_in_pool_inner(&cover);

    phasm_core::set_memory_budget(None);
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED"); }

    assert_eq!(stego_rung0.len(), stego_rung1.len(), "byte length must match");
    assert!(
        stego_rung0 == stego_rung1,
        "rung 0 vs rung 1 differ at offset {:?}",
        first_diff(&stego_rung0, &stego_rung1),
    );
}

fn encode_with_shadows_in_pool_inner(cover: &[u8]) -> Vec<u8> {
    phasm_core::ghost_encode_with_shadows(
        cover,
        "primary message that is longer than each shadow",
        &[],
        "primary-pass-determinism",
        &[
            phasm_core::ShadowLayer {
                message: "shadow one".to_string(),
                passphrase: "shadow-pass-1".to_string(),
                files: vec![],
            },
            phasm_core::ShadowLayer {
                message: "shadow two".to_string(),
                passphrase: "shadow-pass-2".to_string(),
                files: vec![],
            },
        ],
        None,
    )
    .expect("ghost_encode_with_shadows")
}

fn num_threads_default() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(2)
        .max(2)
}

fn first_diff(a: &[u8], b: &[u8]) -> Option<usize> {
    a.iter().zip(b.iter()).position(|(x, y)| x != y)
}
