// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The global cancel flag (`progress::cancel`) and the STC search that video
//! encode, probe and decode share with the image pipelines.
//!
//! Only the image pipelines clear the flag (`progress::init`). STC treats a
//! set flag as "no path", so before `progress::clear_cancel` existed, one
//! cancelled image operation — or a cancelled video encode, whose overlay
//! sets the same flag — left every later video encode finding no STC path in
//! any GOP: refused as too large, or a zero capacity, until an image
//! operation happened to reset it. The mobile bridges now expose
//! `clear_cancel` and the apps call it before every video operation.
//!
//! Its own test binary on purpose: the flag is process-global, and setting it
//! inside the lib's parallel unit tests would fail unrelated STC tests.

use phasm_core::progress;
use phasm_core::stego::stc::embed::{stc_embed, stc_is_feasible};
use phasm_core::stego::stc::hhat::generate_hhat;

const H: usize = 7;

fn instance() -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<Vec<u32>>, usize) {
    let w = 4;
    let n_msg = 512;
    let n = n_msg * w;
    let cover: Vec<u8> = (0..n).map(|i| ((i * 7 + 3) % 5 % 2) as u8).collect();
    let costs: Vec<f32> = (0..n).map(|i| 1.0 + (i % 13) as f32).collect();
    let msg: Vec<u8> = (0..n_msg).map(|i| ((i * 31 + 1) % 3 % 2) as u8).collect();
    let hhat = generate_hhat(H, w, &[0x5A; 32]);
    (cover, costs, msg, hhat, w)
}

#[test]
fn a_stale_cancel_poisons_stc_until_cleared() {
    let (cover, costs, msg, hhat, w) = instance();
    progress::clear_cancel();
    assert!(stc_is_feasible(&cover, &costs, &msg, &hhat, H, w), "fixture must be feasible");
    assert!(stc_embed(&cover, &costs, &msg, &hhat, H, w).is_some());

    // A cancel from an operation that has long finished: nothing on the video
    // path resets it, and the next STC search finds "no path".
    progress::cancel();
    assert!(!stc_is_feasible(&cover, &costs, &msg, &hhat, H, w));
    assert!(stc_embed(&cover, &costs, &msg, &hhat, H, w).is_none());

    // What the apps now do before every video operation.
    progress::clear_cancel();
    assert!(!progress::is_cancelled());
    assert!(stc_is_feasible(&cover, &costs, &msg, &hhat, H, w));
    assert!(stc_embed(&cover, &costs, &msg, &hhat, H, w).is_some());
}
