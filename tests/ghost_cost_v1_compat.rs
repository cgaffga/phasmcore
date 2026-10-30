// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Images made by phasm-core ≤ 0.4.x / app releases ≤ 1.8 ("v1": binary
//! STC on the non-zero AC set, v1 J-UNIWARD costs, shadow layout v1) must
//! keep decoding — primary AND every shadow — next to Ghost v3.
//!
//! The fixtures in `test-vectors/image/ghost-cost-v1-compat/` were encoded
//! by the released 0.4.3 CLI (tag v0.4.3, v1 costs) with `regen.sh` there:
//! JPEG and PNG (SI / Deep Cover) covers, 1–4 shadows, short and
//! pool-filling shadows. The larger ones (12 MP photo, real photos) live
//! in the gitignored `tests/real_photos/ghost-cost-v1-compat/` and are
//! skipped when absent.
//!
//! * The primary message never depended on cost values, only on the set
//!   of non-zero AC coefficients, which v1 and v2 share.
//! * Shadow layout v1 (≤ 0.4.x) chose a shadow's cost pool by cost. Every
//!   shadow the 0.4.3 encoder actually produced in our sweeps (these
//!   fixtures + 97 more encodes) sits in the 100 % pool, where costs do not
//!   matter: a shadow's own LSB writes reshuffle the cheap pool's priority
//!   order, so the encoder's first, tight-pool verify fails and its cascade
//!   keeps fraction 1. That pool is the current decoder's legacy tier (the
//!   keyed hash order, direct LSB), so these fixtures decode without any
//!   cost computation; they pin that nothing regressed on real encoder
//!   output.
//! * A v1 shadow in a tight pool (5–50 %) — possible in principle, never
//!   observed — is found by the decoder's legacy fallback
//!   (`compute_positions_streaming_legacy_v1` + v1's cost pools; mechanics:
//!   `pipeline::tests::shadow_fallback_finds_a_pool_only_the_legacy_order_selects`).
//! * New encodes (Ghost v3: keyed-region STC shadows on the all-AC set)
//!   need no costs to decode either.

use phasm_core::stego::cost::uniward::compute_positions_streaming_legacy_v1;
use phasm_core::stego::shadow::{shadow_extract_cost_pools, shadow_extract_legacy, shadow_extract_v3};
use phasm_core::{ghost_decode, ghost_encode_with_shadows, ghost_shadow_decode, smart_decode, JpegImage, ShadowLayer};

/// A fixture message: literal text, or `regen.sh`'s `hexmsg <seed> <n>`
/// (`n` hex chars of SHA-256("<seed>.<i>"), i = 0, 1, …).
#[derive(Clone, Copy)]
enum M {
    L(&'static str),
    H(&'static str, usize),
}

impl M {
    fn text(self) -> String {
        match self {
            M::L(s) => s.to_string(),
            M::H(seed, n) => {
                use sha2::{Digest, Sha256};
                let mut out = String::new();
                let mut i = 0;
                while out.len() < n {
                    let d = Sha256::digest(format!("{seed}.{i}").as_bytes());
                    for b in d {
                        out.push_str(&format!("{b:02x}"));
                    }
                    i += 1;
                }
                out.truncate(n);
                out
            }
        }
    }
}

struct Fixture {
    path: &'static str,
    primary_pass: &'static str,
    primary: M,
    shadows: &'static [(&'static str, M)],
}

const P1: M = M::L("The quick brown fox jumps over the lazy dog, primary layer one.");
const P2: M = M::L("Primary message for the cost-v1 compatibility fixture, a little longer than the shadows.");

const SMALL: &str = "test-vectors/image/ghost-cost-v1-compat";
const LARGE: &str = "tests/real_photos/ghost-cost-v1-compat";

/// Mirrors `regen.sh`.
const FIXTURES: &[Fixture] = &[
    Fixture { path: "S/v1_320x240_s1.jpg", primary_pass: "pp-a", primary: P1, shadows: &[("sp-a1", M::L("first shadow"))] },
    Fixture { path: "S/v1_640x480_s2.jpg", primary_pass: "pp-b", primary: P2, shadows: &[("sp-b1", M::L("shadow one, b")), ("sp-b2", M::L("shadow two"))] },
    Fixture {
        path: "S/v1_640x480_s4.jpg", primary_pass: "pp-c", primary: P2,
        shadows: &[("sp-c1", M::L("s1")), ("sp-c2", M::L("second shadow")), ("sp-c3", M::L("third shadow layer")), ("sp-c4", M::L("the fourth shadow, longest"))],
    },
    Fixture { path: "S/v1_640x480_si_s1.jpg", primary_pass: "pp-d", primary: P1, shadows: &[("sp-d1", M::L("si shadow 1"))] },
    Fixture { path: "S/v1_640x480_tight_s1.jpg", primary_pass: "pp-t", primary: M::H("prim-t", 1200), shadows: &[("sp-t1", M::H("tight-1", 300))] },
    Fixture {
        path: "S/v1_640x480_si_tight_s2.jpg", primary_pass: "pp-u", primary: M::H("prim-u", 1500),
        shadows: &[("sp-u1", M::H("tight-u1", 280)), ("sp-u2", M::L("short one"))],
    },
    Fixture { path: "S/v1_640x480_si_s3.jpg", primary_pass: "pp-e", primary: P2, shadows: &[("sp-e1", M::L("si one")), ("sp-e2", M::L("si two")), ("sp-e3", M::L("si three!"))] },
    Fixture { path: "L/v1_12mp_s1.jpg", primary_pass: "pp-f", primary: P2, shadows: &[("sp-f1", M::L("12mp shadow"))] },
    Fixture { path: "L/v1_12mp_s2.jpg", primary_pass: "pp-g", primary: P2, shadows: &[("sp-g1", M::L("12mp one")), ("sp-g2", M::L("12mp two, b"))] },
    Fixture {
        path: "L/v1_12mp_s4.jpg", primary_pass: "pp-h", primary: P2,
        shadows: &[("sp-h1", M::L("a")), ("sp-h2", M::L("bb shadow")), ("sp-h3", M::L("ccc shadow three")), ("sp-h4", M::L("dddd shadow four"))],
    },
    Fixture {
        path: "L/v1_12mp_tight_s2.jpg", primary_pass: "pp-l", primary: M::H("prim-l", 30000),
        shadows: &[("sp-l1", M::H("tight-l1", 18000)), ("sp-l2", M::H("tight-l2", 2000))],
    },
    // A primary frame over 256 KB: the 0.4.3 encoder wrote it, the 0.4.3
    // decoder rejected it (frame bound), the current decoder must read it.
    Fixture { path: "L/v1_12mp_bigframe.jpg", primary_pass: "pp-big", primary: M::H("bigframe", 600000), shadows: &[] },
    Fixture { path: "L/v1_whatsapp_s3.jpg", primary_pass: "pp-i", primary: P2, shadows: &[("sp-i1", M::L("wa one")), ("sp-i2", M::L("wa two")), ("sp-i3", M::L("wa three"))] },
    Fixture { path: "L/v1_png_si_s2.jpg", primary_pass: "pp-j", primary: P2, shadows: &[("sp-j1", M::L("png si one")), ("sp-j2", M::L("png si two"))] },
    Fixture {
        path: "L/v1_png_si_s4.jpg", primary_pass: "pp-k", primary: P2,
        shadows: &[("sp-k1", M::L("k1")), ("sp-k2", M::L("k2 shadow")), ("sp-k3", M::L("k3 shadow 3")), ("sp-k4", M::L("k4 shadow four"))],
    },
];

fn load(f: &Fixture) -> Option<Vec<u8>> {
    let p = if let Some(rest) = f.path.strip_prefix("S/") {
        format!("{SMALL}/{rest}")
    } else {
        format!("{LARGE}/{}", &f.path[2..])
    };
    match std::fs::read(&p) {
        Ok(b) => Some(b),
        Err(_) if f.path.starts_with("L/") => {
            eprintln!("skip {p} (gitignored large fixture absent)");
            None
        }
        Err(e) => panic!("committed fixture {p} missing: {e}"),
    }
}

/// Cost-sorted Y positions, as the shadow decoder sorts them.
fn sorted(mut p: Vec<phasm_core::stego::permute::CoeffPos>) -> Vec<phasm_core::stego::permute::CoeffPos> {
    p.sort_by(|a, b| a.cost.total_cmp(&b.cost));
    p
}

#[test]
fn v1_images_decode_primary_and_every_shadow() {
    let mut shadows_total = 0;
    let mut missed_by_legacy_tier = 0;
    let mut missed_names: Vec<String> = Vec::new();
    for f in FIXTURES {
        let Some(bytes) = load(f) else { continue };
        let name = f.path;

        // Primary: unaffected by the cost change.
        let p = ghost_decode(&bytes, f.primary_pass).unwrap_or_else(|e| panic!("{name}: primary: {e:?}"));
        assert_eq!(p.text, f.primary.text(), "{name}: primary text");
        let (sp, _) = smart_decode(&bytes, f.primary_pass).unwrap_or_else(|e| panic!("{name}: smart primary: {e:?}"));
        assert_eq!(sp.text, f.primary.text(), "{name}: smart primary text");

        let img = JpegImage::from_bytes(&bytes).unwrap();
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt = img.quant_table(qt_id).unwrap();
        let v1 = sorted(compute_positions_streaming_legacy_v1(img.dct_grid(0), qt, None).unwrap());

        for &(pass, msg) in f.shadows {
            let msg = msg.text();
            let msg = msg.as_str();
            shadows_total += 1;
            // v1's own pools under the v1 costs find it (the image IS a v1 image)...
            let legacy = shadow_extract_cost_pools(&img, &v1, pass, &[20, 10, 5, 2, 1])
                .unwrap_or_else(|e| panic!("{name} {pass}: v1 costs: {e:?}"));
            assert_eq!(legacy.text, msg, "{name} {pass}: v1 text");
            // ...the cost-free legacy tier (v1's 100 % pool) may not...
            if !matches!(shadow_extract_legacy(&img, pass), Ok(ref p) if p.text == msg) {
                missed_by_legacy_tier += 1;
                missed_names.push(format!("{name}:{pass}"));
            }
            // ...and the public decoders (with the fallback) always do.
            let d = ghost_shadow_decode(&bytes, pass).unwrap_or_else(|e| panic!("{name} {pass}: shadow decode: {e:?}"));
            assert_eq!(d.text, msg, "{name} {pass}: shadow text");
            let (s, _) = smart_decode(&bytes, pass).unwrap_or_else(|e| panic!("{name} {pass}: smart decode: {e:?}"));
            assert_eq!(s.text, msg, "{name} {pass}: smart text");
        }
    }
    eprintln!("{shadows_total} v1 shadows decoded; {missed_by_legacy_tier} needed the cost-pool fallback ({missed_names:?})");
    assert!(shadows_total >= 14, "committed fixtures hold 14 shadows");
}

/// A wrong passphrase still fails cleanly (the fallback adds a second
/// attempt, not a false positive).
#[test]
fn wrong_passphrase_still_fails() {
    let bytes = std::fs::read(format!("{SMALL}/v1_640x480_s2.jpg")).unwrap();
    assert!(ghost_shadow_decode(&bytes, "not-the-pass").is_err());
    assert!(smart_decode(&bytes, "not-the-pass").is_err());
}

/// A new encode's shadows (Ghost v3) are found by the v3 layout directly
/// and therefore never depend on the fallback.
#[test]
fn v3_shadows_decode_without_the_fallback() {
    let cover = std::fs::read("test-vectors/image/photo_640x480_q75_420.jpg").unwrap();
    let shadows: Vec<ShadowLayer> = [("n1", "new shadow one"), ("n2", "new shadow two"), ("n3", "third")]
        .iter()
        .map(|&(p, m)| ShadowLayer { message: m.into(), passphrase: p.into(), files: vec![] })
        .collect();
    let p2 = P2.text();
    // Shadow layout v1 rejected this for about 1 in 6 salts (the three
    // shadows overwrote each other in the shared 100 % pool beyond what RS
    // could correct); the v3 layout keeps the layers disjoint, so every salt
    // encodes. (Only this test encodes, so the process-global seed is safe.)
    let mut stego = Vec::new();
    for s in 1..=12u64 {
        unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", s.to_string()) };
        stego = ghost_encode_with_shadows(&cover, &p2, &[], "np", &shadows, None)
            .unwrap_or_else(|e| panic!("salt {s}: {e:?}"));
    }
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
    let img = JpegImage::from_bytes(&stego).unwrap();
    for s in &shadows {
        let got = shadow_extract_v3(&img, &s.passphrase).unwrap();
        assert_eq!(got.text, s.message);
    }
    assert_eq!(ghost_decode(&stego, "np").unwrap().text, p2);
}
