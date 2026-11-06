// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Armor v2 (core 0.6 / apps 1.10): compatibility with released stegos, the
//! decoder-only grid fix, exact v2 capacity, luma headroom, and the
//! legacy-index wire variant.
//! Fixtures: `test-vectors/image/armor-compat/`, written by the 0.4.3 / 0.5.1
//! release encoders (eval/runs/2026-10-31-armor-v2/scripts/make_fixtures.py).

use phasm_core::stego::armor::embedding::StdmRounding;
use phasm_core::stego::armor::pipeline::{
    armor_decode_opts, armor_encode_opts, ArmorDecodeOpts, ArmorEncodeOpts, StdmVariant,
};
use phasm_core::stego::armor::fortress::{fortress_capacity, fortress_capacity_compact};
use phasm_core::{armor_capacity, armor_capacity_info, armor_decode, armor_encode, smart_decode, JpegImage};

const FIXTURE_MSG: &str = "STDM fixture: the quick brown fox jumps over the lazy dog.";

fn vector(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/image/{name}")).unwrap()
}

#[test]
fn released_stdm_stegos_decode() {
    for f in ["armor-compat/armor_043_stdm_320.jpg", "armor-compat/armor_051_stdm_320.jpg"] {
        let st = vector(f);
        assert_eq!(armor_decode(&st, "fixture").unwrap().0.text, FIXTURE_MSG, "{f} armor_decode");
        assert_eq!(smart_decode(&st, "fixture").unwrap().0.text, FIXTURE_MSG, "{f} smart_decode");
        let (_, _, info) = armor_decode_opts(&st, "fixture", &ArmorDecodeOpts::default()).unwrap();
        assert_eq!(info.path, "legacy");
        assert!(armor_decode(&st, "wrong").is_err());
    }
}

#[test]
fn grid_geometry_fix_reads_444_odd_cover_after_420_recompression() {
    // 0.5.1 stego on a 312x232 4:4:4 cover (39x29 blocks), re-encoded QF75 4:2:0
    // (MCU-padded grid 40x30): the 0.5.x decoder reads the wrong blocks.
    let st = vector("armor-compat/armor_051_444odd_stdm_q75_420.jpg");
    assert!(armor_decode_opts(&st, "fixture", &ArmorDecodeOpts::shipped()).is_err());
    let (p, _, info) = armor_decode_opts(&st, "fixture", &ArmorDecodeOpts::default()).unwrap();
    assert_eq!(p.text, FIXTURE_MSG);
    assert!(!info.native_grid);
    assert_eq!(info.enc_grid, (39, 29));
    assert_eq!(smart_decode(&st, "fixture").unwrap().0.text, FIXTURE_MSG);
}

#[test]
fn v2_is_the_default_and_old_decoder_rejects_it() {
    let cover = vector("photo_640x480_q75_420.jpg");
    // One payload byte beyond the Fortress capacity (Fortress T or TL).
    let (_, over) = fit_prefix(armor_capacity_info(&cover).unwrap().fortress_capacity);
    let msg = over.as_str();
    let st = armor_encode(&cover, msg, "pw").unwrap();
    assert_eq!(armor_decode(&st, "pw").unwrap().0.text, msg);
    assert_eq!(smart_decode(&st, "pw").unwrap().0.text, msg);
    let (_, _, di) = armor_decode_opts(&st, "pw", &ArmorDecodeOpts::default()).unwrap();
    assert_eq!(di.path, "v2");
    // The core <= 0.5.x decoder cannot read a v2 message.
    assert!(armor_decode_opts(&st, "pw", &ArmorDecodeOpts::shipped()).is_err());
    assert!(armor_decode(&st, "wrong").is_err());
}

#[test]
fn v2_capacity_is_exact() {
    for name in ["photo_640x480_q75_420.jpg", "photo_320x240_q75_420.jpg", "nonaligned_13x13_q80.jpg"] {
        let cover = vector(name);
        let Ok(img) = JpegImage::from_bytes(&cover) else { continue };
        let cap = armor_capacity(&img).unwrap();
        if cap < 20 {
            continue;
        }
        // Prefixes of one random printable string; `size(n)` is the payload size
        // (what the capacity counts) of the first n characters.
        let mut s = 0x1234_5678u64;
        let long: String = (0..4 * cap)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (b'!' + (s % 94) as u8) as char
            })
            .collect();
        let size = |n: usize| phasm_core::stego::payload::compressed_payload_size(&long[..n], &[]);
        let fit = (1..4 * cap).take_while(|&n| size(n) <= cap).last().unwrap();
        let opts = ArmorEncodeOpts { force_stdm: true, ..ArmorEncodeOpts::default() };
        let (st, _) = armor_encode_opts(&cover, &long[..fit], "pw", &opts)
            .unwrap_or_else(|e| panic!("{name}: {e:?} at payload {} / cap {cap}", size(fit)));
        assert_eq!(armor_decode(&st, "pw").unwrap().0.text, &long[..fit]);
        let over = (fit + 1..4 * cap).find(|&n| size(n) > cap).unwrap();
        assert!(armor_encode_opts(&cover, &long[..over], "pw", &opts).is_err(), "{name}: payload {} must not fit", size(over));
    }
}

#[test]
fn legacy_index_variant_is_read_by_the_shipped_decoder() {
    let cover = vector("photo_640x480_q75_420.jpg");
    let msg = "legacy wire, index-domain step";
    let opts = ArmorEncodeOpts {
        stdm: StdmVariant::LegacyIdx { h: 1.0 },
        rounding: StdmRounding::Search { tol: 0.05 },
        force_stdm: true,
        ..ArmorEncodeOpts::legacy()
    };
    let (st, _) = armor_encode_opts(&cover, msg, "pw", &opts).unwrap();
    assert_eq!(armor_decode_opts(&st, "pw", &ArmorDecodeOpts::shipped()).unwrap().0.text, msg);
}

#[test]
fn legacy_encoder_still_round_trips() {
    let cover = vector("photo_640x480_q75_420.jpg");
    let msg = "legacy encoder";
    let opts = ArmorEncodeOpts { force_stdm: true, ..ArmorEncodeOpts::legacy() };
    let (st, _) = armor_encode_opts(&cover, msg, "pw", &opts).unwrap();
    assert_eq!(armor_decode_opts(&st, "pw", &ArmorDecodeOpts::shipped()).unwrap().0.text, msg);
    assert_eq!(armor_decode(&st, "pw").unwrap().0.text, msg);
}

/// Prefixes of one random printable string; returns (text, payload size) of
/// the longest prefix whose payload is <= cap, and the shortest one above it.
fn fit_prefix(cap: usize) -> (String, String) {
    let mut s = 0x2545_f491u64;
    let long: String = (0..4 * cap + 64)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (b'!' + (s % 94) as u8) as char
        })
        .collect();
    let size = |n: usize| phasm_core::stego::payload::compressed_payload_size(&long[..n], &[]);
    let fit = (1..long.len()).take_while(|&n| size(n) <= cap).last().unwrap();
    let over = (fit + 1..long.len()).find(|&n| size(n) > cap).unwrap();
    (long[..fit].to_string(), long[..over].to_string())
}

#[test]
fn fortress_t_is_the_default_with_exact_capacity() {
    let cover = vector("photo_640x480_q75_420.jpg");
    let img = JpegImage::from_bytes(&cover).unwrap();
    let info = armor_capacity_info(&cover).unwrap();
    assert_eq!(info.fortress_capacity, fortress_capacity(&img).unwrap());
    for (pass, cap) in [("pw", info.fortress_capacity), ("", fortress_capacity_compact(&img).unwrap())] {
        assert!(cap > 8, "Fortress T fits this cover ({cap})");
        let (fit, over) = fit_prefix(cap);
        let st = armor_encode(&cover, &fit, pass).unwrap();
        let (p, q, di) = armor_decode_opts(&st, pass, &ArmorDecodeOpts::default()).unwrap();
        assert_eq!(p.text, fit);
        assert!(q.fortress_used);
        assert_eq!(di.path, "fortress");
        // core <= 0.5.x reads only the old Fortress magic
        assert!(armor_decode_opts(&st, pass, &ArmorDecodeOpts::shipped()).is_err());
        // one payload byte more takes the STDM path
        let st = armor_encode(&cover, &over, pass).unwrap();
        let (p, q, di) = armor_decode_opts(&st, pass, &ArmorDecodeOpts::default()).unwrap();
        assert_eq!(p.text, over);
        assert!(!q.fortress_used, "{pass:?}: over-capacity message must not be Fortress");
        assert_eq!(di.path, "v2");
    }
}

#[test]
fn legacy_fortress_still_decodes() {
    let cover = vector("photo_640x480_q75_420.jpg");
    let opts = ArmorEncodeOpts::legacy();
    // the legacy format needs r >= 15: tiny message only
    let (st, info) = armor_encode_opts(&cover, "hi", "pw", &opts).unwrap();
    if info.fortress {
        assert_eq!(armor_decode(&st, "pw").unwrap().0.text, "hi");
        assert_eq!(armor_decode_opts(&st, "pw", &ArmorDecodeOpts::shipped()).unwrap().0.text, "hi");
    }
}
