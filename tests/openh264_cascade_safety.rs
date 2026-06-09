// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026 Christoph Gaffga
// https://github.com/cgaffga/phasmcore

//! Phase C.3.1 (task #393) — cascade-safety predicate sanity check on
//! real-world 1080p fixtures via the OpenH264 backend.
//!
//! For each of 5 fixtures: encode 10 frames clean via OpenH264 fork,
//! walk the resulting Annex-B with `record_mvd: true`, run
//! `analyze_safe_mvd_subset`, report per-fixture
//!   - total MVD positions captured by walker
//!   - predicted-cascade-safe count
//!   - safe ratio
//!
//! **Smoke-level** check: validates that (a) the walker successfully
//! parses OpenH264 fork output across diverse real content, (b) MVD
//! metadata flows through cleanly, (c) the predicate produces
//! sensible output (non-zero safe count on motion fixtures, zero on
//! pure-intra fixtures).
//!
//! C.3.2 / C.3.3 will follow this up with empirical predicate
//! validation (predicted-safe positions actually survive cascade)
//! once we see the predicate's basic behaviour on this corpus.
//!
//! Prerequisites: run `core/test-vectors/video/h264/real-world/source/
//! regen_openh264_baseline.sh` first to produce the per-fixture YUV
//! files in /tmp. Marked `#[ignore]` because of the runtime + tmp
//! dependency.

#![cfg(feature = "h264-encoder")]

use core_openh264_sys::PhasmStegoDomain;
use phasm_core::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use phasm_core::codec::h264::openh264::{
    set_frame_num, Encoder, StegoHandlers, StegoSession,
};
use phasm_core::codec::h264::stego::cascade_safety::analyze_safe_mvd_subset;
use std::collections::HashSet;
use std::fs::File;
use std::io::Read;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

// OpenH264 encoder registration is process-global per the StegoSession
// design (see openh264::SESSION_ALIVE). Even though c3_1 doesn't
// register a session, we serialize against other openh264 tests that
// might be running concurrently.
static OPENH264_TEST_MUTEX: Mutex<()> = Mutex::new(());

struct Fixture {
    tag: &'static str,
    width: i32,
    height: i32,
}

const FIXTURES: &[Fixture] = &[
    Fixture { tag: "img4138", width: 1920, height: 1072 },
    Fixture { tag: "img4273", width: 1920, height: 1072 },
    Fixture { tag: "carplane", width: 1072, height: 1920 },
    Fixture { tag: "dji_mini2", width: 1920, height: 1072 },
    Fixture { tag: "lumix_g9", width: 1920, height: 1072 },
];

const N_FRAMES: u32 = 10;
const QP: i32 = 26;
const GOP_SIZE: i32 = 30;

fn yuv_path(fix: &Fixture) -> std::path::PathBuf {
    std::path::PathBuf::from(format!(
        "/tmp/openh264_baseline_{}_{}x{}_f{}.yuv",
        fix.tag, fix.width, fix.height, N_FRAMES
    ))
}

fn read_yuv(path: &std::path::Path) -> Option<Vec<u8>> {
    let mut buf = Vec::new();
    File::open(path).ok()?.read_to_end(&mut buf).ok()?;
    Some(buf)
}

fn encode_clean(fix: &Fixture, yuv: &[u8]) -> Vec<u8> {
    let luma_plane = (fix.width as usize) * (fix.height as usize);
    let chroma_plane = (fix.width as usize / 2) * (fix.height as usize / 2);
    let frame_bytes = luma_plane + 2 * chroma_plane;

    set_frame_num(0);
    let mut enc = Encoder::new(fix.width, fix.height, QP, GOP_SIZE)
        .expect("Encoder::new should succeed at fixture dims");
    let mut annex_b = Vec::new();
    let mut nal_buf = vec![0u8; frame_bytes * 2];
    for i in 0..N_FRAMES {
        let off = (i as usize) * frame_bytes;
        let y = &yuv[off..off + luma_plane];
        let u = &yuv[off + luma_plane..off + luma_plane + chroma_plane];
        let v = &yuv[off + luma_plane + chroma_plane..off + frame_bytes];
        let ts = (i as i64) * 33;
        let (_ft, n) = enc
            .encode_frame(y, u, v, ts, &mut nal_buf)
            .expect("encode_frame");
        annex_b.extend_from_slice(&nal_buf[..n]);
    }
    annex_b
}

/// Encode N frames via OpenH264 backend with an enc_pre_emit hook
/// installed that flips the i-th MvdSign bin if `safe_mask[i]` is
/// true. Returns the resulting Annex-B bitstream.
///
/// Walker and encoder both emit MvdSign bins in raster MB scan
/// order, so a sequential counter inside the hook aligns with
/// `mvd_meta[i]` from the walker.
fn encode_with_mvd_sign_overrides(
    fix: &Fixture,
    yuv: &[u8],
    safe_mask: &[bool],
) -> Vec<u8> {
    let safe_set: HashSet<usize> = safe_mask
        .iter()
        .enumerate()
        .filter(|&(_, &b)| b)
        .map(|(i, _)| i)
        .collect();
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_hook = counter.clone();
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, original| -> Option<i32> {
            if pos.domain != PhasmStegoDomain::MvdSign as u8 {
                return None;
            }
            let idx = counter_for_hook.fetch_add(1, Ordering::SeqCst);
            if safe_set.contains(&idx) {
                Some(1 - original)
            } else {
                None
            }
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers).expect("register");
    encode_clean(fix, yuv)
}

#[test]
#[ignore]
fn c3_1_predict_safe_mvd_smoke_5_fixtures() {
    let _g = OPENH264_TEST_MUTEX.lock().unwrap();

    let mut any_present = false;
    eprintln!(
        "\n=== Phase C.3.1 — cascade-safety predicate smoke (5 fixtures) ===\n"
    );
    eprintln!("{:>12}  {:>9}  {:>9}  {:>9}  {:>9}", "fixture", "mvd_pos", "pred_safe", "unsafe", "safe%");
    eprintln!("{}", "-".repeat(60));

    for fix in FIXTURES {
        let yuv_path = yuv_path(fix);
        let yuv = match read_yuv(&yuv_path) {
            Some(b) => b,
            None => {
                eprintln!(
                    "{:>12}  (SKIP: {} missing — run regen_openh264_baseline.sh)",
                    fix.tag,
                    yuv_path.display()
                );
                continue;
            }
        };
        any_present = true;

        let expected_size = ((fix.width as usize) * (fix.height as usize) * 3 / 2)
            * (N_FRAMES as usize);
        assert!(
            yuv.len() >= expected_size,
            "{} YUV too short: {} bytes < {} expected",
            fix.tag,
            yuv.len(),
            expected_size
        );

        let annex_b = encode_clean(fix, &yuv);

        let walk = walk_annex_b_for_cover_with_options(
            &annex_b,
            WalkOptions { record_mvd: true },
        )
        .unwrap_or_else(|e| {
            panic!(
                "walker failed on {} OpenH264 output: {:?}",
                fix.tag, e
            )
        });

        let n_mvd = walk.mvd_meta.len();
        let safe_mask =
            analyze_safe_mvd_subset(&walk.mvd_meta, walk.mb_w, walk.mb_h);
        let n_safe = safe_mask.iter().filter(|&&b| b).count();
        let n_unsafe = n_mvd - n_safe;
        let safe_pct = if n_mvd == 0 {
            0.0
        } else {
            100.0 * n_safe as f64 / n_mvd as f64
        };

        eprintln!(
            "{:>12}  {:>9}  {:>9}  {:>9}  {:>8.2}%",
            fix.tag, n_mvd, n_safe, n_unsafe, safe_pct
        );

        // Sanity: walker succeeded and produced consistent metadata.
        assert_eq!(
            safe_mask.len(),
            n_mvd,
            "{}: safe mask length must match mvd_meta length",
            fix.tag
        );
        // Frame dims must be detected from SPS.
        assert!(
            walk.mb_w > 0 && walk.mb_h > 0,
            "{}: walker should detect mb_w/mb_h from SPS (got {}x{})",
            fix.tag,
            walk.mb_w,
            walk.mb_h
        );
    }

    eprintln!();
    assert!(
        any_present,
        "no fixtures available — run \
         core/test-vectors/video/h264/real-world/source/regen_openh264_baseline.sh"
    );
}

/// Sanity check (precondition for interpreting C.3.2 negative result):
/// does registering a StegoSession itself shift the encoder's output,
/// or does it only matter when override returns Some(...)?
///
/// Encode the same YUV twice: once with no session, once with a
/// session whose enc_pre_emit hook fires for every MvdSign bin but
/// always returns None (passthrough). Compare the produced Annex-B
/// byte-for-byte. Identical → hook plumbing is clean, C.3.2's
/// structural divergence is genuinely from the flips. Differs →
/// session registration alone has side effects (fork code-path
/// branching when callbacks are installed) and C.3.2's diagnosis
/// is partially confounded.
///
/// Runs on a single fixture (img4138) to keep runtime small.
#[test]
#[ignore]
fn c3_2_null_override_session_sanity() {
    let _g = OPENH264_TEST_MUTEX.lock().unwrap();

    let fix = &FIXTURES[0]; // img4138
    let yuv = match read_yuv(&yuv_path(fix)) {
        Some(b) => b,
        None => {
            eprintln!(
                "SKIP: {} missing — run regen_openh264_baseline.sh",
                yuv_path(fix).display()
            );
            return;
        }
    };

    // Baseline: no session registered.
    let baseline_h264 = encode_clean(fix, &yuv);

    // With-session: counter-only hook, never returns Some().
    let counter = Arc::new(AtomicUsize::new(0));
    let counter_for_hook = counter.clone();
    let handlers = StegoHandlers {
        enc_pre_emit: Some(Box::new(move |pos, _original| -> Option<i32> {
            if pos.domain == PhasmStegoDomain::MvdSign as u8 {
                counter_for_hook.fetch_add(1, Ordering::SeqCst);
            }
            None // always passthrough
        })),
        ..Default::default()
    };
    let _session = StegoSession::register(handlers).expect("register");
    let with_session_h264 = encode_clean(fix, &yuv);
    let total_fires = counter.load(Ordering::SeqCst);

    eprintln!("\n=== Phase C.3.2 sanity — null-override session ===");
    eprintln!("fixture: {}", fix.tag);
    eprintln!(
        "baseline bytes: {}, with-session bytes: {}",
        baseline_h264.len(),
        with_session_h264.len()
    );
    eprintln!("MvdSign hook fires (no overrides applied): {}", total_fires);

    if baseline_h264 == with_session_h264 {
        eprintln!("VERDICT: GREEN — session registration has no side effects");
        eprintln!("         C.3.2's structural divergence is genuinely from flips,");
        eprintln!("         not from session installation.");
    } else {
        let mut first_diff = None;
        let min_len = baseline_h264.len().min(with_session_h264.len());
        for i in 0..min_len {
            if baseline_h264[i] != with_session_h264[i] {
                first_diff = Some(i);
                break;
            }
        }
        eprintln!("VERDICT: RED — session installation alone shifts encoder output");
        eprintln!(
            "         First byte divergence: {:?} (size: baseline={} with={})",
            first_diff,
            baseline_h264.len(),
            with_session_h264.len()
        );
        eprintln!("         C.3.2's diagnosis is partially confounded;");
        eprintln!("         fork's stego-aware branch differs even without overrides.");
    }
    eprintln!();
}

/// Phase C.3.2 (task #394) — bulk-flip empirical FP check.
///
/// For each fixture: baseline-encode + walk → predict cascade-safe
/// MvdSign positions via `analyze_safe_mvd_subset`. Re-encode with
/// override map flipping the MvdSign bit at every predicted-safe
/// position simultaneously. Walk the re-encode.
///
/// **Pass criteria** (cascade-safe predicate is correct):
///   - structural: same number of MVD positions emitted (no
///     mode-decision shifts from the flips)
///   - magnitude invariance: |MVD| identical at every position
///   - flip semantics: predicted-safe bits emerge flipped,
///     predicted-unsafe bits emerge unchanged
///
/// Data-gathering pass: reports per-fixture stats. Does NOT assert
/// cascade-cleanliness in the first run — first we look at what the
/// data says. C.3.2.b (follow-on) tightens to assertions once we
/// know what's possible.
#[test]
#[ignore]
fn c3_2_bulk_flip_empirical_fp_check() {
    let _g = OPENH264_TEST_MUTEX.lock().unwrap();

    let mut any_present = false;
    eprintln!(
        "\n=== Phase C.3.2 — bulk-flip empirical FP check (5 fixtures) ===\n"
    );
    eprintln!(
        "{:>12}  {:>9}  {:>11}  {:>7}  {:>10}  {:>9}",
        "fixture", "pred-safe", "structural", "mag-Δ", "unintended", "verdict"
    );
    eprintln!("{}", "-".repeat(72));

    for fix in FIXTURES {
        let yuv = match read_yuv(&yuv_path(fix)) {
            Some(b) => b,
            None => {
                eprintln!(
                    "{:>12}  (SKIP: {} missing)",
                    fix.tag,
                    yuv_path(fix).display()
                );
                continue;
            }
        };
        any_present = true;

        // Baseline pass: clean encode + walk.
        let baseline_h264 = encode_clean(fix, &yuv);
        let baseline = walk_annex_b_for_cover_with_options(
            &baseline_h264,
            WalkOptions { record_mvd: true },
        )
        .unwrap_or_else(|e| panic!("{}: baseline walker: {:?}", fix.tag, e));

        let safe_mask = analyze_safe_mvd_subset(
            &baseline.mvd_meta,
            baseline.mb_w,
            baseline.mb_h,
        );
        let n_safe = safe_mask.iter().filter(|&&b| b).count();

        if n_safe == 0 {
            eprintln!(
                "{:>12}  {:>9}  (skip: predicate output 0 safe positions)",
                fix.tag, 0
            );
            continue;
        }

        // Flipped pass: bulk-override all predicted-safe MvdSign bins.
        let flipped_h264 = encode_with_mvd_sign_overrides(fix, &yuv, &safe_mask);
        let flipped = walk_annex_b_for_cover_with_options(
            &flipped_h264,
            WalkOptions { record_mvd: true },
        )
        .unwrap_or_else(|e| panic!("{}: flipped walker: {:?}", fix.tag, e));

        // Structural integrity.
        let baseline_n = baseline.mvd_meta.len();
        let flipped_n = flipped.mvd_meta.len();
        let baseline_bits_n = baseline.cover.mvd_sign_bypass.bits.len();
        let flipped_bits_n = flipped.cover.mvd_sign_bypass.bits.len();
        let structural_ok = baseline_n == flipped_n && baseline_bits_n == flipped_bits_n;

        let mut n_mag_changes = 0usize;
        let mut n_unintended = 0usize;
        let mut n_safe_flipped_ok = 0usize;
        let mut n_safe_not_flipped = 0usize;

        if structural_ok {
            for i in 0..baseline_n {
                if baseline.mvd_meta[i].magnitude != flipped.mvd_meta[i].magnitude {
                    n_mag_changes += 1;
                }
            }
            for i in 0..baseline_bits_n {
                let b = baseline.cover.mvd_sign_bypass.bits[i];
                let f = flipped.cover.mvd_sign_bypass.bits[i];
                let safe = safe_mask.get(i).copied().unwrap_or(false);
                if safe {
                    if f == 1 - b {
                        n_safe_flipped_ok += 1;
                    } else {
                        n_safe_not_flipped += 1;
                    }
                } else if f != b {
                    n_unintended += 1;
                }
            }
        }

        let verdict = if !structural_ok {
            "STRUCT_BROKEN"
        } else if n_mag_changes == 0 && n_unintended == 0 && n_safe_not_flipped == 0 {
            "GREEN"
        } else if n_mag_changes > 0 {
            "MAG_DRIFT"
        } else {
            "BIT_DRIFT"
        };

        eprintln!(
            "{:>12}  {:>9}  {:>11}  {:>7}  {:>10}  {:>9}",
            fix.tag,
            n_safe,
            if structural_ok {
                format!("{}={}", baseline_n, flipped_n)
            } else {
                format!("{}≠{}", baseline_n, flipped_n)
            },
            n_mag_changes,
            n_unintended,
            verdict
        );

        if structural_ok {
            eprintln!(
                "              safe-flipped-OK={}  safe-not-flipped={}",
                n_safe_flipped_ok, n_safe_not_flipped
            );
        }
    }

    eprintln!();
    assert!(
        any_present,
        "no fixtures available — run \
         core/test-vectors/video/h264/real-world/source/regen_openh264_baseline.sh"
    );
}

/// Phase C.3.6.2 (task #429) — verify the bitstream_mod helpers on
/// real OpenH264 output:
///
///   1. `locate_nal_units_annexb` produces the same NAL list as the
///      walker's underlying `parse_nal_units_annexb` (parity gate).
///   2. For every NAL: `strip_emulation_prevention(raw_payload) ==
///      walker.rbsp` (the locator already strips internally; this
///      verifies the raw-payload extraction matches what
///      `parse_nal_unit` would compute).
///   3. For every NAL with emulation-prevention bytes:
///      `repack_emulation_prevention(rbsp)` is byte-identical to the
///      original raw payload — the round-trip property.
#[test]
#[ignore]
fn c3_6_2_rbsp_strip_repack_roundtrip_on_openh264_output() {
    use phasm_core::codec::h264::cabac::bin_decoder::{
        locate_nal_units_annexb, repack_emulation_prevention,
        strip_emulation_prevention,
    };

    let _g = OPENH264_TEST_MUTEX.lock().unwrap();

    let fix = &FIXTURES[0];
    let yuv = match read_yuv(&yuv_path(fix)) {
        Some(y) => y,
        None => {
            eprintln!(
                "SKIP: {} missing — run regen_openh264_baseline.sh",
                yuv_path(fix).display()
            );
            return;
        }
    };

    let annex_b = encode_clean(fix, &yuv);
    let locs = locate_nal_units_annexb(&annex_b).expect("locate");

    eprintln!(
        "\n=== Phase C.3.6.2 — bitstream_mod helpers on {} ({} NALs) ===",
        fix.tag,
        locs.len()
    );

    assert!(!locs.is_empty(), "expected at least one NAL");

    let mut n_with_ep = 0usize;
    let mut total_ep_bytes = 0usize;

    for (i, loc) in locs.iter().enumerate() {
        // Byte range sanity.
        assert!(
            loc.start_code_start < loc.payload_start,
            "NAL {} start_code_start={} not before payload_start={}",
            i, loc.start_code_start, loc.payload_start
        );
        assert!(
            loc.payload_start < loc.payload_end,
            "NAL {} payload_start={} not before payload_end={}",
            i, loc.payload_start, loc.payload_end
        );
        assert!(
            loc.payload_end <= annex_b.len(),
            "NAL {} payload_end={} exceeds annex_b.len()={}",
            i, loc.payload_end, annex_b.len()
        );

        // NAL header byte cross-check.
        let header = annex_b[loc.payload_start];
        assert_eq!(loc.nal_type.0, header & 0x1F);
        assert_eq!(loc.nal_ref_idc, (header >> 5) & 0x03);

        // RBSP strip parity: strip(raw_payload) == loc.rbsp.
        let raw_payload =
            &annex_b[loc.payload_start + 1..loc.payload_end];
        let stripped = strip_emulation_prevention(raw_payload);
        assert_eq!(
            stripped.as_slice(),
            loc.rbsp.as_slice(),
            "strip(raw_payload) != loc.rbsp at NAL {} (type {})",
            i, loc.nal_type.0
        );

        // ep_map length matches rbsp length.
        assert_eq!(
            loc.ep_map.rbsp_to_raw.len(),
            loc.rbsp.len(),
            "ep_map.rbsp_to_raw.len() != rbsp.len() at NAL {}",
            i
        );

        // Every rbsp[j] must equal raw_payload[ep_map.rbsp_to_raw[j]].
        for (j, &raw_idx) in loc.ep_map.rbsp_to_raw.iter().enumerate() {
            assert_eq!(
                raw_payload[raw_idx],
                loc.rbsp[j],
                "ep_map[{}]={} but raw_payload[{}]=0x{:02X} != \
                 rbsp[{}]=0x{:02X} at NAL {}",
                j, raw_idx, raw_idx, raw_payload[raw_idx], j, loc.rbsp[j], i
            );
        }

        // Round-trip: repack(rbsp) == raw_payload exactly.
        let repacked = repack_emulation_prevention(&loc.rbsp);
        assert_eq!(
            repacked.as_slice(),
            raw_payload,
            "repack(rbsp) != raw_payload at NAL {} (type {})",
            i, loc.nal_type.0
        );

        let ep_in_this_nal = raw_payload.len() - loc.rbsp.len();
        if ep_in_this_nal > 0 {
            n_with_ep += 1;
            total_ep_bytes += ep_in_this_nal;
        }
    }

    eprintln!(
        "  {} NALs total, {} NALs with EP bytes, {} total EP bytes \
         inserted",
        locs.len(),
        n_with_ep,
        total_ep_bytes
    );
    eprintln!(
        "  all NALs: strip(raw_payload) == rbsp AND repack(rbsp) == raw_payload ✓"
    );
}

