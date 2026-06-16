// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! WV.6.g.4 SHIP GATE — the streaming shadow encode must emit a
//! **byte-identical** stego clip to the whole-clip `h264_encode_with_shadows`.
//!
//! This is the proof obligation for the gated parallel build (§9 / §9.1 of
//! `docs/design/video/h264/oh264-wv6-streaming-shadow-unification.md`): the
//! truly-streaming 2-sweep pipeline replaces the whole-clip function's body
//! increment by increment, and at EVERY increment this assert must hold —
//! same bytes ⇒ wire-compatible (existing stego files keep decoding) AND
//! correct (the streaming path made the same decisions). The call site is
//! swapped to the streaming entry point only once this is permanently green.
//!
//! At g.4.0 the streaming fn is a scaffold (pull GOPs forward → reassemble →
//! delegate), so the gate also proves the `GopYuvSource` surface + the GOP
//! slicing/reassembly math (incl. a partial trailing GOP) before any real
//! streaming logic lands.
//!
//! Run the encode gate (process-global OH264 fork state ⇒ single-threaded):
//!   cargo test -p phasm-core --features h264-encoder --test \
//!     oh264_g4_streaming_byte_identical -- --ignored --test-threads=1 --nocapture

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::h264::openh264_stego::{
    gop_clean_cover, gop_provisional_step, gop_safe_msl_prov, h264_encode_with_shadows,
    h264_encode_with_shadows_streaming, prep_primary_payload, streaming_provisional_plan,
    streaming_shadow_verify, sweep_a, sweep_b_emit, whole_clip_baseline_cover,
    whole_clip_primary_plan,
    whole_clip_provisional_cover, whole_clip_resolved_tier, whole_clip_safe_msl_prov,
    CallbackYuvSource, EncodeOpts, FileYuvSource, GopYuvSource, ShadowSelectionSweep,
    SliceYuvSource,
};
use phasm_core::codec::h264::stego::shadow::build_shadow_rs_frame;
use phasm_core::codec::h264::stego::orchestrate::DomainCosts;
use phasm_core::codec::h264::stego::shadow::{prepare_shadow_over_emit_cover, ShadowSlot};
use phasm_core::codec::h264::stego::{CostWeights, DomainBits, DomainCover};
use phasm_core::stego::shadow_layer::{ShadowLayer, SHADOW_PARITY_TIERS};

/// Deterministic textured tight-I420 clip (same formula as the lib-test +
/// `oh264_video_shadow_roundtrip` generators — real DCT energy so the
/// shadow path has cover positions to work with).
fn synth_yuv(w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let frame_size = (w * h * 3 / 2) as usize;
    let mut out = Vec::with_capacity(frame_size * n_frames as usize);
    let mut s: u32 = 0xCAFE_F00D;
    for frame in 0..n_frames {
        for j in 0..h {
            for i in 0..w {
                out.push(((i + frame * 2) ^ (j + frame * 3)) as u8);
            }
        }
        for _plane in 0..2 {
            for j in 0..(h / 2) {
                for i in 0..(w / 2) {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let pos = (i + j + frame) as u8;
                    out.push(((s >> 16) as u8).wrapping_add(pos));
                }
            }
        }
    }
    out
}

/// Fast, no-encode: `SliceYuvSource` pulled forward GOP-by-GOP must
/// reassemble the exact input buffer — including a partial trailing GOP.
#[test]
fn slice_yuv_source_reassembles_exactly() {
    let (w, h) = (64u32, 48u32);
    let frame_bytes = (w * h * 3 / 2) as usize;
    // (n_frames, gop_size): exact multiple, partial tail, single GOP,
    // gop larger than the clip, and a 1-frame clip.
    for &(n, gop) in &[(20u32, 10u32), (25, 10), (10, 10), (7, 16), (1, 4), (33, 8)] {
        let yuv = synth_yuv(w, h, n);
        let n_gops = n.div_ceil(gop).max(1);
        let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
        src.rewind().expect("rewind");
        let mut reassembled = Vec::with_capacity(yuv.len());
        for g in 0..n_gops {
            reassembled.extend_from_slice(&src.gop_yuv(g).expect("gop_yuv"));
        }
        assert_eq!(
            reassembled.len(),
            yuv.len(),
            "reassembled length mismatch for n={n} gop={gop} (frame_bytes={frame_bytes})"
        );
        assert_eq!(reassembled, yuv, "reassembled bytes differ for n={n} gop={gop}");
        // Past-the-end GOP index yields empty (the encoder never asks, but
        // the contract guarantees it).
        assert!(
            src.gop_yuv(n_gops).expect("past-end gop_yuv").is_empty(),
            "past-end GOP should be empty for n={n} gop={gop}"
        );
    }
}

/// FOUNDATION the gate rests on. The shadow encode is non-deterministic BY
/// DESIGN: `crypto::encrypt` draws a random AES salt+nonce per call, so each
/// encode produces a different shadow (and primary) ciphertext → different
/// stego bytes. `PHASM_DETERMINISTIC_SEED` forces a ChaCha8-seeded reproducible
/// salt+nonce. With it pinned, two calls with identical input MUST be
/// byte-identical — which proves the orchestrator carries NO other
/// process-global state leak between encodes (the 2-sweep streaming design
/// re-encodes each GOP across sweeps in one process and depends on that).
/// Isolates the orchestrator from all streaming code (no SliceYuvSource).
#[test]
#[ignore = "OH264 shadow encode ~2×5s; run with --ignored --test-threads=1"]
fn whole_clip_shadow_deterministic_with_pinned_seed() {
    // SAFETY: single-threaded gate; set+remove the process-global diagnostic
    // seed around the two encodes so both draw identical AES salt+nonce.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (320u32, 240u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let primary_msg = "primary message, longer than the shadow";
    let primary_pass = "primary-pass";
    let shadows = [ShadowLayer { message: "shdw", passphrase: "s", files: &[] }];
    let yuv = synth_yuv(w, h, n);

    eprintln!("=== determinism (seed pinned): call 1 ===");
    let b1 = h264_encode_with_shadows(
        &yuv, w, h, n, opts, primary_msg, &[], primary_pass, &shadows, &weights,
    )
    .expect("encode 1");
    eprintln!("=== determinism (seed pinned): call 2 (same input) ===");
    let b2 = h264_encode_with_shadows(
        &yuv, w, h, n, opts, primary_msg, &[], primary_pass, &shadows, &weights,
    )
    .expect("encode 2");

    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    let first_diff = b1.iter().zip(b2.iter()).position(|(a, b)| a != b);
    eprintln!(
        "WHOLE-CLIP determinism (seed pinned): b1={} B, b2={} B, first_diff={first_diff:?}",
        b1.len(),
        b2.len(),
    );
    assert_eq!(
        b1, b2,
        "whole-clip shadow orchestrator diverges run-to-run EVEN with crypto \
         pinned (first_diff={first_diff:?}) — there is a process-global state \
         leak beyond crypto; WV.6.g's per-sweep re-encode would break."
    );
}

/// WV.6.g.4.1 — Sweep A's per-GOP clean-cover primitive must reproduce the
/// whole-clip baseline cover EXACTLY. Validates the frame_idx local→global
/// remap + the "per-GOP-standalone encode ≡ whole-clip GOP slice" property in
/// isolation (clean encodes only, no shadow crypto), so a remap / off-by-one
/// bug is localised here instead of surfacing buried in the full streaming gate.
#[test]
#[ignore = "OH264 clean encodes (whole + per-GOP); run with --ignored --test-threads=1"]
fn gop_clean_cover_assembles_to_whole_clip_baseline() {
    let (w, h, gop, n) = (128u32, 96u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let yuv = synth_yuv(w, h, n);
    let frame_bytes = (w * h * 3 / 2) as usize;
    let n_gops = n.div_ceil(gop);

    let reference =
        whole_clip_baseline_cover(&yuv, w, h, n, opts).expect("whole-clip baseline cover");

    fn append(dst: &mut DomainBits, src: &DomainBits) {
        dst.positions.extend_from_slice(&src.positions);
        dst.bits.extend_from_slice(&src.bits);
        dst.magnitudes.extend_from_slice(&src.magnitudes);
    }
    let mut asm = DomainCover::default();
    for g in 0..n_gops {
        let start = g * gop;
        let frames_in_gop = (n - start).min(gop);
        let byte_start = start as usize * frame_bytes;
        let byte_end = (start + frames_in_gop) as usize * frame_bytes;
        let gc = gop_clean_cover(&yuv[byte_start..byte_end], w, h, frames_in_gop, start, opts)
            .expect("gop clean cover");
        append(&mut asm.coeff_sign_bypass, &gc.coeff_sign_bypass);
        append(&mut asm.coeff_suffix_lsb, &gc.coeff_suffix_lsb);
        append(&mut asm.mvd_sign_bypass, &gc.mvd_sign_bypass);
        append(&mut asm.mvd_suffix_lsb, &gc.mvd_suffix_lsb);
    }

    // Per-domain digest: (raw position keys, cover bits, magnitudes).
    let dig = |c: &DomainCover| -> Vec<(Vec<u64>, Vec<u8>, Vec<u16>)> {
        [
            &c.coeff_sign_bypass,
            &c.coeff_suffix_lsb,
            &c.mvd_sign_bypass,
            &c.mvd_suffix_lsb,
        ]
        .iter()
        .map(|d| {
            (
                d.positions.iter().map(|k| k.raw()).collect::<Vec<u64>>(),
                d.bits.clone(),
                d.magnitudes.clone(),
            )
        })
        .collect()
    };
    let names = ["CSB", "CSL", "MSB", "MSL"];
    let (da, dr) = (dig(&asm), dig(&reference));
    for (i, (a, r)) in da.iter().zip(dr.iter()).enumerate() {
        let first = a.0.iter().zip(r.0.iter()).position(|(x, y)| x != y);
        eprintln!(
            "WV.6.g.4.1 [{}] assembled={} pos, reference={} pos, first_key_diff={first:?}",
            names[i],
            a.0.len(),
            r.0.len(),
        );
        assert_eq!(
            a.0, r.0,
            "{} position keys diverge (first_diff={first:?}) — per-GOP \
             encode+walk+remap does not reproduce the whole-clip cover slice",
            names[i],
        );
        assert_eq!(a.1, r.1, "{} cover bits diverge", names[i]);
        assert_eq!(a.2, r.2, "{} magnitudes diverge", names[i]);
    }
    let total: usize = dr.iter().map(|d| d.0.len()).sum();
    assert!(total > 0, "no cover positions produced — vacuous fixture");
    eprintln!("WV.6.g.4.1 GREEN — assembled per-GOP cover ≡ whole-clip baseline ({total} positions)");
}

/// WV.6.g.4.1b — the streaming shadow-position selector
/// (`ShadowSelectionSweep`) must reproduce the whole-clip
/// `prepare_shadow_over_emit_cover` selection EXACTLY: same slots, same order.
///
/// Isolates the Sweep-A selection MACHINERY — per-GOP push with running global
/// per-domain offsets, the `safe_msl` gate applied to per-GOP mask slices, and
/// the `StreamingTopN` heap's `priority_slots` tie-break — from the
/// provisional-emit decomposition that produces the real `safe_msl_prov`
/// (proven separately in g.4.1c). The streaming side consumes the per-GOP CLEAN
/// covers (g.4.1a) GOP-by-GOP; the reference selects over the whole-clip clean
/// cover. They must agree for every `safe_msl` shape — `None` (MSL excluded),
/// all-safe, and a deterministic ~50/50 mixed mask — with a shadow large enough
/// that the selection reaches the tiny MvdSuffixLsb domain (else the MSL gate is
/// never exercised). A wrong global offset, a misaligned mask slice, or a
/// mis-remapped `frame_idx` (→ wrong priority) would diverge here, localised,
/// instead of buried in the full byte-identical gate.
#[test]
#[ignore = "OH264 clean encodes (whole + per-GOP); run with --ignored --test-threads=1"]
fn sweep_a_selection_machinery_matches_priority_slots() {
    let (w, h, gop, n) = (128u32, 96u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let yuv = synth_yuv(w, h, n);
    let frame_bytes = (w * h * 3 / 2) as usize;
    let n_gops = n.div_ceil(gop);

    // Expensive part (OH264 encodes), done once: the whole-clip clean cover
    // (the reference's cover) + the per-GOP clean covers (streaming's covers).
    let whole = whole_clip_baseline_cover(&yuv, w, h, n, opts).expect("whole-clip clean cover");
    let mut per_gop: Vec<DomainCover> = Vec::with_capacity(n_gops as usize);
    for g in 0..n_gops {
        let start = g * gop;
        let frames_in_gop = (n - start).min(gop);
        let byte_start = start as usize * frame_bytes;
        let byte_end = (start + frames_in_gop) as usize * frame_bytes;
        per_gop.push(
            gop_clean_cover(&yuv[byte_start..byte_end], w, h, frames_in_gop, start, opts)
                .expect("gop clean cover"),
        );
    }

    let n_msl = whole.mvd_suffix_lsb.len();
    assert!(n_msl > 0, "fixture has no MvdSuffixLsb positions — the MSL gate would be vacuous");

    // A big, HIGH-ENTROPY shadow so n_total reaches a deep fraction of the
    // cover (the tiny MvdSuffixLsb domain participates at ~that depth). The
    // payload is Brotli-compressed (`encode_payload`), so a repeated string
    // collapses to nothing — use deterministic pseudo-random printable ASCII
    // that barely compresses. ~10 K chars → tens of Kbit RS frame, comfortably
    // under this fixture's ~117 K eligible positions.
    let shadow_msg: String = (0..10_000u32)
        .map(|i| {
            let mut z = i.wrapping_mul(0x9E37_79B1).wrapping_add(0x7F4A_7C15);
            z ^= z >> 15;
            z = z.wrapping_mul(0x85EB_CA77);
            z ^= z >> 13;
            char::from(33u8 + (z % 94) as u8) // printable ASCII 33..=126
        })
        .collect();
    let shadows = [ShadowLayer { message: &shadow_msg, passphrase: "sweep-a", files: &[] }];
    let parity = SHADOW_PARITY_TIERS[0];

    // Three mask shapes aligned to the whole-clip MSL positions: None (MSL
    // excluded entirely), all-true (MSL fully eligible), and a deterministic
    // ~50/50 hash mask (the realistic mixed case).
    let mixed: Vec<bool> = whole
        .mvd_suffix_lsb
        .positions
        .iter()
        .map(|k| (k.raw().wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 40) & 1 == 0)
        .collect();
    let all_true = vec![true; n_msl];
    let variants: [(&str, Option<&[bool]>); 3] =
        [("none", None), ("all-safe", Some(&all_true)), ("mixed", Some(&mixed))];

    for (label, safe_msl) in variants {
        // Reference: whole-clip selection over the whole-clip clean cover.
        let reference = prepare_shadow_over_emit_cover(
            &whole, shadows[0].passphrase, shadows[0].message, shadows[0].files,
            parity, None, safe_msl,
        )
        .unwrap_or_else(|e| panic!("[{label}] reference prepare_shadow failed: {e:?}"));
        let n_total = reference.n_total;

        // Streaming: push each GOP's clean cover with its slice of the mask.
        // The per-GOP MSL positions concatenate (in GOP order) to the whole-clip
        // MSL vector (g.4.1a), so the mask slices line up by accumulated offset.
        let mut sweep = ShadowSelectionSweep::new(&shadows, &[n_total]).expect("sweep new");
        let mut off_msl = 0usize;
        for gc in &per_gop {
            let gop_msl = gc.mvd_suffix_lsb.len();
            let slice = safe_msl.map(|m| &m[off_msl..off_msl + gop_msl]);
            sweep.push_gop(gc, slice);
            off_msl += gop_msl;
        }
        assert_eq!(off_msl, n_msl, "[{label}] per-GOP MSL counts must sum to the whole-clip count");
        let streamed: Vec<ShadowSlot> =
            sweep.finish().pop().expect("one shadow").into_iter().take(n_total).collect();

        // Compare slot-by-slot: (domain, intra_index, priority).
        let key = |s: &ShadowSlot| (s.domain as u8, s.intra_index, s.priority);
        let msl_selected = streamed.iter().filter(|s| s.domain as u8 == 3).count();
        let first_diff = reference
            .positions
            .iter()
            .zip(streamed.iter())
            .position(|(r, s)| key(r) != key(s));
        eprintln!(
            "WV.6.g.4.1b [{label}] n_total={n_total} ref={} streamed={} \
             MSL_selected={msl_selected} first_diff={first_diff:?}",
            reference.positions.len(),
            streamed.len(),
        );
        assert_eq!(
            reference.positions.len(),
            streamed.len(),
            "[{label}] selected-count mismatch (ref {} vs streamed {})",
            reference.positions.len(),
            streamed.len(),
        );
        for (i, (r, s)) in reference.positions.iter().zip(streamed.iter()).enumerate() {
            assert_eq!(
                key(r),
                key(s),
                "[{label}] slot {i} diverges: ref={:?} streamed={:?} — the streaming \
                 selector is not bit-identical to priority_slots",
                key(r),
                key(s),
            );
        }
        // The mask shapes must genuinely exercise MSL: all-safe/mixed should
        // pull some MvdSuffixLsb into the top-N (else the gate is vacuous).
        if label != "none" {
            assert!(
                msl_selected > 0,
                "[{label}] no MvdSuffixLsb slots selected — raise the shadow size \
                 so the MSL gate is actually exercised"
            );
        } else {
            assert_eq!(msl_selected, 0, "[none] MSL must be excluded when safe_msl=None");
        }
    }
    eprintln!("WV.6.g.4.1b GREEN — streaming selector ≡ whole-clip priority_slots (3 mask shapes)");
}

/// WV.6.g.4.1c — `safe_msl_prov` must be GOP-DECOMPOSABLE. The cascade-safety
/// analysis (`analyze_safe_mvd_subset` + `derive_msl_safe_from_msb`) is strictly
/// frame-local — every neighbour lookup / slot-key is keyed by `frame_idx`,
/// `shift_bound` accumulates per-position — so restricting the provisional walk
/// to each GOP's frames, running the analysis per-GOP, and concatenating must
/// reproduce the whole-clip `safe_msl_prov` EXACTLY. Also asserts Fact 2: under
/// `wire_only` the provisional primary emit moves no cover position, so
/// `prov_cover`'s positions equal the clean baseline cover's. Both are
/// load-bearing facts for Sweep A, proven here on a REAL provisional walk (not a
/// synthetic meta) so GOP-boundary slicing is exercised on real frame data.
#[test]
#[ignore = "OH264 baseline+provisional encodes; run with --ignored --test-threads=1"]
fn gop_safe_msl_prov_assembles_to_whole_clip() {
    let (w, h, gop, n) = (128u32, 96u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let yuv = synth_yuv(w, h, n);
    let n_gops = n.div_ceil(gop);

    // Moderate high-entropy primary so the provisional emit does real STC work
    // (Brotli would otherwise collapse a repeated string to nothing). Whether
    // the resulting safe-set is all-safe or a mix, the decomposition must hold —
    // frame-locality is structural (every cascade lookup is keyed by frame_idx);
    // this is the empirical backstop on a real provisional walk.
    let primary: String = (0..800u32)
        .map(|i| {
            let mut z = i.wrapping_mul(0x9E37_79B1).wrapping_add(0x1234_5677);
            z ^= z >> 15;
            z = z.wrapping_mul(0x85EB_CA77);
            z ^= z >> 13;
            char::from(33u8 + (z % 94) as u8)
        })
        .collect();

    let pw = whole_clip_provisional_cover(
        &yuv, w, h, n, opts, &primary, &[], "primary-pass", &weights,
    )
    .expect("whole-clip provisional cover");

    // ── Fact 2: the provisional emit moves no position (wire_only). ──
    let dom = |c: &DomainCover| -> Vec<Vec<u64>> {
        [
            &c.coeff_sign_bypass,
            &c.coeff_suffix_lsb,
            &c.mvd_sign_bypass,
            &c.mvd_suffix_lsb,
        ]
        .iter()
        .map(|d| d.positions.iter().map(|k| k.raw()).collect())
        .collect()
    };
    let names = ["CSB", "CSL", "MSB", "MSL"];
    let (clean_pos, prov_pos) = (dom(&pw.clean_cover), dom(&pw.prov_cover));
    for (i, (cl, pr)) in clean_pos.iter().zip(prov_pos.iter()).enumerate() {
        assert_eq!(
            cl.len(),
            pr.len(),
            "[Fact 2/{}] provisional emit changed the {} position COUNT ({} → {})",
            names[i],
            names[i],
            cl.len(),
            pr.len(),
        );
        assert_eq!(
            cl, pr,
            "[Fact 2/{}] provisional emit moved a {} position (wire_only should keep \
             the position set identical to the clean cover)",
            names[i], names[i],
        );
    }

    // ── Cascade decomposition: per-GOP assembly == whole-clip. ──
    let whole = whole_clip_safe_msl_prov(&pw);
    let mut assembled: Vec<bool> = Vec::with_capacity(whole.len());
    for g in 0..n_gops {
        assembled.extend(gop_safe_msl_prov(&pw, gop, g));
    }
    let safe_count = whole.iter().filter(|&&b| b).count();
    let first_diff = whole
        .iter()
        .zip(assembled.iter())
        .position(|(a, b)| a != b);
    eprintln!(
        "WV.6.g.4.1c — MSL positions: whole={} assembled={} safe={safe_count} \
         first_diff={first_diff:?}",
        whole.len(),
        assembled.len(),
    );
    assert!(!whole.is_empty(), "fixture has no MvdSuffixLsb positions — vacuous gate");
    assert_eq!(
        whole.len(),
        assembled.len(),
        "per-GOP MSL counts must sum to the whole-clip count ({} vs {})",
        whole.len(),
        assembled.len(),
    );
    assert_eq!(
        whole, assembled,
        "per-GOP safe_msl_prov assembly != whole-clip (first_diff={first_diff:?}) — \
         the cascade-safety analysis is NOT frame-local / GOP-decomposable"
    );
    eprintln!("WV.6.g.4.1c GREEN — safe_msl_prov is GOP-decomposable + Fact 2 holds");
}

/// WV.6.g.4.1d — the per-GOP primary STC (`embed_primary_one_gop` driven by
/// `streaming_provisional_plan` with a forward `cursor`-carry) must produce a
/// plan byte-identical to the whole-clip `embed_primary_per_gop_4domain` (via
/// `whole_clip_primary_plan`). Isolates the cursor-carry decomposition (cross-GOP
/// dep 2): the per-domain positions are GOP-contiguous, so each GOP's plan slice
/// concatenated in order must reconstruct the whole-clip plan block. The
/// decomposition is independent of cost VALUES (both sides use identical costs),
/// so UNIFORM costs + a SYNTHETIC primary isolate the allocation from the
/// content_costs/tier plumbing (deps 1 + 3 → g.4.2). Cover is real (g.4.1a).
#[test]
#[ignore = "OH264 clean encodes (whole + per-GOP); run with --ignored --test-threads=1"]
fn streaming_provisional_plan_byte_identical_to_whole_clip() {
    let (w, h, gop, n) = (128u32, 96u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let yuv = synth_yuv(w, h, n);
    let frame_bytes_yuv = (w * h * 3 / 2) as usize;
    let n_gops = n.div_ceil(gop);

    // Real cover (whole + per-GOP). gop_clean_cover assembles to the whole
    // baseline (g.4.1a), so the per-GOP STC cover slices line up exactly.
    let cover = whole_clip_baseline_cover(&yuv, w, h, n, opts).expect("whole-clip cover");
    let mut per_gop: Vec<DomainCover> = Vec::with_capacity(n_gops as usize);
    for g in 0..n_gops {
        let start = g * gop;
        let fig = (n - start).min(gop);
        let bs = start as usize * frame_bytes_yuv;
        let be = (start + fig) as usize * frame_bytes_yuv;
        per_gop.push(gop_clean_cover(&yuv[bs..be], w, h, fig, start, opts).expect("gop cover"));
    }

    // Uniform 1.0 costs (decomposition is cost-value-independent).
    let uniform = |c: &DomainCover| DomainCosts {
        coeff_sign_bypass: vec![1.0; c.coeff_sign_bypass.len()],
        coeff_suffix_lsb: vec![1.0; c.coeff_suffix_lsb.len()],
        mvd_sign_bypass: vec![1.0; c.mvd_sign_bypass.len()],
        mvd_suffix_lsb: vec![1.0; c.mvd_suffix_lsb.len()],
    };
    let whole_costs = uniform(&cover);
    let per_gop_costs: Vec<DomainCosts> = per_gop.iter().map(uniform).collect();

    // Synthetic primary — the decomposition holds for any payload + seed.
    let frame_bytes: Vec<u8> =
        (0..1500u32).map(|i| (i.wrapping_mul(37) ^ (i >> 2)) as u8).collect();
    let hhat_seed = [0x5Au8; 32];

    let reference = whole_clip_primary_plan(
        &cover, &whole_costs, &frame_bytes, &hhat_seed, gop, n_gops, &weights,
    )
    .expect("whole-clip primary plan");
    let streamed = streaming_provisional_plan(
        &per_gop, &per_gop_costs, &frame_bytes, &hhat_seed, &weights,
    )
    .expect("streaming provisional plan");

    let names = ["CSB", "CSL", "MSB", "MSL"];
    let pairs = [
        (&reference.coeff_sign_bypass, &streamed.coeff_sign_bypass),
        (&reference.coeff_suffix_lsb, &streamed.coeff_suffix_lsb),
        (&reference.mvd_sign_bypass, &streamed.mvd_sign_bypass),
        (&reference.mvd_suffix_lsb, &streamed.mvd_suffix_lsb),
    ];
    for (i, (r, s)) in pairs.iter().enumerate() {
        let first = r.iter().zip(s.iter()).position(|(a, b)| a != b);
        eprintln!(
            "WV.6.g.4.1d [{}] ref={} streamed={} first_diff={first:?}",
            names[i],
            r.len(),
            s.len(),
        );
        assert_eq!(r.len(), s.len(), "[{}] plan length mismatch", names[i]);
        assert_eq!(
            *r, *s,
            "[{}] per-GOP primary plan diverges from whole-clip (first_diff={first:?}) — \
             the cursor-carry decomposition is wrong",
            names[i],
        );
    }

    // Non-vacuous: the primary must actually have been embedded (plan flips
    // some cover bits), else the gate would pass on two all-cover plans.
    let flips = |plan: &[u8], cov: &[u8]| plan.iter().zip(cov).filter(|(a, b)| a != b).count();
    let total_flips = flips(&reference.coeff_sign_bypass, &cover.coeff_sign_bypass.bits)
        + flips(&reference.coeff_suffix_lsb, &cover.coeff_suffix_lsb.bits)
        + flips(&reference.mvd_sign_bypass, &cover.mvd_sign_bypass.bits)
        + flips(&reference.mvd_suffix_lsb, &cover.mvd_suffix_lsb.bits);
    assert!(total_flips > 0, "plan embedded nothing — vacuous gate");
    eprintln!(
        "WV.6.g.4.1d GREEN — streaming primary plan ≡ whole-clip ({total_flips} flips, \
         cursor-carry decomposition holds)"
    );
}

/// WV.6.g.4.2a — the per-GOP provisional Sweep-A step (`gop_provisional_step`)
/// must reproduce the whole-clip provisional emit. Assembled per-GOP provisional
/// covers (digest: positions + bits + magnitudes) + `safe_msl_prov` must equal
/// `whole_clip_provisional_cover`'s. This validates the FULL per-GOP provisional
/// emit — all three cross-GOP deps at once: dep 1 (local-frame `content_costs`),
/// dep 2 (cursor-carry primary), dep 3 (whole-clip `tier_idx` threaded in). Seed
/// pinned so the once-prepped primary (`prep_primary_payload`) matches the
/// reference's internal prep.
#[test]
#[ignore = "OH264 per-GOP clean+prov encodes; run with --ignored --test-threads=1"]
fn gop_provisional_step_assembles_to_whole_clip() {
    // SAFETY: single-threaded gate. Pin the crypto seed so the primary prepped
    // once below is byte-identical to the reference's internal prep.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (128u32, 96u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let yuv = synth_yuv(w, h, n);
    let frame_bytes_yuv = (w * h * 3 / 2) as usize;
    let n_gops = n.div_ceil(gop);
    let primary_msg = "primary payload for the g.4.2a provisional decomposition gate";
    let primary_pass = "primary-pass";

    // Reference: whole-clip provisional (preps the primary internally; seed
    // pinned ⇒ identical to prep_primary_payload below).
    let pw = whole_clip_provisional_cover(
        &yuv, w, h, n, opts, primary_msg, &[], primary_pass, &weights,
    )
    .expect("whole-clip provisional");
    let whole_safe = whole_clip_safe_msl_prov(&pw);

    // Streaming: prep ONCE, resolve the tier whole-clip, loop the per-GOP step.
    let (frame_bytes, hhat_seed) =
        prep_primary_payload(primary_msg, &[], primary_pass).expect("prep primary");
    let tier_idx = whole_clip_resolved_tier(&pw.clean_cover, opts.qp, frame_bytes.len());

    fn append(dst: &mut DomainBits, src: &DomainBits) {
        dst.positions.extend_from_slice(&src.positions);
        dst.bits.extend_from_slice(&src.bits);
        dst.magnitudes.extend_from_slice(&src.magnitudes);
    }
    let mut asm = DomainCover::default();
    let mut asm_safe: Vec<bool> = Vec::new();
    let mut cursor = 0usize;
    for g in 0..n_gops {
        let start = g * gop;
        let fig = (n - start).min(gop);
        let bs = start as usize * frame_bytes_yuv;
        let be = (start + fig) as usize * frame_bytes_yuv;
        let (gc, smp, nc) = gop_provisional_step(
            &yuv[bs..be], w, h, fig, start, opts, &frame_bytes, &hhat_seed,
            cursor, g, n_gops, tier_idx, &weights,
        )
        .expect("gop provisional step");
        cursor = nc;
        append(&mut asm.coeff_sign_bypass, &gc.coeff_sign_bypass);
        append(&mut asm.coeff_suffix_lsb, &gc.coeff_suffix_lsb);
        append(&mut asm.mvd_sign_bypass, &gc.mvd_sign_bypass);
        append(&mut asm.mvd_suffix_lsb, &gc.mvd_suffix_lsb);
        asm_safe.extend(smp);
    }
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    // Per-domain digest (positions + bits + magnitudes) of the PROVISIONAL cover.
    let dig = |c: &DomainCover| -> Vec<(Vec<u64>, Vec<u8>, Vec<u16>)> {
        [
            &c.coeff_sign_bypass,
            &c.coeff_suffix_lsb,
            &c.mvd_sign_bypass,
            &c.mvd_suffix_lsb,
        ]
        .iter()
        .map(|d| {
            (
                d.positions.iter().map(|k| k.raw()).collect::<Vec<u64>>(),
                d.bits.clone(),
                d.magnitudes.clone(),
            )
        })
        .collect()
    };
    let names = ["CSB", "CSL", "MSB", "MSL"];
    let (da, dr) = (dig(&asm), dig(&pw.prov_cover));
    for (i, (a, r)) in da.iter().zip(dr.iter()).enumerate() {
        let first = a.0.iter().zip(r.0.iter()).position(|(x, y)| x != y);
        eprintln!(
            "WV.6.g.4.2a [{}] assembled={} prov={} first_key_diff={first:?}",
            names[i],
            a.0.len(),
            r.0.len(),
        );
        assert_eq!(
            a.0, r.0,
            "[{}] per-GOP provisional cover positions diverge from whole-clip (first={first:?}) — \
             the per-GOP provisional emit does not reproduce the whole-clip provisional walk",
            names[i],
        );
        assert_eq!(a.1, r.1, "[{}] provisional cover bits diverge", names[i]);
        assert_eq!(a.2, r.2, "[{}] provisional cover magnitudes diverge", names[i]);
    }
    let first_safe = whole_safe.iter().zip(asm_safe.iter()).position(|(a, b)| a != b);
    eprintln!(
        "WV.6.g.4.2a safe_msl_prov: whole={} assembled={} first_diff={first_safe:?}",
        whole_safe.len(),
        asm_safe.len(),
    );
    assert_eq!(
        whole_safe, asm_safe,
        "per-GOP safe_msl_prov assembly != whole-clip (first={first_safe:?})"
    );
    eprintln!("WV.6.g.4.2a GREEN — per-GOP provisional emit (deps 1+2+3) ≡ whole-clip");
}

/// WV.6.g.4.2b-1 — the real Sweep A (`sweep_a`: loop `gop_provisional_step` +
/// `ShadowSelectionSweep`) must produce the same shadow selection the whole-clip
/// `prepare_shadow_over_emit_cover` does. Composes g.4.2a (real per-GOP
/// `safe_msl_prov`) with g.4.1b (selection machinery) into the actual Sweep A,
/// gated end-to-end (seed pinned so the prepped-once primary matches).
#[test]
#[ignore = "OH264 per-GOP clean+prov encodes; run with --ignored --test-threads=1"]
fn sweep_a_selection_matches_whole_clip() {
    // SAFETY: single-threaded gate; pin the seed so the once-prepped primary
    // matches the reference's internal prep.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (128u32, 96u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let yuv = synth_yuv(w, h, n);
    let primary_msg = "primary payload for the g.4.2b-1 Sweep-A selection gate";
    let primary_pass = "primary-pass";
    let shadow_msg = "shadow message — fits at parity tier 0";
    let shadows = [ShadowLayer { message: shadow_msg, passphrase: "s", files: &[] }];

    // Reference: whole-clip provisional + shadow selection at the first tier.
    let pw = whole_clip_provisional_cover(
        &yuv, w, h, n, opts, primary_msg, &[], primary_pass, &weights,
    )
    .expect("whole-clip provisional");
    let whole_safe = whole_clip_safe_msl_prov(&pw);
    let parity = SHADOW_PARITY_TIERS[0];
    let ref_state = prepare_shadow_over_emit_cover(
        &pw.prov_cover, shadows[0].passphrase, shadows[0].message, shadows[0].files,
        parity, None, Some(&whole_safe),
    )
    .expect("reference shadow selection");
    let n_total = ref_state.n_total;

    // Streaming Sweep A (capacity = this tier's n_total).
    let (frame_bytes, hhat_seed) =
        prep_primary_payload(primary_msg, &[], primary_pass).expect("prep primary");
    let tier_idx = whole_clip_resolved_tier(&pw.clean_cover, opts.qp, frame_bytes.len());
    let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
    let result = sweep_a(
        &mut src, w, h, n, opts, &frame_bytes, &hhat_seed, &shadows, &[n_total], tier_idx, &weights,
    )
    .expect("sweep_a");
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    let streamed = &result.selections[0];
    let key = |s: &phasm_core::codec::h264::stego::shadow::ShadowSlot| {
        (s.domain as u8, s.intra_index, s.priority)
    };
    let msl_sel = streamed.iter().filter(|s| s.domain as u8 == 3).count();
    let first_diff = ref_state
        .positions
        .iter()
        .zip(streamed.iter())
        .position(|(r, s)| key(r) != key(s));
    eprintln!(
        "WV.6.g.4.2b-1 n_total={n_total} ref={} streamed={} MSL_sel={msl_sel} first_diff={first_diff:?}",
        ref_state.positions.len(),
        streamed.len(),
    );
    assert_eq!(
        ref_state.positions.len(),
        streamed.len(),
        "selected-count mismatch (ref {} vs streamed {})",
        ref_state.positions.len(),
        streamed.len(),
    );
    for (i, (r, s)) in ref_state.positions.iter().zip(streamed.iter()).enumerate() {
        assert_eq!(
            key(r),
            key(s),
            "slot {i} diverges: ref={:?} streamed={:?} — Sweep A selection != whole-clip",
            key(r),
            key(s),
        );
    }
    eprintln!("WV.6.g.4.2b-1 GREEN — real Sweep A selection ≡ whole-clip prepare_shadow");
}

/// WV.6.g.4.2b-2a — the Sweep B EMIT (`sweep_a` + `sweep_b_emit`) must reproduce
/// the whole-clip orchestrator's stego bytes. This is the byte-identical proof
/// for the emit half: same fixture as the main gate, real `h264_encode_with_shadows`
/// as the reference (no duplicated reference). `sweep_b_emit` is run at each
/// parity tier; the tier whose bytes match the orchestrator's output is the tier
/// the orchestrator's verify passed at — at least one must match.
#[test]
#[ignore = "OH264 shadow encode (orchestrator + per-GOP sweeps); --ignored --test-threads=1"]
fn sweep_b_emit_matches_whole_clip() {
    // SAFETY: single-threaded gate; pin the seed so the once-prepped primary +
    // the per-tier shadow RS frames match the orchestrator's internal prep.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (320u32, 240u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let primary_msg = "primary message, longer than the shadow";
    let primary_pass = "primary-pass";
    let shadows = [ShadowLayer { message: "shdw", passphrase: "s", files: &[] }];
    let yuv = synth_yuv(w, h, n);

    // Reference: the real whole-clip orchestrator (succeeds — same as main gate).
    let reference = h264_encode_with_shadows(
        &yuv, w, h, n, opts, primary_msg, &[], primary_pass, &shadows, &weights,
    )
    .expect("orchestrator encode");

    // Sweep A (capacity = n_total at the largest parity tier — select once).
    let (frame_bytes, hhat_seed) =
        prep_primary_payload(primary_msg, &[], primary_pass).expect("prep primary");
    let cover_p1 = whole_clip_baseline_cover(&yuv, w, h, n, opts).expect("baseline cover");
    let tier_idx = whole_clip_resolved_tier(&cover_p1, opts.qp, frame_bytes.len());
    let capacities: Vec<usize> = shadows
        .iter()
        .map(|s| {
            build_shadow_rs_frame(s.passphrase, s.message, s.files, *SHADOW_PARITY_TIERS.last().unwrap())
                .expect("rs frame")
                .0
                .len()
        })
        .collect();
    let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
    let sa = sweep_a(
        &mut src, w, h, n, opts, &frame_bytes, &hhat_seed, &shadows, &capacities, tier_idx, &weights,
    )
    .expect("sweep_a");

    // Find the parity tier whose sweep_b_emit reproduces the orchestrator output.
    let mut matched: Option<usize> = None;
    for &parity in &SHADOW_PARITY_TIERS {
        let shadow_rs: Vec<(Vec<u8>, usize)> = shadows
            .iter()
            .map(|s| build_shadow_rs_frame(s.passphrase, s.message, s.files, parity).expect("rs frame"))
            .collect();
        let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
        let (bytes, _gop_lens) = sweep_b_emit(
            &mut src, w, h, n, opts, &frame_bytes, &hhat_seed, &sa, &shadow_rs, parity, tier_idx,
            &weights,
        )
        .expect("sweep_b_emit");
        let first_diff = reference.iter().zip(bytes.iter()).position(|(a, b)| a != b);
        eprintln!(
            "WV.6.g.4.2b-2a parity={parity}: ref={} streamed={} first_diff={first_diff:?}",
            reference.len(),
            bytes.len(),
        );
        if bytes == reference {
            matched = Some(parity);
            break;
        }
    }
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    assert!(
        matched.is_some(),
        "no parity tier's sweep_b_emit reproduced the orchestrator's {} bytes — \
         the Sweep B emit is not byte-identical to the whole-clip Phase-4 emit",
        reference.len(),
    );
    eprintln!(
        "WV.6.g.4.2b-2a GREEN — sweep_b_emit ≡ orchestrator ({} B) at parity {}",
        reference.len(),
        matched.unwrap(),
    );
}

/// WV.6.g.5c — `streaming_shadow_verify` (the O(GOP) per-tier verify that replaces
/// the cascade's whole-clip `walk_final` re-decode) must make the SAME decode
/// decision the decoder would: ACCEPT the emitted clip under the correct
/// passphrases, REJECT it under a wrong one. The main byte-identical gate already
/// proves it selects the same tier (output unchanged); this nails the negative
/// path — that it actually decodes the emitted bits, not just returns `true`.
#[test]
#[ignore = "OH264 shadow encode (per-GOP sweeps + streaming verify); --ignored --test-threads=1"]
fn streaming_shadow_verify_accepts_correct_rejects_wrong_pass() {
    // SAFETY: single-threaded gate; pin the seed so the prepped primary + per-tier
    // shadow RS frames match across the emit + verify.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (320u32, 240u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let primary_msg = "primary message, longer than the shadow";
    let primary_pass = "primary-pass";
    let shadows = [ShadowLayer { message: "shdw", passphrase: "s", files: &[] }];
    let yuv = synth_yuv(w, h, n);

    let (frame_bytes, hhat_seed) =
        prep_primary_payload(primary_msg, &[], primary_pass).expect("prep primary");
    let cover_p1 = whole_clip_baseline_cover(&yuv, w, h, n, opts).expect("baseline cover");
    let tier_idx = whole_clip_resolved_tier(&cover_p1, opts.qp, frame_bytes.len());
    let capacities: Vec<usize> = shadows
        .iter()
        .map(|s| {
            build_shadow_rs_frame(s.passphrase, s.message, s.files, *SHADOW_PARITY_TIERS.last().unwrap())
                .expect("rs frame")
                .0
                .len()
        })
        .collect();
    let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
    let sa = sweep_a(
        &mut src, w, h, n, opts, &frame_bytes, &hhat_seed, &shadows, &capacities, tier_idx, &weights,
    )
    .expect("sweep_a");

    // Mirror the cascade: find the first tier whose emit streaming-verifies.
    let mut found: Option<(Vec<u8>, Vec<usize>, Vec<usize>)> = None;
    for &parity in &SHADOW_PARITY_TIERS {
        let shadow_rs: Vec<(Vec<u8>, usize)> = shadows
            .iter()
            .map(|s| build_shadow_rs_frame(s.passphrase, s.message, s.files, parity).expect("rs frame"))
            .collect();
        if shadow_rs.iter().enumerate().any(|(i, (b, _))| sa.selections[i].len() < b.len()) {
            continue;
        }
        let mut src2 = SliceYuvSource::new(&yuv, w, h, n, gop);
        let (bytes, gop_lens) = sweep_b_emit(
            &mut src2, w, h, n, opts, &frame_bytes, &hhat_seed, &sa, &shadow_rs, parity, tier_idx,
            &weights,
        )
        .expect("sweep_b_emit");
        let n_totals: Vec<usize> = shadow_rs.iter().map(|(b, _)| b.len()).collect();
        if streaming_shadow_verify(&bytes, &gop_lens, w, h, gop, &shadows, &n_totals, None).expect("verify") {
            found = Some((bytes, gop_lens, n_totals));
            break;
        }
    }
    let (bytes, gop_lens, n_totals) =
        found.expect("at least one tier must emit + streaming-verify (== the accepted tier)");

    // Negative control: a wrong passphrase derives a different perm_seed → reads
    // different positions → must NOT decode. Proves the verify truly decodes.
    let wrong = [ShadowLayer { message: shadows[0].message, passphrase: "WRONG-passphrase", files: &[] }];
    let wrong_ok = streaming_shadow_verify(&bytes, &gop_lens, w, h, gop, &wrong, &n_totals, None)
        .expect("verify (wrong pass)");

    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    assert!(
        !wrong_ok,
        "streaming_shadow_verify accepted a WRONG passphrase — it is not actually decoding",
    );
    eprintln!(
        "WV.6.g.5c GREEN — streaming verify accepts correct pass ({} B, {} GOPs), rejects wrong pass",
        bytes.len(),
        gop_lens.len(),
    );
}

/// WV.6.g.4.4a — `CallbackYuvSource` (the FFI-friendly pull source the bridges
/// will wrap around their native decoder) must drive the streaming encoder to
/// the SAME bytes as `SliceYuvSource`. Proves the callback-based source is a
/// correct `GopYuvSource` end-to-end (through tier pre-pass + Sweep A + cascade +
/// Sweep B + verify), so the only remaining per-bridge work is implementing the
/// decode callback (AVAssetReader / MediaCodec / demux).
#[test]
#[ignore = "OH264 streaming shadow encode ×2; run with --ignored --test-threads=1"]
fn callback_yuv_source_byte_identical_to_slice() {
    // SAFETY: single-threaded gate; pin the seed so the two encodes match.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (320u32, 240u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let primary_msg = "primary message, longer than the shadow";
    let primary_pass = "primary-pass";
    let shadows = [ShadowLayer { message: "shdw", passphrase: "s", files: &[] }];
    let yuv = synth_yuv(w, h, n);
    let frame_bytes_yuv = (w * h * 3 / 2) as usize;

    // Reference: the whole-buffer slicer.
    let mut src1 = SliceYuvSource::new(&yuv, w, h, n, gop);
    let bytes_slice = h264_encode_with_shadows_streaming(
        &mut src1, w, h, n, opts, primary_msg, &[], primary_pass, &shadows, &weights, None,
    )
    .expect("slice-source encode");

    // CallbackYuvSource: closures slice the same buffer per GOP (a real bridge
    // would re-decode the source video here instead). `rewind` is a no-op for the
    // stateless slicer.
    let decode_gop = |g: u32| -> Result<Vec<u8>, _> {
        let start_f = g * gop;
        if start_f >= n {
            return Ok(Vec::new());
        }
        let end_f = ((g + 1) * gop).min(n);
        Ok(yuv[start_f as usize * frame_bytes_yuv..end_f as usize * frame_bytes_yuv].to_vec())
    };
    let mut src2 = CallbackYuvSource::new(decode_gop, || Ok(()));
    let bytes_cb = h264_encode_with_shadows_streaming(
        &mut src2, w, h, n, opts, primary_msg, &[], primary_pass, &shadows, &weights, None,
    )
    .expect("callback-source encode");

    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    let first_diff = bytes_slice.iter().zip(bytes_cb.iter()).position(|(a, b)| a != b);
    eprintln!(
        "WV.6.g.4.4a — slice={} B callback={} B first_diff={first_diff:?}",
        bytes_slice.len(),
        bytes_cb.len(),
    );
    assert_eq!(
        bytes_slice, bytes_cb,
        "CallbackYuvSource drove the streaming encoder to different bytes than \
         SliceYuvSource (first_diff={first_diff:?}) — the pull source is not a \
         faithful GopYuvSource"
    );
    eprintln!("WV.6.g.4.4a GREEN — CallbackYuvSource ≡ SliceYuvSource ({} B)", bytes_slice.len());
}

/// The ship gate: streaming == whole-clip, byte-for-byte, on a real
/// primary + shadow encode (2 GOPs, primary longer than the shadow so the
/// n-shadow auto-sort keeps it primary).
#[test]
#[ignore = "OH264 shadow encode ~2×5s; run with --ignored --test-threads=1"]
fn streaming_shadow_byte_identical_to_whole_clip() {
    // Pin the diagnostic crypto seed so both encodes draw identical AES
    // salt+nonce — otherwise the random per-encrypt salt/nonce alone would
    // make any two encodes differ (proven by the determinism test above).
    // SAFETY: single-threaded gate.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "42") };

    let (w, h, gop, n) = (320u32, 240u32, 10u32, 20u32);
    let opts = EncodeOpts { qp: 26, intra_period: gop as i32 };
    let weights = CostWeights::default();
    let primary_msg = "primary message, longer than the shadow";
    let primary_pass = "primary-pass";
    let shadows = [ShadowLayer { message: "shdw", passphrase: "s", files: &[] }];

    let yuv = synth_yuv(w, h, n);

    eprintln!("=== g.4 gate: whole-clip reference encode ===");
    let reference = h264_encode_with_shadows(
        &yuv, w, h, n, opts, primary_msg, &[], primary_pass, &shadows, &weights,
    )
    .expect("whole-clip encode");

    eprintln!("=== g.4 gate: streaming encode (SliceYuvSource) ===");
    let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
    let streamed = h264_encode_with_shadows_streaming(
        &mut src, w, h, n, opts, primary_msg, &[], primary_pass, &shadows, &weights, None,
    )
    .expect("streaming encode");

    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };

    let first_diff = reference
        .iter()
        .zip(streamed.iter())
        .position(|(a, b)| a != b);
    eprintln!(
        "WV.6.g.4 GATE — {w}x{h} x{n}f gop={gop} primary+1shadow: \
         reference={} B, streamed={} B, first_diff={first_diff:?} \
         (None + equal len ⇒ byte-identical)",
        reference.len(),
        streamed.len(),
    );
    assert_eq!(
        reference, streamed,
        "streaming shadow encode diverged from the whole-clip path \
         (first_diff={first_diff:?}) — wire-compat / correctness gate FAILED"
    );
}

/// WV.6.g CLI.1 — `FileYuvSource` reads the same per-GOP YUV that `SliceYuvSource`
/// slices from memory, but from disk (O(GOP) RAM). Prove the two are byte-identical
/// per GOP over the same tight-I420 clip — including the past-the-last-GOP empty
/// signal. Fast (no encode), so NOT `#[ignore]`.
#[test]
fn file_yuv_source_matches_slice_yuv_source() {
    let (w, h, gop, n) = (64u32, 48u32, 4u32, 11u32); // 11 frames, gop 4 → 3 GOPs (4+4+3)
    let yuv = synth_yuv(w, h, n);

    // Write the clip to a unique temp file.
    let path = std::env::temp_dir().join(format!(
        "phasm_fileyuvsource_gate_{}.yuv",
        std::process::id()
    ));
    std::fs::write(&path, &yuv).expect("write temp yuv");

    let mut slice = SliceYuvSource::new(&yuv, w, h, n, gop);
    let mut file = FileYuvSource::open(&path, w, h, n, gop).expect("open FileYuvSource");

    // Compare every GOP plus two past-the-end indices (both must be empty).
    let n_gops = n.div_ceil(gop);
    for g in 0..(n_gops + 2) {
        let a = slice.gop_yuv(g).expect("slice gop");
        let b = file.gop_yuv(g).expect("file gop");
        assert_eq!(a, b, "FileYuvSource != SliceYuvSource at GOP {g}");
        if g >= n_gops {
            assert!(a.is_empty(), "GOP {g} past end must be empty");
        }
    }

    // rewind() then re-read GOP 0 — still correct after an absolute reset.
    file.rewind().expect("file rewind");
    slice.rewind().expect("slice rewind");
    assert_eq!(
        slice.gop_yuv(0).expect("slice gop0"),
        file.gop_yuv(0).expect("file gop0"),
        "GOP 0 mismatch after rewind"
    );

    let _ = std::fs::remove_file(&path);
    eprintln!("WV.6.g CLI.1 GREEN — FileYuvSource ≡ SliceYuvSource ({n_gops} GOPs, {} B)", yuv.len());
}
