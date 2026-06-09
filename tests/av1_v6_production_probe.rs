// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D.5 — V6 multi-GOP production probe.
//!
//! Multi-GOP analog of TG-2 (`av1_gop_boundary_probe.rs`). Where TG-2
//! runs N independent single-frame encodes (each via legacy
//! `av1_stego_embed`), D.5 runs ONE `Av1StreamingEncodeSession` with
//! N push_frame calls, splits the emitted bytes back into per-GOP
//! slabs via the D.4 OBU walker, decodes each slab independently,
//! and checks the stego-damage cliff across GOPs.
//!
//! ## What V6 architecture guarantees
//!
//! Per `[[phase-c-streaming-session-v6.md]]` § 2, V6 invariants
//! (per-GOP fresh `Encoder::new`, chunk_frame header, pre-encrypted +
//! split, gop_idx=0 seed sharing) ensure each GOP is structurally
//! independent. So D.5's multi-GOP cliff should equal TG-2's
//! single-encode cliff — within measurement noise.
//!
//! ## What this probe catches
//!
//! - **Cross-GOP state leak** in the session encode path. If
//!   anything (RDO state, rayon thread pool state, cached buffers,
//!   etc.) leaked from GOP N to GOP N+1, the per-GOP damage would
//!   correlate with position-in-session and the cliff would widen.
//! - **chunk_frame wire format drift**. If a future change broke
//!   the v2 header layout, the per-GOP STC would still run but the
//!   decode-side assembly would corrupt the payload bytes. The
//!   round-trip assertion catches that.
//! - **OBU walker bugs**. If `split_av1_into_gops` mis-attributed
//!   bytes between slabs, the per-slab decode would produce
//!   garbage YUV → stego_psnr would collapse → cliff would spike.
//!
//! ## Why this exists (H.264 lesson)
//!
//! H.264's V5 path SHIPPED and passed every cheap gate
//! (round-trip, cover-bit count, drift, AUC) — failing only on
//! the V5 production probe at 1080p × 45f IBPBP, with GOP-1
//! frame-31 cliff at Y -10 to -14 dB. V5 was wrong by construction;
//! V6 (per-GOP STC) closes the bug class. AV1's V6 architecture
//! adopted V6 from day 1, but the cliff probe is the regression
//! gate that locks the invariant.
//!
//! Reference: `memory/h264_v5_production_finding_2026_05_07.md`.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::path::PathBuf;
use std::process::Command;

use phasm_core::codec::av1::stego::session::{
    Av1StreamingDecodeSession, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};

// D.5 uses the SAME fixture + seek points as TG-2
// (`av1_gop_boundary_probe.rs`) so the multi-GOP cliff can be
// compared directly with TG-2's single-encode cliff at the same
// content. carplane @ q=30 is the calibration reference for the
// locked B.1.5.6 defaults; iphone_img4138 has materially different
// content variance and a different per-frame baseline.
const W: u32 = 144;
const H: u32 = 256;
const Q: usize = 30;
const SOURCE: &str = "Artlist_CarPlane.mp4";

/// Damage-cliff threshold. TG-2 (single-encode) measures 1.90 dB at
/// locked B.1.5.6 defaults on the same fixture; D.5 (multi-GOP
/// session) should be at most ~0.5 dB worse to demonstrate the V6
/// architecture's per-GOP independence. 3.0 dB gives margin for
/// per-frame content variance without hiding real cross-GOP
/// regressions.
const D5_DAMAGE_CLIFF_DB: f64 = 3.0;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(seek_s: f32) -> Vec<u8> {
    let src = corpus_root().join(SOURCE);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!("scale={W}:{H}:force_original_aspect_ratio=disable");
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v",
            "1",
            "-vf",
            &vf,
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    let expected = (W * H * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

fn build_ivf_single_frame(obus: &[u8], width: u16, height: u16) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + 12 + obus.len());
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"AV01");
    out.extend_from_slice(&width.to_le_bytes());
    out.extend_from_slice(&height.to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(obus.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(obus);
    out
}

fn decode_av1_to_yuv(av1_bytes: &[u8]) -> Vec<u8> {
    // Pipe IVF via stdin to avoid temp-file path collisions when
    // multiple tests in this binary run concurrently — cargo's
    // default in-binary parallelism can race two `decode_av1_to_yuv`
    // calls that happen to produce same-sized packets, both writing
    // to the same `d5_decode_PID_LEN.ivf` path.
    use std::io::Write;
    use std::process::Stdio;

    let ivf = build_ivf_single_frame(av1_bytes, W as u16, H as u16);
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", "pipe:0"])
        .args([
            "-frames:v",
            "1",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rawvideo",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("ffmpeg decode spawn");
    child
        .stdin
        .as_mut()
        .expect("ffmpeg stdin")
        .write_all(&ivf)
        .expect("write ivf to ffmpeg stdin");
    let out = child.wait_with_output().expect("ffmpeg wait");
    assert!(out.status.success(), "ffmpeg decode failed: {:?}", out);
    out.stdout
}

fn compute_psnr_y(source: &[u8], reconstructed: &[u8]) -> f64 {
    let y_size = (W * H) as usize;
    assert!(source.len() >= y_size);
    assert!(reconstructed.len() >= y_size);
    let mut sum_sq_err: u64 = 0;
    for i in 0..y_size {
        let diff = source[i] as i32 - reconstructed[i] as i32;
        sum_sq_err += (diff * diff) as u64;
    }
    let mse = sum_sq_err as f64 / y_size as f64;
    if mse < 0.001 {
        return 100.0;
    }
    10.0 * (65025.0 / mse).log10()
}

/// D.4-style OBU walker for the probe. We need to split the
/// session's concatenated output into per-GOP slabs to decode each
/// frame separately. This is the same logic the decode session uses
/// internally; we re-implement here so the test doesn't depend on
/// the `pub(crate)` visibility of `split_av1_into_gops`.
fn split_into_gops(bytes: &[u8]) -> Vec<Vec<u8>> {
    const OBU_SEQUENCE_HEADER: u8 = 1;
    let mut slabs: Vec<Vec<u8>> = Vec::new();
    let mut current: Option<Vec<u8>> = None;
    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let hb = bytes[cursor];
        if hb & 0x80 != 0 {
            break;
        }
        let obu_type = (hb >> 3) & 0x0f;
        let has_extension = (hb >> 2) & 1 != 0;
        let has_size = (hb >> 1) & 1 != 0;
        let header_len = 1 + (has_extension as usize);
        if cursor + header_len > bytes.len() {
            break;
        }
        let (payload_len, size_field_len) = if has_size {
            let mut v: u64 = 0;
            let mut shift = 0u32;
            let mut i = cursor + header_len;
            let mut c = 0usize;
            let mut got = false;
            while i < bytes.len() && c < 8 {
                let b = bytes[i];
                v |= ((b & 0x7f) as u64) << shift;
                c += 1;
                i += 1;
                if b & 0x80 == 0 {
                    got = true;
                    break;
                }
                shift += 7;
            }
            if !got {
                break;
            }
            (v, c)
        } else {
            ((bytes.len() - cursor - header_len) as u64, 0)
        };
        let total = header_len + size_field_len + payload_len as usize;
        if cursor + total > bytes.len() {
            break;
        }
        if obu_type == OBU_SEQUENCE_HEADER {
            if let Some(s) = current.take() {
                slabs.push(s);
            }
            current = Some(Vec::new());
        }
        if let Some(ref mut buf) = current {
            buf.extend_from_slice(&bytes[cursor..cursor + total]);
        }
        cursor += total;
    }
    if let Some(s) = current {
        slabs.push(s);
    }
    slabs
}

/// D.5 control: encode the SAME 3 frames via 3 separate single-GOP
/// sessions (one session per frame, each producing 1 GOP). If the
/// multi-GOP cliff matches the control cliff, the variance is
/// content-driven, not cross-GOP state leak. If multi-GOP is
/// systematically worse, V6 invariant #1 (per-GOP fresh state) is
/// being violated.
#[test]
fn d5_control_independent_sessions_3_gops() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260604");
    }
    let passphrase = "d5-control-2026-06-04";
    let seeks = [0.5_f32, 1.0, 1.5];

    let mut stego_damage: Vec<f64> = Vec::with_capacity(seeks.len());
    for (i, &seek_s) in seeks.iter().enumerate() {
        let yuv = extract_yuv420_frame(seek_s);
        let params = Av1StreamingEncodeParams {
            width: W,
            height: H,
            quantizer: Q,
            gop_size: 1,
            total_frames_hint: 1,
        };
        // Per-frame message proportional to D.5's per-GOP chunk
        // payload (~39 bytes). Independent passphrase deviation
        // doesn't matter for the visual measurement — STC picks
        // positions based on cost, not which message bits land.
        let per_frame_message = format!("D.5 control GOP {i}");
        let mut session = Av1StreamingEncodeSession::create(
            passphrase,
            per_frame_message.as_bytes(),
            params,
        )
        .unwrap();
        let mut out = Vec::new();
        session.push_frame(&yuv, &mut out).unwrap();
        session.finish(&mut out).unwrap();

        let slabs = split_into_gops(&out);
        assert_eq!(slabs.len(), 1);
        let stego_decoded = decode_av1_to_yuv(&slabs[0]);
        let stego_psnr = compute_psnr_y(&yuv, &stego_decoded);
        const NATURAL_BASELINE_DB: f64 = 47.0;
        let damage = NATURAL_BASELINE_DB - stego_psnr;
        stego_damage.push(damage);

        eprintln!(
            "[D.5-control] GOP {i} (seek {seek_s:.1}s, isolated session): \
             stego Y-PSNR {:.2} dB, damage_vs_47 {:.3} dB",
            stego_psnr, damage
        );
    }

    let cliff = stego_damage
        .iter()
        .cloned()
        .fold(f64::MIN, f64::max)
        - stego_damage.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!(
        "[D.5-control] independent-session cliff = {cliff:.3} dB; \
         compare with multi-GOP cliff to distinguish content variance from cross-GOP leak"
    );
    // No assertion — diagnostic only. The main d5_v6_production_probe_3_gops
    // assertion is the ship gate.
}

#[test]
fn d5_v6_production_probe_3_gops() {
    // Deterministic crypto for reproducible cliff measurement (same
    // pattern as TG-2). Without it, every embed call randomizes
    // salt+nonce → different STC plan → different stego pixels.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260604");
    }

    let message = b"D.5 V6 production probe - multi-GOP stego visual fidelity";
    let passphrase = "d5-v6-probe-2026-06-04";
    let seeks = [0.5_f32, 1.0, 1.5];

    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: seeks.len() as u32,
    };

    // Encode via Av1StreamingEncodeSession (the production multi-GOP
    // path). Concatenated stego output gets split back into per-GOP
    // slabs via the OBU walker for per-frame PSNR measurement.
    let mut session = Av1StreamingEncodeSession::create(passphrase, message, params).unwrap();
    let mut concat_stego: Vec<u8> = Vec::new();
    let mut source_yuvs: Vec<Vec<u8>> = Vec::new();
    for &seek_s in &seeks {
        let yuv = extract_yuv420_frame(seek_s);
        source_yuvs.push(yuv.clone());
        session.push_frame(&yuv, &mut concat_stego).unwrap();
    }
    session.finish(&mut concat_stego).unwrap();

    let slabs = split_into_gops(&concat_stego);
    assert_eq!(
        slabs.len(),
        seeks.len(),
        "[D.5] OBU walker found {} slabs, expected {} (one per push_frame)",
        slabs.len(),
        seeks.len()
    );

    // Round-trip the message via the decode session, separate from
    // the per-frame measurement below — proves the OBU walker + STC
    // extract + chunk_frame assembly + decrypt chain works end-to-end.
    let mut dec = Av1StreamingDecodeSession::create(passphrase);
    dec.push_bytes(&concat_stego);
    let plaintext = dec.finish().unwrap();
    assert_eq!(plaintext, message, "[D.5] round-trip mismatch");

    // Per-GOP stego Y-PSNR. Each slab is a valid keyframe-only AV1
    // packet; decode via ffmpeg + IVF wrap.
    let mut stego_damage: Vec<f64> = Vec::with_capacity(seeks.len());
    for (i, slab) in slabs.iter().enumerate() {
        let stego_decoded = decode_av1_to_yuv(slab);
        let source = &source_yuvs[i];
        // Per-GOP stego PSNR vs source.
        let stego_psnr = compute_psnr_y(source, &stego_decoded);

        // For damage, we'd ideally compare stego vs natural; but
        // re-encoding the same frame naturally to compute that
        // doubles the test time. Use a known TG-2-style baseline:
        // natural-encode Y-PSNR for this fixture is ≥ 46 dB at q=30
        // (see corpus_validation baselines). damage = baseline -
        // stego_psnr is a conservative upper bound.
        const NATURAL_BASELINE_DB: f64 = 47.0;
        let damage = NATURAL_BASELINE_DB - stego_psnr;
        stego_damage.push(damage);

        eprintln!(
            "[D.5] GOP {i} (seek {:.1}s): slab {} bytes, stego Y-PSNR {:.2} dB, damage_vs_47 {:.3} dB",
            seeks[i], slab.len(), stego_psnr, damage
        );
    }

    let damage_max = stego_damage.iter().cloned().fold(f64::MIN, f64::max);
    let damage_min = stego_damage.iter().cloned().fold(f64::MAX, f64::min);
    let damage_cliff = damage_max - damage_min;

    assert!(
        damage_cliff <= D5_DAMAGE_CLIFF_DB,
        "[D.5] V6 multi-GOP cliff: stego_damage variance {:.3} dB exceeds threshold {} dB. \
         Per-GOP damage values: {:?}. V6 architecture should produce independent per-GOP \
         results; a cliff this wide indicates cross-GOP state leak in the session \
         (see `memory/h264_v5_production_finding_2026_05_07.md` for the H.264 V5 \
         precedent).",
        damage_cliff, D5_DAMAGE_CLIFF_DB, stego_damage
    );

    eprintln!(
        "[D.5] V6 multi-GOP cliff = {:.3} dB (threshold {} dB) ✓",
        damage_cliff, D5_DAMAGE_CLIFF_DB
    );
}
