// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! F.8 — V6 multi-GOP production probe with shadows active.
//!
//! Combines D.5's V6 per-GOP-independence cliff measurement with
//! F.A's shadow integration. Each GOP encodes primary + N shadows;
//! per-GOP stego Y-PSNR is measured; the cliff (max - min damage
//! across GOPs) must stay under threshold. Round-trip on primary
//! AND each shadow's passphrase recovers the original messages.
//!
//! ## What this probe catches
//!
//! - **Per-GOP shadow density anomalies**. If shadows somehow added
//!   variable per-GOP damage (e.g. INF-cost overlay interacting with
//!   STC's content-adaptive cost weighting), the cliff would widen
//!   compared to a no-shadow baseline run at the SAME message size
//!   + fixture.
//! - **Shadow + V6 architectural compatibility**. F.A asserted
//!   per-GOP fresh state; F.8 measures the resulting visual fidelity
//!   stays inside a comparison envelope.
//! - **Multi-shadow visual cost**. F.8 quantifies how much extra
//!   stego damage shadows add per GOP. The empirical answer informs
//!   v0.7+ stealth optimization choices.
//!
//! ## Why a relative gate, not absolute
//!
//! D.5 measured a 0.97 dB cliff at its specific message length.
//! Initial F.8 attempt used a 3.0 dB absolute threshold, but the
//! V6 cliff turns out to be CONTENT-SENSITIVE: the no-shadow
//! baseline at F.8's (different) message length measured ~7 dB —
//! not a regression, just different STC position selection picking
//! different per-GOP positions. The right F.8 assertion is
//! shadow_cliff ≤ baseline_cliff + RELATIVE_MARGIN, comparing the
//! same fixture + same primary message length with/without shadows.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Stdio};

use phasm_core::codec::av1::stego::session::{
    Av1ShadowSpec, Av1StreamingDecodeSession, Av1StreamingEncodeParams,
    Av1StreamingEncodeSession,
};
use phasm_core::stego::payload;

/// Local OBU walker — same as D.5's `split_into_gops`. Splits a
/// concatenated AV1 OBU byte stream at every `OBU_SEQUENCE_HEADER`
/// (type=1) boundary into per-GOP slabs.
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

const W: u32 = 144;
const H: u32 = 256;
const Q: usize = 30;
const SOURCE: &str = "Artlist_CarPlane.mp4";

/// Maximum additional cliff widening F.8 tolerates from adding
/// shadows. Both the baseline (no-shadows) and the with-shadows
/// runs are measured at the SAME fixture + same primary message
/// length, so the cliff difference isolates shadow's contribution.
///
/// 2.0 dB allows for STC re-allocation noise when shadows displace
/// primary's flips into different cover positions, while still
/// catching a true shadow-driven blow-up (which would manifest as
/// e.g. 10+ dB widening).
const F8_SHADOW_CLIFF_MARGIN_DB: f64 = 2.0;

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv(seek_s: f32) -> Vec<u8> {
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

fn build_ivf(obus: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(32 + 12 + obus.len());
    out.extend_from_slice(b"DKIF");
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"AV01");
    out.extend_from_slice(&(W as u16).to_le_bytes());
    out.extend_from_slice(&(H as u16).to_le_bytes());
    out.extend_from_slice(&30u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&0u32.to_le_bytes());
    out.extend_from_slice(&(obus.len() as u32).to_le_bytes());
    out.extend_from_slice(&0u64.to_le_bytes());
    out.extend_from_slice(obus);
    out
}

fn decode_av1(av1_bytes: &[u8]) -> Vec<u8> {
    let ivf = build_ivf(av1_bytes);
    let mut child = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i", "pipe:0"])
        .args([
            "-frames:v", "1", "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
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
        .expect("write ivf");
    let out = child.wait_with_output().expect("ffmpeg wait");
    assert!(out.status.success(), "ffmpeg decode failed: {:?}", out);
    out.stdout
}

fn psnr_y(source: &[u8], reconstructed: &[u8]) -> f64 {
    let y_size = (W * H) as usize;
    let mut sum_sq: u64 = 0;
    for i in 0..y_size {
        let d = source[i] as i32 - reconstructed[i] as i32;
        sum_sq += (d * d) as u64;
    }
    let mse = sum_sq as f64 / y_size as f64;
    if mse < 0.001 {
        return 100.0;
    }
    10.0 * (65025.0 / mse).log10()
}

/// Helper: encode a 3-GOP session (optionally with shadows) on the
/// carplane fixture, measure per-GOP stego Y-PSNR, return per-GOP
/// damage values + total stego byte count.
fn measure_v6_cliff(
    primary_msg: &[u8],
    primary_pass: &str,
    shadows: Vec<Av1ShadowSpec>,
    parity_len: usize,
) -> (Vec<f64>, usize, Vec<Vec<u8>>) {
    let seeks = [0.5_f32, 1.0, 1.5];
    let params = Av1StreamingEncodeParams {
        width: W,
        height: H,
        quantizer: Q,
        gop_size: 1,
        total_frames_hint: seeks.len() as u32,
    };
    let mut enc = Av1StreamingEncodeSession::create_with_shadows(
        primary_pass,
        primary_msg,
        params,
        shadows,
        parity_len,
    )
    .expect("create_with_shadows");
    let mut concat_stego: Vec<u8> = Vec::new();
    let mut source_yuvs: Vec<Vec<u8>> = Vec::new();
    for &seek_s in &seeks {
        let yuv = extract_yuv(seek_s);
        source_yuvs.push(yuv.clone());
        enc.push_frame(&yuv, &mut concat_stego).expect("push_frame");
    }
    enc.finish(&mut concat_stego).expect("finish");

    let slabs = split_into_gops(&concat_stego);
    assert_eq!(slabs.len(), seeks.len());

    const NATURAL_BASELINE_DB: f64 = 47.0;
    let mut damage = Vec::with_capacity(slabs.len());
    for (i, slab) in slabs.iter().enumerate() {
        let stego_decoded = decode_av1(slab);
        let stego_psnr = psnr_y(&source_yuvs[i], &stego_decoded);
        damage.push(NATURAL_BASELINE_DB - stego_psnr);
        eprintln!(
            "  GOP {i} (seek {:.1}s): slab {} bytes, stego Y-PSNR {:.2} dB, damage {:.3} dB",
            seeks[i], slab.len(), stego_psnr, damage[i]
        );
    }
    let cliff = damage.iter().cloned().fold(f64::MIN, f64::max)
        - damage.iter().cloned().fold(f64::MAX, f64::min);
    eprintln!("  cliff = {cliff:.3} dB");
    (damage, concat_stego.len(), slabs)
}

/// F.8 — head-to-head cliff comparison: same fixture, same primary
/// message length, ±shadows. The shadow's incremental cliff effect
/// must stay below `F8_SHADOW_CLIFF_MARGIN_DB`.
#[test]
fn f8_v6_shadows_dont_widen_cliff_beyond_margin() {
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260605");
    }
    // Same length so STC's choice of n_used + w matches between
    // baseline and with-shadow runs — isolates shadow's effect.
    let primary_msg = b"F.8 production probe primary msg padded to len";
    let primary_pass = "f8-pass";

    eprintln!("[F.8-baseline] no shadows:");
    let (baseline_damage, baseline_stego_bytes, _) =
        measure_v6_cliff(primary_msg, primary_pass, Vec::new(), 16);
    let baseline_cliff = baseline_damage.iter().cloned().fold(f64::MIN, f64::max)
        - baseline_damage.iter().cloned().fold(f64::MAX, f64::min);

    let shadow_text = "F.8 shadow content for cliff-delta probe";
    let shadow_payload = payload::encode_payload(shadow_text, &[]).unwrap();
    let shadows = vec![Av1ShadowSpec {
        passphrase: "f8-shadow".into(),
        message: shadow_payload,
    }];

    eprintln!("[F.8-with-shadow] 1 shadow:");
    let (shadow_damage, shadow_stego_bytes, slabs) =
        measure_v6_cliff(primary_msg, primary_pass, shadows, 16);
    let shadow_cliff = shadow_damage.iter().cloned().fold(f64::MIN, f64::max)
        - shadow_damage.iter().cloned().fold(f64::MAX, f64::min);

    eprintln!(
        "[F.8] baseline cliff {:.3} dB, with-shadow cliff {:.3} dB, delta {:+.3} dB (margin {} dB)",
        baseline_cliff,
        shadow_cliff,
        shadow_cliff - baseline_cliff,
        F8_SHADOW_CLIFF_MARGIN_DB
    );
    eprintln!(
        "[F.8] stego bytes: baseline {baseline_stego_bytes}, with-shadow {shadow_stego_bytes}"
    );

    // Round-trip primary + shadow via the session decode (smoke-tests
    // the F.A pipeline end-to-end alongside the cliff measurement).
    let concat: Vec<u8> = slabs.iter().flat_map(|s| s.iter().copied()).collect();
    let mut dec = Av1StreamingDecodeSession::create(primary_pass);
    dec.push_bytes(&concat);
    let recovered_shadow = dec
        .finish_shadow_first_match("f8-shadow")
        .expect("[F.8] shadow round-trip");
    let recovered_primary = dec.finish().expect("[F.8] primary round-trip");
    assert_eq!(recovered_primary.as_slice(), primary_msg);
    assert_eq!(recovered_shadow.text, shadow_text);

    assert!(
        shadow_cliff <= baseline_cliff + F8_SHADOW_CLIFF_MARGIN_DB,
        "[F.8] shadow widens cliff by {:+.3} dB (baseline {:.3} → with-shadow {:.3}); margin \
         is {} dB. A wider gap means shadow's INF-cost overlay or LSB writes are causing \
         primary STC to make systematically worse per-GOP decisions. Per-GOP damages: \
         baseline={:?}, with-shadow={:?}.",
        shadow_cliff - baseline_cliff,
        baseline_cliff,
        shadow_cliff,
        F8_SHADOW_CLIFF_MARGIN_DB,
        baseline_damage,
        shadow_damage
    );
}
