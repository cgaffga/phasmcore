// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D2 residual instrument — how much primary capacity the shadows take.
//!
//! The per-GOP capacity the HUD shows and the planner plans against is
//! measured with no shadows in the cover. A shadow encode stamps every
//! shadow's bits into the cover first and fences those positions off from the
//! primary STC with ∞ cost, so the primary works with fewer finite positions
//! than the probe saw. D2 (2026-10-20) left that as a known residual: "both
//! the plan and the display can be optimistic by whatever the shadows
//! consume". This measures it on real 1080p clips, per GOP, from one clean
//! encode per GOP (`h264_shadow_loaded_gop_caps`), and reports what it does to
//! the primary ceiling the user is shown (`max_message_len`).
//!
//! The loads are the largest a user can actually get through today:
//!   * 1 shadow at the reported per-shadow cap,
//!   * 2 shadows at the reported cap (verified to embed by the D10 gate),
//!   * 4 shadows at 55% of the reported cap — N ≥ 3 over-promises (D19) and
//!     the D10 instrument found N = 4 embeds only well below its figure.
//! All at the 128-byte RS parity tier, the largest footprint a shadow takes.
//!
//! Prints, asserts nothing. Run:
//!   cargo test --release --features h264-encoder --test d2_shadow_loading \
//!     -- --ignored --nocapture --test-threads=1

#![cfg(feature = "h264-encoder")]

mod common;

use common::video_fixture::{ensure_yuv, probe_aligned_dims, session_guard, VideoFixture};
use phasm_core::codec::h264::openh264_stego::{
    h264_shadow_loaded_gop_caps, EncodeOpts, SliceYuvSource,
};
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingProbeSession, YuvFrameRef,
};
use phasm_core::h264_shadow_capacity_for_n;
use phasm_core::stego::balanced_allocation::max_message_len;
use phasm_core::stego::calibration::AllocationCalibration;
use phasm_core::stego::payload::compressed_payload_size;
use phasm_core::stego::shadow_layer::ShadowLayer;

const GOP: u32 = 30;
const N_FRAMES: u32 = 120;
const QP: i32 = 26;

fn lcg_text(len: usize, seed: u64) -> String {
    let mut st = seed ^ 0x9E37_79B9_7F4A_7C15;
    (0..len)
        .map(|_| {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (0x21u8 + ((st >> 33) % 0x5d) as u8) as char
        })
        .collect()
}

/// Longest `lcg_text` whose FRAMED payload (after Brotli) is ≤ `target`.
fn text_with_framed_size(target: usize, seed: u64) -> String {
    let fits = |l: usize| compressed_payload_size(&lcg_text(l, seed), &[]) <= target;
    let (mut lo, mut hi) = (1usize, target * 2 + 64);
    while lo < hi {
        let mid = (lo + hi).div_ceil(2);
        if fits(mid) { lo = mid } else { hi = mid - 1 }
    }
    let mut best = lo;
    for l in lo + 1..lo + 64 {
        if fits(l) {
            best = l;
        }
    }
    lcg_text(best, seed)
}

/// Returns (shadow pool bits, unloaded per-GOP caps) from the production probe.
fn probe(yuv: &[u8], w: u32, h: u32) -> (usize, Vec<usize>) {
    let mut p = StreamingProbeSession::create(EncodeSessionParams {
        width: w,
        height: h,
        fps_num: 30,
        fps_den: 1,
        qp: QP,
        gop_size: GOP,
        total_frames_hint: N_FRAMES,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    })
    .expect("probe session");
    let (ys, cs) = ((w * h) as usize, (w * h / 4) as usize);
    for f in 0..N_FRAMES as usize {
        let base = f * (ys + 2 * cs);
        p.push_frame(YuvFrameRef {
            y: &yuv[base..base + ys],
            y_stride: w as usize,
            u: &yuv[base + ys..base + ys + cs],
            u_stride: (w / 2) as usize,
            v: &yuv[base + ys + cs..base + ys + 2 * cs],
            v_stride: (w / 2) as usize,
        })
        .expect("probe push");
    }
    let (res, per_gop) = p.finish_with_per_gop().expect("probe finish");
    (res.shadow_pool_bits, per_gop)
}

fn ceiling(caps: &[usize]) -> usize {
    let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
    max_message_len(&samples, caps.len(), &AllocationCalibration::H264_1080P_QP26)
}

#[test]
#[ignore = "instrument: real 1080p probes + shadow sweeps, ~15-30 min"]
fn d2_residual_shadow_loading_on_real_clips() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let fixtures = [
        VideoFixture { name: "horse_flag", source_mp4: "Artlist_HorseFlag.mp4", long_side_cap: Some(1920), n_frames: N_FRAMES, qp: QP },
        VideoFixture { name: "asia_bottle", source_mp4: "Artlist_AsiaBottle.mp4", long_side_cap: Some(1920), n_frames: N_FRAMES, qp: QP },
        VideoFixture { name: "iphone7", source_mp4: "iphone7_1080p_30fps_h264_high.mov", long_side_cap: Some(1920), n_frames: N_FRAMES, qp: QP },
    ];
    let opts = EncodeOpts { qp: QP, intra_period: GOP as i32 };
    for fx in &fixtures {
        let (w, h) = probe_aligned_dims(fx);
        let yuv = ensure_yuv(fx, w, h);
        let (pool, unloaded_probe) = probe(&yuv, w, h);
        eprintln!(
            "\n=== {} {w}x{h} x{N_FRAMES}f  pool {pool} bits  probe caps {unloaded_probe:?}  \
             ceiling {} B",
            fx.name,
            ceiling(&unloaded_probe),
        );
        // (n shadows, fraction of the reported per-shadow cap)
        for (n, frac) in [(1usize, 1.00f64), (2, 1.00), (4, 0.55)] {
            let cap = h264_shadow_capacity_for_n(pool, n);
            let size = ((cap as f64) * frac) as usize;
            let msgs: Vec<String> =
                (0..n).map(|k| text_with_framed_size(size, 0xD2 + k as u64 * 104729)).collect();
            let passes: Vec<String> = (0..n).map(|k| format!("d2-shadow-{k}")).collect();
            let layers: Vec<ShadowLayer> = msgs
                .iter()
                .zip(&passes)
                .map(|(m, p)| ShadowLayer { message: m, passphrase: p, files: &[] })
                .collect();
            let mut src = SliceYuvSource::new(&yuv, w, h, N_FRAMES, GOP);
            let r = match h264_shadow_loaded_gop_caps(
                &mut src, w, h, N_FRAMES, opts, "d2 primary", "d2-primary", &layers,
                &CostWeights::default(), 128,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("  n={n} x {size} B: could not select ({e:?})");
                    continue;
                }
            };
            let su: usize = r.unloaded.iter().sum();
            let sl: usize = r.loaded.iter().sum();
            let worst = r
                .unloaded
                .iter()
                .zip(&r.loaded)
                .map(|(&u, &l)| if u == 0 { 0.0 } else { 100.0 * (u as f64 - l as f64) / u as f64 })
                .fold(0.0f64, f64::max);
            let taken: usize = r.finite_taken.iter().sum();
            let total: usize = r.finite_total.iter().sum();
            let (cu, cl) = (ceiling(&r.unloaded), ceiling(&r.loaded));
            eprintln!(
                "  n={n} x {size} B framed (tier {}): finite taken {taken}/{total} ({:.2}%) | \
                 Σcap {su} -> {sl} ({:.2}% lost, worst GOP {worst:.2}%) | \
                 shown ceiling {cu} -> {cl} B ({:.2}% lost)",
                r.tier_idx,
                100.0 * taken as f64 / total.max(1) as f64,
                100.0 * (su as f64 - sl as f64) / su.max(1) as f64,
                100.0 * (cu as f64 - cl as f64) / cu.max(1) as f64,
            );
            eprintln!("      per GOP unloaded {:?}\n      per GOP loaded   {:?}", r.unloaded, r.loaded);
        }
    }
}
