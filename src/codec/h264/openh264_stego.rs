// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Production stego ENCODE orchestrator on top of the OpenH264 fork —
// the sole H.264 encoder (`h264-encoder` feature). The live primitive
// is `h264_encode_gop_framed_bits`: a 2-pass
// per-GOP, single combined-cover STC over all four bypass-bin domains
// (CoeffSign, CoeffSuffixLsb, MvdSign, MvdSuffixLsb) in canonical
// CS→CSL→MVDs→MVDsl order, driven by the streaming session
// (`streaming_session.rs`). Passphrase-derived hhat seeds; per-position
// Tier-3 content-adaptive J-UNIWARD costs; cascade-safety + Track-1
// tier ∞-cost gates. There is NO decode entry in this file — decode is
// the engine-agnostic walker (`StreamingDecodeSession`).
//
// **Cascade elimination**: Pass 2 re-encodes with the planned bits
// applied via the OH264 fork's `wire_only` scratch-table overrides at
// the CABAC bypass-bin emit site (default ON; `PHASM_USE_WIRE_ONLY=0`
// opts into the legacy mutating + dual-recon path). In wire-only mode
// the encoder's reconstruction (pDecPic) stays clean by construction —
// no residual / MV mutation — so there is no inter-frame cascade. A
// #533 Pass-1→Pass-2 mode-decision DecisionCache (Capture/Replay) keeps
// the two passes byte-identical.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// #549 Phase C diagnostic — thread-local pass index, set by
// `h264_encode_gop_framed_bits_auto` before each `encode_once`
// call. Read by the `md_cost` callback in encode_once when
// `PHASM_549_MD_DUMP=1` is set, to tag dumped MB-decision records as
// Pass 1 vs Pass 2 so a downstream diff can find the first divergent MB.
thread_local! {
    static PHASM_549_PASS_IDX: Cell<u32> = const { Cell::new(0) };
    /// P3.3b — per-MB coefficient capture buffer.
    static COEFF_CAPTURE: RefCell<Option<Vec<(u32, u16, u16, [i16; 256])>>> =
        const { RefCell::new(None) };
    /// D2.2 — per-row DPB correction state. Stored as raw pointers
    /// because the callback fires synchronously during encode_frame
    /// (encoder paused between rows). Valid only while encode_frame
    /// is in progress.
    static ROW_DPB_STATE: RefCell<Option<RowDpbState>> = const { RefCell::new(None) };
}

struct RowDpbState {
    encoder_handle: *mut core_openh264_sys::PhasmEncoderHandle,
    enc_height: u32,
    width: u32,
    height: u32,
    override_map: *const std::collections::HashMap<u64, u8>,
    cover: *const super::stego::inject::DomainCover,
    frame_idx: u32,
    qp: i32,
}

unsafe impl Send for RowDpbState {}

extern "C" fn row_complete_callback(
    _frame_num: u32, row_y: u16,
    _bs_byte_pos: i32, _bs_bits_left: i32,
) {
    ROW_DPB_STATE.with(|cell| {
        let borrow = cell.borrow();
        let Some(ref state) = *borrow else { return };
        let mut y_ptr: *mut u8 = core::ptr::null_mut();
        let mut y_stride: i32 = 0;
        let rv = unsafe {
            core_openh264_sys::phasm_encoder_get_dec_pic_y(
                state.encoder_handle, &mut y_ptr, &mut y_stride,
            )
        };
        if rv != 0 || y_ptr.is_null() || y_stride <= 0 {
            return;
        }
        let total = y_stride as usize * state.enc_height as usize;
        let dec_pic_y = unsafe { std::slice::from_raw_parts_mut(y_ptr, total) };
        let override_map = unsafe { &*state.override_map };
        let cover = unsafe { &*state.cover };
        super::stego::dpb_correction::compute_and_apply_deltas_for_row(
            dec_pic_y,
            y_stride as usize,
            state.width,
            state.height,
            override_map,
            cover,
            state.frame_idx,
            row_y as u32,
            state.qp,
        );
    });
}

pub(crate) fn set_549_pass_idx(idx: u32) {
    PHASM_549_PASS_IDX.with(|c| c.set(idx));
}

fn get_549_pass_idx() -> u32 {
    PHASM_549_PASS_IDX.with(|c| c.get())
}

/// P3.3b — C callback for the post-quant hook in the OH264 fork.
/// Captures the luma coefficient array (256 i16) per MB into the
/// thread-local COEFF_CAPTURE buffer. Only captures when the buffer
/// is Some (enabled via `enable_coeff_capture`).
extern "C" fn post_quant_callback(
    frame_num: u32, mb_x: u16, mb_y: u16,
    coeffs: *const i16, coeff_count: i32,
    _cbp_luma: u8, _cbp_chroma: u8, _qp: i32,
) {
    if coeffs.is_null() || coeff_count < 256 { return; }
    COEFF_CAPTURE.with(|buf| {
        if let Some(ref mut v) = *buf.borrow_mut() {
            let slice = unsafe { std::slice::from_raw_parts(coeffs, 256) };
            let mut arr = [0i16; 256];
            arr.copy_from_slice(slice);
            if std::env::var("P33B_TRACE").as_deref() == Ok("1") {
                let nz = arr.iter().filter(|&&c| c != 0).count();
                eprintln!("[CAP] f={frame_num} mb=({mb_x},{mb_y}) nz={nz} first4=[{},{},{},{}]",
                    arr[0], arr[1], arr[2], arr[3]);
            }
            v.push((frame_num, mb_x, mb_y, arr));
        }
    });
}

fn enable_coeff_capture() {
    COEFF_CAPTURE.with(|b| *b.borrow_mut() = Some(Vec::new()));
    unsafe { core_openh264_sys::phasm_set_post_quant_callback(Some(post_quant_callback)); }
}

fn take_coeff_capture() -> Vec<(u32, u16, u16, [i16; 256])> {
    // B-full.6b.3 (#894): do NOT unregister the global post-quant callback here
    // (the old `disable_coeff_capture()` call is gone). The COEFF_CAPTURE buffer is
    // thread_local (per-worker, safe), but `phasm_set_post_quant_callback(None)` is a
    // PROCESS global — unregistering it would truncate an OVERLAPPING producer's
    // capture mid-encode. Leaving the callback registered is correct: it fires
    // per-thread, writes the calling thread's buffer, and no-ops when that buffer is
    // None (so Pass-2 / plain / consumer encodes — capture off — stay byte-identical).
    // `enable_coeff_capture`'s registration is idempotent (same fn ptr).
    COEFF_CAPTURE.with(|b| b.borrow_mut().take().unwrap_or_default())
}

use core_openh264_sys::{
    encoder_pos_to_phasm_position_key, PHASM_MB_TYPE_OTHER, ZZ_SCAN_4X4,
};

use super::cabac::bin_decoder::slice::{
    walk_annex_b_for_cover_with_options, CoverWalkOutput, WalkOptions,
};
use super::stego::cost_weights::{
    combine_bits_4domain, combine_cover_4domain, split_plan_4domain, CostWeights,
};
use super::stego::orchestrate::DomainCosts;
use super::openh264::{
    set_frame_num, set_pass_mode, Encoder, EncoderError, MbDecision, PassMode, StegoHandlers,
    StegoSession,
};
use super::pass2_cache::DecisionCache;
use super::stego::hook::{BinKind, EmbedDomain, SyntaxPath};
use super::stego::inject::{remap_cover_frame_idx, DomainCover};
use crate::stego::error::StegoError;
use crate::stego::stc::embed::stc_embed;
use crate::stego::stc::extract::stc_extract;
use crate::stego::stc::hhat::generate_hhat;

/// STC constraint length. Must match between encode and decode (decoder
/// brute-forces `m_total` but treats `STC_H` as fixed).
const STC_H: usize = 4;

/// Cheap env-var check for `PHASM_PERF_TRACE=1`. When set, the 4-domain
/// encode path prints per-phase wall-clock timing to stderr. Off by
/// default — single `env::var` call per encode (negligible cost).
#[inline]
fn perf_trace() -> bool {
    std::env::var("PHASM_PERF_TRACE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Phase-boundary current-RSS tap for the #847 memory audit. When
/// `PHASM_MEM_TRACE=1`, prints the process resident set (MB) at a named
/// point in the per-GOP encode so the ~0.7-1.1 GB peak can be localised to
/// a specific phase (Pass1 / walk / costs / combine / STC / Pass2). Shells
/// out to `ps -o rss=` — pure std, no FFI, ~8 calls/GOP so it doesn't
/// perturb the footprint. Off by default (one `env::var` check).
fn mem_trace(label: &str) {
    if std::env::var("PHASM_MEM_TRACE").as_deref() != Ok("1") {
        return;
    }
    let pid = std::process::id();
    let rss_mb = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|kb| kb as f64 / 1024.0)
        .unwrap_or(0.0);
    eprintln!("[PHASM_MEM_TRACE] {label:<26} {rss_mb:>8.0} MB");
}

/// Encoder configuration knobs for the production stego path. The
/// `qp` / `intra_period` defaults track the OpenH264 fork's recommended
/// values for visual quality + reasonable IDR cadence.
#[derive(Debug, Clone, Copy)]
pub struct EncodeOpts {
    /// Quantization parameter (initial). Lower = higher quality + larger
    /// bitstream + more cover bits. Range 0..=51; typical 22..32.
    pub qp: i32,
    /// IDR period in frames. The encoder emits IDR at every N-th frame
    /// boundary. 60 = once per second at 60 fps.
    pub intra_period: i32,
}

impl Default for EncodeOpts {
    fn default() -> Self {
        Self {
            qp: 26,
            intra_period: 60,
        }
    }
}

/// #493.4 Phase 3 — OH264 4-domain per-GOP encode primitive.
///
/// Embeds `frame_bits` across all 4 stego domains (CoeffSign,
/// CoeffSuffix, MvdSign, MvdSuffix) via a single combined-cover STC
/// plan. The override map is keyed by canonical `PositionKey::raw()`
/// across all 4 domains; the OH264 fork's per-domain emit hooks look
/// up by the same key and apply the planned bit.
///
/// Phase 0 walker-symmetry parity gates (#493.1) verified that the
/// canonical keys match between OH264 fork emit + Rust walker for
/// all 4 (engine, domain) pairs, so the override-map lookup works
/// uniformly.
///
/// Cost vector (STEGO.A.3): per-position Tier-3 content-adaptive
/// J-UNIWARD wavelet costs from `content_costs::compute_content_costs_yuv`
/// feed the combined cover, multiplied by the per-domain `CostWeights`,
/// so STC concentrates flips in textured / high-detectability-headroom
/// regions. (`PHASM_DIAG_UNIFORM_COSTS=1` forces uniform 1.0 for the
/// cascade-leak diagnostic.) Costs are transparent to the decoder —
/// STC's syndrome equation depends only on which cover bits the encoder
/// ultimately wrote.
///
/// This 4-domain combined-cover STC primitive is the live production
/// encode path: the streaming session drives it per GOP via
/// [`h264_encode_gop_framed_bits`].
pub fn h264_encode_gop_framed_bits_auto(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    frame_bits: &[u8],
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
) -> Result<Vec<u8>, StegoError> {
    h264_encode_gop_framed_bits(
        yuv, width, height, n_frames, opts, frame_bits, hhat_seed, weights,
        super::stego::tier_filter::CascadeTier::Auto,
        super::stego::tier_filter::DEFAULT_HEADROOM,
    )
    .map(|(bytes, _tier)| bytes)
}

/// D'.3-OH264 (#795) — explicit cascade-tier variant of
/// [`h264_encode_gop_framed_bits_auto`]. Mirrors the pure-Rust
/// `h264_stego_encode_yuv_4domain_scheme_a_with_tier` (encode_pixels.rs).
///
/// Tier filter sets ∞-cost on CSB/CSL positions whose estimated pixel
/// impact exceeds the tier threshold. STC steers around them. Wire
/// format is unchanged — decoder is tier-agnostic.
#[allow(clippy::too_many_arguments)]
pub fn h264_encode_gop_framed_bits(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    frame_bits: &[u8],
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
    cascade_tier: super::stego::tier_filter::CascadeTier,
    headroom: f32,
) -> Result<(Vec<u8>, super::stego::tier_filter::CascadeTier), StegoError> {
    // B-full.6b.1 (#894) — the per-GOP encode is split into a payload-INDEPENDENT
    // producer (clean cover + costs) and a payload-DEPENDENT consumer (STC + emit).
    // This wrapper preserves the original one-call contract; the streaming session
    // (`embed_gop_roundtrip_safe`) calls `produce_gop_cover` ONCE then loops
    // `consume_gop_emit` (shrink-carry) so the clean encode is not repeated per
    // shrink — and N producers can run in parallel (4b, B-full per-instance state).
    let products = produce_gop_cover(yuv, width, height, n_frames, opts)?;
    consume_gop_emit(
        &products, yuv, width, height, n_frames, opts, frame_bits, hhat_seed,
        weights, cascade_tier, headroom,
    )
}

/// Payload-INDEPENDENT products of the Pass-1 cover capture for one GOP —
/// produced by [`produce_gop_cover`], consumed by [`consume_gop_emit`]. The
/// split is the 4b (#894) parallelism seam: these products depend only on the
/// GOP YUV + dims + opts, so N producers run concurrently (each its own encoder
/// instance, B-full per-instance stego state) while the sequential consumer
/// carries the cross-GOP STC cursor.
#[cfg(feature = "h264-encoder")]
pub(crate) struct GopCoverProducts {
    /// Pass-1 clean bitstream (retained for the PHASM_PERF_TRACE symmetry probe).
    baseline_bitstream: Vec<u8>,
    /// 4-domain cover from the Pass-1 walk (record_mvd: true).
    baseline_walk: CoverWalkOutput,
    /// J-UNIWARD content costs with the cascade-safety MSL ∞-gate ALREADY applied.
    /// (The tier filter is payload-dependent ⇒ applied in the consumer.)
    content_costs: DomainCosts,
    /// Captured coefficients for Pass-2 replay (P3.3b.4).
    captured_coeffs: Vec<(u32, u16, u16, [i16; 256])>,
    /// Pass-1 mode-decision cache for Pass-2 replay (wire_only); None off-path.
    decision_cache: Option<Arc<Mutex<DecisionCache>>>,
    /// Resolved wire_only flag — the consumer must use the SAME value so Pass-1
    /// and Pass-2 take symmetric code paths through the fork.
    wire_only: bool,
}

/// 4b producer (#894) — the payload-INDEPENDENT half of the per-GOP encode:
/// Pass-1 clean+capture, the 4-domain walk, J-UNIWARD content costs, and the
/// cascade-safety MSL ∞-gate. Byte-identical substitute for the front half of
/// the old monolithic `h264_encode_gop_framed_bits` (gated by oh264_g4).
#[cfg(feature = "h264-encoder")]
pub(crate) fn produce_gop_cover(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<GopCoverProducts, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;

    // #548 v1.0 BLOCKER fix (2026-05-18) — reset the libencoder-side fork
    // statics before every encode session. On the production fork the per-MB
    // bypass scratch + last-MB sentinels + wire-only flag are PROCESS-GLOBAL, so
    // two sequential encode calls otherwise share fork state across the
    // encoder-instance teardown: Call 2 produces 541 ChromaAc CS Sign diffs vs
    // Call 1's 0 on the same YUV, breaking REPLAY byte-identity. Reproducer:
    // `oh264_wire_only_two_sequential_calls_bisect` in
    // `core/tests/oh264_streaming_530_repro.rs`.
    unsafe { core_openh264_sys::phasm_reset_encoder_session_state() };

    let trace = perf_trace();
    mem_trace("00 enter");

    let mb_width = width / 16;
    let mb_height = height / 16;
    let mb_per_frame = (mb_width * mb_height) as usize;
    let _ = mb_height;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // #533.4.8: Pass-2 replay cache. Pass 1 streams per-MB mode
    // decisions in; Pass 2 reads them back so the encoder takes
    // bit-identical mode-decision paths across the two encodes. This
    // closes the inter-pass OH264 non-determinism (~200-byte first_diff
    // observed between two clean encodes of the same YUV) that landed
    // the last residual walker-vs-plan diff on the streaming fixture.
    // Gated on wire_only — the C.8 dual_recon path has its own drift
    // handling, and capture/replay adds cost we only need under
    // wire_only's mode-decision-sensitive scratch override.
    //
    // 2026-05-18: default flipped ON after Phase 4 closure (#538). The
    // wire_only path is now the production v1.0 ship default —
    // single-GOP + multi-GOP green at 1072×1920 × 30f (#530 + #548
    // closed via fork commits bb7f91fa + 1b1205ad). Set
    // `PHASM_USE_WIRE_ONLY=0` to opt back into the legacy mutating
    // + C.8 dual_recon path during the #539 deletion-window
    // transition (the legacy path will be removed in #539).
    let wire_only = std::env::var("PHASM_USE_WIRE_ONLY")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true);
    let decision_cache: Option<Arc<Mutex<DecisionCache>>> = if wire_only {
        Some(Arc::new(Mutex::new(DecisionCache::with_capacity(
            mb_per_frame * n_frames as usize,
        ))))
    } else {
        None
    };

    // Pass 1 — clean encode + walker for 4-domain cover capture.
    // record_mvd: true (Phase 0 finding: walker default leaves MVD
    // cover empty otherwise).
    let t_pass1 = if trace { Some(std::time::Instant::now()) } else { None };
    set_549_pass_idx(1);
    /* #549 v1.0 fix (2026-05-19): set wire_only_global before Pass 1 too.
     * The fork's `apply_coeff_hooks_to_level` selects mutation vs scratch
     * path based on `g_phasm_use_wire_only_overrides`. If Pass 1 runs
     * with flag=0 (mutation path) and Pass 2 runs with flag=1 (scratch
     * path), the two passes take structurally different code paths
     * through the encoder even with empty override_map. That asymmetry
     * — not the override application itself — drives the cascade.
     * Closed-loop walker test `pass2_walker_sees_N_cs_flips_real_carplane_480p`
     * has symmetric wire_only across passes and shows zero positional
     * drift; orchestrator without this line has +95 CS drift. */
    if wire_only {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
    }
    // P3.3b: enable coefficient capture during Pass 1 so we can
    // replay stored coefficients (with STC flips) in Pass 3.
    enable_coeff_capture();
    let baseline_bitstream = encode_once(
        yuv, width, height, n_frames, opts,
        override_map.clone(),
        mb_type_table.clone(),
        applied.clone(),
        mb_width,
        mb_per_frame,
        /* dual_recon = */ false,
        if wire_only { PassMode::Capture } else { PassMode::Passthrough },
        decision_cache.clone(),
        None, // P3.3a: no DPB correction on Pass 1
        None, // P3.3b.4: no coefficient replay on Pass 1 (capture only)
        0, // frame_idx_base: whole-clip / per-GOP no-shadow path (no offset)
    )?;
    let captured_coeffs = take_coeff_capture();
    let dt_pass1 = t_pass1.map(|t| t.elapsed());
    let pass1_bytes = baseline_bitstream.len();
    mem_trace("01 after Pass1+capture");

    let t_walk = if trace { Some(std::time::Instant::now()) } else { None };
    let baseline_walk = walk_annex_b_for_cover_with_options(
        &baseline_bitstream,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("baseline walk: {e}")))?;
    let dt_walk = t_walk.map(|t| t.elapsed());
    mem_trace("02 after walk (cover)");

    // STEGO.A.3 — Tier 3 content-adaptive costs. Computes per-position
    // J-UNIWARD wavelet costs (luma + chroma planes via
    // `content_costs::compute_content_costs_yuv`) and feeds them into
    // the combined cover. Transparent to the decoder — STC's syndrome
    // equation doesn't depend on costs, only on which cover bits the
    // encoder ultimately wrote. Content-adaptive costs let STC
    // concentrate flips in textured / high-detectability-headroom
    // regions, lowering steganalysis detection rates.
    //
    // PHASM_DIAG_UNIFORM_COSTS=1 forces uniform 1.0 costs for the
    // STEGO.A.10 cascade-leak diagnostic — isolates whether the
    // leak is Tier-3-induced or independent.
    let t_cc = if trace { Some(std::time::Instant::now()) } else { None };
    let use_uniform_costs = std::env::var("PHASM_DIAG_UNIFORM_COSTS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut content_costs = if use_uniform_costs {
        DomainCosts::default()
    } else {
        super::stego::content_costs::compute_content_costs_yuv(
            yuv, width, height, n_frames, &baseline_walk.cover, opts.qp,
        )?
    };
    if let Some(t) = t_cc {
        eprintln!(
            "[PHASM_PERF_TRACE]   ├─ content_costs(J-UNIWARD) {:>8.1} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
    mem_trace("03 after content_costs");

    // STEGO.A.10 fix — apply cascade-safety MSL gate as ∞-cost in
    // STC's input (not as override-map filter). Unsafe MvdSuffixLsb
    // positions cascade through the median MV predictor, so flipping
    // them changes downstream cover bits that the decoder will see.
    // STC must never pick them. Setting their cost to f32::INFINITY
    // makes the syndrome solver avoid them by construction.
    //
    // Critical for Tier 3 cost mode: content-adaptive costs assign
    // LOW cost to motion-heavy MBs (textured = hard to detect), and
    // motion-heavy MBs are exactly where MSL cascade risk concentrates.
    // Without this gate, Tier 3 steers STC into cascade-leaky territory
    // (empirically: school_fight CRC mismatch at 480×272×8).
    let safe_msb = super::stego::cascade_safety::analyze_safe_mvd_subset(
        &baseline_walk.mvd_meta, baseline_walk.mb_w, baseline_walk.mb_h,
    );
    let safe_msl = super::stego::cascade_safety::derive_msl_safe_from_msb(
        &baseline_walk.cover.mvd_sign_bypass.positions,
        &safe_msb,
        &baseline_walk.cover.mvd_suffix_lsb.positions,
    );
    let n_msl = content_costs.mvd_suffix_lsb.len();
    let safe_msl_len = safe_msl.len().min(n_msl);
    for i in 0..safe_msl_len {
        if !safe_msl[i] {
            content_costs.mvd_suffix_lsb[i] = f32::INFINITY;
        }
    }
    // Positions past safe_msl_len (shouldn't happen, defensive): mark INF.
    for i in safe_msl_len..n_msl {
        content_costs.mvd_suffix_lsb[i] = f32::INFINITY;
    }

    if trace {
        let ms = |d: Option<std::time::Duration>| {
            d.map(|x| x.as_secs_f64() * 1000.0).unwrap_or(0.0)
        };
        eprintln!(
            "[PHASM_PERF_TRACE]   produce: pass1 {:>8.1} ms  walk {:>8.1} ms  pass1_bs={} bytes",
            ms(dt_pass1), ms(dt_walk), pass1_bytes,
        );
    }

    Ok(GopCoverProducts {
        baseline_bitstream,
        baseline_walk,
        content_costs,
        captured_coeffs,
        decision_cache,
        wire_only,
    })
}

/// 4b consumer (#894) — the payload-DEPENDENT half: tier resolution, combine,
/// the STC plan over the message bits, the override-map build, the Pass-2
/// wire-only stego emit, and the diagnostics. The cross-GOP cursor-carry lives
/// in the caller (`embed_gop_roundtrip_safe`), which loops this with shrinking
/// `frame_bits` while reusing ONE [`GopCoverProducts`] (so the clean Pass-1 is
/// not re-run per shrink). Byte-identical substitute for the back half of the
/// old monolithic `h264_encode_gop_framed_bits` (gated by oh264_g4).
#[cfg(feature = "h264-encoder")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn consume_gop_emit(
    products: &GopCoverProducts,
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    frame_bits: &[u8],
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
    cascade_tier: super::stego::tier_filter::CascadeTier,
    headroom: f32,
) -> Result<(Vec<u8>, super::stego::tier_filter::CascadeTier), StegoError> {
    let m_total = frame_bits.len();
    if m_total == 0 {
        return Err(StegoError::InvalidVideo("empty frame bits".into()));
    }

    let trace = perf_trace();
    let t_start = if trace { Some(std::time::Instant::now()) } else { None };

    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // Reuse the producer's payload-independent cover. `content_costs` is cloned
    // because the tier filter below mutates it in place and one `GopCoverProducts`
    // is shared across shrink iterations (and, later, the parallel producers).
    let baseline_bitstream = &products.baseline_bitstream;
    let baseline_walk = &products.baseline_walk;
    let mut content_costs = products.content_costs.clone();
    let captured_coeffs = &products.captured_coeffs;
    let decision_cache = products.decision_cache.clone();
    let wire_only = products.wire_only;

    let t_stc = if trace { Some(std::time::Instant::now()) } else { None };
    let t_after_costs = if trace { Some(std::time::Instant::now()) } else { None };

    // D'.3-OH264 (#795) — Track 1 cascade-safety tier filter. Mirrors
    // the pure-Rust integration in encode_pixels.rs:975-1010. Resolves
    // `Auto` to the highest tier whose capacity ≥ msg_bytes × headroom,
    // then sets ∞-cost on CSB/CSL positions failing the tier predicate.
    // STC steers around them; wire format unchanged; decoder is
    // tier-agnostic (same as the safe_msl gate above).
    //
    // PHASM_TIER_OVERRIDE / PHASM_TIER_DEBUG env vars match the
    // pure-Rust path for cross-encoder bisects.
    let resolved_tier = {
        use super::stego::tier_filter::{
            apply_tier_filter, auto_select_tier, CascadeTier,
        };
        let qp_value = opts.qp;
        let msg_bytes = frame_bits.len() / 8;
        let csb_qp_slice: Vec<i32> = vec![qp_value; baseline_walk.cover.coeff_sign_bypass.len()];
        let csl_qp_slice: Vec<i32> = vec![qp_value; baseline_walk.cover.coeff_suffix_lsb.len()];
        let resolved_tier = match std::env::var("PHASM_TIER_OVERRIDE").ok().as_deref() {
            Some(s) if s != "auto" => s.parse::<u8>().ok()
                .and_then(CascadeTier::from_u8)
                .unwrap_or(CascadeTier::Tier0),
            _ => match cascade_tier {
                CascadeTier::Auto => auto_select_tier(
                    &baseline_walk.cover, &csb_qp_slice, &csl_qp_slice,
                    msg_bytes, headroom,
                ),
                explicit => explicit,
            },
        };
        if std::env::var("PHASM_TIER_DEBUG").is_ok() {
            eprintln!(
                "[tier_filter/oh264] msg_bytes={msg_bytes} resolved_tier={} csb={} csl={}",
                resolved_tier.as_u8(),
                baseline_walk.cover.coeff_sign_bypass.len(),
                baseline_walk.cover.coeff_suffix_lsb.len(),
            );
        }
        let tier_idx = resolved_tier.as_u8();
        if tier_idx > 0 {
            let csb_keep = apply_tier_filter(
                &baseline_walk.cover.coeff_sign_bypass, &csb_qp_slice, tier_idx,
            );
            let csl_keep = apply_tier_filter(
                &baseline_walk.cover.coeff_suffix_lsb, &csl_qp_slice, tier_idx,
            );
            for (i, &keep) in csb_keep.iter().enumerate() {
                if !keep && i < content_costs.coeff_sign_bypass.len() {
                    content_costs.coeff_sign_bypass[i] = f32::INFINITY;
                }
            }
            for (i, &keep) in csl_keep.iter().enumerate() {
                if !keep && i < content_costs.coeff_suffix_lsb.len() {
                    content_costs.coeff_suffix_lsb[i] = f32::INFINITY;
                }
            }
        }
        resolved_tier
    };

    let (combined_cover, combined_costs, boundaries) =
        combine_cover_4domain(&baseline_walk.cover, &content_costs, weights);
    if let Some(t) = t_after_costs {
        eprintln!(
            "[PHASM_PERF_TRACE]   ├─ safety+tier+combine     {:>8.1} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
    let t_before_stc = if trace { Some(std::time::Instant::now()) } else { None };
    mem_trace("04 after combine");
    let n_cover = combined_cover.len();
    if n_cover == 0 {
        return Err(StegoError::InvalidVideo(
            "openh264 4-domain encode: combined cover empty".into(),
        ));
    }
    let w = n_cover / m_total;
    if w == 0 {
        return Err(StegoError::MessageTooLarge);
    }
    let used_cover = m_total * w;

    // STC plan on combined cover.
    let cover_slice: Vec<u8> = combined_cover[..used_cover].to_vec();
    let cost_slice: Vec<f32> = combined_costs[..used_cover].to_vec();
    let hhat = generate_hhat(STC_H, w, hhat_seed);
    let plan = stc_embed(&cover_slice, &cost_slice, frame_bits, &hhat, STC_H, w)
        .ok_or(StegoError::MessageTooLarge)?;
    if let Some(t) = t_before_stc {
        eprintln!(
            "[PHASM_PERF_TRACE]   └─ stc_embed(Viterbi)      {:>8.1} ms",
            t.elapsed().as_secs_f64() * 1000.0
        );
    }
    let dt_stc = t_stc.map(|t| t.elapsed());
    mem_trace("05 after STC plan");

    // Extend STC result to full combined cover length (positions past
    // `used_cover` stay unchanged) so split_plan_4domain sees the
    // full vector.
    let t_overrides = if trace { Some(std::time::Instant::now()) } else { None };
    let mut full_stego_bits = Vec::with_capacity(n_cover);
    full_stego_bits.extend_from_slice(&plan.stego_bits);
    full_stego_bits.extend_from_slice(&combined_cover[used_cover..]);
    debug_assert_eq!(full_stego_bits.len(), n_cover);
    let domain_plan = split_plan_4domain(&full_stego_bits, &boundaries);

    // Build per-domain override entries keyed by canonical
    // PositionKey::raw(). All 4 domains contribute; OH264's per-
    // domain emit hooks look up by the same key.
    let mut overrides_map: HashMap<u64, u8> = HashMap::new();
    for (positions, bits, cover_bits) in [
        (&baseline_walk.cover.coeff_sign_bypass.positions,
         &domain_plan.coeff_sign_bypass,
         &baseline_walk.cover.coeff_sign_bypass.bits),
        (&baseline_walk.cover.coeff_suffix_lsb.positions,
         &domain_plan.coeff_suffix_lsb,
         &baseline_walk.cover.coeff_suffix_lsb.bits),
        (&baseline_walk.cover.mvd_sign_bypass.positions,
         &domain_plan.mvd_sign_bypass,
         &baseline_walk.cover.mvd_sign_bypass.bits),
        (&baseline_walk.cover.mvd_suffix_lsb.positions,
         &domain_plan.mvd_suffix_lsb,
         &baseline_walk.cover.mvd_suffix_lsb.bits),
    ] {
        let n = positions.len().min(bits.len()).min(cover_bits.len());
        for i in 0..n {
            if bits[i] != cover_bits[i] {
                overrides_map.insert(positions[i].raw(), bits[i]);
            }
        }
    }
    // A.1.9 flip-set bisection (CASCADE.V2, #778) — keep only flips whose
    // GOP-relative frame_idx (PositionKey bits 0..23) is in
    // PHASM_DIAG_FLIP_FRAMES=lo-hi; drop the rest. selfextract will diff at
    // dropped positions, but the per-domain COUNT match (p1 vs p2) is the
    // desync indicator and is independent of which flips are applied. Pins
    // which frame's flip desyncs the walker. Diagnostic-only; default unset.
    let parse_range = |spec: &str| -> Option<(u32, u32)> {
        spec.split_once('-')
            .and_then(|(a, b)| Some((a.trim().parse().ok()?, b.trim().parse().ok()?)))
    };
    if let Ok(spec) = std::env::var("PHASM_DIAG_FLIP_FRAMES") {
        if let Some((lo, hi)) = parse_range(&spec) {
            let before = overrides_map.len();
            overrides_map.retain(|&k, _| {
                let f = (k & 0xFF_FFFF) as u32;
                f >= lo && f <= hi
            });
            eprintln!("[PHASM_DIAG_FLIP_FRAMES={lo}-{hi}] kept {}/{} flips", overrides_map.len(), before);
        }
    }
    if let Ok(spec) = std::env::var("PHASM_DIAG_FLIP_MBS") {
        if let Some((lo, hi)) = parse_range(&spec) {
            let before = overrides_map.len();
            overrides_map.retain(|&k, _| {
                let mb = ((k >> 24) & 0xFF_FFFF) as u32;
                mb >= lo && mb <= hi
            });
            eprintln!("[PHASM_DIAG_FLIP_MBS={lo}-{hi}] kept {}/{} flips", overrides_map.len(), before);
        }
    }
    if std::env::var("PHASM_DIAG_FLIP_DUMP").as_deref() == Ok("1") && overrides_map.len() <= 40 {
        let mut ks: Vec<u64> = overrides_map.keys().copied().collect();
        ks.sort_unstable();
        for k in ks {
            let (frame, mb) = ((k & 0xFF_FFFF) as u32, ((k >> 24) & 0xFF_FFFF) as u32);
            let (dom, syn) = (((k >> 48) & 0xF) as u8, ((k >> 52) & 0xFFF) as u16);
            let (tag, payload) = ((syn & 0x7) as u8, syn >> 3);
            eprintln!("[FLIP] frame={frame} mb={mb} dom={dom} tag={tag} payload=0x{payload:03X} val={}", overrides_map[&k]);
        }
    }

    let n_flips = overrides_map.len();
    {
        let mut map = override_map.lock().expect("override map lock");
        map.clear();
        for (k, v) in overrides_map.iter() {
            map.insert(*k, *v);
        }
    }
    let dt_overrides = t_overrides.map(|t| t.elapsed());

    // Pass 2 — re-encode with overrides. Default path is wire-only.
    //
    // #538 Phase 4.6: with `wire_only` ON (the default — see the
    // `let wire_only` block above) Pass 2 runs the OH264 fork in
    // wire-only override mode. In that mode `apply_coeff_hooks_to_level`
    // + `phasm_apply_mvd_hooks` populate a per-MB scratch table instead
    // of mutating *level / MV state. The CABAC emit reads the scratch at
    // the bypass-bin site and writes the override bin directly to the
    // wire. Encoder's pDecPic stays clean by construction — no C.8.x
    // cascade-break needed for that mode, so `dual_recon` is also forced
    // false.
    //
    // Only `PHASM_USE_WIRE_ONLY=0` falls back to the legacy mutating +
    // dual_recon path; that path + the C.8.x machinery are slated for
    // deletion in #539 Phase 5.
    let t_pass2 = if trace { Some(std::time::Instant::now()) } else { None };
    // Keep an Arc clone of `applied` so the probe below can read the fired count.
    let applied_for_probe = applied.clone();
    if wire_only {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1) };
        if trace {
            eprintln!(
                "[PHASM_PERF_TRACE wire_only] enabled for Pass 2; dual_recon disabled; pass2 replay from cache (len={})",
                decision_cache.as_ref().map(|c| c.lock().map(|g| g.len()).unwrap_or(0)).unwrap_or(0),
            );
        }
    }
    set_549_pass_idx(2);
    let result = encode_once(
        yuv, width, height, n_frames, opts,
        override_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ !wire_only,
        if wire_only { PassMode::Replay } else { PassMode::Passthrough },
        decision_cache.clone(),
        // P3.3a DISABLED (2026-05-27): post-frame DPB correction changes
        // Pass 2 bitstream vs Pass 1 → cover positions diverge → roundtrip
        // decode fails (FrameCorrupted) on all real 1080p content, even with
        // 20-byte payloads. Same structural issue as D2.2: any encoder
        // reconstruction change in the 2-pass architecture breaks the STC plan.
        // Unit tests pass only on tiny synthetic fixtures where cascade is nil.
        None, // was: Some(&baseline_walk.cover),
        // P3.3b.4: pass captured coefficients for replay with STC flips.
        Some(captured_coeffs),
        0, // frame_idx_base: whole-clip / per-GOP no-shadow path (no offset)
    );
    /* #549 Bug 4 fix (2026-05-19): do NOT flip wire_only_global back to 0
     * here. The determinism check + bisect-A/B below re-run encode_once
     * to validate that empty-override re-encodes match Pass 1 byte-for-
     * byte. With wire_only=1 in Pass 1 (Bug 2 fix) but wire_only=0 here,
     * those re-encodes take a STRUCTURALLY DIFFERENT code path through
     * `apply_coeff_hooks_to_level` (the wire_only branch computes
     * scratch keys + calls dispatch_hook + skips level mutation; the
     * !wire_only branch calls dispatch_hook only). With empty overrides,
     * neither branch fires any flips, but the encoder appears to drift
     * tiny accumulating differences across many frames that compound to
     * `first_diff=Some(995)` at 6f, `Some(5870)` at 12f, etc. Keeping
     * wire_only=1 for the rest of the orchestrator preserves Pass-1↔
     * Pass-1-redo symmetry, closing the determinism gap. */
    set_549_pass_idx(0);
    let dt_pass2 = t_pass2.map(|t| t.elapsed());
    mem_trace("06 after Pass2 (encode)");

    // #530 PROBE — verify encoder/walker symmetry on Pass-2 output.
    // For round-trip to work, walker on Pass-2 must recover
    // combined_cover == STC plan's full_stego_bits.
    if trace {
        let applied_count = *applied_for_probe.lock().expect("applied lock");
        eprintln!(
            "[PHASM_PERF_TRACE applied] override entries={} fired={}",
            n_flips, applied_count,
        );
        // Determinism check: re-encode WITHOUT overrides + dual_recon=false
        // (same args as Pass 1). Expect byte-identical to baseline_bitstream.
        let empty_om: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
        let det_mbt: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
        let det_app: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        if let Ok(re_bitstream) = encode_once(
            yuv, width, height, n_frames, opts,
            empty_om, det_mbt, det_app,
            mb_width, mb_per_frame,
            /* dual_recon = */ false,
            PassMode::Passthrough,
            None,
            None, // P3.3a: diagnostic re-encode — no DPB correction
            None, // P3.3b.4: no coefficient replay
            0, // frame_idx_base: diagnostic re-encode (whole-clip, no offset)
        ) {
            let cmp = baseline_bitstream.len().min(re_bitstream.len());
            let mut first_det_diff: Option<usize> = None;
            for i in 0..cmp {
                if baseline_bitstream[i] != re_bitstream[i] {
                    first_det_diff = Some(i);
                    break;
                }
            }
            eprintln!(
                "[PHASM_PERF_TRACE determinism] pass1_bytes={} re_bytes={} first_diff={:?} (None=byte-identical)",
                baseline_bitstream.len(), re_bitstream.len(), first_det_diff,
            );
        }
        // Bisect A: empty overrides + dual_recon=TRUE. Isolates whether
        // dual_recon flag alone changes output even when no flips are
        // applied. If this matches Pass-1, dual_recon is byte-clean; the
        // drift must come from override-application.
        let empty_om_a: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
        let det_mbt_a: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
        let det_app_a: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
        if let Ok(re_a) = encode_once(
            yuv, width, height, n_frames, opts,
            empty_om_a, det_mbt_a, det_app_a,
            mb_width, mb_per_frame,
            /* dual_recon = */ true,
            PassMode::Passthrough,
            None,
            None, // P3.3a: diagnostic re-encode — no DPB correction
            None, // P3.3b.4: no coefficient replay
            0, // frame_idx_base: diagnostic re-encode (whole-clip, no offset)
        ) {
            let cmp = baseline_bitstream.len().min(re_a.len());
            let mut first_a_diff: Option<usize> = None;
            for i in 0..cmp {
                if baseline_bitstream[i] != re_a[i] {
                    first_a_diff = Some(i);
                    break;
                }
            }
            eprintln!(
                "[PHASM_PERF_TRACE bisect-A] empty_overrides + dual_recon=TRUE: bytes={} first_diff={:?} (None=matches Pass-1)",
                re_a.len(), first_a_diff,
            );
        }
        // Bisect B: ONE override entry + dual_recon=TRUE. Isolates whether
        // any single override fire alone triggers drift past the override
        // site. If yes → C.8 dual-recon doesn't fully restore pDecPic to
        // clean. If no → drift is N>1 cumulative.
        if let Some((&one_k, &one_v)) = overrides_map.iter().next() {
            let one_om: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new({
                let mut m = HashMap::new();
                m.insert(one_k, one_v);
                m
            }));
            let det_mbt_b: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
            let det_app_b: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
            if let Ok(re_b) = encode_once(
                yuv, width, height, n_frames, opts,
                one_om, det_mbt_b, det_app_b.clone(),
                mb_width, mb_per_frame,
                /* dual_recon = */ true,
                PassMode::Passthrough,
                None,
                None, // P3.3a: diagnostic re-encode — no DPB correction
                None, // P3.3b.4: no coefficient replay
                0, // frame_idx_base: diagnostic re-encode (whole-clip, no offset)
            ) {
                let cmp = baseline_bitstream.len().min(re_b.len());
                let mut first_b_diff: Option<usize> = None;
                let mut total_b_diff = 0usize;
                for i in 0..cmp {
                    if baseline_bitstream[i] != re_b[i] {
                        if first_b_diff.is_none() { first_b_diff = Some(i); }
                        total_b_diff += 1;
                    }
                }
                let fired_b = *det_app_b.lock().expect("app b lock");
                eprintln!(
                    "[PHASM_PERF_TRACE bisect-B] ONE override (key=0x{:016x} bit={}): bytes={} diff_bytes={} first_diff={:?} fired={}",
                    one_k, one_v, re_b.len(),
                    (re_b.len() as i64) - (baseline_bitstream.len() as i64),
                    first_b_diff, fired_b,
                );
                eprintln!(
                    "[PHASM_PERF_TRACE bisect-B] total_b_diff_bytes_in_common_prefix={}",
                    total_b_diff,
                );
            }
        }
        if let Ok(ref stego_bytes) = result {
            // Bitstream byte-level diff first — tells us whether Pass-1
            // and Pass-2 diverge before or after the first override
            // position. Pass-1 bytes already on hand as
            // `baseline_bitstream`.
            let p1_len = baseline_bitstream.len();
            let p2_len = stego_bytes.len();
            let cmp_bytes = p1_len.min(p2_len);
            let mut byte_first_diff: Option<usize> = None;
            for i in 0..cmp_bytes {
                if baseline_bitstream[i] != stego_bytes[i] {
                    byte_first_diff = Some(i);
                    break;
                }
            }
            eprintln!(
                "[PHASM_PERF_TRACE bitstream] p1_bytes={} p2_bytes={} (diff_bytes={}) first_byte_diff={:?}",
                p1_len, p2_len, (p2_len as i64) - (p1_len as i64), byte_first_diff,
            );

            if let Ok(p2_walk) = walk_annex_b_for_cover_with_options(
                stego_bytes,
                WalkOptions { record_mvd: true, ..Default::default() },
            ) {
                let diag_costs = DomainCosts::default();
                let (p2_combined, _, p2_boundaries) =
                    combine_cover_4domain(&p2_walk.cover, &diag_costs, weights);
                let p1_n = boundaries.total();
                let p2_n = p2_boundaries.total();
                eprintln!(
                    "[PHASM_PERF_TRACE symmetry] p1_total={} p2_total={} (diff={})",
                    p1_n, p2_n, (p2_n as i64) - (p1_n as i64),
                );
                // Per-domain position-count delta — if any domain count
                // changes, Pass-2 emits different MB structure than Pass-1
                // and the index-wise diff below is artifically inflated
                // by misalignment, not necessarily real bit divergence.
                eprintln!(
                    "[PHASM_PERF_TRACE symmetry] per-domain p1 CS={} CSL={} MVDs={} MVDsl={}",
                    boundaries.n_coeff_sign, boundaries.n_coeff_suffix,
                    boundaries.n_mvd_sign, boundaries.n_mvd_suffix,
                );
                eprintln!(
                    "[PHASM_PERF_TRACE symmetry] per-domain p2 CS={} CSL={} MVDs={} MVDsl={}",
                    p2_boundaries.n_coeff_sign, p2_boundaries.n_coeff_suffix,
                    p2_boundaries.n_mvd_sign, p2_boundaries.n_mvd_suffix,
                );
                let cmp_n = p1_n.min(p2_n);
                let plan_slice = &full_stego_bits[..cmp_n];
                let recover_slice = &p2_combined[..cmp_n];
                let mut diff = 0usize;
                let mut first_diff = None;
                let cs_end = boundaries.n_coeff_sign;
                let csl_end = cs_end + boundaries.n_coeff_suffix;
                let mvds_end = csl_end + boundaries.n_mvd_sign;
                let mut diff_cs = 0;
                let mut diff_csl = 0;
                let mut diff_mvds = 0;
                let mut diff_mvdsl = 0;
                for i in 0..cmp_n {
                    if plan_slice[i] != recover_slice[i] {
                        diff += 1;
                        if first_diff.is_none() { first_diff = Some(i); }
                        if i < cs_end { diff_cs += 1; }
                        else if i < csl_end { diff_csl += 1; }
                        else if i < mvds_end { diff_mvds += 1; }
                        else { diff_mvdsl += 1; }
                    }
                }
                let domain_of = |i: usize| -> &'static str {
                    if i < cs_end { "CS" }
                    else if i < csl_end { "CSL" }
                    else if i < mvds_end { "MVDs" }
                    else { "MVDsl" }
                };
                eprintln!(
                    "[PHASM_PERF_TRACE symmetry] plan vs walker diffs={diff} of {cmp_n} (CS={diff_cs} CSL={diff_csl} MVDs={diff_mvds} MVDsl={diff_mvdsl}) first_diff={:?} domain={}",
                    first_diff, first_diff.map(domain_of).unwrap_or("--"),
                );
                // #778 decisive probe — run the DECODER's extract logic on
                // the encoder's OWN post-Pass2 walk, and compare to the
                // embedded frame_bits. If this recovers frame_bits exactly,
                // the encode→walk→extract chain is self-consistent and the
                // Mode-B failure is 100% in the StreamingDecodeSession slab
                // path (it must be walking different bytes). If it does NOT
                // recover, the bug is in combine/extract alignment here.
                {
                    let p2_n2 = p2_combined.len();
                    let m2 = frame_bits.len();
                    let w2 = if m2 > 0 { p2_n2 / m2 } else { 0 };
                    if w2 > 0 {
                        let used2 = m2 * w2;
                        let hhat2 = generate_hhat(STC_H, w2, hhat_seed);
                        let rec = stc_extract(&p2_combined[..used2], &hhat2, w2);
                        let cmpb = rec.len().min(frame_bits.len());
                        let mut fb_diff = 0usize;
                        let mut fb_first = None;
                        for i in 0..cmpb {
                            if rec[i] != frame_bits[i] {
                                fb_diff += 1;
                                if fb_first.is_none() { fb_first = Some(i); }
                            }
                        }
                        eprintln!(
                            "[PHASM_PERF_TRACE selfextract] n_cover={} m_total={} w={} used={} frame_bits_diff={}/{} first_diff_bit={:?}",
                            p2_n2, m2, w2, used2, fb_diff, cmpb, fb_first,
                        );
                    }
                }
                // #538.4.7 diag: dump first ~20 diffs with their position
                // keys so we can map them back to encoder hooks.
                let mut dumped = 0usize;
                for i in 0..cmp_n {
                    if dumped >= 20 { break; }
                    if plan_slice[i] != recover_slice[i] {
                        // Domain-local index = i - domain_start
                        let (domain, local_idx) = if i < cs_end {
                            ("CS", i)
                        } else if i < csl_end {
                            ("CSL", i - cs_end)
                        } else if i < mvds_end {
                            ("MVDs", i - csl_end)
                        } else {
                            ("MVDsl", i - mvds_end)
                        };
                        let p1_pos = match domain {
                            "CS" => p2_walk.cover.coeff_sign_bypass.positions.get(local_idx).copied(),
                            "CSL" => p2_walk.cover.coeff_suffix_lsb.positions.get(local_idx).copied(),
                            "MVDs" => p2_walk.cover.mvd_sign_bypass.positions.get(local_idx).copied(),
                            "MVDsl" => p2_walk.cover.mvd_suffix_lsb.positions.get(local_idx).copied(),
                            _ => None,
                        };
                        // Decode the PositionKey to identify which encoder
                        // hook produced it. PositionKey layout:
                        //   bits 0..23: frame_idx
                        //   bits 24..47: mb_addr
                        //   bits 48..51: domain enum
                        //   bits 52..63: syntax_path packed (3-bit tag + 9-bit payload)
                        // SyntaxPath payload:
                        //   tag 0 Luma4x4:        block_idx(4) | coeff_idx(4)<<4 | kind(1)<<8
                        //   tag 1 LumaAcIntra16x16: block_idx(4) | coeff_idx(4)<<4 | kind(1)<<8
                        //   tag 2 ChromaAc:       plane(1) | block_idx(2)<<1 | coeff_idx(4)<<3 | kind(1)<<7
                        //   tag 3 ChromaDc:       plane(1) | coeff_idx(2)<<1 | kind(1)<<3
                        //   tag 4 LumaDcIntra16x16: coeff_idx(4) | kind(1)<<4
                        let pk_decoded = p1_pos.map(|p| {
                            let raw = p.raw();
                            let frame_idx = (raw & 0xFFFFFF) as u32;
                            let mb_addr = ((raw >> 24) & 0xFFFFFF) as u32;
                            let domain_id = ((raw >> 48) & 0xF) as u8;
                            let syntax = ((raw >> 52) & 0xFFF) as u16;
                            let tag = (syntax & 0x7) as u8;
                            let payload = (syntax >> 3) as u16;
                            let tag_name = match tag {
                                0 => "Luma4x4",
                                1 => "LumaAcI16",
                                2 => "ChromaAc",
                                3 => "ChromaDc",
                                4 => "LumaDcI16",
                                _ => "Unknown",
                            };
                            format!(
                                "tag={tag_name} payload=0x{payload:03X} dom={domain_id} mb={mb_addr} frame={frame_idx}"
                            )
                        }).unwrap_or_default();
                        eprintln!(
                            "[PHASM_PERF_TRACE diff#{dumped}] cmp_i={i} domain={domain} local={local_idx} plan_bit={} walker_bit={} :: {pk_decoded}",
                            plan_slice[i], recover_slice[i],
                        );
                        dumped += 1;
                    }
                }
            }
        }
    }

    if trace {
        let dt_total = t_start.unwrap().elapsed();
        eprintln!(
            "[PHASM_PERF_TRACE oh264_4domain] {w}x{h} f={f} n_cover={nc} m_total={mt} w={ww} flips={nf}",
            w = width, h = height, f = n_frames,
            nc = n_cover, mt = m_total, ww = w, nf = n_flips,
        );
        let report = |label: &str, dt: Option<std::time::Duration>| {
            if let Some(d) = dt {
                let ms = d.as_secs_f64() * 1000.0;
                let pct = ms / (dt_total.as_secs_f64() * 1000.0) * 100.0;
                eprintln!("[PHASM_PERF_TRACE]   {label:<22} {ms:>9.1} ms  ({pct:>5.1}%)");
            }
        };
        // (Pass-1 encode + walk are timed in `produce_gop_cover` now.)
        report("tier + combine + STC", dt_stc);
        report("override map build", dt_overrides);
        report("Pass 2 OH264 stego", dt_pass2);
        eprintln!(
            "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms  (consume half)",
            "CONSUME TOTAL", dt_total.as_secs_f64() * 1000.0,
        );
    }

    // #798 — return the tier the encoder actually resolved/applied so
    // the streaming session can stash it for the mobile success-screen
    // badge (mobile has no ffmpeg to re-resolve from YUV).
    result.map(|bytes| (bytes, resolved_tier))
}

/// D.0.7.2 — expose the MSB-first byte→bit helper for the streaming
/// session to convert its per-chunk `chunk_frame` + inner-stego-frame
/// concatenation into bits before calling
/// [`h264_encode_gop_framed_bits`].
pub(crate) fn bytes_to_bits_msb_first_pub(bytes: &[u8]) -> Vec<u8> {
    bytes_to_bits_msb_first(bytes)
}

/// #796 Mode A — 4-domain OH264 baseline-walk capacity probe.
///
/// Runs one baseline OH264 encode + walker and returns per-domain
/// cover counts so `h264_stego_capacity_4domain` can report numbers
/// that match the OH264 streaming session's actual per-GOP STC budget.
///
/// The pure-Rust `pass1_count_4domain` walker (used by the legacy
/// capacity path) produces wildly different cover sizes on easy-to-
/// compress content (lumix, dji) vs OH264 — measured 32× over-report
/// on lumix at 480p × 30f. This walker fixes that.
///
/// Returns the OH264-walked `DomainCover` (with positions + magnitudes
/// in all 4 domains) so the caller can re-run the tier filter,
/// allocator, and any other cover-derived computation against the
/// actual OH264 wire structure.
pub fn h264_walk_cover(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<super::stego::DomainCover, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let empty_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let bitstream = encode_once(
        yuv, width, height, n_frames, opts,
        empty_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        None,
        None,
        None,
        0, // frame_idx_base: whole-clip (no offset)
    )?;
    let walk = walk_annex_b_for_cover_with_options(
        &bitstream,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    Ok(walk.cover)
}

// ─────────────────── CAP2.1 — accurate per-GOP capacity ────────────────
//
// The legacy `CoeffSign_cover_bits × 0.40` heuristic over-reports the true
// STC budget 8–350× (it ignores the runtime `safe_msl` cascade-safety gate,
// the tier ∞-costs, and the Tier-3 content-cost landscape that makes STC's
// achievable rate 0.003–0.12, not 0.40). These helpers rebuild the EXACT
// cover the streaming encode sees per GOP, then binary-search the largest
// chunk_frame payload STC can embed — accurate by construction, not by
// calibration. Design: docs/design/video/h264/capacity-spreading-plan.md.
//
// NOTE (CAP2 follow-on): the cover construction below MIRRORS the encode
// (`h264_encode_gop_framed_bits`: content_costs →
// safe_msl → tier → combine). The reported-vs-real-encoder gate
// (`core/tests/horse_flag_capacity_797.rs`) guards divergence. Extracting a
// single shared helper that both encode + probe call is tracked for CAP2.2
// / the pure-Rust port (capacity-spreading-plan.md §9).

/// Fixed hhat seed for capacity trials. STC capacity is a property of the
/// cost distribution + (h, w) — NOT the specific random parity-check — so a
/// constant seed yields the same budget as the passphrase-derived hhat the
/// encode uses, and lets the probe estimate capacity without a passphrase.
const CAP2_TRIAL_SEED: [u8; 32] = [0x2c; 32];

/// Safety derating on the per-GOP STC budget. The trial uses ONE
/// representative hhat (the probe can't know the real passphrase-derived
/// parity-check), so the single-sample estimate carries ±~20% hhat-jitter
/// near the ∞-cost margin. 5/6 (−16.7%) turns it into a conservative floor:
/// on the real-world corpus at NATIVE resolution + full length (#810,
/// capacity-spreading-plan.md §11) the worst over-report was lumix 1.05×,
/// so 5/6 brings ALL native-res reports ≤ 1.0× the real encoder.
///
/// NOTES (capacity-spreading-plan.md §10–§11):
/// - Flooring over MANY random hhats was REJECTED — random parity-checks
///   include pathological instances 3–4× worse than any real passphrase
///   draw (→ 0.24×, uselessly conservative).
/// - The "fast-motion over-report" that motivated a content-adaptive floor
///   was a DOWNSCALE artifact (horse_flag 1.29× at 720p but 0.60× at native
///   1080p). The residual per-GOP jitter is best stabilised in CAP2.2,
///   where Σ aggregation makes per-GOP accuracy drive the total (under the
///   even-split `min` it doesn't — aggregation dominates).
const CAP2_SAFETY_NUM: usize = 5;
const CAP2_SAFETY_DEN: usize = 6;

/// Per-GOP capacity: max chunk_frame PAYLOAD bytes per cascade tier
/// (index 0..=4), plus the CoeffSign cover-bit count (kept so the #475
/// progress probe needn't re-walk the GOP for its callback).
pub(crate) struct GopCapacity {
    pub per_tier_payload: [usize; 5],
    pub coeff_sign_cover_bits: usize,
    /// #809 — total injectable cover bits in the 3 shadow domains
    /// (CoeffSign + CoeffSuffixLsb + MvdSign) for THIS GOP, read straight
    /// off the OH264 cover walk. Lets `h264_video_capacity`
    /// compute the shadow capacity from the same encoder it will actually
    /// use — instead of a separate pure-Rust whole-video `pass1_count_4domain`
    /// (the ~45 s/clip bottleneck the #809 profiling pinned).
    pub injectable_cover_bits: usize,
}

/// Deterministic ~incompressible payload bytes for an STC trial. The encoder
/// embeds the encrypted (random-looking) frame, so a pseudo-random trial
/// message reproduces the encode's syndrome reachability under the ∞-cost
/// gate. A structured/all-zero payload would be unrepresentatively easy.
fn cap2_trial_payload(len: usize) -> Vec<u8> {
    let mut v = Vec::with_capacity(len);
    let mut s: u64 = 0x9E37_79B9_7F4A_7C15;
    for _ in 0..len {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v.push((s >> 33) as u8);
    }
    v
}

/// Binary-search the largest chunk_frame PAYLOAD (bytes) that STC can embed
/// into this GOP's combined cover. Reproduces the streaming encode's exact
/// per-GOP `stc_embed → None ⇒ MessageTooLarge` boundary, including the
/// chunk_frame header bits and the inline→u32 length escape. Exponential
/// probe from the low end (cheap STC calls at large `w`) then bisect to an
/// 8-byte tolerance, reporting the embeddable (conservative) end.
fn cap2_stc_trial_max_payload(combined_cover: &[u8], combined_costs: &[f32]) -> usize {
    let n_cover = combined_cover.len();
    let embeddable = |payload: usize| -> bool {
        let body = cap2_trial_payload(payload);
        let framed = match super::stego::chunk_frame::build_first_chunk_frame_v3_1(
            body.len().max(1) as u32,
            &body,
        ) {
            Ok(f) => f,
            Err(_) => return false,
        };
        let frame_bits = bytes_to_bits_msb_first(&framed);
        let m_total = frame_bits.len();
        if m_total == 0 {
            return true;
        }
        let w = n_cover / m_total;
        if w == 0 {
            return false;
        }
        let used = m_total * w;
        let hhat = generate_hhat(STC_H, w, &CAP2_TRIAL_SEED);
        stc_embed(
            &combined_cover[..used],
            &combined_costs[..used],
            &frame_bits,
            &hhat,
            STC_H,
            w,
        )
        .is_some()
    };

    if !embeddable(0) {
        return 0;
    }
    let header = super::stego::chunk_frame::CHUNK_FRAME_FIRST_HEADER_LEN_V3_1;
    let hard_max = (n_cover / 8).saturating_sub(header);
    if hard_max == 0 {
        return 0;
    }
    // Exponential probe for a failing upper bound.
    let mut lo = 0usize; // embeddable
    let mut hi = 1usize;
    loop {
        if hi >= hard_max {
            if embeddable(hard_max) {
                return hard_max;
            }
            hi = hard_max;
            break;
        }
        if embeddable(hi) {
            lo = hi;
            hi = hi.saturating_mul(2);
        } else {
            break;
        }
    }
    // Bisect [lo (embeddable) .. hi (not)] to 8-byte tolerance.
    while lo + 8 < hi {
        let mid = lo + (hi - lo) / 2;
        if embeddable(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    // Residual safety derating — see `CAP2_SAFETY_NUM`/`DEN`.
    lo.saturating_mul(CAP2_SAFETY_NUM) / CAP2_SAFETY_DEN
}

/// CAP2.1 — accurate per-tier capacity for ONE GOP. Baseline-encode + walk
/// (the same cover the streaming encode sees), compute content costs +
/// `safe_msl` gate ONCE (tier-independent), then per tier apply the ∞-cost
/// filter, combine, and STC-trial. `gop_yuv` must hold exactly
/// `gop_n_frames` frames.
pub(crate) fn h264_gop_capacity(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    gop_n_frames: u32,
    opts: EncodeOpts,
    weights: &CostWeights,
    // #809 — when false, compute ONLY tier 0 (the reported primary capacity)
    // and fill tiers 1..4 from it. The tier filter is capacity-neutral on
    // real content (#814 / D'.7), so this is exact for the live mobile path
    // at 5× fewer STC Viterbis. `true` (CLI diagnostic / calibration tests)
    // computes the true per-tier breakdown.
    full_tiers: bool,
) -> Result<GopCapacity, StegoError> {
    validate_dims(gop_yuv, width, height, gop_n_frames)?;
    let prof = std::env::var("PHASM_CAP_PROFILE").is_ok();
    let mut t = std::time::Instant::now();
    macro_rules! lap {
        ($n:expr) => {
            if prof {
                eprintln!("  [cap-prof] {:<16} {:?}", $n, t.elapsed());
                t = std::time::Instant::now();
            }
        };
    }
    // 4a (#893): global-free clean cover — no StegoSession / fork-global touches,
    // so the per-GOP probe is thread-safe for the parallel-GOP runner
    // (`streaming_map_ordered`). Byte-identical to the prior
    // `encode_once(Passthrough, dual_recon=false, frame_idx_base=0)` it replaces
    // (B-lite.1 gate `blite1_clean_encode_matches_plain`), so capacity is
    // unchanged. Everything downstream (walk / content costs / safe-MSL gate /
    // STC trial) is already pure Rust.
    let bitstream = clean_encode_gop(gop_yuv, width, height, gop_n_frames, opts)?;
    let baseline_walk = walk_annex_b_for_cover_with_options(
        &bitstream,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("cap2 walk: {e}")))?;

    let coeff_sign_cover_bits = baseline_walk.cover.coeff_sign_bypass.bits.len();
    // #809 — 3 shadow-injectable domains, same definition the shadow formula
    // uses (CS + CSL + MvdSign; MvdSuffixLsb is excluded — it's the safe-MSL
    // gated domain, not part of the shadow LSB pool).
    let injectable_cover_bits = baseline_walk.cover.coeff_sign_bypass.bits.len()
        + baseline_walk.cover.coeff_suffix_lsb.bits.len()
        + baseline_walk.cover.mvd_sign_bypass.bits.len();
    lap!("encode+walk");

    // content costs + safe_msl gate — computed ONCE (tier-independent).
    let use_uniform = std::env::var("PHASM_DIAG_UNIFORM_COSTS")
        .map(|v| v == "1")
        .unwrap_or(false);
    let mut base_costs = if use_uniform {
        DomainCosts::default()
    } else {
        super::stego::content_costs::compute_content_costs_yuv(
            gop_yuv, width, height, gop_n_frames, &baseline_walk.cover, opts.qp,
        )?
    };
    lap!("content_costs");
    {
        let safe_msb = super::stego::cascade_safety::analyze_safe_mvd_subset(
            &baseline_walk.mvd_meta, baseline_walk.mb_w, baseline_walk.mb_h,
        );
        let safe_msl = super::stego::cascade_safety::derive_msl_safe_from_msb(
            &baseline_walk.cover.mvd_sign_bypass.positions,
            &safe_msb,
            &baseline_walk.cover.mvd_suffix_lsb.positions,
        );
        let n_msl = base_costs.mvd_suffix_lsb.len();
        let safe_len = safe_msl.len().min(n_msl);
        for i in 0..safe_len {
            if !safe_msl[i] {
                base_costs.mvd_suffix_lsb[i] = f32::INFINITY;
            }
        }
        for i in safe_len..n_msl {
            base_costs.mvd_suffix_lsb[i] = f32::INFINITY;
        }
    }
    lap!("safe_msl");

    let csb_qp = vec![opts.qp; baseline_walk.cover.coeff_sign_bypass.len()];
    let csl_qp = vec![opts.qp; baseline_walk.cover.coeff_suffix_lsb.len()];

    let mut per_tier_payload = [0usize; 5];
    let last_tier: u8 = if full_tiers { 4 } else { 0 };
    for tier in 0u8..=last_tier {
        let mut costs = base_costs.clone();
        if tier > 0 {
            use super::stego::tier_filter::apply_tier_filter;
            let csb_keep =
                apply_tier_filter(&baseline_walk.cover.coeff_sign_bypass, &csb_qp, tier);
            let csl_keep =
                apply_tier_filter(&baseline_walk.cover.coeff_suffix_lsb, &csl_qp, tier);
            for (i, &keep) in csb_keep.iter().enumerate() {
                if !keep && i < costs.coeff_sign_bypass.len() {
                    costs.coeff_sign_bypass[i] = f32::INFINITY;
                }
            }
            for (i, &keep) in csl_keep.iter().enumerate() {
                if !keep && i < costs.coeff_suffix_lsb.len() {
                    costs.coeff_suffix_lsb[i] = f32::INFINITY;
                }
            }
        }
        let (cover, costs_v, _b) =
            combine_cover_4domain(&baseline_walk.cover, &costs, weights);
        per_tier_payload[tier as usize] = cap2_stc_trial_max_payload(&cover, &costs_v);
    }
    // #809 — tier filter is capacity-neutral on real content (#814); fill the
    // un-probed tiers from tier 0 so callers see consistent values.
    if !full_tiers {
        let t0 = per_tier_payload[0];
        for t in 1..5 {
            per_tier_payload[t] = t0;
        }
    }
    lap!("tier_loop");

    Ok(GopCapacity {
        per_tier_payload,
        coeff_sign_cover_bits,
        injectable_cover_bits,
    })
}

// ---------- internals ----------

fn validate_dims(yuv: &[u8], width: u32, height: u32, n_frames: u32) -> Result<(), StegoError> {
    if width % 16 != 0 || height % 16 != 0 {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    let expected = frame_size * n_frames as usize;
    if yuv.len() != expected {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {} ({}x{} x {} frames)",
            yuv.len(), expected, width, height, n_frames
        )));
    }
    Ok(())
}

/// CAP2.3 §6 — encode ONE plain (no-stego) GOP for the concentrate+tail plain
/// tail. A single `encode_once` (no Pass-1 walk, no STC, no content-cost
/// computation — the cheap path the capacity probe uses at
/// `h264_gop_capacity`), so a small-payload clip's tail GOPs cost
/// ~plain-encode speed instead of a full stego encode.
///
/// MANDATORY reset: the streaming stego GOPs that precede the tail leave the
/// OH264 fork's per-session scratch state dirty (the stego path resets at its
/// entry — openh264_stego.rs:291 — but `encode_once` does NOT self-reset the
/// fork statics). Without this reset the first plain GOP would inherit the last
/// stego GOP's fork state (#548 class: two sequential calls sharing fork state
/// → spurious diffs). Reset here so every plain GOP encodes from a clean
/// session, exactly like the stego path.
#[cfg(feature = "h264-encoder")]
pub(crate) fn oh264_plain_encode_gop(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<Vec<u8>, StegoError> {
    // #548 v1.0 BLOCKER fix — reset the process-global fork statics (bypass
    // scratch + last-MB sentinels + wire-only flag) before this encode session,
    // exactly like the stego path. They are process-global on the production
    // fork, so a prior GOP's state would otherwise leak into this plain-tail
    // encode (see `produce_gop_cover`).
    unsafe { core_openh264_sys::phasm_reset_encoder_session_state() };
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let empty_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    encode_once(
        gop_yuv, width, height, n_frames, opts,
        empty_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        None,
        None,
        None,
        0, // frame_idx_base: whole-clip plain-tail GOP (no offset)
    )
}

/// Global-free clean cover encode of one GOP — a bare [`Encoder`]
/// (`dual_recon=false`) with NO `StegoSession`, NO
/// `phasm_reset_encoder_session_state`, NO `set_frame_num` / `set_pass_mode`.
/// The encode loop is byte-for-byte the same as [`oh264_plain_encode_gop`]'s
/// (same `Encoder` params, frame slicing, and `frame*33` timestamp) — the only
/// differences are the omitted global touches, which are no-ops on the clean
/// path: the sequential capacity probe (`h264_gop_capacity`) is its only caller
/// and runs with no stego `StegoSession` registered, so the fork's MB hooks read
/// the (null) process-global callbacks and no-op. Gated byte-identical to
/// `oh264_plain_encode_gop` by `blite1_clean_encode_matches_plain`.
#[cfg(feature = "h264-encoder")]
pub(crate) fn clean_encode_gop(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<Vec<u8>, StegoError> {
    validate_dims(gop_yuv, width, height, n_frames)?;
    let mut encoder = Encoder::new_with_dual_recon(
        width as i32,
        height as i32,
        opts.qp,
        opts.intra_period,
        /* dual_recon = */ false,
    )
    .map_err(encoder_err_to_stego)?;

    let frame_y = (width * height) as usize;
    let frame_uv = (width * height / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;

    let mut out = vec![0u8; 4 * 1024 * 1024];
    let mut bitstream = Vec::with_capacity(2 * 1024 * 1024);
    for frame in 0..n_frames {
        let base = (frame as usize) * frame_total;
        let y = &gop_yuv[base..base + frame_y];
        let u = &gop_yuv[base + frame_y..base + frame_y + frame_uv];
        let v = &gop_yuv[base + frame_y + frame_uv..base + frame_total];
        let (_, n) = encoder
            .encode_frame(y, u, v, (frame as i64) * 33, &mut out)
            .map_err(encoder_err_to_stego)?;
        bitstream.extend_from_slice(&out[..n]);
    }
    Ok(bitstream)
}

fn encode_once(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    override_map: Arc<Mutex<HashMap<u64, u8>>>,
    mb_type_table: Arc<Mutex<Vec<u8>>>,
    applied: Arc<Mutex<u32>>,
    mb_width: u32,
    mb_per_frame: usize,
    // C.9.0 (#482): false on the Pass-1 cover probe (no overrides will
    // fire; pVisualRecPic isn't needed because the bitstream is walked
    // then discarded). true on Pass-2 stego where C.8 cascade-break needs
    // the mirror to keep pDecPic clean. Bitstream is byte-identical
    // either way — pVisualRecPic is encoder-internal.
    dual_recon: bool,
    // #533.4.8: Pass-2 replay wiring. `pass_mode` selects encoder
    // behaviour: Passthrough (default; encoder runs RDO/ME normally),
    // Capture (encoder streams per-MB mode decisions into `decision_cache`),
    // Replay (encoder fetches decisions from `decision_cache`; cache
    // miss falls back to RDO/ME). When `decision_cache` is `Some` and
    // `pass_mode` is Capture, the cache is cleared and populated; when
    // Replay, it's read-only.
    pass_mode: PassMode,
    decision_cache: Option<Arc<Mutex<DecisionCache>>>,
    // P3.3a: post-frame DPB correction cover. When `Some`, after each
    // frame's encode the function applies IDCT deltas for all flipped
    // CSB/CSL coefficients to pDecPic's Y plane so the next frame's
    // ME reads post-flip reconstruction. Pass `None` for Pass-1
    // (clean encode) and diagnostic re-encodes.
    dpb_cover: Option<&DomainCover>,
    // P3.3b.4: captured coefficient buffer from Pass 1. When `Some`,
    // enables coefficient replay mode: for each frame, builds a flat
    // buffer of n_mbs * 256 i16 values with STC flips applied, sets
    // the replay pointer, and lets WelsEncInterY auto-advance through
    // the buffer. The override_map is consulted for CSB sign flips and
    // CSL magnitude flips. Pass `None` for Pass-1, capacity probes,
    // and diagnostic re-encodes.
    captured_coeffs: Option<&[(u32, u16, u16, [i16; 256])]>,
    // WV.6.b.2: global frame-index offset applied to `set_frame_num`. Pass `0`
    // for a whole-clip encode (positions + DecisionCache keyed 0..n_frames-1).
    // For a per-GOP-standalone call inside the streaming shadow orchestrator,
    // pass `g * gop_size` so the per-GOP emit keys every cover position AND
    // `DecisionCache` entry with the SAME global frame_idx the union-cover walk
    // assigns (the walker counts VCL NALs monotonically across the
    // concatenation, ignoring SPS/IDR boundaries). Every fork frame_num — coeff
    // hooks, MVD ctx, capture/replay — derives from `PhasmStegoGetFrameNum()`,
    // i.e. this offset, never the encoder's internal counter, so a non-zero
    // base relabels positions without changing the emitted bytes.
    frame_idx_base: u32,
) -> Result<Vec<u8>, StegoError> {
    // PHASM_PROFILE — the OpenH264 C encoder (per-GOP encode_frame loop). Called
    // from every pass: probe / clean cover / Pass-2 stego re-emit / tier / sweeps.
    let _prof = crate::codec::h264::profile::scope("oh264.encode_once");
    // Reset per-encode state.
    {
        let mut t = mb_type_table.lock().expect("mb_type lock");
        for x in t.iter_mut() {
            *x = 0xff;
        }
    }
    *applied.lock().expect("applied lock") = 0;
    if matches!(pass_mode, PassMode::Capture) {
        if let Some(ref cache) = decision_cache {
            cache.lock().expect("cache lock").clear();
        }
    }

    let map_for_hook = override_map.clone();
    let applied_for_hook = applied;
    let mb_type_for_md = mb_type_table;
    // P3.3b: suppress coefficient-domain overrides ONLY when replay is
    // actively applying the flips through the captured-coefficient buffer.
    // When replay is disabled, the emit hook must apply overrides normally
    // — otherwise no flips reach the wire and round-trip decode fails
    // (D2.2 regression: all override entries show fired=0).
    //
    // P3.3b replay is currently disabled (see replay_active = false below),
    // so this must be false. If/when replay is re-enabled, gate on the
    // actual replay_active flag, not just the presence of captured coefficients.
    let suppress_coeff_overrides = false;
    let capture_cache = match pass_mode {
        PassMode::Capture => decision_cache.clone(),
        _ => None,
    };
    let replay_cache = match pass_mode {
        PassMode::Replay => decision_cache.clone(),
        _ => None,
    };
    let handlers = StegoHandlers {
        capture_mb_decision: capture_cache.map(|cache| {
            Box::new(move |d: &MbDecision| {
                if let Ok(mut c) = cache.lock() {
                    c.insert(*d);
                }
            }) as Box<dyn FnMut(&MbDecision) + 'static>
        }),
        replay_mb_decision: replay_cache.map(|cache| {
            Box::new(move |fnum, mx, my| {
                cache.lock().ok().and_then(|c| c.get(fnum, mx, my))
            }) as Box<dyn FnMut(u32, u16, u16) -> Option<MbDecision> + 'static>
        }),
        enc_pre_emit: Some(Box::new(move |pos, _orig| {
            // #549 Phase C — dump every emit-side hook fire when
            // PHASM_549_HOOK_DUMP=1, so the per-MB encoder-internal
            // emission sequence can be diff'd between Pass 1 and Pass 2.
            if std::env::var("PHASM_549_HOOK_DUMP").as_deref() == Ok("1") {
                eprintln!(
                    "[PHASM_549_HOOK pass={} frame={} mb=({},{}) dom={} bc={} sb={} ci={} part={} ref={} mvc={} orig={}]",
                    get_549_pass_idx(),
                    pos.frame_num, pos.mb_x, pos.mb_y,
                    pos.domain, pos.block_cat, pos.sub_block, pos.coeff_idx,
                    pos.partition_idx, pos.ref_idx, pos.mv_component, _orig,
                );
            }
            // P3.3b: suppress coefficient-domain overrides when replay
            // is active — the coefficients are already flipped in the
            // replay buffer. Applying overrides again = double flip.
            // Domain values: 0=CoeffSuffixLsb, 1=CoeffSign (from
            // PhasmStegoDomain enum in wels_stego.h).
            if suppress_coeff_overrides && (pos.domain == 0 || pos.domain == 1) {
                return None;
            }
            let map = map_for_hook.lock().ok()?;
            if map.is_empty() {
                return None;
            }
            let mb_addr = (pos.mb_y as usize) * (mb_width as usize) + (pos.mb_x as usize);
            if mb_addr >= mb_per_frame {
                return None;
            }
            // C.8.13(b) root-cause fix 2026-05-13: always pass
            // `PHASM_MB_TYPE_OTHER` here. The mb_type filter inside
            // `encoder_pos_to_phasm_position_key` would otherwise drop
            // valid overrides because `md_cost` fires AFTER the hook
            // (svc_encode_slice.cpp:2076 vs the hook in WelsEncInterY).
            // For frame N+1, the table still holds frame N's mb_type;
            // if that was I_16x16, `block_cat_matches_mb_type(I_16x16,
            // BC=2)` returns false and silently filters out frame N+1's
            // P-frame Luma 4×4 overrides. The walker's key set already
            // gates overrides to actual wire positions, so the mb_type
            // filter is redundant. See `audit_b_single_flip_probe` +
            // `audit_b_single_flip_probe_filter_bypassed` in
            // `core/tests/openh264_cascade_gap_audit.rs`.
            let key = encoder_pos_to_phasm_position_key(pos, PHASM_MB_TYPE_OTHER, mb_width)?;
            map.get(&key).map(|&t| {
                if let Ok(mut a) = applied_for_hook.lock() {
                    *a += 1;
                }
                t as i32
            })
        })),
        md_cost: Some(Box::new(move |cost| {
            let mb_addr = (cost.mb_y as usize) * (mb_width as usize) + (cost.mb_x as usize);
            if mb_addr < mb_per_frame {
                if let Ok(mut t) = mb_type_for_md.lock() {
                    t[mb_addr] = cost.mb_type;
                }
            }
            // #549 Phase C — env-var-gated per-MB decision dump.
            // Tagged by thread-local pass index so a downstream diff
            // can find the first divergent MB between Pass 1 and Pass 2.
            // No-op when PHASM_549_MD_DUMP isn't "1".
            if std::env::var("PHASM_549_MD_DUMP").as_deref() == Ok("1") {
                eprintln!(
                    "[PHASM_549_MD pass={} frame={} mb=({},{}) type={} cbp={}]",
                    get_549_pass_idx(),
                    cost.frame_num, cost.mb_x, cost.mb_y, cost.mb_type, cost.cbp,
                );
            }
        })),
        ..Default::default()
    };
    let mut encoder =
        Encoder::new_with_dual_recon(width as i32, height as i32, opts.qp, opts.intra_period, dual_recon)
            .map_err(encoder_err_to_stego)?;

    // Register the stego handlers process-globally for this encode. No
    // per-instance `PhasmStegoState` is installed, so the fork's MB hooks fall
    // back to the process-global callback table (`has_session_state == 0`): the
    // Pass-1 capture / Pass-2 replay / emit-override dispatch all run through
    // `handlers`. The `StegoSession` RAII guard unregisters on drop, and each
    // `encode_once` scopes its own session — correct for the sequential encode
    // path (producer Pass-1, then consumer Pass-2, never concurrent).
    let _session = StegoSession::register(handlers)
        .map_err(|e| StegoError::InvalidVideo(format!("openh264 session: {e}")))?;

    let frame_y = (width * height) as usize;
    let frame_uv = (width * height / 4) as usize;
    let frame_total = frame_y + 2 * frame_uv;

    // P3.3b replay disabled for v0.4.G ship. Works at 2-4 kB but hits
    // a structural prediction-divergence wall at 8+ kB. P3.3a (post-
    // frame DPB correction) handles inter-frame cascade at all payloads.
    // The proper intra-frame cascade fix is D.2 (per-row windowed STC,
    // single-pass) — see docs/design/video/h264/d2-per-row-windowed-stc.md.
    let replay_active = false;

    let mut out = vec![0u8; 4 * 1024 * 1024];
    let mut bitstream = Vec::with_capacity(2 * 1024 * 1024);
    set_pass_mode(pass_mode);
    // P3.3b.4: reusable per-frame flat buffer for replay coefficients.
    // Allocated once outside the loop, resized per frame if needed.
    let mut replay_buf: Vec<i16> = if replay_active {
        vec![0i16; mb_per_frame * 256]
    } else {
        Vec::new()
    };
    let mut frame_err: Option<StegoError> = None;
    for frame in 0..n_frames {
        // WV.6.b.2: global frame_idx for the stego hooks (`frame_idx_base` is
        // `g*gop_size` for a per-GOP-standalone shadow call, `0` otherwise).
        // The encoder's INTERNAL frame_num / POC / `encode_frame` timestamp
        // stay GOP-local below — only the phasm position/cache keys go global.
        set_frame_num(frame_idx_base + frame);
        let base = (frame as usize) * frame_total;
        let y = &yuv[base..base + frame_y];
        let u = &yuv[base + frame_y..base + frame_y + frame_uv];
        let v = &yuv[base + frame_y + frame_uv..base + frame_total];

        // P3.3b.4: build per-frame replay buffer with STC flips applied.
        // For frames that have captured coefficients (inter frames), we
        // populate the flat buffer and set the replay pointer. For frames
        // without captures (IDR/I-frames), we temporarily disable replay
        // so WelsEncInterY runs its normal quantize path.
        if replay_active {
            let cap = captured_coeffs.unwrap();
            let frame_caps: Vec<&(u32, u16, u16, [i16; 256])> = cap.iter()
                .filter(|(f, _, _, _)| *f == frame)
                .collect();
            let trace = std::env::var("P33B_TRACE").as_deref() == Ok("1");
            if frame_caps.is_empty() {
                if trace { eprintln!("[RPL] f={frame} NO CAPS — replay OFF"); }
                unsafe { core_openh264_sys::phasm_set_coeff_replay_mode(0); }
            } else {
                if trace {
                    eprintln!("[RPL] f={frame} caps={} mb_per_frame={mb_per_frame} replay ON",
                        frame_caps.len());
                }
                unsafe { core_openh264_sys::phasm_set_coeff_replay_mode(1); }
                // Zero the buffer so any MB without captured coefficients
                // gets an all-zero replay (safe: MB was likely I-mode).
                for v in replay_buf.iter_mut() { *v = 0; }
                // Place each captured MB's coefficients into the flat buffer
                // at its raster-order slot.
                let mb_height_u = height / 16;
                for &&(_, mx, my, ref coeffs) in frame_caps.iter() {
                    let mb_idx = (my as usize) * (mb_width as usize) + (mx as usize);
                    if mb_idx < mb_per_frame {
                        let dst = &mut replay_buf[mb_idx * 256..(mb_idx + 1) * 256];
                        dst.copy_from_slice(coeffs);
                    }
                }
                // Apply STC flips from override_map to the replay buffer.
                let map = override_map.lock().expect("override map lock for replay");
                if !map.is_empty() {
                    if let Some(cover) = dpb_cover {
                        // CSB sign flips: for each CSB position in this
                        // frame, if override_map says the bit should change,
                        // flip the coefficient's sign in the replay buffer.
                        apply_csb_flips_to_replay(
                            &mut replay_buf, &map, &cover.coeff_sign_bypass,
                            frame, mb_width, mb_per_frame, mb_height_u,
                        );
                        // CSL magnitude flips: for each CSL position in this
                        // frame, if override_map says the bit should change,
                        // adjust the coefficient's magnitude by +/-1.
                        apply_csl_flips_to_replay(
                            &mut replay_buf, &map, &cover.coeff_suffix_lsb,
                            frame, mb_width, mb_per_frame, mb_height_u,
                        );
                    }
                }
                // Set the replay pointer for this frame. WelsEncInterY
                // auto-advances by 256 after each MB.
                unsafe {
                    core_openh264_sys::phasm_set_replay_coeffs(
                        replay_buf.as_ptr(),
                        (mb_per_frame * 256) as i32,
                    );
                }
            }
        }

        // D2.2 per-row DPB correction DISABLED (2026-05-27): modifying
        // pDecPic between MB rows during encode_frame changes subsequent
        // rows' predictions → cover positions diverge vs Pass 1 → STC
        // syndrome invalid → roundtrip decode fails (FrameCorrupted).
        // The code + infrastructure (RowDpbState, row_complete_callback,
        // dpb_correction::compute_and_apply_deltas_for_row) is retained
        // for a future single-pass architecture that doesn't rely on
        // Pass 1/Pass 2 cover stability.

        match encoder.encode_frame(y, u, v, (frame as i64) * 33, &mut out) {
            Ok((_, n)) => {

                bitstream.extend_from_slice(&out[..n]);
                // P3.3a: post-frame DPB correction (inter-frame).
                if let Some(cover) = dpb_cover {
                    let map = override_map.lock().expect("override map lock for DPB correction");
                    if !map.is_empty() {
                        if let Some((dec_y, dec_stride)) = encoder.get_dec_pic_y_mut() {
                            super::stego::dpb_correction::compute_and_apply_deltas(
                                dec_y,
                                dec_stride,
                                width,
                                height,
                                &map,
                                cover,
                                frame,
                                opts.qp,
                            );
                        }
                    }
                }
            }
            Err(e) => {
                frame_err = Some(encoder_err_to_stego(e));
                break;
            }
        }
    }
    // P3.3b.4: disable replay mode after the frame loop.
    if replay_active {
        unsafe { core_openh264_sys::phasm_set_coeff_replay_mode(0); }
    }
    set_pass_mode(PassMode::Passthrough);
    if let Some(e) = frame_err {
        return Err(e);
    }
    Ok(bitstream)
}

/// P3.3b.4: apply CSB (CoeffSignBypass) sign flips from the override_map
/// to the per-frame replay buffer. For each CSB position in this frame
/// whose PositionKey is in the override_map, flip the coefficient's sign
/// in the flat replay buffer.
///
/// Only handles `SyntaxPath::Luma4x4` positions (BC=2 inter luma +
/// BC=1 I_16x16 luma AC). Other syntax paths (ChromaAc, ChromaDc,
/// LumaDcIntra16x16) are not in the luma replay buffer and continue
/// to be handled by the existing enc_pre_emit hook path.
fn apply_csb_flips_to_replay(
    buf: &mut [i16],
    map: &HashMap<u64, u8>,
    csb: &super::stego::inject::DomainBits,
    frame: u32,
    _mb_width: u32,
    mb_per_frame: usize,
    _mb_height: u32,
) {
    for pos in csb.positions.iter() {
        if pos.frame_idx() != frame { continue; }
        if !map.contains_key(&pos.raw()) { continue; }
        // Only handle Luma4x4 positions in the replay buffer.
        if let SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind } = pos.syntax_path() {
            if kind != BinKind::Sign { continue; }
            let mb_addr = pos.mb_addr() as usize;
            if mb_addr >= mb_per_frame { continue; }
            // coeff_idx is in scan order; convert to raster for the
            // flat coefficient array.
            let raster_ci = ZZ_SCAN_4X4[(coeff_idx & 0xF) as usize] as usize;
            let flat_idx = mb_addr * 256 + (block_idx as usize) * 16 + raster_ci;
            if flat_idx < buf.len() && buf[flat_idx] != 0 {
                buf[flat_idx] = -buf[flat_idx];
            }
        }
    }
}

/// P3.3b.4: apply CSL (CoeffSuffixLsb) magnitude flips from the
/// override_map to the per-frame replay buffer. For each CSL position
/// in this frame whose PositionKey is in the override_map, adjust the
/// coefficient's absolute value by +/-1 (preserving sign), using the
/// same `flipped_magnitude` logic as `inject.rs`.
///
/// CSL threshold = 16: only coefficients with |coeff| >= 16 participate.
/// flipped_magnitude(abs, 16) = abs+1 if abs==16, else abs-1.
fn apply_csl_flips_to_replay(
    buf: &mut [i16],
    map: &HashMap<u64, u8>,
    csl: &super::stego::inject::DomainBits,
    frame: u32,
    _mb_width: u32,
    mb_per_frame: usize,
    _mb_height: u32,
) {
    const THRESHOLD: i16 = 16;
    for pos in csl.positions.iter() {
        if pos.frame_idx() != frame { continue; }
        if !map.contains_key(&pos.raw()) { continue; }
        if let SyntaxPath::Luma4x4 { block_idx, coeff_idx, kind } = pos.syntax_path() {
            if kind != BinKind::SuffixLsb { continue; }
            let mb_addr = pos.mb_addr() as usize;
            if mb_addr >= mb_per_frame { continue; }
            let raster_ci = ZZ_SCAN_4X4[(coeff_idx & 0xF) as usize] as usize;
            let flat_idx = mb_addr * 256 + (block_idx as usize) * 16 + raster_ci;
            if flat_idx >= buf.len() { continue; }
            let coeff = buf[flat_idx];
            let abs_val = coeff.unsigned_abs() as i16;
            if abs_val < THRESHOLD { continue; }
            // flipped_magnitude: at threshold boundary go up, else go down.
            let new_abs = if abs_val == THRESHOLD { abs_val + 1 } else { abs_val - 1 };
            buf[flat_idx] = if coeff < 0 { -new_abs } else { new_abs };
        }
    }
}

fn encoder_err_to_stego(e: EncoderError) -> StegoError {
    StegoError::InvalidVideo(format!("openh264 encoder: {e}"))
}

fn bytes_to_bits_msb_first(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in 0..8 {
            out.push((b >> (7 - i)) & 1);
        }
    }
    out
}

/// WV.6.a — embed the primary as per-GOP `chunk_frame` v3 over the union
/// 4-domain cover, replacing the single whole-video `frame_bits` STC.
///
/// The combined cover (`combine_cover_4domain` output, canonical
/// CS→CSL→MVDs→MVDsl) is partitioned by `PositionKey.frame_idx() /
/// gop_size` — the SAME GOP segmentation the decoder reproduces by
/// splitting the Annex-B at SPS boundaries (`split_annex_b_into_gops`)
/// and re-combining each slab (`h264_gop_cover_bits`). A `chunk_frame`
/// STC runs per GOP (GOP 0 carries the `total_bytes` first chunk; later
/// GOPs carry subsequent chunks until the payload drains), so each GOP
/// slab decodes via the fast `try_primary_decode` path instead of the
/// whole-video `m_total` brute-force.
///
/// Returns the full-length stego-bit vector: primary flips applied at the
/// per-GOP STC's chosen positions, every other position (including any
/// shadow LSBs already injected into `combined_cover`) preserved
/// byte-for-byte. The caller maps it back with `split_plan_4domain`. The
/// shadow's ∞-cost overlay (in `combined_costs`) keeps the per-GOP STC
/// off shadow positions exactly as the whole-video STC did.
///
/// Allocation is greedy uniform-with-carry, shrunk on STC-infeasibility
/// (the no-probe fallback `drain_one_gop` uses). `MessageTooLarge` if the
/// payload can't drain into the available per-GOP covers.
#[allow(clippy::too_many_arguments)]
fn embed_primary_per_gop_4domain(
    cover: &DomainCover,
    // §6.cover: taken BY VALUE and mutated in place — it IS the returned
    // full-length stego buffer (the per-GOP STC overwrites only its GOP's
    // positions; everything else, including injected shadow LSBs, is preserved).
    // Saves a whole union-sized `Vec<u8>` copy per call vs the old
    // `full_stego = combined_cover.to_vec()`.
    mut combined_cover: Vec<u8>,
    combined_costs: &[f32],
    gop_size: u32,
    n_gops: u32,
    frame_bytes: &[u8],
    hhat_seed: &[u8; 32],
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::chunk_frame::{build_chunk_frame_v3_1, build_first_chunk_frame_v3_1};

    let n_cs = cover.coeff_sign_bypass.len();
    let n_csl = cover.coeff_suffix_lsb.len();
    let n_mvds = cover.mvd_sign_bypass.len();
    let off_csl = n_cs;
    let off_mvds = n_cs + n_csl;
    let off_mvdsl = n_cs + n_csl + n_mvds;
    debug_assert_eq!(
        combined_cover.len(),
        off_mvdsl + cover.mvd_suffix_lsb.len(),
        "combined cover length must equal Σ domain lengths"
    );
    debug_assert_eq!(combined_cover.len(), combined_costs.len());

    let total_bytes = u32::try_from(frame_bytes.len()).map_err(|_| StegoError::MessageTooLarge)?;
    if total_bytes == 0 {
        return Err(StegoError::InvalidVideo("per-GOP primary: empty frame".into()));
    }
    let gop_size = gop_size.max(1);

    // `combined_cover` (owned) IS the working stego buffer — the per-GOP STC
    // overwrites only this GOP's positions with the primary chunk's stego bits;
    // every other position (incl. injected shadow LSBs) stays byte-for-byte.
    let mut full_stego = combined_cover;
    let mut cursor = 0usize; // frame_bytes consumed so far

    for g in 0..n_gops {
        // Gather GOP g's combined-vector indices in canonical
        // CS→CSL→MVDs→MVDsl order — identical to the decoder's per-slab
        // `combine_cover_4domain`, so the STC extract lands on the same
        // cover the encoder embedded into.
        let mut idx: Vec<usize> = Vec::new();
        for (i, p) in cover.coeff_sign_bypass.positions.iter().enumerate() {
            if p.frame_idx() / gop_size == g {
                idx.push(i);
            }
        }
        for (i, p) in cover.coeff_suffix_lsb.positions.iter().enumerate() {
            if p.frame_idx() / gop_size == g {
                idx.push(off_csl + i);
            }
        }
        for (i, p) in cover.mvd_sign_bypass.positions.iter().enumerate() {
            if p.frame_idx() / gop_size == g {
                idx.push(off_mvds + i);
            }
        }
        for (i, p) in cover.mvd_suffix_lsb.positions.iter().enumerate() {
            if p.frame_idx() / gop_size == g {
                idx.push(off_mvdsl + i);
            }
        }

        // GOP 0 is ALWAYS a stego GOP (its first chunk carries total_bytes).
        // Later GOPs stop carrying primary once the payload is fully embedded
        // — they become a plain tail the decoder never STC-extracts (it stops
        // at `accumulated == total_bytes`).
        let is_first = g == 0;
        if !is_first && cursor >= frame_bytes.len() {
            continue;
        }

        // Read from `full_stego` (== the owned `combined_cover`): GOP g's
        // positions are gathered before any write, and GOPs partition the
        // positions disjointly, so a later GOP never reads a position an
        // earlier GOP already overwrote.
        let gop_cover: Vec<u8> = idx.iter().map(|&k| full_stego[k]).collect();
        let gop_costs: Vec<f32> = idx.iter().map(|&k| combined_costs[k]).collect();
        let n_cover = gop_cover.len();

        // Greedy uniform-with-carry: this GOP targets its even share of the
        // remaining bytes; a GOP that can't hold its share embeds less and the
        // rest rolls forward.
        let gops_remaining = (n_gops - g) as usize; // ≥ 1, includes this GOP
        let remaining = frame_bytes.len() - cursor;
        let mut want = if gops_remaining <= 1 {
            remaining
        } else {
            remaining.div_ceil(gops_remaining)
        };
        loop {
            let chunk = &frame_bytes[cursor..cursor + want];
            let framed = if is_first {
                build_first_chunk_frame_v3_1(total_bytes, chunk)?
            } else {
                build_chunk_frame_v3_1(chunk)?
            };
            let frame_bits = bytes_to_bits_msb_first(&framed);
            let m = frame_bits.len();
            let w = if m == 0 { 0 } else { n_cover / m };
            if w >= 1 {
                let used = m * w;
                let hhat = generate_hhat(STC_H, w, hhat_seed);
                if let Some(plan) =
                    stc_embed(&gop_cover[..used], &gop_costs[..used], &frame_bits, &hhat, STC_H, w)
                {
                    for (k, &gi) in idx.iter().enumerate() {
                        full_stego[gi] = if k < used { plan.stego_bits[k] } else { gop_cover[k] };
                    }
                    cursor += want;
                    break;
                }
            }
            // Infeasible at `want`. want == 0 means even the header won't fit
            // this GOP's cover — carry everything forward (GOP 0 failing here
            // surfaces as MessageTooLarge via the drain check below).
            if want == 0 {
                break;
            }
            want -= 1usize.max(want / 8); // ~12% step, always reaches 0
        }
    }

    if cursor < frame_bytes.len() {
        return Err(StegoError::MessageTooLarge);
    }
    Ok(full_stego)
}

/// WV.6.b.2 — run one `encode_once` pass of the streaming shadow
/// orchestrator as **N independent per-GOP encodes**, concatenated in GOP
/// order. Replaces the old whole-clip `encode_once(yuv, n_frames, …)` calls
/// (Pass 1 Capture, provisional Pass 2, cascade Pass 2) so no single encode
/// holds the whole clip's encoder working set — the convergence onto the
/// no-shadow `drain_one_gop` per-GOP pattern.
///
/// Each GOP is a standalone encoder session (fresh `Encoder`, fork statics
/// reset per #548). The per-GOP emit is keyed with **global** `frame_idx`
/// (`frame_idx_base = g * gop_size`) so positions align with the union-cover
/// walk and the global override map (see `encode_once`'s `frame_idx_base` doc +
/// the walker's VCL-NAL frame counting).
///
/// **No DecisionCache (WV.6.g.1b).** A per-GOP-standalone encode is bit-
/// deterministic across two runs (verified on real 1080p,
/// `oh264_g1_clean_reencode_determinism`), so every pass just re-encodes
/// clean (`PassMode::Passthrough`) and the Pass-2 cover reproduces Pass-1's
/// exactly — overrides apply at CABAC emit (wire_only) without any captured
/// decisions. This drops the whole-clip per-GOP-cache memory (~1.6 MB/frame at
/// 1080p, the dominant O(clip) term).
///
/// **Sources each GOP by slicing the whole `yuv`** (the session's in-RAM
/// buffer). WV.6.g will replace this with per-GOP re-decode of the SOURCE
/// video (never a decoded-YUV spill — decoded YUV is ~150× the source).
#[cfg(feature = "h264-encoder")]
#[allow(clippy::too_many_arguments)]
fn shadow_encode_all_gops(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    override_map: &Arc<Mutex<HashMap<u64, u8>>>,
    mb_type_table: &Arc<Mutex<Vec<u8>>>,
    applied: &Arc<Mutex<u32>>,
    mb_width: u32,
    mb_per_frame: usize,
    dual_recon: bool,
    pass_mode: PassMode,
    gop_size: u32,
    n_gops: u32,
    wire_only: bool,
) -> Result<Vec<u8>, StegoError> {
    // Mirror encode_once's own tight-I420 frame stride (frame_y + 2·frame_uv).
    let frame_y = (width * height) as usize;
    let frame_total = frame_y + 2 * (frame_y / 4);
    let mut out: Vec<u8> = Vec::new();
    for g in 0..n_gops {
        let start = g * gop_size;
        if start >= n_frames {
            break;
        }
        let frames_in_gop = (n_frames - start).min(gop_size);
        let byte_start = start as usize * frame_total;
        let byte_end = (start + frames_in_gop) as usize * frame_total;
        if byte_end > yuv.len() {
            return Err(StegoError::InvalidVideo(format!(
                "shadow per-GOP source: GOP {g} byte range {byte_start}..{byte_end} \
                 exceeds yuv len {}",
                yuv.len()
            )));
        }
        let gop_yuv = &yuv[byte_start..byte_end];

        // #548 — reset fork statics before EVERY standalone GOP session. The
        // per-GOP shadow stream is N sequential encodes (the exact #548
        // reproducer), so the bypass scratch + last-MB sentinels + wire-only
        // flag must not carry across. Re-arm wire_only after the reset (it
        // clears `g_phasm_use_wire_only_overrides`).
        unsafe { core_openh264_sys::phasm_reset_encoder_session_state() };
        if wire_only {
            unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
        }

        let gop_bytes = encode_once(
            gop_yuv, width, height, frames_in_gop, opts,
            override_map.clone(), mb_type_table.clone(), applied.clone(),
            mb_width, mb_per_frame,
            dual_recon,
            pass_mode,
            None, // WV.6.g.1b: no DecisionCache — Pass 2 re-encodes clean
            None, // P3.3a: per-GOP shadow stream — no DPB correction
            None, // P3.3b.4: no coefficient replay on shadow path
            /* frame_idx_base = */ start,
        )?;
        out.extend_from_slice(&gop_bytes);
    }
    Ok(out)
}

/// STEGO.A.6 — n-shadow video stego on the OH264 backend (Scheme A
/// combined STC + Tier 3 content-adaptive costs + cascade-safe
/// MvdSuffixLsb gating + shadow ∞-cost overlay).
///
/// Multi-message video stego with plausible deniability. The primary
/// message is carried by Scheme A combined STC over the 4-domain
/// cover with content-adaptive costs (STEGO.A.1). Each shadow
/// message is carried by direct LSB writes + Reed-Solomon parity at
/// one of the [`SHADOW_PARITY_TIERS`] rungs, with positions chosen
/// by passphrase-derived priority over the primary-emit cover.
///
/// Round-trips through `StreamingDecodeSession` (the unified
/// Scheme A decoder from STEGO.A.4): shadow_extract runs
/// first (cheap, AES-GCM-SIV auth-gated), Scheme A combined STC
/// extract handles the primary.
///
/// Wire format: per-GOP `chunk_frame` v3 primary (WV.6.a — same as the
/// no-shadow path) plus shadow LSB injections at the cascade-safe shadow
/// positions. No new wire-format flags.
///
/// Architecture (per `docs/design/video/h264/shadow2-oh264-port.md`):
///
/// 1. **Pass 1** — clean OH264 encode + walker → `cover_p1` (4-domain
///    canonical cover, decision cache populated).
/// 2. **Provisional Pass 2** — primary-only STC plan + wire_only emit
///    + walk → `primary_emit_cover` and the cascade-safety mask
///    `safe_msl_prov` derived from the provisional MVD meta.
/// 3. **Cascade loop** over [`SHADOW_PARITY_TIERS`]: at each rung,
///    select shadow positions over the emit cover with the
///    cascade-safety filter, translate them to `cover_p1` indexing,
///    inject shadow bits + ∞-cost into the packed STC input, plan
///    primary STC, defensive-stamp shadow bits onto the plan, build a
///    cascade-safe override map (MvdSuffixLsb gated by safety mask),
///    re-emit Pass 2, walk + verify per-shadow extract. First rung
///    where every shadow extracts → return.
///
/// Returns `ShadowEmbedFailed` if all 6 parity rungs exhaust.
///
/// Reuses encoder-agnostic primitives from
/// [`super::stego::shadow`] and [`super::stego::cascade_safety`].
/// No Pass 1B equivalent: OH264 wire_only keeps encoder state clean
/// (residual emitted on the wire reflects the encoder's natural MVs,
/// not the overridden ones), so the residual cover after Pass 2 is
/// the residual cover the decoder will see.
///
/// **Empty shadows shortcut**: when `shadows.is_empty()`, falls
/// through to [`h264_encode_gop_framed_bits_auto`] — no
/// provisional pass, no cascade loop, single STC + emit.
#[allow(clippy::too_many_arguments)]
pub fn h264_encode_with_shadows<'a>(
    // WV.6.g: the cover YUV is currently the whole-clip buffer the session
    // holds (the streaming-invariant gap on the shadow path) — to be replaced
    // by per-GOP re-decode of the SOURCE video (never spilled; decoded YUV is
    // ~150× the source). `shadow_encode_all_gops` slices it per GOP for now.
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    primary_message: &str,
    primary_files: &[crate::stego::payload::FileEntry],
    primary_passphrase: &str,
    shadows: &'a [crate::stego::shadow_layer::ShadowLayer<'a>],
    weights: &CostWeights,
) -> Result<Vec<u8>, StegoError> {
    use super::stego::shadow::{
        apply_shadow_to_plan, embed_shadow_lsb,
        overlay_infinity_costs, prepare_shadow_over_emit_cover,
        shadow_extract, translate_shadow_state, ShadowState,
    };
    use super::stego::cascade_safety::{
        analyze_safe_mvd_subset, derive_msl_safe_from_msb,
    };
    use crate::stego::shadow_layer::SHADOW_PARITY_TIERS;
    use super::stego::gop_pattern::GopPattern;

    // ── Phase 1: validate + prep primary payload ─────────────────
    // Share the validation + payload-prep with the pure-Rust shadow
    // pipeline (P0.1.f.1 + P0.1.f.2 helpers). Synthesize a GopPattern
    // matching the OH264 default (IBPBP, b_count=1 since §Default-
    // flip.IBPBP) for validate_n_shadow_inputs's pattern check.
    let pattern = GopPattern::Ibpbp {
        gop: opts.intra_period as usize,
        b_count: 1,
    };
    // `validate_n_shadow_inputs` only reads the byte length (its `yuv_len`
    // param) — pass the buffer's length.
    let _setup = super::stego::oh264_capacity::validate_n_shadow_inputs(
        yuv.len(),
        width, height, n_frames as usize, pattern,
        primary_passphrase, shadows,
    )?;
    let primary = super::stego::oh264_capacity::prep_n_shadow_primary_payload(
        primary_message, primary_files, primary_passphrase,
    )?;

    let primary_hhat_seed = primary.keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    // ── Empty-shadows shortcut: route to the primary-only orchestrator.
    // Dead in production — `create_with_shadows` routes 0-shadow clips to the
    // per-GOP `Oh264` path, so this fn is only reached with ≥1 shadow.
    if shadows.is_empty() {
        return h264_encode_gop_framed_bits_auto(
            yuv, width, height, n_frames, opts,
            &primary.frame_bits, &primary_hhat_seed, weights,
        );
    }

    // ── Phase 2: Pass 1 — clean OH264 encode + walker capture ────
    // WV.6.b.2: Pass 1 is now N standalone per-GOP encodes concatenated in GOP
    // order (`shadow_encode_all_gops`), not one whole-clip `encode_once`. The
    // fork-state reset is per-GOP inside the helper (#548).
    let mb_width = width / 16;
    let mb_height = height / 16;
    let mb_per_frame = (mb_width * mb_height) as usize;
    let _ = mb_height;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // WV.6.a/b — the primary is embedded per-GOP `chunk_frame` (one chunk per
    // GOP slab) so each slab decodes via the fast `try_primary_decode`. GOP g
    // owns the cover positions whose `frame_idx / gop_size == g`. The whole
    // encode is N standalone per-GOP sessions (WV.6.b); the union cover is
    // `walk(concatenation)`, whose VCL-NAL frame counting yields global
    // `frame_idx` 0..N-1 — identical to a single-session walk.
    let gop_size = (opts.intra_period.max(1)) as u32;
    let n_gops = n_frames.div_ceil(gop_size).max(1);

    let wire_only = std::env::var("PHASM_USE_WIRE_ONLY")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true);
    // WV.6.g.1b — NO DecisionCache. A per-GOP-standalone encode is bit-
    // deterministic across runs (verified, `oh264_g1_clean_reencode_determinism`),
    // so every pass re-encodes clean (`Passthrough`) and the Pass-2 cover
    // reproduces Pass-1's exactly; overrides apply at CABAC emit (wire_only)
    // with no captured decisions. Drops the whole-clip per-GOP cache (the
    // dominant O(clip) memory term, ~1.6 MB/frame at 1080p).

    set_549_pass_idx(1);
    let baseline_bitstream = shadow_encode_all_gops(
        yuv, width, height, n_frames, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        gop_size, n_gops, wire_only,
    )?;

    let baseline_walk = walk_annex_b_for_cover_with_options(
        &baseline_bitstream,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("baseline walk: {e}")))?;
    let cover_p1 = &baseline_walk.cover;

    // ── Phase 3: Provisional Pass 2 — primary-only emit + walk ───
    // STEGO.A.3 — content-adaptive costs from Tier 3 wavelet pre-pass.
    let mut content_costs = super::stego::content_costs::compute_content_costs_yuv(
        yuv, width, height, n_frames, cover_p1, opts.qp,
    )?;
    // STEGO.A.10 fix — apply cascade-safety MSL gate as ∞-cost in the
    // STC input (not as override-map filter). See the matching block
    // in `h264_encode_gop_framed_bits_auto` for rationale.
    let safe_msb_p1 = super::stego::cascade_safety::analyze_safe_mvd_subset(
        &baseline_walk.mvd_meta, baseline_walk.mb_w, baseline_walk.mb_h,
    );
    let safe_msl_baseline = super::stego::cascade_safety::derive_msl_safe_from_msb(
        &cover_p1.mvd_sign_bypass.positions,
        &safe_msb_p1,
        &cover_p1.mvd_suffix_lsb.positions,
    );
    {
        let n = content_costs.mvd_suffix_lsb.len();
        let safe_len = safe_msl_baseline.len().min(n);
        for i in 0..safe_len {
            if !safe_msl_baseline[i] {
                content_costs.mvd_suffix_lsb[i] = f32::INFINITY;
            }
        }
        for i in safe_len..n {
            content_costs.mvd_suffix_lsb[i] = f32::INFINITY;
        }
    }

    // D'.8 (#793) — Track 1 cascade-safety tier filter on shadow primary
    // STC plan. The shadow orchestrator has its own STC plan path
    // (provisional + final) separate from the no-shadow encoder; tier
    // filter must apply here too. Auto-tier picks against the primary
    // payload size (shadow capacity ladder is independent — shadows use
    // their own priority sort on a different cost vector).
    //
    // Pure-Rust shadow orchestrator (`encode_pixels.rs` ~ line 1880+)
    // has a sister gate added in parallel.
    {
        use super::stego::tier_filter::{
            apply_tier_filter, auto_select_tier, CascadeTier,
        };
        let qp_value = opts.qp;
        let msg_bytes = primary.frame_bits.len() / 8;
        let csb_qp_slice: Vec<i32> = vec![qp_value; cover_p1.coeff_sign_bypass.len()];
        let csl_qp_slice: Vec<i32> = vec![qp_value; cover_p1.coeff_suffix_lsb.len()];
        let resolved_tier = match std::env::var("PHASM_TIER_OVERRIDE").ok().as_deref() {
            Some(s) if s != "auto" => s.parse::<u8>().ok()
                .and_then(CascadeTier::from_u8)
                .unwrap_or(CascadeTier::Tier0),
            _ => auto_select_tier(
                cover_p1, &csb_qp_slice, &csl_qp_slice, msg_bytes,
                super::stego::tier_filter::DEFAULT_HEADROOM,
            ),
        };
        if std::env::var("PHASM_TIER_DEBUG").is_ok() {
            eprintln!(
                "[tier_filter/oh264-shadow] msg_bytes={msg_bytes} resolved_tier={} csb={} csl={}",
                resolved_tier.as_u8(),
                cover_p1.coeff_sign_bypass.len(),
                cover_p1.coeff_suffix_lsb.len(),
            );
        }
        let tier_idx = resolved_tier.as_u8();
        if tier_idx > 0 {
            let csb_keep = apply_tier_filter(
                &cover_p1.coeff_sign_bypass, &csb_qp_slice, tier_idx,
            );
            let csl_keep = apply_tier_filter(
                &cover_p1.coeff_suffix_lsb, &csl_qp_slice, tier_idx,
            );
            for (i, &keep) in csb_keep.iter().enumerate() {
                if !keep && i < content_costs.coeff_sign_bypass.len() {
                    content_costs.coeff_sign_bypass[i] = f32::INFINITY;
                }
            }
            for (i, &keep) in csl_keep.iter().enumerate() {
                if !keep && i < content_costs.coeff_suffix_lsb.len() {
                    content_costs.coeff_suffix_lsb[i] = f32::INFINITY;
                }
            }
        }
    }

    // (`gop_size` / `n_gops` computed once at the top of Phase 2.)

    // Build provisional plan from cover_p1 (no shadow injection). The
    // provisional emit must lay down the SAME per-GOP primary the final pass
    // does, so the shadow is placed over the real primary-emit cover.
    let provisional_plan = {
        let (combined_cover, combined_costs, boundaries) =
            combine_cover_4domain(cover_p1, &content_costs, weights);
        if combined_cover.is_empty() {
            return Err(StegoError::InvalidVideo(
                "shadow encode: combined cover empty".into(),
            ));
        }
        let full_stego_bits = embed_primary_per_gop_4domain(
            cover_p1, combined_cover, &combined_costs,
            gop_size, n_gops, &primary.frame_bytes, &primary_hhat_seed,
        )?;
        split_plan_4domain(&full_stego_bits, &boundaries)
    };

    // Build provisional override map (no msl gate yet — we don't have a
    // safe set; the provisional pass is what produces it).
    {
        let mut overrides_map = override_map.lock().expect("override map lock");
        overrides_map.clear();
        for (positions, bits, cover_bits) in [
            (&cover_p1.coeff_sign_bypass.positions,
             &provisional_plan.coeff_sign_bypass,
             &cover_p1.coeff_sign_bypass.bits),
            (&cover_p1.coeff_suffix_lsb.positions,
             &provisional_plan.coeff_suffix_lsb,
             &cover_p1.coeff_suffix_lsb.bits),
            (&cover_p1.mvd_sign_bypass.positions,
             &provisional_plan.mvd_sign_bypass,
             &cover_p1.mvd_sign_bypass.bits),
            (&cover_p1.mvd_suffix_lsb.positions,
             &provisional_plan.mvd_suffix_lsb,
             &cover_p1.mvd_suffix_lsb.bits),
        ] {
            let n = positions.len().min(bits.len()).min(cover_bits.len());
            for i in 0..n {
                if bits[i] != cover_bits[i] {
                    overrides_map.insert(positions[i].raw(), bits[i]);
                }
            }
        }
    }

    set_549_pass_idx(2);
    let bytes_prov = shadow_encode_all_gops(
        yuv, width, height, n_frames, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ !wire_only,
        PassMode::Passthrough, // WV.6.g.1b: clean re-encode, no cache replay
        gop_size, n_gops, wire_only,
    )?;

    let walk_prov = walk_annex_b_for_cover_with_options(
        &bytes_prov,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("provisional walk: {e}")))?;

    let safe_msb_prov = analyze_safe_mvd_subset(
        &walk_prov.mvd_meta, walk_prov.mb_w, walk_prov.mb_h,
    );
    let safe_msl_prov = derive_msl_safe_from_msb(
        &walk_prov.cover.mvd_sign_bypass.positions,
        &safe_msb_prov,
        &walk_prov.cover.mvd_suffix_lsb.positions,
    );

    // Translate the safe_msl mask from walk_prov.cover.mvd_suffix_lsb
    // indexing to cover_p1.mvd_suffix_lsb indexing via PositionKey.
    // (Used as the override-map MvdSuffixLsb gate.)
    let safe_msl_p1: Vec<bool> = {
        let prov_map: HashMap<u64, usize> = walk_prov.cover.mvd_suffix_lsb.positions
            .iter().enumerate().map(|(i, k)| (k.raw(), i)).collect();
        cover_p1.mvd_suffix_lsb.positions
            .iter()
            .map(|k| {
                prov_map.get(&k.raw())
                    .copied()
                    .and_then(|i| safe_msl_prov.get(i).copied())
                    .unwrap_or(false)
            })
            .collect()
    };

    // ── Phase 4: Cascade loop over SHADOW_PARITY_TIERS ───────────
    for parity_len in SHADOW_PARITY_TIERS {
        // (a) Prepare shadow states over primary_emit_cover.
        let shadow_states_emit: Vec<ShadowState> = match shadows
            .iter()
            .map(|s| prepare_shadow_over_emit_cover(
                &walk_prov.cover, s.passphrase, s.message, s.files,
                parity_len, None, Some(&safe_msl_prov),
            ))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(v) => v,
            Err(StegoError::MessageTooLarge) => continue, // next parity
            Err(e) => return Err(e),
        };

        // (b) Translate to cover_p1 indexing. OH264 single 4-domain
        // cover serves as both mvd_target and coeff_target.
        let shadow_states_p1: Vec<ShadowState> = shadow_states_emit
            .iter()
            .map(|s| translate_shadow_state(s, &walk_prov.cover, cover_p1, cover_p1))
            .collect();

        // (c-d) Build the shadow-embedded cover bits + ∞-cost overlay.
        // §6.cover: clone ONLY the per-domain BITS (1 byte/pos), not the whole
        // `cover_p1` DomainCover (11 bytes/pos). `combine_bits_4domain` reads
        // only bits, and `embed_primary_per_gop_4domain` reads only positions —
        // which are IDENTICAL to `cover_p1`'s, since the shadow embed flips
        // bits alone. So the 10 bytes/pos of positions+magnitudes are shared
        // with `cover_p1` by reference instead of cloned per rung.
        // STEGO.A.3 — costs start from the Tier 3 content-adaptive baseline;
        // shadow ∞-overlay layers on top.
        let mut sb_csb = cover_p1.coeff_sign_bypass.bits.clone();
        let mut sb_csl = cover_p1.coeff_suffix_lsb.bits.clone();
        let mut sb_msb = cover_p1.mvd_sign_bypass.bits.clone();
        let mut sb_msl = cover_p1.mvd_suffix_lsb.bits.clone();
        let mut costs_for_stc = content_costs.clone();
        for state in &shadow_states_p1 {
            embed_shadow_lsb(&mut sb_csb, &mut sb_csl, &mut sb_msb, &mut sb_msl, state);
            overlay_infinity_costs(
                &mut costs_for_stc.coeff_sign_bypass,
                &mut costs_for_stc.coeff_suffix_lsb,
                &mut costs_for_stc.mvd_sign_bypass,
                &mut costs_for_stc.mvd_suffix_lsb,
                state,
            );
        }

        // (e-f) Combine + per-GOP primary STC over packed cover. The shadow
        // LSBs are already injected into the `sb_*` bits and ∞-masked in
        // `costs_for_stc`, so the per-GOP STC steers the primary around them
        // exactly as the old whole-video STC did — the only change is the
        // primary's per-GOP `chunk_frame` framing (WV.6.a).
        let (combined_cover, combined_costs, boundaries) =
            combine_bits_4domain(&sb_csb, &sb_csl, &sb_msb, &sb_msl, &costs_for_stc, weights);
        // The bits are packed into `combined_cover` now — release the per-domain
        // copies before the primary STC builds its own working set.
        drop((sb_csb, sb_csl, sb_msb, sb_msl));
        let full_stego_bits = match embed_primary_per_gop_4domain(
            cover_p1, combined_cover, &combined_costs,
            gop_size, n_gops, &primary.frame_bytes, &primary_hhat_seed,
        ) {
            Ok(b) => b,
            Err(StegoError::MessageTooLarge) => continue, // primary infeasible at this parity; next rung
            Err(e) => return Err(e),
        };
        let mut domain_plan = split_plan_4domain(&full_stego_bits, &boundaries);

        // (g-h) Defensive shadow stamp — re-write shadow bits onto the
        // primary plan in case STC's allocation drifted them.
        for state in &shadow_states_p1 {
            apply_shadow_to_plan(
                &mut domain_plan.coeff_sign_bypass,
                &mut domain_plan.coeff_suffix_lsb,
                &mut domain_plan.mvd_sign_bypass,
                &mut domain_plan.mvd_suffix_lsb,
                state,
            );
        }

        // (i) Build override map. MvdSuffixLsb entries are gated by
        // safe_msl_p1 — cascade-unsafe MvdSuffixLsb positions are
        // never injected on the wire (the same protection the
        // primary-only orchestrator gets implicitly through the
        // cascade-safety pre-filter on the shadow position selector;
        // here we re-apply it as a final-stage guard).
        {
            let mut overrides_map = override_map.lock().expect("override map lock");
            overrides_map.clear();

            // CSB
            let n = cover_p1.coeff_sign_bypass.positions.len()
                .min(domain_plan.coeff_sign_bypass.len())
                .min(cover_p1.coeff_sign_bypass.bits.len());
            for i in 0..n {
                if domain_plan.coeff_sign_bypass[i] != cover_p1.coeff_sign_bypass.bits[i] {
                    overrides_map.insert(
                        cover_p1.coeff_sign_bypass.positions[i].raw(),
                        domain_plan.coeff_sign_bypass[i],
                    );
                }
            }
            // CSL
            let n = cover_p1.coeff_suffix_lsb.positions.len()
                .min(domain_plan.coeff_suffix_lsb.len())
                .min(cover_p1.coeff_suffix_lsb.bits.len());
            for i in 0..n {
                if domain_plan.coeff_suffix_lsb[i] != cover_p1.coeff_suffix_lsb.bits[i] {
                    overrides_map.insert(
                        cover_p1.coeff_suffix_lsb.positions[i].raw(),
                        domain_plan.coeff_suffix_lsb[i],
                    );
                }
            }
            // MSB
            let n = cover_p1.mvd_sign_bypass.positions.len()
                .min(domain_plan.mvd_sign_bypass.len())
                .min(cover_p1.mvd_sign_bypass.bits.len());
            for i in 0..n {
                if domain_plan.mvd_sign_bypass[i] != cover_p1.mvd_sign_bypass.bits[i] {
                    overrides_map.insert(
                        cover_p1.mvd_sign_bypass.positions[i].raw(),
                        domain_plan.mvd_sign_bypass[i],
                    );
                }
            }
            // MSL — cascade-safety gate
            let n = cover_p1.mvd_suffix_lsb.positions.len()
                .min(domain_plan.mvd_suffix_lsb.len())
                .min(cover_p1.mvd_suffix_lsb.bits.len());
            for i in 0..n {
                if domain_plan.mvd_suffix_lsb[i] != cover_p1.mvd_suffix_lsb.bits[i]
                    && safe_msl_p1.get(i).copied().unwrap_or(false)
                {
                    overrides_map.insert(
                        cover_p1.mvd_suffix_lsb.positions[i].raw(),
                        domain_plan.mvd_suffix_lsb[i],
                    );
                }
            }
        }

        // (j) Pass 2 emit with shadow-aware override map (per-GOP stream).
        let bytes = shadow_encode_all_gops(
            yuv, width, height, n_frames, opts,
            &override_map, &mb_type_table, &applied,
            mb_width, mb_per_frame,
            /* dual_recon = */ !wire_only,
            PassMode::Passthrough, // WV.6.g.1b: clean re-encode, no cache replay
            gop_size, n_gops, wire_only,
        )?;

        // (k-l) Walk emitted bytes + cascade-safety + per-shadow extract.
        let walk_final = match walk_annex_b_for_cover_with_options(
            &bytes,
            WalkOptions { record_mvd: true, ..Default::default() },
        ) {
            Ok(w) => w,
            Err(_) => continue, // walker failure → next rung
        };

        let safe_msb_final = analyze_safe_mvd_subset(
            &walk_final.mvd_meta, walk_final.mb_w, walk_final.mb_h,
        );
        let safe_msl_final = derive_msl_safe_from_msb(
            &walk_final.cover.mvd_sign_bypass.positions,
            &safe_msb_final,
            &walk_final.cover.mvd_suffix_lsb.positions,
        );

        let mut all_ok = true;
        for s in shadows {
            if shadow_extract(
                &walk_final.cover, s.passphrase, None, Some(&safe_msl_final),
            ).is_err() {
                all_ok = false;
                break;
            }
        }

        if all_ok {
            return Ok(bytes);
        }
        // Verify failed at this rung — try next parity.
    }

    // All parity rungs exhausted.
    Err(StegoError::ShadowEmbedFailed)
}

// ─── WV.6.g — streaming shadow encode (O(GOP), not O(clip)) ───────────
//
// The gated parallel build of the truly-streaming shadow path. The source
// abstraction below is the g.3 surface (one shared Rust trait for tests +
// all three bridges); `h264_encode_with_shadows_streaming` is the new
// entry point. Its body is replaced increment by increment (the §9.1
// 2-sweep pipeline), each step held to the byte-identical gate against the
// whole-clip `h264_encode_with_shadows` above. See
// `docs/design/video/h264/oh264-wv6-streaming-shadow-unification.md` §9.

/// Forward-streaming source of decoded GOP YUV for the streaming shadow
/// encode (WV.6.g.3). The encoder pulls one GOP at a time, forward, so
/// nothing wider than a single GOP of decoded YUV is ever live.
///
/// Implementations: [`SliceYuvSource`] (whole-buffer slicer — tests + the
/// not-yet-streamed bridge); the bridge-native AVAssetReader / MediaCodec /
/// CLI demux (lands with the call-site swap).
///
/// **Contract.** `gop_yuv` is called with monotonically increasing
/// `gop_index` within a sweep; `rewind` restarts the forward stream at GOP
/// 0 between the encoder's two sweeps. The source is never seeked to an
/// arbitrary mid-clip GOP — a backward jump is only ever a full rewind to
/// the start, which the native decoders do by re-initialising their reader.
pub trait GopYuvSource {
    /// Tight-I420 YUV for `gop_index`: frames
    /// `[gop_index*gop_size .. min((gop_index+1)*gop_size, n_frames))`,
    /// where `gop_size` / `n_frames` match the encode call. The returned
    /// buffer is `gop_frames * width * height * 3/2` bytes.
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, StegoError>;
    /// Restart the forward stream at GOP 0 (called between encode sweeps).
    fn rewind(&mut self) -> Result<(), StegoError>;
}

/// Whole-buffer [`GopYuvSource`]: slices a contiguous tight-I420 clip into
/// per-GOP YUV. Used by the byte-identical gate and by the bridge until the
/// native forward-stream decoders land. It holds the whole clip, so it does
/// NOT itself bound memory — it is the gated-build stand-in that lets the
/// streaming code path be proven byte-identical before the source is real.
pub struct SliceYuvSource<'a> {
    yuv: &'a [u8],
    width: u32,
    height: u32,
    n_frames: u32,
    gop_size: u32,
}

impl<'a> SliceYuvSource<'a> {
    pub fn new(yuv: &'a [u8], width: u32, height: u32, n_frames: u32, gop_size: u32) -> Self {
        Self { yuv, width, height, n_frames, gop_size: gop_size.max(1) }
    }
}

impl GopYuvSource for SliceYuvSource<'_> {
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, StegoError> {
        let frame_bytes = (self.width * self.height * 3 / 2) as usize;
        let start_f = gop_index.saturating_mul(self.gop_size);
        if start_f >= self.n_frames {
            return Ok(Vec::new());
        }
        let end_f = ((gop_index + 1) * self.gop_size).min(self.n_frames);
        let start = start_f as usize * frame_bytes;
        let end = end_f as usize * frame_bytes;
        self.yuv
            .get(start..end)
            .map(<[u8]>::to_vec)
            .ok_or_else(|| {
                StegoError::InvalidVideo(format!(
                    "SliceYuvSource: GOP {gop_index} range {start}..{end} out of {} bytes",
                    self.yuv.len(),
                ))
            })
    }
    fn rewind(&mut self) -> Result<(), StegoError> {
        Ok(()) // stateless — slicing reads directly from the borrowed buffer
    }
}

/// WV.6.g.progress — fraction (0.0–1.0) progress reporter for the streaming
/// shadow encode. The encode does ~4 forward passes over the GOPs (tier
/// pre-pass + Sweep A + Sweep B emit + verify), so `total` is `4 * n_gops` and
/// each pass `tick()`s once per GOP. The emitted fraction is clamped to 0.99
/// while working (a rare cascade-tier retry over-runs the estimate but must not
/// hit 100% early); only `finish()` emits 1.0, on success — so the bar leaves
/// 0% on the first GOP and reaches 100% only at the very end, with no dead time
/// at either edge. Interior-mutable so one reporter is shared by `&` across the
/// source wrapper (passes 1–3) and the verify pass (pass 4).
struct ShadowProgress<'p> {
    done: std::cell::Cell<usize>,
    total: usize,
    cb: std::cell::RefCell<Option<&'p mut dyn FnMut(f32)>>,
}

impl<'p> ShadowProgress<'p> {
    fn new(total_units: usize, cb: Option<&'p mut dyn FnMut(f32)>) -> Self {
        Self {
            done: std::cell::Cell::new(0),
            total: total_units.max(1),
            cb: std::cell::RefCell::new(cb),
        }
    }
    /// One GOP completed in some pass — advance and emit (clamped below 1.0).
    fn tick(&self) {
        let d = self.done.get() + 1;
        self.done.set(d);
        if let Some(cb) = self.cb.borrow_mut().as_mut() {
            cb((d as f32 / self.total as f32).min(0.99));
        }
    }
    /// The encode succeeded — emit a clean 1.0 (the only path to 100%).
    fn finish(&self) {
        if let Some(cb) = self.cb.borrow_mut().as_mut() {
            cb(1.0);
        }
    }
}

/// [`GopYuvSource`] decorator that `tick()`s a [`ShadowProgress`] each time a
/// GOP's YUV is pulled — transparently meters the three source-pulling passes
/// (tier pre-pass, Sweep A, Sweep B emit) without touching their signatures.
/// The verify pass re-decodes the emitted clip (not the source), so it ticks
/// the same reporter directly.
struct ProgressYuvSource<'s, 'p> {
    inner: &'s mut dyn GopYuvSource,
    prog: &'s ShadowProgress<'p>,
}

impl GopYuvSource for ProgressYuvSource<'_, '_> {
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, StegoError> {
        let r = self.inner.gop_yuv(gop_index)?;
        self.prog.tick();
        Ok(r)
    }
    fn rewind(&mut self) -> Result<(), StegoError> {
        self.inner.rewind()
    }
}

/// WV.6.g.4.2b — resolve the cascade tier by an O(GOP) forward pre-pass
/// (cross-GOP dep 3). `auto_select_tier` → `capacity_at_tier` reads per-position
/// magnitudes, so the tier is whole-clip — but `capacity_at_tier` returns a
/// COUNT, and the per-position magnitude filter is frame-local, so the per-GOP
/// counts sum to the whole-clip count. This clean-encodes each GOP, accumulates
/// `capacity_at_tier(gop_cover, t)` per tier (5 running totals; the cover is
/// dropped), and applies `auto_select_tier`'s decision to the sums. Mirrors
/// [`whole_clip_resolved_tier`] (PHASM_TIER_OVERRIDE / conservative env).
pub fn streaming_tier_prepass(
    source: &mut dyn GopYuvSource,
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    msg_bytes: usize,
) -> Result<u8, StegoError> {
    use super::stego::tier_filter::{capacity_at_tier, CascadeTier, DEFAULT_HEADROOM};

    if let Some(s) = std::env::var("PHASM_TIER_OVERRIDE").ok().filter(|s| s != "auto") {
        return Ok(s
            .parse::<u8>()
            .ok()
            .and_then(CascadeTier::from_u8)
            .unwrap_or(CascadeTier::Tier0)
            .as_u8());
    }
    if std::env::var("PHASM_AUTO_TIER_CONSERVATIVE").map(|v| v == "1").unwrap_or(false) {
        return Ok(0);
    }

    let gop_size = opts.intra_period.max(1) as u32;
    let n_gops = n_frames.div_ceil(gop_size).max(1);
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let wire_only = wire_only_enabled();
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    let mut summed = [0usize; 5]; // capacity_at_tier per tier (0..=4)
    source.rewind()?;
    for g in 0..n_gops {
        let gop_yuv = source.gop_yuv(g)?;
        let fig = (n_frames - g * gop_size).min(gop_size);
        set_549_pass_idx(1);
        override_map.lock().expect("lock").clear();
        let clean = shadow_encode_all_gops(
            &gop_yuv, width, height, fig, opts, &override_map, &mb_type_table, &applied,
            mb_width, mb_per_frame, false, PassMode::Passthrough, fig.max(1), 1, wire_only,
        )?;
        let walk = walk_annex_b_for_cover_with_options(
            &clean,
            WalkOptions { record_mvd: true, ..Default::default() },
        )
        .map_err(|e| StegoError::InvalidVideo(format!("tier prepass walk: {e}")))?;
        let cover = walk.cover;
        let csb_qp: Vec<i32> = vec![opts.qp; cover.coeff_sign_bypass.len()];
        let csl_qp: Vec<i32> = vec![opts.qp; cover.coeff_suffix_lsb.len()];
        for (t, slot) in summed.iter_mut().enumerate() {
            *slot += capacity_at_tier(&cover, &csb_qp, &csl_qp, t as u8);
        }
    }

    // `auto_select_tier`'s decision, on the summed per-tier capacities.
    let required_bits = (msg_bytes as f32 * DEFAULT_HEADROOM * 8.0) as usize;
    for t in [4u8, 3, 2, 1, 0] {
        if summed[t as usize] >= required_bits {
            return Ok(t);
        }
    }
    Ok(0)
}

/// WV.6.g.4.4 — a [`GopYuvSource`] backed by caller-supplied closures: the
/// FFI-friendly **pull** source. The bridge's native decoder (iOS AVAssetReader /
/// Android MediaCodec / CLI demux) supplies each GOP's tight-I420 YUV on demand,
/// so the whole decoded clip is never held — only the source video (on disk,
/// ~150× smaller than decoded YUV) + one GOP in flight. `rewind` re-initialises
/// the decoder between the streaming encoder's sweeps (it rewinds 3×: tier
/// pre-pass, Sweep A, each Sweep B tier).
///
/// The bridge FFI wraps its C decode-callback (+ context pointer) into the two
/// closures; pure-Rust callers (CLI, tests) pass Rust closures directly. Holds no
/// buffer of its own — bounded by whatever the closures hold (one GOP).
pub struct CallbackYuvSource<G, R>
where
    G: FnMut(u32) -> Result<Vec<u8>, StegoError>,
    R: FnMut() -> Result<(), StegoError>,
{
    decode_gop: G,
    reinit: R,
}

impl<G, R> CallbackYuvSource<G, R>
where
    G: FnMut(u32) -> Result<Vec<u8>, StegoError>,
    R: FnMut() -> Result<(), StegoError>,
{
    /// `decode_gop(g)` returns GOP `g`'s tight-I420 YUV
    /// (`frames_in_gop * width * height * 3/2` bytes); `reinit()` restarts the
    /// forward stream at GOP 0.
    pub fn new(decode_gop: G, reinit: R) -> Self {
        Self { decode_gop, reinit }
    }
}

impl<G, R> GopYuvSource for CallbackYuvSource<G, R>
where
    G: FnMut(u32) -> Result<Vec<u8>, StegoError>,
    R: FnMut() -> Result<(), StegoError>,
{
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, StegoError> {
        (self.decode_gop)(gop_index)
    }
    fn rewind(&mut self) -> Result<(), StegoError> {
        (self.reinit)()
    }
}

/// WV.6.g — a [`GopYuvSource`] that reads one GOP at a time from a raw tight-I420
/// YUV file on disk (e.g. the CLI's ffmpeg-decoded temp) via absolute seek+read.
/// Working set is O(GOP): the whole clip stays on disk; only one GOP of YUV is in
/// RAM at a time. The CLI uses this so a long clip no longer loads the entire
/// decoded YUV into memory (the CLI-side analogue of the mobile O(clip) OOM).
///
/// The file MUST be exactly `n_frames * width * height * 3/2` bytes of tight-I420
/// (planar Y, then U, then V; no padding), frames in order — the same layout
/// [`SliceYuvSource`] expects over an in-memory slice, so the two are
/// interchangeable (and byte-identical per GOP).
pub struct FileYuvSource {
    file: std::fs::File,
    width: u32,
    height: u32,
    n_frames: u32,
    gop_size: u32,
}

impl FileYuvSource {
    /// Open `path` as the tight-I420 YUV backing store.
    pub fn open(
        path: &std::path::Path,
        width: u32,
        height: u32,
        n_frames: u32,
        gop_size: u32,
    ) -> Result<Self, StegoError> {
        let file = std::fs::File::open(path).map_err(|e| {
            StegoError::InvalidVideo(format!("FileYuvSource: open {}: {e}", path.display()))
        })?;
        Ok(Self { file, width, height, n_frames, gop_size: gop_size.max(1) })
    }
}

impl GopYuvSource for FileYuvSource {
    fn gop_yuv(&mut self, gop_index: u32) -> Result<Vec<u8>, StegoError> {
        use std::io::{Read, Seek, SeekFrom};
        let frame_bytes = (self.width as u64) * (self.height as u64) * 3 / 2;
        let start_f = (gop_index as u64).saturating_mul(self.gop_size as u64);
        if start_f >= self.n_frames as u64 {
            return Ok(Vec::new()); // past the last GOP (matches SliceYuvSource)
        }
        let end_f = (((gop_index as u64) + 1) * self.gop_size as u64).min(self.n_frames as u64);
        let offset = start_f * frame_bytes;
        let len = ((end_f - start_f) * frame_bytes) as usize;
        self.file
            .seek(SeekFrom::Start(offset))
            .map_err(|e| StegoError::InvalidVideo(format!("FileYuvSource: seek {offset}: {e}")))?;
        let mut buf = vec![0u8; len];
        self.file.read_exact(&mut buf).map_err(|e| {
            StegoError::InvalidVideo(format!("FileYuvSource: read {len} @ {offset}: {e}"))
        })?;
        Ok(buf)
    }
    fn rewind(&mut self) -> Result<(), StegoError> {
        use std::io::{Seek, SeekFrom};
        // gop_yuv seeks absolutely, so this is just a defensive reset.
        self.file
            .seek(SeekFrom::Start(0))
            .map(|_| ())
            .map_err(|e| StegoError::InvalidVideo(format!("FileYuvSource: rewind: {e}")))
    }
}

/// Streaming shadow encode entry point (WV.6.g). Same contract + **byte-identical
/// output** as [`h264_encode_with_shadows`], but pulls cover YUV one GOP at a time
/// through `source` (the §9.1 2-sweep pipeline) so the ENCODE working set is
/// O(GOP) + O(N-shadow-positions), not O(clip).
///
/// **g.4.2b-2b (current):** the real 2-sweep pipeline. Prep the primary once,
/// resolve the tier ([`streaming_tier_prepass`]), [`sweep_a`] (select), then the
/// parity-tier cascade: per tier [`sweep_b_emit`] + [`streaming_shadow_verify`]
/// (re-decode mirroring the orchestrator's `shadow_extract`, ONE GOP at a time),
/// returning at the first verified tier. Both the emit and the verify are now
/// O(GOP) — only the emitted clip itself (the output) is O(clip). Empty shadows
/// → delegate (rare; not the
/// streaming target — the bridge routes 0-shadow clips to the per-GOP `Oh264`
/// path).
#[allow(clippy::too_many_arguments)]
pub fn h264_encode_with_shadows_streaming<'a>(
    source: &mut dyn GopYuvSource,
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    primary_message: &str,
    primary_files: &[crate::stego::payload::FileEntry],
    primary_passphrase: &str,
    shadows: &'a [crate::stego::shadow_layer::ShadowLayer<'a>],
    weights: &CostWeights,
    // Optional fraction (0.0–1.0) progress sink, ticked per-GOP across the ~4
    // forward passes (see `ShadowProgress`). `None` = no reporting.
    progress: Option<&mut dyn FnMut(f32)>,
) -> Result<Vec<u8>, StegoError> {
    use super::stego::gop_pattern::GopPattern;
    use super::stego::shadow::build_shadow_rs_frame;
    use crate::stego::shadow_layer::SHADOW_PARITY_TIERS;

    // ── Phase 1: validate (shares the whole-clip validator). ──
    let pattern = GopPattern::Ibpbp { gop: opts.intra_period as usize, b_count: 1 };
    let frame_bytes_per = (width * height * 3 / 2) as usize;
    let expected_len = frame_bytes_per * n_frames as usize;
    let _setup = super::stego::oh264_capacity::validate_n_shadow_inputs(
        expected_len, width, height, n_frames as usize, pattern, primary_passphrase, shadows,
    )?;

    // Empty-shadows: reassemble + delegate (the per-GOP `Oh264` no-shadow path is
    // the bridge's real 0-shadow route; this is a defensive O(clip) fallback).
    if shadows.is_empty() {
        let gop_size = opts.intra_period.max(1) as u32;
        let n_gops = n_frames.div_ceil(gop_size).max(1);
        let mut yuv = Vec::with_capacity(expected_len);
        source.rewind()?;
        for g in 0..n_gops {
            yuv.extend_from_slice(&source.gop_yuv(g)?);
        }
        return h264_encode_with_shadows(
            &yuv, width, height, n_frames, opts,
            primary_message, primary_files, primary_passphrase, shadows, weights,
        );
    }

    let (frame_bytes, hhat_seed) =
        prep_primary_payload(primary_message, primary_files, primary_passphrase)?;

    // WV.6.g.progress — meter the ~4 forward passes (3 source-pulling + verify).
    // Wrapping the source ticks per pulled GOP across tier pre-pass / Sweep A /
    // Sweep B emit; the verify pass ticks the same reporter directly. Total
    // units = 4 * n_gops (the happy single-tier path).
    let prog_gop_size = opts.intra_period.max(1) as u32;
    let prog_n_gops = n_frames.div_ceil(prog_gop_size).max(1) as usize;
    let prog = ShadowProgress::new(prog_n_gops * 4, progress);
    let mut wrapped = ProgressYuvSource { inner: source, prog: &prog };

    // ── Tier (dep 3, O(GOP) pre-pass) + Sweep A. ──
    let _t_tier = std::time::Instant::now();
    let tier_idx =
        streaming_tier_prepass(&mut wrapped, width, height, n_frames, opts, frame_bytes.len())?;
    crate::codec::h264::profile::record("phase.1_tier_prepass", _t_tier.elapsed());
    // Capacity = n_total at the largest parity tier (select once; prefix per tier).
    let capacities: Vec<usize> = shadows
        .iter()
        .map(|s| {
            build_shadow_rs_frame(
                s.passphrase, s.message, s.files, *SHADOW_PARITY_TIERS.last().unwrap(),
            )
            .map(|(bits, _)| bits.len())
        })
        .collect::<Result<_, _>>()?;
    let _t_sa = std::time::Instant::now();
    let sa = sweep_a(
        &mut wrapped, width, height, n_frames, opts, &frame_bytes, &hhat_seed, shadows,
        &capacities, tier_idx, weights,
    )?;
    crate::codec::h264::profile::record("phase.2_sweep_a", _t_sa.elapsed());

    // ── Phase 4: parity-tier cascade (Sweep B emit + streaming verify). ──
    let gop_size = opts.intra_period.max(1) as u32;
    for parity_len in SHADOW_PARITY_TIERS {
        let shadow_rs: Vec<(Vec<u8>, usize)> = shadows
            .iter()
            .map(|s| build_shadow_rs_frame(s.passphrase, s.message, s.files, parity_len))
            .collect::<Result<_, _>>()?;
        // Feasibility: this tier needs `n_total` selected positions per shadow.
        // (The whole-clip path's `MessageTooLarge` continue — eligible < n_total.)
        if shadow_rs
            .iter()
            .enumerate()
            .any(|(s, (bits, _))| sa.selections[s].len() < bits.len())
        {
            continue;
        }

        let _t_sb = std::time::Instant::now();
        let (bytes, gop_lens) = sweep_b_emit(
            &mut wrapped, width, height, n_frames, opts, &frame_bytes, &hhat_seed, &sa, &shadow_rs,
            parity_len, tier_idx, weights,
        )?;
        crate::codec::h264::profile::record("phase.3_sweep_b_emit", _t_sb.elapsed());

        // WV.6.g.5 — streaming verify: re-decode the emitted clip ONE GOP at a
        // time (no O(clip) walk_final), faithfully mirroring the decoder's
        // shadow_extract. `n_totals` is this tier's RS-frame bit count per shadow.
        // Ticks `prog` per GOP (pass 4 of 4) so the bar keeps moving through verify.
        let n_totals_tier: Vec<usize> = shadow_rs.iter().map(|(bits, _)| bits.len()).collect();
        let mut verify_tick = || prog.tick();
        let _t_verify = std::time::Instant::now();
        let verified = streaming_shadow_verify(
            &bytes, &gop_lens, width, height, gop_size, shadows, &n_totals_tier,
            Some(&mut verify_tick),
        )?;
        crate::codec::h264::profile::record("phase.4_shadow_verify", _t_verify.elapsed());
        if verified {
            prog.finish();
            return Ok(bytes);
        }
    }

    Err(StegoError::ShadowEmbedFailed)
}

/// Read the `PHASM_USE_WIRE_ONLY` flag (production default: ON). Shared by the
/// streaming clean-cover helpers below so they encode exactly as the orchestrator.
fn wire_only_enabled() -> bool {
    std::env::var("PHASM_USE_WIRE_ONLY")
        .map(|v| !(v == "0" || v.eq_ignore_ascii_case("false")))
        .unwrap_or(true)
}


/// WV.6.g.4.1 — Sweep A's per-GOP clean-cover primitive. Encode ONE GOP
/// standalone (clean `Passthrough`, #548 fork reset via `shadow_encode_all_gops`),
/// walk it, and relabel the cover to GLOBAL frame_idx (`frame_idx +=
/// gop_global_start`). Reproduces the whole-clip path's GOP-`g` cover slice
/// exactly: the per-GOP-standalone encode is bit-deterministic (g.1) and
/// `frame_idx_base` does not change bytes, so the walked positions / bits /
/// magnitudes match — only the frame_idx label is local, which the remap fixes.
///
/// This is the foundational, byte-identical-critical Sweep A primitive; the
/// streaming selection loop calls it per GOP. `gop_global_start` is
/// `g * gop_size` (the global index of this GOP's first frame).
pub fn gop_clean_cover(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    frames_in_gop: u32,
    gop_global_start: u32,
    opts: EncodeOpts,
) -> Result<DomainCover, StegoError> {
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let wire_only = wire_only_enabled();

    set_549_pass_idx(1);
    // One standalone session: gop_size == frames_in_gop, n_gops == 1, so
    // `shadow_encode_all_gops` resets fork state (#548) and runs one encode_once.
    let bytes = shadow_encode_all_gops(
        gop_yuv, width, height, frames_in_gop, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        /* gop_size = */ frames_in_gop.max(1),
        /* n_gops = */ 1,
        wire_only,
    )?;
    let walk = walk_annex_b_for_cover_with_options(
        &bytes,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("gop_clean_cover walk: {e}")))?;
    let mut cover = walk.cover;
    if gop_global_start > 0 {
        remap_cover_frame_idx(&mut cover, gop_global_start);
    }
    Ok(cover)
}

/// WV.6.g.4.1 — the whole-clip clean cover, produced exactly as the shadow
/// orchestrator's Pass 1: `shadow_encode_all_gops` over all GOPs as standalone
/// per-GOP sessions, then ONE walk of the concatenation (→ global frame_idx).
/// The reference the per-GOP [`gop_clean_cover`] assembly must reproduce.
pub fn whole_clip_baseline_cover(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<DomainCover, StegoError> {
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let wire_only = wire_only_enabled();
    let gop_size = opts.intra_period.max(1) as u32;
    let n_gops = n_frames.div_ceil(gop_size).max(1);

    set_549_pass_idx(1);
    let bytes = shadow_encode_all_gops(
        yuv, width, height, n_frames, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame,
        false, PassMode::Passthrough, gop_size, n_gops, wire_only,
    )?;
    let walk = walk_annex_b_for_cover_with_options(
        &bytes,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("whole_clip_baseline_cover walk: {e}")))?;
    Ok(walk.cover)
}

/// WV.6.g.4.1c — the whole-clip PROVISIONAL walk: the primary-only emit cover
/// over which shadow positions are selected, plus the MVD meta + frame
/// dimensions Sweep A's `safe_msl_prov` derivation needs. Mirrors the shadow
/// orchestrator's Phase 2 (clean baseline encode + walk) + Phase 3
/// (content-adaptive costs, the `safe_msl_baseline` + tier ∞-gates, the per-GOP
/// primary-only provisional emit, and the walk of the result).
///
/// This is the gated reference the per-GOP [`gop_safe_msl_prov`] decomposition
/// is proven against, and the whole-clip building block g.4.2's Sweep A reuses.
/// The cover here is whole-clip (global `frame_idx`); the per-GOP *emit*
/// decomposition (content_costs[g] + cursor-carry primary + whole-clip tier) is
/// g.4.1d. Because the cascade-safety analysis is frame-local (below), the
/// `safe_msl_prov` derivation already decomposes per-GOP over THIS walk.
pub struct ProvisionalWalk {
    /// Clean Pass-1 baseline cover (global `frame_idx`). Kept for the Fact-2
    /// check: under `wire_only` the provisional emit moves no position, so its
    /// position set equals the clean cover's.
    pub clean_cover: DomainCover,
    /// Provisional primary-emit cover — the cover shadow selection runs over.
    pub prov_cover: DomainCover,
    /// Provisional walk MVD meta, parallel to `prov_cover.mvd_sign_bypass`
    /// (one entry per logged MVD-sign bypass bin, same order).
    pub(crate) prov_mvd_meta: Vec<super::stego::hook::MvdPositionMeta>,
    pub(crate) mb_w: u32,
    pub(crate) mb_h: u32,
}

/// ∞-gate `content_costs.mvd_suffix_lsb` at cascade-UNSAFE positions (STEGO.A.10:
/// applied as STC ∞-cost, not an override filter). Shared by the whole-clip
/// reference + the per-GOP `gop_provisional_step` so both gate identically.
fn apply_safe_msl_gate(
    content_costs: &mut super::stego::orchestrate::DomainCosts,
    safe_msl: &[bool],
) {
    let n = content_costs.mvd_suffix_lsb.len();
    let safe_len = safe_msl.len().min(n);
    for i in 0..safe_len {
        if !safe_msl[i] {
            content_costs.mvd_suffix_lsb[i] = f32::INFINITY;
        }
    }
    for i in safe_len..n {
        content_costs.mvd_suffix_lsb[i] = f32::INFINITY;
    }
}

/// Apply the resolved cascade tier as ∞-cost on CSB/CSL (D'.8). Per-position, so
/// GOP-local-safe — the whole-clip reference and `gop_provisional_step` call it
/// with the SAME `tier_idx` (resolved whole-clip via [`whole_clip_resolved_tier`]).
fn apply_tier_gate(
    cover: &DomainCover,
    content_costs: &mut super::stego::orchestrate::DomainCosts,
    qp: i32,
    tier_idx: u8,
) {
    if tier_idx == 0 {
        return;
    }
    use super::stego::tier_filter::apply_tier_filter;
    let csb_qp: Vec<i32> = vec![qp; cover.coeff_sign_bypass.len()];
    let csl_qp: Vec<i32> = vec![qp; cover.coeff_suffix_lsb.len()];
    let csb_keep = apply_tier_filter(&cover.coeff_sign_bypass, &csb_qp, tier_idx);
    let csl_keep = apply_tier_filter(&cover.coeff_suffix_lsb, &csl_qp, tier_idx);
    for (i, &keep) in csb_keep.iter().enumerate() {
        if !keep && i < content_costs.coeff_sign_bypass.len() {
            content_costs.coeff_sign_bypass[i] = f32::INFINITY;
        }
    }
    for (i, &keep) in csl_keep.iter().enumerate() {
        if !keep && i < content_costs.coeff_suffix_lsb.len() {
            content_costs.coeff_suffix_lsb[i] = f32::INFINITY;
        }
    }
}

/// WV.6.g.4.2a — resolve the cascade tier exactly as the orchestrator's Phase 3:
/// `PHASM_TIER_OVERRIDE` env if set, else `auto_select_tier` over the cover.
/// Whole-clip by nature (`auto_select_tier` reads Σ cover sizes + `msg_bytes`),
/// so the streaming Sweep A resolves it ONCE up front (a Σ-sizes pre-pass) and
/// threads the resulting `tier_idx` into every `gop_provisional_step`.
pub fn whole_clip_resolved_tier(cover: &DomainCover, qp: i32, msg_bytes: usize) -> u8 {
    use super::stego::tier_filter::{auto_select_tier, CascadeTier};
    let csb_qp: Vec<i32> = vec![qp; cover.coeff_sign_bypass.len()];
    let csl_qp: Vec<i32> = vec![qp; cover.coeff_suffix_lsb.len()];
    let resolved = match std::env::var("PHASM_TIER_OVERRIDE").ok().as_deref() {
        Some(s) if s != "auto" => s
            .parse::<u8>()
            .ok()
            .and_then(CascadeTier::from_u8)
            .unwrap_or(CascadeTier::Tier0),
        _ => auto_select_tier(
            cover, &csb_qp, &csl_qp, msg_bytes,
            super::stego::tier_filter::DEFAULT_HEADROOM,
        ),
    };
    resolved.as_u8()
}

/// WV.6.g.4.2a — prep the primary payload ONCE (frame_bytes + hhat_seed) for the
/// streaming GOP loop to reuse. Re-prepping per GOP would re-encrypt with a fresh
/// random salt+nonce (the g.4.0 determinism finding), so the byte-identical gate
/// must prep once here. Mirrors the orchestrator's Phase-1 primary prep.
pub fn prep_primary_payload(
    message: &str,
    files: &[crate::stego::payload::FileEntry],
    passphrase: &str,
) -> Result<(Vec<u8>, [u8; 32]), StegoError> {
    let primary = super::stego::oh264_capacity::prep_n_shadow_primary_payload(
        message, files, passphrase,
    )?;
    let hhat_seed = primary
        .keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;
    Ok((primary.frame_bytes, hhat_seed))
}

/// WV.6.g.4.1c — run the orchestrator's Phase 2 + Phase 3 (primary-only) and
/// return the [`ProvisionalWalk`]. No shadow inputs: Phases 2–3 are
/// shadow-independent (shadows enter only at the Phase-4 cascade loop).
#[allow(clippy::too_many_arguments)]
pub fn whole_clip_provisional_cover(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    primary_message: &str,
    primary_files: &[crate::stego::payload::FileEntry],
    primary_passphrase: &str,
    weights: &CostWeights,
) -> Result<ProvisionalWalk, StegoError> {
    use super::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};

    let primary = super::stego::oh264_capacity::prep_n_shadow_primary_payload(
        primary_message, primary_files, primary_passphrase,
    )?;
    let primary_hhat_seed = primary
        .keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let gop_size = opts.intra_period.max(1) as u32;
    let n_gops = n_frames.div_ceil(gop_size).max(1);
    let wire_only = wire_only_enabled();

    // ── Phase 2 — clean baseline encode + walk. ───────────────────
    set_549_pass_idx(1);
    let baseline = shadow_encode_all_gops(
        yuv, width, height, n_frames, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame, false, PassMode::Passthrough,
        gop_size, n_gops, wire_only,
    )?;
    let baseline_walk = walk_annex_b_for_cover_with_options(
        &baseline,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("baseline walk: {e}")))?;
    let cover_p1 = baseline_walk.cover.clone();

    // ── Phase 3 — content costs + safe_msl_baseline + tier ∞-gates. ─
    let mut content_costs = super::stego::content_costs::compute_content_costs_yuv(
        yuv, width, height, n_frames, &cover_p1, opts.qp,
    )?;
    let safe_msb_p1 = analyze_safe_mvd_subset(
        &baseline_walk.mvd_meta, baseline_walk.mb_w, baseline_walk.mb_h,
    );
    let safe_msl_baseline = derive_msl_safe_from_msb(
        &cover_p1.mvd_sign_bypass.positions, &safe_msb_p1,
        &cover_p1.mvd_suffix_lsb.positions,
    );
    apply_safe_msl_gate(&mut content_costs, &safe_msl_baseline);
    apply_tier_gate(
        &cover_p1, &mut content_costs, opts.qp,
        whole_clip_resolved_tier(&cover_p1, opts.qp, primary.frame_bits.len() / 8),
    );

    // Provisional plan (primary-only) → override map → emit → walk.
    let provisional_plan = {
        let (combined_cover, combined_costs, boundaries) =
            combine_cover_4domain(&cover_p1, &content_costs, weights);
        if combined_cover.is_empty() {
            return Err(StegoError::InvalidVideo(
                "provisional: combined cover empty".into(),
            ));
        }
        let full = embed_primary_per_gop_4domain(
            &cover_p1, combined_cover, &combined_costs,
            gop_size, n_gops, &primary.frame_bytes, &primary_hhat_seed,
        )?;
        split_plan_4domain(&full, &boundaries)
    };
    {
        let mut m = override_map.lock().expect("override map lock");
        m.clear();
        for (positions, bits, cover_bits) in [
            (&cover_p1.coeff_sign_bypass.positions, &provisional_plan.coeff_sign_bypass, &cover_p1.coeff_sign_bypass.bits),
            (&cover_p1.coeff_suffix_lsb.positions, &provisional_plan.coeff_suffix_lsb, &cover_p1.coeff_suffix_lsb.bits),
            (&cover_p1.mvd_sign_bypass.positions, &provisional_plan.mvd_sign_bypass, &cover_p1.mvd_sign_bypass.bits),
            (&cover_p1.mvd_suffix_lsb.positions, &provisional_plan.mvd_suffix_lsb, &cover_p1.mvd_suffix_lsb.bits),
        ] {
            let k = positions.len().min(bits.len()).min(cover_bits.len());
            for i in 0..k {
                if bits[i] != cover_bits[i] {
                    m.insert(positions[i].raw(), bits[i]);
                }
            }
        }
    }
    set_549_pass_idx(2);
    let bytes_prov = shadow_encode_all_gops(
        yuv, width, height, n_frames, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame, !wire_only, PassMode::Passthrough,
        gop_size, n_gops, wire_only,
    )?;
    let walk_prov = walk_annex_b_for_cover_with_options(
        &bytes_prov,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("provisional walk: {e}")))?;

    Ok(ProvisionalWalk {
        clean_cover: cover_p1,
        prov_cover: walk_prov.cover,
        prov_mvd_meta: walk_prov.mvd_meta,
        mb_w: walk_prov.mb_w,
        mb_h: walk_prov.mb_h,
    })
}

/// Whole-clip `safe_msl_prov` over the provisional walk — exactly the
/// orchestrator's Phase-3 tail. The reference the per-GOP assembly reproduces.
pub fn whole_clip_safe_msl_prov(pw: &ProvisionalWalk) -> Vec<bool> {
    let safe_msb = super::stego::cascade_safety::analyze_safe_mvd_subset(
        &pw.prov_mvd_meta, pw.mb_w, pw.mb_h,
    );
    super::stego::cascade_safety::derive_msl_safe_from_msb(
        &pw.prov_cover.mvd_sign_bypass.positions, &safe_msb,
        &pw.prov_cover.mvd_suffix_lsb.positions,
    )
}

/// WV.6.g.4.1c — per-GOP `safe_msl_prov[g]`: restrict the provisional MVD meta
/// + MVD-domain positions to GOP `g`'s frames (`frame_idx / gop_size == g`),
/// then run the SAME cascade-safety analysis over that subset alone.
///
/// `analyze_safe_mvd_subset` and `derive_msl_safe_from_msb` are strictly
/// frame-local — every neighbour lookup and slot-key is keyed by `frame_idx`,
/// and `shift_bound` accumulates per-position, so positions in different frames
/// never interact. A GOP is a contiguous frame range, so the concatenation of
/// `gop_safe_msl_prov(.., g)` for `g = 0..n_gops` (the walk lists positions in
/// frame-ascending order ⇒ GOP-contiguous) equals [`whole_clip_safe_msl_prov`].
/// That equality is what g.4.1c gates — and it is the Sweep-A per-GOP primitive.
pub fn gop_safe_msl_prov(pw: &ProvisionalWalk, gop_size: u32, g: u32) -> Vec<bool> {
    let gs = gop_size.max(1);
    // mvd_meta and mvd_sign_bypass.positions are parallel; slice both by the
    // SAME GOP predicate so safe_msb_g aligns with msb_pos_g.
    let msb_pos = &pw.prov_cover.mvd_sign_bypass.positions;
    let mut meta_g = Vec::new();
    let mut msb_pos_g = Vec::new();
    for (i, m) in pw.prov_mvd_meta.iter().enumerate() {
        if m.frame_idx / gs == g {
            meta_g.push(*m);
            if let Some(&k) = msb_pos.get(i) {
                msb_pos_g.push(k);
            }
        }
    }
    let msl_pos_g: Vec<_> = pw
        .prov_cover
        .mvd_suffix_lsb
        .positions
        .iter()
        .copied()
        .filter(|k| k.frame_idx() / gs == g)
        .collect();
    let safe_msb_g = super::stego::cascade_safety::analyze_safe_mvd_subset(
        &meta_g, pw.mb_w, pw.mb_h,
    );
    super::stego::cascade_safety::derive_msl_safe_from_msb(
        &msb_pos_g, &safe_msb_g, &msl_pos_g,
    )
}

/// WV.6.g.4.1d — one GOP's primary STC, replicating
/// [`embed_primary_per_gop_4domain`]'s per-GOP loop body for the streaming
/// Sweep-A/B path (the whole-clip function stays the untouched gold reference;
/// `streaming_provisional_plan_byte_identical_to_whole_clip` gates this against
/// it). Greedy uniform-with-carry: this GOP targets its even share of the
/// remaining `frame_bytes`; whatever it can't hold rolls forward via the
/// returned cursor. `gop_cover` / `gop_costs` are the GOP's combined 4-domain
/// cover (canonical CS→CSL→MVDs→MVDsl order, e.g. from `combine_cover_4domain`).
/// Returns the GOP's combined stego bits (length == `gop_cover.len()`) + the
/// advanced cursor.
fn embed_primary_one_gop(
    gop_cover: &[u8],
    gop_costs: &[f32],
    frame_bytes: &[u8],
    cursor: usize,
    g: u32,
    n_gops: u32,
    hhat_seed: &[u8; 32],
) -> Result<(Vec<u8>, usize), StegoError> {
    use crate::stego::chunk_frame::{build_chunk_frame_v3_1, build_first_chunk_frame_v3_1};

    let total_bytes =
        u32::try_from(frame_bytes.len()).map_err(|_| StegoError::MessageTooLarge)?;
    let is_first = g == 0;
    // Plain-tail GOP: payload already fully embedded — copy the cover through
    // (the decoder stops at `accumulated == total_bytes` and never STC-extracts
    // these). Matches the `continue` in the whole-clip loop.
    if !is_first && cursor >= frame_bytes.len() {
        return Ok((gop_cover.to_vec(), cursor));
    }

    let n_cover = gop_cover.len();
    let gops_remaining = (n_gops - g) as usize; // ≥ 1, includes this GOP
    let remaining = frame_bytes.len() - cursor;
    let mut want = if gops_remaining <= 1 {
        remaining
    } else {
        remaining.div_ceil(gops_remaining)
    };
    let mut out = gop_cover.to_vec();
    let mut new_cursor = cursor;
    loop {
        let chunk = &frame_bytes[cursor..cursor + want];
        let framed = if is_first {
            build_first_chunk_frame_v3_1(total_bytes, chunk)?
        } else {
            build_chunk_frame_v3_1(chunk)?
        };
        let frame_bits = bytes_to_bits_msb_first(&framed);
        let m = frame_bits.len();
        let w = if m == 0 { 0 } else { n_cover / m };
        if w >= 1 {
            let used = m * w;
            let hhat = generate_hhat(STC_H, w, hhat_seed);
            if let Some(plan) =
                stc_embed(&gop_cover[..used], &gop_costs[..used], &frame_bits, &hhat, STC_H, w)
            {
                for (k, slot) in out.iter_mut().enumerate().take(used) {
                    *slot = plan.stego_bits[k];
                }
                new_cursor = cursor + want;
                break;
            }
        }
        // Infeasible at `want`. want == 0 means even the header won't fit this
        // GOP's cover — leave the cover bits through (GOP 0 failing here surfaces
        // as MessageTooLarge via the caller's drain check).
        if want == 0 {
            break;
        }
        want -= 1usize.max(want / 8); // ~12% step, always reaches 0
    }
    Ok((out, new_cursor))
}

/// WV.6.g.4.1d — thin pub reference: the whole-clip primary plan exactly as the
/// orchestrator's Phase 3 builds it (`combine_cover_4domain` →
/// [`embed_primary_per_gop_4domain`] → `split_plan_4domain`). The gold standard
/// [`streaming_provisional_plan`] is gated against. Calls the UNTOUCHED
/// whole-clip function — no replication on the reference side.
#[allow(clippy::too_many_arguments)]
pub fn whole_clip_primary_plan(
    cover: &DomainCover,
    content_costs: &super::stego::orchestrate::DomainCosts,
    frame_bytes: &[u8],
    hhat_seed: &[u8; 32],
    gop_size: u32,
    n_gops: u32,
    weights: &CostWeights,
) -> Result<super::stego::orchestrate::DomainPlan, StegoError> {
    let (combined_cover, combined_costs, boundaries) =
        combine_cover_4domain(cover, content_costs, weights);
    let full = embed_primary_per_gop_4domain(
        cover, combined_cover, &combined_costs,
        gop_size, n_gops, frame_bytes, hhat_seed,
    )?;
    Ok(split_plan_4domain(&full, &boundaries))
}

/// WV.6.g.4.1d — assemble the whole-clip primary plan by streaming per GOP:
/// `combine_cover_4domain` each GOP's clean cover + costs, run
/// [`embed_primary_one_gop`] (forward `cursor`-carry), `split_plan_4domain`,
/// concatenate. Byte-identical to [`whole_clip_primary_plan`] over the whole
/// clip — the per-domain positions are GOP-contiguous (frame-ascending), so the
/// GOP-ordered concatenation of each GOP's per-domain plan slice reconstructs
/// the whole-clip per-domain plan block.
///
/// `per_gop_clean[g]` is GOP `g`'s clean cover (`gop_clean_cover`, GLOBAL
/// frame_idx); `per_gop_costs[g]` its content costs (the GOP slice; in g.4.2 the
/// per-frame `fill_frame_costs`). Nothing wider than one GOP is built at a time.
pub fn streaming_provisional_plan(
    per_gop_clean: &[DomainCover],
    per_gop_costs: &[super::stego::orchestrate::DomainCosts],
    frame_bytes: &[u8],
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
) -> Result<super::stego::orchestrate::DomainPlan, StegoError> {
    assert_eq!(
        per_gop_clean.len(),
        per_gop_costs.len(),
        "streaming_provisional_plan: one cost set per GOP cover",
    );
    let n_gops = per_gop_clean.len() as u32;
    let mut plan = super::stego::orchestrate::DomainPlan::default();
    let mut cursor = 0usize;
    for (g, (gc, costs_g)) in per_gop_clean.iter().zip(per_gop_costs).enumerate() {
        let (gop_combined, gop_costs, boundaries) =
            combine_cover_4domain(gc, costs_g, weights);
        let (gop_stego, new_cursor) = embed_primary_one_gop(
            &gop_combined, &gop_costs, frame_bytes, cursor, g as u32, n_gops, hhat_seed,
        )?;
        cursor = new_cursor;
        let gop_plan = split_plan_4domain(&gop_stego, &boundaries);
        plan.coeff_sign_bypass.extend_from_slice(&gop_plan.coeff_sign_bypass);
        plan.coeff_suffix_lsb.extend_from_slice(&gop_plan.coeff_suffix_lsb);
        plan.mvd_sign_bypass.extend_from_slice(&gop_plan.mvd_sign_bypass);
        plan.mvd_suffix_lsb.extend_from_slice(&gop_plan.mvd_suffix_lsb);
    }
    // Whole-clip drain check (matches `embed_primary_per_gop_4domain`'s tail).
    if cursor < frame_bytes.len() {
        return Err(StegoError::MessageTooLarge);
    }
    Ok(plan)
}

/// WV.6.g.4.2a — the per-GOP provisional Sweep-A step. Clean-encode GOP `g`
/// standalone, compute its content costs (LOCAL `frame_idx` — `gop_yuv` +
/// `cover_local`, since `compute_content_costs_yuv` indexes YUV by the cover's
/// frame_idx), apply the `safe_msl_baseline` + tier ∞-gates, run the cursor-carry
/// primary STC, emit the provisional bytes with the per-GOP override map (LOCAL
/// keys, `frame_idx_base = 0`), walk them, and derive `safe_msl_prov[g]`. Returns
/// the GLOBAL-remapped provisional cover (== the clean cover positions under
/// `wire_only`, Fact 2) + `safe_msl_prov[g]` + the advanced cursor.
///
/// Resolves cross-GOP deps 1 (local-frame content_costs) + 2 (cursor-carry
/// primary) + 3 (`tier_idx` passed in, resolved whole-clip by the caller via
/// [`whole_clip_resolved_tier`]). Working set is O(GOP) — one GOP's YUV + covers.
/// `frame_bytes` / `hhat_seed` come from [`prep_primary_payload`] (prepped ONCE).
#[allow(clippy::too_many_arguments)]
pub fn gop_provisional_step(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    frames_in_gop: u32,
    gop_global_start: u32,
    opts: EncodeOpts,
    frame_bytes: &[u8],
    hhat_seed: &[u8; 32],
    cursor: usize,
    g: u32,
    n_gops: u32,
    tier_idx: u8,
    weights: &CostWeights,
) -> Result<(DomainCover, Vec<bool>, usize), StegoError> {
    use super::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};

    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let wire_only = wire_only_enabled();
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // ── clean encode + walk (LOCAL frame_idx 0..frames_in_gop). ──
    set_549_pass_idx(1);
    let clean_bytes = shadow_encode_all_gops(
        gop_yuv, width, height, frames_in_gop, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame, false, PassMode::Passthrough,
        frames_in_gop.max(1), 1, wire_only,
    )?;
    let clean_walk = walk_annex_b_for_cover_with_options(
        &clean_bytes,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("gop clean walk: {e}")))?;
    let cover_local = clean_walk.cover;

    // ── content costs (LOCAL cover + GOP yuv) + ∞-gates. ──
    let mut content_costs = super::stego::content_costs::compute_content_costs_yuv(
        gop_yuv, width, height, frames_in_gop, &cover_local, opts.qp,
    )?;
    let safe_msb = analyze_safe_mvd_subset(
        &clean_walk.mvd_meta, clean_walk.mb_w, clean_walk.mb_h,
    );
    let safe_msl_baseline = derive_msl_safe_from_msb(
        &cover_local.mvd_sign_bypass.positions, &safe_msb,
        &cover_local.mvd_suffix_lsb.positions,
    );
    apply_safe_msl_gate(&mut content_costs, &safe_msl_baseline);
    apply_tier_gate(&cover_local, &mut content_costs, opts.qp, tier_idx);

    // ── cursor-carry primary STC → provisional override map (LOCAL keys). ──
    let (gop_combined, gop_costs, boundaries) =
        combine_cover_4domain(&cover_local, &content_costs, weights);
    let (gop_stego, new_cursor) = embed_primary_one_gop(
        &gop_combined, &gop_costs, frame_bytes, cursor, g, n_gops, hhat_seed,
    )?;
    let gop_plan = split_plan_4domain(&gop_stego, &boundaries);
    {
        let mut m = override_map.lock().expect("override map lock");
        m.clear();
        for (positions, bits, cover_bits) in [
            (&cover_local.coeff_sign_bypass.positions, &gop_plan.coeff_sign_bypass, &cover_local.coeff_sign_bypass.bits),
            (&cover_local.coeff_suffix_lsb.positions, &gop_plan.coeff_suffix_lsb, &cover_local.coeff_suffix_lsb.bits),
            (&cover_local.mvd_sign_bypass.positions, &gop_plan.mvd_sign_bypass, &cover_local.mvd_sign_bypass.bits),
            (&cover_local.mvd_suffix_lsb.positions, &gop_plan.mvd_suffix_lsb, &cover_local.mvd_suffix_lsb.bits),
        ] {
            let k = positions.len().min(bits.len()).min(cover_bits.len());
            for i in 0..k {
                if bits[i] != cover_bits[i] {
                    m.insert(positions[i].raw(), bits[i]);
                }
            }
        }
    }

    // ── provisional emit + walk → safe_msl_prov[g]. ──
    set_549_pass_idx(2);
    let prov_bytes = shadow_encode_all_gops(
        gop_yuv, width, height, frames_in_gop, opts,
        &override_map, &mb_type_table, &applied,
        mb_width, mb_per_frame, !wire_only, PassMode::Passthrough,
        frames_in_gop.max(1), 1, wire_only,
    )?;
    let prov_walk = walk_annex_b_for_cover_with_options(
        &prov_bytes,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("gop prov walk: {e}")))?;
    let safe_msb_prov = analyze_safe_mvd_subset(
        &prov_walk.mvd_meta, prov_walk.mb_w, prov_walk.mb_h,
    );
    let safe_msl_prov = derive_msl_safe_from_msb(
        &prov_walk.cover.mvd_sign_bypass.positions, &safe_msb_prov,
        &prov_walk.cover.mvd_suffix_lsb.positions,
    );

    let mut prov_cover = prov_walk.cover;
    if gop_global_start > 0 {
        remap_cover_frame_idx(&mut prov_cover, gop_global_start);
    }
    Ok((prov_cover, safe_msl_prov, new_cursor))
}

/// WV.6.g.4.2b — Sweep A's output: per-shadow sorted top-N selections (capacity
/// `n_total_max`) + per-GOP `safe_msl_prov` (Sweep B reuses both). Bounded:
/// O(N-shadow-positions) + O(Σ MvdSuffixLsb positions) — never O(clip).
pub struct SweepAResult {
    /// Per shadow (in input order): its sorted top-N [`ShadowSlot`]s.
    pub selections: Vec<Vec<super::stego::shadow::ShadowSlot>>,
    /// Per GOP (in forward order): the provisional cascade-safety mask for
    /// MvdSuffixLsb — Sweep B's override MSL gate + shadow-embed gate.
    pub per_gop_safe_msl: Vec<Vec<bool>>,
}

/// WV.6.g.4.2b — **Sweep A**: forward over GOPs (one in flight), run
/// [`gop_provisional_step`] (clean + provisional emit, O(GOP)), and stream each
/// GOP's positions — gated by that GOP's `safe_msl_prov` — into the per-shadow
/// [`ShadowSelectionSweep`]. Returns the per-shadow sorted top-N + per-GOP
/// `safe_msl_prov`. Byte-identical selection to the whole-clip
/// `prepare_shadow_over_emit_cover`: selection reads only positions (= the clean
/// cover, Fact 2 — `gop_provisional_step` returns the provisional cover whose
/// positions equal the clean cover's) + the `safe_msl` mask, both proven to
/// decompose per-GOP (g.4.1b + g.4.1c + g.4.2a).
#[allow(clippy::too_many_arguments)]
pub fn sweep_a(
    source: &mut dyn GopYuvSource,
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    frame_bytes: &[u8],
    hhat_seed: &[u8; 32],
    shadows: &[crate::stego::shadow_layer::ShadowLayer<'_>],
    capacities: &[usize],
    tier_idx: u8,
    weights: &CostWeights,
) -> Result<SweepAResult, StegoError> {
    let gop_size = opts.intra_period.max(1) as u32;
    let n_gops = n_frames.div_ceil(gop_size).max(1);
    let mut sweep = ShadowSelectionSweep::new(shadows, capacities)?;
    let mut per_gop_safe_msl: Vec<Vec<bool>> = Vec::with_capacity(n_gops as usize);
    let mut cursor = 0usize;
    source.rewind()?;
    for g in 0..n_gops {
        let gop_yuv = source.gop_yuv(g)?;
        let start = g * gop_size;
        let fig = (n_frames - start).min(gop_size);
        let (prov_cover, safe_msl, nc) = gop_provisional_step(
            &gop_yuv, width, height, fig, start, opts,
            frame_bytes, hhat_seed, cursor, g, n_gops, tier_idx, weights,
        )?;
        cursor = nc;
        // Fact 2: prov_cover positions == clean cover positions, so the priority
        // computed from each key matches the whole-clip path's.
        sweep.push_gop(&prov_cover, Some(&safe_msl));
        per_gop_safe_msl.push(safe_msl);
    }
    Ok(SweepAResult {
        selections: sweep.finish(),
        per_gop_safe_msl,
    })
}

/// WV.6.g.4.2b — **Sweep B emit** (one parity tier). Per GOP, forward: re-encode
/// clean → `cover_p1[g]`, embed this GOP's shadow LSBs + ∞-mask, run the
/// cursor-carry primary STC, build the override map (MSL-gated by Sweep A's
/// `safe_msl_prov[g]`), emit, and accumulate the bytes. Byte-identical to the
/// whole-clip orchestrator's Phase-4 emit at this tier — each GOP's standalone
/// slab equals the orchestrator's GOP-`g` slab (g.4.2a's `frame_idx_base`
/// argument), and the per-GOP STC over the shadow-embedded cover reproduces
/// `embed_primary_per_gop_4domain`'s plan (g.4.1d).
///
/// `shadow_rs[s]` = `(rs_bits, frame_data_len)` for shadow `s` at this tier
/// (from [`super::stego::shadow::build_shadow_rs_frame`]); `rs_bits[i]` pairs
/// with `sweep_a_result.selections[s][i]` (the sorted top-N). The global→GOP-local
/// position map uses running per-domain offsets (`cover_p1[g]` sizes), matching
/// the whole-clip concatenation. Working set is O(GOP) + the bounded Sweep A
/// state; the emitted clip itself is O(clip) (streamed out, not the working set).
#[allow(clippy::too_many_arguments)]
pub fn sweep_b_emit(
    source: &mut dyn GopYuvSource,
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    primary_frame_bytes: &[u8],
    hhat_seed: &[u8; 32],
    sweep_a_result: &SweepAResult,
    shadow_rs: &[(Vec<u8>, usize)],
    parity_len: usize,
    tier_idx: u8,
    weights: &CostWeights,
) -> Result<(Vec<u8>, Vec<usize>), StegoError> {
    use super::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};
    use super::stego::shadow::{
        apply_shadow_to_plan, embed_shadow_lsb, overlay_infinity_costs, ShadowSlot, ShadowState,
    };

    let n_shadows = sweep_a_result.selections.len();
    assert_eq!(n_shadows, shadow_rs.len(), "sweep_b_emit: one rs frame per shadow");
    let gop_size = opts.intra_period.max(1) as u32;
    let n_gops = n_frames.div_ceil(gop_size).max(1);
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let wire_only = wire_only_enabled();
    // n_total per shadow at this tier = the RS-frame bit count.
    let n_totals: Vec<usize> = shadow_rs.iter().map(|(bits, _)| bits.len()).collect();

    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    let mut offs = [0usize; 4]; // running per-domain global offsets (CSB,CSL,MSB,MSL)
    let mut cursor = 0usize;
    let mut out: Vec<u8> = Vec::new();
    let mut gop_lens: Vec<usize> = Vec::with_capacity(n_gops as usize);
    source.rewind()?;

    for g in 0..n_gops {
        let gop_yuv = source.gop_yuv(g)?;
        let fig = (n_frames - g * gop_size).min(gop_size);

        // 1. clean encode → cover_local (LOCAL frame_idx) + mvd_meta.
        override_map.lock().expect("lock").clear();
        set_549_pass_idx(1);
        let clean_bytes = shadow_encode_all_gops(
            &gop_yuv, width, height, fig, opts, &override_map, &mb_type_table, &applied,
            mb_width, mb_per_frame, false, PassMode::Passthrough, fig.max(1), 1, wire_only,
        )?;
        let clean_walk = walk_annex_b_for_cover_with_options(
            &clean_bytes,
            WalkOptions { record_mvd: true, ..Default::default() },
        )
        .map_err(|e| StegoError::InvalidVideo(format!("sweep_b clean walk: {e}")))?;
        let cover_local = clean_walk.cover;

        // 2. content_costs (local) + safe_msl_baseline + tier ∞-gates.
        let mut content_costs = super::stego::content_costs::compute_content_costs_yuv(
            &gop_yuv, width, height, fig, &cover_local, opts.qp,
        )?;
        let safe_msb = analyze_safe_mvd_subset(
            &clean_walk.mvd_meta, clean_walk.mb_w, clean_walk.mb_h,
        );
        let safe_msl_baseline = derive_msl_safe_from_msb(
            &cover_local.mvd_sign_bypass.positions, &safe_msb,
            &cover_local.mvd_suffix_lsb.positions,
        );
        apply_safe_msl_gate(&mut content_costs, &safe_msl_baseline);
        apply_tier_gate(&cover_local, &mut content_costs, opts.qp, tier_idx);

        let sizes = [
            cover_local.coeff_sign_bypass.len(),
            cover_local.coeff_suffix_lsb.len(),
            cover_local.mvd_sign_bypass.len(),
            cover_local.mvd_suffix_lsb.len(),
        ];

        // 3. per-GOP ShadowState_local for each shadow: the global selection's
        //    positions that fall in THIS GOP's per-domain ranges, intra_index
        //    relabelled local (global − offset). `rs_bits[i]` ↔ selection[i].
        let mut states: Vec<ShadowState> = Vec::with_capacity(n_shadows);
        for s in 0..n_shadows {
            let sel = &sweep_a_result.selections[s];
            let bits = &shadow_rs[s].0;
            let mut st = ShadowState {
                positions: Vec::new(),
                bits: Vec::new(),
                n_total: 0,
                parity_len,
                frame_data_len: shadow_rs[s].1,
            };
            for i in 0..n_totals[s].min(sel.len()) {
                let slot = sel[i];
                let d = slot.domain as usize;
                if slot.intra_index >= offs[d] && slot.intra_index < offs[d] + sizes[d] {
                    st.positions.push(ShadowSlot {
                        domain: slot.domain,
                        intra_index: slot.intra_index - offs[d],
                        priority: slot.priority,
                    });
                    st.bits.push(bits[i]);
                }
            }
            st.n_total = st.positions.len();
            states.push(st);
        }

        // 4. embed shadow LSBs into the cover bits + ∞-overlay the costs.
        let mut sb_csb = cover_local.coeff_sign_bypass.bits.clone();
        let mut sb_csl = cover_local.coeff_suffix_lsb.bits.clone();
        let mut sb_msb = cover_local.mvd_sign_bypass.bits.clone();
        let mut sb_msl = cover_local.mvd_suffix_lsb.bits.clone();
        for st in &states {
            embed_shadow_lsb(&mut sb_csb, &mut sb_csl, &mut sb_msb, &mut sb_msl, st);
            overlay_infinity_costs(
                &mut content_costs.coeff_sign_bypass,
                &mut content_costs.coeff_suffix_lsb,
                &mut content_costs.mvd_sign_bypass,
                &mut content_costs.mvd_suffix_lsb,
                st,
            );
        }

        // 5. combine + cursor-carry primary STC + split.
        let (combined, combined_costs, boundaries) =
            combine_bits_4domain(&sb_csb, &sb_csl, &sb_msb, &sb_msl, &content_costs, weights);
        drop((sb_csb, sb_csl, sb_msb, sb_msl));
        let (gop_stego, nc) = embed_primary_one_gop(
            &combined, &combined_costs, primary_frame_bytes, cursor, g, n_gops, hhat_seed,
        )?;
        cursor = nc;
        let mut domain_plan = split_plan_4domain(&gop_stego, &boundaries);

        // 6. defensive shadow stamp.
        for st in &states {
            apply_shadow_to_plan(
                &mut domain_plan.coeff_sign_bypass,
                &mut domain_plan.coeff_suffix_lsb,
                &mut domain_plan.mvd_sign_bypass,
                &mut domain_plan.mvd_suffix_lsb,
                st,
            );
        }

        // 7. override map (LOCAL keys; CSB/CSL/MSB unconditional, MSL gated by
        //    Sweep A's safe_msl_prov[g]).
        let safe_msl_prov = &sweep_a_result.per_gop_safe_msl[g as usize];
        {
            let mut m = override_map.lock().expect("override map lock");
            m.clear();
            for (positions, plan, cover_bits) in [
                (&cover_local.coeff_sign_bypass.positions, &domain_plan.coeff_sign_bypass, &cover_local.coeff_sign_bypass.bits),
                (&cover_local.coeff_suffix_lsb.positions, &domain_plan.coeff_suffix_lsb, &cover_local.coeff_suffix_lsb.bits),
                (&cover_local.mvd_sign_bypass.positions, &domain_plan.mvd_sign_bypass, &cover_local.mvd_sign_bypass.bits),
            ] {
                let n = positions.len().min(plan.len()).min(cover_bits.len());
                for i in 0..n {
                    if plan[i] != cover_bits[i] {
                        m.insert(positions[i].raw(), plan[i]);
                    }
                }
            }
            let n = cover_local.mvd_suffix_lsb.positions.len()
                .min(domain_plan.mvd_suffix_lsb.len())
                .min(cover_local.mvd_suffix_lsb.bits.len());
            for i in 0..n {
                if domain_plan.mvd_suffix_lsb[i] != cover_local.mvd_suffix_lsb.bits[i]
                    && safe_msl_prov.get(i).copied().unwrap_or(false)
                {
                    m.insert(
                        cover_local.mvd_suffix_lsb.positions[i].raw(),
                        domain_plan.mvd_suffix_lsb[i],
                    );
                }
            }
        }

        // 8. emit GOP g standalone with the shadow-aware override map.
        set_549_pass_idx(2);
        let bytes = shadow_encode_all_gops(
            &gop_yuv, width, height, fig, opts, &override_map, &mb_type_table, &applied,
            mb_width, mb_per_frame, !wire_only, PassMode::Passthrough, fig.max(1), 1, wire_only,
        )?;
        gop_lens.push(bytes.len());
        out.extend_from_slice(&bytes);

        for d in 0..4 {
            offs[d] += sizes[d];
        }
    }
    Ok((out, gop_lens))
}

/// WV.6.g.5 — **streaming per-tier verify**: the O(GOP) replacement for the
/// cascade's whole-clip `walk_final` re-decode. Re-slices the emitted clip into
/// per-GOP Annex-B chunks (`gop_byte_lens` from [`sweep_b_emit`]), walks ONE GOP
/// at a time, re-derives that GOP's **emitted** priority order + cascade-safety
/// (`safe_msl`), and streams its eligible positions (with emitted bits, GLOBAL
/// keys + intra_index) into each shadow's bounded [`ShadowBitTopN`]. After the
/// last GOP, decodes each shadow from its retained priority-ordered bits via
/// [`decode_shadow_from_priority_lsbs`]. Returns `true` iff every shadow decodes.
///
/// Faithful to the whole-clip `shadow_extract` it replaces: (a) under `wire_only`
/// the emitted cover's positions/keys equal the clean cover's, so the per-GOP
/// GLOBAL-remapped priority order equals `priority_slots(whole_emitted_clip)`;
/// (b) `n_totals[s]` (this tier's RS-frame bit count) retains exactly the frame
/// region, which `decode_shadow_from_priority_lsbs` self-delimits identically to
/// the full eligible stream (large frames ≥255 B keep the peek path; small
/// frames use brute-force either way); (c) cascade-safety is frame-local, so the
/// per-GOP `safe_msl` assembles to the whole-clip mask. Working set: one GOP walk
/// + the bounded heaps — no O(clip) walked cover.
#[allow(clippy::too_many_arguments)]
pub fn streaming_shadow_verify(
    emitted: &[u8],
    gop_byte_lens: &[usize],
    width: u32,
    height: u32,
    gop_size: u32,
    shadows: &[crate::stego::shadow_layer::ShadowLayer<'_>],
    n_totals: &[usize],
    // WV.6.g.progress — ticked once per GOP (pass 4 of 4). A closure (not the
    // private `ShadowProgress`) keeps this public signature leak-free.
    mut on_gop: Option<&mut dyn FnMut()>,
) -> Result<bool, StegoError> {
    use super::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};
    use super::stego::shadow::{
        decode_shadow_from_priority_lsbs, for_each_eligible_position, ShadowBitTopN,
    };

    assert_eq!(
        shadows.len(),
        n_totals.len(),
        "streaming_shadow_verify: one n_total per shadow",
    );
    // One bounded bit-selector per shadow (perm_seed from passphrase, exactly as
    // priority_slots / shadow_extract derive it).
    let mut verifiers: Vec<ShadowBitTopN> = Vec::with_capacity(shadows.len());
    for (s, &nt) in shadows.iter().zip(n_totals) {
        let seed = crate::stego::crypto::derive_shadow_structural_key(s.passphrase)?;
        verifiers.push(ShadowBitTopN::new(&seed, nt));
    }

    let mut offs = [0usize; 4]; // running global per-domain offsets (CSB,CSL,MSB,MSL)
    let mut start = 0usize;
    // Cumulative ACTUAL frames decoded so far — mirrors the streaming decoder's
    // `frames_so_far`; cross-checked against g*gop_size below (remap-base contract).
    let mut cumulative_frames = 0u32;
    for (g, &len) in gop_byte_lens.iter().enumerate() {
        if let Some(f) = on_gop.as_mut() {
            f(); // pass 4 of 4 — keep the bar moving through the verify re-decode
        }
        let end = start + len;
        let gop_bytes = emitted.get(start..end).ok_or_else(|| {
            StegoError::InvalidVideo(format!(
                "streaming_shadow_verify: GOP {g} range {start}..{end} out of {} bytes",
                emitted.len(),
            ))
        })?;
        start = end;

        let walk = walk_annex_b_for_cover_with_options(
            gop_bytes,
            WalkOptions { record_mvd: true, ..Default::default() },
        )
        .map_err(|e| StegoError::InvalidVideo(format!("streaming_shadow_verify walk gop {g}: {e}")))?;
        // Cascade-safety on the EMITTED meta. The mask is aligned to the msl
        // positions by index (remap-invariant), so compute it on the LOCAL cover.
        let safe_msb = analyze_safe_mvd_subset(&walk.mvd_meta, walk.mb_w, walk.mb_h);
        let mut cover = walk.cover;
        let safe_msl = derive_msl_safe_from_msb(
            &cover.mvd_sign_bypass.positions,
            &safe_msb,
            &cover.mvd_suffix_lsb.positions,
        );
        // GLOBAL-remap the keys so the priority order matches the decoder's
        // priority_slots over the whole emitted clip.
        //
        // Remap-base contract: we use g*gop_size here, but the streaming DECODER
        // (`StreamingDecodeSession::try_shadow_streaming`) remaps by the
        // CUMULATIVE count of frames it has actually decoded (`frames_so_far`).
        // The two agree iff every non-final GOP holds exactly gop_size frames
        // (true for OH264's fixed-GOP output today). A future variable-GOP
        // encoder would diverge the two priority orders and silently break
        // shadow decode — the debug_assert catches it here, where both the
        // formula base and the actual frame count are known.
        let gstart = (g as u32) * gop_size;
        let mb_per_frame = (walk.mb_w as usize) * (walk.mb_h as usize);
        let gop_frames = if mb_per_frame > 0 { (walk.n_mb / mb_per_frame) as u32 } else { 0 };
        debug_assert_eq!(
            gstart, cumulative_frames,
            "remap-base divergence at GOP {g}: g*gop_size={gstart} != cumulative actual \
             frames={cumulative_frames}; would break try_shadow_streaming's frames_so_far order"
        );
        cumulative_frames += gop_frames;
        if gstart > 0 {
            remap_cover_frame_idx(&mut cover, gstart);
        }
        let sizes = [
            cover.coeff_sign_bypass.len(),
            cover.coeff_suffix_lsb.len(),
            cover.mvd_sign_bypass.len(),
            cover.mvd_suffix_lsb.len(),
        ];

        // Eligibility + domain order via the shared `for_each_eligible_position`
        // (safe_csl = None ⇒ all CSL; emitted cascade-safe MSL only) — the same
        // rule `priority_slots` + the streaming decoder use, so the retained
        // bit order is bit-identical.
        for v in &mut verifiers {
            for_each_eligible_position(&cover, None, Some(safe_msl.as_slice()), |domain, i, key| {
                let bit = match domain {
                    EmbedDomain::CoeffSignBypass => cover.coeff_sign_bypass.bits[i],
                    EmbedDomain::CoeffSuffixLsb => cover.coeff_suffix_lsb.bits[i],
                    EmbedDomain::MvdSignBypass => cover.mvd_sign_bypass.bits[i],
                    EmbedDomain::MvdSuffixLsb => cover.mvd_suffix_lsb.bits[i],
                };
                v.push(domain, offs[domain as usize] + i, *key, bit);
            });
        }

        for d in 0..4 {
            offs[d] += sizes[d];
        }
    }

    // Decode each shadow from its retained priority-ordered emitted bits.
    for (v, s) in verifiers.into_iter().zip(shadows) {
        let all_lsbs = v.into_sorted_bits();
        if decode_shadow_from_priority_lsbs(&all_lsbs, s.passphrase).is_err() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// WV.6.g.4.1b — Sweep A's streaming shadow-position selector.
///
/// Holds one [`StreamingTopN`] per shadow plus the running **global**
/// per-domain offsets. It is fed one GOP's clean cover at a time, in forward
/// order, so nothing wider than a single GOP is ever live; each
/// [`push_gop`](Self::push_gop) streams that GOP's eligible positions into
/// every shadow's heap with `intra_index = global_offset[domain] + local`.
/// Because the whole-clip per-domain `positions` vector is exactly the
/// GOP-ordered concatenation of the per-GOP slices (proven byte-identical by
/// `gop_clean_cover_assembles_to_whole_clip_baseline`), that reconstructed
/// `intra_index` equals the `enumerate()` index the whole-clip
/// `priority_slots` assigns.
///
/// Eligibility mirrors `priority_slots` exactly: CoeffSignBypass +
/// CoeffSuffixLsb + MvdSignBypass are always injectable (the orchestrator
/// passes `safe_csl = None`); MvdSuffixLsb is included only at positions where
/// the GOP-local `safe_msl` mask is `true` (and the domain is excluded
/// entirely when no mask is supplied). [`finish`](Self::finish) drains each
/// selector to its sorted top-N — bit-identical to
/// `priority_slots(whole_clip_cover, None, safe_msl).take(capacity)` (g.2's
/// `StreamingTopN` equivalence, lifted to the per-GOP-streamed cover).
///
/// The keys must already be GLOBAL-remapped (`gop_clean_cover` does this), so
/// priorities — `ChaCha20(seed).seek(key.raw()*2)` — match the whole-clip walk.
pub struct ShadowSelectionSweep {
    selectors: Vec<super::stego::shadow::StreamingTopN>,
    off_csb: usize,
    off_csl: usize,
    off_msb: usize,
    off_msl: usize,
}

impl ShadowSelectionSweep {
    /// One selector per shadow, in `shadows` order. Each selector's permutation
    /// seed is derived from that shadow's passphrase exactly as `prepare_shadow`
    /// does (`crypto::derive_shadow_structural_key`), so the streamed priorities
    /// match the whole-clip path. `capacities[i]` is shadow `i`'s `n_total_max`
    /// (the largest parity tier's RS-bit count); the sorted output is
    /// prefix-`take`n per parity tier downstream (§9.1 Fact 1).
    pub fn new(
        shadows: &[crate::stego::shadow_layer::ShadowLayer<'_>],
        capacities: &[usize],
    ) -> Result<Self, StegoError> {
        assert_eq!(
            shadows.len(),
            capacities.len(),
            "ShadowSelectionSweep: one capacity per shadow",
        );
        let mut selectors = Vec::with_capacity(shadows.len());
        for (s, &cap) in shadows.iter().zip(capacities) {
            let seed = crate::stego::crypto::derive_shadow_structural_key(s.passphrase)?;
            selectors.push(super::stego::shadow::StreamingTopN::new(&seed, cap));
        }
        Ok(Self { selectors, off_csb: 0, off_csl: 0, off_msb: 0, off_msl: 0 })
    }

    /// Stream GOP `g`'s clean cover (already GLOBAL-remapped) into every
    /// selector. `safe_msl_gop` is this GOP's MvdSuffixLsb cascade-safety mask,
    /// aligned to `gop_clean.mvd_suffix_lsb.positions`; `None` excludes the MSL
    /// domain entirely (matching `priority_slots`' `safe_msl = None` default).
    pub fn push_gop(&mut self, gop_clean: &DomainCover, safe_msl_gop: Option<&[bool]>) {
        // Snapshot the offsets so every selector sees GOP g at the SAME global
        // indices (the offsets advance once, after all selectors are fed).
        let (o_csb, o_csl, o_msb, o_msl) =
            (self.off_csb, self.off_csl, self.off_msb, self.off_msl);
        for sel in &mut self.selectors {
            // CoeffSignBypass — always injectable (sign-only, no cascade).
            for (i, &key) in gop_clean.coeff_sign_bypass.positions.iter().enumerate() {
                sel.push(EmbedDomain::CoeffSignBypass, o_csb + i, key);
            }
            // CoeffSuffixLsb — safe_csl = None ⇒ every position eligible.
            for (i, &key) in gop_clean.coeff_suffix_lsb.positions.iter().enumerate() {
                sel.push(EmbedDomain::CoeffSuffixLsb, o_csl + i, key);
            }
            // MvdSignBypass — always injectable (sign-only bitstream override).
            for (i, &key) in gop_clean.mvd_sign_bypass.positions.iter().enumerate() {
                sel.push(EmbedDomain::MvdSignBypass, o_msb + i, key);
            }
            // MvdSuffixLsb — cascade-safe positions only, when a mask is given.
            if let Some(mask) = safe_msl_gop {
                for (i, &key) in gop_clean.mvd_suffix_lsb.positions.iter().enumerate() {
                    if mask.get(i).copied().unwrap_or(false) {
                        sel.push(EmbedDomain::MvdSuffixLsb, o_msl + i, key);
                    }
                }
            }
        }
        // Advance the global per-domain offsets by this GOP's domain counts.
        self.off_csb += gop_clean.coeff_sign_bypass.len();
        self.off_csl += gop_clean.coeff_suffix_lsb.len();
        self.off_msb += gop_clean.mvd_sign_bypass.len();
        self.off_msl += gop_clean.mvd_suffix_lsb.len();
    }

    /// Drain every selector to its sorted top-N (`ShadowSlot`s in the
    /// `priority_slots` total order, ascending priority).
    pub fn finish(self) -> Vec<Vec<super::stego::shadow::ShadowSlot>> {
        self.selectors.into_iter().map(|s| s.into_sorted()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::openh264::SESSION_TEST_MUTEX;

    fn synth_yuv(w: u32, h: u32, n_frames: u32) -> Vec<u8> {
        let frame_size = (w * h * 3 / 2) as usize;
        let mut out = Vec::with_capacity(frame_size * n_frames as usize);
        let mut s: u32 = 0xCAFE_F00D;
        for frame in 0..n_frames {
            for j in 0..h {
                for i in 0..w {
                    let v = ((i + frame * 2) ^ (j + frame * 3)) as u8;
                    out.push(v);
                }
            }
            for _plane in 0..2 {
                for j in 0..(h / 2) {
                    for i in 0..(w / 2) {
                        s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                        let texture = (s >> 16) as u8;
                        let pos = (i + j + frame) as u8;
                        out.push(texture.wrapping_add(pos));
                    }
                }
            }
        }
        out
    }

    /// B-lite.1 (#890) gate — the global-free `clean_encode_gop` producer is
    /// byte-identical to the production `oh264_plain_encode_gop` on the clean
    /// path. Running plain FIRST leaves the fork globals clean (its StegoSession
    /// drops to a null callback table; pass_mode reset to Passthrough), so the
    /// clean producer (which touches none of them) sees the same state. A
    /// divergence here means the StegoSession / reset / set_frame_num the clean
    /// producer omits are NOT no-ops on the clean path.
    #[cfg(feature = "h264-encoder")]
    #[test]
    fn blite1_clean_encode_matches_plain() {
        let _g = SESSION_TEST_MUTEX.lock().unwrap();
        const W: u32 = 176;
        const H: u32 = 96;
        const NF: u32 = 8;
        let yuv = synth_yuv(W, H, NF);
        let plain = oh264_plain_encode_gop(&yuv, W, H, NF, EncodeOpts { qp: 26, intra_period: 30 })
            .expect("plain encode");
        let clean = clean_encode_gop(&yuv, W, H, NF, EncodeOpts { qp: 26, intra_period: 30 })
            .expect("clean encode");
        assert!(plain.len() > 64, "plain encode suspiciously small: {}", plain.len());
        assert_eq!(
            clean,
            plain,
            "clean_encode_gop diverged from oh264_plain_encode_gop ({} vs {} bytes) — \
             the omitted StegoSession/reset/set_frame_num are NOT no-ops on the clean path",
            clean.len(),
            plain.len()
        );
        eprintln!(
            "blite1: clean_encode_gop byte-identical to oh264_plain_encode_gop ({} bytes, {W}x{H} {NF}f)",
            clean.len()
        );
    }

    // The full round-trip lives in the `openh264_stego_roundtrip`
    // integration-test binary (`core/tests/openh264_stego_roundtrip.rs`).
    // Lib tests can't host it: C.8.7's `phasm_set_mv_override_active`
    // fork-side global is process-wide, and the 1300+ other openh264-
    // touching lib tests share state with it under cargo's single test
    // binary, producing 1-2 wire-flip cascade leaks. In an isolated
    // integration-test binary the state is pristine and the round-trip
    // is byte-exact (C.8.12 corpus suite is the structural proof).
}
