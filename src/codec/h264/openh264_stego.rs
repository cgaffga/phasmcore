// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase C.8.13 (#446) — production stego orchestrator on top of the
// OpenH264 backend. Single-domain (CoeffSign) STC encode + brute-force
// decode over walker-aligned cover, with passphrase-derived seeds.
//
// **STATUS: v1.0 ALPHA** — API surface + capacity primitive + encode/decode
// shape are shipped, but the round-trip is **not yet robust** against the
// 0.003 % residual cascade-leak rate observed at higher flip counts
// (~100 + flips). For small messages (≤ 10-20 flips, ~10 message bytes) the
// round-trip is essentially deterministic; above that, occasional 1-2 bit
// wire divergences from cascade leak break STC syndrome and fail decode.
// C.8.13 ship gate is documented in
// `docs/design/video/h264/phase-c8-visual-recon-plan.md` §C.8.13.
//
// Two follow-ons clear the alpha tag:
//   * C.8.13(b) — cascade-break gap audit: localise which domain
//     (CoeffSign / chroma DC/AC / deblock / …) leaks 2 / 76 384 wire
//     bits on 124-flip plans and fix structurally.
//   * C.8.13(c) — fallback: WET-INFINITY cost at empirically-detected
//     unsafe positions per b10_cascade_safe_roundtrip pattern (slow but
//     always correct), used as the ship-safe path until (b) lands.
//
// **Single-domain v1.0**: only CoeffSign positions carry the message.
// CoeffSuffixLsb / MvdSign / MvdSuffixLsb are reserved for v1.1+ once
// cascade-safety analysis (per-domain `cascade_safety.rs` equivalents)
// is wired through the OpenH264 path.
//
// **Cascade-safety**: relies on the C.8.3-11 dual-recon (`pVisualRecPic`)
// to keep mode-decision identical between baseline and stego encodes.
// On the 4-fixture C.8.12 corpus this holds for the small flip set in
// that test (≤ 3 flips); higher-flip-count plans intermittently observe
// the residual leak described above.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

// #549 Phase C diagnostic — thread-local pass index, set by
// `encode_yuv_with_pre_framed_bits_4domain` before each `encode_once`
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

fn disable_coeff_capture() {
    unsafe { core_openh264_sys::phasm_set_post_quant_callback(None); }
}

fn take_coeff_capture() -> Vec<(u32, u16, u16, [i16; 256])> {
    disable_coeff_capture();
    COEFF_CAPTURE.with(|b| b.borrow_mut().take().unwrap_or_default())
}

use core_openh264_sys::{
    encoder_pos_to_phasm_position_key, PHASM_MB_TYPE_OTHER, ZZ_SCAN_4X4,
};

use super::cabac::bin_decoder::slice::{
    walk_annex_b_for_cover, walk_annex_b_for_cover_with_options, WalkOptions,
};
use super::stego::cost_weights::{
    combine_cover_4domain, split_plan_4domain, CostWeights,
};
use super::stego::orchestrate::DomainCosts;
use super::openh264::{
    set_frame_num, set_pass_mode, Encoder, EncoderError, MbDecision, PassMode,
    StegoHandlers, StegoSession,
};
use super::pass2_cache::DecisionCache;
use super::stego::hook::{BinKind, EmbedDomain, SyntaxPath};
use super::stego::inject::DomainCover;
use super::stego::keys::CabacStegoMasterKeys;
use crate::stego::{crypto, frame, payload};
use crate::stego::error::StegoError;
use crate::stego::stc::embed::stc_embed;
use crate::stego::stc::extract::{stc_extract, stc_extract_prefix};
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

/// Encode `message` (plus optional `files`) into a stego H.264 Annex-B
/// stream using the OpenH264 backend with C.8.3-11 visual_recon cascade
/// safety.
///
/// The encode flow:
/// 1. Encrypt the payload with the passphrase (AES-256-GCM-SIV +
///    Argon2id key derive).
/// 2. Wrap in the standard phasm v1/v2 frame format.
/// 3. Baseline encode → walk → walker-aligned `PositionKey` cover.
/// 4. STC plan over `cover[0..m_total * w]` with `w = n_cover / m_total`.
/// 5. Re-encode with an `enc_pre_emit` hook that translates encoder hook
///    fires to canonical `PositionKey` and applies the planned flips.
///
/// Returns the raw Annex-B bytes ready to mux into MP4 (or store raw).
///
/// # Arguments
/// * `yuv` — raw YUV420p bytes, `width * height * 3 / 2 * n_frames` long.
/// * `width`, `height` — 16-aligned encode dimensions.
/// * `n_frames` — number of frames in `yuv`.
/// * `opts` — encoder knobs (QP, intra_period).
/// * `message` — UTF-8 text to embed (typically short; <1 KB).
/// * `files` — file attachments embedded alongside the text.
/// * `passphrase` — derives the AES key + STC hhat seed.
///
/// # Errors
/// * `StegoError::InvalidVideo` — dims not 16-aligned, yuv length wrong.
/// * `StegoError::MessageTooLarge` — cover capacity insufficient.
/// * Encryption / encoding failures bubble up as their respective
///   `StegoError` variants.
pub fn openh264_stego_encode_yuv_string(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    message: &str,
    files: &[payload::FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;

    // 1. Build the encrypted, framed payload bits.
    let payload_bytes = payload::encode_payload(message, files)?;
    let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct);
    let frame_bits = bytes_to_bits_msb_first(&frame_bytes);

    // 2. Derive the per-domain hhat seed from passphrase.
    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    // 3-6. Per-chunk 2-pass: baseline encode → walk → STC → stego encode.
    encode_yuv_with_pre_framed_bits(yuv, width, height, n_frames, opts, &frame_bits, &hhat_seed)
}

/// D.0.7.2 — exposed core of the OH264 stego encode for the streaming
/// session. Takes already-framed-and-encrypted payload bits + the
/// passphrase-derived STC seed, runs the 2-pass (baseline encode →
/// walker → STC plan → stego encode), returns the stego Annex-B.
///
/// The one-shot wrapper [`openh264_stego_encode_yuv_string`] computes
/// `frame_bits` from a UTF-8 message + AES-256-GCM-SIV encryption +
/// phasm v1 frame format. The streaming session (per-GOP) computes
/// the same bits but with an additional `chunk_frame` header
/// (`stego::chunk_frame`) wrapping the chunk-of-payload-bytes.
///
/// # Errors
/// * [`StegoError::InvalidVideo`] — dims not 16-aligned, yuv length
///   wrong, cover empty, or encoder failure.
/// * [`StegoError::MessageTooLarge`] — `n_cover / m_total < 1`
///   (chunk too big for this carrier).
pub(crate) fn encode_yuv_with_pre_framed_bits(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    frame_bits: &[u8],
    hhat_seed: &[u8; 32],
) -> Result<Vec<u8>, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;
    let m_total = frame_bits.len();

    let trace = perf_trace();
    let t_start = if trace { Some(std::time::Instant::now()) } else { None };

    // Baseline encode + walker for cover capture.
    let mb_width = width / 16;
    let mb_height = height / 16;
    let mb_per_frame = (mb_width * mb_height) as usize;
    let _ = mb_height; // dim sanity already done in validate_dims
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // Pass 1: cover probe. No overrides will fire, pVisualRecPic is not
    // needed (bitstream is walked then discarded). C.9.0 (#482) disables
    // the entire visual_recon mirror pool for this pass — ~30-50% encode
    // wall-clock savings, byte-identical bitstream.
    let t_pass1 = if trace { Some(std::time::Instant::now()) } else { None };
    let baseline_bitstream = encode_once(
        yuv, width, height, n_frames, opts,
        override_map.clone(),
        mb_type_table.clone(),
        applied.clone(),
        mb_width,
        mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        None,
        None, // P3.3a: no DPB correction on Pass 1
        None, // P3.3b.4: no coefficient replay on Pass 1
    )?;
    let dt_pass1 = t_pass1.map(|t| t.elapsed());
    let pass1_bytes = baseline_bitstream.len();

    let t_walk = if trace { Some(std::time::Instant::now()) } else { None };
    let baseline_walk = walk_annex_b_for_cover(&baseline_bitstream)
        .map_err(|e| StegoError::InvalidVideo(format!("baseline walk: {e}")))?;
    let dt_walk = t_walk.map(|t| t.elapsed());
    let baseline_positions = baseline_walk.cover.coeff_sign_bypass.positions;
    let baseline_bits = baseline_walk.cover.coeff_sign_bypass.bits;
    let n_cover = baseline_bits.len();

    if n_cover == 0 {
        return Err(StegoError::InvalidVideo("openh264 cover empty".into()));
    }
    if m_total == 0 {
        return Err(StegoError::InvalidVideo("empty frame bits".into()));
    }

    // Compute STC params: w = n_cover / m_total (must be >= 2 in practice
    // for the h=4 Viterbi to find non-trivial plans). The cover slice
    // used is m_total * w bits; remaining cover stays untouched.
    let w = n_cover / m_total;
    if w == 0 {
        return Err(StegoError::MessageTooLarge);
    }
    let used_cover = m_total * w;
    let cover_slice: Vec<u8> = baseline_bits[..used_cover].to_vec();

    // STC plan.
    let t_stc = if trace { Some(std::time::Instant::now()) } else { None };
    let costs: Vec<f32> = vec![1.0; used_cover];
    let hhat = generate_hhat(STC_H, w, hhat_seed);
    let plan = stc_embed(&cover_slice, &costs, frame_bits, &hhat, STC_H, w)
        .ok_or(StegoError::MessageTooLarge)?;
    let dt_stc = t_stc.map(|t| t.elapsed());

    // Build the override map keyed by canonical PositionKey::raw().
    let t_overrides = if trace { Some(std::time::Instant::now()) } else { None };
    let mut overrides_map: HashMap<u64, u8> = HashMap::new();
    for i in 0..used_cover {
        if plan.stego_bits[i] != cover_slice[i] {
            overrides_map.insert(baseline_positions[i].raw(), plan.stego_bits[i]);
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

    // Pass 2: stego encode with overrides. C.8 cascade-break needs the
    // visual_recon mirror so pDecPic stays clean for next-frame ME —
    // dual_recon=true here.
    let t_pass2 = if trace { Some(std::time::Instant::now()) } else { None };
    let result = encode_once(
        yuv, width, height, n_frames, opts,
        override_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ true,
        PassMode::Passthrough,
        None,
        None, // P3.3a: legacy dual_recon path — DPB correction not applicable
        None, // P3.3b.4: legacy 1-domain path — no coefficient replay
    );
    let dt_pass2 = t_pass2.map(|t| t.elapsed());

    if trace {
        let dt_total = t_start.unwrap().elapsed();
        eprintln!(
            "[PHASM_PERF_TRACE oh264_1domain] {w}x{h} f={f} n_cover={nc} m_total={mt} w={ww} flips={nf} pass1_bs={pb}",
            w = width, h = height, f = n_frames,
            nc = n_cover, mt = m_total, ww = w, nf = n_flips, pb = pass1_bytes,
        );
        let report = |label: &str, dt: Option<std::time::Duration>| {
            if let Some(d) = dt {
                let ms = d.as_secs_f64() * 1000.0;
                let pct = ms / (dt_total.as_secs_f64() * 1000.0) * 100.0;
                eprintln!("[PHASM_PERF_TRACE]   {label:<22} {ms:>9.1} ms  ({pct:>5.1}%)");
            }
        };
        report("Pass 1 OH264 encode", dt_pass1);
        report("walker (1-domain)", dt_walk);
        report("STC plan", dt_stc);
        report("override map build", dt_overrides);
        report("Pass 2 OH264 stego", dt_pass2);
        eprintln!(
            "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms  (100.0%)",
            "TOTAL", dt_total.as_secs_f64() * 1000.0,
        );
    }

    result
}

/// #493.4 Phase 3 — OH264 4-domain per-GOP encode primitive.
///
/// Same shape as [`encode_yuv_with_pre_framed_bits`] but embeds
/// `frame_bits` across all 4 stego domains (CoeffSign, CoeffSuffix,
/// MvdSign, MvdSuffix) via a single combined-cover STC plan. The
/// override map is keyed by canonical `PositionKey::raw()` across
/// all 4 domains; the OH264 fork's per-domain emit hooks look up
/// by the same key and apply the planned bit.
///
/// Phase 0 walker-symmetry parity gates (#493.1) verified that the
/// canonical keys match between OH264 fork emit + Rust walker for
/// all 4 (engine, domain) pairs, so the override-map lookup works
/// uniformly.
///
/// Cost vector: OH264 has no Pass-1-side per-position content-adaptive
/// cost (unlike pure-Rust which gets J-UNIWARD-style costs from
/// `pass1_count_with_mode`). We use uniform 1.0 baseline distortion
/// here; per-domain `CostWeights` × 1.0 = the domain weight, so STC's
/// allocation is driven purely by domain weight (per Phase 0.5
/// finding: cascade is binary, so domain weight is the right lever
/// at this resolution). v1.2 research can layer in content-adaptive
/// distortion costs if Phase 5 stealth gates require finer control.
///
/// **WIRING STATUS (Phase 3 / 2026-05-16):** primitive is callable
/// directly but NOT yet wired into the streaming session
/// (`oh264_finish` still uses the CS-only `encode_yuv_with_pre_framed_bits`).
/// Phase 4 / #493.5 swaps the call site once the matching combined-
/// cover decoder lands.
pub fn encode_yuv_with_pre_framed_bits_4domain(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    frame_bits: &[u8],
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
) -> Result<Vec<u8>, StegoError> {
    encode_yuv_with_pre_framed_bits_4domain_with_tier(
        yuv, width, height, n_frames, opts, frame_bits, hhat_seed, weights,
        super::stego::tier_filter::CascadeTier::Auto,
        super::stego::tier_filter::DEFAULT_HEADROOM,
    )
    .map(|(bytes, _tier)| bytes)
}

/// D'.3-OH264 (#795) — explicit cascade-tier variant of
/// [`encode_yuv_with_pre_framed_bits_4domain`]. Mirrors the pure-Rust
/// `h264_stego_encode_yuv_4domain_scheme_a_with_tier` (encode_pixels.rs).
///
/// Tier filter sets ∞-cost on CSB/CSL positions whose estimated pixel
/// impact exceeds the tier threshold. STC steers around them. Wire
/// format is unchanged — decoder is tier-agnostic.
#[allow(clippy::too_many_arguments)]
pub fn encode_yuv_with_pre_framed_bits_4domain_with_tier(
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
    validate_dims(yuv, width, height, n_frames)?;
    let m_total = frame_bits.len();
    if m_total == 0 {
        return Err(StegoError::InvalidVideo("empty frame bits".into()));
    }

    // #548 v1.0 BLOCKER fix (2026-05-18) — Reset libencoder-side fork
    // statics before every encode session. Without this, two
    // sequential `encode_yuv_with_pre_framed_bits_4domain` calls
    // share fork-side state across the encoder-instance teardown:
    // Call 2 produces 541 ChromaAc CS Sign diffs vs Call 1's 0
    // diffs on the same YUV. The bypass scratch + last-MB sentinels
    // + wire-only flag all carry state that breaks REPLAY-byte-
    // identity guarantees on subsequent calls. Reproducer:
    // `oh264_wire_only_two_sequential_calls_bisect` in
    // `core/tests/oh264_streaming_530_repro.rs`.
    unsafe { core_openh264_sys::phasm_reset_encoder_session_state() };

    let trace = perf_trace();
    let t_start = if trace { Some(std::time::Instant::now()) } else { None };

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
    )?;
    let captured_coeffs = take_coeff_capture();
    let dt_pass1 = t_pass1.map(|t| t.elapsed());
    let pass1_bytes = baseline_bitstream.len();

    let t_walk = if trace { Some(std::time::Instant::now()) } else { None };
    let baseline_walk = walk_annex_b_for_cover_with_options(
        &baseline_bitstream,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("baseline walk: {e}")))?;
    let dt_walk = t_walk.map(|t| t.elapsed());

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
    let t_stc = if trace { Some(std::time::Instant::now()) } else { None };
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
    let dt_stc = t_stc.map(|t| t.elapsed());

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

    // Pass 2 — re-encode with overrides + C.8 dual_recon cascade-break.
    //
    // #538 Phase 4.6 EXPERIMENT: setting `PHASM_USE_WIRE_ONLY=1` flips
    // the OH264 fork into wire-only override mode for Pass 2. In that
    // mode `apply_coeff_hooks_to_level` + `phasm_apply_mvd_hooks`
    // populate a per-MB scratch table instead of mutating *level / MV
    // state. The CABAC emit reads the scratch at the bypass-bin site
    // and writes the override bin directly to the wire. Encoder's
    // pDecPic stays clean by construction — no C.8.x cascade-break
    // needed for that mode, so `dual_recon` is also forced false.
    //
    // Default OFF: Pass 2 keeps the mutating + dual_recon path. This
    // gate is the measurement step for #530 — once validated, the
    // default flips and C.8.x machinery can be deleted (#539 Phase 5).
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
        Some(&captured_coeffs),
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
        report("Pass 1 OH264 encode", dt_pass1);
        report("walker (4-domain)", dt_walk);
        report("combine + STC plan", dt_stc);
        report("override map build", dt_overrides);
        report("Pass 2 OH264 stego", dt_pass2);
        eprintln!(
            "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms  (100.0%)  pass1_bs={} bytes",
            "TOTAL", dt_total.as_secs_f64() * 1000.0, pass1_bytes,
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
/// [`encode_yuv_with_pre_framed_bits`].
pub(crate) fn bytes_to_bits_msb_first_pub(bytes: &[u8]) -> Vec<u8> {
    bytes_to_bits_msb_first(bytes)
}

/// Convenience entry: encode a text-only message (no file attachments).
pub fn openh264_stego_encode_yuv_text(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    openh264_stego_encode_yuv_string(yuv, width, height, n_frames, opts, message, &[], passphrase)
}

/// Decode a stego Annex-B stream produced by `openh264_stego_encode_yuv_string`.
///
/// Walks the bitstream with the phasm CABAC walker, brute-forces
/// `m_total` over byte-aligned increments, and on each candidate runs
/// STC extract → frame parse (CRC oracle) → decrypt → payload decode.
///
/// Returns the recovered `PayloadData` (text + file attachments).
pub fn openh264_stego_decode_yuv(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<payload::PayloadData, StegoError> {
    let trace = perf_trace();
    let t_start = if trace { Some(std::time::Instant::now()) } else { None };

    let t_walk = if trace { Some(std::time::Instant::now()) } else { None };
    let walk = walk_annex_b_for_cover(annex_b)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    let dt_walk = t_walk.map(|t| t.elapsed());
    let cover_bits = walk.cover.coeff_sign_bypass.bits;
    let n_cover = cover_bits.len();
    if n_cover == 0 {
        return Err(StegoError::InvalidVideo("empty cover".into()));
    }

    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    let min_m = frame::FRAME_OVERHEAD * 8;
    let max_m = frame::MAX_FRAME_BITS.min(n_cover);

    let t_search = if trace { Some(std::time::Instant::now()) } else { None };

    // Brute-force m_total search. Sequential with early termination.
    //
    // #516.B NEGATIVE (3 strategies tried 2026-05-17):
    //   1. naive par_iter().find_map_any: 49 → 56 ms (rayon default
    //      chunks too large, cancellation lag)
    //   2. par_chunks(8) + find_map_any: 49 → 51 ms (overhead at
    //      8000+ chunks)
    //   3. explicit outer-batched parallel: 49 → 51 ms (rayon batch
    //      setup overhead)
    //
    // Root cause: after #516.2's length-prefix early-reject, wrong-
    // try cost dropped to ~290 µs each, so parallel can shrink the
    // 16 ms of wrong-try work to ~2 ms. But the WINNING try with
    // full extract + Argon2id (key derive) + AES-GCM-SIV decrypt is
    // ~5-8 ms by itself and runs single-threaded by security design
    // (Argon2id is intentionally slow). Parallel wall is bounded
    // below by the winning-try sequential cost.
    //
    // Parallel gains would only emerge for LONG messages (1000+
    // tries) where wrong-try work dominates. Filed as deferred
    // follow-on if message-length distribution shifts.
    let mut m_total = min_m;
    let mut tries = 0usize;
    while m_total <= max_m {
        tries += 1;
        if let Some(plaintext) = try_decode_at(&cover_bits, &hhat_seed, m_total, passphrase) {
            if trace {
                let dt_search = t_search.unwrap().elapsed();
                let dt_total = t_start.unwrap().elapsed();
                eprintln!(
                    "[PHASM_PERF_TRACE oh264_decode] annex_b={} bytes n_cover={} tries={} m_hit={}",
                    annex_b.len(), n_cover, tries, m_total,
                );
                eprintln!(
                    "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms",
                    "walker", dt_walk.unwrap().as_secs_f64() * 1000.0,
                );
                eprintln!(
                    "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms  ({} tries)",
                    "STC search + decrypt", dt_search.as_secs_f64() * 1000.0, tries,
                );
                eprintln!(
                    "[PHASM_PERF_TRACE]   {:<22} {:>9.1} ms  (100.0%)",
                    "TOTAL", dt_total.as_secs_f64() * 1000.0,
                );
            }
            return Ok(plaintext);
        }
        m_total += 8;
    }
    Err(StegoError::FrameCorrupted)
}

/// Text-only convenience wrapper around [`openh264_stego_decode_yuv`].
pub fn openh264_stego_decode_yuv_string(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<String, StegoError> {
    openh264_stego_decode_yuv(annex_b, passphrase).map(|p| p.text)
}

/// Capacity primitive (#424) — encode the YUV once into a baseline H.264
/// stream, walk it, and report the maximum CoeffSign cover bits + the
/// max embeddable message length (in bytes, after framing overhead).
///
/// The reported byte count is the *theoretical* max with `w = 1`; STC
/// at `w = 1` is degenerate, so practical capacity is roughly half this.
/// A caller building a UI capacity meter should display `max / 2` as
/// the safe budget.
#[derive(Debug, Clone, Copy)]
pub struct OpenH264StegoCapacity {
    /// Total CoeffSign cover bits the walker recovers from a baseline
    /// encode of this YUV with these opts.
    pub cover_bits: usize,
    /// `(cover_bits / 8) - FRAME_OVERHEAD`. Upper bound on message
    /// bytes; practical limit is roughly half (STC needs `w >= 2`).
    pub max_message_bytes: usize,
}

/// Predict the cover capacity of a YUV stream by running one baseline
/// encode + walker. No stego overrides applied.
pub fn openh264_stego_capacity_yuv(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<OpenH264StegoCapacity, StegoError> {
    validate_dims(yuv, width, height, n_frames)?;

    let mb_width = width / 16;
    let mb_height = height / 16;
    let mb_per_frame = (mb_width * mb_height) as usize;

    let empty_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

    // C.9.0 (#482): capacity probe is a pure cover-walk; bitstream
    // discarded immediately. Disable dual_recon to skip visual_recon
    // mirror work — byte-identical bitstream.
    let bitstream = encode_once(
        yuv, width, height, n_frames, opts,
        empty_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        None,
        None, // P3.3a: capacity probe — no DPB correction
        None, // P3.3b.4: no coefficient replay
    )?;
    let walk = walk_annex_b_for_cover(&bitstream)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    let cover_bits = walk.cover.coeff_sign_bypass.bits.len();
    let max_message_bytes = (cover_bits / 8).saturating_sub(frame::FRAME_OVERHEAD);
    Ok(OpenH264StegoCapacity {
        cover_bits,
        max_message_bytes,
    })
}

/// #424 D.0.6 — per-GOP cover-bits probe for the streaming capacity
/// session. Encodes the given YUV chunk through the OH264 fork in
/// baseline mode (no overrides applied) and walks the emitted Annex-B
/// for CoeffSign cover positions. Returns the cover-bit count for the
/// chunk.
///
/// Used by `StreamingProbeSession::push_frame` once per GOP. The
/// emitted bitstream is discarded — only the position count matters.
/// Cost is ~equal to the actual stego encode's first pass (the STC
/// plan + override application is what makes the second pass slow,
/// not the OH264 encode itself).
// Retained for the CAP2.5 Phase-1 instant capacity estimate (cheap
// CoeffSign-only walk, no STC trial); the accurate per-GOP probe now uses
// `oh264_gop_capacity_per_tier` (CAP2.1), so this has no caller today.
#[allow(dead_code)]
pub(crate) fn count_cover_bits_for_gop(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<usize, StegoError> {
    validate_dims(gop_yuv, width, height, n_frames)?;
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let empty_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    // C.9.0 (#482): per-GOP cover-bits probe — bitstream discarded after
    // the walk. Disable dual_recon for the same reason as the capacity
    // probe above.
    let bitstream = encode_once(
        gop_yuv, width, height, n_frames, opts,
        empty_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        None,
        None, // P3.3a: per-GOP cover probe — no DPB correction
        None, // P3.3b.4: no coefficient replay
    )?;
    let walk = walk_annex_b_for_cover(&bitstream)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    Ok(walk.cover.coeff_sign_bypass.bits.len())
}

/// #796 Mode A — 4-domain OH264 baseline-walk capacity probe.
///
/// Same as [`openh264_stego_capacity_yuv`] but returns per-domain
/// counts so `h264_stego_capacity_4domain` can report numbers that
/// match the OH264 streaming session's actual per-GOP STC budget.
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
pub fn count_cover_4domain_oh264(
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
// (`encode_yuv_with_pre_framed_bits_4domain_with_tier`: content_costs →
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
    /// off the OH264 cover walk. Lets `h264_stego_capacity_4domain_oh264`
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
        let framed = match super::stego::chunk_frame::build_chunk_frame(0, 1, &body) {
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
    let header = super::stego::chunk_frame::CHUNK_HEADER_LEN;
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
pub(crate) fn oh264_gop_capacity_per_tier(
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
    let mb_width = width / 16;
    let mb_per_frame = ((width / 16) * (height / 16)) as usize;
    let empty_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));
    let bitstream = encode_once(
        gop_yuv, width, height, gop_n_frames, opts,
        empty_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        PassMode::Passthrough,
        None,
        None,
        None,
    )?;
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
/// `oh264_gop_capacity_per_tier`), so a small-payload clip's tail GOPs cost
/// ~plain-encode speed instead of a full stego encode.
///
/// MANDATORY reset: the streaming stego GOPs that precede the tail leave the
/// OH264 fork's per-session scratch state dirty (the stego path resets at its
/// entry — openh264_stego.rs:509 — but `encode_once` does NOT self-reset the
/// fork statics). Without this reset the first plain GOP would inherit the last
/// stego GOP's fork state (#548 class: two sequential calls sharing fork state
/// → spurious diffs). Reset here so every plain GOP encodes from a clean
/// session, exactly like the stego path.
#[cfg(feature = "openh264-backend")]
pub(crate) fn oh264_plain_encode_gop(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    opts: EncodeOpts,
) -> Result<Vec<u8>, StegoError> {
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
    )
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
) -> Result<Vec<u8>, StegoError> {
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
    let _session = StegoSession::register(handlers)
        .map_err(|e| StegoError::InvalidVideo(format!("openh264 session: {e}")))?;

    let mut encoder =
        Encoder::new_with_dual_recon(width as i32, height as i32, opts.qp, opts.intra_period, dual_recon)
            .map_err(encoder_err_to_stego)?;

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
        set_frame_num(frame);
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

fn try_decode_at(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    m_total: usize,
    passphrase: &str,
) -> Option<payload::PayloadData> {
    if m_total == 0 {
        return None;
    }
    let n_cover = cover_bits.len();
    if m_total > n_cover {
        return None;
    }
    let w = n_cover / m_total;
    if w == 0 {
        return None;
    }
    let used = m_total * w;
    let hhat = generate_hhat(STC_H, w, hhat_seed);

    // #516.2 perf — length-prefix early-reject.
    //
    // The phasm v1 frame's first 2 bytes are `plaintext_len: u16 BE`.
    // For ANY candidate `m_total`, only ONE specific u16 makes the
    // identity `(FRAME_OVERHEAD + plaintext_len) * 8 == m_total`
    // hold. For wrong `m_total`, the partial syndrome is essentially
    // random → prob(consistent) ≈ 1/65536. Almost every wrong
    // candidate rejects after just 16 partial-extract message bits
    // (= 16 * w syndrome XORs) instead of the full m_total * w.
    //
    // v2 sentinel (`first u16 == 0`, plaintext > 64 KB): extend
    // partial extract to 48 bits and check u32 length identity
    // `(FRAME_OVERHEAD_EXT + plaintext_len) * 8 == m_total`. Rare
    // path; the 16-bit fast path covers all messages ≤ 65535 bytes.
    let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat, w, 16);
    if prefix_bits.len() < 16 {
        return None;
    }
    let prefix_bytes = bits_to_bytes_msb_first(&prefix_bits);
    let v1_len = u16::from_be_bytes([prefix_bytes[0], prefix_bytes[1]]) as usize;
    let mut len_consistent =
        (frame::FRAME_OVERHEAD + v1_len) * 8 == m_total;

    if !len_consistent && v1_len == 0 {
        // v2 sentinel — pull more bits for u32 length field
        let ext = stc_extract_prefix(&cover_bits[..used], &hhat, w, 48);
        if ext.len() >= 48 {
            let ext_bytes = bits_to_bytes_msb_first(&ext);
            let v2_len = u32::from_be_bytes([
                ext_bytes[2], ext_bytes[3], ext_bytes[4], ext_bytes[5],
            ]) as usize;
            if v2_len > u16::MAX as usize
                && (frame::FRAME_OVERHEAD_EXT + v2_len) * 8 == m_total
            {
                len_consistent = true;
            }
        }
    }

    if !len_consistent {
        return None;
    }

    // Length-prefix consistent: do the full extract + CRC + decrypt
    // path. CRC kills the rare 1/65536 false-positive that snuck
    // past the length filter; AEAD then verifies key + integrity.
    let extracted = stc_extract(&cover_bits[..used], &hhat, w);
    let frame_bits = &extracted[..m_total.min(extracted.len())];
    let frame_bytes = bits_to_bytes_msb_first(frame_bits);
    let parsed = frame::parse_frame(&frame_bytes).ok()?;
    let plaintext =
        crypto::decrypt(&parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce).ok()?;
    payload::decode_payload(&plaintext).ok()
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

fn bits_to_bytes_msb_first(bits: &[u8]) -> Vec<u8> {
    let n_bytes = bits.len() / 8;
    let mut out = Vec::with_capacity(n_bytes);
    for byte_idx in 0..n_bytes {
        let mut byte = 0u8;
        for i in 0..8 {
            byte |= (bits[byte_idx * 8 + i] & 1) << (7 - i);
        }
        out.push(byte);
    }
    out
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
/// Round-trips through `h264_stego_smart_decode_video` (the unified
/// Scheme A decoder from STEGO.A.4): shadow_extract_all4_safe runs
/// first (cheap, AES-GCM-SIV auth-gated), Scheme A combined STC
/// extract handles the primary.
///
/// Wire format: identical to `encode_yuv_with_pre_framed_bits_4domain`
/// (Scheme A combined STC) plus shadow LSB injections at the cascade-
/// safe shadow positions. No new wire-format flags.
///
/// Multi-message video stego with plausible deniability. The primary
/// message is carried by STC over the 4-domain cover (same as the
/// no-shadow path). Each shadow message is carried by direct LSB
/// writes + Reed-Solomon parity at one of the [`SHADOW_PARITY_TIERS`]
/// rungs, with positions chosen by passphrase-derived priority over
/// the primary-emit cover (= the cover the decoder will see).
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
/// through to [`encode_yuv_with_pre_framed_bits_4domain`] — no
/// provisional pass, no cascade loop, single STC + emit.
#[allow(clippy::too_many_arguments)]
pub fn encode_yuv_with_n_shadows_with_pattern_and_files<'a>(
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
        apply_shadow_to_plan_all4, embed_shadow_lsb_all4,
        overlay_infinity_costs_all4, prepare_shadow_over_emit_cover_safe,
        shadow_extract_all4_safe, translate_shadow_state, ShadowState,
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
    let _setup = super::stego::encode_pixels::validate_n_shadow_inputs(
        yuv, width, height, n_frames as usize, pattern,
        primary_passphrase, shadows,
    )?;
    let primary = super::stego::encode_pixels::prep_n_shadow_primary_payload(
        primary_message, primary_files, primary_passphrase,
    )?;

    let primary_hhat_seed = primary.keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    // ── Empty-shadows shortcut: route to the primary-only orchestrator
    if shadows.is_empty() {
        return encode_yuv_with_pre_framed_bits_4domain(
            yuv, width, height, n_frames, opts,
            &primary.frame_bits, &primary_hhat_seed, weights,
        );
    }

    // ── Phase 2: Pass 1 — clean OH264 encode + walker capture ────
    unsafe { core_openh264_sys::phasm_reset_encoder_session_state() };

    let mb_width = width / 16;
    let mb_height = height / 16;
    let mb_per_frame = (mb_width * mb_height) as usize;
    let _ = mb_height;
    let override_map: Arc<Mutex<HashMap<u64, u8>>> = Arc::new(Mutex::new(HashMap::new()));
    let mb_type_table: Arc<Mutex<Vec<u8>>> = Arc::new(Mutex::new(vec![0xff; mb_per_frame]));
    let applied: Arc<Mutex<u32>> = Arc::new(Mutex::new(0));

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

    set_549_pass_idx(1);
    if wire_only {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(1); }
    }
    let baseline_bitstream = encode_once(
        yuv, width, height, n_frames, opts,
        override_map.clone(), mb_type_table.clone(), applied.clone(),
        mb_width, mb_per_frame,
        /* dual_recon = */ false,
        if wire_only { PassMode::Capture } else { PassMode::Passthrough },
        decision_cache.clone(),
        None, // P3.3a: shadow Pass 1 — no DPB correction
        None, // P3.3b.4: no coefficient replay on shadow Pass 1
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
    // in `encode_yuv_with_pre_framed_bits_4domain` for rationale.
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

    let m_total = primary.m_total;

    // Build provisional plan from cover_p1 (no shadow injection).
    let (provisional_plan, w_stc) = {
        let (combined_cover, combined_costs, boundaries) =
            combine_cover_4domain(cover_p1, &content_costs, weights);
        let n_cover = combined_cover.len();
        if n_cover == 0 {
            return Err(StegoError::InvalidVideo(
                "shadow encode: combined cover empty".into(),
            ));
        }
        let w = n_cover / m_total;
        if w == 0 {
            return Err(StegoError::MessageTooLarge);
        }
        let used_cover = m_total * w;
        let cover_slice: Vec<u8> = combined_cover[..used_cover].to_vec();
        let cost_slice: Vec<f32> = combined_costs[..used_cover].to_vec();
        let hhat = generate_hhat(STC_H, w, &primary_hhat_seed);
        let plan = stc_embed(&cover_slice, &cost_slice, &primary.frame_bits, &hhat, STC_H, w)
            .ok_or(StegoError::MessageTooLarge)?;
        let mut full_stego_bits = Vec::with_capacity(n_cover);
        full_stego_bits.extend_from_slice(&plan.stego_bits);
        full_stego_bits.extend_from_slice(&combined_cover[used_cover..]);
        debug_assert_eq!(full_stego_bits.len(), n_cover);
        (split_plan_4domain(&full_stego_bits, &boundaries), w)
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
    let bytes_prov = encode_once(
        yuv, width, height, n_frames, opts,
        override_map.clone(), mb_type_table.clone(), applied.clone(),
        mb_width, mb_per_frame,
        /* dual_recon = */ !wire_only,
        if wire_only { PassMode::Replay } else { PassMode::Passthrough },
        decision_cache.clone(),
        None, // P3.3a: shadow provisional Pass 2 — DPB correction deferred
        None, // P3.3b.4: no coefficient replay on shadow path
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
            .map(|s| prepare_shadow_over_emit_cover_safe(
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

        // (c-d) Build cover_for_stc + costs_for_stc with shadow LSBs
        // injected + ∞-cost overlay at shadow positions.
        // STEGO.A.3 — costs start from the Tier 3 content-adaptive
        // baseline; shadow ∞-overlay layers on top.
        let mut cover_for_stc = cover_p1.clone();
        let mut costs_for_stc = content_costs.clone();
        for state in &shadow_states_p1 {
            embed_shadow_lsb_all4(
                &mut cover_for_stc.coeff_sign_bypass.bits,
                &mut cover_for_stc.coeff_suffix_lsb.bits,
                &mut cover_for_stc.mvd_sign_bypass.bits,
                &mut cover_for_stc.mvd_suffix_lsb.bits,
                state,
            );
            overlay_infinity_costs_all4(
                &mut costs_for_stc.coeff_sign_bypass,
                &mut costs_for_stc.coeff_suffix_lsb,
                &mut costs_for_stc.mvd_sign_bypass,
                &mut costs_for_stc.mvd_suffix_lsb,
                state,
            );
        }

        // (e-f) Combine + primary STC over packed cover with shadow ∞-mask.
        let (combined_cover, combined_costs, boundaries) =
            combine_cover_4domain(&cover_for_stc, &costs_for_stc, weights);
        let n_cover = combined_cover.len();
        let used_cover = m_total * w_stc;
        if used_cover > n_cover {
            return Err(StegoError::MessageTooLarge);
        }
        let cover_slice: Vec<u8> = combined_cover[..used_cover].to_vec();
        let cost_slice: Vec<f32> = combined_costs[..used_cover].to_vec();
        let hhat = generate_hhat(STC_H, w_stc, &primary_hhat_seed);
        let plan = match stc_embed(
            &cover_slice, &cost_slice, &primary.frame_bits, &hhat, STC_H, w_stc,
        ) {
            Some(p) => p,
            None => continue, // STC infeasible at this parity; next rung
        };

        let mut full_stego_bits = Vec::with_capacity(n_cover);
        full_stego_bits.extend_from_slice(&plan.stego_bits);
        full_stego_bits.extend_from_slice(&combined_cover[used_cover..]);
        let mut domain_plan = split_plan_4domain(&full_stego_bits, &boundaries);

        // (g-h) Defensive shadow stamp — re-write shadow bits onto the
        // primary plan in case STC's allocation drifted them.
        for state in &shadow_states_p1 {
            apply_shadow_to_plan_all4(
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

        // (j) Pass 2 emit with shadow-aware override map.
        let bytes = encode_once(
            yuv, width, height, n_frames, opts,
            override_map.clone(), mb_type_table.clone(), applied.clone(),
            mb_width, mb_per_frame,
            /* dual_recon = */ !wire_only,
            if wire_only { PassMode::Replay } else { PassMode::Passthrough },
            decision_cache.clone(),
            None, // P3.3a: shadow per-layer Pass 2 — DPB correction deferred
            None, // P3.3b.4: no coefficient replay on shadow path
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
            if shadow_extract_all4_safe(
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

    // The full round-trip lives in the `openh264_stego_roundtrip`
    // integration-test binary (`core/tests/openh264_stego_roundtrip.rs`).
    // Lib tests can't host it: C.8.7's `phasm_set_mv_override_active`
    // fork-side global is process-wide, and the 1300+ other openh264-
    // touching lib tests share state with it under cargo's single test
    // binary, producing 1-2 wire-flip cascade leaks. In an isolated
    // integration-test binary the state is pristine and the round-trip
    // is byte-exact (C.8.12 corpus suite is the structural proof).

    #[test]
    fn capacity_reports_positive_bits() {
        let _g = SESSION_TEST_MUTEX.lock().unwrap();
        let yuv = synth_yuv(320, 240, 2);
        let opts = EncodeOpts { qp: 22, intra_period: 60 };
        let cap = openh264_stego_capacity_yuv(&yuv, 320, 240, 2, opts).expect("capacity");
        assert!(cap.cover_bits > 1000, "expect non-trivial cover, got {}", cap.cover_bits);
        assert!(cap.max_message_bytes > 0);
    }
}
