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

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use core_openh264_sys::{
    encoder_pos_to_phasm_position_key, PHASM_MB_TYPE_OTHER,
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
use super::stego::hook::EmbedDomain;
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
    )?;
    let dt_pass1 = t_pass1.map(|t| t.elapsed());
    let pass1_bytes = baseline_bitstream.len();

    let t_walk = if trace { Some(std::time::Instant::now()) } else { None };
    let baseline_walk = walk_annex_b_for_cover_with_options(
        &baseline_bitstream,
        WalkOptions { record_mvd: true, ..Default::default() },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("baseline walk: {e}")))?;
    let dt_walk = t_walk.map(|t| t.elapsed());

    // Combine 4-domain cover. Costs are dummy (1.0 baseline before
    // weight multiplication) since OH264 doesn't expose per-position
    // distortion costs. Per-domain weight drives allocation.
    let t_stc = if trace { Some(std::time::Instant::now()) } else { None };
    let dummy_costs = DomainCosts::default();
    let (combined_cover, combined_costs, boundaries) =
        combine_cover_4domain(&baseline_walk.cover, &dummy_costs, weights);
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
    let result = encode_once(
        yuv, width, height, n_frames, opts,
        override_map, mb_type_table, applied,
        mb_width, mb_per_frame,
        /* dual_recon = */ !wire_only,
        if wire_only { PassMode::Replay } else { PassMode::Passthrough },
        decision_cache.clone(),
    );
    if wire_only {
        unsafe { core_openh264_sys::phasm_set_use_wire_only_overrides(0) };
    }
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
                let (p2_combined, _, p2_boundaries) =
                    combine_cover_4domain(&p2_walk.cover, &dummy_costs, weights);
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

    result
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
    )?;
    let walk = walk_annex_b_for_cover(&bitstream)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    Ok(walk.cover.coeff_sign_bypass.bits.len())
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

    let mut out = vec![0u8; 4 * 1024 * 1024];
    let mut bitstream = Vec::with_capacity(2 * 1024 * 1024);
    set_pass_mode(pass_mode);
    let mut frame_err: Option<StegoError> = None;
    for frame in 0..n_frames {
        set_frame_num(frame);
        let base = (frame as usize) * frame_total;
        let y = &yuv[base..base + frame_y];
        let u = &yuv[base + frame_y..base + frame_y + frame_uv];
        let v = &yuv[base + frame_y + frame_uv..base + frame_total];
        match encoder.encode_frame(y, u, v, (frame as i64) * 33, &mut out) {
            Ok((_, n)) => bitstream.extend_from_slice(&out[..n]),
            Err(e) => {
                frame_err = Some(encoder_err_to_stego(e));
                break;
            }
        }
    }
    set_pass_mode(PassMode::Passthrough);
    if let Some(e) = frame_err {
        return Err(e);
    }
    Ok(bitstream)
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
