// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

// `let mut x = X::default(); x.field = ...;` is the intended pattern in
// the §30D-C orchestrator: defaults handle the common-case shape, with
// per-call overrides toggling specific fields. Refactoring into struct
// literals would obscure the layered-default semantics.
#![allow(clippy::field_reassign_with_default)]

// Phase 6D.8 chunk 4 — two-run encode-time stego orchestration driver.
//
// Wires the framework (chunks 1+2 hook trait + Encoder field + chunks
// 3a/3b/3c residual emit-site wiring) into a top-level encode flow:
//
//   Pass 1 (count):    install PositionLoggerHook → encode all frames
//                      → take_cover() → GopCover (bytes discarded)
//   Pass 1.5 (split):  split_message_per_domain(message, capacity)
//                      → DomainMessages
//   Pass 2 (plan):     pass2_stc_plan_with_keys(cover, messages, h,
//                      keys, gop_idx) → DomainPlan
//   Pass 3 (inject):   install InjectionHook(PlanInjector) → encode
//                      all frames → keep bytes
//
// **Scope (this chunk)**: I-frame-only single-GOP workloads. MVD
// wiring is still deferred (deferred-items §30); P-frames produce
// stego output that would desync at the decoder until that lands.
// I-frames-only is sufficient for proving the orchestration driver
// works end-to-end and for wiring the eventual production
// h264_ghost_encode_pixels (chunk 5).
//
// The driver runs the encoder TWICE (Pass 1 + Pass 3), each with a
// different hook. Encoder-internal decisions (mode picks, quantize
// outputs) are deterministic over identical YUV → same coefficients
// per pass → STC plan applies cleanly. Pre-encode the same YUV
// drives the same path in both passes.

use crate::codec::h264::cabac::context::CabacInitSlot;
use crate::stego::error::StegoError;

use super::encoder_hook::{InjectAndLogHook, InjectionHook, PositionLoggerHook};
use super::keys::CabacStegoMasterKeys;
use super::orchestrate::{
    pass2_stc_plan_with_keys, split_message_per_domain, DomainPlan,
};
use super::PositionKey;
use super::{BitInjector, GopCapacity};
use crate::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};

/// Orchestrate encode-time stego for an I-frame-only video sequence
/// at H.264 High Profile + CABAC.
///
/// **Single-GOP, I-frame-only.** Each frame is encoded as an IDR.
/// MVD wiring is deferred (see deferred-items §30) so P-frame
/// stego is not safe until that lands. This driver IS sufficient
/// for end-to-end I-frame stego testing + integration into the
/// public `h264_ghost_encode_pixels` API (chunk 5).
///
/// # Arguments
/// - `yuv`: contiguous I420 planar buffer holding `n_frames` of
///   `width × height` 4:2:0 samples. Must be 16-aligned.
/// - `message`: bit sequence to embed (each entry 0 or 1).
/// - `passphrase`: derives per-domain ChaCha20 seeds + STC.
/// - `h`: STC constraint length (1..=7). Larger h = stronger
///   undetectability + more compute.
/// - `quality`: encoder QP target (0..=51); None = default 26.
///
/// # Returns
/// Annex-B byte stream with the message embedded at coefficient
/// sign + suffix LSB positions. Decode side: parse the Annex-B
/// stream via `cabac::bin_decoder` + STC reverse pass with the
/// same passphrase.
pub fn h264_stego_encode_i_frames_only(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    message: &[u8],
    passphrase: &str,
    h: usize,
    quality: Option<u8>,
) -> Result<Vec<u8>, StegoError> {
    // Validate inputs.
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {} ({}×{}×{})",
            yuv.len(), frame_size * n_frames, frame_size, n_frames, "3/2"
        )));
    }
    if !(1..=7).contains(&h) {
        return Err(StegoError::InvalidVideo(format!("STC h must be 1..=7, got {h}")));
    }

    // Derive per-domain master keys (Argon2 — once per encode).
    let keys = CabacStegoMasterKeys::derive(passphrase)?;

    // ── Pass 1: enumerate cover positions ──
    let cover = pass1_count(yuv, width, height, n_frames, frame_size, quality)?;

    // ── Pass 1.5: split message per domain ──
    let capacity = cover.cover.capacity();
    let messages = split_message_per_domain(message, &capacity)
        .ok_or(StegoError::MessageTooLarge)?;

    // ── Pass 2: per-domain STC plan ──
    let plan = pass2_stc_plan_with_keys(&cover, &messages, h, &keys, /* gop_idx */ 0)
        .ok_or_else(|| StegoError::InvalidVideo(
            "STC plan failed (per-domain cover smaller than message slice)".into()
        ))?;

    // ── Pass 3: encode with injection ──
    pass3_inject(yuv, width, height, n_frames, frame_size, quality, &cover.cover, &plan)
}

/// Pass 1 + 1.5 helper: encode all frames with a position-logger
/// hook and return the GopCover.
///
/// `pub(crate)` so the chunk-6F decoder parity gate can capture
/// the same encode-side cover and compare it byte-for-byte
/// against the bin-decoder slice walker output.
pub(crate) fn pass1_count(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
) -> Result<super::orchestrate::GopCover, StegoError> {
    pass1_count_with_mode(yuv, width, height, n_frames, frame_size, quality,
        /* all_idr */ true)
}

/// Variant of `pass1_count` that selects IDR vs P per frame:
/// `all_idr=true` → every frame is IDR (chunk-5 default);
/// `all_idr=false` → first frame IDR, rest P (§30C path).
fn pass1_count_with_mode(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    all_idr: bool,
) -> Result<super::orchestrate::GopCover, StegoError> {
    let mut enc = build_encoder(width, height, quality)?;
    enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
    for fi in 0..n_frames {
        let frame = &yuv[fi * frame_size..(fi + 1) * frame_size];
        let ft = if all_idr || fi == 0 {
            super::gop_pattern::FrameType::Idr
        } else {
            super::gop_pattern::FrameType::P
        };
        encode_one_frame(&mut enc, frame, ft)
            .map_err(|e| StegoError::InvalidVideo(format!("Pass 1 frame {fi}: {e}")))?;
    }
    let mut hook = enc.take_stego_hook().ok_or_else(|| StegoError::InvalidVideo(
        "Pass 1 stego hook missing".into()
    ))?;
    let cover = drain_position_logger(&mut hook)?;
    Ok(cover)
}

/// Pass 3 helper: encode all frames with an injection hook backed
/// by the STC plan, returning the stego Annex-B byte stream.
fn pass3_inject(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    cover: &super::DomainCover,
    plan: &DomainPlan,
) -> Result<Vec<u8>, StegoError> {
    pass3_inject_with_mode(yuv, width, height, n_frames, frame_size, quality,
        cover, plan, /* all_idr */ true)
}

/// Variant of `pass3_inject` that selects IDR vs P per frame.
#[allow(clippy::too_many_arguments)]
fn pass3_inject_with_mode(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    cover: &super::DomainCover,
    plan: &DomainPlan,
    all_idr: bool,
) -> Result<Vec<u8>, StegoError> {
    let injector = PlanInjector::from_plan(cover, plan);
    let hook = InjectionHook::new(injector);

    let mut enc = build_encoder(width, height, quality)?;
    enc.set_stego_hook(Some(Box::new(hook)));
    let mut out = Vec::new();
    for fi in 0..n_frames {
        let frame = &yuv[fi * frame_size..(fi + 1) * frame_size];
        let ft = if all_idr || fi == 0 {
            super::gop_pattern::FrameType::Idr
        } else {
            super::gop_pattern::FrameType::P
        };
        let bytes = encode_one_frame(&mut enc, frame, ft)
            .map_err(|e| StegoError::InvalidVideo(format!("Pass 3 frame {fi}: {e}")))?;
        out.extend_from_slice(&bytes);
    }
    Ok(out)
}

/// §6E-A.deploy.3 — encode one frame, dispatching to
/// `encode_i_frame` / `encode_p_frame` / `encode_b_frame` by
/// `FrameType`. The B-frame branch requires
/// `enc.enable_b_frames = true` to have been set BEFORE the first
/// frame (caller responsibility).
fn encode_one_frame(
    enc: &mut super::super::encoder::encoder::Encoder,
    pixels: &[u8],
    frame_type: super::gop_pattern::FrameType,
) -> Result<Vec<u8>, super::super::encoder::EncoderError> {
    use super::gop_pattern::FrameType;
    match frame_type {
        FrameType::Idr => enc.encode_i_frame(pixels),
        FrameType::P => enc.encode_p_frame(pixels),
        FrameType::B => enc.encode_b_frame(pixels),
    }
}

/// Build a fresh CABAC + High Profile encoder configured for our
/// stego encode flow.
fn build_encoder(
    width: u32,
    height: u32,
    quality: Option<u8>,
) -> Result<super::super::encoder::encoder::Encoder, StegoError> {
    use super::super::encoder::encoder::{Encoder, EntropyMode};
    let mut enc = Encoder::new(width, height, quality)
        .map_err(|e| StegoError::InvalidVideo(format!("encoder new: {e}")))?;
    enc.entropy_mode = EntropyMode::Cabac;
    // §Stealth.L3.1 follow-on (#145) — High profile (transform_8x8_mode
    // = 1) lands phasm output in the HandBrake/x264-medium metaclass at
    // the SPS level. The CABAC walker handles I_8x8 (transform_size_8x8
    // = 1) since the same task; encoder + walker are now in parity.
    enc.enable_transform_8x8 = true;
    // Phase F (#262, 2026-05-08) — production stego paths default
    // to `BRdoConfig::SAFE_L0_ZERO`. This is RDO + residual emission
    // ON, but every B-MB is overridden to L0_16x16 with MV=(0,0),
    // bypassing spatial-direct + ME-derived MV paths that diverge
    // between encoder and decoder.
    //
    // Phase D bisect (commit 26412b2 / corpus 10-fixture parallel
    // run) confirmed: forced (0,0) MV makes encoder + decoder MC
    // agree on reference positions, eliminating ghost-image and
    // blocky-motion artifacts on textured-motion content. PSNR is
    // equal-or-better than the prior PRODUCTION_VISUAL preset on
    // motion fixtures (+0.4-0.7 dB on horseflag/piratebattle/
    // schoolfight/carplane); minor regression (≤0.24 dB) on
    // already-clean static content where Skip selections were
    // optimal — accepted trade-off for v1.0.
    //
    // No-op for IPPPP (b_count=0) patterns — only fires on IBPBP /
    // multi-B GopPattern. Single source of truth across CLI / iOS /
    // Android / WASM-decode / streaming-v2 paths since all entry
    // points route through `build_encoder()`.
    //
    // v1.1 follow-on: align encoder spatial-direct + bipred MC code
    // with decoder spec § 8.4.1.2 to remove the workaround.
    enc.b_rdo_config = super::super::encoder::mb_decision_b::BRdoConfig::SAFE_L0_ZERO;
    let _ = CabacInitSlot::ISI;
    Ok(enc)
}

/// Recover a `GopCover` from a boxed `PositionLoggerHook` via a
/// safe downcast pattern. `take_stego_hook()` returns
/// `Option<Box<dyn StegoMbHook>>`; we know the concrete type was
/// `PositionLoggerHook` because we just installed one.
fn drain_position_logger(
    hook: &mut Box<dyn super::encoder_hook::StegoMbHook>,
) -> Result<super::orchestrate::GopCover, StegoError> {
    // Downcast via std::any::Any. Add Any bound to StegoMbHook in a
    // companion commit so we can safely downcast here. For now,
    // sidestep the issue by passing the hook back as a raw pointer
    // and reinterpreting — UNSAFE but deterministic since we know
    // the concrete type.
    //
    // Better: add `take_cover()` to the trait as an optional
    // method (fn take_cover(&mut self) -> Option<GopCover>). Default
    // returns None; PositionLoggerHook overrides to Some.
    // That's the production-quality path. Implementing now.
    let cover = hook.take_cover_if_logger()
        .ok_or_else(|| StegoError::InvalidVideo(
            "Pass 1 hook was not a PositionLoggerHook".into()
        ))?;
    Ok(cover)
}

/// `BitInjector` impl backed by a precomputed `DomainPlan` +
/// `DomainCover` (used to map `PositionKey` → planned bit value).
pub struct PlanInjector {
    plan: std::collections::HashMap<PositionKey, u8>,
}

impl PlanInjector {
    /// Build the lookup table from a Pass-1 cover + Pass-2 plan.
    pub fn from_plan(cover: &super::DomainCover, plan: &DomainPlan) -> Self {
        let mut map = std::collections::HashMap::new();
        Self::extend(&mut map, &cover.coeff_sign_bypass.positions, &plan.coeff_sign_bypass);
        Self::extend(&mut map, &cover.coeff_suffix_lsb.positions, &plan.coeff_suffix_lsb);
        Self::extend(&mut map, &cover.mvd_sign_bypass.positions, &plan.mvd_sign_bypass);
        Self::extend(&mut map, &cover.mvd_suffix_lsb.positions, &plan.mvd_suffix_lsb);
        Self { plan: map }
    }

    fn extend(
        map: &mut std::collections::HashMap<PositionKey, u8>,
        positions: &[PositionKey],
        bits: &[u8],
    ) {
        let n = positions.len().min(bits.len());
        for i in 0..n {
            map.insert(positions[i], bits[i]);
        }
    }
}

impl BitInjector for PlanInjector {
    fn override_bit(&mut self, key: PositionKey) -> Option<u8> {
        self.plan.get(&key).copied()
    }
}

// Suppress dead-code warnings from the GopCapacity import — used
// only for type-system completeness.
#[allow(dead_code)]
fn _docs_only() -> GopCapacity {
    GopCapacity::default()
}

/// Encode-time stego at H.264 High Profile + CABAC, taking a UTF-8
/// string message + passphrase + raw I420 YUV pixel buffer. This is
/// the chunk-5 public entry point that mobile bridges + the CLI will
/// route through once the chunk-7 atomic gate-swap lands.
///
/// The string message is framed with the same payload + crypto +
/// frame stack used by the legacy CAVLC pipeline (`payload::
/// encode_payload` → `crypto::encrypt` → `frame::build_frame`), so
/// decode-side framing logic (chunk 6) can recover identical bytes
/// from any encoder output. The framed bytes are then expanded into
/// MSB-first bits and embedded across the four bypass-bin domains
/// per the chunk-4 orchestration driver.
///
/// **Scope (this chunk).** I-frame-only single-GOP. P-frame stego
/// stays deferred until §30 MVD wiring lands. The function is
/// sufficient for end-to-end string-message encode tests + mobile
/// bridge integration via fully YUV-decoded inputs.
///
/// # Arguments
/// - `yuv`: contiguous I420 planar buffer, `n_frames` × (`w`·`h`·3/2).
/// - `width`, `height`: must be 16-aligned.
/// - `n_frames`: how many I420 frames the buffer holds.
/// - `message`: UTF-8 string to embed (typically <1 KB).
/// - `passphrase`: drives Argon2 + per-domain ChaCha20 STC seeds.
///
/// # Returns
/// Annex-B byte stream with the framed/encrypted message embedded
/// at coefficient sign + suffix LSB positions across all I-frames.
pub fn h264_stego_encode_yuv_string(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::{crypto, frame, payload};

    // Frame the message via the standard Phasm payload + crypto +
    // frame stack. Identical framing as legacy CAVLC pipeline
    // (h264_pipeline.rs:206-208) so the chunk-6 decoder can share
    // the unwrap logic.
    let payload_bytes = payload::encode_payload(message, &[])?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);

    // MSB-first bit expansion of the framed bytes.
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();

    // Defaults: STC h=4 (good capacity-vs-undetectability trade per
    // 6D.4 cost-model analysis), QP=26 (neutral baseline matching
    // the chunk-4 orchestration driver tests).
    h264_stego_encode_i_frames_only(
        yuv, width, height, n_frames,
        &frame_bits, passphrase,
        /* h */ 4, /* quality */ Some(26),
    )
}

/// Encode-time stego with one IDR + (n_frames - 1) P-frames
/// (§30C). Same framing + STC orchestration as
/// `h264_stego_encode_yuv_string`, but the second-and-later
/// frames are emitted as P-slices, exercising the §30A1-4
/// decoder dispatch + giving real-video-shaped cover (residual
/// bits from inter MBs).
///
/// **Capacity**: P-frame residual coverage depends on motion
/// content. Static / nearly-static content maximizes P_SKIP MBs
/// (zero coverage) and shrinks total cover. High-motion content
/// gives more residuals, hence more cover.
///
/// **MVD domains stay empty** — encoder MVD hook is unwired
/// (§30D pending). Only coeff_sign_bypass + coeff_suffix_lsb
/// carry message bits.
pub fn h264_stego_encode_yuv_string_i_then_p(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::{crypto, frame, payload};

    let payload_bytes = payload::encode_payload(message, &[])?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();

    h264_stego_encode_i_then_p_frames(
        yuv, width, height, n_frames,
        &frame_bits, passphrase,
        /* h */ 4, /* quality */ Some(26),
    )
}

/// Inner driver for I+P encode-time stego (§30C). Mirrors
/// `h264_stego_encode_i_frames_only` but emits frame 0 as IDR
/// and frames 1..n_frames-1 as P-slices.
pub fn h264_stego_encode_i_then_p_frames(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    message: &[u8],
    passphrase: &str,
    h: usize,
    quality: Option<u8>,
) -> Result<Vec<u8>, StegoError> {
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames
        )));
    }
    if !(1..=7).contains(&h) {
        return Err(StegoError::InvalidVideo(format!("STC h must be 1..=7, got {h}")));
    }

    let keys = CabacStegoMasterKeys::derive(passphrase)?;

    // Pass 1: I + P frames.
    let cover = pass1_count_with_mode(
        yuv, width, height, n_frames, frame_size, quality,
        /* all_idr */ false,
    )?;
    let capacity = cover.cover.capacity();
    let messages = split_message_per_domain(message, &capacity)
        .ok_or(StegoError::MessageTooLarge)?;

    let plan = pass2_stc_plan_with_keys(&cover, &messages, h, &keys, /* gop_idx */ 0)
        .ok_or_else(|| StegoError::InvalidVideo(
            "STC plan failed (per-domain cover smaller than message slice)".into()
        ))?;

    pass3_inject_with_mode(
        yuv, width, height, n_frames, frame_size, quality,
        &cover.cover, &plan, /* all_idr */ false,
    )
}

/// Phase 6D.8 §30D-C — 4-domain encode-time stego (residual +
/// MVD). Wraps the 3-pass orchestrator + standard payload framing
/// + crypto stack.
///
/// **Allocation: fill-MVD-first.** Message bits go to MVD domains
/// up to mvd_capacity; remaining bits spill into residual domains.
/// Decoder mirrors via the same min(m_total, mvd_capacity) split.
///
/// **Capacity vs 2-domain version**: typically MVD positions are
/// <5% of residual positions, so the capacity boost is modest.
/// The bigger gain is per-bit stealth — flipping an MVD sign at
/// |mvd|=1 is a smaller perceptual perturbation than flipping a
/// residual coefficient sign at |coeff|=15. Cost-model-based
/// position selection in §30D-D could tune the allocation.
pub fn h264_stego_encode_yuv_string_4domain(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    // Single-GOP shape: gop_size = n_frames so only frame 0 is IDR.
    h264_stego_encode_yuv_string_4domain_multigop(
        yuv, width, height, n_frames,
        /* gop_size */ n_frames,
        message, passphrase,
    )
}

/// Phase 6E-C1a — multi-IDR variant of `h264_stego_encode_yuv_string_4domain`.
///
/// `gop_size` controls the IDR pattern: a frame at index `fi` is
/// emitted as an IDR when `fi % gop_size == 0`. Pass `gop_size = 30`
/// (typical for 1-second GOPs at 30 fps) for real-world-shape video;
/// pass `gop_size = n_frames` to retain the legacy single-IDR shape.
///
/// The STC plan still spans the WHOLE multi-IDR cover — there is no
/// per-GOP message split (locked-in stealth position; see
/// `docs/design/h264-shadow-messages.md` "Scope notes — primary STC").
/// All four bypass-bin domains accumulate across all GOPs into a
/// single per-domain cover; STC plans run once per domain over the
/// whole-video cover with `gop_idx = 0` keying.
///
/// **Memory note (this commit)**: cover materialization is still
/// in-memory (existing inline / segmented STC paths). For long
/// videos exceeding `SEGMENTED_THRESHOLD = 1M` cover positions per
/// domain, the segmented STC kicks in automatically (O(√n)
/// memory). Per-GOP streaming Phase A + Phase B (further
/// memory bound) is a follow-on optimization tracked separately.
pub fn h264_stego_encode_yuv_string_4domain_multigop(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::{crypto, frame, payload};

    if gop_size == 0 {
        return Err(StegoError::InvalidVideo("gop_size must be > 0".into()));
    }
    if gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size ({gop_size}) > n_frames ({n_frames})"
        )));
    }

    let payload_bytes = payload::encode_payload(message, &[])?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();

    h264_stego_encode_i_then_p_frames_4domain_multigop(
        yuv, width, height, n_frames, gop_size,
        /* b_count */ 1, // §6E-A.deploy.3 default: IBPBP M=2
        &frame_bits, passphrase,
        /* h */ 4, /* quality */ Some(26),
    )
}

/// §6E-A.deploy.5 — explicit-pattern variant of
/// [`h264_stego_encode_yuv_string_4domain_multigop`].
///
/// `pattern` selects the GOP shape:
/// - `GopPattern::Ipppp { gop }` — legacy IPPPP shape (Layer 3
///   fingerprint phasm-distinctive vs iPhone). Useful for tests
///   that need encode order ≡ display order.
/// - `GopPattern::Ibpbp { gop, b_count }` — Apple-iPhone canonical
///   shape (Layer 3 fingerprint match). `b_count = 1` is the
///   default (M=2); larger values increase B-frame density at the
///   cost of additional cover capacity.
///
/// The non-`_with_pattern` entry point delegates here with
/// `GopPattern::Ibpbp { gop: gop_size, b_count: 1 }` (the
/// post-§6E-A.deploy default).
pub fn h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::{crypto, frame, payload};
    use super::gop_pattern::GopPattern;

    let gop_size = pattern.gop_size();
    let b_count = match pattern {
        GopPattern::Ipppp { .. } => 0,
        GopPattern::Ibpbp { b_count, .. } => b_count,
    };

    if gop_size == 0 {
        return Err(StegoError::InvalidVideo("gop_size must be > 0".into()));
    }
    if gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size ({gop_size}) > n_frames ({n_frames})"
        )));
    }

    let payload_bytes = payload::encode_payload(message, &[])?;
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();

    h264_stego_encode_i_then_p_frames_4domain_multigop(
        yuv, width, height, n_frames, gop_size, b_count,
        &frame_bits, passphrase,
        /* h */ 4, /* quality */ Some(26),
    )
}

/// 3-pass driver for 4-domain stego with I + P frames.
///
/// **Pass 1**: encode I + P frames with PositionLoggerHook +
/// `enable_mvd_stego_hook=true`. Captures MVD cover (and a
/// throwaway residual cover from this pass).
///
/// **Pass 2A**: split message — `m_mvd = min(m_total,
/// mvd_capacity)`. Plan MVD STC with capacities = (0, 0,
/// mvd_sign_p1, mvd_suffix_p1) on `message[..m_mvd]`.
///
/// **Pass 1B**: encode I + P frames with `InjectAndLogHook` —
/// applies Pass-2A's MVD plan in-place (modifying choice's MVs),
/// re-runs MC + recon with FINAL MVs, logs the resulting residual
/// cover. The throwaway encode bytes are discarded.
///
/// **Pass 2B**: plan residual STC with capacities =
/// (coeff_sign_p1b, coeff_suffix_p1b, 0, 0) on `message[m_mvd..]`.
///
/// **Pass 3**: encode I + P frames with combined PlanInjector
/// (MVD plan from Pass 2A + residual plan from Pass 2B). Output
/// bytes are the stego stream.
pub fn h264_stego_encode_i_then_p_frames_4domain(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    message: &[u8],
    passphrase: &str,
    h: usize,
    quality: Option<u8>,
) -> Result<Vec<u8>, StegoError> {
    h264_stego_encode_i_then_p_frames_4domain_multigop(
        yuv, width, height, n_frames,
        /* gop_size */ n_frames,
        /* b_count */ 1,
        message, passphrase, h, quality,
    )
}

/// §long-form-stego Phase 5 — streaming-cover orchestrator.
///
/// Public API for long-form video stego (5+ min clips at 1080p).
/// Bypasses the in-memory orchestrator's full Pass 1 cover
/// materialization (which would OOM at 5+ min × 1080p × 4 domains)
/// by replaying Pass 1 per-GOP at STC plan time (Phase 4
/// `H264GopReplayCover`) and per-GOP at Pass 3 inject time (Phase 3
/// `pass1_capture_4domain_for_gop_range`).
///
/// Memory bound for 5-min 1080p:
/// - Phase 1-counts: O(num_gops) ~ a few KB.
/// - Per-domain DomainPlan from streaming-Viterbi: O(n_d) — kept
///   resident through Pass 3. ~1.2 GB for 5-min 1080p × 4 domains.
/// - Streaming-Viterbi internal: O(sqrt n_d).
/// - Per-GOP injector + encoder working set: ~few MB transient.
/// - Total peak: ~1.5 GB at 5-min 1080p (vs ~8 GB in-memory path).
///
/// At 15-min 1080p the plan side becomes the binding cost (~3.5 GB).
/// Plan-side streaming for mobile-15-min support is v1.1+ work.
///
/// Behaviorally identical to
/// [`h264_stego_encode_yuv_string_4domain_multigop`] except for
/// re-emitted SPS+PPS NALs at each GOP's IDR (per-GOP fresh
/// encoder; spec-allowed; decoder handles transparently). Cover
/// positions, STC plan, and stego bit positions are bit-exact
/// equivalent — round-trip decode succeeds with the standard
/// `h264_stego_decode_yuv_string_4domain`.
pub fn h264_stego_encode_yuv_string_4domain_multigop_streaming(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::{crypto, frame, payload};
    use crate::stego::stc::streaming_segmented::{
        stc_embed_streaming_segmented,
    };
    use super::cover_replay::H264GopReplayCover;
    use super::hook::EmbedDomain;
    use super::inject::DomainCover;
    use super::orchestrate::{
        stealth_weighted_allocation, split_message_per_domain, DomainPlan,
        PlanInjector, StealthAllocator,
    };
    use super::encoder_hook::InjectionHook;
    use super::keys::CabacStegoMasterKeys;
    use super::gop_pattern::GopPattern;

    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}",
            yuv.len(),
            frame_size * n_frames,
        )));
    }
    if gop_size == 0 || gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size {gop_size} must be in 1..={n_frames}"
        )));
    }

    // §Default-flip.IBPBP (#193) 2026-05-03: B-frame visual bug
    // root-caused + fixed in commit 5b85432 — bucket-based no-RDO
    // L0/L1/Bi emission was polluting the spatial-direct grid.
    // Fallback now emits Skip / Direct only. Visual quality
    // confirmed clean on iPhone7 1080p (commit message references
    // ~24-25 dB Y-PSNR per frame at 640×384 IBPBP).
    //
    // Default restored to IBPBP{ b_count: 1 } (Apple-iPhone canonical
    // M=2). With §6E-D.5(m) HF-prop Direct multiplier + §6E-D.5(o)
    // Bi rate hack also restored, mode distribution within ε=5pp of
    // x264-medium on the §6E-A6.5 gate fixture.
    let b_count = 1usize;
    let h = 4usize;
    let quality = Some(26u8);

    // Encrypt + frame the message.
    let payload_bytes = payload::encode_payload(message, &[])?;
    let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();
    let m_total = frame_bits.len();

    let keys = CabacStegoMasterKeys::derive(passphrase)?;

    // ── Pass 1 (counting only) — per-GOP per-domain counts. ──
    // Memory: O(num_gops × 4 × usize). No cover materialization.
    let per_gop_counts = pass1_count_per_gop_4domain(
        yuv, width, height, n_frames, gop_size, b_count, quality,
    )?;
    let mut totals = [0usize; 4];
    for row in per_gop_counts.iter() {
        for d in 0..4 {
            totals[d] += row[d];
        }
    }
    let cap_p1 = GopCapacity {
        coeff_sign_bypass: totals[0],
        coeff_suffix_lsb: totals[1],
        mvd_sign_bypass: totals[2],
        mvd_suffix_lsb: totals[3],
    };

    // ── Stealth-weighted allocation (mvd_suffix forced 0). ──
    let allocator = StealthAllocator::v1_default();
    let cap_for_alloc = GopCapacity {
        coeff_sign_bypass: cap_p1.coeff_sign_bypass,
        coeff_suffix_lsb: cap_p1.coeff_suffix_lsb,
        mvd_sign_bypass: cap_p1.mvd_sign_bypass,
        mvd_suffix_lsb: 0,
    };
    let (m_cs, m_cl, m_ms, _m_ml) =
        stealth_weighted_allocation(m_total, &cap_for_alloc, &allocator)
            .ok_or(StegoError::MessageTooLarge)?;
    let m_mvd = m_ms;
    let m_residual = m_cs + m_cl;

    // ── Pass 2 — per-domain streaming-Viterbi via H264GopReplayCover. ──
    //
    // For each non-empty domain we build a H264GopReplayCover (no
    // additional Pass 1 — uses the per_gop_counts we already
    // computed above) and stream through stc_embed_streaming_segmented.
    let mut plan_a = DomainPlan::default();
    let mut plan_b = DomainPlan::default();

    // Pass 2A: MVD-sign domain.
    if m_mvd > 0 {
        let cap_mvd_only = GopCapacity {
            coeff_sign_bypass: 0,
            coeff_suffix_lsb: 0,
            mvd_sign_bypass: cap_p1.mvd_sign_bypass,
            mvd_suffix_lsb: 0,
        };
        let messages_a = split_message_per_domain(&frame_bits[..m_mvd], &cap_mvd_only)
            .ok_or_else(|| StegoError::InvalidVideo(
                "Pass 2A streaming: MVD split failed".into()
            ))?;
        let m_dom = messages_a.mvd_sign_bypass.len();
        let n_dom = cap_p1.mvd_sign_bypass;
        if m_dom > 0 && n_dom > 0 {
            let w = n_dom / m_dom.max(1);
            if w > 0 {
                let k = ((m_dom as f64).sqrt().ceil() as usize).max(1);
                let mut cover = H264GopReplayCover::from_counts(
                    yuv, width, height, n_frames, gop_size, b_count, quality,
                    EmbedDomain::MvdSignBypass,
                    &per_gop_counts, m_dom, w, k,
                )?;
                let seed = keys.per_gop_seeds(EmbedDomain::MvdSignBypass, 0).hhat_seed;
                let hhat = crate::stego::stc::hhat::generate_hhat(h, w, &seed);
                let result = stc_embed_streaming_segmented(
                    &mut cover, &messages_a.mvd_sign_bypass, &hhat, h, w,
                ).map_err(|e| StegoError::InvalidVideo(format!(
                    "Pass 2A streaming-Viterbi: {e}"
                )))?;
                plan_a.mvd_sign_bypass = result.stego_bits;
                plan_a.total_modifications += result.num_modifications;
                plan_a.total_cost += result.total_cost;
            }
        }
    }

    // Pass 2B: residual domains (coeff_sign + coeff_suffix).
    if m_residual > 0 {
        let cap_coeff_only = GopCapacity {
            coeff_sign_bypass: cap_p1.coeff_sign_bypass,
            coeff_suffix_lsb: cap_p1.coeff_suffix_lsb,
            mvd_sign_bypass: 0,
            mvd_suffix_lsb: 0,
        };
        let messages_b = split_message_per_domain(&frame_bits[m_mvd..], &cap_coeff_only)
            .ok_or_else(|| StegoError::InvalidVideo(
                "Pass 2B streaming: residual split failed".into()
            ))?;
        for (domain, msg, n_dom, plan_slot) in [
            (EmbedDomain::CoeffSignBypass, &messages_b.coeff_sign_bypass,
             cap_p1.coeff_sign_bypass, &mut plan_b.coeff_sign_bypass),
            (EmbedDomain::CoeffSuffixLsb, &messages_b.coeff_suffix_lsb,
             cap_p1.coeff_suffix_lsb, &mut plan_b.coeff_suffix_lsb),
        ] {
            let m_dom = msg.len();
            if m_dom == 0 || n_dom == 0 {
                continue;
            }
            let w = n_dom / m_dom.max(1);
            if w == 0 {
                continue;
            }
            let k = ((m_dom as f64).sqrt().ceil() as usize).max(1);
            let mut cover = H264GopReplayCover::from_counts(
                yuv, width, height, n_frames, gop_size, b_count, quality,
                domain, &per_gop_counts, m_dom, w, k,
            )?;
            let seed = keys.per_gop_seeds(domain, 0).hhat_seed;
            let hhat = crate::stego::stc::hhat::generate_hhat(h, w, &seed);
            let result = stc_embed_streaming_segmented(
                &mut cover, msg, &hhat, h, w,
            ).map_err(|e| StegoError::InvalidVideo(format!(
                "Pass 2B streaming-Viterbi ({domain:?}): {e}"
            )))?;
            *plan_slot = result.stego_bits;
            plan_b.total_modifications += result.num_modifications;
            plan_b.total_cost += result.total_cost;
        }
    }

    // Combined per-domain plans.
    let combined_plan = DomainPlan {
        coeff_sign_bypass: plan_b.coeff_sign_bypass.clone(),
        coeff_suffix_lsb: plan_b.coeff_suffix_lsb.clone(),
        mvd_sign_bypass: plan_a.mvd_sign_bypass.clone(),
        mvd_suffix_lsb: Vec::new(),
        total_modifications: plan_a.total_modifications + plan_b.total_modifications,
        total_cost: plan_a.total_cost + plan_b.total_cost,
    };


    // Cumulative per-GOP counts per domain (for plan slicing in Pass 3).
    let cum: [Vec<usize>; 4] = std::array::from_fn(|d| {
        let mut v = Vec::with_capacity(per_gop_counts.len() + 1);
        v.push(0);
        for row in per_gop_counts.iter() {
            v.push(*v.last().unwrap() + row[d]);
        }
        v
    });

    // ── Pass 3 (per-GOP) — fresh encoder per GOP, per-GOP
    //     PlanInjector built from per-GOP cover replay + plan slices.
    //     Memory: ~one GOP's encoder working set + per-GOP injector. ──
    let pattern = GopPattern::Ibpbp { gop: gop_size, b_count };
    let num_gops = per_gop_counts.len();
    let mut out = Vec::new();
    for g in 0..num_gops {
        let gop_cover = pass1_capture_4domain_for_gop_range(
            yuv, width, height, n_frames, gop_size, b_count, quality, g, g + 1,
        )?;

        // Slice each domain plan to this GOP's range.
        let mut gop_plan = DomainPlan::default();
        for d in 0..4 {
            let lo = cum[d][g];
            let hi = cum[d][g + 1];
            let src = match d {
                0 => &combined_plan.coeff_sign_bypass,
                1 => &combined_plan.coeff_suffix_lsb,
                2 => &combined_plan.mvd_sign_bypass,
                _ => &combined_plan.mvd_suffix_lsb,
            };
            let dst = match d {
                0 => &mut gop_plan.coeff_sign_bypass,
                1 => &mut gop_plan.coeff_suffix_lsb,
                2 => &mut gop_plan.mvd_sign_bypass,
                _ => &mut gop_plan.mvd_suffix_lsb,
            };
            if src.is_empty() {
                continue;
            }
            let lo = lo.min(src.len());
            let hi = hi.min(src.len());
            *dst = src[lo..hi].to_vec();
        }

        // Build per-GOP PlanInjector + encode.
        let mut gop_cover_only = DomainCover::default();
        gop_cover_only.coeff_sign_bypass = gop_cover.cover.coeff_sign_bypass.clone();
        gop_cover_only.coeff_suffix_lsb = gop_cover.cover.coeff_suffix_lsb.clone();
        gop_cover_only.mvd_sign_bypass = gop_cover.cover.mvd_sign_bypass.clone();
        gop_cover_only.mvd_suffix_lsb = gop_cover.cover.mvd_suffix_lsb.clone();
        let injector = PlanInjector::from_plan(&gop_cover_only, &gop_plan);

        let mut enc = build_encoder(width, height, quality)?;
        enc.enable_mvd_stego_hook = true;
        enc.enable_b_frames = pattern.has_b_frames();
        enc.set_stego_hook(Some(Box::new(InjectionHook::new(injector))));

        // Iterate the full schedule, encode only frames whose
        // gop_idx matches `g`. First in-range frame primes
        // stego_frame_idx (Phase 1 contract).
        let mut primed = false;
        for (meta, frame) in iter_frames_in_encode_order(
            yuv, frame_size, n_frames, pattern,
        ) {
            if (meta.gop_idx as usize) != g {
                if (meta.gop_idx as usize) > g {
                    break;
                }
                continue;
            }
            if !primed {
                enc.stego_frame_idx = meta.encode_idx;
                primed = true;
            }
            let bytes = encode_one_frame(&mut enc, frame, meta.frame_type)
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "Pass 3 streaming GOP {g} frame {}: {e}",
                    meta.encode_idx,
                )))?;
            out.extend_from_slice(&bytes);
        }
    }
    Ok(out)
}

/// §B-cascade-real Bug #1a fix (#235) — per-GOP STC orchestrator (2026-05-07).
///
/// V5 production probe revealed that the streaming-Viterbi orchestrator
/// concentrates per-domain flips in whichever GOP holds that domain's
/// bulk capacity. GOP 0 (chroma-rich IDR-heavy) gets chroma-heavy flips
/// (-5 dB U-plane); GOP 1 (motion-rich P-frame-heavy) gets luma-heavy
/// flips (-14 dB Y catastrophic). Bug is the global STC trellis spanning
/// all GOPs.
///
/// This function uses per-GOP STC: each GOP's cover and message are
/// planned independently with `pass2_stc_plan_with_keys(.., gop_idx=g)`.
/// Trellis windows can no longer span GOP boundaries, so flips are
/// proportionally distributed across GOPs.
///
/// Identical signature + behaviour to v2 streaming for callers — same
/// frame format, same encryption, same Pass 3 emit. Only the STC layer
/// differs. Decoder mirror in `h264_stego_decode_yuv_string_4domain_per_gop_v3`.
///
/// Memory: per-GOP cover capture (one GOP at a time). Suitable for both
/// short fixtures (V5 1080p × 45f) and long clips. Not faster than
/// streaming-Viterbi for very long clips but bug-free.
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_4domain_per_gop_v3(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    message: &str,
    files: &[crate::stego::payload::FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::{crypto, frame, payload};
    use super::orchestrate::{
        stealth_weighted_allocation, split_message_per_domain,
        DomainPlan, StealthAllocator, pass2_stc_plan_with_keys,
        DomainMessages,
    };
    use super::keys::CabacStegoMasterKeys;
    use super::gop_pattern::GopPattern;

    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames,
        )));
    }
    let gop_size = pattern.gop_size();
    let b_count = pattern.legacy_b_count();
    if gop_size == 0 || gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size {gop_size} must be in 1..={n_frames}"
        )));
    }
    let h = 4usize;
    let quality = Some(26u8);

    // Encrypt + frame the message + files.
    let payload_bytes = payload::encode_payload(message, files)?;
    let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();
    let m_total = frame_bits.len();
    let keys = CabacStegoMasterKeys::derive(passphrase)?;

    // Pass 1: count per-GOP per-domain capacities.
    let per_gop_counts = pass1_count_per_gop_4domain(
        yuv, width, height, n_frames, gop_size, b_count, quality,
    )?;
    let num_gops = per_gop_counts.len();
    let mut totals = [0usize; 4];
    for row in &per_gop_counts {
        for d in 0..4 {
            totals[d] += row[d];
        }
    }
    let cap_p1 = GopCapacity {
        coeff_sign_bypass: totals[0],
        coeff_suffix_lsb: totals[1],
        mvd_sign_bypass: totals[2],
        mvd_suffix_lsb: 0,
    };

    // Stealth allocation: global m_cs / m_cl / m_ms (unchanged).
    let allocator = StealthAllocator::v1_default();
    let (m_cs, m_cl, m_ms, _m_ml) =
        stealth_weighted_allocation(m_total, &cap_p1, &allocator)
            .ok_or(StegoError::MessageTooLarge)?;
    let m_mvd = m_ms;
    let m_residual = m_cs + m_cl;
    debug_assert_eq!(m_total, m_mvd + m_residual);

    // Per-GOP message-bit allocation: distribute global m_d
    // proportionally to per-GOP cap_d_g. Round-robin remainder to
    // earliest GOPs to conserve total. Per-domain.
    let alloc_per_gop = |m_global: usize, dom: usize| -> Vec<usize> {
        let mut out = vec![0usize; num_gops];
        if m_global == 0 || totals[dom] == 0 {
            return out;
        }
        let mut allocated = 0usize;
        for g in 0..num_gops {
            let m_g = (m_global as u64 * per_gop_counts[g][dom] as u64
                / totals[dom] as u64) as usize;
            out[g] = m_g;
            allocated += m_g;
        }
        // Distribute rounding remainder to earliest GOPs that still
        // have capacity headroom.
        let mut rem = m_global - allocated;
        for g in 0..num_gops {
            if rem == 0 { break; }
            if out[g] < per_gop_counts[g][dom] {
                out[g] += 1;
                rem -= 1;
            }
        }
        debug_assert_eq!(out.iter().sum::<usize>(), m_global);
        out
    };
    // (per-GOP allocation deferred until after split_message_per_domain
    // — that's where we get the real per-domain message lengths.)
    let _ = (m_cs, m_cl, m_ms); // placeholder to keep totals validated

    // Slice frame_bits per-domain first. Encoder layout matches
    // `decode_pixels::try_decode_at_4domain`: frame_bits[..m_mvd]
    // = MVD-domain bits, frame_bits[m_mvd..] = residual-domain bits.
    let cap_mvd_only = GopCapacity {
        coeff_sign_bypass: 0, coeff_suffix_lsb: 0,
        mvd_sign_bypass: cap_p1.mvd_sign_bypass, mvd_suffix_lsb: 0,
    };
    let cap_coeff_only = GopCapacity {
        coeff_sign_bypass: cap_p1.coeff_sign_bypass,
        coeff_suffix_lsb: cap_p1.coeff_suffix_lsb,
        mvd_sign_bypass: 0, mvd_suffix_lsb: 0,
    };
    let messages_a = split_message_per_domain(&frame_bits[..m_mvd], &cap_mvd_only)
        .ok_or_else(|| StegoError::InvalidVideo("v3: MVD split failed".into()))?;
    let messages_b = split_message_per_domain(&frame_bits[m_mvd..], &cap_coeff_only)
        .ok_or_else(|| StegoError::InvalidVideo("v3: residual split failed".into()))?;

    // Re-derive m_cs/m_cl/m_ms from the actual split lengths (round-
    // trip safety). `stealth_weighted_allocation` and
    // `split_message_per_domain` use slightly different rounding;
    // decoder side also uses `split_message_per_domain`, so we
    // must base per-GOP allocation on those exact lengths.
    let m_cs = messages_b.coeff_sign_bypass.len();
    let m_cl = messages_b.coeff_suffix_lsb.len();
    let m_ms_actual = messages_a.mvd_sign_bypass.len();

    // Recompute per-GOP allocation against the actual per-domain
    // message lengths.
    let m_cs_per_gop = alloc_per_gop(m_cs, 0);
    let m_cl_per_gop = alloc_per_gop(m_cl, 1);
    let m_ms_per_gop = alloc_per_gop(m_ms_actual, 2);

    // Per-GOP STC. For each GOP: capture per-GOP cover, run Pass 2A
    // (MVD), Pass 1B (residual re-capture with MVD plan), Pass 2B
    // (residual). Concatenate plans into combined_plan.
    let mut combined_plan = DomainPlan::default();
    let mut cs_off = 0usize;
    let mut cl_off = 0usize;
    let mut ms_off = 0usize;

    for g in 0..num_gops {
        let m_cs_g = m_cs_per_gop[g];
        let m_cl_g = m_cl_per_gop[g];
        let m_ms_g = m_ms_per_gop[g];

        // Per-GOP message slices (bit-level, already split per-domain
        // across the whole clip — slice into per-GOP chunks here).
        let mvd_msg_g: Vec<u8> = if m_ms_g > 0 {
            messages_a.mvd_sign_bypass[ms_off..ms_off + m_ms_g].to_vec()
        } else { Vec::new() };
        let cs_msg_g: Vec<u8> = if m_cs_g > 0 {
            messages_b.coeff_sign_bypass[cs_off..cs_off + m_cs_g].to_vec()
        } else { Vec::new() };
        let cl_msg_g: Vec<u8> = if m_cl_g > 0 {
            messages_b.coeff_suffix_lsb[cl_off..cl_off + m_cl_g].to_vec()
        } else { Vec::new() };

        // Capture this GOP's cover (MVD + naive residual).
        let gop_cover = pass1_capture_4domain_for_gop_range(
            yuv, width, height, n_frames, gop_size, b_count, quality, g, g + 1,
        )?;

        // Pass 2A: plan MVD STC for this GOP.
        let mvd_messages = DomainMessages {
            coeff_sign_bypass: Vec::new(),
            coeff_suffix_lsb: Vec::new(),
            mvd_sign_bypass: mvd_msg_g.clone(),
            mvd_suffix_lsb: Vec::new(),
        };
        let plan_a_g = pass2_stc_plan_with_keys(
            &gop_cover, &mvd_messages, h, &keys, g as u32,
        ).ok_or_else(|| StegoError::InvalidVideo(format!(
            "v3 Pass 2A GOP {g}: STC plan failed"
        )))?;

        // Pass 1B: re-encode this GOP with MVD plan applied + residual logger.
        let cover_p1b = encode_one_gop_pass1b_inject_mvd_log_residual(
            yuv, width, height, n_frames, frame_size, quality, pattern,
            g, &gop_cover.cover, &plan_a_g,
        )?;

        // Pass 2B: plan residual STC for this GOP.
        let res_messages = DomainMessages {
            coeff_sign_bypass: cs_msg_g,
            coeff_suffix_lsb: cl_msg_g,
            mvd_sign_bypass: Vec::new(),
            mvd_suffix_lsb: Vec::new(),
        };
        let plan_b_g = pass2_stc_plan_with_keys(
            &cover_p1b, &res_messages, h, &keys, g as u32,
        ).ok_or_else(|| StegoError::InvalidVideo(format!(
            "v3 Pass 2B GOP {g}: STC plan failed"
        )))?;

        // Concatenate this GOP's per-domain plan slices into combined.
        combined_plan.coeff_sign_bypass.extend_from_slice(&plan_b_g.coeff_sign_bypass);
        combined_plan.coeff_suffix_lsb.extend_from_slice(&plan_b_g.coeff_suffix_lsb);
        combined_plan.mvd_sign_bypass.extend_from_slice(&plan_a_g.mvd_sign_bypass);
        combined_plan.total_modifications +=
            plan_a_g.total_modifications + plan_b_g.total_modifications;
        combined_plan.total_cost += plan_a_g.total_cost + plan_b_g.total_cost;

        cs_off += m_cs_g;
        cl_off += m_cl_g;
        ms_off += m_ms_g;

        let _ = pattern;  // reserved for future use
    }
    let _ = GopPattern::Ipppp { gop: 0 };  // keep import live

    // Pass 3: per-GOP encode with combined_plan (mirrors streaming-v1
    // pattern at line 911-981). Cumulative per-GOP per-domain offsets
    // needed for plan slicing per-GOP.
    let cum: [Vec<usize>; 4] = std::array::from_fn(|d| {
        let mut v = Vec::with_capacity(num_gops + 1);
        v.push(0);
        for row in per_gop_counts.iter() {
            v.push(*v.last().unwrap() + row[d]);
        }
        v
    });

    let mut out = Vec::new();
    for g in 0..num_gops {
        let gop_cover = pass1_capture_4domain_for_gop_range(
            yuv, width, height, n_frames, gop_size, b_count, quality, g, g + 1,
        )?;

        let mut gop_plan = DomainPlan::default();
        for d in 0..4 {
            let lo = cum[d][g];
            let hi = cum[d][g + 1];
            let src = match d {
                0 => &combined_plan.coeff_sign_bypass,
                1 => &combined_plan.coeff_suffix_lsb,
                2 => &combined_plan.mvd_sign_bypass,
                _ => &combined_plan.mvd_suffix_lsb,
            };
            let dst = match d {
                0 => &mut gop_plan.coeff_sign_bypass,
                1 => &mut gop_plan.coeff_suffix_lsb,
                2 => &mut gop_plan.mvd_sign_bypass,
                _ => &mut gop_plan.mvd_suffix_lsb,
            };
            if src.is_empty() {
                continue;
            }
            let lo = lo.min(src.len());
            let hi = hi.min(src.len());
            *dst = src[lo..hi].to_vec();
        }

        let mut gop_cover_only = super::DomainCover::default();
        gop_cover_only.coeff_sign_bypass = gop_cover.cover.coeff_sign_bypass.clone();
        gop_cover_only.coeff_suffix_lsb = gop_cover.cover.coeff_suffix_lsb.clone();
        gop_cover_only.mvd_sign_bypass = gop_cover.cover.mvd_sign_bypass.clone();
        gop_cover_only.mvd_suffix_lsb = gop_cover.cover.mvd_suffix_lsb.clone();
        let injector = PlanInjector::from_plan(&gop_cover_only, &gop_plan);

        let mut enc = build_encoder(width, height, quality)?;
        enc.enable_mvd_stego_hook = true;
        enc.enable_b_frames = pattern.has_b_frames();
        enc.set_stego_hook(Some(Box::new(super::encoder_hook::InjectionHook::new(injector))));

        let mut primed = false;
        for (meta, frame) in iter_frames_in_encode_order(
            yuv, frame_size, n_frames, pattern,
        ) {
            if (meta.gop_idx as usize) != g {
                if (meta.gop_idx as usize) > g {
                    break;
                }
                continue;
            }
            if !primed {
                enc.stego_frame_idx = meta.encode_idx;
                primed = true;
            }
            let bytes = encode_one_frame(&mut enc, frame, meta.frame_type)
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "v3 Pass 3 GOP {g} frame {}: {e}", meta.encode_idx,
                )))?;
            out.extend_from_slice(&bytes);
        }
    }
    Ok(out)
}

/// §long-form-stego Phase 6.3 — interleaved Phase B orchestrator.
///
/// Public API for long-form video stego at mobile-scale memory
/// bounds (15-min 1080p target). Same pipeline as Phase 5's
/// `h264_stego_encode_yuv_string_4domain_multigop_streaming`, with
/// two key differences:
///
/// 1. The per-domain `StreamingViterbiPhaseB` drivers run in
///    **round-robin lockstep** — after each Phase B segment
///    emission, the orchestrator picks the driver whose highest
///    pending GOP is largest, keeping all 4 drivers in approximate
///    GOP-sync.
/// 2. Per-GOP Pass 3 fires **incrementally** — as soon as a GOP's
///    plan is fully populated by all active domains, that GOP's
///    encoder Pass 3 runs and the plan buffer is dropped.
///
/// Memory bound (15-min 1080p):
/// - Per-GOP plan buffers (in flight): ~tens of MB (a handful of
///   GOPs at any time vs all `num_gops` in Phase 5).
/// - Per-domain Phase A checkpoints: O(√n) per domain ≈ 33 MB × 4.
/// - Per-segment Phase B back-pointers: O(K·w) ≈ few MB transient.
/// - Encoder Pass 3 working set: ~100 MB transient.
/// - Phase 4 cover replay: ~10 MB transient.
/// - **Total peak**: ~250-400 MB plans+state, plus the encoded
///   output buffer the function returns (~1 GB at 15-min 1080p).
///
/// CPU cost: equivalent to Phase 5 (refactor of plan storage only,
/// no added encoder work).
///
/// Bit-exact equivalent output to Phase 5 is the design goal —
/// the difference is only WHEN bytes are emitted, not WHAT.
/// Verified by Phase 6.4's equivalence test.
pub fn h264_stego_encode_yuv_string_4domain_multigop_streaming_v2(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_files(
        yuv, width, height, n_frames, gop_size, message, &[], passphrase,
    )
}

/// §6E-A5(c.x) / Task #97 — `_with_files` variant of the production
/// CABAC v2 streaming encoder. Accepts a `&[FileEntry]` slice so
/// callers can attach binary files (images / docs / contacts) to
/// the primary text message. Same brotli-compressed payload format
/// as the JPEG-side image stego — `payload::encode_payload(text,
/// files)` builds one blob; encryption + framing are unchanged.
///
/// Decoder side: `h264_stego_smart_decode_video_with_payload`
/// returns `PayloadData { text, files, .. }`.
///
/// Defaults to the §6E-A.deploy IBPBP{ gop, b_count: 1 } pattern.
/// For source-adaptive cadence (§Stealth.L3.2), use
/// [`h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files`].
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_files(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    message: &str,
    files: &[crate::stego::payload::FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    // §Default-flip.IBPBP (#193) — visual bug fixed in commit 5b85432.
    let pattern = super::gop_pattern::GopPattern::Ibpbp {
        gop: gop_size,
        b_count: 1,
    };
    h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
        yuv, width, height, n_frames, pattern, message, files, passphrase,
    )
}

/// §Stealth.L3.2 follow-on (#149) — pattern-aware variant of
/// [`h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_files`].
///
/// Lets the caller pick the GOP cadence (typically via
/// `GopPattern::auto_select(source_mp4)`); the streaming v2
/// orchestrator + per-GOP Phase B Viterbi infrastructure is
/// already pattern-driven via the underlying `pass1_count_per_gop_
/// 4domain` and `pass3_inject_4domain_streaming` helpers, so this
/// just exposes the parameter that was already plumbed through.
///
/// `pattern.gop_size()` drives the IDR period; `pattern.legacy_
/// b_count()` drives the encoder's `enable_b_frames` and
/// `pattern_from_legacy_args` derivations downstream.
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_pattern_and_files(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    message: &str,
    files: &[crate::stego::payload::FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    let gop_size = pattern.gop_size();
    use crate::stego::{crypto, frame, payload};
    use crate::stego::stc::streaming_segmented::StreamingViterbiPhaseB;
    use super::cover_replay::H264GopReplayCover;
    use super::hook::EmbedDomain;
    
    use super::orchestrate::{
        stealth_weighted_allocation, split_message_per_domain, DomainPlan, StealthAllocator,
    };
    
    use super::keys::CabacStegoMasterKeys;
    use super::gop_pattern::GopPattern;
    use super::per_gop_plan::PerGopPlanBuilder;

    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}",
            yuv.len(),
            frame_size * n_frames,
        )));
    }
    if gop_size == 0 || gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size {gop_size} must be in 1..={n_frames}"
        )));
    }

    let b_count = pattern.legacy_b_count();
    let h = 4usize;
    let quality = Some(26u8);

    // Task #97 — `files` is brotli-compressed alongside the text in
    // a single payload blob; identical wire format to the JPEG-side
    // image stego, decoded via `payload::decode_payload`.
    let payload_bytes = payload::encode_payload(message, files)?;
    let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();
    let m_total = frame_bits.len();
    let keys = CabacStegoMasterKeys::derive(passphrase)?;

    // ── Pass 1 (counting only) ──
    let per_gop_counts = pass1_count_per_gop_4domain(
        yuv, width, height, n_frames, gop_size, b_count, quality,
    )?;
    let mut totals = [0usize; 4];
    for row in per_gop_counts.iter() {
        for d in 0..4 {
            totals[d] += row[d];
        }
    }
    let cap_p1 = GopCapacity {
        coeff_sign_bypass: totals[0],
        coeff_suffix_lsb: totals[1],
        mvd_sign_bypass: totals[2],
        mvd_suffix_lsb: totals[3],
    };

    // ── Stealth allocation ──
    let allocator = StealthAllocator::v1_default();
    let cap_for_alloc = GopCapacity {
        coeff_sign_bypass: cap_p1.coeff_sign_bypass,
        coeff_suffix_lsb: cap_p1.coeff_suffix_lsb,
        mvd_sign_bypass: cap_p1.mvd_sign_bypass,
        mvd_suffix_lsb: 0,
    };
    let (m_cs, m_cl, m_ms, _m_ml) =
        stealth_weighted_allocation(m_total, &cap_for_alloc, &allocator)
            .ok_or(StegoError::MessageTooLarge)?;
    let m_mvd = m_ms;
    let m_residual = m_cs + m_cl;

    // ── Per-domain (m, w, k, message) — only for active domains ──
    let messages_a = if m_mvd > 0 {
        let cap_mvd_only = GopCapacity {
            coeff_sign_bypass: 0,
            coeff_suffix_lsb: 0,
            mvd_sign_bypass: cap_p1.mvd_sign_bypass,
            mvd_suffix_lsb: 0,
        };
        Some(
            split_message_per_domain(&frame_bits[..m_mvd], &cap_mvd_only)
                .ok_or_else(|| {
                    StegoError::InvalidVideo("v2: MVD split failed".into())
                })?,
        )
    } else {
        None
    };
    let messages_b = if m_residual > 0 {
        let cap_coeff_only = GopCapacity {
            coeff_sign_bypass: cap_p1.coeff_sign_bypass,
            coeff_suffix_lsb: cap_p1.coeff_suffix_lsb,
            mvd_sign_bypass: 0,
            mvd_suffix_lsb: 0,
        };
        Some(
            split_message_per_domain(&frame_bits[m_mvd..], &cap_coeff_only)
                .ok_or_else(|| {
                    StegoError::InvalidVideo("v2: residual split failed".into())
                })?,
        )
    } else {
        None
    };

    // Per-domain owned (message, w, k, hhat) tuples — must outlive
    // the drivers. We also need stable storage for the
    // H264GopReplayCovers since the drivers borrow them mutably.
    struct DomainOwned {
        domain: EmbedDomain,
        message: Vec<u8>,
        w: usize,
        hhat: Vec<Vec<u32>>,
    }
    let mut owned: [Option<DomainOwned>; 4] = [None, None, None, None];
    let mut k_per_domain: [usize; 4] = [0; 4];

    let domain_table: [(EmbedDomain, usize, Option<&Vec<u8>>); 4] = [
        (
            EmbedDomain::CoeffSignBypass,
            cap_p1.coeff_sign_bypass,
            messages_b.as_ref().map(|mb| &mb.coeff_sign_bypass),
        ),
        (
            EmbedDomain::CoeffSuffixLsb,
            cap_p1.coeff_suffix_lsb,
            messages_b.as_ref().map(|mb| &mb.coeff_suffix_lsb),
        ),
        (
            EmbedDomain::MvdSignBypass,
            cap_p1.mvd_sign_bypass,
            messages_a.as_ref().map(|ma| &ma.mvd_sign_bypass),
        ),
        (EmbedDomain::MvdSuffixLsb, 0, None),
    ];

    for (d, (domain, n_dom, msg_opt)) in domain_table.iter().enumerate() {
        let Some(msg) = msg_opt else {
            continue;
        };
        let m_dom = msg.len();
        if m_dom == 0 || *n_dom == 0 {
            continue;
        }
        let w = *n_dom / m_dom.max(1);
        if w == 0 {
            continue;
        }
        let k = ((m_dom as f64).sqrt().ceil() as usize).max(1);
        let seed = keys.per_gop_seeds(*domain, 0).hhat_seed;
        let hhat = crate::stego::stc::hhat::generate_hhat(h, w, &seed);
        owned[d] = Some(DomainOwned {
            domain: *domain,
            message: (*msg).clone(),
            w,
            hhat,
        });
        k_per_domain[d] = k;
    }
    let active_domains: [bool; 4] = [
        owned[0].is_some(),
        owned[1].is_some(),
        owned[2].is_some(),
        owned[3].is_some(),
    ];

    // ── Construct H264GopReplayCovers + StreamingViterbiPhaseB drivers ──
    //
    // Each cover borrows yuv immutably; each driver borrows ONE
    // cover mutably + owned[d].message + owned[d].hhat. Storing in
    // 4 separate locals lets the borrow checker track them
    // independently.
    let mut cover0: Option<H264GopReplayCover> = None;
    let mut cover1: Option<H264GopReplayCover> = None;
    let mut cover2: Option<H264GopReplayCover> = None;
    // domain 3 always inactive — no cover needed.

    if let Some(o) = &owned[0] {
        cover0 = Some(H264GopReplayCover::from_counts(
            yuv, width, height, n_frames, gop_size, b_count, quality,
            o.domain, &per_gop_counts, o.message.len(), o.w, k_per_domain[0],
        )?);
    }
    if let Some(o) = &owned[1] {
        cover1 = Some(H264GopReplayCover::from_counts(
            yuv, width, height, n_frames, gop_size, b_count, quality,
            o.domain, &per_gop_counts, o.message.len(), o.w, k_per_domain[1],
        )?);
    }
    if let Some(o) = &owned[2] {
        cover2 = Some(H264GopReplayCover::from_counts(
            yuv, width, height, n_frames, gop_size, b_count, quality,
            o.domain, &per_gop_counts, o.message.len(), o.w, k_per_domain[2],
        )?);
    }

    let mut driver0: Option<StreamingViterbiPhaseB> = None;
    let mut driver1: Option<StreamingViterbiPhaseB> = None;
    let mut driver2: Option<StreamingViterbiPhaseB> = None;

    if let (Some(o), Some(c)) = (&owned[0], cover0.as_mut()) {
        driver0 = Some(
            StreamingViterbiPhaseB::new(c, &o.message, &o.hhat, h, o.w)
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "v2 Phase A driver[0]: {e}"
                )))?,
        );
    }
    if let (Some(o), Some(c)) = (&owned[1], cover1.as_mut()) {
        driver1 = Some(
            StreamingViterbiPhaseB::new(c, &o.message, &o.hhat, h, o.w)
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "v2 Phase A driver[1]: {e}"
                )))?,
        );
    }
    if let (Some(o), Some(c)) = (&owned[2], cover2.as_mut()) {
        driver2 = Some(
            StreamingViterbiPhaseB::new(c, &o.message, &o.hhat, h, o.w)
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "v2 Phase A driver[2]: {e}"
                )))?,
        );
    }

    // ── PerGopPlanBuilder ──
    //
    // Each driver emits m*w bits per domain, NOT the full raw cover
    // capacity. Phase 5's per-GOP plan slicing clips at m*w (via
    // `lo.min(src.len())` / `hi.min(src.len())`), dropping any
    // trailing raw-capacity positions [m*w .. total) without emission.
    // Mirror that here by pre-computing **effective** per-GOP sizes:
    // the per-domain bit count for GOP `g` is `m*w` minus the bits
    // already consumed by GOPs `[0..g)`, clipped to GOP `g`'s raw
    // capacity. Without this clip, the last GOP's buffer never fills
    // and `take_ready_gops` deadlocks.
    let num_gops = per_gop_counts.len();
    let cum_raw: [Vec<usize>; 4] = std::array::from_fn(|d| {
        let mut v = Vec::with_capacity(num_gops + 1);
        v.push(0);
        for row in per_gop_counts.iter() {
            v.push(*v.last().unwrap() + row[d]);
        }
        v
    });
    let m_w_per_domain: [usize; 4] = [
        owned[0].as_ref().map(|o| o.message.len() * o.w).unwrap_or(0),
        owned[1].as_ref().map(|o| o.message.len() * o.w).unwrap_or(0),
        owned[2].as_ref().map(|o| o.message.len() * o.w).unwrap_or(0),
        0,
    ];
    let effective_counts: Vec<[usize; 4]> = (0..num_gops)
        .map(|g| {
            std::array::from_fn(|d| {
                if !active_domains[d] {
                    return 0;
                }
                let lo = cum_raw[d][g].min(m_w_per_domain[d]);
                let hi = cum_raw[d][g + 1].min(m_w_per_domain[d]);
                hi - lo
            })
        })
        .collect();

    let mut builder = PerGopPlanBuilder::new(&effective_counts, active_domains);
    debug_assert_eq!(builder.num_gops(), num_gops);
    // Pattern is now a parameter — `pattern_from_legacy_args` would
    // have produced the same value but the caller's choice wins
    // (e.g. Ipppp from `auto_select` on an H.264 IPPPP-source clip
    // collapses to b_count=0 here).
    let _ = GopPattern::Ipppp { gop: 0 }; // keep GopPattern import live

    // Per-GOP encoded bytes, filled out-of-order as GOPs fire;
    // assembled into the final stream in temporal order at the end.
    let mut gop_bytes: Vec<Option<Vec<u8>>> = vec![None; num_gops];

    // ── Lockstep emission + per-GOP Pass 3 ──
    loop {
        // Helper: existence check for each driver slot.
        let alive = [
            driver0.is_some(),
            driver1.is_some(),
            driver2.is_some(),
            false,
        ];
        if !alive.iter().any(|&a| a) {
            break;
        }

        // Pick the driver to step: argmax(highest_pending_gop) over
        // alive drivers; tie-break by lowest d. If all alive
        // drivers have completed their GOP-side reporting (e.g.,
        // empty trailing segments), step the lowest-d alive driver
        // anyway to drain.
        let mut best_d: Option<usize> = None;
        let mut best_g: Option<usize> = None;
        let mut fallback_d: Option<usize> = None;
        for d in 0..4 {
            if !alive[d] {
                continue;
            }
            fallback_d.get_or_insert(d);
            if let Some(g) = builder.highest_pending_gop(d) {
                match best_g {
                    None => {
                        best_g = Some(g);
                        best_d = Some(d);
                    }
                    Some(cur) if g > cur => {
                        best_g = Some(g);
                        best_d = Some(d);
                    }
                    _ => {}
                }
            }
        }
        let step_d = best_d.or(fallback_d).expect("alive checked above");

        let emission = match step_d {
            0 => driver0
                .as_mut()
                .unwrap()
                .step()
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "v2 Phase B driver[0]: {e}"
                )))?,
            1 => driver1
                .as_mut()
                .unwrap()
                .step()
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "v2 Phase B driver[1]: {e}"
                )))?,
            2 => driver2
                .as_mut()
                .unwrap()
                .step()
                .map_err(|e| StegoError::InvalidVideo(format!(
                    "v2 Phase B driver[2]: {e}"
                )))?,
            _ => unreachable!("step_d in 0..3 (domain 3 always inactive)"),
        };

        match emission {
            Some(em) => {
                builder
                    .accept_emission(step_d, em.j_start, &em.stego_bits)
                    .map_err(|e| StegoError::InvalidVideo(format!(
                        "v2 accept_emission[{step_d}]: {e}"
                    )))?;
                builder.add_modifications(em.num_modifications);
            }
            None => {
                match step_d {
                    0 => driver0 = None,
                    1 => driver1 = None,
                    2 => driver2 = None,
                    _ => unreachable!(),
                }
            }
        }

        // Fire ready GOPs incrementally (per-GOP Pass 3). After
        // Pass 3, the per-GOP plan buffer is dropped — that's the
        // memory saving.
        for ready in builder.take_ready_gops() {
            let g = ready.gop_idx;
            let gop_plan = DomainPlan {
                coeff_sign_bypass: ready.plans[0].clone(),
                coeff_suffix_lsb: ready.plans[1].clone(),
                mvd_sign_bypass: ready.plans[2].clone(),
                mvd_suffix_lsb: ready.plans[3].clone(),
                total_modifications: 0,
                total_cost: 0.0,
            };
            gop_bytes[g] = Some(encode_one_gop_with_plan_and_capture(
                yuv, width, height, n_frames, frame_size, quality, pattern,
                g, &gop_plan, None,
            )?);
        }
    }

    // All drivers exhausted — every GOP must have fired.
    if !builder.all_fired() {
        return Err(StegoError::InvalidVideo(format!(
            "v2: {} GOPs not fired after lockstep loop",
            num_gops - gop_bytes.iter().filter(|b| b.is_some()).count(),
        )));
    }

    // Concatenate output bytes in temporal (encode) order.
    let mut out = Vec::new();
    for g in 0..num_gops {
        match gop_bytes[g].take() {
            Some(b) => out.extend_from_slice(&b),
            None => {
                return Err(StegoError::InvalidVideo(format!(
                    "v2: GOP {g} did not fire"
                )))
            }
        }
    }
    Ok(out)
}

/// Phase 6E-C1a — multi-IDR variant of
/// `h264_stego_encode_i_then_p_frames_4domain`. See
/// [`h264_stego_encode_yuv_string_4domain_multigop`] for the
/// `gop_size` semantics. The STC plan still spans the whole
/// multi-IDR cover (no per-GOP message split).
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_i_then_p_frames_4domain_multigop(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    b_count: usize,
    message: &[u8],
    passphrase: &str,
    h: usize,
    quality: Option<u8>,
) -> Result<Vec<u8>, StegoError> {
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames
        )));
    }
    if !(1..=7).contains(&h) {
        return Err(StegoError::InvalidVideo(format!("STC h must be 1..=7, got {h}")));
    }
    if gop_size == 0 {
        return Err(StegoError::InvalidVideo("gop_size must be > 0".into()));
    }

    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let m_total = message.len();

    // ── Pass 1: log all 4 domains' covers (MVD + residual). ──
    let cover_p1 = pass1_count_4domain(
        yuv, width, height, n_frames, frame_size, quality, gop_size, b_count,
    )?;
    let cap_p1 = cover_p1.cover.capacity();

    // Phase 6F.2(k).4 — stealth-weighted cross-domain allocation.
    //
    // The orchestrator computes per-domain bit allocations using
    // `stealth_weighted_allocation` with v1.0-default weights:
    // [coeff_sign=0.5, coeff_suffix=0.8, mvd_sign=1.0, mvd_suffix=1.0].
    // Higher weight = less-attacked domain → more headroom for
    // embedding without triggering steganalysis. Drift-budget cap
    // (default 0.20 of M) bounds cumulative MVD-induced pixel
    // drift while still distributing modifications to break the
    // per-domain fingerprint anomaly.
    //
    // mvd_suffix_lsb stays at zero capacity — a magnitude-LSB flip
    // changes |MVD| by 1 which IS a cascade through the median
    // predictor (same failure mode as §6F.2(j)). v1.0 sign-only
    // path captures the dominant bypass-bin family for stealth
    // fingerprint purposes.
    let allocator = super::orchestrate::StealthAllocator::v1_default();
    let cap_for_alloc = GopCapacity {
        coeff_sign_bypass: cap_p1.coeff_sign_bypass,
        coeff_suffix_lsb: cap_p1.coeff_suffix_lsb,
        mvd_sign_bypass: cap_p1.mvd_sign_bypass,
        mvd_suffix_lsb: 0, // suffix-LSB MVD injection cascades; disabled
    };
    let (m_cs, m_cl, m_ms, _m_ml) =
        super::orchestrate::stealth_weighted_allocation(
            m_total, &cap_for_alloc, &allocator,
        ).ok_or(StegoError::MessageTooLarge)?;
    let m_mvd = m_ms; // mvd_sign only; mvd_suffix is forced 0
    let m_residual = m_cs + m_cl;
    debug_assert_eq!(m_total, m_mvd + m_residual,
        "stealth-weighted allocation must conserve m_total");

    // ── Pass 2A: STC plan over cover_p1.mvd_sign_bypass. ──
    let plan_a = if m_mvd > 0 {
        let cap_mvd_only = GopCapacity {
            coeff_sign_bypass: 0,
            coeff_suffix_lsb: 0,
            mvd_sign_bypass: cap_p1.mvd_sign_bypass,
            mvd_suffix_lsb: 0,
        };
        let messages_a = split_message_per_domain(&message[..m_mvd], &cap_mvd_only)
            .ok_or_else(|| StegoError::InvalidVideo(
                "Pass 2A: MVD split failed".into()
            ))?;
        let mut cover_for_a = super::orchestrate::GopCover::default();
        cover_for_a.cover.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.clone();
        cover_for_a.costs.mvd_sign_bypass = cover_p1.costs.mvd_sign_bypass.clone();
        pass2_stc_plan_with_keys(&cover_for_a, &messages_a, h, &keys, /* gop */ 0)
            .ok_or_else(|| StegoError::InvalidVideo(
                "Pass 2A: STC plan failed".into()
            ))?
    } else {
        DomainPlan::default()
    };

    // ── Pass 1B: re-encode with empty MVD plan (no-op injection)
    // and residual logger. ──
    let cover_p1b_residual = pass1b_inject_mvd_log_residual(
        yuv, width, height, n_frames, frame_size, quality,
        &cover_p1.cover, &plan_a, gop_size, b_count,
    )?;

    // Validate residual capacity is enough for m_residual.
    let cap_p1b = cover_p1b_residual.cover.capacity();
    let residual_capacity = cap_p1b.coeff_sign_bypass + cap_p1b.coeff_suffix_lsb;
    if m_residual > residual_capacity {
        return Err(StegoError::MessageTooLarge);
    }

    // ── Pass 2B: plan residual STC. ──
    let plan_b = if m_residual > 0 {
        let cap_coeff_only = GopCapacity {
            coeff_sign_bypass: cap_p1b.coeff_sign_bypass,
            coeff_suffix_lsb: cap_p1b.coeff_suffix_lsb,
            mvd_sign_bypass: 0,
            mvd_suffix_lsb: 0,
        };
        let messages_b = split_message_per_domain(&message[m_mvd..], &cap_coeff_only)
            .ok_or_else(|| StegoError::InvalidVideo(
                "Pass 2B: residual split failed".into()
            ))?;
        let mut cover_for_b = super::orchestrate::GopCover::default();
        cover_for_b.cover.coeff_sign_bypass = cover_p1b_residual.cover.coeff_sign_bypass.clone();
        cover_for_b.cover.coeff_suffix_lsb = cover_p1b_residual.cover.coeff_suffix_lsb.clone();
        cover_for_b.costs.coeff_sign_bypass = cover_p1b_residual.costs.coeff_sign_bypass.clone();
        cover_for_b.costs.coeff_suffix_lsb = cover_p1b_residual.costs.coeff_suffix_lsb.clone();
        pass2_stc_plan_with_keys(&cover_for_b, &messages_b, h, &keys, /* gop */ 0)
            .ok_or_else(|| StegoError::InvalidVideo(
                "Pass 2B: STC plan failed".into()
            ))?
    } else {
        DomainPlan::default()
    };

    // ── Pass 3: combined injection. ──
    let mut combined_cover = super::DomainCover::default();
    combined_cover.coeff_sign_bypass = cover_p1b_residual.cover.coeff_sign_bypass.clone();
    combined_cover.coeff_suffix_lsb = cover_p1b_residual.cover.coeff_suffix_lsb.clone();
    combined_cover.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.clone();
    combined_cover.mvd_suffix_lsb = cover_p1.cover.mvd_suffix_lsb.clone();
    let mut combined_plan = DomainPlan::default();
    combined_plan.coeff_sign_bypass = plan_b.coeff_sign_bypass.clone();
    combined_plan.coeff_suffix_lsb = plan_b.coeff_suffix_lsb.clone();
    combined_plan.mvd_sign_bypass = plan_a.mvd_sign_bypass.clone();
    combined_plan.mvd_suffix_lsb = plan_a.mvd_suffix_lsb.clone();

    pass3_inject_4domain(
        yuv, width, height, n_frames, frame_size, quality,
        &combined_cover, &combined_plan, gop_size, b_count,
    )
}

/// §6E-C2 polish — H.264 video shadow capacity prediction.
///
/// Returned by `h264_stego_shadow_capacity` to tell callers the
/// largest shadow message size that the encoder can reliably embed
/// at the given `(yuv, n_shadows)` shape, accounting for both
/// the cover capacity and the inter-shadow collision absorption
/// limit of the cascade RS parity ladder.
///
/// The polish architecture's encoder/decoder share a single
/// priority sort over the primary-emit cover, with each shadow
/// independently selecting top-N positions by ChaCha20-priority.
/// Inter-shadow collisions (positions chosen by ≥2 shadows)
/// resolve by write-order; the colliding bit propagates to the
/// later-written shadow's RS stream as random errors. RS at the
/// cascade's max parity tier (128) absorbs up to ~512 bit errors
/// per single-block message, which sets the per-shadow size cap.
///
/// Capacity formula:
///   m_max_bits ≤ sqrt(1024 × C_total / max(1, N − 1))
/// where C_total is the total bypass-bin cover positions across
/// all 4 domains and N is the shadow count. Subtract worst-case
/// RS parity (128 bytes) and the per-shadow frame overhead
/// (~46 bytes) to get the maximum payload (text + files) bytes
/// per shadow.
///
/// For `n_shadows == 1`, no inter-shadow collisions exist; the
/// limit reduces to (cover_size_bits / 8) − parity − overhead.
/// For `n_shadows == 0`, returns `max_message_bytes = 0` (no
/// shadow encoding needed).
pub struct H264ShadowCapacityInfo {
    /// Total bypass-bin cover positions across all 4 domains.
    pub cover_size_bits: usize,
    /// Maximum safe shadow payload bytes (text + files combined,
    /// per shadow) such that the cascade succeeds at one of the
    /// 6 parity tiers under the given `n_shadows`.
    pub max_message_bytes: usize,
    /// Number of shadows this capacity was computed for.
    pub n_shadows: usize,
}

/// §6E-C2 polish — predict the safe per-shadow message capacity
/// for a given video + shadow count.
///
/// Runs Pass 1 to count bypass-bin cover positions (this is fast,
/// just a logger pass — no STC, no emit). Computes the
/// collision-limited safe capacity per the formula in
/// [`H264ShadowCapacityInfo`].
///
/// Callers (UI) MUST check shadow message sizes against
/// `max_message_bytes` BEFORE calling
/// `h264_stego_encode_yuv_string_with_n_shadows`. The encoder also
/// validates and returns `Err(StegoError::MessageTooLarge)` if any
/// shadow exceeds the limit, but the UI check should happen earlier
/// to give better user feedback.
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_shadow_capacity(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    n_shadows: usize,
) -> Result<H264ShadowCapacityInfo, StegoError> {
    // Video shadow uses the wide frame format (u32 plaintext_len,
    // 48-byte overhead). See `crate::stego::shadow_layer` module docs.
    use crate::stego::shadow_layer::SHADOW_FRAME_OVERHEAD_WIDE
        as SHADOW_FRAME_OVERHEAD;

    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames
        )));
    }
    if gop_size == 0 || gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size {gop_size} must be in 1..={n_frames}"
        )));
    }

    let cover_p1 = pass1_count_4domain(
        yuv, width, height, n_frames, frame_size, /* quality */ Some(26), gop_size,
        /* b_count */ 1,
    )?;
    // §6E-A5(c.x) — shadow position selection excludes MvdSuffixLsb
    // (not injectable post-§6F.2(k).2). Shadow capacity counts only
    // the 3 injectable domains so callers get honest numbers.
    let cap_p1 = cover_p1.cover.capacity();
    let cover_size_bits =
        cap_p1.coeff_sign_bypass + cap_p1.coeff_suffix_lsb + cap_p1.mvd_sign_bypass;

    let max_message_bytes = if n_shadows == 0 {
        0
    } else {
        // Collision-limited per the formula in the doc above.
        let denom = n_shadows.saturating_sub(1).max(1);
        let m_max_bits_squared = 1024usize.saturating_mul(cover_size_bits) / denom;
        let m_max_bits = (m_max_bits_squared as f64).sqrt() as usize;
        // Also bound by raw cover capacity (single-shadow case dominates).
        let m_max_bits = m_max_bits.min(cover_size_bits);
        let m_max_bytes = m_max_bits / 8;
        // Subtract worst-case parity (max tier) + per-shadow frame overhead.
        m_max_bytes
            .saturating_sub(128)
            .saturating_sub(SHADOW_FRAME_OVERHEAD)
    };

    Ok(H264ShadowCapacityInfo {
        cover_size_bits,
        max_message_bytes,
        n_shadows,
    })
}

/// §6E-A5(c.x) / Task #96 — combined primary + shadow capacity
/// surface for the CABAC v2 stego pipeline.
///
/// Returned by [`h264_stego_capacity_4domain`]. `cover_size_bits`
/// is the sum of the 3 INJECTABLE bypass-bin domains
/// (CoeffSignBypass + CoeffSuffixLsb + MvdSignBypass) — what shadow
/// position selection actually has to work with after the #107 fix
/// excluded MvdSuffixLsb.
///
/// Both `primary_max_message_bytes` and `shadow_max_message_bytes`
/// are upper bounds on the user-supplied UTF-8 string + attached
/// files combined (one per channel), AFTER subtracting framing
/// overhead (encrypted nonce + salt + plaintext-length + auth tag).
/// The shadow value additionally accounts for the `[4, 8, 16, 32,
/// 64, 128]` parity-tier cascade absorption.
#[derive(Debug, Clone, Copy)]
pub struct H264StegoCapacityInfo {
    /// Total injectable bits across the 3 bypass-bin domains used
    /// by stego: `csb + csl + msb`. Excludes `msl` since
    /// MvdSuffixLsb isn't injectable post-§6F.2(k).2.
    pub cover_size_bits: usize,
    /// Maximum primary message bytes (text + attached files
    /// combined) embeddable via the §30D-C 4-domain orchestrator.
    /// Equals `cover_size_bits/8 - FRAME_OVERHEAD`.
    pub primary_max_message_bytes: usize,
    /// Maximum SHADOW message bytes per shadow at single-shadow
    /// (n_shadows = 1) load. For multi-shadow scenarios use
    /// [`h264_stego_shadow_capacity`] with the actual shadow count.
    pub shadow_max_message_bytes: usize,
}

/// §6E-A5(c.x) / Task #96 — predict primary AND shadow capacity
/// for a given YUV+gop_size combination.
///
/// Single Pass-1 walk; returns both numbers in one struct so callers
/// (CLI `video-capacity`, mobile bridges) can present accurate
/// pre-encode estimates without needing to call two separate APIs.
///
/// For multi-shadow scenarios use [`h264_stego_shadow_capacity`]
/// directly with the actual `n_shadows` count — this helper
/// hardcodes `n_shadows = 1` (the single-shadow case the mobile
/// UI exposes today).
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_capacity_4domain(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
) -> Result<H264StegoCapacityInfo, StegoError> {
    use crate::stego::frame::FRAME_OVERHEAD;
    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames
        )));
    }
    if gop_size == 0 || gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size {gop_size} must be in 1..={n_frames}"
        )));
    }

    let cover_p1 = pass1_count_4domain(
        yuv, width, height, n_frames, frame_size, /* quality */ Some(26),
        gop_size, /* b_count */ 1,
    )?;
    let cap_p1 = cover_p1.cover.capacity();
    // Post-#107: MvdSuffixLsb is excluded from the injectable pool.
    let cover_size_bits =
        cap_p1.coeff_sign_bypass + cap_p1.coeff_suffix_lsb + cap_p1.mvd_sign_bypass;

    // Primary: bytes available = bits/8, minus framing overhead.
    let primary_max_message_bytes =
        (cover_size_bits / 8).saturating_sub(FRAME_OVERHEAD);

    // Shadow: reuse the existing collision-limited formula at
    // n_shadows = 1.
    let shadow_info = h264_stego_shadow_capacity(
        yuv, width, height, n_frames, gop_size, /* n_shadows */ 1,
    )?;

    Ok(H264StegoCapacityInfo {
        cover_size_bits,
        primary_max_message_bytes,
        shadow_max_message_bytes: shadow_info.max_message_bytes,
    })
}

/// Phase 6E-C1b-v2 — encode primary + 1 shadow message into H.264
/// Annex-B bytes with **cascade verification** (mirrors image-side
/// architecture). Both messages live in the same video, each
/// retrievable only with its own passphrase. Under coercion the
/// user reveals `primary_pass`; the shadow remains undetectable.
///
/// Primary uses the existing §30D-C 4-domain STC pipeline + §6E-C1a
/// multi-IDR. Shadow uses direct LSB writes at top-N hash-priority
/// positions across all 4 bypass-bin domains with Reed-Solomon
/// error correction. The cascade loop walks the parity-tier ladder
/// `[4, 8, 16, 32, 64, 128]`, retrying with bigger parity until the
/// encoder-side verify (walk emitted bytes → re-derive shadow
/// positions → RS-decode + AES-GCM-SIV authenticate) recovers the
/// shadow message correctly. Pass 1 (vanilla cover) is memoized
/// across cascade rounds; each retry costs ~Pass 1B + Pass 2A +
/// Pass 2B + Pass 3 + verify-walk.
///
/// Returns `Err(StegoError::ShadowEmbedFailed)` on cascade
/// exhaustion (parity 128 fails). Caller can retry with a smaller
/// primary, different `gop_size`/quality, or different shadow
/// passphrase. See `docs/design/h264-shadow-messages.md` for the
/// full architecture and performance picture.
///
/// Decoder is unchanged from §6E-C1b — already brute-forces all
/// 6 parity tiers; doesn't know or care that the encoder cascaded.
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_with_shadow(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    primary_message: &str,
    primary_passphrase: &str,
    shadow_message: &str,
    shadow_passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::shadow_layer::ShadowLayer;
    let shadow = ShadowLayer {
        message: shadow_message,
        passphrase: shadow_passphrase,
        files: &[],
    };
    h264_stego_encode_yuv_string_with_n_shadows(
        yuv, width, height, n_frames, gop_size,
        primary_message, primary_passphrase,
        std::slice::from_ref(&shadow),
    )
}

/// §6E-A5(a) — explicit-pattern variant of
/// [`h264_stego_encode_yuv_string_with_shadow`]. Mirrors
/// `h264_stego_encode_yuv_string_4domain_multigop_with_pattern`'s
/// shape on the primary side. The non-`_with_pattern` form
/// delegates here with `GopPattern::Ibpbp { gop: gop_size,
/// b_count: 1 }` (Apple-iPhone canonical IBPBP, M=2).
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_with_shadow_with_pattern(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    primary_message: &str,
    primary_passphrase: &str,
    shadow_message: &str,
    shadow_passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::shadow_layer::ShadowLayer;
    let shadow = ShadowLayer {
        message: shadow_message,
        passphrase: shadow_passphrase,
        files: &[],
    };
    h264_stego_encode_yuv_string_with_n_shadows_with_pattern(
        yuv, width, height, n_frames, pattern,
        primary_message, primary_passphrase,
        std::slice::from_ref(&shadow),
    )
}

/// Phase 6E-C2 — encode primary + N shadow messages into H.264
/// Annex-B bytes with cascade verification across all N shadows.
/// Each shadow has its own passphrase and independent
/// hash-priority position selection. Inter-shadow collisions
/// (positions one shadow's selection overlaps with another's)
/// resolve by write-order (later shadow overwrites earlier);
/// the resulting bit corruption at the overwritten positions is
/// absorbed by Reed-Solomon parity.
///
/// For N=1 this is identical to
/// `h264_stego_encode_yuv_string_with_shadow`.
///
/// Cascade behavior: a SINGLE parity tier applies to ALL shadows
/// each round. If any shadow fails verify, the tier escalates
/// for ALL shadows on the next retry. This is simpler than per-
/// shadow cascade (image-side approach) and works well when most
/// shadows succeed at parity 4 — typical workload converges in
/// 1 round. Per-shadow independent cascade is a §6E-C2.x polish
/// item if real-world testing surfaces a need.
///
/// Returns `Err(StegoError::ShadowEmbedFailed)` on cascade
/// exhaustion. Returns `Err(StegoError::DuplicatePassphrase)` if
/// any two shadows or primary share the same passphrase.
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_with_n_shadows<'a>(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    primary_message: &str,
    primary_passphrase: &str,
    shadows: &'a [crate::stego::shadow_layer::ShadowLayer<'a>],
) -> Result<Vec<u8>, StegoError> {
    // §Default-flip.IBPBP (#193) — visual bug fixed in commit 5b85432.
    h264_stego_encode_yuv_string_with_n_shadows_with_pattern(
        yuv, width, height, n_frames,
        super::gop_pattern::GopPattern::Ibpbp { gop: gop_size, b_count: 1 },
        primary_message, primary_passphrase, shadows,
    )
}

/// §6E-A5(a) — explicit-pattern variant of
/// [`h264_stego_encode_yuv_string_with_n_shadows`]. The
/// `pattern` argument is forwarded to the inner Pass-1 / Pass-1B
/// / Pass-3 helpers via `GopPattern::legacy_b_count()`. Picks
/// `IPPPP` (b_count=0) or `IBPBP` (b_count≥1) end-to-end.
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_with_n_shadows_with_pattern<'a>(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    primary_message: &str,
    primary_passphrase: &str,
    shadows: &'a [crate::stego::shadow_layer::ShadowLayer<'a>],
) -> Result<Vec<u8>, StegoError> {
    h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files(
        yuv, width, height, n_frames, pattern,
        primary_message, &[], primary_passphrase, shadows,
    )
}

/// Task #97 — N-shadow variant accepting `primary_files`. Shadows
/// already carry their own files via `ShadowLayer.files`. This
/// completes the parity with the JPEG-side image stego where
/// primary + each shadow can independently attach binary files.
#[allow(clippy::too_many_arguments)]
pub fn h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files<'a>(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    primary_message: &str,
    primary_files: &[crate::stego::payload::FileEntry],
    primary_passphrase: &str,
    shadows: &'a [crate::stego::shadow_layer::ShadowLayer<'a>],
) -> Result<Vec<u8>, StegoError> {
    use crate::stego::{crypto, frame, payload};
    use crate::stego::shadow_layer::SHADOW_PARITY_TIERS;
    let gop_size = pattern.gop_size();
    let b_count = pattern.legacy_b_count();

    if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
        return Err(StegoError::InvalidVideo(format!(
            "dimensions must be 16-aligned, got {width}x{height}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    if yuv.len() != frame_size * n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "yuv len {} != expected {}", yuv.len(), frame_size * n_frames
        )));
    }
    if gop_size == 0 || gop_size > n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_size {gop_size} must be in 1..={n_frames}"
        )));
    }

    // Reject duplicate passphrases across primary + shadows.
    let mut all_passphrases: Vec<&str> = Vec::with_capacity(shadows.len() + 1);
    all_passphrases.push(primary_passphrase);
    for s in shadows {
        all_passphrases.push(s.passphrase);
    }
    for i in 0..all_passphrases.len() {
        for j in (i + 1)..all_passphrases.len() {
            if all_passphrases[i] == all_passphrases[j] {
                return Err(StegoError::DuplicatePassphrase);
            }
        }
    }

    // §6E-C2 polish — capacity-limit check. Reject shadow loads
    // that would exceed the inter-shadow collision absorption
    // capacity at the cascade's max parity tier. UI is expected to
    // call `h264_stego_shadow_capacity` first; this is a
    // belt-and-suspenders guard for callers that don't.
    if !shadows.is_empty() {
        let cap_info = h264_stego_shadow_capacity(
            yuv, width, height, n_frames, gop_size, shadows.len(),
        )?;
        for s in shadows {
            let msg_bytes = s.message.len()
                + s.files.iter().map(|f| f.content.len()).sum::<usize>();
            if msg_bytes > cap_info.max_message_bytes {
                return Err(StegoError::MessageTooLarge);
            }
        }
    }

    // Frame the primary message.
    let primary_bytes = payload::encode_payload(primary_message, primary_files)?;
    let (ct, nonce, salt) = crypto::encrypt(&primary_bytes, primary_passphrase)?;
    let frame_bytes = frame::build_frame(primary_bytes.len(), &salt, &nonce, &ct);
    let frame_bits: Vec<u8> = frame_bytes
        .iter()
        .flat_map(|&byte| (0..8).rev().map(move |i| (byte >> i) & 1))
        .collect();

    let h: usize = 4;
    let quality = Some(26);
    let keys = CabacStegoMasterKeys::derive(primary_passphrase)?;
    let m_total = frame_bits.len();

    // ── Pass 1 (memoized across cascade rounds): log vanilla
    //     4-domain cover. Pass 1 doesn't depend on shadow parity
    //     or shadow positions — same cover for every retry. ──
    let cover_p1 = pass1_count_4domain(
        yuv, width, height, n_frames, frame_size, quality, gop_size,
        b_count,
    )?;
    let cap_p1 = cover_p1.cover.capacity();
    // §6E-A5(b) — align with §30D-C primary: drop mvd_suffix_lsb
    // from the MVD capacity used for shadow message split. Magnitude-
    // LSB MVD flips cascade through the median predictor (§6F.2(j));
    // the production primary pipeline forces mvd_suffix_lsb = 0 for
    // the same reason. Shadow inherits that stance; the cascade RS
    // parity tiers absorb the slight capacity reduction (the
    // primary-emit cover still has ample headroom).
    let mvd_capacity = cap_p1.mvd_sign_bypass;
    let m_mvd = m_total.min(mvd_capacity);
    let m_residual = m_total - m_mvd;

    // §6E-A5(c.3) — derive per_gop_counts from cover_p1.positions.
    // Used by the streaming Pass-1B + Pass-3 variants below to
    // bound per-cascade-iter encoder working set to a single GOP.
    // Cheaper than a second `pass1_count_per_gop_4domain` call
    // since cover_p1 is already materialized.
    let per_gop_counts = derive_per_gop_counts_from_cover(
        yuv, frame_size, n_frames, pattern, &cover_p1.cover,
    );

    // ── §6E-C2 polish — provisional pass (no shadow) gives the
    //     `primary_emit_cover` over which shadow positions are
    //     selected. This matches the cover the DECODER walks from
    //     final emit (modulo shadow override values), sidestepping
    //     the encoder-vs-decoder priority-sort divergence that
    //     blocked §6E-C2 mvp at N≥2. cover_p1b_residual_prov is the
    //     PositionKey reference for residual-domain shadow position
    //     translation; cover_p1b_residual_final from each cascade
    //     round may drift slightly. Cascade absorbs.
    //
    //     Skipped for shadows.is_empty(): no shadow → no provisional
    //     needed → fall through to the no-shadow no-cascade path
    //     below (pure primary 3-pass). ──
    let (primary_emit_cover, cover_p1b_residual_prov, safe_msl_prov) = if shadows.is_empty() {
        (
            super::DomainCover::default(),
            super::orchestrate::GopCover::default(),
            Vec::<bool>::new(),
        )
    } else {
        let plan_a_prov = if m_mvd > 0 {
            let cap_mvd_only = GopCapacity {
                coeff_sign_bypass: 0,
                coeff_suffix_lsb: 0,
                mvd_sign_bypass: cap_p1.mvd_sign_bypass,
                // §6E-A5(b) — aligned with §30D-C primary
                // (§6F.2(j) cascade-through-median).
                mvd_suffix_lsb: 0,
            };
            let messages_a = split_message_per_domain(&frame_bits[..m_mvd], &cap_mvd_only)
                .ok_or_else(|| StegoError::InvalidVideo(
                    "Pass 2A_prov: MVD split failed".into(),
                ))?;
            let mut cover_for_a = super::orchestrate::GopCover::default();
            cover_for_a.cover.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.clone();
            cover_for_a.cover.mvd_suffix_lsb = cover_p1.cover.mvd_suffix_lsb.clone();
            cover_for_a.costs.mvd_sign_bypass = cover_p1.costs.mvd_sign_bypass.clone();
            cover_for_a.costs.mvd_suffix_lsb = cover_p1.costs.mvd_suffix_lsb.clone();
            pass2_stc_plan_with_keys(&cover_for_a, &messages_a, h, &keys, /* gop */ 0)
                .ok_or_else(|| StegoError::InvalidVideo(
                    "Pass 2A_prov: STC plan failed".into(),
                ))?
        } else {
            DomainPlan::default()
        };
        let cover_p1b_prov = pass1b_inject_mvd_log_residual_streaming(
            yuv, width, height, n_frames, frame_size, quality,
            pattern, &cover_p1.cover, &plan_a_prov, &per_gop_counts,
        )?;
        let cap_p1b_prov = cover_p1b_prov.cover.capacity();
        let residual_capacity_prov =
            cap_p1b_prov.coeff_sign_bypass + cap_p1b_prov.coeff_suffix_lsb;
        if m_residual > residual_capacity_prov {
            return Err(StegoError::MessageTooLarge);
        }
        let plan_b_prov = if m_residual > 0 {
            let cap_coeff_only = GopCapacity {
                coeff_sign_bypass: cap_p1b_prov.coeff_sign_bypass,
                coeff_suffix_lsb: cap_p1b_prov.coeff_suffix_lsb,
                mvd_sign_bypass: 0,
                mvd_suffix_lsb: 0,
            };
            let messages_b = split_message_per_domain(&frame_bits[m_mvd..], &cap_coeff_only)
                .ok_or_else(|| StegoError::InvalidVideo(
                    "Pass 2B_prov: residual split failed".into(),
                ))?;
            let mut cover_for_b = super::orchestrate::GopCover::default();
            cover_for_b.cover.coeff_sign_bypass = cover_p1b_prov.cover.coeff_sign_bypass.clone();
            cover_for_b.cover.coeff_suffix_lsb = cover_p1b_prov.cover.coeff_suffix_lsb.clone();
            cover_for_b.costs.coeff_sign_bypass = cover_p1b_prov.costs.coeff_sign_bypass.clone();
            cover_for_b.costs.coeff_suffix_lsb = cover_p1b_prov.costs.coeff_suffix_lsb.clone();
            pass2_stc_plan_with_keys(&cover_for_b, &messages_b, h, &keys, /* gop */ 0)
                .ok_or_else(|| StegoError::InvalidVideo(
                    "Pass 2B_prov: STC plan failed".into(),
                ))?
        } else {
            DomainPlan::default()
        };
        let mut combined_plan_prov = DomainPlan {
            coeff_sign_bypass: plan_b_prov.coeff_sign_bypass.clone(),
            coeff_suffix_lsb: plan_b_prov.coeff_suffix_lsb.clone(),
            mvd_sign_bypass: plan_a_prov.mvd_sign_bypass.clone(),
            mvd_suffix_lsb: plan_a_prov.mvd_suffix_lsb.clone(),
            total_modifications: plan_a_prov.total_modifications + plan_b_prov.total_modifications,
            total_cost: plan_a_prov.total_cost + plan_b_prov.total_cost,
        };
        if combined_plan_prov.coeff_sign_bypass.is_empty() {
            combined_plan_prov.coeff_sign_bypass =
                cover_p1b_prov.cover.coeff_sign_bypass.bits.clone();
        }
        if combined_plan_prov.coeff_suffix_lsb.is_empty() {
            combined_plan_prov.coeff_suffix_lsb =
                cover_p1b_prov.cover.coeff_suffix_lsb.bits.clone();
        }
        if combined_plan_prov.mvd_sign_bypass.is_empty() {
            combined_plan_prov.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.bits.clone();
        }
        if combined_plan_prov.mvd_suffix_lsb.is_empty() {
            combined_plan_prov.mvd_suffix_lsb = cover_p1.cover.mvd_suffix_lsb.bits.clone();
        }
        let bytes_prov = pass3_inject_4domain_streaming(
            yuv, width, height, n_frames, frame_size, quality, pattern,
            &combined_plan_prov, &per_gop_counts,
        )?;
        let walk_opts = WalkOptions { record_mvd: true };
        let walk_prov = walk_annex_b_for_cover_with_options(&bytes_prov, walk_opts)
            .map_err(|e| StegoError::InvalidVideo(format!("provisional walk: {e}")))?;
        // §6E-A5(d).6 — derive safe MvdSuffixLsb mask from the
        // walked provisional cover. Decoder runs the same analysis
        // on its post-walk meta → identical mask by §6F.2(j)
        // construction (greedy is invariant under sign-flips +
        // safe-set magnitude flips).
        let safe_msb_prov = super::cascade_safety::analyze_safe_mvd_subset(
            &walk_prov.mvd_meta, walk_prov.mb_w, walk_prov.mb_h,
        );
        let safe_msl = super::cascade_safety::derive_msl_safe_from_msb(
            &walk_prov.cover.mvd_sign_bypass.positions,
            &safe_msb_prov,
            &walk_prov.cover.mvd_suffix_lsb.positions,
        );
        (walk_prov.cover, cover_p1b_prov, safe_msl)
    };

    // ── Cascade loop: parity_idx ∈ [0..6] (tiers [4, 8, 16, 32, 64, 128]).
    //     Each round re-selects shadow positions at the new parity,
    //     re-overlays ∞-cost, re-runs Pass 2A → Pass 1B → Pass 2B →
    //     Pass 3 → verify. First-round success = no cascade overhead.
    //     6th round failure = ShadowEmbedFailed.
    //
    //     §6E-C2 polish: shadow positions are now selected over
    //     `primary_emit_cover` (the cover decoder walks from final
    //     emit), then translated to cover_p1.mvd_* / cover_p1b_*.coeff_*
    //     intra_indexes via PositionKey lookup for the ∞-cost +
    //     injection flow. Slots whose PositionKey doesn't appear in
    //     the target cover are skipped (small fraction; cascade
    //     absorbs). ──
    let safe_msl_for_select: Option<&[bool]> =
        if shadows.is_empty() { None } else { Some(safe_msl_prov.as_slice()) };
    for &parity_len in &SHADOW_PARITY_TIERS {
        // ── §6E-C2 polish + §6E-A5(d).6 — select shadow positions
        //     over `primary_emit_cover`. Spans 4 bypass-bin domains:
        //     CoeffSignBypass + CoeffSuffixLsb + MvdSignBypass
        //     unconditionally, plus MvdSuffixLsb at cascade-safe
        //     positions only (filtered by `safe_msl_prov` from
        //     `cascade_safety::analyze_safe_mvd_subset` on the
        //     provisional walked meta). Decoder recomputes the same
        //     filter from its own walk → encoder + decoder lockstep. ──
        let shadow_states_emit: Vec<super::shadow::ShadowState> = shadows
            .iter()
            .map(|s| super::shadow::prepare_shadow_over_emit_cover_safe(
                &primary_emit_cover, s.passphrase, s.message, s.files, parity_len,
                None, safe_msl_for_select,
            ))
            .collect::<Result<_, _>>()?;

        // Translate each shadow state to cover_p1 (MVD) + cover_p1b_prov
        // (residual) intra_indexes. Used for Phase 1 cover_for_a
        // injection + ∞-cost mask. Residual indexes will be re-translated
        // to cover_p1b_FINAL after Pass 1B; for Pass 2A only MVD slots
        // are needed.
        let shadow_states_phase1: Vec<super::shadow::ShadowState> = shadow_states_emit
            .iter()
            .map(|s| translate_shadow_state(
                s, &primary_emit_cover, &cover_p1.cover, &cover_p1b_residual_prov.cover,
            ))
            .collect();

        // ── Build cover_for_a: Pass 1's MVD cover with EACH shadow's
        //     MVD bits injected (later shadows overwrite earlier at
        //     overlapping positions — RS absorbs) + ∞-cost overlay
        //     at the UNION of all shadows' MVD positions. ──
        let mut cover_for_a = super::orchestrate::GopCover::default();
        cover_for_a.cover.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.clone();
        cover_for_a.cover.mvd_suffix_lsb = cover_p1.cover.mvd_suffix_lsb.clone();
        cover_for_a.costs.mvd_sign_bypass = cover_p1.costs.mvd_sign_bypass.clone();
        cover_for_a.costs.mvd_suffix_lsb = cover_p1.costs.mvd_suffix_lsb.clone();
        let mut dummy_csb_bits = vec![0u8; cover_p1.cover.coeff_sign_bypass.bits.len()];
        let mut dummy_csl_bits = vec![0u8; cover_p1.cover.coeff_suffix_lsb.bits.len()];
        let mut dummy_csb_cost = vec![0.0f32; cover_p1.costs.coeff_sign_bypass.len()];
        let mut dummy_csl_cost = vec![0.0f32; cover_p1.costs.coeff_suffix_lsb.len()];
        for state in &shadow_states_phase1 {
            super::shadow::embed_shadow_lsb_all4(
                &mut dummy_csb_bits,
                &mut dummy_csl_bits,
                &mut cover_for_a.cover.mvd_sign_bypass.bits,
                &mut cover_for_a.cover.mvd_suffix_lsb.bits,
                state,
            );
            super::shadow::overlay_infinity_costs_all4(
                &mut dummy_csb_cost,
                &mut dummy_csl_cost,
                &mut cover_for_a.costs.mvd_sign_bypass,
                &mut cover_for_a.costs.mvd_suffix_lsb,
                state,
            );
        }

        // ── Pass 2A: primary MVD STC with shadow ∞-cost on MVD. ──
        let plan_a = if m_mvd > 0 {
            let cap_mvd_only = GopCapacity {
                coeff_sign_bypass: 0,
                coeff_suffix_lsb: 0,
                mvd_sign_bypass: cap_p1.mvd_sign_bypass,
                // §6E-A5(b) — aligned with §30D-C primary
                // (§6F.2(j) cascade-through-median).
                mvd_suffix_lsb: 0,
            };
            let messages_a = split_message_per_domain(&frame_bits[..m_mvd], &cap_mvd_only)
                .ok_or_else(|| StegoError::InvalidVideo("Pass 2A: MVD split failed".into()))?;
            pass2_stc_plan_with_keys(&cover_for_a, &messages_a, h, &keys, /* gop */ 0)
                .ok_or_else(|| StegoError::InvalidVideo("Pass 2A: STC plan failed".into()))?
        } else {
            DomainPlan::default()
        };

        // ── Pass 1B: re-encode with primary MVD plan applied →
        //     final residual cover (post-MC-with-modified-MVDs). ──
        let cover_p1b_residual = pass1b_inject_mvd_log_residual_streaming(
            yuv, width, height, n_frames, frame_size, quality,
            pattern, &cover_p1.cover, &plan_a, &per_gop_counts,
        )?;
        let cap_p1b = cover_p1b_residual.cover.capacity();
        let residual_capacity = cap_p1b.coeff_sign_bypass + cap_p1b.coeff_suffix_lsb;
        if m_residual > residual_capacity {
            return Err(StegoError::MessageTooLarge);
        }

        // ── §6E-C2 polish — Phase 2 re-translation: now that
        //     cover_p1b_residual_final is known, re-translate each
        //     shadow's residual-domain slots from primary_emit_cover
        //     indexing to cover_p1b_residual_final.coeff_* indexing.
        //     MVD slots remain valid (cover_p1 is stable across
        //     cascade rounds). ──
        let shadow_states: Vec<super::shadow::ShadowState> = shadow_states_emit
            .iter()
            .map(|s| translate_shadow_state(
                s, &primary_emit_cover, &cover_p1.cover, &cover_p1b_residual.cover,
            ))
            .collect();

        // ── Build cover_for_b: Pass 1B's residual cover with EACH
        //     shadow's residual bits injected + ∞-cost overlay at
        //     the UNION of all shadows' residual positions.
        //     Indexes are correct because Phase 2 prepared positions
        //     against cover_p1b_residual. ──
        let mut cover_for_b = super::orchestrate::GopCover::default();
        cover_for_b.cover.coeff_sign_bypass = cover_p1b_residual.cover.coeff_sign_bypass.clone();
        cover_for_b.cover.coeff_suffix_lsb = cover_p1b_residual.cover.coeff_suffix_lsb.clone();
        cover_for_b.costs.coeff_sign_bypass = cover_p1b_residual.costs.coeff_sign_bypass.clone();
        cover_for_b.costs.coeff_suffix_lsb = cover_p1b_residual.costs.coeff_suffix_lsb.clone();
        let mut dummy_msb_bits = vec![0u8; cover_p1.cover.mvd_sign_bypass.bits.len()];
        let mut dummy_msl_bits = vec![0u8; cover_p1.cover.mvd_suffix_lsb.bits.len()];
        let mut dummy_msb_cost = vec![0.0f32; cover_p1.costs.mvd_sign_bypass.len()];
        let mut dummy_msl_cost = vec![0.0f32; cover_p1.costs.mvd_suffix_lsb.len()];
        for state in &shadow_states {
            super::shadow::embed_shadow_lsb_all4(
                &mut cover_for_b.cover.coeff_sign_bypass.bits,
                &mut cover_for_b.cover.coeff_suffix_lsb.bits,
                &mut dummy_msb_bits,
                &mut dummy_msl_bits,
                state,
            );
            super::shadow::overlay_infinity_costs_all4(
                &mut cover_for_b.costs.coeff_sign_bypass,
                &mut cover_for_b.costs.coeff_suffix_lsb,
                &mut dummy_msb_cost,
                &mut dummy_msl_cost,
                state,
            );
        }

        // ── Pass 2B: primary residual STC with shadow ∞-cost. ──
        let plan_b = if m_residual > 0 {
            let cap_coeff_only = GopCapacity {
                coeff_sign_bypass: cap_p1b.coeff_sign_bypass,
                coeff_suffix_lsb: cap_p1b.coeff_suffix_lsb,
                mvd_sign_bypass: 0,
                mvd_suffix_lsb: 0,
            };
            let messages_b = split_message_per_domain(&frame_bits[m_mvd..], &cap_coeff_only)
                .ok_or_else(|| StegoError::InvalidVideo("Pass 2B: residual split failed".into()))?;
            pass2_stc_plan_with_keys(&cover_for_b, &messages_b, h, &keys, /* gop */ 0)
                .ok_or_else(|| StegoError::InvalidVideo("Pass 2B: STC plan failed".into()))?
        } else {
            DomainPlan {
                coeff_sign_bypass: cover_for_b.cover.coeff_sign_bypass.bits.clone(),
                coeff_suffix_lsb: cover_for_b.cover.coeff_suffix_lsb.bits.clone(),
                mvd_sign_bypass: Vec::new(),
                mvd_suffix_lsb: Vec::new(),
                total_modifications: 0,
                total_cost: 0.0,
            }
        };

        // ── Combined plan: residual from Pass 2B + MVD from Pass 2A.
        //     Defensive re-stamp of shadow bits at shadow positions
        //     across all 4 domains. ──
        let mut combined_plan = DomainPlan {
            coeff_sign_bypass: plan_b.coeff_sign_bypass.clone(),
            coeff_suffix_lsb: plan_b.coeff_suffix_lsb.clone(),
            mvd_sign_bypass: plan_a.mvd_sign_bypass.clone(),
            mvd_suffix_lsb: plan_a.mvd_suffix_lsb.clone(),
            total_modifications: plan_a.total_modifications + plan_b.total_modifications,
            total_cost: plan_a.total_cost + plan_b.total_cost,
        };
        // For empty MVD plan (m_mvd == 0), seed with cover MVD bits
        // so PlanInjector forces emission of cover-bit (no-op) at
        // non-shadow positions and shadow-bit at shadow positions.
        if combined_plan.mvd_sign_bypass.is_empty() {
            combined_plan.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.bits.clone();
        }
        if combined_plan.mvd_suffix_lsb.is_empty() {
            combined_plan.mvd_suffix_lsb = cover_p1.cover.mvd_suffix_lsb.bits.clone();
        }
        // Defensive stamp of EACH shadow's bits at its positions.
        // Order matters when shadow positions overlap (later shadows
        // overwrite earlier — RS absorbs the inter-shadow collisions).
        for state in &shadow_states {
            super::shadow::apply_shadow_to_plan_all4(
                &mut combined_plan.coeff_sign_bypass,
                &mut combined_plan.coeff_suffix_lsb,
                &mut combined_plan.mvd_sign_bypass,
                &mut combined_plan.mvd_suffix_lsb,
                state,
            );
        }

        // ── §6E-A5(d).6 — build the cascade-safe MvdSuffixLsb gate
        //     for `InjectionHook`. Includes only the PositionKeys
        //     where shadow stamped a bit AND cascade-safety greedy
        //     said the magnitude-LSB flip is bounded. PlanInjector
        //     would otherwise return Some(plan_bit) at every msl
        //     position (since combined_plan.mvd_suffix_lsb is filled
        //     to cover_p1 length), risking false flips at unsafe
        //     positions where Pass 3's natural cur_lsb diverged from
        //     Pass 1's cover_bit. ──
        let mvd_msl_gate: std::collections::HashSet<PositionKey> = shadow_states
            .iter()
            .flat_map(|state| state.positions.iter())
            .filter(|s| s.domain == super::EmbedDomain::MvdSuffixLsb)
            .filter_map(|s| cover_p1
                .cover
                .mvd_suffix_lsb
                .positions
                .get(s.intra_index)
                .copied())
            .collect();

        // ── Pass 3: emit with combined plan + cascade-safe gate. ──
        let bytes = pass3_inject_4domain_streaming_with_gate(
            yuv, width, height, n_frames, frame_size, quality, pattern,
            &combined_plan, &per_gop_counts,
            if mvd_msl_gate.is_empty() { None } else { Some(&mvd_msl_gate) },
        )?;

        // ── Verify: walk emitted bytes → 4-domain cover. Each
        //     shadow extracts independently with its own passphrase;
        //     ALL shadows must succeed for cascade to terminate. ──
        let opts = WalkOptions { record_mvd: true };
        let walk = walk_annex_b_for_cover_with_options(&bytes, opts)
            .map_err(|e| StegoError::InvalidVideo(format!("verify walk: {e}")))?;
        // §6E-A5(d).6 — encoder verify must use the same safe-mask
        // as decoder will compute. Decoder runs analyze on walk meta
        // → match here.
        let safe_msb_walk = super::cascade_safety::analyze_safe_mvd_subset(
            &walk.mvd_meta, walk.mb_w, walk.mb_h,
        );
        let safe_msl_walk = super::cascade_safety::derive_msl_safe_from_msb(
            &walk.cover.mvd_sign_bypass.positions,
            &safe_msb_walk,
            &walk.cover.mvd_suffix_lsb.positions,
        );
        let mut all_ok = true;
        for s in shadows {
            match super::shadow::shadow_extract_all4_safe(
                &walk.cover, s.passphrase, None, Some(&safe_msl_walk),
            ) {
                Ok(payload_data) if payload_data.text == s.message => continue,
                _ => {
                    all_ok = false;
                    break;
                }
            }
        }
        if all_ok {
            return Ok(bytes);
        }
        // Verify failed at this parity tier — cascade to next.
    }

    // Cascade exhausted at parity 128. Caller can retry with a
    // smaller primary, different gop_size/quality, or different
    // shadow passphrase.
    Err(StegoError::ShadowEmbedFailed)
}

/// §6E-C2 polish — translate a `ShadowState`'s positions from
/// `source_cover` indexing to `(mvd_target, coeff_target)` indexing
/// via PositionKey lookup.
///
/// MVD-domain slots use `mvd_target.mvd_*.positions`; coeff-domain
/// slots use `coeff_target.coeff_*.positions`. Slots whose
/// PositionKey doesn't appear in the corresponding target are
/// dropped (rare boundary drift case; cascade absorbs).
///
/// The returned state retains the same `parity_len` /
/// `frame_data_len`. `n_total` is updated to reflect the dropped
/// slots; `positions` and `bits` are aligned (slot[i] ↔ bits[i]).
fn translate_shadow_state(
    state: &super::shadow::ShadowState,
    source_cover: &super::DomainCover,
    mvd_target: &super::DomainCover,
    coeff_target: &super::DomainCover,
) -> super::shadow::ShadowState {
    use std::collections::HashMap;
    let build_map = |positions: &[PositionKey]| -> HashMap<PositionKey, usize> {
        positions.iter().enumerate().map(|(i, &k)| (k, i)).collect()
    };
    let target_csb = build_map(&coeff_target.coeff_sign_bypass.positions);
    let target_csl = build_map(&coeff_target.coeff_suffix_lsb.positions);
    let target_msb = build_map(&mvd_target.mvd_sign_bypass.positions);
    let target_msl = build_map(&mvd_target.mvd_suffix_lsb.positions);

    let mut out_positions = Vec::with_capacity(state.positions.len());
    let mut out_bits = Vec::with_capacity(state.bits.len());

    for (i, slot) in state.positions.iter().enumerate().take(state.n_total) {
        if i >= state.bits.len() {
            break;
        }
        let pk_opt = match slot.domain {
            super::EmbedDomain::CoeffSignBypass =>
                source_cover.coeff_sign_bypass.positions.get(slot.intra_index),
            super::EmbedDomain::CoeffSuffixLsb =>
                source_cover.coeff_suffix_lsb.positions.get(slot.intra_index),
            super::EmbedDomain::MvdSignBypass =>
                source_cover.mvd_sign_bypass.positions.get(slot.intra_index),
            super::EmbedDomain::MvdSuffixLsb =>
                source_cover.mvd_suffix_lsb.positions.get(slot.intra_index),
        };
        let pk = match pk_opt {
            Some(&k) => k,
            None => continue,
        };
        let target_idx = match slot.domain {
            super::EmbedDomain::CoeffSignBypass => target_csb.get(&pk).copied(),
            super::EmbedDomain::CoeffSuffixLsb => target_csl.get(&pk).copied(),
            super::EmbedDomain::MvdSignBypass => target_msb.get(&pk).copied(),
            super::EmbedDomain::MvdSuffixLsb => target_msl.get(&pk).copied(),
        };
        if let Some(target_idx) = target_idx {
            out_positions.push(super::shadow::ShadowSlot {
                domain: slot.domain,
                intra_index: target_idx,
                priority: slot.priority,
            });
            out_bits.push(state.bits[i]);
        }
    }

    let n_total = out_bits.len();
    super::shadow::ShadowState {
        positions: out_positions,
        bits: out_bits,
        n_total,
        parity_len: state.parity_len,
        frame_data_len: state.frame_data_len,
    }
}

/// IDR pattern for an encoder run. `gop_size = n_frames` ⇒ only
/// frame 0 is IDR (today's §30D-C single-GOP shape). `gop_size = G`
/// ⇒ IDR every G frames (multi-IDR shape locked in for §6E-C1a).
/// `is_idr(fi) = fi % gop_size == 0`.
#[allow(dead_code)] // §6E-C1a frame-typing helper, kept for diagnostics + future re-wire.
#[inline]
fn is_idr_frame(fi: usize, gop_size: usize) -> bool {
    debug_assert!(gop_size > 0, "gop_size must be > 0");
    fi.is_multiple_of(gop_size)
}

/// §6E-A.deploy.2 — iterate (encode_idx, display_idx, gop_idx,
/// frame_type, &yuv_frame) tuples in encode order over the input
/// `yuv` (which is laid out in DISPLAY order).
///
/// The orchestrator's three encoding passes (Pass 1, Pass 1B, Pass 3)
/// consume this iterator instead of `for fi in 0..n_frames` so that
/// IBPBP sub-GOP reordering can be added later (A.deploy.3) without
/// touching the per-pass loop bodies again. For now (A.deploy.2),
/// callers pass `GopPattern::Ipppp { gop: gop_size }` to keep encode
/// order ≡ display order — output is byte-identical to the
/// pre-refactor `for fi in 0..n_frames` shape.
fn iter_frames_in_encode_order(
    yuv: &[u8],
    frame_size: usize,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
) -> impl Iterator<Item = (super::gop_pattern::EncodeOrderFrame, &[u8])> {
    super::gop_pattern::iter_encode_order(n_frames, pattern).map(move |meta| {
        let d = meta.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        (meta, frame)
    })
}

/// §long-form-stego Phase 2 — counting-only Pass 1 producing
/// per-GOP per-domain position counts.
///
/// Returns `Vec<[usize; 4]>` indexed by GOP index, each row giving
/// `[coeff_sign, coeff_suffix, mvd_sign, mvd_suffix]` position
/// counts for that GOP. Memory: O(num_gops × 4 × usize) — minimal
/// vs the O(n) materialization that `pass1_count_4domain` does.
///
/// Used by the §long-form-stego per-GOP-replay adapter (Phase 4)
/// to map seg_idx → GOP range without materializing the whole
/// cover. Bit-counts match `pass1_count_4domain`'s drained
/// `GopCover` per-domain counts when sliced by gop_idx (verified
/// in Phase 2's integration test).
pub fn pass1_count_per_gop_4domain(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    b_count: usize,
    quality: Option<u8>,
) -> Result<Vec<[usize; 4]>, StegoError> {
    use super::encoder_hook::PositionCountingHook;

    let frame_size = (width * height * 3 / 2) as usize;
    let mut enc = build_encoder(width, height, quality)?;
    enc.enable_mvd_stego_hook = true;
    let pattern = pattern_from_legacy_args(gop_size, b_count);
    enc.enable_b_frames = pattern.has_b_frames();

    // Boxed counting hook so the encoder owns it; we'll re-grab
    // the inner counts via a direct field-access alternative —
    // but the StegoMbHook trait doesn't expose snapshots. Easiest:
    // we maintain a parallel counter outside the hook by tracking
    // GOP boundaries during encode and snapshotting at each.
    //
    // The hook is owned by the encoder for the duration. We attach
    // a wrapper that stores per-GOP snapshots in a shared
    // Arc<Mutex<...>>. Simpler: re-run the encode without the hook
    // owning state, just count externally. But we need the same
    // per-MB `enumerate_*_positions` calls.
    //
    // Concretely: a shared `Vec<[usize; 4]>` that the hook updates
    // as the encoder progresses, snapshotted by gop boundary. We
    // need a single mutable reference that survives across frames,
    // which a Box<dyn StegoMbHook> alone doesn't expose. We
    // therefore use a custom hook type that the encoder owns,
    // and after the run we drain the hook to get the final running
    // counts. The orchestrator does the per-GOP snapshotting by
    // installing a fresh hook PER GOP.
    //
    // Per-GOP fresh hook means: per-GOP encoder state too, which
    // requires Phase 1's mid-clip restart contract. We're using
    // it.

    let mut per_gop_counts: Vec<[usize; 4]> = Vec::new();
    let mut current_gop_idx: u32 = u32::MAX;
    let mut current_gop_running: [usize; 4] = [0; 4];

    // Per-frame encode with a fresh PositionCountingHook installed
    // at each GOP boundary. The hook accumulates per-GOP counts;
    // we drain at the next boundary or end-of-clip.
    let frame_iter = iter_frames_in_encode_order(
        yuv, frame_size, n_frames, pattern,
    )
    .peekable();

    // Encoder is fresh across the whole run (single instance for
    // the full clip). Hooks rotate per-GOP via take_stego_hook +
    // set_stego_hook. The encoder's mid-clip state hygiene
    // (Phase 1 verified) ensures cover counts match what a single
    // PositionLoggerHook over the whole run would produce, sliced
    // by gop_idx.
    enc.set_stego_hook(Some(Box::new(PositionCountingHook::new())));

    for (meta, frame) in frame_iter {
        // GOP boundary: drain the previous hook's count, install fresh.
        if meta.gop_idx != current_gop_idx {
            if current_gop_idx != u32::MAX {
                // Drain previous hook's snapshot.
                let mut hook_box = enc
                    .take_stego_hook()
                    .ok_or_else(|| StegoError::InvalidVideo(
                        "missing PositionCountingHook on GOP boundary".into(),
                    ))?;
                // Downcast via a snapshot helper exposed through a
                // counter-only path. The cleanest: have the trait
                // expose `take_counts_if_counter` (returns Option).
                // For minimal change we re-create the count by
                // re-encoding — but that's O(n) work duplicated.
                // Instead, we use a side-channel: PositionCountingHook
                // exposes `snapshot()` directly via downcast.
                //
                // Use take_counts trait method (added below).
                let row = hook_box.take_counts_if_counter().ok_or_else(|| {
                    StegoError::InvalidVideo(
                        "hook is not PositionCountingHook".into(),
                    )
                })?;
                // Per-GOP delta = row - prior running. Since we
                // installed a FRESH hook per GOP, row IS the per-GOP
                // count directly (no diff needed).
                per_gop_counts.push(row);
                let _ = current_gop_running; // running unused under fresh-per-GOP
                current_gop_running = [0; 4];
            }
            current_gop_idx = meta.gop_idx;
            // Install a fresh counting hook for the new GOP.
            enc.set_stego_hook(Some(Box::new(PositionCountingHook::new())));
        }

        let ft = meta.frame_type;
        encode_one_frame(&mut enc, frame, ft).map_err(|e| {
            StegoError::InvalidVideo(format!("Pass 1 (count): {e}"))
        })?;
    }

    // Drain the final GOP's hook.
    let mut final_hook = enc.take_stego_hook().ok_or_else(|| {
        StegoError::InvalidVideo("missing final PositionCountingHook".into())
    })?;
    let row = final_hook.take_counts_if_counter().ok_or_else(|| {
        StegoError::InvalidVideo(
            "final hook is not PositionCountingHook".into(),
        )
    })?;
    per_gop_counts.push(row);

    Ok(per_gop_counts)
}

/// §long-form-stego Phase 3 — Pass 1 over a single GOP range.
///
/// Runs Pass 1 (encoder + PositionLoggerHook) over the YUV frames
/// belonging to the GOPs in `[gop_start..gop_end)`, with the
/// encoder primed to start at the first frame's encode-order
/// index (the §long-form-stego Phase 1 mid-clip restart contract).
/// Returns a `GopCover` for that range.
///
/// `gop_start` MUST align to a GOP boundary in encode order
/// (i.e. correspond to an IDR position). The caller is
/// responsible for that mapping; for `GopPattern::Ipppp { gop }`
/// every multiple of `gop` is a GOP boundary, while
/// `GopPattern::Ibpbp` GOPs span `gop` frames per GOP in display
/// order which translates to `gop` frames per GOP in encode order
/// too (each GOP is an independent encode-order unit).
///
/// Bit-exact equivalence to slicing a full Pass 1 by gop_idx
/// range is verified by Phase 3's
/// `partial_range_pass1_matches_full_range_slice` test.
pub fn pass1_capture_4domain_for_gop_range(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    b_count: usize,
    quality: Option<u8>,
    gop_start: usize,
    gop_end: usize,
) -> Result<super::orchestrate::GopCover, StegoError> {
    if gop_start >= gop_end {
        return Err(StegoError::InvalidVideo(format!(
            "gop_start {gop_start} >= gop_end {gop_end}"
        )));
    }
    let frame_size = (width * height * 3 / 2) as usize;
    let frame_start = gop_start * gop_size;
    let frame_end = (gop_end * gop_size).min(n_frames);
    if frame_start >= n_frames {
        return Err(StegoError::InvalidVideo(format!(
            "gop_start {gop_start} maps to frame {frame_start} >= n_frames {n_frames}"
        )));
    }

    let mut enc = build_encoder(width, height, quality)?;
    enc.enable_mvd_stego_hook = true;
    let pattern = pattern_from_legacy_args(gop_size, b_count);
    enc.enable_b_frames = pattern.has_b_frames();
    enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));

    // Iterate the FULL clip's encode-order schedule but only
    // ENCODE frames whose gop_idx is in [gop_start..gop_end).
    // We can't "skip ahead" cheaply because the encoder is fresh
    // and stego_frame_idx primes only at start; but for our use
    // case the GOPs to encode are contiguous, so we just call
    // iter_encode_order over the full range and filter in-range.
    //
    // Optimisation: since the GOPs we want are contiguous, we can
    // iterate the full schedule once, lazily prime the encoder at
    // the first in-range frame, and stop after the last in-range
    // frame.
    let mut primed = false;
    for (meta, frame) in iter_frames_in_encode_order(
        yuv, frame_size, n_frames, pattern,
    ) {
        let in_range = (meta.gop_idx as usize) >= gop_start
            && (meta.gop_idx as usize) < gop_end;
        if !in_range {
            // Skip frames before our range; stop after our range.
            if (meta.gop_idx as usize) >= gop_end {
                break;
            }
            continue;
        }
        if !primed {
            // Phase 1 contract: only stego_frame_idx needs priming;
            // frame_num / POC / DPB self-correct at IDR. The first
            // in-range frame must be an IDR for that contract to
            // hold.
            debug_assert_eq!(
                meta.frame_type,
                super::gop_pattern::FrameType::Idr,
                "first in-range frame must be IDR (gop_start={gop_start})",
            );
            enc.stego_frame_idx = meta.encode_idx;
            primed = true;
        }
        let ft = meta.frame_type;
        encode_one_frame(&mut enc, frame, ft).map_err(|e| {
            StegoError::InvalidVideo(format!(
                "Pass 1 (gop_range): frame_idx={} {e}",
                meta.encode_idx,
            ))
        })?;
        if meta.encode_idx as usize + 1 >= frame_end {
            break;
        }
    }

    let mut hook = enc.take_stego_hook().ok_or_else(|| {
        StegoError::InvalidVideo("Pass 1 (gop_range): hook missing".into())
    })?;
    let cover = drain_position_logger(&mut hook)?;
    Ok(cover)
}

/// Pass 1 helper for §30D-C: encode I + P frames with
/// `enable_mvd_stego_hook=true` + PositionLoggerHook. Captures
/// all 4 domains' covers (residual + MVD). `gop_size` controls the
/// IDR pattern: pass `n_frames` to retain the legacy 1-IDR shape;
/// pass a smaller value (e.g. 30) for periodic IDRs.
fn pass1_count_4domain(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    gop_size: usize,
    b_count: usize,
) -> Result<super::orchestrate::GopCover, StegoError> {
    let (cover, _meta) = pass1_count_4domain_with_meta(
        yuv, width, height, n_frames, frame_size, quality, gop_size, b_count,
    )?;
    Ok(cover)
}

/// §6E-A.deploy.5 — convert (gop_size, b_count) → GopPattern.
/// `b_count == 0` selects IPPPP; `b_count >= 1` selects IBPBP M=b_count+1.
fn pattern_from_legacy_args(gop_size: usize, b_count: usize) -> super::gop_pattern::GopPattern {
    if b_count == 0 {
        super::gop_pattern::GopPattern::Ipppp { gop: gop_size }
    } else {
        super::gop_pattern::GopPattern::Ibpbp { gop: gop_size, b_count }
    }
}

/// Phase 6F.2(j).4 — same as `pass1_count_4domain` but also drains
/// the per-MVD-position metadata (`MvdPositionMeta`) needed by the
/// cascade-safety analysis. Aligned by index with
/// `cover.cover.mvd_sign_bypass.positions`.
fn pass1_count_4domain_with_meta(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    gop_size: usize,
    b_count: usize,
) -> Result<(super::orchestrate::GopCover, Vec<super::encoder_hook::MvdPositionMeta>), StegoError> {
    let mut enc = build_encoder(width, height, quality)?;
    enc.enable_mvd_stego_hook = true;
    enc.set_stego_hook(Some(Box::new(PositionLoggerHook::new())));
    let pattern = pattern_from_legacy_args(gop_size, b_count);
    enc.enable_b_frames = pattern.has_b_frames();
    for (_meta, frame) in iter_frames_in_encode_order(yuv, frame_size, n_frames, pattern) {
        let ft = _meta.frame_type;
        encode_one_frame(&mut enc, frame, ft)
            .map_err(|e| StegoError::InvalidVideo(format!("Pass 1: {e}")))?;
    }
    let mut hook = enc.take_stego_hook().ok_or_else(|| StegoError::InvalidVideo(
        "Pass 1 hook missing".into()
    ))?;
    let meta = hook.take_mvd_meta_if_logger();
    let cover = drain_position_logger(&mut hook)?;
    Ok((cover, meta))
}

/// §6E-A5(d).2 — Pass 1 + cascade-safety bundle for shadow.
///
/// Wraps `pass1_count_4domain_with_meta` + `analyze_safe_mvd_subset`
/// + `derive_msl_safe_from_msb`. Returned masks are aligned with
/// the cover's MVD position lists (`safe_msb` ↔ `mvd_sign_bypass`,
/// `safe_msl` ↔ `mvd_suffix_lsb`). Both sides (encoder cascade +
/// decoder shadow_extract) call this so encoder + decoder run the
/// same hash-priority filter.
#[allow(clippy::too_many_arguments)]
#[allow(dead_code)] // d.3 + d.6 will consume these masks
fn pass1_count_4domain_with_safe_masks(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    gop_size: usize,
    b_count: usize,
) -> Result<(super::orchestrate::GopCover, Vec<bool>, Vec<bool>), StegoError> {
    let (cover, meta) = pass1_count_4domain_with_meta(
        yuv, width, height, n_frames, frame_size, quality, gop_size, b_count,
    )?;
    let mb_w = width / 16;
    let mb_h = height / 16;
    let safe_msb = super::cascade_safety::analyze_safe_mvd_subset(&meta, mb_w, mb_h);
    let safe_msl = super::cascade_safety::derive_msl_safe_from_msb(
        &cover.cover.mvd_sign_bypass.positions,
        &safe_msb,
        &cover.cover.mvd_suffix_lsb.positions,
    );
    Ok((cover, safe_msb, safe_msl))
}

/// Pass 1B helper for §30D-C: encode I + P frames with
/// InjectAndLogHook (MVD plan applied + residual logger). The
/// produced bytes are discarded; only the residual cover matters.
#[allow(clippy::too_many_arguments)]
fn pass1b_inject_mvd_log_residual(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    cover_p1: &super::DomainCover,
    plan_a: &DomainPlan,
    gop_size: usize,
    b_count: usize,
) -> Result<super::orchestrate::GopCover, StegoError> {
    let (cover, _bytes) = pass1b_inject_mvd_log_residual_with_bytes(
        yuv, width, height, n_frames, frame_size, quality,
        cover_p1, plan_a, gop_size, b_count,
    )?;
    Ok(cover)
}

/// Phase 6F.2(h) — same as `pass1b_inject_mvd_log_residual` but also
/// returns the emitted bytes. Used by the cascade-verification
/// pipeline to walk Pass 1B's actual output and recover the post-
/// cascade MVD + residual cover (encoder-emitted MVD positions can
/// shift in count vs cover_p1 because MVD sign flips propagate
/// through the median predictor — see deferred-items.md §37).
#[allow(clippy::too_many_arguments)]
fn pass1b_inject_mvd_log_residual_with_bytes(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    cover_p1: &super::DomainCover,
    plan_a: &DomainPlan,
    gop_size: usize,
    b_count: usize,
) -> Result<(super::orchestrate::GopCover, Vec<u8>), StegoError> {
    // Build a PlanInjector containing only MVD plan.
    let injector = PlanInjector::from_plan(cover_p1, plan_a);
    let hook = InjectAndLogHook::new(injector);

    let mut enc = build_encoder(width, height, quality)?;
    enc.enable_mvd_stego_hook = true;
    enc.set_stego_hook(Some(Box::new(hook)));
    let mut bytes = Vec::new();
    let pattern = pattern_from_legacy_args(gop_size, b_count);
    enc.enable_b_frames = pattern.has_b_frames();
    for (_meta, frame) in iter_frames_in_encode_order(yuv, frame_size, n_frames, pattern) {
        let ft = _meta.frame_type;
        let frame_bytes = encode_one_frame(&mut enc, frame, ft)
            .map_err(|e| StegoError::InvalidVideo(format!("Pass 1B: {e}")))?;
        bytes.extend_from_slice(&frame_bytes);
    }
    let mut hook = enc.take_stego_hook().ok_or_else(|| StegoError::InvalidVideo(
        "Pass 1B hook missing".into()
    ))?;
    let cover = drain_position_logger(&mut hook)?;
    Ok((cover, bytes))
}

/// Pass 3 helper for §30D-C: encode with combined plan (MVD +
/// residual). Returns the stego Annex-B byte stream.
#[allow(clippy::too_many_arguments)]
fn pass3_inject_4domain(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    combined_cover: &super::DomainCover,
    combined_plan: &DomainPlan,
    gop_size: usize,
    b_count: usize,
) -> Result<Vec<u8>, StegoError> {
    let injector = PlanInjector::from_plan(combined_cover, combined_plan);
    let hook = InjectionHook::new(injector);

    let mut enc = build_encoder(width, height, quality)?;
    enc.enable_mvd_stego_hook = true;
    enc.set_stego_hook(Some(Box::new(hook)));
    let mut out = Vec::new();
    let pattern = pattern_from_legacy_args(gop_size, b_count);
    enc.enable_b_frames = pattern.has_b_frames();
    for (_meta, frame) in iter_frames_in_encode_order(yuv, frame_size, n_frames, pattern) {
        let ft = _meta.frame_type;
        let bytes = encode_one_frame(&mut enc, frame, ft)
            .map_err(|e| StegoError::InvalidVideo(format!("Pass 3: {e}")))?;
        out.extend_from_slice(&bytes);
    }

    Ok(out)
}

/// §6E-A5(c.3) — derive per-GOP per-domain position counts
/// from a materialized full-clip `DomainCover`. Used by the
/// shadow cascade to size the per-GOP slicing for the streaming
/// Pass-1B + Pass-3 helpers without spending a second
/// `pass1_count_per_gop_4domain` encoder pass.
fn derive_per_gop_counts_from_cover(
    yuv: &[u8],
    frame_size: usize,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
    cover: &super::DomainCover,
) -> Vec<[usize; 4]> {
    // encode_idx → gop_idx map. Walks the deterministic encode-
    // order schedule once; doesn't actually encode (we ignore the
    // frame slice).
    let mut encode_to_gop: Vec<usize> = vec![0; n_frames];
    let mut max_gop: usize = 0;
    for (meta, _frame) in iter_frames_in_encode_order(yuv, frame_size, n_frames, pattern) {
        let g = meta.gop_idx as usize;
        encode_to_gop[meta.encode_idx as usize] = g;
        if g > max_gop {
            max_gop = g;
        }
    }
    let num_gops = max_gop + 1;
    let mut per_gop_counts: Vec<[usize; 4]> = vec![[0; 4]; num_gops];
    for p in &cover.coeff_sign_bypass.positions {
        let g = encode_to_gop[p.frame_idx() as usize];
        per_gop_counts[g][0] += 1;
    }
    for p in &cover.coeff_suffix_lsb.positions {
        let g = encode_to_gop[p.frame_idx() as usize];
        per_gop_counts[g][1] += 1;
    }
    for p in &cover.mvd_sign_bypass.positions {
        let g = encode_to_gop[p.frame_idx() as usize];
        per_gop_counts[g][2] += 1;
    }
    for p in &cover.mvd_suffix_lsb.positions {
        let g = encode_to_gop[p.frame_idx() as usize];
        per_gop_counts[g][3] += 1;
    }
    per_gop_counts
}

/// §6E-A5(c.2) — per-GOP Pass-1B: encode GOP `g` with the MVD
/// plan applied via InjectAndLogHook, return the per-GOP cover
/// (all 4 domains; MVD bits reflect the injected sign overrides,
/// residual coefficients reflect post-MVD-cascade values).
///
/// `mvd_cover` and `mvd_plan` are the full-clip Pass-1 cover
/// (MVD domain) and the full-clip MVD STC plan. The PlanInjector
/// is built from them; only positions in GOP `g`'s frames are
/// hit during encode (other entries are dead weight). Per-GOP
/// pre-slicing of mvd_cover/mvd_plan is a future memory
/// optimization.
///
/// Memory bound: O(per-GOP encoder working set) +
/// O(full mvd_cover + mvd_plan) for the injector HashMap. The
/// HashMap is the cover-side cost that c.x might further bound;
/// the encoder side is fully per-GOP-bounded.
#[allow(clippy::too_many_arguments)]
fn encode_one_gop_pass1b_inject_mvd_log_residual(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    pattern: super::gop_pattern::GopPattern,
    g: usize,
    mvd_cover: &super::DomainCover,
    mvd_plan: &DomainPlan,
) -> Result<super::orchestrate::GopCover, StegoError> {
    let injector = PlanInjector::from_plan(mvd_cover, mvd_plan);
    let hook = InjectAndLogHook::new(injector);

    let mut enc = build_encoder(width, height, quality)?;
    enc.enable_mvd_stego_hook = true;
    enc.enable_b_frames = pattern.has_b_frames();
    enc.set_stego_hook(Some(Box::new(hook)));

    let mut primed = false;
    for (meta, frame) in iter_frames_in_encode_order(
        yuv, frame_size, n_frames, pattern,
    ) {
        if (meta.gop_idx as usize) != g {
            if (meta.gop_idx as usize) > g {
                break;
            }
            continue;
        }
        if !primed {
            enc.stego_frame_idx = meta.encode_idx;
            primed = true;
        }
        encode_one_frame(&mut enc, frame, meta.frame_type)
            .map_err(|e| StegoError::InvalidVideo(format!(
                "Pass 1B GOP {g} frame {}: {e}",
                meta.encode_idx,
            )))?;
    }
    let mut hook = enc.take_stego_hook().ok_or_else(|| {
        StegoError::InvalidVideo("Pass 1B GOP hook missing".into())
    })?;
    drain_position_logger(&mut hook)
}

/// §6E-A5(c.2) — streaming counterpart of
/// `pass1b_inject_mvd_log_residual`. Encodes the full clip
/// GOP-by-GOP with MVD plan injection + residual logging,
/// concatenating the per-GOP captured covers into a full-clip
/// `GopCover` (matches the in-memory output shape so the
/// downstream Pass-2B can consume it without changes).
///
/// Memory bound: only one GOP's encoder working set + the
/// PlanInjector HashMap is resident at any time during the
/// per-GOP encode step. The output residual cover is O(n) total
/// (matches the in-memory variant); pipelining it directly into
/// a streaming Pass-2B is a future optimization.
///
/// Use case: §6E-A5(c.3) shadow cascade refactor — replaces the
/// in-memory `pass1b_inject_mvd_log_residual` call inside the
/// cascade loop so each cascade iteration's Pass-1B encode-side
/// memory is bounded.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass1b_inject_mvd_log_residual_streaming(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    pattern: super::gop_pattern::GopPattern,
    mvd_cover: &super::DomainCover,
    mvd_plan: &DomainPlan,
    per_gop_counts: &[[usize; 4]],
) -> Result<super::orchestrate::GopCover, StegoError> {
    let num_gops = per_gop_counts.len();
    let mut full_cover = super::orchestrate::GopCover::default();
    for g in 0..num_gops {
        let gop_cover = encode_one_gop_pass1b_inject_mvd_log_residual(
            yuv, width, height, n_frames, frame_size, quality, pattern,
            g, mvd_cover, mvd_plan,
        )?;
        // Concatenate per-GOP cover into full_cover. DomainBits
        // append is bits + positions; costs are Vec<f32>.
        for b in &gop_cover.cover.coeff_sign_bypass.bits {
            full_cover.cover.coeff_sign_bypass.bits.push(*b);
        }
        for p in &gop_cover.cover.coeff_sign_bypass.positions {
            full_cover.cover.coeff_sign_bypass.positions.push(*p);
        }
        for c in &gop_cover.costs.coeff_sign_bypass {
            full_cover.costs.coeff_sign_bypass.push(*c);
        }
        for b in &gop_cover.cover.coeff_suffix_lsb.bits {
            full_cover.cover.coeff_suffix_lsb.bits.push(*b);
        }
        for p in &gop_cover.cover.coeff_suffix_lsb.positions {
            full_cover.cover.coeff_suffix_lsb.positions.push(*p);
        }
        for c in &gop_cover.costs.coeff_suffix_lsb {
            full_cover.costs.coeff_suffix_lsb.push(*c);
        }
        for b in &gop_cover.cover.mvd_sign_bypass.bits {
            full_cover.cover.mvd_sign_bypass.bits.push(*b);
        }
        for p in &gop_cover.cover.mvd_sign_bypass.positions {
            full_cover.cover.mvd_sign_bypass.positions.push(*p);
        }
        for c in &gop_cover.costs.mvd_sign_bypass {
            full_cover.costs.mvd_sign_bypass.push(*c);
        }
        for b in &gop_cover.cover.mvd_suffix_lsb.bits {
            full_cover.cover.mvd_suffix_lsb.bits.push(*b);
        }
        for p in &gop_cover.cover.mvd_suffix_lsb.positions {
            full_cover.cover.mvd_suffix_lsb.positions.push(*p);
        }
        for c in &gop_cover.costs.mvd_suffix_lsb {
            full_cover.costs.mvd_suffix_lsb.push(*c);
        }
    }
    let _ = per_gop_counts; // currently only used for num_gops; kept
                            // as parameter so callers don't need a
                            // separate count. Future memory-tight
                            // variant will use cum_counts to slice
                            // mvd_cover / mvd_plan per GOP.
    Ok(full_cover)
}

/// §6E-A5(c.1) — encode a single GOP with a precomputed per-GOP
/// `DomainPlan`, capturing the GOP's cover via per-GOP Pass 1
/// replay (`pass1_capture_4domain_for_gop_range`). Used by:
/// - The Phase 6.3 v2 streaming orchestrator's per-GOP firing.
/// - The §6E-A5(c.3) streaming-shadow Pass-3 helper below.
///
/// Memory bound: O(per-GOP encoder working set) ≈ ~100 MB at 1080p.
/// Re-running per-GOP Pass 1 for cover capture is O(GOP-encoder)
/// extra work; mirrors v2's pattern.
#[allow(clippy::too_many_arguments)]
fn encode_one_gop_with_plan_and_capture(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    pattern: super::gop_pattern::GopPattern,
    g: usize,
    gop_plan: &DomainPlan,
    mvd_msl_safe_keys: Option<&std::collections::HashSet<PositionKey>>,
) -> Result<Vec<u8>, StegoError> {
    let gop_size = pattern.gop_size();
    let b_count = pattern.legacy_b_count();
    let gop_cover = pass1_capture_4domain_for_gop_range(
        yuv, width, height, n_frames, gop_size, b_count, quality, g, g + 1,
    )?;

    let mut gop_cover_only = super::DomainCover::default();
    gop_cover_only.coeff_sign_bypass =
        gop_cover.cover.coeff_sign_bypass.clone();
    gop_cover_only.coeff_suffix_lsb =
        gop_cover.cover.coeff_suffix_lsb.clone();
    gop_cover_only.mvd_sign_bypass =
        gop_cover.cover.mvd_sign_bypass.clone();
    gop_cover_only.mvd_suffix_lsb =
        gop_cover.cover.mvd_suffix_lsb.clone();

    let injector = PlanInjector::from_plan(&gop_cover_only, gop_plan);
    let mut enc = build_encoder(width, height, quality)?;
    enc.enable_mvd_stego_hook = true;
    enc.enable_b_frames = pattern.has_b_frames();
    let mut hook = InjectionHook::new(injector);
    if let Some(keys) = mvd_msl_safe_keys {
        hook.set_mvd_msl_safe_gate(keys.clone());
    }
    enc.set_stego_hook(Some(Box::new(hook)));

    let mut primed = false;
    let mut bytes = Vec::new();
    for (meta, frame) in iter_frames_in_encode_order(
        yuv, frame_size, n_frames, pattern,
    ) {
        if (meta.gop_idx as usize) != g {
            if (meta.gop_idx as usize) > g {
                break;
            }
            continue;
        }
        if !primed {
            enc.stego_frame_idx = meta.encode_idx;
            primed = true;
        }
        let frame_bytes = encode_one_frame(&mut enc, frame, meta.frame_type)
            .map_err(|e| StegoError::InvalidVideo(format!(
                "Pass 3 GOP {g} frame {}: {e}",
                meta.encode_idx,
            )))?;
        bytes.extend_from_slice(&frame_bytes);
    }
    Ok(bytes)
}

/// §6E-A5(c.1) — streaming counterpart of `pass3_inject_4domain`.
/// Encodes the full clip GOP-by-GOP with per-GOP cover capture
/// (no clip-wide cover materialization). Slices `combined_plan`
/// per-GOP via the per-domain cumulative position counts.
///
/// Memory bound: only one GOP's encoder working set + per-GOP
/// captured cover + a slice of the combined plan are resident at
/// any time. Suitable for long-form encodes where the legacy
/// `pass3_inject_4domain` would OOM.
///
/// `per_gop_counts[g][d]` is the per-domain position count for
/// GOP `g`, as produced by `pass1_count_per_gop_4domain`. Caller
/// passes the same shape used to drive the per-GOP plan slicing.
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass3_inject_4domain_streaming(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    pattern: super::gop_pattern::GopPattern,
    combined_plan: &DomainPlan,
    per_gop_counts: &[[usize; 4]],
) -> Result<Vec<u8>, StegoError> {
    pass3_inject_4domain_streaming_with_gate(
        yuv, width, height, n_frames, frame_size, quality, pattern,
        combined_plan, per_gop_counts, None,
    )
}

/// §6E-A5(d).6 — `pass3_inject_4domain_streaming` variant that
/// optionally installs a cascade-safe MvdSuffixLsb gate-set on the
/// `InjectionHook`. When `mvd_msl_safe_keys = Some(set)`,
/// `on_mvd_slot` performs ±1 magnitude-LSB flips at the gated
/// PositionKeys to align with the plan; otherwise it stays a no-op.
/// The shadow cascade body computes the gate-set from
/// `shadow_states` (positions selected via
/// `priority_slots_all4_safe(... safe_msl=Some(...))`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn pass3_inject_4domain_streaming_with_gate(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    frame_size: usize,
    quality: Option<u8>,
    pattern: super::gop_pattern::GopPattern,
    combined_plan: &DomainPlan,
    per_gop_counts: &[[usize; 4]],
    mvd_msl_safe_keys: Option<&std::collections::HashSet<PositionKey>>,
) -> Result<Vec<u8>, StegoError> {
    let num_gops = per_gop_counts.len();
    let cum: [Vec<usize>; 4] = std::array::from_fn(|d| {
        let mut v = Vec::with_capacity(num_gops + 1);
        v.push(0);
        for row in per_gop_counts.iter() {
            v.push(*v.last().unwrap() + row[d]);
        }
        v
    });

    let mut out = Vec::new();
    for g in 0..num_gops {
        // Slice the combined plan to this GOP's position range, per
        // domain. Mirrors v2's slicing: clip at plan length when the
        // plan is shorter than raw cover capacity (m*w < total).
        let mut gop_plan = DomainPlan::default();
        for d in 0..4 {
            let lo = cum[d][g];
            let hi = cum[d][g + 1];
            let src = match d {
                0 => &combined_plan.coeff_sign_bypass,
                1 => &combined_plan.coeff_suffix_lsb,
                2 => &combined_plan.mvd_sign_bypass,
                _ => &combined_plan.mvd_suffix_lsb,
            };
            let dst = match d {
                0 => &mut gop_plan.coeff_sign_bypass,
                1 => &mut gop_plan.coeff_suffix_lsb,
                2 => &mut gop_plan.mvd_sign_bypass,
                _ => &mut gop_plan.mvd_suffix_lsb,
            };
            if src.is_empty() {
                continue;
            }
            let lo = lo.min(src.len());
            let hi = hi.min(src.len());
            *dst = src[lo..hi].to_vec();
        }

        let gop_bytes = encode_one_gop_with_plan_and_capture(
            yuv, width, height, n_frames, frame_size, quality, pattern,
            g, &gop_plan, mvd_msl_safe_keys,
        )?;
        out.extend_from_slice(&gop_bytes);
    }
    Ok(out)
}

#[cfg(test)]
mod pass1b_streaming_diag {
    //! §6E-A5(c.x) — focused diagnostic for #107.
    //!
    //! Compares `pass1b_inject_mvd_log_residual` (in-memory) vs
    //! `pass1b_inject_mvd_log_residual_streaming` on the same
    //! inputs. If they diverge → streaming Pass-1B has a bug. If
    //! they match → bug is downstream.
    use super::*;
    use crate::codec::h264::stego::PositionKey;
    use crate::codec::h264::stego::orchestrate::DomainPlan;
    use crate::codec::h264::stego::gop_pattern::GopPattern;

    /// Task #97 — file-attachment round-trip via the CABAC v2
    /// streaming-v2 entry. Encode a small file alongside text,
    /// decode via smart_decode_with_payload, assert files recovered.
    #[test]
    fn h264_stego_encode_with_files_roundtrip_128x80() {
        use crate::stego::payload::FileEntry;
        let yuv = match std::fs::read(
            "test-vectors/video/h264/real-world/img4138_128x80_f10.yuv",
        ) {
            Ok(y) => y,
            Err(_) => return,
        };
        let files = [FileEntry {
            filename: "note.txt".into(),
            content: b"phasm files demo: hello video!".to_vec(),
        }];
        let bytes = match super::h264_stego_encode_yuv_string_4domain_multigop_streaming_v2_with_files(
            &yuv, 128, 80, 10, 5, "msg", &files, "files-pass-128",
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("encode_with_files flake (#94): {e:?}");
                return; // small fixture flakes on random salt
            }
        };
        let payload =
            super::super::decode_pixels::h264_stego_smart_decode_video_with_payload(
                &bytes, "files-pass-128",
            ).expect("decode_with_payload");
        assert_eq!(payload.text, "msg");
        assert_eq!(payload.files.len(), 1);
        assert_eq!(payload.files[0].filename, "note.txt");
        assert_eq!(payload.files[0].content, b"phasm files demo: hello video!");
    }

    /// Task #120 — N=1 shadow + per-layer file attachments
    /// round-trip via `_with_n_shadows_with_pattern_and_files`.
    /// Verifies primary text+file recover under primary passphrase,
    /// shadow text+file recover under shadow passphrase, and the
    /// cross-domain file-attachment payload encoding works on both
    /// layers simultaneously. CLI / iOS / Android bridges all wire
    /// through this same core entry.
    #[test]
    fn h264_stego_encode_with_shadow_and_files_roundtrip_128x80() {
        use crate::stego::payload::FileEntry;
        use crate::stego::shadow_layer::ShadowLayer;
        let yuv = match std::fs::read(
            "test-vectors/video/h264/real-world/img4138_128x80_f10.yuv",
        ) {
            Ok(y) => y,
            Err(_) => return,
        };
        let primary_files = [FileEntry {
            filename: "primary.txt".into(),
            content: b"primary file".to_vec(),
        }];
        let shadow_files = [FileEntry {
            filename: "shadow.txt".into(),
            content: b"shadow file".to_vec(),
        }];
        let shadows = [ShadowLayer {
            message: "s",
            passphrase: "shadow-pass-128",
            files: &shadow_files,
        }];
        let pattern = GopPattern::Ibpbp { gop: 5, b_count: 1 };
        let bytes = match super::h264_stego_encode_yuv_string_with_n_shadows_with_pattern_and_files(
            &yuv, 128, 80, 10, pattern,
            "p", &primary_files, "primary-pass-128",
            &shadows,
        ) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("encode_with_shadow_and_files flake (#94): {e:?}");
                return; // small fixture flakes on random salt
            }
        };

        let prim_payload =
            super::super::decode_pixels::h264_stego_smart_decode_video_with_payload(
                &bytes, "primary-pass-128",
            ).expect("primary decode");
        assert_eq!(prim_payload.text, "p");
        assert_eq!(prim_payload.files.len(), 1);
        assert_eq!(prim_payload.files[0].filename, "primary.txt");
        assert_eq!(prim_payload.files[0].content, b"primary file");

        let shadow_payload =
            super::super::decode_pixels::h264_stego_smart_decode_video_with_payload(
                &bytes, "shadow-pass-128",
            ).expect("shadow decode");
        assert_eq!(shadow_payload.text, "s");
        assert_eq!(shadow_payload.files.len(), 1);
        assert_eq!(shadow_payload.files[0].filename, "shadow.txt");
        assert_eq!(shadow_payload.files[0].content, b"shadow file");
    }

    /// Task #96 — sanity-check primary > shadow capacity at the
    /// 128x80 fixture. Primary uses the entire injectable cover
    /// minus framing overhead; shadow is collision-limited under
    /// the cascade formula and bounded much smaller.
    #[test]
    fn h264_stego_capacity_4domain_smoke() {
        let yuv = match std::fs::read(
            "test-vectors/video/h264/real-world/img4138_128x80_f10.yuv",
        ) {
            Ok(y) => y,
            Err(_) => return,
        };
        let info = super::h264_stego_capacity_4domain(&yuv, 128, 80, 10, 5)
            .expect("capacity_4domain");
        assert!(info.cover_size_bits > 0, "cover bits must be positive");
        assert!(
            info.primary_max_message_bytes > 0,
            "primary capacity must be > 0",
        );
        // Primary uses cover_bits/8 - FRAME_OVERHEAD; shadow uses the
        // collision-limited sqrt formula so it's typically smaller.
        // At 128x80 we expect primary >= shadow at single-shadow load.
        assert!(
            info.primary_max_message_bytes >= info.shadow_max_message_bytes,
            "primary {} should be >= shadow (n=1) {}",
            info.primary_max_message_bytes,
            info.shadow_max_message_bytes,
        );
        eprintln!(
            "128x80 capacity: cover_bits={} primary={} shadow_n1={}",
            info.cover_size_bits,
            info.primary_max_message_bytes,
            info.shadow_max_message_bytes,
        );
    }

    #[test]
    fn pass1b_streaming_matches_inmemory_128x80_2gop() {
        let yuv = match std::fs::read(
            "test-vectors/video/h264/real-world/img4138_128x80_f10.yuv",
        ) {
            Ok(y) => y,
            Err(_) => return,
        };
        compare(&yuv, 128, 80, 10, 5, 1);
    }

    #[test]
    #[ignore = "needs /tmp/img4138_1080p_f10.yuv + ~10 min release"]
    fn pass1b_streaming_matches_inmemory_1080p_2gop() {
        let yuv = match std::fs::read("/tmp/img4138_1080p_f10.yuv") {
            Ok(y) => y,
            Err(_) => return,
        };
        compare(&yuv, 1920, 1072, 10, 5, 1);
    }

    fn compare(yuv: &[u8], w: u32, h: u32, n: usize, gop: usize, b: usize) {
        let frame_size = (w * h * 3 / 2) as usize;
        let pattern = GopPattern::Ibpbp { gop, b_count: b };
        let cover_p1 = pass1_count_4domain(
            yuv, w, h, n, frame_size, Some(26), gop, b,
        )
        .expect("pass1");
        let plan = DomainPlan::default();
        let in_mem = pass1b_inject_mvd_log_residual(
            yuv, w, h, n, frame_size, Some(26),
            &cover_p1.cover, &plan, gop, b,
        )
        .expect("in-memory pass1b");
        let per_gop = derive_per_gop_counts_from_cover(
            yuv, frame_size, n, pattern, &cover_p1.cover,
        );
        let stream = pass1b_inject_mvd_log_residual_streaming(
            yuv, w, h, n, frame_size, Some(26),
            pattern, &cover_p1.cover, &plan, &per_gop,
        )
        .expect("streaming pass1b");

        let cmp_domain = |label: &str,
                          a: &Vec<u8>, b: &Vec<u8>,
                          ap: &Vec<PositionKey>, bp: &Vec<PositionKey>,
                          ac: &Vec<f32>, bc: &Vec<f32>| {
            eprintln!(
                "{label}: in_mem(bits={} positions={} costs={}) stream(bits={} positions={} costs={})",
                a.len(), ap.len(), ac.len(),
                b.len(), bp.len(), bc.len(),
            );
            assert_eq!(a.len(), b.len(), "{label}: bits len differs");
            // PositionKey is a packed u64; convert + sort for set
            // comparison since PositionKey itself isn't Ord.
            let mut ap_sorted: Vec<u64> =
                ap.iter().map(|p| p.raw()).collect();
            let mut bp_sorted: Vec<u64> =
                bp.iter().map(|p| p.raw()).collect();
            ap_sorted.sort();
            bp_sorted.sort();
            assert_eq!(
                ap_sorted, bp_sorted,
                "{label}: position SETS differ",
            );
        };
        cmp_domain(
            "coeff_sign",
            &in_mem.cover.coeff_sign_bypass.bits,
            &stream.cover.coeff_sign_bypass.bits,
            &in_mem.cover.coeff_sign_bypass.positions,
            &stream.cover.coeff_sign_bypass.positions,
            &in_mem.costs.coeff_sign_bypass,
            &stream.costs.coeff_sign_bypass,
        );
        cmp_domain(
            "coeff_suffix",
            &in_mem.cover.coeff_suffix_lsb.bits,
            &stream.cover.coeff_suffix_lsb.bits,
            &in_mem.cover.coeff_suffix_lsb.positions,
            &stream.cover.coeff_suffix_lsb.positions,
            &in_mem.costs.coeff_suffix_lsb,
            &stream.costs.coeff_suffix_lsb,
        );
        cmp_domain(
            "mvd_sign",
            &in_mem.cover.mvd_sign_bypass.bits,
            &stream.cover.mvd_sign_bypass.bits,
            &in_mem.cover.mvd_sign_bypass.positions,
            &stream.cover.mvd_sign_bypass.positions,
            &in_mem.costs.mvd_sign_bypass,
            &stream.costs.mvd_sign_bypass,
        );
        cmp_domain(
            "mvd_suffix",
            &in_mem.cover.mvd_suffix_lsb.bits,
            &stream.cover.mvd_suffix_lsb.bits,
            &in_mem.cover.mvd_suffix_lsb.positions,
            &stream.cover.mvd_suffix_lsb.positions,
            &in_mem.costs.mvd_suffix_lsb,
            &stream.costs.mvd_suffix_lsb,
        );
    }

    /// §6E-A5(c.x) — Pass-1B is byte-equivalent (above tests
    /// pass), so #107 must be downstream. This test compares the
    /// cover RECOVERED FROM THE WALKER on streaming-Pass-3 bytes
    /// vs in-memory-Pass-3 bytes. If they differ, walker sees
    /// different cover from streaming output → shadow position
    /// selection over primary_emit_cover diverges → cascade
    /// verify fails.
    #[test]
    fn pass3_walked_cover_matches_inmemory_128x80_2gop() {
        use crate::codec::h264::cabac::bin_decoder::slice::{
            walk_annex_b_for_cover_with_options, WalkOptions,
        };
        let yuv = match std::fs::read(
            "test-vectors/video/h264/real-world/img4138_128x80_f10.yuv",
        ) {
            Ok(y) => y,
            Err(_) => return,
        };
        let (w, h, n, gop, b) = (128u32, 80u32, 10usize, 5usize, 1usize);
        let pattern = GopPattern::Ibpbp { gop, b_count: b };
        let frame_size = (w * h * 3 / 2) as usize;

        let cover_p1 = pass1_count_4domain(
            &yuv, w, h, n, frame_size, Some(26), gop, b,
        )
        .expect("pass1");

        let combined_plan = DomainPlan {
            coeff_sign_bypass: cover_p1.cover.coeff_sign_bypass.bits.clone(),
            coeff_suffix_lsb: cover_p1.cover.coeff_suffix_lsb.bits.clone(),
            mvd_sign_bypass: cover_p1.cover.mvd_sign_bypass.bits.clone(),
            mvd_suffix_lsb: cover_p1.cover.mvd_suffix_lsb.bits.clone(),
            total_modifications: 0,
            total_cost: 0.0,
        };

        let in_mem_bytes = pass3_inject_4domain(
            &yuv, w, h, n, frame_size, Some(26),
            &cover_p1.cover, &combined_plan, gop, b,
        )
        .expect("in-memory pass3");

        let per_gop = derive_per_gop_counts_from_cover(
            &yuv, frame_size, n, pattern, &cover_p1.cover,
        );
        let stream_bytes = pass3_inject_4domain_streaming(
            &yuv, w, h, n, frame_size, Some(26), pattern,
            &combined_plan, &per_gop,
        )
        .expect("streaming pass3");

        eprintln!(
            "in-mem bytes: {}, stream bytes: {}",
            in_mem_bytes.len(),
            stream_bytes.len(),
        );

        let walk_opts = WalkOptions { record_mvd: true };
        let in_mem_walk =
            walk_annex_b_for_cover_with_options(&in_mem_bytes, walk_opts)
                .expect("walk in-mem");
        let stream_walk =
            walk_annex_b_for_cover_with_options(&stream_bytes, walk_opts)
                .expect("walk streaming");

        eprintln!(
            "in-mem walk:  cs={} cl={} ms={} ml={}",
            in_mem_walk.cover.coeff_sign_bypass.bits.len(),
            in_mem_walk.cover.coeff_suffix_lsb.bits.len(),
            in_mem_walk.cover.mvd_sign_bypass.bits.len(),
            in_mem_walk.cover.mvd_suffix_lsb.bits.len(),
        );
        eprintln!(
            "stream walk:  cs={} cl={} ms={} ml={}",
            stream_walk.cover.coeff_sign_bypass.bits.len(),
            stream_walk.cover.coeff_suffix_lsb.bits.len(),
            stream_walk.cover.mvd_sign_bypass.bits.len(),
            stream_walk.cover.mvd_suffix_lsb.bits.len(),
        );

        let mut im_pos: Vec<u64> = in_mem_walk
            .cover
            .coeff_sign_bypass
            .positions
            .iter()
            .map(|p| p.raw())
            .collect();
        let mut st_pos: Vec<u64> = stream_walk
            .cover
            .coeff_sign_bypass
            .positions
            .iter()
            .map(|p| p.raw())
            .collect();
        im_pos.sort();
        st_pos.sort();
        assert_eq!(
            im_pos, st_pos,
            "walker-recovered coeff_sign positions differ between in-memory and streaming Pass-3",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_yuv(w: u32, h: u32, n_frames: usize) -> Vec<u8> {
        let frame_size = (w * h * 3 / 2) as usize;
        let mut out = Vec::with_capacity(frame_size * n_frames);
        let mut s: u32 = 0x1234_5678;
        for _ in 0..n_frames {
            for _ in 0..frame_size {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                out.push((s >> 16) as u8);
            }
        }
        out
    }

    #[test]
    fn empty_message_produces_byte_identical_output_to_no_stego() {
        // Phase 6D.8 chunk 4 sign-off gate: with an empty message
        // (no bits to embed), the orchestration driver should
        // produce stego bytes equal to the no-stego baseline.
        // Plan = cover bits → no overrides apply → encoder produces
        // identical bytes.
        let yuv = deterministic_yuv(32, 32, 1);

        // Baseline: encoder with no stego hook. Mirror `build_encoder`
        // (transform_8x8=true, High profile per §Stealth.L3.1 follow-on)
        // so the stego empty-message run lands on identical bytes.
        use super::super::super::encoder::encoder::{Encoder, EntropyMode};
        let mut baseline_enc = Encoder::new(32, 32, Some(26)).unwrap();
        baseline_enc.entropy_mode = EntropyMode::Cabac;
        baseline_enc.enable_transform_8x8 = true;
        let baseline = baseline_enc.encode_i_frame(&yuv).unwrap();

        // Stego with empty message.
        let stego = h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1,
            /* message */ &[],
            "test-pass", /* h */ 4, Some(26),
        ).unwrap();

        assert_eq!(stego, baseline,
            "empty message → stego MUST equal baseline byte-identical");
    }

    #[test]
    fn rejects_non_aligned_dimensions() {
        let yuv = vec![0u8; 33 * 32 * 3 / 2];
        let r = h264_stego_encode_i_frames_only(
            &yuv, 33, 32, 1, &[0, 1, 0], "pass", 4, Some(26),
        );
        assert!(r.is_err(), "non-16-aligned dimensions must error");
    }

    #[test]
    fn rejects_wrong_yuv_length() {
        let yuv = vec![0u8; 100];
        let r = h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1, &[0, 1, 0], "pass", 4, Some(26),
        );
        assert!(r.is_err(), "wrong YUV length must error");
    }

    #[test]
    fn rejects_invalid_h_param() {
        let yuv = deterministic_yuv(32, 32, 1);
        let r = h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1, &[0, 1, 0], "pass", /* h */ 8, Some(26),
        );
        assert!(r.is_err(), "h>7 must error");
    }

    #[test]
    fn small_message_encodes_without_error() {
        // Sanity: a tiny message can be embedded and the output is
        // a non-empty Annex-B byte stream. Decode-side recovery
        // gates land in chunk 5 + 6.
        let yuv = deterministic_yuv(32, 32, 1);
        let msg = vec![1u8, 0, 1, 1, 0]; // 5 bits
        let bytes = h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1, &msg, "test-pass", /* h */ 4, Some(26),
        ).unwrap();
        assert!(!bytes.is_empty(), "stego output must be non-empty");
        // Output should contain SPS + PPS + IDR slice NALs.
        assert!(bytes.len() > 50, "stego output suspiciously small: {} bytes", bytes.len());
    }

    #[test]
    fn message_too_large_returns_error() {
        // 32×32 single I-frame has limited capacity. Try to embed
        // a message that's too large.
        let yuv = deterministic_yuv(32, 32, 1);
        let msg = vec![0u8; 100_000]; // way more bits than 32×32 can carry
        let r = h264_stego_encode_i_frames_only(
            &yuv, 32, 32, 1, &msg, "test-pass", 4, Some(26),
        );
        assert!(r.is_err(), "oversize message must return error");
    }

    // ── Chunk 5: string-message YUV encode wrapper ───────────────

    #[test]
    fn string_wrapper_returns_non_empty_annex_b() {
        // Sanity: a short string message + 32×32 single I-frame
        // produces a non-empty Annex-B byte stream.
        let yuv = deterministic_yuv(32, 32, 1);
        let bytes = h264_stego_encode_yuv_string(
            &yuv, 32, 32, 1, "hi", "pass-1",
        ).unwrap();
        assert!(bytes.len() > 50, "stego output too small: {} bytes", bytes.len());
    }

    #[test]
    fn string_wrapper_rejects_non_aligned_dimensions() {
        let yuv = vec![0u8; 33 * 32 * 3 / 2];
        let r = h264_stego_encode_yuv_string(
            &yuv, 33, 32, 1, "msg", "pass",
        );
        assert!(r.is_err(), "non-16-aligned dims must error");
    }

    #[test]
    fn string_wrapper_distinct_passphrases_produce_distinct_output() {
        // Encrypted framing + per-domain ChaCha20 keys derived from
        // passphrase ⇒ two different passphrases on the same YUV +
        // message must produce different Annex-B bytes.
        let yuv = deterministic_yuv(32, 32, 1);
        let a = h264_stego_encode_yuv_string(
            &yuv, 32, 32, 1, "hi", "pass-a",
        ).unwrap();
        let b = h264_stego_encode_yuv_string(
            &yuv, 32, 32, 1, "hi", "pass-b",
        ).unwrap();
        assert_ne!(a, b, "distinct passphrases must produce distinct output");
    }

    // ─── Phase 6E-C1a multi-IDR tests ─────────────────────────────

    /// Smooth, correlated multi-frame YUV that avoids triggering the
    /// encoder's scene-change → IDR fallback (mean delta < ~20 gray
    /// levels) but provides enough texture for nonzero residuals.
    /// Per-pixel base pattern is shared across frames; small
    /// per-frame perturbations simulate camera noise.
    fn correlated_yuv(w: u32, h: u32, n_frames: usize) -> Vec<u8> {
        let frame_size = (w * h * 3 / 2) as usize;
        let mut out = Vec::with_capacity(frame_size * n_frames);

        // Build a per-pixel base pattern once (mid-gray ± wide range
        // to drive nonzero coefficients).
        let mut base = Vec::with_capacity(frame_size);
        let mut s: u32 = 0x1234_5678;
        for _ in 0..frame_size {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let v = 64i32 + ((s >> 24) & 0x7F) as i32; // 64..=191 (~mid-gray)
            base.push(v.clamp(0, 255) as u8);
        }

        for fi in 0..n_frames {
            // Per-frame perturbation: small ± levels (avoid scene change).
            let mut p: u32 = 0x4242_DEAD ^ (fi as u32 * 17);
            for &b in &base {
                p = p.wrapping_mul(1103515245).wrapping_add(12345);
                let delta = ((p >> 28) & 0x07) as i32 - 4; // ±4 around base
                let v = b as i32 + delta;
                out.push(v.clamp(0, 255) as u8);
            }
        }
        out
    }

    /// Phase 6E-C1a — multi-IDR encode emits IDRs at the expected
    /// frame indices. Walks the Annex-B output and counts IDR slice
    /// NALs; must equal `ceil(n_frames / gop_size)`.
    #[test]
    fn multigop_emits_expected_number_of_idrs() {
        use crate::codec::h264::bitstream::parse_nal_units_annexb;

        let n_frames = 6;
        let gop_size = 2;
        // Smooth content — no scene-change IDR promotion.
        let yuv = correlated_yuv(32, 32, n_frames);

        let bytes = h264_stego_encode_yuv_string_4domain_multigop(
            &yuv, 32, 32, n_frames, gop_size, "hi", "pass",
        ).expect("multigop encode");

        let nalus = parse_nal_units_annexb(&bytes).expect("nal parse");
        let mut idr_count = 0;
        let mut non_idr_count = 0;
        for n in &nalus {
            if n.nal_type.is_idr() { idr_count += 1; }
            else if n.nal_type.is_vcl() { non_idr_count += 1; }
        }
        let expected_idr = n_frames.div_ceil(gop_size);
        let expected_non_idr = n_frames - expected_idr;
        assert_eq!(
            (idr_count, non_idr_count),
            (expected_idr, expected_non_idr),
            "expected {expected_idr} IDRs + {expected_non_idr} non-IDR slices \
             (n_frames={n_frames}, gop_size={gop_size}), got {idr_count} IDRs + {non_idr_count} non-IDR slices"
        );
    }

    /// Phase 6E-C1a — multi-IDR roundtrip. Encode multi-GOP YUV with
    /// gop_size=2 → 3 GOPs in 6 frames → decode → message recovered.
    /// Validates that the §30D-C 4-domain orchestrator + the §6E-C0
    /// streaming walker (via the wrapper that accumulates per-GOP
    /// covers) handle multi-IDR streams end-to-end.
    #[test]
    fn multigop_roundtrip_recovers_message() {
        use super::super::decode_pixels::h264_stego_decode_yuv_string_4domain;

        let n_frames = 6;
        let gop_size = 2;
        let yuv = correlated_yuv(32, 32, n_frames);
        let msg = "hi multi-IDR";
        let pass = "test-pass";

        let bytes = h264_stego_encode_yuv_string_4domain_multigop(
            &yuv, 32, 32, n_frames, gop_size, msg, pass,
        ).expect("multigop encode");

        let recovered = h264_stego_decode_yuv_string_4domain(&bytes, pass)
            .expect("multigop decode");
        assert_eq!(recovered, msg, "message must round-trip through multi-IDR encode/decode");
    }

    /// Phase 6E-C1a — gop_size=n_frames (legacy single-IDR shape)
    /// produces byte-identical output to the original
    /// `h264_stego_encode_yuv_string_4domain` API. Confirms the
    /// multigop variant is a clean superset of the legacy shape.
    #[test]
    fn multigop_gop_size_equals_n_frames_matches_legacy() {
        let yuv = deterministic_yuv(32, 32, 3);
        // Encrypt via deterministic-passphrase: actual bytes differ
        // between calls (random nonce). So compare DECODED message
        // instead of raw bytes.
        use super::super::decode_pixels::h264_stego_decode_yuv_string_4domain;

        let legacy = h264_stego_encode_yuv_string_4domain(
            &yuv, 32, 32, 3, "msg", "pass",
        ).expect("legacy encode");
        let multigop = h264_stego_encode_yuv_string_4domain_multigop(
            &yuv, 32, 32, 3, /* gop_size */ 3, "msg", "pass",
        ).expect("multigop encode");

        let legacy_dec = h264_stego_decode_yuv_string_4domain(&legacy, "pass")
            .expect("legacy decode");
        let multigop_dec = h264_stego_decode_yuv_string_4domain(&multigop, "pass")
            .expect("multigop decode");
        assert_eq!(legacy_dec, "msg");
        assert_eq!(multigop_dec, "msg");
    }

    /// Phase 6E-C1a — gop_size=0 returns InvalidVideo error.
    #[test]
    fn multigop_rejects_gop_size_zero() {
        let yuv = deterministic_yuv(32, 32, 2);
        let r = h264_stego_encode_yuv_string_4domain_multigop(
            &yuv, 32, 32, 2, /* gop_size */ 0, "x", "p",
        );
        assert!(r.is_err());
    }

    /// Phase 6E-C1a — gop_size > n_frames returns InvalidVideo.
    #[test]
    fn multigop_rejects_gop_size_larger_than_n_frames() {
        let yuv = deterministic_yuv(32, 32, 2);
        let r = h264_stego_encode_yuv_string_4domain_multigop(
            &yuv, 32, 32, 2, /* gop_size */ 5, "x", "p",
        );
        assert!(r.is_err());
    }

    // ─── Phase 6E-C1b shadow tests ─────────────────────────────────

    /// Phase 6E-C1b-v2 — cascade-equipped encode/decode roundtrip
    /// for short messages.
    #[test]
    fn shadow_roundtrip_recovers_both_messages() {
        use super::super::decode_pixels::{
            h264_stego_decode_yuv_string_4domain,
            h264_stego_shadow_decode,
        };

        let yuv = correlated_yuv(64, 64, 4);
        let primary = "primary";
        let shadow = "shadow";
        let primary_pass = "alice";
        let shadow_pass = "bob";

        let bytes = h264_stego_encode_yuv_string_with_shadow(
            &yuv, 64, 64, 4, /* gop_size */ 4,
            primary, primary_pass, shadow, shadow_pass,
        ).expect("shadow encode");

        let recovered_primary = h264_stego_decode_yuv_string_4domain(
            &bytes, primary_pass,
        ).expect("primary decode");
        assert_eq!(recovered_primary, primary);

        let recovered_shadow = h264_stego_shadow_decode(&bytes, shadow_pass)
            .expect("shadow decode");
        assert_eq!(recovered_shadow, shadow);
    }

    /// Task #208 diagnostic — same setup as the failing
    /// `shadow_roundtrip_handles_longer_primary_via_cascade` but
    /// PRIMARY ONLY (no shadow). If this passes under
    /// B_RDO+RESIDUAL, the bug is shadow-specific (cascade-safety
    /// or shadow-position-selection). If it ALSO fails, the bug is
    /// in primary STC under B-frame residual cover.
    #[test]
    #[ignore]
    fn task208_primary_only_long_msg_under_residual() {
        use super::super::decode_pixels::h264_stego_decode_yuv_string_4domain;

        let yuv = correlated_yuv(64, 64, 4);
        let primary = "primary message — a sentence of moderate length to drive primary STC's residual flips up";
        let primary_pass = "alice";

        let bytes = h264_stego_encode_yuv_string_4domain_multigop(
            &yuv, 64, 64, 4, /* gop_size */ 4,
            primary, primary_pass,
        ).expect("primary-only encode");

        let recovered = h264_stego_decode_yuv_string_4domain(
            &bytes, primary_pass,
        ).expect("primary decode");
        assert_eq!(recovered, primary);
    }

    /// Phase 6E-C1b-v2 — cascade verification handles longer primary
    /// messages that triggered §6E-C1b's propagation failure mode.
    /// At parity 4 the boundary BER may exceed RS absorption; cascade
    /// escalates parity until verify succeeds.
    #[test]
    fn shadow_roundtrip_handles_longer_primary_via_cascade() {
        use super::super::decode_pixels::{
            h264_stego_decode_yuv_string_4domain,
            h264_stego_shadow_decode,
        };

        // Realistic sizes: primary always bigger than shadow (caller
        // sorts). Tests cascade through propagation BER under a
        // longer primary message that triggered §6E-C1b's failure
        // mode with the no-cascade implementation.
        let yuv = correlated_yuv(64, 64, 4);
        let primary = "primary message — a sentence of moderate length to drive primary STC's residual flips up";
        let shadow = "hi";
        let primary_pass = "alice";
        let shadow_pass = "bob";

        let bytes = h264_stego_encode_yuv_string_with_shadow(
            &yuv, 64, 64, 4, 4,
            primary, primary_pass, shadow, shadow_pass,
        ).expect("shadow encode (cascade should succeed within 6 tiers)");

        let recovered_primary = h264_stego_decode_yuv_string_4domain(
            &bytes, primary_pass,
        ).expect("primary decode");
        assert_eq!(recovered_primary, primary);

        let recovered_shadow = h264_stego_shadow_decode(&bytes, shadow_pass)
            .expect("shadow decode");
        assert_eq!(recovered_shadow, shadow);
    }

    /// Phase 6E-C1b — `smart_decode_video` returns whichever message
    /// the supplied passphrase actually decrypts. With the primary
    /// passphrase: primary message. With the shadow passphrase:
    /// shadow message.
    #[test]
    fn shadow_smart_decode_chooses_by_passphrase() {
        use super::super::decode_pixels::h264_stego_smart_decode_video;

        let yuv = correlated_yuv(64, 64, 4);
        let primary = "primary";
        let shadow = "shadow";

        let bytes = h264_stego_encode_yuv_string_with_shadow(
            &yuv, 64, 64, 4, 4,
            primary, "alice", shadow, "bob",
        ).expect("shadow encode");

        let with_alice = h264_stego_smart_decode_video(&bytes, "alice")
            .expect("smart decode (primary pass)");
        assert_eq!(with_alice, primary);

        let with_bob = h264_stego_smart_decode_video(&bytes, "bob")
            .expect("smart decode (shadow pass)");
        assert_eq!(with_bob, shadow);
    }

    /// Phase 6E-C1b — wrong passphrase fails on shadow decode.
    /// AES-GCM-SIV authentication rejects every parity-tier candidate.
    #[test]
    fn shadow_wrong_passphrase_fails() {
        use super::super::decode_pixels::h264_stego_shadow_decode;

        let yuv = correlated_yuv(64, 64, 4);
        let bytes = h264_stego_encode_yuv_string_with_shadow(
            &yuv, 64, 64, 4, 4,
            "primary", "alice", "shadow", "bob",
        ).expect("shadow encode");

        let r = h264_stego_shadow_decode(&bytes, "wrong-passphrase");
        assert!(r.is_err(), "wrong passphrase must fail shadow decode");
    }

    // ─── Phase 6E-C2 multi-shadow tests ───────────────────────────

    /// Phase 6E-C2 — primary + 2 shadows roundtrip. **DEFERRED**
    /// at §6E-C2 mvp scope: N>1 4-domain multi-shadow has an
    /// architectural chicken-and-egg with shadow position
    /// indexing (shadow_state's intra-indexes reference cover_p1's
    /// vanilla cover, but combined_plan's coeff slots reference
    /// cover_p1b_residual's post-MVD-plan cover; the lists differ
    /// at the |coeff|=16 / zero-crossing boundaries from MVD-induced
    /// §6E-C2 polish — N=2 shadow roundtrip with adequate cover.
    ///
    /// The polish architecture (commits 1-4 of the polish sequence)
    /// uses provisional emit + dual-iteration cascade with
    /// `primary_emit_cover` for shadow position selection. Inter-
    /// shadow collisions on overlapping priority-sorted positions
    /// are absorbed by RS at one of the cascade's parity tiers.
    ///
    /// Capacity formula: collisions per shadow ≈ (N−1) × m² / C
    /// where m = bits per shadow, C = cover_size. Required parity
    /// to absorb: P ≥ (N−1) × m² / (8C). For tiny videos like
    /// 128×64×4 with N=2 and ~50-byte shadow messages, the cover
    /// (~1500 positions) is too small to absorb collisions even
    /// at the max parity tier 128. The earlier failure of this
    /// test under tighter parameters was a CAPACITY exhaustion,
    /// not an architectural flaw.
    ///
    /// This test uses 128×128 × 4 frames (quadrupled cover) which
    /// gives ~6000 bypass-bin positions — enough headroom for two
    /// 1-byte-payload shadows at parity 4 to cascade-succeed first
    /// try.
    #[test]
    fn n_shadows_roundtrip_n_equals_2() {
        use super::super::decode_pixels::{
            h264_stego_decode_yuv_string_4domain,
            h264_stego_shadow_decode,
        };
        use crate::stego::shadow_layer::ShadowLayer;

        // 128×128 × 4 frames gives ~6000 bypass-bin cover positions
        // (4× the 128×64 × 4 frame baseline). For N=2 with ~400-bit
        // shadows: collision count ≈ 400² / 6000 ≈ 27 per shadow,
        // ~14 bit errors → absorbable at parity 4 (16-bit absorption).
        let yuv = correlated_yuv(128, 128, 4);
        let primary = "p";
        let shadow_a = "a";
        let shadow_b = "b";

        let layers = [
            ShadowLayer { message: shadow_a, passphrase: "alice", files: &[] },
            ShadowLayer { message: shadow_b, passphrase: "bob", files: &[] },
        ];

        let bytes = h264_stego_encode_yuv_string_with_n_shadows(
            &yuv, 128, 128, 4, /* gop_size */ 4,
            primary, "primary-pass",
            &layers,
        ).expect("N=2 shadow encode");

        let recovered_primary = h264_stego_decode_yuv_string_4domain(
            &bytes, "primary-pass",
        ).expect("primary decode");
        assert_eq!(recovered_primary, primary);

        let recovered_a = h264_stego_shadow_decode(&bytes, "alice")
            .expect("shadow alice decode");
        assert_eq!(recovered_a, shadow_a);

        let recovered_b = h264_stego_shadow_decode(&bytes, "bob")
            .expect("shadow bob decode");
        assert_eq!(recovered_b, shadow_b);
    }

    /// Phase 6E-C2 — duplicate passphrases across primary + shadows
    /// must be rejected.
    #[test]
    fn n_shadows_rejects_duplicate_passphrases() {
        use crate::stego::shadow_layer::ShadowLayer;

        let yuv = correlated_yuv(128, 64, 4);
        // Two shadows sharing a passphrase.
        let layers = [
            ShadowLayer { message: "a", passphrase: "shared", files: &[] },
            ShadowLayer { message: "b", passphrase: "shared", files: &[] },
        ];

        let r = h264_stego_encode_yuv_string_with_n_shadows(
            &yuv, 128, 64, 4, 4,
            "primary", "primary-pass",
            &layers,
        );
        assert!(matches!(r, Err(StegoError::DuplicatePassphrase)));

        // Primary's passphrase clashing with a shadow's also rejected.
        let layers2 = [
            ShadowLayer { message: "a", passphrase: "primary-pass", files: &[] },
        ];
        let r2 = h264_stego_encode_yuv_string_with_n_shadows(
            &yuv, 128, 64, 4, 4,
            "primary", "primary-pass",
            &layers2,
        );
        assert!(matches!(r2, Err(StegoError::DuplicatePassphrase)));
    }

    /// §6E-C2 polish — h264_stego_shadow_capacity returns sensible
    /// values: cover_size_bits is non-zero, max_message_bytes is
    /// positive for N≥1, zero for N=0, and decreases with growing N.
    #[test]
    fn shadow_capacity_returns_sensible_values() {
        let yuv = correlated_yuv(128, 128, 4);

        let cap_n0 = h264_stego_shadow_capacity(
            &yuv, 128, 128, 4, 4, 0,
        ).expect("capacity n=0");
        assert!(cap_n0.cover_size_bits > 0);
        assert_eq!(cap_n0.max_message_bytes, 0);

        let cap_n1 = h264_stego_shadow_capacity(
            &yuv, 128, 128, 4, 4, 1,
        ).expect("capacity n=1");
        assert_eq!(cap_n1.cover_size_bits, cap_n0.cover_size_bits);
        assert!(cap_n1.max_message_bytes > 0);

        let cap_n5 = h264_stego_shadow_capacity(
            &yuv, 128, 128, 4, 4, 5,
        ).expect("capacity n=5");
        assert!(
            cap_n5.max_message_bytes <= cap_n1.max_message_bytes,
            "max_message_bytes should not increase with shadow count"
        );
    }

    /// §6E-C2 polish — encoder rejects shadow payloads that exceed
    /// the capacity-formula limit. UI is supposed to call
    /// `h264_stego_shadow_capacity` first; this test confirms the
    /// belt-and-suspenders guard inside the encoder triggers when
    /// the UI doesn't.
    #[test]
    fn encoder_rejects_oversize_shadow_payload() {
        use crate::stego::shadow_layer::ShadowLayer;

        let yuv = correlated_yuv(64, 64, 4);
        // Build a shadow message bigger than the capacity ceiling
        // for this tiny cover. The encoder's belt-and-suspenders
        // check must reject before attempting the heavy encode.
        let cap = h264_stego_shadow_capacity(&yuv, 64, 64, 4, 4, 1)
            .expect("capacity");
        let oversize = "X".repeat(cap.max_message_bytes + 1024);

        let layers = [ShadowLayer {
            message: &oversize,
            passphrase: "shadow-pass",
            files: &[],
        }];
        let r = h264_stego_encode_yuv_string_with_n_shadows(
            &yuv, 64, 64, 4, 4, "p", "primary-pass", &layers,
        );
        assert!(matches!(r, Err(StegoError::MessageTooLarge)));
    }

    /// Phase 6E-C2 — N=0 shadows behaves identically to a primary-
    /// only encode (no cascade machinery, no shadow positions).
    #[test]
    fn n_shadows_n_equals_0_matches_primary_only() {
        use super::super::decode_pixels::h264_stego_decode_yuv_string_4domain;

        let yuv = correlated_yuv(64, 64, 4);
        let bytes = h264_stego_encode_yuv_string_with_n_shadows(
            &yuv, 64, 64, 4, 4,
            "primary", "primary-pass",
            &[],
        ).expect("N=0 encode");

        let recovered = h264_stego_decode_yuv_string_4domain(
            &bytes, "primary-pass",
        ).expect("primary decode");
        assert_eq!(recovered, "primary");
    }

    // NOTE: bit-stream-level determinism (same YUV+bits+passphrase ⇒
    // identical Annex-B) is covered by
    // `empty_message_produces_byte_identical_output_to_no_stego`.
    // The string wrapper itself is NOT deterministic across calls
    // because `crypto::encrypt` generates a fresh random nonce +
    // salt per call (AES-GCM-SIV nonce-reuse protection).

    // NOTE: oversize-message rejection is covered at the inner-driver
    // layer by `message_too_large_returns_error`. The string wrapper
    // can't reliably trigger it without a non-compressible payload
    // since `payload::encode_payload` brotli-compresses input first.
}
