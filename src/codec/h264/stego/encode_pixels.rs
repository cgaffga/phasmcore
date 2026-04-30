// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
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
    // Don't enable_transform_8x8 for now — not needed for I-frame-
    // only stego and avoids the I_8x8 mode-decision branch.
    enc.enable_transform_8x8 = false;
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
    let cover_size_bits = cover_p1.cover.capacity().total();

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
    use crate::stego::{crypto, frame, payload};
    use crate::stego::shadow_layer::SHADOW_PARITY_TIERS;

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
    let primary_bytes = payload::encode_payload(primary_message, &[])?;
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
        /* b_count */ 1,
    )?;
    let cap_p1 = cover_p1.cover.capacity();
    let mvd_capacity = cap_p1.mvd_sign_bypass + cap_p1.mvd_suffix_lsb;
    let m_mvd = m_total.min(mvd_capacity);
    let m_residual = m_total - m_mvd;

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
    let (primary_emit_cover, cover_p1b_residual_prov) = if shadows.is_empty() {
        (super::DomainCover::default(), super::orchestrate::GopCover::default())
    } else {
        let plan_a_prov = if m_mvd > 0 {
            let cap_mvd_only = GopCapacity {
                coeff_sign_bypass: 0,
                coeff_suffix_lsb: 0,
                mvd_sign_bypass: cap_p1.mvd_sign_bypass,
                mvd_suffix_lsb: cap_p1.mvd_suffix_lsb,
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
        let cover_p1b_prov = pass1b_inject_mvd_log_residual(
            yuv, width, height, n_frames, frame_size, quality,
            &cover_p1.cover, &plan_a_prov, gop_size, /* b_count */ 1,
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
        let mut combined_cover_prov = super::DomainCover::default();
        combined_cover_prov.coeff_sign_bypass = cover_p1b_prov.cover.coeff_sign_bypass.clone();
        combined_cover_prov.coeff_suffix_lsb = cover_p1b_prov.cover.coeff_suffix_lsb.clone();
        combined_cover_prov.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.clone();
        combined_cover_prov.mvd_suffix_lsb = cover_p1.cover.mvd_suffix_lsb.clone();
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
        let bytes_prov = pass3_inject_4domain(
            yuv, width, height, n_frames, frame_size, quality,
            &combined_cover_prov, &combined_plan_prov, gop_size, /* b_count */ 1,
        )?;
        let walk_opts = WalkOptions { record_mvd: true };
        let walk_prov = walk_annex_b_for_cover_with_options(&bytes_prov, walk_opts)
            .map_err(|e| StegoError::InvalidVideo(format!("provisional walk: {e}")))?;
        (walk_prov.cover, cover_p1b_prov)
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
    for &parity_len in &SHADOW_PARITY_TIERS {
        // ── §6E-C2 polish — select shadow positions over
        //     `primary_emit_cover`. All 4 domains in a single
        //     priority sort (replaces the mvp's two-phase MVD-only +
        //     residual-only sorts). ──
        let shadow_states_emit: Vec<super::shadow::ShadowState> = shadows
            .iter()
            .map(|s| super::shadow::prepare_shadow_over_emit_cover(
                &primary_emit_cover, s.passphrase, s.message, s.files, parity_len,
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
                mvd_suffix_lsb: cap_p1.mvd_suffix_lsb,
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
        let cover_p1b_residual = pass1b_inject_mvd_log_residual(
            yuv, width, height, n_frames, frame_size, quality,
            &cover_p1.cover, &plan_a, gop_size, /* b_count */ 1,
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

        // ── PlanInjector lookup cover: MVD from Pass 1, residual
        //     from Pass 1B. ──
        let mut combined_cover = super::DomainCover::default();
        combined_cover.coeff_sign_bypass = cover_p1b_residual.cover.coeff_sign_bypass.clone();
        combined_cover.coeff_suffix_lsb = cover_p1b_residual.cover.coeff_suffix_lsb.clone();
        combined_cover.mvd_sign_bypass = cover_p1.cover.mvd_sign_bypass.clone();
        combined_cover.mvd_suffix_lsb = cover_p1.cover.mvd_suffix_lsb.clone();

        // ── Pass 3: emit with combined plan. ──
        let bytes = pass3_inject_4domain(
            yuv, width, height, n_frames, frame_size, quality,
            &combined_cover, &combined_plan, gop_size, /* b_count */ 1,
        )?;

        // ── Verify: walk emitted bytes → 4-domain cover. Each
        //     shadow extracts independently with its own passphrase;
        //     ALL shadows must succeed for cascade to terminate. ──
        let opts = WalkOptions { record_mvd: true };
        let walk = walk_annex_b_for_cover_with_options(&bytes, opts)
            .map_err(|e| StegoError::InvalidVideo(format!("verify walk: {e}")))?;
        let mut all_ok = true;
        for s in shadows {
            match super::shadow::shadow_extract_all4(&walk.cover, s.passphrase) {
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
fn iter_frames_in_encode_order<'a>(
    yuv: &'a [u8],
    frame_size: usize,
    n_frames: usize,
    pattern: super::gop_pattern::GopPattern,
) -> impl Iterator<Item = (super::gop_pattern::EncodeOrderFrame, &'a [u8])> {
    super::gop_pattern::iter_encode_order(n_frames, pattern).map(move |meta| {
        let d = meta.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        (meta, frame)
    })
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

        // Baseline: encoder with no stego hook.
        use super::super::super::encoder::encoder::{Encoder, EntropyMode};
        let mut baseline_enc = Encoder::new(32, 32, Some(26)).unwrap();
        baseline_enc.entropy_mode = EntropyMode::Cabac;
        baseline_enc.enable_transform_8x8 = false;
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
