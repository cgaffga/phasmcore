// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D.0.7 — streaming H.264 stego sessions.
//!
//! Session-based encode + decode APIs that emit bytes per frame / per NAL
//! group instead of taking a full YUV / Annex-B buffer at once. Required
//! by mobile + CLI to support arbitrary-length video without bounding
//! clip length by available RAM.
//!
//! Design memo: `docs/design/video/h264/d07-streaming-sessions.md`.
//!
//! ## Wiring status
//!
//! - **D.0.7.1**: API surface + scaffold tests (shipped).
//! - **D.0.7.2**: OH264 encode path (shipped here). Per-GOP STC plan
//!   using the `chunk_frame` wire format; one chunk per GOP. Memory
//!   bound: O(gop_size × frame_size).
//! - **D.0.7.3**: pure-Rust encode path (pending). Still stubbed —
//!   returns `InvalidVideo("not yet implemented")`.
//! - **D.0.7.11**: streaming decode session (pending).

use crate::stego::error::StegoError;

// #474.1 — progress event types. Streaming sessions accept an optional
// callback on creation; emission sites wired in #474.2.
use super::progress::{
    DecodePhase, DecodeProgressCallback, EncodePhase, EncodeProgressCallback,
};

// OH264 path imports — gated by the openh264-backend feature.
#[cfg(feature = "openh264-backend")]
use super::openh264_stego::{
    bytes_to_bits_msb_first_pub, encode_yuv_with_pre_framed_bits_4domain, EncodeOpts,
};
// chunk_frame helpers — needed by BOTH the OH264 encode path and (post
// #472.2) the pure-Rust encode path, so unconditional.
use super::stego::chunk_frame::{build_chunk_frame, split_message_into_chunks};

// Decode-side imports (available within the h264-encoder gate that
// already wraps this module; decode doesn't need openh264-backend).
use super::cabac::bin_decoder::slice::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use super::stego::{combine_cover_4domain, CostWeights};
use super::stego::orchestrate::DomainCosts;
use super::stego::chunk_frame::{assemble_chunks, parse_chunk_frame, CHUNK_HEADER_LEN};
use super::stego::hook::EmbedDomain;
use super::stego::keys::CabacStegoMasterKeys;
use crate::stego::stc::extract::{stc_extract, stc_extract_prefix};
use crate::stego::stc::hhat::generate_hhat;
use crate::stego::{crypto, frame, payload};

/// STC constraint length — mirrors `openh264_stego::STC_H`. Both
/// encode and decode sides MUST agree on this value.
const STC_H: usize = 4;

/// Defensive cap on `total_chunks` decoded from a chunk_frame header.
/// Real-world streaming sessions emit at most a few thousand GOPs
/// (1-hour 4K × 30 fps × GOP=30 ≈ 3600). Reject obviously-bogus values
/// from STC false-positive matches early.
const MAX_REASONABLE_CHUNKS: u16 = 10_000;

/// Which H.264 encoder backend a streaming encode session uses.
///
/// Decode is auto-detect via `smart_decode` semantics (the decode
/// session tries OH264 backend first, falls back to pure-Rust); no
/// parallel `DecodeEngineChoice` is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodeEngineChoice {
    /// Cisco OpenH264 fork (production default, v1.0). Statically linked
    /// via `openh264-sys`; only available when the `openh264-backend`
    /// Cargo feature is on.
    Oh264 = 0,
    /// Phasm's pure-Rust encoder. Experimental — gated behind the
    /// HUD picker, opt-in via Settings → "Enable experimental features".
    PureRust = 1,
}

impl EncodeEngineChoice {
    /// Parse from the u8 the FFI bridge receives. `0` = oh264 (default),
    /// any other value falls back to oh264 to keep production stable
    /// on unknown input from older client versions.
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::PureRust,
            _ => Self::Oh264,
        }
    }
}

/// Colour-space VUI fields the app reads from the source MP4 and passes
/// through unchanged into the stego SPS. See design memo § "Colour-space
/// pass-through".
#[derive(Debug, Clone, Copy)]
pub struct ColorParams {
    /// `color_primaries` (ITU-T H.264 § E.2.1 Table E-3).
    pub color_primaries: u8,
    /// `transfer_characteristics` (Table E-4).
    pub transfer_characteristics: u8,
    /// `matrix_coefficients` (Table E-5).
    pub matrix_coefficients: u8,
    /// `video_full_range_flag`: 0 = limited (Y ∈ [16,235]), 1 = full.
    pub video_full_range_flag: u8,
}

impl Default for ColorParams {
    /// Default: BT.709 limited. Matches the most common phone-recorded
    /// source. Callers that read the MP4 `colr` box should override.
    fn default() -> Self {
        Self {
            color_primaries: 1,
            transfer_characteristics: 1,
            matrix_coefficients: 1,
            video_full_range_flag: 0,
        }
    }
}

/// Encoder session creation parameters.
///
/// `Clone` is implemented but `Debug` is not derived: the optional
/// `progress_callback` holds an `Arc<dyn Fn>` which can't be debug-
/// formatted. The struct's `Debug` impl below skips the callback.
#[derive(Clone)]
pub struct EncodeSessionParams {
    /// Encoded width. Must be 16-aligned (callers pad on the app side).
    pub width: u32,
    /// Encoded height. Same 16-aligned constraint.
    pub height: u32,
    /// Frame rate numerator (e.g. 30 for 30 fps, 30000 for 29.97).
    pub fps_num: u32,
    /// Frame rate denominator (1 for integer fps, 1001 for NTSC fractions).
    pub fps_den: u32,
    /// Initial QP. Both encoders treat this as the I-slice anchor.
    pub qp: i32,
    /// GOP size in frames. STC plan is computed per-GOP; smaller GOP
    /// = lower peak RAM + more boundary overhead, larger GOP = inverse.
    pub gop_size: u32,
    /// REQUIRED for streaming sessions. Used to compute `total_chunks`
    /// at session_create so each emitted GOP carries the correct
    /// `total_chunks` header value. Mobile reads from `AVAsset.duration
    /// * fps`; CLI reads from `ffprobe -show_streams`. A zero value
    /// errors at session_create.
    pub total_frames_hint: u32,
    /// Colour-space VUI pass-through.
    pub color: ColorParams,
    /// Which backend to use.
    pub engine: EncodeEngineChoice,
    /// #493 — per-domain STC cost weights for the 4-domain combined
    /// cover plan. Default values (1.0, 3.0, 10.0, 10.0) are validated
    /// by Phase 0.5 (#493.0b) on real-corpus MvdSign cascade
    /// measurement. See `docs/design/video/h264/d07-streaming-4domain.md`.
    pub cost_weights: super::stego::CostWeights,
    /// #474.1 — optional progress callback. When `Some`, the session
    /// emits `EncodePhase` events at phase transitions + throttled
    /// per-unit ticks (≤30 Hz). Emission sites are wired in #474.2;
    /// in #474.1 this field is stored on the session but no events
    /// fire yet. Default is `None` for backwards compatibility — all
    /// existing call sites that build params struct-literally just
    /// add `progress_callback: None`.
    pub progress_callback: Option<EncodeProgressCallback>,
}

impl std::fmt::Debug for EncodeSessionParams {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EncodeSessionParams")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("fps_num", &self.fps_num)
            .field("fps_den", &self.fps_den)
            .field("qp", &self.qp)
            .field("gop_size", &self.gop_size)
            .field("total_frames_hint", &self.total_frames_hint)
            .field("color", &self.color)
            .field("engine", &self.engine)
            .field("cost_weights", &self.cost_weights)
            .field("progress_callback", &self.progress_callback.as_ref().map(|_| "<callback>"))
            .finish()
    }
}

/// One raw I420 planar frame pushed into a streaming encode session.
///
/// Strides may exceed `width` (source decoder may include padding); the
/// session re-packs into the encoder's tight-pitch layout.
#[derive(Debug)]
pub struct YuvFrameRef<'a> {
    pub y: &'a [u8],
    pub y_stride: usize,
    pub u: &'a [u8],
    pub u_stride: usize,
    pub v: &'a [u8],
    pub v_stride: usize,
}

/// Opaque streaming-encode session state.
///
/// FFI bridges pass `Box<StreamingEncodeSession>` as the C-side handle.
pub struct StreamingEncodeSession {
    params: EncodeSessionParams,
    inner: SessionImpl,
}

enum SessionImpl {
    /// OH264 backend session (D.0.7.2). Holds the GOP frame buffer +
    /// per-GOP chunk schedule + STC seed.
    #[cfg(feature = "openh264-backend")]
    Oh264(Oh264SessionState),
    /// Pure-Rust backend session (D.0.7.3.b shipped 2026-05-16 / #472).
    ///
    /// Memory is O(gop_size × frame_size): mirrors OH264 by buffering
    /// only the current in-flight GOP, draining at the GOP boundary
    /// via `h264_stego_encode_one_gop_with_chunk_bits`, then freeing
    /// the buffer. Output is chunk_frame'd per-GOP — decode-compatible
    /// with `StreamingDecodeSession` (the same protocol OH264
    /// streaming uses). Pre-v1.1 whole-clip output format is retired
    /// from streaming; legacy one-shot `phasm_h264_encode/_decode`
    /// callers keep the whole-video STC format independently.
    PureRust(PureRustSessionState),
    /// Requested engine isn't compiled in. push_frame / finish will
    /// error explicitly. Used when caller asks for OH264 but the
    /// crate was built without `openh264-backend`.
    EngineDisabled(EncodeEngineChoice),
}

struct PureRustSessionState {
    /// Pre-split message bytes — one chunk per GOP. Indexed by chunk_idx.
    chunks: Vec<Vec<u8>>,
    /// Total number of GOPs (== chunks.len()).
    total_chunks: u16,
    /// Index of the next GOP to emit (0-based, < total_chunks).
    chunk_idx: u16,
    /// In-flight GOP YUV buffer (tight I420 packing), drained at every
    /// gop_size boundary.
    gop_buffer: Vec<u8>,
    /// Frames currently buffered in `gop_buffer`.
    frames_buffered_in_gop: u32,
    /// STC h-hat seed for CoeffSign — same derivation as the OH264
    /// path (per_gop_seeds(CoeffSign, gop_idx=0)). All GOPs share this
    /// seed so the unified `StreamingDecodeSession` works without a
    /// per-engine fork.
    hhat_seed: [u8; 32],
    /// Original frame count target (bounds-check on push_frame).
    total_frames_target: u32,
    /// Cumulative frame count across all GOPs.
    frames_seen_total: u32,
}

#[cfg(feature = "openh264-backend")]
struct Oh264SessionState {
    /// Inner stego frame split across GOPs. `chunks[i]` is the payload
    /// for the i-th GOP's chunk_frame.
    chunks: Vec<Vec<u8>>,
    /// Number of GOPs (= number of chunks) expected total. Computed at
    /// session_create from `total_frames_hint / gop_size`.
    total_chunks: u16,
    /// Index of the next GOP to emit (0-based, < total_chunks).
    chunk_idx: u16,
    /// In-flight GOP YUV buffer (tight I420 packing).
    gop_buffer: Vec<u8>,
    /// Frames currently buffered in `gop_buffer`.
    frames_buffered: u32,
    /// STC h-hat seed for CoeffSign domain. Same as the one-shot path
    /// derives so cross-format compatibility stays an option.
    hhat_seed: [u8; 32],
    /// Frame byte size (computed once at create: w * h * 3 / 2).
    frame_bytes_size: usize,
}

impl StreamingEncodeSession {
    /// Create a new streaming encode session.
    ///
    /// # Errors
    /// * [`StegoError::InvalidVideo`] if dimensions are zero/non-16-aligned,
    ///   fps is degenerate, gop_size is zero, total_frames_hint is zero,
    ///   or the message can't fit into the computed chunk schedule.
    pub fn create(
        params: EncodeSessionParams,
        message_text: &str,
        passphrase: &str,
    ) -> Result<Self, StegoError> {
        if params.width == 0 || params.height == 0 {
            return Err(StegoError::InvalidVideo(format!(
                "dimensions must be > 0, got {}x{}",
                params.width, params.height
            )));
        }
        if params.width % 16 != 0 || params.height % 16 != 0 {
            return Err(StegoError::InvalidVideo(format!(
                "dimensions must be 16-aligned, got {}x{} (app must pad)",
                params.width, params.height
            )));
        }
        if params.fps_den == 0 || params.fps_num == 0 {
            return Err(StegoError::InvalidVideo(
                "fps_num and fps_den must be > 0".into(),
            ));
        }
        if params.gop_size == 0 {
            return Err(StegoError::InvalidVideo("gop_size must be > 0".into()));
        }
        if params.total_frames_hint == 0 {
            return Err(StegoError::InvalidVideo(
                "total_frames_hint must be > 0 for streaming sessions".into(),
            ));
        }

        let inner = match params.engine {
            #[cfg(feature = "openh264-backend")]
            EncodeEngineChoice::Oh264 => {
                SessionImpl::Oh264(build_oh264_state(&params, message_text, passphrase)?)
            }
            #[cfg(not(feature = "openh264-backend"))]
            EncodeEngineChoice::Oh264 => SessionImpl::EngineDisabled(EncodeEngineChoice::Oh264),
            EncodeEngineChoice::PureRust => {
                SessionImpl::PureRust(build_pure_rust_state(&params, message_text, passphrase)?)
            }
        };
        // #474.2 — emit Setup once after all initialisation succeeds.
        // Disabled engines never reach this path because build_*_state
        // already errored above; this event always corresponds to a
        // usable session.
        emit_encode(&params, EncodePhase::Setup);
        Ok(Self { params, inner })
    }

    /// Push one YUV frame. Emits Annex-B bytes for a closing GOP into
    /// `out` (appended) when the frame completes a GOP. Both OH264 and
    /// PureRust paths drain per-GOP; finish() handles the tail.
    pub fn push_frame(
        &mut self,
        frame: YuvFrameRef<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), StegoError> {
        match &mut self.inner {
            #[cfg(feature = "openh264-backend")]
            SessionImpl::Oh264(state) => oh264_push_frame(&self.params, state, frame, out),
            SessionImpl::PureRust(state) => pure_rust_push_frame(&self.params, state, frame, out),
            SessionImpl::EngineDisabled(engine) => Err(StegoError::InvalidVideo(format!(
                "streaming encode engine {engine:?} not compiled in (rebuild with feature)"
            ))),
        }
    }

    /// Finish the session. For OH264: drains the final partial GOP.
    /// For PureRust: runs the buffered v2 streaming encode and emits
    /// the full Annex-B output. Errors if no frames were pushed or
    /// (OH264 only) fewer than `total_chunks` chunks emitted.
    pub fn finish(self, out: &mut Vec<u8>) -> Result<(), StegoError> {
        match self.inner {
            #[cfg(feature = "openh264-backend")]
            SessionImpl::Oh264(mut state) => oh264_finish(&self.params, &mut state, out),
            SessionImpl::PureRust(state) => pure_rust_finish(&self.params, state, out),
            SessionImpl::EngineDisabled(engine) => Err(StegoError::InvalidVideo(format!(
                "streaming encode engine {engine:?} not compiled in (rebuild with feature)"
            ))),
        }
    }

    /// Snapshot the session parameters.
    pub fn params(&self) -> &EncodeSessionParams {
        &self.params
    }

    /// #474.2 — install or replace the progress callback after the
    /// session was created. Lets FFI callers attach a callback after
    /// `create()` (the create entry points carry enough arguments
    /// already; threading a function-pointer + context through is
    /// cleaner as a separate call).
    pub fn set_progress_callback(&mut self, callback: Option<EncodeProgressCallback>) {
        self.params.progress_callback = callback;
    }
}

/// #474.2 — invoke the progress callback if one is installed on these
/// params. Cheap no-op when `progress_callback` is `None`. Boundary
/// events bypass the 30 Hz throttle by definition (Setup, Mux, Done
/// fire once each); per-unit events are emitted at the natural
/// granularity of push_frame / drain_one_gop, well below 30 Hz on real
/// content, so no per-call throttle is needed.
fn emit_encode(params: &EncodeSessionParams, phase: EncodePhase) {
    if let Some(cb) = &params.progress_callback {
        cb(phase);
    }
}

// ─────────────────────── OH264 path ───────────────────────────────────

#[cfg(feature = "openh264-backend")]
fn build_oh264_state(
    params: &EncodeSessionParams,
    message_text: &str,
    passphrase: &str,
) -> Result<Oh264SessionState, StegoError> {
    // Build the inner stego frame ONCE up-front (same shape as the
    // one-shot path); split bytes across GOPs.
    let payload_bytes = payload::encode_payload(message_text, &[])?;
    let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct);

    let expected_n_gops = params.total_frames_hint.div_ceil(params.gop_size);
    if expected_n_gops == 0 || expected_n_gops > u16::MAX as u32 {
        return Err(StegoError::InvalidVideo(format!(
            "computed expected_n_gops {expected_n_gops} out of [1, {}]",
            u16::MAX
        )));
    }
    let total_chunks = expected_n_gops as u16;
    let chunks = split_message_into_chunks(&frame_bytes, total_chunks)?;

    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    let frame_bytes_size = (params.width as usize) * (params.height as usize) * 3 / 2;
    let gop_buffer = Vec::with_capacity(frame_bytes_size * params.gop_size as usize);

    Ok(Oh264SessionState {
        chunks,
        total_chunks,
        chunk_idx: 0,
        gop_buffer,
        frames_buffered: 0,
        hhat_seed,
        frame_bytes_size,
    })
}

#[cfg(feature = "openh264-backend")]
fn oh264_push_frame(
    params: &EncodeSessionParams,
    state: &mut Oh264SessionState,
    frame: YuvFrameRef<'_>,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    if state.chunk_idx >= state.total_chunks {
        return Err(StegoError::InvalidVideo(format!(
            "pushed more frames than total_frames_hint expected ({} GOPs already emitted)",
            state.total_chunks
        )));
    }

    // Pack the frame into the tight-pitch GOP buffer. Strides may
    // exceed width if the source decoder added row padding; we copy
    // row-by-row to strip it.
    pack_frame_into_buffer(params.width, params.height, &frame, &mut state.gop_buffer)?;
    state.frames_buffered += 1;

    // #474.2 — Pass1Capture event per buffered frame. The actual Pass-1
    // CABAC capture happens inside the C encoder at drain time, but
    // emitting here gives the smoothing engine 1 event per pushed
    // frame so the bar advances smoothly across the YUV-marshalling
    // phase. `frame_idx` is the cumulative count (chunks finalised so
    // far × gop_size + current buffer count); `total_frames` is the
    // session-level hint.
    let cumulative = state.chunk_idx as u32 * params.gop_size + state.frames_buffered;
    emit_encode(
        params,
        EncodePhase::Pass1Capture {
            frame_idx: cumulative,
            total_frames: params.total_frames_hint,
        },
    );

    if state.frames_buffered == params.gop_size {
        drain_one_gop(params, state, out)?;
    }
    Ok(())
}

#[cfg(feature = "openh264-backend")]
fn oh264_finish(
    params: &EncodeSessionParams,
    state: &mut Oh264SessionState,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    if state.frames_buffered > 0 {
        drain_one_gop(params, state, out)?;
    }
    if state.chunk_idx != state.total_chunks {
        return Err(StegoError::InvalidVideo(format!(
            "session finished with {}/{} chunks emitted — total_frames_hint was too high",
            state.chunk_idx, state.total_chunks,
        )));
    }
    // #474.2 — Mux + Done. OH264 path doesn't currently run a separate
    // HandBrake mux step (mp4 muxing is the bridge's responsibility),
    // but we still emit Mux for parity with the design doc so the
    // mobile-side smoothing engine sees a Mux→Done transition even
    // when the underlying mux is a no-op at this layer.
    emit_encode(params, EncodePhase::Mux);
    emit_encode(params, EncodePhase::Done);
    Ok(())
}

#[cfg(feature = "openh264-backend")]
fn drain_one_gop(
    params: &EncodeSessionParams,
    state: &mut Oh264SessionState,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    let frames_in_gop = state.frames_buffered;
    if frames_in_gop == 0 {
        return Ok(());
    }
    // #474.2 — StcPlan fires BEFORE the C encoder call. The OH264
    // backend performs Pass-1 capture + STC compute + Pass-2 replay
    // inside a single call, so we synthesise one StcPlan event per GOP
    // at the natural boundary.
    let total_gops = state.total_chunks as u32;
    emit_encode(
        params,
        EncodePhase::StcPlan {
            gop_idx: state.chunk_idx as u32,
            total_gops,
        },
    );

    let chunk_payload = &state.chunks[state.chunk_idx as usize];
    let framed = build_chunk_frame(state.chunk_idx, state.total_chunks, chunk_payload)?;
    let frame_bits = bytes_to_bits_msb_first_pub(&framed);

    let opts = EncodeOpts {
        qp: params.qp,
        intra_period: params.gop_size as i32,
    };
    let bitstream = encode_yuv_with_pre_framed_bits_4domain(
        &state.gop_buffer,
        params.width,
        params.height,
        frames_in_gop,
        opts,
        &frame_bits,
        &state.hhat_seed,
        &params.cost_weights,
    )?;
    out.extend_from_slice(&bitstream);

    state.chunk_idx += 1;
    state.frames_buffered = 0;
    state.gop_buffer.clear();

    // #474.2 — Pass2Replay fires AFTER the encoder returns. Reports
    // cumulative frames encoded so far so the smoothing engine can
    // close the per-GOP wall-clock fraction.
    let cumulative_done = state.chunk_idx as u32 * params.gop_size;
    let frame_idx = cumulative_done.min(params.total_frames_hint);
    emit_encode(
        params,
        EncodePhase::Pass2Replay {
            frame_idx,
            total_frames: params.total_frames_hint,
        },
    );
    Ok(())
}

// ─────────────────────── pure-Rust path ───────────────────────────────

fn build_pure_rust_state(
    params: &EncodeSessionParams,
    message_text: &str,
    passphrase: &str,
) -> Result<PureRustSessionState, StegoError> {
    // Build payload bytes + chunk-split UP-FRONT — matches OH264.
    let payload_bytes = payload::encode_payload(message_text, &[])?;
    let (ct, nonce, salt) = crypto::encrypt(&payload_bytes, passphrase)?;
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ct);

    let expected_n_gops = params.total_frames_hint.div_ceil(params.gop_size);
    if expected_n_gops == 0 || expected_n_gops > u16::MAX as u32 {
        return Err(StegoError::InvalidVideo(format!(
            "computed expected_n_gops {expected_n_gops} out of [1, {}]",
            u16::MAX
        )));
    }
    let total_chunks = expected_n_gops as u16;
    let chunks = split_message_into_chunks(&frame_bytes, total_chunks)?;

    // Derive hhat seed once — same convention as OH264 (gop_idx=0,
    // CoeffSign domain). The streaming decoder uses the same seed
    // across every GOP.
    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    let frame_bytes_size = (params.width as usize) * (params.height as usize) * 3 / 2;
    let gop_buffer = Vec::with_capacity(frame_bytes_size * params.gop_size as usize);

    Ok(PureRustSessionState {
        chunks,
        total_chunks,
        chunk_idx: 0,
        gop_buffer,
        frames_buffered_in_gop: 0,
        hhat_seed,
        total_frames_target: params.total_frames_hint,
        frames_seen_total: 0,
    })
}

fn pure_rust_push_frame(
    params: &EncodeSessionParams,
    state: &mut PureRustSessionState,
    frame: YuvFrameRef<'_>,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    if state.chunk_idx >= state.total_chunks {
        return Err(StegoError::InvalidVideo(format!(
            "pushed more frames than total_frames_hint expected ({} GOPs already emitted)",
            state.total_chunks
        )));
    }
    if state.frames_seen_total >= state.total_frames_target {
        return Err(StegoError::InvalidVideo(format!(
            "pushed more frames ({}) than total_frames_hint {}",
            state.frames_seen_total + 1, state.total_frames_target,
        )));
    }
    pack_frame_into_buffer(params.width, params.height, &frame, &mut state.gop_buffer)?;
    state.frames_buffered_in_gop += 1;
    state.frames_seen_total += 1;

    // #474.2 — Pass1Capture per pushed frame (pure-Rust path).
    emit_encode(
        params,
        EncodePhase::Pass1Capture {
            frame_idx: state.frames_seen_total,
            total_frames: params.total_frames_hint,
        },
    );

    if state.frames_buffered_in_gop == params.gop_size {
        drain_pure_rust_one_gop(params, state, out)?;
    }
    Ok(())
}

fn pure_rust_finish(
    params: &EncodeSessionParams,
    mut state: PureRustSessionState,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    if state.frames_seen_total == 0 {
        return Err(StegoError::InvalidVideo(
            "pure-Rust streaming session: finish() called with no frames pushed".into(),
        ));
    }
    if state.frames_buffered_in_gop > 0 {
        drain_pure_rust_one_gop(params, &mut state, out)?;
    }
    if state.chunk_idx != state.total_chunks {
        return Err(StegoError::InvalidVideo(format!(
            "pure-Rust streaming session: finished with {}/{} chunks emitted — total_frames_hint was too high",
            state.chunk_idx, state.total_chunks,
        )));
    }
    // #474.2 — Mux + Done (mirror the OH264 finish path).
    emit_encode(params, EncodePhase::Mux);
    emit_encode(params, EncodePhase::Done);
    Ok(())
}

fn drain_pure_rust_one_gop(
    params: &EncodeSessionParams,
    state: &mut PureRustSessionState,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    let frames_in_gop = state.frames_buffered_in_gop;
    if frames_in_gop == 0 {
        return Ok(());
    }
    // #474.2 — StcPlan before the per-GOP encoder call.
    let total_gops = state.total_chunks as u32;
    emit_encode(
        params,
        EncodePhase::StcPlan {
            gop_idx: state.chunk_idx as u32,
            total_gops,
        },
    );

    let chunk_payload = &state.chunks[state.chunk_idx as usize];
    let framed = build_chunk_frame(state.chunk_idx, state.total_chunks, chunk_payload)?;
    let chunk_bits = bytes_to_bits_msb_first(&framed);

    let bitstream = super::stego::encode_pixels::h264_stego_encode_one_gop_with_chunk_bits_4domain(
        &state.gop_buffer,
        params.width,
        params.height,
        frames_in_gop as usize,
        &chunk_bits,
        &state.hhat_seed,
        Some(params.qp as u8),
        &params.cost_weights,
    )?;
    out.extend_from_slice(&bitstream);

    state.chunk_idx += 1;
    state.frames_buffered_in_gop = 0;
    state.gop_buffer.clear();

    // #474.2 — Pass2Replay after the per-GOP encoder returns.
    let cumulative_done = state.chunk_idx as u32 * params.gop_size;
    let frame_idx = cumulative_done.min(params.total_frames_hint);
    emit_encode(
        params,
        EncodePhase::Pass2Replay {
            frame_idx,
            total_frames: params.total_frames_hint,
        },
    );
    Ok(())
}

/// MSB-first byte→bit helper local to streaming_session so the
/// pure-Rust path doesn't depend on the openh264-backend gate. Same
/// shape as `openh264_stego::bytes_to_bits_msb_first_pub`.
fn bytes_to_bits_msb_first(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(bytes.len() * 8);
    for &b in bytes {
        for i in (0..8).rev() {
            out.push((b >> i) & 1);
        }
    }
    out
}

// Shared by both engines (OH264 + PureRust) — strips row-padding from
// the input strides into a tight I420 layout the encoders expect.
fn pack_frame_into_buffer(
    width: u32,
    height: u32,
    frame: &YuvFrameRef<'_>,
    buf: &mut Vec<u8>,
) -> Result<(), StegoError> {
    let w = width as usize;
    let h = height as usize;
    let half_w = w / 2;
    let half_h = h / 2;

    if frame.y_stride < w || frame.u_stride < half_w || frame.v_stride < half_w {
        return Err(StegoError::InvalidVideo(format!(
            "frame stride too small: y={} u={} v={} for {w}x{h}",
            frame.y_stride, frame.u_stride, frame.v_stride
        )));
    }
    if frame.y.len() < frame.y_stride * h
        || frame.u.len() < frame.u_stride * half_h
        || frame.v.len() < frame.v_stride * half_h
    {
        return Err(StegoError::InvalidVideo(format!(
            "frame plane too small for {w}x{h}"
        )));
    }

    for row in 0..h {
        let start = row * frame.y_stride;
        buf.extend_from_slice(&frame.y[start..start + w]);
    }
    for row in 0..half_h {
        let start = row * frame.u_stride;
        buf.extend_from_slice(&frame.u[start..start + half_w]);
    }
    for row in 0..half_h {
        let start = row * frame.v_stride;
        buf.extend_from_slice(&frame.v[start..start + half_w]);
    }
    Ok(())
}

// ─────────────────────── capacity probe ──────────────────────────────
//
// #424 D.0.6 — `StreamingProbeSession` mirrors the encode session API
// but runs each GOP through a no-override baseline encode + cover-bits
// walker (`openh264_stego::count_cover_bits_for_gop`). It returns
// `CapacityProbeResult` with the raw CoeffSign cover count, the GOP
// count, and conservative per-shadow + primary-message budgets.
//
// **Why a real probe, not an analytical estimate**: cover-bit yield
// varies 5-10× across content (static dialogue vs. fast-motion
// landscapes) because OH264's mode-decision drives the residual /
// non-zero coefficient population. An analytical formula based on
// dims alone over- or under-estimates by an order of magnitude in
// realistic content. The probe is one full baseline encode pass
// (~50% the wall-clock of the actual stego encode); for UI it runs
// on a background thread with a "Calculating capacity..." spinner.
//
// **Conservative payload math**:
//   stc_payload_bits = cover_bits × 0.40
//     (STC at h=4 reaches ~0.45-0.50 in practice; 0.40 is the safety
//     floor that leaves margin for cascade-safety filtering loss on
//     ~5-15% of positions plus encoder coefficient drift between probe
//     and real encode.)
//   primary_bytes = stc_payload_bits / 8
//                 - 4 × n_gops   (chunk_frame header per GOP)
//                 - FRAME_OVERHEAD = 50   (encrypt envelope: 2 + salt
//                   16 + nonce 12 + tag 16 + plaintext-len 4)
//
// **Shadow math** is the existing collision-limited formula from
// `h264_stego_shadow_capacity`, working off the same cover_bits
// total. v1.0 streaming session doesn't actually wire shadows
// through yet — the shadow capacity is exposed for v1.1.

#[cfg(feature = "openh264-backend")]
use super::openh264_stego::count_cover_bits_for_gop;

/// Result of a `StreamingProbeSession::finish` — raw cover bits, GOP
/// count, and conservative payload budgets for primary + shadow modes.
#[derive(Debug, Clone, Copy)]
pub struct CapacityProbeResult {
    /// Total CoeffSign cover bits walked back from the baseline
    /// per-GOP encode. Sum across all GOPs.
    pub cover_bits: usize,
    /// Number of GOPs emitted (= number of chunk_frame headers in the
    /// real encode, each consuming 4 bytes of per-GOP payload).
    pub n_gops: u32,
}

/// Conservative STC payload ratio. h=4 STC reaches ~0.45-0.50 in
/// practice on real content; 0.40 leaves margin for cascade-safety
/// filtering + content drift between probe and real encode.
const STC_PAYLOAD_RATIO_NUM: usize = 40;
const STC_PAYLOAD_RATIO_DEN: usize = 100;

impl CapacityProbeResult {
    /// Maximum primary-message bytes (UTF-8 text + attached files
    /// combined, after envelope) that the streaming OH264 stego encode
    /// session can accept at this content + GOP shape.
    ///
    /// Math: `(cover_bits × 0.40 / 8) − (4 × n_gops) − 50`.
    /// Saturates at 0 when cover yield is too small to overcome
    /// envelope + chunk headers.
    pub fn primary_max_message_bytes(&self) -> usize {
        let payload_bits = self.cover_bits
            .saturating_mul(STC_PAYLOAD_RATIO_NUM)
            / STC_PAYLOAD_RATIO_DEN;
        let payload_bytes = payload_bits / 8;
        let chunk_overhead = (self.n_gops as usize)
            .saturating_mul(CHUNK_HEADER_LEN);
        payload_bytes
            .saturating_sub(chunk_overhead)
            .saturating_sub(frame::FRAME_OVERHEAD)
    }

    /// Maximum per-shadow message bytes under `n_shadows` total
    /// shadows. Mirrors the collision-limited formula in
    /// `h264_stego_shadow_capacity` (image-side / pure-Rust path) so
    /// callers get identical numbers across encoders.
    ///
    /// Returns 0 for `n_shadows == 0` (no shadow encoding needed).
    /// For `n_shadows == 1`, no inter-shadow collisions; capacity is
    /// the raw cover budget minus parity + per-shadow envelope.
    pub fn shadow_max_message_bytes(&self, n_shadows: u32) -> usize {
        use crate::stego::shadow_layer::max_shadow_plaintext_bytes;
        if n_shadows == 0 {
            return 0;
        }
        let denom = (n_shadows.saturating_sub(1)).max(1) as usize;
        let m_max_bits_sq = 1024usize
            .saturating_mul(self.cover_bits)
            / denom;
        let m_max_bits = (m_max_bits_sq as f64).sqrt() as usize;
        let m_max_bits = m_max_bits.min(self.cover_bits);
        let m_max_bytes = m_max_bits / 8;
        // 2026-05-21 — picks v1 or v2 frame overhead automatically
        // (v2 gives 4 fewer bytes capacity but unlocks >64KB
        // plaintext).
        max_shadow_plaintext_bytes(m_max_bytes.saturating_sub(128))
    }
}

/// Streaming capacity probe — mirror of `StreamingEncodeSession` but
/// emits no output; on `finish()` returns the cover-bit count and
/// conservative payload budgets.
///
/// Lifecycle matches encode: `create(params)` → repeated `push_frame`
/// → `finish()` returns the result. Per-GOP baseline encode happens
/// inside `push_frame` on every gop_size frames.
///
/// Memory bound: O(gop_size × frame_size) — same as encode session.
/// Wall-clock: ~50% of the actual stego encode (no STC plan, no
/// override application).
pub struct StreamingProbeSession {
    params: EncodeSessionParams,
    inner: ProbeImpl,
    cover_bits: usize,
    n_gops: u32,
    /// #475 — fires once per GOP completion so the caller can render
    /// a refining estimated-capacity in the HUD. `None` when no
    /// progressive estimate is wanted (e.g. CLI batch use).
    progress_callback: Option<super::progress::CapacityProbeCallback>,
}

enum ProbeImpl {
    #[cfg(feature = "openh264-backend")]
    Oh264(Oh264ProbeState),
    /// Pure-Rust capacity probe not exposed in v1.0 — callers should
    /// run the existing `h264_stego_capacity_4domain` directly. The
    /// streaming session's pure-Rust path is itself buffered (#472),
    /// so a streaming-probe-for-pure-Rust would carry the same
    /// caveats. Errors at create() for now.
    PureRust,
    /// Engine requested but compiled out.
    EngineDisabled(EncodeEngineChoice),
}

#[cfg(feature = "openh264-backend")]
struct Oh264ProbeState {
    /// Tight I420 GOP buffer.
    gop_buffer: Vec<u8>,
    /// Frames buffered in `gop_buffer`.
    frames_buffered: u32,
}

impl StreamingProbeSession {
    /// Create a new capacity-probe session.
    ///
    /// # Errors
    /// Same dimension/GOP validation as `StreamingEncodeSession::create`.
    /// Errors if the requested engine isn't compiled in.
    pub fn create(params: EncodeSessionParams) -> Result<Self, StegoError> {
        if params.width == 0 || params.height == 0 {
            return Err(StegoError::InvalidVideo(format!(
                "invalid dimensions: {}x{}",
                params.width, params.height
            )));
        }
        if params.width % 16 != 0 || params.height % 16 != 0 {
            return Err(StegoError::InvalidVideo(format!(
                "dimensions must be 16-aligned, got {}x{}",
                params.width, params.height
            )));
        }
        if params.gop_size == 0 {
            return Err(StegoError::InvalidVideo("gop_size must be > 0".into()));
        }
        if params.total_frames_hint == 0 {
            return Err(StegoError::InvalidVideo(
                "total_frames_hint must be > 0".into(),
            ));
        }

        let inner = match params.engine {
            EncodeEngineChoice::Oh264 => {
                #[cfg(feature = "openh264-backend")]
                {
                    let frame_bytes =
                        (params.width as usize) * (params.height as usize) * 3 / 2;
                    let gop_capacity = frame_bytes * (params.gop_size as usize);
                    ProbeImpl::Oh264(Oh264ProbeState {
                        gop_buffer: Vec::with_capacity(gop_capacity),
                        frames_buffered: 0,
                    })
                }
                #[cfg(not(feature = "openh264-backend"))]
                ProbeImpl::EngineDisabled(EncodeEngineChoice::Oh264)
            }
            EncodeEngineChoice::PureRust => ProbeImpl::PureRust,
        };

        Ok(Self {
            params,
            inner,
            cover_bits: 0,
            n_gops: 0,
            progress_callback: None,
        })
    }

    /// #475 — install a callback that fires after each GOP probe
    /// completes, with `(gops_done, gops_total, cover_bits_so_far)`.
    /// Caller computes the estimated total cover_bits as
    /// `cover_bits_so_far * gops_total / gops_done`. `gops_total` is
    /// derived from `params.total_frames_hint / gop_size`.
    pub fn set_progress_callback(
        &mut self,
        callback: Option<super::progress::CapacityProbeCallback>,
    ) {
        self.progress_callback = callback;
    }

    pub fn with_progress_callback(
        mut self,
        callback: super::progress::CapacityProbeCallback,
    ) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// Emit one progress event. Called from `push_frame`/`finish`
    /// after each GOP drain. No-op when no callback is installed.
    fn emit_estimate(&self) {
        if let Some(cb) = self.progress_callback.as_ref() {
            // Total GOPs target = ceil(total_frames_hint / gop_size).
            // The final partial GOP (if any) bumps `n_gops` past
            // this target on the last drain — that's fine, the
            // mobile UI clamps the displayed denominator at the
            // running max anyway.
            let total_gops = self.params.total_frames_hint
                .div_ceil(self.params.gop_size)
                .max(1);
            cb(self.n_gops, total_gops, self.cover_bits);
        }
    }

    /// Push one YUV frame. When the in-flight GOP fills, runs the
    /// per-GOP probe (baseline OH264 encode + walker) and accumulates
    /// the cover-bits count.
    pub fn push_frame(&mut self, frame: YuvFrameRef<'_>) -> Result<(), StegoError> {
        match &mut self.inner {
            #[cfg(feature = "openh264-backend")]
            ProbeImpl::Oh264(state) => {
                pack_frame_into_buffer(self.params.width, self.params.height, &frame, &mut state.gop_buffer)?;
                state.frames_buffered += 1;
                if state.frames_buffered == self.params.gop_size {
                    drain_one_gop_probe(&self.params, state, &mut self.cover_bits, &mut self.n_gops)?;
                    self.emit_estimate();
                }
                Ok(())
            }
            ProbeImpl::PureRust => Err(StegoError::InvalidVideo(
                "capacity probe not implemented for pure-Rust engine; call \
                 h264_stego_capacity_4domain directly on a YUV buffer"
                    .into(),
            )),
            ProbeImpl::EngineDisabled(c) => Err(StegoError::InvalidVideo(format!(
                "engine {:?} is not compiled into this build",
                c
            ))),
        }
    }

    /// Return the session's params (read-only). Bridges use this to
    /// recover the originally-requested dims for plane-slice sizing
    /// in `push_frame` callers.
    pub fn params(&self) -> &EncodeSessionParams {
        &self.params
    }

    /// Drain the final partial GOP (if any) and return the result.
    /// CONSUMES self — handle is invalid after this call.
    pub fn finish(mut self) -> Result<CapacityProbeResult, StegoError> {
        match &mut self.inner {
            #[cfg(feature = "openh264-backend")]
            ProbeImpl::Oh264(state) => {
                if state.frames_buffered > 0 {
                    drain_one_gop_probe(&self.params, state, &mut self.cover_bits, &mut self.n_gops)?;
                    // Emit one final per-GOP event before returning so
                    // listeners that race a final-state read can see
                    // the full count rather than the second-last one.
                    self.emit_estimate();
                }
                Ok(CapacityProbeResult {
                    cover_bits: self.cover_bits,
                    n_gops: self.n_gops,
                })
            }
            ProbeImpl::PureRust => Err(StegoError::InvalidVideo(
                "capacity probe not implemented for pure-Rust engine".into(),
            )),
            ProbeImpl::EngineDisabled(c) => Err(StegoError::InvalidVideo(format!(
                "engine {:?} is not compiled into this build",
                c
            ))),
        }
    }
}

#[cfg(feature = "openh264-backend")]
fn drain_one_gop_probe(
    params: &EncodeSessionParams,
    state: &mut Oh264ProbeState,
    cover_bits: &mut usize,
    n_gops: &mut u32,
) -> Result<(), StegoError> {
    let frames_in_gop = state.frames_buffered;
    if frames_in_gop == 0 {
        return Ok(());
    }
    let opts = super::openh264_stego::EncodeOpts {
        qp: params.qp,
        intra_period: params.gop_size as i32,
    };
    let bits = count_cover_bits_for_gop(
        &state.gop_buffer,
        params.width,
        params.height,
        frames_in_gop,
        opts,
    )?;
    *cover_bits = cover_bits.saturating_add(bits);
    *n_gops = n_gops.saturating_add(1);
    state.frames_buffered = 0;
    state.gop_buffer.clear();
    Ok(())
}

// ─────────────────────── decode side ──────────────────────────────────

/// Opaque streaming-decode session state.
///
/// Decodes chunk-framed streaming stego: each GOP is independently
/// STC-extracted, the per-GOP chunk_frame header steers reassembly,
/// and the concatenated payload bytes are decrypted as a single
/// phasm v1 frame. The legacy (whole-video STC) format produced by
/// the one-shot `openh264_stego_encode_yuv_string` is NOT handled
/// here — callers fall back to `openh264_stego_decode_yuv` when this
/// path fails.
///
/// Auto-detection across both formats is the consumer-side job
/// (mobile `smart_decode_video` will try streaming first, fall back
/// to one-shot legacy).
pub struct StreamingDecodeSession {
    passphrase: String,
    accumulator: Vec<u8>,
    /// #493.5 Phase 4 — must match the encoder's CostWeights for the
    /// 4-domain combined-cover STC extract to recover the original
    /// chunk_bits. `create(passphrase)` defaults to `CostWeights::default()`
    /// (same as the encoder default); callers using non-default
    /// weights should use `create_with_weights`.
    cost_weights: CostWeights,
    /// #474.1 — optional progress callback. Emit sites in `finish`
    /// invoke this on each `DecodePhase` transition. Wrapped in
    /// `Arc<dyn Fn>` so the session itself stays Send + Sync without
    /// constraining the callback shape (e.g. a mobile bridge marshals
    /// to the main thread asynchronously).
    progress_callback: Option<DecodeProgressCallback>,
}

/// Per-decode-session result returned by `finish`.
#[derive(Debug, Clone)]
pub struct DecodeSessionResult {
    pub text: String,
    /// 1 = Ghost, 2 = Armor. Always 1 for the H.264 stego path.
    pub mode_id: u8,
}

impl StreamingDecodeSession {
    pub fn create(passphrase: &str) -> Result<Self, StegoError> {
        Self::create_with_weights(passphrase, CostWeights::default())
    }

    /// #493.5 — create with explicit `CostWeights`. Must match the
    /// encoder's `EncodeSessionParams.cost_weights` for the 4-domain
    /// combined-cover extract to recover the message.
    pub fn create_with_weights(
        passphrase: &str,
        cost_weights: CostWeights,
    ) -> Result<Self, StegoError> {
        Ok(Self {
            passphrase: passphrase.to_string(),
            accumulator: Vec::new(),
            cost_weights,
            progress_callback: None,
        })
    }

    /// #474.1 — install a progress callback on this session. See
    /// [`DecodePhase`] for the event vocabulary; events fire from
    /// `push_annex_b` (Demux), the per-GOP brute-force loop in
    /// `finish` (Walker / ShadowExtract), and the decrypt step.
    pub fn with_progress_callback(mut self, callback: DecodeProgressCallback) -> Self {
        self.progress_callback = Some(callback);
        self
    }

    /// #474.2 — mutator variant of `with_progress_callback` for FFI
    /// callers that own the handle and want to attach a callback after
    /// `create()`.
    pub fn set_progress_callback(&mut self, callback: Option<DecodeProgressCallback>) {
        self.progress_callback = callback;
    }

    /// Internal helper — invoke the progress callback if installed.
    /// Cheap no-op when no callback is registered.
    #[allow(dead_code)]
    pub(crate) fn emit_progress(&self, phase: DecodePhase) {
        if let Some(cb) = &self.progress_callback {
            cb(phase);
        }
    }

    pub fn push_annex_b(&mut self, nals: &[u8]) -> Result<(), StegoError> {
        self.accumulator.extend_from_slice(nals);
        Ok(())
    }

    /// Finalise the session and return the recovered message.
    ///
    /// Splits the accumulated Annex-B at SPS NAL boundaries (every
    /// GOP starts SPS+PPS+IDR in the streaming session's output),
    /// brute-forces STC extract + chunk_frame parse per GOP, then
    /// assembles chunks by `chunk_index`, runs `parse_frame` on the
    /// concatenated payload, decrypts, and returns the message text.
    pub fn finish(self) -> Result<DecodeSessionResult, StegoError> {
        // #474.2 — Demux boundary event. Fires once after the buffered
        // Annex-B has been split into per-GOP slabs.
        self.emit_progress(DecodePhase::Demux);

        let slabs = split_annex_b_into_gops(&self.accumulator);
        if slabs.is_empty() {
            return Err(StegoError::InvalidVideo(
                "decode session: no GOPs found in accumulated Annex-B".into(),
            ));
        }

        // Derive the STC seed (same as encode side).
        let keys = CabacStegoMasterKeys::derive(&self.passphrase)?;
        let hhat_seed = keys
            .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
            .hhat_seed;

        let mut collected: Vec<(u16, Vec<u8>)> = Vec::with_capacity(slabs.len());
        let mut seen_total_chunks: Option<u16> = None;

        let total_slabs = slabs.len() as u32;
        for (gop_idx, slab) in slabs.iter().enumerate() {
            let candidate = try_extract_chunk_from_gop(
                slab, &hhat_seed, gop_idx as u16, &self.cost_weights,
            )
                .ok_or_else(|| {
                    StegoError::InvalidVideo(format!(
                        "decode session: no chunk_frame found in GOP {gop_idx}"
                    ))
                })?;
            // Cross-GOP consistency: all chunks must agree on total_chunks.
            match seen_total_chunks {
                None => seen_total_chunks = Some(candidate.total_chunks),
                Some(t) if t == candidate.total_chunks => {}
                Some(t) => {
                    return Err(StegoError::InvalidVideo(format!(
                        "decode session: GOP {gop_idx} reports total_chunks={} but earlier GOPs reported {}",
                        candidate.total_chunks, t,
                    )));
                }
            }
            collected.push((candidate.chunk_idx, candidate.payload));

            // #474.2 — Walker progress event per slab. Streaming decode
            // doesn't expose per-frame walker granularity (each slab is
            // walked + STC-extracted as one unit), so we report at slab
            // granularity. `total_frames` here means "total slabs" for
            // the smoothing engine's fraction math — it doesn't care
            // whether the unit is a frame or a GOP.
            self.emit_progress(DecodePhase::Walker {
                frame_idx: (gop_idx as u32) + 1,
                total_frames: total_slabs,
            });
        }

        let total_chunks = seen_total_chunks.expect("at least one GOP processed");
        if (slabs.len() as u32) != total_chunks as u32 {
            return Err(StegoError::InvalidVideo(format!(
                "decode session: extracted {} GOPs but chunk_frame header reports total_chunks={}",
                slabs.len(),
                total_chunks,
            )));
        }

        let frame_bytes = assemble_chunks(collected, total_chunks).ok_or_else(|| {
            StegoError::InvalidVideo(
                "decode session: chunk reassembly failed (missing/duplicate index)".into(),
            )
        })?;

        // #474.2 — StcExtract: brute-force STC retrieval already
        // happened inside `try_extract_chunk_from_gop` (combined walk
        // + STC extract per slab). Fire StcExtract once here as the
        // boundary marker between Walker and Decrypt — gives the
        // smoothing engine a clean phase transition.
        self.emit_progress(DecodePhase::StcExtract);

        let parsed = frame::parse_frame(&frame_bytes)?;
        // #474.2 — Decrypt fires immediately before crypto::decrypt.
        // Always a single event (decrypt is sub-millisecond on short
        // messages and not worth per-chunk granularity).
        self.emit_progress(DecodePhase::Decrypt);
        let plaintext = crypto::decrypt(
            &parsed.ciphertext,
            &self.passphrase,
            &parsed.salt,
            &parsed.nonce,
        )?;
        let payload_data = payload::decode_payload(&plaintext)?;
        // #474.2 — Done boundary fires AFTER the message text is ready
        // so the UI sees the final state before this function returns.
        self.emit_progress(DecodePhase::Done);
        Ok(DecodeSessionResult {
            text: payload_data.text,
            mode_id: 1, // Ghost = H.264 stego path
        })
    }

    #[allow(dead_code)]
    pub(crate) fn buffered_bytes(&self) -> usize {
        self.accumulator.len()
    }

    #[allow(dead_code)]
    pub(crate) fn passphrase(&self) -> &str {
        &self.passphrase
    }
}

// ─────────────────────── decode helpers ───────────────────────────────

/// One successful chunk extract from a single GOP slab.
struct ChunkCandidate {
    chunk_idx: u16,
    total_chunks: u16,
    payload: Vec<u8>,
}

/// Split the Annex-B stream into per-GOP slabs at SPS NAL boundaries.
///
/// The streaming encode session restarts the OpenH264 encoder per
/// GOP (each `encode_yuv_with_pre_framed_bits` call), and OpenH264
/// emits SPS+PPS at every IDR. So SPS-position-marked NALs are
/// reliable GOP boundaries in our streaming output. Any prefix
/// before the first SPS is dropped.
fn split_annex_b_into_gops(annex_b: &[u8]) -> Vec<&[u8]> {
    let mut sps_offsets: Vec<usize> = Vec::new();
    let mut i = 0;
    while i + 3 < annex_b.len() {
        let (sc_len, nal_off) = if annex_b[i..].starts_with(&[0, 0, 0, 1]) {
            (4usize, i + 4)
        } else if annex_b[i..].starts_with(&[0, 0, 1]) {
            (3usize, i + 3)
        } else {
            i += 1;
            continue;
        };
        if nal_off < annex_b.len() {
            let nal_type = annex_b[nal_off] & 0x1F;
            if nal_type == 7 {
                sps_offsets.push(i);
            }
        }
        i = nal_off;
        let _ = sc_len; // silence unused-var on some cfg combos
    }
    if sps_offsets.is_empty() {
        return Vec::new();
    }
    let mut slabs = Vec::with_capacity(sps_offsets.len());
    let mut prev = sps_offsets[0];
    for &next in &sps_offsets[1..] {
        slabs.push(&annex_b[prev..next]);
        prev = next;
    }
    slabs.push(&annex_b[prev..]);
    slabs
}

/// Walk a GOP slab and brute-force the STC m_total until a chunk_frame
/// header parses cleanly. Validates `chunk_idx == expected_chunk_idx`
/// so the streaming session's strict-ordering invariant catches false
/// positives early.
fn try_extract_chunk_from_gop(
    slab: &[u8],
    hhat_seed: &[u8; 32],
    expected_chunk_idx: u16,
    weights: &CostWeights,
) -> Option<ChunkCandidate> {
    // #493.5 Phase 4 — walk with MVD recording on so the 4-domain
    // combine sees all 4 cover vectors. Default walker leaves MVD
    // cover empty (Phase 0 finding).
    let walk = walk_annex_b_for_cover_with_options(
        slab,
        WalkOptions { record_mvd: true, ..Default::default() },
    ).ok()?;
    // Combine 4-domain cover in canonical order CS → CSL → MVDs →
    // MVDsl, matching the encoder side. STC extract only needs the
    // combined bit vector; costs are unused at extract (dummy here).
    let dummy_costs = DomainCosts::default();
    let (cover_bits, _, _boundaries) =
        combine_cover_4domain(&walk.cover, &dummy_costs, weights);
    let n_cover = cover_bits.len();
    if n_cover < CHUNK_HEADER_LEN * 8 {
        return None;
    }

    // m_total iterates in 8-bit steps starting from the smallest
    // possible value (just the chunk header). bits_to_bytes truncates
    // partial bytes, so non-aligned m_total wouldn't recover anything
    // useful anyway.
    let min_m = CHUNK_HEADER_LEN * 8;
    let max_m = n_cover;

    // #519 / #516.3 — chunk-idx early-reject. Same shape as the
    // single-GOP `try_decode_at` length-prefix early-reject (#516.2):
    // the chunk_frame's first 2 bytes are `chunk_idx: u16 BE`. For
    // the right m_total the prefix matches `expected_chunk_idx`; for
    // wrong m_total the prefix is essentially random and the equality
    // holds with probability 1/65536. Almost every wrong candidate
    // rejects after a 16-bit partial extract (O(16 * w) ops) instead
    // of the full m_total * w STC extract.
    //
    // At 1080p × 30f IBPBP the combined 4-domain cover is ~100k+
    // bits → ~12k brute-force candidates. Without this filter each
    // candidate does an O(n_cover) extract = O(n_cover²) total work
    // which is multi-second on a phone. With this filter, total work
    // drops from O(n_cover²) to O(n_cover · n_tries / 65536) plus
    // ~1 full extract on the winner.
    let trace = std::env::var("PHASM_PERF_TRACE").map(|v| v == "1").unwrap_or(false);
    let mut tries = 0usize;
    let mut idx_match = 0usize;
    let mut parse_ok = 0usize;
    let mut promoted = 0usize;

    let mut m_total = min_m;
    while m_total <= max_m {
        tries += 1;
        let w = n_cover / m_total;
        if w == 0 {
            break;
        }
        let used = m_total * w;
        let hhat = generate_hhat(STC_H, w, hhat_seed);

        // Partial extract — first 16 syndrome bits = chunk_idx u16 BE.
        let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat, w, 16);
        if prefix_bits.len() < 16 {
            m_total += 8;
            continue;
        }
        let prefix_bytes = bits_to_bytes_msb_first(&prefix_bits);
        let candidate_chunk_idx = u16::from_be_bytes([prefix_bytes[0], prefix_bytes[1]]);
        if candidate_chunk_idx != expected_chunk_idx {
            m_total += 8;
            continue;
        }
        idx_match += 1;

        // chunk_idx matches — promote to full extract + parse +
        // total_chunks sanity check. False-positive probability is
        // 1/65536; the total_chunks bound + caller's CRC/AEAD filter
        // catches those.
        promoted += 1;
        let extracted = stc_extract(&cover_bits[..used], &hhat, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = bits_to_bytes_msb_first(bits);
        if let Some((chunk_idx, total_chunks, payload)) = parse_chunk_frame(&bytes) {
            parse_ok += 1;
            if total_chunks > 0
                && total_chunks <= MAX_REASONABLE_CHUNKS
                && chunk_idx == expected_chunk_idx
            {
                if trace {
                    eprintln!(
                        "[PHASM_PERF_TRACE chunk_gop] expected={} n_cover={} tries={} idx_match={} parse_ok={} → HIT m_total={} total_chunks={}",
                        expected_chunk_idx, n_cover, tries, idx_match, parse_ok, m_total, total_chunks,
                    );
                }
                return Some(ChunkCandidate {
                    chunk_idx,
                    total_chunks,
                    payload: payload.to_vec(),
                });
            }
        }
        m_total += 8;
    }
    if trace {
        eprintln!(
            "[PHASM_PERF_TRACE chunk_gop] expected={} n_cover={} tries={} idx_match={} promoted={} parse_ok={} → MISS",
            expected_chunk_idx, n_cover, tries, idx_match, promoted, parse_ok,
        );
    }
    None
}

/// Local copy of the MSB-first bit→byte helper. Inlined here to keep
/// the decode side independent of `openh264_stego` (which is gated by
/// `openh264-backend`).
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

    fn ok_params() -> EncodeSessionParams {
        EncodeSessionParams {
            width: 1920, height: 1080, fps_num: 30, fps_den: 1,
            qp: 23, gop_size: 30, total_frames_hint: 300,
            color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264,
            cost_weights: CostWeights::default(),
            progress_callback: None,
        }
    }

    #[test]
    fn engine_choice_from_u8_defaults_to_oh264() {
        assert_eq!(EncodeEngineChoice::from_u8(0), EncodeEngineChoice::Oh264);
        assert_eq!(EncodeEngineChoice::from_u8(1), EncodeEngineChoice::PureRust);
        assert_eq!(EncodeEngineChoice::from_u8(255), EncodeEngineChoice::Oh264);
    }

    #[test]
    fn create_rejects_zero_dims() {
        let mut bad = ok_params();
        bad.width = 0;
        assert!(StreamingEncodeSession::create(bad, "hi", "pw").is_err());
    }

    #[test]
    fn create_rejects_non_16_aligned_dims() {
        let mut bad = ok_params();
        bad.width = 1234; // not 16-aligned
        assert!(StreamingEncodeSession::create(bad, "hi", "pw").is_err());
    }

    #[test]
    fn create_rejects_zero_gop() {
        let mut bad = ok_params();
        bad.gop_size = 0;
        assert!(StreamingEncodeSession::create(bad, "hi", "pw").is_err());
    }

    #[test]
    fn create_rejects_zero_total_frames_hint() {
        let mut bad = ok_params();
        bad.total_frames_hint = 0;
        assert!(StreamingEncodeSession::create(bad, "hi", "pw").is_err());
    }

    #[test]
    fn decode_session_buffers_and_errors_on_finish() {
        let mut s = StreamingDecodeSession::create("pw").unwrap();
        let nal = b"\x00\x00\x00\x01dummy nal";
        s.push_annex_b(nal).unwrap();
        assert_eq!(s.buffered_bytes(), nal.len());
        assert!(s.finish().is_err());
    }

    #[test]
    fn encode_params_accept_progress_callback() {
        use std::sync::{Arc, Mutex};
        let recorded: Arc<Mutex<Vec<EncodePhase>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded_clone = Arc::clone(&recorded);
        let cb: EncodeProgressCallback = Arc::new(move |phase| {
            recorded_clone.lock().unwrap().push(phase);
        });

        let mut p = ok_params();
        p.progress_callback = Some(Arc::clone(&cb));
        assert!(p.progress_callback.is_some());

        // Debug impl tolerates the unprintable Arc<dyn Fn>.
        let dbg = format!("{:?}", p);
        assert!(dbg.contains("<callback>"), "Debug should show placeholder, got {dbg}");

        // Cloning the params clones the Arc, both still call.
        let p2 = p.clone();
        (p.progress_callback.as_ref().unwrap())(EncodePhase::Setup);
        (p2.progress_callback.as_ref().unwrap())(EncodePhase::Done);
        let got = recorded.lock().unwrap().clone();
        assert_eq!(got, vec![EncodePhase::Setup, EncodePhase::Done]);
    }

    #[test]
    fn decode_session_with_progress_callback_round_trip() {
        use std::sync::{Arc, Mutex};
        let recorded: Arc<Mutex<Vec<DecodePhase>>> = Arc::new(Mutex::new(Vec::new()));
        let recorded_clone = Arc::clone(&recorded);
        let cb: DecodeProgressCallback = Arc::new(move |phase| {
            recorded_clone.lock().unwrap().push(phase);
        });

        let s = StreamingDecodeSession::create("pw").unwrap().with_progress_callback(cb);
        s.emit_progress(DecodePhase::Demux);
        s.emit_progress(DecodePhase::Done);
        let got = recorded.lock().unwrap().clone();
        assert_eq!(got, vec![DecodePhase::Demux, DecodePhase::Done]);
    }

    #[cfg(feature = "openh264-backend")]
    mod oh264 {
        use super::*;
        use crate::codec::h264::openh264::SESSION_TEST_MUTEX;

        fn synth_yuv_frame(w: u32, h: u32, seed: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
            let mut y = Vec::with_capacity((w * h) as usize);
            for j in 0..h {
                for i in 0..w {
                    let v = ((i + seed * 2) ^ (j + seed * 3)) as u8;
                    y.push(v);
                }
            }
            let half_w = w / 2;
            let half_h = h / 2;
            let mut u = Vec::with_capacity((half_w * half_h) as usize);
            let mut v = Vec::with_capacity((half_w * half_h) as usize);
            let mut s: u32 = 0xCAFE_F00D ^ seed;
            for j in 0..half_h {
                for i in 0..half_w {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let tex = (s >> 16) as u8;
                    let pos = (i + j + seed) as u8;
                    u.push(tex.wrapping_add(pos));
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    let tex2 = (s >> 16) as u8;
                    v.push(tex2.wrapping_add(pos));
                }
            }
            (y, u, v)
        }

        #[test]
        fn streaming_session_oh264_emits_annex_b() {
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            const GOP: u32 = 2;
            const N: u32 = 4; // 2 GOPs of 2 frames each.

            let params = EncodeSessionParams {
                width: W, height: H,
                fps_num: 30, fps_den: 1,
                qp: 26, gop_size: GOP, total_frames_hint: N,
                color: ColorParams::default(),
                engine: EncodeEngineChoice::Oh264,
                cost_weights: CostWeights::default(),
                progress_callback: None,
            };
            let mut sess =
                StreamingEncodeSession::create(params, "hi", "pw").expect("create");
            let mut out = Vec::new();
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                let frame = YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                };
                sess.push_frame(frame, &mut out).expect("push");
            }
            sess.finish(&mut out).expect("finish");
            assert!(
                out.len() > 100,
                "expected non-trivial Annex-B output, got {} bytes",
                out.len()
            );
            // Sanity: Annex-B starts with a NAL start code.
            assert!(
                out.windows(4).any(|w| w == [0, 0, 0, 1]) || out.starts_with(&[0, 0, 1]),
                "expected Annex-B start code in output"
            );
        }

        #[test]
        fn streaming_session_oh264_rejects_overflow() {
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            let params = EncodeSessionParams {
                width: W, height: H,
                fps_num: 30, fps_den: 1,
                qp: 26, gop_size: 2, total_frames_hint: 2, // exactly 1 GOP
                color: ColorParams::default(),
                engine: EncodeEngineChoice::Oh264,
                cost_weights: CostWeights::default(),
                progress_callback: None,
            };
            let mut sess =
                StreamingEncodeSession::create(params, "hi", "pw").expect("create");
            let mut out = Vec::new();
            for f in 0..2 {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                let frame = YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                };
                sess.push_frame(frame, &mut out).expect("push");
            }
            // First GOP completed → chunk_idx == total_chunks.
            // A 3rd frame should now be rejected.
            let (y, u, v) = synth_yuv_frame(W, H, 2);
            let frame = YuvFrameRef {
                y: &y, y_stride: W as usize,
                u: &u, u_stride: (W / 2) as usize,
                v: &v, v_stride: (W / 2) as usize,
            };
            let err = sess.push_frame(frame, &mut out).unwrap_err();
            assert!(matches!(err, StegoError::InvalidVideo(_)));
        }

        #[test]
        fn streaming_session_pure_rust_emits_annex_b() {
            // D.0.7.3.b (#472) per-GOP streaming: pure-Rust streaming
            // session drains each GOP independently into `out` as
            // frames arrive — matches the OH264 path. Memory is
            // O(gop_size × frame_size) at any moment, not
            // O(total_frames × frame_size). Tiny fixture (128×80 × 4f
            // × GOP=2 → 2 GOPs) verifies both per-GOP drain and
            // finish-side tail handling.
            const W: u32 = 128;
            const H: u32 = 80;
            const N: u32 = 4;
            const GOP: u32 = 2;
            let params = EncodeSessionParams {
                width: W, height: H,
                fps_num: 30, fps_den: 1,
                qp: 26, gop_size: GOP, total_frames_hint: N,
                color: ColorParams::default(),
                engine: EncodeEngineChoice::PureRust,
                cost_weights: CostWeights::default(),
                progress_callback: None,
            };
            let mut sess =
                StreamingEncodeSession::create(params, "hi", "pw").expect("create");
            let mut out = Vec::new();
            let mut bytes_after_gop1 = 0;
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                let frame = YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                };
                sess.push_frame(frame, &mut out).expect("push");
                if f == GOP - 1 {
                    bytes_after_gop1 = out.len();
                }
            }
            // After the first complete GOP (f == GOP-1), bytes MUST
            // have been emitted — proves the per-GOP drain is live.
            assert!(
                bytes_after_gop1 > 0,
                "PureRust path should emit bytes after each complete GOP, \
                 but `out` was empty after GOP 1 (f={})",
                GOP - 1
            );
            sess.finish(&mut out).expect("finish");
            assert!(
                out.len() > bytes_after_gop1,
                "expected finish() to add bytes for GOP 2 — \
                 got {} after GOP 1, {} after finish",
                bytes_after_gop1, out.len()
            );
        }

        #[test]
        fn streaming_session_oh264_decode_session_roundtrip() {
            // D.0.7.11 end-to-end: encode via streaming session → push the
            // emitted Annex-B through the streaming decode session →
            // verify the recovered text matches the input. Single-GOP
            // first to exercise the chunk_frame header path with the
            // smallest valid configuration.
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            const GOP: u32 = 2;
            const N: u32 = 2; // 1 GOP — start small for first round-trip.
            const MSG: &str = "hi";
            const PASS: &str = "pw";

            let params = EncodeSessionParams {
                width: W, height: H,
                fps_num: 30, fps_den: 1,
                qp: 26, gop_size: GOP, total_frames_hint: N,
                color: ColorParams::default(),
                engine: EncodeEngineChoice::Oh264,
                cost_weights: CostWeights::default(),
                progress_callback: None,
            };
            let mut enc = StreamingEncodeSession::create(params, MSG, PASS).expect("encode create");
            let mut annex_b = Vec::new();
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                let frame = YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                };
                enc.push_frame(frame, &mut annex_b).expect("push");
            }
            enc.finish(&mut annex_b).expect("finish encode");

            let mut dec = StreamingDecodeSession::create(PASS).expect("decode create");
            dec.push_annex_b(&annex_b).expect("push annex");
            let result = dec.finish().expect("finish decode");
            assert_eq!(
                result.text, MSG,
                "round-trip text mismatch: got {:?}", result.text
            );
            assert_eq!(result.mode_id, 1);
        }

        #[test]
        fn streaming_session_pure_rust_decode_session_roundtrip() {
            // #472.3 — round-trip pure-Rust streaming encode through the
            // SAME StreamingDecodeSession the OH264 path uses. The
            // pure-Rust per-GOP encode emits chunk_frame'd CoeffSign
            // bits per GOP with the same hhat_seed convention as
            // OH264, so decode must work with zero changes.
            //
            // Two-GOP fixture (320×240 × 4f × GOP=2 → 2 GOPs) verifies
            // the cross-GOP chunk reassembly path, not just single-GOP.
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            const GOP: u32 = 2;
            const N: u32 = 4; // 2 GOPs
            const MSG: &str = "hi from rust";
            const PASS: &str = "pw";

            let params = EncodeSessionParams {
                width: W, height: H,
                fps_num: 30, fps_den: 1,
                qp: 26, gop_size: GOP, total_frames_hint: N,
                color: ColorParams::default(),
                engine: EncodeEngineChoice::PureRust,
                cost_weights: CostWeights::default(),
                progress_callback: None,
            };
            let mut enc = StreamingEncodeSession::create(params, MSG, PASS).expect("encode create");
            let mut annex_b = Vec::new();
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                let frame = YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                };
                enc.push_frame(frame, &mut annex_b).expect("push");
            }
            enc.finish(&mut annex_b).expect("finish encode");

            let mut dec = StreamingDecodeSession::create(PASS).expect("decode create");
            dec.push_annex_b(&annex_b).expect("push annex");
            let result = dec.finish().expect("finish decode");
            assert_eq!(
                result.text, MSG,
                "round-trip text mismatch: got {:?}", result.text
            );
            assert_eq!(result.mode_id, 1);
        }

        #[test]
        fn streaming_session_emits_progress_events_round_trip() {
            // #474.2 — end-to-end: install encode + decode progress
            // callbacks, run a 2-GOP pure-Rust round-trip, verify the
            // event sequences are well-formed (Setup first, Done last,
            // monotonic frame_idx / gop_idx). Pure-Rust path is used
            // because OH264 needs the SESSION_TEST_MUTEX and the encode
            // session lifecycle here is identical between backends.
            const W: u32 = 128;
            const H: u32 = 80;
            const GOP: u32 = 2;
            const N: u32 = 4;
            const MSG: &str = "progress probe";
            const PASS: &str = "pw";

            use std::sync::{Arc, Mutex};
            let enc_events: Arc<Mutex<Vec<EncodePhase>>> = Arc::new(Mutex::new(Vec::new()));
            let enc_clone = Arc::clone(&enc_events);
            let enc_cb: EncodeProgressCallback = Arc::new(move |phase| {
                enc_clone.lock().unwrap().push(phase);
            });

            let params = EncodeSessionParams {
                width: W, height: H,
                fps_num: 30, fps_den: 1,
                qp: 26, gop_size: GOP, total_frames_hint: N,
                color: ColorParams::default(),
                engine: EncodeEngineChoice::PureRust,
                cost_weights: CostWeights::default(),
                progress_callback: Some(enc_cb),
            };
            let mut enc = StreamingEncodeSession::create(params, MSG, PASS).expect("create");
            let mut annex_b = Vec::new();
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                let frame = YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                };
                enc.push_frame(frame, &mut annex_b).expect("push");
            }
            enc.finish(&mut annex_b).expect("finish encode");

            let got = enc_events.lock().unwrap().clone();
            // Sanity-check the shape of the encode event stream.
            assert_eq!(got.first(), Some(&EncodePhase::Setup), "first event must be Setup");
            assert_eq!(got.last(), Some(&EncodePhase::Done), "last event must be Done");
            assert!(
                got.iter().filter(|p| matches!(p, EncodePhase::Pass1Capture { .. })).count() as u32 == N,
                "expected {N} Pass1Capture events (one per push_frame), got {:?}", got
            );
            assert!(
                got.iter().filter(|p| matches!(p, EncodePhase::StcPlan { .. })).count() as u32 == N / GOP,
                "expected {} StcPlan events (one per GOP)", N / GOP
            );
            assert!(
                got.iter().filter(|p| matches!(p, EncodePhase::Pass2Replay { .. })).count() as u32 == N / GOP,
                "expected {} Pass2Replay events (one per GOP)", N / GOP
            );
            assert_eq!(
                got.iter().filter(|p| matches!(p, EncodePhase::Mux)).count(), 1,
                "Mux fires once at finish"
            );

            // Decode side.
            let dec_events: Arc<Mutex<Vec<DecodePhase>>> = Arc::new(Mutex::new(Vec::new()));
            let dec_clone = Arc::clone(&dec_events);
            let dec_cb: DecodeProgressCallback = Arc::new(move |phase| {
                dec_clone.lock().unwrap().push(phase);
            });
            let mut dec = StreamingDecodeSession::create(PASS)
                .expect("decode create")
                .with_progress_callback(dec_cb);
            dec.push_annex_b(&annex_b).expect("push annex");
            let result = dec.finish().expect("finish decode");
            assert_eq!(result.text, MSG);

            let dec_got = dec_events.lock().unwrap().clone();
            assert_eq!(dec_got.first(), Some(&DecodePhase::Demux), "first decode event must be Demux");
            assert_eq!(dec_got.last(), Some(&DecodePhase::Done), "last decode event must be Done");
            assert!(
                dec_got.iter().any(|p| matches!(p, DecodePhase::Walker { .. })),
                "decode must emit at least one Walker event"
            );
            assert_eq!(
                dec_got.iter().filter(|p| matches!(p, DecodePhase::StcExtract)).count(), 1,
                "StcExtract fires once"
            );
            assert_eq!(
                dec_got.iter().filter(|p| matches!(p, DecodePhase::Decrypt)).count(), 1,
                "Decrypt fires once"
            );
        }
    }
}
