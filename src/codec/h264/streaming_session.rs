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
//! - **D.0.7.3**: pure-Rust encode path — retired. The clean-room
//!   pure-Rust H.264 encoder was deleted in the 2026-06
//!   video-retirement; OH264 is the sole encode backend. A build
//!   without the `h264-encoder` feature falls into the
//!   `EngineDisabled` arm, which errors explicitly at push/finish.
//! - **D.0.7.11**: streaming decode session (shipped). See
//!   [`StreamingDecodeSession`] below.

use crate::stego::error::StegoError;

// #474.1 — progress event types. Streaming sessions accept an optional
// callback on creation; emission sites wired in #474.2.
use super::progress::{
    DecodePhase, DecodeProgressCallback, EncodePhase, EncodeProgressCallback,
};

// OH264 path imports — gated by the h264-encoder feature.
#[cfg(feature = "h264-encoder")]
use super::openh264_stego::{
    bytes_to_bits_msb_first_pub, consume_gop_emit, oh264_plain_encode_gop, produce_gop_cover,
    EncodeOpts, GopCoverProducts,
};
#[cfg(feature = "h264-encoder")]
use super::stego::tier_filter::{CascadeTier, DEFAULT_HEADROOM};
// chunk_frame helpers — needed by BOTH the OH264 encode path and (post
// #472.2) the pure-Rust encode path, so unconditional.
use super::stego::chunk_frame::{build_chunk_frame_v3_1, build_first_chunk_frame_v3_1};

// Decode-side imports (available within the h264-encoder gate that
// already wraps this module; decode doesn't need h264-encoder).
use super::cabac::bin_decoder::slice::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use super::stego::{combine_cover_4domain, CostWeights};
use super::stego::orchestrate::DomainCosts;
use super::stego::chunk_frame::{
    parse_chunk_frame_v3_1, parse_first_chunk_frame_v3_1,
    CHUNK_FRAME_FIRST_HEADER_LEN_V3_1, CHUNK_FRAME_NEXT_HEADER_LEN_V3_1,
};
use super::stego::hook::EmbedDomain;
use super::stego::keys::CabacStegoMasterKeys;
// #852 — whole-video SHADOW decode fallback. Decode-side / shared (not
// behind `h264-encoder`), so the App Clip + decode-only WASM get it too.
use super::stego::cascade_safety::{analyze_safe_mvd_subset, derive_msl_safe_from_msb};
use super::stego::shadow::{decode_shadow_from_priority_lsbs, for_each_eligible_position};
use crate::stego::stc::extract::{stc_extract, stc_extract_prefix};
use crate::stego::stc::hhat::generate_hhat;
use crate::stego::{crypto, frame, payload};

/// STC constraint length — mirrors `openh264_stego::STC_H`. Both
/// encode and decode sides MUST agree on this value.
const STC_H: usize = 4;

/// chunk_frame v3 §3.2 — sanity cap on the first-chunk `total_bytes`.
/// Rejects most wrong-passphrase decodes within GOP 0 (< 1 s on mobile);
/// the AEAD-final check catches any garbage that slips through. Same
/// value the AV1 v3 decoder uses.
const MAX_TOTAL_PAYLOAD_BYTES: u32 = 256 * 1024 * 1024;

/// Which H.264 encoder backend a streaming encode session uses.
///
/// The decode session is engine-agnostic: the pure-Rust CABAC
/// bin-walker decodes any Annex-B regardless of which encoder produced
/// it, so no parallel `DecodeEngineChoice` is needed. (Since OH264 is
/// now the sole encode backend, this is moot in practice, but the
/// walker was always producer-agnostic.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EncodeEngineChoice {
    /// Cisco OpenH264 fork (production default, v1.0). Statically linked
    /// via `openh264-sys`; only available when the `h264-encoder`
    /// Cargo feature is on.
    ///
    /// Since the pure-Rust encoder retirement this is the only encode
    /// engine. The enum is retained (single-variant) so the FFI bridge's
    /// engine selector and the session params keep a stable shape.
    Oh264 = 0,
}

impl EncodeEngineChoice {
    /// Parse from the u8 the FFI bridge receives. Always resolves to
    /// `Oh264`: the pure-Rust encoder was retired, so any value
    /// (including legacy `1`) maps to the sole OpenH264 backend, keeping
    /// production stable on input from older client versions.
    pub fn from_u8(_v: u8) -> Self {
        Self::Oh264
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
/// Phase G.2.P4 (#841) — outcome of
/// [`StreamingEncodeSession::plan_safe_balanced`]. Mirrors AV1's
/// `Av1PlanOutcome` (the balanced-allocation planner is codec-agnostic).
#[derive(Debug, Clone, PartialEq)]
pub enum H264PlanOutcome {
    /// Plan installed; `window` = number of leading stego GOPs (W).
    /// Caller may `push_frame` immediately; GOPs `[W, n_gops)` are
    /// emitted as a plain (no-stego) tail the decoder skips.
    Installed { window: u32 },
    /// Caller must probe these additional GOP indices, append their
    /// `(gop_idx, cap_bytes)` measurements to `samples`, and call again.
    NeedMoreSamples {
        positions: Vec<usize>,
        total_target: usize,
    },
    /// Even at full-corpus coverage the message doesn't fit — user error.
    MessageTooLarge,
}

pub struct StreamingEncodeSession {
    params: EncodeSessionParams,
    inner: SessionImpl,
}

enum SessionImpl {
    /// OH264 backend session (D.0.7.2). Holds the GOP frame buffer +
    /// per-GOP chunk schedule + STC seed.
    #[cfg(feature = "h264-encoder")]
    Oh264(Oh264SessionState),
    /// Requested engine isn't compiled in. push_frame / finish will
    /// error explicitly. Used when caller asks for OH264 but the
    /// crate was built without `h264-encoder`.
    EngineDisabled(EncodeEngineChoice),
}

#[cfg(feature = "h264-encoder")]
struct Oh264SessionState {
    /// CAP2.2 — the full inner stego frame (crypto frame bytes), allocated
    /// to GOPs DYNAMICALLY at drain (uniform-spread + carry-remainder)
    /// rather than pre-split, so one weak GOP no longer caps the clip
    /// (the even-split `n_gops × min` failure, e.g. woman_subway 0.16×).
    msg_frame: Vec<u8>,
    /// Bytes of `msg_frame` allocated to GOPs so far (allocation cursor).
    cursor: usize,
    /// Number of GOPs (= number of chunks) expected total. Computed at
    /// session_create from `total_frames_hint / gop_size`.
    total_chunks: u16,
    /// Index of the next GOP to emit (0-based, < total_chunks).
    chunk_idx: u16,
    /// In-flight GOP YUV buffer (tight I420 packing).
    gop_buffer: Vec<u8>,
    /// Frames currently buffered in `gop_buffer`.
    frames_buffered: u32,
    /// STC h-hat seed for CoeffSign domain. Derived as
    /// `per_gop_seeds(CoeffSignBypass, 0)` — the exact same derivation
    /// the decode session ([`StreamingDecodeSession::finish`]) and the
    /// OH264 stego primitive use, so encode and decode agree on the STC
    /// h-hat without carrying the seed on the wire.
    hhat_seed: [u8; 32],
    /// #798 — cascade tier the encoder resolved for the FIRST GOP
    /// (auto-tier is payload-driven; representative of the encode).
    /// `None` until the first GOP drains. Surfaced via
    /// [`StreamingEncodeSession::resolved_tier`] for the mobile
    /// success-screen badge.
    resolved_tier: Option<u8>,
    /// CAP2.2 §12 — optional precise per-GOP byte-allocation plan (from
    /// `allocate_chunks_proportional` run over a probe pass on re-readable
    /// input — the CLI two-pass path). `Some(plan)` ⇒ `drain_one_gop` sizes
    /// each GOP's chunk from the plan (carry-absorbing) instead of the online
    /// uniform-spread; `None` (default, the push-once mobile path) ⇒
    /// uniform-spread + carry. Either way each per-GOP embed is round-trip
    /// verified + shrink-carried, so a plan never risks silent data loss.
    gop_alloc_plan: Option<Vec<usize>>,
    /// CAP2.3 §4 — count W of leading STEGO GOPs when a concentrate+tail plan
    /// is installed (`gop_alloc_plan[W..]` are all 0 = the plain tail). `None`
    /// when no plan / uniform-spread is active (then every GOP is a stego GOP).
    /// Stego chunks carry `total_chunks = W` (< n_gops when a plan installs a
    /// plain tail), so the decoder learns the window from GOP 0 and skips the
    /// `[W, n_gops)` tail. With no plan (`None`) the wire value falls back to
    /// the GOP count (legacy even-split — every GOP carries a chunk).
    stego_window: Option<u16>,
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
        Self::create_with_files(params, message_text, &[], passphrase)
    }

    /// Like [`Self::create`] but the payload carries file attachments
    /// alongside (or instead of) the text. Files are part of the encrypted
    /// payload — the per-GOP stego channel is byte-identical to the
    /// text-only path; only `payload::encode_payload` differs. The decoder
    /// recovers them via `StreamingDecodeSession::finish().files`.
    pub fn create_with_files(
        params: EncodeSessionParams,
        message_text: &str,
        files: &[payload::FileEntry],
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
            #[cfg(feature = "h264-encoder")]
            EncodeEngineChoice::Oh264 => {
                SessionImpl::Oh264(build_oh264_state(&params, message_text, files, passphrase)?)
            }
            #[cfg(not(feature = "h264-encoder"))]
            EncodeEngineChoice::Oh264 => SessionImpl::EngineDisabled(EncodeEngineChoice::Oh264),
        };
        // #474.2 — emit Setup once after all initialisation succeeds.
        // Disabled engines never reach this path because build_*_state
        // already errored above; this event always corresponds to a
        // usable session.
        emit_encode(&params, EncodePhase::Setup);
        Ok(Self { params, inner })
    }

    /// Push one YUV frame. Emits Annex-B bytes for a closing GOP into
    /// `out` (appended) when the frame completes a GOP. The OH264 path
    /// drains per-GOP; finish() handles the tail.
    pub fn push_frame(
        &mut self,
        frame: YuvFrameRef<'_>,
        out: &mut Vec<u8>,
    ) -> Result<(), StegoError> {
        match &mut self.inner {
            #[cfg(feature = "h264-encoder")]
            SessionImpl::Oh264(state) => oh264_push_frame(&self.params, state, frame, out),
            SessionImpl::EngineDisabled(engine) => Err(StegoError::InvalidVideo(format!(
                "streaming encode engine {engine:?} not compiled in (rebuild with feature)"
            ))),
        }
    }

    /// #798 — the cascade tier the encoder resolved (auto-tier is
    /// payload-driven), as `CascadeTier::as_u8()` (0..=4). `None` until
    /// the first GOP has drained, and always `None` for the shadow /
    /// engine-disabled paths (only the OH264 primary path
    /// stashes it today). Mobile reads this after `finish()` to light
    /// up the success-screen quality badge — mobile has no ffmpeg to
    /// re-resolve the tier from YUV, so the encoder reports it directly.
    pub fn resolved_tier(&self) -> Option<u8> {
        match &self.inner {
            #[cfg(feature = "h264-encoder")]
            SessionImpl::Oh264(state) => state.resolved_tier,
            _ => None,
        }
    }

    /// CAP2.2 §12 — the framed (encrypted + headered) message length in
    /// bytes: the exact `Σ` that a [`set_gop_alloc_plan`](Self::set_gop_alloc_plan)
    /// plan must equal, and the `message_len` the CLI two-pass orchestrator
    /// feeds to
    /// [`allocate_chunks_proportional`](super::stego::chunk_frame::allocate_chunks_proportional)
    /// alongside the probe's per-GOP caps. `None` for non-OH264 backends.
    #[cfg(feature = "h264-encoder")]
    pub fn framed_message_len(&self) -> Option<usize> {
        match &self.inner {
            SessionImpl::Oh264(state) => Some(state.msg_frame.len()),
            _ => None,
        }
    }

    /// CAP2.2 §12 — install a precise per-GOP byte-allocation plan, overriding
    /// the default online uniform-spread used by `drain_one_gop`. `plan[i]` is
    /// GOP `i`'s target byte count; produce it with
    /// [`allocate_chunks_proportional`](super::stego::chunk_frame::allocate_chunks_proportional)
    /// over a probe pass (the CLI two-pass path). `drain_one_gop` still
    /// round-trip-verifies + shrink-carries each GOP, so an over-optimistic
    /// plan entry degrades to carry-forward, never silent data loss.
    ///
    /// Call after `create()` and before pushing frames. Only the OH264 backend
    /// honours a plan; the push-once mobile path leaves it `None`.
    ///
    /// # Errors
    /// * [`StegoError::InvalidVideo`] if `plan.len()` != the session's GOP
    ///   count, `Σ plan` != [`framed_message_len`](Self::framed_message_len)
    ///   (the plan must account for every framed byte exactly), or the backend
    ///   isn't OH264.
    #[cfg(feature = "h264-encoder")]
    pub fn set_gop_alloc_plan(&mut self, plan: Vec<usize>) -> Result<(), StegoError> {
        match &mut self.inner {
            SessionImpl::Oh264(state) => {
                if plan.len() != state.total_chunks as usize {
                    return Err(StegoError::InvalidVideo(format!(
                        "alloc plan length {} != GOP count {}",
                        plan.len(),
                        state.total_chunks
                    )));
                }
                let sigma: usize = plan.iter().sum();
                if sigma != state.msg_frame.len() {
                    return Err(StegoError::InvalidVideo(format!(
                        "alloc plan Σ {sigma} != framed message length {}",
                        state.msg_frame.len()
                    )));
                }
                state.gop_alloc_plan = Some(plan);
                Ok(())
            }
            _ => Err(StegoError::InvalidVideo(
                "set_gop_alloc_plan requires the OH264 backend".into(),
            )),
        }
    }

    /// CAP2.3 §4 — install a **concentrate+tail** allocation plan (name kept
    /// `plan_proportional` for FFI stability; supersedes #804's uniform
    /// proportional spread, whose `r_target` was inert). Fills each GOP
    /// sequentially to `⌊r_target·cap⌋`, leaving a plain tail — so `r_target`
    /// genuinely bounds per-GOP flip density (a lower value spreads the message
    /// across more leading GOPs at lower density). Runs
    /// `allocate_chunks_concentrate_tail` over the probe's per-GOP caps + this
    /// session's framed length, installs the plan via
    /// [`set_gop_alloc_plan`](Self::set_gop_alloc_plan), and records the stego
    /// window `W`: stego chunks carry `total_chunks = W` (< n_gops) and the
    /// `[W, n_gops)` tail is emitted plain (no stego), which the decoder skips.
    ///
    /// `gop_caps` is the per-GOP tier-0 byte vector from
    /// [`StreamingProbeSession::finish_with_per_gop`] (same content + GOP
    /// shape). `r_target` is the rate ceiling: `CAP2_DEFAULT_R_TARGET` (0.5) is
    /// the stealth-leaning default; `1.0` is greedy-concentrate (max per-GOP
    /// rate). Calibration is #806.
    ///
    /// Returns `Ok(true)` if a plan was installed, `Ok(false)` if it falls back
    /// to the default uniform-spread — when the message exceeds Σ caps (the
    /// encode then surfaces `MessageTooLarge`), the backend isn't OH264, or
    /// `gop_caps` doesn't match the session's GOP shape. Falling back is always
    /// safe: uniform-spread + the per-GOP verify/carry never loses data.
    #[cfg(feature = "h264-encoder")]
    pub fn plan_proportional(
        &mut self,
        gop_caps: &[usize],
        r_target: f64,
    ) -> Result<bool, StegoError> {
        let Some(framed) = self.framed_message_len() else {
            return Ok(false); // not OH264 — nothing to plan
        };
        // CAP2.3 §4 — concentrate+tail: fill each GOP sequentially to
        // ⌊r_target·cap⌋, leaving a plain tail. `window` = count of leading
        // stego GOPs. This supersedes #804's uniform-proportional spread,
        // whose r_target cancelled in the proportions and was inert; here
        // r_target genuinely bounds per-GOP rate.
        let Some((plan, window)) =
            super::stego::chunk_frame::allocate_chunks_concentrate_tail(gop_caps, framed, r_target)
        else {
            return Ok(false); // message > Σ caps — leave uniform (→ MessageTooLarge)
        };
        // `set_gop_alloc_plan` re-validates len == GOP count and Σ == framed
        // (both still hold: the plan spans every GOP, tail entries are 0). A
        // probe/encode GOP-count disagreement degrades to uniform (safe).
        if self.set_gop_alloc_plan(plan).is_err() {
            return Ok(false);
        }
        // Record the stego window. The drain reads this as the wire
        // `total_chunks` (= W < n_gops) and emits a plain (no-stego) tail for
        // GOPs [W, n_gops); the decoder learns W from GOP 0 and skips the tail.
        if let SessionImpl::Oh264(state) = &mut self.inner {
            state.stego_window = Some(window as u16);
        }
        Ok(true)
    }

    /// Phase G.2.P4 (#841) — caller-driven balanced safe planner: the
    /// H.264 mirror of `Av1StreamingEncodeSession::plan_safe_balanced`.
    ///
    /// One iteration of the codec-agnostic balanced allocator
    /// ([`crate::stego::balanced_allocation::plan_safe_balanced`]).
    /// `samples` are the `(gop_idx, cap_bytes)` pairs measured so far,
    /// `gop_size` the frames-per-GOP, and `calibration` the locked
    /// constants — use [`AllocationCalibration::DEFAULT`]
    /// (`AV1_1080P_QP30`) as the v0.8 placeholder until an H.264-specific
    /// `H264_*` calibration is derived. Returns:
    ///
    /// - [`H264PlanOutcome::Installed`] `{ window }` — plan installed;
    ///   `push_frame` immediately.
    /// - [`H264PlanOutcome::NeedMoreSamples`] — probe the requested GOP
    ///   indices (via a `StreamingProbeSession`), extend `samples`, retry.
    /// - [`H264PlanOutcome::MessageTooLarge`] — doesn't fit at full corpus.
    ///
    /// Must be called after `create()` and before the first `push_frame`
    /// (same contract as [`set_gop_alloc_plan`](Self::set_gop_alloc_plan)).
    /// OH264 backend only; other backends return `Err`.
    #[cfg(feature = "h264-encoder")]
    pub fn plan_safe_balanced(
        &mut self,
        samples: &[(usize, usize)],
        gop_size: usize,
        calibration: &crate::stego::calibration::AllocationCalibration,
    ) -> Result<H264PlanOutcome, StegoError> {
        let Some(framed_len) = self.framed_message_len() else {
            return Err(StegoError::InvalidVideo(
                "plan_safe_balanced requires the OH264 streaming backend".into(),
            ));
        };
        // n_gops = the session's GOP count (set at create from
        // total_frames_hint / gop_size); the same value `set_gop_alloc_plan`
        // validates `plan.len()` against.
        let n_gops = match &self.inner {
            SessionImpl::Oh264(state) => state.total_chunks as usize,
            _ => {
                return Err(StegoError::InvalidVideo(
                    "plan_safe_balanced requires the OH264 streaming backend".into(),
                ))
            }
        };
        use crate::stego::balanced_allocation::{plan_safe_balanced as core_plan, PlanOutcome};
        match core_plan(samples, n_gops, framed_len, gop_size, calibration) {
            PlanOutcome::Plan { plan, window } => {
                if window == 0 {
                    return Ok(H264PlanOutcome::Installed { window: 0 });
                }
                // The core plan is full-length (n_gops, trailing zeros) — exactly
                // what `set_gop_alloc_plan` wants (Σ == framed). Then record the
                // stego window so the drain emits a plain tail for GOPs
                // [W, n_gops) — identical install pattern to `plan_proportional`.
                self.set_gop_alloc_plan(plan)?;
                if let SessionImpl::Oh264(state) = &mut self.inner {
                    state.stego_window = Some(window as u16);
                }
                Ok(H264PlanOutcome::Installed { window: window as u32 })
            }
            PlanOutcome::NeedMoreSamples {
                positions,
                total_target,
            } => Ok(H264PlanOutcome::NeedMoreSamples {
                positions,
                total_target,
            }),
            PlanOutcome::MessageTooLarge => Ok(H264PlanOutcome::MessageTooLarge),
        }
    }

    /// Finish the session. For OH264: drains the final partial GOP.
    /// For OH264Shadow: runs the n-shadow orchestrator on the
    /// buffered clip. Errors if no frames were pushed or (OH264 only)
    /// fewer than `total_chunks` chunks emitted.
    pub fn finish(self, out: &mut Vec<u8>) -> Result<(), StegoError> {
        match self.inner {
            #[cfg(feature = "h264-encoder")]
            SessionImpl::Oh264(mut state) => oh264_finish(&self.params, &mut state, out),
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

#[cfg(feature = "h264-encoder")]
fn build_oh264_state(
    params: &EncodeSessionParams,
    message_text: &str,
    files: &[payload::FileEntry],
    passphrase: &str,
) -> Result<Oh264SessionState, StegoError> {
    // Build the inner stego frame ONCE up-front; split bytes across GOPs.
    // File attachments ride the same payload as the text — the per-GOP
    // stego channel is payload-agnostic (chunk_frame wraps opaque payload
    // bytes), so text-only and text+files take an identical wire path.
    let payload_bytes = payload::encode_payload(message_text, files)?;
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

    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let hhat_seed = keys
        .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
        .hhat_seed;

    let frame_bytes_size = (params.width as usize) * (params.height as usize) * 3 / 2;
    let gop_buffer = Vec::with_capacity(frame_bytes_size * params.gop_size as usize);

    Ok(Oh264SessionState {
        msg_frame: frame_bytes,
        cursor: 0,
        total_chunks,
        chunk_idx: 0,
        gop_buffer,
        frames_buffered: 0,
        hhat_seed,
        resolved_tier: None,
        gop_alloc_plan: None,
        stego_window: None,
    })
}

#[cfg(feature = "h264-encoder")]
fn oh264_push_frame(
    params: &EncodeSessionParams,
    state: &mut Oh264SessionState,
    frame: YuvFrameRef<'_>,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    // `total_frames_hint` (→ `total_chunks` GOPs) is an ESTIMATE. The mobile
    // bridges derive it from duration×fps and stream frames one at a time
    // (AVFoundation / MediaCodec), so the true decoded count is only known
    // once the stream ends — and it routinely differs by a frame or two
    // (e.g. an Artlist clip with a leading empty edit, #799). When MORE
    // frames arrive than the hint predicted, the surplus GOPs fall OUTSIDE
    // the stego window (`stego_window` ≤ `total_chunks` ≤ `chunk_idx` here),
    // so `drain_one_gop` encodes them as a PLAIN TAIL: the message is already
    // fully embedded in the leading window, and the decoder's
    // read-until-`total_bytes` loop ignores the extra slabs. So we no longer
    // hard-fail on overflow — we only guard the u16 GOP counter from wrapping
    // (a pathological multi-million-frame clip).
    if state.chunk_idx == u16::MAX {
        return Err(StegoError::InvalidVideo(
            "frame count exceeds the u16 GOP-index range".into(),
        ));
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
    // Clamp to the hint so a frame-count overshoot (actual > estimate) can't
    // drive the progress bar past 100% — same guard the Pass2Replay emit uses.
    let cumulative =
        (state.chunk_idx as u32 * params.gop_size + state.frames_buffered).min(params.total_frames_hint);
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

#[cfg(feature = "h264-encoder")]
fn oh264_finish(
    params: &EncodeSessionParams,
    state: &mut Oh264SessionState,
    out: &mut Vec<u8>,
) -> Result<(), StegoError> {
    if state.frames_buffered > 0 {
        drain_one_gop(params, state, out)?;
    }
    // `total_chunks` (the hint's GOP count) is NOT a finish gate. The hint is
    // an estimate (see `oh264_push_frame`); the decoder reconstructs the
    // message from GOP 0's declared `total_bytes` over however many GOPs the
    // real stream contains, so the only ship invariant is payload
    // completeness. When the decoder yields FEWER GOPs than predicted, the
    // encode still succeeds as long as the (front-loaded) stego window fit
    // inside the GOPs that materialised — the unrealised GOPs were a plain
    // tail anyway. When it yields more, the surplus encoded plain. Either way
    // the test below is the real gate.
    //
    // CAP2.2 — the whole payload must have been embedded. With a concentrate
    // plan (the mobile path) this holds whenever the stego window fit; with
    // the bare uniform-spread fallback it can trip if the real GOP count fell
    // short of the hint — a clean, fail-closed MessageTooLarge (never silent
    // corruption), and the user can retry.
    if state.cursor != state.msg_frame.len() {
        return Err(StegoError::MessageTooLarge);
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

#[cfg(feature = "h264-encoder")]
/// CAP2.2 §14 — embed one chunk into a single GOP, **round-trip-verified**.
/// Tries `payload` (up to its full length); on encode-fail OR verify-fail
/// (the J2 silent-loss zone — encodes but decodes wrong) it shrinks `want`
/// ~12%/step and retries, verifying with the SAME extract the real decoder
/// uses (`verify_chunk_v3`) on the GOP's self-contained slab. An
/// empty chunk always round-trips, so the loop terminates. Returns the GOP
/// Annex-B, the resolved cascade tier, and `consumed` ≤ `payload.len()`
/// (the caller carries the rest forward). Correct-by-construction: never
/// emits a chunk that decodes wrong.
///
/// Shared by the streaming encoder (`drain_one_gop`) and the round-trip-aware
/// capacity estimator (#811), so both define "what fits" identically.
#[cfg(feature = "h264-encoder")]
pub(crate) fn embed_gop_roundtrip_safe(
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    frames_in_gop: u32,
    opts: super::openh264_stego::EncodeOpts,
    payload: &[u8],
    chunk_idx: u16,
    total_bytes: u32,
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
) -> Result<(Vec<u8>, CascadeTier, usize), StegoError> {
    // B-full.6b.1/.2 (#894): the payload-INDEPENDENT clean cover (Pass-1 + walk +
    // content_costs + safe_msl ∞-gate) is produced ONCE (`produce_gop_cover`), then
    // the payload-DEPENDENT round-trip shrink-carry emit loop runs
    // (`consume_gop_roundtrip_safe`). Byte-identical to the old per-shrink full
    // 2-pass encode — gated by the non-shadow streaming round-trip tests.
    let products = produce_gop_cover(gop_yuv, width, height, frames_in_gop, opts)?;
    consume_gop_roundtrip_safe(
        &products, gop_yuv, width, height, frames_in_gop, opts, payload, chunk_idx,
        total_bytes, hhat_seed, weights,
    )
}

/// B-full.6b.2 (#894) — the payload-DEPENDENT half of [`embed_gop_roundtrip_safe`]:
/// the round-trip-verified shrink-carry emit loop over a pre-produced
/// [`GopCoverProducts`]. Tries `payload` (up to its full length); on encode-fail OR
/// verify-fail (the J2 silent-loss zone) it shrinks `want` ~12%/step and retries,
/// verifying with the SAME extract the real decoder uses (`verify_chunk_v3`). An
/// empty chunk always round-trips, so the loop terminates.
#[cfg(feature = "h264-encoder")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn consume_gop_roundtrip_safe(
    products: &GopCoverProducts,
    gop_yuv: &[u8],
    width: u32,
    height: u32,
    frames_in_gop: u32,
    opts: super::openh264_stego::EncodeOpts,
    payload: &[u8],
    chunk_idx: u16,
    total_bytes: u32,
    hhat_seed: &[u8; 32],
    weights: &CostWeights,
) -> Result<(Vec<u8>, CascadeTier, usize), StegoError> {
    let mut want = payload.len();
    loop {
        let chunk = &payload[..want];
        // chunk_frame v3.1 §4: GOP 0 carries u32 total_bytes + u32 m_total;
        // GOP ≥ 1 carries u32 m_total + payload_len (#888).
        let framed = if chunk_idx == 0 {
            build_first_chunk_frame_v3_1(total_bytes, chunk)?
        } else {
            build_chunk_frame_v3_1(chunk)?
        };
        let frame_bits = bytes_to_bits_msb_first_pub(&framed);
        match consume_gop_emit(
            products, gop_yuv, width, height, frames_in_gop, opts, &frame_bits, hhat_seed,
            weights, CascadeTier::Auto, DEFAULT_HEADROOM,
        ) {
            // Empty chunk round-trips trivially; non-empty must decode back to
            // exactly the bytes embedded.
            Ok((b, t))
                if want == 0
                    || verify_chunk_v3(&b, hhat_seed, chunk_idx, total_bytes, chunk, weights) =>
            {
                break Ok((b, t, want));
            }
            Ok(_) | Err(StegoError::MessageTooLarge) => {
                debug_assert!(want > 0, "empty chunk must always succeed");
                want -= 1usize.max(want / 8); // ~12% step, reaches 0
            }
            Err(e) => break Err(e),
        }
    }
}

/// CAP2.2 §12 / CAP2.3 §4 — how many payload bytes the stego GOP `chunk_idx`
/// targets, given the optional precise per-GOP plan (carry-absorbing) or the
/// online uniform-spread fallback. `window` = the count of leading stego GOPs
/// (`stego_window` or `total_chunks`); the caller guarantees `chunk_idx < window`.
/// Used by the sequential `drain_one_gop` path (B-full.6b.2, #894).
#[cfg(feature = "h264-encoder")]
fn gop_want(
    gop_alloc_plan: Option<&[usize]>,
    window: usize,
    chunk_idx: usize,
    cursor: usize,
    msg_len: usize,
) -> usize {
    let gops_remaining = window - chunk_idx; // ≥ 1, incl. this GOP
    let remaining = msg_len - cursor;
    match gop_alloc_plan {
        // CAP2.2 §12 — precise plan (probe two-pass). This GOP's target is its
        // cumulative planned bytes minus what's already consumed, so a shortfall
        // carried from a shrunk earlier GOP rolls into this one.
        Some(plan) => {
            let planned_through_now: usize = plan.iter().take(chunk_idx + 1).sum();
            planned_through_now.saturating_sub(cursor).min(remaining)
        }
        // Fast path (§13): online uniform-spread + carry-remainder.
        None => {
            if gops_remaining <= 1 {
                remaining
            } else {
                remaining.div_ceil(gops_remaining)
            }
        }
    }
}

#[cfg(feature = "h264-encoder")]
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
    // at the natural boundary. (Fired for plain tail GOPs too, so the
    // progress smoothing sees a consistent StcPlan→Pass2Replay per GOP.)
    let total_gops = state.total_chunks as u32;
    emit_encode(
        params,
        EncodePhase::StcPlan {
            gop_idx: state.chunk_idx as u32,
            total_gops,
        },
    );

    // CAP2.3 §4 — the stego window W: GOPs [0, W) carry chunks; [W, n_gops)
    // are a PLAIN tail (no stego). With no plan installed `stego_window` is
    // None → W = n_gops, so every GOP is a stego GOP (no-plan even-split
    // fallback).
    let window = state.stego_window.unwrap_or(state.total_chunks);
    let opts = EncodeOpts {
        qp: params.qp,
        intra_period: params.gop_size as i32,
    };

    if state.chunk_idx < window {
        // ── STEGO GOP ───────────────────────────────────────────────────
        // This GOP takes its planned/uniform share of the remaining payload;
        // a GOP that cannot hold its share embeds less and carries the rest
        // forward (absorbed by later GOPs' headroom), so one weak GOP no
        // longer caps the clip.
        let want = gop_want(
            state.gop_alloc_plan.as_deref(),
            window as usize,
            state.chunk_idx as usize,
            state.cursor,
            state.msg_frame.len(),
        );

        // CAP2.2 §14 — embed this GOP's share, ROUND-TRIP-VERIFIED with
        // shrink-carry. CAP2.3: the wire `total_chunks` carried by every stego
        // chunk is the WINDOW `window` (< n_gops when there is a plain tail),
        // so the decoder learns W from GOP 0 and never walks the tail.
        // `state.total_chunks` (the GOP count) is untouched — it remains the
        // bookkeeping basis for the finish assertion. #798 — Auto tier captures
        // the resolved tier for the mobile success badge.
        let (bitstream, gop_tier, consumed) = embed_gop_roundtrip_safe(
            &state.gop_buffer,
            params.width,
            params.height,
            frames_in_gop,
            opts,
            &state.msg_frame[state.cursor..state.cursor + want],
            state.chunk_idx,
            // chunk_frame v3: GOP 0's first-chunk header carries the full
            // ciphertext length; the decoder reads chunks until it has
            // accumulated this many bytes (W is implicit, not on the wire).
            state.msg_frame.len() as u32,
            &state.hhat_seed,
            &params.cost_weights,
        )?;
        state.cursor += consumed;
        if state.resolved_tier.is_none() {
            state.resolved_tier = Some(gop_tier.as_u8());
        }
        out.extend_from_slice(&bitstream);
    } else {
        // ── PLAIN TAIL GOP (CAP2.3 §6) ──────────────────────────────────
        // The message was fully embedded in GOPs [0, W); this GOP carries NO
        // stego. A single clean `encode_once` (no Pass-1 walk / STC / content
        // costs) — the perf win. The decoder's early-dropout never touches
        // these slabs. The reset inside `oh264_plain_encode_gop` clears the
        // fork state the preceding stego GOPs left (the #548 class).
        let bitstream = oh264_plain_encode_gop(
            &state.gop_buffer,
            params.width,
            params.height,
            frames_in_gop,
            opts,
        )?;
        out.extend_from_slice(&bitstream);
    }

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

// Used by the OH264 shadow buffer-on-finish path — strips row-padding
// from the input strides into a tight I420 layout the encoder expects.
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
// but runs each GOP through an accurate per-GOP STC trial
// (`openh264_stego::h264_gop_capacity`, CAP2.1), the same
// primitive the one-shot reporter uses and the same boundary the real
// encode hits before `MessageTooLarge`. It returns `CapacityProbeResult`
// with the per-tier per-GOP payload caps, the GOP count, and the
// shadow-pool cover count.
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
// **Primary payload math** (CAP2.2 §12 / #824):
//   primary_bytes = per_tier_sum_payload[tier] − FRAME_OVERHEAD
//     The probe sums each GOP's own per-tier STC payload cap
//     (`h264_gop_capacity` already frames the chunk_frame
//     header, so the per-GOP cap is post-header). The carry/proportional
//     encoders fill each GOP to its OWN cap, so the message ceiling is
//     Σ per-GOP caps, not the even-split `n_gops × min`. Only the one-time
//     crypto-frame envelope (`FRAME_OVERHEAD`) is subtracted. This is an
//     UPPER bound — the per-GOP STC cap over-reports ~2% (J2 jitter),
//     absorbed by the encoder's per-GOP round-trip verify + shrink-carry
//     (graceful `MessageTooLarge`, never data loss). The retired
//     `cover_bits × 0.40` heuristic is gone.
//   `CapacityProbeResult::cover_bits` (the CoeffSign cover count) is now
//   retained ONLY for the #475 progress estimate and the shadow-pool
//   formula below — it no longer drives the primary capacity number.
//
// **Shadow math** is the existing collision-limited formula from
// `h264_stego_shadow_capacity`, working off the shadow-domain cover
// pool. The shadow video encode is the per-GOP streaming pull path
// (`h264_encode_with_shadows_streaming`, driven by a `GopYuvSource`); the
// probe's `CapacityProbeResult::shadow_max_message_bytes` reports their budget.


/// Result of a `StreamingProbeSession::finish` — raw cover bits, GOP
/// count, and conservative payload budgets for primary + shadow modes.
#[derive(Debug, Clone, Copy)]
pub struct CapacityProbeResult {
    /// Total CoeffSign cover bits walked back from the baseline
    /// per-GOP encode. Sum across all GOPs. Kept for the #475 progress
    /// callback; the PRIMARY capacity now uses the per-tier STC-trial
    /// budget below (CAP2.1).
    pub cover_bits: usize,
    /// #809 — total injectable bits across the 3 shadow domains
    /// (CoeffSign + CoeffSuffixLsb + MvdSign), summed across GOPs. The
    /// shadow-capacity formula reads THIS (not `cover_bits`, which is
    /// CoeffSign-only) so the streaming HUD matches the one-shot
    /// `h264_video_capacity` shadow number.
    pub shadow_pool_bits: usize,
    /// Number of GOPs emitted (= number of chunk_frame headers in the
    /// real encode).
    pub n_gops: u32,
    /// CAP2.2 §12 (#824) — SUM of per-GOP max-payload (bytes) across all GOPs,
    /// per cascade tier (0..=4). The carry-based encoders (uniform-share + carry
    /// AND proportional, #804) fill each GOP to its OWN cap, so the true message
    /// ceiling is this Σ, not `n_gops × min`. `primary_max_message_bytes` reports
    /// `per_tier_sum_payload[tier] − FRAME_OVERHEAD`. UPPER bound: the per-GOP
    /// STC cap over-reports ~2% (J2 jitter), absorbed by the encoder's per-GOP
    /// round-trip verify + shrink-carry (graceful `MessageTooLarge`, never data
    /// loss) and the #807 upper-bound HUD framing.
    pub per_tier_sum_payload: [usize; 5],
}

impl CapacityProbeResult {
    /// Maximum primary-message bytes (UTF-8 text + attached files
    /// combined, after envelope) that the streaming OH264 stego encode
    /// session can accept at this content + GOP shape.
    ///
    /// CAP2.2 §12 (#824) — Σ per-GOP capacity. The carry-based encoders
    /// (uniform-share + carry AND proportional, #804) fill each GOP to its OWN
    /// cap, so the ceiling is `Σ per_gop_cap`, not the even-split `n_gops × min`
    /// (which under-reported on heterogeneous content). Per-GOP payload is
    /// already post-chunk-header (the STC trial frames the chunk), so we subtract
    /// only the one-time crypto-frame envelope: `per_tier_sum_payload[0] −
    /// FRAME_OVERHEAD`. Saturates at 0. UPPER bound (per-GOP STC cap over-reports
    /// ~2% J2; the encoder's verify + shrink-carry degrades gracefully to
    /// `MessageTooLarge`, never data loss).
    pub fn primary_max_message_bytes(&self) -> usize {
        self.primary_max_message_bytes_at_tier(0)
    }

    /// Per-tier variant of [`Self::primary_max_message_bytes`] (tier 0..=4).
    pub fn primary_max_message_bytes_at_tier(&self, tier: usize) -> usize {
        let sum_payload = self.per_tier_sum_payload.get(tier).copied().unwrap_or(0);
        sum_payload.saturating_sub(frame::FRAME_OVERHEAD)
    }

    /// Maximum per-shadow message bytes under `n_shadows` total
    /// shadows. Mirrors the collision-limited formula in
    /// `h264_stego_shadow_capacity` (image-side path) so callers get
    /// identical numbers across the encode/decode surfaces.
    ///
    /// Returns 0 for `n_shadows == 0` (no shadow encoding needed).
    /// For `n_shadows == 1`, no inter-shadow collisions; capacity is
    /// the raw cover budget minus parity + per-shadow envelope.
    pub fn shadow_max_message_bytes(&self, n_shadows: u32) -> usize {
        // #809 — single source of truth for the collision-limited shadow
        // formula, shared with the one-shot OH264 path. Uses the 3-domain
        // injectable pool (`shadow_pool_bits`), NOT CoeffSign-only
        // (`cover_bits`) — the streaming HUD matches the one-shot number.
        crate::codec::h264::stego::shadow_capacity::shadow_max_message_bytes_from_cover_bits(
            self.shadow_pool_bits,
            n_shadows as usize,
        )
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
    /// #809 — total injectable bits across the 3 shadow domains
    /// (CoeffSign + CoeffSuffixLsb + MvdSign), accumulated per GOP. This
    /// is the shadow-pool basis, distinct from `cover_bits` (CoeffSign-only,
    /// kept for the #475 progress estimate). Unifies the streaming HUD's
    /// shadow number with the one-shot `h264_video_capacity`.
    shadow_pool_bits: usize,
    n_gops: u32,
    /// CAP2.2 §12 — per-GOP tier-0 max-payload (bytes), pushed as each GOP
    /// drains. The full per-GOP VECTOR `allocate_chunks_proportional` needs
    /// to size a proportional plan. Surfaced via `finish_with_per_gop` only —
    /// kept off the `Copy` `CapacityProbeResult`.
    per_gop_tier0_payload: Vec<usize>,
    /// CAP2.2 §12 (#824) — running SUM of per-GOP max-payload per tier (the
    /// carry/proportional message ceiling Σ). Feeds
    /// `CapacityProbeResult::primary_max_message_bytes`.
    per_tier_sum_payload: [usize; 5],
    /// #475 — fires once per GOP completion so the caller can render
    /// a refining estimated-capacity in the HUD. `None` when no
    /// progressive estimate is wanted (e.g. CLI batch use).
    progress_callback: Option<super::progress::CapacityProbeCallback>,
}

enum ProbeImpl {
    #[cfg(feature = "h264-encoder")]
    Oh264(Oh264ProbeState),
    /// Engine requested but compiled out.
    EngineDisabled(EncodeEngineChoice),
}

#[cfg(feature = "h264-encoder")]
struct Oh264ProbeState {
    /// Tight I420 GOP buffer (the in-progress GOP).
    gop_buffer: Vec<u8>,
    /// Frames buffered in `gop_buffer`.
    frames_buffered: u32,
    /// Completed GOP `(gop_n_frames, yuv)` awaiting the per-GOP probe drain.
    /// Holds at most one GOP — each completed GOP is probed (and the buffer
    /// freed) immediately in `push_frame`; the final partial GOP is pushed here
    /// in `finish`.
    pending: Vec<(u32, Vec<u8>)>,
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
                #[cfg(feature = "h264-encoder")]
                {
                    let frame_bytes =
                        (params.width as usize) * (params.height as usize) * 3 / 2;
                    let gop_capacity = frame_bytes * (params.gop_size as usize);
                    ProbeImpl::Oh264(Oh264ProbeState {
                        gop_buffer: Vec::with_capacity(gop_capacity),
                        frames_buffered: 0,
                        pending: Vec::new(),
                    })
                }
                #[cfg(not(feature = "h264-encoder"))]
                ProbeImpl::EngineDisabled(EncodeEngineChoice::Oh264)
            }
        };

        Ok(Self {
            params,
            inner,
            cover_bits: 0,
            shadow_pool_bits: 0,
            n_gops: 0,
            per_gop_tier0_payload: Vec::new(),
            per_tier_sum_payload: [0; 5],
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

    /// Emit one progress event. Called from `push_frame`/`finish`
    /// after each GOP drain. No-op when no callback is installed.
    ///
    /// CAP2.5/#821 + CAP2.2 §12 (#824) — the 3rd callback arg is the running
    /// ACCURATE primary estimate in BYTES, NOT raw cover_bits. It extrapolates
    /// the running Σ per-GOP STC budget (tier 0) to all GOPs:
    /// `(Σ_probed / gops_probed) × total_gops − FRAME_OVERHEAD` (the carry/
    /// proportional encoders fill each GOP to its OWN cap, so the ceiling is Σ).
    /// It replaced the old `cover_bits × 0.40` heuristic that over-reported
    /// ~200× during the probe then snapped to the finish() value; at finish
    /// (all GOPs probed) this lands exactly on `primary_max_message_bytes`, so
    /// the HUD refines toward — and converges on — the final number. Callers
    /// must use this value DIRECTLY (no ×0.40, no re-extrapolation).
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
            // CAP2.2 §12 (#824) — extrapolate the running Σ per-GOP capacity to
            // all GOPs: `(Σ_probed / gops_probed) × total_gops`. The carry/
            // proportional encoders fill each GOP to its own cap, so the ceiling
            // is Σ, not `n_gops × min`. At finish (all GOPs probed) this lands
            // exactly on the final `primary_max_message_bytes`, so the live HUD
            // refines toward — and converges on — the final number (no jump).
            let sum0 = self.per_tier_sum_payload[0];
            let probed = self.n_gops.max(1) as u128;
            let est_total = (sum0 as u128).saturating_mul(total_gops as u128) / probed;
            let est_bytes = (est_total as usize).saturating_sub(frame::FRAME_OVERHEAD);
            cb(self.n_gops, total_gops, est_bytes);
        }
    }

    /// Push one YUV frame. When the in-flight GOP fills, runs the
    /// per-GOP probe (baseline OH264 encode + walker) and accumulates
    /// the cover-bits count.
    pub fn push_frame(&mut self, frame: YuvFrameRef<'_>) -> Result<(), StegoError> {
        match &mut self.inner {
            #[cfg(feature = "h264-encoder")]
            ProbeImpl::Oh264(state) => {
                pack_frame_into_buffer(self.params.width, self.params.height, &frame, &mut state.gop_buffer)?;
                state.frames_buffered += 1;
                if state.frames_buffered == self.params.gop_size {
                    // Sequential per-GOP probe: drain this completed GOP immediately.
                    // (The parallel-GOP batch was reverted — oversubscription made it
                    // a net slowdown.)
                    let gop_n = state.frames_buffered;
                    state.pending.push((gop_n, std::mem::take(&mut state.gop_buffer)));
                    state.frames_buffered = 0;
                    drain_pending_gops_probe(&self.params, &mut state.pending, &mut self.cover_bits, &mut self.shadow_pool_bits, &mut self.n_gops, &mut self.per_gop_tier0_payload, &mut self.per_tier_sum_payload)?;
                    self.emit_estimate();
                }
                Ok(())
            }
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
    pub fn finish(self) -> Result<CapacityProbeResult, StegoError> {
        self.finish_with_per_gop().map(|(result, _per_gop)| result)
    }

    /// CAP2.2 §12 — like [`finish`](Self::finish) but ALSO returns the per-GOP
    /// tier-0 max-payload VECTOR (bytes), in GOP order (length == `n_gops`).
    /// This is the basis the CLI two-pass orchestrator feeds to
    /// [`allocate_chunks_proportional`](super::stego::chunk_frame::allocate_chunks_proportional)
    /// via [`StreamingEncodeSession::plan_proportional`]. `CapacityProbeResult`
    /// itself stays `Copy` and exposes the per-GOP Σ.
    pub fn finish_with_per_gop(
        mut self,
    ) -> Result<(CapacityProbeResult, Vec<usize>), StegoError> {
        match &mut self.inner {
            #[cfg(feature = "h264-encoder")]
            ProbeImpl::Oh264(state) => {
                if state.frames_buffered > 0 {
                    let gop_n = state.frames_buffered;
                    state.pending.push((gop_n, std::mem::take(&mut state.gop_buffer)));
                    state.frames_buffered = 0;
                }
                if !state.pending.is_empty() {
                    // Drain ALL remaining GOPs as the final parallel batch.
                    drain_pending_gops_probe(&self.params, &mut state.pending, &mut self.cover_bits, &mut self.shadow_pool_bits, &mut self.n_gops, &mut self.per_gop_tier0_payload, &mut self.per_tier_sum_payload)?;
                    // Emit one final per-GOP event before returning so
                    // listeners that race a final-state read can see
                    // the full count rather than the second-last one.
                    self.emit_estimate();
                }
                let result = CapacityProbeResult {
                    cover_bits: self.cover_bits,
                    shadow_pool_bits: self.shadow_pool_bits,
                    n_gops: self.n_gops,
                    per_tier_sum_payload: self.per_tier_sum_payload,
                };
                let per_gop = std::mem::take(&mut self.per_gop_tier0_payload);
                Ok((result, per_gop))
            }
            ProbeImpl::EngineDisabled(c) => Err(StegoError::InvalidVideo(format!(
                "engine {:?} is not compiled into this build",
                c
            ))),
        }
    }
}

/// Sequential per-GOP probe drain. Each completed GOP in `pending` is probed
/// independently (`h264_gop_capacity`) and the results accumulate in GOP order,
/// then `pending` is emptied. (Previously batched + run in parallel under the
/// pressure controller; reverted to a sequential map since the parallel-GOP
/// encode was a net slowdown from oversubscription.)
///
/// CAP2.1 — accurate per-GOP STC-trial capacity (same primitive the one-shot
/// reporter uses, the same boundary the real encode hits before MessageTooLarge);
/// `coeff_sign_cover_bits` feeds the #475 progress callback + shadow formula.
///
/// `h264-encoder`-gated: the per-GOP capacity probe calls
/// `openh264_stego::h264_gop_capacity` (the OpenH264 encoder). Both callers
/// (StreamingProbeSession::push_frame) are already in `h264-encoder` blocks;
/// this gate keeps the decoder-only build (`video,h264-decoder`) clean — it
/// was latently ungated and only surfaced in the phasmcore sync's decoder gate.
#[cfg(feature = "h264-encoder")]
fn drain_pending_gops_probe(
    params: &EncodeSessionParams,
    pending: &mut Vec<(u32, Vec<u8>)>,
    cover_bits: &mut usize,
    shadow_pool_bits: &mut usize,
    n_gops: &mut u32,
    per_gop_tier0: &mut Vec<usize>,
    per_tier_sum: &mut [usize; 5],
) -> Result<(), StegoError> {
    if pending.is_empty() {
        return Ok(());
    }
    let opts = super::openh264_stego::EncodeOpts {
        qp: params.qp,
        intra_period: params.gop_size as i32,
    };
    let caps: Vec<_> = pending
        .iter()
        .map(|gop| {
            let (gop_n, buf) = gop;
            super::openh264_stego::h264_gop_capacity(
                buf,
                params.width,
                params.height,
                *gop_n,
                opts,
                &params.cost_weights,
                // #809 — progress-probe path: tier 0 only (5× faster).
                false,
            )
        })
        .collect();
    // Accumulate in GOP order (the sequential map is already index-ordered).
    for cap in caps {
        let cap = cap?;
        *cover_bits = cover_bits.saturating_add(cap.coeff_sign_cover_bits);
        *shadow_pool_bits = shadow_pool_bits.saturating_add(cap.injectable_cover_bits);
        *n_gops = n_gops.saturating_add(1);
        for t in 0..5 {
            // CAP2.2 §12 (#824) — Σ per tier = the true carry/proportional ceiling.
            per_tier_sum[t] = per_tier_sum[t].saturating_add(cap.per_tier_payload[t]);
        }
        // CAP2.2 §12 — record this GOP's tier-0 cap (bytes) for proportional
        // allocation; GOP order matches the encode's chunk order.
        per_gop_tier0.push(cap.per_tier_payload[0]);
    }
    pending.clear();
    Ok(())
}

// ─────────────────────── decode side ──────────────────────────────────

/// Opaque streaming-decode session state.
///
/// Decodes chunk-framed streaming stego: each GOP is independently
/// STC-extracted, the per-GOP chunk_frame header steers reassembly,
/// and the concatenated payload bytes are decrypted as a single
/// phasm v1 frame. This is the only OH264 stego wire format the
/// production encode path emits (the legacy whole-video STC one-shot
/// encode/decode was retired in the video-retirement dead-path sweep).
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
    /// File attachments recovered from the payload (empty for text-only
    /// messages). Same payload bytes as `text` — the H.264 stego channel
    /// is payload-agnostic, so file attachments decode through the exact
    /// same per-GOP path.
    pub files: Vec<crate::stego::payload::FileEntry>,
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

    /// #474 — install a progress callback on this session. See
    /// [`DecodePhase`] for the event vocabulary; events fire from
    /// `push_annex_b` (Demux), the per-GOP brute-force loop in `finish`
    /// (Walker / ShadowExtract), and the decrypt step. FFI callers own
    /// the handle and attach the callback after `create()`.
    pub fn set_progress_callback(&mut self, callback: Option<DecodeProgressCallback>) {
        self.progress_callback = callback;
    }

    /// Internal helper — invoke the progress callback if installed.
    /// Cheap no-op when no callback is registered.
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

        // Decode order mirrors the image `smart_decode` + AV1's
        // `finish_shadow_*`: try the PRIMARY message first (common case,
        // per-GOP chunk_frame), then fall back to the whole-video SHADOW
        // scope. A clip can carry a primary AND any number of shadows under
        // different passphrases; the one matching THIS passphrase wins.
        // Since WV.6.a (#854) a shadow-encoded clip's primary is ALSO per-GOP
        // chunk_frame (only the shadows stay whole-video), so
        // `try_primary_decode` covers both the no-shadow and shadow-mode
        // primary. The #852 whole-video-primary `m_total` brute-force stopgap
        // is retired (WV.6.e) — no production clip carries a whole-video
        // primary anymore.
        match self.try_primary_decode(&slabs) {
            Ok(result) => Ok(result),
            // Per-GOP primary missed under this passphrase → try the whole-video
            // SHADOW scope (bounded: parity tiers × fdl).
            Err(primary_err) => self
                .try_shadow_streaming()
                // Nothing decoded under this passphrase. Surface the friendly
                // crypto error (wrong passphrase / no hidden message) rather
                // than the per-GOP path's internal `InvalidVideo("no v3 first
                // chunk")`, which the mobile UI mislabels as "video format
                // isn't supported" (#851). A genuine non-`InvalidVideo`
                // primary error (e.g. a mid-stream truncation) is preserved.
                .map_err(|_| match primary_err {
                    StegoError::InvalidVideo(_) => StegoError::DecryptionFailed,
                    other => other,
                }),
        }
    }

    /// Primary-message decode (chunk_frame v3 §5.2): accumulate per-GOP
    /// payload until GOP 0's declared `total_bytes` is reached, then parse
    /// + decrypt. `slabs` are the per-GOP Annex-B slices.
    fn try_primary_decode(&self, slabs: &[&[u8]]) -> Result<DecodeSessionResult, StegoError> {
        // Derive the STC seed (same as encode side).
        let keys = CabacStegoMasterKeys::derive(&self.passphrase)?;
        let hhat_seed = keys
            .per_gop_seeds(EmbedDomain::CoeffSignBypass, 0)
            .hhat_seed;

        // GOP 0 is always a stego GOP carrying the u32 `total_bytes`;
        // subsequent GOPs carry only `payload_len`. Stop at
        // `accumulated == total_bytes` and never walk the plain tail (W is
        // implicit, not on the wire — decode cost ∝ W, not n_gops).
        let cover0 = h264_gop_cover_bits(slabs[0], &self.cost_weights).ok_or_else(|| {
            StegoError::InvalidVideo("decode session: GOP 0 cover walk failed".into())
        })?;
        let (total_bytes, payload0) =
            extract_first_chunk_frame_match(&cover0, &hhat_seed, MAX_TOTAL_PAYLOAD_BYTES)
                .ok_or_else(|| {
                    StegoError::InvalidVideo(
                        "decode session: no v3 first chunk found in GOP 0".into(),
                    )
                })?;
        let total_bytes = total_bytes as usize;
        self.emit_progress(DecodePhase::Walker { frame_idx: 1, total_frames: slabs.len() as u32 });

        let mut frame_bytes = payload0;
        let mut gop_idx = 1usize;
        while frame_bytes.len() < total_bytes && gop_idx < slabs.len() {
            let cover = h264_gop_cover_bits(slabs[gop_idx], &self.cost_weights).ok_or_else(|| {
                StegoError::InvalidVideo(format!("decode session: GOP {gop_idx} cover walk failed"))
            })?;
            let remaining = total_bytes - frame_bytes.len();
            let payload =
                extract_chunk_frame_match(&cover, &hhat_seed, remaining).ok_or_else(|| {
                    StegoError::InvalidVideo(format!(
                        "decode session: no v3 chunk in stego GOP {gop_idx} (need {remaining} more bytes)"
                    ))
                })?;
            frame_bytes.extend_from_slice(&payload);
            gop_idx += 1;
            self.emit_progress(DecodePhase::Walker {
                frame_idx: gop_idx as u32,
                total_frames: slabs.len() as u32,
            });
        }
        if frame_bytes.len() != total_bytes {
            return Err(StegoError::InvalidVideo(format!(
                "decode session: accumulated {} bytes but GOP 0 declared total_bytes={total_bytes} (clip truncated mid-stream)",
                frame_bytes.len(),
            )));
        }

        // #474.2 — StcExtract boundary marker between Walker and Decrypt.
        self.emit_progress(DecodePhase::StcExtract);

        let parsed = frame::parse_frame(&frame_bytes)?;
        // #474.2 — Decrypt fires immediately before crypto::decrypt.
        self.emit_progress(DecodePhase::Decrypt);
        let plaintext = crypto::decrypt(
            &parsed.ciphertext,
            &self.passphrase,
            &parsed.salt,
            &parsed.nonce,
        )?;
        let payload_data = payload::decode_payload(&plaintext)?;
        // #474.2 — Done boundary fires AFTER the message text is ready.
        self.emit_progress(DecodePhase::Done);
        Ok(DecodeSessionResult {
            text: payload_data.text,
            files: payload_data.files,
            mode_id: 1, // Ghost = H.264 stego path
        })
    }

    /// #852 + WV.6.g (decode-side) — whole-video SHADOW decode, **O(GOP) walk**.
    /// Replaces the monolithic whole-clip walk (one union cover + mvd_meta held
    /// to the end = O(clip) heap). Splits the accumulated Annex-B into per-GOP
    /// slabs, walks ONE GOP at a time, and accumulates ONLY the priority-ordered
    /// `(priority, domain_rank, intra_index, bit)` tuples — **releasing each
    /// GOP's cover + mvd_meta** — then brute-forces
    /// [`decode_shadow_from_priority_lsbs`] over the accumulated bits.
    ///
    /// Bit-identical priority order to the whole-clip `priority_slots` it
    /// replaces: each GOP's keys are GLOBAL-remapped (`frame_idx += frames so
    /// far`, like the encoder's `gop_clean_cover`), eligibility mirrors
    /// `priority_slots` (CSB + CSL + MSB always; MSL where `safe_msl`; `safe_csl
    /// = None`), and the `(priority, domain_rank, intra_index)` sort reproduces
    /// its stable-sort-with-domain-grouped-insertion order — so it recovers the
    /// exact same shadow. Cascade-safety is frame-local, so the per-GOP
    /// `safe_msl` assembles to the whole-clip mask.
    ///
    /// NOT true O(GOP): the decoder doesn't know the shadow size (up to
    /// MAX_SHADOW_FRAME_BYTES), so the tuples are still O(eligible) — but ~half
    /// the per-position size of the cover they replace, and the whole-clip
    /// `mvd_meta` is gone (released per GOP). Returns the first shadow that
    /// AES-authenticates, or `FrameCorrupted` / a crypto error if none match.
    fn try_shadow_streaming(&self) -> Result<DecodeSessionResult, StegoError> {
        use super::stego::inject::remap_cover_frame_idx;
        use rand::{RngCore, SeedableRng};
        use rand_chacha::ChaCha20Rng;

        let perm_seed = crate::stego::crypto::derive_shadow_structural_key(&self.passphrase)?;
        let mut rng = ChaCha20Rng::from_seed(*perm_seed);

        // (priority, domain_rank, intra_index, bit) — the priority_slots order.
        let mut slots: Vec<(u32, u8, usize, u8)> = Vec::new();
        let mut offs = [0usize; 4]; // running global per-domain offsets
        // Cumulative ACTUAL frames decoded so far — the global-remap base. The
        // encoder's `streaming_shadow_verify` uses g*gop_size for the SAME base;
        // the two agree iff every non-final GOP == gop_size (the remap-base
        // contract, debug_assert'd encoder-side). Keep them in sync.
        let mut frames_so_far: u32 = 0;

        for slab in split_annex_b_into_gops(&self.accumulator) {
            let walk = walk_annex_b_for_cover_with_options(
                slab,
                WalkOptions { record_mvd: true, ..Default::default() },
            )
            .map_err(|_| StegoError::FrameCorrupted)?;
            let safe_msb = analyze_safe_mvd_subset(&walk.mvd_meta, walk.mb_w, walk.mb_h);
            let mut cover = walk.cover;
            let safe_msl = derive_msl_safe_from_msb(
                &cover.mvd_sign_bypass.positions,
                &safe_msb,
                &cover.mvd_suffix_lsb.positions,
            );
            // GLOBAL-remap keys so priorities match the whole-clip priority_slots.
            if frames_so_far > 0 {
                remap_cover_frame_idx(&mut cover, frames_so_far);
            }
            // Eligibility + domain order via the shared `for_each_eligible_position`
            // (safe_csl = None ⇒ all CSL; cascade-safe MSL only) — the exact rule
            // `priority_slots` + the streaming verify use, so the priority order
            // here is bit-identical to the whole-clip `priority_slots`.
            for_each_eligible_position(&cover, None, Some(safe_msl.as_slice()), |domain, i, key| {
                let bit = match domain {
                    EmbedDomain::CoeffSignBypass => cover.coeff_sign_bypass.bits[i],
                    EmbedDomain::CoeffSuffixLsb => cover.coeff_suffix_lsb.bits[i],
                    EmbedDomain::MvdSignBypass => cover.mvd_sign_bypass.bits[i],
                    EmbedDomain::MvdSuffixLsb => cover.mvd_suffix_lsb.bits[i],
                };
                rng.set_word_pos((key.raw() as u128).wrapping_mul(2));
                slots.push((rng.next_u32(), domain as u8, offs[domain as usize] + i, bit));
            });
            offs[0] += cover.coeff_sign_bypass.len();
            offs[1] += cover.coeff_suffix_lsb.len();
            offs[2] += cover.mvd_sign_bypass.len();
            offs[3] += cover.mvd_suffix_lsb.len();
            let mb_per_frame = (walk.mb_w as usize) * (walk.mb_h as usize);
            if mb_per_frame > 0 {
                frames_so_far += (walk.n_mb / mb_per_frame) as u32;
            }
            // `cover` + `walk` (incl. mvd_meta) drop here — O(GOP) peak.
        }

        // priority_slots' total order: stable sort by priority with domain-grouped
        // insertion ⇒ (priority, domain_rank, intra_index) ascending.
        slots.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)));
        let all_lsbs: Vec<u8> = slots.iter().map(|s| s.3).collect();

        let payload = decode_shadow_from_priority_lsbs(&all_lsbs, &self.passphrase)?;
        self.emit_progress(DecodePhase::Done);
        Ok(DecodeSessionResult {
            text: payload.text,
            files: payload.files,
            mode_id: 1, // Ghost = H.264 stego path
        })
    }
}

// ─────────────────────── decode helpers ───────────────────────────────

/// Split the Annex-B stream into per-GOP slabs at SPS NAL boundaries.
///
/// The streaming encode session restarts the OpenH264 encoder per
/// GOP (each `h264_encode_gop_framed_bits` call),
/// and OpenH264 emits SPS+PPS at every IDR. So SPS-position-marked NALs
/// are reliable GOP boundaries in our streaming output. Any prefix
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

/// chunk_frame v3 §4 — walk a GOP slab and combine the 4-domain cover
/// (CS → CSL → MVDs → MVDsl, matching the encoder). Returns the combined
/// cover bit vector the v3 STC extract operates on. Shared by both v3
/// extract variants + the encode-time round-trip verify.
fn h264_gop_cover_bits(slab: &[u8], weights: &CostWeights) -> Option<Vec<u8>> {
    // #493.5 Phase 4 — walk with MVD recording so the 4-domain combine
    // sees all 4 cover vectors (default walker leaves MVD cover empty).
    let walk = walk_annex_b_for_cover_with_options(
        slab,
        WalkOptions { record_mvd: true, ..Default::default() },
    ).ok()?;
    let dummy_costs = DomainCosts::default();
    let (cover_bits, _, _boundaries) =
        combine_cover_4domain(&walk.cover, &dummy_costs, weights);
    Some(cover_bits)
}

/// chunk_frame v3.1 first-chunk extract (GOP 0). Returns `(total_bytes,
/// payload)` for the candidate parsing as a v3.1 first chunk, or `None`.
///
/// Still iterates candidate `m_total` (the STC circularity: `w =
/// n_cover / m_total` is needed before any prefix can be read), but the v3.1
/// header carries the encoder's chosen `m_total` explicitly — an exact
/// `stored_m_total == candidate` reject (1/2³²) replaces v3's canonicality +
/// LEN_SENTINEL-bypass and collapses the false-positive survivors to a single
/// full STC extract per slab. See
/// docs/design/video/h264/chunk-frame-v3.1-decode.md (#888).
fn extract_first_chunk_frame_match(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    max_total_bytes: u32,
) -> Option<(u32, Vec<u8>)> {
    let n_cover = cover_bits.len();
    let min_m = CHUNK_FRAME_FIRST_HEADER_LEN_V3_1 * 8; // 80 bits
    if n_cover < min_m {
        return None;
    }
    let trace = std::env::var("PHASM_PERF_TRACE").map(|v| v == "1").unwrap_or(false);
    let (mut tries, mut prefix_ok) = (0usize, 0usize);
    let mut m_total = min_m;
    while m_total <= n_cover {
        tries += 1;
        let w = n_cover / m_total;
        if w == 0 {
            break;
        }
        let used = m_total * w;
        let hhat = generate_hhat(STC_H, w, hhat_seed);

        // 80-bit prefix: u32 total_bytes + u32 m_total + u16 payload_len.
        let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat, w, 80);
        if prefix_bits.len() < 80 {
            m_total += 8;
            continue;
        }
        let pb = bits_to_bytes_msb_first(&prefix_bits);
        if pb.len() < 10 {
            m_total += 8;
            continue;
        }
        // Exact m_total checksum: only the encoder's candidate satisfies it.
        let stored_m_total = u32::from_be_bytes([pb[4], pb[5], pb[6], pb[7]]) as usize;
        if stored_m_total != m_total {
            m_total += 8;
            continue;
        }
        let cand_total = u32::from_be_bytes([pb[0], pb[1], pb[2], pb[3]]);
        if cand_total == 0 || cand_total > max_total_bytes {
            m_total += 8;
            continue;
        }
        prefix_ok += 1;
        let extracted = stc_extract(&cover_bits[..used], &hhat, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = bits_to_bytes_msb_first(bits);
        if let Some((total_bytes, _m, payload)) = parse_first_chunk_frame_v3_1(&bytes) {
            if trace {
                eprintln!(
                    "[PHASM_PERF_TRACE v3_1_first] n_cover={n_cover} tries={tries} prefix_ok={prefix_ok} → HIT m_total={m_total} total_bytes={total_bytes}",
                );
            }
            return Some((total_bytes, payload.to_vec()));
        }
        m_total += 8;
    }
    if trace {
        eprintln!("[PHASM_PERF_TRACE v3_1_first] n_cover={n_cover} tries={tries} prefix_ok={prefix_ok} → MISS");
    }
    None
}

/// chunk_frame v3.1 subsequent-chunk extract (GOP ≥ 1). Returns `payload`.
/// `max_remaining_bytes` (= `total_bytes − accumulated`) bounds the final
/// payload. Same exact-`m_total` reject as the first-chunk variant (#888).
fn extract_chunk_frame_match(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    max_remaining_bytes: usize,
) -> Option<Vec<u8>> {
    let n_cover = cover_bits.len();
    let min_m = CHUNK_FRAME_NEXT_HEADER_LEN_V3_1 * 8; // 48 bits
    if n_cover < min_m {
        return None;
    }
    let mut m_total = min_m;
    while m_total <= n_cover {
        let w = n_cover / m_total;
        if w == 0 {
            break;
        }
        let used = m_total * w;
        let hhat = generate_hhat(STC_H, w, hhat_seed);

        // 48-bit prefix: u32 m_total + u16 payload_len. Exact m_total reject.
        let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat, w, 48);
        if prefix_bits.len() < 48 {
            m_total += 8;
            continue;
        }
        let pb = bits_to_bytes_msb_first(&prefix_bits);
        if pb.len() < 6 {
            m_total += 8;
            continue;
        }
        let stored_m_total = u32::from_be_bytes([pb[0], pb[1], pb[2], pb[3]]) as usize;
        if stored_m_total != m_total {
            m_total += 8;
            continue;
        }
        let extracted = stc_extract(&cover_bits[..used], &hhat, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = bits_to_bytes_msb_first(bits);
        if let Some((_m, payload)) = parse_chunk_frame_v3_1(&bytes) {
            if payload.len() <= max_remaining_bytes {
                return Some(payload.to_vec());
            }
        }
        m_total += 8;
    }
    None
}

/// Encode-time round-trip verify for one GOP's v3 chunk: re-walk the
/// just-encoded `gop` slab and confirm it extracts back to `expected`.
/// GOP 0 verifies as a first chunk; GOP ≥ 1 as a subsequent chunk.
fn verify_chunk_v3(
    gop: &[u8],
    hhat_seed: &[u8; 32],
    chunk_idx: u16,
    total_bytes: u32,
    expected: &[u8],
    weights: &CostWeights,
) -> bool {
    let Some(cover) = h264_gop_cover_bits(gop, weights) else {
        return false;
    };
    if chunk_idx == 0 {
        extract_first_chunk_frame_match(&cover, hhat_seed, MAX_TOTAL_PAYLOAD_BYTES)
            .is_some_and(|(_t, p)| p == expected)
    } else {
        extract_chunk_frame_match(&cover, hhat_seed, total_bytes as usize)
            .is_some_and(|p| p == expected)
    }
}

/// Local copy of the MSB-first bit→byte helper. Inlined here to keep
/// the decode side independent of `openh264_stego` (which is gated by
/// `h264-encoder`).
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
        // Since the pure-Rust encoder retirement, every u8 (including the
        // legacy `1` that used to select pure-Rust) resolves to the sole
        // OpenH264 backend — keeping the FFI stable for older clients.
        assert_eq!(EncodeEngineChoice::from_u8(0), EncodeEngineChoice::Oh264);
        assert_eq!(EncodeEngineChoice::from_u8(1), EncodeEngineChoice::Oh264);
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

        let mut s = StreamingDecodeSession::create("pw").unwrap();
        s.set_progress_callback(Some(cb));
        s.emit_progress(DecodePhase::Demux);
        s.emit_progress(DecodePhase::Done);
        let got = recorded.lock().unwrap().clone();
        assert_eq!(got, vec![DecodePhase::Demux, DecodePhase::Done]);
    }

    #[cfg(feature = "h264-encoder")]
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
        fn streaming_session_oh264_accepts_overflow_as_plain_tail() {
            // `oh264_push_frame` deliberately no longer hard-fails when more
            // frames arrive than `total_frames_hint` predicted (#799 — the
            // hint is an estimate; mobile streams the true count from
            // AVAsset/MediaCodec). Surplus GOPs fall outside the stego
            // window and route through `drain_one_gop` as PLAIN tail
            // (the message is fully embedded in the leading window; the
            // decoder ignores the extra slabs via its read-until-
            // `total_bytes` loop). Only the u16 GOP-index wraparound
            // (multi-million-frame pathology) still errors.
            //
            // Previously asserted that the 3rd frame errored; updated to
            // assert it succeeds, since the new behavior is the intended
            // one across all mobile / CLI overshoot scenarios.
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
            // A 3rd frame is accepted (plain-tail GOP).
            let (y, u, v) = synth_yuv_frame(W, H, 2);
            let frame = YuvFrameRef {
                y: &y, y_stride: W as usize,
                u: &u, u_stride: (W / 2) as usize,
                v: &v, v_stride: (W / 2) as usize,
            };
            sess.push_frame(frame, &mut out)
                .expect("3rd frame should now be accepted as plain-tail GOP");
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
        fn streaming_session_oh264_alloc_plan_roundtrip() {
            // CAP2.2 §12 — a precise per-GOP allocation plan (the CLI
            // two-pass path) must round-trip identically to the default
            // uniform-spread. 3-GOP fixture; front-load the plan onto GOP 0
            // (a shape uniform-spread would never choose) to exercise the
            // `gop_alloc_plan` branch in `drain_one_gop`, with the round-trip
            // verify + shrink-carry as the safety backstop.
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            const GOP: u32 = 2;
            const N: u32 = 6; // 3 GOPs
            const MSG: &str = "spread payload";
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
            let mut enc =
                StreamingEncodeSession::create(params, MSG, PASS).expect("encode create");

            // Build a valid 3-GOP plan summing to the framed length, front-
            // loaded onto GOP 0 (uniform-spread would split ~evenly instead).
            let framed = enc.framed_message_len().expect("oh264 framed len");
            let g0 = framed / 2;
            let g1 = (framed - g0) / 2;
            let g2 = framed - g0 - g1;
            assert_eq!(g0 + g1 + g2, framed);
            enc.set_gop_alloc_plan(vec![g0, g1, g2]).expect("set plan");

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
                "plan-allocated round-trip mismatch: got {:?}", result.text
            );

            // Plan validation rejects a wrong length or a wrong Σ.
            let mut bad = StreamingEncodeSession::create(
                EncodeSessionParams {
                    width: W, height: H, fps_num: 30, fps_den: 1, qp: 26,
                    gop_size: GOP, total_frames_hint: N,
                    color: ColorParams::default(),
                    engine: EncodeEngineChoice::Oh264,
                    cost_weights: CostWeights::default(),
                    progress_callback: None,
                },
                MSG, PASS,
            ).expect("create bad");
            assert!(bad.set_gop_alloc_plan(vec![framed, 0]).is_err(), "wrong length must reject");
            assert!(bad.set_gop_alloc_plan(vec![1, 1, 1]).is_err(), "wrong Σ must reject");
        }

        #[test]
        fn streaming_session_oh264_proportional_pipeline_roundtrip() {
            // CAP2.2 §12 Increment 2 — the full CLI two-pass pipeline at the
            // session level (no CLI/ffmpeg): probe the clip → per-GOP caps →
            // plan_proportional → encode-with-plan → decode round-trips. Proves
            // probe-vector + allocate + set_plan compose, and that proportional
            // allocation never loses data (the verify/carry backstop holds).
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            const GOP: u32 = 2;
            const N: u32 = 6; // 3 GOPs
            const MSG: &str = "proportional";
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

            // Pass 1 — probe the clip for per-GOP caps.
            let mut probe =
                StreamingProbeSession::create(params.clone()).expect("probe create");
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                probe.push_frame(YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                }).expect("probe push");
            }
            let (probe_result, per_gop_caps) =
                probe.finish_with_per_gop().expect("probe finish");
            assert_eq!(
                per_gop_caps.len(), probe_result.n_gops as usize,
                "per-GOP vector length must equal n_gops"
            );
            assert_eq!(per_gop_caps.len(), 3, "expected 3 GOPs");

            // Pass 2 — encode with a proportional plan from the probe caps.
            let mut enc =
                StreamingEncodeSession::create(params, MSG, PASS).expect("encode create");
            let planned = enc.plan_proportional(&per_gop_caps, 1.0).expect("plan");
            assert!(planned, "proportional plan should install for a fitting message");

            let mut annex_b = Vec::new();
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                enc.push_frame(YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                }, &mut annex_b).expect("push");
            }
            enc.finish(&mut annex_b).expect("finish encode");

            let mut dec = StreamingDecodeSession::create(PASS).expect("decode create");
            dec.push_annex_b(&annex_b).expect("push annex");
            let result = dec.finish().expect("finish decode");
            assert_eq!(
                result.text, MSG,
                "proportional pipeline round-trip mismatch: got {:?}", result.text
            );
        }

        #[test]
        fn streaming_session_oh264_plain_tail_roundtrip() {
            // CAP2.3 §6 — concentrate+tail with a genuine PLAIN
            // tail. A tiny message over 4 GOPs concentrates into GOP 0 (W=1);
            // GOPs 1-3 carry NO stego. Verifies what a round-trip ALONE cannot
            // (the decoder SKIPS the tail, so tail corruption is invisible to it):
            //   1. all 4 GOPs emit as separable H.264 (SPS-per-GOP holds for the
            //      plain `encode_once` tail — else `split_annex_b_into_gops` would
            //      merge them and desync the decoder's index↔GOP map),
            //   2. GOP 0's wire `total_chunks` == W == 1 (< n_gops),
            //   3. the tail GOPs carry NO chunk (genuinely plain, not the
            //      even-split empty-chunk tail),
            //   4. EVERY GOP — incl. the plain tail — walks cleanly (structurally
            //      valid, not fork-state corrupted: the #548-class reset works),
            //   5. the message round-trips via the decoder early-dropout.
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            const GOP: u32 = 2;
            const N: u32 = 8; // 4 GOPs
            const MSG: &str = "hi"; // tiny → W=1, plain tail GOPs 1-3
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

            let mut probe = StreamingProbeSession::create(params.clone()).expect("probe create");
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                probe.push_frame(YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                }).expect("probe push");
            }
            let (probe_result, per_gop_caps) = probe.finish_with_per_gop().expect("probe finish");
            assert_eq!(probe_result.n_gops, 4, "expected 4 GOPs");

            let mut enc = StreamingEncodeSession::create(params, MSG, PASS).expect("enc create");
            // r=1.0 (greedy concentrate) guarantees W=1 for a tiny message.
            assert!(
                enc.plan_proportional(&per_gop_caps, 1.0).expect("plan"),
                "concentrate+tail plan should install"
            );

            let mut annex_b = Vec::new();
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                enc.push_frame(YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                }, &mut annex_b).expect("push");
            }
            enc.finish(&mut annex_b).expect("finish encode");

            // (1) all 4 GOPs emitted + SPS-separable (incl. the plain tail).
            let slabs = split_annex_b_into_gops(&annex_b);
            assert_eq!(slabs.len(), 4, "all 4 GOPs (incl. plain tail) must emit + be SPS-separable");

            let keys = CabacStegoMasterKeys::derive(PASS).expect("keys");
            let hhat_seed = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0).hhat_seed;
            let weights = CostWeights::default();

            // (2) GOP 0 carries the whole tiny message in its v3 first chunk:
            // `total_bytes == GOP 0's payload` (W = 1, no subsequent chunks).
            let cover0 = h264_gop_cover_bits(slabs[0], &weights).expect("GOP 0 cover walk");
            let (total_bytes, payload0) =
                extract_first_chunk_frame_match(&cover0, &hhat_seed, MAX_TOTAL_PAYLOAD_BYTES)
                    .expect("GOP 0 must carry the v3 first chunk");
            assert_eq!(
                total_bytes as usize,
                payload0.len(),
                "tiny message must fit entirely in GOP 0 (W = 1, no subsequent chunks)"
            );

            // (3)+(4) tail GOPs are genuinely plain (no chunk) yet walk cleanly.
            for i in 0..4usize {
                walk_annex_b_for_cover_with_options(
                    slabs[i], WalkOptions { record_mvd: true, ..Default::default() },
                ).unwrap_or_else(|e| panic!("GOP {i} failed to walk (corrupt plain tail?): {e}"));
            }
            for i in 1..4usize {
                let cover_i = h264_gop_cover_bits(slabs[i], &weights).expect("tail GOP cover walk");
                assert!(
                    extract_chunk_frame_match(&cover_i, &hhat_seed, total_bytes as usize).is_none(),
                    "plain tail GOP {i} must NOT carry a v3 chunk"
                );
            }

            // (5) round-trips via decoder early-dropout (collects W=1, skips tail).
            let mut dec = StreamingDecodeSession::create(PASS).expect("dec create");
            dec.push_annex_b(&annex_b).expect("push annex");
            let result = dec.finish().expect("finish decode");
            assert_eq!(result.text, MSG, "plain-tail round-trip mismatch: {:?}", result.text);
        }

        #[test]
        fn streaming_session_oh264_capacity_is_sigma_not_evensplit() {
            // CAP2.2 §12 (#824) — the reported capacity must be Σ per-GOP caps
            // (what the carry/proportional encoders actually fill), i.e.
            // `per_tier_sum_payload[0] − FRAME_OVERHEAD`, NOT the even-split
            // `n_gops × min`. Σ ≥ n_gops × min always.
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
            const W: u32 = 320;
            const H: u32 = 240;
            const GOP: u32 = 2;
            const N: u32 = 6; // 3 GOPs

            let params = EncodeSessionParams {
                width: W, height: H, fps_num: 30, fps_den: 1, qp: 26,
                gop_size: GOP, total_frames_hint: N,
                color: ColorParams::default(),
                engine: EncodeEngineChoice::Oh264,
                cost_weights: CostWeights::default(),
                progress_callback: None,
            };
            let mut probe = StreamingProbeSession::create(params).expect("probe create");
            for f in 0..N {
                let (y, u, v) = synth_yuv_frame(W, H, f);
                probe.push_frame(YuvFrameRef {
                    y: &y, y_stride: W as usize,
                    u: &u, u_stride: (W / 2) as usize,
                    v: &v, v_stride: (W / 2) as usize,
                }).expect("probe push");
            }
            let (result, per_gop_caps) = probe.finish_with_per_gop().expect("finish");
            assert_eq!(per_gop_caps.len(), result.n_gops as usize);

            let sigma: usize = per_gop_caps.iter().sum();
            // The accumulated Σ equals the sum of the per-GOP vector.
            assert_eq!(
                result.per_tier_sum_payload[0], sigma,
                "per_tier_sum[0] must == Σ per-GOP caps"
            );
            // Capacity is Σ − overhead (the new formula), not n_gops × min − overhead.
            let overhead = crate::stego::frame::FRAME_OVERHEAD;
            assert_eq!(
                result.primary_max_message_bytes(),
                sigma.saturating_sub(overhead),
                "primary capacity must be Σ − FRAME_OVERHEAD"
            );
            // Σ is always ≥ the old even-split n_gops × min (the under-report it replaces).
            let min_cap = *per_gop_caps.iter().min().unwrap();
            assert!(
                sigma >= (result.n_gops as usize) * min_cap,
                "Σ ({}) must be ≥ even-split n×min ({} × {})",
                sigma, result.n_gops, min_cap
            );
        }
        #[test]
        fn streaming_session_emits_progress_events_round_trip() {
            // #474.2 — end-to-end: install encode + decode progress
            // callbacks, run a 2-GOP OH264 round-trip, verify the
            // event sequences are well-formed (Setup first, Done last,
            // monotonic frame_idx / gop_idx). The progress-event
            // protocol is backend-agnostic; OH264 is the only encode
            // engine since the pure-Rust retirement, so we take the
            // SESSION_TEST_MUTEX like the other OH264 session tests.
            let _g = SESSION_TEST_MUTEX.lock().unwrap();
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
                engine: EncodeEngineChoice::Oh264,
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
                .expect("decode create");
            dec.set_progress_callback(Some(dec_cb));
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
