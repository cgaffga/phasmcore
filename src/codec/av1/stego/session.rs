// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 streaming session (chunk_frame wire format).
//!
//! See [`phase-c-streaming-session-v6.md`](../../../../../docs/design/video/av1/phase-c-streaming-session-v6.md).
//!
//! ## Scope
//!
//! - Session API surface (`create` / `push_frame` / `finish` on encode;
//!   `create` / `push_bytes` / `finish` on decode).
//! - Per-GOP chunk_frame v3 header on the wire — first stego GOP
//!   carries `[total_bytes u32 BE | payload_len u16 BE | payload]`
//!   (6 bytes), subsequent stego GOPs carry just `[payload_len u16 BE
//!   | payload]` (2 bytes). Decoder stops when `Σ payload_len ==
//!   total_bytes`. See `docs/design/video/chunk-frame-v3.md`.
//! - Pre-encrypt + chunk-split at `session_create` (not lazy).
//! - **Multi-frame-per-GOP supported**. `push_frame` accumulates
//!   `gop_size` frames in `gop_buffer` and drains via the multi-frame
//!   primitive (`av1_stego_encode_one_gop`) when the GOP fills.
//!   `gop_size=1` keeps the byte-exact single-frame path for
//!   capacity-probe math + existing test expectations.
//! - Wire format set in stone here: chunk_frame layout unchanged;
//!   only the encode internals differ between v=1 and v>1 GOPs.
//!
//! ## Multi-frame-per-GOP behavior
//!
//! - `push_frame` accumulates `gop_size` frames in `gop_buffer` before
//!   calling `av1_stego_encode_one_gop` (multi-frame) or the
//!   single-frame primitive for v=1.
//! - Per-GOP `Encoder::new` reset (no cross-GOP state leak); the fork's
//!   `encode_gop_with_phasm_tee` runs a fresh keyframe + P-frame chain
//!   per call.
//! - Decode side OBU walker handles multi-frame slabs unchanged —
//!   `sequence_header_obu` boundaries continue to mark per-GOP slabs
//!   regardless of frames-per-GOP.
//!
//! ## Wire compatibility
//!
//! Streaming-session output is **NOT** wire-compatible with legacy
//! `av1_stego_embed` output — the session always emits the chunk_frame
//! v3 header; legacy never does. Mobile bridges adopt the session;
//! CLI keeps both entry points. v3 wire format also has zero
//! backwards compatibility with v2 chunk_frame (no shipped AV1
//! installed base on v2).
//!
//! ## v3 length-strict invariants
//!
//! Both `parse_first_chunk_frame` and `parse_chunk_frame` are
//! length-strict: extracted bytes must equal `header_len + payload_len`
//! exactly. This forces the m_total brute-force search to land on the
//! encoder's exact value within each w-class. Same disambiguation
//! invariant as v2's `payload_len` field, preserved verbatim.

use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::PhasmFrameRecording;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, encode_gop_with_phasm_tee, make_frame, make_inter_config,
    FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

use crate::stego::chunk_frame::{
    allocate_chunks_concentrate_tail, build_chunk_frame, build_first_chunk_frame,
    split_message_into_chunks,
};
use crate::stego::{crypto, frame};

use super::orchestrator::Av1StegoError;
#[cfg(feature = "av1-encoder")]
use super::orchestrator::{
    av1_stego_embed_payload_bits, av1_stego_encode_one_gop,
    av1_stego_encode_one_gop_with_shadows_parity,
};
#[cfg(feature = "av1-decoder")]
use super::orchestrator::{
    extract_chunk_frame_match, extract_first_chunk_frame_match,
    harvest_cover_bits_from_stego, split_av1_into_gops,
};

/// Configuration for an `Av1StreamingEncodeSession`. Captures the
/// frame dimensions, rav1e quantizer, GOP size, and the
/// total-frames-hint that drives chunk-split sizing.
///
/// `total_frames_hint` MUST equal the actual number of frames the
/// caller will push. There is no streaming-without-frame-count mode
/// (live capture / unknown duration → deferred to a later release).
#[derive(Debug, Clone, Copy)]
pub struct Av1StreamingEncodeParams {
    pub width: u32,
    pub height: u32,
    pub quantizer: usize,
    pub gop_size: u32,
    /// Caller's promise of how many frames will be pushed in total.
    /// Drives `total_chunks = ceil(total_frames_hint / gop_size)`.
    pub total_frames_hint: u32,
}

/// One shadow message to embed alongside the primary message at every
/// GOP. Each shadow is fully recoverable from any single GOP via
/// `av1_stego_extract_shadow` with the shadow's own passphrase.
///
/// **Per-GOP scope**: shadows are per-GOP (each GOP carries the full
/// shadow message independently with per-GOP-cover-derived priority).
/// Whole-video scope (positions selected over the union cover across
/// all GOPs) is available via `create_whole_video_with_shadows` and is
/// stealth-stronger but pays the whole-clip memory cost. Per-GOP scope
/// ships a fully-functional multi-shadow API; the stealth-axis cost is
/// that a statistical analyser comparing per-GOP shadow-bit
/// distributions could infer shadow presence.
#[derive(Debug, Clone)]
pub struct Av1ShadowSpec {
    pub passphrase: String,
    pub message: Vec<u8>,
}

/// Outcome of a single [`Av1StreamingEncodeSession::plan_safe_balanced`]
/// call. See that method's doc for the caller-loop pattern.
#[derive(Debug, Clone, PartialEq)]
pub enum Av1PlanOutcome {
    /// Plan installed on the session. `window` is W (number of leading
    /// stego GOPs); GOPs `[W, n_gops)` are natural tail under the
    /// concentrate-tail allocator.
    Installed { window: u32 },

    /// Caller should probe these additional GOP indices, append their
    /// `(idx, cap_bytes)` measurements to the `samples` vector, and
    /// call `plan_safe_balanced` again.
    NeedMoreSamples {
        positions: Vec<usize>,
        total_target: usize,
    },

    /// Even at full-corpus coverage the message doesn't fit. Surface
    /// as a user-facing error.
    MessageTooLarge,
}

/// Whole-video shadow accumulation state. When present, the session
/// defers ALL encode work to `finish` and runs the whole-video shadow
/// flow (`av1_stego_encode_whole_video_with_shadows`) on the full
/// buffered YUV. Mirrors OH264's `Oh264ShadowSessionState` at
/// `core/src/codec/h264/streaming_session.rs:281`.
///
/// Per-GOP shadow scope (default) and whole-video shadow scope (this)
/// coexist; the constructor picks. Whole-video scope is stealth-
/// stronger but pays the whole-clip memory cost
/// (~93 MB raw I420 at 1080p × 30 s).
struct Av1WholeVideoState {
    primary_message: Vec<u8>,
    primary_passphrase: String,
    shadows: Vec<Av1ShadowSpec>,
    shadow_parity_len: usize,
    /// Packed frame-by-frame tight I420 for the whole clip
    /// (frame 0 Y|U|V, frame 1 Y|U|V, ...). Pre-allocated to
    /// `frame_size × total_frames_hint` at create.
    yuv_buffer: Vec<u8>,
    frames_pushed: u32,
}

/// Streaming AV1 stego encode session.
///
/// **Per-GOP scope (default)** — chunk-framed wire format; each
/// `push_frame` may emit one GOP's stego bytes into `out` once a GOP
/// fills. `create` and `create_with_shadows` produce this mode.
///
/// **Whole-video scope** (opt-in via
/// `create_whole_video_with_shadows`) — all `push_frame` calls
/// accumulate raw YUV; `finish` runs the whole-video shadow flow on
/// the buffered clip and emits stego bytes for all GOPs at once.
/// Trades streaming memory profile for stronger shadow stealth (no
/// per-GOP density anomaly).
#[cfg(feature = "av1-encoder")]
pub struct Av1StreamingEncodeSession {
    params: Av1StreamingEncodeParams,
    chunks: Vec<Vec<u8>>,
    /// Total encrypted+framed message length (in bytes). Emitted on the
    /// wire as the first-chunk `total_bytes` field (u32 BE). The decoder
    /// uses this as the stop signal: read chunks until
    /// `Σ payload_len == total_message_bytes`. Replaces v2's
    /// `total_chunks` wire field.
    total_message_bytes: u32,
    /// Internal counter — number of stego chunks the encoder will emit
    /// (= `W` under the concentrate-tail allocator). Walked by
    /// `drain_one_gop` until `chunk_idx == total_chunks`, after which
    /// GOPs are natural. **No longer emitted on the wire under v3** —
    /// the decoder derives W implicitly by accumulating per-chunk
    /// payload bytes against `total_message_bytes`.
    /// - Without a per-GOP allocation plan: `= derived_total_gops`.
    /// - With a plan: `= W` (count of leading stego GOPs).
    total_chunks: u16,
    /// **GOP** counter — ticks on every drained GOP, including natural
    /// tail GOPs. Distinct from `total_chunks`: after a plan install,
    /// `chunk_idx` walks 0..=`derived_total_gops` while `total_chunks`
    /// stays at W.
    chunk_idx: u16,
    /// Derived from `total_frames_hint + gop_size` at create. Stays
    /// constant; the upper bound on `chunk_idx`. Equals `total_chunks`
    /// pre-plan; `≥ total_chunks` post-plan.
    derived_total_gops: u16,
    /// The encrypted+framed primary bytes, retained so a post-create
    /// `set_gop_alloc_plan` / `plan_proportional` can re-split chunks
    /// according to the new plan. `None` after a successful plan
    /// install (the chunks themselves carry the re-split bytes;
    /// `frame_bytes` is no longer needed).
    frame_bytes: Option<Vec<u8>>,
    hhat_seed: [u8; 32],
    /// Held for symmetry with future per-GOP-derived key paths (e.g.
    /// shadow layers). Currently unused by encode; the hhat_seed alone
    /// drives STC.
    #[allow(dead_code)]
    passphrase: String,
    /// Tight I420 bytes for the in-flight GOP, packed frame-by-frame
    /// (frame 0 Y|U|V, frame 1 Y|U|V, ...). Accumulates up to
    /// `gop_size` frames before draining via `drain_one_gop`.
    /// Unused in whole-video mode.
    gop_buffer: Vec<u8>,
    frames_buffered_in_gop: u32,
    frames_pushed_total: u32,
    /// Shadow messages embedded at every GOP. Empty in the no-shadow
    /// case (the standard `create` API path). In whole-video mode the
    /// shadows live on `whole_video` instead.
    shadows: Vec<Av1ShadowSpec>,
    /// RS parity length used across all shadows. Decoder brute-forces
    /// `AV1_SHADOW_PARITY_TIERS` so the encoder can pick any value
    /// from that set; default 16 (matches the H.264 mid-tier).
    shadow_parity_len: usize,
    /// `Some` if the session was created via
    /// `create_whole_video_with_shadows`. When set, push_frame
    /// accumulates instead of draining, and finish runs the whole-video
    /// encode.
    whole_video: Option<Av1WholeVideoState>,
}

#[cfg(feature = "av1-encoder")]
impl Av1StreamingEncodeSession {
    /// Validate params, encrypt the message once, chunk-split it
    /// across the projected GOP count, and lock the session state.
    /// Equivalent to `create_with_shadows(passphrase, message, params, vec![], 16)`.
    pub fn create(
        passphrase: &str,
        message: &[u8],
        params: Av1StreamingEncodeParams,
    ) -> Result<Self, Av1StegoError> {
        Self::create_with_shadows(passphrase, message, params, Vec::new(), 16)
    }

    /// Create a session that embeds N shadow messages alongside the
    /// primary at every GOP. Each shadow gets its own passphrase;
    /// recovery is via `av1_stego_extract_shadow` per GOP slab
    /// (`Av1StreamingDecodeSession::finish_shadow_first_match` runs
    /// that loop).
    ///
    /// `shadow_parity_len` is the RS parity length used uniformly
    /// across all shadows. Default `16` (matches `create`'s baked
    /// value); higher values tolerate more inter-shadow collisions at
    /// the cost of per-shadow capacity.
    pub fn create_with_shadows(
        passphrase: &str,
        message: &[u8],
        params: Av1StreamingEncodeParams,
        shadows: Vec<Av1ShadowSpec>,
        shadow_parity_len: usize,
    ) -> Result<Self, Av1StegoError> {
        if params.gop_size == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "session.create: gop_size must be >= 1".into(),
            ));
        }
        if params.total_frames_hint == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "session.create: total_frames_hint must be > 0".into(),
            ));
        }
        if params.width == 0 || params.height == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "session.create: width/height must be > 0".into(),
            ));
        }

        let structural_key = crypto::derive_structural_key(passphrase)?;
        let hhat_seed: [u8; 32] = structural_key[32..]
            .try_into()
            .expect("derive_structural_key returns 64 bytes");

        // Encrypt the primary message ONCE per session (NOT per GOP).
        // The encrypted+framed bytes are split across N GOP chunks.
        let (ciphertext, nonce, salt) = crypto::encrypt(message, passphrase)?;
        let frame_bytes = frame::build_frame(message.len(), &salt, &nonce, &ciphertext);

        let expected_n_gops = params
            .total_frames_hint
            .div_ceil(params.gop_size.max(1));
        if expected_n_gops == 0 || expected_n_gops > u16::MAX as u32 {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.create: derived total_chunks {} out of range [1..={}]",
                expected_n_gops,
                u16::MAX
            )));
        }
        let total_chunks = expected_n_gops as u16;
        if frame_bytes.len() > u32::MAX as usize {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.create: framed message {} exceeds u32::MAX",
                frame_bytes.len()
            )));
        }
        let total_message_bytes = frame_bytes.len() as u32;
        let chunks = split_message_into_chunks(&frame_bytes, total_chunks)
            .map_err(Av1StegoError::Stego)?;

        Ok(Self {
            params,
            chunks,
            total_message_bytes,
            total_chunks,
            chunk_idx: 0,
            derived_total_gops: total_chunks,
            frame_bytes: Some(frame_bytes),
            hhat_seed,
            passphrase: passphrase.to_string(),
            gop_buffer: Vec::new(),
            frames_buffered_in_gop: 0,
            frames_pushed_total: 0,
            shadows,
            shadow_parity_len,
            whole_video: None,
        })
    }

    /// Create a session with whole-video shadow scope.
    ///
    /// All `push_frame` calls accumulate raw YUV into the session's
    /// `yuv_buffer`; `finish` runs the whole-video shadow encode on
    /// the buffered clip. The chunk_frame wire format is the same as
    /// per-GOP scope — only the shadow position distribution differs
    /// (top-N globally across the union cover vs per-GOP top-N).
    ///
    /// `shadows` empty is degenerate (no shadows = use per-GOP
    /// `create` instead); errors with `InvalidPacket`.
    ///
    /// Memory: pre-allocates
    /// `frame_size × total_frames_hint × 1.5 bytes` for the YUV
    /// buffer. At 1080p × 30 sec that's ~93 MB. Callers should
    /// size `total_frames_hint` accordingly.
    ///
    /// Requires both `av1-encoder` and `av1-decoder` features — the
    /// whole-video self-verify path in `finish()` calls into the
    /// dav1d-walker `harvest_cover_bits_from_stego`.
    #[cfg(feature = "av1-decoder")]
    pub fn create_whole_video_with_shadows(
        passphrase: &str,
        message: &[u8],
        params: Av1StreamingEncodeParams,
        shadows: Vec<Av1ShadowSpec>,
        shadow_parity_len: usize,
    ) -> Result<Self, Av1StegoError> {
        if shadows.is_empty() {
            return Err(Av1StegoError::InvalidPacket(
                "session.create_whole_video_with_shadows: empty shadows — use \
                 create_with_shadows for per-GOP scope or create for no-shadow"
                    .into(),
            ));
        }
        if params.gop_size == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "session.create_whole_video_with_shadows: gop_size must be >= 1".into(),
            ));
        }
        if params.total_frames_hint == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "session.create_whole_video_with_shadows: total_frames_hint must be > 0".into(),
            ));
        }
        if params.width == 0 || params.height == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "session.create_whole_video_with_shadows: width/height must be > 0".into(),
            ));
        }

        let structural_key = crypto::derive_structural_key(passphrase)?;
        let hhat_seed: [u8; 32] = structural_key[32..]
            .try_into()
            .expect("derive_structural_key returns 64 bytes");

        // Encrypt + frame the primary message — the whole-video encode
        // expects pre-framed primary bytes plumbed via the
        // whole_video state.
        let (ciphertext, nonce, salt) = crypto::encrypt(message, passphrase)?;
        let frame_bytes = frame::build_frame(message.len(), &salt, &nonce, &ciphertext);

        let expected_n_gops = params
            .total_frames_hint
            .div_ceil(params.gop_size.max(1));
        if expected_n_gops == 0 || expected_n_gops > u16::MAX as u32 {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.create_whole_video_with_shadows: derived n_gops {} out of range [1..={}]",
                expected_n_gops,
                u16::MAX
            )));
        }
        let total_chunks = expected_n_gops as u16;
        if frame_bytes.len() > u32::MAX as usize {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.create_whole_video_with_shadows: framed message {} exceeds u32::MAX",
                frame_bytes.len()
            )));
        }
        let total_message_bytes = frame_bytes.len() as u32;
        let chunks = split_message_into_chunks(&frame_bytes, total_chunks)
            .map_err(Av1StegoError::Stego)?;

        let frame_size = expected_i420_size(params.width, params.height);
        let buf_capacity = frame_size
            .checked_mul(params.total_frames_hint as usize)
            .ok_or_else(|| {
                Av1StegoError::InvalidPacket(format!(
                    "session.create_whole_video_with_shadows: total YUV size overflow ({} × {})",
                    frame_size, params.total_frames_hint
                ))
            })?;

        let whole_video = Av1WholeVideoState {
            primary_message: frame_bytes,
            primary_passphrase: passphrase.to_string(),
            shadows: shadows.clone(),
            shadow_parity_len,
            yuv_buffer: Vec::with_capacity(buf_capacity),
            frames_pushed: 0,
        };

        Ok(Self {
            params,
            chunks,
            total_message_bytes,
            total_chunks,
            chunk_idx: 0,
            derived_total_gops: total_chunks,
            // Whole-video mode doesn't go through per-GOP drain; plan
            // installation isn't supported. Set to None so attempts
            // surface a clear error.
            frame_bytes: None,
            hhat_seed,
            passphrase: passphrase.to_string(),
            gop_buffer: Vec::new(),
            frames_buffered_in_gop: 0,
            frames_pushed_total: 0,
            shadows,
            shadow_parity_len,
            whole_video: Some(whole_video),
        })
    }

    /// Total chunks committed at `create` — visible to callers for
    /// progress UI.
    pub fn total_chunks(&self) -> u16 {
        self.total_chunks
    }

    /// Total framed-message bytes (the `Σ` a custom
    /// `set_gop_alloc_plan` plan must equal). `None` after the plan has
    /// been installed (the `frame_bytes` buffer is consumed at that
    /// point) or in whole-video mode (no per-GOP plan).
    pub fn framed_message_len(&self) -> Option<usize> {
        self.frame_bytes.as_ref().map(|b| b.len())
    }

    /// Install a per-GOP byte-allocation plan, overriding the default
    /// even-split. `plan[i]` is the byte count for stego GOP `i`; the
    /// plan's length defines the **stego window** W (the on-wire
    /// `total_chunks` becomes W). GOPs `i ≥ W` get a fully natural
    /// encode — no chunk_frame header, no STC, no cover hooks fire
    /// (see `docs/design/video/av1/` design doc §8.3).
    ///
    /// Constraints:
    /// - `plan.len() ≥ 1` and `plan.len() ≤ derived_total_gops`.
    /// - `Σ plan == framed_message_len()` — the plan must place every
    ///   framed byte exactly.
    /// - Call before the first `push_frame` and before `finish`.
    /// - Per-GOP mode only (not whole-video).
    pub fn set_gop_alloc_plan(&mut self, plan: Vec<usize>) -> Result<(), Av1StegoError> {
        if self.whole_video.is_some() {
            return Err(Av1StegoError::InvalidPacket(
                "set_gop_alloc_plan: not supported in whole-video mode".into(),
            ));
        }
        if self.chunk_idx > 0 || self.frames_buffered_in_gop > 0 {
            return Err(Av1StegoError::InvalidPacket(
                "set_gop_alloc_plan: must be called before the first push_frame".into(),
            ));
        }
        let frame_bytes = self.frame_bytes.take().ok_or_else(|| {
            Av1StegoError::InvalidPacket(
                "set_gop_alloc_plan: plan already installed (no framed bytes to re-split)".into(),
            )
        })?;
        if plan.is_empty() {
            self.frame_bytes = Some(frame_bytes); // restore
            return Err(Av1StegoError::InvalidPacket(
                "set_gop_alloc_plan: window must be >= 1".into(),
            ));
        }
        if plan.len() > self.derived_total_gops as usize {
            self.frame_bytes = Some(frame_bytes);
            return Err(Av1StegoError::InvalidPacket(format!(
                "set_gop_alloc_plan: window {} > derived total GOPs {}",
                plan.len(),
                self.derived_total_gops
            )));
        }
        let sigma: usize = plan.iter().sum();
        let fb_len = frame_bytes.len();
        if sigma != fb_len {
            self.frame_bytes = Some(frame_bytes);
            return Err(Av1StegoError::InvalidPacket(format!(
                "set_gop_alloc_plan: Σ plan = {sigma} != framed message length {fb_len}"
            )));
        }
        // Re-chunk-split: chunks[i] = frame_bytes[off..off+plan[i]] for i < W.
        let mut chunks: Vec<Vec<u8>> = Vec::with_capacity(plan.len());
        let mut off = 0usize;
        for &len in &plan {
            chunks.push(frame_bytes[off..off + len].to_vec());
            off += len;
        }
        self.chunks = chunks;
        self.total_chunks = plan.len() as u16;
        Ok(())
    }

    /// Convenience: build a concentrate+tail plan from a per-GOP
    /// cover-byte budget and install it. `r_target` is the per-GOP
    /// rate ceiling (lower → wider stego window, lower per-GOP flip
    /// density). `0.5` is the H.264-aligned conservative default.
    ///
    /// Returns `Ok(true)` if a plan was installed, `Ok(false)` if the
    /// message exceeds `Σ gop_caps` (the encode then surfaces
    /// `MessageTooLarge` from `push_frame`) OR if `gop_caps.len()`
    /// doesn't match the session's `derived_total_gops`. Falling back
    /// leaves the default even-split, which is always safe.
    pub fn plan_proportional(
        &mut self,
        gop_caps: &[usize],
        r_target: f64,
    ) -> Result<bool, Av1StegoError> {
        if gop_caps.len() != self.derived_total_gops as usize {
            return Ok(false);
        }
        let Some(framed_len) = self.framed_message_len() else {
            return Ok(false); // already planned or WV mode
        };
        let Some((plan, _window)) =
            allocate_chunks_concentrate_tail(gop_caps, framed_len, r_target)
        else {
            // Σ cap < message — leave the even-split and let push_frame
            // surface a real `MessageTooLarge` at the right per-GOP
            // boundary.
            return Ok(false);
        };
        // The concentrate_tail allocator pads the plan with trailing
        // zeros up to gop_caps.len(); trim them so plan.len() == W.
        let window = plan.iter().rposition(|&p| p > 0).map(|i| i + 1).unwrap_or(0);
        if window == 0 {
            return Ok(false);
        }
        self.set_gop_alloc_plan(plan[..window].to_vec())?;
        Ok(true)
    }

    /// Caller-driven balanced safe planner.
    ///
    /// One iteration of the corpus-calibrated balanced allocator from
    /// [`crate::stego::balanced_allocation::plan_safe_balanced`].
    /// Caller supplies `samples` (a list of `(gop_idx, cap_bytes)` pairs
    /// measured so far) plus the per-encode-GOP `gop_size` and the
    /// [`AllocationCalibration`] (typically `AV1_1080P_QP30`). Returns:
    ///
    /// - `Ok(Av1PlanOutcome::Installed { window })` — plan was installed
    ///   on this session; caller can `push_frame` immediately.
    /// - `Ok(Av1PlanOutcome::NeedMoreSamples { positions, total_target })`
    ///   — caller must probe the requested GOP indices (e.g., via a
    ///   second `Av1StreamingProbeSession` seeking to those positions),
    ///   append the results to `samples`, then call this method again.
    /// - `Ok(Av1PlanOutcome::MessageTooLarge)` — even at full-corpus
    ///   coverage the message doesn't fit; user-facing error.
    /// - `Err(...)` — invalid pre-condition (e.g., session already past
    ///   `push_frame`, or in whole-video mode).
    ///
    /// The caller-loop pattern (see
    /// `docs/design/balanced-allocation-v3.md` §3.1):
    ///
    /// ```ignore
    /// let mut samples: Vec<(usize, usize)> = Vec::new();
    /// loop {
    ///     match session.plan_safe_balanced(&samples, gop_size, &cal)? {
    ///         Av1PlanOutcome::Installed { .. } => break,
    ///         Av1PlanOutcome::NeedMoreSamples { positions, .. } => {
    ///             let probed = probe_those_gops(positions);
    ///             samples.extend(probed);
    ///         }
    ///         Av1PlanOutcome::MessageTooLarge => return Err(...),
    ///     }
    /// }
    /// // Now push_frame the encode loop normally.
    /// ```
    pub fn plan_safe_balanced(
        &mut self,
        samples: &[(usize, usize)],
        gop_size: usize,
        calibration: &crate::stego::calibration::AllocationCalibration,
    ) -> Result<Av1PlanOutcome, Av1StegoError> {
        if self.chunk_idx > 0 || self.frames_buffered_in_gop > 0 {
            return Err(Av1StegoError::InvalidPacket(
                "plan_safe_balanced: cannot install plan after push_frame has been called"
                    .into(),
            ));
        }
        let Some(framed_len) = self.framed_message_len() else {
            return Err(Av1StegoError::InvalidPacket(
                "plan_safe_balanced: framed_message_len gone (plan already installed or \
                 whole-video mode)"
                    .into(),
            ));
        };

        let n_gops = self.derived_total_gops as usize;
        use crate::stego::balanced_allocation::{plan_safe_balanced as core_plan, PlanOutcome};
        match core_plan(samples, n_gops, framed_len, gop_size, calibration) {
            PlanOutcome::Plan { plan, window } => {
                // Same trimming convention as plan_proportional: pass
                // only the first `window` entries (set_gop_alloc_plan's
                // contract is plan.len() == W).
                if window == 0 {
                    return Ok(Av1PlanOutcome::Installed { window: 0 });
                }
                self.set_gop_alloc_plan(plan[..window].to_vec())?;
                Ok(Av1PlanOutcome::Installed {
                    window: window as u32,
                })
            }
            PlanOutcome::NeedMoreSamples {
                positions,
                total_target,
            } => Ok(Av1PlanOutcome::NeedMoreSamples {
                positions,
                total_target,
            }),
            PlanOutcome::MessageTooLarge => Ok(Av1PlanOutcome::MessageTooLarge),
        }
    }

    /// Push one frame's tight I420 YUV bytes.
    ///
    /// **Per-GOP mode** (`create` / `create_with_shadows`):
    /// accumulates into the per-GOP buffer; drains when `gop_size`
    /// frames are buffered or when `finish` is called (trailing
    /// partial GOP). For `gop_size=1` each push immediately drains
    /// a 1-frame GOP.
    ///
    /// **Whole-video mode** (`create_whole_video_with_shadows`):
    /// appends to the session's `yuv_buffer` and returns; no
    /// per-frame output bytes. `finish` runs the whole-video encode
    /// and emits all stego bytes at once.
    pub fn push_frame(
        &mut self,
        yuv_i420: &[u8],
        out: &mut Vec<u8>,
    ) -> Result<(), Av1StegoError> {
        let expected = expected_i420_size(self.params.width, self.params.height);
        if yuv_i420.len() != expected {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.push_frame: yuv length {} != expected {} for {}×{}",
                yuv_i420.len(),
                expected,
                self.params.width,
                self.params.height,
            )));
        }

        // Whole-video mode: accumulate without draining.
        if let Some(wv) = self.whole_video.as_mut() {
            if wv.frames_pushed >= self.params.total_frames_hint {
                return Err(Av1StegoError::InvalidPacket(format!(
                    "session.push_frame (whole-video): frames_pushed {} would exceed \
                     total_frames_hint {}",
                    wv.frames_pushed, self.params.total_frames_hint
                )));
            }
            wv.yuv_buffer.extend_from_slice(yuv_i420);
            wv.frames_pushed += 1;
            self.frames_pushed_total += 1;
            return Ok(());
        }

        // Per-GOP mode (default).
        // Cap by `derived_total_gops` (constant after create), not by
        // `total_chunks` (which post-plan equals the smaller stego
        // window W). Natural-tail GOPs i ∈ [W, derived_total_gops) are
        // valid pushes that route to natural-encode in drain_one_gop.
        if self.chunk_idx >= self.derived_total_gops {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.push_frame: chunk_idx {} >= derived_total_gops {} (pushed more frames than hinted)",
                self.chunk_idx, self.derived_total_gops
            )));
        }

        self.gop_buffer.extend_from_slice(yuv_i420);
        self.frames_buffered_in_gop += 1;
        self.frames_pushed_total += 1;

        // Drain on GOP boundary. Trailing partial GOP gets drained by
        // `finish()`. We also stop accumulating once we've reached
        // total_frames_hint even if the last GOP is partial — each
        // chunk_idx gets exactly one GOP.
        if self.frames_buffered_in_gop >= self.params.gop_size
            || self.frames_pushed_total >= self.params.total_frames_hint
        {
            self.drain_one_gop(out)?;
        }
        let _ = out;
        Ok(())
    }

    /// Encode the buffered GOP (1 or N frames) as one stego AV1
    /// chunk_frame'd unit, append to `out`, advance `chunk_idx`.
    ///
    /// Dispatches on `frames_buffered_in_gop`: for 1 frame uses the
    /// existing single-frame primitive (preserves byte-exact behavior
    /// + capacity-probe math). For >1 frame, uses the multi-frame
    /// primitive that runs `encode_gop_with_phasm_tee` for an IDR +
    /// (N-1) P-frame chain and a single STC plan over the combined
    /// GOP cover.
    fn drain_one_gop(&mut self, out: &mut Vec<u8>) -> Result<(), Av1StegoError> {
        if self.frames_buffered_in_gop == 0 {
            return Ok(());
        }

        // chunk_idx ≥ total_chunks routes to a fully natural encode
        // (no chunk_frame, no STC, no cover hooks). Pre-plan,
        // `total_chunks == derived_total_gops` and we never enter this
        // branch.
        if self.chunk_idx >= self.total_chunks {
            let natural_packet = if self.frames_buffered_in_gop == 1 {
                let (pkt, _recording) = encode_one_keyframe(&self.gop_buffer, self.params)?;
                pkt
            } else {
                let per_frame = encode_one_gop_multi(
                    &self.gop_buffer,
                    self.frames_buffered_in_gop,
                    self.params,
                )?;
                let mut concatenated = Vec::new();
                for (pkt, _recording) in per_frame {
                    concatenated.extend_from_slice(&pkt);
                }
                concatenated
            };
            out.extend_from_slice(&natural_packet);
            self.chunk_idx += 1;
            self.frames_buffered_in_gop = 0;
            self.gop_buffer.clear();
            return Ok(());
        }

        let chunk_payload = &self.chunks[self.chunk_idx as usize];
        // v3 wire: first stego GOP carries `total_message_bytes`
        // (clip-level header); subsequent stego GOPs carry only
        // payload_len. Decoder stops when
        // `Σ payload_len == total_message_bytes`.
        let framed = if self.chunk_idx == 0 {
            build_first_chunk_frame(self.total_message_bytes, chunk_payload)
        } else {
            build_chunk_frame(chunk_payload)
        }
        .map_err(Av1StegoError::Stego)?;
        let payload_bits = frame::bytes_to_bits(&framed);

        let shadow_refs: Vec<(&str, &[u8])> = self
            .shadows
            .iter()
            .map(|s| (s.passphrase.as_str(), s.message.as_slice()))
            .collect();

        let stego_packet = if self.frames_buffered_in_gop == 1 {
            // Single-frame path. Bit-exact to the legacy
            // av1_stego_embed pipeline; preserves capacity-probe math
            // and existing test expectations.
            let (natural_packet, recording) =
                encode_one_keyframe(&self.gop_buffer, self.params)?;
            if shadow_refs.is_empty() {
                av1_stego_embed_payload_bits(
                    natural_packet,
                    recording,
                    &payload_bits,
                    &self.hhat_seed,
                )?
            } else {
                super::orchestrator::av1_stego_embed_payload_bits_with_shadows_parity(
                    natural_packet,
                    recording,
                    &payload_bits,
                    &self.hhat_seed,
                    &shadow_refs,
                    self.shadow_parity_len,
                )?
            }
        } else {
            // Multi-frame path. Build N frames from the packed
            // gop_buffer + run the fork's encode_gop_with_phasm_tee +
            // single combined STC.
            let per_frame =
                encode_one_gop_multi(&self.gop_buffer, self.frames_buffered_in_gop, self.params)?;
            if shadow_refs.is_empty() {
                av1_stego_encode_one_gop(per_frame, &payload_bits, &self.hhat_seed)?
            } else {
                av1_stego_encode_one_gop_with_shadows_parity(
                    per_frame,
                    &payload_bits,
                    &self.hhat_seed,
                    &shadow_refs,
                    self.shadow_parity_len,
                )?
            }
        };

        out.extend_from_slice(&stego_packet);
        self.chunk_idx += 1;
        self.frames_buffered_in_gop = 0;
        self.gop_buffer.clear();
        Ok(())
    }

    /// Finalize the session.
    ///
    /// **Per-GOP mode**: `chunk_idx` must equal `total_chunks` after
    /// any trailing partial GOP gets drained.
    ///
    /// **Whole-video mode**: runs
    /// `av1_stego_encode_whole_video_with_shadows` on the buffered
    /// YUV and appends the full-clip stego bytes. `frames_pushed`
    /// must equal `total_frames_hint`.
    pub fn finish(mut self, out: &mut Vec<u8>) -> Result<(), Av1StegoError> {
        // Whole-video mode: run the whole-video shadow encode on the
        // buffered YUV. Gated on `av1-decoder` because the whole-video
        // path self-verifies via `harvest_cover_bits_from_stego`
        // (decoder hook). Under encoder-only builds the whole-video
        // constructor is gated off, so `whole_video` is always `None`
        // and this branch is dead code — cfg-out the whole block.
        #[cfg(feature = "av1-decoder")]
        if let Some(wv) = self.whole_video.take() {
            if wv.frames_pushed != self.params.total_frames_hint {
                return Err(Av1StegoError::InvalidPacket(format!(
                    "session.finish (whole-video): pushed {} frames, expected \
                     total_frames_hint {}",
                    wv.frames_pushed, self.params.total_frames_hint
                )));
            }
            let shadow_refs: Vec<(&str, &[u8])> = wv
                .shadows
                .iter()
                .map(|s| (s.passphrase.as_str(), s.message.as_slice()))
                .collect();
            let bytes = super::whole_video::av1_stego_encode_whole_video_with_shadows(
                &wv.yuv_buffer,
                wv.frames_pushed,
                self.params,
                &wv.primary_message,
                &wv.primary_passphrase,
                &shadow_refs,
                wv.shadow_parity_len,
            )?;
            out.extend_from_slice(&bytes);
            return Ok(());
        }

        // Per-GOP mode (default).
        if self.frames_buffered_in_gop > 0 {
            self.drain_one_gop(out)?;
        }
        // chunk_idx must reach total_chunks (stego window fully
        // drained). Pre-plan total_chunks == derived_total_gops so
        // this is the "drained every GOP" check. Post-plan
        // total_chunks == W and chunk_idx may go further to drain
        // the natural tail; the lower bound is what matters.
        if self.chunk_idx < self.total_chunks {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.finish: drained {} GOPs but on-wire total_chunks={} \
                 (insufficient stego window; frames_pushed_total {})",
                self.chunk_idx, self.total_chunks, self.frames_pushed_total
            )));
        }
        if self.chunk_idx > self.derived_total_gops {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.finish: drained {} GOPs exceeds derived_total_gops {}",
                self.chunk_idx, self.derived_total_gops
            )));
        }
        Ok(())
    }
}

/// Streaming AV1 stego decode session. Accumulates incoming stego AV1
/// bytes; at `finish`, runs the OBU walker to split into per-GOP
/// slabs at `sequence_header_obu` boundaries, then per-slab extracts
/// a chunk_frame, assembles, decrypts.
#[cfg(feature = "av1-decoder")]
pub struct Av1StreamingDecodeSession {
    passphrase: String,
    /// Raw accumulated AV1 OBU bytes — split into per-GOP slabs at
    /// `finish` via `split_av1_into_gops` (the OBU walker).
    accumulator: Vec<u8>,
}

#[cfg(feature = "av1-decoder")]
impl Av1StreamingDecodeSession {
    pub fn create(passphrase: &str) -> Self {
        Self {
            passphrase: passphrase.to_string(),
            accumulator: Vec::new(),
        }
    }

    /// Push any quantity of stego AV1 bytes. The OBU walker at
    /// `finish` discovers per-GOP boundaries. Callers can push at
    /// arbitrary byte granularity (single push of the whole stream,
    /// chunked pushes from a network socket, etc.).
    pub fn push_bytes(&mut self, av1_bytes: &[u8]) {
        if !av1_bytes.is_empty() {
            self.accumulator.extend_from_slice(av1_bytes);
        }
    }

    /// Finalize: walk accumulator into per-GOP slabs, extract one
    /// chunk_frame per slab, assemble payload, decrypt, return
    /// plaintext.
    pub fn finish(self) -> Result<Vec<u8>, Av1StegoError> {
        if self.accumulator.is_empty() {
            return Err(Av1StegoError::ExtractionFailed);
        }
        let gop_slabs = split_av1_into_gops(&self.accumulator);
        if gop_slabs.is_empty() {
            return Err(Av1StegoError::InvalidPacket(
                "session.finish: OBU walker found no sequence_header_obu in accumulated bytes \
                 (input not produced by Av1StreamingEncodeSession?)"
                    .into(),
            ));
        }

        let structural_key = crypto::derive_structural_key(&self.passphrase)?;
        let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

        // v3 wire: first GOP carries `total_bytes` (clip header);
        // subsequent GOPs carry only `payload_len`. Decoder concatenates
        // payloads in GOP order and stops when `accumulated == total_bytes`.
        // The concentrate-tail allocator's natural tail is implicit — once
        // accumulated reaches the target, the remaining slabs (if any)
        // are natural and we never try to STC-extract them.
        let mut assembled: Vec<u8> = Vec::new();
        let mut total_bytes_target: Option<u32> = None;

        // Sanity cap on first-chunk `total_bytes`. Realistic phasm
        // video-stego payloads top out at a few hundred MB (a short
        // video inside a video; arbitrary file attachments). 256 MB
        // covers current use cases with margin while tightening the
        // wrong-passphrase / wrong-w fast-reject filter from 1/2^32 to
        // ~1/16 per brute-force candidate (combined with the length-strict
        // payload_len filter, total false-positive rate drops to ~10^-5
        // per GOP at 1080p — comparable to v2's chunk_idx-match filter).
        // Raise if/when 8K + multi-hour clips need it.
        const MAX_TOTAL_PAYLOAD_BYTES: u32 = 256 * 1024 * 1024;

        for (gop_idx, slab) in gop_slabs.iter().enumerate() {
            let cover_bits = harvest_cover_bits_from_stego(slab)?;
            if gop_idx == 0 {
                let (total_bytes, payload) = extract_first_chunk_frame_match(
                    &cover_bits,
                    &hhat_seed,
                    MAX_TOTAL_PAYLOAD_BYTES,
                )
                .ok_or(Av1StegoError::ExtractionFailed)?;
                total_bytes_target = Some(total_bytes);
                assembled.extend_from_slice(&payload);
            } else {
                let target = total_bytes_target
                    .expect("first chunk parsed total_bytes before loop body iterates");
                let remaining = (target as usize).saturating_sub(assembled.len());
                if remaining == 0 {
                    // Already satisfied — natural-tail GOPs follow.
                    break;
                }
                let payload = extract_chunk_frame_match(&cover_bits, &hhat_seed, remaining)
                    .ok_or(Av1StegoError::ExtractionFailed)?;
                assembled.extend_from_slice(&payload);
            }
            if let Some(target) = total_bytes_target {
                if assembled.len() == target as usize {
                    break;
                }
                if assembled.len() > target as usize {
                    // Should not happen — extract_*_v3_match honors
                    // max_remaining_bytes, but be defensive.
                    return Err(Av1StegoError::InvalidPacket(format!(
                        "session.finish: assembled {} bytes overshot total_bytes={} \
                         (decoder over-read past clip-header target)",
                        assembled.len(),
                        target
                    )));
                }
            }
        }

        let target = total_bytes_target.ok_or(Av1StegoError::ExtractionFailed)?;
        if assembled.len() != target as usize {
            return Err(Av1StegoError::InvalidPacket(format!(
                "session.finish: assembled {} bytes but first-chunk header says total_bytes={} \
                 (stego window incomplete — possibly a truncated stream)",
                assembled.len(),
                target
            )));
        }

        let parsed = frame::parse_frame(&assembled).map_err(Av1StegoError::Stego)?;
        let plaintext = crypto::decrypt(
            &parsed.ciphertext,
            &self.passphrase,
            &parsed.salt,
            &parsed.nonce,
        )
        .map_err(Av1StegoError::Stego)?;
        Ok(plaintext)
    }

    /// Extract a shadow message using a shadow-specific passphrase.
    /// Walks per-GOP slabs (same OBU split as `finish`) and returns
    /// the first GOP that successfully decodes the shadow under
    /// `shadow_pass`. Each GOP carries the full shadow independently
    /// (per-GOP scope); any one GOP recovers the message.
    ///
    /// Borrows `&self` since shadow extract doesn't consume session
    /// state — caller can later call `finish()` for the primary, or
    /// `finish_shadow_first_match` again with a different shadow
    /// passphrase.
    pub fn finish_shadow_first_match(
        &self,
        shadow_pass: &str,
    ) -> Result<crate::stego::payload::PayloadData, crate::stego::error::StegoError> {
        if self.accumulator.is_empty() {
            return Err(crate::stego::error::StegoError::FrameCorrupted);
        }
        let gop_slabs = split_av1_into_gops(&self.accumulator);
        if gop_slabs.is_empty() {
            return Err(crate::stego::error::StegoError::InvalidVideo(
                "session.finish_shadow: OBU walker found no sequence_header_obu in \
                 accumulated bytes".into(),
            ));
        }
        for slab in &gop_slabs {
            // For each slab, run the shadow-extract pipeline. The
            // first GOP that yields a valid AES-authenticated shadow
            // wins; subsequent slabs aren't explored to keep cost
            // bounded.
            if let Ok(payload) =
                super::orchestrator::av1_stego_extract_shadow(slab, shadow_pass)
            {
                return Ok(payload);
            }
        }
        Err(crate::stego::error::StegoError::FrameCorrupted)
    }

    /// Extract a whole-video shadow message.
    ///
    /// Unlike `finish_shadow_first_match` (which walks per-GOP slabs
    /// and tries shadow extract on each individually — the per-GOP
    /// shadow scope), this method harvests the UNION cover across all
    /// GOPs and runs shadow extract once. Required for shadows
    /// produced by `Av1StreamingEncodeSession::create_whole_video_with_shadows`,
    /// whose shadow bits are spread top-N across the whole-video
    /// cover (no single GOP slab carries enough bits to RS-decode).
    ///
    /// Borrows `&self` so the caller can call it multiple times with
    /// different shadow passphrases.
    pub fn finish_shadow_whole_video(
        &self,
        shadow_pass: &str,
    ) -> Result<crate::stego::payload::PayloadData, crate::stego::error::StegoError> {
        if self.accumulator.is_empty() {
            return Err(crate::stego::error::StegoError::FrameCorrupted);
        }
        // Walk the WHOLE accumulator as one stream — no per-GOP
        // split. `harvest_cover_bits_from_stego` produces the union
        // Tier-1 cover (AC_COEFF_SIGN + GOLOMB_TAIL_LSB) across all
        // OBUs in walker emit order. That's the same cover the
        // encoder's verify gate operated on.
        let cover = super::orchestrator::harvest_cover_bits_from_stego(&self.accumulator)
            .map_err(|e| match e {
                super::orchestrator::Av1StegoError::Stego(s) => s,
                _ => crate::stego::error::StegoError::FrameCorrupted,
            })?;
        super::shadow::av1_shadow_extract(&cover, shadow_pass)
    }
}

fn expected_i420_size(width: u32, height: u32) -> usize {
    let y = (width as usize) * (height as usize);
    let uv = ((width as usize) / 2) * ((height as usize) / 2);
    y + 2 * uv
}

/// Encode an N-frame GOP via the phasm-rav1e fork's
/// `encode_gop_with_phasm_tee` helper. Frame 0 is the keyframe; frames
/// 1..N are P-frames referencing the prior reconstructions. Returns
/// per-frame `(natural_packet, recording)` pairs that the orchestrator
/// consumes via `av1_stego_encode_one_gop`.
///
/// `gop_buffer` is the packed I420 YUV for ALL `frames_in_gop` frames
/// (frame 0 Y|U|V, frame 1 Y|U|V, ..., frame N-1 Y|U|V). Length must
/// equal `frames_in_gop × frame_size`.
fn encode_one_gop_multi(
    gop_buffer: &[u8],
    frames_in_gop: u32,
    params: Av1StreamingEncodeParams,
) -> Result<Vec<(Vec<u8>, PhasmFrameRecording<u8>)>, Av1StegoError> {
    let w = params.width as usize;
    let h = params.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let frame_size = y_size + 2 * uv_size;
    let expected_total = frame_size * frames_in_gop as usize;
    if gop_buffer.len() != expected_total {
        return Err(Av1StegoError::InvalidPacket(format!(
            "encode_one_gop_multi: gop_buffer len {} != frames_in_gop {} × frame_size {}",
            gop_buffer.len(),
            frames_in_gop,
            frame_size,
        )));
    }

    // low_latency = true is load-bearing: it disables frame reorder
    // + B-frames. The per-GOP stego flow emits frames in input order
    // and runs a single STC plan over the combined cover; B-frame
    // reorder would break both. multiref also off — keeps ref
    // selection deterministic for stealth profile stability.
    let mut config = EncoderConfig {
        width: w,
        height: h,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: params.quantizer,
        ..Default::default()
    };
    config.low_latency = true;
    config.speed_settings.multiref = false;
    let config = Arc::new(config);
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let sequence = Arc::new(sequence);

    let yuvs: Vec<Arc<phasm_rav1e::Frame<u8>>> = (0..frames_in_gop as usize)
        .map(|i| {
            let off = i * frame_size;
            let mut frame_in = make_frame::<u8>(w, h, ChromaSampling::Cs420);
            frame_in.planes[0].copy_from_raw_u8(&gop_buffer[off..off + y_size], w, 1);
            frame_in.planes[1].copy_from_raw_u8(
                &gop_buffer[off + y_size..off + y_size + uv_size],
                w / 2,
                1,
            );
            frame_in.planes[2].copy_from_raw_u8(
                &gop_buffer[off + y_size + uv_size..off + frame_size],
                w / 2,
                1,
            );
            Arc::new(frame_in)
        })
        .collect();

    Ok(encode_gop_with_phasm_tee::<u8>(&yuvs, config, sequence))
}

/// Encode one keyframe to AV1 OBU bytes + recording. The
/// `frames_buffered_in_gop == 1` path uses this for byte-exact
/// parity with the legacy single-frame primitive (kept for
/// capacity-probe math + existing test expectations). The `> 1` path
/// goes through `encode_one_gop_multi` instead.
fn encode_one_keyframe(
    yuv_i420: &[u8],
    params: Av1StreamingEncodeParams,
) -> Result<(Vec<u8>, PhasmFrameRecording<u8>), Av1StegoError> {
    let config = Arc::new(EncoderConfig {
        width: params.width as usize,
        height: params.height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: params.quantizer,
        ..Default::default()
    });
    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;

    let w = params.width as usize;
    let h = params.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    if yuv_i420.len() != y_size + 2 * uv_size {
        return Err(Av1StegoError::InvalidPacket(format!(
            "encode_one_keyframe: yuv len {} != y_size+uv*2 {}",
            yuv_i420.len(),
            y_size + 2 * uv_size
        )));
    }

    let mut frame_in = make_frame::<u8>(w, h, ChromaSampling::Cs420);
    frame_in.planes[0].copy_from_raw_u8(&yuv_i420[..y_size], w, 1);
    frame_in.planes[1].copy_from_raw_u8(&yuv_i420[y_size..y_size + uv_size], w / 2, 1);
    frame_in.planes[2].copy_from_raw_u8(
        &yuv_i420[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame_in));
    let inter_cfg = make_inter_config(&config);
    Ok(encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg))
}
