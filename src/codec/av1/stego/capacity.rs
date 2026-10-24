// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 capacity API
//! (see docs/design/video/av1/phase-c-capacity-api.md).
//!
//! Two-API + real-probe + conservative ratio. Mirror of the H.264
//! `h264_stego_capacity_4domain` / `h264_stego_shadow_capacity` shape
//! adapted for AV1's joint Tier 1 (AC_COEFF_SIGN + GOLOMB_TAIL_LSB).
//!
//! ## `Av1StreamingProbeSession`
//!
//! Session-shape probe. `push_frame` runs a natural baseline encode
//! of the frame (no STC, no embed) and counts tagged cover positions
//! in the recording. Returns total cover_bits + n_gops at `finish`.
//!
//! ## `av1_capacity`
//!
//! One-shot top-level API. Wraps probe session + closed-form math
//! `primary_max_bits = cover_bits × 0.40` (the STC_PAYLOAD_RATIO
//! safety floor), then subtracts chunk_frame v3 overhead (6 + 2 × (n_gops − 1))
//! + FRAME_OVERHEAD.
//!
//! ## `av1_shadow_capacity`
//!
//! Multi-shadow API. Collision formula
//! `m_max_bits ≤ sqrt(1024 × C / max(1, N-1))` lifted verbatim from
//! H.264 `shadow-messages.md`.
//!
//! ## v0.6 vs design-doc full scope
//!
//! The design doc calls for binary-searching the runtime allocator
//! because H.264's `stealth_weighted_allocation` has per-domain caps
//! that closed-form math over-estimates. AV1 v0.6 has a single joint
//! Tier 1 cover and no per-domain weighted caps — closed-form
//! suffices. When Tier 2/3 land (v1.1+) with per-channel weights,
//! binary-search becomes necessary and replaces the closed-form
//! here.

use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

use crate::stego::calibration::AllocationCalibration;
use crate::stego::chunk_frame::{
    CHUNK_FRAME_FIRST_HEADER_LEN_V3_1, CHUNK_FRAME_NEXT_HEADER_LEN_V3_1,
};
use crate::stego::frame::FRAME_OVERHEAD;
use crate::stego::payload::MAX_INNER_PAYLOAD_BYTES;

/// Maximum per-clip chunk_frame v3.1 overhead at a given GOP count.
/// First GOP carries `CHUNK_FRAME_FIRST_HEADER_LEN_V3_1` bytes (clip header);
/// subsequent up to `n_gops - 1` GOPs carry `CHUNK_FRAME_NEXT_HEADER_LEN_V3_1`
/// bytes each. Returns the **upper-bound** overhead under the assumption
/// every GOP is a stego chunk (i.e., no Increment B natural tail) — the
/// conservative case for capacity reporting.
///
/// v3.1 adds the 4-byte `m_total` checksum per chunk: first-chunk header is
/// 10 B, subsequent 6 B → upper bound `10 + 6*(n_gops-1) = 6*n_gops + 4`.
/// Real overhead under a balanced-W plan is lower still.
#[inline]
fn chunk_frame_v3_1_overhead_bytes(n_gops: usize) -> usize {
    if n_gops == 0 {
        0
    } else {
        CHUNK_FRAME_FIRST_HEADER_LEN_V3_1
            + CHUNK_FRAME_NEXT_HEADER_LEN_V3_1.saturating_mul(n_gops.saturating_sub(1))
    }
}

use super::orchestrator::Av1StegoError;
use super::session::Av1StreamingEncodeParams;

/// AV1 effective STC ratio (v0.6 closed-form). H.264 ships 0.40
/// (STC reaches 0.45-0.50 on real content; 0.40 buys content-drift
/// insurance). AV1's joint Tier 1 (AC_COEFF_SIGN + GOLOMB_TAIL_LSB)
/// has materially less headroom than H.264's 4-domain because:
///
/// 1. The cost-rejection filter rejects ~25-40% of cover positions
///    to INF cost (`|coeff| × factor > THRESHOLD_REJECT`). STC must
///    route syndromes around the INF positions, which fails earlier
///    than raw `cover_bits × w` would suggest.
/// 2. The smoothness gate (default SMOOTHNESS=40 PENALTY=4) elevates
///    costs in smooth regions by 4×. STC strongly avoids those,
///    further shrinking the practically-usable cover.
/// 3. The chunk_frame v3.1 header adds 10 bytes for the first stego GOP
///    + 6 bytes per subsequent stego GOP (≤ 6 × n_gops + 4 bytes total at
///    worst-case W = n_gops; less under Increment B's natural-tail
///    plans).
///
/// Combined effective ratio is closer to 0.12-0.15. We ship 0.15 to
/// keep the API contract — "primary_max is a SAFE upper bound, encoder
/// always succeeds at or below" — and defer the binary-search-the-real-
/// allocator follow-on (see docs/design/video/av1/phase-c-capacity-api.md)
/// that gives a tight bound by running STC at incrementally larger m_bits
/// until it fails.
pub(crate) const STC_PAYLOAD_RATIO_NUM: usize = 15;
pub(crate) const STC_PAYLOAD_RATIO_DEN: usize = 100;

/// Session-shape capacity probe. Runs the same per-GOP encode
/// as `Av1StreamingEncodeSession` but skips STC + override apply,
/// just counts tagged cover positions per GOP.
///
/// Wall-clock is essentially identical to the encode phase of a real
/// stego (the per-frame encode dominates; STC + override apply is
/// microseconds). For the API contract, "real probe not analytical"
/// matters because cover-bit yield varies 5-10× across content
/// (rav1e's RDO drives residual + non-zero coefficient population).
pub struct Av1StreamingProbeSession {
    params: Av1StreamingEncodeParams,
    cover_bits: usize,
    n_gops: u32,
    /// Per-domain breakdown for diagnostics (Av1PerDomainBits).
    ac_sign_bits: usize,
    golomb_tail_bits: usize,
    /// Per-GOP cover-bit count. Semantics depend on `params.gop_size`:
    ///   - `gop_size == 1` (legacy): one entry per `push_frame` call —
    ///     each frame is modeled as a standalone I-frame. Same shape
    ///     `av1_calibrate` and the existing tests have always seen.
    ///   - `gop_size > 1` (T2.4): one entry per **encode-shaped GOP**
    ///     (1 IDR + (gop_size−1) P-frames). Internally the probe
    ///     buffers YUV in `current_gop_yuv` until it has gop_size
    ///     frames, then encodes the GOP via `encode_gop_natural` and
    ///     counts Tier-1 tags on the rav1e recording. Match the
    ///     encode's per-GOP cover production exactly (was systematic
    ///     ~2× over-estimate at gop_size=1 because every frame was an
    ///     I-frame, far more nonzero AC coefficients than P-frames).
    ///
    /// Caller threads this through `set_gop_alloc_plan` /
    /// `plan_proportional` on the encode session for Increment B
    /// concentrate+tail allocation. Same units as `cover_bits`;
    /// convert to byte caps with
    /// `Av1CapacityProbeResult::per_gop_byte_caps`.
    per_gop_cover_bits: Vec<usize>,
    /// T2.4: per-GOP YUV accumulator for `gop_size > 1`. Drained on
    /// flush. Empty when `gop_size == 1`. RAM held at peak =
    /// `gop_size × width × height × 3 / 2` (e.g. 1080p × gop=30 =
    /// ~90 MB transient — same shape as the encode's per-GOP buffer).
    current_gop_yuv: Vec<u8>,
    /// Frames already accumulated into `current_gop_yuv`. When this
    /// hits `params.gop_size`, the GOP gets flushed and the buffer
    /// is cleared. `finish()` flushes any trailing partial GOP.
    frames_in_current_gop: u32,
}

#[derive(Debug, Clone)]
pub struct Av1CapacityProbeResult {
    pub cover_bits: usize,
    pub ac_sign_bits: usize,
    pub golomb_tail_bits: usize,
    pub n_gops: u32,
    /// Per-GOP cover-bit vector. Empty for legacy callers that only
    /// used the whole-clip cover_bits.
    pub per_gop_cover_bits: Vec<usize>,
}

impl Av1StreamingProbeSession {
    pub fn create(params: Av1StreamingEncodeParams) -> Result<Self, Av1StegoError> {
        // T2.4: lifted the `gop_size != 1` hard reject. At gop_size=1
        // we keep the legacy per-frame I-frame probe (calibration-bit-
        // identical); at gop_size > 1 we run the real encode-shaped
        // probe via `flush_current_gop`. See the `per_gop_cover_bits`
        // doc on the struct for semantics.
        if params.gop_size == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "probe.create: gop_size must be >= 1".into(),
            ));
        }
        if params.width == 0 || params.height == 0 {
            return Err(Av1StegoError::InvalidPacket(
                "probe.create: width/height must be > 0".into(),
            ));
        }
        Ok(Self {
            params,
            cover_bits: 0,
            n_gops: 0,
            ac_sign_bits: 0,
            golomb_tail_bits: 0,
            per_gop_cover_bits: Vec::new(),
            current_gop_yuv: Vec::new(),
            frames_in_current_gop: 0,
        })
    }

    /// Push one I420 frame.
    ///
    /// **gop_size = 1 (legacy fast path):** every frame is its own
    /// GOP, encoded as a standalone I-frame via `probe_one_keyframe`.
    /// Counts AC-sign + golomb-tail tags from the recording, appends
    /// one entry to `per_gop_cover_bits`. Behavior bit-identical to
    /// pre-T2.4.
    ///
    /// **gop_size > 1 (T2.4 new path):** the YUV is appended to the
    /// per-GOP buffer; when the buffer hits `gop_size` frames the GOP
    /// is flushed via `flush_current_gop` (calls `encode_gop_natural`
    /// so the rav1e encode runs with the real frame-type pattern —
    /// 1 IDR + (gop_size−1) P-frames — matching what the actual
    /// encode produces).
    pub fn push_frame(&mut self, yuv_i420: &[u8]) -> Result<(), Av1StegoError> {
        let expected = expected_i420_size(self.params.width, self.params.height);
        if yuv_i420.len() != expected {
            return Err(Av1StegoError::InvalidPacket(format!(
                "probe.push_frame: yuv length {} != expected {} for {}×{}",
                yuv_i420.len(),
                expected,
                self.params.width,
                self.params.height,
            )));
        }
        if self.params.gop_size == 1 {
            // Legacy path — bit-identical to pre-T2.4. Each frame is
            // an isolated I-frame, no inter-frame prediction state.
            let (ac_sign, golomb_tail) = probe_one_keyframe(yuv_i420, self.params)?;
            let this_gop_cover = ac_sign + golomb_tail;
            self.ac_sign_bits += ac_sign;
            self.golomb_tail_bits += golomb_tail;
            self.cover_bits += this_gop_cover;
            self.per_gop_cover_bits.push(this_gop_cover);
            self.n_gops += 1;
        } else {
            // T2.4 per-GOP path — buffer YUV, flush at boundary.
            self.current_gop_yuv.extend_from_slice(yuv_i420);
            self.frames_in_current_gop += 1;
            if self.frames_in_current_gop >= self.params.gop_size {
                self.flush_current_gop()?;
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Av1CapacityProbeResult {
        // T2.4: drain any trailing partial GOP. At gop_size=1 this
        // can't happen (legacy path drains in push_frame). At
        // gop_size > 1 the clip may end mid-GOP; the partial GOP gets
        // encoded at its actual frame count (the encode does the same
        // — partial trailing GOPs are first-class for the streaming
        // session). Best-effort: if the flush errors, drop the
        // partial measurement rather than panic; this matches the
        // probe's "best-effort, fallback to even-split" contract.
        if self.params.gop_size > 1 && self.frames_in_current_gop > 0 {
            let _ = self.flush_current_gop();
        }
        Av1CapacityProbeResult {
            cover_bits: self.cover_bits,
            ac_sign_bits: self.ac_sign_bits,
            golomb_tail_bits: self.golomb_tail_bits,
            n_gops: self.n_gops,
            per_gop_cover_bits: self.per_gop_cover_bits,
        }
    }

    /// T2.4: encode the accumulated GOP at the real frame-type pattern
    /// and count Tier-1 tags on the resulting recording. Only called
    /// when `gop_size > 1`. Pulls the natural-encode-only path
    /// (`encode_gop_natural`) rather than `process_one_gop_pass1` —
    /// the probe doesn't need the J-UNIWARD `harvest_gop` cost compute,
    /// which is the dominant per-GOP CPU term in the streaming session
    /// and would erase most of T2.4's wall-clock win otherwise.
    fn flush_current_gop(&mut self) -> Result<(), Av1StegoError> {
        let per_frame = super::whole_video::encode_gop_natural(
            &self.current_gop_yuv,
            self.frames_in_current_gop,
            self.params,
        )?;

        let mut ac_sign = 0usize;
        let mut golomb_tail = 0usize;
        for (_natural, recording) in &per_frame {
            if let Some(tile) = recording.tiles.first() {
                for &tag in tile.bit_tags.iter() {
                    if tag == PHASM_TAG_AC_COEFF_SIGN {
                        ac_sign += 1;
                    } else if tag == PHASM_TAG_GOLOMB_TAIL_LSB {
                        golomb_tail += 1;
                    }
                }
            }
        }
        let this_gop_cover = ac_sign + golomb_tail;

        self.ac_sign_bits += ac_sign;
        self.golomb_tail_bits += golomb_tail;
        self.cover_bits += this_gop_cover;
        self.per_gop_cover_bits.push(this_gop_cover);
        self.n_gops += 1;

        self.current_gop_yuv.clear();
        self.frames_in_current_gop = 0;
        Ok(())
    }
}

impl Av1CapacityProbeResult {
    /// Convert the per-GOP cover-bit vector to a per-GOP byte-cap
    /// vector ready to thread into
    /// `Av1StreamingEncodeSession::plan_proportional`. Applies the same
    /// `STC_PAYLOAD_RATIO` + chunk_frame-v3 overheads
    /// `primary_max_message_bytes` applies whole-clip — just per GOP.
    ///
    /// Per-GOP overhead uses the **subsequent-chunk** v3 header size as a
    /// conservative-but-tight bound. The first stego GOP carries 4 extra
    /// bytes (clip-level `total_bytes`); planner allocators that thread
    /// these caps into `plan_proportional` are expected to deduct the
    /// extra 4 bytes from chunk-0's slot via the `+ FRAME_OVERHEAD` /
    /// clip-level safety margin downstream — same idiom v2 used for
    /// `primary_max_message_bytes`.
    ///
    /// Returns an empty vector if `per_gop_cover_bits` is empty (legacy
    /// probe didn't track it).
    pub fn per_gop_byte_caps(&self) -> Vec<usize> {
        self.per_gop_cover_bits
            .iter()
            .map(|&cb| {
                let bits = cb.saturating_mul(STC_PAYLOAD_RATIO_NUM) / STC_PAYLOAD_RATIO_DEN;
                (bits / 8).saturating_sub(CHUNK_FRAME_NEXT_HEADER_LEN_V3_1)
            })
            .collect()
    }
}

impl Av1CapacityProbeResult {
    /// Conservative primary payload byte cap:
    /// `(cover_bits × 0.15 / 8) - chunk_frame_v3_1_overhead(n_gops) - FRAME_OVERHEAD`.
    ///
    /// Chunk_frame v3.1 overhead is `10 + 6 × (n_gops - 1)` bytes
    /// (first-chunk clip header `u32 total_bytes + u32 m_total + u16 payload_len`
    /// + subsequent per-chunk `u32 m_total + u16 payload_len`). This is
    /// the upper bound under the worst case (every GOP is stego'd);
    /// balanced plans typically use fewer chunks and have lower real
    /// overhead.
    ///
    /// Returns 0 if any subtraction would underflow (degenerate
    /// tiny-cover cases). Callers can show "video too short" to the
    /// user without checking the components.
    pub fn primary_max_message_bytes(&self) -> usize {
        let primary_bits =
            self.cover_bits.saturating_mul(STC_PAYLOAD_RATIO_NUM) / STC_PAYLOAD_RATIO_DEN;
        let primary_bytes_raw = primary_bits / 8;
        primary_bytes_raw
            .saturating_sub(chunk_frame_v3_1_overhead_bytes(self.n_gops as usize))
            .saturating_sub(FRAME_OVERHEAD)
            // Never advertise more than `encode_payload` accepts.
            .min(MAX_INNER_PAYLOAD_BYTES)
    }

    /// Shadow collision formula. `m_max_bits ≤ sqrt(1024 × C / max(1, N-1))`,
    /// then bound by raw cover; minus worst-case RS parity tier (128 bytes).
    /// Per-shadow byte cap such that cascade succeeds across all N
    /// shadows. Returns 0 if cover is too small or n_shadows=0.
    ///
    /// Delegates to the free function [`av1_shadow_capacity_for_n`]
    /// so the one-shot capacity probe + the live HUD shadow-segment
    /// bar agree bit-for-bit on the formula.
    pub fn shadow_max_message_bytes(&self, n_shadows: u32) -> usize {
        av1_shadow_capacity_for_n(self.cover_bits, n_shadows)
    }
}

/// N-aware shadow byte cap for the AV1 streaming cover.
///
/// Constant-time at the Rust level — callers pass the raw Tier-1
/// cover-bit total they already know (from `Av1CapacityProbeResult.
/// cover_bits` or from the cached stratified-probe estimate
/// `cover_bits_mean_per_gop × n_gops`) and this returns the per-
/// shadow byte cap under the collision formula
/// `m_max_bits ≤ sqrt(1024 × C / max(1, N - 1))`, bounded by raw
/// cover, minus worst-case RS parity (128 B) + `FRAME_OVERHEAD`.
///
/// Mirror of H.264's `h264_shadow_capacity_for_n` (`core/src/codec/
/// h264/stego/oh264_capacity.rs:39-41`). The point of this helper is
/// that the mobile HUD can resize the shadow-segment bar when the
/// user adds / removes a shadow WITHOUT re-probing the clip —
/// capacity falls out from the already-known cover total in O(1).
///
/// Returns 0 if cover is too small for the cascade overhead, or if
/// `n_shadows == 0`.
pub fn av1_shadow_capacity_for_n(cover_bits: usize, n_shadows: u32) -> usize {
    if n_shadows == 0 {
        return 0;
    }
    let denom = (n_shadows.saturating_sub(1)).max(1) as usize;
    let m_max_bits_sq = 1024usize.saturating_mul(cover_bits) / denom;
    let m_max_bits = (m_max_bits_sq as f64).sqrt() as usize;
    let m_max_bits = m_max_bits.min(cover_bits);
    let m_max_bytes = m_max_bits / 8;
    m_max_bytes
        .saturating_sub(128) // worst-case RS parity tier
        .saturating_sub(FRAME_OVERHEAD)
}

/// One-shot capacity API. Buffers all `n_frames` from `yuv`,
/// runs probe session, returns full capacity info struct including
/// per-domain breakdown and convenience shadow-n=1 byte cap.
///
/// `yuv` is laid out as `n_frames` consecutive I420 frames. Caller
/// is responsible for matching width/height to the YUV.
pub fn av1_capacity(
    yuv: &[u8],
    params: Av1StreamingEncodeParams,
) -> Result<Av1CapacityInfo, Av1StegoError> {
    let frame_size = expected_i420_size(params.width, params.height);
    let total_size = frame_size.saturating_mul(params.total_frames_hint as usize);
    if yuv.len() != total_size {
        return Err(Av1StegoError::InvalidPacket(format!(
            "av1_capacity: yuv length {} != expected {} ({} frames × {} bytes)",
            yuv.len(),
            total_size,
            params.total_frames_hint,
            frame_size
        )));
    }
    let mut probe = Av1StreamingProbeSession::create(params)?;
    for i in 0..params.total_frames_hint as usize {
        let start = i * frame_size;
        let end = start + frame_size;
        probe.push_frame(&yuv[start..end])?;
    }
    let result = probe.finish();
    Ok(Av1CapacityInfo {
        cover_size_bits: result.cover_bits,
        primary_max_message_bytes: result.primary_max_message_bytes(),
        per_domain_bits: Av1PerDomainBits {
            ac_sign: result.ac_sign_bits,
            golomb_tail: result.golomb_tail_bits,
            _tier2_reserved: 0,
            _tier3_reserved: 0,
        },
        n_gops: result.n_gops,
        shadow_max_message_bytes_n1: result.shadow_max_message_bytes(1),
    })
}

/// One-shot multi-shadow capacity API.
pub fn av1_shadow_capacity(
    yuv: &[u8],
    params: Av1StreamingEncodeParams,
    n_shadows: u32,
) -> Result<Av1ShadowCapacityInfo, Av1StegoError> {
    let frame_size = expected_i420_size(params.width, params.height);
    let total_size = frame_size.saturating_mul(params.total_frames_hint as usize);
    if yuv.len() != total_size {
        return Err(Av1StegoError::InvalidPacket(format!(
            "av1_shadow_capacity: yuv length {} != expected {} ({} frames × {} bytes)",
            yuv.len(),
            total_size,
            params.total_frames_hint,
            frame_size
        )));
    }
    let mut probe = Av1StreamingProbeSession::create(params)?;
    for i in 0..params.total_frames_hint as usize {
        let start = i * frame_size;
        let end = start + frame_size;
        probe.push_frame(&yuv[start..end])?;
    }
    let result = probe.finish();
    let max_message_bytes = result.shadow_max_message_bytes(n_shadows);
    let primary_residual_bytes = result
        .primary_max_message_bytes()
        .saturating_sub(max_message_bytes.saturating_mul(n_shadows as usize));
    Ok(Av1ShadowCapacityInfo {
        cover_size_bits: result.cover_bits,
        max_message_bytes,
        primary_residual_bytes,
        shadow_slots_total: result.n_gops,
        n_shadows,
    })
}

#[derive(Debug, Clone, Copy)]
pub struct Av1CapacityInfo {
    pub cover_size_bits: usize,
    pub primary_max_message_bytes: usize,
    pub per_domain_bits: Av1PerDomainBits,
    pub n_gops: u32,
    pub shadow_max_message_bytes_n1: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Av1PerDomainBits {
    pub ac_sign: usize,
    pub golomb_tail: usize,
    pub _tier2_reserved: usize,
    pub _tier3_reserved: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct Av1ShadowCapacityInfo {
    pub cover_size_bits: usize,
    pub max_message_bytes: usize,
    pub primary_residual_bytes: usize,
    pub shadow_slots_total: u32,
    pub n_shadows: u32,
}

fn expected_i420_size(width: u32, height: u32) -> usize {
    let y = (width as usize) * (height as usize);
    let uv = ((width as usize) / 2) * ((height as usize) / 2);
    y + 2 * uv
}

// ────────────────────────────────────────────────────────────────
// T2.6 — Stratified capacity probe
// ────────────────────────────────────────────────────────────────

/// T2.6 primitive — encode exactly the GOPs at `positions` through
/// `source`, run the real inter-frame-shaped per-GOP encode, and
/// return `(gop_idx, cover_bits)` samples ready to feed into
/// [`crate::stego::balanced_allocation::plan_safe_balanced`].
///
/// **Speed:** probe wall-clock is `positions.len() / n_gops` of the
/// full-corpus `av1_capacity` probe (which is itself ~equal to a full
/// encode's wall-clock). For K=8 against a 60-GOP clip that's ~13%
/// of the full probe — the main lever that makes `plan_safe_balanced`'s
/// statistical lower bound usable at mobile cost.
///
/// **Memory:** holds exactly one GOP's YUV at a time (`source.gop_yuv`
/// is dropped between iterations). Peak = `gop_size × frame_size` ≈
/// 93 MB at 1080p × gop=30, independent of clip length.
///
/// **Contract:** `positions` must be strictly increasing (the
/// `Av1GopYuvSource` trait requires monotonically-increasing
/// `gop_index` within a sweep — see trait doc on `whole_video.rs`).
/// Non-monotonic positions return `InvalidPacket`.
///
/// Caller is responsible for `source.rewind()` before the first call
/// (same contract as `av1_encode_streaming` / shadow entry points).
pub fn av1_probe_gops_at(
    source: &mut dyn super::whole_video::Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    positions: &[u32],
) -> Result<Vec<(usize, usize)>, Av1StegoError> {
    av1_probe_gops_at_with_progress(
        source,
        n_frames,
        params,
        positions,
        &mut |_, _, _| {},
    )
}

/// Same as [`av1_probe_gops_at`] but calls `progress_fn(gops_done,
/// gops_total, cover_bits_so_far)` AFTER each probed GOP's cover-bit
/// count is in. `gops_total` is `positions.len()` (what the probe is
/// targeting, not `n_frames.div_ceil(gop_size)`); `cover_bits_so_far`
/// is the running sum across all GOPs probed to date.
///
/// Mirrors H.264's `PhasmProbeProgressListener` callback shape so the
/// Swift / Kotlin wrappers that drive both codecs can share a single
/// `onProbeProgress(done, total, coverBits)` signature. The UI
/// extrapolates a partial capacity estimate from `(coverBits × total /
/// done)` — see the H.264-side listener doc for the formula.
///
/// `progress_fn` is called synchronously on the probing thread; FFI
/// trampolines are expected to marshal to a UI thread if needed.
pub fn av1_probe_gops_at_with_progress(
    source: &mut dyn super::whole_video::Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    positions: &[u32],
    progress_fn: &mut dyn FnMut(u32, u32, u64),
) -> Result<Vec<(usize, usize)>, Av1StegoError> {
    if n_frames == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_probe_gops_at: n_frames == 0".into(),
        ));
    }
    if params.width == 0 || params.height == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_probe_gops_at: width/height must be > 0".into(),
        ));
    }
    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);
    let frame_size = expected_i420_size(params.width, params.height);

    // Monotonicity gate — the trait requires strictly-increasing GOP
    // indices within one sweep. Also reject out-of-range.
    for pair in positions.windows(2) {
        if pair[0] >= pair[1] {
            return Err(Av1StegoError::InvalidPacket(format!(
                "av1_probe_gops_at: positions must be strictly \
                 increasing (got {:?} then {:?})",
                pair[0], pair[1],
            )));
        }
    }
    if let Some(&last) = positions.last() {
        if last >= n_gops {
            return Err(Av1StegoError::InvalidPacket(format!(
                "av1_probe_gops_at: position {last} >= n_gops {n_gops}",
            )));
        }
    }

    let total = positions.len() as u32;
    let mut cover_bits_running: u64 = 0;
    let mut done: u32 = 0;
    let mut samples: Vec<(usize, usize)> = Vec::with_capacity(positions.len());
    for &gop_idx in positions {
        let frames_in_this_gop = {
            let start = gop_idx * gop_size;
            let end = (start + gop_size).min(n_frames);
            end - start
        };
        if frames_in_this_gop == 0 {
            continue; // past end of clip
        }
        let yuv = source.gop_yuv(gop_idx)?;
        let expected = (frames_in_this_gop as usize) * frame_size;
        if yuv.len() != expected {
            return Err(Av1StegoError::InvalidPacket(format!(
                "av1_probe_gops_at: source returned {} bytes for GOP {gop_idx}, \
                 expected {expected} ({frames_in_this_gop}f × {frame_size})",
                yuv.len(),
            )));
        }
        let per_frame = super::whole_video::encode_gop_natural(
            &yuv,
            frames_in_this_gop,
            params,
        )?;
        let mut cover_bits = 0usize;
        for (_natural, recording) in &per_frame {
            if let Some(tile) = recording.tiles.first() {
                for &tag in tile.bit_tags.iter() {
                    if tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB {
                        cover_bits += 1;
                    }
                }
            }
        }
        samples.push((gop_idx as usize, cover_bits));
        cover_bits_running = cover_bits_running.saturating_add(cover_bits as u64);
        done = done.saturating_add(1);
        progress_fn(done, total, cover_bits_running);
    }
    Ok(samples)
}

/// Stratified-sample capacity estimate for a streaming
/// [`Av1GopYuvSource`](super::whole_video::Av1GopYuvSource). Pulls
/// `calibration.k_min` full GOPs at stratified positions (at least one
/// whenever the clip has a full GOP) plus the partial trailing GOP, if
/// any, encodes them naturally (no STC) and extrapolates the whole-clip
/// capacity: the full GOPs via the same 99.9%-confidence statistical
/// lower bound + `pessimism_factor` the planner uses, the tail from its
/// exact cover (see [`Av1StratifiedCapacity`]).
///
/// **Why this exists:** `av1_capacity` requires the entire clip in
/// RAM and probes every GOP (wall-clock ≈ full encode). Stratified
/// probing samples a small K, extrapolates with a confidence bound,
/// and runs through the per-GOP streaming source. On a 60-GOP /
/// 10-second clip with K=8, that's ~13% of the full probe wall-clock
/// and the memory peak drops from O(clip) to O(gop_size).
///
/// Caller is responsible for `source.rewind()` before invocation and
/// may call this function multiple times at different K values
/// (passing a progressively larger `calibration.k_min`) when more
/// precision is needed — mirrors the planner's `NeedMoreSamples`
/// escalation shape at the capacity layer.
pub fn av1_capacity_stratified(
    source: &mut dyn super::whole_video::Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    calibration: &AllocationCalibration,
) -> Result<Av1StratifiedCapacity, Av1StegoError> {
    av1_capacity_stratified_with_progress(
        source,
        n_frames,
        params,
        calibration,
        &mut |_, _, _| {},
    )
}

/// Same as [`av1_capacity_stratified`] but calls `progress_fn(gops_done,
/// gops_total, cover_bits_so_far)` AFTER each probed GOP. Mirrors
/// H.264's `PhasmProbeProgressListener` callback shape so the mobile
/// encode screen can show a "narrowing-down" primary estimate during
/// the probe wall-clock instead of a plain "Calculating…" state for
/// the whole K/n_gops × encode seconds.
///
/// The UI extrapolates a partial-estimate primary cap via
///   `est_total_cover ≈ cover_bits_so_far × gops_total / gops_done`
/// and then re-applies the same STC_PAYLOAD_RATIO + chunk_frame
/// overhead subtraction the final result uses — see the H.264 listener
/// doc for the formula + the Swift / Kotlin wrappers for how both
/// codecs share one progress interface.
///
/// Behaviour on the final GOP: `progress_fn(k, k, final_cover_total)`
/// fires BEFORE this function returns the full result; a caller that
/// only reads the final progress event has an estimate that converges
/// to the returned `Av1StratifiedCapacity.extrapolated_cover_bits_total`
/// modulo the pessimism-factor adjustment the progress estimate
/// deliberately omits (it reports raw running cover, letting the UI
/// apply codec-specific derived values on top).
pub fn av1_capacity_stratified_with_progress(
    source: &mut dyn super::whole_video::Av1GopYuvSource,
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    calibration: &AllocationCalibration,
    progress_fn: &mut dyn FnMut(u32, u32, u64),
) -> Result<Av1StratifiedCapacity, Av1StegoError> {
    if n_frames == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_capacity_stratified: n_frames == 0".into(),
        ));
    }
    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);
    // A partial trailing GOP is not a sample of the full-GOP population: it
    // holds `n_frames % gop_size` frames and correspondingly less cover.
    // Treated as one, it widened the lower bound for every GOP (iphone5,
    // 363 vs 360 frames: 341 828 vs 552 612 B advertised). It is probed
    // exactly instead and added on top of the full-GOP extrapolation.
    let (positions, n_full, tail_gop) =
        stratified_probe_positions(n_frames, gop_size, calibration.k_min);

    let samples = av1_probe_gops_at_with_progress(
        source, n_frames, params, &positions, progress_fn,
    )?;
    if samples.is_empty() {
        return Err(Av1StegoError::InvalidPacket(
            "av1_capacity_stratified: no samples collected".into(),
        ));
    }
    let full: Vec<usize> = samples
        .iter()
        .filter(|&&(g, _)| g < n_full)
        .map(|&(_, c)| c)
        .collect();
    let tail_cover = tail_gop.and_then(|t| {
        samples.iter().find(|&&(g, _)| g == t).map(|&(_, c)| c)
    });
    let (mu, cap_per_gop_lower, cap_per_gop_safe, extrapolated_cover_bits) =
        stratified_cover_estimate(&full, n_full, tail_cover, calibration);

    // Primary/shadow byte caps mirror `Av1CapacityProbeResult` math.
    let primary_bits = extrapolated_cover_bits
        .saturating_mul(STC_PAYLOAD_RATIO_NUM)
        / STC_PAYLOAD_RATIO_DEN;
    let primary_max_message_bytes = (primary_bits / 8)
        .saturating_sub(chunk_frame_v3_1_overhead_bytes(n_gops as usize))
        .saturating_sub(FRAME_OVERHEAD)
        // Never advertise more than `encode_payload` accepts.
        .min(MAX_INNER_PAYLOAD_BYTES);

    // Shadow (N=1) via collision formula.
    let shadow_n1_bits = (extrapolated_cover_bits as f64).min(
        (1024.0 * extrapolated_cover_bits as f64).sqrt(),
    ) as usize;
    let shadow_max_message_bytes_n1 = (shadow_n1_bits / 8)
        .saturating_sub(128) // worst-case RS parity tier
        .saturating_sub(FRAME_OVERHEAD);

    Ok(Av1StratifiedCapacity {
        samples,
        n_gops,
        cover_bits_mean_per_gop: mu as usize,
        cover_bits_lower_99_9_per_gop: cap_per_gop_lower as usize,
        cover_bits_pessimism_adjusted_per_gop: cap_per_gop_safe as usize,
        extrapolated_cover_bits_total: extrapolated_cover_bits,
        primary_max_message_bytes,
        shadow_max_message_bytes_n1,
    })
}

/// The GOPs [`av1_capacity_stratified_with_progress`] probes, ascending:
/// `k_min` stratified full GOPs, plus the partial trailing GOP if the clip
/// has one. Returns `(positions, n_full, tail_gop)`.
fn stratified_probe_positions(
    n_frames: u32,
    gop_size: u32,
    k_min: usize,
) -> (Vec<u32>, usize, Option<usize>) {
    let n_full = (n_frames / gop_size) as usize;
    let tail_gop = (n_frames % gop_size != 0).then_some(n_full);
    // At least one full GOP whenever there is one: an estimate from the
    // partial tail alone (or from nothing) says nothing about the clip.
    let k_target = k_min.max(1).min(n_full);
    let mut positions: Vec<u32> = stratified_positions(k_target, n_full)
        .into_iter()
        .map(|p| p as u32)
        .collect();
    if let Some(t) = tail_gop {
        positions.push(t as u32); // the last index, so `positions` stays ascending
    }
    (positions, n_full, tail_gop)
}

/// Result of [`av1_capacity_stratified`]: the raw probe samples and the
/// capacity estimate derived from them, for UI reporting (the encoder
/// measures every GOP itself in Pass 1 and plans from that).
///
/// The clip's `n_full = n_frames / gop_size` full GOPs are the sampled
/// population; a partial trailing GOP is not a sample of it (it holds
/// fewer frames and correspondingly less cover), so it is probed exactly
/// and added on top. The per-GOP statistics below describe full GOPs only.
#[derive(Debug, Clone)]
pub struct Av1StratifiedCapacity {
    /// `(gop_idx, cover_bits)` for every probed GOP, ascending: the
    /// stratified full GOPs, then the partial trailing GOP (index
    /// `n_full`) if the clip has one. Not a pure full-GOP sample — filter
    /// out the tail before treating it as one.
    pub samples: Vec<(usize, usize)>,
    /// Total GOPs in the clip, the partial trailing GOP included.
    pub n_gops: u32,
    /// Mean cover bits over the sampled full GOPs (0 when the clip is
    /// shorter than one GOP).
    pub cover_bits_mean_per_gop: usize,
    /// One-sided 99.9 % lower bound on a full GOP's cover bits, with the
    /// finite-population correction over `n_full` (0 when the clip is
    /// shorter than one GOP).
    pub cover_bits_lower_99_9_per_gop: usize,
    /// `cover_bits_lower_99_9_per_gop × pessimism_factor`.
    pub cover_bits_pessimism_adjusted_per_gop: usize,
    /// Whole-clip estimate: `(lower × n_full + tail_cover) ×
    /// pessimism_factor`, with `tail_cover` the trailing GOP's exact cover
    /// (0 without one).
    pub extrapolated_cover_bits_total: usize,
    /// Primary payload byte cap from `extrapolated_cover_bits_total`: the
    /// STC payload ratio, minus the chunk-frame overhead for `n_gops` GOPs
    /// and the frame overhead, capped at `MAX_INNER_PAYLOAD_BYTES`.
    pub primary_max_message_bytes: usize,
    /// Shadow (N = 1) byte cap from `extrapolated_cover_bits_total`, via
    /// the collision formula `min(n, √(1024·n))` bits, minus the
    /// worst-case RS parity and the frame overhead.
    pub shadow_max_message_bytes_n1: usize,
}

/// Whole-clip cover estimate from stratified full-GOP samples, plus the
/// exactly probed partial tail GOP (`tail_cover`), if the clip has one.
///
/// Mirrors the statistical-lower-bound math of `plan_safe_balanced`
/// (balanced_allocation.rs): `lower = (μ − t·σ/√K · fpc).max(0)` over the
/// `n_full` full GOPs, `safe = lower × pessimism_factor`, and
/// `extrapolated = (lower × n_full + tail_cover) × pessimism_factor`.
/// Returns `(mu, lower, safe, extrapolated)`. With no full-GOP samples (a
/// clip shorter than one GOP) the estimate is the tail alone.
fn stratified_cover_estimate(
    full_samples: &[usize],
    n_full: usize,
    tail_cover: Option<usize>,
    cal: &AllocationCalibration,
) -> (f64, f64, f64, usize) {
    let k = full_samples.len();
    let (mu, lower) = if k == 0 {
        (0.0, 0.0)
    } else {
        let caps: Vec<f64> = full_samples.iter().map(|&c| c as f64).collect();
        let mu = caps.iter().sum::<f64>() / k as f64;
        let var = if k > 1 {
            caps.iter().map(|&c| (c - mu).powi(2)).sum::<f64>() / (k - 1) as f64
        } else {
            mu // single-sample degenerate — force a pessimistic estimate
        };
        let fpc = if n_full > 1 {
            ((n_full.saturating_sub(k)) as f64 / (n_full - 1) as f64)
                .max(0.0)
                .sqrt()
        } else {
            0.0
        };
        let lower = (mu - t_critical_99_9(k) * var.sqrt() / (k as f64).sqrt() * fpc).max(0.0);
        (mu, lower)
    };
    let safe = lower * cal.pessimism_factor;
    let extrapolated =
        ((lower * n_full as f64 + tail_cover.unwrap_or(0) as f64) * cal.pessimism_factor) as usize;
    (mu, lower, safe, extrapolated)
}

/// Stratified GOP positions: k evenly-spaced points at stratum
/// midpoints in [0, n_gops). Deterministic — matches
/// `balanced_allocation::stratified_positions` so the probe sites
/// agree with the planner's `NeedMoreSamples` request shape.
fn stratified_positions(k: usize, n_gops: usize) -> Vec<usize> {
    if k == 0 || n_gops == 0 {
        return Vec::new();
    }
    let k = k.min(n_gops);
    let mut out = Vec::with_capacity(k);
    let stride = n_gops as f64 / k as f64;
    for i in 0..k {
        let pos = (i as f64 + 0.5) * stride;
        out.push((pos as usize).min(n_gops - 1));
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// 99.9% one-sided t-critical values — mirror of
/// `balanced_allocation::t_critical_99_9`. Duplicated here to avoid a
/// public export from the stego module; same values, same source.
fn t_critical_99_9(k: usize) -> f64 {
    if k <= 1 {
        return 318.31; // df=0 degenerate — use a very large t
    }
    match k - 1 {
        1 => 318.31,
        2 => 22.33,
        3 => 10.21,
        4 => 7.17,
        5 => 5.89,
        6 => 5.21,
        7 => 4.79,
        8 => 4.50,
        9 => 4.30,
        10..=15 => 3.85,
        16..=30 => 3.46,
        31..=60 => 3.23,
        _ => 3.09, // asymptotic z
    }
}

/// Encode one keyframe naturally (no embed) and count Tier-1 cover
/// positions in the encoder-side recording. Returns
/// `(ac_sign_count, golomb_tail_count)`.
///
/// The natural-encode wall-clock dominates the probe cost; counting
/// is O(n) tags traversal after.
///
/// **Speed/sequence alignment (2026-10-18, probe-vs-encoder parity):**
/// applies the production stealth-locked encoder config
/// (`apply_phasm_av1_speed` → speed=9 default; `apply_phasm_av1_sequence`
/// → `enable_large_lru=true` default as of 2026-10-20, was `false`;
/// inert at speed ≥ 8 since LRF is disabled there — see
/// `apply_phasm_av1_sequence`'s doc comment for why we still set it).
/// Before this fix the probe
/// built an `EncoderConfig::default()` which inherited rav1e's default
/// speed=6, and every production encode downstream ran at speed=9 —
/// probe cover_bit counts systematically over-estimated the real
/// encoder's yield (speed=6 does less RDO rejection → more non-zero
/// AC → more Tier-1 cover). The `pessimism_factor=0.5` constant was
/// absorbing this preset-mismatch error on top of its intended
/// within-clip variability role.
fn probe_one_keyframe(
    yuv_i420: &[u8],
    params: Av1StreamingEncodeParams,
) -> Result<(usize, usize), Av1StegoError> {
    let mut config = EncoderConfig {
        width: params.width as usize,
        height: params.height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: params.quantizer,
        ..Default::default()
    };
    super::whole_video::apply_phasm_av1_speed(&mut config);
    let config = Arc::new(config);
    let mut sequence = Sequence::new(&config);
    super::whole_video::apply_phasm_av1_sequence(&mut sequence);
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
    let (_packet, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);

    // Count Tier-1 tags. AV1 v0.6 ships single-tile only, so the
    // first tile holds everything.
    let tile = recording
        .tiles
        .first()
        .ok_or(Av1StegoError::EmptyRecording)?;
    let mut ac_sign = 0usize;
    let mut golomb_tail = 0usize;
    for &tag in tile.bit_tags.iter() {
        if tag == PHASM_TAG_AC_COEFF_SIGN {
            ac_sign += 1;
        } else if tag == PHASM_TAG_GOLOMB_TAIL_LSB {
            golomb_tail += 1;
        }
    }
    Ok((ac_sign, golomb_tail))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tail_does_not_widen_the_lower_bound() {
        let (_, lower, _, extrapolated) = stratified_cover_estimate(
            &[5_000_000; 8],
            12,
            Some(642_285),
            &AllocationCalibration::AV1_1080P_QP30,
        );
        assert_eq!(lower, 5_000_000.0);
        assert_eq!(extrapolated, ((5_000_000 * 12 + 642_285) as f64 * 0.5) as usize);
    }

    #[test]
    fn adding_a_tail_never_lowers_the_estimate() {
        let cal = AllocationCalibration::AV1_1080P_QP30;
        let samples = [5_000_000usize; 8];
        let without = stratified_cover_estimate(&samples, 12, None, &cal).3;
        let with = stratified_cover_estimate(&samples, 12, Some(1), &cal).3;
        assert!(with >= without, "{with} < {without}");
    }

    /// A clip shorter than one GOP has only a tail: estimate from it alone.
    #[test]
    fn probe_samples_a_full_gop_even_with_k_min_zero() {
        // k_min 0 must not leave a clip with full GOPs estimated from its
        // partial tail alone (or from nothing).
        let (positions, n_full, tail) = stratified_probe_positions(95, 30, 0);
        assert_eq!((n_full, tail), (3, Some(3)));
        assert!(
            positions.iter().any(|&g| (g as usize) < n_full),
            "no full GOP sampled: {positions:?}"
        );
        let (positions, _, _) = stratified_probe_positions(90, 30, 0);
        assert_eq!(positions.len(), 1, "one full GOP, no tail: {positions:?}");
        // No full GOP at all: only the tail is probed.
        assert_eq!(stratified_probe_positions(20, 30, 0).0, vec![0]);
    }

    #[test]
    fn clip_shorter_than_one_gop_is_estimated_from_its_tail() {
        let cal = AllocationCalibration::AV1_1080P_QP30;
        let extrapolated = stratified_cover_estimate(&[], 0, Some(400_000), &cal).3;
        assert_eq!(extrapolated, (400_000.0 * 0.5) as usize);
    }
}
