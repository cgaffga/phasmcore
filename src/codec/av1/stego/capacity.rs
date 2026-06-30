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

use crate::stego::chunk_frame::{
    CHUNK_FRAME_FIRST_HEADER_LEN_V3_1, CHUNK_FRAME_NEXT_HEADER_LEN_V3_1,
};
use crate::stego::frame::FRAME_OVERHEAD;

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
const STC_PAYLOAD_RATIO_NUM: usize = 15;
const STC_PAYLOAD_RATIO_DEN: usize = 100;

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
    }

    /// Shadow collision formula. `m_max_bits ≤ sqrt(1024 × C / max(1, N-1))`,
    /// then bound by raw cover; minus worst-case RS parity tier (128 bytes).
    /// Per-shadow byte cap such that cascade succeeds across all N
    /// shadows. Returns 0 if cover is too small or n_shadows=0.
    pub fn shadow_max_message_bytes(&self, n_shadows: u32) -> usize {
        if n_shadows == 0 {
            return 0;
        }
        let denom = (n_shadows.saturating_sub(1)).max(1) as usize;
        let m_max_bits_sq = 1024usize.saturating_mul(self.cover_bits) / denom;
        let m_max_bits = (m_max_bits_sq as f64).sqrt() as usize;
        let m_max_bits = m_max_bits.min(self.cover_bits);
        let m_max_bytes = m_max_bits / 8;
        m_max_bytes
            .saturating_sub(128) // worst-case RS parity tier
            .saturating_sub(FRAME_OVERHEAD)
    }
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

/// Encode one keyframe naturally (no embed) and count Tier-1 cover
/// positions in the encoder-side recording. Returns
/// `(ac_sign_count, golomb_tail_count)`.
///
/// The natural-encode wall-clock dominates the probe cost; counting
/// is O(n) tags traversal after.
fn probe_one_keyframe(
    yuv_i420: &[u8],
    params: Av1StreamingEncodeParams,
) -> Result<(usize, usize), Av1StegoError> {
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
