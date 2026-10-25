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
//! CLI keeps both entry points. The v3.1 wire format has zero
//! backwards compatibility with v2 chunk_frame (no shipped AV1
//! installed base on v2).
//!
//! ## v3.1 length-strict invariants
//!
//! Both `parse_first_chunk_frame_v3_1` and `parse_chunk_frame_v3_1` are
//! length-strict: extracted bytes must equal `header_len + payload_len`
//! exactly. Combined with the explicit `m_total` checksum each header now
//! carries, this pins every STC extract to the encoder's exact w-class —
//! a 1/2³² reject that replaces v3's m_total brute-force search (#888).


use crate::stego::{crypto, frame};

use super::orchestrator::Av1StegoError;
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
        self.primary_plaintext()
    }

    /// [`finish`](Self::finish) without consuming the session, so
    /// [`finish_payload`](Self::finish_payload) can go on to the shadows.
    fn primary_plaintext(&self) -> Result<Vec<u8>, Av1StegoError> {
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

        // v2: every GOP has its own keyed cover order and H-hat seed,
        // derived from the structural master and the GOP index.
        let master = crypto::derive_structural_key(&self.passphrase)?;

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
            // A cancel (the bridges run decodes on the Decode channel) stops
            // at the next GOP instead of decoding the whole clip.
            crate::stego::progress::check_cancelled().map_err(Av1StegoError::Stego)?;
            let (cover_bits, hhat_seed) =
                gop_stc_input(&harvest_cover_bits_from_stego(slab)?, &master, gop_idx as u32);
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

        // Video bound, not the image one: a whole-clip frame is split into
        // per-GOP chunks, so it may legitimately exceed `MAX_FRAME_BYTES`.
        let parsed = frame::parse_frame_with_max(&assembled, frame::VIDEO_MAX_FRAME_BYTES)
            .map_err(Av1StegoError::Stego)?;
        let plaintext = crypto::decrypt(
            &parsed.ciphertext,
            &self.passphrase,
            &parsed.salt,
            &parsed.nonce,
        )
        .map_err(Av1StegoError::Stego)?;
        Ok(plaintext)
    }

    /// Same as `finish`, but interprets the decrypted bytes as a
    /// `payload::encode_payload` frame and returns `(text, files)`.
    ///
    /// Use this when the encoder produced the primary message via
    /// `payload::encode_payload(text, files)` — the canonical mobile
    /// path (matching the H.264 streaming session's payload framing).
    /// Falls back to `text-only` if the plaintext doesn't parse as a
    /// payload frame (treats bytes as raw UTF-8 — only happens for
    /// pre-FILE encoders that pushed raw text directly).
    pub fn finish_primary_payload(
        self,
    ) -> Result<crate::stego::payload::PayloadData, Av1StegoError> {
        Self::primary_payload(self.primary_plaintext()?)
    }

    /// The message the session's passphrase opens, whichever layer it is:
    /// the primary, else the whole-video shadow (what the encoders write
    /// today), else a per-GOP shadow (older encodes). H.264's decode
    /// session falls back to its shadows the same way. This is the entry
    /// a decoder that is only given a passphrase should call —
    /// [`finish_primary_payload`](Self::finish_primary_payload) alone can
    /// never read a shadow back.
    ///
    /// A cancel stops at once; if no layer opens, the primary's error is
    /// returned.
    pub fn finish_payload(self) -> Result<crate::stego::payload::PayloadData, Av1StegoError> {
        use crate::stego::error::StegoError;
        let primary_err = match self.primary_plaintext() {
            Ok(plaintext) => return Self::primary_payload(plaintext),
            Err(Av1StegoError::Stego(StegoError::Cancelled)) => {
                return Err(Av1StegoError::Stego(StegoError::Cancelled));
            }
            Err(e) => e,
        };
        for shadow in [Self::finish_shadow_whole_video] {
            match shadow(&self, &self.passphrase) {
                Ok(payload) => return Ok(payload),
                Err(StegoError::Cancelled) => {
                    return Err(Av1StegoError::Stego(StegoError::Cancelled));
                }
                Err(_) => {}
            }
        }
        Err(primary_err)
    }

    /// Parse a decrypted primary as an `encode_payload` frame, falling
    /// back to raw UTF-8 text for pre-FILE encoders.
    fn primary_payload(
        plaintext: Vec<u8>,
    ) -> Result<crate::stego::payload::PayloadData, Av1StegoError> {
        match crate::stego::payload::decode_payload(&plaintext) {
            Ok(pd) => Ok(pd),
            Err(_) => {
                let text = String::from_utf8(plaintext)
                    .map_err(|_| Av1StegoError::ExtractionFailed)?;
                Ok(crate::stego::payload::PayloadData {
                    text,
                    files: Vec::new(),
                })
            }
        }
    }


    /// Extract a whole-video shadow message.
    ///
    /// Harvests the UNION cover across all GOPs and runs shadow extract
    /// once (the only shadow scope; the per-GOP one was v1 and is gone). Required for shadows
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
        // The union Tier-1 cover (AC_COEFF_SIGN + GOLOMB_TAIL_LSB) across
        // all GOPs in emit order — the cover the encoder's verify gate
        // operated on. Harvested GOP by GOP (each starts at a keyframe, and
        // the union is the per-GOP covers concatenated in GOP order, exactly
        // as the encoder lays it out) so a cancel stops at the next GOP
        // instead of after decoding the whole clip. A stream the walker
        // cannot split is harvested in one walk, as before.
        let to_stego = |e: super::orchestrator::Av1StegoError| match e {
            super::orchestrator::Av1StegoError::Stego(s) => s,
            _ => crate::stego::error::StegoError::FrameCorrupted,
        };
        // v2: each position's damage bucket comes from the same decode, per
        // GOP (frame index and GOP length are GOP-relative), spec § 6.
        let gop_slabs = split_av1_into_gops(&self.accumulator);
        let slabs: Vec<&[u8]> = if gop_slabs.is_empty() {
            vec![&self.accumulator]
        } else {
            gop_slabs.iter().map(|s| s.as_slice()).collect()
        };
        let (mut cover, mut buckets) = (Vec::new(), Vec::new());
        for slab in slabs {
            crate::stego::progress::check_cancelled()?;
            let (c, b) = super::shadow::cover_and_buckets_from_stego(slab).map_err(to_stego)?;
            cover.extend_from_slice(&c);
            buckets.extend_from_slice(&b);
        }
        super::shadow::av1_shadow_extract(&cover, &buckets, shadow_pass)
    }
}

/// v2 STC input for one GOP: the slab's canonical cover gathered through
/// the GOP's keyed order (`stego::cover_order`, spec § 3), and the GOP's
/// own H-hat seed. The order spans the whole GOP cover, as on the encode
/// side.
pub(crate) fn gop_stc_input(cover: &[u8], master: &[u8; 64], gop: u32) -> (Vec<u8>, [u8; 32]) {
    use crate::stego::cover_order::{gop_cover_order, gop_seeds, VideoCodec};
    let seeds = gop_seeds(master, VideoCodec::Av1, gop);
    let order = gop_cover_order(cover.len(), &seeds.order);
    (order.iter().map(|&i| cover[i as usize]).collect(), seeds.hhat)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::cover_order::{gop_cover_order, gop_seeds, VideoCodec};

    fn master(fill: u8) -> [u8; 64] {
        std::array::from_fn(|i| fill.wrapping_add(i as u8))
    }

    /// The decoder's STC input is the slab's canonical cover gathered
    /// through the GOP's keyed order, with that GOP's own H-hat seed.
    #[test]
    fn gop_stc_input_is_the_keyed_gather() {
        let cover: Vec<u8> = (0..500u32).map(|i| (i * 7 % 3 == 0) as u8).collect();
        let m = master(5);
        let (perm, hhat) = gop_stc_input(&cover, &m, 3);
        let seeds = gop_seeds(&m, VideoCodec::Av1, 3);
        let want: Vec<u8> =
            gop_cover_order(cover.len(), &seeds.order).iter().map(|&i| cover[i as usize]).collect();
        assert_eq!(perm, want);
        assert_eq!(hhat, seeds.hhat);
        assert_ne!(perm, cover, "the order must not be the identity");
    }

    #[test]
    fn gop_stc_input_differs_per_gop() {
        let cover: Vec<u8> = (0..500u32).map(|i| (i % 5 == 0) as u8).collect();
        let m = master(9);
        let (p0, h0) = gop_stc_input(&cover, &m, 0);
        let (p1, h1) = gop_stc_input(&cover, &m, 1);
        assert_ne!(h0, h1, "every GOP gets its own H-hat");
        assert_ne!(p0, p1);
    }
}
