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
    allocate_chunks_concentrate_tail, build_chunk_frame_v3_1, build_first_chunk_frame_v3_1,
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
        let plaintext = self.finish()?;
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
///
/// MOBOOM.T3.1: visibility raised to `pub(super)` so the new
/// `av1_encode_streaming` no-shadow streaming entry in `whole_video.rs`
/// can call this directly (must mirror the session's per-GOP encode
/// byte-for-byte for the T3.2 byte-identity gate).
pub(super) fn encode_one_gop_multi(
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
    super::whole_video::apply_phasm_av1_speed(&mut config);
    let config = Arc::new(config);
    let mut sequence = Sequence::new(&config);
    super::whole_video::apply_phasm_av1_sequence(&mut sequence);
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
/// MOBOOM.T3.1: visibility raised to `pub(super)` so the new
/// `av1_encode_streaming` no-shadow streaming entry can mirror the
/// session's gop_size=1 path byte-for-byte (T3.2 byte-identity gate).
pub(super) fn encode_one_keyframe(
    yuv_i420: &[u8],
    params: Av1StreamingEncodeParams,
) -> Result<(Vec<u8>, PhasmFrameRecording<u8>), Av1StegoError> {
    let mut config_inner = EncoderConfig {
        width: params.width as usize,
        height: params.height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: params.quantizer,
        ..Default::default()
    };
    super::whole_video::apply_phasm_av1_speed(&mut config_inner);
    let config = Arc::new(config_inner);
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
