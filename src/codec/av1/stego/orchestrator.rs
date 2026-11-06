// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Production av1_stego_encode / av1_stego_decode flow.
//!
//! Ties together:
//!   - phasm-rav1e [`encode_frame_with_phasm_tee`] (Pass 1: bytes + recorder)
//!   - phasm-core [`OverrideMap`] + [`replay_with_overrides`] (Pass 2)
//!   - phasm-core [`stc_embed`] + [`stc_extract`] (cover-bit → message)
//!   - phasm-core [`crypto`] (Argon2id + AES-GCM-SIV)
//!   - phasm-core [`frame`] (length-prefixed CRC-protected payload framing)
//!   - phasm-dav1d [`decode_with_recording`] (decoder side)
//!
//! Flow (see `docs/design/video/av1/streaming-session.md`):
//!
//! ## Encode
//! 1. Encrypt message via crypto::encrypt → (ciphertext, nonce, salt)
//! 2. Build payload frame via frame::build_frame (len + salt + nonce + CT + CRC)
//! 3. Encode via `encode_frame_with_phasm_tee` → OBU bytes + per-tile
//!    recorder + tile_group_offset
//! 4. Filter recorder bit_positions to AC_COEFF_SIGN positions
//! 5. Compute STC plan: embed payload bits into the AC cover bits → flips
//! 6. Build [`OverrideMap`] from the diff (cover_bit != stego_bit → flip)
//! 7. [`replay_with_overrides`] on the recorder storage → stego tile bytes
//!    (length-invariant 50/50 flips)
//! 8. Byte-splice stego tile bytes into natural OBU output at the
//!    tile_group_offset → final stego AV1 bytes
//!
//! ## Decode
//! 1. [`decode_with_recording`] on the stego AV1 bytes → all decoded
//!    50/50 positions + tags
//! 2. Filter to AC_COEFF_SIGN → stego AC bits
//! 3. Brute-force `w` candidates: for each, STC extract → parse_frame →
//!    decrypt. First valid hit wins (CRC + auth tag gate).
//! 4. Return plaintext.
//!
//! Initial AV1 scope: single-frame, single-tile, AC_COEFF_SIGN channel
//! only (see `docs/design/video/av1/channel-design.md`). Strict cursor
//! parity is the load-bearing equivalence — without it, the decoder
//! would extract at different positions than the encoder hid at.

#[cfg(feature = "av1-encoder")]
use phasm_rav1e::phasm_stego::PhasmFrameRecording;
#[cfg(feature = "av1-encoder")]
use crate::stego::cost::av1_uniward::FramePlanes;

#[cfg(feature = "av1-decoder")]
use crate::codec::av1::stego::decoder::{decode_with_recording, Av1DecodeError};

use crate::stego::error::StegoError;
#[cfg(feature = "av1-decoder")]
use crate::stego::stc::extract::stc_extract;
use crate::stego::stc::hhat;
#[cfg(feature = "av1-encoder")]
use crate::stego::crypto;
use crate::stego::frame;

/// STC matrix height — fixed at 7 (matches phasm-core's standard).
const STC_H: usize = 7;

/// Errors from the AV1 stego orchestrator.
#[derive(Debug)]
pub enum Av1StegoError {
    /// Crypto / framing / payload error from phasm-core's stego stack.
    Stego(StegoError),
    /// dav1d-side decode error.
    #[cfg(feature = "av1-decoder")]
    Decode(Av1DecodeError),
    /// Encoder-side recording has no tiles (unexpected for any valid encode).
    EmptyRecording,
    /// Not enough AC_COEFF_SIGN cover positions for the message size.
    MessageTooLarge { needed_bits: usize, available_bits: usize },
    /// STC embedding returned None (cover too small / w bound violated).
    StcInfeasible,
    /// `replay_with_overrides` produced a tile_group of a different size
    /// than the natural encode. Indicates a Tier 1 invariant violation
    /// (50/50 flips should preserve length).
    TileGroupSizeMismatch { expected: usize, actual: usize },
    /// Decoder side: no AC_COEFF_SIGN positions found in the bitstream.
    #[cfg(feature = "av1-decoder")]
    NoCoverPositions,
    /// All brute-force `w` candidates failed to extract a valid frame.
    #[cfg(feature = "av1-decoder")]
    ExtractionFailed,
    /// Streaming-session validation failed (bad params, chunk_idx
    /// overflow, total_chunks drift across GOPs, etc.). Carries a
    /// human-readable message.
    InvalidPacket(String),
}

impl From<StegoError> for Av1StegoError {
    fn from(e: StegoError) -> Self {
        Av1StegoError::Stego(e)
    }
}

#[cfg(feature = "av1-decoder")]
impl From<Av1DecodeError> for Av1StegoError {
    fn from(e: Av1DecodeError) -> Self {
        Av1StegoError::Decode(e)
    }
}

/// Hide an encrypted message inside an AV1 frame's natural encode.
///
/// Takes the natural Pass 1 output (`natural_packet` + `recording`
/// from [`encode_frame_with_phasm_tee`]), the plaintext message, and
/// the passphrase. Returns final stego AV1 bytes that decode to the
/// same picture but carry the encrypted payload at the AC_COEFF_SIGN
/// cover positions per the STC plan.
///
/// # Initial AV1 scope
/// Single-tile, single-frame, AC_COEFF_SIGN-only.
#[cfg(feature = "av1-encoder")]
pub fn av1_stego_embed(
    natural_packet: Vec<u8>,
    recording: PhasmFrameRecording,
    message: &[u8],
    passphrase: &str,
) -> Result<Vec<u8>, Av1StegoError> {
    // v2: one frame is a one-frame GOP through the production cascade
    // (keyed order, per-GOP H-hat, W·S·D costs), so tools built on this
    // single-frame API measure what ships.
    let master = crypto::derive_structural_key(passphrase)?;
    let (ciphertext, nonce, salt) = crypto::encrypt(message, passphrase)?;
    let framed = frame::build_frame(message.len(), &salt, &nonce, &ciphertext);
    let (w, h) = av1_visible_frame_size(&natural_packet)?;
    let frames = vec![(natural_packet, recording)];
    let harvest = super::whole_video::harvest_gop(&frames, w as u32, h as u32)?;
    let gop_n = harvest.len();
    super::whole_video::cascade_with_replan(
        super::whole_video::Av1CascadeGopInputs {
            gop_idx: 0,
            gop_offset: 0,
            gop_n,
            harvest: &harvest,
            frames: &frames,
            chunk: &framed,
        },
        super::whole_video::Av1CascadeGopShared {
            total_message_bytes: framed.len() as u32,
            shadow_states: &[],
            master: &master,
        },
    )?
    .ok_or(Av1StegoError::MessageTooLarge { needed_bits: framed.len() * 8, available_bits: gop_n })
}








/// Rebuild the frame_obu portion of the natural packet with the new
/// stego tile_group bytes, recomputing the ULEB128 size field so the
/// AV1 container stays consistent.
///
/// Handles the TileGroupSizeMismatch case: when stego flips happen
/// near the end of the entropy-coded stream, the range coder's
/// trailing-carry flush can produce a tile_group that's ±1 byte
/// different in length from the natural one. Previously this returned
/// `Av1StegoError::TileGroupSizeMismatch`; now the orchestrator
/// rebuilds the OBU header's size field and stitches the new
/// tile_group in cleanly.
///
/// Layout reminder (per `PhasmFrameRecording`):
/// ```text
/// natural_packet:
///   [0 .. frame_obu_start]              key-frame OBUs + metadata
///   [frame_obu_start]                   1-byte OBU_FRAME header
///   [frame_obu_start+1 .. fh_start]     ULEB128 of (fh_len + tg_len)
///   [fh_start .. tile_group_offset]     frame_header_obu payload
///   [tile_group_offset .. ...]          tile_group bytes
/// ```
#[cfg(feature = "av1-encoder")]
pub(crate) fn rebuild_obu_with_stego_tile_group(
    natural_packet: &[u8],
    recording: &PhasmFrameRecording,
    stego_tile_bytes: &[u8],
) -> Vec<u8> {
    let frame_header_start = recording.tile_group_offset - recording.frame_header_len;
    let frame_header_bytes = &natural_packet[frame_header_start..recording.tile_group_offset];
    let obu_header_byte = natural_packet[recording.frame_obu_start];

    // Compute new ULEB128 size for (frame_header + stego_tile_group).
    let new_payload_size = (recording.frame_header_len + stego_tile_bytes.len()) as u64;
    let new_uleb128 = encode_uleb128(new_payload_size);

    let mut packet = Vec::with_capacity(
        recording.frame_obu_start
            + 1
            + new_uleb128.len()
            + recording.frame_header_len
            + stego_tile_bytes.len(),
    );
    packet.extend_from_slice(&natural_packet[..recording.frame_obu_start]);
    packet.push(obu_header_byte);
    packet.extend_from_slice(&new_uleb128);
    packet.extend_from_slice(frame_header_bytes);
    packet.extend_from_slice(stego_tile_bytes);
    packet
}

/// Extract the visible-region pixel data from each plane of a
/// reconstructed Frame and pack into [`FramePlanes`] (contiguous, no
/// stride padding). The Frame buffer stores each plane with
/// filter-tap padding at `xorigin`/`yorigin` offsets; our J-UNIWARD
/// wavelet decomposition consumes packed YUV.
///
/// Memory cost: one Vec<u8> per plane = ~3 MB for 1080p. Could be
/// avoided via a wavelet decomp that consumes strided input directly
/// (small refactor of `compute_three_subbands`).
#[cfg(feature = "av1-encoder")]
pub(crate) fn pack_visible_planes(
    rec: &std::sync::Arc<phasm_rav1e::Frame<u8>>,
) -> FramePlanes {
    let pack = |plane_idx: usize| -> (Vec<u8>, usize, usize) {
        let p = &rec.planes[plane_idx];
        let w = p.cfg.width;
        let h = p.cfg.height;
        let stride = p.cfg.stride;
        let start = p.cfg.yorigin * stride + p.cfg.xorigin;
        let mut out = Vec::with_capacity(w * h);
        for row in 0..h {
            let row_start = start + row * stride;
            out.extend_from_slice(&p.data[row_start..row_start + w]);
        }
        (out, w, h)
    };
    let (y, luma_w, luma_h) = pack(0);
    let (cb, chroma_w, chroma_h) = pack(1);
    let (cr, _, _) = pack(2);
    FramePlanes {
        y,
        cb,
        cr,
        luma_width: luma_w,
        luma_height: luma_h,
        chroma_width: chroma_w,
        chroma_height: chroma_h,
    }
}

/// [`pack_visible_planes`] for J-UNIWARD's cost input: the aligned planes,
/// with the visible `width`×`height` frame's last column and row repeated
/// into rav1e's 8-alignment padding (chroma at half size, rounded up).
///
/// rav1e aligns plane width and height to 8, and what it reconstructs in
/// that padding is cropped on output: a decoder does not reproduce it, and
/// no viewer or steganalyst sees it. Costs taken over it would differ
/// between the encoder's recording and a decode of the same packets (the
/// Pass-2 rebuild), so both sides cost the replicated edge instead. At
/// sizes that are multiples of 8 this equals `pack_visible_planes`.
#[cfg(feature = "av1-encoder")]
pub(crate) fn pack_cost_planes(
    rec: &std::sync::Arc<phasm_rav1e::Frame<u8>>,
    width: usize,
    height: usize,
) -> FramePlanes {
    let mut planes = pack_visible_planes(rec);
    let replicate = |buf: &mut [u8], pw: usize, ph: usize, vw: usize, vh: usize| {
        let (vw, vh) = (vw.clamp(1, pw), vh.clamp(1, ph));
        for y in 0..vh {
            let row = &mut buf[y * pw..(y + 1) * pw];
            let edge = row[vw - 1];
            row[vw..].fill(edge);
        }
        for y in vh..ph {
            buf.copy_within((vh - 1) * pw..vh * pw, y * pw);
        }
    };
    let (lw, lh) = (planes.luma_width, planes.luma_height);
    let (cw, ch) = (planes.chroma_width, planes.chroma_height);
    replicate(&mut planes.y, lw, lh, width, height);
    replicate(&mut planes.cb, cw, ch, width.div_ceil(2), height.div_ceil(2));
    replicate(&mut planes.cr, cw, ch, width.div_ceil(2), height.div_ceil(2));
    planes
}

/// The visible frame size an AV1 packet's sequence header declares
/// (`max_frame_width` × `max_frame_height`) — the encode's frame size, as
/// rav1e never sets `frame_size_override_flag`. Every keyframe packet
/// carries one; an inter frame's packet does not, which is an error.
///
/// The research embed paths read it here because they get only packets and
/// recordings, and rav1e's reconstructed planes are 8-aligned. (The fuller
/// parser in `codec::mp4::av1_obu_split` is behind the `video` feature.)
#[cfg(feature = "av1-encoder")]
pub(crate) fn av1_visible_frame_size(packet: &[u8]) -> Result<(usize, usize), Av1StegoError> {
    let mut pos = 0usize;
    while pos < packet.len() {
        let header = packet[pos];
        if header & 0x80 != 0 || (header >> 1) & 1 == 0 {
            break; // forbidden bit, or no size field: not walkable
        }
        let ext = ((header >> 2) & 1) as usize;
        let Some((size, leb_len)) = decode_uleb128(packet, pos + 1 + ext) else {
            break;
        };
        let start = pos + 1 + ext + leb_len;
        let Some(end) = usize::try_from(size)
            .ok()
            .and_then(|s| start.checked_add(s))
            .filter(|&e| e <= packet.len())
        else {
            break;
        };
        if (header >> 3) & 0xf == OBU_SEQUENCE_HEADER {
            return sequence_header_frame_size(&packet[start..end]).ok_or_else(|| {
                Av1StegoError::InvalidPacket("unparseable AV1 sequence header".into())
            });
        }
        pos = end;
    }
    Err(Av1StegoError::InvalidPacket(
        "no AV1 sequence header in the packet (embed into a keyframe / the GOP's first packet)"
            .into(),
    ))
}

/// `max_frame_width` × `max_frame_height` from a `sequence_header_obu`
/// payload (AV1 spec § 5.5.1), skipping the fields before them.
#[cfg(feature = "av1-encoder")]
fn sequence_header_frame_size(payload: &[u8]) -> Option<(usize, usize)> {
    let mut pos = 0usize;
    let mut f = |n: u32| -> Option<u64> {
        let mut v = 0u64;
        for _ in 0..n {
            let byte = *payload.get(pos / 8)?;
            v = (v << 1) | u64::from((byte >> (7 - pos % 8)) & 1);
            pos += 1;
        }
        Some(v)
    };
    let _seq_profile = f(3)?;
    let _still_picture = f(1)?;
    let reduced_still_picture_header = f(1)? == 1;
    if reduced_still_picture_header {
        f(5)?; // seq_level_idx[0]
    } else {
        let timing_info_present = f(1)? == 1;
        let mut decoder_model_info_present = false;
        let mut buffer_delay_length = 0u32;
        if timing_info_present {
            f(32)?; // num_units_in_display_tick
            f(32)?; // time_scale
            if f(1)? == 1 {
                // equal_picture_interval: num_ticks_per_picture_minus_1, uvlc()
                let mut leading_zeros = 0u32;
                while f(1)? == 0 {
                    leading_zeros += 1;
                    if leading_zeros >= 32 {
                        return None;
                    }
                }
                f(leading_zeros)?;
            }
            decoder_model_info_present = f(1)? == 1;
            if decoder_model_info_present {
                buffer_delay_length = f(5)? as u32 + 1;
                f(32)?; // num_units_in_decoding_tick
                f(5)?; // buffer_removal_time_length_minus_1
                f(5)?; // frame_presentation_time_length_minus_1
            }
        }
        let initial_display_delay_present = f(1)? == 1;
        let operating_points = f(5)? + 1;
        for _ in 0..operating_points {
            f(12)?; // operating_point_idc
            if f(5)? > 7 {
                f(1)?; // seq_tier
            }
            if decoder_model_info_present && f(1)? == 1 {
                f(buffer_delay_length)?; // decoder_buffer_delay
                f(buffer_delay_length)?; // encoder_buffer_delay
                f(1)?; // low_delay_mode_flag
            }
            if initial_display_delay_present && f(1)? == 1 {
                f(4)?; // initial_display_delay_minus_1
            }
        }
    }
    let width_bits = f(4)? as u32 + 1;
    let height_bits = f(4)? as u32 + 1;
    let width = f(width_bits)? as usize + 1;
    let height = f(height_bits)? as usize + 1;
    Some((width, height))
}


/// Encode a u64 as an AV1 ULEB128 byte sequence (LSB-first 7-bit
/// groups, continuation bit set on all but the last byte).
#[cfg(feature = "av1-encoder")]
fn encode_uleb128(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    loop {
        let mut byte = (value & 0x7f) as u8;
        value >>= 7;
        if value != 0 {
            byte |= 0x80;
            out.push(byte);
        } else {
            out.push(byte);
            return out;
        }
    }
}

/// Decode an AV1 ULEB128 from `bytes[at..]`. Returns
/// `(value, bytes_consumed)`. AV1 spec caps ULEB128 at 8 bytes
/// (56 bits of payload + 8 continuation bits).
pub(crate) fn decode_uleb128(bytes: &[u8], at: usize) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift: u32 = 0;
    let mut i = at;
    let mut consumed = 0usize;
    while i < bytes.len() && consumed < 8 {
        let b = bytes[i];
        let lo7 = (b & 0x7f) as u64;
        if shift >= 64 {
            return None;
        }
        value |= lo7 << shift;
        consumed += 1;
        i += 1;
        if (b & 0x80) == 0 {
            return Some((value, consumed));
        }
        shift += 7;
    }
    None
}

/// AV1 OBU type codes. Only the types we need for the walker —
/// others get classified as `Other`.
pub(crate) const OBU_SEQUENCE_HEADER: u8 = 1;

/// OBU walker: split accumulated AV1 bytes at every
/// `sequence_header_obu` boundary into per-GOP slabs. Bytes BEFORE
/// the first sequence_header are discarded; bytes between consecutive
/// sequence_headers form one slab each.
///
/// AV1 OBU header layout:
/// ```text
///   bit 7    : obu_forbidden_bit (must be 0)
///   bits 3-6 : obu_type           (4 bits)
///   bit 2    : obu_extension_flag (1 byte of extension header if set)
///   bit 1    : obu_has_size_field (ULEB128 size follows the headers)
///   bit 0    : obu_reserved (0)
/// ```
///
/// rav1e emits `obu_has_size_field=1` on every OBU, so we always
/// expect a ULEB128 payload size after the header bytes.
///
/// The walker is lenient on partial / truncated input: if it can't
/// parse a complete OBU, it stops there (returns the slabs gathered
/// so far). The caller's `frame::parse_frame` CRC catches any
/// downstream corruption from a truncated final slab.
#[cfg(feature = "av1-decoder")]
pub(crate) fn split_av1_into_gops(bytes: &[u8]) -> Vec<Vec<u8>> {
    let mut slabs: Vec<Vec<u8>> = Vec::new();
    let mut current: Option<Vec<u8>> = None;
    let mut cursor = 0usize;

    while cursor < bytes.len() {
        // Parse one OBU header.
        let header_byte = bytes[cursor];
        if header_byte & 0x80 != 0 {
            // obu_forbidden_bit set — bitstream malformed, stop walking.
            break;
        }
        let obu_type = (header_byte >> 3) & 0x0f;
        let has_extension = (header_byte >> 2) & 1 != 0;
        let has_size = (header_byte >> 1) & 1 != 0;
        let header_len = 1 + (has_extension as usize);
        if cursor + header_len > bytes.len() {
            break;
        }

        // Payload size: either ULEB128 (when has_size=1) or "to end
        // of buffer" (when 0). rav1e always emits has_size=1; we treat
        // has_size=0 as "consume remainder", which only fires at end-
        // of-stream when something upstream truncated.
        let (payload_len, size_field_len) = if has_size {
            let after_hdr = cursor + header_len;
            match decode_uleb128(bytes, after_hdr) {
                Some(v) => v,
                None => break,
            }
        } else {
            ((bytes.len() - cursor - header_len) as u64, 0)
        };
        let total_obu_len = header_len + size_field_len + payload_len as usize;
        if cursor + total_obu_len > bytes.len() {
            break;
        }

        // OBU_SEQUENCE_HEADER starts a new GOP slab. Anything before
        // the first one is discarded.
        if obu_type == OBU_SEQUENCE_HEADER {
            if let Some(slab) = current.take() {
                slabs.push(slab);
            }
            current = Some(Vec::new());
        }
        if let Some(ref mut buf) = current {
            buf.extend_from_slice(&bytes[cursor..cursor + total_obu_len]);
        }
        cursor += total_obu_len;
    }

    if let Some(slab) = current {
        slabs.push(slab);
    }
    slabs
}


/// Decode stego AV1 bytes via dav1d's recording hook and harvest the
/// Tier-1 cover bits (AC_COEFF_SIGN ∪ GOLOMB_TAIL_LSB) in walker order.
#[cfg(feature = "av1-decoder")]
pub fn harvest_cover_bits_from_stego(
    stego_av1_bytes: &[u8],
) -> Result<Vec<u8>, Av1StegoError> {
    use core_dav1d_sys::DAV1D_PHASM_TAG_AC_COEFF_SIGN;
    let decoded = decode_with_recording(stego_av1_bytes)?;
    let bits: Vec<u8> = decoded
        .iter()
        .filter(|p| {
            crate::stego::cost::av1_damage::av1_in_cover(
                p.tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN,
                p.coeff_magnitude,
            )
        })
        .map(|p| p.decoded_value)
        .collect();
    if bits.is_empty() {
        return Err(Av1StegoError::NoCoverPositions);
    }
    Ok(bits)
}

/// Streaming decode (v3 wire format): extract the **first-chunk** v3
/// frame from `cover_bits`. Returns `(total_bytes, payload)` on
/// success.
///
/// v3.1 semantics (#888, AV1 step 6):
/// - 80-bit prefix `(u32 total_bytes + u32 m_total + u16 payload_len)`.
/// - Exact `stored_m_total == iter_m_total` reject (1/2³² survival rate)
///   replaces v3's canonicality + LEN_SENTINEL-bypass branching and
///   collapses surviving candidates to ~1 full extract per slab.
/// - Total_bytes sanity: ∈ (0, max_total_bytes].
/// - Full extract + `parse_first_chunk_frame_v3_1` on prefix match.
///
/// Iterates `m_total += 8` (not `w in 1..=max_w` as v3 did), so the
/// loop visits the encoder's chosen `m_total` value directly — required
/// for the exact reject to land. Mirrors the H.264 v3.1 implementation
/// in `core/src/codec/h264/streaming_session.rs::extract_first_chunk_frame_match`.
///
/// See `docs/design/video/h264/chunk-frame-v3.1-decode.md` for the wire
/// format and survivor-cost analysis.
#[cfg(feature = "av1-decoder")]
pub(crate) fn extract_first_chunk_frame_match(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    max_total_bytes: u32,
) -> Option<(u32, Vec<u8>)> {
    use crate::stego::chunk_frame::{
        parse_first_chunk_frame_v3_1, CHUNK_FRAME_FIRST_HEADER_LEN_V3_1,
    };
    use crate::stego::stc::extract::stc_extract_prefix;

    let n_cover = cover_bits.len();
    let min_m = CHUNK_FRAME_FIRST_HEADER_LEN_V3_1 * 8; // 80 bits
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
        let hhat_matrix = hhat::generate_hhat_end_set(STC_H, w, hhat_seed);

        // 80-bit prefix: u32 total_bytes + u32 m_total + u16 payload_len.
        let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat_matrix, w, 80);
        if prefix_bits.len() < 80 {
            m_total += 8;
            continue;
        }
        let pb = frame::bits_to_bytes(&prefix_bits);
        if pb.len() < 10 {
            m_total += 8;
            continue;
        }
        // Exact m_total checksum: only the encoder's candidate satisfies it
        // (1/2³² random-match rate; supersedes v3's canonicality + sentinel
        // bypass).
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
        // Prefix passed → full extract + canonical parse.
        let extracted = stc_extract(&cover_bits[..used], &hhat_matrix, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = frame::bits_to_bytes(bits);
        if let Some((total_bytes, _m, payload)) = parse_first_chunk_frame_v3_1(&bytes) {
            return Some((total_bytes, payload.to_vec()));
        }
        m_total += 8;
    }
    None
}

/// Streaming decode (v3.1 wire format): extract a **subsequent-chunk**
/// v3.1 frame from `cover_bits`. Returns `payload` on success.
///
/// - 48-bit prefix `(u32 m_total + u16 payload_len)`. Exact m_total
///   reject (1/2³² survival) — same mechanism as first-chunk variant.
/// - `payload.len() ≤ max_remaining_bytes` upper-bound sanity filter.
/// - Full extract + `parse_chunk_frame_v3_1` on prefix match.
///
/// `max_remaining_bytes` is `total_bytes - accumulated_bytes` from the
/// outer decode loop. Sanity ceiling so the decoder never reads past
/// the encoder's intent.
///
/// Iterates `m_total += 8` (mirrors H.264 v3.1 implementation).
/// See `docs/design/video/h264/chunk-frame-v3.1-decode.md` (#888).
#[cfg(feature = "av1-decoder")]
pub(crate) fn extract_chunk_frame_match(
    cover_bits: &[u8],
    hhat_seed: &[u8; 32],
    max_remaining_bytes: usize,
) -> Option<Vec<u8>> {
    use crate::stego::chunk_frame::{
        parse_chunk_frame_v3_1, CHUNK_FRAME_NEXT_HEADER_LEN_V3_1,
    };
    use crate::stego::stc::extract::stc_extract_prefix;

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
        let hhat_matrix = hhat::generate_hhat_end_set(STC_H, w, hhat_seed);

        // 48-bit prefix: u32 m_total + u16 payload_len. Exact m_total reject.
        let prefix_bits = stc_extract_prefix(&cover_bits[..used], &hhat_matrix, w, 48);
        if prefix_bits.len() < 48 {
            m_total += 8;
            continue;
        }
        let pb = frame::bits_to_bytes(&prefix_bits);
        if pb.len() < 6 {
            m_total += 8;
            continue;
        }
        let stored_m_total = u32::from_be_bytes([pb[0], pb[1], pb[2], pb[3]]) as usize;
        if stored_m_total != m_total {
            m_total += 8;
            continue;
        }
        let extracted = stc_extract(&cover_bits[..used], &hhat_matrix, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = frame::bits_to_bytes(bits);
        if let Some((_m, payload)) = parse_chunk_frame_v3_1(&bytes) {
            if payload.len() <= max_remaining_bytes {
                return Some(payload.to_vec());
            }
        }
        m_total += 8;
    }
    None
}



#[cfg(all(test, feature = "av1-encoder", feature = "av1-decoder"))]
mod tests {
    use super::*;

    /// The single-frame API is the production v2 cascade on a one-frame
    /// GOP, so its output decodes with the v2 session (the stealth tools
    /// built on it measure v2).
    #[test]
    fn single_frame_embed_decodes_under_v2() {
        let (w, h) = (256usize, 144usize);
        let mut s = 0x2545_F491_4F6C_DD1Du64;
        let yuv: Vec<u8> = (0..w * h * 3 / 2)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                (s >> 56) as u8
            })
            .collect();
        let params = crate::Av1StreamingEncodeParams {
            width: w as u32,
            height: h as u32,
            quantizer: 30,
            gop_size: 1,
            total_frames_hint: 1,
            colour: Default::default(),
        };
        let (packet, recording) = super::super::whole_video::encode_gop_natural(&yuv, 1, params)
            .expect("natural encode")
            .pop()
            .unwrap();
        let message = b"single frame, v2 wire";
        let stego = av1_stego_embed(packet, recording, message, "v2-frame").expect("embed");
        let mut dec = super::super::session::Av1StreamingDecodeSession::create("v2-frame");
        dec.push_bytes(&stego);
        assert_eq!(dec.finish().expect("decode"), message);
    }
    use phasm_rav1e::color::ChromaSampling;
    use phasm_rav1e::phasm_stego::{
        encode_frame_with_phasm_tee, make_frame, make_inter_config,
        FrameInvariants, FrameState,
    };
    use phasm_rav1e::prelude::Sequence;
    use phasm_rav1e::EncoderConfig;
    use std::sync::Arc;

    /// Build the same 128×128 gradient frame state used in the
    /// strict cursor parity test. Returns (natural_packet, recording)
    /// from encode_frame_with_phasm_tee.
    fn build_natural_encode() -> (Vec<u8>, PhasmFrameRecording) {
        let config = Arc::new(EncoderConfig {
            width: 128,
            height: 128,
            bit_depth: 8,
            chroma_sampling: ChromaSampling::Cs420,
            quantizer: 30,
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
        // Disable segmentation to avoid kmeans underflow on
        // synthetic gradient (see encode_frame_with_phasm_tee
        // smoke test in phasm-rav1e).
        fi.enable_segmentation = false;
        let mut frame = make_frame::<u8>(128, 128, ChromaSampling::Cs420);
        // CRITICAL: use `Plane::copy_from_raw_u8` to write into the
        // visible region. Manual chunks_mut iterates raw plane.data
        // bytes including filter-tap padding rows, so synthetic
        // content lands in padding while visible stays default-init
        // neutral gray. See feedback_visual_fidelity_is_correctness.
        for (plane_idx, plane) in frame.planes.iter_mut().enumerate() {
            let (w, h) = (plane.cfg.width, plane.cfg.height);
            let mut buf = vec![0u8; w * h];
            for row_idx in 0..h {
                for col_idx in 0..w {
                    buf[row_idx * w + col_idx] = ((row_idx.wrapping_mul(7)
                        + col_idx.wrapping_mul(3)
                        + plane_idx * 13)
                        & 0xff) as u8;
                }
            }
            plane.copy_from_raw_u8(&buf, w, 1);
        }
        let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
        let inter_cfg = make_inter_config(&config);
        encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg)
    }



    /// Stego packet remains a valid AV1 bitstream (dav1d decodes
    /// without error).
    #[test]
    fn w3105_stego_bytes_decode_as_valid_av1() {
        let plaintext = b"valid av1 check";
        let passphrase = "test-pass";

        let (natural_packet, recording) = build_natural_encode();
        let stego_packet = av1_stego_embed(
            natural_packet,
            recording,
            plaintext,
            passphrase,
        )
        .expect("embed");

        // dav1d decode should succeed (no syntax errors from the
        // stego modifications).
        let decode_result = decode_with_recording(&stego_packet);
        assert!(
            decode_result.is_ok(),
            "stego bytes must decode as valid AV1; got {:?}",
            decode_result.as_ref().err()
        );
    }



}
