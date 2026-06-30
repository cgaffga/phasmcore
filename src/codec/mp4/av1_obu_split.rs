// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! VP.M.2 — AV1 OBU stream → MP4 sample splitter.
//!
//! Walks an AV1 OBU byte stream (as emitted by phasm-rav1e
//! `encode_frame_with_phasm_tee` and concatenated by the streaming
//! session) and partitions it into:
//!
//! 1. The leading `sequence_header_obu` bytes, used by [`super::Av1cData`]
//!    to populate the `av1C` ConfigOBUs field.
//! 2. Per-frame OBU byte groups, one MP4 sample each.
//! 3. Sync (sample_is_sync) flags per sample — `true` for keyframes
//!    (the frame immediately following a sequence_header_obu).
//!
//! AV1 OBU layout reference: AV1 spec § 5.3.2. OBU type codes per
//! § 6.2.1. The walker mirrors the bit-level parser in
//! `core/src/codec/av1/stego/orchestrator.rs::split_av1_into_gops`
//! but split at per-frame boundaries instead of per-GOP, and
//! categorizes OBUs into mp4-sample roles.

use super::Mp4Error;

/// AV1 OBU type codes (subset — only the ones the splitter cares about).
/// Exposed for integration tests + external OBU-handling code.
pub const OBU_SEQUENCE_HEADER: u8 = 1;
pub const OBU_TEMPORAL_DELIMITER: u8 = 2;
pub const OBU_FRAME_HEADER: u8 = 3;
pub const OBU_TILE_GROUP: u8 = 4;
pub const OBU_METADATA: u8 = 5;
pub const OBU_FRAME: u8 = 6;

/// Result of [`split_av1_into_samples`].
#[derive(Debug, Clone)]
pub struct Av1SampleSplit {
    /// The first `sequence_header_obu` bytes encountered in the
    /// stream. Used by [`super::Av1cData`] to populate the `av1C`
    /// box's `ConfigOBUs` field. Empty if no sequence header was
    /// found (caller should treat this as an error — every valid
    /// AV1 stream starts with at least one).
    pub sequence_header_obu: Vec<u8>,

    /// Per-frame OBU byte groups. Each entry is the OBU bytes that
    /// constitute one coded picture (typically a single `frame_obu`
    /// containing both frame header and tile data; optionally
    /// preceded by `metadata_obu`s belonging to that frame).
    ///
    /// **In-band SH emission (muxer-sh-in-band-fix, 2026-06-29):**
    /// `sequence_header_obu`s are PREPENDED to sync samples (sample
    /// where `sync[i] == true`). This makes the MP4 robust under
    /// random-access seeks and decodable by ffmpeg/libdav1d/VLC etc.,
    /// matching the convention real-world AV1 encoders use. The same
    /// SH bytes also live in `av1C` (redundant per AV1-ISOBMFF § 2.2.1
    /// but explicitly allowed). Non-sync samples carry no SH.
    /// `temporal_delimiter_obu`s remain STRIPPED (optional per spec,
    /// adding them wastes bits).
    pub samples: Vec<Vec<u8>>,

    /// `true` for samples that are AV1 keyframes — the frame
    /// immediately following a `sequence_header_obu`. Used to
    /// populate the MP4 `stss` (sync sample) box.
    pub sync: Vec<bool>,
}

/// VP.M.2 entry point: walk an AV1 OBU byte stream and split into
/// MP4-ready samples.
///
/// **Liberal on truncated input**: stops cleanly if the stream ends
/// mid-OBU, returning what was successfully parsed. Caller should
/// validate the result (`samples.len() > 0`, `sequence_header_obu`
/// non-empty) before muxing.
///
/// **OBU type handling**:
///
/// | OBU type        | Action |
/// |-----------------|---------------------------------|
/// | SEQUENCE_HEADER | Store first one for `av1C`; cache most-recent bytes for next sync sample; set `pending_keyframe` |
/// | TEMPORAL_DELIM  | STRIPPED (optional per spec) |
/// | FRAME           | Starts new sample; PREPEND cached SH bytes if `pending_keyframe`; sync = pending_keyframe (then cleared) |
/// | FRAME_HEADER    | Starts new sample; PREPEND cached SH bytes if `pending_keyframe`; sync = pending_keyframe (then cleared) |
/// | TILE_GROUP      | Appended to current sample (follows a FRAME_HEADER) |
/// | METADATA        | Appended to current sample (belongs to upcoming/current frame) |
/// | Other           | Appended to current sample if one exists, else discarded |
///
/// **In-band SH emission** (muxer-sh-in-band-fix, 2026-06-29): every sync
/// sample's bytes start with the most-recently-seen `sequence_header_obu`.
/// This makes the MP4 decodable by ffmpeg/libdav1d/VLC/Chromium (which
/// don't carry SH state across random-access seek points), matching the
/// convention real-world AV1 encoders use. av1C still carries the SH
/// (mandatory per AV1-ISOBMFF § 2.3.1; in-band SH is allowed in
/// addition per § 2.2.1).
pub fn split_av1_into_samples(bytes: &[u8]) -> Result<Av1SampleSplit, Mp4Error> {
    let mut sequence_header_obu: Vec<u8> = Vec::new();
    let mut samples: Vec<Vec<u8>> = Vec::new();
    let mut sync: Vec<bool> = Vec::new();

    let mut current_sample: Option<Vec<u8>> = None;
    let mut pending_keyframe = false;
    // Most-recently-seen SH bytes — prepended verbatim to the next sync
    // sample. `pending_sh` is `Some` from the moment we see an SH until
    // the FRAME/FH that consumes it. Real-world encoders (rav1e, libaom,
    // SVT-AV1) emit identical SH bytes on every reset, but we keep the
    // freshest bytes so streams that DO vary their SHs across GOPs
    // (e.g. resolution change) are handled correctly.
    let mut pending_sh: Option<Vec<u8>> = None;

    let mut cursor = 0usize;
    while cursor < bytes.len() {
        let header_byte = bytes[cursor];
        if header_byte & 0x80 != 0 {
            // obu_forbidden_bit set — malformed; stop here.
            break;
        }
        let obu_type = (header_byte >> 3) & 0x0f;
        let has_extension = (header_byte >> 2) & 1 != 0;
        let has_size = (header_byte >> 1) & 1 != 0;
        let header_len = 1 + (has_extension as usize);
        if cursor + header_len > bytes.len() {
            break;
        }

        // Payload size: ULEB128 when has_size=1, else "to end" (rare).
        let (payload_len, size_field_len) = if has_size {
            match decode_uleb128(bytes, cursor + header_len) {
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

        let obu_bytes = &bytes[cursor..cursor + total_obu_len];

        match obu_type {
            OBU_SEQUENCE_HEADER => {
                if sequence_header_obu.is_empty() {
                    sequence_header_obu.extend_from_slice(obu_bytes);
                }
                // Cache for in-band emission on the next sync sample.
                // Per-GOP resets in phasm's streaming session re-emit the
                // same SH bytes; refreshing the cache keeps us correct
                // even if a future encoder varies them.
                pending_sh = Some(obu_bytes.to_vec());
                pending_keyframe = true;
            }
            OBU_TEMPORAL_DELIMITER => {
                // Optional and not part of samples per AV1-ISOBMFF
                // conventions. Skip.
            }
            OBU_FRAME | OBU_FRAME_HEADER => {
                // Start a new sample. Flush the previous one first.
                if let Some(s) = current_sample.take() {
                    samples.push(s);
                }
                let mut new_sample = Vec::new();
                // In-band SH on sync samples. Decoders without av1C
                // splicing (ffmpeg + libdav1d / VLC / Chromium) need
                // the SH inside the sample bytes; mid-clip seeks land
                // here too — every sync sample carries its own SH.
                if pending_keyframe {
                    if let Some(sh) = pending_sh.take() {
                        new_sample.extend_from_slice(&sh);
                    }
                }
                new_sample.extend_from_slice(obu_bytes);
                current_sample = Some(new_sample);
                sync.push(pending_keyframe);
                pending_keyframe = false;
            }
            OBU_TILE_GROUP => {
                // Follows a FRAME_HEADER (separate frame+tile mode).
                // Append to current sample.
                if let Some(s) = current_sample.as_mut() {
                    s.extend_from_slice(obu_bytes);
                }
                // If no current sample (stream starts mid-frame),
                // discard — corrupt input.
            }
            OBU_METADATA => {
                // Metadata belongs to the upcoming or current frame.
                // Append to current sample if one exists; otherwise
                // buffer until the next frame starts (defer-append by
                // pretending it was part of the next sample).
                if let Some(s) = current_sample.as_mut() {
                    s.extend_from_slice(obu_bytes);
                } else {
                    // Pre-frame metadata: stash into a starter sample
                    // that gets completed on the next FRAME/FH OBU.
                    // For simplicity, drop pre-frame metadata in this
                    // MVP — rav1e doesn't emit metadata OBUs.
                }
            }
            _ => {
                // Padding / tile_list / etc. — append if in a sample,
                // else discard.
                if let Some(s) = current_sample.as_mut() {
                    s.extend_from_slice(obu_bytes);
                }
            }
        }

        cursor += total_obu_len;
    }

    // Flush trailing sample.
    if let Some(s) = current_sample {
        samples.push(s);
    }

    if samples.len() != sync.len() {
        // Shouldn't happen — sync is pushed on every FRAME/FH event
        // before the sample is finalized. Defensive check.
        return Err(Mp4Error::InvalidBox(format!(
            "VP.M.2 splitter desync: {} samples vs {} sync flags",
            samples.len(),
            sync.len()
        )));
    }

    Ok(Av1SampleSplit {
        sequence_header_obu,
        samples,
        sync,
    })
}

/// Decode a ULEB128-encoded length from `bytes[at..]`. Returns
/// `(value, bytes_consumed)`. AV1 spec § 4.10.5 caps at 8 bytes.
fn decode_uleb128(bytes: &[u8], at: usize) -> Option<(u64, usize)> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    let mut consumed = 0;
    for i in 0..8 {
        let idx = at + i;
        if idx >= bytes.len() {
            return None;
        }
        let b = bytes[idx];
        value |= ((b & 0x7F) as u64) << shift;
        consumed += 1;
        if b & 0x80 == 0 {
            return Some((value, consumed));
        }
        shift += 7;
        if shift >= 56 {
            return None;
        }
    }
    None
}

// Tests live in core/tests/av1_mp4_obu_split.rs as an integration
// test file — bypasses pre-existing h264 stego compile breakage in
// the lib test harness on the AV1 branch.

// ──────────────────────────────────────────────────────────────────
//  VP.8 — sequence_header_obu parser
// ──────────────────────────────────────────────────────────────────

/// Parsed fields from an AV1 `sequence_header_obu`. Subset of the
/// full spec (§ 5.5.1) — only the fields needed to populate
/// [`super::Av1cData`] for av1C plus the max frame dimensions for
/// the muxer's `tkhd` / `av01` boxes.
///
/// Per spec § 5.5.1 + AV1-ISOBMFF § 2.3.1.
#[derive(Debug, Clone, Copy)]
pub struct Av1SequenceHeaderInfo {
    /// Encoded picture max width (post +1).
    pub max_frame_width: u32,
    /// Encoded picture max height (post +1).
    pub max_frame_height: u32,
    /// AV1 sequence profile (0 = "Main", 1 = "High", 2 = "Professional").
    pub seq_profile: u8,
    /// AV1 sequence level index (0..=31). The fields below match
    /// [`super::Av1cData`] directly.
    pub seq_level_idx_0: u8,
    pub seq_tier_0: u8,
    pub high_bitdepth: bool,
    pub twelve_bit: bool,
    pub monochrome: bool,
    pub chroma_subsampling_x: bool,
    pub chroma_subsampling_y: bool,
    pub chroma_sample_position: u8,
}

/// VP.8 entry point: parse an AV1 `sequence_header_obu` (full OBU
/// bytes including the OBU header + ULEB size + payload) and extract
/// the fields needed for `av1C` + dimensions.
///
/// Returns `None` if the input isn't a sequence_header_obu, is too
/// short, or contains a syntax we don't support yet (e.g. multiple
/// operating points with extended fields). The caller (typically
/// [`super::build::build_mp4_av1`]) should fall back to
/// [`super::Av1cData::default_yuv420_8bit`] on `None`.
///
/// **Coverage**: handles the single-operating-point cases that
/// phasm-rav1e's fork output produces. Multi-operating-point streams
/// fall back to defaults — out of scope for v0.7.
pub fn parse_sequence_header_obu(obu_bytes: &[u8]) -> Option<Av1SequenceHeaderInfo> {
    // Parse OBU header (must be SEQUENCE_HEADER), skip the ULEB size,
    // then parse the payload as a bit stream.
    if obu_bytes.is_empty() {
        return None;
    }
    let header_byte = obu_bytes[0];
    if header_byte & 0x80 != 0 {
        return None; // obu_forbidden_bit set
    }
    let obu_type = (header_byte >> 3) & 0x0f;
    if obu_type != OBU_SEQUENCE_HEADER {
        return None;
    }
    let has_extension = (header_byte >> 2) & 1 != 0;
    let has_size = (header_byte >> 1) & 1 != 0;
    let mut payload_start = 1 + (has_extension as usize);
    if has_size {
        let (_size, size_field_len) = decode_uleb128(obu_bytes, payload_start)?;
        payload_start += size_field_len;
    }
    if payload_start >= obu_bytes.len() {
        return None;
    }

    let mut br = BitReader::new(&obu_bytes[payload_start..]);

    let seq_profile = br.read_bits(3)? as u8;
    let _still_picture = br.read_bits(1)?;
    let reduced_still_picture_header = br.read_bits(1)? != 0;

    let mut seq_level_idx_0: u8 = 0;
    let mut seq_tier_0: u8 = 0;

    if reduced_still_picture_header {
        seq_level_idx_0 = br.read_bits(5)? as u8;
    } else {
        let timing_info_present_flag = br.read_bits(1)? != 0;
        let mut decoder_model_info_present_flag = false;
        if timing_info_present_flag {
            // num_units_in_display_tick u(32) + time_scale u(32)
            // + equal_picture_interval u(1)
            br.read_bits(32)?;
            br.read_bits(32)?;
            let equal_picture_interval = br.read_bits(1)? != 0;
            if equal_picture_interval {
                let _ = read_uvlc(&mut br)?;
            }
            decoder_model_info_present_flag = br.read_bits(1)? != 0;
            if decoder_model_info_present_flag {
                // buffer_delay_length_minus_1 u(5), num_units_in_decoding_tick u(32),
                // buffer_removal_time_length_minus_1 u(5), frame_presentation_time_length_minus_1 u(5)
                br.read_bits(5)?;
                br.read_bits(32)?;
                br.read_bits(5)?;
                br.read_bits(5)?;
            }
        }
        let initial_display_delay_present_flag = br.read_bits(1)? != 0;
        let operating_points_cnt_minus_1 = br.read_bits(5)? as usize;
        if operating_points_cnt_minus_1 > 0 {
            // Multi-operating-point: bail out and let caller use defaults.
            // Phasm-rav1e never emits this.
            return None;
        }
        // Single operating point — i == 0.
        br.read_bits(12)?; // operating_point_idc[0]
        seq_level_idx_0 = br.read_bits(5)? as u8;
        if seq_level_idx_0 > 7 {
            seq_tier_0 = br.read_bits(1)? as u8;
        }
        if decoder_model_info_present_flag {
            let decoder_model_present_for_this_op = br.read_bits(1)? != 0;
            if decoder_model_present_for_this_op {
                // operating_parameters_info: 2 × buffer_delay_length_minus_1 + 1 bits each
                // We bailed out on decoder_model above for known-rav1e streams; skip
                // generically with conservative width (we have no length here without state).
                return None;
            }
        }
        if initial_display_delay_present_flag {
            let initial_display_delay_present_for_this_op = br.read_bits(1)? != 0;
            if initial_display_delay_present_for_this_op {
                br.read_bits(4)?; // initial_display_delay_minus_1[0]
            }
        }
    }

    let frame_width_bits_minus_1 = br.read_bits(4)? as u32;
    let frame_height_bits_minus_1 = br.read_bits(4)? as u32;
    let n = frame_width_bits_minus_1 + 1;
    let m = frame_height_bits_minus_1 + 1;
    let max_frame_width = (br.read_bits(n as usize)? + 1) as u32;
    let max_frame_height = (br.read_bits(m as usize)? + 1) as u32;

    if !reduced_still_picture_header {
        let frame_id_numbers_present_flag = br.read_bits(1)? != 0;
        if frame_id_numbers_present_flag {
            br.read_bits(4)?; // delta_frame_id_length_minus_2
            br.read_bits(3)?; // additional_frame_id_length_minus_1
        }
    }

    let _use_128x128_superblock = br.read_bits(1)?;
    let _enable_filter_intra = br.read_bits(1)?;
    let _enable_intra_edge_filter = br.read_bits(1)?;

    if !reduced_still_picture_header {
        let _enable_interintra_compound = br.read_bits(1)?;
        let _enable_masked_compound = br.read_bits(1)?;
        let _enable_warped_motion = br.read_bits(1)?;
        let _enable_dual_filter = br.read_bits(1)?;
        let enable_order_hint = br.read_bits(1)? != 0;
        if enable_order_hint {
            let _enable_jnt_comp = br.read_bits(1)?;
            let _enable_ref_frame_mvs = br.read_bits(1)?;
        }
        let seq_choose_screen_content_tools = br.read_bits(1)? != 0;
        let seq_force_screen_content_tools = if seq_choose_screen_content_tools {
            2 // SELECT_SCREEN_CONTENT_TOOLS sentinel; not used downstream
        } else {
            br.read_bits(1)? as u32
        };
        if seq_force_screen_content_tools > 0 {
            let seq_choose_integer_mv = br.read_bits(1)? != 0;
            if !seq_choose_integer_mv {
                br.read_bits(1)?; // seq_force_integer_mv
            }
        }
        if enable_order_hint {
            br.read_bits(3)?; // order_hint_bits_minus_1
        }
    }

    let _enable_superres = br.read_bits(1)?;
    let _enable_cdef = br.read_bits(1)?;
    let _enable_restoration = br.read_bits(1)?;

    // color_config()
    let high_bitdepth = br.read_bits(1)? != 0;
    let twelve_bit = if seq_profile == 2 && high_bitdepth {
        br.read_bits(1)? != 0
    } else {
        false
    };
    let monochrome = if seq_profile == 1 {
        false
    } else {
        br.read_bits(1)? != 0
    };
    let color_description_present_flag = br.read_bits(1)? != 0;
    let (cp, tc, mc) = if color_description_present_flag {
        let cp = br.read_bits(8)? as u8;
        let tc = br.read_bits(8)? as u8;
        let mc = br.read_bits(8)? as u8;
        (cp, tc, mc)
    } else {
        (2, 2, 2) // UNSPECIFIED defaults
    };

    let (subsampling_x, subsampling_y, chroma_sample_position) = if monochrome {
        let _color_range = br.read_bits(1)?;
        (true, true, 0u8)
    } else if cp == 1 && tc == 13 && mc == 0 {
        // sRGB
        (false, false, 0u8)
    } else {
        let _color_range = br.read_bits(1)?;
        let (sx, sy) = if seq_profile == 0 {
            (true, true)
        } else if seq_profile == 1 {
            (false, false)
        } else {
            // profile 2: depends on bit depth
            let bit_depth = if high_bitdepth { if twelve_bit { 12 } else { 10 } } else { 8 };
            if bit_depth == 12 {
                let sx = br.read_bits(1)? != 0;
                let sy = if sx { br.read_bits(1)? != 0 } else { false };
                (sx, sy)
            } else {
                (true, false)
            }
        };
        let csp = if sx && sy {
            br.read_bits(2)? as u8
        } else {
            0u8
        };
        (sx, sy, csp)
    };

    Some(Av1SequenceHeaderInfo {
        max_frame_width,
        max_frame_height,
        seq_profile,
        seq_level_idx_0,
        seq_tier_0,
        high_bitdepth,
        twelve_bit,
        monochrome,
        chroma_subsampling_x: subsampling_x,
        chroma_subsampling_y: subsampling_y,
        chroma_sample_position,
    })
}

/// Big-endian MSB-first bit reader over a byte slice. AV1 syntax is
/// defined this way (spec § 4.10.1).
struct BitReader<'a> {
    bytes: &'a [u8],
    bit_pos: usize, // bit offset from start of bytes
}

impl<'a> BitReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, bit_pos: 0 }
    }

    /// Read `n` bits (0..=64) MSB first as a u64. Returns `None` if
    /// the stream would underflow.
    fn read_bits(&mut self, n: usize) -> Option<u64> {
        if n == 0 {
            return Some(0);
        }
        if n > 64 {
            return None;
        }
        let mut value: u64 = 0;
        for _ in 0..n {
            let byte_idx = self.bit_pos >> 3;
            if byte_idx >= self.bytes.len() {
                return None;
            }
            let bit_idx_in_byte = 7 - (self.bit_pos & 7);
            let bit = (self.bytes[byte_idx] >> bit_idx_in_byte) & 1;
            value = (value << 1) | bit as u64;
            self.bit_pos += 1;
        }
        Some(value)
    }
}

/// AV1 uvlc (unsigned variable-length code) reader, spec § 4.10.3.
/// Used to parse `num_ticks_per_picture_minus_1` inside timing_info.
fn read_uvlc(br: &mut BitReader<'_>) -> Option<u64> {
    let mut leading_zeros = 0u32;
    loop {
        let bit = br.read_bits(1)?;
        if bit != 0 {
            break;
        }
        leading_zeros += 1;
        if leading_zeros >= 32 {
            // uvlc cap per spec; treat as overflow → fail.
            return None;
        }
    }
    let value = br.read_bits(leading_zeros as usize)?;
    Some(value + ((1u64 << leading_zeros) - 1))
}
