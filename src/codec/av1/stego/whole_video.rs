// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Whole-video shadow encode flow.
//!
//! Ported from production OH264's
//! `h264_encode_with_shadows` (see
//! `core/src/codec/h264/openh264_stego.rs`). AV1-simplified:
//! `WriterRecorder` + `replay_with_overrides` is wire-clean by
//! construction (encoder state never sees stego flips), so the
//! provisional Pass 2 the OH264 path needs is elided. Pass 1
//! produces both natural OBU bytes AND the cover the decoder will
//! see; the cascade loop runs once-through.
//!
//! Design: see
//! `docs/design/video/av1/phase-c-wv-whole-video-shadow.md` § 3.2.

use std::collections::HashMap;
use std::sync::Arc;

use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::{PhasmFrameRecording, WriterEncoder};
use phasm_rav1e::phasm_stego::{
    encode_gop_with_phasm_tee, make_frame, AcSignMeta,
    PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

use crate::codec::av1::stego::session::Av1StreamingEncodeParams;
use crate::codec::av1::stego::shadow::{
    av1_shadow_extract, prepare_shadows, AV1_SHADOW_PARITY_TIERS,
};
use crate::codec::av1::stego::writer::{replay_with_overrides, OverrideMap};
use crate::stego::chunk_frame::{
    build_chunk_frame, build_first_chunk_frame, split_message_into_chunks,
};
use crate::stego::cost::av1_uniward::{
    compute_av1_uniward_costs_with_state, Av1FramePosition, FramePlanes,
};
use crate::stego::stc::embed::stc_embed;
use crate::stego::stc::hhat;
use crate::stego::{frame, payload};

use super::orchestrator::{
    harvest_cover_bits_from_stego, pack_visible_planes,
    rebuild_obu_with_stego_tile_group, Av1StegoError,
};

const STC_H: usize = 7;

/// Encode a whole video with shadows spread across all GOPs
/// (whole-video shadow scope per `phase-c-shadows.md` § 2).
///
/// `yuv` is the packed tight I420 buffer for ALL `n_frames` frames
/// (frame-major). `primary_framed` is the pre-built primary frame
/// (encrypt + frame::build_frame already done by the caller — the
/// session does this at `create_whole_video_with_shadows`).
///
/// Returns the concatenated Annex-B-equivalent stego AV1 bytes for
/// every GOP, with the chunk_frame protocol per
/// `phase-c-streaming-session-v6.md` § 8 unchanged from per-GOP
/// scope.
///
/// Cascades through `AV1_SHADOW_PARITY_TIERS` `[4, 8, 16, 32, 64,
/// 128]` until the produced bytes round-trip every shadow under
/// dav1d walk + `av1_shadow_extract`. Returns
/// `Av1StegoError::Stego(StegoError::ShadowEmbedFailed)` if all
/// rungs exhaust.
pub fn av1_stego_encode_whole_video_with_shadows(
    yuv: &[u8],
    n_frames: u32,
    params: Av1StreamingEncodeParams,
    primary_framed: &[u8],
    primary_passphrase: &str,
    shadows: &[(&str, &[u8])],
    shadow_parity_len_floor: usize,
) -> Result<Vec<u8>, Av1StegoError> {
    if shadows.is_empty() {
        return Err(Av1StegoError::InvalidPacket(
            "av1_stego_encode_whole_video_with_shadows: empty shadows — caller \
             should route to per-GOP path instead"
                .into(),
        ));
    }
    if n_frames == 0 {
        return Err(Av1StegoError::InvalidPacket(
            "av1_stego_encode_whole_video_with_shadows: n_frames == 0".into(),
        ));
    }
    let frame_size = expected_i420_size(params.width, params.height);
    let expected_total = frame_size
        .checked_mul(n_frames as usize)
        .ok_or_else(|| {
            Av1StegoError::InvalidPacket(format!(
                "yuv size overflow ({} × {})",
                frame_size, n_frames
            ))
        })?;
    if yuv.len() != expected_total {
        return Err(Av1StegoError::InvalidPacket(format!(
            "yuv length {} != expected {} ({} frames × {} bytes)",
            yuv.len(),
            expected_total,
            n_frames,
            frame_size
        )));
    }

    let gop_size = params.gop_size.max(1);
    let n_gops = n_frames.div_ceil(gop_size);
    let total_chunks = u16::try_from(n_gops).map_err(|_| {
        Av1StegoError::InvalidPacket(format!(
            "n_gops {} exceeds u16::MAX",
            n_gops
        ))
    })?;
    // Clip-level total_bytes goes in the first chunk header (v3 wire format).
    if primary_framed.len() > u32::MAX as usize {
        return Err(Av1StegoError::InvalidPacket(format!(
            "primary_framed {} exceeds u32::MAX",
            primary_framed.len()
        )));
    }
    let total_message_bytes = primary_framed.len() as u32;
    let chunks = split_message_into_chunks(primary_framed, total_chunks)
        .map_err(Av1StegoError::Stego)?;

    let structural_key = crate::stego::crypto::derive_structural_key(primary_passphrase)
        .map_err(Av1StegoError::Stego)?;
    let hhat_seed: [u8; 32] = structural_key[32..]
        .try_into()
        .expect("derive_structural_key returns 64 bytes");

    // Pass 1: encode every GOP naturally + harvest per-GOP Tier-1 cover.
    let mut per_gop_natural: Vec<Vec<(Vec<u8>, PhasmFrameRecording)>> = Vec::with_capacity(n_gops as usize);
    let mut per_gop_harvests: Vec<GopHarvest> = Vec::with_capacity(n_gops as usize);

    for gop_idx in 0..n_gops as usize {
        let start_frame = gop_idx as u32 * gop_size;
        let end_frame = (start_frame + gop_size).min(n_frames);
        let frames_in_gop = end_frame - start_frame;
        let gop_yuv = &yuv[(start_frame as usize * frame_size)
            ..(end_frame as usize * frame_size)];
        let pf = encode_gop_natural(gop_yuv, frames_in_gop, params)?;
        let harvest = harvest_gop(&pf)?;
        per_gop_natural.push(pf);
        per_gop_harvests.push(harvest);
    }

    // Compute union cover length + per-GOP offsets.
    let per_gop_cover_lens: Vec<usize> = per_gop_harvests
        .iter()
        .map(|h| h.cover_bits.len())
        .collect();
    let mut per_gop_cover_offsets: Vec<usize> = Vec::with_capacity(n_gops as usize);
    let mut offset = 0usize;
    for &n in &per_gop_cover_lens {
        per_gop_cover_offsets.push(offset);
        offset += n;
    }
    let union_n = offset;

    // Cascade loop over SHADOW_PARITY_TIERS starting at the
    // caller-provided floor. Default floor matches `create_with_shadows`'s
    // 16 unless caller overrode.
    let parity_tiers: Vec<usize> = AV1_SHADOW_PARITY_TIERS
        .iter()
        .copied()
        .filter(|&p| p >= shadow_parity_len_floor)
        .collect();

    if parity_tiers.is_empty() {
        return Err(Av1StegoError::InvalidPacket(format!(
            "shadow_parity_len_floor {} exceeds all SHADOW_PARITY_TIERS",
            shadow_parity_len_floor
        )));
    }

    for parity_len in parity_tiers {
        // Prepare shadow states over the WHOLE-VIDEO union cover
        // (positions are union-cover-indexed).
        let shadow_states =
            match prepare_shadows(union_n, shadows, parity_len) {
                Ok(v) => v,
                Err(_) => continue, // capacity exhausted at this parity → try next
            };

        // For each GOP, encode the GOP with the slice of shadow
        // positions that fall in that GOP's union range.
        let mut output: Vec<u8> = Vec::new();
        let mut encode_ok = true;
        for (gop_idx, harvest) in per_gop_harvests.iter().enumerate() {
            let gop_offset = per_gop_cover_offsets[gop_idx];
            let gop_n = per_gop_cover_lens[gop_idx];

            // Build chunk_frame v3 payload bits for this GOP. First GOP
            // carries clip-level total_message_bytes; subsequent GOPs
            // carry only payload_len.
            let framed = if gop_idx == 0 {
                build_first_chunk_frame(total_message_bytes, &chunks[gop_idx])
            } else {
                build_chunk_frame(&chunks[gop_idx])
            }
            .map_err(Av1StegoError::Stego)?;
            let payload_bits = frame::bytes_to_bits(&framed);
            let m_bits = payload_bits.len();
            if m_bits == 0 || m_bits > gop_n {
                encode_ok = false;
                break;
            }
            let w = (gop_n / m_bits).max(1);
            let n_used = m_bits * w;

            // Project per-GOP-local shadow slots (slot.cover_index -
            // gop_offset for the subset in [gop_offset, gop_offset +
            // gop_n)).
            let mut gop_shadow_positions: Vec<Vec<(usize, u8)>> =
                vec![Vec::new(); shadow_states.len()];
            for (s_idx, state) in shadow_states.iter().enumerate() {
                for (bit_idx, slot) in state.positions.iter().enumerate().take(state.n_total) {
                    if slot.cover_index >= gop_offset
                        && slot.cover_index < gop_offset + gop_n
                    {
                        let local_idx = slot.cover_index - gop_offset;
                        gop_shadow_positions[s_idx].push((local_idx, state.bits[bit_idx]));
                    }
                }
            }

            // Stamp shadow LSBs into a clone of the GOP cover for
            // STC's view. Track positions for ∞-cost overlay +
            // post-STC defensive stamp + out-of-range overrides.
            let mut combined_cover: Vec<u8> = harvest.cover_bits.clone();
            let mut shadow_position_set: std::collections::HashSet<usize> =
                std::collections::HashSet::new();
            for slots in &gop_shadow_positions {
                for &(local_idx, bit) in slots {
                    if local_idx < n_used {
                        combined_cover[local_idx] = bit;
                    }
                    shadow_position_set.insert(local_idx);
                }
            }
            let original_full_cover: Vec<u8> = harvest.cover_bits.clone();

            // ∞-cost overlay.
            let mut costs_for_stc: Vec<f32> = harvest.costs[..n_used].to_vec();
            for &idx in &shadow_position_set {
                if idx < n_used {
                    costs_for_stc[idx] = f32::INFINITY;
                }
            }

            let cover_used = &combined_cover[..n_used];
            let hhat_matrix = hhat::generate_hhat(STC_H, w, &hhat_seed);
            let mut embed = match stc_embed(
                cover_used,
                &costs_for_stc,
                &payload_bits,
                &hhat_matrix,
                STC_H,
                w,
            ) {
                Some(p) => p,
                None => {
                    encode_ok = false;
                    break;
                }
            };

            // Defensive shadow stamp on the STC plan.
            for slots in &gop_shadow_positions {
                for &(local_idx, bit) in slots {
                    if local_idx < n_used {
                        embed.stego_bits[local_idx] = bit;
                    }
                }
            }

            // Split STC stego_bits into per-frame OverrideMaps.
            let mut per_frame_plans: Vec<OverrideMap> =
                (0..harvest.frame_starts.len()).map(|_| OverrideMap::new()).collect();
            for i in 0..n_used {
                if embed.stego_bits[i] != original_full_cover[i] {
                    let (frame_idx, cursor) = harvest.frame_cursor_at(i);
                    per_frame_plans[frame_idx].set(cursor, embed.stego_bits[i] as u16);
                }
            }
            // Out-of-range shadow override entries.
            for slots in &gop_shadow_positions {
                for &(local_idx, bit) in slots {
                    if local_idx >= n_used {
                        if bit != original_full_cover[local_idx] {
                            let (frame_idx, cursor) = harvest.frame_cursor_at(local_idx);
                            per_frame_plans[frame_idx].set(cursor, bit as u16);
                        }
                    }
                }
            }

            // Per-frame replay + splice.
            let frames = &per_gop_natural[gop_idx];
            for (frame_idx, (natural_packet, recording)) in frames.iter().enumerate() {
                let tile = &recording.tiles[0];
                let mut sink = WriterEncoder::new();
                replay_with_overrides(
                    &tile.storage,
                    &tile.bit_positions,
                    &per_frame_plans[frame_idx],
                    &mut sink,
                );
                let stego_tile_bytes = sink.done();

                let final_packet = if stego_tile_bytes.len() == recording.tile_group_len {
                    let mut packet = natural_packet.clone();
                    let dst = &mut packet[recording.tile_group_offset
                        ..recording.tile_group_offset + recording.tile_group_len];
                    dst.copy_from_slice(&stego_tile_bytes);
                    packet
                } else {
                    rebuild_obu_with_stego_tile_group(
                        natural_packet,
                        recording,
                        &stego_tile_bytes,
                    )
                };
                output.extend_from_slice(&final_packet);
            }
        }

        if !encode_ok {
            continue;
        }

        // Verify: walk output via dav1d, harvest union cover, try
        // shadow extract for each shadow. ALL must round-trip
        // through AES-GCM-SIV authentication for this parity rung to
        // be accepted.
        if verify_shadows_round_trip(&output, shadows) {
            return Ok(output);
        }
    }

    Err(Av1StegoError::Stego(
        crate::stego::error::StegoError::FrameCorrupted,
    ))
}

// ────────────────────────────────────────────────────────────────
// Internal helpers
// ────────────────────────────────────────────────────────────────

fn expected_i420_size(width: u32, height: u32) -> usize {
    let y = (width as usize) * (height as usize);
    let uv = ((width as usize) / 2) * ((height as usize) / 2);
    y + 2 * uv
}

/// Encode one GOP naturally (mirror of session.rs::encode_one_gop_multi
/// but inline-d here to avoid an inter-module pub dependency).
fn encode_gop_natural(
    gop_yuv: &[u8],
    frames_in_gop: u32,
    params: Av1StreamingEncodeParams,
) -> Result<Vec<(Vec<u8>, PhasmFrameRecording<u8>)>, Av1StegoError> {
    let w = params.width as usize;
    let h = params.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    let frame_size = y_size + 2 * uv_size;

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
            frame_in.planes[0].copy_from_raw_u8(&gop_yuv[off..off + y_size], w, 1);
            frame_in.planes[1].copy_from_raw_u8(
                &gop_yuv[off + y_size..off + y_size + uv_size],
                w / 2,
                1,
            );
            frame_in.planes[2].copy_from_raw_u8(
                &gop_yuv[off + y_size + uv_size..off + frame_size],
                w / 2,
                1,
            );
            Arc::new(frame_in)
        })
        .collect();

    Ok(encode_gop_with_phasm_tee::<u8>(&yuvs, config, sequence))
}

/// Harvested per-GOP Tier-1 cover + cost + per-bit back-index.
struct GopHarvest {
    cover_bits: Vec<u8>,
    costs: Vec<f32>,
    /// Per-bit (frame_idx, frame_cursor) — same shape as the
    /// `combined_index` in `av1_stego_encode_one_gop_with_shadows_parity`.
    combined_index: Vec<(usize, u64)>,
    /// Inclusive index where each frame's cover bits start (used to
    /// derive frame_cursor_at if combined_index becomes too large to
    /// stash on big resolutions; currently keeps the full
    /// per-bit vector for simplicity).
    frame_starts: Vec<usize>,
}

impl GopHarvest {
    fn frame_cursor_at(&self, idx: usize) -> (usize, u64) {
        self.combined_index[idx]
    }
}

fn harvest_gop(
    per_frame: &[(Vec<u8>, PhasmFrameRecording<u8>)],
) -> Result<GopHarvest, Av1StegoError> {
    let mut cover_bits: Vec<u8> = Vec::new();
    let mut costs: Vec<f32> = Vec::new();
    let mut combined_index: Vec<(usize, u64)> = Vec::new();
    let mut frame_starts: Vec<usize> = Vec::with_capacity(per_frame.len());

    for (frame_idx, (_, recording)) in per_frame.iter().enumerate() {
        if recording.tiles.is_empty() {
            return Err(Av1StegoError::EmptyRecording);
        }
        let tile = &recording.tiles[0];

        let mut frame_cursors: Vec<u64> = Vec::new();
        let mut frame_bits: Vec<u8> = Vec::new();
        let mut frame_metas: Vec<AcSignMeta> = Vec::new();

        for (cursor, ((&(_, value), &tag), &meta)) in tile
            .bit_positions
            .iter()
            .zip(tile.bit_tags.iter())
            .zip(tile.bit_meta.iter())
            .enumerate()
        {
            if tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB {
                frame_cursors.push(cursor as u64);
                frame_bits.push(value as u8);
                frame_metas.push(meta);
            }
        }

        let frame_planes = pack_visible_planes(&recording.reconstructed_planes);
        let av1_positions: Vec<Av1FramePosition> = frame_metas
            .iter()
            .map(|m| Av1FramePosition {
                plane: m.plane,
                plane_px_x: m.plane_px_x,
                plane_px_y: m.plane_px_y,
                tx_width_log2: m.tx_width_log2,
                tx_height_log2: m.tx_height_log2,
                tx_type: m.tx_type,
                scan_pos: m.scan_pos,
                coeff_magnitude: m.coeff_magnitude,
            })
            .collect();
        let frame_costs = compute_av1_uniward_costs_with_state(
            &frame_planes,
            &av1_positions,
            recording.frame_qindex,
            Some(recording.loop_filter_state),
        );

        frame_starts.push(cover_bits.len());
        for &c in &frame_cursors {
            combined_index.push((frame_idx, c));
        }
        cover_bits.extend_from_slice(&frame_bits);
        costs.extend_from_slice(&frame_costs);
    }

    Ok(GopHarvest {
        cover_bits,
        costs,
        combined_index,
        frame_starts,
    })
}

/// Verify gate — walk the assembled output via dav1d, harvest union
/// cover, run each shadow's authenticated extract. AES-GCM-SIV
/// authentication is byte-perfect: a successful decrypt proves the
/// extracted bytes equal the encryptor's input. So extract-succeeds
/// is sufficient — no need to re-compare to the original payload.
fn verify_shadows_round_trip(
    av1_bytes: &[u8],
    shadows: &[(&str, &[u8])],
) -> bool {
    // Walk the full output as ONE blob — av1_shadow_extract operates
    // on the harvested cover regardless of GOP boundaries.
    let cover = match harvest_cover_bits_from_stego(av1_bytes) {
        Ok(v) => v,
        Err(_) => return false,
    };
    for (passphrase, _) in shadows {
        if av1_shadow_extract(&cover, passphrase).is_err() {
            return false;
        }
    }
    true
}
