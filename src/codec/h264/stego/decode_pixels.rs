// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Phase 6D.8 chunk 6G — top-level decode entry point.
//
// `h264_stego_decode_yuv_string(annex_b, passphrase) → String`
//
// Pipeline:
//   1. Walk the Annex-B byte stream via the bin-decoder slice walker
//      → `DomainCover` (positions + bit values, four bypass-bin
//      domains). Chunk 6F's parity gate proves this cover is byte-
//      identical to the encode-side Pass-1 PositionLoggerHook
//      output on the same bitstream.
//   2. Brute-force search candidate `m_total` (total framed-bytes
//      bit count) over a bounded range. For each candidate:
//      a. Re-derive per-domain message lengths via
//         `split_message_per_domain` (encode-side splitter run in
//         reverse on a stub of length m_total).
//      b. Per domain: `w_d = n_d / m_d`; generate `hhat_d` with
//         passphrase-derived seed; `stc_extract` → m_d bits.
//      c. Concatenate per-domain bits in the canonical
//         `coeff_sign | coeff_suffix | mvd_sign | mvd_suffix` order.
//      d. Pack bits → bytes (MSB-first per `frame::build_frame`).
//      e. Try `frame::parse_frame` — CRC validates the candidate.
//      f. On CRC pass: `crypto::decrypt` + `payload::decode_payload`
//         → UTF-8 string.
//
// Brute-force bound: m_total starts at FRAME_OVERHEAD * 8 and ends
// at min(MAX_FRAME_BITS, total_cover_bits). With 32-bit CRC,
// false-positive rate per candidate is 2^-32; the bounded search
// is exact in practice.

use crate::stego::error::StegoError;
use crate::stego::frame::{self, FRAME_OVERHEAD};
use crate::stego::stc::extract::stc_extract;
use crate::stego::stc::hhat::generate_hhat;
use crate::stego::{crypto, payload};

use crate::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover, walk_annex_b_for_cover_with_options,
    walk_nalus_for_cover, WalkOptions,
};
use crate::codec::h264::NalUnit;

use super::hook::EmbedDomain;
use super::keys::CabacStegoMasterKeys;
use super::orchestrate::{split_message_per_domain, DomainMessages};
use super::DomainCover;

/// Decode an Annex-B byte stream produced by
/// `h264_stego_encode_yuv_string` (or the lower-level
/// `h264_stego_encode_i_frames_only`) and recover the original
/// UTF-8 message.
///
/// **Single-GOP I-frame-only scope** matches chunk-5 encode side.
/// P-slice support comes after §30 MVD wiring lands.
pub fn h264_stego_decode_yuv_string(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<String, StegoError> {
    let walk = walk_annex_b_for_cover(annex_b)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    decode_from_cover(walk.cover, passphrase)
}

/// Variant entry point that takes a pre-parsed NAL unit list
/// (e.g., from MP4-demuxed length-prefixed NAL bytes plus the
/// avcC SPS / PPS). Used by the chunk-7 cfg-gated branch in the
/// legacy `h264_ghost_decode` MP4 path.
pub fn h264_stego_decode_nalus_string(
    nalus: &[NalUnit],
    passphrase: &str,
) -> Result<String, StegoError> {
    let walk = walk_nalus_for_cover(nalus)
        .map_err(|e| StegoError::InvalidVideo(format!("walk nalus: {e}")))?;
    decode_from_cover(walk.cover, passphrase)
}

/// Shared body: brute-force m_total over the recovered cover.
fn decode_from_cover(
    cover: DomainCover,
    passphrase: &str,
) -> Result<String, StegoError> {

    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let seeds = [
        (EmbedDomain::CoeffSignBypass,
         keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0).hhat_seed),
        (EmbedDomain::CoeffSuffixLsb,
         keys.per_gop_seeds(EmbedDomain::CoeffSuffixLsb, 0).hhat_seed),
        (EmbedDomain::MvdSignBypass,
         keys.per_gop_seeds(EmbedDomain::MvdSignBypass, 0).hhat_seed),
        (EmbedDomain::MvdSuffixLsb,
         keys.per_gop_seeds(EmbedDomain::MvdSuffixLsb, 0).hhat_seed),
    ];

    // STC constraint length used at encode time. Must match the
    // chunk-5 driver default (h=4 in `h264_stego_encode_yuv_string`).
    const STC_H: usize = 4;

    let capacities = cover.capacity();
    let total_n_bits: usize = capacities.total();
    if total_n_bits == 0 {
        return Err(StegoError::InvalidVideo("empty cover".into()));
    }

    // Brute-force m_total in byte-aligned increments. Frame is
    // byte-aligned so step by 8. Cap at min(MAX_FRAME_BITS,
    // total cover capacity) to bound the search.
    let max_m_total_bits = (frame::MAX_FRAME_BITS).min(total_n_bits);
    let min_m_total_bits = FRAME_OVERHEAD * 8;

    let mut m_total = min_m_total_bits;
    while m_total <= max_m_total_bits {
        if let Some(plaintext) = try_decode_at(
            &cover, &seeds, STC_H, m_total, passphrase,
        ) {
            return Ok(plaintext);
        }
        m_total += 8;
    }

    Err(StegoError::FrameCorrupted)
}

/// Phase 6D.8 §30D-C — 4-domain decode entry point. Pairs with
/// `h264_stego_encode_yuv_string_4domain`. Walker opts into
/// `record_mvd: true` so MVD positions+bits land in the cover;
/// brute force m_total uses fill-MVD-first allocation matching
/// the encoder's 3-pass orchestrator.
pub fn h264_stego_decode_yuv_string_4domain(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<String, StegoError> {
    let opts = WalkOptions { record_mvd: true };
    let walk = walk_annex_b_for_cover_with_options(annex_b, opts)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    decode_from_cover_4domain(walk.cover, passphrase)
}

/// Phase 6E-C1b — decode a shadow message from the Annex-B byte
/// stream produced by `h264_stego_encode_yuv_string_with_shadow`.
/// Pairs the shadow side: walks the stego bytes → 4-domain cover,
/// then brute-forces parity tiers + first-block-peek for `fdl`.
/// AES-256-GCM-SIV authentication validates the chosen tier.
///
/// §6E-C2 polish — uses `shadow_extract_all4` (single-cover priority
/// sort across all 4 domains) to match the encoder's polish-era
/// position selection (`prepare_shadow_over_emit_cover`).
pub fn h264_stego_shadow_decode(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<String, StegoError> {
    let opts = WalkOptions { record_mvd: true };
    let walk = walk_annex_b_for_cover_with_options(annex_b, opts)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    let payload_data = super::shadow::shadow_extract_all4(&walk.cover, passphrase)?;
    Ok(payload_data.text)
}

/// Phase 6E-C1b — shadow-first smart decode. Tries shadow extract
/// first (cheaper — no STC reverse); on shadow miss, falls back to
/// primary decode. Returns the message belonging to whichever
/// passphrase actually authenticates.
pub fn h264_stego_smart_decode_video(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<String, StegoError> {
    let opts = WalkOptions { record_mvd: true };
    let walk = walk_annex_b_for_cover_with_options(annex_b, opts)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;

    // Shadow attempt first — AES-GCM-SIV authentication ensures we
    // only return Ok if THIS passphrase matches a shadow layer.
    // §6E-C2 polish — uses `shadow_extract_all4` to match the
    // encoder's single-cover position selection.
    if let Ok(payload_data) = super::shadow::shadow_extract_all4(&walk.cover, passphrase) {
        return Ok(payload_data.text);
    }

    // Primary fallback.
    decode_from_cover_4domain(walk.cover, passphrase)
}

/// Shared body for 4-domain decode: brute-force m_total over the
/// recovered cover, mirroring the encoder's fill-MVD-first split.
fn decode_from_cover_4domain(
    cover: DomainCover,
    passphrase: &str,
) -> Result<String, StegoError> {
    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    let seeds = [
        (EmbedDomain::CoeffSignBypass,
         keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0).hhat_seed),
        (EmbedDomain::CoeffSuffixLsb,
         keys.per_gop_seeds(EmbedDomain::CoeffSuffixLsb, 0).hhat_seed),
        (EmbedDomain::MvdSignBypass,
         keys.per_gop_seeds(EmbedDomain::MvdSignBypass, 0).hhat_seed),
        (EmbedDomain::MvdSuffixLsb,
         keys.per_gop_seeds(EmbedDomain::MvdSuffixLsb, 0).hhat_seed),
    ];

    const STC_H: usize = 4;
    let total_n_bits = cover.coeff_sign_bypass.len()
        + cover.coeff_suffix_lsb.len()
        + cover.mvd_sign_bypass.len()
        + cover.mvd_suffix_lsb.len();
    if total_n_bits == 0 {
        return Err(StegoError::InvalidVideo("empty cover".into()));
    }

    let max_m_total_bits = (frame::MAX_FRAME_BITS).min(total_n_bits);
    let min_m_total_bits = FRAME_OVERHEAD * 8;

    let mut m_total = min_m_total_bits;
    while m_total <= max_m_total_bits {
        if let Some(plaintext) = try_decode_at_4domain(
            &cover, &seeds, STC_H, m_total, passphrase,
        ) {
            return Ok(plaintext);
        }
        m_total += 8;
    }

    Err(StegoError::FrameCorrupted)
}

/// Try one candidate m_total under the §6F.2(k).4 stealth-
/// weighted cross-domain allocation. Both encoder and decoder
/// run `stealth_weighted_allocation` against the SAME cover
/// shape (cover_p1 ≡ walker_p3 — no cascade because MVD-sign
/// override doesn't mutate slot.value), so the per-domain
/// `m_d` split is identical on both sides by construction.
fn try_decode_at_4domain(
    cover: &DomainCover,
    seeds: &[(EmbedDomain, [u8; 32]); 4],
    h: usize,
    m_total_bits: usize,
    passphrase: &str,
) -> Option<String> {
    use super::hook::GopCapacity;

    // Phase 6F.2(k).4 — mirror encoder's stealth-weighted
    // allocation. mvd_suffix_lsb is forced to zero capacity
    // (cascades through median predictor → can't be inline-mod),
    // so the allocator gets exactly the same view as the
    // encoder's cap_for_alloc.
    let cap_for_alloc = GopCapacity {
        coeff_sign_bypass: cover.coeff_sign_bypass.len(),
        coeff_suffix_lsb: cover.coeff_suffix_lsb.len(),
        mvd_sign_bypass: cover.mvd_sign_bypass.len(),
        mvd_suffix_lsb: 0,
    };
    let allocator = super::orchestrate::StealthAllocator::v1_default();
    let (m_cs, m_cl, m_ms, _m_ml) =
        super::orchestrate::stealth_weighted_allocation(
            m_total_bits, &cap_for_alloc, &allocator,
        )?;
    let m_mvd = m_ms; // mvd_sign only; mvd_suffix is forced 0
    let m_residual = m_cs + m_cl;
    debug_assert_eq!(m_total_bits, m_mvd + m_residual,
        "decoder stealth-weighted alloc must conserve m_total");

    // Stage A split (MVD-sign-only capacity; suffix disabled).
    // Mirrors encoder's `cap_mvd_only`.
    let cap_mvd = GopCapacity {
        coeff_sign_bypass: 0,
        coeff_suffix_lsb: 0,
        mvd_sign_bypass: cover.mvd_sign_bypass.len(),
        mvd_suffix_lsb: 0,
    };
    let stub_mvd = vec![0u8; m_mvd];
    let split_a = split_message_per_domain(&stub_mvd, &cap_mvd)?;

    // Stage B split (coeff-only capacity).
    let cap_coeff = GopCapacity {
        coeff_sign_bypass: cover.coeff_sign_bypass.len(),
        coeff_suffix_lsb: cover.coeff_suffix_lsb.len(),
        mvd_sign_bypass: 0,
        mvd_suffix_lsb: 0,
    };
    let stub_residual = vec![0u8; m_residual];
    let split_b = split_message_per_domain(&stub_residual, &cap_coeff)?;

    // Per-domain STC extract.
    let mvd_sign = extract_one_domain(
        &cover.mvd_sign_bypass.bits,
        split_a.mvd_sign_bypass.len(),
        h, &seeds[2].1,
    )?;
    let mvd_suffix = extract_one_domain(
        &cover.mvd_suffix_lsb.bits,
        split_a.mvd_suffix_lsb.len(),
        h, &seeds[3].1,
    )?;
    let coeff_sign = extract_one_domain(
        &cover.coeff_sign_bypass.bits,
        split_b.coeff_sign_bypass.len(),
        h, &seeds[0].1,
    )?;
    let coeff_suffix = extract_one_domain(
        &cover.coeff_suffix_lsb.bits,
        split_b.coeff_suffix_lsb.len(),
        h, &seeds[1].1,
    )?;

    // Concat in encoder MESSAGE order: MVD bits FIRST (encoder put
    // them at message[..m_mvd]), then coeff bits.
    let mut all_bits = Vec::with_capacity(m_total_bits);
    all_bits.extend_from_slice(&mvd_sign);
    all_bits.extend_from_slice(&mvd_suffix);
    all_bits.extend_from_slice(&coeff_sign);
    all_bits.extend_from_slice(&coeff_suffix);
    if all_bits.len() != m_total_bits {
        return None;
    }

    let frame_bytes = bits_to_bytes_msb_first(&all_bits);
    let parsed = frame::parse_frame(&frame_bytes).ok()?;
    let plaintext = crypto::decrypt(
        &parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce,
    ).ok()?;
    let payload_data = payload::decode_payload(&plaintext).ok()?;
    Some(payload_data.text)
}

/// Try one candidate `m_total` (total framed bit count). Returns
/// `Some(plaintext)` on CRC + decrypt + payload-decode success;
/// `None` on any failure (wrong m_total or wrong passphrase).
fn try_decode_at(
    cover: &DomainCover,
    seeds: &[(EmbedDomain, [u8; 32]); 4],
    h: usize,
    m_total_bits: usize,
    passphrase: &str,
) -> Option<String> {
    // Re-derive per-domain m_d via the encode-side splitter.
    // Use a stub bit-stream of length m_total — split_message_per_domain
    // only inspects LENGTH, not values, when computing m_d.
    let stub: Vec<u8> = vec![0u8; m_total_bits];
    let capacities = cover.capacity();
    let split: DomainMessages = split_message_per_domain(&stub, &capacities)?;

    // Per-domain STC extract.
    let coeff_sign = extract_one_domain(
        &cover.coeff_sign_bypass.bits,
        split.coeff_sign_bypass.len(),
        h, &seeds[0].1,
    )?;
    let coeff_suffix = extract_one_domain(
        &cover.coeff_suffix_lsb.bits,
        split.coeff_suffix_lsb.len(),
        h, &seeds[1].1,
    )?;
    let mvd_sign = extract_one_domain(
        &cover.mvd_sign_bypass.bits,
        split.mvd_sign_bypass.len(),
        h, &seeds[2].1,
    )?;
    let mvd_suffix = extract_one_domain(
        &cover.mvd_suffix_lsb.bits,
        split.mvd_suffix_lsb.len(),
        h, &seeds[3].1,
    )?;

    // Concatenate in canonical order (matches encode-side
    // split_message_per_domain emit order).
    let mut all_bits = Vec::with_capacity(m_total_bits);
    all_bits.extend_from_slice(&coeff_sign);
    all_bits.extend_from_slice(&coeff_suffix);
    all_bits.extend_from_slice(&mvd_sign);
    all_bits.extend_from_slice(&mvd_suffix);
    if all_bits.len() != m_total_bits {
        return None;
    }

    let frame_bytes = bits_to_bytes_msb_first(&all_bits);

    // Frame parse is the CRC oracle.
    let parsed = frame::parse_frame(&frame_bytes).ok()?;

    // Decrypt with passphrase.
    let plaintext = crypto::decrypt(
        &parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce,
    ).ok()?;

    // Decode payload (brotli + inner format).
    let payload_data = payload::decode_payload(&plaintext).ok()?;
    Some(payload_data.text)
}

/// Run STC extract for one domain. Returns `Some(m_d bits)` on
/// success; `None` if the per-domain configuration is degenerate
/// (e.g., m_d > n_d, or w_d = 0).
fn extract_one_domain(
    cover_bits: &[u8],
    m_d: usize,
    h: usize,
    seed: &[u8; 32],
) -> Option<Vec<u8>> {
    if m_d == 0 {
        return Some(Vec::new());
    }
    let n_d = cover_bits.len();
    if m_d > n_d {
        return None;
    }
    let w_d = n_d / m_d;
    if w_d == 0 {
        return None;
    }
    let hhat = generate_hhat(h, w_d, seed);
    // Use exactly m_d * w_d cover bits to match the encode-side
    // STC embed slice (encoder uses w_d = n_d / m_d, embed sees
    // m_d * w_d bits of cover).
    let cover_slice = &cover_bits[..m_d * w_d];
    Some(stc_extract(cover_slice, &hhat, w_d))
}

/// Pack bits (one bit per byte, MSB-first per byte) into bytes.
/// Mirrors the bit-expansion used at encode time:
///   `byte = b7<<7 | b6<<6 | ... | b0`.
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
    use super::super::encode_pixels::h264_stego_encode_yuv_string;

    fn deterministic_yuv(w: u32, h: u32, n_frames: usize) -> Vec<u8> {
        let frame_size = (w * h * 3 / 2) as usize;
        let mut out = Vec::with_capacity(frame_size * n_frames);
        let mut s: u32 = 0xCAFE_F00D;
        for _ in 0..n_frames {
            for _ in 0..frame_size {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                out.push((s >> 16) as u8);
            }
        }
        out
    }

    /// Load-bearing end-to-end roundtrip gate.
    /// Encode "hi" with chunk-5 driver → walk + decode → assert "hi".
    #[test]
    fn roundtrip_short_string_32x32() {
        let yuv = deterministic_yuv(32, 32, 1);
        let pass = "test-pass";
        let msg = "hi";

        let bytes = h264_stego_encode_yuv_string(
            &yuv, 32, 32, 1, msg, pass,
        ).expect("encode");

        let recovered = h264_stego_decode_yuv_string(&bytes, pass)
            .expect("decode");
        assert_eq!(recovered, msg);
    }

    #[test]
    fn roundtrip_wrong_passphrase_fails() {
        let yuv = deterministic_yuv(32, 32, 1);
        let bytes = h264_stego_encode_yuv_string(
            &yuv, 32, 32, 1, "secret", "right",
        ).expect("encode");

        let r = h264_stego_decode_yuv_string(&bytes, "wrong");
        assert!(r.is_err(), "wrong passphrase must fail");
    }

    /// Chunk 8 — multi-row validation. 64x64 has 4x4=16 MBs across
    /// 4 rows; this exercises the new_row neighbor-state propagation
    /// at every row boundary (chunk 6F's regression-fix path).
    #[test]
    fn roundtrip_multi_row_64x64() {
        let yuv = deterministic_yuv(64, 64, 1);
        let pass = "multi-row";
        let msg = "test message across multi-row frame";

        let bytes = h264_stego_encode_yuv_string(
            &yuv, 64, 64, 1, msg, pass,
        ).expect("encode");
        let recovered = h264_stego_decode_yuv_string(&bytes, pass)
            .expect("decode");
        assert_eq!(recovered, msg);
    }

    /// **§30D-C 4-domain end-to-end roundtrip**: encode "msg" with
    /// the 3-pass orchestrator using MVD + residual domains, walk
    /// the resulting Annex-B with `record_mvd=true`, decode via
    /// the fill-MVD-first 4-domain decoder. Validates the entire
    /// 3-pass MVD-stego pipeline.
    #[test]
    fn roundtrip_i_then_p_frames_4domain_32x32() {
        use crate::codec::h264::stego::encode_pixels::
            h264_stego_encode_yuv_string_4domain;

        let yuv = deterministic_yuv(32, 32, 3);
        let pass = "p-4domain";
        let msg = "hello via 4-domain stego";

        let bytes = h264_stego_encode_yuv_string_4domain(
            &yuv, 32, 32, 3, msg, pass,
        ).expect("encode 4-domain");

        let recovered = h264_stego_decode_yuv_string_4domain(&bytes, pass)
            .expect("decode 4-domain");
        assert_eq!(recovered, msg);
    }

    /// **§30C P-slice multi-frame string roundtrip**: encode-walk-
    /// decode on YUV with 1 IDR + N P-frames. Validates STC reverse
    /// on the larger P-residual cover end-to-end.
    #[test]
    fn roundtrip_i_then_p_frames_32x32() {
        use crate::codec::h264::stego::encode_pixels::
            h264_stego_encode_yuv_string_i_then_p;

        // 32×32 × 3 frames (1 IDR + 2 P). High-entropy random YUV
        // ⇒ encoder picks varied P-partition modes; decoder walker
        // (§30A) reproduces byte-identical cover; STC reverse
        // recovers the message.
        let yuv = deterministic_yuv(32, 32, 3);
        let pass = "p-roundtrip";
        let msg = "hello via P-slices";

        let bytes = h264_stego_encode_yuv_string_i_then_p(
            &yuv, 32, 32, 3, msg, pass,
        ).expect("encode I+P");

        let recovered = h264_stego_decode_yuv_string(&bytes, pass)
            .expect("decode");
        assert_eq!(recovered, msg);
    }

    /// Chunk 8 — multi-frame validation. Three IDR frames in a row;
    /// confirms `stego_frame_idx` propagation across NAL-unit
    /// boundaries gives a coherent global cover that decodes back.
    /// Catches the encoder's `frame_num`-resets-per-IDR collision
    /// in `PlanInjector`'s HashMap (fixed alongside this test).
    #[test]
    fn roundtrip_multi_frame_32x32_3_idrs() {
        let yuv = deterministic_yuv(32, 32, 3);
        let pass = "multi-frame";
        let msg = "spread me across 3 IDR frames";

        let bytes = h264_stego_encode_yuv_string(
            &yuv, 32, 32, 3, msg, pass,
        ).expect("encode");
        let recovered = h264_stego_decode_yuv_string(&bytes, pass)
            .expect("decode");
        assert_eq!(recovered, msg);
    }

    #[test]
    fn bits_to_bytes_msb_first_roundtrip() {
        let original = [0xAB, 0xCD, 0x12];
        let bits: Vec<u8> = original.iter()
            .flat_map(|&b| (0..8).rev().map(move |i| (b >> i) & 1))
            .collect();
        let recovered = bits_to_bytes_msb_first(&bits);
        assert_eq!(recovered, original);
    }
}
