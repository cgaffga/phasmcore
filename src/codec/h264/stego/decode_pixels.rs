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
use crate::stego::stc::extract::{stc_extract, stc_extract_prefix};
use crate::stego::stc::hhat::generate_hhat;
use crate::stego::{crypto, payload};
use crate::stego::payload::PayloadData;

use crate::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover, walk_annex_b_for_cover_with_options,
    walk_nalus_for_cover, walk_nalus_for_cover_with_options, WalkOptions,
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
///
/// STEGO.A.11.fix — was calling Scheme B-only `decode_from_cover`,
/// which missed shadow + Scheme A combined STC entirely (iOS users
/// decoding STEGO.A-produced shadow stego saw multi-minute hangs
/// finding nothing). Now routes through the same 3-tier
/// `smart_decode_from_walked_cover` helper that
/// `h264_stego_smart_decode_video` uses.
pub fn h264_stego_decode_nalus_string(
    nalus: &[NalUnit],
    passphrase: &str,
) -> Result<String, StegoError> {
    h264_stego_decode_nalus_with_payload(nalus, passphrase).map(|p| p.text)
}

/// STEGO.A.11.fix — `_with_payload` variant of
/// [`h264_stego_decode_nalus_string`]. Reuses the unified 3-tier
/// cover-based decode (shadow → Scheme A combined → Scheme B fallback).
pub fn h264_stego_decode_nalus_with_payload(
    nalus: &[NalUnit],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let walk = walk_nalus_for_cover_with_options(
        nalus,
        WalkOptions { record_mvd: true, record_offsets: false },
    )
    .map_err(|e| StegoError::InvalidVideo(format!("walk nalus: {e}")))?;
    smart_decode_from_walked_cover(
        walk.cover, &walk.mvd_meta, walk.mb_w, walk.mb_h, passphrase,
    )
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

/// Phase 6D.8 §30D-C — 4-domain decode entry point.
///
/// STEGO.B.P8 (2026-05-24) — migrated to route through the unified
/// 2-tier `smart_decode_from_walked_cover` (shadow → Scheme A
/// combined extract) now that the legacy bridge-shape encoder
/// (`h264_stego_encode_yuv_string_4domain_multigop`) emits Scheme A.
/// Same wire format as the production decode path; tests + the
/// provisional_emit research module keep working against the same
/// public API.
pub fn h264_stego_decode_yuv_string_4domain(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<String, StegoError> {
    let opts = WalkOptions { record_mvd: true, record_offsets: false };
    let walk = walk_annex_b_for_cover_with_options(annex_b, opts)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    smart_decode_from_walked_cover(
        walk.cover, &walk.mvd_meta, walk.mb_w, walk.mb_h, passphrase,
    )
    .map(|p| p.text)
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
    let opts = WalkOptions { record_mvd: true, record_offsets: false };
    let walk = walk_annex_b_for_cover_with_options(annex_b, opts)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    // §6E-A5(d).6 — derive the cascade-safe MvdSuffixLsb mask from
    // the walked meta. Encoder ran the IDENTICAL analysis on its
    // provisional walk (same content modulo shadow-override values
    // at sign-flip-invariant + safe-set-magnitude-flip-invariant
    // positions), so encoder + decoder land on the same priority
    // pool by §6F.2(j) construction.
    let safe_msb = super::cascade_safety::analyze_safe_mvd_subset(
        &walk.mvd_meta, walk.mb_w, walk.mb_h,
    );
    let safe_msl = super::cascade_safety::derive_msl_safe_from_msb(
        &walk.cover.mvd_sign_bypass.positions,
        &safe_msb,
        &walk.cover.mvd_suffix_lsb.positions,
    );
    let payload_data = super::shadow::shadow_extract_all4_safe(
        &walk.cover, passphrase, None, Some(&safe_msl),
    )?;
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
    h264_stego_smart_decode_video_with_payload(annex_b, passphrase).map(|p| p.text)
}

/// Task #97 — `_with_payload` variant returning the full
/// PayloadData (text + attached files) recovered from either the
/// primary STC plan or a matching shadow layer. Use this when the
/// caller needs file attachments in addition to text.
pub fn h264_stego_smart_decode_video_with_payload(
    annex_b: &[u8],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let opts = WalkOptions { record_mvd: true, record_offsets: false };
    let walk = walk_annex_b_for_cover_with_options(annex_b, opts)
        .map_err(|e| StegoError::InvalidVideo(format!("walk: {e}")))?;
    smart_decode_from_walked_cover(
        walk.cover, &walk.mvd_meta, walk.mb_w, walk.mb_h, passphrase,
    )
}

/// STEGO.A.11.fix — shared 3-tier cover-based decode used by BOTH
/// the Annex-B entry ([`h264_stego_smart_decode_video_with_payload`])
/// and the NALU-list entry
/// ([`h264_stego_decode_nalus_string_with_payload`], used by the iOS
/// CABAC path via `h264_pipeline::decode_cabac_via_chunk_6g`).
///
/// Pre-STEGO.A.11.fix, the NALU path called `decode_from_cover`
/// (legacy Scheme B per-domain extract only). That missed shadow
/// + Scheme A combined STC entirely, so iOS users decoding a
/// STEGO.A-produced shadow stego MP4 saw their decode brute-force
/// for minutes finding nothing. This helper fixes that — same 3-tier
/// flow as smart_decode_video, callable from either entry point.
pub(super) fn smart_decode_from_walked_cover(
    cover: DomainCover,
    mvd_meta: &[crate::codec::h264::stego::encoder_hook::MvdPositionMeta],
    mb_w: u32,
    mb_h: u32,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    let safe_msb = super::cascade_safety::analyze_safe_mvd_subset(
        mvd_meta, mb_w, mb_h,
    );
    let safe_msl = super::cascade_safety::derive_msl_safe_from_msb(
        &cover.mvd_sign_bypass.positions,
        &safe_msb,
        &cover.mvd_suffix_lsb.positions,
    );

    // Shadow attempt first — AES-GCM-SIV authentication ensures we
    // only return Ok if THIS passphrase matches a shadow layer.
    if let Ok(payload_data) = super::shadow::shadow_extract_all4_safe(
        &cover, passphrase, None, Some(&safe_msl),
    ) {
        return Ok(payload_data);
    }

    // STEGO.A.4 — Scheme A combined STC extract. Decoder is UNCHANGED
    // by the D' cascade-safety tier feature: encoder applies tier
    // filter via ∞-cost in `content_costs` (steers STC away from
    // filtered positions) — n_cover and the wire bitstream are
    // unchanged. Same pattern as the existing `safe_msl` cascade-safety
    // gate. Decoder walks → STC extract → AES decrypt, regardless of
    // which tier the encoder used.
    //
    // (Earlier D'.4 design used physical position filtering + decoder
    // brute-force; reverted in favor of ∞-cost which preserves wire
    // structure and keeps decoder simple.)
    decode_from_cover_4domain_combined_with_payload(cover, passphrase)
}

/// STEGO.A.4 — Scheme A combined STC extract for the primary
/// (non-shadow) message. Mirrors the encoder side
/// `pass2_stc_plan_combined_with_keys` + the existing
/// `streaming_session::try_extract_chunk_from_gop` extract pattern,
/// but operates on the WHOLE Annex-B's cover (not per-GOP) and
/// recovers a phasm v1/v2 frame (not a chunk_frame).
///
/// Decoder doesn't use content-adaptive costs — STC's syndrome
/// equation is cost-agnostic. We pass `DomainCosts::default()` so the
/// combine ordering matches the encoder (CSB → CSL → MSB → MSL).
fn decode_from_cover_4domain_combined_with_payload(
    cover: DomainCover,
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    use super::cost_weights::{combine_cover_4domain, CostWeights};
    use super::orchestrate::DomainCosts;

    let keys = CabacStegoMasterKeys::derive(passphrase)?;
    // Canonical "primary" seed in Scheme A — same choice the encoder
    // (`pass2_stc_plan_combined_with_keys`) makes.
    let hhat_seed = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0).hhat_seed;

    let dummy_costs = DomainCosts::default();
    let weights = CostWeights::default();
    let (combined_cover, _, _) = combine_cover_4domain(&cover, &dummy_costs, &weights);
    let n_cover = combined_cover.len();
    if n_cover == 0 {
        return Err(StegoError::InvalidVideo("empty cover".into()));
    }

    const STC_H: usize = 4;
    let max_m_total_bits = frame::MAX_FRAME_BITS.min(n_cover);
    let min_m_total_bits = FRAME_OVERHEAD * 8;

    let mut m_total = min_m_total_bits;
    while m_total <= max_m_total_bits {
        let w = n_cover / m_total;
        if w == 0 {
            break;
        }
        let used = m_total * w;
        let hhat = generate_hhat(STC_H, w, &hhat_seed);

        // Length-prefix early-reject: extract first 16 syndrome bits
        // (the v1 plaintext_len u16, or the v2 0x0000 sentinel).
        // Implausible values reject immediately; cuts the extract loop
        // from O(n_cover) to O(16w) per candidate.
        let prefix_bits = stc_extract_prefix(&combined_cover[..used], &hhat, w, 16);
        if prefix_bits.len() < 16 {
            m_total += 8;
            continue;
        }
        let prefix_bytes = bits_to_bytes_msb_first(&prefix_bits);
        let prefix_u16 = u16::from_be_bytes([prefix_bytes[0], prefix_bytes[1]]);
        // Accept if either:
        //  - prefix_u16 is a plausible v1 plaintext_len
        //    (1 ≤ len ≤ payload_capacity_for_m_total)
        //  - prefix_u16 == 0 (v2 sentinel — full extract will check
        //    the v2 length field)
        let payload_capacity = (m_total / 8).saturating_sub(FRAME_OVERHEAD);
        let accept_prefix = prefix_u16 == 0
            || (prefix_u16 as usize >= 1 && prefix_u16 as usize <= payload_capacity);
        if !accept_prefix {
            m_total += 8;
            continue;
        }

        // Full extract + parse + decrypt.
        let extracted = stc_extract(&combined_cover[..used], &hhat, w);
        let bits = &extracted[..m_total.min(extracted.len())];
        let bytes = bits_to_bytes_msb_first(bits);
        if let Ok(parsed) = frame::parse_frame(&bytes) {
            if let Ok(plaintext) = crypto::decrypt(
                &parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce,
            ) {
                if let Ok(payload_data) = payload::decode_payload(&plaintext) {
                    return Ok(payload_data);
                }
            }
        }
        m_total += 8;
    }

    Err(StegoError::FrameCorrupted)
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

    // #529 — length-prefix early-reject. Message emission order
    // (see concat below) is coeff_sign, coeff_suffix, mvd_sign,
    // mvd_suffix; the v1 plaintext_len u16 lives at the start of
    // the first non-empty domain.
    let prefix_domains = [
        PrefixDomain {
            cover_bits: &cover.coeff_sign_bypass.bits,
            seed: &seeds[0].1,
            m_d: split.coeff_sign_bypass.len(),
        },
        PrefixDomain {
            cover_bits: &cover.coeff_suffix_lsb.bits,
            seed: &seeds[1].1,
            m_d: split.coeff_suffix_lsb.len(),
        },
        PrefixDomain {
            cover_bits: &cover.mvd_sign_bypass.bits,
            seed: &seeds[2].1,
            m_d: split.mvd_sign_bypass.len(),
        },
        PrefixDomain {
            cover_bits: &cover.mvd_suffix_lsb.bits,
            seed: &seeds[3].1,
            m_d: split.mvd_suffix_lsb.len(),
        },
    ];
    if !m_total_passes_length_prefix(&prefix_domains, h, m_total_bits) {
        return None;
    }

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

/// #529 — descriptor for one domain's contribution to the message
/// prefix, in message-emission order.
struct PrefixDomain<'a> {
    cover_bits: &'a [u8],
    seed: &'a [u8; 32],
    m_d: usize,
}

/// #529 — extract the first `k_msg_bits` bits of the concatenated
/// message via per-domain [`stc_extract_prefix`] calls. Walks
/// `domains` in message order, taking up to `m_d` bits from each.
///
/// Pairs with [`length_prefix_consistent`] to give the same
/// brute-force m_total search the early-reject speed-up that
/// `decode_legacy_via_walker` (single-domain OpenH264 path) and the
/// streaming session (chunk-idx) already enjoy via #516.2 / #519.
fn extract_message_prefix(
    domains: &[PrefixDomain<'_>],
    h: usize,
    k_msg_bits: usize,
) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(k_msg_bits);
    for d in domains {
        if out.len() >= k_msg_bits {
            break;
        }
        if d.m_d == 0 {
            continue;
        }
        let n_d = d.cover_bits.len();
        if d.m_d > n_d {
            return None;
        }
        let w_d = n_d / d.m_d;
        if w_d == 0 {
            return None;
        }
        let take = (k_msg_bits - out.len()).min(d.m_d);
        let hhat = generate_hhat(h, w_d, d.seed);
        let cover_slice = &d.cover_bits[..d.m_d * w_d];
        let prefix = stc_extract_prefix(cover_slice, &hhat, w_d, take);
        if prefix.len() < take {
            return None;
        }
        out.extend_from_slice(&prefix);
    }
    if out.len() < k_msg_bits {
        return None;
    }
    Some(out)
}

/// #529 — verify the message prefix's length header is consistent
/// with the candidate `m_total_bits`. The phasm frame's first 2
/// bytes are `plaintext_len: u16 BE` (v1) or 0x0000 sentinel
/// followed by `u32 BE` plaintext length (v2). For ANY candidate
/// m_total, only ONE specific length value satisfies
/// `(FRAME_OVERHEAD + plaintext_len) * 8 == m_total` — wrong
/// candidates pass with ≈1/65536 probability for v1, much less for
/// v2. The full extract + CRC then kills the false-positive tail.
fn length_prefix_consistent(prefix_bytes: &[u8], m_total_bits: usize) -> bool {
    if prefix_bytes.len() < 2 {
        return false;
    }
    let v1_len = u16::from_be_bytes([prefix_bytes[0], prefix_bytes[1]]) as usize;
    if (FRAME_OVERHEAD + v1_len) * 8 == m_total_bits {
        return true;
    }
    if v1_len == 0 && prefix_bytes.len() >= 6 {
        let v2_len = u32::from_be_bytes([
            prefix_bytes[2], prefix_bytes[3], prefix_bytes[4], prefix_bytes[5],
        ]) as usize;
        if v2_len > u16::MAX as usize
            && (frame::FRAME_OVERHEAD_EXT + v2_len) * 8 == m_total_bits
        {
            return true;
        }
    }
    false
}

/// #529 — gate `m_total` against the 16-bit (or 48-bit fallback)
/// length prefix without doing the full per-domain extract. Returns
/// `true` if this m_total candidate is consistent with the recovered
/// length header (caller proceeds to full extract + CRC + decrypt),
/// `false` if it can be skipped.
fn m_total_passes_length_prefix(
    domains: &[PrefixDomain<'_>],
    h: usize,
    m_total_bits: usize,
) -> bool {
    let Some(prefix16) = extract_message_prefix(domains, h, 16) else {
        // Degenerate per-domain config (e.g. w_d=0): don't reject; let
        // the full extract path handle the edge case.
        return true;
    };
    let prefix_bytes = bits_to_bytes_msb_first(&prefix16);
    if length_prefix_consistent(&prefix_bytes, m_total_bits) {
        return true;
    }
    let v1_len = u16::from_be_bytes([prefix_bytes[0], prefix_bytes[1]]);
    if v1_len != 0 {
        return false;
    }
    // v2 sentinel — pull 48 bits and check the u32 length field.
    let Some(prefix48) = extract_message_prefix(domains, h, 48) else {
        return true;
    };
    let ext_bytes = bits_to_bytes_msb_first(&prefix48);
    length_prefix_consistent(&ext_bytes, m_total_bits)
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
