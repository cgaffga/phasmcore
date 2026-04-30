// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase 6E-C2 polish — provisional emit + primary-emit cover walk.
//!
//! Phase 2 of the §6E-C2 polish encoder flow needs the cover the
//! decoder will see. Decoder priority-sorts shadow positions over
//! the FINAL EMIT cover; encoder must use the SAME cover for shadow
//! position selection. This helper produces a "provisional emit"
//! (primary plan only, no shadow injection) and walks it back to
//! the 4-domain cover — which serves as the encoder's stand-in for
//! "what the decoder will see," since shadow overrides on top of
//! primary plan only flip values at known positions and don't
//! introduce new bypass-bin positions (modulo boundary cases that
//! cascade absorbs).
//!
//! ## Status (this commit — utility only)
//!
//! Wraps the existing primary-only encode entry point
//! `h264_stego_encode_yuv_string_4domain_multigop` and the
//! existing §6E-C0 walker. No production wiring into the
//! `_with_n_shadows` cascade loop yet — that lands in commit 3 of
//! the polish sequence (single-cover shadow position selection
//! refactor).
//!
//! See `docs/design/h264-shadow-messages.md` § "§6E-C2 polish"
//! for the full architecture.

use crate::codec::h264::cabac::bin_decoder::{
    walk_annex_b_for_cover_with_options, WalkOptions,
};
use crate::stego::error::StegoError;

use super::encode_pixels::h264_stego_encode_yuv_string_4domain_multigop;
use super::inject::DomainCover;

/// Run a provisional Pass 3 emit (primary plan only, no shadow
/// overrides) and walk the resulting Annex-B bytes back to a
/// 4-domain `DomainCover`.
///
/// Used by §6E-C2 polish encoder flow to obtain the cover that
/// shadow position selection will sort over.
///
/// **NB**: this function performs a full primary-only encode (Pass
/// 1 + Pass 2A + Pass 1B + Pass 2B + Pass 3 inside
/// `h264_stego_encode_yuv_string_4domain_multigop`) plus a single
/// walk pass. The total cost is roughly 2× the underlying
/// primary-only encode (encode + walk).
///
/// On success returns `(provisional_annex_b_bytes, primary_emit_cover)`.
pub fn pass3_emit_provisional(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    gop_size: usize,
    primary_message: &str,
    primary_passphrase: &str,
) -> Result<(Vec<u8>, DomainCover), StegoError> {
    let bytes = h264_stego_encode_yuv_string_4domain_multigop(
        yuv, width, height, n_frames, gop_size,
        primary_message, primary_passphrase,
    )?;
    let walk_opts = WalkOptions { record_mvd: true };
    let walk = walk_annex_b_for_cover_with_options(&bytes, walk_opts)
        .map_err(|e| StegoError::InvalidVideo(format!("provisional walk: {e}")))?;
    Ok((bytes, walk.cover))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::h264::stego::decode_pixels::h264_stego_decode_yuv_string_4domain;
    use crate::codec::h264::stego::encode_pixels::h264_stego_encode_yuv_string_with_n_shadows;

    /// Mid-gray ± small per-frame perturbation. Avoids scene-change
    /// IDRs and matches `correlated_yuv` from `encode_pixels::tests`.
    fn make_yuv(width: u32, height: u32, n_frames: usize) -> Vec<u8> {
        let frame_size = (width * height * 3 / 2) as usize;
        let mut out = Vec::with_capacity(frame_size * n_frames);

        let mut base = Vec::with_capacity(frame_size);
        let mut s: u32 = 0x1234_5678;
        for _ in 0..frame_size {
            s = s.wrapping_mul(1103515245).wrapping_add(12345);
            let v = 64i32 + ((s >> 24) & 0x7F) as i32;
            base.push(v.clamp(0, 255) as u8);
        }

        for fi in 0..n_frames {
            let mut p: u32 = 0x4242_DEAD ^ (fi as u32 * 17);
            for &b in &base {
                p = p.wrapping_mul(1103515245).wrapping_add(12345);
                let delta = ((p >> 28) & 0x07) as i32 - 4;
                let v = b as i32 + delta;
                out.push(v.clamp(0, 255) as u8);
            }
        }
        out
    }

    #[test]
    fn provisional_emit_round_trips_primary() {
        let yuv = make_yuv(32, 32, 1);
        let (bytes, cover) = pass3_emit_provisional(
            &yuv, 32, 32, 1, 1,
            "primary message", "primary-pass",
        ).expect("provisional emit");

        assert!(!bytes.is_empty(), "provisional bytes must be non-empty");
        let total_cover_bits = cover.coeff_sign_bypass.len()
            + cover.coeff_suffix_lsb.len()
            + cover.mvd_sign_bypass.len()
            + cover.mvd_suffix_lsb.len();
        assert!(
            total_cover_bits > 0,
            "provisional walk must yield non-empty cover"
        );

        let decoded = h264_stego_decode_yuv_string_4domain(&bytes, "primary-pass")
            .expect("decode provisional bytes");
        assert_eq!(decoded, "primary message");
    }

    /// Equivalence test: provisional emit and the multi-shadow
    /// encoder with an empty shadow slice both decode back to the
    /// same primary message. Raw bytes differ across calls because
    /// `crypto::encrypt` generates a random nonce per call; the
    /// invariant we care about is decoded equivalence.
    #[test]
    fn provisional_emit_decodes_same_as_no_shadow_full_emit() {
        let yuv = make_yuv(32, 32, 1);

        let (provisional_bytes, _cover) = pass3_emit_provisional(
            &yuv, 32, 32, 1, 1,
            "match test", "primary-pass",
        ).expect("provisional emit");

        let full_bytes = h264_stego_encode_yuv_string_with_n_shadows(
            &yuv, 32, 32, 1, 1,
            "match test", "primary-pass",
            &[],
        ).expect("no-shadow full encode");

        let prov_decoded =
            h264_stego_decode_yuv_string_4domain(&provisional_bytes, "primary-pass")
                .expect("decode provisional");
        let full_decoded =
            h264_stego_decode_yuv_string_4domain(&full_bytes, "primary-pass")
                .expect("decode no-shadow full");

        assert_eq!(prov_decoded, "match test");
        assert_eq!(full_decoded, "match test");
        assert_eq!(prov_decoded, full_decoded);
    }

    #[test]
    fn provisional_emit_multi_gop_round_trip() {
        let yuv = make_yuv(32, 32, 4);
        let (bytes, cover) = pass3_emit_provisional(
            &yuv, 32, 32, 4, 2,
            "multi-gop msg", "pass-mg",
        ).expect("provisional multi-gop emit");

        assert!(!bytes.is_empty());
        assert!(
            cover.coeff_sign_bypass.len() + cover.mvd_sign_bypass.len() > 0,
            "multi-GOP provisional walk must yield bypass-bin positions"
        );

        let decoded = h264_stego_decode_yuv_string_4domain(&bytes, "pass-mg")
            .expect("decode multi-gop provisional bytes");
        assert_eq!(decoded, "multi-gop msg");
    }
}
