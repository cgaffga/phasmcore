// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Deep Cover with exact side information: raw pixels in, cover + stego out.
//!
//! The frontends used to compress the cover JPEG themselves (UIImage,
//! Bitmap.compress, canvas, `image` crate) and hand core the JPEG plus the
//! raw pixels. The side information core derived from the pixels was then
//! only an approximation of that encoder's rounding (other DCT, other colour
//! conversion), and `|e| ≈ ½` coefficients became nearly free changes
//! (`docs/design/image/deep-cover-exact-si.md`).
//!
//! Here core writes the cover itself with
//! [`crate::codec::jpeg::cover_writer`] (byte-identical to libjpeg-turbo
//! `cjpeg -quality Q`), so the pre-rounding coefficients — and hence the
//! side information — are exact by construction. The legacy
//! `ghost_encode_si*` entry points also get exact side information when
//! their cover JPEG is this writer's output for the same pixels (checked
//! coefficient by coefficient in `SideInfo::compute`).

use super::pipeline::{self, ShadowLayer};
use super::quality::EncodeQuality;
use super::side_info::{exact_error_code, SideInfo};
use crate::codec::jpeg::cover_writer::{self, Geometry};
use crate::codec::jpeg::JpegImage;
use crate::stego::error::StegoError;
use crate::stego::payload::FileEntry;

/// Default cover quality (IJG scale): the apps' and the CLI's Deep Cover
/// quality so far (UIImage 0.92, Bitmap.compress 92, canvas 0.92, CLI 92).
pub const DEEP_COVER_DEFAULT_QUALITY: u8 = 92;

/// The cover JPEG phasm writes for `raw_rgb` (row-major RGB, 3 bytes per
/// pixel) at IJG `quality`, and its exact side information.
pub fn deep_cover(
    raw_rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<(JpegImage, Vec<u8>, SideInfo), StegoError> {
    crate::stego::validate_encode_dimensions(width, height)?;
    let g = Geometry::new(width as usize, height as usize);
    let (bw, bh) = (g.y_blocks_wide(), g.y_blocks_tall());
    let qy = cover_writer::scaled_quant_table(&cover_writer::STD_LUMA_QT, quality);
    let mut errors = vec![0i8; bw * bh * 64];
    let bytes = cover_writer::encode_rgb(raw_rgb, width, height, quality, |br, bc, raw, q| {
        let base = (br * bw + bc) * 64;
        for k in 0..64 {
            errors[base + k] = exact_error_code(raw[k], q[k], qy[k]);
        }
    })?;
    let img = JpegImage::from_bytes(&bytes)?;
    Ok((img, bytes, SideInfo::exact_from_codes(errors, bw, bh)))
}

/// Just the cover JPEG bytes (what [`ghost_encode_deep_cover`] embeds into).
pub fn deep_cover_jpeg(raw_rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, StegoError> {
    crate::stego::validate_encode_dimensions(width, height)?;
    Ok(cover_writer::encode_rgb(raw_rgb, width, height, quality, |_, _, _, _| {})?)
}

/// Deep Cover encode: write the cover from `raw_rgb` at `quality`, embed
/// with exact side information. Returns the stego JPEG and its quality score.
pub fn ghost_encode_deep_cover(
    raw_rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    let (img, bytes, si) = deep_cover(raw_rgb, width, height, quality)?;
    pipeline::ghost_encode_impl(&bytes, message, files, passphrase, Some(si), Some(img))
}

/// [`ghost_encode_deep_cover`] with shadow messages.
#[allow(clippy::too_many_arguments)]
pub fn ghost_encode_deep_cover_with_shadows(
    raw_rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    shadows: &[ShadowLayer],
) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
    let (img, bytes, si) = deep_cover(raw_rgb, width, height, quality)?;
    pipeline::shadows_encode(&bytes, message, files, passphrase, shadows, Some(si), Some(img))
}

/// Ghost capacity of the Deep Cover cover these pixels produce: the exact-SI
/// maximum (nzAC / 4 frame bits; the writer's cover is exact by construction).
pub fn ghost_capacity_deep_cover(raw_rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<usize, StegoError> {
    let img = JpegImage::from_bytes(&deep_cover_jpeg(raw_rgb, width, height, quality)?)?;
    Ok(super::capacity::ghost_bands_exact(&img, 0, 0)?.max_bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textured(w: usize, h: usize) -> Vec<u8> {
        let mut s = 0x9E37_79B9u32;
        let mut v = Vec::with_capacity(w * h * 3);
        for y in 0..h {
            for x in 0..w {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let n = (s >> 26) as i32; // 0..63
                let base = ((x * 3 + y * 2) % 160) as i32 + 40;
                v.extend_from_slice(&[(base + n) as u8, (base + n / 2) as u8, (200 - base / 2 + n / 3) as u8]);
            }
        }
        v
    }

    #[test]
    fn exact_side_information_is_detected_and_consistent() {
        let (w, h) = (251usize, 211usize);
        let rgb = textured(w, h);
        let (img, bytes, si) = deep_cover(&rgb, w as u32, h as u32, 85).unwrap();
        assert!(si.is_exact());
        // The legacy entry point recognises the writer's cover and computes
        // the same exact side information.
        let qt = img.quant_table(0).unwrap().values;
        let si2 = SideInfo::compute(&rgb, w as u32, h as u32, img.dct_grid(0), &qt).unwrap();
        assert!(si2.is_exact());
        let n = img.dct_grid(0).coeffs().len();
        let mut ties = 0;
        for i in 0..n {
            assert_eq!(si.error_at(i), si2.error_at(i));
            assert!(si.error_at(i).abs() <= 0.5);
            ties += si.is_tie(i) as usize;
        }
        assert!(ties > 0, "islow ties occur at rate ~1/(8q)");
        // A cover from another encoder (one coefficient off) is not exact.
        let mut other = JpegImage::from_bytes(&bytes).unwrap();
        other.dct_grid_mut(0).coeffs_mut()[1] += 1;
        let si3 = SideInfo::compute(&rgb, w as u32, h as u32, other.dct_grid(0), &qt).unwrap();
        assert!(!si3.is_exact());
    }

    #[test]
    fn deep_cover_round_trips() {
        let (w, h) = (320usize, 240usize);
        let rgb = textured(w, h);
        let (stego, _) = ghost_encode_deep_cover(&rgb, w as u32, h as u32, 92, "deep cover exact", &[], "pw-1").unwrap();
        let out = pipeline::ghost_decode(&stego, "pw-1").unwrap();
        assert_eq!(out.text, "deep cover exact");
    }

    // -- Deep Cover capacity bonus (nzAC / 4 with exact side information) --

    use super::super::capacity::{
        count_nonzero_ac, estimate_capacity, estimate_capacity_si, estimate_shadow_capacity,
        estimate_shadow_capacity_si, ghost_bands, ghost_bands_si, plaintext_for_frame_bits,
    };
    use crate::stego::frame::FRAME_OVERHEAD;
    use crate::stego::payload::compressed_payload_size;

    /// A message whose payload is exactly `bytes` (incompressible text).
    fn payload_of(bytes: usize, seed: u64) -> String {
        use rand::{Rng, SeedableRng};
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        let alphabet: Vec<char> = (b'!'..=b'~').map(|b| b as char).collect();
        let mut msg: String = (0..bytes).map(|_| alphabet[rng.r#gen::<usize>() % alphabet.len()]).collect();
        // The payload is the flags byte plus the (raw or compressed) text:
        // trim or pad until it is exactly `bytes`.
        while compressed_payload_size(&msg, &[]) > bytes {
            msg.pop();
        }
        while compressed_payload_size(&msg, &[]) < bytes {
            msg.push(alphabet[rng.r#gen::<usize>() % alphabet.len()]);
        }
        assert_eq!(compressed_payload_size(&msg, &[]), bytes);
        msg
    }

    /// The writer cover of a textured image and its pixels.
    fn writer_cover(w: usize, h: usize) -> (Vec<u8>, Vec<u8>, JpegImage) {
        let rgb = textured(w, h);
        let bytes = deep_cover_jpeg(&rgb, w as u32, h as u32, DEEP_COVER_DEFAULT_QUALITY).unwrap();
        let img = JpegImage::from_bytes(&bytes).unwrap();
        (rgb, bytes, img)
    }

    /// The same cover with one luma AC coefficient changed: not the writer's
    /// output for the pixels, so `encode_si*` falls back to the floored
    /// approximate side information.
    fn foreign_cover(bytes: &[u8]) -> (Vec<u8>, JpegImage) {
        let mut img = JpegImage::from_bytes(bytes).unwrap();
        img.dct_grid_mut(0).coeffs_mut()[1] += 1;
        let out = img.to_bytes().unwrap();
        let img = JpegImage::from_bytes(&out).unwrap();
        (out, img)
    }

    #[test]
    fn exact_side_information_gets_the_nzac_over_4_maximum() {
        let (w, h) = (320usize, 240usize);
        let (rgb, bytes, img) = writer_cover(w, h);
        let nz = count_nonzero_ac(img.dct_grid(0));
        let plain = ghost_bands(&img, 0, 0).unwrap();
        let si = ghost_bands_si(&img, &rgb, w as u32, h as u32, 0, 0).unwrap();
        assert_eq!(plain.max_bytes, plaintext_for_frame_bits(nz / 5));
        assert_eq!(si.max_bytes, plaintext_for_frame_bits(nz / 4));
        assert!(si.max_bytes > plain.max_bytes);
        assert!(si.comfort_bytes > plain.comfort_bytes && si.caution_bytes > plain.caution_bytes);
        assert!(si.comfort_bytes <= si.caution_bytes && si.caution_bytes <= si.max_bytes);
        assert_eq!(estimate_capacity(&img).unwrap(), plain.max_bytes);
        assert_eq!(estimate_capacity_si(&img, &rgb, w as u32, h as u32).unwrap(), si.max_bytes);
        assert_eq!(
            ghost_capacity_deep_cover(&rgb, w as u32, h as u32, DEEP_COVER_DEFAULT_QUALITY).unwrap(),
            si.max_bytes
        );
        // Shadows: half the (exact-SI) maximum per shadow; the bands bound
        // the total in both cases.
        let s_plain = estimate_shadow_capacity(&img).unwrap();
        let s_si = estimate_shadow_capacity_si(&img, &rgb, w as u32, h as u32).unwrap();
        assert_eq!(s_si, crate::stego::shadow::shadow_plaintext_for_bits(nz / 4 / 2));
        assert!(s_si > s_plain);
        let with = ghost_bands_si(&img, &rgb, w as u32, h as u32, 1, 100).unwrap();
        assert!(with.max_bytes < si.max_bytes && with.max_bytes > ghost_bands(&img, 1, 100).unwrap().max_bytes);
        // Same bytes: a parse of the cover gives the same answer.
        let again = JpegImage::from_bytes(&bytes).unwrap();
        assert_eq!(ghost_bands_si(&again, &rgb, w as u32, h as u32, 0, 0).unwrap(), si);
    }

    #[test]
    fn approximate_side_information_keeps_nzac_over_5() {
        let (w, h) = (320usize, 240usize);
        let (rgb, bytes, _) = writer_cover(w, h);
        let (_, foreign) = foreign_cover(&bytes);
        let plain = ghost_bands(&foreign, 0, 0).unwrap();
        assert_eq!(ghost_bands_si(&foreign, &rgb, w as u32, h as u32, 0, 0).unwrap(), plain);
        assert_eq!(
            estimate_shadow_capacity_si(&foreign, &rgb, w as u32, h as u32).unwrap(),
            estimate_shadow_capacity(&foreign).unwrap()
        );
        // Pixels of another size do not belong to the cover: no bonus.
        let small = textured(240, 200);
        let exact = JpegImage::from_bytes(&bytes).unwrap();
        assert_eq!(ghost_bands_si(&exact, &small, 240, 200, 0, 0).unwrap(), ghost_bands(&exact, 0, 0).unwrap());
        // A buffer too short for the stated size is rejected, as encode_si does.
        let img = JpegImage::from_bytes(&bytes).unwrap();
        assert!(matches!(
            ghost_bands_si(&img, &rgb[..rgb.len() - 1], w as u32, h as u32, 0, 0),
            Err(StegoError::ImageTooSmall)
        ));
        assert!(matches!(ghost_bands_si(&img, &rgb, 0, h as u32, 0, 0), Err(StegoError::ImageTooSmall)));
    }

    /// Round trip at exactly the maximum the bands report, for exact SI
    /// (nzAC / 4) and for the floored fallback (nzAC / 5); one byte more
    /// does not fit the maximum (the frame-bit budget is tight), which is
    /// what the apps' capacity check rejects.
    #[test]
    fn round_trips_at_exactly_the_maximum() {
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode_si};
        let (w, h) = (320usize, 240usize);
        let (rgb, bytes, img) = writer_cover(w, h);
        let nz = count_nonzero_ac(img.dct_grid(0));
        let (foreign_bytes, foreign) = foreign_cover(&bytes);
        let nz_f = count_nonzero_ac(foreign.dct_grid(0));
        for (label, cover, cover_img, rate, nzac) in [
            ("exact", &bytes, &img, 4usize, nz),
            ("floored", &foreign_bytes, &foreign, 5, nz_f),
        ] {
            // The encode's own side information agrees with the bands.
            let qt = cover_img.quant_table(0).unwrap().values;
            let si = SideInfo::compute(&rgb, w as u32, h as u32, cover_img.dct_grid(0), &qt).unwrap();
            assert_eq!(si.is_exact(), rate == 4, "{label}");
            let max = ghost_bands_si(cover_img, &rgb, w as u32, h as u32, 0, 0).unwrap().max_bytes;
            assert_eq!(max, plaintext_for_frame_bits(nzac / rate), "{label}");
            assert!((FRAME_OVERHEAD + max) * 8 <= nzac / rate, "{label}: max fits");
            assert!((FRAME_OVERHEAD + max + 1) * 8 > nzac / rate, "{label}: max + 1 does not");
            let msg = payload_of(max, rate as u64);
            let stego = ghost_encode_si(cover, &rgb, w as u32, h as u32, &msg, "dc-bonus").unwrap();
            assert_eq!(ghost_decode(&stego, "dc-bonus").unwrap().text, msg, "{label}");
        }
        // The exact-SI maximum is over the plain one: a plain-JPEG / floored
        // cover's bands reject it.
        let exact_max = ghost_bands_si(&img, &rgb, w as u32, h as u32, 0, 0).unwrap().max_bytes;
        assert!(exact_max > ghost_bands_si(&foreign, &rgb, w as u32, h as u32, 0, 0).unwrap().max_bytes);
    }

    /// Shadows on an exact-SI cover: one shadow at the exact-SI shadow
    /// maximum and a primary at the bands' maximum next to it both decode.
    #[test]
    fn shadows_round_trip_at_the_exact_si_maxima() {
        use crate::stego::ghost::pipeline::{ghost_decode, ghost_encode_si_with_shadows, ghost_shadow_decode};
        let (w, h) = (320usize, 240usize);
        let (rgb, bytes, img) = writer_cover(w, h);
        let s_max = estimate_shadow_capacity_si(&img, &rgb, w as u32, h as u32).unwrap();
        // A shadow above the plain shadow maximum; a little under the
        // exact-SI one so the primary next to it stays the larger payload
        // (the auto-sort would swap them otherwise).
        let shadow_len = s_max - 64;
        assert!(shadow_len > estimate_shadow_capacity(&img).unwrap(), "{shadow_len}");
        let p_max = ghost_bands_si(&img, &rgb, w as u32, h as u32, 1, shadow_len).unwrap().max_bytes;
        assert!(p_max > shadow_len);
        let primary = payload_of(p_max, 11);
        let shadow = payload_of(shadow_len, 12);
        let layers = [ShadowLayer { message: shadow.clone(), passphrase: "dc-shadow".into(), files: vec![] }];
        let stego = ghost_encode_si_with_shadows(&bytes, &rgb, w as u32, h as u32, &primary, &[], "dc-primary", &layers).unwrap();
        assert_eq!(ghost_decode(&stego, "dc-primary").unwrap().text, primary);
        assert_eq!(ghost_shadow_decode(&stego, "dc-shadow").unwrap().text, shadow);
    }
}
