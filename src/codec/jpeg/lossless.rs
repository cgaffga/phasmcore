// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Lossless JPEG operations: metadata stripping and DCT-domain rotation /
//! flipping (what `jpegtran -copy none -trim` does), so a JPEG source can be
//! cleaned and turned upright **without** a decode / re-encode.
//!
//! Why: re-encoding a decoded JPEG at the same size makes an aligned
//! double-compressed cover (comb histogram), which is detectable under
//! Ghost and fatal with side information
//! (`eval/runs/2026-10-31-wand-ghost/RESULTS.md`, arm D). The coefficients
//! of the original file are the best cover there is; these operations keep
//! them (rotation only permutes blocks, transposes / sign-flips
//! coefficients and transposes the quantization tables).
//!
//! - [`JpegImage::strip_metadata`] keeps only what decoding needs: tables,
//!   frame, restart interval, a JFIF APP0 (added when missing, as libjpeg
//!   writes it), an Adobe APP14 (it carries the colour transform), and the
//!   ICC profile when asked. Everything else goes: EXIF (incl. GPS and
//!   thumbnail), XMP, IPTC / Photoshop APP13, MPF, maker APPn, COM. The
//!   bytes after EOI (MPF secondary images, gain maps) are never written.
//! - [`JpegImage::transform`] applies a [`Transform`]. Flips need whole MCUs
//!   along the flipped axis, so a partial MCU column / row at the right /
//!   bottom edge is dropped first (`jpegtran -trim`: at most 15 px).
//!   Transposes need no trim.
//! - [`upright_and_clean`] = EXIF orientation → transform, then strip.
//!
//! Huffman tables: the original ones when they still encode every symbol
//! (always true for a strip; usually not after a rotation, which changes run
//! lengths and DC differences), else the standard Annex K tables — what
//! jpegtran writes without `-optimize`.

use super::cover_writer::std_huffman_spec;
use super::dct::{DctGrid, QuantTable};
use super::error::{JpegError, Result};
use super::frame::parse_sof;
use super::marker::{self, MarkerSegment};
use super::JpegImage;

/// A lossless spatial transform (the eight EXIF orientations).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Transform {
    /// Mirror left ↔ right.
    FlipH,
    /// Mirror top ↔ bottom.
    FlipV,
    /// Swap x and y (mirror across the main diagonal).
    Transpose,
    /// Mirror across the anti-diagonal.
    Transverse,
    /// Rotate 90° clockwise.
    Rot90,
    /// Rotate 180°.
    Rot180,
    /// Rotate 270° clockwise (90° counter-clockwise).
    Rot270,
}

#[derive(Clone, Copy)]
enum Prim {
    T,
    H,
    V,
}

impl Transform {
    /// The transform that displays an image with EXIF `Orientation` tag
    /// `o` upright; `None` for 1 (already upright) and invalid values.
    pub fn from_exif_orientation(o: u16) -> Option<Self> {
        Some(match o {
            2 => Self::FlipH,
            3 => Self::Rot180,
            4 => Self::FlipV,
            5 => Self::Transpose,
            6 => Self::Rot90,
            7 => Self::Transverse,
            8 => Self::Rot270,
            _ => return None,
        })
    }

    fn prims(self) -> &'static [Prim] {
        match self {
            Self::FlipH => &[Prim::H],
            Self::FlipV => &[Prim::V],
            Self::Transpose => &[Prim::T],
            // (x, y) → (H−1−y, W−1−x): transpose, then both flips.
            Self::Transverse => &[Prim::T, Prim::H, Prim::V],
            // (x, y) → (H−1−y, x): transpose, then mirror the new width.
            Self::Rot90 => &[Prim::T, Prim::H],
            Self::Rot180 => &[Prim::H, Prim::V],
            // (x, y) → (y, W−1−x): transpose, then mirror the new height.
            Self::Rot270 => &[Prim::T, Prim::V],
        }
    }
}

fn transpose_block(b: &[i16]) -> [i16; 64] {
    std::array::from_fn(|i| b[(i % 8) * 8 + i / 8])
}

/// Coefficient (u, v) of a mirrored block: × (−1)^v for a horizontal mirror
/// (v = horizontal frequency, the column), × (−1)^u for a vertical one.
fn flip_block(b: &[i16], horizontal: bool) -> [i16; 64] {
    std::array::from_fn(|i| {
        let odd = if horizontal { i % 8 } else { i / 8 } & 1 == 1;
        if odd { -b[i] } else { b[i] }
    })
}

impl JpegImage {
    /// The EXIF `Orientation` tag (1–8) of the first EXIF APP1 segment.
    pub fn exif_orientation(&self) -> Option<u16> {
        self.raw_segments
            .iter()
            .filter(|s| s.marker == 0xE1 && s.data.starts_with(b"Exif\0\0"))
            .find_map(|s| exif_orientation_of(&s.data[6..]))
    }

    /// Drop every metadata segment (see the module docs); `keep_icc` keeps
    /// the ICC profile (APP2 `ICC_PROFILE`), which some viewers need for the
    /// right colours (e.g. Display P3 photos).
    pub fn strip_metadata(&mut self, keep_icc: bool) {
        let ncomp = self.frame.components.len();
        self.raw_segments.retain(|s| match s.marker {
            marker::DQT | marker::DHT | marker::DRI | 0xC0..=0xC3 => true,
            0xE0 => s.data.starts_with(b"JFIF\0"),
            0xE2 => keep_icc && s.data.starts_with(b"ICC_PROFILE\0"),
            0xEE => s.data.starts_with(b"Adobe"),
            _ => false,
        });
        let has_jfif = self.raw_segments.iter().any(|s| s.marker == 0xE0);
        let has_adobe = self.raw_segments.iter().any(|s| s.marker == 0xEE);
        if !has_jfif && !has_adobe && (ncomp == 1 || ncomp == 3) {
            // libjpeg's JFIF header: version 1.01, no units, density 1:1.
            self.raw_segments.insert(
                0,
                MarkerSegment { marker: 0xE0, data: b"JFIF\0\x01\x01\x00\x00\x01\x00\x01\x00\x00".to_vec() },
            );
        }
    }

    /// Apply `t` in the DCT domain (lossless; see the module docs for the
    /// edge trim). Errors when a flip leaves no whole MCU.
    pub fn transform(&mut self, t: Transform) -> Result<()> {
        for &p in t.prims() {
            match p {
                Prim::T => self.transpose(),
                Prim::H => self.flip(true)?,
                Prim::V => self.flip(false)?,
            }
        }
        self.reset_huffman_if_needed();
        Ok(())
    }

    fn transpose(&mut self) {
        let mut grids = Vec::with_capacity(self.grids.len());
        for g in &self.grids {
            let (bw, bh) = (g.blocks_wide(), g.blocks_tall());
            let mut n = DctGrid::new(bh, bw);
            for r in 0..bh {
                for c in 0..bw {
                    n.block_mut(c, r).copy_from_slice(&transpose_block(g.block(r, c)));
                }
            }
            grids.push(n);
        }
        self.grids = grids;
        let used: Vec<usize> = self.frame.components.iter().map(|c| c.quant_table_id as usize).collect();
        for id in 0..4 {
            if !used.contains(&id) {
                continue;
            }
            if let Some(qt) = &self.quant_tables[id] {
                let v = qt.values;
                self.quant_tables[id] = Some(QuantTable::new(std::array::from_fn(|i| v[(i % 8) * 8 + i / 8])));
            }
        }
        self.rebuild_dqt_segments();
        let (w, h) = (self.frame.width, self.frame.height);
        let comps: Vec<(u8, u8, u8, u8)> =
            self.frame.components.iter().map(|c| (c.id, c.v_sampling, c.h_sampling, c.quant_table_id)).collect();
        self.set_frame(h, w, &comps);
    }

    fn flip(&mut self, horizontal: bool) -> Result<()> {
        let f = &self.frame;
        let (mcu_px, len) = if horizontal { (f.mcu_width, f.width) } else { (f.mcu_height, f.height) };
        let mcus = len / mcu_px; // whole MCUs only (trim)
        if mcus == 0 {
            return Err(JpegError::InvalidDimensions);
        }
        let mut grids = Vec::with_capacity(self.grids.len());
        for (sci, g) in self.grids.iter().enumerate() {
            let comp = &self.frame.components[self.scan_components.get(sci).map_or(sci, |s| s.comp_idx)];
            let samp = if horizontal { comp.h_sampling } else { comp.v_sampling } as usize;
            let keep = mcus as usize * samp;
            let (bw, bh) = (g.blocks_wide(), g.blocks_tall());
            let (nw, nh) = if horizontal { (keep, bh) } else { (bw, keep) };
            let mut n = DctGrid::new(nw, nh);
            for r in 0..nh {
                for c in 0..nw {
                    let (sr, sc) = if horizontal { (r, keep - 1 - c) } else { (keep - 1 - r, c) };
                    n.block_mut(r, c).copy_from_slice(&flip_block(g.block(sr, sc), horizontal));
                }
            }
            grids.push(n);
        }
        self.grids = grids;
        let trimmed = mcus * mcu_px;
        let (w, h) = if horizontal { (trimmed, self.frame.height) } else { (self.frame.width, trimmed) };
        let comps: Vec<(u8, u8, u8, u8)> =
            self.frame.components.iter().map(|c| (c.id, c.h_sampling, c.v_sampling, c.quant_table_id)).collect();
        self.set_frame(w, h, &comps);
        Ok(())
    }

    /// New frame header (`width × height`, components `(id, h, v, tq)`), in
    /// `self.frame` and in the SOF0 segment.
    fn set_frame(&mut self, width: u16, height: u16, comps: &[(u8, u8, u8, u8)]) {
        let mut sof = vec![8u8];
        sof.extend_from_slice(&height.to_be_bytes());
        sof.extend_from_slice(&width.to_be_bytes());
        sof.push(comps.len() as u8);
        for &(id, h, v, tq) in comps {
            sof.extend_from_slice(&[id, (h << 4) | v, tq]);
        }
        self.frame = parse_sof(&sof).expect("valid frame");
        for s in self.raw_segments.iter_mut() {
            if (0xC0..=0xC3).contains(&s.marker) {
                *s = MarkerSegment { marker: marker::SOF0, data: sof.clone() };
            }
        }
    }

    /// Keep the Huffman tables when they still encode the coefficients,
    /// else install the standard Annex K tables (table 0 for the first
    /// component, table 1 for the others, as libjpeg assigns them).
    fn reset_huffman_if_needed(&mut self) {
        if self.to_bytes().is_ok() {
            return;
        }
        self.dc_huff_specs = [None, None, None, None];
        self.ac_huff_specs = [None, None, None, None];
        for sc in self.scan_components.iter_mut() {
            let id = if sc.comp_idx == 0 { 0 } else { 1 };
            sc.dc_table = id;
            sc.ac_table = id;
            self.dc_huff_specs[id] = Some(std_huffman_spec(0, id as u8));
            self.ac_huff_specs[id] = Some(std_huffman_spec(1, id as u8));
        }
        // SOS selectors: Ns, then (Cs, Td<<4 | Ta) per component, then Ss Se Ah/Al.
        let ns = self.sos_data.first().copied().unwrap_or(0) as usize;
        for i in 0..ns.min(self.scan_components.len()) {
            let t = self.scan_components[i].dc_table as u8;
            if let Some(b) = self.sos_data.get_mut(2 + 2 * i) {
                *b = (t << 4) | t;
            }
        }
        self.raw_segments.retain(|s| s.marker != marker::DHT);
        let sof_pos = self.raw_segments.iter().position(|s| s.marker == marker::SOF0).map_or(self.raw_segments.len(), |p| p + 1);
        let mut at = sof_pos;
        for id in 0..2 {
            for spec in [&self.dc_huff_specs[id], &self.ac_huff_specs[id]].into_iter().flatten() {
                let mut d = vec![(spec.class << 4) | spec.id];
                d.extend_from_slice(&spec.bits);
                d.extend_from_slice(&spec.huffval);
                self.raw_segments.insert(at, MarkerSegment { marker: marker::DHT, data: d });
                at += 1;
            }
        }
    }
}

/// `Orientation` (tag 0x0112) of IFD0 in a TIFF block (the EXIF payload
/// after `Exif\0\0`).
fn exif_orientation_of(t: &[u8]) -> Option<u16> {
    let le = match t.get(..4)? {
        b"II*\0" => true,
        b"MM\0*" => false,
        _ => return None,
    };
    let u16_at = |o: usize| t.get(o..o + 2).map(|b| if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) });
    let u32_at = |o: usize| {
        t.get(o..o + 4).map(|b| {
            let a = [b[0], b[1], b[2], b[3]];
            if le { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) }
        })
    };
    let ifd = u32_at(4)? as usize;
    let n = u16_at(ifd)? as usize;
    for i in 0..n {
        let e = ifd + 2 + 12 * i;
        if u16_at(e)? == 0x0112 && u16_at(e + 2)? == 3 {
            let v = u16_at(e + 8)?;
            return (1..=8).contains(&v).then_some(v);
        }
    }
    None
}

/// The JPEG made upright (EXIF orientation applied in the DCT domain) and
/// stripped of metadata, without re-encoding: `jpegtran -copy none -trim`
/// (+ `-copy icc` when `keep_icc`), with the orientation taken from EXIF.
pub fn upright_and_clean(jpeg: &[u8], keep_icc: bool) -> Result<Vec<u8>> {
    let mut img = JpegImage::from_bytes(jpeg)?;
    if let Some(t) = img.exif_orientation().and_then(Transform::from_exif_orientation) {
        img.transform(t)?;
    }
    img.strip_metadata(keep_icc);
    img.to_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::jpeg::cover_writer::encode_rgb;
    use crate::codec::jpeg::pixels::idct_block;

    fn textured(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut s = seed | 1;
        (0..w * h)
            .flat_map(|i| {
                s ^= s << 13;
                s ^= s >> 17;
                s ^= s << 5;
                let (x, y) = (i % w, i / w);
                let base = (x * 3 + y * 2) as u32 % 160 + 40;
                let n = s >> 27;
                [(base + n) as u8, (base / 2 + n * 3) as u8, (255 - base - n) as u8]
            })
            .collect()
    }

    fn jpeg(w: usize, h: usize) -> Vec<u8> {
        encode_rgb(&textured(w, h, 0x1234_5678 ^ (w * 31 + h) as u32), w as u32, h as u32, 85, |_, _, _, _| {}).unwrap()
    }

    /// Luma plane (unrounded float IDCT) cropped to the frame size.
    fn luma(img: &JpegImage) -> (Vec<f64>, usize, usize) {
        plane(img, 0)
    }

    /// Component `c`'s plane (unrounded float IDCT), cropped to its size
    /// (`⌈W · h / h_max⌉ × ⌈H · v / v_max⌉`).
    fn plane(img: &JpegImage, c: usize) -> (Vec<f64>, usize, usize) {
        let fi = img.frame_info();
        let comp = &fi.components[c];
        let g = img.dct_grid(c);
        let qt = img.quant_table(comp.quant_table_id as usize).unwrap().values;
        let pw = g.blocks_wide() * 8;
        let mut p = vec![0.0; pw * g.blocks_tall() * 8];
        for br in 0..g.blocks_tall() {
            for bc in 0..g.blocks_wide() {
                let px = idct_block(g.block(br, bc).try_into().unwrap(), &qt);
                for k in 0..64 {
                    p[(br * 8 + k / 8) * pw + bc * 8 + k % 8] = px[k];
                }
            }
        }
        let w = (fi.width as usize * comp.h_sampling as usize).div_ceil(fi.max_h_sampling as usize);
        let h = (fi.height as usize * comp.v_sampling as usize).div_ceil(fi.max_v_sampling as usize);
        ((0..w * h).map(|i| p[(i / w) * pw + i % w]).collect(), w, h)
    }

    /// The transform on a pixel plane, with the same whole-MCU trim.
    fn expect(p: &[f64], w: usize, h: usize, t: Transform, mcu: (usize, usize)) -> (Vec<f64>, usize, usize) {
        let (mut p, mut w, mut h, mut mcu) = (p.to_vec(), w, h, mcu);
        for &pr in t.prims() {
            match pr {
                Prim::T => {
                    p = (0..w * h).map(|i| p[(i % h) * w + i / h]).collect();
                    std::mem::swap(&mut w, &mut h);
                    mcu = (mcu.1, mcu.0);
                }
                Prim::H => {
                    let nw = w / mcu.0 * mcu.0;
                    p = (0..nw * h).map(|i| p[(i / nw) * w + (nw - 1 - i % nw)]).collect();
                    w = nw;
                }
                Prim::V => {
                    let nh = h / mcu.1 * mcu.1;
                    p = (0..w * nh).map(|i| p[(nh - 1 - i / w) * w + i % w]).collect();
                    h = nh;
                }
            }
        }
        (p, w, h)
    }

    fn close(a: &[f64], b: &[f64]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-6)
    }

    const ALL: [Transform; 7] = [
        Transform::FlipH,
        Transform::FlipV,
        Transform::Transpose,
        Transform::Transverse,
        Transform::Rot90,
        Transform::Rot180,
        Transform::Rot270,
    ];

    #[test]
    fn every_transform_moves_the_pixels_like_the_geometric_one() {
        // MCU-aligned and awkward sizes (4:2:0: 16 px MCUs, so 50×37 trims).
        for (w, h) in [(48usize, 32usize), (50, 37), (17, 64)] {
            let src = jpeg(w, h);
            let orig = JpegImage::from_bytes(&src).unwrap();
            let (lp, lw, lh) = luma(&orig);
            for t in ALL {
                let mut img = orig.clone();
                img.transform(t).unwrap();
                let bytes = img.to_bytes().unwrap();
                let back = JpegImage::from_bytes(&bytes).unwrap();
                let (gp, gw, gh) = luma(&back);
                let (ep, ew, eh) = expect(&lp, lw, lh, t, (16, 16));
                assert_eq!((gw, gh), (ew, eh), "{t:?} {w}x{h}");
                assert!(close(&gp, &ep), "{t:?} {w}x{h}");
                for c in 1..3 {
                    // 4:2:0 chroma: 8-sample MCUs on the half-size plane.
                    let (cp, cw, ch) = plane(&orig, c);
                    let (ep, ew, eh) = expect(&cp, cw, ch, t, (8, 8));
                    let (gp, gw, gh) = plane(&back, c);
                    assert_eq!((gw, gh), (ew, eh), "{t:?} {w}x{h} comp {c}");
                    assert!(close(&gp, &ep), "{t:?} {w}x{h} comp {c}");
                }
                // A third-party decoder reads it at the expected size.
                let dec = image::load_from_memory(&bytes).unwrap().to_rgb8();
                assert_eq!((dec.width() as usize, dec.height() as usize), (ew, eh));
            }
        }
    }

    #[test]
    fn four_quarter_turns_and_double_flips_are_the_identity() {
        let src = jpeg(64, 48);
        let orig = JpegImage::from_bytes(&src).unwrap();
        for (t, n) in [(Transform::Rot90, 4), (Transform::FlipH, 2), (Transform::FlipV, 2), (Transform::Transpose, 2)] {
            let mut img = orig.clone();
            for _ in 0..n {
                img.transform(t).unwrap();
            }
            for c in 0..3 {
                assert_eq!(img.dct_grid(c).coeffs(), orig.dct_grid(c).coeffs(), "{t:?} comp {c}");
            }
            assert_eq!(img.quant_table(0).unwrap().values, orig.quant_table(0).unwrap().values);
        }
    }

    fn with_exif(jpeg: &[u8], orientation: u16) -> Vec<u8> {
        let mut tiff = b"MM\x00\x2a\x00\x00\x00\x08\x00\x01".to_vec();
        tiff.extend_from_slice(&[0x01, 0x12, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01]);
        tiff.extend_from_slice(&orientation.to_be_bytes());
        tiff.extend_from_slice(&[0, 0, 0, 0, 0, 0]);
        let mut payload = b"Exif\x00\x00".to_vec();
        payload.extend_from_slice(&tiff);
        let mut out = jpeg[..2].to_vec();
        for (m, d) in [(0xE1u8, payload), (0xFEu8, b"a comment".to_vec()), (0xEDu8, b"Photoshop 3.0\0x".to_vec())] {
            out.extend_from_slice(&[0xFF, m]);
            out.extend_from_slice(&((d.len() + 2) as u16).to_be_bytes());
            out.extend_from_slice(&d);
        }
        out.extend_from_slice(&jpeg[2..]);
        out
    }

    #[test]
    fn strip_drops_metadata_and_keeps_the_coefficients() {
        let src = with_exif(&jpeg(48, 32), 1);
        let orig = JpegImage::from_bytes(&src).unwrap();
        assert_eq!(orig.exif_orientation(), Some(1));
        let mut img = orig.clone();
        img.strip_metadata(false);
        let out = img.to_bytes().unwrap();
        let back = JpegImage::from_bytes(&out).unwrap();
        assert!(back.raw_segments.iter().all(|s| !matches!(s.marker, 0xE1..=0xEF | 0xFE)));
        assert_eq!(back.raw_segments.iter().filter(|s| s.marker == 0xE0).count(), 1);
        for c in 0..3 {
            assert_eq!(back.dct_grid(c).coeffs(), orig.dct_grid(c).coeffs());
        }
        // Same scan bytes as the original (strip is a pure header edit).
        assert!(src.ends_with(&out[out.len() - 200..]));
    }

    #[test]
    fn upright_and_clean_applies_the_exif_orientation() {
        let plain = jpeg(48, 32);
        let (lp, lw, lh) = luma(&JpegImage::from_bytes(&plain).unwrap());
        for o in 1u16..=8 {
            let out = upright_and_clean(&with_exif(&plain, o), false).unwrap();
            let img = JpegImage::from_bytes(&out).unwrap();
            assert_eq!(img.exif_orientation(), None);
            let (gp, gw, gh) = luma(&img);
            let (ep, ew, eh) = match Transform::from_exif_orientation(o) {
                Some(t) => expect(&lp, lw, lh, t, (16, 16)),
                None => (lp.clone(), lw, lh),
            };
            assert_eq!((gw, gh), (ew, eh), "orientation {o}");
            assert!(close(&gp, &ep), "orientation {o}");
        }
    }
}
