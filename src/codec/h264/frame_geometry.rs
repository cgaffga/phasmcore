// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Exact-size H.264 frame geometry: display size vs the coded 16×16 MB grid.
//!
//! The stego encoder accepts any even frame size (≥ 16×16), like a phone
//! encoder. The bitstream codes the MB grid (`ceil(dim / 16)` MBs per axis)
//! and the SPS crops back to the display size on the right / bottom in 4:2:0
//! crop units — 1920×1080 is coded as 120×68 MBs with `crop_bottom = 4`,
//! exactly what an iPhone writes. OpenH264 emits that crop itself when it is
//! configured with the exact size, and pads its internal coded-size picture
//! with a constant fill.
//!
//! Two rules follow, and this module is the single place that owns them:
//!
//! - **MB grid.** Everything on the encode side that indexes macroblocks
//!   (override keys, `mb_type` tables, decision caches, cost-model MB
//!   addresses) uses `ceil(dim / 16)`, the same grid the walker reads from
//!   the SPS (`pic_width_in_mbs`). A floor would key every override on a
//!   wrong `mb_addr` and drop the last MB row / column.
//! - **Padded pixels.** Anything the encoder side derives from pixels (the
//!   J-UNIWARD content costs) must see the coded-size frame the encoder
//!   actually codes: the display pixels plus OpenH264's fill. See
//!   [`FrameGeometry::pad_frame_into`].
//!
//! YUV buffers stay **display-size tight I420** everywhere (sessions,
//! `GopYuvSource`s, whole-clip slices); only the cost model expands a frame to
//! the coded size, one frame at a time. For a 16-aligned size the coded and
//! display geometry coincide and no padding is ever built, so aligned encodes
//! are byte-identical to the pre-cropping code.
//!
//! Design note: `docs/design/video/h264/exact-size-frame-cropping.md`.

use crate::stego::error::StegoError;

/// Luma value OpenH264 writes into the padded (cropped-away) region of its
/// coded-size source picture. Mirrors `CWelsPreProcess::Padding` in the
/// fork's `codec/encoder/core/src/wels_preprocess.cpp` (`memset(.., 0, ..)`).
pub const ENCODER_PAD_LUMA: u8 = 0;

/// Chroma (Cb and Cr) value OpenH264 writes into the padded region
/// (`memset(.., 0x80, ..)` in the same function).
pub const ENCODER_PAD_CHROMA: u8 = 0x80;

/// Smallest frame side OpenH264 accepts (`WelsPreprocessReset` refuses
/// `width < 16 || height < 16`).
pub const MIN_FRAME_DIM: u32 = 16;

/// Macroblock edge length in luma samples.
const MB: u32 = 16;

/// The coded MB grid `(mb_width, mb_height)` for a frame of `width × height`
/// display pixels: `ceil(dim / 16)` per axis, the grid the SPS carries.
#[inline]
pub fn mb_grid(width: u32, height: u32) -> (u32, u32) {
    (width.div_ceil(MB), height.div_ceil(MB))
}

/// Display size + derived coded geometry of one H.264 stego frame.
///
/// Construct through [`FrameGeometry::new`], which enforces the sizes the
/// encoder can produce exactly (even, at least 16×16).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameGeometry {
    width: u32,
    height: u32,
}

impl FrameGeometry {
    /// Validate a display size.
    ///
    /// # Errors
    /// [`StegoError::InvalidVideo`] if either side is zero, odd, or below
    /// [`MIN_FRAME_DIM`]. Odd sizes are refused rather than rounded: a 4:2:0
    /// crop unit is 2 px, so the stream could not carry the requested size
    /// (OpenH264 would silently drop the odd line).
    pub fn new(width: u32, height: u32) -> Result<Self, StegoError> {
        if width == 0 || height == 0 {
            return Err(StegoError::InvalidVideo(format!(
                "dimensions must be > 0, got {width}x{height}"
            )));
        }
        if !width.is_multiple_of(2) || !height.is_multiple_of(2) {
            return Err(StegoError::InvalidVideo(format!(
                "dimensions must be even (4:2:0), got {width}x{height}"
            )));
        }
        if width < MIN_FRAME_DIM || height < MIN_FRAME_DIM {
            return Err(StegoError::InvalidVideo(format!(
                "dimensions must be at least {MIN_FRAME_DIM}x{MIN_FRAME_DIM}, got {width}x{height}"
            )));
        }
        Ok(Self { width, height })
    }

    /// Display (cropped) width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Display (cropped) height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Coded width in macroblocks (`pic_width_in_mbs`).
    pub fn mb_width(&self) -> u32 {
        mb_grid(self.width, self.height).0
    }

    /// Coded height in macroblocks (`pic_height_in_map_units`, progressive).
    pub fn mb_height(&self) -> u32 {
        mb_grid(self.width, self.height).1
    }

    /// Macroblocks per coded frame.
    pub fn mb_per_frame(&self) -> usize {
        self.mb_width() as usize * self.mb_height() as usize
    }

    /// Coded width in pixels (`mb_width × 16`).
    pub fn coded_width(&self) -> u32 {
        self.mb_width() * MB
    }

    /// Coded height in pixels (`mb_height × 16`).
    pub fn coded_height(&self) -> u32 {
        self.mb_height() * MB
    }

    /// True when the display size already sits on the MB grid (no crop).
    pub fn is_mb_aligned(&self) -> bool {
        self.width.is_multiple_of(MB) && self.height.is_multiple_of(MB)
    }

    /// SPS `frame_crop_right_offset` (4:2:0 crop units of 2 px).
    pub fn sps_crop_right(&self) -> u32 {
        (self.coded_width() - self.width) / 2
    }

    /// SPS `frame_crop_bottom_offset` (4:2:0 crop units of 2 px).
    pub fn sps_crop_bottom(&self) -> u32 {
        (self.coded_height() - self.height) / 2
    }

    /// Bytes of one display-size tight I420 frame (`w·h + 2·(w/2)·(h/2)`).
    pub fn frame_bytes(&self) -> usize {
        let (w, h) = (self.width as usize, self.height as usize);
        w * h + 2 * (w / 2) * (h / 2)
    }

    /// Bytes of one coded-size tight I420 frame.
    pub fn coded_frame_bytes(&self) -> usize {
        let (w, h) = (self.coded_width() as usize, self.coded_height() as usize);
        w * h + 2 * (w / 2) * (h / 2)
    }

    /// Expand one display-size tight I420 `frame` into the coded-size tight
    /// I420 picture OpenH264 actually codes, written to `out` (resized to
    /// [`Self::coded_frame_bytes`]).
    ///
    /// Each plane keeps the display pixels in its top-left corner; every
    /// sample right of the display width or below the display height takes
    /// the encoder's constant fill ([`ENCODER_PAD_LUMA`] /
    /// [`ENCODER_PAD_CHROMA`]). That is the same picture
    /// `CWelsPreProcess::Padding` leaves in the encoder's source buffer, so
    /// costs computed over it price the padded MB row / column from the very
    /// pixels the encoder coded. Pinned against the fork by the
    /// `exact_size_padding_matches_openh264_internal_fill` test
    /// (`core/tests/oh264_exact_size_crop.rs`).
    ///
    /// # Panics
    /// If `frame` is shorter than [`Self::frame_bytes`].
    pub fn pad_frame_into(&self, frame: &[u8], out: &mut Vec<u8>) {
        let (w, h) = (self.width as usize, self.height as usize);
        let (cw, ch) = (self.coded_width() as usize, self.coded_height() as usize);
        let y_len = w * h;
        let c_len = (w / 2) * (h / 2);
        assert!(frame.len() >= y_len + 2 * c_len, "frame shorter than {w}x{h} I420");

        out.clear();
        out.reserve(self.coded_frame_bytes());
        pad_plane(&frame[..y_len], w, h, cw, ch, ENCODER_PAD_LUMA, out);
        pad_plane(&frame[y_len..y_len + c_len], w / 2, h / 2, cw / 2, ch / 2, ENCODER_PAD_CHROMA, out);
        pad_plane(
            &frame[y_len + c_len..y_len + 2 * c_len],
            w / 2,
            h / 2,
            cw / 2,
            ch / 2,
            ENCODER_PAD_CHROMA,
            out,
        );
    }
}

/// Append a `cw × ch` plane holding the `w × h` `src` plane top-left and
/// `fill` everywhere else.
fn pad_plane(src: &[u8], w: usize, h: usize, cw: usize, ch: usize, fill: u8, out: &mut Vec<u8>) {
    for row in src.chunks_exact(w).take(h) {
        out.extend_from_slice(row);
        out.resize(out.len() + (cw - w), fill);
    }
    out.resize(out.len() + (ch - h) * cw, fill);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_is_ceil_and_matches_phone_1080p() {
        let g = FrameGeometry::new(1920, 1080).unwrap();
        assert_eq!((g.mb_width(), g.mb_height()), (120, 68));
        assert_eq!((g.coded_width(), g.coded_height()), (1920, 1088));
        assert_eq!((g.sps_crop_right(), g.sps_crop_bottom()), (0, 4));
        assert!(!g.is_mb_aligned());

        let p = FrameGeometry::new(1080, 1920).unwrap();
        assert_eq!((p.sps_crop_right(), p.sps_crop_bottom()), (4, 0));

        // DJI Mini 2 2.7K: 2720×1530 → 170×96 MBs, crop_bottom 3.
        let d = FrameGeometry::new(2720, 1530).unwrap();
        assert_eq!((d.mb_width(), d.mb_height(), d.sps_crop_bottom()), (170, 96, 3));
    }

    #[test]
    fn aligned_sizes_have_no_crop() {
        for &(w, h) in &[(16, 16), (480, 272), (1920, 1072), (848, 480)] {
            let g = FrameGeometry::new(w, h).unwrap();
            assert!(g.is_mb_aligned(), "{w}x{h}");
            assert_eq!((g.mb_width(), g.mb_height()), (w / 16, h / 16));
            assert_eq!((g.sps_crop_right(), g.sps_crop_bottom()), (0, 0));
            assert_eq!(g.frame_bytes(), g.coded_frame_bytes());
        }
    }

    #[test]
    fn rejects_zero_odd_and_tiny() {
        assert!(FrameGeometry::new(0, 16).is_err());
        assert!(FrameGeometry::new(1921, 1080).is_err());
        assert!(FrameGeometry::new(1920, 1081).is_err());
        assert!(FrameGeometry::new(14, 64).is_err());
        assert!(FrameGeometry::new(64, 14).is_err());
        assert!(FrameGeometry::new(16, 18).is_ok());
    }

    #[test]
    fn frame_bytes_is_tight_i420() {
        let g = FrameGeometry::new(200, 120).unwrap();
        assert_eq!(g.frame_bytes(), 200 * 120 * 3 / 2);
        assert_eq!(g.coded_frame_bytes(), 208 * 128 * 3 / 2);
    }

    #[test]
    fn pad_keeps_display_pixels_and_fills_the_rest() {
        let g = FrameGeometry::new(18, 20).unwrap(); // coded 32×32
        let (w, h) = (18usize, 20usize);
        let mut frame = Vec::new();
        for i in 0..w * h {
            frame.push(1 + (i % 200) as u8); // never 0 → distinguishable from luma fill
        }
        for i in 0..2 * (w / 2) * (h / 2) {
            frame.push(1 + (i % 100) as u8); // never 0x80
        }
        let mut out = Vec::new();
        g.pad_frame_into(&frame, &mut out);
        assert_eq!(out.len(), g.coded_frame_bytes());

        let (cw, ch) = (32usize, 32usize);
        for y in 0..ch {
            for x in 0..cw {
                let got = out[y * cw + x];
                if x < w && y < h {
                    assert_eq!(got, frame[y * w + x], "luma ({x},{y})");
                } else {
                    assert_eq!(got, ENCODER_PAD_LUMA, "luma pad ({x},{y})");
                }
            }
        }
        for plane in 0..2 {
            let src_off = w * h + plane * (w / 2) * (h / 2);
            let dst_off = cw * ch + plane * (cw / 2) * (ch / 2);
            for y in 0..ch / 2 {
                for x in 0..cw / 2 {
                    let got = out[dst_off + y * (cw / 2) + x];
                    if x < w / 2 && y < h / 2 {
                        assert_eq!(got, frame[src_off + y * (w / 2) + x], "chroma{plane} ({x},{y})");
                    } else {
                        assert_eq!(got, ENCODER_PAD_CHROMA, "chroma{plane} pad ({x},{y})");
                    }
                }
            }
        }
    }

    #[test]
    fn pad_of_aligned_frame_is_identity() {
        let g = FrameGeometry::new(32, 16).unwrap();
        let frame: Vec<u8> = (0..g.frame_bytes()).map(|i| (i * 7) as u8).collect();
        let mut out = Vec::new();
        g.pad_frame_into(&frame, &mut out);
        assert_eq!(out, frame);
    }
}
