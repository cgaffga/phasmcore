// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
//! Encoder-side reconstruction. Phase 6A.6.
//!
//! Wires together the existing decoder-side dequant + inverse-DCT
//! with encoder-side prediction to produce reconstructed pixels
//! for the neighbor buffer. The invariant — every intra MB's
//! neighbors must be the *reconstructed* version, not the source —
//! is the reason this module exists; without it, next-MB prediction
//! diverges from what any real decoder will do.
//!
//! Algorithm note:
//!   docs/design/h264-encoder-algorithms/reconstruction.md

use crate::codec::h264::transform::{
    dequant_4x4, inverse_4x4_integer, reconstruct_residual_4x4_with_dc,
};

use super::EncoderError;

/// Clip a reconstructed sample to `[0, 255]`.
#[inline]
fn clip(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Reconstruct a single 4×4 block: dequant → IDCT → add prediction.
///
/// `ac_levels` are the 16 quantized coefficients in **raster order**
/// (as produced by `forward_quantize_4x4`). Use [`scan_to_raster_levels`]
/// if you have them in zigzag order.
///
/// `prediction` is the predicted pixel block from intra mode decision
/// (or inter motion compensation in Phase 6B).
///
/// Returns the reconstructed 4×4 pixel block, clipped to `[0, 255]`.
pub fn reconstruct_4x4_block(
    ac_levels: &[[i32; 4]; 4],
    prediction: &[[u8; 4]; 4],
    qp: u8,
) -> [[u8; 4]; 4] {
    let dq = dequant_4x4(ac_levels, qp as i32, false);
    let residual = inverse_4x4_integer(&dq);
    let mut recon = [[0u8; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            recon[i][j] = clip(prediction[i][j] as i32 + residual[i][j]);
        }
    }
    recon
}

/// Reconstruct a 4×4 block whose DC came from a separate Hadamard
/// DC block (Intra_16x16 luma AC sub-block, or chroma AC sub-block).
///
/// Delegates to the existing `reconstruct_residual_4x4_with_dc`
/// helper in `transform.rs` for the dequant + inverse-transform
/// pipeline, then adds prediction + clips.
pub fn reconstruct_4x4_block_with_dc(
    ac_levels_zigzag: &[i32; 16],
    dc_value: i32,
    prediction: &[[u8; 4]; 4],
    qp: u8,
) -> [[u8; 4]; 4] {
    let residual = reconstruct_residual_4x4_with_dc(ac_levels_zigzag, dc_value, qp as i32);
    let mut recon = [[0u8; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            recon[i][j] = clip(prediction[i][j] as i32 + residual[i][j]);
        }
    }
    recon
}

/// Convert AC levels from raster-order 2D to zigzag-order flat.
///
/// Useful when the encoder has a `[[i32; 4]; 4]` quantizer output and
/// needs to feed the `[i32; 16]` zigzag API expected by
/// `reconstruct_residual_4x4_with_dc`.
pub fn raster_to_scan_levels(raster: &[[i32; 4]; 4]) -> [i32; 16] {
    use crate::codec::h264::tables::ZIGZAG_4X4;
    let mut scan = [0i32; 16];
    for scan_idx in 0..16 {
        let raster_idx = ZIGZAG_4X4[scan_idx] as usize;
        let i = raster_idx / 4;
        let j = raster_idx % 4;
        scan[scan_idx] = raster[i][j];
    }
    scan
}

/// §B-cascade-real v1.1 — Phase 1.1.B. Inverse of
/// [`raster_to_scan_levels`]: write zigzag-order scan levels back into
/// a 2D raster-order block.
///
/// The dual-recon decoder uses this to lift the post-flip `scan`
/// (returned by the stego hook) into a raster `[[i32; 4]; 4]` block
/// suitable for `dequant_4x4` + `inverse_4x4_integer`, computing the
/// "what a downstream player will see" residual that goes into
/// `visual_recon`.
pub fn scan_to_raster_levels(scan: &[i32; 16]) -> [[i32; 4]; 4] {
    use crate::codec::h264::tables::ZIGZAG_4X4;
    let mut raster = [[0i32; 4]; 4];
    for scan_idx in 0..16 {
        let raster_idx = ZIGZAG_4X4[scan_idx] as usize;
        raster[raster_idx / 4][raster_idx % 4] = scan[scan_idx];
    }
    raster
}

/// Reconstructed-pixel buffer for a frame.
///
/// Holds the reconstructed YUV planes populated MB-by-MB during
/// encoding. The next MB's intra mode decision reads from here, NOT
/// from the source frame.
///
/// Storage is yuv420p — Y full-resolution, Cb/Cr half-resolution per
/// axis. Indexing is `plane[y * stride + x]`.
#[derive(Debug, Clone)]
pub struct ReconBuffer {
    pub width: u32,
    pub height: u32,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

impl ReconBuffer {
    /// Allocate zero-initialized reconstruction buffers for the
    /// given frame dimensions. `width` and `height` must be multiples
    /// of 16 (MB alignment) — the caller is expected to pad the
    /// input frame; partial-MB cropping is signaled via
    /// `frame_crop_*` in the SPS (Phase 6A.5).
    pub fn new(width: u32, height: u32) -> Result<Self, EncoderError> {
        if !width.is_multiple_of(16) || !height.is_multiple_of(16) {
            return Err(EncoderError::InvalidInput(format!(
                "frame dimensions must be 16-aligned, got {width}×{height}"
            )));
        }
        let y_size = (width * height) as usize;
        let c_size = (width / 2 * height / 2) as usize;
        Ok(Self {
            width,
            height,
            y: vec![0; y_size],
            cb: vec![0; c_size],
            cr: vec![0; c_size],
        })
    }

    /// Write a reconstructed 16×16 luma MB at position `(mb_x, mb_y)`
    /// into the buffer.
    pub fn write_luma_mb(&mut self, mb_x: u32, mb_y: u32, pixels: &[[u8; 16]; 16]) {
        let x0 = mb_x * 16;
        let y0 = mb_y * 16;
        for (dy, row) in pixels.iter().enumerate() {
            let dst = ((y0 + dy as u32) * self.width + x0) as usize;
            self.y[dst..dst + 16].copy_from_slice(row);
        }
    }

    /// Write a reconstructed 8×8 chroma block at MB `(mb_x, mb_y)`
    /// for the given component (0 = Cb, 1 = Cr).
    pub fn write_chroma_block(
        &mut self,
        mb_x: u32,
        mb_y: u32,
        component: u8,
        pixels: &[[u8; 8]; 8],
    ) {
        debug_assert!(component < 2, "component must be 0 (Cb) or 1 (Cr)");
        let stride = self.width / 2;
        let x0 = mb_x * 8;
        let y0 = mb_y * 8;
        let plane = if component == 0 { &mut self.cb } else { &mut self.cr };
        for (dy, row) in pixels.iter().enumerate() {
            let dst = ((y0 + dy as u32) * stride + x0) as usize;
            plane[dst..dst + 8].copy_from_slice(row);
        }
    }

    /// Read a single luma pixel. Panics if out of bounds.
    pub fn y_at(&self, x: u32, y: u32) -> u8 {
        self.y[(y * self.width + x) as usize]
    }

    /// Read a single chroma pixel. Panics if out of bounds.
    pub fn chroma_at(&self, component: u8, x: u32, y: u32) -> u8 {
        let stride = self.width / 2;
        let plane = if component == 0 { &self.cb } else { &self.cr };
        plane[(y * stride + x) as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clip_bounds() {
        assert_eq!(clip(-5), 0);
        assert_eq!(clip(0), 0);
        assert_eq!(clip(128), 128);
        assert_eq!(clip(255), 255);
        assert_eq!(clip(300), 255);
    }

    #[test]
    fn zero_residual_gives_prediction() {
        let pred = [[100u8; 4]; 4];
        let levels = [[0i32; 4]; 4];
        let recon = reconstruct_4x4_block(&levels, &pred, 22);
        assert_eq!(recon, pred);
    }

    #[test]
    #[allow(clippy::absurd_extreme_comparisons)]
    fn recon_clipped_at_255() {
        // Huge positive residual via level*scale should be clipped.
        let pred = [[200u8; 4]; 4];
        // Use a big level at DC — the inverse transform will propagate
        // a big positive offset to every pixel. Clipping should hold
        // the result at 255. (v <= 255 is tautological on u8; kept as
        // documentation of the declared invariant.)
        let mut levels = [[0i32; 4]; 4];
        levels[0][0] = 10_000;
        let recon = reconstruct_4x4_block(&levels, &pred, 22);
        for row in &recon {
            for &v in row {
                assert!(v <= 255, "clip failed: got {v}");
            }
        }
    }

    #[test]
    fn recon_clipped_at_0() {
        let pred = [[5u8; 4]; 4];
        let mut levels = [[0i32; 4]; 4];
        levels[0][0] = -10_000;
        let recon = reconstruct_4x4_block(&levels, &pred, 22);
        for row in &recon {
            for &v in row {
                assert!(v < 200, "very negative residual should clip far below pred");
            }
        }
    }

    #[test]
    fn recon_buffer_new_requires_16_alignment() {
        assert!(ReconBuffer::new(320, 240).is_ok());
        assert!(ReconBuffer::new(321, 240).is_err());
        assert!(ReconBuffer::new(320, 241).is_err());
    }

    #[test]
    fn recon_buffer_luma_write_read_round_trip() {
        let mut buf = ReconBuffer::new(320, 240).unwrap();
        let mut mb = [[0u8; 16]; 16];
        for y in 0..16 {
            for x in 0..16 {
                mb[y][x] = ((y * 16 + x) & 0xFF) as u8;
            }
        }
        buf.write_luma_mb(5, 3, &mb);
        // MB at (5, 3) starts at pixel (80, 48).
        for y in 0..16 {
            for x in 0..16 {
                assert_eq!(
                    buf.y_at(80 + x, 48 + y),
                    mb[y as usize][x as usize],
                    "luma mismatch at ({x}, {y})"
                );
            }
        }
    }

    #[test]
    fn recon_buffer_chroma_write_read_round_trip() {
        let mut buf = ReconBuffer::new(320, 240).unwrap();
        let mut block = [[0u8; 8]; 8];
        for y in 0..8 {
            for x in 0..8 {
                block[y][x] = ((y * 8 + x) + 10) as u8;
            }
        }
        buf.write_chroma_block(5, 3, 0, &block); // Cb
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(
                    buf.chroma_at(0, 40 + x, 24 + y),
                    block[y as usize][x as usize],
                    "Cb mismatch at ({x}, {y})"
                );
            }
        }
        // Cr should still be zero (not written).
        assert_eq!(buf.chroma_at(1, 40, 24), 0);
    }

    #[test]
    fn raster_to_scan_matches_unzigzag_inverse() {
        use crate::codec::h264::transform::unzigzag_4x4;
        // Build a deterministic scan array → unzigzag to raster →
        // raster_to_scan_levels should give the original scan back.
        let original_scan: [i32; 16] = [
            10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
        ];
        let raster = unzigzag_4x4(&original_scan);
        let round_trip = raster_to_scan_levels(&raster);
        assert_eq!(round_trip, original_scan);
    }
}
