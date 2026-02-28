// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! JPEG frame header (SOF0) parsing.
//!
//! Extracts image dimensions, component information, and sampling factors
//! from the Start of Frame marker segment.

use super::error::{JpegError, Result};

/// Information about one image component from SOF.
#[derive(Debug, Clone)]
pub struct Component {
    /// Component ID (typically 1=Y, 2=Cb, 3=Cr).
    pub id: u8,
    /// Horizontal sampling factor (1–4).
    pub h_sampling: u8,
    /// Vertical sampling factor (1–4).
    pub v_sampling: u8,
    /// Quantization table ID (0–3).
    pub quant_table_id: u8,
}

/// Frame information parsed from SOF0/SOF2 marker.
#[derive(Debug, Clone)]
pub struct FrameInfo {
    /// Sample precision in bits (must be 8).
    pub precision: u8,
    /// Image height in pixels.
    pub height: u16,
    /// Image width in pixels.
    pub width: u16,
    /// Components in the frame.
    pub components: Vec<Component>,
    /// Maximum horizontal sampling factor across all components.
    pub max_h_sampling: u8,
    /// Maximum vertical sampling factor across all components.
    pub max_v_sampling: u8,
    /// MCU width in blocks (= max_h_sampling * 8 pixels).
    pub mcu_width: u16,
    /// MCU height in blocks (= max_v_sampling * 8 pixels).
    pub mcu_height: u16,
    /// Number of MCUs horizontally.
    pub mcus_wide: u16,
    /// Number of MCUs vertically.
    pub mcus_tall: u16,
    /// Whether this is a progressive JPEG (SOF2). False for baseline (SOF0).
    pub is_progressive: bool,
}

impl FrameInfo {
    /// Number of 8×8 blocks wide for a given component.
    pub fn blocks_wide(&self, comp_idx: usize) -> usize {
        let comp = &self.components[comp_idx];
        (self.mcus_wide as usize) * (comp.h_sampling as usize)
    }

    /// Number of 8×8 blocks tall for a given component.
    pub fn blocks_tall(&self, comp_idx: usize) -> usize {
        let comp = &self.components[comp_idx];
        (self.mcus_tall as usize) * (comp.v_sampling as usize)
    }
}

/// Parse a SOF0/SOF2 marker segment body (after the 2-byte length).
/// `progressive` should be true for SOF2 markers.
pub fn parse_sof(data: &[u8]) -> Result<FrameInfo> {
    parse_sof_ext(data, false)
}

/// Parse a SOF marker segment body with explicit progressive flag.
pub fn parse_sof_ext(data: &[u8], progressive: bool) -> Result<FrameInfo> {
    if data.len() < 6 {
        return Err(JpegError::UnexpectedEof);
    }

    let precision = data[0];
    if precision != 8 {
        return Err(JpegError::UnsupportedPrecision(precision));
    }

    let height = u16::from_be_bytes([data[1], data[2]]);
    let width = u16::from_be_bytes([data[3], data[4]]);
    let num_components = data[5] as usize;

    if width == 0 || height == 0 {
        return Err(JpegError::InvalidDimensions);
    }
    if data.len() < 6 + num_components * 3 {
        return Err(JpegError::UnexpectedEof);
    }

    let mut components = Vec::with_capacity(num_components);
    let mut max_h = 0u8;
    let mut max_v = 0u8;

    for i in 0..num_components {
        let offset = 6 + i * 3;
        let id = data[offset];
        let sampling = data[offset + 1];
        let h_sampling = sampling >> 4;
        let v_sampling = sampling & 0x0F;
        let quant_table_id = data[offset + 2];

        if h_sampling == 0 || v_sampling == 0 || h_sampling > 4 || v_sampling > 4 {
            return Err(JpegError::InvalidDimensions);
        }
        if quant_table_id > 3 {
            return Err(JpegError::InvalidQuantTableId(quant_table_id));
        }

        max_h = max_h.max(h_sampling);
        max_v = max_v.max(v_sampling);

        components.push(Component {
            id,
            h_sampling,
            v_sampling,
            quant_table_id,
        });
    }

    let mcu_width = (max_h as u16) * 8;
    let mcu_height = (max_v as u16) * 8;
    let mcus_wide = (width + mcu_width - 1) / mcu_width;
    let mcus_tall = (height + mcu_height - 1) / mcu_height;

    Ok(FrameInfo {
        precision,
        height,
        width,
        components,
        max_h_sampling: max_h,
        max_v_sampling: max_v,
        mcu_width,
        mcu_height,
        mcus_wide,
        mcus_tall,
        is_progressive: progressive,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_ycbcr_420() {
        // SOF0 body: precision=8, height=480, width=640, 3 components
        // Y: id=1, h=2, v=2, qt=0
        // Cb: id=2, h=1, v=1, qt=1
        // Cr: id=3, h=1, v=1, qt=1
        let data = [
            8, 1, 0xE0, 2, 0x80, 3, // precision, height=480, width=640, 3 comps
            1, 0x22, 0, // Y: 2x2, qt=0
            2, 0x11, 1, // Cb: 1x1, qt=1
            3, 0x11, 1, // Cr: 1x1, qt=1
        ];

        let fi = parse_sof(&data).unwrap();
        assert_eq!(fi.precision, 8);
        assert_eq!(fi.height, 480);
        assert_eq!(fi.width, 640);
        assert_eq!(fi.components.len(), 3);
        assert_eq!(fi.max_h_sampling, 2);
        assert_eq!(fi.max_v_sampling, 2);
        assert_eq!(fi.mcu_width, 16);
        assert_eq!(fi.mcu_height, 16);
        assert_eq!(fi.mcus_wide, 40);  // 640/16
        assert_eq!(fi.mcus_tall, 30);  // 480/16

        // Blocks for Y: 40*2=80 wide, 30*2=60 tall
        assert_eq!(fi.blocks_wide(0), 80);
        assert_eq!(fi.blocks_tall(0), 60);
        // Blocks for Cb: 40*1=40 wide, 30*1=30 tall
        assert_eq!(fi.blocks_wide(1), 40);
        assert_eq!(fi.blocks_tall(1), 30);
    }

    #[test]
    fn parse_grayscale() {
        let data = [
            8, 0, 64, 0, 64, 1, // 64x64, 1 component
            1, 0x11, 0, // Y: 1x1, qt=0
        ];
        let fi = parse_sof(&data).unwrap();
        assert_eq!(fi.components.len(), 1);
        assert_eq!(fi.mcus_wide, 8);  // 64/8
        assert_eq!(fi.mcus_tall, 8);
    }

    #[test]
    fn parse_non_mcu_aligned() {
        // 10x10 image with 1x1 sampling → 2x2 MCUs (ceil)
        let data = [
            8, 0, 10, 0, 10, 1,
            1, 0x11, 0,
        ];
        let fi = parse_sof(&data).unwrap();
        assert_eq!(fi.mcus_wide, 2); // ceil(10/8)
        assert_eq!(fi.mcus_tall, 2);
    }

    #[test]
    fn reject_12bit() {
        let data = [12, 0, 8, 0, 8, 1, 1, 0x11, 0];
        assert!(matches!(
            parse_sof(&data),
            Err(JpegError::UnsupportedPrecision(12))
        ));
    }
}
