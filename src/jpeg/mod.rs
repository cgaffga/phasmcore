// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Pure-Rust JPEG coefficient codec (zero external dependencies).
//!
//! Reads and writes baseline and progressive JPEG files, providing direct access
//! to quantized DCT coefficients without any pixel-domain processing. This is
//! the foundation for steganographic embedding, which operates entirely in
//! the DCT domain.
//!
//! Supports:
//! - Baseline sequential DCT (SOF0), 8-bit precision
//! - Progressive DCT (SOF2) — read-only (always writes baseline)
//! - YCbCr, grayscale, and arbitrary component counts
//! - Chroma subsampling: 4:2:0, 4:2:2, 4:4:4
//! - Restart markers (DRI/RST)
//! - Byte-for-byte round-trip for unmodified baseline images
//! - Huffman table rebuild for modified coefficients
//!
//! Does NOT support:
//! - Arithmetic coding (SOF9+) -- rejected at parse time
//! - 12-bit precision -- rejected at parse time

pub mod error;
pub mod zigzag;
pub mod dct;
pub mod bitio;
pub mod tables;
pub mod huffman;
pub mod frame;
pub mod marker;
pub mod scan;
pub mod pixels;

use dct::DctGrid;
use error::{JpegError, Result};
use frame::FrameInfo;
use huffman::encode_value;
use marker::{MarkerSegment, iterate_markers, iterate_markers_all, parse_sos, parse_sos_params, parse_dri};
use scan::ScanComponent;
use tables::{HuffmanSpec, parse_dqt, parse_dht};
use zigzag::NATURAL_TO_ZIGZAG;

/// A decoded JPEG image providing access to quantized DCT coefficients.
///
/// Created by parsing a JPEG byte stream with [`JpegImage::from_bytes`].
/// After modifying DCT coefficients (e.g., for steganographic embedding),
/// call [`JpegImage::to_bytes`] to re-encode. If coefficient modifications
/// introduce symbols not present in the original Huffman tables, call
/// [`JpegImage::rebuild_huffman_tables`] first.
#[derive(Clone)]
pub struct JpegImage {
    /// Frame information (dimensions, components, sampling factors).
    frame: FrameInfo,
    /// DCT coefficient grids, one per component in scan order.
    grids: Vec<DctGrid>,
    /// Quantization tables, indexed by table ID (0–3).
    quant_tables: [Option<dct::QuantTable>; 4],
    /// DC Huffman table specs, indexed by table ID (0–3).
    dc_huff_specs: [Option<HuffmanSpec>; 4],
    /// AC Huffman table specs, indexed by table ID (0–3).
    ac_huff_specs: [Option<HuffmanSpec>; 4],
    /// Scan component selectors (component index + table IDs).
    scan_components: Vec<ScanComponent>,
    /// Restart interval (0 = no restarts).
    restart_interval: u16,
    /// Raw marker segments in original order (for header preservation).
    /// Includes all markers between SOI and SOS (exclusive) except SOI itself.
    raw_segments: Vec<MarkerSegment>,
    /// Raw SOS header data (for exact reconstruction).
    sos_data: Vec<u8>,
}

impl JpegImage {
    /// Parse a JPEG file from bytes.
    ///
    /// Supports both baseline (SOF0) and progressive (SOF2) JPEG.
    /// Progressive images are decoded by accumulating all scans, then the
    /// coefficients are stored exactly as in baseline — `to_bytes()` always
    /// writes baseline output.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        // First pass: quick check if this is progressive by scanning for SOF2
        // We use iterate_markers_all which handles multiple SOS markers.
        let is_progressive = Self::check_progressive(data);

        if is_progressive {
            Self::from_bytes_progressive(data)
        } else {
            Self::from_bytes_baseline(data)
        }
    }

    /// Quick check: does this JPEG contain a SOF2 marker?
    fn check_progressive(data: &[u8]) -> bool {
        // Scan for 0xFF 0xC2 in the header area (before any SOS)
        let mut pos = 2; // skip SOI
        while pos + 1 < data.len() {
            if data[pos] == 0xFF {
                let m = data[pos + 1];
                if m == marker::SOF2 {
                    return true;
                }
                if m == marker::SOS {
                    return false; // Reached scan data without finding SOF2
                }
                if m == 0x00 || m == 0xFF || (m >= 0xD0 && m <= 0xD7) || m == marker::SOI || m == marker::EOI {
                    pos += 2;
                    continue;
                }
                // Skip segment
                if pos + 3 < data.len() {
                    let len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                    pos += 2 + len;
                } else {
                    break;
                }
            } else {
                pos += 1;
            }
        }
        false
    }

    /// Parse a baseline (SOF0) JPEG file.
    fn from_bytes_baseline(data: &[u8]) -> Result<Self> {
        let (entries, scan_start) = iterate_markers(data)?;

        let mut frame_info: Option<FrameInfo> = None;
        let mut quant_tables: [Option<dct::QuantTable>; 4] = [None, None, None, None];
        let mut dc_huff_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
        let mut ac_huff_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
        let mut restart_interval: u16 = 0;
        let mut raw_segments = Vec::new();
        let mut sos_data = Vec::new();
        let mut scan_components = Vec::new();

        for entry in &entries {
            match entry.marker {
                marker::SOI => {}
                marker::EOI => {}
                marker::DQT => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    let tables = parse_dqt(&entry.data)?;
                    for (id, qt) in tables {
                        quant_tables[id as usize] = Some(qt);
                    }
                }
                marker::DHT => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    let specs = parse_dht(&entry.data)?;
                    for spec in specs {
                        let id = spec.id as usize;
                        if spec.class == 0 {
                            dc_huff_specs[id] = Some(spec);
                        } else {
                            ac_huff_specs[id] = Some(spec);
                        }
                    }
                }
                marker::SOF0 => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    frame_info = Some(frame::parse_sof(&entry.data)?);
                }
                marker::DRI => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    restart_interval = parse_dri(&entry.data)?;
                }
                marker::SOS => {
                    sos_data = entry.data.clone();
                    let selectors = parse_sos(&entry.data)?;
                    let fi = frame_info
                        .as_ref()
                        .ok_or(JpegError::InvalidMarkerData("SOS before SOF"))?;

                    for (comp_id, dc_id, ac_id) in selectors {
                        let comp_idx = fi
                            .components
                            .iter()
                            .position(|c| c.id == comp_id)
                            .ok_or(JpegError::UnknownComponentId(comp_id))?;
                        scan_components.push(ScanComponent {
                            comp_idx,
                            dc_table: dc_id as usize,
                            ac_table: ac_id as usize,
                        });
                    }
                }
                _ => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                }
            }
        }

        let fi = frame_info.ok_or(JpegError::InvalidMarkerData("no SOF marker found"))?;

        let (grids, _end_pos) = scan::decode_scan(
            data,
            scan_start,
            &fi,
            &scan_components,
            &dc_huff_specs,
            &ac_huff_specs,
            restart_interval,
        )?;

        Ok(Self {
            frame: fi,
            grids,
            quant_tables,
            dc_huff_specs,
            ac_huff_specs,
            scan_components,
            restart_interval,
            raw_segments,
            sos_data,
        })
    }

    /// Parse a progressive (SOF2) JPEG file.
    ///
    /// Progressive JPEG files have multiple SOS markers, each contributing
    /// partial coefficient data. We accumulate all scans into DctGrids,
    /// then store the result as if it were a baseline image.
    fn from_bytes_progressive(data: &[u8]) -> Result<Self> {
        let (entries, scan_starts) = iterate_markers_all(data)?;

        let mut frame_info: Option<FrameInfo> = None;
        let mut quant_tables: [Option<dct::QuantTable>; 4] = [None, None, None, None];
        let mut dc_huff_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
        let mut ac_huff_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
        let mut restart_interval: u16 = 0;
        let mut raw_segments = Vec::new();

        // Collect all SOS entries with their scan start positions
        struct ScanInfo {
            components: Vec<ScanComponent>,
            params: marker::SosParams,
            scan_start: usize,
            #[allow(dead_code)]
            sos_data: Vec<u8>,
        }
        let mut scans: Vec<ScanInfo> = Vec::new();
        let mut sos_index = 0usize;

        for entry in &entries {
            match entry.marker {
                marker::SOI => {}
                marker::EOI => {}
                marker::DQT => {
                    // Only preserve DQT/DRI in raw_segments (first occurrence)
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    let tables = parse_dqt(&entry.data)?;
                    for (id, qt) in tables {
                        quant_tables[id as usize] = Some(qt);
                    }
                }
                marker::DHT => {
                    // For progressive, DHT markers can appear between scans.
                    // We accumulate all Huffman tables (later tables override earlier ones
                    // with the same ID, which is the correct behavior).
                    // Don't preserve DHT in raw_segments — we'll rebuild them.
                    let specs = parse_dht(&entry.data)?;
                    for spec in specs {
                        let id = spec.id as usize;
                        if spec.class == 0 {
                            dc_huff_specs[id] = Some(spec);
                        } else {
                            ac_huff_specs[id] = Some(spec);
                        }
                    }
                }
                marker::SOF2 => {
                    raw_segments.push(MarkerSegment {
                        // Store as SOF0 for baseline output
                        marker: marker::SOF0,
                        data: entry.data.clone(),
                    });
                    frame_info = Some(frame::parse_sof_ext(&entry.data, true)?);
                }
                marker::SOF0 => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    frame_info = Some(frame::parse_sof(&entry.data)?);
                }
                marker::DRI => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    restart_interval = parse_dri(&entry.data)?;
                }
                marker::SOS => {
                    let selectors = parse_sos(&entry.data)?;
                    let params = parse_sos_params(&entry.data)?;
                    let fi = frame_info
                        .as_ref()
                        .ok_or(JpegError::InvalidMarkerData("SOS before SOF"))?;

                    let mut components = Vec::new();
                    for (comp_id, dc_id, ac_id) in selectors {
                        let comp_idx = fi
                            .components
                            .iter()
                            .position(|c| c.id == comp_id)
                            .ok_or(JpegError::UnknownComponentId(comp_id))?;
                        components.push(ScanComponent {
                            comp_idx,
                            dc_table: dc_id as usize,
                            ac_table: ac_id as usize,
                        });
                    }

                    if sos_index < scan_starts.len() {
                        scans.push(ScanInfo {
                            components,
                            params,
                            scan_start: scan_starts[sos_index],
                            sos_data: entry.data.clone(),
                        });
                        sos_index += 1;
                    }
                }
                _ => {
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                }
            }
        }

        let fi = frame_info.ok_or(JpegError::InvalidMarkerData("no SOF marker found"))?;

        // Allocate DctGrids for all components (initialized to zero)
        let mut grids: Vec<DctGrid> = Vec::with_capacity(fi.components.len());
        for comp_idx in 0..fi.components.len() {
            let bw = fi.blocks_wide(comp_idx);
            let bt = fi.blocks_tall(comp_idx);
            grids.push(DctGrid::new(bw, bt));
        }

        // Snapshot the Huffman specs before processing scans, since progressive
        // JPEG can define new DHT tables between scans.
        // We already accumulated all DHTs above, which works for most files.
        // However, some encoders define DHT tables incrementally before each scan.
        // To handle this correctly, we need to re-parse DHTs in scan order.
        // Let's re-parse by walking entries again, updating specs as we go.
        let mut scan_dc_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
        let mut scan_ac_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
        let mut scan_idx = 0usize;

        for entry in &entries {
            match entry.marker {
                marker::DHT => {
                    let specs = parse_dht(&entry.data)?;
                    for spec in specs {
                        let id = spec.id as usize;
                        if spec.class == 0 {
                            scan_dc_specs[id] = Some(spec);
                        } else {
                            scan_ac_specs[id] = Some(spec);
                        }
                    }
                }
                marker::SOS => {
                    if scan_idx < scans.len() {
                        let scan = &scans[scan_idx];
                        scan::decode_progressive_scan(
                            data,
                            scan.scan_start,
                            &fi,
                            &scan.components,
                            &scan_dc_specs,
                            &scan_ac_specs,
                            restart_interval,
                            &scan.params,
                            &mut grids,
                        )?;
                        scan_idx += 1;
                    }
                }
                _ => {}
            }
        }

        // For baseline output, we need to build a single SOS header.
        // Use all components in component order, with table IDs from the
        // frame components (standard convention: luma=table 0, chroma=table 1).
        let mut final_scan_components = Vec::new();
        let mut final_sos_data = Vec::new();
        final_sos_data.push(fi.components.len() as u8);

        for (comp_idx, comp) in fi.components.iter().enumerate() {
            // Use table ID 0 for luminance (first component), 1 for chrominance
            let table_id = if comp_idx == 0 { 0usize } else { 1usize };
            final_scan_components.push(ScanComponent {
                comp_idx,
                dc_table: table_id,
                ac_table: table_id,
            });
            final_sos_data.push(comp.id);
            final_sos_data.push(((table_id as u8) << 4) | (table_id as u8));
        }
        // Append baseline SOS parameters: Ss=0, Se=63, Ah=0, Al=0
        final_sos_data.push(0);  // Ss
        final_sos_data.push(63); // Se
        final_sos_data.push(0);  // Ah=0, Al=0

        // Build a minimal but complete set of baseline Huffman tables.
        // We set the specs to None first, then rebuild from the coefficient data.
        // This ensures the tables match the actual coefficient values.
        let final_dc_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
        let final_ac_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];

        // Create the image with placeholder Huffman tables, then rebuild them.
        let mut img = Self {
            frame: FrameInfo { is_progressive: false, ..fi },
            grids,
            quant_tables,
            dc_huff_specs: final_dc_specs,
            ac_huff_specs: final_ac_specs,
            scan_components: final_scan_components,
            restart_interval,
            raw_segments,
            sos_data: final_sos_data,
        };

        // Rebuild Huffman tables from the actual coefficient data so they
        // encode correctly as baseline. This also inserts DHT segments into
        // raw_segments.
        img.rebuild_huffman_tables();

        Ok(img)
    }

    /// Encode the (possibly modified) image back to JPEG bytes.
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        let mut out = Vec::new();

        // SOI
        out.push(0xFF);
        out.push(marker::SOI);

        // Write all preserved header segments in original order
        for seg in &self.raw_segments {
            out.push(0xFF);
            out.push(seg.marker);
            let length = (seg.data.len() + 2) as u16;
            out.push((length >> 8) as u8);
            out.push(length as u8);
            out.extend_from_slice(&seg.data);
        }

        // Write SOS header
        out.push(0xFF);
        out.push(marker::SOS);
        let sos_length = (self.sos_data.len() + 2) as u16;
        out.push((sos_length >> 8) as u8);
        out.push(sos_length as u8);
        out.extend_from_slice(&self.sos_data);

        // Re-encode scan data
        let scan_bytes = scan::encode_scan(
            &self.frame,
            &self.scan_components,
            &self.grids,
            &self.dc_huff_specs,
            &self.ac_huff_specs,
            self.restart_interval,
        )?;
        out.extend_from_slice(&scan_bytes);

        // EOI
        out.push(0xFF);
        out.push(marker::EOI);

        Ok(out)
    }

    /// Get a reference to the DCT coefficient grid for a component.
    /// Component index is in scan order (typically 0=Y, 1=Cb, 2=Cr).
    pub fn dct_grid(&self, component: usize) -> &DctGrid {
        &self.grids[component]
    }

    /// Get a mutable reference to the DCT coefficient grid for a component.
    pub fn dct_grid_mut(&mut self, component: usize) -> &mut DctGrid {
        &mut self.grids[component]
    }

    /// Get the frame information.
    pub fn frame_info(&self) -> &FrameInfo {
        &self.frame
    }

    /// Get a quantization table by ID.
    pub fn quant_table(&self, id: usize) -> Option<&dct::QuantTable> {
        self.quant_tables[id].as_ref()
    }

    /// Number of components in the scan.
    pub fn num_components(&self) -> usize {
        self.grids.len()
    }

    /// Rebuild Huffman tables from the current coefficient data.
    ///
    /// Call this after modifying DCT coefficients to ensure the Huffman tables
    /// can encode all symbols present in the modified data. This replaces the
    /// DHT segments in `raw_segments` and updates `dc_huff_specs`/`ac_huff_specs`.
    pub fn rebuild_huffman_tables(&mut self) {
        // Collect symbol frequencies per table.
        let mut dc_freq: [Vec<u32>; 4] = [vec![], vec![], vec![], vec![]];
        let mut ac_freq: [Vec<u32>; 4] = [vec![], vec![], vec![], vec![]];

        for sc in &self.scan_components {
            if dc_freq[sc.dc_table].is_empty() {
                dc_freq[sc.dc_table] = vec![0u32; 256];
            }
            if ac_freq[sc.ac_table].is_empty() {
                ac_freq[sc.ac_table] = vec![0u32; 256];
            }
        }

        // Count symbols by simulating the scan encoding.
        // Must match encode_scan exactly, including restart interval DC pred resets.
        let mut dc_pred = vec![0i16; self.scan_components.len()];
        let mut mcu_count = 0usize;

        for mcu_row in 0..self.frame.mcus_tall as usize {
            for mcu_col in 0..self.frame.mcus_wide as usize {
                // Reset DC predictors at restart boundaries (must match encode_scan)
                if self.restart_interval > 0
                    && mcu_count > 0
                    && mcu_count % (self.restart_interval as usize) == 0
                {
                    for pred in &mut dc_pred {
                        *pred = 0;
                    }
                }

                for (sci, sc) in self.scan_components.iter().enumerate() {
                    let comp = &self.frame.components[sc.comp_idx];
                    for v in 0..comp.v_sampling as usize {
                        for h in 0..comp.h_sampling as usize {
                            let br = mcu_row * comp.v_sampling as usize + v;
                            let bc = mcu_col * comp.h_sampling as usize + h;
                            let block = self.grids[sci].block(br, bc);
                            let mut zz = [0i16; 64];
                            for ni in 0..64 {
                                zz[NATURAL_TO_ZIGZAG[ni]] = block[ni];
                            }

                            // DC symbol
                            let dc_diff = zz[0] - dc_pred[sci];
                            dc_pred[sci] = zz[0];
                            let (_, dc_size) = encode_value(dc_diff);
                            dc_freq[sc.dc_table][dc_size as usize] += 1;

                            // AC symbols
                            let mut k = 1;
                            while k < 64 {
                                let mut run = 0usize;
                                while k + run < 64 && zz[k + run] == 0 {
                                    run += 1;
                                }
                                if k + run >= 64 {
                                    // EOB
                                    ac_freq[sc.ac_table][0x00] += 1;
                                    break;
                                }
                                while run >= 16 {
                                    ac_freq[sc.ac_table][0xF0] += 1;
                                    run -= 16;
                                    k += 16;
                                }
                                k += run;
                                let (_, ac_size) = encode_value(zz[k]);
                                let rs = ((run as u8) << 4) | ac_size;
                                ac_freq[sc.ac_table][rs as usize] += 1;
                                k += 1;
                            }
                        }
                    }
                }

                mcu_count += 1;
            }
        }

        // Build Huffman specs from frequency counts and update state.
        for (id, freq) in dc_freq.iter().enumerate() {
            if freq.is_empty() {
                continue;
            }
            let spec = build_huffman_spec(0, id as u8, freq);
            self.dc_huff_specs[id] = Some(spec);
        }
        for (id, freq) in ac_freq.iter().enumerate() {
            if freq.is_empty() {
                continue;
            }
            let spec = build_huffman_spec(1, id as u8, freq);
            self.ac_huff_specs[id] = Some(spec);
        }

        // Replace DHT segments in raw_segments.
        self.raw_segments.retain(|s| s.marker != marker::DHT);

        // Find the position just before SOF0 to insert DHT segments.
        let sof_pos = self
            .raw_segments
            .iter()
            .position(|s| s.marker == marker::SOF0)
            .unwrap_or(self.raw_segments.len());

        // Build new DHT data: combine all tables into one segment.
        let mut dht_data = Vec::new();
        for id in 0..4 {
            if let Some(spec) = &self.dc_huff_specs[id] {
                dht_data.push((spec.class << 4) | (spec.id & 0x0F));
                dht_data.extend_from_slice(&spec.bits);
                dht_data.extend_from_slice(&spec.huffval);
            }
        }
        for id in 0..4 {
            if let Some(spec) = &self.ac_huff_specs[id] {
                dht_data.push((spec.class << 4) | (spec.id & 0x0F));
                dht_data.extend_from_slice(&spec.bits);
                dht_data.extend_from_slice(&spec.huffval);
            }
        }

        self.raw_segments.insert(
            sof_pos,
            MarkerSegment {
                marker: marker::DHT,
                data: dht_data,
            },
        );
    }

    /// Replace a quantization table by ID and rebuild the DQT marker segments.
    ///
    /// Call this after modifying DCT coefficients to reflect new quantization
    /// (e.g., for recompression simulation). Updates both the internal table
    /// and the raw DQT segments so that `to_bytes()` produces correct output.
    pub fn set_quant_table(&mut self, id: usize, qt: dct::QuantTable) {
        self.quant_tables[id] = Some(qt);
        self.rebuild_dqt_segments();
    }

    /// Rebuild DQT marker segments from internal quantization table state.
    ///
    /// Removes all existing DQT entries from `raw_segments` and inserts fresh
    /// ones before the SOF0 marker (matching the standard JPEG header order).
    fn rebuild_dqt_segments(&mut self) {
        use zigzag::ZIGZAG_TO_NATURAL;

        // Remove old DQT segments.
        self.raw_segments.retain(|s| s.marker != marker::DQT);

        // Build new DQT data: one segment containing all defined tables.
        // DQT stores values in zigzag order. Our internal tables are in
        // natural (row-major) order. For each zigzag index zi, we need
        // the natural index: ni = ZIGZAG_TO_NATURAL[zi].
        let mut dqt_data = Vec::new();
        for id in 0..4u8 {
            if let Some(qt) = &self.quant_tables[id as usize] {
                // precision_and_id: precision=0 (8-bit) for values ≤255
                let precision: u8 = if qt.values.iter().all(|&v| v <= 255) { 0 } else { 1 };
                dqt_data.push((precision << 4) | id);
                for zi in 0..64 {
                    let ni = ZIGZAG_TO_NATURAL[zi];
                    if precision == 0 {
                        dqt_data.push(qt.values[ni] as u8);
                    } else {
                        dqt_data.extend_from_slice(&qt.values[ni].to_be_bytes());
                    }
                }
            }
        }

        // Insert before SOF0 (same position strategy as DHT rebuild).
        let sof_pos = self
            .raw_segments
            .iter()
            .position(|s| s.marker == marker::SOF0)
            .unwrap_or(self.raw_segments.len());

        self.raw_segments.insert(
            sof_pos,
            MarkerSegment {
                marker: marker::DQT,
                data: dqt_data,
            },
        );
    }
}

/// Build an optimal Huffman spec from symbol frequency counts.
///
/// Implements JPEG Annex K (Figures K.1–K.4) with the libjpeg pseudo-symbol
/// technique: a dummy symbol 256 with frequency 1 is added before tree
/// construction. This guarantees:
/// - No real symbol gets the all-ones codeword.
/// - The Kraft inequality is strictly satisfied after code-length limiting.
/// - Output tables are fully compatible with libjpeg/libjpeg-turbo.
fn build_huffman_spec(class: u8, id: u8, freq: &[u32]) -> HuffmanSpec {
    // Collect symbols with nonzero frequency (u16 to accommodate pseudo-symbol 256).
    let mut symbols: Vec<(u16, u32)> = freq
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 0)
        .map(|(sym, &f)| (sym as u16, f))
        .collect();

    if symbols.is_empty() {
        // Need at least one symbol. Use symbol 0 (EOB for AC, size-0 for DC).
        symbols.push((0, 1));
    }

    // If only one real symbol, we still need a valid Huffman code (1-bit code).
    if symbols.len() == 1 {
        let sym = symbols[0].0 as u8;
        return HuffmanSpec {
            class,
            id,
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            huffval: vec![sym],
        };
    }

    // Add pseudo-symbol 256 with frequency 1 (libjpeg technique).
    // This symbol will get the longest code, preventing any real symbol from
    // receiving the all-ones codeword and providing a Kraft inequality safety
    // margin after Annex K.3 code-length limiting.
    symbols.push((256, 1));

    let n = symbols.len(); // includes pseudo-symbol

    // Sort ascending by (frequency, symbol) for the tree-building merge.
    // Higher symbol number breaks ties → pseudo-symbol 256 sorts last among
    // freq-1 symbols, ensuring it gets the longest code.
    symbols.sort_by_key(|&(sym, f)| (f, sym));

    // Build Huffman tree using two-queue merge (standard algorithm).
    let total_nodes = 2 * n - 1;
    let mut parent = vec![0usize; total_nodes];
    let mut next_internal = n;

    let mut q1: std::collections::VecDeque<(u64, usize)> = symbols
        .iter()
        .enumerate()
        .map(|(idx, &(_, f))| (f as u64, idx))
        .collect();
    let mut q2: std::collections::VecDeque<(u64, usize)> = std::collections::VecDeque::new();

    let pick_min = |q1: &mut std::collections::VecDeque<(u64, usize)>,
                    q2: &mut std::collections::VecDeque<(u64, usize)>|
     -> (u64, usize) {
        match (q1.front(), q2.front()) {
            (Some(&a), Some(&b)) => {
                if a.0 <= b.0 {
                    q1.pop_front().unwrap()
                } else {
                    q2.pop_front().unwrap()
                }
            }
            (Some(_), None) => q1.pop_front().unwrap(),
            (None, Some(_)) => q2.pop_front().unwrap(),
            (None, None) => unreachable!(),
        }
    };

    for _ in 0..(n - 1) {
        let (f1, idx1) = pick_min(&mut q1, &mut q2);
        let (f2, idx2) = pick_min(&mut q1, &mut q2);
        parent[idx1] = next_internal;
        parent[idx2] = next_internal;
        q2.push_back((f1 + f2, next_internal));
        next_internal += 1;
    }

    // Compute code lengths by walking from each leaf to the root.
    let root = total_nodes - 1;
    let mut code_lengths = vec![0u8; n];
    for i in 0..n {
        let mut depth = 0u8;
        let mut node = i;
        while node != root {
            node = parent[node];
            depth += 1;
        }
        code_lengths[i] = depth;
    }

    // Limit code lengths to 16 bits (JPEG Annex K.3 — Adjust_BITS procedure).
    let max_len = code_lengths.iter().copied().max().unwrap_or(0) as usize;

    let mut bits_count = vec![0u32; max_len + 1];
    for &len in &code_lengths {
        bits_count[len as usize] += 1;
    }

    if max_len > 16 {
        let mut i = max_len;
        while i > 16 {
            while bits_count[i] > 0 {
                // Find a donor level j (j <= i-2) that has codes to split.
                let mut j = i - 2;
                while j > 0 && bits_count[j] == 0 {
                    j -= 1;
                }
                debug_assert!(j > 0, "Annex K.3: no donor found (pseudo-symbol should prevent this)");
                if j == 0 {
                    // Safety fallback (should never happen with pseudo-symbol).
                    bits_count[16] += bits_count[i];
                    bits_count[i] = 0;
                    break;
                }
                bits_count[i] -= 2;
                bits_count[i - 1] += 1;
                bits_count[j + 1] += 2;
                bits_count[j] -= 1;
            }
            i -= 1;
        }

        // Reassign code_lengths from the adjusted bits_count[].
        // Longest codes go to least-frequent symbols (lowest indices).
        let mut pos = 0;
        for len in (1..=16u8).rev() {
            let count = bits_count[len as usize] as usize;
            for _ in 0..count {
                code_lengths[pos] = len;
                pos += 1;
            }
        }
    }

    // Build bits[] and huffval[] arrays, excluding pseudo-symbol 256.
    // Sort by (code_length, symbol_value) for canonical Huffman ordering.
    let mut sym_len: Vec<(u16, u8)> = symbols
        .iter()
        .zip(code_lengths.iter())
        .map(|(&(sym, _), &len)| (sym, len))
        .collect();
    sym_len.sort_by_key(|&(sym, len)| (len, sym));

    let mut bits = [0u8; 16];
    let mut huffval = Vec::with_capacity(n);
    for &(sym, len) in &sym_len {
        // Skip pseudo-symbol 256 — it served its purpose in tree construction.
        if sym == 256 {
            continue;
        }
        if len > 0 && len <= 16 {
            bits[(len - 1) as usize] += 1;
            huffval.push(sym as u8);
        }
    }

    HuffmanSpec {
        class,
        id,
        bits,
        huffval,
    }
}
