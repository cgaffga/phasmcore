pub mod error;
pub mod zigzag;
pub mod dct;
pub mod bitio;
pub mod tables;
pub mod huffman;
pub mod frame;
pub mod marker;
pub mod scan;

use dct::DctGrid;
use error::{JpegError, Result};
use frame::FrameInfo;
use marker::{MarkerSegment, iterate_markers, parse_sos, parse_dri};
use scan::ScanComponent;
use tables::{HuffmanSpec, parse_dqt, parse_dht};

/// A decoded JPEG image providing access to quantized DCT coefficients.
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
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
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
                    // Store raw for preservation
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                    // Parse for decoding
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
                    // Preserve all other markers (APPn, COM, etc.) verbatim
                    raw_segments.push(MarkerSegment {
                        marker: entry.marker,
                        data: entry.data.clone(),
                    });
                }
            }
        }

        let fi = frame_info.ok_or(JpegError::InvalidMarkerData("no SOF marker found"))?;

        // Decode the scan
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
}
