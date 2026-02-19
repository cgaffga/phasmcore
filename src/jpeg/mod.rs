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
use huffman::encode_value;
use marker::{MarkerSegment, iterate_markers, parse_sos, parse_dri};
use scan::ScanComponent;
use tables::{HuffmanSpec, parse_dqt, parse_dht};
use zigzag::NATURAL_TO_ZIGZAG;

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
}

/// Build an optimal Huffman spec from symbol frequency counts.
///
/// Uses a simplified version of the JPEG Annex K algorithm:
/// sorts symbols by frequency, assigns code lengths, limits to 16 bits.
fn build_huffman_spec(class: u8, id: u8, freq: &[u32]) -> HuffmanSpec {
    // Collect symbols with nonzero frequency.
    let mut symbols: Vec<(u8, u32)> = freq
        .iter()
        .enumerate()
        .filter(|&(_, &f)| f > 0)
        .map(|(sym, &f)| (sym as u8, f))
        .collect();

    if symbols.is_empty() {
        // Need at least one symbol. Use symbol 0 (EOB for AC, size-0 for DC).
        symbols.push((0, 1));
    }

    // If only one symbol, we still need a valid Huffman code (1-bit code).
    if symbols.len() == 1 {
        let sym = symbols[0].0;
        return HuffmanSpec {
            class,
            id,
            bits: [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            huffval: vec![sym],
        };
    }

    let n = symbols.len();

    // Sort by frequency descending (most frequent → shortest code).
    symbols.sort_by(|a, b| b.1.cmp(&a.1));

    // Assign code lengths using a simple heuristic: use the "package-merge"
    // idea simplified. For correctness, we use a standard approach:
    // build a Huffman tree via the standard algorithm, then extract lengths.

    // Build Huffman tree using a priority queue (min-heap via sorted vec).
    let mut nodes: Vec<(u64, Option<usize>)> = symbols
        .iter()
        .enumerate()
        .map(|(i, &(_, f))| (f as u64, Some(i)))
        .collect();

    // Also need parent tracking for code length computation.
    let total_nodes = 2 * n - 1;
    let mut parent = vec![0usize; total_nodes];
    let mut next_internal = n;

    // Sort ascending by frequency for the merge process.
    nodes.sort_by_key(|&(f, _)| f);

    // Simple two-queue merge (Huffman algorithm).
    let mut q1: std::collections::VecDeque<(u64, usize)> = nodes
        .iter()
        .enumerate()
        .map(|(idx, &(f, _))| (f, idx))
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
    // Map from sorted index to original symbol.
    let sorted_syms: Vec<u8> = {
        let mut s: Vec<(u64, usize)> = symbols
            .iter()
            .enumerate()
            .map(|(i, &(_, f))| (f as u64, i))
            .collect();
        s.sort_by_key(|&(f, _)| f);
        s.iter().map(|&(_, orig)| symbols[orig].0).collect()
    };

    for i in 0..n {
        let mut depth = 0u8;
        let mut node = i;
        while node != root {
            node = parent[node];
            depth += 1;
        }
        code_lengths[i] = depth;
    }

    // Limit code lengths to 16 bits (JPEG requirement).
    // Use the algorithm from JPEG spec Annex K.3.
    loop {
        let max_len = code_lengths.iter().copied().max().unwrap_or(0);
        if max_len <= 16 {
            break;
        }
        // Find the deepest leaf and reduce its length by promoting it.
        for len in (17..=max_len).rev() {
            while code_lengths.iter().any(|&l| l == len) {
                // Find a node at `len` and one at `len-2` to swap.
                let deep_idx = code_lengths.iter().position(|&l| l == len).unwrap();
                code_lengths[deep_idx] = len - 1;
                // Find another node at the same depth and reduce.
                if let Some(sibling) = code_lengths.iter().position(|&l| l == len) {
                    code_lengths[sibling] = len - 1;
                }
                // Add a new node at len-1 by splitting one from len-2.
                if let Some(donor) = code_lengths.iter().position(|&l| l == len - 2) {
                    code_lengths[donor] = len - 1;
                }
            }
        }
    }

    // Build bits[] and huffval[] arrays.
    // Sort symbols by (code_length, symbol_value) for canonical ordering.
    let mut sym_len: Vec<(u8, u8)> = sorted_syms
        .iter()
        .zip(code_lengths.iter())
        .map(|(&sym, &len)| (sym, len))
        .collect();
    sym_len.sort_by_key(|&(sym, len)| (len, sym));

    let mut bits = [0u8; 16];
    let mut huffval = Vec::with_capacity(n);
    for &(sym, len) in &sym_len {
        if len > 0 && len <= 16 {
            bits[(len - 1) as usize] += 1;
            huffval.push(sym);
        }
    }

    HuffmanSpec {
        class,
        id,
        bits,
        huffval,
    }
}
