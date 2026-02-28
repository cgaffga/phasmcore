// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! JPEG scan data encoding and decoding.
//!
//! Decodes entropy-coded scan data into [`DctGrid`]s (one per component)
//! and re-encodes modified grids back to entropy-coded bytes. Handles
//! interleaved MCU ordering, restart markers, and DC prediction.

use super::bitio::{BitReader, BitWriter};
use super::dct::DctGrid;
use super::error::{JpegError, Result};
use super::frame::FrameInfo;
use super::huffman::{
    encode_value, extend_sign, HuffmanDecodeTable, HuffmanEncodeTable,
};
use super::marker::SosParams;
use super::tables::HuffmanSpec;
use super::zigzag::{NATURAL_TO_ZIGZAG, ZIGZAG_TO_NATURAL};

/// Component selector for one scan component.
#[derive(Clone)]
pub struct ScanComponent {
    /// Index into FrameInfo.components.
    pub comp_idx: usize,
    /// DC Huffman table index.
    pub dc_table: usize,
    /// AC Huffman table index.
    pub ac_table: usize,
}

/// Decode the entropy-coded scan data into DctGrids.
///
/// - `data`: full JPEG file bytes
/// - `scan_start`: byte offset of the first entropy-coded byte (right after SOS header)
/// - `frame`: parsed frame info
/// - `scan_components`: component selectors from SOS
/// - `dc_specs`/`ac_specs`: Huffman table specs, indexed by table ID
/// - `restart_interval`: from DRI marker, 0 = no restarts
///
/// Returns (grids, end_position). `end_position` is the byte after the last scan byte.
pub fn decode_scan(
    data: &[u8],
    scan_start: usize,
    frame: &FrameInfo,
    scan_components: &[ScanComponent],
    dc_specs: &[Option<HuffmanSpec>; 4],
    ac_specs: &[Option<HuffmanSpec>; 4],
    restart_interval: u16,
) -> Result<(Vec<DctGrid>, usize)> {
    // Build Huffman decode tables
    let mut dc_tables: [Option<HuffmanDecodeTable>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<HuffmanDecodeTable>; 4] = [None, None, None, None];

    for sc in scan_components {
        if dc_tables[sc.dc_table].is_none() {
            let spec = dc_specs[sc.dc_table]
                .as_ref()
                .ok_or(JpegError::InvalidHuffmanTableId(sc.dc_table as u8))?;
            dc_tables[sc.dc_table] = Some(HuffmanDecodeTable::build(&spec.bits, &spec.huffval)?);
        }
        if ac_tables[sc.ac_table].is_none() {
            let spec = ac_specs[sc.ac_table]
                .as_ref()
                .ok_or(JpegError::InvalidHuffmanTableId(sc.ac_table as u8))?;
            ac_tables[sc.ac_table] = Some(HuffmanDecodeTable::build(&spec.bits, &spec.huffval)?);
        }
    }

    // Allocate DctGrids
    let mut grids: Vec<DctGrid> = Vec::with_capacity(scan_components.len());
    for sc in scan_components {
        let bw = frame.blocks_wide(sc.comp_idx);
        let bt = frame.blocks_tall(sc.comp_idx);
        grids.push(DctGrid::new(bw, bt));
    }

    // Initialize DC predictors (i32 to prevent overflow from accumulated diffs)
    let mut dc_pred = vec![0i32; scan_components.len()];

    let mut reader = BitReader::new(data, scan_start);
    let mut mcu_count = 0usize;

    for mcu_row in 0..frame.mcus_tall as usize {
        for mcu_col in 0..frame.mcus_wide as usize {
            // Check for restart
            if restart_interval > 0 && mcu_count > 0 && mcu_count % (restart_interval as usize) == 0 {
                reader.byte_align();
                // Look for RST marker — accept any RST without strict sequence
                // validation, matching libjpeg/libjpeg-turbo behavior.
                let _rst = reader.check_restart_marker()?;
                // Reset DC predictors
                for pred in &mut dc_pred {
                    *pred = 0;
                }
            }

            // Decode blocks for each component in this MCU
            for (sci, sc) in scan_components.iter().enumerate() {
                let comp = &frame.components[sc.comp_idx];
                let dc_tab = dc_tables[sc.dc_table].as_ref().unwrap();
                let ac_tab = ac_tables[sc.ac_table].as_ref().unwrap();

                for v in 0..comp.v_sampling as usize {
                    for h in 0..comp.h_sampling as usize {
                        let block_row = mcu_row * (comp.v_sampling as usize) + v;
                        let block_col = mcu_col * (comp.h_sampling as usize) + h;

                        // Bounds check: skip blocks outside the grid (malformed JPEGs)
                        let blocks_tall = grids[sci].blocks_tall();
                        let blocks_wide = grids[sci].blocks_wide();
                        if block_row >= blocks_tall || block_col >= blocks_wide {
                            // Still need to consume the entropy-coded data for this block
                            // to keep the bitstream in sync, so decode but discard.
                            let dc_size = dc_tab.decode(&mut reader)?;
                            if dc_size > 0 {
                                let dc_bits = reader.read_bits(dc_size)?;
                                let dc_diff = extend_sign(dc_bits, dc_size);
                                dc_pred[sci] += dc_diff as i32;
                            }
                            let mut k = 1;
                            while k < 64 {
                                let rs = ac_tab.decode(&mut reader)?;
                                let run = (rs >> 4) as usize;
                                let size = rs & 0x0F;
                                if size == 0 {
                                    if run == 0 || run != 15 { break; }
                                    k += 16;
                                    continue;
                                }
                                k += run;
                                if k >= 64 { return Err(JpegError::HuffmanDecode); }
                                let _ac_bits = reader.read_bits(size)?;
                                k += 1;
                            }
                            continue;
                        }

                        let mut zz = [0i16; 64];

                        // Decode DC coefficient
                        let dc_size = dc_tab.decode(&mut reader)?;
                        if dc_size > 0 {
                            let dc_bits = reader.read_bits(dc_size)?;
                            let dc_diff = extend_sign(dc_bits, dc_size);
                            dc_pred[sci] += dc_diff as i32;
                        }
                        zz[0] = dc_pred[sci].clamp(i16::MIN as i32, i16::MAX as i32) as i16;

                        // Decode AC coefficients
                        let mut k = 1;
                        while k < 64 {
                            let rs = ac_tab.decode(&mut reader)?;
                            let run = (rs >> 4) as usize;
                            let size = rs & 0x0F;

                            if size == 0 {
                                if run == 0 {
                                    // EOB — remaining ACs are zero
                                    break;
                                } else if run == 15 {
                                    // ZRL — skip 16 zeros
                                    k += 16;
                                    continue;
                                } else {
                                    break;
                                }
                            }

                            k += run;
                            if k >= 64 {
                                return Err(JpegError::HuffmanDecode);
                            }
                            let ac_bits = reader.read_bits(size)?;
                            zz[k] = extend_sign(ac_bits, size);
                            k += 1;
                        }

                        // Convert zigzag to natural order and store
                        let block = grids[sci].block_mut(block_row, block_col);
                        for zi in 0..64 {
                            block[ZIGZAG_TO_NATURAL[zi]] = zz[zi];
                        }
                    }
                }
            }

            mcu_count += 1;
        }
    }

    // Find end position: align to byte, then skip past any trailing marker
    let end_pos = reader.position();

    Ok((grids, end_pos))
}

/// Encode DctGrids back to entropy-coded scan data.
///
/// Returns the raw entropy-coded bytes (without SOS header, but including
/// restart markers if restart_interval > 0).
pub fn encode_scan(
    frame: &FrameInfo,
    scan_components: &[ScanComponent],
    grids: &[DctGrid],
    dc_specs: &[Option<HuffmanSpec>; 4],
    ac_specs: &[Option<HuffmanSpec>; 4],
    restart_interval: u16,
) -> Result<Vec<u8>> {
    // Build Huffman encode tables
    let mut dc_tables: [Option<HuffmanEncodeTable>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<HuffmanEncodeTable>; 4] = [None, None, None, None];

    for sc in scan_components {
        if dc_tables[sc.dc_table].is_none() {
            let spec = dc_specs[sc.dc_table]
                .as_ref()
                .ok_or(JpegError::InvalidHuffmanTableId(sc.dc_table as u8))?;
            dc_tables[sc.dc_table] = Some(HuffmanEncodeTable::build(&spec.bits, &spec.huffval));
        }
        if ac_tables[sc.ac_table].is_none() {
            let spec = ac_specs[sc.ac_table]
                .as_ref()
                .ok_or(JpegError::InvalidHuffmanTableId(sc.ac_table as u8))?;
            ac_tables[sc.ac_table] = Some(HuffmanEncodeTable::build(&spec.bits, &spec.huffval));
        }
    }

    // Use a byte accumulator so we can insert restart markers between segments
    let mut output = Vec::new();
    let mut writer = BitWriter::new();
    let mut dc_pred = vec![0i32; scan_components.len()];
    let mut mcu_count = 0usize;
    let mut restart_count = 0u16;

    for mcu_row in 0..frame.mcus_tall as usize {
        for mcu_col in 0..frame.mcus_wide as usize {
            // Insert restart marker if needed
            if restart_interval > 0 && mcu_count > 0 && mcu_count % (restart_interval as usize) == 0 {
                // Flush current segment
                let segment = std::mem::replace(&mut writer, BitWriter::new()).flush();
                output.extend_from_slice(&segment);

                // Write RST marker (not byte-stuffed — markers are outside entropy data)
                let rst_marker = 0xD0 + (restart_count % 8) as u8;
                output.push(0xFF);
                output.push(rst_marker);
                restart_count += 1;

                // Reset DC predictors
                for pred in &mut dc_pred {
                    *pred = 0;
                }
            }

            // Encode blocks for each component in this MCU
            for (sci, sc) in scan_components.iter().enumerate() {
                let comp = &frame.components[sc.comp_idx];
                let dc_tab = dc_tables[sc.dc_table].as_ref().unwrap();
                let ac_tab = ac_tables[sc.ac_table].as_ref().unwrap();

                for v in 0..comp.v_sampling as usize {
                    for h in 0..comp.h_sampling as usize {
                        let block_row = mcu_row * (comp.v_sampling as usize) + v;
                        let block_col = mcu_col * (comp.h_sampling as usize) + h;

                        // Read block and convert natural → zigzag
                        let block = grids[sci].block(block_row, block_col);
                        let mut zz = [0i16; 64];
                        for ni in 0..64 {
                            zz[NATURAL_TO_ZIGZAG[ni]] = block[ni];
                        }

                        // Encode DC
                        let dc_diff = (zz[0] as i32 - dc_pred[sci]) as i16;
                        dc_pred[sci] = zz[0] as i32;
                        let (dc_bits, dc_size) = encode_value(dc_diff);
                        let (dc_code, dc_code_len) = dc_tab.encode(dc_size)?;
                        writer.write_bits(dc_code, dc_code_len);
                        if dc_size > 0 {
                            writer.write_bits(dc_bits, dc_size);
                        }

                        // Encode AC
                        let mut k = 1;
                        while k < 64 {
                            // Find run of zeros
                            let mut run = 0usize;
                            while k + run < 64 && zz[k + run] == 0 {
                                run += 1;
                            }

                            if k + run >= 64 {
                                // EOB
                                let (eob_code, eob_len) = ac_tab.encode(0x00)?;
                                writer.write_bits(eob_code, eob_len);
                                break;
                            }

                            // Emit ZRL (16 zeros) as needed
                            while run >= 16 {
                                let (zrl_code, zrl_len) = ac_tab.encode(0xF0)?;
                                writer.write_bits(zrl_code, zrl_len);
                                run -= 16;
                                k += 16;
                            }

                            k += run;
                            let (ac_bits, ac_size) = encode_value(zz[k]);
                            let rs = ((run as u8) << 4) | ac_size;
                            let (ac_code, ac_code_len) = ac_tab.encode(rs)?;
                            writer.write_bits(ac_code, ac_code_len);
                            if ac_size > 0 {
                                writer.write_bits(ac_bits, ac_size);
                            }
                            k += 1;
                        }
                    }
                }
            }

            mcu_count += 1;
        }
    }

    // Flush final segment
    output.extend_from_slice(&writer.flush());
    Ok(output)
}

/// Decode a single progressive scan into existing DctGrids.
///
/// Progressive JPEG has multiple scans, each contributing partial coefficient data.
/// This function decodes one scan (identified by its SOS parameters) and accumulates
/// the results into the provided grids.
///
/// The four scan types are:
/// - **DC first** (Ss=0, Se=0, Ah=0): Initial DC coefficients, shifted left by Al
/// - **DC refining** (Ss=0, Se=0, Ah>0): One correction bit per DC coefficient
/// - **AC first** (Ss>0, Ah=0): Initial AC coefficients for a spectral band
/// - **AC refining** (Ss>0, Ah>0): Correction bits for previously-decoded AC coefficients
///
/// Returns the byte position after the scan data.
#[allow(unused_assignments)]
pub fn decode_progressive_scan(
    data: &[u8],
    scan_start: usize,
    frame: &FrameInfo,
    scan_components: &[ScanComponent],
    dc_specs: &[Option<HuffmanSpec>; 4],
    ac_specs: &[Option<HuffmanSpec>; 4],
    restart_interval: u16,
    params: &SosParams,
    grids: &mut [DctGrid],
) -> Result<usize> {
    let ss = params.ss as usize;
    let se = params.se as usize;
    let ah = params.ah;
    let al = params.al;

    // Validate parameters
    if ss > 63 || se > 63 || ss > se {
        return Err(JpegError::InvalidMarkerData("invalid spectral selection"));
    }

    // Build Huffman decode tables (only for tables actually needed by this scan)
    let mut dc_tables: [Option<HuffmanDecodeTable>; 4] = [None, None, None, None];
    let mut ac_tables: [Option<HuffmanDecodeTable>; 4] = [None, None, None, None];

    for sc in scan_components {
        // DC tables needed for DC scans (ss == 0)
        if ss == 0 && ah == 0 && dc_tables[sc.dc_table].is_none() {
            let spec = dc_specs[sc.dc_table]
                .as_ref()
                .ok_or(JpegError::InvalidHuffmanTableId(sc.dc_table as u8))?;
            dc_tables[sc.dc_table] = Some(HuffmanDecodeTable::build(&spec.bits, &spec.huffval)?);
        }
        // AC tables needed for AC scans (ss > 0)
        if ss > 0 && ac_tables[sc.ac_table].is_none() {
            let spec = ac_specs[sc.ac_table]
                .as_ref()
                .ok_or(JpegError::InvalidHuffmanTableId(sc.ac_table as u8))?;
            ac_tables[sc.ac_table] = Some(HuffmanDecodeTable::build(&spec.bits, &spec.huffval)?);
        }
    }

    let mut reader = BitReader::new(data, scan_start);
    let mut dc_pred = vec![0i32; scan_components.len()];
    let mut mcu_count = 0usize;

    // For AC scans, track the End-of-Band run counter
    let mut eob_run: u32 = 0;

    // Determine if this is a non-interleaved scan (single component)
    // Non-interleaved scans process blocks in raster order within that component.
    let non_interleaved = scan_components.len() == 1 && (ss > 0 || se > 0 || frame.components.len() == 1);

    if non_interleaved && scan_components.len() == 1 {
        // Non-interleaved scan: iterate blocks in raster order for the single component
        let sc = &scan_components[0];
        let bw = frame.blocks_wide(sc.comp_idx);
        let bt = frame.blocks_tall(sc.comp_idx);

        // For non-interleaved scans, MCU = 1 block
        // Restart interval counts blocks, not MCUs
        let mut block_count = 0usize;

        for block_row in 0..bt {
            for block_col in 0..bw {
                // Handle restart markers
                if restart_interval > 0 && block_count > 0 && block_count % (restart_interval as usize) == 0 {
                    reader.byte_align();
                    let _rst = reader.check_restart_marker()?;
                    dc_pred[0] = 0;
                    eob_run = 0;
                }

                let grid = &mut grids[sc.comp_idx];

                if ss == 0 {
                    // DC scan
                    if ah == 0 {
                        // DC first scan
                        decode_dc_first(&mut reader, &dc_tables, sc, &mut dc_pred[0], al, grid, block_row, block_col)?;
                    } else {
                        // DC refining scan
                        decode_dc_refine(&mut reader, al, grid, block_row, block_col)?;
                    }
                }

                if se > 0 {
                    // AC scan (might also include DC if ss == 0, but for progressive
                    // AC and DC are always in separate scans)
                    let ac_start = if ss == 0 { 1 } else { ss };
                    if ah == 0 {
                        // AC first scan
                        decode_ac_first(&mut reader, &ac_tables, sc, al, ac_start, se, &mut eob_run, grid, block_row, block_col)?;
                    } else {
                        // AC refining scan
                        decode_ac_refine(&mut reader, &ac_tables, sc, al, ac_start, se, &mut eob_run, grid, block_row, block_col)?;
                    }
                }

                block_count += 1;
            }
        }
    } else {
        // Interleaved scan (DC scans with multiple components)
        for mcu_row in 0..frame.mcus_tall as usize {
            for mcu_col in 0..frame.mcus_wide as usize {
                // Handle restart markers
                if restart_interval > 0 && mcu_count > 0 && mcu_count % (restart_interval as usize) == 0 {
                    reader.byte_align();
                    let _rst = reader.check_restart_marker()?;
                    for pred in &mut dc_pred {
                        *pred = 0;
                    }
                    eob_run = 0;
                }

                for (sci, sc) in scan_components.iter().enumerate() {
                    let comp = &frame.components[sc.comp_idx];

                    for v in 0..comp.v_sampling as usize {
                        for h in 0..comp.h_sampling as usize {
                            let block_row = mcu_row * (comp.v_sampling as usize) + v;
                            let block_col = mcu_col * (comp.h_sampling as usize) + h;

                            let grid = &mut grids[sc.comp_idx];

                            if ss == 0 {
                                if ah == 0 {
                                    decode_dc_first(&mut reader, &dc_tables, sc, &mut dc_pred[sci], al, grid, block_row, block_col)?;
                                } else {
                                    decode_dc_refine(&mut reader, al, grid, block_row, block_col)?;
                                }
                            }
                            // Interleaved scans are DC-only in progressive JPEG
                        }
                    }
                }

                mcu_count += 1;
            }
        }
    }

    let end_pos = reader.position();
    Ok(end_pos)
}

/// DC first scan: Huffman-decode the DC difference and shift left by Al.
fn decode_dc_first(
    reader: &mut BitReader,
    dc_tables: &[Option<HuffmanDecodeTable>; 4],
    sc: &ScanComponent,
    dc_pred: &mut i32,
    al: u8,
    grid: &mut DctGrid,
    block_row: usize,
    block_col: usize,
) -> Result<()> {
    let dc_tab = dc_tables[sc.dc_table]
        .as_ref()
        .ok_or(JpegError::InvalidHuffmanTableId(sc.dc_table as u8))?;
    let dc_size = dc_tab.decode(reader)?;
    if dc_size > 0 {
        let dc_bits = reader.read_bits(dc_size)?;
        let dc_diff = extend_sign(dc_bits, dc_size);
        *dc_pred += dc_diff as i32;
    }
    // Store DC coefficient shifted left by Al (successive approximation), clamped to i16 range
    let block = grid.block_mut(block_row, block_col);
    block[0] = ((*dc_pred).clamp(i16::MIN as i32, i16::MAX as i32) as i16) << al;
    Ok(())
}

/// DC refining scan: read one correction bit per DC coefficient.
fn decode_dc_refine(
    reader: &mut BitReader,
    al: u8,
    grid: &mut DctGrid,
    block_row: usize,
    block_col: usize,
) -> Result<()> {
    let bit = reader.read_bits(1)?;
    let block = grid.block_mut(block_row, block_col);
    // Set bit Al of the DC coefficient
    if bit != 0 {
        block[0] |= 1i16 << al;
    }
    Ok(())
}

/// AC first scan: decode AC coefficients for the spectral band [ss..se].
///
/// Similar to baseline AC decoding but operates on a sub-range of zigzag positions
/// and supports EOBn (End of Band) run-length coding across multiple blocks.
fn decode_ac_first(
    reader: &mut BitReader,
    ac_tables: &[Option<HuffmanDecodeTable>; 4],
    sc: &ScanComponent,
    al: u8,
    ss: usize,
    se: usize,
    eob_run: &mut u32,
    grid: &mut DctGrid,
    block_row: usize,
    block_col: usize,
) -> Result<()> {
    let block = grid.block_mut(block_row, block_col);

    if *eob_run > 0 {
        // We are in an EOB run — this block's coefficients in [ss..se] are all zero.
        *eob_run -= 1;
        return Ok(());
    }

    let ac_tab = ac_tables[sc.ac_table]
        .as_ref()
        .ok_or(JpegError::InvalidHuffmanTableId(sc.ac_table as u8))?;

    let mut k = ss;
    while k <= se {
        let rs = ac_tab.decode(reader)?;
        let run = (rs >> 4) as usize;
        let size = rs & 0x0F;

        if size == 0 {
            if run == 15 {
                // ZRL: skip 16 zero positions
                k += 16;
                continue;
            } else {
                // EOBn: End of Band for 2^run + extra blocks
                // run=0 means EOB for this block only (EOB0 = 1 block)
                // run=1..14 means read `run` extra bits to get the EOB run length
                *eob_run = 1u32 << (run as u32);
                if run > 0 {
                    let extra = reader.read_bits(run as u8)? as u32;
                    *eob_run += extra;
                }
                *eob_run -= 1; // Current block is part of the run
                return Ok(());
            }
        }

        k += run;
        if k > se {
            return Err(JpegError::HuffmanDecode);
        }

        let ac_bits = reader.read_bits(size)?;
        let val = extend_sign(ac_bits, size);
        // Store in natural order, shifted left by Al
        block[ZIGZAG_TO_NATURAL[k]] = val << al;
        k += 1;
    }

    Ok(())
}

/// AC refining scan: read correction bits for previously-nonzero coefficients
/// and new nonzero coefficients in the spectral band [ss..se].
///
/// This is the most complex part of progressive JPEG decoding.
/// The algorithm from ITU-T T.81 Figure G.7 interleaves:
/// - Correction bits for coefficients that were already nonzero
/// - New coefficients (with zero-run and EOBn coding)
fn decode_ac_refine(
    reader: &mut BitReader,
    ac_tables: &[Option<HuffmanDecodeTable>; 4],
    sc: &ScanComponent,
    al: u8,
    ss: usize,
    se: usize,
    eob_run: &mut u32,
    grid: &mut DctGrid,
    block_row: usize,
    block_col: usize,
) -> Result<()> {
    let block = grid.block_mut(block_row, block_col);
    let p1 = 1i16 << al;  // 1 in the bit position being corrected
    let m1 = (-1i16) << al; // -1 in the bit position being corrected (= -(1 << al))

    let ac_tab = ac_tables[sc.ac_table]
        .as_ref()
        .ok_or(JpegError::InvalidHuffmanTableId(sc.ac_table as u8))?;

    let mut k = ss;

    if *eob_run > 0 {
        // In an EOB run: just apply correction bits to nonzero coefficients
        while k <= se {
            let ni = ZIGZAG_TO_NATURAL[k];
            if block[ni] != 0 {
                let bit = reader.read_bits(1)?;
                if bit != 0 {
                    if block[ni] > 0 {
                        block[ni] += p1;
                    } else {
                        block[ni] += m1;
                    }
                }
            }
            k += 1;
        }
        *eob_run -= 1;
        return Ok(());
    }

    while k <= se {
        let rs = ac_tab.decode(reader)?;
        let run = (rs >> 4) as usize; // Number of zero-valued coefficients to skip
        let size = rs & 0x0F;

        if size == 0 {
            if run == 15 {
                // ZRL: skip 16 zero-valued positions, applying correction bits to nonzero
                let mut zeros_to_skip = 16usize;
                while k <= se && zeros_to_skip > 0 {
                    let ni = ZIGZAG_TO_NATURAL[k];
                    if block[ni] != 0 {
                        // Apply correction bit
                        let bit = reader.read_bits(1)?;
                        if bit != 0 {
                            if block[ni] > 0 {
                                block[ni] += p1;
                            } else {
                                block[ni] += m1;
                            }
                        }
                    } else {
                        zeros_to_skip -= 1;
                    }
                    k += 1;
                }
                continue;
            } else {
                // EOBn: remaining coefficients in this band get correction bits only
                *eob_run = 1u32 << (run as u32);
                if run > 0 {
                    let extra = reader.read_bits(run as u8)? as u32;
                    *eob_run += extra;
                }
                // Apply correction bits to remaining nonzero coefficients in this block
                while k <= se {
                    let ni = ZIGZAG_TO_NATURAL[k];
                    if block[ni] != 0 {
                        let bit = reader.read_bits(1)?;
                        if bit != 0 {
                            if block[ni] > 0 {
                                block[ni] += p1;
                            } else {
                                block[ni] += m1;
                            }
                        }
                    }
                    k += 1;
                }
                *eob_run -= 1;
                return Ok(());
            }
        } else if size == 1 {
            // New nonzero coefficient after skipping `run` zero-valued positions
            // Read the sign bit
            let sign_bit = reader.read_bits(1)?;
            let new_val = if sign_bit != 0 { p1 } else { m1 };

            // Skip `run` zero-valued coefficients, applying correction bits to nonzero
            let mut zeros_to_skip = run;
            while k <= se {
                let ni = ZIGZAG_TO_NATURAL[k];
                if block[ni] != 0 {
                    // Apply correction bit to existing nonzero
                    let bit = reader.read_bits(1)?;
                    if bit != 0 {
                        if block[ni] > 0 {
                            block[ni] += p1;
                        } else {
                            block[ni] += m1;
                        }
                    }
                } else {
                    if zeros_to_skip == 0 {
                        // Place the new coefficient here
                        block[ni] = new_val;
                        k += 1;
                        break;
                    }
                    zeros_to_skip -= 1;
                }
                k += 1;
            }
            continue;
        } else {
            // Invalid: size must be 0 or 1 in AC refining scans
            return Err(JpegError::HuffmanDecode);
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_value_dc_diff() {
        // DC difference of 5: category 3 (size=3), bits = 5 = 0b101
        let (bits, size) = encode_value(5);
        assert_eq!(size, 3);
        assert_eq!(bits, 5);

        // DC difference of -3: category 2 (size=2), bits = 0 (one's complement of 3 in 2 bits)
        let (bits, size) = encode_value(-3);
        assert_eq!(size, 2);
        let recovered = extend_sign(bits, size);
        assert_eq!(recovered, -3);
    }
}
