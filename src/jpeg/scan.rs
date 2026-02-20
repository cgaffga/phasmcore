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

    // Initialize DC predictors
    let mut dc_pred = vec![0i16; scan_components.len()];

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

                        let mut zz = [0i16; 64];

                        // Decode DC coefficient
                        let dc_size = dc_tab.decode(&mut reader)?;
                        if dc_size > 0 {
                            let dc_bits = reader.read_bits(dc_size)?;
                            let dc_diff = extend_sign(dc_bits, dc_size);
                            dc_pred[sci] += dc_diff;
                        }
                        zz[0] = dc_pred[sci];

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
    let mut dc_pred = vec![0i16; scan_components.len()];
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
                        let dc_diff = zz[0] - dc_pred[sci];
                        dc_pred[sci] = zz[0];
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
