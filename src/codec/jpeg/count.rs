// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! What a parse of a JPEG will hold, read without storing its coefficients:
//! the frame, the coefficient count of its grids and the luma's non-zero AC
//! coefficients (for memory estimates made before a decode,
//! `stego::memory::decode_peak_bytes`).
//!
//! The scans are Huffman-decoded as [`super::JpegImage::from_bytes`] decodes
//! them, but only the luma's AC coefficients are tracked, one bit each (the
//! scans without luma AC are skipped): the working set is the luma's
//! coefficient count / 8 bytes, the time about a parse's.

use super::bitio::BitReader;
use super::error::{JpegError, Result};
use super::frame::{self, FrameInfo};
use super::huffman::HuffmanDecodeTable;
use super::marker::{self, iterate_markers_all, parse_dri, parse_sos, parse_sos_params, SosParams};
use super::scan::{walk_scan, ScanComponent, ScanStep};
use super::tables::{parse_dht, HuffmanSpec};

/// A JPEG's shape as a parse holds it ([`coefficient_shape`]).
#[derive(Debug, Clone)]
pub struct CoefficientShape {
    /// The frame (SOF0 or SOF2, as the parse accepts).
    pub frame: FrameInfo,
    /// Coefficients of every component's MCU-padded grid (2 B each in a
    /// parse).
    pub coefficients: usize,
    /// Coefficients of the luma's MCU-padded grid.
    pub luma_coefficients: usize,
    /// The luma's non-zero AC coefficients, as
    /// `count_nonzero_ac(img.dct_grid(0))` of the parse counts them.
    pub luma_nonzero_ac: usize,
}

/// Most luma coefficients [`coefficient_shape`] counts (a 32 MiB bitmap;
/// 16384 × 16384 px). A larger frame is not counted ([`frame_shape`]
/// bounds it).
pub const MAX_COUNTED_LUMA_COEFFICIENTS: u64 = 1 << 28;

/// Coefficients of `frame`'s MCU-padded grids: every component's and the
/// luma's, in u64 (a header can claim 65535 × 65535 px: more than a 32-bit
/// `usize` holds).
pub fn frame_coefficients(frame: &FrameInfo) -> (u64, u64) {
    let grid = |c: usize| frame.blocks_wide(c) as u64 * frame.blocks_tall(c) as u64 * 64;
    ((0..frame.components.len()).map(grid).sum(), grid(0))
}

/// The frame of the JPEG `data` from its header alone (SOF0 or SOF2 before
/// the first scan), for a bound without the count: every luma AC
/// coefficient taken as non-zero.
pub fn frame_shape(data: &[u8]) -> Result<CoefficientShape> {
    let (entries, _) = super::marker::iterate_markers(data)?;
    let frame = entries
        .iter()
        .find_map(|e| match e.marker {
            marker::SOF2 => Some(frame::parse_sof_ext(&e.data, true)),
            marker::SOF0 => Some(frame::parse_sof(&e.data)),
            _ => None,
        })
        .ok_or(JpegError::InvalidMarkerData("no SOF marker found"))??;
    let (all, luma) = frame_coefficients(&frame);
    let sat = |v: u64| usize::try_from(v).unwrap_or(usize::MAX);
    Ok(CoefficientShape {
        coefficients: sat(all),
        luma_coefficients: sat(luma),
        luma_nonzero_ac: sat(luma / 64 * 63),
        frame,
    })
}

/// The [`CoefficientShape`] of the JPEG `data`. Errors where the parse would
/// (no SOF0 / SOF2 frame, a scan that does not decode), and for a frame of
/// more than [`MAX_COUNTED_LUMA_COEFFICIENTS`] luma coefficients.
pub fn coefficient_shape(data: &[u8]) -> Result<CoefficientShape> {
    let (entries, scan_starts) = iterate_markers_all(data)?;
    let mut frame: Option<FrameInfo> = None;
    let mut dc_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
    let mut ac_specs: [Option<HuffmanSpec>; 4] = [None, None, None, None];
    let mut restart_interval = 0u16;
    let mut nonzero: Vec<u64> = Vec::new();
    let mut sos_index = 0usize;
    for entry in &entries {
        match entry.marker {
            marker::SOF0 | marker::SOF2 => {
                let fi = if entry.marker == marker::SOF2 {
                    frame::parse_sof_ext(&entry.data, true)?
                } else {
                    frame::parse_sof(&entry.data)?
                };
                let (_, luma) = frame_coefficients(&fi);
                if luma > MAX_COUNTED_LUMA_COEFFICIENTS {
                    return Err(JpegError::InvalidMarkerData("frame too large to count"));
                }
                nonzero = vec![0u64; (luma / 64) as usize];
                frame = Some(fi);
            }
            marker::DHT => {
                for spec in parse_dht(&entry.data)? {
                    let id = spec.id as usize;
                    if spec.class == 0 {
                        dc_specs[id] = Some(spec);
                    } else {
                        ac_specs[id] = Some(spec);
                    }
                }
            }
            marker::DRI => restart_interval = parse_dri(&entry.data)?,
            marker::SOS => {
                let fi = frame.as_ref().ok_or(JpegError::InvalidMarkerData("SOS before SOF"))?;
                let start = *scan_starts.get(sos_index).ok_or(JpegError::UnexpectedEof)?;
                sos_index += 1;
                let mut components = Vec::new();
                for (comp_id, dc, ac) in parse_sos(&entry.data)? {
                    let comp_idx = fi.components.iter().position(|c| c.id == comp_id).ok_or(JpegError::UnknownComponentId(comp_id))?;
                    components.push(ScanComponent { comp_idx, dc_table: dc as usize, ac_table: ac as usize });
                }
                let params = parse_sos_params(&entry.data)?;
                // Only scans that code the luma's AC coefficients change the count.
                if params.se > 0 && components.iter().any(|c| c.comp_idx == 0) {
                    count_scan(data, start, fi, &components, &dc_specs, &ac_specs, restart_interval, &params, &mut nonzero)?;
                }
            }
            _ => {}
        }
    }
    let frame = frame.ok_or(JpegError::InvalidMarkerData("no SOF marker found"))?;
    let (all, luma) = frame_coefficients(&frame);
    let (coefficients, luma_coefficients) = (all as usize, luma as usize);
    let luma_nonzero_ac = nonzero.iter().map(|w| w.count_ones() as usize).sum();
    Ok(CoefficientShape { frame, coefficients, luma_coefficients, luma_nonzero_ac })
}

/// Decode one scan that codes luma AC, setting the bit of every luma AC
/// coefficient it makes non-zero (`nonzero`: one bit per coefficient of the
/// padded luma grid, block `b`'s zigzag position `k` at bit `64·b + k`). The
/// other components' blocks of an interleaved scan are decoded and dropped.
#[allow(clippy::too_many_arguments)]
fn count_scan(
    data: &[u8],
    start: usize,
    frame: &FrameInfo,
    components: &[ScanComponent],
    dc_specs: &[Option<HuffmanSpec>; 4],
    ac_specs: &[Option<HuffmanSpec>; 4],
    restart_interval: u16,
    params: &SosParams,
    nonzero: &mut [u64],
) -> Result<()> {
    let (ss, se, ah, al) = (params.ss as usize, params.se as usize, params.ah, params.al);
    if ss > 63 || se > 63 || ss > se || al > 13 {
        return Err(JpegError::InvalidMarkerData("invalid spectral selection"));
    }
    if ss > 0 && components.len() != 1 {
        return Err(JpegError::InvalidMarkerData("interleaved AC scan"));
    }
    let table = |specs: &[Option<HuffmanSpec>; 4], id: usize| -> Result<HuffmanDecodeTable> {
        let spec = specs.get(id).and_then(|s| s.as_ref()).ok_or(JpegError::InvalidHuffmanTableId(id as u8))?;
        HuffmanDecodeTable::build(&spec.bits, &spec.huffval)
    };
    let mut dc_tables = Vec::with_capacity(components.len());
    let mut ac_tables = Vec::with_capacity(components.len());
    for sc in components {
        dc_tables.push(if ss == 0 && ah == 0 { Some(table(dc_specs, sc.dc_table)?) } else { None });
        ac_tables.push(table(ac_specs, sc.ac_table)?);
    }
    // A sequential frame's scans read EOB as the end of the block (as
    // `scan::decode_scan`); a progressive frame's AC scans read EOBn runs.
    let progressive = frame.is_progressive;
    let luma_bw = frame.blocks_wide(0);
    let mut reader = BitReader::new(data, start);
    let mut eob_run: u32 = 0;
    walk_scan(frame, components, restart_interval, |step| -> Result<()> {
        let (sci, br, bc) = match step {
            ScanStep::Row(_) => return Ok(()),
            ScanStep::Restart => {
                reader.byte_align();
                reader.check_restart_marker()?;
                eob_run = 0;
                return Ok(());
            }
            ScanStep::Block { sci, br, bc } => (sci, br, bc),
        };
        let luma = components[sci].comp_idx == 0;
        let base = (br * luma_bw + bc) * 64;
        if ss == 0 {
            if ah == 0 {
                let size = dc_tables[sci].as_ref().unwrap().decode(&mut reader)?;
                if size > 0 {
                    reader.read_bits(size)?;
                }
            } else {
                reader.read_bits(1)?;
            }
        }
        let ac = &ac_tables[sci];
        let lo = ss.max(1);
        if ah == 0 || !progressive {
            // First scan of the band (or a sequential scan): every coded
            // coefficient is non-zero.
            if eob_run > 0 {
                eob_run -= 1;
                return Ok(());
            }
            let mut k = lo;
            while k <= se {
                let rs = ac.decode(&mut reader)?;
                let (run, size) = ((rs >> 4) as usize, rs & 0x0F);
                if size == 0 {
                    if run == 15 {
                        k += 16;
                        continue;
                    }
                    if progressive {
                        eob_run = (1u32 << run) - 1;
                        if run > 0 {
                            eob_run += reader.read_bits(run as u8)? as u32;
                        }
                    }
                    break;
                }
                k += run;
                if k > se {
                    return Err(JpegError::HuffmanDecode);
                }
                reader.read_bits(size)?;
                if luma {
                    set_bit(nonzero, base + k);
                }
                k += 1;
            }
            return Ok(());
        }
        if !luma {
            // A refinement scan of another component never gets here (only
            // scans with luma AC are read, and refinement scans code one
            // component).
            return Err(JpegError::InvalidMarkerData("interleaved refinement scan"));
        }
        // Refinement (T.81 G.1.2.3): a correction bit for every coefficient
        // already non-zero; new non-zero coefficients after their zero run.
        // Refinement scans code one component: here the luma.
        let mut k = lo;
        if eob_run > 0 {
            while k <= se {
                if bit(nonzero, base + k) {
                    reader.read_bits(1)?;
                }
                k += 1;
            }
            eob_run -= 1;
            return Ok(());
        }
        while k <= se {
            let rs = ac.decode(&mut reader)?;
            let (run, size) = ((rs >> 4) as usize, rs & 0x0F);
            match size {
                0 if run == 15 => {
                    // ZRL: 16 zero coefficients (correction bits for the
                    // non-zero ones passed), up to the 16th zero.
                    let mut zeros = 16;
                    while k <= se && zeros > 0 {
                        if bit(nonzero, base + k) {
                            reader.read_bits(1)?;
                        } else {
                            zeros -= 1;
                        }
                        k += 1;
                    }
                }
                0 => {
                    eob_run = 1u32 << run;
                    if run > 0 {
                        eob_run += reader.read_bits(run as u8)? as u32;
                    }
                    break;
                }
                1 => {
                    // The new coefficient's sign bit, then `run` zero
                    // coefficients (correction bits for the non-zero ones
                    // passed); it takes the next zero position (none left in
                    // the band: dropped, as the parse drops it).
                    reader.read_bits(1)?;
                    let mut zeros = run;
                    while k <= se {
                        if bit(nonzero, base + k) {
                            reader.read_bits(1)?;
                        } else {
                            if zeros == 0 {
                                set_bit(nonzero, base + k);
                                k += 1;
                                break;
                            }
                            zeros -= 1;
                        }
                        k += 1;
                    }
                }
                _ => return Err(JpegError::HuffmanDecode),
            }
        }
        if eob_run > 0 {
            // EOBn: the rest of this block's band gets correction bits.
            while k <= se {
                if bit(nonzero, base + k) {
                    reader.read_bits(1)?;
                }
                k += 1;
            }
            eob_run -= 1;
        }
        Ok(())
    })
}

fn set_bit(words: &mut [u64], i: usize) {
    words[i >> 6] |= 1u64 << (i & 63);
}

fn bit(words: &[u64], i: usize) -> bool {
    (words[i >> 6] >> (i & 63)) & 1 == 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::jpeg::JpegImage;

    fn jpegs(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
        let Ok(rd) = std::fs::read_dir(dir) else { return };
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                jpegs(&p, out);
            } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("jpg") || x.eq_ignore_ascii_case("jpeg")) {
                out.push(p);
            }
        }
    }

    /// The count equals the parse's on every JPEG test vector (baseline,
    /// progressive, multi-scan sequential, restarts, 4:4:4 / 4:2:2 / gray).
    #[test]
    fn counts_match_the_parse() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let mut files = Vec::new();
        jpegs(&root.join("test-vectors/image"), &mut files);
        jpegs(&root.join("../test-vectors"), &mut files);
        assert!(files.len() >= 20, "{} test JPEGs", files.len());
        let mut checked = 0;
        for f in &files {
            let data = std::fs::read(f).unwrap();
            let Ok(img) = JpegImage::from_bytes(&data) else { continue };
            let shape = coefficient_shape(&data).unwrap_or_else(|e| panic!("{}: {e:?}", f.display()));
            let grid = img.dct_grid(0);
            let nz: usize = grid.coeffs().chunks_exact(64).map(|b| b[1..].iter().filter(|&&c| c != 0).count()).sum();
            assert_eq!(shape.luma_nonzero_ac, nz, "{}", f.display());
            assert_eq!(shape.luma_coefficients, grid.coeffs().len(), "{}", f.display());
            let all: usize = (0..img.num_components()).map(|c| img.dct_grid(c).coeffs().len()).sum();
            assert_eq!(shape.coefficients, all, "{}", f.display());
            checked += 1;
        }
        assert!(checked >= 20, "{checked} parsed");
    }
}
