// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Spec-correct P-slice bit-level parser for debugging encoder output.
//!
//! Wraps `parse_macroblock_with_recon` with mb_skip_run handling and
//! dumps bit positions per MB. Shows exactly how many bits each MB
//! consumes according to our own macroblock parser, so we can look for
//! divergences against ffmpeg.

use std::env;
use std::fs;

use phasm_core::codec::h264::bitstream::{
    parse_nal_units_annexb, remove_emulation_prevention_with_map, RbspReader,
};
use phasm_core::codec::h264::macroblock::{parse_macroblock_with_recon, NeighborContext};
use phasm_core::codec::h264::mv::MvPredictorContext;
use phasm_core::codec::h264::slice::parse_slice_header;
use phasm_core::codec::h264::sps::{parse_pps, parse_sps};
use phasm_core::codec::h264::NalType;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: {} <input.h264> [max_mbs]", args[0]);
        std::process::exit(2);
    }
    let bytes = fs::read(&args[1]).expect("read input");
    let max_mbs: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(usize::MAX);

    let nals = parse_nal_units_annexb(&bytes).expect("parse nals");
    let sps_nal = nals.iter().find(|n| n.nal_type == NalType::SPS).expect("SPS");
    let pps_nal = nals.iter().find(|n| n.nal_type == NalType::PPS).expect("PPS");
    let sps = parse_sps(&sps_nal.rbsp).expect("parse sps");
    let pps = parse_pps(&pps_nal.rbsp).expect("parse pps");

    let mb_w = sps.pic_width_in_mbs as usize;
    let mb_h = sps.pic_height_in_map_units as usize;
    let total_mbs = mb_w * mb_h;
    eprintln!("Frame: {mb_w}x{mb_h} MBs ({total_mbs} total)");

    for nal in &nals {
        if !matches!(nal.nal_type, NalType::SLICE | NalType::SLICE_IDR) {
            continue;
        }
        let hdr = parse_slice_header(&nal.rbsp, &sps, &pps, nal.nal_type, nal.nal_ref_idc)
            .expect("slice hdr");
        eprintln!("\n=== slice type {:?} ===", hdr.slice_type);

        let (rbsp, ep_map) = remove_emulation_prevention_with_map(&nal.rbsp);
        let mut reader = RbspReader::new(&rbsp);
        reader.skip_bits(hdr.data_bit_offset as u32).expect("skip hdr");

        use phasm_core::codec::h264::slice::SliceType;
        let is_p = matches!(hdr.slice_type, SliceType::P);

        let mut ctx = NeighborContext::new(mb_w as u32, mb_h as u32);
        let mut mv_ctx = MvPredictorContext::new(mb_w as u32, mb_h as u32);
        let mut current_qp = (pps.pic_init_qp_minus26 as i32 + 26) + hdr.slice_qp_delta as i32;

        let mut mb_addr = 0usize;
        let mut skip_remaining = 0u32;
        let mut iter = 0usize;
        while mb_addr < total_mbs && iter < max_mbs {
            let mb_x = (mb_addr % mb_w) as u32;
            let mb_y = (mb_addr / mb_w) as u32;
            let start_bit = reader.bits_read();

            if is_p {
                if skip_remaining > 0 {
                    eprintln!("MB ({mb_x},{mb_y}) [{start_bit}] P_Skip (run)");
                    skip_remaining -= 1;
                    mb_addr += 1;
                    iter += 1;
                    continue;
                }
                // Read mb_skip_run before this (non-skipped) MB.
                let r = reader.read_ue();
                match r {
                    Ok(n) => {
                        if n > 0 {
                            eprintln!(
                                "MB ({mb_x},{mb_y}) [{start_bit}] mb_skip_run = {n} @ +{} bits",
                                reader.bits_read() - start_bit
                            );
                            skip_remaining = n;
                            skip_remaining -= 1;
                            mb_addr += 1;
                            iter += 1;
                            continue;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "MB ({mb_x},{mb_y}) [{start_bit}] FAIL reading mb_skip_run: {e}"
                        );
                        break;
                    }
                }
            }

            let after_skip_bit = reader.bits_read();
            match parse_macroblock_with_recon(
                &mut reader,
                hdr.slice_type,
                mb_x,
                mb_y,
                &sps,
                &pps,
                &mut ctx,
                &ep_map,
                &rbsp,
                &mut current_qp,
                pps.num_ref_idx_l0_default,
                false,
                Some(&mut mv_ctx),
            ) {
                Ok(mb) => {
                    let end_bit = reader.bits_read();
                    eprintln!(
                        "MB ({mb_x},{mb_y}) [{start_bit}] type={:?} layer={} bits (skip_run=0 → +1 bit, layer starts @ {after_skip_bit})",
                        mb.mb_type,
                        end_bit - start_bit
                    );
                }
                Err(e) => {
                    let err_bit = reader.bits_read();
                    eprintln!(
                        "MB ({mb_x},{mb_y}) [{start_bit}] FAIL @ bit {err_bit}: {e}"
                    );
                    break;
                }
            }

            mb_addr += 1;
            iter += 1;
        }
        eprintln!("parsed {iter} iterations; end mb_addr {mb_addr}/{total_mbs}");
    }
}
