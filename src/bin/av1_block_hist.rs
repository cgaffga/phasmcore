// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// stealth-audit-2026-06-29: AV1 per-block histogram collector.
//
// Drives dav1d (via core-dav1d-sys + phasm-dav1d's block_hook
// addition) to walk every decoded Av1Block and tally:
//   - block size (BLOCK_4X4 … BLOCK_128X128)
//   - intra y_mode (DC_PRED … PAETH_PRED + directional)
//   - inter mode (NEARESTMV / NEARMV / GLOBALMV / NEWMV …)
//   - reference frame (LAST / LAST2 / LAST3 / GOLDEN / BWDREF / ALTREF2 / ALTREF)
//   - motion vector magnitude (binned histogram for KS test)
//   - intra vs inter share
//
// Input: section-5 OBU stream (`av1_obu_extract` output for phasm
// .mp4, or rav1e's `--ivf` output unwrapped). Output: TSV with
// header rows per histogram type; consumed by the audit
// comparison harness for χ² / KS tests.
//
// Usage:
//   av1_block_hist <input.obu> > histograms.tsv
//
// Build:
//   cargo build -p phasm-cli --features av1-encoder --bin av1_block_hist

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use std::collections::BTreeMap;
use std::io::Write;

use core_dav1d_sys::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings,
    dav1d_err_again, dav1d_get_picture, dav1d_open, dav1d_picture_unref,
    dav1d_send_data, Dav1dContext, Dav1dData, Dav1dPhasmBlockInfo, Dav1dPhasmHooks,
    Dav1dPicture, Dav1dSettings,
};

/// Aggregator passed back via the dav1d block_hook cookie. Tallies
/// per-block field counts into BTreeMap so output ordering is stable
/// across runs.
#[derive(Default)]
struct BlockTally {
    n_blocks: u64,
    n_intra: u64,
    n_inter: u64,
    n_skip: u64,
    n_skip_mode: u64,
    /// Block size enum (0..21) → count.
    bs_hist: BTreeMap<u8, u64>,
    /// Block partition (0..9) → count.
    bp_hist: BTreeMap<u8, u64>,
    /// Block level (0..4) → count.
    bl_hist: BTreeMap<u8, u64>,
    /// Intra y_mode (0..12) → count, intra-only.
    y_mode_hist: BTreeMap<u8, u64>,
    /// Intra uv_mode → count, intra-only.
    uv_mode_hist: BTreeMap<u8, u64>,
    /// Inter mode (0..24-ish across compound/single) → count.
    inter_mode_hist: BTreeMap<u8, u64>,
    /// Motion mode (TRANSLATION / OBMC / WARP) → count.
    motion_mode_hist: BTreeMap<u8, u64>,
    /// Compound type (NONE / AVG / WEDGE / DIFFWTD / SEG) → count.
    comp_type_hist: BTreeMap<u8, u64>,
    /// Ref frame 0 (0..6: LAST..ALTREF; -1 means intra) → count.
    /// Stored as i16 so -1 fits.
    ref0_hist: BTreeMap<i16, u64>,
    /// Ref frame 1 (0..6 or -1 for non-compound).
    ref1_hist: BTreeMap<i16, u64>,
    /// MV magnitude (1/8-pel L2 norm of (mv0_x, mv0_y)) binned in
    /// log2 buckets: bucket = floor(log2(1+mag))
    ///   0 → mag in [0, 1)
    ///   1 → mag in [1, 3)
    ///   2 → mag in [3, 7)
    ///   ...
    /// One bucket per encountered magnitude — keeps as raw samples
    /// for KS test downstream (see `mv_samples` below for the raw
    /// vector).
    mv_mag_hist: BTreeMap<u8, u64>,
    /// Raw MV magnitudes (1/8-pel scalar) — kept as an unbounded
    /// Vec for KS test in the downstream Python harness. Capped to
    /// avoid OOM on long clips: only first 200_000 sampled (uniform
    /// reservoir-style).
    mv_samples: Vec<u32>,
    /// Frame type histogram — Dav1dFrameType (KEY=0, INTER=1,
    /// INTRA=2, SWITCH=3).
    frame_type_per_block: BTreeMap<u8, u64>,
    /// Distinct frame_offsets observed (for n_frames counter).
    frames_seen: std::collections::BTreeSet<u16>,
    /// Reservoir RNG state.
    reservoir_n: u64,
    reservoir_rng: u64,
}

impl BlockTally {
    fn reservoir_step(&mut self, mag: u32) {
        const CAP: usize = 200_000;
        if self.mv_samples.len() < CAP {
            self.mv_samples.push(mag);
        } else {
            // Reservoir sample: keep with prob CAP/n.
            self.reservoir_n += 1;
            self.reservoir_rng = self
                .reservoir_rng
                .wrapping_mul(0x9E3779B97F4A7C15)
                .wrapping_add(0xC0FFEE);
            let r = (self.reservoir_rng >> 16) as usize;
            let idx = r % self.reservoir_n as usize;
            if idx < CAP {
                self.mv_samples[idx] = mag;
            }
        }
    }
}

/// FFI callback fired by dav1d once per decoded Av1Block. Casts the
/// cookie back to &mut BlockTally, reads the BlockInfo fields, and
/// updates the per-category counters.
///
/// # Safety
/// `cookie` MUST be a valid `*mut BlockTally` pointer set by the
/// driver before any dav1d call. `info` MUST point to a fully-
/// initialised `Dav1dPhasmBlockInfo` (dav1d's `decode_b` zero-inits
/// then field-fills before firing).
unsafe extern "C" fn block_hook_cb(
    cookie: *mut core::ffi::c_void,
    info: *const Dav1dPhasmBlockInfo,
) {
    let tally = unsafe { &mut *(cookie as *mut BlockTally) };
    let info = unsafe { &*info };

    tally.n_blocks += 1;
    if info.intra == 1 {
        tally.n_intra += 1;
        *tally.y_mode_hist.entry(info.y_mode).or_insert(0) += 1;
        *tally.uv_mode_hist.entry(info.uv_mode).or_insert(0) += 1;
    } else {
        tally.n_inter += 1;
        *tally.inter_mode_hist.entry(info.inter_mode).or_insert(0) += 1;
        *tally.motion_mode_hist.entry(info.motion_mode).or_insert(0) += 1;
        *tally.comp_type_hist.entry(info.comp_type).or_insert(0) += 1;
        *tally.ref0_hist.entry(info.ref0 as i16).or_insert(0) += 1;
        *tally.ref1_hist.entry(info.ref1 as i16).or_insert(0) += 1;
        // MV magnitude — 1/8-pel L2 in scalar form via squared sum.
        let dx = info.mv0_x as i32;
        let dy = info.mv0_y as i32;
        let mag2 = dx * dx + dy * dy;
        let mag = (mag2 as f64).sqrt() as u32;
        tally.reservoir_step(mag);
        // Bin into log2 bucket.
        let bucket = if mag == 0 {
            0u8
        } else {
            (32 - mag.leading_zeros()) as u8
        };
        *tally.mv_mag_hist.entry(bucket).or_insert(0) += 1;
    }
    if info.skip == 1 {
        tally.n_skip += 1;
    }
    if info.skip_mode == 1 {
        tally.n_skip_mode += 1;
    }
    *tally.bs_hist.entry(info.bs).or_insert(0) += 1;
    *tally.bp_hist.entry(info.bp).or_insert(0) += 1;
    *tally.bl_hist.entry(info.bl).or_insert(0) += 1;
    *tally.frame_type_per_block.entry(info.frame_type).or_insert(0) += 1;
    tally.frames_seen.insert(info.frame_offset);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("usage: av1_block_hist <input.obu>");
        std::process::exit(2);
    }
    let path = &args[1];
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("av1_block_hist: read {path}: {e}");
            std::process::exit(1);
        }
    };

    let mut tally = BlockTally::default();
    let eagain = dav1d_err_again();
    let mut frames_decoded: u64 = 0;

    unsafe {
        let mut settings: Dav1dSettings = core::mem::zeroed();
        dav1d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        settings.phasm_hooks = Dav1dPhasmHooks {
            cookie: &mut tally as *mut BlockTally as *mut core::ffi::c_void,
            bit_hook: None,
            tag_hook: None,
            meta_hook: None,
            block_hook: Some(block_hook_cb),
        };

        let mut ctx: *mut Dav1dContext = core::ptr::null_mut();
        let rc = dav1d_open(&mut ctx, &settings);
        if rc != 0 {
            eprintln!("av1_block_hist: dav1d_open failed: {rc}");
            std::process::exit(1);
        }

        let mut data: Dav1dData = core::mem::zeroed();
        let buf = dav1d_data_create(&mut data, bytes.len());
        if buf.is_null() {
            dav1d_close(&mut ctx);
            eprintln!("av1_block_hist: dav1d_data_create failed");
            std::process::exit(1);
        }
        core::ptr::copy_nonoverlapping(bytes.as_ptr(), buf, bytes.len());

        // dav1d_send_data refs our entire buffer into c->in then runs
        // gen_picture which parses OBUs and decodes frames until either
        // (a) c->in is exhausted, or (b) an output picture is ready —
        // whichever comes first. After case (b), send_data unrefs OUR
        // Dav1dData (memset to 0) even though there's more data inside
        // c->in to process. So we can't terminate based on `data.sz`;
        // we must drive get_picture until EAGAIN. dav1d's drain bit
        // (set on first get_picture call) tells it to flush internal
        // buffer + return all remaining decoded frames.
        let send_rc = dav1d_send_data(ctx, &mut data);
        if send_rc != 0 && send_rc != eagain {
            dav1d_data_unref(&mut data);
            dav1d_close(&mut ctx);
            eprintln!("av1_block_hist: dav1d_send_data failed: {send_rc}");
            std::process::exit(1);
        }
        loop {
            let mut pic = Dav1dPicture::default();
            let rc = dav1d_get_picture(ctx, &mut pic);
            if rc == 0 {
                frames_decoded += 1;
                dav1d_picture_unref(&mut pic);
                continue;
            }
            if rc == eagain {
                // Try to push more data — c->in may have been
                // consumed by previous send + decode loop; re-feeding
                // from our buffer fails because we already unref'd
                // (data.sz==0 after unref). If get_picture still says
                // EAGAIN after drain mode, decoding is complete.
                if data.sz == 0 {
                    break;
                }
                let send_rc = dav1d_send_data(ctx, &mut data);
                if send_rc != 0 && send_rc != eagain {
                    dav1d_data_unref(&mut data);
                    dav1d_close(&mut ctx);
                    eprintln!("av1_block_hist: dav1d_send_data (loop) failed: {send_rc}");
                    std::process::exit(1);
                }
                continue;
            }
            dav1d_data_unref(&mut data);
            dav1d_close(&mut ctx);
            eprintln!("av1_block_hist: dav1d_get_picture failed: {rc}");
            std::process::exit(1);
        }

        dav1d_data_unref(&mut data);
        dav1d_close(&mut ctx);
    }

    // ─── TSV output ─────────────────────────────────────────────────
    let stdout = std::io::stdout();
    let mut w = stdout.lock();
    writeln!(w, "# av1_block_hist for {path}").ok();
    writeln!(w, "# n_blocks\t{}", tally.n_blocks).ok();
    writeln!(w, "# n_frames_decoded\t{frames_decoded}").ok();
    writeln!(w, "# n_frame_offsets_distinct\t{}", tally.frames_seen.len()).ok();
    writeln!(w, "# n_intra\t{}", tally.n_intra).ok();
    writeln!(w, "# n_inter\t{}", tally.n_inter).ok();
    writeln!(w, "# n_skip\t{}", tally.n_skip).ok();
    writeln!(w, "# n_skip_mode\t{}", tally.n_skip_mode).ok();
    writeln!(w, "# n_mv_samples_kept\t{}", tally.mv_samples.len()).ok();

    write_hist_u8(&mut w, "BS", &tally.bs_hist);
    write_hist_u8(&mut w, "BP", &tally.bp_hist);
    write_hist_u8(&mut w, "BL", &tally.bl_hist);
    write_hist_u8(&mut w, "Y_MODE", &tally.y_mode_hist);
    write_hist_u8(&mut w, "UV_MODE", &tally.uv_mode_hist);
    write_hist_u8(&mut w, "INTER_MODE", &tally.inter_mode_hist);
    write_hist_u8(&mut w, "MOTION_MODE", &tally.motion_mode_hist);
    write_hist_u8(&mut w, "COMP_TYPE", &tally.comp_type_hist);
    write_hist_i16(&mut w, "REF0", &tally.ref0_hist);
    write_hist_i16(&mut w, "REF1", &tally.ref1_hist);
    write_hist_u8(&mut w, "MV_MAG_LOG2_BUCKET", &tally.mv_mag_hist);
    write_hist_u8(&mut w, "FRAME_TYPE_PER_BLOCK", &tally.frame_type_per_block);

    // Raw MV samples (for KS test downstream).
    writeln!(w, "## MV_SAMPLES_RAW").ok();
    for s in tally.mv_samples.iter() {
        writeln!(w, "{s}").ok();
    }
}

fn write_hist_u8(w: &mut impl Write, name: &str, hist: &BTreeMap<u8, u64>) {
    writeln!(w, "## {name}").ok();
    writeln!(w, "key\tcount").ok();
    for (k, v) in hist.iter() {
        writeln!(w, "{k}\t{v}").ok();
    }
}

fn write_hist_i16(w: &mut impl Write, name: &str, hist: &BTreeMap<i16, u64>) {
    writeln!(w, "## {name}").ok();
    writeln!(w, "key\tcount").ok();
    for (k, v) in hist.iter() {
        writeln!(w, "{k}\t{v}").ok();
    }
}
