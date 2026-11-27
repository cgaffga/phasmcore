// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The streamed Ghost encode: Stage 3 of "Large photos with less memory",
//! the memory ladder's lowest rung (`GhostRung::Streamed`;
//! `docs/design/image/ghost-embedding-v3.md`, "Stage 3").
//!
//! A lane reads, per position, its current value (the overlay's, else the
//! cover's), its pair bit, whether it is wet, its side information and its
//! cover cost ([`v31::LaneInput::at_fi`]). [`v31::plan`] reads them from the
//! luma grid, the keys, the side information and the costs in memory, by flat
//! index through the keyed order. Here they are written to disk once, then
//! read front to back:
//!
//! 1. **Records** — one cost pass over the cover writes an 8 B record per luma
//!    AC position ([`Rec`]) in raster order; the luma and the side
//!    information are dropped after it (the stego image is a fresh parse of
//!    the cover file: `pipeline::stego_image`).
//! 2. **Order** — the keyed order is generated (4 B per position in memory,
//!    the peak) and written to disk (`spill::OrderSpill`): the changes' flat
//!    indices are read from it at the end.
//! 3. **Permutation into lane order**, external, in sequential passes: the
//!    order's inverse in memory (4 B per position, built from the order on
//!    disk), one pass over the records distributing them into buckets of
//!    consecutive lane positions (each record with its place in its bucket),
//!    then each bucket read, put in order in memory and appended to the
//!    lane-order file. The λ and median samples are taken from the buckets.
//! 4. **Lanes** — each lane reads its records front to back,
//!    [`v31::LANE_CHUNK`] at a time, through the chunked trellis input
//!    ([`RecordLane`]); the changes' new values come with them, their flat
//!    indices from the order on disk.
//!
//! The same values reach the same computations in the same order, so the
//! plan — and the stego file — is [`v31::plan`]'s, byte for byte. The files
//! hold numbers derived from the cover photo and the passphrase (the order,
//! per-position costs, coefficient values, pair and wet bits, side
//! information), never the message or the passphrase:
//! owner-only, unlinked right after they are opened, freed when the encode
//! ends whatever the outcome (`spill::SpillFile`). On disk at most 24 B per
//! AC position at once (the order, the records and the distribution file);
//! [`stream_disk_bytes`].

use super::side_info::SideInfo;
use super::v31::{
    self, EncodeOpts, LANE_CHUNK, LaneCaps, LaneSrc, LaneWindow, Meter, Overlay, PairKey, SiTerm, V31Plan, WetMask,
};
use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::DctGrid;
use crate::stego::error::StegoError;
use crate::stego::progress;
use crate::stego::spill::{OrderCursor, OrderSpill, SYNC_BYTES, SpillFile, SpillWriter};
use std::sync::atomic::AtomicU64;
#[cfg(test)]
use std::sync::atomic::Ordering;

/// Bytes of a position's record on disk ([`Rec`]).
pub const REC_BYTES: usize = 8;
/// Bytes of a record in the distribution file: the record and its place in
/// its bucket (`u32`).
pub const DIST_BYTES: usize = REC_BYTES + 4;
/// Most a bucket of the permutation holds in memory while it is put in
/// order: its distribution records and its lane-order records.
pub const BUCKET_BYTES: usize = 24 << 20;
/// The distribution's write buffers together (one per bucket).
pub const DIST_BUFFER_BYTES: usize = 16 << 20;
/// Positions of the order read per chunk while it is inverted (256 KiB).
const INV_CHUNK: usize = 1 << 16;
/// Records of the raster-order file read per chunk while they are
/// distributed (1 MiB).
const FLAT_CHUNK: usize = 1 << 17;

const S_BIT: u8 = 1;
const WET_BIT: u8 = 2;

/// What a lane reads of a position: its cover cost `rho`, its current value
/// `c` (the overlay's, else the cover's), its side-information code (the
/// stored `i8`, 0 without), its pair bit `s` and its wet bit.
#[derive(Clone, Copy, Default, Debug, PartialEq)]
pub(crate) struct Rec {
    rho: f32,
    c: i16,
    si: i8,
    flags: u8,
}

impl Rec {
    #[inline]
    fn put(&self, out: &mut [u8]) {
        out[..4].copy_from_slice(&self.rho.to_le_bytes());
        out[4..6].copy_from_slice(&self.c.to_le_bytes());
        out[6] = self.si as u8;
        out[7] = self.flags;
    }

    #[inline]
    fn get(b: &[u8]) -> Self {
        Self { rho: f32::from_le_bytes([b[0], b[1], b[2], b[3]]), c: i16::from_le_bytes([b[4], b[5]]), si: b[6] as i8, flags: b[7] }
    }

    /// `(value, s, a, b)` exactly as [`v31::LaneInput::at_fi`] computes them
    /// (`si`: the side information's kind, exact or not, if there is any).
    #[inline]
    fn quad(&self, si: Option<bool>) -> (i16, i16, f32, f32) {
        let (c, s) = (self.c, (self.flags & S_BIT) as i16);
        if self.flags & WET_BIT != 0 {
            return (c, s, f32::INFINITY, f32::INFINITY);
        }
        let t = si.map(|exact| SiTerm {
            e: SideInfo::error_of_code(self.si, exact),
            exact,
            tie: SideInfo::tie_of_code(self.si, exact),
        });
        let (rp, rm) = v31::pm_costs(self.rho, c, t);
        let (a, b) = if v31::dir2(c, s) > 0 { (rp, rm) } else { (rm, rp) };
        (c, s, a, b)
    }
}

/// A lane whose records are on disk in lane order (positions `start ..
/// start + len` of the lane-order file).
pub(crate) struct RecordLane<'a> {
    file: &'a SpillFile,
    start: usize,
    len: usize,
    si: Option<bool>,
}

impl LaneSrc for RecordLane<'_> {
    fn len(&self) -> usize {
        self.len
    }

    fn each(&self, j0: usize, j1: usize, mut f: impl FnMut(usize, i16, i16, f32, f32)) {
        let mut buf = [0u8; REC_BYTES * LANE_CHUNK];
        let bytes = &mut buf[..(j1 - j0) * REC_BYTES];
        self.file.read_or_zeros(((self.start + j0) * REC_BYTES) as u64, bytes);
        for (k, r) in bytes.chunks_exact(REC_BYTES).enumerate() {
            let (c, s, a, b) = Rec::get(r).quad(self.si);
            f(j0 + k, c, s, a, b);
        }
    }

    fn at(&self, window: &mut LaneWindow, i: usize) -> (i16, i16, f32, f32) {
        if i < window.lo || i >= window.hi {
            let hi = (i + LANE_CHUNK).min(self.len);
            window.bytes.resize(LANE_CHUNK * REC_BYTES, 0);
            self.file.read_or_zeros(((self.start + i) * REC_BYTES) as u64, &mut window.bytes[..(hi - i) * REC_BYTES]);
            (window.lo, window.hi) = (i, hi);
        }
        let o = (i - window.lo) * REC_BYTES;
        Rec::get(&window.bytes[o..o + REC_BYTES]).quad(self.si)
    }

    fn failed(&self) -> bool {
        self.file.failed()
    }
}

/// Bytes on disk at once of a streamed encode over `n` AC positions: the
/// order (4 B per position), the raster-order records (8 B) and the
/// distribution file (12 B) while the records are distributed; then the
/// order, the distribution file and the lane-order records (8 B).
pub fn stream_disk_bytes(n: usize) -> u64 {
    n as u64 * (4 + REC_BYTES + DIST_BYTES) as u64
}

/// Positions per bucket of the permutation ([`BUCKET_BYTES`]).
pub fn bucket_positions(n: usize) -> usize {
    (BUCKET_BYTES / (DIST_BYTES + REC_BYTES)).clamp(1, n.max(1))
}

/// Records per write buffer of the distribution into `buckets` buckets.
pub fn dist_buffer_records(buckets: usize) -> usize {
    (DIST_BUFFER_BYTES / buckets.max(1) / DIST_BYTES).clamp(1 << 10, 1 << 16)
}

/// Bytes of the raster-order records of one cost strip (written per strip):
/// the strip's AC positions, 8 B each.
pub fn strip_record_bytes(blocks_wide: usize) -> usize {
    crate::stego::cost::uniward::cost_strip_rows(blocks_wide) * blocks_wide * 63 * REC_BYTES
}

/// A cancel check of a long pass.
fn checkpoint() -> Result<(), StegoError> {
    // Tests: a cancel at the n-th check, without the process-wide flag (the
    // other tests' encodes share it).
    #[cfg(test)]
    if tests::cancel_now() {
        return Err(StegoError::Cancelled);
    }
    progress::check_cancelled()
}

/// The raster-order records of `cover`'s luma AC positions (one cost pass).
fn write_records(
    cover: &JpegImage,
    si: Option<&SideInfo>,
    wet: Option<&WetMask>,
    overlay: Option<&Overlay>,
    pairs: &PairKey,
) -> Result<SpillFile, StegoError> {
    use crate::stego::cost::uniward;
    let grid = cover.dct_grid(0);
    let qt = v31::luma_qt(cover)?;
    let mut out = SpillWriter::create("flat")?;
    let mut recs: Vec<u8> = Vec::new();
    uniward::ac_cost_strips(grid, qt, uniward::cost_strip_rows(grid.blocks_wide()), |base, strip| {
        checkpoint()?;
        recs.resize(strip.len() / 64 * 63 * REC_BYTES, 0);
        let fill = |(b, out): (usize, &mut [u8])| {
            let fb = base + b * 64;
            for k in 1..64usize {
                let fi = (fb + k) as u32;
                let wet = wet.is_some_and(|w| w.get(fi));
                let rec = Rec {
                    rho: strip[b * 64 + k],
                    c: v31::value(grid, overlay, fi),
                    si: si.map_or(0, |s| s.code_at(fi as usize)),
                    flags: pairs.get(fi) as u8 | if wet { WET_BIT } else { 0 },
                };
                rec.put(&mut out[(k - 1) * REC_BYTES..k * REC_BYTES]);
            }
        };
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            recs.par_chunks_mut(63 * REC_BYTES).enumerate().for_each(fill);
        }
        #[cfg(not(feature = "parallel"))]
        recs.chunks_mut(63 * REC_BYTES).enumerate().for_each(fill);
        out.push(&recs)
    })?;
    out.finish()
}

/// The records of `flat` (raster order of the `n` AC positions) in the
/// order's sequence (`order`, on disk): the lane-order file, and the
/// records of the positions `samples` (order positions, any order, repeats
/// allowed).
fn permute(flat: SpillFile, order: &OrderSpill, n: usize, samples: &[usize]) -> Result<(SpillFile, Vec<Rec>), StegoError> {
    let bsize = bucket_positions(n);
    // Tests: many small buckets (the memory model never sees this).
    #[cfg(test)]
    let bsize = match tests::BUCKET.load(Ordering::Relaxed) {
        0 => bsize,
        t => t,
    };
    let buckets = n.div_ceil(bsize);
    let dist = {
        // The inverse order: the order position of every raster position.
        let mut inv = vec![0u32; n];
        let mut chunk = vec![0u32; INV_CHUNK.min(n)];
        for k0 in (0..n).step_by(INV_CHUNK) {
            checkpoint()?;
            let k1 = (k0 + INV_CHUNK).min(n);
            let c = &mut chunk[..k1 - k0];
            order.read_checked(k0, c)?;
            for (j, &fi) in c.iter().enumerate() {
                // A damaged read (never from a sound file) is an error, not a panic.
                let a = ac_index_checked(fi, n).ok_or(StegoError::ImageTooLarge)?;
                inv[a] = (k0 + j) as u32;
            }
        }
        drop(chunk);
        distribute(&flat, &inv, n, bsize, buckets)?
    };
    drop(flat);
    gather_buckets(&dist, n, bsize, buckets, samples)
}

/// `v31::ac_index(fi)` if `fi` is an AC flat index of a position below `n`.
fn ac_index_checked(fi: u32, n: usize) -> Option<usize> {
    if fi & 63 == 0 {
        return None;
    }
    Some(v31::ac_index(fi)).filter(|&a| a < n)
}

/// One sequential pass over the raster-order records: each goes, with its
/// place in its bucket, to its bucket's region of the distribution file
/// (bucket `b`: order positions `b · bsize ..`), through a write buffer per
/// bucket.
fn distribute(flat: &SpillFile, inv: &[u32], n: usize, bsize: usize, buckets: usize) -> Result<SpillFile, StegoError> {
    let dist = SpillFile::create("dist")?;
    let per = dist_buffer_records(buckets) * DIST_BYTES;
    let mut bufs: Vec<Vec<u8>> = (0..buckets).map(|_| Vec::with_capacity(per)).collect();
    let mut filled = vec![0u64; buckets];
    let mut unsynced = 0u64;
    let mut flush = |b: usize, buf: &mut Vec<u8>, filled: &mut [u64]| -> Result<(), StegoError> {
        dist.write_at((b * bsize * DIST_BYTES) as u64 + filled[b], buf)?;
        filled[b] += buf.len() as u64;
        unsynced += buf.len() as u64;
        buf.clear();
        if unsynced >= SYNC_BYTES {
            dist.sync()?;
            unsynced = 0;
        }
        Ok(())
    };
    let mut rbuf = vec![0u8; FLAT_CHUNK.min(n) * REC_BYTES];
    for a0 in (0..n).step_by(FLAT_CHUNK) {
        checkpoint()?;
        let a1 = (a0 + FLAT_CHUNK).min(n);
        let r = &mut rbuf[..(a1 - a0) * REC_BYTES];
        flat.read_at((a0 * REC_BYTES) as u64, r)?;
        for (rec, &k) in r.chunks_exact(REC_BYTES).zip(&inv[a0..a1]) {
            let k = k as usize;
            let b = k / bsize;
            let buf = &mut bufs[b];
            buf.extend_from_slice(rec);
            buf.extend_from_slice(&((k - b * bsize) as u32).to_le_bytes());
            if buf.len() >= per {
                flush(b, buf, &mut filled)?;
            }
        }
    }
    for (b, buf) in bufs.iter_mut().enumerate() {
        if !buf.is_empty() {
            flush(b, buf, &mut filled)?;
        }
    }
    // Every bucket exactly full: the inverse was a permutation.
    for (b, &f) in filled.iter().enumerate() {
        if f != (bsize.min(n - b * bsize) * DIST_BYTES) as u64 {
            return Err(StegoError::ImageTooLarge);
        }
    }
    dist.sync()?;
    Ok(dist)
}

/// Each bucket of the distribution file read, its records put in order in
/// memory and appended to the lane-order file; the samples' records taken
/// on the way.
fn gather_buckets(dist: &SpillFile, n: usize, bsize: usize, buckets: usize, samples: &[usize]) -> Result<(SpillFile, Vec<Rec>), StegoError> {
    let mut by_k: Vec<(usize, usize)> = samples.iter().enumerate().map(|(j, &k)| (k, j)).collect();
    by_k.sort_unstable();
    let mut sample_recs = vec![Rec::default(); samples.len()];
    let mut sp = 0;
    let mut lanes = SpillWriter::create("lanes")?;
    let cap = bsize.min(n);
    let mut inb = vec![0u8; cap * DIST_BYTES];
    let mut out = vec![0u8; cap * REC_BYTES];
    for b in 0..buckets {
        checkpoint()?;
        let k0 = b * bsize;
        let len = bsize.min(n - k0);
        let src = &mut inb[..len * DIST_BYTES];
        dist.read_at((k0 * DIST_BYTES) as u64, src)?;
        let dst = &mut out[..len * REC_BYTES];
        for e in src.chunks_exact(DIST_BYTES) {
            let at = u32::from_le_bytes([e[8], e[9], e[10], e[11]]) as usize * REC_BYTES;
            dst.get_mut(at..at + REC_BYTES).ok_or(StegoError::ImageTooLarge)?.copy_from_slice(&e[..REC_BYTES]);
        }
        while sp < by_k.len() && by_k[sp].0 < k0 + len {
            let (k, j) = by_k[sp];
            sample_recs[j] = Rec::get(&dst[(k - k0) * REC_BYTES..]);
            sp += 1;
        }
        lanes.push(dst)?;
    }
    Ok((lanes.finish()?, sample_recs))
}

/// [`v31::plan`] streamed (`EncodeOpts::stream`): the same plan, with the
/// cover's luma and the side information dropped after one cost pass
/// (`cover`'s luma grid is left empty and `si` `None`: the stego image is a
/// fresh parse of the cover, and a caller that plans again parses it again
/// and keeps the side information itself: [`SiOnDisk`]). Needs a spill
/// directory with room for [`stream_disk_bytes`] (`ImageTooLarge`
/// otherwise, and on any I/O error); `Cancelled` on a cancel during the
/// passes.
#[allow(clippy::too_many_arguments)]
pub(crate) fn plan_streamed(
    cover: &mut JpegImage,
    frame_bits: &[u8],
    structural_key: &[u8; 64],
    si: &mut Option<SideInfo>,
    wet: Option<&WetMask>,
    overlay: Option<&Overlay>,
    opts: &EncodeOpts,
) -> Result<V31Plan, StegoError> {
    let t = Clock::start();
    let seeds = v31::v31_seeds(structural_key);
    let grid = cover.dct_grid(0);
    let blocks = grid.blocks_wide() * grid.blocks_tall();
    let n = v31::ac_count(grid);
    let m = frame_bits.len();
    if m == 0 || m > n {
        return Err(StegoError::MessageTooLarge);
    }
    let exact = si.as_ref().map(SideInfo::is_exact);
    let l = v31::lanes(n, m);
    let workers = crate::stego::memory::encode_workers();
    let meter = Meter {
        total: (0..l).map(|k| v31::lane_work(v31::share(n, l, k).len(), v31::share(m, l, k).len()) as u64).sum::<u64>()
            + 2 * v31::gather_work(n),
        steps: opts.progress_steps as u64,
        done: AtomicU64::new(0),
    };

    // 1. The records, in the cover's raster order.
    let flat = {
        // As `v31::plan`: without a mask the dummy blocks are the only wet positions.
        let dummies = if wet.is_none() { v31::dummy_block_wet(cover) } else { None };
        let pairs = PairKey::new(&seeds.pair, grid.coeffs().len());
        write_records(cover, si.as_ref(), wet.or(dummies.as_ref()), overlay, &pairs)?
    };
    *si = None;
    *cover.dct_grid_mut(0) = DctGrid::new(0, 0);
    meter.tick(v31::gather_work(n) as usize);
    t.log("records");

    // 2. The keyed order, on disk.
    let order = {
        let ord = v31::order_of_blocks(blocks, &seeds);
        OrderSpill::write(&ord)?
    };
    t.log("order");

    // 3. The records in lane order, and the samples' records.
    let (samples, n_med) = v31::sample_positions(n);
    let (recs, sample_recs) = permute(flat, &order, n, &samples)?;
    drop(samples);
    meter.tick(v31::gather_work(n) as usize);
    t.log("permutation");
    let (lam, median_cost, sample_ab) = v31::sample_stats_ab(n, n_med, sample_recs.len(), m, |j| {
        let (_, _, a, b) = sample_recs[j].quad(exact);
        (a, b)
    });
    drop(sample_recs);

    // 4. The lanes, reading their records front to back.
    let caps = LaneCaps::of(n, l);
    let mut plans = Vec::with_capacity(l);
    let plan_k = |k: usize, nb: usize| {
        let lr = v31::share(n, l, k);
        let inline = opts.lane_inline_bytes.min(v31::lane_inline_share(n, lr.len(), nb, opts.h, workers));
        let src = RecordLane { file: &recs, start: lr.start, len: lr.len(), si: exact };
        let bits = &frame_bits[v31::share(m, l, k)];
        v31::plan_lane(&src, bits, lam, &seeds, opts, inline, v31::lane_threads(nb, workers), caps, &meter)
    };
    v31::plan_batches(0..l, opts.parallel_lanes.min(workers).clamp(1, l), plan_k, &mut plans)?;
    meter.finish();
    drop(recs);
    t.log("lanes");

    // 5. The changes' flat indices from the order on disk.
    let mut cursor = OrderCursor::new(&order);
    let p = v31::assemble_by(plans, n, |k| cursor.get(k), m, median_cost, &sample_ab, opts.bound);
    if order.failed() {
        return Err(StegoError::ImageTooLarge);
    }
    Ok(p)
}

/// Phase timing of the streamed encode (`PHASM_GHOST_V3_TIMING`), debug
/// builds only: release builds log nothing.
struct Clock {
    #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
    t: Option<std::time::Instant>,
}

impl Clock {
    fn start() -> Self {
        Self {
            #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
            t: std::env::var_os("PHASM_GHOST_V3_TIMING").map(|_| std::time::Instant::now()),
        }
    }

    fn log(&self, _what: &str) {
        #[cfg(all(debug_assertions, not(target_arch = "wasm32")))]
        if let Some(t) = self.t {
            eprintln!("v3stream {_what} {:.3}s", t.elapsed().as_secs_f64());
        }
    }
}

/// The side information on disk while a streamed primary does not hold it:
/// a shadow layout retried at a narrower width plans its shadows with it
/// again (`pipeline::shadows_encode_body`). Its codes, 1 B per luma
/// coefficient.
pub(crate) struct SiOnDisk {
    file: SpillFile,
    len: usize,
    blocks_wide: usize,
    blocks_tall: usize,
    exact: bool,
}

impl SiOnDisk {
    pub(crate) fn write(si: &SideInfo) -> Result<Self, StegoError> {
        let codes = si.codes();
        let mut w = SpillWriter::create("si")?;
        let mut buf = Vec::with_capacity(crate::stego::spill::WRITER_BYTES);
        for chunk in codes.chunks(crate::stego::spill::WRITER_BYTES) {
            checkpoint()?;
            buf.clear();
            buf.extend(chunk.iter().map(|&v| v as u8));
            w.push(&buf)?;
        }
        Ok(Self { file: w.finish()?, len: codes.len(), blocks_wide: si.blocks_wide, blocks_tall: si.blocks_tall, exact: si.is_exact() })
    }

    pub(crate) fn read(&self) -> Result<SideInfo, StegoError> {
        let mut codes: Vec<i8> = Vec::with_capacity(self.len);
        let mut buf = vec![0u8; crate::stego::spill::WRITER_BYTES.min(self.len)];
        let mut at = 0;
        while at < self.len {
            let k = buf.len().min(self.len - at);
            self.file.read_at(at as u64, &mut buf[..k])?;
            codes.extend(buf[..k].iter().map(|&v| v as i8));
            at += k;
        }
        Ok(SideInfo::from_codes(codes, self.blocks_wide, self.blocks_tall, self.exact))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    /// Positions per bucket, when set (tests holding the memory budget's
    /// lock): many small buckets.
    pub(crate) static BUCKET: AtomicUsize = AtomicUsize::new(0);
    /// Cancel at the n-th cancel check from now (0: never).
    pub(crate) static CANCEL_AT: AtomicUsize = AtomicUsize::new(0);

    /// Whether this cancel check is the one [`CANCEL_AT`] names.
    pub(crate) fn cancel_now() -> bool {
        CANCEL_AT.try_update(Ordering::Relaxed, Ordering::Relaxed, |k| (k > 0).then(|| k - 1)) == Ok(1)
    }

    use crate::stego::ghost::v31::tests::synth_jpeg;
    use crate::stego::ghost::v31::{plan, value, wet_mask_for};
    use crate::stego::spill;

    /// Random side-information codes for `grid` (exact: ties ±127 among them).
    fn si_of(grid: &DctGrid, exact: bool, seed: u64) -> SideInfo {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        let codes = (0..grid.coeffs().len()).map(|_| ((rng.gen08::<u8>() % 255) as i16 - 127) as i8).collect();
        SideInfo::from_codes(codes, grid.blocks_wide(), grid.blocks_tall(), exact)
    }

    fn bits(m: usize, seed: u64) -> Vec<u8> {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        (0..m).map(|_| rng.gen08::<u8>() & 1).collect()
    }

    fn same(p: &V31Plan, base: &V31Plan, what: &str) {
        assert_eq!(p.changes, base.changes, "{what}");
        assert_eq!(p.stats.distortion.to_bits(), base.stats.distortion.to_bits(), "{what}");
        assert_eq!(p.stats.median_cost.to_bits(), base.stats.median_cost.to_bits(), "{what}");
        assert_eq!(
            (p.stats.n, p.stats.m, p.stats.lanes, p.stats.m2, p.stats.w1, p.stats.w2, p.stats.changes, p.stats.changes2),
            (base.stats.n, base.stats.m, base.stats.lanes, base.stats.m2, base.stats.w1, base.stats.w2, base.stats.changes, base.stats.changes2),
            "{what}"
        );
    }

    /// The streamed plan equals [`plan`]'s: without and with side information
    /// (approximate, exact with ties), shadow layers' wet positions and
    /// overlay, the dummy blocks of an odd-sized cover, one lane and two
    /// (1600 × 1600), buckets of the default size and many small ones, lanes
    /// at once and checkpointed trellises, one worker; it drops the luma and
    /// the side information and leaves no file behind.
    #[cfg(unix)]
    #[test]
    fn streamed_plan_is_the_same() {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let _guard = crate::stego::memory::budget_test_lock();
        let dir = spill::tests::test_dir("stream");
        spill::set_spill_dir(Some(dir.clone()), spill::tests::plenty);
        let key = [0x6bu8; 64];
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(21);
        let streamed = |o: EncodeOpts| EncodeOpts { stream: true, ..o };
        let cases = [
            (0usize, streamed(EncodeOpts::default())),
            (1000, streamed(EncodeOpts { parallel_lanes: 1, lane_inline_bytes: 0, ..Default::default() })),
            (7777, streamed(EncodeOpts { parallel_lanes: 2, lane_inline_bytes: 1 << 20, ..Default::default() })),
        ];
        // (cover, side-information kinds, frame sizes, with and without shadows)
        type Kinds = &'static [Option<bool>];
        let covers: [(u32, u32, Kinds, &[usize], bool); 3] = [
            (512, 512, &[None, Some(false), Some(true)], &[3_000, 20_000], true),
            (504, 520, &[None, Some(true)], &[3_000, 20_000], true),
            (1600, 1600, &[Some(true)], &[40_000], false),
        ];
        for (w, h, kinds, ms, both) in covers {
            let cover = JpegImage::from_bytes(&synth_jpeg(w, h)).unwrap();
            let grid = cover.dct_grid(0);
            // A shadow-like layer: wet positions with values of their own.
            let mut wet = wet_mask_for(&cover);
            let mut overlay = Overlay::default();
            let blocks = (grid.coeffs().len() / 64) as u32;
            for _ in 0..grid.coeffs().len() / 500 {
                let fi = (rng.gen08::<u32>() % blocks) * 64 + 1 + rng.gen08::<u32>() % 63;
                wet.set(fi);
                overlay.set(fi, value(grid, None, fi) + 1);
            }
            let layers: &[(Option<&WetMask>, Option<&Overlay>)] =
                if both { &[(None, None), (Some(&wet), Some(&overlay))] } else { &[(Some(&wet), Some(&overlay))] };
            for &kind in kinds {
                let si = kind.map(|exact| si_of(grid, exact, 5));
                for &m in ms {
                    let bits = bits(m, m as u64);
                    for &(wt, ov) in layers {
                        let base = plan(&cover, &bits, &key, si.as_ref(), wt, ov, &EncodeOpts::default()).unwrap();
                        if w == 1600 {
                            assert_eq!(base.stats.lanes, 2);
                        }
                        for (bucket, opts) in &cases {
                            BUCKET.store(*bucket, Ordering::Relaxed);
                            let (mut c, mut s) = (cover.clone(), kind.map(|exact| si_of(grid, exact, 5)));
                            let p = plan_streamed(&mut c, &bits, &key, &mut s, wt, ov, opts).unwrap();
                            let what = format!("{w}x{h} si {kind:?} m={m} shadows={} bucket {bucket} {opts:?}", wt.is_some());
                            same(&p, &base, &what);
                            assert!(c.dct_grid(0).coeffs().is_empty() && s.is_none(), "{what}: luma and side information dropped");
                        }
                        BUCKET.store(0, Ordering::Relaxed);
                        #[cfg(feature = "parallel")]
                        {
                            let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
                            let (mut c, mut s) = (cover.clone(), kind.map(|exact| si_of(grid, exact, 5)));
                            if w != 1600 {
                                let p = one.install(|| plan_streamed(&mut c, &bits, &key, &mut s, wt, ov, &cases[1].1)).unwrap();
                                same(&p, &base, "one worker");
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "no spill file left");
        spill::set_spill_dir(None, spill::tests::plenty);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A read or write that fails at any point of the streamed plan (the
    /// records, the order, the permutation, the lanes, the assembly), a
    /// cancel at any of its checks and a missing spill directory end it with
    /// a clean error — `ImageTooLarge`, `Cancelled` — never a panic or a
    /// different plan, and leave no file behind.
    #[cfg(unix)]
    #[test]
    fn streamed_plan_fails_cleanly() {
        let _guard = crate::stego::memory::budget_test_lock();
        let dir = spill::tests::test_dir("stream-fail");
        spill::set_spill_dir(Some(dir.clone()), spill::tests::plenty);
        let cover = JpegImage::from_bytes(&synth_jpeg(640, 640)).unwrap();
        let key = [0x17u8; 64];
        let bits = bits(20_000, 3);
        let base = plan(&cover, &bits, &key, None, None, None, &EncodeOpts::default()).unwrap();
        let opts = EncodeOpts { stream: true, ..Default::default() };
        BUCKET.store(5_000, Ordering::Relaxed);
        let run = || {
            let mut c = cover.clone();
            plan_streamed(&mut c, &bits, &key, &mut None, None, None, &opts)
        };
        // The k-th read / write / cancel check from the start, k on a
        // geometric ladder up to past the last one (then the plan succeeds).
        let ladder = || {
            let mut k = 1usize;
            std::iter::from_fn(move || {
                let cur = k;
                k += k / 3 + 1;
                Some(cur)
            })
        };
        for (what, set) in [
            ("read", &spill::tests::FAIL_READ_AT as &std::sync::atomic::AtomicUsize),
            ("write", &spill::tests::FAIL_WRITE_AT),
            ("cancel", &CANCEL_AT),
        ] {
            let mut failures = 0;
            for k in ladder() {
                set.store(k, Ordering::Relaxed);
                let r = run();
                spill::tests::clear_faults();
                CANCEL_AT.store(0, Ordering::Relaxed);
                assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0, "{what} {k}: no file left");
                match r {
                    Ok(p) => {
                        same(&p, &base, what);
                        break;
                    }
                    Err(e) => {
                        let want = if what == "cancel" { StegoError::Cancelled } else { StegoError::ImageTooLarge };
                        assert_eq!(std::mem::discriminant(&e), std::mem::discriminant(&want), "{what} {k}: {e:?}");
                        failures += 1;
                    }
                }
            }
            assert!(failures >= 10, "{what}: {failures} failure points");
        }
        BUCKET.store(0, Ordering::Relaxed);
        // No spill directory: the streamed plan cannot write its files.
        spill::set_spill_dir(None, spill::tests::plenty);
        assert!(matches!(run(), Err(StegoError::ImageTooLarge)));
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The side information's copy on disk reads back as it was.
    #[cfg(unix)]
    #[test]
    fn side_information_round_trips() {
        let _guard = crate::stego::memory::budget_test_lock();
        let dir = spill::tests::test_dir("si");
        spill::set_spill_dir(Some(dir.clone()), spill::tests::plenty);
        let grid = DctGrid::new(301, 77);
        for exact in [false, true] {
            let si = si_of(&grid, exact, 9);
            let back = SiOnDisk::write(&si).unwrap().read().unwrap();
            assert_eq!(back.codes(), si.codes());
            assert_eq!((back.blocks_wide, back.blocks_tall, back.is_exact()), (301, 77, exact));
        }
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 0);
        spill::set_spill_dir(None, spill::tests::plenty);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn record_round_trips() {
        for rec in [
            Rec { rho: 1.5e-3, c: -1023, si: -127, flags: S_BIT | WET_BIT },
            Rec { rho: f32::INFINITY, c: 32767, si: 126, flags: 0 },
            Rec { rho: 0.0, c: 0, si: 0, flags: S_BIT },
        ] {
            let mut b = [0u8; REC_BYTES];
            rec.put(&mut b);
            assert_eq!(Rec::get(&b), rec);
        }
    }
}
