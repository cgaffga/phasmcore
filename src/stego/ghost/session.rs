// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The Ghost v3 primary encode split into tasks, for hosts whose threads do
//! not share memory: the website runs it in Web Workers, each with its own
//! WASM instance (no `SharedArrayBuffer`, so no COOP/COEP headers).
//!
//! A [`GhostEncodeSession`] (the coordinator) holds the cover, the frame, the
//! keyed order and the costs; it hands out self-contained tasks (byte
//! strings) and takes their results back. A [`GhostTaskRunner`] on each
//! helper runs tasks:
//!
//! - **cost band**: the J-UNIWARD costs of a band of block rows, from the
//!   band's sub-grid (`uniward::ac_cost_band`);
//! - **trellis**: one layer of one lane (layer 2, then layer 1 once the
//!   coordinator has the layer-2 flips), either schedule;
//! - **pass A / segment / trace**: a layer too large for the in-memory
//!   schedule, while there are fewer lanes than helpers: pass A on one
//!   helper, then the segments' back pointers recomputed on all helpers
//!   (each keeps its rows) and traced from the last segment to the first.
//!   That walks each layer `1 + 1/helpers` times instead of twice.
//!
//! The coordinator computes the keyed order while the helpers compute the
//! costs. The output is byte-identical to `ghost_encode_with_files` /
//! `ghost_encode_si_with_files` for any number of helpers: the same parse,
//! frame, order, costs (bit-identical by band), λ, lane inputs and
//! assembly (`v3.rs`), and trellises planned by the same code
//! (`stc::trellis`, whose split schedule equals `embed`).

use std::collections::{BTreeMap, HashMap};

use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::{DctGrid, QuantTable};
use crate::stego::cost::uniward;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::payload::FileEntry;
use crate::stego::quality::EncodeQuality;
use crate::stego::side_info::SideInfo;
use crate::stego::stc::trellis::{self, PassA};

use super::pipeline::{self, PreparedEncode};
use super::v31::{self, LaneCache, LaneGeom, LaneInput, LanePlan, Rho, V31Seeds};
use crate::stego::stc::trellis::Code;

const TASK_COST_BAND: u8 = 1;
const TASK_TRELLIS: u8 = 2;
const TASK_PASS_A: u8 = 3;
const TASK_SEGMENT: u8 = 4;
const TASK_TRACE: u8 = 5;

/// Bytes of the header every task and result starts with: kind (u8), lane
/// (u32), layer (u8), segment (u32), target helper (i32, −1 = any).
pub const TASK_HEADER_BYTES: usize = 14;

/// Byte offset of the target helper in a task (a trace must run on the
/// helper that keeps its segment's back pointers).
pub const TASK_TARGET_OFFSET: usize = 10;

/// Positions per segment of a split trellis (16 MB of back pointers at
/// h = 9; a segment is at least one message block).
const SEGMENT_POSITIONS: usize = 1 << 18;

/// A split layer whose trellis input (5 B per position) is larger than this
/// runs its pass A in the coordinator, which already holds the input and is
/// idle meanwhile, instead of copying it to a helper (twice: the task and
/// its parse) — 236 MB per copy for a short message on a 48 MP cover.
const LOCAL_PASS_A_BYTES: usize = 64 << 20;

/// Lanes up to this many positions get a `LaneCache` (10 B per position:
/// 118 MB for a single-lane message on 12 MP); longer ones read the grid
/// and costs directly, as the single-thread encoder does.
const LANE_CACHE_MAX_POSITIONS: usize = 16 << 20;

/// Smallest cost band (block rows): margins are 2 rows on either side.
const MIN_BAND_ROWS: usize = 8;

#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Debug)]
struct Header {
    kind: u8,
    lane: u32,
    layer: u8,
    seg: u32,
    target: i32,
}

// ---------------------------------------------------------------------------
// Byte encoding (little-endian)
// ---------------------------------------------------------------------------

struct W(Vec<u8>);

impl W {
    fn new(hd: Header, capacity: usize) -> Self {
        let mut w = W(Vec::with_capacity(TASK_HEADER_BYTES + capacity));
        w.u8(hd.kind);
        w.u32(hd.lane);
        w.u8(hd.layer);
        w.u32(hd.seg);
        w.0.extend_from_slice(&hd.target.to_le_bytes());
        w
    }
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.0.extend_from_slice(v);
    }
    fn u32s(&mut self, v: &[u32]) {
        for x in v {
            self.u32(*x);
        }
    }
    fn f32s(&mut self, v: &[f32]) {
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    fn f64s(&mut self, v: &[f64]) {
        for x in v {
            self.f64(*x);
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

/// The layer byte of a task.
fn act_layer(task: &[u8]) -> u8 {
    task[5]
}

fn bad() -> StegoError {
    StegoError::InvalidJpeg(crate::codec::jpeg::error::JpegError::InvalidMarkerData("ghost task"))
}

impl<'a> R<'a> {
    fn new(b: &'a [u8]) -> Result<(Header, Self), StegoError> {
        let mut r = R { b, p: 0 };
        let kind = r.u8()?;
        let lane = r.u32()?;
        let layer = r.u8()?;
        let seg = r.u32()?;
        let target = r.u32()? as i32;
        Ok((Header { kind, lane, layer, seg, target }, r))
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], StegoError> {
        let end = self.p.checked_add(n).filter(|&e| e <= self.b.len()).ok_or_else(bad)?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, StegoError> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, StegoError> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, StegoError> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, StegoError> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn len(&mut self, unit: usize) -> Result<usize, StegoError> {
        let n = self.u32()? as usize;
        if n.saturating_mul(unit) > self.b.len() - self.p {
            return Err(bad());
        }
        Ok(n)
    }
    fn u32s(&mut self, n: usize) -> Result<Vec<u32>, StegoError> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, StegoError> {
        Ok(self.take(n * 4)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn f64s(&mut self, n: usize) -> Result<Vec<f64>, StegoError> {
        Ok(self.take(n * 8)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    }
    fn i16s(&mut self, n: usize) -> Result<Vec<i16>, StegoError> {
        Ok(self.take(n * 2)?.chunks_exact(2).map(|c| i16::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

fn pack_bits(bits: &[u8]) -> Vec<u8> {
    let mut out = vec![0u8; bits.len().div_ceil(8)];
    for (i, &b) in bits.iter().enumerate() {
        out[i >> 3] |= (b & 1) << (i & 7);
    }
    out
}

fn unpack_bits(packed: &[u8], out: &mut [u8]) {
    for (i, o) in out.iter_mut().enumerate() {
        *o = (packed[i >> 3] >> (i & 7)) & 1;
    }
}

/// A layer's trellis input: blocks `[b0, b0 + msg.len())` of `code`, their
/// positions' cover bits and costs.
struct LayerInput<'a> {
    code: &'a Code,
    b0: usize,
    msg: &'a [u8],
    cover: &'a [u8],
    costs: &'a [f32],
}

fn write_code(w: &mut W, code: &Code) {
    let (head, _) = code.head();
    w.u32(code.m() as u32);
    w.u32(head as u32);
    w.u64(code.rest_n() as u64);
    for set in code.column_sets() {
        w.u32(set.len() as u32);
        w.u32s(set);
    }
}

fn read_code(r: &mut R<'_>) -> Result<Code, StegoError> {
    let m = r.u32()? as usize;
    let head = r.u32()? as usize;
    let rest_n = r.u64()? as usize;
    let mut sets = Vec::with_capacity(3);
    for _ in 0..3 {
        let len = r.len(4)?;
        sets.push(r.u32s(len)?);
    }
    let long = sets.pop().ok_or_else(bad)?;
    let short = sets.pop().ok_or_else(bad)?;
    let head_cols = sets.pop().ok_or_else(bad)?;
    Code::new(m, head, head_cols, rest_n, short, long).ok_or_else(bad)
}

fn write_layer(w: &mut W, li: &LayerInput<'_>) {
    write_code(w, li.code);
    w.u32(li.b0 as u32);
    w.u32(li.msg.len() as u32);
    w.bytes(li.msg);
    let n = li.code.start(li.b0 + li.msg.len()) - li.code.start(li.b0);
    w.bytes(&li.cover[..n]);
    w.f32s(&li.costs[..n]);
}

struct OwnedLayer {
    code: Code,
    b0: usize,
    msg: Vec<u8>,
    cover: Vec<u8>,
    costs: Vec<f32>,
}

fn read_layer(r: &mut R<'_>) -> Result<OwnedLayer, StegoError> {
    let code = read_code(r)?;
    let b0 = r.u32()? as usize;
    let m = r.len(1)?;
    if b0.checked_add(m).is_none_or(|e| e > code.m()) {
        return Err(bad());
    }
    let msg = r.take(m)?.to_vec();
    let n = code.start(b0 + m) - code.start(b0);
    let cover = r.take(n)?.to_vec();
    let costs = r.f32s(n)?;
    Ok(OwnedLayer { code, b0, msg, cover, costs })
}

// ---------------------------------------------------------------------------
// Helper side
// ---------------------------------------------------------------------------

/// A segment's back pointers kept on its helper until its trace.
struct Held {
    rows: Vec<u64>,
    msg: Vec<u8>,
    code: Code,
    b0: usize,
    h: usize,
}

/// Runs tasks on a helper. Keeps the back pointers of the segments it
/// computed until their trace task arrives (one runner per helper).
#[derive(Default)]
pub struct GhostTaskRunner {
    held: HashMap<(u32, u8, u32), Held>,
}

impl GhostTaskRunner {
    pub fn new() -> Self {
        Self::default()
    }

    /// Run one task; returns its result. Takes the task by value: its
    /// buffer is freed once parsed, before the work allocates.
    pub fn run(&mut self, task: Vec<u8>) -> Result<Vec<u8>, StegoError> {
        let (hd, mut r) = R::new(&task)?;
        let reply = Header { target: -1, ..hd };
        match hd.kind {
            TASK_COST_BAND => {
                let bw = r.u32()? as usize;
                let bt = r.u32()? as usize;
                let br0 = r.u32()? as usize;
                let br1 = r.u32()? as usize;
                let mut qv = [0u16; 64];
                for q in qv.iter_mut() {
                    *q = u16::from_le_bytes(r.take(2)?.try_into().unwrap());
                }
                let g0 = br0.saturating_sub(uniward::COST_BAND_MARGIN_ROWS);
                let g1 = (br1 + uniward::COST_BAND_MARGIN_ROWS).min(bt);
                if br0 >= br1 || br1 > bt {
                    return Err(bad());
                }
                let coeffs = r.i16s((g1 - g0) * bw * 64)?;
                drop(task);
                let mut sub = DctGrid::new(bw, g1 - g0);
                sub.coeffs_mut().copy_from_slice(&coeffs);
                drop(coeffs);
                let costs = uniward::ac_cost_band(&sub, &QuantTable::new(qv), br0, br1, bt)?;
                let mut w = W::new(reply, costs.len() * 4);
                w.f32s(&costs);
                Ok(w.0)
            }
            TASK_TRELLIS => {
                let h = r.u8()? as usize;
                let limit = r.u64()?;
                let li = read_layer(&mut r)?;
                drop(task);
                let plan = trellis::embed_code(&li.cover, &li.costs, &li.msg, &li.code, h, limit as usize, 1, &|_| {})?;
                let mut w = W::new(reply, li.cover.len() / 8 + 16);
                match plan {
                    Some(p) => {
                        w.u8(1);
                        w.bytes(&pack_bits(&p.stego_bits));
                    }
                    None => w.u8(0),
                }
                Ok(w.0)
            }
            TASK_PASS_A => {
                let h = r.u8()? as usize;
                let k = r.u32()? as usize;
                let li = read_layer(&mut r)?;
                drop(task);
                let pa = trellis::pass_a(&li.cover, &li.costs, &li.msg, &li.code, h, k, &|_| {})?;
                let mut w = W::new(reply, pa.as_ref().map_or(0, |p| p.checkpoints.len() * 8) + 32);
                match pa {
                    Some(p) => {
                        w.u8(1);
                        w.u32(p.best as u32);
                        w.f64(p.cost);
                        w.u32(p.checkpoints.len() as u32);
                        w.f64s(&p.checkpoints);
                    }
                    None => w.u8(0),
                }
                Ok(w.0)
            }
            TASK_SEGMENT => {
                let h = r.u8()? as usize;
                let li = read_layer(&mut r)?;
                let ns = 1usize << h.min(trellis::MAX_H);
                let checkpoint = r.f64s(ns)?;
                drop(task);
                let rows = trellis::segment_rows(&li.cover, &li.costs, &li.msg, &li.code, li.b0, h, &checkpoint)?;
                self.held.insert((hd.lane, hd.layer, hd.seg), Held { rows, msg: li.msg, code: li.code, b0: li.b0, h });
                Ok(W::new(reply, 0).0)
            }
            TASK_TRACE => {
                let state = r.u32()? as usize;
                drop(task);
                let held = self.held.remove(&(hd.lane, hd.layer, hd.seg)).ok_or_else(bad)?;
                let mut bits = vec![0u8; held.code.start(held.b0 + held.msg.len()) - held.code.start(held.b0)];
                let start = trellis::segment_trace(&held.rows, &held.msg, &held.code, held.b0, held.h, state, &mut bits);
                let mut w = W::new(reply, bits.len() / 8 + 8);
                w.u32(start as u32);
                w.bytes(&pack_bits(&bits));
                Ok(w.0)
            }
            _ => Err(bad()),
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinator side
// ---------------------------------------------------------------------------

/// A split layer in progress.
struct Split {
    k: usize,
    pass_a: PassA,
    /// Next segment to hand out (from the last one down); `None` when all are out.
    next_issue: Option<usize>,
    /// Segments computed, by segment: the helper keeping their rows.
    ready: BTreeMap<usize, i32>,
    /// The last segment (where the trace chain starts).
    trace_seg_end: usize,
    /// Next segment to trace (from the last one down) and its end state.
    trace_seg: usize,
    trace_state: usize,
    /// A trace task is out.
    tracing: bool,
}

enum LayerRun {
    /// Input built, task not yet handed out.
    Pending,
    /// One task out (trellis or pass A).
    Out,
    Split(Box<Split>),
}

struct ActiveLayer {
    layer: u8,
    code: Code,
    cover: Vec<u8>,
    costs: Vec<f32>,
    /// Stego bits (filled by the trellis result or the traces).
    stego: Vec<u8>,
    run: LayerRun,
    /// Back-pointer bytes its trellis task may keep (as `v31::plan` shares
    /// the large-frame working set among the lanes in flight).
    limit: usize,
    /// Run as pass A + segments.
    split: bool,
}

struct LaneState {
    geom: LaneGeom,
    cache: Option<LaneCache>,
    /// Layer 2's flips (all false before layer 2 is planned, or without it).
    flip2: Vec<bool>,
    /// The layer being planned.
    active: Option<ActiveLayer>,
}

struct Lanes {
    l: usize,
    lam: f64,
    median_cost: f32,
    sample_ab: Vec<(f64, f64)>,
    states: Vec<Option<LaneState>>,
    plans: Vec<Option<LanePlan>>,
    next_lane: usize,
    left: usize,
}

/// The coordinator of a split Ghost v3 encode (see the module docs).
pub struct GhostEncodeSession {
    img: JpegImage,
    frame_bits: Vec<u8>,
    si: Option<SideInfo>,
    seeds: V31Seeds,
    /// v3.1 pair bits of the cover.
    pairs: v31::PairKey,
    h: usize,
    /// Back-pointer bytes any lane trellis may keep (the memory plan's).
    inline_limit: usize,
    helpers: usize,
    bound: bool,
    costs: Vec<f32>,
    /// The dummy (MCU-padding) blocks, never changed (`v31::dummy_block_wet`).
    wet: Option<v31::WetMask>,
    bands: Vec<(usize, usize)>,
    next_band: usize,
    bands_left: usize,
    ord: Option<Vec<u32>>,
    lanes: Option<Lanes>,
    lane_tasks_out: usize,
    work_total: u64,
    work_done: u64,
    stats: Option<v31::V31Stats>,
    /// Input size above which a split layer's pass A runs here.
    local_pass_a_bytes: usize,
    /// Pass A runs done in the coordinator (tests).
    local_pass_a_count: usize,
}

impl GhostEncodeSession {
    /// Start an encode of `message` (+ `files`) into `image_bytes`, with
    /// Deep Cover side information when `pixels` (sRGB, 3 bytes per pixel,
    /// `width × height`) are given — exactly `ghost_encode_with_files` /
    /// `ghost_encode_si_with_files`. `helpers`: how many helpers will run
    /// tasks (sizes the bands and decides when to split a trellis).
    pub fn new(
        image_bytes: &[u8],
        message: &str,
        files: &[FileEntry],
        passphrase: &str,
        pixels: Option<(&[u8], u32, u32)>,
        helpers: usize,
    ) -> Result<Self, StegoError> {
        let (pre, si) = match pixels {
            Some((rgb, w, h)) => {
                let (img, si) = pipeline::parse_with_side_info(image_bytes, rgb, w, h)?;
                (Some(img), si)
            }
            None => (None, None),
        };
        let PreparedEncode { img, frame_bits, opts, .. } =
            pipeline::prepare_encode(image_bytes, message, files, passphrase, si.is_some(), pre)?;
        let key = crypto::derive_structural_key(passphrase)?;
        let seeds = v31::v31_seeds(&key);
        let grid = img.dct_grid(0);
        let (bw, bt) = (grid.blocks_wide(), grid.blocks_tall());
        let n = v31::ac_count(grid);
        let m = frame_bits.len();
        if m == 0 || m > n {
            return Err(StegoError::MessageTooLarge);
        }
        let helpers = helpers.max(1);
        let nb = (2 * helpers).min(bt.div_ceil(MIN_BAND_ROWS)).max(1);
        let bands: Vec<(usize, usize)> = (0..nb).map(|i| (i * bt / nb, (i + 1) * bt / nb)).filter(|(a, b)| a < b).collect();
        let l = v31::lanes(n, m);
        let work_total = (n / 4) as u64
            + (0..l).map(|k| lane_work(v31::share(n, l, k).len(), v31::share(m, l, k).len()) as u64).sum::<u64>();
        let wet = v31::dummy_block_wet(&img);
        let pairs = v31::PairKey::new(&seeds.pair, grid.coeffs().len());
        Ok(Self {
            pairs,
            inline_limit: opts.lane_inline_bytes,
            h: opts.h,
            bound: opts.bound,
            img,
            frame_bits,
            si,
            seeds,
            helpers,
            costs: vec![f32::INFINITY; bw * bt * 64],
            wet,
            bands_left: bands.len(),
            bands,
            next_band: 0,
            ord: None,
            lanes: None,
            lane_tasks_out: 0,
            work_total,
            work_done: 0,
            stats: None,
            local_pass_a_bytes: LOCAL_PASS_A_BYTES,
            local_pass_a_count: 0,
        })
    }

    /// Fraction of the work done (cost pass and trellises), 0..1.
    pub fn progress(&self) -> f64 {
        if self.work_total == 0 { 0.0 } else { (self.work_done as f64 / self.work_total as f64).min(1.0) }
    }

    /// All lanes planned and applied: [`Self::finish`] can write the JPEG.
    pub fn is_done(&self) -> bool {
        self.stats.is_some()
    }

    /// The coordinator's own work, best run right after the first
    /// [`Self::take_tasks`] (while the helpers compute the costs): the keyed
    /// order. Done on demand otherwise.
    pub fn run_local(&mut self) -> Result<(), StegoError> {
        if self.ord.is_none() {
            self.ord = Some(v31::order(self.img.dct_grid(0), &self.seeds));
        }
        self.maybe_start_lanes()
    }

    /// Tasks ready to run, any helper unless the task's target says
    /// otherwise (bytes `TASK_TARGET_OFFSET..+4`, i32). Every cost band at
    /// once; trellis work at most one task per helper at a time.
    pub fn take_tasks(&mut self) -> Result<Vec<Vec<u8>>, StegoError> {
        let mut out = Vec::new();
        while self.next_band < self.bands.len() {
            out.push(self.band_task(self.next_band));
            self.next_band += 1;
        }
        while self.lane_tasks_out < self.helpers {
            match self.next_lane_task()? {
                Some(t) => {
                    self.lane_tasks_out += 1;
                    out.push(t);
                }
                None => break,
            }
        }
        Ok(out)
    }

    /// A helper's result. `helper`: the index of the helper that ran it
    /// (segments stay on it for their trace).
    pub fn submit(&mut self, result: &[u8], helper: i32) -> Result<(), StegoError> {
        let (hd, mut r) = R::new(result)?;
        match hd.kind {
            TASK_COST_BAND => {
                let (br0, br1) = *self.bands.get(hd.seg as usize).ok_or_else(bad)?;
                let bw = self.img.dct_grid(0).blocks_wide();
                let len = (br1 - br0) * bw * 64;
                let costs = r.f32s(len)?;
                self.costs[br0 * bw * 64..br1 * bw * 64].copy_from_slice(&costs);
                self.bands_left -= 1;
                let bt = self.img.dct_grid(0).blocks_tall();
                self.work_done += (v31::ac_count(self.img.dct_grid(0)) / 4 * (br1 - br0) / bt) as u64;
                if self.bands_left == 0 && self.ord.is_none() {
                    self.run_local()?;
                }
                self.maybe_start_lanes()
            }
            TASK_TRELLIS => {
                self.lane_tasks_out = self.lane_tasks_out.saturating_sub(1);
                let k = hd.lane as usize;
                let act = self.active_mut(k, hd.layer)?;
                if r.u8()? == 0 {
                    return Err(StegoError::MessageTooLarge);
                }
                let n = act.cover.len();
                let packed = r.take(n.div_ceil(8))?;
                unpack_bits(packed, &mut act.stego);
                self.work_done += n as u64;
                self.layer_done(k)
            }
            TASK_PASS_A => {
                self.lane_tasks_out = self.lane_tasks_out.saturating_sub(1);
                let k = hd.lane as usize;
                let act = self.active_mut(k, hd.layer)?;
                if r.u8()? == 0 {
                    return Err(StegoError::MessageTooLarge);
                }
                let best = r.u32()? as usize;
                let cost = r.f64()?;
                let nck = r.len(8)?;
                let checkpoints = r.f64s(nck)?;
                let m = act.code.m();
                let kb = segment_blocks(&act.code);
                let done = (act.cover.len() / 2) as u64;
                self.start_split(k, hd.layer, m, kb, PassA { checkpoints, best, cost }, done)
            }
            TASK_SEGMENT => {
                self.lane_tasks_out = self.lane_tasks_out.saturating_sub(1);
                let act = self.active_mut(hd.lane as usize, hd.layer)?;
                let LayerRun::Split(sp) = &mut act.run else { return Err(bad()) };
                sp.ready.insert(hd.seg as usize, helper);
                Ok(())
            }
            TASK_TRACE => {
                self.lane_tasks_out = self.lane_tasks_out.saturating_sub(1);
                let k = hd.lane as usize;
                let act = self.active_mut(k, hd.layer)?;
                let m = act.code.m();
                let LayerRun::Split(sp) = &mut act.run else { return Err(bad()) };
                let seg = hd.seg as usize;
                if seg != sp.trace_seg || !sp.tracing {
                    return Err(bad());
                }
                let start = r.u32()? as usize;
                let (b0, b1) = (seg * sp.k, ((seg + 1) * sp.k).min(m));
                let (p0, p1) = (act.code.start(b0), act.code.start(b1));
                let packed = r.take((p1 - p0).div_ceil(8))?;
                unpack_bits(packed, &mut act.stego[p0..p1]);
                sp.tracing = false;
                sp.trace_state = start;
                let done = (p1 - p0) as u64 / 2;
                if seg == 0 {
                    if start != 0 {
                        return Err(bad());
                    }
                    self.work_done += done;
                    return self.layer_done(k);
                }
                sp.trace_seg = seg - 1;
                self.work_done += done;
                Ok(())
            }
            _ => Err(bad()),
        }
    }

    /// Write the stego JPEG (after [`Self::is_done`]).
    pub fn finish(mut self) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
        let stats = self.stats.take().ok_or_else(bad)?;
        let is_si = self.si.is_some();
        drop(self.si.take());
        drop(std::mem::take(&mut self.costs));
        pipeline::finish_encode(&mut self.img, &stats, is_si)
    }

    // -- internals ----------------------------------------------------------

    fn band_task(&self, i: usize) -> Vec<u8> {
        let grid = self.img.dct_grid(0);
        let (bw, bt) = (grid.blocks_wide(), grid.blocks_tall());
        let (br0, br1) = self.bands[i];
        let g0 = br0.saturating_sub(uniward::COST_BAND_MARGIN_ROWS);
        let g1 = (br1 + uniward::COST_BAND_MARGIN_ROWS).min(bt);
        let coeffs = &grid.coeffs()[g0 * bw * 64..g1 * bw * 64];
        let hd = Header { kind: TASK_COST_BAND, lane: 0, layer: 0, seg: i as u32, target: -1 };
        let mut w = W::new(hd, 16 + 128 + coeffs.len() * 2);
        w.u32(bw as u32);
        w.u32(bt as u32);
        w.u32(br0 as u32);
        w.u32(br1 as u32);
        let qt = v31::luma_qt(&self.img).expect("checked at parse");
        for q in qt.values {
            w.0.extend_from_slice(&q.to_le_bytes());
        }
        for c in coeffs {
            w.0.extend_from_slice(&c.to_le_bytes());
        }
        w.0
    }

    fn maybe_start_lanes(&mut self) -> Result<(), StegoError> {
        if self.lanes.is_some() || self.bands_left > 0 {
            return Ok(());
        }
        let Some(ord) = self.ord.as_ref() else { return Ok(()) };
        let grid = self.img.dct_grid(0);
        let n = ord.len();
        let m = self.frame_bits.len();
        let (samples, n_med) = v31::sample_positions(n);
        let sample_rho: Vec<f32> = samples.iter().map(|&k| self.costs[ord[k] as usize]).collect();
        let (lam, median_cost, sample_ab) =
            v31::sample_stats(ord, &samples, n_med, &sample_rho, m, grid, None, self.wet.as_ref(), self.si.as_ref(), &self.pairs);
        let l = v31::lanes(n, m);
        self.lanes = Some(Lanes {
            l,
            lam,
            median_cost,
            sample_ab,
            states: (0..l).map(|_| None).collect(),
            plans: (0..l).map(|_| None).collect(),
            next_lane: 0,
            left: l,
        });
        Ok(())
    }

    fn lane_input<'a>(&'a self, k: usize, cache: Option<&'a LaneCache>) -> LaneInput<'a> {
        let ord = self.ord.as_ref().expect("lanes start after the order");
        let lanes = self.lanes.as_ref().expect("lanes started");
        let lr = v31::share(ord.len(), lanes.l, k);
        LaneInput {
            grid: self.img.dct_grid(0),
            overlay: None,
            wet: self.wet.as_ref(),
            si: self.si.as_ref(),
            ord: &ord[lr],
            rho: Rho::Flat(&self.costs),
            pairs: &self.pairs,
            cache,
        }
    }

    fn lane_bits(&self, k: usize) -> &[u8] {
        let l = self.lanes.as_ref().expect("lanes started").l;
        &self.frame_bits[v31::share(self.frame_bits.len(), l, k)]
    }

    fn active_mut(&mut self, k: usize, layer: u8) -> Result<&mut ActiveLayer, StegoError> {
        let lanes = self.lanes.as_mut().ok_or_else(bad)?;
        let act = lanes.states.get_mut(k).and_then(|s| s.as_mut()).and_then(|s| s.active.as_mut()).ok_or_else(bad)?;
        if act.layer != layer {
            return Err(bad());
        }
        Ok(act)
    }

    /// Build a lane layer's input (layer 2 first, unless the lane has none).
    fn build_layer(&self, k: usize, geom: &LaneGeom, layer: u8, flip2: &[bool], cache: Option<&LaneCache>) -> ActiveLayer {
        let inp = self.lane_input(k, cache);
        let (cover, costs) = if layer == 2 {
            let lam = self.lanes.as_ref().expect("lanes started").lam;
            v31::layer2_input(&inp, geom, lam)
        } else {
            v31::layer1_input(&inp, geom, flip2)
        };
        let n = cover.len();
        let code = v31::layer_code(&self.seeds, layer, geom, self.h);
        let lanes = self.lanes.as_ref().expect("lanes started");
        let unfinished = lanes.left;
        let total = self.ord.as_ref().expect("order").len();
        let in_flight = self.helpers.min(lanes.l);
        let limit = self.inline_limit.min(v31::lane_inline_share(total, geom.nk, in_flight, self.h, self.helpers));
        let split = trellis::inline_bytes(n, self.h) > limit && unfinished < self.helpers;
        ActiveLayer { layer, code, cover, costs, stego: vec![0u8; n], run: LayerRun::Pending, limit, split }
    }

    fn layer_msg(&self, k: usize, geom: &LaneGeom, layer: u8) -> &[u8] {
        let bits = self.lane_bits(k);
        if layer == 1 { &bits[..geom.m1] } else { &bits[geom.m1..] }
    }

    /// The next trellis task: traces and segments of split layers first
    /// (they keep the trace chain moving), then pending layers, then a new
    /// lane.
    fn next_lane_task(&mut self) -> Result<Option<Vec<u8>>, StegoError> {
        let Some(l) = self.lanes.as_ref().map(|x| x.l) else { return Ok(None) };
        for k in 0..l {
            if let Some(t) = self.split_task(k)? {
                return Ok(Some(t));
            }
        }
        for k in 0..l {
            if let Some(t) = self.pending_task(k)? {
                return Ok(Some(t));
            }
        }
        let lanes = self.lanes.as_ref().expect("lanes started");
        if lanes.next_lane < l {
            let k = lanes.next_lane;
            let geom = LaneGeom::new(v31::share(self.ord.as_ref().expect("order").len(), l, k).len(), self.lane_bits(k).len())?;
            let first = if geom.m2 > 0 { 2 } else { 1 };
            let flip2 = vec![false; geom.nk];
            let walked = geom.nk.min(geom.n1().max(geom.n2()));
            let cache = (geom.nk <= LANE_CACHE_MAX_POSITIONS).then(|| LaneCache::build(&self.lane_input(k, None), walked));
            let act = self.build_layer(k, &geom, first, &flip2, cache.as_ref());
            let lanes = self.lanes.as_mut().expect("lanes started");
            lanes.next_lane += 1;
            lanes.states[k] = Some(LaneState { geom, cache, flip2, active: Some(act) });
            return self.pending_task(k);
        }
        Ok(None)
    }

    /// A pending layer's single task (trellis, or pass A of a split layer;
    /// a large pass A runs here instead and the layer's first segment is
    /// returned).
    fn pending_task(&mut self, k: usize) -> Result<Option<Vec<u8>>, StegoError> {
        let h = self.h;
        let Some(lanes) = self.lanes.as_ref() else { return Ok(None) };
        let Some(st) = lanes.states[k].as_ref() else { return Ok(None) };
        let Some(act) = st.active.as_ref() else { return Ok(None) };
        if !matches!(act.run, LayerRun::Pending) {
            return Ok(None);
        }
        let msg = self.layer_msg(k, &st.geom, act.layer);
        if act.split && act.cover.len() * 5 > self.local_pass_a_bytes {
            let kb = segment_blocks(&act.code);
            let pa = trellis::pass_a(&act.cover, &act.costs, msg, &act.code, h, kb, &|_| {})?
                .ok_or(StegoError::MessageTooLarge)?;
            let m = act.code.m();
            let done = (act.cover.len() / 2) as u64;
            let layer = act.layer;
            self.local_pass_a_count += 1;
            self.start_split(k, layer, m, kb, pa, done)?;
            return self.split_task(k);
        }
        let li = LayerInput { code: &act.code, b0: 0, msg, cover: &act.cover, costs: &act.costs };
        let n = act.cover.len();
        let task = if act.split {
            let hd = Header { kind: TASK_PASS_A, lane: k as u32, layer: act.layer, seg: 0, target: -1 };
            let mut w = W::new(hd, n * 5 + act.code.max_width() * 12 + 32 + msg.len() + 16);
            w.u8(h as u8);
            w.u32(segment_blocks(&act.code) as u32);
            write_layer(&mut w, &li);
            w.0
        } else {
            let hd = Header { kind: TASK_TRELLIS, lane: k as u32, layer: act.layer, seg: 0, target: -1 };
            let mut w = W::new(hd, n * 5 + act.code.max_width() * 12 + 32 + msg.len() + 16);
            w.u8(h as u8);
            w.u64(act.limit as u64);
            write_layer(&mut w, &li);
            w.0
        };
        let act = self.active_mut(k, act_layer(&task))?;
        act.run = LayerRun::Out;
        Ok(Some(task))
    }

    /// Start the segments of a split layer from its pass A.
    fn start_split(&mut self, k: usize, layer: u8, m: usize, kb: usize, pa: PassA, done: u64) -> Result<(), StegoError> {
        let nseg = m.div_ceil(kb);
        if pa.checkpoints.len() != nseg << self.h {
            return Err(bad());
        }
        let best = pa.best;
        let act = self.active_mut(k, layer)?;
        act.run = LayerRun::Split(Box::new(Split {
            k: kb,
            pass_a: pa,
            next_issue: Some(nseg - 1),
            ready: BTreeMap::new(),
            trace_seg_end: nseg - 1,
            trace_seg: nseg - 1,
            trace_state: best,
            tracing: false,
        }));
        self.work_done += done;
        Ok(())
    }

    /// The next task of a split layer: the trace of the next segment once
    /// its rows exist, else the next segment (while fewer than two per
    /// helper are untraced).
    fn split_task(&mut self, k: usize) -> Result<Option<Vec<u8>>, StegoError> {
        let h = self.h;
        let ns = 1usize << h;
        let Some(lanes) = self.lanes.as_ref() else { return Ok(None) };
        let Some(st) = lanes.states[k].as_ref() else { return Ok(None) };
        let Some(act) = st.active.as_ref() else { return Ok(None) };
        let LayerRun::Split(sp) = &act.run else { return Ok(None) };
        let m = act.code.m();
        if !sp.tracing
            && let Some(&holder) = sp.ready.get(&sp.trace_seg)
        {
            let seg = sp.trace_seg;
            let hd = Header { kind: TASK_TRACE, lane: k as u32, layer: act.layer, seg: seg as u32, target: holder };
            let mut wr = W::new(hd, 4);
            wr.u32(sp.trace_state as u32);
            let act = self.active_mut(k, hd.layer)?;
            let LayerRun::Split(sp) = &mut act.run else { unreachable!() };
            sp.tracing = true;
            sp.ready.remove(&seg);
            return Ok(Some(wr.0));
        }
        let Some(seg) = sp.next_issue else { return Ok(None) };
        let untraced = sp.trace_seg + 1 - seg - 1;
        if untraced >= 2 * self.helpers {
            return Ok(None);
        }
        let (b0, b1) = (seg * sp.k, ((seg + 1) * sp.k).min(m));
        let (p0, p1) = (act.code.start(b0), act.code.start(b1));
        let msg = &self.layer_msg(k, &st.geom, act.layer)[b0..b1];
        let li = LayerInput { code: &act.code, b0, msg, cover: &act.cover[p0..p1], costs: &act.costs[p0..p1] };
        // Round robin from the last segment: each helper computes its
        // segments in the order the trace chain needs them, so the chain
        // waits at most for one segment per helper (with random placement it
        // kept waiting for helpers busy with segments it needed much later).
        let target = ((sp.trace_seg_end - seg) % self.helpers) as i32;
        let hd = Header { kind: TASK_SEGMENT, lane: k as u32, layer: act.layer, seg: seg as u32, target };
        let mut wr = W::new(hd, (p1 - p0) * 5 + act.code.max_width() * 12 + 32 + msg.len() + ns * 8 + 16);
        wr.u8(h as u8);
        write_layer(&mut wr, &li);
        wr.f64s(&sp.pass_a.checkpoints[seg * ns..(seg + 1) * ns]);
        let act = self.active_mut(k, hd.layer)?;
        let LayerRun::Split(sp) = &mut act.run else { unreachable!() };
        sp.next_issue = seg.checked_sub(1);
        Ok(Some(wr.0))
    }

    /// A lane layer's stego bits are complete: start layer 1, or finish the
    /// lane (and the encode, after the last lane).
    fn layer_done(&mut self, k: usize) -> Result<(), StegoError> {
        let lanes = self.lanes.as_mut().ok_or_else(bad)?;
        let mut st = lanes.states[k].take().ok_or_else(bad)?;
        let act = st.active.take().ok_or_else(bad)?;
        if act.layer == 2 {
            st.flip2 = v31::layer2_flips(&st.geom, &act.cover, &act.stego);
            drop(act);
            let next = self.build_layer(k, &st.geom, 1, &st.flip2, st.cache.as_ref());
            st.active = Some(next);
            self.lanes.as_mut().expect("lanes").states[k] = Some(st);
            return Ok(());
        }
        let plan = v31::lane_plan(&self.lane_input(k, st.cache.as_ref()), &st.geom, &st.flip2, &act.cover, &act.stego);
        drop((act, st));
        let lanes = self.lanes.as_mut().expect("lanes");
        lanes.plans[k] = Some(plan);
        lanes.left -= 1;
        if lanes.left == 0 {
            self.apply()?;
        }
        Ok(())
    }

    fn apply(&mut self) -> Result<(), StegoError> {
        let lanes = self.lanes.take().ok_or_else(bad)?;
        let ord = self.ord.take().ok_or_else(bad)?;
        let plans: Vec<LanePlan> = lanes.plans.into_iter().map(|p| p.ok_or_else(bad)).collect::<Result<_, _>>()?;
        let p = v31::assemble(plans, &ord, self.img.dct_grid(0), None, self.frame_bits.len(), lanes.median_cost, &lanes.sample_ab, self.bound);
        drop(ord);
        let coeffs = self.img.dct_grid_mut(0).coeffs_mut();
        for &(fi, v) in &p.changes {
            coeffs[fi as usize] = v;
        }
        self.stats = Some(p.stats);
        Ok(())
    }
}

/// Message blocks per segment of a split trellis of width `w`.
fn segment_blocks(code: &Code) -> usize {
    (SEGMENT_POSITIONS / code.n().div_ceil(code.m().max(1)).max(1)).max(1)
}

/// Trellis positions of a lane (both layers), for the progress fraction.
fn lane_work(nk: usize, bits: usize) -> usize {
    LaneGeom::new(nk, bits).map_or(0, |g| g.n1() + g.n2())
}

/// Run a whole split encode in this thread with `helpers` simulated helpers
/// (tests: the split path must give exactly `ghost_encode_with_files`'s
/// bytes). Tasks run in a shuffled order to exercise out-of-order results.
/// Also returns how many tasks of each kind ran (index = kind; index 0 =
/// pass A runs in the coordinator). `local_pass_a_bytes`: override of
/// `LOCAL_PASS_A_BYTES` (`None` = default).
#[doc(hidden)]
pub fn run_split_encode_locally(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    pixels: Option<(&[u8], u32, u32)>,
    helpers: usize,
    local_pass_a_bytes: Option<usize>,
) -> Result<(Vec<u8>, EncodeQuality, [usize; 6]), StegoError> {
    let mut counts = [0usize; 6];
    let mut s = GhostEncodeSession::new(image_bytes, message, files, passphrase, pixels, helpers)?;
    if let Some(b) = local_pass_a_bytes {
        s.local_pass_a_bytes = b;
    }
    let mut runners: Vec<GhostTaskRunner> = (0..helpers.max(1)).map(|_| GhostTaskRunner::new()).collect();
    let mut queue: Vec<Vec<u8>> = s.take_tasks()?;
    s.run_local()?;
    let mut rot = 0usize;
    while !s.is_done() {
        queue.extend(s.take_tasks()?);
        if queue.is_empty() {
            return Err(bad());
        }
        // Take tasks in a rotating (not FIFO) order.
        rot = rot.wrapping_mul(31).wrapping_add(7);
        let task = queue.remove(rot % queue.len());
        let target = i32::from_le_bytes(task[TASK_TARGET_OFFSET..TASK_TARGET_OFFSET + 4].try_into().unwrap());
        let helper = if target >= 0 { target as usize } else { rot % runners.len() };
        counts[(task[0] as usize).min(5)] += 1;
        let result = runners[helper].run(task)?;
        s.submit(&result, helper as i32)?;
    }
    counts[0] = s.local_pass_a_count;
    let (bytes, q) = s.finish()?;
    Ok((bytes, q, counts))
}
