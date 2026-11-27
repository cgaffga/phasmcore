// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The Ghost v3.1 primary encode split into tasks, for hosts whose threads do
//! not share memory: the website runs it in Web Workers, each with its own
//! WASM instance (no `SharedArrayBuffer`, so no COOP/COEP headers).
//!
//! A [`GhostEncodeSession`] (the coordinator) runs the one-call encoder's
//! plan (`v31::plan`, `memory::ghost_lane_plan`) and holds what it holds
//! through the embedding: the cover's luma (the chroma is dropped after the
//! parse and read again from the cover file by [`GhostEncodeSession::finish`]),
//! the side information, the frame, the keyed order, the costs of the
//! current cost group (the whole cover's when the plan has one group), and
//! per lane in flight its bit-packed P2 flips and one layer's cover and
//! stego bits (3 bits per position). Lane inputs are computed on demand,
//! as the one-call encoder's trellises compute them (`v31::LaneLayerInput`):
//! into a task's bytes as the task is written, a segment at a time where a
//! layer's input is larger than the plan lets a lane's trellis hold.
//!
//! It hands out self-contained tasks (byte strings) and takes their results
//! back. A [`GhostTaskRunner`] on each helper runs tasks, reading a task's
//! input in place:
//!
//! - **cost band**: the J-UNIWARD costs of a band of block rows, from the
//!   band's sub-grid (`uniward::ac_cost_band`), once per cost group;
//! - **trellis**: one layer of one lane (layer 2, then layer 1 once the
//!   coordinator has the layer-2 flips), whole, with the input as cover bits
//!   (1 bit) and costs (4 B) per position;
//! - **forward / segment / trace** (a *split* layer: its input larger than
//!   the plan allows a lane's trellis, or fewer lanes left than helpers):
//!   the forward pass as a chain of segment steps (any helper; each returns
//!   the state costs at its end), then the segments' back pointers
//!   recomputed on all helpers (each keeps its rows) and traced from the
//!   last segment to the first (`stc::trellis::segment_forward`, …).
//!
//! The coordinator computes the keyed order while the helpers compute the
//! costs. The output is byte-identical to `ghost_encode_with_files` /
//! `ghost_encode_si_with_files` for any number of helpers and any memory
//! budget: the same parse, frame, order, costs (bit-identical by band), λ,
//! lane inputs, cost groups, trellis plans (every schedule plans the same),
//! changes and JPEG write (`pipeline::stego_image`).

use std::collections::{BTreeMap, HashMap};
use std::ops::Range;
use std::sync::atomic::AtomicU64;

use sha2::{Digest, Sha256};

use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::{DctGrid, QuantTable};
use crate::stego::cost::uniward;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::payload::FileEntry;
use crate::stego::quality::EncodeQuality;
use crate::stego::side_info::SideInfo;
use crate::stego::stc::trellis::{self, Code, TrellisChunk, TrellisInput};

use super::pipeline::{self, PreparedEncode};
use super::v31::{self, LaneCaps, LaneGeom, LaneInput, LanePlan, Layer1Keep, Rho, V31Seeds};

const TASK_COST_BAND: u8 = 1;
const TASK_TRELLIS: u8 = 2;
const TASK_FORWARD: u8 = 3;
const TASK_SEGMENT: u8 = 4;
const TASK_TRACE: u8 = 5;

/// Bytes of the header every task and result starts with: kind (u8), lane
/// (u32), layer (u8), segment (u32), target helper (i32, −1 = any).
pub const TASK_HEADER_BYTES: usize = 14;

/// Byte offset of the target helper in a task (a trace must run on the
/// helper that keeps its segment's back pointers).
pub const TASK_TARGET_OFFSET: usize = 10;

/// Positions per segment of a split trellis (a 1 MB task, 16 MB of back
/// pointers at h = 9; a segment is at least one message block).
const SEGMENT_POSITIONS: usize = 1 << 18;

/// Smallest cost band (block rows): margins are 2 rows on either side.
const MIN_BAND_ROWS: usize = 8;

/// Positions a helper decodes per chunk of a task's input.
const TASK_CHUNK: usize = 1 << 12;

/// Segments of split layers a helper holds at once, issued and not yet
/// traced (16 MB of back pointers each at h = 9), over all split layers.
const HELD_SEGMENTS_PER_HELPER: usize = 2;

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
    fn bytes(&mut self, v: &[u8]) {
        self.0.extend_from_slice(v);
    }
    fn u32s(&mut self, v: &[u32]) {
        for x in v {
            self.u32(*x);
        }
    }
    fn f64s(&mut self, v: &[f64]) {
        for x in v {
            self.0.extend_from_slice(&x.to_le_bytes());
        }
    }
    /// The first `n` bits of bit-packed `words` (bit `j` = bit `j mod 8` of
    /// byte `j / 8`).
    fn bits(&mut self, words: &[u64], n: usize) {
        let nb = n.div_ceil(8);
        for (i, w) in words.iter().enumerate().take(nb.div_ceil(8)) {
            let b = w.to_le_bytes();
            self.0.extend_from_slice(&b[..(nb - i * 8).min(8)]);
        }
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
}

/// The error of a malformed or out-of-protocol task, result or call (a host
/// fault, not the cover's: [`is_protocol_error`]).
fn bad() -> StegoError {
    StegoError::InvalidJpeg(crate::codec::jpeg::error::JpegError::InvalidMarkerData(PROTOCOL_FAULT))
}

const PROTOCOL_FAULT: &str = "ghost task";

/// Whether an error of a [`GhostEncodeSession`] or [`GhostTaskRunner`] is a
/// fault of the host's protocol (a task or result that does not belong to
/// the session, a call out of order, a wrong cover passed to
/// [`GhostEncodeSession::finish`]) rather than of the cover or the message:
/// a host reports it as its own failure (the website falls back to its
/// single-thread encode).
pub fn is_protocol_error(e: &StegoError) -> bool {
    matches!(e, StegoError::InvalidJpeg(crate::codec::jpeg::error::JpegError::InvalidMarkerData(m)) if *m == PROTOCOL_FAULT)
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
    fn f64s(&mut self, n: usize) -> Result<Vec<f64>, StegoError> {
        Ok(self.take(n * 8)?.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// OR `n` bits (bit-packed bytes, bit `j` of `src` = bit `j mod 8` of byte
/// `j / 8`) into the zeroed positions `at..at + n` of `dst`.
fn or_bits_at(dst: &mut [u64], at: usize, src: &[u8], n: usize) {
    for (bi, &byte) in src.iter().enumerate().take(n.div_ceil(8)) {
        let mut v = byte as u64;
        if (bi + 1) * 8 > n {
            v &= (1u64 << (n - bi * 8)) - 1;
        }
        if v == 0 {
            continue;
        }
        let j = at + bi * 8;
        dst[j >> 6] |= v << (j & 63);
        if (j & 63) > 56 {
            let hi = v >> (64 - (j & 63));
            if hi != 0 {
                dst[(j >> 6) + 1] |= hi;
            }
        }
    }
}

/// Bytes of a task's layer input of `n` positions: the cover bits
/// bit-packed, then the costs (f32).
fn input_bytes(n: usize) -> usize {
    n.div_ceil(8) + 4 * n
}

fn code_bytes(code: &Code) -> usize {
    16 + 4 * (3 + code.column_sets().iter().map(|s| s.len()).sum::<usize>())
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

/// Bytes of a task carrying blocks `b0..b0 + m` of `code` (and `extra`).
fn layer_task_bytes(code: &Code, b0: usize, m: usize, extra: usize) -> usize {
    16 + code_bytes(code) + 8 + m + input_bytes(code.start(b0 + m) - code.start(b0)) + extra
}

/// Blocks `[b0, b0 + msg.len())` of `code` (message bits `msg`): their
/// positions' cover bits and costs, read from `input` (layer positions) as
/// they are written.
fn write_layer(w: &mut W, code: &Code, b0: usize, msg: &[u8], input: &dyn TrellisInput) {
    write_code(w, code);
    w.u32(b0 as u32);
    w.u32(msg.len() as u32);
    w.bytes(msg);
    let p0 = code.start(b0);
    let n = code.start(b0 + msg.len()) - p0;
    let at = w.0.len();
    w.0.resize(at + input_bytes(n), 0);
    let (bits, costs) = w.0[at..].split_at_mut(n.div_ceil(8));
    let mut buf = TrellisChunk::default();
    let mut j = 0;
    while j < n {
        let (c, f) = input.chunk(p0 + j, p0 + n, &mut buf);
        for (k, (&cb, &cf)) in c.iter().zip(f).enumerate() {
            let i = j + k;
            bits[i >> 3] |= (cb & 1) << (i & 7);
            costs[4 * i..4 * i + 4].copy_from_slice(&cf.to_le_bytes());
        }
        j += c.len();
    }
}

/// A task's layer input, read in place: cover bits (bit-packed) and costs.
struct TaskInput<'a> {
    bits: &'a [u8],
    costs: &'a [u8],
}

impl TrellisInput for TaskInput<'_> {
    fn chunk<'s>(&'s self, j0: usize, j1: usize, buf: &'s mut TrellisChunk) -> (&'s [u8], &'s [f32]) {
        let j1 = j1.min(j0 + TASK_CHUNK);
        buf.cover.clear();
        buf.costs.clear();
        for j in j0..j1 {
            buf.cover.push((self.bits[j >> 3] >> (j & 7)) & 1);
            buf.costs.push(f32::from_le_bytes(self.costs[4 * j..4 * j + 4].try_into().unwrap()));
        }
        (&buf.cover, &buf.costs)
    }
}

/// A task's layer: blocks `[b0, b0 + msg.len())` of `code` and their input.
struct TaskLayer<'a> {
    code: Code,
    b0: usize,
    msg: &'a [u8],
    input: TaskInput<'a>,
}

fn read_layer<'a>(r: &mut R<'a>) -> Result<TaskLayer<'a>, StegoError> {
    let code = read_code(r)?;
    let b0 = r.u32()? as usize;
    let m = r.len(1)?;
    if b0.checked_add(m).is_none_or(|e| e > code.m()) {
        return Err(bad());
    }
    let msg = r.take(m)?;
    let n = code.start(b0 + m) - code.start(b0);
    let bits = r.take(n.div_ceil(8))?;
    let costs = r.take(4 * n)?;
    Ok(TaskLayer { code, b0, msg, input: TaskInput { bits, costs } })
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

    /// Run one task; returns its result. The task's input is read in place
    /// (no copy); the buffer is freed before the result is written.
    pub fn run(&mut self, task: Vec<u8>) -> Result<Vec<u8>, StegoError> {
        let (hd, mut r) = R::new(&task)?;
        let reply = Header { target: -1, ..hd };
        let ns = |h: usize| 1usize << h.min(trellis::MAX_H);
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
                if br0 >= br1 || br1 > bt {
                    return Err(bad());
                }
                let g0 = br0.saturating_sub(uniward::COST_BAND_MARGIN_ROWS);
                let g1 = (br1 + uniward::COST_BAND_MARGIN_ROWS).min(bt);
                let coeffs = r.take((g1 - g0) * bw * 64 * 2)?;
                let mut sub = DctGrid::new(bw, g1 - g0);
                for (d, c) in sub.coeffs_mut().iter_mut().zip(coeffs.chunks_exact(2)) {
                    *d = i16::from_le_bytes([c[0], c[1]]);
                }
                drop(task);
                let costs = uniward::ac_cost_band(&sub, &QuantTable::new(qv), br0, br1, bt)?;
                drop(sub);
                let mut w = W::new(reply, costs.len() * 4);
                for c in &costs {
                    w.0.extend_from_slice(&c.to_le_bytes());
                }
                Ok(w.0)
            }
            TASK_TRELLIS => {
                let h = r.u8()? as usize;
                let limit = r.u64()? as usize;
                let li = read_layer(&mut r)?;
                if li.b0 != 0 || li.msg.len() != li.code.m() {
                    return Err(bad());
                }
                let n = li.code.n();
                let plan = trellis::embed_packed(&li.input, n, li.msg, &li.code, h, limit, 1, &|_| {})?;
                drop(li);
                drop(task);
                let mut w = W::new(reply, n.div_ceil(8) + 1);
                match plan {
                    Some((words, _)) => {
                        w.u8(1);
                        w.bits(&words, n);
                    }
                    None => w.u8(0),
                }
                Ok(w.0)
            }
            TASK_FORWARD => {
                let h = r.u8()? as usize;
                let li = read_layer(&mut r)?;
                let start = r.f64s(ns(h))?;
                let end = trellis::segment_forward(&li.input, li.msg, &li.code, li.b0, h, &start)?;
                drop(li);
                drop(task);
                let mut w = W::new(reply, end.len() * 8);
                w.f64s(&end);
                Ok(w.0)
            }
            TASK_SEGMENT => {
                let h = r.u8()? as usize;
                let li = read_layer(&mut r)?;
                let checkpoint = r.f64s(ns(h))?;
                let rows = trellis::segment_rows(&li.input, li.msg, &li.code, li.b0, h, &checkpoint)?;
                let TaskLayer { code, b0, msg, .. } = li;
                let msg = msg.to_vec();
                drop(task);
                self.held.insert((hd.lane, hd.layer, hd.seg), Held { rows, msg, code, b0, h });
                Ok(W::new(reply, 0).0)
            }
            TASK_TRACE => {
                let state = r.u32()? as usize;
                drop(task);
                let held = self.held.remove(&(hd.lane, hd.layer, hd.seg)).ok_or_else(bad)?;
                let n = held.code.start(held.b0 + held.msg.len()) - held.code.start(held.b0);
                let mut bits = vec![0u8; n];
                let start = trellis::segment_trace(&held.rows, &held.msg, &held.code, held.b0, held.h, state, &mut bits);
                drop(held);
                let mut w = W::new(reply, n.div_ceil(8) + 4);
                w.u32(start as u32);
                let at = w.0.len();
                w.0.resize(at + n.div_ceil(8), 0);
                for (i, &b) in bits.iter().enumerate() {
                    w.0[at + (i >> 3)] |= (b & 1) << (i & 7);
                }
                Ok(w.0)
            }
            _ => Err(bad()),
        }
    }
}

// ---------------------------------------------------------------------------
// Coordinator side
// ---------------------------------------------------------------------------

/// A split layer in progress: the forward chain, then the segments' back
/// pointers (on the helpers) traced from the last segment.
struct Split {
    /// Message blocks per segment, and segments.
    k: usize,
    nseg: usize,
    /// State costs at the start of each segment the forward chain reached
    /// (`2^h` each).
    checkpoints: Vec<f64>,
    /// Next segment of the forward chain (`nseg`: done) and whether its step
    /// is out.
    forward: usize,
    forward_out: bool,
    /// Next segment to recompute (from the last one down); `None` before the
    /// forward chain ends and when all are out.
    next_issue: Option<usize>,
    /// Segments computed: the helper keeping their rows.
    ready: BTreeMap<usize, i32>,
    /// Next segment to trace (from the last one down) and its end state.
    trace_seg: usize,
    trace_state: usize,
    /// A trace task is out, on this helper.
    tracing: bool,
    trace_holder: usize,
}

enum LayerRun {
    /// Not yet handed out.
    Pending,
    /// The whole-layer trellis task is out.
    Out,
    Split(Box<Split>),
}

struct ActiveLayer {
    layer: u8,
    code: Code,
    /// The cover bits its input handed out (recorded as tasks are written).
    cover: Vec<AtomicU64>,
    /// Stego bits (bit-packed): the trellis result, or the traces.
    stego: Vec<u64>,
    run: LayerRun,
    /// Back-pointer bytes its whole-layer trellis task may keep on a helper
    /// (the cover's in-flight working set shared among the helpers' lanes).
    limit: usize,
    /// Run as forward chain + segments.
    split: bool,
}

struct LaneState {
    geom: LaneGeom,
    /// Layer 2's P2 flips (bit-packed; zero before layer 2 is planned, or
    /// without it).
    flip2: Vec<u64>,
    /// Layer 1's input kept by layer 2's (when it fits the plan's lane
    /// working set), until layer 1's input is written.
    keep: Option<Layer1Keep>,
    /// The layer being planned.
    active: Option<ActiveLayer>,
}

struct Lanes {
    lam: f64,
    median_cost: f32,
    sample_ab: Vec<(f64, f64)>,
    states: Vec<Option<LaneState>>,
    plans: Vec<Option<LanePlan>>,
    next_lane: usize,
    /// Lanes of the current cost group not yet planned.
    left: usize,
}

/// The current cost group's costs.
enum GroupCosts {
    /// Between groups.
    None,
    /// The plan's only group: every luma coefficient's cost by flat index.
    Dense(Vec<f32>),
    /// A group of several, its cost pass running: the costs of its
    /// positions (in order), followed for the first group by the λ samples';
    /// the index once the order exists.
    Gathering { costs: Vec<f32>, gather: Option<Gather> },
    /// A group of several: the costs of its positions, in order.
    Lanes(Vec<f32>),
}

/// Where a band's costs go in a group's gather: the request's indices
/// bucketed by band (`idx[starts[b]..starts[b + 1]]`), 4 B per position
/// (the one-call encoder's sort index: `uniward::gather_ac_costs_parts`).
struct Gather {
    idx: Vec<u32>,
    starts: Vec<u32>,
}

/// The coordinator of a split Ghost v3.1 encode (see the module docs).
pub struct GhostEncodeSession {
    /// The cover's luma (the chroma is dropped).
    img: JpegImage,
    /// SHA-256 of the cover file ([`Self::finish`] parses it again).
    cover_digest: [u8; 32],
    frame_bits: Vec<u8>,
    si: Option<SideInfo>,
    seeds: V31Seeds,
    /// v3.1 pair bits of the cover.
    pairs: v31::PairKey,
    /// The dummy (MCU-padding) blocks, never changed (`v31::dummy_block_wet`).
    wet: Option<v31::WetMask>,
    h: usize,
    /// The plan's back-pointer bytes per lane, lanes at once, cost group
    /// size (`memory::ghost_lane_plan`), and the workers it was made for.
    inline_limit: usize,
    parallel_lanes: usize,
    workers: usize,
    helpers: usize,
    bound: bool,
    /// AC positions, frame bits, lanes, the largest lane's capacities.
    n: usize,
    m: usize,
    l: usize,
    caps: LaneCaps,
    /// Lanes of each cost group, the current group.
    groups: Vec<Range<usize>>,
    group: usize,
    costs: GroupCosts,
    /// Block rows of each cost band, the band of each block row.
    bands: Vec<(usize, usize)>,
    band_of_row: Vec<u32>,
    /// A cost pass is running: bands handed out, results to come.
    pass_open: bool,
    next_band: usize,
    bands_left: usize,
    ord: Option<Vec<u32>>,
    /// Order positions of the λ / median samples (`v31::sample_positions`).
    samples: Vec<usize>,
    n_med: usize,
    lanes: Option<Lanes>,
    lane_tasks_out: usize,
    /// Segments each helper holds (issued, not yet traced).
    held: Vec<usize>,
    work_total: u64,
    work_done: u64,
    stats: Option<v31::V31Stats>,
    changes: Vec<(u32, i16)>,
    /// Override of a whole-layer task's largest size (tests).
    task_cap: usize,
}

impl GhostEncodeSession {
    /// Start an encode of `message` (+ `files`) into `image_bytes`, with
    /// Deep Cover side information when `pixels` (sRGB, 3 bytes per pixel,
    /// `width × height`) are given — exactly `ghost_encode_with_files` /
    /// `ghost_encode_si_with_files`. `helpers`: how many helpers will run
    /// tasks (sizes the bands and decides when to split a trellis). The
    /// cover file is needed again by [`Self::finish`].
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
        let PreparedEncode { mut img, frame_bits, opts, .. } =
            pipeline::prepare_encode(image_bytes, message, files, passphrase, si.is_some(), pre)?;
        let key = crypto::derive_structural_key(passphrase)?;
        pipeline::drop_chroma(&mut img);
        let seeds = v31::v31_seeds(&key);
        let grid = img.dct_grid(0);
        let (bw, bt) = (grid.blocks_wide(), grid.blocks_tall());
        let n = v31::ac_count(grid);
        let m = frame_bits.len();
        if m == 0 || m > n {
            return Err(StegoError::MessageTooLarge);
        }
        let helpers = helpers.max(1);
        // Bands of about a cost strip's working set (their costs, 4 B per
        // coefficient, arrive here one at a time), at least two per helper.
        let strip_rows = uniward::cost_strip_working_bytes(bw) / (bw * 64 * 4).max(1);
        let nb = bt
            .div_ceil(strip_rows.max(MIN_BAND_ROWS))
            .max((2 * helpers).min(bt.div_ceil(MIN_BAND_ROWS)))
            .max(1);
        let bands: Vec<(usize, usize)> = (0..nb).map(|i| (i * bt / nb, (i + 1) * bt / nb)).filter(|(a, b)| a < b).collect();
        let mut band_of_row = vec![0u32; bt];
        for (b, &(r0, r1)) in bands.iter().enumerate() {
            band_of_row[r0..r1].fill(b as u32);
        }
        let l = v31::lanes(n, m);
        let groups = v31::lane_groups(n, l, opts.group_positions);
        let work_total = groups.len() as u64 * (n / 4) as u64
            + (0..l).map(|k| lane_work(v31::share(n, l, k).len(), v31::share(m, l, k).len()) as u64).sum::<u64>();
        let wet = v31::dummy_block_wet(&img);
        let pairs = v31::PairKey::new(&seeds.pair, grid.coeffs().len());
        let (samples, n_med) = v31::sample_positions(n);
        let mut s = Self {
            cover_digest: Sha256::digest(image_bytes).into(),
            pairs,
            inline_limit: opts.lane_inline_bytes,
            parallel_lanes: opts.parallel_lanes,
            workers: crate::stego::memory::encode_workers(),
            h: opts.h,
            bound: opts.bound,
            img,
            frame_bits,
            si,
            seeds,
            helpers,
            n,
            m,
            l,
            caps: v31::LaneCaps::of(n, l),
            groups,
            group: 0,
            costs: GroupCosts::None,
            wet,
            pass_open: false,
            bands_left: 0,
            bands,
            band_of_row,
            next_band: 0,
            ord: None,
            samples,
            n_med,
            lanes: None,
            lane_tasks_out: 0,
            held: vec![0; helpers],
            work_total,
            work_done: 0,
            stats: None,
            changes: Vec::new(),
            task_cap: usize::MAX,
        };
        s.start_cost_pass();
        Ok(s)
    }

    /// Fraction of the work done (cost passes and trellises), 0..1.
    pub fn progress(&self) -> f64 {
        if self.work_total == 0 { 0.0 } else { (self.work_done as f64 / self.work_total as f64).min(1.0) }
    }

    /// All lanes planned: [`Self::finish`] can write the JPEG.
    pub fn is_done(&self) -> bool {
        self.stats.is_some()
    }

    /// The coordinator's own work, best run right after the first
    /// [`Self::take_tasks`] (while the helpers compute the costs): the keyed
    /// order. Done on demand otherwise.
    pub fn run_local(&mut self) -> Result<(), StegoError> {
        self.ensure_order();
        self.maybe_end_cost_pass()
    }

    /// Tasks ready to run ([`Self::next_task`] until there are none).
    pub fn take_tasks(&mut self) -> Result<Vec<Vec<u8>>, StegoError> {
        let mut out = Vec::new();
        while let Some(t) = self.next_task()? {
            out.push(t);
        }
        Ok(out)
    }

    /// The next task ready to run, any helper unless the task's target says
    /// otherwise (bytes `TASK_TARGET_OFFSET..+4`, i32): every cost band of a
    /// cost pass at once, trellis work at most one task per helper at a
    /// time. A host that copies the tasks out takes them one at a time, so
    /// only one is held here at once.
    pub fn next_task(&mut self) -> Result<Option<Vec<u8>>, StegoError> {
        if self.pass_open && self.next_band < self.bands.len() {
            let t = self.band_task(self.next_band);
            self.next_band += 1;
            return Ok(Some(t));
        }
        if self.lane_tasks_out < self.helpers
            && let Some(t) = self.next_lane_task()?
        {
            self.lane_tasks_out += 1;
            return Ok(Some(t));
        }
        Ok(None)
    }

    /// A helper's result. `helper`: the index of the helper that ran it
    /// (segments stay on it for their trace).
    pub fn submit(&mut self, result: &[u8], helper: i32) -> Result<(), StegoError> {
        let (hd, mut r) = R::new(result)?;
        match hd.kind {
            TASK_COST_BAND => {
                let b = hd.seg as usize;
                let (br0, br1) = *self.bands.get(b).ok_or_else(bad)?;
                if !self.pass_open || b >= self.next_band || self.bands_left == 0 {
                    return Err(bad());
                }
                let grid = self.img.dct_grid(0);
                let bw = grid.blocks_wide();
                let bt = grid.blocks_tall();
                let f0 = br0 * bw * 64;
                let costs = r.take((br1 - br0) * bw * 64 * 4)?;
                let cost = |fi: usize| f32::from_le_bytes(costs[4 * (fi - f0)..4 * (fi - f0) + 4].try_into().unwrap());
                if matches!(self.costs, GroupCosts::Gathering { gather: None, .. }) {
                    // A group's gather needs the order (done here if the
                    // result came before `run_local`).
                    self.ensure_order();
                }
                match &mut self.costs {
                    GroupCosts::Dense(dense) => {
                        for (d, c) in dense[f0..br1 * bw * 64].iter_mut().zip(costs.chunks_exact(4)) {
                            *d = f32::from_le_bytes(c.try_into().unwrap());
                        }
                    }
                    GroupCosts::Gathering { costs: out, gather: Some(g) } => {
                        let ord = self.ord.as_ref().ok_or_else(bad)?;
                        let (r0, head) = group_head(self.n, self.l, &self.groups[self.group]);
                        for &i in &g.idx[g.starts[b] as usize..g.starts[b + 1] as usize] {
                            let fi = request_fi(ord, r0, head, &self.samples, i as usize) as usize;
                            out[i as usize] = cost(fi);
                        }
                    }
                    _ => return Err(bad()),
                }
                self.bands_left -= 1;
                self.work_done += ((self.n / 4) * (br1 - br0) / bt) as u64;
                self.maybe_end_cost_pass()
            }
            TASK_TRELLIS => {
                self.lane_tasks_out = self.lane_tasks_out.saturating_sub(1);
                let k = hd.lane as usize;
                let act = self.active_mut(k, hd.layer)?;
                if !matches!(act.run, LayerRun::Out) {
                    return Err(bad());
                }
                if r.u8()? == 0 {
                    return Err(StegoError::MessageTooLarge);
                }
                let n = act.code.n();
                let packed = r.take(n.div_ceil(8))?;
                or_bits_at(&mut act.stego, 0, packed, n);
                self.work_done += n as u64;
                self.layer_done(k)
            }
            TASK_FORWARD => {
                self.lane_tasks_out = self.lane_tasks_out.saturating_sub(1);
                let k = hd.lane as usize;
                let ns = 1usize << self.h;
                let act = self.active_mut(k, hd.layer)?;
                let m = act.code.m();
                let LayerRun::Split(sp) = &mut act.run else { return Err(bad()) };
                let seg = hd.seg as usize;
                if seg != sp.forward || !sp.forward_out {
                    return Err(bad());
                }
                let end = r.f64s(ns)?;
                let (b0, b1) = (seg * sp.k, ((seg + 1) * sp.k).min(m));
                let done = (act.code.start(b1) - act.code.start(b0)) as u64 / 2;
                sp.forward += 1;
                sp.forward_out = false;
                if sp.forward < sp.nseg {
                    sp.checkpoints.extend_from_slice(&end);
                } else {
                    let (best, cost) = trellis::best_state(&end);
                    if cost == f64::INFINITY {
                        return Err(StegoError::MessageTooLarge);
                    }
                    sp.next_issue = Some(sp.nseg - 1);
                    sp.trace_seg = sp.nseg - 1;
                    sp.trace_state = best;
                }
                self.work_done += done;
                Ok(())
            }
            TASK_SEGMENT => {
                // The helper keeps the rows for the trace: it must be one of ours.
                if helper < 0 || helper as usize >= self.helpers {
                    return Err(bad());
                }
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
                sp.tracing = false;
                sp.trace_state = start;
                let holder = sp.trace_holder;
                let last = seg == 0;
                if !last {
                    sp.trace_seg = seg - 1;
                }
                or_bits_at(&mut act.stego, p0, packed, p1 - p0);
                self.held[holder] = self.held[holder].saturating_sub(1);
                self.work_done += (p1 - p0) as u64 / 2;
                if last {
                    if start != 0 {
                        return Err(bad());
                    }
                    return self.layer_done(k);
                }
                Ok(())
            }
            _ => Err(bad()),
        }
    }

    /// Whether `image_bytes` is the cover file [`Self::new`] was given (what
    /// [`Self::finish`] needs; a host can check before it gives the session
    /// up).
    pub fn cover_matches(&self, image_bytes: &[u8]) -> bool {
        <[u8; 32]>::from(Sha256::digest(image_bytes)) == self.cover_digest
    }

    /// Write the stego JPEG (after [`Self::is_done`]) into `image_bytes`,
    /// the cover file [`Self::new`] was given (checked: else a protocol
    /// error, [`is_protocol_error`]): a fresh parse of it with the changes,
    /// as the one-call encoder writes (`pipeline::stego_image`).
    pub fn finish(mut self, image_bytes: &[u8]) -> Result<(Vec<u8>, EncodeQuality), StegoError> {
        if self.stats.is_none() || !self.cover_matches(image_bytes) {
            return Err(bad());
        }
        let stats = self.stats.take().ok_or_else(bad)?;
        let is_si = self.si.is_some();
        let changes = std::mem::take(&mut self.changes);
        drop(self);
        let mut img = pipeline::stego_image(image_bytes, changes)?;
        pipeline::finish_encode(&mut img, &stats, is_si)
    }

    // -- internals ----------------------------------------------------------

    fn ensure_order(&mut self) {
        if self.ord.is_none() {
            self.ord = Some(v31::order(self.img.dct_grid(0), &self.seeds));
        }
        if matches!(self.costs, GroupCosts::Gathering { gather: None, .. }) {
            let (g, c) = self.gather();
            self.costs = GroupCosts::Gathering { costs: c, gather: Some(g) };
        }
    }

    /// Start the cost pass of the current group: its bands, and where their
    /// costs go.
    fn start_cost_pass(&mut self) {
        self.pass_open = true;
        self.next_band = 0;
        self.bands_left = self.bands.len();
        if self.groups.len() == 1 {
            self.costs = GroupCosts::Dense(vec![f32::INFINITY; self.img.dct_grid(0).coeffs().len()]);
        } else {
            self.costs = GroupCosts::Gathering { costs: Vec::new(), gather: None };
            if self.ord.is_some() {
                self.ensure_order();
            }
        }
    }

    /// The current group's gather index and its zeroed cost vector (the
    /// group's positions, then for the first group the samples).
    fn gather(&self) -> (Gather, Vec<f32>) {
        let ord = self.ord.as_ref().expect("the order before a group's gather");
        let (r0, head) = group_head(self.n, self.l, &self.groups[self.group]);
        let total = head + if self.group == 0 { self.samples.len() } else { 0 };
        let bw64 = self.img.dct_grid(0).blocks_wide() * 64;
        let band = |i: usize| self.band_of_row[request_fi(ord, r0, head, &self.samples, i) as usize / bw64] as usize;
        let nb = self.bands.len();
        let mut starts = vec![0u32; nb + 1];
        for i in 0..total {
            starts[band(i) + 1] += 1;
        }
        for b in 0..nb {
            starts[b + 1] += starts[b];
        }
        let mut pos: Vec<u32> = starts[..nb].to_vec();
        let mut idx = vec![0u32; total];
        for i in 0..total {
            let b = band(i);
            idx[pos[b] as usize] = i as u32;
            pos[b] += 1;
        }
        (Gather { idx, starts }, vec![f32::INFINITY; total])
    }

    /// After a cost pass's last band (and the order): the λ and median from
    /// the samples (first group), then the group's lanes.
    fn maybe_end_cost_pass(&mut self) -> Result<(), StegoError> {
        if !self.pass_open || self.bands_left > 0 {
            return Ok(());
        }
        self.ensure_order();
        self.pass_open = false;
        if matches!(self.costs, GroupCosts::Gathering { .. })
            && let GroupCosts::Gathering { costs, .. } = std::mem::replace(&mut self.costs, GroupCosts::None)
        {
            self.costs = GroupCosts::Lanes(costs);
        }
        if self.group == 0 {
            let ord = self.ord.as_ref().expect("order");
            let sample_rho: Vec<f32> = match &mut self.costs {
                GroupCosts::Dense(dense) => self.samples.iter().map(|&k| dense[ord[k] as usize]).collect(),
                GroupCosts::Lanes(costs) => {
                    let (_, head) = group_head(self.n, self.l, &self.groups[0]);
                    costs.split_off(head)
                }
                _ => return Err(bad()),
            };
            let (lam, median_cost, sample_ab) = v31::sample_stats(
                ord,
                &self.samples,
                self.n_med,
                &sample_rho,
                self.m,
                self.img.dct_grid(0),
                None,
                self.wet.as_ref(),
                self.si.as_ref(),
                &self.pairs,
            );
            self.lanes = Some(Lanes {
                lam,
                median_cost,
                sample_ab,
                states: (0..self.l).map(|_| None).collect(),
                plans: (0..self.l).map(|_| None).collect(),
                next_lane: 0,
                left: 0,
            });
        }
        let lanes = self.lanes.as_mut().ok_or_else(bad)?;
        lanes.left = self.groups[self.group].len();
        Ok(())
    }

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

    /// What lane `k` reads: the cover, its slice of the order and the
    /// current group's costs.
    fn lane_input(&self, k: usize) -> LaneInput<'_> {
        let ord = self.ord.as_ref().expect("lanes start after the order");
        let lr = v31::share(self.n, self.l, k);
        let rho = match &self.costs {
            GroupCosts::Dense(d) => Rho::Flat(d),
            GroupCosts::Lanes(costs) => {
                let (r0, _) = group_head(self.n, self.l, &self.groups[self.group]);
                Rho::Lane(&costs[lr.start - r0..lr.end - r0])
            }
            _ => unreachable!("lanes run with their group's costs"),
        };
        LaneInput {
            grid: self.img.dct_grid(0),
            overlay: None,
            wet: self.wet.as_ref(),
            si: self.si.as_ref(),
            ord: &ord[lr],
            rho,
            pairs: &self.pairs,
        }
    }

    fn lane_bits(&self, k: usize) -> &[u8] {
        &self.frame_bits[v31::share(self.m, self.l, k)]
    }

    fn layer_msg(&self, k: usize, geom: &LaneGeom, layer: u8) -> &[u8] {
        let bits = self.lane_bits(k);
        if layer == 1 { &bits[..geom.m1] } else { &bits[geom.m1..] }
    }

    fn active_mut(&mut self, k: usize, layer: u8) -> Result<&mut ActiveLayer, StegoError> {
        let lanes = self.lanes.as_mut().ok_or_else(bad)?;
        let act = lanes.states.get_mut(k).and_then(|s| s.as_mut()).and_then(|s| s.active.as_mut()).ok_or_else(bad)?;
        if act.layer != layer {
            return Err(bad());
        }
        Ok(act)
    }

    /// A lane layer (layer 2 first, unless the lane has none), to be handed
    /// out whole if its input fits what the plan lets a lane's trellis hold
    /// (`trellis::working_bytes` of the one-call encoder's share for it:
    /// the task is written here one at a time), else split; split also
    /// while fewer lanes are left than helpers and its back pointers do not
    /// fit a helper's share.
    fn build_layer(&self, geom: &LaneGeom, layer: u8) -> ActiveLayer {
        let lanes = self.lanes.as_ref().expect("lanes started");
        let group_lanes = self.groups[self.group].len();
        let n = if layer == 2 { geom.n2() } else { geom.n1() };
        let in_flight = self.helpers.min(group_lanes);
        let limit = self.inline_limit.min(v31::lane_inline_share(self.n, geom.nk, in_flight, self.h, self.helpers));
        let (allowance, _) = self.lane_allowance(geom, layer);
        let fits = input_bytes(n) <= allowance.min(self.task_cap);
        let split = !fits || (trellis::inline_bytes(n, self.h) > limit && lanes.left < self.helpers);
        ActiveLayer {
            layer,
            code: v31::layer_code(&self.seeds, layer, geom, self.h),
            cover: self.caps.atomic_bits(n),
            stego: self.caps.bits(n),
            run: LayerRun::Pending,
            limit,
            split,
        }
    }

    /// What the plan lets one lane's trellis of layer `layer` hold
    /// (`trellis::working_bytes` with the one-call encoder's share for the
    /// lanes it plans at once), and those lanes: the most the coordinator
    /// spends on a lane's task and, for all its lanes in flight, their kept
    /// layer-1 inputs.
    fn lane_allowance(&self, geom: &LaneGeom, layer: u8) -> (usize, usize) {
        let group_lanes = self.groups[self.group].len();
        let shape = if layer == 2 { geom.s2 } else { geom.s1 }.expect("the layer exists");
        let nb = self.parallel_lanes.min(self.workers).clamp(1, group_lanes);
        let own = self.inline_limit.min(v31::lane_inline_share(self.n, geom.nk, nb, self.h, self.workers));
        (trellis::working_bytes(shape.n(), shape.avg_w(), self.h, own, v31::lane_threads(nb, self.workers)), nb)
    }

    /// Whether a lane keeps layer 1's input from its layer-2 pass (one
    /// random-access pass over the cover and costs instead of two): when
    /// layer 2 covers layer 1's positions, both layers' tasks fit whole, and
    /// the kept inputs of the lanes in flight with one task fit the plan's
    /// working set of its lanes.
    fn keeps_layer1(&self, geom: &LaneGeom) -> bool {
        if geom.m2 == 0 || geom.n2() < geom.n1() {
            return false;
        }
        let in_flight = self.helpers.min(self.groups[self.group].len());
        let (a1, nb) = self.lane_allowance(geom, 1);
        let (a2, _) = self.lane_allowance(geom, 2);
        let task = input_bytes(geom.n1().max(geom.n2()));
        task <= a1.min(a2).min(self.task_cap)
            && in_flight.saturating_mul(Layer1Keep::bytes(geom.n2())).saturating_add(task) <= nb.saturating_mul(a1.min(a2))
    }

    /// The next trellis task of the current group: steps of split layers
    /// first (they keep their chains moving), then pending layers, then a
    /// new lane.
    fn next_lane_task(&mut self) -> Result<Option<Vec<u8>>, StegoError> {
        if self.pass_open || self.lanes.as_ref().is_none_or(|x| x.left == 0) {
            return Ok(None);
        }
        let ks = self.groups[self.group].clone();
        for k in ks.clone() {
            if let Some(t) = self.split_task(k)? {
                return Ok(Some(t));
            }
        }
        for k in ks.clone() {
            if let Some(t) = self.pending_task(k)? {
                return Ok(Some(t));
            }
        }
        let k = self.lanes.as_ref().expect("lanes started").next_lane;
        if k < ks.end {
            let geom = LaneGeom::new(v31::share(self.n, self.l, k).len(), self.lane_bits(k).len())?;
            let first = if geom.m2 > 0 { 2 } else { 1 };
            let act = self.build_layer(&geom, first);
            let flip2 = self.caps.bits(geom.nk);
            // Sized for every position layer 2 hands out (it records each).
            let keep = self.keeps_layer1(&geom).then(|| Layer1Keep::new(geom.n2()));
            let lanes = self.lanes.as_mut().expect("lanes started");
            lanes.next_lane += 1;
            lanes.states[k] = Some(LaneState { geom, flip2, keep, active: Some(act) });
            return self.pending_task(k);
        }
        Ok(None)
    }

    /// Write a task carrying blocks `[b0, b1)` of lane `k`'s active layer
    /// (their input computed now), `extra` bytes reserved after it.
    fn layer_task(&self, k: usize, hd: Header, pre: &[u8], b0: usize, b1: usize, extra: usize) -> Vec<u8> {
        let lanes = self.lanes.as_ref().expect("lanes started");
        let st = lanes.states[k].as_ref().expect("lane started");
        let act = st.active.as_ref().expect("active layer");
        let msg = &self.layer_msg(k, &st.geom, act.layer)[b0..b1];
        let inp = self.lane_input(k);
        let input = v31::lane_layer_input(&inp, act.layer, lanes.lam, &st.flip2, &act.cover, st.keep.as_ref());
        let mut w = W::new(hd, pre.len() + layer_task_bytes(&act.code, b0, b1 - b0, extra));
        w.bytes(pre);
        write_layer(&mut w, &act.code, b0, msg, &input);
        w.0
    }

    /// A pending layer's first task: the whole-layer trellis, or the first
    /// forward step of a split layer.
    fn pending_task(&mut self, k: usize) -> Result<Option<Vec<u8>>, StegoError> {
        let h = self.h;
        let Some(lanes) = self.lanes.as_ref() else { return Ok(None) };
        let Some(st) = lanes.states[k].as_ref() else { return Ok(None) };
        let Some(act) = st.active.as_ref() else { return Ok(None) };
        if !matches!(act.run, LayerRun::Pending) {
            return Ok(None);
        }
        let layer = act.layer;
        if act.split {
            let m = act.code.m();
            let kb = segment_blocks(&act.code);
            let nseg = m.div_ceil(kb);
            let act = self.active_mut(k, layer)?;
            act.run = LayerRun::Split(Box::new(Split {
                k: kb,
                nseg,
                checkpoints: trellis::initial_costs(h),
                forward: 0,
                forward_out: false,
                next_issue: None,
                ready: BTreeMap::new(),
                trace_seg: 0,
                trace_state: 0,
                tracing: false,
                trace_holder: 0,
            }));
            return self.split_task(k);
        }
        let m = act.code.m();
        let mut pre = vec![h as u8];
        pre.extend_from_slice(&(act.limit as u64).to_le_bytes());
        let hd = Header { kind: TASK_TRELLIS, lane: k as u32, layer, seg: 0, target: -1 };
        let task = self.layer_task(k, hd, &pre, 0, m, 0);
        self.active_mut(k, layer)?.run = LayerRun::Out;
        if layer == 1 {
            // Layer 1's input is written: its kept copy is no longer needed.
            let lanes = self.lanes.as_mut().expect("lanes started");
            lanes.states[k].as_mut().expect("lane started").keep = None;
        }
        Ok(Some(task))
    }

    /// The next task of a split layer: its next forward step; after the
    /// chain, the trace of the next segment once its rows exist, else the
    /// next segment (while fewer than two per helper are untraced).
    fn split_task(&mut self, k: usize) -> Result<Option<Vec<u8>>, StegoError> {
        let h = self.h;
        let ns = 1usize << h;
        let Some(lanes) = self.lanes.as_ref() else { return Ok(None) };
        let Some(st) = lanes.states[k].as_ref() else { return Ok(None) };
        let Some(act) = st.active.as_ref() else { return Ok(None) };
        let LayerRun::Split(sp) = &act.run else { return Ok(None) };
        let (layer, m) = (act.layer, act.code.m());
        let span = |seg: usize| (seg * sp.k, ((seg + 1) * sp.k).min(m));
        if sp.forward < sp.nseg {
            if sp.forward_out {
                return Ok(None);
            }
            let seg = sp.forward;
            let (b0, b1) = span(seg);
            let hd = Header { kind: TASK_FORWARD, lane: k as u32, layer, seg: seg as u32, target: -1 };
            let mut task = self.layer_task(k, hd, &[h as u8], b0, b1, ns * 8);
            let LayerRun::Split(sp) = &mut self.active_mut(k, layer)?.run else { unreachable!() };
            for c in &sp.checkpoints[seg * ns..(seg + 1) * ns] {
                task.extend_from_slice(&c.to_le_bytes());
            }
            sp.forward_out = true;
            return Ok(Some(task));
        }
        if !sp.tracing
            && let Some(&holder) = sp.ready.get(&sp.trace_seg)
        {
            let seg = sp.trace_seg;
            let hd = Header { kind: TASK_TRACE, lane: k as u32, layer, seg: seg as u32, target: holder };
            let mut wr = W::new(hd, 4);
            wr.u32(sp.trace_state as u32);
            let LayerRun::Split(sp) = &mut self.active_mut(k, layer)?.run else { unreachable!() };
            sp.tracing = true;
            sp.trace_holder = holder as usize;
            sp.ready.remove(&seg);
            return Ok(Some(wr.0));
        }
        let Some(seg) = sp.next_issue else { return Ok(None) };
        let untraced = sp.trace_seg - seg;
        if untraced >= 2 * self.helpers {
            return Ok(None);
        }
        let (b0, b1) = span(seg);
        // Round robin from the last segment: each helper computes its
        // segments in the order the trace chain needs them, so the chain
        // waits at most for one segment per helper (with random placement it
        // kept waiting for helpers busy with segments it needed much later).
        let target = ((sp.nseg - 1 - seg) % self.helpers) as i32;
        // A helper holds at most HELD_SEGMENTS_PER_HELPER segments of all
        // split layers (it bounds its memory). Never a deadlock: a layer's
        // next segment to trace is the first it issued, so it is out or
        // held, and its trace frees a slot.
        if self.held[target as usize] >= HELD_SEGMENTS_PER_HELPER {
            return Ok(None);
        }
        let hd = Header { kind: TASK_SEGMENT, lane: k as u32, layer, seg: seg as u32, target };
        let mut task = self.layer_task(k, hd, &[h as u8], b0, b1, ns * 8);
        let LayerRun::Split(sp) = &mut self.active_mut(k, layer)?.run else { unreachable!() };
        for c in &sp.checkpoints[seg * ns..(seg + 1) * ns] {
            task.extend_from_slice(&c.to_le_bytes());
        }
        sp.next_issue = seg.checked_sub(1);
        self.held[target as usize] += 1;
        Ok(Some(task))
    }

    /// A lane layer's stego bits are complete: start layer 1, or finish the
    /// lane (and the group, and the encode after the last group).
    fn layer_done(&mut self, k: usize) -> Result<(), StegoError> {
        let lanes = self.lanes.as_mut().ok_or_else(bad)?;
        let mut st = lanes.states[k].take().ok_or_else(bad)?;
        let act = st.active.take().ok_or_else(bad)?;
        let ActiveLayer { layer, cover, mut stego, .. } = act;
        v31::flips_in_place(&mut stego, &cover);
        drop(cover);
        if layer == 2 {
            st.flip2[..stego.len()].copy_from_slice(&stego);
            drop(stego);
            let next = self.build_layer(&st.geom, 1);
            st.active = Some(next);
            self.lanes.as_mut().expect("lanes").states[k] = Some(st);
            return Ok(());
        }
        let plan = v31::lane_plan_bits(&self.lane_input(k), &st.geom, &st.flip2, &stego);
        drop((stego, st));
        let lanes = self.lanes.as_mut().expect("lanes");
        lanes.plans[k] = Some(plan);
        lanes.left -= 1;
        if lanes.left > 0 {
            return Ok(());
        }
        if self.group + 1 < self.groups.len() {
            self.group += 1;
            self.costs = GroupCosts::None;
            self.start_cost_pass();
            return Ok(());
        }
        self.apply()
    }

    fn apply(&mut self) -> Result<(), StegoError> {
        self.costs = GroupCosts::None;
        let lanes = self.lanes.take().ok_or_else(bad)?;
        let ord = self.ord.take().ok_or_else(bad)?;
        let plans: Vec<LanePlan> = lanes.plans.into_iter().map(|p| p.ok_or_else(bad)).collect::<Result<_, _>>()?;
        let p = v31::assemble(plans, &ord, self.m, lanes.median_cost, &lanes.sample_ab, self.bound);
        self.changes = p.changes;
        self.stats = Some(p.stats);
        Ok(())
    }
}

/// The first order position of a group's lanes and their positions.
fn group_head(n: usize, l: usize, ks: &Range<usize>) -> (usize, usize) {
    let r0 = v31::share(n, l, ks.start).start;
    (r0, v31::share(n, l, ks.end - 1).end - r0)
}

/// Flat index of a group's request position `i`: its positions (order
/// `r0 + i`), then the samples.
#[inline]
fn request_fi(ord: &[u32], r0: usize, head: usize, samples: &[usize], i: usize) -> u32 {
    if i < head { ord[r0 + i] } else { ord[samples[i - head]] }
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
/// Also returns how many tasks of each kind ran (index = kind). `task_cap`:
/// the largest whole-layer task (`Some(0)`: every layer split).
#[doc(hidden)]
pub fn run_split_encode_locally(
    image_bytes: &[u8],
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
    pixels: Option<(&[u8], u32, u32)>,
    helpers: usize,
    task_cap: Option<usize>,
) -> Result<(Vec<u8>, EncodeQuality, [usize; 6]), StegoError> {
    let mut counts = [0usize; 6];
    let mut s = GhostEncodeSession::new(image_bytes, message, files, passphrase, pixels, helpers)?;
    if let Some(c) = task_cap {
        s.task_cap = c;
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
    let (bytes, q) = s.finish(image_bytes)?;
    Ok((bytes, q, counts))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A wrong cover handed to `finish` is the host's fault: the session says
    /// so before it is given up (`cover_matches`), and `finish` reports a
    /// protocol error, not the cover's (`is_protocol_error`).
    #[test]
    fn wrong_cover_at_finish_is_a_protocol_error() {
        let cover = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/test-vectors/image/photo_640x480_q75_420.jpg")).unwrap();
        let other = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/test-vectors/image/photo_320x240_q75_420.jpg")).unwrap();
        let mut s = GhostEncodeSession::new(&cover, "protocol", &[], "pw", None, 2).unwrap();
        let mut runners = [GhostTaskRunner::new(), GhostTaskRunner::new()];
        let mut queue = s.take_tasks().unwrap();
        s.run_local().unwrap();
        while !s.is_done() {
            queue.extend(s.take_tasks().unwrap());
            let task = queue.remove(0);
            let target = i32::from_le_bytes(task[TASK_TARGET_OFFSET..TASK_TARGET_OFFSET + 4].try_into().unwrap());
            let h = target.max(0) as usize;
            let result = runners[h].run(task).unwrap();
            s.submit(&result, h as i32).unwrap();
        }
        assert!(s.cover_matches(&cover) && !s.cover_matches(&other));
        let e = s.finish(&other).unwrap_err();
        assert!(is_protocol_error(&e), "{e:?}");
        assert!(!is_protocol_error(&StegoError::InvalidJpeg(crate::codec::jpeg::error::JpegError::InvalidMarkerData("other"))));
        assert!(!is_protocol_error(&StegoError::MessageTooLarge));
    }

    /// A result reported for a helper the session does not have is a
    /// protocol error (the index picks the helper that keeps segment rows).
    #[test]
    fn result_from_an_unknown_helper_is_a_protocol_error() {
        let cover = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/test-vectors/image/photo_640x480_q75_420.jpg")).unwrap();
        let mut s = GhostEncodeSession::new(&cover, "protocol", &[], "pw", None, 2).unwrap();
        let result = W::new(Header { kind: TASK_SEGMENT, lane: 0, layer: 2, seg: 0, target: -1 }, 0).0;
        for helper in [-1, 2, 99] {
            let e = s.submit(&result, helper).unwrap_err();
            assert!(is_protocol_error(&e), "helper {helper}: {e:?}");
        }
    }
}
