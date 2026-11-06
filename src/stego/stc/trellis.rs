// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Syndrome-trellis embedding for any constraint height `h` (Ghost v3.1).
//!
//! [`super::embed::stc_embed`] packs one trellis step's back pointers into a
//! `u128`, which limits it to `h ≤ 7`. This module stores them as `2^h / 64`
//! words per step, so the height is a parameter. For `h ≤ 7` both produce the
//! same plan bit for bit (same f64 accumulation, same `c1 < c0` tie-break,
//! same lowest-index terminal state; `matches_stc_embed_at_h7`).
//!
//! The block structure is a [`Code`]: message bit `b` is the syndrome of the
//! positions `[start(b), start(b + 1))` under its block's H-hat columns. The
//! constant-width code ([`Code::uniform`], one message bit per `w`
//! positions) is the classic STC; Ghost v3.1 uses *mixed widths* (blocks of
//! `w` and `w + 1` positions spread evenly), so a code of `m` bits uses every
//! one of its `n` positions however `n / m` falls (Filler, Judas, Fridrich
//! 2011, the reference implementation's non-integer `1/α`).
//!
//! Every memory schedule gives the same plan (the forward recursion is the
//! same arithmetic wherever it runs):
//!
//! - **inline**: one forward pass that keeps every step's back pointers
//!   (`n · max(2^h / 8, 8)` bytes);
//! - **checkpointed**: a forward pass that keeps only the state costs at
//!   every `K`-th message boundary, then segment by segment from the end:
//!   recompute that segment's back pointers from its checkpoint and trace it
//!   back. Two refinements cut the second pass:
//!   - *tail*: the forward pass keeps the back pointers of as many final
//!     blocks as the in-memory limit holds (traced back directly), and only
//!     the blocks before them are recomputed, in segments that reuse the tail
//!     buffer — `1 + head/total` of the inline work;
//!   - *waves*: with idle workers (`threads > 1`) up to `threads` segments
//!     are recomputed at once, each into its own part of the buffer, then
//!     traced back in order — `1 + 1/threads` of the inline work.
//!
//! [`embed`] / [`embed_code`] use the inline schedule when the back-pointer
//! table fits `inline_limit_bytes`.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::stego::error::StegoError;
use crate::stego::progress;

/// Largest supported constraint height.
pub const MAX_H: usize = 12;

/// Positions between progress ticks / cancellation checks.
const TICK: usize = 1 << 16;

/// A planned embedding.
#[derive(Debug, Clone)]
pub struct Plan {
    /// Stego bit of each of the `n` walked positions.
    pub stego_bits: Vec<u8>,
    /// Σ cost of the flipped positions (the trellis optimum).
    pub total_cost: f64,
    /// Number of flipped positions.
    pub changes: usize,
}

/// Block structure of a syndrome-trellis code: message bit `b` is the
/// syndrome of positions `[start(b), start(b + 1))` under the H-hat columns
/// of its block.
///
/// - The first `head` blocks have `head_cols.len()` positions each.
/// - The other `m − head` blocks share the next `rest_n` positions evenly:
///   block `head + r` starts at `head · w_head + ⌊r · rest_n / (m − head)⌋`,
///   so its width is `⌊rest_n / (m − head)⌋` (columns `short_cols`) or one
///   more (`long_cols`).
///
/// [`Code::uniform`] is the constant-width code.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Code {
    m: usize,
    head: usize,
    head_cols: Vec<u32>,
    rest_n: usize,
    short_cols: Vec<u32>,
    long_cols: Vec<u32>,
}

impl Code {
    /// `m` blocks of `cols.len()` positions, all with columns `cols`.
    pub fn uniform(cols: &[u32], m: usize) -> Self {
        Self { m, head: 0, head_cols: Vec::new(), rest_n: m * cols.len(), short_cols: cols.to_vec(), long_cols: Vec::new() }
    }

    /// The code described in the type docs; `None` if the column sets do not
    /// match the widths (`short_cols.len() = ⌊rest_n / (m − head)⌋ ≥ 1`,
    /// `long_cols.len() = short + 1` when some blocks are long, `head_cols`
    /// non-empty when `head > 0`).
    pub fn new(
        m: usize,
        head: usize,
        head_cols: Vec<u32>,
        rest_n: usize,
        short_cols: Vec<u32>,
        long_cols: Vec<u32>,
    ) -> Option<Self> {
        if head > m || (head > 0 && head_cols.is_empty()) {
            return None;
        }
        let rm = m - head;
        if rm > 0 {
            let s = rest_n / rm;
            if s == 0 || short_cols.len() != s || (rest_n % rm != 0 && long_cols.len() != s + 1) {
                return None;
            }
        }
        Some(Self { m, head, head_cols, rest_n, short_cols, long_cols })
    }

    /// Message bits (blocks).
    pub fn m(&self) -> usize {
        self.m
    }

    /// Positions the code covers (`start(m)`).
    pub fn n(&self) -> usize {
        self.start(self.m)
    }

    /// Leading blocks of the head width, the head width.
    pub fn head(&self) -> (usize, usize) {
        (self.head, self.head_cols.len())
    }

    /// Positions shared by the blocks after the head.
    pub fn rest_n(&self) -> usize {
        self.rest_n
    }

    /// Column sets: head, short, long.
    pub fn column_sets(&self) -> [&[u32]; 3] {
        [&self.head_cols, &self.short_cols, &self.long_cols]
    }

    /// First position of block `b` (`b ≤ m`).
    #[inline]
    pub fn start(&self, b: usize) -> usize {
        let hw = self.head_cols.len();
        if b <= self.head {
            b * hw
        } else {
            let rm = (self.m - self.head) as u128;
            self.head * hw + ((b - self.head) as u128 * self.rest_n as u128 / rm) as usize
        }
    }

    /// Positions of block `b`.
    #[inline]
    pub fn width(&self, b: usize) -> usize {
        self.start(b + 1) - self.start(b)
    }

    /// Index into [`Self::column_sets`] of block `b`'s columns.
    #[inline]
    fn set_of(&self, b: usize, width: usize) -> usize {
        if b < self.head {
            0
        } else if width == self.short_cols.len() {
            1
        } else {
            2
        }
    }

    /// H-hat columns of block `b`.
    pub fn cols(&self, b: usize) -> &[u32] {
        self.column_sets()[self.set_of(b, self.width(b))]
    }

    /// Widest block.
    pub fn max_width(&self) -> usize {
        let rest = if self.m > self.head { self.short_cols.len() + usize::from(!self.long_cols.is_empty()) } else { 0 };
        self.head_cols.len().max(rest).max(1)
    }

    fn masked_sets(&self, h: usize) -> [Vec<usize>; 3] {
        [masked_cols(&self.head_cols, h), masked_cols(&self.short_cols, h), masked_cols(&self.long_cols, h)]
    }
}

/// Back-pointer words per trellis step.
#[inline]
pub fn words_per_step(h: usize) -> usize {
    ((1usize << h) / 64).max(1)
}

/// Bytes of the inline schedule's back-pointer table for `positions` steps.
#[inline]
pub fn inline_bytes(positions: usize, h: usize) -> usize {
    positions.saturating_mul(words_per_step(h) * 8)
}

/// The checkpointed schedule of a trellis whose table does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Schedule {
    /// Final message blocks whose back pointers the forward pass keeps.
    tail: usize,
    /// Message blocks per recomputed segment of the blocks before the tail.
    seg: usize,
    /// Segments recomputed at once (one buffer part each).
    wave: usize,
}

impl Schedule {
    /// Back-pointer buffer, in message blocks (the tail, then the waves).
    fn buffer_blocks(&self) -> usize {
        self.tail.max(self.wave * self.seg)
    }
}

/// Schedule of a trellis of `m` blocks of (about) `w` steps, the widest
/// `w_max`; `None` = inline.
fn schedule(m: usize, n: usize, w: usize, w_max: usize, h: usize, inline_limit_bytes: usize, threads: usize) -> Option<Schedule> {
    if inline_bytes(n, h) <= inline_limit_bytes {
        return None;
    }
    // m ≥ 1 here (an empty table always fits).
    let threads = if cfg!(feature = "parallel") { threads.max(1) } else { 1 };
    if threads == 1 {
        // A tail at least as long as the optimal segment of the rest also
        // serves as the segment buffer.
        let tail = (inline_limit_bytes / inline_bytes(w_max, h).max(1)).min(m - 1);
        if tail > 0 && tail >= segment_blocks(m - tail, w, h, 1) {
            return Some(Schedule { tail, seg: tail, wave: 1 });
        }
        return Some(Schedule { tail: 0, seg: segment_blocks(m, w, h, 1), wave: 1 });
    }
    let seg = segment_blocks(m, w, h, threads);
    Some(Schedule { tail: 0, seg, wave: threads.min(m.div_ceil(seg)) })
}

/// Schedule of `code`.
fn code_schedule(code: &Code, h: usize, inline_limit_bytes: usize, threads: usize) -> Option<Schedule> {
    let (m, n) = (code.m(), code.n());
    schedule(m, n, n.div_ceil(m.max(1)).max(1), code.max_width(), h, inline_limit_bytes, threads)
}

/// Peak working memory of [`embed`] for `positions` steps of width `w`
/// (back pointers or segment buffers + checkpoints, excluding the caller's
/// inputs and the returned plan). For a mixed-width code pass its average
/// width: an estimate within one position per block.
pub fn working_bytes(positions: usize, w: usize, h: usize, inline_limit_bytes: usize, threads: usize) -> usize {
    let w = w.max(1);
    let m = positions / w;
    match schedule(m, positions, w, w, h, inline_limit_bytes, threads) {
        None => inline_bytes(positions, h),
        Some(s) => inline_bytes(s.buffer_blocks() * w, h) + (m - s.tail).div_ceil(s.seg) * (1usize << h) * 8,
    }
}

/// Positions an [`embed`] call walks (forward passes and recomputation):
/// `positions` inline, up to twice that checkpointed.
pub fn walked_positions(positions: usize, w: usize, h: usize, inline_limit_bytes: usize, threads: usize) -> usize {
    let w = w.max(1);
    let m = positions / w;
    match schedule(m, positions, w, w, h, inline_limit_bytes, threads) {
        None => positions,
        Some(s) => positions + (m - s.tail) * w,
    }
}

/// Message blocks per segment `K` with `p` segment buffers, minimising
/// buffers + checkpoints: `p·K·w·(2^h/8)` bytes of back pointers against
/// `(m/K)·2^h·8` bytes of checkpointed state costs, i.e.
/// `K = √(m·2^h / (p·w·words))`. (A fixed `K ≥ √m` blew the segment buffer up
/// for very wide `w`: a 64-byte frame on a 48 MP cover is one lane of
/// `w` ≈ 190 k, and 16 blocks of it are 196 MB.)
fn segment_blocks(m: usize, w: usize, h: usize, p: usize) -> usize {
    let ns = (1usize << h) as f64;
    let k = ((m as f64) * ns / ((p.max(1) * w.max(1) * words_per_step(h)) as f64)).sqrt().round() as usize;
    k.clamp(1, m.max(1))
}

/// Plan the minimum-cost stego bits whose syndrome under the H-hat columns
/// `cols` (bit `r` of `cols[c]` = row `r`, column `c`) equals `message`:
/// [`embed_code`] with the constant-width code [`Code::uniform`].
#[allow(clippy::too_many_arguments)]
pub fn embed(
    cover: &[u8],
    costs: &[f32],
    message: &[u8],
    cols: &[u32],
    h: usize,
    inline_limit_bytes: usize,
    threads: usize,
    tick: &(dyn Fn(usize) + Sync),
) -> Result<Option<Plan>, StegoError> {
    if cols.is_empty() {
        return Ok(None);
    }
    embed_code(cover, costs, message, &Code::uniform(cols, message.len()), h, inline_limit_bytes, threads, tick)
}

/// Plan the minimum-cost stego bits whose syndrome under `code` equals
/// `message` (`code.m()` bits).
///
/// - `cover`: cover bits (LSB used), at least `code.n()`.
/// - `costs`: flip cost per position (`f32::INFINITY` = wet), same length.
/// - `inline_limit_bytes`: back-pointer bytes the call may keep (above:
///   checkpointed schedule).
/// - `threads`: workers the checkpointed schedule may use for recomputation
///   (rayon; 1 = this thread only). The plan does not depend on it.
/// - `tick(n)`: called with the number of positions processed (summing to
///   exactly `code.n()` over the whole call, in any schedule; from any
///   thread).
///
/// `Ok(None)` if infeasible (every path hits a wet position against its
/// constraint), the cover is shorter than `code.n()` or `message` does not
/// have `code.m()` bits; `Err(Cancelled)` if the encode channel was
/// cancelled.
#[allow(clippy::too_many_arguments)]
pub fn embed_code(
    cover: &[u8],
    costs: &[f32],
    message: &[u8],
    code: &Code,
    h: usize,
    inline_limit_bytes: usize,
    threads: usize,
    tick: &(dyn Fn(usize) + Sync),
) -> Result<Option<Plan>, StegoError> {
    let m = message.len();
    if h == 0 || h > MAX_H || m != code.m() {
        return Ok(None);
    }
    let n = code.n();
    if n > cover.len() || n > costs.len() {
        return Ok(None);
    }
    if m == 0 {
        return Ok(Some(Plan { stego_bits: Vec::new(), total_cost: 0.0, changes: 0 }));
    }
    let sets = code.masked_sets(h);
    let t = Trellis { cover: &cover[..n], costs: &costs[..n], message, code, sets: &sets, b_off: 0, p_off: 0, h };
    match code_schedule(code, h, inline_limit_bytes, threads) {
        None => t.inline(&Progress::new(n, n, tick)),
        Some(s) => t.checkpointed(s, &Progress::new(n + t.pos(m - s.tail), n, tick)),
    }
}

/// A trellis over the blocks `[b_off, b_off + message.len())` of `code`:
/// local block `b` is code block `b_off + b`, local position `j` is code
/// position `p_off + j` (`p_off = code.start(b_off)`).
struct Trellis<'a> {
    cover: &'a [u8],
    costs: &'a [f32],
    message: &'a [u8],
    code: &'a Code,
    sets: &'a [Vec<usize>; 3],
    b_off: usize,
    p_off: usize,
    h: usize,
}

/// Maps the positions walked (`walked`, all passes) onto `report` progress
/// units, exactly, from any thread.
struct Progress<'a> {
    walked: u64,
    report: u64,
    done: AtomicU64,
    tick: &'a (dyn Fn(usize) + Sync),
}

impl<'a> Progress<'a> {
    fn new(walked: usize, report: usize, tick: &'a (dyn Fn(usize) + Sync)) -> Self {
        Self { walked: walked.max(1) as u64, report: report as u64, done: AtomicU64::new(0), tick }
    }

    fn add(&self, k: usize) {
        let before = self.done.fetch_add(k as u64, Ordering::Relaxed);
        let at = |d: u64| (d.min(self.walked) as u128 * self.report as u128 / self.walked as u128) as u64;
        let r = at(before + k as u64) - at(before);
        if r > 0 {
            (self.tick)(r as usize);
        }
    }

    fn ticker(&self) -> Ticker<'_> {
        Ticker { pending: 0, progress: self }
    }
}

/// Per-thread step counter: reports every [`TICK`] positions and checks for
/// cancellation.
struct Ticker<'a> {
    pending: usize,
    progress: &'a Progress<'a>,
}

impl Ticker<'_> {
    #[inline]
    fn step(&mut self) -> Result<(), StegoError> {
        self.pending += 1;
        if self.pending == TICK {
            self.flush()?;
        }
        Ok(())
    }

    fn flush(&mut self) -> Result<(), StegoError> {
        if self.pending > 0 {
            self.progress.add(self.pending);
            self.pending = 0;
        }
        if progress::is_cancelled() {
            return Err(StegoError::Cancelled);
        }
        Ok(())
    }
}

impl Trellis<'_> {
    #[inline]
    fn costs_at(&self, j: usize) -> (f64, f64) {
        let c = self.costs[j] as f64;
        if self.cover[j] & 1 == 0 { (0.0, c) } else { (c, 0.0) }
    }

    /// Local first position of local block `b` (`b ≤ message.len()`).
    #[inline]
    fn pos(&self, b: usize) -> usize {
        self.code.start(self.b_off + b) - self.p_off
    }

    /// Local block `b`: first local position, width, columns.
    #[inline]
    fn block(&self, b: usize) -> (usize, usize, &[usize]) {
        let g = self.b_off + b;
        let s0 = self.code.start(g);
        let w = self.code.start(g + 1) - s0;
        (s0 - self.p_off, w, &self.sets[self.code.set_of(g, w)])
    }

    /// Forward over message blocks `[b0, b1)` from `prev` (state costs at the
    /// start of block `b0`); `rows`, if given, receives each step's back
    /// pointers (`words_per_step(h)` words per step, block `b0`'s first
    /// position first). On return `prev` holds the costs after block
    /// `b1 − 1`'s message shift.
    #[allow(clippy::too_many_arguments)]
    fn forward(
        &self,
        prev: &mut Vec<f64>,
        cur: &mut Vec<f64>,
        b0: usize,
        b1: usize,
        mut rows: Option<&mut [u64]>,
        on_block_start: &mut dyn FnMut(usize, &[f64]),
        ticker: &mut Ticker<'_>,
    ) -> Result<(), StegoError> {
        let ns = 1usize << self.h;
        let wps = words_per_step(self.h);
        let base = self.pos(b0);
        for b in b0..b1 {
            on_block_start(b, prev);
            let (j0, w, cols) = self.block(b);
            let block_rows = rows.as_deref_mut().map(|r| {
                let k = (j0 - base) * wps;
                &mut r[k..k + w * wps]
            });
            self.forward_block(prev, cur, j0, cols, block_rows, ticker)?;
            // Message shift: keep states whose low bit is the message bit.
            let bit = (self.message[b] & 1) as usize;
            for t in 0..ns / 2 {
                cur[t] = prev[2 * t + bit];
            }
            for v in cur[ns / 2..].iter_mut() {
                *v = f64::INFINITY;
            }
            std::mem::swap(prev, cur);
        }
        Ok(())
    }

    /// The trellis steps of one message block (its positions from `j0`, one
    /// per column of `cols`, before its message shift), back pointers into
    /// `rows` (`cols.len() · words_per_step(h)` words) if given.
    ///
    /// Not inlined on wasm32, like [`step`]: a trellis is one `forward` call,
    /// and V8 keeps a call in baseline code (Liftoff) until it returns, so
    /// the per-step bookkeeping ran unoptimized for a whole first pass
    /// (measured: pass A of a 12 MP single-lane layer 5.1 s cold, 2.5 s warm).
    #[cfg_attr(target_arch = "wasm32", inline(never))]
    fn forward_block(
        &self,
        prev: &mut Vec<f64>,
        cur: &mut Vec<f64>,
        j0: usize,
        cols: &[usize],
        mut rows: Option<&mut [u64]>,
        ticker: &mut Ticker<'_>,
    ) -> Result<(), StegoError> {
        let wps = words_per_step(self.h);
        for (c, &col) in cols.iter().enumerate() {
            let (c0, c1) = self.costs_at(j0 + c);
            match rows.as_deref_mut() {
                Some(r) => step::<true>(prev, cur, &mut r[c * wps..(c + 1) * wps], col, c0, c1),
                None => step::<false>(prev, cur, &mut [], col, c0, c1),
            }
            std::mem::swap(prev, cur);
            ticker.step()?;
        }
        Ok(())
    }

    /// Trace blocks `[b0, b1)` back from `state` (the state after block
    /// `b1 − 1`'s shift) through `rows` (back pointers of those blocks, from
    /// block `b0`'s first position), writing their stego bits into `out`
    /// (local positions). Returns the state at the start of `b0`.
    fn traceback(&self, rows: &[u64], b0: usize, b1: usize, mut state: usize, out: &mut [u8]) -> usize {
        let ns = 1usize << self.h;
        let mask = ns - 1;
        let wps = words_per_step(self.h);
        let base = self.pos(b0);
        for b in (b0..b1).rev() {
            state = ((state << 1) | (self.message[b] & 1) as usize) & mask;
            let (j0, w, cols) = self.block(b);
            for c in (0..w).rev() {
                let j = j0 + c;
                let k = (j - base) * wps;
                let (wi, bi) = bp_bit(state, ns);
                let bit = ((rows[k + wi] >> bi) & 1) as u8;
                out[j] = bit;
                if bit == 1 {
                    state ^= cols[c];
                }
            }
        }
        state
    }

    fn finish(&self, stego_bits: Vec<u8>, total_cost: f64) -> Plan {
        let changes = stego_bits.iter().zip(self.cover).filter(|(s, c)| (**s ^ **c) & 1 != 0).count();
        Plan { stego_bits, total_cost, changes }
    }

    fn initial(&self) -> (Vec<f64>, Vec<f64>) {
        let ns = 1usize << self.h;
        let mut prev = vec![f64::INFINITY; ns];
        prev[0] = 0.0;
        (prev, vec![0.0; ns])
    }

    fn inline(&self, progress: &Progress<'_>) -> Result<Option<Plan>, StegoError> {
        let m = self.message.len();
        let n = self.pos(m);
        let mut ticker = progress.ticker();
        let (mut prev, mut cur) = self.initial();
        let mut rows = vec![0u64; n * words_per_step(self.h)];
        self.forward(&mut prev, &mut cur, 0, m, Some(&mut rows), &mut |_, _| {}, &mut ticker)?;
        ticker.flush()?;
        let (best, cost) = best_state(&prev);
        if cost == f64::INFINITY {
            return Ok(None);
        }
        let mut bits = vec![0u8; n];
        let s0 = self.traceback(&rows, 0, m, best, &mut bits);
        debug_assert_eq!(s0, 0);
        Ok(Some(self.finish(bits, cost)))
    }

    fn checkpointed(&self, s: Schedule, progress: &Progress<'_>) -> Result<Option<Plan>, StegoError> {
        let m = self.message.len();
        let n = self.pos(m);
        let head = m - s.tail;
        let k = s.seg;
        let nseg = head.div_ceil(k);
        let wps = words_per_step(self.h);
        let span = |sg: usize| (sg * k, ((sg + 1) * k).min(head));
        // Buffer: the tail, or the widest wave (consecutive segments).
        let mut buf_pos = n - self.pos(head);
        let mut hi = nseg;
        while hi > 0 {
            let lo = hi.saturating_sub(s.wave);
            buf_pos = buf_pos.max(self.pos(span(hi - 1).1) - self.pos(span(lo).0));
            hi = lo;
        }
        let mut buf = vec![0u64; buf_pos * wps];
        let mut ticker = progress.ticker();

        // Pass A: checkpoints at every segment start of the head, back
        // pointers of the tail.
        let (mut prev, mut cur) = self.initial();
        let mut checkpoints: Vec<Vec<f64>> = Vec::with_capacity(nseg);
        self.forward(
            &mut prev,
            &mut cur,
            0,
            head,
            None,
            &mut |b, p| {
                if b % k == 0 {
                    checkpoints.push(p.to_vec());
                }
            },
            &mut ticker,
        )?;
        let tail_words = (n - self.pos(head)) * wps;
        self.forward(&mut prev, &mut cur, head, m, Some(&mut buf[..tail_words]), &mut |_, _| {}, &mut ticker)?;
        ticker.flush()?;
        let (best, cost) = best_state(&prev);
        if cost == f64::INFINITY {
            return Ok(None);
        }
        let mut bits = vec![0u8; n];
        let mut state = self.traceback(&buf[..tail_words], head, m, best, &mut bits);

        // Pass B: the head's segments from the end, `wave` at a time, back
        // pointers recomputed from their checkpoints.
        let mut hi = nseg;
        while hi > 0 {
            let lo = hi.saturating_sub(s.wave);
            let recompute = |sg: usize, rows: &mut [u64]| -> Result<(), StegoError> {
                let (b0, b1) = span(sg);
                let mut prev = checkpoints[sg].clone();
                let mut cur = vec![0.0; prev.len()];
                let mut t = progress.ticker();
                self.forward(&mut prev, &mut cur, b0, b1, Some(rows), &mut |_, _| {}, &mut t)?;
                t.flush()
            };
            // Each segment's rows in its own part of the buffer, in order
            // (consecutive segments: consecutive parts).
            let base = self.pos(span(lo).0);
            let mut parts: Vec<&mut [u64]> = Vec::with_capacity(hi - lo);
            let mut rest = &mut buf[..];
            for sg in lo..hi {
                let (b0, b1) = span(sg);
                let (part, r) = rest.split_at_mut((self.pos(b1) - self.pos(b0)) * wps);
                parts.push(part);
                rest = r;
            }
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                if hi - lo > 1 {
                    let res: Vec<Result<(), StegoError>> =
                        parts.into_par_iter().enumerate().map(|(i, rows)| recompute(lo + i, rows)).collect();
                    res.into_iter().collect::<Result<(), StegoError>>()?;
                } else {
                    for (i, rows) in parts.into_iter().enumerate() {
                        recompute(lo + i, rows)?;
                    }
                }
            }
            #[cfg(not(feature = "parallel"))]
            for (i, rows) in parts.into_iter().enumerate() {
                recompute(lo + i, rows)?;
            }
            for sg in (lo..hi).rev() {
                let (b0, b1) = span(sg);
                let off = (self.pos(b0) - base) * wps;
                let len = (self.pos(b1) - self.pos(b0)) * wps;
                state = self.traceback(&buf[off..off + len], b0, b1, state, &mut bits);
            }
            checkpoints.truncate(lo);
            hi = lo;
        }
        debug_assert_eq!(state, 0);
        Ok(Some(self.finish(bits, cost)))
    }
}

/// Lowest-index state of minimum cost (as `embed::find_best_state`).
fn best_state(costs: &[f64]) -> (usize, f64) {
    let mut best = 0;
    let mut best_cost = f64::INFINITY;
    for (s, &c) in costs.iter().enumerate() {
        if c < best_cost {
            best_cost = c;
            best = s;
        }
    }
    (best, best_cost)
}

/// Where state `s`'s back-pointer bit lives in a step's words. With 128 or
/// more states, each group of 128 states has two words, the even states'
/// bits and the odd states', pair `p` of the group at bit `63 − p` (the
/// order the kernels shift them in); fewer states use one word, state `s`
/// at bit `s`.
#[inline]
fn bp_bit(s: usize, ns: usize) -> (usize, u32) {
    if ns >= 128 { (((s >> 7) << 1) | (s & 1), 63 - ((s >> 1) & 63) as u32) } else { (0, s as u32) }
}

/// One trellis step over all `2^h` states:
/// `cur[s] = min(prev[s] + c0, prev[s ^ col] + c1)`, back-pointer bit `s` =
/// 1 iff the `c1` branch won (`c1 < c0` strictly, so ties keep `c0`). `ROWS =
/// false`: the same state costs without storing the back pointers.
///
/// Not inlined on wasm32: V8 starts every function in its baseline compiler
/// (Liftoff) and swaps in optimized code only at the next call, with no
/// on-stack replacement for Wasm. Inlined into `forward`, a trellis that is
/// one call long ran its whole first lane in baseline code (≈2× slower);
/// as its own function the kernel is optimized after a few thousand steps.
#[cfg_attr(target_arch = "wasm32", inline(never))]
#[cfg_attr(not(target_arch = "wasm32"), inline)]
fn step<const ROWS: bool>(prev: &[f64], cur: &mut [f64], row: &mut [u64], col: usize, c0: f64, c1: f64) {
    #[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
    {
        if prev.len() >= 128 {
            // SAFETY: NEON is enabled at compile time; every access is in
            // bounds (pairs `2p`, `2q` < len, len a power of two ≥ 128; `row`
            // holds `words_per_step` words when `ROWS`).
            unsafe { step_neon::<ROWS>(prev, cur, row, col, c0, c1) };
            return;
        }
    }
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        if prev.len() >= 128 {
            // SAFETY: SIMD128 is enabled at compile time (`.cargo/config.toml`);
            // bounds as for NEON.
            unsafe { step_simd128::<ROWS>(prev, cur, row, col, c0, c1) };
            return;
        }
    }
    step_scalar::<ROWS>(prev, cur, row, col, c0, c1);
}

#[inline]
fn step_scalar<const ROWS: bool>(prev: &[f64], cur: &mut [f64], row: &mut [u64], col: usize, c0: f64, c1: f64) {
    let ns = prev.len();
    #[inline(always)]
    fn one(prev: &[f64], cur: &mut [f64], s: usize, col: usize, c0: f64, c1: f64) -> u64 {
        let a = prev[s] + c0;
        let b = prev[s ^ col] + c1;
        let take = b < a;
        cur[s] = if take { b } else { a };
        take as u64
    }
    if ns >= 128 {
        // Layout of `bp_bit`: per 128 states, even and odd states' words.
        for g in 0..ns / 128 {
            let (mut even, mut odd) = (0u64, 0u64);
            for p in 0..64 {
                let s = 128 * g + 2 * p;
                let t0 = one(prev, cur, s, col, c0, c1);
                let t1 = one(prev, cur, s + 1, col, c0, c1);
                if ROWS {
                    even = (even << 1) | t0;
                    odd = (odd << 1) | t1;
                }
            }
            if ROWS {
                row[2 * g] = even;
                row[2 * g + 1] = odd;
            }
        }
        return;
    }
    let mut word = 0u64;
    for s in 0..ns {
        let t = one(prev, cur, s, col, c0, c1);
        if ROWS {
            word |= t << s;
        }
    }
    if ROWS {
        row[0] = word;
    }
}

/// NEON kernel: states in pairs. With `col`'s low bit set (always, for the
/// end-set H-hat) `prev[s ^ col]` for the pair `(2p, 2p+1)` is the pair
/// `2(p ^ col>>1)` with its lanes swapped. The comparison masks go into a
/// two-lane accumulator (even states, odd states), eight pairs at a time:
/// two words per 128 states, the layout of `bp_bit`.
#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
#[inline]
unsafe fn step_neon<const ROWS: bool>(prev: &[f64], cur: &mut [f64], row: &mut [u64], col: usize, c0: f64, c1: f64) {
    // SAFETY: see `step`.
    unsafe {
        use core::arch::aarch64::*;
        let ns = prev.len();
        let half = col >> 1;
        let vc0 = vdupq_n_f64(c0);
        let vc1 = vdupq_n_f64(c1);
        let pp = prev.as_ptr();
        let cp = cur.as_mut_ptr();
        // One pair: its new state costs stored, its comparison mask returned.
        macro_rules! pair {
            ($p:expr, $swap:expr) => {{
                let p = $p;
                let q = p ^ half;
                let mut gq = vld1q_f64(pp.add(2 * q));
                if $swap {
                    gq = vextq_f64(gq, gq, 1);
                }
                let a = vaddq_f64(vld1q_f64(pp.add(2 * p)), vc0);
                let b = vaddq_f64(gq, vc1);
                let mk = vcltq_f64(b, a);
                vst1q_f64(cp.add(2 * p), vbslq_f64(mk, b, a));
                mk
            }};
        }
        macro_rules! body {
            ($swap:expr) => {
                if ROWS {
                    // Eight pairs' mask bits at once (bits 7..0, first pair
                    // highest), then shifted into the accumulator: its
                    // dependency chain is one shift and one OR per eight
                    // pairs. Per pair (shift, shift-accumulate) the chain
                    // bounded the kernel: 22 % slower at h = 9 (M-series).
                    let k: [uint64x2_t; 8] = core::array::from_fn(|i| vdupq_n_u64(128 >> i));
                    for g in 0..ns / 128 {
                        let mut acc = vdupq_n_u64(0);
                        let mut p = 64 * g;
                        while p < 64 * g + 64 {
                            let m7 = vandq_u64(pair!(p, $swap), k[0]);
                            let m6 = vandq_u64(pair!(p + 1, $swap), k[1]);
                            let m5 = vandq_u64(pair!(p + 2, $swap), k[2]);
                            let m4 = vandq_u64(pair!(p + 3, $swap), k[3]);
                            let m3 = vandq_u64(pair!(p + 4, $swap), k[4]);
                            let m2 = vandq_u64(pair!(p + 5, $swap), k[5]);
                            let m1 = vandq_u64(pair!(p + 6, $swap), k[6]);
                            let m0 = vandq_u64(pair!(p + 7, $swap), k[7]);
                            // An OR tree, not a chain.
                            let x = vorrq_u64(vorrq_u64(vorrq_u64(m7, m6), vorrq_u64(m5, m4)), vorrq_u64(vorrq_u64(m3, m2), vorrq_u64(m1, m0)));
                            acc = vorrq_u64(vshlq_n_u64(acc, 8), x);
                            p += 8;
                        }
                        vst1q_u64(row.as_mut_ptr().add(2 * g), acc);
                    }
                } else {
                    for p in 0..ns / 2 {
                        pair!(p, $swap);
                    }
                }
            };
        }
        if col & 1 == 1 {
            body!(true);
        } else {
            body!(false);
        }
    }
}

/// WASM SIMD128 kernel (`h ≥ 7`), the NEON kernel's pairing and back-pointer
/// layout (`bp_bit`): states in pairs, the partner of pair `p` is pair
/// `p ^ col>>1`, lanes swapped when `col` is odd; the comparison masks are
/// shifted into a two-lane accumulator (even states, odd states) stored as
/// the group's two words. Bit-identical to [`step_scalar`]:
///
/// - the new cost is `v128_bitselect(b, a, b < a)`, i.e. `b < a ? b : a`
///   lane-wise, and the back pointer is the same mask, so ties keep the `c0`
///   branch;
/// - one of `c0`, `c1` is zero at every position (`costs_at`), and adding
///   ±0.0 is the identity on every state cost (a state cost is +0.0, a sum of
///   non-negative costs or +∞, never −0.0), so that add is skipped.
///
/// About 3× the scalar kernel under V8 (Node 25, Apple M-series).
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#[inline]
unsafe fn step_simd128<const ROWS: bool>(prev: &[f64], cur: &mut [f64], row: &mut [u64], col: usize, c0: f64, c1: f64) {
    // SAFETY: see `step`.
    unsafe {
        use core::arch::wasm32::*;
        let ns = prev.len();
        let half = col >> 1;
        let pp = prev.as_ptr();
        let cp = cur.as_mut_ptr();
        let (vc0, vc1) = (f64x2_splat(c0), f64x2_splat(c1));
        // One pair: its new state costs stored, its comparison mask returned.
        macro_rules! pair {
            ($p:expr, $swap:expr, $add0:expr, $add1:expr) => {{
                let p = $p;
                let q = p ^ half;
                let mut gq = v128_load(pp.add(2 * q) as *const v128);
                if $swap {
                    gq = i64x2_shuffle::<1, 0>(gq, gq);
                }
                let mut a = v128_load(pp.add(2 * p) as *const v128);
                if $add0 {
                    a = f64x2_add(a, vc0);
                }
                let b = if $add1 { f64x2_add(gq, vc1) } else { gq };
                let mk = f64x2_lt(b, a);
                v128_store(cp.add(2 * p) as *mut v128, v128_bitselect(b, a, mk));
                mk
            }};
        }
        macro_rules! body {
            ($swap:expr, $add0:expr, $add1:expr) => {
                if ROWS {
                    // As the NEON kernel: eight pairs' mask bits at once
                    // (bits 7..0, first pair highest), then shifted into the
                    // accumulator, whose dependency chain is one shift and
                    // one OR per eight pairs. Per pair the chain bounded the
                    // kernel: 31 % slower at h = 9 (Node 25, M-series).
                    let k: [v128; 8] = core::array::from_fn(|i| u64x2_splat(128 >> i));
                    for g in 0..ns / 128 {
                        let mut acc = u64x2_splat(0);
                        let mut p = 64 * g;
                        while p < 64 * g + 64 {
                            let m7 = v128_and(pair!(p, $swap, $add0, $add1), k[0]);
                            let m6 = v128_and(pair!(p + 1, $swap, $add0, $add1), k[1]);
                            let m5 = v128_and(pair!(p + 2, $swap, $add0, $add1), k[2]);
                            let m4 = v128_and(pair!(p + 3, $swap, $add0, $add1), k[3]);
                            let m3 = v128_and(pair!(p + 4, $swap, $add0, $add1), k[4]);
                            let m2 = v128_and(pair!(p + 5, $swap, $add0, $add1), k[5]);
                            let m1 = v128_and(pair!(p + 6, $swap, $add0, $add1), k[6]);
                            let m0 = v128_and(pair!(p + 7, $swap, $add0, $add1), k[7]);
                            let x = v128_or(v128_or(v128_or(m7, m6), v128_or(m5, m4)), v128_or(v128_or(m3, m2), v128_or(m1, m0)));
                            acc = v128_or(i64x2_shl(acc, 8), x);
                            p += 8;
                        }
                        v128_store(row.as_mut_ptr().add(2 * g) as *mut v128, acc);
                    }
                } else {
                    for p in 0..ns / 2 {
                        pair!(p, $swap, $add0, $add1);
                    }
                }
            };
        }
        match (col & 1 == 1, c0 == 0.0, c1 == 0.0) {
            (true, true, _) => body!(true, false, true),
            (false, true, _) => body!(false, false, true),
            (true, false, true) => body!(true, true, false),
            (false, false, true) => body!(false, true, false),
            (true, false, false) => body!(true, true, true),
            (false, false, false) => body!(false, true, true),
        }
    }
}

/// Pass A of the checkpointed schedule, for a host that recomputes the
/// segments elsewhere ([`segment_rows`], [`segment_trace`]; the website's
/// Web Workers, `ghost::session`): state costs at the start of every `k`-th
/// message block, and the terminal state.
#[derive(Debug, Clone)]
pub struct PassA {
    /// `2^h` state costs at the start of blocks `0, k, 2k, …` (segment order).
    pub checkpoints: Vec<f64>,
    /// Lowest-index terminal state of minimum cost.
    pub best: usize,
    /// Its cost (the trellis optimum).
    pub cost: f64,
}

/// The checkpointed schedule split up: the forward pass over all `m`
/// blocks of `code` keeping the state costs at every `k`-th block boundary.
/// `Ok(None)` if infeasible or the cover is too short. With [`segment_rows`]
/// and [`segment_trace`] run over the segments `[s·k, min((s+1)·k, m))` from
/// the last to the first, the stego bits are bit for bit [`embed_code`]'s.
pub fn pass_a(
    cover: &[u8],
    costs: &[f32],
    message: &[u8],
    code: &Code,
    h: usize,
    k: usize,
    tick: &(dyn Fn(usize) + Sync),
) -> Result<Option<PassA>, StegoError> {
    let m = message.len();
    let k = k.max(1);
    if h == 0 || h > MAX_H || m == 0 || m != code.m() {
        return Ok(None);
    }
    let n = code.n();
    if n > cover.len() || n > costs.len() {
        return Ok(None);
    }
    let sets = code.masked_sets(h);
    let t = Trellis { cover: &cover[..n], costs: &costs[..n], message, code, sets: &sets, b_off: 0, p_off: 0, h };
    let progress = Progress::new(n, n, tick);
    let mut ticker = progress.ticker();
    let (mut prev, mut cur) = t.initial();
    let ns = 1usize << h;
    let mut checkpoints = Vec::with_capacity(m.div_ceil(k) * ns);
    t.forward(
        &mut prev,
        &mut cur,
        0,
        m,
        None,
        &mut |b, p| {
            if b % k == 0 {
                checkpoints.extend_from_slice(p);
            }
        },
        &mut ticker,
    )?;
    ticker.flush()?;
    let (best, cost) = best_state(&prev);
    if cost == f64::INFINITY {
        return Ok(None);
    }
    Ok(Some(PassA { checkpoints, best, cost }))
}

/// Back pointers of one segment — blocks `[b0, b0 + message.len())` of
/// `code`, `cover` / `costs` their positions `[code.start(b0), …)` — from
/// its checkpoint (the segment's `2^h` state costs from [`pass_a`]):
/// `words_per_step(h)` words per position.
pub fn segment_rows(
    cover: &[u8],
    costs: &[f32],
    message: &[u8],
    code: &Code,
    b0: usize,
    h: usize,
    checkpoint: &[f64],
) -> Result<Vec<u64>, StegoError> {
    let nb = message.len();
    let ns = 1usize << h.min(MAX_H);
    if h == 0 || h > MAX_H || b0 + nb > code.m() || checkpoint.len() != ns {
        return Err(StegoError::MessageTooLarge);
    }
    let p_off = code.start(b0);
    let n = code.start(b0 + nb) - p_off;
    if cover.len() < n || costs.len() < n {
        return Err(StegoError::MessageTooLarge);
    }
    let sets = code.masked_sets(h);
    let t = Trellis { cover: &cover[..n], costs: &costs[..n], message, code, sets: &sets, b_off: b0, p_off, h };
    let progress = Progress::new(n, n, &|_| {});
    let mut ticker = progress.ticker();
    let mut prev = checkpoint.to_vec();
    let mut cur = vec![0.0; ns];
    let mut rows = vec![0u64; n * words_per_step(h)];
    t.forward(&mut prev, &mut cur, 0, nb, Some(&mut rows), &mut |_, _| {}, &mut ticker)?;
    Ok(rows)
}

/// Trace one segment (blocks `[b0, b0 + message.len())` of `code`) back
/// from `state` (the state after its last block's message shift:
/// [`PassA::best`] for the last segment, else the previous call's return
/// value) through its [`segment_rows`], writing its stego bits into `out`
/// (the segment's positions). Returns the state at the segment's start.
pub fn segment_trace(rows: &[u64], message: &[u8], code: &Code, b0: usize, h: usize, state: usize, out: &mut [u8]) -> usize {
    let sets = code.masked_sets(h);
    let p_off = code.start(b0);
    let n = code.start(b0 + message.len()) - p_off;
    let t = Trellis { cover: &[], costs: &[], message, code, sets: &sets, b_off: b0, p_off, h };
    t.traceback(rows, 0, message.len(), state, &mut out[..n])
}

/// The [`Plan`] of finished stego bits (changes counted against `cover`).
pub fn plan_of(cover: &[u8], stego_bits: Vec<u8>, total_cost: f64) -> Plan {
    let changes = stego_bits.iter().zip(cover).filter(|(s, c)| (**s ^ **c) & 1 != 0).count();
    Plan { stego_bits, total_cost, changes }
}

fn masked_cols(cols: &[u32], h: usize) -> Vec<usize> {
    cols.iter().map(|&c| (c as usize) & ((1 << h) - 1)).collect()
}

/// Syndrome of `code`'s first `k` blocks over the stego bits `bit(j)`
/// (code positions), e.g. a decoder reading a frame header.
pub fn syndrome_prefix(code: &Code, k: usize, bit: impl Fn(usize) -> bool) -> Vec<u8> {
    let k = k.min(code.m());
    let mut out = Vec::with_capacity(k);
    let mut state = 0u32;
    for b in 0..k {
        let s0 = code.start(b);
        for (c, &col) in code.cols(b).iter().enumerate() {
            if bit(s0 + c) {
                state ^= col;
            }
        }
        out.push((state & 1) as u8);
        state >>= 1;
    }
    out
}

/// Cross-platform pin of the trellis kernel: SHA-256 over the plans
/// (stego bits, total cost bits, change count) of deterministic random
/// cases at h = 1, 6, 7, 9 and 10, both schedules, wet and zero costs
/// included. The WASM bundle exposes it (`__test_trellis_hash`) so Node can
/// check the SIMD128 kernel against the native (NEON / scalar) value.
#[doc(hidden)]
pub fn trellis_test_hash_hex() -> String {
    use crate::stego::rng_compat::Rand08;
    use rand::SeedableRng;
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    for (seed, h, m, w, wet) in [
        (1u64, 9usize, 400usize, 7usize, 0.05f64),
        (2, 9, 300, 1, 0.0),
        (3, 9, 120, 31, 0.3),
        (4, 7, 500, 4, 0.1),
        (5, 10, 200, 6, 0.02),
        (6, 1, 200, 3, 0.0),
        (7, 6, 200, 3, 0.0),
    ] {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        let n = m * w;
        let cover: Vec<u8> = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
        let costs: Vec<f32> = (0..n)
            .map(|_| {
                let r = rng.gen08::<f64>();
                if r < wet {
                    f32::INFINITY
                } else if r < wet + 0.02 {
                    0.0
                } else {
                    rng.gen08::<f32>() * 50.0
                }
            })
            .collect();
        let msg: Vec<u8> = (0..m).map(|_| rng.gen08::<u8>() & 1).collect();
        let hh = crate::stego::stc::hhat::generate_hhat_end_set(h, w, &[seed as u8; 32]);
        let mut cols: Vec<u32> = (0..w).map(|c| crate::stego::stc::hhat::column_packed(&hh, c)).collect();
        if seed == 4 {
            // Even columns too (the end-set H-hat always has bit 0 set).
            for c in cols.iter_mut().step_by(2) {
                *c &= !1;
            }
        }
        for limit in [usize::MAX, 0] {
            match embed(&cover, &costs, &msg, &cols, h, limit, 1, &|_| {}).expect("not cancelled") {
                Some(p) => {
                    hash.update(&p.stego_bits);
                    hash.update(p.total_cost.to_bits().to_le_bytes());
                    hash.update((p.changes as u64).to_le_bytes());
                }
                None => hash.update(b"infeasible"),
            }
        }
    }
    // Mixed-width codes (Ghost v3.1): a head of fixed width, then blocks of
    // w and w + 1.
    for (seed, h, m, n, head, hw, wet) in [
        (8u64, 9usize, 300usize, 300 * 5 + 137usize, 48usize, 4usize, 0.05f64),
        (9, 9, 50, 50 * 31 + 49, 0, 0, 0.1),
        (10, 9, 400, 400 + 399, 48, 1, 0.0),
    ] {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        let cover: Vec<u8> = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
        let costs: Vec<f32> = (0..n)
            .map(|_| if rng.gen08::<f64>() < wet { f32::INFINITY } else { rng.gen08::<f32>() * 50.0 })
            .collect();
        let msg: Vec<u8> = (0..m).map(|_| rng.gen08::<u8>() & 1).collect();
        let code = test_mixed_code(h, m, n, head, hw, seed as u8);
        for limit in [usize::MAX, 0] {
            match embed_code(&cover, &costs, &msg, &code, h, limit, 1, &|_| {}).expect("not cancelled") {
                Some(p) => {
                    hash.update(&p.stego_bits);
                    hash.update(p.total_cost.to_bits().to_le_bytes());
                    hash.update((p.changes as u64).to_le_bytes());
                }
                None => hash.update(b"infeasible"),
            }
        }
    }
    hash.finalize().iter().map(|b| format!("{b:02x}")).collect()
}

/// A mixed-width code of `m` blocks over `n` positions with `head` blocks
/// of width `hw` (end-set H-hats seeded by `seed` and the width).
#[doc(hidden)]
pub fn test_mixed_code(h: usize, m: usize, n: usize, head: usize, hw: usize, seed: u8) -> Code {
    let cols = |w: usize| -> Vec<u32> {
        if w == 0 {
            return Vec::new();
        }
        let hh = crate::stego::stc::hhat::generate_hhat_end_set(h, w, &[seed ^ (w as u8); 32]);
        (0..w).map(|c| crate::stego::stc::hhat::column_packed(&hh, c)).collect()
    };
    let rest_n = n - head * hw;
    let rm = m - head;
    let s = if rm > 0 { rest_n / rm } else { 0 };
    let long = if rm > 0 && rest_n % rm != 0 { cols(s + 1) } else { Vec::new() };
    Code::new(m, head, cols(hw), rest_n, cols(s), long).expect("consistent test code")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::stc::{embed as stc, extract, hhat};
    use crate::stego::rng_compat::Rand08;
    use rand::SeedableRng;

    fn case(seed: u64, n_blocks: usize, w: usize, wet_frac: f64) -> (Vec<u8>, Vec<f32>, Vec<u8>) {
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(seed);
        let n = n_blocks * w;
        let cover = (0..n).map(|_| rng.gen08::<u8>() & 1).collect();
        let costs = (0..n)
            .map(|_| if rng.gen08::<f64>() < wet_frac { f32::INFINITY } else { (rng.gen08::<f32>() * 6.0).exp() })
            .collect();
        let msg = (0..n_blocks).map(|_| rng.gen08::<u8>() & 1).collect();
        (cover, costs, msg)
    }

    fn packed(hh: &[Vec<u32>], w: usize) -> Vec<u32> {
        (0..w).map(|c| hhat::column_packed(hh, c)).collect()
    }

    /// Pinned (aarch64 NEON); the WASM bundle's `__test_trellis_hash` must
    /// return the same value.
    #[test]
    fn trellis_hash_pinned() {
        assert_eq!(trellis_test_hash_hex(), "2390518fe2b3f2c1748932a72423fcd1e25d6e0817ed5964521e940fafdf25e8");
    }

    /// The split checkpointed schedule (pass A, segment rows, traces from
    /// the end) plans exactly what `embed` plans, for any segment length.
    #[test]
    fn split_schedule_equals_embed() {
        for (seed, h, m, w, wet) in [(21u64, 9usize, 300usize, 7usize, 0.05f64), (22, 9, 64, 40, 0.0), (23, 7, 500, 3, 0.2), (24, 9, 5, 1, 0.0)] {
            let (cover, costs, msg) = case(seed, m, w, wet);
            let hh = hhat::generate_hhat_end_set(h, w, &[seed as u8; 32]);
            let cols = packed(&hh, w);
            split_equals_embed(&cover, &costs, &msg, &Code::uniform(&cols, m), h, seed);
        }
        // Mixed widths: head, short and long blocks.
        for (seed, h, m, n, head, hw, wet) in [
            (31u64, 9usize, 300usize, 300 * 5 + 137usize, 48usize, 4usize, 0.05f64),
            (32, 9, 50, 50 * 31 + 49, 0, 0, 0.1),
            (33, 9, 400, 799, 48, 1, 0.0),
            (34, 7, 120, 120 * 9 + 1, 48, 8, 0.2),
        ] {
            let (cover, costs, _) = case(seed, n, 1, wet);
            let msg: Vec<u8> = case(seed + 100, m, 1, 0.0).2;
            let code = test_mixed_code(h, m, n, head, hw, seed as u8);
            split_equals_embed(&cover, &costs, &msg, &code, h, seed);
        }
    }

    fn split_equals_embed(cover: &[u8], costs: &[f32], msg: &[u8], code: &Code, h: usize, seed: u64) {
        let m = msg.len();
        {
            let want = embed_code(cover, costs, msg, code, h, usize::MAX, 1, &|_| {}).unwrap();
            for k in [1usize, 2, 7, 64, m, m + 3] {
                let got = pass_a(cover, costs, msg, code, h, k, &|_| {}).unwrap().map(|pa| {
                    let ns = 1usize << h;
                    let mut bits = vec![0u8; code.n()];
                    let mut state = pa.best;
                    for seg in (0..m.div_ceil(k)).rev() {
                        let (b0, b1) = (seg * k, ((seg + 1) * k).min(m));
                        let (p0, p1) = (code.start(b0), code.start(b1));
                        let rows = segment_rows(&cover[p0..p1], &costs[p0..p1], &msg[b0..b1], code, b0, h, &pa.checkpoints[seg * ns..(seg + 1) * ns]).unwrap();
                        state = segment_trace(&rows, &msg[b0..b1], code, b0, h, state, &mut bits[p0..p1]);
                    }
                    assert_eq!(state, 0);
                    plan_of(cover, bits, pa.cost)
                });
                match (&want, got) {
                    (None, None) => {}
                    (Some(a), Some(b)) => {
                        assert_eq!(a.stego_bits, b.stego_bits, "seed {seed} k {k}");
                        assert_eq!(a.total_cost.to_bits(), b.total_cost.to_bits());
                        assert_eq!(a.changes, b.changes);
                    }
                    _ => panic!("feasibility differs, seed {seed} k {k}"),
                }
            }
        }
    }

    #[test]
    fn matches_stc_embed_at_h7() {
        for (seed, m, w, wet) in [(1u64, 300usize, 5usize, 0.0), (2, 500, 1, 0.0), (3, 200, 12, 0.3), (4, 1000, 3, 0.1)] {
            let (cover, costs, msg) = case(seed, m, w, wet);
            for end_set in [false, true] {
                let hh = if end_set {
                    hhat::generate_hhat_end_set(7, w, &[seed as u8; 32])
                } else {
                    hhat::generate_hhat(7, w, &[seed as u8; 32])
                };
                let a = stc::stc_embed(&cover, &costs, &msg, &hh, 7, w);
                let b = embed(&cover, &costs, &msg, &packed(&hh, w), 7, usize::MAX, 1, &|_| {}).unwrap();
                let c = embed(&cover, &costs, &msg, &packed(&hh, w), 7, 0, 1, &|_| {}).unwrap();
                match (a, b, c) {
                    (None, None, None) => {}
                    (Some(a), Some(b), Some(c)) => {
                        assert_eq!(a.stego_bits, b.stego_bits, "seed {seed} inline");
                        assert_eq!(a.stego_bits, c.stego_bits, "seed {seed} segmented");
                        assert_eq!(a.total_cost.to_bits(), b.total_cost.to_bits());
                        assert_eq!(a.num_modifications, b.changes);
                    }
                    _ => panic!("feasibility differs, seed {seed}"),
                }
            }
        }
    }

    #[test]
    fn heights_8_to_10_roundtrip_and_segmented_equals_inline() {
        for h in [8usize, 9, 10] {
            let (cover, costs, msg) = case(10 + h as u64, 700, 6, 0.05);
            let hh = hhat::generate_hhat_end_set(h, 6, &[h as u8; 32]);
            let cols = packed(&hh, 6);
            let a = embed(&cover, &costs, &msg, &cols, h, usize::MAX, 1, &|_| {}).unwrap().unwrap();
            let b = embed(&cover, &costs, &msg, &cols, h, 0, 1, &|_| {}).unwrap().unwrap();
            assert_eq!(a.stego_bits, b.stego_bits, "h={h}");
            assert_eq!(extract::stc_extract_prefix(&a.stego_bits, &hh, 6, msg.len()), msg, "h={h}");
        }
    }

    #[test]
    fn higher_h_is_not_worse() {
        let (cover, costs, msg) = case(77, 2000, 8, 0.0);
        let mut last = f64::INFINITY;
        for h in [6usize, 7, 8, 9] {
            let hh = hhat::generate_hhat_end_set(h, 8, &[5u8; 32]);
            let p = embed(&cover, &costs, &msg, &packed(&hh, 8), h, usize::MAX, 1, &|_| {}).unwrap().unwrap();
            // Not a theorem for a fixed random matrix, but holds by a wide
            // margin at this size; guards against a broken kernel.
            assert!(p.total_cost < last * 1.02, "h={h}: {} vs {last}", p.total_cost);
            last = p.total_cost;
        }
    }

    #[test]
    fn ticks_sum_to_positions() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let (cover, costs, msg) = case(5, 40_000, 3, 0.0);
        let hh = hhat::generate_hhat_end_set(7, 3, &[1u8; 32]);
        for (limit, threads) in [(usize::MAX, 1), (0, 1), (0, 3), (inline_bytes(70_000, 7), 1)] {
            let sum = AtomicUsize::new(0);
            embed(&cover, &costs, &msg, &packed(&hh, 3), 7, limit, threads, &|k| {
                sum.fetch_add(k, Ordering::Relaxed);
            })
            .unwrap()
            .unwrap();
            assert_eq!(sum.load(Ordering::Relaxed), 120_000, "limit {limit} threads {threads}");
        }
    }

    /// Every schedule gives the inline plan bit for bit: checkpointed, with a
    /// tail of any length (the forward pass keeps the final blocks), and with
    /// waves of recomputed segments (threads > 1), at h = 9.
    #[test]
    fn every_schedule_equals_inline() {
        for (seed, m, w, wet) in [(21u64, 900usize, 7usize, 0.0), (22, 150, 40, 0.2), (23, 3000, 2, 0.05), (24, 7, 300, 0.0)] {
            let (cover, costs, msg) = case(seed, m, w, wet);
            let h = 9;
            let cols = packed(&hhat::generate_hhat_end_set(h, w, &[seed as u8; 32]), w);
            let base = embed(&cover, &costs, &msg, &cols, h, usize::MAX, 1, &|_| {}).unwrap().unwrap();
            let full = inline_bytes(m * w, h);
            let block = inline_bytes(w, h);
            let mut limits = vec![0, block, 3 * block, full / 4, full / 2, full - block, full - 1];
            limits.retain(|&l| l < full);
            for limit in limits {
                for threads in [1usize, 2, 3, 5] {
                    let p = embed(&cover, &costs, &msg, &cols, h, limit, threads, &|_| {}).unwrap().unwrap();
                    assert_eq!(p.stego_bits, base.stego_bits, "seed {seed} limit {limit} threads {threads}");
                    assert_eq!(p.total_cost.to_bits(), base.total_cost.to_bits());
                    assert_eq!(p.changes, base.changes);
                }
            }
        }
    }

    /// Mixed-width codes: every position is used (`n` exactly), widths are
    /// `w` / `w + 1` after the head, every schedule gives the inline plan, and
    /// the syndrome of the plan is the message.
    #[test]
    fn mixed_codes_schedules_and_syndrome() {
        for (seed, h, m, n, head, hw, wet) in [
            (41u64, 9usize, 700usize, 700 * 6 + 333usize, 48usize, 5usize, 0.05f64),
            (42, 9, 90, 90 * 40 + 89, 48, 33, 0.0),
            (43, 9, 3000, 3000 * 2 - 1, 48, 1, 0.02),
            (44, 9, 7, 7 * 300 + 5, 0, 0, 0.0),
        ] {
            let code = test_mixed_code(h, m, n, head, hw, seed as u8);
            assert_eq!(code.n(), n);
            let rm = m - head;
            for b in 0..m {
                let wb = code.width(b);
                if b < head {
                    assert_eq!(wb, hw);
                } else {
                    assert!(wb == (n - head * hw) / rm || wb == (n - head * hw) / rm + 1, "b {b} width {wb}");
                }
            }
            let (cover, costs, _) = case(seed, n, 1, wet);
            let msg: Vec<u8> = case(seed + 7, m, 1, 0.0).2;
            let base = embed_code(&cover, &costs, &msg, &code, h, usize::MAX, 1, &|_| {}).unwrap().unwrap();
            let syn = syndrome_prefix(&code, m, |j| base.stego_bits[j] & 1 == 1);
            assert_eq!(syn, msg, "seed {seed}");
            let full = inline_bytes(n, h);
            for limit in [0, full / 3, full - 1] {
                for threads in [1usize, 3] {
                    let p = embed_code(&cover, &costs, &msg, &code, h, limit, threads, &|_| {}).unwrap().unwrap();
                    assert_eq!(p.stego_bits, base.stego_bits, "seed {seed} limit {limit} threads {threads}");
                    assert_eq!(p.total_cost.to_bits(), base.total_cost.to_bits());
                }
            }
        }
    }

    /// The tail cuts the recomputation: a limit of half the table walks
    /// ~1.5× the positions, nothing kept walks 2×, waves keep it at 2×
    /// (spread over the threads); the working set never exceeds the limit
    /// by more than the checkpoints.
    #[test]
    fn schedule_work_and_memory() {
        let (m, w, h) = (2000usize, 900usize, 9usize);
        let n = m * w;
        let full = inline_bytes(n, h);
        assert_eq!(walked_positions(n, w, h, full, 1), n);
        assert_eq!(walked_positions(n, w, h, 0, 1), 2 * n);
        let half = walked_positions(n, w, h, full / 2, 1);
        assert!((half as f64 / n as f64 - 1.5).abs() < 0.01, "{half}");
        let wb = working_bytes(n, w, h, full / 2, 1);
        assert!(wb <= full / 2 + 4 * (1 << h) * 8, "{wb}");
        if cfg!(feature = "parallel") {
            assert_eq!(walked_positions(n, w, h, 0, 4), 2 * n);
            assert!(working_bytes(n, w, h, 0, 4) < full / 16);
        }
    }

    /// The scalar kernel (WASM, x86) and the platform kernel give the same
    /// state costs and the same back-pointer words at every height.
    #[test]
    fn kernels_agree() {
        use crate::stego::rng_compat::Rand08;
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(99);
        for h in 1..=MAX_H {
            let ns = 1usize << h;
            for _ in 0..20 {
                let prev: Vec<f64> =
                    (0..ns).map(|_| if rng.gen_bool08(0.1) { f64::INFINITY } else { rng.gen08::<f64>() * 100.0 }).collect();
                let col = (rng.gen08::<usize>() & (ns - 1)) | 1 | (ns >> 1);
                let (c0, c1) = if rng.gen_bool08(0.5) { (0.0, rng.gen08::<f64>() * 9.0) } else { (rng.gen08::<f64>() * 9.0, 0.0) };
                let wps = words_per_step(h);
                let (mut ca, mut ra) = (vec![0.0; ns], vec![0u64; wps]);
                let (mut cb, mut rb) = (vec![0.0; ns], vec![0u64; wps]);
                let mut cc = vec![0.0; ns];
                step_scalar::<true>(&prev, &mut ca, &mut ra, col, c0, c1);
                step::<true>(&prev, &mut cb, &mut rb, col, c0, c1);
                step::<false>(&prev, &mut cc, &mut [], col, c0, c1);
                assert!(ca.iter().zip(&cb).all(|(a, b)| a.to_bits() == b.to_bits()), "h={h}");
                assert!(ca.iter().zip(&cc).all(|(a, b)| a.to_bits() == b.to_bits()), "h={h} no rows");
                assert_eq!(ra, rb, "h={h}");
                for s in 0..ns {
                    let take = prev[s ^ col] + c1 < prev[s] + c0;
                    let (wi, bi) = bp_bit(s, ns);
                    assert_eq!((ra[wi] >> bi) & 1 == 1, take, "h={h} s={s}");
                }
            }
        }
    }

    #[test]
    fn infeasible_and_short_cover() {
        let hh = hhat::generate_hhat_end_set(7, 2, &[1u8; 32]);
        let cols = packed(&hh, 2);
        // all wet, message disagrees with the cover's syndrome
        let cover = vec![0u8; 20];
        let costs = vec![f32::INFINITY; 20];
        let msg = vec![1u8; 10];
        assert!(embed(&cover, &costs, &msg, &cols, 7, usize::MAX, 1, &|_| {}).unwrap().is_none());
        assert!(embed(&cover[..19], &costs, &msg, &cols, 7, usize::MAX, 1, &|_| {}).unwrap().is_none());
    }
}
