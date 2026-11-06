// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shadow messages: Y-channel keyed-region STC + RS framing (headerless).
//!
//! Shadow messages provide plausible deniability for Ghost mode steganography.
//! Multiple messages can be hidden in a single image, each with a different
//! passphrase. The system auto-sorts messages by size: the largest becomes
//! the primary, smaller messages become shadow layers.
//!
//! ## Layout (Ghost v3 and v3.1)
//!
//! v3.1 (fix list R3.16-18) changed only the primary; its shadows use the
//! layout v3 (core 0.5 / apps 1.9) introduced, so one shadow decoder reads
//! both generations. The costs of the shadow trellises are v3.1's (J-UNIWARD
//! on the unrounded decompressed image).
//!
//! Every shadow reads along a **keyed hash order** of the Y AC coefficients
//! (zeros included): position `fi` has priority `ChaCha20(seed)` word pair
//! `2·fi` and the order sorts by priority. The set of AC positions never
//! changes, so the decoder rebuilds the order exactly — no costs on the
//! decoder side.
//!
//! A shadow of `n` RS-coded bits owns the *region* = the first `n·w`
//! positions of its hash order and embeds with a syndrome-trellis code of
//! width `w` into the coefficient parity (end-set H-hat, the v3 height). The
//! encoder's Viterbi minimises the J-UNIWARD (or SI-UNIWARD) cost of the
//! flips; each flip is a ±1 change in the v3 direction (toward the precover
//! with side information, else a keyed coin). Layers never touch each
//! other: a later shadow treats every earlier shadow's region as wet, and
//! the primary treats all regions as wet, so the bits arrive error-free; RS
//! parity 4 is kept only as an integrity gate (the decoder requires zero
//! corrections). Widths come from the v3 ladder.
//!
//! ## Decode-only: shadows of core ≤ 0.4.x / apps ≤ 1.8 (layout v1)
//!
//! - **Legacy tier**: direct LSB on the first `n` positions of a keyed hash
//!   order of the Y *non-zero* AC set, RS parity 4…128, brute-forced (the
//!   "100 % pool" where every v1 shadow ended up in practice);
//! - **cost pools**: the cheapest 5–50 % of the nzAC set under the v1
//!   J-UNIWARD costs, read only after everything else failed
//!   ([`shadow_extract_cost_pools`]).
//!
//! ## Frame format (inside RS-encoded data, no header)
//!
//! ```text
//!     [plaintext_len: 2B] [salt: 16B] [nonce: 12B] [ciphertext: N+16B]
//! ```

use crate::codec::jpeg::JpegImage;
use crate::codec::jpeg::dct::DctGrid;
use crate::stego::armor::ecc;
use crate::stego::crypto;
#[cfg(test)]
use crate::stego::crypto::{NONCE_LEN, SALT_LEN};
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, FileEntry, PayloadData};
use crate::stego::permute::CoeffPos;
use crate::stego::progress;
use crate::stego::stc::{extract as stc_extract, hhat, trellis};
use super::side_info::SideInfo;
use super::v31;
use crate::stego::shadow_layer::{
    build_shadow_frame, parse_shadow_frame, peek_shadow_fdl, SHADOW_FRAME_OVERHEAD_V1,
};

/// Frame overhead lower bound (= v1 overhead, 46 bytes). The image encoder
/// picks the v1 frame for plaintexts ≤ u16::MAX.
const SHADOW_FRAME_OVERHEAD: usize = SHADOW_FRAME_OVERHEAD_V1;

/// RS parity of a shadow layer. Its bits arrive error-free (no other layer
/// writes into its region), so the parity is an integrity gate: the decoder
/// requires zero corrections.
pub const SHADOW_STC_PARITY: usize = 4;

/// Smallest shadow region payload the decoder accepts (bits).
pub const SHADOW_MIN_STC_BITS: usize = (SHADOW_FRAME_OVERHEAD + SHADOW_STC_PARITY) * 8;

/// RS parity tiers of the layout-v1 legacy tier (decode-only). Brute-forced.
pub const SHADOW_PARITY_TIERS: [usize; 6] = [4, 8, 16, 32, 64, 128];

/// Shadow layout v1's cost-pool fractions (denominators): 20 = cheapest 5 %,
/// 10 = 10 %, 5 = 20 %, 2 = 50 %. Decode-only; v1's 100 % pool is the
/// legacy tier.
pub const COST_DEPENDENT_FRACTIONS: [usize; 4] = [20, 10, 5, 2];

/// Largest shadow frame the layout-v1 brute force considers (decoder sanity
/// bound, not wire). The released v1 encoder never capped shadow frames at
/// the 256 KB its decoder used, so the bound is the shadow format's own.
const MAX_SHADOW_FRAME_BYTES: usize = crate::stego::shadow_layer::MAX_SHADOW_FRAME_BYTES;


// Thread-local scratch buffer for the brute-force loop's
// bits_to_bytes step. The no-match path can call try_single_fdl
// ~6000× per smart_decode; reusing a per-thread Vec eliminates
// the per-iteration allocation churn. Each rayon worker gets its
// own scratch via thread_local.
thread_local! {
    static RS_BYTES_SCRATCH: std::cell::RefCell<Vec<u8>> =
        std::cell::RefCell::new(Vec::with_capacity(256));
}

/// Try a single (lsbs, fdl, parity, passphrase) combination.
/// Returns `Some(Ok(payload))` on success, `None` on failure.
fn try_single_fdl(
    lsbs: &[u8],
    fdl: usize,
    parity_len: usize,
    passphrase: &str,
) -> Option<Result<PayloadData, StegoError>> {
    let rs_encoded_len = ecc::rs_encoded_len_with_parity(fdl, parity_len);
    let rs_bits_needed = rs_encoded_len * 8;
    if rs_bits_needed > lsbs.len() {
        return None;
    }

    RS_BYTES_SCRATCH.with(|scratch| {
        let mut rs_bytes = scratch.borrow_mut();
        frame::bits_to_bytes_into(&lsbs[..rs_bits_needed], &mut rs_bytes);
        try_single_fdl_with_rs_bytes(&rs_bytes, fdl, parity_len, passphrase)
    })
}

/// Inner body of [`try_single_fdl`] given pre-packed RS bytes.
fn try_single_fdl_with_rs_bytes(
    rs_bytes: &[u8],
    fdl: usize,
    parity_len: usize,
    passphrase: &str,
) -> Option<Result<PayloadData, StegoError>> {
    let decoded = match ecc::rs_decode_blocks_with_parity(rs_bytes, fdl, parity_len) {
        Ok((data, _stats)) => data,
        Err(_) => return None,
    };

    // O(1) format-aware consistency gate: the frame's own length field must
    // agree with the brute-force candidate, or skip the Argon2 + AES work.
    let expected_total = peek_shadow_fdl(&decoded)?;
    if expected_total != fdl {
        return None;
    }
    decrypt_frame(&decoded, passphrase)
}

/// Parse a decoded shadow frame and decrypt it.
fn decrypt_frame(decoded: &[u8], passphrase: &str) -> Option<Result<PayloadData, StegoError>> {
    let fr = match parse_shadow_frame(decoded) {
        Ok(f) => f,
        Err(_) => return None,
    };

    match crypto::decrypt(&fr.ciphertext, passphrase, &fr.salt, &fr.nonce) {
        Ok(plaintext) => {
            let len = fr.plaintext_len as usize;
            if len > plaintext.len() {
                return None;
            }
            Some(payload::decode_payload(&plaintext[..len]))
        }
        Err(_) => None,
    }
}

/// Peek at the first RS block to read `plaintext_len` and derive the exact fdl.
/// Returns the candidate fdl if the first block decodes and the resulting fdl
/// is plausible (>= k, within pool capacity).
fn peek_fdl_from_first_block(
    lsbs: &[u8],
    parity_len: usize,
    max_fdl: usize,
) -> Option<usize> {
    let k = 255usize.saturating_sub(parity_len);
    if k < 2 || lsbs.len() < 255 * 8 {
        return None;
    }

    // Decode the first 255 RS bytes as one full block (k data bytes).
    let first_block_bytes = frame::bits_to_bytes(&lsbs[..255 * 8]);
    let (data, _) = ecc::rs_decode_blocks_with_parity(&first_block_bytes, k, parity_len).ok()?;

    let fdl = peek_shadow_fdl(&data)?;

    // Only valid if fdl >= k (multi-block, so first block IS a full 255-byte block)
    // and within pool capacity.
    if fdl >= k && fdl <= max_fdl {
        Some(fdl)
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Selection: keyed hash orders
// ---------------------------------------------------------------------------

/// The first `k` Y AC flat indices (`block · 64 + 1..63`) of `grid` in the
/// keyed hash order of `seed`: sorted by the priority `ChaCha20(seed)` word
/// pair `2·fi` (the 64-bit word [`select_shadow_positions`] reads for
/// position `fi`). Equal priorities keep raster order. Memory: the result
/// plus a bounded working set (see [`ac_hash_order_visit`]).
pub fn ac_hash_order_prefix(grid: &DctGrid, seed: &[u8; 32], k: usize) -> Vec<u32> {
    let mut out = Vec::with_capacity(k.min(v31::ac_count(grid)));
    ac_hash_order_visit(grid, seed, k, ORDER_GROUP_CANDIDATES, |fi| out.push(fi));
    out
}

/// Coefficient parities along the first `k` positions of the hash order
/// (`parity_along(ac_hash_order_prefix(..))` without holding the order).
fn ac_hash_order_parities(grid: &DctGrid, seed: &[u8; 32], k: usize) -> Vec<u8> {
    let c = grid.coeffs();
    let mut out = Vec::with_capacity(k.min(v31::ac_count(grid)));
    ac_hash_order_visit(grid, seed, k, ORDER_GROUP_CANDIDATES, |fi| out.push((c[fi as usize] & 1) as u8));
    out
}

/// Candidates (priority, position) one pass of [`ac_hash_order_visit`]
/// collects and sorts: 16 bytes each (128 MB transient at most). A full order
/// of a 100 MP cover takes ~14 keystream passes.
const ORDER_GROUP_CANDIDATES: usize = 8 << 20;

/// Top priority bits the histogram of [`ac_hash_order_visit`] buckets by.
const ORDER_BUCKET_BITS: u32 = 16;

/// Blocks per keystream chunk (the unit of parallelism).
const ORDER_CHUNK_BLOCKS: usize = 4096;

/// Run `step(acc, priority, fi)` over every AC position, chunk by chunk of
/// blocks (in parallel with `parallel`); the chunk accumulators come back in
/// block order. Block `b`'s 64 priorities are keystream bytes
/// `512 b .. 512 b + 512` (words `128 b ..`), so a chunk seeks to its first
/// block and reads on, exactly the stream a sequential pass reads.
fn scan_ac_priorities<T: Send>(
    seed: &[u8; 32],
    blocks: usize,
    chunks: std::ops::Range<usize>,
    init: impl Fn() -> T + Sync,
    step: impl Fn(&mut T, u64, u32) + Sync,
) -> Vec<T> {
    use rand::{Rng, SeedableRng};
    let run = |c: usize| -> T {
        let mut acc = init();
        let b0 = c * ORDER_CHUNK_BLOCKS;
        let b1 = (b0 + ORDER_CHUNK_BLOCKS).min(blocks);
        let mut rng = rand_chacha::ChaCha20Rng::from_seed(*seed);
        rng.set_word_pos(b0 as u128 * 128);
        let mut buf = [0u8; 512];
        for b in b0 as u32..b1 as u32 {
            rng.fill_bytes(&mut buf);
            // DC (k = 0) is not an AC position; its word pair is skipped.
            for kk in 1..64u32 {
                let o = kk as usize * 8;
                step(&mut acc, u64::from_le_bytes(buf[o..o + 8].try_into().unwrap()), b * 64 + kk);
            }
        }
        acc
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        chunks.into_par_iter().map(run).collect()
    }
    #[cfg(not(feature = "parallel"))]
    chunks.map(run).collect()
}

/// Visit the first `k` AC positions of the hash order, in order.
///
/// A histogram pass over the top `ORDER_BUCKET_BITS` priority bits finds the
/// buckets that hold the `k` smallest priorities; then each group of buckets
/// holding at most `group_cap` positions (or one bucket) takes one more pass
/// that collects exactly its positions, sorts them by priority (stable, so
/// equal priorities stay in raster order) and visits them. That is the
/// order of a full sort by (priority, position), so every prefix is exact,
/// while the working set is the histogram plus one group — independent of
/// `k` and of the cover (the cost: one keystream pass per group).
fn ac_hash_order_visit(grid: &DctGrid, seed: &[u8; 32], k: usize, group_cap: usize, mut visit: impl FnMut(u32)) {
    let blocks = grid.blocks_wide() * grid.blocks_tall();
    let k = k.min(blocks * 63);
    if k == 0 {
        return;
    }
    let shift = 64 - ORDER_BUCKET_BITS;
    let n_buckets = 1usize << ORDER_BUCKET_BITS;
    let chunks = blocks.div_ceil(ORDER_CHUNK_BLOCKS);
    let mut hist = vec![0u64; n_buckets];
    // Per-chunk histograms (256 KB each) are summed a round at a time, never
    // all held at once.
    const HIST_ROUND_CHUNKS: usize = 32;
    for c0 in (0..chunks).step_by(HIST_ROUND_CHUNKS) {
        let parts = scan_ac_priorities(
            seed,
            blocks,
            c0..(c0 + HIST_ROUND_CHUNKS).min(chunks),
            || vec![0u32; n_buckets],
            |h: &mut Vec<u32>, prio, _| h[(prio >> shift) as usize] += 1,
        );
        for part in &parts {
            for (t, &c) in hist.iter_mut().zip(part) {
                *t += c as u64;
            }
        }
    }

    let mut done = 0usize;
    let mut g0 = 0usize;
    while done < k && g0 < n_buckets {
        // Grow the group while it stays within the cap (at least one bucket).
        let mut g1 = g0;
        let mut count = 0u64;
        while g1 < n_buckets && (g1 == g0 || count + hist[g1] <= group_cap as u64) && done as u64 + count < k as u64 {
            count += hist[g1];
            g1 += 1;
        }
        if count > 0 {
            let (lo, hi) = (g0 as u64, g1 as u64);
            let mut parts = scan_ac_priorities(
                seed,
                blocks,
                0..chunks,
                Vec::new,
                |v: &mut Vec<(u64, u32)>, prio, fi| {
                    let b = prio >> shift;
                    if b >= lo && b < hi {
                        v.push((prio, fi));
                    }
                },
            );
            // Sort each chunk's part (stable: raster order on ties), then
            // merge the parts in place of one concatenated sort: a tie between
            // parts goes to the earlier part, i.e. the earlier position, so
            // the result is the stable sort of the concatenation without its
            // copy or the sort's scratch.
            let sort = |p: &mut Vec<(u64, u32)>| radsort::sort_by_key(p, |(prio, _)| *prio);
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                parts.par_iter_mut().for_each(sort);
            }
            #[cfg(not(feature = "parallel"))]
            parts.iter_mut().for_each(sort);
            let mut heads = vec![0usize; parts.len()];
            let mut heap: std::collections::BinaryHeap<std::cmp::Reverse<(u64, usize)>> = parts
                .iter()
                .enumerate()
                .filter(|(_, p)| !p.is_empty())
                .map(|(i, p)| std::cmp::Reverse((p[0].0, i)))
                .collect();
            let take = (k - done).min(count as usize);
            for _ in 0..take {
                let std::cmp::Reverse((_, i)) = heap.pop().expect("histogram counted these");
                visit(parts[i][heads[i]].1);
                heads[i] += 1;
                if let Some(&(prio, _)) = parts[i].get(heads[i]) {
                    heap.push(std::cmp::Reverse((prio, i)));
                }
            }
            done += take;
        }
        g0 = g1;
    }
}

/// Every position of `positions` (any order) sorted by its keyed priority
/// `ChaCha20(seed, flat_idx)` — the layout-v1 order of the nzAC set.
pub fn hash_order(positions: &[CoeffPos], seed: &[u8; 32]) -> Vec<CoeffPos> {
    select_shadow_positions(positions, 1, positions.len(), seed)
}

/// The H-hat seed of a shadow's STC: 32 bytes of ChaCha20 stream 1 under the
/// shadow's structural key (the hash priorities use stream 0).
fn stc_hhat_seed(perm_seed: &[u8; 32]) -> [u8; 32] {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_chacha::ChaCha20Rng::from_seed(*perm_seed);
    rng.set_stream(1);
    let mut out = [0u8; 32];
    rng.fill_bytes(&mut out);
    out
}

/// A shadow region's H-hat (end-set construction, the v3 height).
fn shadow_hhat(w: usize, perm_seed: &[u8; 32], h: usize) -> Vec<Vec<u32>> {
    hhat::generate_hhat_end_set(h, w, &stc_hhat_seed(perm_seed))
}

/// Coefficient parities along `order` (flat indices).
fn parity_along(grid: &DctGrid, order: &[u32]) -> Vec<u8> {
    let c = grid.coeffs();
    order.iter().map(|&fi| (c[fi as usize] & 1) as u8).collect()
}


// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// A shadow's encrypted frame, built once per encode (one salt + nonce).
#[derive(Clone)]
pub struct ShadowFrame {
    /// Raw frame bytes (before RS).
    pub frame_bytes: Vec<u8>,
    /// Shadow structural key (seed of the hash order and the H-hat).
    pub perm_seed: [u8; 32],
    /// Seed of the direction coin of this layer's changes.
    pub dir_seed: u64,
}

impl ShadowFrame {
    /// Build the payload, encrypt it and frame it.
    pub fn new(shadow_pass: &str, message: &str, files: &[FileEntry]) -> Result<Self, StegoError> {
        let payload_bytes = payload::encode_payload(message, files)?;
        let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, shadow_pass)?;
        let frame_bytes = build_shadow_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
        let perm_seed = crypto::derive_shadow_structural_key(shadow_pass)?;
        Ok(Self { frame_bytes, perm_seed: *perm_seed, dir_seed: v31::dir_seed_from_salt(&salt) })
    }

    /// RS-coded bit count (parity [`SHADOW_STC_PARITY`]).
    pub fn rs_bits(&self) -> usize {
        ecc::rs_encoded_len_with_parity(self.frame_bytes.len(), SHADOW_STC_PARITY) * 8
    }
}

/// One shadow layer planned for embedding.
#[derive(Clone)]
pub struct ShadowState {
    /// The region this layer owns (flat indices, hash order): `n_total · w`
    /// positions. The primary and every later shadow treat them as wet.
    pub positions: Vec<u32>,
    /// The parity each of `positions` holds once embedded.
    pub embed_bits: Vec<u8>,
    /// RS-encoded frame bits, as the decoder extracts them.
    pub n_total: usize,
    /// Shadow structural key.
    pub perm_seed: [u8; 32],
    /// STC width.
    pub stc_w: usize,
    /// STC height.
    pub h: usize,
    /// Direction coin seed.
    pub dir_seed: u64,
}

impl ShadowState {
    /// RS-coded bits of this layer and the region length at width `w`.
    pub fn region_len(frame: &ShadowFrame, w: usize) -> usize {
        frame.rs_bits() * w
    }

    /// Plan a region of width `w` along `order` (the shadow's hash-order
    /// prefix, at least `n_total · w` long) against the cover `img` plus the
    /// values of earlier layers (`overlay`). `cost(i)` is the flip cost of
    /// `order[i]` (∞ for earlier shadows' regions). `inline_limit`: the
    /// trellis back-pointer table kept in memory (above: checkpointed).
    ///
    /// `Ok(None)` when the region does not fit or the trellis is infeasible.
    #[allow(clippy::too_many_arguments)]
    pub fn plan(
        frame: &ShadowFrame,
        order: &[u32],
        w: usize,
        img: &JpegImage,
        overlay: &v31::Overlay,
        cost: impl Fn(usize) -> f32,
        h: usize,
        inline_limit: usize,
        tick: &(dyn Fn(usize) + Sync),
    ) -> Result<Option<Self>, StegoError> {
        let rs_bytes = ecc::rs_encode_blocks_with_parity(&frame.frame_bytes, SHADOW_STC_PARITY);
        let bits = frame::bytes_to_bits(&rs_bytes);
        let n_total = bits.len();
        let region_len = n_total * w;
        if order.len() < region_len {
            return Ok(None);
        }
        let region = &order[..region_len];
        let grid = img.dct_grid(0);
        let cover_bits: Vec<u8> = region.iter().map(|&fi| (v31::value(grid, Some(overlay), fi) & 1) as u8).collect();
        let costs: Vec<f32> = (0..region_len).map(&cost).collect();
        let cols = v31::packed_cols(&shadow_hhat(w, &frame.perm_seed, h), w);
        let threads = crate::stego::memory::encode_workers();
        let Some(planned) = trellis::embed(&cover_bits, &costs, &bits, &cols, h, inline_limit, threads, tick)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            positions: region.to_vec(),
            embed_bits: planned.stego_bits,
            n_total,
            perm_seed: frame.perm_seed,
            stc_w: w,
            h,
            dir_seed: frame.dir_seed,
        }))
    }
}

/// Embed a shadow layer into the overlay: give every position its planned
/// parity by a ±1 change in the v3 direction (toward the precover with side
/// information, else a keyed coin). Returns the number of coefficients
/// changed.
pub fn embed_shadow(img: &JpegImage, overlay: &mut v31::Overlay, state: &ShadowState, si: Option<&SideInfo>) -> usize {
    let mut changed = 0;
    let grid = img.dct_grid(0);
    for (&fi, &bit) in state.positions.iter().zip(&state.embed_bits) {
        let c = v31::value(grid, Some(overlay), fi);
        if (c & 1) as u8 != bit {
            overlay.set(fi, v31::modify_pm1(c, si.map(|s| s.error_at(fi as usize)), state.dir_seed, fi as u64));
            changed += 1;
        }
    }
    changed
}

/// Verify a shadow layer from the current image exactly as the decoder
/// reads it: the STC syndrome along the region, RS-decode with zero
/// corrections, decrypt.
pub fn verify_shadow(img: &JpegImage, state: &ShadowState, passphrase: &str) -> Result<(), StegoError> {
    let lsbs = parity_along(img.dct_grid(0), &state.positions);
    let hh = shadow_hhat(state.stc_w, &state.perm_seed, state.h);
    match try_stc_tier(&lsbs, &hh, state.stc_w, passphrase, lsbs.len() / state.stc_w) {
        Some(Ok(_)) => Ok(()),
        Some(Err(e)) => Err(e),
        None => Err(StegoError::FrameCorrupted),
    }
}

// ---------------------------------------------------------------------------
// Decoder (v3)
// ---------------------------------------------------------------------------

/// A tier's frame data length from its 48-bit syndrome header (v1 header =
/// 2 bytes, v2 = 6 bytes, both at the start of the first RS block), if the
/// RS-coded frame fits `max_bits`.
fn tier_fdl(header_bits: &[u8], max_bits: usize) -> Option<(usize, usize)> {
    if max_bits < SHADOW_MIN_STC_BITS || header_bits.len() < 48 {
        return None;
    }
    let fdl = peek_shadow_fdl(&frame::bits_to_bytes(header_bits))?;
    let rs_len = ecc::rs_encoded_len_with_parity(fdl, SHADOW_STC_PARITY);
    if rs_len.checked_mul(8)? > max_bits {
        return None;
    }
    Some((fdl, rs_len))
}

/// Decode one STC tier from parities along the hash order. Exact: reads the
/// frame length from the first syndrome bits, extracts exactly the frame's
/// bits, and requires the RS decode to correct nothing (a random word passes
/// that with probability 2⁻³² per block), so a wrong passphrase or tier
/// almost never reaches the Argon2 + AES step. `max_bits`: the most bits the
/// region may hold at this width (`n / w`).
fn try_stc_tier(
    lsbs: &[u8],
    hh: &[Vec<u32>],
    w: usize,
    passphrase: &str,
    max_bits: usize,
) -> Option<Result<PayloadData, StegoError>> {
    let head = stc_extract::stc_extract_prefix(lsbs, hh, w, 48);
    let (fdl, rs_len) = tier_fdl(&head, max_bits)?;
    if lsbs.len() < rs_len * 8 * w {
        return None;
    }
    let bits = stc_extract::stc_extract_prefix(lsbs, hh, w, rs_len * 8);
    let rs_bytes = frame::bits_to_bytes(&bits);
    let (decoded, stats) = ecc::rs_decode_blocks_with_parity(&rs_bytes, fdl, SHADOW_STC_PARITY).ok()?;
    if stats.total_errors != 0 {
        return None;
    }
    decrypt_frame(&decoded, passphrase)
}

/// Whether the first RS block of a tier's frame decodes with zero
/// corrections (a random word passes with probability 2⁻³²).
fn first_block_clean(lsbs: &[u8], hh: &[Vec<u32>], w: usize, fdl: usize, rs_len: usize) -> bool {
    let k = 255 - SHADOW_STC_PARITY;
    let (blk_bytes, blk_data) = if fdl <= k { (rs_len, fdl) } else { (255, k) };
    if lsbs.len() < blk_bytes * 8 * w {
        return false;
    }
    let bits = stc_extract::stc_extract_prefix(lsbs, hh, w, blk_bytes * 8);
    matches!(
        ecc::rs_decode_blocks_with_parity(&frame::bits_to_bytes(&bits), blk_data, SHADOW_STC_PARITY),
        Ok((_, s)) if s.total_errors == 0
    )
}

/// Ghost v3 shadow decode: STC regions along the passphrase's hash order of
/// the Y AC set, every width of the v3 ladder.
///
/// Memory is `O(region)`, not `O(cover)`: the hash order is built only as
/// far as the reads need — every width's 48-bit header first, then the first
/// RS block of the widths whose header is plausible, then the full region
/// of the widths whose first block decodes cleanly.
pub fn shadow_extract_v3(img: &JpegImage, passphrase: &str) -> Result<PayloadData, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let grid = img.dct_grid(0);
    let n = v31::ac_count(grid);
    let widths = v31::shadow_widths(n, SHADOW_MIN_STC_BITS);
    if widths.is_empty() {
        return Err(StegoError::FrameCorrupted);
    }
    let h = v31::V31_STC_H;
    let t = v31::tstart();
    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    v31::tlog("shadow key", t);
    progress::check_cancelled()?;

    // Stage 1: every width's header.
    let mut lsbs = ac_hash_order_parities(grid, &perm_seed, 48 * widths[widths.len() - 1]);
    v31::tlog("shadow order", t);
    let mut cand: Vec<(usize, Vec<Vec<u32>>, usize, usize)> = Vec::new(); // (w, hh, fdl, rs_len)
    for &w in &widths {
        let hh = shadow_hhat(w, &perm_seed, h);
        let head = stc_extract::stc_extract_prefix(&lsbs[..48 * w], &hh, w, 48);
        if let Some((fdl, rs_len)) = tier_fdl(&head, n / w) {
            cand.push((w, hh, fdl, rs_len));
        }
    }
    v31::tlog("shadow headers", t);
    progress::check_cancelled()?;

    // Stage 2: first RS block of the plausible widths.
    let blk = |&(w, _, fdl, rs_len): &(usize, Vec<Vec<u32>>, usize, usize)| {
        let k = 255 - SHADOW_STC_PARITY;
        (if fdl <= k { rs_len } else { 255 }) * 8 * w
    };
    let need2 = cand.iter().map(blk).max().unwrap_or(0);
    if need2 > lsbs.len() {
        lsbs = ac_hash_order_parities(grid, &perm_seed, need2);
    }
    cand.retain(|(w, hh, fdl, rs_len)| first_block_clean(&lsbs, hh, *w, *fdl, *rs_len));
    progress::check_cancelled()?;

    // Stage 3: full regions of the surviving widths.
    let need3 = cand.iter().map(|(w, _, _, rs_len)| rs_len * 8 * w).max().unwrap_or(0);
    if need3 > lsbs.len() {
        lsbs = ac_hash_order_parities(grid, &perm_seed, need3);
    }
    for (w, hh, _, _) in &cand {
        if let Some(r) = try_stc_tier(&lsbs, hh, *w, passphrase, n / w) {
            return r;
        }
    }
    Err(StegoError::FrameCorrupted)
}

// ---------------------------------------------------------------------------
// Decoder (layout v1, core ≤ 0.4.x)
// ---------------------------------------------------------------------------

/// The first `k` Y non-zero AC positions (flat indices) in layout v1's
/// keyed hash order (the order `hash_order` gives the nzAC set): one
/// sequential keystream pass, `O(k)` memory.
pub fn nzac_hash_order_prefix(grid: &DctGrid, seed: &[u8; 32], k: usize) -> Vec<u32> {
    use rand::{Rng, SeedableRng};
    let coeffs = grid.coeffs();
    let nb = grid.blocks_wide() * grid.blocks_tall();
    let n = coeffs.chunks_exact(64).map(|b| b[1..].iter().filter(|&&c| c != 0).count()).sum::<usize>();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let mut want = (k as u128) * 5 / 4 + 64;
    loop {
        let thr = if want >= n as u128 { u64::MAX } else { ((want << 64) / n as u128) as u64 };
        let mut cand: Vec<(u64, u32)> = Vec::with_capacity((want as usize).min(n) + (want as usize).min(n) / 8);
        let mut rng = rand_chacha::ChaCha20Rng::from_seed(*seed);
        let mut buf = [0u8; 512];
        for b in 0..nb {
            rng.fill_bytes(&mut buf);
            let blk = &coeffs[b * 64..b * 64 + 64];
            for kk in 1..64usize {
                if blk[kk] == 0 {
                    continue;
                }
                let prio = u64::from_le_bytes(buf[kk * 8..kk * 8 + 8].try_into().unwrap());
                if prio <= thr {
                    cand.push((prio, (b * 64 + kk) as u32));
                }
            }
        }
        if cand.len() >= k || thr == u64::MAX {
            radsort::sort_by_key(&mut cand, |(prio, _)| *prio);
            return cand.into_iter().take(k).map(|(_, fi)| fi).collect();
        }
        want *= 2;
    }
}

/// The first `k` positions of `pool` (any order) in the keyed hash order:
/// threshold selection, `O(k)` memory (as `select_shadow_positions`, which
/// sorts the whole pool).
fn pool_hash_order_prefix(pool: &[CoeffPos], seed: &[u8; 32], k: usize) -> Vec<u32> {
    use rand::{Rng, SeedableRng};
    let n = pool.len();
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let mut rng = rand_chacha::ChaCha20Rng::from_seed(*seed);
    let mut want = (k as u128) * 5 / 4 + 64;
    loop {
        let thr = if want >= n as u128 { u64::MAX } else { ((want << 64) / n as u128) as u64 };
        let mut cand: Vec<(u64, u32)> = Vec::with_capacity((want as usize).min(n) + (want as usize).min(n) / 8);
        for p in pool {
            rng.set_word_pos(p.flat_idx as u128 * 2);
            let prio = rng.next_u64();
            if prio <= thr {
                cand.push((prio, p.flat_idx));
            }
        }
        if cand.len() >= k || thr == u64::MAX {
            radsort::sort_by_key(&mut cand, |(prio, _)| *prio);
            return cand.into_iter().take(k).map(|(_, fi)| fi).collect();
        }
        want *= 2;
    }
}

/// The keyed priorities of the cheapest `len` entries of `cost_sorted`,
/// indexed by cost rank: `prio[r]` = the word pair `2·flat_idx` of
/// `ChaCha20(seed)` for `cost_sorted[r]` — what [`pool_hash_order_prefix`]
/// reads with one `set_word_pos` per position (each refilling a 256-byte
/// block buffer), here from ONE sequential keystream pass over the blocks
/// that hold a pool member (as [`nzac_hash_order_prefix`] reads it).
///
/// The cost pools are nested (each is a prefix of `cost_sorted`), so the
/// priorities of the largest pool serve every smaller one
/// ([`ranked_hash_order_prefix`]).
///
/// Working memory, on top of the result (8 B per member): a membership
/// bitmap (1 bit per Y coefficient), per-block member counts (4 B per
/// block) and the flat-order → rank map (4 B per member), freed on return.
/// `None` if a flat index repeats or lies outside the grid (no real cost
/// list does): the caller then reads per position, as before.
fn pool_priorities(grid: &DctGrid, cost_sorted: &[CoeffPos], len: usize, seed: &[u8; 32]) -> Option<Vec<u64>> {
    use rand::{Rng, SeedableRng};
    let len = len.min(cost_sorted.len());
    let nb = grid.coeffs().len() / 64;
    // One bitmap word per 8×8 block (bit k = coefficient k of the block).
    let mut members = vec![0u64; nb];
    for p in &cost_sorted[..len] {
        let fi = p.flat_idx as usize;
        let (b, bit) = (fi / 64, 1u64 << (fi % 64));
        if b >= nb || members[b] & bit != 0 {
            return None;
        }
        members[b] |= bit;
    }
    // Members in blocks before b; then each rank's slot in flat order.
    let mut before = vec![0u32; nb];
    let mut acc = 0u32;
    for (b, m) in members.iter().enumerate() {
        before[b] = acc;
        acc += m.count_ones();
    }
    let mut rank_at = vec![0u32; len];
    for (r, p) in cost_sorted[..len].iter().enumerate() {
        let fi = p.flat_idx as usize;
        let (b, k) = (fi / 64, fi % 64);
        let j = before[b] + (members[b] & ((1u64 << k) - 1)).count_ones();
        rank_at[j as usize] = r as u32;
    }
    let mut prio = vec![0u64; len];
    let mut rng = rand_chacha::ChaCha20Rng::from_seed(*seed);
    let mut buf = [0u8; 512];
    let mut next_block = 0usize;
    for (b, &m) in members.iter().enumerate() {
        if m == 0 {
            continue;
        }
        if b != next_block {
            // 128 keystream words per block (a word pair per coefficient).
            rng.set_word_pos(b as u128 * 128);
        }
        rng.fill_bytes(&mut buf);
        next_block = b + 1;
        let mut rest = m;
        let mut j = before[b] as usize;
        while rest != 0 {
            let k = rest.trailing_zeros() as usize;
            rest &= rest - 1;
            prio[rank_at[j] as usize] = u64::from_le_bytes(buf[k * 8..k * 8 + 8].try_into().unwrap());
            j += 1;
        }
    }
    Some(prio)
}

/// The first `k` positions of the pool `cost_sorted[..n]` in the keyed hash
/// order, from the rank-indexed priorities of a pool at least as large
/// ([`pool_priorities`]). The same threshold selection as
/// [`pool_hash_order_prefix`], over the same candidates in the same order
/// (pool order) with the same stable sort — so the same output, ties
/// included.
fn ranked_hash_order_prefix(prio: &[u64], cost_sorted: &[CoeffPos], n: usize, k: usize) -> Vec<u32> {
    let k = k.min(n);
    if k == 0 {
        return Vec::new();
    }
    let mut want = (k as u128) * 5 / 4 + 64;
    loop {
        let thr = if want >= n as u128 { u64::MAX } else { ((want << 64) / n as u128) as u64 };
        let mut cand: Vec<(u64, u32)> = Vec::with_capacity((want as usize).min(n) + (want as usize).min(n) / 8);
        for (p, pos) in prio[..n].iter().zip(&cost_sorted[..n]) {
            if *p <= thr {
                cand.push((*p, pos.flat_idx));
            }
        }
        if cand.len() >= k || thr == u64::MAX {
            radsort::sort_by_key(&mut cand, |(prio, _)| *prio);
            return cand.into_iter().take(k).map(|(_, fi)| fi).collect();
        }
        want *= 2;
    }
}

/// Coefficient |LSBs| along flat indices.
fn abs_lsbs(grid: &DctGrid, order: &[u32]) -> Vec<u8> {
    let c = grid.coeffs();
    order.iter().map(|&fi| (c[fi as usize].unsigned_abs() & 1) as u8).collect()
}

/// The layout-v1 brute force (parity × fdl) over an order of `total`
/// positions whose LSB prefix of any length `prefix(k)` returns. Reads only
/// what it needs: the first RS block (255 × 8 positions) for the peek and the
/// small-frame scan, then the whole frame of the (at most six) peeked
/// candidates — the same candidates, bounds and checks as a brute force over
/// all `total` LSBs.
fn legacy_brute_force(total: usize, prefix: impl Fn(usize) -> Vec<u8>, passphrase: &str) -> Result<PayloadData, StegoError> {
    let base = prefix(total.min(255 * 8));
    progress::check_cancelled()?;
    // Peek: the first block announces the frame length (fdl >= k).
    let mut cands: Vec<(usize, usize)> = Vec::new();
    let mut need = 0;
    for &parity_len in &SHADOW_PARITY_TIERS {
        let max_fdl = compute_max_fdl(total / 8, parity_len).min(MAX_SHADOW_FRAME_BYTES);
        if SHADOW_FRAME_OVERHEAD > max_fdl {
            continue;
        }
        if let Some(fdl) = peek_fdl_from_first_block(&base, parity_len, max_fdl) {
            need = need.max(ecc::rs_encoded_len_with_parity(fdl, parity_len) * 8);
            cands.push((parity_len, fdl));
        }
    }
    if !cands.is_empty() {
        let long = if need > base.len() { prefix(need.min(total)) } else { base.clone() };
        for &(parity_len, fdl) in &cands {
            if let Some(r) = try_single_fdl(&long, fdl, parity_len, passphrase) {
                return r;
            }
        }
    }
    // Small frames (fdl < k): all within the first block.
    brute_force_lsb_pools(std::slice::from_ref(&base), passphrase)
}

/// Layout v1's legacy tier: direct LSBs along the passphrase's hash order of
/// the Y nzAC set, brute-force (parity, fdl). AES-256-GCM-SIV validates.
/// Memory is bounded by the frame read, not the cover.
pub fn shadow_extract_legacy(img: &JpegImage, passphrase: &str) -> Result<PayloadData, StegoError> {
    if img.num_components() == 0 {
        return Err(StegoError::NoLuminanceChannel);
    }
    let grid = img.dct_grid(0);
    let total = super::capacity::count_nonzero_ac(grid);
    if total == 0 {
        return Err(StegoError::FrameCorrupted);
    }
    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    legacy_brute_force(total, |k| abs_lsbs(grid, &nzac_hash_order_prefix(grid, &perm_seed, k)), passphrase)
}

/// Shadow layout v1's cost pools (decode-only): the cheapest `1/fraction` of
/// `cost_sorted` (sorted by cost, cheapest first), hash-ordered, then the
/// legacy brute force. For shadows of core ≤ 0.4.x that landed in a tight
/// pool, read under the costs that encoder used.
pub fn shadow_extract_cost_pools(
    img: &JpegImage,
    cost_sorted: &[CoeffPos],
    passphrase: &str,
    fractions: &[usize],
) -> Result<PayloadData, StegoError> {
    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    if cost_sorted.is_empty() {
        return Err(StegoError::FrameCorrupted);
    }
    let grid = img.dct_grid(0);
    // The pools are nested prefixes of `cost_sorted`: one keystream pass
    // gives the priorities of the largest, and every pool reads its
    // prefix of them.
    let largest = fractions.iter().map(|&f| cost_sorted.len() / f.max(1)).max().unwrap_or(0);
    progress::check_cancelled()?;
    let prio = pool_priorities(grid, cost_sorted, largest, &perm_seed);
    let mut last = StegoError::FrameCorrupted;
    for &fraction in fractions {
        let pool = &cost_sorted[..cost_sorted.len() / fraction.max(1)];
        if pool.is_empty() {
            continue;
        }
        let prefix = |k: usize| match &prio {
            Some(prio) => ranked_hash_order_prefix(prio, cost_sorted, pool.len(), k),
            None => pool_hash_order_prefix(pool, &perm_seed, k),
        };
        match legacy_brute_force(pool.len(), |k| abs_lsbs(grid, &prefix(k)), passphrase) {
            Ok(p) => return Ok(p),
            Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
            Err(e) => last = e,
        }
    }
    Err(last)
}

/// Brute-force (pool, parity, fdl) over LSB streams: first-block peek
/// (fdl ≥ k), then the small-fdl scan (fdl < k). AES-256-GCM-SIV validates.
fn brute_force_lsb_pools(
    fraction_lsbs: &[Vec<u8>],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;

        // Phase 2a: First-block peek — decode first RS block to read plaintext_len
        // and derive the exact fdl. Handles messages where fdl >= k (most cases).
        let combos_2a: Vec<(usize, usize)> = (0..fraction_lsbs.len())
            .flat_map(|fi| SHADOW_PARITY_TIERS.iter().map(move |&p| (fi, p)))
            .collect();
        if let Some(result) = combos_2a.par_iter().find_map_first(|&(fi, parity_len)| {
            let lsbs = &fraction_lsbs[fi];
            let k = 255usize.saturating_sub(parity_len);
            if k == 0 { return None; }
            let max_rs_bytes = lsbs.len() / 8;
            let max_fdl = compute_max_fdl(max_rs_bytes, parity_len)
                .min(MAX_SHADOW_FRAME_BYTES);
            if SHADOW_FRAME_OVERHEAD > max_fdl { return None; }
            let fdl = peek_fdl_from_first_block(lsbs, parity_len, max_fdl)?;
            try_single_fdl(lsbs, fdl, parity_len, passphrase)
        }) {
            return result;
        }

        // Phase 2b: Small-fdl fallback — for tiny messages where fdl < k (single
        // partial RS block). Scan fdl from SHADOW_FRAME_OVERHEAD to min(k-1, max_fdl).
        let mut combos: Vec<(usize, usize, usize)> = Vec::new();
        for (fi, lsbs) in fraction_lsbs.iter().enumerate() {
            for &parity_len in &SHADOW_PARITY_TIERS {
                let k = 255usize.saturating_sub(parity_len);
                if k < 2 { continue; }
                let max_rs_bytes = lsbs.len() / 8;
                let max_fdl = compute_max_fdl(max_rs_bytes, parity_len)
                    .min(MAX_SHADOW_FRAME_BYTES);
                // Only scan fdl < k (partial block range); peek handled fdl >= k.
                let small_max = (k - 1).min(max_fdl);
                if SHADOW_FRAME_OVERHEAD > small_max { continue; }
                for fdl in SHADOW_FRAME_OVERHEAD..=small_max {
                    combos.push((fi, parity_len, fdl));
                }
            }
        }

        // Progress: one step per batch (cancellation checked between batches).
        const SHADOW_BRUTE_BATCHES: usize = 32;
        let total_combos = combos.len();
        let batch_size = total_combos.div_ceil(SHADOW_BRUTE_BATCHES).max(1);
        let mut emitted_steps: u32 = 0;
        let mut result: Option<Result<PayloadData, StegoError>> = None;
        for batch in combos.chunks(batch_size) {
            progress::check_cancelled()?;
            let hit = batch.par_iter().find_map_first(|&(fi, parity_len, fdl)| {
                let lsbs = &fraction_lsbs[fi];
                try_single_fdl(lsbs, fdl, parity_len, passphrase)
            });
            progress::advance();
            emitted_steps += 1;
            if hit.is_some() {
                result = hit;
                break;
            }
        }
        while emitted_steps < SHADOW_BRUTE_BATCHES as u32 {
            progress::advance();
            emitted_steps += 1;
        }

        match result {
            Some(ok_or_err) => ok_or_err,
            None => Err(StegoError::FrameCorrupted),
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for all_lsbs in fraction_lsbs {
            if let Some(result) = try_extract_with_lsbs(all_lsbs, passphrase) {
                return result;
            }
        }
        Err(StegoError::FrameCorrupted)
    }
}

/// Try all (parity, fdl) combinations on a given LSB vector using
/// first-block peek + small-fdl fallback.
#[cfg(not(feature = "parallel"))]
fn try_extract_with_lsbs(all_lsbs: &[u8], passphrase: &str) -> Option<Result<PayloadData, StegoError>> {
    for &parity_len in &SHADOW_PARITY_TIERS {
        let k = 255usize.saturating_sub(parity_len);
        if k < 2 { continue; }

        let max_rs_bytes = all_lsbs.len() / 8;
        let max_fdl = compute_max_fdl(max_rs_bytes, parity_len)
            .min(MAX_SHADOW_FRAME_BYTES);
        if SHADOW_FRAME_OVERHEAD > max_fdl { continue; }

        if let Some(fdl) = peek_fdl_from_first_block(all_lsbs, parity_len, max_fdl)
            && let Some(result) = try_single_fdl(all_lsbs, fdl, parity_len, passphrase) {
                return Some(result);
            }

        let small_max = (k - 1).min(max_fdl);
        if SHADOW_FRAME_OVERHEAD > small_max { continue; }
        for fdl in SHADOW_FRAME_OVERHEAD..=small_max {
            if let Some(result) = try_single_fdl(all_lsbs, fdl, parity_len, passphrase) {
                return Some(result);
            }
        }
    }
    None
}

/// Largest single shadow message (plaintext bytes) for a cover with `y_nzac`
/// non-zero Y AC coefficients, JPEG input: an RS-coded frame of half the
/// Ghost maximum (`capacity::band_frame_bits`), so the primary — the largest
/// message, by the auto-sort — still fits next to it. Technically a shadow
/// region can hold far more; this is the stealth policy. A Deep Cover cover
/// with exact side information: `capacity::estimate_shadow_capacity_si`.
pub fn shadow_capacity(y_nzac: usize) -> usize {
    super::capacity::shadow_capacity_for(y_nzac, super::capacity::MAX_RATE_DIV)
}

/// Largest shadow plaintext (bytes) whose RS-coded frame fits `bits`.
pub fn shadow_plaintext_for_bits(bits: usize) -> usize {
    let max_fdl = compute_max_fdl(bits / 8, SHADOW_STC_PARITY);
    max_fdl.saturating_sub(SHADOW_FRAME_OVERHEAD)
}

/// RS-coded bits of a shadow frame carrying `plaintext_bytes`.
pub fn shadow_bits_for_plaintext(plaintext_bytes: usize) -> usize {
    ecc::rs_encoded_len_with_parity(plaintext_bytes + SHADOW_FRAME_OVERHEAD, SHADOW_STC_PARITY) * 8
}

/// Hash-priority selection: the cheapest `1/fraction` of
/// `cost_sorted_positions` (sorted by cost, cheapest first — irrelevant for
/// `fraction = 1`), sorted by the keyed priority `ChaCha20(seed, flat_idx)`,
/// first `n_total` taken. Layout v1, decode-only.
pub(crate) fn select_shadow_positions(
    cost_sorted_positions: &[CoeffPos],
    fraction: usize,
    n_total: usize,
    seed: &[u8; 32],
) -> Vec<CoeffPos> {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha20Rng;

    // Tier 1: cost pool — cheapest 1/fraction of all positions.
    let pool_size = cost_sorted_positions.len() / fraction;
    if pool_size == 0 {
        return Vec::new();
    }
    let pool = &cost_sorted_positions[..pool_size];

    // Tier 2: hash permutation within the cost pool.
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let mut candidates: Vec<(u64, CoeffPos)> = pool.iter().map(|p| {
        rng.set_word_pos(p.flat_idx as u128 * 2);
        let priority = rng.next_u64();
        (priority, p.clone())
    }).collect();
    // LSD radix sort on the u64 priorities (unique in practice, so any sort
    // gives the same order).
    radsort::sort_by_key(&mut candidates, |(priority, _)| *priority);

    candidates.into_iter().map(|(_, p)| p).take(n_total).collect()
}

/// Compute the maximum frame_data_len that fits in `max_rs_bytes` with given parity.
fn compute_max_fdl(max_rs_bytes: usize, parity_len: usize) -> usize {
    let k = 255usize.saturating_sub(parity_len);
    if k == 0 || max_rs_bytes == 0 {
        return 0;
    }
    let full_blocks = max_rs_bytes / 255;
    let remainder = max_rs_bytes % 255;
    let mut max_data = full_blocks * k;
    if remainder > parity_len {
        max_data += remainder - parity_len;
    }
    max_data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_frame_roundtrip() {
        let salt = [1u8; SALT_LEN];
        let nonce = [2u8; NONCE_LEN];
        let ciphertext = vec![0xAA; 20]; // 4 bytes plaintext + 16 tag
        let fr = build_shadow_frame(4, &salt, &nonce, &ciphertext);
        let parsed = parse_shadow_frame(&fr).unwrap();
        assert_eq!(parsed.plaintext_len, 4);
        assert_eq!(parsed.salt, salt);
        assert_eq!(parsed.nonce, nonce);
        assert_eq!(parsed.ciphertext, ciphertext);
    }

    #[test]
    fn shadow_capacity_is_half_the_maximum() {
        assert_eq!(shadow_capacity(0), 0);
        assert_eq!(shadow_capacity(7), 0);
        let cap = shadow_capacity(3_833_089);
        assert!(shadow_bits_for_plaintext(cap) <= 3_833_089 / 5 / 2);
        assert!(shadow_bits_for_plaintext(cap + 1) > 3_833_089 / 5 / 2);
    }

    #[test]
    fn plaintext_bits_conversions_are_inverse() {
        for pt in [0usize, 1, 64, 204, 205, 251, 1000, 70_000] {
            let bits = shadow_bits_for_plaintext(pt);
            assert_eq!(shadow_plaintext_for_bits(bits), pt, "pt {pt}");
            assert!(shadow_plaintext_for_bits(bits - 8) < pt || pt == 0);
        }
        assert_eq!(shadow_plaintext_for_bits(0), 0);
    }

    /// The hash order depends on the set only — not on the input order or
    /// the costs — and is a permutation of it.
    #[test]
    fn hash_order_ignores_input_order_and_costs() {
        let seed = [7u8; 32];
        let a: Vec<CoeffPos> = (0..5000u32).map(|i| CoeffPos { flat_idx: i * 3 + 1, cost: i as f32 }).collect();
        let mut b = a.clone();
        b.reverse();
        for p in b.iter_mut() {
            p.cost = -(p.flat_idx as f32);
        }
        let oa: Vec<u32> = hash_order(&a, &seed).iter().map(|p| p.flat_idx).collect();
        let ob: Vec<u32> = hash_order(&b, &seed).iter().map(|p| p.flat_idx).collect();
        assert_eq!(oa, ob);
        let mut sorted = oa.clone();
        sorted.sort_unstable();
        let mut want: Vec<u32> = a.iter().map(|p| p.flat_idx).collect();
        want.sort_unstable();
        assert_eq!(sorted, want, "a permutation of the set");
        assert_ne!(oa[..100], want[..100], "and not the raster order");
    }

    /// The streamed all-AC prefix equals the per-position (`set_word_pos`)
    /// hash order of the AC set, at every prefix length.
    #[test]
    fn ac_hash_order_prefix_equals_the_hash_order() {
        let img = JpegImage::from_bytes(&std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg").unwrap()).unwrap();
        let grid = img.dct_grid(0);
        let seed = [9u8; 32];
        let all: Vec<CoeffPos> =
            v31::ac_flat_indices(grid).into_iter().map(|fi| CoeffPos { flat_idx: fi, cost: 0.0 }).collect();
        let full: Vec<u32> = hash_order(&all, &seed).iter().map(|p| p.flat_idx).collect();
        for k in [0usize, 1, 7, 100, 999, 5_000, 37_000, 75_599, 75_600, 80_000] {
            let pre = ac_hash_order_prefix(grid, &seed, k);
            assert_eq!(pre, full[..k.min(full.len())], "k = {k}");
            // Many small groups (one keystream pass each) give the same order.
            for cap in [300usize, 7777] {
                let mut small = Vec::new();
                ac_hash_order_visit(grid, &seed, k, cap, |fi| small.push(fi));
                assert_eq!(small, pre, "k = {k}, cap = {cap}");
            }
            let par = ac_hash_order_parities(grid, &seed, k);
            assert_eq!(par, parity_along(grid, &pre), "parities, k = {k}");
        }
    }

    /// The cost pools' one-pass priorities give exactly the per-position
    /// (`set_word_pos`) prefixes, for every nested pool and prefix length.
    /// (Ties between priorities cannot be forced through the keystream;
    /// their order is the same by construction: same candidates in the
    /// same order, same stable sort.)
    #[test]
    fn pool_priorities_match_the_per_position_prefix() {
        let img = JpegImage::from_bytes(&std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg").unwrap()).unwrap();
        let grid = img.dct_grid(0);
        let qt = img.quant_table(img.frame_info().components[0].quant_table_id as usize).unwrap();
        let mut sorted = crate::stego::cost::uniward::compute_positions_streaming_legacy_v1(grid, qt, None).unwrap();
        sorted.sort_by(|a, b| a.cost.total_cmp(&b.cost));
        assert!(sorted.len() > 10_000);
        for seed in [[3u8; 32], [0xa5; 32]] {
            let largest = sorted.len() / 2;
            let prio = pool_priorities(grid, &sorted, largest, &seed).unwrap();
            for frac in [20usize, 10, 5, 2] {
                let n = sorted.len() / frac;
                for k in [0usize, 1, 17, 255 * 8, n / 3, n - 1, n, n + 5] {
                    assert_eq!(
                        ranked_hash_order_prefix(&prio, &sorted, n, k),
                        pool_hash_order_prefix(&sorted[..n], &seed, k),
                        "frac {frac} k {k}"
                    );
                }
            }
        }
        // A repeated flat index falls back to the per-position read.
        let mut dup = sorted[..100].to_vec();
        dup[50] = dup[10].clone();
        assert!(pool_priorities(grid, &dup, 100, &[1u8; 32]).is_none());
    }

    /// Timing probe: keystream vs the prefix order at 12 MP.
    /// `cargo test --release --lib ac_order_timing -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn ac_order_timing() {
        use rand::{Rng, SeedableRng};
        let Ok(b) = std::fs::read("test-vectors/image/photo_original_3557.jpg") else { return };
        let img = JpegImage::from_bytes(&b).unwrap();
        let grid = img.dct_grid(0);
        let nb = grid.blocks_wide() * grid.blocks_tall();
        let t = std::time::Instant::now();
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([3u8; 32]);
        let mut buf = [0u8; 512];
        let mut x = 0u64;
        for _ in 0..nb {
            rng.fill_bytes(&mut buf);
            x ^= buf[7] as u64;
        }
        eprintln!("keystream {:?} ({x})", t.elapsed());
        for k in [1_000usize, 100_000, 1_450_000, 6_000_000] {
            let t = std::time::Instant::now();
            let o = ac_hash_order_prefix(grid, &[3u8; 32], k);
            eprintln!("prefix k={k}: {:?} ({})", t.elapsed(), o.len());
        }
    }

    /// The H-hat seed is a fixed function of the structural key (pinned:
    /// a change here would make every shadow unreadable).
    #[test]
    fn stc_hhat_seed_is_pinned() {
        let s = stc_hhat_seed(&[0u8; 32]);
        let hex: String = s.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, PINNED_HHAT_SEED_ZERO_KEY);
        assert_ne!(stc_hhat_seed(&[1u8; 32]), s);
    }

    /// `rand_chacha` ChaCha20, key = 0, stream 1, first 32 bytes.
    const PINNED_HHAT_SEED_ZERO_KEY: &str = "ef3fdfd6c61578fbf5cf35bd3dd33b8009631634d21e42ac33960bd138e50d32";
}
