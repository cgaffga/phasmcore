// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Fortress T: Fortress CC on a keyed, periodic (tiled) block layout with
//! pilots, so the decoder needs neither the image size nor the crop offset.
//!
//! # Why
//!
//! Fortress and Fortress CC key every block position through one permutation
//! of *all* blocks of the encoded image: a crop of any size (even a whole
//! number of blocks) changes the block count and the offset, and every
//! position is lost. A rotation or a rescale additionally moves the block
//! grid. Fortress T fixes the layout to the block's absolute position modulo
//! a tile, and adds known pilot blocks, the standard construction of
//! crop-robust watermarks (periodic tiling: Kalker, Linnartz, Depovere & Maes
//! 1999; position search by pilot correlation).
//!
//! # Layout
//!
//! - Block (bx, by) of the encoder's real luma grid belongs to tile cell
//!   (bx mod T, by mod T), T = 128 (1024 px).
//! - The header (tier index and repetition class, [`HDR_BITS`] bits, QIM step
//!   12) sits on [`HDR_BITS`] keyed cells of every [`HDR_PERIOD`]-block
//!   sub-tile, so every 256 x 256 px window holds each header bit once: small
//!   covers and heavy crops keep the whole header (randomly placed header
//!   cells left bits uncovered on a 640 x 480 cover, 2026-11-02).
//! - A keyed permutation of the remaining cells: the first [`N_PILOT`] are
//!   pilots (keyed bits, fixed QIM step 12), the rest payload slots: slot j
//!   carries coded bit j mod n of the Fortress CC codeword.
//! - Same BA-QIM, convolutional code, tiers and frame as Fortress CC; the
//!   repetition class (and so the QIM step) is written in the header because a
//!   decoder of a cropped image cannot count the encoder's blocks.
//! - Watson masking against a fixed reference energy ([`E_REF`]) instead of the
//!   image median, which a crop changes.
//! - A block checkerboard of [`SYNC_AMP`] gray levels (key-free sync pattern,
//!   see [`gridsync`](super::gridsync)): before the QIM in busy blocks (kept
//!   in expectation, invisible, no decoding cost), a smaller share after the
//!   QIM in flat blocks.
//!
//! # Decoding
//!
//! Block averages (from the DC coefficients, or sampled through a
//! [`gridsync`](super::gridsync) map) are folded into one tile; the pilot
//! correlation over all T^2 cyclic shifts finds the tile offset (a standard
//! score, threshold [`Z_MIN`]); then header, Viterbi, frame, decryption.
//! - [`decode_aligned`]: the received block grid (untouched, re-saved, or
//!   cropped by whole blocks); on a wrong passphrase only the fold and the
//!   correlation run.
//! - [`decode_geometric`]: rotation, rescale, crops at any offset. The
//!   checkerboard gives the affine map (key-free; an image without it costs
//!   one FFT); the pilots give the tile offset and polish the map.

use crate::stego::rng_compat::Rand08;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::codec::jpeg::JpegImage;
use crate::stego::armor::fortress::{
    adaptive_params, avg_to_dc, cc_code, cc_tiers, dc_to_avg, qim_embed_avg, qim_extract_soft, remap_watson,
    watson_base_factor, FortressEncodeResult, HEADER_QIM_STEP,
};
use crate::stego::armor::{convcode, ecc};
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::PayloadData;

use super::pipeline::DecodeQuality;

/// Tile side in blocks (power of two).
pub const T: usize = 128;
const TM: usize = T - 1;
/// Pilot cells per tile.
pub const N_PILOT: usize = 512;
/// Header bits: format (1: 0 = Fortress T, 1 = Fortress TL), tier index (6),
/// repetition / copy class (2).
pub const HDR_BITS: usize = 9;
/// Side (blocks) of the sub-tile that holds one copy of the header.
pub const HDR_PERIOD: usize = 32;
/// Header cells per tile (one header per sub-tile).
pub const N_HDR: usize = (T / HDR_PERIOD) * (T / HDR_PERIOD) * HDR_BITS;
/// Repetition classes: the encoder embeds with `adaptive_params` of its
/// class value (the step is flat up to r = 15, so most images are in class 0).
const R_CLASS: [usize; 4] = [15, 27, 40, 61];

fn r_class(r: usize) -> usize {
    R_CLASS.iter().rposition(|&c| c <= r).unwrap_or(0)
}
/// Payload slots per tile.
pub const N_SLOTS: usize = T * T - N_PILOT - N_HDR;
/// Pilot correlation threshold (standard score). The maximum of T^2 = 16384
/// null scores sits near 4.1; 6.0 keeps false alarms below ~1e-5 per field.
pub const Z_MIN: f64 = 6.0;
/// Minimum repetition of the coded stream (as Fortress CC).
pub const R_MIN: usize = 3;
/// Header bits a crop may miss entirely before [`decode_at`] gives up (each
/// one doubles the header values it tries; 6 = up to 192 Viterbi passes,
/// only once the pilots matched). A crop 24 blocks (192 px) wide misses 5.
const MAX_UNSEEN_HDR_BITS: usize = 6;

/// The keyed tile: per cell its rank (role), pilot bits.
pub struct TileKey {
    /// rank[cell] (cell = cy * T + cx): < N_PILOT pilot, < N_PILOT + N_HDR header, else payload.
    rank: Vec<u32>,
    /// Pilot cells (cx, cy, +1 / -1).
    pilots: Vec<(u16, u16, f64)>,
    /// Pilot bit per pilot rank.
    pilot_bit: Vec<u8>,
    /// The orientation swaps the axes (a quarter turn or a transpose): the
    /// multi-lane decoder then reads AC10 for AC01 and AC20 for AC02. Signs do
    /// not matter, QIM's lattices are symmetric.
    pub transposed: bool,
}

impl TileKey {
    pub fn new(passphrase: &str) -> Result<TileKey, StegoError> {
        use hmac::{Hmac, KeyInit, Mac};
        use sha2::Sha256;
        let base = crypto::derive_fortress_structural_key(passphrase)?;
        let mut mac = <Hmac<Sha256>>::new_from_slice(&base).expect("HMAC accepts any key length");
        mac.update(b"phasm-fortress-tile-v2");
        let mut seed = [0u8; 32];
        seed.copy_from_slice(&mac.finalize().into_bytes());
        let mut rng = ChaCha20Rng::from_seed(seed);
        let mut rank = vec![u32::MAX; T * T];
        // Header: HDR_BITS keyed cells of the sub-tile, repeated in each one;
        // rank N_PILOT + s * HDR_BITS + b carries bit b (copy s).
        let mut sub: Vec<usize> = (0..HDR_PERIOD * HDR_PERIOD).collect();
        for i in 0..HDR_BITS {
            let j = rng.gen_range08(i as u32..sub.len() as u32) as usize;
            sub.swap(i, j);
        }
        let per_row = T / HDR_PERIOD;
        for s in 0..per_row * per_row {
            let (ox, oy) = ((s % per_row) * HDR_PERIOD, (s / per_row) * HDR_PERIOD);
            for (b, &c) in sub[..HDR_BITS].iter().enumerate() {
                let cell = (oy + c / HDR_PERIOD) * T + ox + c % HDR_PERIOD;
                rank[cell] = (N_PILOT + s * HDR_BITS + b) as u32;
            }
        }
        // Pilots, then payload slots: a keyed permutation of the other cells.
        let mut perm: Vec<u32> = (0..(T * T) as u32).filter(|&c| rank[c as usize] == u32::MAX).collect();
        for i in (1..perm.len()).rev() {
            let j = rng.gen_range08(0..=(i as u32)) as usize;
            perm.swap(i, j);
        }
        for (k, &c) in perm.iter().enumerate() {
            rank[c as usize] = if k < N_PILOT { k } else { k + N_HDR } as u32;
        }
        let pilot_bit: Vec<u8> = (0..N_PILOT).map(|_| (rng.gen08::<u32>() & 1) as u8).collect();
        let pilots = (0..N_PILOT)
            .map(|k| {
                let c = perm[k] as usize;
                ((c % T) as u16, (c / T) as u16, if pilot_bit[k] == 0 { 1.0 } else { -1.0 })
            })
            .collect();
        Ok(TileKey { rank, pilots, pilot_bit, transposed: false })
    }

    /// The keyed parts (rank table, pilot bits), for the layout goldens
    /// (`stego::rng_golden_tests`).
    #[cfg(test)]
    pub(crate) fn keyed_parts(&self) -> (&[u32], &[u8]) {
        (&self.rank, &self.pilot_bit)
    }

    #[inline]
    pub(crate) fn rank_at(&self, bx: usize, by: usize) -> usize {
        self.rank[(by & TM) * T + (bx & TM)] as usize
    }

    /// The key as seen in an image turned or mirrored by `d`: cell roles and
    /// pilots moved inside the tile (the turn's or flip's constant offset is
    /// part of the tile offset the pilot search finds). With it, every decode
    /// step works unchanged on a turned or mirrored image.
    pub fn oriented(&self, d: Dihedral) -> TileKey {
        let mut rank = vec![0u32; T * T];
        for y in 0..T {
            for x in 0..T {
                let (rx, ry) = d.fwd(x, y);
                rank[ry * T + rx] = self.rank[y * T + x];
            }
        }
        let pilots = self
            .pilots
            .iter()
            .map(|&(px, py, sign)| {
                let (x, y) = d.fwd(px as usize, py as usize);
                (x as u16, y as u16, sign)
            })
            .collect();
        let (x, y) = d.fwd(1, 0);
        let transposed = (x == 0) != self.transposed && y != 0;
        TileKey { rank, pilots, pilot_bit: self.pilot_bit.clone(), transposed }
    }
}

/// One of the 8 orientations of the tile: quarter turns (0-3) and the same
/// mirrored (4-7). The checkerboard sync pattern looks the same in all of
/// them, so the grid sync finds a turned or mirrored image's grid; the pilots
/// tell the orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Dihedral(pub u8);

impl Dihedral {
    pub const ALL: [Dihedral; 8] =
        [Dihedral(0), Dihedral(1), Dihedral(2), Dihedral(3), Dihedral(4), Dihedral(5), Dihedral(6), Dihedral(7)];

    /// Received tile cell of encoder tile cell (x, y), up to the tile offset.
    #[inline]
    pub fn fwd(self, x: usize, y: usize) -> (usize, usize) {
        let n = |v: usize| (T - v) & TM;
        match self.0 {
            0 => (x, y),
            1 => (n(y), x),
            2 => (n(x), n(y)),
            3 => (y, n(x)),
            4 => (n(x), y),
            5 => (y, x),
            6 => (x, n(y)),
            _ => (n(y), n(x)),
        }
    }
}

/// Amplitude (gray levels) of the checkerboard sync pattern in busy blocks.
/// 2.0 (sync cost about 0.66 dB landscape / 0.52 dB portrait). With the
/// whitened detector 1.25 decodes every ordinary edit too (0.35 dB), but small
/// crop pieces need the stronger pattern: 16 covers, crops keeping 50-8 % of
/// the area, 4 message lengths decoded 179 / 198 / 206 of 315 at 1.25 / 1.5 /
/// 2.0 (35.16 / 35.07 / 34.86 dB). The maintainer chose 2.0 (2026-11-02;
/// eval/runs/2026-11-02-armor-geometry-2/results/adopted_heavy_crops_amp.txt).
pub const SYNC_AMP: f64 = 2.0;

/// Block count (1600 x 1067 px) at which the sync amplitude is `SYNC_AMP`.
const REF_BLOCKS: f64 = 26_800.0;

/// Share of the sync amplitude a block carries *before* its QIM, from its
/// Watson base factor (0.3 flat .. 1.5 busy): none up to 0.5, full from 1.0.
/// Not in flat blocks: there neighbouring cover averages are alike, the offset
/// would decide their QIM rounding together and print the checkerboard at a
/// full QIM step. The rest of [`FLAT_SYNC_SHARE`] goes on *after* the QIM.
fn sync_weight(wb: f64) -> f64 {
    ((wb - 0.5) / 0.5).clamp(0.0, 1.0)
}

/// Share of the sync amplitude added after the QIM in flat blocks: a small
/// offset off the QIM lattice (it costs that much decision margin, in blocks a
/// re-save barely moves), so smooth images carry the pattern too.
const FLAT_SYNC_SHARE: f64 = 0.5;

fn hdr_bits(format: usize, tier_idx: usize, rc: usize) -> [u8; HDR_BITS] {
    debug_assert!(format < 2 && tier_idx < 64 && rc < 4);
    let v = (format << 8) | (tier_idx << 2) | rc;
    std::array::from_fn(|i| ((v >> (HDR_BITS - 1 - i)) & 1) as u8)
}

pub(crate) fn coded_len(tier: usize) -> usize {
    convcode::coded_len(cc_code(false), 8 * tier)
}

/// Payload blocks of a `bw` x `bh` real block grid (tile offset 0).
fn payload_blocks(key: &TileKey, bw: usize, bh: usize) -> usize {
    let mut n = 0;
    for by in 0..bh {
        for bx in 0..bw {
            if key.rank_at(bx, by) >= N_PILOT + N_HDR {
                n += 1;
            }
        }
    }
    n
}

/// Payload blocks a key leaves in a `bw` x `bh` grid, as a key-free floor.
///
/// A key's count depends on where its 640 pilot and header cells fall among
/// the tile cells the grid repeats more often (the partial tiles at the
/// edges). That number is hypergeometric over keys, so the floor is its mean
/// minus [`FLOOR_SIGMAS`] standard deviations: no passphrase has fewer
/// payload blocks in practice (one-sided P < 1e-15). A worst-case bound (all
/// reserved cells on the most repeated ones) cost one Fortress tier on three
/// of nine common cover sizes.
///
/// The capacity and the encoder's fits-in-Fortress decision use the floor, so
/// the Fortress maximum is a property of the image alone and the apps' HUD
/// (which has no passphrase) shows exactly the sub-mode core writes. The
/// encoder's repetition uses the key's real count.
fn payload_blocks_floor(bw: usize, bh: usize) -> usize {
    let (qx, rx, qy, ry) = (bw / T, bw % T, bh / T, bh % T);
    // (multiplicity of a tile cell in the grid, number of such cells)
    let classes = [
        ((qx + 1) * (qy + 1), rx * ry),
        ((qx + 1) * qy, rx * (T - ry)),
        (qx * (qy + 1), (T - rx) * ry),
        (qx * qy, (T - rx) * (T - ry)),
    ];
    // Reserved blocks = sum over the reserved cells of their multiplicity m:
    // n draws without replacement from N cells, so mean n E[m] and variance
    // n (N - n) / (N - 1) Var[m], m the multiplicity of a uniform cell.
    let (big_n, n) = ((T * T) as f64, (N_PILOT + N_HDR) as f64);
    let (mut e1, mut e2) = (0.0f64, 0.0f64);
    for (m, cells) in classes {
        let p = cells as f64 / big_n;
        e1 += m as f64 * p;
        e2 += (m * m) as f64 * p;
    }
    let var = n * (big_n - n) / (big_n - 1.0) * (e2 - e1 * e1).max(0.0);
    let reserved = n * e1 + FLOOR_SIGMAS * var.sqrt();
    ((bw * bh) as f64 - reserved).floor().max(0.0) as usize
}

/// Standard deviations [`payload_blocks_floor`] keeps below the key mean.
const FLOOR_SIGMAS: f64 = 8.0;

/// Whether a `bw` x `bh` block grid can hold Fortress (T or TL) at all: both
/// sides reach a full header sub-tile, so every header bit has a copy (a
/// thinner cover leaves some header bits unwritten).
pub fn fortress_size_ok(bw: usize, bh: usize) -> bool {
    bw >= HDR_PERIOD && bh >= HDR_PERIOD
}

/// Largest Fortress T frame (bytes) this image carries, for any passphrase
/// (see [`payload_blocks_floor`]).
pub fn t_max_frame_bytes(img: &JpegImage) -> usize {
    let (bw, bh) = super::pipeline::real_luma_blocks(img);
    if !fortress_size_ok(bw, bh) {
        return 0;
    }
    let pb = payload_blocks_floor(bw, bh);
    cc_tiers()
        .into_iter()
        .filter(|&t| {
            let n = coded_len(t);
            n <= N_SLOTS && pb / n >= R_MIN
        })
        .max()
        .unwrap_or(0)
}

/// Largest Fortress frame (bytes) this image carries, for any passphrase:
/// Fortress T, or Fortress TL ([`fortress_tl::Plan`](super::fortress_tl::Plan))
/// when that holds more.
pub fn max_frame_bytes(img: &JpegImage) -> usize {
    t_max_frame_bytes(img).max(super::fortress_tl::Plan::new(img).max_frame_bytes())
}

/// Fortress T encode (frame already built and encrypted; the image already
/// pre-settled to QF75 like Fortress). With `tl` (the cover's
/// [`Plan`](super::fortress_tl::Plan), computed before the pre-settle like
/// the capacity) the payload goes on the AC lanes instead: Fortress TL.
pub fn encode(
    img: &mut JpegImage,
    frame_bytes: &[u8],
    passphrase: &str,
    sync_amp: f64,
    tl: Option<&super::fortress_tl::Plan>,
) -> Result<FortressEncodeResult, StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qtv = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?.values;
    let qt_dc = qtv[0];
    let (bw, bh) = super::pipeline::real_luma_blocks(img);
    let key = TileKey::new(passphrase)?;
    let tiers = cc_tiers();
    let (tier_idx, tier) = tiers
        .iter()
        .copied()
        .enumerate()
        .find(|&(_, t)| t >= frame_bytes.len())
        .ok_or(StegoError::MessageTooLarge)?;
    let n = coded_len(tier);
    // A frame within the capacity always goes out as that format, as the
    // capacity promised (it is computed key-free, see `payload_blocks_floor`
    // and `fortress_tl::Plan`).
    let (format, r, rc) = match tl {
        Some(plan) => {
            if !plan.fits(n) || (plan.bw, plan.bh) != (bw, bh) {
                return Err(StegoError::MessageTooLarge);
            }
            let r = plan.copies(n);
            (1, r, super::fortress_tl::class(r))
        }
        None => {
            let pb = payload_blocks(&key, bw, bh);
            if !fortress_size_ok(bw, bh) || n > N_SLOTS || payload_blocks_floor(bw, bh).max(pb) / n < R_MIN {
                return Err(StegoError::MessageTooLarge);
            }
            // Should a key ever fall below the floor, some bits get one copy fewer.
            let r = (pb / n).max(1);
            (0, r, r_class(r))
        }
    };
    let mut padded = frame_bytes.to_vec();
    padded.resize(tier, 0);
    let coded = convcode::encode(cc_code(false), &frame::bytes_to_bits(&padded));
    let params = adaptive_params(R_CLASS[rc]);
    let hdr = hdr_bits(format, tier_idx, rc);
    let ratio = q75_ratio(&qtv);
    // Smaller images carry the pattern over fewer blocks: up to 2x the
    // amplitude below the 1600 x 1067 reference so it stays detectable.
    let amp = sync_amp * (REF_BLOCKS / (bw * bh).max(1) as f64).sqrt().clamp(1.0, 2.0);
    let grid = img.dct_grid_mut(0);
    for by in 0..bh {
        for bx in 0..bw {
            let k = key.rank_at(bx, by);
            let wb = watson_base_factor(block_energy(grid.block(by, bx), &ratio) / E_REF);
            // Checkerboard sync pattern, added before the QIM (kept in
            // expectation, the QIM decoder unaffected), only in busy blocks.
            let w = sync_weight(wb);
            let sign = if (bx + by) % 2 == 0 { 1.0 } else { -1.0 };
            let avg = dc_to_avg(grid.get(by, bx, 0, 0), qt_dc) + sign * amp * w;
            let new = if k < N_PILOT {
                qim_embed_avg(avg, HEADER_QIM_STEP, key.pilot_bit[k])
            } else if k < N_PILOT + N_HDR {
                qim_embed_avg(avg, HEADER_QIM_STEP, hdr[(k - N_PILOT) % HDR_BITS])
            } else if format == 1 {
                // Fortress TL: no DC payload; the mean carries only the sync.
                avg
            } else {
                let j = k - N_PILOT - N_HDR;
                let f = remap_watson(wb, params.watson_lo, params.watson_hi);
                qim_embed_avg(avg, params.base_step * f, coded[j % n])
            };
            // Not on pilot and header blocks: there the offset would cost the
            // step-12 lattice a third of its margin on a small smooth cover
            // (the amplitude doubles there), enough to sink the pilots.
            let new = if k < N_PILOT + N_HDR { new } else { new + sign * amp * FLAT_SYNC_SHARE * (1.0 - w) };
            grid.set(by, bx, 0, 0, avg_to_dc(new, qt_dc));
        }
    }
    if let Some(plan) = tl {
        super::fortress_tl::embed(grid, &qtv, &key, plan, &coded, n, rc);
    }
    Ok(FortressEncodeResult { repetition_factor: r, parity_symbols: 0 })
}

/// A field of block averages (Fortress DC domain), Watson base factors and
/// erasures, in some block frame whose offset to the encoder grid is unknown.
pub struct BlockField {
    pub bw: usize,
    pub bh: usize,
    pub avg: Vec<f64>,
    pub base: Vec<f64>,
    pub erased: Vec<bool>,
    /// The Fortress TL lanes (AC01, AC10, AC11, AC02, AC20; orthonormal DCT
    /// units), filled with the Watson factors once the pilots matched.
    pub ac: Vec<[f64; 5]>,
}

impl BlockField {
    /// Block averages from the received grid's DC coefficients (the image's
    /// own block grid); Watson factors are left for [`Self::fill_base`].
    pub fn from_image(img: &JpegImage) -> Result<BlockField, StegoError> {
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt_dc = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?.values[0];
        let grid = img.dct_grid(0);
        let (bw, bh) = super::pipeline::real_luma_blocks(img);
        let mut avg = Vec::with_capacity(bw * bh);
        for by in 0..bh {
            for bx in 0..bw {
                avg.push(dc_to_avg(grid.get(by, bx, 0, 0), qt_dc));
            }
        }
        Ok(BlockField { bw, bh, avg, base: Vec::new(), erased: vec![false; bw * bh], ac: Vec::new() })
    }

    /// The Watson base factors (Fortress T) and the AC lanes (Fortress TL) of
    /// a [`Self::from_image`] field, needed once the pilots matched.
    pub fn fill_base(&mut self, img: &JpegImage) -> Result<(), StegoError> {
        use super::fortress_tl::LANES;
        let qt_id = img.frame_info().components[0].quant_table_id as usize;
        let qt = img.quant_table(qt_id).ok_or(StegoError::NoLuminanceChannel)?.values;
        let ratio = q75_ratio(&qt);
        let grid = img.dct_grid(0);
        let mut base = Vec::with_capacity(self.bw * self.bh);
        let mut ac = Vec::with_capacity(self.bw * self.bh);
        for by in 0..self.bh {
            for bx in 0..self.bw {
                let b = grid.block(by, bx);
                base.push(watson_base_factor(block_energy(b, &ratio) / E_REF));
                ac.push(std::array::from_fn(|i| b[LANES[i]] as f64 * qt[LANES[i]] as f64));
            }
        }
        self.base = base;
        self.ac = ac;
        Ok(())
    }

    /// From blocks sampled through a grid map (energies, when sampled, in
    /// QF75 units).
    pub fn from_sampled(s: super::gridsync::SampledBlocks) -> BlockField {
        let base = s.energy.iter().map(|&e| watson_base_factor(e / E_REF)).collect();
        BlockField { bw: s.bw, bh: s.bh, avg: s.avg, base, erased: s.erased, ac: Vec::new() }
    }
}

/// Reference AC energy of the Watson masking (QF75 units): a typical median
/// block of a textured 1600 px photo. Fortress and Fortress CC divide by the
/// image's own median instead, which a crop changes (it can move by 10x when
/// a sky leaves the frame); Fortress T must read the same factors from any
/// crop, so the reference is fixed.
const E_REF: f64 = 100.0;

/// Per-coefficient factor from a table's units to IJG QF75 units.
fn q75_ratio(qt: &[u16; 64]) -> [f64; 64] {
    let q75 = super::pipeline::compute_jpeg_qt(&super::pipeline::JPEG_BASE_LUMINANCE_QT, 75);
    std::array::from_fn(|k| qt[k] as f64 / q75[k] as f64)
}

/// AC energy of a block in IJG QF75 units (coefficients requantized with
/// `ratio`, |c| >= 2 counted), as Fortress' Watson masking measures it.
fn block_energy(block: &[i16], ratio: &[f64; 64]) -> f64 {
    let mut e = 0.0;
    for k in 1..64 {
        let c = (block[k] as f64 * ratio[k]).round();
        if c.abs() >= 2.0 {
            e += c * c;
        }
    }
    e
}

/// Blocks sampled through a map, with what the payload decoders need once
/// the pilots matched: the Watson base factors from the AC energy in QF75
/// units (Fortress T, as [`gridsync::sample_blocks`](super::gridsync::sample_blocks)
/// measures it) and the Fortress TL lanes (the orthonormal DCT, all-ones
/// table). Also the map-frame origin (i0, j0) of the field.
fn sample_full(l: &super::gridsync::Luma<f32>, m: &super::gridsync::GridMap) -> (BlockField, i64, i64) {
    use super::fortress_tl::LANES;
    use crate::codec::jpeg::pixels::dct_block_f64;
    let s = super::gridsync::sample_blocks(l, m, None);
    let q75 = super::pipeline::compute_jpeg_qt(&super::pipeline::JPEG_BASE_LUMINANCE_QT, 75);
    let ones = [1u16; 64];
    let (i0, j0, bw) = (s.i0, s.j0, s.bw);
    let row = |j: usize, base: &mut [f64], ac: &mut [[f64; 5]]| {
        for i in 0..bw {
            if s.erased[j * bw + i] {
                continue;
            }
            let Some(px) = super::gridsync::sample_block(l, m, i0 + i as i64, j0 + j as i64) else { continue };
            let c = dct_block_f64(&px, &q75);
            let mut e = 0.0;
            for &v in &c[1..] {
                if v.abs() >= 2 {
                    e += v as f64 * v as f64;
                }
            }
            base[i] = watson_base_factor(e / E_REF);
            let u = dct_block_f64(&px, &ones);
            ac[i] = std::array::from_fn(|t| u[LANES[t]] as f64);
        }
    };
    let mut base = vec![0.3f64; s.bw * s.bh];
    let mut ac = vec![[0.0f64; 5]; s.bw * s.bh];
    let cw = bw.max(1);
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        base.par_chunks_mut(cw).zip(ac.par_chunks_mut(cw)).enumerate().for_each(|(j, (b, a))| row(j, b, a));
    }
    #[cfg(not(feature = "parallel"))]
    for (j, (b, a)) in base.chunks_mut(cw).zip(ac.chunks_mut(cw)).enumerate() {
        row(j, b, a);
    }
    (BlockField { bw: s.bw, bh: s.bh, avg: s.avg, base, erased: s.erased, ac }, i0, j0)
}

/// Best tile offset (ox, oy) and its pilot standard score: received block
/// (bx, by) is encoder block (bx + ox, by + oy) modulo T.
pub fn find_offset(field: &BlockField, key: &TileKey) -> (usize, usize, f64) {
    let (fold, cnt) = fold_field(field);
    // Pilot at encoder cell (px, py) sits at received cell (px - ox, py - oy).
    let score_row = |oy: usize| -> (usize, f64) {
        let mut best = (0usize, f64::MIN);
        for ox in 0..T {
            let (mut s, mut v) = (0.0f64, 0.0f64);
            for &(px, py, sign) in &key.pilots {
                let c = ((py as usize).wrapping_sub(oy) & TM) * T + ((px as usize).wrapping_sub(ox) & TM);
                s += sign * fold[c];
                v += cnt[c];
            }
            // Null: soft values uniform on [-1, 1], variance 1/3 per block.
            let z = if v > 0.0 { s / (v / 3.0).sqrt() } else { 0.0 };
            if z > best.1 {
                best = (ox, z);
            }
        }
        best
    };
    #[cfg(feature = "parallel")]
    let rows: Vec<(usize, f64)> = {
        use rayon::prelude::*;
        (0..T).into_par_iter().map(score_row).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<(usize, f64)> = (0..T).map(score_row).collect();
    let mut best = (0, 0, f64::MIN);
    for (oy, &(ox, z)) in rows.iter().enumerate() {
        if z > best.2 {
            best = (ox, oy, z);
        }
    }
    best
}

/// The field folded into one tile: normalised step-12 soft values and block
/// counts per tile cell.
fn fold_field(field: &BlockField) -> (Vec<f64>, Vec<f64>) {
    let mut fold = vec![0.0f64; T * T];
    let mut cnt = vec![0.0f64; T * T];
    for by in 0..field.bh {
        for bx in 0..field.bw {
            let i = by * field.bw + bx;
            if field.erased[i] {
                continue;
            }
            let c = (by & TM) * T + (bx & TM);
            fold[c] += qim_extract_soft(field.avg[i], HEADER_QIM_STEP) / (HEADER_QIM_STEP / 2.0);
            cnt[c] += 1.0;
        }
    }
    (fold, cnt)
}

/// The pilot standard score of `key` at offset (`ox`, `oy`) (as [`find_offset`]).
fn pilot_z(fold: &[f64], cnt: &[f64], key: &TileKey, ox: usize, oy: usize) -> f64 {
    let (mut s, mut v) = (0.0f64, 0.0f64);
    for &(px, py, sign) in &key.pilots {
        let c = ((py as usize).wrapping_sub(oy) & TM) * T + ((px as usize).wrapping_sub(ox) & TM);
        s += sign * fold[c];
        v += cnt[c];
    }
    if v > 0.0 { s / (v / 3.0).sqrt() } else { 0.0 }
}

/// [`find_offset`] over all 8 orientations ([`Dihedral`]): the best
/// orientation, offset and standard score.
///
/// The scores of all T^2 offsets are a circular cross-correlation of the fold
/// with the pilot pattern, computed with the in-house FFT (2 forward and 2
/// inverse 128 x 128 transforms per orientation, about the cost of one direct
/// search); the winner's score is then recomputed exactly.
pub fn find_offset_any(field: &BlockField, key: &TileKey) -> (Dihedral, usize, usize, f64) {
    use super::fft2d::{fft2d, ifft2d, Spectrum2D};
    let (fold, cnt) = fold_field(field);
    let f_fold = fft2d(&fold, T, T);
    let f_cnt = fft2d(&cnt, T, T);
    // sum_n a[n] b[n + o] = IFFT(conj(A) B)[o]
    let corr = |a: &Spectrum2D, b: &[f64]| -> Vec<f64> {
        let fb = fft2d(b, T, T);
        let data = a.data.iter().zip(&fb.data).map(|(x, y)| x.conj() * y).collect();
        ifft2d(&Spectrum2D { data, width: T, height: T })
    };
    let one = |d: Dihedral| -> (Dihedral, usize, usize, f64) {
        let (mut sgn, mut ind) = (vec![0.0f64; T * T], vec![0.0f64; T * T]);
        for &(px, py, sign) in &key.pilots {
            let (x, y) = d.fwd(px as usize, py as usize);
            sgn[y * T + x] = sign;
            ind[y * T + x] = 1.0;
        }
        let (s, v) = (corr(&f_fold, &sgn), corr(&f_cnt, &ind));
        let mut best = (d, 0, 0, f64::MIN);
        for o in 0..T * T {
            if v[o] > 0.5 {
                let z = s[o] / (v[o] / 3.0).sqrt();
                if z > best.3 {
                    best = (d, o % T, o / T, z);
                }
            }
        }
        best
    };
    #[cfg(feature = "parallel")]
    let all: Vec<_> = {
        use rayon::prelude::*;
        Dihedral::ALL.par_iter().map(|&d| one(d)).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let all: Vec<_> = Dihedral::ALL.iter().map(|&d| one(d)).collect();
    let mut best = all[0];
    for &b in &all[1..] {
        if b.3 > best.3 {
            best = b;
        }
    }
    let z = pilot_z(&fold, &cnt, &key.oriented(best.0), best.1, best.2);
    (best.0, best.1, best.2, z)
}

/// Decode a Fortress T payload from a block field (any frame).
pub fn decode_field(
    field: &BlockField,
    key: &TileKey,
    passphrase: &str,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let (d, ox, oy, z) = find_offset_any(field, key);
    if z < Z_MIN {
        return Err(StegoError::FrameCorrupted);
    }
    decode_at(field, &key.oriented(d), ox, oy, passphrase)
}

/// Measurement only: per sync candidate (coarse score, pilot z, header value
/// read, number of header cells seen per bit (min), payload slots seen).
#[doc(hidden)]
pub fn debug_geometric(img: &JpegImage, passphrase: &str) -> Vec<(f64, f64, u16, usize, usize)> {
    use super::gridsync::{checker_candidates, checker_map, sample_blocks, Luma};
    let Some((luma, stride, _)) = crate::codec::jpeg::pixels::jpeg_to_luma_f32(img) else { return Vec::new() };
    let fi = img.frame_info();
    let l = Luma { px: &luma, stride, w: fi.width as usize, h: fi.height as usize };
    let key = TileKey::new(passphrase).unwrap();
    let Ok(wfull) = super::gridsync::whitened_full(&l) else { return Vec::new() };
    let lr = &Luma { px: &wfull, stride: l.w, w: l.w, h: l.h };
    let mut out = Vec::new();
    for c in checker_candidates(&l) {
        let Some(m) = checker_map(lr, &c) else { continue };
        let field = BlockField::from_sampled(sample_blocks(&l, &m, None));
        let (ox, oy, z) = find_offset(&field, &key);
        let mut hs = [0.0f64; HDR_BITS];
        let mut hc = [0usize; HDR_BITS];
        let mut seen = vec![false; N_SLOTS];
        for by in 0..field.bh {
            for bx in 0..field.bw {
                let i = by * field.bw + bx;
                if field.erased[i] {
                    continue;
                }
                let k = key.rank_at(bx + ox, by + oy);
                if (N_PILOT..N_PILOT + N_HDR).contains(&k) {
                    hs[(k - N_PILOT) % HDR_BITS] += qim_extract_soft(field.avg[i], HEADER_QIM_STEP);
                    hc[(k - N_PILOT) % HDR_BITS] += 1;
                } else if k >= N_PILOT + N_HDR {
                    seen[k - N_PILOT - N_HDR] = true;
                }
            }
        }
        let v: u16 = hs.iter().fold(0u16, |a, &s| (a << 1) | (s < 0.0) as u16);
        out.push((c.score, z, v, *hc.iter().min().unwrap(), seen.iter().filter(|&&s| s).count()));
    }
    out
}

/// Decode a block field whose tile offset is known.
pub fn decode_at(
    field: &BlockField,
    key: &TileKey,
    ox: usize,
    oy: usize,
    passphrase: &str,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let rank_of = |bx: usize, by: usize| key.rank_at(bx + ox, by + oy);
    // Header: soft sums per bit.
    let mut hs = [0.0f64; HDR_BITS];
    let mut hc = [0usize; HDR_BITS];
    for by in 0..field.bh {
        for bx in 0..field.bw {
            let i = by * field.bw + bx;
            if field.erased[i] {
                continue;
            }
            let k = rank_of(bx, by);
            if (N_PILOT..N_PILOT + N_HDR).contains(&k) {
                hs[(k - N_PILOT) % HDR_BITS] += qim_extract_soft(field.avg[i], HEADER_QIM_STEP);
                hc[(k - N_PILOT) % HDR_BITS] += 1;
            }
        }
    }
    let v: usize = hs.iter().fold(0usize, |a, &s| (a << 1) | (s < 0.0) as usize);
    // A crop narrower than a header sub-tile can miss every copy of some bits:
    // those are tried both ways (header values out of range fail before the
    // Viterbi pass).
    let unseen: Vec<usize> = (0..HDR_BITS).filter(|&b| hc[b] == 0).collect();
    if unseen.len() > MAX_UNSEEN_HDR_BITS {
        return Err(StegoError::FrameCorrupted);
    }
    // Per unseen-bit pattern: the read header, then (frame failed) with its
    // least reliable seen bit flipped, then its second least reliable: a weak
    // header bit costs a Viterbi pass, not the message. Only reached once the
    // pilots matched.
    let mut order: Vec<usize> = (0..HDR_BITS).filter(|&b| hc[b] > 0).collect();
    order.sort_by(|&a, &b| hs[a].abs().total_cmp(&hs[b].abs()));
    let flips: Vec<Option<usize>> = std::iter::once(None).chain(order.iter().take(2).map(|&b| Some(b))).collect();
    let mut first_err = None;
    for flip in flips {
        for pat in 0..1usize << unseen.len() {
            let v = unseen.iter().enumerate().fold(v, |v, (j, &b)| {
                let m = 1 << (HDR_BITS - 1 - b);
                if pat >> j & 1 == 1 { v | m } else { v & !m }
            });
            let v = flip.map_or(v, |b| v ^ (1 << (HDR_BITS - 1 - b)));
            match decode_payload_at(field, key, ox, oy, passphrase, v >> 8, (v >> 2) & 63, v & 3) {
                Ok(r) => return Ok(r),
                Err(StegoError::Cancelled) => return Err(StegoError::Cancelled),
                Err(e) => {
                    first_err.get_or_insert(e);
                }
            }
        }
    }
    Err(first_err.unwrap_or(StegoError::FrameCorrupted))
}

/// Payload decode at offset (`ox`, `oy`) for header values `format`
/// (0 = Fortress T, DC payload; 1 = Fortress TL, AC lanes), `tier_idx`, `rc`.
#[allow(clippy::too_many_arguments)]
fn decode_payload_at(
    field: &BlockField,
    key: &TileKey,
    ox: usize,
    oy: usize,
    passphrase: &str,
    format: usize,
    tier_idx: usize,
    rc: usize,
) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let rank_of = |bx: usize, by: usize| key.rank_at(bx + ox, by + oy);
    let tiers = cc_tiers();
    if tier_idx >= tiers.len() {
        return Err(StegoError::FrameCorrupted);
    }
    let tier = tiers[tier_idx];
    let n = coded_len(tier);
    let (soft, used, r) = if format == 1 {
        if n > super::fortress_tl::N_SLOTS {
            return Err(StegoError::FrameCorrupted);
        }
        let (soft, used) = super::fortress_tl::soft(field, key, ox, oy, n, rc);
        (soft, used, used / n)
    } else {
        if n > N_SLOTS {
            return Err(StegoError::FrameCorrupted);
        }
        let r = R_CLASS[rc];
        let params = adaptive_params(r);
        let mut soft = vec![0.0f64; n];
        let mut used = 0usize;
        for by in 0..field.bh {
            for bx in 0..field.bw {
                let i = by * field.bw + bx;
                if field.erased[i] {
                    continue;
                }
                let k = rank_of(bx, by);
                if k >= N_PILOT + N_HDR {
                    let f = remap_watson(field.base[i], params.watson_lo, params.watson_hi);
                    let step = params.base_step * f;
                    soft[(k - N_PILOT - N_HDR) % n] += qim_extract_soft(field.avg[i], step) / (step / 2.0);
                    used += 1;
                }
            }
        }
        (soft, used, r)
    };
    let bits = convcode::decode(cc_code(false), &soft, 8 * tier);
    let bytes = frame::bits_to_bytes(&bits);
    let use_compact = passphrase.is_empty();
    if bytes.len() < 2 {
        return Err(StegoError::FrameCorrupted);
    }
    let pt_len = u16::from_be_bytes([bytes[0], bytes[1]]) as usize;
    let total = if use_compact { 2 + pt_len + 16 + 4 } else { 2 + 16 + 12 + pt_len + 16 + 4 };
    if pt_len == 0 || total > tier {
        return Err(StegoError::FrameCorrupted);
    }
    let parsed = if use_compact {
        frame::parse_fortress_compact_frame(&bytes[..total])
    } else {
        frame::parse_frame(&bytes[..total])
    }
    .map_err(|_| StegoError::FrameCorrupted)?;
    let plaintext = crypto::decrypt(&parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce)
        .map_err(|_| StegoError::DecryptionFailed)?;
    let len = parsed.plaintext_len as usize;
    if len > plaintext.len() {
        return Err(StegoError::FrameCorrupted);
    }
    let payload = crate::stego::payload::decode_payload(&plaintext[..len]).map_err(|_| StegoError::FrameCorrupted)?;
    let avg_abs = soft.iter().map(|x| x.abs()).sum::<f64>() / used.max(1) as f64;
    let stats = ecc::RsDecodeStats { total_errors: 0, error_capacity: 0, max_block_errors: 0, num_blocks: 1 };
    let mut q = DecodeQuality::from_rs_stats_with_signal(&stats, r.min(255) as u8, 0, avg_abs, 1.0);
    q.fortress_used = true;
    Ok((payload, q))
}

/// Fortress T on the received block grid (untouched, recompressed, or cropped
/// by whole blocks).
pub fn decode_aligned(img: &JpegImage, passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    let key = TileKey::new(passphrase)?;
    let mut field = BlockField::from_image(img)?;
    let (d, ox, oy, z) = find_offset_any(&field, &key);
    if z < Z_MIN {
        return Err(StegoError::FrameCorrupted);
    }
    field.fill_base(img)?;
    decode_at(&field, &key.oriented(d), ox, oy, passphrase)
}

/// Lattice fit of a map at a known absolute tile offset (map-frame block
/// (i, j) is tile cell (i + oxa, j + oya)): the signed step-12 soft values of
/// the pilot blocks plus the unsigned ones of the header blocks. Samples only
/// those blocks (about 4 % of the image).
fn pilot_fit(l: &super::gridsync::Luma<f32>, m: &super::gridsync::GridMap, key: &TileKey, oxa: i64, oya: i64) -> f64 {
    let (i0, j0, bw, bh) = m.block_range(l.w, l.h);
    let t = T as i64;
    let row = |j: i64| -> f64 {
        let mut s = 0.0;
        for i in i0..i0 + bw as i64 {
            let k = key.rank_at((i + oxa).rem_euclid(t) as usize, (j + oya).rem_euclid(t) as usize);
            if k >= N_PILOT + N_HDR {
                continue;
            }
            let Some(px) = super::gridsync::sample_block(l, m, i, j) else { continue };
            let avg = px.iter().sum::<f64>() / 64.0 - 128.0;
            let soft = qim_extract_soft(avg, HEADER_QIM_STEP) / (HEADER_QIM_STEP / 2.0);
            s += if k < N_PILOT {
                if key.pilot_bit[k] == 0 { soft } else { -soft }
            } else {
                soft.abs()
            };
        }
        s
    };
    let js: Vec<i64> = (j0..j0 + bh as i64).collect();
    #[cfg(feature = "parallel")]
    let rows: Vec<f64> = {
        use rayon::prelude::*;
        js.par_iter().map(|&j| row(j)).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<f64> = js.iter().map(|&j| row(j)).collect();
    rows.iter().sum()
}

/// Polish a map whose tile offset is known: coordinate ascent of the pilot
/// fit over rotation, scale and translation (about the image centre), with a
/// parabola step per coordinate, three rounds of halving steps.
#[doc(hidden)]
pub fn polish(
    l: &super::gridsync::Luma<f32>,
    m: super::gridsync::GridMap,
    key: &TileKey,
    oxa: i64,
    oya: i64,
) -> super::gridsync::GridMap {
    let c = [l.w as f64 / 2.0, l.h as f64 / 2.0];
    let mut cur = m;
    let mut f0 = pilot_fit(l, &cur, key, oxa, oya);
    let mut steps = [0.03f64.to_radians(), 5e-4, 0.6, 0.6];
    for _ in 0..3 {
        for p in 0..4 {
            let h = steps[p];
            let mk = |t: f64| match p {
                0 => cur.perturbed(c, t, 0.0, [0.0, 0.0]),
                1 => cur.perturbed(c, 0.0, t, [0.0, 0.0]),
                2 => cur.perturbed(c, 0.0, 0.0, [t, 0.0]),
                _ => cur.perturbed(c, 0.0, 0.0, [0.0, t]),
            };
            let fp = pilot_fit(l, &mk(h), key, oxa, oya);
            let fm = pilot_fit(l, &mk(-h), key, oxa, oya);
            let den = fp - 2.0 * f0 + fm;
            let t = if den < 0.0 {
                (0.5 * (fm - fp) / den * h).clamp(-h, h)
            } else if fp > fm {
                h
            } else {
                -h
            };
            let cand = mk(t);
            let fc = pilot_fit(l, &cand, key, oxa, oya);
            if fc > f0 {
                cur = cand;
                f0 = fc;
            }
        }
        for s in steps.iter_mut() {
            *s *= 0.5;
        }
    }
    cur
}

/// Fortress T after a geometric change: find the checkerboard sync pattern in
/// the received luma (key-free; an image without it stops here), sample the
/// encoder blocks through each candidate map, find the tile offset with the
/// pilots and decode; if that fails, polish the map with the pilots and retry.
pub fn decode_geometric(img: &JpegImage, passphrase: &str) -> Result<(PayloadData, DecodeQuality), StegoError> {
    use super::gridsync::{checker_candidates, checker_map, sample_blocks, GridMap, Luma};
    let (luma, stride, _) = crate::codec::jpeg::pixels::jpeg_to_luma_f32(img).ok_or(StegoError::NoLuminanceChannel)?;
    let fi = img.frame_info();
    let l = Luma { px: &luma, stride, w: fi.width as usize, h: fi.height as usize };
    let cands = checker_candidates(&l);
    if cands.is_empty() {
        return Err(StegoError::FrameCorrupted);
    }
    let key = TileKey::new(passphrase)?;
    let wfull = super::gridsync::whitened_full(&l)?;
    let lr = &Luma { px: &wfull, stride: l.w, w: l.w, h: l.h };
    let mut err = StegoError::FrameCorrupted;
    let done = |m: &GridMap, (p, mut q): (PayloadData, DecodeQuality)| {
        let (rot, s) = m.rotation_scale();
        q.geometry_corrected = true;
        q.estimated_rotation_deg = rot as f32;
        q.estimated_scale = s as f32;
        (p, q)
    };
    let t = T as i64;
    for c in &cands {
        crate::stego::progress::check_cancelled()?;
        let Some(m) = checker_map(lr, c) else { continue };
        // The received grid itself was tried by the aligned decoder.
        if m.is_identity() {
            continue;
        }
        let s = sample_blocks(&l, &m, None);
        let (i0, j0) = (s.i0, s.j0);
        let (d, ox, oy, z) = find_offset_any(&BlockField::from_sampled(s), &key);
        if z < Z_MIN {
            continue;
        }
        let key = key.oriented(d);
        // The pilots matched: sample again with the AC energies (Watson) and
        // the Fortress TL lanes.
        let (field, _, _) = sample_full(&l, &m);
        match decode_at(&field, &key, ox, oy, passphrase) {
            Ok(r) => return Ok(done(&m, r)),
            Err(StegoError::DecryptionFailed) => err = StegoError::DecryptionFailed,
            Err(_) => {}
        }
        // The pilots found the tile: polish the map with them and retry.
        let (oxa, oya) = (ox as i64 - i0, oy as i64 - j0);
        let pm = polish(&l, m, &key, oxa, oya);
        let (field2, i2, j2) = sample_full(&l, &pm);
        let (ox2, oy2) = ((i2 + oxa).rem_euclid(t) as usize, (j2 + oya).rem_euclid(t) as usize);
        match decode_at(&field2, &key, ox2, oy2, passphrase) {
            Ok(r) => return Ok(done(&pm, r)),
            Err(StegoError::DecryptionFailed) => err = StegoError::DecryptionFailed,
            Err(_) => {}
        }
    }
    Err(err)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The key-free floor stays below every key's real count, and it is
    /// exact when the grid is a whole number of tiles.
    #[test]
    fn payload_blocks_floor_is_below_every_key() {
        let keys: Vec<TileKey> = ["pw", "", "a much longer passphrase", "x"]
            .iter()
            .map(|p| TileKey::new(p).unwrap())
            .collect();
        for (bw, bh) in [(200, 134), (134, 200), (80, 60), (135, 169), (128, 128), (256, 128), (300, 7), (5, 5)] {
            let min = payload_blocks_floor(bw, bh);
            for key in &keys {
                let real = payload_blocks(key, bw, bh);
                assert!(min <= real, "{bw}x{bh}: min {min} > real {real}");
                if bw % T == 0 && bh % T == 0 {
                    assert_eq!(min, real, "{bw}x{bh}");
                }
            }
        }
    }

    /// Every HDR_PERIOD x HDR_PERIOD block window, at any offset, holds each
    /// header bit exactly once (small covers and crops keep the whole header),
    /// and the roles add up: N_PILOT pilots, N_HDR header cells per tile.
    #[test]
    fn header_is_in_every_window() {
        for pass in ["pw", "", "other", "a much longer passphrase"] {
            let key = TileKey::new(pass).unwrap();
            let pilots = (0..T * T).filter(|&c| (key.rank[c] as usize) < N_PILOT).count();
            let hdr = (0..T * T).filter(|&c| (N_PILOT..N_PILOT + N_HDR).contains(&(key.rank[c] as usize))).count();
            assert_eq!((pilots, hdr), (N_PILOT, N_HDR), "{pass:?}");
            for (x0, y0) in [(0, 0), (5, 17), (31, 31), (100, 3), (127, 127)] {
                let mut seen = [0usize; HDR_BITS];
                for y in y0..y0 + HDR_PERIOD {
                    for x in x0..x0 + HDR_PERIOD {
                        let k = key.rank_at(x, y);
                        if (N_PILOT..N_PILOT + N_HDR).contains(&k) {
                            seen[(k - N_PILOT) % HDR_BITS] += 1;
                        }
                    }
                }
                assert_eq!(seen, [1; HDR_BITS], "{pass:?} window at ({x0}, {y0})");
            }
        }
    }
}
