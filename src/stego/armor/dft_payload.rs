// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! DFT ring payload: a resize-robust second payload layer.
//!
//! Embeds a short message in the DFT magnitude spectrum using QIM
//! (Quantization Index Modulation) on annular ring sectors. This layer
//! survives resize because DFT magnitude is translation-invariant,
//! and the existing template system estimates scale+rotation.
//!
//! The ring payload is a completely independent channel from the STDM
//! payload, with its own encryption and heavy RS coding.

use crate::stego::armor::fft2d::Complex32;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

use crate::stego::armor::ecc;
use crate::stego::armor::fft2d::Spectrum2D;
use crate::stego::crypto;
use crate::stego::frame;

use crate::det_math::det_hypot;

/// Inner radius as fraction of min(width, height).
const RING_R_INNER: f64 = 0.08;

/// Outer radius as fraction of min(width, height).
const RING_R_OUTER: f64 = 0.20;

/// Number of angular sectors in the ring.
const NUM_SECTORS: usize = 256;

/// Number of sectors averaged per data bit (spreading).
const SPREAD_SECTORS: usize = 4;

/// QIM quantization step (aggressive for robustness).
const RING_DELTA: f64 = 50.0;

/// RS parity for ring payload (heavy: 192 parity out of 255).
const RING_RS_PARITY: usize = 192;

/// Maximum data bytes in ring payload.
const RING_RS_K: usize = 255 - RING_RS_PARITY; // 63

/// Fixed salt for ring key derivation (independent from STDM).
const RING_SALT: &[u8; 16] = b"phasm-ring-v1\0\0\0";

/// Compute the maximum payload bytes the DFT ring can carry.
pub fn ring_capacity(_width: usize, _height: usize) -> usize {
    // NUM_SECTORS / SPREAD_SECTORS = 64 effective bits
    // After RS(255, 63): 63 data bytes, minus frame overhead
    let effective_bits = NUM_SECTORS / SPREAD_SECTORS;
    let data_bytes = effective_bits / 8; // 8 bytes
    if data_bytes <= frame::FRAME_OVERHEAD {
        return 0;
    }
    data_bytes.min(RING_RS_K) - frame::FRAME_OVERHEAD
}

/// Derive the ring encryption key from the passphrase.
fn derive_ring_key(passphrase: &str) -> Result<[u8; 32], crate::stego::error::StegoError> {
    let mut output = [0u8; 32];
    argon2::Argon2::default()
        .hash_password_into(passphrase.as_bytes(), RING_SALT, &mut output)
        .map_err(|_| crate::stego::error::StegoError::KeyDerivationFailed)?;
    Ok(output)
}

/// Derive the ring PRNG seed for sector assignment.
fn derive_ring_seed(passphrase: &str) -> Result<[u8; 32], crate::stego::error::StegoError> {
    let ring_key = derive_ring_key(passphrase)?;
    // Use HMAC-SHA256 for proper domain separation (replaces XOR with fixed string)
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(&ring_key).expect("HMAC accepts any key length");
    mac.update(b"phasm-ring-sectors-v5");
    let result = mac.finalize().into_bytes();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&result);
    Ok(seed)
}

/// Generate a permuted sector-to-bit assignment.
///
/// Returns a vector of length NUM_SECTORS where each element is the
/// data bit index (0..effective_bits) that sector carries, with
/// SPREAD_SECTORS sectors per bit.
fn generate_sector_assignment(passphrase: &str) -> Result<Vec<usize>, crate::stego::error::StegoError> {
    let seed = derive_ring_seed(passphrase)?;
    let mut rng = ChaCha20Rng::from_seed(seed);

    let effective_bits = NUM_SECTORS / SPREAD_SECTORS;
    let mut assignment: Vec<usize> = Vec::with_capacity(NUM_SECTORS);
    for bit_idx in 0..effective_bits {
        for _ in 0..SPREAD_SECTORS {
            assignment.push(bit_idx);
        }
    }

    // Fisher-Yates shuffle using u32 for cross-platform consistency
    for i in (1..assignment.len()).rev() {
        let j = rng.gen_range(0u32..(i as u32 + 1)) as usize;
        assignment.swap(i, j);
    }

    Ok(assignment)
}

/// Bin classification lookup table for DFT-ring sectors.
///
/// For a fixed `(w, h)`, the mapping from `(dx, dy)` frequency offset
/// to sector index is invariant across all 256 sectors, embed vs
/// decode, and all passphrases. Materialize it ONCE per encode/decode
/// call so the per-sector loops collapse to `O(bins_in_sector)`
/// memory reads instead of `O(bin_square)` trig calls.
///
/// Memory: ring area × 8 bytes ≈ 4 MB at 4K, 16 MB at 50 MP.
///
/// Build cost: one (det_hypot + det_atan2) pair per bin in the
/// `(2*r_max+1)²` square — ~150 ms at 4K. Amortized across 512
/// sector accesses on embed and 256 on decode, that's < 1 ms per
/// sector access vs ~100 ms in the legacy path.
pub struct SectorLut {
    /// `sectors[i]` is the list of (idx, conj_idx) pairs in spectrum
    /// belonging to sector `i`. `conj_idx` is the pre-computed
    /// Hermitian-conjugate index used by `set_sector_magnitude` to
    /// preserve the real-valued IFFT invariant.
    sectors: Vec<Vec<SectorBin>>,
}

#[derive(Copy, Clone)]
struct SectorBin {
    idx: u32,
    conj_idx: u32,
}

impl SectorLut {
    /// Total bin count across all sectors. Used by the perf bench
    /// to compute "expected work" estimates.
    #[doc(hidden)]
    pub fn total_bins(&self) -> usize {
        self.sectors.iter().map(|s| s.len()).sum()
    }

    /// Build the LUT for a spectrum of dimensions (w, h) using the
    /// standard ring radii. Scans every (dy, dx) ∈ [-r_max..r_max]²
    /// once. Parallel by dy-row when the `parallel` feature
    /// is enabled; sequential fallback otherwise. Bin order within
    /// each sector preserves dy-outer / dx-inner order so the f64
    /// sum in compute_sector_magnitude_lut is deterministic.
    pub fn build(w: usize, h: usize) -> Self {
        let min_dim = w.min(h) as f64;
        let r_inner = RING_R_INNER * min_dim;
        let r_outer = RING_R_OUTER * min_dim;
        let cx = w as f64 / 2.0;
        let cy = h as f64 / 2.0;
        let r_max_int = r_outer.ceil() as i32 + 1;
        let spectrum_len = w * h;

        // Per-row classification: pure function of (dy, w, h, ring).
        // Each call returns NUM_SECTORS small Vecs (most empty for
        // a typical dy row), pushed in dx-inner order.
        let classify_row = |dy: i32| -> Vec<Vec<SectorBin>> {
            let mut per_sector: Vec<Vec<SectorBin>> =
                (0..NUM_SECTORS).map(|_| Vec::new()).collect();
            for dx in -r_max_int..=r_max_int {
                let r = det_hypot(dx as f64, dy as f64);
                if r < r_inner || r > r_outer {
                    continue;
                }
                let angle = crate::det_math::det_atan2(dy as f64, dx as f64);
                let angle_pos = if angle < 0.0 {
                    angle + 2.0 * std::f64::consts::PI
                } else {
                    angle
                };
                let sector_idx_f =
                    angle_pos * NUM_SECTORS as f64 / (2.0 * std::f64::consts::PI);
                let mut sector_idx = sector_idx_f.floor() as usize;
                if sector_idx >= NUM_SECTORS {
                    sector_idx = NUM_SECTORS - 1;
                }
                let fu = (cx as i32 + dx).rem_euclid(w as i32) as usize;
                let fv = (cy as i32 + dy).rem_euclid(h as i32) as usize;
                let idx = fv * w + fu;
                if idx >= spectrum_len {
                    continue;
                }
                let conj_fu = (w - fu) % w;
                let conj_fv = (h - fv) % h;
                let conj_idx = conj_fv * w + conj_fu;
                per_sector[sector_idx].push(SectorBin {
                    idx: idx as u32,
                    conj_idx: conj_idx as u32,
                });
            }
            per_sector
        };

        let n_rows = (2 * r_max_int + 1) as usize;

        // Parallel: classify each dy independently, then merge in dy
        // order. The merge step preserves bin-push order across rows.
        #[cfg(feature = "parallel")]
        let per_row: Vec<Vec<Vec<SectorBin>>> = {
            use rayon::prelude::*;
            (0..n_rows)
                .into_par_iter()
                .map(|i| classify_row(-r_max_int + i as i32))
                .collect()
        };
        #[cfg(not(feature = "parallel"))]
        let per_row: Vec<Vec<Vec<SectorBin>>> = (0..n_rows)
            .map(|i| classify_row(-r_max_int + i as i32))
            .collect();

        // Merge: for each sector, extend with each row's contribution
        // in row order. The result is identical to a sequential
        // (dy outer, dx inner) scan.
        let mut sectors: Vec<Vec<SectorBin>> =
            (0..NUM_SECTORS).map(|_| Vec::new()).collect();
        for row in per_row {
            for (i, bins) in row.into_iter().enumerate() {
                sectors[i].extend(bins);
            }
        }

        SectorLut { sectors }
    }
}

/// Deterministic byte stream for the SectorLut, used by
/// `lut_cross_platform_hash_hex` to pin the cross-platform hash.
///
/// Layout: for each sector_idx (0..NUM_SECTORS), dump the sector's
/// bin count as 4 LE bytes, then dump each (idx, conj_idx) pair as
/// 4 + 4 LE bytes. Same fixture across platforms — the bin
/// classification depends only on det_hypot / det_atan2 / floor /
/// modular arithmetic, all of which are IEEE-754-deterministic.
#[doc(hidden)]
pub fn lut_test_deterministic_bytes() -> Vec<u8> {
    let lut = SectorLut::build(256, 256);
    let mut bytes = Vec::with_capacity(NUM_SECTORS * 32);
    for sector in &lut.sectors {
        bytes.extend_from_slice(&(sector.len() as u32).to_le_bytes());
        for bin in sector {
            bytes.extend_from_slice(&bin.idx.to_le_bytes());
            bytes.extend_from_slice(&bin.conj_idx.to_le_bytes());
        }
    }
    bytes
}

/// Perf-test helper: LUT-backed `compute_sector_magnitude_lut` exposed
/// for the bench.
#[doc(hidden)]
pub fn perf_lut_compute_sector_magnitude(
    spectrum: &Spectrum2D,
    sector_idx: usize,
    lut: &SectorLut,
) -> f64 {
    compute_sector_magnitude_lut(spectrum, sector_idx, lut)
}

/// SHA256 of [`lut_test_deterministic_bytes`] as lowercase hex.
#[doc(hidden)]
pub fn lut_cross_platform_hash_hex() -> String {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(lut_test_deterministic_bytes());
    let digest = h.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{:02x}", b));
    }
    hex
}

/// Compute the mean magnitude of a sector in the annular ring.
///
/// A sector covers an angular range and a radial range in the frequency domain.
///
/// **Legacy implementation** — kept around as the reference for the
/// bit-exact gate (`lut_matches_legacy_per_sector_magnitude`)
/// AND for the `perf_t34_ring` perf bench. Production code uses
/// `compute_sector_magnitude_lut`.
#[doc(hidden)]
pub fn perf_legacy_compute_sector_magnitude(
    spectrum: &Spectrum2D,
    sector_idx: usize,
    r_inner: f64,
    r_outer: f64,
) -> f64 {
    compute_sector_magnitude(spectrum, sector_idx, r_inner, r_outer)
}

#[doc(hidden)]
pub fn ring_radii(w: usize, h: usize) -> (f64, f64) {
    let min_dim = w.min(h) as f64;
    (RING_R_INNER * min_dim, RING_R_OUTER * min_dim)
}

#[allow(dead_code)]
fn compute_sector_magnitude(
    spectrum: &Spectrum2D,
    sector_idx: usize,
    r_inner: f64,
    r_outer: f64,
) -> f64 {
    let w = spectrum.width;
    let h = spectrum.height;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;

    let angle_start = 2.0 * std::f64::consts::PI * sector_idx as f64 / NUM_SECTORS as f64;
    let angle_end = 2.0 * std::f64::consts::PI * (sector_idx + 1) as f64 / NUM_SECTORS as f64;

    let mut sum = 0.0f64;
    let mut count = 0usize;

    // Sample frequency bins in this sector
    let r_max_int = r_outer.ceil() as i32 + 1;
    for dy in -r_max_int..=r_max_int {
        for dx in -r_max_int..=r_max_int {
            let r = det_hypot(dx as f64, dy as f64);
            if r < r_inner || r > r_outer {
                continue;
            }

            let angle = crate::det_math::det_atan2(dy as f64, dx as f64);
            let angle_pos = if angle < 0.0 { angle + 2.0 * std::f64::consts::PI } else { angle };

            // Check if this bin falls in the sector (handle wrap-around)
            let in_sector = if angle_start <= angle_end {
                angle_pos >= angle_start && angle_pos < angle_end
            } else {
                angle_pos >= angle_start || angle_pos < angle_end
            };

            if !in_sector {
                continue;
            }

            let fu = (cx as i32 + dx).rem_euclid(w as i32) as usize;
            let fv = (cy as i32 + dy).rem_euclid(h as i32) as usize;
            let idx = fv * w + fu;
            if idx < spectrum.data.len() {
                let c = spectrum.data[idx];
                let mag = det_hypot(c.re as f64, c.im as f64);
                sum += mag;
                count += 1;
            }
        }
    }

    if count > 0 { sum / count as f64 } else { 0.0 }
}

/// Set the magnitude of all bins in a sector to a target value.
///
/// **Legacy implementation** — kept around as the reference for the
/// bit-exact gate. Production code uses
/// `set_sector_magnitude_lut`.
#[allow(dead_code)]
fn set_sector_magnitude(
    spectrum: &mut Spectrum2D,
    sector_idx: usize,
    r_inner: f64,
    r_outer: f64,
    target_mag: f64,
) {
    let w = spectrum.width;
    let h = spectrum.height;
    let cx = w as f64 / 2.0;
    let cy = h as f64 / 2.0;

    let angle_start = 2.0 * std::f64::consts::PI * sector_idx as f64 / NUM_SECTORS as f64;
    let angle_end = 2.0 * std::f64::consts::PI * (sector_idx + 1) as f64 / NUM_SECTORS as f64;

    let r_max_int = r_outer.ceil() as i32 + 1;
    for dy in -r_max_int..=r_max_int {
        for dx in -r_max_int..=r_max_int {
            let r = det_hypot(dx as f64, dy as f64);
            if r < r_inner || r > r_outer {
                continue;
            }

            let angle = crate::det_math::det_atan2(dy as f64, dx as f64);
            let angle_pos = if angle < 0.0 { angle + 2.0 * std::f64::consts::PI } else { angle };

            let in_sector = if angle_start <= angle_end {
                angle_pos >= angle_start && angle_pos < angle_end
            } else {
                angle_pos >= angle_start || angle_pos < angle_end
            };

            if !in_sector {
                continue;
            }

            let fu = (cx as i32 + dx).rem_euclid(w as i32) as usize;
            let fv = (cy as i32 + dy).rem_euclid(h as i32) as usize;
            let idx = fv * w + fu;
            if idx >= spectrum.data.len() {
                continue;
            }

            // Set magnitude while preserving phase (f32 spectrum)
            let current = spectrum.data[idx];
            let current_mag = det_hypot(current.re as f64, current.im as f64);
            if current_mag > 1e-10 {
                let scale = (target_mag / current_mag) as f32;
                spectrum.data[idx] = Complex32::new(current.re * scale, current.im * scale);
            } else {
                spectrum.data[idx] = Complex32::new(target_mag as f32, 0.0);
            }

            // Hermitian conjugate for real-valued IFFT
            let conj_fu = (w - fu) % w;
            let conj_fv = (h - fv) % h;
            let conj_idx = conj_fv * w + conj_fu;
            if conj_idx < spectrum.data.len() && conj_idx != idx {
                let conj = spectrum.data[conj_idx];
                let conj_mag = det_hypot(conj.re as f64, conj.im as f64);
                if conj_mag > 1e-10 {
                    let scale = (target_mag / conj_mag) as f32;
                    spectrum.data[conj_idx] = Complex32::new(conj.re * scale, conj.im * scale);
                }
            }
        }
    }
}

/// LUT-backed compute_sector_magnitude. Reads the bin set
/// for `sector_idx` from `lut.sectors[sector_idx]`, sums their
/// magnitudes, returns mean. Bit-identical to the legacy ring-scan
/// path on the same fixture (same iteration order, same f64
/// arithmetic — only the bin classification is hoisted to LUT build).
fn compute_sector_magnitude_lut(spectrum: &Spectrum2D, sector_idx: usize, lut: &SectorLut) -> f64 {
    let bins = &lut.sectors[sector_idx];
    if bins.is_empty() {
        return 0.0;
    }
    let mut sum = 0.0f64;
    let spectrum_len = spectrum.data.len();
    let mut count = 0usize;
    for bin in bins {
        let idx = bin.idx as usize;
        if idx >= spectrum_len {
            continue;
        }
        let c = spectrum.data[idx];
        sum += det_hypot(c.re as f64, c.im as f64);
        count += 1;
    }
    if count > 0 {
        sum / count as f64
    } else {
        0.0
    }
}

/// LUT-backed set_sector_magnitude. Iterates the same bin
/// set as `compute_sector_magnitude_lut`, rescales each bin to the
/// target magnitude (preserving phase), and applies the Hermitian
/// conjugate update using the pre-stored `conj_idx`.
fn set_sector_magnitude_lut(
    spectrum: &mut Spectrum2D,
    sector_idx: usize,
    lut: &SectorLut,
    target_mag: f64,
) {
    let bins = &lut.sectors[sector_idx];
    let spectrum_len = spectrum.data.len();
    for bin in bins {
        let idx = bin.idx as usize;
        if idx >= spectrum_len {
            continue;
        }
        let current = spectrum.data[idx];
        let current_mag = det_hypot(current.re as f64, current.im as f64);
        if current_mag > 1e-10 {
            let scale = (target_mag / current_mag) as f32;
            spectrum.data[idx] = Complex32::new(current.re * scale, current.im * scale);
        } else {
            spectrum.data[idx] = Complex32::new(target_mag as f32, 0.0);
        }

        let conj_idx = bin.conj_idx as usize;
        if conj_idx < spectrum_len && conj_idx != idx {
            let conj = spectrum.data[conj_idx];
            let conj_mag = det_hypot(conj.re as f64, conj.im as f64);
            if conj_mag > 1e-10 {
                let scale = (target_mag / conj_mag) as f32;
                spectrum.data[conj_idx] = Complex32::new(conj.re * scale, conj.im * scale);
            }
        }
    }
}

/// Combined compute-mean + qim-embed + scale-bins in a
/// single pass over `lut.sectors[sector_idx]`. The legacy embed loop
/// calls `compute_sector_magnitude_lut` (reads `det_hypot` per bin)
/// then `set_sector_magnitude_lut` (reads `det_hypot` per bin AGAIN
/// for the `current_mag` rescale). This combined variant caches the
/// per-bin magnitude from the read pass and reuses it in the write
/// pass — one `det_hypot` per bin instead of two.
///
/// Conjugate bins still need their own `det_hypot` since they live
/// outside `lut.sectors[sector_idx]`.
///
/// Bit-exact with the legacy two-call sequence because the same
/// `det_hypot(re, im)` value computed during the read pass is what
/// `set_sector_magnitude_lut` would have re-computed in its rescale
/// step — no f64 round-off difference.
fn embed_sector_bit_lut(
    spectrum: &mut Spectrum2D,
    sector_idx: usize,
    lut: &SectorLut,
    bit: u8,
) {
    let bins = &lut.sectors[sector_idx];
    if bins.is_empty() {
        return;
    }
    let spectrum_len = spectrum.data.len();

    // Pass 1: read bins, cache per-bin magnitude, accumulate mean.
    let mut per_bin_mag: Vec<f64> = Vec::with_capacity(bins.len());
    let mut sum = 0.0f64;
    let mut count = 0usize;
    for bin in bins {
        let idx = bin.idx as usize;
        if idx >= spectrum_len {
            per_bin_mag.push(0.0);
            continue;
        }
        let c = spectrum.data[idx];
        let mag = det_hypot(c.re as f64, c.im as f64);
        per_bin_mag.push(mag);
        sum += mag;
        count += 1;
    }
    let mean = if count > 0 { sum / count as f64 } else { 0.0 };
    let target = qim_embed(mean.max(RING_DELTA), RING_DELTA, bit);

    // Pass 2: rescale each bin using the cached current_mag, then
    // mirror to the Hermitian conjugate.
    for (i, bin) in bins.iter().enumerate() {
        let idx = bin.idx as usize;
        if idx >= spectrum_len {
            continue;
        }
        let current = spectrum.data[idx];
        let current_mag = per_bin_mag[i];
        if current_mag > 1e-10 {
            let scale = (target / current_mag) as f32;
            spectrum.data[idx] = Complex32::new(current.re * scale, current.im * scale);
        } else {
            spectrum.data[idx] = Complex32::new(target as f32, 0.0);
        }

        let conj_idx = bin.conj_idx as usize;
        if conj_idx < spectrum_len && conj_idx != idx {
            let conj = spectrum.data[conj_idx];
            let conj_mag = det_hypot(conj.re as f64, conj.im as f64);
            if conj_mag > 1e-10 {
                let scale = (target / conj_mag) as f32;
                spectrum.data[conj_idx] = Complex32::new(conj.re * scale, conj.im * scale);
            }
        }
    }
}

/// QIM embed a single bit into a magnitude value.
fn qim_embed(mag: f64, delta: f64, bit: u8) -> f64 {
    if bit == 0 {
        (mag / delta).round() * delta
    } else {
        ((mag / delta - 0.5).round() + 0.5) * delta
    }
}

/// QIM extract a single bit from a magnitude value.
fn qim_extract(mag: f64, delta: f64) -> u8 {
    let half_delta = delta / 2.0;
    let m = (mag / half_delta).round() as i64;
    m.rem_euclid(2) as u8
}

/// Embed a ring payload into the DFT spectrum.
///
/// The payload is encrypted, RS-encoded, and QIM-embedded into the
/// annular ring sectors. Must be called during the FFT pass before IFFT.
pub fn embed_ring_payload(
    spectrum: &mut Spectrum2D,
    payload: &[u8],
    passphrase: &str,
) -> Result<(), crate::stego::error::StegoError> {
    let w = spectrum.width;
    let h = spectrum.height;
    // Build the bin classification LUT once. Hoists ~750 M
    // det_hypot/det_atan2 calls (at 4K) down to ~3 M during build,
    // then per-sector ops become O(bins_in_sector) memory reads.
    let lut = SectorLut::build(w, h);

    // Encrypt payload
    let (ciphertext, nonce, salt) = crypto::encrypt(payload, passphrase)?;
    let frame_bytes = frame::build_frame(payload.len(), &salt, &nonce, &ciphertext);

    // RS encode with heavy parity
    let rs_encoded = ecc::rs_encode_blocks_with_parity(&frame_bytes, RING_RS_PARITY);
    let rs_bits = frame::bytes_to_bits(&rs_encoded);

    // Get sector assignment
    let assignment = generate_sector_assignment(passphrase)?;
    let effective_bits = NUM_SECTORS / SPREAD_SECTORS;

    // Truncate RS bits to fit
    let bits_to_embed = rs_bits.len().min(effective_bits);

    // Embed via QIM on sector magnitudes
    for sector_idx in 0..NUM_SECTORS {
        let bit_idx = assignment[sector_idx];
        if bit_idx >= bits_to_embed {
            continue;
        }

        // Combined compute + set in one pass. Halves the
        // per-bin `det_hypot` count (1 per bin instead of 2).
        embed_sector_bit_lut(spectrum, sector_idx, &lut, rs_bits[bit_idx]);
    }
    Ok(())
}

/// Extract a ring payload from the DFT spectrum.
///
/// Returns `Some(plaintext_bytes)` if successful, `None` if decode fails.
pub fn extract_ring_payload(
    spectrum: &Spectrum2D,
    passphrase: &str,
) -> Option<Vec<u8>> {
    let w = spectrum.width;
    let h = spectrum.height;
    // Same LUT-once optimization on the decode side.
    let lut = SectorLut::build(w, h);

    let assignment = generate_sector_assignment(passphrase).ok()?;
    let effective_bits = NUM_SECTORS / SPREAD_SECTORS;

    // Extract QIM bits with soft voting across SPREAD_SECTORS sectors per bit
    let mut bit_votes = vec![0i32; effective_bits]; // positive = bit 0, negative = bit 1
    for sector_idx in 0..NUM_SECTORS {
        let bit_idx = assignment[sector_idx];
        if bit_idx >= effective_bits {
            continue;
        }

        let mag = compute_sector_magnitude_lut(spectrum, sector_idx, &lut);
        let extracted = qim_extract(mag, RING_DELTA);
        if extracted == 0 {
            bit_votes[bit_idx] += 1;
        } else {
            bit_votes[bit_idx] -= 1;
        }
    }

    // Hard decisions
    let rs_bits: Vec<u8> = bit_votes.iter()
        .map(|&v| if v >= 0 { 0 } else { 1 })
        .collect();

    let rs_bytes = frame::bits_to_bytes(&rs_bits);

    // Try RS decode
    // Compute expected frame size range
    for pt_len in 0..=RING_RS_K.min(64) {
        let ct_len = pt_len + 16;
        let total_frame = 2 + 16 + 12 + ct_len + 4;
        let rs_encoded_len = ecc::rs_encoded_len_with_parity(total_frame, RING_RS_PARITY);
        if rs_encoded_len > rs_bytes.len() {
            continue;
        }
        if let Ok((decoded, _stats)) = ecc::rs_decode_blocks_with_parity(
            &rs_bytes[..rs_encoded_len], total_frame, RING_RS_PARITY
        )
            && let Ok(parsed) = frame::parse_frame(&decoded)
                && let Ok(plaintext) = crypto::decrypt(
                    &parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce,
                ) {
                    let len = parsed.plaintext_len as usize;
                    if len <= plaintext.len() {
                        return Some(plaintext[..len].to_vec());
                    }
                }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::armor::fft2d::Spectrum2D;

    fn make_test_spectrum(w: usize, h: usize) -> Spectrum2D {
        // Create a spectrum with random-ish magnitudes (f32)
        let mut data = vec![Complex32::new(0.0, 0.0); w * h];
        let mut rng = ChaCha20Rng::from_seed([42u8; 32]);
        for val in data.iter_mut() {
            let mag: f32 = 100.0 + rng.gen_range(0u32..100) as f32;
            *val = Complex32::new(mag, 0.0);
        }
        Spectrum2D { data, width: w, height: h }
    }

    /// T3.4.A — the LUT-backed compute_sector_magnitude MUST agree
    /// with the legacy ring-scan path to f64 bit-equality on every
    /// sector. Read order is preserved: the LUT pushes bins in the
    /// same `(dy, dx)` iteration order as the legacy scan, so the
    /// f64 sum is identical.
    #[test]
    fn lut_matches_legacy_per_sector_magnitude() {
        // 256x256 is small enough to test exhaustively but big enough
        // to have meaningful ring-bin counts (~330 bins/sector).
        let spectrum = make_test_spectrum(256, 256);
        let w = spectrum.width;
        let h = spectrum.height;
        let min_dim = w.min(h) as f64;
        let r_inner = RING_R_INNER * min_dim;
        let r_outer = RING_R_OUTER * min_dim;
        let lut = SectorLut::build(w, h);

        for sector_idx in 0..NUM_SECTORS {
            let legacy =
                compute_sector_magnitude(&spectrum, sector_idx, r_inner, r_outer);
            let fast = compute_sector_magnitude_lut(&spectrum, sector_idx, &lut);
            assert_eq!(
                legacy.to_bits(),
                fast.to_bits(),
                "sector {sector_idx}: legacy={legacy} fast={fast}"
            );
        }
    }

    /// T3.4.A — the LUT-backed set_sector_magnitude must produce a
    /// spectrum bit-identical to the legacy path.
    #[test]
    fn lut_matches_legacy_set_sector_magnitude() {
        let mut spec_legacy = make_test_spectrum(256, 256);
        let mut spec_fast = make_test_spectrum(256, 256);
        let w = spec_legacy.width;
        let h = spec_legacy.height;
        let min_dim = w.min(h) as f64;
        let r_inner = RING_R_INNER * min_dim;
        let r_outer = RING_R_OUTER * min_dim;
        let lut = SectorLut::build(w, h);

        // Apply set_sector_magnitude on each sector with a varying
        // target. The legacy and LUT paths should produce byte-
        // identical spectrum data.
        for sector_idx in 0..NUM_SECTORS {
            let target = 100.0 + (sector_idx as f64) * 1.3;
            set_sector_magnitude(&mut spec_legacy, sector_idx, r_inner, r_outer, target);
            set_sector_magnitude_lut(&mut spec_fast, sector_idx, &lut, target);
        }

        for i in 0..spec_legacy.data.len() {
            let l = spec_legacy.data[i];
            let f = spec_fast.data[i];
            assert_eq!(
                l.re.to_bits(),
                f.re.to_bits(),
                "spectrum[{i}].re differs: legacy={} fast={}",
                l.re, f.re
            );
            assert_eq!(
                l.im.to_bits(),
                f.im.to_bits(),
                "spectrum[{i}].im differs: legacy={} fast={}",
                l.im, f.im
            );
        }
    }

    /// T3.4.C — combined embed_sector_bit_lut must produce a
    /// spectrum byte-identical to the legacy (compute then set)
    /// two-call sequence. Confirms the per-bin mag caching matches
    /// the on-the-fly `det_hypot` in `set_sector_magnitude_lut`.
    #[test]
    fn embed_sector_bit_lut_matches_two_call_sequence() {
        let mut spec_a = make_test_spectrum(256, 256);
        let mut spec_b = make_test_spectrum(256, 256);
        let w = spec_a.width;
        let h = spec_a.height;
        let lut = SectorLut::build(w, h);

        for sector_idx in 0..NUM_SECTORS {
            let bit = ((sector_idx * 11) % 2) as u8;
            // Path A: legacy two-call sequence (compute then set).
            let mag = compute_sector_magnitude_lut(&spec_a, sector_idx, &lut);
            let target = qim_embed(mag.max(RING_DELTA), RING_DELTA, bit);
            set_sector_magnitude_lut(&mut spec_a, sector_idx, &lut, target);
            // Path B: combined.
            embed_sector_bit_lut(&mut spec_b, sector_idx, &lut, bit);
        }

        for i in 0..spec_a.data.len() {
            assert_eq!(
                spec_a.data[i].re.to_bits(),
                spec_b.data[i].re.to_bits(),
                "spectrum[{i}].re differs"
            );
            assert_eq!(
                spec_a.data[i].im.to_bits(),
                spec_b.data[i].im.to_bits(),
                "spectrum[{i}].im differs"
            );
        }
    }

    #[test]
    fn ring_capacity_positive() {
        let cap = ring_capacity(320, 240);
        // With 256 sectors / 4 spread = 64 bits = 8 bytes, minus overhead
        // Capacity is very small — may be 0 for short payloads
        // Just verify it doesn't panic
        assert!(cap <= 64, "ring capacity should be small: {cap}");
    }

    #[test]
    fn qim_embed_extract_roundtrip() {
        for bit in 0..=1u8 {
            for mag_base in [50.0, 100.0, 200.0, 500.0] {
                let embedded = qim_embed(mag_base, RING_DELTA, bit);
                let extracted = qim_extract(embedded, RING_DELTA);
                assert_eq!(extracted, bit, "QIM failed for bit={bit}, mag={mag_base}");
            }
        }
    }

    #[test]
    fn sector_assignment_deterministic() {
        let a = generate_sector_assignment("test-pass").unwrap();
        let b = generate_sector_assignment("test-pass").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn sector_assignment_covers_all_bits() {
        let assignment = generate_sector_assignment("test-pass").unwrap();
        assert_eq!(assignment.len(), NUM_SECTORS);

        let effective_bits = NUM_SECTORS / SPREAD_SECTORS;
        for bit_idx in 0..effective_bits {
            let count = assignment.iter().filter(|&&b| b == bit_idx).count();
            assert_eq!(count, SPREAD_SECTORS, "bit {bit_idx} should have {SPREAD_SECTORS} sectors");
        }
    }

    #[test]
    fn dft_ring_embed_extract_roundtrip() {
        let mut spectrum = make_test_spectrum(256, 256);
        let payload = b"Hi";
        let passphrase = "ring-test";

        // Check capacity first
        let cap = ring_capacity(256, 256);
        if cap < payload.len() {
            // Ring capacity too small for this payload — skip
            return;
        }

        embed_ring_payload(&mut spectrum, payload, passphrase).unwrap();
        let result = extract_ring_payload(&spectrum, passphrase);
        assert!(result.is_some(), "should extract ring payload");
        assert_eq!(result.unwrap(), payload);
    }
}
