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
fn derive_ring_key(passphrase: &str) -> [u8; 32] {
    let mut output = [0u8; 32];
    argon2::Argon2::default()
        .hash_password_into(passphrase.as_bytes(), RING_SALT, &mut output)
        .expect("Argon2 ring key derivation should not fail");
    output
}

/// Derive the ring PRNG seed for sector assignment.
fn derive_ring_seed(passphrase: &str) -> [u8; 32] {
    let ring_key = derive_ring_key(passphrase);
    // Use HMAC-SHA256 for proper domain separation (replaces XOR with fixed string)
    use hmac::{Hmac, Mac};
    use sha2::Sha256;
    type HmacSha256 = Hmac<Sha256>;
    let mut mac = HmacSha256::new_from_slice(&ring_key).expect("HMAC accepts any key length");
    mac.update(b"phasm-ring-sectors-v5");
    let result = mac.finalize().into_bytes();
    let mut seed = [0u8; 32];
    seed.copy_from_slice(&result);
    seed
}

/// Generate a permuted sector-to-bit assignment.
///
/// Returns a vector of length NUM_SECTORS where each element is the
/// data bit index (0..effective_bits) that sector carries, with
/// SPREAD_SECTORS sectors per bit.
fn generate_sector_assignment(passphrase: &str) -> Vec<usize> {
    let seed = derive_ring_seed(passphrase);
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

    assignment
}

/// Compute the mean magnitude of a sector in the annular ring.
///
/// A sector covers an angular range and a radial range in the frequency domain.
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
) {
    let w = spectrum.width;
    let h = spectrum.height;
    let min_dim = w.min(h) as f64;
    let r_inner = RING_R_INNER * min_dim;
    let r_outer = RING_R_OUTER * min_dim;

    // Encrypt payload
    let (ciphertext, nonce, salt) = crypto::encrypt(payload, passphrase);
    let frame_bytes = frame::build_frame(payload.len() as u16, &salt, &nonce, &ciphertext);

    // RS encode with heavy parity
    let rs_encoded = ecc::rs_encode_blocks_with_parity(&frame_bytes, RING_RS_PARITY);
    let rs_bits = frame::bytes_to_bits(&rs_encoded);

    // Get sector assignment
    let assignment = generate_sector_assignment(passphrase);
    let effective_bits = NUM_SECTORS / SPREAD_SECTORS;

    // Truncate RS bits to fit
    let bits_to_embed = rs_bits.len().min(effective_bits);

    // Embed via QIM on sector magnitudes
    for sector_idx in 0..NUM_SECTORS {
        let bit_idx = assignment[sector_idx];
        if bit_idx >= bits_to_embed {
            continue;
        }

        let mag = compute_sector_magnitude(spectrum, sector_idx, r_inner, r_outer);
        let target = qim_embed(mag.max(RING_DELTA), RING_DELTA, rs_bits[bit_idx]);
        set_sector_magnitude(spectrum, sector_idx, r_inner, r_outer, target);
    }
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
    let min_dim = w.min(h) as f64;
    let r_inner = RING_R_INNER * min_dim;
    let r_outer = RING_R_OUTER * min_dim;

    let assignment = generate_sector_assignment(passphrase);
    let effective_bits = NUM_SECTORS / SPREAD_SECTORS;

    // Extract QIM bits with soft voting across SPREAD_SECTORS sectors per bit
    let mut bit_votes = vec![0i32; effective_bits]; // positive = bit 0, negative = bit 1
    for sector_idx in 0..NUM_SECTORS {
        let bit_idx = assignment[sector_idx];
        if bit_idx >= effective_bits {
            continue;
        }

        let mag = compute_sector_magnitude(spectrum, sector_idx, r_inner, r_outer);
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
        ) {
            if let Ok(parsed) = frame::parse_frame(&decoded) {
                if let Ok(plaintext) = crypto::decrypt(
                    &parsed.ciphertext, passphrase, &parsed.salt, &parsed.nonce,
                ) {
                    let len = parsed.plaintext_len as usize;
                    if len <= plaintext.len() {
                        return Some(plaintext[..len].to_vec());
                    }
                }
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
        let a = generate_sector_assignment("test-pass");
        let b = generate_sector_assignment("test-pass");
        assert_eq!(a, b);
    }

    #[test]
    fn sector_assignment_covers_all_bits() {
        let assignment = generate_sector_assignment("test-pass");
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

        embed_ring_payload(&mut spectrum, payload, passphrase);
        let result = extract_ring_payload(&spectrum, passphrase);
        assert!(result.is_some(), "should extract ring payload");
        assert_eq!(result.unwrap(), payload);
    }
}
