// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Reed-Solomon error correction over GF(2^8).
//!
//! Implements RS(255, k) with the primitive polynomial 0x11D (x^8+x^4+x^3+x^2+1).
//! Supports systematic encoding and Berlekamp-Massey decoding with Chien search
//! and Forney algorithm. Shortened codes are used for payloads smaller than k.

/// Primitive polynomial for GF(2^8): x^8 + x^4 + x^3 + x^2 + 1 = 0x11D.
const PRIM_POLY: u16 = 0x11D;

/// Maximum RS block size.
const N_MAX: usize = 255;

/// Default data symbols per block.
const K_DEFAULT: usize = 191;

/// Number of parity symbols (n - k).
const PARITY_LEN: usize = N_MAX - K_DEFAULT; // 64

/// Error correction capability: t = parity_len / 2 = 32.
pub const T_MAX: usize = PARITY_LEN / 2;

/// Fixed parity tiers for adaptive RS (limits decoder search space to 4 attempts).
pub const PARITY_TIERS: [usize; 4] = [64, 128, 192, 240];

// --- GF(2^8) Arithmetic ---

/// Precomputed log and exp tables for GF(2^8).
struct GfTables {
    exp: [u8; 512],
    log: [u8; 256],
}

/// Build log/exp tables at compile time is not possible in const context with loops
/// in stable Rust, so we build them once at runtime.
fn build_gf_tables() -> GfTables {
    let mut exp = [0u8; 512];
    let mut log = [0u8; 256];

    let mut x: u16 = 1;
    for i in 0..255u16 {
        exp[i as usize] = x as u8;
        exp[(i + 255) as usize] = x as u8; // wrap-around for easy modular access
        log[x as usize] = i as u8;
        x <<= 1;
        if x & 0x100 != 0 {
            x ^= PRIM_POLY;
        }
    }
    // log[0] is undefined (log of 0 doesn't exist), leave as 0
    // exp[510] and exp[511] are unused padding
    exp[510] = exp[0];
    exp[511] = exp[1];

    GfTables { exp, log }
}

fn gf_tables() -> &'static GfTables {
    use std::sync::OnceLock;
    static TABLES: OnceLock<GfTables> = OnceLock::new();
    TABLES.get_or_init(build_gf_tables)
}

/// GF(2^8) multiplication.
fn gf_mul(a: u8, b: u8) -> u8 {
    if a == 0 || b == 0 {
        return 0;
    }
    let t = gf_tables();
    let log_sum = t.log[a as usize] as usize + t.log[b as usize] as usize;
    t.exp[log_sum]
}

/// GF(2^8) addition (same as XOR).
fn gf_add(a: u8, b: u8) -> u8 {
    a ^ b
}

/// GF(2^8) multiplicative inverse. Panics if a == 0.
fn gf_inv(a: u8) -> u8 {
    assert_ne!(a, 0, "cannot invert zero in GF(2^8)");
    let t = gf_tables();
    t.exp[255 - t.log[a as usize] as usize]
}

/// GF(2^8) power: a^n.
#[cfg(test)]
fn gf_pow(a: u8, n: u32) -> u8 {
    if a == 0 {
        return if n == 0 { 1 } else { 0 };
    }
    let t = gf_tables();
    let log_a = t.log[a as usize] as u32;
    let exp_idx = (log_a * n) % 255;
    t.exp[exp_idx as usize]
}

/// Evaluate polynomial at x. poly[0] is the highest-degree coefficient.
fn poly_eval(poly: &[u8], x: u8) -> u8 {
    let mut result = 0u8;
    for &coeff in poly {
        result = gf_add(gf_mul(result, x), coeff);
    }
    result
}

/// Multiply two polynomials. poly[0] is highest-degree coefficient.
fn poly_mul(a: &[u8], b: &[u8]) -> Vec<u8> {
    let mut result = vec![0u8; a.len() + b.len() - 1];
    for (i, &ac) in a.iter().enumerate() {
        for (j, &bc) in b.iter().enumerate() {
            result[i + j] = gf_add(result[i + j], gf_mul(ac, bc));
        }
    }
    result
}

// --- Generator Polynomial ---

/// Build the RS generator polynomial g(x) = prod_{i=0}^{2t-1} (x - alpha^i).
/// Returns coefficients from highest to lowest degree.
fn build_gen_poly(parity_len: usize) -> Vec<u8> {
    let t = gf_tables();
    let mut gpoly = vec![1u8]; // Start with 1

    for i in 0..parity_len {
        let root = t.exp[i]; // alpha^i
        gpoly = poly_mul(&gpoly, &[1, root]);
    }
    gpoly
}

fn gen_poly() -> &'static Vec<u8> {
    use std::sync::OnceLock;
    static GEN: OnceLock<Vec<u8>> = OnceLock::new();
    GEN.get_or_init(|| build_gen_poly(PARITY_LEN))
}

/// Cached generator polynomial for a given parity length.
/// Uses OnceLock per tier to avoid recomputing.
fn gen_poly_for(parity_len: usize) -> &'static Vec<u8> {
    use std::sync::OnceLock;
    static GEN_64: OnceLock<Vec<u8>> = OnceLock::new();
    static GEN_128: OnceLock<Vec<u8>> = OnceLock::new();
    static GEN_192: OnceLock<Vec<u8>> = OnceLock::new();
    static GEN_240: OnceLock<Vec<u8>> = OnceLock::new();

    match parity_len {
        64 => GEN_64.get_or_init(|| build_gen_poly(64)),
        128 => GEN_128.get_or_init(|| build_gen_poly(128)),
        192 => GEN_192.get_or_init(|| build_gen_poly(192)),
        240 => GEN_240.get_or_init(|| build_gen_poly(240)),
        _ => {
            // For the default parity, reuse the existing cache
            if parity_len == PARITY_LEN {
                gen_poly()
            } else {
                panic!("unsupported parity length: {parity_len}")
            }
        }
    }
}

// --- Encoding ---

/// RS-encode a single data block (systematic encoding).
///
/// # Arguments
/// - `data`: Data bytes of length <= `K_DEFAULT` (191).
///
/// # Returns
/// A vector of `data.len() + PARITY_LEN` bytes: the original data followed
/// by 64 parity symbols.
///
/// # Panics
/// Panics if `data.len() > K_DEFAULT`.
///
/// For shortened codes (`data.len() < K_DEFAULT`), the data is conceptually
/// zero-padded at the front to K_DEFAULT, encoded, then the padding is removed.
/// The parity symbols are computed over this virtual full-length block.
pub fn rs_encode(data: &[u8]) -> Vec<u8> {
    assert!(
        data.len() <= K_DEFAULT,
        "data length {} exceeds max {}",
        data.len(),
        K_DEFAULT
    );

    let gpoly = gen_poly();
    let parity_len = PARITY_LEN;

    // Systematic encoding: compute remainder of data * x^parity_len / g(x).
    // Work with the actual data length (shortened code).
    let mut shift_reg = vec![0u8; parity_len];

    for &byte in data {
        let feedback = gf_add(byte, shift_reg[0]);
        // Shift left
        for j in 0..parity_len - 1 {
            shift_reg[j] = gf_add(shift_reg[j + 1], gf_mul(feedback, gpoly[j + 1]));
        }
        shift_reg[parity_len - 1] = gf_mul(feedback, gpoly[parity_len]);
    }

    // Output: data || parity
    let mut encoded = Vec::with_capacity(data.len() + parity_len);
    encoded.extend_from_slice(data);
    encoded.extend_from_slice(&shift_reg);
    encoded
}

/// RS-encode an arbitrarily long payload, splitting into [`K_DEFAULT`]-byte blocks.
///
/// Returns the concatenation of all RS-encoded blocks. Each block has
/// `min(remaining_data, K_DEFAULT) + PARITY_LEN` bytes. The last block
/// may be a shortened code if `payload.len() % K_DEFAULT != 0`.
pub fn rs_encode_blocks(payload: &[u8]) -> Vec<u8> {
    let mut encoded = Vec::new();
    for chunk in payload.chunks(K_DEFAULT) {
        encoded.extend_from_slice(&rs_encode(chunk));
    }
    encoded
}

// --- Decoding ---

/// Compute syndromes S_0 .. S_{2t-1} for a received block (FCR=0).
/// poly_eval treats received as highest-degree-first: r(x) = received[0]*x^{n-1} + ...
fn compute_syndromes(received: &[u8]) -> Vec<u8> {
    let tab = gf_tables();
    let two_t = PARITY_LEN;
    let mut syndromes = vec![0u8; two_t];
    for i in 0..two_t {
        syndromes[i] = poly_eval(received, tab.exp[i]); // S_i = r(α^i)
    }
    syndromes
}

fn syndromes_are_zero(syndromes: &[u8]) -> bool {
    syndromes.iter().all(|&s| s == 0)
}

/// Berlekamp-Massey algorithm.
///
/// Returns sigma(x) coefficients in ascending power: sigma[0]=1, sigma[1]=σ_1, etc.
fn berlekamp_massey(syndromes: &[u8]) -> Vec<u8> {
    let n = syndromes.len(); // 2t

    // C(x) = error locator, ascending power
    let mut c = vec![0u8; n + 1];
    c[0] = 1;
    let mut c_len = 1usize;

    // B(x) = previous C, ascending power
    let mut b = vec![0u8; n + 1];
    b[0] = 1;
    let mut b_len = 1usize;

    let mut ell = 0usize; // current error count estimate
    let mut bval = 1u8; // previous discrepancy
    let mut m = 1usize; // step counter

    for r in 0..n {
        // Discrepancy
        let mut delta = syndromes[r];
        for i in 1..c_len {
            delta = gf_add(delta, gf_mul(c[i], syndromes[r - i]));
        }

        if delta == 0 {
            m += 1;
            continue;
        }

        let factor = gf_mul(delta, gf_inv(bval));

        if 2 * ell <= r {
            // Save C before updating (it becomes the new B)
            let old_c = c.clone();
            let old_c_len = c_len;

            let new_len = (b_len + m).max(c_len);
            c_len = new_len;
            for j in 0..b_len {
                c[j + m] = gf_add(c[j + m], gf_mul(factor, b[j]));
            }

            b[..old_c_len].copy_from_slice(&old_c[..old_c_len]);
            for j in old_c_len..b.len() {
                b[j] = 0;
            }
            b_len = old_c_len;
            ell = r + 1 - ell;
            bval = delta;
            m = 1;
        } else {
            let new_len = (b_len + m).max(c_len);
            c_len = new_len;
            for j in 0..b_len {
                c[j + m] = gf_add(c[j + m], gf_mul(factor, b[j]));
            }
            m += 1;
        }
    }

    c[..c_len].to_vec()
}

/// Evaluate polynomial in ascending power format at x.
fn eval_asc(poly: &[u8], x: u8) -> u8 {
    let mut result = 0u8;
    let mut x_pow = 1u8;
    for &coeff in poly {
        result = gf_add(result, gf_mul(coeff, x_pow));
        x_pow = gf_mul(x_pow, x);
    }
    result
}

/// Chien search: find roots of sigma(x) to determine error positions.
///
/// Convention: poly_eval treats the codeword as c(x) = c[0]*x^{n-1} + c[1]*x^{n-2} + ...
/// An error at array index k affects the coefficient of x^{n-1-k}.
/// The error locator polynomial sigma(x) has roots at X_l^{-1} where X_l = α^{n-1-k_l}.
///
/// Returns (gf_pos, array_pos) pairs.
fn chien_search(sigma_asc: &[u8], n: usize) -> Option<Vec<(usize, usize)>> {
    let tab = gf_tables();
    let num_errors = sigma_asc.len() - 1;
    let mut found = Vec::with_capacity(num_errors);

    // Test sigma at α^{-p} for each GF position p = 0..n-1.
    // If sigma(α^{-p}) = 0, error at GF position p → array index n-1-p.
    for p in 0..n {
        let x = if p == 0 {
            1u8
        } else {
            tab.exp[(255 - (p % 255)) % 255] // α^{-p}
        };
        if eval_asc(sigma_asc, x) == 0 {
            found.push((p, n - 1 - p));
        }
    }

    if found.len() != num_errors {
        return None;
    }

    Some(found)
}

/// Forney algorithm: compute error magnitudes.
///
/// With FCR=0: e_l = X_l * Omega(X_l^{-1}) / Sigma'(X_l^{-1})
/// where X_l = α^{gf_pos}, and Omega = S(x) * Sigma(x) mod x^{2t},
/// S(x) = S_0 + S_1*x + S_2*x^2 + ...
fn forney(
    sigma_asc: &[u8],
    syndromes: &[u8],
    found: &[(usize, usize)],
) -> Vec<u8> {
    let tab = gf_tables();
    let two_t = syndromes.len();

    // Omega(x) = S(x) * Sigma(x) mod x^{2t} (ascending power)
    let mut omega = vec![0u8; two_t];
    for i in 0..sigma_asc.len().min(two_t) {
        for j in 0..two_t {
            if i + j < two_t {
                omega[i + j] = gf_add(omega[i + j], gf_mul(sigma_asc[i], syndromes[j]));
            }
        }
    }

    // Formal derivative of sigma (ascending power):
    // d/dx (a_0 + a_1*x + a_2*x^2 + ...) = a_1 + 2*a_2*x + 3*a_3*x^2 + ...
    // In GF(2^m): even multipliers vanish, odd survive (3=1, 5=1, etc.)
    // So sigma'[j] = sigma_asc[j+1] if (j+1) is odd, else 0
    let deriv_len = sigma_asc.len().saturating_sub(1);
    let mut sigma_prime = vec![0u8; deriv_len];
    for i in (1..sigma_asc.len()).step_by(2) {
        sigma_prime[i - 1] = sigma_asc[i];
    }

    let mut magnitudes = Vec::with_capacity(found.len());
    for &(gf_pos, _) in found {
        let x_val = if gf_pos == 0 {
            1u8
        } else {
            tab.exp[gf_pos % 255] // α^{gf_pos} = X_l
        };
        let x_inv = if gf_pos == 0 {
            1u8
        } else {
            tab.exp[(255 - (gf_pos % 255)) % 255] // α^{-gf_pos} = X_l^{-1}
        };

        let omega_val = eval_asc(&omega, x_inv);
        let sp_val = eval_asc(&sigma_prime, x_inv);

        if sp_val == 0 {
            magnitudes.push(0);
            continue;
        }

        // FCR=0: e = X_l * Omega(X_l^{-1}) / Sigma'(X_l^{-1})
        magnitudes.push(gf_mul(x_val, gf_mul(omega_val, gf_inv(sp_val))));
    }

    magnitudes
}

/// Error returned when RS decoding fails (too many errors).
#[derive(Debug, PartialEq)]
pub struct RsDecodeError;

impl core::fmt::Display for RsDecodeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Reed-Solomon: too many errors to correct")
    }
}

/// RS-decode a single block with error correction.
///
/// # Arguments
/// - `received`: Received block of length `data_len + PARITY_LEN`.
/// - `data_len`: Original data length (before parity was appended).
///
/// # Returns
/// A tuple of (corrected data of length `data_len`, number of symbol errors corrected).
///
/// # Panics
/// Panics if `received.len() != data_len + PARITY_LEN`.
///
/// # Errors
/// Returns [`RsDecodeError`] if there are more than `t=32` symbol errors,
/// or if errors fall in the zero-padded region of a shortened code.
///
/// For shortened codes (`data_len < K_DEFAULT`), the received block is
/// conceptually zero-padded at the front to form a full 255-symbol block
/// during syndrome computation.
pub fn rs_decode(received: &[u8], data_len: usize) -> Result<(Vec<u8>, usize), RsDecodeError> {
    let block_len = data_len + PARITY_LEN;
    assert_eq!(
        received.len(),
        block_len,
        "received length {} != expected {}",
        received.len(),
        block_len
    );

    // For shortened codes, prepend zeros to make a full 255-symbol block
    let padding = N_MAX - block_len;
    let mut full_block = vec![0u8; N_MAX];
    full_block[padding..].copy_from_slice(received);

    // Compute syndromes on the full block
    let syndromes = compute_syndromes(&full_block);

    if syndromes_are_zero(&syndromes) {
        return Ok((received[..data_len].to_vec(), 0));
    }

    // Find error locator polynomial (ascending power)
    let sigma_asc = berlekamp_massey(&syndromes);
    let num_errors = sigma_asc.len() - 1;

    if num_errors > T_MAX {
        return Err(RsDecodeError);
    }

    // Chien search: find (gf_pos, array_pos) pairs in the full 255-symbol block
    let found = chien_search(&sigma_asc, N_MAX).ok_or(RsDecodeError)?;

    // Forney: compute error magnitudes
    let magnitudes = forney(&sigma_asc, &syndromes, &found);

    // Apply corrections
    let mut corrected = full_block;
    for (i, &(_, array_pos)) in found.iter().enumerate() {
        if array_pos < padding {
            // Error in the zero-padded region of a shortened code — can't correct
            return Err(RsDecodeError);
        }
        corrected[array_pos] = gf_add(corrected[array_pos], magnitudes[i]);
    }

    // Verify syndromes are now zero
    let check_syndromes = compute_syndromes(&corrected);
    if !syndromes_are_zero(&check_syndromes) {
        return Err(RsDecodeError);
    }

    // Extract data (skip padding)
    Ok((corrected[padding..padding + data_len].to_vec(), num_errors))
}

/// Statistics from RS decoding across all blocks.
#[derive(Debug, Clone, Default)]
pub struct RsDecodeStats {
    /// Total symbol errors corrected across all blocks.
    pub total_errors: usize,
    /// Maximum correctable errors per block (T_MAX × num_blocks).
    pub error_capacity: usize,
    /// Maximum errors found in any single block.
    pub max_block_errors: usize,
    /// Number of RS blocks decoded.
    pub num_blocks: usize,
}

/// RS-decode a payload that was encoded with [`rs_encode_blocks`].
///
/// Splits the encoded data into blocks based on `total_data_len`, decodes
/// each block independently (correcting up to `t=32` symbol errors per block),
/// and concatenates the results. Returns decode stats with error counts.
///
/// # Arguments
/// - `encoded`: The RS-encoded data (output of `rs_encode_blocks`).
/// - `total_data_len`: The original payload length before encoding.
///
/// # Errors
/// Returns [`RsDecodeError`] if any block has too many errors to correct
/// or the encoded data is too short.
pub fn rs_decode_blocks(encoded: &[u8], total_data_len: usize) -> Result<(Vec<u8>, RsDecodeStats), RsDecodeError> {
    let mut decoded = Vec::with_capacity(total_data_len);
    let mut remaining_data = total_data_len;
    let mut offset = 0;
    let mut stats = RsDecodeStats::default();

    while remaining_data > 0 {
        let chunk_data_len = remaining_data.min(K_DEFAULT);
        let block_len = chunk_data_len + PARITY_LEN;

        if offset + block_len > encoded.len() {
            return Err(RsDecodeError);
        }

        let block = &encoded[offset..offset + block_len];
        let (data, errors) = rs_decode(block, chunk_data_len)?;
        decoded.extend_from_slice(&data);

        stats.total_errors += errors;
        stats.num_blocks += 1;
        if errors > stats.max_block_errors {
            stats.max_block_errors = errors;
        }

        offset += block_len;
        remaining_data -= chunk_data_len;
    }

    stats.error_capacity = stats.num_blocks * T_MAX;
    Ok((decoded, stats))
}

/// Return the RS-encoded length for a given data length.
pub fn rs_encoded_len(data_len: usize) -> usize {
    let full_blocks = data_len / K_DEFAULT;
    let remainder = data_len % K_DEFAULT;
    let mut total = full_blocks * (K_DEFAULT + PARITY_LEN);
    if remainder > 0 {
        total += remainder + PARITY_LEN;
    }
    total
}

/// Return the parity length per block.
pub const fn parity_len() -> usize {
    PARITY_LEN
}

// --- Adaptive RS with configurable parity ---

/// RS-encode a single data block with configurable parity length.
///
/// # Arguments
/// - `data`: Data bytes of length <= `255 - parity_len`.
/// - `parity_len`: Number of parity symbols (must be even, <= 240).
///
/// # Returns
/// A vector of `data.len() + parity_len` bytes.
pub fn rs_encode_with_parity(data: &[u8], parity_len: usize) -> Vec<u8> {
    if parity_len == 0 { return data.to_vec(); }
    let k_max = N_MAX - parity_len;
    assert!(
        data.len() <= k_max,
        "data length {} exceeds max {} for parity_len={}",
        data.len(),
        k_max,
        parity_len
    );
    assert!(parity_len <= 240, "parity_len {} exceeds 240", parity_len);

    let gpoly = gen_poly_for(parity_len);
    let mut shift_reg = vec![0u8; parity_len];

    for &byte in data {
        let feedback = gf_add(byte, shift_reg[0]);
        for j in 0..parity_len - 1 {
            shift_reg[j] = gf_add(shift_reg[j + 1], gf_mul(feedback, gpoly[j + 1]));
        }
        shift_reg[parity_len - 1] = gf_mul(feedback, gpoly[parity_len]);
    }

    let mut encoded = Vec::with_capacity(data.len() + parity_len);
    encoded.extend_from_slice(data);
    encoded.extend_from_slice(&shift_reg);
    encoded
}

/// RS-decode a single block with configurable parity length.
///
/// # Arguments
/// - `received`: Received block of length `data_len + parity_len`.
/// - `data_len`: Original data length.
/// - `parity_len`: Parity symbols used during encoding.
///
/// # Returns
/// (corrected data, number of errors corrected).
pub fn rs_decode_with_parity(
    received: &[u8],
    data_len: usize,
    parity_len: usize,
) -> Result<(Vec<u8>, usize), RsDecodeError> {
    let block_len = data_len + parity_len;
    assert_eq!(
        received.len(),
        block_len,
        "received length {} != expected {}",
        received.len(),
        block_len
    );

    let padding = N_MAX - block_len;
    let mut full_block = vec![0u8; N_MAX];
    full_block[padding..].copy_from_slice(received);

    // Compute syndromes with the given parity length
    let tab = gf_tables();
    let mut syndromes = vec![0u8; parity_len];
    for i in 0..parity_len {
        syndromes[i] = poly_eval(&full_block, tab.exp[i]);
    }

    if syndromes.iter().all(|&s| s == 0) {
        return Ok((received[..data_len].to_vec(), 0));
    }

    let t_max = parity_len / 2;
    let sigma_asc = berlekamp_massey(&syndromes);
    let num_errors = sigma_asc.len() - 1;

    if num_errors > t_max {
        return Err(RsDecodeError);
    }

    let found = chien_search(&sigma_asc, N_MAX).ok_or(RsDecodeError)?;
    let magnitudes = forney(&sigma_asc, &syndromes, &found);

    let mut corrected = full_block;
    for (i, &(_, array_pos)) in found.iter().enumerate() {
        if array_pos < padding {
            return Err(RsDecodeError);
        }
        corrected[array_pos] = gf_add(corrected[array_pos], magnitudes[i]);
    }

    // Verify syndromes are now zero
    let mut check_ok = true;
    for i in 0..parity_len {
        if poly_eval(&corrected, tab.exp[i]) != 0 {
            check_ok = false;
            break;
        }
    }
    if !check_ok {
        return Err(RsDecodeError);
    }

    Ok((corrected[padding..padding + data_len].to_vec(), num_errors))
}

/// RS-encode an arbitrarily long payload with configurable parity, splitting into blocks.
pub fn rs_encode_blocks_with_parity(payload: &[u8], parity_len: usize) -> Vec<u8> {
    let k_max = N_MAX - parity_len;
    let mut encoded = Vec::new();
    for chunk in payload.chunks(k_max) {
        encoded.extend_from_slice(&rs_encode_with_parity(chunk, parity_len));
    }
    encoded
}

/// RS-decode a payload encoded with [`rs_encode_blocks_with_parity`].
pub fn rs_decode_blocks_with_parity(
    encoded: &[u8],
    total_data_len: usize,
    parity_len: usize,
) -> Result<(Vec<u8>, RsDecodeStats), RsDecodeError> {
    let k_max = N_MAX - parity_len;
    let t_max = parity_len / 2;
    let mut decoded = Vec::with_capacity(total_data_len);
    let mut remaining_data = total_data_len;
    let mut offset = 0;
    let mut stats = RsDecodeStats::default();

    while remaining_data > 0 {
        let chunk_data_len = remaining_data.min(k_max);
        let block_len = chunk_data_len + parity_len;

        if offset + block_len > encoded.len() {
            return Err(RsDecodeError);
        }

        let block = &encoded[offset..offset + block_len];
        let (data, errors) = rs_decode_with_parity(block, chunk_data_len, parity_len)?;
        decoded.extend_from_slice(&data);

        stats.total_errors += errors;
        stats.num_blocks += 1;
        if errors > stats.max_block_errors {
            stats.max_block_errors = errors;
        }

        offset += block_len;
        remaining_data -= chunk_data_len;
    }

    stats.error_capacity = stats.num_blocks * t_max;
    Ok((decoded, stats))
}

/// Return the RS-encoded length for a given data length and parity length.
pub fn rs_encoded_len_with_parity(data_len: usize, parity_len: usize) -> usize {
    let k_max = N_MAX - parity_len;
    let full_blocks = data_len / k_max;
    let remainder = data_len % k_max;
    let mut total = full_blocks * (k_max + parity_len);
    if remainder > 0 {
        total += remainder + parity_len;
    }
    total
}

/// Choose the best parity tier for a given frame size and embedding capacity (in bits).
///
/// Picks the largest parity from [`PARITY_TIERS`] where the RS-encoded data
/// (in bits) still fits within `num_units`.
pub fn choose_parity_tier(frame_len: usize, num_units: usize) -> usize {
    let mut best = PARITY_TIERS[0]; // fallback to smallest
    for &tier in &PARITY_TIERS {
        let rs_bits = rs_encoded_len_with_parity(frame_len, tier) * 8;
        if rs_bits <= num_units {
            best = tier;
        } else {
            break;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gf_mul_identity() {
        for a in 0..=255u16 {
            assert_eq!(gf_mul(a as u8, 1), a as u8);
            assert_eq!(gf_mul(1, a as u8), a as u8);
        }
    }

    #[test]
    fn gf_mul_zero() {
        for a in 0..=255u16 {
            assert_eq!(gf_mul(a as u8, 0), 0);
            assert_eq!(gf_mul(0, a as u8), 0);
        }
    }

    #[test]
    fn gf_inverse_roundtrip() {
        for a in 1..=255u16 {
            let inv = gf_inv(a as u8);
            assert_eq!(gf_mul(a as u8, inv), 1, "a={a}, inv={inv}");
        }
    }

    #[test]
    fn gf_pow_consistency() {
        let t = gf_tables();
        for a in 1..=255u16 {
            // a^1 == a
            assert_eq!(gf_pow(a as u8, 1), a as u8);
            // a^0 == 1
            assert_eq!(gf_pow(a as u8, 0), 1);
            // a^255 == 1 (Fermat's little theorem for GF(2^8))
            assert_eq!(gf_pow(a as u8, 255), 1, "a={a}");
        }
        let _ = t;
    }

    #[test]
    fn encode_decode_no_errors() {
        let data = b"Hello, Reed-Solomon!";
        let encoded = rs_encode(data);
        let (decoded, errors) = rs_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 0);
    }

    #[test]
    fn encode_decode_with_errors() {
        let data = b"Test message for RS error correction.";
        let mut encoded = rs_encode(data);

        // Introduce 10 symbol errors (well within t=32 correction capability).
        // Note: data.len()=37, so avoid position 37 in the data region to
        // prevent overlap with the first parity error at data.len().
        encoded[0] ^= 0xFF;
        encoded[5] ^= 0xAA;
        encoded[10] ^= 0x55;
        encoded[15] ^= 0x11;
        encoded[20] ^= 0x22;
        encoded[25] ^= 0x33;
        encoded[30] ^= 0x01;
        encoded[data.len()] ^= 0x77; // error in parity
        encoded[data.len() + 10] ^= 0x88;
        encoded[data.len() + 30] ^= 0x99;

        let (decoded, errors) = rs_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 10);
    }

    #[test]
    fn encode_decode_max_correctable() {
        let data = vec![42u8; 100];
        let mut encoded = rs_encode(&data);

        // Introduce exactly t=32 errors
        for i in 0..32 {
            encoded[i * 3] ^= 0xFF;
        }

        let (decoded, errors) = rs_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 32);
    }

    #[test]
    fn too_many_errors_fails() {
        let data = vec![0u8; 50];
        let mut encoded = rs_encode(&data);

        // Introduce 33 errors (exceeds t=32)
        for i in 0..33 {
            encoded[i] ^= 0xFF;
        }

        assert!(rs_decode(&encoded, data.len()).is_err());
    }

    #[test]
    fn shortened_code_works() {
        // Very short data (much less than K_DEFAULT=191)
        let data = b"Hi";
        let encoded = rs_encode(data);
        assert_eq!(encoded.len(), data.len() + PARITY_LEN);

        let (decoded, errors) = rs_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 0);
    }

    #[test]
    fn shortened_code_with_errors() {
        let data = b"Short";
        let mut encoded = rs_encode(data);
        encoded[0] ^= 0xFF;
        encoded[2] ^= 0xAA;

        let (decoded, errors) = rs_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 2);
    }

    #[test]
    fn blocks_roundtrip() {
        // Data larger than K_DEFAULT, requiring multiple blocks
        let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
        let encoded = rs_encode_blocks(&data);

        // Should be 2 blocks: 191+64=255 and 209+64=273 → 528 total
        assert_eq!(encoded.len(), rs_encoded_len(data.len()));

        let (decoded, stats) = rs_decode_blocks(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(stats.total_errors, 0);
    }

    #[test]
    fn blocks_with_errors() {
        let data: Vec<u8> = (0..400).map(|i| (i % 256) as u8).collect();
        let mut encoded = rs_encode_blocks(&data);

        // Corrupt a few bytes in block 1 (starts at 0, len 255)
        encoded[10] ^= 0xFF;
        encoded[100] ^= 0xAA;
        // Block 2 starts at 255 (len 255)
        encoded[260] ^= 0x55;
        encoded[300] ^= 0x11;
        // Block 3 starts at 510 (len 82)
        encoded[520] ^= 0x33;

        let (decoded, stats) = rs_decode_blocks(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(stats.total_errors, 5);
        assert!(stats.max_block_errors <= 2);
    }

    #[test]
    fn empty_data() {
        let data: &[u8] = &[];
        let encoded = rs_encode(data);
        assert_eq!(encoded.len(), PARITY_LEN);
        let (decoded, errors) = rs_decode(&encoded, 0).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 0);
    }

    #[test]
    fn rs_encoded_len_correct() {
        assert_eq!(rs_encoded_len(100), 100 + 64);
        assert_eq!(rs_encoded_len(191), 191 + 64);
        assert_eq!(rs_encoded_len(192), (191 + 64) + (1 + 64));
        // 400 / 191 = 2 full blocks (382), remainder 18 → 3 blocks
        assert_eq!(rs_encoded_len(400), 2 * (191 + 64) + (18 + 64));
    }

    #[test]
    fn rs_encoded_len_edge_cases() {
        assert_eq!(rs_encoded_len(0), 0);
        assert_eq!(rs_encoded_len(1), 1 + 64);
        // Full block boundary
        assert_eq!(rs_encoded_len(191), 191 + 64);
        // Just over one block
        assert_eq!(rs_encoded_len(192), (191 + 64) + (1 + 64));
    }

    #[test]
    fn single_error_full_block() {
        let data = vec![42u8; K_DEFAULT];
        let mut encoded = rs_encode(&data);
        encoded[50] ^= 0x01;
        let (decoded, errors) = rs_decode(&encoded, K_DEFAULT).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 1);
    }

    #[test]
    fn single_error_shortened() {
        let data = b"Short";
        let mut encoded = rs_encode(data);
        encoded[0] ^= 0xFF;
        let (decoded, errors) = rs_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 1);
    }

    #[test]
    fn two_errors_full_block() {
        let data = vec![42u8; K_DEFAULT];
        let mut encoded = rs_encode(&data);
        encoded[0] ^= 0xFF;
        encoded[50] ^= 0xAA;
        let (decoded, errors) = rs_decode(&encoded, K_DEFAULT).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 2);
    }

    #[test]
    fn two_errors_shortened() {
        let data = b"Short";
        let mut encoded = rs_encode(data);
        encoded[0] ^= 0xFF;
        encoded[2] ^= 0xAA;
        let (decoded, errors) = rs_decode(&encoded, data.len()).unwrap();
        assert_eq!(decoded, data);
        assert_eq!(errors, 2);
    }

    #[test]
    fn generator_polynomial_correct() {
        let gpoly = gen_poly();
        // Gen poly should have degree = PARITY_LEN, so length = PARITY_LEN + 1
        assert_eq!(gpoly.len(), PARITY_LEN + 1);
        // Leading coefficient should be 1
        assert_eq!(gpoly[0], 1);
        // All roots alpha^0 .. alpha^{2t-1} should evaluate to 0
        let t = gf_tables();
        for i in 0..PARITY_LEN {
            assert_eq!(poly_eval(gpoly, t.exp[i]), 0, "root alpha^{i} failed");
        }
    }

    // --- Adaptive RS tests ---

    #[test]
    fn adaptive_rs_roundtrip_each_tier() {
        for &parity in &PARITY_TIERS {
            let k_max = N_MAX - parity;
            let data: Vec<u8> = (0..k_max.min(100)).map(|i| (i % 256) as u8).collect();
            let encoded = rs_encode_with_parity(&data, parity);
            assert_eq!(encoded.len(), data.len() + parity);
            let (decoded, errors) = rs_decode_with_parity(&encoded, data.len(), parity).unwrap();
            assert_eq!(decoded, data, "parity={parity}");
            assert_eq!(errors, 0, "parity={parity}");
        }
    }

    #[test]
    fn adaptive_rs_corrects_errors_at_each_tier() {
        for &parity in &PARITY_TIERS {
            let k_max = N_MAX - parity;
            let t = parity / 2;
            let data: Vec<u8> = (0..k_max.min(50)).map(|i| (i % 256) as u8).collect();
            let mut encoded = rs_encode_with_parity(&data, parity);

            // Introduce t/2 errors (well within correction capability)
            let num_errors = (t / 2).min(encoded.len());
            let elen = encoded.len();
            for i in 0..num_errors {
                encoded[i * 2 % elen] ^= 0xFF;
            }

            let (decoded, errors) = rs_decode_with_parity(&encoded, data.len(), parity).unwrap();
            assert_eq!(decoded, data, "parity={parity}");
            assert!(errors > 0, "parity={parity}");
        }
    }

    #[test]
    fn adaptive_rs_blocks_roundtrip() {
        let data: Vec<u8> = (0..200).map(|i| (i % 256) as u8).collect();
        for &parity in &PARITY_TIERS {
            let encoded = rs_encode_blocks_with_parity(&data, parity);
            assert_eq!(encoded.len(), rs_encoded_len_with_parity(data.len(), parity));
            let (decoded, stats) = rs_decode_blocks_with_parity(&encoded, data.len(), parity).unwrap();
            assert_eq!(decoded, data, "parity={parity}");
            assert_eq!(stats.total_errors, 0, "parity={parity}");
        }
    }

    #[test]
    fn rs_encoded_len_with_parity_correct() {
        // With parity=128, k_max=127
        assert_eq!(rs_encoded_len_with_parity(100, 128), 100 + 128);
        assert_eq!(rs_encoded_len_with_parity(127, 128), 127 + 128);
        // 128 bytes at parity=128: k_max=127, so 2 blocks: 127+128 + 1+128
        assert_eq!(rs_encoded_len_with_parity(128, 128), (127 + 128) + (1 + 128));
    }

    #[test]
    fn choose_parity_tier_picks_largest_fitting() {
        // 100-byte frame, 10000 embedding units
        // tier 64: rs_len = 100+64=164 bytes = 1312 bits → fits
        // tier 128: rs_len = 100+128=228 bytes = 1824 bits → fits
        // tier 192: rs_len = 100+192=292 bytes (but k_max=63, 100>63, so 2 blocks)
        //   = 63+192 + 37+192 = 484 bytes = 3872 bits → fits
        // tier 240: rs_len = 100+240 but k_max=15, many blocks
        //   = ceil(100/15)*255 = 7*255 = 1785 bytes = 14280 bits → exceeds 10000
        let tier = choose_parity_tier(100, 10000);
        assert_eq!(tier, 192);
    }
}
