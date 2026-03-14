// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Shadow message embedding via repetition coding in chrominance.
//!
//! Shadow messages provide plausible deniability: multiple messages can be
//! hidden in one image, each with a different passphrase. Under coercion the
//! user reveals the primary passphrase (innocuous message). Shadow messages
//! remain undetectable without their respective passphrases.
//!
//! Shadow layers use **repetition coding** (R=7, majority vote) in the
//! **Cb+Cr chrominance channels**, which are completely independent from the
//! primary Ghost STC that operates on the Y (luminance) channel. This
//! eliminates all interference between shadow and primary layers.
//!
//! Requires color images (at least 2 components). Grayscale images cannot
//! carry shadow messages.

use crate::jpeg::dct::DctGrid;
use crate::jpeg::JpegImage;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;
use crate::stego::payload::{self, FileEntry, PayloadData};
use crate::stego::pipeline::{flat_get, flat_set};
use crate::stego::permute::{self, CoeffPos};
use crate::stego::side_info::nsf5_modify_coefficient;

/// Repetition factor for shadow embedding. Each bit is repeated R times.
/// Tolerates up to floor(R/2) = 3 bit flips per group (42.8% BER).
const SHADOW_R: usize = 7;

/// Maximum frame size in bytes for shadow decode search.
/// Limits the brute-force scan during extraction.
const MAX_SHADOW_FRAME_BYTES: usize = 8192;

/// Bit 31 flag in flat_idx: set = Cr (component 2), clear = Cb (component 1).
const CR_FLAG: u32 = 1 << 31;

/// Collect non-zero AC positions from both Cb and Cr chrominance channels.
///
/// Returns a combined position list. Positions from Cr have bit 31 set in
/// `flat_idx` to distinguish them from Cb positions during embed/extract.
///
/// Requires `img.num_components() >= 2` (at minimum Cb). If 3 components
/// exist, Cr positions are also included.
pub fn collect_chroma_positions(img: &JpegImage) -> Vec<CoeffPos> {
    let mut positions = Vec::new();

    // Cb (component 1)
    if img.num_components() >= 2 {
        collect_nonzero_ac_into(img.dct_grid(1), 0, &mut positions);
    }

    // Cr (component 2)
    if img.num_components() >= 3 {
        collect_nonzero_ac_into(img.dct_grid(2), CR_FLAG, &mut positions);
    }

    positions
}

/// Scan a DctGrid for non-zero AC coefficients and append them to `out`.
/// `flag` is OR'd into each flat_idx (0 for Cb, CR_FLAG for Cr).
fn collect_nonzero_ac_into(grid: &DctGrid, flag: u32, out: &mut Vec<CoeffPos>) {
    let bw = grid.blocks_wide();
    let bt = grid.blocks_tall();
    out.reserve(bt * bw * 16); // rough estimate
    for br in 0..bt {
        for bc in 0..bw {
            let blk = grid.block(br, bc);
            for k in 1..64 {
                if blk[k] != 0 {
                    let flat_idx = ((br * bw + bc) * 64 + k) as u32 | flag;
                    out.push(CoeffPos {
                        flat_idx,
                        cost: 1.0, // uniform cost — not used for shadow
                    });
                }
            }
        }
    }
}

/// Embed a shadow message into chrominance DCT coefficients using repetition coding.
///
/// Operates on Cb+Cr channels to avoid interference with the primary Ghost
/// STC on luminance.
///
/// # Arguments
/// - `img`: Mutable JPEG image (shadow modifies Cb and/or Cr grids).
/// - `positions`: Combined Cb+Cr positions from [`collect_chroma_positions`].
/// - `shadow_pass`: Passphrase for this shadow layer.
/// - `message`: Text message to embed.
/// - `files`: Optional file attachments.
///
/// # Returns
/// The number of positions used.
pub fn shadow_embed(
    img: &mut JpegImage,
    positions: &[CoeffPos],
    shadow_pass: &str,
    message: &str,
    files: &[FileEntry],
) -> Result<usize, StegoError> {
    // 1. Build payload (text + files + compression).
    let payload_bytes = payload::encode_payload(message, files)?;

    // 2. Encrypt payload.
    let (ciphertext, nonce, salt) = crypto::encrypt(&payload_bytes, shadow_pass)?;

    // 3. Build frame.
    let frame_bytes = frame::build_frame(payload_bytes.len(), &salt, &nonce, &ciphertext);
    let frame_bits = frame::bytes_to_bits(&frame_bytes);

    // 4. Derive shadow structural key and permute positions.
    let perm_seed = crypto::derive_shadow_structural_key(shadow_pass)?;
    let mut perm_positions = positions.to_vec();
    permute::permute_positions(&mut perm_positions, &perm_seed);

    // 5. Check capacity.
    let needed = frame_bits.len() * SHADOW_R;
    if needed > perm_positions.len() {
        return Err(StegoError::MessageTooLarge);
    }

    // 6. Embed each bit R times using repetition coding.
    for (bit_idx, &bit) in frame_bits.iter().enumerate() {
        for r in 0..SHADOW_R {
            let pos_idx = bit_idx * SHADOW_R + r;
            let pos = &perm_positions[pos_idx];
            let (comp_idx, fi) = decode_flat_idx(pos.flat_idx);
            let grid = img.dct_grid(comp_idx);
            let coeff = flat_get(grid, fi);
            let lsb = (coeff.unsigned_abs() & 1) as u8;
            if lsb != bit {
                let modified = nsf5_modify_coefficient(coeff);
                let grid_mut = img.dct_grid_mut(comp_idx);
                flat_set(grid_mut, fi, modified);
            }
        }
    }

    Ok(needed)
}

/// Extract a shadow message from chrominance DCT coefficients.
///
/// # Arguments
/// - `img`: JPEG image to extract from.
/// - `positions`: Combined Cb+Cr positions from [`collect_chroma_positions`].
/// - `passphrase`: Passphrase for the shadow layer to extract.
pub fn shadow_extract(
    img: &JpegImage,
    positions: &[CoeffPos],
    passphrase: &str,
) -> Result<PayloadData, StegoError> {
    // 1. Derive shadow structural key and permute positions.
    let perm_seed = crypto::derive_shadow_structural_key(passphrase)?;
    let mut perm_positions = positions.to_vec();
    permute::permute_positions(&mut perm_positions, &perm_seed);

    // 2. Determine max extractable bits.
    let max_bits = (perm_positions.len() / SHADOW_R).min(MAX_SHADOW_FRAME_BYTES * 8);
    if max_bits == 0 {
        return Err(StegoError::FrameCorrupted);
    }

    // 3. Extract bits via majority vote.
    let mut extracted_bits = Vec::with_capacity(max_bits);
    for bit_idx in 0..max_bits {
        let mut ones = 0u32;
        for r in 0..SHADOW_R {
            let pos_idx = bit_idx * SHADOW_R + r;
            let pos = &perm_positions[pos_idx];
            let (comp_idx, fi) = decode_flat_idx(pos.flat_idx);
            let grid = img.dct_grid(comp_idx);
            let coeff = flat_get(grid, fi);
            let lsb = (coeff.unsigned_abs() & 1) as u8;
            ones += lsb as u32;
        }
        extracted_bits.push(if ones > (SHADOW_R as u32) / 2 { 1u8 } else { 0u8 });
    }

    // 4. Convert bits to bytes, parse frame, decrypt.
    let frame_bytes = frame::bits_to_bytes(&extracted_bits);
    let parsed = frame::parse_frame(&frame_bytes)?;

    let plaintext = crypto::decrypt(
        &parsed.ciphertext,
        passphrase,
        &parsed.salt,
        &parsed.nonce,
    )?;

    let len = parsed.plaintext_len as usize;
    if len > plaintext.len() {
        return Err(StegoError::FrameCorrupted);
    }

    payload::decode_payload(&plaintext[..len])
}

/// Estimate shadow capacity in plaintext bytes from combined chroma positions.
///
/// Each bit requires [`SHADOW_R`] (7) repetitions, so raw bit capacity is
/// `usable_positions / 7`. Frame overhead (salt, nonce, tag, CRC) is then
/// subtracted to yield the usable plaintext space.
pub fn shadow_capacity(usable_positions: usize) -> usize {
    let max_frame_bytes = usable_positions / SHADOW_R / 8;
    max_frame_bytes.saturating_sub(frame::FRAME_OVERHEAD)
}

/// Decode the component index and real flat_idx from an encoded flat_idx.
/// Bit 31 clear = Cb (component 1), bit 31 set = Cr (component 2).
#[inline]
fn decode_flat_idx(encoded: u32) -> (usize, usize) {
    if encoded & CR_FLAG != 0 {
        (2, (encoded & !CR_FLAG) as usize)
    } else {
        (1, encoded as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shadow_capacity_calculation() {
        // 10000 positions / 7 repetitions / 8 bits = 178 frame bytes
        // 178 - 50 (FRAME_OVERHEAD) = 128 plaintext bytes
        assert_eq!(shadow_capacity(10000), 128);
    }

    #[test]
    fn shadow_capacity_too_small() {
        assert_eq!(shadow_capacity(100), 0);
    }

    #[test]
    fn shadow_capacity_zero() {
        assert_eq!(shadow_capacity(0), 0);
    }

    #[test]
    fn decode_flat_idx_cb() {
        let (comp, fi) = decode_flat_idx(1234);
        assert_eq!(comp, 1); // Cb
        assert_eq!(fi, 1234);
    }

    #[test]
    fn decode_flat_idx_cr() {
        let encoded = 1234 | CR_FLAG;
        let (comp, fi) = decode_flat_idx(encoded);
        assert_eq!(comp, 2); // Cr
        assert_eq!(fi, 1234);
    }
}
