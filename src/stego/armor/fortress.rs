//! Fortress sub-mode: BA-QIM embedding in DC block averages.
//!
//! Fortress uses Quantization Index Modulation (QIM) on the average brightness
//! of each 8x8 block (derived from the DC coefficient). Block averages shift
//! by less than 2 pixel levels even through aggressive JPEG recompression
//! (QF 53+), making this near-perfectly recompression-invariant.
//!
//! Fortress is used automatically within Armor mode when the message is short
//! enough to fit. Longer messages fall through to the STDM pipeline.
//!
//! # Embedding domain
//!
//! Each 8x8 block's DC coefficient relates to the block average brightness:
//!   avg = dc * qt_dc / 8.0
//! QIM embeds one bit per block by quantizing `avg` to a step grid offset by
//! the target bit value.
//!
//! # Header
//!
//! A magic byte (0xF5) is embedded in the first 56 permuted blocks using
//! 7-copy majority voting (8 bits x 7 copies = 56 blocks). This allows fast
//! detection on decode: if the magic doesn't match, fall through to STDM.

use crate::jpeg::JpegImage;
use crate::stego::armor::ecc;
use crate::stego::armor::repetition;
use crate::stego::crypto;
use crate::stego::error::StegoError;
use crate::stego::frame;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

/// QIM step size in pixel-level units.
/// Block averages shift <2 levels through QF70+ recompression, so step=8 gives
/// a 2-level margin on each side of the decision boundary — sufficient for
/// pre-settled images (QF70 encode → QF75 WhatsApp recompression).
const QIM_STEP: f64 = 8.0;

/// Minimum margin from the QIM decision boundary for nudge embedding.
/// WhatsApp shifts block averages by max ~1.375 pixel levels for QF70 pre-settled
/// images, so a 2.0 margin provides comfortable headroom.
const QIM_MIN_MARGIN: f64 = 2.0;

/// Minimum repetition factor for Fortress encode.
/// Higher minimum = earlier switchover to STDM for longer messages = better quality.
/// At r=5, a 1200×1600 image holds ~444 chars; a 1280×720 image holds ~197 chars.
const FORTRESS_MIN_R: usize = 5;

/// Maximum repetition factor for Fortress encode.
/// Caps how many copies of the RS bitstream are embedded. With uncapped r, ALL
/// payload blocks are modified (zero-padded repetition fills everything). Capping
/// at 11 leaves most blocks untouched, dramatically improving image quality.
const FORTRESS_MAX_R: usize = 11;

/// Magic byte embedded in the fortress header for fast detection.
const FORTRESS_MAGIC: u8 = 0xF5;

/// Number of blocks used for the magic header: 1 byte x 8 bits x 7 copies.
const FORTRESS_HEADER_BLOCKS: usize = 56;

/// Number of majority-vote copies of the magic byte.
const FORTRESS_HEADER_COPIES: usize = 7;

// --- BA-QIM core functions ---

/// Convert a quantized DC coefficient to block-average brightness.
fn dc_to_avg(dc: i16, qt_dc: u16) -> f64 {
    dc as f64 * qt_dc as f64 / 8.0
}

/// Convert a block-average brightness back to a quantized DC coefficient.
fn avg_to_dc(avg: f64, qt_dc: u16) -> i16 {
    (avg * 8.0 / qt_dc as f64).round() as i16
}

/// QIM embed: quantize `avg` to the nearest grid point for the target bit.
///
/// Bit 0 grid: ..., -step, 0, step, 2*step, ...
/// Bit 1 grid: ..., -step/2, step/2, 3*step/2, ...
fn qim_embed_avg(avg: f64, step: f64, bit: u8) -> f64 {
    let offset = if bit == 0 { 0.0 } else { step / 2.0 };
    let shifted = avg - offset;
    let quantized = (shifted / step).round() * step;
    quantized + offset
}

/// QIM soft extraction: compute log-likelihood ratio.
///
/// Returns positive for bit 0, negative for bit 1.
/// Magnitude indicates confidence.
fn qim_extract_soft(avg: f64, step: f64) -> f64 {
    let half = step / 2.0;
    // Distance to nearest bit-0 grid point
    let d0 = (avg / step).round() * step;
    let dist0 = (avg - d0).abs();
    // Distance to nearest bit-1 grid point
    let d1 = ((avg - half) / step).round() * step + half;
    let dist1 = (avg - d1).abs();
    // LLR: positive means closer to bit-0, negative means closer to bit-1
    dist1 - dist0
}

/// Nudge-toward-center QIM: minimize distortion while ensuring adequate margin.
///
/// If the block already encodes the correct bit with at least `min_margin`
/// distance from the decision boundary, leave it untouched. Otherwise, nudge
/// it just enough to achieve the margin. For blocks encoding the wrong bit,
/// snap to the nearest grid center (standard QIM) for maximum robustness.
///
/// This dramatically reduces average distortion (~+9 dB PSNR improvement)
/// because most natural image blocks already encode one bit or the other
/// with sufficient margin and need no modification at all.
fn qim_embed_avg_nudge(avg: f64, step: f64, bit: u8, min_margin: f64) -> f64 {
    let half = step / 2.0;

    // Determine which bit the block currently encodes and distance from boundary.
    let llr = qim_extract_soft(avg, step);
    let current_bit = if llr >= 0.0 { 0u8 } else { 1u8 };
    let boundary_dist = llr.abs(); // distance from decision boundary

    if current_bit == bit && boundary_dist >= min_margin {
        // Already correct with sufficient margin — no change needed.
        avg
    } else if current_bit == bit {
        // Correct side but insufficient margin — nudge toward grid center.
        let offset = if bit == 0 { 0.0 } else { half };
        let center = ((avg - offset) / step).round() * step + offset;
        let dist_from_center = avg - center;
        let nudge = min_margin - boundary_dist;
        if dist_from_center >= 0.0 {
            avg - nudge
        } else {
            avg + nudge
        }
    } else {
        // Wrong side — snap to nearest grid center for the target bit (standard QIM).
        qim_embed_avg(avg, step, bit)
    }
}

// --- Block permutation ---

/// Generate a permuted list of block indices using Fisher-Yates shuffle.
///
/// Uses `u32` for cross-platform portability (WASM is 32-bit).
fn permute_blocks(total_blocks: usize, seed: &[u8; 32]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..total_blocks).collect();
    let mut rng = ChaCha20Rng::from_seed(*seed);
    let n = indices.len();
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=(i as u32)) as usize;
        indices.swap(i, j);
    }
    indices
}

// --- Fortress capacity ---

/// Compute the capped, odd repetition factor for fortress mode.
///
/// Applies `FORTRESS_MAX_R` cap and forces odd (for clean majority voting).
fn fortress_compute_r(rs_bit_count: usize, payload_blocks: usize) -> usize {
    let r = repetition::compute_r(rs_bit_count, payload_blocks);
    let r = r.min(FORTRESS_MAX_R);
    // Force odd after capping (compute_r already forces odd, but capping may
    // have turned an odd number even if MAX_R is even — currently 11, so fine).
    if r >= 3 && r % 2 == 0 {
        r - 1
    } else {
        r
    }
}

/// Maximum frame bytes that can be embedded in a fortress-encoded image.
///
/// Returns the max frame_bytes (before RS encoding) that fits in the available
/// blocks, accounting for header, RS parity, repetition coding (r >= MIN_R),
/// and the MAX_R cap for sparse embedding.
pub fn fortress_max_frame_bytes(img: &JpegImage) -> Result<usize, StegoError> {
    let grid = img.dct_grid(0);
    let total_blocks = grid.blocks_wide() * grid.blocks_tall();

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Ok(0);
    }
    let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;

    // With sparse embedding (MAX_R cap), total embedded blocks = r * rs_bits.
    // We need: r * rs_bits <= payload_blocks AND r >= FORTRESS_MIN_R.
    // r is capped at FORTRESS_MAX_R, so with a large image and small message,
    // only MAX_R * rs_bits blocks are used (the rest are untouched).
    let mut best_frame_bytes = 0usize;

    for &parity in &ecc::PARITY_TIERS {
        // Binary search for max frame_bytes
        let mut lo = 0usize;
        let mut hi = frame::MAX_FRAME_BYTES;
        while lo < hi {
            let mid = (lo + hi + 1) / 2;
            let rs_encoded_len = ecc::rs_encoded_len_with_parity(mid, parity);
            let rs_bits = rs_encoded_len * 8;
            if rs_bits == 0 || rs_bits > payload_blocks {
                hi = mid - 1;
                continue;
            }
            let r = fortress_compute_r(rs_bits, payload_blocks);
            if r >= FORTRESS_MIN_R && r * rs_bits <= payload_blocks {
                lo = mid;
            } else {
                hi = mid - 1;
            }
        }
        if lo > best_frame_bytes {
            // Verify the solution
            let rs_encoded_len = ecc::rs_encoded_len_with_parity(lo, parity);
            let rs_bits = rs_encoded_len * 8;
            if rs_bits <= payload_blocks {
                let r = fortress_compute_r(rs_bits, payload_blocks);
                if r >= FORTRESS_MIN_R && r * rs_bits <= payload_blocks {
                    best_frame_bytes = lo;
                }
            }
        }
    }

    Ok(best_frame_bytes)
}

/// Estimate fortress plaintext capacity (max message bytes).
pub fn fortress_capacity(img: &JpegImage) -> Result<usize, StegoError> {
    let max_frame = fortress_max_frame_bytes(img)?;
    if max_frame <= frame::FRAME_OVERHEAD {
        return Ok(0);
    }
    Ok(max_frame - frame::FRAME_OVERHEAD)
}

// --- Fortress encode ---

/// Encode a payload frame into an image using BA-QIM on DC block averages.
///
/// The caller must have already built the frame_bytes (encrypted + framed).
/// This function embeds the magic header and the RS+repetition encoded payload
/// into permuted Y-channel block DCs.
pub fn fortress_encode(
    img: &mut JpegImage,
    frame_bytes: &[u8],
    passphrase: &str,
) -> Result<(), StegoError> {
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_dc = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?
        .values[0];

    let grid = img.dct_grid(0);
    let blocks_wide = grid.blocks_wide();
    let blocks_tall = grid.blocks_tall();
    let total_blocks = blocks_wide * blocks_tall;

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Err(StegoError::ImageTooSmall);
    }
    let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;

    // Derive fortress structural key for block permutation.
    let fort_key = crypto::derive_fortress_structural_key(passphrase);
    let perm = permute_blocks(total_blocks, &fort_key);

    // RS-encode with best parity tier (smallest parity where r >= FORTRESS_MIN_R).
    let mut chosen_parity = ecc::PARITY_TIERS[0];
    for &parity in &ecc::PARITY_TIERS {
        let rs_encoded = ecc::rs_encode_blocks_with_parity(frame_bytes, parity);
        let rs_bits_len = rs_encoded.len() * 8;
        if rs_bits_len <= payload_blocks {
            let r = fortress_compute_r(rs_bits_len, payload_blocks);
            if r >= FORTRESS_MIN_R && r * rs_bits_len <= payload_blocks {
                chosen_parity = parity;
                break;
            }
        }
    }

    let rs_encoded = ecc::rs_encode_blocks_with_parity(frame_bytes, chosen_parity);
    let rs_bits = frame::bytes_to_bits(&rs_encoded);

    let r = fortress_compute_r(rs_bits.len(), payload_blocks);
    if r < FORTRESS_MIN_R {
        return Err(StegoError::MessageTooLarge);
    }

    // Sparse embedding: use exact RS bit count, repeat r times.
    // Only r * rs_bits.len() payload blocks are embedded; the rest are untouched.
    let rs_bit_count = rs_bits.len();
    let total_embed = r * rs_bit_count;
    let (rep_bits, _) = repetition::repetition_encode(&rs_bits, total_embed);

    // Build complete bit sequence: header (56 blocks) + payload (sparse)
    let mut all_bits = Vec::with_capacity(FORTRESS_HEADER_BLOCKS + total_embed);

    // Header: 7 copies of magic byte
    for _ in 0..FORTRESS_HEADER_COPIES {
        for bp in (0..8).rev() {
            all_bits.push((FORTRESS_MAGIC >> bp) & 1);
        }
    }

    // Payload bits (only the embedded portion, not all payload_blocks)
    all_bits.extend_from_slice(&rep_bits[..total_embed.min(rep_bits.len())]);

    // Embed using nudge-toward-center BA-QIM on DC coefficients.
    // Header blocks use standard QIM (must be exact for magic detection).
    // Payload blocks use nudge QIM (minimize distortion).
    let grid_mut = img.dct_grid_mut(0);
    for (bit_idx, &block_idx) in perm.iter().enumerate() {
        if bit_idx >= all_bits.len() {
            break;
        }
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid_mut.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        let new_avg = if bit_idx < FORTRESS_HEADER_BLOCKS {
            // Header: standard QIM for reliable magic detection
            qim_embed_avg(avg, QIM_STEP, all_bits[bit_idx])
        } else {
            // Payload: nudge QIM for minimal distortion
            qim_embed_avg_nudge(avg, QIM_STEP, all_bits[bit_idx], QIM_MIN_MARGIN)
        };
        let new_dc = avg_to_dc(new_avg, qt_dc);
        grid_mut.set(br, bc, 0, 0, new_dc);
    }

    Ok(())
}

// --- Fortress decode ---

/// Decode a fortress-encoded payload from a stego JPEG.
///
/// Returns the decoded message and quality info, or an error if the fortress
/// magic is not detected (allowing the caller to fall through to STDM).
pub fn fortress_decode(
    img: &JpegImage,
    passphrase: &str,
) -> Result<(String, super::pipeline::DecodeQuality), StegoError> {
    let step = QIM_STEP;
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt_dc = img
        .quant_table(qt_id)
        .ok_or(StegoError::NoLuminanceChannel)?
        .values[0];

    let grid = img.dct_grid(0);
    let blocks_wide = grid.blocks_wide();
    let blocks_tall = grid.blocks_tall();
    let total_blocks = blocks_wide * blocks_tall;

    if total_blocks <= FORTRESS_HEADER_BLOCKS {
        return Err(StegoError::FrameCorrupted);
    }
    let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;

    // Derive key and permute blocks.
    let fort_key = crypto::derive_fortress_structural_key(passphrase);
    let perm = permute_blocks(total_blocks, &fort_key);

    // Extract all LLRs from DC coefficients.
    let mut all_llrs = Vec::with_capacity(total_blocks);
    for &block_idx in &perm {
        let br = block_idx / blocks_wide;
        let bc = block_idx % blocks_wide;
        let dc = grid.get(br, bc, 0, 0);
        let avg = dc_to_avg(dc, qt_dc);
        all_llrs.push(qim_extract_soft(avg, step));
    }

    // Check magic header: majority vote across 7 copies
    let magic = extract_magic_byte(&all_llrs[..FORTRESS_HEADER_BLOCKS]);
    if magic != FORTRESS_MAGIC {
        return Err(StegoError::FrameCorrupted);
    }

    // Extract payload LLRs (after header)
    let payload_llrs = &all_llrs[FORTRESS_HEADER_BLOCKS..];
    let payload_llrs = &payload_llrs[..payload_blocks.min(payload_llrs.len())];

    // Brute-force search over (frame_len, parity) combinations.
    // For each candidate, compute the exact r using the same formula as the
    // encoder (fortress_compute_r), then vote and RS-decode.
    // This handles both sparse embedding (MAX_R capped) and legacy images.
    for &parity in &ecc::PARITY_TIERS {
        let candidates = compute_fortress_decode_candidates(payload_blocks, parity);

        for (rs_bit_count, r) in candidates {
            if rs_bit_count == 0 || r < 3 {
                continue;
            }
            let used_llrs = rs_bit_count * r;
            if used_llrs > payload_llrs.len() {
                continue;
            }

            let voted_bits =
                repetition::repetition_decode_soft(&payload_llrs[..used_llrs], rs_bit_count);
            let voted_bytes = frame::bits_to_bytes(&voted_bits);

            if let Some((decoded_frame, rs_stats)) =
                try_fortress_rs_decode(&voted_bytes, parity)
            {
                if let Ok(parsed) = frame::parse_frame(&decoded_frame) {
                    if let Ok(plaintext) = crypto::decrypt(
                        &parsed.ciphertext,
                        passphrase,
                        &parsed.salt,
                        &parsed.nonce,
                    ) {
                        let len = parsed.plaintext_len as usize;
                        if len <= plaintext.len() {
                            if let Ok(text) = std::str::from_utf8(&plaintext[..len]) {
                                let quality = super::pipeline::DecodeQuality::from_rs_stats_with_phase2(
                                    &rs_stats,
                                    r as u8,
                                    parity as u16,
                                );
                                let mut q = quality;
                                q.fortress_used = true;
                                return Ok((text.to_string(), q));
                            }
                        }
                    }
                }
            }
        }
    }

    Err(StegoError::FrameCorrupted)
}

/// Extract magic byte from header LLRs using soft majority voting.
fn extract_magic_byte(header_llrs: &[f64]) -> u8 {
    let mut byte = 0u8;
    for bit_pos in 0..8 {
        let mut total = 0.0;
        for copy in 0..FORTRESS_HEADER_COPIES {
            let idx = copy * 8 + bit_pos;
            if idx < header_llrs.len() {
                total += header_llrs[idx];
            }
        }
        // Positive LLR = bit 0, negative = bit 1
        if total < 0.0 {
            byte |= 1 << (7 - bit_pos);
        }
    }
    byte
}

/// Compute distinct (rs_bit_count, r) decode candidates for fortress.
///
/// Iterates all possible frame lengths and computes the matching r using the
/// same formula as the encoder (with MAX_R cap). Returns deduplicated
/// (rs_bit_count, r) pairs — each unique pair represents a distinct repetition
/// layout to try during decode.
fn compute_fortress_decode_candidates(
    payload_blocks: usize,
    parity: usize,
) -> Vec<(usize, usize)> {
    let mut seen = std::collections::BTreeSet::new();
    let mut candidates = Vec::new();

    let min_frame = frame::FRAME_OVERHEAD;
    let max_frame = frame::MAX_FRAME_BYTES;

    for frame_len in min_frame..=max_frame {
        let rs_encoded_len = ecc::rs_encoded_len_with_parity(frame_len, parity);
        let rs_bits = rs_encoded_len * 8;
        if rs_bits == 0 || rs_bits > payload_blocks {
            continue;
        }
        let r = fortress_compute_r(rs_bits, payload_blocks);
        if r >= 3 && r * rs_bits <= payload_blocks {
            let key = (rs_bits, r);
            if seen.insert(key) {
                candidates.push(key);
            }
        }
    }

    candidates
}

/// Try RS decode with a specific parity — mirrors pipeline's try_rs_decode_frame_with_parity.
fn try_fortress_rs_decode(
    extracted_bytes: &[u8],
    parity: usize,
) -> Option<(Vec<u8>, ecc::RsDecodeStats)> {
    let k_max = 255 - parity;
    let min_data = 2usize.min(k_max);

    for data_len in min_data..=k_max.min(extracted_bytes.len().saturating_sub(parity)) {
        let block_len = data_len + parity;
        if block_len > extracted_bytes.len() {
            break;
        }

        if let Ok((first_block_data, first_errors)) =
            ecc::rs_decode_with_parity(&extracted_bytes[..block_len], data_len, parity)
        {
            if first_block_data.len() >= 2 {
                let pt_len =
                    u16::from_be_bytes([first_block_data[0], first_block_data[1]]) as usize;
                let ct_len = pt_len + 16;
                let total_frame_len = 2 + 16 + 12 + ct_len + 4;

                if total_frame_len > frame::MAX_FRAME_BYTES + 1024 {
                    continue;
                }

                if total_frame_len == data_len {
                    let t_max = parity / 2;
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: t_max,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Some((first_block_data, stats));
                }

                if total_frame_len < data_len {
                    let t_max = parity / 2;
                    let stats = ecc::RsDecodeStats {
                        total_errors: first_errors,
                        error_capacity: t_max,
                        max_block_errors: first_errors,
                        num_blocks: 1,
                    };
                    return Some((first_block_data[..total_frame_len].to_vec(), stats));
                }

                if total_frame_len > data_len {
                    let rs_encoded_len =
                        ecc::rs_encoded_len_with_parity(total_frame_len, parity);
                    if rs_encoded_len <= extracted_bytes.len() {
                        if let Ok((decoded, stats)) = ecc::rs_decode_blocks_with_parity(
                            &extracted_bytes[..rs_encoded_len],
                            total_frame_len,
                            parity,
                        ) {
                            return Some((decoded, stats));
                        }
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

    #[test]
    fn qim_embed_extract_roundtrip() {
        for avg in [0.0, 50.0, 100.0, 200.0, -50.0, 127.5] {
            for bit in [0u8, 1] {
                let embedded = qim_embed_avg(avg, QIM_STEP, bit);
                let llr = qim_extract_soft(embedded, QIM_STEP);
                let extracted_bit = if llr >= 0.0 { 0u8 } else { 1u8 };
                assert_eq!(
                    extracted_bit, bit,
                    "Failed for avg={avg}, bit={bit}: embedded={embedded}, llr={llr}"
                );
            }
        }
    }

    #[test]
    fn qim_nudge_leaves_correct_blocks_untouched() {
        // Block at avg=4.0 (step=8): nearest bit-0 center is 0.0, bit-1 center is 4.0.
        // LLR = dist1 - dist0 = |4-4| - |4-0| = 0 - 4 = -4 → bit 1 with margin 4.
        // If target bit is 1, margin 4 >= 2 → should be untouched.
        let avg = 4.0;
        let result = qim_embed_avg_nudge(avg, QIM_STEP, 1, QIM_MIN_MARGIN);
        assert_eq!(result, avg, "Block already encodes correct bit with margin — should be untouched");

        // Block at avg=0.5 (step=8): nearest bit-0 center is 0.0, bit-1 center is 4.0.
        // LLR = |0.5-4| - |0.5-0| = 3.5 - 0.5 = 3.0 → bit 0 with margin 3.
        // If target bit is 0, margin 3 >= 2 → untouched.
        let avg2 = 0.5;
        let result2 = qim_embed_avg_nudge(avg2, QIM_STEP, 0, QIM_MIN_MARGIN);
        assert_eq!(result2, avg2, "Block already correct with sufficient margin");
    }

    #[test]
    fn qim_nudge_nudges_insufficient_margin() {
        // Block at avg=3.5 (step=8): nearest bit-0 center is 0.0, bit-1 center is 4.0.
        // LLR = |3.5-4| - |3.5-0| = 0.5 - 3.5 = -3.0 → bit 1 with margin 3 (passes)
        // BUT: let's pick a point with margin < 2.
        // avg=3.0: LLR = |3-4| - |3-0| = 1 - 3 = -2 → bit 1 with margin 2 (passes at exactly 2)
        // avg=2.9: LLR = |2.9-4| - |2.9-0| = 1.1 - 2.9 = -1.8 → bit 1 with margin 1.8 < 2
        let avg = 2.9;
        let result = qim_embed_avg_nudge(avg, QIM_STEP, 1, QIM_MIN_MARGIN);
        // Should nudge toward center (4.0) just enough to get margin=2
        // Need to move by 2.0 - 1.8 = 0.2 toward center
        assert!(result != avg, "Should nudge when margin insufficient");
        // Verify the result encodes bit 1
        let llr = qim_extract_soft(result, QIM_STEP);
        assert!(llr < 0.0, "Nudged value should still encode bit 1");
        // Verify margin is now >= 2.0
        assert!(llr.abs() >= QIM_MIN_MARGIN - 0.001, "Margin should be at least min_margin");
        // Verify distortion is small (much less than standard QIM)
        assert!((result - avg).abs() < 1.0, "Nudge should be small");
    }

    #[test]
    fn qim_nudge_snaps_wrong_bit() {
        // Block at avg=0.5 encodes bit 0 (LLR positive).
        // If target is bit 1, should snap to grid center (standard QIM).
        let avg = 0.5;
        let result = qim_embed_avg_nudge(avg, QIM_STEP, 1, QIM_MIN_MARGIN);
        let standard = qim_embed_avg(avg, QIM_STEP, 1);
        assert_eq!(result, standard, "Wrong bit should snap to center (standard QIM)");
    }

    #[test]
    fn qim_nudge_roundtrip_all_bits() {
        // Verify all nudge-embedded values extract correctly
        for avg_int in (-200..=200).step_by(1) {
            let avg = avg_int as f64 * 0.1;
            for bit in [0u8, 1] {
                let embedded = qim_embed_avg_nudge(avg, QIM_STEP, bit, QIM_MIN_MARGIN);
                let llr = qim_extract_soft(embedded, QIM_STEP);
                let extracted_bit = if llr >= 0.0 { 0u8 } else { 1u8 };
                assert_eq!(
                    extracted_bit, bit,
                    "Nudge roundtrip failed for avg={avg}, bit={bit}: embedded={embedded}, llr={llr}"
                );
            }
        }
    }

    #[test]
    fn qim_soft_llr_sign_matches_hard() {
        for avg in [10.0, 50.0, 128.0, 250.0] {
            for bit in [0u8, 1] {
                let embedded = qim_embed_avg(avg, QIM_STEP, bit);
                let llr = qim_extract_soft(embedded, QIM_STEP);
                // Positive LLR = bit 0, negative = bit 1
                if bit == 0 {
                    assert!(llr >= 0.0, "Expected positive LLR for bit 0, got {llr}");
                } else {
                    assert!(llr <= 0.0, "Expected negative LLR for bit 1, got {llr}");
                }
            }
        }
    }

    #[test]
    fn dc_avg_roundtrip() {
        for qt_dc in [1u16, 2, 4, 8, 16, 32] {
            for dc in [-100i16, -1, 0, 1, 50, 100] {
                let avg = dc_to_avg(dc, qt_dc);
                let recovered = avg_to_dc(avg, qt_dc);
                assert_eq!(
                    recovered, dc,
                    "Roundtrip failed for dc={dc}, qt_dc={qt_dc}: avg={avg}"
                );
            }
        }
    }

    #[test]
    fn magic_byte_majority_voting() {
        // Embed magic byte in 56 LLRs, perturb some, verify majority vote
        let mut llrs = Vec::with_capacity(56);
        for _ in 0..FORTRESS_HEADER_COPIES {
            for bp in (0..8).rev() {
                let bit = (FORTRESS_MAGIC >> bp) & 1;
                // Perfect LLR: positive for 0, negative for 1
                llrs.push(if bit == 0 { 5.0 } else { -5.0 });
            }
        }

        // Perturb 2 out of 7 copies (flip their signs)
        for i in 0..8 {
            llrs[i] = -llrs[i]; // flip copy 0
            llrs[8 + i] = -llrs[8 + i]; // flip copy 1
        }

        let extracted = extract_magic_byte(&llrs);
        assert_eq!(extracted, FORTRESS_MAGIC, "Majority vote should recover magic byte");
    }

    #[test]
    fn block_permutation_deterministic() {
        let seed = [42u8; 32];
        let a = permute_blocks(100, &seed);
        let b = permute_blocks(100, &seed);
        assert_eq!(a, b);
    }

    #[test]
    fn block_permutation_different_seeds() {
        let a = permute_blocks(100, &[1u8; 32]);
        let b = permute_blocks(100, &[2u8; 32]);
        assert_ne!(a, b);
    }

    #[test]
    fn block_permutation_is_permutation() {
        let perm = permute_blocks(100, &[7u8; 32]);
        let mut sorted = perm.clone();
        sorted.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(sorted, expected);
    }

    #[test]
    fn fortress_compute_r_caps_at_max() {
        // With 100 RS bits and 50000 payload blocks, uncapped r would be 499 (odd).
        // fortress_compute_r should cap at FORTRESS_MAX_R (11).
        let r = fortress_compute_r(100, 50000);
        assert_eq!(r, FORTRESS_MAX_R);
        // Verify it's odd
        assert!(r % 2 == 1, "Capped r should be odd");
    }

    #[test]
    fn fortress_compute_r_no_cap_for_small_r() {
        // When natural r is below MAX_R, cap doesn't change it.
        let r = fortress_compute_r(100, 700); // natural r=7
        assert_eq!(r, 7);
    }

    #[test]
    fn fortress_capacity_reasonable() {
        // A 1280x720 image has 160*90 = 14400 blocks
        // Payload blocks = 14400 - 56 = 14344
        // With MAX_R=11 cap, capacity is limited by MAX_R * rs_bits <= payload_blocks
        // So rs_bits <= 14344 / 11 = 1304 bits = 163 bytes
        // With parity=64, that's ~99 frame bytes → ~49 chars
        let total_blocks = 14400usize;
        let payload_blocks = total_blocks - FORTRESS_HEADER_BLOCKS;
        assert!(payload_blocks > 1000, "Should have many payload blocks");
        // Verify sparse embedding with a small message leaves most blocks untouched.
        // A short message (~20 chars) → ~70 frame bytes → RS with parity=64 → 134 bytes
        // → 1072 RS bits. With r=11, embedded = 11 * 1072 = 11792 out of 14344 (82%).
        // But for a very small message: ~5 chars → 55 frame bytes → RS+parity → ~119 bytes
        // → 952 RS bits. With r=11, embedded = 10472 out of 14344 (73%).
        // The real quality win is from nudge QIM (blocks already correct are untouched)
        // AND from the fact that r is capped vs. the old uncapped approach where all
        // payload_blocks were modified (including zero-padded positions).
        let rs_bits_small = 500; // very short message RS bits
        let r = fortress_compute_r(rs_bits_small, payload_blocks);
        let embedded = r * rs_bits_small;
        assert!(r == FORTRESS_MAX_R, "Small message should hit MAX_R cap");
        assert!(
            embedded < payload_blocks,
            "Sparse embedding should use fewer than all blocks: embedded={embedded}, total={payload_blocks}"
        );
    }

    #[test]
    fn decode_candidates_include_capped_r() {
        let payload_blocks = 14344;
        for &parity in &ecc::PARITY_TIERS {
            let candidates = compute_fortress_decode_candidates(payload_blocks, parity);
            // Should have candidates with r <= MAX_R
            for &(rs_bits, r) in &candidates {
                assert!(r <= FORTRESS_MAX_R, "r={r} should be <= MAX_R={FORTRESS_MAX_R}");
                assert!(r * rs_bits <= payload_blocks, "r*rs_bits should fit");
            }
        }
    }
}
