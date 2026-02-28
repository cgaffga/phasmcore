// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Repetition coding with soft majority voting.
//!
//! Lays out `r` copies of RS-encoded bits sequentially across embedding
//! positions. On extraction, soft majority voting using STDM log-likelihood
//! ratios combines redundant copies for dramatically improved robustness.

/// Compute the repetition factor from RS bit count and total embedding units.
///
/// Forces `r` to be odd (for clean majority voting) unless `r < 2`.
/// Caps at 255 (fits in a u8).
pub fn compute_r(rs_bit_count: usize, num_units: usize) -> usize {
    if rs_bit_count == 0 {
        return 1;
    }
    let r = num_units / rs_bit_count;
    let r = r.min(255);
    if r >= 3 {
        // Force odd, but don't exceed what fits
        let r_odd = r | 1;
        if r_odd * rs_bit_count <= num_units {
            r_odd
        } else {
            // r was even, r|1 = r+1 which overflows — use r-1 (odd)
            (r - 1).max(1)
        }
    } else if r >= 2 {
        // r=2: can't force odd to 3 if it doesn't fit, use as-is
        if 3 * rs_bit_count <= num_units { 3 } else { 1 }
    } else {
        1
    }
}

/// Lay out `r` copies of RS-encoded bits across `num_units` embedding positions.
///
/// Uses sequential layout — copy `j` of the RS bitstream starts at offset
/// `j * rs_bits.len()`. Each copy is a contiguous block.
///
/// Returns (output_bits, r_used). Any remaining positions are zero-padded.
pub fn repetition_encode(rs_bits: &[u8], num_units: usize) -> (Vec<u8>, usize) {
    let r = compute_r(rs_bits.len(), num_units);
    let mut output = vec![0u8; num_units];

    for copy in 0..r {
        let start = copy * rs_bits.len();
        for (i, &bit) in rs_bits.iter().enumerate() {
            if start + i < num_units {
                output[start + i] = bit;
            }
        }
    }

    (output, r)
}

/// Soft majority voting across `r` copies of RS-encoded bits.
///
/// Uses sequential layout — copy `j` of bit `i` is at position
/// `j * rs_bit_count + i`. Sums LLR across all copies and decides:
/// positive total → bit 0, negative → bit 1.
#[cfg(test)]
pub fn repetition_decode_soft(llrs: &[f64], rs_bit_count: usize) -> Vec<u8> {
    if rs_bit_count == 0 {
        return Vec::new();
    }

    let r = llrs.len() / rs_bit_count;
    let mut voted = Vec::with_capacity(rs_bit_count);

    for i in 0..rs_bit_count {
        let mut total_llr = 0.0;
        for copy in 0..r {
            let idx = copy * rs_bit_count + i;
            if idx < llrs.len() {
                total_llr += llrs[idx];
            }
        }
        voted.push(if total_llr >= 0.0 { 0 } else { 1 });
    }

    voted
}

/// Stats from repetition decode for signal quality measurement.
#[derive(Debug, Clone)]
pub struct RepetitionQuality {
    /// Average |LLR| per copy per bit position.
    ///
    /// For each bit position, we sum the LLR across r copies, take the absolute
    /// value, divide by r to get the per-copy average, then average across all
    /// bit positions. This normalizes out the repetition factor so the reference
    /// value is always `delta/2` (STDM) or `step/2` (QIM) regardless of r.
    pub avg_abs_llr_per_copy: f64,
}

/// Soft majority voting with quality statistics.
///
/// Same logic as `repetition_decode_soft`, but also tracks the average |LLR|
/// per copy per bit position for signal quality measurement.
pub fn repetition_decode_soft_with_quality(
    llrs: &[f64],
    rs_bit_count: usize,
) -> (Vec<u8>, RepetitionQuality) {
    if rs_bit_count == 0 {
        return (
            Vec::new(),
            RepetitionQuality {
                avg_abs_llr_per_copy: 0.0,
            },
        );
    }

    let r = llrs.len() / rs_bit_count;
    let mut voted = Vec::with_capacity(rs_bit_count);
    let mut sum_abs_llr_per_copy = 0.0;

    for i in 0..rs_bit_count {
        let mut total_llr = 0.0;
        for copy in 0..r {
            let idx = copy * rs_bit_count + i;
            if idx < llrs.len() {
                total_llr += llrs[idx];
            }
        }
        voted.push(if total_llr >= 0.0 { 0 } else { 1 });
        // |summed_LLR| / r gives per-copy average for this bit position
        if r > 0 {
            sum_abs_llr_per_copy += total_llr.abs() / r as f64;
        }
    }

    let avg_abs_llr_per_copy = if rs_bit_count > 0 {
        sum_abs_llr_per_copy / rs_bit_count as f64
    } else {
        0.0
    };

    (
        voted,
        RepetitionQuality {
            avg_abs_llr_per_copy,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_r_basic() {
        // 100 RS bits, 1000 units → r = 10, forced odd = 11 (11*100=1100>1000, so 9)
        assert_eq!(compute_r(100, 1000), 9);
        // 100 RS bits, 1100 units → r = 11, already odd
        assert_eq!(compute_r(100, 1100), 11);
        // 100 RS bits, 300 units → r = 3, already odd
        assert_eq!(compute_r(100, 300), 3);
        // 100 RS bits, 150 units → r = 1 (1*100 fits, but can't do 3)
        assert_eq!(compute_r(100, 150), 1);
        // 100 RS bits, 100 units → r = 1
        assert_eq!(compute_r(100, 100), 1);
        // Edge: 0 RS bits
        assert_eq!(compute_r(0, 1000), 1);
        // Verify invariant: when bits fit, r * bits always <= num_units
        for bits in [8, 50, 100, 200, 500] {
            for units in [100, 500, 1000, 5000, 50000] {
                if units >= bits {
                    let r = compute_r(bits, units);
                    assert!(r * bits <= units, "r={r}, bits={bits}, units={units}: product {} > {units}", r * bits);
                }
            }
        }
    }

    #[test]
    fn compute_r_caps_at_255() {
        assert_eq!(compute_r(1, 1000), 255);
    }

    #[test]
    fn repetition_encode_decode_no_noise() {
        let rs_bits = vec![0, 1, 1, 0, 1, 0, 0, 1]; // 8 bits
        let num_units = 100;
        let (encoded, r) = repetition_encode(&rs_bits, num_units);
        assert!(r >= 2);
        assert_eq!(encoded.len(), num_units);

        // Simulate perfect LLR extraction (large magnitude, correct sign)
        let used = r * rs_bits.len();
        assert!(used <= num_units, "r*bits should fit within num_units");
        let llrs: Vec<f64> = encoded[..used]
            .iter()
            .map(|&bit| if bit == 0 { 5.0 } else { -5.0 })
            .collect();

        let voted = repetition_decode_soft(&llrs, rs_bits.len());
        assert_eq!(voted, rs_bits);
    }

    #[test]
    fn soft_voting_corrects_noisy_copies() {
        let rs_bits = vec![0, 1, 0, 1];
        let r = 5;
        let rs_bit_count = rs_bits.len();

        // Sequential layout: copy j of bit i at position j * rs_bit_count + i
        // Create LLRs for 5 copies × 4 bits = 20 positions
        let total = rs_bit_count * r;
        let mut llrs = vec![0.0f64; total];
        for i in 0..rs_bit_count {
            let base_llr = if rs_bits[i] == 0 { 3.0 } else { -3.0 };
            for copy in 0..r {
                let idx = copy * rs_bit_count + i;
                if i == 0 && copy < 2 {
                    // Flip the sign for 2 out of 5 copies of bit 0
                    llrs[idx] = -base_llr;
                } else {
                    llrs[idx] = base_llr;
                }
            }
        }

        let voted = repetition_decode_soft(&llrs, rs_bit_count);
        // Majority (3 out of 5) should still get the right answer
        assert_eq!(voted, rs_bits);
    }

    #[test]
    fn r_equals_1_passthrough() {
        let rs_bits = vec![1, 0, 1, 1, 0];
        let num_units = 5; // exactly fits 1 copy
        let (encoded, r) = repetition_encode(&rs_bits, num_units);
        assert_eq!(r, 1);
        assert_eq!(&encoded[..5], &rs_bits);

        let llrs: Vec<f64> = encoded
            .iter()
            .map(|&bit| if bit == 0 { 2.0 } else { -2.0 })
            .collect();
        let voted = repetition_decode_soft(&llrs, rs_bits.len());
        assert_eq!(voted, rs_bits);
    }

    #[test]
    fn with_quality_matches_soft_decode() {
        // Verify that repetition_decode_soft_with_quality returns the same
        // voted bits as repetition_decode_soft.
        let rs_bits = vec![0, 1, 1, 0, 1, 0, 0, 1];
        let num_units = 100;
        let (encoded, r) = repetition_encode(&rs_bits, num_units);
        let used = r * rs_bits.len();
        let llrs: Vec<f64> = encoded[..used]
            .iter()
            .map(|&bit| if bit == 0 { 5.0 } else { -5.0 })
            .collect();

        let voted_plain = repetition_decode_soft(&llrs, rs_bits.len());
        let (voted_quality, quality) = repetition_decode_soft_with_quality(&llrs, rs_bits.len());
        assert_eq!(voted_plain, voted_quality, "Quality variant must produce same bits");
        assert_eq!(voted_quality, rs_bits);
        // For pristine LLRs of magnitude 5.0, the avg_abs_llr_per_copy should be 5.0
        assert!((quality.avg_abs_llr_per_copy - 5.0).abs() < 0.01,
            "Expected avg_abs_llr_per_copy ≈ 5.0, got {}", quality.avg_abs_llr_per_copy);
    }

    #[test]
    fn with_quality_noisy_signal_lower() {
        // Verify that noisy copies produce lower avg_abs_llr_per_copy
        // than pristine copies.
        let rs_bits = vec![0, 1, 0, 1];
        let r = 5;
        let rs_bit_count = rs_bits.len();
        let total = rs_bit_count * r;

        // Pristine: all copies have LLR magnitude 3.0
        let pristine_llrs: Vec<f64> = (0..total)
            .map(|idx| {
                let bit_idx = idx % rs_bit_count;
                if rs_bits[bit_idx] == 0 { 3.0 } else { -3.0 }
            })
            .collect();

        // Noisy: 2 out of 5 copies of bit 0 are flipped
        let mut noisy_llrs = pristine_llrs.clone();
        for copy in 0..2 {
            let idx = copy * rs_bit_count + 0; // bit position 0
            noisy_llrs[idx] = -noisy_llrs[idx]; // flip
        }

        let (_, pristine_q) = repetition_decode_soft_with_quality(&pristine_llrs, rs_bit_count);
        let (voted, noisy_q) = repetition_decode_soft_with_quality(&noisy_llrs, rs_bit_count);

        // Should still decode correctly (3 out of 5 majority)
        assert_eq!(voted, rs_bits);

        // Noisy signal should have lower avg_abs_llr_per_copy
        assert!(noisy_q.avg_abs_llr_per_copy < pristine_q.avg_abs_llr_per_copy,
            "Noisy signal ({}) should be weaker than pristine ({})",
            noisy_q.avg_abs_llr_per_copy, pristine_q.avg_abs_llr_per_copy);
    }

    #[test]
    fn with_quality_empty_input() {
        let (voted, quality) = repetition_decode_soft_with_quality(&[], 0);
        assert!(voted.is_empty());
        assert_eq!(quality.avg_abs_llr_per_copy, 0.0);
    }
}
