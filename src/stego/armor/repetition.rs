//! Repetition coding with soft majority voting.
//!
//! Lays out `r` copies of RS-encoded bits interleaved across embedding
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
/// Returns (output_bits, r_used). The first `r * rs_bits.len()` positions are
/// filled; any remaining positions are zero-padded.
pub fn repetition_encode(rs_bits: &[u8], num_units: usize) -> (Vec<u8>, usize) {
    let r = compute_r(rs_bits.len(), num_units);
    let mut output = vec![0u8; num_units];

    for copy in 0..r {
        let offset = copy * rs_bits.len();
        for (i, &bit) in rs_bits.iter().enumerate() {
            if offset + i < num_units {
                output[offset + i] = bit;
            }
        }
    }

    (output, r)
}

/// Soft majority voting across `r` copies of RS-encoded bits.
///
/// Takes LLR values for all extracted units and the known RS bit count.
/// For each RS bit position, sums LLR across all `r` copies and decides:
/// positive total → bit 0, negative → bit 1.
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

        // Create 5 copies of LLRs, with 2 copies having flipped bit 0
        let mut llrs = Vec::new();
        for copy in 0..r {
            for &bit in &rs_bits {
                let base_llr = if bit == 0 { 3.0 } else { -3.0 };
                if copy < 2 && bit == rs_bits[0] {
                    // Flip the sign for 2 out of 5 copies of bit 0
                    llrs.push(-base_llr);
                } else {
                    llrs.push(base_llr);
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
}
