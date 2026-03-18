// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Encode quality scoring: Stealth (Ghost) and Robustness (Armor).
//!
//! After encoding, these functions compute a 0-100% quality score that
//! indicates how well-hidden (Ghost) or how robust (Armor) the encoded
//! message is. The score is displayed in the mobile app HUD and encode
//! success screen.

/// Encode quality result returned alongside stego bytes.
#[derive(Debug, Clone)]
pub struct EncodeQuality {
    /// Overall score 0-100.
    pub score: u8,
    /// Localization key for a contextual hint (e.g. "hint_high_rate").
    pub hint_key: String,
    /// Mode: 1 = Ghost (stealth), 2 = Armor (robustness).
    pub mode: u8,
}

/// Maximum embedding rate for Ghost stealth scoring.
/// alpha = modifications / positions. At alpha_max, rate_score = 0.
const ALPHA_MAX: f64 = 0.5;

/// Inputs for Ghost stealth score computation.
pub struct GhostMetrics {
    /// Number of STC modifications (cover != stego).
    pub num_modifications: usize,
    /// Number of cover positions used by STC (n_used).
    pub n_used: usize,
    /// STC width parameter.
    pub w: usize,
    /// Total distortion cost from STC embedding.
    pub total_cost: f64,
    /// Median cost across all positions (before STC).
    pub median_cost: f32,
    /// Whether SI-UNIWARD (Deep Cover) was used.
    pub is_si: bool,
    /// Number of shadow LSB modifications (0 if no shadows).
    pub shadow_modifications: usize,
    /// Total number of Y-channel coefficients in the image.
    pub total_coefficients: usize,
}

/// Compute Ghost stealth score (0-100).
///
/// Five weighted factors + shadow penalty:
/// - 30% embedding rate
/// - 25% cost distribution quality (median cost proxy)
/// - 20% STC width
/// - 15% avg distortion per change
/// - 10% SI-UNIWARD bonus
/// - Shadow penalty: up to -15 points for LSB modifications
pub fn ghost_stealth_score(m: &GhostMetrics) -> EncodeQuality {
    let alpha = if m.n_used > 0 {
        m.num_modifications as f64 / m.n_used as f64
    } else {
        1.0
    };

    // Factor 1: Embedding rate (30%)
    // Lower alpha = stealthier. Quadratic falloff.
    let rate_ratio = (alpha / ALPHA_MAX).min(1.0);
    let rate_score = 100.0 * (1.0 - rate_ratio) * (1.0 - rate_ratio);

    // Factor 2: Cost distribution quality (25%)
    // Higher median cost = image has more texture = better hiding.
    // Median cost of 20+ is excellent; below 3 is poor.
    let cost_score = 100.0 * (1.0 - ((m.median_cost as f64 - 3.0).max(0.0) / 17.0).min(1.0));
    // Invert: high median = hiding in texture = good
    let cost_score = 100.0 - cost_score;

    // Factor 3: STC width (20%)
    // Higher w = more coding slack = fewer modifications.
    // tanh(w/5) gives ~100 for w>=5, ~38 for w=1.
    let w_f = m.w as f64;
    let width_score = 100.0 * det_tanh(w_f / 5.0);

    // Factor 4: Avg distortion per change (15%)
    // Lower avg cost per modification = changes blend into noise.
    let avg_cost = if m.num_modifications > 0 {
        m.total_cost / m.num_modifications as f64
    } else {
        0.0
    };
    let distort_score = 100.0 * (1.0 - ((avg_cost - 3.0) / 17.0).clamp(0.0, 1.0));

    // Factor 5: SI-UNIWARD bonus (10%)
    let si_score = if m.is_si { 100.0 } else { 0.0 };

    let base_stealth = 0.30 * rate_score
        + 0.25 * cost_score
        + 0.20 * width_score
        + 0.15 * distort_score
        + 0.10 * si_score;

    // Shadow penalty: up to -15 points for LSB modifications.
    let shadow_penalty = if m.shadow_modifications > 0 && m.total_coefficients > 0 {
        let shadow_mod_ratio = m.shadow_modifications as f64 / m.total_coefficients as f64;
        15.0 * (shadow_mod_ratio / 0.01).min(1.0)
    } else {
        0.0
    };

    let stealth = (base_stealth - shadow_penalty).max(0.0).min(100.0);
    let score = stealth.round() as u8;

    // Pick hint based on weakest factor.
    let factors = [
        (rate_score, "hint_high_rate"),
        (cost_score, "hint_low_texture"),
        (width_score, "hint_low_width"),
        (distort_score, "hint_high_distortion"),
    ];
    let hint_key = if shadow_penalty > 5.0 {
        "hint_shadow_penalty"
    } else if m.is_si && score >= 70 {
        "hint_si_bonus"
    } else {
        // Pick the weakest factor.
        factors.iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, key)| *key)
            .unwrap_or("hint_high_rate")
    };

    EncodeQuality {
        score,
        hint_key: hint_key.to_string(),
        mode: 1,
    }
}

/// Inputs for Armor robustness score computation.
pub struct ArmorMetrics {
    /// Repetition factor (1 = Phase 1, 3+ = Phase 2, 15+ = Fortress).
    pub repetition_factor: usize,
    /// RS parity symbols per block (64/128/192/240).
    pub parity_symbols: usize,
    /// Whether Fortress sub-mode (BA-QIM) was used.
    pub fortress: bool,
    /// Mean quantization table value (AC positions zigzag 1..=15).
    /// Higher = lower QF = more robust STDM embedding.
    pub mean_qt: f64,
    /// Payload fill ratio (0.0-1.0). Lower = more room for redundancy.
    pub fill_ratio: f64,
    /// STDM delta strength.
    pub delta: f64,
}

/// Compute Armor robustness score (0-100).
///
/// When Fortress is active, six weighted factors:
/// - 30% repetition, 20% parity, 15% Fortress bonus, 15% QT resilience,
///   10% fill, 10% delta
///
/// When Fortress is NOT active, the 15% Fortress weight is redistributed
/// proportionally to the other five factors (~35/24/18/12/12%).
/// This prevents a hard cap on standard Armor scores.
pub fn armor_robustness_score(m: &ArmorMetrics) -> EncodeQuality {
    // Factor 1: Repetition factor
    // r >= 7 is strong. Linear scale.
    let rep_score = 100.0 * (m.repetition_factor as f64 / 7.0).min(1.0);

    // Factor 2: Parity ratio
    // 240 is maximum.
    let parity_score = 100.0 * (m.parity_symbols as f64 / 240.0).min(1.0);

    // Factor 3: Fortress activation
    let fortress_score = if m.fortress { 100.0 } else { 0.0 };

    // Factor 4: QT resilience (continuous, from mean quantization table value)
    // Higher mean_qt = lower QF = larger quant steps = more robust STDM.
    // mean_qt >= 20 (roughly QF 55-60) is excellent; mean_qt ~2 (QF 95) is fragile.
    let qt_score = 100.0 * (m.mean_qt / 20.0).min(1.0);

    // Factor 5: Fill ratio
    // Lower fill = more room for redundancy.
    let fill_score = 100.0 * (1.0 - m.fill_ratio.clamp(0.0, 1.0));

    // Factor 6: Delta strength
    // delta >= 40 is excellent.
    let delta_score = 100.0 * (m.delta / 40.0).min(1.0);

    // When Fortress is active, use full 6-factor weights.
    // When not active, redistribute Fortress's 15% to the other 5 factors
    // so standard Armor isn't hard-capped at ~85%.
    let robustness = if m.fortress {
        0.30 * rep_score
            + 0.20 * parity_score
            + 0.15 * fortress_score
            + 0.15 * qt_score
            + 0.10 * fill_score
            + 0.10 * delta_score
    } else {
        // Redistribute: divide each weight by 0.85 (sum of non-fortress weights)
        (0.30 / 0.85) * rep_score
            + (0.20 / 0.85) * parity_score
            + (0.15 / 0.85) * qt_score
            + (0.10 / 0.85) * fill_score
            + (0.10 / 0.85) * delta_score
    };

    let score = robustness.round().min(100.0).max(0.0) as u8;

    // Pick hint based on dominant/weakest factor.
    let hint_key = if m.fortress {
        "hint_fortress_active"
    } else {
        let factors = [
            (rep_score, "hint_low_repetition"),
            (parity_score, "hint_low_parity"),
            (qt_score, "hint_high_qf"),
            (fill_score, "hint_high_fill"),
            (delta_score, "hint_low_delta"),
        ];
        // Pick the weakest factor.
        factors.iter()
            .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(_, key)| *key)
            .unwrap_or("hint_low_repetition")
    };

    EncodeQuality {
        score,
        hint_key: hint_key.to_string(),
        mode: 2,
    }
}

/// Deterministic tanh approximation (avoids f64::tanh which may not be
/// deterministic in WASM). Uses the identity tanh(x) = (e^2x - 1)/(e^2x + 1).
fn det_tanh(x: f64) -> f64 {
    if x > 10.0 { return 1.0; }
    if x < -10.0 { return -1.0; }
    let e2x = (2.0 * x).exp(); // f64::exp is deterministic (IEEE 754)
    (e2x - 1.0) / (e2x + 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ghost_perfect_stealth() {
        let m = GhostMetrics {
            num_modifications: 0,
            n_used: 10000,
            w: 10,
            total_cost: 0.0,
            median_cost: 20.0,
            is_si: true,
            shadow_modifications: 0,
            total_coefficients: 100000,
        };
        let q = ghost_stealth_score(&m);
        assert_eq!(q.mode, 1);
        assert!(q.score >= 90, "expected >= 90 for perfect stealth, got {}", q.score);
    }

    #[test]
    fn ghost_worst_case() {
        let m = GhostMetrics {
            num_modifications: 5000,
            n_used: 10000,
            w: 1,
            total_cost: 100000.0,
            median_cost: 1.0,
            is_si: false,
            shadow_modifications: 0,
            total_coefficients: 100000,
        };
        let q = ghost_stealth_score(&m);
        assert!(q.score <= 30, "expected <= 30 for worst case, got {}", q.score);
    }

    #[test]
    fn ghost_shadow_penalty() {
        let base = GhostMetrics {
            num_modifications: 100,
            n_used: 10000,
            w: 7,
            total_cost: 500.0,
            median_cost: 15.0,
            is_si: false,
            shadow_modifications: 0,
            total_coefficients: 100000,
        };
        let no_shadow = ghost_stealth_score(&base);

        let with_shadow = GhostMetrics {
            num_modifications: 100,
            n_used: 10000,
            w: 7,
            total_cost: 500.0,
            median_cost: 15.0,
            is_si: false,
            shadow_modifications: 1000,
            total_coefficients: 100000,
        };
        let shadow_q = ghost_stealth_score(&with_shadow);
        assert!(shadow_q.score < no_shadow.score, "shadow penalty should reduce score");
        assert_eq!(shadow_q.hint_key, "hint_shadow_penalty");
    }

    #[test]
    fn ghost_si_bonus() {
        let without_si = GhostMetrics {
            num_modifications: 50,
            n_used: 10000,
            w: 8,
            total_cost: 200.0,
            median_cost: 15.0,
            is_si: false,
            shadow_modifications: 0,
            total_coefficients: 100000,
        };
        let with_si = GhostMetrics {
            num_modifications: 50,
            n_used: 10000,
            w: 8,
            total_cost: 200.0,
            median_cost: 15.0,
            is_si: true,
            shadow_modifications: 0,
            total_coefficients: 100000,
        };
        let q1 = ghost_stealth_score(&without_si);
        let q2 = ghost_stealth_score(&with_si);
        assert!(q2.score > q1.score, "SI should increase score");
        assert_eq!(q2.hint_key, "hint_si_bonus");
    }

    #[test]
    fn armor_fortress_high_score() {
        let m = ArmorMetrics {
            repetition_factor: 15,
            parity_symbols: 240,
            fortress: true,
            mean_qt: 10.0,
            fill_ratio: 0.3,
            delta: 12.0,
        };
        let q = armor_robustness_score(&m);
        assert_eq!(q.mode, 2);
        assert!(q.score >= 75, "expected >= 75 for fortress, got {}", q.score);
        assert_eq!(q.hint_key, "hint_fortress_active");
    }

    #[test]
    fn armor_phase1_low_score() {
        let m = ArmorMetrics {
            repetition_factor: 1,
            parity_symbols: 64,
            fortress: false,
            mean_qt: 3.0,
            fill_ratio: 0.9,
            delta: 5.0,
        };
        let q = armor_robustness_score(&m);
        assert!(q.score <= 40, "expected <= 40 for phase1 near capacity, got {}", q.score);
    }

    #[test]
    fn armor_phase2_medium_score() {
        let m = ArmorMetrics {
            repetition_factor: 5,
            parity_symbols: 192,
            fortress: false,
            mean_qt: 15.0,
            fill_ratio: 0.5,
            delta: 20.0,
        };
        let q = armor_robustness_score(&m);
        assert!(q.score >= 50 && q.score <= 85, "expected 50-85, got {}", q.score);
    }

    #[test]
    fn armor_qf50_max_resilience() {
        let m = ArmorMetrics {
            repetition_factor: 7,
            parity_symbols: 240,
            fortress: false,
            mean_qt: 25.0,
            fill_ratio: 0.2,
            delta: 40.0,
        };
        let q = armor_robustness_score(&m);
        assert!(q.score >= 80, "expected >= 80 for QF50 + r=7, got {}", q.score);
    }

    #[test]
    fn score_clamped_0_100() {
        // Extreme metrics shouldn't produce values outside 0-100.
        let q = ghost_stealth_score(&GhostMetrics {
            num_modifications: 100000,
            n_used: 1,
            w: 0,
            total_cost: 999999.0,
            median_cost: 0.0,
            is_si: false,
            shadow_modifications: 100000,
            total_coefficients: 1,
        });
        assert!(q.score <= 100);

        let q = armor_robustness_score(&ArmorMetrics {
            repetition_factor: 100,
            parity_symbols: 500,
            fortress: true,
            mean_qt: 50.0,
            fill_ratio: -1.0,
            delta: 100.0,
        });
        assert!(q.score <= 100);
    }

    #[test]
    fn det_tanh_basics() {
        assert!((det_tanh(0.0)).abs() < 1e-10);
        assert!((det_tanh(1.0) - 0.7615941559557649).abs() < 1e-10);
        assert!((det_tanh(20.0) - 1.0).abs() < 1e-10);
        assert!((det_tanh(-20.0) + 1.0).abs() < 1e-10);
    }
}
