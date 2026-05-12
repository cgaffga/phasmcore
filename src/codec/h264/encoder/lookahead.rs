// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §v1.7 Phase 2 (#324) — Lookahead rate control.
//
// Pre-encode pass over a buffered window of N frames. Computes per-
// frame complexity (mean-SAD vs previous frame, scene-cut detection)
// and per-frame QP offset suggestion that distributes the bit budget
// proportional to complexity.
//
// ## Algorithm
//
// 1. For each frame i in 0..n_frames, compute frame-level complexity:
//    complexity_i = mean-SAD against frame i-1 (stride-8 sampled).
//    Frame 0 has complexity = 0 (reference for itself).
// 2. Compute mean complexity across the window.
// 3. Per-frame QP offset:
//    offset_i = clamp(-strength * log2(complexity_i / mean_complexity),
//                     -MAX_FRAME_QP_OFFSET, +MAX_FRAME_QP_OFFSET)
//    High-complexity frames (motion, scene changes) get NEGATIVE
//    offset → lower QP → more bits.
//    Low-complexity frames (static, repetitive) get POSITIVE offset →
//    higher QP → fewer bits.
// 4. Detect scene cuts at complexity spikes (mean-SAD > scene_threshold).
//
// ## Composition with MB-tree + AQ
//
// Three QP-offset sources at runtime:
// - Lookahead (this module): FRAME-level offset based on temporal complexity
// - MB-tree (#323): per-MB offset based on forward-reference stability
// - AQ-3 (#302): per-MB offset based on current-frame variance
//
// Total per-MB QP = base_qp + lookahead_frame_offset + mb_tree_offset
//                            + aq_offset, clamped to combined range.
//
// ## Limitations vs a full lookahead RC reference baseline
//
// - **No VBV buffer modeling**: the reference fast encoder tracks a real VBV state across
//   the lookahead window; phasm v1.7.0 uses simple log-ratio
//   distribution. v1.7.1+ for full VBV.
// - **Coarse frame-level only**: no per-MB lookahead-driven offset
//   (MB-tree handles that separately).
// - **No B-frame complexity discounting**: the reference fast encoder weights B-frames lower
//   in the budget; phasm v1.7.0 treats all frames equally.
//
// These keep v1.7.0 implementable in ~1 week scope.

use crate::codec::h264::stego::gop_pattern::detect_scene_cuts_yuv_with_stride;

/// Maximum per-frame QP offset from lookahead analysis. Narrower
/// than per-MB ranges since a frame-level offset multiplies across
/// thousands of MBs. ±3 keeps total swing within ±10 QP when
/// combined with AQ ±6 and MB-tree ±6.
const MAX_FRAME_QP_OFFSET: i32 = 3;

/// Default lookahead window size. Matches the reference fast encoder `--rc-lookahead 20`.
pub const DEFAULT_WINDOW: usize = 20;

/// Default frame-complexity strength multiplier. Calibrated so a
/// 2× more-complex-than-mean frame gets -2 QP, etc.
pub const DEFAULT_STRENGTH: u32 = 2;

/// Output of [`analyze_lookahead_window`].
#[derive(Debug, Clone)]
pub struct LookaheadResult {
    pub n_frames: usize,
    /// Per-frame complexity score (mean-Y-SAD vs previous frame,
    /// stride-8 sampled). Frame 0 is 0.
    pub per_frame_complexity: Vec<u32>,
    /// Per-frame QP offset suggestion. Apply additively to base QP.
    pub per_frame_qp_offset: Vec<i32>,
    /// Display-order indices flagged as scene cuts. Pre-existing
    /// SCENE_CUT_THRESHOLD_DEFAULT semantics; caller can pass to
    /// `iter_encode_order_with_scene_cuts` for IDR forcing.
    pub scene_cuts: Vec<usize>,
    /// Mean complexity across the window (excluding frame 0).
    pub mean_complexity: u32,
}

impl LookaheadResult {
    /// Per-frame QP offset for the given frame. Returns 0 if out of
    /// bounds (defensive).
    pub fn qp_offset(&self, frame_idx: usize) -> i32 {
        if frame_idx >= self.n_frames {
            return 0;
        }
        self.per_frame_qp_offset[frame_idx]
    }
}

/// Analyse a window of yuv420p frames and return per-frame complexity
/// + QP-offset suggestions. Pure function: deterministic across
/// architectures (x86 / ARM / WASM).
///
/// `strength` scales the offset magnitude (1..=4 typical; 0 disables).
pub fn analyze_lookahead_window(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    strength: u32,
) -> LookaheadResult {
    let w = width as usize;
    let h = height as usize;
    let frame_size = w * h * 3 / 2;
    let y_size = w * h;

    if strength == 0 || n_frames == 0 || yuv.len() < n_frames * frame_size {
        return LookaheadResult {
            n_frames,
            per_frame_complexity: vec![0; n_frames],
            per_frame_qp_offset: vec![0; n_frames],
            scene_cuts: Vec::new(),
            mean_complexity: 0,
        };
    }

    // Step 1: per-frame complexity (mean-Y-SAD vs prev frame).
    // Frame 0 = 0; frames 1..n use stride-8 sampled SAD against
    // frame i-1. Matches the gop_pattern::detect_scene_cuts_yuv
    // sampling pattern for consistency.
    let mut complexity = vec![0u32; n_frames];
    for d in 1..n_frames {
        let prev_y = &yuv[(d - 1) * frame_size..(d - 1) * frame_size + y_size];
        let cur_y = &yuv[d * frame_size..d * frame_size + y_size];
        let mut total: u64 = 0;
        let mut count: u64 = 0;
        for y in (0..h).step_by(8) {
            for x in (0..w).step_by(8) {
                let idx = y * w + x;
                let p = prev_y[idx] as i32;
                let c = cur_y[idx] as i32;
                total += (p - c).unsigned_abs() as u64;
                count += 1;
            }
        }
        complexity[d] = if count > 0 { (total / count) as u32 } else { 0 };
    }

    // Step 2: mean complexity (excluding frame 0 which is 0 by
    // construction). Use frames 1..n_frames.
    let sum: u64 = complexity[1..].iter().map(|&c| c as u64).sum();
    let denom = (n_frames.saturating_sub(1)).max(1) as u64;
    let mean_complexity = (sum / denom) as u32;

    // Step 3: per-frame QP offset. Frame 0 = 0 (no comparison).
    let mut offsets = vec![0i32; n_frames];
    if mean_complexity > 0 {
        for d in 1..n_frames {
            let c = complexity[d].max(1) as i32;
            let m = mean_complexity.max(1) as i32;
            // log2(c/m) — negative if c < mean (low motion → +QP).
            let log_ratio = log2_signed(c, m);
            let raw = -(log_ratio * strength as i32) / 2;
            offsets[d] = raw.clamp(-MAX_FRAME_QP_OFFSET, MAX_FRAME_QP_OFFSET);
        }
    }

    // Step 4: scene cuts. Re-use existing primitive.
    let scene_cuts = detect_scene_cuts_yuv_with_stride(
        yuv, width, height, n_frames, 1, /* threshold */ 20,
    );

    LookaheadResult {
        n_frames,
        per_frame_complexity: complexity,
        per_frame_qp_offset: offsets,
        scene_cuts,
        mean_complexity,
    }
}

/// Same as [`compute_mb_tree_qp_offsets`]'s log2_signed but local to
/// this module to avoid a cross-module dependency on a private helper.
fn log2_signed(a: i32, b: i32) -> i32 {
    if a <= 0 || b <= 0 {
        return 0;
    }
    if a == b {
        return 0;
    }
    let la = 31 - (a as u32).leading_zeros() as i32;
    let lb = 31 - (b as u32).leading_zeros() as i32;
    la - lb
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_strength_returns_zero_offsets() {
        let yuv = vec![128u8; 4 * 32 * 32 * 3 / 2];
        let result = analyze_lookahead_window(&yuv, 32, 32, 4, 0);
        for &offset in &result.per_frame_qp_offset {
            assert_eq!(offset, 0);
        }
    }

    #[test]
    fn uniform_static_window_zero_complexity() {
        let yuv = vec![128u8; 6 * 32 * 32 * 3 / 2];
        let result = analyze_lookahead_window(&yuv, 32, 32, 6, 2);
        // Frame 0 is always 0; frames 1..5 should all be 0 too
        // (no motion → 0 SAD).
        for (i, &c) in result.per_frame_complexity.iter().enumerate() {
            assert_eq!(c, 0, "frame {} complexity should be 0, got {}", i, c);
        }
        // All offsets zero.
        for &offset in &result.per_frame_qp_offset {
            assert_eq!(offset, 0);
        }
    }

    #[test]
    fn high_motion_frame_gets_negative_offset() {
        // 4 frames: 0 = uniform, 1 = high-contrast change, 2,3 = uniform.
        // Frame 1 should have high complexity → negative QP offset.
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let mut yuv = Vec::with_capacity(4 * frame_size);
        yuv.extend(std::iter::repeat(128u8).take(frame_size));    // 0: uniform
        yuv.extend(std::iter::repeat(0u8).take(frame_size));       // 1: black
        yuv.extend(std::iter::repeat(0u8).take(frame_size));       // 2: black
        yuv.extend(std::iter::repeat(0u8).take(frame_size));       // 3: black
        let result = analyze_lookahead_window(&yuv, w, h, 4, 2);
        // Frame 1's complexity is ~128 (mean SAD); frames 2,3 are 0.
        assert!(result.per_frame_complexity[1] > 100,
            "frame 1 (change) complexity {} should be > 100",
            result.per_frame_complexity[1]);
        // Mean complexity = (128 + 0 + 0) / 3 = ~43.
        // Frame 1 ratio = 128/43 ≈ 3 → log2 ≈ 1+ → offset = -strength*1/2 = -1
        assert!(result.per_frame_qp_offset[1] < 0,
            "high-motion frame 1 should get negative offset, got {}",
            result.per_frame_qp_offset[1]);
        assert!(result.per_frame_qp_offset[2] > 0 || result.per_frame_qp_offset[2] == 0,
            "static frame 2 should get ≥0 offset, got {}",
            result.per_frame_qp_offset[2]);
    }

    #[test]
    fn offsets_clamped_to_max() {
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let mut yuv = Vec::with_capacity(6 * frame_size);
        for i in 0..6 {
            let v = if i % 2 == 0 { 0u8 } else { 255u8 };
            yuv.extend(std::iter::repeat(v).take(frame_size));
        }
        let result = analyze_lookahead_window(&yuv, w, h, 6, 10);
        for &offset in &result.per_frame_qp_offset {
            assert!(offset.abs() <= MAX_FRAME_QP_OFFSET,
                "offset {} exceeds clamp ±{}", offset, MAX_FRAME_QP_OFFSET);
        }
    }

    #[test]
    fn scene_cuts_detected() {
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let mut yuv = Vec::with_capacity(6 * frame_size);
        for d in 0..6 {
            let val = if d < 4 { 128u8 } else { 0u8 };
            yuv.extend(std::iter::repeat(val).take(frame_size));
        }
        let result = analyze_lookahead_window(&yuv, w, h, 6, 2);
        assert!(result.scene_cuts.contains(&4),
            "scene cut at frame 4 expected, got {:?}", result.scene_cuts);
    }
}
