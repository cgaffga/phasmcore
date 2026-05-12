// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §v1.7 Phase 1 (#323) — MB-tree quality propagation.
//
// Content-adaptive per-MB QP offset based on forward-propagation
// analysis. Matches the spec-permitted MB-tree quality-propagation rate-control
// that distributes bits toward macroblocks referenced by many future
// frames (= long-lived content) and away from one-off transitional
// content.
//
// ## Algorithm (simplified vs the reference fast encoder)
//
// the reference fast encoder's full MB-tree runs a complete lookahead encode pass with
// lightweight ME and tracks per-MB downstream-reference graph. Phasm's
// v1.7.0 simplification skips full ME and uses **same-position SAD
// stability** as a proxy for reference-graph weight:
//
// 1. Pre-pass: for each pair of consecutive frames (i, i+1), compute
//    per-MB SAD between same-position 16x16 blocks.
// 2. Per-MB propagation weight: for MB at (frame_i, x, y), count
//    how many future frames i+1..i+window have LOW same-position SAD
//    (below `STABILITY_THRESHOLD`). Each stable future-frame match
//    counts as one reference.
// 3. Per-MB QP offset = clamp(-strength * log2(weight / mean_weight)
//    , -MAX_OFFSET, MAX_OFFSET).
//
// Long-lived MBs (high weight) get NEGATIVE offset → lower QP → more
// bits. One-shot MBs (low weight) get POSITIVE offset → higher QP →
// fewer bits.
//
// ## Limitations vs full MB-tree
//
// - **No motion compensation**: same-position SAD misses MBs that move
//   between frames. A textured MB that consistently moves 5 px would
//   appear "unstable" by same-position SAD but is actually long-lived
//   content the reference fast encoder would weight high. Full ME (Phase 1.x follow-on)
//   addresses this.
// - **Equal frame weighting**: all future frames count equally; the reference fast encoder
//   weights closer frames higher because they're more likely to use
//   the past MB as a direct reference.
// - **No L0/L1 distinction**: the reference fast encoder tracks separate reference graphs for
//   P and B-frames; phasm v1.7.0 collapses both.
//
// These simplifications keep v1.7.0 implementable in 1 week scope. A
// full MB-tree port lands later as v1.7.1+ if Phase 1.0 measurement
// shows meaningful visual benefit on carplane.
//
// ## Composition with AQ-3
//
// AQ-3 (PHASM_AQ_MODE=3) gives per-MB QP offset based on
// CURRENT-frame variance. MB-tree gives per-MB QP offset based on
// FUTURE-frame propagation. The two compose ADDITIVELY (clamped
// to ±MAX_TOTAL_OFFSET) at the per-MB QP application site.

use std::collections::BTreeMap;

/// Maximum per-MB QP offset from MB-tree analysis. Matches the AQ
/// clamp window (±6) so total combined offset stays in a defensible
/// range when MB-tree composes with AQ-3.
const MAX_QP_OFFSET: i32 = 6;

/// Same-position SAD threshold below which an MB is considered
/// "stable" (likely-referenced by future). Mean-absolute-deviation
/// per pixel ≤ 8 → SAD ≤ 8 × 256 = 2048 for a 16x16 luma block.
/// Calibrated empirically; higher = more MBs count as stable.
const STABILITY_SAD_THRESHOLD: u32 = 2048;

/// Default lookahead window for forward-reference counting. 20 frames
/// matches the reference fast encoder default `--rc-lookahead 20` (couples with Phase 2
/// lookahead RC).
pub const DEFAULT_LOOKAHEAD: usize = 20;

/// Default MB-tree strength multiplier (0..=10). 0 disables; higher
/// values amplify the per-MB QP offset magnitude. Tuned so that
/// strength=4 gives a roughly the reference fast encoder-compatible offset distribution.
pub const DEFAULT_STRENGTH: u32 = 4;

/// Output of [`compute_mb_tree_qp_offsets`]: per-frame, per-MB QP
/// offset suitable for additive composition with AQ offsets. Indexed
/// as `result[frame_idx][mb_y * mb_w + mb_x]`.
#[derive(Debug, Clone)]
pub struct MbTreeResult {
    pub mb_w: usize,
    pub mb_h: usize,
    pub n_frames: usize,
    pub offsets: Vec<Vec<i32>>,
    pub mean_weight_q8: u32,
}

impl MbTreeResult {
    /// Per-MB QP offset for the given frame + MB position. Returns 0
    /// if indices are out of bounds (defensive — caller should not
    /// rely on this for correctness).
    pub fn qp_offset(&self, frame_idx: usize, mb_x: usize, mb_y: usize) -> i32 {
        if frame_idx >= self.n_frames || mb_x >= self.mb_w || mb_y >= self.mb_h {
            return 0;
        }
        self.offsets[frame_idx][mb_y * self.mb_w + mb_x]
    }
}

/// Compute per-frame per-MB QP offset using simplified MB-tree
/// analysis. Pure function: deterministic for given inputs across
/// architectures (x86 / ARM / WASM).
///
/// `yuv` is a concatenated yuv420p byte slice covering `n_frames`
/// frames at `(width, height)` resolution. `strength` scales the
/// offset magnitude (typically 1..=10; 0 disables, returns all-zero
/// offsets).
///
/// `lookahead` is the forward window size — for each MB, future
/// frames at positions `frame+1..frame+lookahead` are checked for
/// same-position SAD stability. Higher = more propagation weight,
/// but more compute. `DEFAULT_LOOKAHEAD = 20` is a reasonable default.
pub fn compute_mb_tree_qp_offsets(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: usize,
    strength: u32,
    lookahead: usize,
) -> MbTreeResult {
    let w = width as usize;
    let h = height as usize;
    let mb_w = w / 16;
    let mb_h = h / 16;
    let mb_count = mb_w * mb_h;
    let frame_size = w * h * 3 / 2;
    let y_size = w * h;

    if strength == 0 || n_frames < 2 || yuv.len() < n_frames * frame_size {
        return MbTreeResult {
            mb_w,
            mb_h,
            n_frames,
            offsets: (0..n_frames).map(|_| vec![0; mb_count]).collect(),
            mean_weight_q8: 0,
        };
    }

    // Step 1: per-MB propagation weight. For each (frame, mb) pair,
    // count how many future frames in the lookahead window have low
    // same-position SAD. Weight is normalized to Q8 fraction of
    // available window so end-of-clip frames (smaller available
    // window) don't get artificially low weight just because their
    // future window is truncated.
    let mut weights: Vec<Vec<u32>> = (0..n_frames)
        .map(|_| vec![0u32; mb_count])
        .collect();

    for frame_i in 0..n_frames {
        let cur_y = &yuv[frame_i * frame_size..frame_i * frame_size + y_size];
        let end = (frame_i + lookahead + 1).min(n_frames);
        let window_size = end.saturating_sub(frame_i + 1);
        if window_size == 0 {
            // Last frame has no future window; weight 0 means it
            // contributes nothing to mean but its own offset will be
            // zero (no propagation info to act on).
            continue;
        }
        let mut counts = vec![0u32; mb_count];
        for frame_j in (frame_i + 1)..end {
            let fut_y = &yuv[frame_j * frame_size..frame_j * frame_size + y_size];
            for mb_y in 0..mb_h {
                for mb_x in 0..mb_w {
                    let sad = mb_sad_16x16(cur_y, fut_y, w, mb_x * 16, mb_y * 16);
                    if sad <= STABILITY_SAD_THRESHOLD {
                        counts[mb_y * mb_w + mb_x] += 1;
                    }
                }
            }
        }
        // Normalise to Q8 fraction. 256 = 1.0 = every future frame
        // in window was stable for this MB.
        for (idx, &c) in counts.iter().enumerate() {
            weights[frame_i][idx] = (c * 256) / window_size as u32;
        }
    }

    // Step 2: compute mean weight across all (frame, mb) pairs.
    let mut total: u64 = 0;
    let mut count: u64 = 0;
    for frame_weights in &weights {
        for &w in frame_weights {
            total += w as u64;
            count += 1;
        }
    }
    let mean_weight = if count > 0 { total / count } else { 0 };
    let mean_weight_q8 = (mean_weight << 8) as u32;

    // Step 3: per-MB QP offset = -strength * log2(weight / mean_weight)
    // clamped to ±MAX_QP_OFFSET.
    //
    // Integer-arithmetic log2 of weight relative to mean: returns
    // signed offset suitable for direct QP modulation.
    let mut offsets: Vec<Vec<i32>> = (0..n_frames)
        .map(|_| vec![0i32; mb_count])
        .collect();
    if mean_weight == 0 {
        // No stable MBs anywhere; uniform high-motion fixture. Return
        // zero offsets (rely on base QP).
        return MbTreeResult {
            mb_w,
            mb_h,
            n_frames,
            offsets,
            mean_weight_q8,
        };
    }
    for (frame_i, frame_weights) in weights.iter().enumerate() {
        for (mb_idx, &w) in frame_weights.iter().enumerate() {
            let weight = w.max(1) as i32;
            let mean = mean_weight.max(1) as i32;
            let log_ratio = log2_signed(weight, mean);
            let raw_offset = -(log_ratio * strength as i32) / 4;
            offsets[frame_i][mb_idx] = raw_offset.clamp(-MAX_QP_OFFSET, MAX_QP_OFFSET);
        }
    }

    MbTreeResult {
        mb_w,
        mb_h,
        n_frames,
        offsets,
        mean_weight_q8,
    }
}

/// 16x16 SAD between same-position luma blocks in two frames.
/// Deterministic across architectures (pure integer arithmetic, no
/// SIMD, no float).
fn mb_sad_16x16(plane_a: &[u8], plane_b: &[u8], stride: usize, x0: usize, y0: usize) -> u32 {
    let mut total: u32 = 0;
    for dy in 0..16 {
        for dx in 0..16 {
            let idx = (y0 + dy) * stride + x0 + dx;
            let a = plane_a[idx] as i32;
            let b = plane_b[idx] as i32;
            total += (a - b).unsigned_abs();
        }
    }
    total
}

/// Signed integer log2 of `a/b`. Returns positive if a > b, negative
/// if a < b, zero if a == b. Uses leading-zero counting for the
/// integer log2 approximation; ratio < 1 reflected as negative
/// log2(1/ratio).
fn log2_signed(a: i32, b: i32) -> i32 {
    if a <= 0 || b <= 0 {
        return 0;
    }
    if a == b {
        return 0;
    }
    if a > b {
        // log2(a/b) ≈ floor(log2(a)) - floor(log2(b))
        let la = 31 - (a as u32).leading_zeros() as i32;
        let lb = 31 - (b as u32).leading_zeros() as i32;
        la - lb
    } else {
        // log2(a/b) = -log2(b/a)
        let la = 31 - (a as u32).leading_zeros() as i32;
        let lb = 31 - (b as u32).leading_zeros() as i32;
        la - lb
    }
}

/// Diagnostic summary: count distribution of per-MB QP offsets across
/// the whole video. Used by tests + Phase 1.1 orchestrator integration
/// to sanity-check that MB-tree is producing meaningful variance, not
/// near-zero uniform offsets.
pub fn offset_histogram(result: &MbTreeResult) -> BTreeMap<i32, u32> {
    let mut hist: BTreeMap<i32, u32> = BTreeMap::new();
    for frame in &result.offsets {
        for &offset in frame {
            *hist.entry(offset).or_insert(0) += 1;
        }
    }
    hist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_strength_returns_zero_offsets() {
        let yuv = vec![128u8; 4 * 32 * 32 * 3 / 2];
        let result = compute_mb_tree_qp_offsets(&yuv, 32, 32, 4, 0, DEFAULT_LOOKAHEAD);
        for frame in &result.offsets {
            for &offset in frame {
                assert_eq!(offset, 0, "zero strength → all-zero offsets");
            }
        }
    }

    #[test]
    fn uniform_static_content_no_intra_frame_variance() {
        // All frames identical → all MBs within a frame have the same
        // normalised propagation weight → all MBs within a frame have
        // the same offset. Cross-frame variance is allowed (last frame
        // has no future window). MB-tree's job is to spread bits
        // ACROSS MBs within a frame based on content — uniform content
        // should produce zero MB-to-MB variance per frame.
        let yuv = vec![128u8; 6 * 32 * 32 * 3 / 2];
        let result = compute_mb_tree_qp_offsets(&yuv, 32, 32, 6, 4, DEFAULT_LOOKAHEAD);
        for (frame_idx, frame_offsets) in result.offsets.iter().enumerate() {
            let first = frame_offsets[0];
            for (mb_idx, &offset) in frame_offsets.iter().enumerate() {
                assert_eq!(
                    offset, first,
                    "uniform content: frame {} mb {} offset {} differs from mb 0 offset {}",
                    frame_idx, mb_idx, offset, first,
                );
            }
        }
    }

    #[test]
    fn dimensions_match_input() {
        let yuv = vec![128u8; 4 * 64 * 48 * 3 / 2];
        let result = compute_mb_tree_qp_offsets(&yuv, 64, 48, 4, 4, DEFAULT_LOOKAHEAD);
        assert_eq!(result.mb_w, 4);
        assert_eq!(result.mb_h, 3);
        assert_eq!(result.n_frames, 4);
        assert_eq!(result.offsets.len(), 4);
        assert_eq!(result.offsets[0].len(), 12);
    }

    #[test]
    fn high_motion_mb_gets_positive_offset() {
        // Construct fixture: 4 frames, 32x32 (2x2 MBs in luma).
        // MB(0,0) stays constant across all frames → high stability
        //   weight → log2 of ratio is positive → negative QP offset
        //   (more bits, finer quant).
        // MB(1,1) flips every frame → zero stability weight → log2 of
        //   ratio is negative → positive QP offset (fewer bits).
        // Other MBs at (1,0) and (0,1) stay constant.
        let w = 32u32;
        let h = 32u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let y_size = (w * h) as usize;
        let mut yuv = Vec::with_capacity(4 * frame_size);
        for f in 0..4 {
            let mut frame = vec![128u8; frame_size];
            // MB(1,1) = pixels at rows 16..32, cols 16..32.
            let mb11_value = if f % 2 == 0 { 0u8 } else { 255u8 };
            for y in 16..32 {
                for x in 16..32 {
                    frame[y * 32 + x] = mb11_value;
                }
            }
            // Chroma: keep mid-gray.
            for px in &mut frame[y_size..] {
                *px = 128;
            }
            yuv.extend(frame);
        }
        let result = compute_mb_tree_qp_offsets(&yuv, w, h, 4, 4, DEFAULT_LOOKAHEAD);
        let mb00_offset = result.qp_offset(0, 0, 0);
        let mb11_offset = result.qp_offset(0, 1, 1);
        // MB(0,0) is stable → SHOULD get non-positive offset
        // (negative or zero), meaning lower or equal QP vs base.
        assert!(
            mb00_offset <= 0,
            "stable MB(0,0) should get ≤0 offset (more bits), got {}",
            mb00_offset
        );
        // MB(1,1) flips → SHOULD get non-negative offset (positive or
        // zero), meaning higher or equal QP vs base.
        assert!(
            mb11_offset >= 0,
            "unstable MB(1,1) should get ≥0 offset (fewer bits), got {}",
            mb11_offset
        );
        // And the two must DIFFER — otherwise MB-tree isn't producing
        // meaningful variance.
        assert!(
            mb00_offset < mb11_offset,
            "stable MB(0,0) should have STRICTLY lower offset than unstable MB(1,1); \
             stable={} vs unstable={}",
            mb00_offset,
            mb11_offset,
        );
    }

    #[test]
    fn offsets_clamped_to_max_range() {
        // Extreme case: half static + half flipping. Verify all
        // offsets fall within ±MAX_QP_OFFSET.
        let w = 64u32;
        let h = 64u32;
        let frame_size = (w * h * 3 / 2) as usize;
        let y_size = (w * h) as usize;
        let mut yuv = Vec::with_capacity(8 * frame_size);
        for f in 0..8 {
            let mut frame = vec![128u8; frame_size];
            // Right half flips, left half stays.
            let v = if f % 2 == 0 { 0u8 } else { 255u8 };
            for y in 0..64 {
                for x in 32..64 {
                    frame[y * 64 + x] = v;
                }
            }
            for px in &mut frame[y_size..] {
                *px = 128;
            }
            yuv.extend(frame);
        }
        let result = compute_mb_tree_qp_offsets(&yuv, w, h, 8, 10, DEFAULT_LOOKAHEAD);
        for (frame_idx, frame) in result.offsets.iter().enumerate() {
            for (mb_idx, &offset) in frame.iter().enumerate() {
                assert!(
                    offset.abs() <= MAX_QP_OFFSET,
                    "offset {} at frame {} mb {} exceeds clamp ±{}",
                    offset, frame_idx, mb_idx, MAX_QP_OFFSET,
                );
            }
        }
    }
}
