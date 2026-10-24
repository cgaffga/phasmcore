// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! D2 GATE — the shadow-video primary payload must be planned against the
//! MEASURED per-GOP capacities, not split evenly.
//!
//! `h264_encode_with_shadows_streaming` drains the primary payload one GOP at
//! a time (`embed_primary_one_gop`). Without a plan it aims for an even share
//! of what is left (`remaining / gops_remaining`) and shrinks on infeasibility,
//! carrying the shortfall forward. That carry is not enough: a capacity-rich
//! early GOP is still held to the even share, so its headroom is wasted, and a
//! capacity-poor GOP later cannot absorb the inflated share it inherits.
//!
//! Meanwhile the capacity the UI *displays* comes from
//! `max_message_len` (via `oh264_capacity`'s `achievable_max_message_bytes`) —
//! the PLANNER's ceiling. So before D2 the shadow path advertised the
//! planner's number and embedded with the even split, and the user got a
//! multi-minute encode followed by a refusal for a message the bar had
//! promised would fit. (Same defect class as #841 on the no-shadow path, and
//! as D1: see `docs/notes/2026-10-20-video-encode-ux-defects.md`.)
//!
//! What this gate pins, on the two measured corpus cap vectors:
//!   1. the planner's advertised ceiling is fully drainable WITH the plan, and
//!   2. it is far out of reach of the even split.
//!
//! The drain here is a MODEL of `embed_primary_one_gop`'s loop, not the
//! function itself: real acceptance is "`stc_embed` succeeded", which needs a
//! full OH264 encode per candidate size. The model uses the probe's per-GOP
//! cap as the acceptance bound — which is exactly the quantity the planner and
//! the capacity display are both built on. The end-to-end round-trip is gated
//! separately by `oh264_video_shadow_roundtrip`.

use phasm_core::stego::balanced_allocation::{max_message_len, plan_safe_balanced, PlanOutcome};
use phasm_core::stego::calibration::AllocationCalibration;

/// iphone7, 1920x1072x510f, 17 GOPs, Σ = 15215, min = 320.
/// The minimum is the LAST GOP — the worst case for a forward-only carry,
/// which has nothing after it to spill into. This is the one vector where the
/// even split really does ceiling out at `n × min(cap)`.
const IPHONE7_CAPS: [usize; 17] = [
    806, 1613, 1373, 720, 820, 1060, 586, 1026, 1140, 1526, 1133, 640, 700, 373, 546, 833, 320,
];

/// IMG_4138, 1920x1072x600f, 20 GOPs, Σ = 36227, min = 26 (index 4).
/// Here the carry does help — `n × min` is only 520, while the even split
/// reaches 16309 — because the poorest GOP is early and has 15 GOPs of
/// headroom behind it to spill into. It is still ~2× short of the ceiling.
const IMG4138_CAPS: [usize; 20] = [
    1833, 213, 120, 373, 26, 186, 1180, 2653, 5180, 2213, 4060, 8733, 953, 366, 1013, 833, 526,
    1160, 2700, 1906,
];

const GOP_SIZE: usize = 30;

/// Model of `embed_primary_one_gop`'s per-GOP drain over a whole clip.
/// `plan = None` is the legacy even split. Returns true if all `l` bytes land.
fn drains(caps: &[usize], l: usize, plan: Option<&[usize]>) -> bool {
    let n = caps.len();
    let mut cursor = 0usize;
    for g in 0..n {
        let remaining = l - cursor;
        // Plain-tail GOP: payload already fully embedded.
        if g > 0 && remaining == 0 {
            continue;
        }
        // Mirrors the `want` match in `embed_primary_one_gop`: with a plan the
        // target is CUMULATIVE (catch up to `Σ plan[..=g]`), so a GOP that
        // under-filled is made up by the next one.
        let mut want = match plan {
            // Last GOP takes everything left, plan or no plan.
            _ if n - g <= 1 => remaining,
            Some(p) => p
                .iter()
                .take(g + 1)
                .sum::<usize>()
                .saturating_sub(cursor)
                .min(remaining),
            None => remaining.div_ceil(n - g),
        };
        // The shrink loop, same ~12% step.
        while want > caps[g] && want > 0 {
            want -= 1usize.max(want / 8);
        }
        cursor += want;
    }
    cursor == l
}

/// Largest payload that drains. Scanned exhaustively rather than bisected:
/// acceptance is NOT monotone in payload size (the shrink step's granularity
/// and the planner's window both make it jagged), so a bisection would report
/// whatever the first probe happened to hit.
fn max_drainable(caps: &[usize], planned: bool) -> usize {
    let cal = AllocationCalibration::H264_1080P_QP26;
    let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
    let sigma: usize = caps.iter().sum();
    (1..=sigma)
        .filter(|&l| {
            if planned {
                match plan_safe_balanced(&samples, caps.len(), l, GOP_SIZE, &cal) {
                    PlanOutcome::Plan { plan, .. } => drains(caps, l, Some(&plan)),
                    _ => false,
                }
            } else {
                drains(caps, l, None)
            }
        })
        .max()
        .unwrap_or(0)
}

/// The number the capacity bar shows must be the number the drain can reach.
/// Before D2 the shadow path showed this and then embedded with the even
/// split, which on these vectors stops at 39% / 49% of it.
#[test]
fn planner_ceiling_is_drainable_and_the_even_split_is_not() {
    let cal = AllocationCalibration::H264_1080P_QP26;
    for (name, caps) in [
        ("iphone7", &IPHONE7_CAPS[..]),
        ("img4138", &IMG4138_CAPS[..]),
    ] {
        let n = caps.len();
        let sigma: usize = caps.iter().sum();
        let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
        // What `oh264_capacity` advertises (minus the one-time crypto
        // envelope, which is payload-shape, not allocation).
        let advertised = max_message_len(&samples, n, &cal);
        let planned = max_drainable(caps, true);
        let even = max_drainable(caps, false);

        eprintln!(
            "D2 {name:9} n={n:3} Σ={sigma:6} n·min={:6} advertised={advertised:6} \
             planned={planned:6} even={even:6} gain={:.2}×",
            n * caps.iter().min().unwrap(),
            planned as f64 / even as f64,
        );

        assert_eq!(
            planned, advertised,
            "{name}: the plan must reach the ceiling the capacity bar \
             advertises ({advertised} B), but only drained {planned} B — the \
             displayed capacity would be a promise the encoder cannot keep",
        );
        assert!(
            even < advertised,
            "{name}: the even split reached {even} B of an advertised \
             {advertised} B. If this ever stops failing the D2 fix has become \
             unnecessary — confirm with a real encode before deleting it.",
        );
        // Measured 2.54× (iphone7) and 2.02× (img4138). Guard well below both
        // so a calibration tweak doesn't fail the gate for no reason, but high
        // enough that losing the plan entirely does.
        assert!(
            planned as f64 >= 1.8 * even as f64,
            "{name}: plan {planned} B vs even split {even} B is only \
             {:.2}× — the proportional plan is not being applied",
            planned as f64 / even as f64,
        );
    }
}

/// The plan must never hand a GOP more than it was measured to hold — the
/// invariant that makes the drain's first attempt succeed instead of shrinking.
#[test]
fn plan_never_exceeds_a_measured_cap() {
    let cal = AllocationCalibration::H264_1080P_QP26;
    for (name, caps) in [
        ("iphone7", &IPHONE7_CAPS[..]),
        ("img4138", &IMG4138_CAPS[..]),
    ] {
        let samples: Vec<(usize, usize)> = caps.iter().copied().enumerate().collect();
        let advertised = max_message_len(&samples, caps.len(), &cal);
        for l in [1, 100, advertised / 2, advertised] {
            if l == 0 {
                continue;
            }
            match plan_safe_balanced(&samples, caps.len(), l, GOP_SIZE, &cal) {
                PlanOutcome::Plan { plan, .. } => {
                    assert_eq!(plan.iter().sum::<usize>(), l, "{name}: plan must place all {l} B");
                    for (g, (&got, &cap)) in plan.iter().zip(caps.iter()).enumerate() {
                        assert!(
                            got <= cap,
                            "{name}: L={l} gave GOP {g} {got} B but it measured {cap} B",
                        );
                    }
                }
                other => panic!("{name}: L={l} (≤ advertised {advertised}) refused: {other:?}"),
            }
        }
    }
}

/// A cap vector whose GOP count disagrees with the clip's (a frame-count
/// rounding between the probe and the encode) must never do WORSE than having
/// no plan at all. Without the last-GOP guard in `embed_primary_one_gop` the
/// cumulative target `Σ plan[..=last]` falls short of the payload, strands the
/// remainder, and the drain check reports MessageTooLarge — a plan that turns
/// a working encode into a refusal.
#[test]
fn an_over_long_plan_still_drains_everything() {
    let cal = AllocationCalibration::H264_1080P_QP26;
    // Plan sized for 19 GOPs, drained over the clip's real 17.
    let caps = &IPHONE7_CAPS[..];
    let mut padded: Vec<usize> = caps.to_vec();
    padded.push(500);
    padded.push(500);
    let samples: Vec<(usize, usize)> = padded.iter().copied().enumerate().collect();
    let l = max_message_len(&samples, padded.len(), &cal);
    let plan = match plan_safe_balanced(&samples, padded.len(), l, GOP_SIZE, &cal) {
        PlanOutcome::Plan { plan, .. } => plan,
        other => panic!("expected a plan at L={l}, got {other:?}"),
    };
    let through_17: usize = plan.iter().take(caps.len()).sum();
    assert!(
        through_17 < l,
        "test is vacuous unless the 19-GOP plan's 17-GOP prefix is short of          the payload (prefix {through_17} B, payload {l} B)",
    );
    // The last GOP of the real clip must absorb the rest. It cannot exceed its
    // own cap, so this asserts the guard applies, not that the payload fits.
    let n = caps.len();
    let mut cursor = 0usize;
    for g in 0..n {
        let remaining = l - cursor;
        if g > 0 && remaining == 0 {
            continue;
        }
        let mut want = match () {
            _ if n - g <= 1 => remaining,
            _ => plan
                .iter()
                .take(g + 1)
                .sum::<usize>()
                .saturating_sub(cursor)
                .min(remaining),
        };
        let asked_for_everything_left = g == n - 1 && want == remaining;
        while want > caps[g] && want > 0 {
            want -= 1usize.max(want / 8);
        }
        if g == n - 1 {
            assert!(
                asked_for_everything_left,
                "the last GOP must ask for the whole remainder ({remaining} B),                  not the plan's cumulative target",
            );
        }
        cursor += want;
    }
}
