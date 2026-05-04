// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §6E-A6.5 — End-to-end stealth-distribution gate.
//
// Phasm's IBPBP B-slice mode-decision compared against the
// SAME-FIXTURE x264-medium-CRF23 reference (committed at
// `core/tests/data/x264_reference_mb_histogram_samefixture.txt`,
// rebaselined per §6E-D.6(c) on 2026-05-03).
//
// **Structural invariants** (HARD asserts — must pass):
//   - B-frame direction L0/L1/Bi within ε=5pp of x264 reference
//   - Sub-8x8 partitions = 0% (matching x264 medium; emitting any
//     would narrow the "look-like-x264" crowd to slow-preset)
//   - P-frame: 100% L0, no Bi
//   - I-frame: ≥99% no-MV records
//
// **Partition-shape distribution** (SOFT warn — blocked on Phase 6E-D
// real RDO mode-decision, task #158):
//   - 16x16 / 8x8 / 16x8 / 8x16 share deltas vs x264 reference are
//     printed but NOT asserted. Phasm's hash-bucket mode decision
//     in `mb_decision_b_with_mvs` produces ~99% 16x16 because no
//     content-driven RDO selects partitioned modes.
//   - Closing this gap requires REAL RDO mode-decision (Phase 6E-D),
//     NOT bucket calibration. Per
//     `memory/feedback_no_cosmetic_calibration.md`, padding bucket
//     distribution with Partitioned/B_8x8 variants purely to match
//     marginal histograms is a stealth ANTI-PATTERN — joint
//     (shape × motion × neighbour) steganalysis features expose
//     cosmetic mode diversity that has no content-driven reason
//     to exist. Marginal histogram match without joint structural
//     match is MORE detectable, not less.
//   - 4 rounds of empirical bucket calibration tested 2026-05-03
//     (commits dfc4fa4, 97e91a6, 762cc89). All regressed Σ|Δ| from
//     baseline 38.9pp to 137-180pp. Findings preserved in
//     `memory/h264_distribution_calibration_findings.md`.
//
// Procedure:
//   1. Encode a 1080p × 10-frame IBPBP fixture via the §30D-C
//      orchestrator (`h264_stego_encode_yuv_string_4domain_multigop_with_pattern`).
//   2. Pipe the output Annex-B through
//      `scripts/h264_mb_partition_histogram.py` (PyAV-based; reads
//      MV side-data from libavcodec).
//   3. Parse + compare the resulting B-frame partition shape +
//      direction shares against the x264 reference within ε=5pp.
//   4. Assert P-frame discipline (100% L0) + I-frame discipline
//      (zero MVs) + zero sub-8x8 partitions (matching x264 medium).
//
// `#[ignore]` because this depends on:
//   - `/tmp/img4138_1080p_f10.yuv` (regen via
//     `core/test-vectors/video/h264/real-world/source/regen.sh`)
//   - `python3` + `PyAV`
//   - `scripts/h264_mb_partition_histogram.py`
//   - ~5-10 min runtime at 1080p
//
// Run with:
//   cargo test --features cabac-stego --test h264_stego_distribution_gate -- --ignored

#![cfg(feature = "cabac-stego")]

use phasm_core::{
    h264_stego_encode_yuv_string_4domain_multigop_with_pattern,
    GopPattern,
};

// §6E-D.6(c) — re-baselined 2026-05-03 to use the SAME-FIXTURE x264
// encode (10 frames of IMG_4138 at 1920x1072 IBPBP) as the comparison
// reference. The legacy 271-frame mixed-content batch reference
// (`x264_reference_mb_histogram.txt`) is kept for historical context
// but is NOT used for assertions — its content distribution differs
// significantly from any 4-B-frame slice of IMG_4138, so comparing
// the two was apples-to-oranges. The new same-fixture reference
// reflects what x264-medium does on IDENTICAL content to phasm.
//
// See `memory/h264_phase6e_d_x264_same_fixture.md` for the
// measurement, and `memory/h264_phase6e_d_trace_breakthrough.md`
// for why same-fixture comparison is necessary.
const X264_REFERENCE_PATH: &str = "tests/data/x264_reference_mb_histogram_samefixture.txt";
const EPSILON_PCT: f64 = 5.0;

#[derive(Debug, Default, Clone)]
struct PerPicTypeHist {
    n_mbs: u64,
    no_mv_pct: f64,
    p16x16_pct: f64,
    p8x8_pct: f64,
    p16x8_pct: f64,
    p8x16_pct: f64,
    p_sub8x8_pct: f64,
    l0_pct: f64,
    l1_pct: f64,
    bi_pct: f64,
}

#[derive(Debug, Default)]
struct ParsedHist {
    b: PerPicTypeHist,
    p: PerPicTypeHist,
    i: PerPicTypeHist,
}

fn parse_histogram(text: &str) -> ParsedHist {
    let mut out = ParsedHist::default();
    let mut current: Option<&mut PerPicTypeHist> = None;
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("## B-frame") {
            current = Some(&mut out.b);
        } else if line.starts_with("## P-frame") {
            current = Some(&mut out.p);
        } else if line.starts_with("## I-frame") {
            current = Some(&mut out.i);
        } else if let Some(h) = current.as_deref_mut() {
            // Match patterns like "  16x16 :  N (P%)"
            let pct = extract_pct(line);
            if line.starts_with("no-MV") {
                if let Some(p) = pct {
                    h.no_mv_pct = p;
                }
            } else if line.starts_with("16x16") {
                if let Some(p) = pct {
                    h.p16x16_pct = p;
                }
            } else if line.starts_with("8x8 ") || line.starts_with("8x8:") {
                if let Some(p) = pct {
                    h.p8x8_pct = p;
                }
            } else if line.starts_with("16x8") {
                if let Some(p) = pct {
                    h.p16x8_pct = p;
                }
            } else if line.starts_with("8x16") {
                if let Some(p) = pct {
                    h.p8x16_pct = p;
                }
            } else if line.starts_with("8x4") || line.starts_with("4x8") || line.starts_with("4x4") {
                if let Some(p) = pct {
                    h.p_sub8x8_pct += p;
                }
            } else if line.starts_with("L0") {
                if let Some(p) = pct {
                    h.l0_pct = p;
                }
            } else if line.starts_with("L1") {
                if let Some(p) = pct {
                    h.l1_pct = p;
                }
            } else if line.starts_with("Bi") {
                if let Some(p) = pct {
                    h.bi_pct = p;
                }
            }
        }
    }
    out
}

fn extract_pct(line: &str) -> Option<f64> {
    let open = line.find('(')?;
    let close = line[open..].find('%')?;
    let inner = &line[open + 1..open + close];
    inner.trim().parse().ok()
}

fn assert_within_eps(label: &str, phasm: f64, x264: f64, eps: f64) {
    let delta = (phasm - x264).abs();
    assert!(
        delta <= eps,
        "{label}: phasm {phasm:.2}% vs x264 {x264:.2}% (Δ={delta:.2}pp > ε={eps}pp)"
    );
}

#[test]
#[ignore = "1080p stealth gate; needs /tmp/img4138_1080p_f10.yuv + python3+PyAV + ~5-10 min"]
fn ibpbp_b_frame_distribution_within_5pp_of_x264_medium() {
    // 1. Load fixture.
    let yuv = match std::fs::read("/tmp/img4138_1080p_f10.yuv") {
        Ok(b) => b,
        Err(e) => {
            eprintln!(
                "Skipping: /tmp/img4138_1080p_f10.yuv missing ({e}). \
                 Regen via core/test-vectors/video/h264/real-world/source/regen.sh"
            );
            return;
        }
    };

    // 2. Encode through phasm in IBPBP shape (matches the x264
    //    reference encoded with `--bframes 1`).
    let pattern = GopPattern::Ibpbp { gop: 5, b_count: 1 };
    let stego_bytes = h264_stego_encode_yuv_string_4domain_multigop_with_pattern(
        &yuv, 1920, 1072, 10, pattern,
        /* message */ "x", "stealth-gate",
    )
    .expect("phasm IBPBP encode");

    let stego_path = std::env::temp_dir().join("phasm_6ea6_5_gate.h264");
    std::fs::write(&stego_path, &stego_bytes).expect("write stego h264");

    // 3. Run histogram script.
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let workspace_root = std::path::Path::new(manifest_dir)
        .parent()
        .expect("manifest dir has parent");
    let script = workspace_root.join("scripts/h264_mb_partition_histogram.py");
    if !script.exists() {
        panic!("missing histogram script: {script:?}");
    }

    let phasm_hist_text = match std::process::Command::new("python3")
        .arg(&script)
        .arg(&stego_path)
        .output()
    {
        Ok(out) if out.status.success() => String::from_utf8_lossy(&out.stdout).into_owned(),
        Ok(out) => panic!(
            "histogram script failed: status={:?}\nstdout={}\nstderr={}",
            out.status,
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr),
        ),
        Err(e) => {
            eprintln!("Skipping: python3 not available ({e})");
            return;
        }
    };

    eprintln!("---- phasm histogram ----\n{phasm_hist_text}\n");
    let phasm = parse_histogram(&phasm_hist_text);

    // 4. Load committed x264 reference.
    let x264_text = std::fs::read_to_string(
        std::path::Path::new(manifest_dir).join(X264_REFERENCE_PATH),
    )
    .expect("x264 reference histogram present");
    let x264 = parse_histogram(&x264_text);

    // 5. B-frame partition shape — SOFT warning (informational delta
    //    against x264 medium reference; #155 follow-up will calibrate
    //    `mb_decision_b_with_mvs` bucket distribution).
    eprintln!("---- B-frame partition shapes (soft compare vs x264 medium) ----");
    eprintln!(
        "phasm B: 16x16={:.2}% 8x8={:.2}% 16x8={:.2}% 8x16={:.2}% sub8x8={:.2}%",
        phasm.b.p16x16_pct, phasm.b.p8x8_pct, phasm.b.p16x8_pct,
        phasm.b.p8x16_pct, phasm.b.p_sub8x8_pct,
    );
    eprintln!(
        "x264  B: 16x16={:.2}% 8x8={:.2}% 16x8={:.2}% 8x16={:.2}% sub8x8={:.2}%",
        x264.b.p16x16_pct, x264.b.p8x8_pct, x264.b.p16x8_pct,
        x264.b.p8x16_pct, x264.b.p_sub8x8_pct,
    );
    let warn_delta = |label: &str, p: f64, x: f64| {
        let d = (p - x).abs();
        if d > EPSILON_PCT {
            eprintln!("  WARN {label}: phasm {p:.2}% vs x264 {x:.2}% (Δ={d:.2}pp > ε={EPSILON_PCT}pp) — Phase 6E-D / #158");
        }
    };
    warn_delta("16x16", phasm.b.p16x16_pct, x264.b.p16x16_pct);
    warn_delta("8x8",   phasm.b.p8x8_pct,   x264.b.p8x8_pct);
    warn_delta("16x8",  phasm.b.p16x8_pct,  x264.b.p16x8_pct);
    warn_delta("8x16",  phasm.b.p8x16_pct,  x264.b.p8x16_pct);

    // 6. Sub-8x8 partitions: x264-medium emits ZERO; phasm must too.
    //    Stricter than ε=5pp — emitting ANY would break the
    //    "look like x264 medium" target (only -preset slow does
    //    sub-8x8). See 6E-A6-bslice-partitions.md "Key finding".
    assert!(
        phasm.b.p_sub8x8_pct == 0.0,
        "B-frame sub-8x8 partitions: phasm emitted {:.4}%, x264 medium emits 0.00% — fail",
        phasm.b.p_sub8x8_pct,
    );

    // 7. B-frame direction (L0 / L1 / Bi) within ε=5pp.
    eprintln!(
        "phasm B dir: L0={:.2}% L1={:.2}% Bi={:.2}%",
        phasm.b.l0_pct, phasm.b.l1_pct, phasm.b.bi_pct,
    );
    eprintln!(
        "x264  B dir: L0={:.2}% L1={:.2}% Bi={:.2}%",
        x264.b.l0_pct, x264.b.l1_pct, x264.b.bi_pct,
    );
    assert_within_eps("B-frame L0", phasm.b.l0_pct, x264.b.l0_pct, EPSILON_PCT);
    assert_within_eps("B-frame L1", phasm.b.l1_pct, x264.b.l1_pct, EPSILON_PCT);
    assert_within_eps("B-frame Bi", phasm.b.bi_pct, x264.b.bi_pct, EPSILON_PCT);

    // 8. P-frame discipline: 100% L0, no Bi.
    eprintln!(
        "phasm P dir: L0={:.2}% L1={:.2}% Bi={:.2}%",
        phasm.p.l0_pct, phasm.p.l1_pct, phasm.p.bi_pct,
    );
    if phasm.p.n_mbs > 0 || phasm.p.l0_pct + phasm.p.l1_pct + phasm.p.bi_pct > 0.0 {
        assert!(
            phasm.p.l1_pct == 0.0 && phasm.p.bi_pct == 0.0,
            "P-frame must be 100% L0; got L0={:.2}% L1={:.2}% Bi={:.2}%",
            phasm.p.l0_pct, phasm.p.l1_pct, phasm.p.bi_pct,
        );
    }
    // P-frame sub-8x8 must also be zero.
    assert!(
        phasm.p.p_sub8x8_pct == 0.0,
        "P-frame sub-8x8 partitions: phasm emitted {:.4}%, must be 0",
        phasm.p.p_sub8x8_pct,
    );

    // 9. I-frame discipline: zero MVs.
    //    The histogram script counts intra MBs in `no-MV`. For an
    //    I-frame, that share should be 100% (or close — script's
    //    reading of MV side-data may miss intra-block-copy etc.).
    if phasm.i.no_mv_pct > 0.0 {
        assert!(
            phasm.i.no_mv_pct >= 99.0,
            "I-frame must be ≥99% no-MV; got {:.2}%",
            phasm.i.no_mv_pct,
        );
    }

    let _ = std::fs::remove_file(&stego_path);
}
