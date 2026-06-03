// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! TG-5 — Force-mode bisect harness.
//!
//! Parametric encoder-knob × fixture matrix run as parallel
//! `#[test]` entries. Each test takes a fixture, encodes it with the
//! knob's forced configuration applied, runs the full embed → extract
//! round-trip, asserts payload survives, and prints diagnostic
//! metrics. The test PASSES if it ran to completion (no encoder
//! panic) AND the round-trip survives.
//!
//! ## Why this exists
//!
//! The diagnostic value comes from comparing knob-by-knob metrics
//! when a future encoder bug surfaces — the harness lets you ablate
//! one variable at a time without manual env-var dance. H.264's
//! V14-V24 wall (twelve mode-decision iterations chasing the wrong
//! variable) is exactly what TG-5 prevents.
//!
//! Per `[[feedback_root_cause_first_on_any_bug]]`, the controls
//! ("force-known-clean" knobs) anchor the bisect: if every variant
//! moves the metric modestly but a control variant ALSO moves, the
//! bug is downstream of mode decision. If only one knob is dirty,
//! that's the suspect.
//!
//! ## v0.5 knob set
//!
//! AV1 today (single-keyframe per encode) doesn't have inter mode
//! decision to ablate, so the v0.5 knob set focuses on the in-frame
//! encoder choices rav1e exposes via `SpeedSettings` + base config:
//!
//!  - `Default` — control, baseline
//!  - `Speed0` — slowest encoder preset (max effort, all RD on)
//!  - `Speed10` — fastest encoder preset (min effort)
//!  - `CdefOff` — disable CDEF post-filter
//!  - `LrfOff` — disable loop-restoration post-filter
//!  - `LowQuant` — qindex=10 (high quality, dense AC)
//!  - `HighQuant` — qindex=50 (low quality, sparse AC)
//!
//! v0.6+ adds inter-mode knobs (KeyframeOnly vs IPPPP vs IBPBP) once
//! AV1 ships multi-GOP via [`phase-c-streaming-session-v6.md`].
//!
//! See [`phase-c-test-gates.md`](../../docs/design/video/av1/phase-c-test-gates.md)
//! § 5 for the full design rationale.

#![cfg(all(feature = "av1-encoder", feature = "av1-backend"))]

use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

use phasm_core::codec::av1::stego::orchestrator::{av1_stego_embed, av1_stego_extract};
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::config::SpeedSettings;
use phasm_rav1e::phasm_stego::{
    encode_frame_with_phasm_tee, make_frame, make_inter_config, FrameInvariants, FrameState,
};
use phasm_rav1e::prelude::Sequence;
use phasm_rav1e::EncoderConfig;

#[derive(Debug, Clone, Copy)]
enum ForceMode {
    Default,
    Speed0,
    Speed10,
    CdefOff,
    LrfOff,
    LowQuant,
    HighQuant,
}

impl ForceMode {
    fn label(&self) -> &'static str {
        match self {
            ForceMode::Default => "default",
            ForceMode::Speed0 => "speed0",
            ForceMode::Speed10 => "speed10",
            ForceMode::CdefOff => "cdef_off",
            ForceMode::LrfOff => "lrf_off",
            ForceMode::LowQuant => "low_quant",
            ForceMode::HighQuant => "high_quant",
        }
    }

    /// Whether this knob is a "force-known-clean" control. Controls
    /// SHOULD produce baseline-level metrics; if a control breaks,
    /// the bug is downstream of every knob and the bisect failed —
    /// look at the codec layer, not mode decision.
    fn is_control(&self) -> bool {
        matches!(self, ForceMode::Default)
    }

    fn apply(&self, config: &mut EncoderConfig) {
        match self {
            ForceMode::Default => {}
            ForceMode::Speed0 => {
                config.speed_settings = SpeedSettings::from_preset(0);
            }
            ForceMode::Speed10 => {
                config.speed_settings = SpeedSettings::from_preset(10);
            }
            ForceMode::CdefOff => {
                config.speed_settings.cdef = false;
            }
            ForceMode::LrfOff => {
                config.speed_settings.lrf = false;
            }
            ForceMode::LowQuant => {
                config.quantizer = 10;
            }
            ForceMode::HighQuant => {
                config.quantizer = 50;
            }
        }
    }
}

struct Fixture {
    name: &'static str,
    source: &'static str,
    width: u32,
    height: u32,
    seek_s: f32,
    quantizer: usize,
}

const FIXTURES: &[Fixture] = &[
    Fixture {
        name: "iphone_img4138",
        source: "IMG_4138.MOV",
        width: 256,
        height: 144,
        seek_s: 1.0,
        quantizer: 30,
    },
    Fixture {
        name: "carplane",
        source: "Artlist_CarPlane.mp4",
        width: 144,
        height: 256,
        seek_s: 0.5,
        quantizer: 30,
    },
];

fn corpus_root() -> PathBuf {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("test-vectors/video/h264/real-world/source");
    p
}

fn extract_yuv420_frame(spec: &Fixture) -> Vec<u8> {
    let src = corpus_root().join(spec.source);
    assert!(src.exists(), "corpus fixture missing: {}", src.display());
    let vf = format!(
        "scale={}:{}:force_original_aspect_ratio=disable",
        spec.width, spec.height
    );
    let out = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-ss"])
        .arg(spec.seek_s.to_string())
        .args(["-i"])
        .arg(&src)
        .args([
            "-frames:v", "1", "-vf", &vf, "-pix_fmt", "yuv420p", "-f", "rawvideo", "-",
        ])
        .output()
        .expect("ffmpeg launch");
    assert!(out.status.success(), "ffmpeg yuv extract failed");
    let expected = (spec.width * spec.height * 3 / 2) as usize;
    assert_eq!(out.stdout.len(), expected);
    out.stdout
}

fn run_bisect(spec: &Fixture, knob: ForceMode) {
    // Force deterministic crypto for reproducible per-knob metrics —
    // same seed pattern as TG-2 (see `av1_gop_boundary_probe.rs`).
    // SAFETY: each #[test] runs in its own thread, but cargo test
    // serializes within the same binary; env is process-wide and
    // every test sets the same value.
    unsafe {
        std::env::set_var("PHASM_DETERMINISTIC_SEED", "20260603");
    }

    let yuv = extract_yuv420_frame(spec);

    let mut config = EncoderConfig {
        width: spec.width as usize,
        height: spec.height as usize,
        bit_depth: 8,
        chroma_sampling: ChromaSampling::Cs420,
        quantizer: spec.quantizer,
        ..Default::default()
    };
    knob.apply(&mut config);
    let config = Arc::new(config);

    let mut sequence = Sequence::new(&config);
    sequence.enable_large_lru = false;
    let mut fi = FrameInvariants::<u8>::new_key_frame(
        config.clone(),
        Arc::new(sequence),
        0,
        Box::new([]),
    );
    fi.enable_segmentation = false;

    let mut frame =
        make_frame::<u8>(spec.width as usize, spec.height as usize, ChromaSampling::Cs420);
    let w = spec.width as usize;
    let h = spec.height as usize;
    let y_size = w * h;
    let uv_size = (w / 2) * (h / 2);
    frame.planes[0].copy_from_raw_u8(&yuv[..y_size], w, 1);
    frame.planes[1].copy_from_raw_u8(&yuv[y_size..y_size + uv_size], w / 2, 1);
    frame.planes[2].copy_from_raw_u8(
        &yuv[y_size + uv_size..y_size + 2 * uv_size],
        w / 2,
        1,
    );

    let mut fs = FrameState::new_with_frame(&fi, Arc::new(frame));
    let inter_cfg = make_inter_config(&config);
    let (natural_packet, recording) = encode_frame_with_phasm_tee(&fi, &mut fs, &inter_cfg);

    let message = b"TG-5 force-mode bisect";
    let passphrase = "tg5-bisect-2026-06-03";

    let cover_bits = recording.tiles.first().map_or(0, |t| t.bit_positions.len());
    let stego_packet =
        av1_stego_embed(natural_packet.clone(), recording, message, passphrase).unwrap_or_else(
            |e| {
                panic!(
                    "[TG-5] {} × {}: av1_stego_embed failed: {:?}",
                    spec.name,
                    knob.label(),
                    e
                )
            },
        );

    let extracted = av1_stego_extract(&stego_packet, passphrase).unwrap_or_else(|e| {
        panic!(
            "[TG-5] {} × {}: av1_stego_extract failed: {:?}",
            spec.name,
            knob.label(),
            e
        )
    });
    assert_eq!(
        extracted.as_slice(),
        message,
        "[TG-5] {} × {}: round-trip payload mismatch",
        spec.name,
        knob.label()
    );

    let len_delta = stego_packet.len() as i64 - natural_packet.len() as i64;
    eprintln!(
        "[TG-5] {} × {} (control={}): natural_bytes={} stego_bytes={} Δ={:+} cover_bits={}",
        spec.name,
        knob.label(),
        knob.is_control(),
        natural_packet.len(),
        stego_packet.len(),
        len_delta,
        cover_bits
    );
}

// Generate one #[test] per (fixture × knob). Cargo runs them in
// parallel; per-test panics isolate to that knob's row in the matrix.
macro_rules! bisect_test {
    ($fn_name:ident, $fixture_idx:expr, $knob:expr) => {
        #[test]
        fn $fn_name() {
            run_bisect(&FIXTURES[$fixture_idx], $knob);
        }
    };
}

// iphone_img4138 × 7 knobs
bisect_test!(iphone_img4138_default, 0, ForceMode::Default);
bisect_test!(iphone_img4138_speed0, 0, ForceMode::Speed0);
bisect_test!(iphone_img4138_speed10, 0, ForceMode::Speed10);
bisect_test!(iphone_img4138_cdef_off, 0, ForceMode::CdefOff);
bisect_test!(iphone_img4138_lrf_off, 0, ForceMode::LrfOff);
bisect_test!(iphone_img4138_low_quant, 0, ForceMode::LowQuant);
bisect_test!(iphone_img4138_high_quant, 0, ForceMode::HighQuant);

// carplane × 7 knobs
bisect_test!(carplane_default, 1, ForceMode::Default);
bisect_test!(carplane_speed0, 1, ForceMode::Speed0);
bisect_test!(carplane_speed10, 1, ForceMode::Speed10);
bisect_test!(carplane_cdef_off, 1, ForceMode::CdefOff);
bisect_test!(carplane_lrf_off, 1, ForceMode::LrfOff);
bisect_test!(carplane_low_quant, 1, ForceMode::LowQuant);
bisect_test!(carplane_high_quant, 1, ForceMode::HighQuant);
