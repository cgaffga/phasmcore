// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `phasm video-encode --codec av1` keeps the source's frame times.
//!
//! The CLI used to hand the AV1 muxer nominal times (`i / fps`), so a source
//! with a gap — dropped frames, a variable-rate phone clip — came out
//! uniform: every frame after the gap played early against the audio, and
//! the file contradicted the libavformat transcode it presents as, which
//! keeps such a gap as a multi-slot delta. The apps pass the real times;
//! now the CLI does too.

#![cfg(feature = "av1-encoder")]

use std::path::{Path, PathBuf};
use std::process::Command;

const PASS: &str = "frame-times-pass";
const MSG: &str = "hello";

fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into());
    let dir = PathBuf::from(base).join(format!("phasm_cli_av1times_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Presentation times (s) of the first video stream's frames, as a player
/// decodes them.
fn frame_times(path: &Path) -> Vec<f64> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries", "frame=best_effort_timestamp_time"])
        .args(["-of", "csv=p=0"])
        .arg(path)
        .output()
        .unwrap();
    assert!(out.status.success(), "probe {}", path.display());
    String::from_utf8_lossy(&out.stdout).lines().filter_map(|l| l.trim().trim_end_matches(',').parse().ok()).collect()
}

fn phasm() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_phasm"));
    if let Ok(t) = std::env::var("PHASM_TMP") {
        c.env("TMPDIR", t);
    }
    c
}

/// 30 fps with frames 20–22 dropped and their gap kept: a variable-rate
/// source.
#[test]
fn a_source_gap_survives_the_encode() {
    let dir = scratch("gap");
    let src = dir.join("gap.mp4");
    let made = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2"])
        .args(["-vf", "select='not(between(n\\,20\\,22))'", "-fps_mode", "passthrough"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .arg(&src)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !made {
        eprintln!("skipping: could not build the fixture (ffmpeg / libx264 missing)");
        return;
    }
    let src_t = frame_times(&src);
    assert_eq!(src_t.len(), 57, "fixture frames");

    let out = dir.join("gap.av1.mp4");
    let enc = phasm()
        .arg("video-encode")
        .arg(&src)
        .args(["--codec", "av1", "-m", MSG, "-p", PASS, "-o"])
        .arg(&out)
        .output()
        .unwrap();
    assert!(enc.status.success(), "video-encode failed:\n{}", String::from_utf8_lossy(&enc.stderr));

    let out_t = frame_times(&out);
    assert_eq!(out_t.len(), src_t.len(), "frame count");
    let (s0, o0) = (src_t[0], out_t[0]);
    for (i, (s, o)) in src_t.iter().zip(&out_t).enumerate() {
        assert!(((o - o0) - (s - s0)).abs() < 0.5 / 30.0, "frame {i}: output {o} s, source {s} s");
    }

    let dec = phasm().arg("decode").arg(&out).args(["-p", PASS]).output().unwrap();
    assert!(
        dec.status.success() && String::from_utf8_lossy(&dec.stdout).contains(MSG),
        "decode did not recover the message:\n{}",
        String::from_utf8_lossy(&dec.stderr),
    );
    let _ = std::fs::remove_dir_all(&dir);
}
