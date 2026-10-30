// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `phasm video-encode --codec av1` signals the source's colour.
//!
//! The file presents as a libavformat transcode with the librav1e encoder,
//! which hands rav1e the source's colour tags (unspecified when it has
//! none) and writes `colr` only when all three are specified. The CLI's
//! decode keeps the source's pixels unconverted, so its tags stay true.

#![cfg(feature = "av1-encoder")]

use phasm_core::codec::mp4::av1_obu_split::parse_sequence_header_obu;
use phasm_core::codec::mp4::demux::demux;
use std::path::{Path, PathBuf};
use std::process::Command;

fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into());
    let dir = PathBuf::from(base).join(format!("phasm_cli_av1colour_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn phasm() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_phasm"));
    if let Ok(t) = std::env::var("PHASM_TMP") {
        c.env("TMPDIR", t);
    }
    c
}

/// A one-second test clip, with `tags` (extra encoder arguments) applied.
fn fixture(dir: &Path, name: &str, tags: &[&str]) -> Option<PathBuf> {
    let src = dir.join(name);
    Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=1"])
        .args(["-c:v", "libx264", "-pix_fmt", "yuv420p"])
        .args(tags)
        .arg(&src)
        .status()
        .ok()
        .filter(|s| s.success())
        .map(|_| src)
}

/// (sequence header colour description, full range, `colr` payload) of an
/// AV1 MP4.
fn colour_of(path: &Path) -> (Option<(u8, u8, u8)>, bool, Option<Vec<u8>>) {
    let bytes = std::fs::read(path).unwrap();
    let mp4 = demux(&bytes).expect("demux");
    let t = mp4.tracks.iter().find(|t| &t.handler_type == b"vide").expect("video track");
    let s0 = &t.samples[0];
    let sample = &bytes[s0.offset as usize..s0.offset as usize + s0.size as usize];
    // The sync sample starts with the sequence header OBU.
    let size = sample[1] as usize; // single-byte leb128 for a short header
    let sh = parse_sequence_header_obu(&sample[..2 + size]).expect("sequence header");
    let colr = bytes.windows(8).position(|w| &w[4..] == b"colr").map(|at| {
        let n = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap()) as usize;
        bytes[at + 8..at + n].to_vec()
    });
    (sh.color_description, sh.full_range, colr)
}

fn encode(src: &Path, out: &Path) {
    let enc = phasm()
        .arg("video-encode")
        .arg(src)
        .args(["--codec", "av1", "-m", "hello", "-p", "colour-pass", "-o"])
        .arg(out)
        .output()
        .unwrap();
    assert!(enc.status.success(), "video-encode failed:\n{}", String::from_utf8_lossy(&enc.stderr));
}

#[test]
fn hlg_bt2020_tags_pass_through() {
    let dir = scratch("hlg");
    let Some(src) = fixture(&dir, "hlg.mp4", &["-vf", "setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc"]) else {
        eprintln!("skipping: could not build the fixture (ffmpeg / libx264 missing)");
        return;
    };
    let out = dir.join("hlg.av1.mp4");
    encode(&src, &out);
    let (desc, full, colr) = colour_of(&out);
    assert_eq!(desc, Some((9, 18, 9)));
    assert!(!full);
    assert_eq!(colr.as_deref(), Some(&b"nclx\0\x09\0\x12\0\x09\0"[..]));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn an_untagged_source_signals_unspecified_without_colr() {
    let dir = scratch("untagged");
    let Some(src) = fixture(&dir, "plain.mp4", &[]) else {
        eprintln!("skipping: could not build the fixture (ffmpeg / libx264 missing)");
        return;
    };
    let out = dir.join("plain.av1.mp4");
    encode(&src, &out);
    let (desc, full, colr) = colour_of(&out);
    assert_eq!(desc, Some((2, 2, 2)), "the wrapper always passes a description");
    assert!(!full);
    assert_eq!(colr, None);
    let _ = std::fs::remove_dir_all(&dir);
}
