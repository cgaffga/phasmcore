// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! `phasm video-encode` must write exactly the source's frames, with the
//! source's video duration.
//!
//! Regression (2026-10-25): a 121-frame clip (Artlist_HorseFlag, 24 fps) came
//! out with 122 frames. The CLI's raw-YUV decode resampled to a constant
//! rate anchored at the file's start, so a video track starting after the
//! audio track had its first frame repeated to fill the gap. Every clip in
//! the corpus with that layout was affected (CarPlane 193 -> 194, SchoolFight
//! 361 -> 362, ...); clips whose video starts with the file were not.
//!
//! The duplicate also stood in, one frame coarse, for the late video's
//! lead-in against the audio. Without it the lead-in is carried by holding
//! frame 0 for it (`build_mp4_with_pattern_audio_lead_in`), so these tests
//! also pin the video's placement against the passed-through audio.
//!
//! Each case builds its own fixture, runs the real binary, then checks the
//! output's video sample count, decoded frame count and timing against the
//! source's, and that the message still decodes. Covers the non-shadow and
//! the shadow encode paths, which derive their frame counts separately.

#![cfg(feature = "h264-encoder")]

use phasm_core::codec::mp4::demux::demux;
use std::path::{Path, PathBuf};
use std::process::Command;

const PASS: &str = "frame-count-pass";
const MSG: &str = "hello";

fn scratch(tag: &str) -> PathBuf {
    let base = std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into());
    let dir = PathBuf::from(base).join(format!("phasm_cli_framecount_{tag}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// Build a fixture with the source-generating tool. `None` when it (or
/// libx264) is unavailable, in which case the test is skipped like the other
/// tool-dependent CLI tests.
fn make_fixture(out: &Path, args: &[&str]) -> Option<()> {
    let ok = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error"])
        .args(args)
        .arg(out)
        .status()
        .ok()?;
    ok.success().then_some(())
}

/// Video sample count of the first video track, from phasm's own demuxer.
fn video_samples(path: &Path) -> usize {
    let bytes = std::fs::read(path).unwrap();
    let mp4 = demux(&bytes).unwrap_or_else(|e| panic!("demux {}: {e:?}", path.display()));
    mp4.tracks
        .iter()
        .find(|t| &t.handler_type == b"vide")
        .unwrap_or_else(|| panic!("{}: no video track", path.display()))
        .samples
        .len()
}

/// Player-side view of one stream, independent of phasm's demuxer.
struct Probed {
    /// Decoded frame count (video only).
    frames: Option<usize>,
    /// When the stream's first frame / sample is presented, in seconds.
    start: f64,
    /// Presentation duration in seconds. Edit-list aware, unlike `mdhd` (a
    /// B-frame source's `mdhd` also counts the composition delay its edit
    /// list hides).
    duration: f64,
}

/// Probe the first stream of `kind` ("v" or "a"); `None` if there is none.
fn probed(path: &Path, kind: &str) -> Option<Probed> {
    let out = Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", &format!("{kind}:0"), "-count_frames",
               "-show_entries", "stream=nb_read_frames,start_time,duration",
               "-of", "default=noprint_wrappers=1"])
        .arg(path)
        .output()
        .unwrap();
    assert!(out.status.success(), "probe {}", path.display());
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    if text.trim().is_empty() {
        return None;
    }
    let field = |k: &str| -> String {
        text.lines()
            .find_map(|l| l.strip_prefix(k))
            .unwrap_or_else(|| panic!("{}: no {k} in {text}", path.display()))
            .trim()
            .to_string()
    };
    Some(Probed {
        frames: field("nb_read_frames=").parse().ok(),
        start: field("start_time=").parse().unwrap(),
        duration: field("duration=").parse().unwrap(),
    })
}

fn phasm() -> Command {
    let mut c = Command::new(env!("CARGO_BIN_EXE_phasm"));
    if let Ok(t) = std::env::var("PHASM_TMP") {
        c.env("TMPDIR", t);
    }
    c
}

fn encode_and_check(src: &Path, out: &Path, extra: &[&str]) {
    let enc = phasm()
        .arg("video-encode")
        .arg(src)
        .args(["-m", MSG, "-p", PASS, "-o"])
        .arg(out)
        .args(extra)
        .output()
        .unwrap();
    assert!(
        enc.status.success(),
        "video-encode {} failed:\n{}",
        src.display(),
        String::from_utf8_lossy(&enc.stderr),
    );

    let (src_n, out_n) = (video_samples(src), video_samples(out));
    assert_eq!(
        out_n, src_n,
        "{}: output has {out_n} video samples, source has {src_n}",
        src.display(),
    );
    let sv = probed(src, "v").unwrap();
    let ov = probed(out, "v").unwrap();
    assert_eq!(
        ov.frames, sv.frames,
        "{}: output decodes to {:?} frames, source to {:?}",
        src.display(), ov.frames, sv.frames,
    );
    // The output video always starts at 0 (no edit list, like x264).
    assert!(ov.start.abs() < 0.001, "{}: output video starts at {}", out.display(), ov.start);

    match (probed(src, "a"), probed(out, "a")) {
        (Some(sa), Some(oa)) => {
            // The audio keeps its placement, so the video must keep its own
            // against it: the source's lead-in is held on frame 0 and the
            // video ends exactly where the source's does.
            assert!(
                (oa.start - sa.start).abs() < 0.001,
                "{}: audio starts at {} s, source {} s", out.display(), oa.start, sa.start,
            );
            let (src_end, out_end) = (sv.start + sv.duration, ov.duration);
            assert!(
                (out_end - src_end).abs() < 0.001,
                "{}: output video ends at {out_end} s, source at {src_end} s \
                 (source video starts {} s in)",
                src.display(), sv.start,
            );
        }
        (None, None) => assert!(
            (ov.duration - sv.duration).abs() < 0.001,
            "{}: output video lasts {} s, source {} s",
            src.display(), ov.duration, sv.duration,
        ),
        (s, o) => panic!(
            "{}: audio present in source={} output={}",
            src.display(), s.is_some(), o.is_some(),
        ),
    }

    let dec = phasm().arg("decode").arg(out).args(["-p", PASS]).output().unwrap();
    assert!(
        dec.status.success() && String::from_utf8_lossy(&dec.stdout).contains(MSG),
        "{}: decode did not recover the message:\n{}",
        out.display(),
        String::from_utf8_lossy(&dec.stderr),
    );
}

/// The reported bug's layout: audio from 0, video starting later (here
/// ~0.083 s, two frame slots at 24 fps). The old decode emitted 50 frames
/// for these 48. Two seconds, so the shadow path (which needs at least one
/// whole 30-frame GOP) takes it too.
fn late_video_fixture(dir: &Path) -> Option<PathBuf> {
    let src = dir.join("late_video.mp4");
    make_fixture(&src, &[
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=2.3",
        "-itsoffset", "0.1",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=24:duration=2",
        "-map", "1:v", "-map", "0:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
    ])?;
    Some(src)
}

/// Video only, 37 frames: not a multiple of the 30-frame GOP, so the last
/// GOP is partial.
fn partial_gop_fixture(dir: &Path) -> Option<PathBuf> {
    let src = dir.join("partial_gop.mp4");
    make_fixture(&src, &[
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30",
        "-frames:v", "37",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
    ])?;
    Some(src)
}

fn run_case(tag: &str, fixture: fn(&Path) -> Option<PathBuf>, extra: &[&str]) {
    let dir = scratch(tag);
    let Some(src) = fixture(&dir) else {
        eprintln!("skipping: fixture could not be built (tool or libx264 unavailable)");
        let _ = std::fs::remove_dir_all(&dir);
        return;
    };
    encode_and_check(&src, &dir.join("stego.mp4"), extra);
    let _ = std::fs::remove_dir_all(&dir);
}

const SHADOW: &[&str] = &["--shadow-message-2", "decoy", "--shadow-passphrase-2", "decoy-pass"];

#[test]
fn late_video_with_audio_keeps_frame_count() {
    run_case("late", late_video_fixture, &[]);
}

#[test]
fn late_video_with_audio_keeps_frame_count_shadow() {
    run_case("late_shadow", late_video_fixture, SHADOW);
}

#[test]
fn partial_last_gop_keeps_frame_count() {
    run_case("partial", partial_gop_fixture, &[]);
}

#[test]
fn partial_last_gop_keeps_frame_count_shadow() {
    run_case("partial_shadow", partial_gop_fixture, SHADOW);
}
