// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! AV1 MP4 video timing from the source's frame presentation times.
//!
//! The untimed muxer stamps every frame 512 ticks at `fps × 512` from an
//! integer frame rate: 29.97 / 23.976 sources ran 0.1 % fast against the
//! passed-through audio (~120 ms after two minutes), a variable-rate clip lost
//! its cadence, and a source whose video starts late (a leading empty edit)
//! lost its lead-in. `build_mp4_av1_timed` derives the video track's timing
//! from the source frame times instead: exact constant rates where the clip
//! is one, real cadence where it is not, and the lead-in kept by holding the
//! first frame (the muxer emits no edit list, like HandBrake).

#![cfg(feature = "video")]

use phasm_core::codec::mp4::build::{build_mp4_av1, build_mp4_av1_timed, FrameTiming};

fn make_obu(obu_type: u8, payload_len: usize) -> Vec<u8> {
    let mut out = vec![((obu_type & 0x0F) << 3) | 0b010];
    let mut n = payload_len as u64;
    loop {
        let mut byte = (n & 0x7F) as u8;
        n >>= 7;
        if n != 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if n == 0 {
            break;
        }
    }
    out.extend(std::iter::repeat(0u8).take(payload_len));
    out
}

/// Sequence header + `n` frame OBUs.
fn stream(n: usize) -> Vec<u8> {
    let mut s = make_obu(1, 8);
    for i in 0..n {
        s.extend_from_slice(&make_obu(6, 40 + i % 7));
    }
    s
}

const TIMING_30: FrameTiming = FrameTiming { fps_num: 30, fps_den: 1 };

/// Content range of the first box `fourcc` among `bytes[range]`'s children.
fn child(bytes: &[u8], range: (usize, usize), fourcc: &[u8; 4]) -> (usize, usize) {
    let mut c = range.0;
    while c + 8 <= range.1 {
        let size = u32::from_be_bytes(bytes[c..c + 4].try_into().unwrap()) as usize;
        if &bytes[c + 4..c + 8] == fourcc {
            return (c + 8, c + size);
        }
        c += size;
    }
    panic!("box {} not found", String::from_utf8_lossy(fourcc));
}

fn path(bytes: &[u8], boxes: &[&[u8; 4]]) -> (usize, usize) {
    boxes.iter().fold((0, bytes.len()), |r, b| child(bytes, r, b))
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

/// Video track `mdhd` → (timescale, duration).
fn mdhd(mp4: &[u8]) -> (u32, u64) {
    let (s, _) = path(mp4, &[b"moov", b"trak", b"mdia", b"mdhd"]);
    if mp4[s] == 1 {
        (u32_at(mp4, s + 20), u64::from_be_bytes(mp4[s + 24..s + 32].try_into().unwrap()))
    } else {
        (u32_at(mp4, s + 12), u32_at(mp4, s + 16) as u64)
    }
}

/// Video track `stts` → [(sample_count, delta)].
fn stts(mp4: &[u8]) -> Vec<(u32, u32)> {
    let (s, _) = path(mp4, &[b"moov", b"trak", b"mdia", b"minf", b"stbl", b"stts"]);
    let n = u32_at(mp4, s + 4) as usize;
    (0..n).map(|i| (u32_at(mp4, s + 8 + 8 * i), u32_at(mp4, s + 12 + 8 * i))).collect()
}

/// `mvhd` duration in its 1000/s timescale.
fn mvhd_ms(mp4: &[u8]) -> u64 {
    let (s, _) = path(mp4, &[b"moov", b"mvhd"]);
    u32_at(mp4, s + 16) as u64
}

/// Frame `i` of a `num/den` fps clip, in µs, starting at `start_us`.
fn cfr_times(n: usize, num: i64, den: i64, start_us: i64) -> Vec<i64> {
    (0..n as i64).map(|i| start_us + (i * den * 1_000_000 + num / 2) / num).collect()
}

#[test]
fn integer_rate_without_lead_in_is_byte_identical_to_untimed() {
    let s = stream(60);
    let timed = build_mp4_av1_timed(&s, 64, 64, TIMING_30, &cfr_times(60, 30, 1, 0), None).unwrap();
    assert_eq!(timed, build_mp4_av1(&s, 64, 64, TIMING_30).unwrap());
}

#[test]
fn ntsc_29_97_is_exact() {
    let mp4 = build_mp4_av1_timed(&stream(60), 64, 64, TIMING_30, &cfr_times(60, 30000, 1001, 0), None).unwrap();
    assert_eq!(mdhd(&mp4), (60000, 60 * 2002));
    assert_eq!(stts(&mp4), vec![(60, 2002)]);
    assert_eq!(mvhd_ms(&mp4), 2002);
}

#[test]
fn ntsc_23_976_is_exact() {
    let mp4 = build_mp4_av1_timed(&stream(48), 64, 64, TIMING_30, &cfr_times(48, 24000, 1001, 0), None).unwrap();
    assert_eq!(mdhd(&mp4).0, 48000);
    assert_eq!(stts(&mp4), vec![(48, 2002)]);
}

#[test]
fn lead_in_holds_the_first_frame() {
    // Video starts 0.125 s into the timeline (a leading empty edit): frame 0
    // is shown from 0, every later frame lands on its source time.
    let mp4 =
        build_mp4_av1_timed(&stream(30), 64, 64, TIMING_30, &cfr_times(30, 30000, 1001, 125_000), None).unwrap();
    assert_eq!(mdhd(&mp4).0, 60000);
    assert_eq!(stts(&mp4), vec![(1, 2002 + 7500), (29, 2002)]);
}

#[test]
fn variable_rate_keeps_the_source_cadence() {
    // Alternating 30 ms / 40 ms frames: no constant rate fits.
    let times: Vec<i64> = (0..20i64).map(|i| (i / 2) * 70_000 + (i % 2) * 30_000).collect();
    let mp4 = build_mp4_av1_timed(&stream(20), 64, 64, TIMING_30, &times, None).unwrap();
    assert_eq!(mdhd(&mp4).0, 90000);
    let deltas: Vec<u32> = stts(&mp4).iter().flat_map(|&(n, d)| std::iter::repeat(d).take(n as usize)).collect();
    assert_eq!(deltas.len(), 20);
    assert!(deltas[..19].iter().enumerate().all(|(i, &d)| d == if i % 2 == 0 { 2700 } else { 3600 }));
}

#[test]
fn non_increasing_times_are_made_strictly_increasing() {
    let times = vec![0, 33_367, 33_367, 100_100, 90_000, 166_833];
    let mp4 = build_mp4_av1_timed(&stream(6), 64, 64, TIMING_30, &times, None).unwrap();
    assert!(stts(&mp4).iter().all(|&(_, d)| d > 0));
}

#[test]
fn time_count_mismatch_falls_back_to_untimed() {
    let s = stream(30);
    let timed = build_mp4_av1_timed(&s, 64, 64, TIMING_30, &cfr_times(29, 30000, 1001, 0), None).unwrap();
    assert_eq!(timed, build_mp4_av1(&s, 64, 64, TIMING_30).unwrap());
}

/// The app path: timed video plus the source's audio passed through. The
/// video track is timed from the frame times and the audio track rides
/// along; the movie lasts as long as the longer of the two.
#[test]
fn timed_with_audio_keeps_both_tracks() {
    let src_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("test-vectors/video/h264/real-world/source/Artlist_SchoolFight.mp4");
    let Ok(source) = std::fs::read(&src_path) else {
        eprintln!("skipping: AAC source MP4 missing at {src_path:?}");
        return;
    };
    let times = cfr_times(30, 30000, 1001, 0);
    let mp4 = build_mp4_av1_timed(&stream(30), 64, 64, TIMING_30, &times, Some(&source)).unwrap();
    assert_eq!(mdhd(&mp4).0, 60000);
    assert_eq!(stts(&mp4), vec![(30, 2002)]);
    let moov = path(&mp4, &[b"moov"]);
    let traks = mp4[moov.0..moov.1].windows(4).filter(|w| w == b"trak").count();
    assert_eq!(traks, 2, "video + passed-through audio");
    assert!(mvhd_ms(&mp4) >= 1001, "movie covers at least the video");
}

/// A real demuxer reads the lead-in the same way: ffprobe's packet times
/// for the video track put frame 0 at 0 (held through the lead-in) and
/// frame 1 at its source time, 0.125 s + one 29.97 period.
#[test]
#[ignore = "needs ffprobe; run with --ignored"]
fn ffprobe_sees_the_lead_in_and_the_exact_rate() {
    let times = cfr_times(30, 30000, 1001, 125_000);
    let mp4 = build_mp4_av1_timed(&stream(30), 64, 64, TIMING_30, &times, None).unwrap();
    let dir = std::env::var("PHASM_TMP").unwrap_or_else(|_| "/tmp".into());
    let file = format!("{dir}/av1_mp4_timing_lead_in.mp4");
    std::fs::write(&file, &mp4).unwrap();
    let out = std::process::Command::new("ffprobe")
        .args(["-v", "error", "-select_streams", "v:0", "-show_entries"])
        .args(["packet=pts_time", "-of", "csv=p=0", &file])
        .output()
        .expect("ffprobe launch");
    let _ = std::fs::remove_file(&file);
    let pts: Vec<f64> = String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| l.trim().trim_end_matches(',').parse().ok())
        .collect();
    assert_eq!(pts.len(), 30, "ffprobe packets: {pts:?}");
    assert_eq!(pts[0], 0.0);
    assert!((pts[1] - (0.125 + 1001.0 / 30000.0)).abs() < 1e-4, "{}", pts[1]);
    assert!((pts[29] - (0.125 + 29.0 * 1001.0 / 30000.0)).abs() < 1e-4, "{}", pts[29]);
}
