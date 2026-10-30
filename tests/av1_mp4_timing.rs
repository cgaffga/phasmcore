// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The AV1 MP4: the file a libavformat transcode with the librav1e encoder
//! writes (`docs/design/video/av1/v2-gate-results.md`, "Container").
//!
//! Timing follows that tool: the video time base is `1/rate` with its
//! denominator doubled until it reaches 10000, the rate is the closer of an
//! integer and an NTSC rate to the source's mean frame period, and every
//! frame keeps its source time rounded to a whole frame slot (gaps survive
//! as multi-slot deltas, jitter under half a slot disappears). A video that
//! starts after the audio gets a leading empty edit of whole frames.

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

/// Temporal delimiter + sequence header + `n` frame OBUs (one per temporal
/// unit, a delimiter before each).
fn stream(n: usize) -> Vec<u8> {
    let mut s = Vec::new();
    for i in 0..n {
        s.extend_from_slice(&make_obu(2, 0));
        if i == 0 {
            s.extend_from_slice(&make_obu(1, 8));
        }
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
        c += size.max(8);
    }
    panic!("box {} not found", String::from_utf8_lossy(fourcc));
}

fn path(bytes: &[u8], boxes: &[&[u8; 4]]) -> (usize, usize) {
    boxes.iter().fold((0, bytes.len()), |r, b| child(bytes, r, b))
}

/// Child boxes of a range, in order, as fourccs.
fn kids(bytes: &[u8], range: (usize, usize)) -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    let mut c = range.0;
    while c + 8 <= range.1 {
        let size = u32::from_be_bytes(bytes[c..c + 4].try_into().unwrap()) as usize;
        out.push(bytes[c + 4..c + 8].try_into().unwrap());
        c += size.max(8);
    }
    out
}

fn u32_at(b: &[u8], at: usize) -> u32 {
    u32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

const VSTBL: &[&[u8; 4]] = &[b"moov", b"trak", b"mdia", b"minf", b"stbl"];

/// Video track `mdhd` → (timescale, duration).
fn mdhd(mp4: &[u8]) -> (u32, u64) {
    let (s, _) = path(mp4, &[b"moov", b"trak", b"mdia", b"mdhd"]);
    (u32_at(mp4, s + 12), u32_at(mp4, s + 16) as u64)
}

/// Video track `stts` → [(sample_count, delta)].
fn stts(mp4: &[u8]) -> Vec<(u32, u32)> {
    let (s, _) = path(mp4, &[VSTBL, &[b"stts"]].concat());
    let n = u32_at(mp4, s + 4) as usize;
    (0..n).map(|i| (u32_at(mp4, s + 8 + 8 * i), u32_at(mp4, s + 12 + 8 * i))).collect()
}

/// Video track `elst` → [(segment_duration_ms, media_time)].
fn elst(mp4: &[u8]) -> Vec<(u32, i32)> {
    let (s, _) = path(mp4, &[b"moov", b"trak", b"edts", b"elst"]);
    let n = u32_at(mp4, s + 4) as usize;
    (0..n).map(|i| (u32_at(mp4, s + 8 + 12 * i), u32_at(mp4, s + 12 + 12 * i) as i32)).collect()
}

/// The `av01` sample entry's content range.
fn av01(mp4: &[u8]) -> (usize, usize) {
    let (s, e) = path(mp4, &[VSTBL, &[b"stsd"]].concat());
    child(mp4, (s + 8, e), b"av01")
}

/// Frame `i` of a `num/den` fps clip, in µs, starting at `start_us`.
fn cfr_times(n: usize, num: i64, den: i64, start_us: i64) -> Vec<i64> {
    (0..n as i64).map(|i| start_us + (i * den * 1_000_000 + num / 2) / num).collect()
}

fn deltas(mp4: &[u8]) -> Vec<u32> {
    stts(mp4).iter().flat_map(|&(n, d)| std::iter::repeat(d).take(n as usize)).collect()
}

#[test]
fn ftyp_free_mdat_moov_with_the_av01_brand() {
    let mp4 = build_mp4_av1(&stream(10), 64, 64, TIMING_30).unwrap();
    let top = kids(&mp4, (0, mp4.len()));
    assert_eq!(top, [*b"ftyp", *b"free", *b"mdat", *b"moov"]);
    let (s, e) = path(&mp4, &[b"ftyp"]);
    assert_eq!(&mp4[s..e], b"isom\0\0\x02\0isomav01iso2mp41");
}

#[test]
fn sample_entry_is_librav1e() {
    let mp4 = build_mp4_av1(&stream(10), 64, 64, TIMING_30).unwrap();
    let (s, e) = av01(&mp4);
    // Compressor name: Pascal string at offset 42 of the sample entry.
    let name = &mp4[s + 43..s + 43 + mp4[s + 42] as usize];
    assert_eq!(name, b"Lavc62.28.101 librav1e");
    // Children after the 78-byte VisualSampleEntry: rav1e's own av1C (level
    // 31, no config OBUs), fiel (progressive), btrt — no colr for a sequence
    // header without a colour description.
    assert_eq!(kids(&mp4, (s + 78, e)), [*b"av1C", *b"fiel", *b"btrt"]);
    let (a, b) = child(&mp4, (s + 78, e), b"av1C");
    assert_eq!(&mp4[a..b], &[0x81, 0x1F, 0x0C, 0x00]);
    let (a, b) = child(&mp4, (s + 78, e), b"fiel");
    assert_eq!(&mp4[a..b], &[1, 0]);
    let (a, _) = child(&mp4, (s + 78, e), b"btrt");
    let (buffer, max, avg) = (u32_at(&mp4, a), u32_at(&mp4, a + 4), u32_at(&mp4, a + 8));
    assert_eq!((buffer, max), (0, avg));
    assert!(avg > 0);
}

#[test]
fn identifies_as_lavf_not_handbrake() {
    let mp4 = build_mp4_av1(&stream(10), 64, 64, TIMING_30).unwrap();
    let (s, e) = path(&mp4, &[b"moov", b"udta", b"meta"]);
    assert!(mp4[s..e].windows(13).any(|w| w == b"Lavf62.12.101"));
    assert!(!mp4.windows(9).any(|w| w == b"HandBrake"));
    // No sample dependency table (librav1e flags no disposable frames).
    let (s, e) = path(&mp4, VSTBL);
    assert!(!kids(&mp4, (s, e)).contains(b"sdtp"));
}

#[test]
fn samples_drop_delimiters_and_carry_the_sequence_header_on_sync_samples() {
    let s = stream(4);
    let mp4 = build_mp4_av1(&s, 64, 64, TIMING_30).unwrap();
    let (m, _) = path(&mp4, &[b"mdat"]);
    // First sample: sequence header OBU (type 1), then the frame OBU.
    assert_eq!(mp4[m] >> 3 & 0x0F, 1);
    assert!(!mp4.windows(2).any(|w| w == [0x12, 0x00]), "no temporal delimiter OBU");
    // stss lists only the first sample.
    let (a, _) = path(&mp4, &[VSTBL, &[b"stss"]].concat());
    assert_eq!((u32_at(&mp4, a + 4), u32_at(&mp4, a + 8)), (1, 1));
}

#[test]
fn integer_rate_is_the_lavf_time_base() {
    let mp4 = build_mp4_av1_timed(&stream(60), 64, 64, TIMING_30, &cfr_times(60, 30, 1, 0), None).unwrap();
    assert_eq!(mdhd(&mp4), (15360, 60 * 512));
    assert_eq!(stts(&mp4), vec![(60, 512)]);
    assert_eq!(elst(&mp4), vec![(2000, 0)]);
    assert_eq!(mp4, build_mp4_av1(&stream(60), 64, 64, TIMING_30).unwrap());
}

#[test]
fn ntsc_29_97_comes_from_the_frame_times() {
    // The apps pass an integer nominal rate; the frame times carry 29.97.
    let mp4 = build_mp4_av1_timed(&stream(60), 64, 64, TIMING_30, &cfr_times(60, 30000, 1001, 0), None).unwrap();
    assert_eq!(mdhd(&mp4), (30000, 60 * 1001));
    assert_eq!(stts(&mp4), vec![(60, 1001)]);
}

#[test]
fn ntsc_23_976_comes_from_the_frame_times() {
    let mp4 = build_mp4_av1_timed(&stream(48), 64, 64, TIMING_30, &cfr_times(48, 24000, 1001, 0), None).unwrap();
    assert_eq!(mdhd(&mp4), (24000, 48 * 1001));
    assert_eq!(stts(&mp4), vec![(48, 1001)]);
}

#[test]
fn lead_in_is_an_empty_edit_of_whole_frames() {
    // Video starts 0.125 s in at 29.97: 4 frames (0.1335 s) of empty edit.
    let mp4 =
        build_mp4_av1_timed(&stream(30), 64, 64, TIMING_30, &cfr_times(30, 30000, 1001, 125_000), None).unwrap();
    assert_eq!(stts(&mp4), vec![(30, 1001)]);
    assert_eq!(elst(&mp4), vec![(133, -1), (1001, 0)]);
}

#[test]
fn gaps_survive_as_whole_frame_deltas() {
    // 30 fps with frames 10..=12 missing from the source: one 4-slot delta.
    let times: Vec<i64> = cfr_times(40, 30, 1, 0).into_iter().enumerate().filter(|(i, _)| !(10..=12).contains(i)).map(|(_, t)| t).collect();
    let mp4 = build_mp4_av1_timed(&stream(37), 64, 64, TIMING_30, &times, None).unwrap();
    assert_eq!(mdhd(&mp4).0, 15360);
    assert_eq!(stts(&mp4), vec![(9, 512), (1, 2048), (27, 512)]);
}

#[test]
fn jitter_under_half_a_frame_rounds_away() {
    let mut times = cfr_times(30, 30, 1, 0);
    for (i, t) in times.iter_mut().enumerate() {
        if i % 7 == 3 {
            *t += 4_000;
        }
    }
    let mp4 = build_mp4_av1_timed(&stream(30), 64, 64, TIMING_30, &times, None).unwrap();
    assert_eq!(stts(&mp4), vec![(30, 512)]);
}

#[test]
fn colliding_times_take_the_next_slot() {
    let times = vec![0, 33_367, 33_367, 100_100, 90_000, 166_833];
    let mp4 = build_mp4_av1_timed(&stream(6), 64, 64, TIMING_30, &times, None).unwrap();
    let d = deltas(&mp4);
    assert_eq!(d.len(), 6);
    assert!(d.iter().all(|&x| x > 0 && x % d[d.len() - 1] == 0), "{d:?}");
}

#[test]
fn time_count_mismatch_falls_back_to_the_nominal_rate() {
    let s = stream(30);
    let timed = build_mp4_av1_timed(&s, 64, 64, TIMING_30, &cfr_times(29, 30000, 1001, 0), None).unwrap();
    assert_eq!(timed, build_mp4_av1(&s, 64, 64, TIMING_30).unwrap());
}

/// The app path: timed video plus the source's audio passed through.
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
    assert_eq!(mdhd(&mp4).0, 30000);
    let moov = path(&mp4, &[b"moov"]);
    let traks = kids(&mp4, moov).iter().filter(|k| k == &b"trak").count();
    assert_eq!(traks, 2, "video + passed-through audio");
}

/// A real demuxer reads the lead-in and the rate the same way.
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
    let lead = 4.0 * 1001.0 / 30000.0;
    assert!((pts[0] - lead).abs() < 2e-3, "{}", pts[0]);
    assert!((pts[29] - (lead + 29.0 * 1001.0 / 30000.0)).abs() < 2e-3, "{}", pts[29]);
}
