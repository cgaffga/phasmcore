// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! A stego MP4 built by phasm's muxer carries none of its source file's
//! identity — in particular not through the audio track it copies.
//!
//! An iPhone `.MOV` stamps every track header with its capture time, to the
//! second, and names Apple's framework in the handlers. The passed-through
//! audio track used to keep both, linking the stego video to the original
//! clip in the sender's library. Here real camera files go through the AV1
//! builder and (with `h264-encoder`) the H.264 builder, and the output is
//! walked box by box:
//!
//! - every `mvhd` / `tkhd` / `mdhd` creation and modification time is 0;
//! - no track carries `udta` / `meta` / `uuid` / `free` / `skip`, and the
//!   movie's only user data is the muxer's own `©too`;
//! - every `hdlr` is a neutral ISO media handler (`VideoHandler` /
//!   `SoundHandler`), every data reference self-contained;
//! - none of the source's header times, nor its writer's names, appear
//!   anywhere in `moov`.
//!
//! And nothing the audio needs may change: the same sample bytes, sample
//! tables, sample groups and codec configuration as the source; nor the
//! video: its samples and sample tables are those of a video-only mux.
//!
//! The fixtures live in the gitignored test-vector tree; a missing one is
//! skipped.

#![cfg(feature = "video")]

use std::path::PathBuf;

use phasm_core::codec::mp4::Track;
use phasm_core::codec::mp4::build::{FrameTiming, build_mp4_av1, build_mp4_av1_with_audio};
use phasm_core::codec::mp4::demux::demux;

/// Real sources with an audio track: four iPhone `.MOV`s (QuickTime audio
/// tracks with capture times) and one MP4 written by a desktop tool.
fn fixtures() -> Vec<(&'static str, Vec<u8>)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test-vectors/video/h264/real-world/source");
    let names = [
        "iphone7_1080p_30fps_h264_high.mov",
        "iphone5_1080p_30fps_h264_high.mov",
        "IMG_4138.MOV",
        "IMG_4273.MOV",
        "Artlist_CarPlane.mp4",
    ];
    let mut out = Vec::new();
    for name in names {
        match std::fs::read(dir.join(name)) {
            Ok(bytes) => out.push((name, bytes)),
            Err(_) => eprintln!("skipping {name}: not in the test-vector tree"),
        }
    }
    out
}

// ─── A box walker ───────────────────────────────────────────────────────

struct Found<'a> {
    /// Slash-separated box types from the file root, e.g. `moov/trak/tkhd`.
    path: String,
    parent: [u8; 4],
    typ: [u8; 4],
    payload: &'a [u8],
}

fn fourcc(t: &[u8; 4]) -> String {
    t.iter().map(|&b| if b.is_ascii_graphic() || b == b' ' { b as char } else { '?' }).collect()
}

/// Where a box's children start inside its payload, if it has any.
fn children_at(typ: &[u8; 4], payload: &[u8], parent: &[u8; 4]) -> Option<usize> {
    match typ {
        b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"edts" | b"dinf" | b"udta" | b"tref" | b"wave" => Some(0),
        // ISO `meta` is a full box; QuickTime's (an iPhone's `moov/meta`)
        // starts with its `hdlr` straight away.
        b"meta" => Some(if payload.get(4..8) == Some(b"hdlr") { 0 } else { 4 }),
        b"stsd" | b"dref" => Some(8),
        _ if parent == b"stsd" => {
            // A sample entry: audio (28 bytes + the QuickTime v1 / v2
            // extension) or visual (78 bytes).
            if matches!(typ, b"avc1" | b"av01" | b"hvc1" | b"hev1") {
                return Some(78);
            }
            let version = u16::from_be_bytes([*payload.get(8)?, *payload.get(9)?]);
            Some(
                28 + match version {
                    1 => 16,
                    2 => 36,
                    _ => 0,
                },
            )
        }
        _ => None,
    }
}

/// Every box in `buf`, parents before children. `strict`: a box that does
/// not fit is a test failure (phasm's output); otherwise the walk of that
/// container stops there (a camera file's vendor sample entries).
fn walk<'a>(buf: &'a [u8], parent: &[u8; 4], path: &str, strict: bool, out: &mut Vec<Found<'a>>) {
    let mut off = 0usize;
    while buf.len() - off >= 8 {
        let size32 = u32::from_be_bytes(buf[off..off + 4].try_into().unwrap()) as usize;
        let typ: [u8; 4] = buf[off + 4..off + 8].try_into().unwrap();
        let (header, size) = match size32 {
            0 => (8, buf.len() - off),
            1 => (16, u64::from_be_bytes(buf[off + 8..off + 16].try_into().unwrap()) as usize),
            n => (8, n),
        };
        if size < header || size > buf.len() - off {
            assert!(!strict, "bad box {} at {path}", fourcc(&typ));
            return;
        }
        let payload = &buf[off + header..off + size];
        let here = if path.is_empty() { fourcc(&typ) } else { format!("{path}/{}", fourcc(&typ)) };
        out.push(Found { path: here.clone(), parent: *parent, typ, payload });
        if let Some(at) = children_at(&typ, payload, parent)
            && at <= payload.len()
        {
            walk(&payload[at..], &typ, &here, strict, out);
        }
        off += size;
    }
}

/// Every box of a phasm-built file (or of one track of it).
fn all_boxes(file: &[u8]) -> Vec<Found<'_>> {
    let mut out = Vec::new();
    walk(file, b"    ", "", true, &mut out);
    out
}

/// Every box of a camera file that can be walked.
fn source_boxes(file: &[u8]) -> Vec<Found<'_>> {
    let mut out = Vec::new();
    walk(file, b"    ", "", false, &mut out);
    out
}

/// `(creation, modification)` of an `mvhd` / `tkhd` / `mdhd` payload.
fn header_times(payload: &[u8]) -> (u64, u64) {
    if payload[0] == 1 {
        (
            u64::from_be_bytes(payload[4..12].try_into().unwrap()),
            u64::from_be_bytes(payload[12..20].try_into().unwrap()),
        )
    } else {
        (
            u32::from_be_bytes(payload[4..8].try_into().unwrap()) as u64,
            u32::from_be_bytes(payload[8..12].try_into().unwrap()) as u64,
        )
    }
}

fn moov(file: &[u8]) -> &[u8] {
    all_boxes(file).into_iter().find(|b| b.path == "moov").expect("moov").payload
}

fn contains(hay: &[u8], needle: &[u8]) -> bool {
    hay.windows(needle.len()).any(|w| w == needle)
}

// ─── The checks ─────────────────────────────────────────────────────────

/// No field of `mp4` carries anything of `source`'s identity.
fn assert_no_source_identity(name: &str, mp4: &[u8], source: &[u8]) {
    let boxes = all_boxes(mp4);
    let mut tracks = 0;
    for b in &boxes {
        let what = format!("{name}: {}", b.path);
        match &b.typ {
            b"mvhd" | b"tkhd" | b"mdhd" => {
                assert_eq!(header_times(b.payload), (0, 0), "{what} creation / modification time");
                tracks += (&b.typ == b"tkhd") as usize;
            }
            b"udta" => {
                assert_eq!(b.path, "moov/udta", "{what}: user data inside a track");
                let kids: Vec<&Found<'_>> = boxes.iter().filter(|k| k.path.starts_with("moov/udta/")).collect();
                assert_eq!(kids.len(), 1, "{what}: more than the muxer's own tag");
                assert_eq!(&kids[0].typ, b"\xa9too", "{what}");
            }
            b"meta" | b"uuid" | b"free" | b"skip" | b"XMP_" | b"loci" | b"keys" | b"ilst" | b"\xa9day" | b"\xa9xyz"
            | b"\xa9mak" | b"\xa9mod" | b"\xa9swr" | b"name" => panic!("{what}: a metadata box"),
            b"hdlr" => {
                assert!(b.path.ends_with("/mdia/hdlr"), "{what}: a handler outside mdia (QuickTime data handler?)");
                assert_eq!(&b.payload[..8], &[0u8; 8], "{what}: version/flags or pre_defined (QuickTime mhlr?)");
                assert_eq!(&b.payload[12..24], &[0u8; 12], "{what}: reserved (QuickTime manufacturer?)");
                let expected: &[u8] = match &b.payload[8..12] {
                    b"vide" => b"VideoHandler\0",
                    b"soun" => b"SoundHandler\0",
                    other => panic!("{what}: unexpected handler type {other:?}"),
                };
                assert_eq!(&b.payload[24..], expected, "{what}: handler name");
            }
            b"dref" => {
                let mut entries = Vec::new();
                walk(&b.payload[8..], b"dref", &b.path, true, &mut entries);
                let declared = u32::from_be_bytes(b.payload[4..8].try_into().unwrap()) as usize;
                assert!(declared >= 1 && entries.len() == declared, "{what}: entry count");
                for e in entries {
                    assert_eq!(&e.typ, b"url ", "{what}: {}", e.path);
                    assert_eq!(e.payload, &[0, 0, 0, 1], "{what}: a reference that is not self-contained");
                }
            }
            _ => {}
        }
        if &b.parent == b"stsd" && is_sound_entry(&b.typ) {
            assert_eq!(&b.payload[12..16], &[0u8; 4], "{what}: sound description vendor code");
        }
    }
    assert!(tracks >= 2, "{name}: expected a video and an audio track");

    // Byte-level: nothing of the source's header times or writer names
    // anywhere in the output's movie box.
    let moov = moov(mp4);
    for needle in [&b"Core Media"[..], b"appl", b"Apple", b"iPhone", b"com.apple", b"Lavf"] {
        assert!(!contains(moov, needle), "{name}: {:?} in moov", String::from_utf8_lossy(needle));
    }
    let mut source_times: Vec<u64> = source_boxes(source)
        .iter()
        .filter(|b| matches!(&b.typ, b"mvhd" | b"tkhd" | b"mdhd"))
        .flat_map(|b| {
            let (c, m) = header_times(b.payload);
            [c, m]
        })
        .filter(|&t| t != 0)
        .collect();
    source_times.sort_unstable();
    source_times.dedup();
    for t in source_times {
        assert!(!contains(moov, &t.to_be_bytes()), "{name}: source time {t} (64-bit) in moov");
        if let Ok(t32) = u32::try_from(t) {
            assert!(!contains(moov, &t32.to_be_bytes()), "{name}: source time {t} in moov");
        }
    }
}

fn is_sound_entry(typ: &[u8; 4]) -> bool {
    matches!(typ, b"mp4a" | b"Opus" | b"ac-3" | b"ec-3" | b"alac" | b"lpcm" | b"sowt" | b"twos" | b"fLaC" | b"apac")
}

fn first_audio(file: &[u8]) -> Track {
    let parsed = demux(file).expect("demux");
    parsed.tracks.into_iter().find(|t| &t.handler_type == b"soun").expect("an audio track")
}

/// The payload of the first box of type `typ` anywhere in `trak` (sample
/// entries and QuickTime `wave` atoms included).
fn trak_box<'a>(trak: &'a [u8], typ: &[u8; 4]) -> Option<&'a [u8]> {
    source_boxes(trak).into_iter().find(|b| &b.typ == typ).map(|b| b.payload)
}

fn sample_bytes<'a>(file: &'a [u8], t: &Track) -> Vec<&'a [u8]> {
    t.samples.iter().map(|s| &file[s.offset as usize..s.offset as usize + s.size as usize]).collect()
}

/// The audio comes through untouched: samples, sample tables, sample
/// groups, codec configuration, timing.
fn assert_audio_untouched(name: &str, mp4: &[u8], source: &[u8]) {
    let src = first_audio(source);
    let out = first_audio(mp4);
    assert_eq!(out.codec, src.codec, "{name}: codec");
    assert_eq!(out.timescale, src.timescale, "{name}: media timescale");
    assert_eq!(out.duration, src.duration, "{name}: media duration");
    assert_eq!(out.samples.len(), src.samples.len(), "{name}: sample count");
    let (a, b) = (sample_bytes(mp4, &out), sample_bytes(source, &src));
    for (i, (x, y)) in a.iter().zip(&b).enumerate() {
        assert!(x == y, "{name}: audio sample {i} differs");
    }
    for typ in [b"stts", b"stsz", b"stsc", b"stz2", b"ctts", b"sgpd", b"sbgp", b"esds", b"dOps", b"btrt"] {
        assert_eq!(trak_box(&out.trak_raw, typ), trak_box(&src.trak_raw, typ), "{name}: audio {} changed", fourcc(typ));
    }
    // The edit list keeps its media time (the encoder priming); its
    // segment duration is in the output's movie timescale.
    if let (Some(e_out), Some(e_src)) = (trak_box(&out.trak_raw, b"elst"), trak_box(&src.trak_raw, b"elst")) {
        assert_eq!(e_out.len(), e_src.len(), "{name}: elst");
        if e_src[0] == 0 && u32::from_be_bytes(e_src[4..8].try_into().unwrap()) == 1 {
            assert_eq!(e_out[12..20], e_src[12..20], "{name}: elst media time / rate");
            let movie_ts = movie_timescale(source) as u64;
            let d_src = u32::from_be_bytes(e_src[8..12].try_into().unwrap()) as u64;
            let d_out = u32::from_be_bytes(e_out[8..12].try_into().unwrap()) as u64;
            assert_eq!(d_out, (d_src * 1000 + movie_ts / 2) / movie_ts, "{name}: elst duration rescaled to 1000");
        }
    }
}

fn movie_timescale(file: &[u8]) -> u32 {
    let mvhd = source_boxes(file).into_iter().find(|b| b.path == "moov/mvhd").expect("mvhd").payload;
    let at = if mvhd[0] == 1 { 20 } else { 12 };
    u32::from_be_bytes(mvhd[at..at + 4].try_into().unwrap())
}

/// The video track is what a video-only mux writes: samples, sample tables
/// and media header.
fn assert_video_untouched(name: &str, with_audio: &[u8], video_only: &[u8]) {
    let video = |file: &[u8]| -> Track {
        let parsed = demux(file).expect("demux");
        let idx = parsed.video_track_idx.expect("video track");
        parsed.tracks.into_iter().nth(idx).unwrap()
    };
    let (a, b) = (video(with_audio), video(video_only));
    assert_eq!(sample_bytes(with_audio, &a), sample_bytes(video_only, &b), "{name}: video samples");
    for typ in [b"stsd", b"stts", b"ctts", b"stss", b"stsc", b"stsz", b"stco", b"co64", b"mdhd", b"hdlr"] {
        assert_eq!(trak_box(&a.trak_raw, typ), trak_box(&b.trak_raw, typ), "{name}: video {}", fourcc(typ));
    }
    // The video part of mdat, byte for byte and at the same offsets.
    let start = b.samples.iter().map(|s| s.offset).min().unwrap() as usize;
    let end = b.samples.iter().map(|s| s.offset + s.size as u64).max().unwrap() as usize;
    assert!(with_audio[start..end] == video_only[start..end], "{name}: video mdat bytes");
}

// ─── AV1 builder ────────────────────────────────────────────────────────

/// A synthetic AV1 stream: one sequence header and `n` frame OBUs.
fn synthetic_av1(n: usize) -> Vec<u8> {
    fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
        let mut out = vec![((obu_type & 0x0F) << 3) | 0b010];
        let mut len = payload.len() as u64;
        loop {
            let mut byte = (len & 0x7F) as u8;
            len >>= 7;
            if len != 0 {
                byte |= 0x80;
            }
            out.push(byte);
            if len == 0 {
                break;
            }
        }
        out.extend_from_slice(payload);
        out
    }
    let mut stream = obu(1, &[0u8; 12]);
    for i in 0..n {
        stream.extend_from_slice(&obu(6, &[i as u8; 100]));
    }
    stream
}

#[test]
fn av1_mp4_from_a_camera_file_carries_none_of_its_identity() {
    let timing = FrameTiming { fps_num: 30, fps_den: 1 };
    let obus = synthetic_av1(30);
    let video_only = build_mp4_av1(&obus, 1920, 1080, timing).expect("video-only mux");
    for (name, source) in fixtures() {
        let mp4 = build_mp4_av1_with_audio(&obus, 1920, 1080, timing, &source).expect("mux with audio");
        assert_no_source_identity(name, &mp4, &source);
        assert_audio_untouched(name, &mp4, &source);
        assert_video_untouched(name, &mp4, &video_only);
    }
}

// ─── H.264 builder ──────────────────────────────────────────────────────

/// A minimal Annex-B stream: SPS + PPS + an IDR access unit, then `n - 1`
/// P access units, each led by an AUD.
#[cfg(feature = "h264-encoder")]
fn synthetic_annexb(n: usize) -> Vec<u8> {
    let sc = [0u8, 0, 0, 1];
    let mut out = Vec::new();
    for nal in [
        &[0x67, 0x64, 0x00, 0x1E, 0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80][..],
        &[0x68, 0xEB, 0xE3, 0xCB, 0x22, 0xC0],
        &[0x09, 0x10],
        &[0x65, 0x88, 0x84, 0x00, 0x33, 0xFF, 0xFE],
    ] {
        out.extend_from_slice(&sc);
        out.extend_from_slice(nal);
    }
    for i in 1..n {
        out.extend_from_slice(&sc);
        out.extend_from_slice(&[0x09, 0x30]);
        out.extend_from_slice(&sc);
        out.extend_from_slice(&[0x41, 0x9A, 0x02, i as u8 | 0x10, 0x55, 0xAA]);
    }
    out
}

#[cfg(feature = "h264-encoder")]
#[test]
fn h264_mp4_from_a_camera_file_carries_none_of_its_identity() {
    use phasm_core::GopPattern;
    use phasm_core::codec::mp4::build::{MuxerProfile, build_mp4_with_pattern, build_mp4_with_pattern_audio};

    let n = 30;
    let annexb = synthetic_annexb(n);
    let pattern = GopPattern::Ipppp { gop: 30 };
    let video_only =
        build_mp4_with_pattern(MuxerProfile::HandbrakeX264, &annexb, 1920, 1080, FrameTiming::FPS_30, pattern, n)
            .expect("video-only mux");
    for (name, source) in fixtures() {
        let mp4 = build_mp4_with_pattern_audio(
            MuxerProfile::HandbrakeX264,
            &annexb,
            1920,
            1080,
            FrameTiming::FPS_30,
            pattern,
            n,
            &source,
        )
        .expect("mux with audio");
        assert_no_source_identity(name, &mp4, &source);
        assert_audio_untouched(name, &mp4, &source);
        assert_video_untouched(name, &mp4, &video_only);
    }
}
