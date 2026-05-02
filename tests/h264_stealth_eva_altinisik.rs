// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §Stealth.L4.4 — Container-fingerprint regression test.
//
// Replicates the decision-tree splits from two published forensic
// container classifiers and asserts phasm output lands at the
// HandBrake/FFmpeg+libx264 leaf:
//
//   * Yang et al. (EVA / IEEE JSTSP 14(5) 2020 / arXiv:2101.10795) —
//     97.6% integrity from a single decision tree over container
//     metadata. Sample splits cited in the paper:
//       count(root/ftyp/@minorVersion=0)
//       count(root/uuid/@userType)
//       count(root/moov/udta/XMP_/@stuff)
//       compatibleBrand_1 = "mp42"
//       count(.../avc1/avcC/@AVCLevelIndication=31)
//
//   * Altinisik, Sencar, Tabaa (arXiv:2201.02949v4, IEEE TIFS 2022) —
//     91.14% on 119 classes. Tree-hash the full PathField topology of
//     the MP4; equal hash → equal metaclass.
//
// The clean-room implementations below are deliberately minimal: just
// enough atom-walking to compute the splits the published trees
// document. Their goal is regression coverage — flag any future
// container-mux change that pushes phasm output out of the libx264
// metaclass leaf — not to provide a competitive forensic detector.

#![cfg(feature = "video")]

use phasm_core::codec::mp4::build::{build_mp4, FrameTiming, MuxerProfile};
use phasm_core::codec::mp4::AvccData;

/// Minimal MP4 box header (test-local clone of mp4/mod.rs internals
/// to avoid relying on `pub(crate)` items).
#[derive(Debug, Clone, Copy)]
struct BoxHeader {
    box_type: [u8; 4],
    size: u64,
    header_len: u8,
}

fn parse_box_header(data: &[u8], offset: usize) -> Option<BoxHeader> {
    if offset + 8 > data.len() {
        return None;
    }
    let size32 = u32::from_be_bytes([
        data[offset], data[offset + 1], data[offset + 2], data[offset + 3],
    ]);
    let box_type = [
        data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7],
    ];
    if size32 == 1 {
        if offset + 16 > data.len() {
            return None;
        }
        let size64 = u64::from_be_bytes([
            data[offset + 8], data[offset + 9], data[offset + 10], data[offset + 11],
            data[offset + 12], data[offset + 13], data[offset + 14], data[offset + 15],
        ]);
        Some(BoxHeader { box_type, size: size64, header_len: 16 })
    } else if size32 == 0 {
        Some(BoxHeader {
            box_type,
            size: (data.len() - offset) as u64,
            header_len: 8,
        })
    } else {
        Some(BoxHeader { box_type, size: size32 as u64, header_len: 8 })
    }
}

fn iterate_boxes<F: FnMut(&BoxHeader, usize)>(
    data: &[u8],
    start: usize,
    end: usize,
    mut visitor: F,
) {
    let mut pos = start;
    while pos < end {
        let h = match parse_box_header(data, pos) {
            Some(h) => h,
            None => break,
        };
        if h.size < 8 || pos + h.size as usize > end {
            break;
        }
        let content_start = pos + h.header_len as usize;
        visitor(&h, content_start);
        pos += h.size as usize;
    }
}

#[derive(Debug, PartialEq, Eq)]
enum EvaLeaf {
    /// Adobe / Premiere / Avidemux / external editor — uuid+userType
    /// or moov/udta/XMP_ traces present.
    Adobe,
    /// Apple `.mp4` modern (iPhone 5s+ / iPad / iOS 8+) — compatible
    /// brand 1 is "mp42" with no editor traces.
    AppleMp4,
    /// Apple `.MOV` capture — major brand "qt  ".
    AppleMov,
    /// Android legacy 3GP — compatible brand 1 is "3gp4".
    Android3gp,
    /// FFmpeg / libx264 / HandBrake — compatible brand 1 is "iso2".
    LibX264Class,
    /// Unknown / phasm-distinctive — file did not match any leaf.
    Unknown,
}

/// Run the Yang/EVA decision tree over an MP4 byte slice and return
/// the assigned leaf class. The tree order matches the paper's
/// importance ordering (most-discriminating splits first). Brand
/// pairs are taken from Agent B's container forensics survey
/// §Q2 ("Per-encoder canonical signatures").
fn eva_classify(mp4: &[u8]) -> EvaLeaf {
    if has_box(mp4, b"uuid") || has_path(mp4, &[b"moov", b"udta", b"XMP_"]) {
        return EvaLeaf::Adobe;
    }
    let ftyp = match read_ftyp(mp4) {
        Some(f) => f,
        None => return EvaLeaf::Unknown,
    };
    let cb1 = ftyp.compatibles.get(1).copied();
    let cb1_slice = cb1.as_ref().map(|c| &c[..]);
    match (&ftyp.major_brand, cb1_slice) {
        // Apple QuickTime .MOV — major "qt  " drives the leaf alone.
        (b"qt  ", _) => EvaLeaf::AppleMov,
        // Apple .MP4 (modern iPhone): "mp42" / [mp42 isom 3gp4 mp41].
        (b"mp42", Some(b"isom")) => EvaLeaf::AppleMp4,
        // Android 3GP legacy: "3gp5" / [3gp5 3gp4 isom].
        (b"3gp5", _) => EvaLeaf::Android3gp,
        // FFmpeg / libx264 / HandBrake: "isom" / [isom iso2 avc1 mp41].
        (b"isom", Some(b"iso2")) => EvaLeaf::LibX264Class,
        _ => EvaLeaf::Unknown,
    }
}

/// Compute an Altinisik-style topology signature: a flat list of
/// `/`-joined box-path strings, deduplicated and sorted. Two MP4 files
/// with the same atom topology produce the same signature regardless
/// of dynamic field values (timestamps, sample counts, etc.).
fn topology_signature(mp4: &[u8]) -> Vec<String> {
    let mut paths = Vec::new();
    walk_paths(mp4, 0, mp4.len(), "", &mut paths);
    paths.sort();
    paths.dedup();
    paths
}

fn walk_paths(data: &[u8], start: usize, end: usize, prefix: &str, out: &mut Vec<String>) {
    iterate_boxes(data, start, end, |h, content_start| {
        let name = box_name(&h.box_type);
        let path = if prefix.is_empty() {
            name.clone()
        } else {
            format!("{prefix}/{name}")
        };
        out.push(path.clone());
        let inner_end = content_start + h.size as usize - h.header_len as usize;
        if let Some(child_off) = container_child_offset(&h.box_type) {
            let cs = content_start + child_off;
            if cs <= inner_end {
                walk_paths(data, cs, inner_end, &path, out);
            }
        }
    });
}

/// Render a box-type fourcc as a printable string. Handles the
/// non-UTF-8 0xA9 ("©") byte that QuickTime user-data atoms use.
fn box_name(t: &[u8; 4]) -> String {
    let mut s = String::with_capacity(4);
    for &b in t {
        if b == 0xA9 {
            s.push('©');
        } else if b.is_ascii() {
            s.push(b as char);
        } else {
            s.push('?');
        }
    }
    s
}

/// Number of bytes to skip past a container-box header before its
/// children begin. ISO base-media simple containers (`moov`, `trak`,
/// etc.) have offset 0; FullBox containers like `meta` have 4
/// (version+flags); `dref` and `stsd` have 8 (version+flags +
/// entry_count); `avc1` has 78 (VisualSampleEntry header). Returns
/// `None` for non-containers.
fn container_child_offset(box_type: &[u8; 4]) -> Option<usize> {
    match box_type {
        b"moov" | b"trak" | b"mdia" | b"minf" | b"stbl" | b"dinf"
        | b"udta" | b"edts" => Some(0),
        b"meta" => Some(4),
        b"dref" | b"stsd" => Some(8),
        b"avc1" => Some(78),
        _ => None,
    }
}

#[derive(Debug)]
struct Ftyp {
    major_brand: [u8; 4],
    minor_version: u32,
    compatibles: Vec<[u8; 4]>,
}

fn read_ftyp(mp4: &[u8]) -> Option<Ftyp> {
    let h = parse_box_header(mp4, 0)?;
    if h.box_type != *b"ftyp" {
        return None;
    }
    let cs = h.header_len as usize;
    let end = h.size as usize;
    if cs + 8 > end {
        return None;
    }
    let major_brand = [mp4[cs], mp4[cs + 1], mp4[cs + 2], mp4[cs + 3]];
    let minor_version = u32::from_be_bytes([mp4[cs + 4], mp4[cs + 5], mp4[cs + 6], mp4[cs + 7]]);
    let mut compatibles = Vec::new();
    let mut p = cs + 8;
    while p + 4 <= end {
        compatibles.push([mp4[p], mp4[p + 1], mp4[p + 2], mp4[p + 3]]);
        p += 4;
    }
    Some(Ftyp { major_brand, minor_version, compatibles })
}

fn has_box(mp4: &[u8], target: &[u8; 4]) -> bool {
    let mut found = false;
    walk_for_box(mp4, 0, mp4.len(), target, &mut found);
    found
}

fn walk_for_box(data: &[u8], start: usize, end: usize, target: &[u8; 4], found: &mut bool) {
    if *found {
        return;
    }
    iterate_boxes(data, start, end, |h, content_start| {
        if *found {
            return;
        }
        if h.box_type == *target {
            *found = true;
            return;
        }
        if let Some(child_off) = container_child_offset(&h.box_type) {
            let inner_end = content_start + h.size as usize - h.header_len as usize;
            let cs = content_start + child_off;
            if cs <= inner_end {
                walk_for_box(data, cs, inner_end, target, found);
            }
        }
    });
}

/// Walk a fixed box-path (e.g. `moov/udta/XMP_`) and return whether
/// it exists at exactly that nesting.
fn has_path(data: &[u8], path: &[&[u8; 4]]) -> bool {
    fn go(data: &[u8], start: usize, end: usize, path: &[&[u8; 4]]) -> bool {
        if path.is_empty() {
            return true;
        }
        let mut hit = false;
        iterate_boxes(data, start, end, |h, content_start| {
            if hit {
                return;
            }
            if h.box_type == *path[0] {
                if path.len() == 1 {
                    hit = true;
                } else {
                    let inner_end =
                        content_start + h.size as usize - h.header_len as usize;
                    if go(data, content_start, inner_end, &path[1..]) {
                        hit = true;
                    }
                }
            }
        });
        hit
    }
    go(data, 0, data.len(), path)
}

/// Build the minimum-viable Annex-B for the muxer (single IDR access
/// unit). Reused from the L4.1/L4.2/L4.3 unit tests.
fn build_minimal_annexb() -> Vec<u8> {
    let sps = vec![
        0x67, 0x64, 0x00, 0x1E, 0xAC, 0xD9, 0x40, 0x40, 0x3C, 0x80,
    ];
    let pps = vec![0x68, 0xEB, 0xE3, 0xCB, 0x22, 0xC0];
    let aud = vec![0x09, 0x10];
    let idr = vec![0x65, 0x88, 0x84, 0x00, 0x33, 0xFF, 0xFE];
    let mut out = Vec::new();
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend(&sps);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend(&pps);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend(&aud);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.extend(&idr);
    out
}

#[test]
fn phasm_handbrake_output_lands_at_libx264_class_leaf() {
    let annex_b = build_minimal_annexb();
    let mp4 = build_mp4(
        MuxerProfile::HandbrakeX264,
        &annex_b,
        1920,
        1080,
        FrameTiming::FPS_30,
    )
    .expect("build_mp4");

    let leaf = eva_classify(&mp4);
    assert_eq!(
        leaf,
        EvaLeaf::LibX264Class,
        "phasm HandBrake output must classify as libx264-class — \
         was {leaf:?}, which moves the file out of the populous \
         metaclass and into a Yang/EVA leaf the strategy doc forbids",
    );
}

#[test]
fn phasm_handbrake_output_has_no_editor_traces() {
    // §Stealth.L4.3 — phasm must NOT emit any of the atoms that
    // Yang/EVA's strongest splitters key on (uuid/userType, XMP_,
    // ©mod/©swr/©day/©xyz). All Adobe/Premiere/Apple-iTunes traces
    // must be absent from the HandBrake profile output.
    let annex_b = build_minimal_annexb();
    let mp4 = build_mp4(
        MuxerProfile::HandbrakeX264,
        &annex_b,
        1920,
        1080,
        FrameTiming::FPS_30,
    )
    .unwrap();

    assert!(!has_box(&mp4, b"uuid"), "no `uuid` atom (Adobe/external-editor tell)");
    assert!(!has_box(&mp4, b"meta"), "no `meta` atom in HandBrake output");
    assert!(!has_path(&mp4, &[b"moov", b"udta", b"XMP_"]), "no XMP_");
    assert!(!has_path(&mp4, &[b"moov", b"udta", &[0xA9, b'm', b'o', b'd']]), "no ©mod");
    assert!(!has_path(&mp4, &[b"moov", b"udta", &[0xA9, b's', b'w', b'r']]), "no ©swr");
    assert!(!has_path(&mp4, &[b"moov", b"udta", &[0xA9, b'd', b'a', b'y']]), "no ©day");
    assert!(!has_path(&mp4, &[b"moov", b"udta", &[0xA9, b'x', b'y', b'z']]), "no ©xyz");
    // ©too IS expected — that's the HandBrake plaintext marker.
    assert!(has_path(&mp4, &[b"moov", b"udta", &[0xA9, b't', b'o', b'o']]), "©too present");
}

#[test]
fn phasm_handbrake_topology_matches_libx264_metaclass() {
    // Altinisik 2022 tree-hash equivalent: assert phasm's atom
    // topology equals the canonical HandBrake/x264 shape (no
    // accidental additions, no missing required atoms).
    let annex_b = build_minimal_annexb();
    let mp4 = build_mp4(
        MuxerProfile::HandbrakeX264,
        &annex_b,
        1920,
        1080,
        FrameTiming::FPS_30,
    )
    .unwrap();

    let topology = topology_signature(&mp4);
    let expected: Vec<&str> = vec![
        "ftyp",
        "mdat",
        "moov",
        "moov/mvhd",
        "moov/trak",
        "moov/trak/mdia",
        "moov/trak/mdia/hdlr",
        "moov/trak/mdia/mdhd",
        "moov/trak/mdia/minf",
        "moov/trak/mdia/minf/dinf",
        "moov/trak/mdia/minf/dinf/dref",
        "moov/trak/mdia/minf/dinf/dref/url ",
        "moov/trak/mdia/minf/stbl",
        "moov/trak/mdia/minf/stbl/stco",
        "moov/trak/mdia/minf/stbl/stsc",
        "moov/trak/mdia/minf/stbl/stsd",
        "moov/trak/mdia/minf/stbl/stsd/avc1",
        "moov/trak/mdia/minf/stbl/stsd/avc1/avcC",
        "moov/trak/mdia/minf/stbl/stss",
        "moov/trak/mdia/minf/stbl/stsz",
        "moov/trak/mdia/minf/stbl/stts",
        "moov/trak/mdia/minf/vmhd",
        "moov/trak/tkhd",
        "moov/udta",
        "moov/udta/\u{a9}too",
    ];
    let expected_owned: Vec<String> = expected.iter().map(|s| s.to_string()).collect();
    assert_eq!(topology, expected_owned, "atom topology drift vs HandBrake reference");
}

#[test]
fn classifier_discriminates_apple_mp4_from_libx264() {
    // Negative control: hand-craft a minimal ftyp with `mp42` major +
    // `[mp42, isom, avc1, mp41]` compatibles (modern Apple .mp4) and
    // confirm `eva_classify` routes it to AppleMp4, NOT LibX264Class.
    // Proves the classifier actually discriminates instead of always
    // returning LibX264Class.
    let mut mp4 = Vec::new();
    let ftyp_content = {
        let mut c = Vec::new();
        c.extend_from_slice(b"mp42"); // major
        c.extend_from_slice(&0x0000_0000u32.to_be_bytes()); // minor
        c.extend_from_slice(b"mp42");
        c.extend_from_slice(b"isom");
        c.extend_from_slice(b"avc1");
        c.extend_from_slice(b"mp41");
        c
    };
    mp4.extend_from_slice(&((8 + ftyp_content.len()) as u32).to_be_bytes());
    mp4.extend_from_slice(b"ftyp");
    mp4.extend_from_slice(&ftyp_content);

    assert_eq!(eva_classify(&mp4), EvaLeaf::AppleMp4);
}

#[test]
fn classifier_discriminates_adobe_uuid_trace() {
    // Negative control: an ftyp that would otherwise classify as
    // LibX264Class, plus a uuid box → should route to Adobe.
    let mut mp4 = Vec::new();
    let ftyp_content = {
        let mut c = Vec::new();
        c.extend_from_slice(b"isom");
        c.extend_from_slice(&0x0000_0200u32.to_be_bytes());
        c.extend_from_slice(b"isom");
        c.extend_from_slice(b"iso2");
        c.extend_from_slice(b"avc1");
        c.extend_from_slice(b"mp41");
        c
    };
    mp4.extend_from_slice(&((8 + ftyp_content.len()) as u32).to_be_bytes());
    mp4.extend_from_slice(b"ftyp");
    mp4.extend_from_slice(&ftyp_content);
    // Synthetic uuid box (24 byte content = 16-byte uuid + 8 bytes data).
    mp4.extend_from_slice(&(8 + 24u32).to_be_bytes());
    mp4.extend_from_slice(b"uuid");
    mp4.extend_from_slice(&[0u8; 24]);

    assert_eq!(eva_classify(&mp4), EvaLeaf::Adobe);
}

#[test]
fn phasm_avcc_carries_decodable_sps_pps() {
    // Yang/EVA's `count(.../avc1/avcC/@AVCLevelIndication=31)` split
    // requires a parseable avcC. Make sure phasm's avcC is
    // structurally valid — profile + level fields populated and SPS/
    // PPS NAL counts correct.
    let annex_b = build_minimal_annexb();
    let avcc = AvccData::from_annexb(&annex_b).expect("avcC parses");
    // SPS hand-crafted in build_minimal_annexb has profile_idc=0x64
    // (High), constraint=0x00, level_idc=0x1E (level 3.0).
    assert_eq!(avcc.profile, 0x64);
    assert_eq!(avcc.profile_compat, 0x00);
    assert_eq!(avcc.level, 0x1E);
    assert_eq!(avcc.length_size_minus1, 3);
    assert_eq!(avcc.sps_nalus.len(), 1);
    assert_eq!(avcc.pps_nalus.len(), 1);
}
