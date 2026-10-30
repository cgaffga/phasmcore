// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Exact-size H.264 output — frames whose width / height are not multiples of
//! 16 are coded on the MB grid and cropped back in the SPS, like a phone
//! encoder (1920×1080 → 120×68 MBs, `crop_bottom = 4`). Design note:
//! `docs/design/video/h264/exact-size-frame-cropping.md`.
//!
//! What is pinned here:
//! - the Rust cost model's view of the padded pixels is the encoder's own
//!   (`FrameGeometry::pad_frame_into` == OpenH264's internal fill, proven by
//!   identical slice bytes);
//! - the SPS carries the exact size (MB grid + right/bottom crop only);
//! - primary, attachment and shadow round trips at cropped sizes in both
//!   orientations, plus 1920×1080;
//! - streaming shadow encode == whole-clip shadow encode at a cropped size;
//! - capacity probe vs actual embed at a cropped size;
//! - MP4 `tkhd` / `avc1` carry the display size, and a coded-size mux is
//!   refused;
//! - cropped output decodes at the exact size with the same quality as the
//!   16-floored encode of the same pixels.
//!
//! OpenH264 encodes share process-global fork state, so every test takes the
//! session guard (`--test-threads=1` is not required).
//!
//!   cargo test -p phasm-core --release --features h264-encoder \
//!     --test oh264_exact_size_crop

#![cfg(feature = "h264-encoder")]

mod common;
use common::oh264_stream;
use common::video_fixture::{ffmpeg_decode_annex_b, session_guard};
use common::visual_metrics::{psnr_y, worst_mb_psnr_y};

use phasm_core::codec::h264::bitstream::remove_emulation_prevention;
use phasm_core::codec::h264::frame_geometry::FrameGeometry;
use phasm_core::codec::h264::openh264_stego::{
    h264_encode_with_shadows, h264_encode_with_shadows_streaming, EncodeOpts, SliceYuvSource,
};
use phasm_core::codec::h264::sps::{parse_sps, Sps};
use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingEncodeSession, YuvFrameRef,
};
use phasm_core::codec::mp4::build::{build_mp4_with_pattern, FrameTiming, MuxerProfile};
use phasm_core::stego::payload::FileEntry;
use phasm_core::stego::shadow_layer::ShadowLayer;
use phasm_core::{GopPattern, StegoError};

const QP: i32 = 26;

fn guard() -> std::sync::MutexGuard<'static, ()> {
    session_guard().lock().unwrap_or_else(|e| e.into_inner())
}

/// Deterministic textured clip (same formula as the lib + shadow tests), so
/// every MB — including the padded edge MBs — carries real residual energy.
fn synth_yuv(w: u32, h: u32, n_frames: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity((w * h * 3 / 2 * n_frames) as usize);
    for f in 0..n_frames {
        for j in 0..h {
            for i in 0..w {
                out.push(((i + f * 2) ^ (j + f * 3)) as u8);
            }
        }
        let mut s: u32 = 0xCAFE_F00D ^ f;
        for _plane in 0..2 {
            for j in 0..h / 2 {
                for i in 0..w / 2 {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    out.push(((s >> 16) as u8).wrapping_add((i + j + f) as u8));
                }
            }
        }
    }
    out
}

/// The top-left `cw × ch` window of every frame of a `w × h` clip.
fn crop_clip(yuv: &[u8], w: u32, h: u32, n_frames: u32, cw: u32, ch: u32) -> Vec<u8> {
    let mut out = Vec::new();
    for f in 0..n_frames as usize {
        let (y, u, v) = oh264_stream::frame_planes(yuv, w, h, f);
        for (plane, pw, cpw, cph) in [
            (y, w as usize, cw as usize, ch as usize),
            (u, (w / 2) as usize, (cw / 2) as usize, (ch / 2) as usize),
            (v, (w / 2) as usize, (cw / 2) as usize, (ch / 2) as usize),
        ] {
            for row in 0..cph {
                out.extend_from_slice(&plane[row * pw..row * pw + cpw]);
            }
        }
    }
    out
}

/// Split Annex-B into raw NAL units (header byte included, start codes
/// stripped).
fn split_nals(annex_b: &[u8]) -> Vec<&[u8]> {
    let mut starts = Vec::new();
    let mut i = 0;
    while i + 3 <= annex_b.len() {
        if annex_b[i] == 0 && annex_b[i + 1] == 0 && annex_b[i + 2] == 1 {
            starts.push(i + 3);
            i += 3;
        } else {
            i += 1;
        }
    }
    let mut out = Vec::with_capacity(starts.len());
    for (k, &s) in starts.iter().enumerate() {
        let mut e = if k + 1 < starts.len() { starts[k + 1] - 3 } else { annex_b.len() };
        // A 4-byte start code leaves its leading zero on the previous NAL.
        while e > s && annex_b[e - 1] == 0 && k + 1 < starts.len() {
            e -= 1;
        }
        out.push(&annex_b[s..e]);
    }
    out
}

fn first_sps(annex_b: &[u8]) -> Sps {
    let nal = split_nals(annex_b)
        .into_iter()
        .find(|n| !n.is_empty() && n[0] & 0x1F == 7)
        .expect("no SPS in stream");
    parse_sps(&remove_emulation_prevention(&nal[1..])).expect("parse SPS")
}

/// The SPS must code the MB grid and crop right/bottom only, back to `w × h`.
fn assert_sps_exact(annex_b: &[u8], w: u32, h: u32) {
    let g = FrameGeometry::new(w, h).unwrap();
    let sps = first_sps(annex_b);
    assert_eq!(
        (sps.pic_width_in_mbs, sps.pic_height_in_map_units),
        (g.mb_width(), g.mb_height()),
        "{w}x{h}: SPS MB grid"
    );
    assert!(sps.frame_mbs_only_flag);
    assert_eq!(sps.frame_cropping_flag, !g.is_mb_aligned(), "{w}x{h}: cropping flag");
    assert_eq!(
        (sps.crop_left, sps.crop_right, sps.crop_top, sps.crop_bottom),
        (0, g.sps_crop_right(), 0, g.sps_crop_bottom()),
        "{w}x{h}: crop offsets (4:2:0 units, right/bottom only)"
    );
    assert_eq!((sps.width_in_pixels, sps.height_in_pixels), (w, h), "{w}x{h}: SPS display size");
}

fn params(w: u32, h: u32, n: u32, gop: u32) -> EncodeSessionParams {
    EncodeSessionParams {
        width: w,
        height: h,
        fps_num: 30,
        fps_den: 1,
        qp: QP,
        gop_size: gop,
        total_frames_hint: n,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights: CostWeights::default(),
        progress_callback: None,
    }
}

fn frame_ref(yuv: &[u8], w: u32, h: u32, f: usize) -> YuvFrameRef<'_> {
    let (y, u, v) = oh264_stream::frame_planes(yuv, w, h, f);
    YuvFrameRef { y, y_stride: w as usize, u, u_stride: (w / 2) as usize, v, v_stride: (w / 2) as usize }
}

/// Decode a stream with an independent decoder (ffmpeg, which applies the SPS
/// crop) and return the luma planes of all frames. Asserts the decoder
/// produced exactly `n` frames of `w × h` — i.e. that it honoured the crop and
/// output the exact size — and reported no concealment.
fn decode_luma_exact(annex_b: &[u8], w: u32, h: u32, n: u32, tag: &str) -> Vec<u8> {
    let clip = ffmpeg_decode_annex_b(annex_b, &format!("exact_size_{tag}"));
    assert_eq!(clip.n_conceal, 0, "{tag}: decoder concealment (see {})", clip.log_path);
    assert_eq!(
        clip.yuv.len(),
        (w * h * 3 / 2 * n) as usize,
        "{tag}: decoder output is not {n} frames of {w}x{h} (crop not honoured?)"
    );
    let y = luma_of(&clip.yuv, w, h, n);
    clip.discard_decoded_yuv();
    y
}

fn luma_of(yuv: &[u8], w: u32, h: u32, n: u32) -> Vec<u8> {
    (0..n as usize).flat_map(|f| oh264_stream::frame_planes(yuv, w, h, f).0.to_vec()).collect()
}

// ───────────────────── padding = OpenH264's own fill ─────────────────────

/// The cost model prices padded MBs from `FrameGeometry::pad_frame_into`. That
/// is only right if it is EXACTLY the picture OpenH264 codes. Proof: encode the
/// display-size clip (OpenH264 pads internally) and the Rust-padded coded-size
/// clip (no internal padding) with the same encoder settings. Every PPS and
/// slice NAL must be byte-identical — the encoder saw identical pixels — and
/// only the SPS differs, by the crop.
#[test]
fn exact_size_padding_matches_openh264_internal_fill() {
    let _g = guard();
    for &(w, h) in &[(200u32, 120u32), (120, 200), (854, 480), (66, 34)] {
        let n = 3u32;
        let geom = FrameGeometry::new(w, h).unwrap();
        let (cw, ch) = (geom.coded_width(), geom.coded_height());
        let yuv = synth_yuv(w, h, n);
        let mut padded = Vec::new();
        let mut one = Vec::new();
        for f in 0..n as usize {
            let fb = geom.frame_bytes();
            geom.pad_frame_into(&yuv[f * fb..(f + 1) * fb], &mut one);
            padded.extend_from_slice(&one);
        }
        let exact = oh264_stream::encode_clean(&yuv, w, h, n, QP, n);
        let coded = oh264_stream::encode_clean(&padded, cw, ch, n, QP, n);

        assert_sps_exact(&exact, w, h);
        assert_sps_exact(&coded, cw, ch);

        let a: Vec<&[u8]> = split_nals(&exact).into_iter().filter(|n| n[0] & 0x1F != 7).collect();
        let b: Vec<&[u8]> = split_nals(&coded).into_iter().filter(|n| n[0] & 0x1F != 7).collect();
        assert_eq!(a.len(), b.len(), "{w}x{h}: NAL count");
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert!(
                x == y,
                "{w}x{h}: NAL {i} (type {}) differs between OpenH264-padded and Rust-padded input \
                 — the Rust fill no longer matches the encoder's",
                x[0] & 0x1F
            );
        }
    }
}

/// The whole stego pipeline (Pass-1 cover, coded-frame J-UNIWARD costs,
/// cascade gates, STC plan, Pass-2 wire-only emit) must make exactly the
/// decisions it makes for an ALIGNED encode of the coded picture: stego-encode
/// the exact-size clip, and the same clip padded to the MB grid with the
/// encoder's fill and encoded as an aligned size, under one pinned crypto
/// seed. Every NAL but the SPS must be byte-identical. So an exact-size encode
/// is, by construction, the aligned encode of what the encoder codes plus an
/// SPS crop — any property of its output (capacity, visual quality, stealth)
/// is the aligned pipeline's on those pixels, not a cropping artefact.
#[test]
fn exact_size_stego_equals_aligned_encode_of_padded_frame() {
    let _g = guard();
    // SAFETY: every test in this binary holds the session guard.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "777") };
    for &(w, h, n, gop) in &[(200u32, 120u32, 8u32, 4u32), (120, 200, 8, 4), (1920, 1080, 2, 2)] {
        let geom = FrameGeometry::new(w, h).unwrap();
        let (cw, ch) = (geom.coded_width(), geom.coded_height());
        let yuv = synth_yuv(w, h, n);
        let mut padded = Vec::new();
        let mut one = Vec::new();
        for f in 0..n as usize {
            let fb = geom.frame_bytes();
            geom.pad_frame_into(&yuv[f * fb..(f + 1) * fb], &mut one);
            padded.extend_from_slice(&one);
        }
        let msg = "same decisions as the aligned encode of the coded frame";
        let exact = oh264_stream::encode_with(&yuv, w, h, n, QP, gop, CostWeights::default(), msg, "eq")
            .expect("exact encode");
        let coded = oh264_stream::encode_with(&padded, cw, ch, n, QP, gop, CostWeights::default(), msg, "eq")
            .expect("aligned encode of the padded frame");
        assert_sps_exact(&exact, w, h);
        let a: Vec<&[u8]> = split_nals(&exact).into_iter().filter(|n| n[0] & 0x1F != 7).collect();
        let b: Vec<&[u8]> = split_nals(&coded).into_iter().filter(|n| n[0] & 0x1F != 7).collect();
        assert_eq!(a.len(), b.len(), "{w}x{h}: NAL count");
        let first_diff = a.iter().zip(&b).position(|(x, y)| x != y);
        assert_eq!(first_diff, None, "{w}x{h}: exact-size stego diverges from the aligned encode of its coded frame");
        assert_eq!(oh264_stream::decode_text(&exact, "eq").expect("decode exact"), msg);
    }
    // SAFETY: as above.
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
}

// ───────────────────────── primary round trips ─────────────────────────

fn primary_roundtrip(w: u32, h: u32, n: u32, gop: u32, msg: &str) -> Vec<u8> {
    let yuv = synth_yuv(w, h, n);
    let stego = oh264_stream::encode_with(&yuv, w, h, n, QP, gop, CostWeights::default(), msg, "crop-pass")
        .unwrap_or_else(|e| panic!("{w}x{h}: encode failed: {e:?}"));
    assert_sps_exact(&stego, w, h);
    let text = oh264_stream::decode_text(&stego, "crop-pass")
        .unwrap_or_else(|e| panic!("{w}x{h}: decode failed: {e:?}"));
    assert_eq!(text, msg, "{w}x{h}: message mismatch");
    // Wrong passphrase must not decode.
    assert!(oh264_stream::decode_text(&stego, "not-the-pass").is_err(), "{w}x{h}: wrong pass decoded");
    stego
}

#[test]
fn exact_size_primary_roundtrip_landscape_and_portrait() {
    let _g = guard();
    // Both axes cropped, landscape and portrait, multi-GOP.
    primary_roundtrip(200, 120, 8, 4, "exact-size landscape 200x120");
    primary_roundtrip(120, 200, 8, 4, "exact-size portrait 120x200");
    // Only one axis cropped each way.
    primary_roundtrip(208, 120, 4, 4, "height-only crop");
    primary_roundtrip(200, 128, 4, 4, "width-only crop");
}

#[test]
fn exact_size_primary_roundtrip_480p() {
    let _g = guard();
    // 854×480: the 16:9 480p size the resolution picker needs (54×30 MBs,
    // crop_right = 5).
    let stego = primary_roundtrip(854, 480, 4, 4, "resolution picker 480p");
    let sps = first_sps(&stego);
    assert_eq!((sps.pic_width_in_mbs, sps.crop_right, sps.crop_bottom), (54, 5, 0));
}

/// Real 1080p geometry: 1920×1080 → 120×68 MBs, `crop_bottom = 4` — the same
/// SPS shape as the iPhone / Lumix corpus clips.
#[test]
fn exact_size_primary_roundtrip_1080p() {
    let _g = guard();
    let stego = primary_roundtrip(1920, 1080, 2, 2, "genuine 1920x1080, not 1072");
    let sps = first_sps(&stego);
    assert_eq!(
        (sps.pic_width_in_mbs, sps.pic_height_in_map_units, sps.crop_right, sps.crop_bottom),
        (120, 68, 0, 4)
    );
    decode_luma_exact(&stego, 1920, 1080, 2, "1080p");
}

#[test]
fn exact_size_attachment_roundtrip() {
    let _g = guard();
    let (w, h, n) = (200u32, 120u32, 8u32);
    let yuv = synth_yuv(w, h, n);
    let files = [FileEntry { filename: "note.txt".into(), content: b"exact-size attachment".to_vec() }];
    let stego = oh264_stream::encode_with_files(&yuv, w, h, n, QP, "with a file", &files, "file-pass")
        .expect("encode with files");
    assert_sps_exact(&stego, w, h);
    let res = oh264_stream::decode_full(&stego, "file-pass").expect("decode");
    assert_eq!(res.text, "with a file");
    assert_eq!(res.files.len(), 1);
    assert_eq!(res.files[0].filename, "note.txt");
    assert_eq!(res.files[0].content, b"exact-size attachment");
}

// ───────────────────────────── shadows ─────────────────────────────────

/// Primary + one shadow at a cropped size (both axes), through the streaming
/// pull encoder the bridges use — and the whole-clip encoder must emit the
/// same bytes (the WV.6.g byte-identity contract also holds when cropped).
#[test]
fn exact_size_shadow_roundtrip_and_streaming_equals_whole_clip() {
    let _g = guard();
    // Pinned crypto salt/nonce so the two encoders draw identical randomness
    // (same protocol as `oh264_g4_streaming_byte_identical`). Serialized by
    // the session guard; cleared below.
    // SAFETY: every test in this binary holds the session guard.
    unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", "4242") };
    for &(w, h) in &[(312u32, 232u32), (232, 312)] {
        let (n, gop) = (20u32, 10u32);
        let yuv = synth_yuv(w, h, n);
        let opts = EncodeOpts { qp: QP, intra_period: gop as i32 };
        let shadows = [ShadowLayer { message: "shdw", passphrase: "s", files: &[] }];
        let primary = "primary message, longer than the shadow";

        let mut src = SliceYuvSource::new(&yuv, w, h, n, gop);
        let streamed = h264_encode_with_shadows_streaming(
            &mut src, w, h, n, opts, primary, &[], "p", &shadows, &CostWeights::default(), None, None,
        )
        .unwrap_or_else(|e| panic!("{w}x{h}: streaming shadow encode: {e:?}"));
        assert_sps_exact(&streamed, w, h);
        assert_eq!(oh264_stream::decode_text(&streamed, "p").expect("primary decode"), primary);
        assert_eq!(oh264_stream::decode_text(&streamed, "s").expect("shadow decode"), "shdw");

        let whole = h264_encode_with_shadows(
            &yuv, w, h, n, opts, primary, &[], "p", &shadows, &CostWeights::default(),
        )
        .expect("whole-clip shadow encode");
        assert!(whole == streamed, "{w}x{h}: streaming shadow encode != whole-clip encode");
    }
    // SAFETY: as above.
    unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
}

// ──────────────────────────── capacity ─────────────────────────────────

/// Capacity at a cropped size: the probe counts the padded MB row/column (more
/// cover than the 16-floored clip of the same pixels), what it offers actually
/// embeds and decodes, and it does not wildly under-report (a payload far past
/// the offer is refused).
#[test]
fn exact_size_capacity_probe_matches_encode() {
    let _g = guard();
    let (w, h, n, gop) = (200u32, 120u32, 8u32, 4u32);
    let yuv = synth_yuv(w, h, n);
    let (probe, caps) = oh264_stream::probe_capacity(&yuv, w, h, n, QP, gop).expect("probe");
    assert_eq!(caps.len(), 2);
    assert!(caps.iter().all(|&c| c > 0), "per-GOP caps {caps:?}");

    // The floored (192×112) clip of the same pixels has strictly less cover.
    let floored = crop_clip(&yuv, w, h, n, 192, 112);
    let (probe_floor, _) = oh264_stream::probe_capacity(&floored, 192, 112, n, QP, gop).expect("probe floor");
    assert!(
        probe.cover_bits > probe_floor.cover_bits,
        "cropped 200x120 cover {} must exceed floored 192x112 cover {}",
        probe.cover_bits,
        probe_floor.cover_bits
    );

    let offered = probe.achievable_max_message_bytes;
    assert!(offered > 0);
    // Incompressible payload (a file of pseudo-random bytes), sized to 90% of
    // the offer, through the CLI two-pass flow: probe caps → plan → encode.
    let blob = |len: usize, seed: u64| -> Vec<FileEntry> {
        let mut s = seed;
        let content = (0..len)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s >> 56) as u8
            })
            .collect();
        vec![FileEntry { filename: "b".into(), content }]
    };
    let encode = |files: &[FileEntry]| -> Result<Vec<u8>, StegoError> {
        let mut s = StreamingEncodeSession::create_with_files(params(w, h, n, gop), "", files, "cap-pass")?;
        s.plan_proportional(&caps, 1.0)?;
        let mut out = Vec::new();
        for f in 0..n as usize {
            s.push_frame(frame_ref(&yuv, w, h, f), &mut out)?;
        }
        s.finish(&mut out)?;
        Ok(out)
    };
    let fits = blob(offered * 9 / 10, 7);
    let stego = encode(&fits).expect("a payload at 90% of the probed capacity must embed");
    let res = oh264_stream::decode_full(&stego, "cap-pass").expect("decode at 90% fill");
    assert_eq!(res.files[0].content, fits[0].content);

    let too_big = blob(offered * 2, 9);
    assert!(
        matches!(encode(&too_big), Err(StegoError::MessageTooLarge)),
        "2x the probed capacity must be refused (probe would be under-reporting)"
    );
}

// ───────────────────────────── MP4 ─────────────────────────────────────

/// Find the first box of type `fourcc` and return its offset (start of size).
fn find_box(mp4: &[u8], fourcc: &[u8; 4]) -> usize {
    mp4.windows(4).position(|w| w == fourcc).expect("box present") - 4
}

/// `tkhd` (16.16 fixed point) and `avc1` (u16) both carry the display size.
fn mp4_dims(mp4: &[u8]) -> ((u32, u32), (u32, u32)) {
    let t = find_box(mp4, b"tkhd");
    let version = mp4[t + 8];
    // size+type(8) + version/flags(4) + times/id/reserved/duration
    // (20 for v0, 32 for v1) + reserved(8) + layer/alt/volume/reserved(8)
    // + matrix(36).
    let off = t + 8 + 4 + if version == 1 { 32 } else { 20 } + 8 + 8 + 36;
    let be32 = |o: usize| u32::from_be_bytes(mp4[o..o + 4].try_into().unwrap());
    let tkhd = (be32(off) >> 16, be32(off + 4) >> 16);
    // The first sample entry sits after the stsd header (8) + version/flags
    // (4) + entry_count (4). (`avc1` also occurs earlier, as an ftyp brand.)
    let a = find_box(mp4, b"stsd") + 16;
    assert_eq!(&mp4[a + 4..a + 8], b"avc1", "first stsd entry must be avc1");
    let be16 = |o: usize| u16::from_be_bytes(mp4[o..o + 2].try_into().unwrap()) as u32;
    let avc1 = (be16(a + 32), be16(a + 34));
    (tkhd, avc1)
}

#[test]
fn exact_size_mp4_track_dims_are_display_size() {
    let _g = guard();
    let (w, h, n, gop) = (200u32, 120u32, 8u32, 4u32);
    let yuv = synth_yuv(w, h, n);
    let stego = oh264_stream::encode_with(&yuv, w, h, n, QP, gop, CostWeights::default(), "mp4 dims", "m")
        .expect("encode");
    let pattern = GopPattern::Ipppp { gop: gop as usize };
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::LavfOpenH264, &stego, w, h, FrameTiming::FPS_30, pattern, n as usize,
    )
    .expect("mux at display size");
    let (tkhd, avc1) = mp4_dims(&mp4);
    assert_eq!(tkhd, (w, h), "tkhd must carry the display size");
    assert_eq!(avc1, (w, h), "avc1 must carry the display size");
    let demuxed = phasm_core::codec::mp4::demux::demux(&mp4).expect("demux");
    let track = demuxed.tracks.iter().find(|t| &t.handler_type == b"vide").expect("video track");
    assert_eq!((track.width, track.height), (w, h));

    // Muxing with the coded (MB-grid) size must be refused, not shipped.
    let g = FrameGeometry::new(w, h).unwrap();
    let err = build_mp4_with_pattern(
        MuxerProfile::LavfOpenH264, &stego, g.coded_width(), g.coded_height(), FrameTiming::FPS_30,
        pattern, n as usize,
    );
    assert!(err.is_err(), "coded-size mux must be rejected");

    // The muxed file still decodes. Rebuild Annex-B the way an MP4→Annex-B
    // extractor does: parameter sets re-inserted before every sync sample.
    let avcc = track.avcc_data.as_ref().expect("avcC");
    let mut annex_b = Vec::new();
    for s in &track.samples {
        if s.is_sync {
            for nal in [&avcc.sps_nalus[0], &avcc.pps_nalus[0]] {
                annex_b.extend_from_slice(&[0, 0, 0, 1]);
                annex_b.extend_from_slice(nal);
            }
        }
        let mut p = 0;
        while p + 4 <= s.data.len() {
            let len = u32::from_be_bytes(s.data[p..p + 4].try_into().unwrap()) as usize;
            annex_b.extend_from_slice(&[0, 0, 0, 1]);
            annex_b.extend_from_slice(&s.data[p + 4..p + 4 + len]);
            p += 4 + len;
        }
    }
    assert_eq!(oh264_stream::decode_text(&annex_b, "m").expect("decode from MP4 samples"), "mp4 dims");
}

// ────────────────────────── visual fidelity ────────────────────────────

/// Cropped output reconstructs the cover as well as the 16-floored encode of
/// the same pixels: the decoder outputs the exact size, luma PSNR of the
/// stego clip over the shared region is on par with the floored clip's, and
/// the partly-padded bottom/right MBs are not a hot spot.
#[test]
fn exact_size_visual_fidelity_matches_floored_encode() {
    let _g = guard();
    let (w, h, n, gop) = (200u32, 120u32, 8u32, 4u32);
    let (fw, fh) = (192u32, 112u32);
    let yuv = synth_yuv(w, h, n);
    let floored = crop_clip(&yuv, w, h, n, fw, fh);
    let msg = "visual fidelity at the exact size";

    let exact = oh264_stream::encode_with(&yuv, w, h, n, QP, gop, CostWeights::default(), msg, "v")
        .expect("exact encode");
    let floor = oh264_stream::encode_with(&floored, fw, fh, n, QP, gop, CostWeights::default(), msg, "v")
        .expect("floored encode");
    let clean = oh264_stream::encode_clean(&yuv, w, h, n, QP, gop);

    let exact_y = decode_luma_exact(&exact, w, h, n, "vis_stego");
    let clean_y = decode_luma_exact(&clean, w, h, n, "vis_clean");
    let floor_y = decode_luma_exact(&floor, fw, fh, n, "vis_floor");

    let src_y = luma_of(&yuv, w, h, n);
    let src_floor_y = luma_of(&floored, fw, fh, n);
    let psnr_exact = psnr_y(&src_y, &exact_y);
    let psnr_clean = psnr_y(&src_y, &clean_y);
    let psnr_floor = psnr_y(&src_floor_y, &floor_y);
    // Shared top-left region of the exact clip, for a like-for-like compare.
    let mut exact_shared = Vec::new();
    let mut src_shared = Vec::new();
    for f in 0..n as usize {
        for row in 0..fh as usize {
            let o = f * (w * h) as usize + row * w as usize;
            exact_shared.extend_from_slice(&exact_y[o..o + fw as usize]);
            src_shared.extend_from_slice(&src_y[o..o + fw as usize]);
        }
    }
    let psnr_exact_shared = psnr_y(&src_shared, &exact_shared);
    let worst_exact = (0..n as usize)
        .map(|f| {
            let o = f * (w * h) as usize;
            worst_mb_psnr_y(&src_y[o..o + (w * h) as usize], &exact_y[o..o + (w * h) as usize], w, h).0
        })
        .fold(f64::INFINITY, f64::min);
    let worst_floor = (0..n as usize)
        .map(|f| {
            let o = f * (fw * fh) as usize;
            worst_mb_psnr_y(&src_floor_y[o..o + (fw * fh) as usize], &floor_y[o..o + (fw * fh) as usize], fw, fh).0
        })
        .fold(f64::INFINITY, f64::min);
    eprintln!(
        "exact {w}x{h}: PSNR-Y stego={psnr_exact:.2} (shared region {psnr_exact_shared:.2}) clean={psnr_clean:.2} \
         worst-MB={worst_exact:.2} | floored {fw}x{fh}: PSNR-Y stego={psnr_floor:.2} worst-MB={worst_floor:.2}"
    );
    assert!(
        psnr_exact_shared >= psnr_floor - 1.0,
        "exact-size stego PSNR over the shared region {psnr_exact_shared:.2} dB is more than 1 dB \
         below the floored encode's {psnr_floor:.2} dB"
    );
    assert!(
        psnr_exact >= psnr_clean - 3.0,
        "exact-size stego PSNR {psnr_exact:.2} dB too far below its own clean encode {psnr_clean:.2} dB"
    );
    assert!(
        worst_exact >= worst_floor - 3.0,
        "worst MB (incl. the partly-padded edge MBs) {worst_exact:.2} dB vs floored {worst_floor:.2} dB"
    );
}
