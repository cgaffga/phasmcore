// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

#![cfg(feature = "h264-encoder")]

//! Shared OH264 streaming-session test harness.
//!
//! Replaces the retired one-shot `openh264_stego_encode_yuv_*` /
//! `openh264_stego_decode_yuv_*` / `openh264_stego_capacity_yuv`
//! helpers. The video-retirement Phase 6 deleted the legacy single-domain
//! (CoeffSignBypass-only) one-shot path, so integration tests drive the
//! production 4-domain `StreamingEncodeSession` / `StreamingProbeSession`
//! through these thin wrappers instead — same `(yuv, message, passphrase)`
//! ergonomics, exercising the path that actually ships.
//!
//! `yuv` is a flat I420 buffer of `n_frames` concatenated frames
//! (Y plane `w*h`, then U, then V, each chroma plane `(w/2)*(h/2)`).
//!
//! Text and file attachments share one wire path: files are just payload
//! bytes (`create_with_files` → `decode_full().files`), so the stego
//! channel is identical either way.

use phasm_core::codec::h264::stego::CostWeights;
use phasm_core::codec::h264::streaming_session::{
    ColorParams, DecodeSessionResult, EncodeEngineChoice, EncodeSessionParams,
    StreamingDecodeSession, StreamingEncodeSession, StreamingProbeSession, YuvFrameRef,
};
use phasm_core::stego::payload::FileEntry;
use phasm_core::StegoError;

/// Default QP the retired one-shot path used for its fixtures.
pub const DEFAULT_QP: i32 = 26;

/// Extract the Y/U/V plane slices for frame `f` from a flat I420 clip.
pub fn frame_planes(yuv: &[u8], width: u32, height: u32, f: usize) -> (&[u8], &[u8], &[u8]) {
    let fw = width as usize;
    let fh = height as usize;
    let cw = fw / 2;
    let ch = fh / 2;
    let y_sz = fw * fh;
    let c_sz = cw * ch;
    let frame_sz = y_sz + 2 * c_sz;
    let base = f * frame_sz;
    let y = &yuv[base..base + y_sz];
    let u = &yuv[base + y_sz..base + y_sz + c_sz];
    let v = &yuv[base + y_sz + c_sz..base + frame_sz];
    (y, u, v)
}

fn frame_ref<'a>(width: u32, planes: (&'a [u8], &'a [u8], &'a [u8])) -> YuvFrameRef<'a> {
    let fw = width as usize;
    YuvFrameRef {
        y: planes.0,
        y_stride: fw,
        u: planes.1,
        u_stride: fw / 2,
        v: planes.2,
        v_stride: fw / 2,
    }
}

fn params(
    width: u32,
    height: u32,
    n_frames: u32,
    qp: i32,
    gop_size: u32,
    cost_weights: CostWeights,
) -> EncodeSessionParams {
    EncodeSessionParams {
        width,
        height,
        fps_num: 30,
        fps_den: 1,
        qp,
        gop_size,
        total_frames_hint: n_frames,
        color: ColorParams::default(),
        engine: EncodeEngineChoice::Oh264,
        cost_weights,
        progress_callback: None,
    }
}

fn push_clip(
    mut enc: StreamingEncodeSession,
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
) -> Result<Vec<u8>, StegoError> {
    let mut out = Vec::new();
    for f in 0..n_frames as usize {
        enc.push_frame(frame_ref(width, frame_planes(yuv, width, height, f)), &mut out)?;
    }
    enc.finish(&mut out)?;
    Ok(out)
}

/// Encode a flat I420 clip into stego Annex-B via the OH264 streaming
/// session. `gop_size` drives the per-GOP STC chunk schedule; pass
/// `n_frames` for a single-GOP encode.
pub fn encode_with(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    qp: i32,
    gop_size: u32,
    cost_weights: CostWeights,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    let enc = StreamingEncodeSession::create(
        params(width, height, n_frames, qp, gop_size, cost_weights),
        message,
        passphrase,
    )?;
    push_clip(enc, yuv, width, height, n_frames)
}

/// Single-GOP encode with default cost weights — the common case for the
/// migrated one-shot fixtures (whole clip in one GOP).
pub fn encode(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    qp: i32,
    message: &str,
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    encode_with(
        yuv,
        width,
        height,
        n_frames,
        qp,
        n_frames.max(1),
        CostWeights::default(),
        message,
        passphrase,
    )
}

/// Encode a message + file attachments. Files ride the same per-GOP path
/// as text — they are just payload bytes — so this is `encode` with a
/// non-empty `files` list.
pub fn encode_with_files(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    qp: i32,
    message: &str,
    files: &[FileEntry],
    passphrase: &str,
) -> Result<Vec<u8>, StegoError> {
    // The per-GOP `create_with_files` path — the same entry the bridges'
    // video file-attachment FFI uses. Files ride the per-GOP path as plain
    // payload bytes (no whole-clip buffering).
    let enc = StreamingEncodeSession::create_with_files(
        params(width, height, n_frames, qp, n_frames.max(1), CostWeights::default()),
        message,
        files,
        passphrase,
    )?;
    push_clip(enc, yuv, width, height, n_frames)
}

/// Decode the full result (text + file attachments) from stego Annex-B.
pub fn decode_full(annex_b: &[u8], passphrase: &str) -> Result<DecodeSessionResult, StegoError> {
    let mut dec = StreamingDecodeSession::create(passphrase)?;
    dec.push_annex_b(annex_b)?;
    dec.finish()
}

/// Decode just the message text from stego Annex-B.
pub fn decode_text(annex_b: &[u8], passphrase: &str) -> Result<String, StegoError> {
    decode_full(annex_b, passphrase).map(|r| r.text)
}

/// Cover-bits capacity of a clip via the production capacity probe
/// (replaces the deleted single-domain `openh264_stego_capacity_yuv`).
pub fn probe_cover_bits(
    yuv: &[u8],
    width: u32,
    height: u32,
    n_frames: u32,
    qp: i32,
    gop_size: u32,
) -> Result<usize, StegoError> {
    let mut probe = StreamingProbeSession::create(params(
        width,
        height,
        n_frames,
        qp,
        gop_size,
        CostWeights::default(),
    ))?;
    for f in 0..n_frames as usize {
        probe.push_frame(frame_ref(width, frame_planes(yuv, width, height, f)))?;
    }
    Ok(probe.finish()?.cover_bits)
}
