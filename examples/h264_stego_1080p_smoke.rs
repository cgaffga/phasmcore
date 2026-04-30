// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! T1.1 — 1080p end-to-end stego round-trip smoke. Encode +
//! decode through the §30D-C 4-domain bitstream-mod CABAC stego
//! pipeline at production scale, sweeping a few payload sizes
//! and GOP layouts.
//!
//! Build with `--features cabac-stego`; without it this compiles
//! to a stub `main` that just prints a message.

#[cfg(not(feature = "cabac-stego"))]
fn main() {
    eprintln!(
        "h264_stego_1080p_smoke requires --features cabac-stego"
    );
}

#[cfg(feature = "cabac-stego")]
use std::time::Instant;

#[cfg(feature = "cabac-stego")]
use phasm_core::{
    h264_stego_decode_yuv_string_4domain,
    h264_stego_encode_yuv_string_4domain_multigop,
    StegoError,
};

#[cfg(feature = "cabac-stego")]
fn main() {
    let yuv_path = "/tmp/img4138_1080p_f10.yuv";
    let yuv = std::fs::read(yuv_path).expect("read 1080p yuv");
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = 10;
    let frame_size = (w * h * 3 / 2) as usize;
    assert!(
        yuv.len() >= n_frames * frame_size,
        "yuv too short: {} vs need {}",
        yuv.len(),
        n_frames * frame_size,
    );
    let yuv_slice = &yuv[..n_frames * frame_size];

    let configs: &[(usize, usize)] =
        &[(8, 10), (100, 10), (200, 5), (500, 10)];

    let mut all_ok = true;
    for &(payload_bytes, gop_size) in configs {
        let msg = "x".repeat(payload_bytes);
        let pass = "smoke-pass";

        let t0 = Instant::now();
        let enc_res = h264_stego_encode_yuv_string_4domain_multigop(
            yuv_slice, w, h, n_frames, gop_size, &msg, pass,
        );
        let enc_s = t0.elapsed().as_secs_f64();

        let bytes = match enc_res {
            Ok(b) => b,
            Err(e) => {
                println!(
                    "payload={}B gop={} encode_s={:.2} ENCODE_FAILED={}",
                    payload_bytes,
                    gop_size,
                    enc_s,
                    fmt_err(&e),
                );
                all_ok = false;
                continue;
            }
        };

        let t1 = Instant::now();
        let dec_res = h264_stego_decode_yuv_string_4domain(&bytes, pass);
        let dec_s = t1.elapsed().as_secs_f64();

        match dec_res {
            Ok(recovered) if recovered == msg => {
                println!(
                    "payload={}B gop={} encode_s={:.2} decode_s={:.2} stego_bytes={}  OK",
                    payload_bytes,
                    gop_size,
                    enc_s,
                    dec_s,
                    bytes.len(),
                );
            }
            Ok(recovered) => {
                println!(
                    "payload={}B gop={} encode_s={:.2} decode_s={:.2} stego_bytes={} \
                     RECOVERED_MISMATCH (got {} bytes vs expected {})",
                    payload_bytes,
                    gop_size,
                    enc_s,
                    dec_s,
                    bytes.len(),
                    recovered.len(),
                    msg.len(),
                );
                all_ok = false;
            }
            Err(e) => {
                println!(
                    "payload={}B gop={} encode_s={:.2} decode_s={:.2} stego_bytes={} \
                     DECODE_FAILED={}",
                    payload_bytes,
                    gop_size,
                    enc_s,
                    dec_s,
                    bytes.len(),
                    fmt_err(&e),
                );
                all_ok = false;
            }
        }
    }

    if all_ok {
        eprintln!("ALL_OK");
    } else {
        eprintln!("SOME_FAILED");
        std::process::exit(1);
    }
}

#[cfg(feature = "cabac-stego")]
fn fmt_err(e: &StegoError) -> String {
    match e {
        StegoError::MessageTooLarge => "MessageTooLarge".into(),
        StegoError::FrameCorrupted => "FrameCorrupted".into(),
        StegoError::DecryptionFailed => "DecryptionFailed".into(),
        StegoError::InvalidVideo(s) => format!("InvalidVideo({s})"),
        StegoError::ShadowEmbedFailed => "ShadowEmbedFailed".into(),
        other => format!("{other:?}"),
    }
}
