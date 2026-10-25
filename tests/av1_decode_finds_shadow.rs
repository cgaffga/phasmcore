// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! One AV1 decode call finds whichever message the passphrase opens.
//!
//! Until 2026-10-24 every production AV1 decoder (CLI `decode`, iOS
//! `av1DecodeStreaming`, Android `av1DecodeStreaming`) called only
//! `finish_primary_payload`, so a shadow message, the plausible-deniability
//! layer, could be embedded but never read back. H.264's decode session
//! falls back to its shadows on its own; `finish_payload` gives AV1 the
//! same: the primary, else the whole-video shadow, else a per-GOP shadow.

#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

use phasm_core::Av1StreamingEncodeParams;
use phasm_core::codec::av1::stego::session::Av1StreamingDecodeSession;
use phasm_core::codec::av1::stego::whole_video::{
    Av1SliceYuvSource, Pass1CacheBudget, av1_encode_with_shadows_streaming,
};
use phasm_core::stego::{crypto, frame, payload};

const PRIMARY_PASS: &str = "finds-primary";
const SHADOW_PASS: &str = "finds-shadow";
const PRIMARY_TEXT: &str = "the primary message";
// Starts with 's': its byte, read raw as payload flags, failed the encode
// before the apps framed shadows (daddcd3a).
const SHADOW_TEXT: &str = "secret shadow message";

/// Moving noise: real texture and motion, so 8 frames carry both layers.
fn synthetic_clip(w: u32, h: u32, n: u32) -> Vec<u8> {
    let (w, h, n) = (w as usize, h as usize, n as usize);
    let (cw, ch) = (w / 2, h / 2);
    let (tw, th) = (w + 2 * n, h + n);
    let mut s: u32 = 0x2545_F491;
    let tex: Vec<u8> = (0..tw * th)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            (s >> 24) as u8
        })
        .collect();
    let mut out = Vec::with_capacity((w * h + 2 * cw * ch) * n);
    for f in 0..n {
        for y in 0..h {
            out.extend((0..w).map(|x| tex[(y + f) * tw + x + 2 * f]));
        }
        out.extend(std::iter::repeat_n(128u8, 2 * cw * ch));
    }
    out
}

fn stego_clip() -> Vec<u8> {
    let (w, h, gop, n) = (256u32, 144u32, 4u32, 8u32);
    let params = Av1StreamingEncodeParams {
        width: w,
        height: h,
        quantizer: 60,
        gop_size: gop,
        total_frames_hint: n,
    };
    let yuv = synthetic_clip(w, h, n);
    let primary = payload::encode_payload(PRIMARY_TEXT, &[]).unwrap();
    let (ct, nonce, salt) = crypto::encrypt(&primary, PRIMARY_PASS).unwrap();
    let framed = frame::build_frame(primary.len(), &salt, &nonce, &ct);
    let shadow = payload::encode_payload(SHADOW_TEXT, &[]).unwrap();
    let shadows: &[(&str, &[u8])] = &[(SHADOW_PASS, &shadow)];
    let mut src = Av1SliceYuvSource::new(&yuv, w, h, n, gop);
    av1_encode_with_shadows_streaming(
        &mut src,
        n,
        params,
        &framed,
        PRIMARY_PASS,
        shadows,
        16,
        Pass1CacheBudget::from_env_or_default(),
    )
    .expect("encode with a shadow")
}

fn decode(stego: &[u8], pass: &str) -> Result<String, String> {
    let mut session = Av1StreamingDecodeSession::create(pass);
    session.push_bytes(stego);
    session.finish_payload().map(|p| p.text).map_err(|e| format!("{e:?}"))
}

#[test]
fn finish_payload_opens_primary_or_shadow_by_passphrase() {
    let stego = stego_clip();
    assert_eq!(decode(&stego, PRIMARY_PASS).as_deref(), Ok(PRIMARY_TEXT));
    assert_eq!(decode(&stego, SHADOW_PASS).as_deref(), Ok(SHADOW_TEXT));
    assert!(decode(&stego, "wrong passphrase").is_err());
}
