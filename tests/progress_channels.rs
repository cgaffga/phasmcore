// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Per-channel progress and cancel (`progress::Channel`, `progress::run_decode`).
//!
//! The mobile apps run an encode (or capacity probe) and a decode at once, one
//! per tab. With one global flag, the decode tab's Cancel made the encode
//! tab's STC searches find "no path" (the encode was refused as too large, the
//! probe under-reported), a decode starting cleared the encode's pending
//! cancel, and each overlay showed the other's progress. The bridges now run
//! decodes through `run_decode`, whose work — including every rayon region
//! inside it — reads the decode channel only.
//!
//! Its own binary, serialized: the channels are process-global, and the lib's
//! unit tests run STC searches on the encode channel.

#![cfg(feature = "parallel")]

use std::sync::{Mutex, MutexGuard};

use phasm_core::progress::{self, Channel, run_decode};
use phasm_core::stego::error::StegoError;
use phasm_core::stego::stc::embed::{stc_embed, stc_is_feasible};
use phasm_core::stego::stc::hhat::generate_hhat;
use rayon::prelude::*;

static SERIAL: Mutex<()> = Mutex::new(());

/// Serialize the tests and start each from clean flags.
fn serial() -> MutexGuard<'static, ()> {
    let guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    Channel::Encode.clear_cancel();
    Channel::Decode.clear_cancel();
    guard
}

const H: usize = 7;

fn stc_instance() -> (Vec<u8>, Vec<f32>, Vec<u8>, Vec<Vec<u32>>, usize) {
    let w = 4;
    let n_msg = 512;
    let n = n_msg * w;
    let cover: Vec<u8> = (0..n).map(|i| ((i * 7 + 3) % 5 % 2) as u8).collect();
    let costs: Vec<f32> = (0..n).map(|i| 1.0 + (i % 13) as f32).collect();
    let msg: Vec<u8> = (0..n_msg).map(|i| ((i * 31 + 1) % 3 % 2) as u8).collect();
    let hhat = generate_hhat(H, w, &[0x5A; 32]);
    (cover, costs, msg, hhat, w)
}

#[test]
fn a_decode_cancel_leaves_encode_stc_alone() {
    let _g = serial();
    let (cover, costs, msg, hhat, w) = stc_instance();
    let fits = || stc_is_feasible(&cover, &costs, &msg, &hhat, H, w);

    // The decode tab's Cancel while the encode tab probes / encodes.
    Channel::Decode.cancel();
    assert!(fits(), "an encode-side STC search must not see a decode cancel");
    assert!(stc_embed(&cover, &costs, &msg, &hhat, H, w).is_some());
    assert!(!run_decode(fits), "decode-side work must see its own cancel");

    // And the other way round.
    Channel::Decode.clear_cancel();
    Channel::Encode.cancel();
    assert!(!fits());
    assert!(run_decode(fits), "decode-side work must not see an encode cancel");
    Channel::Encode.clear_cancel();
}

#[test]
fn parallel_regions_inside_run_decode_stay_on_the_decode_channel() {
    let _g = serial();
    let all_are = |want: Channel| {
        (0..20_000u32)
            .into_par_iter()
            .map(|i| {
                // Some work, so jobs spread over the workers and get stolen.
                let mut x = i;
                for _ in 0..200 {
                    x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                }
                (x, Channel::current())
            })
            .all(|(_, c)| c == want)
    };
    assert!(run_decode(|| all_are(Channel::Decode)));
    assert!(all_are(Channel::Encode));

    // Both at once, as with an encode in one tab and a decode in the other:
    // the shared global pool never runs decode jobs, the decode pool never
    // runs encode jobs.
    let (enc, dec) = std::thread::scope(|s| {
        let enc = s.spawn(|| (0..20).all(|_| all_are(Channel::Encode)));
        let dec = s.spawn(|| run_decode(|| (0..20).all(|_| all_are(Channel::Decode))));
        (enc.join().unwrap(), dec.join().unwrap())
    });
    assert!(enc, "an encode-side job ran as decode");
    assert!(dec, "a decode-side job ran as encode");
}

#[test]
fn counters_and_init_are_per_channel() {
    let _g = serial();
    progress::init(5);
    progress::advance();
    Channel::Encode.cancel();

    // A decode starting and moving along.
    run_decode(|| {
        progress::init(10);
        progress::advance();
        progress::advance();
    });
    assert_eq!(Channel::Decode.get(), (2, 10));
    assert_eq!(Channel::Encode.get(), (1, 5), "a decode must not move the encode's bar");
    assert!(Channel::Encode.is_cancelled(), "a decode's init dropped the encode's cancel");
    Channel::Encode.clear_cancel();
}

#[test]
fn a_handed_over_channel_keeps_a_cancel_across_init() {
    let _g = serial();
    // What the mobile bridges do: the app clears the flag itself when an
    // operation starts and ends, so a cancel that lands before the
    // pipeline's init — or before a retry re-inits — must survive it.
    Channel::Decode.hand_over();
    run_decode(|| {
        progress::cancel();
        progress::init(4);
        assert!(progress::is_cancelled());
        assert!(matches!(progress::check_cancelled(), Err(StegoError::Cancelled)));
    });
    Channel::Decode.clear_cancel();
    assert!(!run_decode(progress::is_cancelled));
}

#[test]
fn an_image_decode_follows_its_own_channel() {
    let _g = serial();
    let cover = std::fs::read("test-vectors/image/photo_320x240_q75_420.jpg").unwrap();
    let (msg, pass) = ("per-channel cancel", "pw-channels");

    // An encode is not stopped by a decode cancel.
    Channel::Decode.cancel();
    let stego = phasm_core::ghost_encode(&cover, msg, pass).expect("encode under a decode cancel");
    Channel::Decode.clear_cancel();

    // A decode is not stopped by an encode cancel.
    Channel::Encode.cancel();
    let (decoded, _) = run_decode(|| phasm_core::smart_decode(&stego, pass))
        .expect("decode under an encode cancel");
    assert_eq!(decoded.text, msg);
    Channel::Encode.clear_cancel();

    // But by its own (the bridges hand the channel over, so init keeps it).
    Channel::Decode.hand_over();
    Channel::Decode.cancel();
    let res = run_decode(|| phasm_core::smart_decode(&stego, pass));
    assert!(matches!(res, Err(StegoError::Cancelled)), "got {:?}", res.map(|r| r.0.text));
    Channel::Decode.clear_cancel();
}

#[cfg(feature = "h264-encoder")]
mod video {
    use super::*;
    use phasm_core::codec::h264::stego::CostWeights;
    use phasm_core::codec::h264::streaming_session::{
        ColorParams, EncodeEngineChoice, EncodeSessionParams, StreamingDecodeSession,
        StreamingEncodeSession, YuvFrameRef,
    };

    /// Same textured YUV the streaming-session tests use.
    fn synth_yuv(width: u32, height: u32, frame_idx: u32) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = Vec::with_capacity((width * height) as usize);
        for j in 0..height {
            for i in 0..width {
                y.push(((i + frame_idx * 2) ^ (j + frame_idx * 3)) as u8);
            }
        }
        let (half_w, half_h) = (width / 2, height / 2);
        let mut u = Vec::with_capacity((half_w * half_h) as usize);
        let mut v = Vec::with_capacity((half_w * half_h) as usize);
        let mut s: u32 = 0xCAFE_F00D ^ frame_idx;
        for j in 0..half_h {
            for i in 0..half_w {
                let pos = (i + j + frame_idx) as u8;
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                u.push(((s >> 16) as u8).wrapping_add(pos));
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                v.push(((s >> 16) as u8).wrapping_add(pos));
            }
        }
        (y, u, v)
    }

    fn encode_clip(msg: &str, pass: &str) -> Vec<u8> {
        let (width, height, gop, frames) = (320u32, 240u32, 10u32, 20u32);
        let params = EncodeSessionParams {
            width,
            height,
            fps_num: 30,
            fps_den: 1,
            qp: 26,
            gop_size: gop,
            total_frames_hint: frames,
            color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264,
            cost_weights: CostWeights::default(),
            progress_callback: None,
        };
        let mut enc = StreamingEncodeSession::create(params, msg, pass).expect("encode create");
        let mut annex_b = Vec::new();
        for f in 0..frames {
            let (y, u, v) = synth_yuv(width, height, f);
            let frame = YuvFrameRef {
                y: &y,
                y_stride: width as usize,
                u: &u,
                u_stride: (width / 2) as usize,
                v: &v,
                v_stride: (width / 2) as usize,
            };
            enc.push_frame(frame, &mut annex_b).expect("push frame");
        }
        enc.finish(&mut annex_b).expect("finish encode");
        annex_b
    }

    /// A cancel that lands while an encode is planning a GOP. The STC search
    /// reads the flag as "no path"; the encode used to take that for "does
    /// not fit", shrink the chunk and re-encode the GOP step by step, and end
    /// in `MessageTooLarge` — seconds later, with the wrong error.
    #[test]
    fn a_cancelled_video_encode_stops_and_says_so() {
        use phasm_core::codec::h264::progress::{EncodePhase, EncodeProgressCallback};
        use std::sync::Arc;
        let _g = serial();
        let (width, height, gop, frames) = (320u32, 240u32, 10u32, 20u32);
        let cancel_at_plan: EncodeProgressCallback = Arc::new(|phase| {
            if matches!(phase, EncodePhase::StcPlan { .. }) {
                Channel::Encode.cancel();
            }
        });
        let params = EncodeSessionParams {
            width,
            height,
            fps_num: 30,
            fps_den: 1,
            qp: 26,
            gop_size: gop,
            total_frames_hint: frames,
            color: ColorParams::default(),
            engine: EncodeEngineChoice::Oh264,
            cost_weights: CostWeights::default(),
            progress_callback: Some(cancel_at_plan),
        };
        let mut enc = StreamingEncodeSession::create(params, "cancel the encode", "pw-enc")
            .expect("encode create");
        let mut annex_b = Vec::new();
        let mut outcome = Ok(());
        for f in 0..frames {
            let (y, u, v) = synth_yuv(width, height, f);
            let frame = YuvFrameRef {
                y: &y,
                y_stride: width as usize,
                u: &u,
                u_stride: (width / 2) as usize,
                v: &v,
                v_stride: (width / 2) as usize,
            };
            outcome = enc.push_frame(frame, &mut annex_b);
            if outcome.is_err() {
                break;
            }
        }
        let outcome = outcome.and_then(|_| enc.finish(&mut annex_b));
        Channel::Encode.clear_cancel();
        assert!(
            matches!(outcome, Err(StegoError::Cancelled)),
            "a cancelled encode must report Cancelled, got {outcome:?}"
        );
    }

    fn decode(annex_b: &[u8], pass: &str) -> Result<String, StegoError> {
        let mut dec = StreamingDecodeSession::create(pass)?;
        dec.push_annex_b(annex_b)?;
        run_decode(move || dec.finish()).map(|r| r.text)
    }

    /// The video decode used to be one blocking call that never looked at
    /// the flag: a Cancel waited for the whole walk + brute force, and a
    /// cancel inside it came back as "wrong passphrase".
    #[test]
    fn a_cancelled_video_decode_stops_and_says_so() {
        let _g = serial();
        let (msg, pass) = ("cancel me", "pw-video");
        let clip = encode_clip(msg, pass);

        Channel::Encode.cancel();
        assert_eq!(decode(&clip, pass).expect("an encode cancel must not stop a decode"), msg);
        Channel::Encode.clear_cancel();

        Channel::Decode.cancel();
        assert!(matches!(decode(&clip, pass), Err(StegoError::Cancelled)));
        // A wrong passphrase walks the primary search AND the shadow scope;
        // cancelled, it must not come back as DecryptionFailed.
        assert!(matches!(decode(&clip, "not-the-passphrase"), Err(StegoError::Cancelled)));
        Channel::Decode.clear_cancel();

        assert!(matches!(decode(&clip, "not-the-passphrase"), Err(StegoError::DecryptionFailed)));
    }
}
