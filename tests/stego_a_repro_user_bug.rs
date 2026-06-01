// Diagnostic: reproduce user's bug — encode with empty primary passphrase + shadow "s",
// decode with "s" should recover shadow.

#![cfg(all(
    feature = "h264-encoder",
    feature = "openh264-backend",
    feature = "cabac-stego",
))]

use phasm_core::codec::h264::openh264_stego::{
    encode_yuv_with_n_shadows_with_pattern_and_files, EncodeOpts,
};
use phasm_core::codec::h264::stego::cost_weights::CostWeights;
use phasm_core::h264_stego_smart_decode_video;
use phasm_core::stego::shadow_layer::ShadowLayer;
use std::sync::{Mutex, OnceLock};

static SESSION_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
fn session_guard() -> &'static Mutex<()> {
    SESSION_GUARD.get_or_init(|| Mutex::new(()))
}

fn synth_yuv(width: u32, height: u32, n_frames: u32) -> Vec<u8> {
    let fs = (width * height * 3 / 2) as usize;
    let mut out = Vec::with_capacity(fs * n_frames as usize);
    let w = width as i32;
    let h = height as i32;
    let half_w = w / 2;
    let half_h = h / 2;
    for f in 0..n_frames {
        for j in 0..h {
            for i in 0..w {
                out.push(((i + f as i32 * 2) ^ (j + f as i32 * 3)) as u8);
            }
        }
        let mut s: u32 = 0xCAFE_F00D ^ f;
        for _ in 0..2 {
            for j in 0..half_h {
                for i in 0..half_w {
                    s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                    out.push(((s >> 16) as u8).wrapping_add((i + j + f as i32) as u8));
                }
            }
        }
    }
    out
}

/// User's reported scenario: empty primary passphrase, shadow passphrase "s".
/// Decode with "s" should recover shadow message.
#[test]
fn user_bug_empty_primary_pass_shadow_s() {
    let _g = session_guard().lock().unwrap_or_else(|e| e.into_inner());
    let width: u32 = 480;
    let height: u32 = 272;
    let n_frames: u32 = 10;
    let yuv = synth_yuv(width, height, n_frames);
    let opts = EncodeOpts { qp: 26, intra_period: 5 };
    let weights = CostWeights::default();

    let primary_msg = "my primary text";
    let primary_pass = "";              // user reported: NO passphrase for primary
    let shadow_msg = "my shadow text";
    let shadow_pass = "s";

    let shadows = [ShadowLayer {
        message: shadow_msg,
        passphrase: shadow_pass,
        files: &[],
    }];

    let bytes = match encode_yuv_with_n_shadows_with_pattern_and_files(
        &yuv, width, height, n_frames, opts,
        primary_msg, &[], primary_pass,
        &shadows,
        &weights,
    ) {
        Ok(b) => b,
        Err(e) => panic!("encode failed: {e}"),
    };
    eprintln!("encode produced {} bytes", bytes.len());

    // Decode primary
    let primary_dec = h264_stego_smart_decode_video(&bytes, primary_pass);
    eprintln!("primary decode (pass=''): {:?}", primary_dec);

    // Decode shadow
    let shadow_dec = h264_stego_smart_decode_video(&bytes, shadow_pass);
    eprintln!("shadow decode (pass='s'): {:?}", shadow_dec);

    // Wrong passphrase
    let wrong_dec = h264_stego_smart_decode_video(&bytes, "wrong");
    eprintln!("wrong decode (pass='wrong'): {:?}", wrong_dec);

    assert_eq!(shadow_dec.as_ref().ok().map(|s| s.as_str()), Some(shadow_msg),
        "shadow decode must work — got {:?}", shadow_dec);
    assert_eq!(primary_dec.as_ref().ok().map(|s| s.as_str()), Some(primary_msg),
        "primary decode must work — got {:?}", primary_dec);
    assert!(wrong_dec.is_err(), "wrong passphrase must reject");
}
