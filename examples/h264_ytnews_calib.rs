// Minimal harness for Phase D.2-stealth calibration task #54.
// Encodes 10 frames of ytnews (1280×720) through the CABAC P-path so the
// mode_stats dump can be compared against x264.
//
// Inputs:
//   /tmp/ytnews_720p_f10.yuv  — 10 frames of raw yuv420p (1280×720).
// Outputs (paths env-overridable for parallel sweeps):
//   PHASM_OUT_H264=/tmp/ours_ytnews.h264
//
// Set PHASM_MODE_STATS=1 to print per-frame intra-in-P split to stderr.

use std::fs;
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

fn main() {
    let w: u32 = 1280;
    let h: u32 = 720;
    let fsize = (w * h * 3 / 2) as usize;
    let pixels = fs::read("/tmp/ytnews_720p_f10.yuv")
        .expect("ytnews yuv missing (expected /tmp/ytnews_720p_f10.yuv)");
    assert!(pixels.len() >= 10 * fsize, "yuv too short");

    let q: u8 = 80;
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let mut bytes = enc.encode_i_frame(&pixels[..fsize]).unwrap();
    for fi in 1..10 {
        let src = &pixels[fi * fsize..(fi + 1) * fsize];
        bytes.extend_from_slice(&enc.encode_p_frame(src).unwrap());
    }
    let h264_path = std::env::var("PHASM_OUT_H264")
        .unwrap_or_else(|_| "/tmp/ours_ytnews.h264".to_string());
    fs::write(&h264_path, &bytes).unwrap();
    eprintln!("ytnews encode done: {} bytes", bytes.len());
}
