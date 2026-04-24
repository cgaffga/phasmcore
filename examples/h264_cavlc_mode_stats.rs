// Quick diagnostic: encode N P-frames via CAVLC and dump mode_stats.
// Used 2026-04-23 to measure intra-in-P firing rate on IMG_4138.

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let yuv_path = std::env::var("YUV").unwrap_or_else(|_| "/tmp/img4138_1080p_f90.yuv".into());
    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    enc.entropy_mode = EntropyMode::Cavlc;
    let _ = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    for fi in 1..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        let _ = enc.encode_p_frame(src).unwrap();
    }
}
