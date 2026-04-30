// Generic full-length encode. Reads W, H, N, Q, GOP, YUV, OUT from env.
//
// Example:
//   PHASM_W=1280 PHASM_H=720 PHASM_N=1846 PHASM_Q=26 PHASM_GOP=30 \
//     PHASM_YUV=/tmp/foo.yuv PHASM_OUT_H264=/tmp/out.h264 \
//     cargo run --release --features video,h264-encoder --example h264_encode_generic

use std::time::Instant;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn env<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|s| s.parse().ok())
}

fn main() {
    let w: u32 = env("PHASM_W").expect("PHASM_W");
    let h: u32 = env("PHASM_H").expect("PHASM_H");
    let n_frames: usize = env("PHASM_N").expect("PHASM_N");
    let gop: usize = env("PHASM_GOP").unwrap_or(30);
    let q: u8 = env("PHASM_Q").unwrap_or(26);
    let yuv_path = std::env::var("PHASM_YUV").expect("PHASM_YUV");
    let out_path = std::env::var("PHASM_OUT_H264").expect("PHASM_OUT_H264");

    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    assert_eq!(
        pixels.len(), n_frames * frame_size,
        "yuv size mismatch: got {} bytes, expected {} ({} × {} × 3/2 × {} frames)",
        pixels.len(), n_frames * frame_size, w, h, n_frames,
    );

    println!("{}×{}  {} frames  q={}  gop={}  out={}", w, h, n_frames, q, gop, out_path);

    let mut enc = Encoder::new(w, h, Some(q)).expect("encoder new");
    enc.set_gop_length(gop as u32);

    let mut bytes: Vec<u8> = Vec::with_capacity(64 * 1024 * 1024);
    let t_start = Instant::now();
    for fi in 0..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        let frame_bytes = if fi % gop == 0 {
            enc.encode_i_frame(src).expect("encode_i_frame")
        } else {
            enc.encode_p_frame(src).expect("encode_p_frame")
        };
        bytes.extend_from_slice(&frame_bytes);
        if fi % 60 == 0 || fi == n_frames - 1 {
            let el = t_start.elapsed().as_secs_f64();
            println!(
                "  frame {fi:5}/{n_frames}  elapsed {el:7.1}s  fps {:.2}  bytes {} MB",
                (fi + 1) as f64 / el.max(1e-9),
                bytes.len() / (1024 * 1024),
            );
        }
    }
    std::fs::write(&out_path, &bytes).expect("write .h264");
    let wall = t_start.elapsed();
    println!(
        "DONE {}  size {} bytes ({:.1} MB)  wall {:.1}s",
        out_path, bytes.len(), bytes.len() as f64 / 1_000_000.0, wall.as_secs_f64(),
    );
}
