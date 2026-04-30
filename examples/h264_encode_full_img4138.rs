// Full-length encode of IMG_4138 at 1920x1072 (cropped from 1080 to be
// 16-aligned), Q=26, GOP=30 with manual IDR insertion every 30 frames.
// Default CABAC + intra-in-P + 8x8-transform + deblock + variance-AQ
// (shipping config as of commit b00f052).

use std::time::Instant;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = 625;
    let gop: usize = 30;
    let q: u8 = 26;
    let yuv_path = "/tmp/img4138_full_1072.yuv";
    let out_path = std::env::var("PHASM_OUT_H264")
        .unwrap_or_else(|_| "/tmp/img4138_phasm_cabac.h264".into());

    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(yuv_path).expect("yuv missing");
    assert_eq!(
        pixels.len(),
        n_frames * frame_size,
        "yuv size mismatch: got {} bytes, expected {} (= {} frames * {})",
        pixels.len(),
        n_frames * frame_size,
        n_frames,
        frame_size
    );

    let mut enc = Encoder::new(w, h, Some(q)).expect("encoder new");
    enc.set_gop_length(gop as u32);

    let mut bytes: Vec<u8> = Vec::with_capacity(64 * 1024 * 1024);

    let t_start = Instant::now();
    for fi in 0..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        let is_idr = fi % gop == 0;
        let frame_bytes = if is_idr {
            enc.encode_i_frame(src).expect("encode_i_frame")
        } else {
            enc.encode_p_frame(src).expect("encode_p_frame")
        };
        bytes.extend_from_slice(&frame_bytes);
        if fi % 30 == 0 || fi == n_frames - 1 {
            let elapsed = t_start.elapsed().as_secs_f64();
            let fps = (fi + 1) as f64 / elapsed.max(1e-9);
            let kind = if is_idr { "I" } else { "P" };
            println!(
                "frame {fi:4}/{n_frames} ({kind})  elapsed {elapsed:7.1}s  avg {fps:5.2} fps  bytes_so_far {} MB",
                bytes.len() / (1024 * 1024)
            );
        }
    }
    let wall = t_start.elapsed();

    std::fs::write(&out_path, &bytes).expect("write .h264");
    let out_size = bytes.len();
    println!(
        "\nDONE: wrote {out_path}  size {} bytes ({:.1} MB)  wall {:.1}s  avg {:.2} fps",
        out_size,
        out_size as f64 / 1_000_000.0,
        wall.as_secs_f64(),
        n_frames as f64 / wall.as_secs_f64()
    );
}
