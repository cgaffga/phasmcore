// Full-length encode of IMG_4273 at 1920x1072 (cropped), Q=26,
// GOP=30 with manual IDR every 30 frames. Reads the RDO env knobs
// the same way the encoder's runtime does.

use std::time::Instant;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = 714;
    let gop: usize = 30;
    let q: u8 = 26;
    let yuv_path = "/tmp/img4273_full_1072.yuv";
    let out_path = std::env::var("PHASM_OUT_H264")
        .unwrap_or_else(|_| "/tmp/img4273_phasm_cabac.h264".into());

    let frame_size = (w * h * 3 / 2) as usize;
    let pixels = std::fs::read(yuv_path).expect("yuv missing");
    assert_eq!(
        pixels.len(),
        n_frames * frame_size,
        "yuv size mismatch: got {} bytes, expected {}",
        pixels.len(),
        n_frames * frame_size,
    );

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
        if fi % 30 == 0 || fi == n_frames - 1 {
            let elapsed = t_start.elapsed().as_secs_f64();
            println!(
                "  frame {fi:4}/{n_frames}  elapsed {elapsed:7.1}s  bytes {} MB",
                bytes.len() / (1024 * 1024)
            );
        }
    }
    let wall = t_start.elapsed();
    std::fs::write(&out_path, &bytes).expect("write .h264");
    println!(
        "DONE {out_path}  size {} bytes ({:.1} MB)  wall {:.1}s",
        bytes.len(),
        bytes.len() as f64 / 1_000_000.0,
        wall.as_secs_f64(),
    );
}
