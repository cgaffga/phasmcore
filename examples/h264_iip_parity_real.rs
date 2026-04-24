// Real-world parity test: 128x80 N-frame from img4138 YUV.
// Reports per-frame diff count so we can see if intra-in-P causes drift
// on real content.

use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = 128;
    let h: u32 = 80;
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(10);
    let yuv_path = std::env::var("YUV").unwrap_or_else(|_| {
        "/Users/cgaffga/Development/phasm/core/test-vectors/video/h264/real-world/img4138_128x80_f10.yuv".into()
    });
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let c_size = y_size / 4;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    assert!(
        pixels.len() >= n_frames * frame_size,
        "yuv too short: {} vs need {}",
        pixels.len(),
        n_frames * frame_size
    );

    let intra_on = std::env::var_os("PHASM_CABAC_INTRA_IN_P").is_some();
    println!("intra-in-P: {}", if intra_on { "ENABLED" } else { "disabled" });

    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    let mut enc_recons = vec![(enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone())];

    for fi in 1..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        bytes.extend_from_slice(&enc.encode_p_frame(src).unwrap());
        enc_recons.push((enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone()));
    }

    std::fs::write("/tmp/iip_real_parity.h264", &bytes).unwrap();
    Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error",
            "-f", "h264", "-i", "/tmp/iip_real_parity.h264",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "/tmp/iip_real_parity.yuv",
        ])
        .output()
        .unwrap();
    let decoded = std::fs::read("/tmp/iip_real_parity.yuv").unwrap();
    let n_check = n_frames.min(decoded.len() / frame_size);

    println!("frame | Y-diffs | Cb-diffs | Cr-diffs | Y max-|Δ|");
    for fi in 0..n_check {
        let (ey, ecb, ecr) = &enc_recons[fi];
        let base = fi * frame_size;
        let dy = &decoded[base..base + y_size];
        let dcb = &decoded[base + y_size..base + y_size + c_size];
        let dcr = &decoded[base + y_size + c_size..base + frame_size];
        let mut yd = 0;
        let mut ym = 0i32;
        for i in 0..y_size {
            let d = ey[i] as i32 - dy[i] as i32;
            if d != 0 {
                yd += 1;
                if d.abs() > ym { ym = d.abs(); }
            }
        }
        let cbd = ecb.iter().zip(dcb.iter()).filter(|(a, b)| a != b).count();
        let crd = ecr.iter().zip(dcr.iter()).filter(|(a, b)| a != b).count();
        println!("  {:3}  |  {:5}  |   {:4}   |   {:4}   |   {}", fi, yd, cbd, crd, ym);
    }
}
