// Force intra-in-P on a specific 1080p MB and compare per-MB enc-vs-dec.
// Prints the first diverging MB's diff grid for targeted debug.
//
// Usage:
//   PHASM_FORCE_INTRA_IN_P_MB=36,42 Q=80 N=3 \
//     cargo run --release --features video,h264-encoder \
//     --example h264_iip_1080p_force

use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    if std::env::var_os("PHASM_CABAC_INTRA_IN_P").is_none() {
        unsafe { std::env::set_var("PHASM_CABAC_INTRA_IN_P", "1"); }
    }
    let force_mb = std::env::var("PHASM_FORCE_INTRA_IN_P_MB").unwrap_or_else(|_| "36,42".into());
    let (force_x, force_y) = {
        let mut p = force_mb.split(',');
        (
            p.next().unwrap().parse::<usize>().unwrap(),
            p.next().unwrap().parse::<usize>().unwrap(),
        )
    };
    unsafe {
        std::env::set_var("PHASM_FORCE_INTRA_IN_P_MB", &force_mb);
        std::env::set_var("PHASM_CABAC_INTRA_IN_P_FORCE_MB", &force_mb);
    }
    println!("Forcing intra-in-P on MB ({},{})", force_x, force_y);

    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let yuv_path = std::env::var("YUV").unwrap_or_else(|_| format!("/tmp/img4138_1080p_f{}.yuv", n_frames));
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let _c_size = y_size / 4;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");

    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    let mut enc_recons = vec![(enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone())];
    for fi in 1..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        bytes.extend_from_slice(&enc.encode_p_frame(src).unwrap());
        enc_recons.push((enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone()));
    }

    std::fs::write("/tmp/iip_1080_force.h264", &bytes).unwrap();
    Command::new("ffmpeg").args([
        "-y", "-loglevel", "error", "-f", "h264", "-i", "/tmp/iip_1080_force.h264",
        "-f", "rawvideo", "-pix_fmt", "yuv420p", "/tmp/iip_1080_force.yuv",
    ]).output().unwrap();
    let decoded = std::fs::read("/tmp/iip_1080_force.yuv").unwrap();

    let mb_w = (w / 16) as usize;
    let mb_h = (h / 16) as usize;

    for fi in 0..n_frames.min(decoded.len() / frame_size) {
        let (ey, _, _) = &enc_recons[fi];
        let base = fi * frame_size;
        let dy = &decoded[base..base + y_size];

        let mut total_diff = 0usize;
        let mut max_diff = 0i32;
        let mut first_bad_mb: Option<(usize, usize, usize, i32)> = None;
        for mb_y in 0..mb_h {
            for mb_x in 0..mb_w {
                let mut mb_diff = 0usize;
                let mut mb_max = 0i32;
                for dy_off in 0..16 {
                    for dx_off in 0..16 {
                        let idx = (mb_y * 16 + dy_off) * w as usize + mb_x * 16 + dx_off;
                        let d = ey[idx] as i32 - dy[idx] as i32;
                        if d != 0 {
                            mb_diff += 1;
                            if d.abs() > mb_max { mb_max = d.abs(); }
                        }
                    }
                }
                if mb_diff > 0 && first_bad_mb.is_none() {
                    first_bad_mb = Some((mb_x, mb_y, mb_diff, mb_max));
                }
                total_diff += mb_diff;
                if mb_max > max_diff { max_diff = mb_max; }
            }
        }
        println!("frame {}: {} px differ (max |Δ|={})", fi, total_diff, max_diff);
        if let Some((mx, my, c, m)) = first_bad_mb {
            println!("  first MB: ({},{}) {} px max={}", mx, my, c, m);
            if (mx, my) == (force_x, force_y) {
                println!("  (= forced intra-in-P MB) — dump diff grid:");
                for dy_off in 0..16 {
                    for dx_off in 0..16 {
                        let idx = (my * 16 + dy_off) * w as usize + mx * 16 + dx_off;
                        let d = ey[idx] as i32 - dy[idx] as i32;
                        print!(" {:3}", d);
                    }
                    println!();
                }
            }
        }
    }
}
