// Encode N frames (1 I + N-1 P), dump encoder's internal recon for
// EACH frame, decode the bitstream with ffmpeg, compare per-frame.
// Prints per-frame enc-vs-dec diff AND per-frame src-vs-recon PSNR so
// we can see where the smear emerges.

use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n_frames: usize = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(30);
    let yuv_path = std::env::var("YUV").unwrap_or_else(|_| format!("/tmp/img4138_1080p_f{n_frames}.yuv"));
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let c_size = (w * h / 4) as usize;
    let pixels = std::fs::read(&yuv_path).expect("yuv missing");
    assert!(pixels.len() >= n_frames * frame_size, "yuv too short: {} vs need {}", pixels.len(), n_frames * frame_size);

    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();

    // Snapshot I-frame recon.
    let mut enc_recons = vec![(enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone())];

    for fi in 1..n_frames {
        let src = &pixels[fi * frame_size..(fi + 1) * frame_size];
        bytes.extend_from_slice(&enc.encode_p_frame(src).unwrap());
        enc_recons.push((enc.recon.y.clone(), enc.recon.cb.clone(), enc.recon.cr.clone()));
    }

    let h264_path = std::env::var("PHASM_OUT_H264")
        .unwrap_or_else(|_| "/tmp/nframe_drift.h264".to_string());
    let dec_path = std::env::var("PHASM_OUT_DEC_YUV")
        .unwrap_or_else(|_| "/tmp/nframe_drift_decoded.yuv".to_string());
    std::fs::write(&h264_path, &bytes).unwrap();

    Command::new("ffmpeg").args([
        "-y", "-loglevel", "error", "-f", "h264", "-i", h264_path.as_str(),
        "-f", "rawvideo", "-pix_fmt", "yuv420p", dec_path.as_str(),
    ]).output().unwrap();
    let decoded = std::fs::read(&dec_path).unwrap();
    let n_decoded = decoded.len() / frame_size;
    if n_decoded != n_frames {
        println!("WARN: decoded {n_decoded} frames, encoded {n_frames}");
    }
    let n_check = n_decoded.min(n_frames);

    println!("frame |  enc-vs-dec |  src-vs-recon  | src-vs-dec  | notes");
    println!("------+-------------+----------------+-------------+--------");
    for fi in 0..n_check {
        let (ey, _, _) = &enc_recons[fi];
        let base = fi * frame_size;
        let dy = &decoded[base..base + y_size];
        let src_y = &pixels[base..base + y_size];

        // Enc vs Dec (desync indicator).
        let mut ed_sqe = 0.0f64;
        let mut diff_pixels = 0usize;
        let mut max_diff = 0i32;
        for (a, b) in ey.iter().zip(dy.iter()) {
            let d = *a as i32 - *b as i32;
            if d != 0 { diff_pixels += 1; }
            if d.abs() > max_diff { max_diff = d.abs(); }
            ed_sqe += (d as f64) * (d as f64);
        }
        let ed_mse = ed_sqe / y_size as f64;
        let ed_psnr = if ed_mse > 0.0 { 10.0 * (255.0 * 255.0 / ed_mse).log10() } else { 99.99 };

        // Src vs ENC recon.
        let mut se_sqe = 0.0f64;
        for i in 0..y_size { let d = src_y[i] as f64 - ey[i] as f64; se_sqe += d * d; }
        let se_mse = se_sqe / y_size as f64;
        let se_psnr = 10.0 * (255.0 * 255.0 / se_mse).log10();

        // Src vs DEC recon.
        let mut sd_sqe = 0.0f64;
        for i in 0..y_size { let d = src_y[i] as f64 - dy[i] as f64; sd_sqe += d * d; }
        let sd_mse = sd_sqe / y_size as f64;
        let sd_psnr = 10.0 * (255.0 * 255.0 / sd_mse).log10();

        // First divergent MB on this frame (if any).
        let mb_w = (w / 16) as usize;
        let mb_h = (h / 16) as usize;
        let mut first_bad: Option<(usize, usize, usize, i32)> = None;
        if diff_pixels > 0 {
            'outer: for mby in 0..mb_h {
                for mbx in 0..mb_w {
                    let mut md = 0usize;
                    let mut mm = 0i32;
                    for dy_ in 0..16 {
                        for dx_ in 0..16 {
                            let x = mbx * 16 + dx_;
                            let yy = mby * 16 + dy_;
                            let d = ey[yy * w as usize + x] as i32 - dy[yy * w as usize + x] as i32;
                            if d != 0 { md += 1; }
                            if d.abs() > mm { mm = d.abs(); }
                        }
                    }
                    if md > 0 {
                        first_bad = Some((mbx, mby, md, mm));
                        break 'outer;
                    }
                }
            }
        }

        let note = if diff_pixels == 0 {
            String::from("in sync")
        } else if let Some((mx, my, md, mm)) = first_bad {
            format!("DESYNC {diff_pixels}/{y_size} ({:.2}%) max{max_diff} firstMB=({mx},{my}) {md}/256 px max{mm}",
                diff_pixels as f64 * 100.0 / y_size as f64)
        } else {
            format!("DESYNC {diff_pixels}/{y_size} ({:.2}%) max{max_diff}", diff_pixels as f64 * 100.0 / y_size as f64)
        };
        println!("{fi:5} | {ed_psnr:8.2} dB | {se_psnr:10.2} dB | {sd_psnr:8.2} dB | {note}");

        // Dump the first divergent MB (enc, dec, diff) for the specified frame.
        let dump_frame = std::env::var("DUMP_FRAME").ok().and_then(|s| s.parse::<usize>().ok());
        if dump_frame == Some(fi) {
            let override_mb = std::env::var("DUMP_MB").ok().and_then(|s| {
                let mut p = s.split(',');
                let x: usize = p.next()?.trim().parse().ok()?;
                let y: usize = p.next()?.trim().parse().ok()?;
                Some((x, y, 0usize, 0i32))
            });
            let target = override_mb.or(first_bad);
            if let Some((mx, my, _, _)) = target {
                println!("\n  ENC recon at MB ({mx},{my}) frame {fi}:");
                for dy_ in 0..16 {
                    let mut r = String::new();
                    for dx_ in 0..16 {
                        let x = mx * 16 + dx_;
                        let yy = my * 16 + dy_;
                        r.push_str(&format!("{:4}", ey[yy * w as usize + x]));
                    }
                    println!("  {r}");
                }
                println!("\n  DEC recon at MB ({mx},{my}) frame {fi}:");
                for dy_ in 0..16 {
                    let mut r = String::new();
                    for dx_ in 0..16 {
                        let x = mx * 16 + dx_;
                        let yy = my * 16 + dy_;
                        r.push_str(&format!("{:4}", dy[yy * w as usize + x]));
                    }
                    println!("  {r}");
                }
                println!("\n  DIFF (ENC - DEC) at MB ({mx},{my}) frame {fi}:");
                for dy_ in 0..16 {
                    let mut r = String::new();
                    for dx_ in 0..16 {
                        let x = mx * 16 + dx_;
                        let yy = my * 16 + dy_;
                        let d = ey[yy * w as usize + x] as i32 - dy[yy * w as usize + x] as i32;
                        r.push_str(&format!("{:4}", d));
                    }
                    println!("  {r}");
                }
                // Dump the REFERENCE frame patch at the same MB position (the
                // I-frame recon should match at frame 1 — if both sides read
                // from the same reference, drift starts in MC or residual).
                if fi >= 1 {
                    let ref_y = &enc_recons[fi - 1].0;
                    let ref_dec_base = (fi - 1) * frame_size;
                    let ref_dec = &decoded[ref_dec_base..ref_dec_base + y_size];
                    let mut ref_diff_cnt = 0;
                    for dy_ in 0..16 {
                        for dx_ in 0..16 {
                            let x = mx * 16 + dx_;
                            let yy = my * 16 + dy_;
                            if ref_y[yy * w as usize + x] != ref_dec[yy * w as usize + x] {
                                ref_diff_cnt += 1;
                            }
                        }
                    }
                    println!("\n  Reference frame ({}) at same MB: {ref_diff_cnt}/256 pixels differ enc-vs-dec", fi - 1);
                }
            }
        }
    }
}
