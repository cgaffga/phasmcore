// Encode 1 I + 1 P with real content. Dump encoder's internal recon
// of frame 1. Compare to ffmpeg's decode of same bitstream's frame 1.
// PSNR between these two = encoder-vs-decoder recon mismatch.
use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    let w: u32 = 1920;
    let h: u32 = 1072;
    let n: usize = 2;
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w*h) as usize;
    let c_size = (w*h/4) as usize;
    let pixels = std::fs::read("/tmp/img4138_1080p_f10.yuv").unwrap();

    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    let i_only = std::env::var_os("I_ONLY").is_some();
    let check_frame;
    if !i_only {
        bytes.extend_from_slice(&enc.encode_p_frame(&pixels[frame_size..2*frame_size]).unwrap());
        check_frame = 1;
    } else {
        check_frame = 0;
    }
    std::fs::write("/tmp/recon_test.h264", &bytes).unwrap();

    // Encoder's internal recon AFTER last frame commit.
    let enc_y = enc.recon.y.clone();
    let enc_cb = enc.recon.cb.clone();
    let enc_cr = enc.recon.cr.clone();

    Command::new("ffmpeg").args(["-y","-loglevel","error","-f","h264","-i","/tmp/recon_test.h264",
        "-f","rawvideo","-pix_fmt","yuv420p","/tmp/recon_test_decoded.yuv"]).output().unwrap();
    let decoded = std::fs::read("/tmp/recon_test_decoded.yuv").unwrap();

    let base = check_frame * frame_size;
    let dec_y = &decoded[base..base + y_size];
    let dec_cb = &decoded[base + y_size..base + y_size + c_size];
    let dec_cr = &decoded[base + y_size + c_size..base + y_size + 2*c_size];

    eprintln!("Checking frame {check_frame} (i_only={i_only})");

    // Encoder vs Decoder recon (both are the result of P-frame reconstruction).
    let mut diffs = 0usize;
    let mut max_diff = 0i32;
    let mut total_sqe = 0.0;
    for (a, b) in enc_y.iter().zip(dec_y.iter()) {
        let d = *a as i32 - *b as i32;
        if d != 0 { diffs += 1; }
        if d.abs() > max_diff { max_diff = d.abs(); }
        total_sqe += (d as f64) * (d as f64);
    }
    let mse = total_sqe / y_size as f64;
    let psnr = if mse > 0.0 { 10.0 * (255.0 * 255.0 / mse).log10() } else { 99.99 };
    println!("Y plane: {diffs}/{y_size} pixels differ ({:.2}%)", diffs as f64 * 100.0 / y_size as f64);
    println!("  max diff: {max_diff}, MSE: {mse:.4}, PSNR (enc_recon vs dec_recon): {psnr:.2} dB");

    // Per-MB diff map to find the FIRST diverging MB.
    let mb_w = (w / 16) as usize;
    let mb_h = (h / 16) as usize;
    let mut first_bad = None;
    let mut total_bad_mbs = 0;
    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            let mut mb_diff = 0;
            let mut mb_max = 0i32;
            for dy in 0..16 {
                for dx in 0..16 {
                    let x = mb_x * 16 + dx;
                    let y = mb_y * 16 + dy;
                    let d = enc_y[y * w as usize + x] as i32 - dec_y[y * w as usize + x] as i32;
                    if d != 0 { mb_diff += 1; }
                    if d.abs() > mb_max { mb_max = d.abs(); }
                }
            }
            if mb_diff > 0 {
                total_bad_mbs += 1;
                if first_bad.is_none() {
                    first_bad = Some((mb_x, mb_y, mb_diff, mb_max));
                }
            }
        }
    }
    println!("  MBs with at least 1 diff: {total_bad_mbs}/{}", mb_w * mb_h);
    if let Some((mx, my, d, m)) = first_bad {
        println!("  first diff MB: ({mx},{my}) — {d}/256 pixels differ, max diff {m}");
        // Dump the 16x16 diff for inspection.
        eprintln!("\nEnc-Dec diff map for MB ({mx},{my}):");
        for dy in 0..16 {
            let mut row = String::new();
            for dx in 0..16 {
                let x = mx * 16 + dx;
                let y = my * 16 + dy;
                let d = enc_y[y * w as usize + x] as i32 - dec_y[y * w as usize + x] as i32;
                row.push_str(&format!("{:4}", d));
            }
            eprintln!("{row}");
        }
        // Optional: raw enc/dec pixel dump for the same MB.
        if std::env::var_os("DUMP_RAW_MB").is_some() {
            eprintln!("\nRAW enc pixels for MB ({mx},{my}):");
            for dy in 0..16 {
                let mut row = String::new();
                for dx in 0..16 {
                    let x = mx * 16 + dx;
                    let y = my * 16 + dy;
                    row.push_str(&format!(" {:3}", enc_y[y * w as usize + x]));
                }
                eprintln!("{row}");
            }
            eprintln!("\nRAW dec pixels for MB ({mx},{my}):");
            for dy in 0..16 {
                let mut row = String::new();
                for dx in 0..16 {
                    let x = mx * 16 + dx;
                    let y = my * 16 + dy;
                    row.push_str(&format!(" {:3}", dec_y[y * w as usize + x]));
                }
                eprintln!("{row}");
            }
        }
    }

    // Also compare to source.
    let src_y = &pixels[base..base + y_size];
    let mut src_dec_sqe = 0.0;
    let mut src_enc_sqe = 0.0;
    for i in 0..y_size {
        let s = src_y[i] as f64;
        src_dec_sqe += (s - dec_y[i] as f64).powi(2);
        src_enc_sqe += (s - enc_y[i] as f64).powi(2);
    }
    let src_dec_mse = src_dec_sqe / y_size as f64;
    let src_enc_mse = src_enc_sqe / y_size as f64;
    let src_dec_psnr = 10.0 * (255.0*255.0 / src_dec_mse).log10();
    let src_enc_psnr = 10.0 * (255.0*255.0 / src_enc_mse).log10();
    println!("  src vs dec_recon: {src_dec_psnr:.2} dB");
    println!("  src vs enc_recon: {src_enc_psnr:.2} dB");

    // Chroma.
    let mut cb_diffs = 0usize;
    let mut cb_max = 0i32;
    for (a,b) in enc_cb.iter().zip(dec_cb.iter()) {
        let d = *a as i32 - *b as i32;
        if d != 0 { cb_diffs += 1; }
        if d.abs() > cb_max { cb_max = d.abs(); }
    }
    let mut cr_diffs = 0usize;
    let mut cr_max = 0i32;
    for (a,b) in enc_cr.iter().zip(dec_cr.iter()) {
        let d = *a as i32 - *b as i32;
        if d != 0 { cr_diffs += 1; }
        if d.abs() > cr_max { cr_max = d.abs(); }
    }
    println!("Cb plane: {cb_diffs}/{c_size} differ, max |diff|={cb_max}");
    println!("Cr plane: {cr_diffs}/{c_size} differ, max |diff|={cr_max}");
}
