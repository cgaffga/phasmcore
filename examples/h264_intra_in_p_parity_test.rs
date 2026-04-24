// Diagnostic harness for task #154: force intra-in-P on a specific MB
// and byte-exact compare encoder recon vs ffmpeg decode of the same
// bitstream. Pinpoints the exact MB/plane where enc-dec diverges so
// the root-cause fix can target a specific syntax element or quant
// path.
//
// Usage:
//   PHASM_FORCE_INTRA_IN_P_MB=5,3 \
//     cargo run --release --features video,h264-encoder \
//     --example h264_intra_in_p_parity_test
//
// Default target: MB (5, 3) on a 128×64 test frame = 8×4 MBs. Plenty
// of neighbor context but small enough to eye-inspect the diff.

use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    // Enable diagnostic if caller didn't already set it.
    if std::env::var_os("PHASM_FORCE_INTRA_IN_P_MB").is_none() {
        unsafe { std::env::set_var("PHASM_FORCE_INTRA_IN_P_MB", "5,3"); }
    }
    let (force_x, force_y) = parse_force_mb();
    eprintln!("Forcing intra-in-P on MB ({force_x},{force_y}).");

    let w: u32 = 128;
    let h: u32 = 64;
    let frame_size = (w * h * 3 / 2) as usize;
    let y_size = (w * h) as usize;
    let c_size = (w * h / 4) as usize;

    // Build two frames: flat-grey I, tiny-textured P. Keeps inter cost
    // low most places but forces our diagnostic MB into intra.
    let mut frame0 = vec![128u8; frame_size];
    let mut frame1 = vec![128u8; frame_size];
    // Add a gradient to Y in frame 1 so some MBs see actual motion.
    for yy in 0..h as usize {
        for xx in 0..w as usize {
            frame1[yy * w as usize + xx] =
                (128 + ((xx as i32 - 64).abs().min(30) as u8)) as u8;
        }
    }

    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(80);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let mut bytes = enc.encode_i_frame(&frame0).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&frame1).unwrap());
    std::fs::write("/tmp/intra_in_p_test.h264", &bytes).unwrap();

    let enc_y = enc.recon.y.clone();
    let enc_cb = enc.recon.cb.clone();
    let enc_cr = enc.recon.cr.clone();

    Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error",
            "-f", "h264", "-i", "/tmp/intra_in_p_test.h264",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "/tmp/intra_in_p_test.yuv",
        ])
        .output()
        .unwrap();
    let decoded = std::fs::read("/tmp/intra_in_p_test.yuv").unwrap();

    // Compare P-frame (frame 1) — 1-frame offset.
    let base = frame_size;
    let dec_y = &decoded[base..base + y_size];
    let dec_cb = &decoded[base + y_size..base + y_size + c_size];
    let dec_cr = &decoded[base + y_size + c_size..base + y_size + 2 * c_size];

    // Compute diffs per MB and report.
    let mb_w = (w / 16) as usize;
    let mut y_differ = 0usize;
    let mut y_max = 0i32;
    let mut worst_mb: Option<(usize, usize, i32)> = None;
    for mb_y in 0..(h / 16) as usize {
        for mb_x in 0..mb_w {
            let mut mb_diff = 0i32;
            let mut mb_max = 0i32;
            for dy in 0..16 {
                for dx in 0..16 {
                    let idx = (mb_y * 16 + dy) * w as usize + mb_x * 16 + dx;
                    let d = enc_y[idx] as i32 - dec_y[idx] as i32;
                    if d != 0 {
                        mb_diff += 1;
                        if d.abs() > mb_max { mb_max = d.abs(); }
                    }
                }
            }
            if mb_diff > 0 {
                y_differ += mb_diff as usize;
                if mb_max > y_max { y_max = mb_max; }
                if worst_mb.is_none() || mb_diff > worst_mb.unwrap().2 {
                    worst_mb = Some((mb_x, mb_y, mb_diff));
                }
            }
        }
    }

    eprintln!();
    eprintln!("Y plane: {y_differ}/{y_size} px differ, max |diff| = {y_max}");
    if let Some((mx, my, count)) = worst_mb {
        let is_forced = (mx, my) == (force_x, force_y);
        eprintln!("  worst MB: ({mx},{my}) - {count}/256 px {}",
                  if is_forced { "[FORCED INTRA]" } else { "" });
        // Dump diff grid for the worst MB.
        for dy in 0..16 {
            for dx in 0..16 {
                let idx = (my * 16 + dy) * w as usize + mx * 16 + dx;
                let d = enc_y[idx] as i32 - dec_y[idx] as i32;
                eprint!(" {:3}", d);
            }
            eprintln!();
        }
    }

    let mut cb_diffs = 0;
    let mut cb_max = 0i32;
    for (a, b) in enc_cb.iter().zip(dec_cb.iter()) {
        let d = *a as i32 - *b as i32;
        if d != 0 {
            cb_diffs += 1;
            if d.abs() > cb_max { cb_max = d.abs(); }
        }
    }
    let mut cr_diffs = 0;
    let mut cr_max = 0i32;
    for (a, b) in enc_cr.iter().zip(dec_cr.iter()) {
        let d = *a as i32 - *b as i32;
        if d != 0 {
            cr_diffs += 1;
            if d.abs() > cr_max { cr_max = d.abs(); }
        }
    }
    eprintln!("Cb plane: {cb_diffs}/{c_size} px differ, max |diff| = {cb_max}");
    eprintln!("Cr plane: {cr_diffs}/{c_size} px differ, max |diff| = {cr_max}");

    if y_differ == 0 && cb_diffs == 0 && cr_diffs == 0 {
        eprintln!("\nPASS: enc recon bit-exact with decoder.");
    } else {
        eprintln!("\nFAIL: enc recon diverges from decoder.");
        std::process::exit(1);
    }
}

fn parse_force_mb() -> (usize, usize) {
    let s = std::env::var("PHASM_FORCE_INTRA_IN_P_MB").unwrap_or_else(|_| "5,3".to_string());
    let mut parts = s.split(',');
    let x: usize = parts.next().unwrap().trim().parse().unwrap();
    let y: usize = parts.next().unwrap().trim().parse().unwrap();
    (x, y)
}
