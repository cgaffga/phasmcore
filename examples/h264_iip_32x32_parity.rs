// 32x32 forced-IIP parity test (the exact setup from the Phase D.0-B
// debug session). Encodes 2 frames (I + P) with MB (0,0) of the P frame
// forced to intra-in-P, then ffmpeg-decodes and byte-compares.

use std::process::Command;
use phasm_core::codec::h264::encoder::encoder::Encoder;

fn main() {
    unsafe {
        std::env::set_var("PHASM_CABAC_INTRA_IN_P", "1");
        std::env::set_var("PHASM_CABAC_INTRA_IN_P_FORCE_MB", "0,0");
        std::env::set_var("PHASM_FORCE_INTRA_IN_P_MB", "0,0");
    }

    let w: u32 = 32;
    let h: u32 = 32;
    let y_size = (w * h) as usize;
    let c_size = y_size / 4;
    let frame_size = y_size + 2 * c_size;

    // f0: gradient; f1: shifted gradient (both from original commit note test)
    let mut f0 = vec![0u8; frame_size];
    for i in 0..y_size { f0[i] = (i % 256) as u8; }
    for i in 0..c_size * 2 { f0[y_size + i] = 128; }

    let mut f1 = vec![0u8; frame_size];
    for i in 0..y_size { f1[i] = ((i + 50) % 256) as u8; }
    for i in 0..c_size { f1[y_size + i] = 130; }
    for i in 0..c_size { f1[y_size + c_size + i] = 126; }

    let q: u8 = std::env::var("Q").ok().and_then(|s| s.parse().ok()).unwrap_or(26);
    let mut enc = Encoder::new(w, h, Some(q)).unwrap();
    let mut bytes = enc.encode_i_frame(&f0).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&f1).unwrap());
    std::fs::write("/tmp/iip_32_current.h264", &bytes).unwrap();

    let enc_y = enc.recon.y.clone();
    let enc_cb = enc.recon.cb.clone();
    let enc_cr = enc.recon.cr.clone();

    Command::new("ffmpeg")
        .args([
            "-y", "-loglevel", "error",
            "-f", "h264", "-i", "/tmp/iip_32_current.h264",
            "-f", "rawvideo", "-pix_fmt", "yuv420p",
            "/tmp/iip_32_current.yuv",
        ])
        .output()
        .unwrap();
    let decoded = std::fs::read("/tmp/iip_32_current.yuv").unwrap();

    // Compare P-frame (frame 1), offset past the I-frame.
    let base = frame_size;
    let dec_y = &decoded[base..base + y_size];
    let dec_cb = &decoded[base + y_size..base + y_size + c_size];
    let dec_cr = &decoded[base + y_size + c_size..base + frame_size];

    let mut y_diffs = Vec::new();
    for i in 0..y_size {
        let d = enc_y[i] as i32 - dec_y[i] as i32;
        if d != 0 {
            let row = i / w as usize;
            let col = i % w as usize;
            y_diffs.push((col, row, d));
        }
    }

    let cb_diffs = enc_cb.iter().zip(dec_cb.iter()).filter(|(a, b)| a != b).count();
    let cr_diffs = enc_cr.iter().zip(dec_cr.iter()).filter(|(a, b)| a != b).count();

    println!("32x32 forced-IIP (Q={}), MB (0,0) forced intra-in-P:", q);
    println!("  Y: {}/{} px differ", y_diffs.len(), y_size);
    println!("  Cb: {}/{} px differ", cb_diffs, c_size);
    println!("  Cr: {}/{} px differ", cr_diffs, c_size);
    if !y_diffs.is_empty() {
        println!("  Y diff locations (col,row,delta):");
        for (c, r, d) in y_diffs.iter().take(16) {
            println!("    col={} row={} delta={}", c, r, d);
        }
    }

    if y_diffs.is_empty() && cb_diffs == 0 && cr_diffs == 0 {
        println!("\nPASS: 32x32 forced-IIP is bit-exact.");
    } else {
        println!("\nFAIL: divergence detected.");
        std::process::exit(1);
    }
}
