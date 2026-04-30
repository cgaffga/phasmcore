// §6E-A4(c)-lite — render a sample IBPBP video to disk for visual
// quality assessment.
//
// Generates 30 frames of synthetic content with horizontal motion
// (a vertical-bar pattern sliding right at 4 px/frame). Encodes
// in IBPBP encode order using `encode_b_frame` from §6E-A4(a). Writes
// raw Annex-B bytes to a path given via PHASM_OUT (default
// `~/Desktop/phasm_bframe_test.h264`).
//
// Usage:
//   cargo run --release --features cabac-stego --example h264_bframe_demo
//
// Then play with ffplay or VLC:
//   ffplay ~/Desktop/phasm_bframe_test.h264
//
// Or convert to MP4 for general playback:
//   ffmpeg -y -i ~/Desktop/phasm_bframe_test.h264 \
//     -c copy ~/Desktop/phasm_bframe_test.mp4

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

const W: u32 = 320;
const H: u32 = 240;
const N_FRAMES: usize = 30;

fn main() {
    let out_path = std::env::var("PHASM_OUT").unwrap_or_else(|_| {
        let home = std::env::var("HOME").expect("HOME");
        format!("{home}/Desktop/phasm_bframe_test.h264")
    });

    let mut enc = Encoder::new(W, H, Some(75)).expect("encoder init");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_b_frames = true;

    // Generate frames. Display order: I₀ B₁ P₂ B₃ P₄ B₅ P₆ ...
    // Each P-frame at display index 2k. Each B-frame at 2k+1.
    // Encode order (M=2 IBPBP): I₀ P₂ B₁ P₄ B₃ P₆ B₅ ... — every
    // (P, B) PAIR encodes the P first (display=2(k+1)) then the B
    // that displays at 2k+1.
    //
    // For this demo, all frames have a single sliding bar so motion
    // is visible. Display index → bar position; encode order
    // determines which YUV the encoder receives at each call.
    let mut all_bytes = Vec::new();

    // Encode I-frame at display 0.
    let yuv_i = make_frame(0);
    all_bytes.extend_from_slice(&enc.encode_i_frame(&yuv_i).expect("I"));
    println!("encoded I at display 0");

    let n_pairs = (N_FRAMES - 1) / 2;
    for k in 0..n_pairs {
        let p_display = 2 * (k + 1);
        let b_display = 2 * k + 1;

        // Encode P at display 2(k+1).
        let yuv_p = make_frame(p_display);
        all_bytes.extend_from_slice(&enc.encode_p_frame(&yuv_p).expect("P"));
        println!("encoded P at display {p_display}");

        // Encode B at display 2k+1 (uses prev I/P + just-encoded P).
        let yuv_b = make_frame(b_display);
        all_bytes.extend_from_slice(&enc.encode_b_frame(&yuv_b).expect("B"));
        println!("encoded B at display {b_display}");
    }

    std::fs::write(&out_path, &all_bytes).expect("write output");
    println!(
        "\nWrote {} bytes to {}\n\
         Frame order: 1 IDR + {} (P, B) pairs = {} encoded frames\n\
         Display order: {} frames (1 I + {} P + {} B)\n\n\
         Play with:\n  ffplay {}\n\
         Or convert to MP4:\n  ffmpeg -y -i {} -c copy {}.mp4",
        all_bytes.len(),
        out_path,
        n_pairs,
        1 + n_pairs * 2,
        1 + n_pairs * 2,
        n_pairs,
        n_pairs,
        out_path,
        out_path,
        out_path,
    );
}

/// Generate a YUV420p frame with a vertical bar at horizontal
/// position `display_index * 4`. Wraps modulo width.
fn make_frame(display_index: usize) -> Vec<u8> {
    let frame_size = (W * H * 3 / 2) as usize;
    let mut buf = vec![128u8; frame_size];
    let bar_x = (display_index as u32 * 4) % W;

    // Y plane: vertical bar at bar_x..bar_x+8 is bright (220),
    // background mid-gray (80).
    for y in 0..H {
        for x in 0..W {
            let idx = (y * W + x) as usize;
            let dx = (x as i32 - bar_x as i32).rem_euclid(W as i32);
            buf[idx] = if (0..8).contains(&dx) { 220 } else { 80 };
        }
    }
    // Chroma planes: flat (no color shift).
    let y_size = (W * H) as usize;
    let c_size = (W / 2 * H / 2) as usize;
    for v in &mut buf[y_size..y_size + c_size] {
        *v = 128;
    }
    for v in &mut buf[y_size + c_size..] {
        *v = 128;
    }
    buf
}
