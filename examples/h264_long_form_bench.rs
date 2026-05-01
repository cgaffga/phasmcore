// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// §long-form-stego Phase 6.4 — memory benchmark harness for v1
// (Phase 5 sequential) vs v2 (Phase 6.3 interleaved) streaming
// orchestrators.
//
// Usage:
//   cargo build --release --features cabac-stego --example h264_long_form_bench
//
//   # Linux (GNU time):
//   /usr/bin/time -v ./target/release/examples/h264_long_form_bench \
//       /tmp/img4138_1080p_f10.yuv 1920 1072 10 5 v1
//   /usr/bin/time -v ./target/release/examples/h264_long_form_bench \
//       /tmp/img4138_1080p_f10.yuv 1920 1072 10 5 v2
//
//   # macOS (BSD time):
//   /usr/bin/time -l ./target/release/examples/h264_long_form_bench \
//       /tmp/img4138_1080p_f10.yuv 1920 1072 10 5 v1
//   /usr/bin/time -l ./target/release/examples/h264_long_form_bench \
//       /tmp/img4138_1080p_f10.yuv 1920 1072 10 5 v2
//
// Compare the "maximum resident set size" line between v1 and v2
// runs. The v2 figure should be lower than v1's (plan-side
// streaming saves O(num_gops × per_gop_plan_size) at peak).

#[cfg(feature = "cabac-stego")]
fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 7 {
        eprintln!(
            "usage: {} <yuv-path> <width> <height> <n-frames> <gop-size> <v1|v2>",
            args[0]
        );
        std::process::exit(2);
    }
    let yuv_path = &args[1];
    let width: u32 = args[2].parse().expect("width");
    let height: u32 = args[3].parse().expect("height");
    let n_frames: usize = args[4].parse().expect("n_frames");
    let gop_size: usize = args[5].parse().expect("gop_size");
    let mode = args[6].as_str();

    let yuv = std::fs::read(yuv_path).expect("read yuv");
    let expected = (width as usize) * (height as usize) * 3 / 2 * n_frames;
    if yuv.len() != expected {
        eprintln!(
            "yuv size mismatch: got {} expected {} ({}x{} × {})",
            yuv.len(),
            expected,
            width,
            height,
            n_frames,
        );
        std::process::exit(3);
    }

    eprintln!("yuv: {} ({} bytes)", yuv_path, yuv.len());
    eprintln!(
        "dimensions: {}x{} n_frames={} gop_size={} num_gops={}",
        width,
        height,
        n_frames,
        gop_size,
        n_frames.div_ceil(gop_size),
    );
    eprintln!("mode: {}", mode);

    let start = std::time::Instant::now();
    let bytes = match mode {
        "v1" => {
            phasm_core::h264_stego_encode_yuv_string_4domain_multigop_streaming(
                &yuv, width, height, n_frames, gop_size, "x", "long-form-bench",
            )
            .expect("v1 encode")
        }
        "v2" => {
            phasm_core::h264_stego_encode_yuv_string_4domain_multigop_streaming_v2(
                &yuv, width, height, n_frames, gop_size, "x", "long-form-bench",
            )
            .expect("v2 encode")
        }
        _ => {
            eprintln!("mode must be 'v1' or 'v2', got '{mode}'");
            std::process::exit(4);
        }
    };
    let elapsed = start.elapsed();

    eprintln!("output: {} bytes", bytes.len());
    eprintln!("elapsed: {:.2}s", elapsed.as_secs_f64());
    eprintln!("done.");
}

#[cfg(not(feature = "cabac-stego"))]
fn main() {
    eprintln!("h264_long_form_bench requires --features cabac-stego");
    std::process::exit(1);
}
