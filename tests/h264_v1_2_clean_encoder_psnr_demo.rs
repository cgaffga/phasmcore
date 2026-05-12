// Phase 2.18 follow-on (#287, 2026-05-09) — clean-encoder (no stego)
// PSNR-vs-source diagnostic. Mirrors the v1_2 demo flow but uses
// `Encoder::new()` directly without stego flips. If clean PSNR is
// reasonable at every frame while stego PSNR has a 21.62 dB drop at
// frame 5, the issue is stego flip damage. If clean PSNR is also
// poor at frame 5, the issue is encoder mode-decision quality.

#![cfg(feature = "cabac-stego")]

use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};
use phasm_core::codec::h264::stego::gop_pattern::{iter_encode_order, FrameType, GopPattern};
use phasm_core::codec::mp4::build::{
    build_mp4_with_pattern, FrameTiming, MuxerProfile,
};
use std::time::Instant;

const W: u32 = 1072;
const H: u32 = 1920;
const N: usize = 7;

fn psnr_y(decoded: &[u8], source: &[u8]) -> f64 {
    let n = (W * H) as usize;
    let mut sse: u64 = 0;
    for i in 0..n {
        let d = decoded[i] as i64 - source[i] as i64;
        sse += (d * d) as u64;
    }
    if sse == 0 {
        return 99.0;
    }
    20.0 * (255.0_f64).log10() - 10.0 * (sse as f64 / n as f64).log10()
}

#[test]
#[ignore]
fn clean_encoder_psnr_carplane() {
    let yuv_full = std::fs::read("/tmp/phasm_corpus_artlist_carplane_1072x1920_f30.yuv")
        .expect("missing carplane f30");
    // Allow shifting the start frame via PHASM_DEMO_START_OFFSET so we can
    // tell whether frame-5 is a content/position artifact.
    let start = std::env::var("PHASM_DEMO_START_OFFSET")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    let frame_size_bytes = (W * H * 3 / 2) as usize;
    let yuv = yuv_full[start * frame_size_bytes
        ..(start + N) * frame_size_bytes]
        .to_vec();
    eprintln!("encoding frames {}..{} (N={})", start, start + N, N);
    let frame_size = (W * H * 3 / 2) as usize;

    unsafe {
        std::env::set_var("PHASM_B_RDO", "1");
        std::env::set_var("PHASM_B_RESIDUAL", "1");
    }

    let pattern = if std::env::var_os("PHASM_DEMO_IPPPP").is_some() {
        GopPattern::Ipppp { gop: 30 }
    } else {
        GopPattern::Ibpbp { gop: 30, b_count: 1 }
    };
    let mut enc = Encoder::new(W, H, Some(26)).expect("encoder new");
    enc.entropy_mode = EntropyMode::Cabac;
    enc.enable_transform_8x8 = true;
    enc.enable_b_frames = true;

    let t0 = Instant::now();
    let mut bitstream = Vec::new();
    for eo in iter_encode_order(N, pattern) {
        let d = eo.display_idx as usize;
        let frame = &yuv[d * frame_size..(d + 1) * frame_size];
        let bytes = match eo.frame_type {
            FrameType::Idr => enc.encode_i_frame(frame),
            FrameType::P => enc.encode_p_frame(frame),
            FrameType::B => enc.encode_b_frame(frame),
        }
        .expect("encode");
        bitstream.extend_from_slice(&bytes);
    }
    eprintln!("clean encoder: {} bytes in {:?}", bitstream.len(), t0.elapsed());

    let h264_path = std::env::temp_dir().join("phasm_v12_clean.h264");
    let dec_path = std::env::temp_dir().join("phasm_v12_clean.dec.yuv");
    std::fs::write(&h264_path, &bitstream).expect("write");
    let status = std::process::Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(&h264_path)
        .args(["-f", "rawvideo", "-pix_fmt", "yuv420p"])
        .arg(&dec_path)
        .status()
        .expect("ffmpeg");
    assert!(status.success());
    let decoded = std::fs::read(&dec_path).expect("read");

    eprintln!("CLEAN encoder per-frame Y-PSNR vs source:");
    let mut total = 0.0;
    let mut min_psnr = 99.0_f64;
    let mut min_frame = 0;
    for i in 0..N {
        let off = i * frame_size;
        let p = psnr_y(
            &decoded[off..off + (W * H) as usize],
            &yuv[off..off + (W * H) as usize],
        );
        total += p;
        if p < min_psnr {
            min_psnr = p;
            min_frame = i;
        }
        eprintln!("  frame={:>2} Y-PSNR={:>6.2} dB", i, p);
    }
    eprintln!(
        "CLEAN avg Y-PSNR={:.2} dB  min={:.2} dB at frame {}",
        total / N as f64,
        min_psnr,
        min_frame
    );

    // Also save the clean MP4 to Desktop for visual comparison.
    let mp4 = build_mp4_with_pattern(
        MuxerProfile::HandbrakeX264,
        &bitstream,
        W, H,
        FrameTiming::FPS_30,
        pattern, N,
    )
    .expect("mp4");
    let home = std::env::var("HOME").expect("HOME");
    let suffix = if std::env::var_os("PHASM_DEMO_IPPPP").is_some() {
        "D_clean_IPPPP"
    } else {
        "C_clean_IBPBP_no_stego"
    };
    let path = std::path::PathBuf::from(home)
        .join("Desktop")
        .join(format!("phasm_v12_carplane_{}.mp4", suffix));
    std::fs::write(&path, &mp4).expect("desktop");
    eprintln!("clean MP4 → {}", path.display());
}
