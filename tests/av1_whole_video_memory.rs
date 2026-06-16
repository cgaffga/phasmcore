#![cfg(all(feature = "av1-encoder", feature = "av1-decoder"))]

//! WV.6.g — Peak-RSS regression test for whole-video shadow streaming
//! Pass 1.
//!
//! Validates the streaming-as-invariant property: at 1080p × 30s × 2
//! shadows, `Av1StreamingEncodeSession::create_whole_video_with_shadows`
//! must keep peak RSS under ~500 MB (target ~150-200 MB; 500 MB is
//! generous variance headroom for hosts with high baseline RSS or
//! allocator behaviour quirks). Pre-WV.6 the pre-allocated
//! `Av1WholeVideoState::yuv_buffer` alone was ~2.8 GB at this scale
//! — this test fails loudly if a future change re-introduces a
//! whole-clip buffer.
//!
//! `#[ignore]` because the real 1080p × 30s encode takes ~2-4 min
//! wall on a dev box. Run explicitly:
//!
//!   cargo test --release --features av1-encoder,av1-decoder \
//!     --test av1_whole_video_memory -- --ignored
//!
//! Cross-platform via POSIX `getrusage` (`ru_maxrss`). macOS reports
//! bytes; Linux reports KiB; both normalised to bytes here.
//! Non-Unix falls through with RSS=0 (the test still runs end-to-
//! end on those, just without the RSS gate).

use std::time::Instant;

use phasm_core::codec::av1::stego::session::{
    Av1ShadowSpec, Av1StreamingEncodeParams, Av1StreamingEncodeSession,
};

#[cfg(unix)]
fn peak_rss_bytes() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut ru) } != 0 {
        return 0;
    }
    let max_rss = ru.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        max_rss
    }
    #[cfg(not(target_os = "macos"))]
    {
        max_rss * 1024
    }
}

#[cfg(not(unix))]
fn peak_rss_bytes() -> u64 {
    0
}

/// Build one frame of synthetic I420 YUV. Uses a per-frame gradient
/// plus a small position hash so cover capacity is non-trivial — a
/// constant-gray frame would have ~0 AC coefficients and starve the
/// shadow allocator.
fn gradient_yuv_frame(width: usize, height: usize, frame_idx: usize) -> Vec<u8> {
    let y_size = width * height;
    let uv_size = (width / 2) * (height / 2);
    let mut buf = vec![0u8; y_size + 2 * uv_size];

    for y in 0..height {
        for x in 0..width {
            let hash = ((x * 37 + y * 53 + frame_idx * 11) % 19) as i32 - 9;
            let val = ((x + y + frame_idx * 3) % 256) as i32 + hash;
            buf[y * width + x] = val.clamp(0, 255) as u8;
        }
    }

    let cb_off = y_size;
    let cr_off = y_size + uv_size;
    for y in 0..height / 2 {
        for x in 0..width / 2 {
            let idx = y * (width / 2) + x;
            buf[cb_off + idx] = (128 + (((x + frame_idx) % 32) as i32) - 16) as u8;
            buf[cr_off + idx] = (128 + (((y + frame_idx) % 32) as i32) - 16) as u8;
        }
    }
    buf
}

#[test]
#[ignore = "1080p × 30s real-encode; opt in with --ignored. ~2-4 min wall."]
fn wv6_streaming_pass1_peak_rss_1080p_30s_2shadow() {
    const WIDTH: u32 = 1920;
    const HEIGHT: u32 = 1080;
    const N_FRAMES: u32 = 900; // 30s at 30fps
    const GOP_SIZE: u32 = 30;
    const PEAK_RSS_LIMIT_BYTES: u64 = 500 * 1024 * 1024; // 500 MB

    eprintln!(
        "[wv6.g] {}x{} × {} frames × gop={} × 2 shadows; RSS limit = {} MB",
        WIDTH,
        HEIGHT,
        N_FRAMES,
        GOP_SIZE,
        PEAK_RSS_LIMIT_BYTES / (1024 * 1024)
    );

    let params = Av1StreamingEncodeParams {
        width: WIDTH,
        height: HEIGHT,
        quantizer: 100,
        gop_size: GOP_SIZE,
        total_frames_hint: N_FRAMES,
    };

    let shadows = vec![
        Av1ShadowSpec {
            passphrase: "shadow1_passphrase_wv6g".to_string(),
            message: vec![0xab; 1024],
        },
        Av1ShadowSpec {
            passphrase: "shadow2_passphrase_wv6g".to_string(),
            message: vec![0xcd; 1024],
        },
    ];

    let rss_before = peak_rss_bytes();
    eprintln!(
        "[wv6.g] RSS baseline: {:.1} MB",
        rss_before as f64 / 1_048_576.0
    );

    let mut session = Av1StreamingEncodeSession::create_whole_video_with_shadows(
        "primary_passphrase_wv6g",
        b"WV.6.g regression test primary",
        params,
        shadows,
        16,
    )
    .expect("create_whole_video_with_shadows");

    let t0 = Instant::now();
    let mut out = Vec::new();
    for frame_idx in 0..N_FRAMES {
        let yuv = gradient_yuv_frame(WIDTH as usize, HEIGHT as usize, frame_idx as usize);
        session
            .push_frame(&yuv, &mut out)
            .unwrap_or_else(|e| panic!("push_frame[{}]: {:?}", frame_idx, e));

        // Sample RSS during the push loop too — catches the case
        // where peak hits mid-encode and falls back by finish.
        if frame_idx > 0 && frame_idx.is_multiple_of(GOP_SIZE * 5) {
            let rss_mid = peak_rss_bytes();
            eprintln!(
                "[wv6.g] mid-encode frame={}/{} peak_rss={:.1}MB delta={:.1}MB",
                frame_idx,
                N_FRAMES,
                rss_mid as f64 / 1_048_576.0,
                rss_mid.saturating_sub(rss_before) as f64 / 1_048_576.0
            );
        }
    }
    session.finish(&mut out).expect("finish");
    let wall = t0.elapsed();

    let rss_after = peak_rss_bytes();
    let rss_delta = rss_after.saturating_sub(rss_before);

    eprintln!(
        "[wv6.g] DONE wall={:.1}s peak_rss_after={:.1}MB delta={:.1}MB out_bytes={}",
        wall.as_secs_f64(),
        rss_after as f64 / 1_048_576.0,
        rss_delta as f64 / 1_048_576.0,
        out.len()
    );

    #[cfg(unix)]
    assert!(
        rss_delta < PEAK_RSS_LIMIT_BYTES,
        "Peak RSS delta {:.0} MB exceeds limit {} MB. Streaming Pass 1 has \
         regressed — whole-video shadow path may have re-introduced a \
         whole-clip buffer.",
        rss_delta as f64 / 1_048_576.0,
        PEAK_RSS_LIMIT_BYTES / (1024 * 1024)
    );
}
