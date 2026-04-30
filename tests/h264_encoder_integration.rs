// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/companyrest

//! End-to-end integration tests: encoder → ffmpeg → pixels.
//!
//! Phase 6A.8 tests exercise the I_PCM path (bit-exact lossless).
//! Phase 6A.10 tests exercise the Intra_16x16 CAVLC path (lossy —
//! assertions on decode-cleanliness + PSNR bounds).
//!
//! Gated behind `h264-encoder`.

#![cfg(feature = "h264-encoder")]

mod common;

use common::h264_oracle::{decode_via_ffmpeg_with_format, system_has_ffmpeg};
use phasm_core::codec::h264::encoder::encoder::{Encoder, EntropyMode};

fn deterministic_yuv420p(width: u32, height: u32) -> Vec<u8> {
    let y = (width * height) as usize;
    let c = (width / 2 * height / 2) as usize;
    let mut buf = vec![0u8; y + 2 * c];
    for yy in 0..height {
        for xx in 0..width {
            buf[(yy * width + xx) as usize] = ((xx * 3 + yy * 5) & 0xFF) as u8;
        }
    }
    let y_end = y;
    for yy in 0..(height / 2) {
        for xx in 0..(width / 2) {
            buf[y_end + (yy * (width / 2) + xx) as usize] = ((xx * 7) & 0xFF) as u8;
        }
    }
    let c_end = y + c;
    for yy in 0..(height / 2) {
        for xx in 0..(width / 2) {
            buf[c_end + (yy * (width / 2) + xx) as usize] = ((yy * 11) & 0xFF) as u8;
        }
    }
    buf
}

fn flat_yuv420p(width: u32, height: u32, luma: u8) -> Vec<u8> {
    let y = (width * height) as usize;
    let c = (width / 2 * height / 2) as usize;
    let mut buf = vec![0u8; y + 2 * c];
    for i in 0..y {
        buf[i] = luma;
    }
    // Chroma 128 = neutral.
    for i in 0..2 * c {
        buf[y + i] = 128;
    }
    buf
}

/// MSE of two equal-length byte slices treated as 8-bit samples.
fn mse(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len());
    let mut acc = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        acc += d * d;
    }
    acc / a.len() as f64
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let m = mse(a, b);
    if m < 1e-9 {
        return 100.0;
    }
    10.0 * (255.0 * 255.0 / m).log10()
}

// ─── I_PCM path (Phase 6A.8 — bit-exact lossless) ──────────────────

#[test]
fn i_pcm_encode_ffmpeg_decodes_cleanly_32x32() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).expect("encoder construction");
    let pixels = deterministic_yuv420p(w, h);
    let bytes = enc.encode_i_frame_pcm(&pixels).expect("encode i-frame PCM");
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .unwrap_or_else(|e| panic!("oracle decode failed: {:?}", e));
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
    let expected = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), expected);
}

#[test]
fn i_pcm_encode_pixel_match_32x32() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let pixels = deterministic_yuv420p(w, h);
    let bytes = enc.encode_i_frame_pcm(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264")).unwrap();
    assert_eq!(result.decoded_yuv, pixels, "I_PCM must be bit-exact");
}

#[test]
fn i_pcm_encode_larger_frame_decodes_cleanly() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (64u32, 48u32);
    let mut enc = Encoder::new(w, h, None).unwrap();
    let pixels = deterministic_yuv420p(w, h);
    let bytes = enc.encode_i_frame_pcm(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264")).unwrap();
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
    assert_eq!(result.decoded_yuv, pixels, "64×48 I_PCM should round-trip bit-exact");
}

#[test]
fn i_pcm_encode_two_frames_sequence() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(90)).unwrap();
    let pixels_a = deterministic_yuv420p(w, h);
    let mut pixels_b = pixels_a.clone();
    for v in pixels_b.iter_mut().take(100) {
        *v = v.wrapping_add(50);
    }
    let bytes_a = enc.encode_i_frame_pcm(&pixels_a).unwrap();
    let bytes_b = enc.encode_i_frame_pcm(&pixels_b).unwrap();
    let mut concatenated = bytes_a;
    concatenated.extend_from_slice(&bytes_b);
    let result = decode_via_ffmpeg_with_format(&concatenated, Some("h264")).unwrap();
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes * 2);
    assert_eq!(result.decoded_yuv[..frame_bytes], pixels_a);
    assert_eq!(result.decoded_yuv[frame_bytes..], pixels_b);
}

// ─── Intra_16x16 path (Phase 6A.10 — lossy) ─────────────────────────

#[test]
fn i16x16_encode_ffmpeg_decodes_cleanly_32x32() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let pixels = deterministic_yuv420p(w, h);
    let bytes = enc.encode_i_frame(&pixels).expect("encode Intra_16x16");
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .unwrap_or_else(|e| panic!("oracle decode failed: {:?}", e));
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
    let expected = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), expected);
}

#[test]
fn i16x16_flat_frame_recovers_approximately() {
    // A flat luma=100 source should decode to approximately flat-100.
    // Phase 6A.10-fu fixed the ffmpeg desync (reversed DC-block scan).
    // Remaining drift is pure quantization — each MB's DC prediction
    // has some residual error that propagates to subsequent MBs.
    // At QP=23 (quality=75) a flat source reconstructs within ~10
    // luma levels, so PSNR > 20 dB.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let pixels = flat_yuv420p(w, h, 100);
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264")).unwrap();

    let y_len = (w * h) as usize;
    let luma_psnr = psnr(&result.decoded_yuv[..y_len], &pixels[..y_len]);
    // Phase 6A polish #8 brought this from ~26 dB to > 50 dB by
    // fixing slice_qp_delta (the decoder was using pic_init_qp=26
    // while the encoder quantized at target_crf). Tightened bound
    // to catch regressions.
    assert!(
        luma_psnr > 40.0,
        "flat-frame luma PSNR too low: {luma_psnr:.1} dB"
    );
}

#[test]
fn i16x16_deterministic_frame_recovers_roughly() {
    // Non-flat frame — lower PSNR bound because AC residuals exist
    // and DC-only prediction can't track a gradient.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(90)).unwrap();
    let pixels = deterministic_yuv420p(w, h);
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264")).unwrap();

    let y_len = (w * h) as usize;
    let luma_psnr = psnr(&result.decoded_yuv[..y_len], &pixels[..y_len]);
    // Smoke-level bound — textured input should at least be
    // recognizable (PSNR > 10 dB).
    assert!(
        luma_psnr > 10.0,
        "gradient-frame luma PSNR critically low: {luma_psnr:.1} dB"
    );
}

#[test]
fn i16x16_smaller_than_i_pcm_output() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (64u32, 48u32);
    let mut enc_pcm = Encoder::new(w, h, Some(75)).unwrap();
    let mut enc_i16 = Encoder::new(w, h, Some(75)).unwrap();
    let pixels = deterministic_yuv420p(w, h);
    let pcm_bytes = enc_pcm.encode_i_frame_pcm(&pixels).unwrap();
    let i16_bytes = enc_i16.encode_i_frame(&pixels).unwrap();
    assert!(
        i16_bytes.len() < pcm_bytes.len(),
        "I_16x16 should compress: i16={} pcm={}",
        i16_bytes.len(),
        pcm_bytes.len()
    );
}

// ─── Phase 6B P-frame integration ───────────────────────────────────

#[test]
fn p_frame_without_reference_errors() {
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let pixels = deterministic_yuv420p(w, h);
    assert!(enc.encode_p_frame(&pixels).is_err());
}

#[test]
fn i_then_p_frame_decodes_cleanly() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let pixels = deterministic_yuv420p(w, h);

    // First IDR, then a P-frame (same content — MV=(0,0) should do).
    let mut bytes = enc.encode_i_frame(&pixels).unwrap();
    let p_bytes = enc.encode_p_frame(&pixels).unwrap();
    bytes.extend_from_slice(&p_bytes);

    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .unwrap_or_else(|e| panic!("oracle decode failed: {:?}", e));
    // Two frames decoded → double yuv420p size.
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(
        result.decoded_yuv.len(),
        frame_bytes * 2,
        "expected 2 decoded frames"
    );
}

#[test]
fn i_then_p_same_content_functional() {
    // Pipeline smoke — I+P sequence decodes to approximately the
    // source content through ffmpeg. PSNR bound reflects the same
    // AC-scale calibration gap noted for intra frames (see
    // deferred-items.md §3); quality polish is a separate task.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let pixels = deterministic_yuv420p(w, h);

    let mut bytes = enc.encode_i_frame(&pixels).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&pixels).unwrap());

    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264")).unwrap();
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    let p_frame = &result.decoded_yuv[frame_bytes..];
    let y_len = (w * h) as usize;
    let luma_psnr = psnr(&p_frame[..y_len], &pixels[..y_len]);
    assert!(
        luma_psnr > 10.0,
        "P-frame luma PSNR critically low: {luma_psnr:.1} dB"
    );
}

#[test]
fn i_then_p_shifted_square_decodes_cleanly() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();

    let mut frame1 = vec![0u8; (w * h * 3 / 2) as usize];
    let y_size = (w * h) as usize;
    let c_size = ((w / 2) * (h / 2)) as usize;
    for y in 0..h {
        for x in 0..w {
            let is_sq = (8..24).contains(&x) && (8..24).contains(&y);
            frame1[(y * w + x) as usize] = if is_sq { 180 } else { 100 };
        }
    }
    for i in 0..2 * c_size {
        frame1[y_size + i] = 128;
    }
    let mut frame2 = vec![0u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            let is_sq = (12..28).contains(&x) && (8..24).contains(&y);
            frame2[(y * w + x) as usize] = if is_sq { 180 } else { 100 };
        }
    }
    for i in 0..2 * c_size {
        frame2[y_size + i] = 128;
    }

    let mut bytes = enc.encode_i_frame(&frame1).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&frame2).unwrap());

    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .expect("oracle should decode clean");
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes * 2);
    // Just verify Layer 0+1 succeed; pixel-accuracy is gated by the
    // AC-scale calibration polish.
}

#[test]
fn i_then_p_horizontal_stripe_motion_decodes_cleanly() {
    // Top half of the P-frame moves right by 4 pixels; bottom half
    // stays still. P_16x8 should be preferred over P_16x16.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();

    let y_size = (w * h) as usize;
    let c_size = ((w / 2) * (h / 2)) as usize;

    let mut frame1 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            frame1[(y * w + x) as usize] = ((x * 7 + y * 5) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame1[y_size + i] = 128;
    }

    let mut frame2 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            // Top half: shifted by +4 horizontally. Bottom half: same
            // as frame1.
            let src_x = if y < h / 2 { (x + w - 4) % w } else { x };
            frame2[(y * w + x) as usize] = ((src_x * 7 + y * 5) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame2[y_size + i] = 128;
    }

    let mut bytes = enc.encode_i_frame(&frame1).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&frame2).unwrap());

    let result =
        decode_via_ffmpeg_with_format(&bytes, Some("h264")).expect("oracle should decode clean");
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes * 2);
}

#[test]
fn i4x4_mixed_content_decodes_cleanly() {
    // Content with fine-grained texture. I_4x4 mode decision should
    // win over I_16x16 on at least some MBs — we just verify the
    // resulting bitstream decodes cleanly via ffmpeg.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let y_size = (w * h) as usize;
    let c_size = ((w / 2) * (h / 2)) as usize;

    // Gradient + noise pattern with per-4×4-block offsets to frustrate
    // I_16x16's flat-plane prediction — I_4x4 should win on detail-rich
    // sub-blocks.
    let mut frame = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            let base = (x * 11 + y * 7) & 0xFF;
            let noise = if ((x / 4) + (y / 4)) & 1 == 0 { 20 } else { 0 };
            frame[(y * w + x) as usize] = (base + noise).min(255) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame[y_size + i] = 128;
    }

    let bytes = enc.encode_i_frame(&frame).unwrap();
    let result =
        decode_via_ffmpeg_with_format(&bytes, Some("h264")).expect("oracle should decode clean");
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes);
}

#[test]
fn i_then_p_fine_grained_motion_decodes_cleanly() {
    // Per-8×8-region motion within a single MB: top-left and
    // bottom-right shift one way, top-right and bottom-left shift the
    // other. This splits the MB such that no partition larger than
    // 8×8 can reconstruct it well — should exercise P_8x8 with
    // sub-MB partitions when the SATD win is large enough.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let y_size = (w * h) as usize;
    let c_size = ((w / 2) * (h / 2)) as usize;

    let mut frame1 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            // Busy 4-pixel-period pattern so fine partitions have
            // enough SATD structure to differentiate.
            frame1[(y * w + x) as usize] =
                (((x / 2) * 31 + (y / 2) * 17) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame1[y_size + i] = 128;
    }

    let mut frame2 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            // Each 4×4 sub-block gets its own shift. Fine-grained
            // motion that P_4x4 can hug but P_16x16 / P_8x8 can't.
            let bx = (x / 4) & 3;
            let by = (y / 4) & 3;
            let sx = match (bx, by) {
                (0, _) | (2, _) => (x + w - 4) % w,
                _ => (x + 4) % w,
            };
            let sy = match (bx, by) {
                (_, 0) | (_, 2) => y,
                _ => (y + h - 4) % h,
            };
            frame2[(y * w + x) as usize] = (((sx / 2) * 31 + (sy / 2) * 17) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame2[y_size + i] = 128;
    }

    let mut bytes = enc.encode_i_frame(&frame1).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&frame2).unwrap());

    let result =
        decode_via_ffmpeg_with_format(&bytes, Some("h264")).expect("oracle should decode clean");
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes * 2);
}

#[test]
fn i_then_p_quadrant_motion_decodes_cleanly() {
    // Four quadrants with independent per-quadrant motion. P_8x8
    // (or finer) should be preferred — at minimum the output must
    // decode cleanly through ffmpeg.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();

    let y_size = (w * h) as usize;
    let c_size = ((w / 2) * (h / 2)) as usize;

    let mut frame1 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            frame1[(y * w + x) as usize] = ((x * 13 + y * 9) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame1[y_size + i] = 128;
    }

    let mut frame2 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            // Per-quadrant shifts: top-left +4x, top-right -4x,
            // bottom-left +4y, bottom-right unshifted.
            let (sx, sy) = match (x < w / 2, y < h / 2) {
                (true, true) => ((x + w - 4) % w, y),
                (false, true) => ((x + 4) % w, y),
                (true, false) => (x, (y + h - 4) % h),
                (false, false) => (x, y),
            };
            frame2[(y * w + x) as usize] = ((sx * 13 + sy * 9) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame2[y_size + i] = 128;
    }

    let mut bytes = enc.encode_i_frame(&frame1).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&frame2).unwrap());

    let result =
        decode_via_ffmpeg_with_format(&bytes, Some("h264")).expect("oracle should decode clean");
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes * 2);
}

#[test]
fn i_then_p_vertical_stripe_motion_decodes_cleanly() {
    // Left half of the P-frame moves down by 4 pixels; right half
    // stays still. P_8x16 should be preferred over P_16x16.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();

    let y_size = (w * h) as usize;
    let c_size = ((w / 2) * (h / 2)) as usize;

    let mut frame1 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            frame1[(y * w + x) as usize] = ((x * 5 + y * 7) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame1[y_size + i] = 128;
    }

    let mut frame2 = vec![128u8; (w * h * 3 / 2) as usize];
    for y in 0..h {
        for x in 0..w {
            // Left half: shifted by +4 vertically. Right half: same.
            let src_y = if x < w / 2 { (y + h - 4) % h } else { y };
            frame2[(y * w + x) as usize] = ((x * 5 + src_y * 7) & 0xFF) as u8;
        }
    }
    for i in 0..2 * c_size {
        frame2[y_size + i] = 128;
    }

    let mut bytes = enc.encode_i_frame(&frame1).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&frame2).unwrap());

    let result =
        decode_via_ffmpeg_with_format(&bytes, Some("h264")).expect("oracle should decode clean");
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes * 2);
}

// ─── Phase 6C.6c: first CABAC I-frame ───────────────────────────────

#[test]
fn cabac_i16x16_zero_residual_ffmpeg_decodes_cleanly_32x32() {
    // Phase 6C.6c scope: CABAC I_16x16 with forced-zero residuals.
    // Validates the CABAC byte stream framing (SPS Main profile, PPS
    // with entropy_coding_mode_flag=1, cabac_alignment_one_bit,
    // terminate-coded end_of_slice_flag) by round-tripping through
    // ffmpeg.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let pixels = flat_yuv420p(w, h, 128);
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .unwrap_or_else(|e| panic!("CABAC I-frame ffmpeg decode failed: {:?}", e));
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
    let expected = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), expected);
}

#[test]
fn cabac_i16x16_zero_residual_ffmpeg_decodes_64x48() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (64u32, 48u32);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let pixels = flat_yuv420p(w, h, 64);
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .unwrap_or_else(|e| panic!("CABAC I-frame ffmpeg decode failed: {:?}", e));
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
}

// ─── Phase 6C.6d: full-residual CABAC I-frame ────────────────────

#[test]
fn cabac_i16x16_full_residual_ffmpeg_decodes_32x32() {
    // Non-flat deterministic pattern → non-trivial residuals → exercise
    // the Intra16x16DCLevel, Intra16x16ACLevel, ChromaDCLevel,
    // ChromaACLevel paths + same-MB coded_block_flag derivation.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(90)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let pixels = deterministic_yuv420p(w, h);
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .unwrap_or_else(|e| panic!("CABAC I-frame ffmpeg decode failed: {:?}", e));
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);
    let expected = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), expected);
    let y_size = (w as usize) * (h as usize);
    let y_psnr = psnr(&pixels[..y_size], &result.decoded_yuv[..y_size]);
    // Smoke-level bound matching the CAVLC gradient test — textured
    // input should at least be recognizable (PSNR > 10 dB).
    assert!(
        y_psnr > 10.0,
        "CABAC full-residual Y PSNR critically low: {y_psnr:.2}"
    );
}

#[test]
fn cabac_matches_cavlc_psnr_within_tolerance() {
    // CABAC and CAVLC encode the same coefficients through different
    // entropy coders. On ffmpeg decode, the Y PSNR should be very
    // close (same quantized coefficients → same reconstruction). If
    // CABAC's PSNR is materially worse than CAVLC's, that's a signal
    // the bitstream is mis-parsing.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let pixels = deterministic_yuv420p(w, h);
    let y_size = (w as usize) * (h as usize);

    // CAVLC baseline.
    let mut enc_cavlc = Encoder::new(w, h, Some(90)).unwrap();
    let cavlc_bytes = enc_cavlc.encode_i_frame(&pixels).unwrap();
    let cavlc_result = decode_via_ffmpeg_with_format(&cavlc_bytes, Some("h264")).unwrap();
    let cavlc_psnr = psnr(&pixels[..y_size], &cavlc_result.decoded_yuv[..y_size]);

    // CABAC.
    let mut enc_cabac = Encoder::new(w, h, Some(90)).unwrap();
    enc_cabac.entropy_mode = EntropyMode::Cabac;
    let cabac_bytes = enc_cabac.encode_i_frame(&pixels).unwrap();
    let cabac_result = decode_via_ffmpeg_with_format(&cabac_bytes, Some("h264")).unwrap();
    let cabac_psnr = psnr(&pixels[..y_size], &cabac_result.decoded_yuv[..y_size]);

    // Allow 2 dB tolerance — CABAC/CAVLC reconstruction differs
    // slightly because CAVLC trellis paths aren't aware of CABAC-level
    // bit cost (minor RD-decision divergence).
    assert!(
        (cavlc_psnr - cabac_psnr).abs() < 2.0,
        "CABAC PSNR ({cabac_psnr:.2} dB) diverges from CAVLC ({cavlc_psnr:.2} dB) — likely bitstream bug"
    );
}

fn high_freq_yuv420p(width: u32, height: u32) -> Vec<u8> {
    // Checkerboard-ish pattern with high local variance → SATD-based
    // mode decision tends to prefer I_4x4 over I_16x16 DC/V/H/Plane
    // modes.
    let y_size = (width * height) as usize;
    let c_size = (width / 2 * height / 2) as usize;
    let mut buf = vec![128u8; y_size + 2 * c_size];
    for yy in 0..height {
        for xx in 0..width {
            let v = if ((xx / 2) + (yy / 2)) & 1 == 0 { 40 } else { 200 };
            buf[(yy * width + xx) as usize] = v;
        }
    }
    for i in 0..2 * c_size {
        buf[y_size + i] = 128;
    }
    buf
}

#[test]
fn cabac_i4x4_ffmpeg_decodes_high_freq_pattern() {
    // High-frequency pattern exercises the I_4x4 branch of
    // write_intra_macroblock_cabac. Validates CABAC I_4x4 mode
    // signaling + LumaLevel4x4 residuals + coded_block_pattern same-MB
    // neighbor fix end-to-end via ffmpeg.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(90)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let pixels = high_freq_yuv420p(w, h);
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    let result = decode_via_ffmpeg_with_format(&bytes, Some("h264"))
        .unwrap_or_else(|e| panic!("CABAC I_4x4 ffmpeg decode failed: {:?}", e));
    assert_eq!(result.width, w);
    assert_eq!(result.height, h);

    // Compare to CAVLC baseline on the same pattern.
    let mut enc_cavlc = Encoder::new(w, h, Some(90)).unwrap();
    let cavlc_bytes = enc_cavlc.encode_i_frame(&pixels).unwrap();
    let cavlc_result = decode_via_ffmpeg_with_format(&cavlc_bytes, Some("h264")).unwrap();

    let y_size = (w as usize) * (h as usize);
    let cabac_psnr = psnr(&pixels[..y_size], &result.decoded_yuv[..y_size]);
    let cavlc_psnr = psnr(&pixels[..y_size], &cavlc_result.decoded_yuv[..y_size]);
    // Both paths select the same mode per MB (SATD is identical) and
    // quantize identically → decoded PSNR should match within 2 dB.
    assert!(
        (cavlc_psnr - cabac_psnr).abs() < 2.0,
        "CABAC I_4x4 diverges: cabac={cabac_psnr:.2} cavlc={cavlc_psnr:.2}"
    );
}

#[test]
fn cabac_p_frame_ffmpeg_decodes_cleanly() {
    // I + P frame sequence via CABAC. Validates P-slice syntax:
    // mb_skip_flag, mb_type_P, mvd, cbp (fixed same-MB logic),
    // LumaLevel4x4 inter residuals.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let mut enc = Encoder::new(w, h, Some(90)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;

    let frame1 = deterministic_yuv420p(w, h);
    let mut frame2 = frame1.clone();
    // Slight translation in the Y plane → motion the encoder should
    // pick up.
    let y_size = (w * h) as usize;
    for yy in 0..h {
        for xx in 0..w {
            let src_x = if xx >= 2 { xx - 2 } else { xx };
            let src_y = yy;
            frame2[(yy * w + xx) as usize] = frame1[(src_y * w + src_x) as usize];
        }
    }

    let mut bytes = enc.encode_i_frame(&frame1).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&frame2).unwrap());

    let result =
        decode_via_ffmpeg_with_format(&bytes, Some("h264")).expect("ffmpeg should decode");
    let frame_bytes = (w as usize) * (h as usize) * 3 / 2;
    assert_eq!(result.decoded_yuv.len(), frame_bytes * 2);
    let p_psnr = psnr(&frame2[..y_size], &result.decoded_yuv[frame_bytes..frame_bytes + y_size]);
    assert!(
        p_psnr > 10.0,
        "CABAC P-frame luma PSNR critically low: {p_psnr:.1} dB"
    );
}

#[test]
fn cabac_p_frame_matches_cavlc_psnr() {
    // P-frame parity check: same source through CABAC and CAVLC should
    // produce ffmpeg-decoded Y PSNR within 2 dB.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let (w, h) = (32u32, 32u32);
    let frame1 = deterministic_yuv420p(w, h);
    let mut frame2 = frame1.clone();
    for yy in 0..h {
        for xx in 0..w {
            let src_x = if xx >= 2 { xx - 2 } else { xx };
            frame2[(yy * w + xx) as usize] = frame1[(yy * w + src_x) as usize];
        }
    }
    let y_size = (w as usize) * (h as usize);
    let frame_bytes = y_size * 3 / 2;

    // CAVLC
    let mut enc_cavlc = Encoder::new(w, h, Some(90)).unwrap();
    let mut cav_bytes = enc_cavlc.encode_i_frame(&frame1).unwrap();
    cav_bytes.extend_from_slice(&enc_cavlc.encode_p_frame(&frame2).unwrap());
    let cav_result = decode_via_ffmpeg_with_format(&cav_bytes, Some("h264")).unwrap();
    let cav_p_psnr =
        psnr(&frame2[..y_size], &cav_result.decoded_yuv[frame_bytes..frame_bytes + y_size]);

    // CABAC
    let mut enc_cabac = Encoder::new(w, h, Some(90)).unwrap();
    enc_cabac.entropy_mode = EntropyMode::Cabac;
    let mut cab_bytes = enc_cabac.encode_i_frame(&frame1).unwrap();
    cab_bytes.extend_from_slice(&enc_cabac.encode_p_frame(&frame2).unwrap());
    let cab_result = decode_via_ffmpeg_with_format(&cab_bytes, Some("h264")).unwrap();
    let cab_p_psnr =
        psnr(&frame2[..y_size], &cab_result.decoded_yuv[frame_bytes..frame_bytes + y_size]);

    assert!(
        (cav_p_psnr - cab_p_psnr).abs() < 2.0,
        "CABAC P-frame diverges: cabac={cab_p_psnr:.2} cavlc={cav_p_psnr:.2}"
    );
}

// ─── Real-world content: strict decode (no ffmpeg errors) ────────────

/// Run ffmpeg and return `true` if the stream decodes with ZERO errors
/// on stderr. Oracles' `decode_via_ffmpeg_with_format` passes as long
/// as ffmpeg exits 0 — but ffmpeg exits 0 even with concealment-covered
/// parse errors. This helper counts non-empty stderr lines after an
/// `-loglevel error` run.
fn ffmpeg_decodes_cleanly(bytes: &[u8]) -> Result<(), String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-f",
            "null",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg spawn: {e}"))?;
    let mut stdin = child.stdin.take().unwrap();
    let bytes = bytes.to_vec();
    let h = std::thread::spawn(move || stdin.write_all(&bytes));
    let out = child.wait_with_output().map_err(|e| format!("ffmpeg wait: {e}"))?;
    h.join().unwrap().map_err(|e| format!("stdin writer: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let errors: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    if errors.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "ffmpeg emitted {} stderr lines. first 5:\n{}",
            errors.len(),
            errors.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
        ))
    }
}

fn load_real_world(name: &str) -> Vec<u8> {
    std::fs::read(format!("test-vectors/video/h264/real-world/{name}"))
        .unwrap_or_else(|e| panic!("load {name}: {e}"))
}

/// Decode a raw Annex-B H.264 stream through ffmpeg to yuv420p raw
/// pixels. Returns the decoded YUV bytes (all frames concatenated) or
/// an error with ffmpeg's stderr summary. Zero-tolerance on stderr:
/// any error line fails the call.
fn ffmpeg_decode_to_yuv(bytes: &[u8]) -> Result<Vec<u8>, String> {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "h264",
            "-i",
            "pipe:0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "yuv420p",
            "-",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("ffmpeg spawn: {e}"))?;
    let mut stdin = child.stdin.take().unwrap();
    let bytes = bytes.to_vec();
    let h = std::thread::spawn(move || stdin.write_all(&bytes));
    let out = child.wait_with_output().map_err(|e| format!("ffmpeg wait: {e}"))?;
    h.join().unwrap().map_err(|e| format!("stdin writer: {e}"))?;
    let stderr = String::from_utf8_lossy(&out.stderr);
    let errors: Vec<&str> = stderr.lines().filter(|l| !l.trim().is_empty()).collect();
    if !errors.is_empty() {
        return Err(format!(
            "ffmpeg emitted {} stderr lines. first 5:\n{}",
            errors.len(),
            errors.iter().take(5).cloned().collect::<Vec<_>>().join("\n")
        ));
    }
    Ok(out.stdout)
}

/// Per-frame Y-PSNR between `src` and `dec` (both yuv420p, same
/// dimensions, potentially multiple frames). Returns average PSNR (dB)
/// across all matching frames.
fn average_y_psnr(src: &[u8], dec: &[u8], w: u32, h: u32) -> f64 {
    let frame = (w * h * 3 / 2) as usize;
    let luma = (w * h) as usize;
    let frames = src.len().min(dec.len()) / frame;
    assert!(frames > 0, "no full frames to compare");
    let mut psnr_sum = 0.0;
    for f in 0..frames {
        let a = &src[f * frame..f * frame + luma];
        let b = &dec[f * frame..f * frame + luma];
        let mse = mse(a, b);
        let psnr = if mse == 0.0 {
            100.0
        } else {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        };
        psnr_sum += psnr;
    }
    psnr_sum / frames as f64
}

#[test]
fn real_world_img4138_32x32_cavlc_decodes_cleanly() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let pixels = load_real_world("img4138_32x32_f1.yuv");
    let mut enc = Encoder::new(32, 32, Some(80)).unwrap();
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("CAVLC 32x32 real content decode errors:\n{e}"));
}

#[test]
fn real_world_img4138_32x32_cabac_decodes_cleanly() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let pixels = load_real_world("img4138_32x32_f1.yuv");
    let mut enc = Encoder::new(32, 32, Some(80)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("CABAC 32x32 real content decode errors:\n{e}"));
}

#[test]
fn real_world_img4138_pcm_decodes_cleanly() {
    // I_PCM sanity check — bit-exact lossless. If this fails, framing
    // (SPS/PPS/NAL/emulation-prevention/slice header) is broken;
    // otherwise the bug is in the residual/intra path.
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let pixels = load_real_world("img4138_32x32_f1.yuv");
    let mut enc = Encoder::new(32, 32, None).unwrap();
    let bytes = enc.encode_i_frame_pcm(&pixels).unwrap();
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("I_PCM 32x32 real content decode errors:\n{e}"));
}

#[test]
fn probe_deterministic_pattern_strict_decode() {
    if !system_has_ffmpeg() {
        return;
    }
    let (w, h) = (32u32, 32u32);
    let pixels = deterministic_yuv420p(w, h);
    let mut enc = Encoder::new(w, h, Some(80)).unwrap();
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    match ffmpeg_decodes_cleanly(&bytes) {
        Ok(()) => println!("PASS: deterministic decodes cleanly"),
        Err(e) => panic!("FAIL: deterministic has errors: {e}"),
    }
}

#[test]
fn probe_flat_strict_decode() {
    if !system_has_ffmpeg() {
        return;
    }
    let (w, h) = (32u32, 32u32);
    let pixels = flat_yuv420p(w, h, 100);
    let mut enc = Encoder::new(w, h, Some(75)).unwrap();
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    match ffmpeg_decodes_cleanly(&bytes) {
        Ok(()) => println!("PASS: flat decodes cleanly"),
        Err(e) => panic!("FAIL: flat has errors: {e}"),
    }
}

/// Decode our own encoder's output via OUR own decoder. Returns Err
/// at the first MB that fails to parse; the bit position tells us
/// where in the stream the encoder→decoder pair desynced.
fn self_decode(bytes: &[u8]) -> Result<usize, String> {
    use phasm_core::codec::h264::bitstream::{
        parse_nal_units_annexb, remove_emulation_prevention_with_map, RbspReader,
    };
    use phasm_core::codec::h264::macroblock::{parse_macroblock_with_recon, NeighborContext};
    use phasm_core::codec::h264::slice::parse_slice_header;
    use phasm_core::codec::h264::sps::{parse_pps, parse_sps};
    use phasm_core::codec::h264::NalType;

    let nals = parse_nal_units_annexb(bytes).map_err(|e| format!("NAL parse: {e}"))?;
    let sps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::SPS)
        .ok_or("no SPS")?;
    let pps_nal = nals
        .iter()
        .find(|n| n.nal_type == NalType::PPS)
        .ok_or("no PPS")?;
    let slice_nal = nals
        .iter()
        .find(|n| matches!(n.nal_type, NalType::SLICE_IDR | NalType::SLICE))
        .ok_or("no SLICE")?;

    let sps = parse_sps(&sps_nal.rbsp).map_err(|e| format!("parse_sps: {e}"))?;
    let pps = parse_pps(&pps_nal.rbsp).map_err(|e| format!("parse_pps: {e}"))?;

    let hdr = parse_slice_header(
        &slice_nal.rbsp,
        &sps,
        &pps,
        slice_nal.nal_type,
        slice_nal.nal_ref_idc,
    )
    .map_err(|e| format!("parse_slice_header: {e}"))?;

    let (rbsp, ep_map) = remove_emulation_prevention_with_map(&slice_nal.rbsp);
    let mut reader = RbspReader::new(&rbsp);
    reader
        .skip_bits(hdr.data_bit_offset as u32)
        .map_err(|e| format!("skip header: {e}"))?;

    let mb_w = sps.pic_width_in_mbs;
    let mb_h = sps.pic_height_in_map_units;
    let mut ctx = NeighborContext::new(mb_w, mb_h);
    let mut current_qp = (pps.pic_init_qp_minus26 as i32 + 26) + hdr.slice_qp_delta as i32;

    for mb_y in 0..mb_h {
        for mb_x in 0..mb_w {
            parse_macroblock_with_recon(
                &mut reader,
                hdr.slice_type,
                mb_x,
                mb_y,
                &sps,
                &pps,
                &mut ctx,
                &ep_map,
                &rbsp,
                &mut current_qp,
                pps.num_ref_idx_l0_default,
                false,
                None,
            )
            .map_err(|e| format!("MB ({mb_x},{mb_y}): {e}"))?;
        }
    }
    Ok((mb_w * mb_h) as usize)
}

#[test]
fn self_decode_real_world_img4138_32x32_cavlc() {
    let pixels = load_real_world("img4138_32x32_f1.yuv");
    let mut enc = Encoder::new(32, 32, Some(80)).unwrap();
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    match self_decode(&bytes) {
        Ok(n) => println!("self-decode OK: {n} MBs parsed cleanly"),
        Err(e) => panic!("self-decode FAILED: {e}"),
    }
}

#[test]
fn self_decode_deterministic_pattern() {
    let (w, h) = (32u32, 32u32);
    let pixels = deterministic_yuv420p(w, h);
    let mut enc = Encoder::new(w, h, Some(80)).unwrap();
    // self_decode only supports CAVLC — pin entropy mode for this test.
    enc.entropy_mode = EntropyMode::Cavlc;
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    match self_decode(&bytes) {
        Ok(n) => println!("self-decode OK: {n} MBs parsed cleanly"),
        Err(e) => panic!("self-decode FAILED: {e}"),
    }
}

#[test]
fn real_world_img4138_64x48_cavlc_decodes_cleanly() {
    if !system_has_ffmpeg() {
        return;
    }
    let pixels = load_real_world("img4138_64x48_f5.yuv");
    let frame_size = (64 * 48 * 3 / 2) as usize;
    let frame0 = &pixels[..frame_size];
    let mut enc = Encoder::new(64, 48, Some(80)).unwrap();
    let bytes = enc.encode_i_frame(frame0).unwrap();
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("CAVLC 64x48 real content decode errors:\n{e}"));
}

#[test]
fn real_world_img4138_128x80_cavlc_decodes_cleanly() {
    if !system_has_ffmpeg() {
        return;
    }
    let pixels = load_real_world("img4138_128x80_f10.yuv");
    let frame_size = (128 * 80 * 3 / 2) as usize;
    let frame0 = &pixels[..frame_size];
    let mut enc = Encoder::new(128, 80, Some(80)).unwrap();
    let bytes = enc.encode_i_frame(frame0).unwrap();
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("CAVLC 128x80 real content decode errors:\n{e}"));
}

#[test]
fn real_world_img4138_64x48_cavlc_i_then_p_decodes_cleanly() {
    if !system_has_ffmpeg() {
        return;
    }
    let pixels = load_real_world("img4138_64x48_f5.yuv");
    let frame_size = (64 * 48 * 3 / 2) as usize;
    let mut enc = Encoder::new(64, 48, Some(80)).unwrap();
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    bytes.extend_from_slice(
        &enc.encode_p_frame(&pixels[frame_size..2 * frame_size])
            .unwrap(),
    );
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("I+P 64x48 real content decode errors:\n{e}"));
}

#[test]
fn probe_deterministic_64x48_strict() {
    if !system_has_ffmpeg() {
        return;
    }
    let pixels = deterministic_yuv420p(64, 48);
    let mut enc = Encoder::new(64, 48, Some(80)).unwrap();
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    if let Err(e) = ffmpeg_decodes_cleanly(&bytes) {
        panic!("64x48 deterministic decode errors:\n{e}");
    }
}

#[test]
fn self_decode_deterministic_64x48_strict() {
    let pixels = deterministic_yuv420p(64, 48);
    let mut enc = Encoder::new(64, 48, Some(80)).unwrap();
    // self_decode only supports CAVLC — pin entropy mode for this test.
    enc.entropy_mode = EntropyMode::Cavlc;
    let bytes = enc.encode_i_frame(&pixels).unwrap();
    match self_decode(&bytes) {
        Ok(n) => println!("self-decode OK: {n} MBs parsed"),
        Err(e) => panic!("self-decode FAILED: {e}"),
    }
}

/// Best-profile CAVLC real-world visual quality test — encoder defaults
/// (all P-partition types enabled), largest available real-world vector
/// (10-frame 128×80 IMG_4138 sequence). Asserts ffmpeg-clean decode +
/// Y-PSNR ≥ 25 dB averaged over all frames. Regression gate: any change
/// that crashes quality to concealment levels (< 15 dB) or breaks
/// decodability fails here.
#[test]
fn cavlc_best_profile_real_world_ffmpeg_and_psnr() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let pixels = load_real_world("img4138_128x80_f10.yuv");
    let frame_size = (128 * 80 * 3 / 2) as usize;
    let n_frames = pixels.len() / frame_size;
    assert!(n_frames >= 2, "need at least I + 1 P frame");

    let mut enc = Encoder::new(128, 80, Some(80)).unwrap();
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    for f in 1..n_frames {
        let start = f * frame_size;
        bytes.extend_from_slice(
            &enc.encode_p_frame(&pixels[start..start + frame_size])
                .unwrap(),
        );
    }
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("ffmpeg decode dirty on CAVLC best-profile sequence:\n{e}"));
    let decoded = ffmpeg_decode_to_yuv(&bytes)
        .unwrap_or_else(|e| panic!("ffmpeg decode to yuv failed:\n{e}"));
    let psnr = average_y_psnr(&pixels[..frame_size * n_frames], &decoded, 128, 80);
    println!("CAVLC best-profile Y-PSNR over {n_frames} frames: {psnr:.2} dB");
    assert!(
        psnr >= 25.0,
        "CAVLC best-profile Y-PSNR {psnr:.2} dB below 25 dB threshold"
    );
}

/// Intra-in-P fallback test — I-frame is solid black, P-frame is random
/// noise, so every P-MB's best inter prediction has huge SATD against
/// the flat black reference, while I_16x16 SATD from noisy neighbors
/// is the only reasonable option. Drives the fallback branch for most
/// P-MBs. Regression-gate: ffmpeg must decode the resulting P-slice
/// cleanly (i.e. the P-slice intra mb_type codenum 5..=30 + mv_grid
/// "intra" marking is spec-conformant).
#[test]
fn intra_in_p_fallback_decodes_cleanly() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let w: u32 = 32;
    let h: u32 = 32;
    let frame_size = (w * h * 3 / 2) as usize;

    // I-frame: flat mid-gray so its reconstruction is uniform.
    let mut i_frame = vec![128u8; frame_size];
    for v in i_frame.iter_mut().take((w * h) as usize) {
        *v = 128;
    }
    // P-frame: deterministic "noise" with strong spatial structure so
    // intra 16x16 DC/Plane predictors can match reasonably but the
    // motion-compensated copy of the flat I-frame does not.
    let mut p_frame = vec![128u8; frame_size];
    for y in 0..h {
        for x in 0..w {
            // High-contrast checkerboard: tiles of 128 vs 16.
            let tile = ((x / 4) + (y / 4)) & 1;
            p_frame[(y * w + x) as usize] = if tile == 0 { 16 } else { 240 };
        }
    }

    let mut enc = Encoder::new(w, h, Some(80)).unwrap();
    let mut bytes = enc.encode_i_frame(&i_frame).unwrap();
    bytes.extend_from_slice(&enc.encode_p_frame(&p_frame).unwrap());

    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("ffmpeg decode dirty on intra-in-P fallback test:\n{e}"));
}

/// Best-profile CABAC real-world visual quality test — same vector and
/// bar as the CAVLC counterpart, but with `EntropyMode::Cabac`. Same
/// regression-gate intent.
#[test]
fn cabac_best_profile_real_world_ffmpeg_and_psnr() {
    if !system_has_ffmpeg() {
        eprintln!("skipping — ffmpeg not installed");
        return;
    }
    let pixels = load_real_world("img4138_128x80_f10.yuv");
    let frame_size = (128 * 80 * 3 / 2) as usize;
    let n_frames = pixels.len() / frame_size;
    assert!(n_frames >= 2, "need at least I + 1 P frame");

    let mut enc = Encoder::new(128, 80, Some(80)).unwrap();
    enc.entropy_mode = EntropyMode::Cabac;
    let mut bytes = enc.encode_i_frame(&pixels[..frame_size]).unwrap();
    for f in 1..n_frames {
        let start = f * frame_size;
        bytes.extend_from_slice(
            &enc.encode_p_frame(&pixels[start..start + frame_size])
                .unwrap(),
        );
    }
    ffmpeg_decodes_cleanly(&bytes)
        .unwrap_or_else(|e| panic!("ffmpeg decode dirty on CABAC best-profile sequence:\n{e}"));
    let decoded = ffmpeg_decode_to_yuv(&bytes)
        .unwrap_or_else(|e| panic!("ffmpeg decode to yuv failed:\n{e}"));
    let psnr = average_y_psnr(&pixels[..frame_size * n_frames], &decoded, 128, 80);
    println!("CABAC best-profile Y-PSNR over {n_frames} frames: {psnr:.2} dB");
    assert!(
        psnr >= 25.0,
        "CABAC best-profile Y-PSNR {psnr:.2} dB below 25 dB threshold"
    );
}

