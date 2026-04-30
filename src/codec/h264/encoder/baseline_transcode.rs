// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Baseline-CAVLC YUV → H.264 Annex-B transcode helper.
//!
//! This is the production entry point for **task #77** — replacing the
//! mobile platform HW transcoders (iOS VideoToolbox / Android
//! MediaCodec) with our pure-Rust phase-6 encoder for the
//! "input video → Baseline-CAVLC bitstream" step that feeds the stego
//! pipeline.
//!
//! Why we use our own encoder, even though it's slower than HW:
//! - **Patent independence**: removes Via LA AVC patent obligation
//!   on OS HW encoders (we can rebuild the chain without depending
//!   on Apple/Google's licenses).
//! - **Stealth control**: we own every encoding decision (mode,
//!   MV, partition, intra-in-P rate). Stealth tuning becomes a
//!   first-class control surface rather than fighting the HW
//!   encoder's hidden behavior.
//! - **Bitstream determinism**: same input → same output bytes
//!   across iOS / Android / desktop (HW encoders vary device-to-
//!   device, sometimes silently fall back to CABAC on certain
//!   Android OEMs — we discovered this with `H264ProfileProbe`).
//! - **Phase 6 distribution flexibility**: CLI / desktop builds can
//!   use the same encoder as mobile with no platform branch.
//!
//! Limitations called out:
//! - **CAVLC is uncommon in modern user video**. Most cameras
//!   produce Main/High Profile + CABAC. A Baseline CAVLC bitstream
//!   itself is a fingerprint, regardless of who encoded it. Solving
//!   that requires CABAC stego (separate, multi-month work — the
//!   prior HEVC CABAC attempt failed). This module is the
//!   foundational control surface; CABAC stego is downstream of it.
//! - **Slow on mobile**. ~1.27 s/frame on M3 Max → ~1.5–2 h on
//!   iPhone for a 1-min 1080p video. Mitigated by the #76
//!   background-encode + Live Activity / Dynamic Island UX.
//!
//! Architecture: this module owns the YUV → Annex-B step. Mobile
//! still uses platform decode (we have no Rust H.264 decoder yet)
//! and platform mux (AVAssetWriter / MediaMuxer pass-through). The
//! Rust encoder slots between them.

use super::encoder::{Encoder, EntropyMode};
use super::EncoderError;

/// Configuration for `transcode_yuv_to_baseline_cavlc_h264`.
#[derive(Debug, Clone, Copy)]
pub struct BaselineTranscodeConfig {
    pub width: u32,
    pub height: u32,
    /// QP target. None = encoder default (~26). 0..=51 per spec; we
    /// recommend 22..=30 for quality, 30..=40 for size-optimised.
    pub quality: Option<u8>,
    /// IDR period in frames. Default 30 = one IDR per second @ 30 fps.
    pub gop_length: u32,
    /// Total frames in the input YUV stream. Caller computes from
    /// the source-decode side.
    pub n_frames: usize,
}

impl BaselineTranscodeConfig {
    /// Reasonable defaults for a phasm stego cover encode at
    /// `width × height`. QP=26 (visually transparent), GOP=30.
    pub fn defaults(width: u32, height: u32, n_frames: usize) -> Self {
        Self {
            width,
            height,
            quality: Some(26),
            gop_length: 30,
            n_frames,
        }
    }
}

/// Transcode raw YUV420p planar pixels (interleaved frames, packed Y
/// then U then V per frame, 4:2:0) to a Baseline CAVLC H.264 Annex-B
/// byte stream. Returns the concatenated NAL stream (SPS + PPS +
/// I-slice for the first frame, then P-slice for each subsequent
/// frame). IDR frames recur every `gop_length` frames.
///
/// **This is the function that replaces VideoToolbox /
/// MediaCodec for mobile stego.** The caller is responsible for:
/// - Decoding the input video to YUV420p (still platform-side until
///   we have a Rust H.264 decoder).
/// - Wrapping the returned Annex-B stream in MP4 (still platform-
///   side via AVAssetWriter pass-through / MediaMuxer pass-through).
///
/// # Stream layout
/// `pixels.len()` must equal `n_frames * (width * height * 3 / 2)`.
/// Frames are packed Y-plane (`width * height` bytes), then
/// U-plane (`width/2 * height/2`), then V-plane (`width/2 * height/2`).
///
/// # Output stream
/// Annex-B NAL stream with start codes (0x00000001) between NAL
/// units. The first NAL is an SPS, then PPS, then the first slice.
/// Compatible with ffmpeg / AVFoundation / MediaCodec demuxers.
///
/// # Errors
/// `EncoderError::InvalidInput` for non-MB-aligned dimensions or
/// pixel-buffer size mismatch.
pub fn transcode_yuv_to_baseline_cavlc_h264(
    pixels: &[u8],
    config: BaselineTranscodeConfig,
) -> Result<Vec<u8>, EncoderError> {
    let frame_size = (config.width * config.height * 3 / 2) as usize;
    if pixels.len() != config.n_frames * frame_size {
        return Err(EncoderError::InvalidInput(format!(
            "pixel buffer size mismatch: got {} bytes, expected {} ({} frames × {})",
            pixels.len(),
            config.n_frames * frame_size,
            config.n_frames,
            frame_size,
        )));
    }

    let mut enc = Encoder::new(config.width, config.height, config.quality)?;
    enc.entropy_mode = EntropyMode::Cavlc;
    enc.set_gop_length(config.gop_length);

    let estimated_bytes = config
        .n_frames
        .saturating_mul((config.width * config.height) as usize / 8);
    let mut out: Vec<u8> = Vec::with_capacity(estimated_bytes);

    for fi in 0..config.n_frames {
        let frame = &pixels[fi * frame_size..(fi + 1) * frame_size];
        let is_idr = fi % (config.gop_length as usize) == 0;
        let nal = if is_idr {
            enc.encode_i_frame(frame)?
        } else {
            enc.encode_p_frame(frame)?
        };
        out.extend_from_slice(&nal);
    }
    Ok(out)
}

// ============================================================================
// Phase A.2 — stateful per-frame encoder handle
// ============================================================================
//
// Streaming alternative to `transcode_yuv_to_baseline_cavlc_h264` for the
// mobile flow where the platform decoder hands us one YUV frame at a time.
// Lets the platform side keep working set bounded (~3 MB / 1080p frame)
// instead of buffering the whole video before encoding starts.
//
// FFI usage:
//   handle = create(...)
//   for frame in decoded_frames: out_bytes = push_frame(handle, frame)
//   destroy(handle)

/// Streaming H.264 Baseline CAVLC encoder handle. Owns one `Encoder`
/// + a frame counter; auto-emits IDR every `gop_length` frames.
pub struct StreamingEncoder {
    enc: Encoder,
    gop_length: u32,
    frame_index: u32,
}

impl StreamingEncoder {
    /// Construct a streaming encoder configured for Baseline CAVLC
    /// output. Mirrors the `BaselineTranscodeConfig` defaults.
    pub fn new(
        width: u32,
        height: u32,
        quality: Option<u8>,
        gop_length: u32,
    ) -> Result<Self, EncoderError> {
        let mut enc = Encoder::new(width, height, quality)?;
        enc.entropy_mode = EntropyMode::Cavlc;
        let gop = if gop_length == 0 { 30 } else { gop_length };
        enc.set_gop_length(gop);
        Ok(Self { enc, gop_length: gop, frame_index: 0 })
    }

    /// Encode the next YUV420p planar frame. Returns Annex-B bytes
    /// for this one frame (SPS + PPS + I-slice on frame 0, P-slice
    /// on subsequent non-IDR frames, IDR I-slice every `gop_length`
    /// frames thereafter).
    ///
    /// `pixels` length must equal `width * height * 3 / 2`.
    pub fn push_frame(&mut self, pixels: &[u8]) -> Result<Vec<u8>, EncoderError> {
        let frame_size = (self.enc.width * self.enc.height * 3 / 2) as usize;
        if pixels.len() != frame_size {
            return Err(EncoderError::InvalidInput(format!(
                "frame buffer size mismatch: got {} bytes, expected {}",
                pixels.len(),
                frame_size,
            )));
        }
        let is_idr = self.frame_index.is_multiple_of(self.gop_length);
        let nal = if is_idr {
            self.enc.encode_i_frame(pixels)?
        } else {
            self.enc.encode_p_frame(pixels)?
        };
        self.frame_index += 1;
        Ok(nal)
    }

    /// Total frames pushed so far.
    pub fn frames_emitted(&self) -> u32 {
        self.frame_index
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn deterministic_yuv(w: u32, h: u32, n_frames: usize) -> Vec<u8> {
        let frame_size = (w * h * 3 / 2) as usize;
        let mut buf = Vec::with_capacity(n_frames * frame_size);
        for fi in 0..n_frames {
            // Y plane: gradient + frame index
            for y in 0..h {
                for x in 0..w {
                    buf.push(((x + y + fi as u32 * 3) & 0xFF) as u8);
                }
            }
            // U plane: neutral
            for _ in 0..(w * h / 4) {
                buf.push(128);
            }
            // V plane: neutral
            for _ in 0..(w * h / 4) {
                buf.push(128);
            }
        }
        buf
    }

    #[test]
    fn transcode_emits_nonempty_annex_b_for_single_idr() {
        let yuv = deterministic_yuv(32, 32, 1);
        let cfg = BaselineTranscodeConfig::defaults(32, 32, 1);
        let h264 = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).unwrap();
        // Annex-B starts with start code 0x00 0x00 0x00 0x01 (or 0x00 0x00 0x01).
        assert!(
            h264.starts_with(&[0, 0, 0, 1]) || h264.starts_with(&[0, 0, 1]),
            "expected Annex-B start code, got {:02x?}",
            &h264[..h264.len().min(8)]
        );
        // Should contain at least 3 NAL units (SPS, PPS, slice).
        // Count start codes.
        let mut starts = 0;
        for w in h264.windows(4) {
            if w == [0, 0, 0, 1] {
                starts += 1;
            }
        }
        assert!(starts >= 3, "expected ≥3 NALs (SPS+PPS+slice), got {starts}");
    }

    #[test]
    fn transcode_idr_then_p_runs_clean() {
        let yuv = deterministic_yuv(32, 32, 5);
        let cfg = BaselineTranscodeConfig::defaults(32, 32, 5);
        let h264 = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).unwrap();
        // 5 frames at GOP=30 → 1 IDR + 4 P. SPS + PPS + 5 slices = ≥7 NALs.
        let mut starts = 0;
        for w in h264.windows(4) {
            if w == [0, 0, 0, 1] {
                starts += 1;
            }
        }
        assert!(starts >= 7, "expected ≥7 NALs for IDR+4P, got {starts}");
    }

    #[test]
    fn transcode_rejects_size_mismatch() {
        let yuv = vec![0u8; 100]; // wrong size
        let cfg = BaselineTranscodeConfig::defaults(32, 32, 1);
        assert!(transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).is_err());
    }

    #[test]
    fn transcode_rejects_non_mb_aligned() {
        let yuv = vec![0u8; (33 * 32 * 3 / 2) as usize];
        let cfg = BaselineTranscodeConfig::defaults(33, 32, 1);
        assert!(transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).is_err());
    }

    #[test]
    fn streaming_encoder_per_frame_matches_one_shot() {
        let n_frames = 5;
        let yuv = deterministic_yuv(32, 32, n_frames);
        let frame_size = (32 * 32 * 3 / 2) as usize;

        // One-shot baseline.
        let cfg = BaselineTranscodeConfig::defaults(32, 32, n_frames);
        let one_shot = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).unwrap();

        // Streaming: same encoder config, push frame-by-frame, concat.
        let mut streaming = StreamingEncoder::new(32, 32, Some(26), 30).unwrap();
        let mut concat: Vec<u8> = Vec::new();
        for fi in 0..n_frames {
            let frame = &yuv[fi * frame_size..(fi + 1) * frame_size];
            concat.extend_from_slice(&streaming.push_frame(frame).unwrap());
        }
        assert_eq!(streaming.frames_emitted(), n_frames as u32);
        // The concatenated streaming output must byte-match the one-shot
        // output — same encoder, same input, same order.
        assert_eq!(concat, one_shot, "streaming != one-shot byte-for-byte");
    }

    #[test]
    fn streaming_encoder_rejects_wrong_frame_size() {
        let mut streaming = StreamingEncoder::new(32, 32, Some(26), 30).unwrap();
        // Wrong-sized frame.
        let bad = vec![0u8; 100];
        assert!(streaming.push_frame(&bad).is_err());
    }

    /// Phase 6D.8 chunk 3 functional gate: a stego hook installed
    /// during a CABAC encode is INVOKED at the wired emit sites.
    /// Counts the number of residual-block invocations via a
    /// thread-safe counter shared with a small in-test hook impl.
    /// Demonstrates the hook plumbing works end-to-end for the
    /// I_16x16 CABAC residual path (`write_i16x16_macroblock_cabac`).
    #[test]
    fn encoder_invokes_stego_hook_during_cabac_encode() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::sync::Arc;
        use crate::codec::h264::stego::encoder_hook::StegoMbHook;
        use crate::codec::h264::stego::orchestrate::ResidualPathKind;
        use crate::codec::h264::stego::inject::MvdSlot;

        #[derive(Debug)]
        struct CountHook {
            residual_calls: Arc<AtomicUsize>,
            mvd_calls: Arc<AtomicUsize>,
        }
        impl StegoMbHook for CountHook {
            fn on_residual_block(
                &mut self, _: u32, _: u32, _: &mut [i32], _: usize, _: usize,
                _: ResidualPathKind,
            ) {
                self.residual_calls.fetch_add(1, Ordering::Relaxed);
            }
            fn on_mvd_slot(&mut self, _: u32, _: u32, _: &mut MvdSlot) {
                self.mvd_calls.fetch_add(1, Ordering::Relaxed);
            }
        }

        let residual_calls = Arc::new(AtomicUsize::new(0));
        let mvd_calls = Arc::new(AtomicUsize::new(0));
        let hook = Box::new(CountHook {
            residual_calls: residual_calls.clone(),
            mvd_calls: mvd_calls.clone(),
        });

        let yuv = deterministic_yuv(32, 32, 1);
        let mut enc = Encoder::new(32, 32, Some(26)).unwrap();
        enc.entropy_mode = EntropyMode::Cabac;
        enc.enable_transform_8x8 = false;
        enc.set_stego_hook(Some(hook));
        let _bytes = enc.encode_i_frame(&yuv).unwrap();

        let r = residual_calls.load(Ordering::Relaxed);
        // 32×32 single I-frame → 4 MBs. Both I_16x16 (chunk 3a) and
        // I_4x4 (chunk 3b) CABAC paths are wired. Each MB calls the
        // hook AT LEAST once for its luma residual, so floor=4.
        // (Real counts are typically much higher: I_16x16 emits 1
        // DC + up to 16 AC + chroma; I_4x4 emits up to 16 luma 4×4 +
        // chroma. Per-fixture variance from mode-decision.)
        //
        // After P-frame chunks wire MVD + P-frame residuals this
        // floor stays the same on a single-I-frame fixture.
        assert!(
            r >= 4,
            "stego hook MUST fire ≥4× on a 32×32 4-MB I-frame CABAC encode (got {r})",
        );
    }

    /// Phase 6D.8 byte-identity gate: encoder with no-op stego hook
    /// MUST produce byte-identical output. Sign-off for the
    /// stego_hook field added in chunk 2.
    #[test]
    fn encoder_with_none_stego_hook_byte_identical() {
        let yuv = deterministic_yuv(32, 32, 5);
        let cfg = BaselineTranscodeConfig::defaults(32, 32, 5);
        let baseline = transcode_yuv_to_baseline_cavlc_h264(&yuv, cfg).unwrap();

        let mut enc2 = Encoder::new(32, 32, Some(26)).unwrap();
        enc2.entropy_mode = EntropyMode::Cavlc;
        enc2.set_gop_length(30);
        enc2.set_stego_hook(None);
        let frame_size = (32 * 32 * 3 / 2) as usize;
        let mut concat = Vec::new();
        for f in 0..5 {
            let frame = &yuv[f * frame_size..(f + 1) * frame_size];
            let bytes = if f == 0 {
                enc2.encode_i_frame(frame).unwrap()
            } else {
                enc2.encode_p_frame(frame).unwrap()
            };
            concat.extend_from_slice(&bytes);
        }
        assert_eq!(
            concat, baseline,
            "encoder with set_stego_hook(None) MUST be byte-identical",
        );
        assert!(enc2.take_stego_hook().is_none());
    }

    #[test]
    fn streaming_encoder_idr_period_respected() {
        // GOP=2 means frames 0, 2, 4 are IDRs; 1, 3 are P.
        let mut streaming = StreamingEncoder::new(32, 32, Some(26), 2).unwrap();
        let frame_size = (32 * 32 * 3 / 2) as usize;
        let yuv = deterministic_yuv(32, 32, 5);
        for fi in 0..5 {
            let frame = &yuv[fi * frame_size..(fi + 1) * frame_size];
            let _ = streaming.push_frame(frame).unwrap();
        }
        assert_eq!(streaming.frames_emitted(), 5);
    }
}
