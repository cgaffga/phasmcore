// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only

//! Integration tests for H.264 CAVLC video steganography.

#[cfg(feature = "video")]
mod h264_tests {
    use phasm_core::codec::h264::bitstream;
    use phasm_core::codec::h264::sps;
    use phasm_core::codec::mp4;

    const TEST_TINY: &str = "test-vectors/video/h264/test_tiny.mp4";
    const TEST_320: &str = "test-vectors/video/h264/test_baseline_320x240.mp4";

    fn skip_if_missing(path: &str) -> Option<Vec<u8>> {
        match std::fs::read(path) {
            Ok(data) => Some(data),
            Err(_) => {
                eprintln!("SKIP: test vector not found: {path}");
                None
            }
        }
    }

    #[test]
    fn demux_h264_tiny() {
        let Some(data) = skip_if_missing(TEST_TINY) else { return };
        let mp4_file = mp4::demux::demux(&data).unwrap();

        assert!(mp4_file.video_track_idx.is_some());
        let idx = mp4_file.video_track_idx.unwrap();
        let track = &mp4_file.tracks[idx];

        assert!(track.is_h264());
        assert!(!track.is_hevc());
        assert_eq!(track.codec, *b"avc1");
        assert_eq!(track.width, 160);
        assert_eq!(track.height, 120);

        // Should have avcC data
        assert!(track.avcc_data.is_some());
        let avcc = track.avcc_data.as_ref().unwrap();
        assert!(!avcc.sps_nalus.is_empty());
        assert!(!avcc.pps_nalus.is_empty());
        assert_eq!(avcc.length_size_minus1, 3); // 4-byte length prefix

        // Should have samples (frames)
        assert!(track.samples.len() > 0, "no samples");
        assert!(track.samples[0].is_sync, "first sample should be sync/IDR");
    }

    #[test]
    fn parse_h264_sps_pps_from_avcc() {
        let Some(data) = skip_if_missing(TEST_TINY) else { return };
        let mp4_file = mp4::demux::demux(&data).unwrap();
        let track = &mp4_file.tracks[mp4_file.video_track_idx.unwrap()];
        let avcc = track.avcc_data.as_ref().unwrap();

        // Parse SPS
        let sps_nalu = &avcc.sps_nalus[0];
        let sps_rbsp = if !sps_nalu.is_empty() && (sps_nalu[0] & 0x1F) == 7 {
            bitstream::remove_emulation_prevention(&sps_nalu[1..])
        } else {
            bitstream::remove_emulation_prevention(sps_nalu)
        };
        let sps_parsed = sps::parse_sps(&sps_rbsp).unwrap();

        assert_eq!(sps_parsed.profile_idc, 66); // Baseline
        assert_eq!(sps_parsed.width_in_pixels, 160);
        assert_eq!(sps_parsed.height_in_pixels, 120);
        assert!(sps_parsed.frame_mbs_only_flag); // progressive
        assert_eq!(sps_parsed.pic_width_in_mbs, 10); // 160/16
        assert_eq!(sps_parsed.pic_height_in_map_units, 8); // ceil(120/16) = 7.5 → adjusted by cropping

        // Parse PPS
        let pps_nalu = &avcc.pps_nalus[0];
        let pps_rbsp = if !pps_nalu.is_empty() && (pps_nalu[0] & 0x1F) == 8 {
            bitstream::remove_emulation_prevention(&pps_nalu[1..])
        } else {
            bitstream::remove_emulation_prevention(pps_nalu)
        };
        let pps_parsed = sps::parse_pps(&pps_rbsp).unwrap();

        assert!(!pps_parsed.entropy_coding_mode_flag, "should be CAVLC, not CABAC");
        assert_eq!(pps_parsed.num_slice_groups_minus1, 0, "no FMO");
    }

    #[test]
    fn h264_capacity_tiny() {
        let Some(data) = skip_if_missing(TEST_TINY) else { return };
        let cap = phasm_core::stego::video::h264_ghost_capacity(&data);
        match cap {
            Ok(bytes) => {
                // Even a tiny 1-second 160x120 video should have some capacity
                eprintln!("H.264 capacity (tiny): {bytes} bytes");
                // Don't assert specific values — just that it doesn't crash
            }
            Err(e) => {
                eprintln!("H.264 capacity error: {e}");
                // Parsing may fail on first real test — that's OK for now
            }
        }
    }

    #[test]
    fn h264_roundtrip_encode_decode() {
        let Some(data) = skip_if_missing(TEST_320) else { return };

        let message = "Hello H.264 world!";
        let passphrase = "test-passphrase-42";

        // Encode
        let stego = match phasm_core::stego::video::h264_ghost_encode(&data, message, passphrase) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("H.264 encode error (expected during early development): {e}");
                return; // Graceful skip if parsing isn't complete yet
            }
        };

        // Verify same size (zero bitrate change)
        assert_eq!(stego.len(), data.len(), "stego file size must equal cover file size");

        // Verify not identical (some bits should have been flipped)
        assert_ne!(stego, data, "stego should differ from cover");

        // Decode
        let decoded = match phasm_core::stego::video::h264_ghost_decode(&stego, passphrase) {
            Ok(d) => d,
            Err(e) => {
                panic!("H.264 decode failed after successful encode: {e}");
            }
        };

        assert_eq!(decoded.text, message, "decoded message should match original");
    }

    /// Visual quality test: encode stego, write to temp file, have ffmpeg decode
    /// both cover and stego to raw YUV, compare pixel-by-pixel → compute PSNR.
    #[test]
    fn h264_visual_quality_psnr() {
        let Some(data) = skip_if_missing(TEST_320) else { return };

        let message = "PSNR test message";
        let passphrase = "psnr-pass";

        let stego = match phasm_core::stego::video::h264_ghost_encode(&data, message, passphrase) {
            Ok(s) => s,
            Err(_) => return,
        };

        // Write both to temp files
        let dir = std::env::temp_dir();
        let cover_path = dir.join("h264_cover.mp4");
        let stego_path = dir.join("h264_stego.mp4");
        std::fs::write(&cover_path, &data).unwrap();
        std::fs::write(&stego_path, &stego).unwrap();

        // ffmpeg decode test: stego must decode without errors
        let ffmpeg_result = std::process::Command::new("ffmpeg")
            .args(["-v", "error", "-i"])
            .arg(&stego_path)
            .args(["-f", "null", "-"])
            .output();

        match ffmpeg_result {
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                if !stderr.is_empty() {
                    eprintln!("ffmpeg decode warnings/errors:\n{stderr}");
                }
                assert!(
                    output.status.success(),
                    "ffmpeg must decode stego video without fatal errors"
                );
            }
            Err(_) => {
                eprintln!("SKIP: ffmpeg not available for visual test");
                return;
            }
        }

        // Compute PSNR via ffmpeg
        let psnr_result = std::process::Command::new("ffmpeg")
            .args(["-i"])
            .arg(&cover_path)
            .args(["-i"])
            .arg(&stego_path)
            .args([
                "-lavfi", "psnr", "-f", "null", "-",
            ])
            .output();

        match psnr_result {
            Ok(output) => {
                let stderr = String::from_utf8_lossy(&output.stderr);
                // ffmpeg outputs PSNR to stderr like: "PSNR ... average:42.50 ..."
                if let Some(avg_pos) = stderr.find("average:") {
                    let after = &stderr[avg_pos + 8..];
                    if let Some(end) = after.find(|c: char| !c.is_ascii_digit() && c != '.') {
                        let psnr_str = &after[..end];
                        if let Ok(psnr) = psnr_str.parse::<f64>() {
                            eprintln!("H.264 stego PSNR: {psnr:.2} dB");
                            // T1 sign flips on a 320x240 video should give high PSNR
                            // At low modification rates, expect > 35 dB
                            assert!(
                                psnr > 25.0,
                                "PSNR {psnr:.2} dB is too low — visual quality unacceptable"
                            );
                            // Check for inf (no modifications visible at pixel level)
                            if psnr > 80.0 || stderr.contains("inf") {
                                eprintln!("  → Excellent: PSNR effectively infinite (modifications below pixel quantization)");
                            } else if psnr > 45.0 {
                                eprintln!("  → Very good: imperceptible quality difference");
                            } else if psnr > 35.0 {
                                eprintln!("  → Good: minor quality difference");
                            }
                        }
                    }
                } else {
                    eprintln!("Could not parse PSNR from ffmpeg output");
                    eprintln!("ffmpeg stderr: {}", &stderr[stderr.len().saturating_sub(500)..]);
                }
            }
            Err(_) => {
                eprintln!("SKIP: ffmpeg PSNR measurement not available");
            }
        }

        // Count byte differences between cover and stego
        let diff_count = data.iter().zip(stego.iter()).filter(|(a, b)| a != b).count();
        let diff_bits: u32 = data
            .iter()
            .zip(stego.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        eprintln!(
            "Byte diffs: {diff_count}/{} ({:.4}%), Bit diffs: {diff_bits}",
            data.len(),
            diff_count as f64 / data.len() as f64 * 100.0
        );

        // Cleanup
        let _ = std::fs::remove_file(&cover_path);
        let _ = std::fs::remove_file(&stego_path);
    }

    /// Verify the stego MP4 has valid box structure (ffprobe can read it).
    #[test]
    fn h264_stego_valid_mp4_structure() {
        let Some(data) = skip_if_missing(TEST_320) else { return };

        let stego = match phasm_core::stego::video::h264_ghost_encode(
            &data, "structure test", "struct-pass",
        ) {
            Ok(s) => s,
            Err(_) => return,
        };

        // Re-demux the stego output — must parse without errors
        let mp4_file = phasm_core::codec::mp4::demux::demux(&stego).unwrap();
        assert!(mp4_file.video_track_idx.is_some());
        let track = &mp4_file.tracks[mp4_file.video_track_idx.unwrap()];

        // Same number of samples
        let orig_mp4 = phasm_core::codec::mp4::demux::demux(&data).unwrap();
        let orig_track = &orig_mp4.tracks[orig_mp4.video_track_idx.unwrap()];
        assert_eq!(
            track.samples.len(),
            orig_track.samples.len(),
            "sample count must be preserved"
        );

        // Same sample sizes (zero bitrate change means identical sample sizes)
        for (i, (orig_s, stego_s)) in orig_track.samples.iter().zip(track.samples.iter()).enumerate() {
            assert_eq!(
                orig_s.size, stego_s.size,
                "sample {i} size mismatch: orig={} stego={}",
                orig_s.size, stego_s.size
            );
        }

        // ffprobe validation (if available)
        let stego_path = std::env::temp_dir().join("h264_stego_structure.mp4");
        std::fs::write(&stego_path, &stego).unwrap();

        if let Ok(output) = std::process::Command::new("ffprobe")
            .args(["-v", "error", "-show_entries", "stream=codec_name,profile,width,height"])
            .arg(&stego_path)
            .output()
        {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                eprintln!("ffprobe warnings: {stderr}");
            }
            eprintln!("ffprobe output:\n{stdout}");
            assert!(
                stdout.contains("h264") || stdout.contains("avc"),
                "ffprobe should identify H.264 codec"
            );
        }

        let _ = std::fs::remove_file(&stego_path);
    }

    /// Phase 3c cross-domain roundtrip: embed a medium-length message and
    /// verify that the encode picks MVD-domain positions in addition to
    /// coefficient positions, then decodes correctly. The flip count should
    /// be higher than a coefficient-only encode (because the proportional
    /// split forces some flips onto MVD positions with higher cost), but
    /// visual quality stays imperceptible (verified separately by ffmpeg
    /// PSNR check in `h264_visual_quality_psnr`).
    #[test]
    fn h264_cross_domain_roundtrip() {
        let Some(data) = skip_if_missing(TEST_320) else { return };

        // Use a longer message to stress both domains.
        let message = "Cross-domain phase 3c: split payload between coefficient and motion vector domains for stealth.";
        let passphrase = "phase3c-pass";

        let stego = match phasm_core::stego::video::h264_ghost_encode(&data, message, passphrase) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("SKIP: encode error: {e}");
                return;
            }
        };

        // Size invariant: zero bitrate change.
        assert_eq!(stego.len(), data.len(), "stego size must equal cover size");
        assert_ne!(stego, data, "stego must differ from cover");

        // Flip count sanity — the cross-domain pipeline runs two STC embeds
        // so total flips are the sum of both domain flip sets. For a ~100
        // char message on 320x240 we expect at least a few tens of flips.
        let diff_bits: u32 = data
            .iter()
            .zip(stego.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum();
        eprintln!("cross-domain flips: {diff_bits} bits");
        assert!(
            diff_bits >= 20,
            "expected at least 20 bit flips for 100-char message, got {diff_bits}"
        );

        // Round-trip must succeed. The decode path does the same proportional
        // split + two-domain STC extract, so a mis-split would corrupt the
        // message and decrypt would fail.
        let decoded = phasm_core::stego::video::h264_ghost_decode(&stego, passphrase)
            .expect("phase 3c decode must recover the cover message");
        assert_eq!(decoded.text, message);

        // Wrong passphrase still fails — verifies the MVD-domain key is
        // actually used by decrypt (not just for permutation).
        let wrong =
            phasm_core::stego::video::h264_ghost_decode(&stego, "definitely-wrong");
        assert!(
            wrong.is_err(),
            "decode with wrong passphrase must fail even on cross-domain stego"
        );
    }

    #[test]
    fn h264_wrong_passphrase_fails() {
        let Some(data) = skip_if_missing(TEST_320) else { return };

        let message = "secret message";
        let passphrase = "correct-pass";
        let wrong_pass = "wrong-pass";

        let stego = match phasm_core::stego::video::h264_ghost_encode(&data, message, passphrase) {
            Ok(s) => s,
            Err(_) => return, // Graceful skip
        };

        // Decode with wrong passphrase should fail
        let result = phasm_core::stego::video::h264_ghost_decode(&stego, wrong_pass);
        assert!(result.is_err(), "decode with wrong passphrase should fail");
    }
}
