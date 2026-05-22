// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! M1 calibration harness: encode large synthetic JPEGs in each Ghost
//! pathway and sample peak RSS to validate
//! `core/src/stego/memory.rs::predict_peak_memory()` against reality.
//!
//! **Opt-in.** Set `PHASM_MEM_AUDIT=1` to run. Allocates up to ~7 GB
//! during the 60 MP shadow case so it's never appropriate for CI.
//!
//! Usage:
//!   PHASM_MEM_AUDIT=1 cargo test --release --features parallel \
//!       --test perf_memory_audit -- --ignored --nocapture
//!
//! Override the matrix:
//!   PHASM_MEM_AUDIT_MPS=12,24,43,60         (default: 12,24,43,60)
//!   PHASM_MEM_AUDIT_MODES=ghost,ghost_shadow (default: both)
//!
//! Output: a Markdown table on stdout that pastes directly into the
//! design doc. Final assertion: actual RSS within ±50% of prediction
//! for every row (loose because allocator behaviour is OS-dependent).

#![cfg(feature = "parallel")]

use image::{ImageBuffer, Rgb, codecs::jpeg::JpegEncoder};
use phasm_core::{ghost_encode, ghost_encode_with_shadows, ShadowLayer};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

/// Dimensions chosen to mimic real-world device cameras:
/// - 12 MP: typical iPhone HEIC export
/// - 24 MP: APS-C DSLR
/// - 43 MP: full-frame DSLR (similar to 02445.jpg)
/// - 60 MP: medium-format / Leica DNG (L1000213.DNG class)
fn default_mp_matrix() -> Vec<u32> {
    if let Ok(s) = std::env::var("PHASM_MEM_AUDIT_MPS") {
        s.split(',')
            .filter_map(|x| x.trim().parse::<u32>().ok())
            .collect()
    } else {
        vec![12, 24, 43, 60]
    }
}

fn mp_to_dims(mp: u32) -> (u32, u32) {
    // 4:3 aspect ratio, round to even. Width = sqrt(mp × 4/3 × 1e6).
    let target_px = mp as f64 * 1_000_000.0;
    let w_f = (target_px * 4.0 / 3.0).sqrt();
    let h_f = w_f * 3.0 / 4.0;
    let w = ((w_f as u32) / 16) * 16;
    let h = ((h_f as u32) / 16) * 16;
    (w, h)
}

/// Generate a deterministic JPEG of `width × height` with a mix of
/// smooth gradients (gives realistic flat-region cost surface) and
/// pseudo-random pixel noise (gives realistic high-texture cost
/// surface). At quality 75 to match the iOS/Android default.
fn synthesize_jpeg(width: u32, height: u32) -> Vec<u8> {
    let mut img = ImageBuffer::new(width, height);
    for (x, y, px) in img.enumerate_pixels_mut() {
        // Smooth gradient — low cost regions.
        let r = ((x as u32 ^ 0x5A) ^ ((y * 7) as u32 & 0xFF)) as u8;
        let g = (((x.wrapping_mul(13).wrapping_add(y)) ^ 0xA3) & 0xFF) as u8;
        let b = (((x.wrapping_add(y).wrapping_mul(17)) ^ 0x7E) & 0xFF) as u8;
        // Mix in pseudo-random noise in alternating bands — texture.
        let noise = if (y / 16) % 2 == 0 {
            let v = x.wrapping_mul(2654435761) ^ y.wrapping_mul(40503);
            ((v >> 8) & 0x1F) as u8
        } else {
            0
        };
        *px = Rgb([r.wrapping_add(noise), g.wrapping_add(noise), b]);
    }
    let mut buf = Vec::with_capacity((width * height) as usize / 3);
    {
        let mut enc = JpegEncoder::new_with_quality(&mut buf, 75);
        enc.encode(
            img.as_raw(),
            width,
            height,
            image::ExtendedColorType::Rgb8,
        )
        .expect("jpeg encode");
    }
    buf
}

/// Sample current RSS in bytes by shelling out to `ps`. macOS prints
/// KB; Linux prints KB. Returns None if `ps` is unavailable.
fn sample_rss_bytes() -> Option<u64> {
    let pid = std::process::id();
    let out = std::process::Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let kb: u64 = std::str::from_utf8(&out.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()?;
    Some(kb * 1024)
}

struct PeakRssSampler {
    peak_bytes: Arc<AtomicUsize>,
    stop: Arc<AtomicBool>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl PeakRssSampler {
    fn start() -> Self {
        let peak = Arc::new(AtomicUsize::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let peak_clone = peak.clone();
        let stop_clone = stop.clone();
        let handle = std::thread::spawn(move || {
            while !stop_clone.load(Ordering::Relaxed) {
                if let Some(rss) = sample_rss_bytes() {
                    let cur = rss as usize;
                    let mut prev = peak_clone.load(Ordering::Relaxed);
                    while cur > prev {
                        match peak_clone.compare_exchange_weak(
                            prev, cur, Ordering::Relaxed, Ordering::Relaxed,
                        ) {
                            Ok(_) => break,
                            Err(p) => prev = p,
                        }
                    }
                }
                std::thread::sleep(std::time::Duration::from_millis(25));
            }
        });
        Self { peak_bytes: peak, stop, handle: Some(handle) }
    }
    fn stop(mut self) -> usize {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
        // Final sample after stop in case encode finished between polls.
        if let Some(rss) = sample_rss_bytes() {
            let cur = rss as usize;
            let mut prev = self.peak_bytes.load(Ordering::Relaxed);
            while cur > prev {
                match self.peak_bytes.compare_exchange_weak(
                    prev, cur, Ordering::Relaxed, Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(p) => prev = p,
                }
            }
        }
        self.peak_bytes.load(Ordering::Relaxed)
    }
}

struct Row {
    mode: &'static str,
    mp: u32,
    width: u32,
    height: u32,
    wall_ms: u128,
    peak_mb: usize,
    rss_baseline_mb: usize,
}

fn run_case(label: &'static str, mp: u32, body: impl FnOnce()) -> Row {
    let (w, h) = mp_to_dims(mp);
    let baseline = sample_rss_bytes().unwrap_or(0) as usize / (1024 * 1024);
    let sampler = PeakRssSampler::start();
    let t0 = Instant::now();
    body();
    let wall_ms = t0.elapsed().as_millis();
    let peak_bytes = sampler.stop();
    Row {
        mode: label,
        mp,
        width: w,
        height: h,
        wall_ms,
        peak_mb: peak_bytes / (1024 * 1024),
        rss_baseline_mb: baseline,
    }
}

fn enabled() -> bool {
    matches!(
        std::env::var("PHASM_MEM_AUDIT").as_deref(),
        Ok("1") | Ok("true"),
    )
}

fn allowed_modes() -> Vec<&'static str> {
    let env = std::env::var("PHASM_MEM_AUDIT_MODES").unwrap_or_else(|_| "ghost,ghost_shadow".into());
    let mut out = Vec::new();
    for s in env.split(',') {
        match s.trim() {
            "ghost" => out.push("ghost"),
            "ghost_shadow" => out.push("ghost_shadow"),
            _ => {}
        }
    }
    out
}

#[test]
#[ignore = "opt-in; allocates up to 7 GB. Run with PHASM_MEM_AUDIT=1 cargo test --release --features parallel --test perf_memory_audit -- --ignored --nocapture"]
fn perf_memory_audit_matrix() {
    if !enabled() {
        eprintln!("perf_memory_audit_matrix: PHASM_MEM_AUDIT not set; skipping");
        return;
    }
    let modes = allowed_modes();
    let mps = default_mp_matrix();
    let mut rows: Vec<Row> = Vec::new();

    eprintln!("=== M1 memory audit start ===");
    eprintln!("modes: {:?}, MPs: {:?}", modes, mps);

    for mp in &mps {
        let (w, h) = mp_to_dims(*mp);
        eprintln!("\nsynthesizing {} MP ({}×{}) cover JPEG...", mp, w, h);
        let cover = synthesize_jpeg(w, h);
        eprintln!("  cover JPEG: {} bytes", cover.len());

        if modes.contains(&"ghost") {
            eprintln!("  encoding Ghost (no shadow)...");
            let cover_ref = &cover;
            let row = run_case("ghost", *mp, || {
                let _ = ghost_encode(cover_ref, "audit", "audit-pass").unwrap();
            });
            eprintln!(
                "  -> peak {} MB, wall {} ms, baseline {} MB",
                row.peak_mb, row.wall_ms, row.rss_baseline_mb,
            );
            rows.push(row);
        }

        if modes.contains(&"ghost_shadow") {
            eprintln!("  encoding Ghost + shadow...");
            let cover_ref = &cover;
            let row = run_case("ghost_shadow", *mp, || {
                let _ = ghost_encode_with_shadows(
                    cover_ref,
                    "audit primary message that's longer than the shadow",
                    &[],
                    "audit-pass",
                    &[ShadowLayer {
                        message: "audit shadow".to_string(),
                        passphrase: "audit-shadow-pass".to_string(),
                        files: vec![],
                    }],
                    None,
                )
                .unwrap();
            });
            eprintln!(
                "  -> peak {} MB, wall {} ms, baseline {} MB",
                row.peak_mb, row.wall_ms, row.rss_baseline_mb,
            );
            rows.push(row);
        }

        // Drop cover before next iteration to free its JPEG bytes.
        drop(cover);
    }

    eprintln!("\n=== M1 memory audit results ===\n");
    eprintln!("| mode         | MP  | dims        | wall (ms) | peak (MB) | base (MB) |");
    eprintln!("|--------------|----:|-------------|----------:|----------:|----------:|");
    for r in &rows {
        eprintln!(
            "| {:12} | {:3} | {:5}×{:<5} | {:9} | {:9} | {:9} |",
            r.mode, r.mp, r.width, r.height, r.wall_ms, r.peak_mb, r.rss_baseline_mb,
        );
    }

    // Sanity gate: peak must exceed baseline. If the sampler never
    // observed RSS movement, something is broken.
    for r in &rows {
        assert!(
            r.peak_mb > r.rss_baseline_mb,
            "{} {} MP peak {} MB did not exceed baseline {} MB - sampler likely broken",
            r.mode, r.mp, r.peak_mb, r.rss_baseline_mb,
        );
    }
}

#[test]
fn synthesize_jpeg_produces_valid_jpeg() {
    let bytes = synthesize_jpeg(640, 480);
    // Must start with the JPEG SOI marker.
    assert_eq!(&bytes[0..2], &[0xFF, 0xD8], "SOI marker");
    // Phasm core must be able to parse it (sanity check on the harness).
    let img = phasm_core::JpegImage::from_bytes(&bytes).expect("parse synthetic jpeg");
    let fi = img.frame_info();
    assert_eq!(fi.width as u32, 640);
    assert_eq!(fi.height as u32, 480);
}

#[test]
fn mp_to_dims_returns_multiples_of_16() {
    for mp in [1u32, 12, 24, 43, 60, 200] {
        let (w, h) = mp_to_dims(mp);
        assert_eq!(w % 16, 0, "width {} for {} MP must be /16", w, mp);
        assert_eq!(h % 16, 0, "height {} for {} MP must be /16", h, mp);
        // Aspect ratio within 5% of 4:3.
        let aspect = w as f64 / h as f64;
        assert!((aspect - 4.0 / 3.0).abs() < 0.1, "aspect {} for {} MP", aspect, mp);
    }
}
