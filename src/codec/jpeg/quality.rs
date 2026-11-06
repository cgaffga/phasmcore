// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! JPEG quality estimate: the IJG quality (1-100) a JPEG's quantization
//! tables correspond to, for the apps' quality chip ("QF~96" for a photo
//! that passes through unchanged).
//!
//! The standard approach: scale the IJG / Annex K tables for every quality
//! 1..=100 exactly as libjpeg does (`jpeg_quality_scaling`,
//! `force_baseline`) and pick the quality whose tables are closest to the
//! file's luma table (and its chroma table, if any). Tables that some
//! quality reproduces entry for entry give an exact result; others (Apple,
//! Samsung and most camera encoders use their own tables) give the closest
//! IJG quality, flagged as approximate. Only the header is read (marker
//! walk to the first scan), never the entropy-coded data.

use super::cover_writer::{scaled_quant_table, STD_CHROMA_QT, STD_LUMA_QT};
use super::tables::parse_dqt;

/// The estimated IJG quality of a JPEG.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QualityEstimate {
    /// IJG quality 1..=100 whose tables are closest.
    pub quality: u8,
    /// The file's tables are exactly the IJG tables at `quality`.
    pub exact: bool,
}

/// Quantization tables and the luma / chroma table ids from the frame
/// header, collected by a marker walk up to the first SOS.
fn header_tables(b: &[u8]) -> Option<([Option<[u16; 64]>; 4], u8, Option<u8>)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut tables: [Option<[u16; 64]>; 4] = [None; 4];
    let mut ids: Option<(u8, Option<u8>)> = None;
    let mut p = 2usize;
    while p + 4 <= b.len() {
        if b[p] != 0xFF {
            p += 1;
            continue;
        }
        let m = b[p + 1];
        if m == 0xFF {
            p += 1;
            continue;
        }
        if m == 0x00 || m == 0x01 || (0xD0..=0xD8).contains(&m) {
            p += 2;
            continue;
        }
        if m == 0xD9 || m == 0xDA {
            break;
        }
        let len = u16::from_be_bytes([b[p + 2], b[p + 3]]) as usize;
        if len < 2 {
            return None;
        }
        let body = b.get(p + 4..p + 2 + len)?;
        match m {
            0xDB => {
                for (id, qt) in parse_dqt(body).ok()? {
                    tables[id as usize] = Some(qt.values);
                }
            }
            // Any SOF (baseline, extended, progressive, lossless, ...): the
            // component table selectors. Luma = the first component.
            0xC0..=0xCF if m != 0xC4 && m != 0xC8 && m != 0xCC => {
                let n = *body.get(5)? as usize;
                let comp = |i: usize| body.get(6 + i * 3 + 2).copied();
                let luma = comp(0)?;
                let chroma = if n >= 3 { comp(1) } else { None };
                if luma > 3 || chroma.is_some_and(|c| c > 3) {
                    return None;
                }
                ids = Some((luma, chroma));
            }
            _ => {}
        }
        p += 2 + len;
    }
    let (luma, chroma) = ids?;
    Some((tables, luma, chroma))
}

/// Squared log-ratio distance of `actual` to `expected` (scale-invariant:
/// a step of 2 vs 1 weighs like 200 vs 100).
fn distance(actual: &[u16; 64], expected: &[u16; 64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(&a, &e)| {
            let r = (a.max(1) as f64 / e.max(1) as f64).ln();
            r * r
        })
        .sum()
}

/// Estimate the IJG quality of `jpeg` from its quantization tables.
///
/// `None` when the bytes are no JPEG with a frame header and its luma
/// table (or the tables are degenerate).
pub fn estimate_jpeg_quality(jpeg: &[u8]) -> Option<QualityEstimate> {
    let (tables, luma_id, chroma_id) = header_tables(jpeg)?;
    let luma = tables[luma_id as usize]?;
    // A chroma table that is the luma table itself (one table for all
    // components) carries no extra information.
    let chroma = chroma_id.and_then(|id| tables[id as usize]).filter(|_| chroma_id != Some(luma_id));
    if luma.iter().any(|&v| v == 0) {
        return None;
    }
    let mut exact: Option<u8> = None;
    let mut best = (f64::INFINITY, 0u8);
    for q in 1..=100u8 {
        let el = scaled_quant_table(&STD_LUMA_QT, q);
        let ec = chroma.map(|_| scaled_quant_table(&STD_CHROMA_QT, q));
        // The highest exactly matching quality (at the lowest qualities
        // every entry clamps to 255, so several qualities can match).
        if el == luma && ec == chroma {
            exact = Some(q);
        }
        let d = distance(&luma, &el) + match (chroma.as_ref(), ec.as_ref()) {
            (Some(c), Some(e)) => distance(c, e),
            _ => 0.0,
        };
        // `<=`: on equal distance the higher quality wins (same reason).
        if d <= best.0 {
            best = (d, q);
        }
    }
    Some(match exact {
        Some(q) => QualityEstimate { quality: q, exact: true },
        None => QualityEstimate { quality: best.1, exact: false },
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stego::ghost::deep_cover::deep_cover_jpeg;

    fn textured(w: u32, h: u32) -> Vec<u8> {
        (0..w * h).flat_map(|i| {
            let (x, y) = (i % w, i / w);
            [(x * 7 + y * 3) as u8, (x * 5 ^ y * 11) as u8, (x + y * 13) as u8]
        })
        .collect()
    }

    #[test]
    fn ijg_tables_give_the_exact_quality() {
        let px = textured(256, 200);
        for q in 10..=100u8 {
            let jpeg = deep_cover_jpeg(&px, 256, 200, q).unwrap();
            assert_eq!(estimate_jpeg_quality(&jpeg), Some(QualityEstimate { quality: q, exact: true }), "QF{q}");
        }
    }

    #[test]
    fn every_ijg_quality_has_distinct_tables_from_10_up() {
        // The exact estimate is unambiguous over the range the apps use.
        for q in 10..100u8 {
            assert_ne!(
                (scaled_quant_table(&STD_LUMA_QT, q), scaled_quant_table(&STD_CHROMA_QT, q)),
                (scaled_quant_table(&STD_LUMA_QT, q + 1), scaled_quant_table(&STD_CHROMA_QT, q + 1)),
                "QF{q} vs QF{}",
                q + 1
            );
        }
    }

    #[test]
    fn libjpeg_test_vectors() {
        // Written by libjpeg (cjpeg / PIL) at the quality in their names.
        // (`photo_640x480_q75_420.jpg` is not: its tables are custom.)
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-vectors/image");
        for (name, q) in [
            ("photo_320x240_q75_420.jpg", 75),
            ("blue_64x64_q90_420.jpg", 90),
            ("fractal_100x75_q85_420.jpg", 85),
            ("green_64x64_q50_422.jpg", 50),
            ("red_64x64_q75_444.jpg", 75),
            ("nonaligned_13x13_q80.jpg", 80),
            ("tiny_8x8_q95.jpg", 95),
            ("gray_64x64_q75.jpg", 75),
        ] {
            let bytes = std::fs::read(dir.join(name)).unwrap();
            let e = estimate_jpeg_quality(&bytes).unwrap();
            assert_eq!(e.quality, q, "{name}: {e:?}");
        }
    }

    #[test]
    fn custom_tables_give_the_closest_quality_flagged_approximate() {
        // An IJG QF90 file whose luma table is nudged (as camera encoders'
        // own tables are): approximate, near 90.
        let px = textured(256, 200);
        let mut jpeg = deep_cover_jpeg(&px, 256, 200, 90).unwrap();
        // First DQT body: FF DB len(2) Pq/Tq then 64 zigzag entries.
        let p = jpeg.windows(2).position(|w| w == [0xFF, 0xDB]).unwrap();
        for i in 0..64 {
            let v = &mut jpeg[p + 5 + i];
            *v = v.saturating_add((i % 3) as u8);
        }
        let e = estimate_jpeg_quality(&jpeg).unwrap();
        assert!(!e.exact && (85..=91).contains(&e.quality), "{e:?}");
    }

    #[test]
    fn not_a_jpeg() {
        assert_eq!(estimate_jpeg_quality(b"\x89PNG\r\n\x1a\n"), None);
        assert_eq!(estimate_jpeg_quality(&[]), None);
        assert_eq!(estimate_jpeg_quality(&[0xFF, 0xD8, 0xFF, 0xD9]), None);
    }

    /// Prints the estimate of every photo in the gitignored
    /// `tests/real_photos/` (phone JPEGs); each must land in 50..=100.
    #[test]
    fn real_photos_land_in_a_plausible_range() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/real_photos");
        let Ok(entries) = std::fs::read_dir(&dir) else { return };
        for entry in entries.flatten() {
            let path = entry.path();
            let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_ascii_lowercase();
            if ext != "jpg" && ext != "jpeg" {
                continue;
            }
            let bytes = std::fs::read(&path).unwrap();
            let e = estimate_jpeg_quality(&bytes);
            eprintln!("{}: {e:?}", path.display());
            let e = e.unwrap();
            assert!((50..=100).contains(&e.quality), "{}: {e:?}", path.display());
        }
    }
}
