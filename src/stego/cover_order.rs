// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Keyed per-GOP embedding order for video stego (wire v2), shared by H.264
//! and AV1. Spec: `docs/design/video/stego-cover-order-and-damage-costs.md`
//! § 3.
//!
//! Embedding the primary message over the cover in canonical (raster / emit)
//! order put every GOP's header and first message bits at its key frame's
//! top-left, where the STC had to flip whatever the costs said, and a flip
//! there floods the picture. Each GOP's whole combined cover is instead
//! walked in one keyed order. Every domain stays in the cover; a domain that
//! is not admitted is wet, not removed. The encoder, the decoder and the
//! capacity probe must all apply the same order.
//!
//! Seeds: one order seed and one H-hat seed per GOP and codec, derived from
//! the passphrase's structural master with a codec-neutral `v2` prefix. The
//! labels are the wire version; changing anything here breaks every v2 video
//! (pinned by tests).

/// Video codec a seed is derived for. Keeps the two codecs' seeds independent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoCodec {
    H264,
    Av1,
}

/// Per-GOP seeds: the cover order and the STC's H-hat.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GopSeeds {
    pub order: [u8; 32],
    pub hhat: [u8; 32],
}

/// Seeds for GOP `gop` of `codec`, from the 64-byte structural master
/// (`crypto::derive_structural_key`: first 32 bytes order source, last 32
/// H-hat source).
pub fn gop_seeds(master: &[u8; 64], codec: VideoCodec, gop: u32) -> GopSeeds {
    let codec = match codec {
        VideoCodec::H264 => &b"h264"[..],
        VideoCodec::Av1 => &b"av1"[..],
    };
    GopSeeds {
        order: derive(&master[..32], codec, b"order-v2", gop),
        hhat: derive(&master[32..], codec, b"hhat-v2", gop),
    }
}

/// SHA-256 over the v2 prefix, the master half, codec, role and GOP index.
/// Every variable-length field is length-prefixed so no two inputs collide.
fn derive(master_half: &[u8], codec: &[u8], role: &[u8], gop: u32) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut h = Sha256::new();
    h.update(b"phasm-video-gop-v2");
    for field in [master_half, codec, role] {
        h.update((field.len() as u32).to_le_bytes());
        h.update(field);
    }
    h.update(gop.to_le_bytes());
    h.finalize().into()
}

/// Embedding order for one GOP's combined cover of `n` positions: a keyed
/// permutation of `0..n` (portable Fisher-Yates over ChaCha20 with `u32`
/// draws, as `stego::permute`, so it is identical on every platform).
/// Position `order[i]` of the canonical cover is the `i`-th STC input.
pub fn gop_cover_order(n: usize, seed: &[u8; 32]) -> Vec<u32> {
    use rand::{Rng, SeedableRng};
    assert!(u32::try_from(n).is_ok(), "cover of {n} positions exceeds u32");
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut rng = rand_chacha::ChaCha20Rng::from_seed(*seed);
    for i in (1..n).rev() {
        let j = rng.gen_range(0..=(i as u32)) as usize;
        order.swap(i, j);
    }
    order
}

#[cfg(test)]
mod tests {
    use super::*;

    fn master(fill: u8) -> [u8; 64] {
        let mut m = [0u8; 64];
        for (i, b) in m.iter_mut().enumerate() {
            *b = fill.wrapping_add(i as u8);
        }
        m
    }

    #[test]
    fn order_is_a_permutation() {
        for n in [0usize, 1, 2, 7, 1000] {
            let mut order = gop_cover_order(n, &[9u8; 32]);
            assert_eq!(order.len(), n);
            order.sort_unstable();
            assert!(order.iter().enumerate().all(|(i, &v)| v as usize == i), "n={n}");
        }
    }

    #[test]
    fn order_is_deterministic_and_keyed() {
        let a = gop_cover_order(500, &[1u8; 32]);
        assert_eq!(a, gop_cover_order(500, &[1u8; 32]));
        assert_ne!(a, gop_cover_order(500, &[2u8; 32]));
        // Not the identity: the start of the cover must not stay first.
        assert_ne!(a[..16], (0..16).collect::<Vec<u32>>()[..]);
    }

    #[test]
    fn seeds_differ_by_gop_codec_and_role() {
        let m = master(3);
        let s0 = gop_seeds(&m, VideoCodec::Av1, 0);
        assert_eq!(s0, gop_seeds(&m, VideoCodec::Av1, 0));
        assert_ne!(s0.order, s0.hhat);
        assert_ne!(s0, gop_seeds(&m, VideoCodec::Av1, 1));
        assert_ne!(s0, gop_seeds(&m, VideoCodec::H264, 0));
        assert_ne!(s0, gop_seeds(&master(4), VideoCodec::Av1, 0));
        // Every GOP gets its own H-hat (H.264 reused GOP 0's before v2).
        assert_ne!(s0.hhat, gop_seeds(&m, VideoCodec::Av1, 1).hhat);
    }

    /// Wire-format pins: every v2 video depends on these exact bytes, on
    /// every platform. If one changes, every v2 video stops decoding — only
    /// change them with a deliberate wire bump, and say why here.
    #[test]
    fn v2_seeds_and_order_are_pinned() {
        let mut m = [0u8; 64];
        for (i, b) in m.iter_mut().enumerate() {
            *b = i as u8;
        }
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        let av1 = gop_seeds(&m, VideoCodec::Av1, 0);
        assert_eq!(hex(&av1.order), "279d90fdc51a05e31cc088c95b1124901255c2bef2929d1112e9a179b8561bca");
        assert_eq!(hex(&av1.hhat), "b681e092060e8fdaac0ae57c0a493ff66bc11798303eb637260ab67594f794f0");
        let h264 = gop_seeds(&m, VideoCodec::H264, 7);
        assert_eq!(hex(&h264.order), "0d2cf3a70ece923c53fee64f61e08861c27dae94e16c792a83e5c59c5ef4ec71");
        assert_eq!(hex(&h264.hhat), "65bb50c7257374c881113d3d626f11439393eb4dfe8ff00f50b91dcec6c4ed7e");
        assert_eq!(
            gop_cover_order(1000, &[7u8; 32])[..16],
            [843, 199, 593, 447, 785, 219, 152, 31, 276, 818, 800, 507, 869, 174, 381, 879]
        );
    }
}
