// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Primary-STC keys for H.264 video stego, wire v2.
//!
//! Spec: `docs/design/video/stego-cover-order-and-damage-costs.md` § 3, § 8.
//!
//! The primary runs ONE STC per GOP over the whole combined 4-domain cover
//! (CS | CSL | MVDs | MVDsl), walked in one keyed order. Each GOP gets its
//! own order seed and its own H-hat seed, both from the passphrase's
//! structural master (`crypto::derive_structural_key`, one Argon2 run per
//! encode / decode) through the shared [`crate::stego::cover_order::gop_seeds`]
//! with `VideoCodec::H264`. The v2 seed labels are the wire version.
//!
//! v1 (until 2026-10) derived per-domain seeds from two Argon2 masters
//! (`CabacStegoMasterKeys`) and used GOP 0's CoeffSign H-hat seed for every
//! GOP and no order. There is no v1 decoder (spec § 9).

use zeroize::Zeroizing;

use crate::stego::cover_order::{gop_seeds, GopSeeds, VideoCodec};
use crate::stego::crypto::derive_structural_key;
use crate::stego::error::StegoError;

/// The primary's per-GOP STC keys for one passphrase.
#[derive(Clone)]
pub struct H264StcKeys {
    master: Zeroizing<[u8; 64]>,
}

impl std::fmt::Debug for H264StcKeys {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("H264StcKeys(..)")
    }
}

impl H264StcKeys {
    /// Derive from a passphrase (one Argon2 run, cached process-wide).
    pub fn derive(passphrase: &str) -> Result<Self, StegoError> {
        Ok(Self { master: derive_structural_key(passphrase)? })
    }

    /// Keys from an explicit 64-byte master (tests and pinned fixtures).
    pub fn from_master(master: [u8; 64]) -> Self {
        Self { master: Zeroizing::new(master) }
    }

    /// Order + H-hat seeds of GOP `gop` (the GOP's index in the stream, i.e.
    /// its slab index for the decoder).
    pub fn gop(&self, gop: u32) -> GopSeeds {
        gop_seeds(&self.master, VideoCodec::H264, gop)
    }

    /// EXPERIMENTAL (Path B, `v2_knobs::seed_variants`): seed variant `a` of
    /// GOP `gop`. Variant 0 is [`Self::gop`]; variant `a > 0` uses the same
    /// SHA-256 construction as `cover_order::gop_seeds` with the roles
    /// `order-v2/a{a}` and `hhat-v2/a{a}`, so every variant is an
    /// independent keyed (order, H-hat) pair.
    pub fn gop_variant(&self, gop: u32, a: u8) -> GopSeeds {
        if a == 0 {
            return self.gop(gop);
        }
        fn derive(master_half: &[u8], role: &[u8], gop: u32) -> [u8; 32] {
            use sha2::{Digest, Sha256};
            let mut h = Sha256::new();
            h.update(b"phasm-video-gop-v2");
            for field in [master_half, &b"h264"[..], role] {
                h.update((field.len() as u32).to_le_bytes());
                h.update(field);
            }
            h.update(gop.to_le_bytes());
            h.finalize().into()
        }
        let order_role = format!("order-v2/a{a}");
        let hhat_role = format!("hhat-v2/a{a}");
        GopSeeds {
            order: derive(&self.master[..32], order_role.as_bytes(), gop),
            hhat: derive(&self.master[32..], hhat_role.as_bytes(), gop),
        }
    }

    /// The seed variants the encoder may choose from and the decoder tries,
    /// in order: `gop_variant(gop, 0..v2_knobs::seed_variants())`. One entry
    /// (= [`Self::gop`]) unless the experiment knob is set.
    pub fn gop_variants(&self, gop: u32) -> Vec<GopSeeds> {
        (0..super::v2_knobs::seed_variants()).map(|a| self.gop_variant(gop, a)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seed_variants_are_independent_and_variant_0_is_the_v2_seed() {
        let k = H264StcKeys::derive("variants").unwrap();
        for g in [0u32, 3] {
            assert_eq!(k.gop_variant(g, 0), k.gop(g));
            let v: Vec<GopSeeds> = (0..4).map(|a| k.gop_variant(g, a)).collect();
            for i in 0..4 {
                for j in i + 1..4 {
                    assert_ne!(v[i].order, v[j].order);
                    assert_ne!(v[i].hhat, v[j].hhat);
                }
            }
            assert_eq!(k.gop_variant(g, 2), H264StcKeys::derive("variants").unwrap().gop_variant(g, 2));
        }
        // Default knob: one variant, the v2 seed.
        assert_eq!(k.gop_variants(5), vec![k.gop(5)]);
    }

    #[test]
    fn deterministic_and_keyed() {
        let a = H264StcKeys::derive("same").unwrap();
        let b = H264StcKeys::derive("same").unwrap();
        let c = H264StcKeys::derive("other").unwrap();
        for g in [0u32, 1, 7, 100] {
            assert_eq!(a.gop(g), b.gop(g));
            assert_ne!(a.gop(g), c.gop(g));
        }
    }

    /// Every GOP has its own order and H-hat (v1 reused GOP 0's H-hat).
    #[test]
    fn per_gop_seeds_differ() {
        let k = H264StcKeys::derive("gops").unwrap();
        let (s0, s1) = (k.gop(0), k.gop(1));
        assert_ne!(s0.order, s1.order);
        assert_ne!(s0.hhat, s1.hhat);
        assert_ne!(s0.order, s0.hhat);
    }

    /// The H.264 seeds are the shared primitive's, with the H.264 codec tag.
    #[test]
    fn matches_the_shared_primitive() {
        let mut m = [0u8; 64];
        for (i, b) in m.iter_mut().enumerate() {
            *b = i as u8;
        }
        let k = H264StcKeys::from_master(m);
        assert_eq!(k.gop(7), gop_seeds(&m, VideoCodec::H264, 7));
    }
}
