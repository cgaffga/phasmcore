// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Per-domain ChaCha20 seed derivation for encode-time CABAC stego.
//
// Architecture decision (see
// docs/design/video/_archive/h264/encoder-algorithms/cabac-bypass-bin-stego.md):
// the four
// stego target domains (CoeffSignBypass, CoeffSuffixLsb,
// MvdSignBypass, MvdSuffixLsb) MUST use uncorrelated per-domain
// seeds for the STC permutation + H-hat matrices. Otherwise an
// attacker breaking one domain's syndrome could attack the others
// using the same matrix.
//
// This module:
//   1. Wraps the existing Argon2-expensive master derivations
//      (`derive_structural_key` for coefficient domains,
//      `derive_h264_mvd_structural_key` for MVD domains) so the
//      caller pays Argon2 once per encode/decode.
//   2. Mixes the masters with a domain-specific label via the
//      existing `derive_per_gop_seed` SHA-256 helper to
//      produce per-(domain, GOP) seeds — cheap, deterministic,
//      cross-platform identical (same `det_math` discipline as the
//      rest of the encoder).
//
// Same structure the retired CAVLC pipeline used; just extended to
// the four CABAC domains.

use zeroize::Zeroizing;

use crate::stego::crypto::{
    derive_h264_mvd_structural_key, derive_per_gop_seed,
    derive_structural_key,
};
use crate::stego::error::StegoError;

use super::EmbedDomain;

/// Master keys for the four CABAC stego domains. Computed once per
/// encode/decode (Argon2 cost), then used to derive per-GOP per-
/// domain seeds via cheap SHA-256 mixing.
///
/// **Two distinct masters**, one for the coefficient domains and
/// one for the MVD domains, mirroring the retired CAVLC pipeline's
/// `coeff_master` + `mvd_master` split. This guarantees that the
/// four per-domain seeds derived below are uncorrelated even at
/// the same GOP index.
pub struct CabacStegoMasterKeys {
    /// 64-byte master for `CoeffSignBypass` + `CoeffSuffixLsb`.
    /// Layout: first 32 = perm_seed source, last 32 = hhat_seed source.
    coeff_master: Zeroizing<[u8; 64]>,
    /// 64-byte master for `MvdSignBypass` + `MvdSuffixLsb`. Same
    /// layout as `coeff_master`.
    mvd_master: Zeroizing<[u8; 64]>,
}

/// Per-GOP STC seed pair for one domain. STC requires two seeds:
/// one for the cover-position permutation, one for the H-hat
/// matrix generation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DomainSeeds {
    pub perm_seed: [u8; 32],
    pub hhat_seed: [u8; 32],
}

impl CabacStegoMasterKeys {
    /// Derive both Argon2 masters from a passphrase. ~Hundreds of
    /// milliseconds; called once per encode or decode.
    pub fn derive(passphrase: &str) -> Result<Self, StegoError> {
        let coeff_master = derive_structural_key(passphrase)?;
        let mvd_master = derive_h264_mvd_structural_key(passphrase)?;
        Ok(Self { coeff_master, mvd_master })
    }

    /// Derive `(perm_seed, hhat_seed)` for the given domain at the
    /// given GOP index. Cheap (two SHA-256 calls).
    pub fn per_gop_seeds(&self, domain: EmbedDomain, gop_idx: u32) -> DomainSeeds {
        let (master_perm_src, master_hhat_src, perm_label, hhat_label) = match domain {
            EmbedDomain::CoeffSignBypass => (
                self.coeff_master_perm(),
                self.coeff_master_hhat(),
                b"cabac-coeff-sign-perm" as &[u8],
                b"cabac-coeff-sign-hhat" as &[u8],
            ),
            EmbedDomain::CoeffSuffixLsb => (
                self.coeff_master_perm(),
                self.coeff_master_hhat(),
                b"cabac-coeff-suffix-perm" as &[u8],
                b"cabac-coeff-suffix-hhat" as &[u8],
            ),
            EmbedDomain::MvdSignBypass => (
                self.mvd_master_perm(),
                self.mvd_master_hhat(),
                b"cabac-mvd-sign-perm" as &[u8],
                b"cabac-mvd-sign-hhat" as &[u8],
            ),
            EmbedDomain::MvdSuffixLsb => (
                self.mvd_master_perm(),
                self.mvd_master_hhat(),
                b"cabac-mvd-suffix-perm" as &[u8],
                b"cabac-mvd-suffix-hhat" as &[u8],
            ),
        };
        DomainSeeds {
            perm_seed: derive_per_gop_seed(master_perm_src, gop_idx, perm_label),
            hhat_seed: derive_per_gop_seed(master_hhat_src, gop_idx, hhat_label),
        }
    }

    fn coeff_master_perm(&self) -> &[u8; 32] {
        Self::first_half(&self.coeff_master)
    }
    fn coeff_master_hhat(&self) -> &[u8; 32] {
        Self::second_half(&self.coeff_master)
    }
    fn mvd_master_perm(&self) -> &[u8; 32] {
        Self::first_half(&self.mvd_master)
    }
    fn mvd_master_hhat(&self) -> &[u8; 32] {
        Self::second_half(&self.mvd_master)
    }

    fn first_half(buf: &[u8; 64]) -> &[u8; 32] {
        // SAFETY: u8 slice → &[u8; 32] cast for the first 32 bytes.
        // Length-checked at compile time (input is &[u8; 64]).
        let first: &[u8] = &buf[..32];
        first.try_into().expect("32-byte slice")
    }
    fn second_half(buf: &[u8; 64]) -> &[u8; 32] {
        let second: &[u8] = &buf[32..];
        second.try_into().expect("32-byte slice")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derive_succeeds_with_any_nonempty_passphrase() {
        let keys = CabacStegoMasterKeys::derive("test-passphrase").unwrap();
        let _ = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0);
    }

    #[test]
    fn same_passphrase_produces_same_seeds() {
        let a = CabacStegoMasterKeys::derive("same").unwrap();
        let b = CabacStegoMasterKeys::derive("same").unwrap();
        for &domain in &[
            EmbedDomain::CoeffSignBypass,
            EmbedDomain::CoeffSuffixLsb,
            EmbedDomain::MvdSignBypass,
            EmbedDomain::MvdSuffixLsb,
        ] {
            for gop in [0u32, 1, 5, 100] {
                assert_eq!(
                    a.per_gop_seeds(domain, gop),
                    b.per_gop_seeds(domain, gop),
                    "determinism: {domain:?} gop={gop}",
                );
            }
        }
    }

    #[test]
    fn different_passphrase_produces_different_seeds() {
        let a = CabacStegoMasterKeys::derive("alpha").unwrap();
        let b = CabacStegoMasterKeys::derive("beta").unwrap();
        let sa = a.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0);
        let sb = b.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0);
        assert_ne!(sa, sb);
    }

    #[test]
    fn four_domains_produce_distinct_seeds_at_same_gop() {
        let keys = CabacStegoMasterKeys::derive("isolation").unwrap();
        let cs = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0);
        let cl = keys.per_gop_seeds(EmbedDomain::CoeffSuffixLsb, 0);
        let ms = keys.per_gop_seeds(EmbedDomain::MvdSignBypass, 0);
        let ml = keys.per_gop_seeds(EmbedDomain::MvdSuffixLsb, 0);
        // All four must be pairwise distinct — this is the cross-
        // domain isolation property.
        let all = [cs, cl, ms, ml];
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "domain seeds {i} and {j} collided");
            }
        }
    }

    #[test]
    fn perm_and_hhat_seeds_are_distinct() {
        let keys = CabacStegoMasterKeys::derive("perm-vs-hhat").unwrap();
        let s = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0);
        assert_ne!(
            s.perm_seed, s.hhat_seed,
            "perm and hhat seeds must be uncorrelated within one domain",
        );
    }

    #[test]
    fn different_gops_produce_different_seeds() {
        let keys = CabacStegoMasterKeys::derive("gop-isolation").unwrap();
        let s0 = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0);
        let s1 = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 1);
        assert_ne!(s0, s1);
    }

    #[test]
    fn coeff_and_mvd_masters_are_uncorrelated() {
        // The coeff_master and mvd_master come from different Argon2
        // salts. Their first-32-byte slices must differ.
        let keys = CabacStegoMasterKeys::derive("master-split").unwrap();
        // Indirect check: at gop=0 the coeff-domain seeds must
        // differ from the MVD-domain seeds even with the same
        // label structure.
        let coeff = keys.per_gop_seeds(EmbedDomain::CoeffSignBypass, 0);
        let mvd = keys.per_gop_seeds(EmbedDomain::MvdSignBypass, 0);
        assert_ne!(coeff.perm_seed, mvd.perm_seed);
        assert_ne!(coeff.hhat_seed, mvd.hhat_seed);
    }
}
