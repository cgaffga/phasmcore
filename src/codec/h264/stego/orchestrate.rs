// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore
//
// Shared per-domain stego types for the H.264 CABAC bypass-bin scheme.
//
// This module once hosted the pure-Rust encoder's three-pass
// orchestration driver: a `GopDecisionCache` of
// `MbDecision`s, a `split_message_per_domain` splitter, and a
// Pass-1/2/3 cover→plan→apply loop. That driver retired with the
// pure-Rust H.264 encoder — the production OH264 path derives per-MB
// decisions through the FFI capture/replay cache (`pass2_cache.rs`)
// and injects at the CABAC emit site (`wire_only`), not by mutating a
// Rust-side decision cache.
//
// What survives is the decode-shared type surface the walker + cost
// model still use:
//   - `ResidualPathKind` — residual-block → `SyntaxPath` tag (walker)
//   - `DomainCosts`      — per-domain cost vector (cost model)
//   - `DomainPlan`       — per-domain STC plan (cost model)

use super::{BinKind, SyntaxPath};

/// Tag describing which [`SyntaxPath`] variant a residual block uses.
/// Lets the walker carry a residual block's path identity as plain
/// data (clonable + serializable) without storing closures.
#[derive(Copy, Clone, Debug)]
pub enum ResidualPathKind {
    Luma4x4 { block_idx: u8 },
    Luma8x8 { block_idx: u8 },
    ChromaAc { plane: u8, block_idx: u8 },
    ChromaDc { plane: u8 },
    LumaDcIntra16x16,
}

impl ResidualPathKind {
    /// Build a SyntaxPath for the given coefficient index + bin kind.
    pub fn path(self, coeff_idx: u8, kind: BinKind) -> SyntaxPath {
        match self {
            ResidualPathKind::Luma4x4 { block_idx } => SyntaxPath::Luma4x4 {
                block_idx, coeff_idx, kind,
            },
            ResidualPathKind::Luma8x8 { block_idx } => SyntaxPath::Luma8x8 {
                block_idx, coeff_idx, kind,
            },
            ResidualPathKind::ChromaAc { plane, block_idx } => SyntaxPath::ChromaAc {
                plane, block_idx, coeff_idx, kind,
            },
            ResidualPathKind::ChromaDc { plane } => SyntaxPath::ChromaDc {
                plane, coeff_idx, kind,
            },
            ResidualPathKind::LumaDcIntra16x16 => SyntaxPath::LumaDcIntra16x16 {
                coeff_idx, kind,
            },
        }
    }

}

/// Cost vectors per domain, indices aligned with the `DomainCover`
/// bits the cover walk produced.
#[derive(Default, Clone, Debug)]
pub struct DomainCosts {
    pub coeff_sign_bypass: Vec<f32>,
    pub coeff_suffix_lsb: Vec<f32>,
    pub mvd_sign_bypass: Vec<f32>,
    pub mvd_suffix_lsb: Vec<f32>,
}

/// Per-domain STC bit plan. Indices align with each domain's
/// `DomainBits.positions` from the cover walk; the emit site writes
/// these target bits at the corresponding bypass positions.
#[derive(Default, Clone, Debug)]
pub struct DomainPlan {
    pub coeff_sign_bypass: Vec<u8>,
    pub coeff_suffix_lsb: Vec<u8>,
    pub mvd_sign_bypass: Vec<u8>,
    pub mvd_suffix_lsb: Vec<u8>,
    /// Total modifications across all four domains.
    pub total_modifications: usize,
    /// Total cost across all four domains.
    pub total_cost: f64,
}
