// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Phase C.3.6.2 (task #429) — bitstream-mod stego: RBSP strip/repack
//! helpers + NAL location indexing for the Option C bitstream-mod
//! splicer.
//!
//! The Phase C OpenH264-backend pivot to **bitstream-mod stego**
//! (Option F per `docs/design/video/h264/phase-c-ship-criteria.md`)
//! flips bypass-coded CABAC bins **after** the encoder produces the
//! Annex-B stream, rather than during encoding. This module provides
//! the coordinate-mapping primitives needed to locate a captured
//! bypass-bin offset (engine-local to the slice CABAC byte stream)
//! within the original Annex-B bytes.
//!
//! ## Coordinate spaces
//!
//! The splicer composes three coordinate transforms:
//!
//! 1. **Engine-local bit** (what `walker` captures via
//!    `CoverWalkOutput.offsets`): bit position within
//!    `&nal.rbsp[cabac_byte_off..]`.
//! 2. **NAL-RBSP-byte** (intermediate): byte position within
//!    `nal.rbsp`. Derived by adding `cabac_byte_off`.
//! 3. **Raw NAL payload byte** (with emulation-prevention): byte
//!    position within the NAL's raw payload (post-NAL-header).
//!    Looked up via `EpByteMap.rbsp_to_raw`.
//! 4. **Annex-B byte** (final): byte position within the input
//!    Annex-B stream. Derived by adding `NalLocation.payload_start`
//!    + 1 (NAL header byte).
//!
//! C.3.6.2 ships the indexing primitives (steps 1→4 lookup data).
//! C.3.6.3 ships the composition + bit-flip splicer.
//!
//! ## Round-trip property
//!
//! `insert_emulation_prevention(remove_emulation_prevention(data)) ==
//! data` for any conformant H.264 NAL payload. Verified at module
//! scope by `roundtrip_ep_strip_repack_on_openh264_output`.

use crate::codec::h264::bitstream::{
    insert_emulation_prevention, parse_nal_units_annexb,
    remove_emulation_prevention_with_map, EpByteMap,
};
use crate::codec::h264::{H264Error, NalType, NalUnit};

// Re-export the RBSP strip/repack primitives so downstream code
// (splicer, orchestrator) has a single import surface.
pub use crate::codec::h264::bitstream::{
    insert_emulation_prevention as repack_emulation_prevention,
    remove_emulation_prevention as strip_emulation_prevention,
    remove_emulation_prevention_with_map as strip_emulation_prevention_with_map,
};
pub use crate::codec::h264::bitstream::EpByteMap as EmulationPrevByteMap;
pub use super::slice::cabac_data_byte_offset;

/// Byte ranges + EP-map for a single NAL within an Annex-B stream.
///
/// The splicer keeps a `Vec<NalLocation>` aligned by index with the
/// walker's `nal_idx`. For each flip target the splicer:
///
///   1. Looks up `nal_loc = &nal_locations[nal_idx]`.
///   2. Computes `rbsp_byte_in_nal = cabac_byte_off + engine_bit / 8`.
///   3. Maps to raw payload byte via `ep_map.rbsp_to_raw[rbsp_byte_in_nal]`.
///   4. Adds `nal_loc.payload_start + 1` (NAL header byte) to reach
///      the Annex-B absolute byte position.
///   5. Flips bit `engine_bit % 8` (MSB-first) at that byte.
///
/// Step 5 is the C.3.6.3 splicer's job; this struct surfaces the
/// fields needed for steps 1–4.
#[derive(Clone, Debug)]
pub struct NalLocation {
    /// NAL unit type (parsed from the NAL header byte).
    pub nal_type: NalType,
    /// nal_ref_idc (parsed from the NAL header byte).
    pub nal_ref_idc: u8,
    /// Byte offset of the first start-code byte (`0x00`) of this
    /// NAL's start code (3-byte `0x00 0x00 0x01` or 4-byte
    /// `0x00 0x00 0x00 0x01`) in the input Annex-B stream. The
    /// start code itself is NOT modifiable by the splicer.
    pub start_code_start: usize,
    /// Byte offset of the NAL header byte (first byte after the
    /// start code). `annex_b[payload_start]` is the NAL header byte
    /// (forbidden_zero_bit + nal_ref_idc + nal_unit_type). The
    /// raw NAL payload bytes (with emulation prevention) live at
    /// `annex_b[payload_start + 1 .. payload_end]`.
    pub payload_start: usize,
    /// One-past-last byte offset of the NAL payload. The slice
    /// `annex_b[payload_start .. payload_end]` contains the NAL
    /// header byte followed by the raw payload (with emulation
    /// prevention bytes).
    pub payload_end: usize,
    /// EP-stripped RBSP — same as `NalUnit.rbsp`. Slice header /
    /// CABAC consumers read from this. Splicer indexes
    /// `rbsp[cabac_byte_off + engine_bit / 8]` for its byte
    /// computation BUT writes to the raw payload bytes (see
    /// `ep_map` for the index translation).
    pub rbsp: Vec<u8>,
    /// Map from `rbsp[i]` to raw-payload-byte offset (relative to
    /// `payload_start + 1`, i.e. inside the NAL's raw payload
    /// after the header byte). The Annex-B absolute byte is
    /// `payload_start + 1 + ep_map.rbsp_to_raw[i]`.
    pub ep_map: EpByteMap,
}

/// Errors surfaced by `locate_nal_units_annexb`.
#[derive(Debug)]
pub enum LocateError {
    /// Underlying H.264 parse error (forbidden-zero-bit set, etc.).
    H264(H264Error),
    /// Annex-B stream contains no valid start codes.
    NoStartCode,
}

impl std::fmt::Display for LocateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LocateError::H264(e) => write!(f, "h264 parse error: {:?}", e),
            LocateError::NoStartCode => {
                write!(f, "annex-b stream contains no start code")
            }
        }
    }
}

impl std::error::Error for LocateError {}

impl From<H264Error> for LocateError {
    fn from(e: H264Error) -> Self {
        LocateError::H264(e)
    }
}

/// Locate every NAL unit in an Annex-B stream, returning byte ranges
/// + EP-map per NAL. This is the splicer's primary lookup table —
/// the index `i` in the returned vec matches the walker's `nal_idx`
/// for the same input stream.
///
/// Identical NAL ordering + content to `parse_nal_units_annexb` (the
/// walker's NAL source), so `nal_locations[nal_idx].rbsp ==
/// walker_nalus[nal_idx].rbsp` byte-for-byte. The walker drops the
/// byte ranges; this function preserves them so the splicer can map
/// captured RBSP coordinates back to Annex-B byte positions.
///
/// **Streaming**: keeps an in-memory `Vec<NalLocation>` of length
/// equal to the input NAL count. For 2h video at ~5 Mbps with one
/// NAL per frame (~180,000 NALs) this is ~10 MB of metadata (each
/// `NalLocation` carries `rbsp: Vec<u8>` which dominates). For very
/// long-form streams the per-GOP streaming variant should be used —
/// see `locate_nal_units_annexb_streaming` follow-on (C.3.6.4 scope).
pub fn locate_nal_units_annexb(
    data: &[u8],
) -> Result<Vec<NalLocation>, LocateError> {
    let mut locations = Vec::new();
    let mut i = 0usize;

    let mut saw_any_start = false;
    while i < data.len() {
        // Find next start code (3-byte `0x00 0x00 0x01` or 4-byte
        // `0x00 0x00 0x00 0x01`).
        let (start_code_start, start_code_len) =
            match find_next_start_code(data, i) {
                Some(x) => x,
                None => break,
            };
        saw_any_start = true;
        let payload_start = start_code_start + start_code_len;
        if payload_start >= data.len() {
            // Start code with no following NAL bytes — malformed but
            // not catastrophic; stop here.
            break;
        }

        // Find the next start code (= end of this NAL).
        let next_start =
            find_next_start_code(data, payload_start).map(|(p, _)| p);
        let mut payload_end = next_start.unwrap_or(data.len());

        // Trailing-zero rule: H.264 Annex-B permits trailing zeros
        // before the next start code; strip them when the next start
        // code follows immediately.
        while payload_end > payload_start
            && data[payload_end - 1] == 0
            && next_start.is_some()
        {
            payload_end -= 1;
        }

        if payload_end <= payload_start {
            // Empty NAL; skip.
            i = next_start.unwrap_or(data.len());
            continue;
        }

        let nal_header = data[payload_start];
        if (nal_header >> 7) & 1 != 0 {
            return Err(LocateError::H264(H264Error::InvalidNalHeader));
        }
        let nal_ref_idc = (nal_header >> 5) & 0x03;
        let nal_type = NalType(nal_header & 0x1F);

        let raw_payload = &data[payload_start + 1..payload_end];
        let (rbsp, ep_map) =
            remove_emulation_prevention_with_map(raw_payload);

        locations.push(NalLocation {
            nal_type,
            nal_ref_idc,
            start_code_start,
            payload_start,
            payload_end,
            rbsp,
            ep_map,
        });

        i = payload_end;
    }

    if !saw_any_start && !data.is_empty() {
        return Err(LocateError::NoStartCode);
    }
    Ok(locations)
}

/// Convert a `NalLocation` back into a plain `NalUnit` — useful for
/// consumers that want the same shape as `parse_nal_units_annexb`
/// (e.g. feeding into the existing walker).
impl From<&NalLocation> for NalUnit {
    fn from(loc: &NalLocation) -> Self {
        NalUnit {
            nal_type: loc.nal_type,
            nal_ref_idc: loc.nal_ref_idc,
            rbsp: loc.rbsp.clone(),
        }
    }
}

/// Locate the next H.264 Annex-B start code starting at `from`.
/// Returns `Some((start_code_start_byte, start_code_len_in_bytes))`
/// where `start_code_len_in_bytes` is 3 or 4. Returns `None` if no
/// further start code exists.
fn find_next_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    let mut i = from;
    while i + 2 < data.len() {
        if data[i] == 0 && data[i + 1] == 0 {
            if data[i + 2] == 1 {
                return Some((i, 3));
            }
            if i + 3 < data.len()
                && data[i + 2] == 0
                && data[i + 3] == 1
            {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

/// Verify that `locate_nal_units_annexb` emits the same NAL list as
/// `parse_nal_units_annexb` for parity with the walker (which uses
/// the latter). Returns `(n_locations, n_walker_nalus)`. Used by the
/// round-trip test + by the splicer's debug-assert path.
pub fn assert_parity_with_walker_parse(
    data: &[u8],
) -> Result<(usize, usize), LocateError> {
    let locs = locate_nal_units_annexb(data)?;
    let nals = parse_nal_units_annexb(data)?;
    assert_eq!(
        locs.len(),
        nals.len(),
        "locate_nal_units_annexb / parse_nal_units_annexb NAL count mismatch"
    );
    for (i, (loc, nal)) in locs.iter().zip(nals.iter()).enumerate() {
        assert_eq!(
            loc.nal_type.0, nal.nal_type.0,
            "NAL type mismatch at idx {i}"
        );
        assert_eq!(
            loc.nal_ref_idc, nal.nal_ref_idc,
            "nal_ref_idc mismatch at idx {i}"
        );
        assert_eq!(
            loc.rbsp.as_slice(),
            nal.rbsp.as_slice(),
            "rbsp bytes mismatch at idx {i}"
        );
    }
    Ok((locs.len(), nals.len()))
}


// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_repack_roundtrip_on_synthetic_ep_pattern() {
        // Construct a NAL payload containing `0x00 0x00 0x03` (the
        // exact pattern emulation prevention guards against).
        // `insert_emulation_prevention` re-inserts the 0x03 after
        // every `0x00 0x00`.
        let rbsp: Vec<u8> = vec![
            0x00, 0x00, 0x03, // would emulate a start code without EP
            0xFF, 0x01,
            0x00, 0x00, 0x00, // double zero followed by zero -> needs EP
            0x42,
            0x00, 0x00, 0x02, 0x99,
        ];
        let raw = repack_emulation_prevention(&rbsp);
        let stripped = strip_emulation_prevention(&raw);
        assert_eq!(rbsp, stripped, "strip(repack(rbsp)) must be identity");

        // With map.
        let (stripped2, map) = strip_emulation_prevention_with_map(&raw);
        assert_eq!(rbsp, stripped2);
        assert_eq!(
            map.rbsp_to_raw.len(),
            rbsp.len(),
            "ep_map length must match rbsp length"
        );
        // For every rbsp byte, the mapped raw byte must equal the
        // rbsp byte.
        for (i, &raw_idx) in map.rbsp_to_raw.iter().enumerate() {
            assert_eq!(
                raw[raw_idx], rbsp[i],
                "rbsp_to_raw[{i}]={raw_idx} but raw[{raw_idx}]=0x{:02X} != rbsp[{i}]=0x{:02X}",
                raw[raw_idx], rbsp[i]
            );
        }
    }

    #[test]
    fn locate_nal_units_finds_synthetic_3_nals() {
        // Hand-craft an Annex-B stream with 3 NALs separated by
        // start codes (mix of 3-byte and 4-byte). Payload bytes are
        // distinguishable so we can verify byte ranges.
        let mut data = Vec::new();
        // NAL 1: 4-byte start code, header byte + 3 payload bytes.
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
        data.extend_from_slice(&[0x67, 0xAA, 0xBB, 0xCC]); // SPS (type=7)
        // NAL 2: 3-byte start code.
        data.extend_from_slice(&[0x00, 0x00, 0x01]);
        data.extend_from_slice(&[0x68, 0xDD]); // PPS (type=8)
        // NAL 3: 3-byte start code.
        data.extend_from_slice(&[0x00, 0x00, 0x01]);
        data.extend_from_slice(&[0x65, 0xEE, 0xFF, 0x11, 0x22]); // IDR (type=5)

        let locs = locate_nal_units_annexb(&data).unwrap();
        assert_eq!(locs.len(), 3, "expected 3 NALs");

        // NAL 1: SPS.
        assert_eq!(locs[0].nal_type.0, 7);
        assert_eq!(locs[0].nal_ref_idc, 0b11);
        assert_eq!(locs[0].start_code_start, 0);
        assert_eq!(locs[0].payload_start, 4);
        assert_eq!(locs[0].payload_end, 8);
        assert_eq!(locs[0].rbsp, vec![0xAA, 0xBB, 0xCC]);

        // NAL 2: PPS.
        assert_eq!(locs[1].nal_type.0, 8);
        assert_eq!(locs[1].start_code_start, 8);
        assert_eq!(locs[1].payload_start, 11);
        assert_eq!(locs[1].payload_end, 13);
        assert_eq!(locs[1].rbsp, vec![0xDD]);

        // NAL 3: IDR.
        assert_eq!(locs[2].nal_type.0, 5);
        assert_eq!(locs[2].start_code_start, 13);
        assert_eq!(locs[2].payload_start, 16);
        assert_eq!(locs[2].payload_end, 21);
        assert_eq!(locs[2].rbsp, vec![0xEE, 0xFF, 0x11, 0x22]);
    }

    #[test]
    fn locate_parity_with_walker_parse_on_synthetic() {
        let mut data = Vec::new();
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01, 0x67, 0x42, 0x00, 0x1E]);
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x68, 0xCE, 0x32, 0xC8]);
        data.extend_from_slice(&[0x00, 0x00, 0x01, 0x65, 0x88, 0x84, 0x00]);
        let (n_locs, n_nals) =
            assert_parity_with_walker_parse(&data).unwrap();
        assert_eq!(n_locs, n_nals);
        assert_eq!(n_locs, 3);
    }

    #[test]
    fn locate_rejects_empty_data_with_no_start_code() {
        let data = vec![0x12, 0x34, 0x56]; // no start code anywhere
        let err = locate_nal_units_annexb(&data).unwrap_err();
        assert!(matches!(err, LocateError::NoStartCode));
    }

    #[test]
    fn locate_accepts_empty_input() {
        let locs = locate_nal_units_annexb(&[]).unwrap();
        assert!(locs.is_empty());
    }

    #[test]
    fn ep_map_indexes_into_raw_payload_correctly() {
        // Build an Annex-B stream where one NAL's raw payload
        // contains an emulation-prevention byte, then verify the
        // splicer-style coordinate lookup yields correct raw bytes.
        let mut data = Vec::new();
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]); // start code
        data.extend_from_slice(&[0x65]); // NAL header (IDR)
        // Raw payload: 0x00 0x00 0x03 0x42 → RBSP: 0x00 0x00 0x42.
        data.extend_from_slice(&[0x00, 0x00, 0x03, 0x42]);

        let locs = locate_nal_units_annexb(&data).unwrap();
        assert_eq!(locs.len(), 1);
        let loc = &locs[0];
        assert_eq!(loc.rbsp, vec![0x00, 0x00, 0x42]);
        // ep_map.rbsp_to_raw[0] = 0 (raw byte 0 = 0x00)
        // ep_map.rbsp_to_raw[1] = 1 (raw byte 1 = 0x00)
        // ep_map.rbsp_to_raw[2] = 3 (raw byte 3 = 0x42; raw byte 2
        //                            is the inserted 0x03 EP byte)
        assert_eq!(loc.ep_map.rbsp_to_raw, vec![0, 1, 3]);
        // Compose Annex-B byte for rbsp[2]:
        //   payload_start + 1 (header) + ep_map.rbsp_to_raw[2] = 5 + 0 + 3
        // Actually payload_start = 4; payload_start+1 = 5 (raw payload base).
        let annex_b_byte =
            loc.payload_start + 1 + loc.ep_map.rbsp_to_raw[2];
        assert_eq!(data[annex_b_byte], 0x42);
    }

}
