// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Debug and diagnostic tests for steganographic internals.

use phasm_core::JpegImage;
use phasm_core::stego::cost::uniward::compute_uniward;
use phasm_core::stego::permute;
use phasm_core::stego::crypto;
use phasm_core::stego::frame::{self, MAX_FRAME_BITS};
use phasm_core::stego::stc::{embed, extract, hhat};

#[test]
fn rebuild_huffman_lossless() {
    let data = std::fs::read("test-vectors/photo_320x240_q75_420.jpg").unwrap();
    let mut img = JpegImage::from_bytes(&data).unwrap();
    let grid_before = img.dct_grid(0).clone();
    img.rebuild_huffman_tables();
    let bytes = img.to_bytes().unwrap();
    let img2 = JpegImage::from_bytes(&bytes).unwrap();
    let grid_after = img2.dct_grid(0);
    for br in 0..grid_before.blocks_tall() {
        for bc in 0..grid_before.blocks_wide() {
            for i in 0..8 {
                for j in 0..8 {
                    assert_eq!(grid_before.get(br, bc, i, j), grid_after.get(br, bc, i, j));
                }
            }
        }
    }
}

/// Full round-trip diagnostic: manually trace encode and decode.
#[test]
fn full_pipeline_diagnostic() {
    let cover_data = std::fs::read("test-vectors/photo_320x240_q75_420.jpg").unwrap();
    let passphrase = "test-pass";
    let message = "Hello!";

    // === ENCODE SIDE ===
    let mut img = JpegImage::from_bytes(&cover_data).unwrap();
    let qt_id = img.frame_info().components[0].quant_table_id as usize;
    let qt = img.quant_table(qt_id).unwrap();
    let cost_map = compute_uniward(img.dct_grid(0), qt);

    let structural_key = crypto::derive_structural_key(passphrase);
    let perm_seed: [u8; 32] = structural_key[..32].try_into().unwrap();
    let hhat_seed: [u8; 32] = structural_key[32..].try_into().unwrap();

    let all_positions = permute::select_and_permute(&cost_map, &perm_seed);
    let n = all_positions.len();
    // m_max adapts to the image: capped at MAX_FRAME_BITS but also at n
    // so that small images still work (w >= 1). Both encoder and decoder
    // compute the same m_max from the same image.
    let m_max = MAX_FRAME_BITS.min(n);
    let w = n / m_max;
    let n_used = m_max * w;
    eprintln!("ENCODE: n={n}, w={w}, n_used={n_used}, m_max={m_max}");

    let positions = &all_positions[..n_used];
    let bw = img.dct_grid(0).blocks_wide();

    let flat_get = |grid: &phasm_core::DctGrid, flat_idx: usize| -> i16 {
        let bi = flat_idx / 64;
        let pos = flat_idx % 64;
        grid.get(bi / bw, bi % bw, pos / 8, pos % 8)
    };

    // Build frame
    let (ciphertext, nonce, salt) = crypto::encrypt(message.as_bytes(), passphrase);
    let frame_bytes = frame::build_frame(message.len() as u16, &salt, &nonce, &ciphertext);
    let frame_bits = frame::bytes_to_bits(&frame_bytes);
    let mut padded_bits = vec![0u8; m_max];
    padded_bits[..frame_bits.len()].copy_from_slice(&frame_bits);

    // Cover bits
    let grid = img.dct_grid(0);
    let cover_bits: Vec<u8> = positions.iter().map(|p| {
        (flat_get(grid, p.flat_idx).unsigned_abs() & 1) as u8
    }).collect();
    let costs: Vec<f64> = positions.iter().map(|p| p.cost).collect();

    // STC embed
    let hhat_matrix = hhat::generate_hhat(7, w, &hhat_seed);
    let result = embed::stc_embed(&cover_bits, &costs, &padded_bits, &hhat_matrix, 7, w).unwrap();

    // Verify pre-JPEG extraction
    let pre_extracted = extract::stc_extract(&result.stego_bits, &hhat_matrix, w);
    assert_eq!(&pre_extracted[..m_max], &padded_bits[..], "pre-JPEG extraction mismatch");

    // Apply modifications (same logic as pipeline)
    let grid_mut = img.dct_grid_mut(0);
    for (idx, pos) in positions.iter().enumerate() {
        if cover_bits[idx] != result.stego_bits[idx] {
            let bi = pos.flat_idx / 64;
            let p = pos.flat_idx % 64;
            let (br, bc, i, j) = (bi / bw, bi % bw, p / 8, p % 8);
            let coeff = grid_mut.get(br, bc, i, j);
            let modified = if coeff == 1 { 2 }
                else if coeff == -1 { -2 }
                else if coeff > 1 { coeff - 1 }
                else if coeff < -1 { coeff + 1 }
                else { coeff };
            grid_mut.set(br, bc, i, j, modified);
        }
    }

    // Read stego LSBs from in-memory image (before JPEG write)
    let grid = img.dct_grid(0);
    let mem_stego_bits: Vec<u8> = positions.iter().map(|p| {
        (flat_get(grid, p.flat_idx).unsigned_abs() & 1) as u8
    }).collect();
    assert_eq!(mem_stego_bits, result.stego_bits, "in-memory LSBs != stego_bits");

    // Write and re-read JPEG
    img.rebuild_huffman_tables();
    let stego_data = img.to_bytes().unwrap();
    let stego_img = JpegImage::from_bytes(&stego_data).unwrap();

    // Check coefficients survived JPEG round-trip
    let stego_grid = stego_img.dct_grid(0);
    let jpeg_stego_bits: Vec<u8> = positions.iter().map(|p| {
        (flat_get(stego_grid, p.flat_idx).unsigned_abs() & 1) as u8
    }).collect();

    let mut jpeg_mismatches = 0;
    for i in 0..n_used {
        if jpeg_stego_bits[i] != result.stego_bits[i] {
            jpeg_mismatches += 1;
        }
    }
    eprintln!("JPEG round-trip LSB mismatches: {jpeg_mismatches}");
    assert_eq!(jpeg_mismatches, 0, "JPEG round-trip changed LSBs");

    // === DECODE SIDE (using stego image positions) ===
    let stego_cost_map = compute_uniward(stego_img.dct_grid(0), stego_img.quant_table(qt_id).unwrap());
    let stego_all_positions = permute::select_and_permute(&stego_cost_map, &perm_seed);
    let n_stego = stego_all_positions.len();
    let m_max_stego = MAX_FRAME_BITS.min(n_stego);
    let w_stego = n_stego / m_max_stego;
    let n_used_stego = m_max_stego * w_stego;
    eprintln!("DECODE: n={n_stego}, w={w_stego}, n_used={n_used_stego}");

    assert_eq!(n, n_stego, "n differs between encoder and decoder");
    assert_eq!(w, w_stego, "w differs");
    assert_eq!(n_used, n_used_stego, "n_used differs");

    let stego_positions = &stego_all_positions[..n_used_stego];

    // Check position order matches
    for i in 0..n_used {
        assert_eq!(
            positions[i].flat_idx, stego_positions[i].flat_idx,
            "position {i} differs"
        );
    }

    // Decoder's stego bits
    let decoder_stego_bits: Vec<u8> = stego_positions.iter().map(|p| {
        (flat_get(stego_grid, p.flat_idx).unsigned_abs() & 1) as u8
    }).collect();
    assert_eq!(decoder_stego_bits, jpeg_stego_bits, "decoder reads different bits");

    // STC extract from decoder's perspective
    let hhat_matrix_decode = hhat::generate_hhat(7, w_stego, &hhat_seed);
    assert_eq!(hhat_matrix, hhat_matrix_decode, "H-hat matrices differ");

    let decoded_bits = extract::stc_extract(&decoder_stego_bits, &hhat_matrix_decode, w_stego);
    assert_eq!(&decoded_bits[..m_max], &padded_bits[..], "decoded bits != original padded bits");
}
