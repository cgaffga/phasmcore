// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Known-answer tests for the crypto every phasm message depends on: the
//! Argon2id key derivation and AES-256-GCM-SIV. A dependency upgrade that
//! changes either output would make every released message undecodable, so
//! these values are frozen (computed with argon2 0.5 / aes-gcm-siv 0.11 on
//! 2026-11-02; unchanged by argon2 0.6 / aes-gcm-siv 0.12).

use phasm_core::stego::crypto;

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn argon2id_key_derivation_is_frozen() {
    let k = crypto::derive_encryption_key("correct horse", &[7u8; 16]).unwrap();
    assert_eq!(hex(&k[..]), "7132e6a6028b7abb94eb36beb767146777cd7c75426211a827c8a7a5c50d7307");
}

#[test]
fn aes_gcm_siv_compact_frame_is_frozen() {
    let ct = crypto::encrypt_with(b"hello phasm", "pw", &crypto::FORTRESS_EMPTY_SALT, &crypto::FORTRESS_EMPTY_NONCE).unwrap();
    assert_eq!(hex(&ct), "8790ea4e97b4e22a3a4e6af9a3e0f3b43f4bb32db6ebc87c9c37ce");
}
