// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Cryptographic primitives for payload encryption.
//!
//! Implements a two-tier key derivation scheme using Argon2id:
//!
//! - **Tier 1 (structural)**: Deterministic key derived from passphrase + fixed
//!   salt. Produces `perm_seed` (coefficient permutation) and `hhat_seed`
//!   (STC matrix generation). Both encoder and decoder derive identical keys.
//!
//! - **Tier 2 (encryption)**: AES-256-GCM-SIV key derived from passphrase +
//!   random salt. The random salt is embedded in the payload frame, so the
//!   decoder recovers it from the extracted data.
//!
//! AES-256-GCM-SIV is chosen over AES-256-GCM for its nonce-misuse resistance,
//! which provides an extra safety margin since the nonce is randomly generated
//! and embedded alongside the ciphertext.

use aes_gcm_siv::{Aes256GcmSiv, KeyInit, Nonce};
use aes_gcm_siv::aead::Aead;
use argon2::Argon2;
use zeroize::Zeroizing;
use crate::stego::error::StegoError;

/// Fixed salt for Ghost Tier-1 (structural) key derivation.
/// This is intentionally fixed so the decoder can reproduce perm_seed/hhat_seed
/// from the passphrase alone, before extracting the payload.
const STRUCTURAL_SALT: &[u8; 16] = b"phasm-ghost-v1\0\0";

/// Fixed salt for Armor Tier-1 (structural) key derivation.
/// Different from Ghost so the same passphrase produces different permutations.
const ARMOR_STRUCTURAL_SALT: &[u8; 16] = b"phasm-armor-v1\0\0";

/// Fixed salt for DFT template key derivation (Phase 3 geometry resilience).
/// Independent from structural/armor keys so the template peaks are uncorrelated.
const TEMPLATE_SALT: &[u8; 16] = b"phasm-tmpl-v1\0\0\0";

/// Fixed salt for Fortress structural key derivation (BA-QIM block permutation).
/// Independent from Armor STDM keys so the block order is uncorrelated.
const FORTRESS_STRUCTURAL_SALT: &[u8; 16] = b"phasm-fort-v1\0\0\0";

/// AES-GCM-SIV nonce length in bytes.
pub const NONCE_LEN: usize = 12;
/// Argon2 salt length in bytes.
pub const SALT_LEN: usize = 16;

/// Fixed salt for Fortress empty-passphrase optimization.
/// When passphrase is empty, we use this deterministic salt so it doesn't need
/// to be embedded in the frame (saving 16 bytes). The message is still
/// AES-encrypted so the payload looks random for steganalysis resistance.
/// NOT secret — just a constant to feed into AES key derivation.
pub const FORTRESS_EMPTY_SALT: [u8; SALT_LEN] = *b"phasm-fe-salt00\0";

/// Fixed nonce for Fortress empty-passphrase optimization.
/// When passphrase is empty, we use this deterministic nonce so it doesn't
/// need to be embedded in the frame (saving 12 bytes).
/// NOT secret — just a constant to feed into AES-GCM-SIV.
pub const FORTRESS_EMPTY_NONCE: [u8; NONCE_LEN] = *b"ph-fe-nonce\0";

/// Derive the structural key (Tier 1) from a passphrase.
///
/// Returns a 64-byte buffer: first 32 bytes = perm_seed, last 32 bytes = hhat_seed.
/// This key is deterministic given the passphrase so both encoder and decoder agree.
pub fn derive_structural_key(passphrase: &str) -> Zeroizing<[u8; 64]> {
    let mut output = Zeroizing::new([0u8; 64]);
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), STRUCTURAL_SALT, &mut *output)
        .expect("Argon2 structural key derivation should not fail");
    output
}

/// Derive the Armor structural key (Tier 1) from a passphrase.
///
/// Same structure as Ghost but uses a different salt so the same passphrase
/// produces different permutation/spreading seeds.
pub fn derive_armor_structural_key(passphrase: &str) -> Zeroizing<[u8; 64]> {
    let mut output = Zeroizing::new([0u8; 64]);
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), ARMOR_STRUCTURAL_SALT, &mut *output)
        .expect("Argon2 structural key derivation should not fail");
    output
}

/// Derive the DFT template key from a passphrase.
///
/// Returns a 32-byte key used as a ChaCha20 seed for generating template
/// peak positions. Independent from Ghost/Armor structural keys.
pub fn derive_template_key(passphrase: &str) -> [u8; 32] {
    let mut output = [0u8; 32];
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), TEMPLATE_SALT, &mut output)
        .expect("Argon2 template key derivation should not fail");
    output
}

/// Derive the Fortress structural key from a passphrase.
///
/// Returns a 32-byte key used as a ChaCha20 seed for generating block
/// permutation. Independent from Ghost/Armor/Template structural keys.
pub fn derive_fortress_structural_key(passphrase: &str) -> [u8; 32] {
    let mut output = [0u8; 32];
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), FORTRESS_STRUCTURAL_SALT, &mut output)
        .expect("Argon2 fortress key derivation should not fail");
    output
}

/// Derive the AES-256 encryption key (Tier 2) from passphrase + random salt.
pub fn derive_encryption_key(passphrase: &str, salt: &[u8]) -> Zeroizing<[u8; 32]> {
    let mut key = Zeroizing::new([0u8; 32]);
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), salt, &mut *key)
        .expect("Argon2 encryption key derivation should not fail");
    key
}

/// Encrypt plaintext with AES-256-GCM-SIV.
///
/// Returns (ciphertext_with_tag, nonce, salt).
/// The ciphertext includes the 16-byte authentication tag appended by AES-GCM-SIV.
pub fn encrypt(plaintext: &[u8], passphrase: &str) -> (Vec<u8>, [u8; NONCE_LEN], [u8; SALT_LEN]) {
    use rand::RngCore;
    let mut rng = rand::thread_rng();

    let mut salt = [0u8; SALT_LEN];
    rng.fill_bytes(&mut salt);

    let mut nonce_bytes = [0u8; NONCE_LEN];
    rng.fill_bytes(&mut nonce_bytes);

    let key = derive_encryption_key(passphrase, &salt);
    let cipher = Aes256GcmSiv::new_from_slice(&*key).expect("valid key length");
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext).expect("AES-GCM-SIV encrypt should not fail");

    (ciphertext, nonce_bytes, salt)
}

/// Encrypt plaintext with AES-256-GCM-SIV using caller-provided salt and nonce.
///
/// Used by the Fortress compact frame path where salt and nonce are fixed
/// constants rather than random values.
pub fn encrypt_with(
    plaintext: &[u8],
    passphrase: &str,
    salt: &[u8; SALT_LEN],
    nonce_bytes: &[u8; NONCE_LEN],
) -> Vec<u8> {
    let key = derive_encryption_key(passphrase, salt);
    let cipher = Aes256GcmSiv::new_from_slice(&*key).expect("valid key length");
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher.encrypt(nonce, plaintext).expect("AES-GCM-SIV encrypt should not fail")
}

/// Decrypt ciphertext with AES-256-GCM-SIV.
///
/// Returns the plaintext or `StegoError::DecryptionFailed` if the passphrase is wrong
/// or data is corrupted.
pub fn decrypt(
    ciphertext: &[u8],
    passphrase: &str,
    salt: &[u8],
    nonce_bytes: &[u8; NONCE_LEN],
) -> Result<Vec<u8>, StegoError> {
    let key = derive_encryption_key(passphrase, salt);
    let cipher = Aes256GcmSiv::new_from_slice(&*key).expect("valid key length");
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| StegoError::DecryptionFailed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let msg = b"Hello, steganography!";
        let passphrase = "secret123";

        let (ct, nonce, salt) = encrypt(msg, passphrase);
        let pt = decrypt(&ct, passphrase, &salt, &nonce).unwrap();
        assert_eq!(pt, msg);
    }

    #[test]
    fn wrong_passphrase_fails() {
        let msg = b"secret message";
        let (ct, nonce, salt) = encrypt(msg, "correct");
        let result = decrypt(&ct, "wrong", &salt, &nonce);
        assert!(matches!(result, Err(StegoError::DecryptionFailed)));
    }

    #[test]
    fn empty_message_works() {
        let msg = b"";
        let passphrase = "pass";
        let (ct, nonce, salt) = encrypt(msg, passphrase);
        let pt = decrypt(&ct, passphrase, &salt, &nonce).unwrap();
        assert_eq!(pt, msg.to_vec());
    }

    #[test]
    fn structural_key_deterministic() {
        let a = derive_structural_key("mypass");
        let b = derive_structural_key("mypass");
        assert_eq!(a, b);
    }

    #[test]
    fn structural_key_differs_by_passphrase() {
        let a = derive_structural_key("pass1");
        let b = derive_structural_key("pass2");
        assert_ne!(a, b);
    }

    #[test]
    fn ghost_and_armor_structural_keys_differ() {
        let ghost = derive_structural_key("same_pass");
        let armor = derive_armor_structural_key("same_pass");
        assert_ne!(ghost, armor, "Ghost and Armor keys must differ for the same passphrase");
    }

    #[test]
    fn armor_structural_key_deterministic() {
        let a = derive_armor_structural_key("mypass");
        let b = derive_armor_structural_key("mypass");
        assert_eq!(a, b);
    }

    #[test]
    fn template_key_deterministic() {
        let a = derive_template_key("mypass");
        let b = derive_template_key("mypass");
        assert_eq!(a, b);
    }

    #[test]
    fn fortress_key_deterministic() {
        let a = derive_fortress_structural_key("mypass");
        let b = derive_fortress_structural_key("mypass");
        assert_eq!(a, b);
    }

    #[test]
    fn fortress_key_differs_from_others() {
        let ghost = derive_structural_key("same_pass");
        let armor = derive_armor_structural_key("same_pass");
        let fortress = derive_fortress_structural_key("same_pass");
        let template = derive_template_key("same_pass");
        assert_ne!(&ghost[..32], &fortress[..]);
        assert_ne!(&armor[..32], &fortress[..]);
        assert_ne!(&template[..], &fortress[..]);
    }

    #[test]
    fn template_key_differs_from_structural() {
        let ghost = derive_structural_key("same_pass");
        let armor = derive_armor_structural_key("same_pass");
        let template = derive_template_key("same_pass");
        assert_ne!(&ghost[..32], &template[..]);
        assert_ne!(&armor[..32], &template[..]);
    }

    #[test]
    fn encryption_key_differs_by_salt() {
        let key1 = derive_encryption_key("pass", &[0u8; 16]);
        let key2 = derive_encryption_key("pass", &[1u8; 16]);
        assert_ne!(key1, key2);
    }

    #[test]
    fn ciphertext_differs_per_encryption() {
        // Even with the same plaintext and passphrase, each encryption
        // should produce different ciphertext (due to random salt + nonce).
        let msg = b"same message";
        let (ct1, _, _) = encrypt(msg, "pass");
        let (ct2, _, _) = encrypt(msg, "pass");
        assert_ne!(ct1, ct2, "repeated encryptions should produce different ciphertext");
    }
}
