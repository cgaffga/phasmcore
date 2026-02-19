use aes_gcm::{Aes256Gcm, KeyInit, Nonce};
use aes_gcm::aead::Aead;
use argon2::Argon2;
use crate::stego::error::StegoError;

/// Fixed salt for Tier-1 (structural) key derivation.
/// This is intentionally fixed so the decoder can reproduce perm_seed/hhat_seed
/// from the passphrase alone, before extracting the payload.
const STRUCTURAL_SALT: &[u8; 16] = b"phasm-ghost-v1\0\0";

/// AES-GCM nonce length in bytes.
pub const NONCE_LEN: usize = 12;
/// Argon2 salt length in bytes.
pub const SALT_LEN: usize = 16;

/// Derive the structural key (Tier 1) from a passphrase.
///
/// Returns a 64-byte buffer: first 32 bytes = perm_seed, last 32 bytes = hhat_seed.
/// This key is deterministic given the passphrase so both encoder and decoder agree.
pub fn derive_structural_key(passphrase: &str) -> [u8; 64] {
    let mut output = [0u8; 64];
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), STRUCTURAL_SALT, &mut output)
        .expect("Argon2 structural key derivation should not fail");
    output
}

/// Derive the AES-256 encryption key (Tier 2) from passphrase + random salt.
pub fn derive_encryption_key(passphrase: &str, salt: &[u8]) -> [u8; 32] {
    let mut key = [0u8; 32];
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), salt, &mut key)
        .expect("Argon2 encryption key derivation should not fail");
    key
}

/// Encrypt plaintext with AES-256-GCM.
///
/// Returns (ciphertext_with_tag, nonce, salt).
/// The ciphertext includes the 16-byte authentication tag appended by AES-GCM.
pub fn encrypt(plaintext: &[u8], passphrase: &str) -> (Vec<u8>, [u8; NONCE_LEN], [u8; SALT_LEN]) {
    use rand::RngCore;
    let mut rng = rand::thread_rng();

    let mut salt = [0u8; SALT_LEN];
    rng.fill_bytes(&mut salt);

    let mut nonce_bytes = [0u8; NONCE_LEN];
    rng.fill_bytes(&mut nonce_bytes);

    let key = derive_encryption_key(passphrase, &salt);
    let cipher = Aes256Gcm::new_from_slice(&key).expect("valid key length");
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext).expect("AES-GCM encrypt should not fail");

    (ciphertext, nonce_bytes, salt)
}

/// Decrypt ciphertext with AES-256-GCM.
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
    let cipher = Aes256Gcm::new_from_slice(&key).expect("valid key length");
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
}
