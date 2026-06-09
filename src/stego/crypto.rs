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
use std::cell::RefCell;
use zeroize::Zeroizing;
use crate::stego::error::StegoError;

// ─── Argon2id structural-key cache ───────────────────────────────────
//
// Argon2id derivation is ~200 ms per call (OWASP minimum: t=2,
// m=19 MiB, p=1). A smart_decode call runs Ghost / Ghost-shadow /
// Armor / Fortress / Template each with its own fixed salt — so
// 4-5 derivations per decode attempt, ~1 s total on a mid-tier
// phone.
//
// Each mode uses a DIFFERENT fixed salt → can't share Argon2
// outputs across modes within a single decode call. But the same
// (passphrase, salt) input repeats whenever the user decodes
// another image in the same session — that's where the cache pays
// off. A small thread-local LRU absorbs that cost.
//
// Per-frame AES keys via `derive_encryption_key` use random
// per-frame salts and are NOT cached (would just pollute the LRU).
//
// Security: entries hold a copy of the passphrase bytes and the
// derived key, both wrapped in `Zeroizing` so they're scrubbed on
// eviction or thread exit. Callers wanting explicit teardown can
// use `clear_key_cache()`. Thread-local — no cross-thread leakage.

/// Maximum entries in the structural-key cache. Sized to comfortably
/// hold one entry per fixed-salt derivation type (6 today) plus a
/// few extra for multi-passphrase use within one session.
const KEY_CACHE_CAPACITY: usize = 16;

struct KeyCacheEntry {
    passphrase: Zeroizing<Vec<u8>>,
    salt: [u8; 16],
    output: Zeroizing<Vec<u8>>,
}

thread_local! {
    static KEY_CACHE: RefCell<Vec<KeyCacheEntry>> =
        RefCell::new(Vec::with_capacity(KEY_CACHE_CAPACITY));
}

fn cache_lookup(passphrase: &str, salt: &[u8; 16], out_len: usize) -> Option<Vec<u8>> {
    KEY_CACHE.with(|c| {
        let cache = c.borrow();
        cache.iter().find_map(|e| {
            if e.salt == *salt
                && e.output.len() == out_len
                && e.passphrase.as_slice() == passphrase.as_bytes()
            {
                Some(e.output.to_vec())
            } else {
                None
            }
        })
    })
}

fn cache_store(passphrase: &str, salt: &[u8; 16], output: &[u8]) {
    KEY_CACHE.with(|c| {
        let mut cache = c.borrow_mut();
        if cache.len() >= KEY_CACHE_CAPACITY {
            // Simple FIFO eviction. The Zeroizing wrappers ensure
            // both passphrase + derived-key bytes are scrubbed.
            cache.remove(0);
        }
        cache.push(KeyCacheEntry {
            passphrase: Zeroizing::new(passphrase.as_bytes().to_vec()),
            salt: *salt,
            output: Zeroizing::new(output.to_vec()),
        });
    });
}

/// Clear the thread-local Argon2id structural-key cache. Each entry
/// is zeroized on drop. Callers that want to scrub keys at the end
/// of a session (e.g., when a user logs out of an app, or after a
/// CLI batch finishes) should call this. Otherwise entries are
/// evicted FIFO at capacity or when the thread exits.
pub fn clear_key_cache() {
    KEY_CACHE.with(|c| {
        c.borrow_mut().clear();
    });
}

/// Argon2id derivation with thread-local cache. All `derive_*`
/// functions below route through this; cache lookup is keyed on
/// `(passphrase, salt, output.len())`.
fn argon2_derive_cached(
    passphrase: &str,
    salt: &[u8; 16],
    output: &mut [u8],
) -> Result<(), StegoError> {
    if let Some(cached) = cache_lookup(passphrase, salt, output.len()) {
        output.copy_from_slice(&cached);
        return Ok(());
    }
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), salt, output)
        .map_err(|_| StegoError::KeyDerivationFailed)?;
    cache_store(passphrase, salt, output);
    Ok(())
}

/// Fixed salt for Ghost Tier-1 (structural) key derivation.
/// This is intentionally fixed so the decoder can reproduce perm_seed/hhat_seed
/// from the passphrase alone, before extracting the payload.
const STRUCTURAL_SALT: &[u8; 16] = b"phasm-ghost-v1\0\0";

/// Fixed salt for Armor Tier-1 (structural) key derivation.
/// Different from Ghost so the same passphrase produces different permutations.
const ARMOR_STRUCTURAL_SALT: &[u8; 16] = b"phasm-armor-v1\0\0";

/// Fixed salt for DFT template key derivation (Armor geometry resilience).
/// Independent from structural/armor keys so the template peaks are uncorrelated.
const TEMPLATE_SALT: &[u8; 16] = b"phasm-tmpl-v1\0\0\0";

/// Fixed salt for Fortress structural key derivation (BA-QIM block permutation).
/// Independent from Armor STDM keys so the block order is uncorrelated.
const FORTRESS_STRUCTURAL_SALT: &[u8; 16] = b"phasm-fort-v1\0\0\0";

/// AES-GCM-SIV nonce length in bytes.
pub const NONCE_LEN: usize = 12;
/// Argon2 salt length in bytes.
pub const SALT_LEN: usize = 16;

/// Fixed salt for Shadow layer structural key derivation (repetition coding).
/// Independent from all other keys so shadow permutations are uncorrelated.
const SHADOW_STRUCTURAL_SALT: &[u8; 16] = b"phasm-shdw-v1\0\0\0";

/// Fixed salt for the H.264 MVD-domain structural key. Independent from
/// the Ghost structural salt so the MVD-domain permutation + STC matrix do
/// not correlate with the coefficient-domain ones, keeping the two
/// cross-domain flip sets statistically uncorrelated.
const H264_MVD_STRUCTURAL_SALT: &[u8; 16] = b"phasm-h264mvd-v1";

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
pub fn derive_structural_key(passphrase: &str) -> Result<Zeroizing<[u8; 64]>, StegoError> {
    let mut output = Zeroizing::new([0u8; 64]);
    argon2_derive_cached(passphrase, STRUCTURAL_SALT, &mut *output)?;
    Ok(output)
}

/// Derive the H.264 MVD-domain structural key.
///
/// Independent from `derive_structural_key` so the two cross-domain
/// STC runs use uncorrelated permutations and HHat matrices. Same 64-byte
/// layout as the main structural key: first 32 = perm_seed, last 32 =
/// hhat_seed.
pub fn derive_h264_mvd_structural_key(
    passphrase: &str,
) -> Result<Zeroizing<[u8; 64]>, StegoError> {
    let mut output = Zeroizing::new([0u8; 64]);
    argon2_derive_cached(passphrase, H264_MVD_STRUCTURAL_SALT, &mut *output)?;
    Ok(output)
}

/// Derive a per-GOP H.264 seed by mixing a master 32-byte seed
/// with `gop_idx` and a domain label using SHA-256.
///
/// The master seed is already passphrase-derived via `derive_structural_key`
/// or `derive_h264_mvd_structural_key` (each Argon2-expensive but only run
/// once per encode/decode), so this per-GOP derivation can be a fast SHA-256
/// pass — the key material is already secret. Deterministic: same master +
/// `gop_idx` + label → same output across iOS / Android / x86_64 / WASM.
///
/// `label` should be a short distinguishing tag (e.g. `b"coeff-perm"`,
/// `b"coeff-hhat"`) so the four per-GOP seeds (perm + hhat × coeff + MVD)
/// are mutually uncorrelated even when they share a master.
pub fn derive_per_gop_seed(
    master_seed: &[u8; 32],
    gop_idx: u32,
    label: &[u8],
) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(b"phasm-h264-gop-v1");
    hasher.update(master_seed);
    hasher.update(label);
    hasher.update(gop_idx.to_le_bytes());
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Derive the Armor structural key (Tier 1) from a passphrase.
///
/// Same structure as Ghost but uses a different salt so the same passphrase
/// produces different permutation/spreading seeds.
pub fn derive_armor_structural_key(passphrase: &str) -> Result<Zeroizing<[u8; 64]>, StegoError> {
    let mut output = Zeroizing::new([0u8; 64]);
    argon2_derive_cached(passphrase, ARMOR_STRUCTURAL_SALT, &mut *output)?;
    Ok(output)
}

/// Derive the DFT template key from a passphrase.
///
/// Returns a 32-byte key used as a ChaCha20 seed for generating template
/// peak positions. Independent from Ghost/Armor structural keys.
pub fn derive_template_key(passphrase: &str) -> Result<[u8; 32], StegoError> {
    let mut output = [0u8; 32];
    argon2_derive_cached(passphrase, TEMPLATE_SALT, &mut output)?;
    Ok(output)
}

/// Derive the Fortress structural key from a passphrase.
///
/// Returns a 32-byte key used as a ChaCha20 seed for generating block
/// permutation. Independent from Ghost/Armor/Template structural keys.
pub fn derive_fortress_structural_key(passphrase: &str) -> Result<[u8; 32], StegoError> {
    let mut output = [0u8; 32];
    argon2_derive_cached(passphrase, FORTRESS_STRUCTURAL_SALT, &mut output)?;
    Ok(output)
}

/// Derive the Shadow structural key from a passphrase.
///
/// Returns a 32-byte key used as a ChaCha20 seed for generating position
/// permutation for shadow layer repetition coding. Independent from all
/// other structural keys. Wrapped in `Zeroizing` to prevent key material
/// from lingering in memory after use.
pub fn derive_shadow_structural_key(passphrase: &str) -> Result<Zeroizing<[u8; 32]>, StegoError> {
    let mut output = Zeroizing::new([0u8; 32]);
    argon2_derive_cached(passphrase, SHADOW_STRUCTURAL_SALT, &mut *output)?;
    Ok(output)
}

/// Derive the AES-256 encryption key (Tier 2) from passphrase + random salt.
pub fn derive_encryption_key(passphrase: &str, salt: &[u8]) -> Result<Zeroizing<[u8; 32]>, StegoError> {
    let mut key = Zeroizing::new([0u8; 32]);
    Argon2::default()
        .hash_password_into(passphrase.as_bytes(), salt, &mut *key)
        .map_err(|_| StegoError::KeyDerivationFailed)?;
    Ok(key)
}

/// Encrypt plaintext with AES-256-GCM-SIV.
///
/// Returns (ciphertext_with_tag, nonce, salt).
/// The ciphertext includes the 16-byte authentication tag appended by AES-GCM-SIV.
pub fn encrypt(plaintext: &[u8], passphrase: &str) -> Result<(Vec<u8>, [u8; NONCE_LEN], [u8; SALT_LEN]), StegoError> {
    use rand::RngCore;

    let mut salt = [0u8; SALT_LEN];
    let mut nonce_bytes = [0u8; NONCE_LEN];

    // Diagnostic override: `PHASM_DETERMINISTIC_SEED=<u64>` env var
    // replaces the OsRng salt+nonce with a ChaCha8-seeded RNG so
    // consecutive encode runs produce byte-identical output (modulo
    // any other non-determinism in the encoder). Diagnostic-only:
    // production paths leave the env var unset, falling back to
    // `rand::thread_rng` for cryptographic salt+nonce per spec.
    if let Ok(seed_str) = std::env::var("PHASM_DETERMINISTIC_SEED")
        && let Ok(seed) = seed_str.parse::<u64>()
    {
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        rng.fill_bytes(&mut salt);
        rng.fill_bytes(&mut nonce_bytes);
    } else {
        let mut rng = rand::thread_rng();
        rng.fill_bytes(&mut salt);
        rng.fill_bytes(&mut nonce_bytes);
    }

    let key = derive_encryption_key(passphrase, &salt)?;
    let cipher = Aes256GcmSiv::new_from_slice(&*key).expect("valid key length");
    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, plaintext).expect("AES-GCM-SIV encrypt should not fail");

    Ok((ciphertext, nonce_bytes, salt))
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
) -> Result<Vec<u8>, StegoError> {
    let key = derive_encryption_key(passphrase, salt)?;
    let cipher = Aes256GcmSiv::new_from_slice(&*key).expect("valid key length");
    let nonce = Nonce::from_slice(nonce_bytes);

    Ok(cipher.encrypt(nonce, plaintext).expect("AES-GCM-SIV encrypt should not fail"))
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
    let key = derive_encryption_key(passphrase, salt)?;
    let cipher = Aes256GcmSiv::new_from_slice(&*key).expect("valid key length");
    let nonce = Nonce::from_slice(nonce_bytes);

    cipher
        .decrypt(nonce, ciphertext)
        .map_err(|_| StegoError::DecryptionFailed)
}

/// Test-only serialization for the process-global `PHASM_DETERMINISTIC_SEED`
/// env var (read by [`encrypt`] to force a deterministic salt+nonce for
/// repeatable encode runs). Env vars are process-wide, so a test that sets
/// the seed must not run concurrently with one that needs the production
/// (random) crypto path — e.g. `ciphertext_differs_per_encryption`. All
/// seed-toggling tests in the lib test binary acquire this lock.
#[cfg(test)]
pub(crate) static SEED_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// RAII guard: lock [`SEED_ENV_LOCK`] and set `PHASM_DETERMINISTIC_SEED`;
/// clears the var (and releases the lock) on drop. Use in any lib test that
/// needs a deterministic salt+nonce.
#[cfg(test)]
pub(crate) struct DeterministicSeedGuard(std::sync::MutexGuard<'static, ()>);

#[cfg(test)]
impl DeterministicSeedGuard {
    pub(crate) fn set(seed: &str) -> Self {
        let g = SEED_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: writes are serialized by SEED_ENV_LOCK; cleared on drop.
        unsafe { std::env::set_var("PHASM_DETERMINISTIC_SEED", seed) };
        Self(g)
    }
}

#[cfg(test)]
impl Drop for DeterministicSeedGuard {
    fn drop(&mut self) {
        // SAFETY: still holding SEED_ENV_LOCK while we clear the var.
        unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let msg = b"Hello, steganography!";
        let passphrase = "secret123";

        let (ct, nonce, salt) = encrypt(msg, passphrase).unwrap();
        let pt = decrypt(&ct, passphrase, &salt, &nonce).unwrap();
        assert_eq!(pt, msg);
    }

    #[test]
    fn wrong_passphrase_fails() {
        let msg = b"secret message";
        let (ct, nonce, salt) = encrypt(msg, "correct").unwrap();
        let result = decrypt(&ct, "wrong", &salt, &nonce);
        assert!(matches!(result, Err(StegoError::DecryptionFailed)));
    }

    #[test]
    fn empty_message_works() {
        let msg = b"";
        let passphrase = "pass";
        let (ct, nonce, salt) = encrypt(msg, passphrase).unwrap();
        let pt = decrypt(&ct, passphrase, &salt, &nonce).unwrap();
        assert_eq!(pt, msg.to_vec());
    }

    #[test]
    fn structural_key_deterministic() {
        let a = derive_structural_key("mypass").unwrap();
        let b = derive_structural_key("mypass").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn structural_key_differs_by_passphrase() {
        let a = derive_structural_key("pass1").unwrap();
        let b = derive_structural_key("pass2").unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn ghost_and_armor_structural_keys_differ() {
        let ghost = derive_structural_key("same_pass").unwrap();
        let armor = derive_armor_structural_key("same_pass").unwrap();
        assert_ne!(ghost, armor, "Ghost and Armor keys must differ for the same passphrase");
    }

    #[test]
    fn armor_structural_key_deterministic() {
        let a = derive_armor_structural_key("mypass").unwrap();
        let b = derive_armor_structural_key("mypass").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn template_key_deterministic() {
        let a = derive_template_key("mypass").unwrap();
        let b = derive_template_key("mypass").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn fortress_key_deterministic() {
        let a = derive_fortress_structural_key("mypass").unwrap();
        let b = derive_fortress_structural_key("mypass").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn shadow_key_deterministic() {
        let a = derive_shadow_structural_key("mypass").unwrap();
        let b = derive_shadow_structural_key("mypass").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn shadow_key_differs_from_others() {
        let ghost = derive_structural_key("same_pass").unwrap();
        let armor = derive_armor_structural_key("same_pass").unwrap();
        let fortress = derive_fortress_structural_key("same_pass").unwrap();
        let template = derive_template_key("same_pass").unwrap();
        let shadow = derive_shadow_structural_key("same_pass").unwrap();
        assert_ne!(&ghost[..32], &shadow[..]);
        assert_ne!(&armor[..32], &shadow[..]);
        assert_ne!(&fortress[..], &shadow[..]);
        assert_ne!(&template[..], &shadow[..]);
    }

    #[test]
    fn fortress_key_differs_from_others() {
        let ghost = derive_structural_key("same_pass").unwrap();
        let armor = derive_armor_structural_key("same_pass").unwrap();
        let fortress = derive_fortress_structural_key("same_pass").unwrap();
        let template = derive_template_key("same_pass").unwrap();
        assert_ne!(&ghost[..32], &fortress[..]);
        assert_ne!(&armor[..32], &fortress[..]);
        assert_ne!(&template[..], &fortress[..]);
    }

    #[test]
    fn template_key_differs_from_structural() {
        let ghost = derive_structural_key("same_pass").unwrap();
        let armor = derive_armor_structural_key("same_pass").unwrap();
        let template = derive_template_key("same_pass").unwrap();
        assert_ne!(&ghost[..32], &template[..]);
        assert_ne!(&armor[..32], &template[..]);
    }

    #[test]
    fn encryption_key_differs_by_salt() {
        let key1 = derive_encryption_key("pass", &[0u8; 16]).unwrap();
        let key2 = derive_encryption_key("pass", &[1u8; 16]).unwrap();
        assert_ne!(key1, key2);
    }

    #[test]
    fn ciphertext_differs_per_encryption() {
        // Even with the same plaintext and passphrase, each encryption
        // should produce different ciphertext (due to random salt + nonce).
        // Serialize against deterministic-seed tests and force the random
        // path: PHASM_DETERMINISTIC_SEED is process-global, so a parallel
        // AV1 deterministic-seed test could otherwise pin encrypt()'s nonce.
        let _seed_lock = super::SEED_ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        // SAFETY: holding SEED_ENV_LOCK; restore the production (unset) path.
        unsafe { std::env::remove_var("PHASM_DETERMINISTIC_SEED") };
        let msg = b"same message";
        let (ct1, _, _) = encrypt(msg, "pass").unwrap();
        let (ct2, _, _) = encrypt(msg, "pass").unwrap();
        assert_ne!(ct1, ct2, "repeated encryptions should produce different ciphertext");
    }

    // ── T1.1 Argon2id cache tests ──────────────────────────────────

    /// Cache hit must return byte-identical key to a fresh derivation.
    /// Run on its own thread so we don't pollute the main test
    /// thread's cache.
    #[test]
    fn cache_hit_matches_uncached() {
        std::thread::spawn(|| {
            clear_key_cache();
            let pass = "cache-test-pass-hit";
            // First call populates the cache.
            let a = derive_structural_key(pass).unwrap();
            // Second call hits the cache.
            let b = derive_structural_key(pass).unwrap();
            assert_eq!(*a, *b, "cached key must match fresh derivation");
        })
        .join()
        .unwrap();
    }

    /// Different passphrases must produce different cached keys —
    /// cache must not return entry A's key for passphrase B.
    #[test]
    fn cache_separates_passphrases() {
        std::thread::spawn(|| {
            clear_key_cache();
            let a = derive_structural_key("pass-A").unwrap();
            let b = derive_structural_key("pass-B").unwrap();
            assert_ne!(*a, *b);
        })
        .join()
        .unwrap();
    }

    /// Different fixed-salt derivations must NOT collide in the cache
    /// even when called with the same passphrase. Each mode has its
    /// own (passphrase, salt) tuple → distinct entries → distinct
    /// derived keys.
    #[test]
    fn cache_separates_salts() {
        std::thread::spawn(|| {
            clear_key_cache();
            let ghost = derive_structural_key("same").unwrap();
            let armor = derive_armor_structural_key("same").unwrap();
            let shadow = derive_shadow_structural_key("same").unwrap();
            let fortress = derive_fortress_structural_key("same").unwrap();
            let template = derive_template_key("same").unwrap();
            assert_ne!(*ghost, *armor);
            assert_ne!(&(*ghost)[..32], &(*shadow)[..]);
            assert_ne!(&(*ghost)[..32], &fortress[..]);
            assert_ne!(&(*ghost)[..32], &template[..]);
        })
        .join()
        .unwrap();
    }

    /// `clear_key_cache` actually empties the cache. Subsequent
    /// derivations re-run Argon2 (verified by correctness — output
    /// must still match the pre-clear derivation since Argon2id is
    /// deterministic for a given input).
    #[test]
    fn cache_clear_works() {
        std::thread::spawn(|| {
            clear_key_cache();
            let pass = "clear-test";
            let pre = derive_structural_key(pass).unwrap();
            clear_key_cache();
            let post = derive_structural_key(pass).unwrap();
            // Output must match — Argon2 is deterministic.
            assert_eq!(*pre, *post);
        })
        .join()
        .unwrap();
    }

    /// Eviction at capacity: pushing more than KEY_CACHE_CAPACITY
    /// distinct (passphrase, salt) entries evicts the oldest.
    /// Older entry re-derives correctly (Argon2 is deterministic).
    #[test]
    fn cache_evicts_at_capacity() {
        std::thread::spawn(|| {
            clear_key_cache();
            // Populate one extra past capacity using distinct
            // passphrases. derive_template_key is fastest (32-byte
            // output) so this stays under a few seconds.
            for i in 0..KEY_CACHE_CAPACITY + 1 {
                let pass = format!("cap-test-{i}");
                let _ = derive_template_key(&pass).unwrap();
            }
            // Cache should hold exactly KEY_CACHE_CAPACITY entries.
            KEY_CACHE.with(|c| {
                let len = c.borrow().len();
                assert!(
                    len <= KEY_CACHE_CAPACITY,
                    "cache len {len} should be ≤ {KEY_CACHE_CAPACITY}"
                );
            });
            // First passphrase should have been evicted — re-deriving
            // it works (no caching artifact corruption).
            let first = derive_template_key("cap-test-0").unwrap();
            let again = derive_template_key("cap-test-0").unwrap();
            assert_eq!(first, again, "evicted-then-rederived key still consistent");
        })
        .join()
        .unwrap();
    }
}
