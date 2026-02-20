pub mod error;
pub mod cost;
pub mod stc;
pub mod crypto;
pub mod frame;
pub mod permute;
pub mod capacity;
mod pipeline;
pub mod armor;

pub use error::StegoError;
pub use pipeline::ghost_encode;
pub use pipeline::ghost_decode;
pub use capacity::estimate_capacity as ghost_capacity;
pub use armor::pipeline::{armor_encode, armor_decode};
pub use armor::capacity::estimate_armor_capacity as armor_capacity;

/// Unified decode: auto-detects Ghost or Armor mode from the embedded frame.
///
/// Tries Ghost first, then Armor. Returns the decoded message and the mode used.
pub fn smart_decode(stego_bytes: &[u8], passphrase: &str) -> Result<(String, u8), StegoError> {
    // Try Ghost first
    match ghost_decode(stego_bytes, passphrase) {
        Ok(text) => return Ok((text, frame::MODE_GHOST)),
        Err(StegoError::DecryptionFailed) => {
            // Could be wrong passphrase for Ghost — still try Armor
        }
        Err(StegoError::FrameCorrupted) | Err(StegoError::UnknownFrameMode(_)) => {
            // Likely not Ghost — try Armor
        }
        Err(e) => {
            // Fundamental error (bad JPEG, too small, etc.) — try Armor anyway
            // in case Ghost fails for mode-specific reasons
            match armor_decode(stego_bytes, passphrase) {
                Ok(text) => return Ok((text, frame::MODE_ARMOR)),
                Err(_) => return Err(e), // Return original Ghost error
            }
        }
    }

    // Try Armor
    match armor_decode(stego_bytes, passphrase) {
        Ok(text) => Ok((text, frame::MODE_ARMOR)),
        Err(_) => Err(StegoError::DecryptionFailed),
    }
}
