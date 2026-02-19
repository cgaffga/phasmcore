use core::fmt;

/// Errors that can occur during steganographic encoding or decoding.
#[derive(Debug)]
pub enum StegoError {
    /// The cover image could not be parsed as a valid JPEG.
    InvalidJpeg(crate::jpeg::error::JpegError),
    /// The image is too small or has too few usable coefficients.
    ImageTooSmall,
    /// The message is too large for the cover image's embedding capacity.
    MessageTooLarge,
    /// CRC check failed on the extracted payload frame.
    FrameCorrupted,
    /// The payload frame has an unrecognized mode byte.
    UnknownFrameMode(u8),
    /// AES-GCM decryption failed (wrong passphrase or corrupted data).
    DecryptionFailed,
    /// The extracted plaintext is not valid UTF-8.
    InvalidUtf8,
    /// The cover image has no luminance component.
    NoLuminanceChannel,
}

impl fmt::Display for StegoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJpeg(e) => write!(f, "invalid JPEG: {e}"),
            Self::ImageTooSmall => write!(f, "image too small for embedding"),
            Self::MessageTooLarge => write!(f, "message too large for this image"),
            Self::FrameCorrupted => write!(f, "payload frame CRC mismatch"),
            Self::UnknownFrameMode(m) => write!(f, "unknown frame mode: 0x{m:02x}"),
            Self::DecryptionFailed => write!(f, "decryption failed (wrong passphrase?)"),
            Self::InvalidUtf8 => write!(f, "extracted text is not valid UTF-8"),
            Self::NoLuminanceChannel => write!(f, "image has no luminance channel"),
        }
    }
}

impl From<crate::jpeg::error::JpegError> for StegoError {
    fn from(e: crate::jpeg::error::JpegError) -> Self {
        Self::InvalidJpeg(e)
    }
}
