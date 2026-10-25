// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Error types for the steganography pipeline.
//!
//! [`StegoError`] covers all failure modes from JPEG parsing through
//! encryption and frame extraction.

use core::fmt;

/// Errors that can occur during steganographic encoding or decoding.
#[derive(Debug)]
pub enum StegoError {
    /// The cover image could not be parsed as a valid JPEG.
    InvalidJpeg(crate::jpeg::error::JpegError),
    /// The image is too small or has too few usable coefficients.
    ImageTooSmall,
    /// The image dimensions exceed the maximum allowed (16384px / 200MP).
    ImageTooLarge,
    /// The message is too large for the cover image's embedding capacity.
    MessageTooLarge,
    /// CRC check failed on the extracted payload frame.
    FrameCorrupted,
    /// AES-GCM decryption failed (wrong passphrase or corrupted data).
    DecryptionFailed,
    /// The extracted plaintext is not valid UTF-8.
    InvalidUtf8,
    /// The cover image has no luminance component.
    NoLuminanceChannel,
    /// The operation was cancelled by the user.
    Cancelled,
    /// Argon2 key derivation failed (invalid parameters or internal error).
    KeyDerivationFailed,
    /// Duplicate passphrase: each shadow layer must use a unique passphrase.
    DuplicatePassphrase,
    /// Shadow embedding failed cascade — encoder verification couldn't
    /// recover the shadow message after escalating through all parity
    /// tiers `[4, 8, 16, 32, 64, 128]`. Caller can retry with a smaller
    /// primary message (less propagation), different `gop_size`/quality,
    /// or a different shadow passphrase.
    ShadowEmbedFailed,
    /// The video file is invalid or uses unsupported features.
    InvalidVideo(String),
}

impl fmt::Display for StegoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJpeg(e) => write!(f, "invalid JPEG: {e}"),
            Self::ImageTooSmall => write!(f, "image too small for embedding"),
            Self::ImageTooLarge => write!(f, "image too large (max 16384px / 200MP)"),
            // "cover", not "image": this variant is returned by the video
            // paths too (per-GOP plan + STC trial), and "too large for this
            // image" reached users verbatim over a .MOV on 2026-10-20.
            Self::MessageTooLarge => write!(f, "message too large for this cover"),
            Self::FrameCorrupted => write!(f, "payload frame CRC mismatch"),
            Self::DecryptionFailed => write!(f, "decryption failed (wrong passphrase?)"),
            Self::InvalidUtf8 => write!(f, "extracted text is not valid UTF-8"),
            Self::NoLuminanceChannel => write!(f, "image has no luminance channel"),
            Self::Cancelled => write!(f, "operation cancelled by user"),
            Self::KeyDerivationFailed => write!(f, "key derivation failed"),
            Self::DuplicatePassphrase => write!(f, "duplicate passphrase (each layer must use a unique passphrase)"),
            Self::ShadowEmbedFailed => write!(f, "shadow embed failed: cascade exhausted at parity tier 128 — try a smaller primary message, different gop_size/quality, or different shadow passphrase"),
            // The reason code is for machines (bridges, apps); people get
            // the detail after it.
            Self::InvalidVideo(s) => {
                let detail = s
                    .split_once(':')
                    .filter(|(code, _)| VideoErrorReason::from_code(code).is_some())
                    .map_or(s.as_str(), |(_, rest)| rest);
                write!(f, "invalid video: {detail}")
            }
        }
    }
}

impl std::error::Error for StegoError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidJpeg(e) => Some(e),
            _ => None,
        }
    }
}

impl From<crate::jpeg::error::JpegError> for StegoError {
    fn from(e: crate::jpeg::error::JpegError) -> Self {
        Self::InvalidJpeg(e)
    }
}

#[cfg(feature = "video")]
impl From<crate::codec::h264::H264Error> for StegoError {
    fn from(e: crate::codec::h264::H264Error) -> Self {
        // A bitstream the H.264 parser rejects.
        Self::invalid_video(VideoErrorReason::Damaged, e)
    }
}

#[cfg(feature = "video")]
impl From<crate::codec::mp4::Mp4Error> for StegoError {
    fn from(e: crate::codec::mp4::Mp4Error) -> Self {
        use crate::codec::mp4::Mp4Error as M;
        let reason = match e {
            M::NoVideoTrack => VideoErrorReason::NoVideoTrack,
            M::TruncatedFile => VideoErrorReason::Truncated,
            M::FragmentedMp4 => VideoErrorReason::FragmentedMp4,
            M::UnsupportedCodec => VideoErrorReason::UnsupportedCodec,
            M::UnexpectedEof | M::InvalidBox(_) => VideoErrorReason::Damaged,
        };
        Self::invalid_video(reason, e)
    }
}

/// Why a video could not be read — the stable sub-code apps map to a
/// message. Every video error crosses the FFI as
/// `INVALID_VIDEO:<REASON>:<detail>`: the apps phrase `<REASON>` from their
/// string catalogs and log `<detail>`, which stays free English. The code
/// strings are a protocol between core, bridges and both apps: never
/// rename one (tests pin them).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VideoErrorReason {
    /// The file has no video track.
    NoVideoTrack,
    /// The file ends early (an interrupted recording or copy).
    Truncated,
    /// Fragmented MP4 (fMP4 / DASH segments), which phasm does not read.
    FragmentedMp4,
    /// The video is DRM-protected.
    DrmProtected,
    /// The video codec is not one phasm reads.
    UnsupportedCodec,
    /// The frame size is outside what the encoders take.
    UnsupportedDimensions,
    /// The file or its bitstream is malformed.
    Damaged,
    /// Anything else.
    Other,
}

impl VideoErrorReason {
    /// Every reason, for tests and for callers listing them.
    pub const ALL: [Self; 8] = [
        Self::NoVideoTrack,
        Self::Truncated,
        Self::FragmentedMp4,
        Self::DrmProtected,
        Self::UnsupportedCodec,
        Self::UnsupportedDimensions,
        Self::Damaged,
        Self::Other,
    ];

    /// The wire code.
    pub fn code(self) -> &'static str {
        match self {
            Self::NoVideoTrack => "NO_VIDEO_TRACK",
            Self::Truncated => "TRUNCATED",
            Self::FragmentedMp4 => "FRAGMENTED_MP4",
            Self::DrmProtected => "DRM_PROTECTED",
            Self::UnsupportedCodec => "UNSUPPORTED_CODEC",
            Self::UnsupportedDimensions => "UNSUPPORTED_DIMENSIONS",
            Self::Damaged => "DAMAGED",
            Self::Other => "OTHER",
        }
    }

    /// The reason whose code is `code`.
    pub fn from_code(code: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|r| r.code() == code)
    }
}

impl StegoError {
    /// An [`InvalidVideo`](Self::InvalidVideo) error carrying `reason`:
    /// its detail is `"<REASON>:<detail>"`.
    pub fn invalid_video(reason: VideoErrorReason, detail: impl fmt::Display) -> Self {
        Self::InvalidVideo(format!("{}:{detail}", reason.code()))
    }
}

/// The wire form of an `InvalidVideo` detail: unchanged when it already
/// starts with a reason code, else prefixed with `OTHER:` — so every
/// `INVALID_VIDEO:` error the bridges emit names a reason.
pub fn invalid_video_wire(detail: &str) -> String {
    let head = detail.split(':').next().unwrap_or_default();
    if VideoErrorReason::from_code(head).is_some() {
        detail.to_string()
    } else {
        format!("{}:{detail}", VideoErrorReason::Other.code())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The codes are a protocol with both apps: pin every one.
    #[test]
    fn reason_codes_are_pinned() {
        let codes: Vec<&str> = VideoErrorReason::ALL.iter().map(|r| r.code()).collect();
        assert_eq!(
            codes,
            [
                "NO_VIDEO_TRACK",
                "TRUNCATED",
                "FRAGMENTED_MP4",
                "DRM_PROTECTED",
                "UNSUPPORTED_CODEC",
                "UNSUPPORTED_DIMENSIONS",
                "DAMAGED",
                "OTHER",
            ]
        );
        for r in VideoErrorReason::ALL {
            assert_eq!(VideoErrorReason::from_code(r.code()), Some(r));
        }
        assert_eq!(VideoErrorReason::from_code("no video track"), None);
    }

    #[test]
    fn invalid_video_carries_its_reason() {
        let e = StegoError::invalid_video(VideoErrorReason::Truncated, "ends at byte 42");
        assert!(matches!(&e, StegoError::InvalidVideo(d) if d == "TRUNCATED:ends at byte 42"));
    }

    /// People read Display (the CLI prints it); the code is for machines.
    #[test]
    fn display_leaves_the_reason_code_out() {
        let e = StegoError::invalid_video(VideoErrorReason::NoVideoTrack, "no video track in MP4 file");
        assert_eq!(e.to_string(), "invalid video: no video track in MP4 file");
        assert_eq!(StegoError::InvalidVideo("walk: x".into()).to_string(), "invalid video: walk: x");
    }

    #[test]
    fn wire_form_always_names_a_reason() {
        assert_eq!(invalid_video_wire("DAMAGED:bad box"), "DAMAGED:bad box");
        assert_eq!(invalid_video_wire("NO_VIDEO_TRACK"), "NO_VIDEO_TRACK");
        assert_eq!(invalid_video_wire("walk: bad slice"), "OTHER:walk: bad slice");
        assert_eq!(invalid_video_wire("damaged: lower case is not a code"), "OTHER:damaged: lower case is not a code");
        assert_eq!(invalid_video_wire(""), "OTHER:");
    }

    #[cfg(feature = "video")]
    #[test]
    fn mp4_errors_map_to_reasons() {
        use crate::codec::mp4::Mp4Error;
        let reason = |e: Mp4Error| match StegoError::from(e) {
            StegoError::InvalidVideo(d) => d.split(':').next().unwrap().to_string(),
            other => panic!("{other:?}"),
        };
        assert_eq!(reason(Mp4Error::NoVideoTrack), "NO_VIDEO_TRACK");
        assert_eq!(reason(Mp4Error::TruncatedFile), "TRUNCATED");
        assert_eq!(reason(Mp4Error::FragmentedMp4), "FRAGMENTED_MP4");
        assert_eq!(reason(Mp4Error::UnsupportedCodec), "UNSUPPORTED_CODEC");
        assert_eq!(reason(Mp4Error::UnexpectedEof), "DAMAGED");
        assert_eq!(reason(Mp4Error::InvalidBox("stsz".into())), "DAMAGED");
    }
}
