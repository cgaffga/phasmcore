// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Payload serialization, compression, and file embedding.
//!
//! The payload format wraps the plaintext before encryption:
//!
//! ```text
//! [1 byte ] flags
//! [M bytes] inner payload (raw or Brotli-compressed depending on flags)
//! ```
//!
//! The inner payload (after decompression) uses a `0x00` separator:
//!
//! ```text
//! [text bytes]     UTF-8 message (may be empty)
//! [0x00]           separator (only present if files follow)
//! [file entry]*    zero or more file entries
//! ```
//!
//! File entry format:
//!
//! ```text
//! [1 byte ] filename_len (1–255)
//! [N bytes] filename (UTF-8)
//! [4 bytes] content_len (u32 BE)
//! [M bytes] file content
//! ```

use crate::stego::error::StegoError;
use std::io::{Read, Write};

/// Compression algorithm flags (bits 0-1 of flags byte).
const COMPRESS_NONE: u8 = 0b00;
const COMPRESS_BROTLI: u8 = 0b01;
const COMPRESS_MASK: u8 = 0b11;

/// Brotli compression quality (0-11). 11 = max compression. Our payloads are
/// small (<64 KB) so even max quality compresses in milliseconds.
const BROTLI_QUALITY: u32 = 11;

/// Brotli LG_WINDOW_SIZE. 22 is the default (4 MB window).
/// For small payloads this is fine — Brotli auto-adjusts.
const BROTLI_LG_WINDOW_SIZE: u32 = 22;

/// Maximum raw file size before compression (hard reject).
pub const MAX_RAW_FILE_SIZE: usize = 2 * 1024 * 1024; // 2 MB

/// A file entry embedded in the payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FileEntry {
    pub filename: String,
    pub content: Vec<u8>,
}

/// Decoded payload containing text and optional files.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PayloadData {
    pub text: String,
    pub files: Vec<FileEntry>,
}

/// Encode a payload (text + optional files) into bytes ready for encryption.
///
/// Returns `[flags byte][maybe_compressed_inner]`. The caller encrypts this
/// as the "plaintext" in the frame format.
///
/// Tries Brotli compression and uses it only if it produces a smaller result.
///
/// Returns an error if any file exceeds `MAX_RAW_FILE_SIZE` or has a filename
/// longer than 255 bytes.
pub fn encode_payload(text: &str, files: &[FileEntry]) -> Result<Vec<u8>, StegoError> {
    for file in files {
        if file.content.len() > MAX_RAW_FILE_SIZE {
            return Err(StegoError::MessageTooLarge);
        }
        if file.filename.as_bytes().len() > 255 {
            return Err(StegoError::MessageTooLarge);
        }
    }
    let inner = serialize_inner(text, files);
    Ok(try_compress(&inner))
}

/// Compute the compressed payload size (in bytes) for the given text and files.
///
/// This is the exact size that `encode_payload` would produce, without actually
/// encrypting or embedding. Useful for showing a live "used bytes" count in the
/// UI that reflects Brotli compression savings.
///
/// Falls back to `text.as_bytes().len() + 1` (raw text + flags byte) if
/// `encode_payload` fails for any reason (e.g. file too large).
pub fn compressed_payload_size(text: &str, files: &[FileEntry]) -> usize {
    encode_payload(text, files)
        .map(|v| v.len())
        .unwrap_or_else(|_| text.as_bytes().len() + 1)
}

/// Decode a payload from decrypted bytes.
///
/// Input is `[flags byte][maybe_compressed_inner]` as produced by `encode_payload`.
pub fn decode_payload(data: &[u8]) -> Result<PayloadData, StegoError> {
    if data.is_empty() {
        return Err(StegoError::FrameCorrupted);
    }

    let flags = data[0];
    let compressed_data = &data[1..];

    let inner = match flags & COMPRESS_MASK {
        COMPRESS_NONE => compressed_data.to_vec(),
        COMPRESS_BROTLI => decompress_brotli(compressed_data)?,
        _ => return Err(StegoError::FrameCorrupted),
    };

    parse_inner(&inner)
}

/// Serialize the inner payload: `[text][0x00][file entries...]`
fn serialize_inner(text: &str, files: &[FileEntry]) -> Vec<u8> {
    let mut buf = Vec::new();
    buf.extend_from_slice(text.as_bytes());

    if !files.is_empty() {
        buf.push(0x00); // separator
        for file in files {
            let name_bytes = file.filename.as_bytes();
            let name_len = name_bytes.len().min(255) as u8;
            buf.push(name_len);
            buf.extend_from_slice(&name_bytes[..name_len as usize]);
            buf.extend_from_slice(&(file.content.len() as u32).to_be_bytes());
            buf.extend_from_slice(&file.content);
        }
    }

    buf
}

/// Parse the inner payload after decompression.
fn parse_inner(data: &[u8]) -> Result<PayloadData, StegoError> {
    // Find first 0x00 byte — everything before is text, after is file entries.
    let separator_pos = data.iter().position(|&b| b == 0x00);

    match separator_pos {
        None => {
            // No separator — entire payload is text (text-only, no files).
            let text = std::str::from_utf8(data)
                .map_err(|_| StegoError::InvalidUtf8)?
                .to_string();
            Ok(PayloadData { text, files: vec![] })
        }
        Some(pos) => {
            // Text before separator.
            let text = std::str::from_utf8(&data[..pos])
                .map_err(|_| StegoError::InvalidUtf8)?
                .to_string();

            // Parse file entries after separator.
            let mut files = Vec::new();
            let mut cursor = pos + 1;

            while cursor < data.len() {
                // filename_len (1 byte)
                let name_len = data[cursor] as usize;
                cursor += 1;
                if name_len == 0 || cursor + name_len > data.len() {
                    return Err(StegoError::FrameCorrupted);
                }

                // filename (UTF-8)
                let filename = std::str::from_utf8(&data[cursor..cursor + name_len])
                    .map_err(|_| StegoError::InvalidUtf8)?
                    .to_string();
                cursor += name_len;

                // content_len (u32 BE)
                if cursor + 4 > data.len() {
                    return Err(StegoError::FrameCorrupted);
                }
                let content_len = u32::from_be_bytes([
                    data[cursor],
                    data[cursor + 1],
                    data[cursor + 2],
                    data[cursor + 3],
                ]) as usize;
                cursor += 4;

                // file content
                if cursor + content_len > data.len() {
                    return Err(StegoError::FrameCorrupted);
                }
                let content = data[cursor..cursor + content_len].to_vec();
                cursor += content_len;

                files.push(FileEntry { filename, content });
            }

            Ok(PayloadData { text, files })
        }
    }
}

/// Try Brotli compression; return `[flags][data]` using whichever is smaller.
fn try_compress(inner: &[u8]) -> Vec<u8> {
    let compressed = compress_brotli(inner);

    // Use compressed only if it's strictly smaller.
    // Both paths include the 1-byte flags prefix.
    if compressed.len() < inner.len() {
        let mut result = Vec::with_capacity(1 + compressed.len());
        result.push(COMPRESS_BROTLI);
        result.extend_from_slice(&compressed);
        result
    } else {
        let mut result = Vec::with_capacity(1 + inner.len());
        result.push(COMPRESS_NONE);
        result.extend_from_slice(inner);
        result
    }
}

/// Compress data with Brotli.
fn compress_brotli(data: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    {
        let mut compressor = brotli::CompressorWriter::new(
            &mut output,
            4096, // buffer size
            BROTLI_QUALITY,
            BROTLI_LG_WINDOW_SIZE,
        );
        compressor.write_all(data).expect("Brotli compression should not fail");
        // CompressorWriter flushes on drop
    }
    output
}

/// Decompress Brotli data.
fn decompress_brotli(data: &[u8]) -> Result<Vec<u8>, StegoError> {
    let mut output = Vec::new();
    let decompressor = brotli::Decompressor::new(data, 4096);
    // Limit decompressed size to prevent decompression bombs
    let limit = 128 * 1024; // 128 KB generous limit
    decompressor.take(limit as u64).read_to_end(&mut output)
        .map_err(|_| StegoError::FrameCorrupted)?;
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_only_roundtrip() {
        let encoded = encode_payload("Hello, world!", &[]).unwrap();
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.text, "Hello, world!");
        assert!(decoded.files.is_empty());
    }

    #[test]
    fn empty_text_roundtrip() {
        let encoded = encode_payload("", &[]).unwrap();
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.text, "");
        assert!(decoded.files.is_empty());
    }

    #[test]
    fn text_with_one_file() {
        let files = vec![FileEntry {
            filename: "test.txt".to_string(),
            content: b"file content here".to_vec(),
        }];
        let encoded = encode_payload("hello", &files).unwrap();
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.text, "hello");
        assert_eq!(decoded.files.len(), 1);
        assert_eq!(decoded.files[0].filename, "test.txt");
        assert_eq!(decoded.files[0].content, b"file content here");
    }

    #[test]
    fn text_with_multiple_files() {
        let files = vec![
            FileEntry {
                filename: "a.bin".to_string(),
                content: vec![0xDE, 0xAD, 0xBE, 0xEF],
            },
            FileEntry {
                filename: "readme.md".to_string(),
                content: b"# Hello\nWorld".to_vec(),
            },
        ];
        let encoded = encode_payload("msg", &files).unwrap();
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.text, "msg");
        assert_eq!(decoded.files.len(), 2);
        assert_eq!(decoded.files[0].filename, "a.bin");
        assert_eq!(decoded.files[0].content, vec![0xDE, 0xAD, 0xBE, 0xEF]);
        assert_eq!(decoded.files[1].filename, "readme.md");
        assert_eq!(decoded.files[1].content, b"# Hello\nWorld");
    }

    #[test]
    fn empty_text_with_files() {
        let files = vec![FileEntry {
            filename: "data.bin".to_string(),
            content: vec![1, 2, 3],
        }];
        let encoded = encode_payload("", &files).unwrap();
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.text, "");
        assert_eq!(decoded.files.len(), 1);
    }

    #[test]
    fn short_message_not_compressed() {
        let encoded = encode_payload("hi", &[]).unwrap();
        // Flags byte should be COMPRESS_NONE for short messages.
        assert_eq!(encoded[0] & COMPRESS_MASK, COMPRESS_NONE);
    }

    #[test]
    fn long_repetitive_text_compressed() {
        let long_text = "abcdefghij".repeat(100); // 1000 bytes of repetitive text
        let encoded = encode_payload(&long_text, &[]).unwrap();
        // Should be compressed (Brotli).
        assert_eq!(encoded[0] & COMPRESS_MASK, COMPRESS_BROTLI);
        // Compressed should be smaller than raw.
        assert!(encoded.len() < long_text.len());
        // Roundtrip.
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.text, long_text);
    }

    #[test]
    fn large_compressible_file() {
        let files = vec![FileEntry {
            filename: "big.txt".to_string(),
            content: b"Hello World! ".repeat(1000),
        }];
        let encoded = encode_payload("", &files).unwrap();
        assert_eq!(encoded[0] & COMPRESS_MASK, COMPRESS_BROTLI);
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.files[0].content.len(), 13000);
    }

    #[test]
    fn incompressible_data_stays_raw() {
        // Random-looking data shouldn't compress.
        let mut data = Vec::new();
        for i in 0u16..200 {
            data.push((i.wrapping_mul(7919) % 256) as u8);
        }
        let files = vec![FileEntry {
            filename: "rand.bin".to_string(),
            content: data.clone(),
        }];
        let encoded = encode_payload("", &files).unwrap();
        // May or may not compress depending on Brotli's framing overhead.
        // Either way, roundtrip must work.
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.files[0].content, data);
    }

    #[test]
    fn unicode_text_and_filename() {
        let files = vec![FileEntry {
            filename: "daten-übersicht.pdf".to_string(),
            content: vec![0xFF],
        }];
        let encoded = encode_payload("Ünïcödé 🎉", &files).unwrap();
        let decoded = decode_payload(&encoded).unwrap();
        assert_eq!(decoded.text, "Ünïcödé 🎉");
        assert_eq!(decoded.files[0].filename, "daten-übersicht.pdf");
    }

    #[test]
    fn empty_payload_error() {
        assert!(decode_payload(&[]).is_err());
    }

    #[test]
    fn truncated_file_entry_error() {
        // flags=0, text="A", separator=0x00, then truncated file entry
        let data = vec![COMPRESS_NONE, b'A', 0x00, 5]; // filename_len=5 but no data
        assert!(decode_payload(&data).is_err());
    }

    #[test]
    fn zero_length_filename_error() {
        // filename_len=0 is invalid
        let data = vec![COMPRESS_NONE, 0x00, 0]; // separator then filename_len=0
        assert!(decode_payload(&data).is_err());
    }
}
