use super::error::{JpegError, Result};

/// JPEG marker constants.
pub const SOI: u8 = 0xD8;
pub const EOI: u8 = 0xD9;
pub const SOF0: u8 = 0xC0;
pub const SOF2: u8 = 0xC2;
pub const DHT: u8 = 0xC4;
pub const DQT: u8 = 0xDB;
pub const DRI: u8 = 0xDD;
pub const SOS: u8 = 0xDA;
pub const COM: u8 = 0xFE;

/// A raw marker segment preserving the original bytes.
#[derive(Debug, Clone)]
pub struct MarkerSegment {
    /// The marker byte (e.g., 0xDB for DQT). Does NOT include the 0xFF prefix.
    pub marker: u8,
    /// The segment data NOT including the marker or the 2-byte length field.
    pub data: Vec<u8>,
}

/// Parsed marker with position information.
pub struct MarkerEntry {
    pub marker: u8,
    /// Segment data (empty for standalone markers like SOI, EOI, RST).
    pub data: Vec<u8>,
    /// Byte offset of the marker (the 0xFF byte) in the original data.
    pub offset: usize,
}

/// Iterate over JPEG markers from a byte slice.
///
/// Returns markers and their segment data in order.
/// Stops when SOS is encountered (caller handles entropy-coded data).
pub fn iterate_markers(data: &[u8]) -> Result<(Vec<MarkerEntry>, usize)> {
    let mut entries = Vec::new();
    // Check SOI
    if data.len() < 2 || data[0] != 0xFF || data[1] != SOI {
        return Err(JpegError::InvalidSoi);
    }
    entries.push(MarkerEntry {
        marker: SOI,
        data: Vec::new(),
        offset: 0,
    });
    let mut pos = 2;

    loop {
        // Find next 0xFF
        while pos < data.len() && data[pos] != 0xFF {
            pos += 1;
        }
        if pos + 1 >= data.len() {
            return Err(JpegError::UnexpectedEof);
        }

        // Skip padding 0xFF bytes
        while pos + 1 < data.len() && data[pos + 1] == 0xFF {
            pos += 1;
        }
        if pos + 1 >= data.len() {
            return Err(JpegError::UnexpectedEof);
        }

        let marker_offset = pos;
        let marker = data[pos + 1];
        pos += 2;

        // Skip 0xFF00 (byte-stuffed, shouldn't appear outside scan but handle gracefully)
        if marker == 0x00 {
            continue;
        }

        // Standalone markers (no length field)
        if marker == EOI || (marker >= 0xD0 && marker <= 0xD7) {
            entries.push(MarkerEntry {
                marker,
                data: Vec::new(),
                offset: marker_offset,
            });
            if marker == EOI {
                return Ok((entries, pos));
            }
            continue;
        }

        // Check for unsupported markers
        if is_unsupported(marker) {
            return Err(JpegError::UnsupportedMarker(marker));
        }

        // Read segment length
        if pos + 2 > data.len() {
            return Err(JpegError::UnexpectedEof);
        }
        let length = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
        if length < 2 || pos + length > data.len() {
            return Err(JpegError::InvalidMarkerData("invalid segment length"));
        }
        let segment_data = data[pos + 2..pos + length].to_vec();

        entries.push(MarkerEntry {
            marker,
            data: segment_data,
            offset: marker_offset,
        });

        pos += length;

        // Stop at SOS — scan data follows
        if marker == SOS {
            return Ok((entries, pos));
        }
    }
}

fn is_unsupported(marker: u8) -> bool {
    matches!(
        marker,
        0xC1 // SOF1 extended sequential
        | 0xC2 // SOF2 progressive
        | 0xC3 // SOF3 lossless
        | 0xC5..=0xC7 // SOF5-7 differential
        | 0xC9..=0xCB // SOF9-11 arithmetic
        | 0xCD..=0xCF // SOF13-15 differential arithmetic
    )
}

/// Parse an SOS (Start of Scan) header.
/// Returns component selectors: (component_id, dc_table_id, ac_table_id) per scan component.
pub fn parse_sos(data: &[u8]) -> Result<Vec<(u8, u8, u8)>> {
    if data.is_empty() {
        return Err(JpegError::InvalidMarkerData("empty SOS"));
    }
    let num_components = data[0] as usize;
    if data.len() < 1 + num_components * 2 + 3 {
        return Err(JpegError::UnexpectedEof);
    }

    let mut selectors = Vec::with_capacity(num_components);
    for i in 0..num_components {
        let offset = 1 + i * 2;
        let comp_id = data[offset];
        let td_ta = data[offset + 1];
        let dc_id = td_ta >> 4;
        let ac_id = td_ta & 0x0F;
        selectors.push((comp_id, dc_id, ac_id));
    }

    Ok(selectors)
}

/// Parse DRI (Define Restart Interval) marker data.
pub fn parse_dri(data: &[u8]) -> Result<u16> {
    if data.len() < 2 {
        return Err(JpegError::UnexpectedEof);
    }
    Ok(u16::from_be_bytes([data[0], data[1]]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iterate_minimal_jpeg() {
        // Minimal: SOI + EOI
        let data = [0xFF, 0xD8, 0xFF, 0xD9];
        let (entries, end_pos) = iterate_markers(&data).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].marker, SOI);
        assert_eq!(entries[1].marker, EOI);
        assert_eq!(end_pos, 4);
    }

    #[test]
    fn invalid_soi() {
        let data = [0x00, 0x00];
        assert!(matches!(iterate_markers(&data), Err(JpegError::InvalidSoi)));
    }

    #[test]
    fn reject_progressive() {
        // SOI then SOF2 (progressive)
        let data = [
            0xFF, 0xD8, // SOI
            0xFF, 0xC2, // SOF2
            0x00, 0x02, // length = 2 (minimal)
        ];
        assert!(matches!(
            iterate_markers(&data),
            Err(JpegError::UnsupportedMarker(0xC2))
        ));
    }

    #[test]
    fn parse_sos_header() {
        // 2 components: comp1 uses DC0/AC0, comp2 uses DC1/AC1
        let data = [2, 1, 0x00, 2, 0x11, 0, 63, 0]; // Ss=0, Se=63, Ah/Al=0
        let sels = parse_sos(&data).unwrap();
        assert_eq!(sels.len(), 2);
        assert_eq!(sels[0], (1, 0, 0));
        assert_eq!(sels[1], (2, 1, 1));
    }

    #[test]
    fn parse_dri_value() {
        let data = [0x00, 0x0A]; // restart interval = 10
        assert_eq!(parse_dri(&data).unwrap(), 10);
    }
}
