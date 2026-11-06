// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! What identifying metadata a JPEG carries, for the CLI's warning when the
//! output keeps the source's metadata (fix list R3.10, decided 2026-11-02:
//! keep it by default, but say what is kept; `--optimize` strips it).
//!
//! Reads the segments before the first scan and walks EXIF IFD0, IFD1, the
//! Exif IFD and the GPS IFD once each (no recursion, every read
//! bounds-checked: the input is untrusted).

/// Human-readable names of the identifying metadata in `jpeg`, in a fixed
/// order; empty when there is none worth a warning.
pub fn identifying_metadata(jpeg: &[u8]) -> Vec<&'static str> {
    let mut found = Found::default();
    if jpeg.len() < 4 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        return Vec::new();
    }
    let mut p = 2;
    while p + 4 <= jpeg.len() {
        if jpeg[p] != 0xFF {
            break;
        }
        let m = jpeg[p + 1];
        if m == 0xFF {
            p += 1;
            continue;
        }
        if m == 0xDA || m == 0xD9 {
            break;
        }
        let len = u16::from_be_bytes([jpeg[p + 2], jpeg[p + 3]]) as usize;
        let Some(body) = jpeg.get(p + 4..p + 2 + len.max(2)) else { break };
        match m {
            0xE1 if body.starts_with(b"Exif\0") => {
                if let Some(tiff) = body.get(6..) {
                    exif(tiff, &mut found);
                }
            }
            0xE1 if body.starts_with(b"http://ns.adobe.com/") => found.xmp = true,
            0xED => found.iptc = true,
            0xE2 if body.starts_with(b"MPF\0") => found.mpf = true,
            0xFE => found.comment = true,
            _ => {}
        }
        p += 2 + len;
    }
    found.list()
}

#[derive(Default)]
struct Found {
    gps: bool,
    gps_other: bool,
    camera: bool,
    serial: bool,
    maker_note: bool,
    owner: bool,
    date: bool,
    unique_id: bool,
    software: bool,
    text: bool,
    thumbnail: bool,
    xmp: bool,
    iptc: bool,
    mpf: bool,
    comment: bool,
}

impl Found {
    fn list(&self) -> Vec<&'static str> {
        [
            (self.gps, "GPS location"),
            (self.gps_other, "GPS data"),
            (self.camera, "camera / lens make and model"),
            (self.serial, "serial numbers"),
            (self.maker_note, "maker notes (may hold the serial number)"),
            (self.owner, "owner / artist / copyright"),
            (self.date, "capture date and time"),
            (self.unique_id, "unique image ID"),
            (self.software, "software"),
            (self.text, "descriptions / comments in EXIF"),
            (self.thumbnail, "an EXIF thumbnail of the original"),
            (self.xmp, "XMP (edit history, location, people…)"),
            (self.iptc, "IPTC / Photoshop data"),
            (self.mpf, "embedded extra images (MPF: preview, depth or HDR map)"),
            (self.comment, "JPEG comments"),
        ]
        .into_iter()
        .filter_map(|(on, name)| on.then_some(name))
        .collect()
    }
}

fn exif(t: &[u8], f: &mut Found) {
    let le = match t.get(..4) {
        Some(b"II*\0") => true,
        Some(b"MM\0*") => false,
        _ => return,
    };
    let u16_at = |o: usize| t.get(o..o + 2).map(|b| if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) });
    let u32_at = |o: usize| {
        t.get(o..o + 4).map(|b| if le { u32::from_le_bytes([b[0], b[1], b[2], b[3]]) } else { u32::from_be_bytes([b[0], b[1], b[2], b[3]]) })
    };
    // (tags, next-IFD offset) of the IFD at `off`.
    let ifd = |off: usize| -> Option<(Vec<(u16, u32)>, u32)> {
        let n = u16_at(off)? as usize;
        let tags = (0..n).filter_map(|i| Some((u16_at(off + 2 + 12 * i)?, u32_at(off + 2 + 12 * i + 8)?))).collect();
        Some((tags, u32_at(off + 2 + 12 * n).unwrap_or(0)))
    };
    let Some(ifd0_at) = u32_at(4) else { return };
    let Some((ifd0, next)) = ifd(ifd0_at as usize) else { return };
    f.thumbnail = next != 0 && next != ifd0_at && ifd(next as usize).is_some_and(|(tags, _)| !tags.is_empty());
    let pointer = |tags: &[(u16, u32)], id: u16| tags.iter().find(|(t, _)| *t == id).map(|&(_, v)| v as usize);
    let exif_ifd = pointer(&ifd0, 0x8769).and_then(ifd).map(|(t, _)| t).unwrap_or_default();
    let gps_ifd = pointer(&ifd0, 0x8825).and_then(ifd).map(|(t, _)| t).unwrap_or_default();
    for &(tag, _) in ifd0.iter().chain(&exif_ifd) {
        match tag {
            0x010F | 0x0110 | 0xA433 | 0xA434 => f.camera = true,
            0xA431 | 0xA435 | 0xC62F => f.serial = true,
            0x927C => f.maker_note = true,
            0x013B | 0x8298 | 0xA430 => f.owner = true,
            0x0132 | 0x9003 | 0x9004 => f.date = true,
            0xA420 => f.unique_id = true,
            0x0131 => f.software = true,
            0x010E | 0x9286 | 0x9C9B..=0x9C9F => f.text = true,
            _ => {}
        }
    }
    // Latitude / longitude (tags 1-4), or other GPS data (altitude, time).
    f.gps = gps_ifd.iter().any(|&(t, _)| (1..=4).contains(&t));
    f.gps_other = !f.gps && !gps_ifd.is_empty();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A big-endian TIFF block: IFD0 with Make + GPS pointer, a GPS IFD with
    /// latitude, and an IFD1 (thumbnail).
    fn tiff() -> Vec<u8> {
        let mut t = b"MM\0*\0\0\0\x08".to_vec();
        // IFD0 at 8: 2 entries, next IFD at 8 + 2 + 24 + 4 = 38.
        t.extend_from_slice(&[0, 2]);
        t.extend_from_slice(&[0x01, 0x0F, 0, 2, 0, 0, 0, 4, b'A', b'B', b'C', 0]);
        t.extend_from_slice(&[0x88, 0x25, 0, 4, 0, 0, 0, 1, 0, 0, 0, 56]);
        t.extend_from_slice(&[0, 0, 0, 38]);
        // IFD1 at 38: 1 entry (Compression), no next.
        t.extend_from_slice(&[0, 1, 0x01, 0x03, 0, 3, 0, 0, 0, 1, 0, 6, 0, 0, 0, 0, 0, 0]);
        // GPS IFD at 56: GPSLatitudeRef.
        t.extend_from_slice(&[0, 1, 0, 1, 0, 2, 0, 0, 0, 2, b'N', 0, 0, 0, 0, 0, 0, 0]);
        t
    }

    fn jpeg_with(segments: &[(u8, Vec<u8>)]) -> Vec<u8> {
        let mut j = vec![0xFF, 0xD8];
        for (m, body) in segments {
            j.extend_from_slice(&[0xFF, *m]);
            j.extend_from_slice(&((body.len() + 2) as u16).to_be_bytes());
            j.extend_from_slice(body);
        }
        j.extend_from_slice(&[0xFF, 0xDA, 0, 2, 0xFF, 0xD9]);
        j
    }

    #[test]
    fn lists_exif_xmp_and_comments() {
        let mut exif = b"Exif\0\0".to_vec();
        exif.extend_from_slice(&tiff());
        let j = jpeg_with(&[
            (0xE1, exif),
            (0xE1, b"http://ns.adobe.com/xap/1.0/\0<x/>".to_vec()),
            (0xFE, b"hello".to_vec()),
        ]);
        assert_eq!(
            identifying_metadata(&j),
            vec![
                "GPS location",
                "camera / lens make and model",
                "an EXIF thumbnail of the original",
                "XMP (edit history, location, people…)",
                "JPEG comments"
            ]
        );
    }

    #[test]
    fn plain_and_malformed_files_list_nothing() {
        assert!(identifying_metadata(&jpeg_with(&[(0xE0, b"JFIF\0\x01\x01\0\0\x01\0\x01\0\0".to_vec())])).is_empty());
        // EXIF whose IFD offsets point past the end, and one pointing at itself.
        let mut bad = b"Exif\0\0MM\0*\xFF\xFF\xFF\x00".to_vec();
        assert!(identifying_metadata(&jpeg_with(&[(0xE1, bad.clone())])).is_empty());
        bad.truncate(6);
        bad.extend_from_slice(b"MM\0*\0\0\0\x08\0\x01\x87\x69\0\x04\0\0\0\x01\0\0\0\x08\0\0\0\x08");
        assert!(identifying_metadata(&jpeg_with(&[(0xE1, bad)])).is_empty());
        assert!(identifying_metadata(b"\xFF\xD8\xFF\xE1\xFF").is_empty());
    }
}
