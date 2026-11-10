// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! The AVIF container (ISO BMFF, MIAF), written and read without dependencies.
//!
//! Written: one AV1 image item, the boxes libavif writes for a plain still image (`ftyp`; `meta` with
//! `hdlr`, `pitm`, `iloc`, `iinf`/`infe`, `iprp` holding `ispe`, `pixi`, `av1C`, `colr` nclx; `mdat`), no
//! metadata. Read: the primary item of any single-image AVIF whose data is one `av01` item (no grids,
//! no alpha handling); anything else is [`PhotoError::UnsupportedAvif`] and the apps show the file as
//! a plain attachment.

use super::PhotoError;

/// Colour description of an image (ISO/IEC 23091-2 code points), from `colr` nclx.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nclx {
    pub primaries: u16,
    pub transfer: u16,
    pub matrix: u16,
    pub full_range: bool,
}

/// What we write: sRGB primaries and transfer, the JPEG (BT.601) matrix, full range.
pub const NCLX_SRGB_BT601_FULL: Nclx = Nclx { primaries: 1, transfer: 13, matrix: 6, full_range: true };

/// The primary image of an AVIF: its size, colour and the AV1 data to decode (the `av1C` config
/// OBUs followed by the item's OBUs).
#[derive(Debug)]
pub struct AvifImage {
    pub width: u32,
    pub height: u32,
    pub nclx: Option<Nclx>,
    pub av1: Vec<u8>,
}

// ---- OBUs ---------------------------------------------------------------------------------------

const OBU_SEQUENCE_HEADER: u8 = 1;
const OBU_TEMPORAL_DELIMITER: u8 = 2;

fn leb128(b: &[u8], mut i: usize) -> Option<(u64, usize)> {
    let mut v = 0u64;
    for k in 0..8 {
        let byte = *b.get(i)?;
        i += 1;
        v |= ((byte & 0x7f) as u64) << (7 * k);
        if byte & 0x80 == 0 {
            return Some((v, i));
        }
    }
    None
}

/// Split a low-overhead AV1 bitstream into (OBU type, whole OBU bytes).
fn obus(b: &[u8]) -> Result<Vec<(u8, &[u8])>, PhotoError> {
    let bad = || PhotoError::UnsupportedAvif("malformed AV1 OBUs");
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
        let start = i;
        let hdr = b[i];
        let ty = (hdr >> 3) & 0x0f;
        let has_ext = hdr & 0x04 != 0;
        let has_size = hdr & 0x02 != 0;
        i += 1 + has_ext as usize;
        if !has_size {
            return Err(bad());
        }
        let (size, j) = leb128(b, i).ok_or_else(bad)?;
        let end = j.checked_add(size as usize).filter(|&e| e <= b.len()).ok_or_else(bad)?;
        out.push((ty, &b[start..end]));
        i = end;
    }
    Ok(out)
}

struct Bits<'a> {
    b: &'a [u8],
    pos: usize,
}

impl Bits<'_> {
    fn f(&mut self, n: u32) -> Option<u32> {
        let mut v = 0u32;
        for _ in 0..n {
            let byte = *self.b.get(self.pos / 8)?;
            v = (v << 1) | ((byte >> (7 - self.pos % 8)) & 1) as u32;
            self.pos += 1;
        }
        Some(v)
    }

    fn uvlc(&mut self) -> Option<u32> {
        let mut zeros = 0;
        while self.f(1)? == 0 {
            zeros += 1;
            if zeros >= 32 {
                return None;
            }
        }
        Some(self.f(zeros)? + ((1u32 << zeros) - 1))
    }
}

/// `seq_profile`, `seq_level_idx[0]` and `seq_tier[0]` of a sequence header OBU (AV1 spec 5.5.1).
fn seq_profile_level_tier(obu: &[u8]) -> Option<(u8, u8, u8)> {
    let has_ext = obu[0] & 0x04 != 0;
    let (_, payload) = leb128(obu, 1 + has_ext as usize)?;
    let mut r = Bits { b: &obu[payload..], pos: 0 };
    let profile = r.f(3)? as u8;
    let _still = r.f(1)?;
    let reduced = r.f(1)?;
    if reduced == 1 {
        return Some((profile, r.f(5)? as u8, 0));
    }
    let timing_info = r.f(1)?;
    let mut decoder_model_info = 0;
    let mut buffer_delay_len = 0;
    if timing_info == 1 {
        r.f(32)?;
        r.f(32)?;
        if r.f(1)? == 1 {
            r.uvlc()?;
        }
        decoder_model_info = r.f(1)?;
        if decoder_model_info == 1 {
            buffer_delay_len = r.f(5)? + 1;
            r.f(32)?;
            r.f(5)?;
            r.f(5)?;
        }
    }
    let initial_display_delay = r.f(1)?;
    let count = r.f(5)? + 1;
    let mut first = None;
    for _ in 0..count {
        r.f(12)?;
        let level = r.f(5)? as u8;
        let tier = if level > 7 { r.f(1)? as u8 } else { 0 };
        if decoder_model_info == 1 && r.f(1)? == 1 {
            r.f(buffer_delay_len)?;
            r.f(buffer_delay_len)?;
            r.f(1)?;
        }
        if initial_display_delay == 1 && r.f(1)? == 1 {
            r.f(4)?;
        }
        first.get_or_insert((level, tier));
    }
    let (level, tier) = first?;
    Some((profile, level, tier))
}

const OBU_FRAME_HEADER: u8 = 3;
const OBU_FRAME: u8 = 6;

/// `av1` up to the end of its first frame (a frame OBU, or a frame header with its tile groups):
/// an image item that carries more frames (someone else's file) has only its first one decoded.
pub fn first_frame(av1: &[u8]) -> Result<&[u8], PhotoError> {
    let mut end = 0;
    let mut seen_frame = false;
    for (ty, obu) in obus(av1)? {
        if ty == OBU_FRAME || ty == OBU_FRAME_HEADER {
            if seen_frame {
                break;
            }
            seen_frame = true;
        }
        end += obu.len();
        if ty == OBU_FRAME {
            break;
        }
    }
    Ok(&av1[..end])
}

// ---- writing ------------------------------------------------------------------------------------

fn bx(ty: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(8 + body.len());
    v.extend_from_slice(&((8 + body.len()) as u32).to_be_bytes());
    v.extend_from_slice(ty);
    v.extend_from_slice(body);
    v
}

fn full(ty: &[u8; 4], version: u8, flags: u32, body: &[u8]) -> Vec<u8> {
    let mut b = Vec::with_capacity(4 + body.len());
    b.push(version);
    b.extend_from_slice(&flags.to_be_bytes()[1..]);
    b.extend_from_slice(body);
    bx(ty, &b)
}

/// Wrap the AV1 output of the encoder (8-bit 4:2:0, `width` x `height`) in an AVIF file. The
/// temporal delimiter is dropped; the sequence header goes into `av1C` and stays in the item data.
pub fn write_avif(av1: &[u8], width: u32, height: u32, nclx: Nclx) -> Result<Vec<u8>, PhotoError> {
    let units = obus(av1)?;
    let seq = units
        .iter()
        .find(|(t, _)| *t == OBU_SEQUENCE_HEADER)
        .map(|(_, o)| *o)
        .ok_or(PhotoError::UnsupportedAvif("no sequence header"))?;
    let (profile, level, tier) =
        seq_profile_level_tier(seq).ok_or(PhotoError::UnsupportedAvif("malformed sequence header"))?;
    let item: Vec<u8> = units
        .iter()
        .filter(|(t, _)| *t != OBU_TEMPORAL_DELIMITER)
        .flat_map(|(_, o)| o.iter().copied())
        .collect();

    let ftyp = bx(b"ftyp", b"avif\0\0\0\0avifmif1miaf");
    let hdlr = full(b"hdlr", 0, 0, b"\0\0\0\0pict\0\0\0\0\0\0\0\0\0\0\0\0\0");
    let pitm = full(b"pitm", 0, 0, &1u16.to_be_bytes());
    let iinf = {
        let infe = full(b"infe", 2, 0, b"\0\x01\0\0av01\0");
        let mut b = 1u16.to_be_bytes().to_vec();
        b.extend_from_slice(&infe);
        full(b"iinf", 0, 0, &b)
    };
    let ispe = {
        let mut b = width.to_be_bytes().to_vec();
        b.extend_from_slice(&height.to_be_bytes());
        full(b"ispe", 0, 0, &b)
    };
    let pixi = full(b"pixi", 0, 0, &[3, 8, 8, 8]);
    let av1c = {
        // 8-bit, 4:2:0 (subsampling x and y), chroma sample position 0, no presentation delay
        let mut b = vec![0x81, (profile << 5) | level, (tier << 7) | 0b0000_1100, 0];
        b.extend_from_slice(seq);
        bx(b"av1C", &b)
    };
    let colr = {
        let mut b = b"nclx".to_vec();
        b.extend_from_slice(&nclx.primaries.to_be_bytes());
        b.extend_from_slice(&nclx.transfer.to_be_bytes());
        b.extend_from_slice(&nclx.matrix.to_be_bytes());
        b.push(if nclx.full_range { 0x80 } else { 0 });
        bx(b"colr", &b)
    };
    let ipco = bx(b"ipco", &[ispe, pixi, av1c, colr].concat());
    // item 1: ispe, pixi, av1C (essential), colr
    let ipma = full(b"ipma", 0, 0, &[0, 0, 0, 1, 0, 1, 4, 0x01, 0x02, 0x83, 0x04]);
    let iprp = bx(b"iprp", &[ipco, ipma].concat());
    let iloc = |offset: u32| {
        let mut b = vec![0x44, 0x00];
        b.extend_from_slice(&1u16.to_be_bytes()); // item count
        b.extend_from_slice(&1u16.to_be_bytes()); // item ID
        b.extend_from_slice(&0u16.to_be_bytes()); // data reference index
        b.extend_from_slice(&1u16.to_be_bytes()); // extent count
        b.extend_from_slice(&offset.to_be_bytes());
        b.extend_from_slice(&(item.len() as u32).to_be_bytes());
        full(b"iloc", 0, 0, &b)
    };
    let meta_of = |offset: u32| full(b"meta", 0, 0, &[hdlr.clone(), pitm.clone(), iloc(offset), iinf.clone(), iprp.clone()].concat());
    let offset = (ftyp.len() + meta_of(0).len() + 8) as u32;
    Ok([ftyp, meta_of(offset), bx(b"mdat", &item)].concat())
}

// ---- reading ------------------------------------------------------------------------------------

/// The boxes in `b`: (type, body).
fn boxes(b: &[u8]) -> Result<Vec<([u8; 4], &[u8])>, PhotoError> {
    let bad = || PhotoError::UnsupportedAvif("malformed box");
    let mut out = Vec::new();
    let mut i = 0;
    while i + 8 <= b.len() {
        let size32 = u32::from_be_bytes(b[i..i + 4].try_into().unwrap()) as usize;
        let ty: [u8; 4] = b[i + 4..i + 8].try_into().unwrap();
        let (hdr, size) = match size32 {
            0 => (8, b.len() - i),
            1 => {
                let s = b.get(i + 8..i + 16).ok_or_else(bad)?;
                (16, usize::try_from(u64::from_be_bytes(s.try_into().unwrap())).map_err(|_| bad())?)
            }
            s => (8, s),
        };
        let end = i.checked_add(size).filter(|&e| e <= b.len() && size >= hdr).ok_or_else(bad)?;
        out.push((ty, &b[i + hdr..end]));
        i = end;
    }
    Ok(out)
}

fn find<'a>(list: &[([u8; 4], &'a [u8])], ty: &[u8; 4]) -> Option<&'a [u8]> {
    list.iter().find(|(t, _)| t == ty).map(|(_, b)| *b)
}

struct Rd<'a> {
    b: &'a [u8],
    i: usize,
}

impl<'a> Rd<'a> {
    fn n(&mut self, bytes: usize) -> Result<u64, PhotoError> {
        let s = self.b.get(self.i..self.i + bytes).ok_or(PhotoError::UnsupportedAvif("truncated box"))?;
        self.i += bytes;
        Ok(s.iter().fold(0u64, |v, &x| (v << 8) | x as u64))
    }
    fn take(&mut self, bytes: usize) -> Result<&'a [u8], PhotoError> {
        let s = self.b.get(self.i..self.i + bytes).ok_or(PhotoError::UnsupportedAvif("truncated box"))?;
        self.i += bytes;
        Ok(s)
    }
}

/// Whether `bytes` look like an AVIF file (an `ftyp` box naming `avif` as a brand).
pub fn is_avif(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || &bytes[4..8] != b"ftyp" {
        return false;
    }
    let size = (u32::from_be_bytes(bytes[0..4].try_into().unwrap()) as usize).min(bytes.len());
    size >= 16 && bytes[8..size].chunks_exact(4).enumerate().any(|(k, c)| k != 1 && c == b"avif")
}

/// The primary image of an AVIF file.
pub fn read_avif(file: &[u8]) -> Result<AvifImage, PhotoError> {
    if !is_avif(file) {
        return Err(PhotoError::UnsupportedAvif("not an AVIF file"));
    }
    let top = boxes(file)?;
    let meta = find(&top, b"meta").ok_or(PhotoError::UnsupportedAvif("no meta box"))?;
    let meta = boxes(meta.get(4..).ok_or(PhotoError::UnsupportedAvif("truncated meta"))?)?;

    let primary = match find(&meta, b"pitm") {
        Some(p) => {
            let mut r = Rd { b: p, i: 0 };
            let v = r.n(1)?;
            r.n(3)?;
            r.n(if v == 0 { 2 } else { 4 })? as u32
        }
        None => 1,
    };

    // the primary item must be an av01 image
    let iinf = find(&meta, b"iinf").ok_or(PhotoError::UnsupportedAvif("no iinf box"))?;
    let mut r = Rd { b: iinf, i: 0 };
    let v = r.n(1)?;
    r.n(3)?;
    r.n(if v == 0 { 2 } else { 4 })?;
    let mut item_type = None;
    for (ty, infe) in boxes(&iinf[r.i..])? {
        if &ty != b"infe" {
            continue;
        }
        let mut e = Rd { b: infe, i: 0 };
        let ver = e.n(1)?;
        e.n(3)?;
        if ver < 2 {
            continue;
        }
        let id = e.n(if ver == 2 { 2 } else { 4 })? as u32;
        e.n(2)?;
        let t = e.take(4)?;
        if id == primary {
            item_type = Some(<[u8; 4]>::try_from(t).unwrap());
        }
    }
    if item_type != Some(*b"av01") {
        return Err(PhotoError::UnsupportedAvif("the primary item is not a plain AV1 image"));
    }

    // properties of the primary item
    let iprp = boxes(find(&meta, b"iprp").ok_or(PhotoError::UnsupportedAvif("no iprp box"))?)?;
    let ipco = boxes(find(&iprp, b"ipco").ok_or(PhotoError::UnsupportedAvif("no ipco box"))?)?;
    let ipma = find(&iprp, b"ipma").ok_or(PhotoError::UnsupportedAvif("no ipma box"))?;
    let mut r = Rd { b: ipma, i: 0 };
    let ver = r.n(1)?;
    let flags = r.n(3)?;
    let mut props: Vec<usize> = Vec::new();
    for _ in 0..r.n(4)? {
        let id = r.n(if ver < 1 { 2 } else { 4 })? as u32;
        let n = r.n(1)?;
        for _ in 0..n {
            let a = r.n(if flags & 1 != 0 { 2 } else { 1 })?;
            let index = if flags & 1 != 0 { a & 0x7fff } else { a & 0x7f } as usize;
            if id == primary && index > 0 {
                props.push(index - 1);
            }
        }
    }
    let (mut width, mut height, mut nclx, mut config) = (0, 0, None, Vec::new());
    for p in props {
        let Some((ty, body)) = ipco.get(p) else { continue };
        match ty {
            b"ispe" => {
                let mut e = Rd { b: body, i: 4 };
                width = e.n(4)? as u32;
                height = e.n(4)? as u32;
            }
            b"av1C" => config = body.get(4..).unwrap_or_default().to_vec(),
            b"colr" if body.len() >= 11 && &body[0..4] == b"nclx" => {
                nclx = Some(Nclx {
                    primaries: u16::from_be_bytes([body[4], body[5]]),
                    transfer: u16::from_be_bytes([body[6], body[7]]),
                    matrix: u16::from_be_bytes([body[8], body[9]]),
                    full_range: body[10] & 0x80 != 0,
                });
            }
            b"auxC" => return Err(PhotoError::UnsupportedAvif("auxiliary image")),
            _ => {}
        }
    }

    // the item's data
    let iloc = find(&meta, b"iloc").ok_or(PhotoError::UnsupportedAvif("no iloc box"))?;
    let mut r = Rd { b: iloc, i: 0 };
    let ver = r.n(1)?;
    r.n(3)?;
    let sizes = r.n(1)?;
    let (off_size, len_size) = ((sizes >> 4) as usize, (sizes & 0xf) as usize);
    let sizes = r.n(1)?;
    let base_size = (sizes >> 4) as usize;
    let index_size = if ver >= 1 { (sizes & 0xf) as usize } else { 0 };
    // the only field sizes ISO/IEC 14496-12 allows; a length size of 0 ("to the end") is only taken
    // for a single extent, so no item can copy the file more than once
    if [off_size, len_size, base_size, index_size].iter().any(|&n| !matches!(n, 0 | 4 | 8)) {
        return Err(PhotoError::UnsupportedAvif("bad iloc field size"));
    }
    let count = r.n(if ver < 2 { 2 } else { 4 })?;
    let idat = find(&meta, b"idat");
    let mut data = None;
    for _ in 0..count {
        let id = r.n(if ver < 2 { 2 } else { 4 })? as u32;
        let method = if ver >= 1 { r.n(2)? & 0xf } else { 0 };
        r.n(2)?;
        let base = r.n(base_size)?;
        let extents = r.n(2)?;
        if len_size == 0 && extents > 1 {
            return Err(PhotoError::UnsupportedAvif("bad iloc extents"));
        }
        let src = match method {
            0 => file,
            1 => idat.ok_or(PhotoError::UnsupportedAvif("no idat box"))?,
            _ => return Err(PhotoError::UnsupportedAvif("item data in another item")),
        };
        // only the primary item's data is copied, and never more than its source holds
        let mut bytes = Vec::new();
        for _ in 0..extents {
            r.n(index_size)?;
            let off = r.n(off_size)?;
            let len = r.n(len_size)?;
            if id != primary {
                continue;
            }
            let range = || -> Option<core::ops::Range<usize>> {
                let start = usize::try_from(base.checked_add(off)?).ok()?;
                let end = if len == 0 { src.len() } else { start.checked_add(usize::try_from(len).ok()?)? };
                (start <= end && end <= src.len()).then_some(start..end)
            };
            let part = range().ok_or(PhotoError::UnsupportedAvif("item data out of range"))?;
            if bytes.len() + part.len() > src.len() {
                return Err(PhotoError::UnsupportedAvif("item data larger than its source"));
            }
            bytes.extend_from_slice(&src[part]);
        }
        if id == primary {
            if data.is_some() {
                return Err(PhotoError::UnsupportedAvif("the primary item is located twice"));
            }
            data = Some(bytes);
        }
    }
    let data = data.ok_or(PhotoError::UnsupportedAvif("no data for the primary item"))?;
    if width == 0 || height == 0 {
        return Err(PhotoError::UnsupportedAvif("no image size"));
    }
    Ok(AvifImage { width, height, nclx, av1: [config, data].concat() })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny well-formed AV1 stream: temporal delimiter, a reduced still-picture sequence header
    /// (profile 0, level 0), a frame OBU with a dummy payload. Enough for the container code.
    fn av1_stub() -> Vec<u8> {
        let mut v = vec![0x12, 0x00]; // temporal delimiter
        v.extend_from_slice(&[0x0a, 0x03, 0b0001_1000, 0x00, 0x00]); // sequence header: still + reduced, level 0
        v.extend_from_slice(&[0x32, 0x02, 0xaa, 0xbb]); // frame
        v
    }

    #[test]
    fn writes_and_reads_back() {
        let file = write_avif(&av1_stub(), 64, 48, NCLX_SRGB_BT601_FULL).unwrap();
        assert!(is_avif(&file));
        let img = read_avif(&file).unwrap();
        assert_eq!((img.width, img.height), (64, 48));
        assert_eq!(img.nclx, Some(NCLX_SRGB_BT601_FULL));
        // config OBU (sequence header) + item data (sequence header + frame, no temporal delimiter)
        assert_eq!(img.av1.len(), 5 + 5 + 4);
    }

    #[test]
    fn every_truncation_is_an_error_not_a_panic() {
        let file = write_avif(&av1_stub(), 64, 48, NCLX_SRGB_BT601_FULL).unwrap();
        for n in 0..file.len() {
            let _ = read_avif(&file[..n]);
        }
    }

    #[test]
    fn huge_box_size_is_refused() {
        let mut file = write_avif(&av1_stub(), 64, 48, NCLX_SRGB_BT601_FULL).unwrap();
        // the box after ftyp (meta) gets a 64-bit size near usize::MAX
        let at = u32::from_be_bytes(file[0..4].try_into().unwrap()) as usize;
        file[at..at + 4].copy_from_slice(&1u32.to_be_bytes());
        file.splice(at + 8..at + 8, 0xffff_ffff_ffff_fff0u64.to_be_bytes());
        assert!(read_avif(&file).is_err());
    }

    #[test]
    fn iloc_cannot_multiply_the_file() {
        let file = write_avif(&av1_stub(), 64, 48, NCLX_SRGB_BT601_FULL).unwrap();
        let at = file.windows(4).position(|w| w == b"iloc").unwrap() - 4;
        // iloc v0 with length size 0 ("to the end") and 60000 extents
        let mut body = vec![0, 0, 0, 0, 0x40, 0x00, 0, 1, 0, 1, 0, 0];
        body.extend_from_slice(&60000u16.to_be_bytes());
        for _ in 0..60000 {
            body.extend_from_slice(&0u32.to_be_bytes());
        }
        let mut iloc = ((8 + body.len()) as u32).to_be_bytes().to_vec();
        iloc.extend_from_slice(b"iloc");
        iloc.extend_from_slice(&body);
        let old = u32::from_be_bytes(file[at..at + 4].try_into().unwrap()) as usize;
        let mut hostile = file[..at].to_vec();
        hostile.extend_from_slice(&iloc);
        hostile.extend_from_slice(&file[at + old..]);
        // fix the meta box size
        let meta = u32::from_be_bytes(hostile[0..4].try_into().unwrap()) as usize;
        let meta_size = u32::from_be_bytes(hostile[meta..meta + 4].try_into().unwrap()) as usize + iloc.len() - old;
        hostile[meta..meta + 4].copy_from_slice(&(meta_size as u32).to_be_bytes());
        assert!(read_avif(&hostile).is_err());
    }

    #[test]
    fn first_frame_stops_after_one_frame() {
        let mut av1 = av1_stub();
        let one = av1.len();
        av1.extend_from_slice(&[0x32, 0x02, 0xcc, 0xdd]); // a second frame
        assert_eq!(first_frame(&av1).unwrap().len(), one);
    }
}
