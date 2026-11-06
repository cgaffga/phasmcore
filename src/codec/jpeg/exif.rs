// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! EXIF of a re-written cover (fix list 1.2, decided 2026-11-02: with the
//! magic wand off, a cover phasm writes itself — Deep Cover from HEIC or a
//! camera capture, the resolution or quality chip — keeps the source's
//! metadata, *cleaned*).
//!
//! Before, iOS prepended a 1x1 ImageIO EXIF to the cover's own JFIF
//! (duplicate JFIF/EXIF, `PixelXDimension = 1`) and Android copied a few
//! tags with `ExifInterface`: different, and inconsistent with the image.
//! Now every frontend hands core the original file and the cover:
//!
//! 1. [`source_exif`] finds the raw EXIF (TIFF) block of the source: JPEG
//!    APP1, HEIF/HEIC `Exif` item, PNG `eXIf`, WebP `EXIF`.
//! 2. [`exif_for_cover`] rewrites it for the cover. Everything is kept
//!    (GPS, maker notes, serial numbers, owner) except what would contradict
//!    the cover: the thumbnail IFD (IFD1 and its image) is dropped,
//!    `Orientation` = 1 (the pixels are upright), `PixelXDimension` /
//!    `PixelYDimension` (and IFD0 `ImageWidth` / `ImageLength`, if present)
//!    = the cover's size, `ColorSpace` = sRGB (phasm's covers are sRGB), and
//!    tags pointing at image data that is not copied (strip / tile offsets,
//!    SubIFDs) are removed.
//!    The maker note stays at its original offset when the rewritten
//!    directories fit in front of it: many cameras' maker notes hold offsets
//!    relative to the TIFF header and break when moved. Otherwise it moves.
//! 3. [`with_exif`] puts the block into the cover as the first segment
//!    (APP1 `Exif`, like a camera file) and drops the writer's JFIF APP0
//!    (a JFIF + Exif mix is not a valid Exif file) and any older EXIF.
//!
//! [`cover_with_source_metadata`] does all three. The DCT coefficients are
//! untouched, so Deep Cover's exactness check and the encode are unaffected.
//! Neither file is decoded: a marker scan finds the APP1 and splices the
//! segments (fix list R3.13).

use super::error::{JpegError, Result};

const TAG_IMAGE_WIDTH: u16 = 0x0100;
const TAG_IMAGE_LENGTH: u16 = 0x0101;
const TAG_STRIP_OFFSETS: u16 = 0x0111;
const TAG_ORIENTATION: u16 = 0x0112;
const TAG_STRIP_BYTE_COUNTS: u16 = 0x0117;
const TAG_TILE_OFFSETS: u16 = 0x0144;
const TAG_TILE_BYTE_COUNTS: u16 = 0x0145;
const TAG_SUB_IFDS: u16 = 0x014A;
const TAG_JPEG_IF: u16 = 0x0201;
const TAG_JPEG_IF_LEN: u16 = 0x0202;
const TAG_EXIF_IFD: u16 = 0x8769;
const TAG_GPS_IFD: u16 = 0x8825;
const TAG_INTEROP_IFD: u16 = 0xA005;
const TAG_MAKER_NOTE: u16 = 0x927C;
const TAG_COLOR_SPACE: u16 = 0xA001;
const TAG_PIXEL_X: u16 = 0xA002;
const TAG_PIXEL_Y: u16 = 0xA003;
const TAG_YCBCR_POSITIONING: u16 = 0x0213;
const TAG_EXIF_VERSION: u16 = 0x9000;
const TAG_COMPONENTS_CONFIGURATION: u16 = 0x9101;
const TAG_FLASHPIX_VERSION: u16 = 0xA000;
const TAG_GPS_VERSION_ID: u16 = 0x0000;

/// Largest EXIF block an APP1 segment can hold (64 KiB − length − "Exif\0\0").
const MAX_TIFF_BYTES: usize = 65_535 - 2 - 6;

/// Bytes per value of a TIFF field type (0 = unknown type).
fn type_size(t: u16) -> usize {
    match t {
        1 | 2 | 6 | 7 => 1,
        3 | 8 => 2,
        4 | 9 | 11 | 13 => 4,
        5 | 10 | 12 => 8,
        _ => 0,
    }
}

#[derive(Clone, Copy)]
struct Order(bool); // little-endian?

impl Order {
    fn u16(self, b: &[u8], o: usize) -> Option<u16> {
        let s = b.get(o..o + 2)?;
        Some(if self.0 { u16::from_le_bytes([s[0], s[1]]) } else { u16::from_be_bytes([s[0], s[1]]) })
    }
    fn u32(self, b: &[u8], o: usize) -> Option<u32> {
        let s = b.get(o..o + 4)?;
        let a = [s[0], s[1], s[2], s[3]];
        Some(if self.0 { u32::from_le_bytes(a) } else { u32::from_be_bytes(a) })
    }
    fn put16(self, v: u16) -> [u8; 2] {
        if self.0 { v.to_le_bytes() } else { v.to_be_bytes() }
    }
    fn put32(self, v: u32) -> [u8; 4] {
        if self.0 { v.to_le_bytes() } else { v.to_be_bytes() }
    }
}

/// One directory entry with its value bytes (in the block's byte order).
#[derive(Clone)]
struct Entry {
    tag: u16,
    typ: u16,
    count: u32,
    value: Vec<u8>,
    /// Where the value sat in the source block (out-of-line values only).
    src_offset: Option<usize>,
}

/// A parsed directory; `children` are the sub-directories its pointer tags
/// lead to (Exif, GPS, Interop), in tag order.
#[derive(Clone)]
struct Ifd {
    entries: Vec<Entry>,
    children: Vec<(u16, Ifd)>,
}

/// Directories one block may hold at most (IFD0, Exif, GPS, Interop: 4 in
/// a camera file).
const MAX_IFDS: usize = 16;
/// Directory entries one block may hold at most, over all directories.
const MAX_ENTRIES: usize = 4096;
/// Value bytes copied at most (a block's values fit in it; overlapping
/// out-of-line values must not multiply it).
const MAX_VALUE_BYTES: usize = 1 << 21;

/// Bounds of one block's walk (fix list R3.3: a directory whose entries all
/// point back at itself was parsed ~N³ times; 26 s and 783 MB at N = 60).
/// Every directory offset is entered once, each pointer tag is followed once
/// per directory, and the totals are capped; a block over a cap is dropped
/// as unreadable.
#[derive(Default)]
struct Walk {
    visited: Vec<usize>,
    entries: usize,
    value_bytes: usize,
    /// A cap was hit: the block is not used.
    over: bool,
}

fn parse_ifd(t: &[u8], ord: Order, off: usize, depth: usize, walk: &mut Walk) -> Option<Ifd> {
    if depth > 3 || walk.visited.contains(&off) {
        return None;
    }
    if walk.visited.len() >= MAX_IFDS {
        walk.over = true;
        return None;
    }
    walk.visited.push(off);
    let n = ord.u16(t, off)? as usize;
    walk.entries += n;
    if walk.entries > MAX_ENTRIES {
        walk.over = true;
        return None;
    }
    let mut entries = Vec::with_capacity(n);
    let mut children: Vec<(u16, Ifd)> = Vec::new();
    let mut followed: Vec<u16> = Vec::new();
    for i in 0..n {
        let e = off.checked_add(2 + 12 * i)?;
        let (tag, typ, count) = (ord.u16(t, e)?, ord.u16(t, e + 2)?, ord.u32(t, e + 4)?);
        if type_size(typ) == 0 {
            continue; // unknown type: cannot be copied safely
        }
        let size = type_size(typ).checked_mul(count as usize)?;
        if matches!(tag, TAG_EXIF_IFD | TAG_GPS_IFD | TAG_INTEROP_IFD) && count == 1 && matches!(typ, 4 | 13) {
            if !followed.contains(&tag) {
                followed.push(tag);
                let sub = ord.u32(t, e + 8)? as usize;
                if let Some(ifd) = parse_ifd(t, ord, sub, depth + 1, walk) {
                    children.push((tag, ifd));
                }
            }
            continue;
        }
        let (value, src_offset) = if size <= 4 {
            (t.get(e + 8..e + 8 + size)?.to_vec(), None)
        } else {
            let o = ord.u32(t, e + 8)? as usize;
            match t.get(o..o.checked_add(size)?) {
                Some(v) => {
                    walk.value_bytes += size;
                    if walk.value_bytes > MAX_VALUE_BYTES {
                        walk.over = true;
                        return None;
                    }
                    (v.to_vec(), Some(o))
                }
                None => continue, // dangling value: drop the entry
            }
        };
        entries.push(Entry { tag, typ, count, value, src_offset });
    }
    Some(Ifd { entries, children })
}

impl Ifd {
    fn set_scalar(&mut self, ord: Order, tag: u16, v: u32, only_if_present: bool) {
        let (typ, value) = if v <= u16::MAX as u32 && self.entries.iter().find(|e| e.tag == tag).is_some_and(|e| e.typ == 3) {
            (3, ord.put16(v as u16).to_vec())
        } else {
            (4, ord.put32(v).to_vec())
        };
        if let Some(e) = self.entries.iter_mut().find(|e| e.tag == tag) {
            *e = Entry { tag, typ, count: 1, value, src_offset: None };
        } else if !only_if_present {
            self.entries.push(Entry { tag, typ, count: 1, value, src_offset: None });
        }
    }

    /// Set `tag` to exactly this field, replacing any existing entry.
    fn set(&mut self, tag: u16, typ: u16, count: u32, value: Vec<u8>) {
        self.entries.retain(|e| e.tag != tag);
        self.entries.push(Entry { tag, typ, count, value, src_offset: None });
    }

    /// Add `tag` when the directory lacks it.
    fn ensure(&mut self, tag: u16, typ: u16, count: u32, value: Vec<u8>) {
        if !self.entries.iter().any(|e| e.tag == tag) {
            self.entries.push(Entry { tag, typ, count, value, src_offset: None });
        }
    }

    fn child_mut(&mut self, tag: u16) -> Option<&mut Ifd> {
        self.children.iter_mut().find(|(t, _)| *t == tag).map(|(_, c)| c)
    }

    /// Entries + pointer slots, in ascending tag order (TIFF requires it).
    fn sorted_tags(&self) -> Vec<u16> {
        let mut tags: Vec<u16> = self.entries.iter().map(|e| e.tag).chain(self.children.iter().map(|(t, _)| *t)).collect();
        tags.sort_unstable();
        tags
    }
}

/// Size of a directory's table: count, 12-byte entries, next-IFD offset.
fn table_bytes(ifd: &Ifd) -> usize {
    2 + 12 * (ifd.entries.len() + ifd.children.len()) + 4
}

/// The directories in writing order (depth first): `(directory, parent
/// index and pointer tag)`.
fn flatten<'a>(ifd: &'a Ifd, parent: Option<(usize, u16)>, out: &mut Vec<(&'a Ifd, Option<(usize, u16)>)>) {
    let me = out.len();
    out.push((ifd, parent));
    for (tag, child) in &ifd.children {
        flatten(child, Some((me, *tag)), out);
    }
}

/// Byte size of the directory tables of the block (they all precede the
/// values).
fn tables_end(ifd0: &Ifd) -> usize {
    let mut all = Vec::new();
    flatten(ifd0, None, &mut all);
    8 + all.iter().map(|(d, _)| table_bytes(d)).sum::<usize>()
}

/// Serialize the block (header included). The tables come first; with
/// `maker_at` the maker note's value is written exactly there, other values
/// fill the room in front of it and continue after it.
fn serialize(ifd0: &Ifd, ord: Order, maker_at: Option<usize>) -> Vec<u8> {
    let mut all = Vec::new();
    flatten(ifd0, None, &mut all);
    let mut table_at = Vec::with_capacity(all.len());
    let mut cursor = 8;
    for (d, _) in &all {
        table_at.push(cursor);
        cursor += table_bytes(d);
    }
    let maker_len = |d: &Ifd| d.entries.iter().find(|e| e.tag == TAG_MAKER_NOTE).map_or(0, |e| e.value.len());
    let maker_end = maker_at.map(|at| at + all.iter().map(|(d, _)| maker_len(d)).max().unwrap_or(0));
    let mut after = maker_end.map(|e| e.next_multiple_of(2));
    // (directory, entry) -> value offset
    let mut placed: Vec<Vec<Option<usize>>> = all.iter().map(|(d, _)| vec![None; d.entries.len()]).collect();
    for (i, (d, _)) in all.iter().enumerate() {
        for (j, e) in d.entries.iter().enumerate() {
            if e.value.len() <= 4 {
                continue;
            }
            let size = e.value.len().next_multiple_of(2);
            placed[i][j] = Some(match maker_at {
                Some(at) if e.tag == TAG_MAKER_NOTE => at,
                Some(at) if cursor + size > at => {
                    let o = after.unwrap();
                    after = Some(o + size);
                    o
                }
                _ => {
                    let o = cursor;
                    cursor += size;
                    o
                }
            });
        }
    }
    let end = cursor.max(after.unwrap_or(0)).max(maker_end.unwrap_or(0));
    let mut out = vec![0u8; end];
    out[..4].copy_from_slice(if ord.0 { b"II*\0" } else { b"MM\0*" });
    out[4..8].copy_from_slice(&ord.put32(8));
    for (i, (d, _)) in all.iter().enumerate() {
        let tags = d.sorted_tags();
        let t = table_at[i];
        out[t..t + 2].copy_from_slice(&ord.put16(tags.len() as u16));
        for (k, tag) in tags.iter().enumerate() {
            let e_off = t + 2 + 12 * k;
            out[e_off..e_off + 2].copy_from_slice(&ord.put16(*tag));
            if let Some(j) = d.entries.iter().position(|e| e.tag == *tag) {
                let e = &d.entries[j];
                out[e_off + 2..e_off + 4].copy_from_slice(&ord.put16(e.typ));
                out[e_off + 4..e_off + 8].copy_from_slice(&ord.put32(e.count));
                match placed[i][j] {
                    None => out[e_off + 8..e_off + 8 + e.value.len()].copy_from_slice(&e.value),
                    Some(o) => {
                        out[e_off + 8..e_off + 12].copy_from_slice(&ord.put32(o as u32));
                        out[o..o + e.value.len()].copy_from_slice(&e.value);
                    }
                }
            } else {
                // A pointer to a sub-directory.
                let child = all.iter().position(|(_, p)| *p == Some((i, *tag))).unwrap();
                out[e_off + 2..e_off + 4].copy_from_slice(&ord.put16(4));
                out[e_off + 4..e_off + 8].copy_from_slice(&ord.put32(1));
                out[e_off + 8..e_off + 12].copy_from_slice(&ord.put32(table_at[child] as u32));
            }
        }
        // Next-IFD offset stays 0: no IFD1 (thumbnail).
    }
    out
}

/// The source's EXIF block as raw TIFF bytes (starting `II*\0` / `MM\0*`):
/// JPEG APP1 `Exif`, HEIF/HEIC `Exif` item, PNG `eXIf`, WebP `EXIF`.
///
/// Every walk is bounds-checked with checked arithmetic (fix list R3.6 /
/// R3.20: WASM builds have no overflow checks, so a wrapped offset would
/// loop forever; the apps build with `panic = "abort"`, so an out-of-range
/// index would kill them): a malformed file gives `None`, never a panic or
/// a hang.
pub fn source_exif(file: &[u8]) -> Option<Vec<u8>> {
    let tiff = |b: &[u8]| -> Option<Vec<u8>> {
        let b = b.strip_prefix(b"Exif\0\0").unwrap_or(b);
        (b.starts_with(b"II*\0") || b.starts_with(b"MM\0*")).then(|| b.to_vec())
    };
    if file.starts_with(&[0xFF, 0xD8]) {
        let (segs, _) = jpeg_header_segments(file)?;
        return segs
            .iter()
            .map(|&(m, s, e)| (m, &file[s + 4..e]))
            .find(|(m, d)| *m == 0xE1 && d.starts_with(b"Exif\0\0"))
            .and_then(|(_, d)| tiff(d));
    }
    if file.starts_with(b"\x89PNG\r\n\x1a\n") {
        let mut o = 8usize;
        while o.checked_add(8)? <= file.len() {
            let len = u32::from_be_bytes(file.get(o..o + 4)?.try_into().ok()?) as usize;
            let kind = file.get(o + 4..o + 8)?;
            let data = file.get(o + 8..(o + 8).checked_add(len)?)?;
            if kind == b"eXIf" {
                return tiff(data);
            }
            if kind == b"IDAT" || kind == b"IEND" {
                break;
            }
            o = o.checked_add(12)?.checked_add(len)?;
        }
        return None;
    }
    if file.starts_with(b"RIFF") && file.get(8..12) == Some(b"WEBP") {
        let mut o = 12usize;
        while o.checked_add(8)? <= file.len() {
            let len = u32::from_le_bytes(file.get(o + 4..o + 8)?.try_into().ok()?) as usize;
            if file.get(o..o + 4)? == b"EXIF" {
                return tiff(file.get(o + 8..(o + 8).checked_add(len)?)?);
            }
            o = o.checked_add(8)?.checked_add(len.checked_next_multiple_of(2)?)?;
        }
        return None;
    }
    heif_exif(file).and_then(|b| tiff(&b))
}

/// ISOBMFF boxes in `b[start..end]`: (type, payload start, payload end).
fn boxes(b: &[u8], start: usize, end: usize) -> Vec<([u8; 4], usize, usize)> {
    let mut out = Vec::new();
    let end = end.min(b.len());
    let mut o = start;
    while o.saturating_add(8) <= end {
        let size = u32::from_be_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
        let kind = [b[o + 4], b[o + 5], b[o + 6], b[o + 7]];
        let (hdr, size) = match size {
            1 if o + 16 <= end => {
                let large = u64::from_be_bytes(b[o + 8..o + 16].try_into().unwrap());
                match usize::try_from(large) {
                    Ok(s) => (16, s),
                    Err(_) => break,
                }
            }
            0 => (8, end - o),
            s => (8, s),
        };
        if size < hdr || o.saturating_add(size) > end {
            break;
        }
        out.push((kind, o + hdr, o + size));
        o += size;
    }
    out
}

/// Largest `Exif` item read from a HEIF file (a real one is a few KB; the
/// cover's APP1 holds at most 64 KB).
const MAX_HEIF_EXIF_BYTES: usize = 1 << 20;
/// `iloc` extents examined at most (zero-size offset / length fields read
/// nothing, so a count alone must bound the walk).
const MAX_ILOC_EXTENTS: u64 = 1 << 16;

/// The `Exif` item's payload of a HEIF/HEIC file (after its 4-byte TIFF
/// header offset), from `meta` → `iinf` / `iloc` (construction method 0).
/// Every read is bounds-checked: truncated boxes give `None`.
fn heif_exif(b: &[u8]) -> Option<Vec<u8>> {
    let top = boxes(b, 0, b.len());
    if top.first().map(|t| &t.0) != Some(b"ftyp") {
        return None;
    }
    let &(_, ms, me) = top.iter().find(|t| &t.0 == b"meta")?;
    let inner = boxes(b, ms.checked_add(4)?, me); // meta is a full box
    let &(_, is, ie) = inner.iter().find(|t| &t.0 == b"iinf")?;
    let iv = *b.get(is)?;
    let count_len = if iv == 0 { 2 } else { 4 };
    let mut exif_id = None;
    for (kind, s, _) in boxes(b, is.checked_add(4 + count_len)?, ie) {
        if &kind != b"infe" {
            continue;
        }
        let v = *b.get(s)?;
        if v < 2 {
            continue;
        }
        let (id, p) = if v == 2 {
            (u16::from_be_bytes(b.get(s + 4..s + 6)?.try_into().ok()?) as u32, s + 6)
        } else {
            (u32::from_be_bytes(b.get(s + 4..s + 8)?.try_into().ok()?), s + 8)
        };
        if b.get(p + 2..p + 6) == Some(b"Exif") {
            exif_id = Some(id);
            break;
        }
    }
    let exif_id = exif_id?;
    let &(_, ls, _) = inner.iter().find(|t| &t.0 == b"iloc")?;
    let v = *b.get(ls)?;
    let sizes = *b.get(ls + 4)?;
    let (off_size, len_size) = ((sizes >> 4) as usize, (sizes & 15) as usize);
    let sizes2 = *b.get(ls + 5)?;
    let (base_size, idx_size) = ((sizes2 >> 4) as usize, if v >= 1 { (sizes2 & 15) as usize } else { 0 });
    if off_size > 8 || len_size > 8 || base_size > 8 || idx_size > 8 {
        return None;
    }
    let read = |o: usize, n: usize| -> Option<u64> {
        let s = b.get(o..o.checked_add(n)?)?;
        Some(s.iter().fold(0u64, |a, &x| (a << 8) | x as u64))
    };
    let mut p = ls + 6;
    let mut extents_seen = 0u64;
    let count = if v < 2 { let c = read(p, 2)?; p += 2; c } else { let c = read(p, 4)?; p += 4; c };
    for _ in 0..count {
        let id = if v < 2 { let i = read(p, 2)?; p += 2; i } else { let i = read(p, 4)?; p += 4; i } as u32;
        let method = if v >= 1 { let m = read(p, 2)? & 15; p += 2; m } else { 0 };
        p += 2; // data_reference_index
        let base = read(p, base_size)?;
        p += base_size;
        let extents = read(p, 2)?;
        p += 2;
        extents_seen += extents;
        if extents_seen > MAX_ILOC_EXTENTS {
            return None;
        }
        let mut data = Vec::new();
        for _ in 0..extents {
            p += idx_size;
            let off = read(p, off_size)?;
            p += off_size;
            let len = read(p, len_size)?;
            p += len_size;
            if id == exif_id && method == 0 {
                let s = usize::try_from(base.checked_add(off)?).ok()?;
                let len = usize::try_from(len).ok()?;
                if data.len().checked_add(len)? > MAX_HEIF_EXIF_BYTES {
                    return None;
                }
                data.extend_from_slice(b.get(s..s.checked_add(len)?)?);
            }
        }
        if id == exif_id {
            if method != 0 || data.len() < 4 {
                return None;
            }
            let skip = u32::from_be_bytes(data[..4].try_into().ok()?) as usize;
            return data.get(skip.checked_add(4)?..).map(|d| d.to_vec());
        }
    }
    None
}

/// Rewrite the EXIF block `tiff` for a `width` x `height` sRGB cover (see the
/// module docs). `None` if it cannot be parsed.
pub fn exif_for_cover(tiff: &[u8], width: u32, height: u32) -> Option<Vec<u8>> {
    let ord = match tiff.get(..4)? {
        b"II*\0" => Order(true),
        b"MM\0*" => Order(false),
        _ => return None,
    };
    let ifd0_off = ord.u32(tiff, 4)? as usize;
    let mut walk = Walk::default();
    let parsed = parse_ifd(tiff, ord, ifd0_off, 0, &mut walk)?;
    if walk.over {
        return None;
    }
    let maker_at = parsed
        .children
        .iter()
        .find(|(t, _)| *t == TAG_EXIF_IFD)
        .and_then(|(_, e)| e.entries.iter().find(|e| e.tag == TAG_MAKER_NOTE && e.value.len() > 4))
        .and_then(|m| m.src_offset);
    // The rewritten directory tables fit in front of the maker note's old
    // place (values can go after it).
    let fits = |ifd: &Ifd| maker_at.is_some_and(|at| tables_end(ifd) <= at && at % 2 == 0);
    // With the tags a JPEG's EXIF should carry added; without them when only
    // that keeps the maker note in place (its offsets may be absolute).
    let mut ifd0 = prepare(&parsed, ord, width, height, true);
    if maker_at.is_some() && !fits(&ifd0) {
        let lean = prepare(&parsed, ord, width, height, false);
        if fits(&lean) {
            ifd0 = lean;
        }
    }
    let mut out = serialize(&ifd0, ord, if fits(&ifd0) { maker_at } else { None });
    if out.len() > MAX_TIFF_BYTES {
        // Too large for one APP1 (a huge maker note): drop the maker note.
        if let Some(exif) = ifd0.child_mut(TAG_EXIF_IFD) {
            exif.entries.retain(|e| e.tag != TAG_MAKER_NOTE);
        }
        out = serialize(&ifd0, ord, None);
        if out.len() > MAX_TIFF_BYTES {
            return None;
        }
    }
    Some(out)
}

/// The edits of [`exif_for_cover`] on a copy of the parsed block. `add`:
/// also add what is missing (dimensions, colour space, the tags a JPEG's EXIF
/// must carry per Exif 2.32 that HEIF sources and some cameras omit);
/// without it only tags already present change.
fn prepare(parsed: &Ifd, ord: Order, width: u32, height: u32, add: bool) -> Ifd {
    // Data this block no longer carries.
    const DROPPED: [u16; 7] = [
        TAG_STRIP_OFFSETS, TAG_STRIP_BYTE_COUNTS, TAG_TILE_OFFSETS, TAG_TILE_BYTE_COUNTS,
        TAG_SUB_IFDS, TAG_JPEG_IF, TAG_JPEG_IF_LEN,
    ];
    let short = |v: u16| ord.put16(v).to_vec();
    let mut ifd0 = parsed.clone();
    ifd0.entries.retain(|e| !DROPPED.contains(&e.tag));
    if add || ifd0.entries.iter().any(|e| e.tag == TAG_ORIENTATION) {
        ifd0.set(TAG_ORIENTATION, 3, 1, short(1));
    }
    ifd0.set_scalar(ord, TAG_IMAGE_WIDTH, width, true);
    ifd0.set_scalar(ord, TAG_IMAGE_LENGTH, height, true);
    if add {
        ifd0.ensure(TAG_YCBCR_POSITIONING, 3, 1, short(1));
        if ifd0.child_mut(TAG_EXIF_IFD).is_none() {
            ifd0.children.push((TAG_EXIF_IFD, Ifd { entries: Vec::new(), children: Vec::new() }));
        }
        if let Some(gps) = ifd0.child_mut(TAG_GPS_IFD) {
            // 2.3: what phones write (GPSHPositioningError needs it).
            gps.ensure(TAG_GPS_VERSION_ID, 1, 4, vec![2, 3, 0, 0]);
        }
    }
    if let Some(exif) = ifd0.child_mut(TAG_EXIF_IFD) {
        exif.entries.retain(|e| !DROPPED.contains(&e.tag));
        exif.set_scalar(ord, TAG_PIXEL_X, width, !add);
        exif.set_scalar(ord, TAG_PIXEL_Y, height, !add);
        if add || exif.entries.iter().any(|e| e.tag == TAG_COLOR_SPACE) {
            exif.set(TAG_COLOR_SPACE, 3, 1, short(1));
        }
        if add {
            exif.ensure(TAG_EXIF_VERSION, 7, 4, b"0232".to_vec());
            exif.ensure(TAG_COMPONENTS_CONFIGURATION, 7, 4, vec![1, 2, 3, 0]);
            exif.ensure(TAG_FLASHPIX_VERSION, 7, 4, b"0100".to_vec());
        }
    }
    ifd0
}

/// A JPEG's segments before its first scan, found by a marker scan alone
/// (no table parse, no entropy decode: R3.13, a full parse of a 48 MP
/// source or cover only to read or replace its APP1 cost ~145 MB each):
/// `(segments, sos)` with each segment as `(marker, start, end)` (`start`
/// at its 0xFF, `end` one past its data) and `sos` the offset of the first
/// SOS marker. `None` if the header is malformed or has no scan.
fn jpeg_header_segments(b: &[u8]) -> Option<(Vec<(u8, usize, usize)>, usize)> {
    if b.len() < 4 || b[0] != 0xFF || b[1] != 0xD8 {
        return None;
    }
    let mut segs = Vec::new();
    let mut p = 2usize;
    loop {
        // Skip garbage and fill bytes up to the next marker (as libjpeg).
        while *b.get(p)? != 0xFF {
            p += 1;
        }
        while *b.get(p + 1)? == 0xFF {
            p += 1;
        }
        let m = b[p + 1];
        match m {
            0x00 => p += 2,
            0x01 | 0xD0..=0xD7 => p += 2, // standalone (TEM, RSTn)
            0xD8 | 0xD9 => return None,   // SOI / EOI before a scan
            0xDA => return Some((segs, p)),
            _ => {
                let len = u16::from_be_bytes([*b.get(p + 2)?, *b.get(p + 3)?]) as usize;
                let end = p.checked_add(2)?.checked_add(len)?;
                if len < 2 || end > b.len() {
                    return None;
                }
                segs.push((m, p, end));
                p = end;
            }
        }
    }
}

/// Width and height from a JPEG's SOF (any SOF0-SOF15 but DHT / JPG / DAC),
/// header scan only.
fn jpeg_dims(b: &[u8], segs: &[(u8, usize, usize)]) -> Option<(u32, u32)> {
    let &(_, s, _) = segs.iter().find(|(m, _, _)| (0xC0..=0xCF).contains(m) && !matches!(m, 0xC4 | 0xC8 | 0xCC))?;
    let d = b.get(s + 4..s + 9)?;
    Some((u16::from_be_bytes([d[3], d[4]]) as u32, u16::from_be_bytes([d[1], d[2]]) as u32))
}

/// `cover` with the EXIF block `tiff` as its first segment (APP1), the
/// writer's JFIF APP0 and any earlier EXIF removed. The other segments and
/// the scan data are copied byte for byte (no decode: coefficients
/// untouched).
pub fn with_exif(cover: &[u8], tiff: &[u8]) -> Result<Vec<u8>> {
    if tiff.len() > MAX_TIFF_BYTES {
        return Err(JpegError::InvalidMarkerData("EXIF block larger than one APP1 segment"));
    }
    let (segs, sos) = jpeg_header_segments(cover).ok_or(JpegError::InvalidMarkerData("no JPEG header"))?;
    let mut out = Vec::with_capacity(cover.len() + tiff.len() + 10);
    out.extend_from_slice(&[0xFF, 0xD8, 0xFF, 0xE1]);
    out.extend_from_slice(&((tiff.len() + 8) as u16).to_be_bytes());
    out.extend_from_slice(b"Exif\0\0");
    out.extend_from_slice(tiff);
    for &(m, s, e) in &segs {
        let data = &cover[s + 4..e];
        let dropped = (m == 0xE0 && data.starts_with(b"JFIF\0")) || (m == 0xE1 && data.starts_with(b"Exif\0\0"));
        if !dropped {
            out.extend_from_slice(&cover[s..e]);
        }
    }
    out.extend_from_slice(&cover[sos..]);
    Ok(out)
}

/// The cover with the source file's metadata, cleaned (module docs); the
/// cover unchanged when the source carries no readable EXIF. Header scans
/// only: neither file's coefficients are decoded.
pub fn cover_with_source_metadata(cover: &[u8], source: &[u8]) -> Result<Vec<u8>> {
    let (segs, _) = jpeg_header_segments(cover).ok_or(JpegError::InvalidMarkerData("no JPEG header"))?;
    let (w, h) = jpeg_dims(cover, &segs).ok_or(JpegError::InvalidMarkerData("no SOF marker found"))?;
    match source_exif(source).and_then(|t| exif_for_cover(&t, w, h)) {
        Some(t) => with_exif(cover, &t),
        None => Ok(cover.to_vec()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::jpeg::JpegImage;

    /// A camera-like block: IFD0 (Make, Orientation 6, Exif + GPS pointers)
    /// → Exif IFD (PixelX/Y, DateTimeOriginal, ColorSpace 0xFFFF, a maker
    /// note at a fixed offset) → GPS IFD; IFD1 with a JPEG thumbnail.
    fn camera_tiff(le: bool) -> (Vec<u8>, usize, Vec<u8>) {
        let ord = Order(le);
        let mut t = vec![0u8; 1024];
        t[..4].copy_from_slice(if le { b"II*\0" } else { b"MM\0*" });
        t[4..8].copy_from_slice(&ord.put32(8));
        let entry = |t: &mut Vec<u8>, at: usize, tag: u16, typ: u16, count: u32, v: [u8; 4]| {
            t[at..at + 2].copy_from_slice(&ord.put16(tag));
            t[at + 2..at + 4].copy_from_slice(&ord.put16(typ));
            t[at + 4..at + 8].copy_from_slice(&ord.put32(count));
            t[at + 8..at + 12].copy_from_slice(&v);
        };
        let short = |v: u16| { let b = ord.put16(v); [b[0], b[1], 0, 0] };
        // IFD0 at 8: 4 entries.
        t[8..10].copy_from_slice(&ord.put16(4));
        entry(&mut t, 10, 0x010F, 2, 6, ord.put32(200)); // Make -> "Apple\0" at 200
        entry(&mut t, 22, TAG_ORIENTATION, 3, 1, short(6));
        entry(&mut t, 34, TAG_EXIF_IFD, 4, 1, ord.put32(300));
        entry(&mut t, 46, TAG_GPS_IFD, 4, 1, ord.put32(500));
        t[58..62].copy_from_slice(&ord.put32(600)); // next IFD = IFD1
        t[200..206].copy_from_slice(b"Apple\0");
        // Exif IFD at 300: 5 entries.
        t[300..302].copy_from_slice(&ord.put16(5));
        entry(&mut t, 302, 0x9003, 2, 20, ord.put32(400)); // DateTimeOriginal
        entry(&mut t, 314, 0x927C, 7, 32, ord.put32(800)); // MakerNote at 800
        entry(&mut t, 326, TAG_COLOR_SPACE, 3, 1, short(0xFFFF));
        entry(&mut t, 338, TAG_PIXEL_X, 4, 1, ord.put32(4032));
        entry(&mut t, 350, TAG_PIXEL_Y, 4, 1, ord.put32(3024));
        t[362..366].copy_from_slice(&ord.put32(0));
        t[400..420].copy_from_slice(b"2026:11:02 10:00:00\0");
        let maker: Vec<u8> = (0..32u8).map(|i| i ^ 0x5A).collect();
        t[800..832].copy_from_slice(&maker);
        // GPS IFD at 500: latitude ref.
        t[500..502].copy_from_slice(&ord.put16(1));
        entry(&mut t, 502, 0x0001, 2, 2, *b"N\0\0\0");
        t[514..518].copy_from_slice(&ord.put32(0));
        // IFD1 at 600: thumbnail at 900, 100 bytes.
        t[600..602].copy_from_slice(&ord.put16(2));
        entry(&mut t, 602, TAG_JPEG_IF, 4, 1, ord.put32(900));
        entry(&mut t, 614, TAG_JPEG_IF_LEN, 4, 1, ord.put32(100));
        t[626..630].copy_from_slice(&ord.put32(0));
        for b in &mut t[900..1000] {
            *b = 0xEE;
        }
        (t, 800, maker)
    }

    fn get(t: &[u8], ord: Order, ifd: usize, tag: u16) -> Option<(u16, u32, usize)> {
        let n = ord.u16(t, ifd)? as usize;
        (0..n).map(|i| ifd + 2 + 12 * i).find(|&e| ord.u16(t, e) == Some(tag)).map(|e| (ord.u16(t, e + 2).unwrap(), ord.u32(t, e + 4).unwrap(), e + 8))
    }

    #[test]
    fn cleans_a_camera_block_in_both_byte_orders() {
        for le in [true, false] {
            let ord = Order(le);
            let (src, maker_at, maker) = camera_tiff(le);
            let out = exif_for_cover(&src, 1600, 1200).unwrap();
            let ifd0 = ord.u32(&out, 4).unwrap() as usize;
            let n0 = ord.u16(&out, ifd0).unwrap() as usize;
            assert_eq!(ord.u32(&out, ifd0 + 2 + 12 * n0), Some(0), "no IFD1");
            assert!(!out.windows(100).any(|w| w.iter().all(|&b| b == 0xEE)), "thumbnail gone");
            let (_, _, o) = get(&out, ord, ifd0, TAG_ORIENTATION).unwrap();
            assert_eq!(ord.u16(&out, o), Some(1));
            let (_, _, mk) = get(&out, ord, ifd0, 0x010F).unwrap();
            let mk = ord.u32(&out, mk).unwrap() as usize;
            assert_eq!(&out[mk..mk + 6], b"Apple\0");
            let exif = ord.u32(&out, get(&out, ord, ifd0, TAG_EXIF_IFD).unwrap().2).unwrap() as usize;
            let (_, _, px) = get(&out, ord, exif, TAG_PIXEL_X).unwrap();
            let (_, _, py) = get(&out, ord, exif, TAG_PIXEL_Y).unwrap();
            assert_eq!((ord.u32(&out, px), ord.u32(&out, py)), (Some(1600), Some(1200)));
            let (_, _, cs) = get(&out, ord, exif, TAG_COLOR_SPACE).unwrap();
            assert_eq!(ord.u16(&out, cs), Some(1));
            let (_, _, mn) = get(&out, ord, exif, TAG_MAKER_NOTE).unwrap();
            assert_eq!(ord.u32(&out, mn), Some(maker_at as u32), "maker note kept at its offset");
            assert_eq!(&out[maker_at..maker_at + 32], &maker[..]);
            let gps = ord.u32(&out, get(&out, ord, ifd0, TAG_GPS_IFD).unwrap().2).unwrap() as usize;
            let (_, _, lat) = get(&out, ord, gps, 0x0001).unwrap();
            assert_eq!(&out[lat..lat + 1], b"N");
            // Tags ascending in every directory.
            for ifd in [ifd0, exif, gps] {
                let n = ord.u16(&out, ifd).unwrap() as usize;
                let tags: Vec<u16> = (0..n).map(|i| ord.u16(&out, ifd + 2 + 12 * i).unwrap()).collect();
                assert!(tags.windows(2).all(|w| w[0] < w[1]), "{tags:?}");
            }
        }
    }

    #[test]
    fn puts_the_block_first_and_keeps_the_coefficients() {
        let px: Vec<u8> = (0..256 * 192 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let cover = super::super::cover_writer::encode_rgb(&px, 256, 192, 92, |_, _, _, _| {}).unwrap();
        let (src, _, _) = camera_tiff(false);
        let mut source = vec![0xFF, 0xD8];
        let mut app1 = b"Exif\0\0".to_vec();
        app1.extend_from_slice(&src);
        source.extend_from_slice(&[0xFF, 0xE1]);
        source.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        source.extend_from_slice(&app1);
        source.extend_from_slice(&cover[2..]);
        let out = cover_with_source_metadata(&cover, &source).unwrap();
        assert_eq!(&out[2..4], &[0xFF, 0xE1], "EXIF is the first segment");
        let a = JpegImage::from_bytes(&cover).unwrap();
        let b = JpegImage::from_bytes(&out).unwrap();
        assert_eq!(a.dct_grid(0).coeffs(), b.dct_grid(0).coeffs());
        assert!(!b.raw_segments.iter().any(|s| s.marker == 0xE0), "writer's JFIF dropped");
        assert_eq!(b.exif_orientation(), Some(1));
        let tiff = source_exif(&out).unwrap();
        let ord = Order(false);
        let ifd0 = ord.u32(&tiff, 4).unwrap() as usize;
        let exif = ord.u32(&tiff, get(&tiff, ord, ifd0, TAG_EXIF_IFD).unwrap().2).unwrap() as usize;
        assert_eq!(ord.u32(&tiff, get(&tiff, ord, exif, TAG_PIXEL_X).unwrap().2), Some(256));
        // No EXIF in the source: the cover as it was.
        assert_eq!(cover_with_source_metadata(&cover, &cover).unwrap(), cover);
    }

    #[test]
    fn finds_exif_in_png_webp_and_heif() {
        let (tiff, _, _) = camera_tiff(true);
        let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
        png.extend_from_slice(&(tiff.len() as u32).to_be_bytes());
        png.extend_from_slice(b"eXIf");
        png.extend_from_slice(&tiff);
        png.extend_from_slice(&[0; 4]);
        assert_eq!(source_exif(&png).unwrap(), tiff);

        let mut webp = b"RIFF\0\0\0\0WEBP".to_vec();
        webp.extend_from_slice(b"EXIF");
        webp.extend_from_slice(&(tiff.len() as u32 + 6).to_le_bytes());
        webp.extend_from_slice(b"Exif\0\0");
        webp.extend_from_slice(&tiff);
        assert_eq!(source_exif(&webp).unwrap(), tiff);

        // Minimal HEIF: ftyp, meta { iinf { infe v2 id 7 'Exif' }, iloc v0 }, mdat.
        let bx = |kind: &[u8], payload: &[u8]| {
            let mut b = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
            b.extend_from_slice(kind);
            b.extend_from_slice(payload);
            b
        };
        let infe = bx(b"infe", &[2, 0, 0, 0, 0, 7, 0, 0, b'E', b'x', b'i', b'f', 0]);
        let mut iinf_p = vec![0, 0, 0, 0, 0, 1];
        iinf_p.extend_from_slice(&infe);
        let iinf = bx(b"iinf", &iinf_p);
        let ftyp = bx(b"ftyp", b"heic\0\0\0\0mif1heic");
        let mut payload = vec![0, 0, 0, 6];
        payload.extend_from_slice(b"Exif\0\0");
        payload.extend_from_slice(&tiff);
        // iloc v0: offset_size 4, length_size 4, base 0; 1 item, 1 extent.
        let iloc_len = 8 + 4 + 2 + 2 + 2 + 2 + 2 + 4 + 4;
        let meta_len = 8 + 4 + iinf.len() + iloc_len;
        let data_at = ftyp.len() + meta_len + 8;
        let mut iloc_p = vec![0, 0, 0, 0, 0x44, 0x00, 0, 1, 0, 7, 0, 0, 0, 1];
        iloc_p.extend_from_slice(&(data_at as u32).to_be_bytes());
        iloc_p.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        let iloc = bx(b"iloc", &iloc_p);
        let mut meta_p = vec![0, 0, 0, 0];
        meta_p.extend_from_slice(&iinf);
        meta_p.extend_from_slice(&iloc);
        let mut heif = ftyp.clone();
        heif.extend_from_slice(&bx(b"meta", &meta_p));
        heif.extend_from_slice(&bx(b"mdat", &payload));
        assert_eq!(source_exif(&heif).unwrap(), tiff);
    }

    /// IFD0 at 8 with `n` entries, all Exif-IFD pointers to offset 8 (itself).
    fn self_pointing_tiff(n: usize) -> Vec<u8> {
        let mut t = b"II*\0".to_vec();
        t.extend_from_slice(&8u32.to_le_bytes());
        t.extend_from_slice(&(n as u16).to_le_bytes());
        for _ in 0..n {
            t.extend_from_slice(&TAG_EXIF_IFD.to_le_bytes());
            t.extend_from_slice(&4u16.to_le_bytes());
            t.extend_from_slice(&1u32.to_le_bytes());
            t.extend_from_slice(&8u32.to_le_bytes());
        }
        t.extend_from_slice(&0u32.to_le_bytes());
        t
    }

    fn jpeg_with_app1(cover: &[u8], tiff: &[u8]) -> Vec<u8> {
        let mut app1 = b"Exif\0\0".to_vec();
        app1.extend_from_slice(tiff);
        let mut out = vec![0xFF, 0xD8, 0xFF, 0xE1];
        out.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        out.extend_from_slice(&app1);
        out.extend_from_slice(&cover[2..]);
        out
    }

    #[test]
    fn self_pointing_directories_are_walked_once() {
        // R3.3: N = 60 took 26 s and 783 MB (each pointer re-parsed the
        // directory, ~N³ parses).
        for n in [60usize, 1000, 5000] {
            let t = self_pointing_tiff(n);
            let start = std::time::Instant::now();
            let out = exif_for_cover(&t, 100, 100);
            assert!(start.elapsed() < std::time::Duration::from_millis(500), "n={n}: {:?}", start.elapsed());
            if n <= MAX_ENTRIES {
                // The loop is cut: IFD0 with the added tags, no Exif child from the loop.
                let out = out.expect("readable");
                assert!(out.len() < 1024, "n={n}: {} B", out.len());
            } else {
                assert!(out.is_none(), "n={n}: over the entry cap");
            }
        }
        // Through a JPEG source, as build_cover sees it.
        let px: Vec<u8> = (0..64 * 48 * 3).map(|i| (i * 13 % 251) as u8).collect();
        let cover = super::super::cover_writer::encode_rgb(&px, 64, 48, 90, |_, _, _, _| {}).unwrap();
        let src = jpeg_with_app1(&cover, &self_pointing_tiff(60));
        let start = std::time::Instant::now();
        assert!(cover_with_source_metadata(&cover, &src).is_ok());
        assert!(start.elapsed() < std::time::Duration::from_millis(500), "{:?}", start.elapsed());
    }

    #[test]
    fn pointer_cycles_and_duplicate_pointers_terminate() {
        // IFD0 (8) → Exif (100) → GPS pointer back to IFD0, Interop to Exif;
        // IFD0 also carries the Exif pointer twice.
        let ord = Order(true);
        let mut t = vec![0u8; 256];
        t[..4].copy_from_slice(b"II*\0");
        t[4..8].copy_from_slice(&ord.put32(8));
        let entry = |t: &mut Vec<u8>, at: usize, tag: u16, v: u32| {
            t[at..at + 2].copy_from_slice(&ord.put16(tag));
            t[at + 2..at + 4].copy_from_slice(&ord.put16(4));
            t[at + 4..at + 8].copy_from_slice(&ord.put32(1));
            t[at + 8..at + 12].copy_from_slice(&ord.put32(v));
        };
        t[8..10].copy_from_slice(&ord.put16(2));
        entry(&mut t, 10, TAG_EXIF_IFD, 100);
        entry(&mut t, 22, TAG_EXIF_IFD, 100);
        t[100..102].copy_from_slice(&ord.put16(2));
        entry(&mut t, 102, TAG_GPS_IFD, 8);
        entry(&mut t, 114, TAG_INTEROP_IFD, 100);
        let out = exif_for_cover(&t, 10, 10).expect("readable");
        let ifd0 = ord.u32(&out, 4).unwrap() as usize;
        let n = ord.u16(&out, ifd0).unwrap() as usize;
        let tags: Vec<u16> = (0..n).map(|i| ord.u16(&out, ifd0 + 2 + 12 * i).unwrap()).collect();
        assert_eq!(tags.iter().filter(|&&t| t == TAG_EXIF_IFD).count(), 1, "{tags:?}");
        assert!(!tags.contains(&TAG_GPS_IFD), "the cycle back to IFD0 is not followed: {tags:?}");
    }

    #[test]
    fn overlapping_values_are_capped() {
        // 4000 entries whose 16 KB values all overlap: 64 MB copied before.
        let n = 4000usize;
        let value_at = 8 + 2 + 12 * n + 4;
        let mut t = b"II*\0".to_vec();
        t.extend_from_slice(&8u32.to_le_bytes());
        t.extend_from_slice(&(n as u16).to_le_bytes());
        for i in 0..n {
            t.extend_from_slice(&(0x1000u16 + i as u16).to_le_bytes());
            t.extend_from_slice(&7u16.to_le_bytes()); // UNDEFINED
            t.extend_from_slice(&16_000u32.to_le_bytes());
            t.extend_from_slice(&(value_at as u32).to_le_bytes());
        }
        t.extend_from_slice(&0u32.to_le_bytes());
        t.resize(value_at + 16_000, 0x55);
        let start = std::time::Instant::now();
        assert!(exif_for_cover(&t, 10, 10).is_none());
        assert!(start.elapsed() < std::time::Duration::from_millis(500), "{:?}", start.elapsed());
    }

    fn bx(kind: &[u8], payload: &[u8]) -> Vec<u8> {
        let mut b = ((payload.len() + 8) as u32).to_be_bytes().to_vec();
        b.extend_from_slice(kind);
        b.extend_from_slice(payload);
        b
    }

    #[test]
    fn truncated_heif_boxes_give_none() {
        // R3.6: these panicked (index out of range), which kills an app
        // built with panic = abort.
        let ftyp = bx(b"ftyp", b"heic\0\0\0\0mif1heic");
        let file = |meta_payload: &[u8]| {
            let mut f = ftyp.clone();
            f.extend_from_slice(&bx(b"meta", meta_payload));
            f
        };
        // (a) empty iinf at the end of the file.
        let mut a = vec![0u8; 4];
        a.extend_from_slice(&bx(b"iinf", &[]));
        // (b) iinf with an empty infe at the end.
        let mut iinf = vec![0u8, 0, 0, 0, 0, 1];
        iinf.extend_from_slice(&bx(b"infe", &[]));
        let mut b = vec![0u8; 4];
        b.extend_from_slice(&bx(b"iinf", &iinf));
        // (c) a valid Exif infe, then an iloc with 2 payload bytes.
        let mut iinf_c = vec![0u8, 0, 0, 0, 0, 1];
        iinf_c.extend_from_slice(&bx(b"infe", &[2, 0, 0, 0, 0, 1, 0, 0, b'E', b'x', b'i', b'f']));
        let mut c = vec![0u8; 4];
        c.extend_from_slice(&bx(b"iinf", &iinf_c));
        c.extend_from_slice(&bx(b"iloc", &[0, 0]));
        // (d) a short infe (version 2, no item id).
        let mut iinf_d = vec![0u8, 0, 0, 0, 0, 1];
        iinf_d.extend_from_slice(&bx(b"infe", &[2, 0]));
        let mut d = vec![0u8; 4];
        d.extend_from_slice(&bx(b"iinf", &iinf_d));
        // (e) an iloc whose one item claims 65535 extents with zero-size
        // offset / length fields, many times over (no bytes consumed).
        let mut e = vec![0u8; 4];
        e.extend_from_slice(&bx(b"iinf", &iinf_c));
        let mut iloc = vec![1u8, 0, 0, 0, 0x00, 0x00, 0xFF, 0xFF];
        for _ in 0..0xFFFF {
            iloc.extend_from_slice(&[0, 2, 0, 0, 0, 0, 0xFF, 0xFF]); // item 2, not the Exif item
        }
        e.extend_from_slice(&bx(b"iloc", &iloc));
        // (f) a box with a 64-bit size beyond the file.
        let mut f = ftyp.clone();
        f.extend_from_slice(&[0, 0, 0, 1, b'm', b'e', b't', b'a', 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF]);
        let px: Vec<u8> = (0..64 * 48 * 3).map(|i| (i * 13 % 251) as u8).collect();
        let cover = super::super::cover_writer::encode_rgb(&px, 64, 48, 90, |_, _, _, _| {}).unwrap();
        for (name, bytes) in [("a", file(&a)), ("b", file(&b)), ("c", file(&c)), ("d", file(&d)), ("e", file(&e)), ("f", f)] {
            let start = std::time::Instant::now();
            assert_eq!(source_exif(&bytes), None, "{name}");
            assert_eq!(cover_with_source_metadata(&cover, &bytes).unwrap(), cover, "{name}");
            assert!(start.elapsed() < std::time::Duration::from_millis(500), "{name}: {:?}", start.elapsed());
        }
    }

    #[test]
    fn huge_webp_and_png_chunk_lengths_terminate() {
        // R3.20: `o += 8 + len` wrapped on wasm32 (no overflow checks) and
        // the walk never ended.
        for len in [u32::MAX, u32::MAX - 1, u32::MAX - 12, 0x7FFF_FFFF] {
            let mut webp = b"RIFF\0\0\0\0WEBP".to_vec();
            webp.extend_from_slice(b"VP8X");
            webp.extend_from_slice(&len.to_le_bytes());
            webp.extend_from_slice(&[0; 32]);
            assert_eq!(source_exif(&webp), None, "webp len {len:#x}");
            let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
            png.extend_from_slice(&len.to_be_bytes());
            png.extend_from_slice(b"tEXt");
            png.extend_from_slice(&[0; 32]);
            assert_eq!(source_exif(&png), None, "png len {len:#x}");
        }
        // Truncated chunk headers.
        assert_eq!(source_exif(b"RIFF\0\0\0\0WEBPEXI"), None);
        assert_eq!(source_exif(b"\x89PNG\r\n\x1a\n\0\0\0"), None);
    }

    #[test]
    fn splices_without_decoding() {
        let px: Vec<u8> = (0..120 * 80 * 3).map(|i| (i * 7 % 251) as u8).collect();
        let cover = super::super::cover_writer::encode_rgb(&px, 120, 80, 90, |_, _, _, _| {}).unwrap();
        let (tiff, _, _) = camera_tiff(true);
        let t = exif_for_cover(&tiff, 120, 80).unwrap();
        let out = with_exif(&cover, &t).unwrap();
        // SOI, the new APP1, then the cover's segments minus its JFIF, and
        // the scan data byte for byte.
        let (segs, sos) = jpeg_header_segments(&cover).unwrap();
        assert!(segs.iter().any(|&(m, _, _)| m == 0xE0), "the writer's JFIF");
        let (osegs, osos) = jpeg_header_segments(&out).unwrap();
        assert_eq!(osegs[0].0, 0xE1);
        assert_eq!(&out[osegs[0].1 + 10..osegs[0].2], &t[..]);
        assert!(!osegs.iter().any(|&(m, _, _)| m == 0xE0));
        assert_eq!(osegs.len(), segs.len(), "JFIF out, EXIF in");
        assert_eq!(&out[osos..], &cover[sos..]);
        assert_eq!(jpeg_dims(&out, &osegs), Some((120, 80)));
        // A second EXIF replaces the first.
        let again = with_exif(&out, &t).unwrap();
        assert_eq!(again, out);
        // A progressive cover stays progressive (a decode + re-encode made it baseline).
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-vectors/image/multiscan/prog420_264x200.jpg");
        let prog = std::fs::read(path).unwrap();
        let src = {
            let mut s = vec![0xFF, 0xD8, 0xFF, 0xE1];
            s.extend_from_slice(&((tiff.len() + 8) as u16).to_be_bytes());
            s.extend_from_slice(b"Exif\0\0");
            s.extend_from_slice(&tiff);
            s.extend_from_slice(&prog[2..]);
            s
        };
        assert_eq!(source_exif(&src).unwrap(), tiff);
        let pout = cover_with_source_metadata(&prog, &src).unwrap();
        let (_, psos) = jpeg_header_segments(&prog).unwrap();
        assert!(pout.ends_with(&prog[psos..]));
        let a = JpegImage::from_bytes(&prog).unwrap();
        let b = JpegImage::from_bytes(&pout).unwrap();
        assert_eq!(a.dct_grid(0).coeffs(), b.dct_grid(0).coeffs());
        // Malformed headers are errors, not panics.
        assert!(with_exif(&cover[..20], &t).is_err());
        assert!(cover_with_source_metadata(b"\xFF\xD8\xFF", &src).is_err());
        assert_eq!(source_exif(&[0xFF, 0xD8, 0xFF, 0xE1, 0xFF, 0xFF]), None);
    }
}
