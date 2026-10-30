// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Rewrite a passed-through audio `trak` as plain ISO-BMFF.
//!
//! The stego MP4 builders copy the source's audio track rather than
//! re-muxing it, patching `stco` and the track id. From an iPhone `.mov`
//! that track is QuickTime-flavoured:
//!
//! - the `mp4a` sample entry is a SoundDescription **v1** (or v2), with its
//!   `esds` nested inside a `wave` atom (`frma`, `mp4a`, `esds`, terminator)
//!   and a `chan` channel layout beside it;
//! - `mdia/hdlr` carries the QuickTime component type `mhlr`;
//! - `minf` holds a second, data-handler `hdlr` (`dhlr` / `url `).
//!
//! ffmpeg reads that fine. Android's `MPEG4Extractor` does not: a stego video
//! made by the CLI from a `.mov` showed duration / size NULL in MediaStore,
//! 0:00 in the picker and no poster frame, while the same streams remuxed by
//! ffmpeg (which writes a v0 `mp4a` with `esds` directly inside) were fine.
//! An ISO reader expects exactly that shape, so the copied track is
//! normalized to it here. Anything this does not recognise is left exactly
//! as the source had it.
//!
//! The copy also carried fields counted in the SOURCE's movie timescale —
//! `tkhd.duration` and the edit list's segment durations — into a file whose
//! `mvhd` counts in another one. An iPhone `.mov` counts in 600ths of a
//! second and the stego MP4 in 1000ths, so a 17 s clip's audio edit said
//! 10200 / 600 = 17 s in the source and 10.2 s in the output: players that
//! honour edit lists (QuickTime, Android, browsers) cut the audio off at 60%.
//! Those fields are rescaled to the output's movie timescale.
//!
//! And the copy carried the source file's identity. A camera writes its
//! capture time, to the second, into the track's `tkhd` and `mdhd`
//! (creation and modification); it names itself in the media handler
//! (`Core Media Audio`, QuickTime manufacturer `appl`), and a track can carry
//! user data, metadata and vendor boxes of its own. In a stego video that is
//! a link back to the original clip in the sender's library. phasm writes 0
//! and a neutral handler name for its own video track, so the audio track is
//! brought in line ([`scrub`]): always, whether or not the rewrites above
//! applied, and if the track cannot be walked far enough to clear it, the
//! mux fails rather than pass the source's fields on. Nothing the audio
//! itself depends on changes: samples, sample tables, edit list, sample
//! groups and codec configuration stay exactly as they were.

use super::Mp4Error;

/// One box: its four-character type, and its payload (everything after the
/// header).
struct Box<'a> {
    typ: [u8; 4],
    payload: &'a [u8],
}

/// Split `buf` into its top-level boxes. `None` if a size does not fit.
fn boxes(buf: &[u8]) -> Option<Vec<Box<'_>>> {
    match split_boxes(buf)? {
        (out, []) => Some(out),
        _ => None,
    }
}

/// [`boxes`], except that a remainder too short to hold a box header (a
/// QuickTime 32-bit zero terminator) ends the list, as it does for the
/// demuxer, and is returned for the caller to keep as it was.
fn split_boxes(buf: &[u8]) -> Option<(Vec<Box<'_>>, &[u8])> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < buf.len() {
        if buf.len() - off < 8 {
            return Some((out, &buf[off..]));
        }
        let size32 = u32::from_be_bytes(buf[off..off + 4].try_into().ok()?) as usize;
        let typ: [u8; 4] = buf[off + 4..off + 8].try_into().ok()?;
        let (header, size) = match size32 {
            0 => (8, buf.len() - off),
            1 => {
                if buf.len() - off < 16 {
                    return None;
                }
                let large = u64::from_be_bytes(buf[off + 8..off + 16].try_into().ok()?);
                (16, usize::try_from(large).ok()?)
            }
            n => (8, n),
        };
        if size < header || size > buf.len() - off {
            return None;
        }
        out.push(Box { typ, payload: &buf[off + header..off + size] });
        off += size;
    }
    Some((out, &[]))
}

/// Serialize one box with a freshly computed size.
fn write_box(out: &mut Vec<u8>, typ: &[u8; 4], payload: &[u8]) {
    let size = payload.len() + 8;
    if let Ok(size32) = u32::try_from(size) {
        out.extend_from_slice(&size32.to_be_bytes());
        out.extend_from_slice(typ);
    } else {
        out.extend_from_slice(&1u32.to_be_bytes());
        out.extend_from_slice(typ);
        out.extend_from_slice(&((size + 8) as u64).to_be_bytes());
    }
    out.extend_from_slice(payload);
}

/// Return `trak` (a whole `trak` box, header included) as it should go into
/// the output: plain ISO-BMFF, with its movie-timescale fields converted from
/// `from_movie_timescale` (the source's `mvhd`) to `to_movie_timescale` (the
/// output's), and nothing left that identifies the source file
/// ([`scrub`]). The ISO and timescale steps leave a layout they do not
/// understand as the source had it; the scrub does not — when it cannot
/// walk the track down to the fields it clears, this is an error and the
/// caller fails the mux.
pub(crate) fn passthrough_audio_trak(
    trak: &[u8],
    from_movie_timescale: u32,
    to_movie_timescale: u32,
) -> Result<Vec<u8>, Mp4Error> {
    let iso = normalize(trak).unwrap_or_else(|| trak.to_vec());
    let timed = rescale_movie_time(&iso, from_movie_timescale, to_movie_timescale).unwrap_or(iso);
    scrub(&timed).ok_or_else(|| {
        Mp4Error::InvalidBox(
            "audio track: cannot clear the source's timestamps and metadata \
             (box layout not understood)"
                .into(),
        )
    })
}

/// `mvhd.timescale` of a whole MP4 / MOV file, from its top-level `moov`.
pub(crate) fn movie_timescale(file: &[u8]) -> Option<u32> {
    let moov = boxes(file)?.into_iter().find(|b| &b.typ == b"moov")?;
    let mvhd = boxes(moov.payload)?.into_iter().find(|b| &b.typ == b"mvhd")?;
    // version/flags(4), then creation + modification: 4+4 (v0) or 8+8 (v1).
    let at = if mvhd.payload.first()? == &1 { 20 } else { 12 };
    let ts = u32::from_be_bytes(mvhd.payload.get(at..at + 4)?.try_into().ok()?);
    (ts > 0).then_some(ts)
}

/// Rescale `tkhd.duration` and every `elst` segment duration (both counted
/// in the movie timescale) by `to / from`. Media times in `elst` are in the
/// track's own timescale and stay.
fn rescale_movie_time(trak: &[u8], from: u32, to: u32) -> Option<Vec<u8>> {
    if from == 0 || to == 0 || from == to {
        return None;
    }
    let scale = |v: u64| -> u64 { ((v as u128 * to as u128 + from as u128 / 2) / from as u128) as u64 };
    let top = boxes(trak)?;
    if top.len() != 1 || &top[0].typ != b"trak" {
        return None;
    }
    let mut payload = Vec::with_capacity(trak.len());
    for child in boxes(top[0].payload)? {
        match &child.typ {
            b"tkhd" => {
                let mut t = child.payload.to_vec();
                match t.first()? {
                    0 if t.len() >= 24 => {
                        let d = u32::from_be_bytes(t[20..24].try_into().ok()?);
                        let n = u32::try_from(scale(d as u64)).ok()?;
                        t[20..24].copy_from_slice(&n.to_be_bytes());
                    }
                    1 if t.len() >= 36 => {
                        let d = u64::from_be_bytes(t[28..36].try_into().ok()?);
                        t[28..36].copy_from_slice(&scale(d).to_be_bytes());
                    }
                    _ => return None,
                }
                write_box(&mut payload, b"tkhd", &t);
            }
            b"edts" => {
                let mut edts = Vec::new();
                for e in boxes(child.payload)? {
                    if &e.typ != b"elst" {
                        write_box(&mut edts, &e.typ, e.payload);
                        continue;
                    }
                    let mut l = e.payload.to_vec();
                    let version = *l.first()?;
                    let count = u32::from_be_bytes(l.get(4..8)?.try_into().ok()?) as usize;
                    let entry_len = if version == 1 { 20 } else { 12 };
                    if l.len() < 8 + count * entry_len {
                        return None;
                    }
                    for i in 0..count {
                        let at = 8 + i * entry_len;
                        if version == 1 {
                            let d = u64::from_be_bytes(l[at..at + 8].try_into().ok()?);
                            l[at..at + 8].copy_from_slice(&scale(d).to_be_bytes());
                        } else {
                            let d = u32::from_be_bytes(l[at..at + 4].try_into().ok()?);
                            let n = u32::try_from(scale(d as u64)).ok()?;
                            l[at..at + 4].copy_from_slice(&n.to_be_bytes());
                        }
                    }
                    write_box(&mut edts, b"elst", &l);
                }
                write_box(&mut payload, b"edts", &edts);
            }
            t => write_box(&mut payload, t, child.payload),
        }
    }
    let mut out = Vec::with_capacity(trak.len());
    write_box(&mut out, b"trak", &payload);
    Some(out)
}

fn normalize(trak: &[u8]) -> Option<Vec<u8>> {
    let top = boxes(trak)?;
    if top.len() != 1 || &top[0].typ != b"trak" {
        return None;
    }
    let mut payload = Vec::with_capacity(trak.len());
    for child in boxes(top[0].payload)? {
        match &child.typ {
            b"mdia" => write_box(&mut payload, b"mdia", &normalize_mdia(child.payload)?),
            t => write_box(&mut payload, t, child.payload),
        }
    }
    let mut out = Vec::with_capacity(trak.len());
    write_box(&mut out, b"trak", &payload);
    Some(out)
}

fn normalize_mdia(mdia: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(mdia.len());
    for child in boxes(mdia)? {
        match &child.typ {
            b"hdlr" => write_box(&mut out, b"hdlr", &iso_media_hdlr(child.payload)?),
            b"minf" => write_box(&mut out, b"minf", &normalize_minf(child.payload)?),
            t => write_box(&mut out, t, child.payload),
        }
    }
    Some(out)
}

/// The media `hdlr` with QuickTime's component type (`mhlr`) and component
/// manufacturer / flags zeroed, as ISO defines `pre_defined` and `reserved`.
fn iso_media_hdlr(hdlr: &[u8]) -> Option<Vec<u8>> {
    // version/flags(4) pre_defined(4) handler_type(4) reserved(12) name…
    if hdlr.len() < 24 {
        return None;
    }
    let mut out = hdlr.to_vec();
    out[4..8].fill(0);
    out[12..24].fill(0);
    Some(out)
}

fn normalize_minf(minf: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(minf.len());
    for child in boxes(minf)? {
        match &child.typ {
            // QuickTime's data-handler `hdlr` (`dhlr` / `url `); ISO has
            // none here (`dinf/dref` covers it).
            b"hdlr" => {}
            b"stbl" => write_box(&mut out, b"stbl", &normalize_stbl(child.payload)?),
            t => write_box(&mut out, t, child.payload),
        }
    }
    Some(out)
}

fn normalize_stbl(stbl: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(stbl.len());
    for child in boxes(stbl)? {
        match &child.typ {
            b"stsd" => write_box(&mut out, b"stsd", &normalize_stsd(child.payload)?),
            t => write_box(&mut out, t, child.payload),
        }
    }
    Some(out)
}

fn normalize_stsd(stsd: &[u8]) -> Option<Vec<u8>> {
    // version/flags(4) entry_count(4) entries…
    if stsd.len() < 8 {
        return None;
    }
    let mut out = stsd[..8].to_vec();
    for entry in boxes(&stsd[8..])? {
        match &entry.typ {
            b"mp4a" => write_box(&mut out, b"mp4a", &iso_mp4a(entry.payload)?),
            t => write_box(&mut out, t, entry.payload),
        }
    }
    Some(out)
}

/// An `mp4a` AudioSampleEntry as ISO v0 with `esds` directly inside.
///
/// SampleEntry: reserved(6) data_reference_index(2). Then the sound
/// description: version(2) revision(2) vendor(4) channel_count(2)
/// sample_size(2) compression_id(2) packet_size(2) sample_rate(4, 16.16)
/// — 28 bytes in all, where ISO's children start. QuickTime v1 appends 16
/// bytes, v2 36 (with its own channel count and a 64-bit float rate).
fn iso_mp4a(entry: &[u8]) -> Option<Vec<u8>> {
    const V0_LEN: usize = 28;
    if entry.len() < V0_LEN {
        return None;
    }
    let version = u16::from_be_bytes([entry[8], entry[9]]);
    let (fields_len, channels, rate_16_16) = match version {
        0 | 1 => {
            let channels = u16::from_be_bytes([entry[16], entry[17]]);
            let rate = u32::from_be_bytes(entry[24..28].try_into().ok()?);
            (if version == 0 { V0_LEN } else { V0_LEN + 16 }, channels, rate)
        }
        2 => {
            if entry.len() < V0_LEN + 36 {
                return None;
            }
            let rate = f64::from_be_bytes(entry[V0_LEN + 4..V0_LEN + 12].try_into().ok()?);
            let channels = u32::from_be_bytes(entry[V0_LEN + 12..V0_LEN + 16].try_into().ok()?);
            if !(rate > 0.0 && rate < 65536.0) || channels == 0 || channels > u16::MAX as u32 {
                return None;
            }
            (V0_LEN + 36, channels as u16, (rate as u32) << 16)
        }
        _ => return None,
    };
    if entry.len() < fields_len {
        return None;
    }

    // Children: keep the ISO ones, take `esds` out of QuickTime's `wave`,
    // drop `wave` and the QuickTime channel layout `chan`.
    let mut esds: Option<&[u8]> = None;
    let mut kept: Vec<Box<'_>> = Vec::new();
    for child in boxes(&entry[fields_len..])? {
        match &child.typ {
            b"esds" => esds = Some(child.payload),
            b"wave" => {
                for inner in boxes(child.payload).unwrap_or_default() {
                    if &inner.typ == b"esds" && esds.is_none() {
                        esds = Some(inner.payload);
                    }
                }
            }
            b"chan" => {}
            _ => kept.push(child),
        }
    }
    // Without an `esds` the decoder config is unknown: leave it alone.
    let esds = esds?;

    let mut out = Vec::with_capacity(entry.len());
    out.extend_from_slice(&[0u8; 6]); // reserved
    out.extend_from_slice(&entry[6..8]); // data_reference_index
    out.extend_from_slice(&[0u8; 8]); // version 0, revision, vendor
    out.extend_from_slice(&channels.to_be_bytes());
    out.extend_from_slice(&16u16.to_be_bytes()); // sample_size
    out.extend_from_slice(&[0u8; 4]); // compression_id, packet_size
    out.extend_from_slice(&rate_16_16.to_be_bytes());
    write_box(&mut out, b"esds", esds);
    for child in kept {
        write_box(&mut out, &child.typ, child.payload);
    }
    Some(out)
}

// ─── Source identity ──────────────────────────────────────────────────

/// Boxes [`scrub`] drops wherever it meets them in the track (`trak`,
/// `mdia`, `minf`, `stbl`): user data (`©day`, `©xyz`, `loci`, `name`, …),
/// ISO / QuickTime metadata (`keys` / `ilst`, XMP), vendor `uuid` extensions
/// (XMP, maker notes) and `free` / `skip`, which can hold the stale bytes of
/// metadata a writer blanked in place. No reader needs any of them to decode
/// or time the track.
const SOURCE_METADATA_BOXES: [&[u8; 4]; 5] = [b"udta", b"meta", b"uuid", b"free", b"skip"];

/// The `hdlr` name phasm gives an audio track, as the video track's
/// `VideoHandler` (`build.rs`) is the libavformat default.
const SOUND_HANDLER_NAME: &[u8] = b"SoundHandler";

/// Where in the track [`scrub_children`] is.
#[derive(Clone, Copy)]
enum Level {
    Trak,
    Mdia,
    Minf,
    Stbl,
}

/// Clear what the track says about the file it came from:
///
/// - `tkhd` / `mdhd` creation and modification time → 0, in the 32-bit
///   (version 0) and 64-bit (version 1) layouts alike, as phasm writes for
///   its video track;
/// - the media `hdlr` → `pre_defined` / `reserved` zero (QuickTime's `mhlr`
///   and manufacturer `appl` sit there) and the name [`SOUND_HANDLER_NAME`]
///   (the source's names the framework that wrote it, "Core Media Audio");
/// - QuickTime's data-handler `hdlr` inside `minf` → dropped (it is named
///   after the writer too; ISO has none there);
/// - `dinf` → one self-contained `url ` per data reference, as phasm writes
///   for video (a `url ` location or a QuickTime `alis` record can name a
///   path, a volume, another file);
/// - each sound description's QuickTime `vendor` code → 0;
/// - [`SOURCE_METADATA_BOXES`] → dropped.
///
/// Everything else is copied byte for byte: the sample tables, `edts`,
/// `sgpd` / `sbgp`, the codec configuration, and any box not named here.
/// `None` when the layout cannot be walked down to these fields; nothing is
/// returned that could not be cleared.
fn scrub(trak: &[u8]) -> Option<Vec<u8>> {
    let (top, tail) = split_boxes(trak)?;
    if top.len() != 1 || &top[0].typ != b"trak" || !tail.is_empty() {
        return None;
    }
    let mut out = Vec::with_capacity(trak.len());
    write_box(&mut out, b"trak", &scrub_children(top[0].payload, Level::Trak)?);
    Some(out)
}

fn scrub_children(container: &[u8], level: Level) -> Option<Vec<u8>> {
    let (children, tail) = split_boxes(container)?;
    let mut out = Vec::with_capacity(container.len());
    for child in children {
        if SOURCE_METADATA_BOXES.contains(&&child.typ) {
            continue;
        }
        match (level, &child.typ) {
            (Level::Trak, b"tkhd") | (Level::Mdia, b"mdhd") => {
                write_box(&mut out, &child.typ, &zero_creation_modification(child.payload)?)
            }
            (Level::Trak, b"mdia") => {
                write_box(&mut out, b"mdia", &scrub_children(child.payload, Level::Mdia)?)
            }
            (Level::Mdia, b"hdlr") => write_box(&mut out, b"hdlr", &neutral_media_hdlr(child.payload)?),
            (Level::Mdia, b"minf") => {
                write_box(&mut out, b"minf", &scrub_children(child.payload, Level::Minf)?)
            }
            // QuickTime's data handler ("Core Media Data Handler", `appl`).
            (Level::Minf, b"hdlr") => {}
            (Level::Minf, b"dinf") => write_box(&mut out, b"dinf", &self_contained_dinf(child.payload)),
            (Level::Minf, b"stbl") => {
                write_box(&mut out, b"stbl", &scrub_children(child.payload, Level::Stbl)?)
            }
            (Level::Stbl, b"stsd") => write_box(&mut out, b"stsd", &zero_sound_vendor(child.payload)?),
            (_, t) => write_box(&mut out, t, child.payload),
        }
    }
    out.extend_from_slice(tail);
    Some(out)
}

/// A `tkhd` / `mdhd` payload with creation and modification time 0: the
/// 4 + 4 bytes after version/flags in version 0, 8 + 8 in version 1.
fn zero_creation_modification(full_box: &[u8]) -> Option<Vec<u8>> {
    let end = match full_box.first()? {
        0 => 12,
        1 => 20,
        _ => return None,
    };
    if full_box.len() < end {
        return None;
    }
    let mut out = full_box.to_vec();
    out[4..end].fill(0);
    Some(out)
}

/// The media `hdlr` as phasm writes its own: version/flags, `pre_defined`
/// and `reserved` zero, the handler type kept, a neutral NUL-terminated
/// name — [`SOUND_HANDLER_NAME`] for `soun`, empty otherwise.
fn neutral_media_hdlr(hdlr: &[u8]) -> Option<Vec<u8>> {
    // version/flags(4) pre_defined(4) handler_type(4) reserved(12) name…
    let handler_type = hdlr.get(8..12)?;
    let name: &[u8] = if handler_type == b"soun" { SOUND_HANDLER_NAME } else { b"" };
    let mut out = vec![0u8; 8];
    out.extend_from_slice(handler_type);
    out.extend_from_slice(&[0u8; 12]);
    out.extend_from_slice(name);
    out.push(0);
    Some(out)
}

/// A `dinf` payload holding one self-contained `url ` reference (flags 1,
/// no location — after the mux the samples are in this file) for each
/// entry the source's `dref` had, at least one, so every sample
/// description's `data_reference_index` still resolves. With one entry it
/// is byte for byte the video track's.
fn self_contained_dinf(dinf: &[u8]) -> Vec<u8> {
    let entries = split_boxes(dinf)
        .and_then(|(children, _)| children.into_iter().find(|b| &b.typ == b"dref"))
        .and_then(|dref| dref.payload.get(8..).and_then(split_boxes).map(|(e, _)| e.len()))
        .unwrap_or(0)
        .max(1);
    let mut dref = Vec::with_capacity(8 + 12 * entries);
    dref.extend_from_slice(&[0u8; 4]); // version/flags
    dref.extend_from_slice(&(entries as u32).to_be_bytes());
    for _ in 0..entries {
        write_box(&mut dref, b"url ", &[0, 0, 0, 1]); // version 0, flags = self-contained
    }
    let mut out = Vec::with_capacity(dref.len() + 8);
    write_box(&mut out, b"dref", &dref);
    out
}

/// `stsd` with the QuickTime `vendor` code of each sound description
/// zeroed. SampleEntry reserved(6) data_reference_index(2), then
/// version(2) revision(2) vendor(4): bytes 12..16, which ISO's
/// AudioSampleEntry reserves as zero. Nothing else in an entry changes —
/// not the channel count or rate, not the codec configuration after it.
fn zero_sound_vendor(stsd: &[u8]) -> Option<Vec<u8>> {
    // version/flags(4) entry_count(4) entries…
    let head = stsd.get(..8)?;
    let (entries, tail) = split_boxes(&stsd[8..])?;
    let mut out = Vec::with_capacity(stsd.len());
    out.extend_from_slice(head);
    for entry in entries {
        let mut payload = entry.payload.to_vec();
        if payload.len() >= 28 {
            payload[12..16].fill(0);
        }
        write_box(&mut out, &entry.typ, &payload);
    }
    out.extend_from_slice(tail);
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// No timescale change (same movie timescale on both sides).
    fn passthrough_audio_trak_same(trak: &[u8]) -> Vec<u8> {
        passthrough_audio_trak(trak, 1000, 1000).expect("the track can be cleared")
    }

    fn bx(typ: &[u8; 4], payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        write_box(&mut out, typ, payload);
        out
    }

    fn cat(parts: &[Vec<u8>]) -> Vec<u8> {
        parts.concat()
    }

    const ESDS: &[u8] = &[0, 0, 0, 0, 3, 0x19, 0, 0, 0, 4, 0x11, 0x40, 0x15, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5, 2, 0x12, 0x10, 6, 1, 2];

    /// A QuickTime-style audio trak as an iPhone `.mov` carries it.
    fn quicktime_trak() -> Vec<u8> {
        let mut v1 = vec![0u8; 6];
        v1.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        v1.extend_from_slice(&1u16.to_be_bytes()); // version 1
        v1.extend_from_slice(&[0u8; 6]); // revision, vendor
        v1.extend_from_slice(&2u16.to_be_bytes()); // channels
        v1.extend_from_slice(&16u16.to_be_bytes());
        v1.extend_from_slice(&0xFFFEu16.to_be_bytes()); // compression_id -2
        v1.extend_from_slice(&0u16.to_be_bytes());
        v1.extend_from_slice(&(44100u32 << 16).to_be_bytes());
        v1.extend_from_slice(&[0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]); // v1 extension
        let wave = bx(b"wave", &cat(&[
            bx(b"frma", b"mp4a"),
            bx(b"mp4a", &[0, 0, 0, 0]),
            bx(b"esds", ESDS),
            vec![0, 0, 0, 8, 0, 0, 0, 0], // terminator
        ]));
        v1.extend_from_slice(&wave);
        v1.extend_from_slice(&bx(b"chan", &[0u8; 24]));
        let stsd = bx(b"stsd", &cat(&[vec![0, 0, 0, 0, 0, 0, 0, 1], bx(b"mp4a", &v1)]));
        let stbl = bx(b"stbl", &cat(&[stsd, bx(b"stts", &[0u8; 16]), bx(b"stco", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 9])]));
        let mut mdhdlr = vec![0u8; 4];
        mdhdlr.extend_from_slice(b"mhlr");
        mdhdlr.extend_from_slice(b"soun");
        mdhdlr.extend_from_slice(b"appl");
        mdhdlr.extend_from_slice(&[0u8; 8]);
        mdhdlr.extend_from_slice(b"\x0cCore Media Audio");
        let mut dhlr = vec![0u8; 4];
        dhlr.extend_from_slice(b"dhlr");
        dhlr.extend_from_slice(b"url ");
        dhlr.extend_from_slice(&[0u8; 13]);
        let minf = bx(b"minf", &cat(&[bx(b"smhd", &[0u8; 8]), bx(b"hdlr", &dhlr), bx(b"dinf", &bx(b"dref", &[0u8; 8])), stbl]));
        let mdia = bx(b"mdia", &cat(&[bx(b"mdhd", &[0u8; 24]), bx(b"hdlr", &mdhdlr), minf]));
        let mut tkhd = vec![0u8; 84];
        tkhd[20..24].copy_from_slice(&10199u32.to_be_bytes()); // duration, movie ts 600
        let mut elst = vec![0u8, 0, 0, 0, 0, 0, 0, 1];
        elst.extend_from_slice(&10199u32.to_be_bytes()); // segment duration, movie ts 600
        elst.extend_from_slice(&0i32.to_be_bytes()); // media time
        elst.extend_from_slice(&[0, 1, 0, 0]); // rate 1.0
        let edts = bx(b"edts", &bx(b"elst", &elst));
        bx(b"trak", &cat(&[bx(b"tkhd", &tkhd), edts, mdia]))
    }

    #[test]
    fn movie_time_fields_follow_the_output_timescale() {
        // An iPhone .mov counts in 600ths, the stego MP4 in 1000ths. Copied
        // verbatim, the audio edit said 10.2 s of a 17 s clip.
        let out = passthrough_audio_trak(&quicktime_trak(), 600, 1000).unwrap();
        let tkhd = find(&out, &[b"trak", b"tkhd"]).unwrap();
        assert_eq!(u32::from_be_bytes(tkhd[20..24].try_into().unwrap()), 16998);
        let elst = find(&out, &[b"trak", b"edts", b"elst"]).unwrap();
        assert_eq!(u32::from_be_bytes(elst[8..12].try_into().unwrap()), 16998);
        assert_eq!(i32::from_be_bytes(elst[12..16].try_into().unwrap()), 0, "media time is not movie time");
        // And it is still normalized.
        let stsd = find(&out, &[b"trak", b"mdia", b"minf", b"stbl", b"stsd"]).unwrap();
        let mp4a = boxes(&stsd[8..]).unwrap().remove(0).payload.to_vec();
        assert_eq!(u16::from_be_bytes([mp4a[8], mp4a[9]]), 0);
    }

    #[test]
    fn reads_the_movie_timescale_of_a_file() {
        let mut mvhd = vec![0u8; 100];
        mvhd[12..16].copy_from_slice(&600u32.to_be_bytes());
        let file = cat(&[bx(b"ftyp", b"qt  \0\0\0\0"), bx(b"mdat", &[1, 2, 3]), bx(b"moov", &bx(b"mvhd", &mvhd))]);
        assert_eq!(movie_timescale(&file), Some(600));
        assert_eq!(movie_timescale(&file[..20]), None);
    }

    fn find<'a>(buf: &'a [u8], path: &[&[u8; 4]]) -> Option<&'a [u8]> {
        let mut cur = buf;
        for (i, typ) in path.iter().enumerate() {
            let children = if i == 0 { boxes(cur)? } else { boxes(cur)? };
            cur = children.into_iter().find(|b| &&b.typ == typ)?.payload;
        }
        Some(cur)
    }

    #[test]
    fn a_quicktime_audio_trak_becomes_plain_iso() {
        let out = passthrough_audio_trak_same(&quicktime_trak());

        let mdia_hdlr = find(&out, &[b"trak", b"mdia", b"hdlr"]).unwrap();
        assert_eq!(&mdia_hdlr[4..8], &[0, 0, 0, 0], "QuickTime component type left in");
        assert_eq!(&mdia_hdlr[8..12], b"soun");
        assert_eq!(&mdia_hdlr[12..24], &[0u8; 12]);
        assert_eq!(&mdia_hdlr[24..], b"SoundHandler\0", "the writer's name left in");

        let minf = find(&out, &[b"trak", b"mdia", b"minf"]).unwrap();
        assert!(boxes(minf).unwrap().iter().all(|b| &b.typ != b"hdlr"), "data-handler hdlr left in");

        let stsd = find(&out, &[b"trak", b"mdia", b"minf", b"stbl", b"stsd"]).unwrap();
        let entries = boxes(&stsd[8..]).unwrap();
        assert_eq!(entries.len(), 1);
        let mp4a = entries[0].payload;
        assert_eq!(u16::from_be_bytes([mp4a[8], mp4a[9]]), 0, "not version 0");
        assert_eq!(u16::from_be_bytes([mp4a[16], mp4a[17]]), 2, "channel count");
        assert_eq!(u32::from_be_bytes(mp4a[24..28].try_into().unwrap()), 44100 << 16);
        let children = boxes(&mp4a[28..]).unwrap();
        let types: Vec<&[u8; 4]> = children.iter().map(|b| &b.typ).collect();
        assert_eq!(types, vec![b"esds"], "esds must sit directly in mp4a, alone");
        assert_eq!(children[0].payload, ESDS);

        // Siblings untouched; the result still parses end to end.
        assert!(find(&out, &[b"trak", b"mdia", b"minf", b"stbl", b"stco"]).is_some());
        assert!(find(&out, &[b"trak", b"tkhd"]).is_some());
    }

    #[test]
    fn an_iso_trak_comes_back_unchanged_in_content() {
        let once = passthrough_audio_trak_same(&quicktime_trak());
        assert_eq!(passthrough_audio_trak_same(&once), once, "normalizing is idempotent");
    }

    #[test]
    fn anything_unrecognised_is_copied_verbatim() {
        let junk = vec![0, 0, 0, 9, b't', b'r', b'a', b'k', 7];
        assert_eq!(passthrough_audio_trak_same(&junk), junk);
        // An mp4a without any esds: the decoder config is unknown, keep it.
        let no_esds = {
            let mut t = quicktime_trak();
            let at = t.windows(4).position(|w| w == b"esds").unwrap();
            t[at..at + 4].copy_from_slice(b"xxxx");
            t
        };
        let out = passthrough_audio_trak_same(&no_esds);
        let stsd = find(&out, &[b"trak", b"mdia", b"minf", b"stbl", b"stsd"]).unwrap();
        let mp4a = boxes(&stsd[8..]).unwrap().remove(0).payload.to_vec();
        assert_eq!(u16::from_be_bytes([mp4a[8], mp4a[9]]), 1, "entry without esds must stay as it was");
    }

    // ─── Source identity ──────────────────────────────────────────────

    /// What [`iso_trak`] puts into a synthetic ISO audio track.
    #[derive(Clone, Copy)]
    struct Synth {
        tkhd_version: u8,
        mdhd_version: u8,
        /// `tkhd` and `mdhd` creation / modification time.
        times: (u64, u64),
        /// `udta` / `meta` / `skip` in `trak`, `udta` in `mdia`, `free` in
        /// `minf`, `uuid` in `stbl`.
        metadata_boxes: bool,
        /// The media handler as a camera names it (`mhlr`, `appl`,
        /// "Core Media Audio") plus QuickTime's data-handler `hdlr`.
        writer_names: bool,
        /// A data reference that names a file (`alis`, not self-contained).
        external_dref: bool,
    }

    impl Synth {
        /// What the output must be: no source identity at all.
        fn cleared(self) -> Self {
            Synth { times: (0, 0), metadata_boxes: false, writer_names: false, external_dref: false, ..self }
        }
    }

    /// 2026-05-02 11:54:26 / 11:54:27 in seconds since 1904, the capture time
    /// of the iPhone 7 fixture's tracks.
    const CAPTURE_TIMES: (u64, u64) = (3_860_567_666, 3_860_567_667);

    const UNITY_MATRIX: [u8; 36] =
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x40, 0, 0, 0];

    /// version(1) + flags(3), then creation + modification in the version's width.
    fn full_box_times(version: u8, flags: u32, times: (u64, u64)) -> Vec<u8> {
        let mut b = vec![version];
        b.extend_from_slice(&flags.to_be_bytes()[1..]);
        if version == 1 {
            b.extend_from_slice(&times.0.to_be_bytes());
            b.extend_from_slice(&times.1.to_be_bytes());
        } else {
            b.extend_from_slice(&(times.0 as u32).to_be_bytes());
            b.extend_from_slice(&(times.1 as u32).to_be_bytes());
        }
        b
    }

    fn duration(version: u8, d: u64) -> Vec<u8> {
        if version == 1 { d.to_be_bytes().to_vec() } else { (d as u32).to_be_bytes().to_vec() }
    }

    /// The parts of the synthetic track the scrub must leave byte for byte.
    fn untouched_stbl_tables() -> Vec<Vec<u8>> {
        vec![
            bx(b"stts", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 4, 0]),
            bx(b"stsc", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1]),
            bx(b"stsz", &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 1, 0x11, 0, 0, 1, 0x22, 0, 0, 1, 0x33]),
            bx(b"stco", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0x10, 0]),
            bx(b"sgpd", &[1, 0, 0, 0, b'r', b'o', b'l', b'l', 0, 0, 0, 2, 0, 0, 0, 1, 0xFF, 0xFF]),
            bx(b"sbgp", &[0, 0, 0, 0, b'r', b'o', b'l', b'l', 0, 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 1]),
        ]
    }

    /// An ISO audio `trak` (already in the shape `normalize` writes, so the
    /// output can be predicted exactly), with what `s` asks for.
    fn iso_trak(s: Synth) -> Vec<u8> {
        let mut tkhd = full_box_times(s.tkhd_version, 0x7, s.times);
        tkhd.extend_from_slice(&7u32.to_be_bytes()); // track_ID
        tkhd.extend_from_slice(&[0u8; 4]);
        tkhd.extend_from_slice(&duration(s.tkhd_version, 17_000)); // movie timescale 1000
        tkhd.extend_from_slice(&[0u8; 8]);
        tkhd.extend_from_slice(&[0, 0, 0, 1, 1, 0, 0, 0]); // layer, alternate_group 1, volume 1.0
        tkhd.extend_from_slice(&UNITY_MATRIX);
        tkhd.extend_from_slice(&[0u8; 8]); // width, height

        let mut elst = vec![0u8, 0, 0, 0, 0, 0, 0, 1];
        elst.extend_from_slice(&17_000u32.to_be_bytes());
        elst.extend_from_slice(&2112i32.to_be_bytes()); // priming, in media time
        elst.extend_from_slice(&[0, 1, 0, 0]);

        let mut mdhd = full_box_times(s.mdhd_version, 0, s.times);
        mdhd.extend_from_slice(&48_000u32.to_be_bytes());
        mdhd.extend_from_slice(&duration(s.mdhd_version, 816_000));
        mdhd.extend_from_slice(&[0x55, 0xC4, 0, 0]); // "und", pre_defined

        let media_hdlr = if s.writer_names {
            cat(&[b"\0\0\0\0mhlrsounappl".to_vec(), vec![0u8; 8], b"\x10Core Media Audio".to_vec()])
        } else {
            cat(&[vec![0u8; 8], b"soun".to_vec(), vec![0u8; 12], b"SoundHandler\0".to_vec()])
        };

        let dref_entry = if s.external_dref {
            // QuickTime alias record, flags 0: the data lives in another file.
            bx(b"alis", b"\0\0\0\0Macintosh HD\0/Users/someone/DCIM/IMG_0001.MOV\0")
        } else {
            bx(b"url ", &[0, 0, 0, 1])
        };
        let dinf = bx(b"dinf", &bx(b"dref", &cat(&[vec![0, 0, 0, 0, 0, 0, 0, 1], dref_entry])));

        let mut mp4a = vec![0u8; 6];
        mp4a.extend_from_slice(&1u16.to_be_bytes()); // data_reference_index
        mp4a.extend_from_slice(&[0u8; 8]); // version 0, revision, vendor
        mp4a.extend_from_slice(&2u16.to_be_bytes());
        mp4a.extend_from_slice(&16u16.to_be_bytes());
        mp4a.extend_from_slice(&[0u8; 4]);
        mp4a.extend_from_slice(&(48_000u32 << 16).to_be_bytes());
        mp4a.extend_from_slice(&bx(b"esds", ESDS));
        mp4a.extend_from_slice(&bx(b"btrt", &[0, 0, 0, 0, 0, 1, 0xF4, 0, 0, 1, 0xF4, 0]));
        let stsd = bx(b"stsd", &cat(&[vec![0, 0, 0, 0, 0, 0, 0, 1], bx(b"mp4a", &mp4a)]));

        let mut stbl = vec![stsd];
        stbl.extend(untouched_stbl_tables());
        if s.metadata_boxes {
            let mut xmp =
                vec![0xBE, 0x7A, 0xCF, 0xCB, 0x97, 0xA9, 0x42, 0xE8, 0x9C, 0x71, 0x99, 0x94, 0x91, 0xE3, 0xAF, 0xAC];
            xmp.extend_from_slice(b"<x:xmpmeta xmp:CreateDate=\"2026-05-02T11:54:26\"/>");
            stbl.push(bx(b"uuid", &xmp));
        }

        let mut minf = vec![bx(b"smhd", &[0u8; 8])];
        if s.writer_names {
            minf.push(bx(
                b"hdlr",
                &cat(&[b"\0\0\0\0dhlralisappl".to_vec(), vec![0u8; 8], b"\x17Core Media Data Handler".to_vec()]),
            ));
        }
        minf.push(dinf);
        if s.metadata_boxes {
            minf.push(bx(b"free", b"\xa9day2026-05-02T11:54:26+0100"));
        }
        minf.push(bx(b"stbl", &cat(&stbl)));

        let mut mdia = vec![bx(b"mdhd", &mdhd), bx(b"elng", b"\0\0\0\0und\0"), bx(b"hdlr", &media_hdlr)];
        if s.metadata_boxes {
            mdia.push(bx(b"udta", &bx(b"name", b"Audio from IMG_0001")));
        }
        mdia.push(bx(b"minf", &cat(&minf)));

        let mut trak = vec![bx(b"tkhd", &tkhd), bx(b"edts", &bx(b"elst", &elst))];
        if s.metadata_boxes {
            trak.push(bx(
                b"udta",
                &cat(&[
                    bx(b"\xa9day", b"\0\x182026-05-02T11:54:26+0100"),
                    bx(b"\xa9xyz", b"\0\x12+38.7130-009.1569/"),
                    bx(b"\xa9mak", b"\0\x05\0\0Apple"),
                ]),
            ));
        }
        trak.push(bx(b"tref", &bx(b"fall", &[0, 0, 0, 2]))); // not metadata: kept
        trak.push(bx(b"mdia", &cat(&mdia)));
        if s.metadata_boxes {
            trak.push(bx(
                b"meta",
                &cat(&[vec![0u8; 4], bx(b"hdlr", &cat(&[vec![0u8; 8], b"mdta".to_vec(), vec![0u8; 13]]))]),
            ));
            trak.push(bx(b"skip", b"com.apple.quicktime.creationdate"));
        }
        bx(b"trak", &cat(&trak))
    }

    fn times_of(full_box: &[u8]) -> (u64, u64) {
        if full_box[0] == 1 {
            (
                u64::from_be_bytes(full_box[4..12].try_into().unwrap()),
                u64::from_be_bytes(full_box[12..20].try_into().unwrap()),
            )
        } else {
            (
                u32::from_be_bytes(full_box[4..8].try_into().unwrap()) as u64,
                u32::from_be_bytes(full_box[8..12].try_into().unwrap()) as u64,
            )
        }
    }

    const VERSIONS: [(u8, u8); 4] = [(0, 0), (1, 1), (0, 1), (1, 0)];

    #[test]
    fn source_times_and_user_data_are_cleared_in_both_header_versions() {
        for (tkhd_version, mdhd_version) in VERSIONS {
            let source = Synth {
                tkhd_version,
                mdhd_version,
                times: CAPTURE_TIMES,
                metadata_boxes: true,
                writer_names: false,
                external_dref: false,
            };
            let input = iso_trak(source);
            assert_eq!(times_of(find(&input, &[b"trak", b"tkhd"]).unwrap()), CAPTURE_TIMES);
            assert!(find(&input, &[b"trak", b"udta"]).is_some());

            let out = passthrough_audio_trak_same(&input);
            let tkhd = find(&out, &[b"trak", b"tkhd"]).unwrap();
            let mdhd = find(&out, &[b"trak", b"mdia", b"mdhd"]).unwrap();
            assert_eq!(tkhd[0], tkhd_version, "tkhd version must not change");
            assert_eq!(mdhd[0], mdhd_version, "mdhd version must not change");
            assert_eq!(times_of(tkhd), (0, 0), "tkhd v{tkhd_version} times");
            assert_eq!(times_of(mdhd), (0, 0), "mdhd v{mdhd_version} times");
            assert!(find(&out, &[b"trak", b"udta"]).is_none(), "trak/udta left in");
            // Everything else, byte for byte: the same track built without
            // the times and the user data.
            assert_eq!(
                out,
                iso_trak(source.cleared()),
                "v{tkhd_version}/v{mdhd_version}: more than the source identity changed"
            );
        }
    }

    #[test]
    fn every_source_identifying_field_is_cleared() {
        for (tkhd_version, mdhd_version) in VERSIONS {
            let source = Synth {
                tkhd_version,
                mdhd_version,
                times: CAPTURE_TIMES,
                metadata_boxes: true,
                writer_names: true,
                external_dref: true,
            };
            let out = passthrough_audio_trak_same(&iso_trak(source));
            assert_eq!(out, iso_trak(source.cleared()), "v{tkhd_version}/v{mdhd_version}");
            for needle in [
                &b"Core Media"[..],
                b"appl",
                b"Apple",
                b"Macintosh HD",
                b"IMG_0001",
                b"2026-05-02",
                b"xmpmeta",
                b"creationdate",
            ] {
                assert!(
                    !out.windows(needle.len()).any(|w| w == needle),
                    "{:?} left in",
                    String::from_utf8_lossy(needle)
                );
            }
            for typ in SOURCE_METADATA_BOXES {
                assert!(!out.windows(4).any(|w| w == typ), "a {:?} box left in", String::from_utf8_lossy(typ));
            }
            // The capture time is nowhere in the track, in either width.
            assert!(!out.windows(4).any(|w| w == (CAPTURE_TIMES.0 as u32).to_be_bytes()));
            assert!(!out.windows(8).any(|w| w == CAPTURE_TIMES.0.to_be_bytes()));
        }
    }

    #[test]
    fn what_the_audio_needs_is_copied_byte_for_byte() {
        let source = Synth {
            tkhd_version: 0,
            mdhd_version: 0,
            times: CAPTURE_TIMES,
            metadata_boxes: true,
            writer_names: true,
            external_dref: true,
        };
        let input = iso_trak(source);
        let out = passthrough_audio_trak_same(&input);
        let stbl_in = find(&input, &[b"trak", b"mdia", b"minf", b"stbl"]).unwrap();
        let stbl_out = find(&out, &[b"trak", b"mdia", b"minf", b"stbl"]).unwrap();
        for typ in [b"stsd", b"stts", b"stsc", b"stsz", b"stco", b"sgpd", b"sbgp"] {
            assert_eq!(find(stbl_out, &[typ]), find(stbl_in, &[typ]), "{} changed", String::from_utf8_lossy(typ));
        }
        assert_eq!(find(&out, &[b"trak", b"edts"]), find(&input, &[b"trak", b"edts"]), "edit list changed");
        assert_eq!(find(&out, &[b"trak", b"tref"]), find(&input, &[b"trak", b"tref"]), "an unrecognised box changed");
        assert_eq!(find(&out, &[b"trak", b"mdia", b"elng"]), find(&input, &[b"trak", b"mdia", b"elng"]));
        let tkhd_in = find(&input, &[b"trak", b"tkhd"]).unwrap();
        let tkhd_out = find(&out, &[b"trak", b"tkhd"]).unwrap();
        assert_eq!(tkhd_out[..4], tkhd_in[..4]);
        assert_eq!(tkhd_out[12..], tkhd_in[12..], "track id, duration, volume, matrix must stay");
        let mdhd_in = find(&input, &[b"trak", b"mdia", b"mdhd"]).unwrap();
        let mdhd_out = find(&out, &[b"trak", b"mdia", b"mdhd"]).unwrap();
        assert_eq!(mdhd_out[12..], mdhd_in[12..], "timescale, duration, language must stay");
    }

    #[test]
    fn a_cleared_track_comes_back_unchanged() {
        for (tkhd_version, mdhd_version) in VERSIONS {
            let clean = iso_trak(Synth {
                tkhd_version,
                mdhd_version,
                times: (0, 0),
                metadata_boxes: false,
                writer_names: false,
                external_dref: false,
            });
            assert_eq!(passthrough_audio_trak_same(&clean), clean);
        }
    }

    #[test]
    fn the_source_identity_is_cleared_where_the_iso_rewrite_gives_up() {
        // A QuickTime track the ISO rewrite leaves alone (its mp4a has no
        // esds anywhere), with the capture time, a vendor code and user data.
        let mut trak = quicktime_trak();
        let at = trak.windows(4).position(|w| w == b"esds").unwrap();
        trak[at..at + 4].copy_from_slice(b"xxxx");
        let tkhd_at = trak.windows(4).position(|w| w == b"tkhd").unwrap() + 4;
        trak[tkhd_at + 4..tkhd_at + 8].copy_from_slice(&(CAPTURE_TIMES.0 as u32).to_be_bytes());
        trak[tkhd_at + 8..tkhd_at + 12].copy_from_slice(&(CAPTURE_TIMES.1 as u32).to_be_bytes());
        let mdhd_at = trak.windows(4).position(|w| w == b"mdhd").unwrap() + 4;
        trak[mdhd_at + 4..mdhd_at + 8].copy_from_slice(&(CAPTURE_TIMES.0 as u32).to_be_bytes());
        let mp4a_at = trak.windows(4).position(|w| w == b"mp4a").unwrap() + 4;
        trak[mp4a_at + 12..mp4a_at + 16].copy_from_slice(b"appl");
        let mut payload = boxes(&trak).unwrap()[0].payload.to_vec();
        payload.extend_from_slice(&bx(b"udta", &bx(b"\xa9day", b"\0\x182026-05-02T11:54:26+0100")));
        let trak = bx(b"trak", &payload);
        assert!(normalize(&trak).is_none(), "this track must be one the ISO rewrite cannot handle");

        let out = passthrough_audio_trak_same(&trak);
        assert_eq!(times_of(find(&out, &[b"trak", b"tkhd"]).unwrap()), (0, 0));
        assert_eq!(times_of(find(&out, &[b"trak", b"mdia", b"mdhd"]).unwrap()), (0, 0));
        assert!(find(&out, &[b"trak", b"udta"]).is_none());
        let hdlr = find(&out, &[b"trak", b"mdia", b"hdlr"]).unwrap();
        assert_eq!(hdlr, cat(&[vec![0u8; 8], b"soun".to_vec(), vec![0u8; 12], b"SoundHandler\0".to_vec()]));
        let minf = find(&out, &[b"trak", b"mdia", b"minf"]).unwrap();
        assert!(boxes(minf).unwrap().iter().all(|b| &b.typ != b"hdlr"), "data-handler hdlr left in");
        assert_eq!(
            find(minf, &[b"dinf"]).unwrap(),
            bx(b"dref", &cat(&[vec![0, 0, 0, 0, 0, 0, 0, 1], bx(b"url ", &[0, 0, 0, 1])]))
        );
        let stsd = find(&out, &[b"trak", b"mdia", b"minf", b"stbl", b"stsd"]).unwrap();
        let mp4a = boxes(&stsd[8..]).unwrap().remove(0).payload.to_vec();
        assert_eq!(&mp4a[12..16], &[0u8; 4], "vendor code left in");
        // The rest of the QuickTime entry is as it was.
        let stsd_in = find(&trak, &[b"trak", b"mdia", b"minf", b"stbl", b"stsd"]).unwrap();
        let mp4a_in = boxes(&stsd_in[8..]).unwrap().remove(0).payload.to_vec();
        assert_eq!(mp4a[..12], mp4a_in[..12]);
        assert_eq!(mp4a[16..], mp4a_in[16..]);
        for needle in [&b"Core Media"[..], b"appl", b"2026-05-02"] {
            assert!(!out.windows(needle.len()).any(|w| w == needle), "{:?} left in", String::from_utf8_lossy(needle));
        }
    }

    #[test]
    fn every_data_reference_becomes_self_contained() {
        let two = cat(&[
            vec![0, 0, 0, 0, 0, 0, 0, 2],
            bx(b"alis", &[0, 0, 0, 1]),
            bx(b"url ", b"\0\0\0\0file:///Users/someone/clip.mov\0"),
        ]);
        let out = self_contained_dinf(&bx(b"dref", &two));
        assert_eq!(
            out,
            bx(b"dref", &cat(&[vec![0, 0, 0, 0, 0, 0, 0, 2], bx(b"url ", &[0, 0, 0, 1]), bx(b"url ", &[0, 0, 0, 1])]))
        );
        // Nothing usable: one reference, which the sample entries' index 1 needs.
        let one = bx(b"dref", &cat(&[vec![0, 0, 0, 0, 0, 0, 0, 1], bx(b"url ", &[0, 0, 0, 1])]));
        assert_eq!(self_contained_dinf(&[]), one);
        assert_eq!(self_contained_dinf(b"garbage"), one);
    }

    #[test]
    fn a_track_that_cannot_be_cleared_is_refused() {
        // Truncated: the trak box claims more bytes than there are.
        assert!(passthrough_audio_trak(&quicktime_trak()[..40], 1000, 1000).is_err());
        // A tkhd version with no known layout.
        let mut v2 = iso_trak(Synth {
            tkhd_version: 0,
            mdhd_version: 0,
            times: CAPTURE_TIMES,
            metadata_boxes: false,
            writer_names: false,
            external_dref: false,
        });
        let at = v2.windows(4).position(|w| w == b"tkhd").unwrap() + 4;
        v2[at] = 2;
        assert!(passthrough_audio_trak(&v2, 600, 1000).is_err());
        // An mdhd too short to hold its times.
        let short = bx(b"trak", &bx(b"mdia", &bx(b"mdhd", &[0, 0, 0, 0, 0, 0])));
        assert!(passthrough_audio_trak(&short, 1000, 1000).is_err());
        // A child whose size runs past its parent.
        let mut overrun = iso_trak(Synth {
            tkhd_version: 1,
            mdhd_version: 1,
            times: CAPTURE_TIMES,
            metadata_boxes: true,
            writer_names: true,
            external_dref: true,
        });
        let at = overrun.windows(4).position(|w| w == b"minf").unwrap() - 4;
        overrun[at..at + 4].copy_from_slice(&0x7FFF_FFFFu32.to_be_bytes());
        assert!(passthrough_audio_trak(&overrun, 1000, 1000).is_err());
    }

    #[test]
    fn a_short_zero_terminator_is_kept_and_the_rest_still_cleared() {
        // QuickTime may end a container with a 32-bit zero; the demuxer
        // accepts it, so the scrub must too.
        let source = Synth {
            tkhd_version: 0,
            mdhd_version: 0,
            times: CAPTURE_TIMES,
            metadata_boxes: true,
            writer_names: true,
            external_dref: false,
        };
        let mut payload = boxes(&iso_trak(source)).unwrap()[0].payload.to_vec();
        payload.extend_from_slice(&[0, 0, 0, 0]);
        let out = passthrough_audio_trak_same(&bx(b"trak", &payload));
        let mut expected = boxes(&iso_trak(source.cleared())).unwrap()[0].payload.to_vec();
        expected.extend_from_slice(&[0, 0, 0, 0]);
        assert_eq!(out, bx(b"trak", &expected));
    }
}
