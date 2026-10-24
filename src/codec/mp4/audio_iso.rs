// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Rewrite a passed-through audio `trak` as plain ISO-BMFF.
//!
//! The stego MP4 builders copy the source's audio track verbatim, patching
//! only `stco` and the track id. From an iPhone `.mov` that track is
//! QuickTime-flavoured:
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

/// One box: its four-character type, and its payload (everything after the
/// header).
struct Box<'a> {
    typ: [u8; 4],
    payload: &'a [u8],
}

/// Split `buf` into its top-level boxes. `None` if a size does not fit.
fn boxes(buf: &[u8]) -> Option<Vec<Box<'_>>> {
    let mut out = Vec::new();
    let mut off = 0usize;
    while off < buf.len() {
        if buf.len() - off < 8 {
            return None;
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
    Some(out)
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
/// output's). Each step that meets a layout it does not understand leaves
/// that part as the source had it.
pub(crate) fn passthrough_audio_trak(
    trak: &[u8],
    from_movie_timescale: u32,
    to_movie_timescale: u32,
) -> Vec<u8> {
    let iso = normalize(trak).unwrap_or_else(|| trak.to_vec());
    rescale_movie_time(&iso, from_movie_timescale, to_movie_timescale).unwrap_or(iso)
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The ISO normalization alone (same movie timescale on both sides).
    fn passthrough_audio_trak_same(trak: &[u8]) -> Vec<u8> {
        passthrough_audio_trak(trak, 1000, 1000)
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
        let out = passthrough_audio_trak(&quicktime_trak(), 600, 1000);
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
        let truncated = &quicktime_trak()[..40];
        assert_eq!(passthrough_audio_trak_same(truncated), truncated);
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
}
