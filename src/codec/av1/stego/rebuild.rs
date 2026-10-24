// Copyright (c) 2026 Christoph Gaffga
// SPDX-License-Identifier: GPL-3.0-only
// https://github.com/cgaffga/phasmcore

//! Rebuild a GOP's Pass-2 replay data from its natural packets.
//!
//! The encode's Pass 2 embeds each GOP by replaying its range-coder
//! symbols with overrides ([`super::writer::replay_with_overrides`]), which
//! needs per frame what phasm-rav1e's `WriterRecorder` records at encode
//! time: the `(fl, fh, nms)` storage tuples, the 50/50 bit positions with
//! their tags and AC-sign meta, the reconstructed planes and the frame's
//! qindex / loop-filter state for the J-UNIWARD harvest, and the OBU splice
//! offsets. Holding that between the passes costs 300–400 MB per 1080p GOP;
//! re-encoding it costs a full rav1e encode.
//!
//! [`rebuild_gop_recordings`] gets the same data from one dav1d decode of
//! the GOP's natural packets (~0.3 s per 1080p GOP against ~45 s to
//! re-encode), through phasm-dav1d's `sym_hook` / `mag_hook` (see
//! `docs/design/video/av1/dav1d-hook-sites.md` § 11). It is bit-exact with
//! the encoder's recording — measured over 86 M symbols — and checks itself:
//! each frame's rebuilt symbols, replayed without overrides, must reproduce
//! the packet's tile bytes, or the rebuild fails instead of handing Pass 2
//! data the encoder never wrote.

use std::sync::Arc;

use core_dav1d_sys::{
    dav1d_close, dav1d_data_create, dav1d_data_unref, dav1d_default_settings, dav1d_err_again,
    dav1d_get_picture, dav1d_open, dav1d_picture_unref, dav1d_send_data,
    phasm_dav1d_frame_lf_state, Dav1dContext, Dav1dData, Dav1dPhasmAcSignMeta, Dav1dPhasmHooks,
    Dav1dPicture, Dav1dSettings, PhasmDav1dFrameLf,
};
use phasm_rav1e::color::ChromaSampling;
use phasm_rav1e::ec::{
    PhasmFrameLoopFilterState, PhasmFrameRecording, PhasmTileRecording, WriterEncoder,
};
use phasm_rav1e::phasm_stego::{
    make_frame, AcSignMeta, PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB,
};

use super::decoder::{copy_plane, Av1DecodeError, Dav1dPictureView};
use super::orchestrator::{decode_uleb128, Av1StegoError};
use super::writer::{replay_with_overrides, OverrideMap};

/// AV1 `OBU_FRAME`: a frame header followed by its tile group.
const OBU_FRAME: u8 = 6;

/// Bound on send/receive rounds per packet; dav1d only answers EAGAIN on
/// both sides once the packet is consumed, so this only stops a misbehaving
/// decoder from spinning.
const MAX_ROUNDS_PER_PACKET: usize = 64;

/// One frame's hook output, gathered while dav1d decodes it.
#[derive(Default)]
struct SymbolRecorder {
    storage: Vec<(u16, u16, u16)>,
    bit_positions: Vec<(u32, u16)>,
    bit_tags: Vec<u8>,
    bit_meta: Vec<AcSignMeta>,
    /// Sticky meta from `meta_hook`, as the encoder's `phasm_current_meta`.
    current_meta: AcSignMeta,
    /// Enrolled entries of the coefficient being decoded, waiting for its
    /// full magnitude from `mag_hook` (the golomb tail follows the sign).
    pending_mag: Vec<usize>,
    /// `tracked` symbols seen; each must be followed by one `bit_hook`.
    tracked: usize,
    /// First broken hook invariant, reported when the frame closes (the
    /// callbacks run inside C and must not panic).
    fault: Option<&'static str>,
}

/// # Safety
/// `cookie` is the `*mut SymbolRecorder` registered by
/// [`rebuild_gop_recordings`]; dav1d runs single-threaded and calls hooks
/// only inside its synchronous `send_data` / `get_picture`.
unsafe extern "C" fn sym_cb(
    cookie: *mut core::ffi::c_void,
    fl: u16,
    fh: u16,
    nms: u16,
    tracked: core::ffi::c_uint,
) {
    let r = unsafe { &mut *(cookie as *mut SymbolRecorder) };
    r.storage.push((fl, fh, nms));
    if tracked != 0 {
        r.tracked += 1;
    }
}

/// Fires right after `sym_cb` for every tracked (`bool_equi`) symbol.
///
/// # Safety
/// As [`sym_cb`].
unsafe extern "C" fn bit_cb(cookie: *mut core::ffi::c_void, bit: core::ffi::c_uint, tag: u8) {
    let r = unsafe { &mut *(cookie as *mut SymbolRecorder) };
    let Some(idx) = r.storage.len().checked_sub(1) else {
        r.fault.get_or_insert("bit_hook before any symbol");
        return;
    };
    r.bit_positions.push((idx as u32, bit as u16));
    r.bit_tags.push(tag);
    r.bit_meta.push(r.current_meta);
    if tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB {
        r.pending_mag.push(r.bit_meta.len() - 1);
    }
}

/// # Safety
/// As [`sym_cb`]; `meta` points into dav1d's stack for this call only.
unsafe extern "C" fn meta_cb(cookie: *mut core::ffi::c_void, meta: *const Dav1dPhasmAcSignMeta) {
    let r = unsafe { &mut *(cookie as *mut SymbolRecorder) };
    if meta.is_null() {
        r.fault.get_or_insert("null meta");
        return;
    }
    let m = unsafe { *meta };
    r.current_meta = AcSignMeta {
        plane: m.plane,
        plane_px_x: m.plane_px_x,
        plane_px_y: m.plane_px_y,
        tx_width_log2: m.tx_width_log2,
        tx_height_log2: m.tx_height_log2,
        tx_type: m.tx_type,
        scan_pos: m.scan_pos,
        coeff_magnitude: 0,
    };
}

/// # Safety
/// As [`sym_cb`].
unsafe extern "C" fn mag_cb(cookie: *mut core::ffi::c_void, magnitude: core::ffi::c_uint) {
    let r = unsafe { &mut *(cookie as *mut SymbolRecorder) };
    // Saturated as the encoder records it.
    let m = magnitude.min(u16::MAX as u32) as u16;
    for i in r.pending_mag.drain(..) {
        r.bit_meta[i].coeff_magnitude = m;
    }
}

/// A decoded frame before its OBU layout is known.
struct DecodedFrame {
    tile: PhasmTileRecording,
    planes: phasm_rav1e::Frame<u8>,
    lf: PhasmDav1dFrameLf,
}

/// Owns the hook recorder for one decode. dav1d writes to it through the
/// cookie from inside its calls, so Rust reaches it only through that same
/// raw pointer — never through a `&mut` over a `Box`, which a later reborrow
/// would invalidate while C still holds the cookie — and frees it on drop.
struct RecorderGuard(*mut SymbolRecorder);

impl Drop for RecorderGuard {
    fn drop(&mut self) {
        // SAFETY: from `Box::into_raw` in `decode_recording`, freed exactly
        // once here, after the context that held the cookie has closed.
        drop(unsafe { Box::from_raw(self.0) });
    }
}

/// Closes the dav1d context on every exit path.
struct ContextGuard(*mut Dav1dContext);

impl Drop for ContextGuard {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: opened by `dav1d_open`, closed exactly once here.
            unsafe { dav1d_close(&mut self.0) };
        }
    }
}

/// Releases a packet's dav1d data buffer on every exit path (a no-op once
/// dav1d has consumed it).
struct DataGuard(Dav1dData);

impl Drop for DataGuard {
    fn drop(&mut self) {
        // SAFETY: zero-initialised or filled by `dav1d_data_create`.
        unsafe { dav1d_data_unref(&mut self.0) };
    }
}

fn invalid(what: impl std::fmt::Display) -> Av1StegoError {
    Av1StegoError::InvalidPacket(format!("rebuild: {what}"))
}

/// Every rebuilt 50/50 bit position must name the symbol the embed would
/// flip: the `bool_equi` tuple for its recorded natural value — `(32768,
/// 16384, 2)` for 0, `(16384, 0, 1)` for 1, as the fork's hooked decoder
/// reports them. The replay check in [`tile_layout`] proves `storage`; this
/// proves the positions laid over it, so Pass 2 never overrides a
/// context-coded symbol (which would desync every symbol after it) or
/// against a wrong natural bit. The parallel tag and meta vectors must
/// match the positions one to one.
fn check_bit_positions(tile: &PhasmTileRecording) -> Result<(), Av1StegoError> {
    let n = tile.bit_positions.len();
    if tile.bit_tags.len() != n || tile.bit_meta.len() != n {
        return Err(invalid(format_args!(
            "{n} bit positions, {} tags, {} metas",
            tile.bit_tags.len(),
            tile.bit_meta.len()
        )));
    }
    for (i, &(idx, value)) in tile.bit_positions.iter().enumerate() {
        let expected = match value {
            0 => (32768, 16384, 2),
            1 => (16384, 0, 1),
            _ => return Err(invalid(format_args!("bit position {i} does not point at its 50/50 symbol"))),
        };
        if tile.storage.get(idx as usize) != Some(&expected) {
            return Err(invalid(format_args!("bit position {i} does not point at its 50/50 symbol")));
        }
    }
    Ok(())
}

/// Rebuild what `encode_gop_natural` returned for a GOP — per frame the
/// natural packet and its [`PhasmFrameRecording`] — from the GOP's natural
/// packets alone, with one dav1d decode.
///
/// `width`×`height` is the encode's visible frame size; the recordings'
/// `reconstructed_planes` hold the decoded visible pixels (the alignment
/// padding is left as `make_frame` creates it, which the cost input
/// replaces anyway — `orchestrator::pack_cost_planes`).
///
/// Fails rather than returning data the encoder never wrote: on a dav1d
/// error, a picture count other than one per packet, a broken hook
/// invariant, a 50/50 position that does not name its symbol
/// ([`check_bit_positions`]), or a frame whose rebuilt symbols do not
/// replay to its tile bytes ([`tile_layout`]).
///
/// Keeps every frame's planes (tests compare them); Pass 2 uses
/// [`rebuild_gop_streamed`].
#[cfg(test)]
pub(super) fn rebuild_gop_recordings(
    packets: &[Vec<u8>],
    width: u32,
    height: u32,
) -> Result<Vec<(Vec<u8>, PhasmFrameRecording<u8>)>, Av1StegoError> {
    rebuild_gop_streamed(packets, width, height, &mut |_, _| Ok(()))
}

/// [`rebuild_gop_recordings`], handing each frame to `on_frame(k, entry)`
/// the moment it is decoded and checked — before the next frame's packet
/// is decoded — so the caller can take what it needs from the planes and
/// drop them. Pass 2 harvests each frame there: the decoder never holds
/// more than one frame of pictures, and the harvest works on one frame at
/// a time (a whole GOP's at once peaked at 1.2–1.4 GB at 1080p and ~2.5 GB
/// at 2720×1530). The returned frames are whatever `on_frame` left.
pub(super) fn rebuild_gop_streamed(
    packets: &[Vec<u8>],
    width: u32,
    height: u32,
    on_frame: &mut dyn FnMut(usize, &mut (Vec<u8>, PhasmFrameRecording<u8>)) -> Result<(), Av1StegoError>,
) -> Result<Vec<(Vec<u8>, PhasmFrameRecording<u8>)>, Av1StegoError> {
    let mut out: Vec<(Vec<u8>, PhasmFrameRecording<u8>)> = Vec::with_capacity(packets.len());
    decode_recording(packets, width as usize, height as usize, &mut |frame| {
        let k = out.len();
        let packet = packets
            .get(k)
            .ok_or_else(|| invalid(format_args!("more pictures than the {} packets", packets.len())))?;
        let mut entry = finish_frame(packet, frame)?;
        on_frame(k, &mut entry)?;
        out.push(entry);
        Ok(())
    })?;
    if out.len() != packets.len() {
        return Err(invalid(format_args!("{} pictures from {} packets", out.len(), packets.len())));
    }
    Ok(out)
}

/// One decoded frame's recording, checked against its natural `packet`.
fn finish_frame(
    packet: &[u8],
    frame: DecodedFrame,
) -> Result<(Vec<u8>, PhasmFrameRecording<u8>), Av1StegoError> {
    check_bit_positions(&frame.tile)?;
    let mut sink = WriterEncoder::new();
    replay_with_overrides(&frame.tile.storage, &frame.tile.bit_positions, &OverrideMap::new(), &mut sink);
    let (frame_obu_start, tile_group_offset, tile_group_len, frame_header_len) =
        tile_layout(packet, &sink.done())?;
    let lf = frame.lf;
    Ok((
        packet.to_vec(),
        PhasmFrameRecording {
            tiles: vec![frame.tile],
            tile_group_offset,
            tile_group_len,
            frame_header_len,
            frame_obu_start,
            reconstructed_planes: Arc::new(frame.planes),
            frame_qindex: lf.base_q_idx,
            loop_filter_state: PhasmFrameLoopFilterState {
                deblock_levels: lf.deblock_levels,
                cdef_y_strengths: lf.cdef_y_strengths,
                cdef_uv_strengths: lf.cdef_uv_strengths,
                cdef_bits: lf.cdef_bits,
                cdef_enabled: lf.cdef_enabled != 0,
            },
        },
    ))
}

/// Decode `packets` with the recording hooks, closing one frame record per
/// output picture and handing it to `sink` right away (between dav1d calls,
/// after the picture is released), so no picture outlives its frame.
fn decode_recording(
    packets: &[Vec<u8>],
    width: usize,
    height: usize,
    sink: &mut dyn FnMut(DecodedFrame) -> Result<(), Av1StegoError>,
) -> Result<(), Av1StegoError> {
    // Declared before the context guard, so the context closes (and stops
    // using the cookie) before the recorder is freed.
    let rec = RecorderGuard(Box::into_raw(Box::new(SymbolRecorder::default())));
    let eagain = dav1d_err_again();
    // SAFETY: dav1d runs single-threaded with one frame of delay, so every
    // hook fires inside the synchronous calls below while `rec` is alive,
    // and every picture is unreferenced before the next call.
    unsafe {
        let mut settings: Dav1dSettings = core::mem::zeroed();
        dav1d_default_settings(&mut settings);
        settings.n_threads = 1;
        settings.max_frame_delay = 1;
        settings.phasm_hooks = Dav1dPhasmHooks {
            cookie: rec.0 as *mut core::ffi::c_void,
            bit_hook: Some(bit_cb),
            tag_hook: None,
            meta_hook: Some(meta_cb),
            block_hook: None,
            sym_hook: Some(sym_cb),
            mag_hook: Some(mag_cb),
        };
        let mut ctx = ContextGuard(core::ptr::null_mut());
        let rc = dav1d_open(&mut ctx.0, &settings);
        if rc != 0 {
            return Err(Av1StegoError::Decode(Av1DecodeError::DavxdOpenFailed(rc)));
        }

        let take_pictures = |sink: &mut dyn FnMut(DecodedFrame) -> Result<(), Av1StegoError>| -> Result<(), Av1StegoError> {
            loop {
                let mut pic = Dav1dPicture::default();
                let rc = dav1d_get_picture(ctx.0, &mut pic);
                if rc == eagain {
                    return Ok(());
                }
                if rc != 0 {
                    return Err(Av1StegoError::Decode(Av1DecodeError::DavxdGetPictureFailed(rc)));
                }
                let frame = close_frame(rec.0, &pic, width, height);
                dav1d_picture_unref(&mut pic);
                // dav1d is between calls and the recorder is not borrowed:
                // the sink may run any Rust code.
                sink(frame?)?;
            }
        };

        for packet in packets {
            let mut data = DataGuard(core::mem::zeroed());
            let buf = dav1d_data_create(&mut data.0, packet.len());
            if buf.is_null() {
                return Err(Av1StegoError::Decode(Av1DecodeError::DataCreateFailed));
            }
            core::ptr::copy_nonoverlapping(packet.as_ptr(), buf, packet.len());
            let mut rounds = 0usize;
            while data.0.sz > 0 {
                rounds += 1;
                if rounds > MAX_ROUNDS_PER_PACKET {
                    return Err(invalid("dav1d made no progress on a packet"));
                }
                let rc = dav1d_send_data(ctx.0, &mut data.0);
                if rc != 0 && rc != eagain {
                    return Err(Av1StegoError::Decode(Av1DecodeError::DavxdSendDataFailed(rc)));
                }
                take_pictures(sink)?;
            }
        }
        take_pictures(sink)?;
    }
    Ok(())
}

/// Take the frame `rec` gathered for `pic`, leaving `rec` empty for the
/// next one.
///
/// # Safety
/// `pic` is a picture `dav1d_get_picture` just returned, not yet unref'd;
/// `rec` is the live recorder of [`decode_recording`], and dav1d is not
/// inside a call (so nothing else is using it).
unsafe fn close_frame(
    rec: *mut SymbolRecorder,
    pic: &Dav1dPicture,
    width: usize,
    height: usize,
) -> Result<DecodedFrame, Av1StegoError> {
    // SAFETY: see above — a short-lived exclusive view taken from the
    // cookie pointer itself while dav1d is between calls.
    let rec = unsafe { &mut *rec };
    let tracked = std::mem::take(&mut rec.tracked);
    let pending = std::mem::take(&mut rec.pending_mag);
    let tile = PhasmTileRecording {
        storage: std::mem::take(&mut rec.storage),
        bit_positions: std::mem::take(&mut rec.bit_positions),
        bit_tags: std::mem::take(&mut rec.bit_tags),
        bit_meta: std::mem::take(&mut rec.bit_meta),
    };
    if let Some(fault) = rec.fault.take() {
        return Err(invalid(fault));
    }
    if !pending.is_empty() {
        return Err(invalid("coefficient magnitude missing"));
    }
    if tracked != tile.bit_positions.len() {
        return Err(invalid("50/50 symbols and bit hooks disagree"));
    }

    let view = unsafe { &*(pic as *const Dav1dPicture as *const Dav1dPictureView) };
    check_picture_shape(view.p_w, view.p_h, view.p_layout, view.p_bpc, width, height)?;
    let (cw, ch) = (width.div_ceil(2), height.div_ceil(2));
    // SAFETY: `pic` is live; its plane pointers and strides cover the
    // visible width × height (chroma at half, rounded up).
    let (y, u, v) = unsafe {
        (
            copy_plane(view.data[0] as *const u8, view.stride[0], width, height),
            copy_plane(view.data[1] as *const u8, view.stride[1], cw, ch),
            copy_plane(view.data[2] as *const u8, view.stride[1], cw, ch),
        )
    };
    let mut planes = make_frame::<u8>(width, height, ChromaSampling::Cs420);
    planes.planes[0].copy_from_raw_u8(&y, width, 1);
    planes.planes[1].copy_from_raw_u8(&u, cw, 1);
    planes.planes[2].copy_from_raw_u8(&v, cw, 1);

    let mut lf = PhasmDav1dFrameLf::default();
    // SAFETY: `pic` is live; the shim only reads its headers.
    if unsafe { phasm_dav1d_frame_lf_state(pic as *const Dav1dPicture as *const _, &mut lf) } != 0 {
        return Err(invalid("picture without frame headers"));
    }
    Ok(DecodedFrame { tile, planes, lf })
}

/// The rebuild reads only what this encode makes: an 8-bit I420 picture
/// (`Dav1dPixelLayout` 1) of the encode's `width`×`height`.
fn check_picture_shape(
    w: i32,
    h: i32,
    layout: i32,
    bpc: i32,
    width: usize,
    height: usize,
) -> Result<(), Av1StegoError> {
    if bpc != 8 {
        return Err(invalid(format_args!("{bpc}-bit picture; the encode is 8-bit")));
    }
    if layout != 1 || (w as usize, h as usize) != (width, height) {
        return Err(invalid(format_args!(
            "picture {w}x{h} layout {layout} for a {width}x{height} I420 encode"
        )));
    }
    Ok(())
}

/// Where `packet`'s tile group sits, given the tile bytes a replay of the
/// rebuilt symbols produced: `(frame_obu_start, tile_group_offset,
/// tile_group_len, frame_header_len)`, the fields
/// `orchestrator::rebuild_obu_with_stego_tile_group` splices with.
///
/// rav1e writes one tile per frame and the frame OBU last, so the tile
/// group is the end of that OBU's payload — and must be byte-identical to
/// `replayed_tile`. A mismatch (several tiles, a fork or encoder drift, a
/// damaged packet) is an error: Pass 2 must never embed into symbols that
/// do not encode the packet.
fn tile_layout(
    packet: &[u8],
    replayed_tile: &[u8],
) -> Result<(usize, usize, usize, usize), Av1StegoError> {
    let mut pos = 0usize;
    let mut frame_obu = None;
    while pos < packet.len() {
        let header = packet[pos];
        if (header >> 1) & 1 == 0 {
            return Err(invalid("OBU without a size field"));
        }
        let ext = ((header >> 2) & 1) as usize;
        let (size, leb_len) =
            decode_uleb128(packet, pos + 1 + ext).ok_or_else(|| invalid("bad OBU size"))?;
        let payload_start = pos + 1 + ext + leb_len;
        let end = usize::try_from(size)
            .ok()
            .and_then(|s| payload_start.checked_add(s))
            .filter(|&e| e <= packet.len())
            .ok_or_else(|| invalid("OBU overruns the packet"))?;
        if (header >> 3) & 0xf == OBU_FRAME {
            frame_obu = Some((pos, payload_start, end));
        }
        pos = end;
    }
    let (obu_start, payload_start, end) = frame_obu.ok_or_else(|| invalid("no frame OBU"))?;
    if end != packet.len() {
        return Err(invalid("the frame OBU is not the packet's last OBU"));
    }
    let len = replayed_tile.len();
    if len > end - payload_start || packet[end - len..end] != *replayed_tile {
        return Err(invalid("replayed tile does not reproduce the natural tile bytes"));
    }
    Ok((obu_start, end - len, len, end - len - payload_start))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::av1::stego::orchestrator::pack_visible_planes;
    use crate::codec::av1::stego::session::Av1StreamingEncodeParams;
    use crate::codec::av1::stego::whole_video::{
        cascade_one_gop, encode_gop_natural, harvest_gop, noise_gop_sized, Av1CascadeGopInputs,
        Av1CascadeGopShared,
    };
    use phasm_rav1e::phasm_stego::{PHASM_TAG_AC_COEFF_SIGN, PHASM_TAG_GOLOMB_TAIL_LSB};

    type Frames = Vec<(Vec<u8>, PhasmFrameRecording<u8>)>;

    fn enrolled(tag: u8) -> bool {
        tag == PHASM_TAG_AC_COEFF_SIGN || tag == PHASM_TAG_GOLOMB_TAIL_LSB
    }

    /// Visible `w`×`h` region of a plane buffer packed `pw` wide.
    fn visible(buf: &[u8], pw: usize, w: usize, h: usize) -> Vec<u8> {
        (0..h).flat_map(|r| buf[r * pw..r * pw + w].iter().copied()).collect()
    }

    /// Every field Pass 2 reads, frame by frame; `Err` names the first
    /// difference.
    fn compare_frames(enc: &Frames, dec: &Frames, w: usize, h: usize) -> Result<(), String> {
        if enc.len() != dec.len() {
            return Err(format!("{} encoded frames, {} rebuilt", enc.len(), dec.len()));
        }
        for (i, ((ep, er), (dp, dr))) in enc.iter().zip(dec).enumerate() {
            let (et, dt) = (&er.tiles[0], &dr.tiles[0]);
            let checks: [(&str, bool); 12] = [
                ("packet", ep == dp),
                ("tile count", er.tiles.len() == 1 && dr.tiles.len() == 1),
                ("storage", et.storage == dt.storage),
                ("bit_positions", et.bit_positions == dt.bit_positions),
                ("bit_tags", et.bit_tags == dt.bit_tags),
                (
                    "bit_meta",
                    et.bit_tags
                        .iter()
                        .zip(et.bit_meta.iter().zip(&dt.bit_meta))
                        .all(|(&t, (a, b))| !enrolled(t) || a == b),
                ),
                ("frame_qindex", er.frame_qindex == dr.frame_qindex),
                (
                    "loop filter",
                    er.loop_filter_state.deblock_levels == dr.loop_filter_state.deblock_levels
                        && er.loop_filter_state.cdef_enabled == dr.loop_filter_state.cdef_enabled
                        && er.loop_filter_state.cdef_y_strengths[0]
                            == dr.loop_filter_state.cdef_y_strengths[0]
                        && er.loop_filter_state.cdef_uv_strengths[0]
                            == dr.loop_filter_state.cdef_uv_strengths[0],
                ),
                ("tile_group_offset", er.tile_group_offset == dr.tile_group_offset),
                ("tile_group_len", er.tile_group_len == dr.tile_group_len),
                (
                    "frame header / OBU start",
                    er.frame_header_len == dr.frame_header_len
                        && er.frame_obu_start == dr.frame_obu_start,
                ),
                ("visible planes", {
                    let (a, b) = (
                        pack_visible_planes(&er.reconstructed_planes),
                        pack_visible_planes(&dr.reconstructed_planes),
                    );
                    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
                    visible(&a.y, a.luma_width, w, h) == visible(&b.y, b.luma_width, w, h)
                        && visible(&a.cb, a.chroma_width, cw, ch)
                            == visible(&b.cb, b.chroma_width, cw, ch)
                        && visible(&a.cr, a.chroma_width, cw, ch)
                            == visible(&b.cr, b.chroma_width, cw, ch)
                }),
            ];
            if let Some((name, _)) = checks.iter().find(|(_, ok)| !ok) {
                return Err(format!("frame {i}: {name} differs"));
            }
        }
        Ok(())
    }

    /// `cascade_one_gop` output for a chunk of `frac` × cover bits / 8.
    fn embed(frames: &Frames, w: u32, h: u32, frac: f64) -> Option<Vec<u8>> {
        let harvest = harvest_gop(frames, w, h).expect("harvest");
        let n = harvest.cover_len();
        let chunk: Vec<u8> =
            (0..((n as f64 * frac) / 8.0) as usize).map(|i| (i * 131 + 7) as u8).collect();
        cascade_one_gop(
            Av1CascadeGopInputs {
                gop_idx: 0,
                gop_offset: 0,
                gop_n: n,
                harvest: &harvest,
                frames,
                chunk: &chunk,
            },
            Av1CascadeGopShared {
                total_message_bytes: chunk.len() as u32,
                shadow_states: &[],
                hhat_seed: &[9u8; 32],
            },
        )
        .expect("cascade")
    }

    /// The rebuilt GOP hands Pass 2 exactly what the encoder recorded:
    /// every field, the harvest bit for bit, and the embedded bytes.
    fn assert_rebuild_matches(yuv: &[u8], params: Av1StreamingEncodeParams, n: u32) {
        let (w, h) = (params.width, params.height);
        let enc = encode_gop_natural(yuv, n, params).expect("encode");
        let packets: Vec<Vec<u8>> = enc.iter().map(|(p, _)| p.clone()).collect();
        let dec = rebuild_gop_recordings(&packets, w, h).expect("rebuild");
        compare_frames(&enc, &dec, w as usize, h as usize).unwrap();
        let (he, hd) = (harvest_gop(&enc, w, h).unwrap(), harvest_gop(&dec, w, h).unwrap());
        assert!(he.cover_len() > 0, "fixture must have cover");
        assert!(he.bit_identical(&hd), "harvests differ");
        for frac in [0.01, 0.05, 0.08] {
            let a = embed(&enc, w, h, frac);
            assert!(a.is_some(), "{frac}: infeasible on the encoder side");
            assert_eq!(a, embed(&dec, w, h, frac), "{frac}: embedded bytes differ");
        }
    }

    #[test]
    fn rebuild_matches_encoder_recording_aligned() {
        let (yuv, params) = noise_gop_sized(256, 144, 4);
        assert_rebuild_matches(&yuv, params, 4);
    }

    /// 250×142: rav1e aligns to 256×144, so the padding differs between the
    /// two sides; the cost input replicates the visible edge on both.
    #[test]
    fn rebuild_matches_encoder_recording_unaligned() {
        let (yuv, params) = noise_gop_sized(250, 142, 4);
        assert_rebuild_matches(&yuv, params, 4);
    }

    /// The rebuild only accepts what this encode makes: an 8-bit 4:2:0
    /// picture of the encode's size.
    #[test]
    fn check_picture_shape_accepts_only_the_encodes_pictures() {
        assert!(check_picture_shape(256, 144, 1, 8, 256, 144).is_ok());
        for (w, h, layout, bpc) in [(256, 144, 1, 10), (250, 144, 1, 8), (256, 142, 1, 8), (256, 144, 3, 8)] {
            assert!(
                matches!(
                    check_picture_shape(w, h, layout, bpc, 256, 144),
                    Err(Av1StegoError::InvalidPacket(_))
                ),
                "{w}x{h} layout {layout} {bpc}-bit accepted"
            );
        }
    }

    #[test]
    fn rebuild_rejects_truncated_packet() {
        let (yuv, params) = noise_gop_sized(256, 144, 4);
        let mut packets: Vec<Vec<u8>> = encode_gop_natural(&yuv, 4, params)
            .expect("encode")
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        let len = packets[1].len();
        packets[1].truncate(len - 16);
        assert!(rebuild_gop_recordings(&packets, 256, 144).is_err());
    }

    /// A rebuilt 256×144 noise GOP's tiles.
    fn rebuilt_tiles() -> Vec<PhasmTileRecording> {
        let (yuv, params) = noise_gop_sized(256, 144, 2);
        let packets: Vec<Vec<u8>> = encode_gop_natural(&yuv, 2, params)
            .expect("encode")
            .into_iter()
            .map(|(p, _)| p)
            .collect();
        rebuild_gop_recordings(&packets, 256, 144)
            .expect("rebuild")
            .into_iter()
            .map(|(_, mut r)| r.tiles.remove(0))
            .collect()
    }

    #[test]
    fn check_bit_positions_accepts_a_rebuilt_tile() {
        let tiles = rebuilt_tiles();
        assert!(tiles.iter().all(|t| !t.bit_positions.is_empty()), "fixture must have 50/50 symbols");
        for (f, tile) in tiles.iter().enumerate() {
            assert!(check_bit_positions(tile).is_ok(), "frame {f}");
        }
    }

    /// The embed flips the symbols `bit_positions` names; one that names a
    /// context-coded symbol, or records the wrong natural value, must be
    /// caught before Pass 2 embeds into it.
    #[test]
    fn check_bit_positions_rejects_a_shifted_or_flipped_position() {
        let tile = rebuilt_tiles().remove(0);
        let half = |t: (u16, u16, u16)| t == (32768, 16384, 2) || t == (16384, 0, 1);
        // An entry whose next symbol is not a 50/50 one.
        let (i, idx) = tile
            .bit_positions
            .iter()
            .enumerate()
            .map(|(i, &(idx, _))| (i, idx as usize))
            .find(|&(_, idx)| idx + 1 < tile.storage.len() && !half(tile.storage[idx + 1]))
            .expect("a 50/50 symbol followed by a context-coded one");

        let mut shifted = tile.clone();
        shifted.bit_positions[i].0 = (idx + 1) as u32;
        assert!(matches!(check_bit_positions(&shifted), Err(Av1StegoError::InvalidPacket(_))), "shifted");

        let mut flipped = tile.clone();
        flipped.bit_positions[i].1 ^= 1;
        assert!(matches!(check_bit_positions(&flipped), Err(Av1StegoError::InvalidPacket(_))), "flipped");

        let mut past_end = tile.clone();
        past_end.bit_positions[i].0 = tile.storage.len() as u32;
        assert!(matches!(check_bit_positions(&past_end), Err(Av1StegoError::InvalidPacket(_))), "past end");

        let mut short_tags = tile.clone();
        short_tags.bit_tags.pop();
        assert!(matches!(check_bit_positions(&short_tags), Err(Av1StegoError::InvalidPacket(_))), "tags");
    }

    #[test]
    fn tile_layout_rejects_mismatched_replay() {
        let (yuv, params) = noise_gop_sized(256, 144, 1);
        let (packet, rec) = encode_gop_natural(&yuv, 1, params).expect("encode").remove(0);
        let tile = &packet[rec.tile_group_offset..rec.tile_group_offset + rec.tile_group_len];
        assert_eq!(
            tile_layout(&packet, tile).expect("natural tile"),
            (rec.frame_obu_start, rec.tile_group_offset, rec.tile_group_len, rec.frame_header_len)
        );
        let mut flipped = tile.to_vec();
        let mid = flipped.len() / 2;
        flipped[mid] ^= 0x10;
        assert!(matches!(tile_layout(&packet, &flipped), Err(Av1StegoError::InvalidPacket(_))));
    }

    fn env_u32(k: &str, d: u32) -> u32 {
        std::env::var(k).ok().and_then(|v| v.parse().ok()).unwrap_or(d)
    }

    /// Real-content parity, for fork and encoder bumps. Env:
    /// `PHASM_O3_CLIP` (default IMG_4138), `PHASM_O3_W` / `_H` (256×144),
    /// `PHASM_O3_FRAMES` (8, one GOP), `PHASM_O3_Q` (100). Prints the
    /// encode / harvest / rebuild timings.
    #[test]
    #[ignore = "needs ffmpeg + corpus; run with --ignored --nocapture"]
    fn rebuild_matches_encoder_on_corpus() {
        let clip = std::env::var("PHASM_O3_CLIP").unwrap_or_else(|_| {
            format!(
                "{}/test-vectors/video/h264/real-world/source/IMG_4138.MOV",
                env!("CARGO_MANIFEST_DIR")
            )
        });
        let (w, h) = (env_u32("PHASM_O3_W", 256), env_u32("PHASM_O3_H", 144));
        let n = env_u32("PHASM_O3_FRAMES", 8);
        let q = env_u32("PHASM_O3_Q", 100) as usize;
        let vf = format!("scale={w}:{h}:force_original_aspect_ratio=disable");
        let out = std::process::Command::new("ffmpeg")
            .args(["-y", "-loglevel", "error", "-i", &clip])
            .args(["-frames:v", &n.to_string(), "-vf", &vf])
            .args(["-pix_fmt", "yuv420p", "-f", "rawvideo", "-"])
            .output()
            .expect("ffmpeg");
        assert!(out.status.success(), "ffmpeg failed on {clip}");
        let params = Av1StreamingEncodeParams {
            width: w,
            height: h,
            quantizer: q,
            gop_size: n,
            total_frames_hint: n,
        };

        let t = std::time::Instant::now();
        let enc = encode_gop_natural(&out.stdout, n, params).expect("encode");
        let t_encode = t.elapsed();
        let packets: Vec<Vec<u8>> = enc.iter().map(|(p, _)| p.clone()).collect();
        let t = std::time::Instant::now();
        let dec = rebuild_gop_recordings(&packets, w, h).expect("rebuild");
        let t_rebuild = t.elapsed();
        let symbols: usize = dec.iter().map(|(_, r)| r.tiles[0].storage.len()).sum();
        compare_frames(&enc, &dec, w as usize, h as usize).unwrap();
        let t = std::time::Instant::now();
        let he = harvest_gop(&enc, w, h).unwrap();
        let t_harvest = t.elapsed();
        let hd = harvest_gop(&dec, w, h).unwrap();
        assert!(he.bit_identical(&hd), "harvests differ");
        for frac in [0.01, 0.05, 0.08] {
            assert_eq!(embed(&enc, w, h, frac), embed(&dec, w, h, frac), "{frac}: embed differs");
        }
        eprintln!(
            "[rebuild] {clip} {w}x{h} n={n} q={q}: {symbols} symbols, {} cover bits, all equal | \
             rav1e encode {:.3}s, harvest {:.3}s, rebuild (decode) {:.3}s, natural {} B",
            he.cover_len(),
            t_encode.as_secs_f64(),
            t_harvest.as_secs_f64(),
            t_rebuild.as_secs_f64(),
            packets.iter().map(Vec::len).sum::<usize>()
        );
    }
}
