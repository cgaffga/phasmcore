// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// Raw FFI bindings for cgaffga/phasm-dav1d (phasm-stego fork of
// dav1d AV1 decoder).
//
// **W2.2 SKELETON STATE (2026-05-20)**: this file is intentionally
// near-empty. It establishes the crate's compilation unit so cargo
// can build core-dav1d-sys with the workspace.
//
// Real FFI will land in W3 alongside the phasm-stego branch on the
// phasm-dav1d fork. The FFI surface is expected to be small (mirrors
// openh264-sys's "1 enum, 2 plain-data structs, 3 callback function
// types, 1 callback table struct, 3 entry points" pattern):
//
//   - 1 enum: PhasmDav1dDomain (Tier 1 channel identifier)
//   - 2-3 plain-data structs: PhasmDav1dPos, PhasmDav1dSymbolMeta
//   - 2-3 callback function types (L(1) decode site, mode-decision
//     read site, CdefIdx read site)
//   - 1 callback table struct: PhasmDav1dCallbacks
//   - 2-3 entry points: register_callbacks, abi_version, reset_state
//
// Higher-level Rust API (callback registration RAII guard, panic
// catching, position-key translation) will live in phasm-core under
// `core/src/codec/av1/` — see channel-design.md § 2.2 + streaming-
// session.md § 10 for the architectural shape.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]

// W2.2: stub presence marker. Returns 0 always. Confirms the C shim
// linked successfully and gives downstream Rust code something to
// call to verify the sys crate is built. Will be removed in W3 once
// the real FFI surface arrives.
unsafe extern "C" {
    pub fn phasm_dav1d_stub_present() -> core::ffi::c_int;
}

/// W2.2 skeleton sanity check. Returns 0 if the C shim linked
/// correctly. Removed in W3 once real FFI lands.
pub fn skeleton_present() -> bool {
    // SAFETY: phasm_dav1d_stub_present is a no-arg, pure C function
    // returning a constant. Cannot panic, doesn't deref pointers.
    unsafe { phasm_dav1d_stub_present() == 0 }
}

// ============================================================
// W3.D.2.1+: phasm-stego FFI surface (Dav1dPhasmHooks + tags).
// ============================================================
//
// Hand-rolled Rust FFI surface for the W3.D fork patches on
// phasm-dav1d. Matches the C-side declarations in
// `vendor/phasm-dav1d/include/dav1d/dav1d.h` (W3.D.2.1).
//
// The W3.D.4 phasm-core decode wrapper uses this to register hooks
// on a Dav1dContext. ABI parity with the C declarations is enforced
// at test time by the `phasm_abi_*` checks against the shim's
// `phasm_dav1d_abi_*` reporter functions (see tests below).

/// Tag constants — must match `DAV1D_PHASM_TAG_*` in
/// `vendor/phasm-dav1d/include/dav1d/dav1d.h` AND
/// `vendor/phasm-rav1e/src/ec.rs` PHASM_TAG_* constants on the
/// encoder side. Cross-checked at test time.
pub const DAV1D_PHASM_TAG_OTHER: u8 = 0;
pub const DAV1D_PHASM_TAG_AC_COEFF_SIGN: u8 = 1;
pub const DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB: u8 = 2;

/// Per-bit decode callback. Fires AFTER each 50/50 binary symbol
/// (`dav1d_msac_decode_bool_equi`) returns, with the decoded value
/// + currently-set tag.
pub type Dav1dPhasmBitHook =
    Option<unsafe extern "C" fn(cookie: *mut core::ffi::c_void, bit: core::ffi::c_uint, tag: u8)>;

/// Tag-change callback. Optional; fires when `dav1d_msac_phasm_set_tag`
/// is called (typically from inside the AC sign / golomb tail decode
/// sites in src/recon_tmpl.c after W3.D.3 lands).
pub type Dav1dPhasmTagHook =
    Option<unsafe extern "C" fn(cookie: *mut core::ffi::c_void, tag: u8)>;

/// Phase B.1.1.b: per-AC-sign-emission spatial metadata. Field
/// layout MUST match `Dav1dPhasmAcSignMeta` in
/// `vendor/phasm-dav1d/include/dav1d/dav1d.h` and the encoder-side
/// `AcSignMeta` in `vendor/phasm-rav1e/src/ec.rs`.
///
/// Pixel coords are PER-PLANE (already chroma-subsampled). Used by
/// phasm-core's J-UNIWARD cost computation to map cover bits to
/// reconstructed-pixel patches.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Dav1dPhasmAcSignMeta {
    pub plane: u8,
    pub plane_px_x: u16,
    pub plane_px_y: u16,
    pub tx_width_log2: u8,
    pub tx_height_log2: u8,
    pub tx_type: u8,
    pub scan_pos: u16,
}

/// Meta-change callback. Fires from `dav1d_msac_phasm_set_meta` at
/// each AC sign decode site BEFORE the actual sign bit decode.
/// Only fires when `tag == DAV1D_PHASM_TAG_AC_COEFF_SIGN` is about
/// to be set.
pub type Dav1dPhasmMetaHook = Option<
    unsafe extern "C" fn(
        cookie: *mut core::ffi::c_void,
        meta: *const Dav1dPhasmAcSignMeta,
    ),
>;

/// stealth-audit-2026-06-29: per-block metadata mirror. Field layout
/// MUST match `Dav1dPhasmBlockInfo` in
/// `vendor/phasm-dav1d/include/dav1d/dav1d.h`.
///
/// Fields populated by dav1d's decode_b at the end of per-block
/// decode. Intra/inter unions are flattened: when `intra=1`,
/// `y_mode`/`uv_mode` are set and inter fields are zero-init; when
/// `intra=0`, ref/mv/inter_mode fields are set and intra fields are
/// zero-init.
///
/// Used by the Layer-3 fingerprint comparison binary
/// (`av1_block_hist`) to tally partition/mode/MV/ref distributions
/// per encoded file.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Dav1dPhasmBlockInfo {
    pub bx: u16,
    pub by: u16,
    pub bs: u8,
    pub bl: u8,
    pub bp: u8,
    pub intra: u8,
    pub skip: u8,
    pub skip_mode: u8,
    pub seg_id: u8,
    pub y_mode: u8,
    pub uv_mode: u8,
    pub ref0: i8,
    pub ref1: i8,
    pub inter_mode: u8,
    pub motion_mode: u8,
    pub comp_type: u8,
    pub mv0_x: i16,
    pub mv0_y: i16,
    pub mv1_x: i16,
    pub mv1_y: i16,
    pub frame_type: u8,
    pub _pad0: u8,
    pub frame_offset: u16,
}

/// stealth-audit-2026-06-29: block_hook callback type. Fires once
/// per Av1Block at the end of dav1d's decode_b.
pub type Dav1dPhasmBlockHook = Option<
    unsafe extern "C" fn(
        cookie: *mut core::ffi::c_void,
        info: *const Dav1dPhasmBlockInfo,
    ),
>;

/// Hook registration struct — embedded in `Dav1dSettings` (and copied
/// down through `Dav1dContext` to each `MsacContext` per
/// `dav1d-hook-sites.md` § 5).
///
/// Zero-init (NULL callbacks) = no-op = byte-identical decode behaviour
/// to upstream dav1d. Field order + sizes MUST match the C struct in
/// `vendor/phasm-dav1d/include/dav1d/dav1d.h`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dav1dPhasmHooks {
    pub cookie: *mut core::ffi::c_void,
    pub bit_hook: Dav1dPhasmBitHook,
    pub tag_hook: Dav1dPhasmTagHook,
    /// Phase B.1.1.b addition.
    pub meta_hook: Dav1dPhasmMetaHook,
    /// stealth-audit-2026-06-29 addition.
    pub block_hook: Dav1dPhasmBlockHook,
}

// ABI verification helpers exposed by the shim. Each returns the
// C-side value so the Rust test can cross-check.
unsafe extern "C" {
    fn phasm_dav1d_abi_phasm_hooks_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_phasm_hooks_align() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_tag_other() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_tag_ac_coeff_sign() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_tag_golomb_tail_lsb() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_set_tag_fn_ptr() -> *mut core::ffi::c_void;
    // W3.D.4.1: struct size reporters for the larger public-API types.
    fn phasm_dav1d_abi_settings_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_data_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_picture_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_logger_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_pic_allocator_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_data_props_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_user_data_size() -> core::ffi::c_uint;
    fn phasm_dav1d_abi_settings_phasm_hooks_offset() -> core::ffi::c_uint;
}

// ============================================================
// W3.D.4.1: real FFI bindings for dav1d public C API.
// ============================================================
//
// Minimum subset needed to drive a decode round-trip from
// phasm-core: open context with phasm_hooks → send_data → get_picture
// → unref_picture → close.
//
// Struct layouts mirror C declarations exactly; sizes cross-checked
// against shim reporters at test time (see phasm_abi_* tests below).
// Picture is treated as a sized opaque blob (we don't read its
// contents; just need lifecycle management).

/// Opaque dav1d decoder context. Created by `dav1d_open`, destroyed
/// by `dav1d_close`. Internals defined in `vendor/phasm-dav1d/src/
/// internal.h::Dav1dContext` — not exposed at FFI boundary.
#[repr(C)]
pub struct Dav1dContext {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Opaque reference-counting wrapper. Used internally by dav1d for
/// `Dav1dData.ref`, `Dav1dPicture.ref`, etc. We only ever hold
/// `*mut Dav1dRef` pointers and let dav1d's ref-counting manage
/// lifecycle.
#[repr(C)]
pub struct Dav1dRef {
    _opaque: [u8; 0],
    _marker: core::marker::PhantomData<(*mut u8, core::marker::PhantomPinned)>,
}

/// Mirror of C `Dav1dLogger`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dav1dLogger {
    pub cookie: *mut core::ffi::c_void,
    pub callback: Option<
        unsafe extern "C" fn(
            cookie: *mut core::ffi::c_void,
            format: *const core::ffi::c_char,
            // va_list — we never call this from Rust; pass NULL.
            // C layout is opaque; on aarch64 it's a struct, on x86_64
            // it's a pointer. We use `*mut c_void` placeholder + only
            // use the default logger callback which dav1d provides.
            ap: *mut core::ffi::c_void,
        ),
    >,
}

/// Mirror of C `Dav1dPicAllocator`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dav1dPicAllocator {
    pub cookie: *mut core::ffi::c_void,
    pub alloc_picture_callback: Option<
        unsafe extern "C" fn(
            pic: *mut core::ffi::c_void, // really *mut Dav1dPicture; opaque here
            cookie: *mut core::ffi::c_void,
        ) -> core::ffi::c_int,
    >,
    pub release_picture_callback: Option<
        unsafe extern "C" fn(
            pic: *mut core::ffi::c_void,
            cookie: *mut core::ffi::c_void,
        ),
    >,
}

/// Mirror of C `Dav1dUserData`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dav1dUserData {
    pub data: *const u8,
    pub r#ref: *mut Dav1dRef,
}

/// Mirror of C `Dav1dDataProps`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dav1dDataProps {
    pub timestamp: i64,
    pub duration: i64,
    pub offset: i64,
    pub size: usize,
    pub user_data: Dav1dUserData,
}

/// Mirror of C `Dav1dData`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dav1dData {
    pub data: *const u8,
    pub sz: usize,
    pub r#ref: *mut Dav1dRef,
    pub m: Dav1dDataProps,
}

/// `Dav1dInloopFilterType` — opaque enum exposed as `c_uint`.
/// `DAV1D_INLOOPFILTER_ALL = 7` per dav1d.h.
pub type Dav1dInloopFilterType = core::ffi::c_uint;
pub const DAV1D_INLOOPFILTER_ALL: Dav1dInloopFilterType = 7;

/// `Dav1dDecodeFrameType` — opaque enum exposed as `c_uint`.
/// `DAV1D_DECODEFRAMETYPE_ALL = 0` per dav1d.h.
pub type Dav1dDecodeFrameType = core::ffi::c_uint;
pub const DAV1D_DECODEFRAMETYPE_ALL: Dav1dDecodeFrameType = 0;

/// Mirror of C `Dav1dSettings` (`vendor/phasm-dav1d/include/dav1d/
/// dav1d.h::78`). Field order MUST match C exactly. ABI cross-checked
/// at test time via `phasm_dav1d_abi_settings_size` +
/// `phasm_dav1d_abi_settings_phasm_hooks_offset`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Dav1dSettings {
    pub n_threads: core::ffi::c_int,
    pub max_frame_delay: core::ffi::c_int,
    pub apply_grain: core::ffi::c_int,
    pub operating_point: core::ffi::c_int,
    pub all_layers: core::ffi::c_int,
    pub frame_size_limit: core::ffi::c_uint,
    pub allocator: Dav1dPicAllocator,
    pub logger: Dav1dLogger,
    pub strict_std_compliance: core::ffi::c_int,
    pub output_invisible_frames: core::ffi::c_int,
    pub inloop_filters: Dav1dInloopFilterType,
    pub decode_frame_type: Dav1dDecodeFrameType,
    pub reserved: [u8; 16],
    /// phasm-stego (W3.D.2.1): per-decode hook callbacks. Zero-init
    /// = NULL = no hooks = byte-identical decode behaviour to upstream
    /// dav1d. Phasm-core's decode wrapper (W3.D.4.2) fills this in
    /// to register a recording bit_hook.
    pub phasm_hooks: Dav1dPhasmHooks,
}

/// Sized opaque blob for `Dav1dPicture`. Size cross-checked against
/// `phasm_dav1d_abi_picture_size` at test time. We never read or
/// write picture contents directly from Rust — just construct a
/// zero-init `Dav1dPicture`, hand it to `dav1d_get_picture`, then
/// call `dav1d_picture_unref` to free it.
///
/// Size is generous (1024 bytes) to allow upstream dav1d field
/// growth; the ABI test catches if actual sizeof exceeds this.
#[repr(C, align(16))]
pub struct Dav1dPicture {
    _blob: [u8; 1024],
}

impl Default for Dav1dPicture {
    fn default() -> Self {
        Self { _blob: [0u8; 1024] }
    }
}

// dav1d public C entry points. Mirror declarations from
// vendor/phasm-dav1d/include/dav1d/dav1d.h + data.h + picture.h.
unsafe extern "C" {
    pub fn dav1d_version() -> *const core::ffi::c_char;
    pub fn dav1d_version_api() -> core::ffi::c_uint;
    pub fn dav1d_default_settings(s: *mut Dav1dSettings);
    pub fn dav1d_open(
        c_out: *mut *mut Dav1dContext,
        s: *const Dav1dSettings,
    ) -> core::ffi::c_int;
    pub fn dav1d_close(c_out: *mut *mut Dav1dContext);

    pub fn dav1d_data_create(data: *mut Dav1dData, sz: usize) -> *mut u8;
    pub fn dav1d_data_unref(data: *mut Dav1dData);

    pub fn dav1d_send_data(
        c: *mut Dav1dContext,
        in_: *mut Dav1dData,
    ) -> core::ffi::c_int;
    pub fn dav1d_get_picture(
        c: *mut Dav1dContext,
        out: *mut Dav1dPicture,
    ) -> core::ffi::c_int;
    pub fn dav1d_picture_unref(p: *mut Dav1dPicture);
}

// phasm-stego helper to retrieve platform-specific DAV1D_ERR(EAGAIN)
// value (-11 on Linux, -35 on macOS). dav1d returns this from
// send_data + get_picture to signal "buffer full / need more data".
unsafe extern "C" {
    fn phasm_dav1d_eagain_err() -> core::ffi::c_int;
}

/// Returns dav1d's `DAV1D_ERR(EAGAIN)` value for the current
/// platform. Use this to detect the "try the other call first"
/// condition from `dav1d_send_data` + `dav1d_get_picture`.
pub fn dav1d_err_again() -> i32 {
    // SAFETY: phasm_dav1d_eagain_err is a no-arg pure function that
    // returns a constant. Cannot panic, no pointers.
    unsafe { phasm_dav1d_eagain_err() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn skeleton_links() {
        // The C shim compiled + linked (true in both stub and real
        // build modes — phasm_dav1d_stub_present is in the shim
        // either way).
        assert!(skeleton_present());
    }

    #[test]
    fn real_build_succeeded() {
        // W2.3 acceptance: when the submodule is present, build.rs
        // takes the meson + ninja path; phasm_dav1d_stub cfg is
        // NOT set. The CI environment may legitimately run without
        // submodule (e.g., source-archive distributions), so this
        // test only asserts when phasm_dav1d_stub IS unset — i.e.,
        // it's a "did the real build work" check, not a "must be
        // real build" check.
        if !cfg!(phasm_dav1d_stub) {
            // Real build path: libdav1d.a was produced and linked.
            // Calling skeleton_present already proves link succeeded.
            assert!(skeleton_present());
        }
    }

    // ========================================================
    // W3.D.2.5: phasm-stego FFI ABI surface verification.
    // ========================================================
    //
    // Cross-check the Rust-side Dav1dPhasmHooks struct + tag
    // constants against the C-side declarations in
    // vendor/phasm-dav1d/include/dav1d/dav1d.h (W3.D.2.1) via the
    // phasm_dav1d_abi_* reporters in the shim. Catches:
    //   - Struct size / alignment drift (Rust struct layout doesn't
    //     match C struct).
    //   - Tag value drift (Rust constants != C #defines).
    //   - Missing dav1d_msac_phasm_set_tag symbol (W3.D.2.2 not in
    //     libdav1d.a — e.g. submodule SHA is too old, or symbol got
    //     stripped).
    //
    // Doesn't actually fire the hook end-to-end — that's W3.D.4's
    // round-trip test (which needs real dav1d_open + decode flow).

    #[test]
    fn phasm_abi_struct_size() {
        if cfg!(phasm_dav1d_stub) {
            return; // shim-only build skips dav1d-h-dependent checks
        }
        let rust_size = std::mem::size_of::<Dav1dPhasmHooks>() as core::ffi::c_uint;
        let c_size = unsafe { phasm_dav1d_abi_phasm_hooks_size() };
        assert_eq!(
            rust_size, c_size,
            "Dav1dPhasmHooks size mismatch: Rust={}, C={}",
            rust_size, c_size
        );
    }

    #[test]
    fn phasm_abi_struct_alignment() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust_align = std::mem::align_of::<Dav1dPhasmHooks>() as core::ffi::c_uint;
        let c_align = unsafe { phasm_dav1d_abi_phasm_hooks_align() };
        assert_eq!(
            rust_align, c_align,
            "Dav1dPhasmHooks alignment mismatch: Rust={}, C={}",
            rust_align, c_align
        );
    }

    #[test]
    fn phasm_abi_tag_constants() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        unsafe {
            assert_eq!(
                DAV1D_PHASM_TAG_OTHER as core::ffi::c_uint,
                phasm_dav1d_abi_tag_other(),
                "DAV1D_PHASM_TAG_OTHER value mismatch"
            );
            assert_eq!(
                DAV1D_PHASM_TAG_AC_COEFF_SIGN as core::ffi::c_uint,
                phasm_dav1d_abi_tag_ac_coeff_sign(),
                "DAV1D_PHASM_TAG_AC_COEFF_SIGN value mismatch"
            );
            assert_eq!(
                DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB as core::ffi::c_uint,
                phasm_dav1d_abi_tag_golomb_tail_lsb(),
                "DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB value mismatch"
            );
        }
    }

    #[test]
    fn phasm_abi_set_tag_symbol_present() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let fn_ptr = unsafe { phasm_dav1d_abi_set_tag_fn_ptr() };
        assert!(
            !fn_ptr.is_null(),
            "dav1d_msac_phasm_set_tag symbol resolved to NULL — \
             W3.D.2.2 patches may not be in the current phasm-dav1d \
             submodule SHA"
        );
    }

    // ========================================================
    // W3.D.4.1: public-API struct ABI parity checks.
    // ========================================================
    //
    // Cross-check each Rust struct mirror against the C-side via
    // shim reporters. Catches drift if dav1d's public API changes
    // upstream (or we miss a rebase) before downstream code
    // (phasm-core W3.D.4.2 decode wrapper) hits a layout bug at
    // runtime.

    #[test]
    fn phasm_abi_settings_size() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::size_of::<Dav1dSettings>() as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_settings_size() };
        assert_eq!(rust, c, "Dav1dSettings size mismatch: Rust={}, C={}", rust, c);
    }

    #[test]
    fn phasm_abi_settings_phasm_hooks_offset() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::offset_of!(Dav1dSettings, phasm_hooks) as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_settings_phasm_hooks_offset() };
        assert_eq!(
            rust, c,
            "phasm_hooks offset in Dav1dSettings mismatch: Rust={}, C={}",
            rust, c
        );
    }

    #[test]
    fn phasm_abi_data_size() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::size_of::<Dav1dData>() as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_data_size() };
        assert_eq!(rust, c, "Dav1dData size mismatch: Rust={}, C={}", rust, c);
    }

    #[test]
    fn phasm_abi_picture_size_at_least_c_size() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::size_of::<Dav1dPicture>() as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_picture_size() };
        // Rust uses a generous 1024-byte blob; C's actual size must
        // fit within it. If C exceeds 1024, bump the blob and the
        // assertion; this is intentionally a one-sided check (we want
        // the blob to be ≥ C size, with some headroom).
        assert!(
            rust >= c,
            "Dav1dPicture blob too small: Rust={}, C={}",
            rust, c
        );
    }

    #[test]
    fn phasm_abi_logger_size() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::size_of::<Dav1dLogger>() as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_logger_size() };
        assert_eq!(rust, c, "Dav1dLogger size mismatch: Rust={}, C={}", rust, c);
    }

    #[test]
    fn phasm_abi_pic_allocator_size() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::size_of::<Dav1dPicAllocator>() as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_pic_allocator_size() };
        assert_eq!(
            rust, c,
            "Dav1dPicAllocator size mismatch: Rust={}, C={}",
            rust, c
        );
    }

    #[test]
    fn phasm_abi_data_props_size() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::size_of::<Dav1dDataProps>() as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_data_props_size() };
        assert_eq!(
            rust, c,
            "Dav1dDataProps size mismatch: Rust={}, C={}",
            rust, c
        );
    }

    #[test]
    fn phasm_abi_user_data_size() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        let rust = std::mem::size_of::<Dav1dUserData>() as core::ffi::c_uint;
        let c = unsafe { phasm_dav1d_abi_user_data_size() };
        assert_eq!(rust, c, "Dav1dUserData size mismatch: Rust={}, C={}", rust, c);
    }

    /// W3.D.4.1 open+close smoke test: default_settings populates the
    /// allocator + logger; dav1d_open creates a context; dav1d_close
    /// destroys it. Validates the basic context-lifecycle FFI works
    /// without needing actual AV1 input.
    #[test]
    fn dav1d_open_close_smoke() {
        if cfg!(phasm_dav1d_stub) {
            return;
        }
        unsafe {
            // Zero-init then call default_settings (which fills the
            // allocator + logger but not phasm_hooks or reserved).
            let mut settings: Dav1dSettings = std::mem::zeroed();
            dav1d_default_settings(&mut settings);
            // n_threads=1 + max_frame_delay=1 force the single-tile,
            // single-frame mode per dav1d-hook-sites.md § 1.
            settings.n_threads = 1;
            settings.max_frame_delay = 1;
            // phasm_hooks stays zero-initialised → NULL callbacks =
            // no-op = byte-identical decode behaviour.
            assert!(settings.phasm_hooks.bit_hook.is_none());

            let mut ctx: *mut Dav1dContext = std::ptr::null_mut();
            let rc = dav1d_open(&mut ctx, &settings);
            assert_eq!(rc, 0, "dav1d_open returned error code {}", rc);
            assert!(!ctx.is_null(), "dav1d_open returned NULL context");

            dav1d_close(&mut ctx);
            assert!(ctx.is_null(), "dav1d_close should null out the context pointer");
        }
    }
}
