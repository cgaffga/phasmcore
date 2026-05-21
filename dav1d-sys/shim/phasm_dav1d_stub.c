/* SPDX-License-Identifier: GPL-3.0-only
 * Copyright (c) 2026, Christoph Gaffga
 *
 * Stub C shim for core-dav1d-sys W2.2 SKELETON state.
 *
 * Defines fail-fast no-op implementations of the FFI surface we'll
 * eventually need from libdav1d / phasm-dav1d's stego hook API.
 *
 * W2.2 state: this file is intentionally near-empty. The first
 * real symbols arrive in W2.3 / W3 as we discover the exact FFI
 * surface we need. For now it just provides a compilation unit so
 * cc::Build::new().compile() has something to consume.
 *
 * When real symbols land in W3, each will follow the pattern:
 *
 *     int phasm_dav1d_some_function(...) {
 *       return -1;  // PHASM_DAV1D_E_STUBBED
 *     }
 *
 * Runtime callers see -1 and route through the degraded fallback
 * (or panic with a clear "real backend not available" message).
 */

/* No symbols yet — file exists only so cc::Build has a compilation
 * unit to consume. W2.3 / W3 will fill in stub function bodies as
 * the FFI surface is defined. */

static int phasm_dav1d_stub_marker = 0;

/* Reference the marker from a public function so it doesn't get
 * stripped by aggressive dead-code elimination in release builds.
 * Returns 0 always; not called by anything yet. */
int phasm_dav1d_stub_present(void) {
    return phasm_dav1d_stub_marker;
}

/* phasm-stego (W3.D.2.5): ABI surface verification.
 *
 * The Rust side (core-dav1d-sys src/lib.rs phasm_abi tests) declares
 * its own copy of Dav1dPhasmHooks + DAV1D_PHASM_TAG_* constants and
 * cross-checks them against these C-side reporters. Catches drift
 * between phasm-dav1d's dav1d.h header (W3.D.2.1) and the Rust FFI
 * surface that phasm-core depends on. See dav1d-hook-sites.md § 8
 * for the planned hand-written FFI bindings.
 */
#include "dav1d/dav1d.h"

unsigned phasm_dav1d_abi_phasm_hooks_size(void) {
    return (unsigned)sizeof(Dav1dPhasmHooks);
}

unsigned phasm_dav1d_abi_phasm_hooks_align(void) {
    return (unsigned)_Alignof(Dav1dPhasmHooks);
}

unsigned phasm_dav1d_abi_tag_other(void) {
    return (unsigned)DAV1D_PHASM_TAG_OTHER;
}

unsigned phasm_dav1d_abi_tag_ac_coeff_sign(void) {
    return (unsigned)DAV1D_PHASM_TAG_AC_COEFF_SIGN;
}

unsigned phasm_dav1d_abi_tag_golomb_tail_lsb(void) {
    return (unsigned)DAV1D_PHASM_TAG_GOLOMB_TAIL_LSB;
}

/* Returns NULL if the linker couldn't resolve dav1d_msac_phasm_set_tag —
 * means W3.D.2.2's symbol export got stripped or the phasm-dav1d
 * submodule isn't at a SHA that includes it.
 *
 * Forward-declare struct MsacContext (private dav1d type) so we can
 * take the function pointer without pulling in src/msac.h (a private
 * dav1d header not on the shim's include path).
 */
#include <stdint.h>
#include <stddef.h>  /* offsetof */
struct MsacContext;
extern void dav1d_msac_phasm_set_tag(struct MsacContext *s, uint8_t tag);

void *phasm_dav1d_abi_set_tag_fn_ptr(void) {
    return (void *)dav1d_msac_phasm_set_tag;
}

/* phasm-stego (W3.D.4.1): struct size reporters for the larger
 * dav1d public-API types that the phasm-core decode wrapper needs to
 * mirror. Used by core-dav1d-sys's ABI tests to cross-check that the
 * Rust struct layouts match C's view (catches drift if dav1d's
 * public API ever shifts upstream + we miss the rebase).
 */
#include "dav1d/data.h"
#include "dav1d/picture.h"

unsigned phasm_dav1d_abi_settings_size(void) {
    return (unsigned)sizeof(Dav1dSettings);
}

unsigned phasm_dav1d_abi_data_size(void) {
    return (unsigned)sizeof(Dav1dData);
}

unsigned phasm_dav1d_abi_picture_size(void) {
    return (unsigned)sizeof(Dav1dPicture);
}

unsigned phasm_dav1d_abi_logger_size(void) {
    return (unsigned)sizeof(Dav1dLogger);
}

unsigned phasm_dav1d_abi_pic_allocator_size(void) {
    return (unsigned)sizeof(Dav1dPicAllocator);
}

unsigned phasm_dav1d_abi_data_props_size(void) {
    return (unsigned)sizeof(Dav1dDataProps);
}

unsigned phasm_dav1d_abi_user_data_size(void) {
    return (unsigned)sizeof(Dav1dUserData);
}

/* Offset of phasm_hooks inside Dav1dSettings — must match what the
 * Rust mirror computes. Catches "added a field before phasm_hooks
 * without updating Rust" drift.
 */
unsigned phasm_dav1d_abi_settings_phasm_hooks_offset(void) {
    return (unsigned)offsetof(Dav1dSettings, phasm_hooks);
}

/* phasm-stego (W3.D.4.2): EAGAIN errno → DAV1D_ERR(EAGAIN). dav1d's
 * send_data + get_picture flow returns DAV1D_ERR(EAGAIN) to signal
 * "buffer full, call the other one first" / "need more data". Errno
 * values are platform-specific (EAGAIN=11 on Linux, 35 on macOS),
 * so we expose the platform value via this shim helper.
 */
#include <errno.h>
int phasm_dav1d_eagain_err(void) {
    return -EAGAIN;
}
