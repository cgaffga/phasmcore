// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// build.rs for core-openh264-sys.
//
// Drives a meson + ninja static build of the phasm-openh264 submodule
// at ../../vendor/phasm-openh264 and emits Cargo link directives so
// downstream Rust code can call into wels_stego.h via extern "C".
//
// Prerequisites checked at build time:
//   - vendor/phasm-openh264 submodule initialized (meson.build present)
//   - meson + ninja on PATH
//
// Behaviour:
//   - First build: meson setup _build_static --default-library=static
//                  --buildtype=release ; ninja -C _build_static
//   - Incremental: ninja -C _build_static (meson setup is idempotent
//     and skipped when the build dir already has build.ninja)
//   - Re-run trigger: any change under vendor/phasm-openh264/codec/
//
// Task #381 (Phase B.12) — graceful fallback when the submodule is
// absent (e.g., the phasmcore mirror, which is `git archive`'d and
// doesn't carry submodules). Detection + stub-compile path lives in
// `build_stub` below; the function emits cargo:warning, skips meson,
// and compiles `shim/phasm_openh264_stub.c` as a no-op library that
// satisfies the Rust FFI surface in `src/lib.rs`. Calls into the
// stubbed backend return error codes at runtime so callers fall back
// to the pure-Rust walker path.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    // CARGO_MANIFEST_DIR = core/openh264-sys
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // Locate submodule at <project root>/vendor/phasm-openh264.
    // From core/openh264-sys/, the project root is two levels up.
    let submodule_path_raw = manifest_dir
        .join("..")
        .join("..")
        .join("vendor")
        .join("phasm-openh264");

    // Task #381 — detect missing-submodule state. Two failure modes
    // both flow into the stub path:
    //   1. canonicalize() fails (path doesn't exist at all)
    //   2. canonicalize() succeeds but meson.build is missing (e.g.,
    //      directory was created but submodule wasn't checked out)
    let submodule_path = match submodule_path_raw.canonicalize() {
        Ok(p) if p.join("meson.build").exists() => p,
        _ => {
            build_stub(&manifest_dir, &submodule_path_raw);
            return;
        }
    };

    // Phase B.8 step 5 — SHA pin. The Rust-side scan-table + hook-key
    // translation in `core-openh264-sys` depends on Phase A.5's hook
    // firing convention (raster within block for HOOK-A/B/E, POST-scan
    // for HOOK-F/G, partition_idx-as-plane for HOOK-C/D, etc.). If the
    // fork's hook insertion sites shift in a future Phase A.5 stage,
    // those tables silently go wrong. Pin to a known-good SHA so any
    // future fork update fails the build loudly and the maintainer
    // can re-validate empirically.
    //
    // To unpin during fork development on a feature branch, export
    //   PHASM_OPENH264_SHA_PIN_OVERRIDE=<sha>
    // or
    //   PHASM_OPENH264_SHA_PIN_OVERRIDE=*  (accept any SHA)
    {
        const PINNED_SHAS: &[&str] = &[
            // 2026-05-11: A.5(j) wired md_cost_capture (ABI 0x010100).
            "acf7db39843183f30771ffba2830f4861d73088c",
            // 2026-05-12: B.9.2.0 + .1 decoder-side emit helpers
            // (codec/common/inc/wels_stego_dec_helpers.h + impls in
            // wels_stego.cpp). No ABI bump; no encoder behavior change.
            "3ff049e3335b8f998341671b01cd38cab960e206",
            // 2026-05-12: B.9.2.2 CoeffSign decoder hook +
            // wels_stego_common.cpp split (state + accessors + dec
            // helpers move to libcommon so libdecoder TU resolves
            // them). Encoder helpers slimmed; direct g_phasm_callbacks
            // access via accessor functions. dec_post_read now fires
            // 106,725× on the 320×240 IDR fixture — exact walker match.
            "c266256d5ce2ab121fca0d0fcf4b74e74c11f7ba",
            // 2026-05-12: B.9.2.3 MvdSignBypass decoder hook at
            // ParseMvdInfoCabac (post-DecodeBypassCabac, inside the
            // |MVD|>0 branch). Verified by b9_2_3_dec_mvd_sign_
            // fires_match_walker: decoder fires 2 vs walker 2 on a
            // 3-frame IDR+P+P shifted-noise fixture.
            "a9102398a208a00711b324a2a843c84468a662d9",
            // 2026-05-12: B.9.2.4 MvdSuffixLsb decoder hook at
            // ParseMvdInfoCabac (post-sign-flip, |MVD|>=9 only).
            "e262a6cd65051d1c2fb7ad2a05014e571c7cfc26",
            // 2026-05-12: B.9.2.5 CoeffSuffixLsb decoder hook in
            // ParseSignificantCoeffCabac (|coeff|>=16 — phasm walker's
            // WET-eligible threshold). Closes B.9.2 fork-side: all 4
            // stego domains now fire decoder hooks bit-exact vs walker.
            "715ac507da3dcb37f222aaadce4e8e1c9b795744",
            // 2026-05-12: fork CI fix — PhasmStegoAbi.VersionNotZero
            // hardcoded 0x010000 but PHASM_STEGO_ABI_VERSION had been
            // bumped to 0x010100 for the dec_post_read additions.
            // Test now reads the macro from wels_stego.h directly.
            "6bf4565af7f2d827781cac180539a8c738b70ae4",
            // 2026-05-12: C.8.1 pVisualRecPic per-DqLayer buffer
            // allocator. Adds parallel SRefList::pVisualRef[] pool
            // (1:1 with pRef[]) + pCtx->pVisualDecPic +
            // SDqLayer::pVisualRecPic plumbing. Buffer-infra only --
            // no dual-write hooks yet, encoder bit-identical (21/21
            // openh264-backend lib tests including session_round_trip
            // + b9_3_full_roundtrip + b10_cascade_safe_roundtrip).
            "eda532623010a93da7a340d4c25d9713840b8d98",
            // 2026-05-12: C.8.2 dual-recon hook ABI. Bumps
            // PHASM_STEGO_ABI_VERSION 1.1.0 -> 1.2.0 (additive). Adds
            // PhasmStegoDualReconFn + dual_recon_observe field +
            // internal phasm_dual_recon_writeback helper. ABI-only
            // change; encoder bit-identical, 22/22 openh264-backend
            // lib tests + 4 new gtest cases for the writeback helper.
            "a70255e8ca91f0b39d4cb6933b53a0f38d897c04",
            // 2026-05-12: C.8.3 I_16x16 dual-recon. WelsEncRecI16x16Y
            // snapshots post-quant clean coefficients before HOOK-A/B,
            // reruns dequant+DC-reinject+IDCT after the existing stego
            // recon writes pPred, then mirrors stego into pVisualRecPic
            // via phasm_dual_recon_writeback. pDecPic = clean,
            // pVisualRecPic = stego on I_16x16 MBs. Gated on
            // PhasmStegoGetEncPreEmit()!=NULL so no-stego encodes stay
            // byte-identical to upstream. 22/22 lib tests pass.
            "d1b91289395e2f4892f7a140f5cac81c622df01b",
            // 2026-05-13: C.8.4 I_4x4 dual-recon + C.8.3 sizeof bug fix
            // in svc_encode_mb.cpp. WelsEncRecI4x4Y now snapshots
            // post-quant clean per sub-block, reruns dequant+IDCT after
            // the original IDCT writes stego, mirrors stego to
            // pVisualRecPic. C.8.3 recompute previously used
            // sizeof(ENFORCE_STACK_ALIGN_1D-name) which evaluates to
            // pointer size (8B) instead of the array byte count -- pPred
            // was being written with stack garbage; b10 passed via
            // deterministic garbage. Fixed to sizeof(int16_t)*N. 22/22
            // lib tests genuinely green now (including b10's seq-diverge
            // check on real clean pixels).
            "4497206d87fdb287a67598141cfd5ac020b55df1",
            // 2026-05-13: C.8.5 chroma DC + AC dual-recon (stash +
            // Pskip mirror only -- chroma intra + inter clean recompute
            // was missing due to a silent Edit-failure in this session).
            "eb94726fa79f5b66bba497653b1217b55a6a2896",
            // 2026-05-13: C.8.5 completion -- chroma intra (Site C-3,
            // WelsIMbChromaEncode) + chroma inter (Site C-4,
            // OutputPMbWithoutConstructCsRsNoCopy) clean recompute now
            // wired. With C.8.3/C.8.4 luma + this commit's chroma
            // dual-recon active, the chroma cascade through MB N+1+'s
            // intra-chroma prediction is broken (= visual_recon working
            // as designed). The 1-byte stc_drives bitstream-differ
            // signal that pre-C.8.5 carried the cascade leak is now
            // absorbed -- 6 sign-bit flips compress to the same CABAC
            // bytes in the 320x240 QP=18 fixture. b9_3 round-trip via
            // decoder hook continues to verify wire-level override
            // propagation rigorously. 22/22 openh264-backend lib tests
            // + 1365/1365 total lib tests pass.
            "f50fef5d52b85657bb80cf75631542794d25c91f",
            // 2026-05-13: C.8.6 P-frame luma dual-recon. Closes the
            // largest remaining cascade surface (P-frame in-place luma
            // IDCT in OutputPMbWithoutConstructCsRsNoCopy was writing
            // stego luma into pDecY after HOOK-F flipped coefficients).
            // New phasm_stash_p_luma_clean_pres helper carries CLEAN
            // post-quant + mirror-suppression+dequant luma residual
            // from WelsEncInterY to the IDCT site. Pattern mirrors
            // C.8.5 inter chroma: snapshot MC pred, live IDCT, snapshot
            // stego, restore pred, re-IDCT with clean stash, mirror to
            // pVisualRecPic[Y]. Residual-coefficient cascade now closed
            // for all 4 shapes (I_16x16/I_4x4/intra chroma/inter Y+UV).
            // Still leaking: MvdSign MC (C.8.7), deblock (C.8.8), ref
            // promotion audit (C.8.9). 22/22 + 1365/1365 tests pass.
            "54bf85afd1f40731acc80223ca6906d2f73fa4b8",
            // 2026-05-13: C.8 observe-callback fix. All 5 dual-recon
            // call sites (C.8.3-C.8.6) now pass compact clean snapshots
            // to phasm_dual_recon_writeback so the observe callback
            // actually fires (helper had `clean_pixels != NULL &&
            // stego_pixels != NULL` guard; we previously passed NULL
            // for clean). Enables the cascade-break probe test
            // c8_pdecpic_clean_under_coeff_sign_overrides which
            // captured 0 cascade_leak_blocks across 10727 observed
            // blocks on a 2-frame 320x240 IDR+P fixture — direct
            // evidence C.8.3-6 cascade-break works as intended.
            "9b4f66df8ce7f80150be7db72d3cb4ba56f02869",
            // C.8.7 (#440) — MvdSign cascade-break for P_16x16
            // (HOOK-H1 luma + chroma). On MV-override MBs, HOOK-H1
            // captures clean MC into a per-MB stash; OutputPMb shifts
            // pDecPic by (CLEAN_MC − STEGO_MC) so encoder reference
            // stays clean while pVisualRecPic mirrors the actual
            // decoder reconstruction. Partitioned modes (H2..H7)
            // remain pending.
            "5505f3e0b2bf04d4b2c5eead8f8a6702b0f3dcc6",
            // C.8.7 v1.1 (#453) — extends cascade-break to partitioned
            // P modes: P_16x8 (H2), P_8x16 (H3), P_8x8/SUB_MB_TYPE_8x8
            // (H4) — luma via the centralised partition helper +
            // chroma inline at each site. H5/H6/H7 (sub-MB 4x4/8x4/4x8)
            // closed-by-design: OpenH264 upstream itself disables
            // sub-MB-partition selection at svc_mode_decision.cpp:628
            // (#if 0 //Disable for sub8x8 modes for now), so those
            // hooks can never fire.
            "5bdca8b84a3e363e04b2bd0ac3726a507ee28756",
            // C.8.8 (#441) — deblock dual-pass. After slice recon,
            // run the deblock filter twice: once on pDecPic (encoder
            // reference for ME), once on pVisualRecPic (decoder-
            // equivalent reconstruction for mp4 output). Boundary
            // strengths come from SMB syntax → identical for both.
            // Only pixel content differs. ~2× deblock cost in stego
            // mode; zero cost outside stego.
            "e9988005c752db426bb55061a93e32f0bada9308",
            // C.8.9 (#442) — cascade-safety runtime assertions at the
            // two DPB promotion sites (ref_list_mgr_svc.cpp:403 short-
            // ref + encoder_ext.cpp:2833 ME ref attachment). Both scan
            // pVisualRef[] and abort if a stego-mirror pointer ever
            // reaches the clean reference path. Structural decoupling
            // argued in docs/design/video/h264/c89-cascade-safety-audit
            // section 2.
            "6588fb54cc906ef798cd6c3f752120e304007d6f",
            // C.8.10 (#443) — fsnr redirect to pVisualDecPic for
            // encoder-internal debug output paths (ENABLE_FRAME_DUMP +
            // PSNR readback). The mp4 muxer is downstream of the
            // encoder library, so it receives the wire bitstream and
            // any decoder naturally produces stego pixels. Only the
            // in-library debug output needed redirecting.
            "e210d008c06775a507c79f390f96a58e38c26416",
            // 222457ce - C.8.13(b) fix HOOK-F coeff_idx — pass raster
            // not scan. BC=2 canonical-key translation applies
            // INV_ZZ_SCAN expecting raster (matches HOOK-E intra
            // convention). HOOK-F (P-inter Luma 4x4) was passing
            // scan, silently misrouting overrides on non-fixed-point
            // scan indices. Root-caused via the 32-seed audit in
            // `core/tests/openh264_cascade_gap_audit.rs`.
            "222457ceebeaf9a5c72f09c7a46ba8a9419f5c2b",
        ];
        let head_output = Command::new("git")
            .arg("-C")
            .arg(&submodule_path)
            .arg("rev-parse")
            .arg("HEAD")
            .output();
        if let Ok(out) = head_output {
            if out.status.success() {
                let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let override_env = std::env::var("PHASM_OPENH264_SHA_PIN_OVERRIDE").ok();
                let accept = override_env.as_deref() == Some("*")
                    || override_env.as_deref() == Some(&sha)
                    || PINNED_SHAS.iter().any(|s| *s == sha);
                if !accept {
                    panic!(
                        "phasm-openh264 submodule HEAD {} is not in the\n\
                         known-good SHA pin list {:?}.\n\
                         Phase B.8's Rust-side scan-table translation in\n\
                         core-openh264-sys depends on Phase A.5's hook firing\n\
                         convention. Verify empirically by re-running:\n\
                         \n\
                         cargo test --all-features --lib openh264::tests::b8_full_translation\n\
                         \n\
                         If it passes, add this SHA to PINNED_SHAS in\n\
                         core/openh264-sys/build.rs. If it fails, the fork's\n\
                         hook firing convention has shifted and the\n\
                         translation tables need updating.\n\
                         \n\
                         Override (for fork-side dev branches):\n\
                         PHASM_OPENH264_SHA_PIN_OVERRIDE={}\n\
                         PHASM_OPENH264_SHA_PIN_OVERRIDE=* (accept any)",
                        sha, PINNED_SHAS, sha
                    );
                }
                // Re-run build.rs if the submodule HEAD changes.
                println!(
                    "cargo:rerun-if-changed={}",
                    submodule_path.join(".git").display()
                );
            }
            // If `git rev-parse` fails (e.g. submodule isn't a git
            // checkout), skip the pin silently — the build can still
            // proceed, just without the safety check.
        }
        println!("cargo:rerun-if-env-changed=PHASM_OPENH264_SHA_PIN_OVERRIDE");
    }

    check_tool_available("meson", "Install via `brew install meson` (macOS) or `pip install meson` (Linux).");
    check_tool_available("ninja", "Install via `brew install ninja` (macOS) or `apt install ninja-build` (Linux).");

    let build_dir = out_dir.join("openh264-static");

    // Meson setup (skip if already configured).
    let needs_setup = !build_dir.join("build.ninja").exists();
    if needs_setup {
        std::fs::create_dir_all(&build_dir).expect("create build dir");
        let status = Command::new("meson")
            .arg("setup")
            .arg("--default-library=static")
            .arg("--buildtype=release")
            .arg("-Db_lto=false")  // LTO clashes with cargo's own LTO downstream
            .arg(&build_dir)
            .arg(&submodule_path)
            .status()
            .expect("failed to spawn meson");
        if !status.success() {
            panic!("meson setup failed (exit {:?}); check output above", status.code());
        }
    }

    // Ninja build. We deliberately build all default targets — Meson's
    // default_library=static still produces an aggregate libopenh264.a
    // plus per-component .a files, but the individual archive layout
    // varies by meson version, so the simplest portable thing is to
    // just build the whole project. The C++ unit tests are a sub-target;
    // they get built too but ignored for linking. (TODO follow-on: pass
    // --target libopenh264.a once we verify the artifact name is stable
    // across meson versions.)
    let status = Command::new("ninja")
        .arg("-C")
        .arg(&build_dir)
        .status()
        .expect("failed to spawn ninja");
    if !status.success() {
        panic!("ninja build failed (exit {:?}); check output above", status.code());
    }

    // OpenH264 with default-library=static produces an aggregate
    // libopenh264.a at the build root that bundles encoder + decoder +
    // common + processing. Linking that single archive avoids worrying
    // about cross-archive symbol resolution.
    println!("cargo:rustc-link-search=native={}", build_dir.display());
    println!("cargo:rustc-link-lib=static=openh264");

    // Compile the C++ shims alongside libopenh264.a. Encoder shim wraps
    // ISVCEncoder, decoder shim wraps ISVCDecoder — both expose plain-C
    // entry points so the Rust side doesn't have to model the C++ vtable.
    // See core/openh264-sys/shim/phasm_{encoder,decoder}_shim.{h,cc}.
    //
    // Both shims compile into a single static library `phasm_shims` so the
    // cargo cc invocation runs once; the file order doesn't matter (both
    // TUs are independent).
    let shim_dir = manifest_dir.join("shim");
    cc::Build::new()
        .cpp(true)
        .file(shim_dir.join("phasm_encoder_shim.cc"))
        .file(shim_dir.join("phasm_decoder_shim.cc"))
        .include(&shim_dir)
        .include(submodule_path.join("codec/api/wels"))
        .flag_if_supported("-std=c++14")
        // Match OpenH264's release build flags closely.
        .flag_if_supported("-fno-rtti")
        .flag_if_supported("-fno-exceptions")
        .opt_level(2)
        .warnings(true)
        .compile("phasm_shims");
    println!("cargo:rerun-if-changed={}", shim_dir.display());

    // C++ stdlib link.
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else if cfg!(target_os = "linux") {
        println!("cargo:rustc-link-lib=stdc++");
    } else if cfg!(target_os = "windows") {
        // MSVC bundles its stdlib; nothing to do.
    }

    // Expose include path so phasm-core can #include wels_stego.h via
    // bindgen later if we adopt it. For Phase B.3's hand-written
    // bindings, this is informational only.
    println!(
        "cargo:include={}",
        submodule_path.join("codec/api/wels").display()
    );

    // Trigger rebuild on any source/header change in the submodule.
    println!(
        "cargo:rerun-if-changed={}",
        submodule_path.join("codec").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        submodule_path.join("meson.build").display()
    );
}

fn check_tool_available(tool: &str, install_hint: &str) {
    let status = Command::new(tool).arg("--version").output();
    match status {
        Ok(output) if output.status.success() => {}
        _ => panic!(
            "Required build tool `{}` not found on PATH.\n{}",
            tool, install_hint
        ),
    }
}

/// Task #381 — Phase B.12: stub-build fallback when the
/// `vendor/phasm-openh264` submodule is absent.
///
/// Phasmcore mirror consumers receive a `git archive`'d snapshot of
/// `core/` that does NOT carry the submodule. Without this fallback,
/// `cargo build --features openh264-backend` would panic in the
/// canonicalize() / meson.build check above. Instead, this function:
///
/// 1. Emits a loud `cargo:warning=...` pointing at the README setup
///    instructions, so users see the situation at build time.
/// 2. Compiles `shim/phasm_openh264_stub.c` (a plain-C TU defining
///    every FFI symbol as a fail-fast no-op) as `phasm_shims`.
/// 3. Sets `cfg=phasm_openh264_stub` so Rust-side code can pick a
///    different path if it wants (optional — runtime detection via
///    `WelsStegoAbiVersion() == 0` works too).
/// 4. Skips the `cargo:rustc-link-lib=static=openh264` directive
///    (no archive was built).
/// 5. Skips the C++ stdlib link (stub is plain C).
///
/// Runtime: every backend call returns -1 / NULL. Rust-side wrappers
/// (`StegoSession::new`, `Decoder::new`) propagate that as an error,
/// and the higher-level orchestrator falls back to the pure-Rust
/// walker path. No production breakage; just degraded backend
/// availability with a visible warning.
fn build_stub(manifest_dir: &std::path::Path, expected_submodule_path: &std::path::Path) {
    println!(
        "cargo:warning=phasm-openh264 submodule not found at {}",
        expected_submodule_path.display()
    );
    println!(
        "cargo:warning=  → building openh264-sys with fail-fast stubs only."
    );
    println!(
        "cargo:warning=  → backend calls will return errors at runtime."
    );
    println!(
        "cargo:warning=  → to enable the real backend, see core/README.md"
    );
    println!(
        "cargo:warning=    \"Optional OpenH264 backend (experimental)\" section."
    );
    println!(
        "cargo:warning=    TL;DR: from the monorepo, run"
    );
    println!(
        "cargo:warning=      git submodule update --init --recursive"
    );
    println!(
        "cargo:warning=    from the phasmcore mirror, additionally clone"
    );
    println!(
        "cargo:warning=    https://github.com/cgaffga/phasm-openh264 into"
    );
    println!(
        "cargo:warning=    vendor/phasm-openh264 on the SHA pinned in"
    );
    println!(
        "cargo:warning=    core/openh264-sys/build.rs."
    );

    // Compile the stub TU as `phasm_shims` so the rest of cargo's
    // link directives (already emitted by cc) wire it up
    // automatically. Plain C — no C++ stdlib needed.
    let shim_dir = manifest_dir.join("shim");
    cc::Build::new()
        .file(shim_dir.join("phasm_openh264_stub.c"))
        .include(&shim_dir)
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("phasm_shims");

    // Surface the stub state to downstream Rust code as a cfg flag.
    // Optional consumer: tests can `#[cfg(not(phasm_openh264_stub))]`
    // to skip when the backend isn't really there.
    println!("cargo:rustc-check-cfg=cfg(phasm_openh264_stub)");
    println!("cargo:rustc-cfg=phasm_openh264_stub");

    // Re-run if the user later checks out the submodule.
    println!(
        "cargo:rerun-if-changed={}",
        expected_submodule_path.display()
    );
    println!("cargo:rerun-if-changed={}", shim_dir.display());
}
