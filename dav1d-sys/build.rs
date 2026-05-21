// SPDX-License-Identifier: GPL-3.0-only
// Copyright (c) 2026, Christoph Gaffga
//
// build.rs for core-dav1d-sys.
//
// W2.3 (2026-05-21) — real meson + ninja static build of the
// phasm-dav1d submodule at ../../vendor/phasm-dav1d. Emits Cargo
// link directives so downstream Rust code can call into libdav1d
// via extern "C". Falls back to a fail-fast stub build when the
// submodule is absent (e.g., source archive distributions).
//
// W2.4 (2026-05-21) — iOS / Android meson cross-file generation
// (parallel port of openh264-sys's ios_cross / android_cross
// helpers). Host builds get None (meson auto-detects); cross-
// compile targets get a generated meson cross-file path. dav1d is
// plain C — the cross-file's cpp_* fields are mirror-style only;
// meson ignores cpp config for a C-only project.
//
// Prerequisites checked at build time:
//   - vendor/phasm-dav1d submodule initialized (meson.build present)
//   - meson + ninja + nasm on PATH (host build)
//
// Behaviour:
//   - First build: meson setup _build_static --default-library=static
//                  --buildtype=release -Denable_tools=false
//                  -Denable_tests=false ; ninja -C _build_static
//   - Incremental: ninja -C _build_static (meson setup idempotent;
//     skipped when build dir already has build.ninja)
//   - Re-run trigger: any change under vendor/phasm-dav1d/src/

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR"));
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR"));

    // Locate submodule at <project root>/vendor/phasm-dav1d (two
    // levels up from core/dav1d-sys/).
    let submodule_path_raw = manifest_dir
        .join("..")
        .join("..")
        .join("vendor")
        .join("phasm-dav1d");

    let submodule_path = match submodule_path_raw.canonicalize() {
        Ok(p) if p.join("meson.build").exists() => p,
        _ => {
            build_stub(&manifest_dir, &submodule_path_raw);
            return;
        }
    };

    // SHA pin: catch silent submodule drift. v0.3-AV1 starts with the
    // 1718ff9a (videolan/dav1d master at 2026-05-15, parity-clean)
    // pin. Add new SHAs to PINNED_SHAS as upstream rebases land on
    // phasm-stego.
    //
    // Override during fork-side dev with:
    //   PHASM_DAV1D_SHA_PIN_OVERRIDE=<sha>  (accept specific SHA)
    //   PHASM_DAV1D_SHA_PIN_OVERRIDE=*       (accept any SHA)
    {
        const PINNED_SHAS: &[&str] = &[
            // 2026-05-15: phasm-stego branch initial state (parity-
            // clean with videolan/dav1d master). No stego hooks yet —
            // those land in W3.D.2+ as the decoder-side counterpart to
            // phasm-rav1e's WriterStego intercept.
            "1718ff9aded99f0a89f5c7940d6afb8948301e33",
            // 2026-05-21: W3.D.2.1 + W3.D.2.2 — Dav1dPhasmHooks
            // struct + msac.c hook call (C-path). Foundation patches
            // per dav1d-hook-sites.md § 2.
            "42650b80a25c97429c207c60c19defc4126acb3b",
            // 2026-05-21: W3.D.2.3 — NEON + SSE2 asm wrappers for
            // bool_equi hook. Required for hooks to fire on
            // production ARM/x86 builds (where asm path bypasses
            // the C variant). Per dav1d-hook-sites.md § 4.
            "b9ec593b9ebb43e881d67fa63d6c70761fd71d1b",
            // 2026-05-21: W3.D.2.4 — hook propagation Settings →
            // Dav1dContext → MsacContext at tile init. Closes the
            // wiring gap so user-supplied callbacks actually fire
            // on the runtime decode path. Per dav1d-hook-sites.md § 5.
            "1648fff68ebc3f6ef8071b4bfe8f6e8ea8f7eaec",
            // 2026-05-21: W3.D.3 — AC sign + golomb tail tag site
            // patches in recon_tmpl.c (qmatrix + no-qmatrix paths).
            // Symmetric to phasm-rav1e W3.9.0 encoder-side tagging.
            // Per dav1d-hook-sites.md § 3.
            "715fee0a07d48e63f62f039ba602ff8430441bd5",
            // 2026-05-21: W3.D.3-fix — DC golomb tail tag site
            // patches in recon_tmpl.c (qmatrix + no-qmatrix DC
            // residual sites). Closes the W3.10.4 GOLOMB tag
            // attribution gap. Paired with phasm-rav1e 6254b700
            // (replay() resets dest tag).
            "619908efee1279c99f2fe56bc0d226848085ea3a",
            // 2026-05-21: README update — Phasm-fork-specific
            // overview replacing upstream README. Phase A status
            // (W3.D shipped), license posture, sibling-fork links.
            // No source change.
            "8966e7226ed4380cb04e2b517db56d5c1d4e83d3",
        ];

        // phasm-rav1e PINNED_SHAS (parallel pin list — documentation
        // only). Most recent: 6254b700 (W3.10.4-fix replay() resets
        // dest tag — root-cause for AC + GOLOMB tag attribution gap).
        let head_output = Command::new("git")
            .arg("-C")
            .arg(&submodule_path)
            .arg("rev-parse")
            .arg("HEAD")
            .output();
        if let Ok(out) = head_output {
            if out.status.success() {
                let sha = String::from_utf8_lossy(&out.stdout).trim().to_string();
                let override_env = std::env::var("PHASM_DAV1D_SHA_PIN_OVERRIDE").ok();
                let accept = override_env.as_deref() == Some("*")
                    || override_env.as_deref() == Some(&sha)
                    || PINNED_SHAS.iter().any(|s| *s == sha);
                if !accept {
                    panic!(
                        "phasm-dav1d submodule HEAD {} is not in the known-good\n\
                         SHA pin list {:?}.\n\
                         When phasm-stego accumulates hooks, the hook firing\n\
                         convention + FFI surface in core-dav1d-sys depends on a\n\
                         specific decoder layout. Verify empirically before\n\
                         adding to PINNED_SHAS.\n\n\
                         Override during fork-side dev:\n\
                         PHASM_DAV1D_SHA_PIN_OVERRIDE={}\n\
                         PHASM_DAV1D_SHA_PIN_OVERRIDE=* (accept any)",
                        sha, PINNED_SHAS, sha
                    );
                }
                println!(
                    "cargo:rerun-if-changed={}",
                    submodule_path.join(".git").display()
                );
            }
        }
        println!("cargo:rerun-if-env-changed=PHASM_DAV1D_SHA_PIN_OVERRIDE");
    }

    check_tool_available(
        "meson",
        "Install via `brew install meson` (macOS) or `pip install meson` (Linux).",
    );
    check_tool_available(
        "ninja",
        "Install via `brew install ninja` (macOS) or `apt install ninja-build` (Linux).",
    );
    // nasm is required only on x86; not enforced here (meson detects
    // and errors loudly if it's needed and missing).

    // W2.4: generate meson cross-file for cross-targets (iOS / Android).
    // Host builds get None (meson auto-detects).
    let cross_file = generate_meson_cross_file(&out_dir);

    let build_dir = out_dir.join("dav1d-static");

    let needs_setup = !build_dir.join("build.ninja").exists();
    if needs_setup {
        std::fs::create_dir_all(&build_dir).expect("create build dir");
        let mut cmd = Command::new("meson");
        cmd.arg("setup")
            .arg("--default-library=static")
            .arg("--buildtype=release")
            // Skip dav1d's CLI tools and tests — we only need libdav1d.a.
            .arg("-Denable_tools=false")
            .arg("-Denable_tests=false")
            // Skip docs (saves time + avoids doxygen dep check).
            .arg("-Denable_docs=false")
            // Disable b_lto — clashes with cargo's own LTO downstream
            // (same lesson openh264-sys learned the hard way).
            .arg("-Db_lto=false");
        if let Some(ref path) = cross_file {
            cmd.arg(format!("--cross-file={}", path.display()));
        }
        cmd.arg(&build_dir).arg(&submodule_path);
        let status = cmd
            .status()
            .expect("failed to spawn meson");
        if !status.success() {
            panic!("meson setup failed (exit {:?}); see output above", status.code());
        }
    }

    let status = Command::new("ninja")
        .arg("-C")
        .arg(&build_dir)
        .status()
        .expect("failed to spawn ninja");
    if !status.success() {
        panic!("ninja build failed (exit {:?}); see output above", status.code());
    }

    // libdav1d.a lands at <build_dir>/src/libdav1d.a in the standard
    // meson layout for dav1d's project structure (src/meson.build's
    // static_library() target).
    let lib_search_dir = build_dir.join("src");
    println!("cargo:rustc-link-search=native={}", lib_search_dir.display());
    println!("cargo:rustc-link-lib=static=dav1d");

    // Compile the phasm_dav1d_stub.c shim alongside libdav1d.a so
    // downstream stub functions remain available. When W3.3 adds
    // real stego hooks to phasm-dav1d itself (via fork patches),
    // the shim will pivot to a phasm_dav1d_hooks.c that registers
    // those hooks via the dav1d API. For W2.3 the stub is still
    // compiled so phasm_dav1d_stub_present() remains callable.
    let shim_dir = manifest_dir.join("shim");
    cc::Build::new()
        .file(shim_dir.join("phasm_dav1d_stub.c"))
        .include(&shim_dir)
        // phasm-stego (W3.D.2.5): shim's phasm_dav1d_abi_* reporter
        // functions #include <dav1d/dav1d.h> to cross-check the
        // Dav1dPhasmHooks struct layout + tag constants against the
        // Rust FFI declarations. Need the dav1d include dir on the
        // header search path.
        .include(submodule_path.join("include"))
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("phasm_dav1d_shims");
    println!("cargo:rerun-if-changed={}", shim_dir.display());

    // dav1d's headers live at include/dav1d/. Expose for downstream
    // bindgen consumers if we ever adopt it.
    println!(
        "cargo:include={}",
        submodule_path.join("include").display()
    );

    // dav1d transitively depends on libdl (Linux), libm, librt (Linux).
    // libc++ NOT needed (dav1d is plain C). Conditional per target OS.
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    match target_os.as_str() {
        "linux" => {
            println!("cargo:rustc-link-lib=m");
            println!("cargo:rustc-link-lib=dl");
            println!("cargo:rustc-link-lib=rt");
        }
        "macos" | "ios" => {
            // libm is in libSystem on Apple; no explicit link needed.
        }
        "android" => {
            println!("cargo:rustc-link-lib=m");
            // dl + log come from libc / Bionic; no extra link needed.
        }
        _ => {}
    }

    println!(
        "cargo:rerun-if-changed={}",
        submodule_path.join("src").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        submodule_path.join("meson.build").display()
    );
    println!(
        "cargo:rerun-if-changed={}",
        submodule_path.join("include").display()
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

/// W2.4 — generate a meson cross-file for the current cargo target.
/// Returns `Some(path)` for known cross-compile destinations (iOS
/// device / simulator / Android arm64 / Android x86_64); returns
/// `None` for host builds (meson auto-detects) and for unsupported
/// cross-targets (meson tries host detection, probably fails loudly).
///
/// dav1d's meson.build understands `system = 'ios'` and
/// `system = 'android'` for its NEON / cpu-feature detection; no
/// fork-side patches needed.
fn generate_meson_cross_file(out_dir: &std::path::Path) -> Option<PathBuf> {
    let target = env::var("TARGET").unwrap_or_default();
    let host = env::var("HOST").unwrap_or_default();

    if target == host || target.is_empty() {
        return None;
    }

    let content = match target.as_str() {
        // iOS device (arm64).
        "aarch64-apple-ios" => ios_cross(/*sim=*/ false, "arm64", "aarch64"),
        // iOS simulator on Apple Silicon.
        "aarch64-apple-ios-sim" => ios_cross(/*sim=*/ true, "arm64", "aarch64"),
        // iOS simulator on Intel mac (legacy CI matrices).
        "x86_64-apple-ios" => ios_cross(/*sim=*/ true, "x86_64", "x86_64"),
        // Android arm64 (primary mobile target).
        "aarch64-linux-android" => android_cross("aarch64-linux-android", "arm64", "aarch64"),
        // Android x86_64 (emulator).
        "x86_64-linux-android" => android_cross("x86_64-linux-android", "x86_64", "x86_64"),
        // macOS x86_64 cross from Apple Silicon host — used by W7
        // (cross-arch determinism check via Rosetta 2). Apple's
        // bundled clang accepts `-arch x86_64` natively on aarch64
        // hosts; the macOS SDK serves both arches.
        "x86_64-apple-darwin" => macos_cross("x86_64", "x86_64"),
        // macOS aarch64 cross from x86_64 host (less common; CI
        // matrices that build Apple Silicon binaries from Intel).
        "aarch64-apple-darwin" => macos_cross("arm64", "aarch64"),
        _ => return None,
    };

    let path = out_dir.join("phasm-dav1d-cross.ini");
    std::fs::write(&path, content).expect("write meson cross-file");

    println!("cargo:rerun-if-env-changed=TARGET");
    println!("cargo:rerun-if-env-changed=HOST");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=NDK_HOME");

    Some(path)
}

/// Build a meson cross-file for an iOS device or simulator target.
///
/// Uses `clang -target <triple>` for the deployment-target encoding
/// (clang derives -mios-version-min vs -mios-simulator-version-min
/// from the triple). Min iOS = 15.0 to match phasm-ios deployment
/// target. Mirrors openh264-sys's ios_cross() verbatim modulo file
/// naming.
fn ios_cross(simulator: bool, arch: &str, cpu_family: &str) -> String {
    let sdk = if simulator { "iphonesimulator" } else { "iphoneos" };
    let min_ver = "15.0";
    let suffix = if simulator { "-simulator" } else { "" };
    let triple = format!("{arch}-apple-ios{min_ver}{suffix}");
    format!(
        "[binaries]\n\
         c = ['xcrun', '--sdk', '{sdk}', 'clang']\n\
         cpp = ['xcrun', '--sdk', '{sdk}', 'clang++']\n\
         ar = ['xcrun', '--sdk', '{sdk}', 'ar']\n\
         strip = ['xcrun', '--sdk', '{sdk}', 'strip']\n\
         \n\
         [built-in options]\n\
         c_args = ['-target', '{triple}']\n\
         cpp_args = ['-target', '{triple}']\n\
         c_link_args = ['-target', '{triple}']\n\
         cpp_link_args = ['-target', '{triple}']\n\
         \n\
         [host_machine]\n\
         system = 'ios'\n\
         cpu_family = '{cpu_family}'\n\
         cpu = '{arch}'\n\
         endian = 'little'\n"
    )
}

/// Build a meson cross-file for a same-OS (macOS) different-arch
/// target. Apple's bundled clang accepts `-arch <arch>` natively
/// without needing `xcrun --sdk`; the macOS SDK serves both arm64
/// and x86_64. Used by W7 (cross-arch determinism via Rosetta 2).
fn macos_cross(arch: &str, cpu_family: &str) -> String {
    format!(
        "[binaries]\n\
         c = ['clang', '-arch', '{arch}']\n\
         cpp = ['clang++', '-arch', '{arch}']\n\
         ar = ['ar']\n\
         strip = ['strip']\n\
         \n\
         [built-in options]\n\
         c_args = []\n\
         cpp_args = []\n\
         c_link_args = []\n\
         cpp_link_args = []\n\
         \n\
         [host_machine]\n\
         system = 'darwin'\n\
         cpu_family = '{cpu_family}'\n\
         cpu = '{arch}'\n\
         endian = 'little'\n"
    )
}

/// Build a meson cross-file for an Android NDK target.
///
/// Target API auto-detected (highest API-versioned clang shim in
/// the NDK install), capped at MIN_API. PHASM_ANDROID_API env var
/// overrides. Adds -Wl,-z,max-page-size=16384 to link args (Android
/// 15+ arm64 requires 16KB page alignment for upload acceptance).
///
/// Mirrors openh264-sys's android_cross() verbatim modulo dav1d-
/// specific tweaks (no C++ source so cpp_args/cxx unused in practice
/// but still emitted for symmetry).
fn android_cross(clang_prefix: &str, arch: &str, cpu_family: &str) -> String {
    const MIN_API: u32 = 24; // NDK ABI floor; phasm-android minSdk
    let ndk = locate_android_ndk();
    let host_tag = ndk_host_tag();
    let toolchain = std::path::Path::new(&ndk)
        .join("toolchains")
        .join("llvm")
        .join("prebuilt")
        .join(&host_tag)
        .join("bin");
    let api = resolve_android_api(&toolchain, clang_prefix, MIN_API);
    let cc = toolchain.join(format!("{clang_prefix}{api}-clang"));
    let cxx = toolchain.join(format!("{clang_prefix}{api}-clang++"));
    let ar = toolchain.join("llvm-ar");
    let strip = toolchain.join("llvm-strip");

    if !cc.exists() {
        panic!(
            "Android NDK clang not found at expected path:\n  {}\n\
             ANDROID_NDK_HOME={}\n\
             host_tag={}\n\
             api={}\n\
             Verify the NDK install includes API-{} clang shims.",
            cc.display(),
            ndk,
            host_tag,
            api,
            api
        );
    }

    // Also point cc::Build at the same NDK toolchain so the shim
    // compilation step uses the right clang. Mirrors openh264-sys.
    let target = env::var("TARGET").unwrap_or_default();
    let target_underscored = target.replace('-', "_");
    // SAFETY: set_var is single-threaded here — build.rs runs before
    // cc::Build spawns its compiler subprocess. No concurrent reads.
    unsafe {
        env::set_var(format!("CC_{target}"), cc.as_os_str());
        env::set_var(format!("CXX_{target}"), cxx.as_os_str());
        env::set_var(format!("AR_{target}"), ar.as_os_str());
        env::set_var(format!("CC_{target_underscored}"), cc.as_os_str());
        env::set_var(format!("CXX_{target_underscored}"), cxx.as_os_str());
        env::set_var(format!("AR_{target_underscored}"), ar.as_os_str());
    }

    format!(
        "[binaries]\n\
         c = '{cc}'\n\
         cpp = '{cxx}'\n\
         ar = '{ar}'\n\
         strip = '{strip}'\n\
         \n\
         [built-in options]\n\
         c_link_args = ['-Wl,-z,max-page-size=16384']\n\
         cpp_link_args = ['-Wl,-z,max-page-size=16384']\n\
         \n\
         [host_machine]\n\
         system = 'android'\n\
         cpu_family = '{cpu_family}'\n\
         cpu = '{arch}'\n\
         endian = 'little'\n",
        cc = cc.display(),
        cxx = cxx.display(),
        ar = ar.display(),
        strip = strip.display(),
    )
}

fn locate_android_ndk() -> String {
    for var in &["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT", "NDK_HOME"] {
        if let Ok(p) = env::var(var) {
            if !p.is_empty() {
                return p;
            }
        }
    }
    panic!(
        "Android cross-compile needs the NDK install path. Set one of:\n\
         \n  ANDROID_NDK_HOME=/path/to/android-ndk-r28\n\
         \n  ANDROID_NDK_ROOT=...\n\
         \n  NDK_HOME=...\n\
         \nInstall via `sdkmanager 'ndk;28.0.13004108'` or download from\n\
         https://developer.android.com/ndk/downloads ."
    )
}

fn resolve_android_api(toolchain_bin: &std::path::Path, clang_prefix: &str, min_api: u32) -> u32 {
    println!("cargo:rerun-if-env-changed=PHASM_ANDROID_API");
    if let Ok(s) = env::var("PHASM_ANDROID_API") {
        if let Ok(n) = s.parse::<u32>() {
            return n;
        }
    }

    let needle = format!("{clang_prefix}");
    let suffix = "-clang";
    let mut best: Option<u32> = None;
    if let Ok(rd) = std::fs::read_dir(toolchain_bin) {
        for entry in rd.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if let Some(rest) = name.strip_prefix(&needle) {
                if let Some(api_str) = rest.strip_suffix(suffix) {
                    if let Ok(n) = api_str.parse::<u32>() {
                        if n >= min_api && best.is_none_or(|b| n > b) {
                            best = Some(n);
                        }
                    }
                }
            }
        }
    }

    best.unwrap_or_else(|| {
        panic!(
            "No Android NDK clang shim found for prefix '{}' (API >= {}) in:\n  {}\n\
             Override with PHASM_ANDROID_API=<N> or install a newer NDK.",
            clang_prefix,
            min_api,
            toolchain_bin.display()
        )
    })
}

fn ndk_host_tag() -> String {
    let host = env::var("HOST").unwrap_or_default();
    if host.contains("apple-darwin") {
        "darwin-x86_64".to_string()
    } else if host.contains("linux") {
        "linux-x86_64".to_string()
    } else if host.contains("windows") {
        "windows-x86_64".to_string()
    } else {
        panic!("Unsupported NDK host triple: {host}")
    }
}

/// Stub fallback when vendor/phasm-dav1d submodule is absent.
/// Mirrors openh264-sys's build_stub. Compiles the no-op
/// phasm_dav1d_stub.c so cargo link succeeds; downstream FFI calls
/// (none yet in W2.3) return error codes at runtime.
fn build_stub(manifest_dir: &std::path::Path, expected_submodule_path: &std::path::Path) {
    println!(
        "cargo:warning=phasm-dav1d submodule not found at {}",
        expected_submodule_path.display()
    );
    println!("cargo:warning=  → building dav1d-sys with fail-fast stubs only.");
    println!("cargo:warning=  → FFI calls return error codes at runtime.");
    println!("cargo:warning=  → to enable the real backend, run:");
    println!("cargo:warning=      git submodule update --init --recursive");

    let shim_dir = manifest_dir.join("shim");
    cc::Build::new()
        .file(shim_dir.join("phasm_dav1d_stub.c"))
        .include(&shim_dir)
        .flag_if_supported("-std=c11")
        .warnings(true)
        .compile("phasm_dav1d_shims");

    println!("cargo:rustc-check-cfg=cfg(phasm_dav1d_stub)");
    println!("cargo:rustc-cfg=phasm_dav1d_stub");
    println!(
        "cargo:rerun-if-changed={}",
        expected_submodule_path.display()
    );
    println!("cargo:rerun-if-changed={}", shim_dir.display());
}
