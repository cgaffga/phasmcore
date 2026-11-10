# Building phasm-core from source

This repository builds everything phasm-core can do, including H.264 and AV1
video steganography. The three video codecs are phasm forks:

| Fork | Fork of | Used by | Comes from | Licence |
|------|---------|---------|------------|---------|
| [phasm-openh264](https://github.com/cgaffga/phasm-openh264) | Cisco OpenH264 (C/C++) | `h264-encoder` | git submodule `vendor/phasm-openh264` | BSD-2-Clause |
| [phasm-dav1d](https://github.com/cgaffga/phasm-dav1d) | VideoLAN dav1d (C) | `av1-decoder` | git submodule `vendor/phasm-dav1d` | BSD-2-Clause |
| [libaom](https://aomedia.googlesource.com/aom) | AOMedia libaom (C), unmodified, at a release tag | `photo-encoder` | git submodule `vendor/libaom` | BSD-2-Clause + AOMedia Patent License 1.0 |
| [phasm-rav1e](https://github.com/cgaffga/phasm-rav1e) | xiph rav1e (Rust + asm) | `av1-encoder` | fetched by cargo from GitHub ([The rav1e fork](#the-rav1e-fork)) | BSD-2-Clause |

Each is pinned to an exact commit of its `phasm-stego` branch. None of them is
needed for image stego: a plain, non-recursive `git clone` builds the default
features (and `h264-decoder`) without fetching any fork.

If you only want image steganography, `cargo add phasm-core` from crates.io is
simpler: that package has no video encoders and needs no C toolchain.

## Quick start

```bash
git clone --recursive https://github.com/cgaffga/phasmcore
cd phasmcore
make check-deps        # lists missing tools, with the install command for your OS
make cli-video         # builds target/release/phasm with H.264 + AV1
```

Without make, a video build needs one extra step: cargo has to be told to use
the phasm rav1e fork (see [The rav1e fork](#the-rav1e-fork)).
`scripts/video-cargo.sh` does that and passes everything else to cargo:

```bash
sh scripts/video-cargo.sh build --release -p phasm-cli --features h264-encoder,av1-encoder
./target/release/phasm --help
```

Already cloned without `--recursive`? Run `git submodule update --init --recursive`
before a video build. Image-only builds need neither the submodules nor the
rav1e fork.

## Prerequisites

All builds need Rust. `rust-toolchain.toml` pins the compiler version, and
[rustup](https://rustup.rs) installs it on the first build. The default
features (image stego) need nothing else.

The video features build C code with meson and ninja:

| Feature | Needs |
|---------|-------|
| `h264-encoder` | meson, ninja, a C and a C++ compiler; nasm on x86 / x86_64 |
| `av1-decoder` | meson, ninja, a C compiler; nasm 2.14 or newer on x86 / x86_64 |
| `av1-encoder` | nasm on x86_64, a C compiler on aarch64 (rav1e's `asm` kernels); plus everything `av1-decoder` needs, which it implies; network access to github.com, where cargo fetches the rav1e fork |
| `photo-encoder` | CMake 3.16 or newer, ninja, a C compiler; nasm on x86_64 |

On Apple Silicon and other aarch64 machines nasm is not needed.

The CLI's `video-encode` and `video-capacity` commands also call `ffmpeg` and
`ffprobe` at **runtime** to read the cover video. You don't need them to build,
and `phasm decode` doesn't use them.

### macOS

```bash
xcode-select --install                 # C/C++ compiler
brew install meson ninja nasm          # nasm only matters on Intel Macs
brew install ffmpeg                    # optional: CLI video-encode at runtime
```

### Debian / Ubuntu

```bash
sudo apt install build-essential git meson ninja-build nasm
sudo apt install ffmpeg                # optional: CLI video-encode at runtime
```

### Fedora

```bash
sudo dnf install gcc gcc-c++ git meson ninja-build nasm
sudo dnf install ffmpeg-free           # optional; or ffmpeg from RPM Fusion
```

### Windows

Image-only builds work natively with the MSVC toolchain (`rustup` default).
The video features are **untested on native Windows**: the C codecs build with
meson, and their static-library and linker setup has only been exercised on
macOS and Linux. Use **WSL2** (Ubuntu) and follow the Debian/Ubuntu steps. If
you try native Windows anyway, you need Visual Studio Build Tools ("Desktop
development with C++"), `pip install meson ninja`, and nasm on `PATH`
(`winget install NASM.NASM`), and you should run the builds from a "Developer
PowerShell for VS". Reports and fixes are welcome.

`scripts/check-deps.sh` checks all of the above (`make check-deps` runs it).
It only reports; it never installs anything.

## Feature matrix

phasm-core features (the library, `Cargo.toml` at the repo root):

| Feature | What you get | Extra build tools | On crates.io |
|---------|--------------|-------------------|--------------|
| *(default)* | Image stego: Ghost, Armor, shadow messages, Deep Cover | none | yes |
| `parallel` | Rayon multi-threading for costs, STC and Armor decode | none | yes |
| `video` | MP4 demux/mux, H.264 bitstream parsing (base of the H.264 features) | none | yes |
| `h264-decoder` | H.264 stego **decode** (pure-Rust CABAC walker, no C) | none | yes |
| `h264-encoder` | H.264 stego **encode** through phasm-openh264; implies `h264-decoder`, `parallel`, `simd` | meson, ninja, C/C++, nasm (x86) | **no** |
| `av1-decoder` | AV1 stego decode through phasm-dav1d | meson, ninja, C, nasm (x86) | **no** |
| `av1-encoder` | AV1 stego encode through phasm-rav1e; implies `av1-decoder`, `parallel` | as `av1-decoder`, plus nasm on x86_64 | **no** |
| `photo-encoder` | photo attachments: fit a photo into a message as a small AVIF file (`photo::fit`, libaom). Showing one (`photo::decode`) needs `av1-decoder` | CMake, ninja, C, nasm (x86_64) | **no** |
| `simd` | NEON kernel for the H.264 stego walker (aarch64) | none | yes |
| `wasm` | wasm-bindgen glue for the web build (image stego only, never video) | none | yes |

The CLI (`cli/`, binary `phasm`) forwards the same names: `video`,
`h264-decoder`, `h264-encoder` (encode + decode) and `av1-encoder` (AV1 encode
+ decode; `av1-decoder` is an alias for it in the CLI). The pre-built CLI
binaries on GitHub Releases are built with default features and contain no
video code.

### Common commands

Commands with `av1-encoder` go through `scripts/video-cargo.sh` (written
`vcargo` below), which is cargo plus the rav1e patch. Everything else is plain
cargo.

```bash
vcargo() { sh scripts/video-cargo.sh "$@"; }

# Library
cargo build --release                                    # image stego
cargo build --release --features parallel
cargo build --release --features h264-decoder            # + H.264 decode, still no C
cargo build --release --features h264-encoder            # + H.264 encode (submodules)
vcargo build --release --features h264-encoder,av1-encoder,av1-decoder  # everything

# CLI
cargo build --release -p phasm-cli                                   # image stego
cargo build --release -p phasm-cli --features h264-decoder           # + H.264 decode
cargo build --release -p phasm-cli --features h264-encoder           # + H.264 encode
vcargo build --release -p phasm-cli --features h264-encoder,av1-encoder  # + AV1
vcargo install --path cli --features h264-encoder,av1-encoder        # into ~/.cargo/bin

# Tests
cargo test --release --lib                                           # library, default features
vcargo test --release --features h264-encoder,av1-encoder,av1-decoder --lib streaming_session_oh264_
vcargo test --release --features h264-encoder,av1-encoder,av1-decoder --test av1_stratified_probe
```

### Make targets

`make` with no target prints this list. Every target runs the plain cargo
commands above.

| Target | Does |
|--------|------|
| `make check-deps` | report missing tools and how to install them |
| `make submodules` | `git submodule update --init --recursive` |
| `make build` / `make build-video` | library, image only / with every video codec |
| `make cli` / `make cli-video` | the `phasm` binary, image only / with H.264 + AV1 |
| `make install-cli` / `make install-cli-video` | `cargo install` the CLI |
| `make test` | library tests, default features |
| `make test-video` | H.264 and AV1 encode-and-decode round trips |
| `make smoke-video` | CLI `video-encode` + `decode` on a sample clip, both codecs (needs ffmpeg) |
| `make publish-dry-run` | package the crates.io flavour in a temporary copy |
| `make clean` | `cargo clean` |

Variables: `JOBS=4` limits cargo's parallel jobs, `PROFILE=dev` builds unoptimised.

`build-video` and `cli-video` finish with `scripts/check-video-backends.sh`,
which fails if OpenH264 or dav1d was built as a stub (see
[Troubleshooting](#troubleshooting)).

## The rav1e fork

The AV1 encoder is [phasm-rav1e](https://github.com/cgaffga/phasm-rav1e), a
fork of rav1e with the stego hooks. It is used from GitHub source only and
is never published to crates.io.

`Cargo.toml` declares it as the crates.io package it forks:

```toml
phasm-rav1e = { package = "rav1e", version = "=0.8.0", optional = true, default-features = false, features = ["asm", "threading"] }
```

Cargo resolves every dependency in the manifest, optional ones included, to
write the lock file. A registry dependency costs only an index lookup, and
nothing is downloaded unless `av1-encoder` actually compiles it. A git
dependency, or a path into a submodule, would be cloned or required on disk
for every build, image-only included.

Builds with `av1-encoder` therefore replace that package with the fork,
using a `[patch.crates-io]` entry at the pinned commit. `scripts/video-cargo.sh`
(which the Makefile's video targets use) passes the patch on cargo's command
line. Typed out, it is:

```bash
cargo --config 'patch.crates-io.rav1e.git="https://github.com/cgaffga/phasm-rav1e"' \
      --config 'patch.crates-io.rav1e.rev="5d17e129310bc0d8a80a6218fa1965e990755a5d"' \
      build --release -p phasm-cli --features h264-encoder,av1-encoder
```

The patch is on the command line, not in `Cargo.toml`, because cargo
resolves `[patch]` entries eagerly. In the manifest it would clone the fork
for image-only builds too.

`PHASM_RAV1E_REV=<full sha>` (another fork commit) and
`PHASM_RAV1E_GIT=<url>` (for example a local `file://` clone) override the
script's defaults.

**Without the patch** (plain `cargo build --features av1-encoder`), cargo
compiles upstream rav1e 0.8.0 from crates.io. It has no stego hooks, so the
build stops at a deliberate check of the fork's `phasm_stego` module:

```
error[E0432]: unresolved import `phasm_rav1e::phasm_stego`
  --> src/codec/av1/mod.rs
   |
   | use phasm_rav1e::phasm_stego as _; // av1-encoder needs the phasm rav1e fork: build with `make build-video` or see BUILDING.md
```

The same error appears if cargo prints `patch for rav1e … was not used in
the crate graph`. That happens when the fork's version no longer matches the
`=0.8.0` requirement, and cargo then falls back to upstream.

**Cargo.lock.** The repository has no lock file. A build with the patch
records rav1e as a git source in your local `Cargo.lock`, and a build without
it records the crates.io package, so switching between image-only and video
builds in one checkout rewrites that entry. Don't combine `--locked` with
both kinds of build in one checkout.

## Using the CLI with video

```bash
phasm video-capacity clip.mp4                                  # H.264 capacity
phasm video-encode clip.mp4 -m "secret" -p "pass"              # H.264 (default)
phasm video-encode clip.mp4 --codec av1 -m "secret" -p "pass"  # AV1
phasm decode clip.stego.mp4 -p "pass"                          # detects image / H.264 / AV1
```

`phasm video-encode --help` lists the resolution presets, GOP size, shadow
messages (`--shadow-message-2` …) and file attachments.

## phasm-core as a git dependency

Cargo fetches the submodules of a git dependency itself (OpenH264 and dav1d),
so the H.264 features work straight from the repository:

```toml
[dependencies]
phasm-core = { git = "https://github.com/cgaffga/phasmcore", tag = "vX.Y.Z", features = ["h264-encoder"] }
```

With `av1-encoder`, also add the rav1e patch to **your** workspace's root
`Cargo.toml`. Cargo ignores `[patch]` in a dependency's manifest, so
phasm-core cannot do this for you:

```toml
[dependencies]
phasm-core = { git = "https://github.com/cgaffga/phasmcore", tag = "vX.Y.Z", features = ["h264-encoder", "av1-encoder"] }

[patch.crates-io]
rav1e = { git = "https://github.com/cgaffga/phasm-rav1e", rev = "5d17e129310bc0d8a80a6218fa1965e990755a5d" }
```

Take the `rev` from `PHASM_RAV1E_DEFAULT_REV` in `scripts/video-cargo.sh` of
the phasmcore commit you depend on (it is `5d17e129310bc0d8a80a6218fa1965e990755a5d`
for this one). Without the patch, the build stops at the error shown in
[The rav1e fork](#the-rav1e-fork).

Pin a `tag` or `rev` of phasmcore: the submodule commits are part of the
pinned commit, so the build is reproducible. Use a release newer than v0.4.3;
earlier commits of this repository did not carry the submodules or declare
the video features. Your build machine needs the tools from
[Prerequisites](#prerequisites) for the features you enable. Cargo fetches
the two submodules for any git dependency on this repository, even with no
video feature on (they are small). It never fetches rav1e unless you add the
patch.

The CLI installs the same way. `cargo install` uses the phasmcore
manifest, so the patch goes on the command line:

```bash
cargo install --git https://github.com/cgaffga/phasmcore phasm-cli --features h264-encoder,av1-encoder \
  --config 'patch.crates-io.rav1e.git="https://github.com/cgaffga/phasm-rav1e"' \
  --config 'patch.crates-io.rav1e.rev="5d17e129310bc0d8a80a6218fa1965e990755a5d"'
```

## Patents

phasm publishes **source code only**. This affects what you may do with a
binary you build:

- **H.264** encoders are covered by the
  [Via LA AVC patent pool](https://www.via-la.com/licensing-2/avc-h-264/).
  Building `h264-encoder` for your own use is your own business, but whoever
  **distributes** binaries that contain the H.264 encoder takes on the pool's
  licensing obligations. That is why the crates.io package has no
  `h264-encoder` feature and the pre-built CLI binaries contain no video code.
  The phasm mobile apps are licensed separately. Review the Via LA terms before
  you publish a binary built with `h264-encoder`.
- **AV1** (rav1e, dav1d) is royalty-free under the
  [AOMedia Patent License 1.0](https://aomedia.org/license/patent-license/).
- `h264-decoder` contains no H.264 encoder and reconstructs no pixels. It
  reads the stego bits from the entropy-coded bitstream.

This is not legal advice.

## What crates.io gets

The crates.io package is made from this repository at publish time:
`scripts/strip-for-crates-io.py` removes `h264-encoder`, `av1-encoder`,
`av1-decoder`, the three codec dependencies and the targets that need them.
Then `cargo publish` runs (`.github/workflows/publish-crates.yml`). A registry
package can't carry the submodule sources. `av1-encoder` would resolve, but it
needs the rav1e fork patched in, and a registry package can't give its users a
patch. The script's header explains each removal.

To check the crates.io flavour locally without touching your checkout, run
`make publish-dry-run`. It strips a temporary copy and runs
`cargo publish --dry-run`, which builds the package and uploads nothing.

## Troubleshooting

**``unresolved import `phasm_rav1e::phasm_stego` ``** (the line says "av1-encoder
needs the phasm rav1e fork"). cargo built upstream rav1e instead of the fork.
Use `make build-video` / `scripts/video-cargo.sh`, or add the patch (see
[The rav1e fork](#the-rav1e-fork)). If you did patch it, look for cargo's
`patch … was not used in the crate graph` warning: the fork's version must
match the `=0.8.0` in `Cargo.toml`.

**``failed to load source for dependency `rav1e` ``, or a failed fetch of
`phasm-rav1e`.** The video build couldn't reach github.com, or the rev isn't
on the fork. Image-only builds don't need the network beyond the crates.io
index.

**`warning: phasm-openh264 submodule not found`.** The build did not fail.
Instead `core-openh264-sys` built a fail-fast **stub**: the binary compiles,
but every H.264 encode returns an error at runtime. Fetch the submodules and
rebuild; cargo re-runs the build script by itself. To confirm a build is real,
run `scripts/check-video-backends.sh` (or `make check-video-backends`). Cargo
shows this warning only for packages in your own workspace, so with phasm-core
as a git dependency a stub build is silent. Check your own target directory:
`grep -l phasm_openh264_stub target/release/build/core-openh264-sys-*/output`
must print nothing.

**`phasm-dav1d submodule not found`, then `'dav1d/dav1d.h' file not found`.**
Same cause for `av1-decoder` / `av1-encoder`. dav1d's stub needs the
submodule's headers, so this build fails instead of linking a stub. Fetch the
submodules.

**`meson` / `ninja` not found, or `NASM build failed`.** A build tool is
missing. Run `make check-deps` to get the install command for your OS.

**`phasm-openh264 submodule HEAD … is not in the known-good SHA pin list`.**
`openh264-sys/build.rs` only accepts fork commits whose stego hook layout has
been verified. Use the commit this repository pins: `git submodule update
--recursive`. When you work on the fork yourself,
`PHASM_OPENH264_SHA_PIN_OVERRIDE=<sha>` (or `=*`) skips the check. The same
mechanism exists for dav1d (`PHASM_DAV1D_SHA_PIN_OVERRIDE`).

**`video-encode` fails with "ffmpeg not found on PATH".** The CLI needs `ffmpeg` and
`ffprobe` at runtime to read the cover video. Install them (see
[Prerequisites](#prerequisites)).

**The meson build fails after you moved the checkout, or after an interrupted
build.** Run `cargo clean` (or delete `target/`). The meson build directories
live under `target/`, and a half-configured one is not always detected.
