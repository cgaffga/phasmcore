#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Copyright (c) 2026, Christoph Gaffga
"""Turn a phasmcore checkout into the manifest that crates.io gets.

The phasmcore repository builds everything, video encoders included, from a
recursive clone (see BUILDING.md). The crates.io package is deliberately
smaller: image stego plus the `video` / `h264-decoder` reader, no encoders,
no C. This script removes the difference **in place**, right before
`cargo publish`:

    python3 scripts/strip-for-crates-io.py [ROOT]      # ROOT defaults to the repo root
    cargo publish -p phasm-core --allow-dirty           # the tree is now dirty

It is destructive by design. Run it in CI (publish-crates.yml does) or on a
throwaway copy (`make publish-dry-run` does that for you); in a working
checkout, undo it with `git checkout -- .`.

What it removes, from `Cargo.toml` and `cli/Cargo.toml`:

  * the features `h264-encoder`, `av1-encoder`, `av1-decoder`, `photo-encoder`;
  * the dependencies behind them: `core-openh264-sys`, `core-dav1d-sys`,
    `phasm-rav1e`, `core-aom-sys` (plus the comment block that describes each
    line; `core-aom-sys` is path-only and builds libaom from vendor/libaom);
  * `openh264-sys`, `dav1d-sys` and `aom-sys` from `[workspace] members`;
  * every `[[bin]]` / `[[test]]` / `[[example]]` / `[[bench]]` target whose
    `required-features` names a removed feature, and the `src/bin/*.rs`
    source of each removed bin;
  * and it adds `exclude = ["vendor/"]` to `[package]`, so the vendored C of
    a recursive clone can never end up inside the .crate;
  * it declares the removed features to rustc (`[lints.rust]
    unexpected_cfgs` check-cfg): the source keeps its
    `#[cfg(feature = "h264-encoder")]` gates, which are then always off, and
    without the declaration every one of them warns "unexpected `cfg`
    condition value" (91 warnings in a `h264-decoder` build).

Why each piece cannot go to crates.io
-------------------------------------

`h264-encoder`: the `core-openh264-sys` dep is path-only (no crates.io
version), so `cargo publish` rejects the manifest. Even if it had a version,
shipping a feature that pulls in the OpenH264 fork via build-script
meson+ninja runs afoul of "no H.264 binary code in build/deploy" (Via LA AVC
patent obligations). Since the pure-Rust encoder was retired
(video-retirement Phase 3) and `h264-encoder` now denotes the OpenH264
encoder, crates.io ships NO H.264 encoder; `h264-decoder` (the pure-Rust
bin-walker, no C) remains as the decode-only path.

`av1-encoder`: `phasm-rav1e` is declared as the crates.io package `rav1e`
(`version = "=0.8.0"`), so unlike the two -sys crates it would pass
`cargo publish`. It still has to go. The stego hooks exist only in the
cgaffga/phasm-rav1e fork, which is consumed from GitHub source and never
published to crates.io; phasmcore builds get it through a `[patch.crates-io]`
entry (scripts/video-cargo.sh, or the snippet in BUILDING.md). Cargo ignores
`[patch]` in a dependency's manifest, so a registry user of phasm-core could
not be given that patch, and `--features av1-encoder` would compile UPSTREAM
rav1e and then fail at the fork guard in src/codec/av1/mod.rs. A feature
that cannot build does not belong in the published manifest; the dependency
line goes with it so the registry package does not even resolve rav1e.

`av1-decoder`: `core-dav1d-sys` is path-only like `core-openh264-sys`, and
its `build.rs` stub mode is not self-contained: the stub
`phasm_dav1d_stub.c` `#include`s `dav1d/dav1d.h`, which lives in
`vendor/phasm-dav1d/include/`. A registry `--features av1-decoder` build
would fail with "fatal error: 'dav1d/dav1d.h' file not found".

If the AV1 forks are ever published to crates.io under their own names (the
current decision is that phasm-rav1e is not), and dav1d-sys's stub is
self-contained, drop the AV1 entries here. The AV1
patent posture is already royalty-free (AOMedia Patent License 1.0); the gap
is purely toolchain plumbing.

The bin sources go too because cargo AUTO-DISCOVERS `src/bin/*.rs`: with the
`[[bin]]` block gone, the file would build unconditionally and fail, since
each is cfg-gated to nothing without its feature. And cargo-dist enumerates
`[[bin]]` targets from the manifest, so a dangling target fails the Release
workflow ("failed to find bin av1_block_hist"); that bit the v0.4.0 tag.
"""

from __future__ import annotations

import pathlib
import re
import sys

STRIPPED_FEATURES = ("h264-encoder", "av1-encoder", "av1-decoder", "photo-encoder")
STRIPPED_DEPS = ("core-openh264-sys", "core-dav1d-sys", "phasm-rav1e", "core-aom-sys")
STRIPPED_MEMBERS = ("openh264-sys", "dav1d-sys", "aom-sys")
DROP_PREFIXES = tuple(f"{name} =" for name in STRIPPED_FEATURES + STRIPPED_DEPS)
TARGET_HEADERS = ("[[test]]", "[[bin]]", "[[example]]", "[[bench]]")
CHECK_CFG_FEATURES = "cfg(feature, values(" + ",".join(f'"{n}"' for n in STRIPPED_FEATURES) + "))"


def drop_lines(lines: list[str], log: list[str]) -> list[str]:
    """Remove feature / dependency lines and the comment block above each."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if any(stripped.startswith(p) for p in DROP_PREFIXES):
            log.append(f"line: {stripped.split('=')[0].strip()}")
            # rewind through the preceding comment lines
            j = len(out) - 1
            while j >= 0 and out[j].lstrip().startswith("#"):
                j -= 1
            out = out[: j + 1]
            i += 1
            # swallow trailing blank lines
            while i < len(lines) and lines[i].strip() == "":
                i += 1
            continue
        out.append(lines[i])
        i += 1
    return out


def drop_targets(lines: list[str], log: list[str]) -> tuple[list[str], list[str]]:
    """Remove target blocks that require a stripped feature.

    Returns the new lines and the source paths of the removed bins.
    """
    out: list[str] = []
    removed_bin_sources: list[str] = []
    k = 0
    while k < len(lines):
        if lines[k].strip() in TARGET_HEADERS:
            # The block is the header plus its key lines; it ends at the
            # first blank line, comment or next table, so the comment that
            # introduces the NEXT target never counts as part of this one.
            block = [lines[k]]
            m = k + 1
            while m < len(lines) and lines[m].strip() and not lines[m].lstrip().startswith(("[", "#")):
                block.append(lines[m])
                m += 1
            body = "".join(block)
            req = re.search(r"^\s*required-features\s*=\s*\[([^\]]*)\]", body, re.M)
            if req and any(f'"{feat}"' in req.group(1) for feat in STRIPPED_FEATURES):
                name = re.search(r'^\s*name\s*=\s*"([^"]+)"', body, re.M)
                path = re.search(r'^\s*path\s*=\s*"([^"]+)"', body, re.M)
                kind = lines[k].strip()
                log.append(f"target: {kind} {name.group(1) if name else '?'}")
                if kind == "[[bin]]" and name:
                    removed_bin_sources.append(
                        path.group(1) if path else f"src/bin/{name.group(1)}.rs"
                    )
                # drop the block plus its preceding comment / blank lines
                while out and out[-1].lstrip().startswith("#"):
                    out.pop()
                while out and out[-1].strip() == "":
                    out.pop()
                k = m
                continue
            out.extend(block)
            k = m
            continue
        out.append(lines[k])
        k += 1
    return out, removed_bin_sources


def drop_members(text: str, log: list[str]) -> str:
    for sub in STRIPPED_MEMBERS:
        new = re.sub(
            r'(members\s*=\s*\[[^\]]*?),?\s*"' + re.escape(sub) + '"', r"\1", text
        )
        if new != text:
            log.append(f"workspace member: {sub}")
        text = new
    return text


def add_vendor_exclude(text: str, log: list[str]) -> str:
    """Add `exclude = ["vendor/"]` to [package] (root manifest only)."""
    pkg = re.search(r"^\[package\]\s*$", text, re.M)
    if not pkg:
        return text
    nxt = re.search(r"^\[", text[pkg.end():], re.M)
    end = pkg.end() + (nxt.start() if nxt else len(text[pkg.end():]))
    section = text[pkg.end():end]
    if re.search(r"^\s*exclude\s*=", section, re.M):
        if '"vendor/"' not in section and '"vendor"' not in section:
            sys.exit(
                "strip-for-crates-io: [package] already has an `exclude` list "
                "without vendor/; add it there by hand."
            )
        return text
    if re.search(r"^\s*include\s*=", section, re.M):
        return text  # an include list already keeps vendor/ out
    section = section.rstrip("\n") + '\nexclude = ["vendor/"]\n\n'
    log.append('package: exclude = ["vendor/"]')
    return text[: pkg.end()] + section + text[end:].lstrip("\n")


def declare_stripped_features(text: str, log: list[str]) -> str:
    """Tell rustc the stripped features exist, so their `#[cfg]` gates stay quiet.

    The source still carries `#[cfg(feature = "h264-encoder")]` and friends
    (they are simply never on); a manifest that no longer declares the feature
    makes rustc warn at each one. Declared through `check-cfg` in
    `[lints.rust] unexpected_cfgs`: appended to the existing list when the
    manifest has one, else the line (and table, if need be) is added.
    """
    entry = f"'{CHECK_CFG_FEATURES}'"
    existing = re.search(
        r"^unexpected_cfgs\s*=\s*\{[^\n]*?check-cfg\s*=\s*\[([^\]\n]*)\][^\n]*\}[ \t]*$",
        text,
        re.M,
    )
    if existing:
        if CHECK_CFG_FEATURES in existing.group(0):
            return text
        items = existing.group(1).rstrip()
        sep = "" if not items or items.endswith(",") else ","
        line = (
            existing.group(0)[: existing.start(1) - existing.start(0)]
            + items + sep + (" " if items else "") + entry
            + existing.group(0)[existing.end(1) - existing.start(0):]
        )
        text = text[: existing.start()] + line + text[existing.end():]
    else:
        line = f'unexpected_cfgs = {{ level = "warn", check-cfg = [{entry}] }}\n'
        table = re.search(r"^\[lints\.rust\]\s*$", text, re.M)
        if table:
            text = text[: table.end()] + "\n" + line.rstrip("\n") + text[table.end():]
        else:
            text = text.rstrip("\n") + "\n\n[lints.rust]\n" + line
    log.append("lints: unexpected_cfgs check-cfg for the stripped features")
    return text


def strip_manifest(path: pathlib.Path, root: pathlib.Path, is_root: bool) -> list[str]:
    log: list[str] = []
    lines = path.read_text().splitlines(keepends=True)
    lines = drop_lines(lines, log)
    lines, bin_sources = drop_targets(lines, log)
    text = drop_members("".join(lines), log)
    if is_root:
        text = add_vendor_exclude(text, log)
    text = declare_stripped_features(text, log)
    path.write_text(text)
    for rel in bin_sources:
        src = path.parent / rel
        if src.is_file():
            src.unlink()
            log.append(f"bin source: {src.relative_to(root)}")
    return log


def verify(root: pathlib.Path) -> None:
    """Fail loudly if anything that cannot go to crates.io survived."""
    problems: list[str] = []
    for rel in ("Cargo.toml", "cli/Cargo.toml"):
        p = root / rel
        if not p.is_file():
            continue
        text = p.read_text()
        for name in STRIPPED_FEATURES + STRIPPED_DEPS:
            if re.search(rf"^\s*{re.escape(name)}\s*=", text, re.M):
                problems.append(f"{rel}: `{name}` is still declared")
        if re.search(r'"[^"]*vendor/', text) and rel == "Cargo.toml":
            for line in text.splitlines():
                if "vendor/" in line and not line.lstrip().startswith(("#", "exclude")):
                    problems.append(f"{rel}: still references vendor/: {line.strip()}")
        for sub in STRIPPED_MEMBERS:
            m = re.search(r"members\s*=\s*\[([^\]]*)\]", text)
            if m and f'"{sub}"' in m.group(1):
                problems.append(f"{rel}: workspace member `{sub}` remains")
    if problems:
        sys.exit("strip-for-crates-io: verification failed:\n  " + "\n  ".join(problems))


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if any(a in ("-h", "--help") for a in sys.argv[1:]):
        print(__doc__)
        return
    root = pathlib.Path(args[0] if args else pathlib.Path(__file__).resolve().parent.parent)
    root = root.resolve()
    manifest = root / "Cargo.toml"
    if not manifest.is_file() or 'name = "phasm-core"' not in manifest.read_text():
        sys.exit(f"strip-for-crates-io: {root} is not a phasm-core checkout")

    print(f"strip-for-crates-io: stripping {root}")
    for rel, is_root in (("Cargo.toml", True), ("cli/Cargo.toml", False)):
        p = root / rel
        if not p.is_file():
            continue
        for entry in strip_manifest(p, root, is_root):
            print(f"  - {rel}: {entry}")
    verify(root)
    print("strip-for-crates-io: done; publish with `cargo publish -p phasm-core --allow-dirty`")


if __name__ == "__main__":
    main()
