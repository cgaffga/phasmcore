# Contributing to phasm

Thank you for your interest. **phasm** is a steganography engine
for hiding encrypted messages in media. Today it ships
production-grade JPEG image stego (Ghost / Armor / SI-UNIWARD /
Shadow) and an in-development H.264 video stego pipeline; future
media types (audio, PDF, animated formats) are planned.

This file lives alongside the public mirror at
[github.com/cgaffga/phasmcore](https://github.com/cgaffga/phasmcore)
and applies to all media surfaces. Please read it end-to-end
before opening a PR.

## How this repo relates to the upstream monorepo

phasmcore is a **mirror**. Authoritative source lives in a
private monorepo; published here under GPL-3.0-only as a single
squashed commit per sync (see commit subjects under `git log`).

Practical consequences:

- **PRs are not merged via the GitHub button.** Accepted
  contributions are ported into the monorepo with attribution,
  then the next sync brings them out as part of a squashed
  commit. Expect a delay of days between approval and visibility
  on `main`.
- **Force-pushes happen** when a sync rewrites history. Don't
  rely on commit SHAs being permanent — tagged releases are
  stable, branches are not.
- **Keep PRs small + atomic.** A 2,000-line PR is harder to port
  than three focused ones, and is more likely to be rejected on
  review-cost grounds alone.

## Code of Conduct

Participation is governed by the
[Contributor Covenant 2.1](./CODE_OF_CONDUCT.md). Reports go to
**conduct@phasm.app**. By contributing, you agree to abide by
those terms.

## Reporting security issues

**Do not open public issues for security findings.** Email
**security@phasm.app** — see [SECURITY.md](./SECURITY.md) for
the full disclosure process. Three classes of finding count as
security here:

1. **Crypto vulnerabilities** — key-derivation weaknesses, side
   channels, oracle leaks.
2. **Detectability findings** — a steganalysis attack that
   distinguishes phasm output from cover. Treat these as
   security issues; this is a stego project.
3. **Stealth-impact bugs** — anything that increases the
   embedding fingerprint (cost-function regression, mode-decision
   bug affecting Layer 3 fingerprint, etc.).

## What kinds of contributions are welcome

Welcomeness varies by **medium** and by **type**. Use the tables
below as a first filter; if you're unsure, open an issue before
coding.

### Image stego (mature surface)

| Color | Examples |
|---|---|
| 🟢 Green | Bug fixes in the JPEG codec, new test vectors, doc/typo fixes, perf improvements with benchmarks, cross-platform fixes, SIMD additions, Armor robustness improvements. |
| 🟡 Yellow (discuss first) | New cost functions, new embedding domains, new image formats (PNG-direct, WebP, HEIC, AVIF), new shadow-message strategies. |
| 🔴 Red | Changes to the AES-256-GCM-SIV / Argon2id / ChaCha20 primitives. New ciphers. Breaking JPEG byte-for-byte round-trip. Adding heavy crate dependencies. |

### Video stego (in development, source-only)

| Color | Examples |
|---|---|
| 🟢 Green | Bug fixes in the CABAC parser, encoder corrections with H.264-spec citation, small synthetic test vectors, determinism fixes. |
| 🟡 Yellow | New `EmbedDomain`, mode-decision changes (re-runs steganalysis), B-frame partition variants, capacity-API work, file attachments in CABAC stego, default `gop_size` tuning. |
| 🔴 Red | Direct ports from `x264` / `libx264` / `OpenH264` / `ffmpeg` source — see clean-room policy below. Adding `h264-encoder` to default features (Via LA AVC patent obligations). Reviving HEVC (archived behind `hevc-archive` for a reason). Bundling large fixture videos. |

### Future media types (planned)

For audio (MP3 / Opus / FLAC), PDF, document, animated images,
or any other new medium:

| Color | Examples |
|---|---|
| 🟡 Yellow | Always discuss first. New media types are architectural decisions, not drive-by PRs. |

The proposed flow for a new medium:

1. Open an issue with a design sketch — cost function, embedding
   domain, capacity bound, threat model.
2. Wait for "go" before writing code.
3. Follow the existing layout: `core/src/codec/<format>/` for the
   format parser, `core/src/stego/<format>/` for the stego layer,
   `core/tests/<format>_*.rs` for integration tests.
4. Add platform bridges only after the Rust core lands and is
   stable.

## Cross-cutting requirements (apply to all media)

These invariants are non-negotiable, regardless of which medium
your contribution targets:

1. **Determinism + bit-exactness across iOS arm64 / Android
   arm64-v8a / Linux x86_64 / macOS arm64 / WASM.** No
   `f64::sin/cos/atan2/hypot` — use `det_math`. No FMA in
   hand-SIMD kernels (FMA differs across architectures). No
   `HashMap` iteration in code paths that affect output bytes
   (the default hasher is randomized; iteration order varies
   per process).
2. **Pure-Rust core.** No new C/C++ FFI in `core/`. Platform
   bridges (`ios-bridge/`, `android-bridge/`, `wasm-bridge/`) are
   the only place FFI lives.
3. **All encryption is AES-256-GCM-SIV + Argon2id key derivation
   + ChaCha20 PRNG.** Don't propose alternatives. Per-call
   non-determinism (random salt + nonce per encrypt) is
   intentional security hygiene; round-trip is the test gate, not
   byte-equality across runs.
4. **Clean-room provenance for codec code.** If a codec subtree
   has a clean-room audit, contributions must follow the
   spec-as-reference workflow: study the upstream spec → write an
   algorithm note in `docs/design/<codec>-encoder-algorithms/` →
   implement from your own note → cite **spec sections**, not
   third-party encoder source. Never copy-paste from other
   encoders. Never name your function after one (`wels_*`,
   `x264_*`, etc.).
5. **Test vectors.** Small synthetic vectors in `tests/` are
   fine. Real-world fixtures (camera footage, full-resolution
   images) stay gitignored — place sources in
   `test-vectors/<format>/real-world/source/` and let `regen.sh`
   produce derived files to `/tmp` on demand. PRs adding large
   binary blobs will be reverted.
6. **No conditional spaghetti per format / variant.** Design ONE
   universal code path. Branching internally on "is this Baseline
   profile vs Main profile" is a smell; the universal path should
   handle both.

## Development setup

Toolchain: stable Rust (any recent version).

Feature matrix:

| Command | What it builds |
|---|---|
| `cargo build` | default features — image stego only, no video, no encoder |
| `cargo build --features video` | adds image stego + video decode-only |
| `cargo build --features video,h264-encoder` | full H.264 stack (source-only build; never enabled in cargo-dist GitHub-release binaries) |
| `cargo build --features cabac-stego` | production CABAC v2 path (implies the above) |
| `cargo build --features wasm` | WASM target (use the `wasm-bridge/` crate for the actual bundling) |

Test suite: `cargo test --features cabac-stego` runs the lib tests
(~1100+) plus the cabac-stego integration tests.

`ffmpeg` is required for some integration tests and for the CLI's
CABAC path. `brew install ffmpeg` on macOS, package manager on
Linux.

## Coding conventions

- **No `f64::sin/cos/atan2/hypot`** — use `det_math`. WASM
  compiles those to non-deterministic `Math.*`.
- **No FMA in hand-SIMD kernels.** Bit-exactness across
  architectures.
- **No `HashMap` iteration in output-affecting paths.** Lookups
  by key are fine. Use `BTreeMap` or `Vec<(K, V)>` if you need
  ordered iteration.
- **No `unwrap()` / `expect()` on production paths.** Tests +
  examples are exempt; production code must return `Result`.
- **No `// TODO` without an issue link.** Floating TODOs rot —
  tag with `// TODO #N: ...`.
- **No `unsafe`** outside SIMD intrinsic wrappers + FFI bridges.
  Justify in a comment block when you do use it.
- **Always cross-check numeric spec tables** against an
  authoritative reference (the actual spec PDF, not Stack
  Overflow). Spec-table bugs are easy to miss in review and
  expensive to debug.
- **Single-letter spec variables (`u`, `v`, `i`, `j`, `k`, `x`,
  `y`)** are allowed when they match the spec's notation. Clippy
  silences `many_single_char_names` project-wide for this reason.
- **Index-based loops over spec-indexed arrays** are allowed for
  the same reason. Clippy silences `needless_range_loop`.

## Commit + PR conventions

- Subject: imperative mood, ≤72 chars. Subsystem prefix when
  helpful: `jpeg: ...`, `h264 cabac: ...`, `stego/armor: ...`,
  `det_math: ...`, `cli: ...`. See `git log --oneline` for the
  in-tree style.
- Body: explain the *why*, not the *what*. Reviewers can read
  the diff.
- One logical change per commit. Three bug fixes = three commits.
- PR description: link the issue you're addressing; include a
  test plan checklist. The
  [PR template](.github/pull_request_template.md) prompts for
  this.
- Tests are required for behavior changes. Code without tests
  gets rejected unless the change is genuinely untestable.
- Sign your commits with `--signoff` for DCO (see License + DCO
  below).

## How reviews work

- **Solo maintainer.** Expect 1-7 days for a first response,
  longer for non-trivial changes.
- The maintainer may ask for: extra tests, benchmark re-runs, a
  steganalysis re-measure, a clean-room provenance attestation.
- "Approved" ≠ "merged." Merging happens via the monorepo port
  + next sync (see top of this file). Expect a delay between
  approval and visibility on `main`.

## License + DCO

- All contributions are under **GPL-3.0-only** (see
  [LICENSE](./LICENSE)).
- We use the **Developer Certificate of Origin (DCO)** rather
  than a CLA. Every commit must be signed off:
  ```bash
  git commit --signoff
  ```
  The signoff certifies that you wrote the code or have
  permission to contribute it under GPL-3.0. Enforcement is
  manual at port time today; please sign off so we don't have
  to ask.

## Questions

| Topic | Email |
|---|---|
| General questions | `info@phasm.app` |
| Code of Conduct reports | `conduct@phasm.app` |
| Security disclosure | `security@phasm.app` |
| Press / media | `press@phasm.app` |

For most contributor questions, prefer opening a GitHub issue
tagged `question` (when issue templates are configured) so
others can benefit from the answer.
