# Security policy

## Supported versions

phasm is in active development. Only the `main` branch of
[phasmcore](https://github.com/cgaffga/phasmcore) receives
security fixes. Tagged releases get patches for high-severity
issues for as long as the maintainer can support them; check the
release notes for the most recent patched version.

## Reporting a vulnerability

**Do not open a public GitHub issue for security findings.**
Email **security@phasm.app**. The address forwards to the
maintainer; replies will come from the maintainer's primary
inbox.

You can also use GitHub's
[Report a vulnerability](https://github.com/cgaffga/phasmcore/security/advisories/new)
button on the repo's Security tab — that opens a private
advisory thread.

In your report, please include:

- A description of the issue and its impact.
- Steps to reproduce, ideally with a minimal test case.
- The phasm version (commit SHA or tagged release) you tested
  against.
- The platform (OS, architecture, build feature flags).
- Any proof-of-concept code or sample inputs (attach as files,
  do not paste binary content inline).

## What counts as a security issue

phasm is a steganography + encryption project, so the security
surface is broader than the typical Rust crate. The following
categories all count:

1. **Crypto vulnerabilities** — weaknesses in key derivation
   (Argon2id), authenticated encryption (AES-256-GCM-SIV), or
   the deterministic PRNG (ChaCha20). Side-channel leaks
   (timing, cache, EM) on the platforms phasm targets.
2. **Detectability findings** — a practical steganalysis attack
   that distinguishes phasm-encoded media from cover with
   meaningfully better-than-random accuracy. This is a security
   issue here even though the traditional CVE vocabulary doesn't
   describe it.
3. **Stealth-impact bugs** — implementation defects that
   measurably increase the embedding fingerprint without
   improving capacity (e.g., a cost-function regression that
   biases STC choices, a mode-decision bug that produces
   unusual encoder traces).

Bugs in robustness, capacity, performance, or general
correctness are normal issues — please file them publicly.

## Disclosure

The default disclosure window is **90 days** from the date
the maintainer acknowledges the report. We're flexible: shorter
if the fix is straightforward and the issue is high-severity,
longer if a coordinated disclosure with downstream consumers
is needed.

If we cannot reproduce or do not consider the report a security
issue, we will say so within 14 days and explain why.

## PGP / signed mail

phasm does not currently publish a PGP key. If you need
encrypted email, request a key in a first message to
`security@phasm.app` and we will provision one.

## Hall of fame

Security researchers who report valid findings will be credited
in the release notes for the fix (and in a `SECURITY.md` hall
of fame, when one exists), unless they prefer to remain
anonymous.
