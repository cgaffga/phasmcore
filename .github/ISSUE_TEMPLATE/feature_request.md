---
name: Feature request
about: Propose a new feature, improvement, or new media type
title: "[feature] "
labels: enhancement
---

## Use case

<!-- What problem are you trying to solve? Who benefits? Concrete
scenarios are more persuasive than abstract wishes. -->

## Proposed solution

<!-- What you'd like phasm to do. If you have a sketch of the
API or the user-facing flow, include it. -->

## Alternatives considered

<!-- What else could solve this? Why is the proposed solution
better? Often the alternative is "users do this manually
outside phasm" — that's a valid baseline. -->

## Medium

<!-- Tick the medium this targets. -->

- [ ] Image stego
- [ ] Video stego
- [ ] New media type (audio, PDF, document, animated, …)
- [ ] CLI tool
- [ ] Mobile / Web bridge
- [ ] Tooling / docs

## For new media types only

<!--
Please fill in this section if you're proposing a new medium.
The maintainer needs this to scope the work realistically.
-->

- **Cover format**: <!-- e.g. MP3 LAME-encoded, PDF 1.7, ... -->
- **Embedding domain**: <!-- where the message bits live (DCT
  coefficients, MDCT lines, glyph kerning, …) -->
- **Capacity estimate**: <!-- bits per second, bits per page, … -->
- **Threat model**: <!-- which adversaries do you want to defeat?
  recompression, format conversion, recipient-side steganalysis -->
- **Robustness vs stealth tradeoff**: <!-- where on the
  Ghost-vs-Armor axis does this medium sit? -->

## Out of scope

<!-- Are there parts of the proposal you do NOT want included?
Drawing the boundary up front saves review cycles. -->

## Additional context

<!-- Prior art, papers, links to similar projects, etc. -->
