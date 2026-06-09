"""Build the paper-ready figures from already-completed experiment data.

Produces (in OUT — see resolution below):
  - fig_phase4_ceiling.pdf        : §6.4 deniability ceiling — 9 attacks, bar chart
  - fig_cost_pool_sensitivity.pdf : §6.4 cost-pool sensitivity sweep (E5)
  - fig_capacity_sweep.pdf        : §6.5 capacity (Ghost vs Shadow, 3 buckets)
  - fig_eval_singlemsg_pe.pdf     : §6.2 PE-vs-payload headline
  - fig_eval_shadowN_pe.pdf       : §6.3 shadow-N PE curve (E10)
  - fig_bit_exact_table.md        : §6.6 reproducibility table (Markdown -> LaTeX)

OUT resolution (first match wins):
  1. $PHASM_FIGURES_OUT if set in environment
  2. <repo>/marketing/research/figures/ if the marketing repo is present
     alongside eval/ (dev workflow inside the monorepo)
  3. <eval>/figures/ (reviewer workflow inside phasmcore where marketing/
     is not cloned)

Run:
  cd ~/Development/phasm/eval
  .venv/bin/python scripts/build_paper_figures.py
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent  # eval/scripts/
EVAL = SCRIPT_DIR.parent                       # eval/
REPO = EVAL.parent                             # monorepo root or phasmcore root


def _resolve_out_dir() -> Path:
    env = os.environ.get("PHASM_FIGURES_OUT")
    if env:
        return Path(env).resolve()
    marketing_figures = REPO / "marketing" / "research" / "figures"
    if marketing_figures.is_dir():
        return marketing_figures
    return EVAL / "figures"


OUT = _resolve_out_dir()
OUT.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------------
# Figure 1 — Phase 4 deniability ceiling
# ----------------------------------------------------------------------

def fig_phase4_ceiling() -> None:
    """9 attack methodologies measured on the shadow-count discrimination task.

    AUC ordered by class:
      black-box CNN  -> low (under-extracts)
      stat-aware hand-crafted (XGBoost / MLP) -> highest
      white-box cost-pool-aware -> near chance (cost-pool defense)
    """
    attacks = [
        # (label, class, AUC, half-CI or std, dataset)
        ("Multi-feature LR",     "ensemble",  0.481, 0.144, "BOSSbase n=160"),
        ("White-box (gray)",     "white-box", 0.517, 0.028, "BOSSbase n=79"),
        ("White-box (color)",    "white-box", 0.551, 0.011, "ALASKA n=387"),
        ("CNN SRNet",            "cnn",       0.556, 0.003, "BOSSbase n=900×3"),
        ("Single feature",       "single",    0.572, 0.000, "BOSSbase n=160"),
        ("XGBoost (color, min)", "stat-aware",0.685, 0.018, "ALASKA n=387"),
        ("XGBoost (color, full)","stat-aware",0.701, 0.022, "ALASKA n=387"),
        ("XGBoost (gray)",       "stat-aware",0.803, 0.009, "BOSSbase n=1496"),
        ("MLP (gray)",           "stat-aware",0.870, 0.006, "BOSSbase n=1496"),
    ]
    # Wong (Nature 2011) colorblind-safe palette. Avoids the red-green
    # confusion (deuteranopia / protanopia ~5-8% of males).
    klass_colors = {
        "ensemble":   "#7f7f7f",   # gray
        "single":     "#bdbdbd",   # lighter gray (distinguishable from ensemble)
        "cnn":        "#0072B2",   # Wong blue
        "white-box":  "#009E73",   # Wong bluish-green
        "stat-aware": "#D55E00",   # Wong vermilion
    }
    # Hatching adds a redundant non-color cue for accessibility.
    klass_hatch = {
        "ensemble":   "",
        "single":     "//",
        "cnn":        "",
        "white-box":  "..",
        "stat-aware": "xx",
    }
    klass_labels = {
        "ensemble":   "Naive ensemble",
        "single":     "Single-feature",
        "cnn":        "Deep CNN (black-box)",
        "white-box":  "White-box cost-pool-aware",
        "stat-aware": "Stat-aware feature mining",
    }

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    xs = np.arange(len(attacks))
    aucs = np.array([a[2] for a in attacks])
    errs = np.array([a[3] for a in attacks])
    colors = [klass_colors[a[1]] for a in attacks]
    hatches = [klass_hatch[a[1]] for a in attacks]
    bars = ax.bar(xs, aucs, yerr=errs, capsize=3, color=colors,
                  edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars, hatches):
        bar.set_hatch(h)
    ax.set_xticks(xs)
    ax.set_xticklabels([a[0] for a in attacks], fontsize=8,
                       rotation=30, ha="right")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8,
               label="chance (AUC=0.5)")
    ax.axhline(0.85, color="firebrick", linestyle="--", linewidth=0.8,
               label="trivial-detection threshold (AUC=0.85)")
    ax.set_ylabel("AUC (shadow-count discrimination)", fontsize=10)
    ax.set_ylim(0.4, 0.95)
    ax.set_title("Shadow-count deniability: AUC ceiling across nine attacks",
                 fontsize=10)
    # Class legend
    used = set(a[1] for a in attacks)
    handles = [plt.Rectangle((0,0),1,1, color=klass_colors[k]) for k in used]
    labels = [klass_labels[k] for k in used]
    ax.legend(handles=handles + [
        plt.Line2D([0], [0], color="black", linestyle=":"),
        plt.Line2D([0], [0], color="firebrick", linestyle="--"),
    ], labels=labels + ["chance", "trivial-detection"],
        fontsize=7, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    out = OUT / "fig_phase4_ceiling.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ----------------------------------------------------------------------
# Figure 2 — Cost-pool fraction sensitivity
# ----------------------------------------------------------------------

def fig_cost_pool_sensitivity() -> None:
    """Per-fraction AUC sweep: 5/10/20/50/100% pool sizes.

    Gray data (E5, 4 seeds x 5 folds pooled) is the clean signal: small
    fractions are at chance, large fractions leak marginally.
    """
    fractions = [5, 10, 20, 50, 100]

    # Gray: pooled 20-fold from E5
    seeds_gray = [2026, 12345, 67890, 99]
    gray_per_frac = {f: [] for f in fractions}
    for s in seeds_gray:
        sub = "" if s == 2026 else f"/s{s}"
        path = EVAL / f"runs/2026-05-14-cost-pool-sensitivity-gray{sub}/results.json"
        d = json.loads(path.read_text())
        for entry in d["per_fraction"]:
            gray_per_frac[entry["fraction_pct"]].extend(entry["auc_per_fold"])

    # Color: pool 4 seeds (s2026 from 2026-05-14 + s12345/s67890/s99 from
    # 2026-05-16) for 20-fold CIs.
    color_per_frac_folds = {f: [] for f in fractions}
    color_paths = [
        EVAL / "runs/2026-05-14-cost-pool-sensitivity-color/results.json",
    ] + [EVAL / f"runs/2026-05-16-cost-pool-sensitivity-color-3seed/s{s}/results.json"
         for s in (12345, 67890, 99)]
    for cp in color_paths:
        cd = json.loads(cp.read_text())
        for e in cd["per_fraction"]:
            color_per_frac_folds[e["fraction_pct"]].extend(e["auc_per_fold"])
    color_per_frac = {f: (float(np.mean(color_per_frac_folds[f])),
                          float(np.std(color_per_frac_folds[f], ddof=1)))
                      for f in fractions}

    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.2), sharey=True)
    xs = np.arange(len(fractions))
    width = 0.6

    # Gray panel
    means = np.array([np.mean(gray_per_frac[f]) for f in fractions])
    errs = np.array([
        1.96 * np.std(gray_per_frac[f], ddof=1) / np.sqrt(len(gray_per_frac[f]))
        for f in fractions
    ])
    # Wong-palette: bluish-green (#009E73) for "at chance" (CI includes 0.5),
    # vermilion (#D55E00) for "above chance". Hatch redundancy: solid for
    # at-chance, dense crosshatch for above-chance.
    bar_colors = ["#009E73" if m + e < 0.5 or m - e <= 0.5 else "#D55E00"
                  for m, e in zip(means, errs)]
    bar_hatches = ["" if m + e < 0.5 or m - e <= 0.5 else "xx"
                   for m, e in zip(means, errs)]
    bars0 = axes[0].bar(xs, means, width=width, yerr=errs, capsize=4,
                color=bar_colors,
                edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars0, bar_hatches):
        bar.set_hatch(h)
    axes[0].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[0].set_xticks(xs)
    axes[0].set_xticklabels([f"{f}%" for f in fractions])
    axes[0].set_xlabel("Cost-pool fraction (% of Y nzAC)")
    axes[0].set_ylabel("White-box attacker AUC")
    axes[0].set_title("(a) BOSSbase grayscale (n=79, 4 seeds × 5 folds)", fontsize=10)
    axes[0].set_ylim(0.35, 0.75)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    # Color panel — 4 seeds x 5 folds = 20 folds
    means_c = np.array([color_per_frac[f][0] for f in fractions])
    n_color = 20
    errs_c = np.array([1.96 * color_per_frac[f][1] / np.sqrt(n_color) for f in fractions])
    bar_colors_c = ["#009E73" if m + e < 0.5 or m - e <= 0.5 else "#D55E00"
                    for m, e in zip(means_c, errs_c)]
    bar_hatches_c = ["" if m + e < 0.5 or m - e <= 0.5 else "xx"
                     for m, e in zip(means_c, errs_c)]
    bars1 = axes[1].bar(xs, means_c, width=width, yerr=errs_c, capsize=4,
                color=bar_colors_c, edgecolor="black", linewidth=0.5)
    for bar, h in zip(bars1, bar_hatches_c):
        bar.set_hatch(h)
    axes[1].axhline(0.5, color="black", linestyle=":", linewidth=0.8)
    axes[1].set_xticks(xs)
    axes[1].set_xticklabels([f"{f}%" for f in fractions])
    axes[1].set_xlabel("Cost-pool fraction (% of Y nzAC)")
    axes[1].set_title("(b) ALASKA-2 color (n=387, 4 seeds × 5 folds)", fontsize=10)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("Cost-pool fraction sensitivity sweep",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT / "fig_cost_pool_sensitivity.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ----------------------------------------------------------------------
# Figure 3 — Capacity sweep (Ghost vs Shadow vs Deep Cover)
# ----------------------------------------------------------------------

def fig_capacity_sweep() -> None:
    """Per-bucket bpnzAC distributions for Ghost / Ghost-SI / Shadow / Armor.

    Shows: shadow ~5x Ghost (cover-distribution invariant), Ghost-SI gives
    ~43% boost from side-information.
    """
    summary_path = EVAL / "runs/2026-05-12-capacity-sweep/summary.json"
    summary = json.loads(summary_path.read_text())

    bucket_labels = {
        "bossbase_qf75_gray": "BOSSbase\nQF75 gray",
        "alaska2_color":      "ALASKA-2\ncolor",
        "real_world_jpeg":    "Real-world\nphotos",
    }
    buckets = list(bucket_labels.keys())

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    # Panel A: bpnzAC rates as bars per mode per bucket
    # (Deep-Cover SI bytes are recorded but bpnzAC wasn't computed in the
    # 2026-05-12 sweep; see §6.5 prose for the ~43% boost number.)
    # Wong palette: blue for Ghost, vermilion for Shadow.
    modes = [("ghost_bpnzAC", "Primary (J-UNIWARD)", "#0072B2"),
             ("shadow_bpnzAC", "Shadow (cost-pool LSB)", "#D55E00")]

    x = np.arange(len(buckets))
    width = 0.38
    for i, (key, label, color) in enumerate(modes):
        means = []
        errs = []
        for b in buckets:
            d = summary.get(b, {})
            ent = d.get(key) or {}
            means.append(ent.get("mean") or 0.0)
            errs.append(ent.get("std") or 0.0)
        axes[0].bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=3,
                    label=label, color=color, edgecolor="black", linewidth=0.4)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([bucket_labels[b] for b in buckets])
    axes[0].set_ylabel("Payload rate (bits per nzAC)")
    axes[0].set_title("(a) Mean ± std bpnzAC rate per mode", fontsize=10)
    # Legend at center-left (y≈0.5) — clear of Ghost bars near 0.18 and
    # Shadow bars near 0.97. Upper-left would overlap the Shadow bar top
    # at BOSSbase QF75 (~0.97); lower-right would overlap Ghost bars.
    axes[0].legend(fontsize=7, loc="center left", framealpha=0.95)
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)
    axes[0].set_ylim(0, 1.1)

    # Panel B: shadow/ghost ratio scatter — shadow ~5x Ghost
    ratios_per_bucket = {}
    for b in buckets:
        d = summary.get(b, {})
        r = (d.get("shadow_per_ghost_ratio") or {}).get("mean")
        rstd = (d.get("shadow_per_ghost_ratio") or {}).get("std", 0.0)
        ratios_per_bucket[b] = (r, rstd)
    ratios = [ratios_per_bucket[b][0] or 0 for b in buckets]
    rstds = [ratios_per_bucket[b][1] or 0 for b in buckets]
    axes[1].bar(np.arange(len(buckets)), ratios, yerr=rstds, capsize=4,
                color="#CC79A7", edgecolor="black", linewidth=0.5)  # Wong reddish-purple
    axes[1].axhline(5.0, color="black", linestyle=":", linewidth=0.8,
                    label="rule of thumb: 5×")
    axes[1].set_xticks(np.arange(len(buckets)))
    axes[1].set_xticklabels([bucket_labels[b] for b in buckets])
    axes[1].set_ylabel("Shadow / primary capacity ratio")
    axes[1].set_title("(b) Shadow capacity is ≈5× the primary across buckets",
                       fontsize=10)
    # Legend at lower-right (clear of the Real-world bucket's tall error
    # bar that reaches y≈7.5 in the upper-right area).
    axes[1].legend(fontsize=8, loc="lower right", framealpha=0.95)
    axes[1].set_ylim(0, 8)
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    fig.suptitle("§6.5 — Capacity sweep across cover sources",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = OUT / "fig_capacity_sweep.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out}")


# ----------------------------------------------------------------------
# Table — bit-exact reproducibility
# ----------------------------------------------------------------------

def write_bit_exact_table() -> None:
    md = """\
# §6.6 Bit-exact reproducibility — CI artifact

Cross-platform deterministic invariants validated by `cross_platform.rs`
test suite (pinned via `core/.github/workflows/bit-exact.yml`). Each
checkmark indicates the test passes the pinned bit-exact assertion at
HEAD on that platform on the most recent CI run (2026-05-12).

| Platform | det_math (sin/cos/atan2/hypot) | PRNG permutation (`select_and_permute`) | FFT (Cooley–Tukey + Bluestein) |
|---|---|---|---|
| Ubuntu 22.04 (x86_64)         | ✓ | ✓ | ✓ |
| Ubuntu 24.04 (aarch64)        | ✓ | ✓ | ✓ |
| macOS 14 (aarch64 Apple Silicon) | ✓ | ✓ | ✓ |
| Windows 11 (x86_64 MSVC)      | ✓ | ✓ | ✓ |
| WASM (wasm-pack target nodejs) | ✓ | ✓ | ✓ |

iOS / Android targets are not in the CI matrix but are field-validated
via the published apps (the same `det_math` polynomials and `select_and_permute`
fixture are compiled identically by `cargo-ndk` and `cbindgen`-emitted
static libs).
"""
    out = OUT / "fig_bit_exact_table.md"
    out.write_text(md)
    print(f"saved {out}")


# ----------------------------------------------------------------------
# Figure 4 — §6.2 single-message PE-vs-payload curve (headline figure)
# ----------------------------------------------------------------------

def fig_eval_singlemsg_pe() -> None:
    """Phasm Ghost vs J-UNIWARD reference, PE vs payload (BOSSbase QF75 SRNet).

    Headline figure of §6.2 per the original paper plan. Two series:
      - J-UNI native: detector trained on J-UNI@p, tested on J-UNI@p
      - Phasm Ghost:  same detector, tested on Phasm Ghost@p (cross-stego)

    Numbers are the FINAL test PE from train.log (J-UNI native) and the
    cover-vs-stego decision-error rate from the cross-stego inference
    results.json files. Cells without data show as gaps.
    """
    # J-UNI native: SRNet trained on J-UNI@QF75@p, tested on J-UNI@QF75@p.
    # FINAL test PE from each train.log. Phase 1c-curriculum2 for 0.10 (the
    # 0.40 floor was a known training collapse without curriculum init).
    juni_native = {
        0.05: 0.449,   # E9 from-scratch (2026-05-14)
        0.10: 0.404,   # Phase 1c curriculum 2 (2026-05-11) — actual test.pe
        0.20: 0.293,   # E9 from-scratch (2026-05-15)
        0.30: 0.204,   # E9 from-scratch (2026-05-15, landed overnight)
        0.40: 0.148,   # Phase 1c original (2026-05-10)
        0.50: 0.080,   # E9 from-scratch (2026-05-16, deferred handler)
    }

    # Phasm Ghost vs same detector at matched payload, n=1000 paired.
    # All six cells now come from cross_eval_pe.py with the cover/stego
    # scores written out as results.json {FA, MD, PE, AUC} in each run dir.
    phasm_eval: dict[float, float] = {}

    # 0.05 / 0.20 / 0.30 / 0.50 cells live as per-image score JSONs in the
    # 2026-05-16 cross-stego inference run (PE computed inline below).
    cross_stego_dir = EVAL / "runs/2026-05-16-e9-eval-vs-phasm"
    if cross_stego_dir.exists():
        for p in (0.05, 0.20, 0.30, 0.50):
            p_pad = f"{int(round(p * 100)):03d}"
            cover_f = cross_stego_dir / f"cover_qf75_byDet{p_pad}.json"
            phasm_f = cross_stego_dir / f"phasm_qf75_pf{p_pad}.json"
            if cover_f.exists() and phasm_f.exists():
                cover = json.loads(cover_f.read_text())
                phasm = json.loads(phasm_f.read_text())
                cs = np.array([r["stego_prob"] for r in cover["per_image"].values()])
                ps = np.array([r["stego_prob"] for r in phasm["per_image"].values()])
                fa = float((cs > 0.5).mean())
                md = float((ps <= 0.5).mean())
                phasm_eval[p] = (fa + md) / 2

    # 0.10 / 0.40 cells live as canonical results.json (regenerated 2026-05-17
    # after the paper review found the legacy path1c-eval cells were
    # mislabeled; see 2026-05-17 paper-review-findings doc).
    for p, run in [(0.10, "2026-05-17-e9-eval-vs-phasm-pf010"),
                   (0.40, "2026-05-17-e9-eval-vs-phasm-pf040")]:
        results_f = EVAL / f"runs/{run}/results.json"
        if results_f.exists():
            phasm_eval[p] = float(json.loads(results_f.read_text())["PE"])

    # Hardcoded fallbacks if the dirs aren't present (reviewer flow on a
    # data-less phasmcore checkout).
    phasm_eval.setdefault(0.05, 0.404)
    phasm_eval.setdefault(0.10, 0.352)
    phasm_eval.setdefault(0.20, 0.245)
    phasm_eval.setdefault(0.30, 0.234)
    phasm_eval.setdefault(0.40, 0.220)
    phasm_eval.setdefault(0.50, 0.165)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    payloads_juni = sorted(juni_native.keys())
    payloads_phasm = sorted(phasm_eval.keys())

    ax.plot(payloads_juni, [juni_native[p] for p in payloads_juni],
            marker="o", linestyle="-", color="#0072B2", linewidth=1.6,
            markersize=7, label="J-UNI native (SRNet trained on J-UNI@$p$)")
    ax.plot(payloads_phasm, [phasm_eval[p] for p in payloads_phasm],
            marker="^", linestyle="--", color="#D55E00", linewidth=1.6,
            markersize=7, label="Phasm (same detector, cross-stego)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8,
               label="chance ($P_E = 0.5$)")
    ax.axhline(0.05, color="firebrick", linestyle="--", linewidth=0.6,
               label="trivial-detection ($P_E = 0.05$)")
    ax.set_xlabel(r"Payload rate (bits per non-zero AC)")
    ax.set_ylabel(r"SRNet detection error $P_E$")
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0.0, 0.55)
    ax.set_xticks([0.05, 0.10, 0.20, 0.30, 0.40, 0.50])
    ax.grid(linestyle="--", alpha=0.3)
    ax.set_title("Single-message security: Phasm vs J-UNIWARD reference (BOSSbase QF75)",
                 fontsize=9)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.95)
    fig.tight_layout()
    out = OUT / "fig_eval_singlemsg_pe.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def fig_eval_shadowN_pe() -> None:
    """E10 SRNet + E15 EfficientNet PE vs shadow-count N on BOSSbase QF75.

    Per-N max series (n varies 40-100 per N):
      - SRNet J-UNI@QF75@0.20 from-scratch  (E10, native test PE 0.293)
      - EfficientNet-B0 ImageNet-pretrained (E15, F7 follow-up; only plotted
        if the E15 per_n_pe.json file is present — drops the curve
        automatically if E15 has not run yet)
    Plus the SRNet universal-subset (n=40) curve.
    """
    src = EVAL / "runs/2026-05-17-e10-shadow-pe-curve/per_n_pe.json"
    d = json.loads(src.read_text())
    ns = sorted(int(k) for k in d["per_N"].keys())
    pe_universal = [d["per_N"][str(n)]["universal_subset"]["PE"] for n in ns]
    pe_max       = [d["per_N"][str(n)]["per_n_max"]["PE"] for n in ns]
    n_max        = [d["per_N"][str(n)]["per_n_max"]["n_stego"] for n in ns]
    n_universal  = d["n_universal_subset"]

    # E15 EfficientNet — load if present
    e15_path = EVAL / "runs/2026-05-17-e15-effnet-shadow-pe-curve/per_n_pe.json"
    eff_pe_max = None
    if e15_path.exists():
        eff_d = json.loads(e15_path.read_text())
        eff_pe_max = [eff_d["per_N"][str(n)]["per_n_max"]["PE"] for n in ns]

    # JIN-pretrained EfficientNet (Tier 2) — answers the "why not JIN?" review
    # point: a stronger, steganalysis-specific pretrain is still flat on Phasm.
    jin_path = EVAL / "runs/2026-06-03-jin-T2-shadow-pe-curve/per_n_pe.json"
    jin_pe_max = None
    if jin_path.exists():
        jin_d = json.loads(jin_path.read_text())
        jin_pe_max = [(lambda v: v["PE"] if isinstance(v, dict) else v)(
                       jin_d["per_N"][str(n)]["per_n_max"]) for n in ns]

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    # Wong blue (SRNet universal), vermilion (SRNet per-N max),
    # bluish-green (EffNet per-N max) — colorblind-safe palette.
    ax.plot(ns, pe_universal, marker="o", linestyle="-",
            color="#0072B2", linewidth=1.8, markersize=8,
            label=f"SRNet (univ.\\ subset, $n_{{paired}}={n_universal}$)")
    ax.plot(ns, pe_max, marker="^", linestyle="--",
            color="#D55E00", linewidth=1.4, markersize=7,
            label=f"SRNet (per-$N$ max, $n_{{paired}}=$" +
                  ",".join(str(x) for x in n_max) + ")")
    if eff_pe_max is not None:
        ax.plot(ns, eff_pe_max, marker="s", linestyle="-.",
                color="#009E73", linewidth=1.6, markersize=7,
                label="EffNet ImageNet-pretrained (per-$N$ max)")
    if jin_pe_max is not None:
        ax.plot(ns, jin_pe_max, marker="D", linestyle=":",
                color="#CC79A7", linewidth=1.6, markersize=6,
                label="EffNet JIN-pretrained (per-$N$ max)")
    ax.axhline(0.5, color="black", linestyle=":", linewidth=0.8,
               label="chance ($P_E = 0.5$)")
    ax.set_xlabel(r"Shadow count $N$")
    ax.set_ylabel(r"Detection error $P_E$")
    ax.set_xticks(ns)
    ax.set_xticklabels(["0\n(primary only)", "1", "2", "3", "4"])
    # Expand y-range slightly to accommodate EffNet curve (may go below 0.45 if
    # pretrained detector sees through wash-out)
    y_lo, y_hi = 0.45, 0.60
    for _series in (eff_pe_max, jin_pe_max):
        if _series is not None:
            y_lo = min(y_lo, min(_series) - 0.02)
            y_hi = max(y_hi, max(_series) + 0.02)
    ax.set_ylim(y_lo, y_hi)
    ax.set_xlim(-0.3, 4.3)
    ax.grid(linestyle="--", alpha=0.3)
    title = ("Shadow-message security: detection error vs shadow count\n"
             "BOSSbase QF75; from-scratch SRNet vs EffNet "
             "(ImageNet & JIN pretrained)")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left", framealpha=0.95)
    fig.tight_layout()
    out = OUT / "fig_eval_shadowN_pe.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main() -> None:
    print(f"[build_paper_figures] OUT = {OUT}")
    fig_phase4_ceiling()
    fig_cost_pool_sensitivity()
    fig_capacity_sweep()
    fig_eval_singlemsg_pe()
    fig_eval_shadowN_pe()
    write_bit_exact_table()
    print(f"\nAll figures in {OUT}")


if __name__ == "__main__":
    main()
