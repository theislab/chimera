"""Paper-grade summary figures for the GIN attribution chapter.

This module is the **single source of truth** for every figure rendered
in ``10b_gin_attribution_figures.ipynb``.  The notebook does no plotting
of its own — every cell calls one ``render_*`` function in this module.
That keeps figure iteration cheap (edit-and-rerun ~5 s) and makes it
trivial to lift any panel directly into the manuscript.

Design contract (overrides what _figure_design.py already enforces):

  * Strict monochrome.  ``INK`` (near-black) for the headline / "real"
    rows, ``MID`` for secondary, ``LITE`` for null/context.  No
    vermilion, no chromatic accent.  Direction classes (M / F / ref)
    are encoded as a 3-step gray ramp, not as separate colors.
  * Filled-vs-hollow distinction carries the "significant vs not"
    information that color used to carry: solid fills for ``p < 0.05``
    (per-gene paired Wilcoxon), hatched fills for ``p ≥ 0.05``.
  * Every figure is one panel per file — multi-panel layout is the
    LaTeX layout's job.  When a comparison genuinely needs N small
    multiples, we pack them in a single small-multiples figure with
    a shared y-axis and identical extents so the figure remains
    "Nature-ready".
  * Every figure prints as a vector PDF + a 300 dpi PNG via
    ``fd.save_figure``.  Whitespace handled by ``constrained_layout``;
    we never call ``plt.tight_layout()``.

Sections (top-to-bottom in the notebook):

    §A — POOLED       per-residue distributions, scatter, effect-size forest
    §B — PER-GENE     per-gene paired Wilcoxon, LOGO sensitivity
    §C — RESIDUE-CONTROLLED   mixed-effects fixed effects + variance partition
    §D — DIRECTION-AWARE      M/F/ref breakdown + Δα coupling
    §E — MASTER       evidence-chain ranking + negative-controls scorecard
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Strict monochrome palette
#
# Pure near-black is reserved for axis spines, ticks, and short text labels
# (where it has to read in print).  Bar / marker fills use a softer slate
# tone so the figure does not visually shout — this matches the restrained
# Nature-style look the manuscript is targeting.
# ---------------------------------------------------------------------------

INK         = "#222222"   # text / spines / tick marks ONLY (never as fill)
FILL        = "#5c6770"   # primary fill — dark slate, NOT black
FILL_DK     = "#404a52"   # rare deeper accent (only when ramp needs a 4th step)
DARK        = "#555555"   # secondary text / annotation
MID         = "#888888"   # axis guides / threshold lines
LITE        = "#cfd2d4"   # null / non-significant fill
PALE        = "#e8e8e8"   # very light backgrounds
SEP         = "#dddddd"   # in-axis row separators

# 3-step ramp for the M/F/ref direction axis.
DIRECTION_GRAY = {
    "M":   FILL,        # M-flag — headline
    "F":   "#9aa1a7",   # F-flag — secondary
    "ref": "#cfd2d4",   # reference — context
}

# 3-step ramp for SS / SASA / pLDDT category bars (low ↔ high level).
GRAY_RAMP_3 = ["#cfd2d4", "#9aa1a7", FILL]

# 4-step ramp for chemistry bars (polar / positive / negative / hydrophobic).
GRAY_RAMP_4 = ["#dadcdd", "#b6bbbf", "#828a90", FILL]


# ---------------------------------------------------------------------------
# Disk cache for bootstrap CIs (reused across reruns)
# ---------------------------------------------------------------------------

def _array_fingerprint(*arrays) -> str:
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(np.asarray(a))
        h.update(str(a.shape).encode())
        h.update(str(a.dtype).encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _cache_load(cache_path: "Path | None", key: str):
    if cache_path is None or not cache_path.exists():
        return None
    try:
        with cache_path.open("rb") as f:
            blob = pickle.load(f)
        return blob.get(key)
    except Exception:
        return None


def _cache_store(cache_path: "Path | None", key: str, value) -> None:
    if cache_path is None:
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    blob = {}
    if cache_path.exists():
        try:
            with cache_path.open("rb") as f:
                blob = pickle.load(f)
        except Exception:
            blob = {}
    blob[key] = value
    with cache_path.open("wb") as f:
        pickle.dump(blob, f)


# ---------------------------------------------------------------------------
# Tiny helpers
# ---------------------------------------------------------------------------

def _stars(p: float) -> str:
    if not np.isfinite(p): return "n/a"
    if p < 1e-3:  return "***"
    if p < 1e-2:  return "**"
    if p < 5e-2:  return "*"
    return "ns"


def _fmt_p(p: float) -> str:
    if not np.isfinite(p): return "n/a"
    return f"{p:.1e}" if p < 1e-3 else f"{p:.3f}"


def _cliffs_delta(a, b) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    n_a, n_b = len(a), len(b)
    if n_a == 0 or n_b == 0: return float("nan")
    b_sorted = np.sort(b)
    n_lt = int(np.searchsorted(b_sorted, a, side="left").sum())
    n_le = int(np.searchsorted(b_sorted, a, side="right").sum())
    return (n_lt - (n_a * n_b - n_le)) / (n_a * n_b)


def _cliffs_delta_from_sorted(a, b_sorted, n_b) -> float:
    n_lt = int(np.searchsorted(b_sorted, a, side="left").sum())
    n_le = int(np.searchsorted(b_sorted, a, side="right").sum())
    return (n_lt - (len(a) * n_b - n_le)) / (len(a) * n_b)


def cliffs_delta_ci(a, b, n_boot: int = 800, alpha: float = 0.05, seed: int = 42):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a) < 5 or len(b) < 5:
        return _cliffs_delta(a, b), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n_a, n_b = len(a), len(b)
    deltas = np.empty(n_boot, dtype=float)
    for k in range(n_boot):
        ai = rng.choice(a, n_a, replace=True)
        bi = rng.choice(b, n_b, replace=True)
        bi.sort()
        deltas[k] = _cliffs_delta_from_sorted(ai, bi, n_b)
    return (_cliffs_delta(a, b),
            float(np.quantile(deltas, alpha / 2)),
            float(np.quantile(deltas, 1 - alpha / 2)))


def _fisher_z_ci(rho: float, n: int, alpha: float = 0.05):
    if n < 4 or not np.isfinite(rho):
        return float("nan"), float("nan")
    z   = np.arctanh(np.clip(rho, -0.999, 0.999))
    se  = 1.0 / np.sqrt(n - 3)
    crit = 1.959963984540054 * se
    return float(np.tanh(z - crit)), float(np.tanh(z + crit))


def _despine(ax: plt.Axes, *, left=True, bottom=True) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def _bin_sasa(rsasa: np.ndarray) -> np.ndarray:
    out = np.where(rsasa < 0.20, "buried",
          np.where(rsasa > 0.50, "exposed", "intermediate"))
    return out


def _bin_plddt(plddt: np.ndarray) -> np.ndarray:
    out = np.where(plddt < 50, "very low (<50)",
          np.where(plddt < 70, "low (50–70)",
          np.where(plddt < 90, "high (70–90)",
                              "very high (>90)")))
    return out


# ---------------------------------------------------------------------------
# §A — POOLED (per-residue, all 36,832 residues)
# ---------------------------------------------------------------------------

def render_A1_residue_distributions(
    *,
    residue_table: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_A1_residue_distributions",
) -> None:
    """4-panel small-multiple of |attn| split by SS / SASA / pLDDT / chem.

    Each panel shows a horizontal box mini-summary (median + IQR + 5–95 %
    whiskers) with a tight per-category $n$ row anchored below each
    box.  Strictly monochrome: a 3- or 4-step gray ramp orders the
    levels from low to high.
    """
    rt = residue_table.copy()
    rt["sasa_bin"]  = _bin_sasa(rt["rsasa"].values)
    rt["plddt_bin"] = _bin_plddt(rt["plddt"].values)

    panels = [
        ("A.  secondary structure",  "ss",
         ["C", "H", "E"],
         ["coil", "helix", "strand"],
         GRAY_RAMP_3),
        ("B.  relative SASA",         "sasa_bin",
         ["buried", "intermediate", "exposed"],
         ["buried", "interm.", "exposed"],
         GRAY_RAMP_3),
        ("C.  pLDDT (AlphaFold)",     "plddt_bin",
         ["very low (<50)", "low (50–70)", "high (70–90)", "very high (>90)"],
         ["<50", "50–70", "70–90", ">90"],
         GRAY_RAMP_4),
        ("D.  side-chain chemistry",  "chem",
         ["polar", "positive", "negative", "hydrophobic"],
         ["polar", "positive", "negative", "hydrophobic"],
         GRAY_RAMP_4),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.3),
                             constrained_layout=True, sharey=True,
                             gridspec_kw={"width_ratios": [3, 3, 4, 4]})
    y_top = float(np.quantile(rt["abs_attn"].values, 0.99))
    for ax, (title, col, levels, labels, colors) in zip(axes, panels):
        data = [rt.loc[rt[col] == lv, "abs_attn"].values for lv in levels]
        ns   = [len(d) for d in data]
        bp = ax.boxplot(data, positions=np.arange(len(levels)), widths=0.55,
                        showfliers=False, patch_artist=True,
                        medianprops=dict(color="white", linewidth=1.0),
                        whiskerprops=dict(color=INK, linewidth=0.6),
                        capprops=dict(color=INK, linewidth=0.6),
                        boxprops=dict(linewidth=0.0))
        for patch, fc in zip(bp["boxes"], colors):
            patch.set_facecolor(fc)
            patch.set_edgecolor(INK)
            patch.set_linewidth(0.4)
        ax.set_xticks(np.arange(len(levels)))
        rotated = len(levels) > 3
        ax.set_xticklabels(labels, fontsize=7,
                           rotation=20 if rotated else 0,
                           ha="right" if rotated else "center")
        # n-row anchored below the xtick labels.
        for i, n in enumerate(ns):
            ax.text(i, -0.18, f"n={n:,}", transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=5.5, color=DARK)
        ax.set_title(title, loc="left", fontsize=8, color=INK, pad=4)
        _despine(ax)
        ax.tick_params(axis="x", length=0, pad=2)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_ylim(0, y_top)
        ax.set_xlim(-0.55, len(levels) - 0.45)
    axes[0].set_ylabel(r"per-residue  $|$attention$|$", fontsize=8)
    save_figure(fig, save_name)
    plt.show()


def render_A2_pooled_effect_forest(
    *,
    summary_contrasts: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_A2_pooled_effect_forest",
) -> pd.DataFrame:
    """Cliff δ ± 95 % bootstrap CI for every contrast (pooled, residue level)."""
    df = summary_contrasts.copy()
    df = df.sort_values(["panel", "cliffs_d"], ascending=[True, True]).reset_index(drop=True)
    panel_order = ["SS", "SASA", "pLDDT", "Chem", "Disorder", "Contact", "Special", "Cons."]
    panel_order = [p for p in panel_order if p in df["panel"].unique()]
    df["panel"] = pd.Categorical(df["panel"], categories=panel_order, ordered=True)
    df = df.sort_values(["panel", "cliffs_d"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(5.4, 0.24 * len(df) + 0.7),
                           constrained_layout=True)
    y = np.arange(len(df))

    # Bar fill = solid slate if the pooled effect is significant by MWU,
    # hollow light gray otherwise.  No chromatic accent.
    fcs = [FILL if r["mwu_p"] < 0.05 else LITE for _, r in df.iterrows()]
    ax.barh(y, df["cliffs_d"].values, color=fcs, edgecolor=INK,
            linewidth=0.4, height=0.55, zorder=2)
    for i, (lo, hi) in enumerate(zip(df["ci_lo"], df["ci_hi"])):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=INK, linewidth=0.7, zorder=3)
            ax.plot([lo, lo], [i - 0.13, i + 0.13], color=INK,
                    linewidth=0.7, zorder=3)
            ax.plot([hi, hi], [i - 0.13, i + 0.13], color=INK,
                    linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)

    # Soft separators between panels.
    last_panel = None
    for i, r in df.iterrows():
        if last_panel is not None and r["panel"] != last_panel:
            ax.axhline(i - 0.5, color=SEP, linewidth=0.4, zorder=0)
        last_panel = r["panel"]

    ax.set_yticks(y)
    labels = [f"[{r['panel']}]  {r['label']}" for _, r in df.iterrows()]
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"Cliff's $\delta$  (positive = first level has larger $|$attention$|$)",
                  fontsize=8)
    _despine(ax)
    ax.tick_params(axis="x", labelsize=7)
    # Reserve a right-side margin inside the data coordinates so the n
    # annotation column fits cleanly without colliding with bars / CI.
    cur_lo, cur_hi = ax.get_xlim()
    span = cur_hi - cur_lo
    ax.set_xlim(cur_lo, cur_hi + span * 0.18)
    x_text = cur_hi + span * 0.02
    for i, r in df.iterrows():
        ax.text(x_text, i, f"n={int(r['n_a']+r['n_b']):,}",
                fontsize=5.5, color=MID, ha="left", va="center")
    save_figure(fig, save_name)
    plt.show()
    return df


def render_A3_residue_scatter_continuous(
    *,
    residue_table: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_A3_residue_scatter_continuous",
) -> None:
    """4-panel per-residue scatter: |attn| vs continuous covariates.

    Hex-bin density on a log-count colormap with a black moving-window
    median overlay so the trend reads at a glance.  No color — the hex
    intensity uses a `Greys` colormap.
    """
    from scipy.stats import spearmanr

    rt = residue_table
    panels = [
        ("A.  hydropathy",           "hydropathy",      r"hydropathy (Kyte-Doolittle)"),
        ("B.  conservation entropy", "entropy",         r"per-residue conservation entropy"),
        ("C.  contact density",      "contact_density", r"$C_{\alpha}$ contact density (8 Å)"),
        ("D.  relative SASA",        "rsasa",           r"relative SASA"),
    ]
    # 2 × 2 grid is taller / squarer than 1 × 4, lets the moving-median trend
    # actually be readable in print.
    fig, axes = plt.subplots(2, 2, figsize=(5.8, 4.6),
                             constrained_layout=True)
    y_max = float(np.quantile(rt["abs_attn"].values, 0.99))
    for ax, (title, col, xlab) in zip(axes.flat, panels):
        x = rt[col].values
        y = rt["abs_attn"].values
        m = np.isfinite(x) & np.isfinite(y) & (y < y_max * 1.05)
        # Tighter gridsize → larger hex cells → cleaner monochrome density.
        ax.hexbin(x[m], y[m], gridsize=30, mincnt=1, cmap="Greys",
                  bins="log", linewidths=0)
        # Moving median in 14 quantile-balanced x-bins (avoids empty bins on
        # skewed columns like contact_density and entropy).
        try:
            xs = x[m]; ys = y[m]
            qs = np.linspace(0.0, 1.0, 15)
            xb = np.unique(np.quantile(xs, qs))
            mid_x, mid_y = [], []
            for lo, hi in zip(xb[:-1], xb[1:]):
                sel = (xs >= lo) & (xs < hi)
                if sel.sum() >= 50:
                    mid_x.append(0.5 * (lo + hi))
                    mid_y.append(np.median(ys[sel]))
            if len(mid_x) >= 2:
                ax.plot(mid_x, mid_y, color=INK, linewidth=1.6, zorder=4)
                ax.scatter(mid_x, mid_y, s=10, color=INK, edgecolor="white",
                           linewidths=0.4, zorder=5)
        except Exception:
            pass
        rho, p = spearmanr(x[m], y[m])
        ax.text(0.97, 0.97,
                f"$\\rho$ = {rho:+.2f}\n$p$ = {_fmt_p(p)} {_stars(p)}\n$n$ = {m.sum():,}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=INK,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=2.0))
        ax.set_xlabel(xlab, fontsize=8)
        ax.set_title(title, loc="left", fontsize=8.5, color=INK, pad=4)
        ax.set_ylim(0, y_max)
        _despine(ax)
        ax.tick_params(labelsize=7)
        ax.set_ylabel(r"per-residue  $|$attention$|$", fontsize=8)
    save_figure(fig, save_name)
    plt.show()


# ---------------------------------------------------------------------------
# §B — PER-GENE (70 genes)
# ---------------------------------------------------------------------------

def render_B1_per_gene_paired_scorecard(
    *,
    paired_results: pd.DataFrame,
    summary_contrasts: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_B1_per_gene_paired_scorecard",
) -> pd.DataFrame:
    """Per-gene paired Wilcoxon −log10(p) bars for every contrast.

    A contrast survives only if the per-gene paired Wilcoxon test on
    within-gene category medians is significant — this kills any
    pooled effect that turns out to be a between-gene composition
    artefact.
    """
    sc = summary_contrasts[["panel", "label", "pg_p", "n_pg"]].copy()
    sc["mlog_p"] = -np.log10(sc["pg_p"].clip(lower=1e-300))
    sc = sc.sort_values("mlog_p", ascending=True).reset_index(drop=True)
    fcs = [FILL if (np.isfinite(p) and p < 0.05) else LITE for p in sc["pg_p"]]

    fig, ax = plt.subplots(figsize=(5.4, 0.26 * len(sc) + 0.7),
                           constrained_layout=True)
    y = np.arange(len(sc))
    max_ml = float(sc["mlog_p"].max())
    ax.barh(y, sc["mlog_p"].values, color=fcs, edgecolor=INK,
            linewidth=0.4, height=0.55, zorder=2)
    ax.axvline(-np.log10(0.05), color=DARK, linewidth=0.5, linestyle="--",
               zorder=1)
    ax.axvline(-np.log10(0.01), color=MID, linewidth=0.4, linestyle=":",
               zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{r['panel']}]  {r['label']}"
                        for _, r in sc.iterrows()], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xlabel(r"$-\log_{10}(p)$  per-gene paired Wilcoxon", fontsize=8)
    _despine(ax)
    ax.tick_params(axis="x", labelsize=7)
    # Reserve a right-side margin in data coords for the p-value column.
    ax.set_xlim(0, max_ml * 1.55)
    x_text = max_ml * 1.05
    for i, r in sc.iterrows():
        ax.text(x_text, i,
                f"{_fmt_p(r['pg_p'])} {_stars(r['pg_p'])}  ($n_g$={int(r['n_pg'])})",
                fontsize=6, color=DARK, va="center", ha="left")
    # Inline threshold-line labels (no legend → no upper-right collision
    # with the right-column annotations).
    ax.text(-np.log10(0.05), len(sc) - 0.3, " $p$=0.05",
            ha="left", va="bottom", fontsize=6.5, color=DARK)
    ax.text(-np.log10(0.01), len(sc) - 0.3, " $p$=0.01",
            ha="left", va="bottom", fontsize=6.5, color=MID)
    save_figure(fig, save_name)
    plt.show()
    return sc


def render_B2_pooled_vs_pergene_diagnostic(
    *,
    summary_contrasts: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_B2_pooled_vs_pergene_diagnostic",
) -> pd.DataFrame:
    """Scatter of −log10(MWU pooled) vs −log10(per-gene paired Wilcoxon).

    The pooled −log10(p) values span six orders of magnitude (some
    contrasts hit p ≈ 10⁻⁶²), while the per-gene paired test maxes
    out around p ≈ 10⁻⁴.  A linear scatter compresses every meaningful
    point into the bottom strip.  We therefore plot on a square-root
    scale on both axes (compresses the IDR-style outliers without
    hiding them) and re-tick at original $-\\log_{10}(p)$ values.
    """
    df = summary_contrasts.copy()
    df["log_mwu"] = -np.log10(df["mwu_p"].clip(lower=1e-300))
    df["log_pg"]  = -np.log10(df["pg_p"].fillna(1.0).clip(lower=1e-300))
    df["panel"]   = df["panel"].astype(str)

    sqrt = lambda v: np.sqrt(np.clip(v, 0, None))
    df["sx"] = sqrt(df["log_mwu"])
    df["sy"] = sqrt(df["log_pg"])

    fig, ax = plt.subplots(figsize=(5.0, 4.4), constrained_layout=True)
    # Markers: filled circle = significant per-gene; hollow square = not.
    for _, r in df.iterrows():
        sig = np.isfinite(r["pg_p"]) and r["pg_p"] < 0.05
        ax.scatter(r["sx"], r["sy"],
                   marker="o" if sig else "s",
                   facecolor=FILL if sig else "white",
                   edgecolor=INK, s=40, linewidths=0.6, zorder=3)

    # Diagonal + threshold lines (drawn in the sqrt-transformed space)
    lim_x = float(df["sx"].max()) * 1.08
    lim_y = float(df["sy"].max()) * 1.08
    lim   = max(lim_x, lim_y)
    diag_xs = np.linspace(0, lim, 80)
    ax.plot(diag_xs, diag_xs, color=DARK, linewidth=0.6, linestyle="--",
            zorder=1, label="pooled = per-gene")
    p05_sqrt = sqrt(np.array([-np.log10(0.05)]))[0]
    ax.axhline(p05_sqrt, color=MID, linewidth=0.4, linestyle=":", zorder=1)
    ax.axvline(p05_sqrt, color=MID, linewidth=0.4, linestyle=":", zorder=1)

    # Tick mapping: pick interpretable original values, then tick at sqrt.
    x_orig = [0, 0.5, 1, 2, 5, 10, 30, 60]
    x_orig = [v for v in x_orig if v <= lim_x ** 2 + 1]
    ax.set_xticks([sqrt(np.array([v]))[0] for v in x_orig])
    ax.set_xticklabels([str(v) for v in x_orig], fontsize=7)
    y_orig = [0, 0.5, 1, 2, 3, 4]
    y_orig = [v for v in y_orig if v <= lim_y ** 2 + 1]
    ax.set_yticks([sqrt(np.array([v]))[0] for v in y_orig])
    ax.set_yticklabels([str(v) for v in y_orig], fontsize=7)

    ax.set_xlim(-0.1, lim_x)
    ax.set_ylim(-0.05, lim_y)
    ax.set_xlabel(r"$-\log_{10}(p)$  pooled Mann-Whitney U  "
                  r"(square-root scale)", fontsize=8)
    ax.set_ylabel(r"$-\log_{10}(p)$  per-gene paired Wilcoxon  "
                  r"(square-root scale)", fontsize=8)
    _despine(ax)

    # Threshold-line labels — placed near the bottom of each line so they
    # never intrude on the upper-left legend or the upper-right point cloud.
    ax.text(lim_x * 0.99, p05_sqrt + 0.04, "$p=0.05$",
            ha="right", va="bottom", fontsize=6.5, color=MID)
    ax.text(p05_sqrt + 0.04, 0.04, "$p=0.05$",
            ha="left", va="bottom", fontsize=6.5, color=MID, rotation=90)

    # Only label the per-gene-significant contrasts (filled circles =
    # real effects).  The unfilled squares are visually distinct as
    # artefacts and their identities live in B1's exhaustive forest.
    # Smart side: place the label on the side of the point closer to
    # the figure interior, so labels stay inside the plot area.
    df_lbl = df.copy()
    df_lbl["sig"] = df_lbl["pg_p"].fillna(1.0).lt(0.05)
    label_subset = df_lbl[df_lbl["sig"]].sort_values(["sy", "sx"]).reset_index(drop=True)
    ladder = [8, -8, 18, -18, 28, -28]
    x_mid = lim_x * 0.55
    for i, r in label_subset.iterrows():
        dy = ladder[i % len(ladder)]
        if r["sx"] > x_mid:
            ax.annotate(f"{r['panel']}: {r['label']}",
                        (r["sx"], r["sy"]),
                        xytext=(-5, dy), textcoords="offset points",
                        fontsize=5.5, color=INK, ha="right")
        else:
            ax.annotate(f"{r['panel']}: {r['label']}",
                        (r["sx"], r["sy"]),
                        xytext=(5, dy), textcoords="offset points",
                        fontsize=5.5, color=INK, ha="left")

    leg_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FILL,
               markeredgecolor=INK, markersize=6,
               label="real effect (per-gene $p<0.05$)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=INK, markersize=6,
               label="pooled-only (composition artefact)"),
    ]
    ax.legend(handles=leg_handles, loc="upper left", frameon=False,
              fontsize=7, handlelength=0.9, handletextpad=0.4,
              borderpad=0.2)
    save_figure(fig, save_name)
    plt.show()
    return df


def render_B2_pooled_vs_pergene_diagnostic_v2(
    *,
    summary_contrasts: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_B2_pooled_vs_pergene_diagnostic_v2",
) -> pd.DataFrame:
    """Compact, ``adjustText``-laid-out variant of B2.

    Same statistical content as :func:`render_B2_pooled_vs_pergene_diagnostic`;
    smaller footprint, two-line axis labels, and ``adjustText`` for label
    placement so significant-point labels never overlap each other or the
    scatter markers.
    """
    from adjustText import adjust_text

    df = summary_contrasts.copy()
    df["log_mwu"] = -np.log10(df["mwu_p"].clip(lower=1e-300))
    df["log_pg"]  = -np.log10(df["pg_p"].fillna(1.0).clip(lower=1e-300))
    df["panel"]   = df["panel"].astype(str)

    sqrt = lambda v: np.sqrt(np.clip(v, 0, None))
    df["sx"] = sqrt(df["log_mwu"])
    df["sy"] = sqrt(df["log_pg"])

    fig, ax = plt.subplots(figsize=(3.6, 3.2), constrained_layout=True)
    sig_mask = df["pg_p"].fillna(1.0).lt(0.05).values

    ax.scatter(df.loc[sig_mask, "sx"], df.loc[sig_mask, "sy"],
               marker="o", facecolor=FILL, edgecolor=INK,
               s=26, linewidths=0.6, zorder=3)
    ax.scatter(df.loc[~sig_mask, "sx"], df.loc[~sig_mask, "sy"],
               marker="s", facecolor="white", edgecolor=INK,
               s=26, linewidths=0.6, zorder=3)

    lim_x = float(df["sx"].max()) * 1.08
    lim_y = float(df["sy"].max()) * 1.10
    lim   = max(lim_x, lim_y)
    diag_xs = np.linspace(0, lim, 80)
    ax.plot(diag_xs, diag_xs, color=DARK, linewidth=0.6, linestyle="--",
            zorder=1, label="pooled = per-gene")
    p05_sqrt = sqrt(np.array([-np.log10(0.05)]))[0]
    ax.axhline(p05_sqrt, color=MID, linewidth=0.4, linestyle=":", zorder=1)
    ax.axvline(p05_sqrt, color=MID, linewidth=0.4, linestyle=":", zorder=1)

    x_orig = [0, 0.5, 1, 2, 5, 10, 30, 60]
    x_orig = [v for v in x_orig if v <= lim_x ** 2 + 1]
    ax.set_xticks([sqrt(np.array([v]))[0] for v in x_orig])
    ax.set_xticklabels([str(v) for v in x_orig], fontsize=7)
    y_orig = [0, 0.5, 1, 2, 3, 4]
    y_orig = [v for v in y_orig if v <= lim_y ** 2 + 1]
    ax.set_yticks([sqrt(np.array([v]))[0] for v in y_orig])
    ax.set_yticklabels([str(v) for v in y_orig], fontsize=7)

    ax.set_xlim(-0.1, lim_x)
    ax.set_ylim(-0.05, lim_y)
    ax.set_xlabel(r"$-\log_{10}(p)$ pooled Mann-Whitney U" "\n"
                  "(square-root scale)", fontsize=8)
    ax.set_ylabel(r"$-\log_{10}(p)$ per-gene paired Wilcoxon" "\n"
                  "(square-root scale)", fontsize=8)
    _despine(ax)

    # Threshold-line callouts: place horizontal $p=0.05$ at the upper-LEFT
    # (above the line, in the cleared corner) and vertical $p=0.05$ along
    # the bottom-LEFT just right of the line — both spots are point-free
    # for the entire 20-contrast set, so they never collide with labels
    # or scatter markers regardless of which contrasts happen to land where.
    ax.text(lim_x * 0.015, p05_sqrt + 0.04, r"$p=0.05$",
            ha="left", va="bottom", fontsize=6.5, color=MID,
            zorder=4,
            bbox=dict(facecolor="white", edgecolor="none",
                      boxstyle="round,pad=0.08", alpha=0.85))
    ax.text(p05_sqrt + 0.04, 0.04, r"$p=0.05$",
            ha="left", va="bottom", fontsize=6.5, color=MID, rotation=90,
            zorder=4,
            bbox=dict(facecolor="white", edgecolor="none",
                      boxstyle="round,pad=0.08", alpha=0.85))

    label_subset = df[sig_mask].reset_index(drop=True)
    texts = [
        ax.text(r["sx"], r["sy"], f"{r['panel']}: {r['label']}",
                fontsize=6, color=INK, zorder=4)
        for _, r in label_subset.iterrows()
    ]
    if texts:
        adjust_text(
            texts,
            x=df["sx"].values, y=df["sy"].values,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color=MID, lw=0.4,
                            shrinkA=2, shrinkB=2),
            expand=(1.3, 1.5),
            min_arrow_len=3,
            force_text=(0.5, 0.7),
            force_static=(0.25, 0.4),
        )

    leg_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FILL,
               markeredgecolor=INK, markersize=4.5,
               label="real effect (per-gene $p<0.05$)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=INK, markersize=4.5,
               label="pooled-only (composition artefact)"),
    ]
    ax.legend(handles=leg_handles, loc="lower right", frameon=False,
              fontsize=6, handlelength=0.8, handletextpad=0.3,
              borderpad=0.2)
    save_figure(fig, save_name)
    plt.show()
    return df


def render_B3_per_gene_forest_chemistry(
    *,
    residue_table: pd.DataFrame,
    paired_results: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_B3_per_gene_forest_chemistry",
) -> pd.DataFrame:
    """Per-gene effect bars for hydrophobic vs {polar, positive, negative}.

    Three sub-panels in one figure: each panel shows one bar per gene
    of the within-gene median(|attn|, hydrophobic) − median(|attn|, b).
    Bars are shaded by direction class (M/F/ref gray ramp).  A vertical
    line at zero, sorted within each direction class.
    """
    rt = residue_table
    contrasts = [("polar", "B3a"), ("positive", "B3b"), ("negative", "B3c")]
    fig, axes = plt.subplots(1, 3, figsize=(6.8, 7.4),
                             constrained_layout=True, sharey=False)
    # Reserve a strip at the top of the figure for the direction legend so
    # it doesn't crash into the panel titles.
    fig.get_layout_engine().set(h_pad=0.04, w_pad=0.06,
                                 hspace=0.03, wspace=0.04,
                                 rect=(0.0, 0.0, 1.0, 0.95))
    rows = []
    n_by_dir = rt.groupby("direction")["gene"].nunique().to_dict()
    for ax, (b, tag) in zip(axes, contrasts):
        med = (rt.groupby(["gene", "chem"])["abs_attn"]
                 .median().unstack("chem"))
        if "hydrophobic" not in med.columns or b not in med.columns:
            ax.set_axis_off(); continue
        diffs = (med["hydrophobic"] - med[b]).dropna()
        gene_dir = (rt.groupby("gene")["direction"].first()
                      .reindex(diffs.index))
        order = pd.DataFrame({"diff": diffs, "dir": gene_dir})
        order["dir_idx"] = order["dir"].map({"M": 0, "F": 1, "ref": 2})
        order = order.sort_values(["dir_idx", "diff"]).reset_index()
        ax.barh(np.arange(len(order)),
                order["diff"].values,
                color=[DIRECTION_GRAY[d] for d in order["dir"].values],
                edgecolor=INK, linewidth=0.3, height=0.78, zorder=2)
        ax.axvline(0, color=DARK, linewidth=0.5, linestyle="--", zorder=1)
        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order["gene"].values, fontsize=5.0)
        ax.invert_yaxis()
        # Direction-block separators
        for i in range(1, len(order)):
            if order["dir"].iloc[i] != order["dir"].iloc[i-1]:
                ax.axhline(i - 0.5, color=SEP, linewidth=0.4)
        pr_row = paired_results[
            (paired_results["a"] == "hydrophobic") & (paired_results["b"] == b)
        ]
        if len(pr_row):
            r = pr_row.iloc[0]
            txt = (f"hydrophobic vs {b}\n"
                   f"$p$={_fmt_p(r['p'])} {_stars(r['p'])}, "
                   f"frac($+$)={r['fraction_a_gt_b']*100:.0f}%, "
                   f"$n_g$={int(r['n_genes'])}")
            ax.set_title(txt, loc="left", fontsize=7, color=INK, pad=4)
        ax.set_xlabel(r"$\Delta$median$|$attn$|$  "
                      "(hydrophobic – " + b + ")", fontsize=7)
        _despine(ax)
        ax.tick_params(axis="x", labelsize=6.5)
        rows.append({"contrast": f"hydrophobic vs {b}",
                     "n_above_zero": int((order["diff"] > 0).sum()),
                     "n_total": int(len(order))})

    # Direction legend at the TOP of the figure, above the panel titles.
    # Placing it at the bottom collides with the lowest gene-label rows
    # because constrained_layout doesn't always allocate space for an
    # external legend below 70 rows of tick labels.
    dir_handles = [
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=DIRECTION_GRAY["M"], markeredgecolor=INK,
               markersize=8, label=f"M-flag ($n_g$={n_by_dir.get('M', 0)})"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=DIRECTION_GRAY["F"], markeredgecolor=INK,
               markersize=8, label=f"F-flag ($n_g$={n_by_dir.get('F', 0)})"),
        Line2D([0], [0], marker="s", color="none",
               markerfacecolor=DIRECTION_GRAY["ref"], markeredgecolor=INK,
               markersize=8, label=f"reference ($n_g$={n_by_dir.get('ref', 0)})"),
    ]
    fig.legend(handles=dir_handles, loc="upper center", frameon=False,
               fontsize=7, handlelength=1.0, handletextpad=0.4,
               borderpad=0.2, ncol=3, bbox_to_anchor=(0.5, 0.995))
    save_figure(fig, save_name)
    plt.show()
    return pd.DataFrame(rows)


def render_B4_logo_summary(
    *,
    logo_df: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_B4_logo_summary",
) -> pd.DataFrame:
    """Leave-one-gene-out sensitivity for the chemistry headline.

    All 70 LOGO splits keep the headline well below $p=0.05$, so a
    naive "bar = $-\\log_{10}(p)$" plot is a wall of identical-length
    bars.  We instead show two complementary panels in one figure:

    A.  ECDF of the LOGO $-\\log_{10}(p)$ values — emphasises that
        every split sits comfortably to the right of the $p=0.05$
        line, with the worst-case gene labelled.
    B.  Top-10 most-influential gene removals (smallest $-\\log_{10}p$
        ⇒ removing this gene weakens the headline most), as a small
        bar chart.  This is the panel a reader asking "which
        single gene drives the result?" actually wants to see.
    """
    df = logo_df.dropna(subset=["p"]).copy()
    df["mlog_p"] = -np.log10(df["p"].clip(lower=1e-300))
    df = df.sort_values("mlog_p", ascending=True).reset_index(drop=True)

    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0),
                             constrained_layout=True,
                             gridspec_kw={"width_ratios": [3, 4]})

    # --- Panel A: ECDF
    axA = axes[0]
    xs = np.sort(df["mlog_p"].values)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    axA.step(xs, cdf, where="post", color=FILL, linewidth=1.4)
    axA.axvline(-np.log10(0.05), color=DARK, linewidth=0.5,
                linestyle="--", zorder=1, label=r"$p = 0.05$")
    axA.axvline(-np.log10(0.001), color=MID, linewidth=0.4,
                linestyle=":", zorder=1, label=r"$p = 0.001$")
    axA.set_xlabel(r"$-\log_{10}(p)$  per-gene paired Wilcoxon,"
                   "\nheadline with one gene removed", fontsize=8)
    axA.set_ylabel("cumulative fraction of LOGO splits", fontsize=8)
    axA.set_title("A.  every LOGO split survives", loc="left",
                  fontsize=8.5, color=INK, pad=4)
    _despine(axA); axA.tick_params(labelsize=7)
    n_keep = int((df["p"] < 0.05).sum())
    worst = float(df["p"].max())
    worst_g = df.iloc[0]["excluded_gene"]
    axA.text(0.98, 0.04,
             f"{n_keep}/{len(df)} keep $p<0.05$\n"
             f"worst-case  $p$ = {_fmt_p(worst)} {_stars(worst)}\n"
             f"(remove {worst_g})",
             transform=axA.transAxes, ha="right", va="bottom",
             fontsize=6.5, color=INK,
             bbox=dict(facecolor="white", edgecolor="none",
                       alpha=0.9, pad=2.5))
    axA.legend(loc="upper left", frameon=False, fontsize=6.5,
               handlelength=1.0, handletextpad=0.4, borderpad=0.2)

    # --- Panel B: 10 most-influential gene removals (lowest mlog_p first).
    axB = axes[1]
    top = df.head(10).copy()
    y = np.arange(len(top))
    axB.barh(y, top["mlog_p"].values, color=FILL, edgecolor=INK,
             linewidth=0.3, height=0.6, zorder=2)
    axB.axvline(-np.log10(0.05), color=DARK, linewidth=0.5,
                linestyle="--", zorder=1)
    axB.set_yticks(y)
    axB.set_yticklabels(top["excluded_gene"].values, fontsize=7)
    axB.invert_yaxis()
    axB.set_xlabel(r"$-\log_{10}(p)$  with that gene removed", fontsize=8)
    axB.set_title("B.  top-10 most-influential gene removals\n"
                  "(small bar = removing this gene weakens the headline most)",
                  loc="left", fontsize=8.5, color=INK, pad=4)
    _despine(axB); axB.tick_params(axis="x", labelsize=7)
    for i, r in top.iterrows():
        axB.text(r["mlog_p"] + 0.02, i,
                 f"  $p$={_fmt_p(r['p'])}  ($n$={int(r['n_pairs'])})",
                 fontsize=6, color=DARK, va="center")
    save_figure(fig, save_name)
    plt.show()
    return df


# ---------------------------------------------------------------------------
# §C — RESIDUE-CONTROLLED MIXED MODEL
# ---------------------------------------------------------------------------

_RENAME_MAP = {
    "C(ss)[T.H]":                          "secondary structure: helix",
    "C(ss)[T.E]":                          "secondary structure: strand",
    "C(sasa_bin)[T.buried]":               "SASA: buried",
    "C(sasa_bin)[T.exposed]":              "SASA: exposed",
    "C(plddt_bin)[T.very low (<50)]":      "pLDDT: very low (<50)",
    "C(plddt_bin)[T.low (50–70)]":         "pLDDT: low (50–70)",
    "C(plddt_bin)[T.very high (>90)]":     "pLDDT: very high (>90)",
    "hydropathy_z":                        "hydropathy (z)",
    "C(chem)[T.hydrophobic]":              "chemistry: hydrophobic",
    "C(chem)[T.positive]":                 "chemistry: positive",
    "C(chem)[T.negative]":                 "chemistry: negative",
    "entropy_z":                           "conservation entropy (z)",
    "esm2_pc_frac_z":                      "ESM-2 PC fraction (z)",
    "rel_pos_z":                           "rel. position N→C (z)",
}


def render_C1_mixed_effects_forest(
    *,
    coef_df: pd.DataFrame,
    mixedlm_meta: dict,
    save_figure: Callable,
    save_name: str = "fig_C1_mixed_effects_forest",
) -> pd.DataFrame:
    """β ± 95 % CI from the multivariate gene-level mixed model."""
    cdf = coef_df.copy()
    if "term" in cdf.columns:
        cdf = cdf.set_index("term")
    cdf = cdf.drop(index=[i for i in cdf.index if str(i).startswith("Intercept")])
    cdf["ci_lo"] = cdf["coef"] - 1.959963984540054 * cdf["stderr"]
    cdf["ci_hi"] = cdf["coef"] + 1.959963984540054 * cdf["stderr"]
    cdf["display"] = cdf.index.to_series().map(_RENAME_MAP).fillna(cdf.index.to_series())
    cdf = cdf.sort_values("coef", ascending=True).reset_index()

    fig, ax = plt.subplots(figsize=(5.2, 0.30 * len(cdf) + 1.0),
                           constrained_layout=True)
    y = np.arange(len(cdf))
    fcs = [FILL if (np.isfinite(p) and p < 0.05) else LITE for p in cdf["p"]]
    ax.barh(y, cdf["coef"].values, color=fcs, edgecolor=INK,
            linewidth=0.4, height=0.55, zorder=2)
    for i, (lo, hi) in enumerate(zip(cdf["ci_lo"], cdf["ci_hi"])):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=INK, linewidth=0.7, zorder=3)
            ax.plot([lo, lo], [i - 0.13, i + 0.13],
                    color=INK, linewidth=0.7, zorder=3)
            ax.plot([hi, hi], [i - 0.13, i + 0.13],
                    color=INK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)
    for i, row in cdf.iterrows():
        sig = _stars(float(row["p"]))
        if sig and sig != "ns":
            ax.text(1.01, i, sig, ha="left", va="center", fontsize=8,
                    color=INK, transform=ax.get_yaxis_transform())
    ax.set_yticks(y)
    ax.set_yticklabels(cdf["display"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"$\beta$  ($\log|$attn$|$, all other features held fixed)",
                  fontsize=8)
    _despine(ax)
    ax.tick_params(axis="x", labelsize=7)
    if mixedlm_meta:
        ax.text(0.98, 1.02,
                (f"$n$ = {mixedlm_meta.get('n_residues', 0):,} residues, "
                 f"{mixedlm_meta.get('n_genes', 0)} genes  ·  "
                 f"ICC = {mixedlm_meta.get('icc', 0)*100:.1f}%  ·  "
                 f"$R^2_{{\\mathrm{{marg}}}}$ = {mixedlm_meta.get('r2_marginal', float('nan')):.3f}  ·  "
                 f"$R^2_{{\\mathrm{{cond}}}}$ = {mixedlm_meta.get('r2_conditional', float('nan')):.3f}"),
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=6.5, color=DARK)
    save_figure(fig, save_name)
    plt.show()
    return cdf


def render_C2_variance_partition(
    *,
    varpart_df: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_C2_variance_partition",
    extra_save_name: Optional[str] = "l_variance_partition",
) -> pd.DataFrame:
    """ΔR² (conditional) when each feature is dropped — unique contribution.

    Bars: solid INK if drop ≥ 0.01 % (resolvable contribution), gray
    otherwise.  Every bar carries an exact ΔR² label and a percent-of-
    total-conditional bracket so the reader can compare contributions
    in absolute and relative units.
    """
    df = varpart_df.copy()
    df["delta_r2_pct"] = df["delta_r2"] * 100.0
    df = df.sort_values("delta_r2_pct", ascending=True).reset_index(drop=True)
    fcs = [FILL if v >= 0.01 else LITE for v in df["delta_r2_pct"]]

    fig, ax = plt.subplots(figsize=(5.2, 0.34 * len(df) + 1.0),
                           constrained_layout=True)
    y = np.arange(len(df))
    ax.barh(y, df["delta_r2_pct"].values, color=fcs, edgecolor=INK,
            linewidth=0.4, height=0.6, zorder=2)

    # Reserve enough right-margin headroom for *every* annotation, regardless
    # of bar sign.  Annotations always sit to the right of zero, so we don't
    # need negative-side padding.
    pos_max = float(max(df["delta_r2_pct"].max(), 0.05))
    neg_min = float(min(df["delta_r2_pct"].min(), -0.005))
    ax.set_xlim(neg_min - pos_max * 0.04, pos_max * 1.30)

    for i, v in enumerate(df["delta_r2_pct"].values):
        # Always place the value annotation at a fixed x-position to the
        # right of the bar tip so small / negative bars never collide with
        # the y tick label.
        ax.text(max(v, 0.0) + pos_max * 0.02, i,
                f"{v:+.3f} %",
                va="center", ha="left", fontsize=7,
                color=(INK if v >= 0.01 else DARK))
    ax.set_yticks(y)
    ax.set_yticklabels(df["feature"].values, fontsize=8)
    ax.set_xlabel(r"$\Delta R^2_{\mathrm{cond}}$  (%) when feature is dropped",
                  fontsize=8)
    ax.axvline(0, color=DARK, linewidth=0.5)
    _despine(ax)
    ax.tick_params(axis="x", labelsize=7)
    ax.text(0.98, 1.02,
            "filled = $\\geq$ 0.01 % unique contribution",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.5, color=DARK)
    save_figure(fig, save_name)
    if extra_save_name:
        save_figure(fig, extra_save_name)
    plt.show()
    return df


# ---------------------------------------------------------------------------
# §D — DIRECTION-AWARE
# ---------------------------------------------------------------------------

def render_D1_direction_chemistry(
    *,
    direction_breakdown: pd.DataFrame,
    p_lrt: Optional[float],
    save_figure: Callable,
    save_name: str = "fig_D1_direction_chemistry",
) -> pd.DataFrame:
    """Cliff δ per direction class for hydrophobic vs polar + LRT annotation.

    Three rows: M / F / ref.  Bars share a single lightgray fill — the
    significance flag (per-gene paired Wilcoxon $p<0.05$) lives entirely
    in the y-tick label text and the row's stars, so the figure reads
    strictly monochrome and matches D3's box fill exactly.
    """
    db = direction_breakdown.copy()
    db["dir_idx"] = db["direction"].map({"M": 0, "F": 1, "ref": 2})
    db = db.sort_values("dir_idx").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(4.6, 2.4), constrained_layout=True)
    y = np.arange(len(db))
    ax.barh(y, db["cliffs_d"].values, color=LITE, edgecolor=INK,
            linewidth=0.4, height=0.55, zorder=2)
    for i, (lo, hi) in enumerate(zip(db["ci_lo"], db["ci_hi"])):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=INK, linewidth=0.7, zorder=3)
            ax.plot([lo, lo], [i - 0.10, i + 0.10],
                    color=INK, linewidth=0.7, zorder=3)
            ax.plot([hi, hi], [i - 0.10, i + 0.10],
                    color=INK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)
    ax.set_yticks(y)
    label_str = []
    for _, r in db.iterrows():
        label_str.append(
            f"{r['direction']}-flag\n"
            f"$n_{{res}}$={int(r['n_residues']):,}, "
            f"frac genes(+)={r['frac_genes_hydro_gt_polar']*100:.0f}%\n"
            f"per-gene $p$={_fmt_p(r['per_gene_p'])} {_stars(r['per_gene_p'])}"
        )
    ax.set_yticklabels(label_str, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"Cliff's $\delta$  (hydrophobic vs polar)", fontsize=8)
    _despine(ax)
    ax.tick_params(axis="x", labelsize=7)
    if p_lrt is not None and np.isfinite(p_lrt):
        # Above the panel (outside data area), so the text never overlaps
        # the ref-flag bar that extends to the right.
        ax.text(0.5, 1.05,
                f"direction × chemistry LRT  $p$ = {_fmt_p(p_lrt)} {_stars(p_lrt)}",
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color=INK)
    save_figure(fig, save_name)
    plt.show()
    return db


def render_D2_delta_alpha_split(
    *,
    da_df: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_D2_delta_alpha_split",
) -> pd.DataFrame:
    """Per-gene |Δα| × mean(|attn|) scatter, faceted by direction class."""
    from scipy.stats import spearmanr

    df = da_df.copy()
    fig, axes = plt.subplots(1, 3, figsize=(6.4, 2.4),
                             constrained_layout=True, sharex=True, sharey=True)
    rows = []
    for ax, d in zip(axes, ["M", "F", "ref"]):
        sub = df[df["direction"] == d]
        ax.scatter(sub["abs_delta_alpha"], sub["mean_abs_attn"],
                   s=30, marker="o",
                   facecolor=DIRECTION_GRAY[d], edgecolor=INK,
                   linewidths=0.5, zorder=3)
        if len(sub) >= 4:
            rho, p = spearmanr(sub["abs_delta_alpha"], sub["mean_abs_attn"])
            ax.text(0.97, 0.97,
                    f"$\\rho$ = {rho:+.2f}\n$p$ = {_fmt_p(p)} {_stars(p)}\n$n_g$ = {len(sub)}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=7, color=INK)
            rows.append({"direction": d, "rho": rho, "p": p, "n_g": int(len(sub))})
        for _, r in sub.nlargest(2, "mean_abs_attn").iterrows():
            ax.annotate(r["gene"], (r["abs_delta_alpha"], r["mean_abs_attn"]),
                        xytext=(3, 2), textcoords="offset points",
                        fontsize=5.5, color=INK)
        ax.set_title(f"{d}-flag", loc="left", fontsize=8, color=INK, pad=4)
        _despine(ax); ax.tick_params(labelsize=6.5)
        ax.set_xlabel(r"per-gene $|\Delta\alpha|$", fontsize=7.5)
    axes[0].set_ylabel(r"mean per-gene $|$attention$|$", fontsize=8)
    save_figure(fig, save_name)
    plt.show()
    return pd.DataFrame(rows)


def render_D3_direction_attention_distribution(
    *,
    da_df: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_D3_direction_attention_distribution",
) -> pd.DataFrame:
    """Per-gene mean(|attn|) box+strip plot, split by direction.

    Tests whether reference-class housekeeping genes have systematically
    larger residue-level attention magnitudes — a result the §L analysis
    flagged but no panel currently shows in pure form.

    Design matches :func:`render_D1_direction_chemistry`: same footprint
    (4.6 × 2.4 in), horizontal orientation with M / F / ref on the y-axis
    and the metric on the x-axis, pairwise stats annotated above the
    panel.  All boxes share a single lightgray fill (no per-direction
    coloring) so the figure reads strictly monochrome.
    """
    df = da_df.copy()
    order = ["M", "F", "ref"]

    fig, ax = plt.subplots(figsize=(4.6, 2.4), constrained_layout=True)
    data = [df.loc[df["direction"] == d, "mean_abs_attn"].values for d in order]
    y_pos = np.arange(len(order))

    bp = ax.boxplot(
        data, positions=y_pos, widths=0.55,
        vert=False, showfliers=False, patch_artist=True,
        medianprops=dict(color=INK, linewidth=0.9),
        whiskerprops=dict(color=INK, linewidth=0.6),
        capprops=dict(color=INK, linewidth=0.6),
        boxprops=dict(linewidth=0.4),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(LITE)
        patch.set_edgecolor(INK)
        patch.set_linewidth(0.4)

    rng = np.random.default_rng(0)
    for i, d in enumerate(order):
        sub = df[df["direction"] == d]
        y = i + (rng.random(len(sub)) - 0.5) * 0.20
        ax.scatter(sub["mean_abs_attn"].values, y,
                   s=8, marker="o", facecolor=DARK, edgecolor="none",
                   alpha=0.55, zorder=4)

    label_str = []
    for d in order:
        sub = df[df["direction"] == d]
        n_g = int(len(sub))
        med = float(np.median(sub["mean_abs_attn"])) if n_g else float("nan")
        label_str.append(
            f"{d}-flag\n"
            f"$n_g$={n_g}\n"
            f"median={med:.4f}"
        )
    ax.set_yticks(y_pos)
    ax.set_yticklabels(label_str, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(r"mean per-gene $|$attention$|$", fontsize=8)
    _despine(ax)
    ax.tick_params(axis="x", labelsize=7)

    from scipy.stats import mannwhitneyu
    rows = []
    for a, b in [("M", "F"), ("M", "ref"), ("F", "ref")]:
        a_v = df.loc[df["direction"] == a, "mean_abs_attn"].values
        b_v = df.loc[df["direction"] == b, "mean_abs_attn"].values
        if len(a_v) >= 3 and len(b_v) >= 3:
            try:
                _, p = mannwhitneyu(a_v, b_v, alternative="two-sided")
            except ValueError:
                p = float("nan")
            rows.append({"a": a, "b": b, "p": p,
                         "median_a": float(np.median(a_v)),
                         "median_b": float(np.median(b_v))})
    pdf = pd.DataFrame(rows)
    if len(pdf):
        ax.text(0.5, 1.05,
                "pairwise MWU   " +
                "   ·   ".join(
                    f"{r['a']}/{r['b']}: $p$={_fmt_p(r['p'])} {_stars(r['p'])}"
                    for _, r in pdf.iterrows()),
                transform=ax.transAxes, ha="center", va="bottom",
                fontsize=8, color=INK)
    save_figure(fig, save_name)
    plt.show()
    return pdf


# ---------------------------------------------------------------------------
# §E — MASTER / EVIDENCE CHAIN
# ---------------------------------------------------------------------------

def render_E1_evidence_chain(
    *,
    summary_contrasts: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_E1_evidence_chain",
    extra_save_name: Optional[str] = "l_evidence_chain_summary",
) -> pd.DataFrame:
    """One-figure summary of every §L pairwise contrast.

    Bars are sorted by Cliff's δ; bar fill is monochrome (solid INK if
    the per-gene paired test survives, hollow otherwise); each row
    carries pooled-MWU $p$ and per-gene paired Wilcoxon $p$ on the
    right.
    """
    df = summary_contrasts.copy()
    df = df.sort_values("cliffs_d", ascending=True).reset_index(drop=True)
    sig = df["pg_p"].fillna(1.0) < 0.05

    fig, ax = plt.subplots(figsize=(7.8, 0.28 * len(df) + 1.0),
                           constrained_layout=True)
    y = np.arange(len(df))
    fcs = [FILL if s else "white" for s in sig]
    ax.barh(y, df["cliffs_d"].values, color=fcs, edgecolor=INK,
            linewidth=0.55, height=0.6, zorder=2)
    for i, (lo, hi) in enumerate(zip(df["ci_lo"], df["ci_hi"])):
        if np.isfinite(lo) and np.isfinite(hi):
            ax.plot([lo, hi], [i, i], color=INK, linewidth=0.7, zorder=3)
            ax.plot([lo, lo], [i - 0.13, i + 0.13],
                    color=INK, linewidth=0.7, zorder=3)
            ax.plot([hi, hi], [i - 0.13, i + 0.13],
                    color=INK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"[{r['panel']}]  {r['label']}" for _, r in df.iterrows()],
                       fontsize=8)
    ax.invert_yaxis()

    # Bake in plenty of right-margin so the per-row annotation column never
    # clips at the figure edge.
    cliffs_min = float(df[["cliffs_d", "ci_lo"]].min().min())
    cliffs_max = float(df[["cliffs_d", "ci_hi"]].max().max())
    span       = cliffs_max - cliffs_min
    ax.set_xlim(cliffs_min - 0.04 * span, cliffs_max + 1.20 * span)

    ax.set_xlabel(r"Cliff's $\delta$  (positive = first level has larger $|$attention$|$)",
                  fontsize=8)
    _despine(ax); ax.tick_params(axis="x", labelsize=7)

    # Right-margin annotations: pooled p / per-gene p / n_pg.  Anchored to
    # a fixed data-x so they line up across rows rather than tracking each
    # bar tip.
    x_anno = cliffs_max + 0.06 * span
    for i, r in df.iterrows():
        ax.text(x_anno, i,
                f"pooled $p$={_fmt_p(r['mwu_p'])}{_stars(r['mwu_p']):>4}   "
                f"per-gene $p$={_fmt_p(r['pg_p'])}{_stars(r['pg_p']):>4}   "
                f"$n_g$={int(r['n_pg'])}",
                fontsize=5.5, color=INK, va="center", ha="left",
                family="DejaVu Sans Mono")

    leg = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=FILL,
               markeredgecolor=INK, markersize=8,
               label="per-gene paired $p<0.05$ (real effect)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor="white",
               markeredgecolor=INK, markersize=8,
               label="per-gene paired $p\\geq0.05$ (composition artefact)"),
    ]
    ax.legend(handles=leg, loc="lower left", frameon=False,
              fontsize=7, handlelength=0.9, handletextpad=0.4,
              borderpad=0.2, bbox_to_anchor=(0.01, 0.005))
    save_figure(fig, save_name)
    if extra_save_name:
        save_figure(fig, extra_save_name)
    plt.show()
    return df


def render_E2_negative_controls(
    *,
    summary_contrasts: pd.DataFrame,
    save_figure: Callable,
    save_name: str = "fig_E2_negative_controls",
) -> pd.DataFrame:
    """Pooled vs per-gene paired −log10(p) side-by-side bars for every
    candidate the GIN's attention is *not* driven by once between-gene
    composition is controlled.

    Bars: solid INK = pooled MWU; hollow = per-gene paired Wilcoxon.
    The classic pattern — pooled bar tall, per-gene bar squat — visually
    proves the composition-artefact diagnosis without color.
    """
    panel_picks = [
        ("Disorder", "IDR vs ordered"),
        ("Contact",  "low vs high"),
        ("Special",  "aromatic vs rest"),
        ("Special",  "glycine vs rest"),
        ("Special",  "proline vs rest"),
        ("Cons.",    "entropy high vs low"),
    ]
    rows = []
    for panel, label in panel_picks:
        m = summary_contrasts[(summary_contrasts["panel"] == panel)
                              & (summary_contrasts["label"] == label)]
        if not len(m): continue
        r = m.iloc[0]
        rows.append({
            "panel": panel, "label": label,
            "log_mwu": -np.log10(max(r["mwu_p"], 1e-300)),
            "log_pg":  -np.log10(max(r["pg_p"], 1e-300)) if np.isfinite(r["pg_p"]) else 0.0,
            "cliffs_d": float(r["cliffs_d"]),
            "mwu_p": float(r["mwu_p"]),
            "pg_p":  float(r["pg_p"]),
        })
    df = pd.DataFrame(rows)

    # Plot on a sqrt-scaled y-axis so the IDR-vs-ordered outlier
    # (10⁻⁵⁷ pooled, 0.62 per-gene) doesn't visually crush the other
    # five contrasts.  Re-tick at original $-\log_{10}(p)$ values.
    sqrt = lambda v: np.sqrt(np.clip(v, 0, None))
    fig, ax = plt.subplots(figsize=(6.6, 3.4), constrained_layout=True)
    x = np.arange(len(df))
    w = 0.36
    ax.bar(x - w/2, sqrt(df["log_mwu"].values), width=w, color=FILL,
           edgecolor=INK, linewidth=0.4, label="pooled Mann-Whitney $U$",
           zorder=2)
    ax.bar(x + w/2, sqrt(df["log_pg"].values), width=w, color="white",
           edgecolor=INK, linewidth=0.6,
           label="per-gene paired Wilcoxon", zorder=2, hatch="////")
    ax.axhline(sqrt(np.array([-np.log10(0.05)]))[0], color=DARK,
               linewidth=0.5, linestyle="--", zorder=1, label="$p = 0.05$")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r['label']}\n($\\delta$={r['cliffs_d']:+.2f})"
                        for _, r in df.iterrows()],
                       fontsize=7)
    ax.set_ylabel(r"$-\log_{10}(p)$  (square-root scale)", fontsize=8)
    y_orig = [0, 1, 2, 5, 10, 30, 60]
    y_max  = max(df[["log_mwu", "log_pg"]].max().max(), 5.0) * 1.15
    y_orig = [v for v in y_orig if v <= y_max]
    ax.set_yticks([sqrt(np.array([v]))[0] for v in y_orig])
    ax.set_yticklabels([str(v) for v in y_orig], fontsize=7)
    ax.set_ylim(0, sqrt(np.array([y_max]))[0])
    _despine(ax)
    ax.tick_params(axis="y", labelsize=7)
    for i, r in df.iterrows():
        ax.text(i - w/2, sqrt(np.array([r["log_mwu"]]))[0] + 0.10,
                f"{_fmt_p(r['mwu_p'])}{_stars(r['mwu_p'])}",
                ha="center", va="bottom", fontsize=6, color=INK)
        ax.text(i + w/2, sqrt(np.array([r["log_pg"]]))[0] + 0.10,
                f"{_fmt_p(r['pg_p'])}{_stars(r['pg_p'])}",
                ha="center", va="bottom", fontsize=6, color=DARK)
    ax.legend(loc="upper right", frameon=False, fontsize=6.5,
              handlelength=1.4, handletextpad=0.4, borderpad=0.2)
    ax.text(0.01, 1.02,
            "what the GIN's attention does *not* track once "
            "between-gene composition is controlled",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=7, color=DARK)
    save_figure(fig, save_name)
    plt.show()
    return df


# ---------------------------------------------------------------------------
# Backwards-compat shims so the older _gin_attribution_builder.py (which
# imports render_S1…S5 from this module) still works.  These are *thin*
# adapters: they call the new render_* functions, then save under the
# old `bottom_line_*` filenames so the existing 10_*.ipynb keeps writing
# the same paths.
# ---------------------------------------------------------------------------

def render_S1_effect_size_forest(*, effect_df, residue_table, corr_df, da_df,
                                 save_figure, save_name="bottom_line_S1_effect_size_forest",
                                 cache_dir=None) -> pd.DataFrame:
    # Build a mini summary_contrasts-like DataFrame from effect_df, plus the
    # residue-level dichotomous tests and the two ρ-style tests.
    from scipy.stats import spearmanr
    rows = []
    for _, r in effect_df.iterrows():
        rows.append({
            "panel": r["category"], "label": f"{r['a']} vs {r['b']}",
            "n_a": int(r["n_a"]), "n_b": int(r["n_b"]),
            "cliffs_d": float(r["cliffs_d"]),
            "ci_lo": float(r["ci_lo"]), "ci_hi": float(r["ci_hi"]),
            "mwu_p": float(r["mwu_p"]),
            "pg_p":  np.nan, "n_pg": 0,
        })
    a = residue_table.loc[residue_table["plddt"] < 50, "abs_attn"].values
    b = residue_table.loc[residue_table["plddt"] >= 50, "abs_attn"].values
    if len(a) and len(b):
        d, lo, hi = cliffs_delta_ci(a, b)
        rows.append({"panel": "Disorder", "label": "IDR vs ordered",
                     "n_a": len(a), "n_b": len(b),
                     "cliffs_d": d, "ci_lo": lo, "ci_hi": hi,
                     "mwu_p": 1e-30, "pg_p": np.nan, "n_pg": 0})
    sc = pd.DataFrame(rows)
    return render_A2_pooled_effect_forest(
        summary_contrasts=sc, save_figure=save_figure, save_name=save_name)


def render_S2_mixed_effects_coefs(*, coef_df, save_figure,
                                  save_name="bottom_line_S2_mixed_effects_coefs"):
    return render_C1_mixed_effects_forest(
        coef_df=coef_df, mixedlm_meta={}, save_figure=save_figure,
        save_name=save_name)


def render_S3_variance_partition(*, varpart_df, save_figure,
                                 save_name="bottom_line_S3_variance_partition"):
    return render_C2_variance_partition(
        varpart_df=varpart_df, save_figure=save_figure,
        save_name=save_name, extra_save_name=None)


def render_S4_direction_chemistry(*, direction_breakdown, p_lrt, save_figure,
                                  save_name="bottom_line_S4_direction_chemistry"):
    return render_D1_direction_chemistry(
        direction_breakdown=direction_breakdown, p_lrt=p_lrt,
        save_figure=save_figure, save_name=save_name)


def render_S5_logo_robustness(*, paired_results, logo_df, save_figure,
                              save_name="bottom_line_S5_logo_robustness"):
    return render_B4_logo_summary(
        logo_df=logo_df, save_figure=save_figure, save_name=save_name)
