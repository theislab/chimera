"""The panels of the residue-level attribution audit, redrawn at print size.

    supp_residue_audit/pooled.pdf           the 20 residue contrasts over all residues: Cliff's δ with its 95% interval,
                                            filled where the pooled Mann-Whitney test has p < 0.05; number of residues
    supp_residue_audit/per_gene.pdf         the same contrasts within proteins: -log10 p of the paired Wilcoxon test on
                                            per-gene medians, filled where p < 0.05; p and the number of genes
    supp_residue_audit/variance.pdf         change of the conditional R² of the residue-level mixed model when one
                                            feature is left out
    supp_residue_audit/diagnostic.pdf       every contrast by its pooled p and its per-gene p (square-root axes)
    supp_residue_audit/by_class.pdf         mean |attribution| of each protein, by curated class (box and every protein)
    supp_residue_audit/class_chemistry.pdf  Cliff's δ of hydrophobic against polar residues within each curated class
    supp_residue_audit/controls.pdf         six contrasts: pooled p beside per-gene p (square-root axis)

Inputs: ``notebooks/cache/gin_attribution/L_analysis/`` ``summary_contrasts.pkl`` (one row per contrast),
``variance_partition.pkl`` (one row per feature), ``delta_alpha_attention.pkl`` (one row per protein) and
``direction_breakdown.pkl`` (one row per curated class); nothing is tested, fitted or resampled here. In the first two
panels the rows are the contrasts in one order: by residue feature, then by Cliff's δ. Both have the same page height and
the same row positions, so the second needs no row names; they are composed with ``X(...)`` in one row, and so are the
other two. The panels by curated class have one page width and stand in a column beside the stored notebook panel of the
second float. Colors, names of the residue features and of the curated classes and the forms of p values are those of
``figures/redraw/residue.py``.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.residue_audit [panel ...]
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from figures.redraw import _common as C
from figures.redraw.integration_scores import save_exact
from figures.redraw.residue import (
    CURATED_CLASS,
    DARK,
    FEATURE,
    FILL,
    L_CACHE,
    LITE,
    axes_at,
    contrast_name,
    fmt_p,
    key_row,
    put,
    sheet,
    stars,
    width_of,
)

FIGURE = "supp_residue_audit"
ORDER = ["SS", "SASA", "pLDDT", "Chem", "Disorder", "Contact", "Special", "Cons."]   # residue features, top to bottom
# Geometry (bp at print): the two panels and a gap of 7 bp fill the text width.
W_POOLED, W_PER_GENE = 300.0, 179.0
PITCH, TOP, BOTTOM = 10.5, 14.0, 45.0     # one row; column headings above; ticks, axis label and key below
ROW, SMALL = 6.0, 6.0                     # pt: row names; numbers beside the bars


def contrasts() -> pd.DataFrame:
    src = L_CACHE / "summary_contrasts.pkl"
    C.need(src)
    df = pd.read_pickle(src)
    if len(df) != 20 or set(df["panel"]) != set(ORDER):
        raise SystemExit(f"{src} does not hold the 20 contrasts of the eight residue features")
    df["panel"] = pd.Categorical(df["panel"], categories=ORDER, ordered=True)
    return df.sort_values(["panel", "cliffs_d"]).reset_index(drop=True)


def rows(ax, df: pd.DataFrame) -> None:
    """Row geometry shared by both panels: first contrast at the top, a hairline between residue features."""
    n = len(df)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks([])
    for i in range(1, n):
        if df["panel"].iloc[i] != df["panel"].iloc[i - 1]:
            ax.axhline(i - 0.5, color=C.RULE, linewidth=0.4, zorder=0)


def baseline(i: int) -> float:
    """Baseline of the text of row ``i``, in bp below the top edge of the page."""
    return TOP + (i + 0.5) * PITCH + 0.36 * ROW


def pooled() -> list:
    """(a) Cliff's δ of every contrast over all residues, with its interval and the number of residues."""
    df = contrasts()
    n = len(df)
    fig = sheet(W_POOLED, TOP + n * PITCH + BOTTOM)
    feature = [FEATURE[p] for p in df["panel"]]
    left = 1 + max(width_of(fig, s, ROW) for s in feature) + 7 + max(width_of(fig, contrast_name(s), ROW) for s in df["label"]) + 5
    x_n = W_POOLED - 3.0                                                     # right edge of the column of residue counts
    ax = axes_at(fig, left, TOP, x_n - 31 - left, n * PITCH)
    y = np.arange(n)
    ax.barh(y, df["cliffs_d"], color=[FILL if p < 0.05 else LITE for p in df["mwu_p"]], edgecolor=C.INK, linewidth=0.5, height=0.6, zorder=2)
    for i, (lo, hi) in enumerate(zip(df["ci_lo"], df["ci_hi"], strict=True)):
        ax.plot([lo, hi], [i, i], color=C.INK, linewidth=0.7, zorder=3)
        for v in (lo, hi):
            ax.plot([v, v], [i - 0.13, i + 0.13], color=C.INK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)
    lo, hi = float(df[["cliffs_d", "ci_lo"]].min().min()), float(df[["cliffs_d", "ci_hi"]].max().max())
    ax.set_xlim(lo - 0.04 * (hi - lo), hi + 0.04 * (hi - lo))
    ax.set_xticks([-0.1, 0.0, 0.1, 0.2])
    rows(ax, df)
    ax.set_xlabel("Cliff's δ\n(positive: first level has larger |attribution|)", linespacing=1.15)

    head = TOP - 5.0
    put(fig, 1, head, "residue feature", fontsize=C.SIZE["title"])
    put(fig, left - 5, head, "contrast", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_n, head, "residues", fontsize=C.SIZE["title"], ha="right")
    for i, r in df.iterrows():
        if i == 0 or r["panel"] != df["panel"].iloc[i - 1]:
            put(fig, 1, baseline(i), feature[i], fontsize=ROW)              # once, at the first contrast of the feature
        put(fig, left - 5, baseline(i), contrast_name(r["label"]), fontsize=ROW, ha="right")
        put(fig, x_n, baseline(i), f"{int(r['n_a'] + r['n_b']):,}", fontsize=SMALL, ha="right")
    key_row(fig, 1, TOP + n * PITCH + BOTTOM - 3.5, [("patch", FILL, C.INK, "pooled $p$ < 0.05 (Mann-Whitney, all residues)"), ("patch", LITE, C.INK, "pooled $p$ ≥ 0.05")], size=SMALL)
    print(df[["panel", "label", "n_a", "n_b", "cliffs_d", "ci_lo", "ci_hi", "mwu_p"]].to_string(index=False))
    return [save_exact(fig, FIGURE, "pooled")]


def per_gene() -> list:
    """(b) The same contrasts within proteins, on the rows of (a): -log10 p of the paired test, p and the number of genes."""
    df = contrasts()
    n = len(df)
    fig = sheet(W_PER_GENE, TOP + n * PITCH + BOTTOM)
    x_p, x_g = W_PER_GENE - 31.0, W_PER_GENE - 3.0                          # right edges of the columns of p and of genes
    left = 6.0
    ax = axes_at(fig, left, TOP, x_p - 32 - left, n * PITCH)
    score = -np.log10(df["pg_p"].clip(lower=1e-300))
    ax.barh(np.arange(n), score, color=[FILL if p < 0.05 else LITE for p in df["pg_p"]], edgecolor=C.INK, linewidth=0.5, height=0.6, zorder=2)
    for p, style in ((0.05, "--"), (0.01, ":")):
        ax.axvline(-np.log10(p), color=DARK, linewidth=0.55, linestyle=style, zorder=1)
    ax.set_xlim(0, float(score.max()) * 1.04)
    ax.set_xticks([0, 1, 2, 3])
    rows(ax, df)
    ax.set_xlabel("−log₁₀ $p$\n(per-gene paired Wilcoxon)", linespacing=1.15)

    head = TOP - 5.0
    over = ax.get_xaxis_transform()                                          # x in data, y in axes heights: the two thresholds named above the axes
    ax.text(-np.log10(0.05) - 0.06, 1.0 + 3.0 / (n * PITCH), "$p$ = 0.05", ha="right", va="bottom", fontsize=SMALL, color=DARK, transform=over)
    ax.text(-np.log10(0.01) + 0.06, 1.0 + 3.0 / (n * PITCH), "0.01", ha="left", va="bottom", fontsize=SMALL, color=DARK, transform=over)
    put(fig, x_p, head, "$p$", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_g, head, "genes", fontsize=C.SIZE["title"], ha="right")
    for i, r in df.iterrows():
        put(fig, x_p, baseline(i), fmt_p(r["pg_p"]), fontsize=SMALL, ha="right")
        put(fig, x_p + 2.5, baseline(i), stars(r["pg_p"]), fontsize=SMALL, color=C.INK if stars(r["pg_p"]) != "ns" else C.MUTED)
        put(fig, x_g, baseline(i), f"{int(r['n_pg'])}", fontsize=SMALL, ha="right")
    key_row(fig, left, TOP + n * PITCH + BOTTOM - 3.5, [("patch", FILL, C.INK, "per-gene $p$ < 0.05"), ("patch", LITE, C.INK, "$p$ ≥ 0.05")], size=SMALL)
    print(df[["panel", "label", "pg_p", "n_pg"]].to_string(index=False))
    return [save_exact(fig, FIGURE, "per_gene")]


# ----------------------------------------------------------------------------- variance partition and diagnostic
W_VARIANCE, W_DIAGNOSTIC, H_LOWER = 179.0, 300.0, 190.0
PARTITION = {   # feature as stored -> name on the panel
    "SASA bin": "solvent accessibility", "chemistry": "side-chain chemistry", "pLDDT bin": "pLDDT",
    "rel. position N→C": "relative position N→C", "ESM-2 PC fraction": "ESM-2 component fraction", "hydropathy": "hydropathy",
    "conservation entropy": "conservation entropy", "secondary structure": "secondary structure",
}
RESOLVED = 0.01   # percent: a bar is filled from this unique contribution on
# Where the name of a contrast with per-gene p < 0.05 stands: offset of the left end of the text from its point (pt), and
# the place on the text box where its leader starts (0 to 1 across, 0 to 1 up). The three contrasts whose points lie on
# each other share one block of three lines, anchored at the first of them.
PLACE = {
    ("Chem", "hydrophobic vs positive"): ((6.0, 0.0), (0.0, 0.5)),
    ("SASA", "intermediate vs buried"): ((-14.0, 46.0), (0.1, 0.0)),
    ("Contact", "low vs high"): ((6.0, 13.5), (0.0, 0.3)),
    ("Chem", "hydrophobic vs negative"): ((11.0, 10.0), (0.0, 0.3)),
}
BLOCK = [("Chem", "hydrophobic vs negative"), ("Chem", "hydrophobic vs polar"), ("Contact", "low vs medium")]
TWO_LINES = {("Chem", "hydrophobic vs positive")}   # a name longer than the room at the right of its point: the contrast stands under the feature


def variance() -> list:
    """(c) Unique contribution of each feature to the conditional R² of the mixed model, in percent."""
    src = L_CACHE / "variance_partition.pkl"
    C.need(src)
    df = pd.read_pickle(src)
    df["percent"] = df["delta_r2"] * 100.0
    df = df.sort_values("percent", ascending=False).reset_index(drop=True)
    n = len(df)
    fig = sheet(W_VARIANCE, H_LOWER)
    names = [PARTITION[f] for f in df["feature"]]
    left = 2 + max(width_of(fig, s, ROW) for s in names) + 4
    top, bottom = 8.0, 45.0
    ax = axes_at(fig, left, top, W_VARIANCE - left - 4, H_LOWER - top - bottom)
    y = np.arange(n)
    ax.barh(y, df["percent"], color=[FILL if v >= RESOLVED else LITE for v in df["percent"]], edgecolor=C.INK, linewidth=0.5, height=0.6, zorder=2)
    ax.axvline(0, color=DARK, linewidth=0.6, zorder=1)
    high = float(df["percent"].max())
    ax.set_xlim(min(float(df["percent"].min()), 0.0) - 0.03 * high, high * 1.85)       # room for the value beside the longest bar
    ax.set_xticks([0.0, 0.2, 0.4])
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_yticks(y, names, fontsize=ROW)
    ax.tick_params(axis="y", length=0, pad=3.0)
    ax.spines["left"].set_visible(False)                                               # the line at zero is the base of the bars
    for i, v in enumerate(df["percent"]):
        ax.annotate(f"{v:+.3f} %".replace("-", "−"), xy=(max(v, 0.0), i), xytext=(3.0, 0), textcoords="offset points", ha="left", va="center_baseline",
                    fontsize=SMALL, color=C.INK)
    ax.set_xlabel("change of conditional R² (%)\nwhen the feature is left out", linespacing=1.15, loc="right")
    key_row(fig, 2, H_LOWER - 3.5, [("patch", FILL, C.INK, f"{RESOLVED} % or more"), ("patch", LITE, C.INK, "less")], size=SMALL)
    print(df[["feature", "delta_r2", "percent"]].to_string(index=False))
    return [save_exact(fig, FIGURE, "variance")]


def diagnostic() -> list:
    """(d) Pooled against per-gene evidence of every contrast, both as -log10 p on square-root axes; the contrasts with
    per-gene p < 0.05 are named."""
    from matplotlib.lines import Line2D
    from matplotlib.text import Text

    df = contrasts()
    df["panel"] = df["panel"].astype(str)
    df["x"] = np.sqrt(-np.log10(df["mwu_p"].clip(lower=1e-300)))
    df["y"] = np.sqrt(-np.log10(df["pg_p"].fillna(1.0).clip(lower=1e-300)))
    within = (df["pg_p"].fillna(1.0) < 0.05).to_numpy()
    fig = sheet(W_DIAGNOSTIC, H_LOWER)
    left, top, bottom, right = 42.0, 6.0, 44.0, 24.0
    ax = axes_at(fig, left, top, W_DIAGNOSTIC - left - right, H_LOWER - top - bottom)
    ax.scatter(df.loc[within, "x"], df.loc[within, "y"], marker="o", facecolor=FILL, edgecolor=C.INK, s=22, linewidths=0.6, zorder=3)
    ax.scatter(df.loc[~within, "x"], df.loc[~within, "y"], marker="s", facecolor="white", edgecolor=C.INK, s=20, linewidths=0.6, zorder=3)
    lim_x, lim_y = float(df["x"].max()) * 1.08, float(df["y"].max()) * 1.10
    ax.plot([0, lim_y], [0, lim_y], color=DARK, linewidth=0.6, linestyle="--", zorder=1)            # pooled p = per-gene p
    threshold = float(np.sqrt(-np.log10(0.05)))
    ax.axhline(threshold, color=DARK, linewidth=0.5, linestyle=":", zorder=1)
    ax.axvline(threshold, color=DARK, linewidth=0.5, linestyle=":", zorder=1)
    ax.set_xlim(-0.1, lim_x)
    ax.set_ylim(-0.05, lim_y)
    ax.set_xticks([np.sqrt(v) for v in (0, 1, 2, 5, 10, 30, 60)], ["0", "1", "2", "5", "10", "30", "60"])
    ax.set_yticks([np.sqrt(v) for v in (0, 0.5, 1, 2, 3, 4) if np.sqrt(v) <= lim_y], [s for s, v in (("0", 0), ("0.5", 0.5), ("1", 1), ("2", 2), ("3", 3), ("4", 4)) if np.sqrt(v) <= lim_y])
    ax.set_xlabel("−log₁₀ $p$, pooled Mann-Whitney over all residues (square-root scale)")
    ax.set_ylabel("−log₁₀ $p$, per-gene paired\nWilcoxon (square-root scale)", linespacing=1.15)
    ax.text(-0.06, threshold - 0.03, "$p$ = 0.05", ha="left", va="top", fontsize=SMALL, color=DARK)
    ax.text(threshold + 0.06, -0.02, "$p$ = 0.05", ha="left", va="bottom", fontsize=SMALL, color=DARK, rotation=90)

    def name(key) -> str:
        return f"{FEATURE[key[0]]}:{chr(10) if key in TWO_LINES else ' '}{contrast_name(key[1])}"

    texts = []
    for key, (offset, start) in PLACE.items():
        r = df[(df["panel"] == key[0]) & (df["label"] == key[1])].iloc[0]
        lines = BLOCK if key == BLOCK[0] else [key]
        texts.append(ax.annotate("\n".join(name(k) for k in lines), xy=(r["x"], r["y"]), xytext=offset, textcoords="offset points", ha="left", va="center",
                                 fontsize=SMALL, color=C.INK, linespacing=1.2, zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.6),
                                 arrowprops=dict(arrowstyle="-", color=DARK, linewidth=0.4, shrinkA=0.5, shrinkB=3.0, relpos=start)))
    named = {k for k in PLACE} | set(BLOCK)
    missing = [(p, lab) for p, lab in zip(df.loc[within, "panel"], df.loc[within, "label"], strict=True) if (p, lab) not in named]
    if missing:
        raise SystemExit(f"contrasts with per-gene p < 0.05 without a place for their name (PLACE, BLOCK): {missing}")
    # no name may lie on a point
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    points = ax.transData.transform(df[["x", "y"]].to_numpy())
    for t in texts:
        box = Text.get_window_extent(t, renderer)                              # the text alone, without its leader
        hit = [df["label"].iloc[i] for i, (px, py) in enumerate(points) if box.x0 - 2 <= px <= box.x1 + 2 and box.y0 - 2 <= py <= box.y1 + 2]
        if hit:
            raise SystemExit(f"the name '{t.get_text()}' lies on the point of: {hit}")
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=FILL, markeredgecolor=C.INK, markersize=4.2, label="per-gene $p$ < 0.05"),
               Line2D([0], [0], marker="s", color="none", markerfacecolor="white", markeredgecolor=C.INK, markersize=4.0, label="per-gene $p$ ≥ 0.05"),
               Line2D([0], [0], color=DARK, linewidth=0.6, linestyle="--", label="pooled $p$ = per-gene $p$")]
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(left / W_DIAGNOSTIC, 0.0), ncol=3, frameon=False, fontsize=SMALL, handlelength=1.4,
               handletextpad=0.4, columnspacing=1.0, borderpad=0.2)
    print(df[["panel", "label", "mwu_p", "pg_p", "x", "y"]].to_string(index=False))
    return [save_exact(fig, FIGURE, "diagnostic")]


# ----------------------------------------------------------------------------- curated classes and controls
CLASSES = ["M", "F", "ref"]              # rows of the two panels by curated class, top to bottom
W_CLASS = 225.24                         # bp: with the gap of 7 bp, the width left beside the notebook panel of the float
H_BY_CLASS, H_CHEMISTRY = 118.0, 117.86  # bp: the two panels, their letters and the gap make the height of that panel
X_AXES, X_RIGHT = 84.0, 66.0             # bp: left edge of both axes; width kept at the right for the number columns
W_CONTROLS, H_CONTROLS = 486.0, 158.0
CONTROLS = [("Disorder", "IDR vs ordered"), ("Contact", "low vs high"), ("Special", "aromatic vs rest"),
            ("Special", "glycine vs rest"), ("Special", "proline vs rest"), ("Cons.", "entropy high vs low")]


def class_rows(fig, ax, top: float, height: float, second: list) -> None:
    """Row names of a panel by curated class: the class and, under it, a count."""
    ax.set_ylim(len(CLASSES) - 0.5, -0.5)
    ax.set_yticks([])
    pitch = height / len(CLASSES)
    for i, (c, n) in enumerate(zip(CLASSES, second, strict=True)):
        mid = top + (i + 0.5) * pitch
        put(fig, X_AXES - 5, mid - 1.2, CURATED_CLASS[c], fontsize=ROW, ha="right")
        put(fig, X_AXES - 5, mid + 6.4, n, fontsize=SMALL, ha="right", color=DARK)


def by_class() -> list:
    """(e) Mean |attribution| of each protein by curated class: box (median, quartiles, whiskers to 1.5 IQR) and every protein."""
    src = L_CACHE / "delta_alpha_attention.pkl"
    C.need(src)
    df = pd.read_pickle(src)
    data = [df.loc[df["direction"] == c, "mean_abs_attn"].to_numpy() for c in CLASSES]
    if sum(len(d) for d in data) != len(df):
        raise SystemExit(f"{src} holds classes other than {CLASSES}")
    fig = sheet(W_CLASS, H_BY_CLASS)
    top, bottom = 13.0, 26.0
    height = H_BY_CLASS - top - bottom
    ax = axes_at(fig, X_AXES, top, W_CLASS - X_AXES - X_RIGHT, height)
    y = np.arange(len(CLASSES))
    ax.boxplot(data, orientation="horizontal", positions=y, widths=0.55, showfliers=False, patch_artist=True, manage_ticks=False,
               medianprops=dict(color=C.INK, linewidth=0.9), boxprops=dict(facecolor=LITE, edgecolor=C.INK, linewidth=0.5),
               whiskerprops=dict(color=C.INK, linewidth=0.6), capprops=dict(color=C.INK, linewidth=0.6))
    rng = np.random.default_rng(0)
    for i, d in enumerate(data):
        ax.scatter(d, i + (rng.random(len(d)) - 0.5) * 0.20, s=5, marker="o", facecolor=DARK, edgecolor="none", alpha=0.6, zorder=4)
    high = float(max(d.max() for d in data))
    ax.set_xlim(-0.15, high * 1.04)
    ax.set_xticks([0, 2, 4, 6])
    class_rows(fig, ax, top, height, [f"{len(d)} genes" for d in data])
    ax.set_xlabel("per-gene mean |attribution|")
    x_m = W_CLASS - 16.0
    put(fig, x_m, top - 5.0, "median", fontsize=C.SIZE["title"], ha="right")
    pitch = height / len(CLASSES)
    for i, d in enumerate(data):
        put(fig, x_m, top + (i + 0.5) * pitch + 0.36 * ROW, f"{np.median(d):.2f}", fontsize=SMALL, ha="right")
    print(pd.DataFrame({"class": CLASSES, "genes": [len(d) for d in data], "median": [float(np.median(d)) for d in data],
                        "max": [float(d.max()) for d in data]}).to_string(index=False))
    return [save_exact(fig, FIGURE, "by_class")]


def class_chemistry() -> list:
    """(f) Hydrophobic against polar residues within each curated class: Cliff's δ with its interval, the share of
    genes whose own contrast is positive, and the per-gene p."""
    src = L_CACHE / "direction_breakdown.pkl"
    C.need(src)
    df = pd.read_pickle(src).set_index("direction").loc[CLASSES]
    fig = sheet(W_CLASS, H_CHEMISTRY)
    top, bottom = 21.0, 34.0
    height = H_CHEMISTRY - top - bottom
    ax = axes_at(fig, X_AXES, top, W_CLASS - X_AXES - X_RIGHT, height)
    y = np.arange(len(CLASSES))
    ax.barh(y, df["cliffs_d"], color=[FILL if p < 0.05 else LITE for p in df["per_gene_p"]], edgecolor=C.INK, linewidth=0.5, height=0.55, zorder=2)
    for i, (lo, hi) in enumerate(zip(df["ci_lo"], df["ci_hi"], strict=True)):
        ax.plot([lo, hi], [i, i], color=C.INK, linewidth=0.7, zorder=3)
        for v in (lo, hi):
            ax.plot([v, v], [i - 0.11, i + 0.11], color=C.INK, linewidth=0.7, zorder=3)
    ax.axvline(0, color=DARK, linewidth=0.6, linestyle="--", zorder=1)
    lo, hi = float(df["ci_lo"].min()), float(df["ci_hi"].max())
    ax.set_xlim(min(lo, 0.0) - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_xticks([0.0, 0.04, 0.08])
    class_rows(fig, ax, top, height, [f"{int(n):,} residues" for n in df["n_residues"]])
    ax.set_xlabel("Cliff's δ (hydrophobic vs polar)")
    x_p, x_g = W_CLASS - 16.0, W_CLASS - 44.0                               # right edges of the columns of p and of the share of genes
    put(fig, x_g, top - 12.5, "genes", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_g, top - 4.5, "δ > 0", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_p + 13.0, top - 12.5, "per-gene", fontsize=C.SIZE["title"], ha="right")
    put(fig, x_p, top - 4.5, "$p$", fontsize=C.SIZE["title"], ha="right")
    pitch = height / len(CLASSES)
    for i, c in enumerate(CLASSES):
        r = df.loc[c]
        base = top + (i + 0.5) * pitch + 0.36 * ROW
        put(fig, x_g, base, f"{r['frac_genes_hydro_gt_polar'] * 100:.0f}%", fontsize=SMALL, ha="right")
        put(fig, x_p, base, fmt_p(r["per_gene_p"]), fontsize=SMALL, ha="right")
        put(fig, x_p + 2.5, base, stars(r["per_gene_p"]), fontsize=SMALL, color=C.INK if stars(r["per_gene_p"]) != "ns" else C.MUTED)
    key_row(fig, X_AXES, H_CHEMISTRY - 3.5, [("patch", FILL, C.INK, "per-gene $p$ < 0.05"), ("patch", LITE, C.INK, "$p$ ≥ 0.05")], size=SMALL)
    print(df[["n_residues", "cliffs_d", "ci_lo", "ci_hi", "per_gene_p", "frac_genes_hydro_gt_polar"]].to_string())
    return [save_exact(fig, FIGURE, "class_chemistry")]


def controls() -> list:
    """(g) Six contrasts that are often assumed to drive such maps: -log10 p of the pooled test beside that of the
    per-gene test, on a square-root axis; p with its stars above each bar."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    df = contrasts()
    df["panel"] = df["panel"].astype(str)
    picked = [df[(df["panel"] == p) & (df["label"] == lab)] for p, lab in CONTROLS]
    if any(len(r) != 1 for r in picked):
        raise SystemExit("a contrast of CONTROLS is not in summary_contrasts.pkl")
    df = pd.concat(picked).reset_index(drop=True)

    def root(v):
        return np.sqrt(np.clip(-np.log10(np.clip(np.asarray(v, dtype=float), 1e-300, None)), 0, None))

    fig = sheet(W_CONTROLS, H_CONTROLS)
    left, top, bottom = 42.0, 8.0, 36.0
    ax = axes_at(fig, left, top, W_CONTROLS - left - 6.0, H_CONTROLS - top - bottom)
    x, w = np.arange(len(df)), 0.44
    pooled_h, gene_h = root(df["mwu_p"]), root(df["pg_p"].fillna(1.0))
    ax.bar(x - w / 2, pooled_h, width=w, color=FILL, edgecolor=C.INK, linewidth=0.5, zorder=2)
    ax.bar(x + w / 2, gene_h, width=w, color="white", edgecolor=C.INK, linewidth=0.5, hatch="////", zorder=2)
    ax.axhline(float(root(0.05)), color=DARK, linewidth=0.55, linestyle="--", zorder=1)
    high = max(float(-np.log10(df["mwu_p"].min())), 5.0) * 1.15
    ticks = [v for v in (0, 1, 2, 5, 10, 30, 60) if v <= high]
    ax.set_ylim(0, float(np.sqrt(high)))
    ax.set_yticks([np.sqrt(v) for v in ticks], [str(v) for v in ticks])
    ax.set_xlim(-0.55, len(df) - 0.45)
    ax.set_xticks(x, [f"{FEATURE[r['panel']]}:\n{r['label']}\n(δ = {r['cliffs_d']:+.2f})".replace("-", "−") for _, r in df.iterrows()], fontsize=ROW, linespacing=1.2)
    ax.tick_params(axis="x", length=0, pad=3.0)
    ax.set_ylabel("−log₁₀ $p$ (square-root scale)")
    for i, r in df.iterrows():
        for dx, h, p in ((-w / 2, pooled_h[i], r["mwu_p"]), (w / 2, gene_h[i], r["pg_p"])):
            ax.annotate(f"{fmt_p(p)} {stars(p)}", xy=(i + dx, h), xytext=(0, 2.0), textcoords="offset points", ha="center", va="bottom",
                        fontsize=SMALL, color=C.INK, zorder=4, bbox=dict(facecolor="white", edgecolor="none", pad=0.5))
    handles = [Patch(facecolor=FILL, edgecolor=C.INK, linewidth=0.5, label="pooled Mann-Whitney over all residues"),
               Patch(facecolor="white", edgecolor=C.INK, linewidth=0.5, hatch="////", label="per-gene paired Wilcoxon"),
               Line2D([0], [0], color=DARK, linewidth=0.55, linestyle="--", label="$p$ = 0.05")]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=SMALL, handlelength=1.6, handletextpad=0.5,
              labelspacing=0.35, borderpad=0.0, borderaxespad=0.2)
    print(df[["panel", "label", "cliffs_d", "mwu_p", "pg_p"]].to_string(index=False))
    return [save_exact(fig, FIGURE, "controls")]


PANELS = {"pooled": pooled, "per_gene": per_gene, "variance": variance, "diagnostic": diagnostic,
          "by_class": by_class, "class_chemistry": class_chemistry, "controls": controls}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
