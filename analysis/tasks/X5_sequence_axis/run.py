"""Sequence measures against Chimera's readouts on the same genes, in the style of the prior-vs-alpha panels.

Rows (same genes in each cohort): dN/dS vs identity; alpha vs identity; Delta-alpha vs identity;
Delta-alpha vs dN/dS. Identity is the raw prior; alpha and Delta-alpha are seed consensus values.
Delta-alpha is D (median-centred, against the prior the run anchors on). Blue genes have count
evidence (S_g nonzero), grey genes are masked and their Delta-alpha is not a count call.

Usage: python analysis/tasks/X5_sequence_axis/run.py           # the table of correlations, then the figure
       python -c "from analysis.tasks.X5_sequence_axis import run; run.draw_panels()"          # the figure alone, from the cached tables
The figure is one panel, figures/panels/supp_sequence_axis/sequence_axis.pdf, drawn at its size on the page (house style: figures/style.py).
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
from scipy.stats import spearmanr

from analysis.common import gene_ids
from figures import style

RES = gene_ids.REPO_ROOT / "results" / "analysis"
OUT = RES / "X5"
COHORTS = [(c, style.cohort_name(c)) for c in ("liv_hmouse", "liv_4sp", "test_hmouse")]
LIMS = {"liv_hmouse": (-0.32, 0.22), "liv_4sp": (-0.42, 0.22), "test_hmouse": (-0.05, 0.12)}  # Δα window per cohort; genes beyond it are drawn on the frame
PI_MIN = 0.4  # left end of the identity axis; genes below it are drawn on the frame

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this figure
BLUE, GREY, TREND = style.COUNT_EVIDENCE, style.IMPUTED, style.TREND  # genes with count evidence, genes without it, median trend
INK, MUTED, REF_LINE = style.INK, style.MUTED, "#888888"  # REF_LINE: zero and identity lines
S_TICK = style.SIZE_TICK  # axis labels and titles take their sizes from style.apply()
PI, DNDS, ALPHA, DELTA = "protein identity π", "log₁₀ dN/dS (human–mouse)", "α", "Δα"
KEY_EVIDENCE, KEY_IMPUTED, KEY_TREND, KEY_BEYOND = "genes with count evidence", "genes without count evidence (Δα imputed from features)", "median in 14 equal-count bins", "genes beyond an axis"
RASTER_DPI = 360  # rasterized gene clouds: above 300 ppi on the page


def trend(ax, x, y, bins=14):
    q = pd.qcut(x, bins, duplicates="drop")
    m = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True).median()
    ax.plot(m.x, m.y, color=TREND, lw=1.1)


def cloud(ax, x, y, supported):
    ax.scatter(x[~supported], y[~supported], s=1.2, alpha=0.18, color=GREY, linewidths=0, rasterized=True)
    ax.scatter(x[supported], y[supported], s=1.2, alpha=0.22, color=BLUE, linewidths=0, rasterized=True)


def beyond(ax, x, y, supported) -> int:
    """Genes outside the axes, drawn on the frame as triangles that point outward, so no gene is cut without a mark. Call after the limits
    are set. Returns the number of such genes."""
    x, y, supported = np.asarray(x, float), np.asarray(y, float), np.asarray(supported, bool)
    (x0, x1), (y0, y1) = sorted(ax.get_xlim()), sorted(ax.get_ylim())
    in_y = (y >= y0) & (y <= y1)
    n = 0
    for sel, mk in ((y > y1, "^"), (y < y0, "v"), ((x < x0) & in_y, "<"), ((x > x1) & in_y, ">")):
        for cls, col in ((~supported, GREY), (supported, BLUE)):
            s = sel & cls
            if s.any():
                ax.scatter(np.clip(x[s], x0, x1), np.clip(y[s], y0, y1), marker=mk, s=5, color=col, linewidths=0, clip_on=False, zorder=6)
                n += int(s.sum())
    return n


def note(ax, text: str, loc: str = "upper left") -> None:
    """A statistic in a corner of the axes, on a white patch where it lies on the cloud."""
    x, y, ha, va = {"upper left": (0.03, 0.93, "left", "top"), "lower left": (0.03, 0.06, "left", "bottom"), "upper right": (0.97, 0.93, "right", "top")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=S_TICK, color=INK, bbox=dict(fc="white", ec="none", alpha=0.8, pad=1))


def signed(v: float, _=None) -> str:
    """Tick label of a signed quantity, with a true minus sign."""
    return "0" if abs(v) < 1e-12 else f"{v:+.2f}".replace("-", "−")


def key_handles(with_identity: bool = False) -> list:
    """Legend entries for the gene clouds, the trend line and the marks on the frame."""
    h = [Line2D([], [], marker="o", ls="", color=BLUE, ms=3, label=KEY_EVIDENCE), Line2D([], [], marker="o", ls="", color=GREY, ms=3, label=KEY_IMPUTED),
         Line2D([], [], color=TREND, lw=1.1, label=KEY_TREND)]
    if with_identity:
        h.append(Line2D([], [], color=REF_LINE, lw=0.6, ls="--", label="α = π"))
    return h + [Line2D([], [], marker="^", ls="", color=MUTED, ms=3, label=KEY_BEYOND)]


def genes(c: str, gold: pd.DataFrame) -> dict:
    """The genes of one cohort with a human-mouse dN/dS: identity, log dN/dS, α, Δα and whether the gene has count evidence."""
    d = pd.read_csv(RES / f"T3_3/{c}__consensus.csv").set_index("hsapiens_gene_id")
    d = d[~d.index.duplicated()].join(gold).dropna(subset=["dNdS", "D", "alpha", "prior_raw"])
    d = d[d.dNdS > 0]
    return dict(pi=d.prior_raw.to_numpy(), logw=np.log10(d.dNdS.to_numpy().clip(1e-3)), a=d.alpha.to_numpy(), D=d.D.to_numpy(), sup=(d.S_ema.abs() > 1e-6).to_numpy())


def panels(g: dict) -> list:
    """The four rows of the figure for one cohort: x, y, their axis labels and the name of the row in the table."""
    return [(g["pi"], g["logw"], PI, DNDS, "pi_vs_dNdS"), (g["pi"], g["a"], PI, ALPHA, "pi_vs_alpha"), (g["pi"], g["D"], PI, DELTA, "pi_vs_D"),
            (g["logw"], g["D"], DNDS, DELTA, "dNdS_vs_D")]


def statistics() -> pd.DataFrame:
    """Spearman correlation of every row of the figure, per cohort: over all genes shown and over the genes with count evidence."""
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)[["dNdS"]]
    rows = []
    for c, _ in COHORTS:
        g = genes(c, gold)
        sup = g["sup"]
        for x, y, _, _, name in panels(g):
            r, rs = spearmanr(x, y)[0], spearmanr(x[sup], y[sup])[0]
            rows.append(dict(cohort=c, panel=name, n=len(x), rho=r, n_count_evidence=int(sup.sum()), rho_count_evidence=rs))
    return pd.DataFrame(rows)


def draw_panels(t: pd.DataFrame | None = None) -> None:
    """The figure, from the cached tables: gene clouds from the consensus tables and the dN/dS table, the printed correlations from
    `t` (default: results/analysis/X5/sequence_axis_rho.csv)."""
    style.apply()
    if t is None:
        t = pd.read_csv(OUT / "sequence_axis_rho.csv")
    rho = t.set_index(["cohort", "panel"]).rho
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)[["dNdS"]]
    W, left, right, gap_x, top, bottom, ax_h, row_gap = style.TEXTWIDTH, 40.0, 8.0, 28.0, 12.0, 43.0, 80.0, 26.0
    ax_w = (W - left - right - 2 * gap_x) / 3
    H = top + 4 * ax_h + 3 * row_gap + bottom
    fig = plt.figure(figsize=(style.inch(W), style.inch(H)))
    for j, (c, title) in enumerate(COHORTS):
        g = genes(c, gold)
        pi, sup = g["pi"], g["sup"]
        for i, (x, y, xl, yl, name) in enumerate(panels(g)):
            ax = fig.add_axes([(left + j * (ax_w + gap_x)) / W, (bottom + (3 - i) * (ax_h + row_gap)) / H, ax_w / W, ax_h / H])
            cloud(ax, x, y, sup)
            if name == "pi_vs_alpha":
                ax.plot([PI_MIN, 1], [PI_MIN, 1], color=REF_LINE, lw=0.6, ls="--")
            else:
                trend(ax, x, y)
            if name != "pi_vs_dNdS" and name != "pi_vs_alpha":
                ax.axhline(0, color=REF_LINE, lw=0.6, ls=":")
                ax.set_ylim(*LIMS[c])
                ax.yaxis.set_major_formatter(FuncFormatter(signed))
            if x is pi:
                ax.set_xlim(PI_MIN, 1.0)
            beyond(ax, x, y, sup)
            note(ax, f"ρ = {rho[(c, name)]:+.2f}".replace("-", "−"), "lower left" if name == "pi_vs_dNdS" else "upper left")
            ax.set_xlabel(xl, labelpad=2)
            if j == 0:
                ax.set_ylabel(yl.replace(" (", "\n("))
            if i == 0:
                ax.set_title(title, pad=4)
    fig.legend(handles=key_handles(with_identity=True), loc="lower center", bbox_to_anchor=(0.5, 2 / H), ncol=3, handletextpad=0.5, columnspacing=1.6, borderaxespad=0,
               labelspacing=0.3)
    with plt.rc_context({"savefig.dpi": RASTER_DPI}):
        style.save_panel(fig, "supp_sequence_axis/sequence_axis", exact=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t = statistics()
    t.to_csv(OUT / "sequence_axis_rho.csv", index=False)
    draw_panels(t)
    print(t.round(3).to_string())


if __name__ == "__main__":
    main()
