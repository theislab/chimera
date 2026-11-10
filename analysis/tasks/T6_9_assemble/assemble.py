"""T6.9: BH family (m = 7, 05 section 7) and the supplementary functional-layer figure.

Four panels, one file each in figures/panels/supp_functional_layer/, drawn at their exact size on the page (house style:
figures/style.py): pathways (F1, coherence), interactions (F2, coherence), interologs (F3, conservation), regulatory
(F4, conservation).

    python -m analysis.tasks.T6_9_assemble.assemble                                                  # the family table, then the panels
    python -c "from analysis.tasks.T6_9_assemble import assemble; assemble.draw_panels()"            # the panels alone, from the cached tables
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from analysis.common import controls as K  # noqa: E402
from analysis.common import gene_ids  # noqa: E402
from analysis.tasks.T4_3_scoreboard import stats as ST  # noqa: E402
from figures import style  # noqa: E402

RES = gene_ids.REPO_ROOT / "results/analysis"
OUT = RES / "T6_9"

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this figure
FAMILY = "supp_functional_layer"
CLAB = {c: style.cohort_name(c) for c in ("liv_4sp", "liv_hmouse", "test_3sp")}
# Readout of a row -> its name and its mark. r1 and r2 are Δα with covariates regressed out, r2 also with 64 components of ESM-2.
RLAB = {"PI": "protein prior π", "S1": style.READOUT["count_evidence"].name, "r1": "Δα, adjusted", "r2": "Δα, adjusted + ESM-2"}
MARK = {"PI": style.READOUT["protein_identity"], "S1": style.READOUT["count_evidence"], "r1": style.READOUT["delta_alpha"],
        "r2": style.READOUT["delta_alpha"]}
MOUSE_SOURCE = {"Skinnider_liver": "mouse liver interaction map", "STRING_mouse": "STRING, mouse"}  # sources of the mouse interactions
REF_LINE, NULL = "#888888", "#C9C9C9"  # REF_LINE: zero lines; NULL: bars of the permutation nulls

# Sizes in bp. Panels a and b stand beside a column of c over d. The three widths and two gaps fill the text width, and
# the column with its two letter bands and the gap between them is as tall as a and b (figures/compose.py: band 9.5, gap 6).
WIDTH = {"a": 150.0, "b": 176.0, "c": 146.0, "d": 146.0}
HEIGHT = {"a": 266.0, "b": 266.0, "c": 113.5, "d": 137.0}
LEFT = {"a": 76.0, "b": 86.0, "c": 76.0, "d": 52.0}  # room for the row labels
BELOW_AXES = {"a": 51.0, "b": 51.0, "c": 37.0, "d": 37.0}  # tick labels, axis label and the key
ABOVE_AXES, RIGHT, EDGE = 13.0, 6.0, 1.5
PLAIN = FuncFormatter(lambda v, _: "0" if v == 0 else f"{v:g}".replace("-", "−"))  # tick labels with a true minus sign


def new_panel(name, title):
    """A figure of the exact size of a panel: one axes at the margins of the panel, its title at the left edge. The rows
    are named by their labels, so the axes has no line and no tick marks on the left."""
    w, h = WIDTH[name], HEIGHT[name]
    fig = plt.figure(figsize=(style.inch(w), style.inch(h)))
    ax = fig.add_axes([LEFT[name] / w, BELOW_AXES[name] / h, (w - LEFT[name] - RIGHT) / w, (h - ABOVE_AXES - BELOW_AXES[name]) / h])
    fig.text(EDGE / w, 1 - EDGE / h, title, ha="left", va="top", fontsize=style.SIZE_TITLE)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    return fig, ax


def heading(fig, ax, y, text) -> None:
    """A group heading (cohort, source, readout) above its rows, from the left edge of the panel."""
    ax.text(EDGE / (72 * fig.get_figwidth()), y, text, transform=blended_transform_factory(fig.transFigure, ax.transData), ha="left", va="center",
            fontsize=style.SIZE_LABEL, fontweight="bold")


def key(fig, name, handles, labels, ncol=1) -> None:
    """The key of a panel, under its axis label, from the left edge of the panel."""
    fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(EDGE / WIDTH[name], (BELOW_AXES[name] - 27.0) / HEIGHT[name]),
               fontsize=style.SIZE_TEXT, handlelength=1.5, labelspacing=0.25, columnspacing=1.2, borderaxespad=0, borderpad=0, ncol=ncol)


def null_bar():
    return Line2D([], [], color=NULL, lw=4, solid_capstyle="butt")


def save(fig, name) -> None:
    """Save a panel at its exact size (placed with X(...) in the figure spec); nothing may reach beyond the page."""
    box = fig.get_tightbbox(fig.canvas.get_renderer())
    inside = box.x0 >= 0 and box.y0 >= 0 and box.x1 <= fig.get_figwidth() and box.y1 <= fig.get_figheight()
    assert inside, f"panel {name}: content box {[round(72 * v, 1) for v in box.extents]} bp beyond its page"
    style.save_panel(fig, f"{FAMILY}/{name}", exact=True)


def bh_family() -> pd.DataFrame:
    f1 = pd.read_csv(RES / "T6_3/F1_coherence.csv")
    f2 = pd.read_csv(RES / "T6_4/F2_moran.csv")
    f4 = pd.read_csv(RES / "T6_6/F4_regulatory.csv").set_index("version")
    rows = []
    for c in ("liv_4sp", "liv_hmouse"):
        r = f1[(f1.cohort == c) & (f1.library == "pooled") & (f1.readout == "r2") & (f1.null == "pi_x_mean")].iloc[0]
        rows.append(dict(test=f"F1 {c}", estimate=r["index"], null_ref=r.null_q975, p=r.p_index))
    for c in ("liv_4sp", "liv_hmouse"):
        for g in ("HuRI", "STRING"):
            r = f2[(f2.cohort == c) & (f2.graph == g) & (f2.edge_set == "all") & (f2.weights == "row")
                   & (f2.readout == "r2")].iloc[0]
            rows.append(dict(test=f"F2 {g} {c}", estimate=r.I, null_ref=max(r.perm_q975, r.ms_q975), p=r.p_larger))
    r = f4.loc["primary"]
    rows.append(dict(test="F4 liver - testis", estimate=r["diff"], null_ref=0.0, p=r.p_diff))
    t = pd.DataFrame(rows)
    t["q_BH"] = ST.bh(t.p.tolist())
    t["passes_q05"] = t.q_BH < 0.05
    K.write_csv(t, OUT / "bh_family.csv")
    return t


def panel_pathways() -> None:
    f = pd.read_csv(RES / "T6_3/F1_coherence.csv")
    f = f[(f.library == "pooled") & (f.null == "pi_x_mean")]
    fig, ax = new_panel("a", "Pathway (coherence)")
    y, ticks, labels = 0, [], []
    for c in CLAB:
        heading(fig, ax, y + 0.8, CLAB[c])
        for rd in ("PI", "S1", "r1", "r2"):
            r = f[(f.cohort == c) & (f.readout == rd)].iloc[0]
            ax.plot([r.null_q025, r.null_q975], [y, y], color=NULL, lw=4, solid_capstyle="butt", zorder=1)
            ax.plot(r["index"], y, MARK[rd].marker, color=MARK[rd].color, ms=3.8, zorder=3, mec="white", mew=0.4)
            ticks.append(y)
            labels.append(RLAB[rd])
            y -= 1
        y -= 0.9
    ax.set_xscale("log")
    ax.set_xlim(0.6, 14)
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xticklabels(["1", "2", "5", "10"])
    ax.minorticks_off()
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 0.6, 1.3)
    ax.set_xlabel("pathway coherence index", loc="right")
    key(fig, "a", [null_bar()], ["95% interval of the permutation null"])
    save(fig, "pathways")


def panel_interactions() -> None:
    f = pd.read_csv(RES / "T6_4/F2_moran.csv")
    f = f[f.weights == "row"]
    fig, ax = new_panel("b", "Protein interaction (coherence)")
    at_edge = blended_transform_factory(fig.transFigure, ax.transData)
    y, ticks, labels = 0, [], []
    for c in CLAB:
        heading(fig, ax, y + 0.8, CLAB[c])
        for g in ("HuRI", "STRING"):
            ax.text((EDGE + 4.0) / WIDTH["b"], y - 0.5, g, transform=at_edge, rotation=90, ha="center", va="center", fontsize=style.SIZE_TICK)
            for rd in ("r1", "r2"):
                a = f[(f.cohort == c) & (f.graph == g) & (f.edge_set == "all") & (f.readout == rd)].iloc[0]
                b = f[(f.cohort == c) & (f.graph == g) & (f.edge_set == "no_top1pct_esm") & (f.readout == rd)].iloc[0]
                hi = max(a.perm_q975, a.ms_q975)
                ax.plot([-0.03, hi], [y, y], color=NULL, lw=4, solid_capstyle="butt", zorder=1)
                ax.plot(b.I, y, MARK[rd].marker, mfc="white", mec=MARK[rd].color, mew=0.7, ms=3.6, zorder=3)
                ax.plot(a.I, y, MARK[rd].marker, color=MARK[rd].color, ms=3.8, zorder=4, mec="white", mew=0.4)
                ticks.append(y)
                labels.append(RLAB[rd])
                y -= 1
        y -= 0.9
    ax.axvline(0, color=REF_LINE, lw=style.LW_REF)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 0.6, 1.3)
    ax.set_xlim(-0.03, 0.24)
    ax.set_xlabel("Moran's I", loc="right")
    m = MARK["r1"]
    key(fig, "b", [Line2D([], [], marker=m.marker, color=m.color, ms=3.8, ls="", mec="white", mew=0.4),
              Line2D([], [], marker=m.marker, mfc="white", mec=m.color, mew=0.7, ms=3.6, ls=""), null_bar()],
        ["all edges", "without the 1% most similar protein pairs", "null, up to its 97.5th percentile"])
    save(fig, "interactions")


def panel_interologs() -> None:
    t = pd.read_csv(RES / "T6_4/F3_interolog.csv")
    fig, ax = new_panel("c", "Interolog retention (conservation)")
    y, ticks, labels = 0, [], []
    for src, name in MOUSE_SOURCE.items():
        heading(fig, ax, y + 0.8, name)
        for rd in ("r1", "r2"):
            r = t[(t.source == src) & (t.readout == rd)].iloc[0]
            ax.plot([r.null_q025, r.null_q975], [y, y], color=NULL, lw=4, solid_capstyle="butt", zorder=1)
            ax.plot([r.ci_lo, r.ci_hi], [y, y], color=MARK[rd].color, lw=style.LW_CI, zorder=2)
            ax.plot(r.rho_partial, y, MARK[rd].marker, color=MARK[rd].color, ms=3.8, zorder=3, mec="white", mew=0.4)
            ticks.append(y)
            labels.append(RLAB[rd])
            y -= 1
        y -= 1.0
    ax.axvline(0, color=REF_LINE, lw=style.LW_REF)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 1.4, 1.3)
    ax.set_xlim(-0.1, 0.1)
    ax.set_xticks([-0.1, 0, 0.1])
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("partial ρ with interolog retention", loc="right")
    key(fig, "c", [Line2D([], [], color=MARK["r1"].color, lw=style.LW_CI), null_bar()], ["95% CI", "permutation null"], ncol=2)
    save(fig, "interologs")


def panel_regulatory() -> None:
    f = pd.read_csv(RES / "T6_6/F4_regulatory.csv").set_index("version")
    fig, ax = new_panel("d", "Regulatory (conservation)")
    y, ticks, labels = 0, [], []
    for v, rd in (("primary", "r2"), ("r1", "r1")):
        r = f.loc[v]
        heading(fig, ax, y + 0.8, RLAB[rd])
        for name, e, lo, hi in (("liver", r.rho_liver, r.rho_liver_lo, r.rho_liver_hi),
                                ("testis", r.rho_testis, r.rho_testis_lo, r.rho_testis_hi),
                                ("liver − testis", r["diff"], r.diff_lo, r.diff_hi)):
            ax.plot([lo, hi], [y, y], color=MARK[rd].color, lw=style.LW_CI, zorder=2)
            if "−" in name:  # the difference of the two rows above it: an open diamond
                ax.plot(e, y, "D", mfc="white", mec=MARK[rd].color, mew=0.7, ms=3.4, zorder=3)
            else:
                ax.plot(e, y, MARK[rd].marker, color=MARK[rd].color, ms=3.8, zorder=3, mec="white", mew=0.4)
            ticks.append(y)
            labels.append(name)
            y -= 1
        y -= 1.0
    ax.axvline(0, color=REF_LINE, lw=style.LW_REF)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y + 1.4, 1.3)
    ax.set_xlim(-0.05, 0.09)
    ax.set_xticks([-0.04, 0, 0.04, 0.08])
    ax.xaxis.set_major_formatter(PLAIN)
    ax.set_xlabel("Spearman ρ with conserved TF binding", loc="right")
    key(fig, "d", [Line2D([], [], color=MARK["r1"].color, lw=style.LW_CI)], ["95% CI"])
    save(fig, "regulatory")


def draw_panels() -> None:
    """The four panels from the cached tables of the functional-layer tests; nothing is computed."""
    style.apply()
    panel_pathways()
    panel_interactions()
    panel_interologs()
    panel_regulatory()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = bh_family()
    pd.set_option("display.width", 160)
    print(t.round(4).to_string(index=False))
    draw_panels()


if __name__ == "__main__":
    main()
