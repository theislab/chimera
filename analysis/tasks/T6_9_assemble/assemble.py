"""T6.9: BH family (m = 7, 05 section 7) and the supplementary functional-layer figure.

One panel per noun of M2: a pathway (F1), b protein interaction (F2 coherence, F3 conservation),
c regulatory conservation (F4). PDF with embedded fonts through ``notebooks/_figure_design.py``.

    python analysis/tasks/T6_9_assemble/assemble.py <path to supp_functional_layer.pdf>
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import git_sha
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.T8_4_figures import common as CM

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as FD  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

RES = gene_ids.REPO_ROOT / "results/analysis"
OUT = RES / "T6_9"
CLAB = {"liv_4sp": "Liver 4-sp", "liv_hmouse": "Liver H+Mouse", "test_3sp": "Testis 3-sp (secondary)"}
RLAB = {"PI": r"$\pi$", "S1": "S", "r1": "r1", "r2": "r2"}
INK, GREY, OBS2 = "#222222", "#C9C9C9", "#222222"


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


def panel_a(ax) -> None:
    f = pd.read_csv(RES / "T6_3/F1_coherence.csv")
    f = f[(f.library == "pooled") & (f.null == "pi_x_mean")]
    y, ticks, labels = 0, [], []
    for c in CLAB:
        ax.text(0.62, y + 0.72, CLAB[c], fontsize=6.5, color=INK, ha="left", va="center")
        for rd in ("PI", "S1", "r1", "r2"):
            r = f[(f.cohort == c) & (f.readout == rd)].iloc[0]
            ax.plot([r.null_q025, r.null_q975], [y, y], color=GREY, lw=4, solid_capstyle="butt", zorder=1)
            ax.plot(r["index"], y, "o", color=INK, ms=3.5, zorder=3, mec="white", mew=0.4)
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
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.6, 1.3)
    ax.set_xlabel("Pathway coherence index")
    ax.spines["left"].set_visible(False)


def panel_b(ax, ax2) -> None:
    f = pd.read_csv(RES / "T6_4/F2_moran.csv")
    f = f[f.weights == "row"]
    y, ticks, labels = 0, [], []
    for c in CLAB:
        ax.text(0.004, y + 0.72, CLAB[c], fontsize=6.5, color=INK, ha="left", va="center")
        for g in ("HuRI", "STRING"):
            for rd, lab in (("r1", "r1"), ("r2", "r2")):
                a = f[(f.cohort == c) & (f.graph == g) & (f.edge_set == "all") & (f.readout == rd)].iloc[0]
                b = f[(f.cohort == c) & (f.graph == g) & (f.edge_set == "no_top1pct_esm") & (f.readout == rd)].iloc[0]
                hi = max(a.perm_q975, a.ms_q975)
                ax.plot([-0.03, hi], [y, y], color=GREY, lw=4, solid_capstyle="butt", zorder=1)
                ax.plot(b.I, y, "o", mfc="white", mec=INK, mew=0.7, ms=3.5, zorder=3)
                ax.plot(a.I, y, "o", color=INK, ms=3.5, zorder=4, mec="white", mew=0.4)
                ticks.append(y)
                labels.append(f"{g} {lab}")
                y -= 1
        y -= 0.9
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.6, 1.3)
    ax.set_xlim(-0.03, 0.24)
    ax.set_xlabel("Moran's I")
    ax.spines["left"].set_visible(False)
    ax.plot([], [], "o", color=INK, ms=3.5, label="all edges")
    ax.plot([], [], "o", mfc="white", mec=INK, mew=0.7, ms=3.5, label="without top 1% ESM")
    ax.plot([], [], color=GREY, lw=4, label="below null 97.5%")
    ax.legend(loc="center right", frameon=False, fontsize=6, handlelength=1.2, borderaxespad=0.1)
    t = pd.read_csv(RES / "T6_4/F3_interolog.csv")
    y, ticks, labels = 0, [], []
    for src, lab in (("Skinnider_liver", "Skinnider"), ("STRING_mouse", "STRING")):
        for rd in ("r1", "r2"):
            r = t[(t.source == src) & (t.readout == rd)].iloc[0]
            ax2.plot([r.null_q025, r.null_q975], [y, y], color=GREY, lw=4, solid_capstyle="butt", zorder=1)
            ax2.plot([r.ci_lo, r.ci_hi], [y, y], color=INK, lw=0.8, zorder=2)
            ax2.plot(r.rho_partial, y, "o", color=INK, ms=3.5, zorder=3, mec="white", mew=0.4)
            ticks.append(y)
            labels.append(f"{lab} {rd}")
            y -= 1
    ax2.axvline(0, color=INK, lw=0.5)
    ax2.set_yticks(ticks)
    ax2.set_yticklabels(labels)
    ax2.tick_params(axis="y", length=0)
    ax2.set_ylim(y + 0.3, 0.9)
    ax2.set_xlim(-0.1, 0.1)
    ax2.set_xticks([-0.1, 0, 0.1])
    ax2.set_xticklabels(["-0.1", "0", "0.1"])
    ax2.set_xlabel(r"Partial $\rho$, retention")
    ax2.spines["left"].set_visible(False)


def panel_c(ax) -> None:
    f = pd.read_csv(RES / "T6_6/F4_regulatory.csv").set_index("version")
    y, ticks, labels = 0, [], []
    for v, lab in (("primary", "r2"), ("r1", "r1")):
        r = f.loc[v]
        for name, e, lo, hi in (("Liver", r.rho_liver, r.rho_liver_lo, r.rho_liver_hi),
                                ("Testis", r.rho_testis, r.rho_testis_lo, r.rho_testis_hi),
                                ("Liver - testis", r["diff"], r.diff_lo, r.diff_hi)):
            ax.plot([lo, hi], [y, y], color=INK, lw=0.8, zorder=2)
            mf = "white" if name.startswith("Liver -") else INK
            ax.plot(e, y, "o", mfc=mf, mec=INK, mew=0.7, ms=3.5, zorder=3)
            ticks.append(y)
            labels.append(f"{name}, {lab}")
            y -= 1
        y -= 0.7
    ax.axvline(0, color=INK, lw=0.5)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y + 0.3, 0.8)
    ax.set_xlim(-0.05, 0.09)
    ax.set_xticks([-0.04, 0, 0.04, 0.08])
    ax.set_xticklabels(["-0.04", "0", "0.04", "0.08"])
    ax.set_xlabel(r"Spearman $\rho$($B_g$, residual)")
    ax.spines["left"].set_visible(False)


def main() -> None:
    dest = Path(sys.argv[1])
    OUT.mkdir(parents=True, exist_ok=True)
    t = bh_family()
    pd.set_option("display.width", 160)
    print(t.round(4).to_string(index=False))
    FD.apply_design()
    fig = plt.figure(figsize=(FD.DOUBLE_W + 0.4, 4.0))
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.0, 1.15, 1.0], height_ratios=[1, 1],
                  left=0.045, right=0.99, top=0.95, bottom=0.11, wspace=0.42, hspace=0.42)
    a = fig.add_subplot(gs[:, 0])
    b = fig.add_subplot(gs[:, 1])
    b2 = fig.add_subplot(gs[0, 2])
    c = fig.add_subplot(gs[1, 2])
    panel_a(a)
    panel_b(b, b2)
    panel_c(c)
    for ax, tag in ((a, "Pathway (coherence)"), (b, "Protein interaction (coherence)"),
                    (b2, "Interolog retention (conservation)"), (c, "Regulatory (conservation)")):
        ax.set_title(tag, fontsize=7.5, loc="left", pad=3)
    # One PDF per panel for F1_figures/compose.py, which adds the panel letters.
    for k, ax in (("a", a), ("b_left", b), ("b_right", b2), ("c", c)):
        CM.save_axes(fig, f"supp_functional_layer_{k}", [ax], out_dir=dest.parent)
    tmp = dest.with_name(dest.name + ".tmp")
    fig.savefig(tmp, format="pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUT / "supp_functional_layer.png", format="png", dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    os.replace(tmp, dest)
    (OUT / "_provenance.json").write_text(json.dumps(dict(git_sha=git_sha(), figure=str(dest))))


if __name__ == "__main__":
    main()
