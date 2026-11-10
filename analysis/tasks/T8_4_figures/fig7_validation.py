"""Fig 7 (new main figure): independent validation and benchmark.

Panels, one PDF each (``figures/revision/fig7{a..e}_*.pdf``):
  a  forest of H1-H4: rho_p of Delta-alpha (D) with 95% block-bootstrap CI, primary cohort and the
     multi-species cohort used in Figs 4-6, status letter (T4.3 ``scoreboard.csv``, ``row.json``).
  b  set discrimination, the 9 pre-specified H5/H6 tests (T4.3 ``sets.csv``, family F2).
  c  head-to-head: Delta = rho_p(D) - rho_p(baseline) for B1b, B1c, B2b, B3a, B1z on each gold
     (T4.3 ``head_to_head.csv``); gene-level tool scalars of SATURN and CAMEX against the
     three-seed D (T5.6). Testis tool rows are drawn as a marked placeholder until T5.6 writes them.
  d  incremental information: each score given covariates and all other simple baselines
     (X6 ``incremental_rho.csv``, exploratory).
  e  sequence axis: rank correlations of protein identity, dN/dS, alpha and Delta-alpha in every cohort
     (H1 ``axis.csv``; later cohorts join when H1 adds them; equal to X5 ``sequence_axis_rho.csv`` where both exist).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from analysis.tasks.T8_4_figures import common as cm

T43 = cm.RES / "T4_3"
HS = ["H1", "H3", "H2", "H4"]  # liver first, then testis; level before pattern
MULTI = {"H1": "liv_4sp", "H2": "test_3sp", "H3": "liv_4sp", "H4": "test_3sp"}
STATUS_WORD = {"V": "V", "N": "N", "R": "R", "W": "W"}
V_BAR = 0.10


def _ci(ax, y, est, lo, hi, color, marker, filled=True, size=4.2, lw=1.0, z=3):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=z)
    ax.plot([est], [y], marker=marker, ms=size, color=color, mfc=color if filled else "white",
            mec=color, mew=0.9, zorder=z + 1, linestyle="none")


def panel_forest() -> plt.Figure:
    sb = pd.read_csv(T43 / "scoreboard.csv")
    row = json.loads((T43 / "row.json").read_text())
    status = row["step1_status"]
    # Exact size and axes, shared with the set panel beside it (F1): same axis baseline and height.
    fig = plt.figure(figsize=(2.98, 2.3))
    ax = fig.add_axes([0.98 / 2.98, 0.56 / 2.3, 1.62 / 2.98, 1.62 / 2.3])
    yt, yl, y = [], [], 0.0
    rows = []
    for h in HS:
        p = sb[(sb.id == f"{h}:D")].iloc[0]
        m = sb[(sb.id == f"{h}:multi:D")].iloc[0]
        rows.append((h, p, m))
    for h, p, m in rows:
        _ci(ax, y, p.rho_p, p.ci_lo, p.ci_hi, cm.STY["D"][0], "o", True)
        yt.append(y)
        yl.append(f"{cm.GOLD_SHORT[h]} ({h})\n{cm.COHORT[p.cohort]}")
        ax.text(1.02, y, status[h], transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=7.5, fontweight="bold", color=cm.INK)
        y += 1
        _ci(ax, y, m.rho_p, m.ci_lo, m.ci_hi, cm.STY["D"][0], "o", False)
        yt.append(y)
        yl.append(f"{cm.COHORT[m.cohort]}")
        # multi-species estimates sit outside the decision procedure: no status of their own
        ax.text(1.02, y, "–",
                transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6, color=cm.MUTED)
        y += 1.6
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
    ax.axvline(V_BAR, color=cm.RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(V_BAR, -0.9, "0.10", ha="center", va="bottom", fontsize=5.5, color=cm.MUTED)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6)
    for t, lab in zip(ax.get_yticklabels(), yl):
        if "\n" not in lab:
            t.set_color(cm.MUTED)
    ax.set_ylim(y - 1.2, -1.0)
    ax.set_xlim(-0.3, 0.2)
    ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xlabel("Δα vs independent measure, $\\rho_p$ (95% CI)", fontsize=6.5, labelpad=1)
    ax.text(1.02, -0.95, "status", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=6, color=cm.MUTED)
    fig.legend(handles=[Line2D([], [], marker="o", color=cm.STY["D"][0], ls="none", ms=4, label="primary cohort"),
                        Line2D([], [], marker="o", color=cm.STY["D"][0], mfc="white", ls="none", ms=4,
                               label="multi-species cohort")],
               loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=5.8, handletextpad=0.2, frameon=False,
               borderpad=0)
    return fig


def _tool_rows() -> tuple[pd.DataFrame, bool]:
    """Tool head-to-head rows; final T5.6 output if present, else the liver-first run."""
    final = cm.RES / "T5_6" / "scoreboard" / "head_to_head.csv"
    src = final if final.exists() else cm.RES / "T5_6" / "liver_first" / "scoreboard" / "head_to_head.csv"
    t = pd.read_csv(src)
    t = t[t.analysis == "tool_vs_D3"].copy()
    return t, final.exists()


TOOL_LABEL = {"S_deploy_native": "SATURN gene score, native genes",
              "S_deploy_1to1": "SATURN gene score, 1:1 orthologs",
              "C_cos": "CAMEX gene-embedding cosine"}


def panel_head_to_head() -> plt.Figure:
    hh = pd.read_csv(T43 / "head_to_head.csv")
    hh = hh[hh.analysis == "head_to_head"]
    row = json.loads((T43 / "row.json").read_text())
    outcome = row["step2_head_to_head"]
    tools, final = _tool_rows()
    order = ["B1b", "B1c", "B2b", "B3a", "B1z"]
    # Exact size, axes shared with the incremental panel below it (F1).
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.35), sharey=True,
                             gridspec_kw=dict(left=2.42 / 6.75, right=0.995, top=1 - 0.31 / 2.35, bottom=0.53 / 2.35,
                                              wspace=0.09))
    labels = [cm.STY[b][2] for b in order] + [TOOL_LABEL[k] for k in TOOL_LABEL]
    ys = list(range(len(order))) + [len(order) + 1.15 + i for i in range(len(TOOL_LABEL))]
    for ax, h in zip(axes, HS):
        sub = hh[hh.hyp == h].set_index("comparator")
        for y, b in zip(ys, order):
            r = sub.loc[b]
            ax.plot([r.sim95_lo, r.sim95_hi], [y, y], color=cm.STY[b][0], lw=0.6, alpha=0.6, zorder=2)
            _ci(ax, y, r.delta, r.pc_lo, r.pc_hi, cm.STY[b][0], cm.STY[b][1], True, size=3.4, lw=1.4)
        ts = tools[tools.hyp == h].set_index("comparator") if len(tools) else pd.DataFrame()
        for y, k in zip(ys[len(order):], TOOL_LABEL):
            if k in ts.index:
                r = ts.loc[k]
                ax.plot([r.sim95_lo, r.sim95_hi], [y, y], color=cm.INK, lw=0.6, alpha=0.6, zorder=2)
                _ci(ax, y, r.delta, r.pc_lo, r.pc_hi, cm.INK, "d", False, size=3.4, lw=1.4)
        if h in ("H2", "H4") and not (len(ts) and set(TOOL_LABEL) & set(ts.index)):
            y0, y1 = ys[len(order)] - 0.4, ys[-1] + 0.4
            ax.add_patch(plt.Rectangle((-0.5, y0), 0.8, y1 - y0, facecolor="#F2F2F2", edgecolor=cm.RULE,
                                       lw=0.5, ls=(0, (2, 2)), zorder=2))
            ax.text(-0.1, (y0 + y1) / 2, "testis tool scores\npending", ha="center", va="center",
                    fontsize=5.8, color=cm.MUTED, style="italic", zorder=3)
        ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
        ax.axhline(len(order) - 0.45, color=cm.RULE, lw=0.4)
        ax.set_title(f"{cm.GOLD_SHORT[h]} ({h})\nresult: {outcome[h].lower()}", fontsize=6.5)
        ax.set_xlim(-0.5, 0.3)
        ax.set_xticks([-0.4, -0.2, 0, 0.2])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=6)
    axes[0].set_ylim(ys[-1] + 0.7, -0.7)
    axes[0].text(-0.03, len(order) + 0.1, "gene-level tools (vs Δα mean of 3 seeds,\non the genes each tool scores)",
                 transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=5.3,
                 color=cm.MUTED, style="italic")
    fig.text(0.5 * (2.42 / 6.75 + 0.995), 0.01, "difference in $\\rho_p$, Δα minus comparator (95% CI; thin: simultaneous)\n"
             "← comparator closer to independent measure        Δα closer →", fontsize=6.3, ha="center", va="bottom")
    return fig


SETS = [("liver", "C1", "housekeeping genes", "above"), ("liver", "C2", "ribosome, proteasome, spliceosome", "above"),
        ("liver", "V1", "xenobiotic and bile-acid metabolism", "below"),
        ("liver", "V2", "interferon and inflammatory response", "below"),
        ("liver", "V3", "secreted (extracellular region)", "below"),
        ("testis", "C1", "housekeeping genes", "above"), ("testis", "C2", "ribosome, proteasome, spliceosome", "above"),
        ("testis", "V2", "interferon and inflammatory response", "below"),
        ("testis", "V3", "secreted (extracellular region)", "below")]


def panel_sets() -> plt.Figure:
    s = pd.read_csv(T43 / "sets.csv")
    s = s[(s.family == "F2") & (s.score == "D")].set_index(["tissue", "set"])
    assert len(s) == 9, len(s)
    # Exact size and axes, shared with the forest panel beside it (F1).
    fig = plt.figure(figsize=(3.67, 2.3))
    ax = fig.add_axes([2.04 / 3.67, 0.56 / 2.3, 1.3 / 3.67, 1.62 / 2.3])
    yt, yl = [], []
    y = 0.0
    for i, (tis, st, name, direc) in enumerate(SETS):
        if i == 5:
            y += 0.6
        r = s.loc[(tis, st)]
        _ci(ax, y, r.auroc, r.auroc_ci_lo, r.auroc_ci_hi, cm.STY["D"][0], "o", True)
        arrow = "↑" if direc == "above" else "↓"
        yt.append(y)
        yl.append(f"{tis}: {name} {arrow}")
        ax.text(1.02, y, f"{r.q_bh:.2f}", transform=ax.get_yaxis_transform(), ha="left", va="center",
                fontsize=6, color=cm.INK)
        y += 1
    ax.text(1.02, -0.9, "q", transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=6,
            color=cm.MUTED, style="italic")
    ax.axvline(0.5, color=cm.MUTED, lw=0.6, zorder=1)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=6)
    ax.set_ylim(y - 0.4, -1.0)
    ax.set_xlim(0.2, 0.7)
    fig.text(0.995, 0.01, "set vs matched background, AUROC of Δα (95% CI)\n"
             "arrow: predicted direction (↑ above 0.5, ↓ below)", fontsize=6.3, ha="right", va="bottom")
    return fig


INC_GOLDS = [("liv_hmouse", "G_H1", "H1"), ("liv_hmouse", "G_H3", "H3"),
             ("test_hmouse", "G_H2", "H2"), ("test_hmouse", "G_H4", "H4")]
INC_SCORES = [("D", "D", "all simple baselines"), ("S", "S", "all simple baselines"),
              ("Sdir", "Sdir", "all simple baselines"), ("B1b", "B1b_raw", "other simple baselines"),
              ("B1c", "B1c_raw", "other simple baselines"), ("B2b", "B2b", "other simple baselines"),
              ("B1z", "B1z_raw", "other simple baselines"), ("B3a", "B3a", "other simple baselines")]


def panel_incremental() -> plt.Figure:
    inc = pd.read_csv(cm.RES / "X6" / "incremental_rho.csv")
    inc = inc[inc.variant == "C*"]
    # Exact size, axes shared with the head-to-head panel above it (F1).
    fig, axes = plt.subplots(1, 4, figsize=(6.75, 2.2), sharey=True,
                             gridspec_kw=dict(left=2.42 / 6.75, right=0.995, top=1 - 0.31 / 2.2, bottom=0.42 / 2.2,
                                              wspace=0.09))
    for ax, (coh, gold, h) in zip(axes, INC_GOLDS):
        d = inc[(inc.cohort == coh) & (inc.gold == gold)]
        n = int(d.n.iloc[0])
        for y, (key, score, given) in enumerate(INC_SCORES):
            r = d[(d.score == score) & (d.given == given)].iloc[0]
            col, mk, _ = cm.STY[key]
            _ci(ax, y + (0.4 if y >= 3 else 0), r.rho_p, r.ci_lo, r.ci_hi, col, mk, True, size=3.4, lw=1.2)
        ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
        ax.axhline(2.7, color=cm.RULE, lw=0.4)
        ax.set_title(f"{cm.GOLD_SHORT[h]} ({h})\nn = {n:,}", fontsize=6.5)
        ax.set_xlim(-0.27, 0.22)
        ax.set_xticks([-0.2, 0, 0.2])
    ys = [y + (0.4 if y >= 3 else 0) for y in range(len(INC_SCORES))]
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([cm.STY[k][2] for k, _, _ in INC_SCORES], fontsize=6)
    axes[0].set_ylim(ys[-1] + 0.6, -0.6)
    fig.text(0.5 * (2.42 / 6.75 + 0.995), 0.01,
             "$\\rho_p$ with independent measure, given covariates and all other simple baselines (95% CI)",
             fontsize=6.3, ha="center", va="bottom")
    return fig


SEQ_ROWS = [("pi_vs_dNdS", "rho_pi_dNdS", "protein identity vs dN/dS"), ("pi_vs_alpha", "rho_alpha_pi", "protein identity vs α"),
            ("pi_vs_D", "rho_D_pi", "protein identity vs Δα"), ("dNdS_vs_D", "rho_D_dNdS", "dN/dS vs Δα")]
SEQ_TISSUE = {"liver": ("o", "#0072B2"), "testis": ("^", "#D55E00"), "bone marrow": ("s", "#CC79A7"),
              "kidney": ("D", "#009E73"), "islet": ("v", "#E69F00"), "eye": ("P", "#5B3E80")}


def panel_sequence_axis() -> plt.Figure:
    ax_t = pd.read_csv(cm.RES / "H1" / "axis.csv")
    x5 = pd.read_csv(cm.RES / "X5" / "sequence_axis_rho.csv").set_index(["cohort", "panel"])
    ax_t = ax_t[ax_t.tissue.isin(SEQ_TISSUE)]  # H1 also lists the C1 cohorts; Fig 4g keeps the 16 until its layout is extended
    A, E = ax_t[ax_t.universe == "all"].set_index("cohort"), ax_t[ax_t.universe == "count_support"].set_index("cohort")
    for (coh, panel), r in x5.iterrows():  # the same statistic as X5 on the cohorts both have
        col = dict((k, c) for k, c, _ in SEQ_ROWS)[panel]
        assert abs(A.loc[coh, col] - r.rho) < 1e-6, (coh, panel, A.loc[coh, col], r.rho)
        assert abs(E.loc[coh, col] - r.rho_count_evidence) < 1e-6, (coh, panel)
    # Exact size and axes, shared with the margin-support panel above it in Fig 4 (F1).
    fig = plt.figure(figsize=(3.4, 1.85))
    ax = fig.add_axes([1.06 / 3.4, 0.62 / 1.85, 2.3 / 3.4, 1.21 / 1.85])
    cohorts = list(A.index)
    for i, (_, col, _lab) in enumerate(SEQ_ROWS):
        for j, c in enumerate(cohorts):
            mk, color = SEQ_TISSUE[A.loc[c, "tissue"]]
            y = i + (j - (len(cohorts) - 1) / 2) * 0.052
            ax.plot([A.loc[c, col]], [y], marker=mk, ms=3, color=color, ls="none", mew=0, zorder=3, alpha=0.9)
            ax.plot([E.loc[c, col]], [y], marker=mk, ms=3, color=color, mfc="white", mew=0.5, ls="none", zorder=2)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
    ax.set_yticks(range(len(SEQ_ROWS)))
    ax.set_yticklabels([lab for _, _, lab in SEQ_ROWS], fontsize=5.8)
    ax.set_ylim(len(SEQ_ROWS) - 0.5, -0.5)
    ax.set_xlim(-1, 1)
    ax.tick_params(axis="x", labelsize=6)
    ax.set_xlabel("Spearman ρ across genes, one point per cohort", fontsize=6, labelpad=1)
    h = [Line2D([], [], marker=m, color=col, ls="none", ms=3.2, mew=0, label=t) for t, (m, col) in SEQ_TISSUE.items()]
    h.append(Line2D([], [], marker="o", color=cm.INK, mfc="white", mew=0.6, ls="none", ms=3.2,
                    label="open: genes with count evidence only"))
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=5.3, handletextpad=0.1,
               columnspacing=0.7, frameon=False, borderpad=0, labelspacing=0.15)
    return fig


def main() -> list:
    cm.setup()
    out = []
    for name, fn in [("fig7a_forest_H1_H4", panel_forest), ("fig7b_set_discrimination", panel_sets),
                     ("fig7c_head_to_head", panel_head_to_head), ("fig7d_incremental_information", panel_incremental),
                     ("fig7e_sequence_axis", panel_sequence_axis)]:
        out.append(cm.save(fn(), name, exact=True))
    return out


if __name__ == "__main__":
    for p in main():
        print(p, cm.check_pdf(p))
