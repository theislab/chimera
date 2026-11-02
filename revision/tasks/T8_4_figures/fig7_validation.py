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
  e  sequence axis: rank correlations of protein identity, dN/dS, alpha and Delta-alpha
     (X5 ``sequence_axis_rho.csv``).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from revision.tasks.T8_4_figures import common as cm

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
    fig, ax = plt.subplots(figsize=(3.5, 2.95), layout="constrained")
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
        holds = (np.sign(m.rho_p) == np.sign(p.rho_p)) and (m.ci_lo > 0 or m.ci_hi < 0)
        yt.append(y)
        yl.append(f"{cm.COHORT[m.cohort]}")
        ax.text(1.02, y, "holds" if (holds and status[h] in ("V", "W")) else "–",
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
    ax.set_xlabel("Δα vs independent measure, $\\rho_p$ (95% CI)")
    ax.text(1.02, -0.95, "status", transform=ax.get_yaxis_transform(), ha="left", va="bottom",
            fontsize=6, color=cm.MUTED)
    fig.legend(handles=[Line2D([], [], marker="o", color=cm.STY["D"][0], ls="none", ms=4, label="primary cohort"),
                        Line2D([], [], marker="o", color=cm.STY["D"][0], mfc="white", ls="none", ms=4,
                               label="multi-species cohort (Figs 4-6)")],
               loc="outside lower center", ncol=2, fontsize=5.8, handletextpad=0.2)
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
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.7), sharey=True, layout="constrained")
    labels = [cm.STY[b][2] for b in order] + [TOOL_LABEL[k] for k in TOOL_LABEL]
    ys = list(range(len(order))) + [len(order) + 0.6 + i for i in range(len(TOOL_LABEL))]
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
        ax.axhline(len(order) - 0.2, color=cm.RULE, lw=0.4)
        ax.set_title(f"{cm.GOLD_SHORT[h]} ({h})\nresult: {outcome[h].lower()}", fontsize=6.5)
        ax.set_xlim(-0.5, 0.3)
        ax.set_xticks([-0.4, -0.2, 0, 0.2])
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels(labels, fontsize=6)
    axes[0].set_ylim(ys[-1] + 0.7, -0.7)
    axes[0].text(-0.03, len(order) + 0.05, "gene-level tools (vs Δα mean of 3 seeds,\non the genes each tool scores)",
                 transform=axes[0].get_yaxis_transform(), ha="right", va="center", fontsize=5.3,
                 color=cm.MUTED, style="italic")
    fig.supxlabel("difference in $\\rho_p$, Δα minus comparator (95% CI; thin: simultaneous)\n"
                  "← comparator closer to independent measure        Δα closer →", fontsize=6.5)
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
    fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="constrained")
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
    ax.set_xlabel("set vs matched background, AUROC of Δα (95% CI)\n"
                  "arrow: predicted direction (↑ above 0.5, ↓ below)", fontsize=6.5)
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
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.5), sharey=True, layout="constrained")
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
    fig.supxlabel("$\\rho_p$ with independent measure, given covariates and all other simple baselines (95% CI)",
                  fontsize=6.5)
    return fig


SEQ_ROWS = [("pi_vs_dNdS", "protein identity vs dN/dS"), ("pi_vs_alpha", "protein identity vs α"),
            ("pi_vs_D", "protein identity vs Δα"), ("dNdS_vs_D", "dN/dS vs Δα")]
SEQ_COH = [("liv_hmouse", "o"), ("liv_4sp", "s"), ("test_hmouse", "^")]


def panel_sequence_axis() -> plt.Figure:
    x5 = pd.read_csv(cm.RES / "X5" / "sequence_axis_rho.csv").set_index(["cohort", "panel"])
    fig, ax = plt.subplots(figsize=(3.4, 2.6), layout="constrained")
    for i, (key, _) in enumerate(SEQ_ROWS):
        for j, (coh, mk) in enumerate(SEQ_COH):
            r = x5.loc[(coh, key)]
            y = i + (j - 1) * 0.22
            ax.plot([r.rho], [y], marker=mk, ms=4, color=cm.INK, ls="none", zorder=3)
            ax.plot([r.rho_count_evidence], [y], marker=mk, ms=4, color=cm.INK, mfc="white",
                    mew=0.9, ls="none", zorder=3)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
    ax.set_yticks(range(len(SEQ_ROWS)))
    ax.set_yticklabels([lab for _, lab in SEQ_ROWS], fontsize=6.5)
    ax.set_ylim(len(SEQ_ROWS) - 0.5, -0.5)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Spearman ρ across genes")
    h = [Line2D([], [], marker=mk, color=cm.INK, ls="none", ms=4, label=cm.COHORT[c]) for c, mk in SEQ_COH]
    h.append(Line2D([], [], marker="o", color=cm.INK, mfc="white", ls="none", ms=4,
                    label="open: genes with count evidence only"))
    fig.legend(handles=h, loc="outside lower center", ncol=2, fontsize=5.6, handletextpad=0.1,
               columnspacing=0.8)
    return fig


def main() -> list:
    cm.setup()
    out = []
    for name, fn in [("fig7a_forest_H1_H4", panel_forest), ("fig7b_set_discrimination", panel_sets),
                     ("fig7c_head_to_head", panel_head_to_head), ("fig7d_incremental_information", panel_incremental),
                     ("fig7e_sequence_axis", panel_sequence_axis)]:
        out.append(cm.save(fn(), name))
    return out


if __name__ == "__main__":
    for p in main():
        print(p, cm.check_pdf(p))
