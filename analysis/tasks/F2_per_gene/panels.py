"""Per-gene gate panels: Δα read gene by gene from counts (per-gene gate) beside the protein-feature gate, on the
independent golds. Exploratory (decided after the pre-specified test); every number comes from the outputs of
Z3_free_gate or from ``F2_per_gene/data.py``, which recomputes them with the harness of Z3_free_gate and checks them
against its tables.

Panels, one PDF each at its exact size (``figures/panels/main_validation/*.pdf``):
  a  forest: rho_p with 95% block-bootstrap CI on H1-H4 and the 4-species liver estimates, per-gene gate (free5) vs
     protein-feature gate (can5, same seeds), all genes and genes with count evidence; pre-specified value marked;
     paired difference with its CI (Z3 compare_hyp.csv, compare_pairs.csv); status letters only for the frozen
     pre-specified test of the protein-feature gate (tab:validation), with their key on the panel; the default gate is drawn first
  b  testis level: Δα rank against EVEE testis level, raw and given the covariates C, binned (data.py; the reversal)
  c  Δα against the count evidence S_g of the same fit, per gene, liver and testis (Z3 consensus)
  d  named findings: spermatocyte-peak genes and hepatocyte catabolic genes, matched excess under each gate
     (Z3 compare_y3.csv)
  e  separate axis: rho(Δα, dN/dS) and rho(Δα, protein prior) under each gate (Z3 compare_x5.csv)

Inputs: ``results/analysis/Z3/`` (``compare_hyp.csv``, ``compare_pairs.csv``, ``compare_reversal.csv``,
``compare_y3.csv``, ``compare_x5.csv``) and ``results/analysis/F2/`` (``testis_level_residuals.csv``,
``gate_vs_S.csv``), bundle ``analysis_results``. "Z3" in this file is the task Z3_free_gate.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.F2_per_gene.panels
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import spearmanr

from analysis.tasks.T8_4_figures import common as cm

Z3 = cm.RES / "Z3"
F2 = cm.RES / "F2"
PREVIEW = F2 / "preview"

# Colors and markers of the house style (figures/style.py): the default gate as in every other figure.
CAN = (cm.style.READOUT["delta_alpha"].color, cm.style.READOUT["delta_alpha"].marker, "Δα, protein-feature gate")
FREE = (cm.style.READOUT["delta_alpha_per_gene"].color, cm.style.READOUT["delta_alpha_per_gene"].marker, "Δα, per-gene gate")
SG = (cm.style.READOUT["count_evidence"].color, cm.style.READOUT["count_evidence"].marker, "count evidence $S_g$")
FS = 6.0  # tick and row labels
FL = 6.3  # axis labels


def save(fig: plt.Figure, name: str):
    """Panel at its exact page size (the design default is a tight crop), so the axes line up in the composed figure."""
    pdf = cm.style.save_panel(fig, cm.style.panel_for(name), exact=True, preview=False, close=False)
    with plt.rc_context({"savefig.bbox": None}):
        fig.savefig(PREVIEW / f"{name}.png", format="png", dpi=250)
    plt.close(fig)
    return pdf


def _ci(ax, y, est, lo, hi, sty, filled=True, size=3.6, lw=1.1, z=3):
    color, marker = sty[0], sty[1]
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt", zorder=z)
    ax.plot([est], [y], marker=marker, ms=size * (0.85 if marker == "D" else 1), color=color,
            mfc=color if filled else "white", mec=color, mew=0.8, zorder=z + 1, ls="none")


def _style(ax):
    ax.tick_params(labelsize=FS, length=2.5, pad=1.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ------------------------------------------------------------------------------------------------ a
ROWS = [("H1", "all", "liver level (H1)\nLiver Human+Mouse"), ("H1", "multi", "Liver 4 species"),
        ("H3", "all", "liver pattern (H3)\nLiver Human+Mouse"), ("H3", "multi", "Liver 4 species"),
        ("H2", "all", "testis level (H2)\nTestis Human+Mouse"), ("H4", "all", "testis pattern (H4)\nTestis Human+Mouse")]
UNI = {("all", "all"): "all", ("all", "cs"): "count_supported", ("multi", "all"): "multi_4sp_or_3sp",
       ("multi", "cs"): "multi_4sp_count_supported"}


STATUS_KEY = "pre-specified status: V supported, N null, R reversed; –, estimate outside the test"
PANEL_H, PANEL_BOTTOM, PANEL_TOP = 2.48, 0.74, 0.26  # forest and testis-level panels share one height and axes bottom (one row)


def panel_forest() -> plt.Figure:
    """Points with CIs for both gates; the only status letters are the frozen pre-specified ones of the
    protein-feature gate (tab:validation), in their own column with a key. The per-gene gate gets no status."""
    T = pd.read_csv(Z3 / "compare_hyp.csv")
    P = pd.read_csv(Z3 / "compare_pairs.csv")
    H = PANEL_H
    left, bottom, top = 1.15, PANEL_BOTTOM, PANEL_TOP
    ah = H - bottom - top
    aw = [0.98, 0.98, 0.78]
    gaps = [0.5, 0.12]
    W = left + sum(aw) + sum(gaps) + 0.14
    fig = plt.figure(figsize=(W, H))
    axes, x = [], left
    for i, w in enumerate(aw):
        axes.append(fig.add_axes([x / W, bottom / H, w / W, ah / H]))
        x += w + (gaps[i] if i < len(gaps) else 0)
    ys, y = [], 0.0
    for h, kind, _ in ROWS:
        ys.append(y)
        y += 1.0 if kind == "all" and h in ("H1", "H3") else 1.55
    ys = np.array(ys)

    def get(h, uni, line):
        return T[(T.hyp == h) & (T.universe == uni) & (T.line == line)].iloc[0]

    for j, sub in enumerate(("all", "cs")):
        ax = axes[j]
        for yy, (h, kind, _) in zip(ys, ROWS):
            u = UNI[(kind, sub)]
            f, c = get(h, u, "free5"), get(h, u, "can5")
            _ci(ax, yy - 0.17, c.rho_p, c.ci_lo, c.ci_hi, CAN)  # the default gate on top
            _ci(ax, yy + 0.17, f.rho_p, f.ci_lo, f.ci_hi, FREE)
            if sub == "all" and kind == "all":
                fz = get(h, "all", "frozen")
                ax.plot([fz.rho_p], [yy - 0.17], marker="|", ms=7, mew=1.0, color=cm.INK, zorder=6, ls="none")
            if sub == "all":  # frozen ten-seed status of the protein-feature gate; H1-H4 Human+Mouse only
                lab = get(h, "all", "frozen").status_rule if kind == "all" else "–"
                ax.text(1.0 + gaps[0] / 2 / aw[0], yy - 0.17, lab, transform=ax.get_yaxis_transform(),
                        ha="center", va="center", fontsize=FS, color=cm.INK if kind == "all" else cm.MUTED,
                        fontweight="bold" if kind == "all" else "normal")
        ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
        ax.axvline(0.10, color=cm.RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
        ax.set_xlim(-0.29, 0.21)
        ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
        ax.set_xticklabels(["−0.2", "", "0", "", "0.2"])
        ax.set_title(["all genes", "count evidence only"][j], fontsize=FL, pad=2)
    axes[0].text(1.0 + gaps[0] / 2 / aw[0], -0.62, "pre-specified\ntest (protein-\nfeature gate)",
                 transform=axes[0].get_yaxis_transform(), ha="center", va="bottom", fontsize=5.5, color=cm.MUTED,
                 linespacing=1.0)
    # paired difference
    ax = axes[2]
    for yy, (h, kind, _) in zip(ys, ROWS):
        for k, uni in enumerate(("all", "count_supported")):
            x_ = P[(P.hyp == h) & (P.universe == uni) & (P.comparator == "D_can5") & (P.multi == (kind == "multi"))]
            r = x_.iloc[0]
            _ci(ax, yy + (-0.17 if k == 0 else 0.17), r.delta, r.pc_lo, r.pc_hi, (cm.INK, "o"), filled=(k == 0),
                size=3.3)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
    for v in (-0.03, 0.03):
        ax.axvline(v, color=cm.RULE, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.set_xlim(-0.06, 0.3)
    ax.set_xticks([0, 0.1, 0.2, 0.3])
    ax.set_xticklabels(["0", "0.1", "0.2", "0.3"])
    ax.set_title("per-gene minus\nprotein-feature gate", fontsize=FL, pad=2, linespacing=1.0)
    for ax in axes:
        _style(ax)
        ax.set_ylim(ys[-1] + 0.6, -0.6)
        ax.set_yticks(ys)
        ax.set_yticklabels([])
    labs = [lab for _, _, lab in ROWS]
    axes[0].set_yticklabels(labs, fontsize=FS, linespacing=1.05)
    for t, lab in zip(axes[0].get_yticklabels(), labs):
        if "\n" not in lab:
            t.set_color(cm.MUTED)
    axes[1].set_yticks([])
    axes[2].set_yticks([])
    fig.text((left + aw[0] + gaps[0] / 2 + aw[1] / 2 - 0.2) / W, 0.325 / H,
             "Δα vs independent data, $\\rho_p$ (95% CI)", fontsize=FL, ha="center", va="bottom")
    x2 = left + aw[0] + gaps[0] + aw[1] + gaps[1] + aw[2] / 2
    fig.text(x2 / W, 0.325 / H, "difference (95% CI)", fontsize=FL, ha="center", va="bottom")
    h = [Line2D([], [], marker=CAN[1], color=CAN[0], ls="none", ms=3.1, label="protein-feature gate"),
         Line2D([], [], marker=FREE[1], color=FREE[0], ls="none", ms=3.6, label="per-gene gate"),
         Line2D([], [], marker="|", color=cm.INK, ls="none", ms=6, mew=1.0, label="pre-specified, 10 seeds"),
         Line2D([], [], marker="o", color=cm.INK, ls="none", ms=3.3, label="difference: all genes"),
         Line2D([], [], marker="o", color=cm.INK, mfc="white", ls="none", ms=3.3, label="count evidence only")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.13 / H), ncol=5, fontsize=5.5,
               handletextpad=0.1, columnspacing=0.45, frameon=False, borderpad=0)
    fig.text(0.5, 0.03 / H, STATUS_KEY, fontsize=5.5, ha="center", va="bottom", color=cm.MUTED)
    return fig


# ------------------------------------------------------------------------------------------------ b
def _binned(x, y, nb=20):
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    b = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1)
    xm = np.array([x[b == i].mean() for i in range(nb)])
    ym = np.array([y[b == i].mean() for i in range(nb)])
    se = np.array([y[b == i].std(ddof=1) / np.sqrt((b == i).sum()) for i in range(nb)])
    return xm, ym, se


def panel_testis_level() -> plt.Figure:
    d = pd.read_csv(F2 / "testis_level_residuals.csv")
    rev = pd.read_csv(Z3 / "compare_reversal.csv")
    W, H = 2.45, PANEL_H
    fig = plt.figure(figsize=(W, H))
    left, bottom, top, gap = 0.5, PANEL_BOTTOM, PANEL_TOP, 0.16
    aw = (W - left - gap - 0.12) / 2
    ah = H - bottom - top
    axes = [fig.add_axes([left / W, bottom / H, aw / W, ah / H]),
            fig.add_axes([(left + aw + gap) / W, bottom / H, aw / W, ah / H])]
    spec = [("rank", "ranks", "spearman"), ("resid", "given covariates", "rho_p")]
    for ax, (pre, title, stat) in zip(axes, spec):
        gx = d[f"{pre}_G"].to_numpy()
        if pre == "rank":
            gx = gx - gx.mean()
        for key, sty, lw in (("D_can5", CAN, 1.0), ("D_free5", FREE, 1.0), ("S_free5", SG, 0.7)):
            yv = d[f"{pre}_{key}"].to_numpy()
            yv = yv - yv.mean()
            xm, ym, se = _binned(gx, yv)
            if key == "S_free5":
                ax.plot(xm, ym, color=sty[0], lw=lw, ls=(0, (2.5, 1.5)), zorder=2)
                continue
            ax.fill_between(xm, ym - 1.96 * se, ym + 1.96 * se, color=sty[0], alpha=0.15, lw=0, zorder=1)
            ax.plot(xm, ym, color=sty[0], lw=lw, marker=sty[1], ms=2.1 if sty[1] == "D" else 2.4, mew=0, zorder=3)
        ax.axhline(0, color=cm.MUTED, lw=0.5, zorder=0)
        ax.set_title(title, fontsize=FL, pad=2)
        _style(ax)
        r = rev[(rev.universe == "all")].set_index("line")
        s = rev[(rev.universe == "all_S_ema")].set_index("line")
        vals = [(CAN, r.loc["can5", stat]), (FREE, r.loc["free5", stat]), (SG, s.loc["free5", stat])]
        x0 = 0.62 if pre == "rank" else 0.05  # the empty corner: lower right of the raw panel, lower left given C
        y0 = 0.30
        ax.text(x0, y0, "ρ" if stat == "spearman" else "$\\rho_p$", transform=ax.transAxes, fontsize=5.6,
                va="center", ha="left", color=cm.MUTED)
        for k, (sty, v) in enumerate(vals):
            yy = y0 - (k + 1) * 0.085
            if sty is SG:
                ax.plot([x0, x0 + 0.07], [yy, yy], transform=ax.transAxes, color=sty[0], lw=0.8, ls=(0, (2.5, 1.5)),
                        clip_on=False)
            else:
                ax.plot([x0 + 0.035], [yy], transform=ax.transAxes, marker=sty[1], color=sty[0], ms=2.6, mew=0,
                        clip_on=False)
            ax.text(x0 + 0.1, yy, f"{v:+.2f}".replace("-", "−"), transform=ax.transAxes, fontsize=5.6, va="center",
                    ha="left", color=cm.INK)
    lim = max(abs(np.array(ax.get_ylim())).max() for ax in axes)
    for ax in axes:
        ax.set_ylim(-lim, lim)
    axes[0].set_ylabel("Δα, mean rank in bin (centered)", fontsize=FL, labelpad=1)
    axes[1].set_yticklabels([])
    fig.text((left + aw + gap / 2) / W, 0.325 / H, "EVEE testis expression level, rank\n(right: residual rank given covariates)", fontsize=FL, linespacing=1.05, ha="center",
             va="bottom")
    h = [Line2D([], [], marker=CAN[1], color=CAN[0], ms=2.2, mew=0, lw=1, label="protein-feature gate"),
         Line2D([], [], marker=FREE[1], color=FREE[0], ms=2.6, mew=0, lw=1, label="per-gene gate"),
         Line2D([], [], color=SG[0], lw=0.8, ls=(0, (2.5, 1.5)), label="$S_g$")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.55, 0.115 / H), ncol=3, fontsize=5.5, handletextpad=0.3,
               columnspacing=0.8, frameon=False, borderpad=0, handlelength=1.6)
    return fig


# ------------------------------------------------------------------------------------------------ c
def panel_gate_vs_s() -> plt.Figure:
    g = pd.read_csv(F2 / "gate_vs_S.csv")
    H = 2.3
    left, bottom, top, gap, vg = 0.5, 0.56, 0.2, 0.1, 0.1
    a = (H - bottom - top - vg) / 2
    W = left + 2 * a + gap + 0.14
    fig = plt.figure(figsize=(W, H))
    pos = {}
    for i, coh in enumerate(("liv_hmouse", "test_hmouse")):
        for j, gate in enumerate(("can5", "free5")):
            pos[(coh, gate)] = fig.add_axes([(left + j * (a + gap)) / W, (bottom + (1 - i) * (a + vg)) / H, a / W, a / H])
    for (coh, gate), ax in pos.items():
        d = g[(g.cohort == coh) & g.count_supported]
        rs = (d[f"S_{gate}"].rank() / len(d)).to_numpy()
        rd = (d[f"D_{gate}"].rank() / len(d)).to_numpy()
        sty = FREE if gate == "free5" else CAN
        ax.scatter(rs, rd, s=0.6, color=sty[0], alpha=0.25, lw=0, rasterized=True)
        rho = spearmanr(d[f"S_{gate}"], d[f"D_{gate}"])[0]
        ax.text(0.04, 0.96, f"ρ = {rho:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=5.8, color=cm.INK,
                bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.set_yticks([0, 0.5, 1])
        ax.set_aspect("equal")
        _style(ax)
        if gate == "free5":
            ax.set_yticklabels([])
        else:
            ax.set_yticklabels(["0", "0.5", "1"])
            ax.set_ylabel("liver" if coh == "liv_hmouse" else "testis", fontsize=FL, labelpad=1)
        if coh == "liv_hmouse":
            ax.set_xticklabels([])
            ax.set_title("per-gene gate" if gate == "free5" else "protein-feature gate", fontsize=FL, pad=2)
        else:
            ax.set_xticklabels(["0", "0.5", "1"])
    fig.text(0.06 / W, (bottom + a + vg / 2) / H, "Δα, rank", rotation=90, fontsize=FL, ha="left", va="center")
    fig.text((left + a + gap / 2) / W, 0.2 / H, "$S_g$ of the same fit, rank\n(genes with count evidence)",
             fontsize=FL, ha="center", va="bottom", linespacing=1.05)
    return fig


# ------------------------------------------------------------------------------------------------ d
C42_ROWS = [("SC", "all", "spermatocyte peak, 1,103"), ("SC", "count_supported", "with count evidence"),
            ("SC_strict", "all", "strict peak, 166"), ("SC_strict", "count_supported", "with count evidence")]
C43_ROWS = [("liv_4sp", "prog63_noMT", "program, 63"), ("liv_4sp", "core15", "core genes, 15"),
            ("liv_hmouse", "prog63_noMT", "program, 63"), ("liv_hmouse", "core15", "core genes, 15")]
C43_Y = [0.75, 1.75, 3.5, 4.5]  # group headers (cohort) at 0 and 2.75


def panel_named() -> plt.Figure:
    t = pd.read_csv(Z3 / "compare_y3.csv")
    W, H = 2.6, 2.3
    fig = plt.figure(figsize=(W, H))
    left, bottom, top, gap = 1.28, 0.56, 0.2, 0.26
    ah_total = H - bottom - top
    ah1 = (ah_total - gap) * 4.0 / 9.6  # row slots: 4 above, 5.6 below (two cohort headers)
    ah2 = ah_total - gap - ah1
    aw = W - left - 0.1
    ax1 = fig.add_axes([left / W, (bottom + ah2 + gap) / H, aw / W, ah1 / H])
    ax2 = fig.add_axes([left / W, bottom / H, aw / W, ah2 / H])

    def get(item, coh, line, group, value, subset):
        return t[(t.item == item) & (t.cohort == coh) & (t.line == line) & (t.group == group) & (t.value == value)
                 & (t.subset == subset)].iloc[0]

    for k, (grp, sub, _) in enumerate(C42_ROWS):
        for off, line, val, sty in ((-0.26, "can5", "D", CAN), (0.0, "free5", "D", FREE), (0.26, "free5", "S", SG)):
            r = get("C42", "test_hmouse", line, grp, val, sub)
            _ci(ax1, k + off, r.excess_d, r.ci_lo, r.ci_hi, sty, size=3.0, lw=0.9)
    for k, (coh, grp, _) in enumerate(C43_ROWS):
        y = C43_Y[k]
        for off, line, val, sty in ((-0.26, "can5", "D_raw", CAN), (0.0, "free5", "D_raw", FREE), (0.26, "free5", "S", SG)):
            r = get("C43", coh, line, grp, val, "all")
            _ci(ax2, y + off, r.excess_d, r.ci_lo, r.ci_hi, sty, size=3.0, lw=0.9)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([lab for _, _, lab in C42_ROWS], fontsize=FS)
    for i, tk in enumerate(ax1.get_yticklabels()):
        tk.set_color(cm.MUTED if i % 2 else cm.INK)
    ax1.set_ylim(3.5, -0.5)
    ax1.set_xlim(0, 1)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xticklabels(["0", "0.5", "1"])
    ax1.set_title("testis, spermatocyte genes", fontsize=FL, pad=2)
    y2 = [0.0, *C43_Y[:2], 2.75, *C43_Y[2:]]
    labs2 = ["Liver 4 species"] + [lab for _, _, lab in C43_ROWS[:2]] + ["Liver Human+Mouse"] + [lab for _, _, lab in C43_ROWS[2:]]
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labs2, fontsize=FS)
    for i, tk in enumerate(ax2.get_yticklabels()):
        if i in (0, 3):
            tk.set_color(cm.MUTED)
            ax2.yaxis.get_major_ticks()[i].tick1line.set_visible(False)
    ax2.set_ylim(y2[-1] + 0.6, -0.5)
    ax2.set_xlim(-4.9, 0.3)
    ax2.set_xticks([-4, -3, -2, -1, 0])
    ax2.set_xticklabels(["−4", "−3", "−2", "−1", "0"])
    ax2.set_title("hepatocyte catabolic genes", fontsize=FL, pad=2)
    for ax in (ax1, ax2):
        ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
        _style(ax)
    fig.text(0.5, 0.205 / H, "matched excess, SD units (95% CI)", fontsize=FL, ha="center", va="bottom")
    h = [Line2D([], [], marker=CAN[1], color=CAN[0], ls="none", ms=2.6, label="protein-feature gate"),
         Line2D([], [], marker=FREE[1], color=FREE[0], ls="none", ms=3.0, label="per-gene gate"),
         Line2D([], [], marker=SG[1], color=SG[0], ls="none", ms=3.0, label="$S_g$")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=5.5, handletextpad=0.15,
               columnspacing=0.7, frameon=False, borderpad=0)
    return fig


# ------------------------------------------------------------------------------------------------ e
X5_COH = [("liv_hmouse", "Liver Human+Mouse"), ("liv_4sp", "Liver 4 species"), ("test_hmouse", "Testis Human+Mouse")]
X5_PANELS = [("dNdS_vs_D", "Δα vs dN/dS"), ("pi_vs_D", "Δα vs protein prior π")]


def panel_axes() -> plt.Figure:
    x = pd.read_csv(Z3 / "compare_x5.csv")
    W, H = 2.15, 2.3
    fig = plt.figure(figsize=(W, H))
    left, bottom, top, gap = 1.1, 0.56, 0.2, 0.3
    ah = (H - bottom - top - gap) / 2
    aw = W - left - 0.13
    axs = [fig.add_axes([left / W, (bottom + ah + gap) / H, aw / W, ah / H]), fig.add_axes([left / W, bottom / H, aw / W, ah / H])]
    for ax, (pan, title) in zip(axs, X5_PANELS):
        for k, (coh, _) in enumerate(X5_COH):
            for off, line, sty in ((-0.16, "can5", CAN), (0.16, "free5", FREE)):
                r = x[(x.cohort == coh) & (x.line == line) & (x.panel == pan)].iloc[0]
                ms = 3.0 if sty[1] == "o" else 2.6
                ax.plot([r.rho], [k + off], marker=sty[1], color=sty[0], ms=ms, mew=0.8, ls="none", zorder=3)
                ax.plot([r.rho_count_evidence], [k + off], marker=sty[1], color=sty[0], mfc="white", ms=ms, mew=0.8,
                        ls="none", zorder=3)
                ax.plot([r.rho, r.rho_count_evidence], [k + off, k + off], color=sty[0], lw=0.5, zorder=2)
        ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=1)
        ax.set_xlim(-0.3, 0.3)
        ax.set_xticks([-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3])
        ax.set_xticklabels(["−0.3", "", "", "0", "", "", "0.3"])
        ax.set_yticks(range(3))
        ax.set_yticklabels([lab for _, lab in X5_COH], fontsize=FS)
        ax.set_ylim(2.5, -0.5)
        ax.set_title(title, fontsize=FL, pad=2)
        _style(ax)
    fig.text((left + aw / 2 - 0.15) / W, 0.205 / H, "Spearman ρ across genes", fontsize=FL, ha="center", va="bottom")
    h = [Line2D([], [], marker="o", color=cm.INK, ls="none", ms=3.0, label="all genes"),
         Line2D([], [], marker="o", color=cm.INK, mfc="white", ls="none", ms=3.0, label="count evidence")]
    fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, fontsize=5.5, handletextpad=0.1,
               columnspacing=0.6, frameon=False, borderpad=0)
    return fig


# ------------------------------------------------------------------------------------------------ main figure, panels d and e
# The two panels of the main validation figure (fig:validation d, e), drawn small. Type in pt at print size: block
# headings (bold) and axis titles; row labels, tick labels and the key; notes.
M_TITLE, M_LABEL, M_TICK, M_NOTE = 6.5, 6.5, 6.0, 5.5
M_ORDER = (SG, CAN, FREE)  # marks of a row from top to bottom: the order of figures/style.py (count evidence, then the two gates)
M_NAME = ("count evidence $S_g$", "Δα, protein-feature gate", "Δα, per-gene gate")  # their names in the key, which stands in panel d for both panels
# bp: the two pages have one height and their lowest axes one bottom, so that the x axes of the row lie on one line; the
# widths are on the 0.24 bp grid of the layout tool and fill the text width with the gap between the panels.
M_H, M_BOTTOM, M_TOP = 123.12, 20.0, 9.5
M_WIDTH = (264.0, 215.04)
M_ZERO = dict(color=cm.RULE, lw=0.6, zorder=1)  # the zero line of both panels, behind the marks
M_AXIS_TITLE = 3.8  # bp from the lower edge of the page to the baseline of the axis titles
# The four independent references: hypothesis of the pre-specified plan -> block and row of the panel.
REFERENCES = [("Cell-state pattern", [("H3", "Liver Human+Mouse"), ("H4", "Testis Human+Mouse")]),
              ("Expression level", [("H1", "Liver Human+Mouse"), ("H2", "Testis Human+Mouse")])]
# What the paper prints for the twelve estimates (three decimals) and the status of the pre-specified test.
PRINTED = {"S": {"H3": 0.148, "H4": 0.116, "H1": 0.041, "H2": 0.014},
           "can": {"H3": 0.110, "H4": 0.015, "H1": 0.013, "H2": -0.230},
           "free": {"H3": 0.145, "H4": 0.123, "H1": 0.037, "H2": 0.029}}
PRINTED_STATUS = {"H3": "V", "H4": "N", "H1": "N", "H2": "R"}
STATUS_WORD = {"V": "supported", "N": "null", "R": "reversed"}  # the status of the table of results, as a word on the panel
READOUT_ROWS = {"S": ("all_S_ema", "frozen"), "can": ("all", "frozen"), "free": ("all", "free5")}  # (universe, line) of compare_hyp.csv


def readout_estimates() -> pd.DataFrame:
    """The twelve estimates of ``panel_readouts``, read from ``results/analysis/Z3/compare_hyp.csv`` (index: readout,
    hypothesis; columns ``rho_p``, ``ci_lo``, ``ci_hi``, ``status_rule``). Nothing is computed. Stops if an estimate or
    a status differs from what the paper prints (``PRINTED``, ``PRINTED_STATUS``)."""
    T = pd.read_csv(Z3 / "compare_hyp.csv")
    out = []
    for key, (universe, line) in READOUT_ROWS.items():
        for h in ("H1", "H2", "H3", "H4"):
            r = T[(T.hyp == h) & (T.universe == universe) & (T.line == line)]
            assert len(r) == 1, f"compare_hyp.csv: {len(r)} rows for {h}, {universe}, {line}"
            r = r.iloc[0]
            assert abs(r.rho_p - PRINTED[key][h]) < 0.00051, f"{key} {h}: {r.rho_p:.4f} in compare_hyp.csv, the paper prints {PRINTED[key][h]}"
            if key == "can":
                assert r.status_rule == PRINTED_STATUS[h], f"{h}: status {r.status_rule} in compare_hyp.csv, the paper prints {PRINTED_STATUS[h]}"
            out.append(dict(readout=key, hyp=h, rho_p=r.rho_p, ci_lo=r.ci_lo, ci_hi=r.ci_hi, status_rule=r.status_rule))
    return pd.DataFrame(out).set_index(["readout", "hyp"])


def panel_readouts() -> plt.Figure:
    """Main validation figure, panel d (``main_validation/readouts_vs_references``, 264 x 123 bp): the three readouts
    against the four independent references.

    What it draws. x: the partial rank correlation with independent data given the covariates, with its 95%
    block-bootstrap interval; dotted line and light band: the bar of 0.10 set in advance and the side that clears it.
    Two blocks of two rows: cell-state pattern (liver, testis) and expression level (liver, testis). In each row three
    marks: the count evidence S_g (square), Δα with the protein-feature gate as pre-specified, ten seeds (circle), and
    Δα with the per-gene gate, five seeds, evaluated afterward (hexagon); the key of the three marks stands in the
    upper left corner of the plot and serves panel e as well. The column at the right, headed "pre-specified test",
    gives the status of that test as a word (supported, null, reversed; ``status_rule`` V, N, R of the table), which
    exists for the protein-feature gate only. The per-gene gate and S_g carry no status.

    Reads ``results/analysis/Z3/compare_hyp.csv`` and nothing else (columns ``hyp``, ``universe``, ``line``, ``rho_p``,
    ``ci_lo``, ``ci_hi``, ``status_rule``): S_g from ``universe == all_S_ema, line == frozen``; the protein-feature
    gate from ``universe == all, line == frozen``; the per-gene gate from ``universe == all, line == free5``; H3 and H4
    are the pattern references, H1 and H2 the level references. No value is computed; ``readout_estimates`` checks the
    twelve estimates and the four status letters against the values the paper prints. The panel stops if an interval
    reaches into the corner that holds the key.

        python -m analysis.tasks.F2_per_gene.panels --panel main_validation/readouts_vs_references     # in chimera_env, from the repository root
    """
    E = readout_estimates()
    W, H = M_WIDTH[0], M_H
    left, right, bottom, top = 72.0, 40.0, M_BOTTOM, M_TOP
    aw, ah = W - left - right, H - bottom - top
    fig = plt.figure(figsize=(cm.style.inch(W), cm.style.inch(H)))
    ax = fig.add_axes([left / W, bottom / H, aw / W, ah / H])
    xlim = (-0.275, 0.2)
    ys, heads, y = {}, [], 0.0
    for block, rows in REFERENCES:  # a heading line, then the rows of the block
        heads.append((y, block))
        y += 0.9
        for h, lab in rows:
            ys[h] = (y, lab)
            y += 1.1
        y += 0.15
    ylim = (y - 0.75, -0.5)
    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)
    ax.axvspan(0.10, xlim[1], color="#F1F1F1", lw=0, zorder=0)
    ax.axvline(0.10, color=cm.MUTED, lw=0.6, ls=(0, (1, 1.5)), zorder=1)
    ax.axvline(0, **M_ZERO)
    tr = ax.get_yaxis_transform()
    for yy, block in heads:
        ax.text(-(left - 2.0) / aw, yy, block, transform=tr, ha="left", va="center_baseline", fontsize=M_TITLE, fontweight="bold", color=cm.INK, clip_on=False)
    for h, (yy, lab) in ys.items():
        ax.text(-4.0 / aw, yy, lab, transform=tr, ha="right", va="center_baseline", fontsize=M_TICK, color=cm.INK, clip_on=False)
        for off, key, sty in zip((-0.3, 0.0, 0.3), ("S", "can", "free"), M_ORDER):
            r = E.loc[(key, h)]
            _ci(ax, yy + off, r.rho_p, r.ci_lo, r.ci_hi, sty, size=3.3, lw=1.0)
        ax.text(1.0 + 4.0 / aw, yy, STATUS_WORD[E.loc[("can", h)].status_rule], transform=tr, ha="left", va="center_baseline", fontsize=M_TICK, fontweight="bold", color=CAN[0], clip_on=False)
    # over the four words, what they are the status of
    ax.annotate("pre-specified\ntest", (1.0, 1.0), xycoords="axes fraction", xytext=(4.0, 1.6 - 1.1 * M_NOTE), textcoords="offset points", ha="left", va="baseline",
                fontsize=M_NOTE, color=CAN[0], linespacing=1.1, annotation_clip=False)
    ax.text(0.10, 1.0 + 1.6 / ah, "bar set in advance", transform=ax.get_xaxis_transform(), ha="center", va="baseline", fontsize=M_NOTE, color=cm.MUTED, clip_on=False)
    # The key of the three marks, for this panel and the one beside it: in the white corner left of zero, its three lines
    # centered on the line of the first heading, above the first row.
    step, names = 0.45, []
    for k, (sty, lab) in zip((-1, 0, 1), zip(M_ORDER, M_NAME)):
        ax.plot([7.0 / aw], [k * step], marker=sty[1], ms=3.3, color=sty[0], mec=sty[0], mew=0.8, ls="none", transform=tr, zorder=4, clip_on=False)
        names.append(ax.text(11.5 / aw, k * step, lab, transform=tr, ha="left", va="center_baseline", fontsize=M_TICK, color=cm.INK))
    fig.canvas.draw()
    corner = ax.transData.inverted().transform((max(t.get_window_extent().x1 for t in names) + 2.0 * fig.dpi / 72, 0))[0]  # right end of the key, with 2 bp of air
    reach = min(float(E.loc[(key, h)].ci_lo) for key in ("S", "can", "free") for h, (yy, _) in ys.items() if yy - 0.3 < step + 0.3)
    assert corner < min(reach, 0.0), "an interval or the zero line reaches into the key"
    ax.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
    ax.set_xticklabels(["−0.2", "−0.1", "0", "+0.1", "+0.2"])
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    _style(ax)
    ax.tick_params(labelsize=M_TICK, length=2.5, width=0.5, pad=1.5)
    fig.text(0.5, M_AXIS_TITLE / H, "partial rank correlation with independent data, $\\rho_p$ (95% CI)", fontsize=M_LABEL, ha="center", va="baseline")
    return fig


def panel_named_compact() -> plt.Figure:
    """Main validation figure, panel e (``main_validation/named_findings``, 215 x 123 bp): the two named findings
    under both gates. The estimates of ``panel_named`` (the panel of the supplementary figure), drawn small, with the
    count evidence first in each row as in panel d.

    What it draws. x: the excess of a gene set over matched background genes in SD units, with its 95% interval; the
    gray line is zero. Top: genes that peak in human spermatocytes, on testis Human+Mouse (all 1,103 and the 166 with
    a strict peak; for each, all genes and those with count evidence): above the prior. Bottom: the hepatocyte
    catabolic program on four-species liver and liver Human+Mouse (63 genes and its 15 core genes): below the prior.
    In each row three marks: the count evidence S_g (square), Δα with the protein-feature gate (circle), Δα with the
    per-gene gate (hexagon); their key stands in panel d, beside this panel.

    Reads ``results/analysis/Z3/compare_y3.csv`` and nothing else (columns ``item``, ``cohort``, ``line``, ``group``,
    ``value``, ``subset``, ``excess_d``, ``ci_lo``, ``ci_hi``): item ``C42`` (spermatocyte genes; value ``D`` for the
    gates, ``S`` for the count evidence) and item ``C43`` (catabolic program; value ``D_raw`` and ``S``); lines
    ``can5`` (protein-feature gate) and ``free5`` (per-gene gate, also the source of S_g, as in ``panel_named``).
    No value is computed. Two estimates are checked against the text of the paper (+0.34 and −1.56).

        python -m analysis.tasks.F2_per_gene.panels --panel main_validation/named_findings              # in chimera_env, from the repository root
    """
    t = pd.read_csv(Z3 / "compare_y3.csv")
    W, H = M_WIDTH[1], M_H
    left, right, bottom, top, gap = 85.0, 6.0, M_BOTTOM, M_TOP, 18.0
    ah_total = H - bottom - top - gap
    ah1 = ah_total * 4.0 / 9.6  # row slots: 4 above, 5.6 below (two cohort lines)
    ah2 = ah_total - ah1
    aw = W - left - right
    fig = plt.figure(figsize=(cm.style.inch(W), cm.style.inch(H)))
    ax1 = fig.add_axes([left / W, (bottom + ah2 + gap) / H, aw / W, ah1 / H])
    ax2 = fig.add_axes([left / W, bottom / H, aw / W, ah2 / H])

    def get(item, coh, line, group, value, subset):
        r = t[(t.item == item) & (t.cohort == coh) & (t.line == line) & (t.group == group) & (t.value == value) & (t.subset == subset)]
        assert len(r) == 1, f"compare_y3.csv: {len(r)} rows for {item}, {coh}, {line}, {group}, {value}, {subset}"
        return r.iloc[0]

    assert round(get("C42", "test_hmouse", "can5", "SC", "D", "all").excess_d, 2) == 0.34 and round(get("C42", "test_hmouse", "free5", "SC", "D", "all").excess_d, 2) == 0.34
    assert round(get("C43", "liv_4sp", "free5", "prog63_noMT", "D_raw", "all").excess_d, 2) == -1.56
    marks = ((-0.27, "free5", "S", SG), (0.0, "can5", None, CAN), (0.27, "free5", None, FREE))  # the order of M_ORDER
    for k, (grp, sub, _) in enumerate(C42_ROWS):
        for off, line, val, sty in marks:
            r = get("C42", "test_hmouse", line, grp, val or "D", sub)
            _ci(ax1, k + off, r.excess_d, r.ci_lo, r.ci_hi, sty, size=2.8, lw=0.9)
    for k, (coh, grp, _) in enumerate(C43_ROWS):
        for off, line, val, sty in marks:
            r = get("C43", coh, line, grp, val or "D_raw", "all")
            _ci(ax2, C43_Y[k] + off, r.excess_d, r.ci_lo, r.ci_hi, sty, size=2.8, lw=0.9)
    ax1.set_yticks(range(4))
    ax1.set_yticklabels([lab for _, _, lab in C42_ROWS], fontsize=M_TICK)
    for i, tk in enumerate(ax1.get_yticklabels()):
        tk.set_color(cm.MUTED if i % 2 else cm.INK)
    ax1.set_ylim(3.5, -0.5)
    ax1.set_xlim(-0.06, 1)
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xticklabels(["0", "+0.5", "+1"])
    y2 = [0.0, *C43_Y[:2], 2.75, *C43_Y[2:]]
    labs2 = ["Liver 4 species"] + [lab for _, _, lab in C43_ROWS[:2]] + ["Liver Human+Mouse"] + [lab for _, _, lab in C43_ROWS[2:]]
    ax2.set_yticks(y2)
    ax2.set_yticklabels(labs2, fontsize=M_TICK)
    for i, tk in enumerate(ax2.get_yticklabels()):
        if i in (0, 3):
            tk.set_color(cm.MUTED)
    ax2.set_ylim(y2[-1] + 0.6, -0.5)
    ax2.set_xlim(-4.9, 0.3)
    ax2.set_xticks([-4, -3, -2, -1, 0])
    ax2.set_xticklabels(["−4", "−3", "−2", "−1", "0"])
    for ax, title in ((ax1, "Testis, spermatocyte-peak genes"), (ax2, "Liver, hepatocyte catabolic genes")):
        ax.axvline(0, **M_ZERO)  # the zero line as in panel d: no left spine, the rows named beside the plot
        _style(ax)
        ax.spines["left"].set_visible(False)
        ax.tick_params(labelsize=M_TICK, length=2.5, width=0.5, pad=1.5)
        ax.tick_params(axis="y", length=0, pad=4.0)
        ax.text(-(left - 2.0) / aw, 1.0 + 1.5 / (ax.get_position().height * H), title, transform=ax.transAxes, ha="left", va="bottom", fontsize=M_TITLE, fontweight="bold", color=cm.INK, clip_on=False)
    fig.text((W - 3.0) / W, M_AXIS_TITLE / H, "excess over matched genes, SD units (95% CI)", fontsize=M_LABEL, ha="right", va="baseline")
    return fig


# panels of the main figure that are saved under their own names: panel -> the function that draws it
MAIN_PANELS = {"main_validation/readouts_vs_references": panel_readouts, "main_validation/named_findings": panel_named_compact}


def main(only=None) -> list:
    """Draw the five panels of the supplementary figure and the two panels of the main figure (``only``: names of
    ``MAIN_PANELS`` to draw just those)."""
    cm.setup()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = []
    if not only:
        for name, fn in [("fig_per_gene_a", panel_forest), ("fig_per_gene_b", panel_testis_level),
                         ("fig_per_gene_c", panel_gate_vs_s), ("fig_per_gene_d", panel_named), ("fig_per_gene_e", panel_axes)]:
            out.append(save(fn(), name))
    for name in only or MAIN_PANELS:  # at their exact size; the preview goes to figures/_work/panels/
        out.append(cm.style.save_panel(MAIN_PANELS[name](), name, exact=True))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", choices=sorted(MAIN_PANELS), metavar="NAME", help="draw only these panels of the main figure (family/name)")
    for p in main(ap.parse_args().panel):
        print(p, cm.check_pdf(p))
