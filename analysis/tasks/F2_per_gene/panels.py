"""F2 per-gene figure: Δα read gene by gene from counts (per-gene gate) beside the protein-feature gate, on the
independent golds. Exploratory (decided after the pre-specified test); every number comes from Z3 outputs or from
``F2_per_gene/data.py``, which recomputes them with Z3's harness and checks them against Z3's tables.

Panels, one PDF each at its exact size (``figures/revision/fig_per_gene_{a..e}.pdf``):
  a  forest: rho_p with 95% block-bootstrap CI on H1-H4 and the 4-species liver estimates, per-gene gate (free5) vs
     protein-feature gate (can5, same seeds), all genes and genes with count evidence; pre-specified value marked;
     paired difference with its CI (Z3 compare_hyp.csv, compare_pairs.csv); status letters only for the frozen
     pre-specified test of the protein-feature gate (Table 1), with their key on the panel; the default gate is drawn first
  b  testis level: Δα rank against EVEE testis level, raw and given the covariates C, binned (data.py; the reversal)
  c  Δα against the count evidence S_g of the same fit, per gene, liver and testis (Z3 consensus)
  d  named findings: spermatocyte-peak genes (C42) and hepatocyte catabolic genes (C43), matched excess under each gate
     (Z3 compare_y3.csv)
  e  separate axis: rho(Δα, dN/dS) and rho(Δα, protein prior) under each gate (Z3 compare_x5.csv)

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
    protein-feature gate (Table 1), in their own column with a key. The per-gene gate gets no status."""
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


def main() -> list:
    cm.setup()
    PREVIEW.mkdir(parents=True, exist_ok=True)
    out = []
    for name, fn in [("fig_per_gene_a", panel_forest), ("fig_per_gene_b", panel_testis_level),
                     ("fig_per_gene_c", panel_gate_vs_s), ("fig_per_gene_d", panel_named), ("fig_per_gene_e", panel_axes)]:
        out.append(save(fn(), name))
    return out


if __name__ == "__main__":
    for p in main():
        print(p, cm.check_pdf(p))
