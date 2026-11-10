"""Y2 (EXPLORATORY): one compact figure. Usage: PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m analysis.tasks.Y2_functional.figures"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.common import gene_ids

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import _figure_design as fd  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results/analysis/Y2"
STY = {"S": ("#2a78d6", "o", "$S_g$"), "D": ("#eb6834", "s", "$\\Delta\\alpha$"), "B1b_raw": ("#1baf7a", "^", "B1b"), "B1c_raw": ("#eda100", "D", "B1c"),
       "B1z_raw": ("#e87ba4", "v", "B1z"), "B2b": ("#008300", "P", "B2b"), "B3a": ("#6250d6", "X", "B3a"), "random": ("#8a8a86", ".", "random")}
INK = "#333333"


def zero(ax, vertical=True):
    (ax.axvline if vertical else ax.axhline)(0, color=INK, lw=0.6, ls=":", zorder=0)


def hz(ax, y, r, s, filled=True, ms=4):
    col, mk, _ = STY[s]
    ax.plot([r.ci_lo, r.ci_hi], [y, y], color=col, lw=1.0, solid_capstyle="round", zorder=2)
    ax.plot(r.est, y, mk, ms=ms, mfc=col if filled else "white", mec=col, mew=0.8, zorder=3)


def panel_a(ax, con):
    order = ["S", "D", "B1b_raw", "B1c_raw", "B1z_raw", "B2b", "B3a", "random"]
    x = con[(con["adj"] == "C_Y2") & (con["mode"] == "system") & (con["stat"] == "Delta_match")]
    for i, s in enumerate(order):
        hz(ax, i - 0.17, x[(x.universe == "un") & (x.score == s)].iloc[0], s, True)
        hz(ax, i + 0.17, x[(x.universe == "ms") & (x.score == s)].iloc[0], s, False, 3.2)
    ax.set_yticks(range(len(order)), [STY[s][2] for s in order])
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel("$\\Delta_{match}$: tissue-matched minus mismatched $\\rho_p$")
    ax.set_title("Phenotype translation, liver and testis\nfilled unmasked, open margin-supported", fontsize=7, loc="left")


def panel_b(ax, m):
    t = [("liver", "Liver"), ("testis", "Testis"), ("bone_marrow", "Bone marrow"), ("kidney", "Kidney")]
    x = m[(m.stat == "diag_minus_offdiag") & (m.universe == "un")]
    for i, (k, _) in enumerate(t):
        for j, s in enumerate(("S", "D", "B1b_raw")):
            hz(ax, i + (j - 1) * 0.24, x[(x.tissue == k) & (x.score == s)].iloc[0], s)
    ax.set_yticks(range(4), [b for _, b in t])
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel("own system minus other systems, $\\rho_p$")
    ax.set_title("Each tissue's readout on four systems", fontsize=7, loc="left")


def panel_c(ax, a):
    adj = ["C", "C+tau", "C_Y2", "C_Y2+class", "C_Y2+ESM64"]
    x = a[(a.universe == "un") & (a.stat == "Delta_match")]
    for j, s in enumerate(("S", "D", "B1b_raw", "B3a")):
        col, mk, lab = STY[s]
        pts = [x[(x["adj"] == k) & (x.score == s)].iloc[0] for k in adj]
        xs = np.arange(len(adj)) + (j - 1.5) * 0.12
        ax.plot(xs, [p.est for p in pts], color=col, lw=0.8, zorder=2)
        for xi, p in zip(xs, pts):
            ax.plot([xi, xi], [p.ci_lo, p.ci_hi], color=col, lw=0.8, solid_capstyle="round", zorder=2)
            ax.plot(xi, p.est, mk, ms=3.6, color=col, zorder=3)
    ax.set_xticks(range(len(adj)), ["C", "+$\\tau$", "$C_{Y2}$", "+cl.", "+ESM"], fontsize=6.5)
    zero(ax, False)
    ax.set_ylabel("$\\Delta_{match}$")
    ax.set_title("What removes it", fontsize=7, loc="left")


def panel_d(ax, r):
    t = [("liver", "Liver"), ("testis", "Testis"), ("bone_marrow", "Bone marrow"), ("kidney", "Kidney")]
    x = r[(r.universe == "un") & (r.outcome == "phylop_promoter") & r.stat.isna()]
    for i, (k, _) in enumerate(t):
        for j, s in enumerate(("S", "D", "B1b_raw")):
            hz(ax, i + (j - 1) * 0.24, x[(x.tissue == k) & (x.score == s)].iloc[0], s)
    ax.set_yticks(range(4), [b for _, b in t])
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel("$\\rho_p$ with promoter phyloP (241 mammals)")
    ax.set_title("Promoter conservation, unmasked", fontsize=7, loc="left")


def panel_e(ax, cal, cal2):
    rows = [("testis", "PDhigh", "y_repro", "Testis, $\\pi$ low, placed high:\nreproductive concordance"),
            ("liver", "PClow", "y_hepmet", "Liver, $\\pi$ high, placed low:\nhepatic-metabolic concordance"),
            ("liver", "PClow", "phylop_utr5", "Liver, $\\pi$ high, placed low:\n5'UTR phyloP"),
            ("testis", "PClow", "phylop_utr3", "Testis, $\\pi$ high, placed low:\n3'UTR phyloP")]
    both = pd.concat([cal.assign(eov=cal["obs"] - cal["null_mean"]), cal2.assign(eov=cal2["obs"] - cal2["null_mean"])], ignore_index=True)
    for i, (t, pre, y, _) in enumerate(rows):
        for j, (tag, s) in enumerate((("S_un", "S"), ("D_ms", "D"), ("B1b_un", "B1b_raw"))):
            g = both[(both.tissue == t) & (both.cls == f"{pre}_{tag}") & (both.outcome == y)]
            if not len(g):
                continue
            g = g.iloc[0]
            yy = i + (j - 1) * 0.24
            col, mk, _l = STY[s]
            ax.plot([g.eov - 1.96 * g.null_sd, g.eov + 1.96 * g.null_sd], [yy, yy], color=col, lw=1.0, solid_capstyle="round")
            ax.plot(g.eov, yy, mk, ms=4, color=col)
    ax.set_yticks(range(len(rows)), [r[3] for r in rows], fontsize=5.8)
    ax.invert_yaxis()
    zero(ax)
    ax.set_xlabel("excess over random $\\pi$-band sets, SD")
    ax.set_title("Discordant gene classes", fontsize=7, loc="left")


def panel_f(ax, lit):
    u = lit.drop_duplicates(["tissue", "human_symbol"])
    v = lit[lit["visible"].astype(float) == 1].drop_duplicates(["tissue", "human_symbol"]).sort_values(["tissue", "human_symbol"])
    for i, (_, r) in enumerate(v.iterrows()):
        for j, (c, s) in enumerate((("pct_S", "S"), ("pct_D", "D"), ("pct_B1b_raw", "B1b_raw"), ("pct_B1c_raw", "B1c_raw"))):
            if s in ("S", "D") and r.status == "masked":
                continue
            col, mk, _ = STY[s]
            ax.plot(r[c], i + (j - 1.5) * 0.18, mk, ms=3.6, color=col)
    ax.axvline(0.25, color=INK, lw=0.6, ls=":")
    ax.set_yticks(range(len(v)), [f"{r.human_symbol}{' (masked)' if r.status == 'masked' else ''}" for _, r in v.iterrows()], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel("percentile in matched genes (low = recovered)")
    ax.set_title("Published differences visible in counts", fontsize=7, loc="left")


def main():
    fd.apply_design()
    con = pd.read_csv(OUT / "pheno_contrast.csv")
    m = pd.read_csv(OUT / "pheno_matrix.csv")
    a = pd.read_csv(OUT / "F1_F2_adjustment.csv")
    r = pd.read_csv(OUT / "regulatory.csv")
    cal = pd.read_csv(OUT / "F7_discordant_calibrated.csv")
    cal2 = pd.read_csv(OUT / "F5_calibration.csv")
    lit = pd.read_csv(OUT / "literature_table.csv")
    fig, ax = plt.subplots(2, 3, figsize=(fd.DOUBLE_W, 5.2), constrained_layout=True)
    panel_a(ax[0, 0], con)
    panel_b(ax[0, 1], m)
    panel_c(ax[0, 2], a)
    panel_d(ax[1, 0], r)
    panel_e(ax[1, 1], cal, cal2)
    panel_f(ax[1, 2], lit)
    for a_, lab in zip(ax.ravel(), "abcdef"):
        a_.text(-0.02, 1.08, lab, transform=a_.transAxes, fontweight="bold", fontsize=8, ha="right", va="bottom")
        a_.tick_params(labelsize=6)
        a_.xaxis.label.set_size(6.5)
        a_.yaxis.label.set_size(6.5)
    h = [plt.Line2D([], [], color=STY[s][0], marker=STY[s][1], ls="", ms=4, label=STY[s][2]) for s in STY]
    fig.legend(handles=h, loc="outside lower center", ncol=8, frameon=False, fontsize=6.5, handletextpad=0.1, columnspacing=0.9)
    d = OUT / "figures"
    d.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"fig_y2_functional.{ext}", dpi=300)


if __name__ == "__main__":
    main()
