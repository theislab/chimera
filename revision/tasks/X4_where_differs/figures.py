"""X4 (EXPLORATORY): main panel candidate from the X4 CSVs. PDF with embedded fonts (TrueType, type 42).

Writes results/revision/X4/figures/fig_main_where_differs.{pdf,png}.
"""

from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import TwoSlopeNorm  # noqa: E402

from revision.tasks.X4_where_differs import common as CM  # noqa: E402

sys.path.insert(0, str(CM.REPO / "notebooks"))
import _figure_design as FD  # noqa: E402

OUT = CM.OUT
FIG = OUT / "figures"
# one colour and one marker per readout, so identity is never colour alone
STY = {"D": ("#D55E00", "o", r"$\Delta\alpha$"), "S": ("#0072B2", "s", r"margin $S_g$"),
       "B1b": ("#555555", "^", "profile correlation B1b"), "Sdir": ("#009E73", "D", "direct margin, no model"),
       "Bk": ("#555555", "^", "per-type direct level"),
       "Sp": ("#CC79A7", "P", "direct margin: pattern gain"), "Sl": ("#E69F00", "v", "direct margin: level cost")}
GOLD = {"G_H1": "L level", "G_H3": "L pattern", "G_H2": "T level", "G_H4": "T pattern", "G_H4_primates": "T prim."}


def letter(ax, s):
    ax.text(-0.02, 1.08, s, transform=ax.transAxes, fontweight="bold", fontsize=9, va="bottom", ha="right")


def err(ax, x, r, key, **kw):
    c, m, _ = STY[key]
    ax.errorbar(x, r.rho_p, yerr=[[r.rho_p - r.ci_lo], [r.ci_hi - r.rho_p]], fmt=m, color=c, ms=3.6, lw=0.8, capsize=0,
                mec="white", mew=0.3, **kw)


def panel_map(axs):
    x = pd.read_csv(OUT / "map_crosstab.csv")
    x = x[x.axis == "dNdS"]
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=0, vmax=1.5)
    last = None
    for ax, (c, sc, title) in zip(axs, (("liv_hmouse", "residual", r"liver, $\Delta\alpha$ given covariates"),
                                        ("test_hmouse", "raw", r"testis, raw-prior $\Delta\alpha$"),
                                        ("test_hmouse", "residual", r"testis, $\Delta\alpha$ given covariates"))):
        t = x[(x.cohort == c) & (x.scale == sc)].pivot(index="t_constraint", columns="t_D", values="ratio")
        last = ax.imshow(np.log2(t.to_numpy()), cmap="RdBu_r", norm=norm, aspect="equal")
        for i in range(3):
            for j in range(3):
                v = t.to_numpy()[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if abs(np.log2(v)) > 0.9 else "#222222")
        ax.set_xticks(range(3), ["low", "mid", "high"])
        ax.set_yticks(range(3), ["low", "mid", "high"] if ax is axs[0] else ["", "", ""])
        ax.set_title(title, fontsize=7)
        ax.set_xlabel(r"$\Delta\alpha$ tertile")
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].set_ylabel("dN/dS tertile")
    return last


def panel_strata(ax, gold, cohort, title, legend=False):
    r = pd.read_csv(OUT / "right_strata.csv")
    r = r[(r.cohort == cohort) & (r.gold == gold) & (r.axis == "dNdS") & (r.universe == "common")]
    for k, (st, _) in enumerate((("constrained", 0), ("middle", 1), ("relaxed", 2))):
        for off, (key, col) in zip((-0.2, 0, 0.2), (("D", "D_raw"), ("S", "S"), ("B1b", "B1b_raw"))):
            q = r[(r.stratum == st) & (r.score == col)].iloc[0]
            err(ax, k + off, q, key)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(range(3), ["constrained", "middle", "relaxed"])
    ax.set_xlabel("dN/dS tertile")
    ax.set_ylabel(r"$\rho_p$ with gold")
    ax.set_title(title, fontsize=7)
    ax.set_ylim(-0.06, 0.38)
    if legend:
        for key in ("D", "S", "B1b"):
            ax.plot([], [], STY[key][1], color=STY[key][0], ms=3.6, label=STY[key][2])
        ax.legend(loc="upper left", ncol=1, handletextpad=0.2)


def panel_h2h(ax):
    h = pd.read_csv(OUT / "disagree_head_to_head.csv")
    xs = [("D_raw", r"$\Delta\alpha$"), ("S", r"$S_g$"), ("Sdir", "direct\nmargin"), ("B1z_raw", "B1z"), ("B2b", "B2b"),
          ("B1c_raw", "B1c")]
    for off, (c, gold, m, col, lab) in zip((-0.18, 0.18), (("liv_hmouse", "G_H3", "o", "#222222", "liver TS/TMS"),
                                                          ("test_hmouse", "G_H4", "s", "#999999", "testis held-out"))):
        t = h[(h.cohort == c) & (h.gold == gold)].set_index("X")
        y = [t.loc[x, "excess_d"] for x, _ in xs]
        lo = [t.loc[x, "ci_lo"] for x, _ in xs]
        hi = [t.loc[x, "ci_hi"] for x, _ in xs]
        pos = np.arange(len(xs)) + off
        ax.errorbar(y, pos, xerr=[np.array(y) - lo, np.array(hi) - y], fmt=m, color=col, ms=3.6, lw=0.8, label=lab,
                    mec="white", mew=0.3)
    ax.axvline(0, color="#999999", lw=0.5)
    ax.set_yticks(range(len(xs)), [b for _, b in xs])
    ax.invert_yaxis()
    ax.set_xlabel("gold: B1b-high X-low minus B1b-low X-high (SD)\n< 0 sides with X, > 0 sides with B1b")
    ax.legend(loc="lower right", bbox_to_anchor=(1.02, 0.99), ncol=2, handletextpad=0.1, columnspacing=0.6, fontsize=6)
    ax.set_xlim(-0.25, 0.75)


def panel_terms(ax):
    """Direct margin split exactly into pattern gain and level cost, against level and pattern golds."""
    m = pd.read_csv(OUT / "mechanism_rho.csv")
    m = m[m.analysis == "terms_vs_gold"]
    gs = [("liv_hmouse", "G_H1"), ("liv_hmouse", "G_H3"), ("test_hmouse", "G_H2"), ("test_hmouse", "G_H4")]
    for k, (c, g) in enumerate(gs):
        for off, (key, col) in zip((-0.2, 0, 0.2), (("Sp", "Sdir_pattern"), ("Sl", "Sdir_level"), ("S", "S"))):
            err(ax, k + off, m[(m.cohort == c) & (m.gold == g) & (m.score == col)].iloc[0], key)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(range(4), [GOLD[g] for _, g in gs], fontsize=6)
    ax.set_ylabel(r"$\rho_p$ with gold")
    ax.set_ylim(-0.3, 0.42)
    for key in ("Sp", "Sl", "S"):
        ax.plot([], [], STY[key][1], color=STY[key][0], ms=3.6, label=STY[key][2])
    ax.legend(loc="upper left", handletextpad=0.2, fontsize=6)


def panel_adds(ax):
    a = pd.read_csv(OUT / "adds_rho.csv")
    a = a[(a.analysis == "given_panel") & (a.panel == "B1b+B1c+B1z+B2b")]
    gs = ["G_H1", "G_H3", "G_H2", "G_H4", "G_H4_primates"]
    for k, g in enumerate(gs):
        for off, (key, col) in zip((-0.2, 0, 0.2), (("D", "D_raw"), ("S", "S"), ("Sdir", "Sdir"))):
            err(ax, k + off, a[(a.gold == g) & (a.score == col)].iloc[0], key)
    ax.axhline(0, color="#999999", lw=0.5)
    ax.set_xticks(range(len(gs)), [GOLD[g] for g in gs], fontsize=6)
    ax.set_ylabel(r"$\rho_p$ given all direct baselines")
    ax.set_ylim(-0.32, 0.17)
    for key in ("D", "S", "Sdir"):
        ax.plot([], [], STY[key][1], color=STY[key][0], ms=3.6, label=STY[key][2])
    ax.legend(loc="lower right", handletextpad=0.2, fontsize=6)


def panel_drivers(ax):
    d = pd.read_csv(OUT / "drivers_rho.csv")
    d = d[d.given == "C"]
    types = [("liv_hmouse", "Hepatocytes", "Hep."), ("liv_hmouse", "Endothelial", "Endo."), ("liv_hmouse", "fibroblasts", "Fibro."),
             ("liv_hmouse", "Kupffer cells", "Macro."), ("test_hmouse", "SG", "SG"), ("test_hmouse", "SC", "SC"), ("test_hmouse", "SD", "SD"),
             ("test_hmouse", "somatic", "Somatic")]
    for k, (c, t, _) in enumerate(types):
        q = d[(d.cohort == c) & (d.type == t)]
        ck = q[q.score.isin([f"C_{t}", "Ck"])].iloc[0]
        cd = q[q.score.isin([f"Cdir_{t}", "Cdir_k"])].iloc[0]
        bk = q[q.score == "Bk"]
        err(ax, k - 0.2, ck, "S")
        err(ax, k, cd, "Sdir")
        if len(bk):
            err(ax, k + 0.2, bk.iloc[0], "Bk")
    ax.axhline(0, color="#999999", lw=0.5)
    ax.axvline(3.5, ymax=0.62, color="#cccccc", lw=0.5)
    ax.set_xticks(range(len(types)), [x[2] for x in types], fontsize=5.5, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_ylabel(r"$\rho_p$ with per-type level gold")
    ax.set_ylim(-0.07, 0.32)
    for key, lab in (("S", "Chimera contribution"), ("Sdir", "direct contribution, no model"), ("Bk", "per-type direct level")):
        ax.plot([], [], STY[key][1], color=STY[key][0], ms=3.6, label=lab)
    ax.legend(loc="upper right", handletextpad=0.2, fontsize=6)


def vignettes():
    """Rule of rules.py: rank 1 per class and cohort. Raw per-type log2(1 + CPM) in both species beside the readouts."""
    v = pd.read_csv(OUT / "vignettes.csv")
    v = v[v.rank_in_class == 1]
    classes = [("constrained_lowD", "constrained protein, low Δα"), ("relaxed_highD", "relaxed protein, high Δα"),
               ("B1bhigh_Dlow", "B1b high, Δα low"), ("B1blow_Dhigh", "B1b low, Δα high")]
    fig, axes = plt.subplots(2, 4, figsize=(FD.DOUBLE_W, 4.1))
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.15, wspace=0.28, hspace=0.95)
    for i, c in enumerate(("liv_hmouse", "test_hmouse")):
        for j, (k, title) in enumerate(classes):
            ax = axes[i, j]
            r = v[(v.cohort == c) & (v.cls == k)].iloc[0]
            types = [t[len("x_Mouse_"):] for t in v.columns if t.startswith("x_Mouse_") and f"x_Human_{t[8:]}" in v.columns
                     and pd.notna(r[t]) and pd.notna(r[f"x_Human_{t[8:]}"])]
            x = np.arange(len(types))
            ax.bar(x - 0.19, [r[f"x_Human_{t}"] for t in types], 0.36, color=FD.SPECIES_PALETTE["Human"], label="Human")
            ax.bar(x + 0.19, [r[f"x_Mouse_{t}"] for t in types], 0.36, color=FD.SPECIES_PALETTE["Mouse"], label="Mouse")
            ax.set_xticks(x, [t.replace("Hepatocytes", "Hep").replace("Endothelial", "Endo").replace("fibroblasts", "Fib").replace("Cholangio", "Chol")
                              .replace("Other_somatic", "Som").replace("Sertoli", "Sert") for t in types], fontsize=5.5, rotation=35, ha="right",
                          rotation_mode="anchor")
            ax.tick_params(axis="y", labelsize=6)
            b = "n/a" if pd.isna(r.B1b_raw) else f"{r.B1b_raw:+.2f}"
            ax.set_title(f"{r.symbol}\ndN/dS {r.dNdS:.2f}  π {r.prior_raw:.2f}  B1b {b}\nΔα {r.D_raw:+.2f}  $S_g$ {r.S:+.2f}", fontsize=6, loc="left")
            if j == 0:
                ax.set_ylabel(("Liver" if i == 0 else "Testis") + "\nlog2(1 + CPM)", fontsize=6.5)
            if i == 0:
                ax.text(0, 1.5, title, transform=ax.transAxes, fontsize=6.5, fontweight="bold")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, fontsize=6.5, handlelength=0.9, bbox_to_anchor=(0.5, 0.0))
    FD.save_figure(fig, "fig_supp_vignettes", out_dir=FIG, formats=("pdf", "png"), close=True)


def main():
    FD.apply_design()
    FIG.mkdir(parents=True, exist_ok=True)
    vignettes()
    fig = plt.figure(figsize=(FD.DOUBLE_W, 7.6))
    gs = fig.add_gridspec(3, 6, hspace=0.62, wspace=1.15, height_ratios=[0.8, 1, 1], left=0.08, right=0.985, top=0.965, bottom=0.075)
    a1, a2, a3 = fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6])
    im = panel_map([a1, a2, a3])
    cb = fig.colorbar(im, ax=[a1, a2, a3], fraction=0.015, pad=0.015, ticks=[-1, 0, 1])
    cb.ax.set_yticklabels(["0.5", "1", "2"])
    cb.set_label("observed / expected", fontsize=6)
    cb.outline.set_visible(False)
    letter(a1, "a")
    b1, b2 = fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4])
    panel_strata(b1, "G_H3", "liv_hmouse", "liver, TS vs TMS pattern", legend=True)
    panel_strata(b2, "G_H4", "test_hmouse", "testis, 8 held-out species pattern")
    letter(b1, "b")
    c = fig.add_subplot(gs[1, 4:6])
    panel_h2h(c)
    letter(c, "c")
    d = fig.add_subplot(gs[2, 0:2])
    panel_terms(d)
    letter(d, "d")
    e = fig.add_subplot(gs[2, 2:4])
    panel_adds(e)
    letter(e, "e")
    f = fig.add_subplot(gs[2, 4:6])
    panel_drivers(f)
    letter(f, "f")
    FD.save_figure(fig, "fig_main_where_differs", out_dir=FIG, formats=("pdf", "png"), close=True)
    print("figure written", flush=True)


if __name__ == "__main__":
    main()
