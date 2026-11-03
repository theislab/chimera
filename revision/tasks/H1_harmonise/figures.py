"""H1 figures, one PDF per panel in results/revision/H1/panels/, composed by F1_figures/compose.py with specs_h1.py. Supplementary: one figure per new tissue with the first submission's
standard panels, same axes, palette, order and fonts as the liver and testis originals. Main candidate: one reading across six tissues. Scripts read results/revision/H1/ only; no model is loaded.

Panels reuse the originals' forms: notebooks/_alpha_master_builder.py (A1.1-A1.3), _alpha_biology_simpler_builder.py (category boxes with d and q), _pathway_builder.py (Hallmark volcano),
revision/tasks/T8_4_figures/fig4_additions.py (landscape, cell-type drivers), X5_sequence_axis/run.py (axis scatter). Design: notebooks/_figure_design.py.

    PYTHONPATH=<chimera repo> python revision/tasks/H1_harmonise/figures.py [--reference]
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from adjustText import adjust_text  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy.stats import spearmanr, wilcoxon  # noqa: E402

from revision.tasks.H1_harmonise import rules as RU  # noqa: E402
from revision.tasks.H1_harmonise.recurrence import bh  # noqa: E402
from revision.tasks.T8_4_figures import common as cm  # noqa: E402

fd = cm.fd
OUT = RU.OUT
BLUE, GREY, TREND = "#2b6ca3", "#9a9a9a", "#c8501e"  # X5_sequence_axis
UP, DOWN = "#D55E00", "#0072B2"  # fig4_margin_support
THEME_COL = ["#0072B2", "#5B3E80", "#D55E00", "#7F7F7F"]  # fig4c
CT_COL = ["#E69F00", "#56B4E9", "#009E73", "#CC79A7", "#BBBBBB"]  # fig4_celltype_drivers
YLIM_D = (-0.32, 0.32)
TISSUE_LABEL = {"liver": "Liver", "testis": "Testis", "bone marrow": "Bone marrow", "kidney": "Kidney", "islet": "Islet", "eye": "Eye", "cortex": "Cortex", "lung": "Lung",
                "skeletal muscle": "Muscle", "tumour myeloid": "Tumour myeloid"}
FIG_NAME = {"bone marrow": "supp_h1_bone_marrow", "kidney": "supp_h1_kidney", "islet": "supp_h1_islet", "eye": "supp_h1_eye", "cortex": "supp_h1_cortex", "lung": "supp_h1_lung",
            "skeletal muscle": "supp_h1_muscle", "tumour myeloid": "supp_h1_tumour"}
SUPP_PRIMARY = {**RU.PRIMARY, **RU.SENS_EXTRA, "tumour myeloid": "tumour_zilionis"}
MODEL_COL = ["#D55E00", "#0072B2", "#009E73", "#222222", "#CC79A7", "#E69F00", "#56B4E9"]
SET_LABEL = {"heme Metabolism": "Heme Metabolism", "Pperoxisome": "Peroxisome", "TF": "Transcription factors", "TF_homeodomain": "Homeodomain TFs", "TF_C2H2_ZF": "C2H2 zinc-finger TFs", "TF_nuclear_rec": "Nuclear receptors", "protein_kinase": "Protein kinases",
             "receptor": "Receptors", "GPCR": "GPCRs", "ion_channel": "Ion channels", "SLC_transporter": "SLC transporters", "chromatin_reg": "Chromatin regulators", "enzyme": "Enzymes",
             "mitochondrial": "Mitochondrial proteins"}


TYPE_LABEL = {"pt": "proximal tubule", "tal": "thick asc. limb", "dct": "distal conv. tubule", "cnt": "connecting tubule", "pc": "principal cell", "ic": "intercalated cell", "atl": "thin asc. limb",
              "dtl": "thin desc. limb", "pec": "parietal epithelium", "pod": "podocyte", "ec": "endothelium", "fib": "fibroblast", "vsm_p": "smooth muscle", "pape": "papillary epithelium",
              "trabecular meshwork cell": "trabecular meshwork", "ciliary muscle cell": "ciliary muscle", "vascular endothelial cell": "vascular endothelium", "canal endothelial cell": "Schlemm's canal"}


def type_label(t: str) -> str:
    return TYPE_LABEL.get(str(t), str(t))


def genes(c: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / "genes" / f"{c}.parquet")


def named(g: pd.DataFrame) -> pd.DataFrame:
    return g[g.nameable & g.symbol.notna()]


def gene_label(r) -> str:
    """^ alpha above 0.95 in a seed; section sign, pair flagged by the ortholog audit (kept, read with care)."""
    return f"{r.symbol}{'^' if r.saturated_flag else ''}{'§' if bool(r.o1_T2) else ''}"


PANELS = OUT / "panels"


def panel_rc() -> None:
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 5.6, "ytick.labelsize": 5.6, "legend.fontsize": 5.4})


def psave(fig: plt.Figure, name: str) -> None:
    PANELS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PANELS / f"{name}.pdf", format="pdf", bbox_inches="tight", pad_inches=0.015)
    plt.close(fig)


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


# ---------------------------------------------------------------- standard panels
def p_scatter(ax, g):
    fd.density_scatter(ax, g.pi_raw, g.alpha)
    fd.style_alpha_axis(ax, "x")
    ax.set_xlabel(fd.LABEL["prior"])
    fd.style_alpha_axis(ax, "y")
    fd.add_identity_line(ax, lim=(0, 1))
    fd.annotate_stat(ax, f"Spearman $\\rho = {spearmanr(g.pi_raw, g.alpha)[0]:.3f}$", "lower right")


def p_hist(ax, g):
    ax.hist(g.D, bins=fd.DELTA_ALPHA_BINS, color="#bdbdbd", edgecolor="#888888", linewidth=0.4)
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--")
    fd.style_delta_alpha_axis(ax, "x")
    ax.set_xlabel(r"$\Delta\alpha$ (centred)")
    ax.set_ylabel(fd.LABEL["n_genes"])
    ax.text(0.98, 0.97, f"frac $> +0.05$: {(g.D > 0.05).mean():.2f}\nfrac $< -0.05$: {(g.D < -0.05).mean():.2f}\nSD: {g.D.std():.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=5.8, color=cm.INK)


def p_overlay(ax, g):
    ax.hist(g.pi_raw, bins=fd.ALPHA_BINS, color="#bbbbbb", edgecolor="none", label="conservation prior", density=True)
    ax.hist(g.alpha, bins=fd.ALPHA_BINS, color="#222222", histtype="step", linewidth=1.0, label=fd.LABEL["alpha"], density=True)
    fd.style_alpha_axis(ax, "x")
    ax.set_ylabel(fd.LABEL["density"])
    fd.thin_legend(ax, loc="upper left")
    ax.text(0.03, 0.56, f"$n={len(g):,}$ genes\nmean $\\alpha={g.alpha.mean():.3f}$\nstd $={g.alpha.std():.3f}$", transform=ax.transAxes, fontsize=5.6, color="#222222", va="top")


def lineage_named(tissue, g):
    n = named(g).set_index("symbol")
    rows = []
    for (theme, syms), col in zip(RU.LINEAGE[tissue].items(), THEME_COL):
        for s in syms:
            if s in n.index:
                rows.append(dict(symbol=s, theme=theme, color=col, D=n.loc[s, "D"], pi=n.loc[s, "pi_raw"], var_name=n.loc[s, "var_name"] if "var_name" in n else None))
    return pd.DataFrame(rows, columns=["symbol", "theme", "color", "D", "pi", "var_name"]).astype({"D": float, "pi": float})


def top_named(g, k=10):
    n = named(g)
    return pd.concat([n.nlargest(k, "D"), n.nsmallest(k, "D")])


def p_landscape(g, tissue) -> plt.Figure:
    fig = plt.figure(figsize=(4.05, 1.9))
    gs = fig.add_gridspec(2, 2, width_ratios=[6, 1], height_ratios=[1, 6], hspace=0.05, wspace=0.03)
    ax = fig.add_subplot(gs[1, 0])
    top = fig.add_subplot(gs[0, 0], sharex=ax)
    right = fig.add_subplot(gs[1, 1], sharey=ax)
    fd.density_scatter(ax, g.pi_raw, g.D, s=1.6, alpha=0.25, color="#BBBBBB")
    ax.axhline(0, color="#888888", lw=0.5, ls="--", zorder=2)
    nm = lineage_named(tissue, g)
    texts = []
    for theme, s in nm.groupby("theme", sort=False):
        ax.scatter(s.pi, s.D, s=11, color=s.color.iloc[0], edgecolor="white", linewidth=0.3, zorder=4, label=theme)
    sel = nm.reindex(nm.D.abs().sort_values(ascending=False).index).head(18)
    for r in sel.itertuples():
        texts.append(ax.text(r.pi, r.D, r.symbol, fontsize=4.8, ha="center", va="center", color=cm.INK, zorder=5))
    ax.set_xlabel("protein-identity prior π")
    ax.set_ylabel("Δα (centred)")
    ax.set_xlim(0.15, 1.0)
    ax.set_ylim(-0.6, 0.6)
    top.hist(g.pi_raw, bins=60, color="#999999", edgecolor="none")
    right.hist(g.D.clip(-0.6, 0.6), bins=60, color="#999999", edgecolor="none", orientation="horizontal")
    for a, hide in ((top, ("top", "right", "left")), (right, ("top", "right", "bottom"))):
        for sp in hide:
            a.spines[sp].set_visible(False)
    top.set_yticks([])
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    right.set_xticks([])
    right.tick_params(axis="y", left=False, labelleft=False)
    if len(sel):
        adjust_text(texts, x=sel.pi.to_numpy(), y=sel.D.to_numpy(), ax=ax, expand=(1.25, 1.6), force_text=(0.4, 0.7), force_static=(0.3, 0.5), arrowprops=dict(arrowstyle="-", color="#999999", lw=0.3), min_arrow_len=1.0)
    if len(nm):
        ax.legend(loc="lower left", fontsize=5, handletextpad=0.1, borderaxespad=0.1, frameon=False)
    return fig


def p_categories(ax, g, tissue, members):
    cats = members[members.family == "broad"][["set", "hsapiens_gene_id"]]
    d = g.merge(cats, on="hsapiens_gene_id")
    order = d.groupby("set").D.median().sort_values().index.tolist()
    data = [d.loc[d.set == c, "D"].values for c in order]
    ax.boxplot(data, vert=False, positions=range(len(order)), widths=0.65, patch_artist=True, medianprops=dict(color="#0072B2", linewidth=1.2),
               boxprops=dict(facecolor="#dddddd", edgecolor="#222222", linewidth=0.5), whiskerprops=dict(color="#222222", linewidth=0.5), capprops=dict(color="#222222", linewidth=0.5),
               flierprops=dict(marker="o", markersize=1.0, markerfacecolor="#222222", markeredgecolor="none", alpha=0.4))
    ax.axvline(0, color="#888888", linewidth=0.6, linestyle="--", zorder=0)
    ax.set_xlim(-0.20, 0.20)
    ax.set_xticks([-0.20, 0.0, 0.20])
    ax.set_xticklabels([f"{v:.2f}" for v in [-0.20, 0.0, 0.20]])
    ax.set_xlabel(r"$\Delta\alpha$ (centred)")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=5.6)
    sd = g.D.std()
    ps = [wilcoxon(x, alternative="two-sided", zero_method="wilcox")[1] if len(x) > 5 else 1.0 for x in data]
    q = bh(ps)
    for i, c in enumerate(order):
        ax.text(1.02, i, rf"$d\!=\!{np.mean(data[i]) / sd:+.2f}$" + ("  ★" if q[i] < 0.05 else ""), ha="left", va="center", fontsize=5, color="#444444", transform=ax.get_yaxis_transform(), clip_on=False)
        ax.text(-0.04, i - 0.32, f"$n${len(data[i])}", ha="right", va="center", fontsize=4.6, color="#999999", transform=ax.get_yaxis_transform(), clip_on=False)


def clean_term(t, max_len=30):
    s = re.sub(r"^HALLMARK_", "", str(t).split("__")[-1]).replace("_", " ").strip()
    return s if len(s) <= max_len else s[: max_len - 1].rstrip() + "…"


def p_volcano(ax, df, nes_lim=3.2, nlf_max=3.5):
    d = df.copy()
    d["nlf"] = -np.log10(d["FDR q-val"].clip(1e-300, 1.0)).clip(upper=nlf_max)
    d["sig"] = d["FDR q-val"] < 0.05
    ns, sp, sn = d[~d.sig], d[d.sig & (d.NES > 0)], d[d.sig & (d.NES < 0)]
    ax.scatter(ns.NES, ns.nlf, s=12, c="#cccccc", alpha=0.7, linewidths=0, rasterized=True, zorder=1)
    ax.scatter(sp.NES, sp.nlf, s=22, c="#b2182b", alpha=0.9, linewidths=0.3, edgecolor="#222222", zorder=2, label="NES > 0")
    ax.scatter(sn.NES, sn.nlf, s=22, c="#2166ac", alpha=0.9, linewidths=0.3, edgecolor="#222222", zorder=2, label="NES < 0")
    ax.axhline(-np.log10(0.05), color="#888888", linewidth=0.5, linestyle="--", zorder=0)
    ax.axvline(0, color="#444444", linewidth=0.5, zorder=0)
    texts = []
    pts = []
    for sub in (sp, sn):
        for _, r in sub.assign(a=sub.NES.abs()).nlargest(3, "a").iterrows():
            texts.append(ax.text(r.NES, r.nlf, clean_term(r.Term, 22), fontsize=4.7, ha="center", va="bottom", color="#222222"))
            pts.append((r.NES, r.nlf))
    ax.set_xlim(-nes_lim, nes_lim)
    ax.set_ylim(-0.1, nlf_max * 1.3)
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("NES  (Δα-ranked GSEA)")
    ax.set_ylabel(r"$-\log_{10}(\mathrm{FDR}\;q)$")
    if texts:
        adjust_text(texts, x=[p[0] for p in pts], y=[p[1] for p in pts], ax=ax, expand=(1.15, 1.9), force_text=(0.3, 1.0), arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.3))
    fd.thin_legend(ax, loc="lower left")


def p_archetypes(ax, c):
    t = pd.read_csv(OUT / "archetypes" / f"{c}.csv")
    s = pd.read_csv(OUT / "archetypes" / "summary.csv")
    s = s[s.cohort == c]
    k = sorted(t.archetype.unique())
    ax.boxplot([t.D[t.archetype == a] for a in k], positions=k, widths=0.65, patch_artist=True, medianprops=dict(color="#0072B2", linewidth=1.2), boxprops=dict(facecolor="#dddddd", edgecolor="#222222", linewidth=0.5),
               whiskerprops=dict(color="#222222", linewidth=0.5), capprops=dict(color="#222222", linewidth=0.5), flierprops=dict(marker="o", markersize=0.8, markerfacecolor="#222222", markeredgecolor="none", alpha=0.3))
    ax.axhline(0, color="#888888", lw=0.6, ls="--", zorder=0)
    ax.set_ylim(*YLIM_D)
    ax.set_xticks(k)
    ax.set_xticklabels([f"$A_{a + 1}$" for a in k])
    ax.set_xlabel(f"GIN hidden-state archetype ($k={int(s.k.iloc[0])}$)")
    ax.set_ylabel("Δα (centred)")
    fd.annotate_stat(ax, f"silhouette {s.silhouette.iloc[0]:.3f}", "upper left")


def dominant_cells(c):
    f = OUT / "drivers" / f"{c}.parquet"
    if not f.exists():
        return None
    return pd.read_parquet(f)


def p_drivers(ax, c, g, tissue):
    """Fig 4d form: contribution of each cell type to S_g of the genes labelled in the landscape; top-weight types, the rest as other."""
    C = dominant_cells(c)
    nm = lineage_named(tissue, g)
    if C is None:
        ax.text(0.5, 0.5, "forward pass pending", transform=ax.transAxes, ha="center")
        return
    sel = pd.concat([nm.nlargest(7, "D"), nm.nsmallest(7, "D")]).drop_duplicates("symbol").sort_values("D")
    if len(sel) < 8:
        extra = top_named(g, 6)
        extra = extra[~extra.symbol.isin(sel.symbol)].assign(pi=lambda d: d.pi_raw, theme="", color="")[["symbol", "theme", "color", "D", "pi"]]
        sel = pd.concat([sel, extra]).sort_values("D")
    gs = g.set_index("symbol")
    shares = pd.read_csv(OUT / "drivers" / f"{c}__shares.csv").set_index("cell_type")
    tops = shares.sort_values("share_abs_S", ascending=False).index[:4].tolist()
    groups = [(t, [t], col) for t, col in zip(tops, CT_COL)] + [("other cell types", None, CT_COL[-1])]
    for y, r in enumerate(sel.itertuples()):
        vn = gs.index.get_loc(r.symbol)
        row = C.loc[g.index[g.symbol == r.symbol][0]]
        pos = neg = 0.0
        for name, ts, col in groups:
            v = row[[t for t in row.index if t not in tops]].sum() if ts is None else row.reindex(ts).sum()
            ax.barh(y, v, left=pos if v >= 0 else neg, color=col, height=0.72, lw=0)
            pos, neg = (pos + v, neg) if v >= 0 else (pos, neg + v)
        ax.plot([row.sum()], [y], marker="|", ms=5, color=cm.INK, mew=0.9, zorder=4)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(sel)))
    ax.set_yticklabels([f"{r.symbol}  ({r.D:+.2f})" for r in sel.itertuples()], fontsize=5.0)
    ax.tick_params(axis="y", pad=1)
    ax.set_ylim(len(sel) - 0.4, -0.6)
    ax.set_xlabel("contribution to $S_g$, by cell type\n(tick: total $S_g$; Δα in brackets)", fontsize=6)
    w = shares.weight
    hs = [Patch(color=col, label=f"{type_label(name)} ({w.reindex(ts).sum():.2f})" if ts else name) for name, ts, col in groups]
    ax.legend(handles=hs, loc="upper center", bbox_to_anchor=(0.3, -0.36), ncol=2, fontsize=4.7, handlelength=0.9, columnspacing=0.6, frameon=False)


def p_top(ax, g, c, k=12):
    """Top count-supported genes by Delta-alpha, each direction, with the cell type carrying most of the margin."""
    n = named(g)
    C = dominant_cells(c)
    t = pd.concat([n.nlargest(k, "D"), n.nsmallest(k, "D").iloc[::-1]]).drop_duplicates("symbol")
    ys = np.arange(len(t))
    ax.barh(ys, t.D, color=[UP if v > 0 else DOWN for v in t.D], height=0.68, lw=0)
    ax.axvline(0, color=cm.MUTED, lw=0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([gene_label(r) for r in t.itertuples()], fontsize=5.0)
    ax.set_ylim(len(t) - 0.4, -0.6)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xlabel("Δα (centred)")
    if C is not None:
        for y, vn, d in zip(ys, t.index, t.D):
            row = C.loc[vn]
            dom = row.abs().idxmax()
            ax.text(-0.02 if d > 0 else 0.02, y, type_label(dom)[:20], fontsize=4.6, color=cm.MUTED, ha="right" if d > 0 else "left", va="center")


def p_count(ax, g):
    """Delta-alpha against the count evidence on supported genes: the count dependence behind the reading."""
    u = g[g.count_support]
    x = pd.Series(u.S_ema).rank(pct=True).to_numpy()
    fd.density_scatter(ax, x, u.D, s=1.4, alpha=0.2, color=BLUE)
    ax.axhline(0, color="#888888", lw=0.5, ls="--")
    ax.set_ylim(*YLIM_D)
    ax.set_xlim(0, 1)
    ax.set_xlabel("$S_g$ (rank, count-supported genes)")
    ax.set_ylabel("Δα (centred)")
    fd.annotate_stat(ax, f"Spearman $\\rho = {spearmanr(u.S_ema, u.D)[0]:+.2f}$", "upper left")


def p_support(ax, g):
    """Fig 4e form: fraction of tail genes whose Delta-alpha is backed by count evidence of the same sign."""
    un = g.count_support.to_numpy()
    thr = np.median(g.S_ema.abs()[un])
    ok = g.count_support & (np.sign(g.S_ema) == np.sign(g.delta_alpha)) & (g.S_ema.abs() >= thr)
    for k, (sub, col, lab) in enumerate([(g.delta_alpha < -RU.TAIL, DOWN, "Δα < −0.05"), (g.delta_alpha > RU.TAIL, UP, "Δα > +0.05")]):
        f = ok[sub].mean() if sub.any() else 0.0
        ax.barh(k, f, color=col, height=0.6, lw=0)
        ax.text(f + 0.02, k, f"{f:.2f}  (n {int(sub.sum()):,})", va="center", fontsize=5.5, color=cm.INK)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["negative tail", "positive tail"], fontsize=6)
    ax.set_ylim(-0.7, 1.7)
    ax.set_xlim(0, 1.35)
    ax.set_xticks([0, 0.5, 1.0])
    ax.set_xlabel("fraction of tail genes backed by\ncount evidence of the same sign", fontsize=6)


def p_models(ax, tissue):
    """Delta-alpha of every model of one tissue (eye, cortex): the model of the thread first."""
    cs = [c for c, d in RU.cohorts().items() if d["tissue"] == tissue]
    cs = sorted(cs, key=lambda c: (c != RU.PRIMARY[tissue], c))
    for c, col in zip(cs, MODEL_COL):
        ax.hist(genes(c).D, bins=fd.DELTA_ALPHA_BINS, histtype="step", color=col, lw=0.9, density=True,
                label=RU.cohorts()[c]["label"].replace("Eye ", "").replace("Cortex ", "").replace(", own table", " own"))
    fd.style_delta_alpha_axis(ax, "x")
    ax.set_xlabel(r"$\Delta\alpha$ (centred)")
    ax.set_ylabel(fd.LABEL["density"])
    ax.axvline(0, color="#888888", lw=0.6, ls="--")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.0)
    ax.legend(loc="upper left", bbox_to_anchor=(0.56, 1.04), fontsize=4.7, ncol=1, handlelength=1.0, labelspacing=0.15, frameon=False, borderaxespad=0)


def one(draw, size, *args, **kw) -> plt.Figure:
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    clean_axes(ax)
    draw(ax, *args, **kw)
    return fig


def supp_panels(tissue: str, c: str, stem: str) -> None:
    g = genes(c)
    g["var_name"] = g.index
    mem = pd.read_csv(OUT / "sets_membership.csv")
    gsea = pd.read_csv(OUT / "gsea" / f"{c}.csv")
    psave(one(p_scatter, (2.2, 1.2), g), f"{stem}_a")
    psave(one(p_hist, (2.2, 1.2), g), f"{stem}_b")
    psave(one(p_overlay, (2.2, 1.2), g), f"{stem}_c")
    psave(p_landscape(g, tissue), f"{stem}_d")
    psave(one(p_categories, (2.3, 1.9), g, tissue, mem), f"{stem}_e")
    psave(one(p_volcano, (2.0, 1.7), gsea, nes_lim=3.2, nlf_max=3.5), f"{stem}_f")
    psave(one(p_drivers, (2.45, 1.75), c, g, tissue), f"{stem}_g")
    psave(one(p_top, (2.2, 1.7), g, c, 8), f"{stem}_h")
    psave(one(p_archetypes, (2.2, 1.05), c), f"{stem}_i")
    psave(one(p_count, (2.2, 1.05), g), f"{stem}_j")
    if tissue in ("eye", "cortex"):
        psave(one(p_models, (2.2, 1.05), tissue), f"{stem}_k")
    else:
        psave(one(p_support, (2.2, 1.05), g), f"{stem}_k")


# ---------------------------------------------------------------- main candidate
def trend(ax, x, y, bins=14):
    q = pd.qcut(x, bins, duplicates="drop")
    m = pd.DataFrame({"x": x, "y": y}).groupby(q, observed=True).median()
    ax.plot(m.x, m.y, color=TREND, lw=1.1)


def cloud(ax, x, y, sup):
    ax.scatter(x[~sup], y[~sup], s=1.0, alpha=0.16, color=GREY, linewidths=0, rasterized=True)
    ax.scatter(x[sup], y[sup], s=1.0, alpha=0.2, color=BLUE, linewidths=0, rasterized=True)


def main_axis() -> plt.Figure:
    ax_t = pd.read_csv(OUT / "axis.csv")
    fig, axes = plt.subplots(3, len(RU.TISSUE_ORDER), figsize=(fd.DOUBLE_W + 0.15, 2.85), sharey="row", layout="constrained", gridspec_kw=dict(height_ratios=[0.7, 1, 1]))
    for j, t in enumerate(RU.TISSUE_ORDER):
        c = RU.PRIMARY[t]
        g = genes(c)
        r_all = ax_t[(ax_t.cohort == c) & (ax_t.universe == "all")].iloc[0]
        r_sup = ax_t[(ax_t.cohort == c) & (ax_t.universe == "count_support")].iloc[0]
        ax = axes[0, j]
        ax.hist(g.D.clip(-0.4, 0.4), bins=fd.DELTA_ALPHA_BINS, color="#bdbdbd", edgecolor="none", density=True)
        ax.axvline(0, color="#888888", lw=0.5, ls="--")
        ax.set_xlim(-0.3, 0.3)
        ax.set_xticks([-0.2, 0, 0.2])
        ax.set_xlabel("Δα (centred)", fontsize=6)
        ax.set_title(TISSUE_LABEL[t], fontsize=7.5, fontweight="bold", pad=3)
        fd.annotate_stat(ax, f"SD {g.D.std():.3f}\n↑ {(g.D > 0.05).mean():.0%}  ↓ {(g.D < -0.05).mean():.0%}", "upper left")
        for tx in ax.texts:
            tx.set_fontsize(5.2)
        ax = axes[1, j]
        d = g.dropna(subset=["dNdS"])
        d = d[d.dNdS > 0]
        sup = d.count_support.to_numpy()
        cloud(ax, d.pi_raw.to_numpy(), d.D.to_numpy(), sup)
        trend(ax, d.pi_raw.to_numpy(), d.D.to_numpy())
        ax.axhline(0, color="#333333", lw=0.5, ls=":")
        ax.set_xlim(0.3, 1.0)
        ax.set_ylim(*YLIM_D)
        ax.set_xticks([0.4, 0.7, 1.0])
        ax.set_xlabel("protein identity π", fontsize=6)
        ax.text(0.03, 0.03, f"Δα ρ = {r_all.rho_D_pi:+.2f}\n$S_g$ ρ = {r_sup.rho_S_pi:+.2f}", transform=ax.transAxes, fontsize=5.2, va="bottom", color=cm.INK)
        ax = axes[2, j]
        x = np.log10(d.dNdS.clip(lower=1e-3)).to_numpy()
        cloud(ax, x, d.D.to_numpy(), sup)
        trend(ax, x, d.D.to_numpy())
        ax.axhline(0, color="#333333", lw=0.5, ls=":")
        ax.set_xlim(-2.5, 0.3)
        ax.set_ylim(*YLIM_D)
        ax.set_xticks([-2, -1, 0])
        ax.set_xlabel("dN/dS (log10)", fontsize=6)
        ax.text(0.03, 0.03, f"Δα ρ = {r_all.rho_D_dNdS:+.2f}\nπ ρ = {r_all.rho_pi_dNdS:+.2f}", transform=ax.transAxes, fontsize=5.2, va="bottom", color=cm.INK)
    axes[0, 0].set_ylabel("density", fontsize=6)
    axes[1, 0].set_ylabel("Δα (centred)", fontsize=6)
    axes[2, 0].set_ylabel("Δα (centred)", fontsize=6)
    for ax in axes.ravel():
        ax.tick_params(labelsize=5.5, length=2)
    hs = [Line2D([], [], ls="none", marker="o", ms=3, color=BLUE, label="count evidence"), Line2D([], [], ls="none", marker="o", ms=3, color=GREY, label="no count evidence (imputed from features)"),
          Line2D([], [], color=TREND, lw=1.1, label="median trend")]
    fig.legend(handles=hs, loc="outside lower center", ncol=3, fontsize=5.5, frameon=False)
    return fig


def label_set(k: str) -> str:
    fam, s = k.split("|", 1)
    return SET_LABEL.get(s, s) + {"hallmark": "", "class": " (class)", "broad": " (category)"}[fam]


def recurrence_rows(mode: str):
    R = pd.read_csv(OUT / "recurrence.csv")
    tabs = {k: R[R.readout == k].set_index("key") for k in R.readout.unique()}
    D = tabs["D"]
    ts = pd.read_csv(OUT / "tissue_specific.csv")
    ts = ts[(ts.readout == "D") & ts.tissue_specific]
    if mode == "strong":  # recurrent on D, on the count evidence, under both nulls, and matched on cell-type specificity
        keep = D.recurrent & tabs["D_tau"].recurrent & tabs["D_block"].recurrent & tabs["S"].recurrent & tabs["S_block"].recurrent
    else:
        keep = D.recurrent
    rec = D[keep]
    ab = rec[rec.direction == "above"].sort_values("stouffer", ascending=False)
    be = rec[rec.direction != "above"].sort_values("stouffer", ascending=False)
    keys = list(ab.index) + list(be.index)
    spec = [k for k in ts.sort_values("tissue").key if k not in keys]
    zc = [f"z_{t}" for t in RU.TISSUE_ORDER]
    opp = D[(D.n_up >= 1) & (D.n_down >= 1) & (D[zc].abs().max(1) >= 5) & ~D.index.isin(keys + spec) & (D.family == "hallmark")].sort_values("stouffer", ascending=False).index.tolist()
    return tabs, keys, len(ab), spec, opp


def main_recurrence(mode: str = "strong") -> plt.Figure:
    tabs, keys, n_ab, spec, opp = recurrence_rows(mode)
    D, S, B, PI, W = tabs["D"], tabs["S"], tabs["B1"], tabs["pi"], tabs["dNdS"]
    rows = keys + spec + opp
    cols = RU.TISSUE_ORDER
    M = D.loc[rows, [f"z_{t}" for t in cols]].to_numpy(float)
    Q = D.loc[rows, [f"q_{t}" for t in cols]].to_numpy(float)
    rh = 0.112
    w_label, w_col, w_side = 1.78, 0.43, 0.36
    nc = len(RU.TISSUE_ORDER)
    W_in = w_label + nc * w_col + 0.08 + 5 * w_side
    top_in, bot_in = 0.42, 0.04
    H = rh * len(rows) + top_in + bot_in
    fig = plt.figure(figsize=(W_in, H))
    ax = fig.add_axes([w_label / W_in, bot_in / H, nc * w_col / W_in, rh * len(rows) / H])
    vmax = 8.0
    ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if not np.isfinite(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=5, color=cm.MUTED)
                continue
            col = "#ffffff" if abs(v) > 0.62 * vmax else "#222222"
            ax.text(j, i, f"{v:+.1f}" + ("*" if Q[i, j] < 0.05 else ""), ha="center", va="center", fontsize=5, color=col)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([TISSUE_LABEL[t] for t in cols], fontsize=5.6, rotation=30, ha="left", rotation_mode="anchor")
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([label_set(k) for k in rows], fontsize=5.3)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if spec:
        ax.axhline(len(keys) - 0.5, color="#222222", lw=0.8)
    if opp:
        ax.axhline(len(keys) + len(spec) - 0.5, color="#222222", lw=0.8)
    if 0 < n_ab < len(keys):
        ax.axhline(n_ab - 0.5, color="#ffffff", lw=2)
    ax2 = fig.add_axes([(w_label + nc * w_col + 0.08) / W_in, bot_in / H, 5 * w_side / W_in, rh * len(rows) / H], sharey=ax)
    ax2.set_xlim(-0.5, 4.5)
    for j, hd in enumerate(["$q$", "$S_g$", "B1", "π", "dN/dS"]):
        ax2.text(j, -1.0, hd, fontsize=5.6, ha="center", va="bottom", color=cm.INK)
    for i, k in enumerate(rows):
        q = D.q_rec[k]
        ax2.text(0, i, f"{q:.3f}" if q >= 0.001 else "<.001", fontsize=4.9, ha="center", va="center", color=cm.INK if k in keys else cm.MUTED)
        for j, tab in zip((1, 2), (S, B)):
            r = tab.loc[k]
            ax2.text(j, i, f"{int(r.n_up)}↑{int(r.n_down)}↓", fontsize=4.9, ha="center", va="center", color=cm.INK if r.recurrent else cm.MUTED)
        for j, tab in zip((3, 4), (PI, W)):
            r = tab.loc[k]
            ax2.text(j, i, f"{r.mean_z:+.1f}", fontsize=4.9, ha="center", va="center", color=cm.INK if abs(r.mean_z) >= RU.Z_CALL else cm.MUTED)
    ax2.axis("off")
    ax2.set_ylim(ax.get_ylim())
    return fig


def main_lineage() -> plt.Figure:
    L = pd.read_csv(OUT / "lineage_z.csv")
    prim = {RU.PRIMARY[t]: t for t in RU.TISSUE_ORDER}
    L = L[L.cohort.isin(prim) & L.readout.isin(["D", "S"])].copy()
    L["tissue"] = L.cohort.map(prim)
    L["own"] = L.key.str.split("|").str[1]
    L["set"] = L.key.str.split("|").str[2]
    rows = [f"lineage|{t}|{s}" for t in RU.TISSUE_ORDER for s in RU.LINEAGE[t]]
    fig, axes = plt.subplots(1, 2, figsize=(fd.DOUBLE_W, 0.135 * len(rows) + 0.75), layout="constrained", sharey=True)
    for ax, rd, ttl in zip(axes, ["D", "S"], ["Δα", "count evidence $S_g$"]):
        P = L[L.readout == rd].pivot(index="key", columns="tissue", values="z").reindex(index=rows, columns=RU.TISSUE_ORDER)
        M = P.to_numpy(float)
        ax.imshow(M, cmap="RdBu_r", vmin=-8, vmax=8, aspect="auto", interpolation="nearest")
        for i in range(M.shape[0]):
            own = RU.TISSUE_ORDER.index(rows[i].split("|")[1])
            ax.add_patch(Rectangle((own - 0.5, i - 0.5), 1, 1, fill=False, ec="#222222", lw=0.9))
            for j in range(M.shape[1]):
                if np.isfinite(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:+.1f}", ha="center", va="center", fontsize=5, color="#ffffff" if abs(M[i, j]) > 5 else "#222222")
                else:
                    ax.text(j, i, "—", ha="center", va="center", fontsize=5, color=cm.MUTED)
        ax.set_xticks(range(len(RU.TISSUE_ORDER)))
        ax.set_xticklabels([TISSUE_LABEL[t] for t in RU.TISSUE_ORDER], fontsize=6, rotation=30, ha="right")
        ax.tick_params(length=0)
        ax.set_title(ttl, fontsize=7, pad=3)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([f"{r.split('|')[2]} ({TISSUE_LABEL[r.split('|')[1]].lower()})" for r in rows], fontsize=5.6)
    return fig


def main_genes(k_top=4, k_lin=5, k_down=5) -> plt.Figure:
    nr = 2 if len(RU.TISSUE_ORDER) > 6 else 1
    nc = int(np.ceil(len(RU.TISSUE_ORDER) / nr))
    fig, axes = plt.subplots(nr, nc, figsize=(fd.DOUBLE_W + 0.15, 1.95 * nr), layout="constrained", sharex=True)
    for ax, t in zip(np.ravel(axes), RU.TISSUE_ORDER):
        c = RU.PRIMARY[t]
        n = named(genes(c))
        n = n[~n.symbol.str.startswith("MT-")]  # mitochondrially encoded genes are not measured in every tissue and track RNA retention; see the caption
        panel = {s for v in RU.LINEAGE[t].values() for s in v}
        up = n[n.D > 0]
        lin = up[up.symbol.isin(panel)].nlargest(k_lin, "D")
        top = up[~up.symbol.isin(panel)].nlargest(k_top, "D")
        down = n[n.D < 0].nsmallest(k_down, "D").iloc[::-1]
        sel = pd.concat([top, lin, down])
        y = np.r_[np.arange(len(top)), len(top) + 0.5 + np.arange(len(lin)), len(top) + len(lin) + 1 + np.arange(len(down))]
        ax.barh(y, sel.D, color=[UP if v > 0 else DOWN for v in sel.D], height=0.68, lw=0)
        ax.axvline(0, color=cm.MUTED, lw=0.5)
        ax.set_ylim(k_top + k_lin + k_down + 0.6, -0.6)
        ax.set_yticks([])
        ax.set_xlim(-0.6, 0.6)
        ax.set_xticks([-0.4, 0, 0.4])
        ax.tick_params(labelsize=5.5, length=2)
        if t in RU.TISSUE_ORDER[-nc:]:
            ax.set_xlabel("Δα (centred)", fontsize=6)
        ax.set_title(TISSUE_LABEL[t], fontsize=7.5, fontweight="bold", pad=3)
        ax.spines["left"].set_visible(False)
        for yy, r in zip(y, sel.itertuples()):
            ax.text(-0.012 if r.D > 0 else 0.012, yy, gene_label(r), fontsize=5.2, ha="right" if r.D > 0 else "left", va="center", color=cm.INK, fontweight="bold" if r.symbol in panel else "normal")
    return fig


def y1_masked() -> None:
    """Y1's page without its own letters, so that compose labels its six panels in sequence. Y1 writes the page twice
    (``figure.py --slot``): with letters (the page box everyone reads; also copied to the manuscript as
    fig_h1_y1_slot.pdf) and without them, saved with a box that differs only by a margin on the left and bottom. That
    margin is trimmed here so every panel keeps the position the regions in specs_h1.py were read on. No letter is
    covered or hidden: the letters are not drawn."""
    y1 = cm.RES / "Y1" / "figures"
    ref, slot = y1 / "fig_y1_gene_tissue.pdf", y1 / "fig_y1_gene_tissue_slot.pdf"

    def size(p):
        o = subprocess.run([str(cm.POPPLER / "pdfinfo"), str(p)], capture_output=True, text=True).stdout
        w, h = re.search(r"Page size:\s+([\d.]+) x ([\d.]+)", o).groups()
        return float(w), float(h)

    (W, H), (Ws, Hs) = size(ref), size(slot)
    trim = f"{Ws - W:.4f}bp {Hs - H:.4f}bp 0bp 0bp"
    tex = ("\\documentclass{article}\\usepackage[paperwidth=%sbp,paperheight=%sbp,margin=0pt]{geometry}\\usepackage{graphicx}\\pagestyle{empty}\\setlength{\\parindent}{0pt}"
           "\\begin{document}\\noindent\\includegraphics[trim=%s,clip]{%s}\\end{document}" % (W, H, trim, slot))
    with tempfile.TemporaryDirectory() as td:
        (Path(td) / "m.tex").write_text(tex)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "m.tex"], cwd=td, capture_output=True, check=True)
        PANELS.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(Path(td) / "m.pdf", PANELS / "y1_masked.pdf")
    shutil.copyfile(ref, cm.FIG_OUT / "fig_h1_y1_slot.pdf")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", action="store_true", help="also render the standard panels for liver and testis, to compare with the originals")
    a = ap.parse_args()
    cm.PREVIEW = OUT / "preview"
    cm.setup()
    panel_rc()
    for t in RU.NEW_TISSUES:
        supp_panels(t, SUPP_PRIMARY[t], FIG_NAME[t])
    if a.reference:
        for t in ("liver", "testis"):
            supp_panels(t, RU.PRIMARY[t], f"h1_reference_{t.replace(' ', '_')}")
    for name, fn in (("fig_h1_axis", main_axis), ("fig_h1_recurrence", main_recurrence), ("supp_h1_recurrence_all", lambda: main_recurrence("all")), ("supp_h1_lineage", main_lineage), ("fig_h1_genes", main_genes)):
        psave(fn(), name)
    y1_masked()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
