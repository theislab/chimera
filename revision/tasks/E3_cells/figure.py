"""E3 candidate panels in the T8.4 / F1 style; composed with F1's compose.py into results/revision/E3/fig_ms
(kept out of the manuscript). Reads E3 result tables only.

    PYTHONPATH=$PWD python revision/tasks/E3_cells/figure.py
    CHIMERA_MANUSCRIPT=$PWD/results/revision/E3/fig_ms PYTHONPATH=$PWD python -m revision.tasks.F1_figures.compose --spec revision/tasks/E3_cells/fig_spec.py --all
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("CHIMERA_MANUSCRIPT", str(ROOT / "results/revision/E3/fig_ms"))
(Path(os.environ["CHIMERA_MANUSCRIPT"]) / "figures" / "revision").mkdir(parents=True, exist_ok=True)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.tasks.E3_cells import rules as R  # noqa: E402
from revision.tasks.H1_harmonise import rules as RU  # noqa: E402
from revision.tasks.T8_4_figures import common as cm  # noqa: E402

cm.PREVIEW = ROOT / "results/revision/E3/fig_preview"
cm.PREVIEW.mkdir(parents=True, exist_ok=True)
E = R.OUT
REG = RU.cohorts()
COMP = ["specialised", "stromal", "vascular", "immune", "broad"]
COMP_LAB = {"specialised": "specialised cells", "stromal": "stromal", "vascular": "vascular", "immune": "immune", "broad": "no home type\n(broadly used)"}
COMP_COL = {"specialised": "#D55E00", "stromal": "#009E73", "vascular": "#0072B2", "immune": "#CC79A7", "broad": "#777777"}
FREE = cm.STY["D"][0]
B1C = cm.STY["B1b"][0]
LVL = "#999999"


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvline(0, color=cm.RULE, lw=0.6, zorder=0)


def excess(universe="count_supported"):
    e = pd.read_csv(E / "excess.csv")
    return e[e.universe == universe]


def panel_compartments():
    e = excess()
    e = e[e.level == "compartment"]
    cols = [("D_free", "Δα, per-gene read-out", FREE, "o"), ("D", "Δα, canonical (main model)", FREE, "o"),
            ("B1", "profile correlation (B1)", B1C, "^"), ("DL", "level similarity at\nthe home type (direct)", LVL, "v")]
    fig, axes = plt.subplots(1, 4, figsize=(6.6, 2.15), sharey=True)
    rng = np.random.default_rng(0)
    for ax, (sc, title, col, mk) in zip(axes, cols):
        for i, c in enumerate(COMP):
            x = e[(e.group == c) & (e.score == sc)].dropna(subset=["excess_d"])
            y = len(COMP) - 1 - i + rng.uniform(-0.18, 0.18, len(x))
            sig = (x.p < 0.01).to_numpy()
            ax.scatter(x.excess_d[sig], y[sig], s=9, color=col, marker=mk, lw=0, alpha=0.85)
            ax.scatter(x.excess_d[~sig], y[~sig], s=9, facecolor="none", edgecolor=col, marker=mk, lw=0.6, alpha=0.85)
            med = x.excess_d.median()
            ax.plot([med, med], [len(COMP) - 1 - i - 0.32, len(COMP) - 1 - i + 0.32], color=cm.INK, lw=1.2)
            if sc in ("D_free", "DL"):
                npos, n = int((x.excess_d > 0).sum()), len(x)
                ax.text(1.02, len(COMP) - 1 - i, f"{npos}/{n} > 0", transform=ax.get_yaxis_transform(), fontsize=5.5, va="center", color=cm.MUTED)
        ax.set_title(title, loc="left", fontsize=6.5)
        ax.set_xlabel("excess over matched genes (SD)", fontsize=6)
        ax.set_xlim(-0.65, 1.25)
        _clean(ax)
    axes[0].set_yticks(range(len(COMP)), [COMP_LAB[c] for c in COMP[::-1]])
    fig.tight_layout(w_pad=1.6)
    return cm.save(fig, "e3a_compartments")


def panel_pattern_vs_level():
    e = excess()
    e = e[(e.level == "subgroup") & (e.group != "broad")]
    th = list(R.THREAD.values()) + list(R.SENSITIVITY.values())
    t = e[e.cohort.isin(th)].pivot_table(index=["cohort", "group"], columns="score", values="excess_d").dropna(subset=["D_free", "DL"]).reset_index()
    comp = {}
    for c in th:
        for k, (cp, sg) in R.type_map(REG[c]["tissue"]).items():
            comp[(c, sg)] = cp
    t["comp"] = [comp.get((c, g), "immune") for c, g in zip(t.cohort, t.group)]
    fig, ax = plt.subplots(figsize=(2.9, 2.6))
    for cp in COMP[:-1]:
        s = t[t.comp == cp]
        ax.scatter(-s.DL, s.D_free, s=12, color=COMP_COL[cp], lw=0, label=COMP_LAB[cp], alpha=0.9)
    lab = {("liv_hmouse", "endothelial"): "liver sinusoidal\nendothelium", ("liv_hmouse", "hepatocyte"): "hepatocyte",
           ("kidney", "thick ascending limb"): "kidney TAL", ("kidney", "thin limb"): "kidney thin limb",
           ("test_hmouse", "germline: spermatid"): "spermatid", ("vz_hmouse", "outflow: trabecular meshwork"): "trabecular\nmeshwork",
           ("brain_m1_hmouse", "excitatory neuron"): "excitatory\nneuron", ("islet", "endocrine: beta"): "beta cell",
           ("brain_m1_hmouse", "microglia"): "microglia", ("muscle_ageing", "muscle stem cell"): "muscle stem cell"}
    for (c, g), txt in lab.items():
        r = t[(t.cohort == c) & (t.group == g)]
        if len(r):
            off = {"trabecular\nmeshwork": (4, 4), "excitatory\nneuron": (-38, -14), "kidney thin limb": (4, 2), "muscle stem cell": (4, -5), "spermatid": (6, -9)}.get(txt, (3, 2))
            ax.annotate(txt, (-r.DL.iloc[0], r.D_free.iloc[0]), fontsize=5.5, xytext=off, textcoords="offset points", color=cm.INK)
    ax.axhline(0, color=cm.RULE, lw=0.6, zorder=0)
    ax.axvline(0, color=cm.RULE, lw=0.6, zorder=0)
    ax.set_xlabel("direct level gap at the home type,\nexcess over matched genes (SD)", fontsize=6)
    ax.set_ylabel("Δα per-gene read-out,\nexcess over matched genes (SD)", fontsize=6)
    ax.set_title("Cell-type programmes keep pattern\nwhere their level moves", loc="left", fontsize=6.5)
    ax.legend(frameon=False, fontsize=5.5, loc="upper left", handletextpad=0.2, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return cm.save(fig, "e3b_pattern_vs_level")


LINEAGE_ROWS = [
    ("liver", ["endothelial", "stellate/fibroblast", "macrophage", "cholangiocyte", "hepatocyte"]),
    ("eye", ["outflow: trabecular meshwork", "ciliary muscle", "Schwann cell", "melanocyte", "cornea", "outflow: Schlemm canal"]),
    ("cortex", ["excitatory neuron", "inhibitory neuron", "oligodendrocyte", "astrocyte", "microglia"]),
    ("testis", ["germline: spermatocyte", "germline: spermatid", "Sertoli", "germline: spermatogonia"]),
]


def panel_lineages():
    e = excess()
    e = e[e.level == "subgroup"]
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.3), gridspec_kw=dict(wspace=1.55))
    for ax, (tis, groups) in zip(axes, LINEAGE_ROWS):
        cs = [c for c in REG if REG[c]["tissue"] == tis]
        for i, g in enumerate(groups):
            y0 = len(groups) - 1 - i
            for sc, col, mk, dy in (("D_free", FREE, "o", 0.13), ("B1", B1C, "^", -0.13)):
                x = e[(e.group == g) & (e.score == sc) & e.cohort.isin(cs)].excess_d
                ax.scatter(x, np.full(len(x), y0 + dy), s=8, color=col, marker=mk, lw=0, alpha=0.8)
                if len(x):
                    ax.plot([x.median()] * 2, [y0 + dy - 0.12, y0 + dy + 0.12], color=col, lw=1.1)
        ax.set_yticks(range(len(groups)), [g.replace("outflow: ", "").replace("germline: ", "") for g in groups[::-1]], fontsize=6)
        ax.set_title(f"{tis} ({len(cs)} models)", loc="left", fontsize=6.5)
        ax.set_xlabel("excess (SD)", fontsize=6)
        if tis == "testis":
            ax.set_xticks([-0.2, 0.2])
        _clean(ax)
    axes[0].scatter([], [], color=FREE, marker="o", s=8, label="Δα, per-gene read-out")
    axes[0].scatter([], [], color=B1C, marker="^", s=8, label="profile correlation (B1)")
    fig.legend(frameon=False, fontsize=6, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.06))
    fig.subplots_adjust(bottom=0.27)
    return cm.save(fig, "e3c_lineages")


NAMED = [("liv_hmouse", "endothelial", "liver sinusoidal endothelium"), ("vz_full", "outflow: trabecular meshwork", "eye trabecular meshwork"),
         ("brain_m1_hmouse", "inhibitory neuron", "cortex inhibitory neurons"), ("brain_m1_hmouse", "excitatory neuron", "cortex excitatory neurons")]


def panel_named():
    n = pd.read_csv(E / "named_genes.csv")
    fig, axes = plt.subplots(1, 4, figsize=(6.7, 1.9), gridspec_kw=dict(wspace=0.55))
    for ax, (c, g, title) in zip(axes, NAMED):
        s = n[(n.cohort == c) & (n.subgroup == g) & (n.sign == "above") & (n.rep_same_sign == 1.0) & (n.n_peer_cohorts >= 4)].head(7)
        peers = [k for k in REG if REG[k]["tissue"] == REG[c]["tissue"]]
        M = s[[f"z__{k}" for k in peers]].to_numpy(float)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_yticks(range(len(s)), s.symbol, fontsize=6)
        short = {k: REG[k]["label"].split(" ", 1)[1].replace("DLPFC ", "").replace("own table", "own").replace("fascicularis", "fasc.").replace("M. mulatta", "M. mul.") for k in peers}
        ax.set_xticks(range(len(peers)), [short[k] for k in peers], rotation=90, fontsize=5.5)
        ax.set_title(title, loc="left", fontsize=6.5)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cb.set_label("Δα per-gene, z", fontsize=6)
    cb.ax.tick_params(labelsize=5, length=2)
    cb.outline.set_visible(False)
    return cm.save(fig, "e3d_named")


def panel_branch():
    rates = pd.read_csv(E / "departures" / "rates.csv")
    fig, ax0 = plt.subplots(figsize=(2.05, 1.95))
    axes = [ax0]
    names = {"brain_dlpfc": ("Departures grow with branch length\n(cortex, 4 primates, one fit)", {"Human": "human", "Chimpanzee": "chimp.", "Macaque": "macaque", "Marmoset": "marmoset"})}
    for ax, (c, (title, lab)) in zip(axes, names.items()):
        r = rates[rates.cohort == c].groupby(["readout", "species"])[["tested", "departures"]].sum()
        r["rate"] = r.departures / r.tested
        bl = R.BRANCH_MY[c]
        for rd, col, mk, ls in (("per-gene", FREE, "o", "-"), ("canonical", FREE, "o", ":"), ("direct", B1C, "^", "-")):
            if rd not in r.index.get_level_values(0):
                continue
            t = r.loc[rd].reindex(list(lab))
            x = np.array([bl[s] for s in t.index])
            o = np.argsort(x)
            ax.plot(x[o], t.rate.to_numpy()[o] * 100, ls=ls, color=col, marker=mk, ms=3.2, lw=0.8,
                    mfc=col if ls == "-" else "white", label={"per-gene": "Chimera, per-gene", "canonical": "Chimera, canonical", "direct": "direct pseudobulk"}[rd])
        t = r.loc["per-gene"].reindex(list(lab))
        for s in t.index:
            ax.annotate(lab[s], (bl[s], t.rate[s] * 100), fontsize=5.5, xytext=(3, -7 if s in ("Human", "macaca_fascicularis") else 3), textcoords="offset points")
        ax.set_xlabel("terminal branch length (My)", fontsize=6)
        ax.set_ylabel("species-specific departures\n(% of gene × cell type)", fontsize=6)
        ax.set_title(title, loc="left", fontsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=5.5, loc="upper left")
    return cm.save(fig, "e3e_branch")


def panel_mtg():
    m = pd.read_csv(E / "departures" / "mtg_check.csv")
    tests = [("marmoset minus macaque branch vs MTG gap difference", "marmoset vs macaque\nbranch"),
             ("human departures vs MTG human-specific gap", "human branch"),
             ("chimpanzee departures vs MTG human-specific gap (control)", "chimp. branch\n(control)")]
    fig, ax = plt.subplots(figsize=(2.0, 1.95))
    for i, (tt, lab) in enumerate(tests):
        for j, (rd, col, mk) in enumerate((("per-gene", FREE, "o"), ("canonical", FREE, "s"), ("direct", B1C, "^"))):
            r = m[(m.test == tt) & (m.readout == rd)].iloc[0]
            y = len(tests) - 1 - i + 0.18 - j * 0.18
            ax.errorbar(r.rho, y, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt=mk, color=col, ms=3, lw=0.8, capsize=0,
                        mfc=col if rd != "canonical" else "white")
    ax.set_yticks(range(len(tests)), [t[1] for t in tests[::-1]], fontsize=6)
    ax.set_xlabel("ρ with held-out MTG level gaps\n(partial on expression)", fontsize=6)
    ax.set_title("Departures replicate\nin another region and lab", loc="left", fontsize=6.5)
    _clean(ax)
    return cm.save(fig, "e3f_mtg")


def panel_microglia():
    m = pd.read_csv(E / "departures" / "ma2022_genes.csv")
    m = m[(m.readout == "per-gene") & m.gene.isin(["FOXP2", "CACNA1D", "DSCAM", "MELTF", "PKD2L1"])]
    sp = ["Human", "Chimpanzee", "Macaque", "Marmoset"]
    col = {"Human": "#222222", "Chimpanzee": "#56B4E9", "Macaque": "#0072B2", "Marmoset": "#E69F00"}
    fig, ax = plt.subplots(figsize=(2.1, 1.95))
    for i, r in enumerate(m.itertuples()):
        v = dict(x.split(":") for x in r.S_rows.split(";"))
        for j, s in enumerate(sp):
            ax.bar(i + (j - 1.5) * 0.19, float(v[s]), width=0.18, color=col[s], label=s if i == 0 else None)
    ax.set_xticks(range(len(m)), [f"{g}\n{ {'Micro': 'microgl.', 'L3_5_IT': 'L3-5 IT'}.get(t, t)}" for g, t in zip(m.gene, m.cell_type)], fontsize=5, rotation=90)
    ax.axhline(0, color=cm.RULE, lw=0.6)
    ax.set_ylabel("per-cell count margin $S_{gsk}$\nin the named cell type", fontsize=6)
    ax.set_title("Human-lineage genes (Ma et al. 2022):\nthe human row departs", loc="left", fontsize=6.5)
    ax.legend(frameon=False, fontsize=5.5, ncol=2, loc="upper left")
    ax.set_ylim(-0.5, 1.75)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return cm.save(fig, "e3g_human_lineage")


if __name__ == "__main__":
    cm.setup()
    for f in (panel_compartments, panel_pattern_vs_level, panel_lineages, panel_named, panel_branch, panel_mtg, panel_microglia):
        print(f())
