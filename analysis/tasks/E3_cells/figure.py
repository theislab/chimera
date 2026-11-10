"""E3 panels: home compartments, cell lineages and species-specific departures. Seven panels of three figures, saved
through figures/style.py to figures/panels/<figure>/; reads the result tables of results/analysis/E3 only.

Every panel is drawn at its size on the page and saved at that exact size, so the figures compose at scale 1:
supp_cell_compartments (compartments, pattern_vs_level), supp_cell_lineages (lineages, named) and
supp_primate_branches (branch, mtg, human_lineage). The panel compartments is also a panel of the main figure on tissues.

    PYTHONPATH=$PWD python -m analysis.tasks.E3_cells.figure [panel ...]     # every panel, or the named ones, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                         # dry run: panels under <folder>/panels
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from analysis.tasks.E3_cells import rules as R  # noqa: E402
from analysis.tasks.H1_harmonise import rules as RU  # noqa: E402
from figures import style  # noqa: E402

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
E = R.OUT
REG = RU.cohorts()
WIDTH = style.TEXTWIDTH  # bp; a full-width panel
ROW_H = 165  # bp; height of the three panels that share the row of supp_primate_branches
S_TITLE, S_LABEL, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_LABEL, style.SIZE_TICK, style.SIZE_TEXT
INK, MUTED, RULE = style.INK, style.MUTED, style.RULE
NA_CELL = "#F2F2F2"  # heat-map cell without a value
GATE, PER_GENE, PROFILE = style.READOUT["delta_alpha"], style.READOUT["delta_alpha_per_gene"], style.READOUT["profile_correlation"]
LEVEL = style.Mark("#999999", "v", "level similarity at the home type (direct)")  # a direct level measure; not in the house table
# Species-specific departures: Chimera's count margin under each gate, and the direct pseudobulk analogue (keys of rates.csv, mtg_check.csv).
DEPARTURES = [("canonical", style.Mark(GATE.color, GATE.marker, "Chimera, protein-feature gate")),
              ("per-gene", style.Mark(PER_GENE.color, PER_GENE.marker, "Chimera, per-gene gate")),
              ("direct", style.Mark(PROFILE.color, PROFILE.marker, "direct pseudobulk"))]
COMP = ["specialised", "stromal", "vascular", "immune", "broad"]  # compartment keys of the result tables
COMP_LAB = {"specialised": "specialized cells", "stromal": "stromal", "vascular": "vascular", "immune": "immune", "broad": "no home type\n(broadly used)"}
COMP_COL = {"specialised": "#D55E00", "stromal": "#009E73", "vascular": "#0072B2", "immune": "#CC79A7", "broad": "#777777"}
COMP_MK = {"specialised": "o", "stromal": "s", "vascular": "^", "immune": "D"}
SPECIES = ["Human", "Chimpanzee", "Macaque", "Marmoset"]  # the four primates of the prefrontal-cortex fit, as named in the tables
CELL_TYPE = {"Micro": "microglia", "PVALB": "PVALB interneurons", "L3_5_IT": "L3-5 IT neurons"}
LEADER = dict(arrowstyle="-", lw=0.4, color=MUTED, shrinkA=0, shrinkB=1.5)  # line from a label to its point


def save(fig, name: str):
    """One panel at its exact size: figures/panels/<figure>/<name>.pdf."""
    return style.save_panel(fig, name, exact=True)


def signed(v, _=None) -> str:
    """Tick label with one decimal and a true minus sign."""
    return f"{v:.1f}".replace("-", "−")


def cohort_label(k: str) -> str:
    """Name of a cohort under a heading that already names its tissue: "DLPFC Human+Macaque, own ortholog table"."""
    name, tissue = style.cohort_name(REG[k]["label"]), REG[k]["tissue"]
    if name.lower().startswith(tissue + " "):
        name = name[len(tissue) + 1:]
    return {"primates": "3 primates"}.get(name, name).replace(", own table", ",\nown ortholog table")


def _clean(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)


def excess(universe="count_supported"):
    e = pd.read_csv(E / "excess.csv")
    return e[e.universe == universe]


def panel_compartments():
    """Genes by home compartment: excess over matched genes, one mark per cohort (filled, p < 0.01), for four scores."""
    e = excess()
    e = e[e.level == "compartment"]
    # the two gates first, then the direct baselines; a count of cohorts above zero beside every row
    cols = [("D", GATE.name, GATE), ("D_free", PER_GENE.name, PER_GENE), ("B1", PROFILE.name, PROFILE),
            ("DL", "level similarity at the\nhome type (direct)", LEVEL)]
    fig, axes = plt.subplots(1, 4, figsize=(style.inch(WIDTH), style.inch(150)), sharey=True, layout="constrained")
    rng = np.random.default_rng(0)
    for ax, (sc, title, mark) in zip(axes, cols):
        for i, c in enumerate(COMP):
            x = e[(e.group == c) & (e.score == sc)].dropna(subset=["excess_d"])
            y = len(COMP) - 1 - i + rng.uniform(-0.18, 0.18, len(x))
            sig = (x.p < 0.01).to_numpy()
            ax.scatter(x.excess_d[sig], y[sig], s=9, color=mark.color, marker=mark.marker, lw=0, alpha=0.85)
            ax.scatter(x.excess_d[~sig], y[~sig], s=9, facecolor="none", edgecolor=mark.color, marker=mark.marker, lw=0.6, alpha=0.85)
            med = x.excess_d.median()
            ax.plot([med, med], [len(COMP) - 1 - i - 0.32, len(COMP) - 1 - i + 0.32], color=INK, lw=1.2)
            npos, n = int((x.excess_d > 0).sum()), len(x)
            ax.text(1.02, len(COMP) - 1 - i, f"{npos}/{n} > 0", transform=ax.get_yaxis_transform(), fontsize=S_TEXT, va="center", color=MUTED)
        ax.set_title(title, loc="left", fontsize=S_TITLE)
        ax.set_xlim(-0.9, 1.3)  # holds every cohort (lowest -0.84, highest 1.19)
        ax.set_xticks([-0.5, 0, 0.5, 1])
        ax.xaxis.set_major_formatter(FuncFormatter(signed))
        ax.tick_params(labelsize=S_TICK)
        _clean(ax)
    axes[0].set_yticks(range(len(COMP)), [COMP_LAB[c] for c in COMP[::-1]])
    fig.supxlabel("excess over matched genes (SD)", fontsize=S_LABEL)
    return save(fig, "supp_cell_compartments/compartments")


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
    fig, ax = plt.subplots(figsize=(style.inch(300), style.inch(250)), layout="constrained")
    for cp in COMP[:-1]:
        s = t[t.comp == cp]
        ax.scatter(-s.DL, s.D_free, s=12, color=COMP_COL[cp], marker=COMP_MK[cp], lw=0, label=COMP_LAB[cp], alpha=0.9)
    # named subgroups: (cohort, subgroup) -> label, offset of the label from its point (points), horizontal and vertical anchor
    lab = {("liv_hmouse", "endothelial"): ("liver sinusoidal\nendothelium", (8, -2), "left", "center"),
           ("liv_hmouse", "hepatocyte"): ("hepatocyte", (-6, -9), "right", "top"),
           ("kidney", "thick ascending limb"): ("kidney thick\nascending limb", (8, -3), "left", "center"),
           ("kidney", "thin limb"): ("kidney thin limb", (9, 9), "left", "bottom"),
           ("test_hmouse", "germline: spermatid"): ("spermatid", (12, 8), "left", "bottom"),
           ("vz_hmouse", "outflow: trabecular meshwork"): ("trabecular\nmeshwork", (-4, 12), "center", "bottom"),
           ("brain_m1_hmouse", "excitatory neuron"): ("excitatory\nneuron", (-32, 14), "right", "bottom"),
           ("islet", "endocrine: beta"): ("beta cell", (10, 0), "left", "center"),
           ("brain_m1_hmouse", "microglia"): ("microglia", (10, 0), "left", "center"),
           ("muscle_ageing", "muscle stem cell"): ("muscle stem cell", (8, 10), "left", "bottom")}
    for (c, g), (txt, off, ha, va) in lab.items():
        r = t[(t.cohort == c) & (t.group == g)]
        if len(r):
            ax.annotate(txt, (-r.DL.iloc[0], r.D_free.iloc[0]), fontsize=S_TEXT, xytext=off, textcoords="offset points", color=INK, ha=ha, va=va,
                        linespacing=1.0, arrowprops=LEADER)
    ax.axhline(0, color=RULE, lw=0.6, zorder=0)
    ax.axvline(0, color=RULE, lw=0.6, zorder=0)
    ax.set_xlim(-0.52, 1.02)  # room on the right for the label of the right-most named point
    ax.set_xlabel("direct level gap at the home type,\nexcess over matched genes (SD)", fontsize=S_LABEL)
    ax.set_ylabel(f"{PER_GENE.name},\nexcess over matched genes (SD)", fontsize=S_LABEL)
    ax.xaxis.set_major_formatter(FuncFormatter(signed))
    ax.yaxis.set_major_formatter(FuncFormatter(signed))
    ax.tick_params(labelsize=S_TICK)
    ax.legend(frameon=False, fontsize=S_TICK, loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    return save(fig, "supp_cell_compartments/pattern_vs_level")


LINEAGE_ROWS = [  # tissue, its home-type groups (top to bottom), x ticks
    ("liver", ["endothelial", "stellate/fibroblast", "macrophage", "cholangiocyte", "hepatocyte"], [0, 0.5, 1]),
    ("eye", ["outflow: trabecular meshwork", "ciliary muscle", "Schwann cell", "melanocyte", "cornea", "outflow: Schlemm canal"], [0, 0.2, 0.4]),
    ("cortex", ["excitatory neuron", "inhibitory neuron", "oligodendrocyte", "astrocyte", "microglia"], [-0.2, 0, 0.2]),
    ("testis", ["germline: spermatocyte", "germline: spermatid", "Sertoli", "germline: spermatogonia"], [-0.2, 0, 0.2, 0.4]),
]


def panel_lineages():
    """Home-type groups of four tissues: excess over matched genes in every cohort of the tissue, per-gene gate and profile correlation."""
    e = excess()
    e = e[e.level == "subgroup"]
    fig, axes = plt.subplots(1, 4, figsize=(style.inch(WIDTH), style.inch(150)), layout="constrained")
    for ax, (tis, groups, ticks) in zip(axes, LINEAGE_ROWS):
        cs = [c for c in REG if REG[c]["tissue"] == tis]
        for i, g in enumerate(groups):
            y0 = len(groups) - 1 - i
            for sc, mark, dy in (("D_free", PER_GENE, 0.13), ("B1", PROFILE, -0.13)):
                x = e[(e.group == g) & (e.score == sc) & e.cohort.isin(cs)].excess_d
                ax.scatter(x, np.full(len(x), y0 + dy), s=8, color=mark.color, marker=mark.marker, lw=0, alpha=0.8)
                if len(x):
                    ax.plot([x.median()] * 2, [y0 + dy - 0.12, y0 + dy + 0.12], color=mark.color, lw=1.1)
        ax.set_yticks(range(len(groups)), [g.replace("outflow: ", "").replace("germline: ", "") for g in groups[::-1]], fontsize=S_TICK)
        ax.set_title(f"{style.TISSUE[tis].name} ({len(cs)} cohorts)", loc="left", fontsize=S_TITLE)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(FuncFormatter(signed))
        ax.tick_params(axis="x", labelsize=S_TICK)
        _clean(ax)
    fig.supxlabel("excess over matched genes (SD)", fontsize=S_LABEL)
    handles = [Line2D([], [], ls="", marker=m.marker, color=m.color, ms=3.5, label=m.name) for m in (PER_GENE, PROFILE)]
    fig.legend(handles=handles, frameon=False, fontsize=S_TICK, loc="outside upper right", ncol=2, handletextpad=0.3, borderaxespad=0.2)
    return save(fig, "supp_cell_lineages/lineages")


NAMED = [("liv_hmouse", "endothelial", "Liver,\nsinusoidal endothelium"), ("vz_full", "outflow: trabecular meshwork", "Eye,\ntrabecular meshwork"),
         ("brain_m1_hmouse", "inhibitory neuron", "Cortex,\ninhibitory neurons"), ("brain_m1_hmouse", "excitatory neuron", "Cortex,\nexcitatory neurons")]
Z_MAX = 3.0  # color limit of the z maps; cells above it take the end color and print their value


def panel_named():
    """Seven named genes of four cell types: z of the per-gene gate in every cohort of the tissue, value in every cell."""
    n = pd.read_csv(E / "named_genes.csv")
    sets = []
    for c, g, title in NAMED:
        s = n[(n.cohort == c) & (n.subgroup == g) & (n.sign == "above") & (n.rep_same_sign == 1.0) & (n.n_peer_cohorts >= 4)].head(7)
        sets.append((title, s, [k for k in REG if REG[k]["tissue"] == REG[c]["tissue"]]))
    fig, axes = plt.subplots(1, 4, figsize=(style.inch(WIDTH), style.inch(190)), layout="constrained",
                             gridspec_kw=dict(width_ratios=[len(p) for _, _, p in sets]))
    for ax, (title, s, peers) in zip(axes, sets):
        M = s[[f"z__{k}" for k in peers]].to_numpy(float)
        ax.set_facecolor(NA_CELL)
        im = ax.imshow(np.ma.masked_invalid(M), cmap=style.DIVERGING, vmin=-Z_MAX, vmax=Z_MAX, aspect="auto", interpolation="nearest")
        for (i, j), v in np.ndenumerate(M):
            if np.isfinite(v):
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=S_TEXT, color="white" if abs(v) > 0.6 * Z_MAX else INK)
        ax.set_yticks(range(len(s)), s.symbol, fontsize=S_TICK)
        ax.set_xticks(range(len(peers)), [cohort_label(k) for k in peers], rotation=90, fontsize=S_TEXT, linespacing=1.0)
        ax.set_title(title, loc="right", fontsize=S_TITLE, linespacing=1.05)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.016, pad=0.01, extend="max")
    cb.set_label(f"z of {PER_GENE.name}", fontsize=S_LABEL)
    cb.ax.tick_params(labelsize=S_TICK, length=2)
    cb.outline.set_visible(False)
    return save(fig, "supp_cell_lineages/named")


def departures_legend(fig):
    """The three series of the departure panels, above the axes."""
    handles = [Line2D([], [], ls="", marker=m.marker, color=m.color, ms=3.8, label=m.name) for _, m in DEPARTURES]
    fig.legend(handles=handles, frameon=False, fontsize=S_TICK, loc="outside upper left", ncol=1, handletextpad=0.3, borderaxespad=0.2)


def panel_branch():
    """Prefrontal cortex, four primates in one fit: share of gene by cell-type pairs in which a species departs, species in the
    order of their terminal branch length (printed under each name)."""
    rates = pd.read_csv(E / "departures" / "rates.csv")
    r = rates[rates.cohort == "brain_dlpfc"].groupby(["readout", "species"])[["tested", "departures"]].sum()
    r["rate"] = r.departures / r.tested
    bl = R.BRANCH_MY["brain_dlpfc"]
    sp = sorted(SPECIES, key=lambda s: bl[s])  # stable: human and chimpanzee share one branch length
    fig, ax = plt.subplots(figsize=(style.inch(186), style.inch(ROW_H)), layout="constrained")
    for (rd, mark), dx in zip(DEPARTURES, (-0.22, 0.0, 0.22)):
        ax.plot(np.arange(len(sp)) + dx, r.loc[rd].reindex(sp).rate.to_numpy() * 100, color=mark.color, marker=mark.marker, ms=4, lw=0.7)
    ax.set_xticks(range(len(sp)), [f"{s.lower()}\n{bl[s]:g}" for s in sp], fontsize=S_TICK)
    ax.set_xlim(-0.5, len(sp) - 0.5)
    ax.tick_params(axis="y", labelsize=S_TICK)
    ax.set_xlabel("species, terminal branch length (My)", fontsize=S_LABEL)
    ax.set_ylabel("species-specific departures\n(% of gene × cell type pairs)", fontsize=S_LABEL)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    departures_legend(fig)
    return save(fig, "supp_primate_branches/branch")


def panel_mtg():
    """Three checks of the departures against level gaps of a held-out cortical region: rank correlation with its interval."""
    m = pd.read_csv(E / "departures" / "mtg_check.csv")
    tests = [("marmoset minus macaque branch vs MTG gap difference", "marmoset vs macaque\nbranch"),
             ("human departures vs MTG human-specific gap", "human branch"),
             ("chimpanzee departures vs MTG human-specific gap (control)", "chimpanzee branch\n(control)")]
    fig, ax = plt.subplots(figsize=(style.inch(150), style.inch(ROW_H)), layout="constrained")
    for i, (tt, lab) in enumerate(tests):
        for j, (rd, mark) in enumerate(DEPARTURES):
            r = m[(m.test == tt) & (m.readout == rd)].iloc[0]
            y = len(tests) - 1 - i + 0.2 - j * 0.2
            ax.errorbar(r.rho, y, xerr=[[r.rho - r.lo], [r.hi - r.rho]], fmt=mark.marker, color=mark.color, ms=3.2, lw=1.0, capsize=0)
    ax.set_yticks(range(len(tests)), [t[1] for t in tests[::-1]], fontsize=S_TICK)
    ax.set_xticks([0, 0.1, 0.2])
    ax.xaxis.set_major_formatter(FuncFormatter(signed))
    ax.tick_params(axis="x", labelsize=S_TICK)
    fig.supxlabel("ρ with held-out MTG level gaps\n(partial on expression)", fontsize=S_LABEL)
    _clean(ax)
    departures_legend(fig)
    return save(fig, "supp_primate_branches/mtg")


HUMAN_LINEAGE = ["FOXP2", "CACNA1D", "DSCAM", "MELTF", "PKD2L1"]  # the human-lineage genes drawn, each in its named cell type


def panel_human_lineage():
    """Count margin per cell of each species for five human-lineage genes in their named cell type (per-gene gate fit)."""
    m = pd.read_csv(E / "departures" / "ma2022_genes.csv")
    m = m[(m.readout == "per-gene") & m.gene.isin(HUMAN_LINEAGE)]
    fig, ax = plt.subplots(figsize=(style.inch(136), style.inch(ROW_H)), layout="constrained")
    for i, r in enumerate(m.itertuples()):
        v = dict(x.split(":") for x in r.S_rows.split(";"))
        for j, s in enumerate(SPECIES):
            ax.bar(i + (j - 1.5) * 0.19, float(v[s]), width=0.18, color=style.SPECIES[s.lower()], label=s.lower() if i == 0 else None)
    ax.set_xticks(range(len(m)), [f"{g}\n{CELL_TYPE.get(t, t)}" for g, t in zip(m.gene, m.cell_type)], fontsize=S_TEXT, rotation=90, linespacing=1.0)
    ax.axhline(0, color=RULE, lw=0.6)
    ax.set_ylabel("count margin per cell\nin the named cell type", fontsize=S_LABEL)
    ax.yaxis.set_major_formatter(FuncFormatter(signed))
    ax.tick_params(axis="y", labelsize=S_TICK)
    ax.set_ylim(-0.5, 1.75)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.legend(frameon=False, fontsize=S_TICK, ncol=2, loc="outside upper center", handlelength=1.0, handletextpad=0.4, columnspacing=1.0)
    return save(fig, "supp_primate_branches/human_lineage")


# panel name in figures/panels -> the function that draws it
DRAW = {"supp_cell_compartments/compartments": panel_compartments, "supp_cell_compartments/pattern_vs_level": panel_pattern_vs_level,
        "supp_cell_lineages/lineages": panel_lineages, "supp_cell_lineages/named": panel_named,
        "supp_primate_branches/branch": panel_branch, "supp_primate_branches/mtg": panel_mtg, "supp_primate_branches/human_lineage": panel_human_lineage}


def draw_panels(names=None) -> list:
    """Draw the panels of this task from the cached tables of results/analysis/E3 and nothing else. ``names``: panel names
    as in ``DRAW`` (default: all seven). Returns the files written."""
    style.apply()
    return [DRAW[n]() for n in (DRAW if names is None else names)]


if __name__ == "__main__":
    for path in draw_panels(sys.argv[1:] or None):
        print(path)
