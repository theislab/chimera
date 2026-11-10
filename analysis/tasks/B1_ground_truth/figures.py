"""B1 figures, one panel per task, from the tidy tables. Saved through figures/style.py to figures/panels/supp_ground_truth/; never to the manuscript.

Every panel is drawn at its size on the page (the text width, 486 bp) and saved at that exact size, so the figures compose at scale 1.
Heat maps use one hue from the value of no detection (AUROC 0.5, power 0.05) to 1; lower values take the lightest color and the
color bar says so; a cell that was not run is light gray. Scores keep the colors, markers and names of the house table.

    PYTHONPATH=$PWD python -m analysis.tasks.B1_ground_truth.figures        # every panel, from the cached tables (draw_panels)
    CHIMERA_FIGURES_OUT=<folder> ... the same command                        # dry run: panels under <folder>/panels
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

from analysis.tasks.B1_ground_truth import rules as RU
from figures import style

# ---------------------------------------------------------------- house style (figures/style.py) and the names of this task
PANELS = "supp_ground_truth"  # figures/panels/<PANELS>/<name>.pdf
WIDTH = style.inch(style.TEXTWIDTH)  # in; every panel fills the text width
S_TITLE, S_LABEL, S_TICK, S_TEXT = style.SIZE_TITLE, style.SIZE_LABEL, style.SIZE_TICK, style.SIZE_TEXT
INK, RULE = style.INK, style.RULE
REF = "#888888"  # reference lines (chance, nominal rate)
NA_CELL = "#F2F2F2"  # heat-map cell that was not run
COH = {"liv_hmouse": "Liver", "test_hmouse": "Testis", "kidney": "Kidney"}  # the three Human+Mouse cohorts, named by tissue
TISSUE = {"liv_hmouse": style.TISSUE["liver"], "test_hmouse": style.TISSUE["testis"], "kidney": style.TISSUE["kidney"]}
MULTI = {"test_3sp": style.COHORT["test_3sp"], "liv_4sp": style.COHORT["liv_4sp"]}  # timed for Chimera alone (practical cost)
ORDER = ["S_g", "Delta_alpha", "Delta_alpha_free", "Delta_alpha_hybrid", "B1z", "S_fwd", "Sdir", "B1", "B1b", "B1c", "B2b", "Lfc", "B3a", "S_deploy", "S_cos", "C_cos", "GeneSpectra", "CACIMAR", "SAMap", "pi", "dNdS", "LOEUF", "expr", "random"]
# Blocks of ORDER; a gap separates them wherever the scores are rows.
FAMILY = {"Chimera readouts": ["S_g", "Delta_alpha", "Delta_alpha_free", "Delta_alpha_hybrid", "B1z", "S_fwd"],
          "direct baselines": ["Sdir", "B1", "B1b", "B1c", "B2b", "Lfc", "B3a"],
          "other tools": ["S_deploy", "S_cos", "C_cos", "GeneSpectra", "CACIMAR", "SAMap"],
          "sequence measures": ["pi", "dNdS", "LOEUF"], "controls": ["expr", "random"]}
# Scores with an entry in the house table of readouts (name, color, marker). The internal codes are only keys and never reach a figure.
READOUT = {"S_g": "count_evidence", "Delta_alpha": "delta_alpha", "Delta_alpha_free": "delta_alpha_per_gene", "B1z": "latent_cluster_correlation",
           "Sdir": "direct_margin", "B1": "profile_correlation", "B1c": "rank_matched_level", "B2b": "specificity_similarity", "Lfc": "fold_change",
           "B3a": "gene_features", "SAMap": "samap", "pi": "protein_identity", "dNdS": "dnds", "LOEUF": "loeuf", "expr": "expression"}
# Reader names of every score; the benchmark overview (B2_benchmark/overview.py) takes its row names from here.
NAME = {**{m: style.READOUT[k].name for m, k in READOUT.items()},
        "Delta_alpha_hybrid": "Δα, gate + per-gene bias", "c_g_free": "$c_g$, per-gene gate", "S_fwd": "$S_g$ without masking (exploratory)",
        "B1b": "profile correlation (residual)", "S_deploy": "SATURN gene score", "S_cos": "SATURN weight cosine",
        "C_cos": "CAMEX gene-embedding cosine", "GeneSpectra": "GeneSpectra", "CACIMAR": "CACIMAR", "random": "random"}
STY = {**{m: (style.READOUT[k].color, style.READOUT[k].marker) for m, k in READOUT.items()}, "S_fwd": ("#56B4E9", "*")}
OPEN = {m for m, k in READOUT.items() if style.READOUT[k].open}  # drawn with a white face
# Alterations and nulls, and the size of an alteration (the x axis of a dose response and the phrase in a title)
TASK = {"level": "level change", "pattern": "pattern change", "loss": "loss", "placebo": "placebo sets", "sham_swap": "sham move",
        "thin_both": "both species thinned"}
EFFECT_LABEL = {"pattern": "share of the move, q", "level": "fold of thinning, f", "loss": "share removed, e"}
TOP_EFFECT = {"level": "8", "pattern": "1", "loss": "1"}  # the strongest effect of each alteration
QUARTILE = {"low": "1", "midlow": "2", "midhigh": "3", "high": "4"}  # expression bands of rules.BANDS: quartiles, lowest to highest
QUARTILE_WORD = {"low": "first", "midlow": "second", "midhigh": "third", "high": "fourth"}
IDENT = {"id_low": "low", "id_mid": "mid", "id_high": "high"}  # tertiles of protein identity
# Kidney cell types in upper-case abbreviations, as in Fig 7 (no lower-case "fi" pair for the text check)
KTYPE = {"atl": "ATL", "cnt": "CNT", "dct": "DCT", "dtl": "DTL", "ec": "EC", "fib": "FIB", "ic": "IC", "lymphoid": "LYM",
         "myeloid": "MYE", "pape": "PapE", "pc": "PC", "pec": "PEC", "pod": "POD", "pt": "PT", "tal": "TAL", "vsm_p": "VSM/P"}
# Methods of the practical-cost panel: name, color, marker. The two other tools are grays, so no readout color is reused for a tool.
COST = {"chimera": ("Chimera", "#0072B2", "o"), "direct": ("direct baselines", "#009E73", "D"), "saturn": ("SATURN", "#555555", "s"),
        "camex": ("CAMEX", "#AAAAAA", "^")}
BEFORE, AFTER = "#BBBBBB", "#0072B2"  # raw counts before and after an alteration
BAND = "#F4F4F4"  # light band under every second row of a dot plot
SEQ = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#a6c8e4", "#2a78d6", "#08306b"])


def setup():
    style.apply()


def save(fig, name):
    """One panel at its exact size: figures/panels/<PANELS>/<name>.pdf."""
    style.save_panel(fig, f"{PANELS}/{name}", exact=True)


def parse(t: pd.DataFrame) -> pd.DataFrame:
    kv = t["setting"].str.split(";", expand=True)
    for i, k in enumerate(["cohort", "effect", "band", "share", "identity"]):
        t[k] = kv[i].str.split("=").str[1]
    return t


def pick(t, **k):
    d = t
    for a, b in k.items():
        d = d[d[a] == str(b)] if a in ("cohort", "effect", "band", "share", "identity") else d[d[a] == b]
    return d


def mark(m: str, ms: float = 3.5) -> dict:
    """Color and marker of a score, as keywords of plot: the house table, with a white face where it says open."""
    c, mk = STY[m]
    return dict(color=c, marker=mk, ms=ms, markerfacecolor="white" if m in OPEN else c, markeredgecolor=c, markeredgewidth=0.7 if m in OPEN else 0.0)


def family_starts(methods: list) -> list:
    """Indices of the rows that begin a new block of FAMILY (the first row left out)."""
    block = {m: k for k, ms in enumerate(FAMILY.values()) for m in ms}
    return [i for i in range(1, len(methods)) if block.get(methods[i]) != block.get(methods[i - 1])]


def row_positions(methods: list, gap: float = 0.6) -> np.ndarray:
    """y of every row when the scores are rows of a dot plot: one unit apart, a further gap between blocks of FAMILY."""
    starts = set(family_starts(methods))
    return np.cumsum([0.0] + [1.0 + (gap if i in starts else 0.0) for i in range(1, len(methods))])


def effect_phrase(task: str, effect) -> str:
    """An alteration at one effect size, in words: 'level change (fold of thinning, f = 8)'."""
    return f"{TASK[task]} ({EFFECT_LABEL[task]} = {effect})"


def legend_name(m: str) -> str:
    return NAME[m].replace(", clusters", ",\nclusters")


def heat_panel(ax, M: pd.DataFrame, vmin=0.5, vmax=1.0, fmt="{:.2f}", cmap=SEQ, gaps=()):
    """Values as colored cells with the number printed in the cell. A cell without a value is light gray and empty; ``gaps`` are
    the rows that begin a new block, set off by a white rule."""
    cm = cmap.copy()
    cm.set_bad(NA_CELL)
    im = ax.imshow(np.ma.masked_invalid(M.to_numpy(dtype=float)), aspect="auto", vmin=vmin, vmax=vmax, cmap=cm)
    ax.set_xticks(range(M.shape[1]), M.columns, rotation=0)
    ax.set_yticks(range(M.shape[0]), M.index)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=S_TEXT, color="white" if (v - vmin) / (vmax - vmin) > 0.6 else INK)
    for k in gaps:
        ax.axhline(k - 0.5, color="white", lw=3.0)
    return im


def color_bar(fig, im, axes, label: str, values, ticks: list, **kw):
    """Color bar beside heat maps; ``ticks`` start at the lower limit of the scale. Where values lie under that limit they take
    the lightest color: the bar then ends in an arrow and its lowest tick reads "≤ limit"."""
    low = float(np.nanmin(np.asarray(values, dtype=float))) < im.norm.vmin
    cb = fig.colorbar(im, ax=axes, location="right", extend="min" if low else "neither", ticks=ticks, **kw)
    if low:
        cb.set_ticklabels([f"≤ {ticks[0]:g}"] + [f"{v:g}" for v in ticks[1:]])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, labelsize=S_TICK)
    cb.set_label(label, fontsize=S_TICK)
    return cb


def fig_task(t: pd.DataFrame, task: str, effects: list, top_effect):
    """Top: AUROC by score x cohort x expression quartile at the strongest effect (share marginal), and power over the three
    cohorts together. Bottom: dose response per cohort in the third expression quartile."""
    tt = t[t.task == task]
    methods = [m for m in ORDER if m in set(tt.method)]
    bands = RU.BANDS
    cols = [(c, b) for c in COH for b in bands]
    gaps = family_starts(methods)
    fig = plt.figure(figsize=(WIDTH, style.inch(414)), constrained_layout=True)
    fig.suptitle(effect_phrase(task, top_effect), x=0.008, ha="left", fontsize=S_TITLE)
    gs = fig.add_gridspec(2, 1, height_ratios=[2.5, 1])
    top = gs[0].subgridspec(1, 2, width_ratios=[len(cols), 1.3 * len(bands)])
    axh = fig.add_subplot(top[0])
    d = tt[(tt.metric == "AUROC") & (tt.effect == str(top_effect)) & (tt.share == "all") & (tt.identity == "all")]
    M = pd.DataFrame({f"{c} {b}": d[(d.cohort == c) & (d.band == b)].set_index("method")["value"].reindex(methods) for c, b in cols})
    M.index = [NAME[m] for m in methods]
    im = heat_panel(axh, M, gaps=gaps)
    # expression quartiles as ticks, cohorts once above their four columns
    axh.set_xticks(range(len(cols)), [QUARTILE[b] for _, b in cols])
    axh.set_xlabel("expression quartile (1 lowest, 4 highest)")
    for k in range(1, len(COH)):
        axh.axvline(k * len(bands) - 0.5, color="white", lw=1.5)
    head = axh.secondary_xaxis("top")
    head.set_xticks([k * len(bands) + (len(bands) - 1) / 2 for k in range(len(COH))], list(COH.values()))
    head.tick_params(length=0)
    head.spines["top"].set_visible(False)
    axh.set_title("AUROC against matched controls", loc="left")
    color_bar(fig, im, axh, "AUROC", M, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fraction=0.04, pad=0.012, aspect=38)
    axp = fig.add_subplot(top[1])
    d = tt[(tt.metric == "power_05") & (tt.effect == str(top_effect)) & (tt.share == "all") & (tt.identity == "all") & (tt.cohort == "all")]
    P = pd.DataFrame({b: d[d.band == b].set_index("method")["value"].reindex(methods) for b in bands})
    P.index = [NAME[m] for m in methods]
    imp = heat_panel(axp, P, vmin=0.05, vmax=1.0, gaps=gaps)
    axp.set_xticks(range(len(bands)), [QUARTILE[b] for b in bands])
    axp.set_xlabel("expression quartile")
    axp.set_yticklabels([])
    headp = axp.secondary_xaxis("top")
    headp.set_xticks([(len(bands) - 1) / 2], [f"all {len(COH)} cohorts"])
    headp.tick_params(length=0)
    headp.spines["top"].set_visible(False)
    axp.set_title("power at 5% false positives", loc="left")
    color_bar(fig, imp, axp, "power", P, [0.05, 0.25, 0.5, 0.75, 1.0], fraction=0.11, pad=0.03, aspect=38)
    show = [m for m in ("S_g", "Delta_alpha", "Delta_alpha_free", "B1z", "Sdir", "B1", "B1c", "Lfc") if m in methods]
    bottom = gs[1].subgridspec(1, len(COH))
    for i, c in enumerate(COH):
        ax = fig.add_subplot(bottom[i])
        d = tt[(tt.metric == "AUROC") & (tt.cohort == c) & (tt.band == RU.CENTRAL_BAND) & (tt.share == "all") & (tt.identity == "all") & (tt.effect != "all")]
        for m in show:
            s = d[d.method == m].assign(e=lambda x: x.effect.astype(float)).sort_values("e")
            ax.errorbar(range(len(s)), s["value"], yerr=[s["value"] - s["ci_lo"], s["ci_hi"] - s["value"]], lw=1, capsize=0, label=legend_name(m), **mark(m))
        ax.set_xticks(range(len(effects)), effects)
        ax.axhline(0.5, color=REF, lw=0.6, ls="--")
        ax.set_ylim(0.4, 1.0)
        ax.set_title(COH[c], loc="left")
        ax.set_xlabel(EFFECT_LABEL[task])
        if i == 0:
            ax.set_ylabel(f"AUROC, {QUARTILE_WORD[RU.CENTRAL_BAND]}\nexpression quartile")
        if i == len(COH) - 1:
            ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    save(fig, f"task_{task}")


def fig_low(t):
    """Power by expression quartile at the strongest effect of each alteration, over the three cohorts together (top), and the
    share of altered genes whose S_g is masked to exactly 0 (bottom)."""
    tasks = ["level", "pattern", "loss"]
    fig, ax = plt.subplots(2, 3, figsize=(WIDTH, style.inch(244)), sharex=True, constrained_layout=True)
    for j, task in enumerate(tasks):
        top = TOP_EFFECT[task]
        d = t[(t.task == task) & (t.effect == top) & (t.share == "all") & (t.identity == "all") & (t.cohort == "all")]
        for m in ("S_g", "S_fwd", "Delta_alpha", "Delta_alpha_free", "Sdir", "B1", "Lfc"):
            s = d[(d.metric == "power_05") & (d.method == m)].set_index("band").reindex(RU.BANDS)
            ax[0, j].plot(range(4), s["value"], lw=1, label=NAME[m], **mark(m))
        ax[0, j].axhline(0.05, color=REF, lw=0.6, ls="--")
        ax[0, j].set_ylim(0, 1)
        ax[0, j].set_title(effect_phrase(task, top).replace(" (", "\n("), loc="left")
        s = d[(d.metric == "frac_zero") & (d.method == "S_g")].set_index("band").reindex(RU.BANDS)
        ax[1, j].bar(range(4), s["value"], color=STY["S_g"][0], width=0.6)
        ax[1, j].set_xticks(range(4), [QUARTILE[b] for b in RU.BANDS])
        ax[1, j].set_xlabel("expression quartile (1 lowest, 4 highest)")
        ax[1, j].set_ylim(0, 1)
    ax[0, 0].set_ylabel("power at 5% false positives")
    ax[1, 0].set_ylabel("share of altered genes\nwith $S_g$ exactly 0")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="outside lower center", ncols=4)
    save(fig, "low_expression")


def fig_nulls(t):
    """Rejection rate at 0.05 on the three nulls, per score (rows, in the blocks of FAMILY) and cohort. The markers of the three
    cohorts stand side by side inside a row, so that equal rates do not hide each other."""
    tasks = [("placebo", f"{TASK['placebo']}, unaltered fits"), ("sham_swap", TASK["sham_swap"]), ("thin_both", f"{TASK['thin_both']} 4-fold")]
    methods = [m for m in ORDER if m in set(t.method) and m not in ("random", )]
    y = row_positions(methods)
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, style.inch(252)), sharey=True, constrained_layout=True)
    for j, (task, lab) in enumerate(tasks):
        d = t[(t.task == task) & (t.metric == "reject05") & (t.band == "all") & (t.effect == "all") & (t.share == "all") & (t.identity == "all") & (t.cohort.isin(COH))]
        for i, c in enumerate(COH):
            s = d[d.cohort == c].set_index("method")["value"].reindex(methods)
            ax[j].plot(s.values, y + (i - (len(COH) - 1) / 2) * 0.27, ls="", marker=TISSUE[c].marker, ms=3.2, color=TISSUE[c].color, markeredgewidth=0, label=COH[c])
        for yk in y[::2]:  # every second row on a light band, to follow a row across the three plots
            ax[j].axhspan(yk - 0.5, yk + 0.5, color=BAND, lw=0, zorder=0)
        ax[j].axvline(0.05, color=REF, lw=0.6, ls="--")
        ax[j].set_title(lab, loc="left")
        ax[j].set_xlim(-0.03, 1.03)
        ax[j].set_xlabel("rejection rate at α = 0.05")
        ax[j].tick_params(axis="y", length=0)
    ax[0].set_ylim(y[-1] + 0.75, y[0] - 0.75)
    ax[0].set_yticks(y, [NAME[m] for m in methods])
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside lower center", ncols=len(COH))
    save(fig, "type1_error")


def fig_share(t):
    """AUROC against the share of altered genes, at the strongest level effect (solid) and pattern effect (dotted), per cohort.
    One key for the scores, one for the two alterations; the AUROC axis is that of the dose responses."""
    scores = ("S_g", "Delta_alpha", "Delta_alpha_free", "B1")
    kinds = (("level", "-"), ("pattern", ":"))
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, style.inch(150)), sharey=True, constrained_layout=True)
    for j, c in enumerate(COH):
        d = t[(t.task == "level") & (t.cohort == c) & (t.metric == "AUROC") & (t.effect == TOP_EFFECT["level"]) & (t.band == "all") & (t.identity == "all") & (t.share != "all")]
        d2 = t[(t.task == "pattern") & (t.cohort == c) & (t.metric == "AUROC") & (t.effect == TOP_EFFECT["pattern"]) & (t.band == "all") & (t.identity == "all") & (t.share != "all")]
        for m in scores:
            for dd, ls in ((d, kinds[0][1]), (d2, kinds[1][1])):
                s = dd[dd.method == m].assign(x=lambda x: x.share.astype(float)).sort_values("x")
                ax[j].plot(s["x"] * 100, s["value"], lw=1, ls=ls, **mark(m, 3))
        ax[j].axhline(0.5, color=REF, lw=0.6, ls="--")
        ax[j].set_ylim(0.4, 1.0)
        ax[j].set_xticks([100 * q for q in RU.SHARES])
        ax[j].set_title(COH[c], loc="left")
        ax[j].set_xlabel("share of altered genes (%)")
    ax[0].set_ylabel("AUROC")
    key = ax[-1].legend(handles=[Line2D([], [], lw=1, label=NAME[m], **mark(m, 3)) for m in scores], loc="upper left", bbox_to_anchor=(1.0, 1.03))
    ax[-1].add_artist(key)
    ax[-1].legend(handles=[Line2D([], [], color=INK, lw=1, ls=ls, label=effect_phrase(task, TOP_EFFECT[task]).replace(" (", "\n(")) for task, ls in kinds],
                  loc="lower left", bbox_to_anchor=(1.0, -0.03))
    save(fig, "share")


def fig_identity_cells():
    """Identity tertiles from the per-fit table (subsets are scored within each fit)."""
    c = pd.read_parquet(RU.RES / "per_cell.parquet")
    c = c[(c.metric == "AUROC") & c.subset.isin(RU.IDENTITY_BANDS) & c.alt.isin(["level", "pattern", "loss"])]
    methods = ["S_g", "Delta_alpha", "Delta_alpha_free", "B1", "Sdir", "B1c"]
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, style.inch(128)), constrained_layout=True)
    drawn = []
    for j, task in enumerate(("level", "pattern", "loss")):
        M = c[c.alt == task].groupby(["method", "subset"])["value"].mean().unstack().reindex(methods)[RU.IDENTITY_BANDS]
        M.index = [NAME[m] for m in methods]
        M.columns = [IDENT[b] for b in M.columns]
        im = heat_panel(ax[j], M, gaps=family_starts(methods))
        drawn.append(M.to_numpy(dtype=float))
        ax[j].set_title(TASK[task], loc="left")
        ax[j].set_xlabel("protein identity tertile")
        if j:
            ax[j].set_yticklabels([])
    color_bar(fig, im, list(ax), "AUROC", np.concatenate(drawn), [0.5, 0.6, 0.7, 0.8, 0.9, 1.0], fraction=0.025, pad=0.012, aspect=16)
    save(fig, "identity")
    return c


def fig_identity(t):
    methods = ["S_g", "Delta_alpha", "Delta_alpha_free", "B1", "Sdir", "B1c"]
    rows = []
    for task in ("level", "pattern", "loss"):
        for m in methods:
            for ib in RU.IDENTITY_BANDS:
                d = t[(t.task == task) & (t.method == m) & (t.metric == "AUROC") & (t.identity == ib)]
                rows.append(dict(task=task, method=NAME[m], identity=ib, value=d["value"].mean()))
    r = pd.DataFrame(rows)
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, style.inch(128)), constrained_layout=True)
    for j, task in enumerate(("level", "pattern", "loss")):
        M = r[r.task == task].pivot(index="method", columns="identity", values="value").loc[[NAME[m] for m in methods], RU.IDENTITY_BANDS]
        M.columns = [IDENT[b] for b in M.columns]
        heat_panel(ax[j], M, gaps=family_starts(methods))
        ax[j].set_title(TASK[task], loc="left")
        ax[j].set_xlabel("protein identity tertile")
        if j:
            ax[j].set_yticklabels([])
    save(fig, "identity")


def fig_examples():
    e = pd.read_parquet(RU.RES / "qc" / "example_counts.parquet")
    e = e[(e.cohort == "kidney") & (e.band == "high") & e.alt.isin(["pattern", "level", "loss"])]
    picks = e.groupby("alt")["gene"].first()
    from analysis.common import gene_ids
    sym = gene_ids.symbols("kidney", list(picks.values))
    sym = {g: (v if isinstance(v, str) and v else g) for g, v in zip(picks.values, list(sym))}
    fig, ax = plt.subplots(3, 2, figsize=(WIDTH, style.inch(352)), constrained_layout=True, sharex="col")
    for i, (alt, gene) in enumerate(picks.items()):
        d = e[(e.alt == alt) & (e.gene == gene)]
        sp = d["species_altered"].iloc[0]
        types = sorted(d.cell_type.unique())
        # cell types an alteration acts on: the gene's top type for a loss, its top and bottom types for a pattern change; a level
        # change thins every cell of the species, so no type is singled out
        acted = [d[k].iloc[0] for k in {"loss": ["type_top"], "pattern": ["type_top", "type_bottom"]}.get(alt, [])]
        for j, s in enumerate((sp, "Human" if sp == "Mouse" else "Mouse")):
            bb = d[(d.state == "before") & (d.species == s)].groupby("cell_type")["count"].mean().reindex(types)
            aa = d[(d.state == "after") & (d.species == s)].groupby("cell_type")["count"].mean().reindex(types)
            x = np.arange(len(types))
            ax[i, j].bar(x - 0.2, bb, width=0.38, color=BEFORE)
            ax[i, j].bar(x + 0.2, aa, width=0.38, color=AFTER)
            ax[i, j].set_title(f"$\\mathit{{{sym[gene]}}}$, {s.lower()}: {TASK[alt] if j == 0 else 'untouched'}", loc="left")
            ax[i, j].set_xticks(x, [KTYPE.get(t, t) for t in types], rotation=90)
            if j == 0:
                ax[i, j].set_ylabel("mean raw count per cell")
                top = float(max(bb.max(), aa.max()))
                for k in acted:
                    ax[i, j].plot([types.index(k)], [max(bb[k], aa[k]) + 0.07 * top], ls="", marker="v", ms=4, color=INK, markeredgewidth=0)
                ax[i, j].set_ylim(0, 1.16 * top)
    fig.legend(handles=[Line2D([], [], ls="", marker="s", ms=5, color=BEFORE, markeredgewidth=0, label="before the alteration"),
                        Line2D([], [], ls="", marker="s", ms=5, color=AFTER, markeredgewidth=0, label="after"),
                        Line2D([], [], ls="", marker="v", ms=4, color=INK, markeredgewidth=0, label="cell type the alteration acts on")],
               loc="outside upper right", ncols=3)
    save(fig, "raw_counts")


def fig_trackd():
    """Practical cost of one run on 16 CPUs: wall time and peak memory per method and cohort (bars), and wall time against the
    number of testis cells (lines). The three- and four-species cohorts were timed for Chimera alone: they form a group of
    their own under a rule, so that no bar stands beside a method that was not run."""
    t = pd.read_csv(RU.RES / "track_d" / "runtime.csv")
    t = t[t.finished]
    meth = list(COST)
    full = t[t.object == "full"].groupby(["cohort", "method"]).agg(w=("wall_s", "mean"), r=("peak_rss_gb", "max")).reset_index()
    alone = [c for c in MULTI if set(full.loc[full.cohort == c, "method"]) == {"chimera"}]
    y = np.arange(len(COH), dtype=float)  # one cluster of bars per Human+Mouse cohort
    ya = len(COH) - 0.05 + 0.5 * np.arange(len(alone))  # single bars of the cohorts timed for Chimera alone
    fig, ax = plt.subplots(1, 3, figsize=(WIDTH, style.inch(150)), constrained_layout=True)
    for i, m in enumerate(meth):
        s = full[full.method == m].set_index("cohort").reindex(list(COH))
        ax[0].barh(y + (i - 1.5) * 0.2, s["w"] / 3600, height=0.18, color=COST[m][1])
        ax[1].barh(y + (i - 1.5) * 0.2, s["r"], height=0.18, color=COST[m][1])
    s = full[full.method == "chimera"].set_index("cohort").reindex(alone)
    ax[0].barh(ya, s["w"] / 3600, height=0.18, color=COST["chimera"][1])
    ax[1].barh(ya, s["r"], height=0.18, color=COST["chimera"][1])
    names = [f"{COH[c]} Human+Mouse" for c in COH] + [MULTI[c] for c in alone]
    for k, l in ((0, "wall time (h), 16 CPUs"), (1, "peak memory (GB)")):
        ax[k].set_yticks(list(y) + list(ya), names if k == 0 else [""] * len(names))
        ax[k].set_ylim((ya[-1] if alone else y[-1]) + 0.5, -0.55)
        ax[k].tick_params(axis="y", length=0)
        ax[k].set_xlabel(l)
        if alone:
            ax[k].axhline(len(COH) - 0.47, color=RULE, lw=0.5)
            ax[k].text(0.98, float(np.mean(ya)), "Chimera only", transform=ax[k].get_yaxis_transform(), ha="right", va="center", fontsize=S_TICK, color=style.MUTED)
    frac = {"f25": 0.25, "f50": 0.5, "full": 1.0}
    s = t[t.cohort == "test_hmouse"].copy()
    s["f"] = s["object"].map(frac)
    for m in meth:
        q = s[s.method == m].groupby("f")["wall_s"].mean()
        ax[2].plot(q.index * 21241 / 1000, q.values / 3600, color=COST[m][1], marker=COST[m][2], ms=3.5, lw=1, markeredgewidth=0)
    ax[2].set_xlabel("testis cells (thousands)")
    ax[2].set_ylabel("wall time (h), 16 CPUs")
    fig.legend(handles=[Line2D([], [], color=COST[m][1], marker=COST[m][2], ms=3.5, lw=1, markeredgewidth=0, label=COST[m][0]) for m in meth],
               loc="outside upper center", ncols=len(meth))
    save(fig, "practical_cost")


def draw_panels():
    """Plotting only: the nine panels from the cached tables (B1_tidy.csv, per_cell.parquet, qc/example_counts.parquet,
    track_d/runtime.csv). Nothing is scored."""
    setup()
    t = parse(pd.read_csv(RU.RES / "B1_tidy.csv", dtype={"setting": str}))
    fig_task(t, "pattern", [0.25, 0.5, 1.0], TOP_EFFECT["pattern"])
    fig_task(t, "level", [2, 4, 8], TOP_EFFECT["level"])
    fig_task(t, "loss", [0.5, 0.9, 1.0], TOP_EFFECT["loss"])
    fig_low(t)
    fig_nulls(t)
    fig_share(t)
    fig_identity_cells()
    fig_examples()
    try:
        fig_trackd()
    except FileNotFoundError:
        print("no track D yet")


def main():
    draw_panels()


if __name__ == "__main__":
    main()
