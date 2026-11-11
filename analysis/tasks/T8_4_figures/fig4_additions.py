"""Panels on the liver reading: prior landscape, cell-type drivers, margin support, tissue comparison.

  fig4c_prior_landscape    Prior landscape (fig:dalpha-biology): protein-identity prior vs Delta-alpha on
                           liver 4 species, with four named gene groups and no
                           internal run name on any axis. A dagger marks a named tail gene (|Delta-alpha|
                           > 0.05) that lacks margin support (T2_5_celltype_margin: count evidence S_g unmasked, of
                           the sign of Delta-alpha and at least the cohort median |S_g|), i.e. imputed from features.
  fig4_celltype_drivers    Where each named gene's count-evidence margin S_g comes from: the exact
                           per-cell-type split C_gk (posterior-mean z) of T2_5_celltype_margin, liver 4 species.
  fig4_margin_support      Fraction of tail genes (|Delta-alpha| > 0.05) with margin support, per cohort and
                           tail, one trained run per cohort: tables of T2_5_celltype_margin on the canonical s42
                           runs, and the same flag (``margin.margin_flag``) on the first kept run of every other
                           cohort in ``runs.tsv`` of H1_harmonise, so later cohorts join when H1_harmonise lists
                           them. Checked against T2_5_celltype_margin.
  fig4_cross_tissue        Tissue comparison (T2_3_tissue, primary version): Delta-alpha against
                           the raw prior, centered on the cohort median, same species pair, genes joined
                           on Ensembl ID.
The names on the left are those the script saves under; ``figures/panel_names.tsv`` gives the panel file of each
(``figures/panels/main_delta_alpha_liver/``).
Sources: ``results/analysis/T2_5/`` (canonical s42 runs), ``results/analysis/T2_3/tissue_comparison.csv``.
"""

from __future__ import annotations

import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import spearmanr

from analysis.tasks.T8_4_figures import common as cm

T25 = cm.RES / "T2_5"
T23 = cm.RES / "T2_3"
# Gene groups of the prior landscape panel (notebooks/_alpha_biology_builder.py, H_THEMES), same colors.
THEMES = {
    "urea cycle / gluconeogenesis": (["CPS1", "ARG1", "ASL", "OTC", "PCK1", "PAH"], "#0072B2"),
    "mitochondrially encoded OXPHOS": (["MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND4",
                                        "MT-ATP6"], "#5B3E80"),
    "supporting tissue": (["SELP", "SELE", "VCAM1", "F3", "CR2", "CD40LG", "COL17A1", "PDGFRB", "KIT", "C1QC"],
                      "#D55E00"),
    "conserved controls": (["DDX4", "RPL3", "EEF1A1", "EEF2", "ACTB", "TUBA1A", "COX4I1", "NDUFA1", "HNF4A"], "#7F7F7F"),
}
DAGGER = "†"
# Not drawn (ortholog audit O1): OTC has no human counts in the source matrix (a source zero); COL17A1 and
# CD40LG have wrong zebrafish one-to-one targets (col19a1, identity 5; tnfsf10l4, identity 18); SELE's
# zebrafish target is selp and it has no mouse counts; SELP's zebrafish target is an unnamed low-confidence
# gene (identity 13). Applies to panels c and d alike.
O1_DROPPED = {"OTC", "COL17A1", "CD40LG", "SELE", "SELP"}


# Genes of the lower tail that pass every check of the named-gene table (same sign in all five seeds, margin support,
# counts in every species): named in bold in the landscape panel, which otherwise names genes that the text qualifies.
EVERY_CHECK = ["HAL", "ARG2", "CYP7A1"]
EVERY_CHECK_NAME, EVERY_CHECK_COLOR = "enzymes passing every check", "#111111"
# Where their symbols stand: offset of a corner of the symbol from the marker (bp) and the corner. Fixed, because two of
# the three markers nearly coincide; the other symbols are placed around them.
EVERY_CHECK_PLACE = {"CYP7A1": ((-2.6, 2.6), "right", "bottom"), "ARG2": ((-2.6, -2.6), "right", "top"), "HAL": ((2.6, -2.6), "left", "top")}


def genes_4sp() -> pd.DataFrame:
    return pd.read_parquet(T25 / "liv_4sp__genes.parquet")


def every_check(g: pd.DataFrame) -> pd.DataFrame:
    """The rows of ``g`` for the genes of ``EVERY_CHECK``, after confirming in the stored tables what the name of the group
    says: each gene is in the plotted table with margin support and hepatocytes as the cell type that contributes most
    to its count evidence (``T2_5/liv_4sp__genes.parquet``), and the named-gene table gives it the class "confident",
    the same sign in all five seeds and counts in every species (``X2/named_genes.csv``)."""
    s = g[g.symbol.isin(EVERY_CHECK)].copy()
    assert sorted(s.symbol) == sorted(EVERY_CHECK), f"not in the plotted table: {sorted(set(EVERY_CHECK) - set(s.symbol))}"
    assert s["tail"].all() and s.margin_support.all() and not s.lacks_margin_support.any(), "a gene of EVERY_CHECK lacks margin support"
    assert (s.dominant_cell_type == "Hepatocytes").all(), "a gene of EVERY_CHECK is not led by hepatocytes"
    t = pd.read_csv(cm.RES / "X2" / "named_genes.csv").set_index("symbol").loc[EVERY_CHECK]
    passed = (t["class"] == "confident") & (t.sign_consistency == 1.0) & (t.margin_support.astype(str) == "True") & (t.complete.astype(str) == "True")
    assert passed.all(), f"not passing every check in X2/named_genes.csv: {list(t.index[~passed])}"
    s["theme"], s["color"], s["label"] = EVERY_CHECK_NAME, EVERY_CHECK_COLOR, s.symbol
    return s


def named(g: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for theme, (genes, col) in THEMES.items():
        s = g[g.symbol.isin(set(genes) - O1_DROPPED)].copy()
        s["theme"], s["color"] = theme, col
        rows.append(s)
    out = pd.concat(rows)
    # Dagger: T2.5 flag ``lacks_margin_support`` (a tail gene, |Delta-alpha| > 0.05, whose count evidence
    # does not back its sign), as planned for the main panel. Near-zero genes outside the tail carry no
    # claim and no dagger.
    out["label"] = out.symbol + np.where(out.lacks_margin_support, DAGGER, "")
    return out


def panel_landscape() -> plt.Figure:
    """Main Δα figure, panel c (``main_delta_alpha_liver/prior_landscape``, 203 x 205 bp): every gene of the four-species
    liver model on the plane of protein prior and Δα, with five groups of named genes.

    What it draws. One gray dot per gene, protein prior π on the x axis and Δα on the y axis; the dashed line is Δα = 0.
    The two margins hold the distribution of each quantity over all genes (60 bins). The genes of ``THEMES`` are
    colored markers with their symbols (those of ``O1_DROPPED`` are left out); a dagger after a symbol marks a tail gene
    (|Δα| > 0.05) whose Δα is not backed by count evidence of the same sign. The genes of ``EVERY_CHECK`` are black
    markers with bold symbols: three hepatocyte enzymes below the prior that pass every check (``every_check``). The key
    stands above the plot, so that the x axis lies on one line with the panels beside it
    (``figures/redraw/main_figures.py`` takes the height of its two panels from this one).

    Reads ``results/analysis/T2_5/liv_4sp__genes.parquet``, columns ``symbol``, ``prior``, ``delta_alpha`` and
    ``lacks_margin_support`` (the dagger), and for the three genes of ``EVERY_CHECK`` their class in
    ``results/analysis/X2/named_genes.csv``. No value is computed.

    Symbols are placed by ``adjust_text``, which keeps them off the markers and off each other and stops by itself; the
    same placement comes out on every run. The three bold symbols stand at a fixed place beside their markers
    (``EVERY_CHECK_PLACE``). The panel stops when a gene lies outside the axes, or when a symbol still
    lies on a marker or on another symbol.

        python -m analysis.tasks.T8_4_figures.fig4_additions --panel main_delta_alpha_liver/prior_landscape     # in chimera_env, from the repository root
    """
    from matplotlib.transforms import Bbox

    g = genes_4sp()
    nm = pd.concat([named(g), every_check(g)])
    W, H = 203.5, 205.1  # bp, before the crop to the ink
    left, bottom, margin, gap, key = 25.0, 17.0, 12.0, 1.5, 23.5  # bp: y-axis band, x-axis band, depth of a margin plot, its gap, band of the key
    w, h = W - left - gap - margin - 0.6, H - bottom - gap - margin - key
    xlim, ylim = (0.15, 1.0), (-0.56, 0.23)
    assert g.prior.between(*xlim).all() and g.delta_alpha.between(*ylim).all(), "a gene lies outside the axes"
    mark, size, line = 11.0, 5.3, 0.5  # area of a named marker (pt^2); pt of gene symbols and key; pt of the axes
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_axes([left / W, bottom / H, w / W, h / H])
    top = fig.add_axes([left / W, (bottom + h + gap) / H, w / W, margin / H], sharex=ax)
    right = fig.add_axes([(left + w + gap) / W, bottom / H, margin / W, h / H], sharey=ax)
    cm.fd.density_scatter(ax, g.prior, g.delta_alpha, s=1.2, alpha=0.25, color="#BBBBBB")
    texts, fixed = [], []
    groups = [(theme, col) for theme, (_, col) in THEMES.items()] + [(EVERY_CHECK_NAME, EVERY_CHECK_COLOR)]
    for theme, col in groups:
        s = nm[nm.theme == theme]
        ax.scatter(s.prior, s.delta_alpha, s=mark, color=col, edgecolor="white", linewidth=0.35, zorder=4)
        for r in s.itertuples():
            if theme == EVERY_CHECK_NAME:
                offset, ha, va = EVERY_CHECK_PLACE[r.symbol]
                fixed.append(ax.annotate(r.label, (r.prior, r.delta_alpha), xytext=offset, textcoords="offset points", fontsize=size, ha=ha, va=va,
                                         color=cm.INK, zorder=5, fontweight="bold"))
            else:
                texts.append(ax.text(r.prior, r.delta_alpha, r.label, fontsize=size, ha="center", va="center", color=cm.INK, zorder=5))
    ax.axhline(0, color="#888888", lw=line, ls="--", zorder=2)
    ax.set_xlabel("protein prior π", fontsize=6.2, labelpad=2.1)
    ax.set_ylabel("Δα (liver 4 species)", fontsize=6.2, labelpad=1.5)
    ax.tick_params(labelsize=5.5, length=2.0, width=0.4, pad=1.5)
    ax.set_xlim(*xlim)
    ax.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylim(*ylim)
    ax.set_yticks(np.arange(-5, 3) / 10)
    top.hist(g.prior, bins=60, color="#999999", edgecolor="none")
    right.hist(g.delta_alpha, bins=60, color="#999999", edgecolor="none", orientation="horizontal")
    for a, hide in ((top, ("top", "right", "left")), (right, ("top", "right", "bottom"))):
        for sp in hide:
            a.spines[sp].set_visible(False)
    for a in (ax, top, right):
        for sp in a.spines.values():
            sp.set_linewidth(line)
    top.set_yticks([])
    top.tick_params(axis="x", bottom=False, labelbottom=False)
    right.set_xticks([])
    right.tick_params(axis="y", left=False, labelleft=False)
    # The key, above the plot, in two columns: the four groups, then the genes that pass every check (bold, as their
    # symbols) and the dagger.
    for i, (theme, col) in enumerate(groups):
        x, y = left + 0.5 + 106.0 * (i // 2 if i < 4 else 0), H - 4.9 - 6.9 * (i % 2 if i < 4 else 2)
        fig.add_artist(Line2D([(x + 1.8) / W], [(y + 1.9) / H], transform=fig.transFigure, ls="none", marker="o", ms=mark**0.5,
                              mfc=col, mec="white", mew=0.35))
        fig.text((x + 5.5) / W, y / H, theme, fontsize=size, ha="left", va="baseline", color=cm.INK, fontweight="bold" if i == 4 else "normal")
    fig.text((left + 0.5 + 106.0 + 1.8) / W, (H - 18.7) / H, DAGGER, fontsize=size, ha="center", va="baseline", color=cm.INK)
    fig.text((left + 0.5 + 106.0 + 5.5) / W, (H - 18.7) / H, "no margin support", fontsize=size, ha="left", va="baseline", color=cm.INK)
    # Symbols: pushed off each other, off a box of 3.2 bp around every marker (the marker and some air) and off the
    # three symbols that stand at a fixed place.
    rx, ry = 3.2 * (xlim[1] - xlim[0]) / w, 3.2 * (ylim[1] - ylim[0]) / h
    disks = [Bbox.from_extents(x - rx, y - ry, x + rx, y + ry) for x, y in zip(nm.prior, nm.delta_alpha)]
    fig.canvas.draw()
    held = [t.get_window_extent().transformed(ax.transData.inverted()) for t in fixed]
    adjust_text(texts, x=nm.prior.to_numpy(), y=nm.delta_alpha.to_numpy(), objects=disks + held, ax=ax, expand=(1.05, 1.2),
                force_text=(0.25, 0.35), force_static=(0.25, 0.35), force_explode=(0.2, 0.4), pull_threshold=35, iter_lim=400,
                arrowprops=dict(arrowstyle="-", color="#999999", lw=0.3), min_arrow_len=1.0)
    texts += fixed  # in the order of the markers (nm): the check below names a symbol by its place in this list
    fig.canvas.draw()
    boxes = [t.get_window_extent() for t in texts]
    points = ax.transData.transform(nm[["prior", "delta_alpha"]].to_numpy())
    reach = (mark**0.5 / 2 + 0.35 + 0.5) * fig.dpi / 72  # a marker, its edge and 0.5 bp of air, in pixels
    for i, b in enumerate(boxes):
        on = [nm.label.iloc[j] for j, (x, y) in enumerate(points) if np.hypot(max(b.x0 - x, 0, x - b.x1), max(b.y0 - y, 0, y - b.y1)) < reach]
        on += [texts[j].get_text() for j in range(i) if b.overlaps(boxes[j])]
        assert not on, f"the symbol {texts[i].get_text()} lies on {', '.join(on)}"
    return fig


CT_GROUPS = [("hepatocytes", ["Hepatocytes"], "#E69F00"), ("endothelial", ["Endothelial"], "#56B4E9"),
             ("stellate / fibroblasts", ["fibroblasts"], "#009E73"),
             ("Kupffer cells and macrophages", ["Kupffer cells", "Macrophages"], "#CC79A7"),
             ("other cell types", None, "#BBBBBB")]


def panel_drivers() -> plt.Figure:
    g = genes_4sp()
    nm = named(g).sort_values("delta_alpha")
    C = pd.read_parquet(T25 / "liv_4sp__C_gk.parquet")
    C = C[C.var_name.isin(nm.var_name)]
    named_types = {t for _, ts, _ in CT_GROUPS if ts for t in ts}
    W = C.groupby("cell_type").weight_k.first()
    fig, ax = plt.subplots(figsize=(3.25, 5.05), layout="constrained")
    for y, r in enumerate(nm.itertuples()):
        c = C[C.var_name == r.var_name].set_index("cell_type").C_gk_meanz
        pos = neg = 0.0
        for name, ts, col in CT_GROUPS:
            v = c[[t for t in c.index if t not in named_types]].sum() if ts is None else c.reindex(ts).fillna(0).sum()
            if v >= 0:
                ax.barh(y, v, left=pos, color=col, height=0.72, lw=0)
                pos += v
            else:
                ax.barh(y, v, left=neg, color=col, height=0.72, lw=0)
                neg += v
        ax.plot([c.sum()], [y], marker="|", ms=5, color=cm.INK, mew=0.9, zorder=4)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(nm)))
    ax.set_yticklabels([f"{r.label}  ({r.delta_alpha:+.2f})" for r in nm.itertuples()], fontsize=5.6)
    ax.set_ylim(len(nm) - 0.4, -0.6)
    ax.set_xlabel("contribution to count evidence $S_g$, by cell type\n(liver 4 species; tick: total $S_g$)",
                  fontsize=6.5)
    hs = [Patch(color=col, label=f"{name} (weight {W.reindex(ts).sum():.2f})" if ts else name)
          for name, ts, col in CT_GROUPS]
    fig.legend(handles=hs, loc="outside lower center", ncol=2, fontsize=5.6, handlelength=1.0,
               columnspacing=0.8)
    ax.text(0.01, 0.01, f"(Δα in parentheses)   {DAGGER} no margin support", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=5.6, color=cm.MUTED)
    return fig


# The seven hepatocyte catabolic enzymes that the text names as conserved proteins used at different levels.
ENZYMES = ["ALAS1", "ACOX1", "PCK1", "PDK4", "HAL", "GOT1", "ALDOB"]
ENZYME_PANEL = "main_validation/celltype_drivers_enzymes"


def panel_drivers_enzymes() -> plt.Figure:
    """Main validation figure, panel c (``main_validation/celltype_drivers_enzymes``, 159 x 120 bp): the cell types
    behind the count evidence of seven hepatocyte enzymes. It is ``panel_drivers`` for the genes of ``ENZYMES``, drawn small.

    What it draws. One bar per gene on four-species liver: the count evidence S_g split exactly into the
    contributions of the cell types (posterior-mean z), stacked from zero, negative parts to the left and positive
    parts to the right, in the colors of the liver cell types of ``figures/style.py``; the black tick is the sum, the
    count evidence of the gene. For these seven genes nearly all of it comes from hepatocytes.

    Reads ``results/analysis/T2_5/liv_4sp__C_gk.parquet`` and nothing else (columns ``symbol``, ``cell_type``,
    ``C_gk_meanz``): the rows of the seven genes, summed within the cell-type groups of ``CT_GROUPS`` as
    ``panel_drivers`` does. No other value is computed. The panel stops unless hepatocytes carry the largest part
    of every gene, which is what the text says.

        python -m analysis.tasks.T8_4_figures.fig4_additions --panel main_validation/celltype_drivers_enzymes     # in chimera_env, from the repository root
    """
    C = pd.read_parquet(T25 / "liv_4sp__C_gk.parquet")
    C = C[C.symbol.isin(ENZYMES)]
    assert set(C.symbol) == set(ENZYMES), f"liv_4sp__C_gk.parquet lacks {sorted(set(ENZYMES) - set(C.symbol))}"
    named_types = {t for _, ts, _ in CT_GROUPS if ts for t in ts}
    W, H = 159.0, 120.0
    left, right, bottom, top = 28.0, 4.0, 42.0, 31.0
    fig = plt.figure(figsize=(cm.style.inch(W), cm.style.inch(H)))
    ax = fig.add_axes([left / W, bottom / H, (W - left - right) / W, (H - bottom - top) / H])
    for y, sym in enumerate(ENZYMES):
        c = C[C.symbol == sym].set_index("cell_type").C_gk_meanz
        parts = [(c[[t for t in c.index if t not in named_types]].sum() if ts is None else c.reindex(ts).fillna(0).sum(), col) for _, ts, col in CT_GROUPS]
        assert abs(parts[0][0]) == max(abs(v) for v, _ in parts), f"{sym}: hepatocytes do not carry the largest part of the count evidence"
        pos = neg = 0.0
        for v, col in parts:
            if v >= 0:
                ax.barh(y, v, left=pos, color=col, height=0.7, lw=0)
                pos += v
            else:
                ax.barh(y, v, left=neg, color=col, height=0.7, lw=0)
                neg += v
        ax.plot([c.sum()], [y], marker="|", ms=5.5, color=cm.INK, mew=0.9, zorder=4)
    ax.axvline(0, color=cm.MUTED, lw=0.6, zorder=0)
    ax.set_yticks(range(len(ENZYMES)))
    ax.set_yticklabels(ENZYMES, fontsize=5.5)
    ax.set_ylim(len(ENZYMES) - 0.45, -0.55)
    ax.set_xlim(-0.42, 0.06)
    ax.set_xticks([-0.4, -0.3, -0.2, -0.1, 0])
    ax.set_xticklabels(["−0.4", "−0.3", "−0.2", "−0.1", "0"])
    ax.tick_params(labelsize=5.5, length=2, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("contribution to count evidence $S_g$ (tick: total)", fontsize=5.5, labelpad=1.5)
    fig.text(2.0 / W, (H - 1.2) / H, "Cell types behind\nthe count evidence", ha="left", va="top", fontsize=6.5, fontweight="bold", color=cm.INK, linespacing=1.12)
    fig.text(2.0 / W, (H - 22.6) / H, "seven hepatocyte enzymes, liver 4 species", ha="left", va="baseline", fontsize=5.5, color=cm.INK)
    hs = [Patch(color=col, label=name) for name, _, col in CT_GROUPS]
    fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(2.0 / W, 1.0 / H), ncol=2, fontsize=5.2, handlelength=0.9, handleheight=0.7, columnspacing=0.8, handletextpad=0.35,
               labelspacing=0.25, frameon=False, borderpad=0, borderaxespad=0)
    return fig


MS_COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
TISSUES = ["liver", "testis", "bone marrow", "kidney", "islet", "eye"]


def cohort_rows() -> pd.DataFrame:
    """Every cohort of H1 in tissue order: cohort, tissue, label, first kept run."""
    runs = pd.read_csv(cm.RES / "H1" / "runs.tsv", sep="\t")
    runs = runs[runs.kept.astype(str) == "True"].sort_values("seed").groupby("cohort", sort=False).first().reset_index()
    table = pd.read_csv(cm.RES / "H1" / "cohort_table.csv").set_index("cohort")
    labels = table.label
    runs = runs.set_index("cohort").reindex(table.index).reset_index()
    runs["label"] = [cm.style.cohort_name(c if c in cm.COHORT else labels[c]) for c in runs.cohort]
    runs = runs[runs.tissue.isin(TISSUES)]  # H1 also lists the C1 cohorts (cortex, lung, muscle, tumor); the margin-support panel keeps the 16 until its layout is extended
    runs["order"] = [TISSUES.index(t) for t in runs.tissue]
    return runs.sort_values("order", kind="stable").reset_index(drop=True)


def _run_tail(c: str, run: str) -> pd.DataFrame:
    """One run: Delta-alpha against the prior it anchors on, S_g_ema, margin flag of T2.5."""
    sys.path.insert(0, str(cm.REPO / "analysis" / "tasks" / "T2_5_celltype_margin"))
    import margin as MG

    if c in MS_COHORTS:
        g = pd.read_parquet(T25 / f"{c}__genes.parquet")
        return g[g["tail"]][["delta_alpha", "margin_support"]]
    with np.load(cm.REPO / "training" / "runs" / run / "final_state.npz", allow_pickle=True) as z:
        da, S = z["alpha"] - z["prior"], z["S_g_ema"]
    t = pd.DataFrame(dict(delta_alpha=da, margin_support=MG.margin_flag(da, S)))
    return t[np.abs(da) > MG.TAIL]


def panel_margin_support() -> plt.Figure:
    h = pd.read_csv(T25 / "headline_margin_support.csv")
    h = h[(h.run_label == h.cohort) & (h.group_key == "cell_type")].set_index("cohort")
    rows = cohort_rows()
    # Exact size and axes, shared with the sequence-axis panel below it.
    pitch = 12.2 / 72
    H = pitch * len(rows) + 0.66
    fig = plt.figure(figsize=(3.4, H))
    ax = fig.add_axes([1.06 / 3.4, 0.48 / H, 2.3 / 3.4, pitch * len(rows) / H])
    source = []
    for i, r0 in enumerate(rows.itertuples()):
        c = r0.cohort
        t = _run_tail(c, r0.run)
        neg, pos = t[t.delta_alpha < 0], t[t.delta_alpha > 0]
        for k, (sub, col, name) in enumerate([(neg, "#0072B2", "negative tail (Δα < −0.05)"),
                                               (pos, "#D55E00", "positive tail (Δα > +0.05)")]):
            y = i + (k - 0.5) * 0.46
            source.append(dict(cohort=c, run=r0.run, tail=name[:8], n=len(sub),
                               frac=sub.margin_support.mean() if len(sub) else np.nan))
            if len(sub) == 0:
                ax.text(0.01, y, "no genes", va="center", fontsize=5.9, color=cm.MUTED)
                continue
            f = sub.margin_support.mean()
            ax.barh(y, f, height=0.42, color=col, lw=0, label=name if not ax.containers or len(ax.containers) < 2 else None)
            ax.text(f + 0.01, y, f"{f:.2f}  (n = {len(sub):,})", va="center", fontsize=5.9, color=cm.INK)
        if c in MS_COHORTS:
            assert abs(h.loc[c, "frac_tail_with_margin_support"] - t.margin_support.mean()) < 1e-9
    pd.DataFrame(source).to_csv(cm.RES / "T8_4" / "fig4_margin_support_source.csv", index=False)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows.label, fontsize=5.8)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    for i in np.where(np.diff(rows.order))[0]:
        ax.axhline(i + 0.5, color=cm.RULE, lw=0.4)
    ax.set_xlim(0, 1.5)  # room for the numbers beside the bars
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("fraction of tail genes whose Δα is backed by\ncount evidence of the same sign (margin support)",
                  fontsize=6, labelpad=1)
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5.9, handlelength=1.0, frameon=False,
               borderpad=0, columnspacing=0.8, handletextpad=0.3)
    return fig


PAIRS = [("liv_hmonkey", "test_hmac", "H+Monkey vs H+Macaque", "liver Human+Monkey vs testis Human+Macaque"),
         ("liv_hmouse", "test_hmouse", "H+Mouse vs H+Mouse", "liver Human+Mouse vs testis Human+Mouse")]


def _centred(c: str) -> pd.DataFrame:
    g = pd.read_parquet(T25 / f"{c}__genes.parquet").dropna(subset=["hsapiens_gene_id"])
    g = g.drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["da"] = g.delta_alpha_raw - g.delta_alpha_raw.median()
    return g


def panel_cross_tissue() -> plt.Figure:
    ref = pd.read_csv(T23 / "tissue_comparison.csv")
    ref = ref[ref.version == "raw_centred"].set_index("pair")
    fig, axes = plt.subplots(1, 2, figsize=(6.75, 1.95), layout="constrained")
    for ax, (lc, tc, key, title) in zip(axes, PAIRS):
        L, T = _centred(lc), _centred(tc)
        ix = L.index.intersection(T.index)
        x, y = L.loc[ix, "da"], T.loc[ix, "da"]
        rho = spearmanr(x, y).statistic
        r = ref.loc[key]
        assert abs(rho - r.gene_rho) < 1e-9 and len(ix) == r.n_genes, (key, rho, r.gene_rho, len(ix))
        cm.fd.density_scatter(ax, x, y, s=1.4, alpha=0.2, color="#888888")
        mt = L.loc[ix, "symbol"].astype(str).str.startswith("MT-")
        ax.scatter(x[mt], y[mt], s=14, color="#5B3E80", edgecolor="white", lw=0.4, zorder=4,
                   label="mitochondrially encoded genes")
        ax.axhline(0, color=cm.MUTED, lw=0.5)
        ax.axvline(0, color=cm.MUTED, lw=0.5)
        lim = 1.08 * max(np.nanquantile(np.abs(np.r_[x, y]), 0.995), np.abs(np.r_[x[mt], y[mt]]).max())
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("Δα in liver (raw prior, centered)", fontsize=6.5)
        ax.set_title(title, fontsize=6.3)
        cm.fd.annotate_stat(ax, f"Spearman ρ = {rho:+.3f}, n = {len(ix):,}\nsign differs in {r.flip_rate_observed:.0%}"
                                f" of genes\n(expected if unrelated: {r.flip_rate_expected:.0%})", "lower right")
    axes[0].set_ylabel("Δα in testis (raw prior, centered)", fontsize=6.5)
    axes[0].legend(loc="upper left", fontsize=5.5, handletextpad=0.2, borderaxespad=0.1)
    return fig


def main(only=None) -> list:
    """Draw the panels of this script; ``only``: panel names ("family/name") to draw just those."""
    cm.setup()
    out = []
    for name, fn in [("fig4c_prior_landscape", panel_landscape), ("fig4_celltype_drivers", panel_drivers),
                     ("fig4_margin_support", panel_margin_support), ("fig4_cross_tissue", panel_cross_tissue)]:
        if not only or cm.style.panel_for(name) in only:
            out.append(cm.save(fn(), name, exact=name == "fig4_margin_support"))
    if not only or ENZYME_PANEL in only:  # at its exact size; the preview goes to figures/_work/panels/
        out.append(cm.style.save_panel(panel_drivers_enzymes(), ENZYME_PANEL, exact=True))
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--panel", nargs="+", metavar="NAME", help="draw only these panels (family/name, as in figures/panels/)")
    for p in main(ap.parse_args().panel):
        print(p, cm.check_pdf(p))
