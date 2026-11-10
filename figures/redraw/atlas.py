"""Panels of the protein-space atlas figures of the Supplement, redrawn at print size from stored model outputs.

    supp_atlas_archetype_trajectory/trajectory.pdf   mean Δα of the three gene archetypes at the stored training epochs

    supp_atlas_spatial_autocorrelation/coherence_{alpha,delta_alpha,prior}.pdf
                                                     kNN coherence of α, Δα and the prior among protein-space neighbors,
                                                     four liver cohorts, with two nulls
    supp_atlas_spatial_autocorrelation/moran_{alpha,delta_alpha,prior}.pdf
                                                     Moran's I of the same three quantities on the same neighbor graph
    supp_atlas_spatial_autocorrelation/legend.pdf    the one legend of the six bar panels

    supp_atlas_archetype_enrichment/categories.pdf         35 gene categories x 3 archetypes, Pearson residuals and stars
    supp_atlas_archetype_enrichment/broad_categories.pdf   the same for the 10 broad categories

Inputs. Four-species liver run: ``alpha_snapshots/alpha_epoch_*.npz`` (α and prior every five epochs) and
``tables/gene_archetypes.csv`` (the archetype and the gene category of every gene, as saved by the archetype analysis;
no clustering here); the category vocabularies ``notebooks/_gene_categories.py`` and ``notebooks/_gene_categories_broad.py``.
Four liver runs: ``final_state.npz`` (α and prior) and the 64 ESM-2 components of every gene (``varm["gene_features"]``
of the cohort's data object). The neighbor statistics and their nulls follow sections B.1 and B.2 of
``notebooks/05_esm_alpha_geometry.ipynb`` draw for draw (about one minute per cohort); the enrichment
follows section C of ``notebooks/_gene_archetypes_builder.py``.
The six bar panels are pages with one axes box at fixed margins, the two heat maps are pages with the same row pitch
and the same bottom margin, so that they line up when composed with ``Panel(..., ink=False)``.

    PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m figures.redraw.atlas [panel ...]
"""
from __future__ import annotations

import sys
from functools import lru_cache

import numpy as np
import pandas as pd

from figures.redraw import _common as C

ARCHETYPE_COLOR = ("#0072B2", "#D55E00", "#009E73")   # A0, A1, A2 as in the main atlas figure
EPOCH_CUTOFF = 100

AUTOCORRELATION, ENRICHMENT = "supp_atlas_spatial_autocorrelation", "supp_atlas_archetype_enrichment"
TOTAL, GAP = 486.0, 7.0                                # bp: text width, gap between panels in a row

# Neighbor statistics (sections B.1 and B.2 of notebook 05)
LIVER = ("liver_hmonkey", "liver_hmouse", "liver_hzeb", "liver_4sp")
QUANTITY = {"alpha": "α", "delta_alpha": "Δα", "prior": "protein prior π"}      # stored value -> heading of its panel
STATISTIC = {"coherence": "kNN coherence (Spearman ρ)", "moran": "Moran's $I$"}   # y label of a row of panels
K_NEIGHBORS = 15           # neighbors of a gene in the space of the 64 ESM-2 components
N_SHUFFLE = 50             # null 1: the values shuffled among the genes, the neighbor graph kept
N_RANDOM = 5               # null 2: the values kept, the graph built on Gaussian random features of the same size
RNG_SEED = 0
COHERENCE_RANGE, LINEAR_BELOW = (-0.005, 1.0), 0.01    # symmetric-log axis of the coherence row, as on the first version
MORAN_RANGE = (-0.02, 0.40)                            # one linear axis for the three Moran panels
BAR_WIDTH = 0.27
BAR = {"real": dict(color="#bbbbbb", edgecolor="#444444", linewidth=0.5),
       "shuffle": dict(color="#dddddd", edgecolor="#666666", linewidth=0.4, hatch="/////"),
       "random": dict(color="#eeeeee", edgecolor="#666666", linewidth=0.4, hatch="xxxx")}
BAR_LABEL = {"real": "ESM-2 neighbors (real)", "shuffle": f"null: values shuffled ({N_SHUFFLE} draws)",
             "random": f"null: random features ({N_RANDOM} draws)"}
CUT_LABEL = "bar ends below the axis"
Y_LABEL, LEFT, RIGHT, TOP, BOTTOM, AXES_HEIGHT = 11.0, 21.0, 2.0, 13.0, 33.0, 100.0   # bp, margins of a bar panel

# Archetype by category (section C of the archetype notebook)
MIN_GENES = 10             # a category is a row when it has at least this many genes
Q = 0.05                   # Benjamini-Hochberg level of the stars
N_ARCHETYPES = 3
PITCH, CELL = 11.5, 20.0   # bp: height of a row, width of a cell
MAP_TOP, MAP_BOTTOM = 13.0, 19.0                       # bp above the first row (column headings) and below the last (axis label)
MAP_PAGE = 196.0           # bp, width of either heat-map page
ROW_SIZE, STAR_SIZE = 5.8, 8.0                         # pt: row names, stars
COLORBAR_ROWS = 8          # length of a color bar, in rows


def archetypes() -> np.ndarray:
    """Archetype (0, 1, 2) of every gene of the four-species liver run, in the gene order of the stored model outputs."""
    d = C.run_dir("liver_4sp")
    t = d / "tables" / "gene_archetypes.csv"
    C.need(t)
    tab = pd.read_csv(t)
    final = np.load(d / "final_state.npz")
    if len(tab) != len(final["alpha"]) or not np.allclose(tab["alpha"].to_numpy(), final["alpha"], atol=1e-6):
        raise SystemExit("gene_archetypes.csv is not in the gene order of final_state.npz")
    return tab["archetype"].to_numpy()


def archetype_trajectory() -> list:
    import matplotlib.pyplot as plt

    C.design()
    d = C.run_dir("liver_4sp")
    snaps = sorted((d / "alpha_snapshots").glob("alpha_epoch_*.npz"))
    if not snaps:
        raise SystemExit(f"no snapshots in {d / 'alpha_snapshots'}")
    arch = archetypes()
    epochs, traj = [], []
    for p in snaps:
        s = np.load(p)
        da = (s["alpha"] - s["prior"]).astype(np.float32)
        epochs.append(int(s["epoch"]))
        traj.append([float(da[arch == a].mean()) for a in range(3)])
    epochs, traj = np.array(epochs), np.array(traj)
    keep = epochs <= EPOCH_CUTOFF
    fig, ax = plt.subplots(figsize=(3.3, 2.5), layout="constrained")
    for a in range(3):
        ax.plot(epochs[keep], traj[keep, a], color=ARCHETYPE_COLOR[a], linewidth=1.0, marker="o", markersize=2.0,
                label=f"A{a}  (n = {int((arch == a).sum()):,})")
    ax.axhline(0, color="#888888", linewidth=0.5, linestyle="--")
    ax.set_xlim(0, EPOCH_CUTOFF)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean Δα of the archetype's genes")
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.56), frameon=False, title="archetype", title_fontsize=C.SIZE["legend"])
    print("archetype means at the last shown epoch:", np.round(traj[keep][-1], 4))
    return [C.save_panel(fig, "supp_atlas_archetype_trajectory", "trajectory")]


# ---------------------------------------------------------------------------------------------------------------
# Coherence of α, Δα and the prior among protein-space neighbors (six bar panels and their legend)
# ---------------------------------------------------------------------------------------------------------------

def protein_space(cohort: str) -> dict:
    """The 64 ESM-2 components of every gene (the first 64 gate inputs stored with the data), α, the prior and Δα of one run."""
    import anndata as ad

    path = C.DATA / C.ADATA[cohort]
    C.need(path)
    final = np.load(C.run_dir(cohort) / "final_state.npz")
    a = ad.read_h5ad(path, backed="r")
    features = np.asarray(a.varm["gene_features"]).astype(np.float32)
    a.file.close()
    if len(features) != len(final["alpha"]):
        raise SystemExit(f"{path.name} has {len(features)} genes, the run has {len(final['alpha'])}")
    return dict(esm=features[:, :64].copy(), alpha=final["alpha"].astype(np.float32), prior=final["prior"].astype(np.float32),
                delta_alpha=(final["alpha"] - final["prior"]).astype(np.float32))


NEIGHBOR_THREADS = 20


def neighbors(features: np.ndarray) -> np.ndarray:
    """Indices of the K_NEIGHBORS nearest genes of every gene (Euclidean): the first of the K_NEIGHBORS + 1 found is dropped.

    In the four-species cohort 91 genes have the same 64 components as another gene, so which of the genes at distance
    zero count among the neighbors is an arbitrary choice, and that choice follows from how the search is split over
    threads. The values of the paper are those of a search on 20 threads, which this call repeats (they are reproduced
    to five decimals). With 1 to 8 threads, or on a machine with fewer than 20 cores, the four-species values differ
    from them by at most 0.0008; the other three cohorts have no such ties and do not change.
    """
    from sklearn.neighbors import NearestNeighbors
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=NEIGHBOR_THREADS):
        _, index = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, algorithm="auto").fit(features).kneighbors(features)
    return index[:, 1:]


def coherence(values: np.ndarray, graph: np.ndarray) -> float:
    """Spearman correlation between the value of a gene and the mean value of its neighbors."""
    from scipy import stats

    rho, _ = stats.spearmanr(values, values[graph].mean(axis=1))
    return float(rho) if np.isfinite(rho) else 0.0


def moran(values: np.ndarray, graph: np.ndarray) -> float:
    """Moran's I with row-normalized weights (every gene gives 1 / K_NEIGHBORS to each of its neighbors)."""
    from scipy import sparse

    n, k = graph.shape
    weights = sparse.coo_matrix((np.ones(n * k, dtype=np.float64) / k, (np.repeat(np.arange(n), k), graph.ravel())), shape=(n, n)).tocsr()
    x = values - values.mean()
    total = float((x ** 2).sum())
    return float((x * (weights @ x)).sum()) / total if total > 0 else 0.0


@lru_cache(maxsize=1)
def autocorrelation_table() -> pd.DataFrame:
    """Both statistics for α, Δα and the prior of the four liver runs: the real value and the two nulls.

    The random draws are those of the notebook: a generator started from RNG_SEED for every cohort and quantity
    (coherence) or once per cohort for its three quantities in turn (Moran's I), giving first the N_SHUFFLE shuffles
    and then the N_RANDOM random feature matrices. A null is summarized by the median of its draws and their 97.5% quantile.
    """
    def row(statistic, cohort, quantity, real, shuffled, random):
        return dict(statistic=statistic, cohort=cohort, quantity=quantity, real=real,
                    shuffle_median=float(np.median(shuffled)), shuffle_q975=float(np.quantile(shuffled, 0.975)),
                    random_median=float(np.median(random)), random_q975=float(np.quantile(random, 0.975)))

    rows = []
    for cohort in LIVER:
        d = protein_space(cohort)
        graph = neighbors(d["esm"])
        for statistic, measure in (("coherence", coherence), ("moran", moran)):
            rng = np.random.default_rng(RNG_SEED)
            for quantity in QUANTITY:
                if statistic == "coherence":
                    rng = np.random.default_rng(RNG_SEED)
                values = d[quantity]
                shuffled = [measure(rng.permutation(values), graph) for _ in range(N_SHUFFLE)]
                random = [measure(values, neighbors(rng.standard_normal(d["esm"].shape).astype(np.float32))) for _ in range(N_RANDOM)]
                rows.append(row(statistic, cohort, quantity, measure(values, graph), shuffled, random))
        print(f"{cohort}: {len(d['alpha']):,} genes, neighbor statistics done", flush=True)
    return pd.DataFrame(rows)


def _bar_panel(statistic: str, quantity: str, table: pd.DataFrame, first: bool):
    """One bar panel: the four liver cohorts, real value and two nulls (median, error bar to the 97.5% quantile)."""
    import matplotlib.pyplot as plt

    t = table[(table["statistic"] == statistic) & (table["quantity"] == quantity)].set_index("cohort").loc[list(LIVER)]
    left = LEFT + (Y_LABEL if first else 0.0)
    axes_width = (TOTAL - 2 * GAP - Y_LABEL) / 3 - LEFT - RIGHT
    width, height = left + axes_width + RIGHT, TOP + AXES_HEIGHT + BOTTOM
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([left / width, BOTTOM / height, axes_width / width, AXES_HEIGHT / height])
    x = np.arange(len(t))
    error = dict(ecolor=C.INK, error_kw=dict(linewidth=0.5, capsize=1.5, capthick=0.5))
    ax.bar(x - BAR_WIDTH, t["real"], width=BAR_WIDTH, **BAR["real"])
    for null, offset in (("shuffle", 0.0), ("random", BAR_WIDTH)):
        median, upper = t[f"{null}_median"].to_numpy(), t[f"{null}_q975"].to_numpy()
        ax.bar(x + offset, median, width=BAR_WIDTH, yerr=[np.zeros(len(t)), (upper - median).clip(min=0)], **BAR[null], **error)
    ax.axhline(0, color="#888888", linewidth=0.5)
    cut = []
    if statistic == "coherence":
        low, high = COHERENCE_RANGE
        ax.set_yscale("symlog", linthresh=LINEAR_BELOW)
        ax.set_ylim(low, high)
        ax.set_yticks([0, 0.01, 0.1, 1], ["0", "0.01", "0.1", "1"])
        ax.minorticks_off()
        for null, offset in (("shuffle", 0.0), ("random", BAR_WIDTH)):       # a null bar that ends below the axis: a mark under it
            for i, v in enumerate(t[f"{null}_median"]):
                if v < low:
                    ax.plot([i + offset], [0], marker="v", markersize=2.6, color=C.INK, markeredgewidth=0, clip_on=False,
                            transform=ax.get_xaxis_transform() + _shift(fig, -2.4), zorder=6)
                    cut.append(f"{null} null of {quantity}, {t.index[i]}: {v:+.4f}")
    else:
        ax.set_ylim(*MORAN_RANGE)
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4], ["0", "0.1", "0.2", "0.3", "0.4"])
    ax.set_xticks(x, [C.COHORT_SHORT[c].replace("+", "+\n") for c in t.index], fontsize=C.SIZE["text"], linespacing=1.1)
    ax.tick_params(axis="x", pad=4.5)
    ax.set_xlabel("liver cohort")
    ax.xaxis.set_label_coords(0.5, -(BOTTOM - 9.5) / AXES_HEIGHT)
    if first:
        ax.set_ylabel(STATISTIC[statistic])
        ax.yaxis.set_label_coords(-(left - 9.5) / axes_width, 0.5)                # the label fills the band of Y_LABEL bp
    ax.set_title(QUANTITY[quantity], fontsize=C.SIZE["title"], pad=3.5)
    if (t[["real", "shuffle_q975", "random_q975"]].to_numpy() > ax.get_ylim()[1]).any() or (statistic == "moran" and
            (t[["real", "shuffle_median", "random_median"]].to_numpy() < MORAN_RANGE[0]).any()):
        raise SystemExit(f"{statistic} of {quantity}: a bar leaves the axis")
    return fig, cut


def _shift(fig, points: float):
    from matplotlib.transforms import ScaledTranslation

    return ScaledTranslation(0, points / 72, fig.dpi_scale_trans)


def spatial_autocorrelation() -> list:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    C.design()
    mpl.rcParams.update({"hatch.linewidth": 0.3})
    table = autocorrelation_table()
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    out, cut = [], []
    for statistic in STATISTIC:
        for i, quantity in enumerate(QUANTITY):
            fig, c = _bar_panel(statistic, quantity, table, first=i == 0)
            cut += c
            out.append(C.save_panel(fig, AUTOCORRELATION, f"{statistic}_{quantity}", exact=True))
    print("null bars that end below the coherence axis (marked under the axis):\n   " + "\n   ".join(cut or ["none"]))
    fig = plt.figure(figsize=(TOTAL / 72, 11.0 / 72))
    handles = [Patch(label=BAR_LABEL[k], facecolor=s["color"], **{p: v for p, v in s.items() if p != "color"}) for k, s in BAR.items()]
    handles.append(Line2D([], [], linestyle="none", marker="v", markersize=2.6, color=C.INK, markeredgewidth=0, label=CUT_LABEL))
    fig.legend(handles=handles, loc="center", ncols=len(handles), frameon=False, fontsize=C.SIZE["legend"], handlelength=1.7,
               handleheight=0.9, handletextpad=0.5, columnspacing=1.5, borderpad=0, borderaxespad=0)
    out.append(C.save_panel(fig, AUTOCORRELATION, "legend", exact=True))
    return out


# ---------------------------------------------------------------------------------------------------------------
# Gene categories enriched in each archetype (two heat maps)
# ---------------------------------------------------------------------------------------------------------------

def text_width(strings, size: float) -> float:
    """Width in bp of the longest of some strings in the figure font."""
    import matplotlib as mpl
    from matplotlib.font_manager import FontProperties
    from matplotlib.textpath import TextPath

    prop = FontProperties(family=mpl.rcParams["font.family"])
    return max(TextPath((0, 0), s, size=size, prop=prop).get_extents().width for s in strings)


@lru_cache(maxsize=None)
def enrichment(vocabulary: str) -> dict:
    """Archetype by category table of the four-species liver run for the "narrow" or the "broad" vocabulary.

    Counts (archetypes x categories with at least MIN_GENES genes, in the order of the vocabulary), the global chi-square
    test, the Pearson residual of every cell, and its one-sided Fisher exact test (more genes than expected) with the
    Benjamini-Hochberg correction over the whole table.
    """
    from scipy import stats
    from statsmodels.stats.multitest import multipletests

    C.design()                                      # puts the notebooks folder on the path
    import _gene_categories as gc                   # narrow vocabulary: order of the categories
    import _gene_categories_broad as gcb            # broad vocabulary: narrow category -> broad category

    genes = pd.read_csv(C.run_dir("liver_4sp") / "tables" / "gene_archetypes.csv", usecols=["category"])
    genes["archetype"] = archetypes()
    if vocabulary == "broad":
        genes["category"] = genes["category"].map(lambda c: gcb.NARROW_TO_BROAD.get(c, gcb.OTHER))
        order = [c for c in gcb.CATEGORY_ORDER_BROAD if c != gcb.OTHER]
    else:
        order = [c for c in gc.CATEGORY_ORDER_EXTENDED if c != gc.OTHER]
    named = genes[genes["category"].isin(order)]
    counts = pd.crosstab(named["archetype"], named["category"]).reindex(index=range(N_ARCHETYPES), columns=order, fill_value=0)
    counts = counts.loc[:, counts.sum(axis=0) >= MIN_GENES]
    chi2, p_global, dof, expected = stats.chi2_contingency(counts.to_numpy())
    residual = (counts.to_numpy() - expected) / np.sqrt(expected.clip(min=1.0))
    total = int(counts.to_numpy().sum())
    p = np.ones(counts.shape)
    for a in range(counts.shape[0]):
        in_archetype = int(counts.iloc[a].sum())
        for c in range(counts.shape[1]):
            both, in_category = int(counts.iloc[a, c]), int(counts.iloc[:, c].sum())
            p[a, c] = stats.fisher_exact([[both, in_archetype - both], [in_category - both, total - in_category - in_archetype + both]],
                                         alternative="greater")[1]
    q = multipletests(p.ravel(), method="fdr_bh")[1].reshape(p.shape)
    return dict(counts=counts, residual=residual.T, q=q.T, chi2=float(chi2), dof=int(dof), p_global=float(p_global), categories=len(order))


def _heatmap(vocabulary: str, name: str, colorbar_at_top: bool):
    """Categories (rows) by archetypes (columns): Pearson residual as color, a star where q < Q; its own color bar."""
    import matplotlib.pyplot as plt

    C.design()
    e = enrichment(vocabulary)
    residual, star = e["residual"], e["q"] < Q
    n = len(residual)
    if vocabulary == "narrow":
        names = C.category_names()
        labels = [C.category(c, names) for c in e["counts"].columns]
    else:
        labels = list(e["counts"].columns)                                   # the broad names are words already
    left = 2.0 + text_width(labels, ROW_SIZE) + 3.0
    width, height = MAP_PAGE, MAP_TOP + PITCH * n + MAP_BOTTOM
    if left + N_ARCHETYPES * CELL + 44.0 > width:
        raise SystemExit(f"{name}: the row names need {left:.0f} bp, the page of {width:.0f} bp is too narrow")
    fig = plt.figure(figsize=(width / 72, height / 72))
    ax = fig.add_axes([left / width, MAP_BOTTOM / height, N_ARCHETYPES * CELL / width, PITCH * n / height])
    limit = float(np.abs(residual).max() * 1.05 + 1e-6)                      # color range of this heat map
    im = ax.imshow(residual, cmap="RdBu_r", aspect="auto", vmin=-limit, vmax=limit, interpolation="nearest")
    for i, j in zip(*np.nonzero(star)):                                      # the glyph sits high in its line: moved down to the cell center
        ax.annotate("*", (j, i), xytext=(0, -0.27 * STAR_SIZE), textcoords="offset points", ha="center", va="center", fontsize=STAR_SIZE,
                    color=C.INK if abs(residual[i, j]) < 0.55 * limit else "#eeeeee")
    ax.xaxis.tick_top()
    ax.set_xticks(range(N_ARCHETYPES), [f"A{a}" for a in range(N_ARCHETYPES)], fontsize=C.SIZE["title"])
    ax.set_yticks(range(n), labels, fontsize=ROW_SIZE)
    ax.tick_params(length=0, pad=2.5)
    ax.tick_params(axis="x", pad=2.0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("archetype", labelpad=3.5)
    bar = PITCH * min(COLORBAR_ROWS, n)
    bottom = MAP_BOTTOM + (PITCH * n - bar if colorbar_at_top else (PITCH * n - bar) / 2)
    cax = fig.add_axes([(left + N_ARCHETYPES * CELL + 8.0) / width, bottom / height, 6.0 / width, bar / height])
    cb = fig.colorbar(im, cax=cax, ticks=[-8, -4, 0, 4, 8])
    cb.ax.set_yticklabels(["−8", "−4", "0", "+4", "+8"])
    cb.set_label("Pearson χ² residual", fontsize=C.SIZE["tick"], labelpad=2.0)
    cb.ax.tick_params(labelsize=6.0, length=2.0, pad=1.5)
    cb.outline.set_linewidth(0.4)
    print(f"{name}: {e['categories']} categories, {n} with {MIN_GENES} genes or more; chi-square {e['chi2']:.1f}, dof {e['dof']}, "
          f"p {e['p_global']:.2e}; stars {int(star.sum())} of {star.size}; color range ±{limit:.3f}")
    for a in range(N_ARCHETYPES):
        print(f"   A{a}: " + ", ".join(c for c, s in zip(e["counts"].columns, star[:, a]) if s))
    return C.save_panel(fig, ENRICHMENT, name, exact=True)


def archetype_enrichment() -> list:
    return [_heatmap("narrow", "categories", colorbar_at_top=True), _heatmap("broad", "broad_categories", colorbar_at_top=False)]


PANELS = {"archetype_trajectory": archetype_trajectory, "spatial_autocorrelation": spatial_autocorrelation,
          "archetype_enrichment": archetype_enrichment}

if __name__ == "__main__":
    for name in sys.argv[1:] or list(PANELS):
        for p in PANELS[name]():
            print("wrote", p)
