"""Notebook helpers for Chimera paper-quality validation.

Loads a training run's artefacts (config.json, metrics.csv, alpha_snapshots,
final_state.npz, pretrain_checks.json, report.md, summary.json, AnnData), and
offers a small set of plotting helpers that produce figures at a consistent
Nature-style look.

Everything is read-only with respect to the run directory. No model reload is
required to render the figures in notebooks 10 and 11.

Usage (from a notebook cell)::

    from _paper_utils import load_run, set_paper_style
    set_paper_style()
    run = load_run("iter16-slow-classifier")
    alpha = run["final_state"]["alpha"]
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(os.environ.get("CHIMERA_ROOT") or Path(__file__).resolve().parents[1])
RUNS_ROOT = REPO_ROOT / "training" / "runs"
DATA_PROCESSED = REPO_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# Run loading
# ---------------------------------------------------------------------------

def _find_metrics_csv(run_dir: Path) -> Path | None:
    hits = list((run_dir / "lightning").rglob("metrics.csv"))
    return sorted(hits, key=lambda p: p.stat().st_mtime, reverse=True)[0] if hits else None


def load_run(run_name: str) -> dict[str, Any]:
    """Load a training run's artefacts into a dict.

    Keys (present when files exist):
        run_dir, run_name, config, pretrain_checks, metrics (pd.DataFrame),
        snapshots (list of dict-like npz), final_state (dict),
        summary (summary.json), report_md (str).
    """
    run_dir = RUNS_ROOT / run_name
    if not run_dir.exists():
        raise FileNotFoundError(f"Run not found: {run_dir}")
    out: dict[str, Any] = {"run_dir": run_dir, "run_name": run_name}

    for name, rel in [
        ("config", "config.json"),
        ("pretrain_checks", "pretrain_checks.json"),
        ("summary", "summary.json"),
    ]:
        p = run_dir / rel
        if p.exists():
            out[name] = json.loads(p.read_text())

    mp = _find_metrics_csv(run_dir)
    if mp is not None:
        out["metrics"] = pd.read_csv(mp)

    snap_dir = run_dir / "alpha_snapshots"
    if snap_dir.exists():
        snaps = sorted(snap_dir.glob("alpha_epoch_*.npz"))
        out["snapshots"] = [dict(np.load(p)) for p in snaps]

    fs = run_dir / "final_state.npz"
    if fs.exists():
        out["final_state"] = dict(np.load(fs, allow_pickle=False))

    rep = run_dir / "report.md"
    if rep.exists():
        out["report_md"] = rep.read_text()

    return out


def load_adata_for_run(run: dict[str, Any]):
    """Load the AnnData referenced by run['config']['adata_path']. Lazy import."""
    import anndata as ad
    cfg = run.get("config", {})
    adata_rel = cfg.get("adata_path")
    if adata_rel is None:
        raise ValueError("run config has no adata_path")
    return ad.read_h5ad(REPO_ROOT / adata_rel)


# ---------------------------------------------------------------------------
# Gene identity
#
# ``ortho_N`` numbers are table-local: the liver, testis and van Zyl ortholog
# tables number their rows independently, so the same ``ortho_N`` names
# different genes in different tables. Every lookup therefore states its
# cohort, and genes from different tables are joined on ``hsapiens_gene_id``.
# All lookups delegate to ``revision/common/gene_ids.py``.
# ---------------------------------------------------------------------------

def _gene_ids():
    import sys
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from revision.common import gene_ids
    return gene_ids


def cohort_for(name: str) -> str:
    """Ortholog-table cohort for a cohort name, a tissue, or a notebook registry key.

    ``pancreas``/``panc_*``/``pancreas_*`` are the legacy names of the liver
    cohorts. Unknown names raise, so no lookup silently falls back to a table.
    """
    gi = _gene_ids()
    n = str(name)
    if n in gi.TABLES:
        return n
    if n == "pancreas" or n.startswith(("panc_", "pancreas_", "liver_", "liv_")):
        return "liver"
    if n.startswith(("test_", "testis_")):
        return "testis"
    if n.startswith("van_zyl"):
        return "van_zyl"
    raise KeyError(f"no ortholog table for {name!r}; expected one of {sorted(gi.TABLES)}, "
                   "a tissue (liver/pancreas/testis) or a panc_*/test_* registry key")


def load_ortholog_table(cohort: str) -> pd.DataFrame:
    """The ortholog table that built ``cohort``, indexed by ortholog_group_id."""
    return _gene_ids().load_table(cohort_for(cohort)).copy()


def annotate_genes_with_names(var_names: Iterable[str], cohort: str) -> pd.Series:
    """Map adata.var_names like 'ortho_42' to the human gene symbol in ``cohort``'s table.

    Returns a pd.Series indexed by var_names with values = human gene name
    (fallback: the var name itself if the table has no symbol).
    """
    sym = _gene_ids().symbols(cohort_for(cohort), var_names)
    idx = sym.index
    return pd.Series(np.where(sym.isna(), idx.to_numpy(dtype=object), sym.to_numpy(dtype=object)),
                     index=idx, dtype=object)


def ensembl_ids(var_names: Iterable[str], cohort: str) -> pd.Series:
    """Human Ensembl gene ID per var_name in ``cohort``'s table: the cross-tissue join key."""
    return _gene_ids().ensembl(cohort_for(cohort), var_names)


def shared_genes(var_names_a: Iterable[str], cohort_a: str,
                 var_names_b: Iterable[str], cohort_b: str) -> tuple[np.ndarray, np.ndarray]:
    """Positions ``(ia, ib)`` of the genes two runs share, joined on hsapiens_gene_id.

    Valid within and across ortholog tables; never join runs on var_names
    across tissues. The order follows run A.
    """
    ea = ensembl_ids(var_names_a, cohort_a).to_numpy(dtype=object)
    eb = pd.Index(ensembl_ids(var_names_b, cohort_b).to_numpy(dtype=object))
    if pd.Index(ea).has_duplicates or eb.has_duplicates:
        raise ValueError("duplicated hsapiens_gene_id within a cohort; join is ambiguous")
    ib = eb.get_indexer(ea)
    ia = np.flatnonzero((ib >= 0) & pd.notna(ea))
    return ia, ib[ia]


# ---------------------------------------------------------------------------
# Plotting style
# ---------------------------------------------------------------------------

def set_paper_style():
    """Set matplotlib rcParams for Nature-ish look. Idempotent.

    Notebooks also import seaborn on top; rcParams are preserved across cells.
    """
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 220,
        "figure.constrained_layout.use": True,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "image.cmap": "viridis",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


# Qualitative colour set inspired by the Okabe-Ito palette (colour-blind friendly).
SPECIES_PALETTE = [
    "#0072B2",  # blue
    "#E69F00",  # orange
    "#009E73",  # green
    "#D55E00",  # vermilion
    "#CC79A7",  # magenta
    "#56B4E9",  # sky
    "#F0E442",  # yellow
    "#333333",  # black-ish
]


def species_colors(species_names: Iterable[str]) -> dict[str, str]:
    """Return a deterministic name → hex colour mapping."""
    return {s: SPECIES_PALETTE[i % len(SPECIES_PALETTE)] for i, s in enumerate(species_names)}


# ---------------------------------------------------------------------------
# Diagnostic computations (beyond the 32 gates)
# ---------------------------------------------------------------------------

def nonlinear_z_leakage_probe(
    z: np.ndarray,
    species_labels: np.ndarray,
    hidden: tuple[int, ...] = (64, 32),
    epochs: int = 200,
    lr: float = 5e-3,
    test_frac: float = 0.3,
    seed: int = 0,
) -> dict[str, float]:
    """Train a small 2-hidden-layer MLP on z → species and report heldout accuracy.

    Round2 Caveat 34 / cross-ref §1609: the linear adversarial classifier used in
    training only catches linearly-separable z-leakage. A nonlinear probe on a
    test split is the canonical post-hoc confirmation that z is species-free.

    Returns
    -------
    dict with keys: train_acc, test_acc, chance, margin_over_chance, K
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    rng = np.random.default_rng(seed)
    n = z.shape[0]
    idx = rng.permutation(n)
    n_te = int(test_frac * n)
    te, tr = idx[:n_te], idx[n_te:]

    z_t = torch.tensor(z, dtype=torch.float32)
    y_t = torch.tensor(species_labels, dtype=torch.long)
    K = int(y_t.max().item()) + 1

    layers: list[nn.Module] = []
    d = z.shape[1]
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU()]
        d = h
    layers += [nn.Linear(d, K)]
    mlp = nn.Sequential(*layers)
    opt = torch.optim.Adam(mlp.parameters(), lr=lr)

    for _ in range(epochs):
        opt.zero_grad()
        logits = mlp(z_t[tr])
        loss = F.cross_entropy(logits, y_t[tr])
        loss.backward()
        opt.step()

    mlp.eval()
    with torch.no_grad():
        p_tr = mlp(z_t[tr]).argmax(-1)
        p_te = mlp(z_t[te]).argmax(-1)
    tr_acc = float((p_tr == y_t[tr]).float().mean().item())
    te_acc = float((p_te == y_t[te]).float().mean().item())
    chance = 1.0 / K
    return {
        "train_acc": tr_acc,
        "test_acc": te_acc,
        "chance": chance,
        "margin_over_chance": te_acc - chance,
        "K": K,
    }


def per_dim_z_stats(z_mean: np.ndarray) -> pd.DataFrame:
    """Per-dim Var(z_mean), abs-mean, and a rough KL-to-prior approximation.

    The full per-dim KL needs z_logvar which we do not save post-hoc. A useful
    proxy is `0.5 * (Var(z_mean) + mean(z_mean)^2 - 1 - log Var(z_mean))` assuming
    negligible within-cell q variance — a *lower bound* on the true per-dim KL
    under a unit-Normal prior. Round2 OAI §904 flags any dim with KL < 0.01
    (collapse) and > 5.0 nats (species-encoding signature).
    """
    m = z_mean.mean(0)
    v = z_mean.var(0, ddof=0)
    # lower bound on KL: assumes logvar ≈ 0; in practice logvar < 0 slightly,
    # so this is close but not exact
    kl_lower = 0.5 * (v + m ** 2 - 1.0 - np.log(np.clip(v, 1e-12, None)))
    return pd.DataFrame({
        "dim": np.arange(z_mean.shape[1]),
        "mean": m,
        "var": v,
        "kl_lower_bound": kl_lower,
    })


def adv_tail_stability(
    metrics: pd.DataFrame,
    n_species: int,
    tail_frac: float = 0.2,
    min_tail: int = 10,
    tolerance: float = 0.05,
) -> dict[str, Any]:
    """Round2 OAI §905/§933 stopping criterion: adv_acc within ±tolerance of chance
    for the last `max(min_tail, tail_frac * max_epochs)` epochs.

    Our E6.2 is a *point* check; this is a *tail-stability* check that rounds
    out the OAI criterion.
    """
    if metrics is None or "adversarial_accuracy" not in metrics.columns:
        return {"pass": None, "note": "no adversarial_accuracy column"}
    acc = metrics["adversarial_accuracy"].dropna()
    if not len(acc):
        return {"pass": None}
    max_epoch = int(metrics["epoch"].max())
    n_tail = max(min_tail, int(tail_frac * max_epoch))
    tail = acc.tail(n_tail)
    chance = 1.0 / n_species
    frac_ok = float(((tail >= chance - tolerance) & (tail <= chance + tolerance)).mean())
    return {
        "tail_n": int(len(tail)),
        "chance": chance,
        "tolerance": tolerance,
        "mean_tail_acc": float(tail.mean()),
        "min_tail_acc": float(tail.min()),
        "max_tail_acc": float(tail.max()),
        "frac_within_tolerance": frac_ok,
    }


def saturation_stats(alpha: np.ndarray, low: float = 0.02, high: float = 0.98) -> dict[str, float]:
    """Tight saturation thresholds from Caveat 29 / BEG §1317.

    Spec wants frac(α < `low`) < 5% and frac(α > `high`) < 5% (a few-percent
    upper bound on saturation). E3.2 uses far looser thresholds; this is a
    supplementary tighter check.
    """
    alpha = np.asarray(alpha).ravel()
    return {
        "low_thresh": low,
        "high_thresh": high,
        "frac_below_low": float((alpha < low).mean()),
        "frac_above_high": float((alpha > high).mean()),
        "frac_saturated": float(((alpha < low) | (alpha > high)).mean()),
    }


def collect_alpha_trajectory(snapshots: list[dict]) -> dict[str, np.ndarray]:
    """Stack alpha snapshots into an (E, G) array along with epochs."""
    if not snapshots:
        return {"epochs": np.empty(0), "alpha": np.empty((0, 0))}
    epochs = np.array([int(s["epoch"][()] if s["epoch"].ndim == 0 else s["epoch"]) for s in snapshots])
    alpha = np.stack([s["alpha"] for s in snapshots], axis=0)
    corr = np.stack([s["correction"] for s in snapshots], axis=0) if "correction" in snapshots[0] else None
    out = {"epochs": epochs, "alpha": alpha}
    if corr is not None:
        out["correction"] = corr
    return out


def pretty_report_markdown(run: dict) -> str:
    """Return the report.md content, or a one-line summary if missing."""
    if "report_md" in run:
        return run["report_md"]
    if "summary" in run:
        s = run["summary"]["expectation_summary"]
        return f"Summary: {s['pass']}/{s['pass']+s['fail']+s['skip']} expectations pass"
    return "(no report available)"


def species_name_map(adata) -> dict[int, str]:
    """Return {species_idx: species_name} from an AnnData's species categorical."""
    if "species" not in adata.obs.columns:
        raise ValueError("adata.obs must have 'species'")
    codes = adata.obs["species"].astype("category")
    return {i: str(c) for i, c in enumerate(codes.cat.categories)}


# ---------------------------------------------------------------------------
# Fast integration metrics + UMAP (for notebook 12)
# ---------------------------------------------------------------------------

def lognorm_from_counts(adata, target_sum: float = 1e4, max_cells: int | None = None,
                        seed: int = 0) -> np.ndarray:
    """Return a dense log1p(CPM-normalised) matrix from an AnnData's raw counts.

    Chimera's AnnData stores raw integer counts in `.X`. For a "pre-integration"
    UMAP/baseline we need a library-size-normalised, log-transformed matrix.
    Optionally subsample cells.
    """
    from scipy import sparse
    X = adata.X
    rng = np.random.default_rng(seed)
    if max_cells is not None and adata.n_obs > max_cells:
        cells = rng.choice(adata.n_obs, size=max_cells, replace=False)
        cells.sort()
        X = X[cells]
    else:
        cells = np.arange(adata.n_obs)
    X = X.toarray() if sparse.issparse(X) else np.asarray(X)
    lib = X.sum(axis=1, keepdims=True)
    lib = np.clip(lib, 1.0, None)
    X = X / lib * target_sum
    X = np.log1p(X)
    return X.astype(np.float32), cells


def fit_umap(
    X: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.3,
    metric: str = "euclidean",
    seed: int = 0,
    pca_first: int | None = 50,
) -> np.ndarray:
    """Return a 2-D UMAP embedding of `X`. Optional PCA preprocessing for speed.

    Returns
    -------
    (n, 2) float32 array. Deterministic given the same seed.
    """
    import umap
    Y = X
    if pca_first is not None and X.shape[1] > pca_first:
        from sklearn.decomposition import PCA
        Y = PCA(n_components=pca_first, random_state=seed).fit_transform(X)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
        n_jobs=1,  # deterministic
    )
    return reducer.fit_transform(Y).astype(np.float32)


def knn_purity(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
    max_query: int | None = None,
    seed: int = 0,
) -> float:
    """Fraction of k nearest neighbours (in X) that share the label.

    - For species labels: HIGH purity = strong species structure (bad integration).
      Random chance = max class frequency. 1/K for balanced.
    - For cell-type labels: HIGH purity = cell-type structure preserved (good).

    Complexity is O(N log N) via sklearn NN; for N > 20k we subsample queries.
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(X)
    if max_query is not None and X.shape[0] > max_query:
        rng = np.random.default_rng(seed)
        qi = rng.choice(X.shape[0], size=max_query, replace=False)
    else:
        qi = np.arange(X.shape[0])
    _, idx = nn.kneighbors(X[qi])
    idx = idx[:, 1:]  # drop self
    same = (labels[idx] == labels[qi][:, None]).mean(axis=1)
    return float(same.mean())


def knn_mixing_entropy(
    X: np.ndarray,
    labels: np.ndarray,
    k: int = 30,
    max_query: int | None = None,
    seed: int = 0,
) -> float:
    """Mean normalised Shannon entropy of per-cell k-NN label distributions.

    1.0 = perfectly mixed (each neighbourhood is uniform over labels).
    0.0 = fully segregated (each neighbourhood is single-label).

    Returns natural-log entropy divided by log(K).
    """
    from sklearn.neighbors import NearestNeighbors
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    K = int(np.max(labels) + 1) if labels.dtype.kind in "iu" else len(np.unique(labels))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    if max_query is not None and X.shape[0] > max_query:
        rng = np.random.default_rng(seed)
        qi = rng.choice(X.shape[0], size=max_query, replace=False)
    else:
        qi = np.arange(X.shape[0])
    _, idx = nn.kneighbors(X[qi])
    neigh = labels[idx[:, 1:]]
    ents = []
    for row in neigh:
        _, counts = np.unique(row, return_counts=True)
        p = counts / counts.sum()
        ents.append(-(p * np.log(p)).sum())
    H = np.array(ents)
    return float((H / max(np.log(K), 1e-9)).mean())


def silhouette_subsample(
    X: np.ndarray,
    labels: np.ndarray,
    max_n: int = 5000,
    seed: int = 0,
) -> float:
    """Subsampled silhouette score. Returns 0 if only one class."""
    from sklearn.metrics import silhouette_score
    X = np.asarray(X, dtype=np.float32)
    labels = np.asarray(labels)
    if len(np.unique(labels)) < 2:
        return 0.0
    rng = np.random.default_rng(seed)
    if X.shape[0] > max_n:
        pick = rng.choice(X.shape[0], size=max_n, replace=False)
        X = X[pick]
        labels = labels[pick]
    return float(silhouette_score(X, labels, metric="euclidean"))


def integration_scorecard(
    X: np.ndarray,
    species_labels: np.ndarray,
    celltype_labels: np.ndarray | None,
    k: int = 15,
    max_query: int = 8000,
    seed: int = 0,
) -> dict[str, float]:
    """Bundle of fast integration diagnostics for a single embedding `X`.

    Keys:
        knn_species_purity (lower = better integration)
        knn_species_entropy (higher = better integration)
        knn_celltype_purity (higher = biology preserved)
        silhouette_celltype (higher = biology preserved)
    """
    out = {
        "knn_species_purity": knn_purity(X, species_labels, k=k,
                                         max_query=max_query, seed=seed),
        "knn_species_entropy": knn_mixing_entropy(X, species_labels, k=max(k, 30),
                                                  max_query=max_query, seed=seed),
    }
    if celltype_labels is not None:
        out["knn_celltype_purity"] = knn_purity(X, celltype_labels, k=k,
                                                max_query=max_query, seed=seed)
        out["silhouette_celltype"] = silhouette_subsample(X, celltype_labels,
                                                          max_n=5000, seed=seed)
    return out


# ---------------------------------------------------------------------------
# Biological gene sets (for notebook 13)
# ---------------------------------------------------------------------------
#
# Small, well-established categories. These are prefix-based (robust across the
# human gene-symbol mapping) or short curated lists (canonical housekeeping,
# cell cycle, interferon response). Not a replacement for MsigDB/GO — just
# enough to ground biological interpretation of α and Δα without a new
# dependency. See notebook 13 §0 for provenance notes.

RIBOSOMAL_PREFIXES = ("RPL", "RPS", "MRPL", "MRPS")
# Histone genes start with H1/H2A/H2B/H3/H4 or HIST*. Match via full regex below.
HISTONE_PREFIXES = ("HIST", "H1-", "H2A", "H2B", "H2AC", "H2BC", "H3-", "H4-", "H1F")
MITOCHONDRIAL_PREFIXES = ("MT-", "MTRNR", "MTATP", "MTCO", "MTCY", "MTND")

# Canonical housekeeping genes (Eisenberg & Levanon 2013 top tier + widely
# used qPCR controls). Deliberately short.
HOUSEKEEPING = frozenset([
    "ACTB", "GAPDH", "TUBB", "TUBA1B", "HPRT1", "PGK1", "PPIA",
    "B2M", "RPLP0", "RPL13A", "TBP", "UBC", "PSMB2", "PSMB4",
    "SDHA", "YWHAZ", "CHMP2A", "EMC7", "GPI", "REEP5", "RAB7A",
])

# Cell-cycle / proliferation (Tirosh et al. 2016 + canonical)
CELL_CYCLE = frozenset([
    "MKI67", "PCNA", "TOP2A", "MCM2", "MCM3", "MCM4", "MCM5", "MCM6",
    "MCM7", "CDK1", "CDK2", "CDK4", "CDK6", "CCNA2", "CCNB1", "CCNB2",
    "CCND1", "CCND2", "CCNE1", "CCNE2", "AURKA", "AURKB", "BIRC5",
    "CENPA", "CENPE", "CENPF", "FOXM1", "E2F1", "E2F2", "PLK1", "BUB1",
])

# Interferon-stimulated / innate-immune genes (canonical type-I IFN response)
INTERFERON = frozenset([
    "IFI6", "IFI27", "IFI35", "IFI44", "IFI44L", "IFIT1", "IFIT2",
    "IFIT3", "IFITM1", "IFITM2", "IFITM3", "ISG15", "ISG20",
    "OAS1", "OAS2", "OAS3", "OASL", "MX1", "MX2", "STAT1", "STAT2",
    "IRF7", "IRF9", "USP18", "RSAD2",
])

# Immediate-early (stress/activation response — often species-divergent regulation)
IMMEDIATE_EARLY = frozenset([
    "FOS", "FOSB", "JUN", "JUNB", "JUND", "EGR1", "EGR2", "EGR3",
    "NR4A1", "NR4A2", "NR4A3", "ATF3", "DUSP1", "IER2", "IER3",
])


def classify_gene_category(symbol: str) -> str:
    """Assign a gene symbol to one of the built-in biological categories, or 'other'.

    Prefix rules (ribosomal / histone / mitochondrial) take priority over the
    small curated lists so that e.g. RPL13A resolves to `ribosomal` rather than
    `housekeeping`.
    """
    if not isinstance(symbol, str):
        return "other"
    s = symbol.upper()
    if s.startswith(RIBOSOMAL_PREFIXES):
        return "ribosomal"
    if any(s.startswith(p) for p in HISTONE_PREFIXES):
        return "histone"
    if s.startswith(MITOCHONDRIAL_PREFIXES):
        return "mitochondrial"
    if s in HOUSEKEEPING:
        return "housekeeping"
    if s in CELL_CYCLE:
        return "cell_cycle"
    if s in INTERFERON:
        return "interferon"
    if s in IMMEDIATE_EARLY:
        return "immediate_early"
    return "other"


CATEGORY_ORDER = (
    "ribosomal", "mitochondrial", "histone", "housekeeping",
    "cell_cycle", "immediate_early", "interferon", "other",
)


def annotate_gene_categories(var_names: Iterable[str], cohort: str) -> pd.DataFrame:
    """Build a DataFrame with columns [var_name, symbol, category]."""
    var_names = pd.Index(list(var_names))
    sym = annotate_genes_with_names(var_names, cohort)
    cats = sym.map(classify_gene_category)
    return pd.DataFrame({"var_name": var_names, "symbol": sym.values, "category": cats.values},
                        index=var_names)


def bh_fdr(p_values) -> np.ndarray:
    """Benjamini–Hochberg FDR-adjusted p-values (q-values).

    Returns an array the same shape as ``p_values`` with each entry's
    BH-corrected q-value. Use ``q < 0.05`` (or your chosen FDR cutoff)
    to decide significance — that controls the *expected* false-
    discovery rate at 5 %, which is the standard genomics convention
    for enrichment-heatmap stars.

    NaNs in the input are passed through (NaN out, NaN ignored when
    counting tests).
    """
    p = np.asarray(p_values, dtype=float)
    n_total = p.size
    if n_total == 0:
        return p.copy()
    valid = np.isfinite(p)
    n = int(valid.sum())
    out = np.full(p.shape, np.nan, dtype=float)
    if n == 0:
        return out
    pv = p[valid]
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1, dtype=float)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    inv = np.empty_like(adj)
    inv[order] = adj
    out[valid] = inv
    return out


def perm_zscore_category(
    values: np.ndarray,
    in_set_mask: np.ndarray,
    n_perm: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Permutation-test z-scores for two non-Fisher statistics on a gene
    category, computed against a null in which the same number of genes
    is randomly drawn from the dataset (without replacement).

    Returns a dict with:
      ``z_med``  — z-score of the category's *median* of ``values`` under the
                   null. Catches *directional location shifts* of the category.
      ``z_mag``  — z-score of the category's *mean of |values|* under the null.
                   Catches *bidirectional amplitude inflation* (genes pulled
                   strongly in either direction). This is the test that fires
                   on Mitochondria & OXPHOS in testis where the median barely
                   shifts but |Δα| is dramatically larger than chance.
      ``p_med``  — empirical two-sided p-value for ``z_med``.
      ``p_mag``  — empirical one-sided p-value for ``z_mag`` (only "magnitude
                   greater than null" is biologically interesting).
      ``obs_med``, ``obs_mag`` — the observed test statistics.

    Vectorised: builds an ``(n_perm, n_in)`` index matrix once, then
    computes all null statistics with a single ``np.median`` /
    ``np.mean`` reduction across axis 1.
    """
    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    in_set_mask = np.asarray(in_set_mask, dtype=bool)
    n = int(values.size)
    n_in = int(in_set_mask.sum())
    av = np.abs(values)

    indices = np.empty((n_perm, n_in), dtype=np.int64)
    for i in range(n_perm):
        indices[i] = rng.choice(n, size=n_in, replace=False)
    null_med = np.median(values[indices], axis=1)
    null_mag = np.mean(av[indices], axis=1)

    obs_med = float(np.median(values[in_set_mask]))
    obs_mag = float(np.mean(av[in_set_mask]))

    sd_med = float(null_med.std(ddof=1))
    sd_mag = float(null_mag.std(ddof=1))
    z_med = float((obs_med - null_med.mean()) / sd_med) if sd_med > 0 else 0.0
    z_mag = float((obs_mag - null_mag.mean()) / sd_mag) if sd_mag > 0 else 0.0
    p_med = float(2.0 * min((null_med >= obs_med).mean(), (null_med <= obs_med).mean()))
    p_med = max(p_med, 1.0 / n_perm)
    p_mag = max(float((null_mag >= obs_mag).mean()), 1.0 / n_perm)
    return {
        "z_med": z_med, "z_mag": z_mag,
        "p_med": p_med, "p_mag": p_mag,
        "obs_med": obs_med, "obs_mag": obs_mag,
    }


def fisher_enrichment(
    in_set_mask: np.ndarray,
    hit_mask: np.ndarray,
    alternative: str = "two-sided",
) -> dict[str, float]:
    """Fisher exact test on contingency table (hit × in_set).

    hit_mask : bool array of gene-level "hits" (e.g. α in top 10 %).
    in_set_mask : bool array of genes in a reference set (e.g. ribosomal).

    `alternative` is forwarded to ``scipy.stats.fisher_exact`` with one
    extra mode supported here:

    - ``"two-sided"`` (default) — symmetric H1: OR ≠ 1. Conservative.
    - ``"greater"`` / ``"less"`` — one-sided as in scipy.
    - ``"directional"`` — pick ``"greater"`` if the observed OR ≥ 1 and
      ``"less"`` otherwise. Use this for enrichment heatmaps where the
      cell color already encodes direction and the star should flag
      significant deviation from OR=1 in *either* direction. Roughly
      halves p-values vs two-sided, which is appropriate when the
      direction-of-effect is read off the heatmap, not pre-specified.

    Returns dict with keys: odds_ratio, p_value, n_hit_in_set, n_hit,
    n_in_set, n_total.
    """
    from scipy.stats import fisher_exact
    hit = np.asarray(hit_mask, dtype=bool)
    s = np.asarray(in_set_mask, dtype=bool)
    a = int(np.sum(hit & s))
    b = int(np.sum(hit & ~s))
    c = int(np.sum(~hit & s))
    d = int(np.sum(~hit & ~s))
    table = [[a, b], [c, d]]
    try:
        odds_ts, _ = fisher_exact(table, alternative="two-sided")
        if alternative == "directional":
            side = "greater" if (np.isfinite(odds_ts) and odds_ts >= 1.0) else "less"
            _, p = fisher_exact(table, alternative=side)
            odds = odds_ts
        else:
            odds, p = fisher_exact(table, alternative=alternative)
    except ValueError:
        odds, p = float("nan"), 1.0
    return {
        "odds_ratio": float(odds),
        "p_value": float(p),
        "n_hit_in_set": a,
        "n_hit": a + b,
        "n_in_set": a + c,
        "n_total": a + b + c + d,
    }
