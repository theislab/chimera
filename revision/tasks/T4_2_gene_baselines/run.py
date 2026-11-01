"""T4.2: simple gene-level baselines B1b, B1c, B1z, B2b, B3a, B3b for every cohort.

Definitions follow the frozen analysis plan (05 section 5) and audit A5 action A4. No gold
standard is read and no association with one is computed here; the scoreboard (T4.3) does that.

Shared preprocessing
- ``.X`` is checked to hold raw counts (non-negative integers) over the whole matrix.
- Labels: liver ``obs.cell_type``; testis ``obs.cell_type`` through the fixed coarse map
  (SG, SC, rSD, eSD, Sertoli, Other_somatic). Library key: liver ``batch``, testis
  ``sample_batch`` (testis ``batch`` is the species).
- Pseudobulk per library: for each (species s, group t, library l) with >= 30 cells
  (>= 20 for Leiden clusters), ``x[g,s,t] = mean_l log2(1 + 1e6 C[g,s,t,l] / sum_g C[.,s,t,l])``.
  Detection f[g,s,t] is the fraction of expressing cells over the cells of those libraries.
- A gene is scorable in a species pair if f >= 1% in at least one group of the set used, in
  both species.

Scores
- E (per pair): Pearson across the pair's shared groups of the species-centred profiles,
  undefined for a constant profile. One pair: E = r. Several pairs: Fisher-z mean over the
  scorable pairs weighted by K_pair - 3; pairs with K_pair <= 3 carry no weight and are dropped.
- B1b = E on cell types, residualised on additive natural cubic splines (df 4, knots at rank
  quartiles, tied knots merged) of rank pi_raw, rank species-balanced mean log1p CP10K and rank species-balanced
  detection, fitted on the scorable genes. ``B1b_raw`` = E.
- B1c: within each type shared by all species of the cohort, genes are rank-normalised per
  species (mid-ranks / n genes); L = -mean_t |r_s - r_s'|, mean over scorable pairs; then
  residualised as B1b. ``B1c_raw`` = L.
- B1z: E across Leiden clusters of Chimera's latent (canonical checkpoint,
  ``ChimeraModel.load(...).get_latent_representation()``), 15-NN exact graph, resolution 1,
  igraph Leiden, fixed seed; residualised as B1b. ``B1z_raw`` = E on clusters.
- B2b: p[g,s,t] = 2^x / sum_t 2^x on the types shared by all species; cosine(p_s, p_s'),
  mean over scorable pairs. Only when that K >= 4; otherwise NA for every gene (liver
  H+Zebrafish K=3 and 4-sp K=2), never forced.
- B3a, B3b: 5-fold cross-fitted RidgeCV on the 72 GIN features (``varm['gene_features']``,
  standardised inside each fold) predicting E (B1b_raw) and delta-alpha. Genes with a target
  get their out-of-fold prediction; genes without one get the mean of the five fold models
  (none of which saw them). ``*_oof`` flags which.

The delta-alpha source for B3b is an argument: ``canonical`` (alpha - prior of the canonical s42
run, the clamped prior on clamped recipes) or a CSV/parquet path, ``{cohort}`` substituted, keyed
by ``hsapiens_gene_id`` with the value in ``--delta-column`` (the T3.3 consensus D).

Outputs ``results/revision/T4_2/baselines__<cohort>.csv`` (one row per human ENSG),
``pseudobulk_units__<cohort>.csv``, ``summary__<cohort>.json``, ``marker_check.csv``.

Usage::

    python revision/tasks/T4_2_gene_baselines/run.py [--cohorts all] [--delta-alpha canonical]
        [--delta-column D] [--threads 8] [--outdir DIR] [--force]
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
import warnings

if __name__ == "__main__":  # fix BLAS threads before numpy loads, for a reproducible rerun
    _t = "8"
    if "--threads" in sys.argv:
        _t = sys.argv[sys.argv.index("--threads") + 1]
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = _t

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata

from revision.common import cohorts as C
from revision.common import gene_ids
from revision.common.outputs import TaskOutput, add_common_args

TASK = "T4_2"
SEED = 0
MIN_CELLS_TYPE = 30
MIN_CELLS_CLUSTER = 20
DET_MIN = 0.01
N_FOLDS = 5
RIDGE_ALPHAS = np.logspace(-2, 5, 29)
LEIDEN_RESOLUTION = 1.0
N_NEIGHBORS = 15
LIBRARY_KEY = {"liver": "batch", "testis": "sample_batch"}
TESTIS_COARSE = {
    "Undifferentiated_Spermatogonia": "SG", "Differentiated_Spermatogonia": "SG", "Spermatogonia": "SG",
    "Leptotene_Spermatocytes": "SC", "Zygotene_Spermatocytes": "SC", "Pachytene_Spermatocytes": "SC",
    "Spermatocytes": "SC",
    "Early_Round_spermatids": "rSD", "Late_Round_spermatids": "rSD", "Round_spermatids": "rSD",
    "Elongated_spermatids": "eSD", "Sertoli": "Sertoli", "Other_somatic": "Other_somatic",
}
MARKERS = {  # gene symbol -> expected peak group (genes in the 1:1 vocabularies)
    "testis": {"PRM2": "eSD", "TNP1": "eSD", "ACRV1": "rSD", "SPACA1": "rSD", "SYCP1": "SC",
               "SPO11": "SC", "PIWIL1": "SC", "DDX4": "SG", "ZBTB16": "SG", "PIWIL4": "SG",
               "FGFR3": "SG", "SOX9": "Sertoli", "WT1": "Sertoli", "ACTA2": "Other_somatic",
               "MYH11": "Other_somatic"},
    "liver": {"HNF4A": "Hepatocytes", "KDR": "Endothelial", "STAB2": "Endothelial",
              "DCN": "fibroblasts"},
}
GERM = {"SG", "SC", "rSD", "eSD"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def tissue(cohort: str) -> str:
    return "testis" if C.is_testis(cohort) else "liver"


def check_raw_counts(X) -> dict:
    d = X.data if sp.issparse(X) else np.asarray(X).ravel()
    ok_int = bool(np.all(d == np.round(d)))
    ok_nonneg = bool(np.all(d >= 0))
    if not (ok_int and ok_nonneg):
        raise AssertionError(f".X is not raw counts: integer={ok_int} non-negative={ok_nonneg}")
    return dict(X_dtype=str(d.dtype), X_integer=ok_int, X_nonnegative=ok_nonneg, X_max=float(d.max()),
                X_nnz=int(d.size))


def ns_basis(x: np.ndarray) -> np.ndarray:
    """Natural cubic spline basis without intercept, knots at min, quartiles, max (df 4).

    Tied knots (a covariate with a mass point, e.g. the testis prior ceiling) are merged,
    which lowers df by one per merged knot."""
    k = np.unique(np.quantile(x, [0.0, 0.25, 0.5, 0.75, 1.0]))
    if len(k) < 3:
        return x[:, None]

    def d(j):
        return (np.clip(x - k[j], 0, None) ** 3 - np.clip(x - k[-1], 0, None) ** 3) / (k[-1] - k[j])

    cols = [x] + [d(j) - d(len(k) - 2) for j in range(len(k) - 2)]
    return np.column_stack(cols)


def residualise(score: pd.Series, cov: pd.DataFrame) -> pd.Series:
    """score - OLS fit on additive splines of the within-universe ranks of each covariate."""
    m = score.notna() & cov.notna().all(1)
    out = pd.Series(np.nan, index=score.index)
    if m.sum() < 50:
        return out
    blocks = [np.ones((m.sum(), 1))]
    for c in cov.columns:
        r = rankdata(cov.loc[m, c].to_numpy()) / m.sum()
        blocks.append(ns_basis(r))
    Z = np.hstack(blocks)
    y = score[m].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
    out[m] = y - Z @ beta
    return out


def pseudobulk(X, species: np.ndarray, group: np.ndarray, library: np.ndarray, min_cells: int):
    """Mean over qualifying libraries of log2(1 + CPM); detection over those cells.

    Returns ``x``, ``f``: dict (species, group) -> gene vector, and the unit table.
    """
    units = (pd.DataFrame(dict(species=species, group=group, library=library))
             .value_counts().rename("n_cells").reset_index()
             .sort_values(["species", "group", "library"]).reset_index(drop=True))
    units["used"] = units["n_cells"] >= min_cells
    x, f = {}, {}
    for (s, t), u in units[units.used].groupby(["species", "group"], sort=True):
        prof, nexp, ncell = [], 0, 0
        for lib in u["library"]:
            mk = (species == s) & (group == t) & (library == lib)
            Xs = X[mk]
            cnt = np.asarray(Xs.sum(0)).ravel()
            prof.append(np.log2(1.0 + 1e6 * cnt / cnt.sum()))
            nexp = nexp + np.asarray((Xs > 0).sum(0)).ravel()
            ncell += int(mk.sum())
        x[(s, t)] = np.mean(prof, 0)
        f[(s, t)] = nexp / ncell
    return x, f, units


def shared_groups(x: dict, species_list: list[str], among: list[str]) -> list[str]:
    groups = None
    for s in among:
        g = {t for (ss, t) in x if ss == s}
        groups = g if groups is None else groups & g
    return sorted(groups or [])


def pair_corr(x: dict, f: dict, s1: str, s2: str, groups: list[str], n_genes: int):
    """Pearson across groups of the species-centred profiles; NaN where unscorable or constant."""
    r = np.full(n_genes, np.nan)
    if len(groups) < 2:
        return r, np.zeros(n_genes, bool)
    A = np.column_stack([x[(s1, t)] for t in groups])
    B = np.column_stack([x[(s2, t)] for t in groups])
    det = (np.column_stack([f[(s1, t)] for t in groups]).max(1) >= DET_MIN) & \
          (np.column_stack([f[(s2, t)] for t in groups]).max(1) >= DET_MIN)
    A = A - A.mean(1, keepdims=True)
    B = B - B.mean(1, keepdims=True)
    na, nb = np.sqrt((A ** 2).sum(1)), np.sqrt((B ** 2).sum(1))
    ok = det & (na > 1e-12) & (nb > 1e-12)
    r[ok] = (A[ok] * B[ok]).sum(1) / (na[ok] * nb[ok])
    return np.clip(r, -1.0, 1.0), det


def pooled_E(x: dict, f: dict, species_list: list[str], n_genes: int):
    """E over species pairs, each on its own maximal shared group set; Fisher-z pooled by K-3."""
    pairs = list(itertools.combinations(species_list, 2))
    info, rs, ws = [], [], []
    for s1, s2 in pairs:
        g = shared_groups(x, species_list, [s1, s2])
        r, det = pair_corr(x, f, s1, s2, g, n_genes)
        w = 1.0 if len(pairs) == 1 else float(max(len(g) - 3, 0))
        info.append(dict(pair=f"{s1}-{s2}", K=len(g), groups=g, weight=w,
                         n_scorable=int(det.sum()), n_defined=int(np.isfinite(r).sum())))
        if w > 0:
            rs.append(r)
            ws.append(w)
    if not rs:
        return np.full(n_genes, np.nan), np.zeros(n_genes, int), info
    R = np.column_stack(rs)
    W = np.broadcast_to(np.array(ws), R.shape)
    fin = np.isfinite(R)
    Zf = np.arctanh(np.clip(np.where(fin, R, 0.0), -0.999999, 0.999999))
    wsum = (W * fin).sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        E = np.tanh((W * Zf * fin).sum(1) / wsum)
    E[wsum == 0] = np.nan
    if len(pairs) == 1:
        E = R[:, 0]
    return E, fin.sum(1), info


def level_and_specificity(x: dict, f: dict, species_list: list[str], n_genes: int):
    """B1c (L) and B2b (cosine of specificity vectors) on the groups shared by all species."""
    groups = shared_groups(x, species_list, species_list)
    ranks = {(s, t): rankdata(x[(s, t)]) / n_genes for s in species_list for t in groups}
    Ls, Cs = [], []
    for s1, s2 in itertools.combinations(species_list, 2):
        if not groups:
            break
        det = (np.column_stack([f[(s1, t)] for t in groups]).max(1) >= DET_MIN) & \
              (np.column_stack([f[(s2, t)] for t in groups]).max(1) >= DET_MIN)
        L = -np.mean([np.abs(ranks[(s1, t)] - ranks[(s2, t)]) for t in groups], 0)
        Ls.append(np.where(det, L, np.nan))
        if len(groups) >= 4:
            P1 = np.column_stack([2.0 ** x[(s1, t)] for t in groups])
            P2 = np.column_stack([2.0 ** x[(s2, t)] for t in groups])
            P1 /= P1.sum(1, keepdims=True)
            P2 /= P2.sum(1, keepdims=True)
            cos = (P1 * P2).sum(1) / (np.linalg.norm(P1, axis=1) * np.linalg.norm(P2, axis=1))
            Cs.append(np.where(det, cos, np.nan))
    nan = np.full(n_genes, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        L =np.nanmean(np.column_stack(Ls), 1) if Ls else nan
        B2 = np.nanmean(np.column_stack(Cs), 1) if Cs else nan
    nL = np.isfinite(np.column_stack(Ls)).sum(1) if Ls else np.zeros(n_genes, int)
    nB = np.isfinite(np.column_stack(Cs)).sum(1) if Cs else np.zeros(n_genes, int)
    return L, nL, B2, nB, groups


def cross_fit_ridge(F: np.ndarray, y: np.ndarray, seed: int = SEED):
    """Out-of-fold RidgeCV predictions where y is known; fold-model mean elsewhere."""
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    known = np.isfinite(y)
    idx = np.flatnonzero(known)
    pred = np.full(len(y), np.nan)
    other = np.zeros(len(y))
    alphas = []
    kf = KFold(N_FOLDS, shuffle=True, random_state=seed)
    for tr, te in kf.split(idx):
        mdl = make_pipeline(StandardScaler(), RidgeCV(alphas=RIDGE_ALPHAS))
        mdl.fit(F[idx[tr]], y[idx[tr]])
        pred[idx[te]] = mdl.predict(F[idx[te]])
        if (~known).any():
            other[~known] += mdl.predict(F[~known]) / N_FOLDS
        alphas.append(float(mdl[-1].alpha_))
    pred[~known] = other[~known]
    yk = y[known]
    r2 = 1 - np.sum((yk - pred[known]) ** 2) / np.sum((yk - yk.mean()) ** 2)
    return pred, known, dict(cv_r2=float(r2), n_train=int(known.sum()), alphas=alphas)


def chimera_latent(cohort: str, adata) -> np.ndarray:
    from chimera import ChimeraModel

    run_dir = C.RUNS_DIR / C.CANONICAL[cohort]
    model = ChimeraModel.load(str(run_dir / "model"), adata=adata, accelerator="cpu")
    return np.asarray(model.get_latent_representation(), dtype=np.float32)


def leiden_clusters(z: np.ndarray) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=np.zeros((z.shape[0], 1), dtype=np.float32))
    a.obsm["X_z"] = z
    sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS, use_rep="X_z", transformer="sklearn", random_state=SEED)
    sc.tl.leiden(a, resolution=LEIDEN_RESOLUTION, flavor="igraph", n_iterations=2, directed=False,
                 random_state=SEED)
    return a.obs["leiden"].astype(str).map(lambda s: f"L{int(s):03d}").to_numpy()


def load_delta_alpha(source: str, column: str, cohort: str, genes: pd.DataFrame) -> pd.Series:
    if source == "canonical":
        return genes["delta_alpha"].astype(float)
    path = source.replace("{cohort}", cohort)
    tab = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    if tab["hsapiens_gene_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated hsapiens_gene_id")
    m = tab.set_index("hsapiens_gene_id")[column]
    return genes["hsapiens_gene_id"].map(m).astype(float)


# ---------------------------------------------------------------------------
# Per cohort
# ---------------------------------------------------------------------------

def run_cohort(cohort: str, args, out: TaskOutput) -> pd.DataFrame:
    tis = tissue(cohort)
    print(f"[{cohort}] loading", flush=True)
    co = C.load_cohort(cohort, keep_adata=True, covariates=True)
    a, g = co.adata, co.genes
    raw = check_raw_counts(a.X)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    n = a.n_vars
    species = a.obs["species"].astype(str).to_numpy()
    species_list = sorted(set(species))
    libkey = LIBRARY_KEY[tis]
    library = a.obs[libkey].astype(str).to_numpy()
    if pd.isna(a.obs[libkey]).any():
        raise AssertionError(f"{cohort}: missing {libkey}")
    lab = a.obs["cell_type"].astype(str)
    if tis == "testis":
        if not set(lab.unique()) <= set(TESTIS_COARSE):
            raise AssertionError(f"{cohort}: unmapped testis labels {set(lab.unique()) - set(TESTIS_COARSE)}")
        lab = lab.map(TESTIS_COARSE)
    ctype = lab.to_numpy()

    # cell-type pseudobulk
    x, f, units = pseudobulk(X, species, ctype, library, MIN_CELLS_TYPE)
    units.insert(0, "grouping", "cell_type")
    E, nE, pinfo = pooled_E(x, f, species_list, n)
    L, nL, B2, nB2, all_groups = level_and_specificity(x, f, species_list, n)

    # Leiden on Chimera z
    print(f"[{cohort}] latent + Leiden", flush=True)
    z = chimera_latent(cohort, a)
    cl = leiden_clusters(z)
    xz, fz, uz = pseudobulk(X, species, cl, library, MIN_CELLS_CLUSTER)
    uz.insert(0, "grouping", "leiden_z")
    Ez, nEz, zinfo = pooled_E(xz, fz, species_list, n)

    cov = g[["prior_raw", "log_cp10k_spmean", "detection_spmean"]]
    res = pd.DataFrame(index=g.index)
    res["hsapiens_gene_id"] = g["hsapiens_gene_id"].values
    res["var_name"] = g.index
    res["symbol"] = g["symbol"].values
    res["B1b"] = residualise(pd.Series(E, index=g.index), cov)
    res["B1b_raw"] = E
    res["n_pairs_B1b"] = nE
    res["B1c"] = residualise(pd.Series(L, index=g.index), cov)
    res["B1c_raw"] = L
    res["n_pairs_B1c"] = nL
    res["B1z"] = residualise(pd.Series(Ez, index=g.index), cov)
    res["B1z_raw"] = Ez
    res["n_pairs_B1z"] = nEz
    res["B2b"] = B2
    res["n_pairs_B2b"] = nB2

    F = np.asarray(co.state["gene_features"], dtype=float)
    if F.shape != (n, 72):
        raise AssertionError(f"{cohort}: gene_features shape {F.shape}")
    b3a, oofa, infoa = cross_fit_ridge(F, np.asarray(E, dtype=float))
    da = load_delta_alpha(args.delta_alpha, args.delta_column, cohort, g)
    b3b, oofb, infob = cross_fit_ridge(F, da.to_numpy(dtype=float))
    res["B3a"] = b3a
    res["B3a_oof"] = oofa
    res["B3b"] = b3b
    res["B3b_oof"] = oofb

    # scorable counts: per species, number of all-species shared types with detection >= 1%
    for s in species_list:
        res[f"n_types_det_{s}"] = (np.column_stack([f[(s, t)] for t in all_groups]) >= DET_MIN).sum(1) \
            if all_groups else 0
    res["prior_raw"] = g["prior_raw"].values
    res["log_cp10k_spmean"] = g["log_cp10k_spmean"].values
    res["detection_spmean"] = g["detection_spmean"].values
    res = res.sort_values("hsapiens_gene_id").reset_index(drop=True)
    if res["hsapiens_gene_id"].isna().any() or res["hsapiens_gene_id"].duplicated().any():
        raise AssertionError(f"{cohort}: ENSG missing or duplicated")

    # marker sanity check on cell-type profiles
    mk_rows = []
    sym = g["symbol"].astype(str).to_numpy()
    for gene, expect in MARKERS[tis].items():
        hit = np.flatnonzero(sym == gene)
        if len(hit) != 1:
            mk_rows.append(dict(cohort=cohort, symbol=gene, expected=expect, note=f"{len(hit)} matches"))
            continue
        j = hit[0]
        for s in species_list:
            prof = {t: float(x[(s, t)][j]) for (ss, t) in x if ss == s}
            peak = max(prof, key=prof.get)
            order = sorted(prof, key=lambda t: -prof[t])
            germ_ok = (peak in GERM) == (expect in GERM) if tis == "testis" else np.nan
            mk_rows.append(dict(cohort=cohort, symbol=gene, hsapiens_gene_id=g["hsapiens_gene_id"].iloc[j],
                                var_name=g.index[j], species=s, expected=expect, peak=peak,
                                peak_ok=peak == expect,
                                expected_rank=order.index(expect) + 1 if expect in order else np.nan,
                                germ_soma_ok=germ_ok, peak_log2cpm=prof[peak],
                                expected_log2cpm=prof.get(expect, np.nan),
                                profile=";".join(f"{t}={v:.2f}" for t, v in sorted(prof.items()))))

    n_clusters = len(np.unique(cl))
    summary = dict(
        cohort=cohort, run=co.run, h5ad=gene_ids.H5ADS[cohort], n_genes=n, n_cells=int(a.n_obs),
        species=species_list, library_key=libkey,
        libraries={s: sorted(set(library[species == s])) for s in species_list},
        label_key="cell_type" + (" (testis coarse map)" if tis == "testis" else ""),
        raw_counts=raw, min_cells_type=MIN_CELLS_TYPE, min_cells_cluster=MIN_CELLS_CLUSTER,
        detection_min=DET_MIN, celltype_pairs=pinfo, all_species_types=all_groups,
        K_all_species=len(all_groups), B2b_defined=len(all_groups) >= 4,
        leiden=dict(n_clusters=n_clusters, resolution=LEIDEN_RESOLUTION, n_neighbors=N_NEIGHBORS,
                    latent_dim=int(z.shape[1]), pairs=zinfo),
        B3a=infoa, B3b=dict(infob, delta_alpha_source=args.delta_alpha,
                            delta_column=None if args.delta_alpha == "canonical" else args.delta_column),
        n_scored={c: int(res[c].notna().sum()) for c in
                  ["B1b", "B1b_raw", "B1c", "B1c_raw", "B1z", "B1z_raw", "B2b", "B3a", "B3b"]},
        seed=SEED, threads=args.threads,
    )
    out.csv(res, f"baselines__{cohort}.csv")
    out.csv(pd.concat([units, uz], ignore_index=True), f"pseudobulk_units__{cohort}.csv")
    out.json(summary, f"summary__{cohort}.json")
    print(f"[{cohort}] n={n} scored {summary['n_scored']} K_all={len(all_groups)} "
          f"clusters={n_clusters} B3a R2={infoa['cv_r2']:.3f} B3b R2={infob['cv_r2']:.3f}", flush=True)
    return pd.DataFrame(mk_rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--cohorts", default="all", help="comma list or 'all'")
    p.add_argument("--delta-alpha", default="canonical",
                   help="'canonical' (alpha - prior, canonical s42) or a CSV/parquet path with {cohort}")
    p.add_argument("--delta-column", default="D", help="value column of a --delta-alpha file")
    p.add_argument("--threads", type=int, default=8)
    add_common_args(p)
    args = p.parse_args(argv)
    cohorts = C.COHORT_ORDER if args.cohorts == "all" else args.cohorts.split(",")
    for c in cohorts:
        if c not in C.CANONICAL:
            raise KeyError(c)
    import torch

    torch.set_num_threads(args.threads)
    out = TaskOutput(TASK, args.outdir, prerun=args.prerun, force=args.force)
    markers = [run_cohort(c, args, out) for c in cohorts]
    mk = pd.concat(markers, ignore_index=True)
    out.csv(mk, "marker_check.csv")
    out.finish(dict(cohorts=cohorts, delta_alpha=args.delta_alpha, seed=SEED,
                    library_key=LIBRARY_KEY, testis_coarse_map=TESTIS_COARSE))


if __name__ == "__main__":
    main()
