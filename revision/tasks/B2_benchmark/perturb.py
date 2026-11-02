"""B2 Track B: build the object of one setting. Pure numpy / scipy on the canonical object; every draw comes from a seeded
generator, so an object regenerates exactly from its grid row (cohort, setting, rep). The h5ad is removed after its fit is scored.

Written to the object:
  obs['b2_label']    cell annotation the direct baselines read in this setting;
  obs['b2_library']  library key of the direct baselines;
  obs[model_key]     annotation the recipe reads (conditional MMD), where the recipe reads one;
  uns['p_null']      recomputed as the pipeline does (precompute_p_null, eps 1);
  varm['gene_features'] expression columns moved with the data, F' = F + (stat' - stat) / SD(stat) over genes (as Track A);
  conservation_prior and sequence features stay.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.common import gene_ids
from revision.tasks.B2_benchmark import rules as RU
from revision.tasks.T4_2_gene_baselines.run import TESTIS_COARSE

EXPR_COLS = [66, 67, 68]
CHEM_FILE = RU.RES / "inputs" / "chemistry_profile.parquet"


def csr(X) -> sp.csr_matrix:
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)


def read_base(cohort: str):
    import anndata as ad

    return ad.read_h5ad(RU.REPO / RU.ALL_COHORTS[cohort]["h5ad"])


def base_labels(a, cohort: str) -> np.ndarray:
    cfg = RU.ALL_COHORTS[cohort]
    lab = a.obs[cfg["label_key"]].astype(str)
    if cfg["coarse_map"]:
        lab = lab.map(TESTIS_COARSE)
    assert lab.notna().all()
    return lab.to_numpy()


def expr_stats(X: sp.csr_matrix) -> np.ndarray:
    """compute_expression_stats, as columns log_mean_expr, log_expr_var, frac_nonzero."""
    X = X.astype(np.float64)
    m = np.asarray(X.mean(0)).ravel()
    X2 = X.copy()
    X2.data = X2.data ** 2
    v = np.maximum(np.asarray(X2.mean(0)).ravel() - m ** 2, 0.0)
    f = np.asarray((X > 0).sum(0)).ravel() / X.shape[0]
    return np.column_stack([np.log1p(m), np.log1p(v), f])


def p_null(X: sp.csr_matrix, spc: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """precompute_p_null, rows in sorted species order."""
    G = X.shape[1]
    out = []
    for s in sorted(set(spc)):
        Xs = X[spc == s]
        out.append((np.asarray(Xs.sum(0)).ravel() + eps) / (Xs.sum() + eps * G))
    return np.vstack(out)


def features_after(F: np.ndarray, X0: sp.csr_matrix, X1: sp.csr_matrix) -> np.ndarray:
    s0, s1 = expr_stats(X0), expr_stats(X1)
    F1 = np.array(F, copy=True)
    sd = s0.std(0, ddof=0)
    for k, c in enumerate(EXPR_COLS):
        F1[:, c] = F1[:, c] + (s1[:, k] - s0[:, k]) / sd[k]
    return F1


def shared_by_size(lab: np.ndarray, spc: np.ndarray) -> list[str]:
    """Types with >= 30 cells in both species, largest smaller-species count first."""
    t = pd.crosstab(lab, spc)
    t = t[(t >= RU.MIN_CELLS_LABEL).all(1)]
    return t.min(1).sort_values(ascending=False, kind="mergesort").index.tolist()


def thin(rng, X: sp.csr_matrix, keep) -> sp.csr_matrix:
    """x' ~ Binomial(x, keep). keep is a number or one probability per gene."""
    X = X.tocsr().copy()
    d = X.data.astype(np.int64)
    if np.ndim(keep) == 0:
        X.data = rng.binomial(d, float(keep)).astype(X.dtype)
    else:
        col = np.asarray(keep)[X.indices]
        X.data = rng.binomial(d, col).astype(X.dtype)
    X.eliminate_zeros()
    return X


def relabel(rng, lab: np.ndarray, spc: np.ndarray, q: float) -> np.ndarray:
    """X6's noise model. In each species a fraction q of cells gets another label, in proportion to abundance."""
    out = lab.copy()
    for s in np.unique(spc):
        idx = np.flatnonzero(spc == s)
        types, cnt = np.unique(lab[idx], return_counts=True)
        pick = rng.choice(idx, int(round(q * len(idx))), replace=False)
        for i in pick:
            w = np.where(types == lab[i], 0.0, cnt.astype(float))
            out[i] = types[rng.choice(len(types), p=w / w.sum())]
    return out


def relabel_near(rng, X, lab, spc, q: float) -> np.ndarray:
    """Confusion: in each species a fraction q of cells, each gets the label of its type's most similar other type
    (Pearson of mean log1p CP10K over the 2,000 most variable genes of the species)."""
    out = lab.copy()
    L = np.asarray(X.sum(1)).ravel()
    Xn = sp.diags(1e4 / np.maximum(L, 1)) @ X
    Xn.data = np.log1p(Xn.data)
    for s in np.unique(spc):
        idx = np.flatnonzero(spc == s)
        types = np.unique(lab[idx])
        P = np.column_stack([np.asarray(Xn[idx[lab[idx] == t]].mean(0)).ravel() for t in types])
        hv = np.argsort(-P.var(1))[:2000]
        C = np.corrcoef(P[hv].T)
        np.fill_diagonal(C, -np.inf)
        near = {t: types[int(C[i].argmax())] for i, t in enumerate(types)}
        pick = rng.choice(idx, int(round(q * len(idx))), replace=False)
        for i in pick:
            out[i] = near[lab[i]]
    return out


def _hvg_pcs(X, n_top=2000, n_pcs=20, seed=0):
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=X.astype(np.float32).copy())
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=n_top)
    h = a[:, a.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(h, max_value=10)
    n = min(n_pcs, min(h.shape) - 1)
    sc.tl.pca(h, n_comps=n, svd_solver="arpack", random_state=seed)
    return h.obsm["X_pca"], a.X


def fine_labels(X, lab, spc, seed: int) -> np.ndarray:
    """Over-fine annotation: each type split in FINE_K by k-means, in each species separately; sub-labels are matched across species
    inside the parent type by Hungarian assignment on the Pearson of z-scored mean profiles over the parent's HVGs."""
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans

    out = lab.astype(object).copy()
    t = pd.crosstab(lab, spc)
    for ty in t.index:
        if not all(s in t.columns and t.loc[ty, s] >= RU.FINE_MIN for s in RU.SPECIES):
            continue
        prof = {}
        ids = {}
        for s in RU.SPECIES:
            idx = np.flatnonzero((lab == ty) & (spc == s))
            pcs, Xl = _hvg_pcs(X[idx], seed=seed)
            k = KMeans(RU.FINE_K, n_init=5, random_state=seed).fit_predict(pcs)
            ids[s] = (idx, k)
            prof[s] = np.column_stack([np.asarray(Xl[k == j].mean(0)).ravel() for j in range(RU.FINE_K)])
        v = (prof["Human"].var(1) + prof["Mouse"].var(1))
        hv = np.argsort(-v)[:500]
        A, B = prof["Human"][hv], prof["Mouse"][hv]
        A = (A - A.mean(1, keepdims=True)) / (A.std(1, keepdims=True) + 1e-9)
        B = (B - B.mean(1, keepdims=True)) / (B.std(1, keepdims=True) + 1e-9)
        R = np.corrcoef(A.T, B.T)[:RU.FINE_K, RU.FINE_K:]
        hi, mi = linear_sum_assignment(-R)
        name_h = {h: f"{ty}__{j}" for j, h in enumerate(hi)}
        name_m = {m: f"{ty}__{j}" for j, m in enumerate(mi)}
        idx, k = ids["Human"]
        out[idx] = [name_h[v_] for v_ in k]
        idx, k = ids["Mouse"]
        out[idx] = [name_m[v_] for v_ in k]
    return out.astype(str)


def species_diff(lab, spc) -> np.ndarray:
    """Different vocabularies: in mouse every second shared type by size (second, fourth, ...) carries a mouse-only name, as
    human macrophages and mouse Kupffer cells do in the liver data. The recipe's conditional MMD skips such types by construction."""
    out = lab.astype(object).copy()
    types = shared_by_size(lab, spc)
    for t in types[1::2]:
        out[(spc == "Mouse") & (lab == t)] = f"Mouse_{t}"
    return out.astype(str)


def nolabel(X, spc, lab_true, seed: int, r_min: float) -> np.ndarray:
    """Annotation-free direct workflow of X6: per-species Leiden on HVGs, clusters matched across species by reciprocal best hits
    of marker-profile correlation at r >= r_min; matched pairs share a label, the rest keep species-own labels."""
    from revision.tasks.X6_adds_beyond import degrade as X6
    from revision.tasks.X6_adds_beyond import rules as X6R

    keep_p, keep_r = dict(X6R.LEIDEN), dict(X6R.LEIDEN_RMIN)
    try:
        X6R.LEIDEN.update(RU.LEIDEN | dict(seed=seed))
        X6R.LEIDEN_RMIN.clear()
        X6R.LEIDEN_RMIN["m"] = r_min
        outs, _ = X6.leiden_matched(X, spc, lab_true, "b2")
    finally:
        X6R.LEIDEN.clear()
        X6R.LEIDEN.update(keep_p)
        X6R.LEIDEN_RMIN.clear()
        X6R.LEIDEN_RMIN.update(keep_r)
    return outs["m"]


def chem_profile() -> pd.Series:
    """Empirical per-gene capture factor, human Ensembl ID -> keep probability.

    Tabula Sapiens liver holds 10x 3' v3 and Smart-seq2 cells of the same tissue. r_g = mean over the types with >= 30 cells in both
    assays of (CPM_ss2 + 1) / (CPM_10x + 1); keep = clip(r_g / q90(r_g), 1 / CHEM_CLIP, 1). Thinning one species by this profile moves its
    gene-wise capture toward Smart-seq2, a real chemistry difference across genes, with no change in the biology. Built once, no Chimera
    readout or gold association is read. The gold G_H3 uses the 10x cells of the same file; disclosed."""
    import anndata as ad

    if CHEM_FILE.exists():
        return pd.read_parquet(CHEM_FILE)["keep"]
    a = ad.read_h5ad(RU.REPO / "data/revision/gold/G2/ts_liver.h5ad")
    X = a.raw.X if a.raw is not None else a.X
    X = csr(X).astype(np.float64)
    var = a.raw.var if a.raw is not None else a.var
    ids = var.index.astype(str).str.replace(r"\.\d+$", "", regex=True)
    assay = a.obs["assay"].astype(str).to_numpy()
    ct = a.obs["cell_type"].astype(str).to_numpy()
    is10x = np.array(["10x" in v for v in assay])
    isss = np.array(["Smart-seq" in v for v in assay])
    t = pd.DataFrame(dict(ct=ct[is10x | isss], k=np.where(is10x, "x", "s")[is10x | isss]))
    tab = pd.crosstab(t.ct, t.k)
    types = tab.index[(tab >= 30).all(1)].tolist()
    rs = []
    for ty in types:
        m1, m2 = (ct == ty) & is10x, (ct == ty) & isss
        c1, c2 = np.asarray(X[m1].sum(0)).ravel(), np.asarray(X[m2].sum(0)).ravel()
        rs.append(np.log2((1e6 * c2 / c2.sum() + 1) / (1e6 * c1 / c1.sum() + 1)))
    r = 2.0 ** np.mean(rs, 0)
    q = np.quantile(r[np.isfinite(r)], 0.9)
    keep = np.clip(r / q, 1 / RU.CHEM_CLIP, 1.0)
    d = pd.DataFrame(dict(keep=keep, r=r), index=pd.Index(ids, name="hsapiens_gene_id"))
    d = d[~d.index.duplicated()]
    CHEM_FILE.parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(CHEM_FILE)
    (CHEM_FILE.parent / "chemistry_profile.json").write_text(json.dumps(dict(
        types=types, q90=float(q), median_keep=float(np.median(keep)), n_genes=int(len(d)),
        assays=sorted(set(assay))), indent=1))
    return d["keep"]


def donors_kept(rng, donor, spc, n, rep: int) -> np.ndarray:
    keep = np.zeros(len(spc), bool)
    for s in np.unique(spc):
        ds = pd.Series(donor[spc == s]).value_counts()
        pick = ds.index[:n] if rep == 0 else rng.choice(ds.index.to_numpy(), n, replace=False)
        keep |= (spc == s) & np.isin(donor, list(pick))
    return keep


def species_sizes(rng, spc, ratio: int, rep: int) -> np.ndarray:
    """Cells kept for an imbalance 1:ratio. rep 0 and 2: mouse is the small species; rep 1: human. 1:1 cuts the larger species."""
    n = {s: int((spc == s).sum()) for s in RU.SPECIES}
    keep = np.ones(len(spc), bool)
    if ratio == 1:
        small = min(n, key=n.get)
        large = max(n, key=n.get)
        cut, target = large, n[small]
    else:
        cut = "Mouse" if rep in (0, 2) else "Human"
        other = "Human" if cut == "Mouse" else "Mouse"
        target = n[other] // ratio
    idx = np.flatnonzero(spc == cut)
    if target < len(idx):
        drop = rng.choice(idx, len(idx) - target, replace=False)
        keep[drop] = False
    return keep


def build(row: dict, out_path=None, check: bool = True):
    """Returns (adata, info). Row: cohort, setting, rep."""
    cohort, setting, rep = row["cohort"], row["setting"], int(row["rep"])
    cfg = RU.ALL_COHORTS[cohort]
    st = RU.SETTINGS["full" if setting == "ref" else setting]
    rng = np.random.default_rng(RU.useed("perturb", cohort, RU.SETTINGS.get(setting, {}).get("object_of", setting), rep))
    a = read_base(cohort)
    X0 = csr(a.X)
    assert np.allclose(X0.data, np.round(X0.data)), "X is not integer"
    spc = a.obs["species"].astype(str).to_numpy()
    lab = base_labels(a, cohort)
    library = a.obs[cfg["library_key"]].astype(str).to_numpy()
    batch = a.obs[cfg["batch_key"]].astype(str).to_numpy()
    model_lab = a.obs[cfg["model_key"]].astype(str).to_numpy() if cfg["model_key"] else None
    info = dict(cohort=cohort, setting=setting, rep=rep, n_cells_base=int(a.n_obs))
    kind = st["kind"]
    X1 = X0.astype(np.float64)
    keep = np.ones(a.n_obs, bool)
    lab1 = lab.copy()

    def model_of(direct):
        if not cfg["coarse_map"]:
            return direct
        return np.array([RU.TESTIS_MODEL_PARENT[v] for v in direct])

    if kind == "none":
        pass
    elif kind == "relabel":
        lab1 = relabel(rng, lab, spc, st["q"])
        model_lab = model_of(lab1) if model_lab is not None else None
    elif kind == "relabel_near":
        lab1 = relabel_near(rng, X1, lab, spc, st["q"])
        model_lab = model_of(lab1) if model_lab is not None else None
    elif kind == "coarse":
        lab1 = pd.Series(lab).map(RU.COARSE[cohort]).to_numpy()
        assert not pd.isna(lab1).any()
        model_lab = lab1 if model_lab is not None else None
    elif kind == "fine":
        lab1 = fine_labels(X1, lab, spc, RU.useed("fine", cohort, rep) % 100000)
        model_lab = lab1 if model_lab is not None else None
    elif kind == "species_diff":
        lab1 = species_diff(lab, spc)
        model_lab = lab1 if model_lab is not None else None
    elif kind == "nolabel":
        lab1 = nolabel(X1, spc, lab, RU.useed("leiden", cohort, rep) % 100000, st["r_min"])
        model_lab = None  # no-label form: recipe without conditional MMD
    elif kind == "type_missing":
        types = shared_by_size(lab, spc)
        ty = types[1] if rep in (0, 1) else types[2]
        sp_ = "Human" if rep == 1 else "Mouse"
        keep &= ~((lab == ty) & (spc == sp_))
        info.update(type_removed=ty, species_removed=sp_)
    elif kind == "rare_types":
        big = shared_by_size(lab, spc)[:2]
        for s in RU.SPECIES:
            for ty in set(lab[spc == s]) - set(big):
                idx = np.flatnonzero((lab == ty) & (spc == s))
                if len(idx) > RU.RARE_CELLS:
                    keep[rng.choice(idx, len(idx) - RU.RARE_CELLS, replace=False)] = False
        info.update(types_kept=big)
    elif kind == "size":
        keep &= species_sizes(rng, spc, st["ratio"], rep)
    elif kind == "cells":
        for s in RU.SPECIES:
            idx = np.flatnonzero(spc == s)
            keep[rng.choice(idx, len(idx) - int(round(st["frac"] * len(idx))), replace=False)] = False
    elif kind == "depth":
        X1 = thin(rng, X1, st["frac"])
    elif kind == "few_donors":
        donor = a.obs[cfg["donor_key"]].astype(str).to_numpy()
        keep &= donors_kept(rng, donor, spc, RU.FEW_DONORS[cohort], rep)
    elif kind == "extra_batch":
        b1, lib1 = batch.astype(object).copy(), library.astype(object).copy()
        parts = []
        for s in RU.SPECIES:
            idx = np.flatnonzero(spc == s)
            pick = rng.choice(idx, len(idx) // 2, replace=False)
            factor = np.exp(-np.abs(rng.normal(0.0, RU.XB_SIGMA, X1.shape[1]))) * RU.XB_DEPTH
            parts.append((pick, factor))
            b1[pick] = [f"{v}_x" for v in batch[pick]]
            lib1[pick] = [f"{v}_x" for v in library[pick]]
        X1 = X1.tocsr().copy()
        for pick, factor in parts:
            Y = thin(rng, X1[pick], factor)
            mask = np.ones(X1.shape[0], bool)
            mask[pick] = False
            order = np.concatenate([np.flatnonzero(mask), pick])
            inv = np.argsort(order)
            X1 = sp.vstack([X1[mask], Y]).tocsr()[inv]
        batch, library = b1.astype(str), lib1.astype(str)
    elif kind == "chemistry":
        sp_ = "Mouse" if rep == 1 else "Human"
        prof = chem_profile()
        ids = np.asarray(gene_ids.ensembl(cfg["gene_table"], a.var_names))
        k = pd.Series(ids).map(prof).to_numpy(dtype=float)
        k = np.where(np.isfinite(k), k, float(np.nanmedian(k)))
        m = spc == sp_
        Y = thin(rng, X1[m], k)
        order = np.concatenate([np.flatnonzero(~m), np.flatnonzero(m)])
        X1 = sp.vstack([X1[~m], Y]).tocsr()[np.argsort(order)]
        info.update(species_thinned=sp_, median_keep=float(np.median(k)))
    else:
        raise ValueError(kind)

    L = np.asarray(X1.sum(1)).ravel()
    n_zero = int(((L < RU.MIN_LIB) & keep).sum())
    keep &= L >= RU.MIN_LIB
    info.update(n_cells_zero_dropped=n_zero, n_cells=int(keep.sum()))
    b = a[keep].copy()
    X1k = X1[keep].tocsr()
    b.X = X1k.astype(np.float32)
    b.obs["b2_label"] = lab1[keep]
    b.obs["b2_library"] = library[keep]
    b.obs[cfg["batch_key"]] = batch[keep]
    if cfg["model_key"]:
        b.obs[cfg["model_key"]] = (model_lab if model_lab is not None else lab1)[keep]
    b.uns["p_null"] = p_null(X1k.astype(np.float64), spc[keep])
    Fn = features_after(np.asarray(a.varm["gene_features"]), X0, X1k)
    b.varm["gene_features"] = Fn.astype(np.asarray(a.varm["gene_features"]).dtype)
    s1 = spc[keep]
    info.update(n_by_species={s: int((s1 == s).sum()) for s in RU.SPECIES},
                n_by_label=pd.crosstab(lab1[keep], s1).to_dict())
    if out_path is not None:
        b.write_h5ad(out_path)
        if check:
            from revision.common.check_cohort import check as cc

            ct = cfg["model_key"] if st["kind"] != "nolabel" else None
            errs = cc(str(out_path), batch_key=cfg["batch_key"], celltype_key=ct)
            if errs:
                raise AssertionError(f"check_cohort: {errs}")
    return b, info
