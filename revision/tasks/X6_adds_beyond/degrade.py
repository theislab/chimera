"""X6 (EXPLORATORY): label sets and the label-dependent direct scores under each (rules.py). No gold is read.

Writes results/revision/X6/cache/labels__<cohort>.parquet (one column per condition),
leiden_match__<cohort>.csv (cluster pairs, r, majority true label, purity), and
degraded__<cohort>.parquet (B1b_raw, B1c_raw, B2b, Sdir per condition, keyed by var_name), and
checks that the full condition reproduces T4.2 (B1b_raw, B1c_raw, B2b) and X4 (Sdir).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import zlib

import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.common import cohorts as CO
from revision.common import controls as K
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.X4_where_differs.direct_margin import nb_ll_diff
from revision.tasks.X6_adds_beyond import rules as RU

OUT = T42.gene_ids.REPO_ROOT / "results" / "revision" / "X6"
CACHE = OUT / "cache"


def useed(*parts) -> int:
    return (zlib.crc32("|".join(map(str, parts)).encode()) ^ RU.SEED) & 0x7FFFFFFF


def base_labels(c: str, a) -> np.ndarray:
    lab = a.obs["cell_type"].astype(str)
    if c.startswith("test"):
        lab = lab.map(T42.TESTIS_COARSE)
    assert lab.notna().all()
    return lab.to_numpy()


def noisy(lab: np.ndarray, spc: np.ndarray, q: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = lab.copy()
    for s in np.unique(spc):
        idx = np.flatnonzero(spc == s)
        types, cnt = np.unique(lab[idx], return_counts=True)
        pick = rng.choice(idx, int(round(q * len(idx))), replace=False)
        for i in pick:
            w = np.where(types == lab[i], 0.0, cnt.astype(float))
            out[i] = types[rng.choice(len(types), p=w / w.sum())]
    assert all(abs((out[spc == s] != lab[spc == s]).mean() - q) < 1e-3 for s in np.unique(spc))
    return out


def leiden_species(X, P: dict) -> np.ndarray:
    import anndata as ad
    import scanpy as sc

    a = ad.AnnData(X=X.copy())
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat", n_top_genes=P["n_top_genes"])
    h = a[:, a.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=P["n_pcs"], svd_solver="arpack", random_state=P["seed"])
    sc.pp.neighbors(h, n_neighbors=P["n_neighbors"], n_pcs=P["n_pcs"], random_state=P["seed"])
    sc.tl.leiden(h, resolution=P["resolution"], flavor="igraph", n_iterations=2, directed=False,
                 random_state=P["seed"])
    return h.obs["leiden"].astype(str).to_numpy()


def leiden_matched(X, spc: np.ndarray, lab_true: np.ndarray, c: str):
    P = RU.LEIDEN
    species = sorted(set(spc))
    assert species == ["Human", "Mouse"]
    cl = np.empty(len(spc), dtype=object)
    prof, info = {}, []
    for s in species:
        m = np.flatnonzero(spc == s)
        Xs = X[m]
        k = leiden_species(Xs, P)
        cl[m] = [f"{s}_c{int(v):02d}" for v in k]
        L = np.asarray(Xs.sum(1)).ravel()
        Xn = sp.diags(1e4 / L) @ Xs
        Xn.data = np.log1p(Xn.data)
        Z, D, names = [], [], []
        for v in sorted(set(cl[m])):
            mm = cl[m] == v
            if mm.sum() < P["min_cells"]:
                continue
            Z.append(np.asarray(Xn[mm].mean(0)).ravel())
            D.append(np.asarray((Xs[mm] > 0).mean(0)).ravel())
            names.append(v)
            tl, tc = np.unique(lab_true[m][mm], return_counts=True)
            info.append(dict(species=s, cluster=v, n_cells=int(mm.sum()), majority=tl[tc.argmax()],
                             purity=float(tc.max() / tc.sum())))
        Z = np.array(Z)
        sd = Z.std(0)
        Z = np.where(sd > 0, (Z - Z.mean(0)) / np.where(sd > 0, sd, 1), 0.0)
        prof[s] = (names, Z, np.array(D))
    markers = set()
    for s in species:
        names, Z, D = prof[s]
        for i in range(len(names)):
            z = np.where(D[i] >= P["marker_det"], Z[i], -np.inf)
            markers |= set(np.argsort(-z)[:P["n_markers"]].tolist())
    M = np.array(sorted(markers))
    (hn, hZ, _), (mn, mZ, _) = prof["Human"], prof["Mouse"]
    A, B = hZ[:, M], mZ[:, M]
    A = (A - A.mean(1, keepdims=True)) / A.std(1, keepdims=True)
    B = (B - B.mean(1, keepdims=True)) / B.std(1, keepdims=True)
    R = A @ B.T / len(M)
    pairs = []
    for i in range(len(hn)):
        j = int(R[i].argmax())
        if int(R[:, j].argmax()) == i and R[i, j] >= P["r_min"]:
            pairs.append((hn[i], mn[j], float(R[i, j])))
    out = cl.copy()
    for k, (h, m_, r) in enumerate(pairs):
        out[cl == h] = f"P{k:02d}"
        out[cl == m_] = f"P{k:02d}"
    inf = pd.DataFrame(info)
    pair_of = {h: (f"P{k:02d}", r) for k, (h, m_, r) in enumerate(pairs)} | {m_: (f"P{k:02d}", r) for k, (h, m_, r) in enumerate(pairs)}
    inf["pair"] = inf.cluster.map(lambda v: pair_of.get(v, (None, np.nan))[0])
    inf["r"] = inf.cluster.map(lambda v: pair_of.get(v, (None, np.nan))[1])
    inf["best_r"] = [float(R[hn.index(v)].max()) if v in hn else float(R[:, mn.index(v)].max()) for v in inf.cluster]
    inf["n_markers_union"] = len(M)
    inf.insert(0, "cohort", c)
    return out.astype(str), inf


def specificity_cos(x: dict, f: dict, species: list[str], n: int) -> np.ndarray:
    """B2b on the groups shared by all species, for any K >= 2 (T4.2 formula)."""
    groups = T42.shared_groups(x, species, species)
    if len(groups) < 2:
        return np.full(n, np.nan)
    s1, s2 = species
    det = (np.column_stack([f[(s1, t)] for t in groups]).max(1) >= T42.DET_MIN) & \
          (np.column_stack([f[(s2, t)] for t in groups]).max(1) >= T42.DET_MIN)
    P1 = np.column_stack([2.0 ** x[(s1, t)] for t in groups])
    P2 = np.column_stack([2.0 ** x[(s2, t)] for t in groups])
    P1 /= P1.sum(1, keepdims=True)
    P2 /= P2.sum(1, keepdims=True)
    cos = (P1 * P2).sum(1) / (np.linalg.norm(P1, axis=1) * np.linalg.norm(P2, axis=1))
    return np.where(det, cos, np.nan)


def sdir(X, spc: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """X4 direct margin with the given labels (direct_margin.run, total only)."""
    L = np.asarray(X.sum(1)).ravel()
    G = X.shape[1]
    species = sorted(set(spc))
    Ksp = len(species)
    groups = sorted({(s, t) for s, t in zip(spc, lab)})
    rate, n_sk = {}, {}
    for s, t in groups:
        m = (spc == s) & (lab == t)
        n_sk[(s, t)] = int(m.sum())
        rate[(s, t)] = (np.asarray(X[m].sum(0)).ravel() + 0.5) / (L[m].sum() + 0.5 * G)
    p = {s: (np.asarray(X[spc == s].sum(0)).ravel() + 0.5) / (L[spc == s].sum() + 0.5 * G) for s in species}
    rbar = {t: np.mean([rate[(s, t)] for s in species if (s, t) in rate], 0) for t in {t for _, t in groups}}
    num, den = np.zeros(G), np.zeros(G)
    for s, t in groups:
        m = np.where((spc == s) & (lab == t))[0]
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            mu = np.outer(L[mm], rate[(s, t)])
            xx = np.asarray(X[mm].todense())
            num += (((xx - mu) ** 2) - mu).sum(0)
            den += (mu ** 2).sum(0)
    th = 1.0 / np.clip(num / np.maximum(den, 1e-12), 1e-3, 1e3)
    out = np.zeros(G)
    for s, t in groups:
        m = np.where((spc == s) & (lab == t))[0]
        tot = np.zeros(G)
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            xx = np.asarray(X[mm].todense())
            tot += nb_ll_diff(xx, np.outer(L[mm], rbar[t]), np.outer(L[mm], p[s]), th).sum(0)
        out += n_sk[(s, t)] / (np.sum(spc == s) * Ksp) * tot / n_sk[(s, t)]
    return out


def scores(X, spc, lab, library, min_cells: int, n: int) -> tuple[dict, dict]:
    species = sorted(set(spc))
    x, f, units = T42.pseudobulk(X, spc, lab, library, min_cells)
    E, _, pinfo = T42.pooled_E(x, f, species, n)
    Lv, _, B2, _, groups = T42.level_and_specificity(x, f, species, n)
    if len(groups) < 4:
        B2 = specificity_cos(x, f, species, n)
    S = sdir(X, spc, lab)
    meta = dict(K_pair=pinfo[0]["K"], groups=";".join(pinfo[0]["groups"]), K_all=len(groups),
                B2b_note="K3 exploratory" if len(groups) < 4 else "",
                units_used=int(units.used.sum()), units_total=len(units))
    return dict(B1b_raw=E, B1c_raw=Lv, B2b=B2, Sdir=S), meta


def run(c: str):
    a = CO.read_adata(c)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    library = a.obs[T42.LIBRARY_KEY["testis" if c.startswith("test") else "liver"]].astype(str).to_numpy()
    lab0 = base_labels(c, a)
    labs = {"full": lab0, "coarse3": pd.Series(lab0).map(RU.COARSE3[c]).to_numpy()}
    assert not pd.isna(labs["coarse3"]).any()
    labs["leiden"], match = leiden_matched(X, spc, lab0, c)
    for k, q in RU.NOISE.items():
        for r in range(RU.NOISE_REPS):
            labs[f"{k}_r{r}"] = noisy(lab0, spc, q, useed(c, k, r))
    lt = pd.DataFrame(labs, index=a.obs_names)
    lt["species"] = spc
    CACHE.mkdir(parents=True, exist_ok=True)
    lt.to_parquet(CACHE / f"labels__{c}.parquet")
    K.write_csv(match, CACHE / f"leiden_match__{c}.csv")
    cols, metas = {}, []
    n = a.n_vars
    for cond in RU.conditions():
        mc = RU.MIN_CELLS["noise" if cond.startswith("noise") else cond]
        sc_, meta = scores(X, spc, labs[cond], library, mc, n)
        metas.append(dict(cohort=c, condition=cond, min_cells=mc, **meta,
                          **{f"n_{k}": int(np.isfinite(v).sum()) for k, v in sc_.items()}))
        for k, v in sc_.items():
            cols[f"{k}__{cond}"] = v
        print(c, cond, metas[-1], flush=True)
    df = pd.DataFrame(cols, index=pd.Index(a.var_names, name="var_name")).reset_index()
    df.to_parquet(CACHE / f"degraded__{c}.parquet")
    K.write_csv(pd.DataFrame(metas), CACHE / f"degraded_meta__{c}.csv")
    # reproduction of the full condition
    b = pd.read_csv(T42.gene_ids.REPO_ROOT / f"results/revision/T4_3/baselines_consensusD/baselines__{c}.csv").set_index("var_name")
    xs = pd.read_parquet(T42.gene_ids.REPO_ROOT / f"results/revision/X4/cache/Sdir__{c}.parquet").set_index("var_name")
    d = df.set_index("var_name")
    rows = []
    for k in ("B1b_raw", "B1c_raw", "B2b"):
        u, v = d[f"{k}__full"], b[k].reindex(d.index)
        rows.append(dict(cohort=c, score=k, same_defined=bool((u.notna() == v.notna()).all()),
                         max_abs_diff=float(np.nanmax(np.abs(u - v)))))
    u, v = d["Sdir__full"], xs["Sdir"].reindex(d.index)
    rows.append(dict(cohort=c, score="Sdir", same_defined=bool((u.notna() == v.notna()).all()),
                     max_abs_diff=float(np.nanmax(np.abs(u - v)))))
    rep = pd.DataFrame(rows)
    K.write_csv(rep, CACHE / f"degraded_repro__{c}.csv")
    print(rep.to_string(), flush=True)


def main():
    import sys

    for c in (sys.argv[1:] or RU.COHORTS):
        run(c)


if __name__ == "__main__":
    main()
