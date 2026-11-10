"""X7 (EXPLORATORY): baselines beside the per-cell residual share and the species margins (rules.py).

Per run label: cell table with the Chimera per-cell outputs (forward npz) and normalised kNN species purity in
Chimera z and (canonical runs) scVI z. Per cohort, from the training counts (T4.2 pseudobulk): per-type direct
divergence between species, gene x type direct divergence Ddir_gk with level and detection, and pairwise E per
species pair (T4.2 E on that pair's shared types). Writes results/analysis/X7/baselines/.

Usage: python -m analysis.tasks.X7_architecture_outputs.baselines
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X7_architecture_outputs import rules as RU
from analysis.tasks.X7_architecture_outputs.forward import FWD, OUT, run_table

BD = OUT / "baselines"


def tissue(c):
    return "testis" if c.startswith("test") else "liver"


def coarse(c: str, lab: np.ndarray) -> np.ndarray:
    s = pd.Series(lab)
    if tissue(c) == "testis":
        return s.map(T42.TESTIS_COARSE).to_numpy()
    return s.replace(RU.LIVER_MACRO).to_numpy()


def npurity(z: np.ndarray, species: np.ndarray, k: int = RU.K_NN) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, idx = nn.kneighbors(z)
    idx = idx[:, 1:]
    pur = (species[idx] == species[:, None]).mean(1)
    f = pd.Series(species).map(pd.Series(species).value_counts(normalize=True)).to_numpy()
    return (pur - f) / (1 - f)


def scvi_z(c: str) -> np.ndarray:
    stem = gene_ids.H5ADS[c]
    d = C.RUNS_DIR / "baselines_scvi" / f"{stem}-s{RU.SCVI_SEED}-e100-nl10-bk_{RU.LIB_KEY[tissue(c)]}"
    return np.load(d / "z_mean.npy")


def cell_tables():
    BD.mkdir(parents=True, exist_ok=True)
    for lab, (c, run) in run_table().items():
        f = FWD / f"{lab}.npz"
        if not f.exists():
            print("missing forward", lab, flush=True)
            continue
        z = np.load(f, allow_pickle=False)
        spn = z["species"][z["obs_species"]]
        ct = z["obs_cell_type"]
        t = pd.DataFrame(dict(obs_name=z["obs_names"], species=spn, cell_type=ct, coarse=coarse(c, ct),
                              tv=z["tv"], gain=z["gain"], lshare=z["lshare"], lib=z["lib"]))
        t["npur_chimera"] = npurity(z["z"], spn)
        if lab == c:
            zs = scvi_z(c)
            if zs.shape[0] != len(t):
                raise AssertionError(f"{c}: scVI cells {zs.shape[0]} vs {len(t)}")
            t["npur_scvi"] = npurity(zs, spn)
            # order check: scVI neighbours should share cell type far above chance
            nn = NearestNeighbors(n_neighbors=16).fit(zs)
            _, idx = nn.kneighbors(zs)
            same = (ct[idx[:, 1:]] == ct[:, None]).mean()
            print(f"  {c}: scVI kNN same-type share {same:.2f}", flush=True)
            if same < 0.5:
                raise AssertionError(f"{c}: scVI z does not follow the h5ad cell order")
        t.to_parquet(BD / f"{lab}__cells.parquet")
        print("cells", lab, len(t), flush=True)


def pseudobulk(c: str):
    a = C.read_adata(c)
    X = (a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lib = a.obs[RU.LIB_KEY[tissue(c)]].astype(str).to_numpy()
    ct = a.obs["cell_type"].astype(str).to_numpy()
    res = {"coarse": T42.pseudobulk(X, spc, coarse(c, ct), lib, RU.MIN_CELLS)[:2]}
    if tissue(c) == "testis":
        res["fine"] = T42.pseudobulk(X, spc, ct, lib, RU.MIN_CELLS)[:2]
    hs = gene_ids.ensembl(c, pd.Index(a.var_names)).to_numpy()
    return res, sorted(set(spc)), np.array(a.var_names, dtype=str), hs


def direct(c: str):
    res, species, vn, hs = pseudobulk(c)
    rows = []
    for level, (x, f) in res.items():
        types = sorted({t for _, t in x})
        for s1, s2 in itertools.combinations(species, 2):
            for t in types:
                if (s1, t) not in x or (s2, t) not in x:
                    continue
                ok = (f[(s1, t)] >= RU.DET_MIN) & (f[(s2, t)] >= RU.DET_MIN)
                r = spearmanr(x[(s1, t)][ok], x[(s2, t)][ok])[0]
                rows.append(dict(cohort=c, level=level, s1=s1, s2=s2, type=t, n_genes=int(ok.sum()), rho=r, div=1 - r))
    # gene x type direct divergence (coarse)
    x, f = res["coarse"]
    types = sorted({t for _, t in x})
    gt = pd.DataFrame(dict(var_name=vn, hsapiens_gene_id=hs))
    for t in types:
        have = [s for s in species if (s, t) in x]
        if len(have) < 2:
            continue
        Xs = np.column_stack([x[(s, t)] for s in have])
        Fs = np.column_stack([f[(s, t)] for s in have])
        ok = (Fs >= RU.DET_MIN).all(1)
        off = np.array([np.median(Xs[ok, j] - Xs[ok].mean(1)) for j in range(len(have))])
        Xc = Xs - off[None, :]
        gt[f"Ddir_{t}"] = np.where(ok, Xc.std(1), np.nan)
        gt[f"lev_{t}"] = Xs.mean(1)
        gt[f"det_{t}"] = Fs.mean(1)
        for j, s in enumerate(have):
            gt[f"x_{s}_{t}"] = Xs[:, j]
            gt[f"f_{s}_{t}"] = Fs[:, j]
    # pairwise E (T4.2 definition, each pair's shared coarse types)
    pe = pd.DataFrame(dict(var_name=vn, hsapiens_gene_id=hs))
    for s1, s2 in itertools.combinations(species, 2):
        g = T42.shared_groups(x, species, [s1, s2])
        r, _ = T42.pair_corr(x, f, s1, s2, g, len(vn))
        pe[f"E_{s1}_{s2}"] = r
        pe[f"K_{s1}_{s2}"] = len(g)
    return pd.DataFrame(rows), gt, pe


def main():
    cell_tables()
    allr = []
    for c in C.CANONICAL:
        r, gt, pe = direct(c)
        allr.append(r)
        gt.to_parquet(BD / f"{c}__gene_type_direct.parquet")
        pe.to_parquet(BD / f"{c}__pair_E.parquet")
        print("direct", c, flush=True)
    pd.concat(allr).to_csv(BD / "type_divergence_direct.csv", index=False)


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        main()
