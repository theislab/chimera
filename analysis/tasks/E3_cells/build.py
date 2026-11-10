"""E3 build: per cohort, one table of (species, cell type) groups with pseudobulk expression and the
count margin per group, for both read-outs.

* pseudobulk: summed counts / summed library x 1e4 (CP10K) and detection, per (species, type), from the
  cohort's training h5ad (H1 registry; the cells the model saw).
* canonical gate: S_gsk (mean over cells of the per-cell log-likelihood ratio, posterior-mean z) from the
  existing forward passes H1's drivers read (T2.5, P12, P13, Y1, T11, C1), mean over the seeds with a file.
* per-gene gate: the same from E3's forward passes of the Z4 fits (``forward_free.py``), when present.

    PYTHONPATH=$PWD python analysis/tasks/E3_cells/build.py [--cohort C]

Output ``results/analysis/E3/build/<cohort>.npz``: var_names, species, groups (species_idx, label, n),
w (T2.5 weight n_sk / (n_s K)), cp10k, det, S_can, S_can_within, S_free, S_free_within (NaN if missing),
seeds_can, seeds_free.
"""
from __future__ import annotations

import argparse

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.tasks.H1_harmonise import drivers as HD
from analysis.tasks.H1_harmonise import rules as RU
from analysis.tasks.T2_5_celltype_margin import margin as MG

RES = RU.RES
OUT = RES / "E3" / "build"


def can_files(c: str, d: dict, seeds: list[int]) -> tuple[list, str]:
    if d["role"] == "reference":
        base = RES / "T2_5" / "forward"
        fs = [(42, base / f"{c}.npz")] + [(s, base / f"{c}_s{s}.npz") for s in (43, 44)]
        return [(s, f) for s, f in fs if f.exists()], ("cell_ontology_class" if c.startswith("test") else "cell_type")
    if d.get("source") == "C1":
        return [(s, RES / "C1" / c / "drivers" / f"fw_s{s}.npz") for s in seeds], "cell_ontology_class"
    return HD.fw_files(c, seeds)


def load_groups(f, key: str) -> tuple[dict, str]:
    fw = dict(np.load(f, allow_pickle=False))
    k = key if f"{key}__groups" in fw else str(fw["keys"][0])
    if k == "y1half":
        fw, k = HD.fold_halves(fw, k)
    return fw, k


def margin_table(files: list, key: str, species: list[str], labels: list[tuple[int, str]], var_names: np.ndarray):
    """Seed-mean S_gsk (meanz, within) on the h5ad's (species, label) groups and genes."""
    acc, accw, used = None, None, []
    for s, f in files:
        if not f.exists():
            continue
        fw, k = load_groups(f, key)
        dec = MG.decompose(fw, k)
        sp_names = [str(x) for x in fw["species"]]
        gi = {(sp_names[i], l): j for j, (i, l) in enumerate(zip(dec["species_idx"], dec["labels"]))}
        vn = pd.Index(fw["var_names"].astype(str))
        cols = vn.get_indexer(var_names)
        assert (cols >= 0).all(), "forward var_names differ from the h5ad"
        M = np.full((len(labels), len(var_names)), np.nan)
        W = np.full_like(M, np.nan)
        for r, (si, l) in enumerate(labels):
            j = gi.get((species[si], l))
            if j is not None:
                M[r] = dec["S_gsk_meanz"][j][cols]
                W[r] = dec["S_gsk_within"][j][cols]
        acc = M if acc is None else acc + M
        accw = W if accw is None else accw + W
        used.append(s)
    if not used:
        return None, None, []
    return acc / len(used), accw / len(used), used


def build(c: str) -> None:
    d = RU.cohorts()[c]
    runs = pd.read_csv(RU.OUT / "runs.tsv", sep="\t")
    seeds = sorted(runs[(runs.cohort == c) & runs.kept].seed.astype(int))
    a = ad.read_h5ad(d["h5ad"])
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = a.obs[d["ct"]].astype(str).to_numpy()
    species = sorted(set(spc))
    labels = sorted({(species.index(s), l) for s, l in zip(spc, lab)})
    gidx = {p: i for i, p in enumerate(labels)}
    grp = np.array([gidx[(species.index(s), l)] for s, l in zip(spc, lab)])
    G = X.shape[1]
    lib = np.asarray(X.sum(1)).ravel()
    ind = sp.csr_matrix((np.ones(len(grp)), (grp, np.arange(len(grp)))), shape=(len(labels), len(grp)))
    sums = np.asarray((ind @ X).todense())
    det = np.asarray((ind @ (X > 0).astype(np.float64)).todense())
    n = np.asarray(ind.sum(1)).ravel()
    libs = ind @ lib
    cp10k = sums / libs[:, None] * 1e4
    det = det / n[:, None]
    n_s = np.array([n[[i for i, (s, _) in enumerate(labels) if s == k]].sum() for k in range(len(species))])
    w = np.array([n[i] / n_s[s] / len(species) for i, (s, _) in enumerate(labels)])
    vn = np.asarray(a.var_names, dtype=str)

    files, key = can_files(c, d, seeds)
    S_can, W_can, seeds_can = margin_table(files, key, species, labels, vn)
    ff = sorted((RES / "E3" / "forward_free" / c).glob("fw_s*.npz"))
    free = [(int(f.stem.split("_s")[-1]), f) for f in ff if not f.name.endswith(".tmp.npz")]
    S_free, W_free, seeds_free = margin_table(free, d["ct"], species, labels, vn)
    nanM = np.full((len(labels), G), np.nan)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{c}.npz", var_names=vn, species=np.array(species), labels=np.array([l for _, l in labels]),
                        species_idx=np.array([s for s, _ in labels]), n=n, w=w, cp10k=cp10k, det=det,
                        S_can=S_can if S_can is not None else nanM, S_can_within=W_can if W_can is not None else nanM,
                        S_free=S_free if S_free is not None else nanM, S_free_within=W_free if W_free is not None else nanM,
                        seeds_can=np.array(seeds_can, int), seeds_free=np.array(seeds_free, int))
    print(f"{c}: {len(labels)} groups, {G} genes, canonical seeds {seeds_can}, free seeds {seeds_free}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=None)
    a = ap.parse_args()
    for c in ([a.cohort] if a.cohort else list(RU.cohorts())):
        build(c)
