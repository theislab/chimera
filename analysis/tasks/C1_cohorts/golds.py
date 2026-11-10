"""C1: per-gene human-partner profile correlation in data never used in training (T4.2 `pair_corr` rule on unit pseudobulks).

Mean over units with at least 30 cells of log2(1 + CPM) on the 1:1 human-partner genes, per class; a gene is scored if detected in >= 1% of a class in both species (T4.2);
Pearson over shared classes of species-centred profiles. No model and no Chimera readout is read.

    python analysis/tasks/C1_cohorts/golds.py          # writes results/analysis/C1/golds/<region>_<partner>.parquet
"""
from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from analysis.tasks.C1_cohorts import configs as CF
from analysis.tasks.C1_cohorts import orthologs as OR
from analysis.tasks.T4_2_gene_baselines import run as T42

OUT = CF.RES / "golds"
MIN_CELLS = 30


def prof(pb: ad.AnnData, genes: pd.Index) -> tuple[dict, dict]:
    keep = pb.obs["cell_ontology_class"].astype(str) != "DROP"
    pb = pb[keep.to_numpy()]
    idx = pb.var_names.get_indexer(genes)
    X = pb.X.tocsr() if hasattr(pb.X, "tocsr") else pb.X
    Dn = pb.layers["n_detected"]
    x, f = {}, {}
    G = len(genes)
    total = np.asarray(X.sum(1)).ravel()  # library over all genes of the species, as CPM in T4.2 on the species' vocabulary after the 1:1 join below
    for t, g in pb.obs.reset_index(drop=True).groupby("cell_ontology_class", observed=True):
        rows = g.index[g["n_cells"].astype(int) >= MIN_CELLS].to_numpy()
        if len(rows) == 0:
            continue
        A = np.full((len(rows), G), 0.0)
        cc = np.zeros(G)
        nn = 0
        for j, r in enumerate(rows):
            v = np.zeros(G)
            present = idx >= 0
            v[present] = np.asarray(X[r, idx[present]].todense()).ravel()
            A[j] = np.log2(1.0 + 1e6 * v / max(v.sum(), 1.0))
            d = np.zeros(G)
            d[present] = np.asarray(Dn[r, idx[present]].todense()).ravel()
            cc += d
            nn += int(pb.obs["n_cells"].iloc[r])
        x[str(t)] = A.mean(0)
        f[str(t)] = cc / nn
    return x, f


def gold(src: str, partner: str) -> pd.DataFrame:
    t, _ = OR.pair_rows(partner)
    a = ad.read_h5ad(CF.BUILD / "species" / src / "hsapiens" / "pseudobulk.h5ad")
    b = ad.read_h5ad(CF.BUILD / "species" / src / partner / "pseudobulk.h5ad")
    t = t[t["hsapiens_gene_id"].isin(a.var_names) & t[f"{partner}_gene_id"].isin(b.var_names)].reset_index(drop=True)
    xa, fa = prof(a, pd.Index(t["hsapiens_gene_id"]))
    xb, fb = prof(b, pd.Index(t[f"{partner}_gene_id"]))
    groups = sorted(set(xa) & set(xb))
    xx = {("h", k): v for k, v in xa.items()} | {("o", k): v for k, v in xb.items()}
    ff = {("h", k): v for k, v in fa.items()} | {("o", k): v for k, v in fb.items()}
    r, det = T42.pair_corr(xx, ff, "h", "o", groups, len(t))
    out = pd.DataFrame(dict(hsapiens_gene_id=t["hsapiens_gene_id"], gold=r, scorable=det, n_groups=len(groups)))
    print(src, partner, len(t), "groups", len(groups), "scored", int(np.isfinite(r).sum()), flush=True)
    # Direct level comparison, baseline for the level reference (E-B7): per class log2(1 + CPM) human minus partner; mean of the signed gap and of its size over shared classes.
    if groups:
        G = np.stack([xa[k] - xb[k] for k in groups])
        out["signed_gap"] = G.mean(0)
        out["level_gap"] = np.abs(G).mean(0)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    for src, partner, name in (("m1", "cjacchus", "M1_marmoset"), ("mtg", "cjacchus", "MTG_marmoset"), ("mtg", "mmulatta", "MTG_macaque"),
                               ("dlpfc", "cjacchus", "DLPFC_marmoset"), ("dlpfc", "mmulatta", "DLPFC_macaque"), ("dlpfc", "ptroglodytes", "DLPFC_chimpanzee"), ("m1", "mmusculus", "M1_mouse")):
        t = gold(src, partner)
        t.to_parquet(OUT / f"{name}.parquet")
        if name == "DLPFC_chimpanzee":
            t.to_parquet(OUT / "DLPFC_chimpanzee_level.parquet")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
