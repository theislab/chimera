"""Y1: independent pattern golds for kidney and pancreas, T4.3 H3 recipe unchanged. No Chimera readout is read.

Tabula Sapiens (10x 3' v3 cells only) vs Tabula Muris Senis (10x), raw counts in ``raw.X``, ages pooled. Types: committed
``gold_label_map.tsv``, every coarse type >= 30 cells in both atlases (checked). Per atlas and type: counts pooled over all
cells, x = log2(1 + 1e6 C / sum C), library size over the human-mouse 1:1 vocabulary (kidney ortholog table, the 16,010 groups
of Compara 110 from both dumps) present in both atlases. Scorable if detected in >= 1% of the cells of at least one type in both
atlases. G = Pearson across the K types of the atlas-centred profiles (T4.3 ``profile_corr``). NaN for a constant profile.

Also written: atlas covariates (mean log2 CPM, detection, amplitude = SD across types, mean of the two atlases) and cell-half
reliability (TS kidney has one donor, TS pancreas two usable, so donor halves do not exist; labelled technical).

    PYTHONPATH=<repo> python -m revision.tasks.Y1_gene_tissue.atlas_gold
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.common import gene_ids
from revision.tasks.T4_3_scoreboard import golds as GD
from revision.tasks.Y1_gene_tissue import rules as RU

FILES = {"kidney": ("ts_kidney/ts_kidney.h5ad", "tms_kidney/tms_kidney_10x.h5ad"),
         "pancreas": ("ts_pancreas/ts_pancreas.h5ad", "tms_pancreas/tms_pancreas_10x.h5ad")}


def load_atlas(path, mapping: dict, ids: np.ndarray, assay: str | None):
    a = ad.read_h5ad(path)
    obs = a.obs
    lab = obs["cell_type"].astype(str).to_numpy()
    ct = np.array([mapping.get(x, "") for x in lab])
    mk = ct != ""
    if assay is not None:
        mk &= obs["assay"].astype(str).to_numpy() == assay
    X = a.raw.X[np.flatnonzero(mk)]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    cols = pd.Index(a.raw.var_names).get_indexer(ids)
    if (cols < 0).any():
        raise AssertionError("gene missing from the atlas")
    X = X[:, cols].astype(np.float64)
    d = X.data
    if not (np.all(d >= 0) and np.all(d == np.round(d))):
        raise AssertionError(f"{path}: raw.X is not raw counts")
    return X, ct[mk], obs["donor_id"].astype(str).to_numpy()[mk]


def profiles(X, ct, types, cells=None):
    P, F, N = [], [], []
    for t in types:
        m = ct == t
        if cells is not None:
            m = m & cells
        Xt = X[np.flatnonzero(m)]
        S = np.asarray(Xt.sum(0)).ravel()
        P.append(np.log2(1 + 1e6 * S / max(S.sum(), 1)))
        Xb = Xt.copy()
        Xb.data = (Xb.data > 0).astype(np.float64)
        F.append(np.asarray(Xb.sum(0)).ravel() / max(m.sum(), 1))
        N.append(int(m.sum()))
    return np.column_stack(P), np.column_stack(F), N


def build(gold: str, types_keep=None):
    lm = pd.read_csv(RU.TASK / "gold_label_map.tsv", sep="\t", dtype=str)
    lm = lm[lm.gold == gold]
    types = list(dict.fromkeys(lm.coarse_type))
    if types_keep is not None:
        types = [t for t in types if t in types_keep]
    tsmap = dict(zip(lm.ts_cell_type.dropna(), lm.coarse_type[lm.ts_cell_type.notna()]))
    tmsmap = dict(zip(lm.tms_cell_type.dropna(), lm.coarse_type[lm.tms_cell_type.notna()]))
    tab = gene_ids.load_table("kidney")
    tab = tab[tab.hsapiens_gene_id.notna() & tab.mmusculus_gene_id.notna()]
    fts, ftms = FILES[gold]
    tsv = pd.Index(ad.read_h5ad(RU.ATLAS / fts, backed="r").raw.var_names)
    tmsv = pd.Index(ad.read_h5ad(RU.ATLAS / ftms, backed="r").raw.var_names)
    keep = tab.hsapiens_gene_id.isin(tsv) & tab.mmusculus_gene_id.isin(tmsv)
    tab = tab[keep]
    hs, mm = tab.hsapiens_gene_id.to_numpy(), tab.mmusculus_gene_id.to_numpy()
    XT, cT, dT = load_atlas(RU.ATLAS / fts, tsmap, hs, "10x 3' v3")
    XM, cM, dM = load_atlas(RU.ATLAS / ftms, tmsmap, mm, None)
    PT, FT, NT = profiles(XT, cT, types)
    PM, FM, NM = profiles(XM, cM, types)
    meta = dict(gold=gold, K=len(types), types=types, cells_TS=dict(zip(types, NT)), cells_TMS=dict(zip(types, NM)),
                donors_TS=int(len(set(dT))), donors_TMS=int(len(set(dM))), n_vocabulary=len(hs))
    if min(NT + NM) < GD.MIN_CELLS:
        raise AssertionError(f"{gold}: a type is under {GD.MIN_CELLS} cells: {meta}")
    G = GD.profile_corr(PT, PM, FT.max(1) >= GD.DET_MIN, FM.max(1) >= GD.DET_MIN)
    out = pd.DataFrame(dict(G=G, atlas_expr=(PT.mean(1) + PM.mean(1)) / 2, atlas_det=(FT.mean(1) + FM.mean(1)) / 2,
                            atlas_amp=(PT.std(1) + PM.std(1)) / 2), index=pd.Index(hs, name="hsapiens_gene_id"))
    rng = np.random.default_rng(RU.SEED)
    A, B = [], []
    for _ in range(20):  # cell halves stratified by type in each atlas
        def half(ct):
            h = np.zeros(len(ct), bool)
            for t in types:
                i = np.flatnonzero(ct == t)
                h[rng.permutation(i)[: len(i) // 2]] = True
            return h
        hT, hM = half(cT), half(cM)
        gs = []
        for sT, sM in ((hT, hM), (~hT, ~hM)):
            pT, fT, _ = profiles(XT, cT, types, sT)
            pM, fM, _ = profiles(XM, cM, types, sM)
            gs.append(GD.profile_corr(pT, pM, fT.max(1) >= GD.DET_MIN, fM.max(1) >= GD.DET_MIN))
        A.append(gs[0])
        B.append(gs[1])
    meta["n_defined"] = int(np.isfinite(G).sum())
    return out, np.array(A, dtype=np.float32), np.array(B, dtype=np.float32), meta


def main():
    d = RU.OUT / "gold"
    d.mkdir(parents=True, exist_ok=True)
    allm = {}
    for gold, keep, tag in (("kidney", None, "kidney"), ("pancreas", None, "pancreas"),
                            ("pancreas", ["ductal", "endothelial", "beta", "alpha", "leukocyte"], "pancreas_noacinar")):
        t, A, B, m = build(gold, keep)
        t.to_parquet(d / f"gold_{tag}.parquet")
        np.savez_compressed(d / f"splits_{tag}.npz", a=A, b=B, genes=t.index.to_numpy().astype(str))
        allm[tag] = m
        print(json.dumps(m), flush=True)
    (d / "gold_meta.json").write_text(json.dumps(allm, indent=1))


if __name__ == "__main__":
    main()
