"""B2 Track C: atlas level golds, written before any readout is joined. No Chimera readout is read.

G_lvl_<tissue>: per-type rank-matched level between two independent atlases, the B1c formula on the atlas profiles: within each type, genes are rank-normalised per
atlas (mid-ranks / n), L = -mean over types of |r_TS - r_TMS|; defined if detected in >= 1% of the cells of at least one type in both atlases. Liver from the H3 profiles;
kidney and pancreas from the committed Y1 label maps (TS vs TMS). Writes results/analysis/B2/inputs/atlas_level_golds.parquet (index: human ENSG).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from analysis.common import gene_ids
from analysis.tasks.B2_benchmark import rules as RU
from analysis.tasks.T4_3_scoreboard import golds as GD


def level(PT, PM, okT, okM) -> np.ndarray:
    n = PT.shape[0]
    d = []
    for j in range(PT.shape[1]):
        d.append(np.abs(rankdata(PT[:, j]) / n - rankdata(PM[:, j]) / n))
    L = -np.mean(d, 0)
    return np.where(okT & okM, L, np.nan)


def atlas(gold: str) -> pd.Series:
    import anndata as ad

    from analysis.tasks.Y1_gene_tissue import atlas_gold as AG
    from analysis.tasks.Y1_gene_tissue import rules as YR

    lm = pd.read_csv(YR.TASK / "gold_label_map.tsv", sep="\t", dtype=str)
    lm = lm[lm.gold == gold]
    types = list(dict.fromkeys(lm.coarse_type))
    tsmap = dict(zip(lm.ts_cell_type.dropna(), lm.coarse_type[lm.ts_cell_type.notna()]))
    tmsmap = dict(zip(lm.tms_cell_type.dropna(), lm.coarse_type[lm.tms_cell_type.notna()]))
    tab = gene_ids.load_table("kidney")
    tab = tab[tab.hsapiens_gene_id.notna() & tab.mmusculus_gene_id.notna()]
    fts, ftms = AG.FILES[gold]
    tsv = pd.Index(ad.read_h5ad(YR.ATLAS / fts, backed="r").raw.var_names)
    tmsv = pd.Index(ad.read_h5ad(YR.ATLAS / ftms, backed="r").raw.var_names)
    tab = tab[tab.hsapiens_gene_id.isin(tsv) & tab.mmusculus_gene_id.isin(tmsv)]
    hs, mm = tab.hsapiens_gene_id.to_numpy(), tab.mmusculus_gene_id.to_numpy()
    XT, cT, _ = AG.load_atlas(YR.ATLAS / fts, tsmap, hs, "10x 3' v3")
    XM, cM, _ = AG.load_atlas(YR.ATLAS / ftms, tmsmap, mm, None)
    PT, FT, _ = AG.profiles(XT, cT, types)
    PM, FM, _ = AG.profiles(XM, cM, types)
    return pd.Series(level(PT, PM, FT.max(1) >= GD.DET_MIN, FM.max(1) >= GD.DET_MIN), index=pd.Index(hs, name="hsapiens_gene_id"))


def main():
    out = {}
    z = np.load(RU.REPO / "results/analysis/T4_3/golds/h3_profiles.npz", allow_pickle=True)
    print({k: z[k].shape for k in z.files})
    ids = z["genes"]
    PT, PM = z["TS_x"], z["TMS_x"]
    FT = z["TS_f"]
    FM = z["TMS_f"]
    out["G_lvl_liver"] = pd.Series(level(PT, PM, FT.max(1) >= GD.DET_MIN, FM.max(1) >= GD.DET_MIN), index=pd.Index(ids, name="hsapiens_gene_id"))
    for g, name in (("kidney", "G_lvl_kidney"), ("pancreas", "G_lvl_pancreas")):
        out[name] = atlas(g)
    d = pd.DataFrame(out)
    (RU.RES / "inputs").mkdir(parents=True, exist_ok=True)
    d.to_parquet(RU.RES / "inputs" / "atlas_level_golds.parquet")
    print(d.notna().sum())


if __name__ == "__main__":
    main()
