"""Y1 post hoc golds (ledger addition C): living donor kidney vs TMS kidney; Muraro pancreas vs TMS pancreas. T4.3 H3 recipe unchanged.

Sub-golds: kidney_nephron = pt, tal, dct, pc, endothelial, mesangial (no immune type; bone marrow has none of them);
islet_noacinar = without acinar (the islet fit has no acinar class). No Chimera readout is read.

    python -I revision/tasks/Y1_gene_tissue/launch.py revision.tasks.Y1_gene_tissue.atlas_gold_posthoc
"""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

from revision.common import gene_ids
from revision.tasks.T4_3_scoreboard import golds as GD
from revision.tasks.Y1_gene_tissue import atlas_gold as AG
from revision.tasks.Y1_gene_tissue import rules as RU

FILES = {"kidney_fine": ("hs_kidney_living/living_donor_kidney.h5ad", "tms_kidney/tms_kidney_10x.h5ad"),
         "islet_fine": ("muraro_pancreas/muraro_pancreas.h5ad", "tms_pancreas/tms_pancreas_10x.h5ad")}
TAGS = [("kidney_fine", None, "kidney_fine"), ("kidney_fine", ["pt", "tal", "dct", "pc", "endothelial", "mesangial"], "kidney_nephron"),
        ("islet_fine", None, "islet_fine"), ("islet_fine", ["alpha", "beta", "delta", "pp", "ductal"], "islet_noacinar")]


def build(gold, keep):
    lm = pd.read_csv(RU.TASK / "gold_label_map_posthoc.tsv", sep="\t", dtype=str)
    lm = lm[lm.gold == gold]
    types = list(dict.fromkeys(lm.coarse_type))
    if keep is not None:
        types = [t for t in types if t in keep]
    hmap = dict(zip(lm.human_cell_type.dropna(), lm.coarse_type[lm.human_cell_type.notna()]))
    mmap = dict(zip(lm.tms_cell_type.dropna(), lm.coarse_type[lm.tms_cell_type.notna()]))
    hmap = {k: v for k, v in hmap.items() if v in types}
    mmap = {k: v for k, v in mmap.items() if v in types}
    tab = gene_ids.load_table("kidney")
    tab = tab[tab.hsapiens_gene_id.notna() & tab.mmusculus_gene_id.notna()]
    fh, fm = FILES[gold]
    hv = pd.Index(ad.read_h5ad(RU.ATLAS / fh, backed="r").raw.var_names)
    mv = pd.Index(ad.read_h5ad(RU.ATLAS / fm, backed="r").raw.var_names)
    tab = tab[tab.hsapiens_gene_id.isin(hv) & tab.mmusculus_gene_id.isin(mv)]
    hs, mm = tab.hsapiens_gene_id.to_numpy(), tab.mmusculus_gene_id.to_numpy()
    XH, cH, dH = AG.load_atlas(RU.ATLAS / fh, hmap, hs, None)
    XM, cM, dM = AG.load_atlas(RU.ATLAS / fm, mmap, mm, None)
    PH, FH, NH = AG.profiles(XH, cH, types)
    PM, FM, NM = AG.profiles(XM, cM, types)
    meta = dict(gold=gold, K=len(types), types=types, cells_human=dict(zip(types, NH)), cells_TMS=dict(zip(types, NM)), donors_human=len(set(dH)), donors_TMS=len(set(dM)), n_vocabulary=len(hs))
    if min(NH + NM) < GD.MIN_CELLS:
        raise AssertionError(f"{gold}: a type under {GD.MIN_CELLS} cells: {meta}")
    G = GD.profile_corr(PH, PM, FH.max(1) >= GD.DET_MIN, FM.max(1) >= GD.DET_MIN)
    meta["n_defined"] = int(np.isfinite(G).sum())
    return pd.DataFrame(dict(G=G, atlas_expr=(PH.mean(1) + PM.mean(1)) / 2, atlas_det=(FH.mean(1) + FM.mean(1)) / 2, atlas_amp=(PH.std(1) + PM.std(1)) / 2),
                        index=pd.Index(hs, name="hsapiens_gene_id")), meta


def main():
    d = RU.OUT / "gold"
    meta = {}
    for g, keep, tag in TAGS:
        t, m = build(g, keep)
        t.to_parquet(d / f"gold_{tag}.parquet")
        meta[tag] = m
        print(json.dumps(m), flush=True)
    (d / "gold_meta_posthoc.json").write_text(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
