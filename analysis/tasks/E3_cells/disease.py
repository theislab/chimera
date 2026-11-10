"""E3 analysis 4 (post hoc, labelled): does the healthy count margin read IN THE DISEASE CELL TYPE predict which
genes change the same way in human and mouse disease, beside the whole-gene readouts? C1's H-D1 statistic
unchanged (rho_p(score, Q | C, |l^h|, SE(m)), Q = sign(l^h) m; C1's DE, universe, permutations and ESM-block
bootstrap via ``tests_c1.rho_table``), with new scores: S_type = mean of the human and mouse rows' per-cell
margin S_gsk in the matching type (per-gene gate and canonical), and D_free.

    PYTHONPATH=$PWD python analysis/tasks/E3_cells/disease.py
Output ``results/analysis/E3/disease_hd1_type.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tasks.C1_cohorts import configs as CF
from analysis.tasks.C1_cohorts import tests_c1 as T
from analysis.common import gene_ids
from analysis.tasks.E3_cells import compartments as CP
from analysis.tasks.E3_cells import rules as R

CASES = {
    "muscle": ("muscle_ageing", {"artec": ["ArtEC"], "capec": ["CapEC"], "fb": ["FB"], "macrophage": ["Macrophage"], "musc": ["MuSC"],
                                 "pericyte": ["Pericyte"], "smc": ["SMC"], "venec": ["VenEC"]}),
    "lung": ("lung_healthy", {"at2": ["AT2"], "dc": ["cDC"], "endothelial": ["Aerocyte", "Capillary", "Artery", "Vein"],
                              "macrophage": ["Macrophage_alveolar"], "t": ["T"]}),
    "ad": ("brain_m1_hmouse", {"microglia": ["Micro"], "astrocyte": ["Astro"], "oligodendrocyte": ["Oligo"]}),
}


def type_margin(c: str, key: str, labels: list[str]) -> pd.Series:
    b = CP.load_build(c)
    sp = [str(s).lower() for s in b["species"]]
    lab = b["labels"].astype(str)
    rows = [i for i, (si, l) in enumerate(zip(b["species_idx"], lab)) if l in labels and sp[si] in ("human", "mouse")]
    return pd.Series(np.nanmean(b[key][rows], axis=0), index=b["var_names"])


def main() -> int:
    rng = np.random.default_rng(T.SEED + 1)
    out = []
    for case, (model, tmap) in CASES.items():
        g = T.genes(model)
        e3 = pd.read_parquet(R.OUT / "genes" / f"{model}.parquet")
        g["D_free"] = e3["D_free"].reindex(g["var_name"]).to_numpy()
        tab = gene_ids.load_table(model)
        ids = pd.Index(g["var_name"]).str.replace("ortho_", "", regex=False).astype(int)
        g["mouse_id"] = tab.loc[ids, "mmusculus_gene_id"].to_numpy()
        de = CF.RES / case / "de"
        for t, labels in tmap.items():
            fh, fm = de / f"human_primary_{t}.tsv", de / f"mouse_primary_{t}.tsv"
            if not (fh.exists() and fm.exists()):
                continue
            h = pd.read_csv(fh, sep="\t").dropna(subset=["log2fc_mle", "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
            m = pd.read_csv(fm, sep="\t").dropna(subset=["log2fc_fixed", "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
            u = g.copy()
            u["S_type_free"] = type_margin(model, "S_free", labels).reindex(u["var_name"]).to_numpy()
            u["S_type_can"] = type_margin(model, "S_can", labels).reindex(u["var_name"]).to_numpy()
            u["l"] = h["log2fc_mle"].reindex(u["hsapiens_gene_id"]).to_numpy()
            u["det_h"] = h["det_frac"].reindex(u["hsapiens_gene_id"]).to_numpy()
            u["m"] = m["log2fc_fixed"].reindex(u["mouse_id"]).to_numpy()
            u["se_m"] = m["lfcse_fixed"].reindex(u["mouse_id"]).to_numpy()
            u["det_m"] = m["det_frac"].reindex(u["mouse_id"]).to_numpy()
            u = u[(u.det_h >= 0.05) & (u.det_m >= 0.05)].dropna(subset=["l", "m", "se_m"]).copy()
            u["abs_l"] = u["l"].abs()
            u["Q"] = np.sign(u["l"]) * u["m"]
            out += T.rho_table(u, "Q", ["D", "D_free", "S_type_free", "S_type_can", "S_ema", "B1"], 2000, 2000, rng,
                               dict(case=case, cell_type=t, model=model), extra_nuis=("abs_l", "se_m"))
            print(case, t, flush=True)
    pd.DataFrame(out).to_csv(R.OUT / "disease_hd1_type.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
