"""Per-tissue description (depth, K, support) and cell-half reliability of each gold on its universe.

    python -I launch.py analysis.tasks.Y1_gene_tissue.describe
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.Y1_gene_tissue import common as Y
from analysis.tasks.Y1_gene_tissue import contrast as CT
from analysis.tasks.Y1_gene_tissue import rules as RU


def main():
    rows = []
    for t in RU.TISSUES:
        m = json.loads((Y.CACHE / f"tissue_meta__{t}.json").read_text())
        rows.append(dict(tissue=t, K=m["K"], cells=sum(m["n_cells"].values()), cells_fit=m["n_cells_fit"], n_seeds=len(m["seeds"]), n_units=m["n_units"], n_genes=m["n_genes"], n_support=m["n_support"],
                         n_strict=m["n_strict"], n_B1=m["n_B1"], **{f"median_lib_{k.lower()}": v for k, v in m["median_library"].items()}))
    pd.DataFrame(rows).to_csv(RU.OUT / "tissues.csv", index=False)
    print(pd.DataFrame(rows).to_string())
    D = CT.Data()
    rel = []
    z4 = np.load(Y.RES / "T4_3" / "golds" / "splits.npz", allow_pickle=True)
    for gold, T in {**CT.GOLDS, "pancreas_noacinar": "islet"}.items():
        col = f"G_{gold}"
        g = CT.universe(D, T, col)
        Z = CT.design_T(D, T, g, CT.PAT, col)
        if gold in ("liver", "testis"):
            pre = "h3" if gold == "liver" else "h4"
            ids = z4[f"{pre}_genes"].astype(str)
            a, b = z4[f"{pre}_a"], z4[f"{pre}_b"]
            kind = "donor halves" if gold == "liver" else "library halves"
        else:
            z = np.load(RU.OUT / "gold" / f"splits_{gold}.npz")
            ids, a, b = z["genes"].astype(str), z["a"], z["b"]
            kind = "cell halves (technical)"
        ix = pd.Series(np.arange(len(ids)), index=ids).reindex(g).to_numpy()
        ok = np.isfinite(ix)
        r = S.split_rel(a[:, ix[ok].astype(int)].astype(float), b[:, ix[ok].astype(int)].astype(float), Z[ok])
        rel.append(dict(gold=gold, kind=kind, n=len(g), **r, sd_G=float(D.gold[col].reindex(g).std())))
    pd.DataFrame(rel).to_csv(RU.OUT / "gold_reliability.csv", index=False)
    print(pd.DataFrame(rel).round(3).to_string())


if __name__ == "__main__":
    main()
