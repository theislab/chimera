"""E3 robustness of the home-group excess (post hoc, labelled): per-gene gate seed by seed (each kept Z4 seed's
alpha minus the clamped prior, centred), canonical seed by seed, the strict home rule (peak minus mean >= 2 log2,
detection >= 20%), and all genes as universe (read from excess.csv).

    PYTHONPATH=$PWD python analysis/tasks/E3_cells/robust.py
Output ``results/analysis/E3/robust.csv``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.tasks.E3_cells import compartments as CP
from analysis.tasks.E3_cells import rules as R
from analysis.tasks.H1_harmonise import rules as RU


def strict_home(c: str, g: pd.DataFrame) -> pd.Series:
    b = CP.load_build(c)
    vn = pd.Index(b["var_names"])
    cols = vn.get_indexer(g.index)
    hi = CP.human_index(b)
    lab = b["labels"].astype(str)
    hrows = np.where((b["species_idx"] == hi) & (b["n"] >= R.MIN_HUMAN_CELLS))[0]
    H = np.log2(1 + b["cp10k"][:, cols] * 100)[hrows].T
    pk = H.argmax(1)
    spec = H.max(1) - H.mean(1)
    fpk = b["det"][hrows][:, cols].T[np.arange(len(g)), pk]
    tmap = R.type_map(RU.cohorts()[c]["tissue"])
    home = np.where((spec >= 2.0) & (fpk >= 0.20), lab[hrows][pk], "broad")
    return pd.Series([tmap[h][1] if h != "broad" else "broad" for h in home], index=g.index)


def main() -> int:
    rows = []
    rng = np.random.default_rng(R.SEED)
    for c in RU.cohorts():
        g = pd.read_parquet(R.OUT / "genes" / f"{c}.parquet")
        z = pd.read_parquet(R.RES / "Z4" / "h1" / "genes" / f"{c}.parquet").reindex(g.index)
        h = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet").reindex(g.index)
        for frame, tag in ((z, "free"), (h, "can")):
            for col in [k for k in frame.columns if k.startswith("alpha_s")]:
                d = frame[col] - frame.pi_clamped
                g[f"D_{tag}_{col[6:]}"] = d - d.median()
        g["subgroup_strict"] = strict_home(c, g)
        u = g.count_support.to_numpy()
        gu = g[u]
        Zr = gu[R.COV_PAT].to_numpy(float)
        seedcols = [k for k in g.columns if k.startswith("D_free_s") or k.startswith("D_can_s")]
        for grp in sorted(gu.subgroup.unique()):
            flag = (gu.subgroup == grp).to_numpy()
            for sc in seedcols:
                r = CP.knn_excess(gu[sc].to_numpy(float), Zr, flag, 300, rng)
                rows.append(dict(cohort=c, group=grp, variant=sc, **r))
        for grp in sorted(gu.subgroup_strict.unique()):
            flag = (gu.subgroup_strict == grp).to_numpy()
            for sc in ("D_free", "D", "S_ema", "B1", "DL"):
                r = CP.knn_excess(gu[sc].to_numpy(float), Zr, flag, 300, rng)
                rows.append(dict(cohort=c, group=grp, variant=f"strict_{sc}", **r))
        print(c, flush=True)
    pd.DataFrame(rows).to_csv(R.OUT / "robust.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
