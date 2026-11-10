"""T12.3 exploratory (decided after E2 was seen; not part of the section 12 test or success rule).

E2 compares Shay divergent genes (COE < 0.25 and a log2 range of lineage means >= 2 in both species)
with COE-conserved genes (COE above the median), which carry no lineage-range requirement. S_g, and so
D, rewards cell-state pattern the shared decoder explains, which is larger for lineage-variable genes.
This repeats E2 with the conserved set restricted to the same range rule (>= 2 in both species), and
reports the stratified AUROC of the range itself as a check of the imbalance. Same strata and
bootstrap as ``tests.py``; the range-matched set is smaller, so its cells are kept on the absolute
reading (>= 5 conserved genes per cell, the section 4 Family 3 sensitivity).

    PYTHONPATH=<chimera repo> python analysis/tasks/T12_3_tests/explore_e2.py
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.tasks.T12_3_tests import tests as T
from analysis.tasks.T4_3_scoreboard import stats as S

SCORES = ["D", "S_ema", "B1_pi", "range_min"]


def main() -> int:
    rng = np.random.default_rng(T.SEED + 7)
    tab, _ = T.load()
    coe = pd.read_csv(T.COE, sep="\t").drop_duplicates("hsapiens_gene_id", keep=False).set_index("hsapiens_gene_id")
    rmin = np.minimum(coe["human_log2_range_lineage_means"], coe["mouse_log2_range_lineage_means"])
    tab["range_min"] = rmin.reindex(tab["hsapiens_gene_id"]).to_numpy()
    voc = tab.dropna(subset=["D"] + T.C_COLS).reset_index(drop=True)
    fine = pd.Series(S.cells(voc["prior_raw"].to_numpy(), voc["log_cp10k_spmean"].to_numpy()), index=voc["var_name"])
    u = tab.dropna(subset=["D", "COE", "B1_raw", "B1_pi"] + T.C_COLS).reset_index(drop=True)
    med = float(u["COE"].median())
    rows = []
    for name, cons in (("conserved, any range (pre-specified E2)", u["COE"] > med),
                       ("conserved with range >= 2 in both species", (u["COE"] > med) & (u["range_min"] >= 2))):
        w = u[u["divergent"] | cons].reset_index(drop=True)
        inset = w["divergent"].to_numpy()
        lab = fine.loc[w["var_name"]].to_numpy()
        keep = S.kept_strata(inset, lab, ratio=5, absolute=name.startswith("conserved with"))
        V = w.loc[keep, SCORES].to_numpy(dtype=float)
        ik, lk = inset[keep], lab[keep]
        draws = T.joint_auroc_boot(V, ik, lk, 2000, rng)
        for j, s in enumerate(SCORES):
            lo, hi = S.pct_ci(draws[:, j])
            rows.append(dict(comparison=name, score=s, n_divergent=int(ik.sum()), n_conserved=int((~ik).sum()),
                             auroc=S.strat_auroc(V[:, j], ik, lk), ci_lo=lo, ci_hi=hi))
    res = pd.DataFrame(rows)
    out = T.OUT / "explore_e2_range_matched.tsv"
    res.to_csv(out, sep="\t", index=False)
    print(res.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
