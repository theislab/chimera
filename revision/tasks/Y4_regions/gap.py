"""Y4 (EXPLORATORY): gap map between sequence similarity and deployment. Descriptive, rules.py.

Per region and split, residualised on expression and detection only (not pi): mean percentile of id_true minus mean
percentile of S, of D_raw and of the pattern gold (the gold column only when ``gold`` is passed); share of genes with
id_true in the top tertile and S residual in the bottom tertile against the expected 1/9; ratio of the SD of the S residual
in the region to the split's SD. `gap odd`, then `gap even` after the carried list is committed.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.Y4_regions import rules as RU
from revision.tasks.Y4_regions import screen as SC

PATTERN = {"liver": "G_H3", "testis": "G_H4"}


def pct_resid(T, col, Z):
    d = T[col].notna().to_numpy()
    e = np.full(len(T), np.nan)
    e[d] = ST.resid(ST.midrank(T.loc[d, col].to_numpy(float)), Z[d])
    s = pd.Series(e)
    return (s.rank() / s.notna().sum()).to_numpy()


def run(split: str) -> pd.DataFrame:
    rows = []
    for c in RU.COHORTS:
        t = SC.table(c)
        cv = ["log_cp10k_spmean", "detection_spmean"]
        ok = t[cv + ["id_true", "S"]].notna().all(1).to_numpy() & t["split"].eq(split).to_numpy() & ~t["masked"].to_numpy()
        T = t.loc[ok]
        Z = ST.design(T, cv)
        P = pct_resid(T, "id_true", Z)
        Ps, Pd = pct_resid(T, "S", Z), pct_resid(T, "D_raw", Z)
        g = PATTERN[SC.tissue(c)]
        Pg = pct_resid(T, g, Z)
        sd0 = np.nanstd(Ps)
        for reg in SC.regs(c):
            m = reg["mask"][ok]
            if m.sum() < RU.MIN_GENES:
                continue
            rows.append(dict(cohort=c, split=split, region=reg["name"], axis=reg["axis"], n=int(m.sum()),
                             id_pct=np.nanmean(P[m]), S_pct=np.nanmean(Ps[m]), D_pct=np.nanmean(Pd[m]), gold_pct=np.nanmean(Pg[m]),
                             gap_S=np.nanmean(P[m] - Ps[m]), gap_D=np.nanmean(P[m] - Pd[m]), gap_gold=np.nanmean(P[m] - Pg[m]),
                             share_hiId_loS=float(np.mean((P[m] > 2 / 3) & (Ps[m] < 1 / 3))),
                             share_hiId_hiS=float(np.mean((P[m] > 2 / 3) & (Ps[m] > 2 / 3))),
                             sd_ratio_S=float(np.nanstd(Ps[m]) / sd0)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    s = sys.argv[1]
    d = run(s)
    d.to_csv(SC.OUT / f"gap_map_{s}.csv", index=False)
    print(d.groupby("cohort").size())
