"""O1 part A, no gold: where do flagged genes sit on D, S_ema and B1? Per cohort and tier. Reads the H1 per-gene tables and the flags.

    PYTHONPATH=$PWD python -m revision.tasks.O1_orthologs.blind
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.tasks.O1_orthologs import rules as RU

H1 = RU.REPO / "results/revision/H1/genes"
TIERS = ["T1", "T2", "T2L", "MAP", "T3", "TZ"]


def main():
    G = pd.read_parquet(RU.RES / "flags_by_cohort.parquet")
    G["MAP"] = G.M_wrong_row | G.M_false_zero
    rows = []
    for c, x in G.groupby("cohort", sort=False):
        g = pd.read_parquet(H1 / f"{c}.parquet").merge(x[["hsapiens_gene_id"] + TIERS].drop_duplicates("hsapiens_gene_id"), on="hsapiens_gene_id", how="left")
        for t in TIERS:
            f = g[t].fillna(False).astype(bool).to_numpy()
            r = dict(cohort=c, tier=t, n=len(g), n_flag=int(f.sum()), share=float(f.mean()), share_supported=float(f[g.count_support].mean()), share_strict=float(f[g.strict_support].mean()))
            if f.sum() < 3:
                rows.append(r)
                continue
            for s in ("D", "S_ema", "B1"):
                v = g[s].to_numpy(float)
                ok = np.isfinite(v)
                sd = np.nanstd(v, ddof=1)
                r[f"{s}_median_diff_sd"] = float((np.nanmedian(v[f & ok]) - np.nanmedian(v[~f & ok])) / sd)
                lo, hi = np.nanquantile(v, [0.05, 0.95])
                r[f"{s}_share_bottom5"] = float(f[ok & (v <= lo)].mean())
                r[f"{s}_share_top5"] = float(f[ok & (v >= hi)].mean())
            u = g[g.count_support]
            r["rho_DS_supported"] = float(spearmanr(u.D, u.S_ema)[0])
            k = u[~u[t].fillna(False).astype(bool)]
            r["rho_DS_supported_unflagged"] = float(spearmanr(k.D, k.S_ema)[0])
            rows.append(r)
    R = pd.DataFrame(rows)
    R.to_csv(RU.RES / "blind_tails.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 40, "display.max_rows", 200)
    print(R[R.tier.isin(["T2", "T3"])][["cohort", "tier", "n_flag", "share", "share_strict", "D_median_diff_sd", "D_share_bottom5", "D_share_top5", "S_ema_median_diff_sd", "S_ema_share_bottom5", "S_ema_share_top5", "B1_share_bottom5", "rho_DS_supported", "rho_DS_supported_unflagged"]].round(3).to_string())


if __name__ == "__main__":
    main()
