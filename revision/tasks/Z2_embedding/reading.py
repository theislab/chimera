"""Gene reading per run (Z1 diagnostics) and agreement with the canonical consensus (H1 tables): Spearman of D with consensus D on the consensus count-supported genes, S_g_ema likewise.
Leave-one-out among canonical seeds first, as the reference. No gold is read; columns taken from H1 are alpha per seed, prior, D, S_ema, count support.
python -m revision.tasks.Z2_embedding.reading [cohorts...]  Output: results/revision/Z2/reading.csv, reading_loo.csv"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.tasks.Z2_embedding import rules as RU

sys.path.insert(0, str(RU.REPO / "revision/tasks/Z1_retune"))
import diagnostics as Z1  # noqa: E402


def consensus(c):
    cols = pd.read_parquet(RU.REPO / f"results/revision/H1/genes/{c}.parquet", columns=None).columns.tolist()
    use = [x for x in cols if x in ("var_name", "pi_clamped", "pi_raw", "D", "S_ema", "count_support") or x.startswith("alpha_s")]
    return pd.read_parquet(RU.REPO / f"results/revision/H1/genes/{c}.parquet", columns=use).set_index("var_name")


def main():
    fac = pd.read_csv(RU.REPO / "revision/tasks/Z2_embedding/factorial.tsv", sep="\t")
    rows, loo = [], []
    for c in ("liv_hmouse", "test_hmouse"):
        H = consensus(c)
        sup = H.count_support.astype(bool).to_numpy()
        al = [x for x in H.columns if x.startswith("alpha_s")]
        for x in al:
            others = [o for o in al if o != x]
            d = (H[x] - H.pi_clamped).to_numpy()
            m = (H[others].mean(1) - H.pi_clamped).to_numpy()
            loo.append(dict(cohort=c, seed=x, rho_loo=spearmanr(d[sup], m[sup])[0], rho_loo_all=spearmanr(d, m)[0]))
        runs = [(r, "base", int(r.rsplit("-s", 1)[1])) for r in RU.base_seed_runs(c).values()]
        runs += [(r.run, r.level, r.seed) for r in fac[fac.cohort == c].itertuples()]
        for run, lv, s in runs:
            if not (RU.RUNS / run / "final_state.npz").exists():
                continue
            r = Z1.one_run(run)
            r.update(cohort=c, level=lv, seed=s)
            if r.get("no_nan"):
                fs = np.load(RU.RUNS / run / "final_state.npz", allow_pickle=True)
                D = np.asarray(fs["alpha"], float) - np.asarray(fs["prior"], float)
                S = np.asarray(fs["S_g_ema"], float)
                r["rho_D_consensus"] = spearmanr(D[sup], H.D.to_numpy()[sup])[0]
                r["rho_D_consensus_all"] = spearmanr(D, H.D.to_numpy())[0]
                r["rho_S_consensus"] = spearmanr(S[sup], H.S_ema.to_numpy()[sup])[0]
                r["sd_S"] = float(S[np.abs(S) >= Z1.MASK].std())
            rows.append(r)
            print(run, flush=True)
    pd.DataFrame(rows).to_csv(RU.OUT / "reading.csv", index=False)
    pd.DataFrame(loo).to_csv(RU.OUT / "reading_loo.csv", index=False)


if __name__ == "__main__":
    main()
