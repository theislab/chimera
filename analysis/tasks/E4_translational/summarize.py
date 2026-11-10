"""Step 4: call shares of targets and disease genes against strata-preserving permutation (pooled over human-mouse tissues), and the
detection gap of robust below and above calls (targets beside all genes). Post hoc summaries of the call shares of rules.py; labeled so.

Writes results/analysis/E4/{callshare_perm.csv, detection_gap.csv}.

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.E4_translational.summarize
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

from analysis.tasks.E4_translational import rules as RU

N_PERM = 2_000


def main() -> int:
    T = pd.read_parquet(RU.OUT / "tables" / "cohorts.parquet")
    sets = pd.read_parquet(RU.OUT / "annot" / "sets.parquet")
    hm = {**RU.HM_THREAD, **{"skeletal muscle": "muscle_ageing"}}
    groups = {"approved targets": lambda t: set(sets.hsapiens_gene_id[sets.set == "target:approved"]),
              "clinical-only targets": lambda t: set(sets.hsapiens_gene_id[sets.set == "target:clinical_only"]),
              "indicated targets, own tissue": lambda t: set(sets.hsapiens_gene_id[sets.set == f"target:ind_{t}"]),
              "disease genes, own tissue": lambda t: set(sets.hsapiens_gene_id[sets.set == f"own:{t}"])}
    rng = np.random.default_rng(RU.SEED)
    rows = []
    for gname, fn in groups.items():
        obs_up = obs_dn = 0
        null_up = np.zeros(N_PERM)
        null_dn = np.zeros(N_PERM)
        n = 0
        for t, c in hm.items():
            u = T[T.cohort == c].reset_index(drop=True)
            m = u.hsapiens_gene_id.isin(fn(t)).to_numpy()
            n += int(m.sum())
            call = u.call.to_numpy()
            obs_up += int((call[m] == 1).sum())
            obs_dn += int((call[m] == -1).sum())
            st = u.stratum.to_numpy()
            # per stratum: number of members k, calls in stratum; null count = hypergeometric draw per stratum
            for s in np.unique(st[m]):
                idx = st == s
                k = int(m[idx].sum())
                cs = call[idx]
                N_, up_, dn_ = len(cs), int((cs == 1).sum()), int((cs == -1).sum())
                null_up += rng.hypergeometric(up_, N_ - up_, k, size=N_PERM)
                null_dn += rng.hypergeometric(dn_, N_ - dn_, k, size=N_PERM)
        rows.append(dict(group=gname, n=n, above=obs_up, exp_above=null_up.mean(), fold_above=obs_up / null_up.mean(),
                         p_above=(1 + (np.abs(null_up - null_up.mean()) >= abs(obs_up - null_up.mean())).sum()) / (N_PERM + 1),
                         below=obs_dn, exp_below=null_dn.mean(), fold_below=obs_dn / null_dn.mean(),
                         p_below=(1 + (np.abs(null_dn - null_dn.mean()) >= abs(obs_dn - null_dn.mean())).sum()) / (N_PERM + 1)))
    R = pd.DataFrame(rows)
    R.to_csv(RU.OUT / "callshare_perm.csv", index=False)
    print(R.round(3).to_string(index=False))
    # detection gap of robust calls (log2 human / partner max detection), targets vs all genes, human-mouse tissues
    appr = set(sets.hsapiens_gene_id[sets.set == "target:approved"])
    X = T[T.cohort.isin(hm.values()) & (T.call != 0)].copy()
    X["gap"] = np.log2((X.det_human + 0.01) / (X.det_partner_max + 0.01))
    X["target"] = X.hsapiens_gene_id.isin(appr)
    G = X.groupby(["target", "call"]).gap.agg(["size", "median", lambda s: (s < -0.5).mean(), lambda s: (s > 0.5).mean()])
    G.columns = ["n", "median_log2_human_vs_partner", "share_partner_broader", "share_human_broader"]
    G = G.reset_index()
    p = mannwhitneyu(X.gap[X.target & (X.call == -1)], X.gap[~X.target & (X.call == -1)]).pvalue
    G["p_target_vs_rest_same_call"] = [np.nan if not r.target else
                                       mannwhitneyu(X.gap[X.target & (X.call == r.call)], X.gap[~X.target & (X.call == r.call)]).pvalue for r in G.itertuples()]
    G.to_csv(RU.OUT / "detection_gap.csv", index=False)
    print(G.round(3).to_string(index=False), p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
