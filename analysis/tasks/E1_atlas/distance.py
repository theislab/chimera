"""E1 step 4: how the programme reading depends on evolutionary distance (definitions in rules.py).

Per pair cohort and gate: Spearman of the cohort's set z with the consensus of the thread tissues other than its own tissue (leave
tissue out, so human-mouse cohorts are not compared with themselves); pole scores (mean effect of AGREE-above and AGREE-below sets);
number of sets at q < 0.05. Per set: within-ladder Spearman of effect with log10 divergence (eye, liver, cortex DLPFC; testis has two
points and enters the sign count only), sign consistency across ladders.

Writes results/analysis/E1/{distance_cohorts.csv, distance_sets.csv}.

    PYTHONPATH=$PWD python -m analysis.tasks.E1_atlas.distance
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.E1_atlas import rules as RU

LADDERS = {"eye": ["vz_hmmul", "vz_hmfas", "vz_hmouse", "vz_hpig"], "liver": ["liv_hmonkey", "liv_hmouse", "liv_hzeb"],
           "cortex": ["brain_dlpfc_hchimp", "brain_dlpfc_hmac", "brain_dlpfc_hmac_own", "brain_dlpfc_hmarm"], "testis": ["test_hmac", "test_hmouse"]}


def main() -> int:
    st = pd.read_parquet(RU.OUT / "set_stats.parquet")
    rec = pd.read_parquet(RU.OUT / "recurrence.parquet")
    A = pd.read_csv(RU.OUT / "atlas.csv", index_col=0)
    above = A.index[A.agree & (A.can_direction == "above")]
    below = A.index[A.agree & (A.can_direction == "below")]
    rows = []
    cohorts = {**{c: v for c, v in RU.PAIRS.items()}, **{c: (sp, None) for c, sp in RU.MULTI.items()}}
    from analysis.tasks.H1_harmonise import rules as H1R
    meta = H1R.cohorts()
    for c, (partner, _) in cohorts.items():
        tissue = meta[c]["tissue"]
        for gate, r in (("canonical", "D_can"), ("per-gene", "D_free")):
            w = st[(st.cohort == c) & (st.readout == r)].set_index("key")
            cons_r = rec[rec.readout == r].set_index("key")
            zc = [f"z_{t}" for t in RU.TISSUES if t != tissue]
            cons = cons_r[zc].sum(1, min_count=3) / np.sqrt(cons_r[zc].notna().sum(1).clip(lower=1))
            j = pd.concat([w.z, cons.rename("cons")], axis=1).dropna()
            rows.append(dict(cohort=c, label=meta[c]["label"], tissue=tissue, gate=gate, partner=partner, divergence_my=RU.DIVERGENCE[partner],
                             pair=c in RU.PAIRS, n_sets=int(w.z.notna().sum()), n_sig=int((w.q < 0.05).sum()),
                             rho_consensus=spearmanr(j.z, j.cons)[0], above_pole=float(w.effect.reindex(above).mean()), below_pole=float(w.effect.reindex(below).mean()),
                             above_pole_z=float(w.z.reindex(above).mean()), below_pole_z=float(w.z.reindex(below).mean()),
                             frac_above_sets_up=float((w.z.reindex(above) >= RU.Z_CALL).mean()), frac_below_sets_down=float((w.z.reindex(below) <= -RU.Z_CALL).mean())))
    dc = pd.DataFrame(rows)
    dc.to_csv(RU.OUT / "distance_cohorts.csv", index=False)
    # per set, within-ladder trends
    srows = []
    for gate, r in (("canonical", "D_can"), ("per-gene", "D_free")):
        w = st[st.readout == r]
        E = w.pivot(index="key", columns="cohort", values="effect")
        for k in E.index:
            d = dict(key=k, gate=gate)
            signs = []
            for lad, cs in LADDERS.items():
                x = np.log10([RU.DIVERGENCE[RU.PAIRS[c][0]] for c in cs])
                y = E.loc[k, cs].to_numpy(float)
                ok = np.isfinite(y)
                if ok.sum() >= 2:
                    slope = np.polyfit(x[ok], y[ok], 1)[0] if len(set(x[ok])) > 1 else np.nan
                    d[f"slope_{lad}"] = slope
                    d[f"rho_{lad}"] = spearmanr(x[ok], y[ok])[0] if ok.sum() >= 3 else np.nan
                    if np.isfinite(slope):
                        signs.append(np.sign(slope))
            d["n_ladders"] = len(signs)
            d["n_rise"] = int(sum(s > 0 for s in signs))
            d["n_fall"] = int(sum(s < 0 for s in signs))
            srows.append(d)
    ds = pd.DataFrame(srows)
    ds.to_csv(RU.OUT / "distance_sets.csv", index=False)
    pd.set_option("display.width", 250)
    print(dc.sort_values(["gate", "tissue", "divergence_my"]).round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
