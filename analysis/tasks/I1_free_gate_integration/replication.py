"""Liver H+Monkey replication of the liver H+Mouse zonation result, fixed before any H+Monkey free-gate value was read.

Pairs: per-gene gate fits z4-liv_hmonkey-m-s42..s46 of Z4_free_gate_all (its gate health, from its health table) against the canonical-recipe CPU
runs o2-ctrl-liv_hmonkey-s42..s46 (control arm of analysis/tasks/O1_orthologs, repair_monkey.py; health by the fixed canonical rule, Z3_free_gate
``health.gate``), paired by seed number.
Test: hepatocyte zone AUROC in the model (Z2_embedding ``hep_zone`` task: Z2_embedding markers, standardized logistic regression, 5-fold, mean over
species), ``score.py`` of this task unchanged. Primary: paired difference free - canonical, one-sided paired t; replicated if the one-sided
95% lower bound is above 0. Beside: seed-range call, rho_self, rho_cross, kNN-AUROC, and the integrated pairs (both seeds with
mixing entropy at least half the median of the canonical kept seeds of H1_harmonise, the criterion of Z4_free_gate_all).

    python -m analysis.tasks.I1_free_gate_integration.replication health    health of the control-arm runs (no zone value, no free value)
    python -m analysis.tasks.I1_free_gate_integration.replication test      the test, after the units are scored
Writes results/analysis/I1/replication_health.csv, replication.csv, replication.json.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from analysis.tasks.I1_free_gate_integration import rules as RU

COHORT = "liv_hmonkey"
SEEDS = [42, 43, 44, 45, 46]
CAN = {s: f"o2-ctrl-liv_hmonkey-s{s}" for s in SEEDS}
FREE = {s: f"z4-liv_hmonkey-m-s{s}" for s in SEEDS}
PRIMARY = "z2|auroc|hep_zone"
BESIDE = ["z2|rho_self|hep_zone", "z2|rho_cross", "z2|within_auroc_knn", "h1|knn_mixing_entropy", "h1|label_transfer_acc"]
HEALTH = RU.OUT / "replication_health.csv"


def health() -> pd.DataFrame:
    from analysis.common import cohorts as C
    from analysis.tasks.Z3_free_gate import health as Z3H
    a = C.read_adata(COHORT, backed=True)
    praw = C.raw_prior(COHORT, C.CANONICAL[COHORT], a.var["conservation_prior"].to_numpy(dtype=float))
    a.file.close()
    rows = []
    for s, r in CAN.items():
        g = Z3H.gate(r, praw)
        rows.append(dict(seed=s, run=r, **{k: g.get(k) for k in ("finished", "no_nan", "sat_high", "sat_low", "n_active_z", "tail_up_raw",
                                                                   "tail_dn_raw", "healthy")}))
    H = pd.DataFrame(rows)
    RU.OUT.mkdir(parents=True, exist_ok=True)
    H.to_csv(HEALTH, index=False)
    print(H.to_string(index=False))
    return H


def lines() -> list[dict]:
    """Rows for rules.lines(): line 'replication_hmonkey'. Canonical inclusion from replication_health.csv; free from Z4_free_gate_all gate health."""
    out = []
    if HEALTH.exists():
        H = pd.read_csv(HEALTH).set_index("seed")
        for s, r in CAN.items():
            ok = bool(H.loc[s, "healthy"]) if s in H.index else False
            out.append(dict(cohort=COHORT, gate="canonical", seed=s, run=r, line="replication_hmonkey", source="O2", included=ok, included_mix=ok))
    zh = RU.RES / "Z4" / "health.csv"
    if zh.exists():
        z = pd.read_csv(zh)
        z = z[(z.cohort == COHORT) & z.finished.astype(bool) & z.seed.isin(SEEDS)]
        for r in z.itertuples():
            ok = bool(r.healthy_gate) and bool(r.no_nan)
            out.append(dict(cohort=COHORT, gate="free", seed=int(r.seed), run=r.run, line="replication_hmonkey", source="Z4", included=ok, included_mix=ok))
    return out


def one_sided(d: np.ndarray) -> dict:
    from scipy.stats import t
    n = len(d)
    if n < 2:
        return dict(n_pairs=n, mean=float(d.mean()) if n else np.nan, lower95=np.nan, p_one_sided=np.nan)
    se = d.std(ddof=1) / np.sqrt(n)
    tt = d.mean() / se if se > 0 else np.inf * np.sign(d.mean())
    return dict(n_pairs=n, mean=float(d.mean()), lower95=float(d.mean() - t.ppf(0.95, n - 1) * se), p_one_sided=float(t.sf(tt, n - 1)))


def test(variant: str = "spec") -> dict:
    """variant 'spec': gate health of Z4_free_gate_all; 'kept_json': per-gene seeds also in Z4_free_gate_all's kept.json (adds Z4_free_gate_all's mixing criterion)."""
    P = pd.read_csv(RU.OUT / "per_seed.csv")
    P = P[(P.line == "replication_hmonkey") & P.included]
    if variant == "kept_json":
        kept = set(json.loads((RU.RES / "Z4" / "kept.json").read_text()).get(COHORT, {}).values())
        P = P[(P.gate == "canonical") | P.run.isin(kept)]
    W = P.pivot_table(index="seed", columns=["gate", "metric"], values="value")
    pairs = [s for s in SEEDS if ("free", PRIMARY) in W and ("canonical", PRIMARY) in W and s in W.index
             and np.isfinite(W.loc[s, ("free", PRIMARY)]) and np.isfinite(W.loc[s, ("canonical", PRIMARY)])]
    res, rows = {}, []
    ent = pd.read_csv(RU.RES / "H1" / "cache" / "integration.csv")
    ent_cut = 0.5 * ent[ent.cohort == COHORT].knn_mixing_entropy.median()
    integ = [s for s in pairs if W.loc[s, ("free", "h1|knn_mixing_entropy")] >= ent_cut and W.loc[s, ("canonical", "h1|knn_mixing_entropy")] >= ent_cut]
    for m in [PRIMARY] + BESIDE:
        if ("free", m) not in W or ("canonical", m) not in W:
            continue
        for name, ss in (("all_pairs", pairs), ("integrated_pairs", integ)):
            if not ss:
                continue
            f, c = W.loc[ss, ("free", m)].to_numpy(float), W.loc[ss, ("canonical", m)].to_numpy(float)
            r = dict(metric=m, pairs=name, seeds=",".join(map(str, ss))) | one_sided(f - c) | RU.compare(f, c, True)
            rows.append(r)
    T = pd.DataFrame(rows)
    sfx = "" if variant == "spec" else f"_{variant}"
    T.to_csv(RU.OUT / f"replication{sfx}.csv", index=False)
    p = T[(T.metric == PRIMARY) & (T.pairs == "all_pairs")]
    res = dict(n_pairs=len(pairs), entropy_cut=ent_cut, integrated_pairs=integ,
               replicated=bool(len(p) and p.lower95.iloc[0] > 0) if len(pairs) >= 3 else None,
               note="" if len(pairs) >= 3 else "fewer than 3 healthy pairs: not tested")
    if len(p):
        res |= {k: float(p[k].iloc[0]) for k in ("mean", "lower95", "p_one_sided", "mean_free", "mean_can")} | dict(call=p.call.iloc[0])
    res["variant"] = variant
    (RU.OUT / f"replication{sfx}.json").write_text(json.dumps(res, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(T.round(3).to_string(index=False))
    print(json.dumps(res, indent=1, default=str))
    return res


if __name__ == "__main__":
    if sys.argv[1] == "health":
        health()
    else:
        test(sys.argv[2] if len(sys.argv) > 2 else "spec")
