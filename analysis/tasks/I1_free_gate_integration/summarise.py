"""I1 summary: per-seed tidy table, Fig 3 scores on the frozen panel bounds, free vs canonical per cohort and metric, verdict.

    python -m analysis.tasks.I1_free_gate_integration.summarise [--cohorts a,b,...]

Writes results/analysis/I1/{per_seed.csv, compare.csv, verdict.json, missing.csv}.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from analysis.tasks.I1_free_gate_integration import rules as RU
from analysis.tasks.I1_free_gate_integration import score as SC

SE = SC.SE
MASTER = SE.OUT / "master_table_ext.csv"


def frozen_bounds() -> dict:
    """dataset -> (min, max) of the direction-corrected raw metrics across the 10 frozen models."""
    T = pd.read_csv(MASTER, float_precision="round_trip")
    out = {}
    for ds, g in T.groupby("dataset"):
        corr = pd.DataFrame({m: SE.direction_correct(g[f"raw__{m}"], m) for m in SE.METRICS})
        out[ds] = (corr.min(), corr.max(), g.set_index("model"))
    return out


def fig3_scores(c: str, raw: dict, B: dict) -> dict:
    ds = RU.FIG3[c]
    lo, hi, panel = B[ds]
    corr = pd.Series({m: float(SE.direction_correct(pd.Series([raw[m]]), m).iloc[0]) for m in SE.METRICS})
    norm = (corr - lo) / (hi - lo)
    mix, bio = norm[SE.BATCH].mean(), norm[SE.BIO].mean()
    out = dict(mixing=mix, biology=bio, total_score=0.4 * mix + 0.6 * bio, outside_bounds=int(((norm < 0) | (norm > 1)).sum()))
    # sensitivity: this seed in place of the Chimera column, panel re-scaled
    r = panel[[f"raw__{m}" for m in SE.METRICS]].copy()
    r.columns = SE.METRICS
    r.loc["Chimera"] = [raw[m] for m in SE.METRICS]
    s = SE.score_table(r).loc["Chimera"]
    out.update(mixing_replace=float(s["mixing"]), biology_replace=float(s["biology"]), total_score_replace=float(s["total_score"]),
               wins_vs_pca=int(s["wins_vs_pca"]))
    return out


def z2_summary(d: pd.DataFrame) -> dict:
    d = d[~d.task.str.endswith(("__odd", "__even"))]
    out = {}
    au = d[d.metric == "auroc_lr"]
    if len(au):
        per_task = au.groupby("task").value.mean()
        out["within_auroc"] = float(per_task.mean())
        for t, v in per_task.items():
            out[f"auroc|{t}"] = float(v)
    ak = d[d.metric == "auroc_knn"]
    if len(ak):
        out["within_auroc_knn"] = float(ak.groupby("task").value.mean().mean())
    rs = d[d.metric == "rho_self"]
    if len(rs):
        per_task = rs.groupby("task").value.mean()
        out["rho_self"] = float(per_task.mean())
        for t, v in per_task.items():
            out[f"rho_self|{t}"] = float(v)
    rc = d[d.metric == "rho_cross"]
    if len(rc):
        out["rho_cross"] = float(rc.groupby("task").value.mean().mean())
    cl = d[d.metric == "clisi"]
    if len(cl):
        out["clisi"] = float(cl.groupby("task").value.mean().mean())
    lf = d[(d.task == "labelfree") & (d.kind == "plain")]
    for m in ("within_var_r2", "lib_r2"):
        q = lf[lf.metric == m]
        if len(q):
            out[m] = float(np.average(q.value, weights=q.n))
    q = lf[lf.metric == "z_within_share"]
    if len(q):
        out["z_within_share"] = float(q.value.iloc[0])
    return out


def enc_summary(c: str, d: pd.DataFrame) -> dict:
    if c == "liv_hmouse":
        g = lambda st, m, dep, mk: float(d[(d.set == st) & (d.metric == m) & (d.depth == dep) & (d.marker == mk)].value.mean())
        return dict(tms_auroc=g("tms", "tms_auroc_lr", "asis", "author"), tms_auroc_depth=g("tms", "tms_auroc_lr", "depth", "author"),
                    tms_rho_self=g("tms", "rho_self", "asis", "all"), ts_rho_self=g("ts", "rho_self", "asis", "all"),
                    ts_to_tms_rho_cross=g("ts_to_tms", "rho_cross", "asis", "all"))
    q = d[(d.set == "macaque") & (d.metric == "stage_auroc_lr") & (d.depth == "asis")]
    r = d[(d.set == "macaque_to_human_in_model") & (d.metric == "rho_cross") & (d.depth == "asis")]
    return dict(macaque_stage_auroc=float(q.value.mean()), macaque_to_human_rho=float(r.value.mean()))


def per_seed(cohorts=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    L = RU.lines()
    if cohorts:
        L = L[L.cohort.isin(cohorts)]
    B = frozen_bounds()
    rows, missing = [], []
    for r in L.itertuples():
        base = dict(cohort=r.cohort, gate=r.gate, seed=r.seed, run=r.run, line=r.line, source=r.source, included=r.included, included_mix=r.included_mix)
        f = SC.path(r.cohort, r.run, "h1")
        if not f.exists():
            missing.append(base | dict(reason="not scored" if (RU.RUNS / r.run / "final_state.npz").exists() else "no final_state"))
            continue
        vals = {}
        h = json.loads(f.read_text())
        vals.update({f"h1|{k}": v for k, v in h.items() if k in RU.FIG3_METRICS or k == "adv_acc"})
        f3 = SC.path(r.cohort, r.run, "fig3")
        if r.cohort in RU.FIG3 and f3.exists():
            raw = json.loads(f3.read_text())
            vals.update({f"fig3|{k}": raw[k] for k in RU.FIG3_METRICS})
            vals.update({f"fig3|{k}": v for k, v in fig3_scores(r.cohort, raw, B).items()})
        sp = SC.path(r.cohort, r.run, "spec")
        if sp.exists():
            vals.update({f"spec|{k}": v for k, v in json.loads(sp.read_text()).items()})
        z2 = SC.path(r.cohort, r.run, "z2")
        if z2.exists():
            vals.update({f"z2|{k}": v for k, v in z2_summary(pd.read_parquet(z2)).items()})
        en = SC.path(r.cohort, r.run, "enc")
        if en.exists():
            vals.update({f"enc|{k}": v for k, v in enc_summary(r.cohort, pd.read_parquet(en)).items()})
        for k, v in vals.items():
            rows.append(base | dict(metric=k, value=float(v)))
    return pd.DataFrame(rows), pd.DataFrame(missing)


def higher_better(metric: str) -> bool:
    m = metric.split("|")[-1]
    return m not in RU.LOWER_BETTER and not m.startswith("outside")


def kept_json() -> dict:
    """Z4's kept.json: cohort -> set of kept free runs (gate health and Z4's mixing criterion)."""
    f = RU.RES / "Z4" / "kept.json"
    return {c: set(v.values()) for c, v in json.loads(f.read_text()).items()} if f.exists() else {}


def comparisons(P: pd.DataFrame) -> pd.DataFrame:
    out = []
    K = kept_json()
    P = P[P.included]
    for (c, line, m), g in P[P.line == "main"].groupby(["cohort", "line", "metric"]):
        # variant: free seeds restricted to Z4's kept.json (coordinator), canonical side unchanged
        if m.endswith(("outside_bounds", "wins_vs_pca")) or c not in K:
            continue
        fr = g[(g.gate == "free") & g.run.isin(K[c])].set_index("seed").value
        ca = g[g.gate == "canonical"].set_index("seed").value
        if len(fr) and len(ca) and len(fr) < (g.gate == "free").sum():
            same = set(fr.index) == set(ca.index)
            out.append(dict(cohort=c, line=line, metric=m, analysis="free_kept_json", seeds_free=",".join(map(str, fr.index)),
                            seeds_can=",".join(map(str, ca.index))) | RU.compare(fr.to_numpy(), ca.to_numpy(), same, higher_better(m)))
    for (c, line, m), g in P.groupby(["cohort", "line", "metric"]):
        if m.endswith(("outside_bounds", "wins_vs_pca")):
            continue
        fr = g[g.gate == "free"].set_index("seed").value
        ca = g[g.gate == "canonical"].set_index("seed").value
        if not len(fr) or not len(ca):
            continue
        same = set(fr.index) == set(ca.index)
        analyses = [("primary", fr, ca, same)]
        shared = sorted(set(fr.index) & set(ca.index))
        if not same and len(shared) >= 2:
            analyses.append(("paired_shared_seeds", fr.loc[shared], ca.loc[shared], True))
        if line == "main" and "included_mix" in g:
            gm = g[g.included_mix]
            fm = gm[gm.gate == "free"].set_index("seed").value
            cm_all = g[g.gate == "canonical"]
            ent = P[(P.cohort == c) & (P.line == line) & (P.gate == "canonical") & (P.metric == "h1|knn_mixing_entropy")].set_index("seed").value
            keep = ent[ent >= 0.5 * ent.median()].index
            cm = cm_all.set_index("seed").value.loc[lambda s: s.index.isin(keep)]
            if len(fm) and len(cm) and (len(fm) < len(fr) or len(cm) < len(ca)):
                analyses.append(("mixing_rule_both_sides", fm, cm, set(fm.index) == set(cm.index)))
        for name, f, cc, paired in analyses:
            if paired:
                idx = sorted(set(f.index) & set(cc.index))
                f, cc = f.loc[idx], cc.loc[idx]
            res = RU.compare(f.to_numpy(), cc.to_numpy(), paired, higher_better(m))
            out.append(dict(cohort=c, line=line, metric=m, analysis=name, seeds_free=",".join(map(str, f.index)),
                            seeds_can=",".join(map(str, cc.index))) | res)
    return pd.DataFrame(out)


def primary_calls(C: pd.DataFrame) -> pd.DataFrame:
    q = C[(C.analysis == "primary") & (C.line == "main")]
    keep = [f"fig3|{m}" for m in RU.PRIMARY_FIG3] + [f"h1|{m}" for m in RU.PRIMARY_H1] + [f"z2|{m}" for m in RU.PRIMARY_Z2]
    return q[q.metric.isin(keep)]


def main():
    cohorts = None
    if "--cohorts" in sys.argv:
        cohorts = sys.argv[sys.argv.index("--cohorts") + 1].split(",")
    P, M = per_seed(cohorts)
    RU.OUT.mkdir(parents=True, exist_ok=True)
    P.to_csv(RU.OUT / "per_seed.csv", index=False)
    M.to_csv(RU.OUT / "missing.csv", index=False)
    C = comparisons(P)
    C.to_csv(RU.OUT / "compare.csv", index=False)
    Q = primary_calls(C)
    v = {}
    z3 = Q[Q.cohort.isin(RU.Z3_COHORTS)]
    v["z3"] = RU.verdict(z3, sorted(z3.metric.unique()))
    v["z3"]["cohorts"] = sorted(z3.cohort.unique())
    v["all"] = RU.verdict(Q, sorted(Q.metric.unique()))
    v["all"]["cohorts"] = sorted(Q.cohort.unique())
    # Fig 3 metrics and H1 metrics separately (Fig 3 rows only have fig3)
    for fam in ("fig3", "h1", "z2"):
        q = Q[Q.metric.str.startswith(fam + "|")]
        if len(q):
            v[f"all_{fam}"] = RU.verdict(q, sorted(q.metric.unique())) | dict(cohorts=sorted(q.cohort.unique()))
    (RU.OUT / "verdict.json").write_text(json.dumps(v, indent=1, default=str))
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
    show = C[(C.analysis == "primary")][["cohort", "line", "metric", "n_free", "n_can", "mean_free", "mean_can", "delta", "ci_lo", "ci_hi", "call"]]
    print(show.round(3).to_string(index=False))
    print(json.dumps(v, indent=1, default=str))
    if len(M):
        print("missing:\n" + M.to_string(index=False))


if __name__ == "__main__":
    main()
