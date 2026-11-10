"""Liver H+Monkey, seeds 42-51 on the identifier-joined ("repaired") object and on the control. Classifier, rate rule and both reference forms were fixed before these runs.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair_monkey rate       # gold-free
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair_monkey compare    # gold-free
    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.repair_monkey golds      # only if rates match
"""

from __future__ import annotations

import itertools
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.tasks.O1_orthologs import rules as RU
from analysis.tasks.O1_orthologs.repair_compare import OUT, obj, repaired_set

sys.path.insert(0, str(RU.REPO / "training"))
from compare_vs_scvi import compute_integration_metrics  # noqa: E402

c = "liv_hmonkey"
SEEDS = list(range(42, 52))
MPS = {42: "iter90-150ep-s42", 43: "iter90-150ep-s43", 44: "iter90-150ep-s44"}
CUT = dict(adv_excess=0.6388 - 0.5098, ent=0.2198, sd=0.0587, tail=0.1545)
ARMS = {"repaired": {s: f"o2-{c}-s{s}" for s in SEEDS}, "control": {s: f"o2-ctrl-{c}-s{s}" for s in SEEDS}, "canonical_mps": MPS}


def classify(run, a, major):
    x = K.read_run(run)
    z = np.load(C.RUNS_DIR / run / "final_state.npz")
    g = compute_integration_metrics(z["z_mean"], a.obs["species"].to_numpy(), a.obs["cell_type"].to_numpy(), k=15, seed=42)["global"]
    d = x["D"] - np.median(x["D"])
    m = dict(adv=float(z["adversarial_accuracy_final"]), adv_excess=float(z["adversarial_accuracy_final"]) - major, ent=g["knn_mixing_entropy"], sd=float(np.std(x["D"])), tail=float((d < -0.05).mean()))
    n = int(m["adv_excess"] <= CUT["adv_excess"]) + int(m["ent"] >= CUT["ent"]) + int(m["sd"] >= CUT["sd"]) + int(m["tail"] >= CUT["tail"])
    return dict(m, n_integrated_markers=n, state="integrated" if n >= 3 else ("poorly mixed" if n <= 1 else "between"))


def rate():
    a0, a1 = C.read_adata(c), ad.read_h5ad(obj(c))
    major = float(a0.obs["species"].value_counts(normalize=True).max())
    rows = []
    for arm, runs in ARMS.items():
        for s, r in runs.items():
            rows.append(dict(arm=arm, seed=s, run=r, **classify(r, a1 if arm == "repaired" else a0, major)))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "monkey_states.csv", index=False)
    pm = T.state.eq("poorly mixed")
    n1, n0 = int(pm[T.arm == "repaired"].sum()), int(pm[T.arm == "control"].sum())
    b1, b0 = int((T.state != "integrated")[T.arm == "repaired"].sum()), int((T.state != "integrated")[T.arm == "control"].sum())
    v = dict(repaired_poor=n1, control_poor=n0, excess=n1 - n0, match=(n1 - n0) <= 2, fisher_one_sided=float(fisher_exact([[n1, 10 - n1], [n0, 10 - n0]], alternative="greater")[1]), with_between_repaired=b1, with_between_control=b0, fisher_with_between=float(fisher_exact([[b1, 10 - b1], [b0, 10 - b0]], alternative="greater")[1]), cuts=CUT)
    (OUT / "monkey_rate.json").write_text(json.dumps(v, indent=1))
    pd.set_option("display.width", 200, "display.max_rows", 40)
    print(T.drop(columns="run").round(3).to_string())
    print(json.dumps(v, indent=1))


def compare():
    T = pd.read_csv(OUT / "monkey_states.csv")
    a0, a1 = C.read_adata(c, backed=True), ad.read_h5ad(obj(c), backed="r")
    v0, v1 = pd.Index(a0.var_names.astype(str)), pd.Index(a1.var_names.astype(str))
    sh = v0.intersection(v1)
    p0, p1 = v0.get_indexer(sh), v1.get_indexer(sh)
    rep = sh.isin(set(repaired_set(c).var_name))
    D, S = {}, {}
    for r in T.itertuples():
        x = K.read_run(r.run)
        p = p1 if r.arm == "repaired" else p0
        D[(r.arm, r.seed)], S[(r.arm, r.seed)] = x["D"][p], x["S_g_ema"][p]
    un = [k for k in D if k[0] != "repaired"]
    st = {(r.arm, r.seed): r.state for r in T.itertuples()}
    rho = lambda a, b: float(spearmanr(a, b)[0])
    rows = []
    for rd, X in (("D", D), ("S", S)):
        for gs, m in (("all", np.ones(len(sh), bool)), ("unrepaired", ~rep), ("repaired", rep)):
            Y = {k: (v[m] - np.median(v[m])) if rd == "D" else v[m] for k, v in X.items()}
            for form in ("pooled", "distinct"):
                for scope in ("all runs", "integrated"):
                    ref = un if scope == "all runs" else [k for k in un if st[k] == "integrated"]
                    new = [k for k in Y if k[0] == "repaired" and (scope == "all runs" or st[k] == "integrated")]
                    other = (lambda k, pool: [q for q in pool if q != k and (form == "pooled" or q[1] != k[1])])
                    A = {k: rho(Y[k], np.mean([Y[q] for q in other(k, ref)], axis=0)) for k in ref}
                    N = {k: rho(Y[k], np.mean([Y[q] for q in other(k, ref)], axis=0)) for k in new}
                    pairs = [rho(Y[a], Y[b]) for a, b in itertools.combinations(ref, 2) if form == "pooled" or a[1] != b[1]]
                    rows.append(dict(readout=rd, genes=gs, form=form, scope=scope, n_genes=int(m.sum()), n_ref=len(ref), n_new=len(new), A_min=min(A.values()), A_med=float(np.median(list(A.values()))), A_max=max(A.values()),
                                     new_min=min(N.values()) if N else np.nan, new_med=float(np.median(list(N.values()))) if N else np.nan, new_max=max(N.values()) if N else np.nan, n_new_below_min=int(sum(v < min(A.values()) for v in N.values())),
                                     pair_min=min(pairs), pair_med=float(np.median(pairs)), pair_max=max(pairs)))
    R = pd.DataFrame(rows)
    R.to_csv(OUT / "monkey_comparison.csv", index=False)
    pd.set_option("display.width", 220, "display.max_columns", 30, "display.max_rows", 100)
    print(R.round(3).to_string())


def golds():
    v = json.loads((OUT / "monkey_rate.json").read_text())
    if not v["match"]:
        raise SystemExit("rates do not match: stop, no gold is read")
    from analysis.tasks.T4_3_scoreboard import run as R
    from analysis.tasks.T4_3_scoreboard import stats as S

    T = pd.read_csv(OUT / "monkey_states.csv")
    T = T[T.state == "integrated"]
    gold = pd.read_csv(RU.REPO / "results/analysis/T4_3/golds/gold_table.csv", index_col=0)
    rep = set(repaired_set(c).var_name)
    rows = []
    for arm in ("control", "repaired"):
        a = ad.read_h5ad(obj(c)) if arm == "repaired" else C.read_adata(c)
        cov = K.cohort_covariates(c, adata=a, prior_raw=a.var["conservation_prior"].to_numpy(dtype=float))
        from analysis.tasks.T4_3_scoreboard import inputs as I

        I.T42_DIR = RU.REPO / "results/analysis/T4_2"
        t = cov.join(gold, on="hsapiens_gene_id")
        t["tau_shared"] = I.tau_shared(a, c)
        runs = T[T.arm == arm].run.tolist()
        M = np.column_stack([K.read_run(r)["D"] for r in runs])
        t["D"] = M.mean(1) - np.median(M.mean(1))
        t["var_name"] = t.index
        blocks = I.esm_blocks(np.asarray(a.varm["gene_features"], dtype=float)[:, :64])[0]
        t["block"] = blocks
        for h, g, cv in (("H1", "G_H1", "C"), ("H3", "G_H3", "C_pat")):
            cols = [g] + R.COVSETS[cv] + ["D"]
            for lab, extra in (("all", np.ones(len(t), bool)), ("no repaired genes", ~t.var_name.isin(rep).to_numpy())):
                m = t[cols].notna().all(1).to_numpy() & extra
                U = t.loc[m]
                Rk = np.column_stack([S.midrank(U[g].to_numpy(float)), S.midrank(U["D"].to_numpy(float))])
                Z = S.design(U, R.COVSETS[cv])
                est = S.rho_p_matrix(Rk, Z)[1]
                B = S.bootstrap(Rk, Z, 10000, np.random.default_rng(RU.SEED), U["block"].to_numpy())[:, 1]
                lo, hi = S.pct_ci(B)
                rows.append(dict(arm=arm, n_seeds=len(runs), hyp_gold=h, genes=lab, n=int(m.sum()), rho_p=float(est), ci_lo=lo, ci_hi=hi))
    D = pd.DataFrame(rows)
    D.to_csv(OUT / "monkey_golds.csv", index=False)
    print(D.round(4).to_string())


if __name__ == "__main__":
    dict(rate=rate, compare=compare, golds=golds)[sys.argv[1]]()
