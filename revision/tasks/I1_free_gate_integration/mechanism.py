"""I1 exploratory, labelled: does the gate decide how much of the within-hepatocyte programme the shared head (and so z) must carry?

Per seed of both gates (liver H+Mouse, liver 4-sp lines): mean alpha of the Z2 zonation markers present in the object, of the top 200 and
2000 genes by within-hepatocyte variance (log CP10K, all species pooled), and of all genes; beside the seed's zone AUROC from I1 units.
    python -m revision.tasks.I1_free_gate_integration.mechanism
Writes results/revision/I1/mechanism_alpha.csv.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.tasks.I1_free_gate_integration import rules as RU
from revision.tasks.Z2_embedding import common as K, rules as Z2R


def gene_sets(c: str):
    a = K.read(c)
    Xn = K.lognorm(a.X)
    hep = a.obs["cell_type"].astype(str).to_numpy() == "Hepatocytes"
    M = Xn[hep]
    mu = np.asarray(M.mean(0)).ravel()
    var = np.asarray(M.multiply(M).mean(0)).ravel() - mu ** 2
    order = np.argsort(-var)
    sym = K.symbols(c, a.var_names).symbol.astype(str).to_numpy()
    zone = np.isin(sym, Z2R.HEP_ZONE["up"] + Z2R.HEP_ZONE["dn"])
    det = np.asarray((M > 0).mean(0)).ravel() >= Z2R.MIN_DET
    return dict(zone=zone & det, hv200=np.isin(np.arange(len(sym)), order[:200]), hv2000=np.isin(np.arange(len(sym)), order[:2000]),
                all=np.ones(len(sym), bool)), sym


def main():
    L = RU.lines()
    P = pd.read_csv(RU.OUT / "per_seed.csv")
    rows = []
    for c in ("liv_hmouse", "liv_4sp"):
        sets, sym = gene_sets(c)
        q = L[(L.cohort == c) & (L.line == "main")]
        for r in q.itertuples():
            with np.load(RU.RUNS / r.run / "final_state.npz", allow_pickle=True) as f:
                al, pi, S = np.asarray(f["alpha"]), np.asarray(f["prior"]), np.asarray(f["S_g_ema"])
            d = dict(cohort=c, gate=r.gate, seed=r.seed, run=r.run)
            for k, m in sets.items():
                d[f"alpha_{k}"] = float(al[m].mean())
                d[f"dalpha_{k}"] = float((al - pi)[m].mean())
                d[f"S_{k}"] = float(S[m].mean())
                d[f"n_{k}"] = int(m.sum())
            v = P[(P.cohort == c) & (P.run == r.run) & (P.line == "main")].set_index("metric").value
            for m in ("z2|within_auroc", "z2|rho_cross", "enc|tms_auroc", "h1|knn_mixing_entropy", "spec|pc1_share", "z2|lib_r2"):
                d[m] = float(v.get(m, np.nan))
            rows.append(d)
    T = pd.DataFrame(rows)
    T.to_csv(RU.OUT / "mechanism_alpha.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    cols = ["cohort", "gate", "seed", "alpha_zone", "dalpha_zone", "alpha_hv200", "alpha_hv2000", "alpha_all", "S_zone", "z2|within_auroc", "enc|tms_auroc", "h1|knn_mixing_entropy"]
    print(T[cols].round(3).to_string(index=False))
    for c, g in T.groupby("cohort"):
        for x in ("alpha_zone", "dalpha_zone", "alpha_hv200", "alpha_hv2000"):
            print(c, x, "Spearman with zone AUROC over both gates' seeds:", round(spearmanr(g[x], g["z2|within_auroc"])[0], 2))


if __name__ == "__main__":
    main()
