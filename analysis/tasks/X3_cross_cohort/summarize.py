"""EXPLORATORY: compact cross-cohort summary tables from the ``analyses.py`` CSVs.

* ``summary_classes.csv``: per class and score, the matched excess (C4) in each cohort, the
  number of cohorts with a positive / negative excess at p < 0.01, and the seed range.
* ``summary_core_sets.csv``: core-group enrichments (classes and programs) at q < 0.05 in at
  least two of the three tissue pairs, same direction, for Chimera D and for the baseline.
* ``summary_depth_programs.csv``: per liver program, matched excess per species pair for D, S
  and B1b, and the monkey-to-mouse change.
* ``summary_coupling.csv``: per cohort, feature predictability of D (this task's ESM-only CV R²
  of residual D; T2_4_covariates's 72-feature CV R² of s42 delta-alpha) beside the D-S residual coupling.

Writes results/analysis/X3/summary_*.csv.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.common import gene_ids

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "X3"
T24 = gene_ids.REPO_ROOT / "results" / "analysis" / "T2_4"
ORDER = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]


def classes() -> pd.DataFrame:
    d = pd.read_csv(OUT / "class_excess.csv")
    d = d[d.matching == "C4"]
    s = pd.read_csv(OUT / "class_excess_seeds.csv")
    e = pd.read_csv(OUT / "esm_class_loadings.csv")
    e = e.rename(columns={"score": "score"})
    e["matching"] = "C4"
    d = pd.concat([d, e[e.score.isin(["non_esm_rD", "esm_pred_rD", "non_esm_rB1b"])]], ignore_index=True)
    rows = []
    for (cls, score), x in d.groupby(["cls", "score"]):
        x = x.set_index("cohort").reindex(ORDER)
        r = dict(cls=cls, score=score)
        for c in ORDER:
            r[f"excess_{c}"] = x.loc[c, "excess_d"]
            r[f"p_{c}"] = x.loc[c, "p"]
        r["n_pos_sig"] = int(((x.excess_d > 0) & (x.p < 0.01)).sum())
        r["n_neg_sig"] = int(((x.excess_d < 0) & (x.p < 0.01)).sum())
        r["n_pos"] = int((x.excess_d > 0).sum())
        r["n_neg"] = int((x.excess_d < 0).sum())
        if score == "D_raw":
            ss = s[s.cls == cls]
            for c in ORDER:
                v = ss[ss.cohort == c].excess_d
                r[f"seed_range_{c}"] = f"{v.min():.2f}..{v.max():.2f}" if len(v) else ""
        rows.append(r)
    return pd.DataFrame(rows)


def core_sets() -> pd.DataFrame:
    e = pd.read_csv(OUT / "core_enrichment.csv")
    e = e[e.kind == "enrichment"].copy()
    e["direction"] = np.where(e.oe > 1, "enriched", "depleted")
    sig = e[e.q < 0.05]
    g = (sig.groupby(["score", "group", "set", "direction"])
         .agg(n_pairs=("liver", "nunique"), pairs=("liver", lambda v: " ".join(sorted(v))),
              oe_min=("oe", "min"), oe_max=("oe", "max"), q_max=("q", "max"))
         .reset_index())
    return g[g.n_pairs >= 2].sort_values(["score", "group", "direction", "n_pairs"], ascending=[True, True, True, False])


def depth_programs() -> pd.DataFrame:
    d = pd.read_csv(OUT / "depth_programs.csv")
    d = d[d.tissue == "liver"]
    w = d.pivot_table(index="set", columns=["score", "cohort"], values="excess_d")
    w.columns = [f"{s}__{c}" for s, c in w.columns]
    q = d.pivot_table(index="set", columns=["score", "cohort"], values="q")
    q.columns = [f"q_{s}__{c}" for s, c in q.columns]
    n = d.groupby("set").n.max().rename("n_genes")
    ident = d[d.score == "D_raw"].pivot_table(index="set", columns="cohort", values="mean_identity")
    ident.columns = [f"identity__{c}" for c in ident.columns]
    out = pd.concat([n, w, q, ident], axis=1)
    for s in ["D_raw", "S", "B1b_raw", "B1c_raw"]:
        out[f"{s}__change_monkey_to_mouse"] = out[f"{s}__liv_hmouse"] - out[f"{s}__liv_hmonkey"]
    return out.reset_index().rename(columns={"index": "set"}).sort_values("D_raw__change_monkey_to_mouse")


def coupling() -> pd.DataFrame:
    a = pd.read_csv(OUT / "agreement_D_S.csv").set_index("cohort")
    r = pd.read_csv(OUT / "esm_cv_r2.csv")
    r = r[r.target == "rD"].set_index("cohort").cv_r2_esm64
    t = pd.read_csv(T24 / "cv_r2.csv")
    out = pd.DataFrame(index=ORDER)
    out["cv_r2_esm64_residual_D"] = r
    tt = t[(t.target == "delta_alpha") & (t.features == "all72")].set_index("cohort").cv_r2
    out["cv_r2_72_T2_4_s42"] = tt.reindex(ORDER)
    out["rho_p_D_S"] = a.rho_p_D_S
    out["spearman_D_S"] = a.spearman_D_S
    out["rho_p_B1b_S"] = a.rho_p_B1b_S
    out["rho_p_B1b_D"] = a.rho_p_B1b_D
    out["sign_agree_unmasked"] = a.sign_agree_unmasked
    out["sign_agree_expected"] = a.sign_agree_expected
    return out.reset_index().rename(columns={"index": "cohort"})


def main():
    K.write_csv(classes(), OUT / "summary_classes.csv")
    K.write_csv(core_sets(), OUT / "summary_core_sets.csv")
    K.write_csv(depth_programs(), OUT / "summary_depth_programs.csv")
    K.write_csv(coupling(), OUT / "summary_coupling.csv")
    print("wrote summary_*.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
