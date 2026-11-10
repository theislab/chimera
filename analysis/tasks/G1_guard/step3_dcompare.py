"""G1 step 3: guard-on fits at full depth (liver H+Mouse, testis H+Mouse; seeds 45, 46) against the canonical D. No gold is read.

D = alpha - prior (final_state.npz), as analysis/common/controls.py. Canonical seed sets: controls.SEED_SETS (the five seeds of the paper's D) and, as sensitivity, the ten H1 seeds.
For a guard run of seed s: rho(D_guard, mean of the canonical seeds other than s), raw Spearman and residual given C (controls.design). The canonical floor F is the leave-one-seed-out rho of each
canonical seed against the mean of the others, on the same rows. Inside seed range: rho_guard >= min F. Also the paired rho with the canonical fit of the same seed.
    python -m analysis.tasks.G1_guard.step3_dcompare
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.tasks.G1_guard.step3_prepare import RES

GUARD = {"liv_hmouse": {45: "rev-g1-liv_hmouse-s45", 46: "rev-g1-liv_hmouse-s46"},
         "test_hmouse": {45: "rev-g1-test_hmouse-s45", 46: "rev-g1-test_hmouse-s46"}}
FORMS = {"symmetric": "rev-g1-", "upper_only": "rev-g1u-"}  # upper-only: analysis/tasks/G1_guard/upper_only.py


def main():
    h1 = pd.read_csv(C.gene_ids.REPO_ROOT / "results/analysis/H1/runs.tsv", sep="\t")
    rows = []
    for cohort, guard in GUARD.items():
        adata = C.read_adata(cohort)
        canon = C.CANONICAL[cohort]
        prior_raw = C.raw_prior(cohort, canon, adata.var["conservation_prior"].to_numpy(dtype=float))
        cov = K.cohort_covariates(cohort, adata=adata, prior_raw=prior_raw)
        ok = cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
        Z = K.design(cov.loc[ok])
        sets = {"five": K.SEED_SETS[cohort],
                "ten": {int(r.seed): r.run for r in h1[(h1.cohort == cohort) & h1.kept].itertuples()}}
        G = {(f, s): (r.replace("rev-g1-", p), K.read_run(r.replace("rev-g1-", p))) for f, p in FORMS.items() for s, r in guard.items() if K.run_finished(r.replace("rev-g1-", p))}
        for name, ss in sets.items():
            M = pd.DataFrame({s: K.read_run(r)["D"] for s, r in ss.items()}).iloc[ok.nonzero()[0]].reset_index(drop=True)
            F = K.floor(M, Z)
            for (form, s), (gname, g) in G.items():
                fin = bool(np.isfinite(g["D"]).all())
                d = g["D"][ok]
                ref = K.consensus_excluding(M, s)
                r = dict(cohort=cohort, seed_set=name, n_canonical=M.shape[1], form=form, guard_seed=s, guard_run=gname, finite=fin)
                if fin:
                    rp = K.rho_pair(d, ref, Z)
                    r.update(rho_raw=rp["spearman"], rho_resid=rp["residual"])
                    if s in M.columns:
                        pp = K.rho_pair(d, M[s].to_numpy(), Z)
                        r.update(paired_raw=pp["spearman"], paired_resid=pp["residual"])
                        r.update(canon_same_seed_raw=float(F.set_index("seed").loc[s, "spearman"]), canon_same_seed_resid=float(F.set_index("seed").loc[s, "residual"]))
                r.update(F_min_raw=F["spearman"].min(), F_med_raw=F["spearman"].median(), F_max_raw=F["spearman"].max(),
                         F_min_resid=F["residual"].min(), F_med_resid=F["residual"].median(), F_max_resid=F["residual"].max())
                if fin:
                    r["inside_raw"] = bool(r["rho_raw"] >= r["F_min_raw"])
                    r["inside_resid"] = bool(r["rho_resid"] >= r["F_min_resid"])
                rows.append(r)
    # symmetric against upper-only of the same seed number (splits differ, so this is two more seeds, not a pair)
    for cohort, guard in GUARD.items():
        for s, r in guard.items():
            a, b = K.read_run(r)["D"], K.read_run(r.replace("rev-g1-", "rev-g1u-"))["D"]
            rows.append(dict(cohort=cohort, seed_set="sym_vs_upper", guard_seed=s, guard_run=r, form="sym_vs_upper", rho_raw=float(pd.Series(a).corr(pd.Series(b), method="spearman"))))
    d = pd.DataFrame(rows)
    K.write_csv(d, RES / "d_full_depth.csv")
    pd.set_option("display.width", 250)
    print(d.round(4).to_string())


if __name__ == "__main__":
    main()
