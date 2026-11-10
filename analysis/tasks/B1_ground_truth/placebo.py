"""B1 type I error: placebo gene sets on unaltered fits. No count is touched.

For every unaltered fit of a cohort: 200 draws per family (level rule, contrast rule) x band, each nested over the three
shares; the same selection and matching as a real run. Per method and share: reject05 (one-sided signed-rank) mean,
AUROC mean and SD, power_05 mean. Selections are shared across fits. results/analysis/B1/placebo/<cohort>.parquet
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import wilcoxon

from analysis.tasks.B1_ground_truth import alter as A
from analysis.tasks.B1_ground_truth import metrics as MT
from analysis.tasks.B1_ground_truth import rules as RU

N_DRAWS = 200


def one(g, cols, fam_alt, band, d, cohort):
    rng = np.random.default_rng(RU.useed("placebo", cohort, fam_alt, band, d))
    p = A.select_pairs(g, fam_alt, band, 42 + d % 2, rng=rng)
    out = []
    G = len(g)
    for sh in RU.SHARES:
        n = min(int(round(sh * G)), len(p))
        a, c = p["altered"].to_numpy()[:n], p["control"].to_numpy()[:n]
        for fit, S in cols.items():
            for m, s in S.items():
                sa, sc = s.reindex(a).to_numpy(), s.reindex(c).to_numpy()
                da, dc = -sa, -sc
                au = MT.auroc_miss(da, dc)[0]
                pair = np.isfinite(da) & np.isfinite(dc)
                diff = (da - dc)[pair]
                pv = float(wilcoxon(diff, alternative="greater", zero_method="zsplit").pvalue) if len(diff) >= 5 and np.any(diff != 0) else 1.0
                out.append((fam_alt, band, sh, n, fit, m, d, au, float(pv < 0.05), MT.power(da, dc)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort")
    ap.add_argument("--jobs", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--arm", default=None)
    a = ap.parse_args()
    c = a.cohort
    base = A.read_base(c)
    g, _ = A.gene_table(base, c)
    seq = MT.sequence_table()
    cols = {}
    runs = RU.COHORTS[c]["none_runs"] if a.arm is None else {s: f"b1{a.arm}-{c}-none-s{s}" for s in RU.ARM_SEEDS}
    methods = MT.METHODS if a.arm is None else MT.arm_methods(a.arm)
    for s, r in runs.items():
        f = RU.RES / "scores" / f"{r}.parquet"
        if not f.exists():
            continue
        sc = MT.join_static(pd.read_parquet(f).set_index("var_name"), seq)
        cols[r] = {m: sign * sc[col].astype(float) for m, (col, sign, _) in methods.items() if col in sc}
    jobs = [(alt, b, d) for alt in ("level", "pattern") for b in RU.BANDS for d in range(N_DRAWS)]
    res = Parallel(n_jobs=a.jobs, batch_size=8)(delayed(one)(g, cols, alt, b, d, c) for alt, b, d in jobs)
    t = pd.DataFrame([x for r in res for x in r],
                     columns=["rule", "band", "share", "n", "fit", "method", "draw", "AUROC", "reject05", "power_05"])
    (RU.RES / "placebo").mkdir(parents=True, exist_ok=True)
    t.to_parquet(RU.RES / "placebo" / (f"{c}.parquet" if a.arm is None else f"{c}__{a.arm}.parquet"))
    print(t.groupby("method")[["AUROC", "reject05", "power_05"]].mean(), flush=True)


if __name__ == "__main__":
    main()
