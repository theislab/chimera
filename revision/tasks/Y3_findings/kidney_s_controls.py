"""Y3 (EXPLORATORY, Family 3): does S_g from a healthy-only kidney model survive H-D1's label-permuted controls?

P13's h1.tsv gives S_ema positive in all nine kidney cell types (rho_p +0.007 to +0.065) with stratified-permutation
p values and block CIs, but h1_controls_summary.tsv holds controls for D only. This recomputes S_ema, D and B1b on the
permuted-label DE fits with the P13 universe and design. H-D1's pass rule concerns D and still fails; nothing here
touches it, the row or any status.

Definitions, fixed before anything below was computed (S_ema primary values were known from h1.tsv, controls were not):
    statistic  rho_p as hd1.quick: ranks of Q = sign(l^h) * m and of the score, residualised on C + splines of
               rank |l^h| and rank SE(m), Pearson. Same universe() per fit.
    controls   every fit in de/perm: human labels permuted (mouse primary), mouse labels permuted (human primary).
    per type   empirical two-sided-style tail: share of control |values| at or above the primary value, one tail
               (control >= primary), per relabelling; p = (1 + k) / (1 + n); BH over nine types within relabelling.
    across     mean rho_p over the nine types, per draw index, where every type has that draw.
    survives   (a) across-type mean above the 95th percentile of both control distributions, and (b) at least one type
               with BH q < 0.05 under both relabellings. Otherwise the result sits inside the controls and is left out.
    baseline   B1b goes through the same procedure, as the comparison, and D as the check against P13's summary.

Writes results/revision/Y3/kidney_controls_draws.csv, kidney_controls_types.csv, kidney_controls_across.csv.

Usage: PYTHONPATH=$PWD python -m revision.tasks.Y3_findings.kidney_s_controls [--workers 16]
"""

from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from revision.common import gene_ids  # noqa: E402
from revision.tasks.P13_kidney import hd1 as P  # noqa: E402
from revision.tasks.T4_3_scoreboard import stats as S  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "revision" / "Y3"
TYPES = ["pt", "tal", "dct", "pc", "ic", "ec", "fib", "myeloid", "lymphoid"]
SCORES = ["S_ema", "D", "B1b"]
TAB = None


def init():
    global TAB
    TAB = pd.read_parquet(P.GENES)


def one(job):
    t, kind, i, hf, mf = job
    u = P.universe(TAB, P.read_de(hf), P.read_de(mf)).dropna(subset=SCORES)
    Z = S.design(u, P.C_COLS + P.NUIS)
    R = np.column_stack([S.midrank(u["Q"])] + [S.midrank(u[s]) for s in SCORES])
    est = S.rho_p_matrix(R, Z)[1:]
    return dict(cell_type=t, kind=kind, draw=i, n=len(u), **{f"rho_{s}": float(e) for s, e in zip(SCORES, est)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for t in TYPES:
        hp, mp = P.DE / f"human_primary_{t}.tsv", P.DE / f"mouse_primary_{t}.tsv"
        jobs.append((t, "primary", -1, hp, mp))
        for f in sorted((P.DE / "perm").glob(f"human_perm*_{t}.tsv")):
            jobs.append((t, "human_permuted", int(f.name.split("perm")[1].split("_")[0]), f, mp))
        for f in sorted((P.DE / "perm").glob(f"mouse_perm*_{t}.tsv")):
            jobs.append((t, "mouse_permuted", int(f.name.split("perm")[1].split("_")[0]), hp, f))
    print(len(jobs), "fits", flush=True)
    with ProcessPoolExecutor(a.workers, initializer=init) as ex:
        d = pd.DataFrame(list(ex.map(one, jobs, chunksize=8)))
    d.to_csv(OUT / "kidney_controls_draws.csv", index=False)
    prim = d[d.kind == "primary"].set_index("cell_type")
    rows, across = [], []
    for kind in ("human_permuted", "mouse_permuted"):
        c = d[d.kind == kind]
        for s in SCORES:
            ps = []
            for t in TYPES:
                v = c.loc[c.cell_type == t, f"rho_{s}"].to_numpy()
                obs = prim.loc[t, f"rho_{s}"]
                p = (1 + (v >= obs).sum()) / (1 + len(v))
                ps.append(p)
                rows.append(dict(cell_type=t, kind=kind, score=s, n_draws=len(v), primary=obs, control_mean=v.mean(), control_sd=v.std(ddof=1),
                                 control_q95=np.quantile(v, 0.95), control_max=v.max(), p_control=p))
            q = S.bh(ps)
            for k, qq in zip(range(-len(TYPES), 0), q):
                rows[k]["q_control"] = qq
            w = c.pivot(index="draw", columns="cell_type", values=f"rho_{s}").dropna()
            m = w.mean(axis=1).to_numpy()
            obs = prim[f"rho_{s}"].mean()
            across.append(dict(kind=kind, score=s, n_draws=len(m), primary_mean=obs, control_mean=m.mean(), control_sd=m.std(ddof=1), control_q95=np.quantile(m, 0.95),
                               control_max=m.max(), p_control=(1 + (m >= obs).sum()) / (1 + len(m))))
    pd.DataFrame(rows).to_csv(OUT / "kidney_controls_types.csv", index=False)
    pd.DataFrame(across).to_csv(OUT / "kidney_controls_across.csv", index=False)


if __name__ == "__main__":
    main()
