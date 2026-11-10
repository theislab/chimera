"""T3.5 offline null: 1,000-permutation closed-form ridge projection (no training).

Model of a count-permuted run without training: the GIN output is taken as a ridge
projection of the fit-quality margin onto the features, ``c_hat = a + b * P_F S`` with
``P_F y = X (X'X + I)^-1 X' y``, ``X = [1, F]`` (A2 ``a2_proj.py``), and
``D_hat = sigmoid(logit(prior) + c_hat) - prior``; ``a, b`` are the OLS fit of the reference
correction on ``P_F S_ref``. A permuted control moves the counts, so its margin at slot g is
``S_ref[perm[g]]``; in arm b the expression scalars (feature columns 66-68) move as well.

Each draw redraws ``perm`` with the arm's scheme (a, b: free; c: within the 20 expression bins
of ``make_control_h5ad.py``) and records rho of ``D_hat`` with the reference D:

* slot: against ``D_ref[g]`` (what sequence would preserve),
* moved: against ``D_ref[perm[g]]`` (what counts would carry),

on the Spearman scale and on the covariate-residual scale (05 section 3; the control's
covariates have the count covariates moved, the slot covariates in place). The central 95%
of the draws is the null interval of section 9: slot rho is high only above its 97.5th
percentile. ``c_spearman_slot`` is A2's statistic, rho(c_ref, P_F S_perm), for comparison
with A2 (free 99th pct |rho| and within-bin mean; ``projection.csv`` of T2.4).

Usage::

    python analysis/tasks/T3_5_count_perm/offline_null.py [--n-perm 1000] [--reference s42|available]
        [--cohorts ...] [--outdir DIR] [--force]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_control_h5ad import CONTROLS_TSV, EXPR_COLS, N_BINS, expression_bins  # noqa: E402

ARMS_BY_COHORT = {"liv_hmouse": ["a", "b", "c"], "test_hmouse": ["a", "b", "c"], "liv_4sp": ["a", "b", "c"]}
CONTROL_DIR = gene_ids.REPO_ROOT / "data" / "analysis" / "controls"


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class Projection:
    """Ridge projection onto [1, F] with the reference calibration of c on P_F S."""

    def __init__(self, F: np.ndarray, S_ref: np.ndarray, c_ref: np.ndarray, prior: np.ndarray):
        self.F = np.asarray(F, dtype=float)
        self.prior = np.asarray(prior, dtype=float)
        self.logit = np.log(self.prior / (1 - self.prior))
        X = np.c_[np.ones(len(self.F)), self.F]
        self._X, self._Inv = X, np.linalg.inv(X.T @ X + np.eye(X.shape[1]))
        p = self.project(self.F, S_ref)
        self.b, self.a = np.polyfit(p, c_ref, 1)

    def project(self, F: np.ndarray, y: np.ndarray) -> np.ndarray:
        if F is self.F:
            return self._X @ (self._Inv @ (self._X.T @ y))
        X = np.c_[np.ones(len(F)), F]
        return X @ np.linalg.solve(X.T @ X + np.eye(X.shape[1]), X.T @ y)

    def predict(self, S: np.ndarray, F: np.ndarray | None = None):
        c = self.a + self.b * self.project(self.F if F is None else F, S)
        return c, _sigmoid(self.logit + c) - self.prior


def draw_perm(arm: str, G: int, bins: np.ndarray, rng) -> np.ndarray:
    if arm in ("a", "b"):
        return rng.permutation(G)
    perm = np.arange(G)
    for k in range(N_BINS):
        m = np.where(bins == k)[0]
        perm[m] = m[rng.permutation(len(m))]
    return perm


def arm_covariates(cov: pd.DataFrame, perm: np.ndarray) -> pd.DataFrame:
    """Covariates of a control: count covariates move with the counts, slot covariates stay."""
    out = cov[K.COVARIATE_COLUMNS].copy()
    for c in K.COUNT_COVARIATES:
        out[c] = cov[c].to_numpy()[perm]
    return out


def slot_moved(arm_vals: np.ndarray, arm_cov: pd.DataFrame, ref_vals: np.ndarray, ref_cov: pd.DataFrame,
               perm: np.ndarray, residual: bool = True) -> dict:
    """rho of a control's per-slot values with the reference by slot and by moved gene."""
    out = dict(spearman_slot=float(spearmanr(arm_vals, ref_vals)[0]),
               spearman_moved=float(spearmanr(arm_vals, ref_vals[perm])[0]))
    if residual:
        U = ref_cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
        Ua = arm_cov[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
        ea = np.full(len(arm_vals), np.nan)
        er = np.full(len(ref_vals), np.nan)
        ea[Ua] = K.residual_rank(arm_vals[Ua], K.design(arm_cov[Ua]))
        er[U] = K.residual_rank(ref_vals[U], K.design(ref_cov[U]))
        m = np.isfinite(ea) & np.isfinite(er)
        out["residual_slot"] = float(np.corrcoef(ea[m], er[m])[0, 1])
        er_m = er[perm]
        m = np.isfinite(ea) & np.isfinite(er_m)
        out["residual_moved"] = float(np.corrcoef(ea[m], er_m[m])[0, 1])
        out["n_residual_slot"] = int((np.isfinite(ea) & np.isfinite(er)).sum())
        out["n_residual_moved"] = int(m.sum())
    return out


def null_draws(arm: str, proj: Projection, S_ref: np.ndarray, c_ref: np.ndarray, D_ref: np.ndarray,
               cov: pd.DataFrame, bins: np.ndarray, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    G = len(S_ref)
    rows = []
    for i in range(n_perm):
        perm = draw_perm(arm, G, bins, rng)
        F = proj.F
        if arm == "b":
            F = proj.F.copy()
            F[:, EXPR_COLS] = proj.F[perm][:, EXPR_COLS]
        c_hat, D_hat = proj.predict(S_ref[perm], F)
        r = dict(draw=i, c_spearman_slot=float(spearmanr(c_ref, c_hat)[0]))
        r.update(slot_moved(D_hat, arm_covariates(cov, perm), D_ref, cov, perm))
        rows.append(r)
    return pd.DataFrame(rows)


def summarise(draws: pd.DataFrame) -> dict:
    out = dict(n_perm=len(draws))
    for col in [c for c in draws.columns if c.startswith(("spearman_", "residual_", "c_spearman"))]:
        v = draws[col].to_numpy()
        out.update({f"{col}__mean": float(v.mean()), f"{col}__sd": float(v.std()),
                    f"{col}__p2_5": float(np.percentile(v, 2.5)), f"{col}__p97_5": float(np.percentile(v, 97.5))})
    out["c_spearman_slot__p99_abs"] = float(np.percentile(np.abs(draws["c_spearman_slot"]), 99))
    return out


def reference(cohort: str, which: str, drop=(), exclude_seed: int | None = None) -> dict:
    """Reference D, S, c (mean over seeds) and the seeds used."""
    if which == "s42":
        runs = {42: C.CANONICAL[cohort]}
    else:
        runs = K.seed_runs(cohort, drop)
    if exclude_seed is not None and len(runs) > 1:
        runs = {s: r for s, r in runs.items() if s != exclude_seed}
    st = [K.read_run(r) for r in runs.values()]
    prior = st[0]["prior"]
    for x in st[1:]:
        if not np.allclose(x["prior"], prior):
            raise AssertionError(f"{cohort}: prior differs between seeds")
    return dict(seeds=sorted(runs), runs=list(runs.values()), prior=prior,
                D=np.mean([x["D"] for x in st], 0), S=np.mean([x["S_g_ema"] for x in st], 0),
                c=np.mean([x["correction"] for x in st], 0))


def cohort_inputs(cohort: str) -> dict:
    a = C.read_adata(cohort)
    import scipy.sparse as sp

    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    cov = K.cohort_covariates(cohort, a)
    return dict(F=np.asarray(a.varm["gene_features"], dtype=float), cov=cov, bins=expression_bins(X),
                var_names=np.asarray(a.var_names))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=35)
    ap.add_argument("--reference", nargs="+", default=["s42", "available"], choices=["s42", "available"])
    ap.add_argument("--cohorts", nargs="+", default=list(ARMS_BY_COHORT))
    add_common_args(ap)
    args = ap.parse_args()
    outdir = args.outdir or str(gene_ids.REPO_ROOT / "results" / "analysis" / "T3_5" / "null")
    out = TaskOutput("T3_5_null", outdir, prerun=args.prerun, force=args.force)
    controls = pd.read_csv(CONTROLS_TSV, sep="\t")
    summ, expected = [], []
    for ci, cohort in enumerate(args.cohorts):
        print("==", cohort, flush=True)
        inp = cohort_inputs(cohort)
        for which in args.reference:
            ref = reference(cohort, which)
            proj = Projection(inp["F"], ref["S"], ref["c"], ref["prior"])
            c_id, D_id = proj.predict(ref["S"])
            ident = slot_moved(D_id, inp["cov"], ref["D"], inp["cov"], np.arange(len(D_id)))
            base = dict(cohort=cohort, reference=which, seeds=";".join(map(str, ref["seeds"])),
                        calib_a=proj.a, calib_b=proj.b, identity_c_spearman=float(spearmanr(ref["c"], c_id)[0]),
                        identity_D_spearman=ident["spearman_slot"], identity_D_residual=ident["residual_slot"])
            for ai, arm in enumerate(ARMS_BY_COHORT[cohort]):
                d = null_draws(arm, proj, ref["S"], ref["c"], ref["D"], inp["cov"], inp["bins"], args.n_perm,
                               args.seed * 1000 + ci * 10 + ai)
                out.csv(d, f"draws/{cohort}__arm{arm}__{which}.csv")
                s = dict(base, arm=arm, **summarise(d))
                summ.append(s)
                print(f"  {which} arm {arm}: slot D {s['spearman_slot__mean']:.3f} "
                      f"[{s['spearman_slot__p2_5']:.3f}, {s['spearman_slot__p97_5']:.3f}], "
                      f"moved D {s['spearman_moved__mean']:.3f}, c slot {s['c_spearman_slot__mean']:.3f} "
                      f"(p99|.| {s['c_spearman_slot__p99_abs']:.3f})", flush=True)
            # projection expectation for the actual control permutations
            for _, row in controls[controls.cohort == cohort].iterrows():
                f = CONTROL_DIR / f"{row['control']}.perm.npz"
                if not f.exists():
                    continue
                perm = np.load(f)["perm"]
                F = inp["F"]
                if row["arm"] == "b":
                    F = F.copy()
                    F[:, EXPR_COLS] = inp["F"][perm][:, EXPR_COLS]
                _, D_hat = proj.predict(ref["S"][perm], F)
                r = slot_moved(D_hat, arm_covariates(inp["cov"], perm), ref["D"], inp["cov"], perm)
                expected.append(dict(control=row["control"], cohort=cohort, arm=row["arm"], reference=which,
                                     **{f"expected_{k}": v for k, v in r.items()}))
    out.csv(pd.DataFrame(summ), "null_summary.csv")
    out.csv(pd.DataFrame(expected), "expected_for_controls.csv")
    out.finish(dict(n_perm=args.n_perm, seed=args.seed, references=args.reference))
    return 0


if __name__ == "__main__":
    sys.exit(main())
