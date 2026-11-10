"""E5 analyses on 26 cohorts, both gates (EXPLORATORY). Definitions: rules.py. Loader, residuals and scan statistic: E2's.

Usage (from chimera/):  mamba run -n chimera_env python -m analysis.tasks.E5_regulatory.analyse --workers 13
Inputs: results/analysis/E5/props.parquet, regulons.parquet (annotate.py), E2 props (protein length), H1 and Z4 tables.
Outputs in results/analysis/E5/: scan.csv, regulon.csv, joint.csv, gene_resid.parquet.
"""
from __future__ import annotations

import argparse
import json
import os
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from analysis.common import gene_ids  # noqa: E402
from analysis.tasks.E2_protein import analyse as E2  # noqa: E402
from analysis.tasks.E5_regulatory import rules as RU  # noqa: E402
from analysis.tasks.T4_3_scoreboard import stats as ST  # noqa: E402

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "analysis" / "E5"
READ = E2.READ  # D_can, D_free, S
C = E2.C  # pi_raw, expr, detection, log_length


def props() -> pd.DataFrame:
    p = pd.read_parquet(OUT / "props.parquet")
    e2 = pd.read_parquet(REPO / "results/analysis/E2/props.parquet", columns=["hsapiens_gene_id", "log_length"])
    return p.merge(e2, on="hsapiens_gene_id", how="left")


def est_only(T: pd.DataFrame, p: str, covs: list[str], extra: np.ndarray | None = None) -> dict:
    R = np.column_stack([ST.midrank(T[p].to_numpy(float))] + [ST.midrank(T[y].to_numpy(float)) for y in READ])
    est = ST.rho_p_matrix(R, ST.design(T, covs, extra=extra))[1:]
    return dict(zip(READ, est))


def ok_binary(T: pd.DataFrame, p: str) -> bool:
    if p not in RU.BINARY:
        return T[p].nunique() >= 3
    npos = int((T[p] > 0).sum())
    return npos >= RU.MIN_FAMILY and len(T) - npos >= RU.MIN_FAMILY


def run_cohort(c: str) -> dict:
    t = E2.load(c, props())
    tissue = str(t.tissue.iloc[0])
    U = t[t.supp].dropna(subset=C).reset_index(drop=True)
    blocks = E2.paralog_blocks(pd.Index(U.hsapiens_gene_id)).to_numpy()
    seed0 = zlib.crc32(("E5" + c).encode()) & 0x7FFFFFFF
    rows = []
    for i, p in enumerate(list(RU.CONTINUOUS) + list(RU.BINARY)):
        m = U[p].notna().to_numpy()
        T = U[m].reset_index(drop=True)
        if not ok_binary(T, p):
            continue
        npos = int((T[p] > 0).sum()) if p in RU.BINARY else np.nan
        base = dict(cohort=c, tissue=tissue, property=p, n=len(T), n_pos=npos)
        r = E2.scan_one(T, p, C, RU.DRAWS, seed0 + i, blocks[m])
        TB = T.dropna(subset=["B1"]).reset_index(drop=True)
        RB = np.column_stack([ST.midrank(TB[p].to_numpy(float)), ST.midrank(TB["B1"].to_numpy(float))])
        r["B1"] = float(ST.rho_p_matrix(RB, ST.design(TB, C))[1])
        rows.append({**base, "cov": "C", **r})
        # sensitivities, estimates only (B1 beside for Ctau and Cbrd)
        sens = {"Ctau": ["tau"], **{k: v for k, v in RU.EXTRA_COVS.items() if p in RU.EXTRA_FOR[k]}}
        for name, add in sens.items():
            add = [a for a in add if a != p]
            if not add:
                continue
            Ts = T.dropna(subset=add).reset_index(drop=True)
            if not ok_binary(Ts, p):
                continue
            r2 = est_only(Ts, p, C + add)
            if name in ("Ctau", "Cbrd"):
                TB = Ts.dropna(subset=["B1"]).reset_index(drop=True)
                RB = np.column_stack([ST.midrank(TB[p].to_numpy(float)), ST.midrank(TB["B1"].to_numpy(float))])
                r2["B1"] = float(ST.rho_p_matrix(RB, ST.design(TB, C + add))[1])
            rows.append({**base, "n": len(Ts), "cov": name, **r2})
        Tc = T.dropna(subset=RU.CLASS_COVS).reset_index(drop=True)
        if ok_binary(Tc, p):
            rows.append({**base, "n": len(Tc), "cov": "Ccls",
                         **est_only(Tc, p, C, extra=Tc[RU.CLASS_COVS].to_numpy(float))})
    scan = pd.DataFrame(rows)

    # standardised residuals (C, C + tau, C + PubMed) for regulons, neighbourhood and named genes
    G = U[["hsapiens_gene_id", "symbol", "pi_raw", "expr", "tau", "nameable", "nameable_free", "saturated_flag",
           "saturated_flag_free", "sign_D_stable", "sign_D_stable_free", "D_can", "D_free", "S", "B1"]].copy()
    for y in READ:
        G[f"r_{y}"] = E2.resid_std(U, y, C)
        G[f"rt_{y}"] = E2.resid_std(U, y, C + ["tau"])
    mb = U.B1.notna().to_numpy()
    G["r_B1"] = np.nan
    G.loc[mb, "r_B1"] = E2.resid_std(U[mb].reset_index(drop=True), "B1", C)
    mp = U.log_pubmed_h.notna().to_numpy()
    for y in ("D_can", "D_free"):
        G[f"rp_{y}"] = np.nan
        G.loc[mp, f"rp_{y}"] = E2.resid_std(U[mp].reset_index(drop=True), y, C + ["log_pubmed_h"])
    G["cohort"], G["tissue"] = c, tissue
    RCOLS = [f"r_{y}" for y in READ] + [f"rt_{y}" for y in READ] + ["r_B1", "rp_D_can", "rp_D_free"]

    # regulon scan (DoRothEA A-C)
    reg = pd.read_parquet(OUT / "regulons.parquet")
    gi = pd.Series(np.arange(len(U)), index=U.hsapiens_gene_id)
    reg = reg[reg.g.isin(gi.index)]
    rrows = []
    for tf, sub in reg.groupby("tf"):
        if len(sub) < RU.MIN_FAMILY:
            continue
        mask = np.zeros(len(U), bool)
        mask[gi[sub.g].to_numpy()] = True
        r = dict(cohort=c, tissue=tissue, tf=tf, n=int(mask.sum()),
                 tf_in_universe=bool(sub.tf_gene.iloc[0] in gi.index) if isinstance(sub.tf_gene.iloc[0], str) else False)
        for col in RCOLS:
            v = G[col].to_numpy()
            a, b = v[mask & np.isfinite(v)], v[~mask & np.isfinite(v)]
            r[f"{col}_mean"] = float(a.mean()) if len(a) else np.nan
            r[f"{col}_p"] = float(sps.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 2 else np.nan
        rrows.append(r)
    regd = pd.DataFrame(rrows)

    # joint model: standardised residual read-out on standardised residual properties (complete cases)
    jp = RU.JOINT
    J = U.dropna(subset=jp).reset_index(drop=True)
    jrows = []
    if len(J) > 500:
        Z = ST.design(J, C)
        X = np.column_stack([ST.resid(ST.midrank(J[p].to_numpy(float))[:, None], Z)[:, 0] for p in jp])
        keep = X.std(0) > 0
        X = X[:, keep] / X[:, keep].std(0)
        names = [p for p, k in zip(jp, keep) if k]
        A = np.column_stack([np.ones(len(J)), X])
        XtX = np.linalg.inv(A.T @ A)
        for y in READ + ["B1"]:
            if y == "B1":
                mb = J.B1.notna().to_numpy()
                yv = np.full(len(J), np.nan)
                yv[mb] = E2.resid_std(J[mb].reset_index(drop=True), "B1", C)
                AA, yy = A[mb], yv[mb]
                XtXb = np.linalg.inv(AA.T @ AA)
            else:
                AA, yy, XtXb = A, E2.resid_std(J, y, C), XtX
            beta, *_ = np.linalg.lstsq(AA, yy, rcond=None)
            res = yy - AA @ beta
            se = np.sqrt(np.diag(XtXb) * res.var(ddof=AA.shape[1]))
            for k, p in enumerate(names):
                jrows.append(dict(cohort=c, tissue=tissue, readout=y, property=p, beta=float(beta[k + 1]),
                                  se=float(se[k + 1]), n=len(yy)))
    return dict(scan=scan, genes=G, regulon=regd, joint=pd.DataFrame(jrows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--cohorts", nargs="*")
    a = ap.parse_args()
    cs = a.cohorts or E2.cohorts()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_cohort, cs))
    for k, fn in (("scan", "scan.csv"), ("regulon", "regulon.csv"), ("joint", "joint.csv")):
        pd.concat([r[k] for r in res], ignore_index=True).to_csv(OUT / fn, index=False)
    pd.concat([r["genes"] for r in res], ignore_index=True).to_parquet(OUT / "gene_resid.parquet")
    json.dump(dict(cohorts=cs, draws=RU.DRAWS), open(OUT / "_run.json", "w"), indent=1)
    print("done", len(cs))


if __name__ == "__main__":
    main()
