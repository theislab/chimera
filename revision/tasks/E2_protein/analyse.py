"""E2 analyses on 26 cohorts, both gates (EXPLORATORY). Definitions: rules.py.

Usage (from chimera/):  mamba run -n chimera_env python -m revision.tasks.E2_protein.analyse --workers 13
Inputs: results/revision/E2/props.parquet (annotate.py), H1 and Z4 per-gene tables, Y5 paralog pairs.
Outputs in results/revision/E2/: scan.csv, family.csv, complex.csv, complex_means.csv, joint.csv, gene_resid.parquet.
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
from scipy.sparse import coo_matrix  # noqa: E402
from scipy.sparse.csgraph import connected_components  # noqa: E402
from scipy.stats import norm  # noqa: E402

from revision.common import gene_ids  # noqa: E402
from revision.tasks.E2_protein import rules as RU  # noqa: E402
from revision.tasks.T4_3_scoreboard import stats as ST  # noqa: E402

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "revision" / "E2"
H1G = REPO / "results" / "revision" / "H1" / "genes"
Z4G = REPO / "results" / "revision" / "Z4" / "h1" / "genes"
READ = ["D_can", "D_free", "S"]
C = ["pi_raw", "expr", "detection", "log_length"]


def cohorts() -> list[str]:
    return sorted(p.stem for p in H1G.glob("*.parquet"))


def paralog_blocks(genes: pd.Index) -> pd.Series:
    pp = pd.read_parquet(REPO / "results/revision/Y5/paralog_pairs_id40.parquet")
    idx = {g: i for i, g in enumerate(genes)}
    pp = pp[pp.g.isin(idx) & pp.h.isin(idx)]
    n = len(genes)
    a = pp.g.map(idx).to_numpy()
    b = pp.h.map(idx).to_numpy()
    M = coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n))
    _, lab = connected_components(M, directed=False)
    return pd.Series(lab, index=genes)


def load(c: str, props: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_parquet(H1G / f"{c}.parquet")
    z = pd.read_parquet(Z4G / f"{c}.parquet", columns=["hsapiens_gene_id", "D_raw", "S_ema", "count_support", "nameable",
                                                        "saturated_flag", "sign_D_stable", "strict_support"])
    z = z.rename(columns={k: f"{k}_free" for k in z.columns if k != "hsapiens_gene_id"})
    t = h.merge(z, on="hsapiens_gene_id", how="inner", validate="1:1")
    assert len(t) == len(h) == len(z), c
    t["D_can"] = t["D_raw"]
    t["D_free"] = t["D_raw_free"]
    t["S"] = t["S_ema"]
    t["supp"] = t["count_support"].astype(bool) & t["count_support_free"].astype(bool)
    t = t.merge(props, on="hsapiens_gene_id", how="left")
    return t


def resid_std(T: pd.DataFrame, y: str, covs: list[str]) -> np.ndarray:
    Z = ST.design(T, covs)
    r = ST.resid(ST.midrank(T[y].to_numpy(float))[:, None], Z)[:, 0]
    return r / r.std()


def scan_one(T: pd.DataFrame, prop: str, covs: list[str], draws: int, seed: int, block: np.ndarray | None):
    R = np.column_stack([ST.midrank(T[prop].to_numpy(float))] + [ST.midrank(T[y].to_numpy(float)) for y in READ])
    Z = ST.design(T, covs)
    est = ST.rho_p_matrix(R, Z)[1:]
    out = dict(zip(READ, est))
    if draws:
        B = ST.bootstrap(R, Z, draws, np.random.default_rng(seed), block)[:, 1:]
        sd = np.nanstd(B, axis=0)
        for k, y in enumerate(READ):
            lo, hi = ST.pct_ci(B[:, k])
            out[f"{y}_lo"], out[f"{y}_hi"] = lo, hi
            out[f"{y}_p"] = float(2 * norm.sf(abs(est[k]) / sd[k])) if sd[k] > 0 else 1.0
    return out


def run_cohort(c: str) -> dict:
    props = pd.read_parquet(OUT / "props.parquet")
    t = load(c, props)
    tissue = str(t.tissue.iloc[0])
    U = t[t.supp].dropna(subset=["pi_raw", "expr", "detection", "log_length"]).reset_index(drop=True)
    blocks = paralog_blocks(pd.Index(U.hsapiens_gene_id)).to_numpy()
    rows = []
    allprops = list(RU.CONTINUOUS) + list(RU.BINARY)
    seed0 = zlib.crc32(c.encode()) & 0x7FFFFFFF
    for i, p in enumerate(allprops):
        m = U[p].notna().to_numpy()
        T = U[m].reset_index(drop=True)
        if p in RU.BINARY:
            npos = int((T[p] > 0).sum())
            if npos < RU.MIN_FAMILY or len(T) - npos < RU.MIN_FAMILY:
                continue
        else:
            npos = np.nan
            if T[p].nunique() < 3:
                continue
        covs = [k for k in C if not (p == "log_length" and k == "log_length")]
        base = dict(cohort=c, tissue=tissue, property=p, n=len(T), n_pos=npos)
        r = scan_one(T, p, covs, RU.DRAWS, seed0 + i, blocks[m])
        TB = T.dropna(subset=["B1"]).reset_index(drop=True)  # comparator: pseudobulk profile correlation, estimate only
        RB = np.column_stack([ST.midrank(TB[p].to_numpy(float)), ST.midrank(TB["B1"].to_numpy(float))])
        r["B1"] = float(ST.rho_p_matrix(RB, ST.design(TB, covs))[1])
        rows.append({**base, "cov": "C", "universe": "supp", **r})
        Tt = T.dropna(subset=["tau"]).reset_index(drop=True)
        rows.append({**base, "cov": "Ctau", "universe": "supp", **scan_one(Tt, p, covs + ["tau"], 0, 0, None)})
        if p not in RU.CLASS_COVS:
            Tc = T.dropna(subset=RU.CLASS_COVS).reset_index(drop=True)
            if p in RU.BINARY and min((Tc[p] > 0).sum(), (Tc[p] == 0).sum()) < RU.MIN_FAMILY:
                pass
            else:
                Zc = Tc[RU.CLASS_COVS].to_numpy(float)
                R = np.column_stack([ST.midrank(Tc[p].to_numpy(float))] + [ST.midrank(Tc[y].to_numpy(float)) for y in READ])
                est = ST.rho_p_matrix(R, ST.design(Tc, covs, extra=Zc))[1:]
                rows.append({**base, "cov": "Ccls", "universe": "supp", **dict(zip(READ, est))})
    # canonical on all genes (imputation included), estimate only
    A = t.dropna(subset=["pi_raw", "expr", "detection", "log_length"]).reset_index(drop=True)
    for p in allprops:
        T = A[A[p].notna()].reset_index(drop=True)
        if p in RU.BINARY and min((T[p] > 0).sum(), (T[p] == 0).sum()) < RU.MIN_FAMILY:
            continue
        covs = [k for k in C if not (p == "log_length" and k == "log_length")]
        R = np.column_stack([ST.midrank(T[p].to_numpy(float)), ST.midrank(T["D_can"].to_numpy(float))])
        est = ST.rho_p_matrix(R, ST.design(T, covs))[1]
        rows.append(dict(cohort=c, tissue=tissue, property=p, n=len(T), cov="C", universe="all", D_can=est))
    scan = pd.DataFrame(rows)

    # standardised residuals on C, universe supp
    G = U[["hsapiens_gene_id", "symbol", "pi_raw", "expr", "tau", "nameable", "nameable_free", "saturated_flag",
           "saturated_flag_free", "sign_D_stable", "sign_D_stable_free", "D_can", "D_free", "S"]].copy()
    for y in READ:
        G[f"r_{y}"] = resid_std(U, y, C)
        G[f"rt_{y}"] = resid_std(U, y, C + ["tau"])
    mb = U.B1.notna().to_numpy()
    G["r_B1"] = np.nan
    G.loc[mb, "r_B1"] = resid_std(U[mb].reset_index(drop=True), "B1", C)
    G["cohort"], G["tissue"] = c, tissue
    RCOLS = [f"r_{y}" for y in READ] + [f"rt_{y}" for y in READ] + ["r_B1"]

    # family scans: Pfam families and TF DBD families
    fam_rows = []
    fams = {}
    for g, s in zip(U.hsapiens_gene_id, U.pfam_list):
        if isinstance(s, str) and s:
            for f in set(s.split(";")):
                fams.setdefault(("pfam", f), []).append(g)
    for g, d in zip(U.hsapiens_gene_id, U.tf_dbd):
        if isinstance(d, str):
            fams.setdefault(("tf_dbd", d), []).append(g)
    gi = pd.Series(np.arange(len(U)), index=U.hsapiens_gene_id)
    for (kind, f), gs in fams.items():
        if len(gs) < RU.MIN_FAMILY:
            continue
        idx = gi[gs].to_numpy()
        mask = np.zeros(len(U), bool)
        mask[idx] = True
        r = dict(cohort=c, tissue=tissue, kind=kind, family=f, n=int(mask.sum()))
        for col in RCOLS:
            v = G[col].to_numpy()
            a, b = v[mask & np.isfinite(v)], v[~mask & np.isfinite(v)]
            y = col[2:] if col.startswith("r_") else "tau_" + col[3:]
            r[f"{y}_mean"] = float(a.mean()) if len(a) else np.nan
            r[f"{y}_p"] = float(sps.ttest_ind(a, b, equal_var=False).pvalue) if len(a) > 2 else np.nan
        fam_rows.append(r)
    fam = pd.DataFrame(fam_rows)

    # complex coherence
    cm = pd.read_parquet(OUT / "complex_members.parquet")
    cm = cm[cm.g.isin(gi.index)]
    pairs = pd.read_parquet(REPO / "results/revision/Y5/paralog_pairs_id40.parquet")
    bl = pd.Series(blocks, index=U.hsapiens_gene_id)
    strata = (pd.qcut(U.pi_raw.rank(method="first"), 5, labels=False).astype(int) * 5
              + pd.qcut(U.expr.rank(method="first"), 5, labels=False).astype(int)).to_numpy()
    by_stratum = {s: np.flatnonzero(strata == s) for s in np.unique(strata)}
    rng = np.random.default_rng(seed0)
    cx_rows = []
    for (cx, nm), sub in cm.groupby(["complex", "name"]):
        gs = list(dict.fromkeys(sub.g))
        for guard in (False, True):
            g2 = gs
            if guard:  # one member per paralog family
                seen, g2 = set(), []
                for g in gs:
                    if bl[g] not in seen:
                        seen.add(bl[g])
                        g2.append(g)
            if len(g2) < RU.MIN_COMPLEX:
                continue
            idx = gi[g2].to_numpy()
            st = strata[idx]
            draws_idx = np.column_stack([by_stratum[s][rng.integers(0, len(by_stratum[s]), RU.NULL_SETS)] for s in st])
            r = dict(cohort=c, tissue=tissue, complex=cx, name=nm, n=len(idx), guard=guard)
            for col in RCOLS:
                y = col[2:] if col.startswith("r_") else "tau_" + col[3:]
                v = G[col].to_numpy()
                if not np.isfinite(v[idx]).all():
                    continue
                v = np.where(np.isfinite(v), v, 0.0)
                sd_obs = v[idx].std(ddof=1)
                sd_null = v[draws_idx].std(axis=1, ddof=1)
                r[f"{y}_mean"] = float(v[idx].mean())
                r[f"{y}_sd"] = float(sd_obs)
                r[f"{y}_z"] = float((sd_obs - sd_null.mean()) / sd_null.std())
                r[f"{y}_pless"] = float((np.sum(sd_null <= sd_obs) + 1) / (RU.NULL_SETS + 1))
            cx_rows.append(r)
    cxd = pd.DataFrame(cx_rows)

    # joint model: standardised residual read-out on standardised residual key properties (complete cases)
    jp = [p for p in RU.KEY_PROPS if p not in ("loeuf",)] + ["loeuf"]
    J = U.dropna(subset=jp + ["tau"]).reset_index(drop=True)
    jrows = []
    if len(J) > 500:
        covs = C
        Z = ST.design(J, covs)
        X = np.column_stack([ST.resid(ST.midrank(J[p].to_numpy(float))[:, None], Z)[:, 0] for p in jp])
        keep = X.std(0) > 0
        X = X[:, keep] / X[:, keep].std(0)
        names = [p for p, k in zip(jp, keep) if k]
        for y in READ:
            yv = resid_std(J, y, covs)
            beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(J)), X]), yv, rcond=None)
            res = yv - np.column_stack([np.ones(len(J)), X]) @ beta
            s2 = res.var(ddof=X.shape[1] + 1)
            XtX = np.linalg.inv(np.column_stack([np.ones(len(J)), X]).T @ np.column_stack([np.ones(len(J)), X]))
            se = np.sqrt(np.diag(XtX) * s2)
            for k, p in enumerate(names):
                jrows.append(dict(cohort=c, tissue=tissue, readout=y, property=p, beta=float(beta[k + 1]),
                                  se=float(se[k + 1]), n=len(J)))
    jd = pd.DataFrame(jrows)
    return dict(scan=scan, genes=G, family=fam, complex=cxd, joint=jd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--cohorts", nargs="*")
    a = ap.parse_args()
    cs = a.cohorts or cohorts()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_cohort, cs))
    for k, fn in (("scan", "scan.csv"), ("family", "family.csv"), ("complex", "complex.csv"), ("joint", "joint.csv")):
        pd.concat([r[k] for r in res], ignore_index=True).to_csv(OUT / fn, index=False)
    pd.concat([r["genes"] for r in res], ignore_index=True).to_parquet(OUT / "gene_resid.parquet")
    json.dump(dict(cohorts=cs, draws=RU.DRAWS), open(OUT / "_run.json", "w"), indent=1)
    print("done", len(cs))


if __name__ == "__main__":
    main()
