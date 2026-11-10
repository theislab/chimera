"""Step 2: Q1-Q3 statistics on both read-outs (definitions in rules.py). EXPLORATORY.

Writes results/analysis/E4/{tables/cohorts.parquet, q1_concordance.csv, q1_per_cohort.csv, q1_not.csv, q23_sets.csv,
q23_contrast.csv, q23_partial.csv, q23_callshare.csv}.

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.E4_translational.q1_q3_stats [--quick]
"""

from __future__ import annotations

import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.api as sm

from analysis.common import controls as K
from analysis.tasks.E1_atlas.genes import calls
from analysis.tasks.E4_translational import rules as RU
from analysis.tasks.H1_harmonise import rules as H1R
from analysis.tasks.H1_harmonise.recurrence import bh, inv_norm_rank, perm_scores, strata

warnings.filterwarnings("ignore")
READOUTS = ["D_can", "D_free", "S", "B1"]
COVS = ["pi_raw", "expr", "det_human", "det_partner_min", "tau", "log_pubmed_h"]
QUICK = "--quick" in sys.argv
TAB = RU.OUT / "tables"


def all_cohorts() -> dict:
    out = {t: c for t, c in RU.HM_THREAD.items()}
    roles = {c: ("thread", t) for t, c in out.items()}
    for t, c in RU.HM_EXTRA.items():
        roles[c] = ("hm_extra", t)
    for t, cs in RU.PRIMATE.items():
        for c in cs:
            roles[c] = ("primate", t)
    return roles


def cohort_tables() -> pd.DataFrame:
    f = TAB / "cohorts.parquet"
    if f.exists():
        return pd.read_parquet(f)
    TAB.mkdir(parents=True, exist_ok=True)
    g = pd.read_parquet(RU.OUT / "annot" / "genes.parquet")
    parts = []
    for c, (role, t) in all_cohorts().items():
        x = calls(c)
        x["role"], x["tissue_e4"] = role, t
        x = x.join(g[["pubmed_h", "pubmed_m", "mouse_symbol", "mouse_1to1_agree"]], on="hsapiens_gene_id")
        parts.append(x)
    T = pd.concat(parts, ignore_index=True)
    T["log_pubmed_h"] = np.log1p(T.pubmed_h.fillna(0))
    T["log_pubmed_m"] = np.log1p(T.pubmed_m.fillna(0))
    T.to_parquet(f)
    return T


# ---------------------------------------------------------------- Q2/Q3 set statistics
def set_job(args):
    c, readout, with_tau, keys, mem, n_perm, keep_null = args
    T = pd.read_parquet(TAB / "cohorts.parquet", filters=[("cohort", "==", c)])
    u = T.dropna(subset=[readout, "expr", "pi_raw"]).copy()
    st = strata(u, True, "pi_raw", with_tau=with_tau)
    v = inv_norm_rank(u[readout].to_numpy(float)).astype(np.float32)
    ix = pd.Index(u.hsapiens_gene_id)
    si = {k: i for i, k in enumerate(keys)}
    mm = mem[mem.hsapiens_gene_id.isin(ix) & mem.set.isin(si)]
    M = sp.csr_matrix((np.ones(len(mm), np.float32), (mm.set.map(si).to_numpy(), ix.get_indexer(mm.hsapiens_gene_id))), shape=(len(keys), len(ix)))
    sizes = np.asarray(M.sum(1)).ravel()
    rng = np.random.default_rng([RU.SEED, sum(map(ord, c)) * 1000 + sum(map(ord, readout)) + 7 * with_tau])
    null = perm_scores(v, st, M, n_perm, rng)
    obs = np.asarray(M @ v).ravel() / np.maximum(sizes, 1)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    z = (obs - mu) / sd
    p = (1 + (np.abs(null - mu) >= np.abs(obs - mu)[None, :]).sum(0)) / (n_perm + 1)
    res = pd.DataFrame(dict(cohort=c, readout=readout, strata="pi_expr_tau" if with_tau else "pi_expr", set=keys, n=sizes.astype(int),
                            effect=obs - mu, z=z, p=np.where(sizes >= 5, p, np.nan)))
    zn = ((null - mu) / sd).astype(np.float32) if keep_null else None
    return res, zn


def partial_job(args):
    """OLS INT(readout) ~ member + natural-spline covariates; member coefficient, gene bootstrap CI."""
    c, readout, keys, mem = args
    T = pd.read_parquet(TAB / "cohorts.parquet", filters=[("cohort", "==", c)])
    u = T.dropna(subset=[readout] + COVS).reset_index(drop=True)
    y = inv_norm_rank(u[readout].to_numpy(float))
    Z = np.c_[np.ones(len(u)), *[K.ns_basis(u[k].rank().to_numpy(), 4) for k in COVS]]
    rng = np.random.default_rng([RU.SEED, sum(map(ord, c)), sum(map(ord, readout))])
    B = rng.integers(0, len(u), size=(RU.N_BOOT if not QUICK else 100, len(u)))
    ids = set(u.hsapiens_gene_id)
    ks, cols = [], []
    for k in keys:
        m = u.hsapiens_gene_id.isin(set(mem.hsapiens_gene_id[mem.set == k]) & ids).to_numpy().astype(float)
        if m.sum() >= 5:
            ks.append(k)
            cols.append(m)
    if not ks:
        return pd.DataFrame()
    Mm = np.stack(cols, 1)

    def fwl(Zb, yb, Mb):  # member coefficient of each set, Frisch-Waugh-Lovell on a shared covariate basis
        Q, _ = np.linalg.qr(Zb)
        yr = yb - Q @ (Q.T @ yb)
        Mr = Mb - Q @ (Q.T @ Mb)
        den = (Mr ** 2).sum(0)
        return np.where(den > 1e-9, (Mr * yr[:, None]).sum(0) / np.where(den > 1e-9, den, 1), np.nan)

    beta = fwl(Z, y, Mm)
    bs = np.stack([fwl(Z[b], y[b], Mm[b]) for b in B])
    return pd.DataFrame(dict(cohort=c, readout=readout, set=ks, n=Mm.sum(0).astype(int), beta=beta,
                             lo=np.nanquantile(bs, 0.025, axis=0), hi=np.nanquantile(bs, 0.975, axis=0)))


def q23(T: pd.DataFrame, sets: pd.DataFrame, pool) -> None:
    keys = sorted(sets.set.unique())
    mem = sets[["set", "hsapiens_gene_id"]].drop_duplicates()
    roles = all_cohorts()
    jobs = []
    for c, (role, t) in roles.items():
        n_perm = (RU.N_PERM_THREAD if role == "thread" else RU.N_PERM_OTHER) if not QUICK else 200
        for r in READOUTS:
            for tau in (False, True):
                jobs.append((c, r, tau, keys, mem, n_perm, (role in ("thread", "hm_extra")) and not tau))
    t0 = time.time()
    res = list(pool.map(set_job, jobs))
    print(f"set jobs {len(jobs)} in {time.time() - t0:.0f}s", flush=True)
    Z = pd.concat([r[0] for r in res], ignore_index=True)
    Z["q"] = np.nan
    for (r, s), x in Z.groupby(["readout", "strata"]).groups.items():
        Z.loc[x, "q"] = bh(Z.loc[x, "p"].to_numpy())
    Z["role"] = Z.cohort.map(lambda c: roles[c][0])
    Z["tissue"] = Z.cohort.map(lambda c: roles[c][1])
    Z.to_csv(RU.OUT / "q23_sets.csv", index=False)
    # own versus other: own z in own thread cohort minus mean z of the same set over the other thread tissues; null from the same draws
    nulls = {(j[0], j[1]): rz[1] for j, rz in zip(jobs, res) if rz[1] is not None}
    thread = [RU.HM_THREAD[t] for t in RU.TISSUES]
    ki = {k: i for i, k in enumerate(keys)}
    rows = []
    own_sets = [k for k in keys if k.startswith("own:") or k.startswith("target:ind_") or k.startswith("do:")]
    for k in own_sets:
        t = k.split(":", 1)[1].replace("ind_", "")
        if k.startswith("target:"):
            t = k.split("ind_", 1)[1]
        c_own = RU.HM_THREAD.get(t) or RU.HM_EXTRA.get(t)
        if c_own is None:
            continue
        others = [c for c in thread if c != c_own]
        for r in READOUTS:
            zz = Z[(Z.readout == r) & (Z.strata == "pi_expr") & (Z.set == k)].set_index("cohort").z
            if c_own not in zz or not np.isfinite(zz.get(c_own, np.nan)):
                continue
            ov = [zz.get(c, np.nan) for c in others]
            obs = zz[c_own] - np.nanmean(ov)
            n_own = nulls[(c_own, r)][:, ki[k]]
            n_oth = np.nanmean(np.stack([nulls[(c, r)][:len(n_own), ki[k]] for c in others]), 0)
            n_ = min(len(n_own), len(n_oth))
            nc = n_own[:n_] - n_oth[:n_]
            nc = nc[np.isfinite(nc)]
            p = (1 + (np.abs(nc) >= abs(obs)).sum()) / (len(nc) + 1)
            rank = 1 + int(sum(1 for v in ov if np.isfinite(v) and v > zz[c_own]))
            rows.append(dict(set=k, tissue=t, readout=r, z_own=zz[c_own], z_other_mean=np.nanmean(ov), contrast=obs, p=p, rank_own=rank,
                             n_tissues=1 + sum(np.isfinite(ov))))
    C = pd.DataFrame(rows)
    C["q"] = np.nan
    for r, x in C.groupby("readout").groups.items():
        C.loc[x, "q"] = bh(C.loc[x, "p"].to_numpy())
    C.to_csv(RU.OUT / "q23_contrast.csv", index=False)
    # partial effects with PubMed, own-tissue cohort sets only (+ approved/clinical everywhere), HM thread and extras, primates for own sets
    pj = []
    for c, (role, t) in roles.items():
        ks = [k for k in keys if k in ("target:approved", "target:clinical_only") or k.endswith(":" + t) or k == f"target:ind_{t}"
              or (k.startswith("panel:") and k in set(sets.set[sets.tissue == t]))]
        for r in READOUTS:
            pj.append((c, r, ks, mem))
    t0 = time.time()
    P = pd.concat(list(pool.map(partial_job, pj)), ignore_index=True)
    print(f"partial jobs {len(pj)} in {time.time() - t0:.0f}s", flush=True)
    P["role"] = P.cohort.map(lambda c: roles[c][0])
    P["tissue"] = P.cohort.map(lambda c: roles[c][1])
    P.to_csv(RU.OUT / "q23_partial.csv", index=False)
    # call shares against stratum-matched expectation (robust calls, both gates)
    rows = []
    for c, (role, t) in roles.items():
        u = T[T.cohort == c]
        rate_dn = u.groupby("stratum").call.apply(lambda s: (s == -1).mean())
        rate_up = u.groupby("stratum").call.apply(lambda s: (s == 1).mean())
        for k in keys:
            m = u[u.hsapiens_gene_id.isin(set(mem.hsapiens_gene_id[mem.set == k]))]
            if len(m) < 5:
                continue
            rows.append(dict(cohort=c, role=role, tissue=t, set=k, n=len(m), below=int((m.call == -1).sum()), above=int((m.call == 1).sum()),
                             exp_below=float(m.stratum.map(rate_dn).sum()), exp_above=float(m.stratum.map(rate_up).sum())))
    pd.DataFrame(rows).to_csv(RU.OUT / "q23_callshare.csv", index=False)


# ---------------------------------------------------------------- Q1 concordance
def logit_fit(df: pd.DataFrame, x: str, covs: list[str], fe: bool):
    X = df[[x] + covs].copy()
    if fe and df.cohort.nunique() > 1:
        X = X.join(pd.get_dummies(df.cohort, drop_first=True, dtype=float))
    X = sm.add_constant(X.astype(float), has_constant="add")
    try:
        m = sm.Logit(df.Y.astype(float).to_numpy(), X.to_numpy()).fit(disp=0, maxiter=200)
        return float(m.params[1]), float(m.bse[1])
    except Exception:
        return np.nan, np.nan


def q1_frame(T: pd.DataFrame, outcome: str, organ: bool) -> pd.DataFrame:
    dg = pd.read_parquet(RU.OUT / "annot" / "disease_genes.parquet")
    ip = pd.read_parquet(RU.OUT / "annot" / "impc.parquet")
    parts = []
    for t, c in {**RU.HM_THREAD, "skeletal muscle": "muscle_ageing"}.items():
        u = T[(T.cohort == c)].copy()
        for k in READOUTS + ["pi_raw"]:
            ok = u[k].notna()
            u.loc[ok, "INT_" + k] = inv_norm_rank(u.loc[ok, k].to_numpy(float))
        for k in ["expr", "det_human", "det_partner_min", "tau", "log_pubmed_h", "log_pubmed_m"]:
            u["C_" + k] = inv_norm_rank(u[k].fillna(u[k].median()).to_numpy(float))
        if outcome == "Y_impc":
            x = u.merge(ip.rename(columns={"Y_impc": "Y"}), on="hsapiens_gene_id")
            x["C_ndis"] = np.log(x.n_mendelian)
        else:
            d = dg[dg.mouse_tested]
            if organ:
                d = d[d[f"organ_{t}"]]
            x = u.merge(d.rename(columns={outcome: "Y"}), on="hsapiens_gene_id")
            x["C_ndis"] = np.log(x.n_disease)
        x["C_pi"] = x["INT_pi_raw"]
        x["tissue_q1"] = t
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


COVS_Q1 = ["C_pi", "C_expr", "C_det_human", "C_det_partner_min", "C_tau", "C_log_pubmed_h", "C_log_pubmed_m", "C_ndis"]
Q1_SETS = [("Y_mgi", False), ("Y_mgi", True), ("Y_mgi_single", False), ("Y_mgi_single", True), ("Y_impc", False)]


def q1_job(args):
    """Pooled (eight-tissue thread, cohort fixed effects) log OR per SD, gene-cluster bootstrap; per-cohort Wald beside."""
    outcome, organ, r = args
    T = pd.read_parquet(TAB / "cohorts.parquet")
    Fm = q1_frame(T, outcome, organ)
    F = Fm[Fm.cohort != "muscle_ageing"]  # pooled = eight-tissue thread; muscle per cohort only
    x = "INT_" + r
    covs = [c for c in COVS_Q1 if not (r == "pi_raw" and c == "C_pi")]
    G = F.dropna(subset=[x]).reset_index(drop=True)
    b_adj, se_adj = logit_fit(G, x, covs, True)
    b_raw, _ = logit_fit(G, x, [], True)
    nb = (RU.N_BOOT if not QUICK else 50) if not outcome.endswith("_single") else 0  # single-marker sensitivity: point estimate only
    rng = np.random.default_rng([RU.SEED, sum(map(ord, outcome)), int(organ), sum(map(ord, r))])
    codes, uniq = pd.factorize(G.hsapiens_gene_id)
    members = pd.Series(np.arange(len(G))).groupby(codes).apply(np.array).to_list()
    bs = []
    for _ in range(nb):
        pick = rng.integers(0, len(uniq), len(uniq))
        bi = np.concatenate([members[j] for j in pick])
        bs.append(logit_fit(G.iloc[bi].reset_index(drop=True), x, covs, True)[0])
    bs = np.array([v for v in bs if np.isfinite(v)])
    lo, hi = (np.quantile(bs, 0.025), np.quantile(bs, 0.975)) if len(bs) else (b_adj - 1.96 * se_adj, b_adj + 1.96 * se_adj)
    row = dict(outcome=outcome, organ_matched=organ, readout=r, n=len(G), n_genes=G.hsapiens_gene_id.nunique(), n_pos=int(G.drop_duplicates("hsapiens_gene_id").Y.sum()),
               logOR_raw=b_raw, logOR=b_adj, se=se_adj, lo=lo, hi=hi, ci="cluster bootstrap" if len(bs) else "Wald",
               OR=np.exp(b_adj), OR_lo=np.exp(lo), OR_hi=np.exp(hi))
    per = []
    for c, Gc in Fm.dropna(subset=[x]).groupby("cohort"):
        if Gc.Y.sum() < 5 or (1 - Gc.Y).sum() < 5:
            continue
        b, se = logit_fit(Gc, x, covs, False)
        br, _ = logit_fit(Gc, x, [], False)
        per.append(dict(outcome=outcome, organ_matched=organ, readout=r, cohort=c, tissue=Gc.tissue_q1.iloc[0], n=len(Gc), n_pos=int(Gc.Y.sum()),
                        logOR_raw=br, logOR=b, lo=b - 1.96 * se, hi=b + 1.96 * se))
    return row, per


def q1(T: pd.DataFrame) -> None:
    covs_all = COVS_Q1
    jobs = [(o, org, r) for o, org in Q1_SETS for r in READOUTS + ["pi_raw"]]
    with ProcessPoolExecutor(16) as pool:
        res = list(pool.map(q1_job, jobs))
    rows = [r for r, _ in res]
    per = [p for _, ps in res for p in ps]
    for outcome, organ in Q1_SETS:
        F = q1_frame(T, outcome, organ)
        F = F[F.cohort != "muscle_ageing"]
    # covariate weights (pooled, adjusted model with D_free)
        G = F.dropna(subset=["INT_D_free"])
        X = sm.add_constant(G[["INT_D_free"] + covs_all].astype(float).join(pd.get_dummies(G.cohort, drop_first=True, dtype=float)))
        m = sm.Logit(G.Y.astype(float), X).fit(disp=0)
        for k in ["INT_D_free"] + covs_all:
            rows.append(dict(outcome=outcome, organ_matched=organ, readout="covariate:" + k, n=len(G), logOR=m.params[k], se=m.bse[k]))
    pd.DataFrame(rows).to_csv(RU.OUT / "q1_concordance.csv", index=False)
    pd.DataFrame(per).to_csv(RU.OUT / "q1_per_cohort.csv", index=False)
    # explicit NOT models versus curated models (genes with either), raw and adjusted
    F = q1_frame(T, "Y_mgi", False)
    dg = pd.read_parquet(RU.OUT / "annot" / "disease_genes.parquet")
    nots = set(dg.hsapiens_gene_id[dg.Y_not])
    G = F[F.Y | F.hsapiens_gene_id.isin(nots)].copy()
    G["Y"] = G.Y.astype(int)
    out = []
    for r in READOUTS + ["pi_raw"]:
        x = "INT_" + r
        Gx = G.dropna(subset=[x])
        b, se = logit_fit(Gx, x, ["C_expr", "C_tau", "C_log_pubmed_h", "C_log_pubmed_m"] + ([] if r == "pi_raw" else ["C_pi"]), True)
        out.append(dict(readout=r, n=len(Gx), n_model=int(Gx.Y.sum()), n_not=int((1 - Gx.Y).sum()), logOR=b, lo=b - 1.96 * se, hi=b + 1.96 * se,
                        mean_model=Gx[x][Gx.Y == 1].mean(), mean_not=Gx[x][Gx.Y == 0].mean()))
    pd.DataFrame(out).to_csv(RU.OUT / "q1_not.csv", index=False)


def main() -> int:
    RU.OUT.mkdir(parents=True, exist_ok=True)
    T = cohort_tables()
    print(T.groupby("cohort").size().to_string(), flush=True)
    sets = pd.read_parquet(RU.OUT / "annot" / "sets.parquet")
    steps = [a for a in sys.argv[1:] if not a.startswith("--")] or ["q1", "q23"]
    if "q1" in steps:
        q1(T)
    if "q23" in steps:
        with ProcessPoolExecutor(16) as pool:
            q23(T, sets, pool)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
