"""Y5 F4 follow-up (amendment 9): why S_gsk profile similarity marks STRING partners on liver H+Mouse, and is it Chimera's.

Profiles on the same (species, label) groups, unmasked genes: S (healthy-seed mean), each healthy seed, the X4 no-model margin per group
(Sd), its pattern part (Sp) and level part (Sl), pseudobulk log CPM (PB), species-centred pseudobulk difference per type (Dpb).
Golds: STRING physical experiments >= 700 and >= 400, HuRI (yeast two-hybrid, not abundance-biased), Reactome, Hallmark.
Strata: 10 x 10 deciles of the two genes' species-balanced expression. All pairs, and pairs with neither gene in abundant machinery
(MACHINERY regex, fixed here). Jackknife over 10 gene folds.

Usage: python -m analysis.tasks.Y5_aspects.f4b <task index 0-7> ; python -m analysis.tasks.Y5_aspects.f4b collect
"""
from __future__ import annotations
import re
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, rankdata
from analysis.common import cohorts as C
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X4_where_differs.direct_margin import nb_ll_diff
from analysis.tasks.Y5_aspects import common as K, f4 as F4, rules as RU

OUT = K.OUT / "F4b"
MACHINERY = re.compile(r"^(RP[LS]\d|RPLP|MRP[LS]\d|NDUF|COX\d|ATP5|UQCR|SDH[ABCD]$|PSM[ABCDEFG]\d|POLR[123]|EIF[1-6]|MT-|SNRP|TOMM|TIMM)")
COH = ["liv_hmouse", "liv_4sp", "test_3sp", "test_hmouse"]


def direct_profiles(c, names):
    a = C.read_adata(c)
    X = sp.csr_matrix(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = a.obs["cell_ontology_class" if c.startswith("test") else "cell_type"].astype(str).to_numpy()
    L = np.asarray(X.sum(1)).ravel()
    G = X.shape[1]
    species = sorted(set(spc))
    groups = sorted({(s, t) for s, t in zip(spc, lab)})
    rate = {(s, t): (np.asarray(X[(spc == s) & (lab == t)].sum(0)).ravel() + .5) / (L[(spc == s) & (lab == t)].sum() + .5 * G) for s, t in groups}
    p = {s: (np.asarray(X[spc == s].sum(0)).ravel() + .5) / (L[spc == s].sum() + .5 * G) for s in species}
    rbar = {t: np.mean([rate[(s, t)] for s in species if (s, t) in rate], 0) for t in {t for _, t in groups}}
    num, den = np.zeros(G), np.zeros(G)
    for s, t in groups:
        m = np.flatnonzero((spc == s) & (lab == t))
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            mu = np.outer(L[mm], rate[(s, t)])
            xx = np.asarray(X[mm].todense())
            num += (((xx - mu) ** 2) - mu).sum(0)
            den += (mu ** 2).sum(0)
    th = 1 / np.clip(num / np.maximum(den, 1e-12), 1e-3, 1e3)
    Ls = {s: L[spc == s].sum() for s in species}
    cfac = {s: p[s] / (sum(L[(spc == a_) & (lab == t_)].sum() * rbar[t_] for (a_, t_) in groups if a_ == s) / Ls[s]) for s in species}
    Sd, Sp = [], []
    for s, t in names:
        m = np.flatnonzero((spc == s) & (lab == t))
        tot, pat = np.zeros(G), np.zeros(G)
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            xx = np.asarray(X[mm].todense())
            tot += nb_ll_diff(xx, np.outer(L[mm], rbar[t]), np.outer(L[mm], p[s]), th).sum(0)
            pat += nb_ll_diff(xx, np.outer(L[mm], rbar[t] * cfac[s]), np.outer(L[mm], p[s]), th).sum(0)
        Sd.append(tot / len(m))
        Sp.append(pat / len(m))
    Sd, Sp = np.column_stack(Sd), np.column_stack(Sp)
    return Sd, Sp, Sd - Sp


def seed_profiles(c, key):
    out = {}
    for f in K.forward_runs(c):
        with np.load(f, allow_pickle=False) as d:
            grp = d[f"{key}__groups"]
            keep = [i for i, (s, l, n) in enumerate(grp) if n >= RU.PROFILE_MIN_CELLS]
            out[f.stem.rsplit("_", 1)[1]] = np.column_stack([d[f"{key}__sum_meanz"][i] / grp[i][2] for i in keep])
    return out


def task(i):
    OUT.mkdir(parents=True, exist_ok=True)
    c, half = [(x, h) for x in COH for h in ("odd", "even")][i]
    key = "cell_ontology_class" if c.startswith("test") else "cell_type"
    S, PB, names, var = F4.profiles(c)
    Sd, Sp, Sl = direct_profiles(c, names)
    # species-centred pseudobulk difference
    Dpb = PB.copy()
    lb = np.array([l for _, l in names])
    for l in np.unique(lb):
        Dpb[:, lb == l] -= Dpb[:, lb == l].mean(1, keepdims=True)
    prof = dict(S=S, PB=PB, Sd=Sd, Sp=Sp, Sl=Sl, Dpb=Dpb)
    for s, M in seed_profiles(c, key).items():
        prof[f"S_{s}"] = M
    g = K.genes(c).set_index("var_name").reindex(var)
    m = (g.half.to_numpy() == half) & ~g.masked.astype(bool).to_numpy() & np.isfinite(g.log_cp10k_spmean.to_numpy())
    ids = g.hsapiens_gene_id.to_numpy()[m]
    u = ~pd.Series(ids).duplicated().to_numpy()
    idx = np.flatnonzero(m)[u]
    ids = ids[u]
    sym = g.symbol.astype(str).to_numpy()[idx]
    mach = np.array([bool(MACHINERY.match(x)) for x in sym])
    lev = g.log_cp10k_spmean.to_numpy()[idx]
    dec = np.digitize(lev, np.quantile(lev, np.linspace(.1, .9, 9)))
    iu = np.triu_indices(len(idx), 1)
    strat = np.minimum(dec[iu[0]], dec[iu[1]]) * 10 + np.maximum(dec[iu[0]], dec[iu[1]])
    Gm = F4.gold_mats(ids)
    from analysis.tasks.T6_common import functional as FN
    pos = {x: k for k, x in enumerate(ids)}
    for nm, ed in (("STRING400", FN.string_edges("9606", 400)), ("HuRI", FN.huri_edges())):
        a_, b_ = ed.a.map(pos), ed.b.map(pos)
        ok = a_.notna() & b_.notna()
        M = sp.csr_matrix((np.ones(ok.sum(), bool), (a_[ok].astype(int), b_[ok].astype(int))), shape=(len(ids), len(ids)))
        Gm[nm] = (M + M.T).astype(bool).tocsr()
    sims = {k: F4.zsim(v[idx])[0][iu] for k, v in prof.items()}
    fold = np.random.default_rng(K.useed(c, half)).integers(0, 10, len(idx))
    rows = []
    for gn, M in Gm.items():
        P = np.asarray(M[iu]).ravel().astype(bool)
        for filt, keep in (("all", np.ones(len(P), bool)), ("no_machinery", ~mach[iu[0]] & ~mach[iu[1]])):
            for k, s in sims.items():
                est = F4.stratified_auc(s[keep], P[keep], strat[keep])
                jk = []
                if not k.startswith("S_"):
                    for f in range(10):
                        kk = keep & (fold[iu[0]] != f) & (fold[iu[1]] != f)
                        jk.append(F4.stratified_auc(s[kk], P[kk], strat[kk]))
                se = float(np.sqrt(9 / 10 * np.sum((np.array(jk) - np.mean(jk)) ** 2))) if jk else np.nan
                rows.append(dict(cohort=c, half=half, gold=gn, filter=filt, profile=k, auc=est, se=se, n_pos=int(P[keep].sum()), n_genes=int(len(idx)), n_machinery=int(mach.sum())))
            print(c, half, gn, filt, flush=True)
        pm = P & (mach[iu[0]] | mach[iu[1]])
        rows.append(dict(cohort=c, half=half, gold=gn, filter="share_of_pos_with_machinery", profile="-", auc=float(pm.sum() / max(P.sum(), 1)), se=np.nan, n_pos=int(P.sum()), n_genes=int(len(idx)), n_machinery=int(mach.sum())))
    pd.DataFrame(rows).to_csv(OUT / f"auc_{i}.csv", index=False)


def collect():
    pd.concat([pd.read_csv(f) for f in sorted(OUT.glob("auc_*.csv"))], ignore_index=True).to_csv(OUT / "auc.csv", index=False)


if __name__ == "__main__":
    collect() if sys.argv[1] == "collect" else task(int(sys.argv[1]))
