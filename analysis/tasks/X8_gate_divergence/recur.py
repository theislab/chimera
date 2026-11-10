"""X8 step (a): do gap calls recur across tissues, how tissue-specific are they, and how do they replicate within a tissue.

Thread: H1's eight-tissue primary cohorts. Per gene R = max(#tissues called up, #called down). Null: call labels (with delta)
permuted within pi x expression strata (5 x 10, H1's) independently in each cohort, N_PERM_REC draws; p = (1 + #{R_null >= R})/(N + 1),
BH over genes present in >= 3 tissues; recurrent if R >= R_MIN and q < 0.05. Tissue-specific: called in exactly one thread tissue,
|delta| < SPECIFIC_MAX_OTHER in every other tissue where present (>= 3 others), count against the same null.
Beside: pairwise cross-cohort Spearman of delta, z_free, z_can (all 26 cohorts, shared genes); one-way ICC over genes in all eight
thread tissues; within-tissue replication of calls in sibling cohorts (shared human data, not independent).

    PYTHONPATH=$PWD python -m analysis.tasks.X8_gate_divergence.recur
Output: results/analysis/X8/recurrence_genes.csv, recurrence_summary.csv, cross_cohort_rho.csv, icc.csv, replication.csv
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.H1_harmonise.recurrence import bh
from analysis.tasks.X8_gate_divergence import rules as RU

OUT = RU.OUT


def load(c: str) -> pd.DataFrame:
    return pd.read_parquet(OUT / "genes" / f"{c}.parquet")


def strata(g: pd.DataFrame) -> np.ndarray:
    e = pd.qcut(g.expr.rank(method="first"), 10, labels=False).to_numpy()
    p = pd.Series(pd.qcut(g.pi_raw.rank(method="average"), 5, labels=False, duplicates="drop")).fillna(0).to_numpy().astype(int)
    return e * 100 + p


def icc(M: np.ndarray) -> float:
    """One-way random-effects ICC(1), genes as groups, tissues as replicates."""
    n, k = M.shape
    gm = M.mean(1)
    msb = k * ((gm - M.mean()) ** 2).sum() / (n - 1)
    msw = ((M - gm[:, None]) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw))


def main() -> int:
    reg = RU.cohorts()
    cs = list(reg)
    T = {c: load(c) for c in cs}
    th = RU.thread()
    tissues = list(th)
    genes = sorted(set().union(*[set(T[th[t]].hsapiens_gene_id) for t in tissues]))
    gi = {g: i for i, g in enumerate(genes)}
    n = len(genes)
    present = np.zeros((n, len(tissues)), bool)
    CALL = np.zeros((n, len(tissues)), np.int8)
    DEL = np.full((n, len(tissues)), np.nan)
    per = []
    for j, t in enumerate(tissues):
        g = T[th[t]]
        ix = g.hsapiens_gene_id.map(gi).to_numpy()
        present[ix, j] = True
        CALL[ix, j] = g.call.to_numpy()
        DEL[ix, j] = g.delta.to_numpy()
        st = strata(g)
        per.append((ix, g.call.to_numpy().astype(np.int8), g.delta.to_numpy(), st))

    def stats(C, Dl):
        up, dn = (C > 0).sum(1), (C < 0).sum(1)
        R = np.maximum(up, dn)
        called1 = (C != 0).sum(1) == 1
        npres = present.sum(1)
        small_other = np.where(present & (C == 0), np.abs(np.nan_to_num(Dl, nan=0.0)) < RU.SPECIFIC_MAX_OTHER, True).all(1)
        spec = called1 & small_other & (npres >= 4)
        return up, dn, R, spec

    up, dn, R, spec = stats(CALL, DEL)
    rng = np.random.default_rng(RU.SEED)
    ge = np.zeros(n)
    nR3_null, nspec_null = [], []
    for _ in range(RU.N_PERM_REC):
        C = np.zeros_like(CALL)
        Dl = np.full_like(DEL, np.nan)
        for j, (ix, cl, dl, st) in enumerate(per):
            o = np.lexsort((rng.random(len(st)), st))  # a permutation within strata: sorted by stratum, random within
            base = np.argsort(st, kind="stable")
            perm = np.empty(len(st), int)
            perm[base] = o
            C[ix, j] = cl[perm]
            Dl[ix, j] = dl[perm]
        _, _, Rn, sn = stats(C, Dl)
        ge += Rn >= R
        nR3_null.append(int(((Rn >= RU.R_MIN) & (present.sum(1) >= 3)).sum()))
        nspec_null.append(int(sn.sum()))
    p = (1 + ge) / (RU.N_PERM_REC + 1)
    elig = present.sum(1) >= 3
    q = np.full(n, np.nan)
    q[elig] = bh(p[elig])
    sym = pd.concat([T[th[t]][["hsapiens_gene_id", "symbol"]] for t in tissues]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id").symbol
    out = pd.DataFrame(dict(hsapiens_gene_id=genes, symbol=sym.reindex(genes).to_numpy(), n_tissues=present.sum(1), n_up=up, n_down=dn, R=R, p_rec=p, q_rec=q,
                            direction=np.where(up > dn, "up", np.where(dn > up, "down", "mixed")), mean_delta=np.nanmean(DEL, 1), tissue_specific=spec))
    out["recurrent"] = (out.R >= RU.R_MIN) & (out.q_rec < 0.05)
    for j, t in enumerate(tissues):
        out[f"delta_{t}"] = DEL[:, j]
        out[f"call_{t}"] = np.where(present[:, j], CALL[:, j], np.nan)
    out.sort_values(["R", "p_rec"], ascending=[False, True]).to_csv(OUT / "recurrence_genes.csv", index=False)
    summ = dict(n_genes=n, n_eligible=int(elig.sum()), n_R_ge3=int(((R >= RU.R_MIN) & elig).sum()), null_R_ge3_mean=float(np.mean(nR3_null)),
                null_R_ge3_hi=float(np.percentile(nR3_null, 97.5)), p_R_ge3=float((1 + (np.array(nR3_null) >= ((R >= RU.R_MIN) & elig).sum()).sum()) / (RU.N_PERM_REC + 1)),
                n_recurrent=int(out.recurrent.sum()), n_R_ge2=int(((R >= 2) & elig).sum()),
                n_tissue_specific=int(spec.sum()), null_specific_mean=float(np.mean(nspec_null)), null_specific_hi=float(np.percentile(nspec_null, 97.5)),
                n_called_any=int(((CALL != 0).sum(1) > 0).sum()))
    pd.DataFrame([summ]).to_csv(OUT / "recurrence_summary.csv", index=False)
    print(summ)
    # cross-cohort correlations
    rows = []
    for a_ in range(len(cs)):
        for b_ in range(a_ + 1, len(cs)):
            A, B = T[cs[a_]].set_index("hsapiens_gene_id"), T[cs[b_]].set_index("hsapiens_gene_id")
            sh = A.index.intersection(B.index)
            if len(sh) < 200:
                continue
            r = dict(a=cs[a_], b=cs[b_], tissue_a=reg[cs[a_]]["tissue"], tissue_b=reg[cs[b_]]["tissue"], n=len(sh))
            r["same_tissue"] = r["tissue_a"] == r["tissue_b"]
            r["thread_pair"] = cs[a_] in th.values() and cs[b_] in th.values()
            for k in ("delta", "z_free", "z_can", "delta_res"):
                r[f"rho_{k}"] = float(spearmanr(A.loc[sh, k], B.loc[sh, k], nan_policy="omit")[0])
            rows.append(r)
    X = pd.DataFrame(rows)
    X.to_csv(OUT / "cross_cohort_rho.csv", index=False)
    print(X.groupby(["same_tissue"])[["rho_delta", "rho_z_free", "rho_z_can", "rho_delta_res"]].median().round(3))
    print(X[X.thread_pair][["rho_delta", "rho_z_free", "rho_z_can", "rho_delta_res"]].median().round(3))
    # ICC over genes in all eight
    full = present.all(1)
    ic = []
    for k in ("delta", "z_free", "z_can", "delta_res"):
        M = np.full((n, len(tissues)), np.nan)
        for j, t in enumerate(tissues):
            g = T[th[t]]
            M[g.hsapiens_gene_id.map(gi).to_numpy(), j] = g[k].to_numpy()
        Mf = M[full & np.isfinite(M).all(1)]
        Mf = (Mf - Mf.mean(0)) / Mf.std(0)
        ic.append(dict(readout=k, n_genes=len(Mf), icc=icc(Mf)))
    pd.DataFrame(ic).to_csv(OUT / "icc.csv", index=False)
    print(pd.DataFrame(ic))
    # within-tissue replication
    rep = []
    for tis, sib in RU.SIBLINGS.items():
        for a_ in sib:
            for b_ in sib:
                if a_ == b_ or a_ not in T or b_ not in T:
                    continue
                A, B = T[a_].set_index("hsapiens_gene_id"), T[b_].set_index("hsapiens_gene_id")
                ca = A[A.call != 0]
                ca = ca[ca.index.isin(B.index)]
                if not len(ca):
                    continue
                bb = B.loc[ca.index]
                rep.append(dict(tissue=tis, a=a_, b=b_, n_calls_a=len(ca), share_same_sign=float((np.sign(bb.delta) == np.sign(ca.delta)).mean()),
                                share_called_same=float((bb.call == ca.call).mean()), base_call_rate_b=float((B.call != 0).mean())))
    pd.DataFrame(rep).to_csv(OUT / "replication.csv", index=False)
    print(pd.DataFrame(rep).groupby("tissue")[["share_same_sign", "share_called_same", "base_call_rate_b"]].median().round(3))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
