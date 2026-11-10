"""G0 (set-level reliability across seeds) and F1 (pathway coherence), pre-specified analysis plan, section 7.

G0: Spearman between seeds of set means of per-seed D, Hallmark and Reactome, against the same
statistic on shuffled gene-to-set assignment and against gene-level Spearman.

F1, per cohort: index = mean over sets of n_s m_s^2 / Var(r), and median variance ratio, on the
pooled collection (Hallmark, Reactome, CollecTRI) and per library. Null: the readout permuted
within pi x log-mean strata, 10,000 draws, one set of permutations per cohort shared by readouts.

    python analysis/tasks/T6_3_pathway/run.py [--outdir results/analysis/T6_3] [--n-perm 10000]
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.common.outputs import git_sha
from analysis.tasks.T4_3_scoreboard import stats as ST
from analysis.tasks.T6_common import functional as FN

BAR = 0.70
N_SHUFFLE = 1000
G0_GATE = ["liv_hmouse", "liv_hmonkey"]
G0_BESIDE = ["liv_4sp", "test_3sp"]


def spearman_cols(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Spearman of each column of A with the same column of B (sets x draws)."""
    Ra = rankdata(A, axis=0)
    Rb = rankdata(B, axis=0)
    Ra = Ra - Ra.mean(0)
    Rb = Rb - Rb.mean(0)
    return (Ra * Rb).sum(0) / np.sqrt((Ra ** 2).sum(0) * (Rb ** 2).sum(0))


def g0_one(M: np.ndarray, members, rng, tag: str) -> dict:
    """M: genes x seeds. members: sets x genes (csr)."""
    seeds = range(M.shape[1])
    n = np.asarray(members.sum(1)).ravel()
    means = np.asarray(members @ M) / n[:, None]
    G = M.shape[0]
    set_rho, gene_rho, shuf_rho = [], [], []
    perms = np.stack([rng.permutation(G) for _ in range(N_SHUFFLE)], 1)
    for i, j in itertools.combinations(seeds, 2):
        set_rho.append(spearmanr(means[:, i], means[:, j])[0])
        gene_rho.append(spearmanr(M[:, i], M[:, j])[0])
        Ma, Mb = M[perms, i], M[perms, j]  # genes x draws
        sa = np.asarray(members @ Ma) / n[:, None]
        sb = np.asarray(members @ Mb) / n[:, None]
        shuf_rho.append(float(np.median(spearman_cols(sa, sb))))
    return dict(scale=tag, n_seeds=M.shape[1], n_pairs=len(set_rho), n_sets=len(n),
                set_rho_median=float(np.median(set_rho)), set_rho_min=float(np.min(set_rho)),
                set_rho_max=float(np.max(set_rho)), gene_rho_median=float(np.median(gene_rho)),
                shuffled_rho_median=float(np.median(shuf_rho)))


def run_g0(out: Path) -> pd.DataFrame:
    rows = []
    for cohort in G0_GATE + G0_BESIDE:
        tab, E = FN.gene_table(cohort)
        Z = FN.designs(tab, E)
        M = FN.seed_matrix(cohort, tab, "D")
        seeds = list(M.columns)
        L = FN.libraries(tab)
        R1 = np.column_stack([ST.resid(FN.inverse_normal(M[s].to_numpy()), Z["Z1"]) for s in seeds])
        R2 = np.column_stack([ST.resid(FN.inverse_normal(M[s].to_numpy()), Z["Z2"]) for s in seeds])
        for lib in ("Hallmark", "Reactome"):
            mem, _ = FN.membership(tab, L[lib])
            rng = np.random.default_rng(20261102)
            for tag, X, sub in (("D", M.to_numpy(), "all"), ("r1", R1, "all"), ("r2", R2, "all")):
                r = g0_one(X, mem, rng, tag)
                rows.append(dict(cohort=cohort, library=lib, seeds_used=sub, gate=cohort in G0_GATE, **r))
            if cohort == "liv_hmouse":
                keep = [seeds.index(s) for s in FN.INTEGRATED_SEEDS]
                for tag, X in (("D", M.to_numpy()), ("r2", R2)):
                    r = g0_one(X[:, keep], mem, rng, tag)
                    rows.append(dict(cohort=cohort, library=lib, seeds_used="integrated", gate=False, **r))
            print(cohort, lib, "done", flush=True)
    df = pd.DataFrame(rows)
    df["passes"] = (df.set_rho_median >= BAR) & (df.set_rho_median > df.gene_rho_median)
    K.write_csv(df, out / "G0_set_reliability.csv")
    return df


def perm_stats(R: np.ndarray, mem, lab: np.ndarray, n_perm: int, seed: int, chunk: int = 250):
    """R: genes x readouts. Returns per-readout arrays of set means and set variances over draws:
    index per library handled by the caller from (means, vars) float32 sets x draws."""
    rng = np.random.default_rng(seed)
    groups = [np.flatnonzero(lab == s) for s in np.unique(lab)]
    G, Q = R.shape
    n = np.asarray(mem.sum(1)).ravel()
    K_ = mem.shape[0]
    Ms = np.empty((Q, K_, n_perm), dtype=np.float32)
    Vs = np.empty((Q, K_, n_perm), dtype=np.float32)
    done = 0
    while done < n_perm:
        c = min(chunk, n_perm - done)
        P = np.empty((c, G), dtype=np.int64)
        for idx in groups:
            P[:, idx] = idx[np.argsort(rng.random((c, len(idx))), axis=1)]
        for q in range(Q):
            X = R[:, q][P].T  # genes x c
            m1 = np.asarray(mem @ X) / n[:, None]
            m2 = np.asarray(mem @ (X ** 2)) / n[:, None]
            Ms[q, :, done:done + c] = m1
            Vs[q, :, done:done + c] = (m2 - m1 ** 2) * (n[:, None] / (n[:, None] - 1))
        done += c
    return Ms, Vs


def summarise(r: np.ndarray, mem, sel: np.ndarray, Mq: np.ndarray, Vq: np.ndarray) -> dict:
    n = np.asarray(mem.sum(1)).ravel()[sel]
    m = sel.astype(bool)
    v_all = r.var()
    obs_m = (np.asarray(mem @ r) / np.asarray(mem.sum(1)).ravel())[sel]
    obs_m2 = (np.asarray(mem @ (r ** 2)) / np.asarray(mem.sum(1)).ravel())[sel]
    obs_v = (obs_m2 - obs_m ** 2) * (n / (n - 1))
    obs_idx = float((n * obs_m ** 2).mean() / v_all)
    null_idx = (n[:, None] * Mq[sel].astype(float) ** 2).mean(0) / v_all
    v0 = Vq[sel].astype(float).mean(1)
    obs_vr = float(np.median(obs_v / v0))
    null_vr = np.median(Vq[sel].astype(float) / v0[:, None], axis=0)
    B = len(null_idx)
    return dict(n_sets=int(len(n)), index=obs_idx, null_mean=float(null_idx.mean()),
                null_q025=float(np.quantile(null_idx, 0.025)), null_q975=float(np.quantile(null_idx, 0.975)),
                ses=float((obs_idx - null_idx.mean()) / null_idx.std()),
                p_index=float((1 + (null_idx >= obs_idx).sum()) / (B + 1)),
                median_VR=obs_vr, null_VR_q025=float(np.quantile(null_vr, 0.025)),
                null_VR_q975=float(np.quantile(null_vr, 0.975)),
                p_VR_below=float((1 + (null_vr <= obs_vr).sum()) / (B + 1)))


def run_f1(out: Path, n_perm: int) -> pd.DataFrame:
    rows, sets_out = [], []
    for ci, cohort in enumerate(FN.PRIMARY + FN.SECONDARY):
        tab, E = FN.gene_table(cohort)
        Z = FN.designs(tab, E)
        extra = {"rI": FN.integrated_D(tab)} if cohort == "liv_hmouse" else None
        Rd = FN.residuals(tab, Z, extra)
        L = FN.libraries(tab)
        pooled = {f"{k}|{t}": v for k, d in L.items() for t, v in d.items()}
        mem, names = FN.membership(tab, pooled)
        libs = np.array([n.split("|")[0] for n in names])
        lab = ST.strata(tab["prior_raw"].to_numpy(), tab["log_cp10k_spmean"].to_numpy())
        lab_e = ST.deciles(tab["log_cp10k_spmean"].to_numpy())
        cols = list(Rd.columns)
        R = Rd[cols].to_numpy()
        Ms, Vs = perm_stats(R, mem, lab, n_perm, 20261102 + ci)
        for q, c in enumerate(cols):
            for lib in ("pooled", "Hallmark", "Reactome", "CollecTRI"):
                sel = np.ones(len(names), bool) if lib == "pooled" else libs == lib
                rows.append(dict(cohort=cohort, n_genes=len(tab), library=lib, readout=c, null="pi_x_mean",
                                 **summarise(R[:, q], mem, np.flatnonzero(sel), Ms[q], Vs[q])))
            print(cohort, c, rows[-4]["index"], rows[-4]["p_index"], flush=True)
        # per-set table, descriptive
        n = np.asarray(mem.sum(1)).ravel()
        for q, c in enumerate(cols):
            if c not in ("r1", "r2", "S2"):
                continue
            m = np.asarray(mem @ R[:, q]) / n
            mq = Ms[q].astype(float)
            p = (1 + (np.abs(mq - mq.mean(1, keepdims=True)) >= np.abs(m - mq.mean(1))[:, None]).sum(1)) / (n_perm + 1)
            df = pd.DataFrame(dict(cohort=cohort, readout=c, library=libs, set=[x.split("|", 1)[1] for x in names],
                                   n=n.astype(int), mean=m, null_mean=mq.mean(1), z=(m - mq.mean(1)) / mq.std(1), p=p))
            df["q_within_library"] = np.nan
            for lib in np.unique(libs):
                s = df.library == lib
                df.loc[s, "q_within_library"] = ST.bh(df.loc[s, "p"].tolist())
            sets_out.append(df)
        # prior comparator also under expression-only strata: pi strata absorb pi itself
        Ms2, Vs2 = perm_stats(R[:, [cols.index("PI")]], mem, lab_e, n_perm, 20261202 + ci)
        rows.append(dict(cohort=cohort, n_genes=len(tab), library="pooled", readout="PI", null="mean_only",
                         **summarise(R[:, cols.index("PI")], mem, np.arange(len(names)), Ms2[0], Vs2[0])))
        del Ms, Vs, Ms2, Vs2
    df = pd.DataFrame(rows)
    K.write_csv(df, out / "F1_coherence.csv")
    K.write_csv(pd.concat(sets_out, ignore_index=True), out / "F1_sets.csv")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=gene_ids.REPO_ROOT / "results/analysis/T6_3")
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--only", choices=["G0", "F1"], default=None)
    a = ap.parse_args()
    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.only != "F1":
        g = run_g0(a.outdir)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(g.round(3).to_string(index=False), flush=True)
    if a.only != "G0":
        f = run_f1(a.outdir, a.n_perm)
        with pd.option_context("display.width", 220, "display.max_columns", 30):
            print(f[f.library == "pooled"].round(4).to_string(index=False), flush=True)
    (a.outdir / "_provenance.json").write_text(json.dumps(dict(git_sha=git_sha(), n_perm=a.n_perm)))


if __name__ == "__main__":
    main()
