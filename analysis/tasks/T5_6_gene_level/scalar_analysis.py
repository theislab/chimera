"""T5.6 analysis of the tool scalars (chimera_env), after scalars.py. Fixed before any result was read.

D_3 = mean of alpha - prior over seeds 42-44, centred on the cohort median (signs within a cohort
only). Covariates C3 = spline of rank pi_raw, rank log mean, rank detection (the M4 rule, 05 section 5);
C3 + rank log CDS length is reported beside it (the section 3 set C).

Reliability. For a scalar over seeds 42-44: pairwise Spearman (median) and r = mean pairwise
correlation of rank residuals on C3, stepped up to the three-seed mean, rel = 3r / (1 + 2r). Same for
D_3 on the same genes. A4: rel below 0.5 is labelled unstable. "Seed reliability measures
determinism": it says nothing about validity (CHIMERA_MECHANISM.md).

L1. Spearman of each scalar with alpha (mean of three seeds), D_3 and pi_raw; headline partial
Spearman with D_3 given C3. CI: ESM-module block bootstrap, 10,000 draws (05 section 3). p: two-sided,
10,000 permutations of residuals within pi decile x expression decile; BH over the whole grid. Class on
the observed |rho_p|: concordant >= 0.3, orthogonal < 0.1, otherwise partial; disattenuated by
sqrt(rel_scalar rel_D3) in brackets, never used for the class.

L2. Per seed, per arm. Modules: SATURN macrogenes (weight >= 0.5, else top 10 per species); CAMEX
Leiden modules at resolution 1.0. Weighted-mean Delta-alpha, mixed-sign index 1 - |n+ - n-| / (n+ + n-)
among members with |D_3| > 0.05 (the T2.5 tail floor) in modules with at least 5 such members. Hard
module: largest mean weight (SATURN), human member's module (CAMEX). eta^2 of D_3 by hard module.
Null: D_3 permuted within pi decile x expression decile, 10,000 draws. p_coherent = P(null eta^2 >=
observed), p_mixed = P(null mixed-sign >= observed).

Native arm, descriptive. SATURN: per species and shared type, macrogene Wilcoxon test against the
other cells of the species on the macrogene values (BH < 0.05, score > 0); conserved macrogene: up in
both species for the type; divergent: up in one; a macrogene in both unions over types leaves both
lists. Genes: top 10 by weight per macrogene and species. Curated sets outside 15-500 genes in the
universe are not tested by the GSEA (T7.9 machinery).
Fisher ORA of the union lists on the curated sets of T4.3 (C1, C2, V1, V2, V3) and MSigDB Hallmark
2020, background the arm's genes (human symbols), BH per list. CAMEX: Fisher ORA of each module at
resolution 1.0 (10 or more genes), BH over modules x sets; a set is recovered if any module passes.
Delta-alpha: preranked weighted-KS GSEA of D_3 (T7.9 machinery, 10,000 gene-set permutations), BH.
A tool recovers a set if q < 0.05 in at least 2 of 3 seeds.

Primary scalars for T4.3: results/analysis/T5_6/tools_primary.csv (S_deploy_native, S_deploy_1to1,
C_cos). TOOL-LOSS is read on these three columns (both SATURN arms, since the plan reports both and
the modifier can only reduce a claim); secondary scalars never enter it.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import fisher_exact, spearmanr

from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.T5_6_gene_level import scalars as SC
from analysis.tasks.T7_9_islet import hd4 as G

RES = gene_ids.REPO_ROOT / "results/analysis"
OUT = RES / "T5_6"
C3 = ["prior_raw", "log_cp10k_spmean", "detection_spmean"]
C4 = C3 + ["log_cds_length"]
SEEDS = [42, 43, 44]
ARMS = {"saturn_native": SC.SATURN_SCALARS, "saturn_1to1": SC.SATURN_SCALARS, "camex": SC.CAMEX_SCALARS}
PRIMARY = {"saturn_native": "S_deploy", "saturn_1to1": "S_deploy", "camex": "C_cos"}
DRAWS = 10_000
PERMS = 10_000
BASE_SEED = 20261102
TAIL = 0.05
MIN_TAIL = 5
SET_NAMES = ["C1", "C2", "V1", "V2", "V3"]


def useed(s: str) -> int:
    return BASE_SEED + sum((i + 1) * ord(c) for i, c in enumerate(s)) % 1_000_003


def reliability(X: np.ndarray, Z: np.ndarray) -> dict:
    prs = [spearmanr(X[:, a], X[:, b])[0] for a, b in itertools.combinations(range(X.shape[1]), 2)]
    E = np.column_stack([S.resid(S.midrank(X[:, j]), Z) for j in range(X.shape[1])])
    rs = [np.corrcoef(E[:, a], E[:, b])[0, 1] for a, b in itertools.combinations(range(X.shape[1]), 2)]
    r = float(np.nanmean(rs))
    return dict(pair_spearman_median=float(np.nanmedian(prs)), r_resid=r, rel=3 * r / (1 + 2 * r) if (1 + 2 * r) > 0 else np.nan)


def rho_block(T: pd.DataFrame, x: str, cols: list[str], tag: str) -> dict:
    y = T["D_3"].to_numpy(float)
    R = np.column_stack([S.midrank(y), S.midrank(T[x].to_numpy(float))])
    Z = S.design(T, cols)
    est = float(S.rho_p_matrix(R, Z)[1])
    rng = np.random.default_rng(useed(tag + x))
    B = S.bootstrap(R, Z, DRAWS, rng, T["block"].to_numpy())[:, 1]
    lo, hi = S.pct_ci(B)
    return dict(rho_p=est, ci_lo=lo, ci_hi=hi)


def l1(cohort: str, arm: str, T: pd.DataFrame, alpha: pd.Series, rel_rows: list) -> list[dict]:
    out = []
    pi = T["prior_raw"].to_numpy(float)
    D = T["D_3"].to_numpy(float)
    ZD = S.design(T, C3)
    Dseed = T[[f"D_s{s}" for s in SEEDS]].to_numpy(float)
    relD = reliability(Dseed, ZD)
    for sca in ARMS[arm]:
        x = T[sca].to_numpy(float)
        rel = reliability(T[[f"{sca}_s{s}" for s in SEEDS]].to_numpy(float), ZD)
        rel_rows.append(dict(cohort=cohort, arm=arm, scalar=sca, n=len(T), primary=sca == PRIMARY[arm], **rel,
                             unstable=bool(rel["rel"] < 0.5) if np.isfinite(rel["rel"]) else True))
        b3 = rho_block(T, sca, C3, f"{cohort}{arm}C3")
        b4 = rho_block(T, sca, C4, f"{cohort}{arm}C4")
        eD = S.resid(S.midrank(D), ZD)
        eX = S.resid(S.midrank(x), ZD)
        lab = S.strata(pi, T["log_cp10k_spmean"].to_numpy(float))
        rng = np.random.default_rng(useed(cohort + arm + sca + "perm"))
        null = S.perm_null(eD, eX, lab, PERMS, rng)
        T0 = float(np.corrcoef(eD, eX)[0, 1])
        a = abs(b3["rho_p"])
        cl = "concordant" if a >= 0.3 else ("orthogonal" if a < 0.1 else "partial")
        den = rel["rel"] * relD["rel"]
        dis = b3["rho_p"] / np.sqrt(den) if np.isfinite(den) and den > 0 else np.nan
        out.append(dict(cohort=cohort, arm=arm, scalar=sca, primary=sca == PRIMARY[arm], n=len(T),
                        spearman_alpha=spearmanr(x, alpha)[0], spearman_D3=spearmanr(x, D)[0], spearman_pi=spearmanr(x, pi)[0],
                        rho_p=b3["rho_p"], ci_lo=b3["ci_lo"], ci_hi=b3["ci_hi"], p_perm=S.p_two_sided(T0, null),
                        rho_p_C4=b4["rho_p"], ci_lo_C4=b4["ci_lo"], ci_hi_C4=b4["ci_hi"],
                        rel_scalar=rel["rel"], rel_D3=relD["rel"], rho_p_disattenuated=dis, klass=cl))
    return out


def read_modules(cohort: str, arm: str, seed: int) -> tuple[sp.csr_matrix, np.ndarray]:
    z = np.load(OUT / f"modules/{cohort}__{arm}__s{seed}.npz")
    A = sp.csr_matrix((np.ones(len(z["indices"]), dtype=np.int8), z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    return A, z["hard"]


def eta2(y: np.ndarray, lab: np.ndarray, n: int) -> float:
    c = np.bincount(lab, minlength=n).astype(float)
    s = np.bincount(lab, y, minlength=n)
    m = c > 0
    return float(((s[m] ** 2) / c[m]).sum() - y.sum() ** 2 / len(y)) / float(((y - y.mean()) ** 2).sum())


def mixed(A_csr: sp.csr_matrix, y: np.ndarray) -> tuple[float, int]:
    p = A_csr @ (y > TAIL).astype(np.int32)
    q = A_csr @ (y < -TAIL).astype(np.int32)
    keep = (p + q) >= MIN_TAIL
    if not keep.any():
        return np.nan, 0
    idx = 1 - np.abs(p[keep] - q[keep]) / (p[keep] + q[keep])
    return float(idx.mean()), int(keep.sum())


def l2(cohort: str, arm: str, tab: pd.DataFrame, d3: np.ndarray) -> list[dict]:
    rows = []
    for s in SEEDS:
        A, hard = read_modules(cohort, arm, s)
        ok = hard >= 0
        A = A[:, np.where(ok)[0]].tocsr()
        hd = hard[ok]
        y = d3[ok]
        pi = tab["prior_raw"].to_numpy(float)[ok]
        ex = tab["log_cp10k_spmean"].to_numpy(float)[ok]
        lab = S.strata(pi, ex)
        n_m = int(hd.max()) + 1
        e0, (m0, nq) = eta2(y, hd, n_m), mixed(A, y)
        rng = np.random.default_rng(useed(f"{cohort}{arm}{s}L2"))
        en, mn = np.empty(PERMS), np.empty(PERMS)
        groups = [np.where(lab == u)[0] for u in np.unique(lab)]
        for i in range(PERMS):
            yp = y.copy()
            for g in groups:
                yp[g] = y[rng.permutation(g)]
            en[i] = eta2(yp, hd, n_m)
            mn[i] = mixed(A, yp)[0]
        rows.append(dict(cohort=cohort, arm=arm, seed=s, n_genes=int(ok.sum()), n_hard_modules=int(len(np.unique(hd))),
                         n_modules_scored=nq, eta2=e0, eta2_null_mean=float(en.mean()),
                         p_coherent=float((1 + (en >= e0).sum()) / (1 + PERMS)), mixed_sign=m0,
                         mixed_sign_null_mean=float(np.nanmean(mn)),
                         p_mixed=float((1 + (mn >= m0).sum()) / (1 + PERMS)) if np.isfinite(m0) else np.nan))
    return rows


def fisher_ora(fg: set, sets: dict, universe: set) -> pd.DataFrame:
    rows = []
    U = len(universe)
    for name, mem in sets.items():
        m = set(mem) & universe
        if len(m) < 5:
            continue
        a = len(fg & m)
        b = len(fg) - a
        c = len(m) - a
        d = U - a - b - c
        rows.append(dict(set=name, size=len(m), overlap=a, p=fisher_exact([[a, b], [c, d]], alternative="greater")[1]))
    df = pd.DataFrame(rows)
    if len(df):
        df["q"] = S.bh(df["p"].tolist())
    return df


def set_tables(tab: pd.DataFrame, gold: pd.DataFrame) -> dict:
    sets = {}
    s = tab.set_index("hsapiens_gene_id")["symbol_h"]
    for c in SET_NAMES:
        ids = gold.index[gold[c].eq(True)]
        sets[f"curated:{c}"] = sorted(set(s.reindex(ids).dropna().astype(str)))
    txt = G.read_gmt(G.SETS["hallmark"])
    sets.update({f"hallmark:{k}": v for k, v in txt.items()})
    return sets


def macrogene_de(work: Path, cohort: str) -> dict:
    import anndata as ad
    import glob
    import scanpy as sc

    f = [p for p in glob.glob(str(work / "saturn_results/*.h5ad"))
         if not p.endswith("_pretrain.h5ad") and "_ep_" not in Path(p).name][0]
    a = ad.read_h5ad(f)
    p = ad.read_h5ad(gene_ids.h5ad_path(cohort), backed="r")
    obs = p.obs[["species", "cell_type"]].astype(str).copy()
    p.file.close()
    a = a[obs.index.to_numpy()]
    lab = obs["cell_type"]
    if cohort.startswith("test"):
        lab = lab.map(T42.TESTIS_COARSE)
    X = np.asarray(a.obsm["macrogenes"], dtype=np.float32)
    types = I.shared_types(cohort)
    up = {}
    for s in ("Human", "Mouse"):
        m = (obs["species"] == s).to_numpy()
        b = ad.AnnData(X=X[m])
        b.obs["group"] = pd.Categorical(lab.to_numpy()[m])
        b.var_names = [str(i) for i in range(X.shape[1])]
        sc.tl.rank_genes_groups(b, "group", method="wilcoxon", use_raw=False, groups=types, reference="rest", tie_correct=True)
        res = b.uns["rank_genes_groups"]
        for t in types:
            nm = np.array(res["names"][t]).astype(int)
            hit = (np.array(res["pvals_adj"][t]) < 0.05) & (np.array(res["scores"][t]) > 0)
            up[(s, t)] = set(nm[hit].tolist())
    cons, div = set(), set()
    per = {}
    for t in types:
        c = up[("Human", t)] & up[("Mouse", t)]
        d = up[("Human", t)] ^ up[("Mouse", t)]
        cons |= c
        div |= d
        per[t] = dict(conserved=len(c), divergent=len(d))
    return dict(conserved=cons - div, divergent=div - cons, per_type=per)


def native_arm(cohort: str, tab: pd.DataFrame, d3: np.ndarray, gold: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    sets = set_tables(tab, gold)
    stem = SC.STEMS[cohort]
    rows, meta = [], {}
    long = []
    for arm in ("saturn_native", "saturn_1to1"):
        for s in SEEDS:
            w = SC.saturn_dir(arm, stem, s)
            import glob
            import pickle

            final = pickle.load(open(glob.glob(str(w / "saturn_results/*_genes_to_macrogenes.pkl"))[0], "rb"))
            de = macrogene_de(w, cohort)
            meta[f"{arm}_s{s}"] = dict(per_type=de["per_type"], n_conserved=len(de["conserved"]), n_divergent=len(de["divergent"]))
            hg = [k for k in final if k.startswith("human_")]
            mg = [k for k in final if k.startswith("mouse_")]
            Wh = np.vstack([final[k] for k in hg])
            Wm = np.vstack([final[k] for k in mg])
            m2h = dict(zip(tab["symbol_m"].astype(str), tab["symbol_h"].astype(str)))
            hs = np.array([k[len("human_"):] for k in hg])
            ms = np.array([m2h.get(k[len("mouse_"):], "") for k in mg])
            universe = set(hs) | (set(ms) - {""})

            def genes(macros):
                out = set()
                for m in macros:
                    out |= set(hs[np.argsort(-Wh[:, m])[:SC.TOPK]])
                    out |= set(ms[np.argsort(-Wm[:, m])[:SC.TOPK]])
                return out - {""}

            for lst in ("conserved", "divergent"):
                df = fisher_ora(genes(sorted(de[lst])), sets, universe)
                if len(df):
                    long.append(df.assign(cohort=cohort, method=arm, list=lst, seed=s))
    for s in SEEDS:
        A, hard = read_modules(cohort, "camex", s)
        ok = np.where(hard >= 0)[0]
        universe = set(tab["symbol_h"].astype(str).to_numpy()[ok])
        sy = tab["symbol_h"].astype(str).to_numpy()
        dfs = []
        for m in range(A.shape[0]):
            g = set(sy[A[m].indices])
            if len(g) >= 10:
                d = fisher_ora(g, sets, universe)
                if len(d):
                    dfs.append(d.assign(module=m))
        if dfs:
            d = pd.concat(dfs, ignore_index=True)
            d["q"] = S.bh(d["p"].tolist())
            d = d.sort_values("q").groupby("set", as_index=False).first()
            long.append(d.assign(cohort=cohort, method="camex", list="module", seed=s))
    L = pd.concat(long, ignore_index=True)
    r = pd.Series(d3, index=tab["symbol_h"].astype(str)).groupby(level=0).mean()
    g = G.gsea(r, sets, PERMS, BASE_SEED)
    g["q"] = S.bh(g["p"].tolist())
    rec = L.assign(hit=L["q"] < 0.05).groupby(["set", "method", "list"])["hit"].sum().unstack(["method", "list"]).fillna(0).astype(int)
    rec.columns = [f"{a}_{b}_seeds" for a, b in rec.columns]
    tb = g.set_index("set")[["size", "nes", "q"]].rename(columns=dict(size="n_gsea", nes="D3_nes", q="D3_q")).join(rec, how="outer")
    tb.insert(0, "cohort", cohort)
    return tb.reset_index().rename(columns={"index": "set"}), dict(L=L, meta=meta)


def main():
    global DRAWS, PERMS, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohorts", nargs="+", default=list(SC.STEMS))
    ap.add_argument("--arms", nargs="+", default=list(ARMS), help="testing only")
    ap.add_argument("--draws", type=int, default=DRAWS, help="testing only")
    ap.add_argument("--perms", type=int, default=PERMS, help="testing only")
    ap.add_argument("--outdir", type=Path, default=OUT, help="testing only")
    a = ap.parse_args()
    DRAWS, PERMS, OUT = a.draws, a.perms, a.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    gold = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)
    L1, REL, L2, NAT, LONG, META, PR = [], [], [], [], [], {}, []
    for cohort in a.cohorts:
        tab, M3, idx = SC.base_table(cohort)
        d3 = M3.mean(1)
        d3 = d3 - np.median(d3)
        run = I.seed_runs_used(cohort)
        from analysis.common import controls as K

        A = np.column_stack([K.read_run(run[s])["alpha"] for s in SEEDS]).mean(1)
        alpha = pd.Series(A, index=tab["var_name"])
        tools = tab[["hsapiens_gene_id"]].assign(cohort=cohort).copy()
        for arm in a.arms:
            T = pd.read_csv(OUT / f"{cohort}__{arm}.csv")
            T = T.dropna(subset=C4)
            L1 += l1(cohort, arm, T, alpha.reindex(T["var_name"]).to_numpy(), REL)
            print(cohort, arm, "L1 done", flush=True)
            L2 += l2(cohort, arm, tab, d3)
            print(cohort, arm, "L2 done", flush=True)
            p = PRIMARY[arm]
            col = {"saturn_native": "S_deploy_native", "saturn_1to1": "S_deploy_1to1", "camex": "C_cos"}[arm]
            tools[col] = T.set_index("var_name")[p].reindex(tab["var_name"]).to_numpy()
        PR.append(tools)
        if set(a.arms) == set(ARMS):
            t, extra = native_arm(cohort, tab, d3, gold)
            NAT.append(t)
            LONG.append(extra["L"])
            META[cohort] = extra["meta"]
            print(cohort, "native arm done", flush=True)
    l1df = pd.DataFrame(L1)
    l1df["q_bh"] = S.bh(l1df["p_perm"].tolist())
    l1df.to_csv(OUT / "summary.csv", index=False)
    pd.DataFrame(REL).to_csv(OUT / "reliability.csv", index=False)
    pd.DataFrame(L2).to_csv(OUT / "l2_modules.csv", index=False)
    if NAT:
        pd.concat(NAT, ignore_index=True).to_csv(OUT / "native_arm.csv", index=False)
        pd.concat(LONG, ignore_index=True).to_csv(OUT / "native_arm_long.csv", index=False)
        (OUT / "native_arm_meta.json").write_text(json.dumps(META, indent=2, default=str))
    tp = pd.concat(PR, ignore_index=True)
    tp = tp.dropna(subset=[c for c in tp.columns if c not in ("hsapiens_gene_id", "cohort")], how="all")
    tp = tp.drop_duplicates(subset=["cohort", "hsapiens_gene_id"])
    tp.to_csv(OUT / "tools_primary.csv", index=False)
    print(l1df.round(3).to_string(), flush=True)
    (OUT / "ANALYSIS_DONE").write_text("ok\n")


if __name__ == "__main__":
    main()
