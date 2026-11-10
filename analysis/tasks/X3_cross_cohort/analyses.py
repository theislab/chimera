"""X3 (EXPLORATORY): cross-cohort synthesis of Chimera readouts over the seven cohorts.

Nothing here is pre-specified, enters section 6 of the frozen plan, or can change a status,
row or modifier. Inputs are the per-gene tables of ``build.py`` (consensus over all finished
seeds) and the gene classes of ``annotations.py`` (committed before any class was compared
with a readout). Cross-cohort joins use ``hsapiens_gene_id`` only.

Readouts: ``D_raw`` = mean-over-seeds alpha minus the raw prior, minus the cohort median
(primary scale); ``D_clamp`` (testis recipes' clamped anchor); ``S`` = mean S_g_ema; T2.5
cell-type contributions. Uncentred delta-alpha is never compared across cohorts.
Baselines beside every Chimera row: ``B1b_raw`` (cross-species cell-type profile correlation,
i.e. pseudobulk correlation per pair), ``B1c_raw`` (rank-matched level), per-type pseudobulk
species-centred differences (``dc_*``).

Covariates: C4 = raw prior, species-balanced log CP10K, species-balanced detection, log human
CDS length (plan section 3 C); C_pat = C4 + cell-type specificity tau.

* Matched excess: per set gene the 40 nearest non-set genes in rank space of the covariates,
  one drawn per set gene, B draws; excess = (set mean - null mean) / SD over the universe;
  two-sided empirical p (floor 1/(B+1)). BH within cohort x score x matching where many sets.
* Residual scale: rank, residualise on spline(C4) by OLS (``analysis/common/controls.py``).
* Reliability of a consensus: Spearman-Brown of the median pairwise seed Spearman of D_raw.

Sections (``--sections``): depth (analysis 1), core (2), classes (3), esm (4), agreement (5).
Reserved sets of the frozen plan are never formed or tested (``annotations.RESERVED_KW``).
No gold standard, held-out species, constraint, dN/dS, disease or functional-layer data is read.

Writes results/analysis/X3/*.csv.
"""

from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sps
from scipy.stats import fisher_exact, rankdata, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.common import controls as K
from analysis.common import gene_ids

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotations as AN  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "X3"
GDIR, SDIR = OUT / "genes", OUT / "seeds"
COHORTS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp", "test_hmac", "test_hmouse", "test_3sp"]
LIVER_PAIRS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb"]
TESTIS_PAIRS = ["test_hmac", "test_hmouse"]
PAIR_SPECIES = {"liv_hmonkey": "Monkey", "liv_hmouse": "Mouse", "liv_hzeb": "Zebrafish",
                "test_hmac": "Macaque", "test_hmouse": "Mouse"}
TISSUE_PAIRS = [("liv_hmouse", "test_hmouse"), ("liv_4sp", "test_3sp"), ("liv_hmonkey", "test_hmac")]
C4 = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]
CPAT = C4 + ["tau"]
MATCHINGS = {"C4": C4, "C_pat": CPAT}
K_NN = 40
FDR = 0.05


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load(c: str) -> pd.DataFrame:
    g = pd.read_parquet(GDIR / f"{c}.parquet")
    g = g[g.hsapiens_gene_id.notna()].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["prior_raw_c"] = g["prior_raw"] - g["prior_raw"].median()
    return g


def load_seeds(c: str) -> pd.DataFrame:
    s = pd.read_parquet(SDIR / f"{c}.parquet")
    return s[s.hsapiens_gene_id.notna()].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    o = np.argsort(pv)
    r = pv[o] * len(pv) / (np.arange(len(pv)) + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty_like(pv)
    out[o] = np.minimum(r, 1)
    q[ok] = out
    return q


def resid(g: pd.DataFrame, col: str, covs=C4) -> pd.Series:
    """Rank of ``col`` residualised on spline(rank covariates); NaN where anything is missing."""
    ok = g[col].notna() & g[covs].notna().all(1)
    out = pd.Series(np.nan, index=g.index)
    if ok.sum() < 50:
        return out
    Z = K.design(g.loc[ok, covs], covs)
    out[ok] = K.residual_rank(g.loc[ok, col].to_numpy(), Z)
    return out / out.std()


def reliability(c: str, g: pd.DataFrame) -> float:
    """Spearman-Brown step-up of the median pairwise seed correlation of D_raw on the C4 residual scale."""
    s = load_seeds(c)
    cols = [x for x in s.columns if x.startswith("D_raw_s")]
    if len(cols) < 2:
        return np.nan
    gs = g[C4].join(s[cols])
    R = {x: resid(gs, x) for x in cols}
    r = np.median([np.corrcoef(R[a].dropna(), R[b].dropna())[0, 1] for a, b in itertools.combinations(cols, 2)])
    n = len(cols)
    return n * r / (1 + (n - 1) * r)


class Matcher:
    """Matched-excess null for one gene set in one universe (covariates fixed)."""

    def __init__(self, g: pd.DataFrame, covs: list[str]):
        self.g = g
        self.covs = covs
        self.okc = g[covs].notna().all(1).to_numpy()

    def excess(self, flag: np.ndarray, col: str, B: int, rng) -> dict:
        v = self.g[col].to_numpy(dtype=float)
        ok = self.okc & np.isfinite(v)
        f = flag[ok]
        nset = int(f.sum())
        if nset < 5 or (~f).sum() < K_NN:
            return dict(n=nset, mean_set=np.nan, mean_null=np.nan, excess_d=np.nan, p=np.nan)
        Zr = self.g.loc[ok, self.covs].to_numpy(dtype=float)
        Z = np.column_stack([rankdata(Zr[:, j]) / ok.sum() for j in range(Zr.shape[1])])
        vv = v[ok]
        pool = np.where(~f)[0]
        _, idx = NearestNeighbors(n_neighbors=K_NN).fit(Z[pool]).kneighbors(Z[f])
        nb = vv[pool[idx]]
        pick = rng.integers(0, K_NN, (B, nset))
        draws = nb[np.arange(nset)[None, :], pick].mean(1)
        obs = vv[f].mean()
        c = draws.mean()
        p = (1 + np.sum(np.abs(draws - c) >= abs(obs - c))) / (B + 1)
        return dict(n=nset, mean_set=obs, mean_null=c, excess_d=(obs - c) / vv.std(), p=p)


def classes_for(index: pd.Index) -> pd.DataFrame:
    return AN.gene_classes(index)


# ---------------------------------------------------------------------------
# 1. divergence depth
# ---------------------------------------------------------------------------

def depth(G: dict, B: int, rng, write) -> None:
    rel = {c: reliability(c, G[c]) for c in COHORTS}
    write(pd.DataFrame([dict(cohort=c, reliability_D_raw=r, n_seeds=int(G[c].n_seeds.iloc[0]))
                        for c, r in rel.items()]), "reliability.csv")
    rows = []
    for tissue, cs in [("liver", LIVER_PAIRS), ("testis", TESTIS_PAIRS)]:
        common = set.intersection(*[set(G[c].index) for c in cs])
        commonB = set.intersection(*[set(G[c].index[G[c].B1b_raw.notna()]) for c in cs])
        for c in cs:
            g = G[c]
            u = g.loc[sorted(common)]
            ub = g.loc[sorted(commonB)]
            sp_ = PAIR_SPECIES[c]
            unm = u.S.abs() > 1e-6
            tail = (u.alpha - u.prior).abs() > 0.05
            rows.append(dict(
                tissue=tissue, cohort=c, species=sp_, n_common=len(u), n_common_B1b=len(ub),
                n_seeds=int(g.n_seeds.iloc[0]), reliability_D_raw=rel[c],
                median_pair_identity=u[f"identity_{sp_}"].median(),
                sd_D_raw=u.D_raw.std(), sd_D_raw_s42=(u.alpha_s42 - u.prior_raw).std(),
                mean_abs_delta_alpha_s42=(u.alpha_s42 - u.prior).abs().mean(),
                frac_tail_s42=float(((u.alpha_s42 - u.prior).abs() > 0.05).mean()),
                frac_S_pos=float((u.S > 1e-6).mean()), frac_S_neg=float((u.S < -1e-6).mean()),
                frac_S_masked=float((~unm).mean()),
                frac_S_neg_of_unmasked=float((u.S[unm] < 0).mean()),
                median_abs_S_unmasked=float(u.S[unm].abs().median()),
                frac_tail_supported=float(u.margin_support[tail].mean()),
                median_B1b_raw_common=ub.B1b_raw.median(), mean_B1b_raw_common=ub.B1b_raw.mean(),
                frac_B1b_raw_neg_common=float((ub.B1b_raw < 0).mean()),
                median_B1b_raw_own=g.B1b_raw.median(), n_B1b_own=int(g.B1b_raw.notna().sum()),
                median_B1c_raw_common=ub.B1c_raw.median(),
                rho_B1b_identity=spearmanr(g.B1b_raw, g[f"identity_{sp_}"], nan_policy="omit")[0],
                rho_D_identity=spearmanr(g.D_raw, g[f"identity_{sp_}"], nan_policy="omit")[0],
                rho_alpha_prior=spearmanr(g.alpha, g.prior_raw)[0],
                rho_alpha_pair_identity=spearmanr(g.alpha, g[f"identity_{sp_}"], nan_policy="omit")[0],
            ))
    write(pd.DataFrame(rows), "depth_summary.csv")

    # gene-level agreement between species pairs (and with the 4-sp / 3-sp joint fits)
    combos = [("liv_hmonkey", "liv_hmouse"), ("liv_hmonkey", "liv_hzeb"), ("liv_hmouse", "liv_hzeb"),
              ("liv_4sp", "liv_hmonkey"), ("liv_4sp", "liv_hmouse"), ("liv_4sp", "liv_hzeb"),
              ("test_hmac", "test_hmouse"), ("test_3sp", "test_hmac"), ("test_3sp", "test_hmouse")]
    rows = []
    for a, b in combos:
        ga, gb = G[a], G[b]
        idx = ga.index.intersection(gb.index)
        for col in ["alpha", "D_raw", "S", "B1b_raw", "B1c_raw"]:
            x, y = ga.loc[idx, col], gb.loc[idx, col]
            ok = x.notna() & y.notna()
            ra, rb = resid(ga, col).loc[idx], resid(gb, col).loc[idx]
            okr = ra.notna() & rb.notna()
            rs = spearmanr(x[ok], y[ok])[0]
            rp = float(np.corrcoef(ra[okr], rb[okr])[0, 1]) if okr.sum() > 50 else np.nan
            dis = np.nan
            if col == "D_raw" and np.isfinite(rel[a]) and np.isfinite(rel[b]):
                dis = rs / np.sqrt(rel[a] * rel[b])
            rows.append(dict(a=a, b=b, score=col, n=int(ok.sum()), spearman=rs, rho_p=rp,
                             spearman_disattenuated=dis))
    write(pd.DataFrame(rows), "depth_gene_agreement.csv")

    # programs and classes per species pair: matched excess of D_raw, S and B1b_raw
    rows = []
    for tissue, cs in [("liver", LIVER_PAIRS), ("testis", TESTIS_PAIRS)]:
        common = pd.Index(sorted(set.intersection(*[set(G[c].index) for c in cs])))
        progs = AN.program_sets(common)
        cl = classes_for(common)
        sets = {**{k: set(common[cl[k].to_numpy()]) for k in AN.PRIMARY_CLASSES + ["TF_homeodomain",
                                                                                    "TF_C2H2_ZF", "GPCR"]},
                **progs}
        for c in cs:
            g = G[c].loc[common]
            M = Matcher(g, C4)
            for name, members in sets.items():
                flag = g.index.isin(members)
                for col in ["D_raw", "S", "B1b_raw", "B1c_raw"]:
                    r = M.excess(flag, col, B, rng)
                    rows.append(dict(tissue=tissue, cohort=c, set=name, score=col, **r,
                                     mean_identity=g.loc[flag, f"identity_{PAIR_SPECIES[c]}"].mean()))
    df = pd.DataFrame(rows)
    df["q"] = np.nan
    for _, ix in df.groupby(["cohort", "score"]).groups.items():
        df.loc[ix, "q"] = bh(df.loc[ix, "p"].to_numpy())
    write(df, "depth_programs.csv")

    # per-species margin inside the joint fits (4-sp liver, 3-sp testis)
    rows = []
    for c in ["liv_4sp", "test_3sp"]:
        g = G[c]
        sp_cols = [x for x in g.columns if x.startswith("Csp_")]
        unm = g.S.abs() > 1e-6
        for col in sp_cols:
            v = g.loc[unm, col]
            rows.append(dict(cohort=c, species=col[4:], n_unmasked=int(unm.sum()),
                             frac_negative=float((v < 0).mean()), median=float(v.median()),
                             sum_neg=float(v[v < 0].sum()), sum_pos=float(v[v > 0].sum()),
                             rho_with_D_raw=spearmanr(v, g.loc[unm, "D_raw"])[0]))
    write(pd.DataFrame(rows), "depth_species_margin.csv")

    # lineage contrast inside a joint fit: per-species mean margin (C_sp / species weight), the
    # difference of ranks between two species, against the same contrast in the direct baselines
    # (pair-cohort B1b_raw difference; mean |species-centred pseudobulk difference| per type).
    rows = []
    T25 = gene_ids.REPO_ROOT / "results" / "analysis" / "T2_5"
    for joint, (s1, c1), (s2, c2) in [("liv_4sp", ("Monkey", "liv_hmonkey"), ("Mouse", "liv_hmouse")),
                                      ("liv_4sp", ("Mouse", "liv_hmouse"), ("Zebrafish", "liv_hzeb")),
                                      ("liv_4sp", ("Monkey", "liv_hmonkey"), ("Zebrafish", "liv_hzeb")),
                                      ("test_3sp", ("Macaque", "test_hmac"), ("Mouse", "test_hmouse"))]:
        g = G[joint]
        W = (pd.read_parquet(T25 / f"{joint}__S_gsk.parquet", columns=["var_name", "species", "cell_type", "weight"])
             .drop_duplicates(["species", "cell_type"]).groupby("species").weight.sum())
        unm = g.S.abs() > 1e-6
        m1, m2 = g[f"Csp_{s1}"] / W[s1], g[f"Csp_{s2}"] / W[s2]
        contrast = (m2.rank(pct=True) - m1.rank(pct=True))[unm]  # > 0: species 2 better explained
        b = (G[c2].B1b_raw - G[c1].B1b_raw).reindex(contrast.index)
        d1 = g[[x for x in g.columns if x.startswith(f"dc_{s1}_")]].abs().mean(1)
        d2 = g[[x for x in g.columns if x.startswith(f"dc_{s2}_")]].abs().mean(1)
        pb = -(d2 - d1).reindex(contrast.index)  # > 0: species 2 closer to human
        dd = (G[c2].D_raw - G[c1].D_raw).reindex(contrast.index)
        okb, okp, okd = b.notna(), pb.notna(), dd.notna()
        rows.append(dict(joint=joint, species_1=s1, species_2=s2, n_unmasked=int(unm.sum()),
                         frac_species2_worse=float((m2[unm] < m1[unm]).mean()),
                         rho_vs_B1b_pair_difference=spearmanr(contrast[okb], b[okb])[0], n_B1b=int(okb.sum()),
                         rho_vs_pseudobulk_difference=spearmanr(contrast[okp], pb[okp])[0], n_pb=int(okp.sum()),
                         rho_vs_pair_D_difference=spearmanr(contrast[okd], dd[okd])[0],
                         rho_B1b_difference_vs_pair_D_difference=spearmanr(b[okb & okd], dd[okb & okd])[0]))
    write(pd.DataFrame(rows), "depth_lineage_contrast.csv")


# ---------------------------------------------------------------------------
# 2. core conserved deployment across tissues
# ---------------------------------------------------------------------------

def core(G: dict, B: int, rng, write) -> None:
    rows, members, enr = [], [], []
    for liv, tes in TISSUE_PAIRS:
        gl, gt = G[liv], G[tes]
        idx = gl.index.intersection(gt.index)
        for score in ["D_raw", "B1b_raw"]:
            rl, rt = resid(gl, score).loc[idx], resid(gt, score).loc[idx]
            ok = rl.notna() & rt.notna()
            rl, rt = rl[ok], rt[ok]
            ql = pd.qcut(rl.rank(method="first"), 3, labels=["low", "mid", "high"])
            qt = pd.qcut(rt.rank(method="first"), 3, labels=["low", "mid", "high"])
            tab = pd.crosstab(ql, qt)
            n = int(ok.sum())
            exp = np.outer(tab.sum(1), tab.sum(0)) / n
            row = dict(liver=liv, testis=tes, score=score, n=n,
                       rho_p=float(np.corrcoef(rl, rt)[0, 1]),
                       spearman_raw=spearmanr(gl.loc[rl.index, score], gt.loc[rl.index, score])[0])
            for a_, b_ in [("high", "high"), ("low", "low"), ("high", "low"), ("low", "high")]:
                row[f"n_{a_}L_{b_}T"] = int(tab.loc[a_, b_])
                row[f"oe_{a_}L_{b_}T"] = float(tab.loc[a_, b_] / exp[list(tab.index).index(a_),
                                                                      list(tab.columns).index(b_)])
            if score == "D_raw":
                sl, st = gl.loc[rl.index, "sign_consistency"], gt.loc[rl.index, "sign_consistency"]
                row["frac_sign_consistent_both"] = float(((sl >= 0.8) & (st >= 0.8)).mean())
            rows.append(row)
            grp = pd.Series("other", index=rl.index)
            grp[(ql == "high").to_numpy() & (qt == "high").to_numpy()] = "core_up"
            grp[(ql == "low").to_numpy() & (qt == "low").to_numpy()] = "core_down"
            grp[(ql == "high").to_numpy() & (qt == "low").to_numpy()] = "liver_up_testis_down"
            grp[(ql == "low").to_numpy() & (qt == "high").to_numpy()] = "liver_down_testis_up"
            members.append(pd.DataFrame(dict(liver=liv, testis=tes, score=score, group=grp,
                                             r_liver=rl, r_testis=rt,
                                             tau_liver=gl.loc[rl.index, "tau"],
                                             tau_testis=gt.loc[rl.index, "tau"],
                                             prior_raw=gl.loc[rl.index, "prior_raw"],
                                             symbol=gl.loc[rl.index, "symbol"])).reset_index())
    write(pd.DataFrame(rows), "core_tissue_tables.csv")
    mem = pd.concat(members, ignore_index=True)
    write(mem, "core_members.csv")

    # replicated core: same group in the H+Mouse and the 4-sp / 3-sp tissue pairs
    rep_rows = []
    for score in ["D_raw", "B1b_raw"]:
        a = mem[(mem.liver == "liv_hmouse") & (mem.score == score)].set_index("hsapiens_gene_id")
        b = mem[(mem.liver == "liv_4sp") & (mem.score == score)].set_index("hsapiens_gene_id")
        idx = a.index.intersection(b.index)
        for grp in ["core_up", "core_down", "liver_up_testis_down", "liver_down_testis_up"]:
            na = (a.loc[idx, "group"] == grp)
            nb = (b.loc[idx, "group"] == grp)
            both = int((na & nb).sum())
            expd = na.sum() * nb.sum() / len(idx)
            rep_rows.append(dict(score=score, group=grp, n_universe=len(idx), n_hmouse=int(na.sum()),
                                 n_4sp3sp=int(nb.sum()), n_both=both, expected=expd, oe=both / expd,
                                 fisher_p=fisher_exact([[both, int(na.sum()) - both],
                                                        [int(nb.sum()) - both,
                                                         len(idx) - int(na.sum()) - int(nb.sum()) + both]],
                                                       alternative="greater")[1]))
    write(pd.DataFrame(rep_rows), "core_replication.csv")

    # Chimera core vs baseline core, and what the groups are (classes, programs, tau)
    for liv, tes in TISSUE_PAIRS:
        sub = mem[(mem.liver == liv)]
        d = sub[sub.score == "D_raw"].set_index("hsapiens_gene_id")
        b = sub[sub.score == "B1b_raw"].set_index("hsapiens_gene_id")
        idx = d.index.intersection(b.index)
        for grp in ["core_up", "core_down"]:
            x, y = d.loc[idx, "group"] == grp, b.loc[idx, "group"] == grp
            both = int((x & y).sum())
            enr.append(dict(kind="chimera_vs_baseline", liver=liv, group=grp, set="B1b same group",
                            n_group=int(x.sum()), n_set=int(y.sum()), n_overlap=both,
                            oe=both / (x.sum() * y.sum() / len(idx)), p=np.nan))
        universe = d.index
        cl = classes_for(universe)
        progs = AN.program_sets(universe)
        sets = {**{k: set(universe[cl[k].to_numpy()]) for k in AN.CLASS_ORDER}, **progs}
        for score in ["D_raw", "B1b_raw"]:
            s = sub[sub.score == score].set_index("hsapiens_gene_id")
            for grp in ["core_up", "core_down", "liver_up_testis_down", "liver_down_testis_up"]:
                ing = s.group == grp
                for name, mset in sets.items():
                    ins = s.index.isin(mset)
                    a_ = int((ing & ins).sum())
                    b_ = int((ing & ~ins).sum())
                    c_ = int((~ing & ins).sum())
                    d_ = int((~ing & ~ins).sum())
                    if a_ + c_ < 10:
                        continue
                    orr, p = fisher_exact([[a_, b_], [c_, d_]])
                    enr.append(dict(kind="enrichment", liver=liv, score=score, group=grp, set=name,
                                    n_group=a_ + b_, n_set=a_ + c_, n_overlap=a_,
                                    oe=a_ / ((a_ + b_) * (a_ + c_) / len(s)), odds_ratio=orr, p=p))
                tl, tt = s.loc[ing, "tau_liver"].median(), s.loc[ing, "tau_testis"].median()
                enr.append(dict(kind="tau", liver=liv, score=score, group=grp, set="median tau",
                                n_group=int(ing.sum()), tau_liver=tl, tau_testis=tt,
                                tau_liver_all=s.tau_liver.median(), tau_testis_all=s.tau_testis.median(),
                                prior_raw=s.loc[ing, "prior_raw"].median(),
                                prior_raw_all=s.prior_raw.median()))
    e = pd.DataFrame(enr)
    e["q"] = np.nan
    m = e.kind == "enrichment"
    for _, ix in e[m].groupby(["liver", "score", "group"]).groups.items():
        e.loc[ix, "q"] = bh(e.loc[ix, "p"].to_numpy())
    write(e, "core_enrichment.csv")

    # tissue discordance vs specificity: |r_liver - r_testis| against tau (rank correlation)
    rows = []
    for (liv, score), s in mem.groupby(["liver", "score"]):
        disc = (s.r_liver - s.r_testis).abs()
        rows.append(dict(liver=liv, score=score, n=len(s),
                         rho_disc_tau_liver=spearmanr(disc, s.tau_liver, nan_policy="omit")[0],
                         rho_disc_tau_testis=spearmanr(disc, s.tau_testis, nan_policy="omit")[0],
                         rho_rliver_tau_liver=spearmanr(s.r_liver, s.tau_liver, nan_policy="omit")[0],
                         rho_rtestis_tau_testis=spearmanr(s.r_testis, s.tau_testis, nan_policy="omit")[0]))
    write(pd.DataFrame(rows), "core_tau.csv")


# ---------------------------------------------------------------------------
# 3. protein classes
# ---------------------------------------------------------------------------

def class_excess(G: dict, B: int, rng, write) -> None:
    rows, seed_rows, desc = [], [], []
    for c in COHORTS:
        g = G[c]
        cl = classes_for(g.index)
        for mname, covs in MATCHINGS.items():
            M = Matcher(g, covs)
            for k in AN.CLASS_ORDER:
                flag = cl[k].to_numpy()
                for col in ["D_raw", "D_clamp", "S", "B1b_raw", "B1c_raw"]:
                    r = M.excess(flag, col, B, rng)
                    rows.append(dict(cohort=c, cls=k, matching=mname, score=col, **r))
        for k in AN.CLASS_ORDER:
            f = cl[k].to_numpy()
            desc.append(dict(cohort=c, cls=k, n=int(f.sum()), median_prior_raw=g.prior_raw[f].median(),
                             median_prior_raw_all=g.prior_raw.median(),
                             median_log_cp10k=g.log_cp10k_spmean[f].median(),
                             median_log_cp10k_all=g.log_cp10k_spmean.median(),
                             median_tau=g.tau[f].median(), median_tau_all=g.tau.median(),
                             frac_margin_support_tail=float(g.margin_support[f & g["tail"].to_numpy()].mean())
                             if (f & g["tail"].to_numpy()).sum() else np.nan))
        s = load_seeds(c)
        cols = [x for x in s.columns if x.startswith("D_raw_s")]
        if len(cols) > 1:
            gs = g.join(s[cols])
            M = Matcher(gs, C4)
            for k in AN.PRIMARY_CLASSES + ["TF_homeodomain", "TF_C2H2_ZF", "GPCR", "mitochondrial"]:
                flag = cl[k].to_numpy()
                for col in cols:
                    r = M.excess(flag, col, B, rng)
                    seed_rows.append(dict(cohort=c, cls=k, seed=int(col.split("_s")[-1]), **r))
    write(pd.DataFrame(rows), "class_excess.csv")
    write(pd.DataFrame(seed_rows), "class_excess_seeds.csv")
    write(pd.DataFrame(desc), "class_description.csv")


# ---------------------------------------------------------------------------
# 4. what the gate learned: ESM-2 substrate
# ---------------------------------------------------------------------------

def knn_graph(X: np.ndarray, k: int = 15) -> sps.csr_matrix:
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True).clip(1e-12)
    _, idx = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xn).kneighbors(Xn)
    n = len(X)
    rows = np.repeat(np.arange(n), k)
    W = sps.csr_matrix((np.full(n * k, 1.0 / k), (rows, idx[:, 1:].ravel())), shape=(n, n))
    return W


def morans_i(W: sps.csr_matrix, v: np.ndarray) -> float:
    z = v - v.mean()
    return float(len(v) / W.sum() * (z @ (W @ z)) / (z @ z))


def strata(g: pd.DataFrame) -> np.ndarray:
    a = pd.qcut(g.prior_raw.rank(method="first"), 10, labels=False)
    b = pd.qcut(g.log_cp10k_spmean.rank(method="first"), 10, labels=False)
    return (a * 10 + b).to_numpy()


def strat_perm(v: np.ndarray, st: np.ndarray, rng) -> np.ndarray:
    """Permutation of ``v`` within strata ``st`` (vectorised)."""
    base = np.argsort(st, kind="stable")
    perm = np.lexsort((rng.random(len(v)), st))
    out = np.empty_like(v)
    out[base] = v[perm]
    return out


def cv_r2(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, np.ndarray]:
    pred = np.full(len(y), np.nan)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(X):
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 29))).fit(X[tr], y[tr])
        pred[te] = m.predict(X[te])
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2), pred


def esm(G: dict, B: int, rng, write) -> None:
    ecols = [f"esm_{i}" for i in range(64)]
    r2_rows, mor_rows, fam_rows, load_rows, pred_store = [], [], [], [], {}
    for c in COHORTS:
        g = G[c].copy()
        g["rD"] = resid(g, "D_raw")
        g["rS"] = resid(g, "S")
        g["rB1b"] = resid(g, "B1b_raw")
        X = g[ecols].to_numpy()
        for col in ["D_raw", "rD", "S", "rS", "B1b_raw", "rB1b", "prior_raw", "log_cp10k_spmean", "tau"]:
            ok = g[col].notna().to_numpy()
            r2, pred = cv_r2(X[ok], g.loc[ok, col].to_numpy())
            r2_rows.append(dict(cohort=c, target=col, n=int(ok.sum()), cv_r2_esm64=r2))
            if col in ("rD", "rB1b"):
                pred_store[(c, col)] = pd.Series(pred, index=g.index[ok])
        # Moran's I on the 15-NN cosine graph of the 64 ESM PCs, stratified permutation null
        st_all = strata(g)
        for col in ["D_raw", "rD", "S", "rS", "B1b_raw", "rB1b", "prior_raw", "log_cp10k_spmean"]:
            ok = g[col].notna().to_numpy() & np.isfinite(st_all.astype(float))
            W = knn_graph(X[ok])
            v = g.loc[ok, col].to_numpy(dtype=float)
            st = st_all[ok]
            obs = morans_i(W, v)
            if col in ("prior_raw", "log_cp10k_spmean"):
                null = np.array([morans_i(W, rng.permutation(v)) for _ in range(B)])
                kind = "free"
            else:
                null = np.array([morans_i(W, strat_perm(v, st, rng)) for _ in range(B)])
                kind = "within prior x expression deciles"
            mor_rows.append(dict(cohort=c, score=col, n=int(ok.sum()), morans_I=obs, null_mean=null.mean(),
                                 null_sd=null.std(), z=(obs - null.mean()) / null.std(),
                                 p=(1 + np.sum(null >= obs)) / (B + 1), null=kind))
        # protein families: between-family variance of residualised scores vs permutation
        cl = classes_for(g.index)
        fam = cl["family"].copy()
        res = AN.reserved_share(g.index)
        share = res.groupby(fam).mean()
        fam[fam.isin(share[share >= 0.5].index)] = np.nan  # reserved-adjacent families never used
        for col in ["rD", "rS", "rB1b"]:
            ok = g[col].notna() & fam.notna()
            sub = pd.DataFrame(dict(v=g.loc[ok, col], f=fam[ok]))
            vc = sub.f.value_counts()
            sub = sub[sub.f.isin(vc[vc >= 5].index)]
            if len(sub) < 50:
                continue
            codes, uniq = pd.factorize(sub.f)
            v = sub.v.to_numpy()

            def between(vv):
                m = np.bincount(codes, vv) / np.bincount(codes)
                return np.sum(np.bincount(codes) * (m - vv.mean()) ** 2) / np.sum((vv - vv.mean()) ** 2)

            obs = between(v)
            null = np.array([between(rng.permutation(v)) for _ in range(B)])
            fam_rows.append(dict(cohort=c, score=col, n_genes=len(v), n_families=len(uniq),
                                 eta2=obs, null_mean=null.mean(), ratio=obs / null.mean(),
                                 p=(1 + np.sum(null >= obs)) / (B + 1)))
            means = sub.groupby("f").v.agg(["mean", "size"])
            means["z"] = means["mean"] * np.sqrt(means["size"])
            for f_, r in means.iterrows():
                load_rows.append(dict(cohort=c, score=col, family=f_, n=int(r["size"]), mean=r["mean"], z=r["z"]))
    write(pd.DataFrame(r2_rows), "esm_cv_r2.csv")
    write(pd.DataFrame(mor_rows), "esm_morans_i.csv")
    write(pd.DataFrame(fam_rows), "esm_family_coherence.csv")
    write(pd.DataFrame(load_rows), "esm_family_means.csv")

    # cross-cohort agreement: ESM-predictable part vs the rest of the residualised D
    rows = []
    for a, b in [("liv_hmonkey", "liv_hmouse"), ("liv_hmouse", "liv_4sp"), ("liv_hmouse", "liv_hzeb"),
                 ("test_hmac", "test_hmouse"), ("test_hmouse", "test_3sp"),
                 ("liv_hmouse", "test_hmouse"), ("liv_4sp", "test_3sp"), ("liv_hmonkey", "test_hmac")]:
        for col in ["rD", "rB1b"]:
            pa, pb = pred_store[(a, col)], pred_store[(b, col)]
            ga, gb = G[a], G[b]
            ya = resid(ga, "D_raw" if col == "rD" else "B1b_raw")
            yb = resid(gb, "D_raw" if col == "rD" else "B1b_raw")
            idx = pa.index.intersection(pb.index)
            ea, eb = (ya.loc[idx] - pa.loc[idx]), (yb.loc[idx] - pb.loc[idx])
            rows.append(dict(a=a, b=b, score=col, n=len(idx),
                             rho_total=spearmanr(ya.loc[idx], yb.loc[idx])[0],
                             rho_esm_part=spearmanr(pa.loc[idx], pb.loc[idx])[0],
                             rho_non_esm_part=spearmanr(ea, eb)[0]))
    write(pd.DataFrame(rows), "esm_cross_cohort_parts.csv")

    # which classes the ESM-predicted direction of residual D separates (matched on C4)
    rows = []
    for c in COHORTS:
        g = G[c].copy()
        g["esm_pred_rD"] = pred_store[(c, "rD")].reindex(g.index)
        g["esm_pred_rB1b"] = pred_store[(c, "rB1b")].reindex(g.index)
        g["non_esm_rD"] = resid(g, "D_raw") - g["esm_pred_rD"]
        g["non_esm_rB1b"] = resid(g, "B1b_raw") - g["esm_pred_rB1b"]
        cl = classes_for(g.index)
        M = Matcher(g, C4)
        for k in AN.CLASS_ORDER:
            for col in ["esm_pred_rD", "non_esm_rD", "esm_pred_rB1b", "non_esm_rB1b"]:
                rows.append(dict(cohort=c, cls=k, score=col, **M.excess(cl[k].to_numpy(), col, B, rng)))
    write(pd.DataFrame(rows), "esm_class_loadings.csv")


# ---------------------------------------------------------------------------
# 5. agreement of delta-alpha and S_g, and which cell types carry the evidence
# ---------------------------------------------------------------------------

def agreement(G: dict, B: int, rng, write) -> None:
    rows = []
    for c in COHORTS:
        g = G[c]
        s = load_seeds(c)
        unm = g.S.abs() > 1e-6
        da = g.alpha - g.prior  # the run's own anchor, the quantity of the equilibrium theorem
        rD, rS = resid(g, "D_raw"), resid(g, "S")
        ok = rD.notna() & rS.notna()
        row = dict(cohort=c, n=len(g), n_unmasked=int(unm.sum()), spearman_D_S=spearmanr(g.D_raw, g.S)[0],
                   rho_p_D_S=float(np.corrcoef(rD[ok], rS[ok])[0, 1]),
                   sign_agree_unmasked=float((np.sign(da[unm]) == np.sign(g.S[unm])).mean()),
                   sign_agree_tail=float((np.sign(da[unm & (da.abs() > 0.05)]) ==
                                          np.sign(g.S[unm & (da.abs() > 0.05)])).mean()),
                   frac_S_pos_unmasked=float((g.S[unm] > 0).mean()),
                   frac_da_pos_unmasked=float((da[unm] > 0).mean()))
        row["sign_agree_expected"] = (row["frac_S_pos_unmasked"] * row["frac_da_pos_unmasked"]
                                      + (1 - row["frac_S_pos_unmasked"]) * (1 - row["frac_da_pos_unmasked"]))
        scols = [x for x in s.columns if x.startswith("S_s")]
        dcols = [x for x in s.columns if x.startswith("D_raw_s")]
        per = [spearmanr(s[d], s[sc])[0] for d, sc in zip(dcols, scols)]
        row["spearman_D_S_per_seed_min"], row["spearman_D_S_per_seed_max"] = min(per), max(per)
        row["spearman_B1b_S"] = spearmanr(g.B1b_raw, g.S, nan_policy="omit")[0]
        row["spearman_B1b_D"] = spearmanr(g.B1b_raw, g.D_raw, nan_policy="omit")[0]
        rB = resid(g, "B1b_raw")
        okb = rB.notna() & rS.notna() & rD.notna()
        row["rho_p_B1b_S"] = float(np.corrcoef(rB[okb], rS[okb])[0, 1])
        row["rho_p_B1b_D"] = float(np.corrcoef(rB[okb], rD[okb])[0, 1])
        rows.append(row)
    write(pd.DataFrame(rows), "agreement_D_S.csv")

    # cell-type shares of positive / negative S mass relative to weight, by D group
    wts = pd.read_csv(OUT / "celltype_weights.csv")
    rows, top_rows = [], []
    for c in COHORTS:
        g = G[c]
        ccols = [x for x in g.columns if x.startswith("C_")]
        types = [x[2:] for x in ccols]
        w = wts[wts.cohort == c].set_index("cell_type")
        # testis contributions are on coarse stages; weights summed from fine types
        if set(types) - set(w.index):
            w = None
        rD = resid(g, "D_raw")
        q = pd.qcut(rD.rank(method="first"), 10, labels=False)
        for grp, mask in [("all", rD.notna()), ("D top decile", q == 9), ("D bottom decile", q == 0)]:
            Cm = g.loc[mask, ccols]
            pos, neg = Cm.clip(lower=0).sum(), (-Cm.clip(upper=0)).sum()
            for t, cc in zip(types, ccols):
                wt = float(w.loc[t, "weight_k"]) if w is not None else np.nan
                rows.append(dict(cohort=c, group=grp, cell_type=t, weight=wt, n_genes=int(mask.sum()),
                                 share_pos=pos[cc] / pos.sum(), share_neg=neg[cc] / neg.sum(),
                                 pos_per_weight=pos[cc] / pos.sum() / wt, neg_per_weight=neg[cc] / neg.sum() / wt))
        # baseline: does the type with the most negative contribution match the type with the
        # largest species-centred pseudobulk difference? (types with pseudobulk only)
        dcols = [x for x in g.columns if x.startswith("dc_")]
        sp_list = sorted({x.split("_")[1] for x in dcols})
        for sp_ in sp_list:
            tt = [x[len(f"dc_{sp_}_"):] for x in dcols if x.startswith(f"dc_{sp_}_")]
            tt = [t for t in tt if f"C_{t}" in g.columns]
            if len(tt) < 2:
                continue
            Cm = g[[f"C_{t}" for t in tt]].to_numpy()
            Dm = g[[f"dc_{sp_}_{t}" for t in tt]].abs().to_numpy()
            fH = g[[f"fH_{t}" for t in tt]].to_numpy()
            okg = (g.S < -1e-6).to_numpy() & np.isfinite(Dm).all(1) & (fH.max(1) >= 0.05)
            a_ = Cm[okg].argmin(1)
            b_ = Dm[okg].argmax(1)
            pa = np.bincount(a_, minlength=len(tt)) / okg.sum()
            pb = np.bincount(b_, minlength=len(tt)) / okg.sum()
            match = float((a_ == b_).mean())
            # gene-wise permutation null of the baseline argmax
            null = np.array([(a_ == rng.permutation(b_)).mean() for _ in range(B)])
            # within-gene rank agreement between -C and |dc| across types
            rr = np.array([spearmanr(-Cm[i], Dm[i])[0] for i in np.where(okg)[0]])
            top_rows.append(dict(cohort=c, species=sp_, types=" ".join(tt), n_genes=int(okg.sum()),
                                 match_rate=match, expected=float(pa @ pb), null_mean=null.mean(),
                                 p=(1 + np.sum(null >= match)) / (B + 1),
                                 median_within_gene_rho=float(np.nanmedian(rr)),
                                 most_negative_type=tt[int(np.bincount(a_).argmax())],
                                 most_divergent_type_baseline=tt[int(np.bincount(b_).argmax())]))
    write(pd.DataFrame(rows), "agreement_celltype_shares.csv")
    write(pd.DataFrame(top_rows), "agreement_celltype_baseline.csv")


# ---------------------------------------------------------------------------

SECTIONS = {"depth": depth, "core": core, "classes": class_excess, "esm": esm, "agreement": agreement}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sections", nargs="+", default=list(SECTIONS))
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args(argv)
    out = Path(a.outdir) if a.outdir else OUT
    out.mkdir(parents=True, exist_ok=True)

    def write(df, name):
        K.write_csv(df, out / name)
        print(f"  wrote {name} ({len(df)} rows)", flush=True)

    G = {c: load(c) for c in COHORTS}
    for s in a.sections:
        print(f"== {s}", flush=True)
        SECTIONS[s](G, a.draws, np.random.default_rng(a.seed), write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
