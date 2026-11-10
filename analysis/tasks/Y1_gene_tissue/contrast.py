"""Y1 (c), (c2): does X in tissue T predict independent pattern agreement in T better than X from another tissue, given C of T?

Joint block draws: one multiplicity per ESM module per draw, shared by all golds and scores. rules.py fixes the family and sensitivities.

    python -I launch.py analysis.tasks.Y1_gene_tissue.contrast [--draws 10000] [--sections main,sens,within]
"""

from __future__ import annotations

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.Y1_gene_tissue import common as Y
from analysis.tasks.Y1_gene_tissue import prep as P
from analysis.tasks.Y1_gene_tissue import rules as RU

MAT, ORDER = P.MAT, RU.TISSUES
OUT = RU.OUT
GOLDS = {"liver": "liver", "testis": "testis", "kidney": "kidney", "pancreas": "islet"}
PAT = ["prior_raw", "expr", "det", "log_cds_length", "tau"]
WORKERS = int(os.environ.get("Y1_WORKERS", "1"))


class Data:
    def __init__(self):
        self.sup = pd.read_parquet(MAT / "support.parquet")
        self.raw = {sc: pd.read_parquet(MAT / f"raw_{sc}.parquet") for sc in ("S", "D", "B1", "Sdir", "Sdir_pattern", "S_fw", "S_unw")}
        self.cov = {c: pd.read_parquet(MAT / f"cov_{c}.parquet") for c in ("prior_raw", "expr", "det", "log_cds_length", "tau", "amp", "dom")}
        self.gold = pd.read_parquet(MAT / "golds.parquet")
        bk = pd.read_parquet(MAT / "blocks.parquet")["block"]
        self.block = bk
        self.ub = np.unique(bk.to_numpy())

    def cov_T(self, T: str, genes) -> pd.DataFrame:
        return pd.DataFrame({c: self.cov[c][T] for c in self.cov}).reindex(genes)


def weights(D: Data, genes, cnt) -> np.ndarray:
    return cnt[np.searchsorted(D.ub, D.block.reindex(genes).to_numpy())]


def block_counts(D: Data, n_draws: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.stack([np.bincount(rng.integers(0, len(D.ub), len(D.ub)), minlength=len(D.ub)) for _ in range(n_draws)]).astype(float)


def rho_table(D: Data, gold_col: str, genes, columns: dict, Z: np.ndarray, CNT: np.ndarray):
    """Point rho_p and draws for many score columns against one gold on one universe and design."""
    names = list(columns)
    R = np.column_stack([S.midrank(D.gold[gold_col].reindex(genes).to_numpy(float))] + [S.midrank(columns[n].to_numpy(float)) for n in names])
    est = S.rho_p_matrix(R, Z)[1:]
    def one(cnt):
        return S.weighted_rho(R, Z, weights(D, genes, cnt))[1:]
    if WORKERS > 1:
        draws = np.array(Parallel(n_jobs=WORKERS, backend="threading", batch_size=64)(delayed(one)(c) for c in CNT))
    else:
        draws = np.array([one(c) for c in CNT])
    return names, est, draws


def p_from(d: np.ndarray) -> float:
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()) + 1 / len(d)))


def summarise(name: str, est: float, draws: np.ndarray, **kw) -> dict:
    lo, hi = S.pct_ci(draws)
    return dict(name=name, est=float(est), lo=lo, hi=hi, p=p_from(draws), **kw)


def universe(D: Data, T: str, gold_col: str, extra=()) -> pd.Index:
    g = D.gold[gold_col].dropna().index
    m = D.sup.reindex(g).fillna(False).astype(bool).all(axis=1) & D.raw["B1"].reindex(g).notna().all(axis=1)
    genes = m[m].index
    cv = D.cov_T(T, genes)[PAT + list(extra)]
    genes = cv.dropna().index
    if gold_col == "G_testis":
        genes = genes[D.gold["H4_nsp"].reindex(genes).notna()]
    return genes


def design_T(D: Data, T: str, genes, cols, gold_col, extra: pd.DataFrame | None = None) -> np.ndarray:
    cv = D.cov_T(T, genes)[cols].copy()
    if gold_col == "G_testis":
        cv["H4_nsp"] = D.gold["H4_nsp"].reindex(genes)
    if extra is not None:
        cv = cv.join(extra)
    return S.design(cv, list(cv.columns))


def source_columns(D: Data, genes, scores) -> dict:
    return {f"{sc}|{t}": D.raw[sc][t].reindex(genes) for sc in scores for t in ORDER}


def contrast(names, est, draws, scores, T) -> tuple[list, dict]:
    idx = {n: i for i, n in enumerate(names)}
    rows, deltas = [], {}
    for sc in scores:
        own = idx[f"{sc}|{T}"]
        oth = [idx[f"{sc}|{t}"] for t in ORDER if t != T]
        dE = est[own] - est[oth].mean()
        dD = draws[:, own] - draws[:, oth].mean(1)
        rows.append(summarise(f"Delta({sc})", dE, dD, score=sc))
        deltas[sc] = (dE, dD)
    return rows, deltas


def run_main(D: Data, CNT) -> None:
    scores = ["S", "B1", "D", "Sdir", "Sdir_pattern"]
    rho_rows, delta_rows, uni = [], [], {}
    all_rho = {}
    for gold, T in GOLDS.items():
        col = f"G_{gold}"
        genes = universe(D, T, col)
        uni[gold] = len(genes)
        Z = design_T(D, T, genes, PAT, col)
        names, est, draws = rho_table(D, col, genes, source_columns(D, genes, scores), Z, CNT)
        all_rho[gold] = (names, est, draws)
        for n, e, d in zip(names, est, draws.T):
            sc, t = n.split("|")
            rho_rows.append(summarise(n, e, d, gold=gold, score=sc, source=t, matched=(t == T), n_genes=len(genes)))
        rows, dl = contrast(names, est, draws, scores, T)
        for r in rows:
            r.update(gold=gold, T=T, n_genes=len(genes))
            delta_rows.append(r)
        for a, b in (("S", "B1"), ("S", "Sdir"), ("D", "B1"), ("S", "D")):
            delta_rows.append(summarise(f"Delta({a})-Delta({b})", dl[a][0] - dl[b][0], dl[a][1] - dl[b][1], score=f"{a}-{b}", gold=gold, T=T, n_genes=len(genes)))
    rho = pd.DataFrame(rho_rows)
    dt = pd.DataFrame(delta_rows)
    fam = dt["score"].isin(["S", "B1", "S-B1"])
    dt.loc[fam, "q"] = S.bh(dt.loc[fam, "p"].tolist())
    for sc in ("D", "Sdir", "Sdir_pattern", "S-Sdir", "D-B1", "S-D"):
        m = dt["score"] == sc
        dt.loc[m, "q_own"] = S.bh(dt.loc[m, "p"].tolist())
    rho.to_csv(OUT / "contrast_rho.csv", index=False)
    dt.to_csv(OUT / "contrast_delta.csv", index=False)
    # crossed interaction over the four matched tissues: each gold on its own universe, same draws
    mt = list(GOLDS.values())
    golds_ = list(GOLDS)
    rows = []
    for sc in scores:
        Me = np.array([[all_rho[g][1][all_rho[g][0].index(f"{sc}|{t}")] for t in mt] for g in golds_])
        Md = np.stack([[all_rho[g][2][:, all_rho[g][0].index(f"{sc}|{t}")] for t in mt] for g in golds_])  # gold, t, draws
        def inter(M):
            return (np.trace(M) / 4 - M.mean(1).mean() - M.mean(0).mean() + M.mean())
        ie = inter(Me)
        id_ = np.array([inter(Md[:, :, b]) for b in range(Md.shape[2])])
        rows.append(summarise(f"interaction({sc})", ie, id_, score=sc))
        if sc == "S":
            iS = id_
        if sc == "B1":
            iB = id_
    rows.append(summarise("interaction(S)-interaction(B1)", rows[0]["est"] - rows[1]["est"], iS - iB, score="S-B1"))
    pd.DataFrame(rows).to_csv(OUT / "contrast_interaction.csv", index=False)
    # incremental: what the matched readout keeps beyond its own baseline and beyond the other tissues' readout
    zS = pd.read_parquet(MAT / "z_S__ED.parquet")
    zB = pd.read_parquet(MAT / "z_B1__ED.parquet")
    zD = pd.read_parquet(MAT / "z_D__ED.parquet")
    rows = []
    for gold, T in GOLDS.items():
        col = f"G_{gold}"
        genes = universe(D, T, col)
        oth = [t for t in ORDER if t != T]
        ex = pd.DataFrame(dict(
            B1_T=D.raw["B1"][T].reindex(genes), S_T=D.raw["S"][T].reindex(genes), Sdir_T=D.raw["Sdir"][T].reindex(genes), D_T=D.raw["D"][T].reindex(genes),
            others_S=zS[oth].reindex(genes).mean(axis=1), others_B1=zB[oth].reindex(genes).mean(axis=1), others_D=zD[oth].reindex(genes).mean(axis=1)))
        specs = [("S_T", "S_T", []), ("S_T | B1_T", "S_T", ["B1_T"]), ("S_T | others S", "S_T", ["others_S"]), ("S_T | B1_T, others S", "S_T", ["B1_T", "others_S"]),
                 ("S_T | B1_T, others S, others B1", "S_T", ["B1_T", "others_S", "others_B1"]), ("S_T | Sdir_T", "S_T", ["Sdir_T"]),
                 ("B1_T", "B1_T", []), ("B1_T | S_T", "B1_T", ["S_T"]), ("B1_T | others B1", "B1_T", ["others_B1"]), ("B1_T | S_T, others B1", "B1_T", ["S_T", "others_B1"]),
                 ("B1_T | S_T, others B1, others S", "B1_T", ["S_T", "others_B1", "others_S"]), ("Sdir_T | S_T", "Sdir_T", ["S_T"]),
                 ("D_T", "D_T", []), ("D_T | S_T", "D_T", ["S_T"]), ("D_T | others D", "D_T", ["others_D"]), ("others S alone", "others_S", []), ("others B1 alone", "others_B1", [])]
        for name, x, cond in specs:
            Z = design_T(D, T, genes, PAT, col, ex[cond] if cond else None)
            n_, est, dr = rho_table(D, col, genes, {x: ex[x]}, Z, CNT)
            rows.append(summarise(name, est[0], dr[:, 0], gold=gold, T=T, n_genes=len(genes)))
    pd.DataFrame(rows).to_csv(OUT / "contrast_incremental.csv", index=False)
    pd.Series(uni).to_csv(OUT / "contrast_universes.csv")
    print(dt.round(3).to_string(), flush=True)


def run_sens(D: Data, CNT) -> None:
    rows = []
    scores = ["S", "B1"]
    # raw per-seed tables
    for gold, T in GOLDS.items():
        col = f"G_{gold}"
        genes = universe(D, T, col)

        def add(tag, est, draws, names, n, TT=T):
            r, dl = contrast(names, est, draws, scores, TT)
            for x in r:
                x.update(sens=tag, gold=gold, n_genes=n)
                rows.append(x)
            rows.append(summarise("Delta(S)-Delta(B1)", dl["S"][0] - dl["B1"][0], dl["S"][1] - dl["B1"][1], score="S-B1", sens=tag, gold=gold, n_genes=n))

        # (i) + every source tissue's expr, det, tau
        ex = pd.DataFrame({f"{c}_{t}": D.cov[c][t].reindex(genes) for c in ("expr", "det", "tau") for t in ORDER})
        Z = design_T(D, T, genes, PAT, col, ex)
        n_, e_, d_ = rho_table(D, col, genes, source_columns(D, genes, scores), Z, CNT)
        add("i_all_source_covariates", e_, d_, n_, len(genes))
        # (ii) matched variability: C_PAT + amplitude in T, atlas amplitude, atlas expression and detection
        ats = [f"amp_{gold}", f"aexpr_{gold}"] + ([f"adet_{gold}"] if gold != "testis" else [])
        g2 = genes[D.gold[ats].reindex(genes).notna().all(axis=1) & D.cov["amp"][T].reindex(genes).notna().to_numpy()]
        ex = D.gold[ats].reindex(g2).assign(amp_T=D.cov["amp"][T].reindex(g2))
        Z = design_T(D, T, g2, PAT, col, ex)
        n_, e_, d_ = rho_table(D, col, g2, source_columns(D, g2, scores), Z, CNT)
        add("ii_matched_variability", e_, d_, n_, len(g2))
        # (iii) + dominant-type enrichment in T
        ex3 = ex.assign(dom_T=D.cov["dom"][T].reindex(g2))
        Z = design_T(D, T, g2, PAT, col, ex3)
        n_, e_, d_ = rho_table(D, col, g2, source_columns(D, g2, scores), Z, CNT)
        add("iii_plus_dominance", e_, d_, n_, len(g2))
        # (iv) pairwise universes: support only in T and t
        for sc in scores:
            own, oth, ownD, othD = [], [], [], []
            for t in ORDER:
                if t == T:
                    continue
                g = D.gold[col].dropna().index
                m = D.sup.reindex(g)[[T, t]].fillna(False).all(axis=1) & D.raw["B1"].reindex(g)[[T, t]].notna().all(axis=1)
                g = m[m].index
                g = D.cov_T(T, g)[PAT].dropna().index
                if col == "G_testis":
                    g = g[D.gold["H4_nsp"].reindex(g).notna()]
                Z = design_T(D, T, g, PAT, col)
                _, e_, d_ = rho_table(D, col, g, {"own": D.raw[sc][T].reindex(g), "oth": D.raw[sc][t].reindex(g)}, Z, CNT)
                own.append(e_[0] - e_[1])
                ownD.append(d_[:, 0] - d_[:, 1])
                rows.append(summarise(f"pair {t}", e_[0] - e_[1], d_[:, 0] - d_[:, 1], score=sc, sens="iv_pairwise", gold=gold, n_genes=len(g)))
            rows.append(summarise("Delta pairwise mean", float(np.mean(own)), np.mean(ownD, axis=0), score=sc, sens="iv_pairwise_mean", gold=gold))
        # (v) unweighted margin vs weighted forward margin, same forward seed
        Z = design_T(D, T, genes, PAT, col)
        for sc in ("S_fw", "S_unw"):
            cols_ = {f"{sc}|{t}": D.raw[sc][t].reindex(genes) for t in ORDER}
            n_, e_, d_ = rho_table(D, col, genes, cols_, Z, CNT)
            r, _ = contrast(n_, e_, d_, [sc], T)
            for x in r:
                x.update(sens="v_forward_margin", gold=gold, n_genes=len(genes))
                rows.append(x)
        # (vi) per-seed S_T against the others' consensus
        sm = pd.read_parquet(MAT / f"seedS__{T}.parquet").reindex(genes)
        for s in sm.columns:
            cols_ = {f"S|{t}": D.raw["S"][t].reindex(genes) for t in ORDER}
            cols_[f"S|{T}"] = sm[s]
            n_, e_, d_ = rho_table(D, col, genes, cols_, Z, CNT)
            r, _ = contrast(n_, e_, d_, ["S"], T)
            for x in r:
                x.update(sens=f"vi_seed_{s}", gold=gold, n_genes=len(genes))
                rows.append(x)
        if T == "liver":
            cols_ = {f"S|{t}": D.raw["S"][t].reindex(genes) for t in ORDER}
            cols_["S|liver"] = sm[[f"S_s{s}" for s in (42, 43, 44, 47, 48, 51)]].mean(axis=1)
            n_, e_, d_ = rho_table(D, col, genes, cols_, Z, CNT)
            r, _ = contrast(n_, e_, d_, ["S"], T)
            for x in r:
                x.update(sens="vi_liver_integrated_seeds", gold=gold, n_genes=len(genes))
                rows.append(x)
        # (viii) leave one source tissue out of the mean, and sources by depth
        for drop in [t for t in ORDER if t != T]:
            names, est, draws = rho_table(D, col, genes, source_columns(D, genes, scores), Z, CNT)
            idx = {n: i for i, n in enumerate(names)}
            for sc in scores:
                oth = [idx[f"{sc}|{t}"] for t in ORDER if t not in (T, drop)]
                rows.append(summarise(f"drop {drop}", est[idx[f"{sc}|{T}"]] - est[oth].mean(), draws[:, idx[f"{sc}|{T}"]] - draws[:, oth].mean(1), score=sc,
                                      sens="viii_drop_source", gold=gold, n_genes=len(genes)))
    # (vii) pancreas gold without acinar
    col, T, gold = "G_pancreas_noacinar", "islet", "pancreas_noacinar"
    genes = universe(D, T, col)
    Z = design_T(D, T, genes, PAT, col)
    names, est, draws = rho_table(D, col, genes, source_columns(D, genes, scores), Z, CNT)
    r, dl = contrast(names, est, draws, scores, T)
    for x in r:
        x.update(sens="vii_no_acinar", gold=gold, n_genes=len(genes))
        rows.append(x)
    rows.append(summarise("Delta(S)-Delta(B1)", dl["S"][0] - dl["B1"][0], dl["S"][1] - dl["B1"][1], score="S-B1", sens="vii_no_acinar", gold=gold, n_genes=len(genes)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "contrast_sensitivity.csv", index=False)
    print(t[~t.sens.str.startswith(("vi_seed", "iv_pairwise", "viii"))].round(3).to_string(), flush=True)


def run_within(D: Data, CNT) -> None:
    """(c2): between two tissues, within a gene: does the readout's tissue difference follow the independent gold's?"""
    rows, flips, h2h = [], [], []
    pairs = list(itertools.combinations(GOLDS, 2))
    for g1, g2 in pairs:
        T1, T2 = GOLDS[g1], GOLDS[g2]
        c1, c2 = f"G_{g1}", f"G_{g2}"
        g = D.gold[[c1, c2]].dropna().index
        m = D.sup.reindex(g)[[T1, T2]].fillna(False).all(axis=1) & D.raw["B1"].reindex(g)[[T1, T2]].notna().all(axis=1)
        g = m[m].index
        g = D.cov_T(T1, g)[PAT].dropna().index.intersection(D.cov_T(T2, g)[PAT].dropna().index)
        ext1 = ["H4_nsp"] if c1 == "G_testis" else []
        ext2 = ["H4_nsp"] if c2 == "G_testis" else []
        if ext1 or ext2:
            g = g[D.gold["H4_nsp"].reindex(g).notna()]

        def zT(y, T, c):
            Z = design_T(D, T, g, PAT, c)
            e = S.resid(S.midrank(y.reindex(g).to_numpy(float)), Z)
            return (e - e.mean()) / e.std()

        dG = zT(D.gold[c1], T1, c1) - zT(D.gold[c2], T2, c2)
        dd = {}
        for sc in ("S", "B1", "D", "Sdir"):
            dd[sc] = zT(D.raw[sc][T1], T1, c1) - zT(D.raw[sc][T2], T2, c2)
        A = np.column_stack([dG] + [dd[k] for k in dd])
        names = list(dd)
        def wr(w):
            mu = np.average(A, axis=0, weights=w)
            E = A - mu
            c = (E[:, 0] * w) @ E
            return (c / np.sqrt(c[0] * ((E * E * w[:, None]).sum(0))))[1:]
        est = wr(np.ones(len(g)))
        dr = np.stack([wr(weights(D, g, cnt)) for cnt in CNT])
        for i, n in enumerate(names):
            rows.append(summarise(f"{g1} vs {g2}", est[i], dr[:, i], score=n, n_genes=len(g), pair=f"{g1}|{g2}"))
        ii = names.index("S"), names.index("B1")
        rows.append(summarise(f"{g1} vs {g2}", est[ii[0]] - est[ii[1]], dr[:, ii[0]] - dr[:, ii[1]], score="S-B1", n_genes=len(g), pair=f"{g1}|{g2}"))
        # flip contrast: strong tissue difference in S (or B1), independent difference there
        for n in ("S", "B1", "Sdir"):
            x = dd[n]
            sd = x.std()
            hi, lo = x > sd, x < -sd
            def eff(w):
                return np.average(dG[hi], weights=w[hi]) - np.average(dG[lo], weights=w[lo])
            flips.append(summarise(f"{g1} vs {g2}", eff(np.ones(len(g))), np.array([eff(weights(D, g, cnt) + 1e-12) for cnt in CNT]), score=n,
                                   pair=f"{g1}|{g2}", n_hi=int(hi.sum()), n_lo=int(lo.sum())))
        # head-to-head on disagreement (X4's frame): S says T1 > T2 and B1 says T1 < T2, or the reverse; > 0 sides with S
        sS, sB = dd["S"], dd["B1"]
        ts, tb = 0.5 * sS.std(), 0.5 * sB.std()
        A1, A2 = (sS > ts) & (sB < -tb), (sS < -ts) & (sB > tb)
        def h(w):
            return np.average(dG[A1], weights=w[A1]) - np.average(dG[A2], weights=w[A2])
        h2h.append(summarise(f"{g1} vs {g2}", h(np.ones(len(g))), np.array([h(weights(D, g, cnt) + 1e-12) for cnt in CNT]), n_S_up=int(A1.sum()), n_S_dn=int(A2.sum()), pair=f"{g1}|{g2}"))
    t = pd.DataFrame(rows)
    for sc in t["score"].unique():
        m = t["score"] == sc
        t.loc[m, "q"] = S.bh(t.loc[m, "p"].tolist())
    t.to_csv(OUT / "within_gene.csv", index=False)
    pd.DataFrame(flips).to_csv(OUT / "within_flip_effect.csv", index=False)
    h2 = pd.DataFrame(h2h)
    h2["q"] = S.bh(h2["p"].tolist())
    h2.to_csv(OUT / "within_head_to_head.csv", index=False)
    print(t.round(3).to_string(), flush=True)
    print(h2.round(3).to_string(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--draws", type=int, default=RU.N_DRAWS)
    p.add_argument("--sections", default="main,sens,within")
    a = p.parse_args()
    D = Data()
    CNT = block_counts(D, a.draws, RU.SEED)
    for s in a.sections.split(","):
        {"main": run_main, "sens": run_sens, "within": run_within}[s](D, CNT)


if __name__ == "__main__":
    main()
