"""(d): stable and flipping genes, what moves in flips, the literature panel, and worked genes with counts.

    python -I launch.py analysis.tasks.Y1_gene_tissue.classes
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.Y1_gene_tissue import common as Y
from analysis.tasks.Y1_gene_tissue import prep as P
from analysis.tasks.Y1_gene_tissue import rules as RU

MAT, ORDER = P.MAT, RU.TISSUES
OUT = RU.OUT
GOLDT = {"liver": "G_liver", "testis": "G_testis", "kidney": "G_kidney", "islet": "G_pancreas"}


def classify(Z: pd.DataFrame, minT: int) -> pd.DataFrame:
    n = Z.notna().sum(axis=1)
    use = n >= minT
    mn, mx = Z.min(axis=1), Z.max(axis=1)
    cl = pd.Series("other", index=Z.index)
    cl[use & (mn >= RU.Z_STABLE)] = "shared_all"
    cl[use & (mx <= -RU.Z_STABLE)] = "not_shared_all"
    cl[use & (mx >= RU.Z_FLIP) & (mn <= -RU.Z_FLIP)] = "flip"
    cl[~use] = "too_few_tissues"
    return pd.DataFrame(dict(cls=cl, n_tissues=n, zmax=mx, zmin=mn, t_hi=Z.where(use).idxmax(axis=1), t_lo=Z.where(use).idxmin(axis=1)))


def expected(Z: pd.DataFrame, bins: pd.Series, minT: int, rng, n=RU.N_NULL) -> pd.DataFrame:
    """z permuted within tissue inside expression bins, keeping each gene's pattern of supported tissues."""
    A = Z.to_numpy()
    b = bins.to_numpy()
    rows = []
    for _ in range(n):
        B = A.copy()
        for j in range(A.shape[1]):
            ok = np.isfinite(A[:, j])
            for k in np.unique(b[ok]):
                i = np.flatnonzero(ok & (b == k))
                B[i, j] = A[rng.permutation(i), j]
        use = np.isfinite(B).sum(1) >= minT
        mn, mx = np.nanmin(B, 1), np.nanmax(B, 1)
        rows.append(dict(shared_all=(use & (mn >= RU.Z_STABLE)).sum(), not_shared_all=(use & (mx <= -RU.Z_STABLE)).sum(), flip=(use & (mx >= RU.Z_FLIP) & (mn <= -RU.Z_FLIP)).sum()))
    return pd.DataFrame(rows)


def class_enrichment(tab: pd.DataFrame, ann: pd.DataFrame, bins: pd.Series, score: str, rng) -> pd.DataFrame:
    rows = []
    uni = tab[tab["cls"] != "too_few_tissues"].index
    b = bins.reindex(uni).to_numpy()
    for cl in ("shared_all", "not_shared_all", "flip"):
        m = (tab.loc[uni, "cls"] == cl).to_numpy()
        for c in ann.columns:
            a = ann[c].reindex(uni).fillna(False).astype(bool).to_numpy()
            if a.sum() < 20:
                continue
            obs = int((m & a).sum())
            null = np.zeros(1000)
            for r in range(1000):
                perm = np.arange(len(a))
                for k in np.unique(b):
                    i = np.flatnonzero(b == k)
                    perm[i] = rng.permutation(i)
                null[r] = (m & a[perm]).sum()
            rows.append(dict(score=score, cls=cl, annotation=c.replace("cls_", ""), n_ann=int(a.sum()), obs=obs, exp=float(null.mean()), fold=obs / max(null.mean(), 1e-9),
                             p=float((1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / 1001)))
    t = pd.DataFrame(rows)
    t["q"] = S.bh(t["p"].tolist())
    return t


def shares(ds, dp, dl) -> tuple[float, float]:
    v = np.var(ds)
    return float(np.cov(ds, dp)[0, 1] / v), float(np.cov(ds, dl)[0, 1] / v)


def main():
    rng = np.random.default_rng(RU.SEED)
    sup = pd.read_parquet(MAT / "support.parquet")
    strict = pd.read_parquet(MAT / "strict.parquet")
    stable = pd.read_parquet(MAT / "sign_stable.parquet")
    sym = pd.read_parquet(MAT / "symbols.parquet")["symbol"]
    expr = pd.read_parquet(MAT / "cov_expr.parquet")
    bk = pd.read_parquet(MAT / "blocks.parquet")["block"]
    ranks = expr.rank(pct=True).mean(axis=1)
    bins = pd.qcut(ranks.rank(method="first"), 20, labels=False)
    zS = pd.read_parquet(MAT / "z_S__ED.parquet")
    zB = pd.read_parquet(MAT / "z_B1__ED.parquet")
    zD = pd.read_parquet(MAT / "z_D__ED.parquet")
    zP = pd.read_parquet(MAT / "z_Sdir_pattern__ED.parquet")
    zL = pd.read_parquet(MAT / "z_Sdir_level__ED.parquet")
    zSd = pd.read_parquet(MAT / "z_Sdir__ED.parquet")
    ctS, ctB = classify(zS, RU.MIN_TISSUES), classify(zB, RU.MIN_TISSUES)
    ctS.join(sym).to_parquet(OUT / "classes_S.parquet")
    ctB.join(sym).to_parquet(OUT / "classes_B1.parquet")
    rows = []
    for n, Z, ct in (("S", zS, ctS), ("B1", zB, ctB)):
        ex = expected(Z, bins, RU.MIN_TISSUES, rng)
        for c in ("shared_all", "not_shared_all", "flip"):
            o = int((ct["cls"] == c).sum())
            rows.append(dict(score=n, cls=c, n_universe=int((ct["cls"] != "too_few_tissues").sum()), obs=o, exp=float(ex[c].mean()), fold=o / max(ex[c].mean(), 1e-9),
                             p_hi=float((1 + (ex[c] >= o).sum()) / (1 + len(ex))), p_lo=float((1 + (ex[c] <= o).sum()) / (1 + len(ex)))))
    cnt = pd.DataFrame(rows)
    cnt.to_csv(OUT / "classes_counts.csv", index=False)
    print(cnt.round(3).to_string(), flush=True)
    xt = pd.crosstab(ctS["cls"], ctB["cls"])
    xt.to_csv(OUT / "classes_S_vs_B1.csv")
    print(xt, flush=True)
    # annotation classes from the X3_cross_cohort and X4_where_differs tables (Lambert, UniProt, HGNC)
    ann = []
    for c in ("liv_hmouse", "test_hmouse"):
        g = pd.read_parquet(Y.RES / "X4" / "genes" / f"{c}.parquet").set_index("hsapiens_gene_id")
        ann.append(g[[k for k in g.columns if k.startswith("cls_")]].astype(float))
    ann = pd.concat(ann).groupby(level=0).max() > 0
    en = pd.concat([class_enrichment(ctS, ann, bins, "S", rng), class_enrichment(ctB, ann, bins, "B1", rng)])
    en.to_csv(OUT / "classes_enrichment.csv", index=False)
    print(en[en.q < 0.1].round(3).to_string(), flush=True)

    # what moves in flips: S difference between the high and the low tissue, split by the direct analogue
    def dframe(ct, Z):
        d = ct[ct["cls"] != "too_few_tissues"]
        ii, hi, lo = d.index, d["t_hi"].to_numpy(), d["t_lo"].to_numpy()
        take = lambda M, tt: np.array([M.at[g, t] if t in M.columns else np.nan for g, t in zip(ii, tt)])  # noqa: E731
        return pd.DataFrame(dict(cls=d["cls"], t_hi=hi, t_lo=lo, ds=take(Z, hi) - take(Z, lo), dp=take(zP, hi) - take(zP, lo), dl=take(zL, hi) - take(zL, lo),
                                 p_lo=take(zP, lo), l_lo=take(zL, lo), p_hi=take(zP, hi), l_hi=take(zL, hi), dsd=take(zSd, hi) - take(zSd, lo)), index=ii).dropna()

    dS = dframe(ctS, zS)
    ub = np.unique(bk.to_numpy())
    rows = []
    def boot(sub):
        idx = np.searchsorted(ub, bk.reindex(sub.index).to_numpy())
        out = []
        for _ in range(1000):
            w = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub))[idx]
            s = np.repeat(np.arange(len(sub)), w)
            out.append(shares(sub["ds"].to_numpy()[s], sub["dp"].to_numpy()[s], sub["dl"].to_numpy()[s]))
        return np.array(out)
    for name, sub in [("flip", dS[dS.cls == "flip"]), ("not flip", dS[dS.cls != "flip"])] + [(f"flip, low in {t}", dS[(dS.cls == "flip") & (dS.t_lo == t)]) for t in ORDER] + \
            [(f"not flip, low in {t}", dS[(dS.cls != "flip") & (dS.t_lo == t)]) for t in ORDER]:
        if len(sub) < 10:
            rows.append(dict(set=name, n=len(sub)))
            continue
        sp, sl = shares(sub["ds"].to_numpy(), sub["dp"].to_numpy(), sub["dl"].to_numpy())
        b = boot(sub)
        rows.append(dict(set=name, n=len(sub), pattern_share=sp, pattern_lo=np.quantile(b[:, 0], .025), pattern_hi=np.quantile(b[:, 0], .975), level_share=sl, level_lo=np.quantile(b[:, 1], .025),
                         level_hi=np.quantile(b[:, 1], .975), frac_lo_level_gap=float((sub["l_lo"] <= -1).mean()), frac_lo_no_pattern=float((sub["p_lo"] <= -1).mean()),
                         frac_hi_level_gap=float((sub["l_hi"] <= -1).mean()), frac_hi_no_pattern=float((sub["p_hi"] <= -1).mean()), rho_S_Sdir_diff=float(np.corrcoef(sub["ds"], sub["dsd"])[0, 1])))
    fl = pd.DataFrame(rows)
    fl.to_csv(OUT / "flips_split.csv", index=False)
    print(fl.round(3).to_string(), flush=True)
    # moves over all tissue pairs: readout difference vs amplitude, level gap, expression differences
    amp = pd.read_parquet(MAT / "cov_amp.parquet")
    ash = pd.read_parquet(MAT / "cov_abs_shift.parquet")
    tau = pd.read_parquet(MAT / "cov_tau.parquet")
    rows = []
    for name, Z in (("S", zS), ("B1", zB), ("D", zD), ("Sdir", zSd)):
        for a, b in itertools.combinations(ORDER, 2):
            ok = Z[a].notna() & Z[b].notna()
            dz = (Z[a] - Z[b])[ok]
            r = dict(score=name, a=a, b=b, n=int(ok.sum()))
            for lab_, M, sgn in (("amp", amp, 1), ("abs_shift", ash, 1), ("expr", expr, 1), ("tau", tau, 1)):
                r[f"rho_{lab_}"] = float(pd.concat([dz, (M[a] - M[b])[ok]], axis=1).corr(method="spearman").iloc[0, 1])
            rows.append(r)
    mv = pd.DataFrame(rows)
    mv.to_csv(OUT / "flips_moves.csv", index=False)
    print(mv.groupby("score")[["rho_amp", "rho_abs_shift", "rho_expr", "rho_tau"]].agg(["mean", "min", "max"]).round(3).to_string(), flush=True)

    # literature panel
    golds = pd.read_parquet(MAT / "golds.parquet")
    inv = {v: k for k, v in sym.items() if isinstance(v, str)}
    rows = []
    for g_, (t, why) in RU.LIT_PANEL.items():
        gid = inv.get(g_)
        if gid is None or gid not in zS.index:
            rows.append(dict(gene=g_, expected_low=t, note="not in any vocabulary"))
            continue
        zs, zb = zS.loc[gid], zB.loc[gid]
        rec = dict(gene=g_, expected_low=t, support=";".join(c for c in ORDER if sup.at[gid, c]), strict=";".join(c for c in ORDER if strict.at[gid, c]), n_read=int(zs.notna().sum()))
        for T in ORDER:
            rec[f"zS_{T}"] = zs[T]
            rec[f"zB1_{T}"] = zb[T]
        if np.isfinite(zs.get(t, np.nan)) and zs.drop(t).notna().any():
            rec["dS"] = zs[t] - zs.drop(t).mean()
            rec["S_right_way"] = bool(rec["dS"] <= -0.5)
        if np.isfinite(zb.get(t, np.nan)) and zb.drop(t).notna().any():
            rec["dB1"] = zb[t] - zb.drop(t).mean()
            rec["B1_right_way"] = bool(rec["dB1"] <= -0.5)
        rows.append(rec)
    pd.DataFrame(rows).to_csv(OUT / "panel_literature.csv", index=False)
    print(pd.DataFrame(rows).round(2).to_string(), flush=True)

    # active scan: named lists and top flips with strict, sign-stable support in the tissues read
    def scan_rows(ids, source):
        out = []
        for gid in ids:
            zs = zS.loc[gid]
            st = strict.loc[gid]
            read = zs[st.reindex(zs.index).fillna(False)].dropna()
            if len(read) < 2:
                continue
            hi, lo = read.idxmax(), read.idxmin()
            rec = dict(gene=sym.get(gid), gid=gid, source=source, t_hi=hi, t_lo=lo, z_hi=read[hi], z_lo=read[lo], dz=read[hi] - read[lo], n_strict=len(read),
                       cls=ctS["cls"].get(gid), zB1_hi=zB.at[gid, hi], zB1_lo=zB.at[gid, lo], dB1=zB.at[gid, hi] - zB.at[gid, lo],
                       zP_hi=zP.at[gid, hi], zP_lo=zP.at[gid, lo], zL_hi=zL.at[gid, hi], zL_lo=zL.at[gid, lo])
            for T, c in GOLDT.items():
                rec[f"G_{T}"] = golds[c].get(gid, np.nan)
            if hi in GOLDT and lo in GOLDT:
                g1, g2 = golds[GOLDT[hi]].get(gid, np.nan), golds[GOLDT[lo]].get(gid, np.nan)
                rec["G_diff_hi_lo"] = g1 - g2
            out.append(rec)
        return out

    rows = []
    for nm, lst in (("regulator", RU.TISSUE_REGULATORS), ("metabolic", RU.METABOLIC), ("immune", RU.IMMUNE), ("panel", list(RU.LIT_PANEL))):
        rows += scan_rows([inv[g] for g in lst if g in inv and inv[g] in zS.index], nm)
    flips_ = ctS[ctS["cls"] == "flip"].index
    rows += scan_rows(flips_, "flip")
    sc = pd.DataFrame(rows).sort_values("dz", ascending=False)
    sc.to_csv(OUT / "scan_named.csv", index=False)
    print(sc[sc.source != "flip"].round(2).head(60).to_string(), flush=True)
    print(sc[sc.source == "flip"].round(2).head(40).to_string(), flush=True)
    ctS[ctS.cls == "shared_all"].join(sym).to_csv(OUT / "genes_shared_all.csv")
    ctS[ctS.cls == "not_shared_all"].join(sym).to_csv(OUT / "genes_not_shared_all.csv")


if __name__ == "__main__":
    main()
