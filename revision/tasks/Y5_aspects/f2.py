"""Y5 F2: phylogenetic depth. Independent axis: Murat held-out species only.

Sections: gold, tests, programs. Usage: python -m revision.tasks.Y5_aspects.f2 <section>
"""
from __future__ import annotations
import itertools
import sys
import warnings
import numpy as np
import pandas as pd
from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.Y5_aspects import common as K, rules as RU

OUT = K.OUT / "F2"


def gold():
    from revision.tasks.T4_3_scoreboard import golds as TG
    OUT.mkdir(parents=True, exist_ok=True)
    inp = TG.h4_inputs()
    res = TG.h4_species_profiles(inp)
    n = len(inp["genes"])
    pairs = {}
    for s1, s2 in itertools.combinations(TG.PRIMATES + TG.OUTGROUPS, 2):
        p1, ok1 = res[s1]
        p2, ok2 = res[s2]
        ok = ok1 & ok2
        r = np.full(n, np.nan)
        if ok.any():
            r[ok] = TG.profile_corr(p1[ok], p2[ok], np.ones(ok.sum(), bool), np.ones(ok.sum(), bool))
        my = RU.MYA.get((s1, s2), RU.MYA.get((s2, s1)))
        pairs[(s1, s2, my)] = np.arctanh(np.clip(r, -0.999999, 0.999999))
    d = pd.DataFrame({"hsapiens_gene_id": inp["genes"]})
    yrs = np.array([k[2] for k in pairs])
    Z = np.column_stack(list(pairs.values()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for b, (lo, hi) in RU.DEPTH_BINS.items():
            m = (yrs >= lo) & (yrs < hi)
            d[f"G_{b}"] = np.nanmean(Z[:, m], 1)
            d[f"n_{b}"] = np.isfinite(Z[:, m]).sum(1)
    # slope per gene on log10 time, >= 8 pairs spanning at least 2 of the deep bins
    lt = np.log10(yrs)
    slope = np.full(n, np.nan)
    for i in range(n):
        ok = np.isfinite(Z[i])
        if ok.sum() >= 8 and (yrs[ok] > 150).sum() >= 3 and (yrs[ok] < 45).sum() >= 3:
            slope[i] = np.polyfit(lt[ok], Z[i, ok], 1)[0]
    d["slope"] = slope
    d["n_species"] = np.sum([res[s][1] for s in TG.PRIMATES + TG.OUTGROUPS], 0)
    d.to_parquet(OUT / "gold_depth.parquet")
    print(d.describe().T[["count", "mean", "std"]])


def table(c):
    g = K.genes(c).merge(pd.read_parquet(OUT / "gold_depth.parquet"), on="hsapiens_gene_id", how="left")
    drop = [k for k in g.columns if k.startswith(("C_", "Csp_"))]
    g = g.drop(columns=drop).merge(K.contribs(c), on="var_name", how="left")
    return g


def multi_gold(T, golds, scores, covs, draws=1000, seed=0):
    """rho_p of every score with every gold on the same genes and the same bootstrap draws."""
    Z = ST.design(T, covs)
    Rs = np.column_stack([ST.midrank(T[s].to_numpy(float)) for s in scores])
    Rg = {g: ST.midrank(T[g].to_numpy(float)) for g in golds}
    est = {g: ST.rho_p_matrix(np.column_stack([Rg[g], Rs]), Z)[1:] for g in golds}
    rng = np.random.default_rng(seed)
    block = T["block"].to_numpy()
    D = {g: np.empty((draws, len(scores))) for g in golds}
    for b in range(draws):
        w = ST.block_weights(block, rng)
        for g in golds:
            D[g][b] = ST.weighted_rho(np.column_stack([Rg[g], Rs]), Z, w)[1:]
    return est, D


def tests():
    rows, slopes, cond = [], [], []
    bins = [f"G_{b}" for b in RU.DEPTH_BINS]
    yrs = np.log10(np.array([6, 30, 170, 318.0]))
    for c in ("test_3sp", "test_hmouse"):
        g = table(c)
        scores = ["S", "D_raw", "B1b", "B1c", "Sdir", "B1z"]
        covs = K.CPAT + ["n_species"]
        for half in ("odd", "even"):
            T = g[(g.half == half) & ~g.masked.astype(bool) & g[bins + scores + covs + ["block"]].notna().all(1)]
            est, D = multi_gold(T, bins, scores, covs, seed=K.useed(c, half))
            for k, s in enumerate(scores):
                for gb in bins:
                    lo, hi = ST.pct_ci(D[gb][:, k])
                    rows.append(dict(cohort=c, half=half, score=s, gold=gb, rho_p=est[gb][k], lo=lo, hi=hi,
                                     p=float(2 * __import__("scipy.stats").stats.norm.sf(abs(est[gb][k]) / D[gb][:, k].std())), n=len(T)))
                sl = np.polyfit(yrs, np.column_stack([D[gb][:, k] for gb in bins]).T, 1)[0]
                slopes.append(dict(cohort=c, half=half, score=s, slope=np.polyfit(yrs, [est[gb][k] for gb in bins], 1)[0], lo=ST.pct_ci(sl)[0], hi=ST.pct_ci(sl)[1]))
                # difference in decay to B1b on the same draws
                kb = scores.index("B1b")
                slb = np.polyfit(yrs, np.column_stack([D[gb][:, kb] for gb in bins]).T, 1)[0]
                sd = sl - slb
                slopes[-1].update(diff_vs_B1b=float(np.polyfit(yrs, [est[gb][k] - est[gb][kb] for gb in bins], 1)[0]), diff_lo=ST.pct_ci(sd)[0], diff_hi=ST.pct_ci(sd)[1])
            # deep pattern given shallow pattern: what holds beyond what is shared among close species
            for deep in ("G_318", "G_160_180"):
                for extra in ([], ["identity_chicken"] if deep == "G_318" else ["identity_opossum"]):
                    Tq = T[T[["G_lt10", "G_10_45"] + extra].notna().all(1)]
                    r = K.scan(Tq, deep, scores + ["prior_raw"] if False else scores, covs + ["G_lt10", "G_10_45"] + extra, seed=K.useed(c, half, deep, str(extra)))
                    r["cohort"], r["half"], r["deep"], r["with_identity"] = c, half, deep, bool(extra)
                    cond.append(r)
            # per-gene decay slope
            Ts = g[(g.half == half) & ~g.masked.astype(bool) & g[["slope"] + scores + covs + ["block"]].notna().all(1)]
            r = K.scan(Ts, "slope", scores, covs, seed=K.useed(c, half, "slope"))
            r["cohort"], r["half"], r["deep"], r["with_identity"] = c, half, "slope", False
            cond.append(r)
            Ts = Ts[Ts.identity_chicken.notna()]
            r = K.scan(Ts, "slope", scores, covs + ["identity_chicken"], seed=K.useed(c, half, "slope", 1))
            r["cohort"], r["half"], r["deep"], r["with_identity"] = c, half, "slope", True
            cond.append(r)
            # protein identity itself on the same axis
            Tp = g[(g.half == half) & g[["slope", "identity_chicken", "block", "log_cp10k_spmean", "detection_spmean", "n_species", "log_cds_length", "tau_shared"] + bins].notna().all(1)]
            for y in ["slope"] + bins:
                r = K.scan(Tp, y, ["identity_chicken", "identity_opossum"] if Tp.identity_opossum.notna().all() else ["identity_chicken"], ["log_cp10k_spmean", "detection_spmean", "log_cds_length", "tau_shared", "n_species"], seed=K.useed(c, half, y, "id"), draws=200)
                r["cohort"], r["half"], r["deep"], r["with_identity"] = c, half, y, "identity_axis"
                cond.append(r)
        # class by depth, on the independent gold
        cls_rows = []
        for cl in RU.DEPTH_CLASSES:
            for half in ("odd", "even"):
                T = g[(g.half == half) & g[bins + [cl, "block"] + K.CPAT + ["n_species"]].notna().all(1)].copy()
                T[cl] = T[cl].astype(float)
                out = K.scan(T, cl, bins + ["S", "D_raw", "B1b"], K.CPAT + ["n_species"], seed=K.useed(c, cl, half))
                out["cohort"], out["half"], out["cls"] = c, half, cl
                cls_rows.append(out)
        pd.concat(cls_rows, ignore_index=True).to_csv(OUT / f"class_depth__{c}.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "depth_rho.csv", index=False)
    pd.DataFrame(slopes).to_csv(OUT / "depth_slopes.csv", index=False)
    pd.concat(cond, ignore_index=True).to_csv(OUT / "depth_conditional.csv", index=False)


def programs():
    """Which programs lose S from a close to a far partner, and is it the same across tissues."""
    from revision.tasks.T6_common import functional as FN
    from revision.tasks.X7_architecture_outputs.forward import OUT as X7
    T = {}
    g = table("liv_4sp")
    g["dS"] = g["Csp_Mouse"] - g["Csp_Monkey"]
    pe = pd.read_parquet(X7 / "baselines" / "liv_4sp__pair_E.parquet")
    print(pe.columns.tolist()[:10], flush=True)
    g = g.merge(pe, on="var_name", how="left") if "var_name" in pe else g
    T["liver"] = g
    g = table("test_3sp")
    g["dS"] = g["Csp_Mouse"] - g["Csp_Macaque"]
    pe = pd.read_parquet(X7 / "baselines" / "test_3sp__pair_E.parquet")
    g = g.merge(pe, on="var_name", how="left") if "var_name" in pe else g
    T["testis"] = g
    a = pd.read_parquet(K.REPO / "results/revision/T11/readout/genes_hmouse.parquet")
    b = pd.read_parquet(K.REPO / "results/revision/T11/readout/genes_hmmul.parquet")
    m = a.merge(b[["hsapiens_gene_id", "S_ema", "E_human_mmul" if "E_human_mmul" in b else "B1_raw"]], on="hsapiens_gene_id", suffixes=("", "_mmul"))
    m["dS"] = m["S_ema"] - m["S_ema_mmul"]
    chrom = pd.read_parquet(K.OUT / "chromosomes.parquet").set_index("hsapiens_gene_id").chrom
    m["half"] = np.where(m.hsapiens_gene_id.map(chrom).isin(RU.ODD), "odd", np.where(m.hsapiens_gene_id.map(chrom).isin(RU.EVEN), "even", "other"))
    m["masked"] = (m.S_ema == 0) | (m.S_ema_mmul == 0)
    m["dE"] = m["B1_raw"] - m["B1_raw_mmul"]
    T["retina"] = m
    sets = FN.raw_libraries()
    sets = {**{f"H:{k}": v for k, v in sets["Hallmark"].items()}, **{f"R:{k}": v for k, v in sets["Reactome"].items()}}
    rows = []
    for tis, g in T.items():
        if tis == "liver":
            g["dE"] = g["E_Human_Mouse"] - g["E_Human_Monkey"] if "E_Human_Mouse" in g else np.nan
        if tis == "testis":
            g["dE"] = g["E_Human_Mouse"] - g["E_Human_Macaque"] if "E_Human_Mouse" in g else np.nan
        for half in ("odd", "even"):
            q = g[(g.half == half) & ~g.masked.astype(bool) & g[K.C4].notna().all(1)]
            for y in ("dS", "dE"):
                if q[y].notna().sum() < 500:
                    continue
                qq = q[q[y].notna()]
                e = ST.resid(ST.midrank(qq[y].to_numpy(float)), ST.design(qq, K.C4))
                e = pd.Series(e / e.std(), index=qq.hsapiens_gene_id)
                e = e[~e.index.duplicated()]
                for name, v in sets.items():
                    ids = e.index.intersection(list(v))
                    if len(ids) >= 8 and len(ids) <= 300:
                        rows.append(dict(tissue=tis, half=half, y=y, set=name, n=len(ids), mean=float(e.loc[ids].mean())))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "programs_delta.csv", index=False)


if __name__ == "__main__":
    dict(gold=gold, tests=tests, programs=programs)[sys.argv[1]]()
