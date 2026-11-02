"""Y5 additions fixed after F1 and T3.1-T3.2 were read, before these ran (ledger amendment).

F1c: GIN protein-only response c_esm and the ridge prediction on ESM against independent pattern golds, odd and even,
within and across tissues. Comparators: B3a (feature ridge on B1b), and the same regressor fit to B1b.
F7: paralog pairs (Compara, >= 60 % identity, one pair per gene, both in the cohort): protein language is fixed by design,
so does the difference in S, D, B1b within a pair follow the difference in the independent gold?

Usage: python -m revision.tasks.Y5_aspects.f17 f1c|f7
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import RidgeCV
from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.Y5_aspects import common as K, rules as RU

COH = RU.PRIMARY + RU.SENS_RAWTAIL


def f1c():
    tabs = {}
    for c in COH:
        g = K.genes(c)
        x3 = pd.read_parquet(K.REPO / f"results/revision/X3/genes/{c}.parquet")
        E = x3[[f"esm_{i}" for i in range(64)]].to_numpy(float)
        for y in ("B1b", "S"):
            m = (~g.masked.astype(bool)) & g[[y] + K.C4].notna().all(1)
            T = g[m]
            res = pd.Series(np.nan, index=g.index)
            res[m] = ST.resid(ST.midrank(T[y].to_numpy(float)), ST.design(T, K.C4))
            tr = m & (g.half == "odd")
            g[f"pred_{y}"] = RidgeCV(alphas=np.logspace(-1, 5, 25)).fit(E[tr], res[tr]).predict(E)
        tabs[c] = g
    rows = []
    golds = {"G_H3": "liver", "G_H4": "testis", "G_H4_primates": "testis"}
    for train in COH:
        src = tabs[train][["hsapiens_gene_id", "c_esm", "c_expr", "pred_B1b", "pred_S", "D_raw", "has_odd_paralog"]].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
        for tgt in COH:
            g = tabs[tgt]
            for gold, tis in golds.items():
                if gold not in g or (tis == "liver") != tgt.startswith("liv"):
                    continue
                q = g[["hsapiens_gene_id", "half", "block", "tau_shared", "H4_nsp", "B3a", gold] + K.C4].copy()
                for k in src.columns:
                    q["t_" + k] = q.hsapiens_gene_id.map(src[k])
                covs = K.CPAT + (["H4_nsp"] if gold.startswith("G_H4") else [])
                ys = ["t_c_esm", "t_c_expr", "t_pred_B1b", "t_pred_S", "t_D_raw", "B3a"]
                for half in ("even", "odd"):
                    T = q[(q.half == half) & q[[gold, "block"] + covs + ys].notna().all(1)]
                    r = K.scan(T, gold, ys, covs, seed=K.useed(train, tgt, gold, half))
                    r["train"], r["target"], r["gold"], r["half"] = train, tgt, gold, half
                    rows.append(r)
                    if half == "even":
                        Tg = T[~T.t_has_odd_paralog.astype(bool)]
                        r = K.scan(Tg, gold, ys, covs, seed=K.useed(train, tgt, gold, half, "g"), draws=200)
                        r["train"], r["target"], r["gold"], r["half"] = train, tgt, gold, "even_noparalog"
                        rows.append(r)
    pd.concat(rows, ignore_index=True).to_csv(K.OUT / "f1c_gold.csv", index=False)


def f7():
    co = pd.read_csv(K.OUT / "compara_human.tsv", sep="\t", names=["g", "id", "typ", "h", "sp", "hid"])
    par = co[(co.sp == "homo_sapiens") & co.typ.str.contains("paralog") & (co.id >= 60) & (co.hid >= 60)]
    rows, shown = [], []
    for c, gold in (("liv_hmouse", "G_H3"), ("liv_4sp", "G_H3"), ("test_3sp", "G_H4"), ("test_hmouse", "G_H4")):
        g = K.genes(c)
        g = g[(~g.masked.astype(bool))].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
        ok = g[[gold] + K.C4].notna().all(1)
        p = par[par.g.isin(g.index[ok]) & par.h.isin(g.index[ok]) & (par.g < par.h)].sort_values("id", ascending=False)
        used, keep = set(), []
        for r in p.itertuples():
            if r.g in used or r.h in used:
                continue
            used |= {r.g, r.h}
            keep.append((r.g, r.h, r.id))
        P = pd.DataFrame(keep, columns=["a", "b", "id"])
        ga, gb = g.loc[P.a], g.loc[P.b]
        cols = [gold, "S", "D_raw", "B1b", "Sdir", "c_esm", "c_expr", "log_cp10k_spmean", "detection_spmean", "prior_raw", "log_cds_length"]
        D = pd.DataFrame({k: ga[k].to_numpy(float) - gb[k].to_numpy(float) for k in cols if k in g.columns})
        D["block"] = np.arange(len(D))
        sim = np.corrcoef(np.column_stack([ga.c_esm, gb.c_esm]).T)[0, 1]
        Z = np.column_stack([np.ones(len(D)), D.log_cp10k_spmean, D.detection_spmean, D.prior_raw, D.log_cds_length])
        for y in ("S", "D_raw", "B1b", "Sdir", "c_esm"):
            if y not in D:
                continue
            m = D[[gold, y]].notna().all(1).to_numpy()
            yy, gg, ZZ = D[y][m].to_numpy(), D[gold][m].to_numpy(), Z[m]
            e1 = ST.resid(ST.midrank(yy), ZZ)
            e2 = ST.resid(ST.midrank(gg), ZZ)
            est = np.corrcoef(e1, e2)[0, 1]
            rng = np.random.default_rng(K.useed(c, y))
            bs = []
            for _ in range(1000):
                i = rng.integers(0, len(e1), len(e1))
                bs.append(np.corrcoef(e1[i], e2[i])[0, 1])
            rows.append(dict(cohort=c, gold=gold, score=y, n_pairs=int(m.sum()), rho_p=est, lo=np.quantile(bs, .025), hi=np.quantile(bs, .975), p=float(2 * norm.sf(abs(est) / np.std(bs))), between_pair_corr_c_esm=sim))
    pd.DataFrame(rows).to_csv(K.OUT / "f7_paralogs.csv", index=False)


def f1d():
    """Properties against D in the existing control runs: is the relation learned from counts or from the features?"""
    from revision.common import cohorts as C
    arms = {"liv_hmouse": ["m", "rand", "zesm", "zexpr", "permC"], "liv_4sp": ["rand", "zesm", "zexpr", "permA"],
            "test_hmouse": ["rand", "permC"], "test_3sp": ["nolab"]}
    rows = []
    props = list(RU.PROPERTIES)
    for c, ar in arms.items():
        g = K.genes(c).copy()
        for arm in ar:
            runs = sorted(p.name for p in C.RUNS_DIR.glob(f"rev-ctl-{c}-{arm}-s*"))
            vals, Ss = [], []
            for r in runs:
                st = C.load_state(r)
                if not (C.RUNS_DIR / r / "summary.json").exists() or not np.isfinite(st["alpha"]).all():
                    continue
                d = np.asarray(st["alpha"], float) - g.prior_raw.to_numpy()
                vals.append(d - np.nanmedian(d))
                Ss.append(np.asarray(st["S_g_ema"], float))
            if not vals:
                continue
            g[f"D_{arm}"] = np.mean(vals, 0)
            g[f"S_{arm}"] = np.mean(Ss, 0)
            ys = [f"D_{arm}", f"S_{arm}", "D_raw", "S"]
            for prop in props:
                covs = K.C3 if prop == "log_length" else K.C4
                for half in ("odd", "even"):
                    m = (~g.masked.astype(bool)) & (g.half == half) & g[[prop, "block"] + covs + ys].notna().all(1)
                    r2 = K.scan(g[m], prop, ys, covs, seed=K.useed(c, arm, prop, half), draws=200)
                    r2["cohort"], r2["arm"], r2["n_runs"], r2["half"], r2["property"] = c, arm, len(vals), half, prop
                    rows.append(r2)
            print(c, arm, len(vals), flush=True)
    pd.concat(rows, ignore_index=True).to_csv(K.OUT / "f1d_controls.csv", index=False)


if __name__ == "__main__":
    dict(f1c=f1c, f7=f7, f1d=f1d)[sys.argv[1]]()
