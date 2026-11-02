"""Y1 post hoc additions A-G (ledger, written after the first reading). Section names as in the ledger.

    python -I launch.py revision.tasks.Y1_gene_tissue.posthoc [--sections A,B,C,D,E,F,G]
"""

from __future__ import annotations

import argparse
import io
import itertools
import zipfile

import numpy as np
import pandas as pd

from revision.tasks.T4_3_scoreboard import stats as S
from revision.tasks.Y1_gene_tissue import common as Y
from revision.tasks.Y1_gene_tissue import contrast as CT
from revision.tasks.Y1_gene_tissue import prep as P
from revision.tasks.Y1_gene_tissue import rules as RU

MAT, ORDER = P.MAT, RU.TISSUES
OUT = RU.OUT / "posthoc"
PAT, LVL = CT.PAT, ["prior_raw", "expr", "det", "log_cds_length"]
NEW = {"kidney_fine": "kidney", "kidney_nephron": "kidney", "islet_fine": "islet", "islet_noacinar": "islet"}
LINEAGE_GOLD = {"liver": ["hepatocyte", "myeloid", "endothelial", "lymphoid", "lymphoid", "stromal"], "testis": ["germ", "germ", "germ", "stromal"],
                "kidney": ["epithelial", "endothelial", "myeloid", "lymphoid", "lymphoid", "lymphoid"],
                "pancreas": ["exocrine", "epithelial", "endothelial", "endocrine", "endocrine", "myeloid"]}
LINEAGE_SRC = {"liver": {"hepatocyte", "epithelial", "endothelial", "stromal", "myeloid", "lymphoid"}, "testis": {"germ", "stromal"},
               "kidney": {"epithelial", "endothelial", "stromal", "myeloid", "lymphoid"}, "islet": {"endocrine", "epithelial", "endothelial", "stromal", "myeloid", "lymphoid"},
               "bone_marrow": {"myeloid", "lymphoid", "erythroid"}, "eye": {"epithelial", "endothelial", "stromal", "melanocyte", "myeloid", "lymphoid"}}


def data() -> CT.Data:
    D = CT.Data()
    for tag in NEW:
        g = pd.read_parquet(RU.OUT / "gold" / f"gold_{tag}.parquet")
        D.gold = D.gold.join(g.rename(columns=dict(G=f"G_{tag}", atlas_amp=f"amp_{tag}", atlas_expr=f"aexpr_{tag}", atlas_det=f"adet_{tag}")), how="outer")
    return D


def z_gold_or_score(D, y: pd.Series, genes, T, col, cols, atlas=None) -> np.ndarray:
    ex = None
    if atlas is not None:
        ex = D.gold[atlas].reindex(genes).assign(amp_T=D.cov["amp"][T].reindex(genes))
    Z = CT.design_T(D, T, genes, cols, col, ex)
    e = S.resid(S.midrank(y.reindex(genes).to_numpy(float)), Z)
    return (e - e.mean()) / e.std()


def atlas_cols(gold):
    return [f"amp_{gold}", f"aexpr_{gold}"] + ([f"adet_{gold}"] if gold != "testis" else [])


def wcorr(A, w):
    mu = np.average(A, axis=0, weights=w)
    E = A - mu
    c = (E[:, 0] * w) @ E
    return (c / np.sqrt(c[0] * ((E * E * w[:, None]).sum(0))))[1:]


def within(D, CNT, pairs, tag, matched: bool) -> list:
    rows = []
    for (g1, T1), (g2, T2) in pairs:
        c1, c2 = f"G_{g1}", f"G_{g2}"
        g = D.gold[[c1, c2]].dropna().index
        m = D.sup.reindex(g)[[T1, T2]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[[T1, T2]].notna().all(axis=1)
        g = m[m].index
        g = D.cov_T(T1, g)[PAT + ["amp"]].dropna().index.intersection(D.cov_T(T2, g)[PAT + ["amp"]].dropna().index)
        if "G_testis" in (c1, c2):
            g = g[D.gold["H4_nsp"].reindex(g).notna()]
        at1, at2 = (atlas_cols(g1), atlas_cols(g2)) if matched else (None, None)
        if matched:
            g = g[D.gold[at1 + at2].reindex(g).notna().all(axis=1)]
        zT = lambda y, T, c, at: z_gold_or_score(D, y, g, T, c, PAT, at)  # noqa: E731
        dG = zT(D.gold[c1], T1, c1, at1) - zT(D.gold[c2], T2, c2, at2)
        dd = {sc: zT(D.raw[sc][T1], T1, c1, at1) - zT(D.raw[sc][T2], T2, c2, at2) for sc in ("S", "B1", "D", "Sdir")}
        A = np.column_stack([dG] + list(dd.values()))
        est = wcorr(A, np.ones(len(g)))
        dr = np.stack([wcorr(A, CT.weights(D, g, c) + 1e-12) for c in CNT])
        nm = list(dd)
        for i, n in enumerate(nm):
            rows.append(CT.summarise(f"{g1} vs {g2}", est[i], dr[:, i], score=n, pair=f"{g1}|{g2}", n_genes=len(g), variant=tag))
        rows.append(CT.summarise(f"{g1} vs {g2}", est[0] - est[1], dr[:, 0] - dr[:, 1], score="S-B1", pair=f"{g1}|{g2}", n_genes=len(g), variant=tag))
    return rows


def sec_A(D, CNT):
    G4 = [("liver", "liver"), ("testis", "testis"), ("kidney", "kidney"), ("pancreas", "islet")]
    r = within(D, CNT, list(itertools.combinations(G4, 2)), "matched_variability", True)
    t = pd.DataFrame(r)
    for sc in t.score.unique():
        m = t.score == sc
        t.loc[m, "q"] = S.bh(t.loc[m, "p"].tolist())
    t.to_csv(OUT / "A_within_matched.csv", index=False)
    print(t.assign(v=t.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f}) q{x.q:.3f}", axis=1)).pivot(index="pair", columns="score", values="v").to_string(), t.groupby("pair").n_genes.first().to_dict(), flush=True)


def evee(D):
    z = zipfile.ZipFile(Y.ROOT / "data" / "revision" / "gold" / "G1" / "evolutionary_statistics.zip")
    var = pd.read_csv(io.BytesIO(z.read("evolutionary_statistics/var.txt")), sep="\t")
    mean = pd.read_csv(io.BytesIO(z.read("evolutionary_statistics/mean.txt")), sep="\t")
    m = pd.read_csv(Y.RES / "T4_1" / "evee_symbol_to_ensg.tsv", sep="\t")
    ok = m.ensg.notna().to_numpy()
    out = {}
    for t in ("liver", "testis", "kidney"):
        v = var[t].to_numpy(float)
        out[f"G_ev_{t}"] = pd.Series(np.where(v > 0, -np.log(np.where(v > 0, v, 1.0)), np.nan), index=m.ensg)[ok]
        out[f"evm_{t}"] = pd.Series(np.where(v > 0, mean[t].to_numpy(float), np.nan), index=m.ensg)[ok]
    E = pd.DataFrame(out)
    E = E[~E.index.duplicated()]
    if "G_ev_liver" not in D.gold.columns:
        D.gold = D.gold.join(E, how="outer")
    return list(out)


def sec_B(D, CNT):
    evee(D)
    D.raw["neg_shift"] = -pd.read_parquet(MAT / "cov_abs_shift.parquet")
    D.raw["Sdir_level"] = pd.read_parquet(MAT / "raw_Sdir_level.parquet")
    scores = ["S", "B1", "D", "Sdir", "Sdir_level", "Sdir_pattern", "neg_shift"]
    rows, rho = [], []
    for T in ("liver", "testis", "kidney"):
        col = f"G_ev_{T}"
        for tag, universe in (("all_six", None), ("pair", 1)):
            if tag == "all_six":
                g = D.gold[col].dropna().index
                m = D.sup.reindex(g).astype(bool).all(axis=1) & D.raw["B1"].reindex(g).notna().all(axis=1)
                g = m[m].index
                g = D.cov_T(T, g)[LVL].dropna().index
                Z = CT.design_T(D, T, g, LVL, col)
                names, est, dr = CT.rho_table(D, col, g, CT.source_columns(D, g, scores), Z, CNT)
                for n, e, d in zip(names, est, dr.T):
                    sc, t = n.split("|")
                    rho.append(CT.summarise(n, e, d, gold=T, score=sc, source=t, n_genes=len(g)))
                r_, dl = CT.contrast(names, est, dr, scores, T)
                for x in r_:
                    x.update(gold=T, n_genes=len(g), universe=tag)
                    rows.append(x)
                rows.append(CT.summarise("Delta(S)-Delta(B1)", dl["S"][0] - dl["B1"][0], dl["S"][1] - dl["B1"][1], score="S-B1", gold=T, n_genes=len(g), universe=tag))
                rows.append(CT.summarise("Delta(S)-Delta(neg_shift)", dl["S"][0] - dl["neg_shift"][0], dl["S"][1] - dl["neg_shift"][1], score="S-neg_shift", gold=T, n_genes=len(g), universe=tag))
    t = pd.DataFrame(rows)
    t["q"] = S.bh(t["p"].tolist())
    t.to_csv(OUT / "B_evee_delta.csv", index=False)
    pd.DataFrame(rho).to_csv(OUT / "B_evee_rho.csv", index=False)
    print(t[["gold", "name", "est", "lo", "hi", "q", "n_genes"]].round(3).to_string(), flush=True)
    # within-gene, three pairs, S vs EVEE tissue difference
    wr = []
    for T1, T2 in itertools.combinations(("liver", "testis", "kidney"), 2):
        c1, c2 = f"G_ev_{T1}", f"G_ev_{T2}"
        g = D.gold[[c1, c2]].dropna().index
        m = D.sup.reindex(g)[[T1, T2]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[[T1, T2]].notna().all(axis=1)
        g = m[m].index
        g = D.cov_T(T1, g)[LVL].dropna().index.intersection(D.cov_T(T2, g)[LVL].dropna().index)
        def z(y, T, c):
            return z_gold_or_score(D, y, g, T, c, LVL)
        dG = z(D.gold[c1], T1, c1) - z(D.gold[c2], T2, c2)
        dd = {sc: z(D.raw[sc][T1], T1, c1) - z(D.raw[sc][T2], T2, c2) for sc in scores}
        A = np.column_stack([dG] + list(dd.values()))
        est = wcorr(A, np.ones(len(g)))
        dr = np.stack([wcorr(A, CT.weights(D, g, c) + 1e-12) for c in CNT])
        for i, n in enumerate(dd):
            wr.append(CT.summarise(f"{T1} vs {T2}", est[i], dr[:, i], score=n, n_genes=len(g)))
    w = pd.DataFrame(wr)
    w["q"] = S.bh(w["p"].tolist())
    w.to_csv(OUT / "B_evee_within.csv", index=False)
    print(w.assign(v=w.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot(index="name", columns="score", values="v").to_string(), w.groupby("name").n_genes.first().to_dict(), flush=True)


def sec_C(D, CNT):
    scores = ["S", "B1", "D", "Sdir"]
    rows, rho = [], []
    for gold, T in NEW.items():
        col = f"G_{gold}"
        # primary universe, and pairwise mean; variability matched
        for tag in ("all_six", "matched_variability", "pairwise"):
            if tag == "pairwise":
                for sc in scores:
                    own, ownD = [], []
                    for t in ORDER:
                        if t == T:
                            continue
                        g = D.gold[col].dropna().index
                        m = D.sup.reindex(g)[[T, t]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[[T, t]].notna().all(axis=1)
                        g = m[m].index
                        g = D.cov_T(T, g)[PAT].dropna().index
                        Z = CT.design_T(D, T, g, PAT, col)
                        _, e_, d_ = CT.rho_table(D, col, g, {"own": D.raw[sc][T].reindex(g), "oth": D.raw[sc][t].reindex(g)}, Z, CNT)
                        own.append(e_[0] - e_[1])
                        ownD.append(d_[:, 0] - d_[:, 1])
                    rows.append(CT.summarise(f"Delta({sc})", float(np.mean(own)), np.mean(ownD, axis=0), score=sc, gold=gold, universe=tag))
                continue
            g = CT.universe(D, T, col)
            ex = None
            if tag == "matched_variability":
                at = atlas_cols(gold)
                g = g[D.gold[at].reindex(g).notna().all(axis=1) & D.cov["amp"][T].reindex(g).notna().to_numpy()]
                ex = D.gold[at].reindex(g).assign(amp_T=D.cov["amp"][T].reindex(g))
            Z = CT.design_T(D, T, g, PAT, col, ex)
            names, est, dr = CT.rho_table(D, col, g, CT.source_columns(D, g, scores), Z, CNT)
            if tag == "all_six":
                for n, e, d in zip(names, est, dr.T):
                    sc, t = n.split("|")
                    rho.append(CT.summarise(n, e, d, gold=gold, score=sc, source=t, matched=t == T, n_genes=len(g)))
            r_, dl = CT.contrast(names, est, dr, scores, T)
            for x in r_:
                x.update(gold=gold, n_genes=len(g), universe=tag)
                rows.append(x)
            rows.append(CT.summarise("Delta(S)-Delta(B1)", dl["S"][0] - dl["B1"][0], dl["S"][1] - dl["B1"][1], score="S-B1", gold=gold, n_genes=len(g), universe=tag))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "C_fine_delta.csv", index=False)
    rt = pd.DataFrame(rho)
    rt.to_csv(OUT / "C_fine_rho.csv", index=False)
    print(t.assign(v=t.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot_table(index=["gold", "universe"], columns="score", values="v", aggfunc="first").to_string(), flush=True)
    for sc in ("S", "B1"):
        print(sc, rt[rt.score == sc].pivot(index="source", columns="gold", values="est").reindex(ORDER).round(3).to_string(), flush=True)
    # incremental on new golds
    zS, zB = pd.read_parquet(MAT / "z_S__ED.parquet"), pd.read_parquet(MAT / "z_B1__ED.parquet")
    inc = []
    for gold, T in NEW.items():
        col = f"G_{gold}"
        g = CT.universe(D, T, col)
        oth = [t for t in ORDER if t != T]
        ex = pd.DataFrame(dict(B1_T=D.raw["B1"][T].reindex(g), S_T=D.raw["S"][T].reindex(g), others_S=zS[oth].reindex(g).mean(axis=1), others_B1=zB[oth].reindex(g).mean(axis=1)))
        for nm, x, cond in (("S_T", "S_T", []), ("S_T | B1_T", "S_T", ["B1_T"]), ("S_T | B1_T, others S", "S_T", ["B1_T", "others_S"]), ("B1_T", "B1_T", []), ("B1_T | S_T", "B1_T", ["S_T"]), ("B1_T | S_T, others B1", "B1_T", ["S_T", "others_B1"])):
            Z = CT.design_T(D, T, g, PAT, col, ex[cond] if cond else None)
            _, e_, d_ = CT.rho_table(D, col, g, {x: ex[x]}, Z, CNT)
            inc.append(CT.summarise(nm, e_[0], d_[:, 0], gold=gold, n_genes=len(g)))
    it = pd.DataFrame(inc)
    it.to_csv(OUT / "C_fine_incremental.csv", index=False)
    print(it.assign(v=it.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot(index="name", columns="gold", values="v").to_string(), flush=True)
    # within-gene pairs with the finer golds
    G4 = [("liver", "liver"), ("testis", "testis"), ("kidney_fine", "kidney"), ("islet_fine", "islet")]
    wr = pd.DataFrame(within(D, CNT, list(itertools.combinations(G4, 2)), "fine_primary", False) + within(D, CNT, list(itertools.combinations(G4, 2)), "fine_matched_variability", True))
    wr.to_csv(OUT / "C_fine_within.csv", index=False)
    print(wr.assign(v=wr.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot_table(index=["variant", "pair"], columns="score", values="v", aggfunc="first").to_string(), flush=True)


def sec_D(D, CNT):
    ct = pd.read_parquet(RU.OUT / "classes_S.parquet")
    zS, zB, zP, zL = (pd.read_parquet(MAT / f"z_{k}__ED.parquet") for k in ("S", "B1", "Sdir_pattern", "Sdir_level"))
    evee(D)
    d = ct[ct["cls"].isin(["flip"])].copy()
    take = lambda M, tt, idx: np.array([M.at[g, t] for g, t in zip(idx, tt)])  # noqa: E731
    d["l_lo"], d["p_lo"], d["b_lo"], d["b_hi"], d["s_lo"], d["s_hi"] = take(zL, d.t_lo, d.index), take(zP, d.t_lo, d.index), take(zB, d.t_lo, d.index), take(zB, d.t_hi, d.index), take(zS, d.t_lo, d.index), take(zS, d.t_hi, d.index)
    d["cat"] = np.select([(d.l_lo <= -1) & (d.p_lo > -1), (d.l_lo > -1) & (d.p_lo <= -1), (d.l_lo <= -1) & (d.p_lo <= -1)], ["level only", "pattern only", "both"], "neither")
    # z of each pattern gold over all genes with gold and C of its tissue
    GT = {"liver": "G_liver", "testis": "G_testis", "kidney": "G_kidney_fine", "islet": "G_islet_fine"}
    zG, zE = {}, {}
    for T, c in GT.items():
        g = D.gold[c].dropna().index
        g = D.cov_T(T, g)[PAT].dropna().index
        if c == "G_testis":
            g = g[D.gold["H4_nsp"].reindex(g).notna()]
        zG[T] = pd.Series(z_gold_or_score(D, D.gold[c], g, T, c, PAT), index=g)
    for T in ("liver", "testis", "kidney"):
        c = f"G_ev_{T}"
        g = D.gold[c].dropna().index
        g = D.cov_T(T, g)[LVL].dropna().index
        zE[T] = pd.Series(z_gold_or_score(D, D.gold[c], g, T, c, LVL), index=g)
    d["dG"] = [zG[h].get(g, np.nan) - zG[l].get(g, np.nan) if h in zG and l in zG else np.nan for g, h, l in zip(d.index, d.t_hi, d.t_lo)]
    d["dE"] = [zE[h].get(g, np.nan) - zE[l].get(g, np.nan) if h in zE and l in zE else np.nan for g, h, l in zip(d.index, d.t_hi, d.t_lo)]
    d.to_parquet(OUT / "D_flip_cells.parquet")
    rows = []
    for cat, sub in list(d.groupby("cat")) + [("all flips", d)]:
        rec = dict(cat=cat, n=len(sub), mean_zB1_lo=sub.b_lo.mean(), mean_zB1_hi=sub.b_hi.mean(), frac_B1_flip_same_sign=float((sub.b_hi - sub.b_lo > 0)[sub.b_hi.notna() & sub.b_lo.notna()].mean()))
        for k in ("dG", "dE"):
            x = sub[k].dropna()
            rec[f"n_{k}"] = len(x)
            rec[f"mean_{k}"] = x.mean()
            if len(x) >= 10:
                idx = np.searchsorted(D.ub, D.block.reindex(x.index).to_numpy())
                bt = [np.average(x.to_numpy(), weights=c[idx] + 1e-12) for c in CNT]
                rec[f"{k}_lo"], rec[f"{k}_hi"] = np.quantile(bt, .025), np.quantile(bt, .975)
        rows.append(rec)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "D_flip_categories.csv", index=False)
    print(t.round(3).to_string(), flush=True)
    print(pd.crosstab(d.t_lo, d.cat).to_string(), flush=True)


def sec_E(D, CNT):
    sh = pd.read_parquet(MAT / "cov_shift.parquet")
    expr, det = pd.read_parquet(MAT / "cov_expr.parquet"), pd.read_parquet(MAT / "cov_det.parquet")
    rows = []
    for a, b in itertools.combinations(ORDER, 2):
        ok = D.sup[a] & D.sup[b]
        rows.append(dict(a=a, b=b, n=int(ok.sum()), spearman_signed_shift=float(sh.loc[ok, [a, b]].corr(method="spearman").iloc[0, 1]),
                         spearman_abs_shift=float(sh.loc[ok, [a, b]].abs().corr(method="spearman").iloc[0, 1])))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "E_shift_pairs.csv", index=False)
    print(t.round(3).to_string(), t.mean(numeric_only=True).round(3).to_dict(), flush=True)


def sec_F(D, CNT):
    r = pd.read_csv(RU.OUT / "contrast_rho.csv")
    tb = pd.read_csv(RU.OUT / "tissues.csv").set_index("tissue")
    r["overlap"] = [np.mean([x in LINEAGE_SRC[s] for x in LINEAGE_GOLD[g]]) for g, s in zip(r.gold, r.source)]
    r["K"] = r.source.map(tb.K)
    r["cells"] = r.source.map(tb.cells)
    r["lib"] = r.source.map(tb.filter(like="median_lib").mean(axis=1))
    rows = []
    for sc in ("S", "B1", "Sdir", "D"):
        o = r[(r.score == sc) & ~r.matched]
        # remove gold means, then Spearman across the off-diagonals
        y = o.est - o.groupby("gold").est.transform("mean")
        row = dict(score=sc, n=len(o))
        for c in ("overlap", "K", "cells", "lib"):
            row[f"rho_{c}"] = float(pd.concat([y, o[c]], axis=1).corr(method="spearman").iloc[0, 1])
        rows.append(row)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "F_composition_depth.csv", index=False)
    print(t.round(3).to_string(), flush=True)
    print(r[(r.score == "S")][["gold", "source", "overlap", "est"]].round(3).to_string(), flush=True)


def sec_G(D, CNT):
    rng = np.random.default_rng(RU.SEED)
    u = pd.read_csv(Y.ROOT / "data" / "revision" / "x3_annotations" / "uniprot_human_reviewed.tsv.gz", sep="\t")
    gene_c = next(c for c in u.columns if "gene" in c.lower() and "primary" in c.lower()) if any("primary" in c.lower() for c in u.columns) else next(c for c in u.columns if "gene" in c.lower())
    kw_c = next(c for c in u.columns if "keyword" in c.lower())
    sym2kw = {}
    for g, k in zip(u[gene_c].astype(str), u[kw_c].astype(str)):
        sym2kw.setdefault(g.split()[0], set()).update(x.strip() for x in k.split(";") if x.strip() and x != "nan")
    ct = pd.read_parquet(RU.OUT / "classes_S.parquet")
    uni = ct[ct["cls"] != "too_few_tissues"]
    expr = pd.read_parquet(MAT / "cov_expr.parquet")
    bins = pd.qcut(expr.rank(pct=True).mean(axis=1).rank(method="first"), 20, labels=False).reindex(uni.index).to_numpy()
    kws = {}
    for i, s in enumerate(uni["symbol"].astype(str)):
        for k in sym2kw.get(s, ()):
            kws.setdefault(k, []).append(i)
    rows = []
    perms = np.tile(np.arange(len(uni)), (1000, 1))
    for b in np.unique(bins):
        i = np.flatnonzero(bins == b)
        for r in range(1000):
            perms[r, i] = rng.permutation(i)
    for cl in ("shared_all", "not_shared_all"):
        m = (uni["cls"] == cl).to_numpy()
        for k, ix in kws.items():
            if len(ix) < 20:
                continue
            a = np.zeros(len(uni), bool)
            a[ix] = True
            obs = int((m & a).sum())
            null = (m[None, :] & a[perms]).sum(1)
            rows.append(dict(cls=cl, keyword=k, n=len(ix), obs=obs, exp=null.mean(), fold=obs / max(null.mean(), 1e-9), p=float((1 + (np.abs(null - null.mean()) >= abs(obs - null.mean())).sum()) / 1001)))
    t = pd.DataFrame(rows)
    t["q"] = S.bh(t["p"].tolist())
    t.sort_values("p").to_csv(OUT / "G_keywords.csv", index=False)
    print(t.sort_values("p").head(25).round(3).to_string(), flush=True)


def sec_H(D, CNT):
    """Added after A-G were read: S against B1 and against Sdir, jointly, in the larger pairwise universes, every gold."""
    rows, inc = [], []
    golds = {**CT.GOLDS, **NEW}
    for gold, T in golds.items():
        col = f"G_{gold}"
        own, ownD = {k: [] for k in ("S", "B1", "Sdir")}, {k: [] for k in ("S", "B1", "Sdir")}
        for t in ORDER:
            if t == T:
                continue
            g = D.gold[col].dropna().index
            m = D.sup.reindex(g)[[T, t]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[[T, t]].notna().all(axis=1)
            g = m[m].index
            g = D.cov_T(T, g)[PAT].dropna().index
            if col == "G_testis":
                g = g[D.gold["H4_nsp"].reindex(g).notna()]
            Z = CT.design_T(D, T, g, PAT, col)
            cols = {f"{sc}|{x}": D.raw[sc][x].reindex(g) for sc in own for x in (T, t)}
            nm, e_, d_ = CT.rho_table(D, col, g, cols, Z, CNT)
            ix = {n: i for i, n in enumerate(nm)}
            for sc in own:
                own[sc].append(e_[ix[f"{sc}|{T}"]] - e_[ix[f"{sc}|{t}"]])
                ownD[sc].append(d_[:, ix[f"{sc}|{T}"]] - d_[:, ix[f"{sc}|{t}"]])
        m_ = {sc: (float(np.mean(own[sc])), np.mean(ownD[sc], axis=0)) for sc in own}
        for sc in own:
            rows.append(CT.summarise(f"Delta({sc}) pairwise", m_[sc][0], m_[sc][1], score=sc, gold=gold))
        for a, b in (("S", "B1"), ("S", "Sdir")):
            rows.append(CT.summarise(f"Delta({a})-Delta({b}) pairwise", m_[a][0] - m_[b][0], m_[a][1] - m_[b][1], score=f"{a}-{b}", gold=gold))
        # own-tissue increment on the tissue's support (no requirement in other tissues)
        g = D.gold[col].dropna().index
        m = D.sup.reindex(g)[[T]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[T].notna()
        g = m[m].index
        g = D.cov_T(T, g)[PAT].dropna().index
        if col == "G_testis":
            g = g[D.gold["H4_nsp"].reindex(g).notna()]
        ex = pd.DataFrame(dict(S_T=D.raw["S"][T].reindex(g), B1_T=D.raw["B1"][T].reindex(g), Sdir_T=D.raw["Sdir"][T].reindex(g), D_T=D.raw["D"][T].reindex(g)))
        for nm_, x, cond in (("S", "S_T", []), ("B1", "B1_T", []), ("Sdir", "Sdir_T", []), ("D", "D_T", []), ("S | B1", "S_T", ["B1_T"]), ("B1 | S", "B1_T", ["S_T"]), ("S | Sdir", "S_T", ["Sdir_T"]),
                             ("Sdir | S", "Sdir_T", ["S_T"]), ("S | B1, Sdir", "S_T", ["B1_T", "Sdir_T"])):
            Z = CT.design_T(D, T, g, PAT, col, ex[cond] if cond else None)
            _, e_, d_ = CT.rho_table(D, col, g, {x: ex[x]}, Z, CNT)
            inc.append(CT.summarise(nm_, e_[0], d_[:, 0], gold=gold, n_genes=len(g)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "H_pairwise_joint.csv", index=False)
    print(t.assign(v=t.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot(index="gold", columns="score", values="v").to_string(), flush=True)
    it = pd.DataFrame(inc)
    it.to_csv(OUT / "H_own_tissue_incremental.csv", index=False)
    print(it.assign(v=it.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot(index="gold", columns="name", values="v").to_string(), it.groupby("gold").n_genes.first().to_dict(), flush=True)


def fine_direct(t: str, key: str):
    """Direct margin on finer author labels (species x class x state), same code as X4; is S's lead over the labelled margin within-label state?"""
    from revision.common import cohorts as C_
    from revision.tasks.X4_where_differs import direct_margin as DM

    a = Y.read(t)
    a_hm, h, m = Y.hm_view(a)
    a_hm.obs["cell_type"] = (a_hm.obs["y1_group"].astype(str) + "|" + a_hm.obs[key].astype(str)).values
    vc = a_hm.obs.groupby(["species", "cell_type"], observed=True).size()
    C_.read_adata = lambda cohort, backed=False: a_hm
    df = DM.run(f"y1_fine_{t}")
    df["n_groups"] = len(vc)
    return df


def sec_I(D, CNT):
    """Added after H: S kept 0.13-0.14 beyond B1 and the labelled margin on kidney golds, 0.12 on liver. Reweighting of the two parts, or more?"""
    OUT.mkdir(exist_ok=True)
    zz = {}
    for t, key in (("kidney", "CellStateLevel2"), ("liver", "SubAnno")):
        f = OUT / f"Sdir_fine__{t}.parquet"
        if not f.exists():
            fine_direct(t, key).to_parquet(f)
        tb = Y.tissue_table(t).set_index("var_name")
        d = pd.read_parquet(f).set_index("var_name")
        zz[t] = pd.Series(d["Sdir"].reindex(tb.index).to_numpy(), index=tb["hsapiens_gene_id"].to_numpy())
    rows = []
    pat, lev = pd.read_parquet(MAT / "raw_Sdir_pattern.parquet"), pd.read_parquet(MAT / "raw_Sdir_level.parquet")
    for gold, T in {**CT.GOLDS, **NEW}.items():
        col = f"G_{gold}"
        g = D.gold[col].dropna().index
        m = D.sup.reindex(g)[[T]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[T].notna()
        g = m[m].index
        g = D.cov_T(T, g)[PAT].dropna().index
        if col == "G_testis":
            g = g[D.gold["H4_nsp"].reindex(g).notna()]
        ex = pd.DataFrame(dict(S_T=D.raw["S"][T].reindex(g), B1_T=D.raw["B1"][T].reindex(g), Sdir_T=D.raw["Sdir"][T].reindex(g), pat_T=pat[T].reindex(g), lev_T=lev[T].reindex(g)))
        spec = [("pattern", "pat_T", []), ("level", "lev_T", []), ("S | pattern, level", "S_T", ["pat_T", "lev_T"]), ("S | pattern, level, B1", "S_T", ["pat_T", "lev_T", "B1_T"]), ("pattern | S", "pat_T", ["S_T"]), ("level | S", "lev_T", ["S_T"])]
        if T in zz:
            g2 = g[zz[T].reindex(g).notna().to_numpy()]
            ex["fine"] = zz[T].reindex(g)
            spec += [("Sdir fine labels", "fine", []), ("S | Sdir fine", "S_T", ["fine"]), ("S | Sdir fine, B1", "S_T", ["fine", "B1_T"]), ("Sdir fine | S", "fine", ["S_T"])]
        for nm_, x, cond in spec:
            Z = CT.design_T(D, T, g, PAT, col, ex[cond] if cond else None)
            _, e_, d_ = CT.rho_table(D, col, g, {x: ex[x]}, Z, CNT)
            rows.append(CT.summarise(nm_, e_[0], d_[:, 0], gold=gold, n_genes=len(g)))
        # per seed: S given B1 and the labelled margin
        sm = pd.read_parquet(MAT / f"seedS__{T}.parquet").reindex(g)
        Z = CT.design_T(D, T, g, PAT, col, ex[["B1_T", "Sdir_T"]])
        vals = []
        for c in sm.columns:
            _, e_, _d = CT.rho_table(D, col, g, {"x": sm[c]}, Z, CNT[:2])
            vals.append(e_[0])
        rows.append(dict(name="S | B1, Sdir per seed", gold=gold, est=float(np.mean(vals)), lo=float(np.min(vals)), hi=float(np.max(vals)), n_genes=len(g)))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "I_reweighting.csv", index=False)
    print(t.assign(v=t.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot(index="name", columns="gold", values="v").to_string(), flush=True)


def model_split(D) -> dict:
    """Pm, Lm per tissue x gene: ppass x human and mouse rows of the split forward pass, mean over the tissue's seeds; masked genes 0. Per-seed tables kept."""
    pm, lm, seedP = {}, {}, {}
    fwS = pd.read_parquet(MAT / "raw_S_fw.parquet")
    checks = []
    for t in ORDER:
        tb = Y.tissue_table(t)
        tb = tb[tb["hsapiens_gene_id"].notna()]
        ps, ls = {}, {}
        for s_ in Y.TISSUES[t]["runs"]:
            z = np.load(Y.FW / f"split_{t}_s{s_}.npz")
            spn = [str(x) for x in z["species"]]
            hi = [i for i, x in enumerate(spn) if x.lower() in ("human", "mouse")]
            ns = np.asarray(z["n_species"], float)
            gs = z["g_species"]
            P_ = sum(z["Pg"][gs == i].sum(0) / ns[i] for i in hi) / 2
            L_ = sum(z["Lg"][gs == i].sum(0) / ns[i] for i in hi) / 2
            fi = pd.Series(np.arange(len(z["var_names"])), index=z["var_names"].astype(str)).reindex(tb["var_name"]).to_numpy()
            w = tb["ppass"].to_numpy(float) * (~tb["masked"].to_numpy(bool))
            ps[f"s{s_}"], ls[f"s{s_}"] = w * P_[fi], w * L_[fi]
        P_m, L_m = pd.DataFrame(ps, index=tb["hsapiens_gene_id"]), pd.DataFrame(ls, index=tb["hsapiens_gene_id"])
        pm[t], lm[t], seedP[t] = P_m.mean(axis=1), L_m.mean(axis=1), P_m
        s42 = (P_m["s42"] + L_m["s42"]).reindex(fwS.index)
        ok = fwS[t].notna() & s42.notna()
        sup = D.sup[t]
        dp = pd.read_parquet(MAT / "raw_Sdir_pattern.parquet")[t]
        dl = pd.read_parquet(MAT / "raw_Sdir_level.parquet")[t]
        checks.append(dict(tissue=t, closure_max=float(np.abs(s42[ok] - fwS.loc[ok, t] - (s42[ok] - fwS.loc[ok, t]).median()).max()), closure_r=float(np.corrcoef(s42[ok], fwS.loc[ok, t])[0, 1]),
                           rho_Pm_direct_pattern=float(pd.concat([pm[t].reindex(sup[sup].index), dp.reindex(sup[sup].index)], axis=1).corr(method="spearman").iloc[0, 1]),
                           rho_Lm_direct_level=float(pd.concat([lm[t].reindex(sup[sup].index), dl.reindex(sup[sup].index)], axis=1).corr(method="spearman").iloc[0, 1]),
                           rho_Pm_S=float(pd.concat([pm[t].reindex(sup[sup].index), D.raw["S"][t].reindex(sup[sup].index)], axis=1).corr(method="spearman").iloc[0, 1]),
                           rho_Pm_B1=float(pd.concat([pm[t].reindex(sup[sup].index), D.raw["B1"][t].reindex(sup[sup].index)], axis=1).corr(method="spearman").iloc[0, 1])))
    D.raw["Pm"], D.raw["Lm"] = pd.DataFrame(pm).reindex(fwS.index), pd.DataFrame(lm).reindex(fwS.index)
    return dict(checks=pd.DataFrame(checks), seedP=seedP)


def sec_J(D, CNT):
    sp_ = model_split(D)
    sp_["checks"].to_csv(OUT / "J_checks.csv", index=False)
    print(sp_["checks"].round(3).to_string(), flush=True)
    D.raw["Dp"] = pd.read_parquet(MAT / "raw_Sdir_pattern.parquet")
    golds = {**CT.GOLDS, **NEW}
    own, dl, sd = [], [], []
    for gold, T in golds.items():
        col = f"G_{gold}"
        g = D.gold[col].dropna().index
        m = D.sup.reindex(g)[[T]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[T].notna()
        g = m[m].index
        g = D.cov_T(T, g)[PAT].dropna().index
        if col == "G_testis":
            g = g[D.gold["H4_nsp"].reindex(g).notna()]
        ex = pd.DataFrame({k: D.raw[k][T].reindex(g) for k in ("S", "Pm", "Lm", "Dp", "B1", "Sdir")})
        Z = CT.design_T(D, T, g, PAT, col)
        nm, e_, d_ = CT.rho_table(D, col, g, {k: ex[k] for k in ex}, Z, CNT)
        ix = {n: i for i, n in enumerate(nm)}
        for n in nm:
            own.append(CT.summarise(n, e_[ix[n]], d_[:, ix[n]], gold=gold, kind="own", n_genes=len(g)))
        for a, b in (("Pm", "S"), ("Pm", "B1"), ("Pm", "Dp"), ("S", "B1")):
            own.append(CT.summarise(f"{a}-{b}", e_[ix[a]] - e_[ix[b]], d_[:, ix[a]] - d_[:, ix[b]], gold=gold, kind="diff", n_genes=len(g)))
        for nm_, x, cond in (("Pm | B1", "Pm", ["B1"]), ("B1 | Pm", "B1", ["Pm"]), ("Pm | direct pattern", "Pm", ["Dp"]), ("direct pattern | Pm", "Dp", ["Pm"]), ("Pm | B1, direct pattern", "Pm", ["B1", "Dp"]),
                             ("Lm | Pm", "Lm", ["Pm"]), ("S | Pm", "S", ["Pm"])):
            Zc = CT.design_T(D, T, g, PAT, col, ex[cond])
            _, ee, dd = CT.rho_table(D, col, g, {x: ex[x]}, Zc, CNT)
            own.append(CT.summarise(nm_, ee[0], dd[:, 0], gold=gold, kind="incr", n_genes=len(g)))
        sm = sp_["seedP"][T]
        vals = [CT.rho_table(D, col, g, {"x": sm[c].reindex(g)}, Z, CNT[:2])[1][0] for c in sm.columns]
        own.append(dict(name="Pm per seed", gold=gold, kind="seed", est=float(np.mean(vals)), lo=float(np.min(vals)), hi=float(np.max(vals)), n_genes=len(g)))
        # tissue-matched: all-six universe (matched variability beside), pairwise mean
        for tag in ("all_six", "matched_variability"):
            g6 = CT.universe(D, T, col)
            exv = None
            if tag == "matched_variability":
                at = atlas_cols(gold)
                g6 = g6[D.gold[at].reindex(g6).notna().all(axis=1) & D.cov["amp"][T].reindex(g6).notna().to_numpy()]
                exv = D.gold[at].reindex(g6).assign(amp_T=D.cov["amp"][T].reindex(g6))
            Z6 = CT.design_T(D, T, g6, PAT, col, exv)
            sc = ["S", "Pm", "B1", "Dp"]
            n6, e6, d6 = CT.rho_table(D, col, g6, CT.source_columns(D, g6, sc), Z6, CNT)
            r_, dd_ = CT.contrast(n6, e6, d6, sc, T)
            for x in r_:
                x.update(gold=gold, n_genes=len(g6), universe=tag)
                dl.append(x)
            for a, b in (("Pm", "S"), ("Pm", "B1"), ("Pm", "Dp")):
                dl.append(CT.summarise(f"Delta({a})-Delta({b})", dd_[a][0] - dd_[b][0], dd_[a][1] - dd_[b][1], score=f"{a}-{b}", gold=gold, n_genes=len(g6), universe=tag))
        o_, oD = {k: [] for k in ("S", "Pm", "B1", "Dp")}, {k: [] for k in ("S", "Pm", "B1", "Dp")}
        for t in ORDER:
            if t == T:
                continue
            gg = D.gold[col].dropna().index
            mm = D.sup.reindex(gg)[[T, t]].astype(bool).all(axis=1) & D.raw["B1"].reindex(gg)[[T, t]].notna().all(axis=1)
            gg = mm[mm].index
            gg = D.cov_T(T, gg)[PAT].dropna().index
            if col == "G_testis":
                gg = gg[D.gold["H4_nsp"].reindex(gg).notna()]
            Zp = CT.design_T(D, T, gg, PAT, col)
            nmp, ep, dp = CT.rho_table(D, col, gg, {f"{sc_}|{x_}": D.raw[sc_][x_].reindex(gg) for sc_ in o_ for x_ in (T, t)}, Zp, CNT)
            ipx = {n: i for i, n in enumerate(nmp)}
            for sc_ in o_:
                o_[sc_].append(ep[ipx[f"{sc_}|{T}"]] - ep[ipx[f"{sc_}|{t}"]])
                oD[sc_].append(dp[:, ipx[f"{sc_}|{T}"]] - dp[:, ipx[f"{sc_}|{t}"]])
        mv = {k: (float(np.mean(o_[k])), np.mean(oD[k], axis=0)) for k in o_}
        for k in mv:
            dl.append(CT.summarise(f"Delta({k})", mv[k][0], mv[k][1], score=k, gold=gold, universe="pairwise"))
        for a, b in (("Pm", "S"), ("Pm", "B1"), ("Pm", "Dp")):
            dl.append(CT.summarise(f"Delta({a})-Delta({b})", mv[a][0] - mv[b][0], mv[a][1] - mv[b][1], score=f"{a}-{b}", gold=gold, universe="pairwise"))
    ot = pd.DataFrame(own)
    ot.to_csv(OUT / "J_own.csv", index=False)
    print(ot.assign(v=ot.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot(index="name", columns="gold", values="v").to_string(), ot.groupby("gold").n_genes.first().to_dict(), flush=True)
    dt = pd.DataFrame(dl)
    dt.to_csv(OUT / "J_delta.csv", index=False)
    print(dt.assign(v=dt.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot_table(index=["gold", "universe"], columns="score", values="v", aggfunc="first").to_string(), flush=True)
    # within gene, C and matched variability, S, Pm, B1
    wr = []
    G4 = [("liver", "liver"), ("testis", "testis"), ("kidney_fine", "kidney"), ("islet_noacinar", "islet")]
    G0 = [("liver", "liver"), ("testis", "testis"), ("kidney", "kidney"), ("pancreas", "islet")]
    for tag, Gs in (("primary", G0), ("fine", G4)):
        for matched in (False, True):
            for (g1, T1), (g2, T2) in itertools.combinations(Gs, 2):
                c1, c2 = f"G_{g1}", f"G_{g2}"
                g = D.gold[[c1, c2]].dropna().index
                m = D.sup.reindex(g)[[T1, T2]].astype(bool).all(axis=1) & D.raw["B1"].reindex(g)[[T1, T2]].notna().all(axis=1)
                g = m[m].index
                g = D.cov_T(T1, g)[PAT + ["amp"]].dropna().index.intersection(D.cov_T(T2, g)[PAT + ["amp"]].dropna().index)
                if "G_testis" in (c1, c2):
                    g = g[D.gold["H4_nsp"].reindex(g).notna()]
                a1, a2 = (atlas_cols(g1), atlas_cols(g2)) if matched else (None, None)
                if matched:
                    g = g[D.gold[a1 + a2].reindex(g).notna().all(axis=1)]
                zT = lambda y, T, c, at: z_gold_or_score(D, y, g, T, c, PAT, at)  # noqa: E731
                dG = zT(D.gold[c1], T1, c1, a1) - zT(D.gold[c2], T2, c2, a2)
                dd = {sc: zT(D.raw[sc][T1], T1, c1, a1) - zT(D.raw[sc][T2], T2, c2, a2) for sc in ("S", "Pm", "B1", "Dp")}
                A = np.column_stack([dG] + list(dd.values()))
                est = wcorr(A, np.ones(len(g)))
                dr = np.stack([wcorr(A, CT.weights(D, g, c) + 1e-12) for c in CNT])
                for i, n in enumerate(dd):
                    wr.append(CT.summarise(f"{g1} vs {g2}", est[i], dr[:, i], score=n, set=tag, matched=matched, n_genes=len(g)))
                wr.append(CT.summarise(f"{g1} vs {g2}", est[1] - est[2], dr[:, 1] - dr[:, 2], score="Pm-B1", set=tag, matched=matched, n_genes=len(g)))
                wr.append(CT.summarise(f"{g1} vs {g2}", est[1] - est[0], dr[:, 1] - dr[:, 0], score="Pm-S", set=tag, matched=matched, n_genes=len(g)))
    wt = pd.DataFrame(wr)
    wt.to_csv(OUT / "J_within.csv", index=False)
    print(wt.assign(v=wt.apply(lambda x: f"{x.est:.3f} ({x.lo:.3f},{x.hi:.3f})", axis=1)).pivot_table(index=["set", "matched", "name"], columns="score", values="v", aggfunc="first").to_string(), flush=True)
    # gene share of the split, complete cases
    sup = pd.read_parquet(MAT / "support.parquet")
    b1 = pd.read_parquet(MAT / "raw_B1.parquet")
    cc = sup.all(axis=1) & b1.notna().all(axis=1)
    genes = cc[cc].index
    cov = {c: pd.read_parquet(MAT / f"cov_{c}.parquet") for c in ("expr", "det", "tau")}
    vs = []
    for dn, cols in (("ED", ["expr", "det"]), ("EDtau", ["expr", "det", "tau"])):
        for sc in ("Pm", "Lm"):
            Z = pd.DataFrame({t: P.z_on(D.raw[sc][t].reindex(genes), pd.DataFrame({c: cov[c][t] for c in cols}).reindex(genes), cols) for t in ORDER})
            pr = [np.corrcoef(Z[a], Z[b])[0, 1] for a, b in itertools.combinations(ORDER, 2)]
            vs.append(dict(score=sc, design=dn, n=len(genes), gene_share=float(np.mean(pr))))
    pd.DataFrame(vs).to_csv(OUT / "J_gene_share.csv", index=False)
    print(pd.DataFrame(vs).round(3).to_string(), flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sections", default="A,B,C,D,E,F,G")
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    D = data()
    D.sup = D.sup.astype(bool)
    CNT = CT.block_counts(D, RU.N_DRAWS, RU.SEED)
    fn = dict(A=sec_A, B=sec_B, C=sec_C, D=sec_D, E=sec_E, F=sec_F, G=sec_G, H=sec_H, I=sec_I, J=sec_J)
    for s in a.sections.split(","):
        print("==", s, flush=True)
        fn[s](D, CNT)


if __name__ == "__main__":
    main()
