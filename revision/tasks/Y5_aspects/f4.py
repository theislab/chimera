"""Y5 F4: modules. Gene-pair coherence, modules replicated on even genes, programs moving as a block.

Sections: pairs, modules, blocks. Usage: python -m revision.tasks.Y5_aspects.f4 <section> [cohort]
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import rankdata, norm, hypergeom
from revision.common import cohorts as C
from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.Y5_aspects import common as K, rules as RU

OUT = K.OUT / "F4"
COH = ["liv_hmouse", "liv_4sp", "test_3sp", "test_hmouse"]


def profiles(c):
    """S_gsk and pseudobulk on the same (species, label) groups, unmasked genes."""
    key = "cell_ontology_class" if c.startswith("test") else "cell_type"
    S, names, var = K.gsk(c, key=key, min_cells=RU.PROFILE_MIN_CELLS)
    a = C.read_adata(c)
    assert list(a.var_names) == list(var)
    X = sp.csr_matrix(a.X).astype(float)
    sp_l = a.obs["species"].astype(str).to_numpy()
    lb = a.obs["cell_ontology_class" if c.startswith("test") else "cell_type"].astype(str).to_numpy()
    P = []
    for s, l in names:
        m = (sp_l == s) & (lb == l)
        cnt = np.asarray(X[m].sum(0)).ravel()
        P.append(np.log2(1 + 1e6 * cnt / max(cnt.sum(), 1)))
    return S, np.column_stack(P), names, var


def zsim(M):
    M = (M - M.mean(0)) / np.where(M.std(0) > 0, M.std(0), 1)
    M = M - M.mean(1, keepdims=True)
    n = np.linalg.norm(M, axis=1, keepdims=True)
    M = M / np.where(n > 0, n, 1)
    return M @ M.T, M


def auc_rank(score, pos):
    r = rankdata(score)
    n1 = pos.sum()
    n0 = len(pos) - n1
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)) if n1 and n0 else np.nan


def stratified_auc(score, pos, strata):
    tot, wsum = 0.0, 0.0
    for k in np.unique(strata):
        m = strata == k
        a = auc_rank(score[m], pos[m])
        if np.isfinite(a):
            w = pos[m].sum()
            tot += a * w
            wsum += w
    return tot / wsum if wsum else np.nan


def gold_mats(ids):
    from revision.tasks.T6_common import functional as FN
    pos = {g: i for i, g in enumerate(ids)}
    n = len(ids)
    L = FN.raw_libraries()
    out = {}
    for nm in ("Reactome", "Hallmark"):
        M = sp.lil_matrix((n, n), dtype=np.bool_)
        for t, v in L[nm].items():
            ix = [pos[g] for g in v if g in pos]
            if RU.PATHWAY_SIZE[0] <= len(ix) <= RU.PATHWAY_SIZE[1]:
                M[np.ix_(ix, ix)] = True
        out[nm] = sp.csr_matrix(M)
    ed = FN.string_edges("9606", RU.STRING_THR)
    a, b = ed.a.map(pos), ed.b.map(pos)
    k = a.notna() & b.notna()
    M = sp.csr_matrix((np.ones(k.sum(), bool), (a[k].astype(int), b[k].astype(int))), shape=(n, n))
    out["STRING"] = (M + M.T).astype(bool).tocsr()
    return out


def pairs_one(c):
    S, P, names, var = profiles(c)
    g = K.genes(c).set_index("var_name").reindex(var)
    Ppack = {}
    rows = []
    for half in ("odd", "even"):
        m = (g.half.to_numpy() == half) & ~g.masked.astype(bool).to_numpy() & np.isfinite(g.log_cp10k_spmean.to_numpy())
        ids = g.hsapiens_gene_id.to_numpy()[m]
        uniq = ~pd.Series(ids).duplicated().to_numpy()
        idx = np.flatnonzero(m)[uniq]
        ids = ids[uniq]
        Ss, _ = zsim(S[idx])
        Ps, _ = zsim(P[idx])
        Gm = gold_mats(ids)
        lev = g.log_cp10k_spmean.to_numpy()[idx]
        lb = np.digitize(lev, np.quantile(lev, [.2, .4, .6, .8]))
        iu = np.triu_indices(len(idx), 1)
        s, p = Ss[iu], Ps[iu]
        strat = lb[iu[0]] * 5 + lb[iu[1]]
        strat = np.minimum(lb[iu[0]], lb[iu[1]]) * 5 + np.maximum(lb[iu[0]], lb[iu[1]])
        pdec = np.digitize(p, np.quantile(p, np.linspace(.1, .9, 9)))
        # gene jackknife folds
        rng = np.random.default_rng(K.useed(c, half))
        fold = rng.integers(0, 10, len(idx))
        for gn, M in Gm.items():
            pos = np.asarray(M[iu]).ravel().astype(bool)
            est = dict(S=stratified_auc(s, pos, strat), PB=stratified_auc(p, pos, strat), S_given_PB=stratified_auc(s, pos, strat * 10 + pdec),
                       PB_given_S=stratified_auc(p, pos, strat * 10 + np.digitize(s, np.quantile(s, np.linspace(.1, .9, 9)))))
            jk = {k: [] for k in est}
            for f in range(10):
                keep = (fold[iu[0]] != f) & (fold[iu[1]] != f)
                sk, pk, sta, pdk, ps = s[keep], p[keep], strat[keep], pdec[keep], pos[keep]
                jk["S"].append(stratified_auc(sk, ps, sta))
                jk["PB"].append(stratified_auc(pk, ps, sta))
                jk["S_given_PB"].append(stratified_auc(sk, ps, sta * 10 + pdk))
                jk["PB_given_S"].append(stratified_auc(pk, ps, sta * 10 + np.digitize(sk, np.quantile(sk, np.linspace(.1, .9, 9)))))
            for k, v in est.items():
                se = np.sqrt(9 / 10 * np.sum((np.array(jk[k]) - np.mean(jk[k])) ** 2))
                rows.append(dict(cohort=c, half=half, gold=gn, score=k, auc=v, se=se, p_vs_half=float(2 * norm.sf(abs(v - .5) / se)) if se > 0 else np.nan,
                                 n_genes=len(idx), n_pos=int(pos.sum())))
            d = np.array(jk["S"]) - np.array(jk["PB"])
            se = np.sqrt(9 / 10 * np.sum((d - d.mean()) ** 2))
            rows.append(dict(cohort=c, half=half, gold=gn, score="S-PB", auc=est["S"] - est["PB"], se=se, p_vs_half=float(2 * norm.sf(abs(est["S"] - est["PB"]) / se)), n_genes=len(idx), n_pos=int(pos.sum())))
        print(c, half, len(idx), flush=True)
    return pd.DataFrame(rows)


def pairs():
    OUT.mkdir(parents=True, exist_ok=True)
    t = pd.concat([pairs_one(c) for c in COH], ignore_index=True)
    t.to_csv(OUT / "pairs_auc.csv", index=False)


def modules():
    import anndata as ad
    import scanpy as sc
    from revision.tasks.T6_common import functional as FN
    OUT.mkdir(parents=True, exist_ok=True)
    L = FN.raw_libraries()
    sets = {**{f"H:{k}": v for k, v in L["Hallmark"].items()}, **{f"R:{k}": v for k, v in L["Reactome"].items()}}
    rows, mem = [], []
    for c in COH:
        S, P, names, var = profiles(c)
        g = K.genes(c).set_index("var_name").reindex(var)
        unm = ~g.masked.astype(bool).to_numpy()
        idx_o = np.flatnonzero(unm & (g.half.to_numpy() == "odd"))
        idx_e = np.flatnonzero(unm & (g.half.to_numpy() == "even"))
        ids = g.hsapiens_gene_id.to_numpy()
        for nm, M in (("S", S), ("PB", P)):
            _, Mz = zsim(M[np.r_[idx_o, idx_e]])
            Mo, Me = Mz[:len(idx_o)], Mz[len(idx_o):]
            a = ad.AnnData(X=np.zeros((len(idx_o), 1), np.float32))
            a.obsm["X"] = Mo.astype(np.float32)
            sc.pp.neighbors(a, n_neighbors=RU.MODULE_K, use_rep="X", transformer="sklearn", random_state=0)
            sc.tl.leiden(a, resolution=RU.MODULE_RES, flavor="igraph", n_iterations=2, directed=False, random_state=0, key_added="m")
            lab = a.obs.m.astype(int).to_numpy()
            cen = np.vstack([Mo[lab == k].mean(0) for k in np.unique(lab)])
            ass = np.argmax(Me @ cen.T, 1)
            mem.append(pd.DataFrame(dict(cohort=c, profile=nm, hsapiens_gene_id=ids[np.r_[idx_o, idx_e]], half=["odd"] * len(idx_o) + ["even"] * len(idx_e), module=np.r_[lab, ass])))
            U_o, U_e = set(ids[idx_o]), set(ids[idx_e])
            for k in np.unique(lab):
                Go, Ge = set(ids[idx_o][lab == k]), set(ids[idx_e][ass == k])
                for t, v in sets.items():
                    vo, ve = v & U_o, v & U_e
                    if not (RU.PATHWAY_SIZE[0] <= len(vo) <= RU.PATHWAY_SIZE[1]) or len(ve) < 5:
                        continue
                    ov = len(Go & vo)
                    if ov < 3:
                        continue
                    po = hypergeom.sf(ov - 1, len(U_o), len(vo), len(Go))
                    oe = len(Ge & ve)
                    pe = hypergeom.sf(oe - 1, len(U_e), len(ve), max(len(Ge), 1)) if oe > 0 else 1.0
                    rows.append(dict(cohort=c, profile=nm, module=int(k), set=t, n_mod_odd=len(Go), n_set_odd=len(vo), ov_odd=ov, p_odd=po,
                                     n_mod_even=len(Ge), n_set_even=len(ve), ov_even=oe, p_even=pe))
        print(c, flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "modules_enrich.csv", index=False)
    pd.concat(mem).to_parquet(OUT / "modules_members.parquet")


def blocks():
    from revision.tasks.T6_common import functional as FN
    OUT.mkdir(parents=True, exist_ok=True)
    L = FN.raw_libraries()
    sets = {**{f"H:{k}": v for k, v in L["Hallmark"].items()}, **{f"R:{k}": v for k, v in FN.prune_jaccard({k: set(v) for k, v in L["Reactome"].items() if 10 <= len(v) <= 300}).items()}}
    rows = []
    for c in COH + ["vz_hmouse", "vz_hmmul", "bm"]:
        if c.startswith(("vz", "bm")):
            g = pd.read_parquet(K.REPO / RU.EXTERNAL_TISSUE[c]).rename(columns={"S_ema": "S"})
            chrom = pd.read_parquet(K.OUT / "chromosomes.parquet").set_index("hsapiens_gene_id").chrom
            g["half"] = np.where(g.hsapiens_gene_id.map(chrom).isin(RU.ODD), "odd", np.where(g.hsapiens_gene_id.map(chrom).isin(RU.EVEN), "even", "other"))
            g["masked"] = g.S == 0
            ys = ["S", "D_raw", "B1b"]
        else:
            g = K.genes(c)
            ys = ["S", "D_raw", "c_esm", "c_expr", "B1b", "B1c", "Sdir"]
        for y in ys:
            for half in ("odd", "even"):
                q = g[(g.half == half) & ~g.masked.astype(bool) & g[K.C4 + [y]].notna().all(1)].drop_duplicates("hsapiens_gene_id")
                if len(q) < 500:
                    continue
                e = ST.resid(ST.midrank(q[y].to_numpy(float)), ST.design(q, K.C4))
                e = pd.Series(e / e.std(), index=q.hsapiens_gene_id)
                for t, v in sets.items():
                    ids = e.index.intersection(list(v))
                    if 8 <= len(ids) <= 300:
                        m = float(e.loc[ids].mean())
                        rows.append(dict(cohort=c, y=y, half=half, set=t, n=len(ids), mean=m, p=float(2 * norm.sf(abs(m) * np.sqrt(len(ids))))))
        print(c, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "blocks.csv", index=False)


if __name__ == "__main__":
    dict(pairs=pairs, modules=modules, blocks=blocks)[sys.argv[1]]()
