"""Y5 F5 species x type margins, F6 X linkage in testis. Usage: python -m revision.tasks.Y5_aspects.f56 f5|f6|cases"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.X7_architecture_outputs.forward import OUT as X7
from revision.tasks.Y5_aspects import common as K, rules as RU

OUT = K.OUT / "F5"
MAP_LIVER = {"Hepatocytes": "hepatocyte", "Endothelial": "endothelial", "fibroblasts": "stellate", "macrophage": "macrophage"}


def contrast_table(c):
    """X_gsk per mouse exception; pseudobulk direct from X7 gene_type_direct."""
    S, names, var = K.gsk(c, key="cell_type", min_cells=30)
    d = pd.DataFrame(S, index=pd.Index(var, name="var_name"), columns=[f"{a}|{b}" for a, b in names])
    return d, names


def f5():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, within_rows = [], []
    h3 = pd.read_parquet(X7 / "golds" / "h3_gene_type.parquet").reset_index()
    mu = pd.read_parquet(X7 / "golds" / "murat_gene_type.parquet").reset_index()
    for c in ("liv_4sp", "test_3sp", "liv_hmouse", "test_hmouse"):
        S, names = contrast_table(c)
        g = K.genes(c)
        tests = c.startswith("test")
        lab = lambda l: T42.TESTIS_COARSE.get(l, l) if tests else ({"Kupffer cells": "macrophage", "Macrophages": "macrophage"}.get(l, l))
        # collapse fine labels into coarse, weight by n ignored (mean over available labels)
        coarse = {}
        for s, l in names:
            coarse.setdefault((s, lab(l)), []).append(f"{s}|{l}")
        M = pd.DataFrame({f"{s}|{k}": S[v].mean(1) for (s, k), v in coarse.items()})
        direct = pd.read_parquet(X7 / "baselines" / f"{c}__gene_type_direct.parquet")
        if "hsapiens_gene_id" not in direct.columns and direct.index.name:
            direct = direct.reset_index()
        print(c, direct.columns.tolist()[:12], flush=True)
        types = sorted({k for (s, k) in coarse})
        gm = g.set_index("var_name")
        q = []
        for k in types:
            sps = [s for (s, kk) in coarse if kk == k and s != "Zebrafish"]
            if "Mouse" not in sps or len(sps) < 2:
                continue
            others = [s for s in sps if s != "Mouse"]
            X = M[f"Mouse|{k}"] - M[[f"{s}|{k}" for s in others]].mean(1)
            t = pd.DataFrame({"var_name": M.index, "k": k, "X": X.values})
            q.append(t)
        q = pd.concat(q).merge(gm[["hsapiens_gene_id", "half", "masked", "block"]].reset_index(), on="var_name")
        q["Xneg"] = -q.X
        q["Xabs"] = q.X.abs()
        if tests:
            gl = mu.melt(id_vars="hsapiens_gene_id", var_name="col", value_name="I")
            gl = gl[gl.col.str.startswith("I_PO_")]
            gl["k"] = gl.col.str.replace("I_PO_", "")
        else:
            gl = h3.melt(id_vars="hsapiens_gene_id", var_name="col", value_name="I")
            gl["kk"] = gl.col.str.replace("I_", "")
            rev = {v: u for u, v in MAP_LIVER.items()}
            gl["k"] = gl.kk.map(rev)
        q = q.merge(gl[["hsapiens_gene_id", "k", "I"]], on=["hsapiens_gene_id", "k"])
        # direct gene x type gap, level and detection from X7 (wide: Ddir_<t>, det_<t>, x_<t> or level_<t>)
        dd = direct.set_index("hsapiens_gene_id") if "hsapiens_gene_id" in direct else direct.set_index("var_name")
        idc = "hsapiens_gene_id" if "hsapiens_gene_id" in direct else "var_name"
        print(sorted({x.split("_")[0] for x in dd.columns}), flush=True)
        q["Ddir"] = [dd.at[i, f"Ddir_{k}"] if f"Ddir_{k}" in dd.columns and i in dd.index else np.nan for i, k in zip(q[idc], q.k)]
        q["det"] = [dd.at[i, f"det_{k}"] if f"det_{k}" in dd.columns and i in dd.index else np.nan for i, k in zip(q[idc], q.k)]
        lvl = [x for x in ("lev", "level", "xbar", "x", "mean") if any(c2.startswith(x + "_") for c2 in dd.columns)][0]
        q["lev"] = [dd.at[i, f"{lvl}_{k}"] if f"{lvl}_{k}" in dd.columns and i in dd.index else np.nan for i, k in zip(q[idc], q.k)]
        levc, detc = ["lev"], ["det"]
        for k, tk in q.groupby("k"):
            for half in ("odd", "even"):
                T = tk[(tk.half == half) & ~tk.masked.astype(bool) & tk[["I", "Xneg", "Xabs", "Ddir", "block"] + levc + detc].notna().all(1)]
                if len(T) < 100:
                    continue
                r = K.scan(T, "I", ["Xneg", "Xabs", "Ddir"], levc + detc, seed=K.useed(c, k, half), delta_to="Xabs")
                r["cohort"], r["k"], r["half"] = c, k, half
                rows.append(r)
        # within gene, across types: exception type vs gold's largest gap
        for half in ("odd", "even"):
            T = q[(q.half == half) & ~q.masked.astype(bool) & q[["I", "Xneg", "Ddir"]].notna().all(1)].copy()
            T = T[T.groupby("var_name").k.transform("size") >= 3]
            for col in ("I", "Xneg", "Ddir"):
                T[col + "_c"] = T[col].rank(pct=True) - T.groupby("var_name")[col].transform(lambda s: s.rank(pct=True).mean())
                T[col + "_cw"] = T[col] - T.groupby("var_name")[col].transform("mean")
            from scipy.stats import spearmanr
            within_rows.append(dict(cohort=c, half=half, n_genes=T.var_name.nunique(), rho_Xneg=spearmanr(T.I_cw, T.Xneg_cw)[0], rho_Ddir=spearmanr(T.I_cw, T.Ddir_cw)[0]))
    pd.concat(rows, ignore_index=True).to_csv(OUT / "exception_rho.csv", index=False)
    pd.DataFrame(within_rows).to_csv(OUT / "exception_within_gene.csv", index=False)


def cases():
    rows = []
    for gene, k, sp_, why in RU.LINEAGE_CASES:
        c = "test_3sp" if k in ("SG", "SC", "rSD", "eSD", "Sertoli") else "liv_4sp"
        g = K.genes(c)
        S, names = contrast_table(c)
        hit = g[g.symbol == gene]
        if hit.empty:
            rows.append(dict(gene=gene, cohort=c, type=k, expectation=why, in_table=False))
            continue
        h = hit.iloc[0]
        lab = lambda l: T42.TESTIS_COARSE.get(l, l) if c.startswith("test") else ({"Kupffer cells": "macrophage", "Macrophages": "macrophage"}.get(l, l))
        out = dict(gene=gene, cohort=c, type=k, expectation=why, in_table=True, masked=bool(h.masked), nameable=h.get("nameable"), S=h.S, D_raw=h.D_raw, B1b=h.B1b)
        for (s, l) in names:
            if lab(l) == k:
                out[f"S_{s}"] = float(S.loc[h.var_name, f"{s}|{l}"])
        rows.append(out)
    pd.DataFrame(rows).to_csv(OUT / "lineage_cases.csv", index=False)


def f6():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in ("test_3sp", "test_hmouse", "liv_4sp", "liv_hmouse"):
        g = K.genes(c)
        drop = [k for k in g.columns if k.startswith(("C_", "Csp_"))]
        g = g.drop(columns=drop).merge(K.contribs(c), on="var_name", how="left")
        g["isX"] = (g.chrom == "X").astype(float)
        g = g[g.chrom.isin(RU.ODD | RU.EVEN | {"X"})]
        ys = ["S", "D_raw", "B1b", "B1c", "Sdir"] + [k for k in g.columns if k.startswith("C_")] + (["G_H4", "G_H4_primates"] if c.startswith("test") else ["G_H3"])
        for y in ys:
            T = g[(~g.masked.astype(bool)) & g[[y, "isX", "block"] + K.C4].notna().all(1)]
            r = K.scan(T, "isX", [y], K.C4, seed=K.useed(c, y))
            r["cohort"], r["nX"] = c, int(T.isX.sum())
            rows.append(r)
    pd.concat(rows, ignore_index=True).to_csv(OUT / "xlinked.csv", index=False)


def f6b():
    """F6 extension, after F6 was read. Human X-linked genes in spermatocytes against held-out species at known sex-chromosome ages.
    X conserved region (XCR, chrX >= 47 Mb): X-linked in all therians, autosomal in platypus and chicken. X added region (XAR, < 47 Mb):
    X-linked in eutherians, autosomal in opossum, platypus, chicken (Ross 2005; Veyrunes 2008). Meiotic silencing depends on being X-linked,
    so SC minus SG is expected lower than autosomes where the region is on the X, and not elsewhere. Beside Chimera's C_SC and the direct contribution."""
    import gzip, re
    from revision.tasks.T4_3_scoreboard import stats as ST
    start = {}
    with gzip.open(K.REPO / "data/revision/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz", "rt") as fh:
        for line in fh:
            if line.startswith("X\t"):
                p = line.split("\t", 5)
                if p[2] == "gene":
                    start[re.search(r'gene_id "([^"]+)"', line).group(1)] = int(p[3])
    z = np.load(X7 / "golds" / "murat_profiles.npz", allow_pickle=False)
    genes = z["genes"].astype(str)
    types = list(z["types"].astype(str))
    chrom = pd.read_parquet(K.OUT / "chromosomes.parquet").set_index("hsapiens_gene_id").chrom
    reg = np.where(pd.Series(genes).map(start).fillna(0).to_numpy() >= 47e6, "XCR", np.where(pd.Series(genes).map(start).notna(), "XAR", "A"))
    auto = pd.Series(genes).map(chrom).isin(RU.ODD | RU.EVEN).to_numpy()
    rows = []
    for spc in ["bonobo", "chimpanzee", "gorilla", "gibbon", "marmoset", "opossum", "platypus", "chicken"]:
        x, f = z[f"{spc}_x"], z[f"{spc}_f"]
        sc, sg = types.index("SC"), types.index("SG")
        ok = (np.nanmax(f, 1) >= 0.05) & np.isfinite(x[:, sc]) & np.isfinite(x[:, sg])
        delta = x[:, sc] - x[:, sg]
        lev = np.nanmean(x, 1)
        for r in ("XCR", "XAR"):
            m = ok & (auto | (reg == r))
            T = pd.DataFrame(dict(d=delta[m], lev=lev[m], isX=(reg[m] == r).astype(float)))
            e = ST.resid(ST.midrank(T.d.to_numpy()), ST.design(T, ["lev"]))
            ex, sa = e[T.isX == 1], e[T.isX == 0]
            rng = np.random.default_rng(0)
            bs = [rng.choice(ex, len(ex)).mean() - rng.choice(sa, len(sa)).mean() for _ in range(2000)]
            rows.append(dict(species=spc, region=r, n_X=len(ex), rank_shift=float((ex.mean() - sa.mean()) / len(T)), lo=float(np.quantile(bs, .025) / len(T)), hi=float(np.quantile(bs, .975) / len(T)),
                             median_delta_X=float(np.median(T.d[T.isX == 1])), median_delta_A=float(np.median(T.d[T.isX == 0]))))
    pd.DataFrame(rows).to_csv(OUT / "xlinked_depth.csv", index=False)
    # Chimera C_SC and direct contribution by region on the training species
    out = []
    for c in ("test_3sp", "test_hmouse"):
        g = K.genes(c)
        g = g.drop(columns=[k for k in g.columns if k.startswith(("C_", "Csp_"))]).merge(K.contribs(c), on="var_name", how="left")
        dr = pd.read_parquet(K.REPO / f"results/revision/X4/cache/Sdir__{c}.parquet")
        cols = [k for k in dr.columns if k.startswith("Cdir_")]
        g = g.merge(dr[["var_name"] + cols] if "var_name" in dr else dr[["hsapiens_gene_id"] + cols], how="left")
        g["start"] = g.hsapiens_gene_id.map(start)
        g["reg"] = np.where(g.start >= 47e6, "XCR", np.where(g.start.notna(), "XAR", "A"))
        g = g[g.chrom.isin(RU.ODD | RU.EVEN | {"X"})]
        for r in ("XCR", "XAR"):
            q = g[g.reg.isin(["A", r]) & ~g.masked.astype(bool)].copy()
            q["isX"] = (q.reg == r).astype(float)
            ys = [k for k in ["C_SC", "Cdir_SC", "C_SG", "Cdir_SG", "C_rSD", "Cdir_rSD", "C_eSD", "Cdir_eSD", "S", "Sdir", "B1b"] if k in q]
            q = q[q[ys + K.C4 + ["block"]].notna().all(1)]
            rr = K.scan(q, "isX", ys, K.C4, seed=K.useed(c, r), draws=1000)
            rr["cohort"], rr["region"], rr["nX"] = c, r, int(q.isX.sum())
            out.append(rr)
    pd.concat(out, ignore_index=True).to_csv(OUT / "xlinked_regions.csv", index=False)


if __name__ == "__main__":
    dict(f5=f5, f6=f6, cases=cases, f6b=f6b)[sys.argv[1]]()
