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
    h3 = pd.read_parquet(X7 / "golds" / "h3_gene_type.parquet")
    mu = pd.read_parquet(X7 / "golds" / "murat_gene_type.parquet")
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
        # direct gene x type gap, level and detection from X7 (long or wide)
        if {"Ddir", "k"} <= set(direct.columns) or {"Ddir", "type"} <= set(direct.columns):
            kcol = "k" if "k" in direct.columns else "type"
            q = q.merge(direct.rename(columns={kcol: "k"}), on=["var_name", "k"], how="left") if "var_name" in direct else q.merge(direct.rename(columns={kcol: "k"}), on=["hsapiens_gene_id", "k"], how="left")
        else:
            print("direct columns unexpected", direct.columns.tolist(), flush=True)
        levc = [x for x in q.columns if x in ("level", "lev", "x_mean", "level_k")]
        detc = [x for x in q.columns if x in ("det", "detection", "f_mean", "det_k")]
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


if __name__ == "__main__":
    dict(f5=f5, f6=f6, cases=cases)[sys.argv[1]]()
