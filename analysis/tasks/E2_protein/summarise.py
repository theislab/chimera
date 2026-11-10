"""E2 summaries: generality across tissues, both gates; families, complexes, gene age, duplication class, named genes.

Usage (from chimera/):  mamba run -n chimera_env python -m analysis.tasks.E2_protein.summarise
Reads results/analysis/E2/{scan,family,complex,joint}.csv, gene_resid.parquet, props.parquet. Writes *_summary.csv beside.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from analysis.common import gene_ids
from analysis.tasks.E2_protein import rules as RU
from analysis.tasks.T4_3_scoreboard import stats as ST

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "E2"
GATES = ["D_can", "D_free"]


def tissue_table(d: pd.DataFrame, val: str) -> pd.DataFrame:
    return d.groupby(["key", "tissue"])[val].median().unstack()


def classify(tm: pd.DataFrame, sig: pd.DataFrame | None) -> pd.DataFrame:
    med = tm.median(1)
    same = (np.sign(tm).eq(np.sign(med), axis=0)).sum(1)
    nt = tm.notna().sum(1)
    out = pd.DataFrame({"median": med, "n_tissues": nt, "same_sign": same})
    out["n_tissue_sig"] = sig.reindex(tm.index).fillna(0).astype(int) if sig is not None else np.nan
    gen = (out.same_sign >= np.minimum(8, out.n_tissues)) & (out.n_tissues >= 7)
    if sig is not None:
        gen &= out.n_tissue_sig >= 5
    out["general"] = gen
    return out


def scan_summary():
    s = pd.read_csv(OUT / "scan.csv")
    p = s[(s["cov"] == "C") & (s.universe == "supp")].copy()
    allp = pd.concat([p[f"{y}_p"] for y in GATES + ["S"]])
    q = pd.Series(ST.bh(allp.fillna(1).tolist()), index=allp.index)
    k = 0
    for y in GATES + ["S"]:
        p[f"{y}_q"] = q.iloc[k:k + len(p)].to_numpy()
        k += len(p)
    p.to_csv(OUT / "scan_primary_q.csv", index=False)
    p["key"] = p.property
    res = {}
    for y in GATES + ["S"]:
        sig = p.assign(sig=(p[f"{y}_q"] < 0.05) & (np.sign(p[y]) != 0)).groupby(["key", "tissue"]).sig.any().groupby("key").sum()
        res[y] = classify(tissue_table(p, y), sig)
    out = pd.concat({y: r for y, r in res.items()}, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    for cov in ("Ctau", "Ccls"):
        sc = s[(s["cov"] == cov) & (s.universe == "supp")].assign(key=lambda d: d.property)
        for y in GATES:
            tm = tissue_table(sc, y)
            out[f"{cov}_{y}_median"] = tm.median(1)
            out[f"{cov}_{y}_same"] = np.sign(tm).eq(np.sign(tm.median(1)), axis=0).sum(1)
    out["B1_median"] = tissue_table(p, "B1").median(1)
    out["B1_same"] = np.sign(tissue_table(p, "B1")).eq(np.sign(out["B1_median"]), axis=0).sum(1)
    a = s[(s.universe == "all")].assign(key=lambda d: d.property)
    out["D_can_allgenes_median"] = tissue_table(a, "D_can").median(1)
    sgn = np.sign(out.D_can_median) == np.sign(out.D_free_median)
    out["verdict"] = np.where(out.D_can_general & out.D_free_general & sgn, "ROBUST",
                              np.where(out.D_free_general, "ONE-GATE free",
                                       np.where(out.D_can_general, "ONE-GATE canonical", "NEGATIVE")))
    out.index.name = "property"
    out = out.sort_values("D_free_median")
    out.to_csv(OUT / "scan_summary.csv")
    # tissue values for figures
    tv = []
    for y in GATES + ["S", "B1"]:
        t = tissue_table(p, y).stack().rename("rho_p").reset_index().assign(readout=y)
        tv.append(t)
    for y in GATES:
        sc = s[(s["cov"] == "Ctau") & (s.universe == "supp")].assign(key=lambda d: d.property)
        tv.append(tissue_table(sc, y).stack().rename("rho_p").reset_index().assign(readout=f"{y}_tau"))
    pd.concat(tv).rename(columns={"key": "property"}).to_csv(OUT / "scan_tissue_values.csv", index=False)
    return out


def family_summary():
    f = pd.read_csv(OUT / "family.csv")
    cl = pd.read_csv(gene_ids.REPO_ROOT / "data/analysis/E2/pfam/Pfam-A.clans.tsv.gz", sep="\t", header=None,
                     names=["pf", "clan", "clanname", "name", "desc"])
    nm = cl.set_index("pf").desc.to_dict()
    ys = ["D_can", "D_free", "S", "tau_D_can", "tau_D_free", "B1"]
    allp = pd.concat([f[f"{y}_p"] for y in GATES])
    q = pd.Series(ST.bh(allp.fillna(1).tolist()), index=allp.index)
    f["D_can_q"] = q.iloc[:len(f)].to_numpy()
    f["D_free_q"] = q.iloc[len(f):].to_numpy()
    f["key"] = f.kind + ":" + f.family
    res = {}
    for y in ys:
        sig = None
        if y in GATES:
            sig = f.assign(sig=f[f"{y}_q"] < 0.05).groupby(["key", "tissue"]).sig.any().groupby("key").sum()
        res[y] = classify(tissue_table(f, f"{y}_mean"), sig)
    out = pd.concat({y: r for y, r in res.items()}, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    sgn = np.sign(out.D_can_median) == np.sign(out.D_free_median)
    out["verdict"] = np.where(out.D_can_general & out.D_free_general & sgn, "ROBUST",
                              np.where(out.D_free_general, "ONE-GATE free",
                                       np.where(out.D_can_general, "ONE-GATE canonical", "NEGATIVE")))
    out["n_genes_median"] = f.groupby("key").n.median()
    out["desc"] = [nm.get(k.split(":", 1)[1], k.split(":", 1)[1]) for k in out.index]
    out.index.name = "family"
    out = out.sort_values("D_free_median")
    out.to_csv(OUT / "family_summary.csv")
    tv = [tissue_table(f, f"{y}_mean").stack().rename("mean_resid").reset_index().assign(readout=y) for y in ys]
    pd.concat(tv).rename(columns={"key": "family"}).to_csv(OUT / "family_tissue_values.csv", index=False)
    return out


def complex_summary():
    x = pd.read_csv(OUT / "complex.csv")
    cm = pd.read_parquet(OUT / "complex_members.parquet")
    G = pd.read_parquet(OUT / "gene_resid.parquet", columns=["cohort", "hsapiens_gene_id"])
    rows, keep_all = [], []
    for c, d in x[x.guard].groupby("cohort"):
        genes = set(G.loc[G.cohort == c, "hsapiens_gene_id"])
        mem = cm[cm.g.isin(genes) & cm.complex.isin(d.complex)].groupby("complex").g.apply(frozenset)
        # non-redundant: identical member sets once; then greedy by size, drop Jaccard > 0.5 with a kept one
        order = mem.map(len).sort_values(ascending=False).index
        kept = []
        for k in order:
            s = mem[k]
            if all(len(s & mem[j]) / len(s | mem[j]) <= 0.5 for j in kept):
                kept.append(k)
        dd = d[d.complex.isin(kept)]
        keep_all.append(dd)
        r = dict(cohort=c, tissue=d.tissue.iloc[0], n_complexes=len(dd))
        for y in ["D_can", "D_free", "S", "tau_D_free", "B1"]:
            z = dd[f"{y}_z"].dropna()
            r[f"{y}_mean_z"] = z.mean()
            r[f"{y}_frac_neg"] = (z < 0).mean()
            r[f"{y}_p_sign"] = sps.binomtest(int((z < 0).sum()), len(z), 0.5).pvalue if len(z) else np.nan
            r[f"{y}_frac_p05"] = (dd[f"{y}_pless"] < 0.05).mean()
        rows.append(r)
    cs = pd.DataFrame(rows)
    cs.to_csv(OUT / "complex_coherence.csv", index=False)
    K = pd.concat(keep_all)
    K.to_csv(OUT / "complex_nonredundant.csv", index=False)
    # complexes placed above or below across tissues (all, non-guarded means)
    d = x[~x.guard].copy()
    d["key"] = d.complex + " | " + d.name
    res = {}
    for y in ["D_can", "D_free", "tau_D_free", "B1"]:
        tm = tissue_table(d, f"{y}_mean")
        res[y] = pd.DataFrame({"median": tm.median(1), "n_tissues": tm.notna().sum(1),
                               "same_sign": np.sign(tm).eq(np.sign(tm.median(1)), axis=0).sum(1)})
    out = pd.concat(res, axis=1)
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_members_median"] = d.groupby("key").n.median()
    out.index.name = "complex"
    out.sort_values("D_free_median").to_csv(OUT / "complex_placement.csv")
    return cs, out


def age_and_duplication():
    G = pd.read_parquet(OUT / "gene_resid.parquet")
    P = pd.read_parquet(OUT / "props.parquet", columns=["hsapiens_gene_id", "gene_age", "ohnolog", "paralog_count",
                                                         "phaplo", "ptriplo", "tf"])
    G = G.merge(P, on="hsapiens_gene_id", how="left")
    names = {v: n for v, n, _ in RU.STRATA}
    names[RU.EUTHERIA] = "Eutheria"
    ys = ["r_D_can", "r_D_free", "rt_D_free", "r_B1"]
    a = G.groupby(["tissue", "cohort", "gene_age"])[ys].agg(["mean", "size"])
    a.columns = [f"{c}_{s}" for c, s in a.columns]
    a = a.reset_index()
    t = a.groupby(["tissue", "gene_age"])[[f"{y}_mean" for y in ys]].median().reset_index()
    t["stratum"] = t.gene_age.map(names)
    t.to_csv(OUT / "age_profile.csv", index=False)
    # duplication class among genes with >= 1 human paralog: ohnolog vs small-scale duplicate (SSD) only; singletons beside
    G["dup_class"] = np.where(G.ohnolog > 0, "ohnolog", np.where(G.paralog_count > 0, "SSD", "singleton"))
    rows = []
    for (c, tis), d in G.groupby(["cohort", "tissue"]):
        r = dict(cohort=c, tissue=tis)
        for y in ys:
            v = {k: d.loc[d.dup_class == k, y].dropna() for k in ("ohnolog", "SSD", "singleton")}
            for k in v:
                r[f"{y}_{k}"] = v[k].mean()
            r[f"{y}_ohno_minus_ssd"] = v["ohnolog"].mean() - v["SSD"].mean()
            r[f"{y}_p"] = sps.mannwhitneyu(v["ohnolog"], v["SSD"]).pvalue
        r.update({f"n_{k}": int((d.dup_class == k).sum()) for k in ("ohnolog", "SSD", "singleton")})
        rows.append(r)
    dc = pd.DataFrame(rows)
    dc.to_csv(OUT / "dup_class.csv", index=False)
    return t, dc


def joint_summary():
    j = pd.read_csv(OUT / "joint.csv")
    j = j[j.property != "log_length"]
    j["key"] = j.readout + ":" + j.property
    tm = tissue_table(j, "beta")
    out = pd.DataFrame({"median_beta": tm.median(1), "same_sign": np.sign(tm).eq(np.sign(tm.median(1)), axis=0).sum(1),
                        "n_tissues": tm.notna().sum(1)})
    out.index.name = "readout_property"
    out.to_csv(OUT / "joint_summary.csv")
    return out


def named_genes(fam: pd.DataFrame):
    """Genes passing the naming rule in both tables; per family/property group, count cohorts and tissues."""
    G = pd.read_parquet(OUT / "gene_resid.parquet")
    P = pd.read_parquet(OUT / "props.parquet", columns=["hsapiens_gene_id", "pfam_list", "ohnolog", "tf_dbd",
                                                         "phaplo", "ptriplo", "hpa_loc_plasma_membrane"])
    ok = (G.nameable.astype(bool) & G.nameable_free.astype(bool) & ~G.saturated_flag.astype(bool)
          & ~G.saturated_flag_free.astype(bool) & G.sign_D_stable.astype(bool) & G.sign_D_stable_free.astype(bool)
          & (np.sign(G.D_can) == np.sign(G.D_free)))
    N = G[ok].merge(P, on="hsapiens_gene_id", how="left")
    agg = N.groupby(["hsapiens_gene_id", "symbol"]).agg(
        n_cohorts=("cohort", "size"), n_tissues=("tissue", "nunique"),
        r_D_can=("r_D_can", "median"), r_D_free=("r_D_free", "median"), sign_frac_pos=("D_free", lambda v: (v > 0).mean()),
        pfam=("pfam_list", "first"), ohnolog=("ohnolog", "first"), tf_dbd=("tf_dbd", "first"),
        phaplo=("phaplo", "first"), ptriplo=("ptriplo", "first")).reset_index()
    agg = agg[(agg.n_cohorts >= 3) & (agg.n_tissues >= 2)]
    agg["consistent"] = (agg.sign_frac_pos == 1) | (agg.sign_frac_pos == 0)
    agg.to_csv(OUT / "named_genes_all.csv", index=False)
    rows = []
    robust = fam[fam.verdict == "ROBUST"]
    for key, r in robust.iterrows():
        kind, f = key.split(":", 1)
        if kind == "pfam":
            m = agg.pfam.fillna("").str.split(";").map(lambda L: f in L)
        else:
            m = agg.tf_dbd == f
        sub = agg[m & agg.consistent & (np.sign(agg.r_D_free) == np.sign(r.D_free_median))]
        sub = sub.reindex(sub.r_D_free.abs().sort_values(ascending=False).index)
        rows.append(dict(family=key, desc=r.desc, direction="above" if r.D_free_median > 0 else "below",
                         n_named=len(sub), genes=", ".join(f"{s} ({n})" for s, n in zip(sub.symbol.head(12), sub.n_tissues.head(12)))))
    pd.DataFrame(rows).to_csv(OUT / "named_by_family.csv", index=False)
    return agg


def main():
    sc = scan_summary()
    fam = family_summary()
    cs, cp = complex_summary()
    age, dc = age_and_duplication()
    js = joint_summary()
    named_genes(fam)
    pd.set_option("display.width", 250)
    print(sc[["D_can_median", "D_can_same_sign", "D_can_n_tissue_sig", "D_free_median", "D_free_same_sign",
              "D_free_n_tissue_sig", "Ctau_D_free_median", "B1_median", "verdict"]].round(3).to_string())
    print(fam.verdict.value_counts())
    print(cs.round(3).to_string())


if __name__ == "__main__":
    main()
