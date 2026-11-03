"""E5 follow-ups written after reading the scan (POST HOC, labelled; EXPLORATORY).

Usage (from chimera/):  mamba run -n chimera_env python -m revision.tasks.E5_regulatory.extras --workers 13
Writes in results/revision/E5/:
  enhancer_split.csv     dELS count given GREAT domain size, and domain size given dELS count (C + the other), both gates, B1
  level_check.csv        partial rho of the main properties with a level-gap proxy (SD over species of logit detection) and
                         with the read-outs given that proxy (does a level gap carry the property's effect?)
  regulon_indeg.csv      regulon mean residual with TF in-degree added to C (regulon beyond being a well-regulated gene)
  regulon_marrow.csv     proliferation regulons (E2F, MYC) by tissue, both gates and B1
  quintiles.csv          residual read-outs by quintile of dELS count, TF in-degree and gene density (panel input)
  named_genes.csv        E2 naming rule on E5 residuals, with the regulatory annotations of each named gene
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats as sps  # noqa: E402

from revision.common import gene_ids  # noqa: E402
from revision.tasks.E2_protein import analyse as E2  # noqa: E402
from revision.tasks.E5_regulatory import analyse as A  # noqa: E402
from revision.tasks.E5_regulatory.neighbour import level_proxy  # noqa: E402
from revision.tasks.T4_3_scoreboard import stats as ST  # noqa: E402

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "revision" / "E5"
C = E2.C
MAIN = ["n_dels_domain", "log_great_domain", "tf_indegree", "abc_links_mean", "es_h3k27me3", "prc2_targets", "housekeeping",
        "promoter_cgi", "gene_density", "bidirectional", "mir_density", "phylop_promoter"]


def rho(T: pd.DataFrame, x: str, y: str, covs: list[str]) -> float:
    T = T.dropna(subset=[x, y] + covs).reset_index(drop=True)
    R = np.column_stack([ST.midrank(T[x].to_numpy(float)), ST.midrank(T[y].to_numpy(float))])
    return float(ST.rho_p_matrix(R, ST.design(T, covs))[1])


def run_cohort(c: str) -> dict:
    t = E2.load(c, A.props())
    tissue = str(t.tissue.iloc[0])
    U = t[t.supp].dropna(subset=C).reset_index(drop=True)
    U["lvl"] = U.hsapiens_gene_id.map(level_proxy(c))
    out = {}
    rows = []
    for y in ["D_can", "D_free", "S", "B1"]:
        rows.append(dict(cohort=c, tissue=tissue, readout=y,
                         dels_given_domain=rho(U, "n_dels_domain", y, C + ["log_great_domain"]),
                         domain_given_dels=rho(U, "log_great_domain", y, C + ["n_dels_domain"]),
                         ccre50_given_density=rho(U, "n_ccre_50kb", y, C + ["gene_density"]),
                         density_given_hk=rho(U, "gene_density", y, C + ["gtex_tau", "housekeeping", "cpg_oe_promoter"])))
    out["enh"] = pd.DataFrame(rows)
    rows = []
    for p in MAIN:
        r = dict(cohort=c, tissue=tissue, property=p, with_level=rho(U, p, "lvl", C))
        for y in ["D_can", "D_free", "B1"]:
            r[f"{y}_given_level"] = rho(U, p, y, C + ["lvl"])
        rows.append(r)
    out["lvl"] = pd.DataFrame(rows)
    # regulons beyond TF in-degree
    reg = pd.read_parquet(OUT / "regulons.parquet")
    V = U.dropna(subset=["tf_indegree"]).reset_index(drop=True)
    res = {y: E2.resid_std(V, y, C + ["tf_indegree"]) for y in ["D_can", "D_free"]}
    mb = V.B1.notna().to_numpy()
    res["B1"] = np.full(len(V), np.nan)
    res["B1"][mb] = E2.resid_std(V[mb].reset_index(drop=True), "B1", C + ["tf_indegree"])
    gi = pd.Series(np.arange(len(V)), index=V.hsapiens_gene_id)
    reg = reg[reg.g.isin(gi.index)]
    rows = []
    for tf, sub in reg.groupby("tf"):
        if len(sub) < 15:
            continue
        m = np.zeros(len(V), bool)
        m[gi[sub.g].to_numpy()] = True
        r = dict(cohort=c, tissue=tissue, tf=tf, n=int(m.sum()))
        for y, v in res.items():
            a, b = v[m & np.isfinite(v)], v[~m & np.isfinite(v)]
            r[y] = float(a.mean())
            r[f"{y}_p"] = float(sps.ttest_ind(a, b, equal_var=False).pvalue)
        rows.append(r)
    out["reg"] = pd.DataFrame(rows)
    # quintile profiles
    G = pd.read_parquet(OUT / "gene_resid.parquet")
    G = G[G.cohort == c].merge(U[["hsapiens_gene_id", "n_dels_domain", "tf_indegree", "gene_density", "housekeeping",
                                  "es_h3k27me3"]], on="hsapiens_gene_id")
    rows = []
    for p in ["n_dels_domain", "tf_indegree", "gene_density"]:
        q = pd.qcut(G[p].rank(method="first"), 5, labels=False)
        for k, d in G.groupby(q):
            rows.append(dict(cohort=c, tissue=tissue, property=p, quintile=int(k) + 1,
                             **{y: d[f"r_{y}"].mean() for y in ["D_can", "D_free", "B1"]}, rt_D_free=d.rt_D_free.mean()))
    for p in ["housekeeping", "es_h3k27me3"]:
        for k, d in G.groupby(G[p] > 0):
            rows.append(dict(cohort=c, tissue=tissue, property=p, quintile=int(k),
                             **{y: d[f"r_{y}"].mean() for y in ["D_can", "D_free", "B1"]}, rt_D_free=d.rt_D_free.mean()))
    out["q"] = pd.DataFrame(rows)
    return out


def named(props: pd.DataFrame) -> pd.DataFrame:
    G = pd.read_parquet(OUT / "gene_resid.parquet")
    ok = (G.nameable.astype(bool) & G.nameable_free.astype(bool) & ~G.saturated_flag.astype(bool)
          & ~G.saturated_flag_free.astype(bool) & G.sign_D_stable.astype(bool) & G.sign_D_stable_free.astype(bool)
          & (np.sign(G.D_can) == np.sign(G.D_free)))
    N = G[ok]
    agg = N.groupby(["hsapiens_gene_id", "symbol"]).agg(
        n_cohorts=("cohort", "size"), n_tissues=("tissue", "nunique"), r_D_can=("r_D_can", "median"),
        r_D_free=("r_D_free", "median"), r_B1=("r_B1", "median"), pi=("pi_raw", "median"),
        frac_pos=("D_free", lambda v: (v > 0).mean())).reset_index()
    agg = agg[(agg.n_cohorts >= 3) & (agg.n_tissues >= 2) & agg.frac_pos.isin([0.0, 1.0])]
    P = props.set_index("hsapiens_gene_id")
    for k in ["n_dels_domain", "log_great_domain", "tf_indegree", "abc_links_mean", "housekeeping", "promoter_cgi",
              "es_h3k27me3", "prc2_targets", "bivalent_npc", "gene_density", "mir_density", "tf", "tf_homeodomain"]:
        agg[k] = agg.hsapiens_gene_id.map(P[k])
    # top-quintile flags over all protein-coding genes
    for k in ["n_dels_domain", "tf_indegree", "gene_density"]:
        agg[f"{k}_q5"] = agg[k] >= P[k].quantile(0.8)
    agg.to_csv(OUT / "named_genes.csv", index=False)
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    cs = E2.cohorts()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_cohort, cs))
    for k, fn in (("enh", "enhancer_split.csv"), ("lvl", "level_check.csv"), ("reg", "regulon_indeg.csv"),
                  ("q", "quintiles.csv")):
        pd.concat([r[k] for r in res], ignore_index=True).to_csv(OUT / fn, index=False)
    pd.set_option("display.width", 250)
    e = pd.read_csv(OUT / "enhancer_split.csv")
    tv = e.groupby(["readout", "tissue"]).median(numeric_only=True).groupby("readout")
    print(tv.median().round(3).to_string())
    print((tv.apply(lambda d: (np.sign(d) == np.sign(d.median())).sum())).to_string())
    lv = pd.read_csv(OUT / "level_check.csv").groupby(["property", "tissue"]).median(numeric_only=True).groupby("property").median()
    print(lv.round(3).to_string())
    r = pd.read_csv(OUT / "regulon_indeg.csv").groupby(["tf", "tissue"])[["D_can", "D_free", "B1"]].median()
    tv = r.groupby("tf").median()
    tv["free_same"] = r.D_free.groupby("tf").apply(lambda v: (np.sign(v) == np.sign(v.median())).sum())
    tv["can_same"] = r.D_can.groupby("tf").apply(lambda v: (np.sign(v) == np.sign(v.median())).sum())
    tv["n_tissues"] = r.groupby("tf").size()
    tv.to_csv(OUT / "regulon_indeg_summary.csv")
    print(tv[(tv.n_tissues >= 8)].sort_values("D_free").round(3).to_string())
    f = pd.read_csv(OUT / "regulon.csv")
    prol = f[f.tf.isin(["E2F1", "E2F4", "E2F3", "E2F2", "MYC", "MAX", "FOXM1", "MYBL2", "TFDP1"])]
    prol.groupby(["tf", "tissue"])[["r_D_can_mean", "r_D_free_mean", "r_B1_mean", "n"]].median().to_csv(OUT / "regulon_marrow.csv")
    named(A.props())


if __name__ == "__main__":
    main()


# ---- second follow-up (POST HOC): stringent covariate set for the leading rows ----
STRICT_ROWS = ["n_dels_domain", "tf_indegree", "abc_links_mean", "es_h3k27me3", "prc2_targets", "bivalent_npc", "housekeeping",
               "promoter_cgi", "gene_density", "mir_density"]
STRICT_COVS = ["tau", "gtex_tau", "ohnolog", "log_pubmed_h"]


def stringent_cohort(c: str) -> pd.DataFrame:
    t = E2.load(c, A.props())
    tissue = str(t.tissue.iloc[0])
    U = t[t.supp].dropna(subset=C + STRICT_COVS + A.RU.CLASS_COVS).reset_index(drop=True)
    rows = []
    for p in STRICT_ROWS:
        T = U.dropna(subset=[p]).reset_index(drop=True)
        covs = C + [k for k in STRICT_COVS if k != p]
        extra = T[A.RU.CLASS_COVS].to_numpy(float)
        r = dict(cohort=c, tissue=tissue, property=p, n=len(T))
        for y in ["D_can", "D_free", "B1"]:
            TT = T.dropna(subset=[y]).reset_index(drop=True)
            R = np.column_stack([ST.midrank(TT[p].to_numpy(float)), ST.midrank(TT[y].to_numpy(float))])
            r[y] = float(ST.rho_p_matrix(R, ST.design(TT, covs, extra=TT[A.RU.CLASS_COVS].to_numpy(float)))[1])
        rows.append(r)
    return pd.DataFrame(rows)


def stringent(workers: int = 13):
    with ProcessPoolExecutor(workers) as ex:
        d = pd.concat(list(ex.map(stringent_cohort, E2.cohorts())), ignore_index=True)
    d.to_csv(OUT / "stringent.csv", index=False)
    tv = d.groupby(["property", "tissue"])[["D_can", "D_free", "B1"]].median()
    s = tv.groupby("property").median()
    for y in ["D_can", "D_free", "B1"]:
        s[f"{y}_same"] = tv[y].groupby("property").apply(lambda v: int((np.sign(v) == np.sign(v.median())).sum()))
    s.to_csv(OUT / "stringent_summary.csv")
    print(s.round(3).to_string())
