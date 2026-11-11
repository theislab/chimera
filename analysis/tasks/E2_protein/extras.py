"""Follow-ups to the scan of this task, written after reading the scan (post hoc).

1. Dosage split: pHaplo given pTriplo and pTriplo given pHaplo (rank residuals on C, then on each other).
2. Complex coherence without core machinery (ribosome, proteasome, spliceosome, respiratory chain, RNA polymerase, mediator,
   translation initiation, nuclear pore, exosome, CCT, ATP synthase) by a fixed regex on the complex name.
3. Gene age within duplication classes: is the vertebrate peak only the 2R ohnologs?
4. Young genes (Eutheria to Tetrapoda strata) against older genes.
5. Immune proteins by tissue.

Usage (from chimera/):  mamba run -n chimera_env python -m analysis.tasks.E2_protein.extras
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as sps

from analysis.common import gene_ids
from analysis.tasks.E2_protein import rules as RU
from analysis.tasks.T4_3_scoreboard import stats as ST

OUT = gene_ids.REPO_ROOT / "results" / "analysis" / "E2"
H1G = gene_ids.REPO_ROOT / "results" / "analysis" / "H1" / "genes"
MACHINERY = (r"ribosom|proteasom|spliceosom|respiratory|NADH|cytochrome|ATP synthase|RNA polymerase|Mediator|"
             r"initiation factor|nuclear pore|exosome|chaperonin|CCT|TRiC|snRNP|Integrator|translocase")


def dosage_split() -> pd.DataFrame:
    G = pd.read_parquet(OUT / "gene_resid.parquet")
    P = pd.read_parquet(OUT / "props.parquet", columns=["hsapiens_gene_id", "phaplo", "ptriplo", "log_length"])
    rows = []
    for c, d in G.groupby("cohort"):
        h = pd.read_parquet(H1G / f"{c}.parquet", columns=["hsapiens_gene_id", "detection"])
        d = d.merge(P, on="hsapiens_gene_id").merge(h, on="hsapiens_gene_id").dropna(subset=["phaplo", "ptriplo", "log_length"])
        d = d.reset_index(drop=True)
        Z = ST.design(d, ["pi_raw", "expr", "detection", "log_length"])
        rh = ST.resid(ST.midrank(d.phaplo.to_numpy())[:, None], Z)[:, 0]
        rt = ST.resid(ST.midrank(d.ptriplo.to_numpy())[:, None], Z)[:, 0]
        r = dict(cohort=c, tissue=d.tissue.iloc[0], n=len(d), rho_haplo_triplo=float(np.corrcoef(rh, rt)[0, 1]))
        for y in ["r_D_can", "r_D_free", "rt_D_free", "r_B1"]:
            v = d[y].to_numpy()
            ok = np.isfinite(v)
            X = np.column_stack([np.ones(ok.sum()), rh[ok] / rh[ok].std(), rt[ok] / rt[ok].std()])
            b = np.linalg.lstsq(X, v[ok], rcond=None)[0]
            r[f"{y}_haplo_given_triplo"], r[f"{y}_triplo_given_haplo"] = float(b[1]), float(b[2])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "dosage_split.csv", index=False)
    return out


def complex_no_machinery() -> pd.DataFrame:
    K = pd.read_csv(OUT / "complex_nonredundant.csv")
    K = K[~K.name.str.contains(MACHINERY, case=False, regex=True)]
    rows = []
    for c, dd in K.groupby("cohort"):
        r = dict(cohort=c, tissue=dd.tissue.iloc[0], n_complexes=len(dd))
        for y in ["D_can", "D_free", "tau_D_free", "B1"]:
            z = dd[f"{y}_z"].dropna()
            r[f"{y}_mean_z"] = z.mean()
            r[f"{y}_frac_neg"] = (z < 0).mean()
            r[f"{y}_p_sign"] = sps.binomtest(int((z < 0).sum()), len(z), 0.5).pvalue
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "complex_coherence_no_machinery.csv", index=False)
    return out


def age_by_dup() -> pd.DataFrame:
    G = pd.read_parquet(OUT / "gene_resid.parquet")
    P = pd.read_parquet(OUT / "props.parquet", columns=["hsapiens_gene_id", "gene_age", "ohnolog", "paralog_count"])
    G = G.merge(P, on="hsapiens_gene_id")
    G["dup"] = np.where(G.ohnolog > 0, "ohnolog", np.where(G.paralog_count > 0, "SSD", "singleton"))
    G["age_bin"] = pd.cut(G.gene_age, [-1, 4, 5, 6, 7, 8, 9, 10],
                          labels=["post-tetrapod", "Euteleostomi", "Gnathostomata", "Vertebrata", "Olfactores", "Bilateria",
                                  "Opisthokonta"])
    a = G.groupby(["tissue", "cohort", "dup", "age_bin"], observed=True)[["r_D_free", "r_D_can"]].agg(["mean", "size"])
    a.columns = [f"{x}_{y}" for x, y in a.columns]
    a = a.reset_index()
    a = a[a.r_D_free_size >= 10]
    t = a.groupby(["dup", "age_bin", "tissue"], observed=True)[["r_D_free_mean", "r_D_can_mean"]].median().reset_index()
    s = t.groupby(["dup", "age_bin"], observed=True).agg(D_free=("r_D_free_mean", "median"), D_can=("r_D_can_mean", "median"),
                                                          n_tissues=("tissue", "size"),
                                                          free_pos=("r_D_free_mean", lambda v: int((v > 0).sum()))).reset_index()
    s.to_csv(OUT / "age_by_dup.csv", index=False)
    # young genes: strata 0-4 against all older, per cohort
    rows = []
    for (c, tis), d in G.groupby(["cohort", "tissue"]):
        y, o = d[d.gene_age <= 4], d[d.gene_age > 4]
        r = dict(cohort=c, tissue=tis, n_young=len(y))
        for k in ["r_D_free", "r_D_can", "r_B1"]:
            r[f"{k}_young_minus_old"] = y[k].mean() - o[k].mean()
            r[f"{k}_p"] = sps.mannwhitneyu(y[k].dropna(), o[k].dropna()).pvalue if len(y) > 5 else np.nan
        rows.append(r)
    yo = pd.DataFrame(rows)
    yo.to_csv(OUT / "young_genes.csv", index=False)
    return s, yo


def level_gap() -> pd.DataFrame:
    """Post hoc: does a property relate to a detection gap between species (a crude level proxy) at equal C?
    gap = SD over species of logit detection (pairs: |difference| / sqrt 2). Lower gap = closer level across species."""
    G = pd.read_parquet(OUT / "gene_resid.parquet", columns=["cohort", "tissue", "hsapiens_gene_id", "r_D_free", "r_B1"])
    P = pd.read_parquet(OUT / "props.parquet")
    P["rrm"] = P.pfam_list.fillna("").str.contains("PF00076").astype(float)
    props = ["hpa_secreted", "hpa_secreted_blood", "hpa_loc_er_golgi", "transmembrane", "phaplo", "ptriplo", "rrm",
             "ohnolog", "hpa_loc_plasma_membrane", "depmap_essentiality", "complex_member"]
    rows = []
    for c, d in G.groupby("cohort"):
        h = pd.read_parquet(H1G / f"{c}.parquet")
        dc = [k for k in h.columns if k.startswith("detection_") and k not in ("detection_spmean", "detection_sprange")]
        L = np.log(np.clip(h[dc].to_numpy(float), 1e-4, 1 - 1e-4) / (1 - np.clip(h[dc].to_numpy(float), 1e-4, 1 - 1e-4)))
        h = h[["hsapiens_gene_id", "pi_raw", "expr", "detection"]].assign(gap=L.std(1))
        d = d.merge(h, on="hsapiens_gene_id").merge(P[["hsapiens_gene_id", "log_length"] + props], on="hsapiens_gene_id")
        d = d.dropna(subset=["log_length", "gap"]).reset_index(drop=True)
        rg = ST.resid(ST.midrank(d.gap.to_numpy())[:, None], ST.design(d, ["pi_raw", "expr", "detection", "log_length"]))[:, 0]
        r = dict(cohort=c, tissue=d.tissue.iloc[0], rho_gap_Dfree=float(np.corrcoef(rg, d.r_D_free)[0, 1]))
        for p in props:
            m = d[p].notna().to_numpy()
            T = d[m].reset_index(drop=True)
            Z = ST.design(T, ["pi_raw", "expr", "detection", "log_length"])
            rp = ST.resid(ST.midrank(T[p].to_numpy(float))[:, None], Z)[:, 0]
            r[f"{p}_gap"] = float(np.corrcoef(rp, rg[m])[0, 1])
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "level_gap.csv", index=False)
    return out


def main():
    pd.set_option("display.width", 250)
    ds = dosage_split()
    print(ds.groupby("tissue").median(numeric_only=True).round(3).to_string())
    cn = complex_no_machinery()
    print(cn.round(3).to_string())
    s, yo = age_by_dup()
    print(s.round(3).to_string())
    print(yo.groupby("tissue").median(numeric_only=True).round(3).to_string())
    lg = level_gap()
    print(lg.groupby("tissue").median(numeric_only=True).round(3).T.to_string())
    sc = pd.read_csv(OUT / "scan_tissue_values.csv")
    print(sc[sc.property == "immune"].pivot(index="tissue", columns="readout", values="rho_p").round(3).to_string())


if __name__ == "__main__":
    main()
