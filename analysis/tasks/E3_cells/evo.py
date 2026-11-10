"""E3 analysis C, continued: do species-specific departures in the 4-primate cortex fit replicate in independent
data, and which programmes depart on which branch. Run after ``departures.py``. Post hoc beside the card,
labelled in the ledger.

* MTG check (Jorstad 2023 MTG, never trained; C1 golds): a gene departing on the marmoset branch more than on the
  macaque branch should show a larger human-marmoset than human-macaque level gap in MTG (Spearman of the two
  differences, partial on expression and detection by rank residuals; gene bootstrap 1,000). Human departures
  against an MTG human-specific score: min(|gap macaque|, |gap marmoset|) where the two signed gaps agree, else 0.
* Ma 2022 human-lineage genes (C1 references.tsv, set human_lineage with a cell type): is the human row a
  departure in the named type, against the cohort's human departure rate in that type.
* Programmes: matched excess (C_pat) of each species' departure fraction for H1's 72 sets.

    PYTHONPATH=$PWD python analysis/tasks/E3_cells/evo.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

from analysis.tasks.E3_cells import compartments as CP
from analysis.tasks.E3_cells import departures as DP
from analysis.tasks.E3_cells import rules as R
from analysis.tasks.H1_harmonise import rules as RU

OUT = R.OUT / "departures"
READS = ["canonical", "per-gene", "direct"]


def resid(y, Z):
    ok = np.isfinite(y) & np.isfinite(Z).all(1)
    A = np.column_stack([np.ones(ok.sum())] + [rankdata(Z[ok, j]) for j in range(Z.shape[1])])
    r = np.full(len(y), np.nan)
    yr = rankdata(y[ok])
    r[ok] = yr - A @ np.linalg.lstsq(A, yr, rcond=None)[0]
    return r


def boot_rho(x, y, B=1000, seed=0):
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    rho = spearmanr(x, y)[0]
    rng = np.random.default_rng(seed)
    bs = [spearmanr(x[i], y[i])[0] for i in (rng.integers(0, len(x), len(x)) for _ in range(B))]
    return rho, np.percentile(bs, 2.5), np.percentile(bs, 97.5), int(ok.sum())


def mtg_check(c: str = "brain_dlpfc") -> pd.DataFrame:
    gd = pd.read_parquet(OUT / f"genes_{c}.parquet")
    g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet").reindex(gd.index)
    mac = pd.read_parquet(R.RES / "C1" / "golds" / "MTG_macaque.parquet").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    mar = pd.read_parquet(R.RES / "C1" / "golds" / "MTG_marmoset.parquet").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    ids = gd.hsapiens_gene_id
    gm, gk = mac.reindex(ids), mar.reindex(ids)
    Z = g[["expr", "detection"]].to_numpy(float)
    y_branch = resid((gk.level_gap - gm.level_gap).to_numpy(float), Z)
    agree = np.sign(gm.signed_gap.to_numpy()) == np.sign(gk.signed_gap.to_numpy())
    y_h = resid(np.where(agree, np.minimum(gm.level_gap.abs(), gk.level_gap.abs()), 0.0), Z)
    sup = gd.count_support.to_numpy()
    rows = []
    for rd in READS:
        if f"{rd}__Marmoset" not in gd:
            continue
        xb = resid((gd[f"{rd}__Marmoset"] - gd[f"{rd}__Macaque"]).to_numpy(float), Z)
        xb[~sup] = np.nan
        rows.append(dict(cohort=c, readout=rd, test="marmoset minus macaque branch vs MTG gap difference", **dict(zip(["rho", "lo", "hi", "n"], boot_rho(xb, y_branch)))))
        xh = resid(gd[f"{rd}__Human"].to_numpy(float), Z)
        xh[~sup] = np.nan
        rows.append(dict(cohort=c, readout=rd, test="human departures vs MTG human-specific gap", **dict(zip(["rho", "lo", "hi", "n"], boot_rho(xh, y_h)))))
        xc = resid(gd[f"{rd}__Chimpanzee"].to_numpy(float), Z)
        xc[~sup] = np.nan
        rows.append(dict(cohort=c, readout=rd, test="chimpanzee departures vs MTG human-specific gap (control)", **dict(zip(["rho", "lo", "hi", "n"], boot_rho(xc, y_h)))))
    return pd.DataFrame(rows)


def ma2022(c: str = "brain_dlpfc") -> pd.DataFrame:
    ref = pd.read_csv(RU.REPO / "analysis" / "tasks" / "C1_cohorts" / "references.tsv", sep="\t")
    ref = ref[(ref.cohort == "brain") & ref.set.isin(["human_lineage", "hominini"]) & ref.cell_type.notna()]
    b = DP.load_build(c)
    g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet")
    z = pd.read_parquet(R.RES / "Z4" / "h1" / "genes" / f"{c}.parquet").reindex(g.index)
    vn = pd.Index(b["var_names"])
    species = [str(s) for s in b["species"]]
    lab, si = b["labels"].astype(str), b["species_idx"]
    types = sorted(set(lab))
    x = np.log2(1 + b["cp10k"] * 100)
    rows = []
    sym2var = dict(zip(g.symbol, g.index))
    for rd, key in (("canonical", "S_can"), ("per-gene", "S_free"), ("direct", None)):
        M = x if key is None else b[key]
        V = np.stack([[M[np.where((si == s) & (lab == k))[0][0]] for k in types] for s in range(len(species))])
        dep = DP.departures(V, key is None)
        hi = species.index("Human")
        DET = np.stack([[b["det"][np.where((si == s) & (lab == k))[0][0]] for k in types] for s in range(len(species))])
        expressed = (DET >= DP.DET_EXPR).any(0)
        for _, r in ref.iterrows():
            if r.gene not in sym2var or r.cell_type not in types:
                continue
            j, t = vn.get_loc(sym2var[r.gene]), types.index(r.cell_type)
            base = dep[hi, t][expressed[t]].mean()
            rows.append(dict(readout=rd, set=r.set, gene=r.gene, cell_type=r.cell_type, expressed=bool(expressed[t, j]),
                             human_departs=bool(dep[hi, t, j]), any_ape_departs=bool(dep[hi, t, j] or dep[species.index("Chimpanzee"), t, j]),
                             base_rate=base, count_support=bool(g.count_support.iloc[j]), S_rows=";".join(f"{species[s]}:{V[s, t, j]:.3f}" for s in range(len(species)))))
    return pd.DataFrame(rows)


def programmes(c: str = "brain_dlpfc") -> pd.DataFrame:
    gd = pd.read_parquet(OUT / f"genes_{c}.parquet")
    g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet").reindex(gd.index)
    sets = pd.read_csv(RU.OUT / "sets_membership.csv")
    sup = gd.count_support.to_numpy()
    Zr = g.loc[sup, R.COV_PAT].to_numpy(float)
    ids = gd.hsapiens_gene_id[sup].to_numpy()
    rng = np.random.default_rng(R.SEED)
    rows = []
    cols = [k for k in gd.columns if "__" in k]
    for key, m in sets.groupby("key"):
        flag = np.isin(ids, m.hsapiens_gene_id.unique())
        if flag.sum() < 15:
            continue
        for col in cols:
            rd, sp = col.split("__")
            r = CP.knn_excess(gd.loc[sup, col].to_numpy(float), Zr, flag, R.B_DRAWS, rng)
            rows.append(dict(cohort=c, set=key, readout=rd, species=sp, **r))
    return pd.DataFrame(rows)


def main() -> int:
    mtg_check().to_csv(OUT / "mtg_check.csv", index=False)
    ma2022().to_csv(OUT / "ma2022_genes.csv", index=False)
    progs = [programmes(c) for c in ("brain_dlpfc", "vz_full", "liv_4sp", "brain_m1", "bone_marrow", "lung_healthy", "test_3sp")]
    pd.concat(progs).to_csv(OUT / "programmes.csv", index=False)
    print(pd.read_csv(OUT / "mtg_check.csv").round(3).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
