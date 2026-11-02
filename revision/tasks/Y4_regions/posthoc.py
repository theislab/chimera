"""Y4 (EXPLORATORY): checks on the survivors, chosen after even genes were read. None of them counts as a replication.

1. Mechanism of the level survivor: rho_p of the direct margin's pattern and level terms with EVEE, inside and outside the region.
2. A second, independent level gold: TS vs TMS per-type level agreement (X4 cache), inside and outside the region.
3. The region by predominant type: hepatocyte-peak against non-parenchymal-peak genes.
4. Named genes with the readout's raw sign matching its residual sign, with per-type expression in each species.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["OPENBLAS_NUM_THREADS"] = "2"

import numpy as np
import pandas as pd

from revision.tasks.Y4_regions import replicate as RP
from revision.tasks.Y4_regions import rules as RU
from revision.tasks.Y4_regions import screen as SC

OUT = SC.OUT
TYPES = ["hepatocyte", "macrophage", "endothelial", "NK", "B", "stellate"]


def atlas_level(t: pd.DataFrame) -> pd.Series:
    """-mean over types of |TS - TMS| log expression, each type's difference centred on its median over genes detected in both."""
    a = pd.read_parquet(SC.REPO / "results/revision/X4/cache/atlas_h3.parquet")
    ds = []
    for k in TYPES:
        if f"TS_x_{k}" in a and f"TMS_x_{k}" in a:
            d = a[f"TMS_x_{k}"] - a[f"TS_x_{k}"]
            ok = (a[f"TS_f_{k}"] >= 0.01) & (a[f"TMS_f_{k}"] >= 0.01)
            ds.append((d - d[ok].median()).where(ok))
    D = pd.concat(ds, axis=1)
    lev = -D.abs().mean(1).where(D.notna().sum(1) >= 2)
    return t["hsapiens_gene_id"].map(lev)


def poly_rho(T: pd.DataFrame, x: str, g: str, covs: list[str]) -> float:
    """Independent of the task's statistics code: rank-order, cubic polynomial of covariate ranks, OLS residual correlation."""
    R = T[[x, g] + covs].rank()
    R = (R - R.mean()) / R.std()
    Z = np.column_stack([np.ones(len(R))] + [R[c].to_numpy() ** k for c in covs for k in (1, 2, 3)])
    e = [R[k].to_numpy() - Z @ np.linalg.lstsq(Z, R[k].to_numpy(), rcond=None)[0] for k in (x, g)]
    return float(np.corrcoef(e[0], e[1])[0, 1])


def extra_checks():
    rows = []
    # 1. all genes against T4.3
    for c, g in (("liv_hmouse", "G_H1"), ("liv_hmouse", "G_H3"), ("test_hmouse", "G_H2"), ("test_hmouse", "G_H4")):
        t = SC.table(c)
        cv = RP.cov_of(c, g)
        rows += RP.simple(c, np.ones(len(t), bool), g, cv, ["D", "S"], draws=500, label=dict(check="all_genes_vs_T4_3"))
    # 2. independent recomputation and spline design, for the replicated small regions, even genes
    for c, g, x, reg in (("liv_hmouse", "G_H1", "S", "expr_T2__tau_T3"), ("liv_hmouse", "G_H3", "S", "pihi__tau_T2"),
                         ("liv_hmouse", "G_H3", "D", "pilo__spread_T3"), ("liv_hmouse", "G_H3", "S", "expr_T2__spread_T1")):
        t = SC.table(c)
        cv = RP.cov_of(c, g)
        m = RP.even_rows(t, RP.mask_of(c, reg)) & t[[g] + cv].notna().all(1).to_numpy()
        T = t.loc[m]
        from revision.tasks.T4_3_scoreboard import stats as ST

        spline = float(ST.rho_p_matrix(np.column_stack([ST.midrank(T[g].to_numpy(float)), ST.midrank(T[x].to_numpy(float))]), ST.design(T, cv))[1])
        rows.append(dict(check="even_recompute", cohort=c, gold=g, score=x, region=reg, n=int(m.sum()), rho_poly=poly_rho(T, x, g, cv), rho_spline_df4=spline))
    # 3. given tau, all liver genes and by tau band (level gold is otherwise adjusted on C without tau)
    c = "liv_hmouse"
    t = SC.table(c)
    rows += RP.simple(c, np.ones(len(t), bool), "G_H1", RU.C, ["S", "Sdir_pattern", "Sdir_level", "B1c_raw"], draws=1000, label=dict(check="level_all_genes_C"))
    rows += RP.simple(c, np.ones(len(t), bool), "G_H1", RU.C_PAT, ["S", "Sdir_pattern", "Sdir_level", "B1c_raw"], draws=1000, label=dict(check="level_all_genes_C_plus_tau"))
    m = RP.mask_of(c, "expr_T2__tau_T3") & t["split"].isin(["odd", "even"]).to_numpy()
    rows += RP.simple(c, m, "G_H1", RU.C_PAT, ["S", "Sdir_pattern", "Sdir_level", "B1c_raw"], draws=1000, label=dict(check="level_region_C_plus_tau"))
    # 4. uncapped identity as a predictor inside the high-pi regions, even genes
    for c, g in (("liv_hmouse", "G_H3"), ("test_hmouse", "G_H4")):
        t = SC.table(c)
        cv = [k for k in RP.cov_of(c, g) if k != "prior_raw"]
        m = RP.even_rows(t, RP.mask_of(c, "pihi__tau_T2"))
        rows += RP.simple(c, m, g, cv, ["id_true", "S"], draws=2000, label=dict(check="high_pi_identity_vs_S"))
    pd.DataFrame(rows).to_csv(OUT / "posthoc_checks.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


def main():
    rows = []
    c = "liv_hmouse"
    t = SC.table(c)
    t["G_atlas_level"] = atlas_level(t)
    for reg in ("expr_T2__tau_T3", "tau_Q4"):
        m = RP.mask_of(c, reg)
        for sp in ("odd", "even"):
            for lab, rr in (("region", m), ("rest", ~m)):
                rw = rr & t["split"].eq(sp).to_numpy()
                for g, cv, what in (("G_H1", RU.C, "EVEE"), ("G_atlas_level", RU.C, "atlas_level")):
                    for r in RP.simple(c, rw, g, cv, ["S", "D", "Sdir", "Sdir_pattern", "Sdir_level", "B1c_raw", "B1b_raw"], draws=2000,
                                       label=dict(region=reg, split=sp, part=lab, what=what)):
                        rows.append(r)
    pd.DataFrame(rows).to_csv(OUT / "posthoc_level_terms.csv", index=False)
    rows = []
    for reg, g, x in (("expr_T2__tau_T3", "G_H1", "S"), ("tau_Q4", "G_H1", "S")):
        m = RP.mask_of(c, reg)
        pk = t["peak_y"].astype(object)
        for lab, sel in (("hepatocyte_peak", pk.eq("Hepatocytes")), ("non_hepatocyte_peak", pk.notna() & ~pk.eq("Hepatocytes"))):
            rw = m & sel.to_numpy() & t["split"].isin(["odd", "even"]).to_numpy()
            rows += RP.simple(c, rw, g, RU.C, [x, "B1c_raw", "Sdir"], draws=2000, label=dict(region=reg, part=lab))
    # the same split outside the region, for comparison
    pk = t["peak_y"].astype(object)
    rest = ~RP.mask_of(c, "tau_Q4")
    for lab, sel in (("hepatocyte_peak", pk.eq("Hepatocytes")), ("non_hepatocyte_peak", pk.notna() & ~pk.eq("Hepatocytes"))):
        rw = rest & sel.to_numpy() & t["split"].isin(["odd", "even"]).to_numpy()
        rows += RP.simple(c, rw, "G_H1", RU.C, ["S", "B1c_raw", "Sdir"], draws=2000, label=dict(region="outside_tau_Q4", part=lab))
    pd.DataFrame(rows).to_csv(OUT / "posthoc_level_peak.csv", index=False)
    extra_checks()
    n = pd.read_csv(OUT / "named_genes.csv")
    n["unit_readout"] = n["unit"].str.split("|").str[2]
    rs = [np.sign(r[f"r_{r.unit_readout}"]) == np.sign(r["S"] if r.unit_readout == "S" else r["D_raw"]) for _, r in n.iterrows()]
    n = n[rs]
    n.to_csv(OUT / "named_genes_sign_checked.csv", index=False)
    print(n.groupby("unit").size())


if __name__ == "__main__":
    main()
