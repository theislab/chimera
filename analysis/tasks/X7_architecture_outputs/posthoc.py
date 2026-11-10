"""X7 (EXPLORATORY, added after the rules.py tests were read): diagnostics of the per-cell residual share and the
species margins.

    cellvar  share of the variance of the per-cell residual share (tv) explained by species, by coarse type and by
             species x type, per run; and run-level mean tv against mean kNN species purity across all 25 runs.
    adds     increments: gene x type R given the direct Ddir (and the reverse); species contrasts given the
             same-count baselines; S_Mouse given E(Human, Mouse).
    own      species-own margin with no sharing: mu1 = the species' own rate in each type x library, mu0 = the
             species rate x library, NB with gene-wise moment dispersion (as X4 direct_margin); Sown_gs = mean over
             cells of s. Compared with Chimera's S_gs and the shared direct margin (Cspdir) gene by gene, and on
             the golds. If z keeps species, the shared decoder can fit each species alone and S_gs follows Sown.
Writes results/analysis/X7/posthoc_*.csv.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.X4_where_differs import rules as XR
from analysis.tasks.X4_where_differs.direct_margin import nb_ll_diff
from analysis.tasks.X7_architecture_outputs import rules as RU
from analysis.tasks.X7_architecture_outputs.analyses import (LIVER_GOLD_SPECIES, gold_gene_type, labels_for,
                                                             residual_by_type, seed_of, species_margin, write)
from analysis.tasks.X7_architecture_outputs.baselines import BD
from analysis.tasks.X7_architecture_outputs.forward import FWD, OUT, run_table
from analysis.tasks.X7_architecture_outputs.golds import GD


def eta2(y, groups):
    y = np.asarray(y, float)
    m = pd.Series(y).groupby(np.asarray(groups)).transform("mean").to_numpy()
    return float(((m - y.mean()) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def sec_cellvar():
    rows = []
    for lab, (c, _) in run_table().items():
        t = pd.read_parquet(BD / f"{lab}__cells.parquet")
        r = dict(cohort=c, run_label=lab, seed=seed_of(lab), tv_mean=float(t.tv.mean()),
                 npur_chimera_mean=float(t.npur_chimera.mean()))
        for m in ("tv", "gain", "lshare"):
            r[f"{m}_eta2_species"] = eta2(t[m], t.species)
            r[f"{m}_eta2_type"] = eta2(t[m], t.coarse)
            r[f"{m}_eta2_species_type"] = eta2(t[m], t.species + "|" + t.coarse)
        rows.append(r)
    d = pd.DataFrame(rows)
    write(d, "posthoc_cellvar.csv")
    print("runs: Spearman(mean tv, mean purity) =", round(spearmanr(d.tv_mean, d.npur_chimera_mean)[0], 3))


def sec_adds():
    units = []
    for c in RU.GENE_TYPE_COHORTS:
        species = (LIVER_GOLD_SPECIES if c.startswith("liv")
                   else sorted(pd.read_parquet(BD / f"{c}__cells.parquet").species.unique()))
        base = CM.load(c)[["var_name", "hsapiens_gene_id", "block"]]
        gd = pd.read_parquet(BD / f"{c}__gene_type_direct.parquet").set_index("var_name")
        gold = gold_gene_type(c)
        for lab in labels_for(c):
            rb, vn = residual_by_type(lab, c, species)
            for k, I in gold.items():
                if k not in rb:
                    continue
                g = base.copy()
                g["R"] = pd.Series(rb[k].std(1), index=vn).reindex(g.var_name).to_numpy()
                Xs = np.column_stack([gd[f"x_{s}_{k}"].reindex(g.var_name).to_numpy() for s in species])
                Fs = np.column_stack([gd[f"f_{s}_{k}"].reindex(g.var_name).to_numpy() for s in species])
                ok = (Fs >= RU.DET_MIN).all(1)
                off = np.array([np.median(Xs[ok, j] - Xs[ok].mean(1)) for j in range(len(species))])
                g["Ddir"] = np.where(ok, (Xs - off[None, :]).std(1), np.nan)
                g["lev"], g["det"] = Xs.mean(1), Fs.mean(1)
                g["I"] = g.hsapiens_gene_id.map(I).to_numpy()
                lbl = dict(analysis="genetype", cohort=c, run_label=lab, seed=seed_of(lab), type=k)
                units.append(dict(g=g, gold="I", covs=["lev", "det"], scores=["R"], extra_spline=["Ddir"],
                                  label=dict(lbl, given="Ddir")))
                units.append(dict(g=g, gold="I", covs=["lev", "det"], scores=["Ddir"], extra_spline=["R"],
                                  label=dict(lbl, given="R")))
    for c in ("test_3sp", "liv_4sp"):
        g0 = CM.load(c)
        sd = pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")
        pe = pd.read_parquet(BD / f"{c}__pair_E.parquet").set_index("var_name")
        if c == "test_3sp":
            g0["Gt"] = g0.hsapiens_gene_id.map(pd.read_parquet(GD / "murat_tiers.parquet").Gt)
            g0["E_contrast"] = (pe.E_Human_Macaque - (pe.E_Human_Mouse + pe.E_Macaque_Mouse) / 2).reindex(g0.var_name).to_numpy()
            g0["dir_contrast"] = ((sd.Cspdir_Human + sd.Cspdir_Macaque) / 2 - sd.Cspdir_Mouse).reindex(g0.var_name).to_numpy()
            gold, covs = "Gt", XR.C_PAT + ["H4_nsp"]
        else:
            g0["E_contrast"] = (pe.E_Human_Mouse - pe.E_Human_Monkey).reindex(g0.var_name).to_numpy()
            g0["E_HMouse"] = pe.E_Human_Mouse.reindex(g0.var_name).to_numpy()
            g0["dir_contrast"] = (sd.Cspdir_Mouse - sd.Cspdir_Monkey).reindex(g0.var_name).to_numpy()
            gold, covs = "G_H3", XR.C_PAT
        for lab in labels_for(c):
            g = g0.copy()
            S = species_margin(lab).reindex(g.var_name)
            Sema = C.load_state(run_table()[lab][1])["S_g_ema"]
            vn = np.load(FWD / f"{lab}.npz")["var_names"]
            rows = (g.var_name.map(pd.Series(np.abs(Sema) >= RU.MASK_EPS, index=vn)).eq(True)
                    & g.nonzero_all_species.eq(True)).to_numpy()
            lbl = dict(analysis="species", cohort=c, run_label=lab, seed=seed_of(lab))
            if c == "test_3sp":
                g["Bt"] = ((S.Human + S.Macaque) / 2 - S.Mouse).to_numpy()
                units.append(dict(g=g, gold=gold, covs=covs, scores=["Bt"], extra_spline=["E_contrast", "dir_contrast"],
                                  rows=rows, label=dict(lbl, given="E_contrast+dir_contrast")))
                units.append(dict(g=g, gold=gold, covs=covs, scores=["E_contrast"], extra_spline=["Bt"],
                                  rows=rows, label=dict(lbl, given="Bt")))
            else:
                g["Bl"] = (S.Mouse - S.Monkey).to_numpy()
                g["S_Mouse"] = S.Mouse.to_numpy()
                units.append(dict(g=g, gold=gold, covs=covs, scores=["Bl"], extra_spline=["E_contrast", "E_HMouse"],
                                  rows=rows, label=dict(lbl, given="E_contrast+E_HMouse")))
                units.append(dict(g=g, gold=gold, covs=covs, scores=["S_Mouse"], extra_spline=["E_HMouse"],
                                  rows=rows, label=dict(lbl, given="E_HMouse")))
                units.append(dict(g=g, gold=gold, covs=covs, scores=["E_HMouse"], extra_spline=["S_Mouse"],
                                  rows=rows, label=dict(lbl, given="S_Mouse")))
    write(CM.run_units(units), "posthoc_adds.csv")


def own_margin(c: str) -> pd.DataFrame:
    a = C.read_adata(c)
    X = (a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = a.obs["cell_type"].astype(str)
    if c.startswith("test"):
        lab = lab.map(T42.TESTIS_COARSE)
    lab = lab.to_numpy()
    L = np.asarray(X.sum(1)).ravel()
    G = X.shape[1]
    species = sorted(set(spc))
    groups = sorted({(s, t) for s, t in zip(spc, lab)})
    rate = {}
    for s, t in groups:
        m = (spc == s) & (lab == t)
        rate[(s, t)] = (np.asarray(X[m].sum(0)).ravel() + 0.5) / (L[m].sum() + 0.5 * G)
    p = {s: (np.asarray(X[spc == s].sum(0)).ravel() + 0.5) / (L[spc == s].sum() + 0.5 * G) for s in species}
    num, den = np.zeros(G), np.zeros(G)
    for s, t in groups:
        m = np.where((spc == s) & (lab == t))[0]
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            mu = np.outer(L[mm], rate[(s, t)])
            xx = np.asarray(X[mm].todense())
            num += (((xx - mu) ** 2) - mu).sum(0)
            den += (mu ** 2).sum(0)
    th = 1.0 / np.clip(num / np.maximum(den, 1e-12), 1e-3, 1e3)
    out = {s: np.zeros(G) for s in species}
    for s, t in groups:
        m = np.where((spc == s) & (lab == t))[0]
        for i in range(0, len(m), 2000):
            mm = m[i:i + 2000]
            xx = np.asarray(X[mm].todense())
            out[s] += nb_ll_diff(xx, np.outer(L[mm], rate[(s, t)]), np.outer(L[mm], p[s]), th).sum(0)
    df = pd.DataFrame({f"Sown_{s}": out[s] / np.sum(spc == s) for s in species}, index=pd.Index(a.var_names, name="var_name"))
    return df


def sec_own():
    rows, units = [], []
    for c in ("liv_hmouse", "liv_4sp", "test_hmouse", "test_3sp"):
        own = own_margin(c)
        own.reset_index().to_parquet(BD / f"{c}__own_margin.parquet")
        sd = pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")
        K_ = len([k for k in sd.columns if k.startswith("Cspdir_")])
        for lab in labels_for(c):
            S = species_margin(lab)
            Sema = C.load_state(run_table()[lab][1])["S_g_ema"]
            unm = pd.Series(np.abs(Sema) >= RU.MASK_EPS, index=S.index)
            pur = pd.read_parquet(BD / f"{lab}__cells.parquet").groupby("species").npur_chimera.mean()
            for s in S.columns:
                m = unm.to_numpy()
                a_, b_, d_ = S[s][m], own[f"Sown_{s}"].reindex(S.index)[m], (K_ * sd[f"Cspdir_{s}"]).reindex(S.index)[m]
                rows.append(dict(cohort=c, run_label=lab, seed=seed_of(lab), species=s, n=int(m.sum()),
                                 npur_chimera=float(pur[s]), rho_S_own=float(spearmanr(a_, b_)[0]),
                                 rho_S_shareddir=float(spearmanr(a_, d_)[0]),
                                 rho_own_shareddir=float(spearmanr(b_, d_)[0])))
        if c == "liv_4sp":
            g = CM.load(c).join(own, on="var_name")
            pe = pd.read_parquet(BD / f"{c}__pair_E.parquet").set_index("var_name")
            g["E_HMouse"] = pe.E_Human_Mouse.reindex(g.var_name).to_numpy()
            S = species_margin(c).reindex(g.var_name)
            g["S_Mouse"] = S.Mouse.to_numpy()
            g["S_Monkey"] = S.Monkey.to_numpy()
            rws = (~g.masked.eq(True) & g.nonzero_all_species.eq(True)).to_numpy()
            units.append(dict(g=g, gold="G_H3", covs=XR.C_PAT, rows=rws, ref="S_Mouse",
                              scores=["S_Mouse", "Sown_Mouse", "S_Monkey", "Sown_Monkey", "E_HMouse"],
                              label=dict(cohort=c, test="own_vs_joint")))
        if c == "test_3sp":
            g = CM.load(c).join(own, on="var_name")
            g["Gt"] = g.hsapiens_gene_id.map(pd.read_parquet(GD / "murat_tiers.parquet").Gt)
            S = species_margin(c).reindex(g.var_name)
            g["Bt"] = ((S.Human + S.Macaque) / 2 - S.Mouse).to_numpy()
            g["Bt_own"] = ((g.Sown_Human + g.Sown_Macaque) / 2 - g.Sown_Mouse).to_numpy()
            rws = (~g.masked.eq(True) & g.nonzero_all_species.eq(True)).to_numpy()
            units.append(dict(g=g, gold="Gt", covs=XR.C_PAT + ["H4_nsp"], rows=rws, ref="Bt",
                              scores=["Bt", "Bt_own"], label=dict(cohort=c, test="own_vs_joint")))
    write(pd.DataFrame(rows), "posthoc_own_vs_joint.csv")
    write(CM.run_units(units), "posthoc_own_gold.csv")


def sec_checks():
    """Two checks of the only positives: liver Bl against the species-own contrast; liver fibroblast R against a
    pooled pseudobulk (all cells of the species and type, no per-library threshold)."""
    units = []
    g = CM.load("liv_4sp")
    own = pd.read_parquet(BD / "liv_4sp__own_margin.parquet").set_index("var_name")
    g = g.join(own, on="var_name")
    pe = pd.read_parquet(BD / "liv_4sp__pair_E.parquet").set_index("var_name")
    g["E_contrast"] = (pe.E_Human_Mouse - pe.E_Human_Monkey).reindex(g.var_name).to_numpy()
    g["E_HMouse"] = pe.E_Human_Mouse.reindex(g.var_name).to_numpy()
    g["Bl_own"] = g.Sown_Mouse - g.Sown_Monkey
    for lab in labels_for("liv_4sp"):
        gg = g.copy()
        S = species_margin(lab).reindex(gg.var_name)
        gg["Bl"] = (S.Mouse - S.Monkey).to_numpy()
        gg["S_Mouse"] = S.Mouse.to_numpy()
        rws = (~gg.masked.eq(True) & gg.nonzero_all_species.eq(True)).to_numpy()
        lbl = dict(check="liver_own", run_label=lab, seed=seed_of(lab))
        units.append(dict(g=gg, gold="G_H3", covs=XR.C_PAT, rows=rws, scores=["Bl", "Bl_own"], ref="Bl",
                          label=dict(lbl, given="none")))
        units.append(dict(g=gg, gold="G_H3", covs=XR.C_PAT, rows=rws, scores=["Bl"],
                          extra_spline=["E_contrast", "E_HMouse", "Bl_own"], label=dict(lbl, given="E+own")))
        units.append(dict(g=gg, gold="G_H3", covs=XR.C_PAT, rows=rws, scores=["S_Mouse"],
                          extra_spline=["E_HMouse", "Sown_Mouse"], label=dict(lbl, given="E_HMouse+Sown_Mouse")))
    # pooled pseudobulk for the liver fibroblast candidate
    for c in ("liv_hmouse", "liv_4sp"):
        a = C.read_adata(c)
        X = (a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)).astype(np.float64)
        spc = a.obs["species"].astype(str).to_numpy()
        ct = a.obs["cell_type"].astype(str).replace(RU.LIVER_MACRO).to_numpy()
        base = CM.load(c)[["var_name", "hsapiens_gene_id", "block"]]
        gold = gold_gene_type(c)
        for k in ("fibroblasts", "Hepatocytes", "Endothelial", "macrophage"):
            P, Fq = [], []
            for s_ in LIVER_GOLD_SPECIES:
                m = (spc == s_) & (ct == k)
                cnt = np.asarray(X[m].sum(0)).ravel()
                P.append(np.log2(1 + 1e6 * cnt / cnt.sum()))
                Fq.append(np.asarray((X[m] > 0).sum(0)).ravel() / m.sum())
            Xs, Fs = np.column_stack(P), np.column_stack(Fq)
            ok = (Fs >= RU.DET_MIN).all(1)
            off = np.array([np.median(Xs[ok, j] - Xs[ok].mean(1)) for j in range(2)])
            pooled = pd.Series(np.where(ok, (Xs - off).std(1), np.nan), index=a.var_names)
            for lab in labels_for(c):
                rb, vn = residual_by_type(lab, c, LIVER_GOLD_SPECIES)
                gg = base.copy()
                gg["R"] = pd.Series(rb[k].std(1), index=vn).reindex(gg.var_name).to_numpy()
                gg["Dpool"] = pooled.reindex(gg.var_name).to_numpy()
                gg["lev"] = pd.Series(Xs.mean(1), index=a.var_names).reindex(gg.var_name).to_numpy()
                gg["det"] = pd.Series(Fs.mean(1), index=a.var_names).reindex(gg.var_name).to_numpy()
                gg["I"] = gg.hsapiens_gene_id.map(gold[k]).to_numpy()
                lbl = dict(check="pooled_direct", cohort=c, type=k, run_label=lab, seed=seed_of(lab))
                units.append(dict(g=gg, gold="I", covs=["lev", "det"], scores=["R", "Dpool"], ref="R",
                                  label=dict(lbl, given="none")))
                units.append(dict(g=gg, gold="I", covs=["lev", "det"], scores=["R"], extra_spline=["Dpool"],
                                  label=dict(lbl, given="Dpool")))
    write(CM.run_units(units), "posthoc_checks.csv")


def sec_pairfit():
    """Does the 4-sp joint-fit mouse margin add beyond the human-mouse pair fit's S (seed consensus, X4 table),
    the labelled direct mouse margin of the 4-sp counts (Cspdir_Mouse) and E(Human, Mouse)?"""
    units = []
    g = CM.load("liv_4sp")
    pair = CM.load("liv_hmouse").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    g["S_pair"] = g.hsapiens_gene_id.map(pair.S)
    g["B1b_pair"] = g.hsapiens_gene_id.map(pair.B1b_raw)
    sd = pd.read_parquet(CM.CACHE / "Sdir__liv_4sp.parquet").set_index("var_name")
    g["Sdir_Mouse"] = sd.Cspdir_Mouse.reindex(g.var_name).to_numpy()
    pe = pd.read_parquet(BD / "liv_4sp__pair_E.parquet").set_index("var_name")
    g["E_HMouse"] = pe.E_Human_Mouse.reindex(g.var_name).to_numpy()
    for lab in labels_for("liv_4sp"):
        gg = g.copy()
        S = species_margin(lab).reindex(gg.var_name)
        gg["S_Mouse"] = S.Mouse.to_numpy()
        pm = pd.Series(np.abs(C.load_state(run_table()[lab][1])["S_g_ema"]) >= RU.MASK_EPS,
                       index=np.load(FWD / f"{lab}.npz")["var_names"])
        rws = (gg.var_name.map(pm).eq(True) & gg.nonzero_all_species.eq(True) & ~pair.masked.reindex(gg.hsapiens_gene_id).eq(True).to_numpy()).to_numpy()
        lbl = dict(check="pairfit", run_label=lab, seed=seed_of(lab))
        units.append(dict(g=gg, gold="G_H3", covs=XR.C_PAT, rows=rws, scores=["S_Mouse", "S_pair", "Sdir_Mouse", "E_HMouse"],
                          ref="S_Mouse", label=dict(lbl, given="none")))
        units.append(dict(g=gg, gold="G_H3", covs=XR.C_PAT, rows=rws, scores=["S_Mouse"],
                          extra_spline=["S_pair", "Sdir_Mouse", "E_HMouse"], label=dict(lbl, given="S_pair+Sdir_Mouse+E_HMouse")))
        units.append(dict(g=gg, gold="G_H3", covs=XR.C_PAT, rows=rws, scores=["S_pair"],
                          extra_spline=["S_Mouse", "Sdir_Mouse", "E_HMouse"], label=dict(lbl, given="S_Mouse+Sdir_Mouse+E_HMouse")))
    write(CM.run_units(units), "posthoc_pairfit.csv")


SECTIONS = dict(cellvar=sec_cellvar, adds=sec_adds, own=sec_own, checks=sec_checks, pairfit=sec_pairfit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    for s in ap.parse_args().sections.split(","):
        print("==", s, flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            SECTIONS[s]()


if __name__ == "__main__":
    main()
