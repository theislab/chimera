"""X7 (EXPLORATORY): tests of rules.py. Sections: types (A-i), cells (descriptive), genetype (A-ii, drivers),
species (Lead B), mixing. Writes results/revision/X7/*.csv.

Usage: python -m revision.tasks.X7_architecture_outputs.analyses [--sections types,cells,genetype,species,mixing]
"""

from __future__ import annotations

import argparse
import itertools
import math
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import cohorts as C
from revision.common import controls as K
from revision.tasks.X4_where_differs import common as CM
from revision.tasks.X4_where_differs import rules as XR
from revision.tasks.X7_architecture_outputs import rules as RU
from revision.tasks.X7_architecture_outputs.baselines import BD, coarse
from revision.tasks.X7_architecture_outputs.forward import FWD, OUT, run_table
from revision.tasks.X7_architecture_outputs.golds import GD

CM.N_JOBS = 12
METRICS = ["tv", "gain", "lshare"]
LIVER_GOLD_SPECIES = ["Human", "Mouse"]


def write(df, name):
    K.write_csv(df, OUT / name)
    print(f"  wrote {name} ({len(df)} rows)", flush=True)


def labels_for(c):
    return [lab for lab, (cc, _) in run_table().items() if cc == c]


def seed_of(lab):
    return int(lab.rsplit("_s", 1)[1]) if "_s" in lab and lab.rsplit("_s", 1)[1].isdigit() else 42


def perm_p(x, y):
    """Exact one-sided permutation p of Spearman rho >= observed (all orderings; n <= 10)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    obs = spearmanr(x, y)[0]
    n = len(x)
    if n > 8:  # 10! is large; Monte Carlo with a fixed seed
        rng = np.random.default_rng(RU.SEED)
        null = np.array([spearmanr(x, rng.permutation(y))[0] for _ in range(20000)])
        return float(obs), float((1 + np.sum(null >= obs - 1e-12)) / (len(null) + 1))
    null = np.array([spearmanr(x, np.array(p))[0] for p in itertools.permutations(y)])
    return float(obs), float(np.mean(null >= obs - 1e-12))


# ---------------------------------------------------------------- golds per type
def gold_types(c, fine=False):
    """Series type -> independent divergence (div, div_adj)."""
    if c.startswith("liv"):
        h = pd.read_csv(GD / "h3_type_divergence.csv").set_index("type")
        inv = {v: k for k, v in RU.LIVER_H3_MAP.items()}
        h = h.loc[[t for t in h.index if t in inv]]
        h.index = [inv[t] for t in h.index]
        return h["div"], None
    if fine:
        m = pd.read_csv(GD / "murat_fine_type_divergence.csv")
    else:
        m = pd.read_csv(GD / "murat_type_divergence.csv")
        tr = RU.TESTIS_TIER[c]
        if tr != "all":
            m = m[m.tier == tr]
    g = m.groupby("type")
    return g["div"].mean(), g["div_adj"].mean()


def type_values(t: pd.DataFrame, col: str, key: str, species: list[str]) -> pd.Series:
    """Unweighted mean over species of the species mean; types present in every species with >= MIN_CELLS."""
    t = t[t.species.isin(species)]
    m = t.groupby([key, "species"])[col].agg(["mean", "size"])
    m = m[m["size"] >= RU.MIN_CELLS]["mean"].unstack()
    m = m.dropna()
    return m.mean(1) if set(species) <= set(m.columns) else pd.Series(dtype=float)


def direct_type(c, species, level="coarse"):
    d = pd.read_csv(BD / "type_divergence_direct.csv")
    d = d[(d.cohort == c) & (d.level == level) & d.s1.isin(species) & d.s2.isin(species)]
    return d.groupby("type")["div"].mean()


def sec_types():
    rows, vals = [], []
    tests = [(c, False) for c in RU.GOLD_COHORTS_A] + [("test_hmac", True), ("test_3sp", True)]
    for c, fine in tests:
        if c.startswith("liv"):
            species = LIVER_GOLD_SPECIES
        elif fine:
            species = ["Human", "Macaque"]
        else:
            species = None
        g, gadj = gold_types(c, fine)
        for lab in labels_for(c):
            t = pd.read_parquet(BD / f"{lab}__cells.parquet")
            sp_ = species or sorted(t.species.unique())
            key = "cell_type" if fine else "coarse"
            cols = METRICS + ["npur_chimera"] + (["npur_scvi"] if "npur_scvi" in t else [])
            tv = {k: type_values(t, k, key, sp_) for k in cols}
            tv["direct_div"] = direct_type(c, sp_, "fine" if fine else "coarse")
            tv["median_lib"] = type_values(t, "lib", key, sp_)
            types = [k for k in g.index if all(k in v.index for v in tv.values())]
            for k in types:
                vals.append(dict(cohort=c, fine=fine, run_label=lab, seed=seed_of(lab), type=k, gold_div=g[k],
                                 gold_div_adj=gadj[k] if gadj is not None else np.nan,
                                 **{m: float(tv[m][k]) for m in tv}))
            for m in tv:
                for gname, gg in (("div", g), ("div_adj", gadj)):
                    if gg is None:
                        continue
                    r, p = perm_p([tv[m][k] for k in types], [gg[k] for k in types])
                    rows.append(dict(cohort=c, fine=fine, run_label=lab, seed=seed_of(lab), score=m, gold=gname,
                                     n_types=len(types), rho=r, p_perm_one_sided=p, types=" ".join(types)))
    write(pd.DataFrame(vals), "types_values.csv")
    write(pd.DataFrame(rows), "types_rho.csv")


# ---------------------------------------------------------------- per-cell descriptive
def sec_cells():
    rows = []
    for lab, (c, _) in run_table().items():
        t = pd.read_parquet(BD / f"{lab}__cells.parquet")
        for (s, k), u in t.groupby(["species", "coarse"]):
            if len(u) < RU.MIN_CELLS:
                continue
            r = dict(cohort=c, run_label=lab, seed=seed_of(lab), species=s, type=k, n=len(u))
            for m in METRICS + ["npur_chimera", "npur_scvi", "lib"]:
                if m in u:
                    r[f"{m}_mean"] = float(u[m].mean())
                    r[f"{m}_median"] = float(u[m].median())
            for m in ("npur_chimera", "npur_scvi", "lib"):
                if m in u:
                    r[f"rho_tv_{m}"] = float(spearmanr(u["tv"], u[m])[0])
            rows.append(r)
    write(pd.DataFrame(rows), "cells_by_type.csv")
    # over all cells, per run: does the residual share restate purity or depth?
    rows = []
    for lab, (c, _) in run_table().items():
        t = pd.read_parquet(BD / f"{lab}__cells.parquet")
        r = dict(cohort=c, run_label=lab, seed=seed_of(lab), n=len(t))
        for a, b in (("tv", "gain"), ("tv", "lshare"), ("tv", "npur_chimera"), ("tv", "npur_scvi"), ("tv", "lib"),
                     ("npur_chimera", "npur_scvi")):
            if a in t and b in t:
                r[f"rho_{a}_{b}"] = float(spearmanr(t[a], t[b])[0])
        for s, u in t.groupby("species"):
            r[f"tv_mean_{s}"] = float(u.tv.mean())
            r[f"npur_chimera_{s}"] = float(u.npur_chimera.mean())
            if "npur_scvi" in u:
                r[f"npur_scvi_{s}"] = float(u.npur_scvi.mean())
        rows.append(r)
    write(pd.DataFrame(rows), "cells_summary.csv")


# ---------------------------------------------------------------- gene x type
def residual_by_type(lab, c, species):
    """r̄_gsk on the coarse map; returns dict type -> (G x species) array and var_names."""
    z = np.load(FWD / f"{lab}.npz", allow_pickle=False)
    grp = z["groups"]
    spn = z["species"][grp["species_idx"]]
    ck = coarse(c, np.array(grp["label"]))
    out = {}
    for k in sorted(set(ck)):
        cols = []
        for s in species:
            m = (spn == s) & (ck == k)
            n = grp["n"][m].sum()
            if n < RU.MIN_CELLS:
                break
            cols.append(z["sum_r"][m].sum(0) / n)
        else:
            out[k] = np.column_stack(cols)
    return out, z["var_names"]


def gold_gene_type(c):
    if c.startswith("liv"):
        g = pd.read_parquet(GD / "h3_gene_type.parquet")
        return {k: g[f"I_{v}"] for k, v in RU.LIVER_H3_MAP.items()}
    g = pd.read_parquet(GD / "murat_gene_type.parquet")
    tr = RU.TESTIS_TIER[c]
    return {k: g[f"I_{tr}_{k}"] for k in RU.TESTIS_TYPES}


def sec_genetype():
    units, drv = [], []
    for c in RU.GENE_TYPE_COHORTS:
        species = (LIVER_GOLD_SPECIES if c.startswith("liv")
                   else sorted(pd.read_parquet(BD / f"{c}__cells.parquet").species.unique()))
        base = CM.load(c)[["var_name", "hsapiens_gene_id", "block", "symbol", "nameable", "margin_support", "masked"]]
        gd = pd.read_parquet(BD / f"{c}__gene_type_direct.parquet").set_index("var_name")
        gold = gold_gene_type(c)
        for lab in labels_for(c):
            rb, vn = residual_by_type(lab, c, species)
            for k, I in gold.items():
                if k not in rb or f"x_{species[0]}_{k}" not in gd:
                    continue
                g = base.copy()
                R = rb[k]
                g["R"] = pd.Series(R.std(1), index=vn).reindex(g.var_name).to_numpy()
                Xs = np.column_stack([gd[f"x_{s}_{k}"].reindex(g.var_name).to_numpy() for s in species])
                Fs = np.column_stack([gd[f"f_{s}_{k}"].reindex(g.var_name).to_numpy() for s in species])
                ok = (Fs >= RU.DET_MIN).all(1)
                off = np.array([np.median(Xs[ok, j] - Xs[ok].mean(1)) for j in range(len(species))])
                g["Ddir"] = np.where(ok, (Xs - off[None, :]).std(1), np.nan)
                g["lev"] = Xs.mean(1)
                g["det"] = Fs.mean(1)
                g["I"] = g.hsapiens_gene_id.map(I).to_numpy()
                units.append(dict(g=g, gold="I", covs=["lev", "det"], scores=["R", "Ddir"], ref="R",
                                  label=dict(cohort=c, run_label=lab, seed=seed_of(lab), type=k)))
                if lab == c:
                    T = g[g.R.notna() & g.I.notna() & g.Ddir.notna()].copy()
                    T["R_rank"] = T.R.rank(pct=True)
                    T["I_rank"] = T.I.rank(pct=True)
                    T["Ddir_rank"] = T.Ddir.rank(pct=True)
                    top = T[T.R_rank >= 0.9]
                    drv.append(dict(cohort=c, type=k, n=len(T), n_top=len(top),
                                    frac_top_R_in_top_I=float((top.I_rank >= 0.9).mean()),
                                    frac_top_Ddir_in_top_I=float((T[T.Ddir_rank >= 0.9].I_rank >= 0.9).mean()),
                                    rho_R_Ddir=float(spearmanr(T.R, T.Ddir)[0])))
                    T["cohort"], T["type"] = c, k
                    T[T.nameable].sort_values("R", ascending=False).head(25)[
                        ["cohort", "type", "symbol", "hsapiens_gene_id", "R", "Ddir", "I", "R_rank", "I_rank", "Ddir_rank"]
                    ].to_csv(OUT / f"drivers__{c}__{k}.csv", index=False)
    write(CM.run_units(units), "genetype_rho.csv")
    write(pd.DataFrame(drv), "genetype_top_decile.csv")


# ---------------------------------------------------------------- Lead B
def species_margin(lab):
    z = np.load(FWD / f"{lab}.npz", allow_pickle=False)
    S = z["sp_l"] / z["n_species"][:, None]
    return pd.DataFrame(S.T, index=pd.Index(z["var_names"], name="var_name"), columns=[str(s) for s in z["species"]])


def type_balanced(c):
    from revision.tasks.T2_5_celltype_margin import margin as MG  # noqa: F401

    key = "__cell_ontology_class" if c.startswith("test") else ""
    s = pd.read_parquet(C.gene_ids.REPO_ROOT / f"results/revision/T2_5/{c}{key}__S_gsk.parquet")
    p = s.pivot_table(index=["var_name", "cell_type"], columns="species", values="S_gsk_meanz")
    p = p.dropna()
    shared = p.reset_index().groupby("cell_type").size()
    return p.groupby("var_name").mean(), sorted(shared.index)


def sec_species():
    units = []
    for c in ("test_3sp", "liv_4sp"):
        g0 = CM.load(c)
        sd = pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")
        pe = pd.read_parquet(BD / f"{c}__pair_E.parquet").set_index("var_name")
        tb, tb_types = type_balanced(c)
        if c == "test_3sp":
            mt = pd.read_parquet(GD / "murat_tiers.parquet")
            g0["Gt"] = g0.hsapiens_gene_id.map(mt.Gt)
            gold, covs = "Gt", XR.C_PAT + ["H4_nsp"]
            g0["E_contrast"] = (pe["E_Human_Macaque"] - (pe["E_Human_Mouse"] + pe["E_Macaque_Mouse"]) / 2).reindex(g0.var_name).to_numpy()
            g0["E_HMac"] = pe["E_Human_Macaque"].reindex(g0.var_name).to_numpy()
            g0["dir_contrast"] = ((sd.Cspdir_Human + sd.Cspdir_Macaque) / 2 - sd.Cspdir_Mouse).reindex(g0.var_name).to_numpy()
            g0["tb_contrast"] = ((tb.Human + tb.Macaque) / 2 - tb.Mouse).reindex(g0.var_name).to_numpy()
            contrasts = {"Bt": lambda S: (S.Human + S.Macaque) / 2 - S.Mouse}
            base_scores = ["E_contrast", "dir_contrast", "E_HMac"]
        else:
            gold, covs = "G_H3", XR.C_PAT
            g0["E_contrast"] = (pe["E_Human_Mouse"] - pe["E_Human_Monkey"]).reindex(g0.var_name).to_numpy()
            g0["E_contrast_zeb"] = (pe["E_Human_Mouse"] - pe["E_Human_Zebrafish"]).reindex(g0.var_name).to_numpy()
            g0["E_HMouse"] = pe["E_Human_Mouse"].reindex(g0.var_name).to_numpy()
            g0["dir_contrast"] = (sd.Cspdir_Mouse - sd.Cspdir_Monkey).reindex(g0.var_name).to_numpy()
            g0["dir_contrast_zeb"] = (sd.Cspdir_Mouse - sd.Cspdir_Zebrafish).reindex(g0.var_name).to_numpy()
            g0["tb_contrast"] = (tb.Mouse - tb.Monkey).reindex(g0.var_name).to_numpy()
            contrasts = {"Bl": lambda S: S.Mouse - S.Monkey, "Bz": lambda S: S.Mouse - S.Zebrafish}
            base_scores = ["E_contrast", "dir_contrast", "E_HMouse"]
        for lab in labels_for(c):
            g = g0.copy()
            S = species_margin(lab).reindex(g.var_name)
            for s in S.columns:
                g[f"S_{s}"] = S[s].to_numpy()
            for nm, f in contrasts.items():
                g[nm] = f(S).to_numpy()
            Sema = C.load_state(run_table()[lab][1])["S_g_ema"]
            vn = np.load(FWD / f"{lab}.npz")["var_names"]
            g["unmasked"] = g.var_name.map(pd.Series(np.abs(Sema) >= RU.MASK_EPS, index=vn)).to_numpy()
            sp_cols = [f"S_{s}" for s in S.columns]
            for uni, rows in (("unmasked_counts_all", (g.unmasked & g.nonzero_all_species.eq(True)).to_numpy()),
                              ("all", np.ones(len(g), bool))):
                lbl = dict(cohort=c, run_label=lab, seed=seed_of(lab), universe=uni, tb_types=" ".join(tb_types))
                main_ = list(contrasts)[0]
                extra = ["tb_contrast"] if lab == c else []
                units.append(dict(g=g, gold=gold, covs=covs, scores=list(contrasts) + base_scores + extra,
                                  rows=rows, ref=main_, label=dict(lbl, set="contrasts")))
                units.append(dict(g=g, gold=gold, covs=covs, scores=sp_cols, rows=rows, label=dict(lbl, set="species")))
                if c == "liv_4sp":
                    units.append(dict(g=g, gold=gold, covs=covs, scores=["Bz", "E_contrast_zeb", "dir_contrast_zeb"],
                                      rows=rows, ref="Bz", label=dict(lbl, set="zebrafish")))
    write(CM.run_units(units), "species_rho.csv")


def sec_mixing():
    rows = []
    for lab, (c, _) in run_table().items():
        t = pd.read_parquet(BD / f"{lab}__cells.parquet")
        for s, u in t.groupby("species"):
            rows.append(dict(cohort=c, run_label=lab, seed=seed_of(lab), species=s, n=len(u),
                             npur_chimera=float(u.npur_chimera.mean()),
                             npur_scvi=float(u.npur_scvi.mean()) if "npur_scvi" in u else np.nan,
                             tv=float(u.tv.mean())))
    write(pd.DataFrame(rows), "mixing_by_species.csv")


SECTIONS = dict(types=sec_types, cells=sec_cells, genetype=sec_genetype, species=sec_species, mixing=sec_mixing)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sections", default=",".join(SECTIONS))
    a = ap.parse_args()
    for s in a.sections.split(","):
        print("==", s, flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            SECTIONS[s]()


if __name__ == "__main__":
    main()
