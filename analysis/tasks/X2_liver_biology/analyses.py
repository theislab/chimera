"""EXPLORATORY: liver biology leads on the per-gene tables (``build.py``) and forward sums.

Every lead is written whatever its outcome. Chimera readouts: consensus D (T3_3_consensus), S_ema and
its exact species x type decomposition (T2_5_celltype_margin; unweighted per-cell means S_gsk, within-type null
W_gsk, species contributions Cs, type contributions Ck). Baselines beside each: B1b / B1b_raw /
B1c (T4_2_gene_baselines) and per-type pseudobulk log2 fold change to human (``lfc__``).

Outputs in results/analysis/X2/:
* ``presence.csv``: genes per cohort with zero counts / absent from the raw object, per species,
  and how they sit in the D tails.
* ``drivers.csv``: matched excess of the figure sets (landmark themes of the prior landscape panel, urea cycle, mtDNA, nuclear
  OXPHOS, archetypes of the gate network, zonation sets) for every readout and baseline; matchings C and C_hep;
  universes all genes and complete genes (counts in every species).
* ``drivers_seeds.csv``: set means of the species contributions per seed (4-sp, 5 seeds;
  H+Mouse, 3 seeds), from the forward sums.
* ``set_species_share.csv``: share of the set's summed margin carried by each species and type.
* ``named_genes.csv``: robustness profile of every gene named on the figures, with a class
  (confident / caveated / do-not-highlight) by the rule in ``classify``.
* ``mtdna.csv``, ``mtdna_species.csv``: per MT gene and per species, D, margin, detection, mt capture.
* ``divergent_conserved.csv``: the high-prior, D-bottom-5 %, margin-supported, sign-stable genes.
* ``calibration.csv``: random hepatocyte-gene sets through the same matched test.

The plan's reserved sets (H5/H6) are never formed or tested. No gold data is read.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import gene_ids

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
import sets as ST  # noqa: E402
import stats as SX  # noqa: E402
from _gene_categories import classify_extended  # noqa: E402

OUT = SX.OUT
FW = OUT / "forward"
LIVER = ["liv_4sp", "liv_hmonkey", "liv_hmouse", "liv_hzeb"]
PAIRS = ["liv_hmonkey", "liv_hmouse", "liv_hzeb"]
SPECIES = ["Human", "Monkey", "Mouse", "Zebrafish"]
SEED_LABELS = {"liv_4sp": ["liv_4sp", "liv_4sp_s43", "liv_4sp_s44", "liv_4sp_s45", "liv_4sp_s46"],
               "liv_hmouse": ["liv_hmouse", "liv_hmouse_s43", "liv_hmouse_s44"]}
B = 2000


def categories(g: pd.DataFrame) -> pd.Series:
    return pd.Series([classify_extended(s) if isinstance(s, str) else "other" for s in g["symbol"]], index=g.index)


def figure_sets(g: pd.DataFrame, cohort: str) -> dict:
    sym = g["symbol"]
    cat = categories(g)
    sets = {f"fig4c:{k}": sym.isin(v).to_numpy() for k, v in ST.FIG4C_THEMES.items()}
    sets["urea_cycle"] = sym.isin(ST.UREA_CYCLE).to_numpy()
    sets["mtDNA"] = sym.astype(str).str.startswith("MT-").to_numpy()
    sets["OXPHOS_nuclear"] = (cat == "oxphos_nuclear_encoded").to_numpy()
    if cohort == "liv_4sp":
        arch = pd.read_csv(C.RUNS_DIR / C.CANONICAL["liv_4sp"] / "tables" / "gene_archetypes.csv").set_index("ortho_id")
        a = arch["archetype"].reindex(g["var_name"]).to_numpy()
        for k in (0, 1, 2):
            sets[f"fig6:A{k}"] = a == k
    ref = pd.read_csv(OUT / "zonation" / "gene_zonation_reference.csv").dropna(subset=["hsapiens_gene_id"])
    ref = ref.drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    e = g["hsapiens_gene_id"]
    for k in ("PP_canonical", "PC_canonical"):
        sets[f"zonation:lit_{k}"] = sym.isin(ST.ZONATION_SETS[k]).to_numpy()
    for k in ("zPP_Mouse", "zPC_Mouse", "zPP_Human", "zPC_Human"):
        sets[f"zonation:{k}"] = e.map(ref[k]).fillna(False).to_numpy(bool)
    return sets


def driver_scores(g: pd.DataFrame) -> list[str]:
    sc = ["D", "S_ema_cons", "B1b_raw", "B1b", "B1c"]
    sc += [c for c in g.columns if c.startswith("Cs__")]
    sc += [c for c in g.columns if c.startswith("Ck__") and c.split("__")[1] in ("Hepatocytes", "Endothelial")]
    for pre in ("S__", "W__"):
        sc += [c for c in g.columns if c.startswith(pre) and c.endswith(("__Hepatocytes", "__Endothelial"))]
    sc += [c for c in g.columns if c.startswith("lfc__") and c.endswith("__Hepatocytes")]
    return sc


def presence_table() -> pd.DataFrame:
    rows = []
    for c in LIVER:
        g = SX.load_genes(c)
        q05, q95 = g.D.quantile([0.05, 0.95])
        for s in SPECIES:
            if f"zero__{s}" not in g:
                continue
            z, a = g[f"zero__{s}"], g[f"absent__{s}"]
            rows.append(dict(cohort=c, species=s, n_genes=len(g), zero=int(z.sum()), absent_from_raw=int(a.sum()),
                             frac_zero=z.mean(), frac_zero_in_D_bottom5=z[g.D <= q05].mean(),
                             frac_zero_in_D_top5=z[g.D >= q95].mean(), median_D_zero=g.D[z].median(),
                             median_D_nonzero=g.D[~z].median(),
                             frac_S_masked_zero=float((g.S_g_ema.abs() < 1e-6)[z].mean()),
                             frac_margin_support_zero_in_tail=float(g.margin_support[z & g["tail"]].mean())))
        rows.append(dict(cohort=c, species="any", n_genes=len(g), zero=int((~g.complete).sum()),
                         frac_zero=(~g.complete).mean(), frac_zero_in_D_bottom5=(~g.complete)[g.D <= q05].mean(),
                         frac_zero_in_D_top5=(~g.complete)[g.D >= q95].mean(),
                         median_D_zero=g.D[~g.complete].median(), median_D_nonzero=g.D[g.complete].median(),
                         frac_S_masked_zero=float((g.S_g_ema.abs() < 1e-6)[~g.complete].mean())))
    return pd.DataFrame(rows)


def drivers() -> pd.DataFrame:
    rows = []
    for c in LIVER:
        g = SX.load_genes(c)
        sets = figure_sets(g, c)
        sc = driver_scores(g)
        for uni, keep in (("all", np.ones(len(g), bool)), ("complete", g["complete"].to_numpy())):
            gg = g[keep].reset_index(drop=True)
            ss = {k: v[keep] for k, v in sets.items()}
            for r in SX.set_excess(gg, c, ss, sc, B=B, seed=7):
                r["universe"] = uni
                rows.append(r)
    return pd.DataFrame(rows)


def seed_species(label: str, cohort: str) -> pd.DataFrame:
    """Cs_gs = sum of the per-cell ratio over species-s cells / (n_s K); sums over species to S_g."""
    z = np.load(FW / f"{label}.npz")
    grp = pd.DataFrame([x.split("|") for x in z["groups"]], columns=["species", "cell_type", "zone"])
    spp = sorted(grp.species.unique())
    out = {}
    for s in spp:
        m = (grp.species == s).to_numpy()
        out[f"Cs__{s}"] = z["sum_l"][m].sum(0) / (z["n"][m].sum() * len(spp))
        h = m & (grp.cell_type == "Hepatocytes").to_numpy()
        out[f"S__{s}__Hepatocytes"] = z["sum_l"][h].sum(0) / z["n"][h].sum()
        out[f"W__{s}__Hepatocytes"] = z["sum_wt"][h].sum(0) / z["n"][h].sum()
    return pd.DataFrame(out, index=pd.Index(z["var_names"], name="var_name"))


def drivers_seeds() -> pd.DataFrame:
    rows = []
    for c, labs in SEED_LABELS.items():
        g = SX.load_genes(c)
        sets = figure_sets(g, c)
        for lab in labs:
            d = seed_species(lab, c).reindex(g["var_name"])
            for name, mask in sets.items():
                for col in d.columns:
                    v = d[col].to_numpy()
                    rows.append(dict(cohort=c, label=lab, set=name, score=col, n=int(mask.sum()),
                                     set_mean=float(np.nanmean(v[mask])) if mask.any() else np.nan,
                                     all_mean=float(np.nanmean(v)), all_sd=float(np.nanstd(v)),
                                     d_vs_all=(float(np.nanmean(v[mask])) - float(np.nanmean(v))) / float(np.nanstd(v))
                                     if mask.any() else np.nan))
    return pd.DataFrame(rows)


def species_share() -> pd.DataFrame:
    rows = []
    for c in LIVER:
        g = SX.load_genes(c)
        sets = figure_sets(g, c)
        cs = [x for x in g.columns if x.startswith("Cs__")]
        ck = [x for x in g.columns if x.startswith("Ck__")]
        for name, mask in sets.items():
            if mask.sum() == 0:
                continue
            sub = g[mask]
            tot = sub["S_full_meanz"].sum()
            r = dict(cohort=c, set=name, n=int(mask.sum()), sum_S=tot)
            for x in cs + ck:
                r[f"share_{x}"] = sub[x].sum() / tot if tot != 0 else np.nan
            rows.append(r)
    return pd.DataFrame(rows)


def dominant(g: pd.DataFrame) -> pd.DataFrame:
    cs = [x for x in g.columns if x.startswith("Cs__")]
    A = g[cs].abs()
    share = A.max(1) / A.sum(1).replace(0, np.nan)
    sp_ = A.idxmax(1).str.replace("Cs__", "")
    # dominant (species, type) cell: w_sk * S_gsk with the T2_5_celltype_margin weights
    w = pd.read_csv(OUT / "genes" / "type_weights.csv")
    w = w[w.cohort == g["cohort"].iloc[0]]
    cells = {}
    for _, r in w.iterrows():
        col = f"S__{r.species}__{r.cell_type}"
        if col in g:
            cells[f"{r.species}|{r.cell_type}"] = g[col] * r.weight
    Cm = pd.DataFrame(cells)
    cell = Cm.abs().idxmax(1)
    cell_share = Cm.abs().max(1) / Cm.abs().sum(1).replace(0, np.nan)
    sign = np.sign(pd.Series([Cm.loc[i, cell[i]] if isinstance(cell[i], str) else np.nan for i in Cm.index],
                             index=Cm.index))
    return pd.DataFrame(dict(dom_species=sp_, dom_species_share=share, dom_cell=cell, dom_cell_share=cell_share,
                             dom_cell_sign=sign))


def named_list() -> list[str]:
    out = []
    for v in ST.FIG4C_THEMES.values():
        out += v
    out += ST.UREA_CYCLE + ST.EXTRA_NAMED
    for k in ("PP_canonical", "PC_canonical", "PP_named", "PC_named"):
        out += ST.ZONATION_SETS[k]
    return list(dict.fromkeys(out))


def theme_of(sym: str) -> str:
    for k, v in ST.FIG4C_THEMES.items():
        if sym in v:
            return f"Fig4c:{k}"
    if sym.startswith("MT-"):
        return "mtDNA (not on Fig 4c)"
    if sym in ST.UREA_CYCLE:
        return "urea cycle (T2.6 list)"
    for k in ("PP_canonical", "PC_canonical", "PP_named", "PC_named"):
        if sym in ST.ZONATION_SETS[k]:
            return f"zonation:{k[:2]}"
    return "other named"


def classify(r: pd.Series) -> tuple[str, str]:
    """Fixed rule (written before the table was read).

    confident: 4-sp sign consistency 1.0, margin support, D outside the 10-90 % band of its 100
    matched genes, counts in every species, and the same sign of D in at least 2 of the 3
    two-species cohorts where the gene exists.
    caveated: sign consistency 1.0 and matched-extreme, but one or more of the other checks fails.
    do-not-highlight: otherwise (including genes masked from the margin, S_ema = 0).
    """
    reasons = []
    if not r["sign_consistency"] == 1.0:
        reasons.append("sign unstable across seeds")
    if not (r["D_matched_pct"] <= 0.10 or r["D_matched_pct"] >= 0.90):
        reasons.append("not extreme vs matched genes")
    core_ok = not reasons
    if abs(r["S_ema_cons"]) < 1e-6:
        reasons.append("margin masked (S=0)")
    elif not r["margin_support"]:
        reasons.append("no margin support")
    if not r["complete"]:
        reasons.append("zero counts in " + r["zero_species"])
    if r["n_pairs_with_gene"] > 0 and r["n_pairs_same_sign"] < min(2, r["n_pairs_with_gene"]):
        reasons.append(f"same sign in {int(r['n_pairs_same_sign'])}/{int(r['n_pairs_with_gene'])} pair cohorts")
    if not core_ok or "margin masked (S=0)" in reasons:
        return "do-not-highlight", "; ".join(reasons)
    if reasons:
        return "caveated", "; ".join(reasons)
    return "confident", ""


def named_genes() -> pd.DataFrame:
    g4 = SX.load_genes("liv_4sp")
    g4 = g4.join(dominant(g4))
    others = {c: SX.load_genes(c).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id") for c in PAIRS}
    rows = []
    names = named_list() + sorted(s for s in g4.symbol.dropna() if s.startswith("MT-"))
    for sym in dict.fromkeys(names):
        m = g4.symbol == sym
        if not m.any():
            rows.append(dict(symbol=sym, theme=theme_of(sym), in_4sp_vocab=False))
            continue
        r0 = g4[m].iloc[0]
        zero_sp = [s for s in SPECIES if r0.get(f"zero__{s}", False)]
        r = dict(symbol=sym, theme=theme_of(sym), in_4sp_vocab=True, hsapiens_gene_id=r0.hsapiens_gene_id,
                 prior_raw=r0.prior_raw, D=r0.D, sd_delta_alpha=r0.sd_delta_alpha,
                 sign_consistency=r0.sign_consistency, S_ema_cons=r0.S_ema_cons, margin_support=r0.margin_support,
                 D_matched_pct=r0.D_matched_pct, B1b_raw=r0.B1b_raw, B1b=r0.B1b, B1c=r0.B1c,
                 dom_species=r0.dom_species, dom_species_share=r0.dom_species_share, dom_cell=r0.dom_cell,
                 dom_cell_share=r0.dom_cell_share, dom_cell_sign=r0.dom_cell_sign,
                 complete=bool(r0.complete), zero_species=",".join(zero_sp),
                 absent_species=",".join(s for s in SPECIES if r0.get(f"absent__{s}", False)),
                 log_cp10k_spmean=r0.log_cp10k_spmean)
        for s in SPECIES:
            r[f"Cs_{s}"] = r0.get(f"Cs__{s}")
            r[f"x_hep_{s}"] = r0.get(f"x__{s}__Hepatocytes")
        npair, nsame = 0, 0
        for c, o in others.items():
            if r0.hsapiens_gene_id in o.index:
                d = o.loc[r0.hsapiens_gene_id]
                r[f"D_{c}"] = d.D
                r[f"sc_{c}"] = d.sign_consistency
                r[f"support_{c}"] = d.margin_support
                npair += 1
                nsame += int(np.sign(d.D) == np.sign(r0.D))
            else:
                r[f"D_{c}"] = np.nan
        r["n_pairs_with_gene"], r["n_pairs_same_sign"] = npair, nsame
        r["class"], r["reasons"] = classify(pd.Series(r))
        rows.append(r)
    return pd.DataFrame(rows)


def mtdna() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, srows = [], []
    for c in LIVER:
        g = SX.load_genes(c)
        a = C.read_adata(c, backed=True)
        obs = a.obs
        for s in sorted(obs.species.astype(str).unique()):
            h = (obs.species.astype(str) == s) & (obs.cell_type.astype(str) == "Hepatocytes")
            mt = g[g.symbol.astype(str).str.startswith("MT-")]
            nonmt = g[~g.symbol.astype(str).str.startswith("MT-")]
            srows.append(dict(cohort=c, species=s, median_pct_counts_mt_hep=float(obs.loc[h, "pct_counts_mt"].median()),
                              mean_detection_mt_hep=float(mt[f"f__{s}__Hepatocytes"].mean()),
                              mean_S_hep_mt=float(mt[f"S__{s}__Hepatocytes"].mean()),
                              mean_S_hep_nonmt=float(nonmt[f"S__{s}__Hepatocytes"].mean()),
                              mean_Cs_mt=float(mt[f"Cs__{s}"].mean()), mean_Cs_nonmt=float(nonmt[f"Cs__{s}"].mean()),
                              mean_x_hep_mt=float(mt[f"x__{s}__Hepatocytes"].mean())))
        for _, r0 in g[g.symbol.astype(str).str.startswith("MT-")].iterrows():
            r = dict(cohort=c, symbol=r0.symbol, prior_raw=r0.prior_raw, D=r0.D, sign_consistency=r0.sign_consistency,
                     S_ema_cons=r0.S_ema_cons, margin_support=r0.margin_support, D_matched_pct=r0.D_matched_pct,
                     B1b_raw=r0.B1b_raw)
            for s in SPECIES:
                if f"Cs__{s}" in g:
                    r[f"Cs_{s}"] = r0[f"Cs__{s}"]
                    r[f"S_hep_{s}"] = r0[f"S__{s}__Hepatocytes"]
                    r[f"det_hep_{s}"] = r0[f"f__{s}__Hepatocytes"]
            rows.append(r)
    return pd.DataFrame(rows), pd.DataFrame(srows)


def divergent_conserved() -> pd.DataFrame:
    out = []
    for c in LIVER:
        g = SX.load_genes(c)
        g = g.join(dominant(g))
        cat = categories(g)
        hp = g.prior_raw >= g.prior_raw.quantile(ST.DIVCONS_PRIOR_Q)
        low = g.D <= g.D.quantile(ST.DIVCONS_D_Q)
        stable = (g.sign_consistency == 1.0) if g.sign_consistency.notna().any() else pd.Series(True, index=g.index)
        sel = hp & low & g.margin_support & stable
        sub = g[sel].copy()
        sub["category"] = cat[sel]
        sub["n_hp"], sub["n_low"], sub["n_hp_low"] = int(hp.sum()), int(low.sum()), int((hp & low).sum())
        sub["n_hp_low_support"] = int((hp & low & g.margin_support).sum())
        keep = ["cohort", "symbol", "hsapiens_gene_id", "category", "prior_raw", "D", "sign_consistency", "S_ema_cons",
                "D_matched_pct", "B1b_raw", "B1b", "complete", "dom_species", "dom_species_share", "dom_cell",
                "dom_cell_share", "dom_cell_sign", "log_cp10k_spmean", "n_hp", "n_low", "n_hp_low", "n_hp_low_support"]
        keep += [x for x in g.columns if x.startswith(("Cs__", "zero__"))]
        keep += [x for x in g.columns if x.startswith("lfc__") and x.endswith("Hepatocytes")]
        out.append(sub[[k for k in keep if k in sub.columns]])
    df = pd.concat(out, ignore_index=True)
    # cross-cohort recurrence by Ensembl ID
    rec = df.groupby("hsapiens_gene_id")["cohort"].apply(lambda s: ",".join(sorted(s)))
    df["cohorts_selected"] = df["hsapiens_gene_id"].map(rec)
    allD = {c: SX.load_genes(c).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")["D"] for c in LIVER}
    for c in LIVER:
        df[f"D_{c}"] = df["hsapiens_gene_id"].map(allD[c])
    return df


def calibration(n_sets: int = 200, size: int = 40) -> pd.DataFrame:
    rows = []
    for c in LIVER:
        g = SX.load_genes(c)
        rng = np.random.default_rng(3)
        pools = {"hep_top_quartile": np.where(g.hep_enrich >= g.hep_enrich.quantile(0.75))[0],
                 "any_gene": np.arange(len(g))}
        for pname, pool in pools.items():
            sets = {}
            for i in range(n_sets):
                m = np.zeros(len(g), bool)
                m[rng.choice(pool, size, replace=False)] = True
                sets[f"{pname}_{i}"] = m
            res = pd.DataFrame(SX.set_excess(g, c, sets, ["D"], B=500, seed=99))
            for mt, d in res.groupby("matching"):
                rows.append(dict(cohort=c, pool=pname, matching=mt, n_sets=n_sets, size=size,
                                 mean_d=d.excess_d.mean(), sd_d=d.excess_d.std(), frac_p_lt_0_05=(d.p < 0.05).mean()))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    presence_table().to_csv(OUT / "presence.csv", index=False)
    print("presence", flush=True)
    named_genes().to_csv(OUT / "named_genes.csv", index=False)
    print("named", flush=True)
    m, ms = mtdna()
    m.to_csv(OUT / "mtdna.csv", index=False)
    ms.to_csv(OUT / "mtdna_species.csv", index=False)
    divergent_conserved().to_csv(OUT / "divergent_conserved.csv", index=False)
    species_share().to_csv(OUT / "set_species_share.csv", index=False)
    drivers_seeds().to_csv(OUT / "drivers_seeds.csv", index=False)
    print("seeds", flush=True)
    drivers().to_csv(OUT / "drivers.csv", index=False)
    print("drivers", flush=True)
    calibration().to_csv(OUT / "calibration.csv", index=False)
    print("calibration", flush=True)


if __name__ == "__main__":
    main()
