"""T2.6: category and landmark claims under expression and length matching; mtDNA check.

Sets, per cohort, with true gene identity (``gene_ids``):
* every narrow category of ``notebooks/_gene_categories.py`` and every broad umbrella of
  ``notebooks/_gene_categories_broad.py`` with >= 5 genes (Fig 4a, b; Fig 6d uses the same
  vocabulary), ``other`` excluded;
* the Fig 4c landmark themes (``notebooks/_alpha_biology_simpler_builder.py`` H_THEMES):
  urea cycle / gluconeogenesis, mtDNA OXPHOS, liver supporting tissue, housekeeping controls;
* the urea-cycle enzyme list and the mtDNA / nuclear OXPHOS categories used in the
  earlier read-only analysis;
* the Fig 6 GIN archetypes A0-A2 (canonical 4-sp run, ``tables/gene_archetypes.csv``), 4-sp only.

For each set: mean delta-alpha raw (uncentred), centred on the cohort median, Cohen-style
d = mean / SD of all genes (uncentred, centred on the cohort median as in the
pre-specified readout D, and centred on the cohort mean), against the clamped prior the run anchors on and against the
raw prior; then the same set against expression-matched and expression- and CDS-length-
matched background genes, with empirical two-sided p. Random gene sets check that the
centred and matched statistics give d ~ 0 in every cohort.

mtDNA: liver mtDNA-encoded genes against non-mt genes with the same cross-species detection
imbalance (range of per-species detection fractions in hepatocytes) and expression level.

Usage::

    python analysis/tasks/T2_6_matched/run.py [--ensembl-dir DIR] [--n-null 2000] [--outdir DIR] [--prerun]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from analysis.common import cds as CDS
from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
from _gene_categories import classify_extended  # noqa: E402
from _gene_categories_broad import classify_broad  # noqa: E402

MIN_SET = 5
NN = 40
RANDOM_SIZES = (10, 30, 100)
N_RANDOM = 2000
N_CALIB = 200

# Fig 4c themes, verbatim from notebooks/_alpha_biology_simpler_builder.py (H_THEMES).
FIG4C_THEMES = {
    "Urea cycle / GNG": ["CPS1", "ARG1", "ASL", "OTC", "PCK1", "PAH"],
    "mtDNA OXPHOS": ["MT-CO1", "MT-CO2", "MT-CO3", "MT-CYB", "MT-ND1", "MT-ND2", "MT-ND4", "MT-ATP6"],
    "Liver support": ["SELP", "SELE", "VCAM1", "F3", "CR2", "CD40LG", "COL17A1", "PDGFRB", "KIT", "C1QC"],
    "Housekeeping": ["DDX4", "RPL3", "EEF1A1", "EEF2", "ACTB", "TUBA1A", "COX4I1", "NDUFA1", "HNF4A"],
}
UREA_CYCLE = ["CPS1", "OTC", "ASS1", "ASL", "ARG1", "NAGS", "ARG2", "SLC25A15", "SLC25A13", "SLC25A2"]
ARCHETYPE_RUN = C.CANONICAL["liv_4sp"]

# Earlier read-only analysis, liver rows (testis rows there used wrong labels):
# (cohort, set) -> (n, da_mean, matched_null_mean, matched_null_sd, emp_p, da_expr_resid)
REFERENCE = {
    ("liv_hmonkey", "mtDNA"): (9, -0.1827, -0.0278, 0.0604, 0.0100, -0.1413),
    ("liv_hmonkey", "OXPHOS_nuclear"): (63, -0.0278, -0.0147, 0.0098, 0.1800, -0.0143),
    ("liv_hmonkey", "urea_cycle"): (7, -0.2827, -0.0300, 0.0951, 0.0115, -0.2318),
    ("liv_hmouse", "mtDNA"): (9, -0.2355, -0.0874, 0.0300, 0.0000, -0.1472),
    ("liv_hmouse", "OXPHOS_nuclear"): (64, -0.0349, -0.0259, 0.0057, 0.1135, -0.0095),
    ("liv_hmouse", "urea_cycle"): (7, -0.0795, -0.0919, 0.0372, 0.7345, -0.0013),
    ("liv_hzeb", "mtDNA"): (9, 0.0403, 0.0755, 0.0123, 0.0050, -0.0336),
    ("liv_hzeb", "OXPHOS_nuclear"): (65, 0.0029, 0.0075, 0.0017, 0.0090, -0.0045),
    ("liv_hzeb", "urea_cycle"): (6, 0.0583, 0.0439, 0.0120, 0.2330, 0.0169),
    ("liv_4sp", "mtDNA"): (9, -0.4230, -0.2299, 0.0231, 0.0000, -0.2028),
    ("liv_4sp", "OXPHOS_nuclear"): (66, -0.0691, -0.0382, 0.0065, 0.0000, -0.0297),
    ("liv_4sp", "urea_cycle"): (7, -0.2975, -0.2453, 0.0270, 0.0515, -0.0701),
}
# Same source, true-label testis mtDNA (11 genes): ranges over the three testis cohorts.
REFERENCE_TESTIS_MT = {"delta_alpha": (0.10, 0.12), "delta_alpha_raw": (0.09, 0.12),
                       "da_expr_resid": (0.04, 0.06)}
# 4-sp MT-CO1 detection in hepatocytes: Human, Monkey, Mouse, Zebrafish
REFERENCE_MTCO1 = {"Human": 0.98, "Monkey": 0.25, "Mouse": 0.69, "Zebrafish": 1.00}


def window_null(values: np.ndarray, x: np.ndarray, set_idx: np.ndarray, pool: np.ndarray, B: int,
                rng, nn: int = NN) -> np.ndarray:
    """1-D matching: each set gene draws from the ``nn`` pool genes adjacent in ``x``."""
    order = pool[np.argsort(x[pool], kind="stable")]
    xs = x[order]
    picks = np.empty((len(set_idx), B), dtype=int)
    for i, g in enumerate(set_idx):
        p = np.searchsorted(xs, x[g])
        lo = max(0, p - nn // 2)
        hi = min(len(xs), lo + nn)
        picks[i] = order[rng.integers(lo, hi, size=B)]
    return values[picks].mean(0)


def knn_null(values: np.ndarray, Z: np.ndarray, set_idx: np.ndarray, pool: np.ndarray, B: int,
             rng, nn: int = NN) -> np.ndarray:
    """k-D matching on standardised rank coordinates ``Z``: nearest ``nn`` pool genes."""
    d = ((Z[set_idx][:, None, :] - Z[pool][None, :, :]) ** 2).sum(-1)
    nb = np.argpartition(d, min(nn, len(pool) - 1), axis=1)[:, :nn]
    nbrs = pool[nb]
    pick = rng.integers(0, nn, size=(len(set_idx), B))
    return values[np.take_along_axis(nbrs, pick, axis=1)].mean(0)


def null_summary(obs: float, null: np.ndarray, sd_all: float, prefix: str) -> dict:
    mu = null.mean()
    p = (1 + (np.abs(null - mu) >= abs(obs - mu)).sum()) / (len(null) + 1)
    return {f"{prefix}_null_mean": mu, f"{prefix}_null_sd": null.std(), f"{prefix}_excess": obs - mu,
            f"{prefix}_d": (obs - mu) / sd_all, f"{prefix}_z": (obs - mu) / null.std() if null.std() > 0 else np.nan,
            f"{prefix}_p": p}


def rank01(x: pd.Series) -> np.ndarray:
    return (x.rank(method="average") / x.notna().sum()).to_numpy()


def define_sets(g: pd.DataFrame, cohort: str) -> dict[str, tuple[str, np.ndarray]]:
    sym = g["symbol"]
    sets: dict[str, tuple[str, np.ndarray]] = {}
    narrow = pd.Series([classify_extended(s) if isinstance(s, str) else "other" for s in sym], index=g.index)
    broad = pd.Series([classify_broad(s) if isinstance(s, str) else "other" for s in sym], index=g.index)
    for vocab, lab, src in (("narrow", narrow, "_gene_categories.classify_extended"),
                            ("broad", broad, "_gene_categories_broad.classify_broad")):
        for cat, n in lab.value_counts().items():
            if cat != "other" and n >= MIN_SET:
                sets[f"{vocab}:{cat}"] = (src, (lab == cat).to_numpy())
    for theme, genes in FIG4C_THEMES.items():
        sets[f"fig4c:{theme}"] = ("Fig 4c theme (H_THEMES)", sym.isin(genes).to_numpy())
    sets["landmark:urea_cycle"] = ("urea-cycle enzymes", sym.isin(UREA_CYCLE).to_numpy())
    sets["landmark:mtDNA"] = ("narrow:mitochondrial_dna_encoded", (narrow == "mitochondrial_dna_encoded").to_numpy())
    sets["landmark:OXPHOS_nuclear"] = ("narrow:oxphos_nuclear_encoded", (narrow == "oxphos_nuclear_encoded").to_numpy())
    if cohort == "liv_4sp":
        arch = pd.read_csv(C.RUNS_DIR / ARCHETYPE_RUN / "tables" / "gene_archetypes.csv").set_index("ortho_id")
        a = arch["archetype"].reindex(g.index)
        for k in sorted(a.dropna().unique()):
            sets[f"fig6:archetype_A{int(k)}"] = ("tables/gene_archetypes.csv", (a == k).to_numpy())
    return {k: v for k, v in sets.items() if v[1].sum() >= 3}


def set_rows(g: pd.DataFrame, cohort: str, sets: dict, B: int, seed: int) -> list[dict]:
    rows = []
    x_expr = g["log_mean"].to_numpy()
    has_len = g["log_cds_length"].notna().to_numpy()
    Z = np.c_[rank01(g["log_mean"]), rank01(g["log_cds_length"])]
    for j, (name, (src, mask)) in enumerate(sets.items()):
        rng = np.random.default_rng(seed + j)
        idx = np.where(mask)[0]
        pool = np.where(~mask)[0]
        r = dict(cohort=cohort, set=name, source=src, n=len(idx),
                 symbols=";".join(sorted(map(str, g["symbol"].iloc[idx].dropna())))[:2000],
                 log_mean_pct=float(g["log_mean"].rank(pct=True).iloc[idx].mean()),
                 S_g_ema_mean=float(g["S_g_ema"].iloc[idx].mean()),
                 frac_masked=float((np.abs(g["S_g_ema"].iloc[idx]) < 1e-6).mean()),
                 da_expr_resid_mean=float(g["da_expr_resid"].iloc[idx].mean()))
        for col, tag in (("delta_alpha", "clamped"), ("delta_alpha_raw", "raw")):
            v = g[col].to_numpy()
            sd, med = v.std(ddof=1), np.median(v)
            obs = v[idx].mean()
            r[f"{tag}_mean"] = obs
            r[f"{tag}_d_uncentred"] = obs / sd
            r[f"{tag}_centred_mean"] = obs - med
            r[f"{tag}_d_centred"] = (obs - med) / sd
            r[f"{tag}_d_centred_on_mean"] = (obs - v.mean()) / sd
            r.update(null_summary(obs, window_null(v, x_expr, idx, pool, B, rng), sd, f"{tag}_exprmatched"))
            il, pl = idx[has_len[idx]], pool[has_len[pool]]
            r["n_with_cds"] = len(il)
            if len(il) >= 3:
                r.update(null_summary(v[il].mean(), knn_null(v, Z, il, pl, B, rng), sd,
                                      f"{tag}_exprlenmatched"))
        rows.append(r)
    return rows


def random_rows(g: pd.DataFrame, cohort: str, B: int, seed: int) -> list[dict]:
    """Random sets: uncentred d inherits the cohort offset; centred and matched d should be ~0."""
    rng = np.random.default_rng(seed)
    rows = []
    x = g["log_mean"].to_numpy()
    G = len(g)
    for col, tag in (("delta_alpha", "clamped"), ("delta_alpha_raw", "raw")):
        v = g[col].to_numpy()
        sd, med = v.std(ddof=1), np.median(v)
        for n in RANDOM_SIZES:
            means = np.array([v[rng.choice(G, n, replace=False)].mean() for _ in range(N_RANDOM)])
            excess, pvals = [], []
            for _ in range(N_CALIB):
                idx = rng.choice(G, n, replace=False)
                pool = np.setdiff1d(np.arange(G), idx)
                null = window_null(v, x, idx, pool, 500, rng)
                s = null_summary(v[idx].mean(), null, sd, "m")
                excess.append(s["m_d"])
                pvals.append(s["m_p"])
            rows.append(dict(cohort=cohort, delta=tag, set_size=n, n_random=N_RANDOM,
                             mean_d_uncentred=float((means / sd).mean()),
                             mean_d_centred_median=float(((means - med) / sd).mean()),
                             mean_d_centred_on_mean=float(((means - v.mean()) / sd).mean()),
                             sd_d_centred=float(((means - med) / sd).std()),
                             frac_uncentred_negative=float((means < 0).mean()),
                             frac_centred_negative=float((means - med < 0).mean()),
                             n_calibration=N_CALIB, mean_d_exprmatched=float(np.mean(excess)),
                             frac_exprmatched_p_lt_0_05=float(np.mean(np.array(pvals) < 0.05))))
    return rows


def mtdna_check(co: C.Cohort, g: pd.DataFrame, B: int, seed: int):
    """mtDNA vs non-mt genes with the same cross-species detection imbalance in hepatocytes."""
    import scipy.sparse as sp

    a = co.adata
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    spl = a.obs["species"].astype(str).to_numpy()
    hep = a.obs["cell_type"].astype(str).to_numpy() == "Hepatocytes"
    species = sorted(set(spl))
    det = {}
    for s in species:
        m = hep & (spl == s)
        det[s] = np.asarray((X[m] > 0).mean(0)).ravel() if m.sum() else np.full(a.n_vars, np.nan)
    D = np.stack([det[s] for s in species])
    imb = np.nanmax(D, 0) - np.nanmin(D, 0)
    g = g.assign(hep_det_range=imb, hep_det_mean=np.nanmean(D, 0))
    for s in species:
        g[f"hep_det_{s}"] = det[s]
    mt = (g["symbol"].astype(str).str.startswith("MT-")).to_numpy()
    idx, pool = np.where(mt)[0], np.where(~mt)[0]
    rows = []
    pct = a.obs["pct_counts_mt"].to_numpy() if "pct_counts_mt" in a.obs else None
    for col, tag in (("delta_alpha", "clamped"), ("delta_alpha_raw", "raw")):
        v = g[col].to_numpy()
        sd = v.std(ddof=1)
        obs = v[idx].mean()
        rng = np.random.default_rng(seed)
        designs = {
            "detection_range": np.c_[rank01(g["hep_det_range"])],
            "detection_range+log_mean": np.c_[rank01(g["hep_det_range"]), rank01(g["log_mean"])],
            "detection_range+hep_detection_mean": np.c_[rank01(g["hep_det_range"]), rank01(g["hep_det_mean"])],
            "log_mean": np.c_[rank01(g["log_mean"])],
        }
        for dname, Z in designs.items():
            null = knn_null(v, Z, idx, pool, B, rng)
            r = dict(cohort=co.name, delta=tag, matched_on=dname, n_mt=len(idx), mt_mean=obs,
                     mt_det_range_mean=float(imb[idx].mean()),
                     background_det_range_median=float(np.median(imb[pool])))
            r.update(null_summary(obs, null, sd, "matched"))
            rows.append(r)
    per_gene = g.loc[mt, ["symbol", "delta_alpha", "delta_alpha_raw", "S_g_ema", "log_mean", "hep_det_range",
                          "hep_det_mean"] + [f"hep_det_{s}" for s in species]].assign(cohort=co.name)
    sp_rows = []
    for s in species:
        m = hep & (spl == s)
        sp_rows.append(dict(cohort=co.name, species=s, n_hepatocytes=int(m.sum()),
                            median_pct_counts_mt_hep=float(np.median(pct[m])) if pct is not None and m.sum() else np.nan,
                            samples=";".join(sorted(a.obs.loc[m, "sample"].astype(str).unique()))
                            if "sample" in a.obs else ""))
    return rows, per_gene.reset_index(), sp_rows


def reproduction(cat: pd.DataFrame, mt_genes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    name = {"mtDNA": "landmark:mtDNA", "OXPHOS_nuclear": "landmark:OXPHOS_nuclear", "urea_cycle": "landmark:urea_cycle"}
    fields = ["n", "clamped_mean", "clamped_exprmatched_null_mean", "clamped_exprmatched_null_sd",
              "clamped_exprmatched_p", "da_expr_resid_mean"]
    for (cohort, s), ref in REFERENCE.items():
        r = cat[(cat.cohort == cohort) & (cat.set == name[s])].iloc[0]
        for f, rv in zip(fields, ref):
            mc = "null" in f or f.endswith("_p")
            tol = 0 if f == "n" else (0.02 if f.endswith("_p") else (0.006 if mc else 0.0006))
            v = float(r[f])
            rows.append(dict(check=f"{cohort} {s} {f}", reference=rv, value=v, abs_diff=abs(v - rv), tolerance=tol,
                             monte_carlo=mc, within_tolerance=abs(v - rv) <= tol + 1e-9))
    for col, (lo, hi) in REFERENCE_TESTIS_MT.items():
        for cohort in ("test_hmac", "test_hmouse", "test_3sp"):
            r = cat[(cat.cohort == cohort) & (cat.set == "landmark:mtDNA")].iloc[0]
            key = {"delta_alpha": "clamped_mean", "delta_alpha_raw": "raw_mean", "da_expr_resid": "da_expr_resid_mean"}[col]
            v = float(r[key])
            rows.append(dict(check=f"{cohort} mtDNA (n={int(r.n)}) {col} in [{lo}, {hi}]", reference=np.nan, value=v,
                             abs_diff=np.nan, tolerance=0.005, monte_carlo=False,
                             within_tolerance=lo - 0.005 <= v <= hi + 0.005))
    m = mt_genes[(mt_genes.cohort == "liv_4sp") & (mt_genes.symbol == "MT-CO1")]
    for s, rv in REFERENCE_MTCO1.items():
        v = float(m[f"hep_det_{s}"].iloc[0]) if len(m) else np.nan
        rows.append(dict(check=f"liv_4sp MT-CO1 hepatocyte detection {s}", reference=rv, value=v,
                         abs_diff=abs(v - rv), tolerance=0.005, monte_carlo=False, within_tolerance=abs(v - rv) <= 0.005))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ensembl-dir", default=str(CDS.ENSEMBL_DIR))
    ap.add_argument("--n-null", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    add_common_args(ap)
    args = ap.parse_args()
    out = TaskOutput("T2_6", args.outdir, prerun=args.prerun, force=args.force)

    cds = CDS.human_cds_table(args.ensembl_dir)
    cat_rows, rand_rows, mt_rows, mt_genes, sp_rows, defs = [], [], [], [], [], []
    for i, cohort in enumerate(C.COHORT_ORDER):
        print("==", cohort, flush=True)
        liver = not C.is_testis(cohort)
        co = C.load_cohort(cohort, keep_adata=liver)
        g = co.genes.join(cds[["log_cds_length"]], on="hsapiens_gene_id")
        g["da_expr_resid"] = C.expression_residual(g["delta_alpha"], g["log_mean"])
        sets = define_sets(g, cohort)
        for k, (src, m) in sets.items():
            defs.append(dict(cohort=cohort, set=k, source=src, n=int(m.sum()),
                             genes=";".join(sorted(map(str, g["symbol"][m].dropna())))))
        cat_rows += set_rows(g, cohort, sets, args.n_null, args.seed + 1000 * i)
        rand_rows += random_rows(g, cohort, args.n_null, args.seed + 1000 * i + 999)
        if liver:
            r, pg, sr = mtdna_check(co, g, args.n_null, args.seed + 1000 * i + 500)
            mt_rows += r
            mt_genes.append(pg)
            sp_rows += sr
        co.adata = None
    cat = pd.DataFrame(cat_rows)
    mt_genes = pd.concat(mt_genes, ignore_index=True)
    rep = reproduction(cat, mt_genes)
    out.csv(cat, "category_matched.csv")
    out.csv(pd.DataFrame(defs), "set_definitions.csv")
    out.csv(pd.DataFrame(rand_rows), "random_sets.csv")
    out.csv(pd.DataFrame(mt_rows), "mtdna_detection_matched.csv")
    out.csv(mt_genes, "mtdna_genes_detection.csv")
    out.csv(pd.DataFrame(sp_rows), "liver_hepatocyte_species.csv")
    out.csv(rep, "reproduction_check.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 500)
    key = cat[cat.set.str.startswith(("landmark:", "fig4c:", "fig6:"))]
    print(key[["cohort", "set", "n", "clamped_mean", "clamped_d_centred", "clamped_exprmatched_excess",
               "clamped_exprmatched_p", "clamped_exprlenmatched_excess", "clamped_exprlenmatched_p",
               "raw_d_centred", "raw_exprlenmatched_p"]].round(3).to_string())
    print(pd.DataFrame(rand_rows).round(3).to_string())
    print(pd.DataFrame(mt_rows).round(3).to_string())
    print(rep.round(4).to_string())
    out.finish(dict(n_null=args.n_null, matching_neighbours=NN, min_set=MIN_SET,
                    reproduction_within_tolerance=int(rep.within_tolerance.sum()), reproduction_rows=len(rep)))


if __name__ == "__main__":
    main()
