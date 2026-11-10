"""Liver versus testis comparison of delta-alpha, genes matched on the human Ensembl gene ID.

Primary version: raw prior (alpha - pi_raw), centered on the cohort median, same species
pair (H+Monkey vs H+Macaque, H+Mouse vs H+Mouse). Reported beside it, in one table: the
clamped prior the runs anchor on, expression-trend removal, and 4-sp vs 3-sp.

Per pair and version: gene-level Spearman, program-level Spearman (narrow categories of
``notebooks/_gene_categories.py`` with >= 15 joined genes, ``other`` excluded), the fraction
of liver-negative programs (and genes) positive in testis, and the flip rate expected from
the marginals under independence. Genes are joined on ``hsapiens_gene_id`` only.

Usage::

    python analysis/tasks/T2_3_tissue/run.py [--outdir DIR] [--prerun] [--force]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args

sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
from _gene_categories import classify_extended  # noqa: E402

MIN_PROGRAM_GENES = 15
PRIMARY = "raw_centred"

# Version -> (column, how the zero point is set). "median": subtract the cohort median
# before joining (D of the pre-specified analysis plan, section 2). "program_mean": signs relative to the mean of program
# means (the reference "cohort-centred" row). "none": as trained.
VERSIONS = {
    "raw_centred": ("delta_alpha_raw", "median"),
    "raw": ("delta_alpha_raw", "none"),
    "clamped": ("delta_alpha", "none"),
    "clamped_centred": ("delta_alpha", "median"),
    "clamped_centred_programs": ("delta_alpha", "program_mean"),
    "clamped_expr_resid": ("da_expr_resid", "none"),
    "raw_expr_resid": ("da_raw_expr_resid", "none"),
}

# Reference values of an independent computation of the same table, for comparison
# (2 decimals, n exact).
REFERENCE = {
    ("n_genes", "clamped"): (8366, 8233, 8879),
    ("gene_rho", "clamped"): (0.01, -0.16, -0.21),
    ("program_rho", "clamped"): (0.05, -0.21, 0.07),
    ("program_rho", "clamped_expr_resid"): (0.38, 0.40, 0.52),
    ("frac_livneg_programs_pos_testis", "clamped"): (0.91, 0.93, 0.89),
    ("frac_livneg_programs_pos_testis", "raw"): (0.00, 0.00, 0.04),
    ("frac_livneg_programs_pos_testis", "clamped_centred_programs"): (0.68, 0.71, 0.43),
    ("frac_livneg_programs_pos_testis", "clamped_expr_resid"): (0.23, 0.11, 0.25),
}
# The sign-quadrant panel compares the per-gene mean over the four liver cohorts with the mean
# over the three testis cohorts (panel e of the residue-audit figure of the Supplement).
FIG7E = ("liver mean (4 cohorts) vs testis mean (3 cohorts)",
         ["liv_hmonkey", "liv_hmouse", "liv_hzeb", "liv_4sp"], ["test_hmac", "test_hmouse", "test_3sp"])
# Reference values of the gene-level sign statistics and of the sign-quadrant panel, for comparison.
CLAIMS_REFERENCE = {
    ("frac_livneg_genes_pos_testis", "clamped", "H+Mouse vs H+Mouse"): 0.76,
    ("frac_livneg_genes_pos_testis", "raw", "H+Mouse vs H+Mouse"): 0.19,
    ("flip_rate_observed", "clamped", FIG7E[0]): 0.575,
    ("flip_rate_expected", "clamped", FIG7E[0]): 0.572,
    ("gene_rho", "clamped", FIG7E[0]): -0.147,
    ("flip_rate_observed", "raw", FIG7E[0]): 0.392,
}
MEAN_COLUMNS = ["delta_alpha", "delta_alpha_raw", "da_expr_resid", "da_raw_expr_resid",
                "delta_alpha__median", "delta_alpha_raw__median", "S_g_ema", "correction",
                "log_mean", "prior", "prior_raw"]


def prepare(cohort: str) -> pd.DataFrame:
    g = C.load_cohort(cohort).genes
    g["da_expr_resid"] = C.expression_residual(g["delta_alpha"], g["log_mean"])
    g["da_raw_expr_resid"] = C.expression_residual(g["delta_alpha_raw"], g["log_mean"])
    for col in ("delta_alpha", "delta_alpha_raw"):
        g[col + "__median"] = g[col] - g[col].median()
    g["category"] = [classify_extended(s) if isinstance(s, str) else "other" for s in g["symbol"]]
    return g


def join(a: pd.DataFrame, b: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    aa = a.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    bb = b.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
    common = aa.index.intersection(bb.index)
    return aa.loc[common], bb.loc[common]


def mean_frame(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Per-gene mean over cohorts of one tissue, indexed by ``hsapiens_gene_id``."""
    ix = [f.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id")
          .set_index("hsapiens_gene_id") for f in frames]
    out = pd.DataFrame({c: pd.concat([f[c] for f in ix], axis=1).mean(1) for c in MEAN_COLUMNS})
    sym = ix[0]["symbol"]
    for f in ix[1:]:
        sym = sym.combine_first(f["symbol"])
    out["symbol"] = sym.reindex(out.index)
    out["category"] = [classify_extended(s) if isinstance(s, str) else "other" for s in out["symbol"]]
    return out.reset_index()


def programs(cat: np.ndarray, x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    d = pd.DataFrame({"category": cat, "liver": x, "testis": y})
    size = d.groupby("category").size()
    keep = [c for c in size.index if size[c] >= MIN_PROGRAM_GENES and c != "other"]
    m = d[d.category.isin(keep)].groupby("category")[["liver", "testis"]].mean()
    m["n_genes"] = size.loc[m.index]
    return m


def flip_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Sign-class change between tissues, observed and expected from the marginals."""
    lneg, tneg = x < 0, y < 0
    lpos, tpos = x > 0, y > 0
    obs = float(((lneg & tpos) | (lpos & tneg)).mean())
    exp = float(lneg.mean() * tpos.mean() + lpos.mean() * tneg.mean())
    return dict(flip_rate_observed=obs, flip_rate_expected=exp,
                frac_liver_to_testis_neg_pos=float((lneg & tpos).mean()),
                frac_liver_to_testis_pos_neg=float((lpos & tneg).mean()))


def compare(A: pd.DataFrame, B: pd.DataFrame, label: str, la: str, lb: str):
    rows, prog_rows = [], []
    for version, (col, zero) in VERSIONS.items():
        use = col + "__median" if zero == "median" else col
        x, y = A[use].to_numpy(float), B[use].to_numpy(float)
        P = programs(A["category"].to_numpy(), x, y)
        px, py = P["liver"].to_numpy(), P["testis"].to_numpy()
        if zero == "program_mean":
            px, py = px - px.mean(), py - py.mean()
        neg = px < 0
        gneg = x < 0
        r = dict(pair=label, liver=la, testis=lb, version=version, primary=version == PRIMARY,
                 n_genes=len(A), gene_rho=float(spearmanr(x, y)[0]),
                 n_programs=len(P), program_rho=float(spearmanr(px, py)[0]),
                 n_livneg_programs=int(neg.sum()),
                 frac_livneg_programs_pos_testis=float((py[neg] > 0).mean()) if neg.any() else np.nan,
                 frac_testis_programs_pos=float((py > 0).mean()),
                 program_flip_rate_observed=float(((px < 0) & (py > 0) | (px > 0) & (py < 0)).mean()),
                 program_flip_rate_expected=float((px < 0).mean() * (py > 0).mean()
                                                  + (px > 0).mean() * (py < 0).mean()),
                 frac_livneg_genes_pos_testis=float((y[gneg] > 0).mean()) if gneg.any() else np.nan,
                 frac_testis_genes_pos=float((y > 0).mean()))
        r.update(flip_stats(x, y))
        rows.append(r)
        P = P.assign(pair=label, version=version, liver_c=px, testis_c=py).reset_index()
        prog_rows.append(P)
    extra = dict(pair=label, n_genes=len(A),
                 gene_rho_S_g_ema=float(spearmanr(A["S_g_ema"], B["S_g_ema"])[0]),
                 gene_rho_correction=float(spearmanr(A["correction"], B["correction"])[0]),
                 gene_rho_log_mean=float(spearmanr(A["log_mean"], B["log_mean"])[0]),
                 gene_rho_prior_raw=float(spearmanr(A["prior_raw"], B["prior_raw"])[0]),
                 gene_rho_prior_clamped=float(spearmanr(A["prior"], B["prior"])[0]),
                 frac_equal_prior_clamped=float((np.abs(A["prior"] - B["prior"]) < 1e-6).mean()),
                 frac_equal_prior_raw=float((np.abs(A["prior_raw"] - B["prior_raw"]) < 1e-6).mean()),
                 symbol_agreement=float((A["symbol"].fillna("") == B["symbol"].fillna("")).mean()))
    return rows, pd.concat(prog_rows, ignore_index=True), extra


def mtdna(A: pd.DataFrame, B: pd.DataFrame, label: str) -> list[dict]:
    m = (A["category"] == "mitochondrial_dna_encoded").to_numpy()
    out = []
    for version, (col, zero) in VERSIONS.items():
        if zero == "program_mean":
            continue
        use = col + "__median" if zero == "median" else col
        out.append(dict(pair=label, version=version, n=int(m.sum()),
                        liver_mean=float(A.loc[m, use].mean()), testis_mean=float(B.loc[m, use].mean()),
                        liver_frac_neg=float((A.loc[m, use] < 0).mean()),
                        testis_frac_pos=float((B.loc[m, use] > 0).mean())))
    return out


def reproduction(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pairs = [p[2] for p in C.TISSUE_PAIRS]
    for (metric, version), ref in REFERENCE.items():
        for pair, rv in zip(pairs, ref):
            v = float(table.loc[(table.pair == pair) & (table.version == version), metric].iloc[0])
            tol = 0 if metric == "n_genes" else 0.005 + 1e-9
            rows.append(dict(source="reference table, 2 decimals", metric=metric, version=version, pair=pair,
                             reference=rv, value=v, abs_diff=abs(v - rv), within_rounding=abs(v - rv) <= tol))
    for (metric, version, pair), rv in CLAIMS_REFERENCE.items():
        v = float(table.loc[(table.pair == pair) & (table.version == version), metric].iloc[0])
        tol = 0.0005 + 1e-9 if round(rv * 1000) % 10 else 0.005 + 1e-9
        rows.append(dict(source="claim ledger", metric=metric, version=version, pair=pair, reference=rv,
                         value=v, abs_diff=abs(v - rv), within_rounding=abs(v - rv) <= tol))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    add_common_args(ap)
    args = ap.parse_args()
    out = TaskOutput("T2_3", args.outdir, prerun=args.prerun, force=args.force)

    data = {}
    loc_rows = []
    for c in C.COHORT_ORDER:
        print("loading", c, flush=True)
        g = data[c] = prepare(c)
        for col in ("delta_alpha", "delta_alpha_raw"):
            v = g[col]
            loc_rows.append(dict(cohort=c, quantity=col, n=len(v), mean=v.mean(), median=v.median(),
                                 sd=v.std(), mean_over_sd=v.mean() / v.std(),
                                 frac_pos=(v > 0).mean(), frac_neg=(v < 0).mean()))
        loc_rows[-1]["frac_prior_at_clamp"] = float((g["prior"] >= g["prior"].max() - 1e-6).mean())
        loc_rows[-1]["prior_clamp_value"] = float(g["prior"].max())
        loc_rows[-1]["frac_prior_raw_at_0.95"] = float((g["prior_raw"] >= 0.95 - 1e-6).mean())

    rows, progs, extras, mt = [], [], [], []
    for la, lb, label in C.TISSUE_PAIRS:
        A, B = join(data[la], data[lb])
        r, p, e = compare(A, B, label, la, lb)
        rows += r
        progs.append(p)
        extras.append(e)
        mt += mtdna(A, B, label)
    label, livs, tess = FIG7E
    A, B = join(mean_frame([data[c] for c in livs]), mean_frame([data[c] for c in tess]))
    r, p, e = compare(A, B, label, "+".join(livs), "+".join(tess))
    rows += r
    progs.append(p)
    extras.append(e)
    mt += mtdna(A, B, label)
    table = pd.DataFrame(rows)
    rep = reproduction(table)
    out.csv(table, "tissue_comparison.csv")
    out.csv(pd.concat(progs, ignore_index=True), "program_means.csv")
    out.csv(pd.DataFrame(extras), "pair_identity_checks.csv")
    out.csv(pd.DataFrame(loc_rows), "cohort_location.csv")
    out.csv(pd.DataFrame(mt), "mtdna_by_version.csv")
    out.csv(rep, "reproduction_check.csv")
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(table.drop(columns=["liver", "testis"]).round(3).to_string())
    print(rep.round(4).to_string())
    out.finish(dict(reproduction_all_within_rounding=bool(rep.within_rounding.all()),
                    primary_version=PRIMARY, min_program_genes=MIN_PROGRAM_GENES))


if __name__ == "__main__":
    main()
