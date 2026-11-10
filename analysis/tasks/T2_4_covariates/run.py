"""T2.4: seven-cohort covariate audit of delta-alpha.

1. Spearman of delta-alpha with log mean, detection, log variance, cell-type specificity tau,
   prior pi (clamped and raw), S_g_ema, human CDS length and GC (Ensembl 110), and the
   partial rank correlation of delta-alpha with S_g_ema given expression (log mean,
   detection, log variance).
2. 5-fold cross-validated R^2 of delta-alpha on the GIN inputs: expression scalars (3),
   ESM-2 PCs (64), ESM + expression, all 72; linear ridge only.
3. S_g masking: genes with |S_g_ema| < 1e-6 and what their delta-alpha does.
4. Closed-form ridge projection c ~ kappa P_f S_g_ema, against free and within-expression-bin
   permutations of S_g_ema.

Outputs ``covariate_table.csv``, ``cv_r2.csv``, ``projection.csv``, ``masking.csv``,
``cds_features.csv``, ``reproduction_check.csv`` and the figure replacing
``fig:supp-alpha-decorrelation``.

Usage::

    python analysis/tasks/T2_4_covariates/run.py [--ensembl-dir DIR] [--n-perm 1000] [--outdir DIR] [--prerun]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict

from analysis.common import cds as CDS
from analysis.common import cohorts as C
from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args

MASK_EPS = 1e-6
TAIL = 0.05
N_EXPR_BINS = 20
FEATURE_GROUPS = {
    "expr3": C.GIN_COLUMNS["expr3"],
    "esm64": C.GIN_COLUMNS["esm64"],
    "esm64+expr3": C.GIN_COLUMNS["esm64"] + C.GIN_COLUMNS["expr3"],
    "all72": list(range(72)),
}
COVARIATES = ["log_mean", "detection", "log_var", "tau", "prior", "prior_raw", "S_g_ema",
              "log_cds_length", "gc", "gc3", "log_cp10k_spmean", "detection_spmean"]
COV_LABEL = {"log_mean": "log mean", "detection": "detection", "log_var": "log variance",
             "tau": "cell-type specificity τ", "prior": "prior π", "prior_raw": "raw prior π",
             "S_g_ema": "fit-quality margin S", "log_cds_length": "CDS length", "gc": "CDS GC",
             "partial_S_given_expr": "partial ρ(Δα, S | expression)"}

# Reference values from the earlier read-only analysis (response plan, section 2.5 table),
# two decimals, cohorts in COHORT_ORDER. Its per-cohort CSVs can be passed with
# --reference-tables for a full-precision comparison.
REFERENCE = {
    ("spearman", "log_mean"): (-0.04, -0.35, 0.39, -0.63, 0.73, 0.73, 0.67),
    ("spearman", "detection"): (-0.05, -0.35, 0.35, -0.64, 0.72, 0.71, 0.66),
    ("spearman", "log_var"): (-0.03, -0.34, 0.42, -0.60, 0.74, 0.74, 0.68),
    ("spearman", "tau"): (0.08, 0.31, -0.05, 0.55, -0.27, -0.34, -0.35),
    ("spearman", "prior"): (-0.00, 0.19, -0.05, 0.02, -0.16, -0.16, -0.18),
    ("spearman", "S_g_ema"): (0.41, 0.28, 0.32, 0.48, 0.50, 0.48, 0.35),
    ("partial", "partial_S_given_expr"): (0.39, 0.17, 0.12, 0.11, 0.15, 0.09, 0.14),
    ("cv_r2", "expr3"): (0.03, 0.24, 0.61, 0.62, 0.68, 0.70, 0.54),
    ("cv_r2", "esm64"): (0.01, 0.19, 0.35, 0.31, 0.28, 0.23, 0.42),
    ("cv_r2", "esm64+expr3"): (0.03, 0.47, 0.82, 0.89, 0.89, 0.85, 0.88),
    ("cv_r2", "all72"): (0.03, 0.48, 0.85, 0.91, 0.92, 0.88, 0.93),
    ("cv_r2_rank", "all72"): (0.02, 0.43, 0.79, 0.84, 0.81, 0.77, 0.84),
    ("cv_r2", "prior_only"): (0.00, 0.03, 0.01, 0.00, 0.05, 0.06, 0.06),
    ("cv_r2", "S_only"): (0.37, 0.19, 0.15, 0.11, 0.17, 0.26, 0.07),
    ("projection", "rho_c_proj_true"): (0.15, 0.62, 0.82, 0.84, 0.96, 0.94, 0.96),
    ("projection", "perm_free_p99_abs"): (0.07, 0.21, 0.27, 0.29, 0.28, 0.27, 0.25),
    ("projection", "perm_within_bins_mean"): (0.00, 0.32, 0.47, 0.61, 0.76, 0.76, 0.68),
    ("masking", "frac_masked"): (0.41, 0.34, 0.46, 0.46, 0.41, 0.41, 0.47),
}
# Same source, prose: masked genes as a share of tail (|da| > 0.05) genes
REFERENCE_MASK_TAIL = {"liv_hmonkey": 0.34, "liv_hmouse": 0.24, "liv_hzeb": 0.04, "liv_4sp": 0.29}
MONTE_CARLO = {("projection", "perm_free_p99_abs"), ("projection", "perm_within_bins_mean")}


def ranks(x) -> np.ndarray:
    return pd.Series(np.asarray(x)).rank().to_numpy()


def partial_rank(a, b, Z) -> float:
    """Pearson of rank(a), rank(b) after OLS on the ranks of Z (reference definition)."""
    ra, rb = ranks(a), ranks(b)
    Z1 = np.c_[np.ones(len(ra)), pd.DataFrame(Z).rank().to_numpy()]
    ea = ra - Z1 @ np.linalg.lstsq(Z1, ra, rcond=None)[0]
    eb = rb - Z1 @ np.linalg.lstsq(Z1, rb, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def cv_r2(X: np.ndarray, y: np.ndarray, seed: int = 0) -> float:
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    p = cross_val_predict(RidgeCV(alphas=np.logspace(-2, 4, 13)), X, y,
                          cv=KFold(5, shuffle=True, random_state=seed))
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def projection(F: np.ndarray, c: np.ndarray, S: np.ndarray, log_mean: np.ndarray, kappa: float,
               n_perm: int, rng) -> dict:
    G = len(c)
    X = np.c_[np.ones(G), F]
    Inv = np.linalg.inv(X.T @ X + np.eye(X.shape[1]))

    def proj(y):
        return X @ (Inv @ (X.T @ y))

    h = kappa * proj(S)
    bins = pd.qcut(pd.Series(log_mean).rank(method="first"), N_EXPR_BINS, labels=False).to_numpy()
    members = [np.where(bins == k)[0] for k in range(N_EXPR_BINS)]
    trend = pd.Series(S).groupby(bins).transform("mean").to_numpy()
    free, within = np.empty(n_perm), np.empty(n_perm)
    for i in range(n_perm):
        free[i] = spearmanr(c, proj(rng.permutation(S)))[0]
        ps = S.copy()
        for m in members:
            ps[m] = S[rng.permutation(m)]
        within[i] = spearmanr(c, proj(ps))[0]
    true = float(spearmanr(c, h)[0])
    return dict(kappa=kappa, rho_c_proj_true=true, pearson_c_proj_true=float(pearsonr(c, h)[0]),
                slope_c_on_proj=float(np.polyfit(h, c, 1)[0]),
                rho_c_proj_expr_trend_only=float(spearmanr(c, proj(trend))[0]),
                n_perm=n_perm, perm_free_mean=float(free.mean()),
                perm_free_p99_abs=float(np.percentile(np.abs(free), 99)),
                perm_free_p_true=float((1 + (np.abs(free) >= abs(true)).sum()) / (n_perm + 1)),
                perm_within_bins_mean=float(within.mean()), perm_within_bins_sd=float(within.std()),
                perm_within_bins_p2_5=float(np.percentile(within, 2.5)),
                perm_within_bins_p97_5=float(np.percentile(within, 97.5)),
                perm_within_bins_max=float(within.max()),
                perm_within_bins_p_true=float((1 + (within >= true).sum()) / (n_perm + 1)),
                frac_true_retained_within_bins=float(within.mean() / true) if true else np.nan)


def audit(cohort: str, cds: pd.DataFrame, n_perm: int, seed: int):
    print("==", cohort, flush=True)
    co = C.load_cohort(cohort)
    g = co.genes.join(cds[["cds_length", "log_cds_length", "gc", "gc3"]], on="hsapiens_gene_id")
    F = co.state["gene_features"]
    da = g["delta_alpha"].to_numpy()
    S = g["S_g_ema"].to_numpy()
    c = g["correction"].to_numpy()
    cov_rows = []
    for target in ("delta_alpha", "delta_alpha_raw"):
        y = g[target].to_numpy()
        for cov in COVARIATES:
            x = g[cov].to_numpy(float)
            ok = np.isfinite(x)
            cov_rows.append(dict(cohort=cohort, target=target, statistic="spearman", covariate=cov,
                                 n=int(ok.sum()), rho=float(spearmanr(y[ok], x[ok])[0])))
        Z = g[["log_mean", "detection", "log_var"]].to_numpy()
        cov_rows.append(dict(cohort=cohort, target=target, statistic="partial", covariate="partial_S_given_expr",
                             n=len(y), rho=partial_rank(y, S, Z)))
        ok = g["log_cds_length"].notna().to_numpy()
        Z2 = g.loc[ok, ["log_mean", "detection", "log_var", "log_cds_length", "gc"]].to_numpy()
        cov_rows.append(dict(cohort=cohort, target=target, statistic="partial",
                             covariate="partial_S_given_expr_length_gc", n=int(ok.sum()),
                             rho=partial_rank(y[ok], S[ok], Z2)))
        cov_rows.append(dict(cohort=cohort, target=target, statistic="partial",
                             covariate="partial_length_given_expr", n=int(ok.sum()),
                             rho=partial_rank(y[ok], g.loc[ok, "log_cds_length"], Z2[:, :3])))
    cov_rows.append(dict(cohort=cohort, target="delta_alpha", statistic="coverage", covariate="frac_with_cds",
                         n=len(g), rho=float(g["log_cds_length"].notna().mean())))

    cv_rows = []
    rk = ranks(da) / len(da)
    for name, idx in FEATURE_GROUPS.items():
        cv_rows.append(dict(cohort=cohort, target="delta_alpha", features=name, n_features=len(idx),
                            cv_r2=cv_r2(F[:, idx], da)))
    cv_rows.append(dict(cohort=cohort, target="delta_alpha_rank", features="all72", n_features=72,
                        cv_r2=cv_r2(F, rk)))
    cv_rows.append(dict(cohort=cohort, target="delta_alpha", features="prior_only", n_features=1,
                        cv_r2=cv_r2(g[["prior"]].to_numpy(), da)))
    cv_rows.append(dict(cohort=cohort, target="delta_alpha", features="S_only", n_features=1,
                        cv_r2=cv_r2(S[:, None], da)))

    masked = np.abs(S) < MASK_EPS
    tail = np.abs(da) > TAIL
    mask_row = dict(cohort=cohort, n_genes=len(S), frac_masked=float(masked.mean()),
                    mean_abs_da_masked=float(np.abs(da[masked]).mean()),
                    mean_abs_da_evidence=float(np.abs(da[~masked]).mean()),
                    sd_da_masked=float(da[masked].std()), sd_da_evidence=float(da[~masked].std()),
                    frac_tail_masked=float(tail[masked].mean()), frac_tail_evidence=float(tail[~masked].mean()),
                    share_of_tail_genes_masked=float(masked[tail].mean()) if tail.any() else np.nan,
                    mean_da_masked=float(da[masked].mean()), mean_da_evidence=float(da[~masked].mean()),
                    rho_da_logmean_masked=float(spearmanr(da[masked], g["log_mean"][masked])[0]),
                    rho_ppass_absS=float(spearmanr(g["ppass"], np.abs(S))[0]),
                    tail_threshold=TAIL)

    cfg = co.state["config"]
    kappa = cfg["lambda_fit"] / (2 * cfg["lambda_anchor"])
    pr = dict(cohort=cohort, **projection(F, c, S, g["log_mean"].to_numpy(), kappa, n_perm,
                                          np.random.default_rng(seed)))
    genes = g[["hsapiens_gene_id", "symbol", "delta_alpha", "delta_alpha_raw", "correction", "S_g_ema",
               "log_mean", "detection", "log_var", "tau", "prior", "prior_raw", "ppass",
               "cds_length", "gc", "gc3"]].assign(cohort=cohort).reset_index()
    return cov_rows, cv_rows, mask_row, pr, genes


def reproduction(cov: pd.DataFrame, cv: pd.DataFrame, proj: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    rows = []
    da = cov[cov.target == "delta_alpha"]
    for (kind, key), ref in REFERENCE.items():
        for cohort, rv in zip(C.COHORT_ORDER, ref):
            if kind in ("spearman", "partial"):
                v = da[(da.cohort == cohort) & (da.covariate == key)].rho.iloc[0]
            elif kind == "cv_r2":
                v = cv[(cv.cohort == cohort) & (cv.target == "delta_alpha") & (cv.features == key)].cv_r2.iloc[0]
            elif kind == "cv_r2_rank":
                v = cv[(cv.cohort == cohort) & (cv.target == "delta_alpha_rank")].cv_r2.iloc[0]
            elif kind == "projection":
                v = proj.set_index("cohort").loc[cohort, key]
            else:
                v = mask.set_index("cohort").loc[cohort, key]
            mc = (kind, key) in MONTE_CARLO
            tol = (0.02 if mc else 0.005) + 1e-9
            rows.append(dict(source="reference table, 2 decimals", kind=kind, quantity=key, cohort=cohort, reference=rv,
                             value=float(v), abs_diff=abs(float(v) - rv), tolerance=tol, monte_carlo=mc,
                             within_rounding=abs(float(v) - rv) <= tol))
    for cohort, rv in REFERENCE_MASK_TAIL.items():
        v = mask.set_index("cohort").loc[cohort, "share_of_tail_genes_masked"]
        rows.append(dict(source="reference text", kind="masking", quantity="share_of_tail_genes_masked",
                         cohort=cohort, reference=rv, value=float(v), abs_diff=abs(v - rv), tolerance=0.005,
                         monte_carlo=False, within_rounding=abs(v - rv) <= 0.005 + 1e-9))
    return pd.DataFrame(rows)


# Full-precision reference: (kind, key) -> (file, row selector, column) in --reference-tables.
FULL_REF = {
    ("spearman", "log_mean"): ("cohort_numbers.csv", "rho_da_log_mean"),
    ("spearman", "detection"): ("cohort_numbers.csv", "rho_da_detection"),
    ("spearman", "log_var"): ("cohort_numbers.csv", "rho_da_log_var"),
    ("spearman", "tau"): ("cohort_numbers.csv", "rho_da_tau_specificity"),
    ("spearman", "prior"): ("cohort_numbers.csv", "rho_da_prior"),
    ("spearman", "prior_raw"): ("cohort_numbers.csv", "rho_da_raw_prior"),
    ("spearman", "S_g_ema"): ("cohort_numbers.csv", "rho_da_Sg"),
    ("partial", "partial_S_given_expr"): ("cohort_numbers.csv", "partial_da_Sg_given_expr"),
    ("masking", "frac_masked"): ("cohort_numbers.csv", "Sg_frac_abs_lt1e6"),
    ("cv_r2", "expr3"): ("r2_table.csv", "cv_expr3"),
    ("cv_r2", "esm64"): ("r2_table.csv", "cv_esm64"),
    ("cv_r2", "esm64+expr3"): ("r2_table.csv", "cv_esm64+expr3"),
    ("cv_r2", "all72"): ("r2_table.csv", "cv_all72"),
    ("cv_r2", "prior_only"): ("r2_table.csv", "cv_prior_only"),
    ("cv_r2", "S_only"): ("r2_table.csv", "cv_Sg_only"),
    ("cv_r2_rank", "all72"): ("rank_cv.csv", "cvR2_rank_all72"),
    ("projection", "rho_c_proj_true"): ("proj_summary.csv", "rho_c_projEMA"),
    ("projection", "slope_c_on_proj"): ("proj_summary.csv", "slope_c_on_projEMA"),
    ("projection", "rho_c_proj_expr_trend_only"): ("proj_summary.csv", "rho_c_proj_exprtrendS"),
    ("projection", "perm_free_p99_abs"): ("proj_summary.csv", "perm_free_p99"),
    ("projection", "perm_within_bins_mean"): ("proj_summary.csv", "perm_withinexpr_mean"),
}


def full_reference(ref_dir, cov, cv, proj, mask) -> pd.DataFrame:
    from pathlib import Path

    ref_dir = Path(ref_dir)
    tabs = {"cohort_numbers.csv": pd.read_csv(ref_dir / "cohort_numbers.csv", index_col=0).T.set_index("cohort"),
            "r2_table.csv": pd.read_csv(ref_dir / "r2_table.csv").query("target == 'delta_alpha'").set_index("cohort"),
            "rank_cv.csv": pd.read_csv(ref_dir / "rank_cv.csv").set_index("cohort"),
            "proj_summary.csv": pd.read_csv(ref_dir / "proj_summary.csv", index_col=0).T}
    da = cov[cov.target == "delta_alpha"]
    rows = []
    for (kind, key), (fname, col) in FULL_REF.items():
        for cohort in C.COHORT_ORDER:
            rv = float(tabs[fname].loc[cohort, col])
            if kind in ("spearman", "partial"):
                v = da[(da.cohort == cohort) & (da.covariate == key)].rho.iloc[0]
            elif kind == "cv_r2":
                v = cv[(cv.cohort == cohort) & (cv.target == "delta_alpha") & (cv.features == key)].cv_r2.iloc[0]
            elif kind == "cv_r2_rank":
                v = cv[(cv.cohort == cohort) & (cv.target == "delta_alpha_rank")].cv_r2.iloc[0]
            elif kind == "projection":
                v = proj.set_index("cohort").loc[cohort, key]
            else:
                v = mask.set_index("cohort").loc[cohort, key]
            mc = (kind, key) in MONTE_CARLO
            tol = 0.04 if mc else 0.001
            rows.append(dict(source="reference CSV, full precision", kind=kind, quantity=key, cohort=cohort,
                             reference=rv, value=float(v), abs_diff=abs(float(v) - rv), tolerance=tol,
                             monte_carlo=mc, within_rounding=abs(float(v) - rv) <= tol))
    return pd.DataFrame(rows)


def figure(cov: pd.DataFrame, cv: pd.DataFrame):
    """One panel: Spearman of delta-alpha with each covariate, plus CV R^2 rows, 7 cohorts."""
    sys.path.insert(0, str(gene_ids.REPO_ROOT / "notebooks"))
    try:
        import _figure_design as fd
        fd.apply_design()
    except Exception as exc:  # noqa: BLE001 - fall back to plain matplotlib
        print("figure design module unavailable, plain matplotlib:", exc)
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    da = cov[cov.target == "delta_alpha"]
    keys = ["log_mean", "detection", "log_var", "tau", "prior", "S_g_ema", "log_cds_length", "gc",
            "partial_S_given_expr"]
    M = np.array([[da[(da.cohort == c) & (da.covariate == k)].rho.iloc[0] for c in C.COHORT_ORDER]
                  for k in keys])
    R2 = np.array([[cv[(cv.cohort == c) & (cv.target == "delta_alpha") & (cv.features == f)].cv_r2.iloc[0]
                    for c in C.COHORT_ORDER] for f in ("expr3", "all72")])
    labels = [COV_LABEL[k] for k in keys] + ["CV R² on expression (3)", "CV R² on all 72 inputs"]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    n1 = len(keys)
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
                   extent=(-0.5, M.shape[1] - 0.5, n1 - 0.5, -0.5))
    ax.imshow(R2, cmap="Greys", vmin=0, vmax=1.6, aspect="auto",
              extent=(-0.5, M.shape[1] - 0.5, n1 + 1.5, n1 - 0.5))
    for i in range(n1):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:+.2f}", ha="center", va="center", fontsize=6,
                    color="white" if abs(v) > 0.55 else "#222222")
    for i in range(2):
        for j in range(R2.shape[1]):
            ax.text(j, n1 + i, f"{R2[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if R2[i, j] > 0.9 else "#222222")
    ax.axhline(n1 - 0.5, color="white", linewidth=2)
    ax.set_xlim(-0.5, M.shape[1] - 0.5)
    ax.set_ylim(n1 + 1.5, -0.5)
    ax.set_xticks(range(M.shape[1]))
    ax.set_xticklabels([C.LABELS[c] for c in C.COHORT_ORDER], fontsize=6, rotation=40, ha="right",
                       rotation_mode="anchor")
    ax.set_yticks(range(n1 + 2))
    ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Spearman ρ with Δα", fontsize=6)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_visible(False)
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ensembl-dir", default=str(CDS.ENSEMBL_DIR))
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--reference-tables", default=None,
                    help="folder with the earlier per-cohort CSVs, for a full-precision check")
    add_common_args(ap)
    args = ap.parse_args()
    out = TaskOutput("T2_4", args.outdir, prerun=args.prerun, force=args.force)

    print("reading Ensembl 110 CDS", flush=True)
    cds = CDS.human_cds_table(args.ensembl_dir)
    out.csv(cds.reset_index(), "cds_features.csv")
    cov, cv, mask, proj, genes = [], [], [], [], []
    for i, cohort in enumerate(C.COHORT_ORDER):
        a, b, m, p, g = audit(cohort, cds, args.n_perm, args.seed + i)
        cov += a
        cv += b
        mask.append(m)
        proj.append(p)
        genes.append(g)
    cov, cv, mask, proj = pd.DataFrame(cov), pd.DataFrame(cv), pd.DataFrame(mask), pd.DataFrame(proj)
    rep = reproduction(cov, cv, proj, mask)
    if args.reference_tables:
        rep = pd.concat([full_reference(args.reference_tables, cov, cv, proj, mask), rep], ignore_index=True)
    out.csv(cov, "covariate_table.csv")
    out.csv(cv, "cv_r2.csv")
    out.csv(proj, "projection.csv")
    out.csv(mask, "masking.csv")
    out.csv(pd.concat(genes, ignore_index=True), "gene_covariates.csv")
    out.csv(rep, "reproduction_check.csv")
    fig = figure(cov, cv)
    out.figure(fig, "fig_supp_covariate_audit.pdf")
    out.figure(fig, "fig_supp_covariate_audit.png")
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 400)
    wide = cov[cov.target == "delta_alpha"].pivot(index="covariate", columns="cohort", values="rho")[C.COHORT_ORDER]
    print(wide.round(3).to_string())
    print(cv.pivot_table(index=["target", "features"], columns="cohort", values="cv_r2")[C.COHORT_ORDER].round(3).to_string())
    print(proj.set_index("cohort").T.round(3).to_string())
    print(mask.set_index("cohort").T.round(3).to_string())
    print(rep[~rep.within_rounding].round(4).to_string())
    for src, r in rep.groupby("source"):
        print(f"reproduction [{src}]: {int(r.within_rounding.sum())} of {len(r)} within tolerance")
    out.finish(dict(reproduction_within_rounding=int(rep.within_rounding.sum()), reproduction_rows=len(rep),
                    n_perm=args.n_perm, ensembl_dir=args.ensembl_dir,
                    n_cds_genes=int(len(cds)), frac_canonical=float((cds.cds_source == "canonical").mean())))


if __name__ == "__main__":
    main()
