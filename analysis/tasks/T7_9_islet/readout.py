"""T7.9: islet gene table for H-D1 and H-D1b (plan sections 2, 3, 5 and 8b).

One row per gene of ``islet_healthy_train.h5ad`` (testis human-mouse 1:1 table, joined by human
Ensembl ID through ``load_table("testis")``):

* covariates C (section 3) on the healthy training object: raw prior (the run's
  ``conservation_prior_raw.npy``, checked against the h5ad prior), species-balanced mean log1p
  CP10K and detection (``cohorts.gene_covariates``), log human CDS length (``controls.cds_table``);
* ESM blocks: Leiden modules of a 15-NN graph on the 64 ESM PC features (``T4_3 inputs.esm_blocks``);
* D (section 2): mean over the seed runs of ``T7_9_recipe_b/seed_map.tsv`` (s42 = the selected
  grid run rev-isl-grid14-s42, s43-s46) that pass the training-health gate (G-D2 / G-V thresholds,
  ``run_report.py``; amendment: a failing seed does not enter D or S_ema) of alpha - prior as
  anchored (clamped at 0.85), minus the cohort median; ``D_all_nonnan`` = the same over every
  non-NaN seed, when that set differs (sensitivity); ``D_raw`` = mean alpha - pi_raw minus its median (named sensitivity); ``S_ema`` =
  mean ``S_g_ema``; leave-one-seed-out D; per-seed leave-one-out rho on the C-residual scale;
  rel(D|C) over the 10 2-vs-3 splits; ``D_unflagged`` = the same mean over the seeds that the
  run report does not flag as poorly mixed (Family 3, only if a seed is flagged);
* baselines on the same object (T4.2 functions): B1b_raw = E (cross-species pseudobulk profile
  correlation over the shared classes, units = ``sample_batch`` with >= 30 cells), B1b = E
  residualised on prior, expression and detection splines, B2b (specificity cosine, K >= 4).

All CPU seeds, so the section 2 MPS pooling clause does not apply. No logFC is read.

    PYTHONPATH=<chimera repo> python -I analysis/tasks/T7_9_islet/readout.py
"""

from __future__ import annotations

import itertools
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from analysis.common import cohorts as C
from analysis.common import controls as K
from analysis.common import gene_ids
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.T4_3_scoreboard import inputs as I

ROOT = gene_ids.REPO_ROOT
RUNS = ROOT / "training" / "runs"
H5AD = ROOT / "data" / "analysis" / "disease_islet" / "islet_healthy_train.h5ad"
OUT = ROOT / "results" / "analysis" / "T7_9" / "readout"
SEED_MAP = ROOT / "analysis" / "tasks" / "T7_9_recipe_b" / "seed_map.tsv"
REPORT = ROOT / "results" / "analysis" / "T7_9" / "recipe_b" / "run_report.tsv"
TABLE = "testis"


def seed_runs(partial: bool = False) -> dict[int, str]:
    m = pd.read_csv(SEED_MAP, sep="\t")
    runs = {}
    for _, r in m.iterrows():
        name = str(r["source"]).split()[0].replace("training/runs/", "")
        if not ((RUNS / name / "summary.json").exists() and (RUNS / name / "final_state.npz").exists()):
            if partial:
                continue
            raise FileNotFoundError(f"seed run not finished: {name}")
        runs[int(r["seed"])] = name
    return runs


def main() -> int:
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default=None, help="test runs only")
    p.add_argument("--partial", action="store_true", help="test runs only: use the finished seeds")
    args = p.parse_args()
    global OUT
    if args.outdir:
        OUT = Path(args.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(H5AD)
    tab_t = gene_ids.load_table(TABLE)
    rho_t = gene_ids.assert_table_matches(a, tab_t)
    all_runs = seed_runs(args.partial)
    # training-health gate (G-D2 / G-V, from run_report.py): a failing seed does not enter D or S_ema
    rep = pd.read_csv(REPORT, sep="\t").set_index("run_name")
    for r in all_runs.values():
        if r not in rep.index or REPORT.stat().st_mtime < (RUNS / r / "final_state.npz").stat().st_mtime:
            raise RuntimeError(f"run_report.tsv is missing or older than {r}; rerun run_report.py")
    health = {int(s): dict(run=r, no_nan=bool(rep.loc[r, "gd2_no_nan"]), pass_gate=bool(rep.loc[r, "gd2_pass_GD2"]),
                           flag_poor_mixing=bool(rep.loc[r, "flag_poor_mixing"])) for s, r in all_runs.items()}
    runs = {s: r for s, r in all_runs.items() if health[s]["pass_gate"]}
    nonnan = {s: r for s, r in all_runs.items() if health[s]["no_nan"]}
    if len(runs) < 2:
        raise RuntimeError(f"fewer than 2 seeds pass the health gate: {health}; report descriptively")
    h5_prior = a.var["conservation_prior"].to_numpy(dtype=float)
    for s, r in nonnan.items():
        raw = np.load(RUNS / r / "conservation_prior_raw.npy")
        if not np.allclose(raw, h5_prior, atol=1e-6):
            raise AssertionError(f"{r}: conservation_prior_raw.npy differs from the h5ad prior")
    praw = h5_prior
    g = C.gene_covariates(a)
    g["hsapiens_gene_id"] = gene_ids.ensembl(TABLE, a.var_names).values
    g["symbol"] = gene_ids.symbols(TABLE, a.var_names).values
    ids = pd.Index(a.var_names).str.replace("ortho_", "", regex=False).astype(int)
    g["mouse_id"] = tab_t.loc[ids, "mmusculus_gene_id"].to_numpy()
    g["prior_raw"] = praw
    g = g.join(K.cds_table()[["log_cds_length"]], on="hsapiens_gene_id")
    F = np.asarray(a.varm["gene_features"], dtype=float)
    blk, binfo = I.esm_blocks(F[:, :I.N_PCS])
    g["block"] = blk

    # seeds -> D
    st_all = {s: K.read_run(r) for s, r in sorted(nonnan.items())}
    st = {s: x for s, x in st_all.items() if s in runs}
    for s, x in st.items():
        if not np.isfinite(x["alpha"]).all():
            raise AssertionError(f"seed {s} ({runs[s]}): non-finite alpha")
        if len(x["alpha"]) != a.n_vars:
            raise AssertionError(f"seed {s}: {len(x['alpha'])} genes, object {a.n_vars}")
    M = pd.DataFrame({s: x["D"] for s, x in st.items()}, index=a.var_names)
    A = pd.DataFrame({s: x["alpha"] for s, x in st.items()}, index=a.var_names)
    md = M.mean(1)
    g["D"] = md - md.median()
    dr = A.mean(1) - praw
    g["D_raw"] = dr - dr.median()
    g["S_ema"] = np.column_stack([x["S_g_ema"] for x in st.values()]).mean(1)
    for s in M.columns:
        g[f"D_loso_{s}"] = M.drop(columns=s).mean(1)
    if set(nonnan) != set(runs):  # sensitivity: D over all non-NaN seeds
        mn = pd.DataFrame({s: x["D"] for s, x in st_all.items()}, index=a.var_names).mean(1)
        g["D_all_nonnan"] = mn - mn.median()
    flagged = [s for s in runs if health[s]["flag_poor_mixing"]]
    if flagged:
        mu = M.drop(columns=flagged).mean(1)
        g["D_unflagged"] = mu - mu.median()
    ok = g[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
    Z = K.design(g.loc[ok], K.COVARIATE_COLUMNS)
    loo = {}
    for s in M.columns:
        e1 = K.residual_rank(M.loc[ok, s].to_numpy(), Z)
        e2 = K.residual_rank(M.loc[ok].drop(columns=s).mean(1).to_numpy(), Z)
        loo[int(s)] = dict(run=runs[s], rho_resid=float(np.corrcoef(e1, e2)[0, 1]),
                           spearman=float(spearmanr(M[s], M.drop(columns=s).mean(1))[0]))
    rs = []
    seeds = list(M.columns)
    for two in itertools.combinations(seeds, 2):
        three = [s for s in seeds if s not in two]
        e1 = K.residual_rank(M.loc[ok, list(two)].mean(1).to_numpy(), Z)
        e2 = K.residual_rank(M.loc[ok, three].mean(1).to_numpy(), Z)
        rs.append(float(np.corrcoef(e1, e2)[0, 1]))
    r = float(np.mean(rs))

    # baselines B1b, B2b (T4.2 definitions) on the healthy object
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    sl = sorted(set(species))
    x, f, units = T42.pseudobulk(X, species, a.obs["cell_ontology_class"].astype(str).to_numpy(),
                                 a.obs["sample_batch"].astype(str).to_numpy(), T42.MIN_CELLS_TYPE)
    E, nE, pinfo = T42.pooled_E(x, f, sl, a.n_vars)
    L, nL, B2, nB2, groups = T42.level_and_specificity(x, f, sl, a.n_vars)
    g["B1b_raw"] = E
    g["B1b"] = T42.residualise(pd.Series(E, index=g.index), g[["prior_raw", "log_cp10k_spmean", "detection_spmean"]])
    g["B2b"] = B2
    g["var_name"] = g.index
    g = g.reset_index(drop=True)
    g.to_parquet(OUT / "islet_genes.parquet")
    units.to_csv(OUT / "baseline_units.csv", index=False)
    meta = dict(seeds={int(k): v for k, v in runs.items()}, health=health,
                seeds_all_nonnan={int(k): v for k, v in nonnan.items()}, n_genes=int(a.n_vars), assert_table_matches=rho_t,
                blocks=binfo, leave_one_seed_out=loo, rel_D_given_C=dict(r_mean_2v3=r, rel=2 * r / (1 + r)),
                prior_clamped=bool(not np.allclose(st[min(st)]["prior"], praw)),
                flagged_poor_mixing=flagged,
                frac_D_pos=float((g["D"] > 0).mean()), sd_D=float(g["D"].std()),
                sd_D_raw=float(g["D_raw"].std()), spearman_D_D_raw=float(spearmanr(g["D"], g["D_raw"])[0]),
                b1b_pairs=[{k: v for k, v in p.items()} for p in pinfo], shared_groups=groups,
                n_B1b=int(np.isfinite(E).sum()), n_B2b=int(np.isfinite(B2).sum()))
    (OUT / "readout_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k not in ("b1b_pairs",)}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
