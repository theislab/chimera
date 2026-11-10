"""T12.3 step 2: bone-marrow gene table, D and baselines (plan sections 2, 3, 5 and 12).

One row per vocabulary gene of build 2 (``gene_ids.h5ad_path("bone_marrow")``, 12,417 genes):

* covariates C (section 3): raw prior (the run's ``conservation_prior_raw.npy``, checked against the
  h5ad prior), species-balanced mean log1p CP10K and detection (``cohorts.gene_covariates``), log
  human CDS length (``controls.cds_table``);
* ESM blocks: Leiden modules of a 15-NN graph on the 64 ESM PCs (``T4_3 inputs.esm_blocks``);
* D (section 2): mean over the seeds admitted by ``seed_health.py`` (``seed_health.json``: seeds
  passing G-V, amendment chimera-private a82eaad) of alpha - prior as anchored (clamped 0.85), minus
  the cohort median. ``D_raw`` = mean alpha - pi_raw minus its median (named sensitivity);
  ``D_all`` = the same D over every finite seed, G-V failures included (the amendment's
  sensitivity); ``S_ema`` = mean ``S_g_ema`` (Family 3);
  per-seed leave-one-out rho on the C-residual scale; rel(D|C) over the 10 2-vs-3 splits;
* B1 (section 12): ``B1_raw`` = E, the T4.2 cross-species pseudobulk profile correlation over the
  nine harmonised types (``cell_ontology_class``), units = ``obs['batch']`` (species | donor |
  tissue) with >= 30 cells, the three species pairs Fisher-z pooled (``T4_2.pooled_E``);
  ``B1_pi`` = E residualised on the spline of rank pi_raw only ("B1 residualised on pi");
  ``B1b`` = E residualised on pi, expression and detection (T4.2 B1b, beside); pairwise
  ``E_human_mouse``, ``E_human_mouse_lemur`` for E4.

No COE, Shay list or other external measurement is read here.

    PYTHONPATH=<chimera repo> python analysis/tasks/T12_3_tests/readout.py
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

COHORT = "bone_marrow"
ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis" / "P12"
OUT = RES / "readout"
UNIT_KEY = "batch"
TYPE_KEY = "cell_ontology_class"


def seeds_in_D() -> dict[int, str]:
    h = json.loads((RES / "seed_health.json").read_text())
    tab = pd.read_csv(RES / "seed_health.tsv", sep="\t")
    keep = set(h["seeds_enter_D"])
    return {int(r.seed): r.run_name for r in tab.itertuples() if int(r.seed) in keep}


def finite_seeds() -> dict[int, str]:
    h = json.loads((RES / "seed_health.json").read_text())
    tab = pd.read_csv(RES / "seed_health.tsv", sep="\t")
    return {int(r.seed): r.run_name for r in tab.itertuples() if int(r.seed) in set(h["seeds_finite"])}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a = ad.read_h5ad(gene_ids.h5ad_path(COHORT))
    rho_t = gene_ids.assert_table_matches(a, gene_ids.load_table(COHORT))
    runs = seeds_in_D()
    if len(runs) < 2:
        raise SystemExit("fewer than 2 seeds pass G-V: descriptive only (amendment a82eaad)")
    first = runs[min(runs)]
    praw = C.raw_prior(COHORT, first, a.var["conservation_prior"].to_numpy(dtype=float))
    g = C.gene_covariates(a)
    g["hsapiens_gene_id"] = gene_ids.ensembl(COHORT, a.var_names).values
    g["symbol"] = gene_ids.symbols(COHORT, a.var_names).values
    g["prior_raw"] = praw
    g = g.join(K.cds_table()[["log_cds_length"]], on="hsapiens_gene_id")
    F = np.asarray(a.varm["gene_features"], dtype=float)
    blk, binfo = I.esm_blocks(F[:, :I.N_PCS])
    g["block"] = blk

    st = {s: K.read_run(r) for s, r in sorted(runs.items())}
    for s, x in st.items():
        if len(x["alpha"]) != a.n_vars or not np.isfinite(x["alpha"]).all():
            raise AssertionError(f"seed {s}: wrong length or non-finite alpha")
    M = pd.DataFrame({s: x["D"] for s, x in st.items()}, index=a.var_names)
    A = pd.DataFrame({s: x["alpha"] for s, x in st.items()}, index=a.var_names)
    md = M.mean(1)
    g["D"] = md - md.median()
    dr = A.mean(1) - praw
    g["D_raw"] = dr - dr.median()
    Sm = np.column_stack([x["S_g_ema"] for x in st.values()])
    g["S_ema"] = Sm.mean(1)
    fin = finite_seeds()
    Mall = pd.DataFrame({s: K.read_run(r)["D"] for s, r in sorted(fin.items())}, index=a.var_names)
    g["D_all"] = Mall.mean(1) - Mall.mean(1).median()  # sensitivity: every finite seed
    for s in M.columns:
        g[f"D_s{s}"] = M[s]
    ok = g[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
    Z = K.design(g.loc[ok], K.COVARIATE_COLUMNS)
    loo = {}
    for s in M.columns:
        e1 = K.residual_rank(M.loc[ok, s].to_numpy(), Z)
        e2 = K.residual_rank(M.loc[ok].drop(columns=s).mean(1).to_numpy(), Z)
        loo[int(s)] = dict(run=runs[s], rho_resid=float(np.corrcoef(e1, e2)[0, 1]),
                           spearman=float(spearmanr(M[s], M.drop(columns=s).mean(1))[0]),
                           spearman_S=float(spearmanr(Sm[:, list(M.columns).index(s)],
                                                      np.delete(Sm, list(M.columns).index(s), 1).mean(1))[0]))
    rs = []
    seeds = list(M.columns)
    if len(seeds) == 5:  # rel(D|C) needs the five-seed core
        for two in itertools.combinations(seeds, 2):
            three = [s for s in seeds if s not in two]
            e1 = K.residual_rank(M.loc[ok, list(two)].mean(1).to_numpy(), Z)
            e2 = K.residual_rank(M.loc[ok, three].mean(1).to_numpy(), Z)
            rs.append(float(np.corrcoef(e1, e2)[0, 1]))
    r = float(np.mean(rs)) if rs else float("nan")

    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    sl = sorted(set(species))
    x, f, units = T42.pseudobulk(X, species, a.obs[TYPE_KEY].astype(str).to_numpy(),
                                 a.obs[UNIT_KEY].astype(str).to_numpy(), T42.MIN_CELLS_TYPE)
    E, nE, pinfo = T42.pooled_E(x, f, sl, a.n_vars)
    g["B1_raw"] = E
    g["B1_n_pairs"] = nE
    g["B1_pi"] = T42.residualise(pd.Series(E, index=g.index), g[["prior_raw"]])
    g["B1b"] = T42.residualise(pd.Series(E, index=g.index), g[["prior_raw", "log_cp10k_spmean", "detection_spmean"]])
    for s1, s2 in itertools.combinations(sl, 2):
        grp = T42.shared_groups(x, sl, [s1, s2])
        rr, _ = T42.pair_corr(x, f, s1, s2, grp, a.n_vars)
        g[f"E_{s1}_{s2}"] = rr
    g["var_name"] = g.index
    g = g.reset_index(drop=True)
    g.to_parquet(OUT / "bm_genes.parquet")
    units.to_csv(OUT / "baseline_units.csv", index=False)
    meta = dict(seeds={int(k): v for k, v in runs.items()}, seeds_D_all=sorted(int(s) for s in fin),
                rho_D_D_all=float(spearmanr(g["D"], g["D_all"])[0]), n_genes=int(a.n_vars), assert_table_matches=rho_t,
                blocks=binfo, leave_one_seed_out=loo, rel_D_given_C=dict(r_mean_2v3=r, rel=2 * r / (1 + r) if rs else None),
                prior_clamped=bool(not np.allclose(st[min(st)]["prior"], praw)),
                frac_D_pos=float((g["D"] > 0).mean()), sd_D=float(g["D"].std()),
                b1_pairs=pinfo, n_B1=int(np.isfinite(E).sum()),
                units_used=int(units["used"].sum()), units_total=int(len(units)))
    (OUT / "readout_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: v for k, v in meta.items() if k != "b1_pairs"}, indent=2, default=str))
    print(json.dumps([{k: v for k, v in p.items() if k != "groups"} | {"groups": p["groups"]} for p in pinfo], default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
