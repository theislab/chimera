"""T11.5: Van Zyl gene table, covariate audit, baselines, H5/H6-style sets, margin support.

Per usable dataset (``vz.usable_datasets``), with sections 2-5 and 9 of the pre-specified plan unchanged:

* seeds in D: ``vz.seeds_in_D`` (latest amendment rows); genes with alpha > 0.95 in a seed are flagged;
* covariates C: ``cohorts.gene_covariates`` (tau over ``cell_type_harmonised``), raw prior from the
  run's ``conservation_prior_raw.npy`` (checked against the h5ad prior), log human CDS length;
* D = mean over D seeds of alpha - prior (as anchored, clamped 0.85 on clamped recipes) minus the
  median; ``D_raw`` against the raw prior; ``S_ema`` = mean ``S_g_ema``;
* covariate audit (T2.4 functions): Spearman of D with each covariate, partial rho(D, S | expression),
  5-fold CV R^2 of D on the GIN feature groups;
* B1 (T4.2 functions): cross-species pseudobulk profile correlation E over ``cell_ontology_class``,
  units ``obs['batch']`` with >= 30 cells, pooled over species pairs; ``B1_pi`` residualised on the
  raw prior, ``B1b`` on prior, expression and detection; pairwise ``E_human_<species>``;
* sets (section 4, T4.3 functions): C1, C2 above, V2, V3 below matched background in raw prior x
  expression decile cells (strata kept with >= 5 background genes per set gene); stratified AUROC,
  1,000 gene bootstraps, 2,000 permutations within strata, BH over the four sets; scores D, D without
  flagged saturated genes, B1b and, stratified on expression only, the raw prior;
* tidy per-gene table ``results/revision/T11/genes/genes_<dataset>.tsv`` (gene id, pi, alpha per seed,
  centred Delta-alpha, S_ema, count support, margin support, saturated flag) for the harmonised panels;
* margin support (section 9, T2.5 functions): per D seed, fraction of |Delta-alpha| > 0.05 genes whose
  S_g_ema agrees in sign and is at least the median |S|; mean over seeds.

No glaucoma gene is read here.

    PYTHONPATH=<chimera repo> python revision/tasks/T11_5_vz_genes/readout.py [--datasets full,hmouse]
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from revision.common import cohorts as C
from revision.common import controls as K
from revision.common import gene_ids
from revision.tasks.T11_5_vz_genes import vz
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.T4_3_scoreboard import inputs as I
from revision.tasks.T4_3_scoreboard import stats as S

sys.path.insert(0, str(vz.ROOT / "revision" / "tasks" / "T2_5_celltype_margin"))
import margin as MG  # noqa: E402


def _load(name: str, path):
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_T24 = _load("t24_covariates_run", vz.ROOT / "revision" / "tasks" / "T2_4_covariates" / "run.py")
cv_r2, partial_rank = _T24.cv_r2, _T24.partial_rank  # T2.4 functions, unchanged

OUT = vz.OUT / "readout"
GOLD = vz.ROOT / "results" / "revision" / "T4_3" / "golds" / "gold_table.csv"
SETS = {"C1": "above", "C2": "above", "V2": "below", "V3": "below"}
AUDIT = ["log_mean", "detection", "log_var", "tau", "prior", "prior_raw", "S_ema", "log_cds_length", "gc",
         "log_cp10k_spmean", "detection_spmean"]
FEATURE_GROUPS = {"expr3": C.GIN_COLUMNS["expr3"], "esm64": C.GIN_COLUMNS["esm64"],
                  "esm64+expr3": C.GIN_COLUMNS["esm64"] + C.GIN_COLUMNS["expr3"], "all72": list(range(72))}
N_BOOT, N_PERM = 1000, 2000


def set_tests(g: pd.DataFrame, gold: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for score, strata in (("D", "pi_expr"), ("D_unsat", "pi_expr"), ("B1b", "pi_expr"), ("prior_raw", "expr")):
        m = g[[score, "prior_raw", "log_cp10k_spmean"]].notna().all(1).to_numpy()
        T = g.loc[m]
        lab = S.cells(T["prior_raw"].to_numpy(float) if strata == "pi_expr" else None,
                      T["log_cp10k_spmean"].to_numpy(float))
        for st, direction in SETS.items():
            inset = T["hsapiens_gene_id"].map(gold[st]).fillna(False).to_numpy(bool)
            keep = S.kept_strata(inset, lab)
            r = dict(score=score, set=st, direction=direction, strata=strata, n_universe=int(m.sum()),
                     n_set_universe=int(inset.sum()), n_set=int(inset[keep].sum()), n_background=int((~inset[keep]).sum()))
            if r["n_set"] >= 3:
                sc, ins, lb = T[score].to_numpy(float)[keep], inset[keep], lab[keep]
                r["auroc"] = S.strat_auroc(sc, ins, lb)
                A, _ = S.auroc_boot(sc, ins, lb, N_BOOT, rng)
                r["auroc_ci_lo"], r["auroc_ci_hi"] = S.pct_ci(A)
                null = S.auroc_perm(sc, ins, lb, N_PERM, rng)
                r["p_perm"] = S.p_one_sided(r["auroc"], null, lower=(direction == "below"))
            rows.append(r)
    t = pd.DataFrame(rows)
    for sc in t["score"].unique():
        m = (t["score"] == sc) & t["p_perm"].notna()
        t.loc[m, "q_bh"] = S.bh(t.loc[m, "p_perm"].tolist())
    t["predicted_direction"] = np.where(t["direction"] == "above", t["auroc"] > 0.5, t["auroc"] < 0.5)
    return t


def margin_support(ds: str, runs: dict, a, ppass: np.ndarray) -> dict:
    out = {}
    for s, run in runs.items():
        f = vz.OUT / "drivers" / f"fw_{ds}_s{s}.npz"
        if not f.exists():
            out[int(s)] = None
            continue
        fw = dict(np.load(f, allow_pickle=False))
        if list(fw["var_names"]) != list(a.var_names):
            raise AssertionError(f"{f.name}: var_names differ")
        dec = MG.decompose(fw, vz.CLASS_KEY)
        st = K.read_run(run)
        genes = pd.DataFrame(dict(delta_alpha=st["D"], S_g_ema=st["S_g_ema"], ppass=ppass), index=a.var_names)
        out[int(s)] = MG.headline(MG.gene_table(fw, dec, genes), ds, vz.CLASS_KEY)
    vals = [v["frac_tail_with_margin_support"] for v in out.values() if v]
    return dict(per_seed=out, mean_frac_tail_with_margin_support=float(np.mean(vals)) if vals else None)


def readout(ds: str, gold: pd.DataFrame) -> None:
    rng = np.random.default_rng(20261102)
    a = vz.read_h5ad(ds)
    rho_t = gene_ids.assert_table_matches(a, gene_ids.load_table(vz.TABLE_COHORT))
    health = vz.seed_health(ds)
    runs = vz.seeds_in_D(ds)
    if len(runs) < vz.MIN_SEEDS:
        raise SystemExit(f"{ds}: fewer than {vz.MIN_SEEDS} seeds pass health")
    first = runs[min(runs)]
    praw = C.raw_prior(vz.TABLE_COHORT, first, a.var["conservation_prior"].to_numpy(dtype=float))
    a_tau = a.copy()
    a_tau.obs["cell_type"] = a.obs[vz.LABEL_KEY].astype(str).values  # tau over harmonised types
    g = C.gene_covariates(a_tau)
    del a_tau
    g["hsapiens_gene_id"] = gene_ids.ensembl(vz.TABLE_COHORT, a.var_names).values
    g["symbol"] = gene_ids.symbols(vz.TABLE_COHORT, a.var_names).values
    g["prior_raw"] = praw
    g = g.join(K.cds_table()[["log_cds_length", "gc"]], on="hsapiens_gene_id")
    F = np.asarray(a.varm["gene_features"], dtype=float)
    blk, binfo = I.esm_blocks(F[:, :I.N_PCS])
    g["block"] = blk

    st = {s: K.read_run(r) for s, r in sorted(runs.items())}
    for s, x in st.items():
        if len(x["alpha"]) != a.n_vars or not np.isfinite(x["alpha"]).all():
            raise AssertionError(f"{ds} s{s}: wrong length or non-finite alpha")
    Dm = pd.DataFrame({s: x["D"] for s, x in st.items()}, index=a.var_names)
    Am = pd.DataFrame({s: x["alpha"] for s, x in st.items()}, index=a.var_names)
    g["prior"] = st[min(st)]["prior"]
    md = Dm.mean(1)
    g["D"] = md - md.median()
    dr = Am.mean(1) - praw
    g["D_raw"] = dr - dr.median()
    g["S_ema"] = np.mean([x["S_g_ema"] for x in st.values()], axis=0)
    g["saturated_flag"] = (Am > 0.95).any(axis=1).to_numpy()
    g["D_unsat"] = g["D"].where(~g["saturated_flag"])  # sensitivity without flagged genes
    for s in Dm.columns:
        g[f"D_s{s}"] = Dm[s]

    # covariate audit (T2.4 definitions)
    aud = [dict(covariate=c, spearman=float(spearmanr(g["D"], g[c], nan_policy="omit")[0])) for c in AUDIT]
    expr = g[["log_mean", "detection", "log_var"]].to_numpy()
    aud.append(dict(covariate="partial_S_given_expr", spearman=partial_rank(g["D"], g["S_ema"], expr)))
    cv = [dict(features=k, cv_r2=cv_r2(F[:, cols], g["D"].to_numpy())) for k, cols in FEATURE_GROUPS.items()]

    # B1 baselines (T4.2)
    X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
    X = X.astype(np.float64)
    species = a.obs["species"].astype(str).to_numpy()
    sl = sorted(set(species))
    x, f, units = T42.pseudobulk(X, species, a.obs[vz.CLASS_KEY].astype(str).to_numpy(),
                                 a.obs["batch"].astype(str).to_numpy(), T42.MIN_CELLS_TYPE)
    E, nE, pinfo = T42.pooled_E(x, f, sl, a.n_vars)
    g["B1_raw"] = E
    g["B1_pi"] = T42.residualise(pd.Series(E, index=g.index), g[["prior_raw"]])
    g["B1b"] = T42.residualise(pd.Series(E, index=g.index), g[["prior_raw", "log_cp10k_spmean", "detection_spmean"]])
    for s2 in sl:
        if s2 == "human":
            continue
        grp = T42.shared_groups(x, sl, ["human", s2])
        rr, _ = T42.pair_corr(x, f, "human", s2, grp, a.n_vars)
        g[f"E_human_{s2}"] = rr

    sets = set_tests(g, gold, rng)
    ms_ = margin_support(ds, runs, a, g["ppass"].to_numpy())

    OUT.mkdir(parents=True, exist_ok=True)
    g.reset_index(names="var_name").to_parquet(OUT / f"genes_{ds}.parquet")
    # Tidy per-gene table for the harmonised panels (one row per vocabulary gene).
    tidy = pd.DataFrame(dict(hsapiens_gene_id=g["hsapiens_gene_id"], symbol=g["symbol"], var_name=g.index,
                             pi=g["prior_raw"], pi_anchored=g["prior"]))
    for s in Am.columns:
        tidy[f"alpha_s{s}"] = Am[s]
    tidy["delta_alpha_centred"] = g["D"]
    tidy["S_ema"] = g["S_ema"]
    tidy["count_support"] = np.abs(g["S_ema"]) >= MG.MASK_EPS  # |S_g_ema| >= 1e-6, X4 definition
    tidy["margin_support"] = MG.margin_flag(md.to_numpy(), g["S_ema"].to_numpy())  # T2.5 flag on the seed mean
    tidy["saturated_flag"] = g["saturated_flag"]
    tidy["log_cp10k_spmean"] = g["log_cp10k_spmean"]
    tidy["detection_spmean"] = g["detection_spmean"]
    tidy["B1b_pseudobulk"] = g["B1b"]
    (vz.OUT / "genes").mkdir(parents=True, exist_ok=True)
    tidy.to_csv(vz.OUT / "genes" / f"genes_{ds}.tsv", sep="\t", index=False)
    health.to_csv(OUT / f"seed_health_{ds}.tsv", sep="\t", index=False)
    pd.DataFrame(aud).to_csv(OUT / f"covariate_audit_{ds}.csv", index=False)
    pd.DataFrame(cv).to_csv(OUT / f"cv_r2_{ds}.csv", index=False)
    sets.to_csv(OUT / f"sets_{ds}.csv", index=False)
    meta = dict(dataset=ds, n_genes=int(a.n_vars), n_cells=int(a.n_obs), assert_table_matches=rho_t,
                seeds_D={int(k): v for k, v in runs.items()}, esm_blocks=binfo,
                n_saturated_flag=int(g["saturated_flag"].sum()),
                sd_D=float(g["D"].std()), frac_D_pos=float((g["D"] > 0).mean()),
                rho_D_Draw=float(spearmanr(g["D"], g["D_raw"])[0]),
                rho_D_S=float(spearmanr(g["D"], g["S_ema"])[0]), rho_D_B1raw=float(spearmanr(g["D"], g["B1_raw"], nan_policy="omit")[0]),
                n_B1=int(np.isfinite(E).sum()), b1_pairs=[{k: v for k, v in p.items() if k != "groups"} for p in pinfo],
                units_used=int(units["used"].sum()), margin_support=ms_,
                integration_reference=vz.integration_reference(a.obs))
    vz.write_json(OUT / f"meta_{ds}.json", meta)
    print(ds, {k: meta[k] for k in ("n_genes", "seeds_D", "n_saturated_flag", "sd_D", "rho_D_S", "rho_D_B1raw")})


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default=None)
    a = p.parse_args()
    gold = pd.read_csv(GOLD, index_col=0)[list(SETS)].astype(bool)
    for ds in (a.datasets.split(",") if a.datasets else vz.usable_datasets()):
        readout(ds, gold)
    return 0


if __name__ == "__main__":
    sys.exit(main())
