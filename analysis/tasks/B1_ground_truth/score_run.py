"""B1: every gene-level readout of one fit, recomputed on that fit's own (altered) object. Writes per-gene scores.

Chimera: S_g (S_g_ema of the final state), S_fwd (the same log-likelihood ratio from a forward pass over all cells at
the posterior mean, no 20-cell mask, added to separate the mask from count evidence; exploratory), D (alpha - prior,
minus the run's median), D_raw (alpha - pi_raw, minus its median), B1z (E on Leiden clusters of the fit's z,
residualised as T4.2), B1z_raw.
Direct, same code as T4.2 and X4/X6 on the altered counts: B1 (E, the profile correlation, raw), B1b, B1c (residualised),
B1c_raw, B2b, B3a (5-fold ridge on the 72 features, as moved by alter.py, predicting E), Sdir (labelled direct margin),
Lfc (minus the absolute log2 ratio of species pseudobulk rates; direct fold change, added here as the plain
comparator on level).
Static: pi, dN/dS, LOEUF. Control: expr (post-alteration species-balanced log CP10K), random.
All scores are oriented so that higher means more conserved (metrics.py negates dN/dS and LOEUF).

A failed or non-finite fit leaves its Chimera columns NaN and is recorded as discarded; the direct, sequence and control readouts of that altered
object are computed all the same, so a Chimera failure removes only Chimera's row.

Also per run: health and mixing. Writes results/analysis/B1/scores/<run>.parquet and health/<run>.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SLURM_CPUS_PER_TASK", "4"))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as CO
from analysis.tasks.B1_ground_truth import alter as A
from analysis.tasks.B1_ground_truth import rules as RU
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X6_adds_beyond.degrade import sdir


def _stats_for_covariates(a, cohort: str) -> pd.DataFrame:
    a2 = a.copy()
    a2.obs["cell_type"] = A.labels(a, cohort)
    return CO.gene_covariates(a2)


def forward_margin(run_dir, a, cfg: dict, batch_size: int = 512) -> np.ndarray:
    """S_fwd: species-averaged mean over all cells of the model's LLR at the posterior mean. No mask."""
    import torch

    from analysis.tasks.T2_5_celltype_margin.run import load_components, nb_llr

    comp = load_components(run_dir, a, cfg)
    X = A.csr(a.X)
    N, G = X.shape
    s_idx, b_idx = comp["species_codes"], comp["batch_codes"]
    S = len(comp["species_names"])
    tot = np.zeros((S, G))
    with torch.no_grad():
        for st in range(0, N, batch_size):
            idx = np.arange(st, min(st + batch_size, N))
            x = torch.tensor(X[idx].toarray(), dtype=torch.float32)
            s_t = torch.tensor(s_idx[idx], dtype=torch.long)
            b_t = torch.tensor(b_idx[idx], dtype=torch.long)
            L = x.sum(1, keepdim=True).double()
            zmu, _ = comp["encode"](x, s_t, b_t)
            mu1 = torch.exp(comp["shared_rate"](zmu).double()) * L
            Lt = mu1.sum(1, keepdim=True).clamp(min=1.0)
            mu0 = (comp["p_null"][s_t] * Lt).clamp(min=1e-8)
            ll = nb_llr(x.double(), mu1, mu0, comp["theta"][b_t]).numpy()
            np.add.at(tot, s_idx[idx], ll)
    n = np.bincount(s_idx, minlength=S)[:, None]
    return (tot / n).mean(0)


def species_mixing(z: np.ndarray, spc: np.ndarray, k: int = 15, seed: int = 0) -> dict:
    """Species entropy in 15-NN of z (normalised to [0, 1]) and purity, on at most 10,000 cells as queries."""
    from sklearn.neighbors import NearestNeighbors

    rng = np.random.default_rng(seed)
    q = rng.choice(len(z), min(10000, len(z)), replace=False)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    ind = nn.kneighbors(z[q], return_distance=False)[:, 1:]
    y = (spc == spc[0]).astype(float)
    p = y[ind].mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(np.where(p > 0, p * np.log2(p), 0) + np.where(p < 1, (1 - p) * np.log2(1 - p), 0))
    return dict(mix_entropy=float(h.mean()), mix_purity=float(np.maximum(p, 1 - p).mean()))


def load_fit(run: str) -> dict | None:
    """Final state of a fit, or None if it has none or a non-finite alpha, z or S_g."""
    if not (CO.RUNS_DIR / run / "final_state.npz").exists():
        return None
    st = CO.load_state(run)
    ok = all(np.isfinite(st[k]).all() for k in ("alpha", "z_mean", "S_g_ema", "prior"))
    return st if ok else None


def score(row: dict, h5ad: str | None, force_keep_model: bool = False) -> None:
    run = row["run"]
    cohort = row["cohort"]
    out = RU.RES / "scores" / f"{run}.parquet"
    if out.exists():
        print("exists", out, flush=True)
        return
    cfg_c = RU.COHORTS[cohort]
    import anndata as ad

    a = ad.read_h5ad(h5ad or (RU.REPO / cfg_c["h5ad"]))
    run_dir = CO.RUNS_DIR / run
    st = load_fit(run)
    X = A.csr(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = A.labels(a, cohort)
    library = a.obs[cfg_c["library_key"]].astype(str).to_numpy()
    G = a.n_vars
    species = sorted(set(spc))
    cov = _stats_for_covariates(a, cohort)
    cov["prior_raw"] = np.asarray(a.var["conservation_prior"], dtype=float)

    res = pd.DataFrame(index=pd.Index(a.var_names.astype(str), name="var_name"))
    res["hsapiens_gene_id"] = np.asarray(CO.gene_ids.ensembl(cfg_c["gene_table"], res.index))
    res["prior_raw"] = cov["prior_raw"].to_numpy()
    for c in ("alpha", "prior", "S_g", "D", "D_raw", "correction", "B1z_raw", "B1z", "S_fwd"):
        res[c] = np.nan
    if st is not None:
        res["alpha"], res["prior"] = st["alpha"].astype(float), st["prior"].astype(float)
        res["S_g"] = st["S_g_ema"].astype(float)
        da = res["alpha"] - res["prior"]
        dr = res["alpha"] - res["prior_raw"]
        res["D"], res["D_raw"] = da - da.median(), dr - dr.median()
        res["correction"] = st["correction"].astype(float)
    res["expr"] = cov["log_cp10k_spmean"].to_numpy()
    res["detection_spmean"] = cov["detection_spmean"].to_numpy()
    res["ppass"] = cov["ppass"].to_numpy()
    res["n_cells_changed_hint"] = np.nan

    # direct, cell-type labels
    x, f, units = T42.pseudobulk(X, spc, lab, library, T42.MIN_CELLS_TYPE)
    E, _, _ = T42.pooled_E(x, f, species, G)
    Lv, _, B2, _, groups = T42.level_and_specificity(x, f, species, G)
    c3 = cov[["prior_raw", "log_cp10k_spmean", "detection_spmean"]]
    res["B1"] = E
    res["B1b"] = T42.residualise(pd.Series(E, index=res.index), c3)
    res["B1c_raw"] = Lv
    res["B1c"] = T42.residualise(pd.Series(Lv, index=res.index), c3)
    res["B2b"] = B2
    res["Sdir"] = sdir(X, spc, lab)
    p0, p1 = [(np.asarray(X[spc == s].sum(0)).ravel() + 0.5) / (X[spc == s].sum() + 0.5 * G) for s in species]
    res["Lfc"] = -np.abs(np.log2(p0 / p1))
    F = np.asarray(a.varm["gene_features"], dtype=float)
    b3a, _, _ = T42.cross_fit_ridge(F, np.asarray(E, dtype=float))
    res["B3a"] = b3a

    # latent
    if st is not None:
        z = np.asarray(st["z_mean"], dtype=np.float32)
        cl = T42.leiden_clusters(z)
        xz, fz, _ = T42.pseudobulk(X, spc, cl, library, T42.MIN_CELLS_CLUSTER)
        Ez, _, _ = T42.pooled_E(xz, fz, species, G)
        res["B1z_raw"] = Ez
        res["B1z"] = T42.residualise(pd.Series(Ez, index=res.index), c3)
        cfg_run = json.loads((run_dir / "config.json").read_text())
        if (run_dir / "model").exists():
            import torch

            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
            res["S_fwd"] = forward_margin(run_dir, a, cfg_run)

    rng = np.random.default_rng(RU.useed("random", run))
    res["random"] = rng.random(G)

    # health and mixing
    if st is None:
        h = dict(run=run, cohort=cohort, finite=False, active_dims=0, discard=True, k_shared=len(groups),
                 note="no finite final state: Chimera columns NaN, direct readouts scored")
    else:
        vz = z.var(0)
        active = int((vz >= RU.ACTIVE_VAR).sum())
        dr = res["D_raw"]
        h = dict(run=run, cohort=cohort, finite=True, active_dims=active, discard=active < RU.N_LATENT,
                 tail_pos=float(((res["alpha"] - res["prior_raw"]) > 0.05).mean()), tail_neg=float(((res["alpha"] - res["prior_raw"]) < -0.05).mean()),
                 sat_hi=float((st["alpha"] > 0.95).mean()), sat_lo=float((st["alpha"] < 0.05).mean()),
                 frac_S_zero=float((np.abs(st["S_g_ema"]) < 1e-6).mean()),
                 rho_D_S=float(pd.Series(res["D"]).corr(pd.Series(res["S_g"]), method="spearman")),
                 lib_median={s_: float(np.median(np.asarray(X[spc == s_].sum(1)).ravel())) for s_ in species},
                 k_shared=len(groups))
        h.update(species_mixing(z, spc))
        try:
            s_ = json.loads((run_dir / "summary.json").read_text())
            h["adv_acc_final"] = s_["key_metrics"].get("adv_acc_final")
            h["n_expect_fail"] = s_["expectation_summary"]["fail"]
            rs = json.loads((run_dir / "run_summary.json").read_text())
            h["elapsed_s"] = rs.get("elapsed_seconds")
        except Exception as exc:  # noqa: BLE001
            h["summary_error"] = str(exc)
    (RU.RES / "health").mkdir(parents=True, exist_ok=True)
    (RU.RES / "health" / f"{run}.json").write_text(json.dumps(h, indent=1))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    res.reset_index().to_parquet(tmp)
    tmp.replace(out)
    print("wrote", out, h, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--h5ad", default=None)
    a = ap.parse_args()
    row = pd.read_csv(a.manifest, sep="\t").iloc[a.index].to_dict()
    score(row, a.h5ad)


if __name__ == "__main__":
    main()
