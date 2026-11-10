"""Track B: every gene-level readout of one grid row, on that row's own object. Writes per-gene scores; reads no gold.

Chimera (when the row has a fit): S_g (S_g_ema), S_fwd (forward-pass margin at the posterior mean, no 20-cell mask, exploratory, as Track A),
D (alpha - prior minus the median), D_raw, B1z (E on Leiden clusters of the fit's z).
Direct, the code of T4_2_gene_baselines, X4_where_differs and X6_adds_beyond on the object's counts and b2_label: B1 (E, raw), B1b, B1c (residualized
as T4_2_gene_baselines), B1c_raw, B2b (any K >= 2, flagged in the metadata when K < 4; X6_adds_beyond), B3a (5-fold ridge on the 72 features as moved
by perturb.py), Sdir (labeled direct margin), Lfc (minus absolute log2 ratio of species pseudobulk rates).
Pooled fallback, only where cells were removed or split (families composition, sampling, technical; no-label settings; B1z): a group that exists in
the unperturbed data (or any cluster or matched pair) with >= the minimum cells in total but no library reaching it is pooled over its libraries
instead of vanishing. This cannot change the full-data values and stops the per-library rule from failing the baselines by itself.
Static scores join later. All scores are oriented so that higher means more conserved.
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SLURM_CPUS_PER_TASK", "4"))

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import cohorts as CO
from analysis.tasks.B2_benchmark import perturb as P
from analysis.tasks.B2_benchmark import rules as RU
from analysis.tasks.T4_2_gene_baselines import run as T42
from analysis.tasks.X6_adds_beyond.degrade import sdir, specificity_cos


def full_groups(cohort: str) -> set:
    """(species, type) groups with a qualifying library in the unperturbed data."""
    cfg = RU.COHORTS[cohort]
    a = P.read_base(cohort)
    lab = P.base_labels(a, cohort)
    t = pd.DataFrame(dict(s=a.obs["species"].astype(str).to_numpy(), t=lab, l=a.obs[cfg["library_key"]].astype(str).to_numpy()))
    n = t.value_counts().reset_index(name="n")
    n = n[n.n >= RU.MIN_CELLS_LABEL]
    return set(zip(n.s, n.t))


def pseudobulk_fb(X, spc, lab, library, min_cells: int, allowed):
    """T4_2_gene_baselines pseudobulk, then the pooled fallback. allowed: None = no fallback; 'all' = any group; a set = those groups."""
    x, f, units = T42.pseudobulk(X, spc, lab, library, min_cells)
    rescued = []
    if allowed is not None:
        g = pd.DataFrame(dict(s=spc, t=lab)).value_counts().reset_index(name="n")
        for s, t, n in g.itertuples(index=False):
            if n < min_cells or (s, t) in x or (allowed != "all" and (s, t) not in allowed):
                continue
            m = (spc == s) & (lab == t)
            cnt = np.asarray(X[m].sum(0)).ravel()
            x[(s, t)] = np.log2(1.0 + 1e6 * cnt / cnt.sum())
            f[(s, t)] = np.asarray((X[m] > 0).sum(0)).ravel() / m.sum()
            rescued.append(f"{s}:{t}")
    return x, f, units, rescued


def forward_margin(run_dir, a, cfg: dict, batch_size: int = 512) -> np.ndarray:
    """S_fwd: species-averaged mean over all cells of the model's log-likelihood ratio at the posterior mean. No mask."""
    import torch

    from analysis.tasks.T2_5_celltype_margin.run import load_components, nb_llr

    comp = load_components(run_dir, a, cfg)
    X = P.csr(a.X)
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


def score(row: dict, h5ad: str | None) -> None:
    run, cohort, setting = row["run"], row["cohort"], row["setting"]
    out = RU.RES / "scores" / f"{run}.parquet"
    if out.exists():
        print("exists", out, flush=True)
        return
    cfg_c = RU.COHORTS[cohort]
    import anndata as ad

    a = ad.read_h5ad(h5ad or (RU.REPO / cfg_c["h5ad"]))
    X = P.csr(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    if "b2_label" in a.obs:
        lab = a.obs["b2_label"].astype(str).to_numpy()
        library = a.obs["b2_library"].astype(str).to_numpy()
    else:
        lab = P.base_labels(a, cohort)
        library = a.obs[cfg_c["library_key"]].astype(str).to_numpy()
    G = a.n_vars
    species = sorted(set(spc))
    a2 = a.copy()
    a2.obs["cell_type"] = lab
    cov = CO.gene_covariates(a2)
    cov["prior_raw"] = np.asarray(a.var["conservation_prior"], dtype=float)
    del a2

    res = pd.DataFrame(index=pd.Index(a.var_names.astype(str), name="var_name"))
    res["hsapiens_gene_id"] = np.asarray(CO.gene_ids.ensembl(cfg_c["gene_table"], res.index))
    res["prior_raw"] = cov["prior_raw"].to_numpy()
    res["expr"] = cov["log_cp10k_spmean"].to_numpy()
    res["detection_spmean"] = cov["detection_spmean"].to_numpy()
    res["ppass"] = cov["ppass"].to_numpy()
    family = RU.SETTINGS[setting]["family"] if setting in RU.SETTINGS else "reference"
    kind = RU.SETTINGS[setting]["kind"] if setting in RU.SETTINGS else "none"
    if kind == "nolabel":
        allowed, min_cells = "all", RU.MIN_CELLS_CLUSTER
    elif family in ("composition", "sampling", "technical"):
        allowed, min_cells = full_groups(cohort), RU.MIN_CELLS_LABEL
    else:
        allowed, min_cells = None, RU.MIN_CELLS_LABEL
    x, f, units, rescued = pseudobulk_fb(X, spc, lab, library, min_cells, allowed)
    E, _, pinfo = T42.pooled_E(x, f, species, G)
    Lv, _, B2, _, groups = T42.level_and_specificity(x, f, species, G)
    if len(groups) < 4 and len(groups) >= 2:
        B2 = specificity_cos(x, f, species, G)
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
    if np.isfinite(E).sum() >= 50:
        res["B3a"] = T42.cross_fit_ridge(F, np.asarray(E, dtype=float))[0]
    else:
        res["B3a"] = np.nan
    rng = np.random.default_rng(RU.useed("random", run))
    res["random"] = rng.random(G)
    meta = dict(run=run, cohort=cohort, setting=setting, rep=int(row["rep"]), K_pair=pinfo[0]["K"] if pinfo else 0, K_all=len(groups),
                B2b_note="K<4, computed as X6" if len(groups) < 4 else "", groups=groups, units_used=int(units.used.sum()),
                units_total=int(len(units)), rescued=rescued, min_cells=min_cells, n_cells=int(a.n_obs),
                n_defined={k: int(res[k].notna().sum()) for k in ("B1", "B1c", "B2b", "B3a", "Sdir")})

    if int(row["fit"]) == 1 or row["setting"] == "ref":
        cr = row["chimera_from"] if row["setting"] == "ref" else run
        run_dir = CO.RUNS_DIR / cr
        st = CO.load_state(cr)
        pr = st["prior"].astype(float)
        al = st["alpha"].astype(float)
        res["alpha"], res["prior"] = al, pr
        res["S_g"] = st["S_g_ema"].astype(float)
        da, dr = res["alpha"] - res["prior"], res["alpha"] - res["prior_raw"]
        res["D"], res["D_raw"] = da - da.median(), dr - dr.median()
        res["correction"] = st["correction"].astype(float)
        z = np.asarray(st["z_mean"], dtype=np.float32)
        if np.isfinite(z).all():
            cl = T42.leiden_clusters(z)
            xz, fz, _, _ = pseudobulk_fb(X, spc, cl, library, RU.MIN_CELLS_CLUSTER, "all")
            Ez, _, _ = T42.pooled_E(xz, fz, species, G)
            res["B1z"] = Ez
        else:
            res["B1z"] = np.nan  # a fit that went non-finite: discarded by the health rule, direct scores of the object still stand
        cfg_run = json.loads((run_dir / "config.json").read_text())
        if (run_dir / "model").exists() and np.isfinite(z).all():
            import torch

            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
            res["S_fwd"] = forward_margin(run_dir, a, cfg_run)
        else:
            res["S_fwd"] = np.nan
        vz = z.var(0)
        fin = bool(np.isfinite(st["alpha"]).all() and np.isfinite(z).all() and np.isfinite(st["S_g_ema"]).all())
        active = int((vz >= RU.ACTIVE_VAR).sum())
        h = dict(run=cr, cohort=cohort, setting=setting, finite=fin, active_dims=active, discard=not (fin and active >= RU.N_LATENT),
                 tail_pos=float((dr > 0.05).mean()), tail_neg=float((dr < -0.05).mean()),
                 sat_hi=float((st["alpha"] > 0.95).mean()), sat_lo=float((st["alpha"] < 0.05).mean()),
                 frac_S_zero=float((np.abs(st["S_g_ema"]) < 1e-6).mean()),
                 rho_D_S=float(pd.Series(res["D"]).corr(pd.Series(res["S_g"]), method="spearman")),
                 lib_median={s: float(np.median(np.asarray(X[spc == s].sum(1)).ravel())) for s in species}, n_cells=int(a.n_obs))
        if np.isfinite(z).all():
            h.update(species_mixing(z, spc))
        try:
            s = json.loads((run_dir / "summary.json").read_text())
            h["adv_acc_final"] = s["key_metrics"].get("adv_acc_final")
            h["n_expect_fail"] = s["expectation_summary"]["fail"]
            h["elapsed_s"] = json.loads((run_dir / "run_summary.json").read_text()).get("elapsed_seconds")
        except Exception as exc:  # noqa: BLE001
            h["summary_error"] = str(exc)
        (RU.RES / "health").mkdir(parents=True, exist_ok=True)
        (RU.RES / "health" / f"{run}.json").write_text(json.dumps(h, indent=1))
        print(h, flush=True)
    (RU.RES / "meta").mkdir(parents=True, exist_ok=True)
    (RU.RES / "meta" / f"{run}.json").write_text(json.dumps(meta, indent=1, default=str))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    res.reset_index().to_parquet(tmp)
    tmp.replace(out)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--h5ad", default=None)
    a = ap.parse_args()
    row = pd.read_csv(a.manifest, sep="\t", keep_default_na=False).iloc[a.index].to_dict()
    score(row, a.h5ad)


if __name__ == "__main__":
    main()
