"""Per-gene readouts of one D1 fit on its own object; reads no gold. Depth rows: B2 score_run's definitions (S_g, S_fwd, D, D_raw, B1z, B1, B1c, B2b, B3a, Sdir, Lfc), with Lfc as the mean over
species pairs, the Track C form, so that more than two species are scored. Held-out rows: alpha, S_g, D, D_raw, correction; arm E also D_E0 (held-out genes forward through the fitted gate with
expression columns at their mean).
    python -m revision.tasks.D1_sparse_genes.score_fit --index I --h5ad path
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SLURM_CPUS_PER_TASK", "4"))

import numpy as np
import pandas as pd

from revision.common import cohorts as CO
from revision.tasks.B2_benchmark import perturb as P
from revision.tasks.B2_benchmark import score_run as SR
from revision.tasks.D1_sparse_genes import rules as RU
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.X6_adds_beyond.degrade import sdir


def gate_forward(run_dir, F: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """alpha from the fitted correction MLP on feature matrix F."""
    import torch

    sd = torch.load(run_dir / "model" / "model.pt", map_location="cpu", weights_only=False)["model_state_dict"]
    idx = sorted({int(k.split(".")[2]) for k in sd if k.startswith("gin.correction_mlp.") and k.endswith("weight")})
    x = torch.tensor(F, dtype=torch.float32)
    with torch.no_grad():
        for n, i in enumerate(idx):
            x = x @ sd[f"gin.correction_mlp.{i}.weight"].T + sd[f"gin.correction_mlp.{i}.bias"]
            if n < len(idx) - 1:
                x = torch.tanh(x)
    c = x.squeeze(-1).numpy()
    return 1 / (1 + np.exp(-(np.log(prior / (1 - prior)) + c)))


def chimera_part(res, st, a, X, spc, library, run, cohort, setting, direct: bool):
    species = sorted(set(spc))
    run_dir = CO.RUNS_DIR / run
    pr, al = st["prior"].astype(float), st["alpha"].astype(float)
    res["alpha"], res["prior"] = al, pr
    res["S_g"] = st["S_g_ema"].astype(float)
    da, dr = res["alpha"] - res["prior"], res["alpha"] - res["prior_raw"]
    res["D"], res["D_raw"] = da - da.median(), dr - dr.median()
    res["correction"] = st["correction"].astype(float)
    z = np.asarray(st["z_mean"], dtype=np.float32)
    cfg_run = json.loads((run_dir / "config.json").read_text())
    if direct:
        cl = T42.leiden_clusters(z)
        xz, fz, _, _ = SR.pseudobulk_fb(X, spc, cl, library, RU.B2.MIN_CELLS_CLUSTER, "all")
        res["B1z"] = T42.pooled_E(xz, fz, species, X.shape[1])[0]
        if (run_dir / "model").exists():
            import torch

            torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
            res["S_fwd"] = SR.forward_margin(run_dir, a, cfg_run)
    vz = z.var(0)
    fin = bool(np.isfinite(st["alpha"]).all() and np.isfinite(z).all() and np.isfinite(st["S_g_ema"]).all())
    active = int((vz >= RU.B2.ACTIVE_VAR).sum())
    h = dict(run=run, cohort=cohort, setting=setting, finite=fin, active_dims=active, discard=not (fin and active >= RU.B2.N_LATENT),
             tail_pos=float((dr > 0.05).mean()), tail_neg=float((dr < -0.05).mean()), sat_hi=float((st["alpha"] > 0.95).mean()),
             sat_lo=float((st["alpha"] < 0.05).mean()), frac_S_zero=float((np.abs(st["S_g_ema"]) < 1e-6).mean()),
             rho_D_S=float(res["D"].corr(res["S_g"], method="spearman")),
             lib_median={s: float(np.median(np.asarray(X[spc == s].sum(1)).ravel())) for s in species}, n_cells=int(a.n_obs))
    if fin:
        h.update(SR.species_mixing(z, spc))
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


def score(row: dict, h5ad: str) -> None:
    run, cohort, setting = row["run"], row["cohort"], row["setting"]
    out = RU.RES / "scores" / f"{run}.parquet"
    cfg_c = RU.B2.COHORTS[cohort]
    import anndata as ad

    a = ad.read_h5ad(h5ad)
    X = P.csr(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    res = pd.DataFrame(index=pd.Index(a.var_names.astype(str), name="var_name"))
    res["hsapiens_gene_id"] = np.asarray(CO.gene_ids.ensembl(cfg_c["gene_table"], res.index))
    res["prior_raw"] = np.asarray(a.var["conservation_prior"], dtype=float)
    st = CO.load_state(run)
    G, species = a.n_vars, sorted(set(spc))
    if row["family"] == "heldout":
        res["held_out"] = a.var["d1_held_out"].to_numpy(dtype=bool)
        library = a.obs[cfg_c["library_key"]].astype(str).to_numpy()
        chimera_part(res, st, a, X, spc, library, run, cohort, setting, direct=False)
        if row["arm"] == "E":
            F = np.array(a.varm["gene_features"], dtype=np.float32)
            F[np.ix_(res["held_out"].to_numpy(), [66, 67, 68])] = 0.0
            al0 = gate_forward(CO.RUNS_DIR / run, F, res["prior"].to_numpy())
            d0 = al0 - res["prior"].to_numpy()
            res["D_E0"] = d0 - np.median(d0)
            chk = gate_forward(CO.RUNS_DIR / run, np.asarray(a.varm["gene_features"], dtype=np.float32), res["prior"].to_numpy())
            assert np.nanmax(np.abs(chk - res["alpha"].to_numpy())) < 1e-3, "gate forward does not reproduce alpha"
    else:
        lab = a.obs["b2_label"].astype(str).to_numpy()
        library = a.obs["b2_library"].astype(str).to_numpy()
        a2 = a.copy()
        a2.obs["cell_type"] = lab
        cov = CO.gene_covariates(a2)
        cov["prior_raw"] = res["prior_raw"].to_numpy()
        del a2
        res["expr"] = cov["log_cp10k_spmean"].to_numpy()
        res["detection_spmean"] = cov["detection_spmean"].to_numpy()
        res["ppass"] = cov["ppass"].to_numpy()
        allowed, min_cells = SR.full_groups(cohort), RU.B2.MIN_CELLS_LABEL
        x, f, units, rescued = SR.pseudobulk_fb(X, spc, lab, library, min_cells, allowed)
        E, _, pinfo = T42.pooled_E(x, f, species, G)
        Lv, _, B2b, _, groups = T42.level_and_specificity(x, f, species, G)
        c3 = cov[["prior_raw", "log_cp10k_spmean", "detection_spmean"]]
        res["B1"] = E
        res["B1b"] = T42.residualise(pd.Series(E, index=res.index), c3)
        res["B1c_raw"] = Lv
        res["B1c"] = T42.residualise(pd.Series(Lv, index=res.index), c3)
        res["B2b"] = B2b
        res["Sdir"] = sdir(X, spc, lab)
        ps = [(np.asarray(X[spc == s].sum(0)).ravel() + 0.5) / (X[spc == s].sum() + 0.5 * G) for s in species]
        lf = [np.abs(np.log2(ps[i] / ps[j])) for i in range(len(ps)) for j in range(i + 1, len(ps))]
        res["Lfc"] = -np.mean(lf, 0)
        F = np.asarray(a.varm["gene_features"], dtype=float)
        res["B3a"] = T42.cross_fit_ridge(F, np.asarray(E, dtype=float))[0] if np.isfinite(E).sum() >= 50 else np.nan
        chimera_part(res, st, a, X, spc, library, run, cohort, setting, direct=True)
        meta = dict(run=run, K_all=len(groups), units_used=int(units.used.sum()), rescued=rescued, n_cells=int(a.n_obs),
                    n_defined={k: int(res[k].notna().sum()) for k in ("B1", "B1c", "B3a", "Sdir")})
        (RU.RES / "meta").mkdir(parents=True, exist_ok=True)
        (RU.RES / "meta" / f"{run}.json").write_text(json.dumps(meta, indent=1, default=str))
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.parquet")
    res.reset_index().to_parquet(tmp)
    tmp.replace(out)
    print("wrote", out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--h5ad", required=True)
    a = ap.parse_args()
    RU.register()
    row = pd.read_csv(RU.TASK / "grid.tsv", sep="\t", keep_default_na=False).iloc[a.index].to_dict()
    score(row, a.h5ad)


if __name__ == "__main__":
    main()
