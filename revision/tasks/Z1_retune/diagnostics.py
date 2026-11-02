"""Z1 training diagnostics: count dependence and health of a Chimera run. Blind: uses only the
run's own state and the training counts.

Per run, on genes with count support (|S_g_ema| >= 1e-6, the X4 definition of unmasked):
* rho_DS: Spearman(alpha - prior, S_g_ema), prior as the run anchored on it;
* rho_DS_partial: same, partial on expression (Pearson of rank residuals on a natural spline of the
  rank of the species-balanced mean log1p CP10K);
* rho_cS: Spearman(correction, S_g_ema) (logit scale, free of the prior's slope);
* rho_DSbar: rho_DS with S_g_ema averaged over the snapshots of the last 25% of the run, which removes
  the noise of the 10-batch EMA in the final value.
Also: fraction with count support, saturation, tails on the raw prior and on the training prior, active
z dims, final adversary accuracy, kNN mixing entropy (k = 15), last epoch, epoch the encoder learning
rate fell below 1e-5, peak learned posterior variance.

    python revision/tasks/Z1_retune/diagnostics.py --runs-tsv <tsv: cohort, run> --out <tsv>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from revision.common.controls import ns_basis  # noqa: E402

RUNS = REPO / "training" / "runs"
MASK = 1e-6
TAIL = 0.05
TAIL_MIN = 0.02
SAT_MAX = 0.10
ZVAR_MIN = 0.01
_cache: dict = {}


def cohort_data(adata_path: str) -> dict:
    """Species labels and species-balanced mean log1p CP10K per gene."""
    if adata_path in _cache:
        return _cache[adata_path]
    import anndata as ad
    import scipy.sparse as sp

    p = Path(adata_path)
    p = p if p.is_absolute() else REPO / p
    if not p.exists():  # runs moved from the original machine
        p = REPO / "data" / "processed" / p.name
    a = ad.read_h5ad(p)
    species = a.obs["species"].astype(str).to_numpy()
    X = a.X.tocsr().astype(np.float64) if sp.issparse(a.X) else sp.csr_matrix(a.X, dtype=np.float64)
    lib = np.asarray(X.sum(1)).ravel()
    cp = sp.diags(1e4 / np.maximum(lib, 1)).dot(X).tocsr()
    cp.data = np.log1p(cp.data)
    expr = np.mean([np.asarray(cp[species == s].mean(0)).ravel() for s in np.unique(species)], axis=0)
    med = {s: float(np.median(lib[species == s])) for s in np.unique(species)}
    _cache[adata_path] = dict(species=species, expr=expr, median_lib=med, n_cells=a.n_obs, n_genes=a.n_vars)
    return _cache[adata_path]


def mixing_entropy(z: np.ndarray, species: np.ndarray, k: int = 15) -> float:
    from sklearn.neighbors import NearestNeighbors

    codes, names = pd.factorize(species, sort=True)
    idx = NearestNeighbors(n_neighbors=k + 1).fit(z).kneighbors(z)[1][:, 1:]
    p = np.stack([(codes[idx] == c).mean(axis=1) for c in range(len(names))], axis=1)
    h = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=1) / np.log(len(names))
    return float(h.mean())


def curves(d: Path) -> dict:
    out: dict = {}
    fs = sorted(d.glob("lightning/**/metrics.csv"))
    if not fs:
        return out
    m = pd.read_csv(fs[-1])
    out["last_epoch"] = int(m["epoch"].max()) + 1
    if "lr/encoder" in m:
        lr = m[["epoch", "lr/encoder"]].dropna()
        below = lr[(lr["lr/encoder"] < 1e-5) & (lr["epoch"] >= 5)]
        out["lr_freeze_epoch"] = int(below["epoch"].iloc[0]) if len(below) else np.nan
        for e in (40, 80):
            s = lr[lr["epoch"] == e]["lr/encoder"]
            out[f"lr_ep{e}"] = float(s.iloc[0]) if len(s) else np.nan
    if "z/mean_learned_var" in m:
        v = m["z/mean_learned_var"].dropna()
        out["max_learned_var"] = float(v[np.isfinite(v)].max()) if np.isfinite(v).any() else np.nan
    for c, k in (("adversarial_accuracy", "adv_acc_last"),):
        if c in m and m[c].notna().any():
            out[k] = float(m[c].dropna().iloc[-1])
    return out


def sbar(d: Path, last_epoch: int, frac: float = 0.25) -> np.ndarray | None:
    files = sorted((d / "alpha_snapshots").glob("alpha_epoch_*.npz"))
    use = [f for f in files if int(f.stem.split("_")[-1]) >= (1 - frac) * last_epoch]
    if not use:
        return None
    return np.mean([np.load(f)["S_g_ema"] for f in use], axis=0).astype(float)


def one_run(run: str, with_mixing: bool = True) -> dict:
    d = RUNS / run
    r: dict = {"run": run, "finished": (d / "final_state.npz").exists()}
    if not r["finished"]:
        return r
    cfg = json.loads((d / "config.json").read_text())
    data = cohort_data(cfg["adata_path"])
    r["n_cells"], r["n_genes"] = data["n_cells"], data["n_genes"]
    fs = np.load(d / "final_state.npz", allow_pickle=True)
    alpha, prior = np.asarray(fs["alpha"], float), np.asarray(fs["prior"], float)
    S = np.asarray(fs["S_g_ema"], float)
    corr = np.asarray(fs["correction"], float)
    z = np.asarray(fs["z_mean"], float)
    r["no_nan"] = bool(np.isfinite(alpha).all() and np.isfinite(z).all() and np.isfinite(S).all())
    r.update(curves(d))
    if not r["no_nan"]:
        r["healthy"] = False
        return r
    fraw = d / "conservation_prior_raw.npy"
    raw = np.load(fraw).astype(float) if fraw.exists() else prior
    D = alpha - prior
    Draw = alpha - raw
    r["sat_hi"], r["sat_lo"] = float((alpha > 0.95).mean()), float((alpha < 0.05).mean())
    r["tail_up_raw"], r["tail_dn_raw"] = float((Draw > TAIL).mean()), float((Draw < -TAIL).mean())
    r["tail_up_tr"], r["tail_dn_tr"] = float((D > TAIL).mean()), float((D < -TAIL).mean())
    r["sd_D"] = float(D.std())
    zv = z.var(axis=0)
    r["n_active_z"], r["z_var_min"] = int((zv >= ZVAR_MIN).sum()), float(zv.min())
    r["healthy"] = bool(r["n_active_z"] == z.shape[1] and r["tail_up_raw"] >= TAIL_MIN and r["tail_dn_raw"] >= TAIL_MIN
                        and r["sat_hi"] <= SAT_MAX and r["sat_lo"] <= SAT_MAX)
    un = np.abs(S) >= MASK
    r["frac_support"] = float(un.mean())
    r["frac_S_pos"] = float((S[un] > 0).mean())
    r["rho_DS"] = float(spearmanr(D[un], S[un])[0])
    r["rho_cS"] = float(spearmanr(corr[un], S[un])[0])
    Z = np.c_[np.ones(un.sum()), ns_basis(pd.Series(data["expr"][un]).rank().to_numpy(), 4)]

    def res(y):
        ry = pd.Series(y).rank().to_numpy()
        return ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]

    r["rho_DS_partial"] = float(np.corrcoef(res(D[un]), res(S[un]))[0, 1])
    r["rho_D_expr"] = float(spearmanr(D[un], data["expr"][un])[0])
    r["rho_S_expr"] = float(spearmanr(S[un], data["expr"][un])[0])
    Sb = sbar(d, r.get("last_epoch", 150))
    if Sb is not None:
        r["rho_DSbar"] = float(spearmanr(D[un], Sb[un])[0])
    if with_mixing:
        r["mix_entropy"] = mixing_entropy(z, data["species"])
    summ = d / "summary.json"
    if summ.exists():
        r["adv_acc"] = json.loads(summ.read_text())["key_metrics"].get("adv_acc_final")
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-tsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-mixing", action="store_true")
    a = ap.parse_args()
    t = pd.read_csv(a.runs_tsv, sep="\t")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = pd.read_csv(out, sep="\t") if out.exists() else pd.DataFrame(columns=["run"])
    rows = done.to_dict("records")
    have = set(done["run"]) if len(done) else set()
    for _, x in t.iterrows():
        if x["run"] in have:
            continue
        print(x["run"], flush=True)
        rows.append({"cohort": x["cohort"], **one_run(x["run"], not a.no_mixing)})
        pd.DataFrame(rows).to_csv(out, sep="\t", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
