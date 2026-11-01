"""T11.2: apply the pre-committed selection rule to the Van Zyl recipe grid (see README.md).

1. Gate G-V on each finished grid run (s42): no NaN; saturation <= 5%; 10/10 active
   z dims; both tails present.
2. Among runs that pass, the integration composite from
   ``training/compare_vs_scvi.py::compute_integration_metrics`` (k=15, seed=42) on
   ``z_mean`` with ``obs['cell_type_harmonised']``: six metrics direction-corrected and
   min-max normalised across the passing runs, 0.4 * mixing + 0.6 * biology (the
   formula of notebooks/07_baselines_master.ipynb). Highest composite wins; ties
   (|diff| < 1e-9) go to the lower grid_id.

No Delta-alpha statistic beyond the gate's tail check, no glaucoma gene and no biology
readout is read here.

    python revision/tasks/T11_2_recipe/select.py [--write-t113-manifest]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from training.compare_vs_scvi import compute_integration_metrics  # noqa: E402

TASK = Path(__file__).resolve().parent
RUNS = REPO / "training" / "runs"
OUT = REPO / "results" / "revision" / "T11_2"
T113 = REPO / "revision" / "tasks" / "T11_3_vz_seeds"
ADATA = REPO / "data" / "revision" / "van_zyl" / "van_zyl_unified.h5ad"
LABEL_KEY = "cell_type_harmonised"
SEEDS_T113 = (43, 44, 45, 46)

BATCH_KEYS = ["knn_species_purity", "knn_mixing_entropy", "sil_species"]
BIO_KEYS = ["knn_celltype_purity", "sil_celltype", "label_transfer_acc"]

# Gate G-V thresholds (fixed before training).
SAT_MAX = 0.05          # each alpha tail: frac(alpha > 0.95) and frac(alpha < 0.05)
ZVAR_MIN = 0.01         # a z dim is active if Var over cells of z_mean >= 0.01 (E6.3 weak-dim bar)
N_ACTIVE = 10
TAIL_DELTA = 0.05       # both tails: frac(alpha - prior > +0.05) and frac(< -0.05) ...
TAIL_MIN = 0.02         # ... each >= 2% (E9.2 criterion)


def gate(run_dir: Path) -> dict:
    r: dict = {"finished": (run_dir / "summary.json").exists() and (run_dir / "final_state.npz").exists()}
    if not r["finished"]:
        r["pass_GV"] = False
        return r
    summ = json.loads((run_dir / "summary.json").read_text())
    fs = np.load(run_dir / "final_state.npz", allow_pickle=True)
    alpha = np.asarray(fs["alpha"], dtype=np.float64)
    prior = np.asarray(fs["prior"], dtype=np.float64)
    z = np.asarray(fs["z_mean"], dtype=np.float64)
    finite = bool(np.isfinite(alpha).all() and np.isfinite(prior).all() and np.isfinite(z).all())
    r["no_nan"] = finite and "E2.1" not in summ.get("failed_expectations", [])
    r["sat_high"] = float((alpha > 0.95).mean())
    r["sat_low"] = float((alpha < 0.05).mean())
    r["saturation_ok"] = r["sat_high"] <= SAT_MAX and r["sat_low"] <= SAT_MAX
    zvar = z.var(axis=0)
    r["n_active_z"] = int((zvar >= ZVAR_MIN).sum())
    r["z_var_min"] = float(zvar.min())
    r["active_ok"] = r["n_active_z"] == N_ACTIVE and z.shape[1] == N_ACTIVE
    d = alpha - prior
    r["tail_up"] = float((d > TAIL_DELTA).mean())
    r["tail_dn"] = float((d < -TAIL_DELTA).mean())
    r["tails_ok"] = r["tail_up"] >= TAIL_MIN and r["tail_dn"] >= TAIL_MIN
    r["pass_GV"] = bool(r["no_nan"] and r["saturation_ok"] and r["active_ok"] and r["tails_ok"])
    metrics = sorted(run_dir.glob("lightning/**/metrics.csv"))
    if metrics:
        ep = pd.read_csv(metrics[-1], usecols=lambda c: c == "epoch")["epoch"]
        r["last_epoch"] = int(ep.max()) + 1 if len(ep.dropna()) else None
    rs = run_dir / "run_summary.json"
    if rs.exists():
        r["elapsed_h"] = round(json.loads(rs.read_text()).get("elapsed_seconds", float("nan")) / 3600, 2)
    return r


def correct(metric: str, s: pd.Series) -> pd.Series:
    if metric == "knn_species_purity":
        return 1.0 - s
    if metric == "sil_species":
        return 1.0 - s.clip(lower=0.0)
    if metric == "sil_celltype":
        return (s + 1.0) / 2.0
    return s


def minmax(s: pd.Series) -> pd.Series:
    mn, mx = s.min(), s.max()
    if not np.isfinite(mn) or not np.isfinite(mx) or mx == mn:
        return s.where(s.isna(), 0.5)
    return (s - mn) / (mx - mn)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-t113-manifest", action="store_true")
    args = ap.parse_args()
    import anndata as ad

    grid = pd.read_csv(TASK / "grid.tsv", sep="\t")
    obs = ad.read_h5ad(ADATA, backed="r").obs
    species = obs["species"].astype(str).to_numpy()
    labels = obs[LABEL_KEY].astype(str).to_numpy()
    OUT.mkdir(parents=True, exist_ok=True)

    rows, raw_metrics = [], {}
    for _, g in grid.iterrows():
        run_dir = RUNS / g.run_name
        r = {"grid_id": int(g.grid_id), "run_name": g.run_name, **gate(run_dir)}
        if r["finished"]:
            z =np.asarray(np.load(run_dir / "final_state.npz")["z_mean"])
            if z.shape[0] != len(obs):
                raise RuntimeError(f"{g.run_name}: z has {z.shape[0]} rows, adata {len(obs)}")
            m = compute_integration_metrics(z, species, labels, k=15, seed=42)
            raw_metrics[g.run_name] = m
            r.update({k: m["global"][k] for k in BATCH_KEYS + BIO_KEYS})
            r["n_shared_celltypes"] = m["global"]["n_shared_celltypes"]
        rows.append(r)
    tab = pd.DataFrame(rows)

    passing = tab[tab["pass_GV"]].copy()
    tab["mixing"] = np.nan
    tab["biology"] = np.nan
    tab["composite"] = np.nan
    tab["composite_unnormalised"] = np.nan
    if len(passing):
        corr = pd.DataFrame({k: correct(k, passing[k].astype(float)) for k in BATCH_KEYS + BIO_KEYS})
        norm = corr.apply(minmax)
        passing["mixing"] = norm[BATCH_KEYS].mean(axis=1)
        passing["biology"] = norm[BIO_KEYS].mean(axis=1)
        passing["composite"] = 0.4 * passing["mixing"] + 0.6 * passing["biology"]
        passing["composite_unnormalised"] = 0.4 * corr[BATCH_KEYS].mean(axis=1) + 0.6 * corr[BIO_KEYS].mean(axis=1)
        for c in ("mixing", "biology", "composite", "composite_unnormalised"):
            tab.loc[passing.index, c] = passing[c]
        best = passing["composite"].max()
        sel = passing[passing["composite"] >= best - 1e-9].sort_values("grid_id").iloc[0]
        selected = {"grid_id": int(sel.grid_id), "run_name": sel.run_name, "composite": float(sel.composite)}
    else:
        selected = None
    tab.to_csv(OUT / "selection.tsv", sep="\t", index=False)
    (OUT / "selection.json").write_text(json.dumps(
        {"rule": "README.md", "label_key": LABEL_KEY, "selected": selected,
         "n_finished": int(tab["finished"].sum()), "n_pass_GV": int(tab["pass_GV"].sum()),
         "integration_metrics": raw_metrics}, indent=2, default=str))
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(tab.drop(columns=[c for c in ("finished",) if c in tab]).to_string(index=False))
    print("selected:", selected)

    if args.write_t113_manifest:
        if selected is None:
            print("no configuration passed G-V: P11 training stops here (README rule)")
            return 2
        base = grid.loc[grid.grid_id == selected["grid_id"], "args"].iloc[0]
        T113.mkdir(parents=True, exist_ok=True)
        with open(T113 / "manifest.tsv", "w") as f:
            f.write("run_name\targs\n")
            for s in SEEDS_T113:
                f.write(f"rev-vz-s{s}\t{base.replace('--seed 42', f'--seed {s}')}\n")
        (T113 / "seed_map.tsv").write_text(
            "run_name\tseed\tsource\n"
            f"rev-vz-s42\t42\t{selected['run_name']} (selected grid run, grid_id {selected['grid_id']})\n"
            + "".join(f"rev-vz-s{s}\t{s}\ttraining/runs/rev-vz-s{s}\n" for s in SEEDS_T113))
        print("wrote", T113 / "manifest.tsv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
