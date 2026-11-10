"""T11.2b: selection per Van Zyl subset, the seed manifest, and the per-run report (README.md).

Selection uses only the T11.2 functions, unchanged: Gate G-V (``select_recipe.gate``) and
the integration composite (``compute_integration_metrics``, k=15, seed 42, labels
``cell_type_harmonised``; direction-corrected, min-max over the subset's passing runs,
0.4 mixing + 0.6 biology). Ties go to the lower grid_id. The per-run report (Delta-alpha
sign profile, mixing) runs only after every subset's selection is written.

    python analysis/tasks/T11_2b_vz_subsets/select_subsets.py select [--subsets hmfas,hpig] [--write-seed-manifest]

Grids end in waves (one array, per-user CPU cap), so ``--subsets`` restricts selection to
subsets whose grid rows have all ended; seeds are written per subset (``seeds_<tag>.tsv``,
``seed_map_<tag>.tsv``) and selection results per subset (``selection_<tag>.json``).
    python analysis/tasks/T11_2b_vz_subsets/select_subsets.py profile
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
sys.path.insert(0, str(REPO / "analysis" / "tasks" / "T11_2_recipe"))

from run_profile import FLAG_RATIO, first_nan_snapshot, sign_profile, species_entropy  # noqa: E402
from select_recipe import BATCH_KEYS, BIO_KEYS, LABEL_KEY, RUNS, correct, gate, minmax  # noqa: E402
from training.compare_vs_scvi import compute_integration_metrics  # noqa: E402

TASK = Path(__file__).resolve().parent
OUT = REPO / "results" / "analysis" / "T11_2b"
SEEDS = (43, 44)  # s42 is the selected grid run


def read_obs(args: str):
    import anndata as ad

    t = args.split()
    return ad.read_h5ad(REPO / t[t.index("--adata") + 1], backed="r")


def select(write_seeds: bool, only: list[str] | None) -> int:
    grid = pd.read_csv(TASK / "grid.tsv", sep="\t")
    if only:
        unknown = set(only) - set(grid["subset"])
        if unknown:
            raise SystemExit(f"unknown subsets {sorted(unknown)}")
        grid = grid[grid["subset"].isin(only)]
    OUT.mkdir(parents=True, exist_ok=True)
    result, tabs = {}, []
    for subset, g in grid.groupby("subset", sort=False):
        seed_rows, seed_map = [], []
        a = read_obs(g["args"].iloc[0])
        species = a.obs["species"].astype(str).to_numpy()
        labels = a.obs[LABEL_KEY].astype(str).to_numpy()
        rows = []
        for _, r in g.iterrows():
            d = RUNS / r.run_name
            row = {"subset": subset, "grid_id": int(r.grid_id), "run_name": r.run_name,
                   "source_row": r.source_row, **gate(d)}
            if row["finished"] and row["no_nan"]:  # a NaN run fails G-V and is not scored
                z = np.asarray(np.load(d / "final_state.npz")["z_mean"])
                if z.shape[0] != a.n_obs:
                    raise RuntimeError(f"{r.run_name}: z rows {z.shape[0]} != {a.n_obs}")
                m = compute_integration_metrics(z, species, labels, k=15, seed=42)["global"]
                row.update({k: m[k] for k in BATCH_KEYS + BIO_KEYS})
            rows.append(row)
        tab = pd.DataFrame(rows)
        tab["composite"] = np.nan
        ok = tab[tab["pass_GV"].astype(bool)]
        sel = None
        if len(ok):
            corr = pd.DataFrame({k: correct(k, ok[k].astype(float)) for k in BATCH_KEYS + BIO_KEYS})
            norm = corr.apply(minmax)
            comp = 0.4 * norm[BATCH_KEYS].mean(axis=1) + 0.6 * norm[BIO_KEYS].mean(axis=1)
            tab.loc[ok.index, "composite"] = comp
            best = tab.loc[comp.index[comp >= comp.max() - 1e-9]].sort_values("grid_id").iloc[0]
            sel = {"grid_id": int(best.grid_id), "run_name": best.run_name, "source_row": best.source_row}
            base = g.loc[g.grid_id == sel["grid_id"], "args"].iloc[0]
            seed_rows += [f"rev-vz-{subset}-s{s}\t{base.replace('--seed 42', f'--seed {s}')}\n" for s in SEEDS]
            seed_map += [f"rev-vz-{subset}-s42\t42\t{sel['run_name']} (selected grid run)\n"] + [
                f"rev-vz-{subset}-s{s}\t{s}\ttraining/runs/rev-vz-{subset}-s{s}\n" for s in SEEDS]
        result[subset] = {"selected": sel, "n_finished": int(tab["finished"].sum()), "n_rows": len(tab)}
        tabs.append(tab)
        tab.to_csv(OUT / f"selection_{subset}.tsv", sep="\t", index=False)
        (OUT / f"selection_{subset}.json").write_text(json.dumps(result[subset], indent=2))
        if write_seeds and sel is not None:
            (TASK / f"seeds_{subset}.tsv").write_text("".join(["run_name\targs\n"] + seed_rows))
            (TASK / f"seed_map_{subset}.tsv").write_text("".join(["run_name\tseed\tsource\n"] + seed_map))
    allt = pd.concat(tabs)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(allt.to_string(index=False))
    print(json.dumps(result, indent=2))
    return 0


def profile(only: list[str] | None) -> int:
    grid = pd.read_csv(TASK / "grid.tsv", sep="\t")
    if only:
        grid = grid[grid["subset"].isin(only)]
    sel = {}
    for subset in grid["subset"].unique():
        f = OUT / f"selection_{subset}.json"
        if not f.exists():
            raise SystemExit(f"no selection for {subset}: the report runs after it")
        sel[subset] = json.loads(f.read_text())
    rows = []
    for subset, g in grid.groupby("subset", sort=False):
        a = read_obs(g["args"].iloc[0])
        species = a.obs["species"].astype(str).to_numpy()
        others = sorted(set(species) - {"human"})
        labels = a.obs[LABEL_KEY].astype(str).to_numpy()
        prior_var = np.asarray(a.var["conservation_prior"], dtype=float)
        chosen = (sel[subset]["selected"] or {}).get("run_name")
        for name in list(g.run_name) + [f"rev-vz-{subset}-s{s}" for s in SEEDS]:
            d = RUNS / name
            if not (d / "summary.json").exists():
                continue
            fs = np.load(d / "final_state.npz", allow_pickle=True)
            alpha, prior, z = (np.asarray(fs[k], dtype=float) for k in ("alpha", "prior", "z_mean"))
            raw_f = d / "conservation_prior_raw.npy"
            raw = np.load(raw_f).astype(float) if raw_f.exists() else prior_var
            fin = bool(np.isfinite(alpha).all() and np.isfinite(z).all())
            m = compute_integration_metrics(z, species, labels, k=15, seed=42)["global"] if fin else {}
            summ = json.loads((d / "summary.json").read_text())
            rows.append({
                "subset": subset, "run_name": name, "selected": name == chosen, "pass_GV": gate(d)["pass_GV"],
                "first_nan_snapshot": first_nan_snapshot(d),
                **sign_profile(alpha - prior, "train"), **sign_profile(alpha - raw, "raw"),
                "knn_mixing_entropy": m.get("knn_mixing_entropy", np.nan),
                **{f"mixing_entropy_{o}": (species_entropy(z, species, o) if fin else np.nan) for o in others},
                "adv_acc_final": summ["key_metrics"].get("adv_acc_final"),
            })
    tab = pd.DataFrame(rows)
    tab["flag_poorly_mixed"] = False
    for subset, t in tab.groupby("subset"):
        ref = t.loc[t["selected"], "knn_mixing_entropy"]
        seeds = t.index[t["run_name"].str.fullmatch(rf"rev-vz-{subset}-s\d+")]
        if len(ref):
            tab.loc[seeds, "flag_poorly_mixed"] = tab.loc[seeds, "knn_mixing_entropy"] < FLAG_RATIO * float(ref.iloc[0])
    tab.to_csv(OUT / f"run_profile_{'_'.join(sorted(grid['subset'].unique()))}.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 300, "display.max_columns", 50):
        print(tab.to_string(index=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["select", "profile"])
    ap.add_argument("--write-seed-manifest", action="store_true")
    ap.add_argument("--subsets", default=None, help="comma-separated subset tags (default: all)")
    a = ap.parse_args()
    only = a.subsets.split(",") if a.subsets else None
    return select(a.write_seed_manifest, only) if a.mode == "select" else profile(only)


if __name__ == "__main__":
    sys.exit(main())
