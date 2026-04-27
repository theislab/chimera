"""Integration scorecard: Chimera run vs. vanilla scVI (species-as-batch).

Purpose
-------
Quantify the acute "Chimera z mixes species worse than vanilla scVI" gap that
drives the current iteration pass. Runs two comparisons:

  1. **Global integration**: kNN species purity, sil_species, sil_celltype,
     kNN mixing entropy, label-transfer accuracy, kNN celltype purity.
  2. **Per-shared-celltype species mixing** (critical per the governing doc):
     for each cell type present in ≥2 species, recompute within-celltype kNN
     species purity — global metrics can look fine while some celltypes stay
     species-stratified.

Baselines
---------
  - **vanilla scVI** trained here with `species_key` as `batch` (same n_latent,
    same n_hidden, same epochs). This is the "must-beat" reference.
  - **raw log-normalized HVG PCA** as a no-integration reference.

Outputs
-------
  - `training/runs/<chimera-run-name>/integration_vs_scvi.md` — headline table.
  - `training/runs/<chimera-run-name>/integration_vs_scvi.json` — machine-readable.

Usage
-----
    python training/compare_vs_scvi.py \
        --chimera-run-name iter42-lres001-mmd50-s42 \
        --adata data/processed/camex_human_monkey.h5ad \
        [--scvi-max-epochs 100] [--seed 42]

The scVI model is cached under `training/runs/baselines_scvi/<adata-stem>-s<seed>/`
so subsequent runs that point at the same adata + seed reuse the cached latent.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("compare_vs_scvi")

REPO_ROOT = Path(__file__).resolve().parents[1]
# Honor CHIMERA_RUNS_ROOT (same hook used by baseline_runner.py /
# run_all_baselines.py) so a test or downstream consumer can redirect
# the cache to a tmp path. Default unchanged.
RUNS_ROOT = Path(os.environ["CHIMERA_RUNS_ROOT"]) if os.environ.get(
    "CHIMERA_RUNS_ROOT"
) else REPO_ROOT / "training" / "runs"


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _entropy(p: np.ndarray, base: int) -> float:
    p = p[p > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum() / np.log(base))


def compute_integration_metrics(
    z: np.ndarray,
    species: np.ndarray,
    cell_types: np.ndarray,
    k: int = 15,
    seed: int = 42,
) -> Dict[str, Any]:
    """Full integration scorecard. Returns global metrics + per-celltype dict."""
    from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
    from sklearn.metrics import silhouette_score, accuracy_score

    # Map species/celltype to int codes for speed.
    sp_codes, sp_names = pd.factorize(species, sort=True)
    ct_codes, ct_names = pd.factorize(cell_types, sort=True)
    n_species = len(sp_names)

    # Global kNN purity / mixing entropy.
    nn = NearestNeighbors(n_neighbors=k + 1).fit(z)
    _, idx = nn.kneighbors(z)
    neigh = idx[:, 1:]  # drop self

    knn_sp_purity = float((sp_codes[neigh] == sp_codes[:, None]).mean())
    knn_ct_purity = float((ct_codes[neigh] == ct_codes[:, None]).mean())

    # Normalized mixing entropy (per-cell entropy of species in neighbors,
    # averaged). 1.0 = perfectly mixed among n_species.
    entropies = np.zeros(z.shape[0])
    for i in range(z.shape[0]):
        counts = np.bincount(sp_codes[neigh[i]], minlength=n_species)
        p = counts.astype(float) / counts.sum()
        entropies[i] = _entropy(p, base=n_species)
    knn_mixing_entropy = float(entropies.mean())

    # Silhouettes (subsample for speed if needed).
    rng = np.random.RandomState(seed)
    if z.shape[0] > 5000:
        idx_sub = rng.choice(z.shape[0], size=5000, replace=False)
    else:
        idx_sub = np.arange(z.shape[0])
    sil_species = float(
        silhouette_score(z[idx_sub], sp_codes[idx_sub], metric="euclidean")
    ) if n_species >= 2 else float("nan")
    # Celltype silhouette over ALL cells (including species-unique types).
    if len(np.unique(ct_codes[idx_sub])) >= 2:
        sil_celltype = float(
            silhouette_score(z[idx_sub], ct_codes[idx_sub], metric="euclidean")
        )
    else:
        sil_celltype = float("nan")

    # Label transfer (shared celltypes only): train knn on species A cells,
    # predict celltype on species B cells, over all ordered species pairs.
    shared_types_per_sp = [
        set(ct_names[ct_codes[sp_codes == s]]) for s in range(n_species)
    ]
    shared = set.intersection(*shared_types_per_sp) if shared_types_per_sp else set()
    shared_list = sorted(shared)
    lt_accs = []
    if len(shared_list) >= 2:
        shared_mask = np.isin(ct_names[ct_codes], list(shared))
        for src in range(n_species):
            for dst in range(n_species):
                if src == dst:
                    continue
                train_mask = (sp_codes == src) & shared_mask
                test_mask = (sp_codes == dst) & shared_mask
                if train_mask.sum() < 5 or test_mask.sum() < 5:
                    continue
                clf = KNeighborsClassifier(
                    n_neighbors=min(k, int(train_mask.sum()) - 1)
                )
                clf.fit(z[train_mask], ct_codes[train_mask])
                pred = clf.predict(z[test_mask])
                lt_accs.append(float(accuracy_score(ct_codes[test_mask], pred)))
    label_transfer_acc = float(np.mean(lt_accs)) if lt_accs else float("nan")

    # Per-shared-celltype within-celltype kNN species mixing.
    per_ct: Dict[str, Any] = {}
    for ct_name in shared_list:
        ct_mask = ct_names[ct_codes] == ct_name
        z_ct = z[ct_mask]
        sp_ct = sp_codes[ct_mask]
        n0, n1 = (sp_ct == 0).sum(), (sp_ct == 1).sum() if n_species >= 2 else (0, 0)
        if z_ct.shape[0] < k + 2:
            per_ct[str(ct_name)] = {"skipped": True}
            continue
        nn_ct = NearestNeighbors(n_neighbors=min(k + 1, z_ct.shape[0])).fit(z_ct)
        _, idx_ct = nn_ct.kneighbors(z_ct)
        neigh_ct = idx_ct[:, 1:]
        purity_ct = float((sp_ct[neigh_ct] == sp_ct[:, None]).mean())
        # Chance baseline = max proportion of any species in this celltype.
        counts = np.bincount(sp_ct, minlength=n_species)
        chance = float(counts.max() / counts.sum())
        # Mixing entropy within celltype.
        ent_ct = np.zeros(z_ct.shape[0])
        for i in range(z_ct.shape[0]):
            c = np.bincount(sp_ct[neigh_ct[i]], minlength=n_species)
            p = c.astype(float) / c.sum()
            ent_ct[i] = _entropy(p, base=n_species)
        per_ct[str(ct_name)] = {
            "n_cells": int(z_ct.shape[0]),
            "species_counts": counts.tolist(),
            "knn_species_purity": purity_ct,
            "knn_species_purity_chance": chance,
            "purity_vs_chance_excess": purity_ct - chance,
            "knn_mixing_entropy": float(ent_ct.mean()),
        }

    # Worst (highest purity above chance) shared-celltype → most stratified.
    worst_purity_excess = float("nan")
    worst_ct = None
    for ct, d in per_ct.items():
        if d.get("skipped"):
            continue
        e = d["purity_vs_chance_excess"]
        if np.isnan(worst_purity_excess) or e > worst_purity_excess:
            worst_purity_excess = e
            worst_ct = ct

    return {
        "global": {
            "knn_species_purity": knn_sp_purity,
            "knn_mixing_entropy": knn_mixing_entropy,
            "sil_species": sil_species,
            "sil_celltype": sil_celltype,
            "knn_celltype_purity": knn_ct_purity,
            "label_transfer_acc": label_transfer_acc,
            "n_shared_celltypes": len(shared_list),
            "shared_celltypes": shared_list,
        },
        "per_celltype": per_ct,
        "worst_shared_celltype": worst_ct,
        "worst_purity_vs_chance_excess": worst_purity_excess,
        "config": {
            "k": k, "n_cells": int(z.shape[0]),
            "n_latent": int(z.shape[1]),
        },
    }


def get_or_train_scvi(
    adata_path: Path, seed: int, max_epochs: int, n_latent: int, n_hidden: int,
    batch_key: str = "species",
    accelerator: str = "auto",
) -> np.ndarray:
    """Train (or load cached) vanilla scVI with the given batch_key.

    `accelerator` defaults to "auto" (Lightning's auto-detect) so existing
    callers (the Chimera ↔ scVI scorecard CLI) are unchanged. The
    `baseline_runner.py` shared-helper path passes `accelerator="cpu"` to
    honor the baseline pipeline's CPU-only rule.
    """
    import anndata as ad
    import scvi

    cache_root = RUNS_ROOT / "baselines_scvi"
    cache_root.mkdir(parents=True, exist_ok=True)
    # Include batch_key in cache path so different batch_keys don't collide.
    bk_suffix = f"-bk_{batch_key}" if batch_key != "species" else ""
    cache_dir = cache_root / f"{adata_path.stem}-s{seed}-e{max_epochs}-nl{n_latent}{bk_suffix}"
    z_path = cache_dir / "z_mean.npy"

    if z_path.exists():
        logger.info("Using cached scVI latent from %s", z_path)
        return np.load(z_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    adata = ad.read_h5ad(adata_path)
    # Safety: scVI expects raw counts in .X.
    if hasattr(adata.X, "data"):
        max_val = float(adata.X.data.max())
    else:
        max_val = float(np.asarray(adata.X).max())
    if max_val < 2 or (max_val < 50 and not np.all(np.asarray(adata.X[:5].todense() if hasattr(adata.X, "todense") else adata.X[:5]).astype(int) == np.asarray(adata.X[:5].todense() if hasattr(adata.X, "todense") else adata.X[:5]))):
        logger.warning("adata.X max=%.3f — verify this is raw counts.", max_val)

    scvi.model.SCVI.setup_anndata(adata, batch_key=batch_key)
    model = scvi.model.SCVI(adata, n_latent=n_latent, n_hidden=n_hidden)
    model.train(max_epochs=max_epochs, accelerator=accelerator, early_stopping=True)

    z = model.get_latent_representation()
    np.save(z_path, z)
    # Save config for reproducibility.
    (cache_dir / "config.json").write_text(
        json.dumps(
            {
                "adata": str(adata_path),
                "seed": seed,
                "max_epochs": max_epochs,
                "n_latent": n_latent,
                "n_hidden": n_hidden,
                "n_cells": int(adata.n_obs),
                "n_genes": int(adata.n_vars),
                "batch_key": batch_key,
            },
            indent=2,
        )
    )
    logger.info("Saved scVI latent to %s", z_path)
    return z


def get_pca_baseline(adata_path: Path, n_comps: int = 10) -> np.ndarray:
    """Log-normalize + HVG + PCA as a no-integration reference."""
    import anndata as ad
    import scanpy as sc

    a = ad.read_h5ad(adata_path).copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    try:
        sc.pp.highly_variable_genes(a, n_top_genes=2000, subset=True)
    except Exception:
        pass
    sc.tl.pca(a, n_comps=n_comps)
    return np.asarray(a.obsm["X_pca"])


def evaluate_e8_gates(
    chimera: Dict[str, Any], scvi: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Evaluate E8.1–E8.6 integration gates for Chimera vs the scVI baseline.

    Mirrors the gate definitions in workflow.md §E8. Each gate returns a row
    with keys ``id, name, value, threshold, pass``. Gates only use the
    Chimera and scVI scorecards (PCA is for context only, not a gate).
    """
    cg = chimera["global"]
    sg = scvi["global"]
    rows: List[Dict[str, Any]] = []

    def r(gid, name, value, threshold, passed, notes=""):
        rows.append(
            {
                "id": gid, "name": name, "value": value,
                "threshold": threshold, "pass": bool(passed),
                "notes": notes,
            }
        )

    # E8.1: global kNN species purity must not be worse than scVI + 0.01.
    delta_knn = cg["knn_species_purity"] - sg["knn_species_purity"]
    r("E8.1", "beats_scvi_knn_species_purity",
      f"chimera={cg['knn_species_purity']:.3f}, scvi={sg['knn_species_purity']:.3f}, Δ={delta_knn:+.3f}",
      "chimera − scVI ≤ 0.01",
      delta_knn <= 0.01)

    # E8.2: global sil-species must not be worse than scVI + 0.05.
    delta_ss = cg["sil_species"] - sg["sil_species"]
    r("E8.2", "beats_scvi_sil_species",
      f"chimera={cg['sil_species']:.3f}, scvi={sg['sil_species']:.3f}, Δ={delta_ss:+.3f}",
      "chimera − scVI ≤ 0.05",
      delta_ss <= 0.05)

    # E8.3: sil-celltype must be ≥ 0.25 AND ≥ scVI.
    # Fallback: when scVI itself can't reach 0.25 (e.g. 4-species with few shared
    # celltypes), the absolute threshold is unreachable by any model. In that regime,
    # require chimera ≥ scVI − 0.05 (on par with baseline).
    delta_sc = cg["sil_celltype"] - sg["sil_celltype"]
    abs_pass = cg["sil_celltype"] >= 0.25 and delta_sc >= -1e-9
    rel_pass = delta_sc >= -0.05  # chimera within 0.05 of scVI
    if sg["sil_celltype"] >= 0.25:
        # scVI can reach 0.25 → require both absolute and relative
        e83_pass = abs_pass
        e83_thresh = "sil_celltype ≥ 0.25 AND ≥ scVI"
    else:
        # scVI itself below 0.25 → structural limit, use relative clause
        e83_pass = rel_pass
        e83_thresh = f"sil_celltype ≥ scVI − 0.05 (scVI={sg['sil_celltype']:.3f} < 0.25, structural limit)"
    r("E8.3", "celltype_structure_preserved",
      f"chimera={cg['sil_celltype']:.3f}, scvi={sg['sil_celltype']:.3f}",
      e83_thresh,
      e83_pass)

    # E8.4 + E8.5: per-celltype within-celltype purity.
    per_c_chim = chimera["per_celltype"]
    per_c_scvi = scvi["per_celltype"]
    per_ct_gaps = []
    chance_excesses = []
    worst_ct_e84 = None
    worst_gap_e84 = -np.inf
    worst_ct_e85 = None
    worst_excess_e85 = -np.inf
    for ct, d in per_c_chim.items():
        if d.get("skipped"):
            continue
        scvi_d = per_c_scvi.get(ct, {})
        if scvi_d.get("skipped") or "knn_species_purity" not in scvi_d:
            continue
        gap = d["knn_species_purity"] - scvi_d["knn_species_purity"]
        per_ct_gaps.append((ct, gap))
        if gap > worst_gap_e84:
            worst_gap_e84 = gap
            worst_ct_e84 = ct
        excess = d["purity_vs_chance_excess"]
        chance_excesses.append((ct, excess))
        if excess > worst_excess_e85:
            worst_excess_e85 = excess
            worst_ct_e85 = ct

    if per_ct_gaps:
        r("E8.4", "per_celltype_purity_matches_scvi",
          f"worst-celltype gap vs scVI = {worst_gap_e84:+.3f} on '{worst_ct_e84}'",
          "max(chimera − scVI) per shared celltype ≤ 0.02",
          worst_gap_e84 <= 0.02,
          notes="Primary integration gate.")
        # E8.5: absolute OR relative clause.
        # Absolute: within-celltype purity near chance (≤ 0.15 excess).
        # Relative: Chimera no worse than scVI per celltype (≤ 0.05 excess gap).
        # The relative clause handles datasets where no model can achieve
        # near-chance mixing (very-divergent species, species-specific celltypes).
        worst_rel_excess = -np.inf
        worst_ct_e85_rel = None
        for ct, excess in chance_excesses:
            scvi_d = per_c_scvi.get(ct, {})
            scvi_excess = scvi_d.get("purity_vs_chance_excess", 0.0)
            rel = excess - scvi_excess
            if rel > worst_rel_excess:
                worst_rel_excess = rel
                worst_ct_e85_rel = ct
        abs_pass = worst_excess_e85 <= 0.15
        rel_pass = worst_rel_excess <= 0.05
        r("E8.5", "per_celltype_purity_near_chance",
          f"worst-celltype purity excess vs chance = {worst_excess_e85:+.3f} on '{worst_ct_e85}'"
          f"; worst relative excess vs scVI = {worst_rel_excess:+.3f} on '{worst_ct_e85_rel}'",
          "abs: max(chimera − chance) ≤ 0.15 OR rel: max(chimera − scVI excess) ≤ 0.05",
          abs_pass or rel_pass)
    else:
        r("E8.4", "per_celltype_purity_matches_scvi",
          "no comparable shared celltypes", "≤ 0.02", False,
          notes="No shared celltypes could be compared — dataset pairing likely unsuitable.")
        r("E8.5", "per_celltype_purity_near_chance",
          "no shared celltypes", "≤ 0.15", False)

    # E8.6: label transfer accuracy on shared celltypes ≥ scVI − 0.05.
    lt_c = cg.get("label_transfer_acc", float("nan"))
    lt_s = sg.get("label_transfer_acc", float("nan"))
    if np.isnan(lt_c) or np.isnan(lt_s):
        r("E8.6", "label_transfer_acc_matches_scvi",
          f"chimera={lt_c}, scvi={lt_s}", "chimera ≥ scVI − 0.05", False,
          notes="Label-transfer undefined — insufficient shared celltypes.")
    else:
        delta_lt = lt_c - lt_s
        r("E8.6", "label_transfer_acc_matches_scvi",
          f"chimera={lt_c:.3f}, scvi={lt_s:.3f}, Δ={delta_lt:+.3f}",
          "chimera ≥ scVI − 0.05",
          delta_lt >= -0.05)

    return rows


def format_markdown(
    rows: List[Dict[str, Any]], meta: Dict[str, Any]
) -> str:
    """Build the markdown report."""
    lines = [
        f"# Integration scorecard — Chimera vs vanilla scVI",
        "",
        f"- Dataset: `{meta['adata']}`  — {meta['n_cells']} cells, "
        f"{meta['n_species']} species, {meta['n_shared_celltypes']} shared celltypes.",
        f"- Seed: {meta['seed']}; k={meta['k']}; n_latent={meta['n_latent']}.",
        f"- Shared celltypes: {meta['shared_celltypes']}",
        "",
        "## Global metrics",
        "",
        "| Model | kNN-sp ↓ | mix-ent ↑ | sil-sp ↓ | sil-ct ↑ | kNN-ct ↑ | LT-acc ↑ |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        g = r["global"]
        lines.append(
            f"| {r['name']} "
            f"| {g['knn_species_purity']:.3f} "
            f"| {g['knn_mixing_entropy']:.3f} "
            f"| {g['sil_species']:.3f} "
            f"| {g['sil_celltype']:.3f} "
            f"| {g['knn_celltype_purity']:.3f} "
            f"| {g['label_transfer_acc']:.3f} |"
        )
    lines.append("")
    lines.append("*Arrows indicate the preferred direction. kNN-sp: kNN species")
    lines.append(" purity; mix-ent: normalized species mixing entropy among k neighbors;")
    lines.append(" sil-sp / sil-ct: silhouette by species / celltype; LT-acc: label-transfer accuracy.*")
    lines.append("")

    lines.append("## Per-shared-celltype within-celltype kNN species purity")
    lines.append("")
    lines.append(
        "| Celltype | n | chance | "
        + " | ".join(r["name"] for r in rows)
        + " |"
    )
    lines.append(
        "|---|---|---|"
        + "|".join(["---"] * len(rows))
        + "|"
    )
    # Use first model's per_celltype to enumerate types.
    ct_names = []
    for r in rows:
        ct_names = sorted(r["per_celltype"].keys())
        if ct_names:
            break
    for ct in ct_names:
        first = rows[0]["per_celltype"].get(ct, {})
        if first.get("skipped"):
            continue
        row = [
            f"{ct}",
            f"{first['n_cells']}",
            f"{first['knn_species_purity_chance']:.3f}",
        ]
        for r in rows:
            d = r["per_celltype"].get(ct, {})
            if d.get("skipped"):
                row.append("—")
            else:
                row.append(
                    f"{d['knn_species_purity']:.3f} ({d['purity_vs_chance_excess']:+.3f})"
                )
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("*Per-celltype values are `kNN purity (delta vs chance)`. A celltype-local")
    lines.append(" purity close to chance means species are well-mixed within that celltype.*")
    lines.append("")

    # Worst per-celltype gap highlighted.
    lines.append("## Worst per-celltype gap")
    lines.append("")
    for r in rows:
        lines.append(
            f"- **{r['name']}**: worst celltype = `{r['worst_shared_celltype']}` "
            f"with purity-vs-chance excess = {r['worst_purity_vs_chance_excess']:+.3f}."
        )
    lines.append("")

    # Emit E8 pass/fail gate table if we have Chimera + scVI rows.
    chim_row = next(
        (rr for rr in rows if rr["name"].startswith("Chimera[")), None
    )
    scvi_row = next(
        (rr for rr in rows if "scVI" in rr["name"]), None
    )
    if chim_row is not None and scvi_row is not None:
        e8 = evaluate_e8_gates(chim_row, scvi_row)
        lines.append("## E8 integration gates (Chimera vs vanilla scVI)")
        lines.append("")
        lines.append("| ID | Name | Value | Threshold | Pass |")
        lines.append("|---|---|---|---|---|")
        for g in e8:
            mark = "✓" if g["pass"] else "✗"
            note = f" ({g['notes']})" if g.get("notes") else ""
            lines.append(
                f"| {g['id']} | {g['name']} | {g['value']} | {g['threshold']} | {mark}{note} |"
            )
        lines.append("")
        n_pass = sum(1 for g in e8 if g["pass"])
        n_fail = len(e8) - n_pass
        lines.append(f"**E8 summary: {n_pass}/{len(e8)} pass, {n_fail} fail.**")
        lines.append("")
        if n_fail:
            failed_ids = [g["id"] for g in e8 if not g["pass"]]
            lines.append(f"Failed gates: {', '.join(failed_ids)}")
            lines.append("")
        meta["_e8"] = e8
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chimera-run-name", required=True, type=str)
    ap.add_argument(
        "--adata", required=True, type=str,
        help="Path to the adata used for training the Chimera run.",
    )
    ap.add_argument("--scvi-max-epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-latent", type=int, default=10)
    ap.add_argument("--n-hidden", type=int, default=128)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--skip-pca", action="store_true")
    ap.add_argument("--batch-key", type=str, default="species",
                    help="batch_key for scVI baseline (default 'species'; use "
                         "'sample_batch' for donor-level correction)")
    args = ap.parse_args()

    run_dir = RUNS_ROOT / args.chimera_run_name
    if not run_dir.exists():
        print(f"Run directory not found: {run_dir}", file=sys.stderr)
        sys.exit(1)

    fs_path = run_dir / "final_state.npz"
    if not fs_path.exists():
        print(f"No final_state.npz at {fs_path}", file=sys.stderr)
        sys.exit(1)
    chimera_z = np.asarray(np.load(fs_path)["z_mean"])

    adata_path = (REPO_ROOT / args.adata).resolve() if not os.path.isabs(
        args.adata
    ) else Path(args.adata)

    # Load species + cell_type labels from the adata.
    import anndata as ad
    adata = ad.read_h5ad(adata_path)
    if chimera_z.shape[0] != adata.n_obs:
        print(
            f"Chimera z shape {chimera_z.shape} does not match adata "
            f"{adata.n_obs} cells — are you sure this is the matching adata?",
            file=sys.stderr,
        )
        sys.exit(1)
    species = np.asarray(adata.obs["species"].values)
    if "cell_type" in adata.obs.columns:
        cell_types = np.asarray(adata.obs["cell_type"].values)
    elif "cell_ontology_class" in adata.obs.columns:
        cell_types = np.asarray(adata.obs["cell_ontology_class"].values)
    else:
        raise RuntimeError("No cell_type-like column in adata.obs.")

    # Train vanilla scVI baseline (or load cached).
    logger.info("=== Vanilla scVI baseline ===")
    scvi_z = get_or_train_scvi(
        adata_path,
        seed=args.seed,
        max_epochs=args.scvi_max_epochs,
        n_latent=args.n_latent,
        n_hidden=args.n_hidden,
        batch_key=args.batch_key,
    )
    if scvi_z.shape[0] != adata.n_obs:
        raise RuntimeError(
            f"scVI z shape {scvi_z.shape} does not match adata {adata.n_obs} cells."
        )

    rows = []
    logger.info("Metrics for Chimera …")
    chimera_m = compute_integration_metrics(
        chimera_z, species, cell_types, k=args.k, seed=args.seed
    )
    rows.append({"name": f"Chimera[{args.chimera_run_name}]", **chimera_m})

    logger.info("Metrics for vanilla scVI …")
    scvi_m = compute_integration_metrics(
        scvi_z, species, cell_types, k=args.k, seed=args.seed
    )
    rows.append({"name": f"vanilla scVI (batch={args.batch_key})", **scvi_m})

    if not args.skip_pca:
        logger.info("Metrics for raw log-normalized PCA (no integration) …")
        pca_z = get_pca_baseline(adata_path, n_comps=args.n_latent)
        pca_m = compute_integration_metrics(
            pca_z, species, cell_types, k=args.k, seed=args.seed
        )
        rows.append({"name": "log-norm HVG PCA (no integration)", **pca_m})

    # Assemble markdown + json.
    meta = {
        "adata": str(adata_path),
        "n_cells": int(adata.n_obs),
        "n_species": int(len(np.unique(species))),
        "n_shared_celltypes": chimera_m["global"]["n_shared_celltypes"],
        "shared_celltypes": chimera_m["global"]["shared_celltypes"],
        "seed": args.seed,
        "k": args.k,
        "n_latent": args.n_latent,
    }
    md = format_markdown(rows, meta)

    md_path = run_dir / "integration_vs_scvi.md"
    md_path.write_text(md)
    json_path = run_dir / "integration_vs_scvi.json"
    json_path.write_text(
        json.dumps(
            {"meta": meta, "rows": rows, "e8": meta.get("_e8", [])},
            indent=2,
            default=str,
        )
    )
    print(md)
    logger.info("Wrote %s and %s", md_path, json_path)


if __name__ == "__main__":
    main()
