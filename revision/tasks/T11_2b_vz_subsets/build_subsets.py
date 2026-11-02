"""T11.2b: Van Zyl subsets and their fixed recipe grids (rule in README.md).

Builds five subsets of the T11.1 object: human paired with M. fascicularis, M. mulatta, pig
and mouse, and the primates (human, M. fascicularis, M. mulatta), with
``training/build_subset.py::build_subset`` unchanged, as notebook 01 cells 34 and 46 do
(``drop_empty_genes=True, min_cells_per_gene=3``; prior and gene features carried over from
the five-species object, p_null recomputed). Then, without changing any value, it drops
category levels left over from the removed species in the categorical obs columns
(``build_subset`` re-categorises only ``species`` and ``batch``), so ``source_batch`` has
only the levels present in the subset; runs ``revision.common.check_cohort`` with
``--batch-key source_batch``; and writes each subset's grid from the README rule
(``grid.tsv``, and ``manifest.tsv`` holding every subset's rows for one array).

Run only after the coordinator confirms the R4 re-audit of the T11.1 object is clean.

    python revision/tasks/T11_2b_vz_subsets/build_subsets.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training"))

from build_subset import build_subset  # noqa: E402
from revision.common.check_cohort import check  # noqa: E402

TASK = Path(__file__).resolve().parent
VZ = REPO / "data" / "revision" / "van_zyl"
SOURCE = VZ / "van_zyl_unified.h5ad"
SUBSETS = {  # tag -> species in obs['species'] (owner instruction 2026-11-02; notebook 01 cell 46)
    "hmfas": ("human", "macaca_fascicularis"),
    "hmmul": ("human", "macaca_mulatta"),
    "hpig": ("human", "pig"),
    "hmouse": ("human", "mouse"),
    "primates": ("human", "macaca_fascicularis", "macaca_mulatta"),
}
CATEGORICAL = ("species", "batch", "source_batch", "chemistry", "cell_type", "Cluster",
               "cell_type_harmonised", "cell_ontology_class", "cell_class")
BATCH_KEY, CT_KEY = "source_batch", "cell_ontology_class"

# Canonical rows (07_EXPERIMENT_MATRIX.md) and the section 11 base at K = 2 and K = 3.
ROWS = {
    "rule_base_k2": dict(epochs=150, adv=100, anchor=0.003, mmd=50, fb=0.5, ct=True, clamp=0.85,
                         role="section 11 base: testis 3-sp row form (conditional MMD, clamp 0.85) with K = 2 weights (Laws 1-5)"),
    "rule_base_k3": dict(epochs=200, adv=133, anchor=0.001, mmd=100, fb=1.0, ct=True, clamp=0.85,
                         role="section 11 base: testis 3-sp row form with K = 3 weights (Law 1 150*sqrt(3/2)=184 -> 200, "
                              "Law 2 0.001, Law 3 50*(K-1)=100, Law 4 0.5*ceil(log2 3)=1.0, Law 5 100*(1-1/3)/(1-1/2)=133)"),
    "testis_3sp": dict(epochs=100, adv=100, anchor=0.001, mmd=50, fb=1.0, ct=True, clamp=0.85,
                       role="canonical testis 3-sp row (iter115)"),
    "liver_4sp": dict(epochs=200, adv=200, anchor=0.001, mmd=100, fb=1.0, ct=False, clamp=None,
                      role="canonical liver 4-sp row (iter101), unclamped"),
    "testis_hmouse": dict(epochs=200, adv=200, anchor=0.003, mmd=50, fb=1.0, ct=True, clamp=0.85,
                          role="canonical testis H+Mouse row (iter111)"),
    "liver_hmouse": dict(epochs=150, adv=200, anchor=0.003, mmd=50, fb=1.0, ct=False, clamp=None,
                         role="canonical liver H+Mouse row (iter90 xds)"),
    "liver_hmonkey": dict(epochs=150, adv=100, anchor=0.003, mmd=50, fb=0.5, ct=False, clamp=None,
                          role="canonical liver H+Monkey row (iter90, defaults)"),
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def args_for(row: dict, adata: str) -> str:
    a = (f"--config A --max-epochs {row['epochs']} --adata {adata} --seed 42 "
         f"--batch-key {BATCH_KEY} --celltype-key {CT_KEY} --adv-weight {row['adv']:g} "
         f"--lambda-anchor {row['anchor']:g} --lambda-mmd {row['mmd']:g} --free-bits {row['fb']:g}")
    if row["ct"]:
        a += " --celltype-conditional-mmd"
    if row["clamp"] is not None:
        a += f" --prior-clamp-max {row['clamp']:g}"
    return a


def grid_rows(n_species: int, frac: float) -> list[str]:
    """README rule: the rule base enters only if frac(prior >= 0.95) >= 0.20 (otherwise the
    rule's start is the unclamped canonical row already in the grid)."""
    if n_species == 2:
        base, keys = "rule_base_k2", ["testis_hmouse", "liver_hmouse", "liver_hmonkey"]
    elif n_species == 3:
        base, keys = "rule_base_k3", ["testis_3sp", "liver_4sp", "liver_hmonkey"]
    else:
        raise ValueError(n_species)
    return ([base] + keys) if frac >= 0.20 else keys


def main() -> int:
    import anndata as ad

    grid, report = [], {"source": str(SOURCE.relative_to(REPO)), "source_sha256": sha256(SOURCE), "subsets": {}}
    manifest = ["run_name\targs\n"]
    for tag, species in SUBSETS.items():
        dest = VZ / f"van_zyl_{tag}.h5ad"
        build_subset(source_path=SOURCE, species=species, dest_path=dest,
                     drop_empty_genes=True, min_cells_per_gene=3)
        a = ad.read_h5ad(dest)
        for c in CATEGORICAL:
            if c in a.obs:
                a.obs[c] = a.obs[c].astype(str).astype("category")
        a.write_h5ad(dest)
        errs = check(str(dest), batch_key=BATCH_KEY, celltype_key=CT_KEY)
        if errs:
            raise SystemExit(f"check_cohort failed for {tag}: {errs}")
        if sorted(a.obs["species"].astype(str).unique()) != sorted(species):
            raise SystemExit(f"{tag}: species {sorted(a.obs['species'].unique())} != {sorted(species)}")
        # Compare in float32: the prior is stored as float32 and its clamp value 0.95 is
        # float32(0.95) < 0.95 as a float64, so a float64 comparison would count no gene.
        prior = np.asarray(a.var["conservation_prior"], dtype=np.float32)
        frac = float((prior >= np.float32(0.95)).mean())
        shared = pd.crosstab(a.obs["cell_type_harmonised"], a.obs["species"])
        report["subsets"][tag] = {
            "species": list(species), "shape": list(a.shape), "sha256": sha256(dest),
            "cells_by_species": a.obs["species"].value_counts().to_dict(),
            "cells_by_source_batch": a.obs[BATCH_KEY].value_counts().to_dict(),
            "frac_prior_ge_0.95": frac, "check_cohort": "OK",
            "n_types_shared": int((shared > 0).all(axis=1).sum()),
            "frac_cells_in_shared_types": float(a.obs["cell_type_harmonised"].isin(
                shared.index[(shared > 0).all(axis=1)]).mean()),
        }
        rel = str(dest.relative_to(REPO))
        for k, key in enumerate(grid_rows(len(species), frac), start=1):
            row = ROWS[key]
            name = f"rev-vz-{tag}-grid{k}-s42"
            args = args_for(row, rel)
            grid.append(dict(subset=tag, grid_id=k, run_name=name, source_row=key, adv_weight=row["adv"],
                             lambda_anchor=row["anchor"], lambda_mmd=row["mmd"], free_bits=row["fb"],
                             prior_clamp_max=row["clamp"] or "none", celltype_conditional_mmd=row["ct"],
                             max_epochs=row["epochs"], role=row["role"], args=args))
            manifest.append(f"{name}\t{args}\n")
    (TASK / "manifest.tsv").write_text("".join(manifest))
    pd.DataFrame(grid).to_csv(TASK / "grid.tsv", sep="\t", index=False)
    (VZ / "subsets_build_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
