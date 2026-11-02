"""C1: pseudobulk DE for the disease cases, the P13 fit unchanged (`P13_kidney/de.py::fit`, `permuted_labels`). Runs in revision_de_env, never chimera_env. No Chimera readout.

    python revision/tasks/C1_cohorts/de_case.py --case ad [--perm-task K --n-chunks N] [--n-cpus 8]

Primary: human l^h = log2fc_mle, mouse m = log2fc_fixed. A unit needs at least 20 nuclei of the type. Label null: relabellings within
strata (`CASES`), N_PERM = 200 or all distinct relabellings where fewer exist, never the observed labels or their swap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from revision.common import gene_ids
from revision.tasks.P13_kidney import de as P

RES = gene_ids.REPO_ROOT / "results" / "revision" / "C1"
SEED = 20261102
N_PERM = 200
MIN_UNITS = 2
CASES = {"ad": dict(types=["microglia", "astrocyte", "oligodendrocyte"], cov=dict(human=["sex"], mouse=[]), strata=dict(human=["sex"], mouse=[])),
         "muscle": dict(types=["fb", "musc", "smc", "pericyte", "capec", "artec", "venec", "macrophage"], cov=dict(human=["assay", "sex"], mouse=[]), strata=dict(human=["assay", "sex"], mouse=[])),
         "lung": dict(types=["macrophage", "at2", "endothelial", "t", "dc"], cov=dict(human=["source"], mouse=[]), strata=dict(human=["source"], mouse=[]),
                      descriptive_only=["fibroblast"])}


def jobs(case: str):
    d = RES / case
    pbs = {s: ad.read_h5ad(d / f"pb_{s}.h5ad") for s in ("human", "mouse")}
    out = []
    for t in CASES[case]["types"]:
        for s, a in pbs.items():
            sub = a[(a.obs["cell_ontology_class"] == t).to_numpy() & (a.obs["n_nuclei"].astype(int) >= P.MIN_CELLS_UNIT).to_numpy()].copy()
            for c in sub.obs.columns:
                if sub.obs[c].dtype.name == "category":
                    sub.obs[c] = sub.obs[c].astype(str)
            out.append((t, s, sub))
    return out


def run_primary(case: str, n_cpus: int) -> None:
    out = RES / case / "de"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t, s, sub in jobs(case):
        n = sub.obs["condition"].value_counts().to_dict()
        if n.get("control", 0) < MIN_UNITS or n.get("disease", 0) < MIN_UNITS:
            rows.append(dict(cell_type=t, species=s, design="skipped", n_control=n.get("control", 0), n_disease=n.get("disease", 0)))
            continue
        r, design = P.fit(sub, CASES[case]["cov"][s], n_cpus)
        r.to_csv(out / f"{s}_primary_{t}.tsv", sep="\t", index=False)
        rows.append(dict(cell_type=t, species=s, design=design, n_control=n.get("control", 0), n_disease=n.get("disease", 0), units=",".join(sub.obs["unit"].astype(str)),
                         n_genes_tested=len(r), n_padj_lt_0_1=int((r["padj"] < 0.1).sum()), frac_abs_fixed_lt_0001=float((r["log2fc_fixed"].abs() < 0.001).mean())))
        pd.DataFrame(rows).to_csv(out / "de_summary.tsv", sep="\t", index=False)
    (out / "de_meta.json").write_text(json.dumps(dict(de_types=CASES[case]["types"], min_cells_unit=P.MIN_CELLS_UNIT, min_counts=P.MIN_COUNTS), indent=2))
    (out / "DONE").write_text("ok\n")


def permuted(obs: pd.DataFrame, strata: list[str], rng, n: int) -> list:
    return P.permuted_labels(obs.assign(_all="a"), strata or ["_all"], rng, n)


def run_perm(case: str, task: int, n_chunks: int, n_cpus: int) -> None:
    types = CASES[case]["types"]
    t = types[task // n_chunks]
    chunk = task % n_chunks
    out = RES / case / "de" / "perm"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t2, s, sub in jobs(case):
        if t2 != t:
            continue
        rng = np.random.default_rng([SEED, types.index(t), 0 if s == "human" else 1])
        labels = permuted(sub.obs, CASES[case]["strata"][s], rng, N_PERM)
        for i, lab in enumerate(labels):
            if i % n_chunks != chunk:
                continue
            f = out / f"{s}_perm{i:02d}_{t}.tsv"
            if f.exists():
                continue
            ss = sub.copy()
            ss.obs["condition"] = lab
            r, design = P.fit(ss, CASES[case]["cov"][s], n_cpus, adaptive=False)
            r.to_csv(f, sep="\t", index=False)
            rows.append(dict(cell_type=t, species=s, perm=i, design=design, controls=",".join(ss.obs["unit"].astype(str)[lab == "control"]), n_draws=len(labels)))
    pd.DataFrame(rows).to_csv(out / f"labels_{t}_c{chunk}.tsv", sep="\t", index=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--perm-task", type=int, default=None)
    ap.add_argument("--n-chunks", type=int, default=1)
    ap.add_argument("--n-cpus", type=int, default=8)
    a = ap.parse_args()
    if a.perm_task is None:
        run_primary(a.case, a.n_cpus)
    else:
        run_perm(a.case, a.perm_task, a.n_chunks, a.n_cpus)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
