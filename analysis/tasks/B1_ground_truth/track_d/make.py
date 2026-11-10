"""Track D: subsampled objects (testis H+Mouse 25%, 50%; same random cells for Chimera and the tools) and the job table.

Cells: the rng(0) choice of the SATURN and CAMEX adapters (analysis/tasks/T5_2_adapters, prep_camex --subsample). Chimera
objects keep the gene set and the canonical features with p_null recomputed and the expression columns moved as in alter.py.
Tool inputs are the input files of T5_2_adapters subset to the same cells. Writes data/analysis/B1/trackD/ and jobs.tsv: one
job per method x object x repeat, all on 16 CPUs, one node.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.tasks.B1_ground_truth import alter as A
from analysis.tasks.B1_ground_truth import rules as RU

OUT = RU.REPO / "data" / "analysis" / "B1" / "trackD"
HERE = Path(__file__).resolve().parent
STEM = {"liv_hmouse": "camex_human_mouse", "test_hmouse": "camex_testis_human_mouse", "kidney": "kidney_healthy_train"}
FRACS = {"f25": 0.25, "f50": 0.5}
RECIPES_EXTRA = {  # more species, same runs as T3_2_seeds and T5_3_runs
    "test_3sp": ("data/processed/camex_testis_unified.h5ad",
                 "--config A --max-epochs 100 --adv-weight 100 --lambda-anchor 0.001 --free-bits 1.0 --lambda-mmd 50 --celltype-conditional-mmd --celltype-key cell_ontology_class --prior-clamp-max 0.85"),
    "liv_4sp": ("data/processed/camex_pancreas_unified.h5ad",
                "--config A --max-epochs 200 --adv-weight 200 --lambda-anchor 0.001 --lambda-mmd 100 --free-bits 1.0"),
}


def subsample():
    c = "test_hmouse"
    a = A.read_base(c)
    X0 = A.csr(a.X)
    rows = {}
    for name, f in FRACS.items():
        n = int(round(f * a.n_obs))
        idx = np.sort(np.random.default_rng(0).choice(a.n_obs, size=n, replace=False))
        b = a[idx].copy()
        X1 = A.csr(b.X)
        b.uns["p_null"] = A.p_null(X1.astype(np.float64), b.obs["species"].astype(str).to_numpy())
        st0, st1 = A.expr_stats(X0), A.expr_stats(X1)
        F = np.array(a.varm["gene_features"], copy=True)
        for k, col in enumerate(A.EXPR_COLS):
            F[:, col] = F[:, col] + (st1[:, k] - st0[:, k]) / st0[:, k].std()
        b.varm["gene_features"] = F
        path = OUT / f"{c}_{name}.h5ad"
        b.write_h5ad(path)
        (OUT / f"{c}_{name}.obs.txt").write_text("\n".join(b.obs_names.astype(str)))
        rows[name] = dict(path=str(path.relative_to(RU.REPO)), n_cells=int(n))
    (OUT / "subsamples.json").write_text(json.dumps(rows, indent=1))
    return rows


def jobs(sub) -> pd.DataFrame:
    rows = []
    for c, cfg in RU.COHORTS.items():
        for rep in (1, 2):
            rows.append(dict(job=f"chimera-{c}-full-r{rep}", method="chimera", cohort=c, object="full", n_species=2, h5ad=cfg["h5ad"], recipe=cfg["recipe"]))
        rows.append(dict(job=f"direct-{c}-full-r1", method="direct", cohort=c, object="full", n_species=2, h5ad=cfg["h5ad"], recipe=""))
        for t in ("saturn", "camex"):
            rows.append(dict(job=f"{t}-{c}-full-r1", method=t, cohort=c, object="full", n_species=2, h5ad=cfg["h5ad"], recipe=""))
    c = "test_hmouse"
    for name, r in sub.items():
        rows.append(dict(job=f"chimera-{c}-{name}-r1", method="chimera", cohort=c, object=name, n_species=2, h5ad=r["path"], recipe=RU.COHORTS[c]["recipe"]))
        rows.append(dict(job=f"direct-{c}-{name}-r1", method="direct", cohort=c, object=name, n_species=2, h5ad=r["path"], recipe=""))
        for t in ("saturn", "camex"):
            rows.append(dict(job=f"{t}-{c}-{name}-r1", method=t, cohort=c, object=name, n_species=2, h5ad=r["path"], recipe=""))
    for k, (p, rec) in RECIPES_EXTRA.items():
        sp = 3 if k == "test_3sp" else 4
        rows.append(dict(job=f"chimera-{k}-full-r1", method="chimera", cohort=k, object="full", n_species=sp, h5ad=p, recipe=rec))
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sub = subsample()
    j = jobs(sub)
    j.to_csv(HERE / "jobs.tsv", sep="\t", index=False)
    print(j.to_string())


if __name__ == "__main__":
    main()
