"""B2 Track C: direct baselines for every Track C cohort, T4.2 and X4/X6 code on each cohort's h5ad, and B1z mean over up to three kept fits. No gold is read.
Labels: canonical T4.2 labels; new cohorts cell_ontology_class. Library: T4.2 keys; kidney and islet sample_batch; bone marrow and eye batch. Reproduces T4.3 B1b_raw, B1c_raw, B2b and H1 B1 where they exist.
Writes results/revision/B2/track_C/baselines__<cohort>.parquet (index: human ENSG).
    python -m revision.tasks.B2_benchmark.baselines_c <cohort>
"""

from __future__ import annotations

import json
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, os.environ.get("SLURM_CPUS_PER_TASK", "4"))

import anndata as ad
import numpy as np
import pandas as pd

from revision.common import cohorts as CO
from revision.common import gene_ids
from revision.tasks.B2_benchmark import perturb as P
from revision.tasks.B2_benchmark import rules as RU
from revision.tasks.B2_benchmark.score_run import pseudobulk_fb
from revision.tasks.H1_harmonise import rules as H1R
from revision.tasks.T4_2_gene_baselines import run as T42
from revision.tasks.X6_adds_beyond.degrade import sdir

OUT = RU.RES / "track_C"
RES = RU.REPO / "results" / "revision"


def run(cohort: str) -> None:
    out = OUT / f"baselines__{cohort}.parquet"
    if out.exists():
        return
    d = H1R.COHORTS[cohort]
    a = ad.read_h5ad(d["h5ad"])
    X = P.csr(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    species = sorted(set(spc))
    G = a.n_vars
    if cohort.startswith("test"):
        lab = a.obs["cell_type"].astype(str).map(T42.TESTIS_COARSE).to_numpy()
        lib = a.obs["sample_batch"].astype(str).to_numpy()
    elif cohort.startswith("liv"):
        lab = a.obs["cell_type"].astype(str).to_numpy()
        lib = a.obs["batch"].astype(str).to_numpy()
    else:
        lab = a.obs["cell_ontology_class"].astype(str).to_numpy()
        lib = a.obs["sample_batch" if cohort in ("kidney", "islet") else "batch"].astype(str).to_numpy()
    a2 = a.copy()
    a2.obs["cell_type"] = lab
    cov = CO.gene_covariates(a2)
    cov["prior_raw"] = np.asarray(a.var["conservation_prior"], dtype=float)
    res = pd.DataFrame(index=pd.Index(np.asarray(gene_ids.ensembl(d["table"], a.var_names)), name="hsapiens_gene_id"))
    res["var_name"] = a.var_names.astype(str)
    x, f, units = T42.pseudobulk(X, spc, lab, lib, T42.MIN_CELLS_TYPE)
    E, _, pinfo = T42.pooled_E(x, f, species, G)
    Lv, _, B2, _, groups = T42.level_and_specificity(x, f, species, G)
    c3 = cov[["prior_raw", "log_cp10k_spmean", "detection_spmean"]]
    c3.index = res.index
    res["B1"] = E
    res["B1b"] = T42.residualise(pd.Series(E, index=res.index), c3)
    res["B1c"] = Lv
    res["B2b"] = B2
    res["Sdir"] = sdir(X, spc, lab)
    ps = [(np.asarray(X[spc == s].sum(0)).ravel() + 0.5) / (X[spc == s].sum() + 0.5 * G) for s in species]
    lf = [np.abs(np.log2(ps[i] / ps[j])) for i in range(len(ps)) for j in range(i + 1, len(ps))]
    res["Lfc"] = -np.mean(lf, 0)
    res["B3a"] = T42.cross_fit_ridge(np.asarray(a.varm["gene_features"], dtype=float), np.asarray(E, dtype=float))[0]
    rt = pd.read_csv(RES / "H1/runs.tsv", sep="\t")
    rt = rt[(rt.cohort == cohort) & (rt.kept)]
    zs = []
    for run_ in rt.run.tolist()[:3]:
        z = np.asarray(CO.load_state(run_)["z_mean"], dtype=np.float32)
        cl = T42.leiden_clusters(z)
        xz, fz, _, _ = pseudobulk_fb(X, spc, cl, lib, RU.MIN_CELLS_CLUSTER, "all")
        zs.append(T42.pooled_E(xz, fz, species, G)[0])
    res["B1z"] = np.nanmean(np.column_stack(zs), 1) if zs else np.nan
    rep = dict(cohort=cohort, K_pairs=pinfo, K_all=len(groups), B1z_fits=len(zs))
    b0 = RES / f"T4_3/baselines_consensusD/baselines__{cohort}.csv"
    if b0.exists():
        t = pd.read_csv(b0).set_index("hsapiens_gene_id").reindex(res.index)
        rep["repro_T43"] = {k: float(np.nanmax(np.abs(res[k] - t[v]))) for k, v in (("B1", "B1b_raw"), ("B1c", "B1c_raw")) if t[v].notna().any()}
    h = pd.read_parquet(RES / f"H1/genes/{cohort}.parquet").set_index("hsapiens_gene_id").reindex(res.index)
    rep["repro_H1_B1"] = float(np.nanmax(np.abs(res["B1"] - h["B1"]))) if h["B1"].notna().any() else None
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"baselines__{cohort}.json").write_text(json.dumps(rep, indent=1, default=str))
    res.reset_index().to_parquet(out)
    print(rep, flush=True)


if __name__ == "__main__":
    from revision.tasks.B2_benchmark.rules_c import COHORTS_C

    run(sys.argv[1] if not sys.argv[1].isdigit() else list(COHORTS_C)[int(sys.argv[1])])
