"""Control of the liver MASH case: Guilliams lean human hepatocyte pseudobulks, for Gribben healthy vs Guilliams lean.

Donor H38 is the only Guilliams nuclei donor with a lean label we can verify (``diet`` in the
CAMEX object for CS170; GEO marks H38 Visium as healthy and gives no diet for the other donors).
CS170 trained the canonical liver H+Mouse model, so it shares a sample with D; CS171 (same donor,
second library, GSE192742) does not, and it is the primary side of the control. CS170 is shown
beside it.

CS171 has no labels in GEO: QC as for the liver MASH objects (analysis/tasks/T7_2_disease_objects;
>= 200 genes, mitochondrial fraction < 20%), then kNN transfer (k = 15, PCA on 2,000 HVG of the
reference) from the CS170 annotation.

    PYTHONPATH=<chimera repo> python analysis/tasks/T7_3_mash/guilliams.py
"""

from __future__ import annotations

import sys

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.neighbors import KNeighborsClassifier

from analysis.common import gene_ids

ROOT = gene_ids.REPO_ROOT
RES = ROOT / "results" / "analysis" / "T7_3"
REF = ROOT / "data" / "raw" / "camex" / "1liver" / "dataset" / "raw-liver-human-Martin.h5ad"
H5 = ROOT / "data" / "analysis" / "disease" / "raw" / "GSE192742_lean_human" / "GSM5764401_filtered_feature_bc_matrix_CS171.h5"
HEP = ["ALB", "APOA1", "CYP2E1", "CPS1", "PCK1", "TF"]


def lognorm(a: ad.AnnData) -> ad.AnnData:
    a = a.copy()
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


def main() -> int:
    RES.mkdir(parents=True, exist_ok=True)
    ref = ad.read_h5ad(REF)
    ref.var_names_make_unique()
    q = sc.read_10x_h5(H5)
    q.var_names_make_unique()
    n0 = q.n_obs
    q.var["mt"] = q.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(q, qc_vars=["mt"], inplace=True, percent_top=None, log1p=False)
    q = q[(q.obs["n_genes_by_counts"] >= 200) & (q.obs["pct_counts_mt"] < 20)].copy()
    genes = ref.var_names.intersection(q.var_names)
    r, qq = lognorm(ref[:, genes]), lognorm(q[:, genes])
    sc.pp.highly_variable_genes(r, n_top_genes=2000, flavor="seurat")
    hv = r.var_names[r.var["highly_variable"]]
    rx = np.asarray(r[:, hv].X.toarray(), dtype=np.float32)
    qx = np.asarray(qq[:, hv].X.toarray(), dtype=np.float32)
    mu, sd = rx.mean(0), rx.std(0) + 1e-6
    from sklearn.decomposition import PCA

    pca = PCA(n_components=30, random_state=0).fit((rx - mu) / sd)
    knn = KNeighborsClassifier(n_neighbors=15).fit(pca.transform((rx - mu) / sd), ref.obs["annot"].astype(str))
    lab = knn.predict(pca.transform((qx - mu) / sd))
    q.obs["annot"] = lab
    hepm = [g for g in HEP if g in genes]
    score = pd.Series(np.asarray(qq[:, hepm].X.mean(1)).ravel(), index=q.obs_names)
    chk = score.groupby(q.obs["annot"]).agg(["size", "mean"]).reset_index()
    chk.insert(0, "sample", "CS171")
    chk.to_csv(RES / "guilliams_label_check.tsv", sep="\t", index=False)
    print(chk, f"QC {n0} -> {q.n_obs}")
    gid_q = q.var["gene_ids"].astype(str).to_numpy()
    gid_r = ref.var["gene_ids"].astype(str).to_numpy()
    a = pd.Series(np.asarray(q[q.obs["annot"] == "Hepatocytes"].X.sum(0)).ravel(), index=gid_q).groupby(level=0).sum()
    b = pd.Series(np.asarray(ref[ref.obs["annot"] == "Hepatocytes"].X.sum(0)).ravel(), index=gid_r).groupby(level=0).sum()
    out = pd.DataFrame({"CS170": b, "CS171": a}).dropna().astype(np.int64)
    out.index.name = "gene_id"
    out.to_csv(RES / "guilliams_hepatocytes.tsv", sep="\t")
    print(out.sum().to_dict(), "nuclei CS171", int((q.obs["annot"] == "Hepatocytes").sum()),
          "CS170", int((ref.obs["annot"] == "Hepatocytes").sum()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
