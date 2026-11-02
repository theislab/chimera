"""Build the object of one grid row. Depth: B2 perturb.build. Held-out: zero the counts of the fold's genes, recompute p_null, keep features (arm P drops expression columns at training
through the recipe's own --gin-feature-op zero_expr_stats). Run in chimera_env.
    python -m revision.tasks.D1_sparse_genes.objects --index I
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.common import gene_ids
from revision.tasks.B2_benchmark import perturb as P
from revision.tasks.D1_sparse_genes import rules as RU


def fold_genes(cohort: str, var_names, fold: str) -> np.ndarray:
    f = pd.read_csv(RU.TASK / "folds.tsv", sep="\t", dtype=str).set_index("fold")
    chroms = set(f.loc[fold, "chromosomes"].split(","))
    ch = pd.read_csv(RU.REPO / "results/revision/X1/genes/chromosome_ensembl110.csv", dtype={"chrom": str}).set_index("hsapiens_gene_id")["chrom"]
    ids = pd.Series(np.asarray(gene_ids.ensembl(cohort, var_names)))
    return ids.map(ch).isin(chroms).to_numpy()


def build_heldout(row: dict, out_path) -> dict:
    cohort, fold = row["cohort"], row["setting"][2]
    cfg = RU.B2.ALL_COHORTS[cohort]
    a = P.read_base(cohort)
    X0 = P.csr(a.X).astype(np.float64)
    hold = fold_genes(cohort, a.var_names, fold)
    X1 = (X0 @ sp.diags((~hold).astype(float))).tocsr()
    X1.eliminate_zeros()
    L = np.asarray(X1.sum(1)).ravel()
    keep = L >= RU.B2.MIN_LIB
    spc = a.obs["species"].astype(str).to_numpy()
    b = a[keep].copy()
    X1k = X1[keep].tocsr()
    b.X = X1k.astype(np.float32)
    b.uns["p_null"] = P.p_null(X1k, spc[keep])
    b.var["d1_held_out"] = hold
    b.write_h5ad(out_path)
    from revision.common.check_cohort import check as cc

    errs = cc(str(out_path), batch_key=cfg["batch_key"], celltype_key=cfg["model_key"])
    if errs:
        raise AssertionError(f"check_cohort: {errs}")
    return dict(cohort=cohort, setting=row["setting"], rep=int(row["rep"]), n_held=int(hold.sum()), n_genes=int(len(hold)), n_cells=int(keep.sum()),
                n_cells_dropped=int((~keep).sum()), umi_held_share=float(1 - X1.sum() / X0.sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    RU.register()
    row = pd.read_csv(RU.TASK / "grid.tsv", sep="\t", keep_default_na=False).iloc[a.index].to_dict()
    RU.H5AD_DIR.mkdir(parents=True, exist_ok=True)
    path = RU.H5AD_DIR / f"{row['run']}.h5ad"
    tmp = path.with_suffix(".tmp.h5ad")
    if row["family"] == "heldout":
        info = build_heldout(row, tmp)
    else:
        _, info = P.build(row, out_path=tmp)
    tmp.replace(path)
    (RU.RES / "objects").mkdir(parents=True, exist_ok=True)
    (RU.RES / "objects" / f"{row['run']}.json").write_text(json.dumps(info, indent=1, default=str))
    print(path, {k: v for k, v in info.items() if k != "n_by_label"}, flush=True)


if __name__ == "__main__":
    main()
