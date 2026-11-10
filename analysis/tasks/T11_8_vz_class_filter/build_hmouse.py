"""Human + mouse subset of the class-filtered Van Zyl object, as T11_2b_vz_subsets (``build_subset``), and cohort checks."""

from __future__ import annotations

import sys

import anndata as ad

from analysis.common import gene_ids
from analysis.common.check_cohort import check
from analysis.tasks.T11_2b_vz_subsets.build_subsets import BATCH_KEY, CATEGORICAL, CT_KEY, build_subset

F50 = gene_ids.REPO_ROOT / "data" / "analysis" / "van_zyl_f50"


def main() -> int:
    errs = check(str(F50 / "van_zyl_unified.h5ad"), batch_key=BATCH_KEY, celltype_key=CT_KEY)
    if errs:
        raise SystemExit(f"check_cohort failed on the full object: {errs}")
    dest = F50 / "van_zyl_hmouse.h5ad"
    build_subset(source_path=F50 / "van_zyl_unified.h5ad", species=("human", "mouse"), dest_path=dest,
                 drop_empty_genes=True, min_cells_per_gene=3)
    a = ad.read_h5ad(dest)
    for c in CATEGORICAL:
        if c in a.obs:
            a.obs[c] = a.obs[c].astype(str).astype("category")
    a.write_h5ad(dest)
    errs = check(str(dest), batch_key=BATCH_KEY, celltype_key=CT_KEY)
    if errs:
        raise SystemExit(f"check_cohort failed on human + mouse: {errs}")
    print("OK", a.shape, a.obs["species"].value_counts().to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
