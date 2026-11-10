"""Print --lambda-anchor for an arm L row, by the formula of analysis/common/depth_anchor.py, and record it. Reads the object and the unperturbed cohort, no fit, no gold.
    python -m analysis.tasks.D1_sparse_genes.anchor_arg --index I --h5ad path
"""

from __future__ import annotations

import argparse
import json

import anndata as ad
import pandas as pd

from analysis.common import depth_anchor as DA
from analysis.tasks.B2_benchmark import perturb as P
from analysis.tasks.D1_sparse_genes import rules as RU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--h5ad", required=True)
    a = ap.parse_args()
    RU.register()
    r = pd.read_csv(RU.TASK / "grid.tsv", sep="\t", keep_default_na=False).iloc[a.index]
    cfg = RU.B2.COHORTS[r["cohort"]]
    o = ad.read_h5ad(a.h5ad)
    b = P.read_base(r["cohort"])
    an, ratio = DA.scaled_anchor(DA.recipe_anchor(cfg["recipe"]), P.csr(o.X), o.obs["species"].astype(str).to_numpy(), P.csr(b.X), b.obs["species"].astype(str).to_numpy())
    (RU.RES / "law").mkdir(exist_ok=True)
    (RU.RES / "law" / f"{r['run']}.json").write_text(json.dumps(dict(run=r["run"], recipe_anchor=DA.recipe_anchor(cfg["recipe"]) or DA.DEFAULT_ANCHOR, depth_ratio=ratio, anchor=an), indent=1))
    print(f"--lambda-anchor {an:.8f}")


if __name__ == "__main__":
    main()
