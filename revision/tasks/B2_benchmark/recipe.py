"""Print the training arguments of one grid row as shell words: run seed fit args."""

from __future__ import annotations

import argparse
import shlex

import pandas as pd

from revision.tasks.B2_benchmark import rules as RU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--h5ad", default=None)
    a = ap.parse_args()
    r = pd.read_csv(a.manifest, sep="\t", keep_default_na=False).iloc[a.index]
    cfg = RU.COHORTS[r["cohort"]]
    nolabel = RU.SETTINGS[r["setting"]]["kind"] == "nolabel" and cfg["nolabel_recipe"]
    args = cfg["nolabel_recipe"] if nolabel else cfg["recipe"]
    if a.h5ad:  # after the object exists: the depth law arm appends the scaled anchor, from the formula alone
        if RU.SETTINGS[r["setting"]].get("depth_law"):
            import anndata as ad

            from revision.common import depth_anchor as DA
            from revision.tasks.B2_benchmark import perturb as P

            o, b = ad.read_h5ad(a.h5ad), P.read_base(r["cohort"])
            v, ratio = DA.scaled_anchor(DA.recipe_anchor(args), P.csr(o.X), o.obs["species"].astype(str).to_numpy(), P.csr(b.X), b.obs["species"].astype(str).to_numpy())
            print(f"--lambda-anchor {v:.8g}")
        return
    print(f"RUN={shlex.quote(r['run'])}; SEED={int(r['seed'])}; FIT={int(r['fit'])}; ARGS={shlex.quote(args)}")


if __name__ == "__main__":
    main()
