"""G1 step 3: one guard-arm row. Builds the row's object with the owning task's own code (B2 perturb.build, through D1's registry for D1 settings), prints the training words: the owning recipe
unchanged, the depth-law anchor of revision/common/depth_anchor.py for law rows, and --clamp-z-logvar. The object is the same draw as the unguarded row of the same (cohort, setting, rep).
    python -m revision.tasks.G1_guard.step3_prepare --index I [--build]
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

import pandas as pd

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
H5 = REPO / "data" / "revision" / "G1" / "h5ad"
RES = REPO / "results" / "revision" / "G1"


def row(i: int) -> dict:
    return pd.read_csv(TASK / "step3_grid.tsv", sep="\t", keep_default_na=False, dtype=str).iloc[i].to_dict()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, required=True)
    ap.add_argument("--build", action="store_true")
    a = ap.parse_args()
    r = row(a.index)
    if r["kind"] in ("bm", "full"):
        print(f"RUN={shlex.quote(r['run'])}; ARGS={shlex.quote(r['args'] + ' --clamp-z-logvar')}; H5=; TRAIN=revision/tasks/T12_2b_debug/nan_probe.py")
        return
    if r["kind"] == "full_upper":  # upper-only clamp, experiment through upper_only.py, package option off
        print(f"RUN={shlex.quote(r['run'])}; ARGS={shlex.quote(r['args'])}; H5=; TRAIN=revision/tasks/G1_guard/upper_only.py")
        return
    from revision.tasks.B2_benchmark import perturb as P
    from revision.tasks.B2_benchmark import rules as B2
    from revision.tasks.D1_sparse_genes import rules as D1

    D1.register()
    h5 = H5 / f"{r['run']}.h5ad"
    if a.build and not h5.exists():
        H5.mkdir(parents=True, exist_ok=True)
        tmp = h5.with_suffix(".tmp.h5ad")
        b = dict(cohort=r["cohort"], setting=r["setting"], rep=int(r["rep"]), family="sampling")
        _, info = P.build(b, out_path=tmp)
        tmp.replace(h5)
        (RES / "objects").mkdir(parents=True, exist_ok=True)
        (RES / "objects" / f"{r['run']}.json").write_text(json.dumps(info, indent=1, default=str))
        # the same draw as the owning task's object of this (cohort, setting, rep), where it recorded one
        own = (B2.RES if r["kind"] == "b2" else D1.RES) / "objects" / f"{r['kind']}-{r['cohort']}-{r['setting']}-r{r['rep']}.json"
        if own.exists():
            o = json.loads(own.read_text())
            for k in ("n_cells", "n_by_species", "n_by_label"):
                if k in o and json.loads(json.dumps(info.get(k), default=str)) != o[k]:
                    raise AssertionError(f"{r['run']}: object differs from {own.name} on {k}")
            print("object matches", own.name, flush=True)
    recipe = B2.COHORTS[r["cohort"]]["recipe"]
    args = recipe
    if r["form"] == "law":
        import anndata as ad

        from revision.common import depth_anchor as DA

        o, base = ad.read_h5ad(h5), P.read_base(r["cohort"])
        v, ratio = DA.scaled_anchor(DA.recipe_anchor(recipe), P.csr(o.X), o.obs["species"].astype(str).to_numpy(), P.csr(base.X), base.obs["species"].astype(str).to_numpy())
        (RES / "law").mkdir(parents=True, exist_ok=True)
        (RES / "law" / f"{r['run']}.json").write_text(json.dumps(dict(run=r["run"], recipe_anchor=DA.recipe_anchor(recipe) or DA.DEFAULT_ANCHOR, depth_ratio=ratio, anchor=v), indent=1))
        args += f" --lambda-anchor {v:.8g}"
    args += f" --seed {int(r['seed'])} --clamp-z-logvar"
    print(f"RUN={shlex.quote(r['run'])}; ARGS={shlex.quote(args)}; H5={shlex.quote(str(h5))}; TRAIN=revision/tasks/T12_2b_debug/nan_probe.py")


if __name__ == "__main__":
    main()
