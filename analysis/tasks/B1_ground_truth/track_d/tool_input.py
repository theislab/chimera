"""B1 Track D: tool inputs for the timing jobs, in the tool's env. Full object: the T5 or B1 inputs; subsample: the same cells as Chimera's object."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1].parent / "T5_2_adapters"))
import cohorts as C5  # noqa: E402

sys.path.insert(0, str(HERE.parents[1]))
import importlib.util as iu  # noqa: E402

REPO = HERE.parents[4]
OUT = REPO / "data" / "analysis" / "B1" / "trackD"
STEM = {"liv_hmouse": "camex_human_mouse", "test_hmouse": "camex_testis_human_mouse", "kidney": "kidney_healthy_train"}


def main():
    tool, cohort, obj, dest = sys.argv[1:5]
    dest = Path(dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    os.environ["B1_TOOL"] = "trackd"
    spec = iu.spec_from_file_location("b1tools", HERE.parents[1] / "tools.py")
    tools = iu.module_from_spec(spec)
    spec.loader.exec_module(tools)
    keep = None
    if obj != "full":
        keep = set((OUT / f"{cohort}_{obj}.obs.txt").read_text().split("\n"))
    # one altered.h5ad-free path: write from the base object through the same prep code
    wd = tools.wdir(dest.parent.name)
    wd.mkdir(parents=True, exist_ok=True)
    base = C5.PROCESSED_OVERRIDE.get(STEM[cohort], C5.PROCESSED / f"{STEM[cohort]}.h5ad")
    a = ad.read_h5ad(base)
    if keep is not None:
        a = a[np.isin(a.obs_names.astype(str), list(keep))].copy()
    a.write_h5ad(wd / "altered.h5ad")
    row = dict(cohort=cohort, run=dest.parent.name)
    if tool == "saturn":
        tools.prep_saturn(row)
        src = wd / "saturn_in"
    else:
        if cohort != "kidney" and keep is not None:
            # T5 raw files subset to the cells
            s5 = C5.INPUTS / STEM[cohort] / "camex"
            src = wd / "camex_in"
            src.mkdir(exist_ok=True)
            for f in s5.glob("raw-*.h5ad"):
                r = ad.read_h5ad(f)
                r[np.isin(r.obs_names.astype(str), list(keep))].copy().write_h5ad(src / f.name)
            shutil.copy(list(s5.glob("gene_matches_*.csv"))[0], src)
            (src / "manifest.json").write_text(json.dumps({"files": {f.name: {} for f in src.glob("raw-*.h5ad")}}))
        else:
            tools.prep_camex(row)
            src = wd / "camex_in"
    for f in src.iterdir():
        shutil.move(str(f), dest / f.name)
    # run.csv of SATURN carries absolute paths of its input files
    if tool == "saturn":
        rc = pd.read_csv(dest / "run.csv")
        rc["path"] = [str(dest / Path(p).name) for p in rc["path"]]
        rc.to_csv(dest / "run.csv", index=False)
    shutil.rmtree(wd)


if __name__ == "__main__":
    main()
