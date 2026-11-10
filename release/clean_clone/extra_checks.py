"""Clean-clone checks on the unpacked deposit, beyond the smoke test.

Run from the root of a checkout that holds the unpacked bundles, inside the environment::

    python -I release/clean_clone/extra_checks.py

1. **Training objects.** Every archived run names its training object in ``config.json``
   (``adata_path``). The check asks, for each run, whether that object is in the checkout.
   Runs on thinned or altered objects, which the task code rebuilds, and runs on the earlier
   bone-marrow build are counted apart, because those objects are left out of the deposit on
   purpose (``skipped.tsv`` of the data bundles).
2. **Model reload.** One model of each bundle family is loaded on CPU with its own object, and
   its per-gene alpha is compared with the archived ``final_state.npz`` of the run.

Exit code 0: no run misses its object and every reloaded model returns its archived alpha.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

RUNS = Path("training/runs")
# Objects that the deposit leaves out on purpose and the task code rebuilds.
REBUILT_BY_TASK_CODE = ("data/analysis/B1/", "data/analysis/B2/", "data/analysis/D1/", "data/analysis/G1/")
# Earlier bone-marrow build and its diagnostic objects (the cohort is data/analysis/bone_marrow/v2).
EARLIER_BUILD = ("bone_marrow/bone_marrow_unified.h5ad", "bone_marrow/debug/")
# One run per family: identifier-joined objects, per-gene gate, other cohorts, latent guard.
RELOAD = ["o2-liv_hmouse-s42", "o2-test_hmouse-s42", "o2-liv_hmonkey-s43", "z3-test_hmouse-m-s42",
          "z3-liv_hmouse-m-s47", "z4-kidney-m-s42", "z4-brain_dlpfc_hmac_own-m-s42", "rev-c1-brain-m1-bA-s42",
          "rev-c1-tumour-zilionis-ax0.000805-s42", "rev-g1-bm2-s44"]
MIN_RHO = 0.999


def current_path(recorded: str) -> str:
    """Path of a training object relative to this checkout.

    A few runs were started from the folder above the checkout and recorded ``chimera/data/...``,
    and runs on rebuilt objects recorded an absolute path; both are reduced to ``data/...``.
    """
    marker = "chimera/data/"
    return recorded[recorded.rindex(marker) + len("chimera/"):] if marker in recorded else recorded


def check_training_objects() -> int:
    """Print where each run's training object is; return the number of runs that miss theirs."""
    present, rebuilt = Counter(), Counter()
    earlier, missing = [], []
    for run in sorted(p for p in RUNS.iterdir() if (p / "config.json").is_file()):
        recorded = json.loads((run / "config.json").read_text()).get("adata_path")
        if not recorded:
            continue
        family = run.name.split("-")[0]
        path = current_path(recorded)
        if Path(path).is_file():
            present[family] += 1
        elif any(part in path for part in REBUILT_BY_TASK_CODE):
            rebuilt[family] += 1
        elif any(part in path for part in EARLIER_BUILD):
            earlier.append((run.name, recorded))
        else:
            missing.append((run.name, recorded))
    print("runs whose training object is present:", dict(present))
    print("runs on thinned or altered objects that the task code rebuilds:", dict(rebuilt))
    print(f"runs on the earlier bone-marrow build or its diagnostic objects (left out on purpose): {len(earlier)}")
    print(f"runs whose training object is missing: {len(missing)}")
    for name, recorded in missing:
        print("   ", name, recorded)
    return len(missing)


def check_model_reload() -> int:
    """Reload one model per family and compare alpha with the archived state; return the failures."""
    import anndata as ad
    import numpy as np
    from scipy.stats import spearmanr

    sys.path.insert(0, os.getcwd())            # the checkout root (this script runs with python -I)
    from analysis.common.load_run import load_run

    failures = 0
    for name in RELOAD:
        run = RUNS / name
        if not run.is_dir():
            print(f"{name}: not in the checkout")
            failures += 1
            continue
        path = current_path(json.loads((run / "config.json").read_text())["adata_path"])
        model = load_run(run, ad.read_h5ad(path))     # canonical loader, plus the per-gene correction of --config M fits
        alpha = np.asarray(model.get_alpha(), float).ravel()
        archived = np.load(run / "final_state.npz", allow_pickle=False)["alpha"]
        rho = float(spearmanr(alpha, archived)[0])
        worst = float(np.max(np.abs(alpha - archived)))
        ok = rho >= MIN_RHO
        failures += not ok
        print(f"{name:40s} {path:60s} genes {len(archived):6d}  rho {rho:.5f}  max|diff| {worst:.1e}  "
              f"{'ok' if ok else 'FAIL'}", flush=True)
    return failures


if __name__ == "__main__":
    n_missing = check_training_objects()
    n_failed = check_model_reload()
    sys.exit(1 if (n_missing or n_failed) else 0)
