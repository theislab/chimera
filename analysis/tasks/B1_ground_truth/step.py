"""One step of a grid row: build the altered object and its truth table. Run in chimera_env."""

from __future__ import annotations

import argparse
import json
import resource
import time

import numpy as np
import pandas as pd

from analysis.tasks.B1_ground_truth import alter as A
from analysis.tasks.B1_ground_truth import rules as RU


def build(row: dict) -> str:
    RU.H5AD_DIR.mkdir(parents=True, exist_ok=True)
    (RU.RES / "truth").mkdir(parents=True, exist_ok=True)
    path = RU.H5AD_DIR / f"{row['run']}.h5ad"
    t0 = time.time()
    tmp = path.with_suffix(".tmp.h5ad")
    if row["alt"] == "none":  # an arm's unaltered fit: the canonical object
        A.read_base(row["cohort"]).write_h5ad(tmp)
        tmp.replace(path)
        return str(path)
    b, truth = A.build_altered(row, out_path=tmp)
    tmp.replace(path)
    tp = RU.RES / "truth" / f"{row['run']}.parquet"
    truth.to_parquet(tp)
    # minimal realism: per-cell depth change, recorded for every run
    a = A.read_base(row["cohort"])
    L0, L1 = np.asarray(a.X.sum(1)).ravel(), np.asarray(b.X.sum(1)).ravel()
    spc = a.obs["species"].astype(str).to_numpy()
    q = {s: dict(median_ratio=float(np.median(L1[spc == s] / np.maximum(L0[spc == s], 1))),
                 p05_ratio=float(np.quantile(L1[spc == s] / np.maximum(L0[spc == s], 1), 0.05)),
                 p95_ratio=float(np.quantile(L1[spc == s] / np.maximum(L0[spc == s], 1), 0.95)))
         for s in sorted(set(spc))}
    (RU.RES / "qc_runs").mkdir(parents=True, exist_ok=True)
    (RU.RES / "qc_runs" / f"{row['run']}.json").write_text(json.dumps(
        dict(run=row["run"], library_ratio=q, n_pairs=len(truth), share_achieved=float(truth["share_achieved"].iloc[0]),
             elapsed_s=time.time() - t0, peak_rss_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2), indent=1))
    print(path, len(truth), flush=True)
    return str(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    row = pd.read_csv(a.manifest, sep="\t").iloc[a.index].to_dict()
    build(row)


if __name__ == "__main__":
    main()
