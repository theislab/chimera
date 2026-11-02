"""One grid row: build the object of a setting and write what was done to it. Run in chimera_env."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from revision.tasks.B2_benchmark import perturb as P
from revision.tasks.B2_benchmark import rules as RU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index", type=int, required=True)
    a = ap.parse_args()
    row = pd.read_csv(a.manifest, sep="\t").iloc[a.index].to_dict()
    RU.H5AD_DIR.mkdir(parents=True, exist_ok=True)
    path = RU.H5AD_DIR / f"{row['run']}.h5ad"
    tmp = path.with_suffix(".tmp.h5ad")
    _, info = P.build(row, out_path=tmp)
    tmp.replace(path)
    (RU.RES / "objects").mkdir(parents=True, exist_ok=True)
    (RU.RES / "objects" / f"{row['run']}.json").write_text(json.dumps(info, indent=1, default=str))
    print(path, info, flush=True)


if __name__ == "__main__":
    main()
