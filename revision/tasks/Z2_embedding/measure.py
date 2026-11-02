"""Within-label metrics, one latent per call so arrays can share a cohort's cached targets.
python -m revision.tasks.Z2_embedding.measure prep <cohort>
python -m revision.tasks.Z2_embedding.measure cohort <cohort>           every latent of the cohort, sequentially, skipping finished ones
python -m revision.tasks.Z2_embedding.measure run <cohort> <run>        one factorial run
Output: results/revision/Z2/measure/<cohort>__<latent>.parquet"""
from __future__ import annotations

import sys

import pandas as pd

from revision.tasks.Z2_embedding import common as K, rules as RU

OUT = RU.OUT / "measure"


def one(c, name, method, seed, path):
    f = OUT / f"{c}__{name}.parquet"
    if f.exists():
        return
    z = K.load_z(path)
    T = K.targets(c)
    if z.shape[0] != len(T["spc"]):
        print("skip, cells differ", c, name, z.shape, flush=True)
        return
    d = K.all_rows(c, z)
    d.insert(0, "seed", seed)
    d.insert(0, "method", method)
    d.insert(0, "latent", name)
    d.insert(0, "cohort", c)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(f)
    print(c, name, len(d), flush=True)


def main():
    k = sys.argv[1]
    if k == "prep":
        K.prep(sys.argv[2])
    elif k == "cohort":
        c = sys.argv[2]
        K.prep(c)
        for name, (m, s, p) in K.latents(c).items():
            one(c, name, m, s, p)
    elif k == "run":
        c, run = sys.argv[2], sys.argv[3]
        one(c, run, "Chimera-factorial", int(run.rsplit("-s", 1)[1]), RU.RUNS / run / "final_state.npz")


if __name__ == "__main__":
    main()
