"""T7.9: calibrate the H-D4 calls against permuted condition labels (as T7.3).

Gene-set permutation ignores correlation between genes, so a raw call count means little alone.
For every DE-able type, the section 8b calls are recomputed with one species' fit replaced by a
permuted-label fit (``de.py --perm``) and the other kept: 20 human draws, 11 mouse relabellings.
Same universe, rank column (``log2fc_fixed``), 10,000 gene-set permutations and seeds as ``hd4.py``,
so only the labels differ. A count is reported with p = (1 + draws >= observed) / (1 + draws); it
counts as supported only at p < 0.05. Mouse draws: 11 relabellings give a floor of 1/12 = 0.083, so
that side cannot reach 0.05 and is reported as a range, not a test.

    PYTHONPATH=<chimera repo> python revision/tasks/T7_9_islet/hd4_null.py [--jobs 8]
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from revision.tasks.T7_9_islet import hd4 as G


def one(args):
    t, kind, path, n_perm, sets, o2o = args
    j = G.type_universe(t, o2o)
    rh = pd.Series(j["log2fc_fixed_h"].to_numpy(), index=j["gene_symbol_h"].to_numpy()).dropna()
    rm = pd.Series(j["log2fc_fixed_m"].to_numpy(), index=j["gene_symbol_h"].to_numpy()).dropna()
    p = pd.read_csv(path, sep="\t").dropna(subset=["log2fc_fixed"]).drop_duplicates("gene_id", keep=False)
    key = "human_id" if kind == "human" else "mouse_id"
    v = pd.Series(p["log2fc_fixed"].to_numpy(), index=p["gene_id"].astype(str).to_numpy())
    x = pd.Series(j[key].map(v).to_numpy(), index=j["gene_symbol_h"].to_numpy()).dropna()
    g = G.call_programs(x if kind == "human" else rh, x if kind == "mouse" else rm, sets, n_perm)
    return dict(cell_type=t, kind=kind, draw=path.name, n_human_sig=int((g["h_q"] < 0.05).sum()),
                n_transfer=int((g["call"] == "transfers").sum()), n_opposite=int((g["call"] == "opposite").sum()),
                nes_spearman=float(g[["h_nes", "m_nes"]].corr(method="spearman").iloc[0, 1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    types = json.loads((G.DE / "de_meta.json").read_text())["de_types"]
    sets = {k: G.read_gmt(v) for k, v in G.SETS.items()}
    o2o = G.one2one()
    jobs = [(t, kind, f, a.n_perm, sets, o2o) for t in types for kind in ("human", "mouse")
            for f in sorted((G.DE / "perm").glob(f"{kind}_*_{t}.tsv"))]
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(one, jobs))
    null = pd.DataFrame(rows)
    null.to_csv(G.OUT / "hd4_null.tsv", sep="\t", index=False)
    obs = pd.read_csv(G.OUT / "hd4_summary.tsv", sep="\t").set_index("cell_type")
    out = []
    for (t, kind), g in null.groupby(["cell_type", "kind"], sort=False):
        r = dict(cell_type=t, kind=kind, n_draws=len(g))
        for c in ("n_human_sig", "n_transfer", "n_opposite"):
            o = int(obs.loc[t, c])
            r.update({f"{c}_obs": o, f"{c}_null_median": float(np.median(g[c])), f"{c}_null_max": int(g[c].max()),
                      f"{c}_p": (1 + int((g[c] >= o).sum())) / (1 + len(g))})
        r["nes_spearman_obs"] = float(obs.loc[t, "nes_spearman_all_sets"])
        r["nes_spearman_null_sd"] = float(g["nes_spearman"].std())
        out.append(r)
    pd.DataFrame(out).to_csv(G.OUT / "hd4_null_summary.tsv", sep="\t", index=False)
    (G.OUT / "DONE_null").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
