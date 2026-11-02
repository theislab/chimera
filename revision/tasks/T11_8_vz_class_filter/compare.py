"""T11.8: apply the README decision rule: class-filtered models against the current Van Zyl models.

Reads alpha and prior of each run only; no gene list, no gold. Writes
``results/revision/T11/class_filter/compare.{json,tsv}``.

    PYTHONPATH=<chimera repo> python revision/tasks/T11_8_vz_class_filter/compare.py
"""

from __future__ import annotations

import itertools
import json
import sys

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import controls as K
from revision.tasks.T11_5_vz_genes import vz

OUT = vz.OUT / "class_filter"
F50 = vz.ROOT / "data" / "revision" / "van_zyl_f50"
WINNER = int(json.loads((vz.ROOT / "results/revision/T11/count_rule/selection.json").read_text())["winner_grid"])
RUNS = {"full": (f"rev-vz-f50-g{WINNER}", "van_zyl_unified.h5ad"), "hmouse": ("rev-vz-f50-hmouse", "van_zyl_hmouse.h5ad")}
SEEDS_TRY = range(42, 52)
N_NEEDED = 2


def delta(run: str, var_names) -> pd.Series:
    with np.load(vz.RUNS / run / "final_state.npz") as z:
        return pd.Series(z["alpha"].astype(float) - z["prior"].astype(float), index=var_names)


def filtered_seeds(name: str) -> list[str]:
    """The first seeds that pass health, in seed order (README: a failing seed is replaced by the next)."""
    t = []
    for s in SEEDS_TRY:
        d = vz.RUNS / f"{name}-s{s}"
        if not d.exists():
            continue
        g = vz.gate(d)
        up, dn = vz.raw_tails(f"{name}-s{s}") if g["finished"] else (0.0, 0.0)
        ok = g["finished"] and g.get("no_nan") and g.get("n_active_z") == 10 and up >= 0.02 and dn >= 0.02 \
            and g["sat_high"] <= vz.SAT_MAX_SEED and g["sat_low"] <= vz.SAT_MAX_SEED
        print(f"{name}-s{s}", {k: g.get(k) for k in ("finished", "no_nan", "sat_high", "n_active_z", "tail_up", "tail_dn", "last_epoch")},
              "health:", bool(ok), flush=True)
        if ok and len(t) < N_NEEDED:
            t.append(f"{name}-s{s}")
    return t


def mean_s(pairs) -> float:
    return float(np.mean([spearmanr(a, b)[0] for a, b in pairs]))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for ds, (name, h5) in RUNS.items():
        cur_a = vz.read_h5ad(ds, backed=True)
        flt_a = ad.read_h5ad(F50 / h5, backed="r")
        shared = cur_a.var_names.intersection(flt_a.var_names)
        cur = {r: delta(r, cur_a.var_names)[shared] for r in vz.seeds_in_D(ds).values()}
        fruns = filtered_seeds(name)
        if len(fruns) < N_NEEDED:
            raise SystemExit(f"{name}: {len(fruns)} healthy seeds, need {N_NEEDED}")
        flt = {r: delta(r, flt_a.var_names)[shared] for r in fruns}
        A_cross = mean_s((a, b) for a in flt.values() for b in cur.values())
        A_within = mean_s(itertools.combinations(cur.values(), 2))
        A_within_f = mean_s(itertools.combinations(flt.values(), 2))
        g = pd.read_parquet(vz.OUT / "readout" / f"genes_{ds}.parquet").set_index("var_name").loc[shared]
        ok = g[K.COVARIATE_COLUMNS].notna().all(1).to_numpy()
        Z = K.design(g.loc[ok], K.COVARIATE_COLUMNS)

        def res(x):
            return K.residual_rank(x.to_numpy()[ok], Z)

        rc = float(np.mean([np.corrcoef(res(a), res(b))[0, 1] for a in flt.values() for b in cur.values()]))
        rw = float(np.mean([np.corrcoef(res(a), res(b))[0, 1] for a, b in itertools.combinations(cur.values(), 2)]))
        m_f, m_c = pd.concat(flt.values(), axis=1).mean(1), pd.concat(cur.values(), axis=1).mean(1)
        rows.append(dict(dataset=ds, n_shared_genes=len(shared), n_current=cur_a.n_vars, n_filtered=flt_a.n_vars,
                         current_seeds=list(cur), filtered_seeds=fruns, A_cross=A_cross, A_within=A_within,
                         keep_current=bool(A_cross >= A_within), A_within_filtered=A_within_f,
                         spearman_seed_means=float(spearmanr(m_f, m_c)[0]), residual_cross=rc, residual_within=rw))
    keep = all(r["keep_current"] for r in rows)
    vz.write_json(OUT / "compare.json", dict(rows=rows, keep_current_models=keep))
    pd.DataFrame(rows).to_csv(OUT / "compare.tsv", sep="\t", index=False)
    print(json.dumps(dict(rows=rows, keep_current_models=keep), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
