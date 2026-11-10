"""Recompute the per-gene tables of the harmonized cohorts from the archived run states.

Checks the two per-gene tables of every harmonized cohort against ``final_state.npz`` of the seeds they pool:

- canonical gate (protein-feature GIN, ``--config A``): ``results/analysis/H1/genes/<cohort>.parquet``,
  seeds = rows of ``results/analysis/H1/runs.tsv`` with ``kept`` true;
- per-gene (free) gate (``--config M``): ``results/analysis/Z4/h1/genes/<cohort>.parquet``,
  seeds = ``results/analysis/Z4/h1/kept_used.json``.

Recomputed from the states alone (no AnnData needed): ``alpha_s<seed>``, ``alpha``, ``pi_clamped``, ``pi_raw``,
``delta_alpha``, ``D``, ``D_raw``, ``S_ema`` and the flags ``count_support``, ``sign_D_stable``, ``sign_S_stable``,
``saturated_flag``, ``strict_support`` (definitions in ``analysis/tasks/H1_harmonise/rules.py``).
``pi_raw`` is ``conservation_prior_raw.npy`` of the run where present, else its prior; the tables took it from the
AnnData, which the builder checked to agree within 1e-5, so ``pi_raw`` and ``D_raw`` are compared at 2e-5 and every
other value at 1e-9; flags must agree on every gene.

Run from the repository root, inside the environment, after unpacking the bundles:

    python analysis/check_gene_tables.py [--out check.csv]

Exit code 0 when every table of both gates matches.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("CHIMERA_ROOT", Path(__file__).resolve().parents[1]))
RUNS = ROOT / "training" / "runs"
RES = ROOT / "results" / "analysis"
MASK_EPS, SAT_HI = 1e-6, 0.95  # analysis/tasks/H1_harmonise/rules.py
FLOATS = ["alpha", "pi_clamped", "delta_alpha", "D", "S_ema"]
RAW = ["pi_raw", "D_raw"]
FLAGS = ["count_support", "sign_D_stable", "sign_S_stable", "saturated_flag", "strict_support"]


def state(run: str) -> dict:
    d = RUNS / run
    with np.load(d / "final_state.npz", allow_pickle=False) as z:
        st = {k: np.asarray(z[k], dtype=float) for k in ("alpha", "prior", "S_g_ema")}
    f = d / "conservation_prior_raw.npy"
    st["raw"] = np.load(f).astype(float) if f.exists() else st["prior"]
    return st


def recompute(kept: dict[int, str]) -> pd.DataFrame:
    seeds = sorted(kept)
    S = {s: state(kept[s]) for s in seeds}
    A = np.stack([S[s]["alpha"] for s in seeds])
    P = np.stack([S[s]["prior"] for s in seeds])
    Sm = np.stack([S[s]["S_g_ema"] for s in seeds])
    if not np.allclose(P, P[0]):
        raise AssertionError("seeds anchor on different priors")
    g = pd.DataFrame({f"alpha_s{s}": A[i] for i, s in enumerate(seeds)})
    g["alpha"] = A.mean(0)
    g["pi_clamped"] = P[0]
    g["pi_raw"] = S[seeds[0]]["raw"]
    g["delta_alpha"] = g["alpha"] - g["pi_clamped"]
    g["D"] = g["delta_alpha"] - g["delta_alpha"].median()
    r = g["alpha"] - g["pi_raw"]
    g["D_raw"] = r - r.median()
    g["S_ema"] = Sm.mean(0)
    sup = np.abs(g["S_ema"].to_numpy()) >= MASK_EPS
    g["count_support"] = sup
    thr = np.median(np.abs(g["S_ema"].to_numpy()[sup]))
    Dm = A - P
    Ds = Dm - np.median(Dm, axis=1, keepdims=True)
    sign = np.sign(g["D"].to_numpy())
    g["sign_D_stable"] = (np.sign(Ds) == sign).all(0)
    g["sign_S_stable"] = (np.sign(Sm) == np.sign(g["S_ema"].to_numpy())).all(0)
    g["saturated_flag"] = (A > SAT_HI).any(0)
    s_ = g["S_ema"].to_numpy()
    g["strict_support"] = sup & (np.abs(s_) >= thr) & (np.sign(s_) == sign) & g["sign_D_stable"].to_numpy() & g["sign_S_stable"].to_numpy()
    return g


def compare(gate: str, cohort: str, kept: dict[int, str], table: Path) -> dict:
    t = pd.read_parquet(table)
    g = recompute(kept)
    if len(t) != len(g):
        raise AssertionError(f"{gate} {cohort}: {len(t)} rows vs {len(g)} genes in the states")
    cols = FLOATS + [c for c in g if c.startswith("alpha_s")]
    if missing := [c for c in cols + RAW + FLAGS if c not in t]:
        raise AssertionError(f"{gate} {cohort}: table lacks {missing}")
    err = max(float(np.nanmax(np.abs(t[c].to_numpy(float) - g[c].to_numpy(float)))) for c in cols)
    err_raw = max(float(np.nanmax(np.abs(t[c].to_numpy(float) - g[c].to_numpy(float)))) for c in RAW)
    flag_diff = int(sum((t[c].astype(bool).to_numpy() != g[c].to_numpy(bool)).sum() for c in FLAGS))
    ok = err < 1e-9 and err_raw < 2e-5 and flag_diff == 0
    return dict(gate=gate, cohort=cohort, n_seeds=len(kept), runs=",".join(kept[s] for s in sorted(kept)), genes=len(t),
                max_abs_diff=err, max_abs_diff_pi_raw_D_raw=err_raw, flag_mismatches=flag_diff, ok=ok)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the per-table result as CSV")
    a = ap.parse_args()
    runs = pd.read_csv(RES / "H1" / "runs.tsv", sep="\t")
    runs = runs[runs.kept.astype(str) == "True"]
    jobs = [("canonical", c, {int(r.seed): r.run for r in g.itertuples()}, RES / "H1" / "genes" / f"{c}.parquet")
            for c, g in runs.groupby("cohort", sort=False)]
    free = json.loads((RES / "Z4" / "h1" / "kept_used.json").read_text())
    jobs += [("free", c, {int(s): r for s, r in k.items()}, RES / "Z4" / "h1" / "genes" / f"{c}.parquet") for c, k in free.items()]
    rows = []
    for gate, c, kept, table in jobs:
        r = compare(gate, c, kept, table)
        rows.append(r)
        print(f"{gate:9s} {c:22s} seeds={r['n_seeds']} genes={r['genes']:6d}  max |table - recomputed| = {r['max_abs_diff']:.1e}"
              f"  pi_raw/D_raw {r['max_abs_diff_pi_raw_D_raw']:.1e}  flag mismatches {r['flag_mismatches']}  {'ok' if r['ok'] else 'FAIL'}", flush=True)
    df = pd.DataFrame(rows)
    if a.out:
        df.to_csv(a.out, index=False)
    n = df.groupby("gate").size().to_dict()
    print(f"tables checked: {n}; failed: {int((~df.ok).sum())}")
    return 0 if df.ok.all() else 1


if __name__ == "__main__":
    sys.exit(main())
