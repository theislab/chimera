"""EXPLORATORY: incremental information (part 1) and label degradation (part 2). Rules in rules.py.

Sections: repro (X4_where_differs numbers first), incremental, degradation. Writes results/analysis/X6/*.csv.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.X4_where_differs import common as CM
from analysis.tasks.X6_adds_beyond import rules as RU
from analysis.tasks.X6_adds_beyond.degrade import CACHE, OUT, useed

CM.N_JOBS = 16


def write(df: pd.DataFrame, name: str) -> None:
    K.write_csv(df, OUT / name)
    print(f"  wrote {name} ({len(df)} rows)", flush=True)


def genes(c: str, degraded: bool = False) -> pd.DataFrame:
    g = CM.load(c)
    sd = pd.read_parquet(CM.CACHE / f"Sdir__{c}.parquet").set_index("var_name")
    g = g.join(sd[["Sdir"]], on="var_name")
    if degraded:
        d = pd.read_parquet(CACHE / f"degraded__{c}.parquet").set_index("var_name")
        g = g.join(d, on="var_name")
    return g


def readers(c: str) -> list[str]:
    return ["D", "S", "Sdir"] + (["D_raw"] if c.startswith("test") else [])


# ---------------------------------------------------------------- reproduction of X4_where_differs
def repro():
    units = []
    for c in RU.COHORTS:
        g = genes(c)
        ok = g[RU.X4_PANEL + ["D_raw", "S", "Sdir"]].notna().all(1).to_numpy()
        rd = ["D_raw", "S", "Sdir"] + (["D"] if c.startswith("test") else [])
        for gold, _, covs in RU.GOLDS[c]:
            units.append(dict(g=g, gold=gold, covs=covs, scores=rd, extra_spline=RU.X4_PANEL, rows=ok,
                              label=dict(cohort=c, analysis="given_panel", panel="B1b+B1c+B1z+B2b")))
    new = CM.run_units(units)
    old = pd.read_csv(CM.OUT / "adds_rho.csv")
    old = old[(old.analysis == "given_panel") & (old.panel == "B1b+B1c+B1z+B2b")]
    m = new.merge(old, on=["cohort", "gold", "score"], suffixes=("", "_X4"))
    m["diff"] = m.rho_p - m.rho_p_X4
    write(m[["cohort", "gold", "score", "n", "n_X4", "rho_p", "rho_p_X4", "diff", "ci_lo", "ci_lo_X4", "ci_hi", "ci_hi_X4"]],
          "repro_X4.csv")
    assert np.abs(m["diff"]).max() < 1e-9, "X4 numbers not reproduced"
    print(m[["cohort", "gold", "score", "n", "rho_p", "rho_p_X4"]].to_string(), flush=True)


# ---------------------------------------------------------------- part 1
def incremental():
    units = []
    for c in RU.COHORTS:
        g = genes(c)
        rd = readers(c)
        ok = g[RU.PANEL + ["D", "D_raw", "S", "Sdir"]].notna().all(1).to_numpy()
        for variant, extra in (("C*", []), ("C*+dNdS+LOEUF", RU.SEQ)):
            rows = ok & g[extra].notna().all(1).to_numpy() if extra else ok
            for gold, _, covs in RU.GOLDS[c]:
                base = dict(cohort=c, variant=variant)
                units.append(dict(g=g, gold=gold, covs=covs, scores=rd + RU.PANEL, extra_spline=extra, rows=rows,
                                  label=dict(base, given="none")))
                units.append(dict(g=g, gold=gold, covs=covs, scores=rd, extra_spline=extra + RU.PANEL, rows=rows,
                                  label=dict(base, given="all simple baselines")))
                for x, y in (("D", "S"), ("S", "D"), ("S", "Sdir"), ("Sdir", "S")) + ((("D_raw", "S"),) if c.startswith("test") else ()):
                    units.append(dict(g=g, gold=gold, covs=covs, scores=[x], extra_spline=extra + RU.PANEL + [y], rows=rows,
                                      label=dict(base, given=f"all simple baselines + {y}")))
                for b in RU.PANEL:
                    units.append(dict(g=g, gold=gold, covs=covs, scores=[b], extra_spline=extra + [p for p in RU.PANEL if p != b],
                                      rows=rows, label=dict(base, given="other simple baselines")))
    write(CM.run_units(units), "incremental_rho.csv")


# ---------------------------------------------------------------- part 2
def joint(g, gold, covs, scores, rows, draws=RU.DRAWS):
    """rho_p of every score with gold on rows, all on the same block-bootstrap draws."""
    T = g.loc[rows]
    R = np.column_stack([S.midrank(T[gold].to_numpy(float))] + [S.midrank(T[s].to_numpy(float)) for s in scores])
    Z = S.design(T, covs)
    est = S.rho_p_matrix(R, Z)[1:]
    rng = np.random.default_rng(useed("joint", gold, len(T)))
    B = S.bootstrap(R, Z, draws, rng, T["block"].to_numpy())[:, 1:]
    return pd.Series(est, index=scores), pd.DataFrame(B, columns=scores)


def _row(base, name, est, draws):
    lo, hi = S.pct_ci(draws.to_numpy())
    return dict(base, score=name, rho_p=float(est), ci_lo=lo, ci_hi=hi)


def degradation():
    conds = RU.conditions()
    est_rows, delta_rows, units = [], [], []
    for c in RU.COHORTS:
        g = genes(c, degraded=True)
        chim = ["D", "S", "B1z_raw"] + (["D_raw"] if c.startswith("test") else [])
        deg = [f"{k}__{cd}" for cd in conds for k in RU.DEG_SCORES]
        for gold, _, covs in RU.GOLDS[c]:
            if gold not in RU.DEG_GOLDS[c]:
                continue
            need = [gold] + covs + chim + deg
            rows = g[need].replace([np.inf, -np.inf], np.nan).notna().all(1).to_numpy()
            scores = chim + deg
            est, B = joint(g, gold, covs, scores, rows)
            # noise replicate means, per draw
            for k in RU.NOISE:
                for sc in RU.DEG_SCORES:
                    cols = [f"{sc}__{k}_r{r}" for r in range(RU.NOISE_REPS)]
                    est[f"{sc}__{k}_mean"] = est[cols].mean()
                    B[f"{sc}__{k}_mean"] = B[cols].mean(1)
            base = dict(cohort=c, gold=gold, n=int(rows.sum()), universe="U_deg")
            for s in est.index:
                sc, _, cd = s.partition("__")
                est_rows.append(_row(dict(base, condition=cd or "model"), sc, est[s], B[s]))
            for ref in ("S", "D") + (("D_raw",) if c.startswith("test") else ()) + ("B1z_raw",):
                for s in est.index:
                    if "__" not in s:
                        continue
                    sc, _, cd = s.partition("__")
                    lo, hi = S.pct_ci((B[ref] - B[s]).to_numpy())
                    delta_rows.append(dict(base, ref=ref, condition=cd, score=sc, delta=float(est[ref] - est[s]),
                                           delta_lo=lo, delta_hi=hi))
            # own-universe sensitivity, each score alone
            for s in scores:
                units.append(dict(g=g, gold=gold, covs=covs, scores=[s], label=dict(cohort=c, universe="own", analysis="own")))
            # secondary: S and D given the condition's degraded panel + B1z + B3a
            for cd in conds:
                pan = [f"B1b_raw__{cd}", f"B1c_raw__{cd}", f"B2b__{cd}", "B1z_raw", "B3a"]
                units.append(dict(g=g, gold=gold, covs=covs, scores=["S", "D"] + (["D_raw"] if c.startswith("test") else []),
                                  extra_spline=pan, rows=rows,
                                  label=dict(cohort=c, universe="U_deg", analysis="given_degraded_panel", condition=cd)))
                units.append(dict(g=g, gold=gold, covs=covs, scores=[f"Sdir__{cd}"], extra_spline=pan, rows=rows,
                                  label=dict(cohort=c, universe="U_deg", analysis="given_degraded_panel", condition=cd)))
    write(pd.DataFrame(est_rows), "degradation_rho.csv")
    write(pd.DataFrame(delta_rows), "degradation_delta.csv")
    write(CM.run_units(units), "degradation_secondary.csv")


SECTIONS = dict(repro=repro, incremental=incremental, degradation=degradation)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--sections", default="repro,incremental,degradation")
    a = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for s in a.sections.split(","):
        print("section", s, flush=True)
        SECTIONS[s]()


if __name__ == "__main__":
    main()
