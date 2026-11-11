"""Per-gene figure: per-gene tables behind the panels, built with the harness of analysis/tasks/Z3_free_gate (no new statistic).

Reads only the outputs of Z3_free_gate and the frozen inputs of T4_3_scoreboard, through ``Z3_free_gate.scoreboard.patch`` and
the unchanged ``prepare``, ``universe``, ``design`` and ``rho_p_matrix`` of T4_3_scoreboard. Every rank correlation that a panel
draws is recomputed here and checked against ``results/analysis/Z3/compare_*.csv`` before it is written.

    PYTHONPATH=$PWD mamba run -n chimera_env python -m analysis.tasks.F2_per_gene.data
Output: results/analysis/F2/
  testis_level_residuals.csv   H2 universe: ranks and covariate residuals of the EVEE testis level gold, D_free5, D_can5, S_free5
  gate_vs_S.csv                per gene, count-supported flag, D and S of each gate (liver H+Mouse, testis H+Mouse)
  named_genes.csv              spermatocyte-peak (strict) and hepatocyte catabolic core genes with D under each gate and
                               its percentile among all genes of the cohort
  checks.csv                   recomputed value, value of Z3_free_gate, absolute difference
  summary.json                 numbers quoted in the caption and text that are not in a table of Z3_free_gate (rho of D with S,
                               D_free vs D_can)
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.T4_3_scoreboard import run as R
from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.Z3_free_gate import rules as RU
from analysis.tasks.Z3_free_gate import scoreboard as SB

OUT = RU.RES / "F2"
Z3 = RU.OUT


def cs_mask(cons: pd.DataFrame) -> np.ndarray:
    return ((cons.S_free5.abs() >= RU.EPS) & (cons.S_can5.abs() >= RU.EPS)).to_numpy()


def testis_level(checks: list) -> None:
    SB.patch("free5")
    R.CFG.update(draws=0, mode="real")
    R.prepare("real", 0)
    tab = R.DATA["test_hmouse"]["tab"]
    hyp = pd.read_csv(Z3 / "compare_hyp.csv")
    scores = ["D_free5", "D_can5", "S_free5"]
    out = {}
    for uni, sub in (("all", None), ("count_supported", "count_supported")):
        rows = R.universe(tab, "G_H2", "C", ["D_free5", "D_can5"], sub)
        T = tab.loc[rows]
        Rk = np.column_stack([S.midrank(T["G_H2"].to_numpy(float))] + [S.midrank(T[s].to_numpy(float)) for s in scores])
        Z = S.design(T, R.COVSETS["C"])
        E = S.resid(Rk, Z)
        rp = S.corr_first(E)[1:]
        for k, s in enumerate(scores):
            line = {"D_free5": "free5", "D_can5": "can5", "S_free5": "free5"}[s]
            u = uni if s != "S_free5" else f"{uni}_S_ema"
            ref = hyp[(hyp.hyp == "H2") & (hyp.universe == u) & (hyp.line == line)].iloc[0]
            sp = spearmanr(T[s], T["G_H2"])[0]
            checks.append(dict(item=f"H2 rho_p {uni} {s}", recomputed=rp[k], z3=ref.rho_p, abs_diff=abs(rp[k] - ref.rho_p)))
            checks.append(dict(item=f"H2 spearman {uni} {s}", recomputed=sp, z3=ref.spearman, abs_diff=abs(sp - ref.spearman)))
        if uni == "all":
            n = len(T)
            d = pd.DataFrame({"var_name": T.var_name.to_numpy(), "hsapiens_gene_id": T.hsapiens_gene_id.to_numpy(),
                              "count_supported": T.count_supported.to_numpy(), "rank_G": Rk[:, 0] / n,
                              "resid_G": E[:, 0] / n})
            for k, s in enumerate(scores):
                d[f"rank_{s}"] = Rk[:, k + 1] / n
                d[f"resid_{s}"] = E[:, k + 1] / n
            out = d
    out.to_csv(OUT / "testis_level_residuals.csv", index=False)


def gate_vs_s(summary: dict) -> None:
    frames = []
    for c in ("liv_hmouse", "test_hmouse", "liv_4sp"):
        cons = pd.read_csv(Z3 / "consensus" / f"{c}.csv")
        cs = cs_mask(cons)
        frames.append(pd.DataFrame(dict(cohort=c, var_name=cons.var_name, count_supported=cs, D_free5=cons.D_free5,
                                        S_free5=cons.S_free5, D_can5=cons.D_can5, S_can5=cons.S_can5)))
        summary[c] = dict(
            n_genes=int(len(cons)), n_count_supported=int(cs.sum()),
            rho_D_S_free5_cs=float(spearmanr(cons.D_free5[cs], cons.S_free5[cs])[0]),
            rho_D_S_can5_cs=float(spearmanr(cons.D_can5[cs], cons.S_can5[cs])[0]),
            rho_Dfree_Dcan_all=float(spearmanr(cons.D_free5, cons.D_can5)[0]),
            rho_Dfree_Dcan_cs=float(spearmanr(cons.D_free5[cs], cons.D_can5[cs])[0]),
            rho_Sfree_Scan_cs=float(spearmanr(cons.S_free5[cs], cons.S_can5[cs])[0]),
            sd_D_free5=float(cons.D_free5.std()), sd_D_can5=float(cons.D_can5.std()),
            masked_D_free5_max_abs_D_raw=float((cons.alpha_free5 - cons.prior_free5)[~cs & (cons.S_free5.abs() < RU.EPS)].abs().max()))
    pd.concat(frames, ignore_index=True).to_csv(OUT / "gate_vs_S.csv", index=False)


def named(summary: dict) -> None:
    from analysis.tasks.X4_where_differs import analyses as A

    rows = []
    g = A.prepare("test_hmouse")
    cons = pd.read_csv(Z3 / "consensus" / "test_hmouse.csv").set_index("var_name")
    cs = pd.Series(cs_mask(cons.reset_index()), index=cons.index)
    t = g[g.peak_strict == "SC"][["var_name", "symbol"]].copy()
    for k in ("D_free5", "D_can5", "S_free5", "S_can5"):
        t[k] = cons[k].reindex(t.var_name).to_numpy()
    t["count_supported"] = cs.reindex(t.var_name).to_numpy()
    for k in ("D_free5", "D_can5"):  # percentile among all genes of the cohort
        t[f"pct_{k}"] = cons[k].rank(pct=True).reindex(t.var_name).to_numpy()
    rows.append(t.assign(cohort="test_hmouse", set="SC_strict"))
    for c in ("liv_4sp", "liv_hmouse"):
        g = A.prepare(c)
        cons = pd.read_csv(Z3 / "consensus" / f"{c}.csv").set_index("var_name")
        cs = pd.Series(cs_mask(cons.reset_index()), index=cons.index)
        t = g[g.symbol.isin(A.PROG_CORE)][["var_name", "symbol"]].copy()
        for k in ("D_raw_free5", "D_raw_can5", "S_free5", "S_can5", "D_free5", "D_can5"):
            t[k] = cons[k].reindex(t.var_name).to_numpy()
        t["count_supported"] = cs.reindex(t.var_name).to_numpy()
        for k in ("D_raw_free5", "D_raw_can5"):
            t[f"pct_{k}"] = cons[k].rank(pct=True).reindex(t.var_name).to_numpy()
        rows.append(t.assign(cohort=c, set="core15"))
    pd.concat(rows, ignore_index=True).to_csv(OUT / "named_genes.csv", index=False)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks, summary = [], {}
    gate_vs_s(summary)
    named(summary)
    testis_level(checks)
    c = pd.DataFrame(checks)
    c.to_csv(OUT / "checks.csv", index=False)
    print(c.to_string())
    if c.abs_diff.max() > 1e-9:
        raise SystemExit("Z3 values not reproduced")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
