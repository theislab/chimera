"""Z3: the T4.3 scoreboard (with the T5.6 tool scalars, as results/analysis/T5_6/scoreboard) on free-gate consensus D,
beside the frozen row and a like-for-like canonical line. T4.3 code imported unchanged (units, RNG streams by unit id,
draws 10,000, permutations 10,000); only the seed runs behind D, S_ema, D_raw and the per-seed matrix are swapped, and
units on count-supported and masked genes are added (Family 3). Frozen folders are only read.

Lines (rules.py, lines.json): frozen (check: must reproduce T4_3 and T5_6/scoreboard), free5, can5, can_state. Testis 3-sp stays canonical.
Baselines, covariates, tau, ESM blocks, golds and B1z (canonical latent clusters) stay frozen: the objects are the
canonical objects. B3b is refitted on each line's D by T4.3's own code.

    PYTHONPATH=$PWD python -m analysis.tasks.Z3_free_gate.scoreboard --line free5 --workers 38
Output: results/analysis/Z3/scoreboard_<line>/
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.tasks.T4_3_scoreboard import inputs as I
from analysis.tasks.T4_3_scoreboard import run as R
from analysis.tasks.Z3_free_gate import rules as RU


def shim() -> Path:
    """R.RES stand-in: frozen golds, T2.5, input cache and baseline split halves; DONE present; no consensus-D B3b file
    (B3b is refitted on each line's D, so T4.3's equality check against the frozen consensus does not apply)."""
    d = RU.OUT / "shim"
    (d / "T4_3").mkdir(parents=True, exist_ok=True)
    (d / "T3_3").mkdir(exist_ok=True)
    for src, dst in ((RU.RES / "T4_3/golds", d / "T4_3/golds"), (RU.RES / "T2_5", d / "T2_5"),
                     (RU.RES / "T4_3/cache", d / "T4_3/cache"), (RU.RES / "T4_3/baseline_splits", d / "T4_3/baseline_splits")):
        if not dst.exists():
            dst.symlink_to(src)
    (d / "T3_3/DONE").touch()
    return d


def healthy_runs(line: str, canonical_4sp: bool = False) -> dict:
    """canonical_4sp: interim run before the 4-sp free seeds exist; 4-sp rows stay canonical (labelled)."""
    L = json.loads((RU.OUT / "lines.json").read_text())
    coh = [c for c in RU.COHORTS if not (canonical_4sp and c == "liv_4sp")]
    out = {c: {int(s): r for s, r in L[line][c].items()} for c in coh}
    for c, v in out.items():
        if len(v) != 5 and not line.endswith("_int"):
            raise SystemExit(f"{c} {line}: {len(v)} seeds, need 5")
    return out


def support_masks(cohorts=RU.COHORTS) -> dict:
    out = {}
    for c in cohorts:
        t = pd.read_csv(RU.OUT / "consensus" / f"{c}.csv")
        if not {"S_free5", "S_can5"} <= set(t.columns):
            continue  # frozen check before the free lines exist
        out[c] = pd.Series(((t.S_free5.abs() >= RU.EPS) & (t.S_can5.abs() >= RU.EPS)).to_numpy(), index=t.var_name.values)
    return out


def patch(line: str, canonical_4sp: bool = False) -> None:
    R.INTEGRATED_SEEDS = {}  # its unit needs canonical seeds 47, 48, 51 on the frozen 10-seed set; not part of a 5-seed line
    runs = None if line == "frozen" else healthy_runs(line, canonical_4sp)
    sup = {} if line == "frozen" else support_masks(tuple(runs))
    o_load = I.load
    if runs is not None:
        I.seed_runs_available = lambda cohort: runs[cohort]

    def load(cohort, cache_dir, consensus=True):
        swap = runs is not None and cohort in runs
        d = o_load(cohort, cache_dir, consensus=not swap)  # consensus=False: D = mean over the swapped seeds, median-centred
        t = d["tab"]
        if cohort in sup:
            cs = sup[cohort].reindex(t["var_name"]).to_numpy()
            t["count_supported"] = cs == True  # noqa: E712
            t["masked_either"] = cs == False  # noqa: E712
            cons = pd.read_csv(RU.OUT / "consensus" / f"{cohort}.csv").set_index("var_name")
            for k in ("free5", "can5", "can_state", "free_int", "can_int"):
                if f"D_{k}" not in cons:
                    continue
                t[f"D_{k}"] = cons[f"D_{k}"].reindex(t["var_name"]).to_numpy()
                t[f"S_{k}"] = cons[f"S_{k}"].reindex(t["var_name"]).to_numpy()
            fz = pd.read_csv(RU.RES / f"T3_3/{cohort}__consensus.csv").set_index("var_name")
            t["D_frozen"] = fz["D"].reindex(t["var_name"]).to_numpy()
            t["S_frozen"] = fz["S_ema"].reindex(t["var_name"]).to_numpy()
            if swap and not np.allclose(t["D"].to_numpy(), t[f"D_{line}"].to_numpy(), atol=1e-9):
                raise AssertionError(f"{cohort}: line D differs from the health.py consensus")
        return d

    I.load = load
    R.RES = shim()
    tools = pd.read_csv(RU.TOOLS)
    tcols = [c for c in tools.columns if c not in ("hsapiens_gene_id", "cohort")]
    o_prep, o_build = R.prepare, R.build_units

    def prepare(mode, null_seed):
        o_prep(mode, null_seed)
        for c in ("liv_hmouse", "test_hmouse"):
            D = R.DATA[c]
            use = (42, 43, 44) if line == "frozen" else tuple(sorted(D["seeds"])[:3])
            idx = [D["seeds"].index(s) for s in use]
            D["tab"]["D3"] = D["M"][:, idx].mean(1)
            per = tools[tools["cohort"] == c].drop(columns="cohort").set_index("hsapiens_gene_id")
            D["tab"] = D["tab"].join(per, on="hsapiens_gene_id")
            D["meta"]["D3_seeds"] = list(use)

    def build_units(_tools):
        U = o_build(tools)
        if line == "frozen":
            return U  # reproduction check: the T5_6/scoreboard units only
        for h, v in R.PRIMARY.items():
            c, g, cov = v["cohort"], v["gold"], v["cov"]
            U.append(R.unit(f"{h}:cs:D", "F3", "count_supported", h, c, g, cov, ["D"], subset="count_supported", perm="one"))
            U.append(R.unit(f"{h}:masked:D", "F3", "masked_either_line", h, c, g, cov, ["D"], subset="masked_either"))
            U.append(R.unit(f"{h}:cs:S_ema", "F3", "count_supported_S_ema", h, c, g, cov, ["S_ema"], subset="count_supported"))
            U.append(R.unit(f"{h}:cs:h2h", "F3", "head_to_head_count_supported", h, c, g, cov, ["D"] + R.J,
                            subset="count_supported"))
            for t in tcols:
                U.append(R.unit(f"{h}:cs:tool:{t}", "F3", "tool_vs_D3_count_supported", h, c, g, cov, ["D3", t],
                                subset="count_supported"))
            if line == "free5":  # paired free vs canonical on the same genes, one joint block bootstrap
                for uni in (None, "count_supported"):
                    tag = "cs" if uni else "all"
                    U.append(R.unit(f"{h}:pair_{tag}:can5", "F3", f"free_vs_can5_{tag}", h, c, g, cov, ["D_free5", "D_can5"], subset=uni))
                    U.append(R.unit(f"{h}:pair_{tag}:frozen", "F3", f"free_vs_frozen_{tag}", h, c, g, cov, ["D_free5", "D_frozen"], subset=uni))
                    U.append(R.unit(f"{h}:pair_{tag}:S_free5", "F3", f"free_vs_own_S_{tag}", h, c, g, cov, ["D_free5", "S_free5"], subset=uni))
                    if c == "liv_hmouse":
                        U.append(R.unit(f"{h}:pair_{tag}:int", "F3", f"free_int_vs_can_int_{tag}", h, c, g, cov, ["D_free_int", "D_can_int"], subset=uni))
                        U.append(R.unit(f"{h}:pair_{tag}:can_state", "F3", f"free_vs_can_state_{tag}", h, c, g, cov, ["D_free5", "D_can_state"], subset=uni))
                    if R.MULTI[h] == "liv_4sp" and not canonical_4sp:
                        U.append(R.unit(f"{h}:multi_pair_{tag}:can5", "F3", f"multi_free_vs_can5_{tag}", h, "liv_4sp", g, cov, ["D_free5", "D_can5"], subset=uni))
            if R.MULTI[h] == "liv_4sp" and not canonical_4sp:
                U.append(R.unit(f"{h}:multi_cs:D", "F3", "multi_species_estimate_count_supported", h, "liv_4sp", g, cov,
                                ["D"], subset="count_supported", perm="one"))
        for tis, sets in R.SETS.items():
            c = R.SET_COHORT[tis]
            for s in sets:
                d = "above" if s.startswith("C") else "below"
                U.append(dict(kind="set", id=f"set:{tis}:{s}:D:cs", family="F3", tissue=tis, cohort=c, set=s, score="D",
                              strata="pi_expr", direction=d, perm=True, subset="count_supported"))
        return U

    R.prepare, R.build_units = prepare, build_units


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--line", choices=["frozen", "free5", "can5", "can_state", "free_int", "can_int"], required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--only", default=None)
    ap.add_argument("--canonical-4sp", action="store_true", help="interim: 4-sp rows canonical (free 4-sp not trained yet)")
    a = ap.parse_args()
    patch(a.line, a.canonical_4sp)
    out = RU.OUT / (f"scoreboard_{a.line}" + ("_c4sp" if a.canonical_4sp else ""))
    argv = ["--mode", "real", "--outdir", str(out), "--workers", str(a.workers)]
    if a.only:
        argv += ["--only", a.only]
    R.main(argv)
    (out / "z3_line.json").write_text(json.dumps(dict(line=a.line, canonical_4sp=a.canonical_4sp, seeds={c: R.DATA[c]["meta"]["seeds"] for c in RU.COHORTS},
                                                      D3={c: R.DATA[c]["meta"].get("D3_seeds") for c in ("liv_hmouse", "test_hmouse")}),
                                                 indent=1, default=str))


if __name__ == "__main__":
    main()
