"""Z2 labelled sensitivity, coordinator's instruction after the decision that the canonical recipe stays: liver H+Mouse trained with label-conditional MMD, the 4 existing healthy seeds (42-45).
Reads golds; enters no status, row or modifier. T4.3 and T5.6 code imported unchanged (units, RNG streams by unit id, draws 10,000, permutations 10,000); only the liver H+Mouse D, S_ema and seed matrix are swapped.
Lines: frozen = the check that this harness returns the frozen values to 1e-9; cmmd4 = seeds 42-45; cmmd_no45 = seeds 42-44, the s45-free second line, stated beside. The frozen T4.3 and T5.6 folders are only read.
Comparators stay frozen, including B1z (clusters on the canonical latent). Reliability, ceilings: the plan's 2-vs-3 splits need 5 seeds, so absent here.
python -m revision.tasks.Z2_embedding.sens_cmmd --workers 14    Output: results/revision/Z2/sens_cmmd/"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd

from revision.common import controls as K
from revision.tasks.T4_3_scoreboard import decide as DEC
from revision.tasks.T4_3_scoreboard import run as T
from revision.tasks.T4_3_scoreboard import stats as S
from revision.tasks.Z2_embedding import rules as RU

OUT = RU.OUT / "sens_cmmd"
CMMD = {42: "z2-liv-cmmd-s42", 43: "z2-liv-cmmd-s43", 44: "z2-liv-cmmd-s44", 45: "z2-liv-cmmd-s45"}
TOOLS = RU.REPO / "results/revision/T5_6/tools_primary.csv"


def consensus(seeds: list[int]) -> dict:
    R = {s: K.read_run(CMMD[s]) for s in seeds}
    M = np.column_stack([R[s]["D"] for s in seeds])
    m = M.mean(1)
    A = np.column_stack([R[s]["alpha"] for s in seeds]).mean(1)
    return dict(M=M, D=m - np.median(m), alpha=A, S=np.column_stack([R[s]["S_g_ema"] for s in seeds]).mean(1))


def set_line(base: dict, line: str) -> None:
    D = T.DATA["liv_hmouse"]
    D.update(M=base["M"].copy(), seeds=list(base["seeds"]))
    tab = D["tab"]
    for c in ("D", "S_ema", "D_raw", "D3"):
        tab[c] = base[c]
    if line == "frozen":
        return
    seeds = [42, 43, 44, 45] if line == "cmmd4" else [42, 43, 44]
    c = consensus(seeds)
    tab["D"], tab["S_ema"] = c["D"], c["S"]
    draw = c["alpha"] - tab["prior_raw"].to_numpy()
    tab["D_raw"] = draw - np.median(draw)
    D.update(M=c["M"], seeds=seeds)
    tab["D3"] = consensus([42, 43, 44])["D"] + np.median(consensus([42, 43, 44])["M"].mean(1))  # as run.py: uncentred mean of seeds 42-44; ranks only


def units(tools: list[str]) -> list[dict]:
    U = []
    for h in ("H1", "H3"):
        v = T.PRIMARY[h]
        c, g, cov = v["cohort"], v["gold"], v["cov"]
        U.append(T.unit(f"{h}:D", "F1", "primary", h, c, g, cov, ["D"], perm="one", gene_ci=True, seeds=True, rel=True))
        U.append(T.unit(f"{h}:h2h", "F1_h2h", "head_to_head", h, c, g, cov, ["D"] + T.J))
        for b in T.J:
            U.append(T.unit(f"{h}:h2h_own:{b}", "F3", "head_to_head_own_universe", h, c, g, cov, ["D", b]))
        U.append(T.unit(f"{h}:common4", "F1_common", "common_universe_of_four_golds", h, c, g, cov, ["D"], subset="common4", perm="one"))
        U.append(T.unit(f"{h}:S_ema", "F3", "S_ema_beside_D", h, c, g, cov, ["S_ema"]))
        U.append(T.unit(f"{h}:alpha_minus_pi_raw", "F3", "alpha_minus_pi_raw", h, c, g, cov, ["D_raw"]))
        U.append(T.unit(f"{h}:margin_support", "F3", "margin_with_support_canonical_flag", h, c, g, cov, ["D"], subset="margin_support"))
        for t in tools:
            U.append(T.unit(f"{h}:tool:{t}", "F1_tools", "tool_vs_D3", h, c, g, cov, ["D3", t]))
    # post hoc T4.3 reading: H3 with hepatocyte enrichment in the design (design and bootstrap as dominance_exploratory; its id and stream differ, checked on value only)
    U.append(T.unit("H3:hep", "F3", "H3_given_hep_enrich", "H3", "liv_hmouse", "G_H3", "C_pat_hep", ["D"], perm="one"))
    U.append(T.unit("H1:hep", "F3", "H1_given_hep_enrich", "H1", "liv_hmouse", "G_H1", "C_hep", ["D"], perm="one"))
    return U


def y4(cm4: dict, cm3: dict) -> pd.DataFrame:
    """Y4 even-gene readings of the liver H+Mouse carried units, code unchanged, D and S columns added to its table."""
    from revision.tasks.Y4_regions import replicate as RP
    from revision.tasks.Y4_regions import screen as SC
    c = "liv_hmouse"
    t = SC.table(c)
    a = pd.read_parquet(RU.REPO / "results/revision/T4_3/cache/liv_hmouse__inputs.parquet", columns=["var_name"]).var_name
    for k, v in (("cm4", cm4), ("cm3", cm3)):
        t[f"D_{k}"] = pd.Series(v["D"], index=a.values).reindex(t.var_name).to_numpy()
        t[f"S_{k}"] = pd.Series(v["S"], index=a.values).reindex(t.var_name).to_numpy()
    rows = []
    for gold, name, region, kind in (("G_H1", "level survivor", "expr_T2__tau_T3", "S"), ("G_H3", "high pi", "pihi__tau_T2", "S"), ("G_H3", "large level gap", "absshift_T3", "D"), ("G_H3", "spread_T1", "spread_T1", "S"), ("G_H3", "low pi", "pilo__spread_T3", "D")):
        m = RP.mask_of(c, region)
        ev = RP.even_rows(t, m)
        for r in RP.simple(c, ev, gold, RP.cov_of(c, gold), [kind, f"{kind}_cm4", f"{kind}_cm3"], label=dict(unit=name, region=region, split="even")):
            rows.append(r)
        rest = RP.even_rows(t, ~np.asarray(m, bool))
        for r in RP.simple(c, rest, gold, RP.cov_of(c, gold), [kind, f"{kind}_cm4", f"{kind}_cm3"], label=dict(unit=name, region=region, split="even_rest")):
            rows.append(r)
    return pd.DataFrame(rows)


def catabolic(frozen: pd.DataFrame, cm4: dict, cm3: dict) -> pd.DataFrame:
    """Where the hepatocyte catabolic list (X2 divergent_conserved, liver 4-sp, the C43 list) sits in liver H+Mouse D and S: mean percentile within genes, frozen and labelled."""
    x = pd.read_csv(RU.REPO / "results/revision/X2/divergent_conserved.csv")
    rows = []
    for co, g in x.groupby("cohort"):
        for cat in [None] + sorted(g.category.unique()):
            gg = g if cat is None else g[g.category == cat]
            ok = frozen.hsapiens_gene_id.isin(gg.hsapiens_gene_id).to_numpy()
            r = dict(list_cohort=co, category=cat or "all", n_list=len(gg), n_in_liv_hmouse=int(ok.sum()))
            for tag, d, s in (("frozen", frozen.D0.to_numpy(), frozen.S0.to_numpy()), ("cmmd4", cm4["D"], cm4["S"]), ("cmmd_no45", cm3["D"], cm3["S"])):
                r[f"D_pct_{tag}"] = float(pd.Series(d).rank(pct=True)[ok].mean())
                r[f"S_pct_{tag}"] = float(pd.Series(s).rank(pct=True)[ok].mean())
            rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--draws", type=int, default=T.N_DRAWS)
    ap.add_argument("--n-perm", type=int, default=T.N_PERM)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    T.CFG.update(draws=a.draws, n_perm=a.n_perm, n_boot_sets=T.N_BOOT_SETS, n_perm_sets=T.N_PERM_SETS, n_calib=T.N_CALIB, n_perm_calib=T.N_PERM_CALIB, mode="real")
    T.prepare("real", 1)
    D = T.DATA["liv_hmouse"]
    tab = D["tab"]
    tools = pd.read_csv(TOOLS)
    per = tools[tools.cohort == "liv_hmouse"].drop(columns="cohort").set_index("hsapiens_gene_id")
    D["tab"] = tab = tab.join(per, on="hsapiens_gene_id")
    tcols = list(per.columns)
    he = pd.read_parquet(RU.REPO / "results/revision/X2/genes/liv_hmouse.parquet").set_index("var_name")["hep_enrich"]
    tab["hep_enrich"] = he.reindex(tab["var_name"]).to_numpy()
    T.COVSETS["C_pat_hep"] = T.COVSETS["C_pat"] + ["hep_enrich"]
    T.COVSETS["C_hep"] = T.COVSETS["C"] + ["hep_enrich"]
    idx = [D["seeds"].index(s) for s in (42, 43, 44)]
    tab["D3"] = D["M"][:, idx].mean(1)
    base = dict(M=D["M"].copy(), seeds=list(D["seeds"]), D=tab["D"].to_numpy().copy(), S_ema=tab["S_ema"].to_numpy().copy(), D_raw=tab["D_raw"].to_numpy().copy(), D3=tab["D3"].to_numpy().copy())
    cm4, cm3 = consensus([42, 43, 44, 45]), consensus([42, 43, 44])
    pd.DataFrame(dict(var_name=tab.var_name, hsapiens_gene_id=tab.hsapiens_gene_id, D_frozen=base["D"], S_frozen=base["S_ema"], D_cmmd4=cm4["D"], S_cmmd4=cm4["S"], D_cmmd_no45=cm3["D"], S_cmmd_no45=cm3["S"])).to_csv(OUT / "consensus.csv", index=False)
    rows, deltas = [], []
    for line in ("frozen", "cmmd4", "cmmd_no45"):
        set_line(base, line)
        U = units(tcols)
        print(line, len(U), "units", flush=True)
        with mp.get_context("fork").Pool(a.workers) as pool:
            res = list(pool.imap_unordered(T.dispatch, U))
        for r in res:
            for x in r["rows"]:
                if "error" in x:
                    raise SystemExit(f"{x['id']}: {x['error']}\n{x.get('traceback', '')[-1200:]}")
                rows.append(dict(x, line=line))
            deltas += [dict(x, line=line) for x in r.get("deltas", [])]
        pd.DataFrame(rows).to_csv(OUT / "scoreboard.csv", index=False)
        pd.DataFrame(deltas).to_csv(OUT / "head_to_head.csv", index=False)
    frz = pd.DataFrame(dict(hsapiens_gene_id=tab.hsapiens_gene_id, D0=base["D"], S0=base["S_ema"]))
    catabolic(frz, cm4, cm3).to_csv(OUT / "catabolic.csv", index=False)
    y4(cm4, cm3).to_csv(OUT / "y4.csv", index=False)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
