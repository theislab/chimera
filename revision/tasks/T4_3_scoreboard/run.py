"""T4.3 validation scoreboard: every pre-specified test of 05 sections 3-6, by a fixed procedure.

Inputs (see MANIFEST.tsv): T3.3 consensus D, T4.2 baselines, golds built by ``golds.py``,
baseline split halves (``baseline_splits.py``), T2.5 margin flags and contributions.

Modes
- ``real``: the official run. Needs ``results/revision/T3_3/DONE``.
- ``null-D``: D (with every per-seed column and the T2.5 contributions, one gene permutation per
  cohort, seeded) is permuted across genes before anything is computed; mechanics check.
- ``null-G``: every gold (and its split halves, the sets and the per-type driver golds) is
  permuted across genes; D is a stand-in from the finished seeds. Mechanics check for the
  baselines against a permuted gold.

Outputs (``--outdir``): scoreboard.csv (H1-H4 for D, baselines and controls, 4-sp estimates),
head_to_head.csv, sets.csv, family3.csv, calibration_sets.csv, units.json; then decide.py writes
row.json and figures.py the figures. Files are written as *.tmp and renamed; DONE last.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[_v] = "1"

import argparse
import itertools
import json
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from revision.common import controls as K
from revision.common import gene_ids
from revision.common.outputs import TaskOutput
from revision.tasks.T4_3_scoreboard import inputs as I
from revision.tasks.T4_3_scoreboard import stats as S

TASK = "T4_3"
REPO = gene_ids.REPO_ROOT
RES = REPO / "results/revision"
COV = list(K.COVARIATE_COLUMNS)
COVSETS = {
    "C": COV,
    "C_pat": COV + ["tau_shared"],
    "C_pat_nsp": COV + ["tau_shared", "H4_nsp"],
    "C_pat_npri": COV + ["tau_shared", "H4_npri"],
    "C_eveemean_liver": COV + ["evee_mean_liver"],
    "C_eveemean_testis": COV + ["evee_mean_testis"],
}
PRIMARY = {  # 05 section 4
    "H1": dict(cohort="liv_hmouse", gold="G_H1", cov="C", tissue="liver", kind="level"),
    "H2": dict(cohort="test_hmouse", gold="G_H2", cov="C", tissue="testis", kind="level"),
    "H3": dict(cohort="liv_hmouse", gold="G_H3", cov="C_pat", tissue="liver", kind="pattern"),
    "H4": dict(cohort="test_hmouse", gold="G_H4", cov="C_pat_nsp", tissue="testis", kind="pattern"),
}
MULTI = {"H1": "liv_4sp", "H3": "liv_4sp", "H2": "test_3sp", "H4": "test_3sp"}  # Figs 4-6 rule estimates
J = ["B1b", "B1c", "B2b", "B3a", "B1z"]
SETS = {"liver": ["C1", "C2", "V1", "V2", "V3"], "testis": ["C1", "C2", "V2", "V3"]}
SET_COHORT = {"liver": "liv_hmouse", "testis": "test_hmouse"}
DRIVER_TYPES = {"Hepatocytes": "hepatocyte", "Endothelial": "endothelial", "fibroblasts": "stellate",
                "Kupffer cells": "macrophage"}
CORE = [42, 43, 44, 45, 46]
N_DRAWS = 10_000
N_PERM = 10_000
N_BOOT_SETS = 2_000
N_PERM_SETS = 10_000
N_CALIB = 200
N_PERM_CALIB = 2_000
BASE_SEED = 20261101

DATA: dict = {}
CFG: dict = {}


def useed(uid: str) -> int:
    return (zlib.crc32(uid.encode()) ^ BASE_SEED) & 0x7FFFFFFF


# ====================================================================== data
def gold_table(mode: str, rng: np.random.Generator):
    g = pd.read_csv(RES / "T4_3/golds/gold_table.csv", index_col=0)
    sp_ = np.load(RES / "T4_3/golds/splits.npz", allow_pickle=True)
    halves = {"G_H3": (sp_["h3_genes"], sp_["h3_a"], sp_["h3_b"]), "G_H4": (sp_["h4_genes"], sp_["h4_a"], sp_["h4_b"])}
    prof = np.load(RES / "T4_3/golds/h3_profiles.npz", allow_pickle=True)
    drv = driver_golds(prof)
    perms = {}
    if mode == "null-G":
        for c in [c for c in g.columns if c.startswith("G_") or c in ("C1", "C2", "V1", "V2", "V3")]:
            p = rng.permutation(len(g))
            perms[c] = p
            g[c] = g[c].to_numpy()[p]
        for c in ("G_H3", "G_H4"):  # halves follow their gold: gene i now carries gene p[i]'s halves
            genes, A, B = halves[c]
            src = np.asarray(g.index)[perms[c]]
            pos = pd.Index(genes).get_indexer(src)
            keep = pos >= 0
            halves[c] = (np.asarray(g.index)[keep], A[:, pos[keep]], B[:, pos[keep]])
        for c in drv.columns:
            drv[c] = drv[c].to_numpy()[rng.permutation(len(drv))]
    return g, halves, drv


def driver_golds(prof) -> pd.DataFrame:
    """Per-type level matching in TS vs TMS: -|rank_TS - rank_TMS| within the type, genes
    detected in >= 1% of the type's cells in both atlases (mid-ranks over the vocabulary / n)."""
    from scipy.stats import rankdata

    types = list(prof["types"])
    n = len(prof["genes"])
    out = {}
    for k, t in enumerate(types):
        rt = rankdata(prof["TS_x"][:, k]) / n
        rm = rankdata(prof["TMS_x"][:, k]) / n
        ok = (prof["TS_f"][:, k] >= 0.01) & (prof["TMS_f"][:, k] >= 0.01)
        out[f"Gdrv_{t}"] = np.where(ok, -np.abs(rt - rm), np.nan)
    return pd.DataFrame(out, index=pd.Index(prof["genes"], name="hsapiens_gene_id"))


def prepare(mode: str, null_seed: int) -> None:
    cache = RES / "T4_3/cache"
    rng = np.random.default_rng(BASE_SEED + 7919 * null_seed)
    gold, halves, drv = gold_table(mode, rng)
    DATA["gold_full"] = gold
    for cohort in sorted({v["cohort"] for v in PRIMARY.values()} | set(MULTI.values())):
        d = I.load(cohort, cache, consensus=(mode == "real"))
        tab = d["tab"]
        n = len(tab)
        F72 = I.features72(cohort)
        cgk = None
        t25 = RES / f"T2_5/{cohort}__C_gk.parquet"
        if cohort == "liv_hmouse" and t25.exists():
            c = pd.read_parquet(t25)
            c = c[c.group_key == "cell_type"].pivot(index="var_name", columns="cell_type", values="C_gk_meanz")
            cgk = c.reindex(tab["var_name"])
        if mode == "null-D":
            p = rng.permutation(n)
            tab["D"] = tab["D"].to_numpy()[p]
            tab["D_raw"] = tab["D_raw"].to_numpy()[p]
            tab["S_ema"] = tab["S_ema"].to_numpy()[p]
            d["M"] = d["M"][p]
            if cgk is not None:
                cgk = pd.DataFrame(cgk.to_numpy()[p], index=cgk.index, columns=cgk.columns)
        tab["B3b"] = I.b3b(F72, tab["D"].to_numpy())
        ref = RES / f"T4_3/baselines_consensusD/baselines__{cohort}.csv"
        if mode == "real" and ref.exists():  # same code path as T4.2 run.py --delta-alpha <consensus>
            rb = pd.read_csv(ref).set_index("var_name")["B3b"].reindex(tab["var_name"]).to_numpy()
            err = float(np.nanmax(np.abs(rb - tab["B3b"].to_numpy())))
            d["meta"]["b3b_vs_T4_2_rerun_max_abs"] = err
            if err > 1e-6:
                raise AssertionError(f"{cohort}: B3b differs from the T4.2 rerun by {err}")
        tab["random"] = np.random.default_rng(useed(f"random-{cohort}")).standard_normal(n)
        tab = tab.join(gold, on="hsapiens_gene_id")
        for c in ("C1", "C2", "V1", "V2", "V3"):
            tab[c] = tab[c].eq(True)
        if cohort == "liv_hmouse":
            tab = tab.join(drv, on="hsapiens_gene_id")
            if cgk is not None:
                for k in DRIVER_TYPES:
                    tab[f"Cgk_{k}"] = cgk[k].to_numpy()
        tab["autosome"] = tab["chromosome"].astype(str).isin([str(i) for i in range(1, 23)])
        seeds = sorted(d["meta"]["seeds"])
        hv = {}
        for gname, (genes, A, B) in halves.items():
            pos = pd.Index(genes).get_indexer(tab["hsapiens_gene_id"])
            Aa = np.where(pos[None, :] >= 0, A[:, np.maximum(pos, 0)], np.nan)
            Bb = np.where(pos[None, :] >= 0, B[:, np.maximum(pos, 0)], np.nan)
            hv[gname] = (Aa, Bb)
        bfile = RES / f"T4_3/baseline_splits/baseline_splits__{cohort}.npz"
        bh = {}
        if bfile.exists():
            z = np.load(bfile, allow_pickle=True)
            pos = pd.Index(z["var_names"]).get_indexer(tab["var_name"])
            for c in I.BASELINE_RAW:
                bh[c] = (z[c][:, 0][:, pos].astype(float), z[c][:, 1][:, pos].astype(float))
        DATA[cohort] = dict(tab=tab, M=d["M"], seeds=seeds, meta=d["meta"], F72=F72, gold_halves=hv,
                            baseline_halves=bh)
        print(f"[prepare] {cohort}: n={n} seeds={seeds} blocks={d['meta']['blocks']}", flush=True)
    # U_g masks of the primary hypotheses and their common gene universe (Step 4)
    ug = {}
    for h, v in PRIMARY.items():
        tab = DATA[v["cohort"]]["tab"]
        ug[h] = set(tab.loc[universe(tab, v["gold"], v["cov"], ["D"]), "hsapiens_gene_id"])
    common = set.intersection(*ug.values())
    for cohort in ("liv_hmouse", "test_hmouse"):
        DATA[cohort]["tab"]["common4"] = DATA[cohort]["tab"]["hsapiens_gene_id"].isin(common)
    DATA["n_common4"] = len(common)
    print(f"[prepare] common universe of the four golds: {len(common)} genes", flush=True)


def universe(tab: pd.DataFrame, gold: str, cov: str, scores: list[str], subset: str | None = None) -> np.ndarray:
    cols = [gold] + COVSETS[cov] + scores
    m = tab[cols].notna().all(1).to_numpy()
    if subset:
        m &= tab[subset].eq(True).to_numpy()
    return m


# ====================================================================== units
def unit(uid, family, analysis, hyp, cohort, gold, cov, scores, **kw):
    u = dict(id=uid, family=family, analysis=analysis, hyp=hyp, cohort=cohort, gold=gold, cov=cov,
             scores=scores, pcs=False, subset=None, perm=None, gene_ci=False, seeds=False, rel=False,
             draws=CFG["draws"], deltas=len(scores) > 1)
    u.update(kw)
    return u


def seed_splits(M: np.ndarray, seeds: list[int], which: str):
    """Half means: 'core' = 2-vs-3 splits of seeds 42-46; 'all' = k-vs-k splits of all seeds."""
    if which == "core":
        idx = [seeds.index(s) for s in CORE if s in seeds]
        if len(idx) != 5:
            return None
        combos = [(list(a), [i for i in idx if i not in a]) for a in itertools.combinations(idx, 2)]
    else:
        if len(seeds) < 10 or len(seeds) % 2:
            return None
        idx = list(range(len(seeds)))
        combos, seen = [], set()
        for a in itertools.combinations(idx, len(idx) // 2):
            b = tuple(i for i in idx if i not in a)
            if frozenset([a, b]) in seen:
                continue
            seen.add(frozenset([a, b]))
            combos.append((list(a), list(b)))
    A = np.array([M[:, a].mean(1) for a, b in combos])
    B = np.array([M[:, b].mean(1) for a, b in combos])
    return A, B, combos


def run_unit(u: dict) -> dict:
    D = DATA[u["cohort"]]
    tab = D["tab"]
    rows = universe(tab, u["gold"], u["cov"], u["scores"], u["subset"])
    n = int(rows.sum())
    base = {k: u[k] for k in ("id", "family", "analysis", "hyp", "cohort", "gold", "cov", "subset")}
    base["pcs"] = u["pcs"]
    if n < 30:
        return dict(rows=[dict(base, score=s, n=n, note="fewer than 30 genes") for s in u["scores"]], deltas=[])
    T = tab.loc[rows]
    G = T[u["gold"]].to_numpy(float)
    R = np.column_stack([S.midrank(G)] + [S.midrank(T[s].to_numpy(float)) for s in u["scores"]])
    Z = S.design(T, COVSETS[u["cov"]], extra=D["F72"][rows][:, :64] if u["pcs"] else None)
    Zpi = S.design(T, ["prior_raw"])
    est = S.rho_p_matrix(R, Z)[1:]
    rpi = S.rho_p_matrix(R, Zpi)[1:]
    blocks = T["block"].to_numpy()
    rng = np.random.default_rng(useed(u["id"]))
    Bb = S.bootstrap(R, Z, u["draws"], rng, blocks)[:, 1:]
    Bg = S.bootstrap(R, Z, u["draws"], rng, None)[:, 1:] if u["gene_ci"] else None
    gfull = DATA["gold_full"][u["gold"]].dropna() if u["gold"] in DATA["gold_full"] else pd.Series(dtype=float)
    out_rows = []
    for k, s in enumerate(u["scores"]):
        x = T[s].to_numpy(float)
        r = dict(base, score=s, n=n, n_blocks=int(len(np.unique(blocks))), sd_G_U=float(np.std(G, ddof=1)),
                 n_gold_full=int(len(gfull)), sd_G_full=float(gfull.std()) if len(gfull) else np.nan,
                 rho_p=float(est[k]), rho_pi=float(rpi[k]), spearman=float(spearmanr(x, G)[0]))
        r["ci_lo"], r["ci_hi"] = S.pct_ci(Bb[:, k])
        r["boot_sd"] = float(np.nanstd(Bb[:, k], ddof=1))
        if Bg is not None:
            r["ci_gene_lo"], r["ci_gene_hi"] = S.pct_ci(Bg[:, k])
            r["var_ratio"] = float(np.nanvar(Bb[:, k]) / np.nanvar(Bg[:, k]))
        out_rows.append(r)
    # permutation test and sign statistic (focal score only)
    if u["perm"]:
        eD = S.resid(R[:, 1], Z)
        eG = S.resid(R[:, 0], Z)
        lab = S.strata(T["prior_raw"].to_numpy(float), T["log_cp10k_spmean"].to_numpy(float))
        null = S.perm_null(eD, eG, lab, CFG["n_perm"], rng)
        T0 = float(np.sum((eD - eD.mean()) * (eG - eG.mean())) / np.sqrt(((eD - eD.mean()) ** 2).sum() * ((eG - eG.mean()) ** 2).sum()))
        r0 = out_rows[0]
        r0["T_perm_stat"] = T0
        r0["p_perm"] = S.p_two_sided(T0, null) if u["perm"] == "two" else S.p_one_sided(T0, null)
        r0["perm_sides"] = u["perm"]
        r0["n_strata"] = int(len(np.unique(lab)))
        r0["null_mean"], r0["null_sd"] = float(null.mean()), float(null.std())
        r0["sign_concordance"] = S.sign_concordance(R[:, 1], R[:, 0], lab)
    # leave-one-seed-out consensus range (focal score D)
    if u["seeds"] and u["scores"][0] in ("D", "D_raw"):
        M = D["M"][rows]
        vals = []
        for j in range(M.shape[1]):
            others = np.delete(M, j, axis=1).mean(1)
            Rr = np.column_stack([R[:, 0], S.midrank(others)])
            vals.append(float(S.rho_p_matrix(Rr, Z)[1]))
        out_rows[0].update(seed_loo_min=min(vals), seed_loo_max=max(vals), seed_loo_n=len(vals),
                           seed_loo=";".join(f"{s}:{v:.4f}" for s, v in zip(D["seeds"], vals)))
    # noise ceilings
    if u["rel"]:
        relG = np.nan
        if u["gold"] in D["gold_halves"]:
            A, B = D["gold_halves"][u["gold"]]
            rg = S.split_rel(A[:, rows], B[:, rows], Z)
            relG = rg["rel"]
            out_rows[0].update(rel_G_r=rg["r"], rel_G_splits=rg["n_splits"], rel_G_n=rg["n_median"])
        for k, s in enumerate(u["scores"]):
            rx, rx_all = np.nan, np.nan
            if s in ("D", "D_raw", "B3b"):
                for which in ("core", "all"):
                    sp_ = seed_splits(D["M"], D["seeds"], which)
                    if sp_ is None:
                        continue
                    A, B, _ = sp_
                    if s == "B3b":
                        if which == "all":
                            continue  # B3b reliability on the plan's 2-vs-3 core splits only
                        A = np.array([I.b3b(D["F72"], a) for a in A])
                        B = np.array([I.b3b(D["F72"], b) for b in B])
                    rr = S.split_rel(A[:, rows], B[:, rows], Z)
                    if which == "core":
                        rx = rr["rel"]
                        out_rows[k]["rel_x_r"] = rr["r"]
                        out_rows[k]["rel_x_splits"] = rr["n_splits"]
                    else:
                        rx_all = rr["rel"]
                        out_rows[k]["rel_x_all_splits"] = rr["n_splits"]
            elif s in D["baseline_halves"]:
                A, B = D["baseline_halves"][s]
                rr = S.split_rel(A[:, rows], B[:, rows], Z)
                rx = rr["rel"]
                out_rows[k].update(rel_x_r=rr["r"], rel_x_splits=rr["n_splits"], rel_x_label="cell halves (technical)")
            r = out_rows[k]
            r["rel_x"], r["rel_x_all"], r["rel_G"] = rx, rx_all, relG
            for tag, rv in (("", rx), ("_all", rx_all)):
                ok = np.isfinite(rv) and np.isfinite(relG) and rv > 0 and relG > 0
                c = float(np.sqrt(rv * relG)) if ok else np.nan
                r[f"ceiling{tag}"] = c
                r[f"disattenuated{tag}"] = float(r["rho_p"] / c) if ok else np.nan
    deltas = []
    if u["deltas"]:
        dd = Bb[:, :1] - Bb[:, 1:]
        de = est[0] - est[1:]
        good = np.isfinite(dd).all(1)
        lo95, hi95, c95 = S.simultaneous_ci(de, dd[good], 0.95)
        lo90, hi90, c90 = S.simultaneous_ci(de, dd[good], 0.90)
        for j, s in enumerate(u["scores"][1:]):
            pl, ph = S.pct_ci(dd[:, j])
            deltas.append(dict(base, focal=u["scores"][0], comparator=s, n_pair=n, rho_p_focal=float(est[0]),
                               rho_p_comparator=float(est[j + 1]), delta=float(de[j]), pc_lo=pl, pc_hi=ph,
                               sim95_lo=float(lo95[j]), sim95_hi=float(hi95[j]), sim90_lo=float(lo90[j]),
                               sim90_hi=float(hi90[j]), c95=c95, c90=c90, n_comparisons=len(de)))
    return dict(rows=out_rows, deltas=deltas)


# ---------------------------------------------------------------------- set units
def set_universe(tab, subset=None):
    m = tab[COV + ["D"]].notna().all(1).to_numpy()
    if subset:
        m &= tab[subset].eq(True).to_numpy()
    return m


def run_set_unit(u: dict) -> dict:
    tab = DATA[u["cohort"]]["tab"]
    m = set_universe(tab, u.get("subset")) & tab[u["score"]].notna().to_numpy()
    T = tab.loc[m]
    inset = T[u["set"]].to_numpy(bool)
    score = T[u["score"]].to_numpy(float)
    if u["strata"] == "pi_expr":
        lab = S.cells(T["prior_raw"].to_numpy(float), T["log_cp10k_spmean"].to_numpy(float))
    else:
        lab = S.cells(None, T["log_cp10k_spmean"].to_numpy(float))
    keep = S.kept_strata(inset, lab, absolute=u.get("bg_rule") == "absolute")
    rng = np.random.default_rng(useed(u["id"]))
    r = {k: u[k] for k in ("id", "family", "tissue", "cohort", "set", "score", "strata", "direction")}
    r["bg_rule"] = u.get("bg_rule", "ratio")
    r["subset"] = u.get("subset")
    r.update(n_universe=int(m.sum()), n_set_universe=int(inset.sum()), n_set=int(inset[keep].sum()),
             n_background=int((~inset[keep]).sum()), n_strata_kept=int(len(np.unique(lab[keep]))),
             n_strata=int(len(np.unique(lab))))
    if r["n_set"] < 3:
        r["note"] = "fewer than 3 set genes in kept strata"
        return dict(rows=[r])
    sc, ins, lb = score[keep], inset[keep], lab[keep]
    r["auroc"] = S.strat_auroc(sc, ins, lb)
    r["auprc"] = S.strat_auprc(sc, ins, lb)
    r["prevalence"] = float(ins.mean())
    A, P = S.auroc_boot(sc, ins, lb, CFG["n_boot_sets"], rng)
    r["auroc_ci_lo"], r["auroc_ci_hi"] = S.pct_ci(A)
    r["auprc_ci_lo"], r["auprc_ci_hi"] = S.pct_ci(P)
    if u.get("perm"):
        null = S.auroc_perm(sc, ins, lb, CFG["n_perm_sets"], rng)
        r["p_perm"] = S.p_one_sided(r["auroc"], null, lower=(u["direction"] == "below"))
        r["null_mean"] = float(null.mean())
    return dict(rows=[r])


def run_calib_unit(u: dict) -> dict:
    """Calibration of the set test on random gene sets of the observed size (uniform draws
    from the set universe): fraction of nominal one-sided p < 0.05 in each direction."""
    tab = DATA[u["cohort"]]["tab"]
    m = set_universe(tab, u.get("subset"))
    T = tab.loc[m]
    score = T["D"].to_numpy(float)
    lab = S.cells(T["prior_raw"].to_numpy(float), T["log_cp10k_spmean"].to_numpy(float))
    rng = np.random.default_rng(useed(u["id"]))
    pu, pl, used = [], [], 0
    for _ in range(CFG["n_calib"]):
        ins = np.zeros(len(T), bool)
        ins[rng.choice(len(T), u["size"], replace=False)] = True
        keep = S.kept_strata(ins, lab)
        if ins[keep].sum() < 3:
            continue
        used += 1
        a = S.strat_auroc(score[keep], ins[keep], lab[keep])
        null = S.auroc_perm(score[keep], ins[keep], lab[keep], CFG["n_perm_calib"], rng)
        pu.append(S.p_one_sided(a, null))
        pl.append(S.p_one_sided(a, null, lower=True))
    pu, pl = np.array(pu), np.array(pl)
    return dict(rows=[dict(id=u["id"], cohort=u["cohort"], subset=u.get("subset"), set_like=u["set"], size=u["size"],
                           n_random_sets=used, frac_p_above_lt_005=float((pu < 0.05).mean()) if used else np.nan,
                           frac_p_below_lt_005=float((pl < 0.05).mean()) if used else np.nan,
                           median_p_above=float(np.median(pu)) if used else np.nan,
                           n_perm=CFG["n_perm_calib"])])


def dispatch(u):
    kind = u.get("kind", "rho")
    try:
        if kind == "set":
            return run_set_unit(u)
        if kind == "calib":
            return run_calib_unit(u)
        return run_unit(u)
    except Exception as e:  # noqa: BLE001 - one failing unit is reported, never silently dropped
        import traceback

        return dict(rows=[dict(id=u["id"], error=repr(e), traceback=traceback.format_exc())], deltas=[])


# ====================================================================== plan
def build_units(tools: pd.DataFrame | None) -> list[dict]:
    U = []
    for h, v in PRIMARY.items():
        c, g, cov = v["cohort"], v["gold"], v["cov"]
        # Family 1: D, all statistics
        U.append(unit(f"{h}:D", "F1", "primary", h, c, g, cov, ["D"], perm="one", gene_ci=True, seeds=True, rel=True))
        # baselines and B3b on their own universes (scoreboard rows), random control
        for b in J + ["B3b"]:
            U.append(unit(f"{h}:{b}", "scoreboard", "baseline", h, c, g, cov, [b], gene_ci=True, rel=True))
        U.append(unit(f"{h}:random", "scoreboard", "control_random", h, c, g, cov, ["random"]))
        # head-to-head on U_pair, joint block bootstrap
        U.append(unit(f"{h}:h2h", "F1_h2h", "head_to_head", h, c, g, cov, ["D"] + J))
        for b in J:
            U.append(unit(f"{h}:h2h_own:{b}", "F3", "head_to_head_own_universe", h, c, g, cov, ["D", b]))
        # Family 3
        U.append(unit(f"{h}:beyond_pcs", "F3", "beyond_features_D_given_C_plus_64PCs", h, c, g, cov, ["D"], pcs=True))
        U.append(unit(f"{h}:B3b_vs_D", "F3", "beyond_features_B3b", h, c, g, cov, ["D", "B3b"]))
        U.append(unit(f"{h}:margin_support", "F3", "margin_with_support", h, c, g, cov, ["D"], subset="margin_support"))
        U.append(unit(f"{h}:margin_lacks", "F3", "margin_without_support", h, c, g, cov, ["D"], subset="lacks_margin_support"))
        U.append(unit(f"{h}:cov_pcs", "F3", "coverage_low_detection_D_given_C_plus_64PCs", h, c, g, cov, ["D"],
                      pcs=True, subset="low_detection"))
        U.append(unit(f"{h}:cov_vs_B3", "F3", "coverage_low_detection_D_vs_B3a_B3b", h, c, g, cov, ["D", "B3a", "B3b"],
                      subset="low_detection"))
        U.append(unit(f"{h}:alpha_minus_pi_raw", "F3", "alpha_minus_pi_raw", h, c, g, cov, ["D_raw"]))
        U.append(unit(f"{h}:S_ema", "F3", "S_ema_beside_D", h, c, g, cov, ["S_ema"]))
        U.append(unit(f"{h}:common4", "F1_common", "common_universe_of_four_golds", h, c, g, cov, ["D"],
                      subset="common4", perm="one"))
        if v["tissue"] == "testis":
            U.append(unit(f"{h}:autosome", "F3", "testis_autosome_only", h, c, g, cov, ["D"], subset="autosome"))
        # Figs 4-6 rule: canonical multi-species cohort, estimate
        mc = MULTI[h]
        U.append(unit(f"{h}:multi:D", "F3_figs4_6", "multi_species_estimate", h, mc, g, cov, ["D"], perm="one",
                      seeds=True, rel=True))
        if mc == "liv_4sp":
            U.append(unit(f"{h}:multi_nonzero:D", "F3", "multi_species_estimate_nonzero_all_species", h, mc, g, cov,
                          ["D"], subset="nonzero_all_species"))
        if tools is not None:
            for t in [c for c in tools.columns if c != "hsapiens_gene_id"]:
                U.append(unit(f"{h}:tool:{t}", "F1_tools", "tool_vs_D3", h, c, g, cov, ["D3", t]))
    U.append(unit("H1:evee_mean", "F3", "evee_mean_in_C", "H1", "liv_hmouse", "G_H1", "C_eveemean_liver", ["D"]))
    U.append(unit("H2:evee_mean", "F3", "evee_mean_in_C", "H2", "test_hmouse", "G_H2", "C_eveemean_testis", ["D"]))
    U.append(unit("H4:primates_only", "F3", "h4_primates_only", "H4", "test_hmouse", "G_H4_primates", "C_pat_npri", ["D"]))
    # H8: two-sided, raw, rho_pi, rho_p
    for c in ("liv_hmouse", "test_hmouse"):
        for g in ("G_LOEUF", "G_dNdS"):
            U.append(unit(f"H8:{c}:{g}", "F3", "H8", "H8", c, g, "C", ["D"], perm="two"))
    # cell-type drivers vs TS/TMS per-type level matching (liver H+Mouse)
    for k, t in DRIVER_TYPES.items():
        U.append(unit(f"driver:{k}", "F3", "celltype_driver_vs_TS_TMS", "driver", "liv_hmouse", f"Gdrv_{t}", "C",
                      [f"Cgk_{k}", "D", "B1c"]))
    # sets
    for tis, sets in SETS.items():
        c = SET_COHORT[tis]
        for s in sets:
            d = "above" if s.startswith("C") else "below"
            U.append(dict(kind="set", id=f"set:{tis}:{s}:D", family="F2", tissue=tis, cohort=c, set=s, score="D",
                          strata="pi_expr", direction=d, perm=True))
            U.append(dict(kind="set", id=f"set:{tis}:{s}:D:absolute_bg", family="F3", tissue=tis, cohort=c, set=s,
                          score="D", strata="pi_expr", direction=d, perm=True, bg_rule="absolute"))
            U.append(dict(kind="set", id=f"set:{tis}:{s}:pi_raw", family="F3", tissue=tis, cohort=c, set=s,
                          score="prior_raw", strata="expr", direction=d))
            for b in J + ["B3b"]:
                U.append(dict(kind="set", id=f"set:{tis}:{s}:{b}", family="F3", tissue=tis, cohort=c, set=s, score=b,
                              strata="pi_expr", direction=d))
    for s in SETS["liver"]:  # 4-sp set estimates (Figs 4-6 cohort) and the nonzero-in-every-species subset
        d = "above" if s.startswith("C") else "below"
        for sub in (None, "nonzero_all_species"):
            U.append(dict(kind="set", id=f"set:liv_4sp:{s}:D:{sub}", family="F3", tissue="liver", cohort="liv_4sp",
                          set=s, score="D", strata="pi_expr", direction=d, perm=True, subset=sub))
    return U


def calib_units() -> list[dict]:
    U = []
    for tis, sets in SETS.items():
        c = SET_COHORT[tis]
        tab = DATA[c]["tab"]
        m = set_universe(tab)
        for s in sets:
            U.append(dict(kind="calib", id=f"calib:{c}:{s}", cohort=c, set=s, size=int(tab.loc[m, s].sum())))
    tab = DATA["liv_4sp"]["tab"]
    for sub in (None, "nonzero_all_species"):
        m = set_universe(tab, sub)
        for s in SETS["liver"]:
            U.append(dict(kind="calib", id=f"calib:liv_4sp:{s}:{sub}", cohort="liv_4sp", set=s, subset=sub,
                          size=int(tab.loc[m, s].sum())))
    return U


# ====================================================================== main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["real", "null-D", "null-G"], required=True)
    ap.add_argument("--null-seed", type=int, default=1)
    ap.add_argument("--outdir", type=Path, default=None)
    ap.add_argument("--draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--n-boot-sets", type=int, default=N_BOOT_SETS)
    ap.add_argument("--n-perm-sets", type=int, default=N_PERM_SETS)
    ap.add_argument("--n-calib", type=int, default=N_CALIB)
    ap.add_argument("--n-perm-calib", type=int, default=N_PERM_CALIB)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--only", default=None, help="regex on unit ids (testing)")
    ap.add_argument("--tools", type=Path, default=None, help="CSV of tool scalars keyed by hsapiens_gene_id")
    a = ap.parse_args(argv)
    if a.mode == "real":
        if not (RES / "T3_3/DONE").exists():
            raise SystemExit("results/revision/T3_3/DONE missing: the consensus D is not final")
        if a.outdir is None:
            a.outdir = RES / TASK
    elif a.outdir is None:
        raise SystemExit("null modes need --outdir (never the official folder)")
    if a.mode != "real" and Path(a.outdir).resolve() == (RES / TASK).resolve():
        raise SystemExit("null modes may not write to the official folder")
    CFG.update(draws=a.draws, n_perm=a.n_perm, n_boot_sets=a.n_boot_sets, n_perm_sets=a.n_perm_sets,
               n_calib=a.n_calib, n_perm_calib=a.n_perm_calib, mode=a.mode)
    out = TaskOutput(TASK, a.outdir, force=True)
    prepare(a.mode, a.null_seed)
    tools = None
    if a.tools is not None and Path(a.tools).exists():
        tools = pd.read_csv(a.tools)
        for c in ("liv_hmouse", "test_hmouse"):
            D = DATA[c]
            idx = [D["seeds"].index(s) for s in (42, 43, 44)]
            D["tab"]["D3"] = D["M"][:, idx].mean(1)
            D["tab"] = D["tab"].join(tools.set_index("hsapiens_gene_id"), on="hsapiens_gene_id")
    units = build_units(tools) + calib_units()
    if a.only:
        import re

        units = [u for u in units if re.search(a.only, u["id"])]
    # heavy units first
    units.sort(key=lambda u: -(u.get("draws", 0) * (4 if u.get("pcs") else 1) * (len(u.get("scores", [1])))
                               + (5e5 if u.get("kind") == "calib" else 0)))
    print(f"{len(units)} units, {a.workers} workers", flush=True)
    import multiprocessing as mp

    with mp.get_context("fork").Pool(a.workers) as pool:
        res = []
        for i, r in enumerate(pool.imap_unordered(dispatch, units)):
            res.append(r)
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(units)} units", flush=True)
    rows = [x for r in res for x in r["rows"]]
    deltas = [x for r in res for x in r.get("deltas", [])]
    errs = [x for x in rows if "error" in x]
    for e in errs:
        print("ERROR", e["id"], e["error"], e.get("traceback", "")[-1500:], flush=True)
    if errs:
        raise SystemExit(f"{len(errs)} units failed")
    R = pd.DataFrame(rows)
    Dl = pd.DataFrame(deltas)
    rho = R[~R["id"].str.startswith(("set:", "calib:"))].copy()
    sets = R[R["id"].str.startswith("set:")].copy()
    calib = R[R["id"].str.startswith("calib:")].copy().dropna(axis=1, how="all")
    # Holm m = 4 over the Family 1 primary tests; on the common universe too (Step 4)
    for fam in ("F1", "F1_common"):
        m = (rho.family == fam) & (rho.score == "D")
        sub = rho[m].set_index("hyp").reindex(list(PRIMARY))
        adj = S.holm(sub["p_perm"].fillna(1.0).tolist())
        rho.loc[m, "p_holm"] = rho.loc[m, "hyp"].map(dict(zip(PRIMARY, adj)))
    # Family 2 BH over the 9 pre-specified set tests of D
    m = (sets.family == "F2") if len(sets) else pd.Series(dtype=bool)
    if m.sum() == 9:
        sets.loc[m, "q_bh"] = S.bh(sets.loc[m, "p_perm"].tolist())
        sets.loc[m, "pass_predicted_direction"] = (sets.loc[m, "q_bh"] < 0.05) & np.where(
            sets.loc[m, "direction"] == "above", sets.loc[m, "auroc"] > 0.5, sets.loc[m, "auroc"] < 0.5)
    # controls with raw Spearman only (05 section 3)
    ctrl = []
    for h, v in PRIMARY.items():
        tab = DATA[v["cohort"]]["tab"]
        m_ = universe(tab, v["gold"], v["cov"], ["D"])
        for s in ("prior_raw", "log_cp10k_spmean"):
            ctrl.append(dict(id=f"{h}:{s}", family="scoreboard", analysis="control_raw_spearman_only", hyp=h,
                             cohort=v["cohort"], gold=v["gold"], cov=v["cov"], score=s, n=int(m_.sum()),
                             spearman=float(spearmanr(tab.loc[m_, s], tab.loc[m_, v["gold"]])[0])))
    rho = pd.concat([rho, pd.DataFrame(ctrl)], ignore_index=True)
    order = {"F1": 0, "scoreboard": 1, "F1_h2h": 2, "F1_common": 3, "F3_figs4_6": 4, "F1_tools": 5, "F3": 6}
    rho["_o"] = rho.family.map(order).fillna(9)
    rho = rho.sort_values(["_o", "hyp", "id"]).drop(columns="_o").dropna(axis=1, how="all")
    score_fams = ("F1", "scoreboard", "F1_h2h", "F1_common", "F3_figs4_6", "F1_tools")
    out.csv(rho[rho.family.isin(score_fams)], "scoreboard.csv")
    out.csv(rho[~rho.family.isin(score_fams)], "family3.csv")
    out.csv(Dl, "head_to_head.csv")
    out.csv(sets.dropna(axis=1, how="all"), "sets.csv")
    out.csv(calib, "calibration_sets.csv")
    meta = dict(mode=a.mode, null_seed=a.null_seed if a.mode != "real" else None, cfg=CFG, n_units=len(units),
                n_common4=DATA["n_common4"],
                cohorts={c: dict(seeds=DATA[c]["meta"]["seeds"], blocks=DATA[c]["meta"]["blocks"],
                                 consensus_check=DATA[c]["meta"].get("consensus_check_max_abs"),
                                 b3b_check=DATA[c]["meta"].get("b3b_vs_T4_2_rerun_max_abs"),
                                 baseline_halves=sorted(DATA[c]["baseline_halves"]))
                         for c in DATA if isinstance(DATA[c], dict) and "tab" in DATA[c]},
                units=[{k: v for k, v in u.items()} for u in units])
    out.json(meta, "units.json")
    if not a.only:
        from revision.tasks.T4_3_scoreboard import decide as DEC
        from revision.tasks.T4_3_scoreboard import figures as FIG

        res = DEC.decide(out.dir)
        out.json(res, "row.json")
        print(json.dumps(dict(status=res["step1_status"], h2h=res["step2_head_to_head"], row=res["step3"],
                              modifiers=res["step4_modifiers"]), default=str), flush=True)
        for f in FIG.make(out.dir):
            out.files.append(str(Path(f).relative_to(out.dir)))
        out.finish(dict(mode=a.mode, null_seed=a.null_seed))


if __name__ == "__main__":
    main()
