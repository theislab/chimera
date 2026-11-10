"""Z4 independent references: the canonical reading's agreement statistics recomputed with the free-gate D beside it, same code,
same golds, covariates, blocks and draws; paired difference free - canonical.

  trackc   B2 Track C units (track_c.table, strata, covariates, EN.Task with B2's seeds and 10,000 block draws) for the liver, testis,
           kidney, islet and bone-marrow golds; strata all, count_supported (B2's, canonical support) and support_both (free and
           canonical). Scores: B2's core set plus D_free, S_free, D_raw_free. h2h: D_free against D, S_g, S_free and the baselines.
  lists    Bone-marrow Shay lists (track_c.list_sets, stratified AUROC, B2's seeds; D_free and S_free appended after B2's methods).
  cortex   C1 E-B1 (tests_c1.rho_table: rho_p with held-out regions | C, permutation and block bootstrap; delta to canonical D) and
           E-B2 (near-identical strata, tests_c1.boot_spear).
  cases    C1 H-D1 (Alzheimer on M1 H+Mouse, lung, muscle): C1's h_d1 statistic with D_free and S_free added (C1's files untouched).

    PYTHONPATH=$PWD python -m analysis.tasks.Z4_free_gate_all.references trackc|lists|cortex|cases|summary [--jobs 6]
Output: results/analysis/Z4/references/
"""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from analysis.common import gene_ids  # noqa: E402
from analysis.tasks.B2_benchmark import engine as EN  # noqa: E402
from analysis.tasks.B2_benchmark import rules as B2R  # noqa: E402
from analysis.tasks.B2_benchmark import rules_c as RC  # noqa: E402
from analysis.tasks.B2_benchmark import track_c as TC  # noqa: E402
from analysis.tasks.C1_cohorts import configs as CF  # noqa: E402
from analysis.tasks.C1_cohorts import tests_c1 as T1  # noqa: E402
from analysis.tasks.T4_3_scoreboard import stats as ST  # noqa: E402
from analysis.tasks.Z4_free_gate_all import rules as RU  # noqa: E402

OUT = RU.OUT / "references"
FREE = RU.OUT / "h1" / "genes"
STRATA = ("all", "count_supported", "support_both")
COMPARATORS = ["D", "S_g", "S_free", "B1", "B1b", "B1c", "B2b", "B3a", "B1z", "Sdir", "Lfc"]


def add_free(t: pd.DataFrame, c: str) -> pd.DataFrame:
    """D_free, S_free, D_raw_free, support_free joined on the human Ensembl ID (unique in both tables)."""
    f = pd.read_parquet(FREE / f"{c}.parquet")
    f = f[f.hsapiens_gene_id.notna() & ~f.hsapiens_gene_id.duplicated(keep=False)].set_index("hsapiens_gene_id")
    key = t["hsapiens_gene_id"]
    t = t.copy()
    t["D_free"] = f["D"].reindex(key).to_numpy(float)
    t["S_free"] = f["S_ema"].reindex(key).to_numpy(float)
    t["D_raw_free"] = f["D_raw"].reindex(key).to_numpy(float)
    t["support_free"] = f["count_support"].reindex(key).fillna(False).to_numpy(bool)
    return t


def unit_list(cohorts: list[str]) -> list[dict]:
    rows = []
    for c in cohorts:
        if c not in RC.COHORTS_C or not (FREE / f"{c}.parquet").exists():
            continue
        for kind, gold, _ in RC.COHORTS_C[c]["golds"]:
            rows += [dict(cohort=c, kind=kind, gold=gold, stratum=s) for s in STRATA]
    return rows


def score_unit(u: dict) -> tuple[list[dict], list[dict]]:
    """As track_c.score_unit, main group only, with the free-gate scores added; B2's seed, draws, universe rule."""
    c, kind, gold, stn = u["cohort"], u["kind"], u["gold"], u["stratum"]
    covs = next(x[2] for x in RC.COHORTS_C[c]["golds"] if x[0] == kind)
    t = add_free(TC.table(c), c)
    stt = TC.strata(t)
    st = (stt["count_supported"] & t["support_free"].to_numpy(bool)) if stn == "support_both" else stt[stn]
    G = t[gold].to_numpy(dtype=float) if gold in t else np.full(len(t), np.nan)
    ok0 = np.isfinite(G) & t[covs].notna().all(1).to_numpy() & st & t["block"].notna().to_numpy()
    base = dict(cohort=c, kind=kind, gold=gold, stratum=stn)
    if ok0.sum() < B2R.STRATUM_MIN:
        return [dict(**base, method="", value=np.nan, n=int(ok0.sum()))], []
    seed_stratum = "count_supported" if stn == "support_both" else stn
    task = EN.Task(t, B2R.N_DRAWS_C, B2R.useed("C", c, kind, seed_stratum))
    scores = {k: t[k].to_numpy(dtype=float) for k in TC.CORE + ["random"] + [x for x in B2R.TOOLS if x in t]}
    in_core = [k for k in TC.CORE if np.isfinite(scores[k][ok0]).mean() >= 0.5]
    U = ok0.copy()
    for k in in_core:
        U &= np.isfinite(scores[k])
    if U.sum() < B2R.STRATUM_MIN:
        U = ok0 & np.isfinite(scores["S_g"]) & np.isfinite(scores["D"])
        in_core = ["S_g", "D"]
    U &= np.isfinite(t["D_free"].to_numpy(float))
    sc = {**{k: scores[k] for k in in_core}, "D_free": t["D_free"].to_numpy(float), "S_free": t["S_free"].to_numpy(float),
          "D_raw": t["D_raw"].to_numpy(float), "D_raw_free": t["D_raw_free"].to_numpy(float)}
    task.add("main", U, G, sc, covs)
    task.run()
    rows = [dict(**base, method=r["method"], value=r["value"], ci_lo=r["ci_lo"], ci_hi=r["ci_hi"], n=r["n"], n_blocks=r["n_blocks"]) for r in task.rows()]
    h = task.h2h(["D_free"], [k for k in COMPARATORS if k in sc], "main") + task.h2h(["S_free"], ["S_g", "D"], "main") + task.h2h(["D_raw_free"], ["D_raw"], "main")
    return rows, [dict(**base, **x) for x in h]


def trackc(jobs: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    units = unit_list(list(RC.COHORTS_C))
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(score_unit, units))
    R = pd.DataFrame([r for x in res for r in x[0]])
    H = pd.DataFrame([r for x in res for r in x[1]])
    R.to_csv(OUT / "trackc_rho.csv", index=False)
    H.to_csv(OUT / "trackc_h2h.csv", index=False)
    # check: canonical D and S_g on strata all and count_supported reproduce B2's unit files (same universe -> same estimate)
    ck = []
    for (c, kind, stn), g in R[R.stratum.isin(["all", "count_supported"])].groupby(["cohort", "kind", "stratum"]):
        f = B2R.RES / "track_C" / "units" / f"{c}__{kind}__{stn}.parquet"
        if not f.exists():
            ck.append(dict(cohort=c, kind=kind, stratum=stn, note="no B2 unit"))
            continue
        b = pd.read_parquet(f).set_index("method")
        for m in ("D", "S_g"):
            mine = g.set_index("method")["value"].get(m, np.nan)
            ck.append(dict(cohort=c, kind=kind, stratum=stn, method=m, z4=mine, b2=float(b["value"].get(m, np.nan)), abs_diff=abs(mine - float(b["value"].get(m, np.nan)))))
    pd.DataFrame(ck).to_csv(OUT / "trackc_check_b2.csv", index=False)
    print(pd.DataFrame(ck).round(4).to_string(), flush=True)


def lists() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for c in [k for k in RC.COHORTS_C if any(x.startswith("Shay") for x in RC.COHORTS_C[k]["lists"])]:
        if not (FREE / f"{c}.parquet").exists():
            continue
        t = add_free(TC.table(c), c)
        cov_ok = t[B2R.COV_C].notna().all(1).to_numpy()
        pi, ex = t["prior_raw"].to_numpy(dtype=float), t["log_cp10k_spmean"].to_numpy(dtype=float)
        methods = TC.CORE + ["pi", "dNdS", "LOEUF"] + [x for x in B2R.TOOLS if x in t] + ["D_free", "S_free"]
        col = {"pi": "pi_score", "dNdS": "dNdS_score", "LOEUF": "LOEUF_score"}
        for name in [x for x in RC.COHORTS_C[c]["lists"] if x.startswith("Shay")]:
            inset, comp = TC.list_sets(t, name)
            rng = np.random.default_rng(B2R.useed("lists", c, name))
            for m in methods:
                v = t[col.get(m, m)].to_numpy(dtype=float)
                use = cov_ok & np.isfinite(v) & comp
                if inset[use].sum() < 15:
                    continue
                idx = np.flatnonzero(use)
                lab = ST.cells(None, ex[idx]) if m == "pi" else ST.cells(pi[idx], ex[idx])
                keep = ST.kept_strata(inset[idx], lab)
                a = ST.strat_auroc(v[idx][keep], inset[idx][keep], lab[keep])
                A, _ = ST.auroc_boot(v[idx][keep], inset[idx][keep], lab[keep], RC.N_BOOT_LISTS, rng, auprc=False)
                lo, hi = ST.pct_ci(A)
                below = RC.LIST_DIRECTION[name] == "below"
                val, l2, h2 = (1 - a, 1 - hi, 1 - lo) if below else (a, lo, hi)
                rows.append(dict(cohort=c, list=name, method=m, value=val, ci_lo=l2, ci_hi=h2, auroc_raw=a, n=int(keep.sum()), n_set=int(inset[idx][keep].sum()),
                                 note=f"expected {RC.LIST_DIRECTION[name]}; oriented so higher is the expected side"))
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "lists.csv", index=False)
    b = pd.read_parquet(B2R.RES / "track_C" / "lists" / "bone_marrow.parquet")
    chk = d.merge(b[["setting", "method", "value"]].rename(columns={"setting": "list", "value": "b2"}), on=["list", "method"], how="left")
    print(chk[chk.method.isin(["D", "D_free", "S_g", "S_free", "B1"])].round(3).to_string(), flush=True)


def c1_genes(c: str) -> pd.DataFrame:
    g = T1.genes(c)
    f = pd.read_parquet(FREE / f"{c}.parquet").reindex(g["var_name"])
    g["D_free"], g["S_free"] = f["D"].to_numpy(float), f["S_ema"].to_numpy(float)
    g["support_both"] = g["count_support"].to_numpy(bool) & f["count_support"].fillna(False).to_numpy(bool)
    return g


def cortex(n_perm: int, n_boot: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(T1.SEED)
    e1, e2 = [], []
    scores = ["D", "D_free", "S_ema", "S_free", "B1", "pi_raw"]
    for c, gl in T1.BRAIN_GOLDS.items():
        if not (FREE / f"{c}.parquet").exists():
            continue
        g = c1_genes(c)
        part = T1.PARTNER[c]
        g["identity_partner"] = g[f"{part}_identity"].astype(float)
        g["log_dnds_mouse"] = np.log10(g["dNdS"].where(g["dNdS"] > 0).clip(lower=1e-3))
        for gname in gl:
            gg = pd.read_parquet(T1.GOLD / f"{gname}.parquet").drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")
            u = g.assign(gold=gg["gold"].reindex(g["hsapiens_gene_id"]).to_numpy())
            for uni in ("all", "support_both"):
                v = u if uni == "all" else u[u.support_both]
                e1 += T1.rho_table(v, "gold", scores, n_perm, n_boot, rng, dict(cohort=c, gold=gname, universe=uni))
            for name, mask in (("identity_ge_98", u["identity_partner"] >= 98), ("top_pi_tertile", u["pi_raw"].rank(pct=True) > 2 / 3), ("identity_ge_99_5", u["identity_partner"] >= 99.5)):
                v = u[mask.to_numpy()]
                for sc in ("identity_partner", "log_dnds_mouse", "D", "D_free", "S_ema", "S_free", "B1"):
                    if v[sc].notna().sum() < 50 or v[sc].nunique() < 3:
                        continue
                    a = v[["gold", sc, "block"]].dropna()
                    r, lo, hi = T1.boot_spear(a[sc].to_numpy(float), a["gold"].to_numpy(float), a["block"].to_numpy(), rng)
                    e2.append(dict(cohort=c, gold=gname, stratum=name, score=sc, n=len(a), spearman=r, ci_lo=lo, ci_hi=hi))
    E1 = pd.DataFrame(e1)
    E1.to_csv(OUT / "cortex_e_b1.tsv", sep="\t", index=False)
    pd.DataFrame(e2).to_csv(OUT / "cortex_e_b2.tsv", sep="\t", index=False)
    c1 = pd.read_csv(CF.RES / "cortex" / "e_b1.tsv", sep="\t")
    chk = E1[E1.universe == "all"].merge(c1[["cohort", "gold", "score", "rho_p"]], on=["cohort", "gold", "score"], how="left", suffixes=("", "_C1"))
    print(chk[["cohort", "gold", "score", "n", "rho_p", "rho_p_C1", "ci_lo", "ci_hi", "delta_to_first", "delta_lo", "delta_hi"]].round(3).to_string(), flush=True)


def h_d1(case: str, model: str, types: list[str], n_perm: int, n_boot: int) -> pd.DataFrame:
    """tests_c1.h_d1 with D_free and S_free added; reads C1's DE files, writes only to Z4."""
    rng = np.random.default_rng(T1.SEED + 1)
    de = CF.RES / case / "de"
    g = c1_genes(model)
    tab = gene_ids.load_table(model)
    ids = pd.Index(g["var_name"]).str.replace("ortho_", "", regex=False).astype(int)
    g["mouse_id"] = tab.loc[ids, "mmusculus_gene_id"].to_numpy()
    rows = []
    for t in types:
        fh, fm = de / f"human_primary_{t}.tsv", de / f"mouse_primary_{t}.tsv"
        if not (fh.exists() and fm.exists()):
            continue
        h = pd.read_csv(fh, sep="\t").dropna(subset=["log2fc_mle", "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
        m = pd.read_csv(fm, sep="\t").dropna(subset=["log2fc_fixed", "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
        u = g.copy()
        u["l"] = h["log2fc_mle"].reindex(u["hsapiens_gene_id"]).to_numpy()
        u["det_h"] = h["det_frac"].reindex(u["hsapiens_gene_id"]).to_numpy()
        u["padj_h"] = h["padj"].reindex(u["hsapiens_gene_id"]).to_numpy()
        u["m"] = m["log2fc_fixed"].reindex(u["mouse_id"]).to_numpy()
        u["se_m"] = m["lfcse_fixed"].reindex(u["mouse_id"]).to_numpy()
        u["det_m"] = m["det_frac"].reindex(u["mouse_id"]).to_numpy()
        u = u[(u.det_h >= 0.05) & (u.det_m >= 0.05)].dropna(subset=["l", "m", "se_m"]).copy()
        u["abs_l"] = u["l"].abs()
        u["Q"] = np.sign(u["l"]) * u["m"]
        rows += T1.rho_table(u, "Q", ["D", "D_free", "S_ema", "S_free", "B1", "pi_raw"], n_perm, n_boot, rng,
                             dict(case=case, cell_type=t, n_human_de=int((u.padj_h < 0.1).sum()), spearman_l_m=float(spearmanr(u.l, u.m)[0])), extra_nuis=("abs_l", "se_m"))
    return pd.DataFrame(rows)


def cases(n_perm: int, n_boot: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    spec = [("ad", "brain_m1_hmouse", ["microglia", "astrocyte", "oligodendrocyte"]),
            ("muscle", "muscle_ageing", ["fb", "musc", "smc", "pericyte", "capec", "artec", "venec", "macrophage"]),
            ("lung", "lung_healthy", ["macrophage", "at2", "endothelial", "t", "dc"])]
    out = []
    for case, model, types in spec:
        if (FREE / f"{model}.parquet").exists():
            out.append(h_d1(case, model, types, n_perm, n_boot))
    d = pd.concat(out)
    d.to_csv(OUT / "cases_h_d1.tsv", sep="\t", index=False)
    c1 = pd.concat([pd.read_csv(CF.RES / k / "h_d1.tsv", sep="\t") for k in ("ad", "muscle", "lung")])
    chk = d.merge(c1[["case", "cell_type", "score", "rho_p"]], on=["case", "cell_type", "score"], how="left", suffixes=("", "_C1"))
    print(chk[["case", "cell_type", "score", "n", "rho_p", "rho_p_C1", "ci_lo", "ci_hi", "delta_to_first", "delta_lo", "delta_hi"]].round(3).to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    if a.cmd == "trackc":
        trackc(a.jobs)
    elif a.cmd == "lists":
        lists()
    elif a.cmd == "cortex":
        cortex(a.n_perm, a.n_boot)
    elif a.cmd == "cases":
        cases(a.n_perm, a.n_boot)


if __name__ == "__main__":
    main()
