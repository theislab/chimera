"""Track C: rho_p, 10,000 block draws, head-to-head, on every cohort x gold x stratum; named lists; Y4_regions replication. Re-runnable: new cohorts join by a row in rules_c.py.
    python -m analysis.tasks.B2_benchmark.track_c units            write the unit list
    python -m analysis.tasks.B2_benchmark.track_c unit --index I   one unit (rho_p)
    python -m analysis.tasks.B2_benchmark.track_c lists --cohort c named lists of one cohort
    python -m analysis.tasks.B2_benchmark.track_c y4               Y4_regions replication
"""

from __future__ import annotations

import argparse
import json
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import numpy as np
import pandas as pd

from analysis.tasks.B2_benchmark import engine as EN
from analysis.tasks.B2_benchmark import inputs as IN
from analysis.tasks.B2_benchmark import rules as RU
from analysis.tasks.B2_benchmark import rules_c as RC
from analysis.tasks.T4_3_scoreboard import stats as ST

OUT = RU.RES / "track_C"
TL = RU.RES / "tool_scores"
CORE = ["S_g", "D", "B1z", "B1", "B1b", "B1c", "B2b", "B3a", "Sdir", "Lfc"]
FOCAL = ["S_g", "D", "B1z"]
TOOL_COHORTS = ["liv_hmouse", "test_hmouse", "kidney", "islet"]


def table(cohort: str) -> pd.DataFrame:
    t = IN.cohort_table(cohort)
    b = pd.read_parquet(OUT / f"baselines__{cohort}.parquet").set_index("hsapiens_gene_id")
    t = t.set_index("hsapiens_gene_id")
    for k in ("B1z", "B1b", "B1c", "B2b", "B3a", "Sdir", "Lfc"):
        t[k] = b[k].reindex(t.index)
    t["B1"] = b["B1"].reindex(t.index)
    t["S_g"] = t["S_ema"]
    t["var_name"] = b["var_name"].reindex(t.index)
    z4 = RU.REPO / "results/analysis/Z4/h1/genes" / f"{cohort}.parquet"
    if z4.exists():  # per-gene-gate Delta-alpha (Z4_free_gate_all, H1_harmonise schema), a Chimera readout beside D
        t["D_free"] = pd.read_parquet(z4).set_index("hsapiens_gene_id")["D"].reindex(t.index)
    rng = np.random.default_rng(RU.useed("random", cohort))
    t["random"] = rng.random(len(t))
    if cohort in TOOL_COHORTS:
        for tool in RU.TOOLS:
            fr = []
            for r in RU.REPS:
                p = TL / tool / f"{RU.run_name(cohort, 'full', r)}.csv"
                if p.exists():
                    fr.append(pd.read_csv(p).set_index("var_name")["score"])
            if fr:  # SATURN one replicate on most settings, CAMEX three, by budget
                M = np.column_stack([f.reindex(t["var_name"]).to_numpy(dtype=float) for f in fr])
                with np.errstate(invalid="ignore"):
                    n = np.isfinite(M).sum(1)
                    v = np.nansum(M, 1) / np.where(n > 0, n, 1)
                v[n < 1] = np.nan
                t[tool] = v
    return t.reset_index()


def strata(t: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {"all": np.ones(len(t), bool), "count_supported": t["count_support"].fillna(False).to_numpy(dtype=bool)}
    idn = t["identity_human_mouse"].to_numpy(dtype=float)
    if np.isfinite(idn).mean() < 0.5:  # cohorts with no human-mouse pair: mean identity prior
        idn = t["prior_raw"].to_numpy(dtype=float)
    for name, lo, hi in RU.ID_BANDS:
        out[name] = (idn >= lo) & (idn < hi)
    e = t["log_cp10k_spmean"].rank(pct=True).to_numpy()
    for q in range(RU.EXPR_BANDS):
        out[f"expr_Q{q + 1}"] = (e > q / RU.EXPR_BANDS) & (e <= (q + 1) / RU.EXPR_BANDS)
    return out


def unit_list() -> list[dict]:
    rows = []
    for c, cfg in RC.COHORTS_C.items():
        if not (RU.REPO / "results/analysis/H1/genes" / f"{c}.parquet").exists():
            continue
        names = ["all", "count_supported"] + [b[0] for b in RU.ID_BANDS] + [f"expr_Q{q + 1}" for q in range(RU.EXPR_BANDS)]
        for kind, gold, covs in cfg["golds"]:
            for s in names:
                rows.append(dict(cohort=c, kind=kind, gold=gold, stratum=s))
    return rows


def score_unit(u: dict) -> None:
    c, kind, gold, stn = u["cohort"], u["kind"], u["gold"], u["stratum"]
    out = OUT / "units" / f"{c}__{kind}__{stn}.parquet"
    covs = next(x[2] for x in RC.COHORTS_C[c]["golds"] if x[0] == kind)
    t = table(c)
    st = strata(t)[stn]
    G = t[gold].to_numpy(dtype=float) if gold in t else np.full(len(t), np.nan)
    ok0 = np.isfinite(G) & t[covs].notna().all(1).to_numpy() & st & t["block"].notna().to_numpy()
    rows = []
    if ok0.sum() < RU.STRATUM_MIN:
        print("too few", u, int(ok0.sum()), flush=True)
        pd.DataFrame([dict(track="C", task=f"C|{kind}|{c}|{stn}", cohort=c, setting=stn, gold=gold, metric="rho_p", method="", value=np.nan, n=int(ok0.sum()))]).to_parquet(out)
        return
    task = EN.Task(t, RU.N_DRAWS_C, RU.useed("C", c, kind, stn))
    scores = {k: t[k].to_numpy(dtype=float) for k in CORE + ["random"] + [x for x in RU.TOOLS if x in t]}
    scores["D_raw"] = t["D_raw"].to_numpy(dtype=float)
    in_core = [k for k in CORE if np.isfinite(scores[k][ok0]).mean() >= 0.5]
    U = ok0.copy()
    for k in in_core:
        U &= np.isfinite(scores[k])
    if U.sum() < RU.STRATUM_MIN:
        U = ok0 & np.isfinite(scores["S_g"]) & np.isfinite(scores["D"])
        in_core = ["S_g", "D"]
    main_extra = {}
    if "D_free" in t and np.isfinite(t["D_free"].to_numpy(dtype=float)[U]).all():  # free gate defined on every gene of U: scored on the same genes as D (Z4_free_gate_all's universe)
        main_extra["D_free"] = t["D_free"].to_numpy(dtype=float)
    task.add("main", U, G, {**{k: scores[k] for k in in_core}, **main_extra, "random": scores["random"], "D_raw": scores["D_raw"]}, covs)
    task.add("pi", U, G, dict(pi=t["pi_score"].to_numpy(dtype=float)), [x for x in covs if x != "prior_raw"])
    side = {"dNdS": t["dNdS_score"].to_numpy(dtype=float), "LOEUF": t["LOEUF_score"].to_numpy(dtype=float)}
    side.update({k: scores[k] for k in CORE if k not in in_core})
    side.update({k: scores[k] for k in RU.TOOLS if k in scores})
    if "D_free" in t and not main_extra:
        side["D_free"] = t["D_free"].to_numpy(dtype=float)
    sides = []
    for k, v in side.items():
        foc = {f: scores[f] for f in FOCAL if f in in_core}
        rw = ok0 & np.isfinite(v)
        for f in foc.values():
            rw &= np.isfinite(f)
        if rw.sum() < RU.STRATUM_MIN:
            continue
        task.add(k, rw, G, {**foc, k: v}, covs)
        sides.append(k)
    task.run()
    base = dict(track="C", task=f"C|{kind}|{c}|{stn}", cohort=c, setting=stn, gold=gold, metric="rho_p", rank_direction="higher", family=kind)
    for r in task.rows():
        if r["group"] in sides and r["method"] in FOCAL:
            continue
        m = r["method"]
        rows.append({**base, "method": m, "value": r["value"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"], "n_blocks": r["n_blocks"],
                     "flag": "given C without pi" if m == "pi" else "", "rank_direction": "" if m in ("random", "D_raw") else "higher"})
    (OUT / "units").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    h = task.h2h(FOCAL + list(main_extra), ["B1", "B1b", "B1c", "B2b", "B3a", "Sdir", "Lfc"], "main") + task.h2h(["S_g", "D"] + list(main_extra), ["B1z"], "main")
    if main_extra:
        h += task.h2h(["D_free"], ["D", "S_g"], "main")
    for k in sides:
        foc = [f for f in FOCAL if f in in_core]
        h += EN.side_h2h(task, k, k, foc)
    if h:
        pd.DataFrame([{**dict(cohort=c, kind=kind, gold=gold, stratum=stn), **x} for x in h]).to_parquet(OUT / "units" / f"{c}__{kind}__{stn}__h2h.parquet")
    print(c, kind, stn, len(rows), flush=True)


def list_sets(t: pd.DataFrame, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Membership and which genes form the comparison. Shay: divergent vs genes with COE above the median; matched: conserved also >= 4-fold across lineages in a species."""
    if name in t and name not in ("Shay", "Shay_matched"):
        v = t[name].fillna(False).to_numpy(dtype=bool)
        return v, np.ones(len(t), bool)
    if name.startswith("Shay"):
        co = t["G_coe"].to_numpy(dtype=float)
        d = t["divergent"].fillna(False).to_numpy(dtype=bool) & np.isfinite(co)
        good = np.isfinite(co) & (co > np.nanmedian(co[np.isfinite(co)]))
        if name == "Shay_matched":
            rg = np.nanmax(t[["human_log2_range_lineage_means", "mouse_log2_range_lineage_means"]].to_numpy(dtype=float), 1)
            good &= rg >= 2
        return d, d | good
    return None


def score_lists(c: str) -> None:
    t = table(c)
    rows = []
    cov_ok = t[RU.COV_C].notna().all(1).to_numpy()
    pi, ex = t["prior_raw"].to_numpy(dtype=float), t["log_cp10k_spmean"].to_numpy(dtype=float)
    methods = CORE + ["pi", "dNdS", "LOEUF"] + [x for x in RU.TOOLS + ["D_free"] if x in t]
    col = {"pi": "pi_score", "dNdS": "dNdS_score", "LOEUF": "LOEUF_score"}
    for name in RC.COHORTS_C[c]["lists"]:
        s = list_sets(t, name)
        if s is None:
            continue
        inset, comp = s
        rng = np.random.default_rng(RU.useed("lists", c, name))
        for m in methods:
            v = t[col.get(m, m)].to_numpy(dtype=float)
            use = cov_ok & np.isfinite(v) & comp
            if inset[use].sum() < 15:
                continue
            idx = np.flatnonzero(use)
            lab = ST.cells(None, ex[idx]) if m == "pi" else ST.cells(pi[idx], ex[idx])
            keep = ST.kept_strata(inset[idx], lab)
            if inset[idx][keep].sum() < 15:
                continue
            a = ST.strat_auroc(v[idx][keep], inset[idx][keep], lab[keep])
            A, _ = ST.auroc_boot(v[idx][keep], inset[idx][keep], lab[keep], RC.N_BOOT_LISTS, rng, auprc=False)
            lo, hi = ST.pct_ci(A)
            below = RC.LIST_DIRECTION[name] == "below"
            val, l2, h2 = (1 - a, 1 - hi, 1 - lo) if below else (a, lo, hi)
            rows.append(dict(track="C", task=f"C|list|{c}|{name}", cohort=c, setting=name, gold=name, family="list", metric="stratified_auroc_directed", method=m, value=val, ci_lo=l2, ci_hi=h2,
                             n=int(keep.sum()), n_set=int(inset[idx][keep].sum()), flag=f"expected {RC.LIST_DIRECTION[name]}; value oriented so that higher is the expected side", rank_direction="higher"))
    (OUT / "lists").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(OUT / "lists" / f"{c}.parquet")
    print(c, len(rows), flush=True)


def y4() -> None:
    """Y4's two survivors on tissues that were not searched: all genes, odd and even autosomes, against the rest, and head-to-head."""
    rows, hh = [], []
    for spec in RC.Y4_REPLICATION:
        for c, kinds in spec["cohorts"].items():
            t = table(c)
            r3 = lambda x: np.ceil((t[x].rank(method="average") / t[x].notna().sum()) * 3).clip(1, 3).to_numpy()  # noqa: E731
            if spec["region"] == "expr_T2__tau_T3":
                reg = (r3("log_cp10k_spmean") == 2) & (r3("tau_shared") == 3)
            else:
                q = np.ceil((t["prior_raw"].rank(method="average") / t["prior_raw"].notna().sum()) * 5).clip(1, 5).to_numpy()
                reg = (q == 5) & (r3("tau_shared") == 2)
            chrom = t["chromosome"].astype(str).to_numpy()
            num = pd.to_numeric(pd.Series(chrom), errors="coerce").to_numpy()
            parts = {"all": np.ones(len(t), bool), "odd": np.isfinite(num) & (num % 2 == 1), "even": np.isfinite(num) & (num % 2 == 0)}
            for kind in kinds:
                gold, covs = next((x[1], x[2]) for x in RC.COHORTS_C[c]["golds"] if x[0] == kind)
                if gold not in t:
                    continue
                G = t[gold].to_numpy(dtype=float)
                for pn, pm in parts.items():
                    task = EN.Task(t, RU.N_DRAWS_C, RU.useed("Y4", spec["region"], c, kind, pn))
                    ok0 = np.isfinite(G) & t[covs].notna().all(1).to_numpy() & pm
                    scores = {k: t[k].to_numpy(dtype=float) for k in ["S_g", "D", "B1z", "B1b", "B1c", "B2b", "B3a", "Sdir"]}
                    inn = ok0 & reg
                    rest = ok0 & ~reg
                    avail = [k for k in scores if np.isfinite(scores[k][inn]).mean() >= 0.5]
                    com = inn.copy()
                    for k in avail:
                        com &= np.isfinite(scores[k])
                    if com.sum() < RU.STRATUM_MIN:
                        rows.append(dict(region=spec["region"], cohort=c, kind=kind, gold=gold, part=pn, method="", value=np.nan, n=int(com.sum())))
                        continue
                    task.add("in", inn & np.isfinite(scores["S_g"]), G, dict(S_g=scores["S_g"], D=scores["D"]), covs)
                    task.add("rest", rest & np.isfinite(scores["S_g"]), G, dict(S_g=scores["S_g"], D=scores["D"]), covs)
                    task.add("common", com, G, {k: scores[k] for k in avail}, covs)
                    task.run()
                    for g, nm, ee in [(x["name"], x["names"][j], x["est"][j]) for x in task.groups for j in range(len(x["names"]))]:
                        e, d = task.get(g, nm)
                        lo, hi = ST.pct_ci(d[np.isfinite(d)])
                        rows.append(dict(region=spec["region"], cohort=c, kind=kind, gold=gold, part=pn, group=g, method=nm, value=e, ci_lo=lo, ci_hi=hi, se=float(np.nanstd(d)), n=len(next(x for x in task.groups if x["name"] == g)["idx"])))
                    e1, d1 = task.get("in", "S_g")
                    e2, d2 = task.get("rest", "S_g")
                    df = d1 - d2
                    rows.append(dict(region=spec["region"], cohort=c, kind=kind, gold=gold, part=pn, group="in_minus_rest", method="S_g", value=e1 - e2, ci_lo=ST.pct_ci(df[np.isfinite(df)])[0], ci_hi=ST.pct_ci(df[np.isfinite(df)])[1], n=len(next(x for x in task.groups if x["name"] == "in")["idx"])))
                    hh += [dict(region=spec["region"], cohort=c, kind=kind, gold=gold, part=pn, **x) for x in task.h2h(["S_g"], ["B1z", "B1b", "B1c", "B2b", "B3a", "Sdir"], "common")]
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "y4_replication.csv", index=False)
    pd.DataFrame(hh).to_csv(OUT / "y4_replication_h2h.csv", index=False)
    print(d[d.group.isin(["in"]) & (d.method == "S_g") & (d.part == "all")].to_string(), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--index", type=int)
    ap.add_argument("--cohort")
    a = ap.parse_args()
    if a.cmd == "units":
        u = unit_list()
        pd.DataFrame(u).to_csv(RU.TASK_DIR / "track_c_units.tsv", sep="\t", index=False)
        print(len(u))
    elif a.cmd == "unit":
        score_unit(pd.read_csv(RU.TASK_DIR / "track_c_units.tsv", sep="\t").iloc[a.index].to_dict())
    elif a.cmd == "lists":
        score_lists(a.cohort if a.cohort else list(RC.COHORTS_C)[a.index])
    elif a.cmd == "y4":
        y4()


if __name__ == "__main__":
    main()
