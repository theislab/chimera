"""Z3 comparison tables, after the scoreboard lines (rule in the private ledger Z3_free_gate.md, committed first).

Sections:
  check     frozen line through the harness against results/analysis/T5_6/scoreboard (and T4_3), every shared value
  tables    H1-H4 (all, count-supported, masked), status and row, head-to-heads, tools, sets, 4-sp estimates, paired
            free-vs-canonical deltas and the verdict by the committed rule
  reversal  H2 raw Spearman, rho_pi, rho_p by line
  x5        X5 run.py definitions on each line's consensus (frozen reproduced first)
  y3        X4 re_x1 (spermatocyte-peak genes, C42) and re_x2 (catabolic genes, C43) matched excess, X4 code and keys
            unchanged, line columns swapped in (frozen reproduced first)

    PYTHONPATH=$PWD python -m analysis.tasks.Z3_free_gate.compare check tables reversal x5 y3
Output: results/analysis/Z3/compare_*.csv, verdict.json
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analysis.tasks.Z3_free_gate import rules as RU

OUT = RU.OUT
FROZEN = RU.RES / "T5_6/scoreboard"
SUFFIX = "_c4sp" if "--c4sp" in sys.argv else ""  # interim: 4-sp rows canonical in every line
LINES = {"frozen": FROZEN, **{k: OUT / f"scoreboard_{k}{SUFFIX}" for k in ("free5", "can5", "can_state", "free_int", "can_int")}}
HYP = ["H1", "H2", "H3", "H4"]
KEYS = ["id", "score", "subset"]


def read(line: str, name: str) -> pd.DataFrame:
    f = LINES[line] / name
    return pd.read_csv(f).assign(line=line) if f.exists() else pd.DataFrame()


def allrho(line: str) -> pd.DataFrame:
    return pd.concat([read(line, "scoreboard.csv"), read(line, "family3.csv")], ignore_index=True)


def check():
    rows = []
    for name in ("scoreboard.csv", "family3.csv", "head_to_head.csv", "sets.csv"):
        a = pd.read_csv(FROZEN / name)
        b = pd.read_csv(OUT / "scoreboard_frozen" / name)
        k = [c for c in ("id", "score", "comparator", "subset") if c in a.columns]
        m = a.merge(b, on=k, suffixes=("_a", "_b"), how="outer", indicator=True)
        num = [c for c in a.columns if c not in k and pd.api.types.is_numeric_dtype(a[c]) and not pd.api.types.is_bool_dtype(a[c]) and c in b.columns]
        worst = max((float(np.nanmax(np.abs(m[f"{c}_a"] - m[f"{c}_b"]))) if m[f"{c}_a"].notna().any() else 0.0) for c in num)
        rows.append(dict(file=name, n_frozen=len(a), n_harness=len(b), unmatched=int((m["_merge"] != "both").sum()), max_abs_diff=worst))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "compare_check.csv", index=False)  # frozen line only, no suffix
    print(t.to_string())


def status(rho, lo, hi, p=None):
    if rho <= -0.03 and hi < 0:
        return "R"
    if p is not None and rho >= 0.10 and lo > 0.03 and p < 0.05:
        return "V"
    if rho >= 0.03 and lo > 0:
        return "W"
    return "N"


def tables():
    R = pd.concat([allrho(k) for k in LINES], ignore_index=True)
    H = pd.concat([read(k, "head_to_head.csv") for k in LINES], ignore_index=True)
    S = pd.concat([read(k, "sets.csv") for k in LINES], ignore_index=True)
    rows = []
    for h in HYP:
        for uni, uid in (("all", f"{h}:D"), ("count_supported", f"{h}:cs:D"), ("masked", f"{h}:masked:D"), ("all_S_ema", f"{h}:S_ema"),
                         ("count_supported_S_ema", f"{h}:cs:S_ema"), ("multi_4sp_or_3sp", f"{h}:multi:D"), ("multi_4sp_count_supported", f"{h}:multi_cs:D"),
                         ("beyond_64PCs", f"{h}:beyond_pcs"), ("B3b", f"{h}:B3b_vs_D")):
            for line in LINES:
                x = R[(R.line == line) & (R.id == uid)]
                if uid.endswith("B3b_vs_D"):
                    x = x[x.score == "B3b"]
                if x.empty:
                    continue
                x = x.iloc[0]
                r = dict(hyp=h, universe=uni, line=line, n=x.n, rho_p=x.rho_p, ci_lo=x.ci_lo, ci_hi=x.ci_hi, spearman=x.spearman, rho_pi=x.rho_pi,
                         p_perm=x.get("p_perm"), p_holm=x.get("p_holm"), seed_loo_min=x.get("seed_loo_min"), seed_loo_max=x.get("seed_loo_max"),
                         ceiling=x.get("ceiling"))
                p = x.get("p_holm") if uni == "all" else x.get("p_perm")
                r["status_rule"] = status(x.rho_p, x.ci_lo, x.ci_hi, p if pd.notna(p) else None)
                rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f"compare_hyp{SUFFIX}.csv", index=False)
    # head-to-heads (J), all and count-supported, and tools
    hh = H[H.id.str.match(r"^H\d:(h2h|cs:h2h)$") | H.id.str.contains(r":(?:cs:)?tool:")].copy()
    hh["universe"] = np.where(hh.id.str.contains(":cs:"), "count_supported", "all")
    hh = hh[["line", "hyp", "universe", "id", "focal", "comparator", "n_pair", "rho_p_focal", "rho_p_comparator", "delta", "pc_lo", "pc_hi", "sim95_lo", "sim95_hi"]]
    hh["call"] = np.where((hh.delta >= 0.03) & (hh.pc_lo > 0), "win", np.where((hh.delta <= -0.03) & (hh.sim95_hi < 0), "loss", "tie"))
    hh.to_csv(OUT / f"compare_h2h{SUFFIX}.csv", index=False)
    # paired free vs canonical
    pr = H[(H.line == "free5") & H.id.str.contains("pair_")].copy()
    pr["universe"] = np.where(pr.id.str.contains("pair_cs"), "count_supported", "all")
    pr["multi"] = pr.id.str.contains("multi_")
    pr["call"] = np.where((pr.delta >= 0.03) & (pr.pc_lo > 0), "free better", np.where((pr.delta <= -0.03) & (pr.pc_hi < 0), "free worse", "no difference shown"))
    pr = pr[["hyp", "cohort", "universe", "multi", "focal", "comparator", "n_pair", "rho_p_focal", "rho_p_comparator", "delta", "pc_lo", "pc_hi", "call"]]
    pr.to_csv(OUT / f"compare_pairs{SUFFIX}.csv", index=False)
    verdict = {}
    for comp in ("D_can5", "D_can_state", "D_frozen", "D_can_int", "S_free5"):
        for uni in ("count_supported", "all"):
            x = pr[(pr.comparator == comp) & (pr.universe == uni) & ~pr.multi]
            if comp == "D_can_state":
                x = pd.concat([x, pr[(pr.comparator == "D_can5") & (pr.universe == uni) & ~pr.multi & (pr.cohort != "liv_hmouse")]])
            b, w = int((x.call == "free better").sum()), int((x.call == "free worse").sum())
            v = "better" if b and not w else "worse" if w and not b else "mixed" if b and w else "no difference shown"
            verdict[f"{comp}:{uni}"] = dict(verdict=v, better=sorted(x.hyp[x.call == "free better"]), worse=sorted(x.hyp[x.call == "free worse"]))
    rowj = {k: json.loads((LINES[k] / "row.json").read_text()) for k in LINES if (LINES[k] / "row.json").exists()}
    verdict["decide"] = {k: dict(status=v["step1_status"], h2h=v["step2_head_to_head"], row=v["step3"]["row"],
                                 modifiers=v["modifiers_applied"]) for k, v in rowj.items()}
    (OUT / f"verdict{SUFFIX}.json").write_text(json.dumps(verdict, indent=1, default=str))
    # sets (Family 2 D and count-supported)
    ss = S[S.id.str.match(r"^set:(liver|testis):[CV]\d:(D|D:cs|D3|S_deploy_native|S_deploy_1to1|C_cos)$")]
    ss = ss[["line", "id", "tissue", "set", "score", "subset", "n_set", "auroc", "auroc_ci_lo", "auroc_ci_hi", "p_perm", "q_bh", "pass_predicted_direction"]]
    ss.to_csv(OUT / f"compare_sets{SUFFIX}.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 30, "display.max_rows", 400)
    print(T.round(3).to_string())
    print(pr.round(3).to_string())
    print(json.dumps(verdict, indent=1, default=str))


def reversal():
    T = pd.read_csv(OUT / f"compare_hyp{SUFFIX}.csv")
    x = T[(T.hyp == "H2") & T.universe.isin(["all", "count_supported", "masked", "all_S_ema", "count_supported_S_ema"])]
    x = x[["universe", "line", "n", "spearman", "rho_pi", "rho_p", "ci_lo", "ci_hi", "status_rule"]]
    x.to_csv(OUT / f"compare_reversal{SUFFIX}.csv", index=False)
    print(x.round(3).to_string())


def x5():
    gold = pd.read_csv(RU.RES / "T4_3/golds/gold_table.csv", index_col=0)[["dNdS"]]
    rows = []
    for c in RU.COHORTS:
        cons = pd.read_csv(RU.OUT / "consensus" / f"{c}.csv")
        ids = pd.read_parquet(RU.RES / f"T4_3/cache/{c}__inputs.parquet", columns=["var_name", "hsapiens_gene_id"])
        fz = pd.read_csv(RU.RES / f"T3_3/{c}__consensus.csv")
        assert (ids.var_name.to_numpy() == cons.var_name.to_numpy()).all() and (fz.var_name.to_numpy() == cons.var_name.to_numpy()).all()
        cons["hsapiens_gene_id"] = ids.hsapiens_gene_id.to_numpy()
        cons["D_frozen"], cons["alpha_frozen"], cons["S_frozen"] = fz.D.to_numpy(), fz.alpha.to_numpy(), fz.S_ema.to_numpy()
        lines = [k for k in ["frozen", "free5", "can5", "can_state", "free_int", "can_int"] if f"D_{k}" in cons.columns]
        d0 = cons.set_index("hsapiens_gene_id")
        d0 = d0[~d0.index.duplicated()].join(gold)
        for line in lines:
            d = d0.dropna(subset=["dNdS", f"D_{line}", f"alpha_{line}", "prior_raw"])
            d = d[d.dNdS > 0]
            pi, logw = d.prior_raw.to_numpy(), np.log10(d.dNdS.to_numpy().clip(1e-3))
            a, D = d[f"alpha_{line}"].to_numpy(), d[f"D_{line}"].to_numpy()
            sup = (d[f"S_{line}"].abs() > 1e-6).to_numpy()
            for x, y, name in ((pi, logw, "pi_vs_dNdS"), (pi, a, "pi_vs_alpha"), (pi, D, "pi_vs_D"), (logw, D, "dNdS_vs_D")):
                rows.append(dict(cohort=c, line=line, panel=name, n=len(x), rho=spearmanr(x, y)[0], n_count_evidence=int(sup.sum()),
                                 rho_count_evidence=spearmanr(x[sup], y[sup])[0]))
    t = pd.DataFrame(rows)
    ref = pd.read_csv(RU.RES / "X5/sequence_axis_rho.csv")
    chk = t[t.line == "frozen"].merge(ref, on=["cohort", "panel"], suffixes=("", "_x5"))
    err = float(np.nanmax(np.abs(chk.rho - chk.rho_x5).tolist() + np.abs(chk.rho_count_evidence - chk.rho_count_evidence_x5).tolist()))
    print("X5 frozen reproduction, max abs diff", err)
    if err > 1e-9:
        raise SystemExit("X5 frozen not reproduced")
    t.to_csv(OUT / f"compare_x5{SUFFIX}.csv", index=False)
    print(t.pivot_table(index=["cohort", "panel"], columns="line", values=["rho", "rho_count_evidence"]).round(3).to_string())


def y3():
    from analysis.tasks.X4_where_differs import analyses as A
    from analysis.tasks.X4_where_differs.common import matched

    rows = []

    def swap(g, c, line):
        if line == "frozen":
            return g
        cons = pd.read_csv(RU.OUT / "consensus" / f"{c}.csv").set_index("var_name")
        g = g.copy()
        g["D"] = cons[f"D_{line}"].reindex(g.var_name).to_numpy()
        g["D_raw"] = cons[f"D_raw_{line}"].reindex(g.var_name).to_numpy()
        g["S"] = cons[f"S_{line}"].reindex(g.var_name).to_numpy()
        return g

    def cs_mask(g, c):
        cons = pd.read_csv(RU.OUT / "consensus" / f"{c}.csv").set_index("var_name")
        if "S_free5" not in cons:
            return np.zeros(len(g), bool)
        return ((cons.S_free5.abs() >= RU.EPS) & (cons.S_can5.abs() >= RU.EPS)).reindex(g.var_name).to_numpy()

    c = "test_hmouse"
    g0 = A.prepare(c)
    cs = cs_mask(g0, c)
    for line in ("frozen", "free5", "can5", "free_int", "can_int"):
        g = swap(g0, c, line)
        ms = g.margin_support.to_numpy()
        for k, f in (("SC", (g.peak == "SC").to_numpy()), ("SC_strict", (g.peak_strict == "SC").to_numpy())):
            for v in ("D_raw", "D", "S"):
                rows.append(dict(item="C42", cohort=c, line=line, group=k, value=v, subset="all", **matched(g, f, ~f, v, A.CP, key=f"{c}{k}{v}")))
                rows.append(dict(item="C42", cohort=c, line=line, group=k, value=v, subset="count_supported",
                                 **matched(g, f & cs, ~f & cs, v, A.CP, key=f"{c}{k}{v}cs")))
            for v in ("D_raw", "S"):
                rows.append(dict(item="C42", cohort=c, line=line, group=k, value=v, subset="margin_supported",
                                 **matched(g, f & ms, ~f & ms, v, A.CP, key=f"{c}{k}{v}ms")))
    lst = pd.read_csv(RU.RES / "X2/divergent_conserved.csv")
    prog = sorted(set(lst[lst.cohort == "liv_4sp"].symbol) - {"MT-CO1"})
    sets = {"prog63_noMT": prog, "core15": A.PROG_CORE}
    for c in ("liv_4sp", "liv_hmouse"):
        g0 = A.prepare(c)
        cs = cs_mask(g0, c)
        chep = A.CP + ["hep_enrich"]
        have = pd.read_csv(RU.OUT / "consensus" / f"{c}.csv", nrows=1).columns
        for line in [k for k in ("frozen", "free5", "can5", "can_state", "free_int", "can_int") if k == "frozen" or f"D_{k}" in have]:
            g = swap(g0, c, line)
            for k, names in sets.items():
                f = g.symbol.isin(names).to_numpy()
                pool = ~g.symbol.isin(prog).to_numpy()
                for v in ("D_raw", "S"):
                    rows.append(dict(item="C43", cohort=c, line=line, group=k, value=v, subset="all", **matched(g, f, pool, v, chep, key=f"{c}{k}{v}")))
                    rows.append(dict(item="C43", cohort=c, line=line, group=k, value=v, subset="count_supported",
                                     **matched(g, f & cs, pool & cs, v, chep, key=f"{c}{k}{v}cs")))
    t = pd.DataFrame(rows)
    # frozen reproduction against X4
    x1 = pd.read_csv(RU.RES / "X4/reread_x1_stages.csv")
    x2 = pd.read_csv(RU.RES / "X4/reread_x2_program.csv")
    a = t[(t.line == "frozen") & (t.item == "C42") & t.subset.isin(["all", "margin_supported"])].merge(x1, on=["cohort", "group", "value", "subset"], suffixes=("", "_x4"))
    b = t[(t.line == "frozen") & (t.item == "C43") & (t.subset == "all")].merge(x2.rename(columns={"set": "group"}), on=["cohort", "group", "value"], suffixes=("", "_x4"))
    err = max(float(np.nanmax(np.abs(a.excess_d - a.excess_d_x4))), float(np.nanmax(np.abs(b.excess_d - b.excess_d_x4))))
    print("Y3 frozen reproduction (X4), rows", len(a), len(b), "max abs diff", err)
    if err > 1e-9:
        raise SystemExit("X4 frozen not reproduced")
    t.to_csv(OUT / f"compare_y3{SUFFIX}.csv", index=False)
    print(t[["item", "cohort", "line", "group", "value", "subset", "n_set", "excess_d", "ci_lo", "ci_hi", "p"]].round(3).to_string())


if __name__ == "__main__":
    for s in [a for a in sys.argv[1:] if not a.startswith("--")]:
        print(f"== {s}", flush=True)
        dict(check=check, tables=tables, reversal=reversal, x5=x5, y3=y3)[s]()
