"""E6 summary tables from results/analysis/E6/auroc.tsv and genes.tsv.gz.

primary.csv     one row per (assignment, set) on the primary cohort, supported universe, tau strata: AUROC [CI] of every score
all_cohorts.csv the same for every cohort
predictions.csv P1-P7 as computed
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as RU  # noqa: E402


def fmt(r):
    return f"{r.auroc:.3f} ({r.lo:.3f}, {r.hi:.3f})" if pd.notna(r.auroc) else ""


def wide(A):
    keys = ["assign_id", "source_id", "circular", "cohort", "set", "comparison"]
    A = A.assign(txt=A.apply(fmt, axis=1))
    W = A.pivot_table(index=keys, columns="score", values=["auroc", "lo", "hi", "p_lower", "txt"], aggfunc="first")
    W.columns = [f"{a}_{b}" for a, b in W.columns]
    n = A.groupby(keys).n_list.max().rename("n")
    return W.join(n).reset_index()


def drop_full_where_core(T):
    """Keep the core of a set where it exists, per assignment (a full set is dropped only if its own assignment has its core)."""
    cores = {(a, s.replace("__core", "")) for a, s in zip(T.assign_id, T.set) if s.endswith("__core")}
    return T[[(a, s) not in cores for a, s in zip(T.assign_id, T.set)]]


def main():
    A = pd.read_csv(RU.OUT / "auroc.tsv", sep="\t", dtype={"circular": str})
    A["is_primary"] = A.cohort == A.primary
    sup = A[(A.universe == "supported") & A.strata.isin(["expr_tau_pi", "expr3_tau3"])]
    W = wide(sup)
    W.to_csv(RU.OUT / "all_cohorts.csv", index=False)
    P = W[W.cohort.isin(A[A.is_primary].cohort.unique())]
    P = P.merge(A[["assign_id", "primary"]].drop_duplicates(), on="assign_id")
    P = P[P.cohort == P.primary].drop(columns="primary")
    P.to_csv(RU.OUT / "primary.csv", index=False)
    # truth sets: not controls; core where it exists
    truth = P[(P.comparison == "vs_matched") & ~P.set.str.match(r"(conserved|text_conserved)")]
    truth = drop_full_where_core(truth)
    rows = []
    ind = truth[(truth.circular == "no") & (truth.n >= RU.MIN_LIST_PRED)]
    below = (ind.hi_S_can < 0.5)
    rows.append(dict(pred="P1", text="non-circular, n>=30: S_can CI upper < 0.5 in at least half", n_pairs=len(ind), n_hit=int(below.sum()),
                     held=bool(below.sum() >= len(ind) / 2)))
    d = (ind.auroc_D_free - 0.5).abs() - (ind.auroc_S_can - 0.5).abs()
    rows.append(dict(pred="P2a", text="|D_free-0.5| within 0.03 of |S_can-0.5|", n_pairs=len(ind), n_hit=int((d.abs() <= 0.03).sum()),
                     held=bool((d.abs() <= 0.03).mean() >= 0.5)))
    c = (ind.auroc_D_can - 0.5).abs() < (ind.auroc_S_can - 0.5).abs()
    rows.append(dict(pred="P2b", text="D_can closer to 0.5 than S_can", n_pairs=len(ind), n_hit=int(c.sum()), held=bool(c.mean() >= 0.5)))
    b = (ind.auroc_B1 < 0.5) & ((0.5 - ind.auroc_B1) >= (0.5 - ind.auroc_S_can))
    rows.append(dict(pred="P3", text="B1 below 0.5 and at least as far as S_can", n_pairs=len(ind), n_hit=int(b.sum()), held=bool(b.mean() >= 0.5)))
    lv = (0.5 - ind.auroc_LVL) > (0.5 - ind.auroc_S_can)
    rows.append(dict(pred="P4", text="LVL further below 0.5 than S_can", n_pairs=len(ind), n_hit=int(lv.sum()), held=bool(lv.mean() >= 0.5)))
    circ = truth[truth.circular == "yes"]
    rows.append(dict(pred="P5", text="median S_can AUROC circular < independent", n_pairs=len(circ),
                     n_hit=float(circ.auroc_S_can.median()), held=bool(circ.auroc_S_can.median() < ind.auroc_S_can.median()),
                     note=f"independent median {ind.auroc_S_can.median():.3f}"))
    ctl = P[(P.comparison == "vs_conserved_control") & ~P.set.str.match(r"(conserved|text_conserved)")]
    ctl = drop_full_where_core(ctl)
    rows.append(dict(pred="P6", text="divergent below conserved on S_can (AUROC<0.5)", n_pairs=len(ctl), n_hit=int((ctl.auroc_S_can < 0.5).sum()),
                     held=bool((ctl.auroc_S_can < 0.5).mean() >= 0.5), note="B1: " + str(int((ctl.auroc_B1 < 0.5).sum()))))
    nt = A[(A.universe == "supported") & (A.strata == "expr_pi") & (A.score == "S_can") & (A.comparison == "vs_matched")]
    m = ind.merge(nt[["assign_id", "cohort", "set", "auroc"]], on=["assign_id", "cohort", "set"])
    up = m.auroc > m.auroc_S_can
    rows.append(dict(pred="P7", text="without tau, S_can AUROC moves up", n_pairs=len(m), n_hit=int(up.sum()), held=bool(up.mean() >= 0.5)))
    pd.DataFrame(rows).to_csv(RU.OUT / "predictions.csv", index=False)
    cols = ["assign_id", "circular", "cohort", "set", "n", "txt_S_can", "txt_D_free", "txt_D_can", "txt_B1", "txt_LVL"]
    with pd.option_context("display.width", 250, "display.max_colwidth", 40, "display.max_rows", 500):
        print(P[P.comparison == "vs_matched"][cols].to_string(index=False))
        print(P[P.comparison == "vs_conserved_control"][cols].to_string(index=False))
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
