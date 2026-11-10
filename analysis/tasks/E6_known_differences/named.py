"""E6 named genes (rules.py naming rule, plus: the cohort's own detection agrees with the paper's direction).
Output results/analysis/E6/named_final.tsv: recovered lists on their primary cohort, S_g lower decile, nameable (H1), sign stable in
both gates, not saturated, O1 clean, not a DLPFC reference artefact, count direction agrees with the paper by at least 0.05 detection.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules as RU  # noqa: E402

LISTS = [("jor_hs", "human_specific__core"), ("jor_hc", "human_vs_chimp_divergent__core"), ("cag_am", "ape_vs_macaque_divergent__core"),
         ("benner", "human_enriched__core"), ("benner", "mouse_enriched__core")]


def main():
    G = pd.read_csv(RU.OUT / "genes.tsv.gz", sep="\t", low_memory=False)
    A = pd.read_csv(Path(__file__).resolve().parent / "assign.tsv", sep="\t", comment="#")
    G = G.merge(A[["assign_id", "primary"]], on="assign_id")
    G = G[G.cohort == G.primary]
    out = []
    for a, s in LISTS:
        x = G[(G.assign_id == a) & (G.set == s)].copy()
        dc = [c for c in x.columns if c.startswith("detection_") and x[c].notna().all()]
        hc = [c for c in dc if "uman" in c][0]
        oc = [c for c in dc if c != hc]
        x["det_human"], x["det_other_mean"] = x[hc], x[oc].mean(1)
        up = x.direction.eq("up_in_human")
        dn = x.direction.isin(["down_in_human", "up_in_mouse"])
        x["count_direction_agrees"] = (up & (x.det_human > x.det_other_mean)) | (dn & (x.det_human < x.det_other_mean))
        ok = (x.nameable & x.stable & ~x.saturated & x.supp_both & ~x.o1_T2 & ~x.reference_artefact
              & (x.cohort_pct_S_can <= 0.10) & x.count_direction_agrees & ((x.det_human - x.det_other_mean).abs() >= 0.05))
        out.append(x[ok])
    N = pd.concat(out).sort_values(["assign_id", "set", "S_can"])
    cols = ["assign_id", "set", "cohort", "symbol", "cell_type_list", "direction", "effect", "S_can", "D_free", "D_can", "B1",
            "pct_B1", "cohort_pct_S_can", "det_human", "det_other_mean"]
    N[cols].to_csv(RU.OUT / "named_final.tsv", sep="\t", index=False)
    with pd.option_context("display.width", 250, "display.max_rows", 200):
        print(N[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
