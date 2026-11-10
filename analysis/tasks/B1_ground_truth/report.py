"""Report tables from the tidy file: main table per alteration x method, predictions against outcomes, low-expression power.

Writes results/analysis/B1/{main_table.csv, low_expression.csv, type1.csv, chimera_vs_best.csv} and prints them.
"""

from __future__ import annotations

import pandas as pd

from analysis.tasks.B1_ground_truth import rules as RU
from analysis.tasks.B1_ground_truth.figures import ORDER, parse

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)
pd.set_option("display.max_rows", 200)
TOP = {"pattern": "1", "level": "8", "loss": "1"}


def main():
    t = parse(pd.read_csv(RU.RES / "B1_tidy.csv"))
    rows = []
    for task, top in TOP.items():
        for c in ("all", "liv_hmouse", "test_hmouse", "kidney"):
            for eff in ("all", top):
                for band in ("all", "low", "midlow", "midhigh", "high"):
                    d = t[(t.task == task) & (t.cohort == c) & (t.effect == eff) & (t.band == band) & (t.share == "all") & (t.identity == "all")]
                    p = d.pivot(index="method", columns="metric", values="value")
                    for m, r in p.iterrows():
                        rows.append(dict(task=task, cohort=c, effect=eff, band=band, method=m, **r.to_dict()))
    m = pd.DataFrame(rows)
    m.to_csv(RU.RES / "main_table.csv", index=False)
    show = m[(m.cohort == "all") & (m.band == "all") & (m.effect != "all")]
    for task in TOP:
        s = show[show.task == task].set_index("method").reindex([x for x in ORDER if x in set(show[show.task == task].method)])
        print(task, s[["AUROC", "AUROC_cov", "AUPRC_genome", "topk_genome", "power_05", "dir_acc", "coverage"]].round(3), sep="\n")
    low = m[(m.effect == "all") & (m.cohort.isin(["liv_hmouse", "test_hmouse", "kidney", "all"]))].pivot_table(index=["task", "cohort", "method"], columns="band", values="power_05")
    low.to_csv(RU.RES / "low_expression.csv")
    print(low.loc[(slice(None), "all", ["S_g", "S_fwd", "Delta_alpha", "Sdir", "B1", "B1c", "Lfc", "expr"]), :].round(3))
    fz = m[(m.effect == "all") & (m.method == "S_g")].pivot_table(index=["task", "cohort"], columns="band", values="frac_zero")
    print(fz.round(3))
    n = t[(t.task.isin(["placebo", "sham_swap", "thin_both"])) & (t.metric.isin(["reject05", "AUROC", "power_05"])) & (t.band == "all") & (t.effect == "all") & (t.share == "all") & (t.identity == "all")]
    y = n.pivot_table(index=["task", "metric", "method"], columns="cohort", values="value")
    y.to_csv(RU.RES / "type1.csv")
    print(y.round(3))


if __name__ == "__main__":
    main()
