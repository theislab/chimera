"""Write the grid manifests. A row whose gene set would equal that of a smaller share (the band has too few eligible
genes for the larger share) is listed with duplicate_of and not trained: it is the same experiment."""

from __future__ import annotations

import pandas as pd

from revision.tasks.B1_ground_truth import alter as A
from revision.tasks.B1_ground_truth import rules as RU


def main():
    rows = pd.DataFrame(RU.grid())
    rows["duplicate_of"] = ""
    rows["n_pairs"] = 0
    rows["share_achieved"] = 0.0
    rows["recipe"] = rows["cohort"].map(lambda c: RU.COHORTS[c]["recipe"])
    for c in RU.COHORTS:
        a = A.read_base(c)
        g, _ = A.gene_table(a, c)
        for (alt, band, seed), sub in rows[(rows.cohort == c) & (rows.alt != "none")].groupby(["alt", "band", "seed"]):
            np_ = len(A.select_pairs(g, alt, band, seed))
            for i, r in sub.iterrows():
                n = min(int(round(r["share"] * len(g))), np_)
                rows.loc[i, "n_pairs"] = n
                rows.loc[i, "share_achieved"] = n / len(g)
            for e, s2 in sub.groupby("effect"):
                s2 = s2.sort_values("share")
                seen = {}
                for i, r in s2.iterrows():
                    n = rows.loc[i, "n_pairs"]
                    if n in seen:
                        rows.loc[i, "duplicate_of"] = seen[n]
                    else:
                        seen[n] = r["run"]
    rows.to_csv(RU.TASK_DIR / "grid_all.tsv", sep="\t", index=False)
    for c in RU.COHORTS:
        d = rows[(rows.cohort == c) & (rows.duplicate_of == "") & (rows.reuse == "")].copy()
        # order: largest effect and middle share first, so early rows answer the main question first
        d["ord"] = d["share"].map({0.05: 0, 0.10: 1, 0.02: 2}) * 10 + d.groupby(["alt"])["effect"].rank(method="dense", ascending=False)
        d = d.sort_values(["ord", "alt", "band", "seed"]).drop(columns="ord")
        d.to_csv(RU.TASK_DIR / f"grid_{c}.tsv", sep="\t", index=False)
        n = rows[(rows.cohort == c) & (rows.reuse != "")]
        n.to_csv(RU.TASK_DIR / f"grid_{c}_none.tsv", sep="\t", index=False)
        print(c, "train rows", len(d), "of", ((rows.cohort == c) & (rows.reuse == "")).sum(), "none rows", len(n), flush=True)


if __name__ == "__main__":
    main()
