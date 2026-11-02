"""Write the D1 fit grid and the chromosome folds. No readout is read. Tiers fix the order of work; whole tiers finished at the stop time enter the analysis, by time alone.
   python -m revision.tasks.D1_sparse_genes.make_grid
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from revision.common import gene_ids
from revision.tasks.B2_benchmark import rules as B2
from revision.tasks.D1_sparse_genes import rules as RU

AUT = [str(i) for i in range(1, 23)]


def genes_per_chrom() -> pd.DataFrame:
    ch = pd.read_csv(RU.REPO / "results/revision/X1/genes/chromosome_ensembl110.csv", dtype={"chrom": str}).set_index("hsapiens_gene_id")["chrom"]
    out = {}
    for c in RU.DEPTH_COHORTS:
        h = pd.read_parquet(RU.REPO / f"results/revision/H1/genes/{c}.parquet")
        out[c] = h.hsapiens_gene_id.map(ch).value_counts()
    return pd.DataFrame(out).reindex(AUT).fillna(0)


def fold(tab: pd.DataFrame, chroms: list[str], rng, draws=20000) -> list[str]:
    tot = tab.sum(0)
    best, bd = None, 9.0
    for _ in range(draws):
        k = rng.integers(2, 8)
        pick = list(rng.choice(chroms, k, replace=False))
        share = (tab.loc[pick].sum(0) / tot).to_numpy()
        d = np.abs(share - RU.HOLD_SHARE).max()
        if d < bd:
            best, bd = pick, d
    return sorted(best, key=int)


def main():
    tab = genes_per_chrom()
    rng = np.random.default_rng(RU.useed("folds"))
    A = fold(tab, [c for c in AUT if int(c) % 2 == 1], rng)
    B = fold(tab, [c for c in AUT if int(c) % 2 == 0], rng)
    tot = tab.sum(0)
    fr = pd.DataFrame([dict(fold="A", chromosomes=",".join(A), **(tab.loc[A].sum(0) / tot).to_dict()),
                       dict(fold="B", chromosomes=",".join(B), **(tab.loc[B].sum(0) / tot).to_dict())])
    RU.RES.mkdir(parents=True, exist_ok=True)
    fr.to_csv(RU.TASK / "folds.tsv", sep="\t", index=False)
    print(fr.to_string())
    rows = []

    def add(tier, cohort, setting, family, rep, seed, arm=""):
        rows.append(dict(tier=tier, cohort=cohort, setting=setting, family=family, rep=rep, seed=seed, run=f"d1-{cohort}-{setting}-r{rep}", fit=1, arm=arm))

    # tier 1: arm P both folds, two seeds, three cohorts; depth r0
    for c in RU.DEPTH_COHORTS:
        for f in "AB":
            for r in (0, 1):
                add(1, c, f"ho{f}_P", "heldout", r, RU.HOLD_SEEDS[r], "P")
        for s in ("depth5", "depth10", "depth25"):
            if not (c == "liv_hmouse" and s == "depth25"):  # B2 fits exist
                add(1, c, s, "sampling", 0, B2.FIT_SEEDS[0])
    # tier 2: depth r1
    for c in RU.DEPTH_COHORTS:
        for s in ("depth5", "depth10", "depth25"):
            if not (c == "liv_hmouse" and s == "depth25"):
                add(2, c, s, "sampling", 1, B2.FIT_SEEDS[1])
    # tier 3: arm E, one seed per fold
    for c in RU.DEPTH_COHORTS:
        for f in "AB":
            add(3, c, f"ho{f}_E", "heldout", 0, RU.HOLD_SEEDS[0], "E")
    # added after both 4-sp fold A seeds failed the health rule; at the end, so row numbers stay
    add(1, "liv_4sp", "hoA_P", "heldout", 2, RU.HOLD_SEEDS[2], "P")
    add(1, "liv_4sp", "hoA_P", "heldout", 3, RU.HOLD_SEEDS[3], "P")
    g = pd.DataFrame(rows)
    g.to_csv(RU.TASK / "grid.tsv", sep="\t", index=False)
    print(g.groupby(["tier", "cohort"]).size())


if __name__ == "__main__":
    main()
