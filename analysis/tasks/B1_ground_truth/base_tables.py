"""Per-cohort gene table (band, identity band, eligibility inputs): results/analysis/B1/base/<cohort>_genes.parquet."""

from __future__ import annotations

from analysis.tasks.B1_ground_truth import alter as A
from analysis.tasks.B1_ground_truth import rules as RU


def main():
    (RU.RES / "base").mkdir(parents=True, exist_ok=True)
    for c in RU.COHORTS:
        g, m = A.gene_table(A.read_base(c), c)
        g.to_parquet(RU.RES / "base" / f"{c}_genes.parquet")
        print(c, g.shape, m["types"], flush=True)


if __name__ == "__main__":
    main()
