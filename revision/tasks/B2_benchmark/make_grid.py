"""Write the Track B manifests: one per cohort, with fits and direct-only rows; the reference rows. Order: settings as in rules."""

from __future__ import annotations

import pandas as pd

from revision.tasks.B2_benchmark import rules as RU


def main():
    rows = pd.DataFrame(RU.grid())
    rows.to_csv(RU.TASK_DIR / "grid_all.tsv", sep="\t", index=False)
    pd.DataFrame(RU.ref_rows()).to_csv(RU.TASK_DIR / "grid_ref.tsv", sep="\t", index=False)
    # fits: per cohort so that each array asks the cohort's resources; direct-only rows wait for their fits only at aggregation
    for c in RU.COHORTS:
        d = rows[rows.cohort == c]
        d.to_csv(RU.TASK_DIR / f"grid_{c}.tsv", sep="\t", index=False)
        print(c, "rows", len(d), "fits", int(d.fit.sum()))


if __name__ == "__main__":
    main()
