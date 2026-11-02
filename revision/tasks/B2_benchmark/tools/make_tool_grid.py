"""Tool runs: SATURN, CAMEX (and the added tools) on the main settings of Track B and the reference seeds, on each cohort; islet full and ref for Track C."""
from __future__ import annotations

import pandas as pd

from revision.tasks.B2_benchmark import rules as RU


def main():
    rows = []
    for c in list(RU.COHORTS) + list(RU.WAVE2):
        settings = RU.TOOL_SETTINGS if c in RU.COHORTS else ["full"]
        for s in settings:
            for r in RU.REPS:
                rows.append(dict(cohort=c, setting=s, rep=r, seed=RU.FIT_SEEDS[r], run=RU.run_name(c, s, r)))
        for r in RU.REPS:
            rows.append(dict(cohort=c, setting="ref", rep=r, seed=RU.TOOL_REF_SEEDS[r], run=RU.run_name(c, "ref", r)))
    d = pd.DataFrame(rows)
    d.to_csv(RU.TASK_DIR / "tools" / "tool_grid.tsv", sep="\t", index=False)
    print(len(d), d.groupby("cohort").size().to_dict())


if __name__ == "__main__":
    main()
