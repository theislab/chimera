"""D1: runs tried and finished per cohort, setting and method, in the schema of B2_benchmark/completion.py. A fit that goes non-finite under a setting is a result and stays here; a job lost to a bad node is infrastructure, is run again and is not recorded.
Chimera: tried = a health record for the grid row or the B2 row; finished = healthy under the health rule. Direct readouts are computed on the object whether or not the fit finished.
    python -m revision.tasks.D1_sparse_genes.completion
"""

from __future__ import annotations

import json

import pandas as pd

from revision.tasks.B2_benchmark import rules as B2
from revision.tasks.D1_sparse_genes import rules as RU


def main():
    rows = []
    g = pd.read_csv(RU.TASK / "grid.tsv", sep="\t", keep_default_na=False)
    seen = set()
    for r in g.itertuples():
        h = RU.RES / "health" / f"{r.run}.json"
        if h.exists():
            d = json.loads(h.read_text())
            seen.add((r.cohort, r.setting, r.rep))
            rows.append(dict(cohort=r.cohort, setting=r.setting, rep=r.rep, method="Chimera", tried=1, finished=int(not d["discard"]), source="D1", run=r.run, finite=d.get("finite")))
    for c in RU.DEPTH_COHORTS + ["kidney"]:
        for s in ("depth25", "depth50"):
            for rep in range(3):
                h = B2.RES / "health" / f"b2-{c}-{s}-r{rep}.json"
                if h.exists() and (c, s, rep) not in seen:
                    d = json.loads(h.read_text())
                    rows.append(dict(cohort=c, setting=s, rep=rep, method="Chimera", tried=1, finished=int(not d["discard"]), source="B2", run=h.stem, finite=d.get("finite")))
    d = pd.DataFrame(rows)
    d.to_csv(RU.RES / "completion_runs.csv", index=False)
    t = d.groupby(["cohort", "setting", "method"], as_index=False)[["tried", "finished"]].sum()
    t.to_csv(RU.RES / "completion.csv", index=False)
    print(t.to_string())


if __name__ == "__main__":
    main()
