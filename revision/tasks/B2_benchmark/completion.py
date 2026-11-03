"""Runs tried and finished, per method and setting: a method's failure under a setting is itself a result of the robustness track. Failures stay out of the scores and are reported here and in the overview.

Chimera: tried = a run folder or health record exists for the grid row; finished = healthy under the health rule (alpha, z, S_g finite, 10 active latent dims).
Chimera also: finished_gate_tail, finished with gate tails of at least 2% (the larger of the two tails beyond 0.05, D1's definition): a flat gate finishes without a reading.
Tools: tried = a score file or a recorded attempt; finished = a score file with at least one finite score. A job lost to a bad node is infrastructure, not a tool failure, and is run again: its attempt is not recorded.
Writes results/revision/B2/completion.csv: cohort, setting, method, tried, finished.    python -m revision.tasks.B2_benchmark.completion
"""

from __future__ import annotations

import json

import pandas as pd

from revision.common import cohorts as CO
from revision.tasks.B2_benchmark import rules as RU


def table() -> pd.DataFrame:
    rows = []
    for r in RU.grid():
        if int(r["fit"]) != 1:
            continue
        h = RU.RES / "health" / f"{r['run']}.json"
        tried = h.exists() or (CO.RUNS_DIR / r["run"]).exists()
        d = json.loads(h.read_text()) if h.exists() else {}
        fin = h.exists() and not d["discard"]
        tails = max(d.get("tail_pos") or 0.0, d.get("tail_neg") or 0.0)  # as D1: share of genes beyond +-0.05 of the raw prior, the larger side
        rows.append(dict(cohort=r["cohort"], setting=r["setting"], rep=r["rep"], method="Chimera", tried=int(tried), finished=int(fin), finished_gate_tail=int(fin and tails >= 0.02)))
    g = pd.read_csv(RU.TASK_DIR / "tools" / "tool_grid.tsv", sep="\t")
    for tool in RU.TOOLS:
        for r in g.itertuples():
            f = RU.RES / "tool_scores" / tool / f"{r.run}.csv"
            a = RU.RES / "attempts" / tool / f"{r.run}.rc"
            fin = False
            if f.exists():
                fin = bool(pd.read_csv(f)["score"].notna().any())
            tried = f.exists() or a.exists()
            rows.append(dict(cohort=r.cohort, setting=r.setting, rep=r.rep, method=tool, tried=int(tried), finished=int(fin), finished_gate_tail=0))
    d = pd.DataFrame(rows)
    return d.groupby(["cohort", "setting", "method"], as_index=False)[["tried", "finished", "finished_gate_tail"]].sum()


def main():
    t = table()
    from revision.tasks.B2_benchmark import tiers as T

    main_set = set(T.MAIN) | {"ref", "depth25_law", "depth50_law"}
    t["tier_status"] = "run"
    t.loc[(t.cohort != "liv_hmouse") & ~t.setting.isin(main_set), "tier_status"] = "planned, not run (tier 3; owner's scope decision)"
    t.to_csv(RU.RES / "completion.csv", index=False)
    print(t[t.tried > t.finished].to_string())


if __name__ == "__main__":
    main()
