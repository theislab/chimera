"""queue.tsv: every task, in the order the feeder submits them: kind, manifest, index, cpus, mem, wall, done_file, tier.

Paths are relative to the repository root, from which the feeder runs.
"""

from __future__ import annotations

import pandas as pd

from analysis.tasks.B1_ground_truth import rules as RU

MEM = {"liv_hmouse": "7G", "test_hmouse": "10G", "kidney": "16G"}  # about 1.3 x observed peak; the per-user cap of 2T is the limit
WALL = {"liv_hmouse": "04:00:00", "test_hmouse": "08:00:00", "kidney": "16:00:00"}
TOOL_MEM = {"camex": "14G", "saturn": "20G"}
# CACIMAR: 4 GB peak on liver and testis; kidney reached the 8 GB request and failed (2026-11-03), so kidney gets 32 GB.
CACIMAR_MEM = {"liv_hmouse": "8G", "test_hmouse": "8G", "kidney": "32G"}


def rel(path) -> str:
    return str(path.relative_to(RU.REPO))


def main():
    rows = []
    for c in ("test_hmouse", "kidney"):
        rows.append(("placebo", c, 0, 8, "32G", "12:00:00", rel(RU.RES / "placebo" / f"{c}.parquet"), -1))
    j = pd.read_csv(RU.TASK_DIR / "track_d" / "jobs.tsv", sep="\t", keep_default_na=False)
    for i, r in j.iterrows():
        rows.append(("trackd", "jobs", i, 16, "96G", "23:50:00", rel(RU.RES / "track_d" / f"{r['job']}.time.txt"), 0))
    for c in RU.COHORTS:
        g = pd.read_csv(RU.TASK_DIR / f"grid_{c}.tsv", sep="\t", keep_default_na=False)
        man = f"analysis/tasks/B1_ground_truth/grid_{c}.tsv"
        for i, r in g.iterrows():
            rows.append(("fit", man, i, 8, MEM[c], WALL[c], rel(RU.RES / "scores" / f"{r['run']}.parquet"), int(r["tier"])))
    t = pd.read_csv(RU.TASK_DIR / "grid_tools_v2.tsv", sep="\t")
    for i, r in t.iterrows():
        for tool in ("camex", "saturn"):
            rows.append((tool, "analysis/tasks/B1_ground_truth/grid_tools_v2.tsv", i, 8 if tool == "camex" else 16, TOOL_MEM[tool], "23:50:00",
                         rel(RU.RES / "tool_scores" / f"{tool}__{r['run']}.parquet"), int(r["tier"])))
    q = pd.DataFrame(rows, columns=["kind", "manifest", "index", "cpus", "mem", "wall", "done_file", "tier"])
    q = q.sort_values("tier", kind="stable")
    q.to_csv(RU.TASK_DIR / "queue.tsv", sep="\t", index=False)
    print(q.groupby(["tier", "kind"]).size())


if __name__ == "__main__":
    main()
