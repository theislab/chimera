"""Write the run order: central settings and main effects first, replicates after, then the rest. Appends to the tool grid without renumbering what ran.
    python -m analysis.tasks.B2_benchmark.make_queue <queue file>
Cells dropped for carrying no information, by design: CAMEX and SAMap on label-only settings, since neither reads labels (the counts and the fit are those of `full`, so scoring uses `full` and marks them label-independent).
"""
from __future__ import annotations

import csv
import sys

import pandas as pd

from analysis.tasks.B2_benchmark import rules as RU

MAIN = ["relabel25", "coarse", "nolabel", "type_missing", "size_1to5", "cells25", "depth25", "extra_batch", "chemistry"]
REST = [s for s in RU.SETTINGS if s not in MAIN + ["full"]]
LABEL_FREE_TOOLS_SKIP = {s for s, c in RU.SETTINGS.items() if c["family"] == "label"}


def lg(n):
    return f"--output=results/analysis/B2/logs/{n}_%A_%a.out"


def main():
    qf = sys.argv[1]
    L = []
    for i in range(11, 17):
        L.append(f"sbatch --array={i} -c 4 --mem=24G --job-name=b2ref {lg('b2ref')} analysis/tasks/B2_benchmark/ref.sbatch analysis/tasks/B2_benchmark/grid_ref.tsv")
    for i in range(3, 14):
        L.append(f"sbatch --array={i} -c 6 --mem=64G --job-name=b2c_base {lg('b2c_base')} analysis/tasks/B2_benchmark/track_c.sbatch baselines")
    fit = {c: pd.read_csv(RU.TASK_DIR / f"grid_{c}.tsv", sep="\t", keep_default_na=False) for c in ("test_hmouse", "kidney")}
    res = dict(test_hmouse=("-c 12 --mem=32G", "b2_test"), kidney=("-c 16 --mem=64G", "b2_kid"))

    def fits(settings, rep):
        out = []
        for s in settings:
            for c, g in fit.items():
                for i in g.index[(g.setting == s) & (g.rep == rep)]:
                    a, n = res[c]
                    out.append(f"sbatch --array={i} {a} --job-name={n} {lg(n)} analysis/tasks/B2_benchmark/array.sbatch analysis/tasks/B2_benchmark/grid_{c}.tsv")
        return out

    L += fits(MAIN, 0)
    # integration, all ten methods
    base = pd.read_csv(RU.TASK_DIR / "integration/baselines.tsv", sep="\t")
    for i in range(len(base)):
        L.append(f"sbatch --array={i} -c 8 --mem=48G --job-name=b2i_base {lg('b2i_base')} analysis/tasks/T11_4_vz_integration/cmd_array.sbatch analysis/tasks/B2_benchmark/integration/baselines.tsv")
    for co in ["bone_marrow", "kidney", "islet", "vz_hmouse"]:
        for m in ("CAMEX", "SATURN", "SATURN-Leiden"):
            L.append(f"sbatch -c 16 --mem=64G --job-name=b2i_{m}_{co} --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/integration/tools.sbatch {co} {m}")
    L += fits(MAIN, 1)
    L += fits(REST, 0)
    L += fits(REST, 1)
    # tools: append the settings not yet in the grid, keep the indices of what ran
    gf = RU.TASK_DIR / "tools" / "tool_grid.tsv"
    g = pd.read_csv(gf, sep="\t")
    have = set(zip(g.cohort, g.setting))
    add = []
    for c in RU.COHORTS:
        for s in RU.SETTINGS:
            if (c, s) in have or (RU.SETTINGS[s]["kind"] == "few_donors" and c not in RU.FEW_DONORS):
                continue
            add.append(dict(cohort=c, setting=s, rep=0, seed=RU.FIT_SEEDS[0], run=RU.run_name(c, s, 0)))
    g = pd.concat([g, pd.DataFrame(add)], ignore_index=True)
    g.to_csv(gf, sep="\t", index=False)
    sat = lambda i: f"sbatch --array={i} -c 16 --mem=48G --job-name=b2t5_saturn {lg('b2t5_saturn')} analysis/tasks/B2_benchmark/tools/t5.sbatch saturn"  # noqa: E731
    order = g.assign(i=g.index, m=g.setting.map(lambda s: 0 if s in ["full", "ref"] + MAIN else 1))
    done = {0, 3, 7, 8, 9, 1, 2, 4, 5, 6}
    for _, r in order[(order.rep == 0)].sort_values(["m", "i"]).iterrows():
        if r.i not in done:
            L.append(sat(r.i))
    # CAMEX: not on label-only settings
    cam = g[~g.setting.isin(LABEL_FREE_TOOLS_SKIP)]
    for co, gg in cam.groupby("cohort", sort=False):
        ids = [int(i) for i in gg.index if i != 0]
        for j in range(0, len(ids), 12):
            L.append(f"sbatch -c 8 --mem=32G --job-name=b2t5b_camex --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/tools/t5_bundle.sbatch camex {' '.join(map(str, ids[j:j + 12]))}")
    tb = {c: RU.ALL_COHORTS[c]["gene_table"] for c in RU.ALL_COHORTS}
    mains = ["full"] + MAIN + REST
    for c in RU.COHORTS:
        sett = [s for s in mains if not (RU.SETTINGS[s]["kind"] == "few_donors" and c not in RU.FEW_DONORS)]
        L.append(f"sbatch -c 10 --mem=32G --job-name=b2add_gs --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/tools/added.sbatch genespectra {c} {tb[c]} 0 {' '.join(sett)}")
        for k in range(0, len(sett), 6):
            L.append(f"sbatch -c 2 --mem=32G --job-name=b2add_ca --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/tools/added.sbatch cacimar {c} {tb[c]} 0 {' '.join(sett[k:k + 6])}")
    L.append(f"sbatch -c 10 --mem=32G --job-name=b2add_gs --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/tools/added.sbatch genespectra islet {tb['islet']} 0 full")
    L.append(f"sbatch -c 2 --mem=32G --job-name=b2add_ca --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/tools/added.sbatch cacimar islet {tb['islet']} 0 full")
    L += fits(MAIN + REST, 2)
    # remaining SATURN replicates on main settings
    for _, r in order[(order.rep > 0) & order.setting.isin(RU.TOOL_SETTINGS + ["ref"])].sort_values("i").iterrows():
        if r.i not in done:
            L.append(sat(r.i))
    # anything of the first queue that already ran stays done: its lines are identical
    open(qf, "w").write("\n".join(L) + "\n")
    print(len(L), "lines")


if __name__ == "__main__":
    main()
