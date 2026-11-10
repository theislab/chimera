"""Run tiers of B2, fixed by the plan's stopping rule while liver alone had been scored in Track B: the narrative pass takes every tier complete at 2026-11-04 14:00, by time alone, never by results.

Main settings: full, relabel25, coarse, nolabel, type_missing, size_1to5, cells25, depth25, extra_batch, chemistry (one per kind of problem, the middle severity, fixed before any scoring as TOOL_SETTINGS).
tier 0  main settings, replicate r0: Chimera and direct baselines, testis and kidney (liver: complete in all tiers). Reference fits scored.
tier 1  main settings, r0: SATURN and CAMEX, each with its reference seed; GeneSpectra and CACIMAR (cheap, on the same objects); integration on the new cohorts, ten methods.
tier 2  main settings, replicates r1, r2: Chimera, baselines, SATURN, CAMEX.
tier 3  remaining settings (other severities and label forms), all replicates: Chimera, baselines; every tool on them; SAMap.
A tier enters scoring only whole. `python -m analysis.tasks.B2_benchmark.tiers status` prints what is complete; `queue <file>` writes the run order.
Liver ran all tiers before this was written, so it is reported in full; testis and kidney enter only by whole tier. Tools outside the complete tiers stay on main settings, stated in the text.
"""

from __future__ import annotations

import sys

import pandas as pd

from analysis.tasks.B2_benchmark import rules as RU

MAIN = list(RU.TOOL_SETTINGS)
REST = [s for s in RU.SETTINGS if s not in MAIN]
LABEL_ONLY = {s for s, c in RU.SETTINGS.items() if c["family"] == "label"}
TOOL_GRID = RU.TASK_DIR / "tools" / "tool_grid.tsv"
TBL = {c: RU.ALL_COHORTS[c]["gene_table"] for c in RU.ALL_COHORTS}
COH = ["test_hmouse", "kidney"]
RES_FIT = dict(test_hmouse=("-c 12 --mem=8G", "b2_test"), kidney=("-c 16 --mem=20G", "b2_kid"))
SAT_MEM = dict(liv_hmouse="12G", test_hmouse="24G", islet="24G", kidney="40G")


def lg(n):
    return f"--output=results/analysis/B2/logs/{n}_%A_%a.out"


def applies(c, s):
    return not (RU.SETTINGS[s]["kind"] == "few_donors" and c not in RU.FEW_DONORS)


def fit_rows():
    out = []
    for c in COH:
        g = pd.read_csv(RU.TASK_DIR / f"grid_{c}.tsv", sep="\t", keep_default_na=False)
        for i, r in g.iterrows():
            t = (0 if int(r["rep"]) == 0 else 2) if r["setting"] in MAIN else 3
            a, n = RES_FIT[c]
            out.append(dict(tier=t, cohort=c, setting=r["setting"], rep=int(r["rep"]), kind="fit", output=f"scores/{r['run']}.parquet",
                            line=f"sbatch --array={i} {a} --job-name={n} {lg(n)} analysis/tasks/B2_benchmark/array.sbatch analysis/tasks/B2_benchmark/grid_{c}.tsv"))
    return out


def tool_rows():
    g = pd.read_csv(TOOL_GRID, sep="\t")
    out = []
    for i, r in g.iterrows():
        c, s, rep = r["cohort"], r["setting"], int(r["rep"])
        main = s in MAIN + ["ref"]
        t = (1 if rep == 0 else 2) if main else 3
        if c == "liv_hmouse" and False:
            continue
        out.append(dict(tier=t, cohort=c, setting=s, rep=rep, kind="SATURN", output=f"tool_scores/SATURN/{r['run']}.csv",
                        line=f"sbatch --array={i} -c 16 --mem={SAT_MEM[c]} --job-name=b2t5_saturn {lg('b2t5_saturn')} analysis/tasks/B2_benchmark/tools/t5.sbatch saturn"))
    cam = g[~g.setting.isin(LABEL_ONLY)]
    for t in (1, 2, 3):
        sub = cam[[(1 if int(r.rep) == 0 else 2) == t if r.setting in MAIN + ["ref"] else t == 3 for r in cam.itertuples()]]
        for c, gg in sub.groupby("cohort", sort=False):
            ids = [int(i) for i in gg.index]
            for j in range(0, len(ids), 12):
                part = ids[j:j + 12]
                for k, i in enumerate(part):
                    run = g.loc[i, "run"]
                    out.append(dict(tier=t, cohort=c, setting=g.loc[i, "setting"], rep=int(g.loc[i, "rep"]), kind="CAMEX", output=f"tool_scores/CAMEX/{run}.csv",
                                    line=(f"sbatch -c 8 --mem=12G --job-name=b2t5b_camex --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/tools/t5_bundle.sbatch camex {' '.join(map(str, part))}" if k == 0 else "")))
    return out


def added_rows():
    out = []
    mem = dict(liv_hmouse="8G", test_hmouse="16G", islet="16G", kidney="30G")
    for c in list(RU.COHORTS) + ["islet"]:
        sett = [s for s in MAIN if applies(c, s)] if c != "islet" else ["full"]
        rest = [s for s in REST if applies(c, s)] if c != "islet" else []
        for t, ss in ((1, sett), (3, rest)):
            if not ss:
                continue
            for k, tool, name in ((0, "genespectra", "GeneSpectra"), (1, "cacimar", "CACIMAR")):
                step = len(ss) if k == 0 else 3
                for j in range(0, len(ss), step):
                    part = ss[j:j + step]
                    for m, s in enumerate(part):
                        res = f"-c 10 --mem={mem[c]}" if k == 0 else "-c 2 --mem=8G"
                        nm = "b2add_gs" if k == 0 else "b2add_ca"
                        out.append(dict(tier=t, cohort=c, setting=s, rep=0, kind=name, output=f"tool_scores/{name}/{RU.run_name(c, s, 0)}.csv",
                                        line=(f"sbatch {res} --job-name={nm} --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/tools/added.sbatch {tool} {c} {TBL[c]} 0 {' '.join(part)}" if m == 0 else "")))
    return out


def integ_rows():
    from analysis.tasks.B2_benchmark.integration import rules as RI

    out = []
    base = pd.read_csv(RU.TASK_DIR / "integration/baselines.tsv", sep="\t")
    for i, r in base.iterrows():
        out.append(dict(tier=1, cohort=r["name"].split(":")[0], setting=r["name"], rep=0, kind="integration", output="",
                        line=f"sbatch --array={i} -c 8 --mem=16G --job-name=b2i_base {lg('b2i_base')} analysis/tasks/T11_4_vz_integration/cmd_array.sbatch analysis/tasks/B2_benchmark/integration/baselines.tsv"))
    for co in RI.COHORTS:
        for m in ("CAMEX", "SATURN", "SATURN-Leiden"):
            out.append(dict(tier=1, cohort=co, setting=m, rep=0, kind="integration", output=f"integration/latents/{co}/{m}/z_mean.npy",
                            line=f"sbatch -c 16 --mem=40G --job-name=b2i_{m}_{co} --output=results/analysis/B2/logs/%x_%j.out analysis/tasks/B2_benchmark/integration/tools.sbatch {co} {m}"))
    return out


def table() -> pd.DataFrame:
    return pd.DataFrame(fit_rows() + tool_rows() + added_rows() + integ_rows())


def status():
    t = table()
    t["done"] = t["output"].map(lambda p: bool(p) and (RU.RES / p).exists())
    t.loc[t.output == "", "done"] = t.loc[t.output == ""].apply(lambda r: None, axis=1)
    print(t.groupby(["tier", "kind"])["done"].agg(["count", "sum"]).to_string())
    for tier in sorted(t.tier.unique()):
        s = t[(t.tier == tier) & (t.output != "")]
        print("tier", tier, "complete" if s["done"].all() else f"incomplete, {int((~s['done'].astype(bool)).sum())} of {len(s)} missing")


def queue(path):
    t = table()
    order = {"fit": 0, "CAMEX": 1, "GeneSpectra": 2, "CACIMAR": 3, "integration": 4, "SATURN": 5}
    t["o"] = t["kind"].map(order)
    t = t[t.line != ""].sort_values(["tier", "o"], kind="stable")
    single = t["kind"].isin(["fit", "SATURN"])  # a bundle stays: it skips what exists
    t = t[~(single & t.output.map(lambda p: bool(p) and (RU.RES / p).exists()))]
    pd.Series(t.line.drop_duplicates()).to_csv(path, index=False, header=False, quoting=3, sep="\t")
    print(t.groupby("tier").size().to_dict())


def freeze():
    """NOT USED: the owner withdrew the stopping time (plan amendment, 2026-11-04); every tier runs to completion and is scored whole as it completes.
    At the stopping time: write which settings and replicates enter scoring, from whole tiers only, by time alone."""
    import json
    import time

    t = table()
    t = t[t.output != ""]
    done = {k: bool(g.output.map(lambda p: (RU.RES / p).exists()).all()) for k, g in t.groupby("tier")}
    sett, reps = [], []
    if done.get(0):
        sett += MAIN
        reps = [0]
    if done.get(0) and done.get(2):
        reps = list(RU.REPS)
    if done.get(0) and done.get(3):
        sett += REST
    out = dict(time=time.strftime("%Y-%m-%dT%H:%M:%S"), tiers_complete={int(k): v for k, v in done.items()}, settings=sett, reps=reps,
               tools_note="tools enter where their tier is complete; tier 1 holds SATURN, CAMEX, GeneSpectra, CACIMAR on main settings and the integration track")
    (RU.RES / "tiers_used.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    {"queue": lambda: queue(sys.argv[2]), "freeze": freeze}.get(sys.argv[1], status)()
