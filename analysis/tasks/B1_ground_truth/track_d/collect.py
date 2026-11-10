"""B1 Track D tables: results/analysis/B1/track_d/{runtime.csv, returns.csv} and tidy rows in B1_trackD_tidy.csv (schema of B1_tidy.csv).

All jobs on one node, 16 CPUs (job.sbatch). A Chimera job counts as finished when its run folder holds final_state.npz and summary.json:
run_experiment.py exits 1 when an expectation check fails, which is not a failed fit.

Tidy tasks (rank_direction lower):
  runtime, memory        wall hours and peak GB of one job, per cohort and object.
  pipeline_runtime       wall hours to get both a cross-species embedding and gene-level scalars on the same cohort: Chimera = one fit;
                         SATURN + direct, CAMEX + direct = the tool's fit plus the labelled direct pipeline. Gene scalars of the tools need the
                         direct pipeline beside them, since their own are our derivations.
  scaling_cells          wall hours on testis H+Mouse at 25, 50, 100% of cells (setting carries n_cells).
  scaling_species        Chimera wall hours per 1,000 cells on 2, 3, 4 species (testis H+Mouse, testis 3-sp, liver 4-sp; descriptive: recipes and
                         epochs differ by cohort and are stated).
"""

from __future__ import annotations

import json
import re

import anndata as ad
import pandas as pd

from analysis.common import cohorts as CO
from analysis.tasks.B1_ground_truth import rules as RU
from analysis.tasks.B1_ground_truth.track_d.make import HERE

OUT = RU.RES / "track_d"
NAME = {"chimera": "Chimera", "direct": "Direct (B1, B1b, B1c, B2b, Sdir, B3a)", "saturn": "SATURN", "camex": "CAMEX"}
RETURNS = [
    dict(method="Chimera", one_fit_returns="cell embedding z; species-blind shared decoder; per gene alpha, Delta-alpha, S_g; S_g by cell type and species (forward pass); decoders for imputation",
         labels_needed="none on liver; cell-type conditional MMD on testis and kidney", gene_scalar="native", extra_runs_for_both="none"),
    dict(method="Direct (B1, B1b, B1c, B2b, Sdir, B3a)", one_fit_returns="gene scalars only; no embedding, no decoder", labels_needed="yes, matched across species",
         gene_scalar="native", extra_runs_for_both="an integration tool for an embedding"),
    dict(method="SATURN", one_fit_returns="cell embedding; macrogene weights; S_deploy, S_cos are our derivations", labels_needed="yes, within species",
         gene_scalar="derived", extra_runs_for_both="the direct pipeline for gene-level scores"),
    dict(method="CAMEX", one_fit_returns="cell embedding; gene node embedding for HVG and DEG genes; C_cos is our derivation", labels_needed="human labels, unused in integration",
         gene_scalar="derived, partial coverage", extra_runs_for_both="the direct pipeline for gene-level scores"),
]


def parse_time(p):
    t = p.read_text()
    sec = float(re.search(r"wall_s: (\S+)", t).group(1))
    rss = int(re.search(r"peak_rss_kb: (\d+)", t).group(1)) / 1024 ** 2
    code = int(re.search(r"exit: (-?\d+)", t).group(1))
    return sec, rss, code


def n_cells(h5ad: str) -> int:
    a = ad.read_h5ad(RU.REPO / h5ad, backed="r")
    n = a.n_obs
    a.file.close()
    return n


def row(task, setting, method, metric, value, n, d="lower"):
    return dict(track="D", task=task, setting=setting, method=method, metric=metric, value=value, ci_lo=None, ci_hi=None, n=n, rank_direction=d)


def main():
    j = pd.read_csv(HERE / "jobs.tsv", sep="\t", keep_default_na=False)
    rows = []
    for _, r in j.iterrows():
        f = OUT / f"{r['job']}.time.txt"
        if not f.exists():
            continue
        sec, rss, code = parse_time(f)
        d = r.drop(labels=["recipe"]).to_dict()
        if r["method"] == "chimera":
            rd = CO.RUNS_DIR / f"b1trackd-{r['job']}"
            ok = (rd / "final_state.npz").exists() and (rd / "summary.json").exists()
            d["epochs"] = json.loads((rd / "config.json").read_text()).get("max_epochs") if ok else None
        else:
            ok = code == 0
        d.update(wall_s=sec, peak_rss_gb=rss, exit=code, finished=ok, n_cells=n_cells(r["h5ad"]))
        fd = OUT / f"{r['job']}.direct.json"
        if fd.exists():
            d["direct_steps"] = json.dumps(json.loads(fd.read_text()))
        rows.append(d)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "runtime.csv", index=False)
    pd.DataFrame(RETURNS).to_csv(OUT / "returns.csv", index=False)
    ok = t[t.finished]
    tidy = []
    g = ok.groupby(["method", "cohort", "object", "n_species"]).agg(w=("wall_s", "mean"), r=("peak_rss_gb", "max"), n=("wall_s", "size"), cells=("n_cells", "first")).reset_index()
    for _, r in g.iterrows():
        s = f"cohort={r['cohort']};object={r['object']};n_species={r['n_species']};n_cells={r['cells']};cpus=16"
        tidy.append(row("runtime", s, NAME[r["method"]], "wall_h", r["w"] / 3600, r["n"]))
        tidy.append(row("memory", s, NAME[r["method"]], "peak_rss_gb", r["r"], r["n"]))
        if r["cohort"] == "test_hmouse":
            tidy.append(row("scaling_cells", s, NAME[r["method"]], "wall_h", r["w"] / 3600, r["n"]))
    full = g[g.object == "full"].set_index(["cohort", "method"])["w"]
    for c in RU.COHORTS:
        s = f"cohort={c};object=full;n_species=2;cpus=16"
        if (c, "chimera") in full:
            tidy.append(row("pipeline_runtime", s, "Chimera (one fit)", "wall_h", full[(c, "chimera")] / 3600, 1))
        for tool in ("saturn", "camex"):
            if (c, tool) in full and (c, "direct") in full:
                tidy.append(row("pipeline_runtime", s, f"{NAME[tool]} + direct", "wall_h", (full[(c, tool)] + full[(c, "direct")]) / 3600, 1))
    ch = ok[(ok.method == "chimera") & (ok.object == "full") & ok.cohort.isin(["test_hmouse", "test_3sp", "liv_4sp"])]
    for _, r in ch.groupby(["cohort", "n_species"]).agg(w=("wall_s", "mean"), cells=("n_cells", "first"), ep=("epochs", "first"), n=("wall_s", "size")).reset_index().iterrows():
        s = f"cohort={r['cohort']};object=full;n_species={r['n_species']};n_cells={r['cells']};epochs={r['ep']};cpus=16"
        tidy.append(row("scaling_species", s, "Chimera", "wall_h_per_1k_cells", r["w"] / 3600 / (r["cells"] / 1000), r["n"]))
    pd.DataFrame(tidy).to_csv(RU.RES / "B1_trackD_tidy.csv", index=False)
    print(t.drop(columns=[c for c in ("direct_steps", "h5ad") if c in t]).to_string())
    print(pd.DataFrame(tidy)[["task", "setting", "method", "value"]].to_string())


if __name__ == "__main__":
    main()
