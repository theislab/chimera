"""X1 (EXPLORATORY): T2.5 margin decomposition on extra testis seeds (checkpoint forward pass only).

Uses the T2.5 forward pass and decomposition code unchanged (``T2_5_celltype_margin/run.py``,
``margin.py``) on finished seed runs (``summary.json`` present). Only the fine
``cell_type`` key is decomposed (the coarse stages are summed from it in ``build.py``).
Writes ``results/revision/X1/forward/<cohort>_s<seed>__{S_gsk,genes}.parquet``; nothing is
written to run dirs.

Usage: python revision/tasks/X1_testis_discovery/seeds.py --runs rev-test_hmouse-s43 ... [--threads 8]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from revision.common import cohorts as C
from revision.common import gene_ids

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "T2_5_celltype_margin"))
import margin as MG  # noqa: E402
import run as T25  # noqa: E402

OUT = gene_ids.REPO_ROOT / "results" / "revision" / "X1" / "forward"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    for run in args.runs:
        if not (C.RUNS_DIR / run / "summary.json").exists():
            print(f"skip {run}: no summary.json", flush=True)
            continue
        cohort = run.removeprefix("rev-").rsplit("-s", 1)[0]
        label = f"{cohort}_s{run.rsplit('-s', 1)[1]}"
        if (OUT / f"{label}__genes.parquet").exists():
            print(f"skip {label}: done", flush=True)
            continue
        fw = T25.forward_sums(label, cohort, run, args.threads, args.batch_size)
        co = C.load_cohort(cohort, run=run)
        genes = co.genes.reset_index()
        if list(genes["var_name"]) != [str(v) for v in fw["var_names"]]:
            raise AssertionError(f"{label}: gene order differs")
        dec = MG.decompose(fw, "cell_type")
        t = MG.gene_table(fw, dec, genes)
        S_long, _ = MG.long_tables(dec, t, "cell_type", None)
        acc = MG.acceptance(dec, t, cohort, "cell_type")
        for r in acc:
            print(f"  {label}: {r['check']} = {r['value']:.4g}", flush=True)
        for name, df in (("S_gsk", S_long), ("genes", t)):
            tmp = OUT / f"{label}__{name}.parquet.tmp"
            df.to_parquet(tmp, index=False)
            tmp.replace(OUT / f"{label}__{name}.parquet")


if __name__ == "__main__":
    main()
