"""Metrics of one scored altered fit: results/analysis/B1/metrics/<run>.parquet, draws/<run>.npz. Reads truth and scores."""

from __future__ import annotations

import argparse
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd

from analysis.tasks.B1_ground_truth import metrics as MT
from analysis.tasks.B1_ground_truth import rules as RU


def tool_scores(run: str) -> pd.DataFrame | None:
    """SATURN and CAMEX scalars of the same altered object, if that tool ran on it."""
    frames = []
    for t in ("saturn", "camex", "genespectra", "cacimar", "samap"):
        f = RU.RES / "tool_scores" / f"{t}__{run}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f).set_index("var_name"))
    return pd.concat(frames, axis=1) if frames else None


def tool_reference(cohort: str) -> pd.DataFrame | None:
    frames = []
    for t in ("saturn", "camex", "genespectra", "cacimar", "samap"):
        fs = [RU.RES / "tool_scores" / f"{t}__b1-{cohort}-none-s{s}.parquet" for s in RU.TOOL_NONE_SEEDS]
        fs = [pd.read_parquet(f).set_index("var_name") for f in fs if f.exists()]
        if fs:
            frames.append(sum(fs) / len(fs))
    return pd.concat(frames, axis=1) if frames else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default=str(RU.TASK_DIR / "grid_all.tsv"))
    ap.add_argument("--index", type=int, default=None)
    ap.add_argument("--run", default=None)
    a = ap.parse_args()
    g = pd.read_csv(a.grid, sep="\t", keep_default_na=False)
    arms_tsv = RU.TASK_DIR / "grid_arms.tsv"
    if arms_tsv.exists() and a.index is None:
        g = pd.concat([g, pd.read_csv(arms_tsv, sep="\t", keep_default_na=False)], ignore_index=True)
        g["arm"] = g["arm"].fillna("")
    g = g[(g.alt != "none") & (g.duplicate_of == "")].reset_index(drop=True)
    row = (g.iloc[a.index] if a.index is not None else g[g.run == a.run].iloc[0]).to_dict()
    run = row["run"]
    if not (RU.RES / "scores" / f"{run}.parquet").exists():
        print("no scores yet", run)
        return
    truth = pd.read_parquet(RU.RES / "truth" / f"{run}.parquet")
    scores = pd.read_parquet(RU.RES / "scores" / f"{run}.parquet")
    arm = row.get("arm") or None
    if isinstance(arm, float):
        arm = None
    ref = MT.reference(row["cohort"], arm)
    seq = MT.sequence_table()
    gt = pd.read_parquet(RU.RES / "base" / f"{row['cohort']}_genes.parquet")
    band_mask = (gt["band"].reindex(scores["var_name"]).to_numpy() == row["band"])
    extra, methods = tool_scores(run), dict(MT.METHODS) if arm is None else MT.arm_methods(arm)
    if extra is not None:
        tr = tool_reference(row["cohort"])
        if tr is not None:
            for c in tr.columns:
                ref[c] = tr[c].reindex(ref.index)
        methods.update({k: v for k, v in MT.TOOLS.items() if k in extra.columns})
    lng = MT.score_run(row, truth, scores, ref, seq, band_mask, methods=methods, extra=extra)
    for k in ("cohort", "alt", "effect", "band", "share", "seed", "run"):
        lng[k] = row[k]
    (RU.RES / "metrics").mkdir(parents=True, exist_ok=True)
    (RU.RES / "draws").mkdir(parents=True, exist_ok=True)
    lng.to_parquet(RU.RES / "metrics" / f"{run}.parquet")
    dr = MT.bootstrap_all(row, truth, scores, ref, seq, methods=methods, extra=extra)
    np.savez_compressed(RU.RES / "draws" / f"{run}.npz", **dr)
    print("done", run, flush=True)


if __name__ == "__main__":
    main()
