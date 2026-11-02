"""D1 statistics: rho_p through X4's rho_unit (T4.3 stats), one universe per call, tidy rows."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np
import pandas as pd

from revision.tasks.B2_benchmark import rules_c as RC
from revision.tasks.D1_sparse_genes import rules as RU
from revision.tasks.T4_3_scoreboard import stats as ST
from revision.tasks.X4_where_differs import common as X4

MIN_N = RU.STRATUM_MIN


def golds(c: str) -> list[tuple[str, str, list[str]]]:
    return list(RC.COHORTS_C[c]["golds"])


def load(c: str) -> pd.DataFrame:
    return pd.read_parquet(RU.RES / "genes" / f"{c}.parquet")


def split_rows(t: pd.DataFrame, split: str) -> np.ndarray:
    return np.ones(len(t), bool) if split == "all" else (t["split"] == split).to_numpy()


def partial(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> float:
    ra = ST.resid(ST.midrank(a), Z)
    rb = ST.resid(ST.midrank(b), Z)
    return float(np.corrcoef(ra, rb)[0, 1])


def rho_rows(t: pd.DataFrame, gold: str, covs: list[str], scores: list[str], rows: np.ndarray, ref: str | None, draws: int, label: dict,
             extra_spline=None, extra_linear=None) -> list[dict]:
    use = [s for s in scores if s in t and np.isfinite(t.loc[rows, s].to_numpy(float)).mean() > 0]
    if ref is not None and ref not in use:
        return []
    return X4.rho_unit(t, gold, covs, use, rows=rows, label=label, draws=draws, ref=ref, extra_spline=extra_spline, extra_linear=extra_linear)
