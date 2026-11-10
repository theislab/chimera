"""I1: does the free per-gene gate (Config M) give a better cell embedding than the canonical protein-feature gate?

Fixed before any free-gate integration value was computed by this task (see ledger I1_free_gate_integration.md for what had been seen).

Metrics, code and keys are the paper's, imported unmodified:
  * Fig 3 / T5.5: ``compute_integration_metrics`` (training/compare_vs_scvi.py), k 15, seed 42, ``species`` and ``cell_type`` of the seven
    T5.5 objects; direction correction, min-max and 0.4/0.6 composite from ``T5_5_benchmark/score_external.score_table``. A seed of either
    gate is placed on the frozen 10-model panel's per-cohort bounds (``master_table_ext.csv``), so no other method's value moves; the
    replace-the-Chimera-column composite is a sensitivity.
  * Harmonised cohorts / H1: the same function on H1's object and cell-type key (``H1_harmonise.tables.integration_row``).
  * Z2 within-label: ``Z2_embedding.common.all_rows`` on Z2's cached targets (zone, PT segment, beta stress, marrow maturation, testis
    stages; within-label variance and library R2), Z2's encoder sets (TS and TMS hepatocytes through liver H+Mouse models, macaque testis
    through testis H+Mouse models) via ``Z2_embedding.encode``.
Lines: Z3 cohorts as Z3 fixed them (``results/analysis/Z3/lines.json``): free5 against can5, liver also free_int against can_int.
Other cohorts: canonical = H1 kept runs; free = Z4 fits with gate health (Z3 amendment 1) passing, mixing NOT used for inclusion (it is the
outcome here); Z4's own healthy flag (gate + mixing) is a labelled sensitivity, applied to both sides.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.common import gene_ids

REPO = gene_ids.REPO_ROOT
RES = REPO / "results" / "analysis"
OUT = RES / "I1"
RUNS = REPO / "training" / "runs"
Z3_COHORTS = ["liv_hmouse", "test_hmouse", "liv_4sp"]
# cohort -> T5.5 dataset key (Fig 3 rows)
FIG3 = {"liv_hmonkey": "panc_hmonkey", "liv_hmouse": "panc_hmouse", "liv_hzeb": "panc_hzeb", "liv_4sp": "panc_4sp",
        "test_hmac": "test_hmac", "test_hmouse": "test_hmouse", "test_3sp": "test_3sp"}
FIG3_METRICS = ["knn_species_purity", "knn_mixing_entropy", "sil_species", "knn_celltype_purity", "sil_celltype", "label_transfer_acc"]
# Primary metrics, in the verdict. Higher is better for all of them.
PRIMARY_FIG3 = ["mixing", "biology", "total_score"]          # Fig 3 rows, frozen-panel bounds
PRIMARY_H1 = ["knn_mixing_entropy", "label_transfer_acc"]   # every cohort, H1 key
PRIMARY_Z2 = ["within_auroc"]                                # Z2 cohorts: mean in-model AUROC (logistic) over the cohort's within-label tasks
LOWER_BETTER = {"knn_species_purity", "sil_species", "lib_r2", "pc1_share"}
CI_LEVEL = 0.95


def _z3_lines() -> dict:
    return json.loads((RES / "Z3" / "lines.json").read_text())


def lines() -> pd.DataFrame:
    """One row per (cohort, gate, seed, run) with inclusion flags. gate in {canonical, free}; line names the comparison set."""
    rows = []
    L = _z3_lines()
    for c in Z3_COHORTS:
        for line, gate in (("free5", "free"), ("can5", "canonical")):
            for s, r in L[line][c].items():
                rows.append(dict(cohort=c, gate=gate, seed=int(s), run=r, line="main", source="Z3", included=True, included_mix=True))
        if c == "liv_hmouse":
            for line, gate in (("free_int", "free"), ("can_int", "canonical")):
                for s, r in L[line][c].items():
                    rows.append(dict(cohort=c, gate=gate, seed=int(s), run=r, line="integrated_only", source="Z3", included=True, included_mix=True))
    h1 = pd.read_csv(RES / "H1" / "runs.tsv", sep="\t")
    h1 = h1[h1.kept.astype(bool) & ~h1.cohort.isin(Z3_COHORTS)]
    for r in h1.itertuples():
        rows.append(dict(cohort=r.cohort, gate="canonical", seed=int(r.seed), run=r.run, line="main", source="H1", included=True, included_mix=True))
    zh = RES / "Z4" / "health.csv"
    if zh.exists():
        z = pd.read_csv(zh)
        z = z[~z.cohort.isin(Z3_COHORTS) & z.finished.astype(bool)]
        for r in z.itertuples():
            gate_ok = bool(r.healthy_gate) and bool(r.no_nan)
            rows.append(dict(cohort=r.cohort, gate="free", seed=int(r.seed), run=r.run, line="main", source="Z4", included=gate_ok,
                             included_mix=gate_ok and bool(r.healthy)))
    from analysis.tasks.I1_free_gate_integration import replication  # amendment 1: liver H+Monkey replication line
    rows += replication.lines()
    return pd.DataFrame(rows)


def mixing_rule_canonical(cohort: str, t: pd.DataFrame) -> pd.Series:
    """Z4's integration criterion applied to canonical seeds (sensitivity): entropy >= half the median of the cohort's kept canonical
    seeds and adversary <= canonical max + 0.10 hold for every canonical seed by construction of the max; only the entropy half applies."""
    e = t.knn_mixing_entropy
    return e >= 0.5 * e.median()


def tcrit(df: int) -> float:
    from scipy.stats import t
    return float(t.ppf(0.5 + CI_LEVEL / 2, df))


def compare(free: np.ndarray, can: np.ndarray, paired: bool, higher_better: bool = True) -> dict:
    """Mean difference free - canonical with a t interval over seeds (paired on seed number, else Welch), and the seed-range call."""
    free, can = np.asarray(free, float), np.asarray(can, float)
    free, can = free[np.isfinite(free)], can[np.isfinite(can)]
    out = dict(n_free=len(free), n_can=len(can), mean_free=free.mean() if len(free) else np.nan, mean_can=can.mean() if len(can) else np.nan,
               min_free=free.min() if len(free) else np.nan, max_free=free.max() if len(free) else np.nan,
               min_can=can.min() if len(can) else np.nan, max_can=can.max() if len(can) else np.nan, paired=paired)
    out["delta"] = out["mean_free"] - out["mean_can"]
    lo = hi = np.nan
    if paired and len(free) == len(can) and len(free) >= 2:
        d = free - can
        h = tcrit(len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        lo, hi = d.mean() - h, d.mean() + h
    elif len(free) >= 2 and len(can) >= 2:
        v1, v2 = free.var(ddof=1) / len(free), can.var(ddof=1) / len(can)
        se = np.sqrt(v1 + v2)
        dfw = (v1 + v2) ** 2 / (v1 ** 2 / (len(free) - 1) + v2 ** 2 / (len(can) - 1)) if se > 0 else 1
        h = tcrit(max(dfw, 1)) * se
        lo, hi = out["delta"] - h, out["delta"] + h
    out["ci_lo"], out["ci_hi"] = lo, hi
    if len(free) and len(can):
        if higher_better:
            better, worse = out["min_free"] > out["max_can"], out["max_free"] < out["min_can"]
        else:
            better, worse = out["max_free"] < out["min_can"], out["min_free"] > out["max_can"]
        out["call"] = "better" if better else ("worse" if worse else "within seed range")
    else:
        out["call"] = "n/a"
    return out


def verdict(calls: pd.DataFrame, metrics: list[str]) -> dict:
    """calls: one row per (cohort, metric) with 'call'. Owner's hypothesis (free integrates better):
    supported if, on at least one primary metric, cohorts better beyond seed range exceed cohorts worse by at least a third of the cohorts
    compared (at least 1), and on no primary metric worse exceeds better; refuted on the mirror; mixed if one metric each way;
    otherwise no consistent difference."""
    per = {}
    up = down = False
    for m in metrics:
        q = calls[calls.metric == m]
        n = int((q.call != "n/a").sum())
        b, w = int((q.call == "better").sum()), int((q.call == "worse").sum())
        bar = max(1, int(np.ceil(n / 3)))
        per[m] = dict(n=n, better=b, worse=w, within=n - b - w, bar=bar)
        up |= (b - w) >= bar
        down |= (w - b) >= bar
    net_bad = any(v["worse"] > v["better"] for v in per.values())
    net_good = any(v["better"] > v["worse"] for v in per.values())
    if up and not net_bad:
        v = "supported"
    elif down and not net_good:
        v = "refuted (free integrates worse)"
    elif up and down:
        v = "mixed"
    else:
        v = "no consistent difference"
    return dict(verdict=v, per_metric=per)
