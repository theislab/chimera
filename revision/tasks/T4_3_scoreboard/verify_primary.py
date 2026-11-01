"""Independent check of the H1-H4 point estimates, and the covariate decomposition of rho_p.

Rebuilds each universe from the CSV inputs by an ENSG join and residualises with a different
spline implementation (scikit-learn cubic B-splines, 5 knots at quantiles), without importing
``stats.py`` or ``controls.py``. A resampled-with-replacement gene bootstrap (500 draws, by row
copying) checks the width of the gene-level CI. The decomposition adds C's covariates one at a
time; it is exploratory and enters no status, row or modifier.

Usage: python -m revision.tasks.T4_3_scoreboard.verify_primary [results/revision/T4_3]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import SplineTransformer

HYP = {"H1": ("liv_hmouse", "G_H1", []), "H2": ("test_hmouse", "G_H2", []),
       "H3": ("liv_hmouse", "G_H3", ["tau_shared"]), "H4": ("test_hmouse", "G_H4", ["tau_shared", "H4_nsp"])}
C4 = ["prior_raw", "log_cp10k_spmean", "detection_spmean", "log_cds_length"]


def basis(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    parts = [np.ones((len(df), 1))]
    for c in cols:
        r = rankdata(df[c].to_numpy(float)).reshape(-1, 1)
        parts.append(SplineTransformer(n_knots=5, degree=3, knots="quantile", include_bias=False).fit_transform(r))
    return np.hstack(parts)


def rho(a: np.ndarray, b: np.ndarray, Z: np.ndarray) -> float:
    ea = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    eb = b - Z @ np.linalg.lstsq(Z, b, rcond=None)[0]
    return float(np.corrcoef(ea, eb)[0, 1])


def r2(a: np.ndarray, Z: np.ndarray) -> float:
    e = a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return float(1 - e.var() / a.var())


def main(out: Path):
    res = out.parents[0]
    gold = pd.read_csv(out / "golds/gold_table.csv", index_col=0)
    sb = pd.read_csv(out / "scoreboard.csv")
    rows, dec = [], []
    rng = np.random.default_rng(5)
    for h, (c, g, extra) in HYP.items():
        d = pd.read_csv(res / f"T3_3/{c}__consensus.csv").set_index("hsapiens_gene_id")
        b = pd.read_csv(res / f"T4_2/baselines__{c}.csv").set_index("hsapiens_gene_id")
        cds = pd.read_csv(res / "T2_4/cds_features.csv").set_index("hsapiens_gene_id")
        cache = pd.read_parquet(out / f"cache/{c}__inputs.parquet").set_index("hsapiens_gene_id")
        t = pd.DataFrame({"D": d["D"], "prior_raw": d["prior_raw"]})
        t = t.join(b[["log_cp10k_spmean", "detection_spmean"]]).join(cds["log_cds_length"]).join(cache["tau_shared"])
        t = t.join(gold[[g] + (["H4_nsp"] if "H4_nsp" in extra else [])])
        cols = C4 + extra
        t = t.dropna(subset=["D", g] + cols)
        rD, rG = rankdata(t["D"]), rankdata(t[g])
        Z = basis(t, cols)
        val = rho(rD, rG, Z)
        boots = []
        for _ in range(500):
            i = rng.integers(0, len(t), len(t))
            boots.append(rho(rD[i], rG[i], Z[i]))
        ref = sb[(sb.family == "F1") & (sb.hyp == h)].iloc[0]
        rows.append(dict(hyp=h, n=len(t), n_scoreboard=int(ref.n), rho_p_independent=val, rho_p_scoreboard=ref.rho_p,
                         diff=val - ref.rho_p, gene_ci_lo=np.quantile(boots, .025), gene_ci_hi=np.quantile(boots, .975),
                         scoreboard_gene_lo=ref.ci_gene_lo, scoreboard_gene_hi=ref.ci_gene_hi))
        for label, cs in (("none", []), ("pi", C4[:1]), ("expr", C4[1:2]), ("detection", C4[2:3]), ("length", C4[3:]),
                          ("pi+expr", C4[:2]), ("pi+expr+det", C4[:3]), ("C", C4), ("C_full", cols)):
            Zc = basis(t, cs)
            dec.append(dict(hyp=h, covariates=label, rho=rho(rD, rG, Zc), r2_rank_D=r2(rD, Zc), r2_rank_G=r2(rG, Zc)))
    v = pd.DataFrame(rows)
    q = pd.DataFrame(dec)
    v.to_csv(out / "verify_primary.csv.tmp", index=False)
    (out / "verify_primary.csv.tmp").replace(out / "verify_primary.csv")
    q.to_csv(out / "rho_decomposition_exploratory.csv.tmp", index=False)
    (out / "rho_decomposition_exploratory.csv.tmp").replace(out / "rho_decomposition_exploratory.csv")
    pd.set_option("display.width", 200)
    print(v.round(4).to_string(index=False))
    print(q.pivot(index="covariates", columns="hyp", values="rho").reindex(
        ["none", "pi", "expr", "detection", "length", "pi+expr", "pi+expr+det", "C", "C_full"]).round(3).to_string())
    print(q[q.covariates.isin(["C", "C_full"])].round(3).to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/revision/T4_3"))
