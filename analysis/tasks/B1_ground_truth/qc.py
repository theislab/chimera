"""Realism: does an altered object look like real data to the same QC? No training, no method score is read.

Rows: every alteration at its strongest effect plus sham_swap and thin_both, band low and high, share 5%, seed 42, all
three cohorts; also pattern q 0.5 and level f 2 in the band midhigh. For each object:
  * the object passes check_cohort and the pretraining checks E1 of train_and_diagnose;
  * library size and genes-per-cell distributions per species against the unaltered object: KS statistic and ratio of medians;
  * gene level, in the altered species: mean-variance and mean-dropout residuals of altered genes against a spline fit on
    untouched genes of the unaltered object, compared with their matched controls (Cohen's d, KS);
  * a 5-fold classifier (logistic on log variance, zero share, Fano, max count in the altered species) of altered genes vs
    untouched genes of the nearest post-alteration mean: AUROC near 0.5 means an altered gene looks like a real gene of its
    new expression. Mean itself is not a feature because level and loss change it by design, and per-type statistics are not
    features because pattern and loss are meant to move them.
Figure: raw per-type counts before and after for three genes of each alteration (panel in figures.py).
Writes results/analysis/B1/qc/{realism.csv, example_counts.parquet}.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from analysis.tasks.B1_ground_truth import alter as A
from analysis.tasks.B1_ground_truth import rules as RU

QC_ROWS = ([("pattern", 1.0), ("level", 8), ("loss", 1.0), ("sham_swap", 1.0), ("thin_both", 4)], ["low", "high"], 0.05)
QC_EXTRA = [("pattern", 0.5, "midhigh"), ("level", 2, "midhigh")]


def gene_stats(X) -> pd.DataFrame:
    X = X.astype(np.float64)
    m = np.asarray(X.mean(0)).ravel()
    X2 = X.copy()
    X2.data = X2.data ** 2
    v = np.maximum(np.asarray(X2.mean(0)).ravel() - m ** 2, 0)
    z = 1 - np.asarray((X > 0).mean(0)).ravel()
    mx = np.asarray(X.max(0).todense()).ravel()
    return pd.DataFrame(dict(log_mean=np.log1p(m), log_var=np.log1p(v), zero=z, fano=np.log1p(v / np.maximum(m, 1e-9)),
                             log_max=np.log1p(mx), log_m=np.log(np.maximum(m, 1e-9))))


def fit_curve(x, y, df=6):
    """Spline fit on untouched genes (polynomial in rank-free log mean, knots at quantiles)."""
    from numpy.polynomial import chebyshev as C

    lo, hi = x.min(), x.max()
    t = 2 * (x - lo) / (hi - lo) - 1
    return lambda q: C.chebval(2 * (q - lo) / (hi - lo) - 1, C.chebfit(t, y, df))


def run_one(row, base, g, meta, X0, examples):
    cohort, alt, eff, band, share, seed = row["cohort"], row["alt"], row["effect"], row["band"], row["share"], row["seed"]
    truth = A.truth_for(g, alt, band, share, seed)
    X1, info = A.apply_alteration(X0, truth, meta, alt, float(eff), seed, g.index)
    spc = meta["spc"]
    rec = dict(cohort=cohort, alt=alt, effect=eff, band=band, n_pairs=len(truth), share_achieved=len(truth) / len(g),
               X_min=int(X1.data.min()) if X1.data.size else 0)
    for s in RU.SPECIES:
        m = spc == s
        L0, L1 = np.asarray(X0[m].sum(1)).ravel(), np.asarray(X1[m].sum(1)).ravel()
        N0, N1 = np.asarray((X0[m] > 0).sum(1)).ravel(), np.asarray((X1[m] > 0).sum(1)).ravel()
        rec[f"lib_ks_{s}"] = ks_2samp(L0, L1).statistic
        rec[f"lib_med_ratio_{s}"] = float(np.median(L1) / np.median(L0))
        rec[f"ngene_ks_{s}"] = ks_2samp(N0, N1).statistic
        rec[f"ngene_med_ratio_{s}"] = float(np.median(N1) / np.median(N0))
    # altered-species gene-level statistics
    alt_mask = {s: truth["species_altered"].to_numpy() == s for s in RU.SPECIES}
    resid_a, resid_c = {"var": [], "zero": []}, {"var": [], "zero": []}
    feats_a, feats_c = [], []
    fcols = ["log_var", "zero", "fano", "log_max"]
    pos = pd.Series(np.arange(len(g)), index=g.index)
    for s in RU.SPECIES:
        m = spc == s
        st0 = gene_stats(X0[m])
        st1 = gene_stats(X1[m])
        touched = np.zeros(len(g), bool)
        touched[pos[truth["altered"]].to_numpy()] = True
        use = (~touched) & (st0["log_mean"].to_numpy() > 0)
        fv = fit_curve(st0["log_m"].to_numpy()[use], st0["log_var"].to_numpy()[use])
        fz = fit_curve(st0["log_m"].to_numpy()[use], st0["zero"].to_numpy()[use])
        ia = pos[truth.loc[alt_mask[s], "altered"]].to_numpy()
        ic = pos[truth.loc[alt_mask[s], "control"]].to_numpy()
        ok = st1["log_mean"].to_numpy()[ia] > 0
        ia = ia[ok]
        ic = ic[ok]
        resid_a["var"].append(st1["log_var"].to_numpy()[ia] - fv(st1["log_m"].to_numpy()[ia]))
        resid_c["var"].append(st1["log_var"].to_numpy()[ic] - fv(st1["log_m"].to_numpy()[ic]))
        resid_a["zero"].append(st1["zero"].to_numpy()[ia] - fz(st1["log_m"].to_numpy()[ia]))
        resid_c["zero"].append(st1["zero"].to_numpy()[ic] - fz(st1["log_m"].to_numpy()[ic]))
        # untouched genes of the nearest post-alteration mean, without replacement
        pool = np.flatnonzero(use)
        taken = np.zeros(len(pool), bool)
        lm = st1["log_m"].to_numpy()
        near = []
        for v in ia:
            dist = np.abs(lm[pool] - lm[v])
            dist[taken] = np.inf
            j = int(dist.argmin())
            taken[j] = True
            near.append(pool[j])
        feats_a.append(st1[fcols].to_numpy()[ia])
        feats_c.append(st1[fcols].to_numpy()[np.array(near, dtype=int)])
    for k in ("var", "zero"):
        a_, c_ = np.concatenate(resid_a[k]), np.concatenate(resid_c[k])
        sd = np.sqrt((a_.var() + c_.var()) / 2)
        rec[f"resid_{k}_d"] = float((a_.mean() - c_.mean()) / max(sd, 1e-9))
        rec[f"resid_{k}_ks"] = float(ks_2samp(a_, c_).statistic)
        rec[f"resid_{k}_abs_alt"] = float(np.median(np.abs(a_)))
        rec[f"resid_{k}_abs_ctl"] = float(np.median(np.abs(c_)))
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    Xc = np.vstack(feats_a + feats_c)
    yc = np.r_[np.ones(sum(map(len, feats_a))), np.zeros(sum(map(len, feats_c)))]
    rec["classifier_auc"] = float(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(max_iter=500)), Xc, yc,
                                                  cv=5, scoring="roc_auc").mean())
    # example genes (first three in the table order, one set per object) for the raw-count panel
    lab = meta["lab"]
    for _, r in truth.head(3).iterrows():
        j = pos[r["altered"]]
        for tag, X in (("before", X0), ("after", X1)):
            x = np.asarray(X[:, j].todense()).ravel()
            examples.append(pd.DataFrame(dict(cohort=cohort, alt=alt, effect=eff, band=band, gene=r["altered"], state=tag,
                                              species=spc, cell_type=lab, count=x.astype(int), type_top=r["type_top"],
                                              type_bottom=r["type_bottom"], species_altered=r["species_altered"])))
    return rec


def main():
    (RU.RES / "qc").mkdir(parents=True, exist_ok=True)
    recs, examples = [], []
    for c in RU.COHORTS:
        a = A.read_base(c)
        g, meta = A.gene_table(a, c)
        X0 = A.csr(a.X).astype(np.int64)
        rows = [dict(cohort=c, alt=al, effect=e, band=b, share=QC_ROWS[2], seed=42) for al, e in QC_ROWS[0] for b in QC_ROWS[1]]
        rows += [dict(cohort=c, alt=al, effect=e, band=b, share=QC_ROWS[2], seed=42) for al, e, b in QC_EXTRA]
        for r in rows:
            recs.append(run_one(r, a, g, meta, X0, examples))
            print(recs[-1], flush=True)
    pd.DataFrame(recs).to_csv(RU.RES / "qc" / "realism.csv", index=False)
    e = pd.concat(examples, ignore_index=True)
    e.to_parquet(RU.RES / "qc" / "example_counts.parquet")


if __name__ == "__main__":
    main()
