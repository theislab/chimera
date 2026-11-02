"""Y6 use 1: genome annotation check, rules.py USE 1. Odd autosomes tune, even autosomes score.

    PYTHONPATH=$PWD python -m revision.tasks.Y6_uses.use1 [--threads 12]
"""

from __future__ import annotations

import argparse
import zlib

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from revision.tasks.Y6_uses import rules as RU
from revision.tasks.Y6_uses.build_rows import forward_files, species_margin

OUT = RU.OUT / "use1"
LABELS = RU.LABELS_PRIMARY + RU.LABELS_SECONDARY


def load() -> pd.DataFrame:
    r = pd.read_parquet(RU.OUT / "rows.parquet")
    L = pd.read_parquet(RU.OUT / "annot/labels.parquet")
    d = r.merge(L, on=["species", "hsapiens_gene_id", "gene_id"], how="left")
    d = d[d.biotype_110 == "protein_coding"].copy()
    d["S_row"] = -d.S_gs
    d["S_gene"] = -d.S_ema
    d["D_neg"] = -d.D
    d["zero_s"] = 1 - d.det_s
    d["zero_gap"] = d.det_other - d.det_s
    d["B1_neg"] = (-d.E_hs).fillna(1.01)
    d["id_neg"] = -d.identity_s
    for c in LABELS:
        d[c] = pd.to_numeric(d[c], errors="coerce").astype(float)
    return d


def ap_setup(score: np.ndarray):
    order = np.argsort(-score, kind="mergesort")
    s = score[order]
    ends = np.r_[np.flatnonzero(s[1:] != s[:-1]), len(s) - 1]
    return order, ends


def wap(y, w, setup):
    order, ends = setup
    wy = (w * y)[order]
    tp = np.cumsum(wy)[ends]
    n = np.cumsum(w[order])[ends]
    if tp[-1] <= 0:
        return np.nan
    dtp = np.diff(np.r_[0.0, tp])
    return float((dtp * (tp / n)).sum() / tp[-1])


def wprf(y, w, flag):
    fl = (w * flag).sum()
    tp = (w * flag * y).sum()
    pos = (w * y).sum()
    if fl == 0 or pos == 0:
        return np.nan, np.nan, np.nan
    p, r = tp / fl, tp / pos
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def tune(score, y):
    best, bv = -1.0, np.inf
    for q in RU.GRID_Q:
        v = np.quantile(score, 1 - q)
        fl = score >= v
        tp = (fl & (y > 0)).sum()
        if tp == 0:
            continue
        f1 = 2 * tp / (fl.sum() + (y > 0).sum())
        if f1 > best:
            best, bv = f1, v
    return bv, best


def block_weights(blocks: np.ndarray, rng, draws: int):
    ub, inv = np.unique(blocks, return_inverse=True)
    for _ in range(draws):
        c = rng.multinomial(len(ub), np.full(len(ub), 1 / len(ub)))
        yield c[inv].astype(float)


def one(fit, lab, label, dd, universe):
    if universe == "expr":
        u = dd[dd.det_other >= RU.DET_EXPR]
    elif universe == "hi":
        u = dd[(dd.det_other >= RU.DET_EXPR) & (dd.identity_s >= RU.ID_HIGH)]
    else:
        u = dd
    u = u[u[label].notna()]
    odd, even = u[u.split == "odd"], u[u.split == "even"]
    npos = int(even[label].sum())
    if npos < RU.MIN_POS or int(odd[label].sum()) < RU.MIN_POS:
        return [dict(fit=fit, species_label=lab, label=label, universe=universe, n=len(even), n_pos=npos, score="", note="under MIN_POS")]
    y = even[label].to_numpy(float)
    base = y.mean()
    out = []
    setups, flags, budget = {}, {}, {}
    n = len(even)
    for s in RU.SCORES_U1:
        sc = even[s].to_numpy(float)
        sc = np.where(np.isfinite(sc), sc, np.nanmin(sc) - 1)
        setups[s] = ap_setup(sc)
        v, f1odd = tune(odd[s].fillna(odd[s].min() - 1).to_numpy(float), odd[label].to_numpy(float))
        flags[s] = sc >= v
        budget[s] = np.zeros(n, bool)
        budget[s][np.argsort(-sc, kind="mergesort")[: max(1, int(round(RU.BUDGET_Q * n)))]] = True
        p, r, f = wprf(y, np.ones(n), flags[s])
        pb, rb, _ = wprf(y, np.ones(n), budget[s])
        out.append(dict(fit=fit, species_label=lab, species=even.species.iloc[0], label=label, universe=universe, score=s, n=n, n_pos=npos, base_rate=base,
                        threshold=v, f1_odd=f1odd, n_flag=int(flags[s].sum()), precision=p, recall=r, f1=f, lift=(p / base if p == p else np.nan),
                        p_budget=pb, r_budget=rb, lift_budget=pb / base, ap=wap(y, np.ones(n), setups[s]), ap_lift=wap(y, np.ones(n), setups[s]) / base,
                        auroc=roc_auc_score(y, sc)))
    point = pd.DataFrame(out).set_index("score")
    rng = np.random.default_rng(RU.SEED + zlib.crc32(f"{fit}|{lab}|{label}|{universe}".encode()) % 100000)
    draws = {s: {k: [] for k in ("ap", "ap_lift", "precision", "recall", "f1", "lift", "lift_budget")} for s in RU.SCORES_U1}
    for w in block_weights(even.block.to_numpy(), rng, RU.DRAWS_PR):
        b = (w * y).sum() / w.sum()
        if b <= 0:
            continue
        for s in RU.SCORES_U1:
            a = wap(y, w, setups[s])
            p, r, f = wprf(y, w, flags[s])
            pb, _, _ = wprf(y, w, budget[s])
            for k, v in (("ap", a), ("ap_lift", a / b), ("precision", p), ("recall", r), ("f1", f), ("lift", p / b), ("lift_budget", pb / b)):
                draws[s][k].append(v)
    arr = {s: {k: np.asarray(v, float) for k, v in dd_.items()} for s, dd_ in draws.items()}
    res = []
    bases = [b for b in RU.BASELINES_U1]
    best_base = point.loc[bases, "ap"].idxmax()
    for r in out:
        s = r["score"]
        for k in draws[s]:
            r[f"{k}_lo"], r[f"{k}_hi"] = np.nanpercentile(arr[s][k], [2.5, 97.5])
        r["best_baseline"] = best_base
        dlt = arr[s]["ap"] - arr[best_base]["ap"]
        r["d_ap_best"] = r["ap"] - point.loc[best_base, "ap"]
        r["d_ap_best_lo"], r["d_ap_best_hi"] = np.nanpercentile(dlt, [2.5, 97.5])
        r["p_d_ap_best"] = 2 * min(np.nanmean(dlt <= 0), np.nanmean(dlt >= 0))
        dz = arr[s]["ap"] - arr["zero_gap"]["ap"]
        r["d_ap_zero_gap"] = r["ap"] - point.loc["zero_gap", "ap"]
        r["d_ap_zero_gap_lo"], r["d_ap_zero_gap_hi"] = np.nanpercentile(dz, [2.5, 97.5])
        res.append(r)
    return res


def logit_beyond(fit, lab, label, dd):
    u = dd[(dd.det_other >= RU.DET_EXPR) & dd[label].notna() & (dd.split == "even")]
    y = u[label].to_numpy(int)
    if y.sum() < RU.MIN_POS:
        return None
    cols = ["S_row", "zero_gap", "id_neg", "B1_neg", "det_other"]
    X = np.column_stack([rankdata(u[c].fillna(u[c].max()).to_numpy()) / len(u) for c in cols])
    X = (X - X.mean(0)) / X.std(0)

    def fit_(Xw, yw, w):
        return LogisticRegression(C=1e4, max_iter=500).fit(Xw, yw, sample_weight=w).coef_[0]

    est = fit_(X, y, np.ones(len(y)))
    rng = np.random.default_rng(RU.SEED + zlib.crc32(f"{fit}|{lab}|{label}".encode()) % 100000)
    B = [fit_(X, y, w) for w in block_weights(u.block.to_numpy(), rng, 200) if (w * y).sum() > 0]
    B = np.array(B)
    return [dict(fit=fit, species_label=lab, label=label, n=len(u), n_pos=int(y.sum()), term=c, coef=est[i], lo=np.percentile(B[:, i], 2.5), hi=np.percentile(B[:, i], 97.5)) for i, c in enumerate(cols)]


def f_abs(d):
    out = []
    for (fit, lab), dd in d.groupby(["fit", "species_label"]):
        hi = dd[(dd.det_other >= RU.DET_EXPR) & (dd.identity_s >= RU.ID_HIGH)]
        fl = hi[hi.det_s < RU.DET_ABSENT]
        for sp in ("even", "odd"):
            h, f = hi[hi.split == sp], fl[fl.split == sp]
            for label in LABELS:
                if h[label].notna().sum() == 0:
                    continue
                b = h[label].mean()
                nf = int(f[label].notna().sum())
                out.append(dict(fit=fit, species_label=lab, split=sp, label=label, n_universe=int(h[label].notna().sum()), base_rate=b, n_flag=nf, n_pos_flag=int(f[label].sum()),
                                precision=(f[label].mean() if nf else np.nan), lift=(f[label].mean() / b if nf and b > 0 else np.nan)))
    return pd.DataFrame(out)


def seed_range(d):
    out = []
    for fit in RU.JOINT:
        spec = RU.FITS[fit]
        files = forward_files(fit, spec)
        for f in files:
            m, _, _ = species_margin([f])
            for lab in spec["species"]:
                dd = d[(d.fit == fit) & (d.species_label == lab)]
                s = -m[lab].reindex(dd.var_name).to_numpy()
                u = (dd.det_other >= RU.DET_EXPR).to_numpy() & (dd.split == "even").to_numpy()
                for label in RU.LABELS_PRIMARY + ["T_no3utr"]:
                    y = dd[label].to_numpy()
                    k = u & np.isfinite(y)
                    if np.nansum(y[k]) < RU.MIN_POS:
                        continue
                    a = wap(y[k], np.ones(k.sum()), ap_setup(s[k]))
                    out.append(dict(fit=fit, species_label=lab, label=label, file=str(f).split("/")[-1], ap=a, ap_lift=a / y[k].mean()))
    return pd.DataFrame(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, default=12)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    groups = [(fit, lab, g) for (fit, lab), g in d.groupby(["fit", "species_label"])]
    jobs = [(fit, lab, label, g, "expr") for fit, lab, g in groups for label in LABELS]
    jobs += [(fit, lab, label, g, u) for fit, lab, g in groups for label in RU.LABELS_PRIMARY + ["T_no3utr"] for u in ("hi", "all") if fit in RU.JOINT]
    res = Parallel(n_jobs=a.threads)(delayed(one)(*j) for j in jobs)
    m = pd.DataFrame([r for x in res for r in x])
    m.to_csv(OUT / "metrics.csv", index=False)
    joint = [(f, l, lb, g) for f, l, g in groups if f in RU.JOINT for lb in RU.LABELS_PRIMARY + ["T_no3utr"]]
    lg = Parallel(n_jobs=a.threads)(delayed(logit_beyond)(*j) for j in joint)
    pd.DataFrame([r for x in lg if x for r in x]).to_csv(OUT / "logit_beyond.csv", index=False)
    f_abs(d).to_csv(OUT / "f_abs.csv", index=False)
    seed_range(d).to_csv(OUT / "seed_range.csv", index=False)
    d.to_parquet(OUT / "scored_rows.parquet")
    print(m.groupby("note", dropna=False).size() if "note" in m else "", flush=True)


if __name__ == "__main__":
    main()
