"""B1 scoring of one altered fit against its truth table. Fixed before any altered run was scored.

Each score is oriented so higher = more conserved; d = -score is the divergence call. The expected direction of every
alteration on every conservation score is "altered below its matched control" (PREDICTIONS.md).

For one run, per method and per gene subset:
  AUROC        altered vs matched controls on d; ties count one half; a gene the method does not score ties with
               everything (AUROC_cov is the value on scored genes only; coverage is the scored share of altered genes).
  AUPRC_match  average precision, altered vs matched controls (prevalence 0.5).
  AUPRC_genome average precision, altered vs every other gene in the object (prevalence = share); unscored genes rank at the median.
  topk_genome  precision among the k = n_altered highest d, all genes of the object (unscored at the median).
  topk_band    the same within the run's expression band.
  power_05     share of altered genes above the 95th percentile of d among their matched controls (a method's own null).
  dir_acc      signed direction: share of pairs with (s_alt - r_alt) < (s_ctl - r_ctl); s in this fit, r the mean over the
               cohort's unaltered fits; ties one half.
  dir_raw      share of altered genes with s_alt < r_alt.
  effect_sd    mean over pairs of (s_alt - r_alt) - (s_ctl - r_ctl), in SD of r over the band's genes.
  wilcoxon_p, reject05   one-sided signed-rank on d_alt - d_ctl over scored pairs (zeros split); type I error on nulls.
  frac_zero    share of altered genes with |score| < 1e-6 (mask; meaningful for S_g).
Subsets: all; identity band; ref_unmasked and ref_masked (both members of the pair count-supported, or both masked, by
the mean over the unaltered fits, so outcome-independent: a gene's own thinning cannot move it between subsets).
Bootstrap: 200 resamples of pairs per run, all methods on the same draws, subset all only.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from revision.tasks.B1_ground_truth import rules as RU

# method -> (column, sign, group)
METHODS = {
    "S_g": ("S_g", 1, "chimera"), "Delta_alpha": ("D", 1, "chimera"), "Delta_alpha_raw": ("D_raw", 1, "chimera"),
    "B1z": ("B1z", 1, "chimera"), "B1z_raw": ("B1z_raw", 1, "chimera"), "S_fwd": ("S_fwd", 1, "chimera_exploratory"),
    "B1": ("B1", 1, "direct"), "B1b": ("B1b", 1, "direct"), "B1c": ("B1c", 1, "direct"), "B1c_raw": ("B1c_raw", 1, "direct"),
    "B2b": ("B2b", 1, "direct"), "B3a": ("B3a", 1, "direct"), "Sdir": ("Sdir", 1, "direct"), "Lfc": ("Lfc", 1, "direct"),
    "pi": ("prior_raw", 1, "sequence"), "dNdS": ("dNdS", -1, "sequence"), "LOEUF": ("LOEUF", -1, "sequence"),
    "expr": ("expr", 1, "control"), "random": ("random", 1, "control"),
}
TOOLS = {"S_deploy": ("S_deploy", 1, "tool"), "S_cos": ("S_cos", 1, "tool"), "C_cos": ("C_cos", 1, "tool"),
         "GeneSpectra": ("GeneSpectra", 1, "tool"), "CACIMAR": ("CACIMAR", 1, "tool"), "SAMap": ("SAMap", 1, "tool")}
BOOT = 200
MASK_EPS = 1e-6
BOOT_METRICS = ["AUROC", "AUPRC_match", "power_05", "dir_acc"]


def auroc_vec(a: np.ndarray, c: np.ndarray) -> float:
    if len(a) == 0 or len(c) == 0:
        return np.nan
    c = np.sort(c)
    return float(((np.searchsorted(c, a, "left") + np.searchsorted(c, a, "right")) / 2).mean() / len(c))


def auroc_miss(a: np.ndarray, c: np.ndarray) -> tuple[float, float]:
    """(AUROC with unscored genes tying, AUROC on scored genes)."""
    fa, fc = np.isfinite(a), np.isfinite(c)
    v = auroc_vec(a[fa], c[fc])
    if not np.isfinite(v):
        return 0.5, np.nan
    w = fa.mean() * fc.mean()
    return float(w * v + (1 - w) * 0.5), v


def avg_prec(y: np.ndarray, s: np.ndarray) -> float:
    """Average precision with tied scores grouped; unscored genes at the median."""
    if y.sum() == 0:
        return np.nan
    s = np.where(np.isfinite(s), s, np.nanmedian(s) if np.isfinite(s).any() else 0.0)
    o = np.argsort(-s, kind="mergesort")
    ys, ss = y[o], s[o]
    last = np.r_[np.flatnonzero(np.diff(ss)), len(ss) - 1]
    tp = np.cumsum(ys)[last]
    fp = last + 1 - tp
    prec = tp / (tp + fp)
    rec = tp / y.sum()
    return float(np.sum(np.diff(np.r_[0.0, rec]) * prec))


def topk(y: np.ndarray, s: np.ndarray, k: int, rng) -> float:
    s = np.where(np.isfinite(s), s, np.nanmedian(s) if np.isfinite(s).any() else 0.0)
    j = rng.random(len(s))  # ties in random order
    o = np.lexsort((j, -s))[:k]
    return float(y[o].mean())


def power(a: np.ndarray, c: np.ndarray) -> float:
    fc = c[np.isfinite(c)]
    if len(fc) == 0 or len(a) == 0:
        return np.nan
    thr = np.quantile(fc, 0.95)
    # a tie at the threshold is not a detection
    return float(np.mean(np.nan_to_num(a, nan=-np.inf) > thr))


def direction(sa, ra, sc, rc) -> tuple[float, float, np.ndarray]:
    diff = (sa - ra) - (sc - rc)
    ok = np.isfinite(diff)
    if ok.sum() == 0:
        return np.nan, np.nan, diff
    dd = diff[ok]
    acc = float(((dd < 0).sum() + 0.5 * (dd == 0).sum()) / len(dd))
    da = (sa - ra)[np.isfinite(sa - ra)]
    raw = float(((da < 0).sum() + 0.5 * (da == 0).sum()) / max(len(da), 1))
    return acc, raw, diff


def one_set(sa, sc, ra, rc, method_group):
    """Point metrics of one method on one set of pairs. sa, sc oriented scores; ra, rc reference."""
    da, dc = -sa, -sc
    au, auc = auroc_miss(da, dc)
    yy = np.r_[np.ones(len(da)), np.zeros(len(dc))]
    ap = avg_prec(yy, np.r_[da, dc])
    pw = power(da, dc)
    acc, raw, diff = direction(sa, ra, sc, rc)
    pair = np.isfinite(da) & np.isfinite(dc)
    if pair.sum() >= 5 and np.any((da - dc)[pair] != 0):
        p = float(wilcoxon((da - dc)[pair], alternative="greater", zero_method="zsplit").pvalue)
    else:
        p = 1.0
    return dict(AUROC=au, AUROC_cov=auc, AUPRC_match=ap, power_05=pw, dir_acc=acc, dir_raw=raw,
                wilcoxon_p=p, reject05=float(p < 0.05), coverage=float(np.isfinite(sa).mean()),
                frac_zero=float(np.mean(np.abs(np.nan_to_num(sa, nan=1.0)) < MASK_EPS)),
                _diff=diff, n=len(sa))


def reference(cohort: str) -> pd.DataFrame:
    """Mean over the cohort's unaltered fits, and the mean masked share."""
    frames = []
    for s, r in RU.COHORTS[cohort]["none_runs"].items():
        f = RU.RES / "scores" / f"{r}.parquet"
        if f.exists():
            frames.append(pd.read_parquet(f).set_index("var_name"))
    if not frames:
        raise FileNotFoundError(f"no unaltered fit scored for {cohort}")
    cols = [c for c in frames[0].columns if c != "hsapiens_gene_id" and pd.api.types.is_numeric_dtype(frames[0][c])]
    ref = sum(f[cols] for f in frames) / len(frames)
    ref["masked_share"] = sum((f["S_g"].abs() < MASK_EPS).astype(float) for f in frames) / len(frames)
    ref["hsapiens_gene_id"] = frames[0]["hsapiens_gene_id"]
    return ref


def sequence_table() -> pd.DataFrame:
    g = pd.read_csv(RU.REPO / "results/revision/T4_3/golds/gold_table.csv", index_col=0)
    g = g[["dNdS", "LOEUF"]]
    g.index = g.index.astype(str)
    return g[~g.index.duplicated()]


def join_static(sc: pd.DataFrame, seq: pd.DataFrame) -> pd.DataFrame:
    ids = sc["hsapiens_gene_id"].astype(str)
    sc = sc.copy()
    sc["dNdS"] = ids.map(seq["dNdS"]).to_numpy()
    sc["LOEUF"] = ids.map(seq["LOEUF"]).to_numpy()
    return sc


def score_run(row: dict, truth: pd.DataFrame, scores: pd.DataFrame, ref: pd.DataFrame, seq: pd.DataFrame,
              band_mask: np.ndarray, methods=None, extra: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long table of metrics of one run. ``band_mask``: genes of the run's expression band, in object order."""
    methods = methods or METHODS
    sc = join_static(scores.set_index("var_name") if "var_name" in scores else scores, seq)
    if extra is not None:
        sc = sc.join(extra)
    rfull = join_static(ref, seq)
    if extra is not None:
        for c in extra.columns:
            rfull[c] = rfull.get(c, np.nan)
    a, c = truth["altered"].to_numpy(), truth["control"].to_numpy()
    rng = np.random.default_rng(RU.useed("metrics", row["run"]))
    subsets = {"all": np.ones(len(truth), bool)}
    for b in RU.IDENTITY_BANDS:
        subsets[b] = truth["identity_band"].to_numpy() == b
    mk = rfull["masked_share"]
    subsets["ref_unmasked"] = (mk.reindex(a).to_numpy() < 0.5) & (mk.reindex(c).to_numpy() < 0.5)
    subsets["ref_masked"] = (mk.reindex(a).to_numpy() >= 0.5) & (mk.reindex(c).to_numpy() >= 0.5)
    y_all = pd.Series(0.0, index=sc.index)
    y_all.loc[a] = 1.0
    rows = []
    for m, (col, sign, grp) in methods.items():
        if col not in sc:
            continue
        s = sign * sc[col].astype(float)
        r = sign * rfull[col].astype(float) if col in rfull else pd.Series(np.nan, index=sc.index)
        sa, sc_, ra, rc = s.reindex(a).to_numpy(), s.reindex(c).to_numpy(), r.reindex(a).to_numpy(), r.reindex(c).to_numpy()
        for name, mask in subsets.items():
            if mask.sum() < 5:
                continue
            res = one_set(sa[mask], sc_[mask], ra[mask], rc[mask], grp)
            diff = res.pop("_diff")
            n = res.pop("n")
            if name == "all":
                sdr = np.nanstd(ra[mask]) if np.isfinite(ra[mask]).any() else np.nan
                res["effect_sd"] = float(np.nanmean(diff) / sdr) if np.isfinite(sdr) and sdr > 1e-9 and np.isfinite(diff).any() else np.nan
                res["AUPRC_genome"] = avg_prec(y_all.to_numpy(), -s.to_numpy())
                res["topk_genome"] = topk(y_all.to_numpy(), -s.to_numpy(), len(a), rng)
                res["topk_band"] = topk(y_all.to_numpy()[band_mask], -s.to_numpy()[band_mask],
                                        min(len(a), int(band_mask.sum())), rng)
            for k, v in res.items():
                rows.append(dict(method=m, group=grp, subset=name, metric=k, value=v, n=n))
    return pd.DataFrame(rows)


def bootstrap_all(row: dict, truth: pd.DataFrame, scores: pd.DataFrame, ref: pd.DataFrame, seq: pd.DataFrame, methods=None,
                  extra: pd.DataFrame | None = None, n_boot: int = BOOT) -> dict:
    methods = methods or METHODS
    sc = join_static(scores.set_index("var_name") if "var_name" in scores else scores, seq)
    if extra is not None:
        sc = sc.join(extra)
    rfull = join_static(ref, seq)
    a, c = truth["altered"].to_numpy(), truth["control"].to_numpy()
    n = len(a)
    rng = np.random.default_rng(RU.useed("boot", row["run"]))
    ix = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for m, (col, sign, _) in methods.items():
        if col not in sc:
            continue
        s = sign * sc[col].astype(float)
        r = sign * rfull[col].astype(float) if col in rfull else pd.Series(np.nan, index=sc.index)
        sa, sc_, ra, rc = s.reindex(a).to_numpy(), s.reindex(c).to_numpy(), r.reindex(a).to_numpy(), r.reindex(c).to_numpy()
        M = np.full((n_boot, len(BOOT_METRICS)), np.nan)
        for b in range(n_boot):
            i = ix[b]
            da, dc = -sa[i], -sc_[i]
            M[b, 0] = auroc_miss(da, dc)[0]
            M[b, 1] = avg_prec(np.r_[np.ones(n), np.zeros(n)], np.r_[da, dc])
            M[b, 2] = power(da, dc)
            M[b, 3] = direction(sa[i], ra[i], sc_[i], rc[i])[0]
        out[m] = M.astype(np.float32)
    return out


def is_discarded(run: str) -> bool:
    f = RU.RES / "health" / f"{run}.json"
    return json.loads(f.read_text()).get("discard", True) if f.exists() else True
