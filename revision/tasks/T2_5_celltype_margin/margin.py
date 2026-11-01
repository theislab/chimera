"""Cell-type decomposition of the fit-quality margin S_g, from forward-pass sums.

Pure numpy / pandas; the forward pass that produces the sums lives in ``run.py``.

Definitions (per gene g, species s, cell type k, K species, n_s cells in s, n_sk in (s, k)):

    l_cg      = log NB(x_cg | mu_shared_cg, theta) - log NB(x_cg | p_null[s,g] L_c, theta)
    S_gsk     = mean_{c in s,k} l_cg                                  (unweighted)
    w_sk      = n_sk / (n_s K)
    C_gk      = sum_s w_sk S_gsk                                      (abundance-weighted)
    S_g       = sum_k C_gk = (1/K) sum_s mean_{c in s} l_cg           (exact identity)

The within-type null replaces p_null[s,g] by the Laplace-smoothed rate of (s, k), so
S^wt_gsk asks whether the shared decoder beats a flat model inside k, rather than
"the gene is correctly low outside its own type".

Margin support flag (fixed before any run): a gene's |delta-alpha| has margin support if
its trained margin S_g_ema is unmasked (|S| >= 1e-6), has the sign of delta-alpha, and
|S_g_ema| is at least the median |S_g_ema| of unmasked genes in the cohort. Tail genes are
|delta-alpha| > 0.05. The same flag on the forward-pass margin is reported as sensitivity.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

MASK_EPS = 1e-6
TAIL = 0.05
MAJOR_WEIGHT = 0.10
ACCEPT_CLOSURE = 0.97
ACCEPT_IDENTITY = 1e-6
ACCEPT_SEED = 0.9


def decompose(fw: dict, key: str) -> dict:
    """Group tables for one label key from the forward sums ``fw``."""
    g = fw[f"{key}__groups"]  # structured: species index, label
    sp_idx = np.asarray(g["species_idx"], dtype=int)
    labels = np.asarray(g["label"]).astype(str)
    n = np.asarray(g["n"], dtype=float)
    n_s = np.asarray(fw["n_species"], dtype=float)
    K = len(n_s)
    w = n / n_s[sp_idx] / K
    out = dict(species_idx=sp_idx, labels=labels, n=n, w=w, K=K, cell_types=sorted(set(labels)))
    for mode in ("meanz", "samp", "within"):
        S_gsk = fw[f"{key}__sum_{mode}"] / n[:, None]
        out[f"S_gsk_{mode}"] = S_gsk
        out[f"C_gk_{mode}"] = {k: (S_gsk[labels == k] * w[labels == k, None]).sum(0) for k in out["cell_types"]}
    for mode in ("meanz", "samp"):
        out[f"S_full_{mode}"] = (fw[f"sp_sum_{mode}"] / n_s[:, None]).mean(0)
    species = [str(s) for s in fw["species"]]
    shared = set.intersection(*[set(labels[sp_idx == i]) for i in range(K)])
    out["shared"] = shared
    out["species"] = species
    return out


def margin_flag(delta_alpha: np.ndarray, S: np.ndarray) -> np.ndarray:
    unmasked = np.abs(S) >= MASK_EPS
    thr = np.median(np.abs(S[unmasked])) if unmasked.any() else np.inf
    return unmasked & (np.sign(S) == np.sign(delta_alpha)) & (np.abs(S) >= thr)


def gene_table(fw: dict, dec: dict, genes: pd.DataFrame) -> pd.DataFrame:
    """Per gene: margins, pass rate, delta-alpha, support flags, dominant cell type."""
    t = genes.copy()
    t["S_full_meanz"] = dec["S_full_meanz"]
    t["S_full_samp"] = dec["S_full_samp"]
    t["ppass_S_samp"] = t["ppass"] * t["S_full_samp"]
    t["S_ema_checkpoint"] = fw["S_ema"]
    da = t["delta_alpha"].to_numpy()
    t["tail"] = np.abs(da) > TAIL
    t["margin_support"] = margin_flag(da, t["S_g_ema"].to_numpy())
    t["margin_support_forward"] = margin_flag(da, t["S_full_samp"].to_numpy())
    t["lacks_margin_support"] = t["tail"] & ~t["margin_support"]
    cts = dec["cell_types"]
    M = np.stack([dec["C_gk_meanz"][k] for k in cts])
    t["dominant_cell_type"] = np.array(cts)[np.abs(M).argmax(0)]
    t["dominant_share_abs"] = np.abs(M).max(0) / np.maximum(np.abs(M).sum(0), 1e-12)
    return t


def long_tables(dec: dict, genes: pd.DataFrame, key: str, boot: dict | None):
    """Long-format S_gsk and C_gk tables."""
    base = genes[["var_name", "hsapiens_gene_id", "symbol"]].reset_index(drop=True)
    G = len(base)
    rows = []
    for j in range(len(dec["labels"])):
        d = base.copy()
        d["group_key"] = key
        d["species"] = dec["species"][dec["species_idx"][j]]
        d["cell_type"] = dec["labels"][j]
        d["n_cells"] = int(dec["n"][j])
        d["weight"] = dec["w"][j]
        d["shared_type"] = dec["labels"][j] in dec["shared"]
        for mode in ("meanz", "samp", "within"):
            d[f"S_gsk_{mode}"] = dec[f"S_gsk_{mode}"][j]
        rows.append(d)
    S_long = pd.concat(rows, ignore_index=True)
    rows = []
    for k in dec["cell_types"]:
        d = base.copy()
        d["group_key"] = key
        d["cell_type"] = k
        d["weight_k"] = dec["w"][dec["labels"] == k].sum()
        d["shared_type"] = k in dec["shared"]
        for mode in ("meanz", "samp", "within"):
            d[f"C_gk_{mode}"] = dec[f"C_gk_{mode}"][k]
        d["boot_lo"] = np.nan
        d["boot_hi"] = np.nan
        if boot is not None:
            lo, hi = boot["lo"][k], boot["hi"][k]
            d.loc[boot["gene_idx"], "boot_lo"] = lo
            d.loc[boot["gene_idx"], "boot_hi"] = hi
        rows.append(d)
    C_long = pd.concat(rows, ignore_index=True)
    assert len(C_long) == G * len(dec["cell_types"])
    return S_long, C_long


def bootstrap(fw: dict, dec: dict, key: str, n_boot: int, seed: int) -> dict | None:
    """Cell bootstrap within (species, type) groups for landmark genes, mean-z margin."""
    if "landmark_l" not in fw or fw["landmark_l"].size == 0:
        return None
    L = fw["landmark_l"]  # cells x landmarks
    grp = fw[f"{key}__cell_group"]
    rng = np.random.default_rng(seed)
    members = [np.where(grp == j)[0] for j in range(len(dec["labels"]))]
    draws = {k: np.empty((n_boot, L.shape[1])) for k in dec["cell_types"]}
    for b in range(n_boot):
        S = np.stack([L[rng.choice(m, len(m), replace=True)].mean(0) for m in members])
        for k in dec["cell_types"]:
            sel = dec["labels"] == k
            draws[k][b] = (S[sel] * dec["w"][sel, None]).sum(0)
    return dict(gene_idx=np.asarray(fw["landmark_idx"], dtype=int),
                lo={k: np.percentile(v, 2.5, axis=0) for k, v in draws.items()},
                hi={k: np.percentile(v, 97.5, axis=0) for k, v in draws.items()})


def acceptance(dec: dict, t: pd.DataFrame, cohort: str, key: str) -> list[dict]:
    rows = []
    S_ema = t["S_g_ema"].to_numpy()
    r = pearsonr(t["ppass_S_samp"], S_ema)[0]
    rows.append(dict(cohort=cohort, group_key=key, check="pearson(pass_rate x S_samp, S_g_ema)",
                     value=float(r), threshold=ACCEPT_CLOSURE, passed=bool(r >= ACCEPT_CLOSURE)))
    rows.append(dict(cohort=cohort, group_key=key, check="pearson(S_samp, S_g_ema), no pass rate",
                     value=float(pearsonr(t["S_full_samp"], S_ema)[0]), threshold=np.nan, passed=np.nan))
    rows.append(dict(cohort=cohort, group_key=key, check="slope S_g_ema on pass_rate x S_samp",
                     value=float(np.polyfit(t["ppass_S_samp"], S_ema, 1)[0]), threshold=np.nan, passed=np.nan))
    for mode in ("meanz", "samp"):
        tot = sum(dec[f"C_gk_{mode}"].values())
        err = float(np.abs(tot - dec[f"S_full_{mode}"]).max())
        rows.append(dict(cohort=cohort, group_key=key, check=f"max |sum_k C_gk - S_g| ({mode})", value=err,
                         threshold=ACCEPT_IDENTITY, passed=bool(err <= ACCEPT_IDENTITY)))
    wsum = float(abs(dec["w"].sum() - 1))
    rows.append(dict(cohort=cohort, group_key=key, check="|sum of weights - 1|", value=wsum,
                     threshold=ACCEPT_IDENTITY, passed=bool(wsum <= ACCEPT_IDENTITY)))
    return rows


def shares(dec: dict, t: pd.DataFrame, cohort: str, key: str) -> list[dict]:
    S = dec["S_full_meanz"]
    v = np.var(S, ddof=1)
    big = np.abs(S) > np.percentile(np.abs(S), 50)
    dom = t["dominant_cell_type"].to_numpy()
    rows = []
    for k in dec["cell_types"]:
        Ck = dec["C_gk_meanz"][k]
        Cw = dec["C_gk_within"][k]
        rows.append(dict(cohort=cohort, group_key=key, cell_type=k, shared_type=k in dec["shared"],
                         weight=float(dec["w"][dec["labels"] == k].sum()),
                         share_var_S=float(np.cov(Ck, S)[0, 1] / v),
                         frac_top_half_genes_dominant=float((dom[big] == k).mean()),
                         rho_C_within_vs_C=float(spearmanr(Cw, Ck)[0]) if np.std(Cw) > 0 and np.std(Ck) > 0 else np.nan,
                         mean_C=float(Ck.mean()), mean_C_within=float(Cw.mean())))
    return rows


def headline(t: pd.DataFrame, cohort: str, key: str) -> dict:
    tail = t["tail"].to_numpy()
    pos, neg = tail & (t["delta_alpha"] > 0), tail & (t["delta_alpha"] < 0)
    sup, supf = t["margin_support"].to_numpy(), t["margin_support_forward"].to_numpy()
    masked = np.abs(t["S_g_ema"]) < MASK_EPS
    return dict(cohort=cohort, group_key=key, n_genes=len(t), n_tail=int(tail.sum()),
                frac_tail_with_margin_support=float(sup[tail].mean()) if tail.any() else np.nan,
                frac_pos_tail_with_support=float(sup[pos].mean()) if pos.any() else np.nan,
                frac_neg_tail_with_support=float(sup[neg].mean()) if neg.any() else np.nan,
                frac_tail_masked=float(masked[tail].mean()) if tail.any() else np.nan,
                frac_tail_with_support_forward_margin=float(supf[tail].mean()) if tail.any() else np.nan,
                frac_all_genes_with_support=float(sup.mean()))


def seed_stability(decs: dict, cohort: str, key: str) -> list[dict]:
    """Spearman of C_gk (mean-z) across seed pairs, per cell type."""
    names = sorted(decs)
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = decs[names[i]], decs[names[j]]
            for k in a["cell_types"]:
                if k not in b["C_gk_meanz"]:
                    continue
                x, y = a["C_gk_meanz"][k], b["C_gk_meanz"][k]
                w = float(a["w"][a["labels"] == k].sum())
                rows.append(dict(cohort=cohort, group_key=key, run_a=names[i], run_b=names[j], cell_type=k,
                                 weight=w, major=w >= MAJOR_WEIGHT,
                                 rho=float(spearmanr(x, y)[0]) if np.std(x) > 0 and np.std(y) > 0 else np.nan,
                                 rho_within=float(spearmanr(a["C_gk_within"][k], b["C_gk_within"][k])[0])))
            rows.append(dict(cohort=cohort, group_key=key, run_a=names[i], run_b=names[j], cell_type="(S_g total)",
                             weight=1.0, major=False, rho=float(spearmanr(a["S_full_meanz"], b["S_full_meanz"])[0]),
                             rho_within=np.nan))
    return rows
