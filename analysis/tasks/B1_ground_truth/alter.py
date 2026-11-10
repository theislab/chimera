"""B1 Track A: select gene sets and alter counts in one species, keeping depth and sparsity real.

Pure numpy / scipy on a CSR count matrix. Every draw comes from a seeded generator, so an altered object
regenerates exactly from its grid row (cohort, alteration, effect, band, share, seed). The h5ad is removed
after a run; the truth table stays.

Selection (paired, nested across shares and effects). Each gene gets one altered species (balanced coin,
seed-level). Eligible: >= 20 counts in each species; pattern, loss and thin/sham share one rule per family
(pattern and loss also need a >= 2-fold contrast between the top and the bottom shared type in that species).
Within the band, one identity tertile at a time, altered genes are drawn at random and each gets the nearest
unused eligible gene in the same tertile on (expression rank, pi rank, log2 contrast rank, z-scored) as its
untouched control. The list is ordered at random across tertiles, so the 2% set lies inside the 5% set inside
the 10% set, and effects of one family share genes. n = round(share x all genes), capped at half the eligible.

Alterations (all in the gene's altered species only, except thin_both)
  pattern q   Top type A and bottom type B of the shared types exchange rates. Total conserved exactly because
              counts only move between cells: target rates s_t = swapped r_t; if the swap needs less mass than
              the gene has, the excess goes to all types as a flat rate b, otherwise all rates are scaled by c.
              r'_t = (1-q) r_t + q s_t. Types that lose mass are thinned binomially; the moved count of a cell
              stays one packet (keeps burstiness) and goes to a cell of a gaining type, chosen by type with
              probability proportional to its gain, then without replacement by library size, larger packets to deeper cells.
  level f     x' ~ Binomial(x, 1/f) in all cells of the altered species.
  loss e      x' ~ Binomial(x, 1-e) in the gene's top shared type of the altered species.
Nulls
  none        no change.
  sham_swap   the gene's counts are permuted among cells of the same type and library decile: rates and totals stay.
  thin_both   x' ~ Binomial(x, 1/f) in both species: expression falls, no divergence.

p_null is recomputed as the pipeline does (precompute_p_null, eps 1). Sequence features and pi stay. The three
expression columns of the GIN features move with the data: F' = F + (stat' - stat) / SD(stat) over genes, with
stat as compute_expression_stats, so an unaltered object keeps exactly the canonical features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.tasks.B1_ground_truth import rules as RU

EXPR_COLS = [66, 67, 68]


# ---------------------------------------------------------------------------
# Base object
# ---------------------------------------------------------------------------

def read_base(cohort: str):
    """AnnData of the canonical object (read fully)."""
    import anndata as ad

    return ad.read_h5ad(RU.REPO / RU.COHORTS[cohort]["h5ad"])


def labels(a, cohort: str) -> np.ndarray:
    cfg = RU.COHORTS[cohort]
    lab = a.obs[cfg["label_key"]].astype(str)
    if cfg["coarse"]:
        from analysis.tasks.T4_2_gene_baselines.run import TESTIS_COARSE

        lab = lab.map(TESTIS_COARSE)
    assert lab.notna().all()
    return lab.to_numpy()


def shared_types(lab: np.ndarray, spc: np.ndarray) -> list[str]:
    t = pd.crosstab(lab, spc)
    return sorted(t.index[(t >= RU.MIN_TYPE_CELLS).all(1)])


def csr(X) -> sp.csr_matrix:
    return X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)


def expr_stats(X: sp.csr_matrix) -> np.ndarray:
    """compute_expression_stats, as columns log_mean_expr, log_expr_var, frac_nonzero."""
    X = X.astype(np.float64)
    m = np.asarray(X.mean(0)).ravel()
    X2 = X.copy()
    X2.data = X2.data ** 2
    v = np.maximum(np.asarray(X2.mean(0)).ravel() - m ** 2, 0.0)
    f = np.asarray((X > 0).sum(0)).ravel() / X.shape[0]
    return np.column_stack([np.log1p(m), np.log1p(v), f])


def p_null(X: sp.csr_matrix, spc: np.ndarray, eps: float = 1.0) -> np.ndarray:
    """precompute_p_null, rows in sorted species order."""
    G = X.shape[1]
    out = []
    for s in sorted(set(spc)):
        Xs = X[spc == s]
        out.append((np.asarray(Xs.sum(0)).ravel() + eps) / (Xs.sum() + eps * G))
    return np.vstack(out)


def gene_table(a, cohort: str) -> tuple[pd.DataFrame, dict]:
    """Per gene: band, identity band, altered-species coin inputs, type rates per species."""
    X = csr(a.X).astype(np.float64)
    spc = a.obs["species"].astype(str).to_numpy()
    lab = labels(a, cohort)
    L = np.asarray(X.sum(1)).ravel()
    types = shared_types(lab, spc)
    cp = sp.diags(1e4 / np.maximum(L, 1)).dot(X).tocsr()
    lcp = cp.copy()
    lcp.data = np.log1p(lcp.data)
    Xb = X.copy()
    Xb.data = (Xb.data > 0).astype(np.float64)
    sp_lcp = np.stack([np.asarray(lcp[spc == s].mean(0)).ravel() for s in RU.SPECIES])
    tot = np.stack([np.asarray(X[spc == s].sum(0)).ravel() for s in RU.SPECIES])
    det = np.stack([np.asarray(Xb[spc == s].mean(0)).ravel() for s in RU.SPECIES])
    g = pd.DataFrame(index=pd.Index(a.var_names.astype(str), name="var_name"))
    g["idx"] = np.arange(a.n_vars)
    g["log_cp10k_spmean"] = sp_lcp.mean(0)
    g["detection_spmean"] = det.mean(0)
    g["prior_raw"] = np.asarray(a.var["conservation_prior"], dtype=float)
    g["total_Human"], g["total_Mouse"] = tot[0], tot[1]
    g["band"] = pd.qcut(g["log_cp10k_spmean"].rank(method="first"), 4, labels=RU.BANDS).astype(str)
    g["identity_band"] = pd.qcut(g["prior_raw"].rank(method="first"), 3, labels=RU.IDENTITY_BANDS).astype(str)
    # type rates with the half-count pseudocount of the direct margin, per altered species
    rates = {}
    for s in RU.SPECIES:
        rs = []
        for t in types:
            m = (spc == s) & (lab == t)
            rs.append((np.asarray(X[m].sum(0)).ravel() + 0.5) / L[m].sum())
        R = np.column_stack(rs)
        rates[s] = R
        g[f"top_{s}"] = np.array(types)[R.argmax(1)]
        g[f"bottom_{s}"] = np.array(types)[R.argmin(1)]
        g[f"log2_contrast_{s}"] = np.log2(R.max(1) / R.min(1))
    return g, dict(types=types, rates=rates, spc=spc, lab=lab, L=L)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

def family_of(alt: str) -> str:
    return {"pattern": "contrast", "loss": "contrast", "sham_swap": "contrast", "level": "any", "thin_both": "any",
            "none": "any"}[alt]


def select_pairs(g: pd.DataFrame, alt: str, band: str, seed: int, n_max: int | None = None,
                 rng: np.random.Generator | None = None) -> pd.DataFrame:
    """Ordered altered-control pairs for the largest share; a share takes the first n rows. Independent of effect."""
    fam = family_of(alt)
    if rng is None:
        rng = np.random.default_rng(RU.useed("select", g.index[0], fam, alt, band, seed))
    coin = np.random.default_rng(RU.useed("coin", g.index[0], seed)).permutation(len(g)) % 2
    alt_sp = np.array(RU.SPECIES)[coin]
    d = g.copy()
    d["species_altered"] = alt_sp
    d["log2_contrast"] = np.where(alt_sp == "Human", d["log2_contrast_Human"], d["log2_contrast_Mouse"])
    d["type_top"] = np.where(alt_sp == "Human", d["top_Human"], d["top_Mouse"])
    d["type_bottom"] = np.where(alt_sp == "Human", d["bottom_Human"], d["bottom_Mouse"])
    ok = (d["band"] == band) & (d["total_Human"] >= RU.MIN_COUNTS) & (d["total_Mouse"] >= RU.MIN_COUNTS)
    if fam == "contrast":
        ok &= d["log2_contrast"] >= RU.MIN_CONTRAST_LOG2
    e = d[ok].copy()
    n_all = int(round(max(RU.SHARES) * len(g))) if n_max is None else n_max
    n_cap = int(len(e) * RU.MAX_PAIRS_FRAC)
    n = min(n_all, n_cap)
    # balanced over identity tertiles
    per = {t: (e["identity_band"] == t).sum() for t in RU.IDENTITY_BANDS}
    want = {t: n // 3 for t in RU.IDENTITY_BANDS}
    for t in RU.IDENTITY_BANDS[: n % 3]:
        want[t] += 1
    for _ in range(3):  # a thin tertile hands its remainder to the others
        spare = sum(max(0, want[t] - per[t] // 2) for t in want)
        for t in want:
            want[t] = min(want[t], per[t] // 2)
        for t in want:
            add = min(spare, max(0, per[t] // 2 - want[t]))
            want[t] += add
            spare -= add
    Z = np.column_stack([e[c].rank().to_numpy() for c in ("log_cp10k_spmean", "prior_raw", "log2_contrast")])
    Z = (Z - Z.mean(0)) / np.maximum(Z.std(0), 1e-9)
    pos = {v: i for i, v in enumerate(e.index)}
    pairs = []
    for t in RU.IDENTITY_BANDS:
        cand = e.index[e["identity_band"] == t].to_numpy()
        cand = cand[rng.permutation(len(cand))]
        chosen = cand[: want[t]]
        pool = np.array([pos[v] for v in cand[want[t]:]])
        used = np.zeros(len(pool), bool)
        for v in chosen:
            dist = ((Z[pool] - Z[pos[v]]) ** 2).sum(1)
            dist[used] = np.inf
            j = int(dist.argmin())
            used[j] = True
            pairs.append((v, e.index[pool[j]], t, float(np.sqrt(dist[j]))))
    p = pd.DataFrame(pairs, columns=["altered", "control", "identity_band", "match_distance"])
    p = p.iloc[rng.permutation(len(p))].reset_index(drop=True)
    p["pair"] = np.arange(len(p))
    p["n_eligible"] = len(e)
    p["n_wanted_max_share"] = n_all
    for col in ("species_altered", "type_top", "type_bottom", "log2_contrast", "log_cp10k_spmean", "prior_raw"):
        p[col] = e.loc[p["altered"], col].to_numpy()
        p[f"ctrl_{col}"] = e.loc[p["control"], col].to_numpy()
    return p


def truth_for(g: pd.DataFrame, alt: str, band: str, share: float, seed: int) -> pd.DataFrame:
    p = select_pairs(g, alt, band, seed)
    n = min(int(round(share * len(g))), len(p))
    t = p.iloc[:n].copy()
    t["share_wanted"] = share
    t["share_achieved"] = n / len(g)
    return t


# ---------------------------------------------------------------------------
# Alteration
# ---------------------------------------------------------------------------

def _thin(rng, x, keep):
    return rng.binomial(x.astype(np.int64), keep)


def swap_targets(T_t: np.ndarray, L_t: np.ndarray, ia: int, ib: int, q: float) -> np.ndarray:
    """Target count totals per type after moving a share q from the observed profile to the swapped one.

    T_t counts and L_t library sums per type (all types of the species). Conserves T_t.sum()."""
    r = T_t / L_t
    s = r.copy()
    s[ia], s[ib] = r[ib], r[ia]
    T = T_t.sum()
    need = (s * L_t).sum()
    s = s + (T - need) / L_t.sum() if need <= T else s * (T / need)
    return ((1 - q) * r + q * s) * L_t


def assign_packets(rng, pk: np.ndarray, idx: np.ndarray, L: np.ndarray, out: np.ndarray) -> None:
    """Add packets to cells ``idx`` without replacement, round after round, so counts do not pile up on a few cells while
    others stay empty. Larger packets go to deeper cells within a round, which keeps the count-depth relation."""
    pk = pk[rng.permutation(len(pk))]
    w = L[idx] / L[idx].sum()
    done = 0
    while done < len(pk):
        k = min(len(idx), len(pk) - done)
        recv = rng.choice(idx, size=k, replace=False, p=w)
        take = np.sort(pk[done:done + k])
        out[recv[np.argsort(L[recv], kind="stable")]] += take
        done += k


def move_counts(rng, x: np.ndarray, lab_idx: np.ndarray, L: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Move counts between cells so that the type totals reach ``target`` in expectation; total exact."""
    T_t = np.bincount(lab_idx, weights=x, minlength=len(target))
    delta = target - T_t
    out = x.copy().astype(np.int64)
    packets = []
    for t in np.where(delta < -1e-9)[0]:
        if T_t[t] <= 0:
            continue
        m = lab_idx == t
        moved = rng.binomial(out[m], min(1.0, -delta[t] / T_t[t]))
        out[m] -= moved
        packets.append(moved[moved > 0])
    if not packets:
        return out
    packets = np.concatenate(packets)
    gain = np.where(delta > 1e-9, delta, 0.0)
    ptype = rng.choice(len(gain), size=len(packets), p=gain / gain.sum())
    for t in np.unique(ptype):
        assign_packets(rng, packets[ptype == t], np.where(lab_idx == t)[0], L, out)
    return out


def sham_move(rng, x: np.ndarray, lab_idx: np.ndarray, L: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """Permute the gene's counts among cells of the same type and library decile: rates and the count distribution stay."""
    out = np.zeros_like(x, dtype=np.int64)
    for t in np.unique(lab_idx):
        idx = np.where(lab_idx == t)[0]
        r = np.argsort(np.argsort(L[idx], kind="stable"), kind="stable")
        bins = (r * n_bins // max(len(idx), 1)).astype(int)
        for b in np.unique(bins):
            j = idx[bins == b]
            out[j] = x[j][rng.permutation(len(j))]
    return out


def apply_alteration(X: sp.csr_matrix, truth: pd.DataFrame, meta: dict, alt: str, effect: float, seed: int,
                     var_names: pd.Index) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """Altered CSR, and per altered gene the total before and after in each species."""
    rng = np.random.default_rng(RU.useed("alter", var_names[0], alt, effect, seed, len(truth), truth["altered"].iloc[0]
                                         if len(truth) else ""))
    spc, lab, L = meta["spc"], meta["lab"], meta["L"]
    Xc = X.tocsc().astype(np.int64)
    pos = pd.Series(np.arange(X.shape[1]), index=var_names)
    alltypes = sorted(set(lab))
    tindex = {t: i for i, t in enumerate(alltypes)}
    lab_i = np.array([tindex[t] for t in lab])
    rows, cols, vals, info = [], [], [], []
    for _, r in truth.iterrows():
        j = int(pos[r["altered"]])
        x = np.asarray(Xc[:, j].todense()).ravel()
        new = x.copy()
        s_mask = spc == r["species_altered"]
        if alt == "level":
            new[s_mask] = _thin(rng, x[s_mask], 1.0 / effect)
        elif alt == "thin_both":
            new = _thin(rng, x, 1.0 / effect)
        elif alt == "loss":
            m = s_mask & (lab == r["type_top"])
            new[m] = _thin(rng, x[m], 1.0 - effect)
        elif alt in ("pattern", "sham_swap"):
            idx = np.where(s_mask)[0]
            li = lab_i[idx]
            if alt == "sham_swap":
                new[idx] = sham_move(rng, x[idx], li, L[idx])
            else:
                L_t = np.bincount(li, weights=L[idx], minlength=len(alltypes))
                T_t = np.bincount(li, weights=x[idx], minlength=len(alltypes)).astype(float)
                present = L_t > 0
                tgt = np.zeros(len(alltypes))
                map_i = {v: k for k, v in enumerate(np.where(present)[0])}
                tgt[present] = swap_targets(T_t[present], L_t[present], map_i[tindex[r["type_top"]]],
                                            map_i[tindex[r["type_bottom"]]], float(effect))
                new[idx] = move_counts(rng, x[idx], li, L[idx], tgt)
        else:
            raise ValueError(alt)
        d = new - x
        nz = np.flatnonzero(d)
        rows.append(nz)
        cols.append(np.full(len(nz), j))
        vals.append(d[nz])
        info.append(dict(altered=r["altered"],
                         total_before_altered_sp=float(x[s_mask].sum()), total_after_altered_sp=float(new[s_mask].sum()),
                         total_before_other_sp=float(x[~s_mask].sum()), total_after_other_sp=float(new[~s_mask].sum()),
                         det_before_altered_sp=float((x[s_mask] > 0).mean()), det_after_altered_sp=float((new[s_mask] > 0).mean()),
                         n_cells_changed=int(len(nz))))
    D = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=X.shape, dtype=np.int64)
    Xn = (X.astype(np.int64) + D).tocsr()
    Xn.eliminate_zeros()
    if Xn.data.size and Xn.data.min() < 0:
        raise AssertionError("negative counts after alteration")
    return Xn, pd.DataFrame(info)


def features_after(F: np.ndarray, X0: sp.csr_matrix, X1: sp.csr_matrix) -> np.ndarray:
    """Canonical features with the expression columns moved by the change in the pipeline statistic."""
    s0, s1 = expr_stats(X0), expr_stats(X1)
    F1 = np.array(F, copy=True)
    sd = s0.std(0, ddof=0)
    for k, c in enumerate(EXPR_COLS):
        F1[:, c] = F1[:, c] + (s1[:, k] - s0[:, k]) / sd[k]
    return F1


def build_altered(row: dict, out_path=None, check: bool = True):
    """Build one altered object from a grid row; returns (adata, truth)."""
    cohort, alt, eff, band, share, seed = (row["cohort"], row["alt"], float(row["effect"]), row["band"],
                                           float(row["share"]), int(row["seed"]))
    a = read_base(cohort)
    X0 = csr(a.X)
    assert np.allclose(X0.data, np.round(X0.data)), "X is not integer"
    g, meta = gene_table(a, cohort)
    truth = truth_for(g, alt, band, share, seed)
    X1, info = apply_alteration(X0, truth, meta, alt, eff, seed, g.index)
    truth = truth.merge(info, on="altered", how="left")
    ids = gene_ids.ensembl(RU.COHORTS[cohort]["gene_table"], truth["altered"])
    truth["hsapiens_gene_id"] = np.asarray(ids)
    truth.insert(0, "run", row["run"])
    b = a.copy()
    b.X = X1.astype(np.float32)
    b.uns["p_null"] = p_null(X1.astype(np.float64), meta["spc"])
    b.varm["gene_features"] = features_after(np.asarray(a.varm["gene_features"]), X0, X1).astype(a.varm["gene_features"].dtype)
    if out_path is not None:
        b.write_h5ad(out_path)
        if check:
            from analysis.common.check_cohort import check as cc

            errs = cc(str(out_path), batch_key="batch")
            if errs:
                raise AssertionError(f"check_cohort: {errs}")
    return b, truth
