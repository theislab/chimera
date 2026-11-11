"""Step 1: per-cohort gap tables, seed noise, calls, calibration null, covariates, ESM-2 neighborhoods and sequence constraint.

    PYTHONPATH=$PWD python -m analysis.tasks.X8_gate_divergence.gap [--jobs 8]
Inputs: results/analysis/Z4/h1/genes/<c>.parquet (per-gene gate), results/analysis/H1/genes/<c>.parquet (canonical gate), the cohort's
training object (64 ESM-2 PCs in varm['gene_features']).
Outputs in results/analysis/X8/: genes/<c>.parquet, cohort_gap.csv, calibration.csv, covariates.csv, neighbours.csv, sequence.csv,
cache/esm__<c>.parquet (PCs and Leiden modules, as in analysis/tasks/T4_3_scoreboard, of the cohort's genes).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

import argparse  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm, spearmanr  # noqa: E402

from analysis.common import controls as K  # noqa: E402
from analysis.tasks.X8_gate_divergence import rules as RU  # noqa: E402

OUT = RU.OUT


def inv_norm_rank(v) -> np.ndarray:
    r = pd.Series(np.asarray(v, float)).rank(method="average").to_numpy()
    return norm.ppf((r - 0.5) / len(r))


def rho(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() > 20 else np.nan


def design(cols: list[np.ndarray]) -> np.ndarray:
    return np.c_[np.ones(len(cols[0])), *[K.ns_basis(pd.Series(c).rank().to_numpy(), 4) for c in cols]]


def resid_rank(y, Z) -> np.ndarray:
    r = pd.Series(np.asarray(y, float)).rank().to_numpy()
    return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]


def partial_rho(a, b, covs) -> float:
    Z = design(covs)
    return float(np.corrcoef(resid_rank(a, Z), resid_rank(b, Z))[0, 1])


def block_boot(fn, n_obs: int, blocks: np.ndarray, n: int, rng) -> tuple[float, float]:
    """Percentile CI of fn(idx) over ESM-module bootstrap draws."""
    ids, inv = np.unique(blocks, return_inverse=True)
    members = [np.flatnonzero(inv == i) for i in range(len(ids))]
    out = []
    for _ in range(n):
        pick = rng.integers(0, len(ids), len(ids))
        idx = np.concatenate([members[i] for i in pick])
        out.append(fn(idx))
    out = np.asarray(out, float)
    out = out[np.isfinite(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) > 10 else (np.nan, np.nan)


# ---------------------------------------------------------------- tables


def universe(c: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = pd.read_parquet(RU.FREE / f"{c}.parquet")
    k = pd.read_parquet(RU.CAN / f"{c}.parquet").reindex(f.index)
    assert (f.hsapiens_gene_id.fillna("") == k.hsapiens_gene_id.fillna("")).all() and np.allclose(f.pi_clamped, k.pi_clamped)
    sup = f.count_support.to_numpy(bool) & k.count_support.to_numpy(bool)
    ok = sup & f.hsapiens_gene_id.notna().to_numpy() & ~f.hsapiens_gene_id.duplicated(keep=False).to_numpy()
    return f[ok].copy(), k[ok].copy()


def seed_z(g: pd.DataFrame) -> np.ndarray:
    cols = sorted([x for x in g.columns if x.startswith("alpha_s")])
    d = g[cols].to_numpy(float) - g.pi_clamped.to_numpy(float)[:, None]
    return np.column_stack([inv_norm_rank(d[:, j]) for j in range(d.shape[1])]), d


def decile(expr: np.ndarray) -> np.ndarray:
    return pd.qcut(pd.Series(expr).rank(method="first"), RU.EXPR_DECILES, labels=False).to_numpy()


def fit_prior(Z: np.ndarray, dec: np.ndarray) -> tuple[float, np.ndarray]:
    """Empirical-Bayes prior for per-gene seed variances (Smyth 2004, limma fitFDist), prior mean by expression decile (limma-trend form).
    Returns prior df d0 (capped at D0_MAX) and the prior variance of each decile."""
    from scipy.optimize import brentq
    from scipy.special import digamma, polygamma

    d = Z.shape[1] - 1
    s2 = np.maximum(Z.var(1, ddof=1), 1e-10)
    e = np.log(s2) - digamma(d / 2) + np.log(d / 2)
    em = pd.Series(e).groupby(dec).mean().reindex(range(RU.EXPR_DECILES)).to_numpy()
    r = e - em[dec]
    evar = float(np.var(r, ddof=RU.EXPR_DECILES)) - float(polygamma(1, d / 2))
    d0 = RU.D0_MAX if evar <= 0 else min(RU.D0_MAX, 2 * brentq(lambda x: float(polygamma(1, x)) - evar, 1e-8, 1e8))
    s0 = np.exp(em + digamma(d0 / 2) - np.log(d0 / 2))
    return float(d0), s0


def moderated(Z: np.ndarray, dec: np.ndarray, prior: tuple[float, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Moderated per-gene variance of seed z and its df. One seed: the borrowed prior (variance s0 of the decile, df d0)."""
    n = Z.shape[1]
    if n < 2:
        assert prior is not None
        d0, s0 = prior
        return s0[dec], np.full(len(dec), d0)
    d0, s0 = fit_prior(Z, dec) if prior is None else prior
    d = n - 1
    v = (d * Z.var(1, ddof=1) + d0 * s0[dec]) / (d + d0)
    return v, np.full(len(dec), d + d0)


def borrowed_prior(c: str) -> tuple[float, np.ndarray]:
    _, k = universe(c)
    Z, _ = seed_z(k)
    return fit_prior(Z, decile(k.expr.to_numpy(float)))


def gap_stats(Za: np.ndarray, Zb: np.ndarray, dec: np.ndarray, prior_b=None) -> dict:
    """delta = mean seed z of A - mean seed z of B; moderated Welch t; BH q; calls."""
    from scipy.stats import t as tdist

    from analysis.tasks.H1_harmonise.recurrence import bh

    va, dfa = moderated(Za, dec)
    vb, dfb = moderated(Zb, dec, prior_b)
    a_, b_ = va / Za.shape[1], vb / Zb.shape[1]
    se = np.sqrt(a_ + b_)
    df = (a_ + b_) ** 2 / (a_ ** 2 / dfa + b_ ** 2 / dfb)
    delta = Za.mean(1) - Zb.mean(1)
    p = 2 * tdist.sf(np.abs(delta) / se, df)
    q = bh(p)
    pair = Za[:, :, None] - Zb[:, None, :]
    agree = (np.sign(pair) == np.sign(delta)[:, None, None]).all((1, 2))
    big = np.abs(delta) >= RU.MIN_GAP
    sgn = np.sign(delta).astype(int)
    return dict(delta=delta, se=se, df=df, p=p, q=q, agree=agree,
                call=np.where((q < RU.Q_CALL) & big & agree, sgn, 0), call_strict=np.where((q < RU.Q_STRICT) & big & agree, sgn, 0))


def split_half(Zf: np.ndarray, Zc: np.ndarray, rng, n_splits: int = 20) -> tuple[float, float]:
    """Split-half reliability of delta: disjoint halves of both gates' seeds, Spearman between the two half-deltas, Spearman-Brown to full."""
    if Zf.shape[1] < 2 or Zc.shape[1] < 2:
        return np.nan, np.nan
    rs = []
    for _ in range(n_splits):
        pf, pc = rng.permutation(Zf.shape[1]), rng.permutation(Zc.shape[1])
        hf, hc = Zf.shape[1] // 2, Zc.shape[1] // 2
        a = Zf[:, pf[:hf]].mean(1) - Zc[:, pc[:hc]].mean(1)
        b = Zf[:, pf[hf:2 * hf]].mean(1) - Zc[:, pc[hc:2 * hc]].mean(1)
        rs.append(rho(a, b))
    r = float(np.mean(rs))
    return r, 2 * r / (1 + r)


def gap_table(c: str) -> tuple[pd.DataFrame, dict]:
    f, k = universe(c)
    Zf, df_ = seed_z(f)
    Zc, dc_ = seed_z(k)
    zf, zc = Zf.mean(1), Zc.mean(1)  # equal weight per seed (Amendment 1 in rules.py)
    e = f.expr.to_numpy(float)
    dec = decile(e)
    G = gap_stats(Zf, Zc, dec, borrowed_prior(RU.BORROW[c]) if Zc.shape[1] < 2 else None)
    delta, se = G["delta"], G["se"]
    t = pd.DataFrame(dict(cohort=c, hsapiens_gene_id=f.hsapiens_gene_id.to_numpy(), symbol=f.symbol.to_numpy(), z_free=zf, z_can=zc, delta=delta, se=se,
                          t=delta / se, df=G["df"], p=G["p"], q=G["q"], seed_pairs_agree=G["agree"], call=G["call"], call_strict=G["call_strict"],
                          z_free_D=inv_norm_rank(df_.mean(1)), z_can_D=inv_norm_rank(dc_.mean(1)),
                          D_free=f.D.to_numpy(float), D_can=k.D.to_numpy(float), S_free=f.S_ema.to_numpy(float), S_can=k.S_ema.to_numpy(float)),
                     index=f.index)
    t["r_f_given_c"] = resid_rank(zf, design([zc])) / len(zf)
    t["r_c_given_f"] = resid_rank(zc, design([zf])) / len(zf)
    for col in ("pi_raw", "pi_clamped", "expr", "detection", "tau", "dNdS", "LOEUF", "identity_human_mouse"):
        t[col] = f[col].to_numpy() if col in f else np.nan
    dets = [x for x in f.columns if x.startswith("detection_")]
    t["det_gap"] = f[dets].max(1).to_numpy() - f[dets].min(1).to_numpy()
    t["abs_S_free"], t["abs_S_can"] = np.abs(t.S_free), np.abs(t.S_can)
    sat = lambda g: (g[[x for x in g.columns if x.startswith("alpha_s")]] > RU.SAT_HI).any(axis=1).to_numpy()  # noqa: E731
    t["saturated_any"] = sat(f) | sat(k)
    for col in ("nameable", "strict_support", "quiet_in_species", "reference_artefact", "o1_T2", "o1_M_wrong_row", "o1_M_false_zero", "o1_zero_any"):
        t[f"{col}_free"] = f[col].to_numpy() if col in f else np.nan
    t["nameable_can"] = k.nameable.to_numpy()
    t["n_seeds_free"], t["n_seeds_can"] = Zf.shape[1], Zc.shape[1]
    vtot = float(np.var(delta))
    summ = dict(cohort=c, tissue=f.tissue.iloc[0], n_U=len(t), n_seeds_free=Zf.shape[1], n_seeds_can=Zc.shape[1], borrowed_var=c in RU.BORROW and Zc.shape[1] < 2,
                rho_zf_zc=rho(zf, zc), var_delta=vtot, mean_se2=float(np.mean(se ** 2)), reliability=1 - float(np.mean(se ** 2)) / vtot,
                seed_rho_free=float(np.nanmean([rho(Zf[:, i], Zf[:, j]) for i in range(Zf.shape[1]) for j in range(i + 1, Zf.shape[1])])) if Zf.shape[1] > 1 else np.nan,
                seed_rho_can=float(np.nanmean([rho(Zc[:, i], Zc[:, j]) for i in range(Zc.shape[1]) for j in range(i + 1, Zc.shape[1])])) if Zc.shape[1] > 1 else np.nan,
                n_up=int((t.call > 0).sum()), n_down=int((t.call < 0).sum()), share_called=float((t.call != 0).mean()),
                n_up_strict=int((t.call_strict > 0).sum()), n_down_strict=int((t.call_strict < 0).sum()),
                rho_delta_vs_Dform=rho(delta, t.z_free_D - t.z_can_D))
    summ["split_half_r"], summ["reliability_split"] = split_half(Zf, Zc, np.random.default_rng([RU.SEED, 7, sum(map(ord, c))]))
    return t, summ


def calibration(c: str, gate: str, rng) -> list[dict]:
    """Same estimator on two disjoint seed groups of one gate: no true gap, so calls are false calls."""
    f, k = universe(c)
    g = f if gate == "free" else k
    Z, _ = seed_z(g)
    n = Z.shape[1]
    dec = decile(g.expr.to_numpy(float))
    sizes = (2, 3) if n >= 5 else (2, 2)
    rows = []
    for s in range(RU.N_SPLITS):
        perm = rng.permutation(n)
        A, B = perm[: sizes[0]], perm[sizes[0]: sizes[0] + sizes[1]]
        G = gap_stats(Z[:, A], Z[:, B], dec)
        rows.append(dict(cohort=c, gate=gate, split=s, n_a=len(A), n_b=len(B), n_U=len(dec), n_calls=int((G["call"] != 0).sum()), rate=float((G["call"] != 0).mean()),
                         n_calls_strict=int((G["call_strict"] != 0).sum()), share_p_lt_05=float((G["p"] < 0.05).mean()),
                         reliability=1 - float(np.mean(G["se"] ** 2)) / float(np.var(G["delta"])),
                         reliability_split=split_half(Z[:, A], Z[:, B], rng, 5)[1]))
    return rows


# ---------------------------------------------------------------- covariates, neighborhoods, sequence


def covariates(t: pd.DataFrame) -> list[dict]:
    u = t.dropna(subset=RU.COVARIATES)
    rows = []
    for x in RU.COVARIATES:
        others = [u[o].to_numpy(float) for o in RU.COVARIATES if o != x]
        for y in ("delta", "z_free", "z_can"):
            rows.append(dict(cohort=t.cohort.iloc[0], readout=y, covariate=x, n=len(u), rho=rho(u[y], u[x]), partial_rho=partial_rho(u[y].to_numpy(), u[x].to_numpy(float), others)))
    Z = design([u[o].to_numpy(float) for o in RU.COVARIATES])
    for y in ("delta", "z_free", "z_can"):
        r = pd.Series(u[y].to_numpy()).rank().to_numpy()
        res = r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]
        rows.append(dict(cohort=t.cohort.iloc[0], readout=y, covariate="ALL_R2", n=len(u), rho=np.nan, partial_rho=1 - float(res.var() / r.var())))
    return rows


def delta_res(t: pd.DataFrame) -> np.ndarray:
    """delta's rank residual on all covariates (NaN where a covariate is missing)."""
    out = np.full(len(t), np.nan)
    ok = t[RU.COVARIATES].notna().all(1).to_numpy()
    Z = design([t.loc[ok, o].to_numpy(float) for o in RU.COVARIATES])
    out[ok] = resid_rank(t.delta.to_numpy()[ok], Z) / ok.sum()
    return out


def esm(c: str) -> pd.DataFrame:
    """64 ESM PCs and Leiden modules (as in T4_3_scoreboard) of every gene of the cohort's object (cached)."""
    import anndata as ad

    from analysis.tasks.T4_3_scoreboard import inputs as I

    f = OUT / "cache" / f"esm__{c}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    a = ad.read_h5ad(RU.cohorts()[c]["h5ad"], backed="r")
    F = np.asarray(a.varm["gene_features"], dtype=float)[:, : RU.N_PCS]
    vn = np.asarray(a.var_names)
    a.file.close()
    b, _ = I.esm_blocks(F)
    out = pd.DataFrame(F, index=pd.Index(vn, name="var_name"), columns=[f"esm{j}" for j in range(RU.N_PCS)])
    out["block"] = b
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f)
    return out


def neighbours(c: str, t: pd.DataFrame, rng) -> tuple[pd.DataFrame, dict]:
    from sklearn.neighbors import NearestNeighbors

    E = esm(c).reindex(t.index)
    F = E[[f"esm{j}" for j in range(RU.N_PCS)]].to_numpy(float)
    nn = NearestNeighbors(n_neighbors=RU.KNN + 1).fit(F)
    dist, idx = nn.kneighbors(F)
    idx, dist = idx[:, 1:], dist[:, 1:]
    zf, zc, dl = t.z_free.to_numpy(), t.z_can.to_numpy(), t.delta.to_numpy()
    m = zf[idx].mean(1)
    mc = zc[idx].mean(1)
    N = zf - m
    t = t.assign(nbr_mean_free=m, nbr_mean_can=mc, nbr_contrast=N, nbr_dist=dist.mean(1), block=E.block.to_numpy(),
                 nbr_symbols=[";".join(t.symbol.to_numpy()[r[:5]].astype(str)) for r in idx], nbr_zfree5=[";".join(f"{v:.2f}" for v in zf[r[:5]]) for r in idx])
    dec = decile(t.expr.to_numpy(float))
    bydec = [np.flatnonzero(dec == d) for d in range(RU.EXPR_DECILES)]
    null, null_c, null_n = [], [], []
    for _ in range(RU.N_NULL_NBR):  # random neighbors of the same expression decile
        R = np.empty_like(idx)
        for d, mem in enumerate(bydec):
            w = np.flatnonzero(dec == d)
            R[w] = rng.choice(mem, size=(len(w), RU.KNN))
        null.append(rho(zf, zf[R].mean(1)))
        null_c.append(rho(zc, zf[R].mean(1)))
        null_n.append(rho(dl, zf - zf[R].mean(1)))
    covs = [t[o].to_numpy(float) for o in RU.COVARIATES]
    ok = np.all([np.isfinite(x) for x in covs], 0)
    blk = t.block.to_numpy()

    def share(v):  # between-module share of variance, excess over random assignment (one-way ICC form)
        s = pd.Series(v)
        sb = (s.groupby(blk).transform("mean") - s.mean()).pow(2).sum()
        st = (s - s.mean()).pow(2).sum()
        nb = len(np.unique(blk))
        e = (nb - 1) / (len(v) - 1)
        return float(sb / st), float((sb / st - e) / (1 - e))

    sd, sde = share(dl)
    sf, sfe = share(zf)
    sc, sce = share(zc)
    r = dict(cohort=c, n=len(t), rho_zf_nbrf=rho(zf, m), null_mean=float(np.mean(null)), null_hi=float(np.percentile(null, 97.5)), rho_zc_nbrf=rho(zc, m),
             null_zc_nbrf_mean=float(np.mean(null_c)), null_zc_nbrf_hi=float(np.percentile(null_c, 97.5)), null_contrast_mean=float(np.mean(null_n)),
             null_contrast_hi=float(np.percentile(null_n, 97.5)),
             rho_zc_nbrc=rho(zc, mc), rho_zf_nbrc=rho(zf, mc), rho_delta_contrast=rho(dl, N),
             prho_delta_contrast=partial_rho(dl[ok], N[ok], [x[ok] for x in covs]), rho_delta_nbrdist=rho(np.abs(dl), dist.mean(1)),
             rho_absdelta_absN=rho(np.abs(dl), np.abs(N)), block_share_delta=sd, block_excess_delta=sde, block_share_zf=sf, block_excess_zf=sfe,
             block_share_zc=sc, block_excess_zc=sce, n_blocks=int(len(np.unique(blk))))
    return t, r


def sequence(t: pd.DataFrame, rng) -> list[dict]:
    rows = []
    for name, col, tr in (("log10_dNdS", "dNdS", lambda v: np.log10(np.clip(v, 1e-3, None))), ("LOEUF", "LOEUF", lambda v: v)):
        u = t[np.isfinite(t[col].to_numpy(float)) & (t[col].to_numpy(float) > 0 if col == "dNdS" else True)]
        u = u.dropna(subset=["pi_raw", "expr", "detection"])
        if len(u) < 200:
            continue
        x = tr(u[col].to_numpy(float))
        covs = np.c_[u.pi_raw.to_numpy(float), u.expr.to_numpy(float), u.detection.to_numpy(float)]
        blk = u.block.to_numpy()
        for y in ("delta", "z_free", "z_can", "delta_res"):
            v = u[y].to_numpy(float)
            okv = np.isfinite(v)
            vv, xx, cc, bb = v[okv], x[okv], covs[okv], blk[okv]
            r0 = rho(vv, xx)
            p0 = partial_rho(vv, xx, list(cc.T))
            lo, hi = block_boot(lambda i: rho(vv[i], xx[i]), len(vv), bb, RU.N_BOOT_SEQ, rng)
            plo, phi = block_boot(lambda i: partial_rho(vv[i], xx[i], list(cc[i].T)), len(vv), bb, RU.N_BOOT_SEQ // 4, rng)
            rows.append(dict(cohort=t.cohort.iloc[0], constraint=name, readout=y, n=len(vv), rho=r0, ci_lo=lo, ci_hi=hi, partial_rho=p0, p_ci_lo=plo, p_ci_hi=phi))
    return rows


def one(c: str) -> dict:
    rng = np.random.default_rng([RU.SEED, sum(map(ord, c))])
    t, summ = gap_table(c)
    t["delta_res"] = delta_res(t)
    t, nb = neighbours(c, t, rng)
    cov = covariates(t)
    seq = sequence(t, rng)
    (OUT / "genes").mkdir(parents=True, exist_ok=True)
    t.to_parquet(OUT / "genes" / f"{c}.parquet")
    print(c, f"n={summ['n_U']} rel={summ['reliability']:.2f} up={summ['n_up']} down={summ['n_down']}", flush=True)
    return dict(summ=summ, nb=nb, cov=cov, seq=seq)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cs = a.only or list(RU.cohorts())
    for c in cs:  # ESM caches first, serially (h5ad reads on the network file system)
        esm(c)
    with ProcessPoolExecutor(a.jobs, mp_context=__import__("multiprocessing").get_context("spawn")) as ex:
        res = list(ex.map(one, cs))
    pd.DataFrame([r["summ"] for r in res]).to_csv(OUT / "cohort_gap.csv", index=False)
    pd.DataFrame([r["nb"] for r in res]).to_csv(OUT / "neighbours.csv", index=False)
    pd.DataFrame([x for r in res for x in r["cov"]]).to_csv(OUT / "covariates.csv", index=False)
    pd.DataFrame([x for r in res for x in r["seq"]]).to_csv(OUT / "sequence.csv", index=False)
    rng = np.random.default_rng(RU.SEED)
    cal = [x for c, g in RU.CALIB for x in calibration(c, g, rng)]
    pd.DataFrame(cal).to_csv(OUT / "calibration.csv", index=False)
    pd.set_option("display.width", 250, "display.max_columns", 40)
    print(pd.DataFrame([r["summ"] for r in res]).round(3).to_string())
    print(pd.DataFrame(cal).groupby(["cohort", "gate"])[["rate", "n_calls", "share_p_lt_05", "reliability"]].mean().round(4).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
