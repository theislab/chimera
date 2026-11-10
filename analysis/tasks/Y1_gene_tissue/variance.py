"""(b): variance split on z, for S, D, B1 (and Sdir). Complete-case universe unless stated. Block CI on the gene share.

gene share = mean Pearson of z over tissue pairs. rel: S = cell-half reliability on the fixed model times seed reliability of the mean over
the tissue's seeds; B1 = cell-half reliability; D = seed reliability only (no cell-half version exists without retraining: noted, it flatters D).
noise = 1 - mean rel; gene x tissue = mean rel - gene share. Raw-value two-way ANOVA is reported for the tissue share as a statement on scale.

    python -I launch.py analysis.tasks.Y1_gene_tissue.variance
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from analysis.tasks.Y1_gene_tissue import prep as P
from analysis.tasks.Y1_gene_tissue import rules as RU

MAT, ORDER = P.MAT, RU.TISSUES
OUT = RU.OUT
N_BOOT = 10000


def load(sc, dn):
    return pd.read_parquet(MAT / f"z_{sc}__{dn}.parquet")


def gene_share(Z: pd.DataFrame, w=None) -> tuple[float, dict]:
    pairs = {}
    for a, b in itertools.combinations(Z.columns, 2):
        ok = Z[a].notna() & Z[b].notna()
        x, y = Z.loc[ok, a].to_numpy(), Z.loc[ok, b].to_numpy()
        if w is None:
            pairs[(a, b)] = float(np.corrcoef(x, y)[0, 1])
        else:
            wi = w[ok.to_numpy()]
            mx, my = np.average(x, weights=wi), np.average(y, weights=wi)
            c = np.average((x - mx) * (y - my), weights=wi)
            pairs[(a, b)] = float(c / np.sqrt(np.average((x - mx) ** 2, weights=wi) * np.average((y - my) ** 2, weights=wi)))
    return float(np.mean(list(pairs.values()))), pairs


def block_ci(Z: pd.DataFrame, blocks: pd.Series, rng) -> tuple[float, float]:
    u, inv = np.unique(blocks.reindex(Z.index).to_numpy(), return_inverse=True)
    out = []
    for _ in range(N_BOOT):
        cnt = np.bincount(rng.integers(0, len(u), len(u)), minlength=len(u)).astype(float)
        w = cnt[inv]
        keep = w > 0
        out.append(gene_share(Z[keep], w[keep])[0])
    return float(np.quantile(out, 0.025)), float(np.quantile(out, 0.975))


def half_rel(a: pd.Series, b: pd.Series, cov_t: pd.DataFrame, cols) -> float:
    za, zb = P.z_on(a, cov_t, cols), P.z_on(b, cov_t, cols)
    i = za.index.intersection(zb.index)
    r = float(np.corrcoef(za[i], zb[i])[0, 1])
    return 2 * r / (1 + r), r


def seed_rel(M: pd.DataFrame, cov_t: pd.DataFrame, cols) -> tuple[float, float]:
    """Reliability of the mean over k seeds: Spearman-Brown of the mean pairwise r of z, k = number of seeds."""
    zs = {c: P.z_on(M[c], cov_t, cols) for c in M.columns}
    rs = [float(np.corrcoef(zs[a], zs[b])[0, 1]) for a, b in itertools.combinations(M.columns, 2)]
    r, k = float(np.mean(rs)), M.shape[1]
    return k * r / (1 + (k - 1) * r), r


def esm_free(Zc: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for t in Zc.columns:
        F = pd.read_parquet(MAT / f"esm__{t}.parquet").reindex(Zc.index).to_numpy(float)
        y = Zc[t].to_numpy(float)
        pred = np.zeros(len(y))
        for tr, te in KFold(5, shuffle=True, random_state=0).split(F):
            pred[te] = RidgeCV(alphas=np.logspace(-2, 5, 29)).fit(F[tr], y[tr]).predict(F[te])
        e = y - pred
        out[t] = (e - e.mean()) / e.std()
    return pd.DataFrame(out, index=Zc.index)


def main():
    sup = pd.read_parquet(MAT / "support.parquet")
    b1 = pd.read_parquet(MAT / "raw_B1.parquet")
    cc = sup.all(axis=1) & b1.notna().all(axis=1)
    genes = cc[cc].index
    blocks = pd.read_parquet(MAT / "blocks.parquet")["block"]
    cov = {c: pd.read_parquet(MAT / f"cov_{c}.parquet") for c in ("expr", "det", "tau", "prior_raw")}
    rng = np.random.default_rng(RU.SEED)
    rows, prs = [], []
    # z recomputed on the complete-case genes, one universe for all scores and designs
    raw = {sc: pd.read_parquet(MAT / f"raw_{sc}.parquet") for sc in ("S", "D", "D_raw", "B1", "Sdir", "Sdir_pattern", "Sdir_level")}
    hS = {k: pd.read_parquet(MAT / f"raw_{k}.parquet") for k in ("S_h0", "S_h1")}
    h_b1 = {t: np.load(RU.OUT / "cache" / f"b1_halves__{t}.npz") for t in ORDER}
    from analysis.tasks.Y1_gene_tissue import common as Y

    rel = {}
    for t in ORDER:
        cvt = pd.DataFrame({c: cov[c][t] for c in cov}).reindex(genes)
        ids = Y.tissue_table(t).set_index("var_name")["hsapiens_gene_id"]
        for dn, cols in P.DESIGNS.items():
            relS, rS = half_rel(hS["S_h0"][t].reindex(genes), hS["S_h1"][t].reindex(genes), cvt, cols)
            sm = pd.read_parquet(MAT / f"seedS__{t}.parquet").reindex(genes)
            dm = pd.read_parquet(MAT / f"seedD__{t}.parquet").reindex(genes)
            rsS, rsS1 = seed_rel(sm, cvt, cols)
            rsD, rsD1 = seed_rel(dm, cvt, cols)
            hb = h_b1[t]
            vn = ids.reindex(hb["var_names"].astype(str)).to_numpy()
            rb = []
            for A, B in zip(hb["a"], hb["b"]):
                a, b = pd.Series(A, index=vn).reindex(genes), pd.Series(B, index=vn).reindex(genes)
                ok = a.notna() & b.notna()
                rb.append(half_rel(a[ok], b[ok], cvt.loc[ok], cols)[1])
            rB1 = float(np.mean(rb))
            relB1 = 2 * rB1 / (1 + rB1)
            rel[(t, dn)] = dict(tissue=t, design=dn, n=len(genes), rel_cell_S=relS, rel_seed_S=rsS, seed_r_S=rsS1, rel_S=relS * rsS,
                                rel_seed_D=rsD, seed_r_D=rsD1, rel_cell_B1=relB1, n_seeds=sm.shape[1])
    relT = pd.DataFrame(rel.values())
    relT.to_csv(OUT / "reliability.csv", index=False)
    for dn, cols in P.DESIGNS.items():
        Zs = {}
        for sc in raw:
            if sc == "D_raw" and dn != "EDpi":
                continue
            Zs[sc] = pd.DataFrame({t: P.z_on(raw[sc][t].reindex(genes), pd.DataFrame({c: cov[c][t] for c in cols}).reindex(genes), cols) for t in ORDER})
        if "D" in Zs:
            Zs["D_ESMfree"] = esm_free(Zs["D"])
            Zs["S_ESMfree"] = esm_free(Zs["S"])
            Zs["B1_ESMfree"] = esm_free(Zs["B1"])
        for sc, Z in Zs.items():
            gs, pr = gene_share(Z)
            lo, hi = block_ci(Z, blocks, rng)
            base = sc.split("_")[0]
            rr = relT[relT.design == dn].set_index("tissue")
            if base == "S" and sc != "Sdir":
                relv = rr["rel_S"].to_numpy()
            elif sc in ("D", "D_raw", "D_ESMfree"):
                relv = rr["rel_seed_D"].to_numpy()
            elif base == "B1":
                relv = rr["rel_cell_B1"].to_numpy()
            else:
                relv = np.full(6, np.nan)
            mrel = float(np.mean(relv))
            dis = [pr[(a, b)] / np.sqrt(rr.loc[a, {"S": "rel_S", "D": "rel_seed_D", "B1": "rel_cell_B1"}.get(base, "rel_S")] *
                                        rr.loc[b, {"S": "rel_S", "D": "rel_seed_D", "B1": "rel_cell_B1"}.get(base, "rel_S")]) for a, b in pr]
            rows.append(dict(score=sc, design=dn, n=len(genes), gene_share=gs, gene_lo=lo, gene_hi=hi, mean_rel=mrel,
                             gxt_share=mrel - gs if np.isfinite(mrel) else np.nan, noise_share=1 - mrel if np.isfinite(mrel) else np.nan,
                             gxt_of_reliable=(mrel - gs) / mrel if np.isfinite(mrel) else np.nan, disattenuated_gene=float(np.mean(dis))))
            for (a, b), v in pr.items():
                prs.append(dict(score=sc, design=dn, a=a, b=b, r=v))
    pd.DataFrame(rows).to_csv(OUT / "variance_split.csv", index=False)
    pd.DataFrame(prs).to_csv(OUT / "variance_pairs.csv", index=False)
    # two-way ANOVA on raw values, tissue share as a statement on scale: complete cases, values standardized over the whole matrix
    an = []
    for sc in ("S", "D", "B1"):
        v = raw[sc].reindex(genes)
        v = (v - np.nanmean(v.values)) / np.nanstd(v.values)
        tot = np.nansum(v.values ** 2)
        tis = (v.mean(0) ** 2).sum() * len(v)
        gen = (v.mean(1) ** 2).sum() * v.shape[1]
        an.append(dict(score=sc, tissue_share_raw=tis / tot, gene_share_raw=gen / tot, residual_raw=1 - (tis + gen) / tot))
    pd.DataFrame(an).to_csv(OUT / "anova_raw.csv", index=False)
    # own-universe gene shares: every count-supported pair (S, D), B1 on its own defined genes
    own = []
    for sc, f in (("S", "z_S__ED"), ("D", "z_D__ED"), ("B1 on support", "z_B1__ED"), ("B1 own genes", "z_B1own__ED")):
        Z = pd.read_parquet(MAT / f"{f}.parquet")
        own.append(dict(score=sc, gene_share=gene_share(Z)[0], n_genes=int(Z.notna().sum(axis=1).ge(2).sum())))
    pd.DataFrame(own).to_csv(OUT / "variance_own_universe.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
