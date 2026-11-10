"""H1 step 4: set z per cohort, recurrence across the six tissues with a within-cohort permutation null, tissue-specific sets, separate axis.

Definitions: rules.py. Reads results/analysis/H1/genes/. Writes results/analysis/H1/{sets_membership.csv, set_z.csv, recurrence.csv, tissue_specific.csv,
lineage_z.csv, axis.csv}.

    PYTHONPATH=<chimera repo> python analysis/tasks/H1_harmonise/recurrence.py [--jobs 10]
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm, spearmanr

from analysis.common import controls as K
from analysis.tasks.H1_harmonise import rules as RU
from analysis.tasks.X3_cross_cohort import annotations as AN

OUT = RU.OUT
sys.path.insert(0, str(RU.REPO / "notebooks"))
import _gene_categories_broad as GB  # noqa: E402

HALLMARK = __import__("pathlib").Path.home() / ".cache/gseapy/Enrichr.MSigDB_Hallmark_2020.gmt"
FAMILY = {"hallmark": "Hallmark", "class": "gene class", "broad": "first-submission category"}


def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q = p[ok]
        o = np.argsort(q)
        r = q[o] * len(q) / np.arange(1, len(q) + 1)
        r = np.minimum.accumulate(r[::-1])[::-1]
        v = np.empty_like(q)
        v[o] = np.minimum(r, 1)
        out[ok] = v
    return out


def read_hallmark() -> dict[str, set]:
    d = {}
    for ln in HALLMARK.read_text().splitlines():
        x = ln.split("\t")
        d[x[0]] = {g.upper() for g in x[2:] if g}
    return d


def gene_sets(genes: pd.DataFrame) -> pd.DataFrame:
    """Long table: family, set, hsapiens_gene_id, over the union of all cohorts' genes."""
    u = genes.dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id")[["hsapiens_gene_id", "symbol"]]
    u["sym"] = u.symbol.astype(str).str.upper()
    rows = []
    for name, mem in read_hallmark().items():
        for e in u.hsapiens_gene_id[u.sym.isin(mem)]:
            rows.append(("hallmark", name, e))
    cl = AN.gene_classes(pd.Index(u.hsapiens_gene_id))
    for k in RU.CLASSES:
        for e in cl.index[cl[k].to_numpy().astype(bool)]:
            rows.append(("class", k, e))
    cat = u.symbol.fillna("").map(lambda s: GB.classify_broad(s) if s else GB.classify_broad("")).to_numpy()
    for e, k in zip(u.hsapiens_gene_id, cat):
        if str(k).lower() != "other":
            rows.append(("broad", str(k), e))
    return pd.DataFrame(rows, columns=["family", "set", "hsapiens_gene_id"])


def lineage_sets(genes: pd.DataFrame) -> pd.DataFrame:
    u = genes.dropna(subset=["hsapiens_gene_id", "symbol"]).drop_duplicates("hsapiens_gene_id")
    m = dict(zip(u.symbol, u.hsapiens_gene_id))
    rows = []
    for t, groups in RU.LINEAGE.items():
        for gname, syms in groups.items():
            for s in syms:
                if s in m:
                    rows.append((t, gname, s, m[s]))
    return pd.DataFrame(rows, columns=["tissue", "set", "symbol", "hsapiens_gene_id"])


def inv_norm_rank(v: np.ndarray) -> np.ndarray:
    r = pd.Series(v).rank(method="average").to_numpy()
    return norm.ppf((r - 0.5) / len(r))


def strata(g: pd.DataFrame, with_pi: bool, pi_col: str = "pi_raw", with_tau: bool = False) -> np.ndarray:
    e = pd.qcut(g.expr.rank(method="first"), RU.EXPR_BINS, labels=False).to_numpy()
    if not with_pi:
        return e
    p = pd.Series(pd.qcut(g[pi_col].rank(method="average"), RU.PI_BINS, labels=False, duplicates="drop")).fillna(0).to_numpy().astype(int)
    out = e * 100 + p
    if with_tau:  # post hoc sensitivity: also matched on cell-type specificity, because the margin rewards having a pattern
        tq = pd.Series(pd.qcut(g.tau.fillna(g.tau.median()).rank(method="first"), RU.PI_BINS, labels=False)).to_numpy().astype(int)
        out = out * 10 + tq
    return out


def block_scores(res: np.ndarray, blocks: np.ndarray, M: sp.csr_matrix, n: int, rng) -> np.ndarray:
    """Post hoc null respecting protein neighbourhoods: genes ordered by ESM module, module order shuffled, values laid back over the same gene slots.
    Keeps most of the within-module correlation that a gene-level permutation breaks. res = values minus their stratum mean."""
    order = np.argsort(blocks, kind="stable")
    b = blocks[order]
    x = res[order]
    starts = np.r_[0, np.flatnonzero(np.diff(b)) + 1]
    ends = np.r_[starts[1:], len(b)]
    sizes = np.asarray(M.sum(1)).ravel()
    out = np.empty((n, M.shape[0]), dtype=np.float32)
    for i in range(n):
        pm = rng.permutation(len(starts))
        y = np.concatenate([x[starts[j]:ends[j]] for j in pm])
        val = np.empty(len(x), dtype=np.float32)
        val[order] = y
        out[i] = (M @ val) / np.maximum(sizes, 1)
    return out


def perm_scores(v: np.ndarray, st: np.ndarray, M: sp.csr_matrix, n: int, rng, batch: int = 200) -> np.ndarray:
    """Set means under permutation of v within strata: draws x sets."""
    out = np.empty((n, M.shape[0]), dtype=np.float32)
    sizes = np.asarray(M.sum(1)).ravel()
    order0 = np.argsort(st, kind="stable")
    st0 = st[order0]
    v0 = v[order0]
    starts = np.r_[0, np.flatnonzero(np.diff(st0)) + 1, len(st0)]
    for b in range(0, n, batch):
        m = min(batch, n - b)
        P = np.empty((len(v), m), dtype=np.float32)
        for j in range(m):
            x = v0.copy()
            for i in range(len(starts) - 1):
                a_, b_ = starts[i], starts[i + 1]
                x[a_:b_] = rng.permutation(x[a_:b_])
            P[order0, j] = x
        out[b:b + m] = (M @ P).T / np.maximum(sizes, 1)
    return out


def one(args):
    """Set z for one cohort x readout. Returns long table and the null z matrix (for the primaries)."""
    c, readout, membership, n_perm = args
    g = pd.read_parquet(OUT / "genes" / f"{c}.parquet").dropna(subset=["hsapiens_gene_id"]).drop_duplicates("hsapiens_gene_id")
    col = {"D": "D", "D_allgenes": "D", "D_raw": "D_raw", "D_hmid": "D", "S": "S_ema", "B1": "B1", "pi": "pi_raw", "dNdS": "dNdS", "D_integrated": None, "D_tau": "D", "S_tau": "S_ema", "D_block": "D", "S_block": "S_ema", "D_nosat": "D", "S_nosat": "S_ema"}[readout]
    if readout == "D_integrated":
        keep = [f"alpha_s{s}" for s in (42, 43, 44, 47, 48, 51) if f"alpha_s{s}" in g]
        x = g[keep].mean(1) - g["pi_clamped"]
        g = g.assign(D_integrated=x - x.median())
        col = "D_integrated"
    if readout == "dNdS":
        g = g[g.dNdS > 0].assign(dNdS=lambda d: np.log10(d.dNdS.clip(lower=1e-3)))
    u = g if readout == "D_allgenes" else g[g.count_support]
    u = u.dropna(subset=[col, "expr"])
    if readout.endswith("_nosat"):
        u = u[~u.saturated_flag]
    if readout == "D_hmid":
        u = u.dropna(subset=["identity_human_mouse"])
    st = strata(u, readout not in ("pi", "dNdS"), "identity_human_mouse" if readout == "D_hmid" else "pi_raw", with_tau=readout.endswith("_tau"))
    v = inv_norm_rank(u[col].to_numpy(float))
    ix = pd.Index(u.hsapiens_gene_id)
    sets = sorted(membership.key.unique())
    si = {s: i for i, s in enumerate(sets)}
    mm = membership[membership.hsapiens_gene_id.isin(ix)]
    M = sp.csr_matrix((np.ones(len(mm), np.float32), (mm.key.map(si).to_numpy(), ix.get_indexer(mm.hsapiens_gene_id))), shape=(len(sets), len(ix)))
    sizes = np.asarray(M.sum(1)).ravel()
    rng = np.random.default_rng([RU.SEED, sum(map(ord, c)) * 1000 + sum(map(ord, readout))])
    if readout.endswith("_block"):
        v = (v - pd.Series(v).groupby(st).transform("mean").to_numpy()).astype(np.float32)
        tissue = RU.cohorts()[c]["tissue"].replace(" ", "_")
        yf = RU.RES / "Y1" / "cache" / f"tissue__{tissue}.parquet"
        tb = pd.read_parquet(yf) if yf.exists() else pd.read_parquet(OUT / "cache" / f"blocks__{c}.parquet")  # cortex, lung: ESM modules of the cohort
        blk = pd.Series(tb.block.to_numpy(), index=tb.hsapiens_gene_id).groupby(level=0).first().reindex(ix)
        codes = pd.factorize(blk.fillna(-1).astype(str))[0]
        obs = np.asarray(M @ v).ravel() / np.maximum(sizes, 1)
        null = block_scores(v, codes, M, n_perm, rng)
    else:
        obs = np.asarray(M @ v).ravel() / np.maximum(sizes, 1)
        null = perm_scores(v.astype(np.float32), st, M, n_perm, rng)
    mu, sd = null.mean(0), null.std(0, ddof=1)
    sd = np.where(sd > 0, sd, np.nan)
    z = (obs - mu) / sd
    zn = (null - mu) / sd
    p = (1 + (np.abs(zn) >= np.abs(z)[None, :]).sum(0)) / (n_perm + 1)
    t = pd.DataFrame(dict(cohort=c, readout=readout, key=sets, n=sizes.astype(int), mean_value=obs, z=z, p=p))
    t.loc[t.n < t.key.map(lambda k: 5 if k.startswith("lineage|") else RU.MIN_SET), ["z", "p"]] = np.nan
    return t, zn.astype(np.float32), sets


def recurrence_table(Z: dict[str, tuple], tissues: list[str], sets: list[str], fam: dict, rmin: int = RU.R_MIN) -> tuple[pd.DataFrame, pd.DataFrame]:
    """R over the tissues for one readout, permutation p, BH; tissue-specific candidates."""
    zo = np.stack([Z[t][0] for t in tissues], 1)  # sets x tissues
    zn = np.stack([Z[t][1] for t in tissues], 2)  # draws x sets x tissues
    pv = np.stack([Z[t][2] for t in tissues], 1)
    qv = np.stack([bh(pv[:, j]) for j in range(pv.shape[1])], 1)

    def stat(a):
        return (np.nan_to_num(a, nan=0.0) >= RU.Z_CALL).sum(-1), (np.nan_to_num(a, nan=0.0) <= -RU.Z_CALL).sum(-1)

    up, dn = stat(zo)
    R = np.maximum(up, dn)
    nup, ndn = stat(zn)
    Rn = np.maximum(nup, ndn)
    prec = (1 + (Rn >= R[None, :]).sum(0)) / (zn.shape[0] + 1)
    qrec = bh(prec)
    rows = pd.DataFrame(dict(key=sets, family=[fam[s] for s in sets], n_tissues=np.isfinite(zo).sum(1), n_up=up, n_down=dn, R=R, direction=np.where(up > dn, "above", np.where(dn > up, "below", "mixed")),
                             p_rec=prec, q_rec=qrec, mean_z=np.nanmean(zo, 1), stouffer=np.nansum(zo, 1) / np.sqrt(np.maximum(np.isfinite(zo).sum(1), 1))))
    rows["recurrent"] = (rows.R >= rmin) & (rows.q_rec < 0.05)
    for j, t in enumerate(tissues):
        rows[f"z_{t}"] = zo[:, j]
        rows[f"q_{t}"] = qv[:, j]
    cand = []
    for i, s in enumerate(sets):
        call = (qv[i] < 0.05) & (np.abs(np.nan_to_num(zo[i])) >= RU.Z_CALL)
        if call.sum() != 1:
            continue
        j = int(np.argmax(call))
        others = np.delete(zo[i], j)
        if np.any(np.abs(np.nan_to_num(others)) >= RU.Z_CALL) or np.isfinite(others).sum() < 3:
            continue
        cobs = zo[i, j] - np.nanmean(others)
        cnull = zn[:, i, j] - np.nanmean(np.delete(zn[:, i, :], j, axis=1), axis=1)
        cand.append(dict(key=s, family=fam[s], tissue=tissues[j], z=zo[i, j], q_in_tissue=qv[i, j], contrast=cobs, p_contrast=(1 + (np.abs(cnull) >= abs(cobs)).sum()) / (len(cnull) + 1), max_abs_z_others=float(np.nanmax(np.abs(others)))))
    ts = pd.DataFrame(cand)
    if len(ts):
        ts["q_contrast"] = bh(ts.p_contrast)
        ts["tissue_specific"] = ts.q_contrast < 0.05
    return rows, ts


def partial_rho(a, b, covs):
    Z = np.c_[np.ones(len(a)), *[K.ns_basis(pd.Series(c).rank().to_numpy(), 4) for c in covs]]

    def res(y):
        r = pd.Series(y).rank().to_numpy()
        return r - Z @ np.linalg.lstsq(Z, r, rcond=None)[0]

    return float(np.corrcoef(res(a), res(b))[0, 1])


def axis_table(cohorts: dict) -> pd.DataFrame:
    """The separate-axis statement per cohort. Per-seed range for D. D on clamped recipes anchors on the clamped prior; D_raw beside it."""
    rows = []
    for c in cohorts:
        g = pd.read_parquet(OUT / "genes" / f"{c}.parquet")
        seeds = [k for k in g.columns if k.startswith("alpha_s")]
        g["logw"] = np.log10(g.dNdS.where(g.dNdS > 0).clip(lower=1e-3))
        d = g.dropna(subset=["logw", "D", "pi_raw"])
        for uni, sub in (("all", d), ("count_support", d[d.count_support])):
            r = dict(cohort=c, tissue=cohorts[c]["tissue"], universe=uni, n=len(sub))
            r["rho_pi_dNdS"] = spearmanr(sub.pi_raw, sub.logw)[0]
            for name, col in (("D", "D"), ("D_raw", "D_raw"), ("S", "S_ema"), ("alpha", "alpha"), ("B1", "B1")):
                s2 = sub.dropna(subset=[col])
                if len(s2) < 50:
                    continue
                r[f"rho_{name}_pi"] = spearmanr(s2[col], s2.pi_raw)[0]
                r[f"rho_{name}_dNdS"] = spearmanr(s2[col], s2.logw)[0]
                r[f"rho_{name}_expr"] = spearmanr(s2[col], s2.expr)[0]
            r["rho_D_hmid"] = spearmanr(sub.D, sub.identity_human_mouse, nan_policy="omit")[0]
            r["rhop_D_dNdS"] = partial_rho(sub.D.to_numpy(), sub.logw.to_numpy(), [sub.pi_raw.to_numpy(), sub.expr.to_numpy(), sub.detection.to_numpy()])
            r["rhop_S_dNdS"] = partial_rho(sub.S_ema.to_numpy(), sub.logw.to_numpy(), [sub.pi_raw.to_numpy(), sub.expr.to_numpy(), sub.detection.to_numpy()])
            r["rhop_D_pi"] = partial_rho(sub.D.to_numpy(), sub.pi_raw.to_numpy(), [sub.expr.to_numpy(), sub.detection.to_numpy()])
            ds = [sub[k] - sub["pi_clamped"] for k in seeds]
            pis = [spearmanr(x, sub.pi_raw)[0] for x in ds]
            w = [spearmanr(x, sub.logw)[0] for x in ds]
            r.update(rho_D_pi_seedmin=min(pis), rho_D_pi_seedmax=max(pis), rho_D_dNdS_seedmin=min(w), rho_D_dNdS_seedmax=max(w))
            rows.append(r)
    return pd.DataFrame(rows)


def ensure_blocks(c: str) -> None:
    """ESM modules for a thread tissue without a Y1 cache: T4.3 esm_blocks on the cohort's 64 ESM PCs, as every other tissue's blocks."""
    import anndata as ad

    from analysis.tasks.T4_3_scoreboard import inputs as I

    f = OUT / "cache" / f"blocks__{c}.parquet"
    if f.exists():
        return
    a = ad.read_h5ad(RU.cohorts()[c]["h5ad"], backed="r")
    F = np.asarray(a.varm["gene_features"], dtype=float)[:, :I.N_PCS]
    vn = np.asarray(a.var_names)
    a.file.close()
    g = pd.read_parquet(OUT / "genes" / f"{c}.parquet").reindex(vn)
    b, _ = I.esm_blocks(F)
    f.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(hsapiens_gene_id=g.hsapiens_gene_id.to_numpy(), block=b)).to_parquet(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--n-perm", type=int, default=RU.N_PERM)
    a = ap.parse_args()
    cohorts = RU.cohorts()
    allg = pd.concat([pd.read_parquet(OUT / "genes" / f"{c}.parquet")[["hsapiens_gene_id", "symbol"]] for c in cohorts])
    gs = gene_sets(allg)
    ls = lineage_sets(allg)
    gs["key"] = gs.family + "|" + gs.set
    lin = ls.assign(key="lineage|" + ls.tissue + "|" + ls.set)[["key", "hsapiens_gene_id"]]
    gs.to_csv(OUT / "sets_membership.csv", index=False)
    ls.to_csv(OUT / "lineage_membership.csv", index=False)
    mem = pd.concat([gs[["key", "hsapiens_gene_id"]], lin], ignore_index=True)
    primary = [RU.PRIMARY[t] for t in RU.TISSUE_ORDER]
    for t in RU.TISSUE_ORDER:
        if not (RU.RES / "Y1" / "cache" / f"tissue__{t.replace(' ', '_')}.parquet").exists():
            ensure_blocks(RU.PRIMARY[t])
    sens = list(RU.SENS_EXTRA.values())
    jobs = [(c, r, mem, a.n_perm) for c in primary for r in ("D", "S", "B1", "pi", "dNdS", "D_allgenes", "D_raw", "D_hmid")]
    jobs += [("liv_hmouse", "D_integrated", mem, a.n_perm)]
    jobs += [(c, r, mem, a.n_perm) for c in primary for r in ("D_tau", "S_tau", "D_nosat", "S_nosat")]
    jobs += [(c, r, mem, 2000) for c in primary for r in ("D_block", "S_block")]
    jobs += [(c, r, mem, a.n_perm) for c in sens for r in ("D", "S")]
    jobs += [(c, r, mem, 2000) for c in cohorts if c not in primary and c not in sens for r in ("D", "S")]
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, jobs))
    long = pd.concat([x[0] for x in res], ignore_index=True)
    long.to_csv(OUT / "set_z.csv", index=False)
    idx = {(j[0], j[1]): i for i, j in enumerate(jobs)}
    fam = {k: k.split("|")[0] for k in mem.key.unique()}
    allrec, allts = [], []
    tests = {k: k for k in ("D", "S", "B1", "pi", "dNdS", "D_allgenes", "D_raw", "D_hmid", "D_tau", "S_tau", "D_block", "S_block", "D_nosat", "S_nosat")}
    sets_all = res[0][2]
    keep = [s for s in sets_all if not s.startswith("lineage|")]
    ki = [sets_all.index(s) for s in keep]
    prim = {**RU.PRIMARY, **RU.SENS_EXTRA}
    runs_ = [(name, r, RU.TISSUE_ORDER, RU.R_MIN) for name, r in list(tests.items()) + [("D_integrated_liver", None)]]
    # The six-tissue thread of the first pass, beside it, so what changes can be read off one file; all of its tests.
    runs_ += [(f"six_{name}", r, RU.SIX, RU.R_MIN) for name, r in tests.items()]
    runs_ += [("D_r4", "D", RU.TISSUE_ORDER, RU.R_MIN_HALF), ("S_r4", "S", RU.TISSUE_ORDER, RU.R_MIN_HALF)]
    runs_ += [("D_with_muscle", "D", RU.TISSUE_ORDER + list(RU.SENS_EXTRA), RU.R_MIN), ("S_with_muscle", "S", RU.TISSUE_ORDER + list(RU.SENS_EXTRA), RU.R_MIN)]
    for name, r, tissues, rmin in runs_:
        Z = {}
        for t in tissues:
            c = prim[t]
            rr = "D_integrated" if (r is None and t == "liver") else ("D" if r is None else r)
            tab, zn, _ = res[idx[(c, rr)]]
            Z[t] = (tab.z.to_numpy()[ki], zn[:, ki], tab.p.to_numpy()[ki])
        rec, ts = recurrence_table(Z, tissues, keep, fam, rmin)
        rec.insert(0, "readout", name)
        allrec.append(rec)
        if len(ts):
            ts.insert(0, "readout", name)
            allts.append(ts)
    pd.concat(allrec, ignore_index=True).to_csv(OUT / "recurrence.csv", index=False)
    pd.concat(allts, ignore_index=True).to_csv(OUT / "tissue_specific.csv", index=False)
    lz = long[long.key.str.startswith("lineage|")].copy()
    lz.to_csv(OUT / "lineage_z.csv", index=False)
    axis_table(cohorts).to_csv(OUT / "axis.csv", index=False)
    rec = pd.concat(allrec, ignore_index=True)
    piv = rec.pivot(index="key", columns="readout", values="mean_z")
    sa = [dict(a="six_D", b="six_" + b, spearman_over_sets=float(spearmanr(piv["six_D"], piv["six_" + b], nan_policy="omit")[0])) for b in ("S", "B1", "pi", "dNdS")]
    sa += [dict(a="D", b=b, spearman_over_sets=float(spearmanr(piv["D"], piv[b], nan_policy="omit")[0])) for b in ("S", "B1", "pi", "dNdS", "D_tau", "D_block", "D_raw", "D_allgenes", "D_hmid")]
    sa += [dict(a="pi", b="dNdS", spearman_over_sets=float(spearmanr(piv["pi"], piv["dNdS"], nan_policy="omit")[0])), dict(a="S", b="B1", spearman_over_sets=float(spearmanr(piv["S"], piv["B1"], nan_policy="omit")[0]))]
    pd.DataFrame(sa).to_csv(OUT / "set_axis.csv", index=False)
    print(pd.DataFrame(sa).round(3).to_string())
    print(rec.groupby("readout").recurrent.sum())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
