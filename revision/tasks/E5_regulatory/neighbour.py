"""E5 neighbourhood co-deployment (EXPLORATORY). Definitions: rules.py (Neighbourhood).

Usage (from chimera/):  mamba run -n chimera_env python -m revision.tasks.E5_regulatory.neighbour --workers 13
Reads gene_resid.parquet, genes_pos.parquet, paralogs_any.parquet (E5), ENCODE contact domains (data/revision/E5/tads, read as
data), H1 tables (detection per species for the level-gap read-out). Writes neighbour.csv, neighbour_tad.csv, pairs_tad.parquet.
Read-outs: C-residual D_can, D_free, S, B1; C+tau residual D_free; and a level-gap proxy (SD over species of logit detection,
C-residual) to ask whether co-deployment is shared pattern or shared level gaps.
"""
from __future__ import annotations

import argparse
import gzip
import os
import zlib

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from revision.common import gene_ids  # noqa: E402
from revision.tasks.E2_protein import analyse as E2  # noqa: E402
from revision.tasks.E5_regulatory import rules as RU  # noqa: E402

REPO = gene_ids.REPO_ROOT
OUT = REPO / "results" / "revision" / "E5"
TADS = REPO / "data" / "revision" / "E5" / "tads"
RCOLS = ["r_D_can", "r_D_free", "r_S", "r_B1", "rt_D_free", "r_level"]
BIN_NAMES = ["<10kb", "10-50kb", "50-200kb", "200-1000kb"]


def all_pairs() -> pd.DataFrame:
    """Protein-coding canonical-TSS pairs within 1 Mb, paralog and overlap flags, TAD labels (cohort independent)."""
    pos = pd.read_parquet(OUT / "genes_pos.parquet")
    pos = pos[pos.biotype == "protein_coding"]
    par = pd.read_parquet(OUT / "paralogs_any.parquet")
    parset = set(zip(par.g, par.h)) | set(zip(par.h, par.g))
    rows = []
    for c, d in pos.groupby("chrom"):
        d = d.sort_values("tss")
        ids, v, st, gs, ge = d.index.to_numpy(), d.tss.to_numpy(), d.strand.to_numpy(), d.gstart.to_numpy(), d.gend.to_numpy()
        for i in range(len(d)):
            j1 = np.searchsorted(v, v[i] + RU.DIST_BINS[-1], "right")
            for j in range(i + 1, j1):
                ov = gs[i] < ge[j] and gs[j] < ge[i]
                bid = st[i] != st[j] and v[j] - v[i] <= RU.BIDIR_MAX and st[i] == "-" and st[j] == "+"
                rows.append((ids[i], ids[j], c, int(v[i]), int(v[j]), int(v[j] - v[i]), ov, bid, (ids[i], ids[j]) in parset,
                             j == i + 1))
    P = pd.DataFrame(rows, columns=["g1", "g2", "chrom", "t1", "t2", "dist", "overlap", "bidir", "paralog", "adjacent_pc"])
    # TAD membership per biosample: a pair is inside one domain if some intra-chromosomal domain spans both TSSs
    same = np.zeros(len(P), int)
    inside1 = np.zeros(len(P), int)
    inside2 = np.zeros(len(P), int)
    nbio = 0
    for f in sorted(TADS.glob("*.bedpe.gz")):
        nbio += 1
        dm = pd.read_csv(f, sep="\t", comment="#", header=None, usecols=[0, 1, 2, 3], names=["c", "a", "b", "c2"])
        dm = dm[dm.c == dm.c2]
        dm["c"] = dm.c.astype(str).str.removeprefix("chr")
        for c, d in dm.groupby("c"):
            m = (P.chrom == c).to_numpy()
            if not m.any():
                continue
            a, b = d.a.to_numpy(), d.b.to_numpy()
            t1, t2 = P.t1.to_numpy()[m], P.t2.to_numpy()[m]
            # chunked containment test
            s_, i1_, i2_ = np.zeros(m.sum(), bool), np.zeros(m.sum(), bool), np.zeros(m.sum(), bool)
            for k in range(0, len(a), 500):
                aa, bb = a[k:k + 500, None], b[k:k + 500, None]
                in1 = (aa <= t1) & (t1 < bb)
                in2 = (aa <= t2) & (t2 < bb)
                s_ |= (in1 & in2).any(0)
                i1_ |= in1.any(0)
                i2_ |= in2.any(0)
            same[m] += s_
            inside1[m] += i1_
            inside2[m] += i2_
    P["n_same_tad"] = same
    P["tad_class"] = np.where(same >= RU.TAD_MIN_BIOSAMPLES, "same",
                              np.where((same == 0) & (inside1 >= RU.TAD_MIN_BIOSAMPLES) & (inside2 >= RU.TAD_MIN_BIOSAMPLES),
                                       "cross", "other"))
    P["n_biosamples"] = nbio
    return P


def level_proxy(c: str) -> pd.Series:
    h = pd.read_parquet(E2.H1G / f"{c}.parquet")
    det = h[[k for k in h.columns if k.startswith("detection_")]].clip(1e-4, 1 - 1e-4)
    lg = np.log(det / (1 - det))
    return pd.Series(lg.std(axis=1).to_numpy(), index=h.hsapiens_gene_id)


def sym_rho(a: np.ndarray, b: np.ndarray) -> float:
    """Pair-order-free rank correlation: each pair entered in both orders."""
    m = np.isfinite(a) & np.isfinite(b)
    x, y = np.r_[a[m], b[m]], np.r_[b[m], a[m]]
    if len(x) < 20:
        return np.nan
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def run_cohort(c: str) -> dict:
    G = pd.read_parquet(OUT / "gene_resid.parquet")
    G = G[G.cohort == c].set_index("hsapiens_gene_id")
    tissue = G.tissue.iloc[0]
    lv = level_proxy(c).reindex(G.index)
    # level proxy residualised on C like the read-outs
    U = pd.read_parquet(E2.H1G / f"{c}.parquet").set_index("hsapiens_gene_id").reindex(G.index)
    e2 = pd.read_parquet(REPO / "results/revision/E2/props.parquet", columns=["hsapiens_gene_id", "log_length"])
    U = U.join(e2.set_index("hsapiens_gene_id"))
    U["lvl"] = lv
    ok = U[E2.C + ["lvl"]].notna().all(1).to_numpy()
    G["r_level"] = np.nan
    G.loc[ok, "r_level"] = E2.resid_std(U[ok].reset_index(drop=True), "lvl", E2.C)
    P = pd.read_parquet(OUT / "pairs_tad.parquet")
    P = P[P.g1.isin(G.index) & P.g2.isin(G.index)].reset_index(drop=True)
    clean = ~P.paralog & ~P.overlap
    vals = {k: (G[k].reindex(P.g1).to_numpy(), G[k].reindex(P.g2).to_numpy()) for k in RCOLS}
    rng = np.random.default_rng(zlib.crc32(("E5n" + c).encode()))
    chrom_of = pd.read_parquet(OUT / "genes_pos.parquet").chrom.reindex(G.index).to_numpy()
    gidx = pd.Series(np.arange(len(G)), index=G.index)
    rows = []

    def add(name: str, m: np.ndarray):
        if m.sum() < 30:
            return
        i1, i2 = gidx[P.g1[m]].to_numpy(), gidx[P.g2[m]].to_numpy()
        r = dict(cohort=c, tissue=tissue, set=name, n_pairs=int(m.sum()))
        # null: second gene replaced by a random universe gene on another chromosome
        draws = []
        for _ in range(RU.NULL_DRAWS):
            j = rng.integers(0, len(G), len(i2))
            bad = chrom_of[j] == chrom_of[i1]
            while bad.any():
                j[bad] = rng.integers(0, len(G), bad.sum())
                bad = chrom_of[j] == chrom_of[i1]
            draws.append(j)
        for k in RCOLS:
            a, b = vals[k][0][m], vals[k][1][m]
            r[k] = sym_rho(a, b)
            v = G[k].to_numpy()
            null = np.array([sym_rho(v[i1], v[j]) for j in draws[:50 if k != "r_D_free" else RU.NULL_DRAWS]])
            r[f"{k}_null_mean"], r[f"{k}_null_sd"] = float(np.nanmean(null)), float(np.nanstd(null))
            r[f"{k}_z"] = (r[k] - r[f"{k}_null_mean"]) / r[f"{k}_null_sd"] if r[f"{k}_null_sd"] > 0 else np.nan
        rows.append(r)

    d = P.dist.to_numpy()
    for lo, hi, nm in zip(RU.DIST_BINS[:-1], RU.DIST_BINS[1:], BIN_NAMES):
        add(f"dist {nm}", clean.to_numpy() & (d >= lo) & (d < hi))
    add("bidirectional", clean.to_numpy() & P.bidir.to_numpy())
    add("paralog pairs <1Mb", P.paralog.to_numpy())
    # adjacent universe genes (consecutive in this cohort's universe along the chromosome), clean
    adj = P.sort_values(["g1", "dist"]).groupby("g1").head(1).index.to_numpy()
    madj = np.zeros(len(P), bool)
    madj[adj] = True
    add("adjacent", madj & clean.to_numpy())
    for lo, hi in RU.TAD_BINS:
        for cls in ("same", "cross"):
            add(f"TAD {cls} {lo // 1000}-{hi // 1000}kb", clean.to_numpy() & (d >= lo) & (d < hi) & (P.tad_class == cls).to_numpy())
    return dict(rows=pd.DataFrame(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=13)
    a = ap.parse_args()
    if not (OUT / "pairs_tad.parquet").exists():
        P = all_pairs()
        P.to_parquet(OUT / "pairs_tad.parquet")
        print(P.tad_class.value_counts(), P.bidir.sum(), P.paralog.sum())
    cs = E2.cohorts()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(run_cohort, cs))
    N = pd.concat([r["rows"] for r in res], ignore_index=True)
    N.to_csv(OUT / "neighbour.csv", index=False)
    s = N.groupby(["set", "tissue"])[RCOLS].median().groupby("set").median()
    s["n_tissue_pos_free"] = N.groupby(["set", "tissue"]).r_D_free.median().gt(0).groupby("set").sum()
    s["n_cohort_z2_free"] = N.assign(x=N.r_D_free_z > 2).groupby("set").x.sum()
    s.to_csv(OUT / "neighbour_summary.csv")
    pd.set_option("display.width", 250)
    print(s.round(3).to_string())


if __name__ == "__main__":
    main()
