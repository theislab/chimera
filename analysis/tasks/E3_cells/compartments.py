"""E3 analyses A and B: home compartments, matched excess on both read-outs and direct baselines, and where
each group's margin sits by cell type and species. Definitions in ``rules.py``.

    PYTHONPATH=$PWD python analysis/tasks/E3_cells/compartments.py [--jobs 8]

Outputs in ``results/analysis/E3/``: ``genes/<c>.parquet`` (per gene: home type, compartment, subgroup, readouts,
direct level gap), ``excess.csv``, ``attribution.csv``, ``species_attribution.csv``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

from analysis.tasks.E3_cells import rules as R
from analysis.tasks.H1_harmonise import rules as RU

OUT = R.OUT
SCORES = ["D", "D_free", "S_ema", "S_free", "B1", "DL"]


def load_build(c: str) -> dict:
    with np.load(OUT / "build" / f"{c}.npz", allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def human_index(b: dict) -> int:
    sp = [str(s) for s in b["species"]]
    return [i for i, s in enumerate(sp) if s.lower() == "human"][0]


def gene_frame(c: str) -> pd.DataFrame:
    d = RU.cohorts()[c]
    tmap = R.type_map(d["tissue"])
    g = pd.read_parquet(RU.OUT / "genes" / f"{c}.parquet")
    z = pd.read_parquet(R.RES / "Z4" / "h1" / "genes" / f"{c}.parquet").reindex(g.index)
    g["D_free"], g["S_free"] = z["D"], z["S_ema"]
    g["count_support_free"], g["nameable_free"], g["saturated_free"] = z["count_support"], z["nameable"], z["saturated_flag"]
    g["sign_D_stable_free"] = z["sign_D_stable"]
    b = load_build(c)
    vn = pd.Index(b["var_names"])
    cols = vn.get_indexer(g.index)
    assert (cols >= 0).all()
    hi = human_index(b)
    sp_idx, labels, n = b["species_idx"], b["labels"].astype(str), b["n"]
    x = np.log2(1 + b["cp10k"][:, cols] * 100)  # log2(1 + CPM)
    det = b["det"][:, cols]
    hrows = np.where((sp_idx == hi) & (n >= R.MIN_HUMAN_CELLS))[0]
    H = x[hrows].T
    pk = H.argmax(1)
    spec = H.max(1) - H.mean(1)
    fpk = det[hrows].T[np.arange(len(g)), pk]
    home = np.where((spec >= R.SPEC_MIN) & (fpk >= R.DET_MIN), labels[hrows][pk], "broad")
    g["home"] = home
    g["home_spec"] = spec
    g["compartment"] = [tmap[h][0] if h != "broad" else "broad" for h in home]
    g["subgroup"] = [tmap[h][1] if h != "broad" else "broad" for h in home]
    # direct per-type level gap at the gene's human peak type among types shared (>= 30 cells) with each other species
    others = [s for s in range(len(b["species"])) if s != hi]
    gaps = []
    order = np.argsort(-H, axis=1)
    for s in others:
        orow = {l: j for j, (si, l, nn) in enumerate(zip(sp_idx, labels, n)) if si == s and nn >= R.MIN_TYPE_CELLS}
        shared = np.array([labels[r] in orow for r in hrows])
        # first shared type in the human order
        first = np.full(len(g), -1)
        for k in range(order.shape[1]):
            cand = order[:, k]
            ok = (first < 0) & shared[cand]
            first[ok] = cand[ok]
        gap = np.full(len(g), np.nan)
        m = first >= 0
        hr = hrows[first[m]]
        orr = np.array([orow[labels[r]] for r in hr])
        gap[m] = np.abs(x[orr, np.where(m)[0]] - x[hr, np.where(m)[0]])
        gaps.append(gap)
    g["lfc_peak_abs"] = np.nanmean(np.column_stack(gaps), axis=1)
    g["DL"] = -g["lfc_peak_abs"]
    return g


def knn_excess(v, Zr, flag, B, rng) -> dict:
    ok = np.isfinite(v) & np.isfinite(Zr).all(1)
    f = flag[ok]
    if f.sum() < 10 or (~f).sum() < 200:
        return dict(n=int(f.sum()), excess_d=np.nan, p=np.nan, mean_set=np.nan, mean_null=np.nan)
    Z = np.column_stack([rankdata(Zr[ok, j]) / ok.sum() for j in range(Zr.shape[1])])
    vv = v[ok]
    pool = np.where(~f)[0]
    _, idx = NearestNeighbors(n_neighbors=R.K_NN).fit(Z[pool]).kneighbors(Z[f])
    nb = vv[pool[idx]]
    draws = nb[np.arange(len(nb))[None, :], rng.integers(0, R.K_NN, (B, len(nb)))].mean(1)
    obs = vv[f].mean()
    c0 = draws.mean()
    p = (1 + np.sum(np.abs(draws - c0) >= abs(obs - c0))) / (B + 1)
    return dict(n=int(f.sum()), mean_set=obs, mean_null=c0, excess_d=(obs - c0) / vv.std(), p=p)


def one(c: str):
    d = RU.cohorts()[c]
    g = gene_frame(c)
    (OUT / "genes").mkdir(parents=True, exist_ok=True)
    g.to_parquet(OUT / "genes" / f"{c}.parquet")
    rng = np.random.default_rng(R.SEED)
    rows = []
    for universe in ("count_supported", "all"):
        u = g.count_support.to_numpy() if universe == "count_supported" else np.ones(len(g), bool)
        gu = g[u]
        Zr = gu[R.COV_PAT].to_numpy(float)
        for level in ("compartment", "subgroup", "home"):
            for grp in sorted(gu[level].unique()):
                flag = (gu[level] == grp).to_numpy()
                for sc in SCORES:
                    r = knn_excess(gu[sc].to_numpy(float), Zr, flag, R.B_DRAWS, rng)
                    rows.append(dict(cohort=c, tissue=d["tissue"], universe=universe, level=level, group=grp, score=sc, **r))
    ex = pd.DataFrame(rows)
    # attribution (canonical forward passes, count-supported genes): positive and negative mass by type / weight
    b = load_build(c)
    vn = pd.Index(b["var_names"])
    att, spa = [], []
    for gate, key in (("canonical", "S_can"), ("per-gene", "S_free")):
        S = b[key]
        if np.isnan(S).all():
            continue
        Cgsk = S * b["w"][:, None]
        labels, sp_idx = b["labels"].astype(str), b["species_idx"]
        types = sorted(set(labels))
        wk = {k: b["w"][labels == k].sum() for k in types}
        sup = g.count_support.to_numpy() if gate == "canonical" else g.count_support_free.fillna(False).to_numpy()
        for level in ("compartment", "subgroup"):
            for grp in sorted(g[level].unique()):
                sel = vn.get_indexer(g.index[(g[level] == grp).to_numpy() & sup])
                if len(sel) < 10:
                    continue
                A = Cgsk[:, sel]
                pos, neg = np.clip(A, 0, None), -np.clip(A, None, 0)
                tp, tn = pos.sum(), neg.sum()
                for k in types:
                    m = labels == k
                    att.append(dict(cohort=c, gate=gate, level=level, group=grp, n=len(sel), cell_type=k, weight=wk[k],
                                    compartment_type=R.type_map(d["tissue"]).get(k, ("?", "?"))[0],
                                    share_pos=pos[m].sum() / tp, share_neg=neg[m].sum() / tn,
                                    enrich_pos=pos[m].sum() / tp / wk[k], enrich_neg=neg[m].sum() / tn / wk[k],
                                    net=A[m].sum() / len(sel)))
                for si, s in enumerate(b["species"]):
                    m = sp_idx == si
                    spa.append(dict(cohort=c, gate=gate, level=level, group=grp, n=len(sel), species=str(s),
                                    share_pos=pos[m].sum() / tp, share_neg=neg[m].sum() / tn, net=A[m].sum() / len(sel)))
    print(c, "done", flush=True)
    return ex, pd.DataFrame(att), pd.DataFrame(spa)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cohort", default=None)
    a = ap.parse_args()
    cs = [a.cohort] if a.cohort else list(RU.cohorts())
    with ProcessPoolExecutor(a.jobs) as ex:
        res = list(ex.map(one, cs))
    sfx = "" if not a.cohort else f"__{a.cohort}"
    pd.concat([r[0] for r in res]).to_csv(OUT / f"excess{sfx}.csv", index=False)
    pd.concat([r[1] for r in res]).to_csv(OUT / f"attribution{sfx}.csv", index=False)
    pd.concat([r[2] for r in res]).to_csv(OUT / f"species_attribution{sfx}.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
