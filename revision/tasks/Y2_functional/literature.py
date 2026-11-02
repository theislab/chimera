"""Y2 (EXPLORATORY) section (d): literature-curated human-mouse differences against readouts and baselines.

``literature_cases.tsv``: the curated list, written with no access to any readout. ``literature_expect.tsv``: each row encoded by hand
from the list's own direction text, committed before the first join, in the coarse labels of the cohort:
  expect: human>mouse or mouse>human in ``type`` (level, on_off); type_switch (peak type differs between species); stage_peak (peak among
  germ stages differs); zonation (signs in zon_h, zon_m: +1 pericentral, -1 periportal, 0 mid); sub_type (difference inside a coarse type,
  not visible at this resolution); uncodable.

Visible in the counts ("supported by these data"), fixed here before the join:
  level, on_off: x_Mouse - x_Human in that type, centred on the median over genes with detection >= 1% in both species (X4 definition), has the
  expected sign with |d| >= 1 (log2 units).
  type_switch, stage_peak: peak of the species' per-type expression differs (both species detected >= 1% in their own peak type).
  zonation: hepatocyte porto-central slope (X2 zone table, PC - PP): human and mouse slope have the expected signs (0 means |slope| < 0.5).
  sub_type, uncodable: not assessable.

Recovered by score X: X defined, and X at or below the 25th percentile of its 200 nearest genes in rank space of expression, detection, tau, pi. Genes with
S exactly 0 are masked: not callable for S and D, though D is still listed. B1b, B1c are called wherever defined.
Usage (from chimera/): PYTHONPATH=$PWD mamba run --no-capture-output -n chimera_env python -m revision.tasks.Y2_functional.literature
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

from revision.common import cohorts as CO
from revision.common import gene_ids
from revision.tasks.Y2_functional import rules as RU

REPO = gene_ids.REPO_ROOT
RES = REPO / "results/revision"
OUT = RES / "Y2"
HERE = REPO / "revision/tasks/Y2_functional"
LFC_MIN = 1.0
ZON_MIN = 0.5
SCORES = ["S", "D", "B1b_raw", "B1c_raw", "B1z_raw", "B2b", "Sdir", "Sdir_pattern", "Sdir_level"]
TYPE_COL = {"Macrophages": ("Macrophages", "Kupffer cells")}


def type_cols(t: str):
    return TYPE_COL.get(t, (t, t))


def percentile_table(g: pd.DataFrame, score: str, query: np.ndarray, pool: np.ndarray, k=200) -> np.ndarray:
    covs = ["log_cp10k_spmean", "detection_spmean", "tau_shared", "prior_raw"]
    ok = g[covs + [score]].notna().all(axis=1).to_numpy()
    pos = np.where(ok & pool)[0]
    out = np.full(len(g), np.nan)
    if len(pos) < k:
        return out
    Z = np.column_stack([rankdata(g[c].to_numpy(float)) / len(g) for c in covs])
    q = np.where(query & ok)[0]
    if len(q) == 0:
        return out
    _, nb = NearestNeighbors(n_neighbors=k).fit(Z[pos]).kneighbors(Z[q])
    v = g[score].to_numpy(float)
    out[q] = (v[pos][nb] < v[q][:, None]).mean(1)
    return out


def centred_d(g: pd.DataFrame, t: str) -> pd.Series:
    th, tm = type_cols(t)
    d = g[f"x_Mouse_{tm}"] - g[f"x_Human_{th}"]
    ok = (g[f"f_Human_{th}"] >= 0.01) & (g[f"f_Mouse_{tm}"] >= 0.01)
    return d - d[ok].median()


def main():
    cases = pd.read_csv(HERE / "literature_cases.tsv", sep="\t", dtype=str)
    ex = pd.read_csv(HERE / "literature_expect.tsv", sep="\t", dtype=str)
    cases = cases.merge(ex, on=["tissue", "human_symbol", "pmid"], how="left", validate="one_to_one")
    assert cases["expect"].notna().all(), cases[cases["expect"].isna()]
    hg = pd.read_csv(REPO / "data/revision/x3_annotations/hgnc_complete_set.txt", sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"]).dropna()
    s2e = dict(zip(hg.symbol, hg.ensembl_gene_id))
    cases["hsapiens_gene_id"] = cases.human_symbol.map(s2e)
    x2z = pd.read_csv(RES / "X2/zonation/gene_zonation_reference.csv")
    rows, raw_rows = [], []
    for tissue, c in RU.COHORTS.items():
        g = pd.read_parquet(RES / f"X4/genes/{c}.parquet")
        d = pd.read_parquet(RES / f"X4/cache/Sdir__{c}.parquet")[["var_name", "Sdir", "Sdir_pattern", "Sdir_level"]]
        g = g.merge(d, on="var_name", how="left")
        un = (g["S"].abs() >= 1e-6).to_numpy()
        tab = gene_ids.load_table(c)
        tt = cases[cases.tissue == tissue]
        idx = {e: i for i, e in enumerate(g.hsapiens_gene_id)}
        query = g.hsapiens_gene_id.isin(set(tt.hsapiens_gene_id.dropna())).to_numpy()
        pct = {}
        for s in SCORES:
            if s in g:
                pool = un if s in ("S", "D", "Sdir", "Sdir_pattern", "Sdir_level") else g[s].notna().to_numpy()
                pct[s] = percentile_table(g, s, query, pool)
        pct["abs_shift"] = percentile_table(g, "abs_shift", query, g["abs_shift"].notna().to_numpy())
        ad = CO.read_adata(c)
        for _, r in tt.iterrows():
            base = dict(tissue=tissue, human_symbol=r.human_symbol, mouse_symbol=r.mouse_symbol, pmid=r.pmid, kind=r.kind, expect=r.expect, type=r.get("type"), confidence=r.confidence, direction=r.direction)
            e = r.hsapiens_gene_id
            if not isinstance(e, str) or e not in idx:
                rows.append(base | dict(status="not_in_vocabulary"))
                continue
            i = idx[e]
            q = g.iloc[i]
            tb = tab[tab.hsapiens_gene_id == e]
            mname = str(tb.mmusculus_gene_name.iloc[0]) if len(tb) else ""
            b = base | dict(status="masked" if abs(q.S) < 1e-6 else "unmasked", mouse_in_table=mname, identity_ok=mname.lower() == str(r.mouse_symbol).lower(),
                            margin_support=bool(q.margin_support), S=q.S, D=q.D, D_raw=q.D_raw, B1b=q.get("B1b_raw"), B1c=q.get("B1c_raw"), B1z=q.get("B1z_raw"), abs_shift=q.get("abs_shift"), prior=q.prior_raw)
            for s, v in pct.items():
                b[f"pct_{s}"] = v[i]
            callable_c = abs(q.S) >= 1e-6
            for s, lab in (("S", "S"), ("D", "D"), ("B1b_raw", "B1b"), ("B1c_raw", "B1c"), ("B1z_raw", "B1z"), ("B2b", "B2b")):
                v = b.get(f"pct_{s}", np.nan)
                ok = np.isfinite(v) and (callable_c or s not in ("S", "D"))
                b[f"rec_{lab}"] = bool(v <= 0.25) if ok else np.nan
            types = sorted(k[len("x_Mouse_"):] for k in g.columns if k.startswith("x_Mouse_"))
            htypes = sorted(k[len("x_Human_"):] for k in g.columns if k.startswith("x_Human_"))
            hp = {t: q[f"x_Human_{t}"] for t in htypes}
            mp = {t: q[f"x_Mouse_{t}"] for t in types}
            for t in htypes:
                raw_rows.append(dict(tissue=tissue, gene=r.human_symbol, species="Human", cell_type=t, log2cpm=q[f"x_Human_{t}"], det=q[f"f_Human_{t}"]))
            for t in types:
                raw_rows.append(dict(tissue=tissue, gene=r.human_symbol, species="Mouse", cell_type=t, log2cpm=q[f"x_Mouse_{t}"], det=q[f"f_Mouse_{t}"]))
            # raw UMIs per species x type
            vs = ad.var_names.get_loc(q.var_name)
            X = ad.X[:, vs]
            x = np.asarray(X.todense()).ravel() if hasattr(X, "todense") else np.asarray(X).ravel()
            ob = pd.DataFrame(dict(sp=ad.obs["species"].astype(str).values, ct=ad.obs["cell_type"].astype(str).values, x=x))
            b["umi_human"], b["umi_mouse"] = int(ob.x[ob.sp == "Human"].sum()), int(ob.x[ob.sp == "Mouse"].sum())
            vis = np.nan
            if r.expect in ("human>mouse", "mouse>human"):
                dd = centred_d(g, r.type).iloc[i]
                b["d_type"] = dd
                vis = bool(abs(dd) >= LFC_MIN and ((dd > 0) == (r.expect == "mouse>human")))
            elif r.expect in ("type_switch", "stage_peak"):
                keep_h = {t: v for t, v in hp.items() if q[f"f_Human_{t}"] >= 0.01}
                keep_m = {t: v for t, v in mp.items() if q[f"f_Mouse_{t}"] >= 0.01}
                if keep_h and keep_m:
                    ph, pm = max(keep_h, key=keep_h.get), max(keep_m, key=keep_m.get)
                    b["peak_h"], b["peak_m"] = ph, pm
                    vis = TYPE_COL.get(ph, (ph, ph))[1] != pm
                else:
                    vis = False
            elif r.expect == "zonation":
                z = x2z[x2z.var_name == q.var_name]
                if len(z) and str(r.type) == "Hepatocytes":
                    sh, sm = float(z.obs_slope_Human.iloc[0]), float(z.obs_slope_Mouse.iloc[0])
                    b["slope_h"], b["slope_m"] = sh, sm

                    def agree(s, want):
                        want = float(want)
                        return abs(s) < ZON_MIN if want == 0 else np.sign(s) == want and abs(s) >= ZON_MIN

                    vis = bool(agree(sh, r.zon_h) and agree(sm, r.zon_m))
            b["visible"] = vis
            rows.append(b)
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "literature_table.csv", index=False)
    pd.DataFrame(raw_rows).drop_duplicates().to_csv(OUT / "literature_raw_pattern.csv", index=False)
    # one line per gene x tissue for rates: a gene counted once, a row for each kind
    u = res.drop_duplicates(["tissue", "human_symbol", "expect"])
    summ = []
    for keys, g in list(u.groupby(["tissue", "expect"])) + [((t, "all"), x) for t, x in u.groupby("tissue")]:
        b = dict(tissue=keys[0], expect=keys[1], n=len(g), not_in_vocab=int((g.status == "not_in_vocabulary").sum()), masked=int((g.status == "masked").sum()), visible=int(g["visible"].astype(float).sum()))
        for lab in ("S", "D", "B1b", "B1c", "B1z", "B2b"):
            c = g[f"rec_{lab}"].dropna().astype(float)
            b[f"{lab}_n"], b[f"{lab}_rec"] = len(c), c.sum()
            vv = g[g["visible"].astype(float) == 1][f"rec_{lab}"].dropna().astype(float)
            b[f"{lab}_vis_n"], b[f"{lab}_vis_rec"] = len(vv), vv.sum()
        summ.append(b)
    pd.DataFrame(summ).to_csv(OUT / "literature_summary.csv", index=False)
    print(pd.DataFrame(summ).T.to_string())


if __name__ == "__main__":
    main()
