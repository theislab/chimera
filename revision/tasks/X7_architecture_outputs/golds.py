"""X7 (EXPLORATORY): independent per-type and gene-by-type divergence, and the Murat tier gold for Lead B.

Definitions in rules.py. Reads the Murat held-out species h5ads, the T4.1 ortholog map and the T4.3 H3
profiles; no Chimera readout is read here. Writes results/revision/X7/golds/:
    murat_type_divergence.csv  per species pair and coarse type: 1 - Spearman across genes, n genes, cells,
                               and a split-library noise-adjusted version where each species has >= 2 libraries
    murat_profiles.npz         per species: genes x 6 types profile and detection (human ENSG rows)
    murat_gene_type.parquet    I_gk per tier (PP, PO, all) and type
    h3_type_divergence.csv     TS vs TMS per type
    h3_gene_type.parquet       I_gk per type
    murat_tiers.parquet        G_H4 tier means zPP, zPO, zOO and Gt = zPP - zPO (T4.3 pipeline, 4 types)
Usage: python -m revision.tasks.X7_architecture_outputs.golds
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import spearmanr

from revision.tasks.T4_3_scoreboard import golds as TG
from revision.tasks.X7_architecture_outputs import rules as RU
from revision.tasks.X7_architecture_outputs.forward import OUT

GD = OUT / "golds"
PRIMATES, OUTGROUPS = TG.PRIMATES, TG.OUTGROUPS
TYPES = RU.TESTIS_TYPES


def tier(s1, s2):
    return "PP" if s1 in PRIMATES and s2 in PRIMATES else "OO" if s1 in OUTGROUPS and s2 in OUTGROUPS else "PO"


def murat_inputs(fine: bool = False):
    """Per species profiles on the coarse map (all 8 species) or the fine stages (primates only)."""
    import anndata as ad

    m = pd.read_csv(TG.T41 / "h4_ortholog_map_one2one_union.tsv.gz", sep="\t", dtype=str)
    human = np.array(sorted(m.hsapiens_gene_id.unique()))
    hidx = pd.Index(human)
    out = {}
    types = RU.FINE_STAGES if fine else TYPES
    for spc in (PRIMATES if fine else PRIMATES + OUTGROUPS):
        a = ad.read_h5ad(TG.REPO / f"data/raw/camex/2testis/dataset/raw-testis-{spc}-Murat.h5ad")
        ids = a.var["gene"].astype(str).str.replace(r"\.\d+$", "", regex=True).to_numpy()
        mm = m[m.murat_species == spc]
        s2h = dict(zip(mm.species_gene_id, mm.hsapiens_gene_id))
        cols = np.array([i for i, g in enumerate(ids) if g in s2h])
        rows = hidx.get_indexer([s2h[ids[i]] for i in cols])
        X = a.X.tocsr() if sp.issparse(a.X) else sp.csr_matrix(a.X)
        X = X[:, cols].astype(np.float64)
        ct = a.obs["cell_type"].astype(str)
        ct = (ct if fine else ct.map(RU.MURAT_COARSE).fillna("")).to_numpy()
        lib = a.obs["sample_batch"].astype(str).to_numpy()
        prof, det, ncell, halves = {}, {}, {}, {}
        for t in types:
            libs = [l for l in sorted(set(lib)) if ((ct == t) & (lib == l)).sum() >= RU.MIN_CELLS]
            if not libs:
                continue
            P, nexp, nc = [], 0, 0
            per_lib = {}
            for l in libs:
                mk = (ct == t) & (lib == l)
                cnt = np.asarray(X[mk].sum(0)).ravel()
                per_lib[l] = np.log2(1 + 1e6 * cnt / cnt.sum())
                P.append(per_lib[l])
                nexp = nexp + np.asarray((X[mk] > 0).sum(0)).ravel()
                nc += int(mk.sum())
            prof[t], det[t], ncell[t] = np.mean(P, 0), nexp / nc, nc
            if len(libs) >= 2:  # split-library reliability: first half of libraries vs the rest
                k = len(libs) // 2
                halves[t] = (np.mean([per_lib[l] for l in libs[:k]], 0), np.mean([per_lib[l] for l in libs[k:]], 0))
        out[spc] = dict(rows=rows, prof=prof, det=det, ncell=ncell, halves=halves)
    return human, out


def type_divergence(human, inp, types=TYPES):
    rows = []
    rel = {}
    for spc, d in inp.items():
        for t, (h1, h2) in d["halves"].items():
            ok = (d["det"][t] >= RU.DET_MIN)
            r = spearmanr(h1[ok], h2[ok])[0]
            rel[(spc, t)] = 2 * r / (1 + r)  # Spearman-Brown, two halves
    for s1, s2 in itertools.combinations(list(inp), 2):
        a, b = inp[s1], inp[s2]
        common = np.intersect1d(a["rows"], b["rows"])
        ia = pd.Index(a["rows"]).get_indexer(common)
        ib = pd.Index(b["rows"]).get_indexer(common)
        for t in types:
            if t not in a["prof"] or t not in b["prof"]:
                continue
            ok = (a["det"][t][ia] >= RU.DET_MIN) & (b["det"][t][ib] >= RU.DET_MIN)
            r = spearmanr(a["prof"][t][ia][ok], b["prof"][t][ib][ok])[0]
            ra, rb = rel.get((s1, t), np.nan), rel.get((s2, t), np.nan)
            rows.append(dict(s1=s1, s2=s2, tier=tier(s1, s2), type=t, n_genes=int(ok.sum()), rho=r, div=1 - r,
                             rel1=ra, rel2=rb, div_adj=1 - r / np.sqrt(ra * rb) if np.isfinite(ra * rb) else np.nan,
                             cells1=a["ncell"][t], cells2=b["ncell"][t]))
    return pd.DataFrame(rows)


def gene_type_murat(human, inp):
    n = len(human)
    full = {}
    for spc, d in inp.items():
        P = np.full((n, len(TYPES)), np.nan)
        Fd = np.full((n, len(TYPES)), np.nan)
        for j, t in enumerate(TYPES):
            if t in d["prof"]:
                P[d["rows"], j] = d["prof"][t]
                Fd[d["rows"], j] = d["det"][t]
        full[spc] = (P, Fd)
    acc = {tr: [] for tr in ("PP", "PO", "all")}
    for s1, s2 in itertools.combinations(PRIMATES + OUTGROUPS, 2):
        (P1, F1), (P2, F2) = full[s1], full[s2]
        D = P1 - P2
        ok = (F1 >= RU.DET_MIN) & (F2 >= RU.DET_MIN)
        D = np.where(ok, D, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            A = np.abs(D - np.nanmedian(D, 0, keepdims=True))
        tr = tier(s1, s2)
        acc["all"].append(A)
        if tr in acc:
            acc[tr].append(A)
    out = pd.DataFrame(index=pd.Index(human, name="hsapiens_gene_id"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for tr, L in acc.items():
            S = np.stack(L)
            npair = np.isfinite(S).sum(0)
            M = np.nanmean(S, 0)
            M[npair < 3] = np.nan
            for j, t in enumerate(TYPES):
                out[f"I_{tr}_{t}"] = M[:, j]
    return out, full


def h3():
    z = np.load(TG.REPO / "results/revision/T4_3/golds/h3_profiles.npz", allow_pickle=True)
    genes, types = z["genes"].astype(str), [str(t) for t in z["types"]]
    rows, gt = [], pd.DataFrame(index=pd.Index(genes, name="hsapiens_gene_id"))
    for j, t in enumerate(types):
        ok = (z["TS_f"][:, j] >= RU.DET_MIN) & (z["TMS_f"][:, j] >= RU.DET_MIN)
        r = spearmanr(z["TS_x"][ok, j], z["TMS_x"][ok, j])[0]
        rows.append(dict(type=t, n_genes=int(ok.sum()), rho=r, div=1 - r))
        d = z["TS_x"][:, j] - z["TMS_x"][:, j]
        d = np.where(ok, d, np.nan)
        gt[f"I_{t}"] = np.abs(d - np.nanmedian(d))
    return pd.DataFrame(rows), gt


def murat_tiers():
    """T4.3 H4 pipeline, 4 types; returns tier means and checks G_H4 against the gold table."""
    inp = TG.h4_inputs()
    res = TG.h4_species_profiles(inp)
    n = len(inp["genes"])
    tiers = {"PP": [], "PO": [], "OO": []}
    for s1, s2 in itertools.combinations(PRIMATES + OUTGROUPS, 2):
        p1, ok1 = res[s1]
        p2, ok2 = res[s2]
        ok = ok1 & ok2
        r = np.full(n, np.nan)
        if ok.any():
            r[ok] = TG.profile_corr(p1[ok], p2[ok], np.ones(ok.sum(), bool), np.ones(ok.sum(), bool))
        tiers[tier(s1, s2)].append(np.arctanh(np.clip(r, -0.999999, 0.999999)))
    npri = np.sum([res[s][1] for s in PRIMATES], 0)
    nout = np.sum([res[s][1] for s in OUTGROUPS], 0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        tm = {k: np.nanmean(np.column_stack(v), 1) for k, v in tiers.items()}
    G = (tm["PP"] + tm["PO"] + tm["OO"]) / 3
    ok = (npri >= 3) & (nout >= 2)
    G[~ok] = np.nan
    df = pd.DataFrame({"zPP": tm["PP"], "zPO": tm["PO"], "zOO": tm["OO"], "G_H4_check": G},
                      index=pd.Index(inp["genes"], name="hsapiens_gene_id"))
    for k in ("zPP", "zPO", "zOO"):
        df.loc[~ok, k] = np.nan
    df["Gt"] = df.zPP - df.zPO
    gold = pd.read_csv(TG.REPO / "results/revision/T4_3/golds/gold_table.csv", index_col=0)["G_H4"]
    g = gold.reindex(df.index)
    both = g.notna() | df.G_H4_check.notna()
    if not np.allclose(g[both].to_numpy(), df.G_H4_check[both].to_numpy(), equal_nan=True, atol=1e-6):
        raise AssertionError("G_H4 not reproduced")
    return df


def main():
    GD.mkdir(parents=True, exist_ok=True)
    human, inp = murat_inputs()
    td = type_divergence(human, inp)
    td.to_csv(GD / "murat_type_divergence.csv", index=False)
    gt, full = gene_type_murat(human, inp)
    gt.to_parquet(GD / "murat_gene_type.parquet")
    np.savez_compressed(GD / "murat_profiles.npz", genes=human, types=np.array(TYPES),
                        **{f"{s}_x": v[0] for s, v in full.items()}, **{f"{s}_f": v[1] for s, v in full.items()})
    hf, inpf = murat_inputs(fine=True)
    tdf = type_divergence(hf, inpf, RU.FINE_STAGES)
    tdf.to_csv(GD / "murat_fine_type_divergence.csv", index=False)
    print(tdf.groupby("type")[["div", "div_adj", "n_genes"]].mean().round(3).to_string())
    h3t, h3g = h3()
    h3t.to_csv(GD / "h3_type_divergence.csv", index=False)
    h3g.to_parquet(GD / "h3_gene_type.parquet")
    mt = murat_tiers()
    mt.to_parquet(GD / "murat_tiers.parquet")
    print(td.groupby(["tier", "type"])[["div", "div_adj", "n_genes"]].mean().round(3).to_string())
    print(h3t.round(3).to_string())
    print("Gt defined", int(mt.Gt.notna().sum()), "G_H4 reproduced")


if __name__ == "__main__":
    main()
