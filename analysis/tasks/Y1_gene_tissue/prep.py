"""Y1 (a): the gene x tissue matrices and the joined gold table. Joins tissues on the human Ensembl ID.

Writes results/analysis/Y1/matrices/: raw_<score>.parquet (genes x tissues, centred on the tissue median of supported genes for S,
D, Sdir, Sdir_pattern, Sdir_level; raw for B1), z_<score>__<design>.parquet (z_gt), support.parquet, strict.parquet, cov_<col>.parquet,
golds.parquet, blocks.parquet, halves_S.npz. Nothing is compared across tissues here.

    python -I launch.py analysis.tasks.Y1_gene_tissue.prep
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd

from analysis.common import controls as K
from analysis.tasks.Y1_gene_tissue import common as Y
from analysis.tasks.Y1_gene_tissue import rules as RU

MAT = RU.OUT / "matrices"
ORDER = RU.TISSUES
SCORES = ["S", "D", "D_raw", "B1", "Sdir", "Sdir_pattern", "Sdir_level"]
COVS = ["expr", "det", "tau", "amp", "dom", "abs_shift", "shift", "prior_raw", "log_cds_length", "ppass", "det_human", "det_mouse",
        "expr_human", "expr_mouse"]
DESIGNS = {"ED": ["expr", "det"], "EDpi": ["expr", "det", "prior_raw"], "EDtau": ["expr", "det", "tau"]}


def wide(tabs: dict, col: str) -> pd.DataFrame:
    return pd.DataFrame({t: tabs[t].set_index("hsapiens_gene_id")[col] for t in ORDER})


def z_on(y: pd.Series, cov: pd.DataFrame, cols: list[str]) -> pd.Series:
    """z of rank(y) given the spline design of cols, within the genes where y and cols are finite."""
    d = pd.concat([y.rename("y"), cov[cols]], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    Z = K.design(d, columns=cols)
    e = K.residual_rank(d["y"].to_numpy(), Z)
    return pd.Series((e - e.mean()) / e.std(), index=d.index)


def z_matrix(raw: pd.DataFrame, keep: pd.DataFrame, cov: dict, cols: list[str]) -> pd.DataFrame:
    """Per tissue, z on the kept genes. keep: boolean genes x tissues."""
    out = {}
    for t in ORDER:
        y = raw[t][keep[t].fillna(False).to_numpy(bool)]
        cv = pd.DataFrame({c: cov[c][t] for c in cols}).reindex(y.index)
        out[t] = z_on(y, cv, cols)
    return pd.DataFrame(out).reindex(raw.index)


def forward_margins(t: str) -> dict:
    """From the y1half forward pass (one seed): weighted margin, its cell halves, unweighted margin, human-mouse rows."""
    seed = {"liver": 42, "testis": 42, "kidney": 42, "islet": 42, "bone_marrow": 42, "eye": 42}[t]
    fw = dict(np.load(Y.FW / f"fw_{t}_s{seed}.npz", allow_pickle=False))
    gr = fw["y1half__groups"]
    sidx, lab, n = np.asarray(gr["species_idx"], int), np.asarray(gr["label"]).astype(str), np.asarray(gr["n"], float)
    typ = np.array([l.rsplit("|", 1)[0] for l in lab])
    hf = np.array([l.rsplit("|", 1)[1] for l in lab])
    sp = [str(x) for x in fw["species"]]
    h = next(i for i, s in enumerate(sp) if s.lower() == "human")
    m = next(i for i, s in enumerate(sp) if s.lower() == "mouse")
    Sum = fw["y1half__sum_meanz"]
    out = {}

    def margin(rows):
        r = np.zeros(Sum.shape[1])
        for s in (h, m):
            mk = rows & (sidx == s)
            r += Sum[mk].sum(0) / n[mk].sum()
        return r / 2

    out["S_fw"] = margin(np.ones(len(lab), bool))
    # halves keep the species x type weights of the whole: rate per (species, type, half), weight n_sk / n_s
    for hn in ("h0", "h1"):
        r = np.zeros(Sum.shape[1])
        for s in (h, m):
            ns = n[sidx == s].sum()
            for k in np.unique(typ[sidx == s]):
                mk = (sidx == s) & (typ == k) & (hf == hn)
                if n[mk].sum() > 0:
                    nk = n[(sidx == s) & (typ == k)].sum()
                    r += (nk / ns) * Sum[mk].sum(0) / n[mk].sum()
        out[f"S_{hn}"] = r / 2
    shared = sorted(set(typ[sidx == h]) & set(typ[sidx == m]))
    u = np.zeros(Sum.shape[1])
    for s in (h, m):
        for k in shared:
            mk = (sidx == s) & (typ == k)
            u += Sum[mk].sum(0) / n[mk].sum()
    out["S_unw"] = u / (2 * len(shared))
    out["var_names"] = fw["var_names"].astype(str)
    return out


def esm_table(t: str, var_names, ids) -> pd.DataFrame:
    a = ad.read_h5ad(Y.TISSUES[t]["h5ad"], backed="r")
    if list(a.var_names) != list(var_names):
        raise AssertionError(t)
    F = np.asarray(a.varm["gene_features"])[:, :64]
    return pd.DataFrame(F, index=ids, columns=[f"pc{i}" for i in range(64)])


def seed_D(t: str, tb: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for s, run in Y.TISSUES[t]["runs"].items():
        out[f"D_s{s}"] = K.read_run(run)["D"]
    return pd.DataFrame(out, index=tb["hsapiens_gene_id"].values)


def golds() -> pd.DataFrame:
    gt = pd.read_csv(Y.RES / "T4_3" / "golds" / "gold_table.csv", index_col=0)
    G = pd.DataFrame(dict(G_liver=gt["G_H3"], G_testis=gt["G_H4"], H4_nsp=gt["H4_nsp"]))
    h3 = pd.read_parquet(Y.RES / "X4" / "cache" / "atlas_h3.parquet")
    ty = ["hepatocyte", "macrophage", "endothelial", "nk", "b_cell", "stellate"]
    xT, xM = h3[[f"TS_x_{k}" for k in ty]].to_numpy(), h3[[f"TMS_x_{k}" for k in ty]].to_numpy()
    fT, fM = h3[[f"TS_f_{k}" for k in ty]].to_numpy(), h3[[f"TMS_f_{k}" for k in ty]].to_numpy()
    liv = pd.DataFrame(dict(amp_liver=(xT.std(1) + xM.std(1)) / 2, aexpr_liver=(xT.mean(1) + xM.mean(1)) / 2,
                            adet_liver=(fT.mean(1) + fM.mean(1)) / 2), index=h3.index)
    h4 = pd.read_parquet(Y.RES / "X4" / "cache" / "atlas_h4_types.parquet")
    xt = h4[[c for c in h4.columns if c.startswith("H4_x_")]].to_numpy()
    tes = pd.DataFrame(dict(amp_testis=np.nanstd(xt, axis=1), aexpr_testis=np.nanmean(xt, axis=1)), index=h4.index)
    tes["adet_testis"] = tes["aexpr_testis"]  # the testis atlas table has no detection: mean expression enters twice, rank of the same column
    G = G.join(liv, how="outer").join(tes, how="outer")
    for tag in ("kidney", "pancreas", "pancreas_noacinar"):
        g = pd.read_parquet(RU.OUT / "gold" / f"gold_{tag}.parquet")
        G = G.join(g.rename(columns=dict(G=f"G_{tag}", atlas_amp=f"amp_{tag}", atlas_expr=f"aexpr_{tag}", atlas_det=f"adet_{tag}")), how="outer")
    return G


def main():
    MAT.mkdir(parents=True, exist_ok=True)
    tabs, hv, esm, dsd = {}, {}, {}, {}
    for t in ORDER:
        tb = Y.tissue_table(t)
        tb = tb[tb["hsapiens_gene_id"].notna()].copy()
        sd = pd.read_parquet(Y.CACHE / f"Sdir__{t}.parquet").set_index("var_name")
        tb = tb.join(sd[["Sdir", "Sdir_pattern", "Sdir_level"]], on="var_name")
        fm = forward_margins(t)
        fi = pd.Series(np.arange(len(fm["var_names"])), index=fm["var_names"]).reindex(tb["var_name"]).to_numpy()
        ppass = tb["ppass"].to_numpy(float)
        unm = ~tb["masked"].to_numpy(bool)
        for k in ("S_fw", "S_h0", "S_h1", "S_unw"):
            tb[k] = np.where(unm, ppass * fm[k][fi], 0.0)
        tabs[t] = tb
        esm[t] = esm_table(t, Y.tissue_table(t)["var_name"].tolist(), Y.tissue_table(t)["hsapiens_gene_id"].tolist()).reindex(tb["hsapiens_gene_id"])
        dsd[t] = seed_D(t, Y.tissue_table(t)).reindex(tb["hsapiens_gene_id"])
    sup = wide(tabs, "support").fillna(False).astype(bool)
    strict = wide(tabs, "strict").fillna(False).astype(bool)
    sup.to_parquet(MAT / "support.parquet")
    strict.to_parquet(MAT / "strict.parquet")
    cov = {c: wide(tabs, c) for c in COVS}
    for c, v in cov.items():
        v.to_parquet(MAT / f"cov_{c}.parquet")
    wide(tabs, "top_type").to_parquet(MAT / "top_type.parquet")
    wide(tabs, "symbol").bfill(axis=1).iloc[:, 0].rename("symbol").to_frame().to_parquet(MAT / "symbols.parquet")
    wide(tabs, "sign_stable_S").to_parquet(MAT / "sign_stable.parquet")
    keep_sup = sup
    for sc in SCORES + ["S_fw", "S_h0", "S_h1", "S_unw"]:
        raw = wide(tabs, sc)
        out = raw.copy()
        if sc != "B1":
            out = raw - raw.where(sup).median()
        out.to_parquet(MAT / f"raw_{sc}.parquet")
        for dn, cols in DESIGNS.items():
            keep = keep_sup & raw.notna()
            if sc.startswith("S_") and dn != "ED":
                continue
            z_matrix(raw, keep, cov, cols).to_parquet(MAT / f"z_{sc}__{dn}.parquet")
    # B1 on its own genes (defined, not limited to count support), beside
    b1 = wide(tabs, "B1")
    z_matrix(b1, b1.notna(), cov, DESIGNS["ED"]).to_parquet(MAT / "z_B1own__ED.parquet")
    # per-seed S and D, z on ED, for seed reliability
    for t in ORDER:
        tb = tabs[t]
        idx = tb["hsapiens_gene_id"].to_numpy()
        ss = tb[[c for c in tb.columns if c.startswith("S_s") and c[3:].isdigit()]].set_axis(idx)
        ss.to_parquet(MAT / f"seedS__{t}.parquet")
        dsd[t].to_parquet(MAT / f"seedD__{t}.parquet")
        esm[t].to_parquet(MAT / f"esm__{t}.parquet")
    # blocks: kidney ESM modules; genes outside the kidney vocabulary fall back to the islet (testis table) modules, then own label
    bk = {}
    for t, off in (("kidney", 0), ("islet", 100000), ("testis", 200000), ("liver", 300000), ("bone_marrow", 400000), ("eye", 500000)):
        for g, b in zip(tabs[t]["hsapiens_gene_id"], tabs[t]["block"]):
            bk.setdefault(g, int(b) + off)
    pd.Series(bk, name="block").rename_axis("hsapiens_gene_id").to_frame().to_parquet(MAT / "blocks.parquet")
    golds().to_parquet(MAT / "golds.parquet")
    print({t: (int(sup[t].sum()), int(strict[t].sum())) for t in ORDER}, "support in all six", int(sup.all(axis=1).sum()),
          "B1 too", int((sup.all(axis=1) & b1.notna().all(axis=1)).sum()), flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
