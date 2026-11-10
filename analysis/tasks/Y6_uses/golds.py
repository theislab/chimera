"""Y6: independent references computed from downloads. No Chimera readout is read. Run with python -I after setting the path argument.

Liver primate gold: Tabula Sapiens (10x 3' v3) vs Tabula Microcebus liver (10x 3' v2, lemur; a held-out primate), T4.3 H3 recipe, on the five types with at
least 30 cells in each atlas (hepatocyte, macrophage, endothelial, nk, b_cell; TM has 8 fibroblasts, so stellate is dropped). G_H3_5 = the same five types of the TS vs
TMS profiles (T4.3 h3_profiles.npz), so the two golds differ only in the species.

Hybrid cis reference: allele counts of human-chimpanzee hybrid cells, see rules.py USE 4. No fit is touched.

    python -I analysis/tasks/Y6_uses/golds.py
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[3]
D = ROOT / "data" / "analysis" / "Y6"
OUT = ROOT / "results" / "analysis" / "Y6" / "gold"
TYPES = ["hepatocyte", "macrophage", "endothelial", "nk", "b_cell"]
TM_MAP = {"hepatocyte": "hepatocyte", "macrophage": "macrophage", "endothelial cell of hepatic sinusoid": "endothelial", "endothelial cell": "endothelial",
          "natural killer cell": "nk", "B cell": "b_cell"}
DET_MIN = 0.01
MIN_ALLELIC = 30
ASE_TYPES = ["CM", "HP", "MN", "PP", "SKM", "RPE"]


def profile_corr(A, B, ok_a, ok_b):
    """Pearson across types of the atlas-centred profiles (T4.3 profile_corr); NaN for a constant profile or a gene not detected."""
    a, b = A - A.mean(1, keepdims=True), B - B.mean(1, keepdims=True)
    sa, sb = np.sqrt((a ** 2).sum(1)), np.sqrt((b ** 2).sum(1))
    with np.errstate(invalid="ignore", divide="ignore"):
        g = (a * b).sum(1) / (sa * sb)
    g[~(ok_a & ok_b) | (sa == 0) | (sb == 0)] = np.nan
    return g


def lemur_liver():
    z = np.load(ROOT / "results/analysis/T4_3/golds/h3_profiles.npz", allow_pickle=True)
    genes, types = np.array([str(x) for x in z["genes"]]), [str(x) for x in z["types"]]
    idx = [types.index(t) for t in TYPES]
    tsx, tsf, tmx, tmf = z["TS_x"][:, idx], z["TS_f"][:, idx], z["TMS_x"][:, idx], z["TMS_f"][:, idx]
    bm = pd.read_csv(ROOT / "data/analysis/bone_marrow/ortholog_table_bone_marrow.csv")
    lem = pd.read_parquet(ROOT / "results/analysis/Y6/annot/compara110_pairs.parquet")
    lem = lem[(lem.species_ens == "microcebus_murinus") & (lem.homology_type == "ortholog_one2one")].drop_duplicates("hsapiens_gene_id")
    m1 = dict(zip(bm.hsapiens_gene_id, bm.mmurinus_gene_id))
    m1.update(dict(zip(lem.hsapiens_gene_id, lem.gene_id)))
    a = ad.read_h5ad(D / "tm_liver/tm_liver.h5ad")
    assay = a.obs["assay"].astype(str).to_numpy() == "10x 3' v2"
    lab = np.array([TM_MAP.get(x, "") for x in a.obs["cell_type"].astype(str)])
    keep = assay & (lab != "")
    X = a.raw.X[np.flatnonzero(keep)]
    X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
    d = X.data
    if not (np.all(d >= 0) and np.all(d == np.round(d))):
        raise AssertionError("TM liver raw.X is not raw counts")
    var = pd.Index(a.raw.var_names)
    want = np.array([m1.get(g, "") for g in genes])
    col = var.get_indexer(want)
    usable = col >= 0
    # Library size over the human-lemur 1:1 genes with the mapped vocabulary (T4.3: over the shared vocabulary).
    Xv = X[:, col[usable]].astype(np.float64)
    lm = lab[keep]
    P, F, N = [], [], []
    for t in TYPES:
        m = lm == t
        Xt = Xv[np.flatnonzero(m)]
        S = np.asarray(Xt.sum(0)).ravel()
        P.append(np.log2(1 + 1e6 * S / max(S.sum(), 1)))
        Xb = Xt.copy()
        Xb.data = (Xb.data > 0).astype(np.float64)
        F.append(np.asarray(Xb.sum(0)).ravel() / max(m.sum(), 1))
        N.append(int(m.sum()))
    print("lemur cells", dict(zip(TYPES, N)), "vocabulary", int(usable.sum()), flush=True)
    if min(N) < 30:
        raise AssertionError("a lemur type is under 30 cells")
    PL, FL = np.column_stack(P), np.column_stack(F)
    Glem = np.full(len(genes), np.nan)
    Glem[usable] = profile_corr(tsx[usable], PL, tsf[usable].max(1) >= DET_MIN, FL.max(1) >= DET_MIN)
    Gh5 = profile_corr(tsx, tmx, tsf.max(1) >= DET_MIN, tmf.max(1) >= DET_MIN)
    out = pd.DataFrame({"G_lemur": Glem, "G_H3_5": Gh5}, index=pd.Index(genes, name="hsapiens_gene_id"))
    out.to_parquet(OUT / "liver_primate_gold.parquet")
    print(out.notna().sum().to_dict(), "both", int(out.notna().all(axis=1).sum()), flush=True)


def _wang(path):
    d = pd.read_csv(path, sep="\t").drop_duplicates("gene").set_index("gene")
    res = {}
    for t in ASE_TYPES:
        hc = [c for c in d.columns if f"_{t}_" in c and c.endswith("_human")]
        cc = [c for c in d.columns if f"_{t}_" in c and c.endswith("_chimp")]
        res[t] = (d[hc].sum(1), d[cc].sum(1), len(hc))
    return res


def ase_wang():
    a = _wang(D / "ase_wang2024/GSE232948_RNAseq_ASE_CM_HP_MN_PP_SKM_RPE_GRCh38.txt.gz")
    b = _wang(D / "ase_wang2024/GSE232948_RNAseq_ASE_CM_HP_MN_PP_SKM_RPE_PanTro6.txt.gz")
    print({t: a[t][2] for t in ASE_TYPES}, flush=True)
    out = pd.DataFrame(index=a["HP"][0].index.union(b["HP"][0].index))
    for t in ASE_TYPES:
        ha, ca = a[t][0].reindex(out.index), a[t][1].reindex(out.index)
        hb, cb = b[t][0].reindex(out.index), b[t][1].reindex(out.index)
        ok = ((ha + ca) >= MIN_ALLELIC) & ((hb + cb) >= MIN_ALLELIC)
        v = (np.log2((ha + 1) / (ca + 1)) + np.log2((hb + 1) / (cb + 1))) / 2
        v = v.where(ok)
        out[f"a_{t}"] = v - v.median()
        out[f"n_{t}"] = (ha + ca + hb + cb) / 2
    cols = [f"a_{t}" for t in ASE_TYPES]
    out["n_types"] = out[cols].notna().sum(1)
    out["P_cis"] = -out[cols].std(axis=1, ddof=1).where(out.n_types >= 4)
    return out


def _agoglia(ref, kind, stem):
    meta = pd.read_csv(D / f"ase_agoglia2021/GSE144825_{kind}_meta.txt.gz", sep="\t")
    d = pd.read_csv(D / f"ase_agoglia2021/GSE144825_{kind}_{stem}.txt.gz", sep="\t", index_col=0)
    hyb = meta[meta.Species == "Hybrid"].Sample.tolist()
    return d[[f"{s}_human" for s in hyb]].sum(1), d[[f"{s}_chimp" for s in hyb]].sum(1), len(hyb)


def ase_agoglia():
    out = {}
    for kind, tag in (("bulk_iPS", "iPS"), ("bulk_CS", "CS")):
        ha, ca, n = _agoglia("h", kind, "GRCh38")
        hb, cb, _ = _agoglia("c", kind, "PanTro5")
        print(kind, "hybrid samples", n, flush=True)
        idx = ha.index.union(hb.index)
        ha, ca, hb, cb = ha.reindex(idx), ca.reindex(idx), hb.reindex(idx), cb.reindex(idx)
        ok = ((ha + ca) >= MIN_ALLELIC) & ((hb + cb) >= MIN_ALLELIC)
        v = ((np.log2((ha + 1) / (ca + 1)) + np.log2((hb + 1) / (cb + 1))) / 2).where(ok)
        out[f"a_{tag}"] = v - v.median()
    return pd.DataFrame(out)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lemur_liver()
    w = ase_wang()
    ag = ase_agoglia()
    all_ = w.join(ag, how="outer")
    all_.index.name = "symbol"
    all_.to_parquet(OUT / "ase_reference.parquet")
    print(all_.filter(like="a_").notna().sum().to_dict(), flush=True)
    print("n_types>=4", int((all_.n_types >= 4).sum()), flush=True)


if __name__ == "__main__":
    main()
