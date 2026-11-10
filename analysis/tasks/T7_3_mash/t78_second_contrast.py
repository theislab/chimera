"""T7.8: second contrast and perturbation (plan section 8), Family 3, no training, runs in
``revision_de_env``.

Second contrast. GSE136103 (Ramachandran 2019), H-D1 with the same D on macrophages. Human: hepatic
CD45+ libraries of 5 healthy and 5 cirrhotic donors (blood and CD45- excluded); mouse: one healthy
and one CCl4 sample of sorted mononuclear phagocytes. GEO gives no cell labels, so cells are labelled
here: QC as T7.2 (>= 200 genes, mitochondrial fraction < 20%), log CP10K, 2,000 HVG, 30 PCs, Leiden
(resolution 2.0) per species; each cluster takes the marker class with the highest mean score,
class "mono_mac" is the macrophage class, as T7.2 labels Kupffer-like and monocyte-derived cells
together. Human DE: donors with >= 20 cells, ~ sex + condition when both sexes occur in both arms,
else ~ condition; PyDESeq2, l^h unshrunk as in hd1.py. Mouse n = 1 per arm, so no dispersion exists: m is the
log2 ratio of CP10K with pseudocount 0.1, SE(m) the Poisson SE on summed counts. This is
cell-level, not replicated, and is reported as such.

Perturbation. rho_p of D against Hagai 2018 response divergence, G = -response_divergence, so that
positive means a conserved response. Fibroblast dsRNA sheet, 955 genes with human padj < 0.01 (the
plan's set), all genes beside it. Bone-marrow phagocyte LPS sheet (mouse, rat, rabbit, pig; no human)
joined on the mouse ID, exploratory. Covariates C only, null and CI as section 3; |human logFC|
spline is a sensitivity analysis.

Neither dataset is a drug response, and Hagai is cultured fibroblasts and phagocytes, not liver.

    PYTHONPATH=<chimera repo> python analysis/tasks/T7_3_mash/t78_second_contrast.py
"""

from __future__ import annotations

import gzip
import logging
import re
import sys
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.io
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.tasks.P13_kidney.xlsx_min import read_xlsx  # standard library only
from analysis.tasks.T4_3_scoreboard import stats as S
from analysis.tasks.T7_3_mash import de as DEmod
from analysis.tasks.T7_3_mash import hd1 as H

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("T7_8")

ROOT = gene_ids.REPO_ROOT
RAW = ROOT / "data" / "analysis" / "disease" / "raw"
OUT = ROOT / "results" / "analysis" / "T7_8"
SEED = 20261102
MIN_CELLS = 20
PSEUDO = 0.1
MARKERS = {
    "mono_mac": ["CD68", "CD14", "LYZ", "C1QA", "C1QB", "CSF1R", "MS4A7", "MAFB", "FCGR1A", "CD163"],
    "dc": ["CD1C", "CLEC9A", "FLT3", "XCR1", "CLEC10A", "LILRA4", "CD209A", "SIGLECH"],
    "t_nk": ["CD3D", "CD3E", "TRAC", "NKG7", "GNLY", "KLRD1", "NCR1"],
    "b_plasma": ["MS4A1", "CD79A", "CD79B", "MZB1", "JCHAIN"],
    "mast_baso": ["TPSAB1", "CPA3", "KIT"],
    "neutrophil": ["S100A8", "S100A9", "FCGR3B", "CSF3R"],
    "stromal_epithelial": ["ALB", "COL1A1", "PECAM1", "KRT19", "VWF", "DCN"],
}
GSM = re.compile(r"(GSM\d+)_(.+)_(barcodes|genes|matrix)\.(tsv|mtx)\.gz$")


def read_tar() -> dict:
    tar = tarfile.open(RAW / "GSE136103" / "GSE136103_RAW.tar")
    parts: dict = {}
    for mem in tar.getmembers():
        m = GSM.match(mem.name)
        if not m:
            continue
        parts.setdefault(m.group(2), {})[m.group(3)] = mem
    keep = [k for k in parts if re.fullmatch(r"(healthy|cirrhotic)\d_cd45\+", k) or k in ("mouse_healthy", "mouse_fibrotic")]
    out = {}
    for k in sorted(keep):
        p = parts[k]
        mat = scipy.io.mmread(gzip.open(tar.extractfile(p["matrix"]))).T.tocsr().astype(np.float32)
        genes = pd.read_csv(gzip.open(tar.extractfile(p["genes"])), sep="\t", header=None, usecols=[0, 1], names=["gene_id", "symbol"])
        bc = pd.read_csv(gzip.open(tar.extractfile(p["barcodes"])), sep="\t", header=None)[0].astype(str)
        a = ad.AnnData(X=mat, obs=pd.DataFrame({"sample": k}, index=[f"{k}_{b}" for b in bc]),
                       var=pd.DataFrame({"gene_id": genes["gene_id"].to_numpy(), "symbol": genes["symbol"].to_numpy()},
                                        index=genes["gene_id"].to_numpy()))
        out[k] = a
        log.info("%s %s", k, a.shape)
    return out


def annotate(a: ad.AnnData, species: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = a.copy()
    sym = x.var["symbol"].astype(str).str.upper()
    mt = sym.str.startswith("MT-").to_numpy()
    tot = np.asarray(x.X.sum(1)).ravel()
    ngen = np.asarray((x.X > 0).sum(1)).ravel()
    mtf = np.asarray(x[:, mt].X.sum(1)).ravel() / np.maximum(tot, 1)
    keep = (ngen >= 200) & (mtf < 0.20)
    log.info("%s QC %d -> %d", species, x.n_obs, keep.sum())
    x = x[keep].copy()
    sc.pp.normalize_total(x, target_sum=1e4)
    sc.pp.log1p(x)
    sc.pp.highly_variable_genes(x, n_top_genes=2000, flavor="seurat")
    h = x[:, x.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(h, max_value=10)
    sc.tl.pca(h, n_comps=30, random_state=0)
    sc.pp.neighbors(h, n_neighbors=15, use_rep="X_pca", random_state=0, transformer="sklearn")
    sc.tl.leiden(h, resolution=2.0, flavor="igraph", n_iterations=2, directed=False, random_state=0, key_added="cl")
    x.obs["cl"] = h.obs["cl"].astype(str).to_numpy()
    scores = {}
    for c, genes in MARKERS.items():
        g = sym[sym.isin(genes)].index
        scores[c] = pd.Series(np.asarray(x[:, g].X.mean(1)).ravel(), index=x.obs_names)
    sc_df = pd.DataFrame(scores)
    m = sc_df.groupby(x.obs["cl"]).mean()
    rel = m / m.max(0)  # ratio rule of T7.2: a class's mean relative to its best cluster
    label = rel.idxmax(1)
    tab = m.round(3)
    tab.insert(0, "label", label)
    tab.insert(0, "n_cells", x.obs["cl"].value_counts())
    tab.insert(0, "species", species)
    x.obs["label"] = x.obs["cl"].map(label).to_numpy()
    return x.obs[["sample", "cl", "label"]], tab.reset_index()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "de").mkdir(exist_ok=True)
    (OUT / "gse136103").mkdir(exist_ok=True)
    data = read_tar()
    hum = ad.concat([v for k, v in data.items() if not k.startswith("mouse")], join="inner", merge="same")
    mus = ad.concat([v for k, v in data.items() if k.startswith("mouse")], join="inner", merge="same")
    lh, th = annotate(hum, "human")
    lm, tm = annotate(mus, "mouse")
    pd.concat([th, tm]).to_csv(OUT / "gse136103" / "cluster_labels.tsv", sep="\t", index=False)
    cnt = pd.concat([lh.groupby(["sample", "label"], observed=True).size(), lm.groupby(["sample", "label"], observed=True).size()])
    cnt.rename("n_cells").reset_index().to_csv(OUT / "gse136103" / "cell_counts.tsv", sep="\t", index=False)
    # human pseudobulks
    soft = (RAW / "GSE136103" / "GSE136103_samples_brief.soft.txt").read_text()
    sex = {}
    for blk in soft.split("^SAMPLE = ")[1:]:
        t = re.search(r"!Sample_title = (\S+)", blk).group(1).lower()
        s = re.search(r"Sex: (\w+)", blk)
        if s:
            sex[t.split("_")[0]] = s.group(1)
    mac = lh[lh["label"] == "mono_mac"]
    samples = sorted(mac["sample"].unique())
    X, nn, dt = [], [], []
    sub = hum[mac.index].copy()
    for s in samples:
        k = (sub.obs["sample"] == s).to_numpy()
        X.append(np.asarray(sub.X[k].sum(0)).ravel())
        dt.append(np.asarray((sub.X[k] > 0).sum(0)).ravel())
        nn.append(int(k.sum()))
    X = np.array(X)
    dt = np.array(dt)
    obs = pd.DataFrame({"unit": [s.split("_")[0] for s in samples], "n_nuclei": nn}, index=samples)
    obs["condition"] = np.where(obs["unit"].str.startswith("healthy"), "control", "disease")
    obs["sex"] = obs["unit"].map(sex)
    obs.to_csv(OUT / "gse136103" / "human_units.tsv", sep="\t")
    obs = obs[obs["n_nuclei"] >= MIN_CELLS]
    idx = [samples.index(s) for s in obs.index]
    pb = ad.AnnData(X=sp.csr_matrix(X[idx]), obs=obs, var=sub.var.copy())
    pb.layers["n_detected"] = sp.csr_matrix(dt[idx])
    tabs = pd.crosstab(obs["sex"], obs["condition"])
    both = (tabs.shape == (2, 2)) and (tabs.values > 0).all()
    design = "~ sex + condition" if both else "~ condition"
    log.info("human macrophage units\n%s\n%s", obs, design)
    r = DEmod.fit_units(pb, design, "condition", "disease", "control", 8)
    r["symbol"] = sub.var.set_index("gene_id")["symbol"].reindex(r["gene_id"]).to_numpy()
    r.to_csv(OUT / "de" / "human_cirrhosis_macrophage.tsv", sep="\t", index=False)
    scar = ["TREM2", "CD9", "SPP1", "GPNMB", "LGALS3", "CD63", "FABP5"]
    r[r["symbol"].isin(scar)][["symbol", "log2fc_mle", "lfcse_mle", "padj", "det_frac"]].to_csv(
        OUT / "gse136103" / "scar_markers_human.tsv", sep="\t", index=False)
    # mouse, n = 1 per arm
    mm = lm[lm["label"] == "mono_mac"]
    subm = mus[mm.index].copy()
    arms = {}
    for s in ("mouse_healthy", "mouse_fibrotic"):
        k = (subm.obs["sample"] == s).to_numpy()
        arms[s] = (np.asarray(subm.X[k].sum(0)).ravel(), np.asarray((subm.X[k] > 0).sum(0)).ravel(), int(k.sum()))
    a, b = arms["mouse_fibrotic"], arms["mouse_healthy"]
    ca, cb = a[0], b[0]
    Na, Nb = ca.sum(), cb.sum()
    pa, pb_ = ca / Na * 1e4, cb / Nb * 1e4
    md = pd.DataFrame({"gene_id": subm.var["gene_id"].to_numpy(), "gene_symbol": subm.var["symbol"].to_numpy(),
                       "log2fc_shrunk": np.log2((pa + PSEUDO) / (pb_ + PSEUDO)),
                       "log2fc_mle": np.log2((pa + PSEUDO) / (pb_ + PSEUDO)),
                       "lfcse_shrunk": np.sqrt(1 / (ca + 0.5) + 1 / (cb + 0.5)) / np.log(2),
                       "det_frac": (a[1] + b[1]) / (a[2] + b[2]), "padj": np.nan, "count_ccl4": ca, "count_healthy": cb})
    md.to_csv(OUT / "de" / "mouse_ccl4_macrophage_n1.tsv", sep="\t", index=False)
    log.info("mouse cells healthy %d CCl4 %d", b[2], a[2])
    # H-D1 with the same D
    tab = H.vocabulary()
    rng = np.random.default_rng(SEED)
    hu = r.dropna(subset=["log2fc_mle", "gene_id"]).drop_duplicates("gene_id", keep=False).set_index("gene_id")
    mo = md.drop_duplicates("gene_id", keep=False).set_index("gene_id")
    u = H.universe(tab, hu, mo)
    log.info("universe %d (joined %d)", len(u), u.attrs["n_join"])
    rows = [{"family": 3, "contrast": "GSE136103 macrophages, human cirrhosis vs mouse CCl4 (mouse n=1)",
             **H.rho_row(u, s, 10000, 10000, rng)} for s in ("D", "S_ema", "B1b", "B2b")]
    rows.extend({"family": 3, "contrast": "GSE136103 macrophages", **x} for x in H.pi_row(u))
    pd.DataFrame(rows).to_csv(OUT / "h1_gse136103.tsv", sep="\t", index=False)
    pd.DataFrame([H.anchor("GSE136103 macrophages", u)]).to_csv(OUT / "h1_gse136103_anchor.tsv", sep="\t", index=False)
    # Hagai
    sheets = read_xlsx(str(RAW / "Hagai2018" / "Hagai2018_Supplementary_Table_4.xlsx"))
    def sheet(name):
        rowsx = sheets[name]
        d = pd.DataFrame(rowsx[1:], columns=rowsx[0])
        for c in d.columns[1:]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
        return d
    fb = sheet("fibroblasts_FC_divergence").set_index("gene")
    ph = sheet("phagocytes_FC_diveregnce").set_index("gene")
    out = []

    def rho_h(v: pd.DataFrame, label: str, nuis=()):
        v = v.dropna(subset=["Q"]).reset_index(drop=True)
        v["abs_l"] = 0.0
        v["se_m"] = 0.0
        for sc_name in ("D", "S_ema", "B1b", "B2b"):
            w = v.dropna(subset=[sc_name]).reset_index(drop=True)
            cols = H.C_COLS + list(nuis)
            Z = S.design(w, cols)
            Rm = np.column_stack([S.midrank(w["Q"]), S.midrank(w[sc_name])])
            E = S.resid(Rm, Z)
            rho = float(np.corrcoef(E[:, 0], E[:, 1])[0, 1])
            lab = S.strata(w["prior_raw"].to_numpy(), w["log_cp10k_spmean"].to_numpy())
            null = S.perm_null(E[:, 1], E[:, 0], lab, 10000, rng)
            bb = S.bootstrap(Rm, Z, 10000, rng, w["block"].to_numpy())[:, 1]
            lo, hi = S.pct_ci(bb)
            out.append(dict(family=3, gold=label, score=sc_name, n=len(w), rho_p=rho, ci_lo=lo, ci_hi=hi,
                            p_one_sided=S.p_one_sided(rho, null), rho_raw=float(H.spearmanr(w[sc_name], w["Q"])[0]),
                            covariates="C" + (" + " + ",".join(nuis) if nuis else "")))

    base = tab.dropna(subset=H.C_COLS).copy()
    a1 = base.copy()
    a1["Q"] = -fb["response_divergence"].reindex(a1["hsapiens_gene_id"]).to_numpy()
    a1["abs_fc"] = fb["human_log2FoldChange"].abs().reindex(a1["hsapiens_gene_id"]).to_numpy()
    a1["hp"] = fb["human_padj"].reindex(a1["hsapiens_gene_id"]).to_numpy()
    rho_h(a1[a1["hp"] < 0.01], "Hagai fibroblasts, human padj < 0.01 (plan set)")
    rho_h(a1[a1["hp"] < 0.01].dropna(subset=["abs_fc"]), "Hagai fibroblasts, human padj < 0.01 (plan set)", nuis=("abs_fc",))
    rho_h(a1, "Hagai fibroblasts, all genes")
    a2 = base.copy()
    a2["Q"] = -ph["response_divergence"].reindex(a2["mouse_id"]).to_numpy()
    a2["hp"] = ph["mouse_padj"].reindex(a2["mouse_id"]).to_numpy()
    rho_h(a2, "Hagai bone-marrow phagocytes, all genes (exploratory)")
    rho_h(a2[a2["hp"] < 0.01], "Hagai bone-marrow phagocytes, mouse padj < 0.01 (exploratory)")
    pd.DataFrame(out).to_csv(OUT / "hagai.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).T.to_string())
    print(pd.DataFrame(out).to_string())
    (OUT / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
