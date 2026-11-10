"""T7.2: annotated disease objects for the liver MASH case study.

Human: GSE202379 (Gribben 2024), ``*-raw_counts.csv.gz`` only (never ``norm_counts``).
Mouse: GSE192742 diet series CS184-CS197 (Guilliams 2022), ``filtered_feature_bc_matrix`` h5.
Hold-out mice are fixed in ``../T7_1_holdouts/holdouts.tsv`` (CS184, CS185, CS195, CS196).
No disease concordance, no logFC and no Chimera readout is computed here.

Steps (task card T7.2, plan section 8; every rule below was fixed before the mouse
type counts were seen):

1. ``_validate_raw_counts``, ``_standardize_obs_columns``, ``_qc_filter`` (min 200 genes,
   mitochondrial fraction < 20%) from ``chimera._data_utils``. The GEO human CSVs carry no
   ``MT-`` rows, so for human the same 20% threshold is applied to the authors'
   ``percent.mt.RNA``.
2. Human coarse type from the authors' ``cell.annotation`` (Seurat ``@meta.data``, extracted
   at T7.1). Checked against coarse marker sets and against kNN label transfer from the
   CAMEX human liver nuclei (CS170).
3. Mouse coarse type: kNN label transfer (reference PCA, k = 15) from CAMEX mouse liver nuclei
   (ABU20, author labels; its 23 "HsPCs" are dropped from the reference), combined with coarse
   marker sets on Leiden clusters of the query (resolution ``LEIDEN_RES``). Per cluster:
   transfer majority T, marker class M = class with the largest ratio of the cluster's mean
   marker expression to that class's highest cluster mean. Margin (best minus second ratio)
   < ``MIN_MARGIN`` -> "unresolved" (mixed or doublet-like). T == M with T carried by
   >= ``MIN_TRANSFER_FRAC`` of the cluster -> T. M a class absent from the reference
   (lymphocyte, neutrophil, mesothelial) -> M. Otherwise "unresolved" (excluded downstream).
4. Disease states (plan section 8, frozen marker sets; human symbols, mouse 1:1 ortholog from
   the testis table, else Ensembl Compara 110 ``ortholog_one2one``): ``scanpy.tl.score_genes``
   on log-normalised counts within the coarse type, state = top quartile of that type within
   species. Scoring population = the species' training nuclei before caps (human: primary
   healthy donors and NASH without cirrhosis; mouse: the 8 training mice).
5. Primary sets. Human healthy = "Healthy control" without P98 (D13), HL1 right lobe only (A6
   3.2; all other primary samples are right lobe). Human MASH = "NASH w/o cirrhosis". Mouse
   training = holdouts.tsv role "training". P98, the other HL1 lobes, NAFLD, cirrhosis,
   end stage and hold-out mice stay in the per-sample pseudobulks.
6. Donor caps for the pooled object: at most ``CAP`` nuclei per species x condition x coarse
   type; within each such stratum, donors share the quota equally (water-filling, a donor with
   fewer nuclei gives its unused share to the others), random draw with ``SEED``.
7. Pooled object as ``notebooks/01_data_preparation.ipynb`` section 8 builds the testis
   cohorts: ``ortholog_table_testis.csv`` (15,732 groups), ``compute_conservation_prior``
   (table ``mean_identity``), ``download_and_process_esm2`` over the table's species
   (hsapiens, mmulatta, mmusculus) as for the canonical testis features, ``pca_esm2``,
   ``compute_esm2_interspecies_variance``, ``construct_unified_adata``,
   ``compute_expression_stats`` and ``assemble_gin_features`` on the pooled training nuclei,
   then, as ``training/build_subset.py`` does for the pair cohorts, genes in < 3 nuclei are
   dropped and ``precompute_p_null`` is run on the result. Genes the human GEO matrices do not
   contain (MT genes and genes the authors removed) are also dropped, because they are absent
   by construction rather than zero. ``obs``: species (Human/Mouse), batch = sample,
   cell_ontology_class, condition (control/disease) and the native condition label.
8. Per-sample pseudobulks per coarse type for every sample (all human, all 12 mice), from all
   QC-passed annotated nuclei (no caps), native gene space, with per-gene nuclei-detected
   counts for T7.3's 5% detection rule.

Gate G-D1 (plan section 8): >= 3 healthy donors (P98 excluded), >= 15 MASH donors, >= 4 SD +
4 WD training mice, >= 4 of 5 types pseudobulk-able at >= 20 nuclei per sample. Reading fixed
here, before the mouse counts: a type is pseudobulk-able if, after dropping the DE units
(human donor, mouse animal) with < 20 nuclei of that type, the remaining units still meet the
gate's own unit minima (3 healthy, 15 MASH, 4 SD and 4 WD training mice). The strict reading
(every unit >= 20) and sensitivity counts (P98 included, all HL1 lobes, all 12 mice) are
reported beside it and do not decide the gate.

Usage::

    PYTHONPATH=<chimera repo> python analysis/tasks/T7_2_disease_objects/build.py [--force]
        [--reannotate] [--jobs 12]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import pickle
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.common import gene_ids
from analysis.common.outputs import TaskOutput, add_common_args, git_sha

from chimera._data_utils import (
    _qc_filter,
    _standardize_obs_columns,
    _validate_raw_counts,
    assemble_gin_features,
    compute_conservation_prior,
    compute_esm2_interspecies_variance,
    compute_expression_stats,
    construct_unified_adata,
    download_and_process_esm2,
    pca_esm2,
    precompute_p_null,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("T7_2")

ROOT = gene_ids.REPO_ROOT
RAW = ROOT / "data" / "analysis" / "disease" / "raw"
META = ROOT / "data" / "analysis" / "disease" / "meta" / "GSE202379_meta.data.tsv.gz"
HOLDOUTS = ROOT / "analysis" / "tasks" / "T7_1_holdouts" / "holdouts.tsv"
REF_MOUSE = ROOT / "data" / "raw" / "camex" / "1liver" / "dataset" / "raw-liver-mouse-Martin.h5ad"
REF_HUMAN = ROOT / "data" / "raw" / "camex" / "1liver" / "dataset" / "raw-liver-human-Martin.h5ad"
TABLE = ROOT / "data" / "processed" / "ortholog_table_testis.csv"
CANON_TESTIS = ROOT / "data" / "processed" / "camex_testis_unified.h5ad"
SATURN = ROOT / "data" / "raw" / "saturn_esm2"
COMPARA = ROOT / "data" / "analysis" / "ensembl110" / "Compara.110.protein_default.homologies.tsv.gz"
OUT = ROOT / "data" / "analysis" / "disease"

SEED = 20261101
CAP = 2000
MIN_GENES = 200
MAX_MT = 20.0
MIN_NUCLEI = 20
LEIDEN_RES = 2.0
MIN_MARGIN = 0.10  # marker ratio of the best class minus the second best, per cluster
MIN_TRANSFER_FRAC = 0.5  # share of the cluster's nuclei carrying the transfer majority label
KNN = 15
N_HVG = 2000
N_PCS = 50
P98 = "98"
HL1 = "HL1"
PRIMARY = ["hepatocyte", "hsc_fibroblast", "endothelial", "macrophage", "cholangiocyte"]
MIN_OTHER = 50  # an extra coarse class enters the pooled object only with >= 50 training nuclei per species

HUMAN_AUTHOR_MAP = {
    "Hepatocytes": "hepatocyte", "Stellate": "hsc_fibroblast", "Endothelial": "endothelial",
    "Macrophages": "macrophage", "Cholangiocytes": "cholangiocyte", "Lymphocytes": "lymphocyte",
    "B-cell 1": "b_cell", "B-cell 2": "b_cell", "Neutrophils": "neutrophil", "unknown": "unassigned",
}
REF_MOUSE_MAP = {
    "Hepatocytes": "hepatocyte", "Endothelial cells": "endothelial", "Fibroblasts": "hsc_fibroblast",
    "Kupffer cells": "macrophage", "Cholangiocytes": "cholangiocyte", "B cells": "b_cell",
}  # "HsPCs" (23 nuclei) dropped from the reference
REF_HUMAN_MAP = {
    "Hepatocytes": "hepatocyte", "Endothelial cells": "endothelial", "Fibroblasts": "hsc_fibroblast",
    "Macrophages": "macrophage", "Cholangiocytes": "cholangiocyte",
}
# Coarse marker sets (human symbols; annotation only, not the frozen state sets).
COARSE_MARKERS = {
    "hepatocyte": ["ALB", "APOB", "APOH", "TTR", "CYP2E1", "ASS1", "PCK1", "HPX"],
    "endothelial": ["KDR", "PTPRB", "STAB2", "FLT1", "PECAM1", "DNASE1L3", "EMCN", "CLEC4G"],
    "hsc_fibroblast": ["RELN", "DCN", "LRAT", "COLEC11", "HGF", "CXCL12", "NGFR"],
    "macrophage": ["CSF1R", "C1QA", "C1QB", "C1QC", "VSIG4", "CD5L", "MARCO", "CD163"],
    "cholangiocyte": ["KRT19", "KRT7", "EPCAM", "SOX9", "PKHD1", "BICC1", "HNF1B"],
    "b_cell": ["CD79A", "CD79B", "MS4A1", "CD19", "PAX5", "BANK1", "MZB1", "JCHAIN"],
    "lymphocyte": ["CD3E", "CD3G", "CD2", "SKAP1", "NKG7", "IL7R", "THEMIS", "CD247"],
    "neutrophil": ["S100A8", "S100A9", "CSF3R", "CXCR2", "MMP9"],
    "mesothelial": ["MSLN", "UPK3B", "LRRN4", "WT1"],
}
NON_REFERENCE = {"lymphocyte", "neutrophil", "mesothelial"}
# Plan section 8, frozen at T0.2 (human symbols).
STATES = {
    "state_scar_lipid_macrophage": ("macrophage", ["TREM2", "CD9", "SPP1", "GPNMB", "CD63", "FABP5", "LGALS3"]),
    "state_activated_hsc": ("hsc_fibroblast", ["COL1A1", "COL1A2", "COL3A1", "ACTA2", "TIMP1", "LUM",
                                                "PDGFRA", "PDGFRB"]),
    "state_stressed_hepatocyte": ("hepatocyte", ["ATF3", "DDIT3", "ATF4", "TRIB3", "GADD45A", "GADD45B",
                                                  "HSPA5", "SQSTM1"]),
}


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def lognorm(adata: ad.AnnData) -> ad.AnnData:
    import scanpy as sc

    a = ad.AnnData(X=adata.X.astype(np.float32), obs=adata.obs[[]].copy(), var=adata.var[[]].copy())
    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    return a


# --------------------------------------------------------------------------- loading


def _read_human_member(args):
    tar_path, name = args
    with tarfile.open(tar_path) as tf:
        with gzip.open(tf.extractfile(name)) as fh:
            df = pd.read_csv(fh, index_col=0, engine="pyarrow")
    X = sp.csr_matrix(df.to_numpy(dtype=np.int32).T)
    return name, df.index.astype(str).tolist(), df.columns.astype(str).tolist(), X


def load_human(jobs: int) -> ad.AnnData:
    tar_path = RAW / "GSE202379" / "GSE202379_RAW.tar"
    with tarfile.open(tar_path) as tf:
        names = sorted(m.name for m in tf.getmembers() if m.name.endswith("-raw_counts.csv.gz"))
    assert len(names) == 59, len(names)
    md = pd.read_csv(META, sep="\t", dtype=str).set_index("cell_barcode")
    parts, genes0 = [], None
    import multiprocessing as mp

    # spawn, not fork: a forked pool left the parent deadlocked in later threaded code
    with ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context("spawn")) as ex:
        for name, genes, cells, X in ex.map(_read_human_member, [(tar_path, n) for n in names]):
            if genes0 is None:
                genes0 = genes
            if genes != genes0:
                raise ValueError(f"{name}: gene rows differ from the first CSV")
            missing = [c for c in cells if c not in md.index]
            if missing:
                raise ValueError(f"{name}: {len(missing)} barcodes not in Seurat metadata")
            obs = md.loc[cells].copy()
            if obs["orig.ident"].nunique() != 1:
                raise ValueError(f"{name}: spans several orig.ident")
            obs["gsm"] = name.split("_")[0]
            obs["geo_file"] = name
            parts.append(ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=genes0)))
            log.info("human %s: %d nuclei", name, X.shape[0])
    adata = ad.concat(parts, join="inner", merge="same")
    adata.obs_names = adata.obs_names.astype(str)
    if adata.obs_names.duplicated().any():
        raise ValueError("duplicate human barcodes across files")
    o = adata.obs
    adata.obs = pd.DataFrame({
        "sample": o["orig.ident"].astype(str), "species": "Human", "donor": o["Patient.ID"].astype(str),
        "condition_label": o["Disease.status"].astype(str), "lobe": o["Lobe"].fillna("NA").astype(str),
        "sex": o["Gender"].astype(str), "age": pd.to_numeric(o["Age"], errors="coerce"),
        "bmi": pd.to_numeric(o["BMI"], errors="coerce"), "saf_score": o["SAF.Score"].astype(str),
        "fibrosis": o["Fibrosis.score..F0.4."].astype(str), "gsm": o["gsm"].astype(str),
        "author_label": o["cell.annotation"].astype(str), "author_cluster": o["seurat_clusters"].astype(str),
        "author_pct_mt": pd.to_numeric(o["percent.mt.RNA"], errors="coerce"),
    }, index=adata.obs_names)
    log.info("human: %d nuclei x %d genes, %d samples (Seurat metadata %d nuclei)",
             adata.n_obs, adata.n_vars, adata.obs["sample"].nunique(), len(md))
    return adata


def load_mouse() -> ad.AnnData:
    import scanpy as sc

    hold = pd.read_csv(HOLDOUTS, sep="\t", dtype=str)
    parts = []
    for _, r in hold.iterrows():
        f = next((RAW / "GSE192742_diet").glob(f"{r.gsm}_filtered_feature_bc_matrix_{r.shortfilename}.h5"))
        a = sc.read_10x_h5(f)
        a.var_names_make_unique()
        a.obs_names = [f"{r.shortfilename}_{b}" for b in a.obs_names]
        a.obs = pd.DataFrame({
            "sample": r.shortfilename, "species": "Mouse", "donor": r.shortfilename, "gsm": r.gsm,
            "diet": r.diet, "time_weeks": int(r.weeks), "role": r.role, "title": r.title,
            "condition_label": r.diet,
        }, index=a.obs_names)
        parts.append(a)
    var = parts[0].var.copy()
    for a in parts[1:]:
        if not a.var["gene_ids"].equals(var["gene_ids"]):
            raise ValueError("mouse h5 feature lists differ")
    adata = ad.concat(parts, join="inner")
    adata.var = var.loc[adata.var_names]
    adata.X = sp.csr_matrix(adata.X)
    log.info("mouse: %d nuclei x %d genes, %d samples", adata.n_obs, adata.n_vars, adata.obs["sample"].nunique())
    return adata


def qc(adata: ad.AnnData, name: str, use_author_mt: bool = False) -> tuple[ad.AnnData, dict]:
    _validate_raw_counts(adata, name)
    adata.obs["batch"] = adata.obs["sample"]  # batch = sample (else the helper would pick "donor")
    adata = _standardize_obs_columns(adata)
    n0 = adata.n_obs
    per0 = adata.obs["sample"].value_counts()
    adata = _qc_filter(adata, min_genes=MIN_GENES, max_mt_pct=MAX_MT)
    n1 = adata.n_obs
    if use_author_mt:
        adata = adata[adata.obs["author_pct_mt"].fillna(0) < MAX_MT].copy()
    per1 = adata.obs["sample"].value_counts()
    info = dict(n_in=int(n0), n_after_qc_filter=int(n1), n_after_author_mt=int(adata.n_obs),
                n_mt_genes=int(adata.var["mt"].sum()),
                per_sample=pd.DataFrame({"before": per0, "after": per1}).fillna(0).astype(int))
    log.info("%s QC: %d -> %d -> %d nuclei", name, n0, n1, adata.n_obs)
    return adata, info


# --------------------------------------------------------------------------- orthologs


def human_to_mouse(symbols: list[str], mouse_var: pd.DataFrame, table: pd.DataFrame) -> dict:
    """Human symbol -> mouse Ensembl ID: testis table 1:1, else Compara 110 one2one, else symbol case."""
    t = table.dropna(subset=["hsapiens_gene_name"]).drop_duplicates("hsapiens_gene_name", keep=False)
    t = t.set_index("hsapiens_gene_name")
    mouse_ids = set(mouse_var["gene_ids"])
    out, need = {}, []
    for s in symbols:
        if s in t.index and t.loc[s, "mmusculus_gene_id"] in mouse_ids:
            out[s] = (t.loc[s, "mmusculus_gene_id"], "testis_table")
        else:
            need.append(s)
    if need:
        hs = _human_ids(need)
        one2one = _compara_one2one(set(hs.values()))
        for s in need:
            mid = one2one.get(hs.get(s))
            if mid in mouse_ids:
                out[s] = (mid, "compara110_one2one")
            else:
                cand = mouse_var.index[mouse_var.index.str.upper() == s.upper()]
                if len(cand) == 1:
                    out[s] = (mouse_var.loc[cand[0], "gene_ids"], "symbol_case")
    return out


def _human_ids(symbols) -> dict:
    import re

    want, out = set(symbols), {}
    path = ROOT / "data" / "analysis" / "ensembl110" / "Homo_sapiens.GRCh38.110.gtf.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            if "\tgene\t" not in line:
                continue
            m = re.search(r'gene_name "([^"]+)"', line)
            if m and m.group(1) in want:
                out.setdefault(m.group(1), re.search(r'gene_id "([^"]+)"', line).group(1))
    return out


_COMPARA_CACHE: dict = {}


def _gtf_unique_name_to_id() -> dict:
    """Ensembl 110 gene_name -> gene_id where the name is unique (annotation for T7.3 joins)."""
    import re

    ids: dict = {}
    path = ROOT / "data" / "analysis" / "ensembl110" / "Homo_sapiens.GRCh38.110.gtf.gz"
    with gzip.open(path, "rt") as f:
        for line in f:
            if "\tgene\t" not in line:
                continue
            m = re.search(r'gene_name "([^"]+)"', line)
            if m:
                ids.setdefault(m.group(1), set()).add(re.search(r'gene_id "([^"]+)"', line).group(1))
    return {k: next(iter(v)) for k, v in ids.items() if len(v) == 1}


def _compara_one2one(human_ids: set) -> dict:
    if all(h in _COMPARA_CACHE for h in human_ids):
        return {h: _COMPARA_CACHE[h] for h in human_ids if _COMPARA_CACHE[h]}
    out = {}
    with gzip.open(COMPARA, "rt") as f:
        next(f)
        for line in f:
            if not line.startswith("ENSG"):
                continue
            p = line.split("\t", 9)
            if p[0] in human_ids and p[4] == "ortholog_one2one" and p[7] == "mus_musculus":
                out[p[0]] = p[5]
    _COMPARA_CACHE.update({h: out.get(h) for h in human_ids})
    return out


# --------------------------------------------------------------------------- annotation


def knn_transfer(ref: ad.AnnData, ref_labels: pd.Series, query: ad.AnnData) -> pd.DataFrame:
    """Reference-PCA kNN label transfer. ``ref`` and ``query`` share var_names (raw counts)."""
    import scanpy as sc
    from sklearn.decomposition import PCA
    from sklearn.neighbors import NearestNeighbors

    genes = ref.var_names.intersection(query.var_names)
    r = lognorm(ref[:, genes])
    q = lognorm(query[:, genes])
    sc.pp.highly_variable_genes(r, n_top_genes=N_HVG, flavor="seurat")
    hv = r.var_names[r.var["highly_variable"].to_numpy()]
    R = r[:, hv].X.toarray()
    mu, sd = R.mean(0), R.std(0)
    sd[sd == 0] = 1.0
    R = np.clip((R - mu) / sd, -10, 10)
    pca = PCA(n_components=N_PCS, random_state=0).fit(R)
    Q = np.clip((q[:, hv].X.toarray() - mu) / sd, -10, 10)
    nn = NearestNeighbors(n_neighbors=KNN).fit(pca.transform(R))
    idx = nn.kneighbors(pca.transform(Q), return_distance=False)
    lab = ref_labels.to_numpy()[idx]
    classes = np.array(sorted(set(ref_labels)))
    votes = np.stack([(lab == c).sum(axis=1) for c in classes], axis=1)
    best = classes[votes.argmax(axis=1)]
    frac = votes.max(axis=1) / KNN
    log.info("transfer: %d genes shared, %d HVG, %d query nuclei", len(genes), len(hv), query.n_obs)
    return pd.DataFrame({"transfer_label": best, "transfer_frac": frac},
                        index=query.obs_names)


def marker_scores(ln: ad.AnnData, markers: dict) -> pd.DataFrame:
    """Per nucleus mean log-normalised expression of each class's markers."""
    out = {}
    for c, genes in markers.items():
        g = [x for x in genes if x in ln.var_names]
        out[c] = np.asarray(ln[:, g].X.mean(axis=1)).ravel() if g else np.full(ln.n_obs, np.nan)
    return pd.DataFrame(out, index=ln.obs_names)


def marker_class(scores: pd.DataFrame, groups: pd.Series) -> pd.DataFrame:
    """Per group: mean marker score per class divided by that class's highest group mean.

    The marker class is the class with the largest ratio (1 = the group is the class's top
    group). A ratio rather than a z-score across groups, because one class (hepatocytes) makes
    up most groups and a z-score would push its own groups below zero.
    """
    m = scores.groupby(groups.to_numpy()).mean()
    rel = m / m.max().replace(0, np.nan)
    res = rel.copy()
    res.columns = [f"rel_{c}" for c in rel.columns]
    res["marker_class"] = rel.idxmax(axis=1)
    res["marker_rel"] = rel.max(axis=1)
    srt = np.sort(rel.fillna(0).to_numpy(), axis=1)
    res["marker_margin"] = srt[:, -1] - srt[:, -2]
    return res


def leiden_clusters(adata: ad.AnnData) -> pd.Series:
    import scanpy as sc

    a = lognorm(adata)
    sc.pp.highly_variable_genes(a, n_top_genes=N_HVG, flavor="seurat")
    a = a[:, a.var["highly_variable"].to_numpy()].copy()
    sc.pp.scale(a, max_value=10)
    sc.tl.pca(a, n_comps=N_PCS, random_state=0)
    # kNN graph built directly (scanpy's neighbors fails on this scipy's csr_array);
    # UMAP fuzzy connectivities as in scanpy, Leiden (modularity) in igraph.
    import random

    import igraph as ig
    from sklearn.neighbors import NearestNeighbors
    from umap.umap_ import fuzzy_simplicial_set

    pcs = a.obsm["X_pca"]
    dist, idx = NearestNeighbors(n_neighbors=15).fit(pcs).kneighbors(pcs)
    conn = fuzzy_simplicial_set(pcs, 15, random_state=0, metric="euclidean",
                                knn_indices=idx, knn_dists=dist)[0].tocoo()
    g = ig.Graph(n=a.n_obs, edges=list(zip(conn.row.tolist(), conn.col.tolist())), directed=True)
    g.es["weight"] = conn.data.tolist()
    g = g.as_undirected(mode="each")
    ig.set_random_number_generator(random.Random(0))
    part = g.community_leiden(objective_function="modularity", weights="weight",
                              resolution=LEIDEN_RES, n_iterations=2)
    return pd.Series(np.asarray(part.membership).astype(str), index=a.obs_names)


def annotate_mouse(m: ad.AnnData, table: pd.DataFrame, res: TaskOutput) -> ad.AnnData:
    ref = ad.read_h5ad(REF_MOUSE)
    ref = ref[ref.obs["annot"].isin(list(REF_MOUSE_MAP))].copy()
    ref.var_names = ref.var["gene_ids"].astype(str)
    ref = ref[:, ~ref.var_names.duplicated()].copy()
    q = m.copy()
    q.var_names = q.var["gene_ids"].astype(str)
    tr = knn_transfer(ref, ref.obs["annot"].map(REF_MOUSE_MAP), q)
    m.obs = m.obs.join(tr)
    m.obs["leiden"] = leiden_clusters(m).to_numpy()

    hm = human_to_mouse(sorted({g for v in COARSE_MARKERS.values() for g in v}), m.var, table)
    id2sym = pd.Series(m.var_names, index=m.var["gene_ids"].to_numpy())
    mk = {c: [id2sym[hm[g][0]] for g in v if g in hm] for c, v in COARSE_MARKERS.items()}
    res.json({c: {"human": COARSE_MARKERS[c], "mouse": mk[c]} for c in mk}, "mouse_coarse_markers.json")
    ln = lognorm(m)
    sc_ = marker_scores(ln, mk)
    cl = marker_class(sc_, m.obs["leiden"])
    tmaj = m.obs.groupby("leiden")["transfer_label"].agg(lambda s: s.value_counts().index[0])
    tfrac = m.obs.groupby("leiden")["transfer_label"].agg(lambda s: s.value_counts(normalize=True).iloc[0])
    cl["n"] = m.obs["leiden"].value_counts()
    cl["transfer_majority"] = tmaj
    cl["transfer_majority_frac"] = tfrac
    cl["mean_transfer_conf"] = m.obs.groupby("leiden")["transfer_frac"].mean()

    def rule(r):
        if r.marker_margin < MIN_MARGIN:
            return "unresolved", "marker_margin_below_min"
        if r.transfer_majority == r.marker_class and r.transfer_majority_frac >= MIN_TRANSFER_FRAC:
            return r.transfer_majority, "agree"
        if r.marker_class in NON_REFERENCE:
            return r.marker_class, "marker_class_absent_from_reference"
        return "unresolved", "disagree_or_weak_transfer"

    cl[["final", "decision"]] = cl.apply(lambda r: pd.Series(rule(r)), axis=1)
    m.obs["cell_ontology_class"] = m.obs["leiden"].map(cl["final"]).to_numpy()
    m.obs["annotation_decision"] = m.obs["leiden"].map(cl["decision"]).to_numpy()
    cl = cl.reset_index().rename(columns={"index": "leiden"})
    res.csv(cl, "mouse_cluster_annotation.csv")
    agree = pd.crosstab(m.obs["transfer_label"], m.obs["cell_ontology_class"])
    res.csv(agree, "mouse_transfer_vs_final.csv", index=True)
    # marker check on final labels
    fin = marker_class(sc_, m.obs["cell_ontology_class"])
    res.csv(fin.reset_index().rename(columns={"index": "cell_ontology_class"}), "mouse_marker_check_final.csv")
    return m


def annotate_human(h: ad.AnnData, res: TaskOutput) -> ad.AnnData:
    h.obs["cell_ontology_class"] = h.obs["author_label"].map(HUMAN_AUTHOR_MAP).fillna("unassigned").to_numpy()
    ref = ad.read_h5ad(REF_HUMAN)
    ref = ref[ref.obs["annot"].isin(list(REF_HUMAN_MAP))].copy()
    ref.var_names_make_unique()
    tr = knn_transfer(ref, ref.obs["annot"].map(REF_HUMAN_MAP), h)
    h.obs = h.obs.join(tr)
    res.csv(pd.crosstab(h.obs["author_label"], h.obs["transfer_label"]), "human_author_vs_transfer.csv", index=True)
    ln = lognorm(h)
    sc_ = marker_scores(ln, COARSE_MARKERS)
    fin = marker_class(sc_, h.obs["author_label"])
    res.csv(fin.reset_index().rename(columns={"index": "author_label"}), "human_marker_check_author.csv")
    return h


# --------------------------------------------------------------------------- primary sets, states


def mark_primary(h: ad.AnnData, m: ad.AnnData) -> None:
    o = h.obs
    healthy = (o["condition_label"] == "Healthy control") & (o["donor"] != P98)
    healthy &= ~((o["donor"] == HL1) & (o["lobe"] != "Right"))
    mash = o["condition_label"] == "NASH w/o cirrhosis"
    o["condition"] = np.where(o["condition_label"] == "Healthy control", "control",
                              np.where(mash, "disease", "other"))
    o["primary"] = (healthy | mash).to_numpy()
    o["role"] = np.where(o["primary"], "training", "not_primary")
    mo = m.obs
    mo["condition"] = np.where(mo["diet"] == "SD", "control", "disease")
    mo["primary"] = True
    mo["time_weeks"] = mo["time_weeks"].astype(int)


def score_states(adata: ad.AnnData, species: str, gene_map: dict, res_rows: list) -> None:
    """Frozen state scores within coarse type on the species' training nuclei; top quartile = state."""
    import scanpy as sc

    train = (adata.obs["role"] == "training").to_numpy()
    for name, (ctype, genes) in STATES.items():
        adata.obs[name.replace("state_", "score_")] = np.nan
        adata.obs[name] = False
        sel = train & (adata.obs["cell_ontology_class"] == ctype).to_numpy()
        sub = lognorm(adata[sel])
        g = [gene_map[x] for x in genes if x in gene_map and gene_map[x] in sub.var_names]
        sc.tl.score_genes(sub, g, score_name="s", random_state=0)
        thr = float(np.quantile(sub.obs["s"], 0.75))
        adata.obs.loc[sub.obs_names, name.replace("state_", "score_")] = sub.obs["s"].to_numpy()
        adata.obs.loc[sub.obs_names, name] = (sub.obs["s"] >= thr).to_numpy()
        res_rows.append(dict(species=species, state=name, coarse_type=ctype, n_markers=len(genes),
                             n_found=len(g), markers_used=",".join(g),
                             missing=",".join(x for x in genes if x not in gene_map or gene_map[x] not in sub.var_names),
                             n_nuclei=int(sel.sum()), threshold_q75=thr,
                             n_state=int((sub.obs["s"] >= thr).sum())))


# --------------------------------------------------------------------------- pseudobulks, gate


def pseudobulk(adata: ad.AnnData, keys: list[str]) -> ad.AnnData:
    o = adata.obs[adata.obs["cell_ontology_class"].isin(["unresolved", "unassigned"]) == False]  # noqa: E712
    sub = adata[o.index]
    grp = sub.obs["sample"].astype(str) + "|" + sub.obs["cell_ontology_class"].astype(str)
    codes, uniq = pd.factorize(grp, sort=True)
    G = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
    X = sub.X.tocsr().astype(np.float64)
    counts = (G @ X).toarray()
    det = (G @ (X > 0).astype(np.float64)).toarray()
    first = sub.obs.groupby(grp.to_numpy(), sort=True)[keys].first()
    obs = first.loc[uniq].copy()
    obs["cell_ontology_class"] = [u.split("|")[1] for u in uniq]
    obs["n_nuclei"] = np.asarray(G.sum(axis=1)).ravel().astype(int)
    obs.index = uniq
    pb = ad.AnnData(X=sp.csr_matrix(counts.astype(np.int64)), obs=obs, var=sub.var[[]].copy())
    pb.layers["n_detected"] = sp.csr_matrix(det.astype(np.int32))
    return pb


def gate_gd1(h: ad.AnnData, m: ad.AnnData) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    def units(obs, unit_col):
        obs = obs.assign(**{unit_col: obs[unit_col].astype(str),
                            "cell_ontology_class": obs["cell_ontology_class"].astype(str)})
        return obs.groupby([unit_col, "cell_ontology_class"]).size().unstack(fill_value=0)

    ho = h.obs
    prim = ho[ho["primary"]]
    hu = units(prim, "donor").reindex(columns=PRIMARY, fill_value=0)
    healthy_d = sorted(prim.loc[prim["condition"] == "control", "donor"].unique())
    mash_d = sorted(prim.loc[prim["condition"] == "disease", "donor"].unique())
    # sensitivity unit sets
    hc = ho[ho["condition_label"] == "Healthy control"]
    hu_p98 = units(hc[~((hc["donor"] == HL1) & (hc["lobe"] != "Right"))], "donor").reindex(columns=PRIMARY, fill_value=0)
    hu_lobes = units(hc[hc["donor"] != P98], "donor").reindex(columns=PRIMARY, fill_value=0)
    mo = m.obs
    mu = units(mo, "sample").reindex(columns=PRIMARY, fill_value=0)
    roles = mo.groupby("sample")[["diet", "role"]].first()
    sd_tr = roles.index[(roles.diet == "SD") & (roles.role == "training")]
    wd_tr = roles.index[(roles.diet == "WD") & (roles.role == "training")]
    sd_all, wd_all = roles.index[roles.diet == "SD"], roles.index[roles.diet == "WD"]

    def n_ok(tab, ids, t):
        return int((tab.reindex(ids, fill_value=0)[t] >= MIN_NUCLEI).sum())

    rows = []
    for t in PRIMARY:
        r = dict(cell_type=t,
                 healthy_ok=n_ok(hu, healthy_d, t), healthy_n=len(healthy_d),
                 mash_ok=n_ok(hu, mash_d, t), mash_n=len(mash_d),
                 sd_train_ok=n_ok(mu, sd_tr, t), sd_train_n=len(sd_tr),
                 wd_train_ok=n_ok(mu, wd_tr, t), wd_train_n=len(wd_tr),
                 sd_all_ok=n_ok(mu, sd_all, t), wd_all_ok=n_ok(mu, wd_all, t),
                 healthy_with_p98_ok=n_ok(hu_p98, hu_p98.index, t), healthy_with_p98_n=len(hu_p98),
                 healthy_all_hl1_lobes_ok=n_ok(hu_lobes, hu_lobes.index, t))
        r["min_healthy_nuclei"] = int(hu.reindex(healthy_d)[t].min())
        r["min_train_mouse_nuclei"] = int(mu.reindex(list(sd_tr) + list(wd_tr))[t].min())
        r["pseudobulkable_primary"] = bool(r["healthy_ok"] >= 3 and r["mash_ok"] >= 15
                                           and r["sd_train_ok"] >= 4 and r["wd_train_ok"] >= 4)
        r["pseudobulkable_strict"] = bool(r["healthy_ok"] == r["healthy_n"] and r["mash_ok"] == r["mash_n"]
                                          and r["sd_train_ok"] == r["sd_train_n"] and r["wd_train_ok"] == r["wd_train_n"])
        r["pseudobulkable_sens_p98_all_mice"] = bool(r["healthy_with_p98_ok"] >= 3 and r["mash_ok"] >= 15
                                                    and r["sd_all_ok"] >= 4 and r["wd_all_ok"] >= 4)
        rows.append(r)
    types = pd.DataFrame(rows)
    n_types = int(types["pseudobulkable_primary"].sum())
    crit = pd.DataFrame([
        dict(criterion="healthy donors >= 3 (P98 excluded)", value=len(healthy_d), threshold=3,
             passed=len(healthy_d) >= 3, detail=",".join(healthy_d)),
        dict(criterion="MASH donors >= 15 (NASH w/o cirrhosis)", value=len(mash_d), threshold=15,
             passed=len(mash_d) >= 15, detail=f"{len(mash_d)} donors"),
        dict(criterion="training mice >= 4 SD + 4 WD", value=f"{len(sd_tr)}+{len(wd_tr)}", threshold="4+4",
             passed=len(sd_tr) >= 4 and len(wd_tr) >= 4, detail=",".join(list(sd_tr) + list(wd_tr))),
        dict(criterion=">= 4 of 5 types pseudobulk-able at >= 20 nuclei per unit", value=n_types, threshold=4,
             passed=n_types >= 4,
             detail=",".join(types.loc[types["pseudobulkable_primary"], "cell_type"])),
    ])
    unit_tab = pd.concat({"human_primary_donor": hu, "human_healthy_with_p98": hu_p98,
                          "human_healthy_all_hl1_lobes": hu_lobes, "mouse": mu}, names=["unit_set", "unit"])
    return crit, types, unit_tab


# --------------------------------------------------------------------------- pooled object


def cap_nuclei(obs: pd.DataFrame, rng: np.random.Generator) -> pd.Index:
    keep = []
    obs = obs[["species", "condition", "cell_ontology_class", "donor"]].astype(str)
    for _, g in obs.groupby(["species", "condition", "cell_ontology_class"], sort=True):
        by = g.groupby("donor").groups
        avail = {d: len(ix) for d, ix in sorted(by.items())}
        target = min(CAP, sum(avail.values()))
        quota = {d: 0 for d in avail}
        left = target
        while left > 0:  # water-filling, one nucleus at a time per open donor in fixed order
            open_ = [d for d in avail if quota[d] < avail[d]]
            share = max(left // len(open_), 1)
            for d in open_:
                add = min(share, avail[d] - quota[d], left)
                quota[d] += add
                left -= add
                if left == 0:
                    break
        for d, ix in sorted(by.items()):
            ix = np.array(sorted(ix))
            keep.extend(rng.choice(ix, size=quota[d], replace=False).tolist())
    return pd.Index(sorted(keep))


def build_pooled(h: ad.AnnData, m: ad.AnnData, table: pd.DataFrame, res: TaskOutput) -> tuple[ad.AnnData, dict]:
    obs_cols = ["sample", "donor", "species", "condition", "condition_label", "cell_ontology_class", "role",
                "sex", "lobe", "diet", "time_weeks", *STATES, *[s.replace("state_", "score_") for s in STATES]]
    hs = h[(h.obs["role"] == "training").to_numpy()].copy()
    ms = m[(m.obs["role"] == "training").to_numpy()].copy()
    for a in (hs, ms):
        for c in obs_cols:
            if c not in a.obs:
                a.obs[c] = np.nan if c == "time_weeks" else "NA"
    hs.obs["time_weeks"] = np.nan
    # coarse classes: the five primary types plus classes with >= MIN_OTHER training nuclei in both species
    ch, cm = hs.obs["cell_ontology_class"].value_counts(), ms.obs["cell_ontology_class"].value_counts()
    extra = sorted(c for c in set(ch.index) & set(cm.index)
                   if c not in PRIMARY and c not in ("unresolved", "unassigned")
                   and ch[c] >= MIN_OTHER and cm[c] >= MIN_OTHER)
    classes = PRIMARY + extra
    hs = hs[hs.obs["cell_ontology_class"].isin(classes).to_numpy()].copy()
    ms = ms[ms.obs["cell_ontology_class"].isin(classes).to_numpy()].copy()
    rng = np.random.default_rng(SEED)
    hs = hs[cap_nuclei(hs.obs, rng)].copy()
    ms = ms[cap_nuclei(ms.obs, rng)].copy()

    # gene IDs: human symbol -> ENSG through unique table names; mouse by Ensembl ID
    t_name = table.dropna(subset=["hsapiens_gene_name"])
    t_name = t_name[~t_name["hsapiens_gene_name"].duplicated(keep=False)]
    sym2ens = dict(zip(t_name["hsapiens_gene_name"], t_name["hsapiens_gene_id"]))
    hkeep = [g for g in hs.var_names if g in sym2ens]
    hs = hs[:, hkeep].copy()
    hs.var_names = [sym2ens[g] for g in hkeep]
    ms.var_names = ms.var["gene_ids"].astype(str).to_numpy()
    ms = ms[:, ~ms.var_names.duplicated()].copy()
    human_present = set(hs.var_names)
    for a in (hs, ms):
        a.obs = a.obs[obs_cols].copy()
        a.var = a.var[[]]

    # features exactly as notebooks/01_data_preparation.ipynb section 8 (testis)
    prior = compute_conservation_prior(table)
    ens_species = ["hsapiens", "mmulatta", "mmusculus"]
    consensus, per_sp = download_and_process_esm2(table, species_list=ens_species, saturn_data_dir=SATURN)
    esm_pca = pca_esm2(consensus, n_components=64)
    esm_var = compute_esm2_interspecies_variance(per_sp)
    G = len(table)
    unified = construct_unified_adata({"hsapiens": hs, "mmusculus": ms}, table,
                                      np.zeros((G, 72), dtype=np.float32), prior)
    unified.X = sp.csr_matrix(unified.X, dtype=np.float32)  # canonical h5ads hold float32 CSR counts
    stats = compute_expression_stats(unified)
    feats, scaler = assemble_gin_features(esm_pca, esm_var, table, stats)
    unified.varm["gene_features"] = feats
    unified.uns["p_null"] = precompute_p_null(unified)
    for c in ("species", "batch", "cell_ontology_class", "condition", "condition_label", "donor", "role",
              "sex", "lobe", "diet"):
        if c == "batch":
            unified.obs["batch"] = unified.obs["sample"].astype(str)
        unified.obs[c] = unified.obs[c].astype(str).astype("category")
    for s in STATES:
        unified.obs[s] = unified.obs[s].astype(bool)

    # structural features must equal the canonical testis features (same table, same species)
    canon = ad.read_h5ad(CANON_TESTIS, backed="r")
    cf = np.asarray(canon.varm["gene_features"])
    struct = [i for i in range(72) if i not in (66, 67, 68)]
    max_diff = float(np.abs(cf[:, struct] - feats[:, struct]).max())
    canon.file.close()

    # drop genes absent from the human GEO matrices and genes in < 3 nuclei (build_subset rule)
    gid = table["hsapiens_gene_id"].to_numpy()
    present = np.array([g in human_present for g in gid])
    nnz = np.asarray((unified.X != 0).sum(axis=0)).ravel()
    gene_mask = present & (nnz >= 3)
    full_path = OUT / "disease_liver_pooled_full_vocab.h5ad"
    _write_h5ad(unified, full_path)
    pooled = unified[:, gene_mask].copy()
    pooled.obs["species"] = pooled.obs["species"].astype(str).astype("category")
    pooled.obs["batch"] = pooled.obs["batch"].astype(str).astype("category")
    pooled.uns["p_null"] = precompute_p_null(pooled, species_key="species")
    info = dict(classes=classes, extra_classes=extra, n_genes_table=G,
                n_genes_absent_in_human_matrix=int((~present).sum()),
                n_genes_lt3_nuclei_among_present=int((present & (nnz < 3)).sum()),
                n_genes_pooled=int(gene_mask.sum()), max_abs_diff_struct_features_vs_canonical_testis=max_diff,
                frac_prior_ge_095=float((pooled.var["conservation_prior"] >= 0.95).mean()),
                frac_table_identity_ge_95=float((table["mean_identity"] >= 95).mean()))
    with open(OUT / "disease_liver_pooled_gin_scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    return pooled, info


def _write_h5ad(adata: ad.AnnData, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    adata.write_h5ad(tmp)
    tmp.replace(path)


# --------------------------------------------------------------------------- main


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(p)
    p.add_argument("--reannotate", action="store_true", help="redo QC and annotation even if cached")
    p.add_argument("--jobs", type=int, default=12)
    a = p.parse_args()
    res = TaskOutput("T7_2", a.outdir, prerun=a.prerun, force=a.force)
    OUT.mkdir(parents=True, exist_ok=True)
    table = pd.read_csv(TABLE)
    if len(table) != 15732:
        raise ValueError(f"testis table has {len(table)} rows, expected 15,732")

    ann_h, ann_m = OUT / "annotated_human.h5ad", OUT / "annotated_mouse.h5ad"
    if a.reannotate or not (ann_h.exists() and ann_m.exists()):
        h, qh = qc(load_human(a.jobs), "GSE202379", use_author_mt=True)
        m, qm = qc(load_mouse(), "GSE192742")
        res.csv(qh.pop("per_sample"), "qc_human_per_sample.csv", index=True)
        res.csv(qm.pop("per_sample"), "qc_mouse_per_sample.csv", index=True)
        res.json(dict(human=qh, mouse=qm, min_genes=MIN_GENES, max_mt_pct=MAX_MT), "qc_summary.json")
        h = annotate_human(h, res)
        m = annotate_mouse(m, table, res)
        mark_primary(h, m)
        rows = []
        hmap = {g: g for _, gs in STATES.values() for g in gs}
        score_states(h, "Human", hmap, rows)
        mm = human_to_mouse(sorted(hmap), m.var, table)
        id2sym = pd.Series(m.var_names, index=m.var["gene_ids"].to_numpy())
        score_states(m, "Mouse", {g: id2sym[v[0]] for g, v in mm.items()}, rows)
        res.csv(pd.DataFrame(rows), "state_scoring.csv")
        res.json({g: list(v) for g, v in mm.items()}, "state_markers_mouse_orthologs.json")
        for x, path in ((h, ann_h), (m, ann_m)):
            for c in x.obs.columns:
                if x.obs[c].dtype == object:
                    x.obs[c] = x.obs[c].astype(str)
            _write_h5ad(x, path)
    else:
        h, m = ad.read_h5ad(ann_h), ad.read_h5ad(ann_m)

    # counts per type x condition x species (all QC-passed nuclei, and training/primary only)
    cnt = pd.concat([x.obs[["species", "condition_label", "role", "cell_ontology_class"]] for x in (h, m)])
    res.csv(cnt.value_counts().rename("n_nuclei").reset_index()
            .sort_values(["species", "role", "condition_label", "cell_ontology_class"]), "counts_all_nuclei.csv")

    # pseudobulks (all samples, no caps)
    pb_h = pseudobulk(h, ["sample", "donor", "species", "condition", "condition_label", "primary", "role", "lobe",
                          "sex", "age", "bmi", "saf_score", "fibrosis", "gsm"])
    pb_h.var["gene_symbol"] = pb_h.var_names
    pb_h.var["gene_id_ens110"] = pb_h.var_names.map(_gtf_unique_name_to_id()).astype(object)
    pb_m = pseudobulk(m, ["sample", "donor", "species", "condition", "condition_label", "role", "diet",
                          "time_weeks", "gsm"])
    pb_m.var["gene_symbol"] = pb_m.var_names
    pb_m.var["gene_id"] = m.var["gene_ids"].reindex(pb_m.var_names).to_numpy()
    for c in pb_h.obs.columns:
        if pb_h.obs[c].dtype == object:
            pb_h.obs[c] = pb_h.obs[c].astype(str)
    paths = {"pseudobulk_human": OUT / "disease_pseudobulk_human.h5ad",
             "pseudobulk_mouse": OUT / "disease_pseudobulk_mouse.h5ad"}
    _write_h5ad(pb_h, paths["pseudobulk_human"])
    _write_h5ad(pb_m, paths["pseudobulk_mouse"])

    crit, types, units = gate_gd1(h, m)
    res.csv(crit, "gd1_criteria.csv")
    res.csv(types, "gd1_types.csv")
    res.csv(units, "gd1_units.csv", index=True)
    gd1_pass = bool(crit["passed"].all())
    log.info("G-D1 %s\n%s\n%s", "PASS" if gd1_pass else "FAIL", crit.to_string(), types.to_string())

    pooled, info = build_pooled(h, m, table, res)
    rho = gene_ids.assert_table_matches(pooled, gene_ids.load_table("testis"))
    paths["pooled"] = OUT / "disease_liver_pooled.h5ad"
    paths["pooled_full_vocab"] = OUT / "disease_liver_pooled_full_vocab.h5ad"
    _write_h5ad(pooled, paths["pooled"])
    pc = pooled.obs.groupby(["species", "condition", "cell_ontology_class"], observed=True).size()
    res.csv(pc.rename("n_nuclei").reset_index(), "counts_pooled.csv")
    res.csv(pooled.obs.groupby(["species", "sample"], observed=True).size().rename("n_nuclei").reset_index(),
            "counts_pooled_per_sample.csv")
    st = pooled.obs.groupby(["species", "condition"], observed=True)[list(STATES)].mean()
    res.csv(st.reset_index(), "pooled_state_fractions.csv")
    info.update(assert_table_matches_spearman=rho, n_obs=int(pooled.n_obs), n_vars=int(pooled.n_vars),
                p_null_shape=list(pooled.uns["p_null"].shape), gd1_pass=gd1_pass, cap=CAP, seed=SEED,
                leiden_res=LEIDEN_RES, knn=KNN)
    res.json(info, "pooled_summary.json")
    paths["gin_scaler"] = OUT / "disease_liver_pooled_gin_scaler.pkl"
    paths["annotated_human"], paths["annotated_mouse"] = ann_h, ann_m
    out_hash = {k: dict(path=str(v.relative_to(ROOT)), bytes=v.stat().st_size, sha256=sha256(v))
                for k, v in paths.items()}
    in_paths = [RAW / "GSE202379" / "GSE202379_RAW.tar", META, HOLDOUTS, REF_MOUSE, REF_HUMAN, TABLE,
                *sorted((RAW / "GSE192742_diet").glob("*.h5"))]
    in_hash = {str(x.relative_to(ROOT)): sha256(x) for x in in_paths}
    res.json(dict(outputs=out_hash, inputs=in_hash, code_sha=git_sha()), "hashes.json")
    res.finish(dict(gd1_pass=gd1_pass))


if __name__ == "__main__":
    sys.exit(main())
