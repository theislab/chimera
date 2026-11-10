"""Build the bone-marrow cohort (human, mouse lemur, mouse) from the CELLxGENE files.

Pre-specified analysis pre-specified analysis plan, section 12. Inputs (``data/analysis/bone_marrow/raw/``, see
MANIFEST.tsv): Tabula Sapiens Bone_Marrow and Spleen, Tabula Microcebus Bone marrow and
Spleen, Tabula Muris Senis Bone marrow and Spleen (10x). Counts are ``raw/X`` (integer UMIs).

Steps, in order:
1. Cells: 10x 3' only (human 10x 3' v3; lemur and mouse 10x 3' v2). Smart-seq2 read counts and
   10x 5' libraries are not on the same count scale as 3' UMIs.
2. QC per file: ``chimera._data_utils._qc_filter`` with min 200 genes and no mitochondrial
   filter in any species (``max_mt_pct=100``), a deviation from the canonical 20%: the lemur
   and mouse CELLxGENE matrices carry no mitochondrial genes, so the 20% rule acted in human
   only (1,137 human cells in build 1) and cannot act evenly across species. Same decision as
   Van Zyl (``T11_1_vanzyl/build_vanzyl.py``) and kidney (``P13_kidney/build.py``). The
   human mitochondrial percentage stays in ``obs['pct_counts_mt']`` for disclosure.
3. Labels: the committed map ``celltype_map.tsv`` (CELLxGENE ``cell_type`` to the nine
   harmonized types; ``DROP`` rows removed). Every label present must be in the map.
4. Spleen rule (pre-specified analysis plan, section 12): if lemur bone-marrow B or NK has fewer than 150 cells after steps
   1-3, the spleen files of all three atlases are added. Decided on these counts only.
5. Caps (build 2, T12_2b_recipe): every species x type, bone marrow and spleen together, at ``CAP``
   cells, drawn uniformly without replacement (numpy default_rng(SEED) over sorted cell IDs).
   ``CAP`` is the largest multiple of 50 for which the capped total does not exceed
   ``SIZE_REF`` = 28,854 cells, the size of the testis 3-species cohort
   (``data/processed/camex_testis_unified.h5ad``) on which the base recipe and its epoch-based
   schedule (KL and FQR warm-ups at epoch 20, adversary at 5, plateau LR decay) were set.
   Reason: at 128,328 cells (build 1, 452 steps per epoch against 29-102 in the canonical
   cohorts) training collapsed the latent or went NaN, and a proportional 25k draw of the same
   cells did not (diagnostics of analysis/tasks/T12_2b_debug). One per-type cap also
   keeps any single type from dominating the cohort: after the spleen addition B and T, not
   neutrophil and erythroid, were the largest classes, so build 1's neutrophil/erythroid-only
   cap (5,000) had lost its reason. Decided on cell counts only.
6. Pipeline, unchanged from ``chimera._data_utils``: ``compute_conservation_prior``,
   ``download_and_process_esm2``, ``pca_esm2``, ``compute_esm2_interspecies_variance``,
   ``construct_unified_adata``, ``compute_expression_stats``, ``assemble_gin_features``,
   ``precompute_p_null``; ortholog table ``ortholog_table_bone_marrow.csv`` (build_orthologs.py).
7. ``analysis.common.gene_ids.assert_table_matches`` on the result.

obs: species (human / mouse_lemur / mouse), batch (species|donor|tissue: one batch per donor and
tissue, so mouse age, which is a donor property, enters through batch), cell_type (CELLxGENE
label), cell_ontology_class and cell_type_harmonised (the nine types), tissue, donor_id,
sample_id, assay, age_months (mouse; NaN otherwise), development_stage, sex.

Writes ``bone_marrow_unified.h5ad``, ``gin_scaler_bone_marrow.pkl``, ``build_report.json`` and
count tables to ``data/analysis/bone_marrow/v2/`` (or ``--out``), never to ``data/processed/``.
Build 1 (20% mt filter, neutrophil/erythroid cap 5,000; 128,328 cells) stays in
``data/analysis/bone_marrow/`` for the record of the T12_2_recipe grid that was trained on it.

    python analysis/tasks/T12_1_bone_marrow/build_bm.py [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from chimera import _data_utils as du  # noqa: E402
from analysis.common import gene_ids  # noqa: E402

TASK = Path(__file__).resolve().parent
BM = REPO / "data" / "analysis" / "bone_marrow"
RAW = BM / "raw"
TABLE = BM / "ortholog_table_bone_marrow.csv"
ESM2 = REPO / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"

# Ensembl short name -> (obs species label, assay kept, {tissue: file})
SPECIES = {
    "hsapiens": ("human", "10x 3' v3", {"bone marrow": "human_TS_bone_marrow.h5ad",
                                        "spleen": "human_TS_spleen.h5ad"}),
    "mmurinus": ("mouse_lemur", "10x 3' v2", {"bone marrow": "lemur_TMicrocebus_bone_marrow.h5ad",
                                              "spleen": "lemur_TMicrocebus_spleen.h5ad"}),
    "mmusculus": ("mouse", "10x 3' v2", {"bone marrow": "mouse_TMS_bone_marrow_10x.h5ad",
                                         "spleen": "mouse_TMS_spleen_10x.h5ad"}),
}
# ESM-2 files, linked under names the loader's filename match resolves to one species each
# (the export's "mouse_lemur_embedding.torch" would otherwise also match "mouse").
ESM2_LINKS = {"human_embedding.torch": "human_embedding.torch",
              "mmurinus_embedding.torch": "mouse_lemur_embedding.torch",
              "mouse_embedding.torch": "mouse_embedding.torch"}
QC = dict(min_genes=200, max_mt_pct=100.0)  # no mt filter in any species, see module docstring
TYPES = ["HSPC", "neutrophil", "monocyte", "macrophage", "B", "plasma", "T", "NK", "erythroid"]
CAPPED = tuple(TYPES)  # every type (build 2); build 1 capped neutrophil and erythroid at 5,000
SIZE_REF = 28854  # testis 3-species cohort, cells (camex_testis_unified.h5ad)
CAP_STEP = 50
SEED = 42
SPLEEN_MIN = 150  # lemur bone-marrow B or NK below this -> add spleen (plan, section 12)
OBS_KEEP = ["donor_id", "sample_id", "assay", "development_stage", "sex", "cell_type",
            "cell_type_ontology_term_id", "age"]

log = logging.getLogger("build_bm")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def content_hash(adata) -> dict:
    """Hashes of the parts of an AnnData that training reads, independent of file layout."""
    import scipy.sparse as sp

    def h(*arrays) -> str:
        m = hashlib.sha256()
        for a in arrays:
            a = np.ascontiguousarray(a)
            m.update(str(a.dtype).encode() + str(a.shape).encode())
            m.update(a.tobytes())
        return m.hexdigest()

    X = adata.X if sp.issparse(adata.X) else sp.csr_matrix(adata.X)
    X = X.tocsr()
    X.sort_indices()
    obs = adata.obs.astype(str)
    return {
        "X": h(X.data, X.indices, X.indptr),
        "obs_names": h(np.asarray(adata.obs_names, dtype="U")),
        "var_names": h(np.asarray(adata.var_names, dtype="U")),
        "obs": h(np.asarray(obs.values, dtype="U"), np.asarray(obs.columns, dtype="U")),
        "conservation_prior": h(adata.var["conservation_prior"].to_numpy()),
        "gene_features": h(adata.varm["gene_features"]),
        "p_null": h(np.asarray(adata.uns["p_null"])),
    }


def load_label_map() -> pd.DataFrame:
    cmap = pd.read_csv(TASK / "celltype_map.tsv", sep="\t", keep_default_na=False)
    if cmap.duplicated(["species", "source_label"]).any():
        raise ValueError("duplicate (species, source_label) in celltype_map.tsv")
    bad = set(cmap["cell_type_harmonised"]) - set(TYPES) - {"DROP"}
    if bad:
        raise ValueError(f"unknown harmonised types in map: {bad}")
    return cmap


def load_file(ens: str, tissue: str, cmap: pd.DataFrame) -> tuple:
    """One CELLxGENE file -> raw-count AnnData of mapped cells, and a step-count record."""
    import anndata as ad
    import scipy.sparse as sp

    label, assay, files = SPECIES[ens]
    fname = files[tissue]
    a = ad.read_h5ad(RAW / fname)
    rec = {"file": fname, "n_cells_file": int(a.n_obs)}
    keep = (a.obs["assay"].astype(str) == assay).to_numpy()
    rec["assays_in_file"] = a.obs["assay"].astype(str).value_counts().to_dict()
    raw = a.raw
    X = raw.X[keep]
    X = sp.csr_matrix(X) if not sp.issparse(X) else X.tocsr()
    obs = a.obs.loc[keep, [c for c in OBS_KEEP if c in a.obs.columns]].copy()
    for c in obs.columns:
        obs[c] = obs[c].astype(str)
    tag = {"bone marrow": "BM", "spleen": "SPL"}[tissue]
    obs.index = [f"{label}:{tag}:{i}" for i in obs.index]
    obs["tissue"] = tissue
    obs["species"] = label
    var = pd.DataFrame(index=pd.Index(raw.var_names.astype(str), name=None))
    var["feature_name"] = raw.var["feature_name"].astype(str).to_numpy()
    sub = ad.AnnData(X=X.astype(np.float32), obs=obs, var=var)
    del a, raw, X
    rec["n_cells_assay"] = int(sub.n_obs)
    du._validate_raw_counts(sub, f"{label} {tissue}")

    # QC on symbol names (mitochondrial prefixes), mapped back to the Ensembl-ID object.
    q = sub.copy()
    q.var_names = pd.Index(sub.var["feature_name"].to_numpy())
    q.var_names_make_unique()
    rec["mt_genes"] = int(q.var_names.str.startswith(("MT-", "Mt-", "mt-", "MTND", "MTCO", "MTATP")).sum())
    q = du._qc_filter(q, **QC)
    sub = sub[q.obs_names].copy()
    for c in ("n_genes", "n_genes_by_counts", "total_counts", "total_counts_mt", "pct_counts_mt"):
        if c in q.obs:
            sub.obs[c] = q.obs[c].to_numpy()
    del q
    rec["n_cells_after_qc"] = int(sub.n_obs)

    # Labels.
    m = cmap[cmap["species"] == label].set_index("source_label")["cell_type_harmonised"]
    lab = sub.obs["cell_type"].map(m)
    if lab.isna().any():
        raise ValueError(f"{label} {tissue}: labels not in celltype_map.tsv: "
                         f"{sorted(sub.obs.loc[lab.isna(), 'cell_type'].unique())}")
    rec["dropped_by_label"] = sub.obs.loc[lab == "DROP", "cell_type"].value_counts().to_dict()
    sub = sub[(lab != "DROP").to_numpy()].copy()
    sub.obs["cell_type_harmonised"] = lab[lab != "DROP"].to_numpy()
    rec["n_cells_mapped"] = int(sub.n_obs)
    rec["mapped_by_type"] = sub.obs["cell_type_harmonised"].value_counts().reindex(TYPES, fill_value=0).astype(int).to_dict()
    log.info("%s %s: %s", label, tissue, rec)
    return sub, rec


def choose_cap(counts: dict) -> int:
    """Largest multiple of CAP_STEP whose capped total (all species x types) is <= SIZE_REF."""
    n = np.array([v for per in counts.values() for v in per.values()], dtype=int)
    c = CAP_STEP
    while np.minimum(n, c + CAP_STEP).sum() <= SIZE_REF and c + CAP_STEP <= n.max():
        c += CAP_STEP
    return c


def cap(adata, rng_seed: int, cap_n: int) -> tuple:
    keep = np.ones(adata.n_obs, dtype=bool)
    rec = {}
    names = np.asarray(adata.obs_names)
    for t in CAPPED:
        idx = np.flatnonzero(adata.obs["cell_type_harmonised"].to_numpy() == t)
        rec[t] = {"before": int(len(idx))}
        if len(idx) > cap_n:
            order = idx[np.argsort(names[idx], kind="stable")]  # sorted cell IDs
            rng = np.random.default_rng(rng_seed)
            chosen = rng.choice(order, size=cap_n, replace=False)
            drop = np.setdiff1d(idx, chosen)
            keep[drop] = False
        rec[t]["after"] = int(min(len(idx), cap_n))
    return adata[keep].copy(), rec


def link_esm(out: Path) -> Path:
    esm_dir = out / "inputs" / "esm2"
    d = esm_dir / "ESM2"
    d.mkdir(parents=True, exist_ok=True)
    for link, target in ESM2_LINKS.items():
        dst = d / link
        if not dst.exists():
            dst.symlink_to(ESM2 / target)
    return esm_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BM / "v2")
    ap.add_argument("--skip-input-hashes", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.is_relative_to((REPO / "data" / "processed").resolve()):
        raise SystemExit("refusing to write under data/processed/")
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("bone_marrow_unified.h5ad", "gin_scaler_bone_marrow.pkl", "build_report.json"):
        (out / stale).unlink(missing_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    np.random.seed(SEED)
    import anndata as ad
    import torch

    torch.manual_seed(SEED)
    report: dict = {
        "repo_sha": subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        "build": 2, "qc": QC, "capped_types": list(CAPPED), "seed": SEED, "spleen_min": SPLEEN_MIN,
        "species": {k: v[:2] for k, v in SPECIES.items()},
    }
    if not args.skip_input_hashes:
        report["inputs"] = {f: sha256(RAW / f) for v in SPECIES.values() for f in v[2].values()}
        report["inputs"][TABLE.name] = sha256(TABLE)
        report["inputs"]["celltype_map.tsv"] = sha256(TASK / "celltype_map.tsv")
        for link, target in ESM2_LINKS.items():
            report["inputs"][f"ESM2/{target}"] = sha256(ESM2 / target)

    cmap = load_label_map()
    parts: dict = {ens: {} for ens in SPECIES}
    files_rec: dict = {}
    for ens in SPECIES:
        parts[ens]["bone marrow"], files_rec[f"{ens}|bone marrow"] = load_file(ens, "bone marrow", cmap)

    # Spleen rule, on lemur bone-marrow counts after QC and label mapping only.
    lem = files_rec["mmurinus|bone marrow"]["mapped_by_type"]
    add_spleen = lem["B"] < SPLEEN_MIN or lem["NK"] < SPLEEN_MIN
    report["spleen_rule"] = {"lemur_bm_B": lem["B"], "lemur_bm_NK": lem["NK"], "threshold": SPLEEN_MIN,
                             "spleen_added": bool(add_spleen)}
    log.info("spleen rule: %s", report["spleen_rule"])
    if add_spleen:
        for ens in SPECIES:
            parts[ens]["spleen"], files_rec[f"{ens}|spleen"] = load_file(ens, "spleen", cmap)
    report["files"] = files_rec

    merged = {ens: ad.concat(list(lt.values()), join="inner", merge="same") if len(lt) > 1 else next(iter(lt.values()))
              for ens, lt in parts.items()}
    pre_cap = {ens: a.obs["cell_type_harmonised"].value_counts().reindex(TYPES, fill_value=0).astype(int).to_dict()
               for ens, a in merged.items()}
    cap_n = choose_cap(pre_cap)
    report["cap"] = cap_n
    report["cap_rule"] = {"size_ref": SIZE_REF, "step": CAP_STEP, "pre_cap_counts": pre_cap,
                          "capped_total": int(sum(min(v, cap_n) for per in pre_cap.values() for v in per.values()))}
    log.info("cap rule: %s", {k: v for k, v in report["cap_rule"].items() if k != "pre_cap_counts"} | {"cap": cap_n})
    per_species, caps = {}, {}
    for ens, a in merged.items():
        a, caps[ens] = cap(a, SEED, cap_n)
        o = a.obs
        o["batch"] = o["species"] + "|" + o["donor_id"] + "|" + o["tissue"]
        if ens == "mmusculus":
            o["age_months"] = o["age"].str.rstrip("m").astype(float)
        else:
            o["age_months"] = np.nan
        o["cell_ontology_class"] = o["cell_type_harmonised"]
        per_species[ens] = a
    report["caps"] = caps
    del parts, merged

    # Ortholog table, prior, ESM-2 features.
    table = pd.read_csv(TABLE)
    prior = du.compute_conservation_prior(table)
    report["n_ortholog_groups"] = int(len(table))
    report["frac_prior_ge_0.95"] = float((prior.numpy() >= 0.95).mean())
    esm_dir = link_esm(out)
    species_list = sorted(SPECIES)
    consensus, per_sp_emb = du.download_and_process_esm2(table, species_list=species_list, saturn_data_dir=esm_dir)
    report["esm2_matched"] = {k: int(np.any(v != 0, axis=1).sum()) for k, v in per_sp_emb.items()}
    esm2_pca = du.pca_esm2(consensus, n_components=64)
    esm2_var = du.compute_esm2_interspecies_variance(per_sp_emb)

    # Unified object, expression stats, GIN features, p_null.
    G = len(table)
    # construct_unified_adata copies one gene column at a time (X[:, j]); on CSR that scans
    # every nonzero per column. Same values in CSC storage make each column read O(nnz_col).
    import scipy.sparse as sp

    for a in per_species.values():
        a.X = sp.csc_matrix(a.X)
    log.info("constructing unified AnnData")
    unified = du.construct_unified_adata(per_species, table, np.zeros((G, 72), dtype=np.float32), prior)
    del per_species
    stats = du.compute_expression_stats(unified)
    features, scaler = du.assemble_gin_features(esm2_pca, esm2_var, table, stats)
    unified.varm["gene_features"] = features
    unified.uns["p_null"] = du.precompute_p_null(unified)
    for col in ("species", "batch", "cell_type", "cell_ontology_class", "cell_type_harmonised", "tissue",
                "donor_id", "sample_id", "assay", "development_stage", "sex", "cell_type_ontology_term_id"):
        unified.obs[col] = pd.Categorical(unified.obs[col].astype(str))
    unified.obs = unified.obs.drop(columns=[c for c in ("age",) if c in unified.obs])

    # Counts per type x species (and tissue).
    ct = pd.crosstab(unified.obs["cell_type_harmonised"], unified.obs["species"]).reindex(TYPES)
    ct.to_csv(out / "counts_type_by_species.tsv", sep="\t")
    ctt = pd.crosstab([unified.obs["cell_type_harmonised"]], [unified.obs["species"], unified.obs["tissue"]]).reindex(TYPES)
    ctt.to_csv(out / "counts_type_by_species_tissue.tsv", sep="\t")
    report["counts_type_by_species"] = {t: {s: int(v) for s, v in row.items()} for t, row in ct.iterrows()}
    report["per_species"] = unified.obs["species"].astype(str).value_counts().to_dict()
    report["n_batches"] = {s: int(unified.obs.loc[unified.obs["species"] == s, "batch"].nunique())
                           for s in unified.obs["species"].cat.categories}
    report["shape"] = list(unified.shape)
    report["p_null_species_order"] = sorted(unified.obs["species"].astype(str).unique())

    # Ortholog-table assertion, via the registered cohort table as well.
    report["assert_table_matches_spearman"] = gene_ids.assert_table_matches(unified, table.set_index("ortholog_group_id"))
    report["assert_table_matches_registry"] = gene_ids.assert_table_matches(unified, gene_ids.load_table("bone_marrow"))

    path = out / "bone_marrow_unified.h5ad"
    unified.write_h5ad(path)
    with open(out / "gin_scaler_bone_marrow.pkl", "wb") as f:
        pickle.dump(scaler, f)
    report["content_hash"] = content_hash(unified)
    report["outputs"] = {path.name: sha256(path)}
    (out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("done: %s %s", path, unified.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
