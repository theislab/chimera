"""T11.1: rebuild the Van Zyl aqueous-humor cohort from the GEO count matrices.

Uses the ``chimera._data_utils`` pipeline unchanged:
``download_van_zyl_raw`` (reads the five GEO CSVs and the SATURN metafile),
``_validate_raw_counts``, ``_qc_filter``, ``compute_conservation_prior``,
``download_and_process_esm2``, ``pca_esm2``, ``compute_esm2_interspecies_variance``,
``construct_unified_adata``, ``compute_expression_stats``, ``assemble_gin_features``,
``precompute_p_null``. The ortholog table ``data/processed/ortholog_table_van_zyl.csv``
(Ensembl 110) is reused as reference data. Cell-type labels are harmonised with the
committed map ``celltype_map.tsv``.

Inputs are fixed by symlink folders under the output directory, so the pipeline reads
exactly these files: the five CSVs and the metafile, and the five species' ESM-2 files
(the ESM-2 export also holds ``mouse_lemur_embedding.torch``, which the loader's
substring match could otherwise pick for mouse).

Writes to ``data/revision/van_zyl/`` (or ``--out``), never to ``data/processed/``.

    python revision/tasks/T11_1_vanzyl/build_vanzyl.py [--out DIR] [--no-compare]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from chimera import _data_utils as du  # noqa: E402
from revision.common import gene_ids  # noqa: E402

TASK = Path(__file__).resolve().parent
RAW = REPO / "data" / "raw" / "van_zyl_raw"
ESM2 = REPO / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"
TABLE = REPO / "data" / "processed" / "ortholog_table_van_zyl.csv"
OLD = REPO / "data" / "processed" / "van_zyl_unified.h5ad"
OLD_RAW_CACHE = REPO / "data" / "raw" / "van_zyl_raw.h5ad"
DEFAULT_OUT = REPO / "data" / "revision" / "van_zyl"

CSVS = [
    "GSE146186_Mouse_count_matrix.csv",
    "GSE146187_Pig_count_matrix.csv",
    "GSE148371_Human_count_matrix.csv",
    "GSE148373_MacaF_count_matrix.csv",
    "GSE148374_MacaqueM_count_matrix.csv",
]
META = "all_five_species_metafile.csv"
# obs['species'] label -> Ensembl species name (column prefix in the ortholog table)
ENSEMBL = {
    "human": "hsapiens",
    "macaca_fascicularis": "mfascicularis",
    "macaca_mulatta": "mmulatta",
    "mouse": "mmusculus",
    "pig": "sscrofa",
}
ESM2_FILES = {
    "hsapiens": "human_embedding.torch",
    "mfascicularis": "macaca_fascicularis_embedding.torch",
    "mmulatta": "macaca_mulatta_embedding.torch",
    "mmusculus": "mouse_embedding.torch",
    "sscrofa": "pig_embedding.torch",
}
QC = dict(min_genes=200, max_mt_pct=20.0)

log = logging.getLogger("build_vanzyl")


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def sha256_gunzip(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with gzip.open(path, "rb") as f:
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


def link_inputs(out: Path) -> tuple[Path, Path]:
    """Symlink folders that fix what the pipeline reads."""
    raw_dir = out / "inputs" / "raw"
    vz = raw_dir / "van_zyl_raw"
    esm_dir = out / "inputs" / "esm2"
    esm = esm_dir / "ESM2"
    for d in (vz, esm):
        d.mkdir(parents=True, exist_ok=True)
    for name in CSVS + [META]:
        dst = vz / name
        if not dst.exists():
            dst.symlink_to(RAW / name)
    for name in ESM2_FILES.values():
        dst = esm / name
        if not dst.exists():
            dst.symlink_to(ESM2 / name)
    return raw_dir, esm_dir


def input_hashes(skip_gunzip_check: bool) -> dict:
    rec = {}
    for name in CSVS:
        csv, gz = RAW / name, RAW / (name + ".gz")
        r = {"csv_gz_sha256": sha256(gz), "csv_sha256": sha256(csv)}
        if not skip_gunzip_check:
            r["csv_equals_gunzip"] = sha256_gunzip(gz) == r["csv_sha256"]
            if not r["csv_equals_gunzip"]:
                raise RuntimeError(f"{csv.name} differs from gunzip of {gz.name}")
        rec[name] = r
    rec[META] = {"sha256": sha256(RAW / META)}
    rec["ortholog_table_van_zyl.csv"] = {"sha256": sha256(TABLE)}
    for sp_, name in ESM2_FILES.items():
        rec[f"ESM2/{name}"] = {"sha256": sha256(ESM2 / name)}
    return rec


def harmonise(obs: pd.DataFrame) -> pd.DataFrame:
    cmap = pd.read_csv(TASK / "celltype_map.tsv", sep="\t")
    if cmap.duplicated(["species", "source_label"]).any():
        raise ValueError("duplicate (species, source_label) in celltype_map.tsv")
    key = pd.MultiIndex.from_arrays([obs["species"].astype(str), obs["cell_type"].astype(str)])
    m = cmap.set_index(["species", "source_label"]).reindex(key)
    if m["cell_type_harmonised"].isna().any():
        bad = sorted(set(key[m["cell_type_harmonised"].isna().to_numpy()]))
        raise ValueError(f"cells without a harmonised label: {bad}")
    used = set(key)
    unused = [k for k in cmap.set_index(["species", "source_label"]).index if k not in used]
    if unused:
        log.warning("map rows not used by any cell: %s", unused)
    out = obs.copy()
    for col in ("cell_type_harmonised", "cell_ontology_class", "cell_class"):
        out[col] = pd.Categorical(m[col].to_numpy())
    return out


def shared_tables(obs: pd.DataFrame, out: Path) -> dict:
    res = {}
    for col in ("cell_type_harmonised", "cell_ontology_class"):
        tab = pd.crosstab(obs[col], obs["species"])
        tab["n_species"] = (tab > 0).sum(axis=1)
        tab = tab.sort_values(["n_species"], ascending=False, kind="stable")
        tab.to_csv(out / f"counts_{col}_by_species.tsv", sep="\t")
        n_sp = obs["species"].nunique()
        shared_all = tab.index[tab["n_species"] == n_sp].tolist()
        frac_cells_shared = float(obs[col].isin(shared_all).mean())
        res[col] = {
            "n_labels": int(len(tab)),
            "n_shared_all_species": len(shared_all),
            "shared_all_species": shared_all,
            "n_shared_ge2_species": int((tab["n_species"] >= 2).sum()),
            "frac_cells_in_types_shared_by_all": frac_cells_shared,
        }
    return res


def compare_old(new, out: Path) -> dict:
    import anndata as ad
    import scipy.sparse as sp

    old = ad.read_h5ad(OLD)
    rec: dict = {"old_path": str(OLD.relative_to(REPO)), "old_sha256": sha256(OLD)}
    rec["shape_old"], rec["shape_new"] = list(old.shape), list(new.shape)
    rec["var_names_identical"] = bool(old.var_names.equals(new.var_names))
    so, sn = set(old.obs_names), set(new.obs_names)
    only_old = sorted(so - sn)
    rec["n_cells_only_old"], rec["n_cells_only_new"] = len(only_old), len(sn - so)
    rec["cells_only_old_by_species"] = (
        old.obs.loc[only_old, "species"].astype(str).value_counts().to_dict() if only_old else {}
    )
    rec["cells_only_old_by_cell_type"] = (
        old.obs.loc[only_old, "cell_type"].astype(str).value_counts().to_dict() if only_old else {}
    )
    rec["per_species_old"] = old.obs["species"].astype(str).value_counts().to_dict()
    rec["per_species_new"] = new.obs["species"].astype(str).value_counts().to_dict()
    common = [c for c in new.obs_names if c in so]
    o, n = old[common], new[common]
    if rec["var_names_identical"]:
        Xo = sp.csr_matrix(o.X)
        Xn = sp.csr_matrix(n.X)
        d = (Xo - Xn)
        rec["X_equal_on_common_cells"] = bool(d.nnz == 0 or np.abs(d.data).max() == 0)
        rec["conservation_prior_max_abs_diff"] = float(
            np.abs(old.var["conservation_prior"].to_numpy() - new.var["conservation_prior"].to_numpy()).max()
        )
        gf = np.abs(np.asarray(old.varm["gene_features"]) - np.asarray(new.varm["gene_features"]))
        rec["gene_features_max_abs_diff"] = float(gf.max())
        rec["gene_features_max_abs_diff_per_block"] = {
            "esm2_pca_0_63": float(gf[:, :64].max()),
            "esm2_interspecies_var_64": float(gf[:, 64].max()),
            "mean_identity_65": float(gf[:, 65].max()),
            "expression_stats_66_68": float(gf[:, 66:69].max()),
            "identity_spread_69_71": float(gf[:, 69:72].max()),
        }
        pn_o, pn_n = np.asarray(old.uns["p_null"]), np.asarray(new.uns["p_null"])
        rec["p_null_max_rel_diff"] = float(np.abs(pn_o - pn_n).max() / np.abs(pn_o).max()) if pn_o.shape == pn_n.shape else None
    rec["cell_type_equal_on_common_cells"] = bool(
        (o.obs["cell_type"].astype(str).to_numpy() == n.obs["cell_type"].astype(str).to_numpy()).all()
    )
    rec["batch_equal_on_common_cells"] = bool(
        (o.obs["batch"].astype(str).to_numpy() == n.obs["batch"].astype(str).to_numpy()).all()
    )
    rec["obs_columns_old"] = list(old.obs.columns)
    rec["obs_columns_new"] = list(new.obs.columns)
    (out / "comparison_old_object.json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-compare", action="store_true", help="skip the cross-check with the old object")
    ap.add_argument("--skip-gunzip-check", action="store_true")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.is_relative_to((REPO / "data" / "processed").resolve()):
        raise SystemExit("refusing to write under data/processed/")
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("van_zyl_unified.h5ad", "gin_scaler_van_zyl.pkl", "build_report.json",
                  "inputs/raw/van_zyl_raw.h5ad"):
        (out / stale).unlink(missing_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    np.random.seed(42)
    import torch

    torch.manual_seed(42)
    report: dict = {"repo_sha": None, "qc": QC, "ensembl_species": ENSEMBL}
    try:
        import subprocess

        report["repo_sha"] = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        pass

    log.info("hashing inputs")
    report["inputs"] = input_hashes(args.skip_gunzip_check)
    raw_dir, esm_dir = link_inputs(out)

    # 1. Raw counts: the five GEO CSVs + metafile, SATURN loader parity.
    raw = du.download_van_zyl_raw(data_dir=raw_dir)
    report["raw"] = {"n_cells": int(raw.n_obs), "n_genes_union": int(raw.n_vars),
                     "per_species": raw.obs["species"].astype(str).value_counts().to_dict()}
    if OLD_RAW_CACHE.exists() and not args.no_compare:
        import anndata as ad
        import scipy.sparse as sp

        old_raw = ad.read_h5ad(OLD_RAW_CACHE)
        same = old_raw.obs_names.equals(raw.obs_names) and old_raw.var_names.equals(raw.var_names)
        if same:
            d = sp.csr_matrix(old_raw.X) - sp.csr_matrix(raw.X)
            same = d.nnz == 0 or float(np.abs(d.data).max()) == 0.0
        report["raw"]["equals_data_raw_van_zyl_raw_h5ad"] = bool(same)
        del old_raw

    # 2. Per species: validate counts, QC filter.
    per_species, qc_rec = {}, {}
    for sp_label in sorted(raw.obs["species"].astype(str).unique()):
        a = raw[raw.obs["species"].astype(str) == sp_label].copy()
        du._validate_raw_counts(a, f"van Zyl {sp_label}")
        detected = np.asarray((a.X > 0).sum(axis=0)).ravel() > 0
        mt_detected = int((a.var_names[detected].str.startswith(("MT-", "Mt-", "mt-", "MTND", "MTCO", "MTATP"))).sum())
        n0 = a.n_obs
        a = du._qc_filter(a, **QC)
        qc_rec[sp_label] = {"n_before": n0, "n_after": int(a.n_obs), "n_removed": n0 - int(a.n_obs),
                            "mt_genes_detected": mt_detected}
        per_species[ENSEMBL[sp_label]] = a
    report["qc_per_species"] = qc_rec
    del raw

    # 3. Ortholog table (reused), prior, ESM-2 features.
    table = pd.read_csv(TABLE)
    prior = du.compute_conservation_prior(table)
    report["n_ortholog_groups"] = int(len(table))
    report["frac_prior_ge_0.95"] = float((prior.numpy() >= 0.95).mean())
    species_list = sorted(ENSEMBL.values())
    consensus, per_sp_emb = du.download_and_process_esm2(table, species_list=species_list,
                                                         saturn_data_dir=esm_dir)
    report["esm2_matched"] = {k: int(np.any(v != 0, axis=1).sum()) for k, v in per_sp_emb.items()}
    esm2_pca = du.pca_esm2(consensus, n_components=64)
    esm2_var = du.compute_esm2_interspecies_variance(per_sp_emb)

    # 4. Unified object, expression stats, GIN features, p_null.
    G = len(table)
    unified = du.construct_unified_adata(per_species, table, np.zeros((G, 72), dtype=np.float32), prior)
    stats = du.compute_expression_stats(unified)
    features, scaler = du.assemble_gin_features(esm2_pca, esm2_var, table, stats)
    unified.varm["gene_features"] = features
    unified.uns["p_null"] = du.precompute_p_null(unified)

    # 5. Harmonised labels.
    unified.obs = harmonise(unified.obs)
    for col in ("species", "batch", "cell_type", "Cluster"):
        unified.obs[col] = pd.Categorical(unified.obs[col].astype(str))
    report["labels"] = shared_tables(unified.obs, out)

    # 6. Ortholog-table assertion (third table: the T2.1 bug class).
    report["assert_table_matches_spearman"] = gene_ids.assert_table_matches(unified, table.set_index("ortholog_group_id"))
    report["shape"] = list(unified.shape)
    report["n_batches"] = int(unified.obs["batch"].nunique())
    report["per_species"] = unified.obs["species"].astype(str).value_counts().to_dict()

    path = out / "van_zyl_unified.h5ad"
    unified.write_h5ad(path)
    with open(out / "gin_scaler_van_zyl.pkl", "wb") as f:
        pickle.dump(scaler, f)
    (out / "inputs" / "raw" / "van_zyl_raw.h5ad").unlink(missing_ok=True)  # intermediate cache
    report["content_hash"] = content_hash(unified)
    report["outputs"] = {"van_zyl_unified.h5ad": sha256(path)}

    if not args.no_compare:
        report["comparison_old_object"] = compare_old(unified, out)
    (out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("done: %s %s", path, unified.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
