"""T11.1: rebuild the Van Zyl aqueous-humor cohort from the GEO count matrices.

Uses the ``chimera._data_utils`` pipeline unchanged:
``download_van_zyl_raw`` (reads the five GEO CSVs and the SATURN metafile),
``_validate_raw_counts``, ``_qc_filter``, ``compute_conservation_prior``,
``download_and_process_esm2``, ``pca_esm2``, ``compute_esm2_interspecies_variance``,
``construct_unified_adata``, ``compute_expression_stats``, ``assemble_gin_features``,
``precompute_p_null``. The ortholog table ``data/processed/ortholog_table_van_zyl.csv``
(Ensembl 110) is reference data; it is restricted to the groups measured in all five species
and written, renumbered, to ``ortholog_table_van_zyl.csv`` in the output folder. Cell-type
labels are harmonised with the committed map ``celltype_map.tsv``.

Measured genes: a group is kept only if, in every species, its gene is in that species' own
count matrix, by symbol or by Ensembl ID (the rhesus and pig matrices name unannotated genes
by ID), each matrix gene serving one group. ``construct_unified_adata`` maps through a single
column, so a gene missing from one species would enter as a false zero there, and genes
present under the other identifier would be lost.

QC: ``min_genes`` only. No mitochondrial filter in this cohort, a deviation from the canonical
20%: mt genes are named ND1/COX1 in rhesus and pig, so the prefix rule cannot see them, and the
cynomolgus matrix has none, so the filter could not act evenly across species.

Thin types: a harmonised cell type is removed from a species where it has fewer than 20 cells
(owner default rule), before any statistic; removals are listed in ``build_report.json``.

Batches: ``obs['source_batch']`` is ``species|chemistry`` (10x v2 or v3, ``batch_chemistry.tsv``);
``obs['batch']`` keeps the 16 samples.

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
SOURCE_TABLE = REPO / "data" / "processed" / "ortholog_table_van_zyl.csv"
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
QC = dict(min_genes=200, max_mt_pct=100.0)  # no mt filter in this cohort, see module docstring
THIN_MIN = 20  # cells per (species, harmonised type); fewer are removed (owner default rule)
OWN_MATRIX = dict(zip(("mmusculus", "sscrofa", "hsapiens", "mfascicularis", "mmulatta"), CSVS))

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
    rec["ortholog_table_van_zyl.csv (source)"] = {"sha256": sha256(SOURCE_TABLE)}
    for sp_, name in ESM2_FILES.items():
        rec[f"ESM2/{name}"] = {"sha256": sha256(ESM2 / name)}
    return rec


def own_gene_list(name: str) -> set:
    """Gene names of one species' own matrix: first column of its CSV (genes by cells)."""
    return set(pd.read_csv(RAW / name, usecols=[0], header=0).iloc[:, 0].astype(str))


def measured_table(table: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, pd.DataFrame]:
    """Restrict ``table`` to groups measured in all species; see the module docstring.

    Returns the renumbered table (ordered by human Ensembl ID, as the bone-marrow table), the
    per-species ``{new group id: matrix gene}`` maps, a report, and a boolean frame (rows of
    ``table``, one column per species) of where each group is measured.
    """
    rep: dict = {"n_groups_source": int(len(table)), "per_species": {}}
    genes, both_rows = {}, []
    for sp in sorted(ENSEMBL.values()):
        own = own_gene_list(OWN_MATRIX[sp])
        name, gid = table[f"{sp}_gene_name"], table[f"{sp}_gene_id"]
        name_unique = ~name.duplicated(keep=False) & name.notna()
        # The Ensembl ID row is the table's gene; a symbol row is used only when the matrix has
        # no row for the ID (audit R4 n1: in rhesus and pig a same-symbol row can be another gene).
        by_id = gid.isin(own)
        by_name = name_unique & name.isin(own) & ~by_id
        both = by_id & name_unique & name.isin(own)
        if both.any():
            both_rows.append(pd.DataFrame({"species": sp, "source_ortholog_group_id": table.loc[both, "ortholog_group_id"],
                                           "symbol_row": name[both], "id_row_used": gid[both]}))
        g = pd.Series(pd.NA, index=table.index, dtype=object)
        g[by_id] = gid[by_id]
        g[by_name] = name[by_name]
        clash = g.notna() & g.duplicated(keep=False)  # a matrix gene may serve one group only
        g[clash] = pd.NA
        genes[sp] = g
        rep["per_species"][sp] = {
            "n_genes_own_matrix": len(own), "n_by_name": int((by_name & ~clash).sum()),
            "n_by_id": int((by_id & ~clash).sum()), "n_clash_dropped": int(clash.sum()),
            "n_not_measured": int(g.isna().sum()), "n_ambiguous_name_rows": int((name.notna() & ~name_unique).sum()),
            "n_symbol_and_id_rows_both_present_id_used": int(both.sum()),
        }
    measured = pd.DataFrame({sp: g.notna() for sp, g in genes.items()})
    keep = measured.all(axis=1).to_numpy()
    new = table.loc[keep].copy()
    new.insert(1, "source_ortholog_group_id", new["ortholog_group_id"])
    order = new["hsapiens_gene_id"].sort_values(kind="stable").index
    new = new.loc[order].reset_index(drop=False).rename(columns={"index": "_row"})
    new["ortholog_group_id"] = np.arange(len(new))
    maps = {sp: dict(zip(new["ortholog_group_id"], g.loc[new["_row"]])) for sp, g in genes.items()}
    new = new.drop(columns="_row")
    rep["n_groups"] = int(len(new))
    rep["symbol_and_id_rows"] = (pd.concat(both_rows).to_dict(orient="records") if both_rows else [])
    return new, maps, rep, measured


def glaucoma_coverage(source: pd.DataFrame, new: pd.DataFrame, measured: pd.DataFrame) -> dict:
    """Coverage of the fixed glaucoma list on the measured vocabulary; the list itself is unchanged."""
    gl = pd.read_csv(TASK / "glaucoma_genes.tsv", sep="\t")
    missing = measured.apply(lambda r: ";".join(c for c in measured.columns if not r[c]), axis=1)
    missing.index = source["hsapiens_gene_id"]
    missing = missing[~missing.index.duplicated()]
    ids = new.set_index("hsapiens_gene_id")["ortholog_group_id"]
    out = gl[["gene", "source", "hsapiens_gene_id", "in_van_zyl_vocab"]].copy()
    out["in_measured_vocab"] = out["hsapiens_gene_id"].isin(ids.index)
    out["ortholog_group_id"] = out["hsapiens_gene_id"].map(ids).astype("Int64")
    out["not_measured_in"] = out["hsapiens_gene_id"].map(missing).fillna("")
    out.to_csv(TASK / "glaucoma_coverage_measured.tsv", sep="\t", index=False)
    van = out["source"].str.contains("vanzyl")
    gh = out["source"].str.contains("gharahkhani")
    return {
        "n_list": int(len(out)), "n_in_source_table": int(out["in_van_zyl_vocab"].sum()),
        "n_in_measured_vocab": int(out["in_measured_vocab"].sum()),
        "van_zyl_named_in_measured": int(out.loc[van, "in_measured_vocab"].sum()), "van_zyl_named": int(van.sum()),
        "gharahkhani_in_measured": int(out.loc[gh, "in_measured_vocab"].sum()), "gharahkhani": int(gh.sum()),
        "lost_from_table": out.loc[out["in_van_zyl_vocab"] & ~out["in_measured_vocab"], "gene"].tolist(),
    }


def add_source_batch(obs: pd.DataFrame) -> pd.DataFrame:
    chem = pd.read_csv(TASK / "batch_chemistry.tsv", sep="\t").set_index("batch")
    b = obs["batch"].astype(str)
    if not b.isin(chem.index).all():
        raise ValueError(f"batches without chemistry: {sorted(set(b) - set(chem.index))}")
    if not (chem.loc[b, "species"].to_numpy() == obs["species"].astype(str).to_numpy()).all():
        raise ValueError("batch_chemistry.tsv species disagrees with obs['species']")
    out = obs.copy()
    out["chemistry"] = pd.Categorical(chem.loc[b, "chemistry"].to_numpy())
    out["source_batch"] = pd.Categorical(obs["species"].astype(str).to_numpy() + "|" + chem.loc[b, "chemistry"].to_numpy())
    return out


def class_filter(raw, class_min: int):
    """Remove a class from a species below ``class_min`` cells, unless a planned test needs the class.

    Rule of the kidney and islet cohorts (checklist item 5). Exception, reason recorded: V2 and V3 compare
    species over shared outflow types, so a class in van Zyl's conventional or uveoscleral outflow groups
    and present in all five species stays.
    """
    t = pd.crosstab(raw.obs["cell_ontology_class"].astype(str), raw.obs["species"].astype(str))
    groups = raw.obs[["cell_ontology_class", "cell_class"]].astype(str).drop_duplicates().set_index("cell_ontology_class")["cell_class"]
    shared_outflow = (t > 0).all(axis=1) & groups.reindex(t.index).isin(["conventional outflow", "uveoscleral outflow"])
    removed, kept = [], []
    for k in t.index:
        for s_ in t.columns:
            if 0 < t.loc[k, s_] < class_min:
                (kept if shared_outflow[k] else removed).append(dict(species=s_, cell_ontology_class=k, n_cells=int(t.loc[k, s_])))
    drop = pd.MultiIndex.from_arrays([raw.obs["species"].astype(str), raw.obs["cell_ontology_class"].astype(str)]).isin(
        [(r["species"], r["cell_ontology_class"]) for r in removed])
    rec = dict(min_cells=class_min, removed=removed, kept_shared_outflow_exception=kept, n_cells_removed=int(drop.sum()),
               exception_reason="V2 and V3 compare species over shared outflow types")
    return raw[~drop].copy(), rec


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


def compare_old(new, table: pd.DataFrame, out: Path) -> dict:
    import anndata as ad
    import scipy.sparse as sp

    old = ad.read_h5ad(OLD)
    rec: dict = {"old_path": str(OLD.relative_to(REPO)), "old_sha256": sha256(OLD)}
    rec["shape_old"], rec["shape_new"] = list(old.shape), list(new.shape)
    rec["n_genes_only_old"] = int(old.n_vars - new.n_vars)
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
    # New groups are a renumbered subset: compare each with its source group in the old object.
    src = [f"ortho_{i}" for i in table["source_ortholog_group_id"].to_numpy()]
    o, n = old[common, src], new[common]
    d = (sp.csr_matrix(o.X) - sp.csr_matrix(n.X)).tocsr()
    d.eliminate_zeros()
    rec["X_equal_on_common_cells_and_genes"] = bool(d.nnz == 0)
    rec["n_genes_gaining_counts_by_species"] = {}
    for s in sorted(new.obs["species"].astype(str).unique()):
        m = (n.obs["species"].astype(str) == s).to_numpy()
        rec["n_genes_gaining_counts_by_species"][s] = int((np.asarray(np.abs(d[m]).sum(axis=0)).ravel() > 0).sum())
    rec["conservation_prior_max_abs_diff"] = float(
        np.abs(o.var["conservation_prior"].to_numpy() - n.var["conservation_prior"].to_numpy()).max()
    )
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
    ap.add_argument("--class-min", type=int, default=None,
                    help="cells per (species, cell_ontology_class); see class_filter (T11.8 sensitivity object)")
    args = ap.parse_args()
    out = args.out.resolve()
    if out.is_relative_to((REPO / "data" / "processed").resolve()):
        raise SystemExit("refusing to write under data/processed/")
    out.mkdir(parents=True, exist_ok=True)
    for stale in ("van_zyl_unified.h5ad", "gin_scaler_van_zyl.pkl", "build_report.json",
                  "ortholog_table_van_zyl.csv",
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

    # 2. Thin cell types (owner default rule): drop a harmonised type from a species where it
    # has fewer than THIN_MIN cells, before any statistic is computed. No cap is applied.
    raw.obs = harmonise(raw.obs)
    counts = raw.obs.groupby(["species", "cell_type_harmonised"], observed=True).size()
    thin = counts[counts < THIN_MIN]
    report["thin_type_rule"] = {"min_cells_per_species": THIN_MIN, "removed": [
        {"species": s, "cell_type_harmonised": t, "n_cells": int(n)} for (s, t), n in thin.items()]}
    drop = pd.MultiIndex.from_arrays([raw.obs["species"].astype(str), raw.obs["cell_type_harmonised"].astype(str)]).isin(
        [(str(s), str(t)) for s, t in thin.index])
    raw = raw[~drop].copy()
    report["thin_type_rule"]["n_cells_removed"] = int(drop.sum())
    if args.class_min:
        raw, report["class_filter"] = class_filter(raw, args.class_min)

    # 3. Per species: validate counts, QC filter.
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

    # 3. Ortholog table restricted to measured groups, prior, ESM-2 features.
    source_table = pd.read_csv(SOURCE_TABLE)
    table, gene_maps, report["measured_genes"], measured = measured_table(source_table)
    table.to_csv(out / "ortholog_table_van_zyl.csv", index=False)
    report["outputs_table_sha256"] = sha256(out / "ortholog_table_van_zyl.csv")
    if out == DEFAULT_OUT.resolve():
        report["glaucoma_coverage"] = glaucoma_coverage(source_table, table, measured)
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
    # construct_unified_adata maps through one column: give it one exact key per group, so a
    # group found by symbol in one species and by Ensembl ID in another maps to the same gene.
    key = table[["ortholog_group_id"]].copy()
    aligned = {}
    for sp, a in per_species.items():
        key[f"{sp}_gene_id"] = [f"group{i}" for i in table["ortholog_group_id"]]
        groups = list(gene_maps[sp])
        a = a[:, [gene_maps[sp][g] for g in groups]].copy()
        a.var_names = [f"group{g}" for g in groups]
        aligned[sp] = a
    del per_species
    unified = du.construct_unified_adata(aligned, key, np.zeros((G, 72), dtype=np.float32), prior)
    stats = du.compute_expression_stats(unified)
    features, scaler = du.assemble_gin_features(esm2_pca, esm2_var, table, stats)
    unified.varm["gene_features"] = features
    unified.uns["p_null"] = du.precompute_p_null(unified)

    # 5. Harmonised labels.
    unified.obs = add_source_batch(harmonise(unified.obs))
    report["cells_by_source_batch"] = unified.obs["source_batch"].astype(str).value_counts().to_dict()
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
        report["comparison_old_object"] = compare_old(unified, table, out)
    (out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    log.info("done: %s %s", path, unified.shape)
    return 0


if __name__ == "__main__":
    sys.exit(main())
