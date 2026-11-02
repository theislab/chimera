"""C1: build a cohort object, the T7.2 / T12.1 / P13 pipeline, one code path for every cohort. No readout or test here.

Per species of a source (cached, shared by the cohorts of that source):

1. Whole-matrix check: `raw/X` when present, else `X`, integer and non-negative in every block of the whole file.
2. Filters of the configuration, then QC: at least 200 detected genes, in every species. No mitochondrial filter in any
   species: most portal matrices carry few or no mitochondrial genes, and one threshold cannot act evenly.
3. Labels through the committed map; an unmapped label stops the build.
4. Pseudobulk per unit x class (summed counts, `layers['n_detected']`, `n_cells`) in native gene space, no caps.

Per cohort:

5. Classes: DROP in the map removed; a class stays only if at least 2 species have it and it has at least 50 cells in
   each species that has it (checklist item 5).
6. Cap: one number for every species x class, the largest multiple of 50 that keeps the total within the size of the
   three-species testis cohort (28,854 cells), donors sharing each quota (T7.2 water-filling).
7. Ortholog table (`orthologs.py`), measured genes only; `compute_conservation_prior`, `download_and_process_esm2`,
   `pca_esm2`, `compute_esm2_interspecies_variance`, `construct_unified_adata`, `compute_expression_stats`,
   `assemble_gin_features`; a group stays only if each of its genes has count > 0 over the species' units and it is
   in at least 3 training cells; `precompute_p_null` on the result. All from `chimera._data_utils`, unchanged.

Batch: `obs['batch']` = species|source|assay, the technical source within species; donors in `obs['sample_batch']`.

    python revision/tasks/C1_cohorts/build.py <cohort> [--species-pass-only] [--force]
    python revision/tasks/C1_cohorts/build.py --subset <tag>
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from anndata.io import read_elem

from chimera._data_utils import (
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
from revision.common import gene_ids
from revision.common.check_cohort import check as check_cohort
from revision.common.outputs import git_sha
from revision.tasks.C1_cohorts import configs as CF
from revision.tasks.C1_cohorts import orthologs as OR
from revision.tasks.T7_2_disease_objects import build as t72

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("C1_build")
BLOCK = 50_000

SATURN = gene_ids.REPO_ROOT / "data" / "raw" / "saturn_esm2" / "protein_embeddings_export" / "ESM2"
NEW_ESM = CF.DATA / "esm" / "ESM2"
ESM_FILES = {"hsapiens": ("human_embedding.torch", SATURN / "human_embedding.torch"),
             "mmusculus": ("mouse_embedding.torch", SATURN / "mouse_embedding.torch"),
             "mmulatta": ("macaca_mulatta_embedding.torch", SATURN / "macaca_mulatta_embedding.torch"),
             # linked under its short name so the loader's substring match cannot pick it for mouse (T12.1)
             "mmurinus": ("mmurinus_embedding.torch", SATURN / "mouse_lemur_embedding.torch"),
             "drerio": ("zebrafish_embedding.torch", SATURN / "zebrafish_embedding.torch"),
             "ptroglodytes": ("chimpanzee_embedding.torch", NEW_ESM / "chimpanzee_embedding.torch"),
             "cjacchus": ("marmoset_embedding.torch", NEW_ESM / "marmoset_embedding.torch"),
             "ggallus": ("chicken_embedding.torch", NEW_ESM / "chicken_embedding.torch")}


def species_dir(src: str, sp_short: str) -> Path:
    d = CF.BUILD / "species" / src / sp_short
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_map(src_cfg: dict, sp_short: str) -> pd.Series:
    m = pd.read_csv(src_cfg["map"], sep="\t", dtype=str, keep_default_na=False)
    if "species" in m:
        m = m[m["species"] == sp_short]
    return m.set_index("author_label")["cell_ontology_class"]


def x_group(f: h5py.File) -> tuple[h5py.Group, h5py.Group, str]:
    if "raw" in f and "X" in f["raw"]:
        return f["raw"]["X"], f["raw"]["var"], "raw/X"
    return f["X"], f["var"], "X"


def species_pass(src: str, sp_short: str, force: bool = False) -> None:
    """One pass over the whole matrices of one species: checks, QC, labels, pseudobulk. Cached."""
    cfg = CF.SOURCES[src]
    sc = cfg["species"][sp_short]
    d = species_dir(src, sp_short)
    if (d / "DONE").exists() and not force:
        return
    ctmap = read_map(cfg, sp_short)
    label_col = sc.get("label_col", cfg.get("label_col"))
    unit_col = sc.get("unit_col", cfg.get("unit_col"))
    filt = sc.get("filters", {})
    keep_cols = sorted({label_col, unit_col, "assay", "sex", *filt})
    obs_parts, var_lists, rep = [], [], {}
    groups: dict[str, int] = {}
    sums, dets = [], []
    genes_ref = None
    for path in sc["files"]:
        with h5py.File(path, "r") as f:
            g, var, where = x_group(f)
            key = var.attrs["_index"]
            ids = pd.Index(read_elem(var[key])).astype(str)
            names = read_elem(var["feature_name"]).astype(str) if "feature_name" in var else ids
            var_lists.append(set(ids))
            o = f["obs"]
            obs = pd.DataFrame({c: read_elem(o[c]) for c in keep_cols if c in o}).astype(str)
            obs.index = pd.Index(read_elem(o[o.attrs["_index"]])).astype(str)
            if ids.has_duplicates:
                raise ValueError(f"{path.name}: duplicated feature IDs")
            if genes_ref is None:
                genes_ref = pd.DataFrame({"gene_symbol": np.asarray(names)}, index=ids)
            elif not genes_ref.index.equals(ids):
                # Same species, different feature order or list: work on the union by re-indexing below.
                genes_ref = genes_ref.reindex(genes_ref.index.union(ids))
                genes_ref.loc[ids, "gene_symbol"] = np.asarray(names)
        rep.setdefault("files", {})[path.name] = dict(n_obs=int(len(obs)), n_var=int(len(ids)), matrix=where)
    allg = pd.Index(sorted(genes_ref.index))
    genes_ref = genes_ref.reindex(allg)
    n = len(allg)
    for path in sc["files"]:
        with h5py.File(path, "r") as f:
            g, var, where = x_group(f)
            ids = pd.Index(read_elem(var[var.attrs["_index"]])).astype(str)
            pos = allg.get_indexer(ids)
            o = f["obs"]
            obs = pd.DataFrame({c: read_elem(o[c]) for c in keep_cols if c in o}).astype(str)
            obs.index = pd.Index(read_elem(o[o.attrs["_index"]])).astype(str)
            obs["file"] = path.name
            ok = np.ones(len(obs), bool)
            for c, vals in filt.items():
                ok &= obs[c].isin(vals).to_numpy()
            obs["pass_filter"] = ok
            lab = obs[label_col].map(ctmap)
            if lab[ok].isna().any():
                raise ValueError(f"{sp_short} {path.name}: unmapped labels {sorted(obs.loc[ok & lab.isna().to_numpy(), label_col].unique())[:20]}")
            obs["cell_ontology_class"] = lab.fillna("DROP").to_numpy()
            indptr = g["indptr"][()]
            n_rows = len(indptr) - 1
            if n_rows != len(obs):
                raise ValueError(f"{path.name}: {n_rows} rows vs {len(obs)} obs")
            ngenes = np.zeros(n_rows, int)
            libs = np.zeros(n_rows, float)
            max_val = 0.0
            for p0 in range(0, n_rows, BLOCK):
                p1 = min(p0 + BLOCK, n_rows)
                lo, hi = int(indptr[p0]), int(indptr[p1])
                dat = g["data"][lo:hi].astype(np.float32)
                ind = pos[g["indices"][lo:hi]]
                X = sp.csr_matrix((dat, ind, indptr[p0:p1 + 1] - lo), shape=(p1 - p0, n))
                a = ad.AnnData(X=X)
                _validate_raw_counts(a, f"{sp_short} {path.name} rows {p0}+")  # raises on non-integer or negative
                max_val = max(max_val, float(dat.max()) if len(dat) else 0.0)
                Xb = X > 0
                ngenes[p0:p1] = np.asarray(Xb.sum(1)).ravel()
                libs[p0:p1] = np.asarray(X.sum(1)).ravel()
                keep = ok[p0:p1] & (ngenes[p0:p1] >= CF.MIN_GENES)
                if keep.any():
                    keyv = (obs[unit_col].to_numpy()[p0:p1] + "||" + obs["cell_ontology_class"].to_numpy()[p0:p1])
                    uniq, codes = np.unique(keyv[keep], return_inverse=True)
                    for k in uniq:
                        if k not in groups:
                            groups[k] = len(groups)
                            sums.append(np.zeros(n, dtype=np.int64))
                            dets.append(np.zeros(n, dtype=np.int64))
                    Gm = sp.csr_matrix((np.ones(len(codes)), (codes, np.arange(len(codes)))), shape=(len(uniq), len(codes)))
                    Xk = X[keep].astype(np.float64)
                    s_ = (Gm @ Xk).toarray()
                    d_ = (Gm @ (Xk > 0).astype(np.float64)).toarray()
                    for j, k in enumerate(uniq):
                        sums[groups[k]] += np.rint(s_[j]).astype(np.int64)
                        dets[groups[k]] += np.rint(d_[j]).astype(np.int64)
                log.info("%s %s rows %d-%d", sp_short, path.name[:8], p0, p1)
            obs["n_genes"] = ngenes
            obs["library"] = libs
            obs["row"] = np.arange(n_rows)
            obs["qc_pass"] = ok & (ngenes >= CF.MIN_GENES)
            rep["files"][path.name]["max_count"] = max_val
            rep["files"][path.name]["integer_whole_matrix"] = True
            obs_parts.append(obs)
    obs = pd.concat(obs_parts)
    if not obs.index.is_unique:
        raise ValueError(f"{sp_short}: cell IDs not unique across files")
    keys = sorted(groups, key=groups.get)
    pb_obs = pd.DataFrame([dict(unit=k.split("||")[0], cell_ontology_class=k.split("||")[1]) for k in keys], index=[k.replace("||", "|") for k in keys])
    cnt = obs[obs.qc_pass].groupby([unit_col, "cell_ontology_class"], observed=True).size()
    pb_obs["n_cells"] = [int(cnt.get((u, c), 0)) for u, c in zip(pb_obs.unit, pb_obs.cell_ontology_class)]
    pb_var = pd.DataFrame({"gene_symbol": genes_ref["gene_symbol"].astype(str).to_numpy()}, index=allg)
    pb = ad.AnnData(X=sp.csr_matrix(np.vstack(sums).astype(np.float64)), obs=pb_obs, var=pb_var)
    pb.layers["n_detected"] = sp.csr_matrix(np.vstack(dets).astype(np.float64))
    t72._write_h5ad(pb, d / "pseudobulk.h5ad")
    obs.to_parquet(d / "obs.parquet")
    measured = set.intersection(*var_lists)
    pd.Series(sorted(measured)).to_csv(d / "measured_genes.txt", index=False, header=False)
    rep.update(n_cells=int(len(obs)), n_filtered_out=int((~obs.pass_filter).sum()),
               n_qc_removed=int((obs.pass_filter & ~obs.qc_pass).sum()), n_qc_pass=int(obs.qc_pass.sum()),
               median_genes=float(obs.loc[obs.qc_pass, "n_genes"].median()), median_library=float(obs.loc[obs.qc_pass, "library"].median()),
               n_measured_all_files=len(measured), n_genes_union=int(n), label_col=label_col, unit_col=unit_col,
               n_mt_named=int(genes_ref["gene_symbol"].astype(str).str.upper().str.startswith("MT-").sum()))
    (d / "report.json").write_text(json.dumps(rep, indent=2))
    (d / "DONE").write_text("ok\n")


def read_cells(paths_rows: dict[str, tuple[Path, np.ndarray]], allg: pd.Index) -> sp.csr_matrix:
    """Read rows per file, native gene order to allg."""
    blocks = []
    for name, (path, rows) in paths_rows.items():
        with h5py.File(path, "r") as f:
            g, var, _ = x_group(f)
            ids = pd.Index(read_elem(var[var.attrs["_index"]])).astype(str)
            pos = allg.get_indexer(ids)
            indptr = g["indptr"][()]
            parts_d, parts_i, lens = [], [], []
            rows = np.sort(rows)
            for p0 in range(0, len(rows), 5000):
                rb = rows[p0:p0 + 5000]
                lo, hi = int(indptr[rb[0]]), int(indptr[rb[-1] + 1])
                dat, ind = g["data"][lo:hi], g["indices"][lo:hi]
                st, ln = indptr[rb] - lo, indptr[rb + 1] - indptr[rb]
                off = np.concatenate([[0], np.cumsum(ln)])
                sel = np.repeat(st - off[:-1], ln) + np.arange(off[-1])
                parts_d.append(dat[sel].astype(np.float32))
                parts_i.append(pos[ind[sel]])
                lens.append(ln)
            lens = np.concatenate(lens)
            blocks.append(sp.csr_matrix((np.concatenate(parts_d), np.concatenate(parts_i), np.r_[0, np.cumsum(lens)]), shape=(len(rows), len(allg))))
    return sp.vstack(blocks).tocsr()


def choose_cap(ct: pd.DataFrame) -> int:
    best = 50
    for cap in range(50, 20001, 50):
        if int(ct.clip(upper=cap).to_numpy().sum()) <= CF.MAX_CELLS:
            best = cap
        else:
            break
    return best


def link_esm(cohort: str, species: list[str]) -> Path:
    d = CF.BUILD / "inputs" / cohort / "ESM2"
    d.mkdir(parents=True, exist_ok=True)
    for s in species:
        name, target = ESM_FILES[s]
        if not target.exists():
            raise FileNotFoundError(f"{target}: ESM embeddings for {s} are not built yet")
        if not (d / name).exists():
            (d / name).symlink_to(target)
    return d.parent


def build_cohort(cohort: str) -> None:
    cc = CF.COHORTS[cohort]
    src = cc["source"]
    cfg = CF.SOURCES[src]
    species = cc["species"]
    (CF.BUILD / "objects").mkdir(parents=True, exist_ok=True)
    out_rep: dict = dict(cohort=cohort, source=src, reference=cfg["reference"], species=species, code_sha=git_sha())
    filt_rows: list[dict] = []
    obsd, pbs, reports = {}, {}, {}
    for s in species:
        species_pass(src, s)
        d = species_dir(src, s)
        obsd[s] = pd.read_parquet(d / "obs.parquet")
        pbs[s] = ad.read_h5ad(d / "pseudobulk.h5ad")
        reports[s] = json.loads((d / "report.json").read_text())
    names = {s: cfg["species"][s]["name"] for s in species}
    for s in species:
        sc_ = cfg["species"][s]
        if sc_.get("filters"):
            filt_rows.append(dict(step="source", item=str(sc_["filters"]), counts={s: reports[s]["n_filtered_out"]}, reason=sc_.get("filter_reason", "")))
    # --- classes
    ct = pd.DataFrame({s: obsd[s].loc[obsd[s].qc_pass, "cell_ontology_class"].value_counts() for s in species}).fillna(0).astype(int)
    out_rep["class_counts_before"] = ct.to_dict()
    classes = []
    for c, row in ct.iterrows():
        if c == "DROP":
            filt_rows.append(dict(step="labels", item="DROP in map", counts=row.to_dict(), reason="labels mapped to DROP: doublets, ambiguous or unresolved states, or a class one species alone defines"))
            continue
        have = row[row > 0]
        if len(have) < 2:
            filt_rows.append(dict(step="thin types", item=c, counts=row.to_dict(), reason="class present in one species only: no cross-species information"))
        elif (have < CF.MIN_CELLS_MODEL).any():
            filt_rows.append(dict(step="thin types", item=c, counts=row.to_dict(), reason=f"fewer than {CF.MIN_CELLS_MODEL} cells in a species that has the class"))
        else:
            classes.append(c)
    ct2 = ct.loc[classes]
    cap = choose_cap(ct2)
    out_rep.update(classes=classes, cap=cap, cap_rule=f"largest multiple of 50 with total <= {CF.MAX_CELLS}", class_counts_kept=ct2.to_dict(),
                   class_counts_after_cap=ct2.clip(upper=cap).to_dict())
    t72.CAP = cap
    rng = np.random.default_rng(CF.SEED)
    # --- cells
    adatas = {}
    for s in species:
        o = obsd[s]
        o = o[o.qc_pass & o.cell_ontology_class.isin(classes)].copy()
        unit_col = cfg["species"][s].get("unit_col", cfg.get("unit_col"))
        cap_obs = pd.DataFrame(dict(species=s, condition="control", cell_ontology_class=o["cell_ontology_class"], donor=o[unit_col]), index=o.index)
        keep = t72.cap_nuclei(cap_obs, rng)
        o = o.loc[keep]
        allg = pd.Index(pbs[s].var_names)
        files = {p.name: p for p in cfg["species"][s]["files"]}
        paths_rows = {fn: (files[fn], g["row"].to_numpy()) for fn, g in o.groupby("file", observed=True)}
        o = pd.concat([g.sort_values("row") for _, g in o.groupby("file", observed=True, sort=False)]) if len(paths_rows) else o
        # read_cells goes file by file in the order of paths_rows; o above has the same order
        order = list(paths_rows)
        o = pd.concat([o[o.file == fn].sort_values("row") for fn in order])
        X = read_cells(paths_rows, allg)
        if X.shape[0] != len(o):
            raise AssertionError("row mismatch")
        a = ad.AnnData(X=sp.csc_matrix(X), obs=pd.DataFrame(index=o.index), var=pd.DataFrame(index=allg))
        a.obs["species"] = names[s]
        a.obs["cell_ontology_class"] = o["cell_ontology_class"].to_numpy()
        a.obs["cell_type"] = a.obs["cell_ontology_class"]
        a.obs["sample_batch"] = o[unit_col].to_numpy()
        a.obs["donor"] = o[unit_col].to_numpy()
        a.obs["assay"] = o["assay"].to_numpy() if "assay" in o else "NA"
        a.obs["sex"] = o["sex"].to_numpy() if "sex" in o else "NA"
        a.obs["n_genes_cell"] = o["n_genes"].to_numpy()
        a.obs["batch"] = names[s] + "|" + cfg["species"][s]["source"] + "|" + a.obs["assay"].astype(str)
        adatas[s] = a
    # --- orthologs: measured = in every file of the species; observed = count > 0 over its units
    partners = [s for s in species if s != "hsapiens"]
    measured = {s: set(pd.read_csv(species_dir(src, s) / "measured_genes.txt", header=None)[0].astype(str)) for s in species}
    if "hsapiens" not in species:
        measured["hsapiens"] = None
    table, rep_t = OR.build(partners, measured)
    table.to_csv(CF.table_path(cohort), index=False)
    out_rep["ortholog_report"] = rep_t
    out_rep["ortholog_sha256"] = OR.sha256(CF.table_path(cohort))
    observed = {}
    for s in species:
        tot = np.asarray(pbs[s].X.sum(axis=0)).ravel()
        observed[s] = set(pbs[s].var_names[tot > 0])
    seen = np.ones(len(table), bool)
    n_never = {}
    for s in species:
        m = table[f"{s}_gene_id"].isin(observed[s]).to_numpy()
        n_never[s] = int((~m).sum())
        seen &= m
    # --- features, all from chimera._data_utils, unchanged
    prior = compute_conservation_prior(table)
    esm_species = cc.get("esm_species", species)
    consensus, per_sp = download_and_process_esm2(table, species_list=esm_species, saturn_data_dir=link_esm(cohort, esm_species))
    esm_matched = {k: int(np.any(v != 0, axis=1).sum()) for k, v in per_sp.items()}
    esm_pca = pca_esm2(consensus, n_components=64)
    esm_var = compute_esm2_interspecies_variance(per_sp)
    G = len(table)
    unified = construct_unified_adata({s: adatas[s] for s in species}, table, np.zeros((G, 72), dtype=np.float32), prior)
    unified.X = sp.csr_matrix(unified.X, dtype=np.float32)
    stats = compute_expression_stats(unified)
    feats, scaler = assemble_gin_features(esm_pca, esm_var, table, stats)
    unified.varm["gene_features"] = feats
    nnz = np.asarray((unified.X != 0).sum(axis=0)).ravel()
    gene_mask = seen & (nnz >= 3)
    obj = unified[:, gene_mask].copy()
    obj.uns["p_null"] = precompute_p_null(obj, species_key="species")
    for c in ("species", "batch", "sample_batch", "cell_type", "cell_ontology_class", "donor", "assay", "sex"):
        obj.obs[c] = obj.obs[c].astype(str).astype("category")
    with open(CF.scaler_path(cohort), "wb") as f:
        pickle.dump(scaler, f)
    path = CF.h5ad(cohort)
    t72._write_h5ad(obj, path)
    # --- checks and record
    errs = check_cohort(str(path), batch_key="batch", celltype_key="cell_ontology_class")
    X = obj.X
    lib = np.asarray(X.sum(1)).ravel()
    sp_lab = obj.obs["species"].astype(str).to_numpy()
    med = {str(k): float(np.median(lib[sp_lab == k])) for k in np.unique(sp_lab)}
    Xc = obj.X.tocsc()
    zero_cells = {str(k): int((np.asarray((Xc[sp_lab == k] > 0).sum(axis=0)).ravel() == 0).sum()) for k in np.unique(sp_lab)}
    out_rep.update(
        n_obs=int(obj.n_obs), n_vars=int(obj.n_vars), steps_per_epoch=int(obj.n_obs / 256), n_genes_table=G,
        n_genes_never_observed_in_a_species=n_never, n_genes_lt3_cells=int((seen & (nnz < 3)).sum()),
        object_genes_zero_training_cells=zero_cells, esm_matched=esm_matched, esm_species=esm_species, n_constant_feature_columns=int((feats.std(axis=0) == 0).sum()),
        frac_prior_ge_095=float((obj.var["conservation_prior"] >= 0.95).mean()), frac_prior_ge_085=float((obj.var["conservation_prior"] >= 0.85).mean()),
        median_library=med, depth_ratio_to_testis3sp=float(np.mean(list(med.values())) / 1097.0), x_max=float(X.data.max()),
        x_integer=bool(np.allclose(X.data, np.round(X.data))), layers_empty=len(obj.layers) == 0, batches=sorted(map(str, obj.obs["batch"].unique())),
        n_sample_batches=int(obj.obs["sample_batch"].nunique()), check_cohort_errors=errs, species_reports=reports,
        spearman_prior_table=float(pd.Series(obj.var["conservation_prior"].to_numpy()).corr(pd.Series(table["mean_identity"].to_numpy()[gene_mask]), method="spearman")),
        per_species_cells=obj.obs["species"].value_counts().astype(int).to_dict(), sha256=OR.sha256(path), filters=filt_rows)
    rd = CF.RES / cohort
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "build_report.json").write_text(json.dumps(out_rep, indent=2, default=str))
    pd.DataFrame(filt_rows).to_csv(rd / "filters.csv", index=False)
    pd.crosstab(obj.obs["cell_ontology_class"].astype(str), obj.obs["species"].astype(str)).to_csv(rd / "counts_type_by_species.tsv", sep="\t")
    if errs:
        raise AssertionError(f"check_cohort: {errs}")
    log.info("%s: %d cells x %d genes, depth ratio %.2f, frac pi>=0.95 %.3f", cohort, obj.n_obs, obj.n_vars, out_rep["depth_ratio_to_testis3sp"], out_rep["frac_prior_ge_095"])


def build_subset(tag: str) -> None:
    sys.path.insert(0, str(gene_ids.REPO_ROOT / "training"))
    from build_subset import build_subset as bs  # noqa: PLC0415

    s = CF.SUBSETS[tag]
    path = CF.h5ad(tag)
    bs(source_path=CF.h5ad(s["parent"]), species=tuple(s["species"]), dest_path=path, drop_empty_genes=True, min_cells_per_gene=3)
    a = ad.read_h5ad(path)
    for c in ("species", "batch", "sample_batch", "cell_type", "cell_ontology_class", "donor"):
        a.obs[c] = a.obs[c].astype(str).astype("category")
    t72._write_h5ad(a, path)
    errs = check_cohort(str(path), batch_key="batch", celltype_key="cell_ontology_class")
    lib = np.asarray(a.X.sum(1)).ravel()
    spl = a.obs["species"].astype(str).to_numpy()
    med = {str(k): float(np.median(lib[spl == k])) for k in np.unique(spl)}
    rd = CF.RES / tag
    rd.mkdir(parents=True, exist_ok=True)
    (rd / "build_report.json").write_text(json.dumps(dict(cohort=tag, parent=s["parent"], n_obs=int(a.n_obs), n_vars=int(a.n_vars), steps_per_epoch=int(a.n_obs / 256),
                                                          median_library=med, depth_ratio_to_testis3sp=float(np.mean(list(med.values())) / 1097.0), check_cohort_errors=errs,
                                                          frac_prior_ge_095=float((a.var["conservation_prior"] >= 0.95).mean()),
                                                          per_species_cells=a.obs["species"].value_counts().astype(int).to_dict(), sha256=OR.sha256(path), code_sha=git_sha()), indent=2))
    pd.crosstab(a.obs["cell_ontology_class"].astype(str), a.obs["species"].astype(str)).to_csv(rd / "counts_type_by_species.tsv", sep="\t")
    if errs:
        raise AssertionError(f"check_cohort: {errs}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cohort", nargs="?")
    ap.add_argument("--subset")
    ap.add_argument("--species-pass", help="source:species, one species pass only")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if a.species_pass:
        src, s = a.species_pass.split(":")
        species_pass(src, s, a.force)
    elif a.subset:
        build_subset(a.subset)
    else:
        build_cohort(a.cohort)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
