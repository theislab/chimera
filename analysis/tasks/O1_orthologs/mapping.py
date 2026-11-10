"""Step 1b: which raw row fed each ortholog group of the canonical objects. Counts of the training objects and raw CAMEX matrices; no readout or gold.

The canonical build joins on the species gene symbol (``construct_unified_adata`` picks the column that matches more var names; all matrices are indexed by
symbol). Raw files keep an Ensembl ID where they have one (liver human and mouse ``gene_ids``, testis ``gene``). So for every group and species:
  name_row   raw row carrying the table's symbol;  id_row  raw row carrying the table's Ensembl ID;
  object_is  'name_row' when the object's column equals the name row cell by cell on the object's cells, 'none' otherwise;
  M_wrong_row  name_row exists and carries another Ensembl ID than the table's, when that ID is a different gene in the Ensembl 110 GTF
               (an ID absent from 110 is reported as retired, not flagged);
  M_false_zero object total 0 and the raw matrix has counts on the id_row, on the object's cells.
Liver macaque and zebrafish matrices carry no ID for named genes: only the symbol can be checked, and they are reported on that basis.

    PYTHONPATH=$PWD python -m analysis.tasks.O1_orthologs.mapping
"""

from __future__ import annotations

import gzip
import re
import warnings

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from analysis.tasks.O1_orthologs import rules as RU

warnings.filterwarnings("ignore", message=".*not unique.*")
OBJ = {"liver": RU.REPO / "data/processed/camex_pancreas_unified.h5ad", "testis": RU.REPO / "data/processed/camex_testis_unified.h5ad"}
SP = {"Human": "hsapiens", "Monkey": "mmulatta", "Macaque": "mmulatta", "Mouse": "mmusculus", "Zebrafish": "drerio"}


def raw_ids(a) -> pd.Series | None:
    for c in ("gene_ids", "gene"):
        if c in a.var:
            return a.var[c].astype(str)
    return None


def one(table: str, species: str, obj) -> pd.DataFrame:
    sp_key = SP[species]
    t = pd.read_csv(RU.TABLES[table]).set_index("ortholog_group_id")
    ids = t[f"{sp_key}_gene_id"].astype(str)
    names = t[f"{sp_key}_gene_name"]
    raw = ad.read_h5ad(RU.RAW_DIR[table] / RU.RAW[table][species])
    raw.var_names_make_unique()
    m = obj[obj.obs["species"].astype(str) == species]
    cells = pd.Index(m.obs_names.astype(str))
    pos = raw.obs_names.get_indexer(cells.str.replace(r"-\d+$", "", regex=True) if False else cells)
    if (pos < 0).mean() > 0.5:  # concat suffix on barcodes
        alt = cells.str.replace(r"-[^-]*$", "", regex=True)
        p2 = raw.obs_names.get_indexer(alt)
        if (p2 >= 0).mean() > (pos >= 0).mean():
            pos = p2
    if len(set(pos[pos >= 0])) < (pos >= 0).sum():
        pos = np.full(len(cells), -1)  # barcodes not unique in this file: compare on totals only
    matched = float((pos >= 0).mean())
    X = sp.csr_matrix(raw.X).astype(np.float64)
    full = np.asarray(X.sum(0)).ravel()
    Xs = X[pos[pos >= 0]] if matched > 0 else X
    rid = raw_ids(raw)
    rnames = pd.Index(raw.var_names.astype(str))
    Xo = sp.csc_matrix(m.X).astype(np.float64)
    tot_obj = np.asarray(Xo.sum(0)).ravel()
    gi = [int(v.split("_")[1]) for v in m.var_names]
    nm = names.reindex(gi).to_numpy()
    idv = ids.reindex(gi).to_numpy()
    name_pos = rnames.get_indexer(pd.Index(pd.Series(nm).fillna("__none__").astype(str)))
    id_pos = pd.Index(rid.to_numpy()).get_indexer(pd.Index(idv)) if rid is not None and rid.is_unique else (pd.Series(rid.to_numpy()).drop_duplicates().reset_index().set_index(0)["index"].reindex(idv).fillna(-1).astype(int).to_numpy() if rid is not None else np.full(len(gi), -1))
    Xc = sp.csc_matrix(Xs)
    tot_sub = np.asarray(Xc.sum(0)).ravel()
    out = pd.DataFrame(dict(table=table, species=sp_key, obj_species=species, ortholog_group_id=gi, var_name=m.var_names.astype(str), table_id=idv, table_name=nm,
                            obj_total=tot_obj, name_row=name_pos, id_row=id_pos))
    out["name_total_sub"] = np.where(name_pos >= 0, tot_sub[np.maximum(name_pos, 0)], np.nan)
    out["id_total_full"] = np.where(id_pos >= 0, full[np.maximum(id_pos, 0)], np.nan)
    out["id_total_sub"] = np.where(id_pos >= 0, tot_sub[np.maximum(id_pos, 0)], np.nan)
    out["name_total_full"] = np.where(name_pos >= 0, full[np.maximum(name_pos, 0)], np.nan)
    out["name_row_id"] = [rid.iloc[p] if (rid is not None and p >= 0) else None for p in name_pos]
    out["cell_match_share"] = matched
    # cell-by-cell check, only where all cells are found
    eq = np.full(len(gi), np.nan)
    if matched == 1.0:
        have = np.flatnonzero(name_pos >= 0)
        A = Xc[:, name_pos[have]]
        B = Xo[:, have]
        diff = (A != B)
        nn = np.asarray(diff.sum(0)).ravel()
        eq[have] = (nn == 0).astype(float)
    out["cellwise_equal_name_row"] = eq
    # object zero, raw has counts under the ID, or under the name
    out["id_differs_row"] = (id_pos >= 0) & (name_pos != id_pos)
    print(f"{table} {species}: cells {m.n_obs}, matched {matched:.3f}, groups {len(gi)}, name row {int((name_pos >= 0).sum())}, id row {int((id_pos >= 0).sum())}, "
          f"cellwise equal {np.nansum(eq):.0f} of {np.isfinite(eq).sum()}", flush=True)
    return out


def add_no_id_routes(m: pd.DataFrame) -> pd.DataFrame:
    """Matrices without an ID on named rows. Liver macaque: the matrix is the M. fascicularis 5.0 annotation (Ensembl 98; every row name is a gene of it), the table is
    M. mulatta. A zero is false when the human gene's Compara 110 one-to-one M. fascicularis gene has counts, under its ID row or its 98 name. Zebrafish: same name, other case."""
    c = pd.read_parquet(RU.DATA / "compara110_human_rows.parquet")
    r = ad.read_h5ad(RU.RAW_DIR["liver"] / RU.RAW["liver"]["Monkey"])
    tot = pd.Series(np.asarray(r.X.sum(0)).ravel(), index=r.var_names.astype(str))
    tot = tot.groupby(level=0).sum()
    n98 = {}
    with gzip.open(RU.Y6 / "gtf98/Macaca_fascicularis.Macaca_fascicularis_5.0.98.gtf.gz", "rt") as f:
        for ln in f:
            if ln.startswith("#"):
                continue
            p = ln.split("\t", 8)
            if p[2] == "gene":
                g = re.search(r'gene_id "([^"]+)"', p[8]).group(1)
                n = re.search(r'gene_name "([^"]+)"', p[8])
                n98[g] = n.group(1) if n else ""
    mf = c[(c.species_ens == "macaca_fascicularis") & (c.source == "protein") & (c.homology_type == "ortholog_one2one")].drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id").gene_id
    t = pd.read_csv(RU.TABLES["liver"]).set_index("ortholog_group_id")
    x = m[(m.table == "liver") & (m.obj_species == "Monkey")]
    gid = t.hsapiens_gene_id.reindex(x.ortholog_group_id).map(mf)
    by_id = tot.reindex(gid.to_numpy()).fillna(0).to_numpy()
    by_nm = tot.reindex(gid.map(n98).to_numpy()).fillna(0).to_numpy()
    m["alt_counts"] = np.nan
    m.loc[x.index, "alt_counts"] = by_id + by_nm
    m.loc[x.index, "M_false_zero"] = (x.obj_total.to_numpy() == 0) & ((by_id + by_nm) > 0)
    r = ad.read_h5ad(RU.RAW_DIR["liver"] / RU.RAW["liver"]["Zebrafish"])
    tot = pd.Series(np.asarray(r.X.sum(0)).ravel(), index=r.var_names.astype(str).str.lower()).groupby(level=0).sum()
    x = m[(m.table == "liver") & (m.obj_species == "Zebrafish")]
    v = tot.reindex(x.table_name.fillna("__none__").str.lower().to_numpy()).fillna(0).to_numpy()
    m.loc[x.index, "alt_counts"] = v
    m.loc[x.index, "M_false_zero"] = (x.obj_total.to_numpy() == 0) & (v > 0)
    m["M_false_zero"] = m["M_false_zero"].astype(bool)
    m["M_structural_zero"] = (m.obj_total == 0) & ~m.M_false_zero
    return m


def main():
    gtf = pd.read_parquet(RU.DATA / "gtf110_genes.parquet")
    frames = []
    for table, path in OBJ.items():
        obj = ad.read_h5ad(path)
        for species in RU.RAW[table]:
            frames.append(one(table, species, obj))
    m = pd.concat(frames, ignore_index=True)
    known = gtf.groupby("species").gene_id.agg(set).to_dict()
    in110 = [(r in known.get(s, set())) if r is not None and str(r).startswith("ENS") else False for r, s in zip(m.name_row_id, m.species)]
    m["name_row_id_in_110"] = in110
    has_id = m.name_row_id.notna() & m.name_row_id.astype(str).str.startswith("ENS")
    m["ids_differ"] = has_id & (m.name_row_id.astype(str) != m.table_id.astype(str))
    m["M_wrong_row"] = m.ids_differ & m.name_row_id_in_110
    m["M_retired_row_id"] = m.ids_differ & ~m.name_row_id_in_110
    m["M_false_zero"] = (m.obj_total == 0) & (m.id_total_sub.fillna(0) > 0)
    m["no_raw_row"] = (m.name_row < 0) & (m.id_row < 0)
    m["M_structural_zero"] = (m.obj_total == 0) & m.no_raw_row
    m = add_no_id_routes(m)
    m.to_parquet(RU.RES / "mapping_rows.parquet")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    g = m.groupby(["table", "obj_species"])
    s = pd.DataFrame(dict(groups=g.size(), zero_obj=g.obj_total.apply(lambda x: int((x == 0).sum())), no_raw_row=g.no_raw_row.sum(), name_row=g.name_row.apply(lambda x: int((x >= 0).sum())),
                          id_row=g.id_row.apply(lambda x: int((x >= 0).sum())), id_row_not_name=g.id_differs_row.sum(), M_wrong_row=g.M_wrong_row.sum(), M_retired=g.M_retired_row_id.sum(),
                          M_false_zero=g.M_false_zero.sum(), structural_zero=g.M_structural_zero.sum(), cellwise_equal=g.cellwise_equal_name_row.sum(), cellwise_n=g.cellwise_equal_name_row.count()))
    s.to_csv(RU.RES / "mapping_summary.csv")
    print(s)


if __name__ == "__main__":
    main()
