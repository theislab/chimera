"""T12.1: Ensembl 110 1:1 ortholog table for human, mouse lemur and mouse.

Source: Ensembl Compara 110 ``protein_default.homologies`` TSV dumps (FTP). The dump writes
each homology once, under one of the two species, so the human-mouse pairs are split between
the Homo sapiens file (``data/analysis/ensembl110/``) and the Mus musculus file; both are read
and the mouse-file rows are re-oriented (human gene and its identity taken from the
``homology_*`` columns). The Microcebus murinus file holds no human pair (checked at build).
Rows with ``homology_type == ortholog_one2one`` for {microcebus_murinus, mus_musculus}; the two
pairwise sets are joined
on the human gene (inner join), as ``chimera._data_utils.query_ortholog_table`` joins its
BioMart queries. ``{sp}_identity`` is the Compara ``identity`` column (percent of the human
protein identical to the ortholog), the quantity BioMart reports as ``{sp}_homolog_perc_id``;
``check_identity_semantics`` verifies this against the canonical testis table (mouse column).
``mean_identity`` is the mean over the two non-reference species, as in the canonical tables.
Gene symbols come from the Ensembl 110 GTFs (``gene_name``; missing names stay empty).

Groups are kept only if each species' gene is in the feature list (``raw.var_names``) of every
CELLxGENE file of that species (``RAW_FILES``): the CELLxGENE lemur and mouse files carry
filtered gene lists (16,400 and 17,943 genes), and a gene absent from one species' list would
enter the cohort as a structural zero in that species. Only gene lists are read, no counts
(Shay et al. 2013 likewise keep orthologs "measured in both compendia").

Rows are ordered by human Ensembl gene ID and numbered from 0 (``ortholog_group_id``) after
that filter, so the table is a deterministic function of the inputs.

    python analysis/tasks/T12_1_bone_marrow/build_orthologs.py [--out CSV]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
E110 = REPO / "data" / "analysis" / "ensembl110"
BM = REPO / "data" / "analysis" / "bone_marrow"
COMPARA = E110 / "Compara.110.protein_default.homologies.tsv.gz"
COMPARA_MM = BM / "ensembl110" / "Compara.110.protein_default.homologies.mus_musculus.tsv.gz"
COMPARA_MIC = BM / "ensembl110" / "Compara.110.protein_default.homologies.microcebus_murinus.tsv.gz"
GTF = {
    "hsapiens": E110 / "Homo_sapiens.GRCh38.110.gtf.gz",
    "mmusculus": BM / "ensembl110" / "Mus_musculus.GRCm39.110.gtf.gz",
    "mmurinus": BM / "ensembl110" / "Microcebus_murinus.Mmur_3.0.110.gtf.gz",
}
COMPARA_SPECIES = {"mmurinus": "microcebus_murinus", "mmusculus": "mus_musculus"}
DEFAULT_OUT = BM / "ortholog_table_bone_marrow.csv"
CANON_TESTIS = REPO / "data" / "processed" / "ortholog_table_testis.csv"
RAW_FILES = {
    "hsapiens": ["human_TS_bone_marrow.h5ad", "human_TS_spleen.h5ad"],
    "mmurinus": ["lemur_TMicrocebus_bone_marrow.h5ad", "lemur_TMicrocebus_spleen.h5ad"],
    "mmusculus": ["mouse_TMS_bone_marrow_10x.h5ad", "mouse_TMS_spleen_10x.h5ad"],
}


def feature_lists() -> dict[str, set]:
    """Per species: intersection of ``raw/var`` index over that species' files (gene IDs only)."""
    import h5py

    out = {}
    for sp, files in RAW_FILES.items():
        sets = []
        for f in files:
            with h5py.File(BM / "raw" / f, "r") as h:
                var = h["raw/var"]
                key = var.attrs["_index"]
                sets.append({x.decode() if isinstance(x, bytes) else str(x) for x in var[key][()]})
        out[sp] = set.intersection(*sets)
    return out


def sha256(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def gtf_names(path: Path) -> dict[str, str]:
    names: dict[str, str] = {}
    gid_re, name_re = re.compile(r'gene_id "([^"]+)"'), re.compile(r'gene_name "([^"]+)"')
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 9)
            if p[2] != "gene":
                continue
            gid = gid_re.search(p[8]).group(1)
            m = name_re.search(p[8])
            if m:
                names[gid] = m.group(1)
    return names


def read_compara() -> tuple[pd.DataFrame, dict]:
    """Human-oriented rows (human in ``gene_stable_id``) for the two species, from all dumps."""
    cols = ["gene_stable_id", "species", "identity", "homology_type",
            "homology_gene_stable_id", "homology_species", "homology_identity"]
    out, rep = [], {}
    for path in (COMPARA, COMPARA_MM, COMPARA_MIC):
        df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str)
        fwd = df[(df["species"] == "homo_sapiens") & df["homology_species"].isin(COMPARA_SPECIES.values())]
        rev = df[(df["homology_species"] == "homo_sapiens") & df["species"].isin(COMPARA_SPECIES.values())]
        rev = rev.rename(columns={"gene_stable_id": "homology_gene_stable_id", "homology_gene_stable_id": "gene_stable_id",
                                  "species": "homology_species", "homology_species": "species",
                                  "identity": "homology_identity", "homology_identity": "identity"})
        rep[path.name] = {"human_first": fwd.groupby(["homology_species", "homology_type"]).size().to_dict(),
                          "human_second": rev.groupby(["homology_species", "homology_type"]).size().to_dict()}
        out += [fwd[cols], rev[cols]]
    df = pd.concat(out, ignore_index=True)
    df["identity"] = pd.to_numeric(df["identity"], errors="coerce")
    n0 = len(df)
    df = df.drop_duplicates(subset=["gene_stable_id", "homology_gene_stable_id", "homology_type"])
    rep["n_duplicate_rows_across_dumps"] = n0 - len(df)
    rep = {k: ({kk: {f"{a}|{b}": int(n) for (a, b), n in vv.items()} for kk, vv in v.items()}
               if isinstance(v, dict) else v) for k, v in rep.items()}
    return df, rep


def build(df: pd.DataFrame, measured: dict[str, set]) -> tuple[pd.DataFrame, dict]:
    names = {sp: gtf_names(p) for sp, p in GTF.items()}
    rep: dict = {"n_gtf_named": {sp: len(v) for sp, v in names.items()}}
    merged = None
    for sp in sorted(COMPARA_SPECIES):  # mmurinus, mmusculus (canonical column order)
        d = df[(df["homology_species"] == COMPARA_SPECIES[sp])
               & (df["homology_type"] == "ortholog_one2one")]
        d = d.rename(columns={"gene_stable_id": "hsapiens_gene_id",
                              "homology_gene_stable_id": f"{sp}_gene_id",
                              "identity": f"{sp}_identity"})
        d = d[["hsapiens_gene_id", f"{sp}_gene_id", f"{sp}_identity"]]
        d = d.drop_duplicates(subset=["hsapiens_gene_id", f"{sp}_gene_id"])
        # 1:1 means unique on both sides; enforce it so a malformed row cannot duplicate a group.
        for c in ("hsapiens_gene_id", f"{sp}_gene_id"):
            dup = d[c].duplicated(keep=False)
            if dup.any():
                rep.setdefault("dropped_non_unique", {})[f"{sp}:{c}"] = int(dup.sum())
                d = d[~dup]
        rep.setdefault("n_one2one", {})[sp] = int(len(d))
        d.insert(2, f"{sp}_gene_name", d[f"{sp}_gene_id"].map(names[sp]))
        merged = d if merged is None else merged.merge(d, on="hsapiens_gene_id", how="inner")
    merged.insert(1, "hsapiens_gene_name", merged["hsapiens_gene_id"].map(names["hsapiens"]))
    rep["n_groups_before_measured_filter"] = int(len(merged))
    keep = np.ones(len(merged), dtype=bool)
    for sp in ("hsapiens", *sorted(COMPARA_SPECIES)):
        ok = merged[f"{sp}_gene_id"].isin(measured[sp]).to_numpy()
        rep.setdefault("n_not_measured", {})[sp] = int((~ok).sum())
        keep &= ok
    merged = merged[keep]
    merged = merged.sort_values("hsapiens_gene_id", kind="stable").reset_index(drop=True)
    merged.insert(0, "ortholog_group_id", np.arange(len(merged)))
    id_cols = [f"{sp}_identity" for sp in sorted(COMPARA_SPECIES)]
    merged["mean_identity"] = merged[id_cols].mean(axis=1)
    rep["n_groups"] = int(len(merged))
    rep["n_missing_name"] = {c: int(merged[c].isna().sum()) for c in merged.columns if c.endswith("_gene_name")}
    rep["frac_mean_identity_ge_95"] = float((merged["mean_identity"] >= 95).mean())
    return merged, rep


def check_identity_semantics(df: pd.DataFrame) -> dict:
    """Compara ``identity`` vs canonical BioMart ``mmusculus_identity`` (testis table)."""
    if not CANON_TESTIS.exists():
        return {"checked": False}
    canon = pd.read_csv(CANON_TESTIS)
    d = df[(df["homology_species"] == "mus_musculus") & (df["homology_type"] == "ortholog_one2one")]
    m = canon.merge(d, left_on=["hsapiens_gene_id", "mmusculus_gene_id"],
                    right_on=["gene_stable_id", "homology_gene_stable_id"], how="inner")
    diff = (m["mmusculus_identity"] - m["identity"]).abs()
    return {"checked": True, "n_pairs": int(len(m)), "n_canonical": int(len(canon)),
            "frac_equal_1e-3": float((diff < 1e-3).mean()), "max_abs_diff": float(diff.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    if args.out.resolve().is_relative_to((REPO / "data" / "processed").resolve()):
        raise SystemExit("refusing to write under data/processed/")
    df, rep_dumps = read_compara()
    table, rep = build(df, feature_lists())
    rep["compara_dumps"] = rep_dumps
    rep["identity_semantics_vs_canonical_testis"] = check_identity_semantics(df)
    rep["inputs"] = {p.name: sha256(p) for p in [COMPARA, COMPARA_MM, COMPARA_MIC, *GTF.values(),
                                         *(BM / "raw" / f for fs in RAW_FILES.values() for f in fs)]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)
    rep["output"] = {args.out.name: sha256(args.out)}
    args.out.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
