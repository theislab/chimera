"""P13: Ensembl 110 human-mouse 1:1 ortholog table for the kidney cohort (plan section 8c).

Source: Ensembl Compara 110 ``protein_default.homologies`` TSV dumps, both the Homo sapiens
file (``data/revision/ensembl110/``) and the Mus musculus file
(``data/revision/bone_marrow/ensembl110/``, fetched in T12.1). The dump writes each homology
once, under one of its two species, so about a third of the human-mouse pairs exist only in the
mouse file; mouse-file rows are re-oriented (T12.1, ``build_orthologs.read_compara`` logic).
Rows with ``homology_type == ortholog_one2one``; 1:1 enforced on both sides.

Columns, in the canonical (BioMart-built) layout: ``ortholog_group_id``, ``hsapiens_gene_id``,
``hsapiens_gene_name``, ``mmusculus_gene_id``, ``mmusculus_gene_name``, ``mmusculus_identity``
(Compara ``identity`` = percent of the human protein identical to the mouse protein, the
quantity BioMart reports as ``mmusculus_homolog_perc_id``; checked against the canonical testis
table), ``mean_identity`` (= ``mmusculus_identity``, the mean over the one non-reference
species), and ``mmusculus_perc_id_r1`` (Compara ``homology_identity``, the reverse direction;
reported only, its name keeps it out of the identity features). Symbols from the Ensembl 110
GTFs.

Groups are kept only if both genes are in the feature list (``raw/var`` index) of the
CELLxGENE file of their species, so no gene enters as a structural zero. Rows are ordered by
human Ensembl gene ID and numbered from 0.

    python -I revision/tasks/P13_kidney/build_orthologs.py <chimera repo> [--out CSV]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


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
            m = name_re.search(p[8])
            if m:
                names[gid_re.search(p[8]).group(1)] = m.group(1)
    return names


def feature_list(path: Path) -> set[str]:
    with h5py.File(path, "r") as h:
        var = h["raw/var"]
        key = var.attrs["_index"]
        return {x.decode() if isinstance(x, bytes) else str(x) for x in var[key][()]}


def read_compara(paths: list[Path]) -> tuple[pd.DataFrame, dict]:
    cols = ["gene_stable_id", "species", "identity", "homology_type",
            "homology_gene_stable_id", "homology_species", "homology_identity"]
    out, rep = [], {}
    for path in paths:
        df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str)
        fwd = df[(df["species"] == "homo_sapiens") & (df["homology_species"] == "mus_musculus")]
        rev = df[(df["homology_species"] == "homo_sapiens") & (df["species"] == "mus_musculus")]
        rev = rev.rename(columns={"gene_stable_id": "homology_gene_stable_id",
                                  "homology_gene_stable_id": "gene_stable_id",
                                  "species": "homology_species", "homology_species": "species",
                                  "identity": "homology_identity", "homology_identity": "identity"})
        rep[path.name] = {"human_first": fwd.groupby("homology_type").size().astype(int).to_dict(),
                          "human_second": rev.groupby("homology_type").size().astype(int).to_dict()}
        out += [fwd[cols], rev[cols]]
    df = pd.concat(out, ignore_index=True)
    for c in ("identity", "homology_identity"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.drop_duplicates(subset=["gene_stable_id", "homology_gene_stable_id", "homology_type"])
    rep["n_duplicate_rows_across_dumps"] = int(n0 - len(df))
    return df, rep


def build(df: pd.DataFrame, names: dict, measured: dict) -> tuple[pd.DataFrame, dict]:
    rep: dict = {}
    d = df[df["homology_type"] == "ortholog_one2one"].rename(columns={
        "gene_stable_id": "hsapiens_gene_id", "homology_gene_stable_id": "mmusculus_gene_id",
        "identity": "mmusculus_identity", "homology_identity": "mmusculus_perc_id_r1"})
    d = d[["hsapiens_gene_id", "mmusculus_gene_id", "mmusculus_identity", "mmusculus_perc_id_r1"]]
    d = d.drop_duplicates(subset=["hsapiens_gene_id", "mmusculus_gene_id"])
    for c in ("hsapiens_gene_id", "mmusculus_gene_id"):
        dup = d[c].duplicated(keep=False)
        rep.setdefault("dropped_non_unique", {})[c] = int(dup.sum())
        d = d[~dup]
    rep["n_one2one"] = int(len(d))
    keep = d["hsapiens_gene_id"].isin(measured["hsapiens"]) & d["mmusculus_gene_id"].isin(measured["mmusculus"])
    rep["n_not_measured"] = {"hsapiens": int((~d["hsapiens_gene_id"].isin(measured["hsapiens"])).sum()),
                             "mmusculus": int((~d["mmusculus_gene_id"].isin(measured["mmusculus"])).sum())}
    d = d[keep].sort_values("hsapiens_gene_id", kind="stable").reset_index(drop=True)
    t = pd.DataFrame({
        "ortholog_group_id": np.arange(len(d)),
        "hsapiens_gene_id": d["hsapiens_gene_id"],
        "hsapiens_gene_name": d["hsapiens_gene_id"].map(names["hsapiens"]),
        "mmusculus_gene_id": d["mmusculus_gene_id"],
        "mmusculus_gene_name": d["mmusculus_gene_id"].map(names["mmusculus"]),
        "mmusculus_identity": d["mmusculus_identity"],
    })
    t["mean_identity"] = t["mmusculus_identity"]
    t["mmusculus_perc_id_r1"] = d["mmusculus_perc_id_r1"].to_numpy()
    if t["mean_identity"].isna().any():
        raise ValueError("missing identity")
    rep["n_groups"] = int(len(t))
    rep["n_missing_name"] = {c: int(t[c].isna().sum()) for c in ("hsapiens_gene_name", "mmusculus_gene_name")}
    rep["frac_mean_identity_ge_95"] = float((t["mean_identity"] >= 95).mean())
    return t, rep


def check_identity_semantics(df: pd.DataFrame, canon_path: Path) -> dict:
    canon = pd.read_csv(canon_path)
    d = df[df["homology_type"] == "ortholog_one2one"]
    m = canon.merge(d, left_on=["hsapiens_gene_id", "mmusculus_gene_id"],
                    right_on=["gene_stable_id", "homology_gene_stable_id"], how="inner")
    diff = (m["mmusculus_identity"] - m["identity"]).abs()
    return {"n_pairs": int(len(m)), "n_canonical": int(len(canon)),
            "frac_equal_1e-3": float((diff < 1e-3).mean()), "max_abs_diff": float(diff.max())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    repo = a.repo.resolve()
    e110, bm = repo / "data" / "revision" / "ensembl110", repo / "data" / "revision" / "bone_marrow" / "ensembl110"
    raw = repo / "data" / "revision" / "disease_kidney" / "raw"
    out = a.out or repo / "data" / "revision" / "disease_kidney" / "ortholog_table_kidney.csv"
    if out.resolve().is_relative_to((repo / "data" / "processed").resolve()):
        raise SystemExit("refusing to write under data/processed/")
    compara = [e110 / "Compara.110.protein_default.homologies.tsv.gz",
               bm / "Compara.110.protein_default.homologies.mus_musculus.tsv.gz"]
    gtf = {"hsapiens": e110 / "Homo_sapiens.GRCh38.110.gtf.gz", "mmusculus": bm / "Mus_musculus.GRCm39.110.gtf.gz"}
    files = {"hsapiens": raw / "human_kidney_v2.h5ad", "mmusculus": raw / "mouse_kidney_v2.h5ad"}
    canon = repo / "data" / "processed" / "ortholog_table_testis.csv"
    df, rep_dumps = read_compara(compara)
    names = {sp: gtf_names(p) for sp, p in gtf.items()}
    measured = {sp: feature_list(p) for sp, p in files.items()}
    table, rep = build(df, names, measured)
    rep["n_features_in_files"] = {sp: len(v) for sp, v in measured.items()}
    rep["compara_dumps"] = rep_dumps
    rep["identity_semantics_vs_canonical_testis"] = check_identity_semantics(df, canon)
    rep["inputs"] = {str(p.relative_to(repo)): sha256(p) for p in [*compara, *gtf.values()]}
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False)
    rep["output"] = {str(out.relative_to(repo)): sha256(out)}
    out.with_suffix(".report.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
