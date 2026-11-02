"""C1: Ensembl 110 one-to-one ortholog table for a set of species, joined on the human gene.

Same rules as T12.1 and P13 (``T12_1_bone_marrow/build_orthologs.py``): Compara 110
``protein_default.homologies`` dumps of both species of each pair, because a dump writes each
homology once under one of its two species; rows with ``homology_type == ortholog_one2one``; 1:1
enforced on both sides; ``<sp>_identity`` = Compara percent of the human protein identical to the
other protein (the quantity BioMart reports, checked by T12.1 against the canonical testis table);
``mean_identity`` = mean over the non-human species of the table; symbols from the Ensembl 110 GTFs.
Human is the hub in every table, also for the regeneration cohort, which has no human cells: Ensembl
IDs of human are the only valid join across cohorts (``revision/common/gene_ids.py``), and the prior
then reads as identity to human in every cohort.

Also kept: ``<sp>_dn``, ``<sp>_ds`` (Compara, human against that species; NaN where absent) for the
per-cohort dN/dS beside the human-mouse gold.

A group is kept only if every gene is in the measured feature list of its species (``measured``):
a gene not in a species' list is a missing gene, not a zero. Rows are ordered by human Ensembl ID and
numbered from 0.
"""

from __future__ import annotations

import gzip
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from revision.common import gene_ids

ROOT = gene_ids.REPO_ROOT
E110 = ROOT / "data" / "revision" / "ensembl110"
BM = ROOT / "data" / "revision" / "bone_marrow" / "ensembl110"
NEW = ROOT / "data" / "revision" / "new_usecases" / "ensembl110"

# Short name -> Compara species, dump, GTF.
SPECIES = {
    "hsapiens": dict(compara="homo_sapiens", dump=E110 / "Compara.110.protein_default.homologies.tsv.gz",
                     gtf=E110 / "Homo_sapiens.GRCh38.110.gtf.gz"),
    "mmusculus": dict(compara="mus_musculus", dump=BM / "Compara.110.protein_default.homologies.mus_musculus.tsv.gz",
                      gtf=BM / "Mus_musculus.GRCm39.110.gtf.gz"),
    "mmurinus": dict(compara="microcebus_murinus", dump=BM / "Compara.110.protein_default.homologies.microcebus_murinus.tsv.gz",
                     gtf=BM / "Microcebus_murinus.Mmur_3.0.110.gtf.gz"),
    "ptroglodytes": dict(compara="pan_troglodytes", dump=E110 / "by_species/pan_troglodytes/Compara.110.protein_default.homologies.tsv.gz",
                         gtf=NEW / "Pan_troglodytes.Pan_tro_3.0.110.gtf.gz", pep=NEW / "Pan_troglodytes.Pan_tro_3.0.pep.all.fa.gz"),
    "cjacchus": dict(compara="callithrix_jacchus", dump=E110 / "by_species/callithrix_jacchus/Compara.110.protein_default.homologies.tsv.gz",
                     gtf=NEW / "Callithrix_jacchus.mCalJac1.pat.X.110.gtf.gz", pep=NEW / "Callithrix_jacchus.mCalJac1.pat.X.pep.all.fa.gz"),
    "mmulatta": dict(compara="macaca_mulatta", dump=NEW / "Compara.110.protein_default.homologies.macaca_mulatta.tsv.gz",
                     gtf=NEW / "Macaca_mulatta.Mmul_10.110.gtf.gz"),
    "drerio": dict(compara="danio_rerio", dump=NEW / "Compara.110.protein_default.homologies.danio_rerio.tsv.gz",
                   gtf=NEW / "Danio_rerio.GRCz11.110.gtf.gz"),
    "ggallus": dict(compara="gallus_gallus", dump=NEW / "Compara.110.protein_default.homologies.gallus_gallus.tsv.gz",
                    gtf=NEW / "Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.110.gtf.gz",
                    pep=NEW / "Gallus_gallus.bGalGal1.mat.broiler.GRCg7b.pep.all.fa.gz"),
}


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


def pair_rows(partner: str, cache: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """One-to-one human-partner rows from both dumps, oriented human first."""
    comp = SPECIES[partner]["compara"]
    cols = ["gene_stable_id", "species", "identity", "homology_type", "homology_gene_stable_id", "homology_species",
            "homology_identity", "dn", "ds"]
    out, rep = [], {}
    for path in (SPECIES["hsapiens"]["dump"], SPECIES[partner]["dump"]):
        df = pd.read_csv(path, sep="\t", usecols=cols, dtype=str)
        df = df[df["homology_type"] == "ortholog_one2one"]
        fwd = df[(df["species"] == "homo_sapiens") & (df["homology_species"] == comp)]
        rev = df[(df["species"] == comp) & (df["homology_species"] == "homo_sapiens")]
        rev = rev.rename(columns={"gene_stable_id": "homology_gene_stable_id", "homology_gene_stable_id": "gene_stable_id",
                                  "species": "homology_species", "homology_species": "species",
                                  "identity": "homology_identity", "homology_identity": "identity"})
        rep[path.name] = dict(human_first=len(fwd), human_second=len(rev))
        out += [fwd[cols], rev[cols]]
    d = pd.concat(out, ignore_index=True).drop_duplicates(["gene_stable_id", "homology_gene_stable_id"])
    for c in ("identity", "homology_identity", "dn", "ds"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    for c in ("gene_stable_id", "homology_gene_stable_id"):
        dup = d[c].duplicated(keep=False)
        rep[f"dropped_non_unique_{c}"] = int(dup.sum())
        d = d[~dup]
    rep["n_one2one"] = len(d)
    return d.rename(columns={"gene_stable_id": "hsapiens_gene_id", "homology_gene_stable_id": f"{partner}_gene_id",
                             "identity": f"{partner}_identity", "homology_identity": f"{partner}_perc_id_r1",
                             "dn": f"{partner}_dn", "ds": f"{partner}_ds"})[
        ["hsapiens_gene_id", f"{partner}_gene_id", f"{partner}_identity", f"{partner}_perc_id_r1", f"{partner}_dn", f"{partner}_ds"]], rep


def build(partners: list[str], measured: dict[str, set], compute_prior_over: list[str] | None = None) -> tuple[pd.DataFrame, dict]:
    """Table over human + ``partners``; ``measured`` per species (set of Ensembl gene IDs, or None for no filter)."""
    rep: dict = {}
    names = {sp: gtf_names(SPECIES[sp]["gtf"]) for sp in ["hsapiens", *partners]}
    t = None
    for sp in partners:
        d, r = pair_rows(sp)
        rep[sp] = r
        t = d if t is None else t.merge(d, on="hsapiens_gene_id", how="inner")
    rep["n_inner_join"] = len(t)
    keep = np.ones(len(t), bool)
    rep["n_not_measured"] = {}
    for sp in ["hsapiens", *partners]:
        if measured.get(sp) is None:
            continue
        m = t[f"{sp}_gene_id"].isin(measured[sp]).to_numpy()
        rep["n_not_measured"][sp] = int((~m).sum())
        keep &= m
    t = t[keep].sort_values("hsapiens_gene_id", kind="stable").reset_index(drop=True)
    t.insert(0, "ortholog_group_id", np.arange(len(t)))
    for sp in ["hsapiens", *partners]:
        t[f"{sp}_gene_name"] = t[f"{sp}_gene_id"].map(names[sp])
    t["mean_identity"] = t[[f"{sp}_identity" for sp in (compute_prior_over or partners)]].mean(axis=1)
    if t["mean_identity"].isna().any():
        raise ValueError("missing identity")
    rep["n_groups"] = len(t)
    rep["frac_mean_identity_ge_95"] = float((t["mean_identity"] >= 95).mean())
    rep["n_missing_name"] = {sp: int(t[f"{sp}_gene_name"].isna().sum()) for sp in ["hsapiens", *partners]}
    return t, rep
