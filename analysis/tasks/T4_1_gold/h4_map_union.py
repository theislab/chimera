"""Rebuild the 1:1 ortholog maps from both sides of the Ensembl Compara homology dumps.

Ensembl writes each homology pair to one species' TSV only, so a map read from the human
file alone misses pairs. This script takes the union of the human file and the partner
species' file (deduplicated on ``homology_id``) and keeps pairs that are 1:1 in both
directions (one partner per human gene and one human gene per partner).

Outputs (``<out>``):
- ``h4_ortholog_map_one2one_union.tsv.gz``: the H4 map (8 Murat species; release 110,
  chicken 105), same columns as ``h4_ortholog_map_one2one.tsv.gz`` plus ``source_file``.
- ``dnds99_hmouse_union.tsv``: human-mouse 1:1, Compara 99, ``dn``, ``ds`` (H8).
- ``ortholog_map_union_counts.tsv``: pairs and eligible genes before and after.

No Chimera readout is read. Usage: python -I h4_map_union.py <chimera_repo> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PRIMATES = ["bonobo", "chimpanzee", "gorilla", "gibbon", "marmoset"]
OUTGROUPS = ["opossum", "platypus", "chicken"]
ENSEMBL_SPECIES = {"bonobo": "pan_troglodytes", "chimpanzee": "pan_troglodytes",
                   "gorilla": "gorilla_gorilla", "gibbon": "nomascus_leucogenys",
                   "marmoset": "callithrix_jacchus", "opossum": "monodelphis_domestica",
                   "platypus": "ornithorhynchus_anatinus", "chicken": "gallus_gallus"}
COLS = ["gene_stable_id", "species", "homology_type", "homology_gene_stable_id", "homology_species",
        "is_high_confidence", "homology_id", "dn", "ds"]


def read_pairs(path: Path, partner: str, human_side: bool) -> pd.DataFrame:
    """1:1 human-partner rows of one dump, as (human, partner) columns."""
    parts = []
    for ch in pd.read_csv(path, sep="\t", dtype=str, usecols=COLS, chunksize=2_000_000):
        ch = ch[ch.homology_type == "ortholog_one2one"]
        if human_side:
            ch = ch[ch.homology_species == partner]
            parts.append(pd.DataFrame({"hsapiens_gene_id": ch.gene_stable_id, "species_gene_id": ch.homology_gene_stable_id,
                                       "is_high_confidence": ch.is_high_confidence, "homology_id": ch.homology_id,
                                       "dn": ch.dn, "ds": ch.ds}))
        else:
            ch = ch[(ch.homology_species == "homo_sapiens") & (ch.species == partner)]
            parts.append(pd.DataFrame({"hsapiens_gene_id": ch.homology_gene_stable_id, "species_gene_id": ch.gene_stable_id,
                                       "is_high_confidence": ch.is_high_confidence, "homology_id": ch.homology_id,
                                       "dn": ch.dn, "ds": ch.ds}))
    return pd.concat(parts, ignore_index=True)


def union(h: pd.DataFrame, s: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    h = h.assign(source_file="human")
    s = s.assign(source_file="partner")
    u = pd.concat([h, s], ignore_index=True)
    both = set(h.homology_id) & set(s.homology_id)
    u.loc[u.homology_id.isin(both), "source_file"] = "both"
    u = u.drop_duplicates("homology_id")
    n0 = len(u)
    # 1:1 in both directions
    u = u[~u.hsapiens_gene_id.duplicated(keep=False) & ~u.species_gene_id.duplicated(keep=False)]
    info = dict(human_file_pairs=len(h), partner_file_pairs=len(s), in_both=len(both),
                union_pairs=n0, union_one2one_both_directions=len(u),
                only_partner_file=int((u.source_file == "partner").sum()))
    return u, info


def main(repo: Path, out: Path):
    out.mkdir(parents=True, exist_ok=True)
    E = repo / "data/analysis/ensembl110"
    rows, maps = [], []
    var = {}
    import anndata as ad

    for sp in PRIMATES + OUTGROUPS:
        a = ad.read_h5ad(repo / f"data/raw/camex/2testis/dataset/raw-testis-{sp}-Murat.h5ad", backed="r")
        var[sp] = set(a.var["gene"].astype(str).str.replace(r"\.\d+$", "", regex=True))
        a.file.close()
    cache = {}
    for sp in PRIMATES + OUTGROUPS:
        es = ENSEMBL_SPECIES[sp]
        rel = "105" if es == "gallus_gallus" else "110"
        if es not in cache:
            h = read_pairs(E / f"Compara.{rel}.protein_default.homologies.tsv.gz", es, human_side=True)
            s = read_pairs(E / "by_species" / es / f"Compara.{rel}.protein_default.homologies.tsv.gz", es,
                           human_side=False)
            cache[es] = union(h, s) + (h,)
        u, info, h = cache[es]
        old = h[h.species_gene_id.isin(var[sp])]
        new = u[u.species_gene_id.isin(var[sp])]
        rows.append(dict(map="H4", species=sp, ensembl_species=es, release=rel, **info,
                         var_genes_mapped_human_file=old.hsapiens_gene_id.nunique(),
                         var_genes_mapped_union=new.hsapiens_gene_id.nunique()))
        maps.append(new.assign(murat_species=sp, ensembl_species=es, ensembl_release=rel))
        print(rows[-1], flush=True)
    m = pd.concat(maps, ignore_index=True)[["hsapiens_gene_id", "murat_species", "ensembl_species", "species_gene_id",
                                            "is_high_confidence", "ensembl_release", "source_file"]]
    m.to_csv(out / "h4_ortholog_map_one2one_union.tsv.gz.tmp", sep="\t", index=False, compression="gzip")
    Path(out / "h4_ortholog_map_one2one_union.tsv.gz.tmp").replace(out / "h4_ortholog_map_one2one_union.tsv.gz")
    old = pd.read_csv(out / "h4_ortholog_map_one2one.tsv.gz", sep="\t", dtype=str)

    def eligible(df):
        pres = df.groupby("hsapiens_gene_id").murat_species.apply(set)
        npri = pres.apply(lambda s: len(s & set(PRIMATES)))
        nout = pres.apply(lambda s: len(s & set(OUTGROUPS)))
        return set(pres.index[(npri >= 3) & (nout >= 2)]), set(pres.index[npri >= 3])

    eo, po = eligible(old)
    en, pn = eligible(m)
    rows.append(dict(map="H4_eligible_genes", species="all", human_file_pairs=len(eo), union_one2one_both_directions=len(en),
                     note=f"primates-only: {len(po)} -> {len(pn)}; lost {len(eo - en)}, gained {len(en - eo)}"))
    # H8 dN/dS, Compara 99, human-mouse
    h = read_pairs(repo / "data/analysis/gold/G8/Compara.99.protein_default.homologies.tsv.gz", "mus_musculus", True)
    s = read_pairs(repo / "data/analysis/gold/G8/mus_musculus/Compara.99.protein_default.homologies.tsv.gz",
                   "mus_musculus", False)
    u, info = union(h, s)
    dn = pd.to_numeric(u.dn.replace("NULL", None), errors="coerce")
    ds = pd.to_numeric(u.ds.replace("NULL", None), errors="coerce")
    u = u.assign(dn=dn, ds=ds)
    hk = h.assign(ds=pd.to_numeric(h.ds.replace("NULL", None), errors="coerce"))
    hk = hk[~hk.hsapiens_gene_id.duplicated(keep=False)]
    ok_old = int(((hk.ds > 0) & (hk.ds <= 3)).sum())
    ok_new = int(((u.ds > 0) & (u.ds <= 3) & u.dn.notna()).sum())
    rows.append(dict(map="H8_dnds99_hmouse", species="mouse", release="99", **info,
                     note=f"genes with dn and ds in (0, 3]: human file {ok_old} -> union {ok_new}"))
    print(rows[-1], flush=True)
    u[["hsapiens_gene_id", "species_gene_id", "dn", "ds", "homology_id", "source_file"]].to_csv(
        out / "dnds99_hmouse_union.tsv.tmp", sep="\t", index=False)
    (out / "dnds99_hmouse_union.tsv.tmp").replace(out / "dnds99_hmouse_union.tsv")
    pd.DataFrame(rows).to_csv(out / "ortholog_map_union_counts.tsv.tmp", sep="\t", index=False)
    (out / "ortholog_map_union_counts.tsv.tmp").replace(out / "ortholog_map_union_counts.tsv")
    print(pd.DataFrame(rows).to_string(), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
