"""Y6 use 1: genome annotation labels, independent of counts. Reads downloaded files only; run with python -I.

For each species gene of every ortholog table (rows: species, human gene, species gene):
  Compara 110: homology type, identity, GOC, WGA, is_high_confidence of the pair (both dumps read; each pair sits under one species).
  Compara 116: is the same pair still ortholog_one2one.
  GTF 98, 110, 116: stable ID present, biotype, contig, strand, gene span, 3' end, CDS length of the longest protein-coding
  transcript, any annotated 3' UTR, number of transcripts, best transcript support level (mouse and human only carry TSL).
Writes results/revision/Y6/annot/genome_labels.parquet and table_pairs.parquet. No count, readout or fit is read here.

    python -I revision/tasks/Y6_uses/annot_genome.py
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
Y6 = ROOT / "data" / "revision" / "Y6"
OUT = ROOT / "results" / "revision" / "Y6" / "annot"
BM = ROOT / "data" / "revision" / "bone_marrow" / "ensembl110"
NU = ROOT / "data" / "revision" / "new_usecases" / "ensembl110"
E110 = ROOT / "data" / "revision" / "ensembl110"

SPECIES = {"mmusculus": "mus_musculus", "mmulatta": "macaca_mulatta", "mfascicularis": "macaca_fascicularis",
           "sscrofa": "sus_scrofa", "drerio": "danio_rerio", "mmurinus": "microcebus_murinus"}
GTF = {
    "mmusculus": {98: Y6 / "gtf98/Mus_musculus.GRCm38.98.gtf.gz", 110: BM / "Mus_musculus.GRCm39.110.gtf.gz", 116: Y6 / "gtf116/Mus_musculus.GRCm39.116.gtf.gz"},
    "mmulatta": {98: Y6 / "gtf98/Macaca_mulatta.Mmul_10.98.gtf.gz", 110: NU / "Macaca_mulatta.Mmul_10.110.gtf.gz", 116: Y6 / "gtf116/Macaca_mulatta.Mmul_10.116.gtf.gz"},
    "mfascicularis": {98: Y6 / "gtf98/Macaca_fascicularis.Macaca_fascicularis_5.0.98.gtf.gz", 110: Y6 / "gtf110/Macaca_fascicularis.Macaca_fascicularis_6.0.110.gtf.gz", 116: Y6 / "gtf116/Macaca_fascicularis.Macaca_fascicularis_6.0.116.gtf.gz"},
    "sscrofa": {98: Y6 / "gtf98/Sus_scrofa.Sscrofa11.1.98.gtf.gz", 110: Y6 / "gtf110/Sus_scrofa.Sscrofa11.1.110.gtf.gz", 116: Y6 / "gtf116/Sus_scrofa.Sscrofa11.1.116.gtf.gz"},
    "drerio": {98: Y6 / "gtf98/Danio_rerio.GRCz11.98.gtf.gz", 110: NU / "Danio_rerio.GRCz11.110.gtf.gz", 116: Y6 / "gtf116/Danio_rerio.GRCz11.116.gtf.gz"},
    "mmurinus": {98: Y6 / "gtf98/Microcebus_murinus.Mmur_3.0.98.gtf.gz", 110: BM / "Microcebus_murinus.Mmur_3.0.110.gtf.gz", 116: Y6 / "gtf116/Microcebus_murinus.Mmur_3.0.116.gtf.gz"},
}
# Same assembly across the two releases, so coordinates compare directly (mouse 98 is GRCm38, fascicularis 98 is 5.0).
SAME_ASSEMBLY_98_110 = {"mmulatta", "sscrofa", "drerio", "mmurinus"}
COMPARA_110 = {"homo_sapiens": E110 / "Compara.110.protein_default.homologies.tsv.gz",
               "mus_musculus": BM / "Compara.110.protein_default.homologies.mus_musculus.tsv.gz",
               "microcebus_murinus": BM / "Compara.110.protein_default.homologies.microcebus_murinus.tsv.gz",
               "danio_rerio": NU / "Compara.110.protein_default.homologies.danio_rerio.tsv.gz",
               "macaca_mulatta": NU / "Compara.110.protein_default.homologies.macaca_mulatta.tsv.gz",
               "sus_scrofa": Y6 / "compara110/sus_scrofa/Compara.110.protein_default.homologies.tsv.gz",
               "macaca_fascicularis": Y6 / "compara110/macaca_fascicularis/Compara.110.protein_default.homologies.tsv.gz"}
COMPARA_116 = {s: Y6 / f"compara116/{s}/Compara.116.protein_default.homologies.tsv.gz"
               for s in ["homo_sapiens", "mus_musculus", "macaca_mulatta", "macaca_fascicularis", "sus_scrofa", "danio_rerio", "microcebus_murinus"]}
TABLES = {"liver": ROOT / "data/processed/ortholog_table.csv", "testis": ROOT / "data/processed/ortholog_table_testis.csv",
          "van_zyl": ROOT / "data/revision/van_zyl/ortholog_table_van_zyl.csv",
          "bone_marrow": ROOT / "data/revision/bone_marrow/ortholog_table_bone_marrow.csv",
          "kidney": ROOT / "data/revision/disease_kidney/ortholog_table_kidney.csv"}
TSV_KEYS = ("gene_stable_id", "homology_gene_stable_id", "homology_species", "species", "homology_type", "identity",
            "homology_identity", "goc_score", "wga_coverage", "is_high_confidence")
RE_ATTR = re.compile(r'(\w+) "([^"]*)"')


def table_pairs() -> pd.DataFrame:
    rows = []
    for name, path in TABLES.items():
        t = pd.read_csv(path)
        for sp in SPECIES:
            if f"{sp}_gene_id" not in t:
                continue
            r = t[["hsapiens_gene_id", f"{sp}_gene_id", f"{sp}_identity"]].dropna(subset=["hsapiens_gene_id", f"{sp}_gene_id"])
            r.columns = ["hsapiens_gene_id", "gene_id", "identity_table"]
            r.insert(0, "species", sp)
            r.insert(0, "table", name)
            rows.append(r)
    return pd.concat(rows, ignore_index=True)


def read_compara(files: dict, pairs: set, want: set) -> pd.DataFrame:
    """Rows of the dumps between human and a species gene; oriented human first. Kept only for wanted species or table pairs."""
    keep = []
    for owner, path in files.items():
        with gzip.open(path, "rt") as f:
            head = f.readline().rstrip("\n").split("\t")
            ix = {k: head.index(k) for k in TSV_KEYS}
            for line in f:
                p = line.rstrip("\n").split("\t")
                s1, s2 = p[ix["species"]], p[ix["homology_species"]]
                if s1 == "homo_sapiens" and s2 in want:
                    h, g, sp, idn = p[ix["gene_stable_id"]], p[ix["homology_gene_stable_id"]], s2, p[ix["homology_identity"]]
                elif s2 == "homo_sapiens" and s1 in want:
                    h, g, sp, idn = p[ix["homology_gene_stable_id"]], p[ix["gene_stable_id"]], s1, p[ix["identity"]]
                else:
                    continue
                if (sp, h, g) not in pairs and (sp, g) not in pairs:
                    continue
                keep.append((sp, h, g, p[ix["homology_type"]], idn, p[ix["goc_score"]], p[ix["wga_coverage"]], p[ix["is_high_confidence"]]))
        print("  read", owner, len(keep), flush=True)
    d = pd.DataFrame(keep, columns=["species_ens", "hsapiens_gene_id", "gene_id", "homology_type", "identity", "goc", "wga", "high_conf"])
    for c in ["identity", "goc", "wga", "high_conf"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.drop_duplicates(["species_ens", "hsapiens_gene_id", "gene_id"])


def read_gtf(path: Path) -> pd.DataFrame:
    genes, tx, cds, utr = {}, {}, {}, set()
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.rstrip("\n").split("\t")
            kind = p[2]
            if kind not in ("gene", "transcript", "CDS", "three_prime_utr"):
                continue
            a = dict(RE_ATTR.findall(p[8]))
            g = a.get("gene_id")
            if kind == "gene":
                genes[g] = dict(contig=p[0], start=int(p[3]), end=int(p[4]), strand=p[6], biotype=a.get("gene_biotype", a.get("gene_type", "")), name=a.get("gene_name", ""))
            elif kind == "transcript":
                t = tx.setdefault(g, dict(n=0, best_tsl=99, n_pc=0))
                t["n"] += 1
                if a.get("transcript_biotype") == "protein_coding":
                    t["n_pc"] += 1
                m = re.match(r"(\d)", a.get("transcript_support_level", ""))
                if m:
                    t["best_tsl"] = min(t["best_tsl"], int(m.group(1)))
            elif kind == "CDS":
                k = (g, a.get("transcript_id"))
                cds[k] = cds.get(k, 0) + int(p[4]) - int(p[3]) + 1
            else:
                utr.add(g)
    best = {}
    for (g, _), v in cds.items():
        best[g] = max(best.get(g, 0), v)
    d = pd.DataFrame.from_dict(genes, orient="index")
    d.index.name = "gene_id"
    d["n_tx"] = pd.Series({g: v["n"] for g, v in tx.items()})
    d["best_tsl"] = pd.Series({g: v["best_tsl"] for g, v in tx.items()})
    d["cds_len"] = pd.Series(best)
    d["has_utr3"] = d.index.isin(utr)
    d["end3"] = d["end"].where(d["strand"] == "+", d["start"])
    return d


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tp = table_pairs()
    tp.to_parquet(OUT / "table_pairs.parquet")
    print(tp.groupby(["table", "species"]).size(), flush=True)
    uniq = tp[["species", "hsapiens_gene_id", "gene_id"]].drop_duplicates()
    uniq["species_ens"] = uniq.species.map(SPECIES)
    keys = set(zip(uniq.species_ens, uniq.hsapiens_gene_id, uniq.gene_id)) | set(zip(uniq.species_ens, uniq.gene_id))
    want = set(uniq.species_ens)
    c110 = read_compara(COMPARA_110, keys, want)
    c116 = read_compara(COMPARA_116, keys, want)
    c110.to_parquet(OUT / "compara110_pairs.parquet")
    c116.to_parquet(OUT / "compara116_pairs.parquet")
    u = uniq.merge(c110.rename(columns=lambda c: c if c in ("species_ens", "hsapiens_gene_id", "gene_id") else c + "_110"),
                   on=["species_ens", "hsapiens_gene_id", "gene_id"], how="left")
    u = u.merge(c116.rename(columns=lambda c: c if c in ("species_ens", "hsapiens_gene_id", "gene_id") else c + "_116"),
                on=["species_ens", "hsapiens_gene_id", "gene_id"], how="left")
    # What the species gene is orthologous to in 116, if not the same human gene (reassigned ortholog).
    other = c116[c116.homology_type.str.startswith("ortholog")].groupby(["species_ens", "gene_id"]).hsapiens_gene_id.agg(lambda x: ",".join(sorted(set(x))))
    u = u.merge(other.rename("human_partners_116").reset_index(), on=["species_ens", "gene_id"], how="left")
    parts = []
    for sp, g in u.groupby("species"):
        g = g.copy()
        tabs = {}
        for rel, path in GTF[sp].items():
            print(" gtf", sp, rel, flush=True)
            tabs[rel] = read_gtf(path)
        for rel in (98, 110, 116):
            t = tabs[rel]
            ids = g.gene_id
            g[f"in_{rel}"] = ids.isin(t.index)
            for c in ("biotype", "contig", "strand", "start", "end", "end3", "cds_len", "has_utr3", "n_tx", "best_tsl"):
                g[f"{c}_{rel}"] = ids.map(t[c])
        parts.append(g)
    out = pd.concat(parts, ignore_index=True)
    # Same-assembly model movement, fixed in rules.py.
    out.to_parquet(OUT / "genome_labels.parquet")
    print(out.shape, flush=True)


if __name__ == "__main__":
    main()
