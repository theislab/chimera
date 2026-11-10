"""T11.1 step 3: fix the glaucoma gene list before any association (plan section 11).

Rule: Mendelian and GWAS glaucoma genes named in van Zyl et al. 2020 (PNAS 117:10339,
doi 10.1073/pnas.2001250117, PMC7229661), plus candidates prioritised in Gharahkhani et al.
2021 (Nat Commun 12:1258, doi 10.1038/s41467-020-20851-4).

Operational definition, fixed here:
  vanzyl_mendelian  the 12 genes of "Mendelian genes assessed were ..." (Results).
  vanzyl_gwas       the 6 genes of "Well-established GWAS loci genes included ..." (Results).
  vanzyl_text       further genes the Results name as glaucoma, IOP or exfoliation risk genes
                    (EFEMP1, GAS7, KALRN, PRSS23, SIX6, POU6F2, CACNA1A). The 462 GWAS
                    Catalog genes screened and the 189 of SI Fig. S3 are not named in the text
                    and are not included. ANGPTL7 and CHI3L1 are named as JCT markers only.
  gharahkhani       every gene named in a gene-evidence column of Supplementary Data 14
                    (functional annotation of all 127 POAG loci: protein-altering, LD with a
                    protein-altering variant, chromatin interaction, eQTL any tissue, eQTL
                    retina, genes in significant pathways, gene-based eQTL integration,
                    gene-based MAGMA, eQTL/sQTL of PAINTOR variants). n_evidence counts the
                    columns naming the gene; nearest gene alone does not qualify.
Symbols map to Ensembl through HGNC (approved, then previous, then alias; ambiguous dropped),
the same mapper as T4.1. Coverage is against data/processed/ortholog_table_van_zyl.csv.
No expression value and no Chimera readout is read.

Usage: python -I build_glaucoma_list.py <chimera_repo> <out.tsv>
"""

from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

VANZYL = {
    "vanzyl_mendelian": ["ANGPT1", "ANGPT2", "CPAMD8", "CYP1B1", "FOXC1", "LOXL1", "LTBP2", "MYOC", "OPTN",
                         "PITX2", "TEK", "TBK1"],
    "vanzyl_gwas": ["TMCO1", "TXNRD2", "CAV1", "CAV2", "ARHGEF12", "ATXN2"],
    "vanzyl_text": ["EFEMP1", "GAS7", "KALRN", "PRSS23", "SIX6", "POU6F2", "CACNA1A"],
}
# Sentences each list is taken from, checked against the saved article page at run time.
VANZYL_QUOTES = [
    "Mendelian genes assessed were ANGPT1 , ANGPT2 , CPAMD8 , CYP1B1 , FOXC1 , LOXL1 , LTBP2 , MYOC , OPTN , "
    "PITX2 , TEK ( TIE2 ), and TBK1 .",
    "Well-established GWAS loci genes included TMCO1 , TXNRD2 , CAV1 , CAV2 , ARHGEF12 , and ATXN2 .",
    "EFEMP1 exhibited robust cell type-specific expression in beam cells and JCT",
    "including GAS7 (CM, Schwann cells), KALRN (CM), PRSS23 (SC, CC, vessel endothelium, and Schwann cells)",
    "a subset, including OPTN , ATXN2 , TMCO1 , and SIX6 , were expressed at the highest levels in RGCs",
    "Several susceptibility genes, including CAV1 , CAV2 , and POU6F2",
    "Two genes associated with exfoliation glaucoma (XFG) and the related exfoliation syndrome (XFS), LOXL1 and CACNA1A",
]
SD14_EVIDENCE = {
    6: "protein_altering",
    7: "ld_protein_altering",
    8: "chromatin_interaction",
    9: "eqtl_any",
    10: "eqtl_retina",
    11: "significant_pathways",
    13: "genebased_eqtl_integration",
    14: "genebased_magma",
    16: "paintor_eqtl_sqtl",
}
TOKEN = re.compile(r"^[A-Z][A-Z0-9-]*[A-Z0-9](\.\d+)?$")
NOT_GENES = {"NO", "YES", "NA", "LD", "SNP", "MULTIPLE"}


def article_text(html: Path) -> str:
    t = re.sub(r"<[^>]+>", " ", html.read_text(encoding="utf-8", errors="replace"))
    return re.sub(r"\s+", " ", t.replace("&amp;", "&"))


def tokens(cell: str) -> list[str]:
    cell = re.sub(r"rs\d+", " ", cell)
    out = []
    for t in re.split(r"[,;/()\s]+", cell.replace("*", " ")):
        t = t.strip()
        if t and TOKEN.match(t) and t.upper() not in NOT_GENES and not t.lower().startswith(("missense", "nonsense")):
            out.append(t)
    return out


def main(repo: Path, out_tsv: Path) -> None:
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "analysis" / "tasks" / "T4_1_gold"))
    from gold_checks import SymbolMapper, gtf_gene_names  # noqa: E402
    sys.path.insert(0, str(repo / "analysis" / "tasks" / "T6_1_functional"))
    from xlsx_min import read_sheet  # noqa: E402

    G = repo / "data/analysis/glaucoma"
    text = article_text(G / "vanZyl2020_PMC7229661.html")
    for q in VANZYL_QUOTES:
        if q not in text:
            raise SystemExit(f"quote not found in the saved van Zyl page: {q[:80]}")
    for tier, genes in VANZYL.items():
        for g in genes:
            if not re.search(rf"\b{re.escape(g)}\b", text):
                raise SystemExit(f"{g} not in the van Zyl text")
    sd = read_sheet(G / "Gharahkhani2021_Supplementary_Data_1-25.xlsx", "Supplementary Data 14")
    hdr = sd[0]
    assert hdr[3].strip() == "Nearest Gene" and hdr[13].startswith("Gene-based") and hdr[14].startswith("Gene-based (MAGMA)"), hdr
    loci = [r for r in sd[1:] if len(r) > 4 and r[2].startswith("rs")]
    if len(loci) != 127:
        raise SystemExit(f"expected 127 loci in Supplementary Data 14, found {len(loci)}")
    ev, loc_of, status_of = defaultdict(set), defaultdict(set), defaultdict(set)
    nearest = defaultdict(set)
    for r in loci:
        r = r + [""] * (17 - len(r))
        for g in tokens(r[3]):
            nearest[g].add(r[2])
        for col, name in SD14_EVIDENCE.items():
            for g in tokens(r[col]):
                ev[g].add(name)
                loc_of[g].add(r[2])
                status_of[g].add(r[4])
    rows = []
    for tier, genes in VANZYL.items():
        for g in genes:
            rows.append({"gene": g, "source": tier, "evidence": "named in van Zyl 2020 Results", "loci": ""})
    for g in sorted(ev):
        rows.append({"gene": g, "source": "gharahkhani", "evidence": ",".join(sorted(ev[g])),
                     "n_evidence": len(ev[g]), "loci": ",".join(sorted(loc_of[g])),
                     "locus_status": ",".join(sorted(status_of[g]))})
    df = pd.DataFrame(rows)
    names, _ = gtf_gene_names(repo / "data/analysis/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz")
    mapper = SymbolMapper(repo / "data/analysis/gold/G10/hgnc_complete_set.txt", names)
    m = mapper.map(sorted(set(df.gene))).set_index("symbol")
    df["hsapiens_gene_id"] = df.gene.map(m.ensg)
    df["id_route"] = df.gene.map(m.route)
    vz = pd.read_csv(repo / "data/processed/ortholog_table_van_zyl.csv", dtype=str)
    in_id = set(vz.hsapiens_gene_id)
    in_sym = set(vz.hsapiens_gene_name.dropna())
    df["in_van_zyl_vocab"] = df.hsapiens_gene_id.isin(in_id)
    df["in_van_zyl_vocab_by_symbol"] = df.gene.isin(in_sym)
    df["nearest_gene_at_loci"] = df.gene.map(lambda g: ",".join(sorted(nearest.get(g, ()))))
    # one row per gene: sources joined; a gene from van Zyl and Gharahkhani keeps both
    agg = df.groupby("gene", sort=False).agg(
        source=("source", lambda s: ";".join(dict.fromkeys(s))),
        hsapiens_gene_id=("hsapiens_gene_id", "first"),
        id_route=("id_route", "first"),
        in_van_zyl_vocab=("in_van_zyl_vocab", "first"),
        in_van_zyl_vocab_by_symbol=("in_van_zyl_vocab_by_symbol", "first"),
        gharahkhani_evidence=("evidence", lambda s: ";".join(x for x in s if x != "named in van Zyl 2020 Results")),
        gharahkhani_n_evidence=("n_evidence", "max") if "n_evidence" in df else ("gene", "size"),
        gharahkhani_loci=("loci", lambda s: ";".join(x for x in s if x)),
        nearest_gene_at_loci=("nearest_gene_at_loci", "first"),
    ).reset_index()
    agg["gharahkhani_n_evidence"] = agg.gharahkhani_n_evidence.fillna(0).astype(int)
    agg["mapped"] = agg.hsapiens_gene_id.notna()
    agg.to_csv(out_tsv, sep="\t", index=False)
    van = agg.source.str.contains("vanzyl")
    gh = agg.source.str.contains("gharahkhani")
    print("genes", len(agg), "| van Zyl", int(van.sum()), "| Gharahkhani", int(gh.sum()), "| both", int((van & gh).sum()))
    print("mapped to ENSG", int(agg.mapped.sum()), "| in van Zyl vocabulary (13,636)", int(agg.in_van_zyl_vocab.sum()))
    print("van Zyl tiers in vocabulary:", {t: int(agg[agg.source.str.contains(t)].in_van_zyl_vocab.sum())
                                         for t in VANZYL}, {t: len(v) for t, v in VANZYL.items()})
    print("Gharahkhani in vocabulary:", int(agg[gh].in_van_zyl_vocab.sum()), "of", int(gh.sum()),
          "| with >=2 evidence columns:", int(agg[gh & (agg.gharahkhani_n_evidence >= 2)].in_van_zyl_vocab.sum()),
          "of", int((gh & (agg.gharahkhani_n_evidence >= 2)).sum()))
    print("van Zyl genes not in vocabulary:", agg[van & ~agg.in_van_zyl_vocab].gene.tolist())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
