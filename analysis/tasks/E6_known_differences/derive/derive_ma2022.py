"""E6 literature list: Ma et al. 2022 Science 377:eabo7257 (dlPFC; human, chimpanzee, rhesus macaque, marmoset).

CIRCULAR: this is the same snRNA-seq data as our DLPFC cohort. Every row has circular = yes.

Usage:
    python -I derive_ma2022.py <converted_dir> <out_tsv>
    converted_dir = chimera/data/analysis/E6/ma2022/converted (TSVs made by xlsx_to_tsv.py)

No genome-wide species-DEG table is published (PMC tables S1-S10 hold no such table; the Zenodo record
10.5281/zenodo.6824097 holds scripts only; the supplementary PDF with methods could not be fetched).
Two parts:
  A. Table S9 "Subclass- and species-specific expression of brain disorder risk genes": a 0/1 matrix
     (species x Enriched/Depleted x gene) over 29 subclasses. Every cell == 1 becomes one row.
     Covers only brain-disorder risk genes (586 species-divergent risk genes were screened; text: 157 genes
     "exhibited subclass- and species-specific expression patterns (table S9)").
       Human rows -> set human_specific; Chimpanzee/Macaque/Marmoset rows -> <species>_specific.
       Enriched -> up_in_<species>; Depleted -> down_in_<species>.
  B. Main-text statements (fixed list below, each with the sentence quoted in the derivation card):
     every sentence that names a gene, a cell type and a species-restricted expression difference
     (human, Hominini, macaque, or primate-vs-mouse). Subtype markers of species-specific subtypes and
     protein-only (IHC) observations are excluded; see card.
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]
SP = {"Human": "human", "Chimpanzee": "chimpanzee", "Macaque": "macaque", "Marmoset": "marmoset"}
ALL = ["human", "chimpanzee", "macaque", "marmoset"]

# (set, gene, cell_type, direction, contrast, text locator)  -- quotes are in lists/ma2022.md
TEXT = [
    ("human_specific_text", "FOXP2", "Micro", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "Abstract; Fig.6; 'FOXP2, which is human-specific in microglia'"),
    ("human_specific_text", "ANXA1", "upper-layer IT subclasses", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "Fig.3C text; 'humans selectively expressed ANXA1'"),
    ("human_specific_text", "PKD2L1", "L3-5 IT", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "Fig.3 text; 'human L3-5 IT subclass enrichment of PKD2L1'"),
    ("human_specific_text", "MELTF", "PVALB ChC", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "fig.S14H text; 'specifically upregulated in PVALB chandelier subtypes in humans'"),
    ("human_specific_text", "NMU", "inhibitory neurons (subclass not stated)", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "fig.S14I text; neuropeptides 'e.g. NMU and PENK' overrepresented in human-enriched genes"),
    ("human_specific_text", "PENK", "inhibitory neurons (subclass not stated)", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "fig.S14I text; neuropeptides 'e.g. NMU and PENK' overrepresented in human-enriched genes"),
    ("human_specific_text", "SLC47A1", "SST HGF", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "Fig.4 text; 'uniquely expressed in both TH-expressing inhibitory neuron subtypes in humans'"),
    ("human_specific_text", "VEGFB", "PC", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "Fig.5F text; 'VEGFB and PGF were enriched in human pericytes'"),
    ("human_specific_text", "PGF", "PC", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "Fig.5F text; 'VEGFB and PGF were enriched in human pericytes'"),
    ("human_specific_text", "DSCAM", "Micro", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "fig.S22C-D text; 'exhibited human-specific expression in microglia'"),
    ("human_specific_text", "MERTK", "L3-5 IT", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "table S9 text; 'enriched in human L3-5 IT excitatory neurons'"),
    ("human_specific_text", "CACNA1D", "Micro", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "table S9 text; 'enriched in human microglia'"),
    ("human_specific_text", "NR4A2", "SMC", "up_in_human", "human_vs_chimpanzee_macaque_marmoset", "table S9 text; 'human-specific expression in smooth muscle cells'"),
    ("hominini_specific_text", "CYP26B1", "L5-6 NP", "up_in_human", "human_chimpanzee_vs_macaque_marmoset", "Fig.3D text; 'enriched in Hominini L5-6 NP and L6 CT subclasses' (up in human and chimpanzee)"),
    ("hominini_specific_text", "CYP26B1", "L6 CT", "up_in_human", "human_chimpanzee_vs_macaque_marmoset", "Fig.3D text; 'enriched in Hominini L5-6 NP and L6 CT subclasses' (up in human and chimpanzee)"),
    ("hominini_specific_text", "CBLN2", "multiple excitatory neuron subclasses", "up_in_human", "human_chimpanzee_vs_macaque_marmoset", "Fig.3D text; 'globally upregulated in multiple Hominini excitatory neuron subclasses' (up in human and chimpanzee)"),
    ("hominini_specific_text", "GBA", "subset of ExN and InN subclasses", "up_in_human", "human_chimpanzee_vs_macaque_marmoset", "fig.S23D text; 'Hominini-enriched expression' (up in human and chimpanzee)"),
    ("hominini_specific_text", "GOSR2", "subset of ExN and InN subclasses", "up_in_human", "human_chimpanzee_vs_macaque_marmoset", "fig.S23D text; 'Hominini-enriched expression' (up in human and chimpanzee)"),
    ("macaque_specific_text", "ANXA3", "deep-layer IT subclasses", "up_in_macaque", "macaque_vs_human_chimpanzee_marmoset", "Fig.3C text; 'macaques specifically expressed ANXA3'"),
    ("primate_vs_mouse_text", "FOXP2", "L3-5 IT", "up_in_primates", "primates_vs_mouse", "Fig.6A text; 'enrichment of FOXP2 expression in L3-5 IT excitatory neurons across primates, that was not detected in mice'"),
    ("primate_vs_mouse_text", "PHLDB2", "L3-5 IT", "up_in_primates", "primates_vs_mouse", "fig.S22C-D text; 'displayed primate-specific expression in these neurons compared to mice'"),
    ("primate_vs_mouse_text", "PLCH1", "L3-5 IT", "up_in_primates", "primates_vs_mouse", "fig.S22C-D text; 'displayed primate-specific expression in these neurons compared to mice'"),
    ("primate_vs_mouse_text", "IL1RAPL2", "L3-5 IT", "up_in_primates", "primates_vs_mouse", "fig.S22C-D text; 'displayed primate-specific expression in these neurons compared to mice'"),
]


def main(conv_dir, out_tsv):
    d = Path(conv_dir)
    s9 = pd.read_csv(d / "NIHMS1837719-supplement-table_S9__Sheet1.tsv", sep="\t", skiprows=1, dtype=str)
    sub_cols = [c for c in s9.columns if c not in ("species", "type", "gene")]
    long = s9.melt(id_vars=["species", "type", "gene"], value_vars=sub_cols, var_name="subclass", value_name="v")
    assert set(long["v"].astype(str)) <= {"0", "1"}, set(long["v"])
    long = long[long["v"] == "1"]
    sp = long["species"].map(SP)
    assert sp.notna().all()
    others = sp.map(lambda s: "_".join(x for x in ALL if x != s))
    a = pd.DataFrame({
        "source_id": "ma2022",
        "set": sp + "_specific",
        "gene_symbol": long["gene"],
        "symbol_species": "human",
        "contrast": sp + "_vs_" + others,
        "cell_type": long["subclass"],
        "direction": long["type"].map({"Enriched": "up_in_", "Depleted": "down_in_"}) + sp,
        "effect": "",
        "evidence": "table S9 subclass- and species-specific expression (brain-disorder risk genes only); thresholds in suppl. methods",
        "circular": "yes",
        "note": "S9_type=" + long["type"] + ";gene_universe=586 species-divergent brain disorder risk genes",
    })
    b = pd.DataFrame([{
        "source_id": "ma2022", "set": s, "gene_symbol": g, "symbol_species": "human", "contrast": c,
        "cell_type": ct, "direction": dr, "effect": "", "evidence": "main-text statement (no table)",
        "circular": "yes", "note": "text: " + loc,
    } for s, g, ct, dr, c, loc in TEXT])
    out = pd.concat([a, b], ignore_index=True)[COLS]
    out.to_csv(out_tsv, sep="\t", index=False)
    print(out.groupby("set").agg(rows=("gene_symbol", "size"), genes=("gene_symbol", "nunique"),
                                 cell_types=("cell_type", "nunique")).to_string())
    print(out.groupby(["set", "direction"]).size().to_string())
    hs = out[out.set == "human_specific"]
    print(hs.groupby("cell_type").size().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
