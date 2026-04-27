"""Extended gene-category vocabulary for Δα biology analysis.

The default 7-category set in [`_paper_utils.py`](_paper_utils.py)
(``classify_gene_category``) is fine for a first sanity check but is too
coarse for a serious cross-dataset enrichment story. This module curates
~30 well-defined biological categories with broader gene coverage,
enough for Fisher enrichment to reach significance on most categories
in an 8K–15K HVG dataset.

**Design rules.**

* Every category is defined as ``(prefix_tuple, gene_set)``. Membership
  is `symbol.startswith(prefix)` OR `symbol in gene_set`.
* Prefix categories take priority over curated lists for the same
  gene (so e.g. ``RPL13A`` resolves to ``ribosomal_structural``, not
  ``housekeeping``, even though it's in both — exactly the convention
  in ``_paper_utils.classify_gene_category``).
* Each gene is assigned to **exactly one** category by the
  priority-ordered ``CATEGORIES`` list. Genes that match no category
  fall through to ``other``.
* ``CATEGORIES`` is a list, not a dict — order matters for priority.
  The first matching category wins.

**Adding categories.** Append a ``Category(...)`` to ``CATEGORIES``.
Keep the order consistent with biological "specificity" — narrow
prefix-based categories (e.g. mitochondrial-DNA-encoded) before
broader curated lists (e.g. cytokines).

**Source provenance.** Curated lists draw from canonical reviews
(Eisenberg & Levanon 2013 housekeeping, Tirosh 2016 cell cycle,
canonical type-I IFN), HGNC family pages (kinase / TF families,
collagens, MHC), and KEGG-style pathway membership for signaling
modules. No external GMT files are required.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class Category:
    """One gene category. Membership = startswith(prefix) OR in gene_set."""
    name: str
    prefixes: tuple[str, ...] = field(default_factory=tuple)
    gene_set: frozenset[str] = field(default_factory=frozenset)

    def matches(self, symbol_upper: str) -> bool:
        if self.prefixes and any(symbol_upper.startswith(p) for p in self.prefixes):
            return True
        return symbol_upper in self.gene_set


# ---------------------------------------------------------------------------
# Category definitions  (priority order — first match wins)
# ---------------------------------------------------------------------------

CATEGORIES: tuple[Category, ...] = (

    # ----- Translation machinery (priority over "housekeeping") -----
    Category(
        name="ribosomal_structural",
        prefixes=("RPL", "RPS"),
    ),
    Category(
        name="ribosomal_mt",
        prefixes=("MRPL", "MRPS"),
    ),
    Category(
        name="ribosome_biogenesis",
        gene_set=frozenset([
            "FBL", "DKC1", "NOP10", "NOP14", "NOP16", "NOP56", "NOP58",
            "NOL6", "NOL8", "NOL10", "NOL11", "NOL12",
            "RRP1", "RRP1B", "RRP7A", "RRP9", "RRP12", "RRP15", "RRP36",
            "EXOSC1", "EXOSC2", "EXOSC3", "EXOSC4", "EXOSC5", "EXOSC6",
            "EXOSC7", "EXOSC8", "EXOSC9", "EXOSC10",
            "BMS1", "WDR12", "WDR43", "WDR74", "WDR75",
            "UTP3", "UTP4", "UTP6", "UTP14A", "UTP15", "UTP18", "UTP20",
            "PWP2", "GNL2", "GNL3", "GNL3L",
            "MPHOSPH10", "MPHOSPH6", "DDX10", "DDX18", "DDX21", "DDX52",
        ]),
    ),
    Category(
        name="translation_initiation",
        prefixes=("EIF",),
    ),
    Category(
        name="translation_elongation",
        prefixes=("EEF",),
        gene_set=frozenset(["PABPC1", "PABPC3", "PABPC4", "PABPC5"]),
    ),

    # ----- Mitochondria & energy -----
    Category(
        name="mitochondrial_dna_encoded",
        prefixes=("MT-", "MTRNR", "MTATP", "MTCO", "MTCY", "MTND"),
    ),
    Category(
        name="oxphos_nuclear_encoded",
        prefixes=("NDUF", "UQCR", "ATP5", "ATP6V", "COX4", "COX5", "COX6", "COX7",
                  "COX8", "COX10", "COX11", "COX15", "COX17", "COX19", "COX20",
                  "SDH"),
        gene_set=frozenset(["CYCS", "CYC1", "TFAM", "MFN1", "MFN2", "OPA1", "DRP1",
                            "DNM1L", "FIS1", "PINK1", "PRKN"]),
    ),

    # ----- Chromatin & nuclear scaffold -----
    Category(
        name="histone",
        prefixes=("HIST", "H1-", "H1F", "H2A", "H2B", "H2AC", "H2BC",
                  "H3-", "H3F", "H4-", "H4C"),
    ),
    Category(
        name="splicing_factors",
        prefixes=("SRSF", "SRRM", "SF3", "U2AF", "PRPF", "SNRP", "HNRN", "HNRNP",
                  "SART", "DHX", "DDX5", "DDX17"),
        gene_set=frozenset(["SF1", "RBM5", "RBM10", "RBM17", "RBM25", "RBM39",
                            "POLR2A", "POLR2B"]),
    ),
    Category(
        name="dna_replication_repair",
        prefixes=("MCM", "RAD51", "RAD52", "RAD54", "XRCC", "MSH", "MLH",
                  "ERCC", "FANC", "POL"),
        gene_set=frozenset([
            "BRCA1", "BRCA2", "ATM", "ATR", "CHEK1", "CHEK2", "TP53BP1",
            "PARP1", "PARP2", "PARP3", "TOP2A", "TOP2B", "TOP1",
            "RPA1", "RPA2", "RPA3", "TIPIN", "TIMELESS",
            "CDC6", "CDC7", "CDC45", "ORC1", "ORC2", "ORC3", "ORC4", "ORC5", "ORC6",
        ]),
    ),

    # ----- Cell cycle -----
    Category(
        name="cell_cycle",
        prefixes=("CDK", "CCN", "AURK", "CENP", "BUB", "PLK", "ANLN", "KIF11",
                  "KIF20", "KIF22", "KIF23"),
        gene_set=frozenset([
            "MKI67", "PCNA", "TOP2A", "FOXM1",
            "E2F1", "E2F2", "E2F3", "E2F4", "E2F5", "E2F6", "E2F7", "E2F8",
            "BIRC5", "BIRC2", "BIRC3",
            "CDC20", "CDC25A", "CDC25B", "CDC25C",
            "RB1", "RBL1", "RBL2", "TP53",
            "MAD2L1", "MAD2L2", "BCCIP", "PTTG1",
        ]),
    ),

    # ----- Protein quality control -----
    Category(
        name="chaperones_hsp",
        prefixes=("HSPA", "HSPB", "HSPC", "HSPD", "HSPE", "HSPH", "DNAJ",
                  "HSP90", "HSP60", "HSP70"),
        gene_set=frozenset(["BAG1", "BAG2", "BAG3", "BAG5", "BAG6",
                            "STIP1", "AHSA1", "AHSA2", "CHORDC1", "CDC37",
                            "CRYAA", "CRYAB"]),
    ),
    Category(
        name="proteasome",
        prefixes=("PSMA", "PSMB", "PSMC", "PSMD", "PSME", "PSMG"),
        gene_set=frozenset(["POMP", "USP14", "UCHL5", "ECPAS", "ADRM1",
                            "PSMF1", "PI31"]),
    ),
    Category(
        name="autophagy",
        prefixes=("ATG", "GABARAP", "MAP1LC3"),
        gene_set=frozenset(["BECN1", "ULK1", "ULK2", "ULK3", "ULK4",
                            "RB1CC1", "WIPI1", "WIPI2", "STX17",
                            "TFEB", "TFE3", "MITF", "SQSTM1", "NBR1",
                            "CALCOCO2", "TAX1BP1", "OPTN"]),
    ),

    # ----- Cytoskeleton -----
    Category(
        name="cytoskeleton_actin",
        prefixes=("ACT", "ARPC", "ARP2", "ARP3", "WASF", "WASL", "FMN", "DIAPH",
                  "MYO", "TPM"),
        gene_set=frozenset(["GSN", "VCL", "TLN1", "TLN2", "FLNA", "FLNB", "FLNC",
                            "PFN1", "PFN2", "CFL1", "CFL2", "CDC42", "RHOA",
                            "RAC1", "RAC2", "RAC3"]),
    ),
    Category(
        name="cytoskeleton_microtubule",
        prefixes=("TUB", "MAP1", "MAP2", "MAP4", "MAP6", "MAP7", "MAPT", "STMN",
                  "DYNL", "DYNC", "KIF"),
        gene_set=frozenset(["DCTN1", "DCTN2", "DCTN3", "PCNT", "CEP170",
                            "TPX2", "NUMA1", "RANBP10"]),
    ),
    Category(
        name="cytoskeleton_intermediate",
        prefixes=("KRT", "NEF"),
        gene_set=frozenset(["VIM", "DES", "GFAP", "PRPH", "SYNE1", "SYNE2",
                            "LMNA", "LMNB1", "LMNB2",
                            "PLEC", "INA", "INF2"]),
    ),

    # ----- Extracellular matrix -----
    Category(
        name="ecm_collagen",
        prefixes=("COL",),
    ),
    Category(
        name="ecm_other",
        prefixes=("LAMA", "LAMB", "LAMC", "FBN", "FBLN", "MMP", "TIMP", "ADAMTS",
                  "ADAM",  "POSTN", "TGFBI", "MFAP", "DPT"),
        gene_set=frozenset(["FN1", "ELN", "VTN", "TNC", "TNXB", "DCN", "BGN",
                            "LUM", "OGN", "PRELP", "ASPN", "FMOD",
                            "VCAN", "ACAN", "HAS1", "HAS2", "HAS3",
                            "HSPG2", "PERLECAN", "NID1", "NID2"]),
    ),

    # ----- Immune & blood -----
    Category(
        name="mhc_class_I",
        prefixes=("HLA-A", "HLA-B", "HLA-C", "HLA-E", "HLA-F", "HLA-G"),
        gene_set=frozenset(["B2M", "TAP1", "TAP2", "TAPBP", "ERAP1", "ERAP2",
                            "CALR", "PDIA3"]),
    ),
    Category(
        name="mhc_class_II",
        prefixes=("HLA-D",),
        gene_set=frozenset(["CD74", "CIITA", "RFX5", "RFXANK", "RFXAP"]),
    ),
    Category(
        name="complement",
        prefixes=("C1Q", "C2", "C3", "C4A", "C4B", "C5", "C6", "C7", "C8", "C9",
                  "CFB", "CFI", "CFH", "CFD", "CFP",
                  "MBL", "MASP", "FCN"),
        gene_set=frozenset(["CD46", "CD55", "CD59", "CR1", "CR2", "C1S", "C1R",
                            "C1RL", "VSIG4", "ITGAX"]),
    ),
    Category(
        name="coagulation_serum",
        prefixes=("F2", "F3", "F5", "F7", "F8", "F9", "F10", "F11", "F12", "F13",
                  "FGA", "FGB", "FGG", "SERPINA", "SERPINC", "SERPIND",
                  "SERPINE", "SERPINF", "SERPING", "SERPINH",
                  "APO", "AHSG"),
        gene_set=frozenset(["VWF", "A2M", "ALB", "TF", "TFRC",
                            "PROC", "PROS1", "PLAT", "PLAU", "PLG",
                            "ANXA1", "ANXA2", "ANXA5", "TFPI", "TFPI2",
                            "ANGPTL3", "ANGPTL4", "ANGPTL8"]),
    ),
    Category(
        name="hemoglobin",
        prefixes=("HBA", "HBB", "HBG", "HBE", "HBM", "HBQ", "HBZ"),
    ),
    Category(
        name="cytokines_chemokines",
        prefixes=("IL", "TNF", "TNFRSF", "TNFSF", "CCL", "CXCL", "CXCR", "CCR",
                  "CSF"),
        gene_set=frozenset(["LTA", "LTB", "LIF", "OSM", "IFNG", "IFNGR1", "IFNGR2",
                            # FAS / FASLG live in apoptosis (below) — but we keep
                            # IL / TNF here as the primary cytokine prefixes.
                            ]),
    ),
    Category(
        name="interferon_response",
        prefixes=("IFI", "IFIT", "IFITM", "ISG", "OAS", "MX",
                  "IRF", "USP18"),
        gene_set=frozenset(["STAT1", "STAT2", "JAK1", "JAK2", "TYK2",
                            "RSAD2", "BST2", "DDX58", "IFIH1", "TRIM22",
                            "TRIM5", "ADAR", "EIF2AK2"]),
    ),
    Category(
        name="immediate_early",
        gene_set=frozenset([
            "FOS", "FOSB", "FOSL1", "FOSL2",
            "JUN", "JUNB", "JUND",
            "EGR1", "EGR2", "EGR3", "EGR4",
            "NR4A1", "NR4A2", "NR4A3",
            "ATF3", "ATF4", "DUSP1", "DUSP2", "DUSP4", "DUSP5", "DUSP6",
            "IER2", "IER3", "IER5",
            "BTG1", "BTG2", "GADD45A", "GADD45B", "GADD45G",
            "ARC", "PCDH8", "MYC", "MYCN",
        ]),
    ),

    # ----- Signaling families -----
    Category(
        name="kinases_signaling",
        prefixes=("MAPK", "MAP2K", "MAP3K", "MAP4K", "AKT", "PIK3", "MTOR",
                  "PTPN", "PTPR", "DUSP",
                  "ERBB", "EGFR", "FGFR", "PDGFR", "VEGFR", "JAK", "TYK",
                  "ROCK", "RPS6K", "CSNK", "PRKC", "PRKA", "PRKD"),
        gene_set=frozenset(["KRAS", "NRAS", "HRAS", "RAF1", "BRAF", "ARAF",
                            "MEK1", "MEK2", "RPS6KB1", "RPS6KB2",
                            "PTEN", "TSC1", "TSC2", "RHEB"]),
    ),
    Category(
        name="wnt_signaling",
        prefixes=("WNT", "FZD", "LRP5", "LRP6", "DVL", "AXIN", "TCF7", "LEF",
                  "DKK", "SFRP", "RSPO"),
        gene_set=frozenset(["CTNNB1", "GSK3A", "GSK3B", "APC", "PORCN",
                            "WLS", "GPC3", "GPC4"]),
    ),
    Category(
        name="notch_signaling",
        prefixes=("NOTCH", "JAG", "DLL", "HES", "HEY", "DTX"),
        gene_set=frozenset(["RBPJ", "MAML1", "MAML2", "MAML3",
                            "PSEN1", "PSEN2", "APH1A", "APH1B",
                            "NCSTN", "NUMBL"]),
    ),
    Category(
        name="tgfb_bmp_signaling",
        prefixes=("TGFB", "BMP", "ACVR", "SMAD", "INHB", "INHA", "BMPR"),
        gene_set=frozenset(["TGFBR1", "TGFBR2", "TGFBR3",
                            "SKI", "SKIL", "SNAI1", "SNAI2"]),
    ),
    Category(
        name="hedgehog_signaling",
        gene_set=frozenset([
            "SHH", "IHH", "DHH", "PTCH1", "PTCH2", "SMO", "GLI1", "GLI2", "GLI3",
            "SUFU", "DISP1", "DISP2", "HHIP", "BOC", "CDON", "GAS1",
        ]),
    ),

    # ----- Transcription factor families -----
    Category(
        name="tf_homeobox",
        prefixes=("HOX", "DLX", "NKX", "MSX", "PBX", "MEIS", "LHX", "ISL",
                  "PAX", "PITX", "OTX", "ZIC", "EN1", "EN2"),
    ),
    Category(
        name="tf_zinc_finger",
        prefixes=("ZNF", "ZFP", "ZFY", "ZFX", "GLI", "KLF", "SP1", "SP2", "SP3",
                  "SP4", "EGR", "ZBTB", "ZSCAN", "ZIC"),
    ),
    Category(
        name="tf_forkhead_etc",
        prefixes=("FOX", "SOX", "GATA", "NFAT", "NFKB", "STAT", "ETS",
                  "ELF", "ELK", "ERG", "ETV", "GABP", "MEF",
                  "TEAD", "RUNX", "TBX", "TCF",
                  "MYB", "MYBL", "MYC", "MYCN", "MAX", "MXI", "MAD",
                  "BHLHE", "ID1", "ID2", "ID3", "ID4", "TWIST", "SCL",
                  "HAND", "MEF2"),
    ),
    Category(
        name="nuclear_receptors",
        prefixes=("NR0", "NR1", "NR2", "NR3", "NR4", "NR5", "NR6",
                  "ESR", "RAR", "RXR", "THR", "PPAR", "PGR", "AR"),
        gene_set=frozenset(["VDR", "GR", "MR", "LXRA", "LXRB"]),
    ),

    # ----- Metabolism subsystems -----
    Category(
        name="cytochrome_p450",
        prefixes=("CYP",),
    ),
    Category(
        name="glutathione_redox",
        prefixes=("GST", "GPX", "GSR", "GSS", "PRDX", "TXN", "TXNRD",
                  "SOD", "CAT", "NQO"),
        gene_set=frozenset(["G6PD", "G6PDH", "TKTL1", "TKTL2", "PGD"]),
    ),
    Category(
        name="metabolism_glycolysis",
        prefixes=("HK", "PGK", "ALDOA", "ALDOB", "ALDOC", "ENO", "PKM", "GAPDH"),
        gene_set=frozenset(["GPI", "PFKL", "PFKM", "PFKP", "PFKFB1", "PFKFB2",
                            "PFKFB3", "PFKFB4", "TPI1", "PGAM1", "PGAM2",
                            "LDHA", "LDHB", "LDHC", "LDHD", "MPC1", "MPC2",
                            "SLC2A1", "SLC2A2", "SLC2A3", "SLC2A4"]),
    ),
    Category(
        name="metabolism_lipid",
        prefixes=("APO", "LPL", "LIPC", "LIPE", "LIPF", "LIPH", "LIPG", "LIPI",
                  "DGAT", "FASN", "ACAC", "ACSL", "ACOX",
                  "CPT", "HSD", "DHCR", "SCD", "ELOVL", "FADS"),
        gene_set=frozenset(["LDLR", "VLDLR", "SREBF1", "SREBF2", "SCAP", "INSIG1",
                            "INSIG2", "MVK", "MVD", "HMGCR", "HMGCS1",
                            "ABCA1", "ABCG1", "PCSK9"]),
    ),

    # ----- Apoptosis -----
    Category(
        name="apoptosis",
        prefixes=("BCL2", "CASP", "BIRC", "TNFRSF",
                  "DIABLO", "MCL"),
        gene_set=frozenset(["BAX", "BAK1", "BID", "BIM", "BAD", "PUMA", "NOXA",
                            "BBC3", "PMAIP1",
                            "FAS", "FASLG", "FADD", "TRADD",
                            "APAF1", "CYCS", "AIFM1",
                            "XIAP", "CFLAR", "CRADD",
                            "ENDOG", "CIDEA", "CIDEB", "CIDEC"]),
    ),

    # ----- Housekeeping (lowest priority — almost everything matches some other) -----
    Category(
        name="housekeeping",
        gene_set=frozenset([
            "ACTB", "GAPDH", "TUBB", "TUBA1B", "HPRT1", "PGK1", "PPIA",
            "B2M", "RPLP0", "RPL13A", "TBP", "UBC", "PSMB2", "PSMB4",
            "SDHA", "YWHAZ", "CHMP2A", "EMC7", "GPI", "REEP5", "RAB7A",
            "PUM1", "MRPL19", "POLR2F", "ALAS1", "RPS17",
        ]),
    ),
)


# Sentinel for "no category matched"
OTHER = "other"


# ---------------------------------------------------------------------------
# Classification API
# ---------------------------------------------------------------------------

def classify_extended(symbol: str) -> str:
    """Return the first matching category from ``CATEGORIES`` (priority order),
    or ``"other"`` if no category matches.

    Identical priority semantics to ``_paper_utils.classify_gene_category``,
    but with much broader category coverage.
    """
    if not isinstance(symbol, str) or not symbol:
        return OTHER
    s = symbol.upper()
    for cat in CATEGORIES:
        if cat.matches(s):
            return cat.name
    return OTHER


def annotate_extended(var_names: Iterable[str],
                      symbol_lookup) -> pd.DataFrame:
    """Build a [var_name, symbol, category] DataFrame using extended categories.

    `symbol_lookup` can be either a callable ``var_name -> symbol`` or a
    pandas Series/dict-like indexed by var_name. The Chimera workspace's
    ortholog table maps ``ortho_X`` -> human gene symbol; pass
    ``_paper_utils.annotate_genes_with_names(var_names)`` and we will
    handle the Series case.
    """
    var_names = pd.Index(list(var_names))
    if hasattr(symbol_lookup, "loc"):
        sym = symbol_lookup.reindex(var_names)
    elif callable(symbol_lookup):
        sym = pd.Series([symbol_lookup(v) for v in var_names], index=var_names)
    else:  # dict-like
        sym = pd.Series([symbol_lookup.get(v, v) for v in var_names], index=var_names)
    cats = sym.map(classify_extended)
    return pd.DataFrame(
        {"var_name": var_names, "symbol": sym.values, "category": cats.values},
        index=var_names,
    )


# Canonical display order: priority order from CATEGORIES + "other" at the end.
CATEGORY_ORDER_EXTENDED: tuple[str, ...] = tuple(c.name for c in CATEGORIES) + (OTHER,)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    samples = [
        ("RPL13A", "ribosomal_structural"),     # prefix beats curated housekeeping
        ("MRPS18A", "ribosomal_mt"),
        ("MT-CO1", "mitochondrial_dna_encoded"),
        ("NDUFA1", "oxphos_nuclear_encoded"),
        ("HIST1H1A", "histone"),
        ("H2AC18", "histone"),
        ("SRSF1", "splicing_factors"),
        ("BRCA1", "dna_replication_repair"),
        ("CCNB1", "cell_cycle"),
        ("HSPA1A", "chaperones_hsp"),
        ("PSMB5", "proteasome"),
        ("ATG5", "autophagy"),
        ("ACTB", "cytoskeleton_actin"),         # prefix ACT beats housekeeping
        ("TUBB", "cytoskeleton_microtubule"),
        ("KRT8", "cytoskeleton_intermediate"),
        ("COL1A1", "ecm_collagen"),
        ("FN1", "ecm_other"),
        ("HLA-A", "mhc_class_I"),
        ("HLA-DRA", "mhc_class_II"),
        ("C3", "complement"),
        ("FGA", "coagulation_serum"),
        ("HBA1", "hemoglobin"),
        ("IL6", "cytokines_chemokines"),
        ("ISG15", "interferon_response"),
        ("FOS", "immediate_early"),
        ("MAPK1", "kinases_signaling"),
        ("WNT3A", "wnt_signaling"),
        ("NOTCH1", "notch_signaling"),
        ("TGFB1", "tgfb_bmp_signaling"),
        ("SHH", "hedgehog_signaling"),
        ("HOXA1", "tf_homeobox"),
        ("ZNF1", "tf_zinc_finger"),
        ("FOXP3", "tf_forkhead_etc"),
        ("ESR1", "nuclear_receptors"),
        ("CYP3A4", "cytochrome_p450"),
        ("GSTP1", "glutathione_redox"),
        ("HK2", "metabolism_glycolysis"),
        ("FASN", "metabolism_lipid"),
        ("BAX", "apoptosis"),
        ("PUM1", "housekeeping"),               # pure housekeeping (no prefix match)
        ("RANDOMGENE", "other"),
    ]
    fail = 0
    for sym, expected in samples:
        got = classify_extended(sym)
        ok = "✓" if got == expected else "✗"
        if got != expected:
            fail += 1
        print(f"  {ok}  {sym:20s} -> {got:30s} (expected {expected})")
    print(f"\n{len(samples) - fail}/{len(samples)} pass; {len(CATEGORY_ORDER_EXTENDED)} categories total.")
