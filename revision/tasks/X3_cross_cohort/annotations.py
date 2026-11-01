"""X3 (EXPLORATORY): gene classes, fixed before any class is compared with a Chimera readout.

Sources (``data/revision/x3_annotations/``, sha256 in ``MANIFEST.tsv``, fetched 2026-11-01):

* Lambert et al. 2018 human TF census (``DatabaseExtract_v_1.01.csv``): ``Is TF? == Yes``;
  DNA-binding-domain families from the ``DBD`` column.
* HGNC complete set: Ensembl gene ID, UniProt accessions, EC numbers (``enzyme_id``).
* UniProtKB/Swiss-Prot human (reviewed): keywords, EC numbers, protein families; mapped to
  Ensembl genes through HGNC ``uniprot_ids``.

Protein classes (analysis 3), each defined from annotation only:

=================  ===============================================================
TF                 Lambert 2018 TF
TF_C2H2_ZF         Lambert DBD == "C2H2 ZF"
TF_homeodomain     Lambert DBD contains "Homeodomain"
TF_bHLH            Lambert DBD == "bHLH"
TF_bZIP            Lambert DBD == "bZIP"
TF_nuclear_rec     Lambert DBD == "Nuclear receptor"
TF_forkhead        Lambert DBD == "Forkhead"
TF_other           any other Lambert TF
protein_kinase     UniProt protein family contains "protein kinase superfamily"
kinase_any         UniProt keyword KW-0418 Kinase
receptor           UniProt keyword KW-0675 Receptor
GPCR               UniProt keyword KW-0297 G-protein coupled receptor
enzyme             EC number in UniProt or HGNC
EC1 ... EC7        first EC digit (oxidoreductase, transferase, hydrolase, lyase,
                   isomerase, ligase, translocase); a gene can carry several
ion_channel        UniProt keyword KW-0407 Ion channel
SLC_transporter    HGNC symbol ^SLC[0-9]
chromatin_reg      UniProt keyword KW-0156 Chromatin regulator, not a Lambert TF
mitochondrial      UniProt keyword KW-0496 Mitochondrion
=================  ===============================================================

Programs (analyses 1-2): UniProt keywords of the categories Biological process, Molecular
function, Cellular component and Domain, with at least ``MIN_PROGRAM`` genes in the
analysed universe. Keywords of the category Disease are never used.

Reserved sets of the frozen plan (05 section 4, H5/H6) are never formed or tested. The
UniProt keywords closest to them are excluded from every program list: ribosomal protein,
ribonucleoprotein, proteasome, spliceosome, mRNA splicing, mRNA processing, secreted,
extracellular matrix, collagen, immunity, innate immunity, adaptive immunity, inflammatory
response, antiviral defense, detoxification, bile acid catabolism. Housekeeping lists are not
read. The protein classes above do not overlap these sets by construction (they are broad
molecular-function classes); this is stated, not tested.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

from revision.common import gene_ids

ANN = gene_ids.REPO_ROOT / "data" / "revision" / "x3_annotations"
MIN_PROGRAM = 40
RESERVED_KW = {
    "KW-0689": "Ribosomal protein", "KW-0687": "Ribonucleoprotein", "KW-0647": "Proteasome",
    "KW-0747": "Spliceosome", "KW-0508": "mRNA splicing", "KW-0507": "mRNA processing",
    "KW-0964": "Secreted", "KW-0272": "Extracellular matrix", "KW-0176": "Collagen",
    "KW-0391": "Immunity", "KW-0399": "Innate immunity", "KW-1064": "Adaptive immunity",
    "KW-0395": "Inflammatory response", "KW-0051": "Antiviral defense",
    "KW-0216": "Detoxification", "KW-1281": "Bile acid catabolism",
}
PROGRAM_CATEGORIES = {"Biological process", "Molecular function", "Cellular component", "Domain"}
CLASS_ORDER = ["TF", "TF_C2H2_ZF", "TF_homeodomain", "TF_bHLH", "TF_bZIP", "TF_nuclear_rec",
               "TF_forkhead", "TF_other", "protein_kinase", "kinase_any", "receptor", "GPCR",
               "enzyme", "EC1", "EC2", "EC3", "EC4", "EC5", "EC6", "EC7", "ion_channel",
               "SLC_transporter", "chromatin_reg", "mitochondrial"]
PRIMARY_CLASSES = ["TF", "protein_kinase", "receptor", "enzyme", "ion_channel", "SLC_transporter",
                   "chromatin_reg"]
EC_NAMES = {"EC1": "oxidoreductase", "EC2": "transferase", "EC3": "hydrolase", "EC4": "lyase",
            "EC5": "isomerase", "EC6": "ligase", "EC7": "translocase"}


def _hgnc() -> pd.DataFrame:
    h = pd.read_csv(ANN / "hgnc_complete_set.txt", sep="\t", dtype=str, low_memory=False)
    h = h[h["ensembl_gene_id"].notna()]
    return h[["symbol", "ensembl_gene_id", "uniprot_ids", "enzyme_id", "gene_group"]]


def _uniprot() -> pd.DataFrame:
    u = pd.read_csv(ANN / "uniprot_human_reviewed.tsv.gz", sep="\t", dtype=str)
    u.columns = ["accession", "gene_primary", "gene_names", "ensembl_tx", "keywords", "keyword_ids",
                 "ec", "families"]
    return u


def uniprot_by_gene() -> pd.DataFrame:
    """One row per Ensembl gene: union of keyword IDs, EC numbers and families of its accessions."""
    h = _hgnc()
    u = _uniprot().set_index("accession")
    rows = []
    for ens, acc in zip(h["ensembl_gene_id"], h["uniprot_ids"].fillna("")):
        kws, ecs, fams = set(), set(), set()
        for a in [x for x in acc.split("|") if x]:
            if a not in u.index:
                continue
            r = u.loc[a]
            if isinstance(r.keyword_ids, str):
                kws |= {k.strip() for k in r.keyword_ids.split(";") if k.strip()}
            if isinstance(r.ec, str):
                ecs |= {e.strip() for e in r.ec.split(";") if e.strip()}
            if isinstance(r.families, str):
                fams |= {f.strip() for f in r.families.split(";") if f.strip()}
        rows.append((ens, kws, ecs, fams))
    out = pd.DataFrame(rows, columns=["hsapiens_gene_id", "kw", "ec", "families"])
    out = out.groupby("hsapiens_gene_id").agg(
        kw=("kw", lambda s: set().union(*s)), ec=("ec", lambda s: set().union(*s)),
        families=("families", lambda s: set().union(*s)))
    hg = h.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")
    out["hgnc_symbol"] = hg["symbol"].reindex(out.index)
    out["hgnc_ec"] = hg["enzyme_id"].reindex(out.index)
    return out


def lambert() -> pd.DataFrame:
    t = pd.read_csv(ANN / "lambert2018_DatabaseExtract_v_1.01.csv", dtype=str)
    t = t[t["Is TF?"] == "Yes"]
    return t[["Ensembl ID", "HGNC symbol", "DBD"]].rename(
        columns={"Ensembl ID": "hsapiens_gene_id", "HGNC symbol": "tf_symbol", "DBD": "tf_dbd"}
    ).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")


def keyword_table() -> pd.DataFrame:
    k = pd.read_csv(ANN / "uniprot_keywords.tsv", sep="\t", dtype=str)
    k.columns = ["kw_id", "kw_name", "kw_category"]
    return k.set_index("kw_id")


def primary_family(fams: set) -> str | float:
    """First listed UniProt family, reduced to its top level (text before the first comma)."""
    if not fams:
        return np.nan
    f = sorted(fams)[0]
    return f.split(",")[0].strip()


def gene_classes(genes: pd.Index) -> pd.DataFrame:
    """Boolean class membership for the given human Ensembl IDs (plus family and TF DBD)."""
    genes = pd.Index(genes, name="hsapiens_gene_id")
    up = uniprot_by_gene().reindex(genes)
    tf = lambert().reindex(genes)
    kw = up["kw"].apply(lambda x: x if isinstance(x, set) else set())
    ec = up["ec"].apply(lambda x: x if isinstance(x, set) else set())
    hec = up["hgnc_ec"].fillna("").astype(str)
    ec_all = [e | {x.strip() for x in h.split("|") if x.strip()} for e, h in zip(ec, hec)]
    fam = up["families"].apply(lambda x: x if isinstance(x, set) else set())
    sym = up["hgnc_symbol"].fillna("").astype(str)
    out = pd.DataFrame(index=genes)
    is_tf = tf["tf_dbd"].notna().to_numpy()
    dbd = tf["tf_dbd"].fillna("").astype(str)
    out["TF"] = is_tf
    out["TF_C2H2_ZF"] = is_tf & (dbd == "C2H2 ZF").to_numpy()
    out["TF_homeodomain"] = is_tf & dbd.str.contains("Homeodomain").to_numpy()
    out["TF_bHLH"] = is_tf & (dbd == "bHLH").to_numpy()
    out["TF_bZIP"] = is_tf & (dbd == "bZIP").to_numpy()
    out["TF_nuclear_rec"] = is_tf & (dbd == "Nuclear receptor").to_numpy()
    out["TF_forkhead"] = is_tf & (dbd == "Forkhead").to_numpy()
    sub = out[["TF_C2H2_ZF", "TF_homeodomain", "TF_bHLH", "TF_bZIP", "TF_nuclear_rec", "TF_forkhead"]].any(axis=1)
    out["TF_other"] = is_tf & ~sub.to_numpy()
    out["protein_kinase"] = [any("protein kinase superfamily" in f.lower() for f in s) for s in fam]
    out["kinase_any"] = ["KW-0418" in s for s in kw]
    out["receptor"] = ["KW-0675" in s for s in kw]
    out["GPCR"] = ["KW-0297" in s for s in kw]
    out["enzyme"] = [len(e) > 0 for e in ec_all]
    for d in range(1, 8):
        out[f"EC{d}"] = [any(x.startswith(f"{d}.") for x in e) for e in ec_all]
    out["ion_channel"] = ["KW-0407" in s for s in kw]
    out["SLC_transporter"] = [bool(re.match(r"^SLC[0-9]", s)) for s in sym]
    out["chromatin_reg"] = np.array(["KW-0156" in s for s in kw]) & ~is_tf
    out["mitochondrial"] = ["KW-0496" in s for s in kw]
    out["family"] = [primary_family(s) for s in fam]
    out["tf_dbd"] = tf["tf_dbd"]
    out["annotated_uniprot"] = up["kw"].notna().to_numpy()
    return out[CLASS_ORDER + ["family", "tf_dbd", "annotated_uniprot"]]


def program_sets(genes: pd.Index, min_size: int = MIN_PROGRAM) -> dict[str, set]:
    """UniProt-keyword programs over ``genes`` (reserved and Disease keywords removed)."""
    genes = pd.Index(genes)
    up = uniprot_by_gene().reindex(genes)
    kt = keyword_table()
    keep = {k for k, r in kt.iterrows() if r.kw_category in PROGRAM_CATEGORIES and k not in RESERVED_KW}
    sets: dict[str, set] = {}
    for g, kws in zip(genes, up["kw"]):
        if not isinstance(kws, set):
            continue
        for k in kws & keep:
            sets.setdefault(k, set()).add(g)
    return {f"{k}:{kt.loc[k, 'kw_name']}": v for k, v in sets.items() if len(v) >= min_size}
