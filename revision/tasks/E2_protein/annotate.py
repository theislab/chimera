"""E2 annotation table: one row per human gene, every E2 property. Run with `python -I` (reads downloaded files only).

Usage (from chimera/):  mamba run -n chimera_env python -I revision/tasks/E2_protein/annotate.py
Writes results/revision/E2/props.parquet and props_coverage.csv. Y5's annot.parquet supplies the properties it defines.
"""
from __future__ import annotations

import gzip
import io
import subprocess
import sys
import zipfile
from pathlib import Path

import re

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # rules.py only (no package import under -I)
import rules as RU  # noqa: E402

DL = REPO / "data" / "revision" / "E2"
OUT = REPO / "results" / "revision" / "E2"
OUT.mkdir(parents=True, exist_ok=True)
UNIPROT_AC = re.compile(r"[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9]([A-Z][A-Z0-9]{2}[0-9]){1,2}")


def hgnc() -> pd.DataFrame:
    h = pd.read_csv(REPO / "data/revision/x3_annotations/hgnc_complete_set.txt", sep="\t", dtype=str,
                    usecols=["symbol", "ensembl_gene_id", "entrez_id", "uniprot_ids", "prev_symbol", "alias_symbol"])
    return h.dropna(subset=["ensembl_gene_id"])


def gene_age(genes: pd.Index) -> pd.Series:
    sp = {s: v for v, _, ss in RU.STRATA for s in ss}
    cond = " || ".join(f'$8=="{s}"' for s in sp)
    tmp = OUT / "compara_strata.tsv"
    cmd = (f"zcat {REPO}/data/revision/ensembl110/Compara.110.protein_default.homologies.tsv.gz | "
           f"awk -F'\\t' '$3==\"homo_sapiens\" && $5 ~ /^ortholog/ && ({cond}) {{print $1\"\\t\"$8}}' | sort -u > {tmp}")
    subprocess.run(cmd, shell=True, check=True)
    co = pd.read_csv(tmp, sep="\t", names=["g", "sp"])
    co["v"] = co.sp.map(sp)
    age = co.groupby("g").v.max()
    return pd.Series(genes.map(age).fillna(RU.EUTHERIA).astype(float), index=genes)


def main():
    y5 = pd.read_parquet(REPO / "results/revision/Y5/annot.parquet")
    t = y5[["hsapiens_gene_id", "uniprot", "log_length", "disorder_frac", "coiled_frac", "ptm_dens", "paralog_count",
            "ppi_degree", "signal_peptide", "transmembrane"]].copy()
    hg = hgnc()
    allg = pd.Index(sorted(set(t.hsapiens_gene_id) | set(hg.ensembl_gene_id)))
    t = t.set_index("hsapiens_gene_id").reindex(allg)
    t.index.name = "hsapiens_gene_id"
    # UniProt Pfam (gene -> accession via Y5's choice, longest reviewed isoform entry)
    u = pd.read_csv(DL / "uniprot/uniprot_human_domains.tsv.gz", sep="\t", dtype=str)
    pf = u.set_index("Entry")["Pfam"].fillna("").str.strip(";")
    fam = t["uniprot"].map(pf)
    t["n_pfam"] = fam.map(lambda s: len(set(s.split(";"))) if isinstance(s, str) and s else (0 if isinstance(s, str) else np.nan))
    t["pfam_list"] = fam
    # Complex Portal (direct members, UniProt accessions, isoform suffix stripped)
    cx = pd.read_csv(DL / "complexportal/complexportal_9606.tsv", sep="\t", dtype=str)
    rows = []
    for ac, nm, ids in zip(cx["#Complex ac"], cx["Recommended name"], cx["Identifiers (and stoichiometry) of molecules in complex"]):
        for part in str(ids).split("|"):
            acc = part.split("(")[0].split("-")[0].strip()
            if UNIPROT_AC.fullmatch(acc):
                rows.append((ac, nm, acc))
    cm = pd.DataFrame(rows, columns=["complex", "name", "acc"]).drop_duplicates()
    acc2g = t["uniprot"].dropna().reset_index().set_index("uniprot")["hsapiens_gene_id"]
    hx = hg.dropna(subset=["uniprot_ids"]).assign(acc=lambda d: d.uniprot_ids.str.split("|")).explode("acc")
    acc2g = pd.concat([acc2g, hx.set_index("acc")["ensembl_gene_id"]])
    acc2g = acc2g[~acc2g.index.duplicated()]
    cm["g"] = cm.acc.map(acc2g)
    cm = cm.dropna(subset=["g"])
    cm[["complex", "name", "g"]].drop_duplicates().to_parquet(OUT / "complex_members.parquet")
    nc = cm.groupby("g").complex.nunique()
    t["n_complexes"] = np.log1p(t.index.map(nc).fillna(0).astype(float))
    t["complex_member"] = (t.index.map(nc).fillna(0) > 0).astype(float)
    # HuRI
    hu = pd.read_csv(REPO / "data/revision/functional/HuRI.tsv", sep="\t", header=None, names=["a", "b"])
    hu = hu[hu.a != hu.b]
    deg = pd.concat([hu.a, hu.b]).value_counts()
    t["huri_degree"] = np.log1p(t.index.map(deg).fillna(0).astype(float))
    # gene age
    t["gene_age"] = gene_age(t.index).values
    # OHNOLOGS
    for nm, fn in (("ohnolog", "hsapiens_2R_pairs_intermediate.tsv"), ("ohnolog_strict", "hsapiens_2R_pairs_strict.tsv")):
        o = pd.read_csv(DL / "ohnologs" / fn, sep="\t")
        s = set(o.Ohno1) | set(o.Ohno2)
        t[nm] = t.index.isin(s).astype(float)
    # Collins dosage (symbols -> Ensembl via HGNC symbol, then previous symbols)
    dz = pd.read_csv(DL / "dosage/Collins_rCNV_2022.dosage_sensitivity_scores.tsv.gz", sep="\t")
    sym2g = hg.set_index("symbol")["ensembl_gene_id"]
    prev = hg.dropna(subset=["prev_symbol"]).assign(p=lambda d: d.prev_symbol.str.split("|")).explode("p").set_index("p")["ensembl_gene_id"]
    prev = prev[~prev.index.duplicated(keep=False)]
    dz["g"] = dz["#gene"].map(sym2g).fillna(dz["#gene"].map(prev))
    dz = dz.dropna(subset=["g"]).drop_duplicates("g").set_index("g")
    t["phaplo"] = t.index.map(dz.pHaplo)
    t["ptriplo"] = t.index.map(dz.pTriplo)
    t["haplo_sensitive"] = (t.phaplo >= 0.86).astype(float).where(t.phaplo.notna())
    t["triplo_sensitive"] = (t.ptriplo >= 0.94).astype(float).where(t.ptriplo.notna())
    # ClinGen HI (Y2 download)
    cg = pd.read_csv(REPO / "data/revision/Y2/clingen/ClinGen_gene_curation_list_GRCh38.tsv", sep="\t", skiprows=5, dtype=str)
    cg.columns = [c.lstrip("#") for c in cg.columns]
    cg["g"] = cg["Gene Symbol"].map(sym2g)
    hi3 = set(cg.loc[cg["Haploinsufficiency Score"] == "3", "g"].dropna())
    t["clingen_hi3"] = t.index.isin(hi3).astype(float)
    # DepMap 24Q4 (Y2 download)
    ent2g = hg.dropna(subset=["entrez_id"]).set_index("entrez_id")["ensembl_gene_id"]
    ge = pd.read_csv(REPO / "data/revision/Y2/depmap/CRISPRGeneEffect_24Q4.csv", index_col=0)
    mean_eff = ge.mean(0)
    ids = mean_eff.index.str.extract(r"\((\d+)\)")[0].values
    me = pd.Series(mean_eff.values, index=pd.Index(ids).map(ent2g))
    me = me[me.index.notna()]
    me = me[~me.index.duplicated()]
    t["depmap_essentiality"] = -t.index.map(me)
    ce = pd.read_csv(REPO / "data/revision/Y2/depmap/CRISPRInferredCommonEssentials_24Q4.csv")
    ceid = ce.iloc[:, 0].str.extract(r"\((\d+)\)")[0].map(ent2g)
    t["common_essential"] = pd.Series(t.index.isin(set(ceid.dropna())).astype(float), index=t.index).where(t.depmap_essentiality.notna())
    # gnomAD v4.1 LOEUF (gold/G7, MANE select on Ensembl IDs)
    gn = pd.read_csv(REPO / "data/revision/gold/G7/gnomad.v4.1.constraint_metrics.tsv", sep="\t", dtype=str,
                     usecols=["gene_id", "mane_select", "lof.oe_ci.upper"])
    gn = gn[gn.gene_id.str.startswith("ENSG") & (gn.mane_select == "true")].drop_duplicates("gene_id").set_index("gene_id")
    t["loeuf"] = t.index.map(pd.to_numeric(gn["lof.oe_ci.upper"], errors="coerce"))
    # GTEx tau
    with gzip.open(DL / "gtex/GTEx_v8_gene_median_tpm.gct.gz", "rt") as fh:
        gt = pd.read_csv(fh, sep="\t", skiprows=2)
    gt["g"] = gt.Name.str.split(".").str[0]
    gt = gt[~gt.g.str.endswith("_PAR_Y")] if gt.g.dtype == object else gt
    X = np.log2(gt.drop(columns=["Name", "Description", "g"]).to_numpy(float) + 1)
    mx = X.max(1)
    tau = np.where(mx > 0, (1 - X / np.where(mx > 0, mx, 1)[:, None]).sum(1) / (X.shape[1] - 1), np.nan)
    tau = pd.Series(tau, index=gt.g)
    tau = tau[~tau.index.duplicated()]
    t["gtex_tau"] = t.index.map(tau)
    # Lambert TFs
    lb = pd.read_csv(REPO / "data/revision/x3_annotations/lambert2018_DatabaseExtract_v_1.01.csv")
    lb = lb[lb["Is TF?"] == "Yes"].set_index("Ensembl ID")
    t["tf"] = t.index.isin(lb.index).astype(float)
    t["tf_dbd"] = t.index.map(lb["DBD"])
    t["tf_c2h2_krab"] = t.index.isin(lb.index[lb["Is C2H2 ZF(KRAB)?"].astype(str) == "True"]).astype(float)
    hd = lb.index[lb.DBD.astype(str).str.contains("Homeodomain|CUT|Paired box|POU", regex=True)]
    t["tf_homeodomain"] = t.index.isin(hd).astype(float)
    t["tf_nuclear_receptor"] = t.index.isin(lb.index[lb.DBD == "Nuclear receptor"]).astype(float)
    # HPA 25.1
    z = zipfile.ZipFile(DL / "hpa/proteinatlas.tsv.zip")
    hp = pd.read_csv(z.open("proteinatlas.tsv"), sep="\t", dtype=str,
                     usecols=["Ensembl", "Protein class", "Secretome location", "Subcellular main location"])
    hp = hp.drop_duplicates("Ensembl").set_index("Ensembl").reindex(t.index)
    sec = hp["Secretome location"]
    has_sec = hp["Protein class"].notna()
    t["hpa_secreted"] = sec.fillna("").str.contains("Secreted").astype(float).where(has_sec)
    t["hpa_secreted_blood"] = sec.fillna("").str.contains("Secreted to blood").astype(float).where(has_sec)
    loc = hp["Subcellular main location"]
    known = loc.notna()
    pats = {"hpa_loc_nucleus": "Nucleoplasm|Nucleoli|Nuclear speckles|Nuclear bodies|Nuclear membrane|^Nucleus|, Nucleus",
            "hpa_loc_cytosol": "Cytosol", "hpa_loc_mitochondria": "Mitochondria", "hpa_loc_plasma_membrane": "Plasma membrane",
            "hpa_loc_vesicles": "Vesicles", "hpa_loc_er_golgi": "Endoplasmic reticulum|Golgi apparatus",
            "hpa_loc_cytoskeleton": "Actin filaments|Microtubules|Intermediate filaments|Centrosome|Centriolar satellite",
            "hpa_loc_junctions": "Cell Junctions|Focal adhesion sites"}
    for k, p in pats.items():
        t[k] = loc.fillna("").str.contains(p, regex=True).astype(float).where(known)
    pc = hp["Protein class"].fillna("")
    t["cd_marker"] = pc.str.contains("CD markers").astype(float).where(has_sec)
    t["enzyme"] = pc.str.contains("Enzymes").astype(float).where(has_sec)
    t["transporter"] = pc.str.contains("Transporters").astype(float).where(has_sec)
    # Immune keyword (UniProt, X3 download)
    ku = pd.read_csv(REPO / "data/revision/x3_annotations/uniprot_human_reviewed.tsv.gz", sep="\t", dtype=str,
                     usecols=["Entry", "Keyword ID"])
    imm = set(ku.loc[ku["Keyword ID"].fillna("").str.contains("KW-0391|KW-0399|KW-1064", regex=True), "Entry"])
    t["immune"] = t["uniprot"].isin(imm).astype(float).where(t.uniprot.notna())
    t = t.reset_index()
    t.to_parquet(OUT / "props.parquet")
    props = list(RU.CONTINUOUS) + list(RU.BINARY)
    cov = pd.DataFrame({"property": props, "n_annotated": [int(t[p].notna().sum()) for p in props],
                        "mean": [float(t[p].mean()) for p in props]})
    cov.to_csv(OUT / "props_coverage.csv", index=False)
    print(cov.to_string())
    print("gene_age counts", t.gene_age.value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
