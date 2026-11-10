"""EXPLORATORY: gene-level phenotype, viability, essentiality and study-depth annotation.

Reads only downloaded files (run with ``python -I``); no Chimera readout enters. Key: human Ensembl gene ID.
Usage (from chimera/): mamba run --no-capture-output -n chimera_env python -I analysis/tasks/Y2_functional/annot_pheno.py
"""

from __future__ import annotations

import gzip
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from analysis.tasks.Y2_functional.rules import HP_SYSTEMS, LETHAL_MP, MP_SYSTEMS  # noqa: E402

DL = REPO / "data/analysis/Y2"
OUT = REPO / "results/analysis/Y2/annot"
TABLES = {
    "liver": REPO / "data/processed/ortholog_table.csv",
    "testis": REPO / "data/processed/ortholog_table_testis.csv",
    "bone_marrow": REPO / "data/analysis/bone_marrow/ortholog_table_bone_marrow.csv",
    "kidney": REPO / "data/analysis/disease_kidney/ortholog_table_kidney.csv",
}


def obo_ancestors(path: Path) -> dict[str, set[str]]:
    """term -> itself plus all is_a ancestors; alt_ids map to the primary term."""
    parents, alt = defaultdict(set), {}
    cur, obsolete = None, False
    for line in path.read_text().splitlines():
        if line == "[Term]":
            cur, obsolete = None, False
        elif line.startswith("id: "):
            cur = line[4:].strip()
        elif line.startswith("alt_id: ") and cur:
            alt[line[8:].strip()] = cur
        elif line.startswith("is_a: ") and cur:
            parents[cur].add(line[6:].split("!")[0].strip())
    memo: dict[str, set[str]] = {}

    def anc(t: str) -> set[str]:
        if t in memo:
            return memo[t]
        out = {t}
        for p in parents.get(t, ()):
            out |= anc(p)
        memo[t] = out
        return out

    sys.setrecursionlimit(100000)
    terms = set(parents) | {p for ps in parents.values() for p in ps}
    res = {t: anc(t) for t in terms}
    for a, t in alt.items():
        if t in res:
            res[a] = res[t]
    return res


def ortholog_map() -> pd.DataFrame:
    rows = []
    for name, f in TABLES.items():
        t = pd.read_csv(f, usecols=lambda c: c in ("hsapiens_gene_id", "mmusculus_gene_id"))
        t["table"] = name
        rows.append(t)
    a = pd.concat(rows).dropna(subset=["hsapiens_gene_id"])
    n = a.dropna(subset=["mmusculus_gene_id"]).groupby("hsapiens_gene_id")["mmusculus_gene_id"].nunique()
    conflict = set(n.index[n > 1])
    print("human genes with different mouse genes across tables (dropped from the mouse side):", len(conflict), flush=True)
    m = a.dropna(subset=["mmusculus_gene_id"]).drop_duplicates("hsapiens_gene_id").set_index("hsapiens_gene_id")["mmusculus_gene_id"]
    m = m[~m.index.isin(conflict)]
    out = pd.DataFrame(index=pd.Index(sorted(a.hsapiens_gene_id.unique()), name="hsapiens_gene_id"))
    out["mouse_ens"] = m.reindex(out.index)
    return out


def mouse_terms():
    """MGI null/knockout single-marker genotypes and IMPC significant calls: gene -> MP terms, by source."""
    al = pd.read_csv(DL / "mgi/MGI_PhenotypicAllele.rpt", sep="\t", comment="#", header=None, dtype=str,
                     names=["allele", "symbol", "name", "type", "attr", "pmid", "marker", "msym", "refseq", "ens", "mp", "syn", "mname"])
    null = set(al.loc[al["attr"].fillna("").str.contains("Null/knockout") & ~al["type"].fillna("").str.contains("Transgenic"), "allele"])
    gp = pd.read_csv(DL / "mgi/MGI_GenePheno.rpt", sep="\t", header=None, dtype=str,
                     names=["comp", "alleles", "allele_ids", "bg", "mp", "pmid", "marker", "genotype", "x"])
    keep = gp["allele_ids"].fillna("").apply(lambda s: bool(s) and all(a in null for a in s.split("|")))
    gp = gp[keep & gp["marker"].notna() & ~gp["marker"].fillna("").str.contains(r"[|,]")]
    print("MGI rows kept", len(gp), "markers", gp.marker.nunique(), flush=True)
    im = pd.read_csv(DL / "impc/genotype-phenotype-assertions-ALL.csv.gz", dtype=str,
                     usecols=["marker_accession_id", "mp_term_id", "zygosity", "pipeline_name"])
    im = im.dropna(subset=["marker_accession_id", "mp_term_id"])
    print("IMPC rows", len(im), "markers", im.marker_accession_id.nunique(), flush=True)
    return gp[["marker", "mp", "comp"]], im[["marker_accession_id", "mp_term_id"]]


def human_terms():
    g2d = pd.read_csv(DL / "hpo/genes_to_disease.txt", sep="\t", dtype=str)
    men = g2d[g2d.association_type.eq("MENDELIAN")]
    men = men.assign(ncbi=men.ncbi_gene_id.str.replace("NCBIGene:", "", regex=False))
    pairs = set(zip(men.ncbi, men.disease_id))
    p = pd.read_csv(DL / "hpo/genes_to_phenotype.txt", sep="\t", dtype=str)
    p = p[[(a, b) in pairs for a, b in zip(p.ncbi_gene_id, p.disease_id)]]
    f = p["frequency"].fillna("-")
    zero = f.str.match(r"^0/\d+$") | f.eq("HP:0040285")
    print("HPO Mendelian rows", len(p), "excluded-frequency", int(zero.sum()), flush=True)
    p = p[~zero]
    return p[["ncbi_gene_id", "hpo_id", "disease_id"]], men


def gene2ens():
    """(human ENSG -> NCBI gene, mouse ENSMUSG -> NCBI gene) from NCBI gene2ensembl and HGNC."""
    hs, mm = {}, {}
    with gzip.open(DL / "ncbi/gene2ensembl.gz", "rt") as fh:
        next(fh)
        for line in fh:
            t, gid, ens = line.split("\t", 3)[:3]
            if t == "9606":
                hs.setdefault(ens, gid)
            elif t == "10090":
                mm.setdefault(ens, gid)
    h = pd.read_csv(REPO / "data/analysis/x3_annotations/hgnc_complete_set.txt", sep="\t", dtype=str,
                    usecols=["hgnc_id", "entrez_id", "ensembl_gene_id"]).dropna(subset=["ensembl_gene_id"])
    # HGNC takes precedence where it gives an NCBI gene
    for e, n in zip(h.ensembl_gene_id, h.entrez_id):
        if isinstance(n, str):
            hs[e] = n
    return hs, mm, h.drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")["hgnc_id"]


def pubmed_counts() -> tuple[pd.Series, pd.Series]:
    t = pd.read_csv(DL / "ncbi/gene2pubmed.gz", sep="\t", dtype=str, usecols=[0, 1])
    t.columns = ["tax", "gene"]
    return t[t.tax.eq("9606")].gene.value_counts(), t[t.tax.eq("10090")].gene.value_counts()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    A = ortholog_map()
    mpa = obo_ancestors(DL / "onto/mp.obo")
    hpa = obo_ancestors(DL / "onto/hp.obo")
    mrk = pd.read_csv(DL / "mgi/MRK_ENSEMBL.rpt", sep="\t", header=None, dtype=str, usecols=[0, 5], names=["mgi", "ens"])
    ens2mgi = mrk.dropna().drop_duplicates("ens").set_index("ens")["mgi"]
    A["mgi"] = A["mouse_ens"].map(ens2mgi)

    # top-level physiological systems, for pleiotropy
    def children(path, root):
        kids, cur = set(), None
        for line in path.read_text().splitlines():
            if line.startswith("id: "):
                cur = line[4:].strip()
            elif line.startswith(f"is_a: {root}"):
                kids.add(cur)
        return kids

    mp_top = children(DL / "onto/mp.obo", "MP:0000001")
    hp_top = children(DL / "onto/hp.obo", "HP:0000118")
    gp, im = mouse_terms()
    terms_all, terms_impc = defaultdict(set), defaultdict(set)
    for m, t in zip(gp.marker, gp.mp):
        terms_all[m].add(t)
    for m, t in zip(im.marker_accession_id, im.mp_term_id):
        terms_all[m].add(t)
        terms_impc[m].add(t)

    def expand(ts):
        out = set()
        for t in ts:
            out |= mpa.get(t, {t})
        return out

    exp_all = {m: expand(ts) for m, ts in terms_all.items()}
    exp_impc = {m: expand(ts) for m, ts in terms_impc.items()}
    for k, ids in MP_SYSTEMS.items():
        s = set(ids)
        A[f"M_{k}"] = [bool(exp_all[m] & s) if m in exp_all else False for m in A.mgi]
        A[f"Mimpc_{k}"] = [bool(exp_impc[m] & s) if m in exp_impc else False for m in A.mgi]
    A["M_any"] = [bool(exp_all[m] & mp_top) if m in exp_all else False for m in A.mgi]
    A["n_mp_sys"] = [len(exp_all[m] & mp_top) if m in exp_all else 0 for m in A.mgi]
    A["Mimpc_any"] = [bool(exp_impc[m] & mp_top) if m in exp_impc else False for m in A.mgi]

    # viability
    v = pd.read_csv(DL / "impc/viability.csv.gz", dtype=str)
    vcol = [c for c in v.columns if "Viability Phenotype" in c or c == "Viability Phenotype HOMs/HEMIs"]
    print("IMPC viability columns:", vcol, v.columns.tolist()[20:27], flush=True)
    state = v.iloc[:, 22].astype(str)  # column holding 'Homozygous - Lethal' etc. (the header is shifted by one here; values printed below)
    vt = pd.DataFrame({"mgi": v["Gene Accession Id"], "s": state})
    vt["lethal"] = vt.s.str.contains("Lethal", case=False)
    vt["subviable"] = vt.s.str.contains("Subviable", case=False)
    vt["viable"] = vt.s.str.contains("Viable", case=False) & ~vt.subviable
    gv = vt.groupby("mgi")[["lethal", "subviable", "viable"]].any()
    print("IMPC viability states", vt.s.value_counts().head(8).to_dict(), flush=True)
    A["impc_tested"] = A.mgi.isin(gv.index)
    # MGI lethal: homozygous null annotation under the lethal terms (heterozygous comp a/+ excluded)
    hom = gp[~gp.comp.fillna("").str.contains(r"/\+|/\?")]
    lm = set(LETHAL_MP)
    leth_mgi = {m for m, t in zip(hom.marker, hom.mp) if mpa.get(t, {t}) & lm}
    im_leth = {m for m, ts in exp_impc.items() if ts & lm}
    A["mouse_lethal_mgi"] = A.mgi.isin(leth_mgi)
    flag = lambda c: A.mgi.map(gv[c]).astype("boolean").fillna(False).astype(bool)
    A["mouse_lethal_impc"] = flag("lethal") | A.mgi.isin(im_leth)
    A["mouse_viable_impc"] = flag("viable") & ~flag("lethal") & ~flag("subviable")
    A["mouse_lethal"] = A.mouse_lethal_mgi | A.mouse_lethal_impc
    # known lethal status: lethal, or viable in IMPC and never annotated lethal
    A["lethal_known"] = np.where(A.mouse_lethal, 1.0, np.where(A.mouse_viable_impc, 0.0, np.nan))

    # human
    hs, mm, hgnc = gene2ens()
    A["ncbi"] = [hs.get(e) for e in A.index]
    p, men = human_terms()
    ht, ht_omim = defaultdict(set), defaultdict(set)
    for g, t, d in zip(p.ncbi_gene_id, p.hpo_id, p.disease_id):
        ht[g] |= hpa.get(t, {t})
        if d.startswith("OMIM:"):
            ht_omim[g] |= hpa.get(t, {t})
    for k, ids in HP_SYSTEMS.items():
        s = set(ids)
        A[f"H_{k}"] = [bool(ht_omim[g] & s) if g in ht_omim else False for g in A.ncbi]
        A[f"Hall_{k}"] = [bool(ht[g] & s) if g in ht else False for g in A.ncbi]
    A["H_any"] = [bool(ht_omim[g] & hp_top) if g in ht_omim else False for g in A.ncbi]
    A["Hall_any"] = [bool(ht[g] & hp_top) if g in ht else False for g in A.ncbi]
    A["n_hp_sys"] = [len(ht_omim[g] & hp_top) if g in ht_omim else 0 for g in A.ncbi]
    # ClinGen
    cg = pd.read_csv(DL / "clingen/Clingen-Gene-Disease-Summary.csv", skiprows=4, dtype=str)
    cg = cg[cg["GENE ID (HGNC)"].astype(str).str.startswith("HGNC:")]
    okc = cg[cg["CLASSIFICATION"].isin(["Definitive", "Strong", "Moderate"])]
    A["clingen_ok"] = A.index.map(hgnc).isin(set(okc["GENE ID (HGNC)"]))
    # DepMap, columns 'SYMBOL (ENTREZ)'
    de = pd.read_csv(DL / "depmap/CRISPRGeneEffect_24Q4.csv", index_col=0)
    ids = de.columns.str.extract(r"\((\d+)\)$")[0]
    de.columns = ids
    de = de.loc[:, de.columns.notna()]
    mean_eff = de.mean(0)
    frac = (de < -0.5).mean(0)
    A["depmap_mean"] = A.ncbi.map(mean_eff)
    A["depmap_frac05"] = A.ncbi.map(frac)
    ce = pd.read_csv(DL / "depmap/CRISPRInferredCommonEssentials_24Q4.csv")
    ce_ids = set(ce.iloc[:, 0].astype(str).str.extract(r"\((\d+)\)$")[0].dropna())
    A["depmap_common"] = np.where(A.depmap_mean.notna(), A.ncbi.isin(ce_ids), np.nan)
    # PubMed
    ph, pm = pubmed_counts()
    A["pubmed_h"] = A.ncbi.map(ph).fillna(0.0)
    A["mouse_ncbi"] = A.mouse_ens.map(mm)
    A["pubmed_m"] = A.mouse_ncbi.map(pm).fillna(0.0)
    A["log_pubmed_h"] = np.log1p(A.pubmed_h)
    A["log_pubmed_m"] = np.log1p(A.pubmed_m)
    A.to_parquet(OUT / "pheno.parquet")
    print(A.shape)
    cov = {c: int(A[c].sum()) for c in A.columns if c.startswith(("M_", "H_", "Mimpc_", "Hall_")) or c in ("impc_tested", "clingen_ok", "mouse_lethal", "depmap_common")}
    print(cov)
    print("both any", int((A.M_any & A.H_any).sum()), "lethal known", A.lethal_known.value_counts().to_dict(), "ncbi", A.ncbi.notna().sum(), "mgi", A.mgi.notna().sum())


if __name__ == "__main__":
    main()
