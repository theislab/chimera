"""Y4 (EXPLORATORY): gene annotations for regions. No readout or gold is read here.

Compara 110 (`data/revision/ensembl110/`): gene-age proxy and paralog status of human genes.
UniProt, HGNC, Lambert (`data/revision/x3_annotations/`), Hallmark (`data/revision/functional/`): classes and sets.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from revision.common import gene_ids
from revision.tasks.Y4_regions import rules as RU

REPO = gene_ids.REPO_ROOT
sys.path.insert(0, str(REPO / "revision/tasks/X3_cross_cohort"))
import annotations as AN  # noqa: E402

CACHE = REPO / "results/revision/Y4/cache"
COMPARA = REPO / "data/revision/ensembl110/Compara.110.protein_default.homologies.tsv.gz"
HALLMARK = REPO / "data/revision/functional/Enrichr.MSigDB_Hallmark_2020.server_response.txt"
CHROM = REPO / "results/revision/X1/genes/chromosome_ensembl110.csv"

# Named classes beyond X3, from UniProt keywords. ``metabolic`` = enzyme with a metabolism keyword.
KW_CLASSES = {
    "ligand": ["KW-0202", "KW-0339", "KW-0372", "KW-0527"],  # cytokine, growth factor, hormone, neuropeptide
    "secreted": ["KW-0964"],
    "membrane": ["KW-0472"],
    "signal_peptide": ["KW-0732"],
    "immune": ["KW-0391", "KW-0399", "KW-1064", "KW-0395"],  # immunity, innate, adaptive, inflammatory response
    "phosphatase": ["KW-0904"],
    "ECM": ["KW-0272"],
    "transcription": ["KW-0804"],
    "cell_adhesion": ["KW-0130"],
}


def chrom_table() -> pd.Series:
    t = pd.read_csv(CHROM, dtype={"chrom": str}).drop_duplicates("hsapiens_gene_id")
    return t.set_index("hsapiens_gene_id")["chrom"]


def split_of(chrom: pd.Series) -> pd.Series:
    out = pd.Series("none", index=chrom.index, dtype=object)
    out[chrom.isin(RU.ODD)] = "odd"
    out[chrom.isin(RU.EVEN)] = "even"
    out[chrom.eq("X")] = "X"
    return out


def compara() -> pd.DataFrame:
    """Per human gene: age class, paralog count, highest paralog identity, paralog on the same chromosome,
    paralog on an odd autosome (leak control for the split)."""
    f = CACHE / "compara_human.parquet"
    if f.exists():
        return pd.read_parquet(f)
    use = ["gene_stable_id", "species", "identity", "homology_type", "homology_gene_stable_id", "homology_species"]
    para, age = [], {}
    rank = {name: i for i, (name, _) in enumerate(RU.AGE_CLADES)}
    sp2 = {s: name for name, ss in RU.AGE_CLADES for s in ss}
    for ch in pd.read_csv(COMPARA, sep="\t", usecols=use, chunksize=2_000_000, dtype=str):
        ch = ch[ch.species == "homo_sapiens"]
        o = ch[ch.homology_type.str.startswith("ortholog") & ch.homology_species.isin(sp2)]
        for g, s in zip(o.gene_stable_id, o.homology_species):
            k = rank[sp2[s]]
            if g not in age or k < age[g]:
                age[g] = k
        para.append(ch.loc[ch.homology_type.eq("within_species_paralog") & ch.homology_species.eq("homo_sapiens"),
                           ["gene_stable_id", "homology_gene_stable_id", "identity"]])
    p = pd.concat(para, ignore_index=True)
    p["identity"] = pd.to_numeric(p["identity"], errors="coerce")
    chrom = chrom_table()
    p["c1"] = p.gene_stable_id.map(chrom)
    p["c2"] = p.homology_gene_stable_id.map(chrom)
    p = p.drop_duplicates(["gene_stable_id", "homology_gene_stable_id"])
    a = p.groupby("gene_stable_id").agg(n_paralog=("identity", "size"), max_paralog_id=("identity", "max"))
    a["paralog_same_chrom"] = p.assign(s=p.c1 == p.c2).groupby("gene_stable_id")["s"].any()
    a["paralog_on_odd"] = p.assign(s=p.c2.isin(RU.ODD)).groupby("gene_stable_id")["s"].any()
    names = [n for n, _ in RU.AGE_CLADES]
    genes = sorted(set(age) | set(a.index))
    out = a.reindex(genes)
    out["n_paralog"] = out["n_paralog"].fillna(0).astype(int)
    out["paralog_same_chrom"] = out["paralog_same_chrom"].astype("boolean").fillna(False).astype(bool)
    out["paralog_on_odd"] = out["paralog_on_odd"].astype("boolean").fillna(False).astype(bool)
    out["age_class"] = [names[age[g]] if g in age else None for g in out.index]
    out.index.name = "hsapiens_gene_id"
    CACHE.mkdir(parents=True, exist_ok=True)
    out.to_parquet(f)
    return out


def hallmark() -> dict[str, set]:
    sets = {}
    for line in HALLMARK.read_text().splitlines():
        p = line.split("\t")
        sets[p[0]] = {x.split(",")[0] for x in p[2:] if x}
    return sets


def class_columns(genes: pd.Index, symbols: pd.Series) -> pd.DataFrame:
    """Boolean columns ``cls_*``, ``kw_*``, ``hm_*``, ``fam_*``, ``grp_*`` for the given human IDs."""
    genes = pd.Index(genes)
    cl = AN.gene_classes(genes)
    out = {f"cls_{k}": cl[k].to_numpy(bool) for k in AN.CLASS_ORDER}
    up = AN.uniprot_by_gene().reindex(genes)
    kw = up["kw"].apply(lambda x: x if isinstance(x, set) else set())
    kt = AN.keyword_table()
    for name, ids in KW_CLASSES.items():
        out[f"cls_{name}"] = np.array([bool(s & set(ids)) for s in kw])
    metab_kw = set(kt.index[kt.kw_name.str.contains("metabolism|biosynthesis|catabolism", case=False, regex=True)
                            & kt.kw_category.eq("Biological process")])
    out["cls_metabolic"] = cl["enzyme"].to_numpy(bool) & np.array([bool(s & metab_kw) for s in kw])
    # every keyword of the four categories; the coverage filter in regions.py decides which are tested
    keep = set(kt.index[kt.kw_category.isin(AN.PROGRAM_CATEGORIES)])
    allkw: dict[str, list] = {}
    for i, s in enumerate(kw):
        for k in s & keep:
            allkw.setdefault(k, []).append(i)
    for k, idx in allkw.items():
        if len(idx) >= 2 * RU.MIN_GENES:
            v = np.zeros(len(genes), bool)
            v[idx] = True
            out["kw_" + kt.loc[k, "kw_name"].replace(" ", "_")] = v
    sy = symbols.reindex(genes).astype(str).to_numpy()
    for name, s in hallmark().items():
        v = np.isin(sy, list(s))
        if v.sum() >= 2 * RU.MIN_GENES:
            out["hm_" + name.replace(" ", "_")] = v
    fam = cl["family"].fillna("").astype(str)
    for name, n in fam.value_counts().items():
        if name and n >= 2 * RU.MIN_GENES:
            out["fam_" + name.replace(" ", "_")[:50]] = (fam == name).to_numpy()
    h = AN._hgnc().drop_duplicates("ensembl_gene_id").set_index("ensembl_gene_id")["gene_group"].reindex(genes)
    grp: dict[str, list] = {}
    for i, s in enumerate(h.fillna("")):
        for x in s.split("|"):
            if x:
                grp.setdefault(x, []).append(i)
    for k, idx in grp.items():
        if len(idx) >= 2 * RU.MIN_GENES:
            v = np.zeros(len(genes), bool)
            v[idx] = True
            out["grp_" + k.replace(" ", "_")[:50]] = v
    return pd.DataFrame(out, index=genes)
