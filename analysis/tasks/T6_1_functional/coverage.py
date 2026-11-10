"""T6.1 ID coverage of the functional-layer sources. Counts only.

No Chimera readout is read. Each number is an ID count in the liver (8,879) and testis
(15,732) ortholog-table vocabularies, the same denominators as A7 section 2.6.

Usage: python -I coverage.py <chimera_repo> <out_dir>
"""

from __future__ import annotations

import gzip
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

LIVER_LINEAGE = re.compile(r"liver|hepat|hep_g2|hep_3b|huh|lo2", re.I)
TFS = ["HNF4A", "CEBPA", "FOXA1", "FOXA2", "HNF1A", "ONECUT1", "RXRA", "PPARA", "NR1H3"]


def log(*a):
    print(*a, flush=True)


def string_gene_map(aliases: Path, prefix: str) -> dict[str, str]:
    """STRING protein -> Ensembl gene, from the ``Ensembl_gene`` alias; multi-gene proteins dropped."""
    m = defaultdict(set)
    with gzip.open(aliases, "rt") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == 3 and p[2] == "Ensembl_gene" and p[1].startswith(prefix):
                m[p[0]].add(p[1])
    return {k: next(iter(v)) for k, v in m.items() if len(v) == 1}


def edges_in(edges: pd.DataFrame, voc: set[str]) -> tuple[int, int]:
    e = edges[edges.a.isin(voc) & edges.b.isin(voc) & (edges.a != edges.b)]
    key = pd.Series([tuple(sorted(x)) for x in zip(e.a, e.b)]).drop_duplicates() if len(e) else pd.Series([], dtype=object)
    nodes = set(e.a) | set(e.b)
    return len(key), len(nodes)


def main(repo: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo / "analysis" / "tasks" / "T4_1_gold"))
    from gold_checks import SymbolMapper, gtf_gene_names  # noqa: E402

    sys.path.insert(0, str(repo / "analysis" / "tasks" / "T6_1_functional"))
    from xlsx_min import read_sheet  # noqa: E402

    F = repo / "data/analysis/functional"
    P = repo / "data/processed"
    L = pd.read_csv(P / "ortholog_table.csv")
    T = pd.read_csv(P / "ortholog_table_testis.csv")
    hv = {"liver": set(L.hsapiens_gene_id), "testis": set(T.hsapiens_gene_id)}
    mv = {"liver": set(L.mmusculus_gene_id.dropna()), "testis": set(T.mmusculus_gene_id.dropna())}
    rows = []

    def add(src, measure, liver, testis, note=""):
        rows.append({"source": src, "measure": measure, "liver": liver, "testis": testis, "note": note})

    # HuRI (ENSG pairs)
    h = pd.read_csv(F / "HuRI.tsv", sep="\t", header=None, names=["a", "b"], dtype=str)
    log("HuRI rows", len(h))
    el, nl = edges_in(h, hv["liver"])
    et, nt = edges_in(h, hv["testis"])
    add("HuRI", "edges both ends in vocabulary (unique, no self)", el, et, f"{len(h)} rows")
    add("HuRI", "nodes on those edges", nl, nt)
    # STRING physical, direct evidence = 'experiments' column only
    for taxon, prefix, voc in (("9606", "ENSG", hv), ("10090", "ENSMUSG", mv)):
        gmap = string_gene_map(F / f"{taxon}.protein.aliases.v12.0.txt.gz", prefix)
        s = pd.read_csv(F / f"{taxon}.protein.physical.links.full.v12.0.txt.gz", sep=" ",
                        usecols=["protein1", "protein2", "experiments", "experiments_transferred", "combined_score"])
        s["a"], s["b"] = s.protein1.map(gmap), s.protein2.map(gmap)
        log(f"STRING {taxon}: rows {len(s)}, proteins mapped {s.a.notna().mean():.3f}")
        for thr in (400, 150):
            d = s[s.experiments >= thr]
            el, nl = edges_in(d, voc["liver"])
            et, nt = edges_in(d, voc["testis"])
            add(f"STRING_{taxon}_direct", f"experiments>={thr}: edges", el, et,
                "mouse IDs via ortholog table mmusculus_gene_id" if taxon == "10090" else "")
            add(f"STRING_{taxon}_direct", f"experiments>={thr}: nodes", nl, nt)
        tr = s[(s.experiments < 400) & (s.experiments_transferred >= 400)]
        add(f"STRING_{taxon}", "transfer-only edges (experiments<400, transferred>=400), excluded", *[edges_in(tr, voc[k])[0] for k in ("liver", "testis")])
    # Symbol mapper for the gene-set sources
    names, _ = gtf_gene_names(repo / "data/analysis/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz")
    mapper = SymbolMapper(repo / "data/analysis/gold/G10/hgnc_complete_set.txt", names)
    # CollecTRI human, gene sets only
    c = pd.read_csv(F / "collectri_human.tsv", sep="\t", dtype=str)
    syms = sorted(set(c.source_genesymbol.dropna()) | set(c.target_genesymbol.dropna()))
    mm = mapper.map(syms).set_index("symbol").ensg
    c["tf"], c["tg"] = c.source_genesymbol.map(mm), c.target_genesymbol.map(mm)
    complexes = int(c.source_genesymbol.str.contains("_", na=False).sum())
    per_tf = {k: c[c.tg.isin(hv[k])].groupby("source_genesymbol").tg.nunique() for k in ("liver", "testis")}
    add("CollecTRI_human", "regulons (TF or complex) with >=1 target in vocabulary",
        len(per_tf["liver"]), len(per_tf["testis"]),
        f"{len(c)} edges, {c.source_genesymbol.nunique()} sources ({complexes} edges from complexes such as AP1)")
    add("CollecTRI_human", "regulons with >=10 targets in vocabulary",
        int((per_tf["liver"] >= 10).sum()), int((per_tf["testis"] >= 10).sum()))
    tf_in = {k: c[c.tg.isin(hv[k]) & c.tf.isin(hv[k])].groupby("tf").tg.nunique() for k in ("liver", "testis")}
    add("CollecTRI_human", "single-gene TFs in vocabulary with >=1 target in vocabulary",
        len(tf_in["liver"]), len(tf_in["testis"]))
    add("CollecTRI_human", "single-gene TFs in vocabulary with >=10 targets in vocabulary",
        int((tf_in["liver"] >= 10).sum()), int((tf_in["testis"] >= 10).sum()))
    # Skinnider 2021 mouse tissue interactomes (mmc2, legend "Table S3"), mouse symbols
    sk = read_sheet(F / "skinnider2021/Skinnider2021_mmc2.xlsx", "Table S3")
    sk = pd.DataFrame(sk[1:], columns=sk[0])
    log("Skinnider rows", len(sk), "tissues", sk.Tissue.value_counts().to_dict())
    msym = {}
    for tab in (L, T):
        for e, s_ in zip(tab.mmusculus_gene_id, tab.mmusculus_gene_name):
            if isinstance(s_, str):
                msym.setdefault(s_, set()).add(e)
    msym = {k: next(iter(v)) for k, v in msym.items() if len(v) == 1}
    sk["a"], sk["b"] = sk["Protein A"].map(msym), sk["Protein B"].map(msym)
    for tissue in ("Liver", None):
        d = sk if tissue is None else sk[sk.Tissue == tissue]
        lab = "all tissues" if tissue is None else tissue
        el, nl = edges_in(d, mv["liver"])
        et, nt = edges_in(d, mv["testis"])
        add("Skinnider2021", f"{lab}: edges, mouse symbol -> ortholog-table mouse ID", el, et,
            f"{len(d)} rows; symbols mapped {d.a.notna().mean():.3f}")
        add("Skinnider2021", f"{lab}: nodes", nl, nt)
    # Gene-set libraries
    hall = Path.home() / ".cache/gseapy/Enrichr.MSigDB_Hallmark_2020.gmt"
    reac24 = Path.home() / ".cache/gseapy/Enrichr.Reactome_Pathways_2024.gmt"
    z = zipfile.ZipFile(F / "ReactomePathways.gmt.zip")
    reac_txt = z.read("ReactomePathways.gmt").decode()

    def gmt(text: str) -> dict[str, list[str]]:
        d = {}
        for line in text.splitlines():
            p = line.split("\t")
            d[p[0]] = [g for g in p[2:] if g]
        return d

    for name, sets in (("Hallmark_2020 (Enrichr)", gmt(hall.read_text())),
                       ("Reactome_Pathways_2024 (Enrichr)", gmt(reac24.read_text())),
                       ("ReactomePathways.gmt (reactome.org current)", gmt(reac_txt))):
        allsym = sorted(set().union(*sets.values()))
        mm = mapper.map(allsym).set_index("symbol").ensg
        res = {}
        for k in ("liver", "testis"):
            sizes = pd.Series({t: len({mm.get(g) for g in v} & hv[k]) for t, v in sets.items()})
            res[k] = (int((sizes >= 15).sum()), int(((sizes >= 15) & (sizes <= 200)).sum()))
        add(name, "sets with >=15 genes in vocabulary", res["liver"][0], res["testis"][0], f"{len(sets)} sets, {len(allsym)} symbols")
        add(name, "sets with 15-200 genes in vocabulary (before Jaccard pruning)", res["liver"][1], res["testis"][1])
    # ChIP-Atlas: experiment counts by cell type, and targets in vocabulary
    ens_by_mouse_sym = {}
    for tab in (L, T):
        for e, s in zip(tab.mmusculus_gene_id, tab.mmusculus_gene_name):
            if isinstance(s, str):
                ens_by_mouse_sym.setdefault(s, set()).add(e)
    chip = []
    for g in ("hg38", "mm10"):
        for tf in TFS:
            name = tf if g == "hg38" else tf[0] + tf[1:].lower()
            f = F / "chipatlas" / g / f"{name}.5.tsv"
            hdr = f.open().readline().rstrip("\n").split("\t")
            exps = [x for x in hdr[2:] if x != "STRING"]
            cells = Counter(x.split("|", 1)[1] for x in exps)
            liver = sum(v for k, v in cells.items() if LIVER_LINEAGE.search(k))
            tg = pd.read_csv(f, sep="\t", usecols=[0], dtype=str).iloc[:, 0]
            if g == "hg38":
                ids = set(mapper.map(sorted(set(tg))).ensg.dropna())
                inl, int_ = len(ids & hv["liver"]), len(ids & hv["testis"])
            else:
                ids = {e for s in tg for e in ens_by_mouse_sym.get(s, ()) if len(ens_by_mouse_sym.get(s, ())) == 1}
                inl, int_ = len(ids & mv["liver"]), len(ids & mv["testis"])
            chip.append({"genome": g, "tf": tf, "experiments": len(exps), "liver_lineage_experiments": liver,
                         "liver_lineage_cell_types": ";".join(f"{k}:{v}" for k, v in cells.items() if LIVER_LINEAGE.search(k)),
                         "target_rows": len(tg), "targets_in_liver_vocab": inl, "targets_in_testis_vocab": int_,
                         "has_STRING_column": hdr[-1] == "STRING"})
    chip = pd.DataFrame(chip)
    chip.to_csv(out / "chipatlas_experiments.tsv", sep="\t", index=False)
    both = chip.pivot(index="tf", columns="genome", values="liver_lineage_experiments")
    log("ChIP-Atlas liver-lineage experiments per TF\n" + both.to_string())
    add("ChIP-Atlas", "TFs of the 9-panel with liver-lineage experiments in both genomes (preliminary label match)",
        int(((both.hg38 > 0) & (both.mm10 > 0)).sum()), None, ",".join(both.index[(both.hg38 > 0) & (both.mm10 > 0)]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "functional_coverage.tsv", sep="\t", index=False)
    log(df.to_string(index=False))
    log(chip.drop(columns="liver_lineage_cell_types").to_string(index=False))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
