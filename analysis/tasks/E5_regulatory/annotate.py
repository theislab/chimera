"""E5 annotation table: one row per human gene, every E5 property. Run with `python -I` (reads downloaded files only).

Usage (from chimera/):  mamba run -n chimera_env python -I analysis/tasks/E5_regulatory/annotate.py
Writes results/analysis/E5/: props.parquet, props_coverage.csv, regulons.parquet (DoRothEA A-C TF -> target gene),
paralogs_any.parquet (Compara 110 human within-species paralog pairs), genes_pos.parquet (canonical TSS, strand, span).
Reused: Y2 regulatory annotation (phyloP, CpG, TATA, ABC, UTR lengths, canonical TSS), E2 props, Y2 PubMed.
"""
from __future__ import annotations

import gzip
import io
import re
import subprocess
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))  # rules.py only (no package import under -I)
import rules as RU  # noqa: E402

DL = REPO / "data" / "analysis" / "E5"
OUT = REPO / "results" / "analysis" / "E5"
OUT.mkdir(parents=True, exist_ok=True)
GTF = REPO / "data/analysis/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz"
COMPARA = REPO / "data/analysis/ensembl110/Compara.110.protein_default.homologies.tsv.gz"
FA = REPO / "data/analysis/Y2/ucsc/hg38.fa.gz"
ATTR = re.compile(r'(\S+) "([^"]*)"')
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]


def hgnc_map() -> dict:
    h = pd.read_csv(REPO / "data/analysis/x3_annotations/hgnc_complete_set.txt", sep="\t", dtype=str,
                    usecols=["symbol", "ensembl_gene_id", "prev_symbol", "alias_symbol"]).dropna(subset=["ensembl_gene_id"])
    m = dict(zip(h.symbol, h.ensembl_gene_id))
    for col in ("prev_symbol", "alias_symbol"):  # current symbols win; previous before alias; ambiguous dropped
        cand = defaultdict(set)
        for s, g in zip(h[col], h.ensembl_gene_id):
            if isinstance(s, str):
                for x in s.split("|"):
                    cand[x.strip()].add(g)
        for x, gs in cand.items():
            if x not in m and len(gs) == 1:
                m[x] = next(iter(gs))
    return m


def parse_gtf():
    tx = []  # gene, transcript, chrom, start0, end, strand, biotype, canonical
    utr3 = defaultdict(list)  # canonical 3'UTR segments
    exons = defaultdict(int)
    genes = {}
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            p = line.rstrip("\n").split("\t")
            if p[0] not in CHROMS or p[2] not in ("gene", "transcript", "exon", "three_prime_utr"):
                continue
            a = dict(ATTR.findall(p[8]))
            g = a["gene_id"]
            s, e = int(p[3]) - 1, int(p[4])
            if p[2] == "gene":
                genes[g] = (p[0], s, e, p[6], a.get("gene_biotype"))
                continue
            can = 'tag "Ensembl_canonical"' in p[8]
            if p[2] == "transcript":
                tx.append((g, a["transcript_id"], p[0], s, e, p[6], a.get("transcript_biotype"), can))
            elif can and p[2] == "exon":
                exons[g] += 1
            elif can and p[2] == "three_prime_utr":
                utr3[g].append((p[0], s, e, p[6]))
    T = pd.DataFrame(tx, columns=["g", "t", "chrom", "start", "end", "strand", "biotype", "canonical"])
    G = pd.DataFrame.from_dict(genes, orient="index", columns=["chrom", "start", "end", "strand", "gene_biotype"])
    return T, G, exons, utr3


def n_distinct(pos: np.ndarray, sep: int) -> int:
    pos = np.sort(pos)
    return int(1 + np.sum(np.diff(pos) > sep)) if len(pos) else 0


def read_fasta_chroms(chroms: set) -> dict:
    seqs, cur, buf = {}, None, []
    with gzip.open(FA, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if cur in chroms:
                    seqs[cur] = "".join(buf)
                cur, buf = line[1:].strip().removeprefix("chr"), []
                continue
            if cur in chroms:
                buf.append(line.strip())
    if cur in chroms:
        seqs[cur] = "".join(buf)
    return seqs


COMP = str.maketrans("ACGTacgtNn", "TGCAtgcaNn")
ARE2 = re.compile(r"(?=[AT][AT]ATTTA[AT][AT])")


def are_features(utr3: dict) -> pd.DataFrame:
    seqs = read_fasta_chroms(set(CHROMS))
    rows = []
    for g, segs in utr3.items():
        strand = segs[0][3]
        segs = sorted(segs, key=lambda x: x[1])
        s = "".join(seqs[c][a:b] for c, a, b, _ in segs).upper()
        if strand == "-":
            s = s.translate(COMP)[::-1]
        L = len(s)
        if L == 0:
            continue
        n5 = len(re.findall(r"(?=ATTTA)", s))
        rows.append((g, L, n5 / L * 1000, float(bool(ARE2.search(s)))))
    return pd.DataFrame(rows, columns=["g", "utr3_seq_len", "are_density", "are_class2"]).set_index("g")


def great_domains(P: pd.DataFrame) -> pd.Series:
    """GREAT basal-plus-extension on protein-coding canonical TSSs; P has chrom, tss, strand."""
    up, down = RU.GREAT_BASAL
    out = {}
    for c, d in P.groupby("chrom"):
        d = d.sort_values("tss")
        tss = d.tss.to_numpy()
        plus = (d.strand == "+").to_numpy()
        bs = np.where(plus, tss - up, tss - down)
        be = np.where(plus, tss + down, tss + up)
        n = len(d)
        # extension: to the nearest basal domain of the neighbouring genes, at most 1 Mb from the TSS
        prev_be = np.r_[-np.inf, np.maximum.accumulate(be)[:-1]]
        next_bs = np.r_[np.minimum.accumulate(bs[::-1])[::-1][1:], np.inf]
        lo = np.minimum(bs, np.maximum(tss - RU.GREAT_MAX, prev_be))
        hi = np.maximum(be, np.minimum(tss + RU.GREAT_MAX, next_bs))
        lo = np.maximum(lo, 0)
        for g, a, b in zip(d.index, lo, hi):
            out[g] = (c, a, b)
        assert n == len(d)
    return pd.DataFrame.from_dict(out, orient="index", columns=["chrom", "dom_lo", "dom_hi"])


def count_in(mids: dict, chrom: str, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    m = mids.get(chrom)
    if m is None:
        return np.zeros(len(lo))
    return np.searchsorted(m, hi, side="right") - np.searchsorted(m, lo, side="left")


def gmt(path: Path) -> dict:
    out = {}
    for line in open(path):
        p = line.rstrip("\n").split("\t")
        out[p[0]] = p[2:]
    return out


def main():
    hg = hgnc_map()
    y2 = pd.read_parquet(REPO / "results/analysis/Y2/annot/regulatory.parquet")
    keep = ["promoter_cgi", "cpg_oe_promoter", "gc_promoter", "tata", "phylop_promoter", "phylop_utr5", "phylop_utr3",
            "phylop_cds", "abc_links_mean", "log_length_utr5", "log_length_utr3", "chrom", "strand", "tss", "biotype"]
    t = y2[keep].copy()
    t.index.name = "hsapiens_gene_id"

    # transcript architecture
    T, G, exons, utr3 = parse_gtf()
    T["tss"] = np.where(T.strand == "+", T.start, T.end - 1)
    T["tes"] = np.where(T.strand == "+", T.end - 1, T.start)
    agg = T.groupby("g").agg(n_tx=("t", "size"), n_pc=("biotype", lambda b: int((b == "protein_coding").sum())))
    agg["n_tss"] = T.groupby("g").tss.apply(lambda v: n_distinct(v.to_numpy(), RU.TSS_SEP))
    agg["n_tes"] = T.groupby("g").tes.apply(lambda v: n_distinct(v.to_numpy(), RU.TSS_SEP))
    t["n_transcripts"] = np.log1p(t.index.map(agg.n_tx).astype(float))
    t["n_pc_transcripts"] = np.log1p(t.index.map(agg.n_pc).astype(float))
    t["n_tss"] = np.log1p(t.index.map(agg.n_tss).astype(float))
    t["n_tes"] = np.log1p(t.index.map(agg.n_tes).astype(float))
    t["n_exons"] = np.log1p(t.index.map(pd.Series(exons)).astype(float))
    t["log_gene_span"] = np.log(t.index.map((G.end - G.start).astype(float)).astype(float))

    # positions of protein-coding canonical TSSs (GREAT, density, bidirectional, neighbourhood)
    pos = t[t.tss.notna() & t.chrom.isin(CHROMS)][["chrom", "strand", "tss", "biotype"]].copy()
    pos["tss"] = pos.tss.astype(np.int64)
    pos["gstart"] = pos.index.map(G.start)
    pos["gend"] = pos.index.map(G.end)
    pos.to_parquet(OUT / "genes_pos.parquet")
    pc = pos[pos.biotype == "protein_coding"]
    dom = great_domains(pc)
    t["log_great_domain"] = np.log(t.index.map((dom.dom_hi - dom.dom_lo).astype(float)).astype(float))

    # cCREs (ENCODE V4)
    cc = pd.read_csv(DL / "encode_ccre/GRCh38-cCREs.V4.bed", sep="\t", header=None, usecols=[0, 1, 2, 5],
                     names=["chrom", "s", "e", "cls"])
    cc["chrom"] = cc.chrom.str.removeprefix("chr")
    cc["mid"] = (cc.s + cc.e) // 2
    mids = {k: {c: np.sort(d.mid.to_numpy()) for c, d in cc[sel].groupby("chrom")}
            for k, sel in (("all", np.ones(len(cc), bool)), ("dELS", (cc.cls == "dELS").to_numpy()),
                           ("pELS", (cc.cls == "pELS").to_numpy()))}
    nd, n50, n2 = {}, {}, {}
    for c, d in pc.groupby("chrom"):
        dd = dom.loc[d.index]
        nd.update(zip(d.index, count_in(mids["dELS"], c, dd.dom_lo.to_numpy(), dd.dom_hi.to_numpy())))
        n50.update(zip(d.index, count_in(mids["all"], c, d.tss.to_numpy() - 50_000, d.tss.to_numpy() + 50_000)))
        n2.update(zip(d.index, count_in(mids["pELS"], c, d.tss.to_numpy() - 2_000, d.tss.to_numpy() + 2_000)))
    t["n_dels_domain"] = np.log1p(t.index.map(pd.Series(nd)).astype(float))
    t["n_ccre_50kb"] = np.log1p(t.index.map(pd.Series(n50)).astype(float))
    t["n_pels_2kb"] = np.log1p(t.index.map(pd.Series(n2)).astype(float))

    # gene density and bidirectional promoters
    dens, bidir = {}, {}
    allpos = pos[pos.biotype.isin(["protein_coding", "lncRNA"])]
    for c, d in pc.groupby("chrom"):
        s = np.sort(d.tss.to_numpy())
        v = d.tss.to_numpy()
        dens.update(zip(d.index, np.searchsorted(s, v + 500_000, "right") - np.searchsorted(s, v - 500_000, "left") - 1))
    for c, d in allpos.groupby("chrom"):
        d = d.sort_values("tss")
        v, st, idx = d.tss.to_numpy(), d.strand.to_numpy(), d.index.to_numpy()
        for i in range(len(d)):
            j0, j1 = np.searchsorted(v, v[i] - RU.BIDIR_MAX), np.searchsorted(v, v[i] + RU.BIDIR_MAX, "right")
            hit = False
            for j in range(j0, j1):
                if j == i or st[j] == st[i]:
                    continue
                # head to head: the partner's TSS lies upstream of (or at) this TSS
                if (st[i] == "+" and v[j] <= v[i] + 100) or (st[i] == "-" and v[j] >= v[i] - 100):
                    hit = True
                    break
            bidir[idx[i]] = float(hit)
    t["gene_density"] = np.log1p(t.index.map(pd.Series(dens)).astype(float))
    t["bidirectional"] = t.index.map(pd.Series(bidir)).astype(float)

    # 3'UTR sequence: AU-rich elements
    are = are_features(utr3)
    t["are_density"] = t.index.map(are.are_density)
    t["are_class2"] = t.index.map(are.are_class2)

    # TargetScan 8 conserved sites of broadly conserved families on the representative transcript
    with zipfile.ZipFile(DL / "targetscan/miR_Family_Info.txt.zip") as z:
        fam = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype=str)
    fam = fam[fam["Species ID"] == "9606"]
    cons = set(fam.loc[fam["Family Conservation?"].astype(int) >= 2, "Seed+m8"])
    with zipfile.ZipFile(DL / "targetscan/Gene_info.txt.zip") as z:
        gi = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype=str)
    gi = gi[(gi["Species ID"] == "9606") & (gi["Representative transcript?"] == "1")]
    gi["g"] = gi["Gene ID"].str.split(".").str[0]
    with zipfile.ZipFile(DL / "targetscan/Summary_Counts.default_predictions.txt.zip") as z:
        sc = pd.read_csv(z.open(z.namelist()[0]), sep="\t", dtype=str,
                         usecols=["Transcript ID", "miRNA family", "Species ID", "Total num conserved sites"])
    sc = sc[(sc["Species ID"] == "9606") & sc["miRNA family"].isin(cons)]
    sc["n"] = sc["Total num conserved sites"].astype(int)
    sc = sc[sc.n > 0]
    per_tx = sc.groupby("Transcript ID").agg(sites=("n", "sum"), fams=("miRNA family", "nunique"))
    gi["sites"] = gi["Transcript ID"].map(per_tx.sites).fillna(0)
    gi["fams"] = gi["Transcript ID"].map(per_tx.fams).fillna(0)
    gi = gi.drop_duplicates("g").set_index("g")
    t["mir_cons_sites"] = np.log1p(t.index.map(gi.sites).astype(float))
    t["mir_cons_families"] = np.log1p(t.index.map(gi.fams).astype(float))
    utr_kb = np.exp(t.log_length_utr3) / 1000
    t["mir_density"] = np.where(utr_kb > 0.05, np.expm1(t.mir_cons_sites) / utr_kb, np.nan)

    # TF in-degree (DoRothEA A-C, CollecTRI) and regulons
    dor = pd.read_csv(DL / "omnipath/dorothea_ABC.tsv", sep="\t", dtype=str)
    col = pd.read_csv(DL / "omnipath/collectri.tsv", sep="\t", dtype=str)
    for d in (dor, col):
        d["tg"] = d.target_genesymbol.map(hg)
        d["sg"] = d.source_genesymbol.map(hg)
    dor = dor[dor.source_genesymbol.str.match(r"^[A-Za-z0-9\-\.]+$", na=False)]
    t["tf_indegree"] = np.log1p(t.index.map(dor.dropna(subset=["tg"]).groupby("tg").source_genesymbol.nunique()).fillna(0)
                                .astype(float))
    t["tf_indegree_ct"] = np.log1p(t.index.map(col.dropna(subset=["tg"]).groupby("tg").source_genesymbol.nunique())
                                   .fillna(0).astype(float))
    reg = dor.dropna(subset=["tg"])[["source_genesymbol", "sg", "tg", "dorothea_level"]].drop_duplicates(["source_genesymbol", "tg"])
    reg = reg[reg.sg != reg.tg]
    reg.rename(columns={"source_genesymbol": "tf", "sg": "tf_gene", "tg": "g"}).to_parquet(OUT / "regulons.parquet")

    # housekeeping, PRC2 / H3K27me3 / bivalent
    hk = pd.read_csv(DL / "hkg/HK_genes.txt", sep="\t", header=None, dtype=str)[0].str.strip()
    hk_g = set(hk.map(hg).dropna())
    t["housekeeping"] = t.index.isin(hk_g).astype(float)
    sets = gmt(DL / "msigdb/c2.cgp.v2025.1.Hs.symbols.gmt")
    # Amendment before any association: MSigDB 2025.1 carries truncated Mikkelsen 2007 sets (the ES HCP bivalent set is
    # absent; HCP with H3K27me3 has 40 genes). bivalent_es = union of the Mikkelsen ES bivalent sets (HCP with H3K27me3,
    # which in ES are bivalent; ICP and LCP with H3K4me3 and H3K27me3); bivalent_npc = Meissner 2008 NPC HCP with H3K4me2
    # and H3K27me3 (351 genes) as a second, larger bivalent set.
    biv = ["MIKKELSEN_ES_HCP_WITH_H3K27ME3", "MIKKELSEN_ES_ICP_WITH_H3K4ME3_AND_H3K27ME3",
           "MIKKELSEN_ES_LCP_WITH_H3K4ME3_AND_H3K27ME3"]
    used = {"prc2_targets": ["BENPORATH_PRC2_TARGETS"], "es_h3k27me3": ["BENPORATH_ES_WITH_H3K27ME3"], "bivalent_es": biv,
            "bivalent_npc": ["MEISSNER_NPC_HCP_WITH_H3K4ME2_AND_H3K27ME3"]}
    for k, ss in used.items():
        gs = {hg.get(x) for s in ss for x in sets[s]} - {None}
        t[k] = t.index.isin(gs).astype(float)
    # PubMed (Y2) and E2 class columns
    ph = pd.read_parquet(REPO / "results/analysis/Y2/annot/pheno.parquet", columns=["log_pubmed_h"])
    t["log_pubmed_h"] = t.index.map(ph.log_pubmed_h)
    e2 = pd.read_parquet(REPO / "results/analysis/E2/props.parquet").set_index("hsapiens_gene_id")
    for k in ["gtex_tau", "tf", "tf_homeodomain", "hpa_loc_mitochondria", "enzyme", "ohnolog"]:
        t[k] = t.index.map(e2[k])
    # restrict binary/target annotations to genes the source can see: protein-coding genes
    t = t[t.biotype == "protein_coding"].copy()
    t.reset_index().to_parquet(OUT / "props.parquet")

    # paralog pairs (any within-species paralog, Compara 110)
    tmp = OUT / "paralogs_any.tsv"
    subprocess.run(f"zcat {COMPARA} | awk -F'\\t' '$3==\"homo_sapiens\" && $8==\"homo_sapiens\" && $5 ~ /paralog/ "
                   f"{{print $1\"\\t\"$6}}' | sort -u > {tmp}", shell=True, check=True)
    pp = pd.read_csv(tmp, sep="\t", names=["g", "h"])
    pp.to_parquet(OUT / "paralogs_any.parquet")
    tmp.unlink()

    cov = pd.DataFrame({"n_nonmissing": t.notna().sum(), "mean": t.select_dtypes("number").mean()})
    cov.to_csv(OUT / "props_coverage.csv")
    with open(OUT / "props_sources.txt", "w") as fh:
        fh.write(f"chromatin sets: {used}\nHK symbols {len(hk)} mapped {len(hk_g)}\nconserved miR families {len(cons)}\n"
                 f"DoRothEA edges {len(dor)}, CollecTRI edges {len(col)}\n")
    print(cov.to_string())


if __name__ == "__main__":
    main()
