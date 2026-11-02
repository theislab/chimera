"""Y2 (EXPLORATORY): regulatory annotation per human gene: phyloP and GC per region, CpG island, TATA, ABC links.

Reads only downloaded files and the Ensembl 110 GTF (run with ``python -I``); no Chimera readout enters.
Regions from the Ensembl canonical transcript: promoter [-1000, +200], 5'UTR, 3'UTR, CDS. Bases in any annotated
CDS of any transcript are removed from the three non-coding regions.
Usage (from chimera/): mamba run --no-capture-output -n chimera_env python -I revision/tasks/Y2_functional/annot_reg.py [--no-phylop]
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from revision.tasks.Y2_functional import bw as BW  # noqa: E402
from revision.tasks.Y2_functional.rules import PHYLOP_CONSTRAINED, PROMOTER, TATA_WINDOW  # noqa: E402

DL = REPO / "data/revision/Y2"
OUT = REPO / "results/revision/Y2/annot"
GTF = REPO / "data/revision/ensembl110/Homo_sapiens.GRCh38.110.gtf.gz"
BIGWIG = DL / "zoonomia/cactus241way.phyloP.bw"
FA = DL / "ucsc/hg38.fa.gz"
ATTR = re.compile(r'(\S+) "([^"]*)"')
CHROMS = [str(i) for i in range(1, 23)] + ["X", "Y"]


def parse_gtf() -> tuple[dict, dict]:
    """regions[gene][region] = list of (start0, end) half-open, on chr; cds_all[chrom] = list of (start0, end)."""
    regions: dict = defaultdict(lambda: dict(chrom=None, strand=None, tss=None, utr5=[], utr3=[], cds=[]))
    cds_all = defaultdict(list)
    with gzip.open(GTF, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            p = line.rstrip("\n").split("\t")
            kind = p[2]
            if kind not in ("transcript", "CDS", "five_prime_utr", "three_prime_utr"):
                continue
            chrom, s, e, strand = p[0], int(p[3]) - 1, int(p[4]), p[6]
            if chrom not in CHROMS:
                continue
            if kind == "CDS":
                cds_all[chrom].append((s, e))
            if 'tag "Ensembl_canonical"' not in p[8]:
                continue
            a = dict(ATTR.findall(p[8]))
            r = regions[a["gene_id"]]
            r["chrom"], r["strand"] = chrom, strand
            if kind == "transcript":
                r["tss"] = s if strand == "+" else e - 1
                r["biotype"] = a.get("transcript_biotype")
            elif kind == "CDS":
                r["cds"].append((s, e))
            elif kind == "five_prime_utr":
                r["utr5"].append((s, e))
            else:
                r["utr3"].append((s, e))
    return dict(regions), cds_all


def promoter_window(tss: int, strand: str) -> tuple[int, int]:
    a, b = PROMOTER
    return (tss + a, tss + b) if strand == "+" else (tss - b + 1, tss - a + 1)


def read_fasta(path: Path, want: set[str]) -> dict[str, bytes]:
    out, cur, buf = {}, None, []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line[0] == ">":
                if cur in want:
                    out[cur] = "".join(buf).upper().encode()
                cur, buf = line[1:].split()[0], []
            elif cur in want:
                buf.append(line.strip())
    if cur in want:
        out[cur] = "".join(buf).upper().encode()
    return out


def seqstats(seq: bytes) -> tuple[float, float]:
    a = np.frombuffer(seq, dtype=np.uint8)
    c, g = (a == 67), (a == 71)
    n = int((c | g | (a == 65) | (a == 84)).sum())
    if n < 10:
        return np.nan, np.nan
    gc = (c.sum() + g.sum()) / n
    cg = int((c[:-1] & g[1:]).sum())
    oe = cg * n / max(int(c.sum()) * int(g.sum()), 1)
    return float(gc), float(oe)


def chrom_job(args):
    chrom, genes, cds_iv, seq, do_phylop = args
    size = len(seq)
    cds_mask = np.zeros(size, dtype=bool)
    for s, e in cds_iv:
        cds_mask[s:e] = True
    arr_seq = np.frombuffer(seq, dtype=np.uint8)
    bwf = BW.BigWig(BIGWIG) if do_phylop else None
    ivs: list[tuple[int, int]] = []
    plan = []
    for g, r in genes.items():
        if r.get("tss") is None:
            continue
        ps, pe = promoter_window(r["tss"], r["strand"])
        ps, pe = max(ps, 0), min(pe, size)
        for name, iv, mask_cds in (("promoter", [(ps, pe)], True), ("utr5", r["utr5"], True), ("utr3", r["utr3"], True), ("cds", r["cds"], False)):
            for s, e in iv:
                if e > s:
                    plan.append((g, name, len(ivs), mask_cds))
                    ivs.append((s, e))
    vals = {}
    if do_phylop and ivs:
        st = np.array([i[0] for i in ivs], dtype=np.int64)
        en = np.array([i[1] for i in ivs], dtype=np.int64)
        vals = bwf.fill("chr" + chrom, st, en)
    acc = defaultdict(lambda: dict(n=0, sum=0.0, nc=0, cov=0, seqs=[]))
    for g, name, k, mask_cds in plan:
        s, e = ivs[k]
        keep = ~cds_mask[s:e] if mask_cds else np.ones(e - s, dtype=bool)
        a = acc[(g, name)]
        a["n"] += int(keep.sum())
        a["seqs"].append(arr_seq[s:e][keep].tobytes())
        if do_phylop:
            v = vals[k][keep]
            okv = np.isfinite(v)
            a["cov"] += int(okv.sum())
            a["sum"] += float(v[okv].sum())
            a["nc"] += int((v[okv] >= PHYLOP_CONSTRAINED).sum())
    rows = []
    for (g, name), a in acc.items():
        gc, oe = seqstats(b"".join(a["seqs"]))
        rows.append(dict(hsapiens_gene_id=g, region=name, length=a["n"], gc=gc, cpg_oe=oe, covered=a["cov"],
                         phylop=a["sum"] / a["cov"] if a["cov"] else np.nan,
                         phylop_fc=a["nc"] / a["cov"] if a["cov"] else np.nan))
    return rows


def tata_table(seqs: dict[str, bytes]) -> pd.DataFrame:
    rows, cur = [], {}
    with open(DL / "epd/Hs_EPDnew.dat") as fh:
        for line in fh:
            if line.startswith("ID "):
                cur = dict(id=line.split()[1])
            elif line.startswith("DR   Ensembl;"):
                cur["gene"] = line.split(";")[1].strip().rstrip(".")
            elif line.startswith("DR   UCSC; hg38"):
                t = line.split()
                cur["chrom"], cur["strand"], cur["tss"] = t[3], t[4], int(t[5])
            elif line.startswith("//"):
                rows.append(cur)
    t = pd.DataFrame(rows).dropna(subset=["gene", "chrom"])
    pat = re.compile(rb"TATA[AT]A[AT]")
    comp = bytes.maketrans(b"ACGT", b"TGCA")
    res = []
    for _, r in t.iterrows():
        s = seqs.get(r.chrom)
        if s is None:
            continue
        a, b = TATA_WINDOW
        # tss is 1-based on hg38; window of motif starts a..b, motif length 7
        if r.strand == "+":
            frag = s[r.tss - 1 + a: r.tss - 1 + b + 7]
        else:
            frag = s[r.tss - 1 - b - 6: r.tss - 1 - a + 1].translate(comp)[::-1]
        res.append(dict(hsapiens_gene_id=r.gene, epd=r.id, tata=bool(pat.search(frag))))
    d = pd.DataFrame(res)
    first = d[d.epd.str.endswith("_1")].drop_duplicates("hsapiens_gene_id")
    print("EPDnew genes", len(first), "TATA share", round(float(first.tata.mean()), 3), flush=True)
    return first.set_index("hsapiens_gene_id")[["tata"]].astype(float)


def abc_table() -> pd.DataFrame:
    h = pd.read_csv(REPO / "data/revision/x3_annotations/hgnc_complete_set.txt", sep="\t", dtype=str,
                    usecols=["symbol", "alias_symbol", "prev_symbol", "ensembl_gene_id"]).dropna(subset=["ensembl_gene_id"])
    primary = dict(zip(h.symbol, h.ensembl_gene_id))
    second = defaultdict(set)
    for _, r in h.iterrows():
        for col in ("alias_symbol", "prev_symbol"):
            if isinstance(r[col], str):
                for x in r[col].split("|"):
                    second[x].add(r.ensembl_gene_id)
    t = pd.read_csv(sorted((DL / "abc").glob("AllPredictions*"))[0], sep="\t", usecols=["chr", "class", "TargetGene", "CellType", "isSelfPromoter"], dtype=str)
    t = t[t["class"].ne("promoter") & t["isSelfPromoter"].ne("True")]
    nb = t.CellType.nunique()

    def ens(s):
        if s in primary:
            return primary[s]
        v = second.get(s)
        return next(iter(v)) if v and len(v) == 1 else None

    t["gene"] = t.TargetGene.map({s: ens(s) for s in t.TargetGene.unique()})
    print("ABC links", len(t), "mapped", int(t.gene.notna().sum()), "biosamples", nb, flush=True)
    t = t.dropna(subset=["gene"])
    out = pd.DataFrame({"abc_links_mean": t.groupby("gene").size() / nb,
                        "abc_chrom": t.groupby("gene")["chr"].agg(lambda x: x.mode().iat[0])})
    liver = t[t.CellType.isin(["liver-ENCODE", "hepatocyte-ENCODE"])]
    out["abc_links_liver"] = liver.groupby("gene").size() / 2
    out["abc_links_liver"] = out["abc_links_liver"].fillna(0.0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-phylop", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    regions, cds_all = parse_gtf()
    print("canonical genes", len(regions), flush=True)
    seqs = read_fasta(FA, {"chr" + c for c in CHROMS})
    print("chromosomes read", len(seqs), flush=True)
    by = defaultdict(dict)
    for g, r in regions.items():
        by[r["chrom"]][g] = r
    jobs = [(c, by[c], cds_all[c], seqs["chr" + c], not a.no_phylop) for c in CHROMS]
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(chrom_job, jobs))
    t = pd.DataFrame([r for x in res for r in x])
    wide = t.pivot(index="hsapiens_gene_id", columns="region", values=["length", "gc", "cpg_oe", "covered", "phylop", "phylop_fc"])
    wide.columns = [f"{v}_{r}" for v, r in wide.columns]
    # regions too short or poorly covered are missing, not zero
    for r in ("promoter", "utr5", "utr3", "cds"):
        bad = wide[f"covered_{r}"].fillna(0) < 20 if not a.no_phylop else wide[f"length_{r}"].fillna(0) < 20
        for v in ("phylop", "phylop_fc", "gc", "cpg_oe"):
            wide.loc[bad, f"{v}_{r}"] = np.nan
    meta = pd.DataFrame({g: dict(chrom=r["chrom"], strand=r["strand"], tss=r.get("tss"), biotype=r.get("biotype")) for g, r in regions.items()}).T
    meta.index.name = "hsapiens_gene_id"
    wide = wide.join(meta)
    # CpG island over the promoter window
    cgi = pd.read_csv(DL / "ucsc/cpgIslandExt.txt.gz", sep="\t", header=None, usecols=[1, 2, 3], names=["chrom", "s", "e"])
    ov = {}
    for c, g in cgi.groupby("chrom"):
        o = np.argsort(g.s.to_numpy())
        ov[c] = (g.s.to_numpy()[o], np.maximum.accumulate(g.e.to_numpy()[o]))
    flags = {}
    for gene, r in regions.items():
        if r.get("tss") is None:
            continue
        s, e = promoter_window(r["tss"], r["strand"])
        c = "chr" + r["chrom"]
        if c not in ov:
            continue
        st, me = ov[c]
        i = np.searchsorted(st, e, side="left")
        flags[gene] = float(i > 0 and me[i - 1] > s)
    wide["promoter_cgi"] = pd.Series(flags).reindex(wide.index)
    wide = wide.join(tata_table(seqs))
    wide = wide.join(abc_table())
    conflict = wide.abc_chrom.notna() & ("chr" + wide.chrom.astype(str) != wide.abc_chrom)
    print("ABC symbol maps on another chromosome, dropped:", int(conflict.sum()), flush=True)
    wide.loc[conflict, ["abc_links_mean", "abc_links_liver"]] = np.nan
    wide[["abc_links_mean", "abc_links_liver"]] = wide[["abc_links_mean", "abc_links_liver"]].where(conflict, wide[["abc_links_mean", "abc_links_liver"]].fillna(0.0))
    for k in ("promoter", "utr5", "utr3", "cds"):
        wide[f"log_length_{k}"] = np.log1p(wide[f"length_{k}"].astype(float))
    for c in wide.columns:
        if c not in ("chrom", "strand", "biotype", "abc_chrom"):
            wide[c] = pd.to_numeric(wide[c], errors="coerce")
    wide.to_parquet(OUT / ("regulatory_nophylop.parquet" if a.no_phylop else "regulatory.parquet"))
    print(wide.shape)
    print(wide[["phylop_promoter", "phylop_utr5", "phylop_utr3", "phylop_cds", "gc_promoter", "promoter_cgi", "tata", "abc_links_mean"]].describe().T.round(3).to_string())


if __name__ == "__main__":
    main()
