"""Y2: checks of the annotation tables only (no readout): bigWig reader against the UCSC API; known genes against their annotation."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from revision.tasks.Y2_functional import bw as BW  # noqa: E402

DL = REPO / "data/revision/Y2"
OUT = REPO / "results/revision/Y2"


def main():
    f = BW.BigWig(DL / "zoonomia/cactus241way.phyloP.bw")
    rng = np.random.default_rng(1)
    rows = []
    for chrom, start in (("chr1", 959256), ("chr7", 117559590), ("chr17", 43044295), ("chrX", 31100000), ("chr22", 21000000)):
        url = f"https://api.genome.ucsc.edu/getData/track?genome=hg38;track=phyloP241wayBW;chrom={chrom};start={start};end={start + 60}"
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                j = json.load(r)
            vals = j["phyloP241wayBW"][chrom] if isinstance(j["phyloP241wayBW"], dict) else j["phyloP241wayBW"]
            api = np.full(60, np.nan)
            for it in vals:
                for p in range(max(it["start"], start), min(it["end"], start + 60)):
                    api[p - start] = it["value"]
        except Exception as e:  # noqa: BLE001
            api = None
            print("API failed", url, repr(e)[:120], flush=True)
        mine = f.fill(chrom, np.array([start]), np.array([start + 60]))[0]
        if api is not None:
            ok = np.isfinite(api) & np.isfinite(mine)
            rows.append(dict(chrom=chrom, start=start, n=int(ok.sum()), max_abs_diff=float(np.abs(api[ok] - mine[ok]).max()) if ok.any() else np.nan))
    print(pd.DataFrame(rows).to_string(), flush=True)
    ph = pd.read_parquet(OUT / "annot/pheno.parquet")
    reg = pd.read_parquet(OUT / "annot/regulatory.parquet")
    hg = pd.read_csv(REPO / "data/revision/x3_annotations/hgnc_complete_set.txt", sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"]).dropna().set_index("symbol")["ensembl_gene_id"]
    genes = ["FAH", "PAH", "ASL", "CPS1", "DDX4", "PRM1", "SYCP3", "DAZL", "GATA1", "PKD1", "WT1", "RHO", "PAX6", "MYC", "RPL3", "TP53", "AKT1"]
    cols = ["M_hepmet", "H_hepmet", "M_repro", "H_repro", "M_male", "H_male", "M_heme", "H_heme", "M_renal", "H_renal", "M_eye", "H_eye", "mouse_lethal", "lethal_known", "depmap_common", "pubmed_h", "pubmed_m"]
    t = ph.reindex(hg.reindex(genes).values)[cols].astype(float)
    t.index = genes
    print(t.round(0).astype("Int64").to_string(), flush=True)
    r = reg.reindex(hg.reindex(genes).values)[["phylop_promoter", "phylop_utr5", "phylop_utr3", "phylop_cds", "gc_promoter", "promoter_cgi", "tata", "abc_links_mean"]]
    r.index = genes
    print(r.round(2).to_string(), flush=True)
    print(ph[[c for c in ph.columns if c.startswith(("M_", "H_"))]].sum().to_dict())


if __name__ == "__main__":
    main()
