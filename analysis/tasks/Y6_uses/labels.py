"""Use 1: labels from the parsed Ensembl fields, rules.py USE 1. No count or readout enters. Run with python -I.

    python -I analysis/tasks/Y6_uses/labels.py
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "results" / "analysis" / "Y6" / "annot"
SAME_98_110 = ("mmulatta", "sscrofa", "drerio", "mmurinus")
Y6 = ROOT / "data" / "analysis" / "Y6"
OLD_GTF = {"mmulatta": Y6 / "gtfold/Macaca_mulatta.Mmul_8.0.1.97.gtf.gz", "sscrofa": Y6 / "gtfold/Sus_scrofa.Sscrofa10.2.89.gtf.gz",
           "drerio": Y6 / "gtfold/Danio_rerio.GRCz10.91.gtf.gz", "mmusculus": Y6 / "gtfold/Mus_musculus.GRCm38.84.gtf.gz",
           "mmurinus": Y6 / "gtfold/Microcebus_murinus.Mmur_2.0.90.gtf.gz", "mfascicularis": Y6 / "gtf98/Macaca_fascicularis.Macaca_fascicularis_5.0.98.gtf.gz"}
MOVE_BP, MOVE_CDS = 2000, 0.10


def moved(g: pd.DataFrame, a: int, b: int) -> pd.Series:
    same = (g[f"contig_{a}"] == g[f"contig_{b}"]) & (g[f"strand_{a}"] == g[f"strand_{b}"])
    shift = (g[f"end3_{a}"] - g[f"end3_{b}"]).abs()
    cds = (g[f"cds_len_{a}"] - g[f"cds_len_{b}"]).abs() / g[[f"cds_len_{a}", f"cds_len_{b}"]].max(axis=1)
    out = (~same) | (shift > MOVE_BP) | (cds > MOVE_CDS)
    return out & g[f"in_{a}"] & g[f"in_{b}"]


def gene_ids(path: Path) -> set:
    ids = set()
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = line.split("\t", 3)
            if p[2] == "gene":
                m = re.search(r'gene_id "([^"]+)"', line)
                ids.add(m.group(1))
    return ids


def main():
    g = pd.read_parquet(OUT / "genome_labels.parquet")
    old = {sp: gene_ids(p) for sp, p in OLD_GTF.items()}
    g["in_old"] = [gid in old[sp] for sp, gid in zip(g.species, g.gene_id)]
    L = g[["species", "hsapiens_gene_id", "gene_id"]].copy()
    L["T_lowconf"] = (g.high_conf_110 == 0) | g.homology_type_110.isna()
    L["T_later"] = (~g.in_116) | (g.biotype_116.fillna("") != "protein_coding") | (g.homology_type_116 != "ortholog_one2one")
    L["T_later_retired"] = ~g.in_116
    L["T_later_biotype"] = g.in_116 & (g.biotype_116.fillna("") != "protein_coding")
    L["T_later_orth"] = g.in_116 & (g.homology_type_116 != "ortholog_one2one")
    L["T_newasm"] = ~g.in_old
    L["T_new98"] = ~g.in_98
    L["T_goc"] = (g.goc_110 < 50).where(g.goc_110.notna())
    L["T_wga"] = (g.wga_110 < 50).where(g.wga_110.notna())
    L["T_no3utr"] = (~g.has_utr3_110.astype("boolean")).astype("boolean").where(g.in_110)
    mv = moved(g, 110, 116)
    mv98 = moved(g, 98, 110).where(g.species.isin(SAME_98_110), False)
    L["T_moved"] = mv | mv98
    L["T_tsl"] = (g.best_tsl_110 > 1).where(g.species == "mmusculus")
    L["goc"], L["wga"], L["high_conf"], L["homology_type_110"], L["homology_type_116"] = g.goc_110, g.wga_110, g.high_conf_110, g.homology_type_110, g.homology_type_116
    L["biotype_110"], L["biotype_116"], L["in_98"], L["in_110"], L["in_116"] = g.biotype_110, g.biotype_116, g.in_98, g.in_110, g.in_116
    L["contig_110"], L["start_110"], L["end_110"], L["strand_110"], L["n_tx_110"], L["best_tsl_110"], L["has_utr3_110"] = g.contig_110, g.start_110, g.end_110, g.strand_110, g.n_tx_110, g.best_tsl_110, g.has_utr3_110
    L["human_partners_116"] = g.human_partners_116
    L.to_parquet(OUT / "labels.parquet")
    tp = pd.read_parquet(OUT / "table_pairs.parquet")
    m = tp.merge(L, on=["species", "hsapiens_gene_id", "gene_id"], how="left")
    print("not protein_coding in 110, per table (left out of use 1):")
    print(m[m.biotype_110 != "protein_coding"].groupby(["table", "species"]).biotype_110.value_counts().unstack(fill_value=0))
    m = m[m.biotype_110 == "protein_coding"]
    labs = ["T_lowconf", "T_later", "T_later_retired", "T_later_biotype", "T_later_orth", "T_newasm", "T_new98", "T_goc", "T_wga", "T_no3utr", "T_moved", "T_tsl"]
    prev = m.groupby(["table", "species"])[labs].agg(lambda x: pd.to_numeric(x, errors="coerce").astype(float).mean()).round(3)
    n = m.groupby(["table", "species"])[labs].agg(lambda x: pd.to_numeric(x, errors="coerce").astype(float).sum()).round(0)
    prev.insert(0, "pairs", m.groupby(["table", "species"]).size())
    prev.to_csv(OUT / "label_prevalence_tables.csv")
    pd.set_option("display.width", 220, "display.max_columns", 30)
    print(prev)
    print(n[["T_lowconf", "T_later", "T_newasm", "T_new98", "T_no3utr", "T_moved"]])
    print("goc non-null", m.groupby("species").goc.apply(lambda x: x.notna().mean()).round(3).to_dict(), "wga", m.groupby("species").wga.apply(lambda x: x.notna().mean()).round(3).to_dict())


if __name__ == "__main__":
    main()
