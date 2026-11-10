"""Derive literature gene lists from Benner et al. 2014 (BMC Genomics 15:620), mouse vs human beta cells.

Usage: python -I derive_benner2014.py <data_dir> <out_tsv>
  <data_dir>: data/analysis/E6/benner2014 (holds converted/ from xlsx_to_tsv.py)
  <out_tsv>:  analysis/tasks/E6_known_differences/lists/benner2014.tsv

Sets (mechanical, the paper's own definitions; see benner2014.md):
  mouse_enriched / human_enriched  Additional file 9: genes with adj. p < 1e-7 (FDR < 0.1%) AND > 10-fold higher in
                     beta cells of one species; status 'unique' (other species RPKM < 1) or 'enriched' (other species
                     RPKM > 1) kept in `note`.
  conserved_control  Additional file 8: the paper's "common beta cell transcriptome" (RPKM > 1 in beta cells of both
                     species), minus any gene that is also in Additional file 9 (those are species-enriched by the
                     paper's own rule).
Symbols are as published (mouse symbols; file 8 gives 'Ins2/INS' for insulin).
"""
import re
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SRC = "benner2014"


def one(d, pattern):
    hits = sorted(Path(d).glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def read(conv, pattern, ncol):
    t = pd.read_csv(one(conv, pattern), sep="\t", header=None, skiprows=4, dtype=str, keep_default_na=False)
    assert t.shape[1] == ncol, t.shape
    return t


def flag(sym):
    return "symbol looks Excel date-converted (as published)" if re.fullmatch(r"\d+(\.\d+)?", sym) else ""


def main(data_dir, out):
    conv = Path(data_dir) / "converted"
    f9 = read(conv, "*MOESM9_ESM__*.tsv", 8)
    f9.columns = ["gene", "acc", "rpkm_mouse", "rpkm_human", "log_fold", "status", "enriched_in", "padj"]
    f8 = read(conv, "*MOESM8_ESM__*.tsv", 5)
    f8.columns = ["gene", "acc", "rpkm_mouse", "rpkm_human", "padj"]
    # paper counts: mouse 725 (569 unique + 156 enriched), human 815 (666 unique + 149 enriched)
    c = f9.groupby(["enriched_in", "status"]).size().to_dict()
    assert c == {("human", "enriched"): 149, ("human", "unique"): 666,
                 ("mouse", "enriched"): 156, ("mouse", "unique"): 569}, c
    rows, viol = [], {"padj>=1e-7": [], "fold<=10": [], "status_vs_RPKM1": []}
    for r in f9.itertuples(index=False):
        pad, rm, rh = float(r.padj), float(r.rpkm_mouse), float(r.rpkm_human)
        # the paper's rule, checked on the table values; rows are kept as published, departures are reported
        hi, lo = (rm, rh) if r.enriched_in == "mouse" else (rh, rm)
        if not pad < 1e-7:
            viol["padj>=1e-7"].append(r.gene)
        if not hi > 10 * lo:
            viol["fold<=10"].append(f"{r.gene}({hi / lo if lo else float('inf'):.2f})")
        if (lo < 1) != (r.status == "unique"):
            viol["status_vs_RPKM1"].append(r.gene)
        nt = [f"status={r.status}", f"accession={r.acc}"]
        if flag(r.gene):
            nt.append(flag(r.gene))
        rows.append(dict(
            set=f"{r.enriched_in}_enriched", gene_symbol=r.gene, symbol_species="mouse", contrast="human_vs_mouse",
            cell_type="beta", direction=f"up_in_{r.enriched_in}",
            effect=f"RPKM_mouse_beta={rm:.3f};RPKM_human_beta={rh:.3f};log_fold={float(r.log_fold):.3f};padj={r.padj}",
            evidence="AddFile9: adj.p<1e-7 (FDR<0.1%) and >10-fold in beta cells of one species; unique = other "
                     "species RPKM<1, enriched = other species RPKM>1 (Results, Fig 5A-B)",
            note=";".join(nt)))
    in9 = set(f9.gene)
    dropped = 0
    for r in f8.itertuples(index=False):
        if r.gene in in9:
            dropped += 1
            continue
        nt = [f"accession={r.acc}"]
        if r.gene == "Ins2/INS":
            nt.append("published as Ins2/INS (mouse/human)")
        if flag(r.gene):
            nt.append(flag(r.gene))
        rows.append(dict(
            set="conserved_control", gene_symbol=r.gene, symbol_species="mouse", contrast="human_vs_mouse",
            cell_type="beta", direction="conserved",
            effect=f"RPKM_mouse_beta={float(r.rpkm_mouse):.3f};RPKM_human_beta={float(r.rpkm_human):.3f};padj={r.padj}",
            evidence="AddFile8: common beta cell transcriptome, RPKM>1 in beta cells of both species; genes also in "
                     "AddFile9 removed",
            note=";".join(nt)))
    df = pd.DataFrame(rows)
    df.insert(0, "source_id", SRC)
    df["circular"] = "no"
    df = df[COLS]
    dup = df.duplicated(["set", "gene_symbol", "contrast", "cell_type"], keep=False)
    print("duplicated (set,gene) rows kept as published:", df[dup].gene_symbol.tolist())
    df.to_csv(out, sep="\t", index=False)
    print("file9 rows departing from the stated rule (kept as published):",
          {k: (len(v), v[:15]) for k, v in viol.items()})
    print(f"file8 rows={len(f8)} (paper text: 9905); removed because in file9: {dropped}")
    print(df.groupby("set").agg(rows=("gene_symbol", "size"), genes=("gene_symbol", "nunique")).to_string())
    print(df[df.set != "conserved_control"].groupby(["set"]).note.apply(
        lambda s: s.str.extract(r"status=(\w+)")[0].value_counts().to_dict()).to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
