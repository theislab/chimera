"""Derive literature gene lists from Bakken et al. 2021 (Nature 598:111), M1 human / marmoset / mouse.
CIRCULAR with the Chimera M1 cohorts (same data): contrast arm only.

Usage: python -I derive_bakken2021.py <extracted_dir> <converted_dir> <out_tsv>
  <extracted_dir>: data/analysis/E6/bakken2021/extracted (Supplementary Tables from MOESM4 zip)
  <converted_dir>: data/analysis/E6/bakken2021/converted (ST20-22 via xlsx_to_tsv.py)

Sources and the paper's own rules (see bakken2021.md):
  ST7  GABAergic subclass markers, ST10 glutamatergic subclass markers: Seurat FindAllMarkers ROC test,
       each subclass vs all other nuclei of the same class within each species (AUC > 0.7, min.pct 0.1);
       `conservation` = species in which the gene is a marker.
  ST20 chandelier vs basket (Pvalb) markers per species (ROC test), 1 = enriched in ChC.
  ST21 L5 ET vs L5 IT markers per species (ROC test), 1 = enriched in L5 ET (human, macaque, marmoset, mouse).
  ST22 131 genes whose L5 ET vs IT log fold change decreases human > macaque > marmoset > mouse, human logFC >= 0.5.

Set from marker status in (human, marmoset, mouse):
  111 conserved_control                 direction conserved
  110 primate_not_mouse_marker          up_in_human (marker in human and marmoset, not mouse)
  100 human_only_marker                 up_in_human
  101 human_mouse_not_marmoset_marker   conserved   (for the human-mouse contrast)
  011 marmoset_mouse_not_human_marker   up_in_mouse
  001 mouse_only_marker                 up_in_mouse
  010 marmoset_only_marker              divergent   (contrast marmoset vs human+mouse)
"direction" here = within-class specificity (marker status) differs, not absolute expression level.
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SRC = "bakken2021"
SETS = {(1, 1, 1): ("conserved_control", "conserved"),
        (1, 1, 0): ("primate_not_mouse_marker", "up_in_human"),
        (1, 0, 0): ("human_only_marker", "up_in_human"),
        (1, 0, 1): ("human_mouse_not_marmoset_marker", "conserved"),
        (0, 1, 1): ("marmoset_mouse_not_human_marker", "up_in_mouse"),
        (0, 0, 1): ("mouse_only_marker", "up_in_mouse"),
        (0, 1, 0): ("marmoset_only_marker", "divergent")}
CONSERVATION = {"human": (1, 0, 0), "marmoset": (0, 1, 0), "mouse": (0, 0, 1),
                "human_marmoset": (1, 1, 0), "human_mouse": (1, 0, 1), "marmoset_mouse": (0, 1, 1),
                "human_marmoset_mouse": (1, 1, 1)}


def contrast_for(key):
    return "marmoset_vs_human_mouse_M1" if key == (0, 1, 0) else "human_vs_mouse_M1(+marmoset)"


def subclass_markers(path, klass, table):
    d = pd.read_csv(path)
    assert set(d.conservation) <= set(CONSERVATION), set(d.conservation)
    assert not d.duplicated(["cluster", "gene"]).any()
    rows = []
    for _, r in d.iterrows():
        key = CONSERVATION[r.conservation]
        s, direction = SETS[key]
        eff = ";".join(f"{sp}_AUC={r[sp + '_AUC']}" for sp in ("human", "marmoset", "mouse"))
        eff += ";" + ";".join(f"{sp}_avg.1={r[sp + '_avg.1']:.3f};{sp}_avg.2={r[sp + '_avg.2']:.3f}"
                              for sp in ("human", "mouse"))
        rows.append(dict(set=s, gene_symbol=r.gene, symbol_species="human", contrast=contrast_for(key),
                         cell_type=r.cluster, direction=direction, effect=eff,
                         evidence=f"{table} Seurat ROC subclass vs rest of {klass} within species, AUC>0.7",
                         note=f"conservation={r.conservation};class={klass}"))
    return rows, d


def binary_table(path, table, cell_type, gene_col, extra_species=()):
    d = pd.read_csv(path, sep="\t")
    d = d.rename(columns={gene_col: "gene"})
    assert not d.gene.duplicated().any(), table
    rows = []
    for r in d.itertuples(index=False):
        key = (int(r.Human), int(r.Marmoset), int(r.Mouse))
        pattern = "_".join(sp for sp in ["Human"] + list(extra_species) + ["Marmoset", "Mouse"] if int(getattr(r, sp)) == 1)
        if key == (0, 0, 0):  # only possible with macaque-only genes in ST21
            s, direction, contrast = "macaque_only_marker", "divergent", "macaque_vs_human_marmoset_mouse_M1"
        else:
            s, direction = SETS[key]
            contrast = contrast_for(key)
        rows.append(dict(set=s, gene_symbol=r.gene, symbol_species="human", contrast=contrast,
                         cell_type=cell_type, direction=direction, effect=f"marker_in={pattern}",
                         evidence=f"{table} Seurat ROC test per species (binary enriched flag)",
                         note=f"marker_in={pattern}"))
    return rows, d


def gradient(path):
    d = pd.read_csv(path, sep="\t")
    d = d.rename(columns={d.columns[0]: "gene"})
    assert len(d) == 131 and not d.gene.duplicated().any()
    # verify the paper's rule holds in the table
    mono = (d.human >= d.macaque) & (d.macaque >= d.marmoset) & (d.marmoset >= d.mouse)
    rows = []
    for r in d.itertuples(index=False):
        rows.append(dict(set="primate_gradient_L5ET_marker", gene_symbol=r.gene, symbol_species="human",
                         contrast="human>macaque>marmoset>mouse_M1", cell_type="L5 ET (vs L5 IT)",
                         direction="up_in_human",
                         effect=f"logFC_ET_vs_IT human={r.human:.3f};macaque={r.macaque:.3f};marmoset={r.marmoset:.3f};mouse={r.mouse:.3f}",
                         evidence="ST22 decreasing ET-vs-IT logFC with distance from human, human logFC>=0.5",
                         note="specificity (ET vs IT) gradient, not absolute level"))
    return rows, d, int(mono.sum())


def main(ext, conv, out):
    ext, conv = Path(ext), Path(conv)
    r7, d7 = subclass_markers(ext / "Supplementary Table 7.csv", "GABAergic", "ST7")
    r10, d10 = subclass_markers(ext / "Supplementary Table 10.csv", "Glutamatergic", "ST10")
    r20, d20 = binary_table(conv / "Supplementary Table 20__Sheet1.tsv", "ST20", "Chandelier (vs Pvalb basket)", "genes")
    r21, d21 = binary_table(conv / "Supplementary Table 21__Sheet1.tsv", "ST21", "L5 ET (vs L5 IT)", "gene",
                            extra_species=("Macaque",))
    r22, d22, nmono = gradient(conv / "Supplementary Table 22__Sheet1.tsv")
    df = pd.DataFrame(r7 + r10 + r20 + r21 + r22)
    df["source_id"], df["circular"] = SRC, "yes"
    df = df[COLS]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    print("ST22 rows monotone h>=mac>=mar>=mus:", nmono, "/", len(d22), "; min human logFC", round(d22.human.min(), 3))
    print("ST21 macaque-only rows:", int(((d21.Human == 0) & (d21.Marmoset == 0) & (d21.Mouse == 0)).sum()))
    df["table"] = df.evidence.str.split().str[0]
    print(df.groupby(["table", "set"]).agg(rows=("gene_symbol", "size"), unique_genes=("gene_symbol", "nunique")).to_string())
    print(df.groupby("set").gene_symbol.nunique().to_string())
    for t in ("ST7", "ST10"):
        x = df[df.table == t]
        print(t, "\n", pd.crosstab(x.cell_type, x.set).to_string())


if __name__ == "__main__":
    main(*sys.argv[1:4])
