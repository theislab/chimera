"""E6: derive primate-vs-mouse cell-type lists from Ezran et al. 2025 Nature 644:173 (Tabula Microcebus atlas,
doi:10.1038/s41586-025-09113-9), Supplementary Table 6 "DEGs across species".

Usage: python -I derive_ezran2025a_tmc.py <tsv_dir> <out.tsv>
  <tsv_dir> = data/analysis/E6/ezran2025a_tmc/tsv (xlsx_to_tsv.py output of Table 6)

Paper's rule (Methods, "Identification of genes with primate-selective expression for each cell type"):
  primate_high: cell-type mean expression > 0.5 in human and lemur, and >5-fold higher with Wilcoxon P < 1e-5 in
                both human and lemur vs the orthologous mouse cell type  -> set primate_enriched, up_in_human
  primate_low:  mean > 0.5 in mouse and >5-fold lower with P < 1e-5 in human and lemur vs mouse
                -> set primate_depleted, up_in_mouse
  conserved:    mean > 0.5 in all three species and >5-fold higher with P < 1e-5 in this cell type vs other cell
                types -> set conserved_control, conserved
All rows kept (all tissue groups: Lung, Marrow & Spleen, Liver, Muscle, Testes). Data = Tabula Sapiens, Tabula
Microcebus, Tabula Muris Senis (testes: other datasets) -> circular yes (testes: no, not in our cohorts).
A fold change of 65535 is the table's value as published (appears to encode division by zero).
"""
import csv
import os
import sys

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]
SRC = "ezran2025a_tmc"
STEM = "2022-08-12359B-Table 6 DEGs across species__%s.tsv"


def main(tsv_dir, out):
    rows = []
    for sheet, set_, direction in [("primate_high", "primate_enriched", "up_in_human"),
                                   ("primate_low", "primate_depleted", "up_in_mouse"),
                                   ("conserved", "conserved_control", "conserved")]:
        with open(os.path.join(tsv_dir, STEM % sheet), newline="") as fh:
            rd = csv.reader(fh, delimiter="\t")
            header = next(rd)
            for vals in rd:
                if not vals or not vals[3]:
                    continue
                d = dict(zip(header, vals))
                tissue = d["tissue group"]
                cell_type = "%s | %s | %s" % (tissue, d["compartment"], d["free annotation (unified)"])
                if sheet == "conserved":
                    contrast = "human_lemur_mouse_celltype_vs_rest"
                    effect = "FC(this cell type vs others)=%s" % d["fold-change (this cell type vs other cell types)"]
                    evidence = ("Wilcoxon p=%s; paper rule mean>0.5 in all 3 species, >5-fold, P<1e-5 vs other cell types"
                                % d["p-val (this cell type vs other cell types)"])
                else:
                    contrast = "primate(human+lemur)_vs_mouse"
                    effect = "FC(human/mouse)=%s; FC(lemur/mouse)=%s" % (d["fold-change (human vs mouse)"],
                                                                        d["fold-change (lemur vs mouse)"])
                    evidence = ("Wilcoxon p(human vs mouse)=%s; p(lemur vs mouse)=%s; paper rule >5-fold and P<1e-5 in "
                                "both primates vs mouse, mean>0.5 in the higher species"
                                % (d["p-val (human vs mouse)"], d["p-val (lemur vs mouse)"]))
                circ = "no" if tissue == "Testes" else "yes"
                note = ("Supp Table 6 sheet %s; data: Tabula Sapiens / Tabula Microcebus / Tabula Muris Senis%s"
                        % (sheet, " (testes: Shami/Ernst datasets, not our cohorts)" if tissue == "Testes" else ""))
                rows.append([SRC, set_, d["Human_NCBI_GeneName"], "human", contrast, cell_type, direction,
                             effect, evidence, circ, note])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows)
    print("rows", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
