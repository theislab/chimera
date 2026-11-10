"""E6: derive human-vs-mouse lung lists from Travaglini et al. 2020 Nature (doi:10.1038/s41586-020-2922-4).

Usage: python -I derive_travaglini2020.py <tsv_dir> <out.tsv>
  <tsv_dir> = data/analysis/E6/travaglini2020/tsv (xlsx_to_tsv.py output of MOESM9 and MOESM11)

Inputs
  MOESM9  = Supplementary Table 7 "Genes specific to mouse and human in each cluster and lung wide";
            one sheet per homologous cluster pair (title row "human_X vs mouse_Y - R = r"), columns
            gene, mouse_avg_exp, human_avg_exp, avg_logFC (natural log, mouse vs human), pct_mouse, pct_human,
            p_val, p_val_adj, log_p_val, enriched. The published table already applies the paper's rule
            (|avg_logFC| >= ln 20, i.e. >= 20-fold, MAST p < 0.05); every row is kept.
  MOESM11 = Supplementary Table 9 "Evolutionary and functional classes of genes" (sheet named Table_S8);
            per gene Evo_Type (0,1,2,3,NA), Gene_Class, Conserved_Num, Human_Spec_Num, Mouse_Spec_Num.
Rules
  Table 7: set species_divergent; direction up_in_mouse if avg_logFC > 0 else up_in_human.
  Table 9: Evo_Type 0 -> conserved_control/conserved; 1 -> evo_type1_gain_loss; 2 -> evo_type2_expansion;
           3 -> evo_type3_switch. Types 2 and 3: direction divergent (type 2 side is not recoverable: the table
           has no gene with Mouse_Spec_Num > 0 and Human_Spec_Num == 0, although Supp Table 8 reports ~1,095
           'expansion in mouse' genes). Type 1: up_in_human if Human_Spec_Num > 0 and Mouse_Spec_Num == 0,
           up_in_mouse if the reverse. Evo_Type NA (gene not expressed in lung) is dropped.
"""
import csv
import glob
import os
import re
import sys

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]
SRC = "travaglini2020"
CIRC_NOTE = ("mouse side = Tabula Muris Senis lung/blood (+522 extra SS2 cells); human side = Travaglini's own "
             "Stanford lung/blood donors (not Tabula Sapiens)")


def gene_note(g):
    if g.startswith("Hs(") or g.startswith("Mm("):
        return "collapsed paralog/homology group in the paper (Hs()/Mm() = duplication collapsed); not 1:1"
    return ""


def main(tsv_dir, out):
    rows = []
    files = sorted(glob.glob(os.path.join(tsv_dir, "41586_2020_2922_MOESM9_ESM__Cluster_*.tsv")),
                   key=lambda p: int(re.search(r"Cluster_(\d+)", p).group(1)))
    for f in files:
        sheet = re.search(r"(Cluster_\d+)", f).group(1)
        with open(f, newline="") as fh:
            lines = list(csv.reader(fh, delimiter="\t"))
        title = lines[0][0]
        m = re.match(r"(.*) - R = ([0-9.]+)", title)
        pair, r = m.group(1), m.group(2)
        header = lines[1]
        if pair == "Human vs Mouse":
            cell_type = "lung (all SS2 cells, lung-wide)"
        else:
            cell_type = pair
        for vals in lines[2:]:
            if not vals or not vals[0]:
                continue
            d = dict(zip(header, vals + [""] * (len(header) - len(vals))))
            lfc = float(d["avg_logFC"])
            note = "; ".join(x for x in [gene_note(d["gene"]),
                                         "Supp Table 7 sheet %s (%s, Pearson R=%s)" % (sheet, title, r),
                                         "enriched(cluster marker in a species)=%s" % (d.get("enriched", "") or "0"),
                                         CIRC_NOTE] if x)
            rows.append([SRC, "species_divergent", d["gene"], "human", "human_vs_mouse", cell_type,
                         "up_in_mouse" if lfc > 0 else "up_in_human",
                         "avg_logFC(ln, mouse/human)=%s; mouse_avg_exp=%s; human_avg_exp=%s; pct_mouse=%s; pct_human=%s"
                         % (d["avg_logFC"], d["mouse_avg_exp"], d["human_avg_exp"], d["pct_mouse"], d["pct_human"]),
                         "MAST p_val=%s; p_val_adj=%s; paper rule >=20-fold (|ln FC|>=3.0) and p<0.05 (SS2 cells)"
                         % (d["p_val"], d["p_val_adj"]),
                         "yes", note])
    # Table 9
    f9 = os.path.join(tsv_dir, "41586_2020_2922_MOESM11_ESM__Table_S8.tsv")
    setmap = {"0": "conserved_control", "1": "evo_type1_gain_loss", "2": "evo_type2_expansion",
              "3": "evo_type3_switch"}
    n_na = 0
    with open(f9, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        header = next(rd)
        for vals in rd:
            d = dict(zip(header, vals))
            gene, et = vals[0], d["Evo_Type"]
            if et not in setmap:
                n_na += 1
                continue
            h, mo = int(float(d["Human_Spec_Num"] or 0)), int(float(d["Mouse_Spec_Num"] or 0))
            if et == "0":
                direction = "conserved"
            elif et in ("2", "3"):
                # type 2: the published table has no gene with Mouse_Spec_Num>0 and Human_Spec_Num==0, although
                # Supp Table 8 reports ~1,095 'expansion in mouse' genes; side not recoverable -> divergent
                direction = "divergent"
            elif h > 0 and mo == 0:
                direction = "up_in_human"
            elif mo > 0 and h == 0:
                direction = "up_in_mouse"
            else:
                direction = "divergent"
            note = "; ".join(x for x in [gene_note(gene), "Supp Table 9 (MOESM11, sheet Table_S8); Gene_Class=%s"
                                         % d["Gene_Class"], CIRC_NOTE] if x)
            rows.append([SRC, setmap[et], gene, "human", "human_vs_mouse",
                         "lung (gene-level pattern across homologous SS2 clusters)", direction,
                         "Evo_Type=%s; Conserved_Num=%s; Human_Spec_Num=%s; Mouse_Spec_Num=%s"
                         % (et, d["Conserved_Num"], d["Human_Spec_Num"], d["Mouse_Spec_Num"]),
                         "binarised expression per cell type (median-based cutoff, Methods); paper's Evo_Type",
                         "yes", note])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows)
    print("rows", len(rows), "dropped Evo_Type NA", n_na, file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
