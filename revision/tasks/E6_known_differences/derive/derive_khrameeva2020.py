"""E6 literature list: Khrameeva et al. 2020 Genome Research 30:776 (snRNA-seq of anterior cingulate cortex,
caudate nucleus and cerebellum; human, chimpanzee, bonobo, rhesus macaque).

Usage:
    python -I derive_khrameeva2020.py <data_dir> <out_tsv>
    data_dir = chimera/data/revision/E6/khrameeva2020 (needs converted/ and hgnc_complete_set.txt)

Source: Supplemental Table S6, sheet "human-specific genes" (Seurat FindMarkers per broad cell type:
human nuclei vs chimpanzee + bonobo + macaque nuclei combined; columns p_val, avg_logFC (natural log),
pct.H, pct.O, p_val_adj, cell_type).
Rule (paper's own, Methods "Human-specific expression differences in snRNA-seq data set"): "fold-change
threshold = 2; Wilcoxon rank-sum test; Bonferroni adjusted P-value < 0.05". The sheet also holds rows with
p_val_adj >= 0.05; those are dropped (count reported). Every kept row already has |avg_logFC| >= ln 2.
Direction from the sign of avg_logFC (positive = higher in human; agrees with pct.H > pct.O in 99.8% of rows).
Genes are published as Ensembl gene IDs; gene_symbol keeps the published ID and the HGNC symbol
(hgnc_complete_set.txt, column ensembl_gene_id) goes into note as hgnc_symbol=...
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type", "direction",
        "effect", "evidence", "circular", "note"]
REGION = {"AC": "anterior_cingulate_cortex_BA24", "CN": "caudate_nucleus", "CB": "cerebellum"}


def main(data_dir, out_tsv):
    d = Path(data_dir)
    t = pd.read_csv(d / "converted/supp_gr.256958.119_Supplemental_Table_S6__human-specific_genes.tsv",
                    sep="\t")
    t = t.rename(columns={t.columns[0]: "ensembl_id"})
    n_all = len(t)
    keep = t[t["p_val_adj"] < 0.05].copy()
    print(f"rows in sheet {n_all}; kept p_val_adj<0.05: {len(keep)}; dropped {n_all - len(keep)}")
    assert (keep["avg_logFC"].abs() >= 0.693).all()

    hg = pd.read_csv(d / "hgnc_complete_set.txt", sep="\t", dtype=str, usecols=["symbol", "ensembl_gene_id"])
    hg = hg.dropna().drop_duplicates("ensembl_gene_id")
    sym = dict(zip(hg["ensembl_gene_id"], hg["symbol"]))
    keep["hgnc"] = keep["ensembl_id"].map(sym).fillna("")
    reg = keep["cell_type"].str.split(" ").str[0].map(REGION)

    out = pd.DataFrame({
        "source_id": "khrameeva2020",
        "set": "human_specific",
        "gene_symbol": keep["ensembl_id"],
        "symbol_species": "human",
        "contrast": "human_vs_chimpanzee_bonobo_macaque",
        "cell_type": keep["cell_type"],
        "direction": keep["avg_logFC"].map(lambda x: "up_in_human" if x > 0 else "down_in_human"),
        "effect": keep["avg_logFC"].map(lambda x: f"lnFC={x:.4g}"),
        "evidence": "TableS6 Seurat FindMarkers Wilcoxon human vs 3 NHP pooled; Bonferroni P<0.05, FC>2",
        "circular": "no",
        "note": ("hgnc_symbol=" + keep["hgnc"] + ";region=" + reg + ";p_val_adj="
                 + keep["p_val_adj"].map(lambda x: f"{x:.3g}") + ";pct.H="
                 + keep["pct.H"].map(lambda x: f"{x:.3g}") + ";pct.O=" + keep["pct.O"].map(lambda x: f"{x:.3g}")),
    })[COLS]
    out.to_csv(out_tsv, sep="\t", index=False)
    print(f"unique ensembl ids {out.gene_symbol.nunique()}; unmapped to HGNC {(keep['hgnc'] == '').sum()} rows / "
          f"{keep.loc[keep.hgnc == '', 'ensembl_id'].nunique()} ids")
    for r in ["AC", "CN", "CB"]:
        sub = out[out.cell_type.str.startswith(r + " ")]
        print(r, "rows", len(sub), "unique genes", sub.gene_symbol.nunique())
    print(out.groupby(["cell_type", "direction"]).size().unstack().to_string())


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
