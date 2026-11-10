"""Derive species-specific cell-type genes from Wang et al. 2020 (Cell Res 30:1109), human vs mouse FETAL liver.

Usage: python -I derive_wang2020_fetal_liver.py <tsv_dir> <out_tsv>

<tsv_dir> holds the per-sheet TSVs of Supplementary Table S3 (41422_2020_378_MOESM12_ESM.xlsx, converted with
xlsx_to_tsv.py): "Genes-to-eigengenes correlations for human and mouse <cell type>", columns Gene,
Correlation_to_human_eigengene, Correlation_to_mouse_eigengene, Species_specific (human/mouse/none).
Paper's rule (Methods): "Genes specifically correlated with the eigengene of one species (Pearson correlations > 0.5) but
not correlated with the other (the differences between Pearson correlations of two species > 0.5) were regarded as
having species-specific expression patterns." We take the published Species_specific column (it matches the rule exactly):
  human -> human_enriched (up_in_human), mouse -> mouse_enriched (up_in_mouse); 'none' rows are not exported.
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]


def main(tsv_dir, out_tsv):
    out = []
    for f in sorted(Path(tsv_dir).glob("41422_2020_378_MOESM12_ESM__*.tsv")):
        lines = [ln.rstrip("\n").split("\t") for ln in open(f, encoding="utf-8")]
        raw = pd.DataFrame([ln + [""] * (4 - len(ln)) for ln in lines]).iloc[:, :4]
        hdr = raw.index[raw[0] == "Gene"][0]
        title = raw.iloc[hdr - 1, 0]
        ct = title.split("for human and mouse", 1)[1].strip().rstrip(".")
        df = raw.iloc[hdr + 1:, :4]
        df.columns = ["Gene", "h", "m", "sp"]
        for _, r in df.iterrows():
            if r["sp"] not in ("human", "mouse"):
                continue
            hum = r["sp"] == "human"
            out.append(dict(
                source_id="wang2020_fetal_liver", set="human_enriched" if hum else "mouse_enriched",
                gene_symbol=r["Gene"], symbol_species="mouse-style symbol of a 1:1 Ensembl orthologue pair",
                contrast="human_vs_mouse", cell_type=f"fetal liver {ct}",
                direction="up_in_human" if hum else "up_in_mouse",
                effect=f"r_human_eigengene={float(r['h']):.3f}; r_mouse_eigengene={float(r['m']):.3f}",
                evidence="Table S3 Species_specific; rule r>0.5 in one species and r difference >0.5",
                circular="no",
                note="FETAL liver (human W5-W19, mouse E11.0-E17.5); species-specific co-expression with the cell-type "
                     "eigengene, not a direct mean-expression contrast"))
    res = pd.DataFrame(out, columns=COLS)
    res.to_csv(out_tsv, sep="\t", index=False)
    print(res.groupby(["cell_type", "set"]).size().to_string())
    print(res.groupby("set").gene_symbol.nunique().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:3])
