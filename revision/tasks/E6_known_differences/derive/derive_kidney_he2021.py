"""Derive human-vs-mouse glomerular cell gene lists from He et al. 2021 Nat Commun 12:2141.

Usage: python -I derive_kidney_he2021.py <data_dir> <out_tsv>
  <data_dir> = data/revision/E6/kidney_he2021 (needs sourcedata/ and tsv/ produced from the
  journal's Source Data zip and Supplementary Data 8-10).

Species-specific sets: the genes shown in the species heatmaps Fig. 4c (MC), 5c (podocyte),
6c (GEC), i.e. Source Data 9/10/11. The paper's criteria (Methods): mean log2-RPKM < 0.1 in one
species and >= 3 in the other; Kruskal-Wallis BH-adjusted P < 0.01; not species-specific in more
than one glomerular cell type. Direction is assigned from the per-cell values in the same file,
with cells mapped to species through Source Data 1/6 (mouse) and 2 (human).
Conserved controls: Supplementary Data 8/9/10 (median log2-RPKM >= 4 in both species).
"""
import sys
from pathlib import Path
import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SID = "kidney_he2021"


def cell_ids(path, idcol):
    df = pd.read_csv(path, sep="\t", comment="#", usecols=[idcol])
    return set(df[idcol].astype(str))


def main(data_dir, out):
    d = Path(data_dir)
    sd = d / "sourcedata"
    mouse = cell_ids(sd / "source_datafile1.txt", "index") | cell_ids(sd / "source_datafile6.txt", "Cell")
    human = cell_ids(sd / "source_datafile2.txt", "index")
    assert not (mouse & human)
    rows = []
    specs = [("source_datafile9.txt", "mesangial cell", "Fig. 4c / Source Data 9"),
             ("source_datafile10.txt", "podocyte", "Fig. 5c / Source Data 10"),
             ("source_datafile11.txt", "glomerular endothelial cell", "Fig. 6c / Source Data 11")]
    for fn, ct, fig in specs:
        df = pd.read_csv(sd / fn, sep="\t", comment="#", index_col=0)
        cols = [str(c) for c in df.columns]
        hc = [c for c in cols if c in human]
        mc = [c for c in cols if c in mouse]
        unk = [c for c in cols if c not in human and c not in mouse]
        if unk:  # cells absent from Source Data 1/2/6 (e.g. plate 308s in Source Data 11): excluded from means
            print(f"NOTE {fn}: {len(unk)} cells not mappable to species, excluded: {sorted(set(c.split('_')[0] for c in unk))}")
        df.columns = cols
        for idx, r in df.iterrows():
            ms, hs = idx.split("_", 1)
            mh, mm = r[hc].mean(), r[mc].mean()
            if mh >= 3 and mm < 0.1:
                direc = "up_in_human"
            elif mm >= 3 and mh < 0.1:
                direc = "up_in_mouse"
            else:
                direc = "up_in_human" if mh > mm else "up_in_mouse"
                print(f"WARN {fn} {idx} does not meet the 3/0.1 rule on per-cell means: h={mh:.2f} m={mm:.2f}")
            rows.append(dict(source_id=SID, set="species_divergent", gene_symbol=hs, symbol_species="human",
                             contrast="human_vs_mouse", cell_type=ct, direction=direc,
                             effect=f"mean_log2RPKM_human={mh:.3f};mean_log2RPKM_mouse={mm:.3f};n_cells_human={len(hc)};n_cells_mouse={len(mc)}",
                             evidence="species-specific per paper: mean log2-RPKM<0.1 in one species and >=3 in other; Kruskal-Wallis BH adj P<0.01; specific to one glomerular cell type",
                             circular="no", note=f"{fig}; mouse symbol {ms}; Smart-seq2 human living-donor biopsies vs C57BL/6J mouse glomeruli"))
    supp = [("41467_2021_22331_MOESM11_ESM__Supplementary_Data_8.tsv", "mesangial cell", "Supplementary Data 8"),
            ("41467_2021_22331_MOESM12_ESM__Supplementary_Data_9.tsv", "podocyte", "Supplementary Data 9"),
            ("41467_2021_22331_MOESM13_ESM__Supplementary_Data_10.tsv", "glomerular endothelial cell", "Supplementary Data 10")]
    for fn, ct, lab in supp:
        df = pd.read_csv(d / "tsv" / fn, sep="\t", skiprows=1)
        for _, r in df.iterrows():
            rows.append(dict(source_id=SID, set="conserved_control", gene_symbol=r["Human_GeneName"], symbol_species="human",
                             contrast="human_vs_mouse", cell_type=ct, direction="conserved",
                             effect=f"median_log2RPKM_human={float(r['Human_median_log2RPKM']):.3f};median_log2RPKM_mouse={float(r['Mouse_median_log2RPKM']):.3f}",
                             evidence="conserved high expression per paper: median log2-RPKM>=4 in both species (1:1 Ensembl v92 homologues only)",
                             circular="no", note=f"{lab}; mouse symbol {r['Mouse_GeneName']}"))
    out_df = pd.DataFrame(rows, columns=COLS)
    out_df.to_csv(out, sep="\t", index=False)
    print(out_df.groupby(["set", "cell_type", "direction"]).gene_symbol.nunique().to_string())
    print("unique genes per set:", out_df.groupby("set").gene_symbol.nunique().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:])
