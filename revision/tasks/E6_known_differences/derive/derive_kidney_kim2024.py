"""Derive human-vs-mouse developing-nephron gene lists from Kim et al. 2024 Dev Cell (McMahon lab).

Usage: python -I derive_kidney_kim2024.py <tsv_dir> <out_tsv>
  <tsv_dir> = data/revision/E6/kidney_kim2024/tsv (xls_to_tsv.py output of mmc5.xls = Table S4 and
  mmc6.xls = Table S5).

Table S5 (mmc6) sheets: 1 = conserved RNA DEGs (352 unique), 2 = divergent human ("human-enriched", 885),
3 = divergent mouse (579). Paper's rule (text): avg.FC > 0.5, pct.diff > 0.3, p.adj < 0.05 per integrated
cluster. All published rows are kept; rows with p_val_adj >= 0.05 are flagged in `note`.
Cluster labels: integrated clusters 0-24 are unnamed in the tables; `cell_type` records the cluster id and
the rank-1 marker of that cluster in Table S4 top50 sheets (mouse scRNA sheet 1 and human snRNA sheet 6).
"""
import sys
from pathlib import Path
import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SID = "kidney_kim2024"
P = "1-s2.0-S1534580724004507-"
RULE = "paper rule: avg.FC>0.5, pct.diff>0.3, p.adj<0.05 within integrated cluster (Table S5)"
CTX = "developing kidney: human fetal snRNA-seq vs mouse embryonic/P0 scRNA-seq, nephron + ureteric epithelium"


def top_marker(path):
    df = pd.read_csv(path, sep="\t", header=None, dtype=str)
    ids = df.iloc[0, 1:].tolist()
    first = df.iloc[2, 1:].tolist()  # row '1' = rank-1 marker
    return {str(int(float(i))): m for i, m in zip(ids, first)}


def main(tsv_dir, out):
    d = Path(tsv_dir)
    mm = top_marker(d / f"{P}mmc5__1.tsv")
    hm = top_marker(d / f"{P}mmc5__6.tsv")
    label = lambda c: f"integrated cluster {c} (top marker mouse {mm.get(c)}, human {hm.get(c)})"
    rows = []
    s1 = pd.read_csv(d / f"{P}mmc6__1.tsv", sep="\t")
    for _, r in s1.iterrows():
        c = str(int(r["cluster"]))
        rows.append(dict(set="conserved_control", gene_symbol=r["rowname"], symbol_species="human", cell_type=label(c),
                         direction="conserved",
                         effect=f"human_avg_logFC={r['human_avg_logFC']:.3f};mouse_avg_logFC={r['mouse_avg_logFC']:.3f};human_pct_diff={r['human_pct_diff']};mouse_pct_diff={r['mouse_pct_diff']};max_padj={max(r['human_p_val_adj'], r['mouse_p_val_adj']):.3g}",
                         evidence="conserved cell-type DEG in both species; " + RULE,
                         note="Table S5 sheet 1 (Conserved, RNA); " + CTX))
    for sheet, st, sp_dir in [(2, "human_enriched", "up_in_human"), (3, "mouse_enriched", "up_in_mouse")]:
        df = pd.read_csv(d / f"{P}mmc6__{sheet}.tsv", sep="\t")
        for _, r in df.iterrows():
            c = str(int(r["cluster"]))
            g = str(r["gene"])
            sym_sp = "human" if sheet == 2 else ("mouse" if g != g.upper() else "mouse_uppercased")
            note = f"Table S5 sheet {sheet} (Divergent, {'human' if sheet == 2 else 'mouse'}, RNA); " + CTX
            if sheet == 2:
                note += f"; mouse symbol {r['Mouse.Symbol']}; Homologene {r['Homologene_ID']}"
            if float(r["p_val_adj"]) >= 0.05:
                note += "; FLAG p_val_adj>=0.05 although listed"
            rows.append(dict(set=st, gene_symbol=g, symbol_species=sym_sp, cell_type=label(c), direction=sp_dir,
                             effect=f"avg_logFC={r['avg_logFC']:.3f};pct1={r['pct.1']};pct2={r['pct.2']};pct_diff={r['pct_diff']};p_val_adj={r['p_val_adj']:.3g}",
                             evidence=f"cluster DEG in {'human' if sheet == 2 else 'mouse'} only (complementary DEGs excluded); " + RULE,
                             note=note))
    out_df = pd.DataFrame(rows)
    out_df["source_id"], out_df["contrast"], out_df["circular"] = SID, "human_vs_mouse", "no"
    out_df = out_df[COLS]
    out_df.to_csv(out, sep="\t", index=False)
    print(out_df.groupby("set").agg(rows=("gene_symbol", "size"), unique=("gene_symbol", "nunique")).to_string())
    print("flagged padj>=0.05:", out_df.note.str.contains("FLAG").sum())
    print(out_df.symbol_species.value_counts().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
