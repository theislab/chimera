"""Derive literature gene lists from Hodge et al. 2019 (Nature 573:61), human MTG vs mouse VISp/ALM.

Usage: python -I derive_hodge2019.py <converted_dir> <out_tsv>
  <converted_dir>: data/analysis/E6/hodge2019/converted (TSVs made by xlsx_to_tsv.py)
  <out_tsv>:       analysis/tasks/E6_known_differences/lists/hodge2019.tsv

Sets (all mechanical, the paper's own definitions; see hodge2019.md):
  human_mouse_divergent : Supplementary Table 5 (MOESM7), genes ranked by homol_expr_score
                          (beta score of log2 human-mouse expression differences across the 37
                          homologous types). Kept: the top 3,382 genes = the paper's "more highly
                          divergent than 95% of 252 housekeeping genes" (23%). Tiers top 3% / top 10%
                          (the paper's other cut-offs) recorded in `note`. Not cell-type resolved.
  conserved_control     : Supplementary Table 4 (MOESM6), homol_shared_markers per homologous type
                          (cluster-enriched in both species; up to 10 per type).
  text                  : statements quoted verbatim from the main text (evidence = text).
"""
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SRC = "hodge2019"
CONTRAST = "human_MTG_vs_mouse_VISp_ALM"
N_HK95 = 3382  # paper: "Twenty-three per cent of genes (3,382) were more highly divergent than 95% of 252 housekeeping genes"


def one(conv, pattern):
    hits = sorted(Path(conv).glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def divergent(conv):
    st5 = pd.read_csv(one(conv, "41586_2019_1506_MOESM7_ESM__*.tsv"), sep="\t", na_values=["NA"],
                      keep_default_na=False)
    s = st5.dropna(subset=["homol_expr_score"]).copy()
    n = len(s)
    assert n == 14553, n  # paper: 14,553 orthologous genes scored
    assert not s.gene.duplicated().any()
    s = s.sort_values("homol_expr_score", ascending=False, kind="mergesort").reset_index(drop=True)
    s["rank"] = range(1, n + 1)
    n3, n10 = round(0.03 * n), round(0.10 * n)
    top = s[s["rank"] <= N_HK95]
    # ties at the cut-off would make the rank rule ambiguous; check there are none
    assert s.loc[N_HK95 - 1, "homol_expr_score"] != s.loc[N_HK95, "homol_expr_score"]
    rows = []
    for r in top.itertuples():
        tier = "top3pct" if r.rank <= n3 else ("top10pct" if r.rank <= n10 else "top23pct")
        rows.append(dict(
            set="human_mouse_divergent", gene_symbol=r.gene, symbol_species="human",
            contrast=CONTRAST, cell_type="all_37_homologous_types", direction="divergent",
            effect=f"homol_expr_score={r.homol_expr_score:.4f};rank={r.rank}/{n};homol_cor={r.homol_cor}",
            evidence=f"SupplTable5 homol_expr_score rank<={N_HK95} (top 23%, paper's >95th pct of housekeeping genes)",
            note=f"tier={tier};top3pct=rank<={n3};top10pct=rank<={n10};fpkm_max_cluster(human)={r.fpkm_max_cluster}"))
    return rows, s, (n, n3, n10)


def conserved(conv):
    st4 = pd.read_csv(one(conv, "41586_2019_1506_MOESM6_ESM__*.tsv"), sep="\t", keep_default_na=False)
    rows = []
    for (hc, cls), g in st4.groupby(["homol_cluster", "homol_class"], sort=False):
        markers = g.homol_shared_markers.unique()
        assert len(markers) == 1, (hc, markers)
        hum = "|".join(g.loc[g.cluster_species == "human", "cluster_name"])
        mus = "|".join(g.loc[g.cluster_species == "mouse", "cluster_name"])
        m = markers[0].strip()
        if m.lower() in ("none", "", "na"):
            continue
        for gene in [x.strip() for x in m.split(",") if x.strip()]:
            rows.append(dict(
                set="conserved_control", gene_symbol=gene, symbol_species="human",
                contrast=CONTRAST, cell_type=hc, direction="conserved", effect="",
                evidence="SupplTable4 homol_shared_markers (>50% of cells in type and <=5 other clusters, both species)",
                note=f"homol_class={cls};human_clusters={hum};mouse_clusters={mus}"))
    return rows, st4


TEXT = [
    # (gene, symbol_species, cell_type, direction, quote)
    ("HTR3A", "human", "CGE interneurons (LAMP5/PAX6, VIP)", "up_in_mouse",
     "Surprisingly, the canonical mouse CGE interneuron marker HTR3A was not expressed in human CGE types (Fig. 3c)."),
    ("NPY", "human", "Inh L3-L6 SST NPY / mouse Sst Chodl (SST, L5-L6)", "up_in_mouse",
     "In situ hybridization showed sparse TH expression in L5-L6 of human MTG and the mouse homologous region (TEa), "
     "suggesting that this gene marks similar cell types in both species, whereas NPY was much more sparsely expressed "
     "in human, indicating differential expression of this closely studied marker between species."),
    ("TH", "human", "Inh L5-L6 SST TH (SST, L5-L6)", "conserved",
     "In situ hybridization showed sparse TH expression in L5-L6 of human MTG and the mouse homologous region (TEa), "
     "suggesting that this gene marks similar cell types in both species"),
    ("VIPR2", "human", "Chandelier (Inh L2-5 PVALB SCUBE3 / mouse Pvalb Vipr2)", "up_in_mouse",
     "For example, chandelier cells express Vipr2 in mouse but COL15A1 and NOG in human (Extended Data Fig. 10d)."),
    ("COL15A1", "human", "Chandelier (Inh L2-5 PVALB SCUBE3 / mouse Pvalb Vipr2)", "up_in_human",
     "For example, chandelier cells express Vipr2 in mouse but COL15A1 and NOG in human (Extended Data Fig. 10d)."),
    ("NOG", "human", "Chandelier (Inh L2-5 PVALB SCUBE3 / mouse Pvalb Vipr2)", "up_in_human",
     "For example, chandelier cells express Vipr2 in mouse but COL15A1 and NOG in human (Extended Data Fig. 10d)."),
]
for g in ["COL24A1", "COL12A1", "GRIK1", "GRIN3A"]:
    TEXT.append((g, "human", "cell-type-shifted (different cell types between species)", "divergent",
                 "Among the top 3% most-divergent genes (Supplementary Table 5), the collagens COL24A1 and COL12A1 and "
                 "glutamate receptor subunits GRIK1 and GRIN3A were expressed in different cell types between species "
                 "and were validated to have different laminar distributions in human and mouse (Fig. 6f, g)."))


def main(conv, out):
    div, s, (n, n3, n10) = divergent(conv)
    con, st4 = conserved(conv)
    rows = div + con
    rank = dict(zip(s.gene, s["rank"]))
    for gene, sp, ct, d, q in TEXT:
        rows.append(dict(set="text", gene_symbol=gene, symbol_species=sp, contrast=CONTRAST, cell_type=ct,
                         direction=d, effect=f"SupplTable5_rank={rank.get(gene, 'NA')}/{n}", evidence="text",
                         note=("Vipr2 written as mouse symbol in text; human ortholog VIPR2 used. " if gene == "VIPR2" else "")
                         + "quote: " + q))
    df = pd.DataFrame(rows)
    df["source_id"], df["circular"] = SRC, "no"
    df = df[COLS]
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False)

    # report
    print(f"scored genes n={n}; top3pct<= {n3}; top10pct<= {n10}; hk95 cut = {N_HK95}")
    print(f"score at cut: {s.loc[N_HK95-1,'homol_expr_score']:.4f}; next {s.loc[N_HK95,'homol_expr_score']:.4f}")
    htr = s[s.gene.str.match(r"^HTR\d")][["gene", "homol_expr_score", "rank"]]
    print("serotonin receptors (paper: 4/7 GPCRs + HTR3A, HTR3B in top 10%):")
    print(htr.assign(top10=htr["rank"] <= n10).to_string(index=False))
    print("text-gene ranks:", {g: rank.get(g) for g in ["COL24A1", "COL12A1", "GRIK1", "GRIN3A", "VIPR2", "COL15A1", "NOG"]})
    print(df.groupby("set").agg(rows=("gene_symbol", "size"), unique_genes=("gene_symbol", "nunique")))
    c = df[df.set == "conserved_control"].groupby("cell_type").gene_symbol.nunique()
    print("conserved_control types with markers:", len(c), "of", st4.homol_cluster.nunique())
    print(c.to_string())


if __name__ == "__main__":
    main(*sys.argv[1:3])
