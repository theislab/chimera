"""Derive literature gene lists from Shami et al. 2020 (Dev Cell 54:529), human / macaque / mouse testis.

Usage: python -I derive_shami2020.py <data_dir> <out_tsv>
  <data_dir>: data/revision/E6/shami2020 (holds converted/ from xlsx_to_tsv.py and PMC7879256_fulltext.html)
  <out_tsv>:  revision/tasks/E6_known_differences/lists/shami2020.tsv

Sets (mechanical, the paper's own definitions; see shami2020.md):
  conserved_control  Table S5 A-C (mmc6): gene in the SAME k-means cluster (K1-K6) in both species of the pair,
                     the paper's "in-phase" genes (Figure 5A, S5A).
  species_divergent  Table S5 A-C: gene in DIFFERENT clusters in the two species, the paper's "phase-shifted"
                     genes (temporal divergence along the aligned germ-cell trajectory, not a level difference).
  species_specific   Table S5 A-C: gene classified (K1-K6) in one species and "NULL" in the other, the paper's
                     "unique" genes (not highly variable and no cluster centroid with r > 0.8 in that species).
  conserved_marker_spg  Table S3 A (mmc4): FindConservedMarkers meta-analysis markers of the six consensus
                     spermatogonia states (SPG1-6), three species.
  primate_msci_escapee  Table S1 E (mmc2): X-linked genes escaping meiotic sex chromosome inactivation in human /
                     macaque spermatocytes; the paper: "the same threshold in mice identified a single gene, Tsga8".
  text_known_difference / text_conserved  statements quoted verbatim from the main text; each quote is checked
                     to occur verbatim in the downloaded PMC full text.
"""
import html
import re
import sys
from pathlib import Path

import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
SRC = "shami2020"
SP = {"Human": "human", "Macaque": "macaque", "Mouse": "mouse"}
S5_SHEETS = {"A._Human_vs._Macaque": ("Human", "Macaque"), "B._Macaque_vs._Mouse": ("Macaque", "Mouse"),
             "C._Human_vs._Mouse": ("Human", "Mouse")}
S5_EVID = ("TableS5{s} (mmc6): k-means k=6 of 1-1-1 orthologue HVGs over 21 ordered germ states "
           "(SPG + 20 aligned GC stages), per species; non-HV genes assigned if r>0.8 to a centroid, else NULL")


def one(d, pattern):
    hits = sorted(Path(d).glob(pattern))
    assert len(hits) == 1, (pattern, hits)
    return hits[0]


def s5(conv):
    rows, human_k = [], {}
    for sheet, (a, b) in S5_SHEETS.items():
        t = pd.read_csv(one(conv, f"*mmc6__{sheet}.tsv"), sep="\t", dtype=str, keep_default_na=False)
        assert list(t.columns) == [f"{a} Gene", f"{b} Gene", f"{a} Cluster", f"{b} Cluster"], t.columns
        ok = {"1", "2", "3", "4", "5", "6", "NULL"}
        assert set(t[f"{a} Cluster"]) <= ok and set(t[f"{b} Cluster"]) <= ok
        assert not ((t[f"{a} Cluster"] == "NULL") & (t[f"{b} Cluster"] == "NULL")).any()
        contrast = f"{SP[a]}_vs_{SP[b]}"
        for g1, g2, k1, k2 in t.itertuples(index=False):
            if a == "Human":
                human_k.setdefault(g1, {})[sheet] = k1
            if k1 == k2:
                st, dirn, cls = "conserved_control", "conserved", "in_phase"
            elif "NULL" in (k1, k2):
                st, dirn, cls = "species_specific", "divergent", "unique"
            else:
                st, dirn, cls = "species_divergent", "divergent", "phase_shifted"
            lab = lambda k: "NA" if k == "NULL" else f"K{k}"
            rows.append(dict(
                set=st, gene_symbol=g1, symbol_species=SP[a], contrast=contrast,
                cell_type=f"germ_kmeans_{lab(k1)}|{lab(k2)}", direction=dirn,
                effect=f"{SP[a]}_cluster={k1};{SP[b]}_cluster={k2};class={cls}",
                evidence=S5_EVID.format(s=sheet[0]),
                note=f"{SP[b]}_symbol={g2};cell_type=cluster in {SP[a]}|{SP[b]} (K1 early ... K6 late, Fig 5A)"))
    # consistency: a human gene's cluster should be the same in sheets A and C (human clustered once)
    both = {g: v for g, v in human_k.items() if len(v) == 2}
    nonnull = {g: v for g, v in both.items() if "NULL" not in v.values()}
    mism = sum(len(set(v.values())) > 1 for v in nonnull.values())
    return rows, dict(human_genes_in_A_and_C=len(both), both_nonnull=len(nonnull), human_cluster_mismatch=mism)


def s3a(conv):
    t = pd.read_csv(one(conv, "*mmc4__A._Meta-analysis.tsv"), sep="\t", dtype=str, keep_default_na=False)
    t = t.rename(columns={t.columns[0]: "gene"})
    rows = []
    for r in t.to_dict("records"):
        eff = ";".join(f"{s}_logFC={r[f'{s}_avg_logFC']},{s}_padj={r[f'{s}_p_val_adj']}"
                       for s in ("Human", "Monkey", "Mouse"))
        rows.append(dict(
            set="conserved_marker_spg", gene_symbol=r["gene"], symbol_species="human",
            contrast="human_vs_macaque_mouse", cell_type=f"SPG{r['Cluster']}", direction="conserved",
            effect=f"{eff};minimump_p={r['minimump_p_val']}",
            evidence="TableS3A (mmc4): FindConservedMarkers (minimump) SPG state vs other SPG states; "
                     ">=10% detection difference and >=1.3-fold in the state (caption)",
            note="1-1-1 orthologues, human symbol; per-species padj in effect (some are 1, the meta-analysis "
                 "does not require significance in every species)"))
    return rows


def s1e(conv):
    t = pd.read_csv(one(conv, "*mmc2__E.MSCI_Escapees.tsv"), sep="\t", header=None, dtype=str,
                    keep_default_na=False, names=range(12))  # ragged rows (human 16, macaque 14 genes)
    assert t.iloc[0, 0] == "Human" and t.iloc[0, 8] == "Macaque", t.iloc[0].tolist()
    body = t.iloc[2:]
    rows = []
    for sp, cols in (("human", (0, 1, 2, 3)), ("macaque", (8, 9, 10, 11))):
        for v in body.itertuples(index=False):
            g, spc, spg, ratio = (v[i] for i in cols)
            if not isinstance(g, str) or not g:
                continue
            for contrast in ([f"{sp}_vs_mouse"]):
                rows.append(dict(
                    set="primate_msci_escapee", gene_symbol=g, symbol_species=sp, contrast=contrast,
                    cell_type="spermatocyte", direction=f"up_in_{sp}",
                    effect=f"mean_SPC={float(spc):.3f};mean_SPG={float(spg):.3f};SPC/SPG={float(ratio):.2f}",
                    evidence="TableS1E (mmc2): X-linked, mean normalized expression >0.5 in spermatocytes and >2-fold "
                             "over spermatogonia; same threshold in mouse identified only Tsga8 (main text)",
                    note="species difference = escape from MSCI in primate spermatocytes vs silencing in mouse"
                         + (";macaque Ensembl ID as published" if g.startswith("ENSMMUG") else "")))
    return rows


# ---- text statements (verbatim; checked against the full text) ----
Q_SPG6 = ("However, we identify many genes that are either expressed in all three species (i.e. FMR1NB , ZCWPW1 , "
          "DPEP3 , IQBP1 , CALR ), in primates only ( BEND2 , ZRANB2 , PAGE4 , ZNFX1 , HLTF , ZBED5 ) or are "
          "singularly expressed in mouse ( Swt1 , Tgfbr1 , Rhox13 , PhTF1 , Fmr1 ), macaque ( ZNF850 , MAGEB17 , "
          "ZMYM1 , TCEA1 ) and human ( PRDM7 , full VCX family, ERBB3 , ERVK13-1 , SSX3 , TRIML2 ) ( Table S3A ).")
Q_SPG2 = ("In SPG2 cells, the noncanonical markers enriched in SPG1 (described above) are diminished, whereas certain "
          "canonical markers for undifferentiated spermatogonia increase in expression ( ZBTB16 (mouse and human), "
          "GFRA1 (all three species)), as do several noncanonical markers ( DUSP6 , TCF4 , L1TD1 (macaque and human "
          "only)) ( Figure 2D , E ( L1TD1 shown); Table S3 ).")
Q_MKI67 = "MKI67 is upregulated in SPG3 of mice and macaques, but not until SPG4 in humans ( Figure S2F ; Table S3 )."
Q_MYOID = ("Unlike in mouse, primate myoid cells don’t express the spermatogonial stem cell proliferative factor GDNF , "
           "but rather express components of other signaling pathways not detected in mouse, like PDGFRA , EGFR , "
           "PTCH1 , and PPARA ( Table S6 ).")
Q_PIWIL4 = "Interestingly, PIWIL4 is a SPG1-specific marker for all three species ( Figure 2D – E ; Table S3 )."

VCX = ["VCX", "VCX2", "VCX3A", "VCX3B"]  # expansion of "full VCX family" (HGNC VCX/Y family, X-linked members)


def text_rows():
    out = []

    def add(st, g, gsp, contrast, ct, dirn, quote, where, note=""):
        out.append(dict(set=st, gene_symbol=g, symbol_species=gsp, contrast=contrast, cell_type=ct,
                        direction=dirn, effect="text", evidence="text",
                        note=f'"{quote}" [{where}]' + (f";{note}" if note else "")))
    w6 = "Results, 'Consensus SPG6', Table S3A"
    for g in ["FMR1NB", "ZCWPW1", "DPEP3", "IQBP1", "CALR"]:
        add("text_conserved", g, "human", "human_vs_macaque_mouse", "SPG6", "conserved", Q_SPG6, w6)
    for g in ["BEND2", "ZRANB2", "PAGE4", "ZNFX1", "HLTF", "ZBED5"]:
        add("text_known_difference", g, "human", "human_vs_mouse", "SPG6", "up_in_human", Q_SPG6, w6, "primates only")
        add("text_known_difference", g, "human", "macaque_vs_mouse", "SPG6", "up_in_macaque", Q_SPG6, w6, "primates only")
    for g in ["Swt1", "Tgfbr1", "Rhox13", "PhTF1", "Fmr1"]:
        add("text_known_difference", g, "mouse", "human_vs_mouse", "SPG6", "up_in_mouse", Q_SPG6, w6, "mouse only")
        add("text_known_difference", g, "mouse", "macaque_vs_mouse", "SPG6", "up_in_mouse", Q_SPG6, w6, "mouse only")
    for g in ["ZNF850", "MAGEB17", "ZMYM1", "TCEA1"]:
        add("text_known_difference", g, "macaque", "human_vs_macaque", "SPG6", "up_in_macaque", Q_SPG6, w6, "macaque only")
        add("text_known_difference", g, "macaque", "macaque_vs_mouse", "SPG6", "up_in_macaque", Q_SPG6, w6, "macaque only")
    for g in ["PRDM7"] + VCX + ["ERBB3", "ERVK13-1", "SSX3", "TRIML2"]:
        nt = "human only" + (";'full VCX family' expanded to VCX,VCX2,VCX3A,VCX3B (assumption)" if g in VCX else "")
        add("text_known_difference", g, "human", "human_vs_macaque", "SPG6", "up_in_human", Q_SPG6, w6, nt)
        add("text_known_difference", g, "human", "human_vs_mouse", "SPG6", "up_in_human", Q_SPG6, w6, nt)
    w2 = "Results, 'Consensus SPG1-2', Fig 2D-E"
    add("text_known_difference", "L1TD1", "human", "human_vs_mouse", "SPG2", "up_in_human", Q_SPG2, w2, "macaque and human only")
    add("text_known_difference", "L1TD1", "human", "macaque_vs_mouse", "SPG2", "up_in_macaque", Q_SPG2, w2, "macaque and human only")
    add("text_conserved", "GFRA1", "human", "human_vs_macaque_mouse", "SPG2", "conserved", Q_SPG2, w2, "all three species")
    add("text_conserved", "PIWIL4", "human", "human_vs_macaque_mouse", "SPG1", "conserved", Q_PIWIL4, w2)
    w3 = "Results, 'Consensus SPG3-5', Fig S2F"
    add("text_known_difference", "MKI67", "human", "human_vs_mouse", "SPG3", "up_in_mouse", Q_MKI67, w3, "timing: SPG3 in mouse, SPG4 in human")
    add("text_known_difference", "MKI67", "human", "human_vs_macaque", "SPG3", "up_in_macaque", Q_MKI67, w3, "timing: SPG3 in macaque, SPG4 in human")
    wm = "Results, somatic cells, peritubular myoid cells, Table S6"
    add("text_known_difference", "GDNF", "human", "human_vs_mouse", "peritubular_myoid", "up_in_mouse", Q_MYOID, wm)
    add("text_known_difference", "GDNF", "human", "macaque_vs_mouse", "peritubular_myoid", "up_in_mouse", Q_MYOID, wm)
    for g in ["PDGFRA", "EGFR", "PTCH1", "PPARA"]:
        add("text_known_difference", g, "human", "human_vs_mouse", "peritubular_myoid", "up_in_human", Q_MYOID, wm)
        add("text_known_difference", g, "human", "macaque_vs_mouse", "peritubular_myoid", "up_in_macaque", Q_MYOID, wm)
    return out, [Q_SPG6, Q_SPG2, Q_MKI67, Q_MYOID, Q_PIWIL4]


def plain_text(p):
    t = open(p, encoding="utf-8").read()
    t = re.sub(r"<script.*?</script>", "", t, flags=re.S)
    t = re.sub(r"<[^>]+>", " ", t)
    return html.unescape(re.sub(r"\s+", " ", t))


def main(data_dir, out):
    data_dir = Path(data_dir)
    conv = data_dir / "converted"
    r5, chk = s5(conv)
    rt, quotes = text_rows()
    full = plain_text(data_dir / "PMC7879256_fulltext.html")
    for q in quotes:
        assert q in full, q[:80]
    rows = r5 + s3a(conv) + s1e(conv) + rt
    df = pd.DataFrame(rows)
    df.insert(0, "source_id", SRC)
    df["circular"] = "no"
    df = df[COLS]
    assert not df.duplicated(["set", "gene_symbol", "contrast", "cell_type"]).any()
    df.to_csv(out, sep="\t", index=False)
    print("S5 consistency:", chk)
    print("rows", len(df))
    g = df.groupby(["set", "contrast"]).agg(rows=("gene_symbol", "size"), genes=("gene_symbol", "nunique"))
    print(g.to_string())
    print("unique genes per set:", df.groupby("set").gene_symbol.nunique().to_dict())
    cc = df[df.set.isin(["conserved_marker_spg", "primate_msci_escapee"]) | df.set.str.startswith("text")]
    print(cc.groupby(["set", "cell_type"]).gene_symbol.nunique().to_string())
    # overlap of the text-named species-specific SPG6 genes with Table S3A
    s3 = set(df[df.set == "conserved_marker_spg"].gene_symbol.str.upper())
    named = set(df[(df.set == "text_known_difference") & (df.cell_type == "SPG6")].gene_symbol.str.upper())
    print("text SPG6 species-specific genes also in TableS3A:", sorted(named & s3))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
