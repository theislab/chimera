"""Derive the mouse-anchor-gene vs human fetal kidney comparison of Lindstrom et al. 2018 JASN 29:785 (Table 2).

Usage: python -I derive_kidney_lindstrom2018.py <table2_jpg> <out_tsv>
Table 2 exists only as an image (PMC5827606, ASN.2017080887t2.jpg); the rows below are a manual
transcription of that image (gene, mouse domain, reproduces primary expression, additional sites,
human 14-16 week expression). The jpg path is only checked for existence/sha256 so the card can cite it.
Mechanical rule (Table 2 columns; green rows = 'conserved expression between mouse and human'):
  reproduces=Yes, additional=No  -> conserved_control, direction conserved
  reproduces=Yes, additional=Yes -> conserved_with_additional_human_sites, direction conserved (in mouse domain)
  reproduces=No,  additional=Yes -> species_divergent, direction up_in_mouse (mouse domain not reproduced in human)
  reproduces=No,  additional=No  -> not_detected_in_human, direction up_in_mouse
"""
import hashlib
import sys
from pathlib import Path
import pandas as pd

COLS = ["source_id", "set", "gene_symbol", "symbol_species", "contrast", "cell_type",
        "direction", "effect", "evidence", "circular", "note"]
DOM = {"EPT": "early proximal tubule (EPT)", "LOH": "loop of Henle (LOH)", "RC": "renal corpuscle (RC)",
       "RV": "renal vesicle (RV)"}
T2 = [  # gene, mouse_domain, reproduces, additional, human expression (14-16 wk)
    ("ACE", "EPT", "No", "Yes", "Inner and outer medullary collecting duct; renal pelvis; proximal segment of S-shaped body"),
    ("ACSM1", "EPT", "No", "Yes", "Inner and outer medullary collecting duct; cap mesenchyme; nephrogenic interstitium; ureteric tip"),
    ("BDH2", "EPT", "Yes", "Yes", "Renal proximal tubule; loop of Henle; renal distal tubule"),
    ("CRYL1", "EPT", "No", "Yes", "Inner and outer medullary collecting duct; loop of henle"),
    ("PRODH2", "EPT", "No", "Yes", "Maturing renal corpuscle; S-shaped body"),
    ("SLC27A2", "EPT", "Yes", "Yes", "Medial segment of S-shaped body; early proximal tubule of capillary loop nephron; loop of Henle"),
    ("SLC3A1", "EPT", "Yes", "Yes", "Renal proximal tubule; loop of Henle"),
    ("SORD", "EPT", "No", "Yes", "Collecting duct; renal pelvis; ureter; ureteric tip; distal segment of S-shaped body"),
    ("TCN2", "EPT", "Yes", "Yes", "Renal proximal tubule; ureteric tip; collecting duct"),
    ("METTL7B", "EPT", "Yes", "Yes", "Renal proximal tubule; loop of Henle"),
    ("SLC22A6", "EPT", "Yes", "No", "Renal proximal tubule (strong in S3)"),
    ("ENTPD5", "EPT", "Yes", "No", "Medial segment of S-shaped body; renal proximal tubule"),
    ("C2", "EPT", "No", "Yes", "Cap mesenchyme; nephrogenic interstitium; inner medullary interstitium"),
    ("AQP11", "EPT", "No", "Yes", "Collecting duct; renal pelvis; cap mesenchyme; S-shaped body"),
    ("FBP1", "EPT", "Yes", "Yes", "Renal pelvis; inner medullary collecting duct; renal proximal tubule"),
    ("FMO2", "EPT", "No", "No", "Negative result - no SISH stain"),
    ("MOGAT2", "EPT", "No", "No", "Negative result - no SISH stain"),
    ("SLC18A1", "EPT", "No", "No", "Negative result - no SISH stain"),
    ("SLC6A20", "EPT", "No", "No", "Negative result - no SISH stain"),
    ("SPP2", "EPT", "No", "No", "Negative result - no SISH stain"),
    ("GLYCTK", "EPT", "Yes", "Yes", "Renal proximal tubule, early proximal tubule of capillary loop nephron, loop of Henle"),
    ("UMOD", "LOH", "Yes", "No", "Loop of Henle ascending limb; renal distal tubule"),
    ("GPSM3", "RC", "Yes", "Yes", "Renal corpuscle, inner medullary collecting duct"),
    ("TDRD5", "RC", "Yes", "Yes", "Maturing renal corpuscle"),
    ("VIP", "RC", "No", "Yes", "Renal pelvis; collecting duct; loop of Henle"),
    ("NPY", "RV", "No", "No", "Negative result - no SISH stain"),
]
RULE = {("Yes", "No"): ("conserved_control", "conserved"),
        ("Yes", "Yes"): ("conserved_with_additional_human_sites", "conserved"),
        ("No", "Yes"): ("species_divergent", "up_in_mouse"),
        ("No", "No"): ("not_detected_in_human", "up_in_mouse")}


def main(jpg, out):
    sha = hashlib.sha256(Path(jpg).read_bytes()).hexdigest()
    rows = []
    for g, dom, rep, add, hum in T2:
        st, di = RULE[(rep, add)]
        rows.append(dict(source_id="kidney_lindstrom2018", set=st, gene_symbol=g, symbol_species="human",
                         contrast="human_vs_mouse", cell_type=DOM[dom], direction=di,
                         effect=f"reproduces_primary_mouse_expression={rep};additional_human_sites={add}",
                         evidence="Table 2 (SISH of GUDMAP mouse anchor genes in human week 14-16 fetal kidney; green rows = conserved)",
                         circular="no",
                         note=f"human expression: {hum}; developmental (fetal human vs mouse anchor-gene domain); transcribed from Table 2 image sha256 {sha[:16]}"))
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(out, sep="\t", index=False)
    print(df.groupby(["set", "direction"]).gene_symbol.nunique().to_string())


if __name__ == "__main__":
    main(*sys.argv[1:])
