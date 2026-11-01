"""Coverage of Ensembl 99 human-mouse 1:1 dn, ds. Counts only; no readout is read.

Usage: python -I dnds_release99_coverage.py <chimera_repo> <out_dir>
"""
import sys
from pathlib import Path

import pandas as pd

repo, out = Path(sys.argv[1]), Path(sys.argv[2])
sys.path.insert(0, str(repo / "revision" / "tasks" / "T4_1_gold"))
from gold_checks import overlap_row, vocabularies  # noqa: E402

voc = vocabularies(repo)
parts = []
for ch in pd.read_csv(repo / "data/revision/gold/G8/Compara.99.protein_default.homologies.tsv.gz", sep="\t",
                      dtype=str, chunksize=2_000_000,
                      usecols=["gene_stable_id", "homology_type", "homology_species", "dn", "ds"]):
    parts.append(ch[ch.homology_species.isin(["mus_musculus", "macaca_mulatta"])])
d = pd.concat(parts)
rows, ov = [], []
for (sp, ht), g in d.groupby(["homology_species", "homology_type"]):
    dn = pd.to_numeric(g.dn.replace("NULL", None), errors="coerce")
    ds = pd.to_numeric(g.ds.replace("NULL", None), errors="coerce")
    ok = dn.notna() & ds.notna()
    rows.append({"release": 99, "species": sp, "homology_type": ht, "rows": len(g), "both_populated": int(ok.sum()),
                 "ds_zero": int((ok & (ds == 0)).sum()), "ds_median": ds[ok].median(), "ds_q99": ds[ok].quantile(.99),
                 "dn_median": dn[ok].median()})
    if ht == "ortholog_one2one":
        ids = set(g.gene_stable_id[ok & (ds > 0)])
        ov.append(overlap_row(f"G8_compara99_{sp}_1to1_dn_ds_ds>0", ids, voc, "candidate only, needs amendment"))
pd.DataFrame(rows).to_csv(out / "dnds_release99.tsv", sep="\t", index=False)
pd.DataFrame(ov).to_csv(out / "id_overlap__dnds99.tsv", sep="\t", index=False)
print(pd.DataFrame(rows).to_string(index=False))
print(pd.DataFrame(ov).drop(columns="note").to_string(index=False))
