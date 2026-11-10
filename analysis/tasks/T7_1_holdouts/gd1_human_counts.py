"""G-D1, human side: nuclei per coarse type per sample from the GSE202379 Seurat metadata.

Reads only the @meta.data TSV written by extract_seurat_meta.R (author annotation
``cell.annotation``). No expression value is read. The mouse side needs the annotation of
analysis/tasks/T7_2_disease_objects.

Usage: python -I gd1_human_counts.py <GSE202379_meta.data.tsv.gz> <out_dir>
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

TYPES = {  # types of the pre-specified analysis plan, section 8 -> author labels
    "hepatocyte": ["Hepatocytes"],
    "hsc_fibroblast": ["Stellate"],
    "endothelial": ["Endothelial"],
    "macrophage": ["Macrophages"],
    "cholangiocyte": ["Cholangiocytes"],
}
MIN_NUCLEI = 20


def main(meta: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    md = pd.read_csv(meta, sep="\t", dtype=str, usecols=["cell_barcode", "orig.ident", "Patient.ID", "Disease.status",
                                                          "cell.annotation", "Gender"])
    inv = {lab: t for t, labs in TYPES.items() for lab in labs}
    md["type"] = md["cell.annotation"].map(inv)
    tab = md.dropna(subset=["type"]).groupby(["Disease.status", "Patient.ID", "orig.ident", "type"]).size().unstack(fill_value=0)
    tab = tab.reindex(columns=list(TYPES), fill_value=0)
    tab.to_csv(out / "gd1_human_nuclei_per_sample_type.tsv", sep="\t")
    lines = [f"nuclei {len(md)}, samples {md['orig.ident'].nunique()}, patients {md['Patient.ID'].nunique()}",
             "author labels: " + "; ".join(f"{k}={v}" for k, v in md["cell.annotation"].value_counts().items())]
    # donor level: sum a donor's samples
    don = tab.groupby(level=["Disease.status", "Patient.ID"]).sum()
    for st in ("Healthy control", "NASH w/o cirrhosis"):
        s = tab.xs(st, level="Disease.status")
        d = don.xs(st, level="Disease.status")
        lines.append(f"{st}: samples {len(s)}, donors {len(d)}")
        for t in TYPES:
            lines.append(f"  {t}: samples with >={MIN_NUCLEI} nuclei {int((s[t] >= MIN_NUCLEI).sum())}/{len(s)}; "
                         f"donors {int((d[t] >= MIN_NUCLEI).sum())}/{len(d)}; median per sample {int(s[t].median())}")
    (out / "gd1_human_counts.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
