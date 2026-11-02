"""C1: Zilionis 2019 (GSE127465) to per-species h5ad, tumour libraries, raw counts from RAW.tar, authors' cell metadata.

The normalised matrices are not used. Cells: those in the authors' metadata, tumour only: human Tissue == tumor, mouse Tumor or healthy == t.
The authors' QC differs by species (human >= 300 UMI, mitochondrial <= 20%; mouse >= 637 UMI, <= 15%), so one rule is applied on top in both species, the stricter pair:
at least 637 UMI and at most 15% mitochondrial (checklist item 4; the cut acts on the authors' columns). The deposit has symbols only. They are mapped through the release they were built on (checklist 2b): all 41,861 human names are in Ensembl 91 and all 28,205 mouse names in Ensembl 81, no other
release checked (75-98) reaches that; names unique in the release; stable ID then joins Ensembl 110 orthologs. An unmapped gene is not listed, so it is a missing gene, not a zero.

    python revision/tasks/C1_cohorts/prepare_tumour.py
"""
from __future__ import annotations

import gzip
import io
import json
import tarfile

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

from revision.tasks.C1_cohorts import configs as CF
from revision.tasks.C1_cohorts import orthologs as OR

D = CF.DATA / "tumour_zilionis"
UMI_MIN, MT_MAX = 637, 15.0
LYMPH = {"tT cells": "T", "tB cells": "B", "tNK cells": "NK", "T cells": "T", "B cells": "B", "NK cells": "NK"}


def label(major: str, minor: str) -> str:
    if major in LYMPH:
        return LYMPH[major]
    return minor if major in ("tMoMacDC", "tNeutrophils", "tpDC", "MoMacDC", "Neutrophils", "pDC") else "other"


def ids_for(names: pd.Index, tag: str) -> tuple[pd.Series, dict]:
    m = OR.release_names(tag)
    out = pd.Series(names, index=names).map(m)
    dup = pd.Index(names)[pd.Index(names).duplicated(keep=False)]
    out[out.index.isin(dup)] = np.nan
    return out, dict(annotation=tag, n_symbols=len(names), n_mapped=int(out.notna().sum()))


def load(meta: pd.DataFrame, tar: tarfile.TarFile, tag: str) -> tuple[sp.csr_matrix, pd.Index, pd.DataFrame]:
    blocks, obs, genes = [], [], None
    for lib, g in meta.groupby("Library", sort=True):
        name = [m for m in tar.getnames() if m.endswith(f"_{tag}{lib}_raw_counts.tsv.gz")]
        if len(name) != 1:
            raise ValueError(f"{tag} {lib}: {name}")
        with gzip.open(io.BytesIO(tar.extractfile(name[0]).read()), "rt") as f:
            head = f.readline().rstrip("\n").split("\t")
            gs = pd.Index(head[1:])
            if genes is None:
                genes = gs
            elif not genes.equals(gs):
                raise ValueError("gene order differs between libraries")
            wanted = set(g["Barcode"])
            rows, bcs = [], []
            for line in f:
                bc, rest = line.split("\t", 1)
                if bc in wanted:
                    v = np.fromstring(rest, sep="\t", dtype=np.float32)
                    if (v != np.round(v)).any() or v.min() < 0:
                        raise ValueError(f"{lib} {bc}: not integer counts")
                    rows.append(sp.csr_matrix(v))
                    bcs.append(bc)
        m = sp.vstack(rows).tocsr()
        gg = g.set_index("Barcode").loc[bcs]
        gg.index = [f"{lib}_{b}" for b in bcs]
        blocks.append(m)
        obs.append(gg)
        print(tag, lib, m.shape, flush=True)
    return sp.vstack(blocks).tocsr(), genes, pd.concat(obs)


def main() -> int:
    h = pd.read_csv(D / "GSE127465_human_cell_metadata_54773x25.tsv.gz", sep="\t")
    m = pd.read_csv(D / "GSE127465_mouse_cell_metadata_15939x12.tsv.gz", sep="\t")
    h = h[h["Tissue"] == "tumor"]
    m = m[m["Tumor or healthy"] == "t"].copy()
    n0 = dict(human=len(h), mouse=len(m))
    keep = lambda t: t[(t["Total counts"] >= UMI_MIN) & (t["Percent counts from mitochondrial genes"] <= MT_MAX)]  # noqa: E731
    h, m = keep(h), keep(m)
    m["Library"] = m["Library"].astype(str)
    rep = dict(common_qc=dict(umi_min=UMI_MIN, mt_max=MT_MAX, before=n0, after=dict(human=len(h), mouse=len(m))))
    with tarfile.open(D / "GSE127465_RAW.tar") as tar:
        for sp_short, meta, tag, unit, out in (("hsapiens", h, "human_", "Patient", "prepared_human.h5ad"), ("mmusculus", m, "mouse_", "Biological replicate", "prepared_mouse.h5ad")):
            X, genes, o = load(meta, tar, tag)
            idm, r = ids_for(genes, {"hsapiens": "hs91", "mmusculus": "mm81"}[sp_short])
            keep = idm.notna().to_numpy()
            ids = idm[keep].to_numpy()
            obs = pd.DataFrame(index=o.index)
            obs["donor_id"] = o[unit].astype(str).to_numpy()
            obs["label"] = [label(a, b) for a, b in zip(o["Major cell type"].astype(str), o["Minor subset"].astype(str))]
            obs["assay"] = "inDrop"
            obs["sex"] = "NA"
            obs["library"] = o["Library"].astype(str).to_numpy()
            var = pd.DataFrame({"feature_name": np.asarray(genes)[keep]}, index=ids)
            a = ad.AnnData(X=X[:, np.flatnonzero(keep)].astype(np.float32), obs=obs, var=var)
            a.write_h5ad(D / out)
            r.update(n_cells=int(a.n_obs), n_genes=int(a.n_vars), labels=obs["label"].value_counts().to_dict(), max_count=float(a.X.max()), median_library=float(np.median(np.asarray(a.X.sum(1)).ravel())))
            rep[sp_short] = r
    (CF.RES / "tumour_zilionis").mkdir(parents=True, exist_ok=True)
    (CF.RES / "tumour_zilionis" / "prepare_report.json").write_text(json.dumps(rep, indent=2))
    print(rep)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
