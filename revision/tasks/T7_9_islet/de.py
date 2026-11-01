"""T7.9: pseudobulk differential expression for the islet T2D case (plan section 8b).

Runs in ``revision_de_env`` (PyDESeq2 0.5.2), never ``chimera_env``. No Chimera readout is read.

Per DE-able type t (``results/revision/T7_9/gd1b_types.csv``), on the DE units of
``donor_roles.tsv`` that carry >= 20 cells of t (the G-D1b unit filter):

* human: ``~ sex + chemistry + condition``, T2D vs ND DE half;
* mouse, primary: ``~ study + condition`` on the 6 untreated libraries (db/db 2+2, mSTZ 1+1);
* mouse, per model (Family 3): one factor ``group`` = study x condition, so dispersion is shared;
  db/db = dbdb_disease vs dbdb_control, mSTZ = mstz_disease vs mstz_control (1 vs 1, descriptive).

Wald tests; apeglm shrinkage of the tested coefficient (``lfc_shrink``). Mouse fits use only genes
measured in both DE studies (``var['measured_dbdb']`` and ``var['measured_mstz']``; elsewhere the
atlas pads a gene missing from a study's reference with zeros). Genes with < 10 counts summed over
the units are not tested. ``det_frac`` = cells with count > 0 / cells, over the units
used, for section 8b's 5% detection rule.

    PYTHONPATH=<chimera repo> python revision/tasks/T7_9_islet/de.py [--types beta,...] [--n-cpus 16]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings

import anndata as ad
import numpy as np
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.default_inference import DefaultInference
from pydeseq2.ds import DeseqStats

from revision.common import gene_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("T7_9_de")

ROOT = gene_ids.REPO_ROOT
DATA = ROOT / "data" / "revision" / "disease_islet"
RES = ROOT / "results" / "revision" / "T7_9"
OUT = RES / "de"
MIN_CELLS_UNIT = 20
MIN_COUNTS = 10
STUDY = {"db/db": "dbdb", "mSTZ": "mstz", "human48": "human48"}


def units_for(pb: ad.AnnData, t: str) -> ad.AnnData:
    o = pb.obs
    keep = (o["cell_ontology_class"] == t) & o["role"].isin(["de_control", "de_disease"]) \
        & (o["n_nuclei"].astype(int) >= MIN_CELLS_UNIT)
    return pb[keep.to_numpy()].copy()


def fit(sub: ad.AnnData, design: str, factor: str, test: str, ref: str, n_cpus: int) -> pd.DataFrame:
    counts = pd.DataFrame(np.asarray(sub.X.toarray(), dtype=np.int64), index=sub.obs_names,
                          columns=sub.var_names)
    keep = counts.sum(axis=0) >= MIN_COUNTS
    counts = counts.loc[:, keep]
    meta = sub.obs.copy()
    for c in meta.columns:
        if meta[c].dtype.name == "category":
            meta[c] = meta[c].astype(str)
    inf = DefaultInference(n_cpus=n_cpus)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts, metadata=meta, design=design, refit_cooks=True,
                           inference=inf, quiet=True)
        dds.deseq2()
        ds = DeseqStats(dds, contrast=[factor, test, ref], inference=inf, quiet=True)
        ds.summary()
        mle = ds.results_df.copy()
        coeff = [c for c in ds.LFC.columns if c.startswith(factor) and test in c]
        if len(coeff) != 1:
            raise RuntimeError(f"coefficient for {factor}={test} not unique: {list(ds.LFC.columns)}")
        ds.lfc_shrink(coeff=coeff[0])
    res = ds.results_df.rename(columns={"log2FoldChange": "log2fc_shrunk", "lfcSE": "lfcse_shrunk"})
    res["log2fc_mle"] = mle["log2FoldChange"]
    res["lfcse_mle"] = mle["lfcSE"]
    res["stat"] = mle["stat"]
    res["pvalue"] = mle["pvalue"]
    res["padj"] = mle["padj"]
    det = np.asarray(sub[:, counts.columns].layers["n_detected"].sum(axis=0)).ravel()
    res["det_frac"] = det / float(sub.obs["n_nuclei"].astype(int).sum())
    res.index.name = "gene_symbol"
    res["gene_id"] = sub.var.loc[counts.columns, "gene_id"].to_numpy()
    return res.reset_index()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--types", default=None)
    p.add_argument("--n-cpus", type=int, default=16)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    types = pd.read_csv(RES / "gd1b_types.csv")
    de_types = types.loc[types["de_able"], "cell_type"].tolist()
    if a.types:
        de_types = [t for t in de_types if t in a.types.split(",")]
    pb_h = ad.read_h5ad(DATA / "islet_pseudobulk_human.h5ad")
    pb_m = ad.read_h5ad(DATA / "islet_pseudobulk_mouse.h5ad")
    pb_m = pb_m[:, (pb_m.var["measured_dbdb"] & pb_m.var["measured_mstz"]).to_numpy()].copy()
    pb_h.obs["chemistry"] = pb_h.obs["assay"].astype(str).map({"10x 3' v2": "v2", "10x 3' v3": "v3"})
    pb_m.obs["study"] = pb_m.obs["study"].astype(str).map(STUDY)
    pb_m.obs["group"] = pb_m.obs["study"] + "_" + pb_m.obs["condition"].astype(str)
    summary = []
    for t in de_types:
        hs = units_for(pb_h, t)
        ms = units_for(pb_m, t)
        jobs = [("human", "t2d_vs_nd", hs, "~ sex + chemistry + condition", "condition", "disease", "control")]
        jobs.append(("mouse", "joint", ms, "~ study + condition", "condition", "disease", "control"))
        for study in ("dbdb", "mstz"):
            g = ms.obs["group"]
            if {f"{study}_disease", f"{study}_control"} <= set(g):
                jobs.append(("mouse", study, ms, "~ group", "group", f"{study}_disease", f"{study}_control"))
        for species, contrast, sub, design, factor, test, ref in jobs:
            if species == "mouse" and factor == "group":
                # reference level = this study's control, so the shrunken coefficient is the contrast
                sub = sub.copy()
                levels = [ref] + sorted(set(sub.obs["group"]) - {ref})
                sub.obs["group"] = pd.Categorical(sub.obs["group"], categories=levels)
            log.info("%s %s %s: %d units (%s)", t, species, contrast, sub.n_obs,
                     sub.obs.groupby("condition", observed=True).size().to_dict())
            r = fit(sub, design, factor, test, ref, a.n_cpus)
            r.to_csv(OUT / f"{species}_{contrast}_{t}.tsv", sep="\t", index=False)
            summary.append(dict(cell_type=t, species=species, contrast=contrast, design=design,
                                n_units=int(sub.n_obs), units=",".join(sub.obs["sample"].astype(str)),
                                n_genes_tested=int(len(r)), n_padj_lt_0_1=int((r["padj"] < 0.1).sum()),
                                n_padj_lt_0_05=int((r["padj"] < 0.05).sum())))
            pd.DataFrame(summary).to_csv(OUT / "de_summary.tsv", sep="\t", index=False)
    skipped = types.loc[~types["de_able"], "cell_type"].tolist()
    (OUT / "de_meta.json").write_text(json.dumps(dict(de_types=de_types, filtered_types=skipped,
                                                      min_cells_unit=MIN_CELLS_UNIT, min_counts=MIN_COUNTS,
                                                      pydeseq2="0.5.2"), indent=2))
    (OUT / "DONE").write_text("ok\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
