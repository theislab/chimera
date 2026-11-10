"""T7.9: pseudobulk differential expression for the islet T2D case (plan section 8b).

Runs in ``revision_de_env`` (PyDESeq2 0.5.2), never ``chimera_env``. No Chimera readout is read.

Per DE-able type t (``results/analysis/T7_9/gd1b_types.csv``), on the DE units of
``donor_roles.tsv`` that carry >= 20 cells of t (the G-D1b unit filter):

* human: ``~ sex + chemistry + condition``, T2D vs ND DE half;
* mouse, primary: ``~ study + condition`` on the 6 untreated libraries (db/db 2+2, mSTZ 1+1);
* mouse, per model (Family 3): one factor ``group`` = study x condition, so dispersion is shared;
  db/db = dbdb_disease vs dbdb_control, mSTZ = mstz_disease vs mstz_control (1 vs 1, descriptive).

Wald tests; PyDESeq2's apeGLM prior on the tested coefficient (``lfc_shrink``), twice:
``log2fc_shrunk`` with the adaptive prior and ``log2fc_fixed`` with prior scale 1 (T7.3). The adaptive
prior collapses the human fits (92-99.8% of genes at |log2FC| < 0.001 in delta, ductal, endothelial
and stellate), so l^h is the unshrunk ``log2fc_mle``, m stays adaptive as the plan says, and H-D4
ranks on ``log2fc_fixed`` in both species. Mouse fits use only genes
measured in both DE studies (``var['measured_dbdb']`` and ``var['measured_mstz']``; elsewhere the
atlas pads a gene missing from a study's reference with zeros). Genes with < 10 counts summed over
the units are not tested. ``det_frac`` = cells with count > 0 / cells, over the units
used, for section 8b's 5% detection rule.

Permuted labels (``--perm``, for the H-D4 calibration and the section 8b controls): human condition
labels within sex x chemistry over the 26 DE donors, 20 draws, the same donor labels in every type;
mouse labels within study, all 11 relabellings other than the true one. Only ``log2fc_fixed`` is
shrunk there. Written to ``de/perm/``.

    PYTHONPATH=<chimera repo> python analysis/tasks/T7_9_islet/de.py [--types beta,...] [--n-cpus 16]
        [--perm]
    PYTHONPATH=<chimera repo> python analysis/tasks/T7_9_islet/de.py --loo <type index>   (loo_array.sbatch)
"""

from __future__ import annotations

import argparse
import itertools
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

from analysis.common import gene_ids

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("T7_9_de")

ROOT = gene_ids.REPO_ROOT
DATA = ROOT / "data" / "analysis" / "disease_islet"
RES = ROOT / "results" / "analysis" / "T7_9"
OUT = RES / "de"
MIN_CELLS_UNIT = 20
MIN_COUNTS = 10
N_PERM_HUMAN = 20
SEED = 20261102
STUDY = {"db/db": "dbdb", "mSTZ": "mstz", "human48": "human48"}


def units_for(pb: ad.AnnData, t: str) -> ad.AnnData:
    o = pb.obs
    keep = (o["cell_ontology_class"] == t) & o["role"].isin(["de_control", "de_disease"]) \
        & (o["n_nuclei"].astype(int) >= MIN_CELLS_UNIT)
    return pb[keep.to_numpy()].copy()


def fit(sub: ad.AnnData, design: str, factor: str, test: str, ref: str, n_cpus: int,
        adaptive: bool = True) -> pd.DataFrame:
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
        # lfc_shrink edits the LFC table in place, so each prior needs its own DeseqStats
        ds2 = DeseqStats(dds, contrast=[factor, test, ref], inference=inf, quiet=True)
        ds2.summary()
        ds2.lfc_shrink(coeff=coeff[0], adapt=False)
        if adaptive:
            ds.lfc_shrink(coeff=coeff[0])
    if adaptive:
        res = ds.results_df.rename(columns={"log2FoldChange": "log2fc_shrunk", "lfcSE": "lfcse_shrunk"})
    else:
        res = mle[["baseMean"]].copy()
    res["log2fc_fixed"] = ds2.results_df["log2FoldChange"]
    res["lfcse_fixed"] = ds2.results_df["lfcSE"]
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


def human_label_draws(obs: pd.DataFrame) -> list:
    """Per draw, donor -> condition, permuted within sex x chemistry over all DE donors."""
    d = obs[obs["role"].isin(["de_control", "de_disease"])].drop_duplicates("donor").set_index("donor")
    d = d[["sex", "chemistry", "condition"]].astype(str).sort_index()
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(N_PERM_HUMAN):
        lab = d["condition"].copy()
        for _, g in d.groupby(["sex", "chemistry"], sort=True):
            lab[g.index] = rng.permutation(g["condition"].to_numpy())
        draws.append(lab.to_dict())
    return draws


def mouse_label_draws(obs: pd.DataFrame) -> list:
    """All within-study relabellings that keep each study's arm sizes, except the true labels."""
    d = obs[obs["role"].isin(["de_control", "de_disease"])].drop_duplicates("sample").set_index("sample")
    d = d[["study", "condition"]].astype(str).sort_index()
    per = []
    for _, g in d.groupby("study", sort=True):
        k = int((g["condition"] == "disease").sum())
        per.append([{u: ("disease" if u in c else "control") for u in g.index}
                    for c in itertools.combinations(g.index, k)])
    true = d["condition"].to_dict()
    return [m for m in ({**a, **b} for a, b in itertools.product(*per)) if m != true]


def run_perm(pb_h: ad.AnnData, pb_m: ad.AnnData, types: list, n_cpus: int) -> None:
    out = OUT / "perm"
    out.mkdir(parents=True, exist_ok=True)
    hd, md = human_label_draws(pb_h.obs), mouse_label_draws(pb_m.obs)
    pd.DataFrame(hd).assign(kind="human").to_csv(out / "labels_human.tsv", sep="\t", index_label="draw")
    pd.DataFrame(md).assign(kind="mouse").to_csv(out / "labels_mouse.tsv", sep="\t", index_label="draw")
    for t in types:
        for species, draws, base, key, design in (
                ("human", hd, units_for(pb_h, t), "donor", "~ sex + chemistry + condition"),
                ("mouse", md, units_for(pb_m, t), "sample", "~ study + condition")):
            for i, lab in enumerate(draws):
                path = out / f"{species}_{i:02d}_{t}.tsv"
                if path.exists():
                    continue
                sub = base.copy()
                sub.obs["condition"] = sub.obs[key].astype(str).map(lab).to_numpy()
                if sub.obs["condition"].nunique() < 2:
                    continue
                r = fit(sub, design, "condition", "disease", "control", n_cpus, adaptive=False)
                tmp = path.with_name(path.name + ".tmp")
                r.to_csv(tmp, sep="\t", index=False)
                tmp.replace(path)
                log.info("perm %s", path.name)
    (out / f"DONE_{'_'.join(types)}").write_text("ok\n")


def run_loo(pb_h: ad.AnnData, pb_m: ad.AnnData, t: str, n_cpus: int) -> None:
    """Leave-one-out primary fits for the H-D1 sensitivity (section 8b): each ND DE-half donor
    (human, ``~ sex + chemistry + condition``) and each mouse DE library (``~ study + condition``;
    ``~ condition`` when the remaining libraries come from one study). A fit needs >= 2 units per
    arm. Human fits skip the adaptive prior (l^h is the MLE); mouse fits keep it (m is adaptive).
    Output ``de/loo/{human,mouse}_loo_<unit>_<type>.tsv`` and the index ``de/loo/loo_<type>.tsv``.
    """
    out = OUT / "loo"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for species, base, key, design in (("human", units_for(pb_h, t), "donor", "~ sex + chemistry + condition"),
                                       ("mouse", units_for(pb_m, t), "sample", "~ study + condition")):
        units = base.obs[key].astype(str)
        left = sorted(units[base.obs["condition"] == "control"].unique()) if species == "human" \
            else sorted(units.unique())
        for u in left:
            sub = base[(units != u).to_numpy()].copy()
            n = sub.obs["condition"].value_counts()
            safe = u.replace("/", "").replace("|", "_").replace(" ", "")
            row = dict(cell_type=t, species=species, left_out=u, file="", n_units=int(sub.n_obs),
                       n_control=int(n.get("control", 0)), n_disease=int(n.get("disease", 0)))
            if n.get("control", 0) < 2 or n.get("disease", 0) < 2:
                rows.append({**row, "design": "skipped: < 2 units in an arm"})
                continue
            d = design
            if species == "mouse" and sub.obs["study"].nunique() < 2:
                d = "~ condition"
            try:
                r = fit(sub, d, "condition", "disease", "control", n_cpus, adaptive=(species == "mouse"))
            except (np.linalg.LinAlgError, ValueError) as e:  # singular design after the removal
                rows.append({**row, "design": f"failed: {e}"})
                log.info("%s %s leave out %s: %s", t, species, u, e)
                continue
            f = f"{species}_loo_{safe}_{t}.tsv"
            r.to_csv(out / f, sep="\t", index=False)
            rows.append({**row, "file": f, "design": d})
            pd.DataFrame(rows).to_csv(out / f"loo_{t}.tsv", sep="\t", index=False)
            log.info("%s %s leave out %s done (%s)", t, species, u, d)
    pd.DataFrame(rows).to_csv(out / f"loo_{t}.tsv", sep="\t", index=False)
    (out / f"DONE_{t}").write_text("ok\n")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--types", default=None)
    p.add_argument("--n-cpus", type=int, default=16)
    p.add_argument("--perm", action="store_true")
    p.add_argument("--loo", type=int, default=None, help="index into the DE-able types: leave-one-out primary fits")
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
    if a.perm:
        run_perm(pb_h, pb_m, de_types, a.n_cpus)
        return 0
    if a.loo is not None:
        run_loo(pb_h, pb_m, de_types[a.loo], a.n_cpus)
        return 0
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
