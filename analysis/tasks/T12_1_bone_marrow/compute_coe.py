"""T12.1: recompute the Shay et al. 2013 conservation of expression (COE) and divergent set.

Rule, from the Methods of Shay et al. 2013 (PNAS 110:2946, PMC3581886), applied to the GEO
series matrices of D-MAP (GSE24759, GPL4685, log2 values) and ImmGen phase 1 (GSE15907,
GPL6246, linear RMA values, log2-transformed here):

* Orthologs: human-mouse one-to-one pairs measured on both platforms (Shay: Ensembl Compara 63;
  here Compara 110, the release of the cohort). Probesets are mapped to genes with the GEO
  platform annotations (GPL4685.annot, GPL6246.annot: "Gene symbol"), and symbols to Ensembl
  110 genes by the Ensembl 110 GTF ``gene_name``; probesets with more than one symbol, and
  symbols naming more than one Ensembl gene, are dropped. (The Ensembl 110 BioMart archive
  was down for the mouse array map; the human BioMart 110 U133A 2.0 map is used only as a
  cross-check of the human probeset mapping.) When a gene has several probesets, the one with
  the highest mean expression over the selected samples is used (Shay).
* Samples: cell types of the seven common lineages (HSPC, GN, MO, DC, NK, T, B), fixed in
  ``shay_lineage_map.tsv``. Shay's own sample table (Dataset S2: 18 human, 44 mouse cell types,
  23 human outlier arrays removed) could not be obtained, so the map is a documented
  approximation from GEO sample annotations and no outlier removal is done.
* Expression filter: a gene is kept if its expression exceeds 120 (ImmGen threshold, linear
  scale; log2 120 = 6.907) in more than three cell types of the common lineages, in each of the
  two species (cell-type value = median of its replicate arrays).
* COE: for each gene and species, the median expression (log2) over the samples of each
  lineage; COE = Pearson correlation of the two seven-vectors.
* Divergent set: COE < 0.25 and at least a fourfold difference between lineage means (log2
  range >= 2) in each species. Shay's 169 also exclude previously reported differences
  (their Dataset S3), which is not available here.

Writes ``data/analysis/bone_marrow/shay_coe.tsv`` and ``shay_coe_report.json``. Reads only the
external measurement: no Chimera output and no baseline is opened.

    python -I analysis/tasks/T12_1_bone_marrow/compute_coe.py [--out DIR]
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]
sys.path.insert(0, str(TASK))
import build_orthologs as bo  # noqa: E402

BM = REPO / "data" / "analysis" / "bone_marrow"
GEO = BM / "geo"
E = BM / "ensembl110"
SERIES = {"human": GEO / "GSE24759_series_matrix.txt.gz", "mouse": GEO / "GSE15907_series_matrix.txt.gz"}
ANNOT = {"human": GEO / "GPL4685.annot.gz", "mouse": GEO / "GPL6246.annot.gz"}
ENS_SP = {"human": "hsapiens", "mouse": "mmusculus"}
BIOMART_HUMAN = E / "biomart110_hsapiens_affy_hg_u133a_2.tsv"  # cross-check only
LINEAGES = ["HSPC", "GN", "MO", "DC", "NK", "T", "B"]
THRESH_LOG2 = float(np.log2(120.0))
MIN_TYPES = 3          # "more than three cell types"
COE_MAX = 0.25
LOG2_FC = 2.0          # fourfold
NAMED = {"DPP4": "B", "NT5E": "B", "CD27": "NK", "TFEC": "B", "LMO2": "B", "IL15": "GN,MO,B",
         "MYBL1": "NK,T", "LDB2": "HSPC,NK", "NFIX": "HSPC,MO", "PRDM1": "MO,NK"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 24):
            h.update(b)
    return h.hexdigest()


def read_series(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    meta: dict[str, list] = {}
    rows, header = [], None
    with gzip.open(path, "rt", encoding="utf8", errors="replace") as fh:
        intable = False
        for line in fh:
            if line.startswith("!series_matrix_table_begin"):
                intable = True
                header = [x.strip('"') for x in next(fh).rstrip("\n").split("\t")]
                continue
            if line.startswith("!series_matrix_table_end"):
                break
            if intable:
                rows.append(line.rstrip("\n").split("\t"))
            elif line.startswith("!Sample_"):
                k = line.split("\t")[0]
                meta.setdefault(k, []).append([x.strip('"') for x in line.rstrip("\n").split("\t")[1:]])
    probes = [r[0].strip('"') for r in rows]
    vals = np.array([[float(x) if x.strip('"') not in ("", "null", "NA") else np.nan for x in r[1:]] for r in rows])
    X = pd.DataFrame(vals, index=probes, columns=header[1:])
    title = meta["!Sample_title"][0]
    chars = meta["!Sample_characteristics_ch1"]
    ct = next((c for c in chars if c[0].startswith("cell type:")), None)
    samples = pd.DataFrame({"gsm": meta["!Sample_geo_accession"][0], "title": title}, index=header[1:])
    if ct is not None:
        samples["cell_type"] = [c.split(":", 1)[1].strip() for c in ct]
    else:
        samples["cell_type"] = [t.split("#")[0] for t in title]
    return X, samples


def read_annot(path: Path) -> pd.DataFrame:
    with gzip.open(path, "rt", encoding="utf8", errors="replace") as fh:
        lines = fh.read().split("\n")
    i = lines.index("!platform_table_begin")
    j = lines.index("!platform_table_end")
    hdr = lines[i + 1].split("\t")
    rows = [ln.split("\t") for ln in lines[i + 2:j]]
    return pd.DataFrame(rows, columns=hdr)[["ID", "Gene symbol"]]


_NAMES: dict = {}


def probe_map(sp: str) -> tuple[pd.Series, dict]:
    """Probeset -> Ensembl 110 gene via the GEO symbol; ambiguous probesets and symbols dropped."""
    a = read_annot(ANNOT[sp])
    a = a[a["Gene symbol"].str.len() > 0]
    single = ~a["Gene symbol"].str.contains("///", regex=False)
    if ENS_SP[sp] not in _NAMES:
        _NAMES[ENS_SP[sp]] = bo.gtf_names(bo.GTF[ENS_SP[sp]])
    names = pd.Series(_NAMES[ENS_SP[sp]])
    counts = names.value_counts()
    uniq = names[names.map(counts) == 1]
    sym2ens = pd.Series(uniq.index, index=uniq.values)
    a1 = a[single]
    gene = a1["Gene symbol"].map(sym2ens)
    out = pd.Series(gene.values, index=a1["ID"].values).dropna()
    rec = {"probes_with_symbol": int(len(a)), "probes_single_symbol": int(single.sum()),
           "probes_mapped_to_ensembl110": int(len(out))}
    return out, rec


def biomart_agreement(pm: pd.Series) -> dict:
    """Share of human probesets whose GEO-symbol gene is among the BioMart 110 genes of the probeset."""
    if not BIOMART_HUMAN.exists():
        return {"checked": False}
    b = pd.read_csv(BIOMART_HUMAN, sep="\t", dtype=str).dropna()
    b.columns = ["gene", "probe"]
    bm = b.groupby("probe")["gene"].apply(set)
    common = pm.index.intersection(bm.index)
    agree = np.mean([pm[p] in bm[p] for p in common]) if len(common) else float("nan")
    return {"checked": True, "n_probes_both": int(len(common)), "frac_agree": float(agree)}


def species_matrix(sp: str, lmap: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    X, samples = read_series(SERIES[sp])
    m = lmap[lmap["species"] == sp].set_index("cell_type")["lineage"]
    missing = sorted(set(m.index) - set(samples["cell_type"]))
    if missing:
        raise ValueError(f"{sp}: cell types in map but not in series: {missing}")
    samples = samples[samples["cell_type"].isin(m.index)].copy()
    samples["lineage"] = samples["cell_type"].map(m)
    X = X[samples.index]
    if sp == "mouse":
        X = np.log2(X.clip(lower=1.0))
    pm, prec = probe_map(sp)
    rec = {"n_probes": int(len(X)), "n_samples": int(X.shape[1]), "n_cell_types": int(samples["cell_type"].nunique()),
           "samples_per_lineage": samples["lineage"].value_counts().reindex(LINEAGES).astype(int).to_dict(),
           "probe_annotation": prec, "n_probes_mapped_unique_gene": int(X.index.isin(pm.index).sum())}
    if sp == "human":
        rec["biomart110_crosscheck"] = biomart_agreement(pm)
    X = X[X.index.isin(pm.index)]
    gene = pm.reindex(X.index)
    mean = X.mean(axis=1)
    best = mean.groupby(gene).idxmax()  # probeset with highest mean expression per gene
    G = X.loc[best.values]
    G.index = best.index
    rec["n_genes"] = int(len(G))
    probe_of_gene = pd.Series(best.values, index=best.index)
    return G, samples, {"rec": rec, "probe": probe_of_gene}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=BM)
    args = ap.parse_args()
    lmap = pd.read_csv(TASK / "shay_lineage_map.tsv", sep="\t")
    report: dict = {"rule": __doc__.split("Writes")[0].strip(), "thresh_log2": THRESH_LOG2}

    # Human-mouse one-to-one orthologs, Compara 110.
    df, _ = bo.read_compara()
    o = df[(df["homology_species"] == "mus_musculus") & (df["homology_type"] == "ortholog_one2one")]
    o = o[["gene_stable_id", "homology_gene_stable_id"]].drop_duplicates()
    for c in o.columns:
        o = o[~o[c].duplicated(keep=False)]
    o.columns = ["hsapiens_gene_id", "mmusculus_gene_id"]
    report["n_one2one_compara110"] = int(len(o))

    mats, samples, extra = {}, {}, {}
    for sp in ("human", "mouse"):
        mats[sp], samples[sp], extra[sp] = species_matrix(sp, lmap)
        report[sp] = extra[sp]["rec"]

    o = o[o["hsapiens_gene_id"].isin(mats["human"].index) & o["mmusculus_gene_id"].isin(mats["mouse"].index)]
    o = o.sort_values("hsapiens_gene_id").reset_index(drop=True)
    report["n_one2one_measured_both"] = int(len(o))

    def per_lineage(sp: str, ids: pd.Series, how: str) -> pd.DataFrame:
        G = mats[sp].loc[ids.values]
        lin = samples[sp]["lineage"]
        out = pd.DataFrame({L: getattr(G.loc[:, lin.index[lin == L]], how)(axis=1).to_numpy() for L in LINEAGES})
        return out

    def n_types_above(sp: str, ids: pd.Series) -> np.ndarray:
        G = mats[sp].loc[ids.values]
        ct = samples[sp]["cell_type"]
        med = pd.DataFrame({c: G.loc[:, ct.index[ct == c]].median(axis=1).to_numpy() for c in ct.unique()})
        return (med.to_numpy() > THRESH_LOG2).sum(axis=1)

    def n_types_variant(sp: str, ids: pd.Series, how: str) -> np.ndarray:
        G = mats[sp].loc[ids.values]
        ct = samples[sp]["cell_type"]
        if how == "samples":
            return (G.to_numpy() > THRESH_LOG2).sum(axis=1)
        if how == "mean_linear":
            v = pd.DataFrame({c: (2.0 ** G.loc[:, ct.index[ct == c]]).mean(axis=1).to_numpy() for c in ct.unique()})
            return (v.to_numpy() > 120.0).sum(axis=1)
        f = {"median_log2": "median", "mean_log2": "mean", "max_log2": "max"}[how]
        v = pd.DataFrame({c: getattr(G.loc[:, ct.index[ct == c]], f)(axis=1).to_numpy() for c in ct.unique()})
        return (v.to_numpy() > THRESH_LOG2).sum(axis=1)

    # Disclosure only: gene counts under other readings of the filter sentence (counts, no COE).
    readings = {}
    for how in ("median_log2", "mean_log2", "mean_linear", "max_log2", "samples"):
        a_ = n_types_variant("human", o["hsapiens_gene_id"], how) > MIN_TYPES
        b_ = n_types_variant("mouse", o["mmusculus_gene_id"], how) > MIN_TYPES
        readings[how] = {"both": int((a_ & b_).sum()), "either": int((a_ | b_).sum()),
                         "human_only_condition": int(a_.sum()), "mouse_only_condition": int(b_.sum())}
    report["filter_readings_gene_counts (disclosure; primary = median_log2/both)"] = readings

    nh = n_types_above("human", o["hsapiens_gene_id"])
    nm = n_types_above("mouse", o["mmusculus_gene_id"])
    keep = (nh > MIN_TYPES) & (nm > MIN_TYPES)
    report["n_pass_filter_both_species"] = int(keep.sum())
    report["n_pass_filter_either_species (reported only)"] = int(((nh > MIN_TYPES) | (nm > MIN_TYPES)).sum())
    report["coe_gene_count_target"] = 5841
    report["coe_gene_count_within_10pct"] = bool(abs(int(keep.sum()) - 5841) <= 584.1)

    o = o[keep].reset_index(drop=True)
    hmed = per_lineage("human", o["hsapiens_gene_id"], "median")
    mmed = per_lineage("mouse", o["mmusculus_gene_id"], "median")
    hmean = per_lineage("human", o["hsapiens_gene_id"], "mean")
    mmean = per_lineage("mouse", o["mmusculus_gene_id"], "mean")
    a = hmed.to_numpy() - hmed.to_numpy().mean(axis=1, keepdims=True)
    b = mmed.to_numpy() - mmed.to_numpy().mean(axis=1, keepdims=True)
    coe = (a * b).sum(axis=1) / np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
    h_range = hmean.max(axis=1) - hmean.min(axis=1)
    m_range = mmean.max(axis=1) - mmean.min(axis=1)
    divergent = (coe < COE_MAX) & (h_range.to_numpy() >= LOG2_FC) & (m_range.to_numpy() >= LOG2_FC)

    names = {k: _NAMES.get(k) or bo.gtf_names(bo.GTF[k]) for k in ("hsapiens", "mmusculus")}
    bm_table = pd.read_csv(BM / "ortholog_table_bone_marrow.csv", usecols=["hsapiens_gene_id"])
    res = pd.DataFrame({
        "hsapiens_gene_id": o["hsapiens_gene_id"],
        "hsapiens_gene_name": o["hsapiens_gene_id"].map(names["hsapiens"]),
        "mmusculus_gene_id": o["mmusculus_gene_id"],
        "mmusculus_gene_name": o["mmusculus_gene_id"].map(names["mmusculus"]),
        "coe": coe,
        "divergent": divergent,
        "human_log2_range_lineage_means": h_range.to_numpy(),
        "mouse_log2_range_lineage_means": m_range.to_numpy(),
        "human_probeset": extra["human"]["probe"].reindex(o["hsapiens_gene_id"]).to_numpy(),
        "mouse_probeset": extra["mouse"]["probe"].reindex(o["mmusculus_gene_id"]).to_numpy(),
        "in_bone_marrow_table": o["hsapiens_gene_id"].isin(bm_table["hsapiens_gene_id"]).to_numpy(),
    })
    for L in LINEAGES:
        res[f"human_median_{L}"] = hmed[L].to_numpy()
    for L in LINEAGES:
        res[f"mouse_median_{L}"] = mmed[L].to_numpy()
    report["n_coe"] = int(len(res))
    report["n_divergent"] = int(res["divergent"].sum())
    report["n_divergent_in_bone_marrow_table"] = int((res["divergent"] & res["in_bone_marrow_table"]).sum())
    report["n_coe_in_bone_marrow_table"] = int(res["in_bone_marrow_table"].sum())
    report["coe_median"] = float(np.median(coe))
    report["frac_coe_gt_0"] = float((coe > 0).mean())
    named = res[res["hsapiens_gene_name"].isin(NAMED)].set_index("hsapiens_gene_name")
    report["named_genes"] = {g: ({"coe": float(named.loc[g, "coe"]), "divergent": bool(named.loc[g, "divergent"]),
                                  "shay_lineage": NAMED[g]} if g in named.index else "not in COE set") for g in NAMED}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "shay_coe.tsv"
    res.to_csv(path, sep="\t", index=False, float_format="%.6g")
    report["inputs"] = {p.name: sha256(p) for p in [*SERIES.values(), *ANNOT.values(), BIOMART_HUMAN,
                                                    bo.GTF["hsapiens"], bo.GTF["mmusculus"],
                                                    TASK / "shay_lineage_map.tsv"]}
    report["output"] = {path.name: sha256(path)}
    (args.out / "shay_coe_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({k: v for k, v in report.items() if k != "rule"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
