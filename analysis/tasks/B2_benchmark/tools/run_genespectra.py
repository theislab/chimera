"""GeneSpectra adapter (Papatheodorou-Group/GeneSpectra). Run in genespectra_env.

PRIMARY SCALAR, declared before any comparison:
    score = H(c, c) if the gene has the same specificity class c in Human and Mouse, else 0, where
    H is the tool's gene class conservation heatmap (plot_cross_species_spec_category_heatmap,
    group_enhanced=False): the harmonic mean, over the two species, of the percentage of the
    genes of a class that fall in that class pair. Range 0-100, higher = more conserved.
    Reason: the heatmap is the only cross-species number of the tool's own cross_species module; its
    diagonal says how conserved a shared class is, and a gene that changed class did not conserve it.

Workflow: the README example (run_classification_sum_cell_pools.py) with its defaults: sum counts
per label within a species, normalize each pool to 1e6, filter_low_counts(min_count=1), classify with
exp_lim=1, enr_fold=4, max_group_n=half of the labels. The tool's Ensembl BioMart query is replaced
by the 1:1 ortholog group of the input (var_name), same merge. A gene is covered if it passes the
filter in both species. Labels are compared within a species only, so they need not be shared.

Secondary columns: class_pair_overlap is the heatmap value of the gene's own class pair, on or off
the diagonal; dist_pair_overlap is the same for the distribution classes; spec_cosine is the cosine
of the spec_score vectors over labels named identically in both species, as in the authors'
cos_sim_homologs notebooks (enriched genes with enriched genes, enhanced with enhanced), NaN elsewhere
and where a spec_score is infinite (no other label expresses the gene).

The tool is deterministic, --seed is recorded only. --cohort is recorded only, var_name is the join.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import b2t_common as C  # noqa: E402

CPUS = C.n_cpus()
C.set_threads(1)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

TOOL_DIR = C.tool_dir("GeneSpectra")
sys.path.insert(0, str(TOOL_DIR))

import anndata as ad  # noqa: E402
import scanpy as sc  # noqa: E402
from genespectra.cross_species.plot_cross_species_categories import (  # noqa: E402
    plot_cross_species_dist_category_heatmap,
    plot_cross_species_spec_category_heatmap,
)
from genespectra.gene_classification.classify_genes import (  # noqa: E402
    ExpressionDataLong,
    GeneClassificationResult,
)
from genespectra.metacells.make_metacells import SummedAnnData  # noqa: E402

LABEL = "b2_label"
PARAMS = dict(pooling="sum per label", target_sum=1_000_000, filter_min_count=1, exp_lim=1, enr_fold=4,
              max_group_n=None, num_gene_batches=10, group_enhanced=False)
ENRICHED = {"cell type enriched", "group enriched"}
ENHANCED = {"cell type enhanced", "group enhanced"}


def classify(X, var_names, labels):
    keep = pd.notna(labels)
    obs = pd.DataFrame({LABEL: labels[keep].astype(str)})
    obs.index = obs.index.astype(str)
    adata = ad.AnnData(X=X[keep].astype(np.float32), obs=obs, var=pd.DataFrame(index=var_names))
    summed = SummedAnnData.create_from_summing_anndata(adata, annotation_col=LABEL)
    summed = SummedAnnData.depth_normalize_counts(summed, target_sum=PARAMS["target_sum"])
    sc.pp.calculate_qc_metrics(summed, log1p=False, inplace=True)
    summed = SummedAnnData.filter_low_counts(summed, min_count=PARAMS["filter_min_count"])
    expr = ExpressionDataLong.create_from_summed_adata(input_summed_adata=summed, anno_col=LABEL)
    res = GeneClassificationResult.create_from_expression_data_long_multiprocess(
        expr, num_gene_batches=PARAMS["num_gene_batches"], max_group_n=PARAMS["max_group_n"],
        exp_lim=PARAMS["exp_lim"], enr_fold=PARAMS["enr_fold"], num_process=CPUS)
    return pd.DataFrame(res).reset_index(drop=True)


def heatmap_formula(pairs, col_h, col_m):
    """The tool's harmonic-mean formula, for objects where its plotting step cannot run."""
    df = pairs.groupby(col_h)[col_m].value_counts().to_frame("count").reset_index()
    a = df.groupby(col_h)["count"].sum()
    b = df.groupby(col_m)["count"].sum()
    df["p1"] = df["count"] / df[col_h].map(a) * 100
    df["p2"] = df["count"] / df[col_m].map(b) * 100
    df["hm"] = 2 / ((1 / df["p1"]) + (1 / df["p2"]))
    return np.round(df.pivot(columns=col_h, index=col_m, values="hm").astype("float32"), 2)


def merge_enhanced(s):
    return s.replace({"group enhanced": "enhanced", "cell type enhanced": "enhanced"})


def conservation_heatmap(both, kind):
    """Heatmap of the tool; falls back to its formula if a class is absent from this object."""
    col_h, col_m = f"{kind}_category_Human", f"{kind}_category_Mouse"
    pairs = both.copy()
    if kind == "spec":
        pairs[col_h], pairs[col_m] = merge_enhanced(pairs[col_h]), merge_enhanced(pairs[col_m])
    own = heatmap_formula(pairs, col_h, col_m)
    try:
        if kind == "spec":
            heat, _ = plot_cross_species_spec_category_heatmap(both.copy(), "Human", "Mouse", group_enhanced=False)
        else:
            heat, _ = plot_cross_species_dist_category_heatmap(both.copy(), "Human", "Mouse")
        source = "tool"
        diff = (heat.sort_index(axis=0).sort_index(axis=1) - own.sort_index(axis=0).sort_index(axis=1)).abs().max().max()
        if diff > 0.011:
            raise SystemExit(f"heatmap formula differs from the tool by {diff}")
    except KeyError as e:  # the tool's plot indexes every class by name
        heat, source = own, f"formula of the tool (a class is absent, its plot raises KeyError: {e})"
    return heat, pairs[col_h].values, pairs[col_m].values, source


def lookup(heat, h, m):
    idx_m = heat.index.get_indexer(m)
    idx_h = heat.columns.get_indexer(h)
    return heat.to_numpy(dtype=float)[idx_m, idx_h]


def spec_vectors(table, shared):
    """gene -> {label: spec_score} over shared labels, as in the authors' notebooks."""
    out = {}
    for gene, groups, score in zip(table["gene"], table["specific_groups"], table["spec_score"]):
        if not isinstance(groups, str) or not groups:
            continue
        v = {g: float(score) for g in groups.split(";") if g in shared}
        if v:
            out[gene] = v
    return out


def spec_cosine(both, labels_h, labels_m):
    shared = set(labels_h) & set(labels_m)
    h = both.rename(columns={"gene": "gene", "specific_groups_Human": "specific_groups", "spec_score_Human": "spec_score"})
    m = both.rename(columns={"gene": "gene", "specific_groups_Mouse": "specific_groups", "spec_score_Mouse": "spec_score"})
    vh, vm = spec_vectors(h, shared), spec_vectors(m, shared)
    out = np.full(len(both), np.nan)
    for i, (g, ch, cm) in enumerate(zip(both["gene"], both["spec_category_Human"], both["spec_category_Mouse"])):
        same_family = (ch in ENRICHED and cm in ENRICHED) or (ch in ENHANCED and cm in ENHANCED)
        if same_family and g in vh and g in vm:
            a, b = vh[g], vm[g]
            with np.errstate(invalid="ignore"):  # spec_score is infinite if no other label expresses the gene
                dot = sum(a[k] * b[k] for k in a if k in b)
                out[i] = dot / (np.sqrt(sum(x * x for x in a.values())) * np.sqrt(sum(x * x for x in b.values())))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--label-key", required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--work", default=None, help="not used by this tool")
    a = p.parse_args(argv)
    t0 = time.time()

    X, var_names, species, labels = C.read_h5ad(a.h5ad, a.label_key)
    tables = {}
    for s in C.SPECIES:
        m = species == s
        if len(set(labels[m & pd.notna(labels)])) < 2:
            raise SystemExit(f"{s}: GeneSpectra needs at least two labels within a species")
        tables[s] = classify(X[m], var_names, labels[m])
        print(s, tables[s]["spec_category"].value_counts().to_dict(), flush=True)

    both = tables["Human"].merge(tables["Mouse"], on="gene", suffixes=("_Human", "_Mouse"))
    both["Mouse_homolog_orthology_type"] = "ortholog_one2one"
    spec_heat, ch, cm, spec_source = conservation_heatmap(both, "spec")
    dist_heat, dh, dm, dist_source = conservation_heatmap(both, "dist")
    overlap = lookup(spec_heat, ch, cm)
    same = ch == cm
    out = pd.DataFrame({
        "var_name": both["gene"].astype(str).values,
        "score": np.where(same, overlap, 0.0),
        "class_pair_overlap": overlap,
        "same_spec_class": same.astype(int),
        "spec_class_human": ch,
        "spec_class_mouse": cm,
        "spec_category_human": both["spec_category_Human"].values,
        "spec_category_mouse": both["spec_category_Mouse"].values,
        "dist_category_human": dh,
        "dist_category_mouse": dm,
        "dist_pair_overlap": lookup(dist_heat, dh, dm),
        "spec_score_human": both["spec_score_Human"].values,
        "spec_score_mouse": both["spec_score_Mouse"].values,
        "tau_human": both["tau_Human"].values,
        "tau_mouse": both["tau_Mouse"].values,
        "specific_groups_human": both["specific_groups_Human"].values,
        "specific_groups_mouse": both["specific_groups_Mouse"].values,
        "spec_cosine": spec_cosine(both, set(labels[(species == "Human") & pd.notna(labels)]),
                                   set(labels[(species == "Mouse") & pd.notna(labels)])),
    })
    C.write_outputs(
        a.out, out, tool="GeneSpectra", tool_info=dict(C.tool_state(TOOL_DIR), repo="Papatheodorou-Group/GeneSpectra", version="0.2.0"),
        primary="score = heatmap diagonal H(c,c) if the specificity class is the same in both species, else 0",
        params=PARAMS, args=a, t0=t0, n_input_genes=var_names.size,
        extra={"label_cells": C.label_table(species, labels), "n_cells_dropped_missing_label": int(pd.isna(labels).sum()),
               "genes_classified": {s: int(len(t)) for s, t in tables.items()},
               "spec_heatmap": spec_heat.to_dict(), "spec_heatmap_source": spec_source, "dist_heatmap_source": dist_source,
               "seed_used": False})


if __name__ == "__main__":
    main()
