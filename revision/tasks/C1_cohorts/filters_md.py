"""C1: section of drafts/tables/data_preparation_filters.md from the build reports, same words across cohorts. Prints markdown; the ledger holder pastes it in once.

    python -I revision/tasks/C1_cohorts/filters_md.py
"""
import json
import sys
from pathlib import Path

RES = Path(__file__).resolve().parents[3] / "results/revision/C1"
PAIR = ["n_one2one", "removed_not_high_confidence", "removed_not_protein_coding_human", "removed_not_protein_coding_partner", "removed_retrocopy_shape", "retrocopy_shape_strong_labels_kept", "removed_name_elsewhere_weak_labels", "name_elsewhere_strong_labels_kept", "n_after_filters"]


def one(c: str) -> str:
    r = json.loads((RES / c / "build_report.json").read_text())
    sp = r["species"]
    lines = [f"### {c}", "", f"Object `data/revision/new_usecases/c1_build/objects/{c}.h5ad`: {r['n_obs']:,} cells x {r['n_vars']:,} genes ({', '.join(f'{k} {v:,}' for k, v in r['per_species_cells'].items())}). {r['reference']}. Median library {r['median_library']}, depth ratio to the three-species testis cohort {r['depth_ratio_to_testis3sp']:.2f}; {r['steps_per_epoch']} steps per epoch; cap {r['cap']}; ESM features from {', '.join(r['esm_species'])}.", "", "| Step | Rule and threshold | Removed | Stated reason |", "|---|---|---|---|"]
    rep = r["species_reports"]
    lines.append(f"| Source filter | As configured per species | {'; '.join(f'{s}: {rep[s]['n_filtered_out']:,}' for s in sp)} | Same chemistry on the UMI scale; same tissue; donors kept out of DE where stated |")
    lines.append(f"| Cell QC | At least 200 genes; no mitochondrial cut in any species | {'; '.join(f'{s}: {rep[s]['n_qc_removed']:,}' for s in sp)} | One threshold cannot act evenly: mitochondrial gene names in matrices: {', '.join(f'{s} {rep[s]['n_mt_named']}' for s in sp)} |")
    for f in r["filters"]:
        lines.append(f"| {f['step']} | {f['item']} | {'; '.join(f'{k}: {v:,}' for k, v in f['counts'].items())} | {f['reason']} |")
    lines.append(f"| Cap | One number for every species x class, largest multiple of 50 with total at most 28,854; donors share each quota | {r['cap']} per species x class | Schedules count epochs; no class dominates |")
    o = r["ortholog_report"]
    for s in sp:
        if s in o:
            x = o[s]
            lines.append(f"| Orthologs, human-{s} | Compara 110 one-to-one, both dumps, high confidence, protein coding in both, no retrocopy target (one exon vs at least 3, unless GOC and WGA above 50), no target with the human name elsewhere unless GOC and WGA above 50; GOC and WGA recorded | {x['n_one2one']:,} one-to-one; removed: not high confidence {x['removed_not_high_confidence']:,}, not protein coding {x['removed_not_protein_coding_human']:,} human and {x['removed_not_protein_coding_partner']:,} partner, retrocopy shape {x['removed_retrocopy_shape']:,} (kept with strong labels {x['retrocopy_shape_strong_labels_kept']:,}), name elsewhere {x['removed_name_elsewhere_weak_labels']:,} (kept {x['name_elsewhere_strong_labels_kept']:,}); {x['n_after_filters']:,} left | Wrong pairs found by Y6 (PIN1, GJA1, RPL36AL) |")
    lines.append(f"| Join | Inner join on the human gene | {o['n_inner_join']:,} groups | All species present |")
    lines.append(f"| Measured genes | Gene in every file of its species | {o['n_not_measured']}; {o['n_groups']:,} groups left | A gene missing from a species' list is missing, not a zero |")
    lines.append(f"| Observed genes | Count > 0 over the species' units | {r['n_genes_never_observed_in_a_species']} | Structural zero |")
    lines.append(f"| Rare genes | At least 3 training cells | {r['n_genes_lt3_cells']} | Not estimable |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    for c in sys.argv[1:] or ["brain_dlpfc", "brain_m1", "brain_m1_hmouse", "brain_dlpfc_hmac_own", "lung_healthy", "tumour_zilionis", "muscle_ageing"]:
        print(one(c))
