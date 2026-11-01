#!/usr/bin/env bash
# T4.1 gold items G1, G7, G8 (see fetch_ensembl.sh), G9, G10. Usage: fetch_gold.sh <chimera_repo>
C="$1"; F="$C/revision/tasks/T4_1_gold/fetch_item.sh"; G="$C/data/revision/gold"
"$F" G1 "https://web.archive.org/web/20220226202014/https://portals.broadinstitute.org/evee/static/data/evolutionary_statistics.zip" "$G/G1/evolutionary_statistics.zip" "$G/G1/MANIFEST.tsv"
"$F" G9 "https://www.tau.ac.il/~elieis/HKG/HK_genes.txt" "$G/G9/HK_genes.txt" "$G/G9/MANIFEST.tsv"
"$F" G7 "https://storage.googleapis.com/gcp-public-data--gnomad/release/4.1/constraint/gnomad.v4.1.constraint_metrics.tsv" "$G/G7/gnomad.v4.1.constraint_metrics.tsv" "$G/G7/MANIFEST.tsv"
# G10: HGNC groups (complete set carries gene_group and gene_group_id), GO extracellular
# region (GO:0005576) from the human GAF plus the ontology, Hallmark from Enrichr
# (copied from the functional folder, which holds the verbatim server response).
"$F" G10 "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt" "$G/G10/hgnc_complete_set.txt" "$G/G10/MANIFEST.tsv"
"$F" G10 "https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/family.csv" "$G/G10/hgnc_family.csv" "$G/G10/MANIFEST.tsv"
"$F" G10 "https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/hierarchy.csv" "$G/G10/hgnc_hierarchy.csv" "$G/G10/MANIFEST.tsv"
"$F" G10 "https://storage.googleapis.com/public-download-files/hgnc/csv/csv/genefamily_db_tables/gene_has_family.csv" "$G/G10/hgnc_gene_has_family.csv" "$G/G10/MANIFEST.tsv"
"$F" G10 "https://current.geneontology.org/annotations/goa_human.gaf.gz" "$G/G10/goa_human.gaf.gz" "$G/G10/MANIFEST.tsv"
"$F" G10 "https://purl.obolibrary.org/obo/go/go-basic.obo" "$G/G10/go-basic.obo" "$G/G10/MANIFEST.tsv"
