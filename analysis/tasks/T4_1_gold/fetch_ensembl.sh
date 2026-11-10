#!/usr/bin/env bash
# Ensembl files of the covariate audit (T2_4_covariates) and of the independent references. Pinned releases, FTP tree, not BioMart.
# Usage: fetch_ensembl.sh <chimera_repo>
C="$1"; F="$C/analysis/tasks/T4_1_gold/fetch_item.sh"
E=https://ftp.ensembl.org/pub
D="$C/data/analysis/ensembl110"; M="$D/MANIFEST.tsv"
"$F" G13 $E/release-110/gtf/homo_sapiens/Homo_sapiens.GRCh38.110.gtf.gz "$D/Homo_sapiens.GRCh38.110.gtf.gz" "$M"
"$F" G13 $E/release-110/gtf/homo_sapiens/CHECKSUMS "$D/CHECKSUMS.gtf" "$M"
"$F" G13 $E/release-110/fasta/homo_sapiens/cds/Homo_sapiens.GRCh38.cds.all.fa.gz "$D/Homo_sapiens.GRCh38.cds.all.fa.gz" "$M"
"$F" G13 $E/release-110/fasta/homo_sapiens/cds/CHECKSUMS "$D/CHECKSUMS.cds" "$M"
"$F" E110C $E/release-110/tsv/ensembl-compara/homologies/homo_sapiens/Compara.110.protein_default.homologies.tsv.gz "$D/Compara.110.protein_default.homologies.tsv.gz" "$M"
D="$C/data/analysis/gold/G8"; M="$D/MANIFEST.tsv"
"$F" G8 $E/release-102/tsv/ensembl-compara/homologies/homo_sapiens/Compara.102.protein_default.homologies.tsv.gz "$D/Compara.102.protein_default.homologies.tsv.gz" "$M"
# Chicken: Murat var IDs are GRCg6a-era ENSGALG IDs. Ensembl 110 gallus_gallus is GRCg7b with new
# IDs (0 of 10,725 Murat chicken IDs found). Release 105 is the last with gallus_gallus = GRCg6a.
D="$C/data/analysis/ensembl110"; M="$D/MANIFEST.tsv"
"$F" E105C $E/release-105/tsv/ensembl-compara/homologies/homo_sapiens/Compara.105.protein_default.homologies.tsv.gz "$D/Compara.105.protein_default.homologies.tsv.gz" "$M"
# dN/dS: release 102 (and 100, 101, 110) have the dn, ds columns but every value is NULL.
# Release 99 (GRCh38, January 2020) is the last with values; H8 uses it
# (amendment log of the pre-specified analysis plan, 2026-11-01).
D="$C/data/analysis/gold/G8"; M="$D/MANIFEST.tsv"
"$F" G8 $E/release-99/tsv/ensembl-compara/homologies/homo_sapiens/Compara.99.protein_default.homologies.tsv.gz "$D/Compara.99.protein_default.homologies.tsv.gz" "$M"
