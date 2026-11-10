#!/usr/bin/env bash
# Non-human Compara homology dumps. Ensembl writes each homology pair to one species' file
# only, so a 1:1 map built from the human file alone misses pairs; the map is the union of
# the human file and the partner species' file (h4_map_union.py).
# Usage: fetch_ensembl_species.sh <chimera_repo>
C="$1"; F="$C/analysis/tasks/T4_1_gold/fetch_item.sh"
E=https://ftp.ensembl.org/pub
D="$C/data/analysis/ensembl110/by_species"; M="$D/MANIFEST.tsv"
for sp in pan_troglodytes gorilla_gorilla nomascus_leucogenys callithrix_jacchus monodelphis_domestica ornithorhynchus_anatinus; do
  "$F" E110S $E/release-110/tsv/ensembl-compara/homologies/$sp/Compara.110.protein_default.homologies.tsv.gz "$D/$sp/Compara.110.protein_default.homologies.tsv.gz" "$M" || exit 1
done
# chicken: release 105 (GRCg6a IDs, as the human-side map)
"$F" E105S $E/release-105/tsv/ensembl-compara/homologies/gallus_gallus/Compara.105.protein_default.homologies.tsv.gz "$D/gallus_gallus/Compara.105.protein_default.homologies.tsv.gz" "$M" || exit 1
# H8 dN/dS: Compara 99 mouse-side file
D="$C/data/analysis/gold/G8"; M="$D/MANIFEST.tsv"
"$F" G8 $E/release-99/tsv/ensembl-compara/homologies/mus_musculus/Compara.99.protein_default.homologies.tsv.gz "$D/mus_musculus/Compara.99.protein_default.homologies.tsv.gz" "$M" || exit 1
