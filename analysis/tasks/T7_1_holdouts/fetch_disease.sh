#!/usr/bin/env bash
# Downloads of the liver disease raw data. Mouse diet series: per-sample h5 files only, CS184-CS197.
# Usage: fetch_disease.sh <chimera_repo>
C="$1"; F="$C/analysis/tasks/T4_1_gold/fetch_item.sh"; D="$C/data/analysis/disease/raw"; M="$D/MANIFEST.tsv"
GEO=https://ftp.ncbi.nlm.nih.gov/geo
get_series () { # gse relpath-under-suppl
    local gse="$1" stub="${1:0:${#1}-3}nnn"
    "$F" "$gse" "$GEO/series/$stub/$gse/suppl/$2" "$D/$gse/$2" "$M"
}
# Large files: NCBI drops long transfers and throttles (500, 403), so fetch in 32 MiB
# range chunks with few parallel requests, then record the manifest row.
get_series_ranged () {
    local gse="$1" stub="${1:0:${#1}-3}nnn"
    "$C/analysis/tasks/T7_1_holdouts/fetch_ranged.sh" "$GEO/series/$stub/$gse/suppl/$2" "$D/$gse/$2" 32 2 \
        && "$F" "$gse" "$GEO/series/$stub/$gse/suppl/$2" "$D/$gse/$2" "$M"
}
get_matrix () {
    local gse="$1" stub="${1:0:${#1}-3}nnn"
    "$F" "$gse" "$GEO/series/$stub/$gse/matrix/${gse}_series_matrix.txt.gz" "$D/$gse/${gse}_series_matrix.txt.gz" "$M"
}
for gse in GSE135251 GSE129516 GSE202379; do get_matrix "$gse"; done
# multi-platform series: one matrix per GPL
for gm in GSE192742-GPL24247 GSE192742-GPL27943 GSE192742-GPL20301 GSE192742-GPL21103 GSE192742-GPL21741 GSE192742-GPL22475 GSE192742-GPL23499 GSE192742-GPL24676 GSE192742-GPL28997 GSE136103-GPL20301 GSE136103-GPL21103; do
    gse="${gm%%-*}"; stub="${gse:0:${#gse}-3}nnn"
    "$F" "$gse" "$GEO/series/$stub/$gse/matrix/${gm}_series_matrix.txt.gz" "$D/$gse/${gm}_series_matrix.txt.gz" "$M"
done
for gse in GSE192742 GSE135251 GSE129516 GSE136103 GSE202379; do get_series "$gse" filelist.txt; done
while read -r gsm cs; do
    stub="${gsm:0:7}nnn"; f="${gsm}_filtered_feature_bc_matrix_${cs}.h5"
    "$F" GSE192742 "$GEO/samples/$stub/$gsm/suppl/$f" "$D/GSE192742_diet/$f" "$M"
done <<LIST
GSM5764312 CS184
GSM5764313 CS185
GSM5764314 CS188
GSM5764315 CS189
GSM5764316 CS190
GSM5764317 CS191
GSM5764318 CS192
GSM5764320 CS193
GSM5764323 CS194
GSM5764327 CS195
GSM5764330 CS196
GSM5764333 CS197
LIST
get_series GSE135251 GSE135251_RAW.tar
get_series_ranged GSE129516 GSE129516_RAW.tar
get_series_ranged GSE136103 GSE136103_RAW.tar
get_series_ranged GSE202379 GSE202379_RAW.tar
get_series_ranged GSE202379 GSE202379_SeuratObject_AllCells.rds.gz
# Hagai et al. 2018 Nature 563:197, Supplementary Table 4 (Nature ESM MOESM4; PMC6347972 lists it as
# NIHMS79113-supplement-Supplementary_Table_4.xlsx). Sheets: fibroblasts_FC_divergence, phagocytes_FC_diveregnce.
"$F" Hagai2018 "https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-018-0657-2/MediaObjects/41586_2018_657_MOESM4_ESM.xlsx" "$D/Hagai2018/Hagai2018_Supplementary_Table_4.xlsx" "$M"
# GEO sample metadata in SOFT text (all samples, all platforms): condition, donor, platform.
for gse in GSE202379 GSE192742 GSE135251 GSE129516 GSE136103; do
    "$F" "$gse" "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=$gse&targ=gsm&form=text&view=brief" "$D/$gse/${gse}_samples_brief.soft.txt" "$M"
done
