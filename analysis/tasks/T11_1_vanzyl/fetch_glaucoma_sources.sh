#!/usr/bin/env bash
# T11.1 step 3 sources. Usage: fetch_glaucoma_sources.sh <chimera_repo>
# van Zyl 2020 (PMC7229661): the PMC article page, from which the gene sentences are quoted
# (publisher blocks the PMC XML). Gharahkhani 2021: Nature Communications ESM MOESM4 holds
# Supplementary Data 1-25 (SD14 = functional annotation of all 127 loci); the XML gives Table 1.
C="$1"; F="$C/analysis/tasks/T4_1_gold/fetch_item.sh"; D="$C/data/analysis/glaucoma"; M="$D/MANIFEST.tsv"
UA='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36'
mkdir -p "$D"
[ -s "$D/vanZyl2020_PMC7229661.html" ] || curl -sSL -A "$UA" -m 300 -o "$D/vanZyl2020_PMC7229661.html" https://pmc.ncbi.nlm.nih.gov/articles/PMC7229661/
"$F" vanZyl2020 https://pmc.ncbi.nlm.nih.gov/articles/PMC7229661/ "$D/vanZyl2020_PMC7229661.html" "$M"
"$F" Gharahkhani2021 "https://www.ebi.ac.uk/europepmc/webservices/rest/PMC7904932/fullTextXML" "$D/Gharahkhani2021_PMC7904932.xml" "$M"
"$F" Gharahkhani2021 "https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-020-20851-4/MediaObjects/41467_2020_20851_MOESM4_ESM.xlsx" "$D/Gharahkhani2021_Supplementary_Data_1-25.xlsx" "$M"
