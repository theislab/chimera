#!/usr/bin/env bash
# T12.1 COE inputs: Ensembl 110 BioMart (archive host jul2023 = "Ensembl Genes 110") probe-to-gene
# maps for the two GEO platforms. GPL4685 (D-MAP, HT_HG-U133A probesets, same IDs and probes as
# U133A 2.0 / U133 Plus 2) and GPL6246 (ImmGen, MoGene 1.0 ST transcript clusters).
# Usage: fetch_probe_maps.sh <chimera_repo>
set -u
C="$1"; E="$C/data/analysis/bone_marrow/ensembl110"; M="$E/MANIFEST.tsv"
H=https://jul2023.archive.ensembl.org/biomart/martservice
mkdir -p "$E"
for i in 1 2 3; do curl -s -m 300 "$H?type=registry" | grep -q 'displayName="Ensembl Genes 110"' && break; sleep 30; done
curl -s -m 300 "$H?type=registry" | grep -q 'displayName="Ensembl Genes 110"' || { echo "archive host is not Ensembl 110" >&2; exit 1; }
q() {  # q <dataset> <probe attribute> <dest>
    local xml="<?xml version=\"1.0\" encoding=\"UTF-8\"?><!DOCTYPE Query><Query virtualSchemaName=\"default\" formatter=\"TSV\" header=\"1\" uniqueRows=\"1\" datasetConfigVersion=\"0.6\"><Dataset name=\"$1\" interface=\"default\"><Attribute name=\"ensembl_gene_id\"/><Attribute name=\"$2\"/></Dataset></Query>"
    local url="$H?query=$(python3 -I -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1]))' "$xml")"
    if [ ! -s "$3" ]; then
        curl -sS -m 1800 --retry 3 -o "$3.part" "$url" || { echo "FAILED $2" >&2; return 1; }
        tail -c 200 "$3.part" | grep -q "Query ERROR" && { echo "BioMart error $2" >&2; return 1; }
        head -c 200 "$3.part" | grep -qi "<html" && { echo "HTML error page for $2" >&2; rm -f "$3.part"; return 1; }
        mv "$3.part" "$3"
    fi
    [ -s "$M" ] || printf 'id\tfile\turl\tdate_utc\tbytes\tsha256\n' > "$M"
    grep -qF "$(basename "$3")	" "$M" || printf '%s\t%s\t%s\t%s\t%s\t%s\n' "E110_BIOMART" "$(basename "$3")" \
        "$H [$1: ensembl_gene_id, $2]" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(stat -c %s "$3")" \
        "$(sha256sum "$3" | cut -d' ' -f1)" >> "$M"
    echo "OK $2 $(wc -l < "$3")"
}
q hsapiens_gene_ensembl affy_hg_u133a_2 "$E/biomart110_hsapiens_affy_hg_u133a_2.tsv"
q hsapiens_gene_ensembl affy_hg_u133_plus_2 "$E/biomart110_hsapiens_affy_hg_u133_plus_2.tsv"
q mmusculus_gene_ensembl affy_mogene_1_0_st_v1 "$E/biomart110_mmusculus_affy_mogene_1_0_st_v1.tsv"
