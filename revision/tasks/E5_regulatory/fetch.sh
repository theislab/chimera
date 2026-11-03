#!/usr/bin/env bash
# E5 (EXPLORATORY): downloads for regulatory and gene-architecture properties of deployment. Files only, never executed.
# Each file lands in data/revision/E5/<resource>/ (its own directory) and gets a row in data/revision/E5/MANIFEST.tsv
# (file, url, date, version, sha256, bytes). Readers open these files with `python -I`, never from inside the download dirs.
# Reused, not downloaded again: Y2 regulatory annotation (Zoonomia 241-way phyloP per region, UCSC CpG islands, EPDnew
# TATA, ABC links, hg38.fa), Ensembl 110 GTF and Compara 110, E2 props (GTEx tau, TF class), HGNC (X3).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/data/revision/E5"
mkdir -p "$OUT"
[ -f "$OUT/MANIFEST.tsv" ] || printf 'file\turl\tdate\tversion\tsha256\tbytes\n' > "$OUT/MANIFEST.tsv"
fetch() {
  local sub="$1" url="$2" name="$3" version="$4"
  mkdir -p "$OUT/$sub"
  local f="$OUT/$sub/$name"
  if grep -qF "	$url	" "$OUT/MANIFEST.tsv"; then echo "have $name"; return; fi
  if ! curl -gfsSL --retry 5 --retry-all-errors --retry-delay 20 --max-time 3600 -o "$f" "$url"; then echo "FAILED $url"; rm -f "$f"; return 1; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$sub/$name" "$url" "$(date -u +%FT%TZ)" "$version" "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" >> "$OUT/MANIFEST.tsv"
  echo "ok $name"
}
fetch encode_ccre https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.bed GRCh38-cCREs.V4.bed "ENCODE SCREEN Registry of cCREs V4, GRCh38"
fetch targetscan https://www.targetscan.org/vert_80/vert_80_data_download/Summary_Counts.default_predictions.txt.zip Summary_Counts.default_predictions.txt.zip "TargetScanHuman 8.0"
fetch targetscan https://www.targetscan.org/vert_80/vert_80_data_download/miR_Family_Info.txt.zip miR_Family_Info.txt.zip "TargetScanHuman 8.0"
fetch targetscan https://www.targetscan.org/vert_80/vert_80_data_download/Gene_info.txt.zip Gene_info.txt.zip "TargetScanHuman 8.0"
fetch omnipath "https://omnipathdb.org/interactions?datasets=dorothea&dorothea_levels=A,B,C&genesymbols=yes&fields=dorothea_level&organisms=9606&license=academic" dorothea_ABC.tsv "OmniPath web service, DoRothEA levels A-C (Garcia-Alonso 2019)"
fetch omnipath "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=yes&organisms=9606&license=academic" collectri.tsv "OmniPath web service, CollecTRI (Mueller-Dott 2023)"
fetch hkg https://www.tau.ac.il/~elieis/HKG/HK_genes.txt HK_genes.txt "Eisenberg and Levanon 2013 housekeeping genes"
fetch msigdb https://data.broadinstitute.org/gsea-msigdb/msigdb/release/2025.1.Hs/c2.cgp.v2025.1.Hs.symbols.gmt c2.cgp.v2025.1.Hs.symbols.gmt "MSigDB 2025.1.Hs C2 CGP"
# TADs: the 3D Genome Browser hg38 bundle returned 404 on 2026-11-03; ENCODE GRCh38 contact domains (Arrowhead, bedpe) instead,
# one released file per biosample, seven diverse biosamples.
for pair in GM12878:ENCFF191MYU IMR-90:ENCFF166QGX HepG2:ENCFF018XKF K562:ENCFF271SAF HUES64:ENCFF395ETE kidney:ENCFF817SLU heart_LV:ENCFF558RLO; do
  bs="${pair%%:*}"; acc="${pair##*:}"
  fetch tads "https://www.encodeproject.org/files/$acc/@@download/$acc.bedpe.gz" "${bs}_$acc.bedpe.gz" "ENCODE contact domains $acc ($bs, GRCh38)"
done
