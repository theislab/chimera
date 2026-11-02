#!/usr/bin/env bash
# Y2 (EXPLORATORY): downloads for the functional-consequences analysis. Files only, never executed.
# Usage: fetch.sh small | big
# Each row: subdir<TAB>URL. Every file lands in data/revision/Y2/<subdir>/ and gets a row in data/revision/Y2/MANIFEST.tsv.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/data/revision/Y2"
mkdir -p "$OUT"
[ -f "$OUT/MANIFEST.tsv" ] || printf 'file\turl\tdate\tsha256\tbytes\n' > "$OUT/MANIFEST.tsv"
fetch() {
  local sub="$1" url="$2"
  local name="${3:-$(basename "${url%%\?*}")}"
  mkdir -p "$OUT/$sub"
  local f="$OUT/$sub/$name"
  if grep -q "	$url	" "$OUT/MANIFEST.tsv"; then echo "have $name"; return; fi
  if ! curl -fsSL --retry 5 --retry-delay 20 -C - -o "$f" "$url"; then echo "FAILED $url"; return 1; fi
  printf '%s\t%s\t%s\t%s\t%s\n' "$sub/$name" "$url" "$(date -u +%FT%TZ)" "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" >> "$OUT/MANIFEST.tsv"
  echo "ok $name"
}
case "$1" in
small)
  fetch impc https://ftp.ebi.ac.uk/pub/databases/impc/all-data-releases/latest/results/genotype-phenotype-assertions-ALL.csv.gz
  fetch impc https://ftp.ebi.ac.uk/pub/databases/impc/all-data-releases/latest/results/viability.csv.gz
  fetch mgi https://www.informatics.jax.org/downloads/reports/MGI_GenePheno.rpt
  fetch mgi https://www.informatics.jax.org/downloads/reports/MGI_PhenoGenoMP.rpt
  fetch mgi https://www.informatics.jax.org/downloads/reports/MRK_ENSEMBL.rpt
  fetch mgi https://www.informatics.jax.org/downloads/reports/MGI_PhenotypicAllele.rpt
  fetch mgi https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt
  fetch onto https://purl.obolibrary.org/obo/mp.obo
  fetch onto https://purl.obolibrary.org/obo/hp.obo
  fetch hpo https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/genes_to_phenotype.txt
  fetch hpo https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/genes_to_disease.txt
  fetch hpo https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa
  fetch clingen https://search.clinicalgenome.org/kb/gene-validity/download Clingen-Gene-Disease-Summary.csv
  fetch clingen https://ftp.clinicalgenome.org/ClinGen_gene_curation_list_GRCh38.tsv
  fetch ucsc https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz
  fetch epd https://epd.expasy.org/ftp/epdnew/H_sapiens/current/Hs_EPDnew.bed
  fetch epd https://epd.expasy.org/ftp/epdnew/H_sapiens/current/Hs_EPDnew.dat
  ;;
big)
  fetch depmap https://ndownloader.figshare.com/files/51064667 CRISPRGeneEffect_24Q4.csv
  fetch depmap https://ndownloader.figshare.com/files/51064916 CRISPRInferredCommonEssentials_24Q4.csv
  fetch ncbi https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2pubmed.gz
  fetch ncbi https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2ensembl.gz
  fetch abc https://mitra.stanford.edu/engreitz/oak/public/Nasser2021/AllPredictions.AvgHiC.ABC0.015.minus150.ForABCPaperV3.txt.gz
  fetch ucsc https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
  fetch zoonomia https://hgdownload.soe.ucsc.edu/goldenPath/hg38/cactus241way/cactus241way.phyloP.bw
  ;;
esac
