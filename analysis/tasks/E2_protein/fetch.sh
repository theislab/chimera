#!/usr/bin/env bash
# E2 (EXPLORATORY): downloads for protein-level and gene-architecture properties of deployment. Files only, never executed.
# Each file lands in data/analysis/E2/<resource>/ (its own directory) and gets a row in data/analysis/E2/MANIFEST.tsv
# (file, url, date, version, sha256, bytes). Readers open these files with `python -I`, never from inside the download dirs.
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/data/analysis/E2"
mkdir -p "$OUT"
[ -f "$OUT/MANIFEST.tsv" ] || printf 'file\turl\tdate\tversion\tsha256\tbytes\n' > "$OUT/MANIFEST.tsv"
fetch() {
  local sub="$1" url="$2" name="$3" version="$4"
  mkdir -p "$OUT/$sub"
  local f="$OUT/$sub/$name"
  if grep -qF "	$url	" "$OUT/MANIFEST.tsv"; then echo "have $name"; return; fi
  if ! curl -gfsSL --retry 5 --retry-all-errors --retry-delay 20 --max-time 1800 -o "$f" "$url"; then echo "FAILED $url"; rm -f "$f"; return 1; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$sub/$name" "$url" "$(date -u +%FT%TZ)" "$version" "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" >> "$OUT/MANIFEST.tsv"
  echo "ok $name"
}
UNI_REL="$(curl -sI 'https://rest.uniprot.org/uniprotkb/search?query=P04637&size=1' | tr -d '\r' | awk -F': ' 'tolower($1)=="x-uniprot-release"{print $2}')"
fetch uniprot "https://rest.uniprot.org/uniprotkb/stream?compressed=true&format=tsv&fields=accession,gene_primary,xref_ensembl,length,xref_pfam,xref_interpro,xref_complexportal,cc_subunit,protein_existence&query=%28organism_id%3A9606%29%20AND%20%28reviewed%3Atrue%29" uniprot_human_domains.tsv.gz "UniProtKB release ${UNI_REL:-unknown}"
fetch pfam https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.clans.tsv.gz Pfam-A.clans.tsv.gz "Pfam current_release (see relnotes)"
fetch pfam https://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/relnotes.txt relnotes.txt "Pfam current_release"
fetch hpa https://www.proteinatlas.org/download/proteinatlas.tsv.zip proteinatlas.tsv.zip "Human Protein Atlas current (www)"
fetch complexportal https://ftp.ebi.ac.uk/pub/databases/intact/complex/current/complextab/9606.tsv complexportal_9606.tsv "Complex Portal current (CORUM server unreachable 2026-11-03)"
fetch ohnologs "http://ohnologs.curie.fr/cgi-bin/DownloadBrowse.cgi?crit=[A]&org=hsapiens&opt=pairs&wgd=2R" hsapiens_2R_pairs_intermediate.tsv "OHNOLOGS v2 (Singh and Isambert 2020), 2R, intermediate"
fetch ohnologs "http://ohnologs.curie.fr/cgi-bin/DownloadBrowse.cgi?crit=[0]&org=hsapiens&opt=pairs&wgd=2R" hsapiens_2R_pairs_strict.tsv "OHNOLOGS v2 (Singh and Isambert 2020), 2R, strict"
fetch dosage https://zenodo.org/records/6347673/files/Collins_rCNV_2022.dosage_sensitivity_scores.tsv.gz Collins_rCNV_2022.dosage_sensitivity_scores.tsv.gz "Collins et al. 2022 Cell, Zenodo 6347673"
fetch gtex https://storage.googleapis.com/adult-gtex/bulk-gex/v8/rna-seq/GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct.gz GTEx_v8_gene_median_tpm.gct.gz "GTEx v8"
