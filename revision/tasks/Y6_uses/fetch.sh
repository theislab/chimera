#!/usr/bin/env bash
# Y6 (EXPLORATORY): downloads for the uses of the fits. Files only, never executed; read with python -I.
# Usage: fetch.sh ensembl | small | big
# Every file lands in data/revision/Y6/<subdir>/ and gets a row in data/revision/Y6/MANIFEST.tsv (url, date, sha256).
set -u
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
OUT="$ROOT/data/revision/Y6"
mkdir -p "$OUT"
[ -f "$OUT/MANIFEST.tsv" ] || printf 'file\turl\tdate\tsha256\tbytes\n' > "$OUT/MANIFEST.tsv"
fetch() {
  local sub="$1" url="$2"
  local name="${3:-$(basename "${url%%\?*}")}"
  mkdir -p "$OUT/$sub"
  local f="$OUT/$sub/$name"
  if grep -q "	$url	" "$OUT/MANIFEST.tsv"; then echo "have $name"; return 0; fi
  if ! curl -fsSL --retry 5 --retry-delay 20 -C - -o "$f" "$url"; then echo "FAILED $url"; return 1; fi
  (flock 9; printf '%s\t%s\t%s\t%s\t%s\n' "$sub/$name" "$url" "$(date -u +%FT%TZ)" "$(sha256sum "$f" | cut -d' ' -f1)" "$(stat -c %s "$f")" >> "$OUT/MANIFEST.tsv") 9>"$OUT/.lock"
  echo "ok $sub/$name"
}
E=https://ftp.ensembl.org/pub
case "$1" in
small)
  M=https://raw.githubusercontent.com/ericminikel/genetic_support/main/data
  for f in pp.tsv merge2.tsv.gz assoc.tsv.gz indic.tsv universe.tsv areas.tsv meta_ccat.tsv meta_hcat.tsv meta_acat.tsv \
           drug_phase_summary.tsv pp_launched_alltime.tsv sim.tsv.gz otg_heavy_hitter_curation.tsv; do
    fetch minikel2024 "$M/$f"
  done
  G=https://ftp.ncbi.nlm.nih.gov/geo/series
  fetch ase_wang2024 "$G/GSE232nnn/GSE232948/suppl/GSE232948_RNAseq_ASE_CM_HP_MN_PP_SKM_RPE_GRCh38.txt.gz"
  fetch ase_wang2024 "$G/GSE232nnn/GSE232948/suppl/GSE232948_RNAseq_ASE_CM_HP_MN_PP_SKM_RPE_PanTro6.txt.gz"
  for f in bulk_iPS_GRCh38 bulk_iPS_PanTro5 bulk_iPS_meta bulk_CS_GRCh38 bulk_CS_PanTro5 bulk_CS_meta single_CS_meta; do
    fetch ase_agoglia2021 "$G/GSE144nnn/GSE144825/suppl/GSE144825_$f.txt.gz"
  done
  fetch tm_liver https://datasets.cellxgene.cziscience.com/c126be6e-bc29-4367-a563-493ebed712e5.h5ad tm_liver.h5ad
  O=https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output
  curl -s "$O/clinical_target/" | grep -o 'href="[^"]*parquet"' | sed 's/href="//;s/"//' | while read -r p; do fetch opentargets/clinical_target "$O/clinical_target/$p"; done
  ;;
ensembl)
  # Compara dumps write each pair once, under one of the two species, so every species' file is needed.
  for sp in sus_scrofa macaca_fascicularis; do
    fetch compara110/$sp "$E/release-110/tsv/ensembl-compara/homologies/$sp/Compara.110.protein_default.homologies.tsv.gz"
  done
  fetch gtf110 "$E/release-110/gtf/sus_scrofa/Sus_scrofa.Sscrofa11.1.110.gtf.gz"
  fetch gtf110 "$E/release-110/gtf/macaca_fascicularis/Macaca_fascicularis.Macaca_fascicularis_6.0.110.gtf.gz"
  for sp in homo_sapiens mus_musculus macaca_mulatta macaca_fascicularis sus_scrofa danio_rerio microcebus_murinus; do
    fetch compara116/$sp "$E/release-116/tsv/ensembl-compara/homologies/$sp/Compara.116.protein_default.homologies.tsv.gz"
  done
  fetch gtf116 "$E/release-116/gtf/homo_sapiens/Homo_sapiens.GRCh38.116.gtf.gz"
  fetch gtf116 "$E/release-116/gtf/mus_musculus/Mus_musculus.GRCm39.116.gtf.gz"
  fetch gtf116 "$E/release-116/gtf/macaca_mulatta/Macaca_mulatta.Mmul_10.116.gtf.gz"
  fetch gtf116 "$E/release-116/gtf/macaca_fascicularis/Macaca_fascicularis.Macaca_fascicularis_6.0.116.gtf.gz"
  fetch gtf116 "$E/release-116/gtf/sus_scrofa/Sus_scrofa.Sscrofa11.1.116.gtf.gz"
  fetch gtf116 "$E/release-116/gtf/danio_rerio/Danio_rerio.GRCz11.116.gtf.gz"
  fetch gtf116 "$E/release-116/gtf/microcebus_murinus/Microcebus_murinus.Mmur_3.0.116.gtf.gz"
  fetch gtf98 "$E/release-98/gtf/mus_musculus/Mus_musculus.GRCm38.98.gtf.gz"
  fetch gtf98 "$E/release-98/gtf/macaca_mulatta/Macaca_mulatta.Mmul_10.98.gtf.gz"
  fetch gtf98 "$E/release-98/gtf/macaca_fascicularis/Macaca_fascicularis.Macaca_fascicularis_5.0.98.gtf.gz"
  fetch gtf98 "$E/release-98/gtf/sus_scrofa/Sus_scrofa.Sscrofa11.1.98.gtf.gz"
  fetch gtf98 "$E/release-98/gtf/danio_rerio/Danio_rerio.GRCz11.98.gtf.gz"
  fetch gtf98 "$E/release-98/gtf/microcebus_murinus/Microcebus_murinus.Mmur_3.0.98.gtf.gz"
  # Older assemblies, for a gene model that did not exist yet.
  fetch gtfold "$E/release-97/gtf/macaca_mulatta/Macaca_mulatta.Mmul_8.0.1.97.gtf.gz"
  fetch gtfold "$E/release-89/gtf/sus_scrofa/Sus_scrofa.Sscrofa10.2.89.gtf.gz"
  fetch gtfold "$E/release-91/gtf/danio_rerio/Danio_rerio.GRCz10.91.gtf.gz"
  fetch gtfold "$E/release-84/gtf/mus_musculus/Mus_musculus.GRCm38.84.gtf.gz"
  fetch gtfold "$E/release-90/gtf/microcebus_murinus/Microcebus_murinus.Mmur_2.0.90.gtf.gz"
  ;;
big)
  O=https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output
  curl -s "$O/association_by_datatype_direct/" | grep -o 'href="[^"]*parquet"' | sed 's/href="//;s/"//' | while read -r p; do fetch opentargets/association_by_datatype_direct "$O/association_by_datatype_direct/$p"; done
  ;;
esac
