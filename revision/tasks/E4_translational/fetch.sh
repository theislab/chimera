#!/usr/bin/env bash
# E4: fetch the translational resources (read as data only, with python -I).
# Run from chimera/: bash revision/tasks/E4_translational/fetch.sh ; writes data/revision/E4/<resource>/ and data/revision/E4/MANIFEST.tsv
# (file, url, UTC date, version note, bytes, sha256). Reused files of earlier tasks are listed in rules.REUSED with their own manifests.
set -euo pipefail
D=data/revision/E4
mkdir -p "$D"
M="$D/MANIFEST.tsv"
[ -f "$M" ] || echo -e "file\turl\tdate_utc\tversion\tbytes\tsha256" > "$M"
get() {
  local url="$1" f="$2" ver="$3"
  mkdir -p "$(dirname "$D/$f")"
  if [ -s "$D/$f" ]; then return 0; fi
  curl -fsSL --retry 8 --retry-delay 10 --retry-connrefused -o "$D/$f" "$url"
  echo -e "$f\t$url\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$ver\t$(stat -c %s "$D/$f")\t$(sha256sum "$D/$f" | cut -d' ' -f1)" >> "$M"
}
MGI=https://www.informatics.jax.org/downloads/reports
# MGI human-mouse disease connection: DO diseases with human genes and mouse models; genotype-level model and NOT-model annotations; homology.
get $MGI/MGI_DO.rpt mgi/MGI_DO.rpt "MGI current (report date in file system)"
get $MGI/MGI_Geno_DiseaseDO.rpt mgi/MGI_Geno_DiseaseDO.rpt "MGI current"
get $MGI/MGI_Geno_NotDiseaseDO.rpt mgi/MGI_Geno_NotDiseaseDO.rpt "MGI current"
get $MGI/HOM_MouseHumanSequence.rpt mgi/HOM_MouseHumanSequence.rpt "MGI current"
get $MGI/MGI_GenePheno.rpt mgi/MGI_GenePheno.rpt "MGI current"
get https://purl.obolibrary.org/obo/doid.obo do/doid.obo "DO purl latest"
get https://thegencc.org/download/action/submissions-export-tsv gencc/gencc-submissions.tsv "GenCC export of fetch date"
# Genomics England PanelApp, a priori panels (ids in rules.PANELS), full JSON per panel (genes with confidence level).
for id in 544 385 528 267 653 6 106 292 487 149 234 548 139 472 293 308 152 26 249 250 658 307 398 159 518 545 1075 200 1638 550 193 285 402 96 162 9 225 185 207 66; do
  get "https://panelapp.genomicsengland.co.uk/api/v1/panels/$id/" panelapp/panel_$id.json "PanelApp API v1, current panel version (in JSON)"
done
OT=https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/latest/output
get $OT/clinical_target/00000000.parquet opentargets/clinical_target/00000000.parquet "Open Targets Platform latest (2026-09 build)"
get $OT/disease/00000000.parquet opentargets/disease/00000000.parquet "Open Targets Platform latest"
get $OT/drug_molecule/00000000.parquet opentargets/drug_molecule/00000000.parquet "Open Targets Platform latest"
get $OT/clinical_indication/00000000.parquet opentargets/clinical_indication/00000000.parquet "Open Targets Platform latest"
for d in evidence_orphanet evidence_gene2phenotype evidence_clingen evidence_genomics_england evidence_impc; do
  for f in $(curl -fsSL --retry 8 --retry-delay 10 --retry-connrefused "$OT/$d/" | grep -o 'href="part-[^"]*parquet"' | sed 's/href="//;s/"//' | sort -u); do
    get "$OT/$d/$f" "opentargets/$d/$f" "Open Targets Platform latest"
  done
done
cat "$M" | cut -f1,4,5 | column -t | tail -5
