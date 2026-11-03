#!/usr/bin/env bash
# E1: fetch GO-slim and the Reactome hierarchy. Hallmark and Reactome gene sets are T6.1's (data/revision/functional).
# Run from chimera/: bash revision/tasks/E1_atlas/fetch.sh ; writes data/revision/E1/ and MANIFEST.tsv (url, UTC date, bytes, sha256).
set -euo pipefail
D=data/revision/E1
mkdir -p "$D"
M="$D/MANIFEST.tsv"
echo -e "file\turl\tdate_utc\tbytes\tsha256" > "$M"
get() {
  local url="$1" f="$2"
  curl -fsSL --retry 3 -o "$D/$f" "$url"
  echo -e "$f\t$url\t$(date -u +%Y-%m-%dT%H:%M:%SZ)\t$(stat -c %s "$D/$f")\t$(sha256sum "$D/$f" | cut -d' ' -f1)" >> "$M"
}
get https://current.geneontology.org/ontology/go-basic.obo go-basic.obo
get https://current.geneontology.org/ontology/subsets/goslim_generic.obo goslim_generic.obo
get https://current.geneontology.org/annotations/goa_human.gaf.gz goa_human.gaf.gz
get https://reactome.org/download/current/ReactomePathwaysRelation.txt ReactomePathwaysRelation.txt
get https://reactome.org/download/current/ReactomePathways.txt ReactomePathways.txt
cat "$M"
