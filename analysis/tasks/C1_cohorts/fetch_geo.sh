#!/usr/bin/env bash
# Fetch GEO series files with patient retries: NCBI answers 403/500 in bursts. No file is executed.
# Usage: fetch_geo.sh <dest dir> <url> [<url> ...]; appends cohort rows to new_usecases/MANIFEST_C1.tsv
dest="$1"; shift
mkdir -p "$dest"
UA='Mozilla/5.0 (X11; Linux x86_64)'
for u in "$@"; do
  f="$dest/$(basename "$u")"
  [[ -s "$f" ]] && { echo "have $f"; continue; }
  ok=0
  for i in $(seq 1 60); do
    code=$(curl -sS -L -A "$UA" -m 7200 -C - -w '%{http_code}' -o "$f.part" "$u" 2>/dev/null || echo fail)
    if [[ "$code" == 200 || "$code" == 206 ]]; then
      if head -c 200 "$f.part" | grep -qi '<html\|<!DOCTYPE'; then rm -f "$f.part"; code=html; else mv "$f.part" "$f"; ok=1; break; fi
    else rm -f "$f.part"; fi
    echo "retry $i $u code=$code"; sleep $((20 + i * 10))
  done
  if [[ $ok == 1 ]]; then
    printf '%s\t%s\t%s\t%s\t%s\n' "$(basename "$dest")" "$(basename "$u")" "$u" "$(date -u +%FT%TZ)" "$(stat -c %s "$f")" >> data/analysis/new_usecases/MANIFEST_C1.tsv
    echo "got $f $(stat -c %s "$f")"
  else
    echo "FAILED $u"
  fi
done
