#!/usr/bin/env bash
# Download the files of a download manifest and check each one against its size and sha256.
# Usage, from the repository root: bash analysis/tasks/C1_cohorts/fetch_manifest.sh [--dry-run] <manifest.tsv> [...]
#   data/analysis/new_usecases/MANIFEST_SHA256.tsv   cortex, lung, skeletal muscle and tumor myeloid cells with their
#                                                    held-out and disease data (44 files, 22 GB; CELLxGENE, GEO, Ensembl)
#   data/raw/MANIFEST.tsv                            what the data preparation notebook downloads: the CAMEX files of
#                                                    liver and testis, the SATURN archives, the eye matrices (47 files,
#                                                    4.7 GB); the notebook then finds the files and downloads nothing
# The script serves every manifest of this layout: the islet cohort uses it with the manifest of its raw
# folder (analysis/tasks/T7_9_islet/README.md). The manifests come with the bundles data_other_cohorts and
# data_liver_testis of the deposit (REPRODUCE.md, section 3).
# A manifest is a tab-separated table whose header row names the columns file, url, bytes and sha256
# (other columns are ignored). `file` is relative to the folder of the manifest, where the build
# scripts read it. A file that is present with the size and the checksum of its row is kept; any other
# is downloaded to <file>.part (resumed when a part exists) and moved into place only when size and
# checksum agree. --dry-run lists what a run would do, compares sizes only and downloads nothing.
# Exit code 0: every file of every manifest is present and correct (dry run: no row has a problem).
# No file is executed.
set -u
usage() { sed -n '2,3p' "$0" | cut -c3-; }
dry=0; manifests=()
for a in "$@"; do
  case "$a" in
    --dry-run) dry=1 ;;
    -h|--help) usage; exit 0 ;;
    -*) echo "unknown option $a" >&2; usage >&2; exit 2 ;;
    *) manifests+=("$a") ;;
  esac
done
[ "${#manifests[@]}" -gt 0 ] || { usage >&2; exit 2; }

# download <url> <part file> <bytes>: patient retries, resuming the part; 0 when the part has the size of the row.
# A part that cannot be completed by resuming is removed; an unfinished one is kept for the next run.
download() {
  local url="$1" part="$2" want="$3" i rc
  for i in 1 2 3 4 5 6; do
    curl -fL -C - --retry 5 --retry-delay 20 --retry-connrefused --connect-timeout 60 -sS -o "$part" "$url"; rc=$?
    [ "$(stat -c %s "$part" 2>/dev/null)" = "$want" ] && return 0
    [ "$rc" = 0 ] && { rm -f "$part"; return 1; }  # complete answer of another size
    [ "$rc" = 33 ] && rm -f "$part"                # the server cannot resume: start again
    [ "$rc" = 22 ] && [ "$i" -ge 2 ] && return 1   # HTTP error (404 and the like): no long retry loop
    echo "  retry $i (curl exit $rc)"; sleep $((30 * i))
  done
  return 1
}

bad=0
for m in "${manifests[@]}"; do
  [ -f "$m" ] || { echo "$m: no such file (the download manifests come with the data bundles of the deposit)" >&2; bad=1; continue; }
  dir="$(dirname "$m")"; n=0; kept=0; got=0; failed=0; todo=0
  rows="$(awk -F'\t' '
    NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i
              if (!(c["file"] && c["url"] && c["bytes"] && c["sha256"])) exit 3; next }
    $c["file"] != "" { u = $c["url"]; b = $c["bytes"]; s = $c["sha256"]
                       print $c["file"] "\t" (u == "" ? "-" : u) "\t" (b == "" ? "-" : b) "\t" (s == "" ? "-" : s) }' "$m")" \
    || { echo "$m: the header row must name the columns file, url, bytes and sha256" >&2; bad=1; continue; }
  while IFS=$'\t' read -r f url bytes sha; do
    [ -n "$f" ] || continue
    n=$((n + 1)); dest="$dir/$f"
    case "/$f/" in */../*|//*) echo "REFUSED $f: not a path below the folder of the manifest" >&2; failed=$((failed + 1)); continue ;; esac
    if ! [[ "$bytes" =~ ^[0-9]+$ && "$sha" =~ ^[0-9a-f]{64}$ && "$url" =~ ^https?:// ]]; then
      echo "REFUSED $f: the row has no usable url, size or sha256" >&2; failed=$((failed + 1)); continue
    fi
    if [ -f "$dest" ] && [ "$(stat -c %s "$dest")" = "$bytes" ]; then
      if [ "$dry" = 1 ]; then echo "have   $dest ($bytes bytes; checksum not computed in a dry run)"; kept=$((kept + 1)); continue; fi
      if [ "$(sha256sum "$dest" | cut -d' ' -f1)" = "$sha" ]; then echo "ok     $dest"; kept=$((kept + 1)); continue; fi
      echo "DIFFERS $dest: size agrees, sha256 does not; move it away and run again" >&2; failed=$((failed + 1)); continue
    fi
    if [ -e "$dest" ]; then
      echo "DIFFERS $dest: $(stat -c %s "$dest") bytes, the manifest says $bytes; move it away and run again" >&2
      failed=$((failed + 1)); continue
    fi
    if [ "$dry" = 1 ]; then echo "fetch  $dest ($bytes bytes) from $url"; todo=$((todo + 1)); continue; fi
    mkdir -p "$(dirname "$dest")"
    echo "fetch  $dest ($bytes bytes)"
    if ! download "$url" "$dest.part" "$bytes"; then
      echo "FAILED $f: $url did not give the $bytes bytes of the manifest (an unfinished part is kept; run again to resume)" >&2
      failed=$((failed + 1)); continue
    fi
    if [ "$(sha256sum "$dest.part" | cut -d' ' -f1)" != "$sha" ]; then
      echo "FAILED $f: sha256 of the download differs from the manifest (the source may have changed the file)" >&2
      rm -f "$dest.part"; failed=$((failed + 1)); continue
    fi
    mv "$dest.part" "$dest"; echo "ok     $dest"; got=$((got + 1))
  done <<< "$rows"
  if [ "$dry" = 1 ]; then
    echo "$m: $n files, $kept present with the size of the manifest, $todo to fetch, $failed with a problem"
  else
    echo "$m: $n files, $kept present and correct, $got downloaded, $failed failed"
  fi
  [ "$failed" = 0 ] && [ "$n" -gt 0 ] || bad=1
done
exit "$bad"
