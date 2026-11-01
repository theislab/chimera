#!/usr/bin/env bash
# Resumable fetch of one URL with a manifest row.
# Usage: fetch_item.sh <id> <url> <dest_file> <manifest.tsv> [expected_size_bytes]
# Writes <dest_file>.part, verifies the size against Content-Length when the
# server gives one, renames, and appends: id, file, url, date, bytes, sha256.
# An existing destination is not refetched; its manifest row is added if missing.
set -u
id="$1"; url="$2"; dest="$3"; manifest="$4"; expected="${5:-}"
mkdir -p "$(dirname "$dest")"
[ -s "$manifest" ] || printf 'id\tfile\turl\tdate_utc\tbytes\tsha256\n' > "$manifest"
if [ ! -s "$dest" ]; then
    if [ -z "$expected" ]; then
        expected=$(curl -sIL --max-time 120 "$url" | tr -d '\r' \
            | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
    fi
    tries=0
    while :; do
        tries=$((tries+1))
        curl -fL -C - --retry 5 --retry-delay 20 --retry-connrefused \
             --connect-timeout 60 -sS -o "$dest.part" "$url" && break
        rc=$?
        # 33 = server refuses resume on a finished file; treat by size below
        if [ -n "$expected" ] && [ "$(stat -c %s "$dest.part" 2>/dev/null)" = "$expected" ]; then break; fi
        # 22 = HTTP error (404 and the like): no point in a long retry loop
        if [ "$rc" = 22 ] && [ "$tries" -ge 2 ]; then echo "FAILED $id $url rc=$rc" >&2; rm -f "$dest.part"; exit 1; fi
        [ "$tries" -ge 6 ] && { echo "FAILED $id $url rc=$rc" >&2; exit 1; }
        sleep 60
    done
    got=$(stat -c %s "$dest.part")
    if [ -n "$expected" ] && [ "$got" != "$expected" ]; then
        echo "SIZE_MISMATCH $id got=$got expected=$expected" >&2; exit 2
    fi
    mv "$dest.part" "$dest"
fi
if ! grep -qF "$(basename "$dest")	$url" "$manifest"; then
    sha=$(sha256sum "$dest" | cut -d' ' -f1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$id" "$(basename "$dest")" "$url" \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(stat -c %s "$dest")" "$sha" >> "$manifest"
fi
echo "OK $id $(basename "$dest") $(stat -c %s "$dest")"
