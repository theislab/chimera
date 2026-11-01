#!/usr/bin/env bash
# Chunked download with HTTP range requests, for large GEO files when single long
# transfers keep dropping. Each chunk is retried until it has the exact length, chunks
# are concatenated, and the total size is checked against Content-Length.
# Usage: fetch_ranged.sh <url> <dest> [chunk_MiB=64] [parallel=4]
set -u
url="$1"; dest="$2"; chunk=$(( ${3:-64} * 1048576 )); par="${4:-4}"
[ -s "$dest" ] && { echo "exists $dest"; exit 0; }
size=$(curl -sIL --max-time 120 "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{v=$2} END{print v}')
[ -n "$size" ] || { echo "no Content-Length for $url" >&2; exit 1; }
dir="$dest.chunks"; mkdir -p "$dir"
n=$(( (size + chunk - 1) / chunk ))
get_chunk () {
    local i=$1 a=$(( $1 * chunk )) b=$(( ($1 + 1) * chunk - 1 ))
    [ "$b" -ge "$size" ] && b=$(( size - 1 ))
    local want=$(( b - a + 1 )) f="$dir/$(printf %05d "$i")"
    for t in $(seq 1 60); do
        [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$want" ] && return 0
        curl -sS --max-time 900 --connect-timeout 60 -r "$a-$b" -o "$f.tmp" "$url" 2>/dev/null
        if [ "$(stat -c %s "$f.tmp" 2>/dev/null)" = "$want" ]; then mv "$f.tmp" "$f"; return 0; fi
        sleep $(( 5 + t ))
    done
    echo "chunk $i failed" >&2; return 1
}
export -f get_chunk; export url dir chunk size
seq 0 $(( n - 1 )) | xargs -P "$par" -I{} bash -c 'get_chunk {}' || { echo "FAILED $url" >&2; exit 1; }
cat $(ls "$dir"/[0-9]* | sort) > "$dest.part"
[ "$(stat -c %s "$dest.part")" = "$size" ] || { echo "SIZE_MISMATCH $dest" >&2; exit 2; }
mv "$dest.part" "$dest" && rm -rf "$dir"
echo "OK ranged $dest $size"
