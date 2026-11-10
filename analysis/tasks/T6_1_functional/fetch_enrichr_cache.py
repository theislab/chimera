"""Fetch the two Enrichr libraries into the gseapy cache so jobs never fetch live.

Writes the file gseapy's own Enrichr.download_libraries would write: name
``Enrichr.<lib>.gmt`` in ``~/.cache/gseapy`` (gseapy.utils.DEFAULT_CACHE_PATH), one
line per term, ``term<TAB>description<TAB>genes...``, with any ``,weight`` suffix cut.
The verbatim server response goes to the functional folder with a manifest row.

Usage: python -I fetch_enrichr_cache.py <functional_dir>
"""
import datetime
import hashlib
import sys
import urllib.request
from pathlib import Path

LIBS = ["MSigDB_Hallmark_2020", "Reactome_Pathways_2024"]
URL = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={}"
CACHE = Path.home() / ".cache" / "gseapy"


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main(functional: Path) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    manifest = functional / "MANIFEST.tsv"
    if not manifest.exists():
        manifest.write_text("id\tfile\turl\tdate_utc\tbytes\tsha256\n")
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for lib in LIBS:
        url = URL.format(lib)
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=300) as r:
            raw = r.read()
        rows, n_with_weight = [], 0
        for line in raw.decode("utf-8").splitlines():
            parts = line.rstrip("\n").split("\t")
            genes = [g.split(",")[0] for g in parts[2:]]
            n_with_weight += sum("," in g for g in parts[2:])
            genes = [g for g in genes if g]
            rows.append("\t".join([parts[0], parts[1], *genes]))
        text = ("\n".join(rows) + "\n").encode("utf-8")
        (functional / f"Enrichr.{lib}.server_response.txt").write_bytes(raw)
        out = CACHE / f"Enrichr.{lib}.gmt"
        out.write_bytes(text)
        with manifest.open("a") as m:
            m.write(f"enrichr\tEnrichr.{lib}.server_response.txt\t{url}\t{now}\t{len(raw)}\t{sha256(raw)}\n")
            m.write(f"enrichr_cache\t{out}\t{url}\t{now}\t{len(text)}\t{sha256(text)}\n")
        print(lib, "terms", len(rows), "raw_sha256", sha256(raw), "cache_sha256", sha256(text),
              "raw_bytes", len(raw), "cache_bytes", len(text), "weights_cut", n_with_weight, "same", raw == text)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
