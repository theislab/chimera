"""Human CDS length and GC per Ensembl gene, from the Ensembl 110 GTF and CDS FASTA.

The CDS of the ``Ensembl_canonical`` transcript is used; genes without a canonical
protein-coding transcript in the FASTA fall back to their longest CDS.
"""

from __future__ import annotations

import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.common import gene_ids

ENSEMBL_DIR = gene_ids.REPO_ROOT / "data" / "analysis" / "ensembl110"
GTF = "Homo_sapiens.GRCh38.110.gtf.gz"
CDS_FASTA = "Homo_sapiens.GRCh38.cds.all.fa.gz"


def canonical_transcripts(gtf: Path) -> dict[str, str]:
    """gene_id -> canonical transcript_id (unversioned)."""
    out = {}
    pat_g = re.compile(r'gene_id "([^"]+)"')
    pat_t = re.compile(r'transcript_id "([^"]+)"')
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.split("\t", 8)
            if len(parts) < 9 or parts[2] != "transcript" or "Ensembl_canonical" not in parts[8]:
                continue
            out[pat_g.search(parts[8]).group(1)] = pat_t.search(parts[8]).group(1)
    return out


def _read_fasta(path: Path):
    tid = gid = None
    seq: list[str] = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.startswith(">"):
                if tid is not None:
                    yield tid, gid, "".join(seq)
                head = line[1:].split()
                tid = head[0].split(".")[0]
                gid = next((h[5:].split(".")[0] for h in head if h.startswith("gene:")), None)
                seq = []
            else:
                seq.append(line.strip())
    if tid is not None:
        yield tid, gid, "".join(seq)


def human_cds_table(ensembl_dir: str | Path = ENSEMBL_DIR) -> pd.DataFrame:
    """Per human gene: ``cds_length`` (nt), ``gc``, ``gc3``, ``cds_source`` (canonical or longest)."""
    ensembl_dir = Path(ensembl_dir)
    canon = canonical_transcripts(ensembl_dir / GTF)
    canon_t = set(canon.values())
    best: dict[str, tuple] = {}
    for tid, gid, s in _read_fasta(ensembl_dir / CDS_FASTA):
        if gid is None or not s:
            continue
        s = s.upper()
        is_canon = tid in canon_t
        cur = best.get(gid)
        if cur is None or (is_canon and not cur[1]) or (is_canon == cur[1] and len(s) > len(cur[0])):
            best[gid] = (s, is_canon)
    rows = []
    for gid, (s, is_canon) in best.items():
        n = len(s)
        gc = (s.count("G") + s.count("C")) / n
        third = s[2::3]
        gc3 = (third.count("G") + third.count("C")) / max(len(third), 1)
        rows.append((gid, n, gc, gc3, "canonical" if is_canon else "longest"))
    df = pd.DataFrame(rows, columns=["hsapiens_gene_id", "cds_length", "gc", "gc3", "cds_source"])
    df["log_cds_length"] = np.log10(df["cds_length"])
    return df.set_index("hsapiens_gene_id")
