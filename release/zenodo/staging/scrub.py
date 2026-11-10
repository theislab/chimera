"""Changes made to member content on its way into a tarball.

Two things can alter a deposited file, and both are recorded in ``scrubbed.tsv``:

* columns named in ``dropped_columns`` are removed from tables (csv, tsv, parquet; csv and tsv
  also when gzipped), because their source does not allow redistribution;
* text rules (``text_rules`` of the declaration plus an optional private rules file) rewrite
  matches of ``replace`` patterns, list matches of ``review`` patterns in ``review_hits.tsv``,
  and stop the build when a ``blocked`` pattern is still present.

Every other file is packed byte for byte.
"""
from __future__ import annotations

import csv
import gzip
import io
import itertools
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .declaration import DroppedColumns, TextRules

DELIMITERS = {".csv": ",", ".tsv": "\t"}
csv.field_size_limit(sys.maxsize)


class BlockedText(RuntimeError):
    """A deposited text file still contains a blocked pattern."""


@dataclass
class Prepared:
    """Content of one member, ready for the tarball."""

    data: bytes | None                                   # None: stream the source file unchanged
    notes: list[str] = field(default_factory=list)       # what was altered (scrubbed.tsv)
    review_hits: list[str] = field(default_factory=list)  # matches of review patterns


def table_kind(path: Path) -> tuple[str, bool] | None:
    """Return (``.csv`` | ``.tsv`` | ``.parquet``, gzipped) for a table file, else None."""
    gzipped = path.suffix == ".gz"
    kind = Path(path.stem).suffix if gzipped else path.suffix
    if kind in DELIMITERS or (kind == ".parquet" and not gzipped):
        return kind, gzipped
    return None


def _open_text(path: Path, gzipped: bool):
    opener = gzip.open if gzipped else open
    return opener(path, "rt", newline="", encoding="utf-8", errors="surrogateescape")


def table_columns(path: Path, kind: str, gzipped: bool) -> list[str]:
    """Column names of a table, read from its header only."""
    if kind == ".parquet":
        import pyarrow.parquet as pq

        return list(pq.read_schema(path).names)
    with _open_text(path, gzipped) as handle:
        return next(csv.reader(handle, delimiter=DELIMITERS[kind]), [])


def _drop_delimited(path: Path, kind: str, gzipped: bool, drop: set[str]) -> bytes:
    """Rewrite a csv/tsv without the dropped columns; every other cell keeps its exact text."""
    out = io.StringIO()
    with _open_text(path, gzipped) as handle:
        reader = csv.reader(handle, delimiter=DELIMITERS[kind])
        writer = csv.writer(out, delimiter=DELIMITERS[kind], lineterminator="\n")
        header = next(reader)
        keep = [i for i, name in enumerate(header) if name not in drop]
        for row in itertools.chain([header], reader):
            writer.writerow([row[i] for i in keep if i < len(row)])
    return out.getvalue().encode("utf-8", errors="surrogateescape")


def _drop_parquet(path: Path, drop: set[str]) -> bytes:
    """Rewrite a parquet file without the dropped columns (schema and other columns unchanged)."""
    import pyarrow.parquet as pq

    table = pq.read_table(path).drop_columns(sorted(drop))
    metadata = dict(table.schema.metadata or {})
    if b"pandas" in metadata:  # keep the pandas column list in step with the columns
        pandas_meta = json.loads(metadata[b"pandas"])
        pandas_meta["columns"] = [c for c in pandas_meta.get("columns", []) if c.get("name") not in drop]
        metadata[b"pandas"] = json.dumps(pandas_meta).encode()
        table = table.replace_schema_metadata(metadata)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


class Scrubber:
    """Apply the dropped-column rule and the text rules to members."""

    def __init__(self, dropped: DroppedColumns, text: TextRules) -> None:
        self.dropped = dropped
        self.text = text
        self._replace = [(re.compile(r.pattern.encode()), r.replacement.encode(), r.note) for r in text.replace]
        # One search that tells whether any replace rule can match, so that most files are read once.
        self._any_replace = (re.compile(b"|".join(b"(?:" + r.pattern.encode() + b")" for r in text.replace))
                             if text.replace else None)
        self._review = [re.compile(p.encode()) for p in text.review]
        self._blocked = [re.compile(p.encode()) for p in text.blocked]

    def _is_text(self, path: Path) -> bool:
        suffix = Path(path.stem).suffix if path.suffix == ".gz" else path.suffix
        return suffix in self.text.suffixes

    def _check_text(self, arc: str, plain: bytes, out: Prepared) -> tuple[bytes, bool]:
        """Run the text rules on ``plain``; return the new bytes and whether they changed."""
        changed = False
        rules = self._replace if self._any_replace is not None and self._any_replace.search(plain) else []
        for regex, replacement, note in rules:
            plain, n = regex.subn(replacement, plain)
            if n:
                changed = True
                if note not in out.notes:      # several rules may share one note
                    out.notes.append(note)
        for regex in self._blocked:
            if regex.search(plain):
                raise BlockedText(f"{arc}: a blocked pattern is present; add a replace rule or leave the file out")
        hits = {m.group(0).decode(errors="replace") for regex in self._review for m in regex.finditer(plain)}
        out.review_hits += sorted(hits)
        return plain, changed

    def prepare(self, arc: str, source: Path) -> Prepared:
        """Return the content of a member after all rules.

        ``Prepared.data`` is None when the source file can be streamed unchanged.
        """
        out = Prepared(data=None)
        gzipped = source.suffix == ".gz"
        plain: bytes | None = None       # uncompressed content, once it has been read
        changed = False
        kind = table_kind(source)
        if kind:
            hit = set(table_columns(source, *kind)) & self.dropped.columns
            if hit:
                plain = _drop_parquet(source, hit) if kind[0] == ".parquet" else _drop_delimited(source, *kind, hit)
                changed = True
                out.notes.append(f"{self.dropped.label} columns dropped: {','.join(sorted(hit))}")
        if self._is_text(source) and (plain is not None or source.stat().st_size <= self.text.max_bytes):
            if plain is None:
                raw = source.read_bytes()
                plain = gzip.decompress(raw) if gzipped else raw
            plain, text_changed = self._check_text(arc, plain, out)
            changed = changed or text_changed
        if plain is None or (gzipped and not changed):
            return out                       # stream the source file
        out.data = gzip.compress(plain, mtime=0) if gzipped else plain
        return out
