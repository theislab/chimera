"""Write the files of one bundle: tarball, manifest, checksums and the staging records.

A bundle is built in a temporary folder beside its final place and moved there only when
every file is complete, so an interrupted run never leaves a bundle that looks finished.

Layout of a finished bundle folder::

    <bundle>/
        upload/                      exactly the files of the Zenodo record
            chimera_<bundle>.tar.gz    members are repository-relative paths
            MANIFEST.tsv               every member: path, bytes, sha256, origin
            README.md, LICENSES.md     what the record is; third-party sources
            <loose files>              copies of selected members for direct viewing
            SHA256SUMS                 sha256 of the files above
        zenodo_metadata.json         deposit metadata with placeholders
        skipped.tsv                  what was left out, with the reason
        scrubbed.tsv                 members whose content was altered
        review_hits.tsv              matches of review patterns (not altered)
        members.tsv                  members with the size and time of their source (for --check)
        staging_info.json            sizes, checksums, fingerprint, commit, time
"""
from __future__ import annotations

import datetime as dt
import gzip
import hashlib
import io
import json
import os
import shutil
import subprocess
import tarfile
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

from .declaration import Bundle, Declaration
from .scrub import Scrubber
from .selection import Member, Selection

PACK_FORMAT = 1          # raise when the layout of the tarball or the way members are altered changes
CHUNK = 2**24
BUILDING_PREFIX = ".building-"
RETIRED_PREFIX = ".retired-"


class PackError(RuntimeError):
    """A bundle could not be written."""


@dataclass
class TarballResult:
    """What writing a tarball produced."""

    manifest: list[tuple[str, int, str, str]]      # path, bytes, sha256, origin
    scrubbed: list[tuple[str, str]]                # path, change
    review_hits: list[tuple[str, str]]             # path, match
    loose: dict[str, bytes]                        # member path -> content, for the loose files


class _HashingReader:
    """File wrapper that hashes what is read through it."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.sha256 = hashlib.sha256()

    def read(self, size: int = -1) -> bytes:
        block = self._handle.read(size)
        self.sha256.update(block)
        return block


def sha256_file(path: Path) -> str:
    """sha256 of a file, read in blocks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(declaration: Declaration, selection: Selection) -> str:
    """Hash of everything that decides the bytes of a tarball.

    It covers the member list (member path, origin, size and modification time of each source
    file), the rules that alter content, and the pack format. Two runs with the same
    fingerprint produce the same tarball, so the bundle need not be rebuilt.
    """
    digest = hashlib.sha256(f"pack-format {PACK_FORMAT}\n{declaration.rules_digest()}\n".encode())
    for m in selection.members:
        digest.update(f"{m.arc}\t{m.origin}\t{m.derive}\t{m.size}\t{m.mtime_ns}\n".encode())
    return digest.hexdigest()


def obs_table(h5ad: Path, cache: Path) -> Path:
    """Write the obs table of an AnnData file as csv.gz into ``cache`` (reused while newer than the source)."""
    if cache.is_file() and cache.stat().st_mtime_ns >= h5ad.stat().st_mtime_ns:
        return cache
    import anndata
    import h5py

    read_elem = anndata.io.read_elem if hasattr(anndata, "io") else anndata.experimental.read_elem
    with h5py.File(h5ad, "r") as handle:
        obs = read_elem(handle["obs"])
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_name(cache.name + ".tmp")
    with (open(tmp, "wb") as raw,
          gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0) as gz,
          io.TextIOWrapper(gz, encoding="utf-8", newline="") as text):
        obs.to_csv(text, index_label="cell_barcode")
    tmp.replace(cache)
    return cache


def _compressor(destination: BinaryIO, jobs: int) -> subprocess.Popen | None:
    """Start pigz writing to ``destination``; None when pigz is not installed."""
    if shutil.which("pigz") is None:
        return None
    return subprocess.Popen(["pigz", "-p", str(max(1, jobs)), "-6", "-n"], stdin=subprocess.PIPE, stdout=destination)


def _tar_info(arc: str, size: int, mtime_ns: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(arc)
    info.size, info.mtime = size, mtime_ns // 10**9
    info.mode, info.uid, info.gid, info.uname, info.gname = 0o644, 0, 0, "", ""
    return info


def _add_member(tar: tarfile.TarFile, member: Member, scrubber: Scrubber, work: Path,
                loose: Sequence[str], result: TarballResult) -> None:
    source = obs_table(member.source, work / "obs_tables" / member.arc) if member.derive == "obs_table" else member.source
    prepared = scrubber.prepare(member.arc, source)
    origin = member.origin
    if prepared.notes:
        result.scrubbed += [(member.arc, note) for note in prepared.notes]
        origin = f"derived: {origin.removeprefix('derived: ')}; {'; '.join(prepared.notes)}"
    result.review_hits += [(member.arc, hit) for hit in prepared.review_hits]
    if prepared.data is None and member.arc in loose:
        prepared.data = source.read_bytes()
    if prepared.data is not None:
        tar.addfile(_tar_info(member.arc, len(prepared.data), member.mtime_ns), io.BytesIO(prepared.data))
        digest, size = hashlib.sha256(prepared.data).hexdigest(), len(prepared.data)
        if member.arc in loose:
            result.loose[member.arc] = prepared.data
    else:
        size = source.stat().st_size
        with open(source, "rb") as handle:
            reader = _HashingReader(handle)
            tar.addfile(_tar_info(member.arc, size, member.mtime_ns), reader)
        digest = reader.sha256.hexdigest()
    result.manifest.append((member.arc, size, digest, origin))


def write_tarball(members: Iterable[Member], destination: Path, scrubber: Scrubber, work: Path,
                  jobs: int, loose: Sequence[str] = ()) -> TarballResult:
    """Write the gzipped tarball of ``members`` to ``destination``.

    Members are stored under their repository-relative path, owned by nobody, mode 644, with
    the modification time of their source, so the same inputs give the same archive.
    """
    result = TarballResult([], [], [], {})
    with open(destination, "wb") as raw:
        pigz = _compressor(raw, jobs)
        stream: BinaryIO = pigz.stdin if pigz else gzip.GzipFile(filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0)
        try:
            with tarfile.open(fileobj=stream, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for member in members:
                    _add_member(tar, member, scrubber, work, loose, result)
        finally:
            stream.close()
            if pigz and pigz.wait() != 0:
                raise PackError(f"pigz failed while writing {destination.name}")
    return result


def split_file(path: Path, limit_bytes: int, part_bytes: int) -> list[Path]:
    """Split ``path`` into ``<name>.part00``, ... when it is larger than 95 % of the record limit."""
    if path.stat().st_size <= 0.95 * limit_bytes:
        return [path]
    parts: list[Path] = []
    with open(path, "rb") as source:
        while True:
            part = path.with_name(f"{path.name}.part{len(parts):02d}")
            written = 0
            with open(part, "wb") as out:
                while written < part_bytes and (block := source.read(min(CHUNK, part_bytes - written))):
                    out.write(block)
                    written += len(block)
            if not written:
                part.unlink()
                break
            parts.append(part)
    path.unlink()
    return parts


def write_tsv(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write a tab-separated table."""
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\t".join(header) + "\n")
        for row in rows:
            handle.write("\t".join(str(cell) for cell in row) + "\n")


def write_checksums(upload: Path, known: dict[str, str] | None = None) -> dict[str, str]:
    """Write ``SHA256SUMS`` for the files of ``upload`` and return {file name: sha256}.

    ``known`` gives digests that need not be computed again (the tarball of an unchanged bundle).
    """
    digests = {}
    for path in sorted(p for p in upload.iterdir() if p.is_file() and p.name != "SHA256SUMS"):
        digests[path.name] = (known or {}).get(path.name) or sha256_file(path)
    (upload / "SHA256SUMS").write_text("".join(f"{d}  {name}\n" for name, d in digests.items()))
    return digests


def swap_into_place(building: Path, final: Path) -> None:
    """Replace ``final`` by ``building`` so that ``final`` is always a complete bundle folder."""
    retired = final.with_name(RETIRED_PREFIX + final.name)
    shutil.rmtree(retired, ignore_errors=True)
    if final.exists():
        final.rename(retired)
    building.rename(final)
    shutil.rmtree(retired, ignore_errors=True)


def remove_leftovers(out: Path) -> None:
    """Remove temporary folders that an interrupted run left behind."""
    for path in out.glob(BUILDING_PREFIX + "*"):
        shutil.rmtree(path, ignore_errors=True)
    for path in out.glob(RETIRED_PREFIX + "*"):
        shutil.rmtree(path, ignore_errors=True)


def build_bundle(bundle: Bundle, declaration: Declaration, selection: Selection, out: Path,
                 upload_texts: Callable[[list[tuple[str, int, str, str]]], dict[str, str]],
                 metadata: dict, jobs: int, repo_state: dict[str, object]) -> dict:
    """Pack ``bundle`` into ``out/<bundle>`` and return its staging information.

    ``upload_texts`` receives the manifest rows and returns the text files that go into the
    record beside the tarball ({file name: text}: README.md, LICENSES.md). ``metadata`` is
    written as ``zenodo_metadata.json``.
    """
    building = out / (BUILDING_PREFIX + bundle.name)
    shutil.rmtree(building, ignore_errors=True)
    try:
        info = _build_in(building, bundle, declaration, selection, out, upload_texts, metadata, jobs, repo_state)
    except BaseException:
        shutil.rmtree(building, ignore_errors=True)   # the staged bundle, if any, stays as it was
        raise
    swap_into_place(building, out / bundle.name)
    return info


def _build_in(building: Path, bundle: Bundle, declaration: Declaration, selection: Selection, out: Path,
              upload_texts: Callable[[list[tuple[str, int, str, str]]], dict[str, str]],
              metadata: dict, jobs: int, repo_state: dict[str, object]) -> dict:
    """Write every file of the bundle into the temporary folder ``building``."""
    upload = building / "upload"
    upload.mkdir(parents=True)
    os.chmod(building, 0o700)

    tarball = upload / f"chimera_{bundle.name}.tar.gz"
    scrubber = Scrubber(declaration.dropped_columns, declaration.text_rules)
    missing_loose = set(bundle.loose_files) - {m.arc for m in selection.members}
    if missing_loose:
        raise PackError(f"bundle {bundle.name}: loose file(s) {sorted(missing_loose)} are not members")
    result = write_tarball(selection.members, tarball, scrubber, out / "_work", jobs, bundle.loose_files)
    parts = split_file(tarball, declaration.deposit.record_limit_bytes, declaration.deposit.split_part_bytes)

    write_tsv(upload / "MANIFEST.tsv", ("path", "bytes", "sha256", "origin"), result.manifest)
    for arc, data in result.loose.items():
        (upload / Path(arc).name).write_bytes(data)
    for name, text in upload_texts(result.manifest).items():
        (upload / name).write_text(text, encoding="utf-8")
    digests = write_checksums(upload)

    write_tsv(building / "skipped.tsv", ("path", "reason"), selection.skipped)
    write_tsv(building / "scrubbed.tsv", ("path", "change"), result.scrubbed)
    write_tsv(building / "review_hits.tsv", ("path", "match"), result.review_hits)
    write_tsv(building / "members.tsv", ("path", "bytes", "mtime_ns", "origin"),
              ((m.arc, m.size, m.mtime_ns, m.origin) for m in selection.members))
    (building / "zenodo_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    files = {p.name: p.stat().st_size for p in sorted(upload.iterdir())}
    info = {
        "bundle": bundle.name,
        "members": len(result.manifest),
        "bytes_uncompressed": sum(row[1] for row in result.manifest),
        "tarball_parts": [p.name for p in parts],
        "upload_files": files,
        "record_bytes": sum(files.values()),
        "sha256": digests,
        "n_skipped": len(selection.skipped),
        "n_scrubbed": len(result.scrubbed),
        "n_review_hits": len(result.review_hits),
        "fingerprint": fingerprint(declaration, selection),
        "pack_format": PACK_FORMAT,
        "private_rules_used": declaration.text_rules.private_rules_used,
        "staged_at": dt.datetime.now().isoformat(timespec="seconds"),
        **repo_state,
    }
    (building / "staging_info.json").write_text(json.dumps(info, indent=2) + "\n")
    return info
