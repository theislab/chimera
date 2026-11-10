"""Texts generated from the declaration: Zenodo metadata, READMEs, size table, bundle table.

Everything here is a pure function of the declaration and of what was packed, with no time
stamp, so regenerating a text gives the same bytes until one of its inputs changes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path

import yaml

from .declaration import Bundle, Declaration

MAX_CONTENT_ROWS = 100


def gigabytes(n_bytes: float) -> str:
    """Human-readable size."""
    if n_bytes >= 10**9:
        return f"{n_bytes / 1e9:.2f} GB"
    if n_bytes >= 10**6:
        return f"{n_bytes / 1e6:.1f} MB"
    return f"{n_bytes / 1e3:.0f} kB"


def creators_and_keywords(repo: Path, citation_file: str) -> tuple[list[dict[str, str]], list[str]]:
    """Creators (Zenodo format) and keywords from the citation file of the repository."""
    cff = yaml.safe_load((repo / citation_file).read_text())
    creators = [{"name": f"{a['family-names']}, {a['given-names']}", "affiliation": a.get("affiliation", "")}
                for a in cff["authors"]]
    return creators, list(cff.get("keywords", []))


def unpack_sentence(declaration: Declaration) -> str:
    """The sentence that every record description ends with."""
    repo = declaration.deposit.code_repository.removeprefix("https://")
    return (f"Unpack in the root of a checkout of {repo} at [TAG] (tar -xzf <file> -C chimera) and verify "
            "with sha256sum -c SHA256SUMS; REPRODUCE.md of the release describes every bundle. Code: [DOI].")


def zenodo_metadata(bundle: Bundle, declaration: Declaration, creators: list[dict[str, str]],
                    keywords: list[str]) -> dict:
    """Deposit metadata of one record, with ``[TAG]`` and ``[DOI ...]`` placeholders."""
    dep = declaration.deposit
    related = [
        {"identifier": "[DOI]", "relation": "isSupplementTo", "resource_type": "software", "scheme": "doi"},
        {"identifier": f"{dep.code_repository}/tree/[TAG]", "relation": "isSupplementTo",
         "resource_type": "software", "scheme": "url"},
        {"identifier": "[DOI paper]", "relation": "isSupplementTo", "resource_type": "publication-article",
         "scheme": "doi"},
    ]
    related += [{"identifier": f"[DOI {other}]", "relation": "references", "resource_type": "dataset",
                 "scheme": "doi"} for other in declaration.bundles if other != bundle.name]
    related += [{"identifier": doi, "relation": "isDerivedFrom", "resource_type": "dataset", "scheme": "doi"}
                for doi in bundle.derived_from]
    metadata = {
        "upload_type": bundle.record_type,
        "title": f"{dep.name}: {bundle.title} (release [TAG])",
        "description": f"{bundle.description} {unpack_sentence(declaration)}",
        "creators": creators,
        "license": dep.license,
        "access_right": "open",
        "version": "[TAG]",
        "keywords": keywords,
        "related_identifiers": related,
    }
    if bundle.record_type == "publication":
        metadata["publication_type"] = "other"
    return {"metadata": metadata}


def _content_rows(manifest: Sequence[tuple[str, int]]) -> list[tuple[str, int, int]]:
    """Group members by folder at the deepest level that still gives a short table."""
    for depth in (3, 2, 1):
        files: Counter[str] = Counter()
        sizes: Counter[str] = Counter()
        for path, n_bytes in manifest:
            parts = path.split("/")
            key = "/".join(parts[:depth]) + ("/" if len(parts) > depth else "")
            files[key] += 1
            sizes[key] += n_bytes
        if len(files) <= MAX_CONTENT_ROWS or depth == 1:
            return [(key, files[key], sizes[key]) for key in sorted(files)]
    return []


def record_readme(bundle: Bundle, declaration: Declaration, manifest: Sequence[tuple[str, int]],
                  tarball_names: Sequence[str]) -> str:
    """README.md of one record: what it is, how to verify and unpack it, what is inside."""
    dep = declaration.deposit
    tarball = f"chimera_{bundle.name}.tar.gz"
    total = sum(n for _, n in manifest)
    lines = [
        f"# {dep.name}: {bundle.title}",
        "",
        bundle.description,
        "",
        f"This record is one bundle (`{bundle.name}`) of the data deposit of {dep.name}. The code is at",
        f"<{dep.code_repository}> (release [TAG]); `REPRODUCE.md` there lists all bundles and how they are used.",
        "",
        "## Files of this record",
        "",
        "| File | Content |",
        "|---|---|",
    ]
    if len(tarball_names) > 1:
        lines.append(f"| `{tarball}.part00` ... | The tarball in {len(tarball_names)} parts; join them with "
                     f"`cat {tarball}.part* > {tarball}` |")
    lines += [
        f"| `{tarball}` | {len(manifest)} files, {gigabytes(total)} unpacked; paths are relative to the root of the code repository |",
        "| `MANIFEST.tsv` | Every file of the tarball: path, bytes, sha256, and where it comes from |",
        "| `LICENSES.md` | Third-party sources of the deposit, their licenses, and whether they are redistributed or linked |",
    ]
    lines += [f"| `{Path(arc).name}` | Copy of `{arc}` from the tarball, for direct viewing |" for arc in bundle.loose_files]
    lines += [
        "| `SHA256SUMS` | sha256 of the files of this record |",
        "",
        "## Use",
        "",
        "```bash",
        "sha256sum -c SHA256SUMS",
        f"tar -xzf {tarball} -C /path/to/chimera     # a checkout of the code repository",
        "```",
        "",
    ]
    if bundle.unpacks_to:
        lines += [f"The files unpack to {bundle.unpacks_to}.", ""]
    lines += ["## Content", "", "| Path | Files | Size |", "|---|---|---|"]
    lines += [f"| `{key}` | {n} | {gigabytes(size)} |" for key, n, size in _content_rows(manifest)]
    if bundle.not_included:
        lines += ["", "## Not included", "", "| Path | Reason |", "|---|---|"]
        lines += [f"| `{e.path}` | {e.reason} |" for e in bundle.not_included]
    return "\n".join(lines) + "\n"


def bundle_table(declaration: Declaration) -> str:
    """Markdown table of the bundles (name, where it unpacks, content), for REPRODUCE.md."""
    lines = ["| Bundle | Unpacks to | Content |", "|---|---|---|"]
    lines += [f"| `{b.name}` | {b.unpacks_to} | {b.description} |" for b in declaration.bundles.values()]
    return "\n".join(lines) + "\n"


def size_table(declaration: Declaration, infos: dict[str, dict]) -> str:
    """Tab-separated table of the staged bundles and their sizes."""
    limit = declaration.deposit.record_limit_bytes
    header = ("bundle", "members", "uncompressed_GB", "record_GB", "fits_one_record", "upload_files",
              "staged_at", "repo_commit")
    rows = [header]
    for name in declaration.bundles:
        info = infos.get(name)
        if info is None:
            rows.append((name, "not staged", "", "", "", "", "", ""))
            continue
        fits = "yes" if info["record_bytes"] < limit else "no, split"
        rows.append((name, info["members"], f"{info['bytes_uncompressed'] / 1e9:.3f}",
                     f"{info['record_bytes'] / 1e9:.3f}", fits, len(info["upload_files"]),
                     info["staged_at"], info.get("repo_commit", "")))
    staged = [infos[n] for n in declaration.bundles if n in infos]
    rows.append(("TOTAL", sum(i["members"] for i in staged),
                 f"{sum(i['bytes_uncompressed'] for i in staged) / 1e9:.3f}",
                 f"{sum(i['record_bytes'] for i in staged) / 1e9:.3f}", "", "", "", ""))
    return "".join("\t".join(str(c) for c in row) + "\n" for row in rows)


def _bundle_rows(declaration: Declaration, infos: dict[str, dict]) -> Iterable[str]:
    for name, bundle in declaration.bundles.items():
        info = infos.get(name)
        if info is None:
            yield f"| `{name}` | {bundle.title} | not staged | | | |"
        else:
            yield (f"| `{name}` | {bundle.title} | {info['members']} | {gigabytes(info['bytes_uncompressed'])} | "
                   f"{gigabytes(info['record_bytes'])} | {info['staged_at']} at `{info.get('repo_commit', '?')}` |")


def staging_readme(declaration: Declaration, infos: dict[str, dict]) -> str:
    """README.md of the staging area."""
    dep = declaration.deposit
    private = all(i.get("private_rules_used") for i in infos.values()) if infos else False
    lines = [
        f"# Zenodo staging area of {dep.name}",
        "",
        "Generated by `release/zenodo/stage.py` of the code repository from `release/zenodo/bundles.yaml`.",
        "Every file in this folder is written by that script; edit the declaration or the repository, not this folder.",
        "Nothing here is uploaded, tagged or pushed by the script.",
        "",
        "## Bundles (one Zenodo record each)",
        "",
        "| Bundle | Record | Members | Unpacked | Record size | Staged |",
        "|---|---|---|---|---|---|",
        *_bundle_rows(declaration, infos),
        "",
        f"Private text rules applied to every staged bundle: {'yes' if private else 'NO'}.",
        "",
        "## Files",
        "",
        "| Path | Upload? | Content |",
        "|---|---|---|",
        "| `<bundle>/upload/` | yes, every file | The files of the record: `chimera_<bundle>.tar.gz` (members are repository-relative paths), `MANIFEST.tsv` (path, bytes, sha256, origin of every member), `README.md`, `LICENSES.md`, loose copies of selected members, `SHA256SUMS` |",
        "| `<bundle>/zenodo_metadata.json` | no; paste into the form or send to the API | Title, description, creators (from `CITATION.cff`), license, related identifiers, with placeholders |",
        "| `<bundle>/skipped.tsv` | no | What was left out of the bundle and why |",
        "| `<bundle>/scrubbed.tsv` | no | Members whose content was altered (dropped columns, replaced text) |",
        "| `<bundle>/review_hits.tsv` | no | Matches of review patterns in text members, for a look; not altered |",
        "| `<bundle>/members.tsv`, `<bundle>/staging_info.json` | no | What was packed from which source state; read by `--check` |",
        "| `size_table.tsv` | no | Members and sizes of every bundle |",
        "| `LICENSES.md` | (inside every record) | Third-party sources, licenses, redistributed or linked |",
        "| `_work/` | no | Derived files that are reused between runs |",
        "",
        f"Zenodo allows {dep.record_limit_bytes / 1e9:.0f} GB per record; a larger tarball is split into `.partNN` files automatically.",
        "",
        "## Keep it up to date",
        "",
        "```bash",
        "python release/zenodo/stage.py --out <this folder> --check     # exit 0: nothing is stale",
        "python release/zenodo/stage.py --out <this folder>             # rebuild what is stale, keep the rest",
        "```",
        "",
        "`--check` compares every bundle with the current repository and storage, and lists storage folders that",
        "are in no bundle and have no stated reason for being left out.",
        "",
        "## Placeholders",
        "",
        "`[TAG]` git tag of the release; `[DOI]` DOI of the code archive; `[DOI <bundle>]` reserved DOI of each data",
        "record; `[DOI paper]` DOI of the article (delete that entry if it is not known at upload).",
        "",
        "## Upload",
        "",
        "1. For each bundle create a new Zenodo upload, reserve its DOI, upload every file of `<bundle>/upload/`,",
        "   and fill the form from `<bundle>/zenodo_metadata.json`. Do not publish yet.",
        "2. Put the reserved DOIs into `REPRODUCE.md` and `README.md` of the code repository, commit, tag the release.",
        "3. Replace the placeholders in the draft records and publish them.",
        "4. Check one record from a machine without a Zenodo login: download the tarball, compare its sha256 with",
        "   `SHA256SUMS`, unpack one file and compare it with `MANIFEST.tsv`.",
        "",
        "## Not redistributed",
        "",
        f"{declaration.dropped_columns.reason} Raw data of every source and protein-language-model embeddings stay",
        "with their sources; `LICENSES.md` lists each source and the download manifests link each file.",
    ]
    return "\n".join(lines) + "\n"
