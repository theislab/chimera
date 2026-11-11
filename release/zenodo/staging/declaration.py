"""Read and validate ``bundles.yaml``, the declaration of the deposit.

The declaration is plain data. This module turns it into typed objects and rejects anything
it does not understand, so that a typing mistake in the file stops the run with a clear
message instead of silently changing what is deposited.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

FORMAT = 1
RECORD_TYPES = ("dataset", "publication")
DOI = re.compile(r"10\.\d{4,9}/\S+")


class DeclarationError(ValueError):
    """The declaration file is malformed."""


@dataclass(frozen=True)
class Deposit:
    """Settings shared by all records."""

    name: str
    code_repository: str
    license: str                            # license of the records, as Zenodo identifies it
    citation_file: str
    licenses_file: str
    docs_naming_every_bundle: tuple[str, ...]
    record_limit_bytes: int
    split_part_bytes: int
    license_name: str = ""                  # the license as a reader is told it; "" = the identifier
    release_tag: str = ""                   # identifiers of the archived release; "" = not declared
    code_doi: str = ""
    paper_doi: str = ""


@dataclass(frozen=True)
class NeverPacked:
    """File names that no selector ever packs."""

    names: frozenset[str]
    suffixes: tuple[str, ...]
    name_contains: tuple[str, ...]

    def matches(self, name: str) -> bool:
        """Return True when a file of this name must stay out of every bundle."""
        return (name in self.names or name.endswith(self.suffixes)
                or any(part in name for part in self.name_contains))


@dataclass(frozen=True)
class DroppedColumns:
    """Columns removed from every deposited table."""

    label: str
    columns: frozenset[str]
    reason: str


@dataclass(frozen=True)
class ReplaceRule:
    """A text pattern that is rewritten in deposited text files."""

    pattern: str
    replacement: str
    note: str


@dataclass(frozen=True)
class TextRules:
    """Checks applied to deposited text files (public rules plus optional private ones)."""

    suffixes: frozenset[str]
    max_bytes: int
    replace: tuple[ReplaceRule, ...]
    review: tuple[str, ...]
    blocked: tuple[str, ...]
    private_rules_used: bool = False


@dataclass(frozen=True)
class RunView:
    """Which files of a training run folder are taken."""

    name: str
    description: str
    requires: tuple[str, ...]
    if_unfinished: str                      # "error" or "skip"
    everything: bool
    top_level_names: frozenset[str]
    small_suffixes: tuple[str, ...]
    small_below_bytes: int
    subtrees: tuple[str, ...]
    files: tuple[str, ...]


@dataclass(frozen=True)
class SkipRule:
    """A rule that leaves files of a task folder out."""

    reason: str
    path_part_contains: str = ""
    name_regex: str = ""
    path_regex: str = ""
    label: str = ""


@dataclass(frozen=True)
class TaskFolderRules:
    """Rules for ``tasks_in`` selectors."""

    not_results_names: frozenset[str]
    not_results_name_contains: tuple[str, ...]
    not_results_reason: str
    listed_skips: tuple[SkipRule, ...]
    counted_skips: tuple[SkipRule, ...]
    compact_drop_suffixes: frozenset[str]
    compact_drop_folders: frozenset[str]
    compact_reason: str


@dataclass(frozen=True)
class Exclusion:
    """A path that is left out on purpose, with the reason."""

    path: str
    reason: str


@dataclass(frozen=True)
class Bundle:
    """One bundle of the deposit = one Zenodo record."""

    name: str
    title: str
    description: str
    record_type: str
    unpacks_to: str
    derived_from: tuple[str, ...]
    members: tuple[dict[str, Any], ...]
    not_included: tuple[Exclusion, ...]
    loose_files: tuple[str, ...]
    doi: str = ""                           # DOI of the record; "" = not declared

    def digest(self) -> str:
        """Hash of everything that decides the content of the tarball."""
        return _digest({"members": list(self.members), "loose_files": list(self.loose_files)})


@dataclass(frozen=True)
class Declaration:
    """The whole declaration."""

    deposit: Deposit
    never_packed: NeverPacked
    dropped_columns: DroppedColumns
    text_rules: TextRules
    run_views: dict[str, RunView]
    task_folders: TaskFolderRules
    bundles: dict[str, Bundle]
    coverage_roots: tuple[str, ...]
    not_deposited: tuple[Exclusion, ...]

    def rules_digest(self) -> str:
        """Hash of the rules that change member content or selection in every bundle."""
        return _digest({
            "never_packed": [sorted(self.never_packed.names), self.never_packed.suffixes,
                             self.never_packed.name_contains],
            "dropped_columns": sorted(self.dropped_columns.columns),
            "text_rules": [sorted(self.text_rules.suffixes), self.text_rules.max_bytes,
                           [(r.pattern, r.replacement) for r in self.text_rules.replace],
                           self.text_rules.blocked],
            "run_views": {k: [v.requires, v.everything, sorted(v.top_level_names), v.small_suffixes,
                              v.small_below_bytes, v.subtrees, v.files]
                          for k, v in sorted(self.run_views.items())},
            "task_folders": [sorted(self.task_folders.not_results_names),
                             self.task_folders.not_results_name_contains,
                             [(r.path_part_contains, r.name_regex, r.path_regex)
                              for r in self.task_folders.listed_skips + self.task_folders.counted_skips],
                             sorted(self.task_folders.compact_drop_suffixes),
                             sorted(self.task_folders.compact_drop_folders)],
        })


def _digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=list).encode()).hexdigest()


def _take(mapping: dict[str, Any], where: str, required: tuple[str, ...] = (),
          optional: tuple[str, ...] = ()) -> dict[str, Any]:
    """Check the keys of a mapping and return it with the optional keys filled with None."""
    if not isinstance(mapping, dict):
        raise DeclarationError(f"{where}: expected a mapping, got {type(mapping).__name__}")
    unknown = set(mapping) - set(required) - set(optional)
    missing = set(required) - set(mapping)
    if unknown:
        raise DeclarationError(f"{where}: unknown key(s) {sorted(unknown)}")
    if missing:
        raise DeclarationError(f"{where}: missing key(s) {sorted(missing)}")
    return {**{k: None for k in optional}, **mapping}


def _exclusions(items: Any, where: str) -> tuple[Exclusion, ...]:
    out = []
    for i, item in enumerate(items or []):
        d = _take(item, f"{where}[{i}]", required=("path", "reason"))
        out.append(Exclusion(str(d["path"]), " ".join(str(d["reason"]).split())))
    return tuple(out)


def _run_view(name: str, raw: dict[str, Any]) -> RunView:
    d = _take(raw, f"run_views.{name}", required=("requires", "if_unfinished"),
              optional=("description", "everything", "top_level", "subtrees", "files"))
    if d["if_unfinished"] not in ("error", "skip"):
        raise DeclarationError(f"run_views.{name}.if_unfinished must be 'error' or 'skip'")
    top = _take(d["top_level"] or {}, f"run_views.{name}.top_level", optional=("names", "small_files"))
    small = _take(top["small_files"] or {}, f"run_views.{name}.top_level.small_files",
                  optional=("suffixes", "below_bytes"))
    return RunView(
        name=name, description=d["description"] or "", requires=tuple(d["requires"]),
        if_unfinished=d["if_unfinished"], everything=bool(d["everything"]),
        top_level_names=frozenset(top["names"] or ()), small_suffixes=tuple(small["suffixes"] or ()),
        small_below_bytes=int(small["below_bytes"] or 0), subtrees=tuple(d["subtrees"] or ()),
        files=tuple(d["files"] or ()))


def _skip_rules(items: Any, where: str) -> tuple[SkipRule, ...]:
    out = []
    for i, item in enumerate(items or []):
        d = _take(item, f"{where}[{i}]", required=("reason",),
                  optional=("path_part_contains", "name_regex", "path_regex", "label"))
        out.append(SkipRule(reason=d["reason"], path_part_contains=d["path_part_contains"] or "",
                            name_regex=d["name_regex"] or "", path_regex=d["path_regex"] or "",
                            label=d["label"] or ""))
    return tuple(out)


def _task_folders(raw: dict[str, Any]) -> TaskFolderRules:
    d = _take(raw, "task_folders", required=("not_results", "compact"),
              optional=("listed_skips", "counted_skips"))
    nr = _take(d["not_results"], "task_folders.not_results", required=("reason",),
               optional=("names", "name_contains"))
    co = _take(d["compact"], "task_folders.compact", required=("reason",),
               optional=("drop_suffixes", "drop_folders"))
    return TaskFolderRules(
        not_results_names=frozenset(nr["names"] or ()),
        not_results_name_contains=tuple(nr["name_contains"] or ()),
        not_results_reason=nr["reason"],
        listed_skips=_skip_rules(d["listed_skips"], "task_folders.listed_skips"),
        counted_skips=_skip_rules(d["counted_skips"], "task_folders.counted_skips"),
        compact_drop_suffixes=frozenset(co["drop_suffixes"] or ()),
        compact_drop_folders=frozenset(co["drop_folders"] or ()),
        compact_reason=co["reason"])


def _doi(value: Any, where: str) -> str:
    """A declared DOI in its bare form (``10.5281/zenodo.N``), or "" when none is declared."""
    doi = str(value or "").strip()
    if doi and not DOI.fullmatch(doi):
        raise DeclarationError(f"{where}: {doi!r} is not a DOI (write it as 10.NNNN/suffix, without https://doi.org/)")
    return doi


def _bundle(name: str, raw: dict[str, Any]) -> Bundle:
    d = _take(raw, f"bundles.{name}", required=("title", "description", "members"),
              optional=("record_type", "unpacks_to", "derived_from", "not_included", "loose_files", "doi"))
    record_type = d["record_type"] or "dataset"
    if record_type not in RECORD_TYPES:
        raise DeclarationError(f"bundles.{name}.record_type must be one of {RECORD_TYPES}")
    if not d["members"]:
        raise DeclarationError(f"bundles.{name}.members is empty")
    description = " ".join(str(d["description"]).split())
    if "<" in description or ">" in description:
        raise DeclarationError(f"bundles.{name}.description: no angle brackets (Zenodo reads a description as "
                               "HTML and drops what stands between them)")
    return Bundle(
        name=name, title=d["title"], description=description,
        record_type=record_type, unpacks_to=d["unpacks_to"] or "",
        derived_from=tuple(d["derived_from"] or ()), members=tuple(d["members"]),
        not_included=_exclusions(d["not_included"], f"bundles.{name}.not_included"),
        loose_files=tuple(d["loose_files"] or ()), doi=_doi(d["doi"], f"bundles.{name}.doi"))


def _text_rules(raw: dict[str, Any], private: dict[str, Any] | None) -> TextRules:
    d = _take(raw, "text_rules", required=("suffixes", "max_mb"),
              optional=("replace", "review", "blocked"))
    p = _take(private or {}, "private rules", optional=("replace", "review", "blocked"))
    replace = []
    for i, item in enumerate((d["replace"] or []) + (p["replace"] or [])):
        r = _take(item, f"text rule replace[{i}]", required=("pattern", "with", "note"))
        replace.append(ReplaceRule(r["pattern"], r["with"], r["note"]))
    return TextRules(
        suffixes=frozenset(d["suffixes"]), max_bytes=int(float(d["max_mb"]) * 2**20),
        replace=tuple(replace), review=tuple((d["review"] or []) + (p["review"] or [])),
        blocked=tuple((d["blocked"] or []) + (p["blocked"] or [])),
        private_rules_used=private is not None)


def load(path: Path, private_rules: Path | None = None) -> Declaration:
    """Read the declaration at ``path``; merge the text rules of ``private_rules`` when given."""
    raw = yaml.safe_load(path.read_text())
    d = _take(raw, str(path), required=("format", "deposit", "never_packed", "dropped_columns",
                                        "text_rules", "run_views", "task_folders", "bundles",
                                        "coverage_roots"), optional=("not_deposited",))
    if d["format"] != FORMAT:
        raise DeclarationError(f"{path}: format {d['format']} is not supported (expected {FORMAT})")
    private = yaml.safe_load(private_rules.read_text()) if private_rules else None

    dep = _take(d["deposit"], "deposit", required=(
        "name", "code_repository", "license", "citation_file", "licenses_file",
        "record_limit_gb", "split_part_gb"),
        optional=("docs_naming_every_bundle", "license_name", "release_tag", "code_doi", "paper_doi"))
    npk = _take(d["never_packed"], "never_packed", optional=("names", "suffixes", "name_contains"))
    dc = _take(d["dropped_columns"], "dropped_columns", required=("label", "columns", "reason"))
    bundles = {name: _bundle(name, raw_b) for name, raw_b in d["bundles"].items()}
    for name in bundles:
        if not name.replace("_", "").isalnum() or name != name.lower():
            raise DeclarationError(f"bundle name {name!r}: use lower-case letters, digits and '_'")
    code_doi = _doi(dep["code_doi"], "deposit.code_doi")
    declared = [doi for doi in (code_doi, *(b.doi for b in bundles.values())) if doi]
    if len(declared) != len(set(declared)):
        raise DeclarationError("deposit.code_doi and the doi of every bundle must differ from each other")
    return Declaration(
        deposit=Deposit(
            name=dep["name"], code_repository=dep["code_repository"], license=dep["license"],
            citation_file=dep["citation_file"], licenses_file=dep["licenses_file"],
            docs_naming_every_bundle=tuple(dep["docs_naming_every_bundle"] or ()),
            record_limit_bytes=int(float(dep["record_limit_gb"]) * 10**9),
            split_part_bytes=int(float(dep["split_part_gb"]) * 10**9),
            license_name=str(dep["license_name"] or dep["license"]).strip(),
            release_tag=str(dep["release_tag"] or "").strip(), code_doi=code_doi,
            paper_doi=_doi(dep["paper_doi"], "deposit.paper_doi")),
        never_packed=NeverPacked(frozenset(npk["names"] or ()), tuple(npk["suffixes"] or ()),
                                 tuple(npk["name_contains"] or ())),
        dropped_columns=DroppedColumns(dc["label"], frozenset(dc["columns"]),
                                       " ".join(str(dc["reason"]).split())),
        text_rules=_text_rules(d["text_rules"], private),
        run_views={name: _run_view(name, v) for name, v in d["run_views"].items()},
        task_folders=_task_folders(d["task_folders"]),
        bundles=bundles,
        coverage_roots=tuple(d["coverage_roots"]),
        not_deposited=_exclusions(d["not_deposited"], "not_deposited"))
