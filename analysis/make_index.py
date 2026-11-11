"""Index of the analysis tasks: one table, three generated views.

``analysis/tasks/INDEX.tsv`` has one row per task folder (what it does, how it is run, what it
writes, where the paper uses it). ``analysis/tasks/PAPER.tsv`` names the figures, tables and
sections of the paper by their LaTeX label. From the two tables this script writes

- the task tables of ``analysis/README.md`` (between the ``index`` markers),
- the paper map of ``REPRODUCE.md`` (between the ``paper-map`` markers),
- ``analysis/tasks/<folder>/README.md`` for every task folder. A README that was written by hand
  is kept; only its "At a glance" block (between markers) is refreshed.

So the index, the map and the folder READMEs cannot disagree. Edit the tables, then run, from the
repository root (standard library only, any Python 3.10+):

    python analysis/make_index.py            # rewrite the generated views
    python analysis/make_index.py --check    # write nothing; exit 1 if a view is stale or a row is wrong
    python analysis/make_index.py --paper-aux <manuscript build>/main.aux   # PAPER.tsv from a build: numbers, order, figures

``--check`` also verifies the rows: every task folder has exactly one row, every script and
module named in ``entry`` exists, every label in ``paper`` is in ``PAPER.tsv``, and (when the
results are on disk) every result folder named in ``outputs`` exists. It warns, without failing,
when a figure of ``PAPER.tsv`` and ``figures/manifest.tsv`` disagree on its key or when a figure
label of the manifest has no row: then correct ``PAPER.tsv`` (columns ``figure_key`` and ``title``).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "analysis" / "tasks"
INDEX = TASKS / "INDEX.tsv"
PAPER = TASKS / "PAPER.tsv"
COLUMNS = ["folder", "group", "status", "summary", "entry", "env", "outputs", "paper", "basis", "notes"]

# Groups in the order of the paper: key -> (title, the question, where in the paper).
GROUPS = {
    "data": ("Data", "Which objects, ortholog tables and independent references do the analyses read?", "Methods; Supplementary Table of data preparation"),
    "training": ("Model fits beyond the canonical runs", "Which further seeds, platform repeats and cohort recipes were trained, and how was each recipe chosen?", "Extended Methods, seeds and recipes"),
    "integration": ("Cell-level integration", "Does one fit align the cells across species, and how does it compare with other tools?", "Results 3.1"),
    "axis": ("Δα as an axis of its own", "Is deployment relative to the protein prior separate from protein sequence?", "Results 3.2"),
    "findings": ("What the readouts find in liver and testis", "Which genes, cell types, pathways and gene classes sit above or below the protein prior?", "Results 3.3"),
    "validation": ("Pre-specified validation", "Do independent data back Δα and the count evidence S_g, and how do direct baselines compare?", "Results 3.4"),
    "reliability": ("Reliability and controls", "How stable are the readouts across seeds, and what do they read from counts and from features?", "Results 3.4"),
    "per_gene_gate": ("The per-gene gate", "What does Δα show when it is read gene by gene from counts, without the protein-feature network?", "Results 3.4"),
    "adds": ("What the count evidence adds", "What do the readouts add to direct baselines and to other tools, on real and on ground-truth data?", "Results 3.4"),
    "tissues": ("One reading across tissues", "Does the reading hold across tissues, and what goes with deployment above the prior?", "Results 3.5"),
    "disease": ("Disease", "Which human disease responses does a mouse model reproduce, and where do drug targets and disease genes sit?", "Results 3.5 (drug targets); Supplementary Results (disease case studies)"),
    "protein": ("Protein space", "Is the residue-level attribution specific?", "Results 3.6"),
    "figures": ("Figures and tables", "Which scripts draw panels or measure table values from cached results?", "all figures"),
    "infrastructure": ("Infrastructure", "Environment and checkpoint checks.", ""),
}
STATUS = {
    "paper": "in the paper",
    "supporting": "supporting (builds inputs, launches runs, or a check reported as a method detail)",
    "exploratory": "exploratory, not in the paper",
    "recipe": "recipe search",
    "infrastructure": "infrastructure",
}
# A task whose result a later task replaced has the status ``retired:<folder of the later task>``. Its code, manifests
# and results stay as the record of what was run; the paper uses the later task.
RETIRED = "retired:"


def status_text(status: str, short: bool = False) -> str:
    if status.startswith(RETIRED):
        return f"retired, replaced by `{status[len(RETIRED):]}`" + ("" if short else " (kept as the record of what was run; the paper uses the later task)")
    return STATUS[status].split(" (")[0] if short else STATUS[status]


BASIS = {
    "numbers": "numbers printed in the paper are read from this task's result files",
    "figure": "a script of this task draws panels of the figure",
    "methods": "described in the Extended Methods",
    "upstream": "read by tasks whose results are in the paper",
    "-": "",
}
# Which deposit bundle holds an output; first match wins. Mirrors release/zenodo/bundles.yaml, which is the authority
# (python release/zenodo/stage.py --dry-run lists the members of every bundle).
BUNDLES = [
    (r"results/analysis/(T5_1|T5_2|T5_3|T5_6)(/|$)", "bundle `tool_runs_saturn_camex`"),
    (r"results/analysis/(B1|B2)(/|$)", "bundle `gene_level_benchmark`"),
    (r"results/analysis/T2_2/values\.csv$", "bundle `analysis_results`"),
    (r"results/analysis/T2_2(/|$)", "not deposited"),
    (r"results/analysis/", "bundle `analysis_results`"),
    (r"data/analysis/O1/repair_", "bundle `data_identifier_joined`"),
    (r"data/analysis/(B1|B2|D1|G1|T5_inputs)(/|$)", "not deposited (rebuilt by the task code)"),
    (r"data/analysis/(E\d|Y\d|gold|functional|ensembl110|glaucoma|x3_annotations|y5_annotations)(/|$)",
     "third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`"),
    (r"data/analysis/", "bundle `data_other_cohorts`"),
    (r"training/runs/o2-", "bundle `data_identifier_joined`"),
    (r"training/runs/(z3-|z4-)", "bundle `models_per_gene_gate`"),
    (r"training/runs/(b1|b2-)", "bundle `gene_level_benchmark` (configuration, summary and final state)"),
    (r"training/runs/(rev-|g1-)", "bundle `models_seeds_controls_cohorts`"),
    (r"training/runs/(z2-|d1-|o1-)", "not deposited (the tables computed from these runs are)"),
    (r"training/runs/baselines_", "bundle `notebook_caches`"),
    (r"notebooks/cache", "bundle `notebook_caches`"),
    (r"notebooks/figures|figures/panels", "tracked in git"),
]


def bundle(output: str) -> str:
    return next(text for pat, text in BUNDLES if re.match(pat, output))


GENERATED = "<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->"
BEGIN, END = "<!-- {name}:begin (generated by analysis/make_index.py; do not edit by hand) -->", "<!-- {name}:end -->"
MODULE = re.compile(r"\banalysis\.tasks\.([A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)+)")
FILEPATH = re.compile(r"\banalysis/(?:tasks|slurm)/[A-Za-z0-9_./-]*[A-Za-z0-9_]")
RESULT = re.compile(r"results/analysis/([A-Za-z0-9_.]+)")


def read_tsv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))


def items(cell: str) -> list[str]:
    return [x.strip() for x in cell.split(" ; ") if x.strip()]


def task_folders() -> list[str]:
    return sorted(p.name for p in TASKS.iterdir() if p.is_dir() and p.name != "__pycache__")


# ---------------------------------------------------------------- paper locations

def paper_names(paper: list[dict]) -> dict[str, dict]:
    return {r["label"]: r for r in paper}


def merged_where(cell: str, names: dict[str, dict], with_label: bool = False) -> str:
    """Paper cell of a row as text; panels of one figure are merged ('Figure 5a, b, c')."""
    out: list[tuple[str, list[str]]] = []
    for it in items(cell):
        m = re.fullmatch(r"(\S+)(?: ([a-z]))?", it)
        label, letter = m.group(1), m.group(2)
        if out and out[-1][0] == label and letter:
            out[-1][1].append(letter)
        else:
            out.append((label, [letter] if letter else []))
    return "; ".join(names[label]["where"] + ", ".join(letters) + (f" (`{label}`)" if with_label else "") for label, letters in out)


# ---------------------------------------------------------------- what a task reads (from its code)

def code_inputs(folder: str, owners: dict[str, list[str]]) -> tuple[list[str], list[str], list[str]]:
    """(result folders of other tasks, data roots, task packages imported) named in the code of a task."""
    text = ""
    for p in sorted((TASKS / folder).rglob("*")):
        if p.suffix in (".py", ".sh", ".sbatch", ".R") and "__pycache__" not in p.parts:
            text += p.read_text(encoding="utf-8", errors="replace") + "\n"
    joined = re.sub(r'"\s*/\s*"', "/", text)  # "results" / "analysis" / "T4_3" -> results/analysis/T4_3
    results = sorted({m for m in RESULT.findall(joined) if folder not in owners.get(m, []) and m in owners})
    data = set()
    for pat, name in ((r"data/processed", "data/processed/"), (r"data/raw", "data/raw/"), (r"training/runs|final_state\.npz", "training/runs/"),
                      (r"notebooks/cache", "notebooks/cache/")):
        if re.search(pat, joined):
            data.add(name)
    data |= {f"data/analysis/{m}/" for m in re.findall(r"data/analysis/([A-Za-z0-9_]+)/", joined)}
    imports = sorted({m.split(".")[0] for m in MODULE.findall(text)} - {folder})
    imports = [m for m in imports if (TASKS / m).is_dir()]
    return results, sorted(data), imports


# ---------------------------------------------------------------- views

def run_lines(entry: str) -> list[str]:
    return items(entry)


def readme_tables(rows: list[dict], names: dict[str, dict]) -> str:
    out = []
    for key, (title, question, _) in GROUPS.items():
        group = [r for r in rows if r["group"] == key]
        if not group:
            continue
        out += [f"### {title}", "", question, "", "| Task folder | What it does | Run (first step) | Writes | Paper | Status |", "|---|---|---|---|---|---|"]
        for r in group:
            cmds = run_lines(r["entry"])
            run = f"`{cmds[0]}`" + (f" (+{len(cmds) - 1} more steps)" if len(cmds) > 1 else "")
            writes = "<br>".join(f"`{o}`" for o in items(r["outputs"]))
            paper = merged_where(r["paper"], names) if r["paper"] else "not in the paper"
            out.append(f"| [`{r['folder']}`](tasks/{r['folder']}/README.md) | {r['summary']} | {run} | {writes} | {paper} | {status_text(r['status'], short=True)} |")
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def glance(r: dict, names: dict[str, dict], owners: dict[str, list[str]]) -> str:
    title, question, _ = GROUPS[r["group"]]
    results, data, imports = code_inputs(r["folder"], owners)
    lines = [r["summary"], "", "| | |", "|---|---|",
             f"| Question | {title}: {question} |",
             f"| Status | {status_text(r['status'])} |",
             f"| Environment | {', '.join(f'`{e}`' if re.fullmatch(r'[a-z_]+_env', e) else e for e in items(r['env']))} |"]
    if r["paper"]:
        lines.append(f"| Used in the paper | {merged_where(r['paper'], names, with_label=True)} |")
        lines.append(f"| How we know | {'; '.join(BASIS[b] for b in items(r['basis']))} |")
    else:
        lines.append("| Used in the paper | no |")
    lines += ["", "**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):", ""]
    lines += [f"    {c}" for c in run_lines(r["entry"])]
    lines += ["", "**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)", ""] + [f"- `{o}`: {bundle(o)}" for o in items(r["outputs"])]
    reads = [f"results of {', '.join(f'`{o}`' for o in sorted(set(sum((owners[m] for m in results), []))))}"] if results else []
    reads += [f"`{d}`" for d in data]
    if reads or imports:
        lines += ["", "**Reads** (as named in the code)", ""] + [f"- {x}" for x in reads]
        if imports:
            lines.append(f"- code of {', '.join(f'`{m}`' for m in imports)}")
    if r["notes"]:
        lines += ["", "**Notes.** " + r["notes"]]
    return "\n".join(lines) + "\n"


def task_readme(r: dict, names: dict[str, dict], owners: dict[str, list[str]], old: str | None) -> str:
    if old is None or old.startswith(GENERATED):
        return f"{GENERATED}\n# {r['folder']}\n\n{glance(r, names, owners)}"
    return replace_block(old, "glance", glance(r, names, owners), after_first_heading=True)


def replace_block(text: str, name: str, body: str, after_first_heading: bool = False) -> str:
    begin, end = BEGIN.format(name=name), END.format(name=name)
    block = f"{begin}\n{body}{end}\n"
    if begin in text:
        head, rest = text.split(begin, 1)
        return head + block + rest.split(end + "\n", 1)[1] if end + "\n" in rest else head + block + rest.split(end, 1)[1]
    if not after_first_heading:
        raise SystemExit(f"marker {begin!r} not found")
    lines = text.split("\n")
    at = next((i + 1 for i, line in enumerate(lines) if line.startswith("# ")), 0)
    return "\n".join(lines[:at]) + "\n\n" + block + "\n" + "\n".join(lines[at:]).lstrip("\n")


def figure_scripts() -> dict[str, list[str]]:
    """Figure label -> the scripts that draw its panels, in panel order, from ``figures/manifest.json``."""
    manifest = ROOT / "figures" / "manifest.json"
    out: dict[str, list[str]] = {}
    if manifest.is_file():
        for fig in json.loads(manifest.read_text(encoding="utf-8"))["figures"]:
            scripts = out.setdefault(fig["label"], [])
            for panel in fig["panels"]:
                if panel.get("script") and panel["script"] not in scripts:
                    scripts.append(panel["script"])
    return out


def paper_map(rows: list[dict], paper: list[dict]) -> str:
    """One row per figure, table and section of the paper: the tasks behind it.

    A figure row whose ``code`` cell is empty names the scripts of its panels from the figure manifest. A title that
    is an extended caption names its figure by the printed number ("Extended caption of Figure 1").
    """
    scripts, names = figure_scripts(), paper_names(paper)
    by_label: dict[str, list[str]] = {}
    for r in rows:
        for it in items(r["paper"]):
            m = re.fullmatch(r"(\S+)(?: ([a-z]))?", it)
            entry = f"`{r['folder']}`" + (f" ({m.group(2)})" if m.group(2) else "")
            if entry.split(" ")[0] not in [e.split(" ")[0] for e in by_label.setdefault(m.group(1), [])] or m.group(2):
                by_label[m.group(1)].append(entry)
    out = ["| In the paper | Label | What | Task folders behind it | Other code behind it |", "|---|---|---|---|---|"]
    for p in paper:
        tasks = by_label.get(p["label"], [])
        merged: dict[str, list[str]] = {}
        for e in tasks:
            name, _, letter = e.partition(" ")
            merged.setdefault(name, []).extend([letter.strip("()")] if letter else [])
        cell = ", ".join(name + (f" ({', '.join(letters)})" if letters else "") for name, letters in merged.items())
        key = f"; figure key `{p['figure_key']}`" if p["figure_key"] else ""
        code = p["code"]
        if not code and p["figure_key"] and scripts.get(p["label"]):
            code = ("panels: " if len(scripts[p["label"]]) > 1 else "panel script: ") + ", ".join(f"`{s}`" for s in scripts[p["label"]])
        title = p["title"]
        if title.startswith("Extended caption of "):
            title = re.sub(r"\b(?:fig|tab):[A-Za-z0-9-]+", lambda m: names[m.group(0)]["where"] if m.group(0) in names else m.group(0), title)
        out.append(f"| {p['where']} | `{p['label']}`{key} | {title} | {cell or '-'} | {code or '-'} |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------- checks

def check_rows(rows: list[dict], names: dict[str, dict]) -> tuple[list[str], list[str]]:
    errors, warnings = [], []
    folders = task_folders()
    listed = [r["folder"] for r in rows]
    errors += [f"task folder without a row: {f}" for f in folders if f not in listed]
    errors += [f"row without a task folder: {f}" for f in listed if f not in folders]
    errors += [f"folder listed twice: {f}" for f in sorted({f for f in listed if listed.count(f) > 1})]
    have_results = (ROOT / "results" / "analysis").is_dir()
    for r in rows:
        f = r["folder"]
        if list(r) != COLUMNS or any(v is None for v in r.values()):
            errors.append(f"{f}: row does not have the {len(COLUMNS)} columns")
            continue
        if r["group"] not in GROUPS:
            errors.append(f"{f}: unknown group {r['group']}")
        if r["status"].startswith(RETIRED):
            later = r["status"][len(RETIRED):]
            if later == f or later not in listed:
                errors.append(f"{f}: status names a task that has no row: {later}")
        elif r["status"] not in STATUS:
            errors.append(f"{f}: unknown status {r['status']}")
        errors += [f"{f}: unknown basis {b}" for b in items(r["basis"]) if b not in BASIS]
        if bool(r["paper"]) != (r["basis"] != "-"):
            errors.append(f"{f}: paper and basis must both be set or both be empty")
        for mod in MODULE.findall(r["entry"]):
            parts = mod.split(".")
            if not (TASKS.joinpath(*parts).with_suffix(".py").is_file() or TASKS.joinpath(*parts).is_dir()):
                errors.append(f"{f}: entry names a module that does not exist: analysis.tasks.{mod}")
        for path in FILEPATH.findall(r["entry"]):
            if "<" not in path and not (ROOT / path).exists():
                errors.append(f"{f}: entry names a file that does not exist: {path}")
        for it in items(r["paper"]):
            if re.fullmatch(r"(\S+)(?: ([a-z]))?", it).group(1) not in names:
                errors.append(f"{f}: paper label not in PAPER.tsv: {it}")
        for o in items(r["outputs"]):
            if not o.startswith(("results/analysis/", "data/analysis/", "training/runs/", "notebooks/", "figures/panels/")):
                errors.append(f"{f}: output outside the known roots: {o}")
            m = RESULT.match(o)
            if m and have_results and not (ROOT / "results" / "analysis" / m.group(1)).exists():
                warnings.append(f"{f}: result folder not on disk: {o}")
    return errors, warnings


def check_figure_keys(paper: list[dict]) -> list[str]:
    """Warnings where PAPER.tsv and figures/manifest.tsv (written by the figure build) disagree on the figures."""
    manifest = ROOT / "figures" / "manifest.tsv"
    if not manifest.is_file():
        return []
    first: dict[str, str] = {}
    for r in read_tsv(manifest):
        first.setdefault(r["label"], r["key"])
    out = []
    for p in paper:
        if p["label"].startswith("fig:") and p["figure_key"] != first.get(p["label"], ""):
            out.append(f"PAPER.tsv: {p['label']} has figure key {p['figure_key'] or '(none)'}, figures/manifest.tsv has {first.get(p['label'], '(no figure)')}")
    known = {p["label"] for p in paper}
    out += [f"PAPER.tsv has no row for {label} (figure key {key} in figures/manifest.tsv)" for label, key in first.items() if label not in known]
    return out


def result_owners(rows: list[dict]) -> dict[str, list[str]]:
    owners: dict[str, list[str]] = {}
    for r in rows:
        for o in items(r["outputs"]):
            m = RESULT.match(o)
            if m and r["folder"] not in owners.setdefault(m.group(1), []):
                owners[m.group(1)].append(r["folder"])
    return owners


def refresh_paper(aux: Path) -> None:
    """Bring PAPER.tsv in line with a build of the manuscript (its LaTeX .aux file) and with the figure manifest.

    Printed numbers and the order of the rows follow the build. Key and title of a figure follow
    ``figures/manifest.tsv`` (written by the figure build); a figure label of the build without a row gets one, and a
    figure row whose label is not in the build is dropped. Titles of sections, of tables and of floats without a
    figure, and the column ``code``, are kept as written; rows of sections and tables are never added or dropped
    here (a label that is not in the build is reported).
    """
    order: dict[str, int] = {}
    number: dict[str, str] = {}
    for m in re.finditer(r"\\newlabel\{([^}@]+)\}\{\{([^}]*)\}\{(\d+)\}", aux.read_text(encoding="utf-8", errors="replace")):
        if m.group(1) not in order:
            order[m.group(1)], number[m.group(1)] = len(order), m.group(2)
    manifest = ROOT / "figures" / "manifest.tsv"
    figures: dict[str, dict] = {}
    for r in read_tsv(manifest) if manifest.is_file() else []:
        figures.setdefault(r["label"], r)

    def where(label: str) -> str:
        noun = {"fig": "Figure", "tab": "Table"}.get(label.split(":")[0], "Section")
        return ("Supplementary " if not number[label][:1].isdigit() else "") + f"{noun} {number[label]}"

    paper = read_tsv(PAPER)
    if sum(p["label"] in order for p in paper) < len(paper) / 2:
        raise SystemExit(f"{aux}: fewer than half of the labels of PAPER.tsv are in this file; it is not a build of the current manuscript. Nothing written.")
    kept, changed = [], 0
    for p in paper:
        label = p["label"]
        if label not in order:
            if label.startswith("fig:"):
                print(f"dropped (figure label not in the build): {label} {p['figure_key']}")
                continue
            print(f"label not in the build (row kept as is): {label}")
        else:
            changed += where(label) != p["where"]
            p["where"] = where(label)
        if label.startswith("fig:") and figures:
            fig = figures.get(label)
            if fig is None and p["figure_key"]:
                print(f"no figure carries {label} any more (was {p['figure_key']}): key removed, check its title")
            if fig is not None and (p["figure_key"], p["title"]) != (fig["key"], fig["title"]):
                print(f"figure of {label}: {p['figure_key'] or '(none)'} -> {fig['key']}")
            p["figure_key"], p["title"] = (fig["key"], fig["title"]) if fig is not None else ("", p["title"])
        kept.append(p)
    known = {p["label"] for p in kept}
    placed = {p["where"] for p in kept}
    for label in order:
        if label.startswith("fig:") and label not in known and where(label) not in placed:  # a second label of a float gets no row
            fig = figures.get(label)
            kept.append({"label": label, "where": where(label), "figure_key": fig["key"] if fig else "",
                         "title": fig["title"] if fig else "Extended caption (text only)", "code": ""})
            print(f"added: {label} {fig['key'] if fig else '(no figure)'}")
    # the order of the paper; a row that is not in the build stays behind the row it followed
    rank, last = {}, -1.0
    for i, p in enumerate(kept):
        last = float(order[p["label"]]) if p["label"] in order else last + 1e-6
        rank[i] = last
    kept = [kept[i] for i in sorted(range(len(kept)), key=lambda i: (rank[i], i))]
    with PAPER.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(paper[0]), delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_NONE)
        w.writeheader()
        w.writerows(kept)
    print(f"{PAPER.name}: {len(kept)} rows ({len(paper)} before), {changed} printed locations changed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--check", action="store_true", help="write nothing; exit 1 if a generated view is stale or a row is wrong")
    ap.add_argument("--paper-aux", type=Path, help="main.aux of a manuscript build: first bring PAPER.tsv in line with it (numbers, order, figure keys and titles)")
    a = ap.parse_args()
    if a.paper_aux:
        refresh_paper(a.paper_aux)
    rows, paper = read_tsv(INDEX), read_tsv(PAPER)
    names = paper_names(paper)
    errors, warnings = check_rows(rows, names)
    warnings += check_figure_keys(paper)
    for w in warnings:
        print("warning:", w)
    if errors:
        print("\n".join("error: " + e for e in errors))
        return 1
    owners = result_owners(rows)
    targets: dict[Path, str] = {}
    readme = ROOT / "analysis" / "README.md"
    targets[readme] = replace_block(readme.read_text(encoding="utf-8"), "index", readme_tables(rows, names))
    reproduce = ROOT / "REPRODUCE.md"
    targets[reproduce] = replace_block(reproduce.read_text(encoding="utf-8"), "paper-map", paper_map(rows, paper))
    for r in rows:
        path = TASKS / r["folder"] / "README.md"
        targets[path] = task_readme(r, names, owners, path.read_text(encoding="utf-8") if path.exists() else None)
    stale = [p for p, text in targets.items() if not p.exists() or p.read_text(encoding="utf-8") != text]
    if a.check:
        for p in stale:
            print("stale:", p.relative_to(ROOT))
        print(f"{len(rows)} tasks, {len(paper)} paper locations, {len(targets)} generated views, {len(stale)} stale")
        return 1 if stale else 0
    for p in stale:
        p.write_text(targets[p], encoding="utf-8")
    print(f"{len(rows)} tasks, {len(paper)} paper locations; wrote {len(stale)} of {len(targets)} views")
    return 0


if __name__ == "__main__":
    sys.exit(main())
