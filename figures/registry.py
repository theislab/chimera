"""The figures of the paper, collected from ``figures/specs/*.py`` and put in paper order.

Every spec module defines ``FIGS = {key: Figure(...)}``. A module with ``IN_PAPER = False`` holds figures that can be
composed but are not part of the paper and not of ``figures.pdf``. ``order.txt`` lists the keys of the paper's figures in
the order of their floats; it is written from the manuscript, so a moved float changes one generated file and no spec.
"""

from __future__ import annotations

import hashlib
import importlib
import pkgutil
from pathlib import Path

from figures import compose as C

ORDER = C.ROOT / "order.txt"


def read_order() -> list:
    return [ln.strip() for ln in ORDER.read_text().splitlines() if ln.strip() and not ln.startswith("#")]


def load(in_paper_only: bool = True) -> dict:
    """key -> Figure. Paper figures first, in the order of ``order.txt``; then, if asked, the others."""
    import figures.specs as specs

    paper, other = {}, {}
    for info in sorted(pkgutil.iter_modules(specs.__path__), key=lambda i: i.name):
        if info.name.startswith("_"):
            continue
        mod = importlib.import_module(f"figures.specs.{info.name}")
        target = paper if getattr(mod, "IN_PAPER", True) else other
        for key, fig in mod.FIGS.items():
            if key in paper or key in other:
                raise ValueError(f"figure key {key!r} is defined twice (second time in specs/{info.name}.py)")
            fig.module = info.name
            target[key] = fig
    order = read_order()
    problems = [f"{k}: in order.txt but in no spec" for k in order if k not in paper]
    problems += [f"{k}: in specs/{paper[k].module}.py but not in order.txt" for k in paper if k not in order]
    problems += [f"{k}: listed twice in order.txt" for k in set(order) if order.count(k) > 1]
    # parts of one label: 1, 2, 3, ... in order, keys <key>, <key>_2, <key>_3
    seen = {}
    for k in order:
        if k not in paper:
            continue
        f = paper[k]
        n = seen.get(f.label, 0) + 1
        seen[f.label] = n
        if f.part != n:
            problems.append(f"{k}: part {f.part} of {f.label}, expected part {n} at this place of order.txt")
        if f.part > 1 and not k.endswith(f"_{f.part}"):
            problems.append(f"{k}: a continuation float is named <key>_{f.part}")
    if problems:
        raise ValueError("figure registry:\n  " + "\n  ".join(problems))
    figs = {k: paper[k] for k in order}
    if not in_paper_only:
        figs.update(other)
    return figs


def signature(fig: C.Figure) -> dict:
    """What a figure is made of: the layout and the checksum of every panel file. A build is current when the stored
    signature equals this one."""
    def node(n) -> dict:
        if isinstance(n, C.Panel):
            d = dict(panel=n.name(), page=n.page, region=n.region, pad=list(n.pad), ink=n.ink, label=n.label,
                     sha256=hashlib.sha256(n.path().read_bytes()).hexdigest())
            if isinstance(n, C.LabelledPage):
                d.update(at=n.at, bottom=n.bottom, pad=None)  # its pad is computed when the page is loaded
            return d
        d = dict(type=type(n).__name__, gap=n.gap, children=[node(c) for c in n.children])
        if isinstance(n, C.Row):
            d.update(same_scale=n.same_scale, width=n.width)
        return d

    return dict(width=fig.width, shrink=fig.shrink, tree=node(fig.tree))


def signature_hash(fig: C.Figure) -> str:
    import json

    return hashlib.sha256(json.dumps(signature(fig), sort_keys=True, default=str).encode()).hexdigest()
