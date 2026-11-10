# Figures of the paper

Every figure of the paper, main and supplementary, is built here: composed from panel files by one layout tool, at
the size it has on the page, collected in one PDF with one figure per page. The paper includes its figures from that
PDF by page number.

| File or folder | What it is |
|---|---|
| `figures.pdf` | the single PDF: one figure per page, in the order of the paper, each page at print size |
| `INDEX.md` | one entry per figure: page, title, what it shows, the panel files it is made of (generated) |
| `manifest.tsv`, `manifest.json` | page, key, label, title, size, spec, panels, smallest text, checks, checksum (generated) |
| `composed/<key>.pdf` | the figures one by one, before they are merged |
| `panels/<family>/<name>.pdf` | the panel files the figures are composed from; `<family>` is the key of the figure that owns the panel |
| `specs/*.py` | the figures: one entry per float of the paper (layout, label, title, plain description) |
| `specs/_allow.py` | the figures that do not pass the house style yet, each with its reason |
| `order.txt` | the keys in the order of the paper (written from the manuscript) |
| `compose.py` | the layout tool: rows, columns and panels, solved exactly, written through pdflatex |
| `registry.py`, `build.py`, `check.py`, `guide.py` | collect the specs; build; check; write the index |
| `style.py` | the house style as code: type sizes, colors and markers of every recurring entity, names, `save_panel` |
| `redraw/` | scripts that draw panels and belong to no analysis task (for example the overview figure) |
| `sources/` | pages that are placed whole because they are not composed from panel files yet |
| `panel_names.tsv` | former file names of panels and their place in `panels/` (used while scripts still save under the old names) |
| `environment.yml` | the small environment of the build |
| `build.sh` | the one command |

## The one command

```bash
bash figures/build.sh              # compose what changed, merge, crop, manifest, index, checks
bash figures/build.sh --check      # write nothing; exit 1 if anything is not current
```

The build needs the environment `chimera_figures_env` (`mamba env create -f figures/environment.yml`; exact versions in
`chimera_figures_env.linux-64.lock.txt`) and `pdflatex`. It reads panel files only: no analysis code runs, no result
table is needed, nothing is recomputed. Output is the same file byte for byte when nothing changed, so `git status`
shows which figures a change touched.

What the build does:

1. **Compose.** Each figure of `specs/` is laid out from its panels and written to `composed/<key>.pdf`. Panels stay
   vector, their fonts stay embedded, and the panel letters are drawn by the tool, bold, under each panel.
2. **Check** each figure against the house style (below).
3. **Merge** the composed figures in the order of `order.txt`.
4. **Crop** the merged file with `pdf-crop-margins -v -p 0 -t 250 -c gr -x 300 -y 300 -a -0.5`. The four options after
   `-p 0` are there because the defaults of the tool trim light strokes (panel frames, light gray labels) at the page
   edges; with them no ink is lost. Only the page boxes change.
5. **Verify the crop**: no page loses ink, no page keeps more than 1.5 bp of white on a side.
6. **Write** `figures.pdf`, the manifest and `INDEX.md`.

## A figure

```python
"main_validation": Figure(
    Col(Row(P("main_validation/prespecified_tests", "a"), P("main_validation/gene_sets", "b"), same_scale=True),
        X("main_validation/head_to_head", "c"), X("main_validation/incremental", "d")),
    label="fig:validation",
    title="Pre-specified tests against independent data and comparison with direct baselines",
    shows="..."),
```

- The key says what the figure is about: `main_<topic>` or `supp_<topic>`. A continuation float is `<key>_2`, `<key>_3`
  with the same `label` and `part=2`, `part=3`. Keys carry no figure number, because numbers move.
- `Row` gives its children one height and fills the width; `Col` stacks; `same_scale=True` keeps the panels of a row at
  one scale; `Row(..., width=330)` is a narrower, centered row.
- `P` crops a panel to its ink; `X` keeps the page of a panel that was saved at its exact size, so axes line up; `NB`
  takes a panel from `notebooks/figures/`; `SRC` takes a file from `sources/`.
- Width is 486 bp (the text width) unless `width=` says otherwise; a figure on a landscape page is 648 bp wide.

```bash
PYTHONPATH=$PWD python -m figures.compose --list             # all keys
PYTHONPATH=$PWD python -m figures.compose main_validation    # compose one figure
```

## Changing a figure

1. **Layout only**: edit its entry in `specs/`, run `bash figures/build.sh`.
2. **A panel**: run the script that draws it (in `chimera_env`), then `bash figures/build.sh`. Panel scripts live with
   the analysis they belong to (`analysis/tasks/<task>/`) or in `figures/redraw/`; they read cached result tables,
   call `figures.style.apply()` and save with `figures.style.save_panel(fig, "family/name")`. For a dry run set
   `CHIMERA_FIGURES_OUT=<folder>`: panels and build products go there and the repository is not touched.
3. **A new figure**: add an entry to the spec module of its part of the paper. Until its float exists in the manuscript
   the figure is *pending*: it is composed, the build names it, and it is not in `figures.pdf`. Then write the order
   and build. `order.txt` is normally written from the manuscript (`sync_figures.py --write-order`), so a float that moves
   in the paper changes one generated file and no spec.

## House style and checks

Panels are drawn at print size, so a font size in a script is the printed size. `style.py` holds the values; the build
checks the result on every figure:

| Check | Rule |
|---|---|
| `text` | no text under 5.0 pt font size at print size; single characters (subscripts, stars) not under 3.5 pt |
| `fonts`, `font_family` | all fonts embedded, none of Type 3; DejaVu Sans only (letters in Helvetica bold) |
| `paths` | no path of a working machine anywhere in a file |
| `page`, `width` | one page of the computed size; 486 bp wide, or the narrower or landscape width of its spec |
| `scale` | no panel placed below 0.9 of its drawn size; no figure shrunk as a whole |
| `letters` | letters run a, b, c ... over a figure and its continuation floats, in reading order |
| `words`, `spelling` | no internal name (cohort keys, method codes, column names) and no British spelling in the artwork |

A figure that does not pass yet is listed in `specs/_allow.py` with its reason; the build prints the list every time,
`--strict` fails on it, and an entry whose check passes is an error, so the list can only shrink.

Colors and markers are fixed per entity (`style.READOUT`, `style.TISSUE`, `style.SPECIES`, `style.ABOVE`, `style.BELOW`):
the same readout, tissue or species has the same color and marker in every figure.

## Comparing two builds

```bash
PYTHONPATH=$PWD python -m figures.check --against reference.json --tsv compare.tsv --diff diff/
```

renders every page of a reference and of the current build so that one pixel is 1/150 inch at print size and reports
each page as identical, equivalent or different (definitions at the top of `check.py`). Use it before and after any
change that should not alter a figure.

## The paper

`figures.pdf` and the manifest are copied to the `figures/` folder of the manuscript (`bash figures/build.sh --export
<manuscript>/figures`). There each float includes its page and names its figure on the same line,

```latex
\centerline{\includegraphics[page=7]{figures/figures.pdf}}% figure: main_validation_2
```

and the manuscript's `figures/sync_figures.py` writes the page numbers from the manifest and checks that paper and
figure file are in step.
