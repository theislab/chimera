# Figures of the paper

Every figure of the paper, main and supplementary, is built here: composed from panel files by one layout tool, at
the size it has on the page, collected in one PDF with one figure per page. The paper includes its figures from that
PDF by page number.

| File or folder | What it is |
|---|---|
| `figures.pdf` | the single PDF: one figure per page, in the order of the paper, each page at print size |
| `INDEX.md` | one entry per figure: page, title, what it shows, the panel files it is made of and the script that draws each (generated) |
| `manifest.tsv`, `manifest.json` | page, key, label, title, size, spec, panels with their scripts, smallest text, checks, checksum (generated) |
| `composed/<key>.pdf` | the figures one by one, before they are merged |
| `panels/<family>/<name>.pdf` | the panel files the figures are composed from; `<family>` is the key of the figure the panel was first drawn for (a panel keeps its folder when it is placed in another figure) |
| `specs/*.py` | the figures: one entry per float of the paper (layout, label, title, plain description) |
| `specs/_allow.py` | the figures that do not pass the house style yet, each with its reason |
| `specs/not_in_paper.py` | figures that can be composed and are not in the paper: extra figures and retired ones |
| `order.txt` | the keys in the order of the paper (written from the manuscript) |
| `numbers.tsv` | key and printed number of every figure, written from the manuscript; the index takes "Figure 6", "Supplementary Figure S26" from it |
| `compose.py` | the layout tool: rows, columns and panels, solved exactly, written through pdflatex |
| `registry.py`, `build.py`, `check.py`, `guide.py` | collect the specs; build; check; write the index |
| `style.py` | the house style as code: type sizes, colors and markers of every recurring entity, names, `save_panel` |
| `redraw/` | scripts that draw panels and belong to no analysis task (for example the overview figure) |
| `producers.tsv` | the script that draws each panel file, with its command: figure -> panel -> script |
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
    Col(Row(X("main_validation/worked_genes", {"a": 88, "b": 252}), X("main_validation/celltype_drivers_enzymes", "c"),
            same_scale=True),
        Row(X("main_validation/readouts_vs_references", "d"), X("main_validation/named_findings", "e"), same_scale=True)),
    label="fig:validation",
    title="Two worked genes, the cell types behind the count evidence, and the readouts against independent data",
    shows="..."),
```

- The key says what the figure is about: `main_<topic>` or `supp_<topic>`. Every main figure is one float. A
  supplementary figure that needs a second page has a continuation float `<key>_2`, `<key>_3` with the same `label`
  and `part=2`, `part=3`. Keys carry no figure number, because numbers move.
- `Row` gives its children one height and fills the width; `Col` stacks; `same_scale=True` keeps the panels of a row at
  one scale; `Row(..., width=330)` is a narrower, centered row.
- `P` crops a panel to its ink; `X` keeps the page of a panel that was saved at its exact size, so axes line up; `NB`
  takes a panel from `notebooks/figures/`.
- A panel file that holds two plots takes two letters: `X("family/name", {"a": 88, "b": 252})`, each with the
  position of its letter in bp from the left edge of the page.
- `notebooks/figures/` holds the panels as the notebooks rendered them (the first drawing of most analyses). The figures of
  the paper take from it only the few files that a spec names with `NB(...)`; every other panel is drawn at print size
  by a script of `producers.tsv`.
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
   `CHIMERA_FIGURES_OUT=<folder>`: panels and build products go there and the repository is not touched
   (`CHIMERA_FIGURES_ORDER=<file>` tries another order of the floats in the same way). Which script draws a panel,
   and with which command, is in `producers.tsv` (and under each figure in `INDEX.md`); a new panel folder gets a row
   there. When a panel shows something else than before, the `shows=` text of its figure in `specs/` and the caption
   in the paper change with it.
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
| panel scripts | every panel of a figure has a row in `producers.tsv` whose script exists (named by the build; an error with `--strict`) |

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
\centerline{\includegraphics[page=6]{figures/figures.pdf}}% figure: main_validation
```

and the manuscript's `figures/sync_figures.py` writes the page numbers from the manifest and checks that paper and
figure file are in step.

## Retired figures

A figure or a panel is retired when a later analysis of the same question on the same data replaced it in the paper.
Its entry moves from its spec module to the dictionary `RETIRED` of `specs/not_in_paper.py`, with a comment that says
what it showed, what replaced it (figure key and analysis task) and when. Nothing is deleted: the panel files stay in
`panels/`, the script that draws them stays in `producers.tsv`, and the figure can still be composed:

```bash
PYTHONPATH=$PWD python -m figures.compose supp_integration_seven_methods    # writes composed/<key>.pdf
```

| Key in `RETIRED` | What it showed | Replaced by |
|---|---|---|
| `supp_alpha_decorrelation` | Δα against detection rate, mean count and count variance, liver Human+Monkey | `supp_covariate_audit`: the same statistics on seven cohorts (task `T2_4_covariates`) |
| `supp_integration_seven_methods` | composite integration score of seven methods on seven cohorts | the benchmark of ten methods: `main_integration`, `supp_integration_ten_methods`, `supp_integration_eleven_cohorts` (tasks `T5_5_benchmark`, `B2_benchmark`) |
| `supp_integration_metrics` | six raw integration metrics of seven methods | `supp_integration_raw_metrics`: the same metrics for ten methods (task `T5_5_benchmark`) |
| `retired_tissue_programs_clamped_prior` | program means of Δα, liver against testis, against the clamped prior (two former panels of `supp_alpha_robustness`) | panel d of `supp_delta_alpha_sources`: the comparison against the raw prior (task `T2_3_tissue`) |
| `retired_tissue_sign_classes_clamped_prior` | genes shared by liver and testis by sign class of Δα, clamped prior (a former panel of `supp_residue_audit_2`) | panel d of `supp_delta_alpha_sources` (task `T2_3_tissue`) |

A retired figure is not in `figures.pdf`, `order.txt` or `INDEX.md`, and the paper does not cite it.
