# Zenodo deposit: how it is built

The data of the paper (processed objects, trained models, tables, analysis outputs) are not
tracked by git. They are deposited at Zenodo as a set of bundles, one record each.
[`REPRODUCE.md`](../../REPRODUCE.md) tells a user how to download and unpack them. This folder
holds what builds them.

| File | Role |
|---|---|
| [`bundles.yaml`](bundles.yaml) | The declaration: which files go into which bundle, what is left out and why, which columns are removed. **Edit this file to change the deposit.** |
| [`stage.py`](stage.py) | Builds the staging area from the declaration and the current repository and storage. Uploads nothing. |
| [`LICENSES.md`](LICENSES.md) | Third-party sources, their licenses, and whether each is redistributed or linked. Copied into every record. |
| [`texts/`](texts/) | Text files that are packed into a bundle (the column description of the gene tables). |
| `staging/` | The modules behind `stage.py`: `declaration` (reads `bundles.yaml`), `selection` (resolves members), `scrub` (dropped columns, text rules), `packing` (tarball, manifest, checksums), `records` (metadata and READMEs), `status` (`--check`). |

## Use

Inside the project environment:

```bash
python release/zenodo/stage.py --list         # the bundles
python release/zenodo/stage.py --dry-run      # what would be packed, with sizes; writes nothing
python release/zenodo/stage.py                # build every bundle that is not up to date
python release/zenodo/stage.py --check        # is the staging area current? exit 1 when not
python release/zenodo/stage.py --help
```

The staging area is written to `--out`, else `$CHIMERA_ZENODO_STAGING`, else `../_zenodo_staging`
beside the checkout (always outside the repository). It needs about as much disk as the deposit
(25 GB) and `pigz` for parallel compression (falls back to single-threaded gzip without it).

## What a staged bundle contains

```
<staging>/<bundle>/
    upload/                         exactly the files of the Zenodo record
        chimera_<bundle>.tar.gz       members are paths relative to the repository root
        MANIFEST.tsv                  every member: path, bytes, sha256, origin
        README.md, LICENSES.md        what the record is; third-party sources
        SHA256SUMS                    sha256 of the files above
    zenodo_metadata.json            title, description, creators, related identifiers ([TAG], [DOI] placeholders)
    skipped.tsv                     what was left out, with the reason
    scrubbed.tsv                    members whose content was altered
    review_hits.tsv                 matches of review patterns (not altered)
    members.tsv, staging_info.json  what was packed from which source state (read by --check)
<staging>/README.md, LICENSES.md, size_table.tsv
```

## Keeping the deposit up to date

`stage.py` is incremental. For every bundle it computes a fingerprint from the member list
(path, size and modification time of each source file) and from the rules that alter content.
A bundle whose fingerprint is unchanged is kept; the others are rebuilt. A bundle is written
to a temporary folder and moved into place only when complete.

Run `--check` after any change to results, data, runs or the declaration. It reports, per
bundle, new, removed and changed members, and it lists storage folders that are in no bundle
and have no stated reason for being left out (`coverage_roots` in the declaration). A new
task folder under `results/analysis/` is picked up automatically; a new folder under
`data/analysis/` or a new family of runs under `training/runs/` needs one line in
`bundles.yaml`: under `members` to deposit it, or under `not_included` / `not_deposited` with
the reason.

The fingerprint trusts sizes and modification times. `--check --deep` also recomputes the
sha256 of the upload files and of every unaltered source file, so it also finds a file whose
content changed while its size and time stayed the same. Run it once before an upload.

## What is altered on the way in

Everything is packed byte for byte except:

- **Dropped columns** (`dropped_columns` in the declaration): per-gene EVEE values, whose source
  states no license, are removed from every csv, tsv and parquet table. Other cells keep their
  exact text. The altered members are listed in `scrubbed.tsv` and marked `derived:` in
  `MANIFEST.tsv`.
- **Derived members**: the cell annotation table (`obs`) of an AnnData file whose expression
  matrix is public and is not redistributed (`obs_table_of` selectors).
- **Text rules** (`text_rules`): text files up to 5 MB are checked against `replace`, `review`
  and `blocked` patterns. The declaration ships none; a maintainer can add private ones with
  `--private-rules <file>` or `$CHIMERA_ZENODO_PRIVATE_RULES`, a YAML file with the same three
  keys.

One member is not produced by code of this repository: the released copy of the pre-specified
analysis plan, `results/analysis/_release/analysis_plan_prespecified.md` (bundle
`analysis_plan`). The authors export it from their working copy of the plan. When the file is
absent, `stage.py` reports that bundle as not built and builds the others.

## Changing the deposit

| To | Do |
|---|---|
| deposit a new folder | add a selector to `members` of a bundle in `bundles.yaml` |
| leave something out | add it to `not_included` of its bundle, with the reason |
| add a bundle | add an entry under `bundles` (title, description, members) and name it in `REPRODUCE.md` |
| rename or remove a bundle | edit `bundles.yaml`, run `stage.py`, then `stage.py --prune` to delete the old folder |
| remove a column everywhere | add it to `dropped_columns.columns` |
| update the bundle table of `REPRODUCE.md` | `python release/zenodo/stage.py --bundle-table` |
