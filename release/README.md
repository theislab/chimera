# Releasing Chimera

A release of Chimera is two things that must fit together:

- the **code** at a git tag of this repository, and
- the **data deposit** at Zenodo: processed objects, trained models, tables and analysis
  outputs. They are too large for git and live in the storage folders that git ignores
  (`data/`, `results/`, `training/runs/`, `notebooks/cache/`).

A reader should be able to take the tag and the deposit and get the numbers of the paper
without anything else. This folder holds the tools that prepare a release and test exactly
that. For *using* a release, see [`REPRODUCE.md`](../REPRODUCE.md).

## The four steps

| Step | What happens | Command | Who |
|---|---|---|---|
| 1. Stage | The deposit is built from the current repository and storage into a staging folder outside the repository: one folder per bundle with a tarball, a manifest, checksums and the Zenodo metadata. | `python release/zenodo/stage.py` | script |
| 2. Test from a clean clone | A fresh `git clone`, an environment built from the lock file and the staged bundles are put together, and the checks run there with the working tree hidden. | `bash release/clean_clone_test.sh all` | script |
| 3. Upload | One Zenodo record per bundle is created from `<staging>/<bundle>/upload/` and `<bundle>/zenodo_metadata.json`; the DOIs are reserved. | by hand (see `<staging>/README.md`) | maintainer |
| 4. Tag | The reserved DOIs and the tag are written where the placeholders `[DOI]` and `[TAG]` stand in the documents (`REPRODUCE.md`, `README.md`), and the release date and DOI into `CITATION.cff` and `CHANGELOG.md`; `git grep -n -F -e '[DOI]' -e '[TAG]' -- '*.md'` lists the places. Then the commit is tagged and the records are published. | by hand | maintainer |

No script here uploads, tags or pushes anything. Steps 1 and 2 can be repeated at any time and
change nothing in the repository.

```bash
python release/zenodo/stage.py --check          # is the staged deposit current? exit 1 when not
python release/zenodo/stage.py                  # rebuild the bundles that are not
python release/zenodo/stage.py --check --deep   # before an upload: recompute every checksum
bash release/clean_clone_test.sh --fresh all    # exit 0: the release reproduces from clone + deposit
```

## What is where

| Path | Role |
|---|---|
| [`zenodo/`](zenodo/README.md) | Step 1. `bundles.yaml` declares the deposit (which files go into which bundle, what is left out and why); `stage.py` builds and checks the staging area; `LICENSES.md` lists third-party sources. |
| [`clean_clone_test.sh`](clean_clone_test.sh), [`clean_clone/`](clean_clone/README.md) | Step 2. The driver and its two check modules. |

Related, outside this folder: `analysis/envs/build_env_from_lock.sh` (the environment of the
test), `analysis/reproduce_smoke.sh` (the checks a user also runs), `analysis/external_tools/`
(the third-party tools some analyses run; they are not part of the deposit).

## Keeping a release consistent

- The bundle names and contents in `REPRODUCE.md` come from the declaration:
  `python release/zenodo/stage.py --bundle-table` prints the table, and `--check` fails when
  `REPRODUCE.md` does not name a bundle.
- After any change to results, data, runs or caches, run `stage.py --check`. It also reports a
  storage folder that is in no bundle and has no stated reason for being left out.
- The clean-clone test uses the committed state of the repository. Commit, stage, then test.
