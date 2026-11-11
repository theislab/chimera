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
| 3. Reserve and upload | One Zenodo record per bundle and one for the code archive are created and their DOIs reserved. The release tag and the DOIs are written into `zenodo/bundles.yaml`; `stage.py` then writes them into the README and the metadata of every record and keeps every tarball whose members did not change, and `stage.py --bundle-table` prints the bundle table of `REPRODUCE.md` with the DOIs. The files of `<staging>/<bundle>/upload/` are uploaded and the form of each record is filled from `<bundle>/zenodo_metadata.json`. | by hand (see `<staging>/README.md`) | maintainer |
| 4. Tag | The tag and the DOI of the code archive are written into `README.md`, `REPRODUCE.md`, `CITATION.cff` (with the release date) and `CHANGELOG.md`; `stage.py --check` reports when `REPRODUCE.md` or `CITATION.cff` does not carry a declared identifier. Then the commit is tagged, the archive of the tagged code is uploaded to its record, and the records are published. | by hand | maintainer |

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
- The identifiers of a release (its tag, the DOI of the code archive, the DOI of each record)
  are declared once, in `zenodo/bundles.yaml`. Declaring them rewrites the generated texts of
  the staged records and leaves the tarballs as they are, so the tarballs that the clean-clone
  test read are the ones that are uploaded.
- The clean-clone test uses the committed state of the repository. Commit, stage, then test.
