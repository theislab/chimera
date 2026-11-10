# Third-party tools

Some analyses run other methods on the same data: SATURN and CAMEX (integration and gene-level
comparison), GeneSpectra, CACIMAR and SAMap (gene-level benchmark). Their code is not part of
this repository. It is cloned from the authors' repositories at pinned commits into a folder
outside the repository, and each tool runs in its own environment.

| File | Role |
|---|---|
| [`tools.tsv`](tools.tsv) | The manifest: name, upstream URL, pinned commit, license, environment, how the clone is used, which scripts use it. **The only place a commit is pinned.** |
| [`fetch.py`](fetch.py) | The producer of the clones: clones each tool, checks out its pinned commit, verifies it, and writes a `README.md` into the folder of the clones. `--check` verifies without changing anything. |
| [`location.py`](location.py) | The one definition of where the clones are and of the pinned commits. Every consumer asks it. |

## Set up

```bash
python3 analysis/external_tools/fetch.py                    # clone and pin all tools
bash analysis/tasks/T5_1_envs/build_envs.sh                 # environments saturn_env, camex_env
bash analysis/tasks/B2_benchmark/tools/build_envs.sh        # environments genespectra_env, cacimar_env, samap_env
python3 analysis/external_tools/fetch.py --check            # exit 0: every clone is as pinned
```

The clones go to `external_tools/` beside the checkout of this repository. Set
`CHIMERA_EXTERNAL_TOOLS=/some/folder` to keep them elsewhere; `fetch.py` and every consumer
follow it. The environment files are in [`analysis/envs/`](../envs/) (`<env>.yml`, with the
installed versions in `<env>.pip_freeze.txt`).

## Who uses which clone

| Tool | Environment | Consumers |
|---|---|---|
| SATURN | `saturn_env` | `analysis/tasks/T5_2_adapters/run_saturn.py`, `analysis/tasks/B2_benchmark/tools/run_saturn_b2.py` (both through `T5_2_adapters/cohorts.py`) |
| CAMEX | `camex_env` | `analysis/tasks/T5_2_adapters/run_camex.py`, `analysis/tasks/B2_benchmark/tools/run_camex_b2.py` (both through `T5_2_adapters/cohorts.py`) |
| GeneSpectra | `genespectra_env` | `analysis/tasks/B2_benchmark/tools/run_genespectra.py` (through `b2t_common.py`) |
| CACIMAR | `cacimar_env` | `analysis/tasks/B2_benchmark/tools/build_envs.sh` installs it into the environment; `run_cacimar.py` and `cacimar_steps.R` use the installed package and record the commit of the clone |
| SAMap | `samap_env` | `analysis/tasks/B2_benchmark/tools/build_envs.sh` installs it into the environment; `blast_samap.sbatch` runs its `map_genes.sh` from the clone; `run_samap.py` records the commit of the clone |
| GeneSpectra_reproducibility | none | Read only: the authors' notebooks that the `spec_cosine` column of the GeneSpectra adapter follows. No script reads it. |

Every adapter writes the commit of the clone it ran and whether the clone was modified into
the record it writes beside its output (`tool_state`), so a result can be traced to the exact
tool code.

## Pin another commit or add a tool

Edit the row in `tools.tsv` (or add one), run `fetch.py`, rebuild the environment if the tool
is installed into it, and rerun the adapters that use it.
