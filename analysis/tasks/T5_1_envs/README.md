<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T5_1_envs

Builds the environments of SATURN and CAMEX and fetches the two tools at their pinned commits.

| | |
|---|---|
| Question | Cell-level integration: Does one fit align the cells across species, and how does it compare with other tools? |
| Status | infrastructure |
| Environment | `saturn_env`, `camex_env` |
| Used in the paper | Supplementary Section C.19.2 (`sssec:rev-benchmark`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/T5_1_envs/build_envs.sh

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `results/analysis/T5_1/`: bundle `tool_runs_saturn_camex`

**Notes.** Tool locations and pinned commits: analysis/external_tools/README.md. Environment files: analysis/envs/.
