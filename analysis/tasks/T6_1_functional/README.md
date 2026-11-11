<!-- Generated from analysis/tasks/INDEX.tsv by analysis/make_index.py. Edit the table, not this file. -->
# T6_1_functional

Downloads the sources of the functional layer (gene sets, protein interaction networks, transcription-factor binding) and reports identifier coverage. Counts only.

| | |
|---|---|
| Question | Data: Which objects, ortholog tables and independent references do the analyses read? |
| Status | supporting (builds inputs, launches runs, or a check reported as a method detail) |
| Environment | `chimera_env` |
| Used in the paper | Supplementary Section C.21.10 (`sssec:rev-functional`) |
| How we know | described in the Extended Methods |

**Run**, from the repository root, in the order given (set `PYTHONPATH=$PWD` for commands that name a file; `-m` commands need nothing):

    bash analysis/tasks/T6_1_functional/fetch_functional.sh $PWD
    python -m analysis.tasks.T6_1_functional.fetch_enrichr_cache data/analysis/functional
    python -m analysis.tasks.T6_1_functional.coverage

**Writes** (with the deposit bundle that holds it, see `REPRODUCE.md`)

- `data/analysis/functional/`: third-party downloads, linked; their manifests (URL, date, sha256) are in bundle `data_other_cohorts`
- `results/analysis/T6_1/`: bundle `analysis_results`

**Reads** (as named in the code)

- `data/analysis/ensembl110/`
- `data/analysis/gold/`
- `data/processed/`

**Notes.** xlsx_min.py is a standard-library reader used by coverage.py. fetch_enrichr_cache.py writes the two Enrichr libraries (Hallmark, Reactome) into the gseapy cache in the home folder (~/.cache/gseapy/), where gseapy itself would put them, and keeps the server responses in data/analysis/functional/; coverage.py and three other tasks (T4_1_gold, T4_3_scoreboard, H1_harmonise) read the libraries from that cache.
