# Clean-clone test

**Question it answers:** do the results reproduce from what is released, and from nothing else?

The test takes only what a reader gets: a fresh `git clone` of this repository, an environment
built from the lock file, and the bundles staged for Zenodo. It unpacks the bundles into the
clone and runs the checks there. While a check runs, the working tree of the authors is hidden
from it, so a check cannot pass by reading a file that is not in the release.

```bash
python release/zenodo/stage.py --check                     # the staged deposit must be current
bash release/clean_clone_test.sh --fresh --threads 4 all   # clone, env, unpack, smoke, extra
bash release/clean_clone_test.sh --help
```

Exit code 0 means every requested step passed. Logs are written to `<work>/logs/`.

## Steps

| Step | What it does | Passes when |
|---|---|---|
| `clone` | `git clone` of the code into `<work>/chimera` | the clone exists |
| `env` | builds the environment from `analysis/envs/chimera_env.linux-64.lock.txt` of the clone | every pinned package installs and `pip check` is clean |
| `unpack` | verifies `SHA256SUMS` of every staged bundle and unpacks it into the clone | all checksums match and `git status` of the clone is still clean |
| `smoke` | `analysis/reproduce_smoke.sh` in the clone, with one checkpoint reload and one notebook | the test suite passes; the per-gene tables recomputed from the archived model states equal the deposited tables; the checkpoint returns its archived values; the notebook runs |
| `extra` | [`clean_clone/extra_checks.py`](extra_checks.py) | every archived run finds its training object in the clone (objects that the task code rebuilds are counted apart); one model of each bundle family reloads on CPU and returns its archived per-gene values |
| `retrain` (optional) | trains one recipe again in the clone, then [`clean_clone/compare_retrain.py`](compare_retrain.py) | no pass mark: the comparison is printed |

`all` runs the first five. `retrain` takes about an hour on 16 threads.

## How the working tree is hidden

Each test command runs inside `unshare -Urm`: a private user and mount namespace, which needs
no administrator rights. Inside it an empty folder is bind-mounted over the working tree, so
every path into the working tree shows an empty folder; the command starts only after that
has been checked. Nothing changes outside that process. On a system that forbids user
namespaces the script stops and asks for `--no-hide`; it never runs unhidden silently.

## Reading the retrain comparison

Training on CPU is not deterministic at a fixed seed, so a retrained run is not expected to
repeat an archived run value by value. `compare_retrain.py` prints the training state of the
retrain (look at it first), its rank correlation with every archived run of the cohort and
with the archived seed consensus, and the same correlation among the archived runs
themselves. A retrain reproduces the result when it falls inside that spread.

## Where things are

| Option | Environment variable | Default |
|---|---|---|
| `--staging` | `CHIMERA_ZENODO_STAGING` | `../_zenodo_staging` beside the checkout |
| `--work` | `CHIMERA_CLEAN_CLONE` | `../_clean_clone` beside the checkout |
| `--env` | | `chimera_clean_env` |
| `--source`, `--ref` | | this repository, current branch; for a published release the repository URL and the tag |

The work folder is disposable (it holds a second copy of the unpacked deposit). Keep the logs
of a test that is cited, then delete it.
