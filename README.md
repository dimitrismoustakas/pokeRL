# pokeRL

Minimal local setup and run notes.

## Environment

This repo uses `uv` and Python `3.13.7`.

```powershell
uv venv --python 3.13.7
uv sync --dev
```

The project package is installed editable into `.venv`.

The no-server training path targets the local BattleStream branch of
`poke-env`. Until that support is available from the published package,
install the sibling checkout editable into the same venv:

```powershell
uv pip install -e ..\poke-env
```

## Local Pokemon Showdown checkout

Smoke tests and training use poke-env's local BattleStream backend. Put a
Pokemon Showdown checkout at `%LOCALAPPDATA%\pokeRL\pokemon-showdown`, or set:

```powershell
$env:POKERL_SHOWDOWN_DIR = "C:\path\to\pokemon-showdown"
```

## Smoke battle

```powershell
& .\.venv\Scripts\python.exe .\scripts\smoke_battle.py --policy max-power --opponent random --max-turns 0 --json
```

## Small training run

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --device cpu --num-envs 2 --rollout-steps 8 --total-timesteps 32 --num-epochs 1 --num-minibatches 1 --log-interval 1
```

Training logs include rollout time, PPO update time, completed episodes,
average completed battle duration/turns, preview PPO KL and clipping, league
payoff coverage, resolution/CANT calibration, and public-field transition
MSE/support. Those consequence labels come only from exact public protocol
transitions; they do not change the sparse terminal win/loss reward.

`--resume` accepts only versioned trainer checkpoints produced by the exact same model class, configuration, categorical vocabulary, and tensor schema. Trainer checkpoints and their uniquely paired league-state files are atomically replaced and cross-checked; together they restore Adam, global step, the exact completed-update boundary, Python/NumPy/PyTorch RNG, league sampling, and team-pool sampling. Lightweight league snapshots are evaluation/opponent artifacts and cannot resume training.

To train on a fixed Regulation F pool with team preview (bring 4 of 6):

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --battle-format gen9vgc2026regf --team-pool-format gen9vgc2026regf --num-envs 8 --opponent-groups 2
```

To evaluate whether a checkpoint is improving against a frozen past self,
sample both trained policies on a pre-planned schedule and mirror every team
pair across the two player roles:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_anchor.py --checkpoint checkpoints\current.pt --anchor checkpoints\anchor.pt --games 64 --action-mode sample --summary-path logs\anchor_eval.json
```

`--games` is the number per role, so this command runs 128 battles. Use
`--action-mode argmax` only as a separate deterministic diagnostic; the same
mode is applied to both checkpoints. Results separate wins/losses/draws, include
95% intervals, and retain per-game teams and preview choices.

For an absolute fixed-bot diagnostic with action, Tera, and team-preview
telemetry:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_fixed_bot.py --checkpoint checkpoints\current.pt --opponent simple-heuristics --games 76 --action-mode sample --summary-path logs\fixed_bot_eval.json
```

To measure whether preview probabilities actually change with the opposing
roster, exactly enumerate all 180 legal bring/lead choices:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_preview_policy.py --checkpoint checkpoints\current.pt --summary-path logs\preview_eval.json
```

The default Regulation F manifest currently contains only teams used for
training. Treat those results as in-distribution until a separate evaluation
population is supplied with `--team-pool-manifest`.

To tune how many frozen opponents are shared across the env batch:

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --device cpu --num-envs 16 --opponent-groups 2
```

The two opponent groups are resampled every update by default, while a new
frozen policy is added to the permanent archive every 10 updates. Discounted
current-policy outcomes drive hard-opponent PFSP sampling, with a uniform
coverage component, and group assignments are distinct whenever the archive
contains enough members. `--league-size` bounds resident frozen models per
inference service rather than archive history. Use `--league-refresh-interval`,
`--league-update-interval`, and the `--league-pfsp-*` options to tune this
schedule. `--seed` records separate league and team streams; paired resume state
also restores the already-sampled opponent assignments so it does not consume
an extra PFSP draw. Asynchronous battles still do not guarantee bitwise-identical
runs.

On CUDA, frozen battle-policy inference uses a bounded CUDA-graph cache by
default. It preserves sampled actions and CUDA RNG state while removing repeated
launch overhead; unsupported devices or capture failures fall back to eager
inference and are reported in league cache diagnostics. Use
`--no-league-cuda-graphs` only for debugging or an eager A/B comparison.

For a short local BattleStream scaling benchmark over two rollout->PPO rounds:

```powershell
& .\.venv\Scripts\python.exe .\scripts\benchmark_parallel_training.py --battle-format gen9vgc2026regf --team-pool-format gen9vgc2026regf --env-counts 4,8,16 --opponent-groups 1,2 --rollout-steps 16 --num-updates 2
```

The benchmark writes per-run logs and a JSON summary under `benchmarks/<timestamp>/`.
Use `--worker-counts`, `--battles-per-worker`, `--runtime-loop-count`,
`--battle-loop-count`, and `--league-coalesce-ms` to test the local
BattleStream knobs used by longer runs. Training also accepts
`--reset-concurrency` to bound simultaneous environment resets when startup
load needs to be limited.

## Building a VGC team pool

The repository includes a helper to fetch public VGC teams, validate them
against the local Pokemon Showdown checkout, and optionally generate extra
legal remix teams from the curated pool.

Then build a Regulation I pool:

```powershell
& .\.venv\Scripts\python.exe .\scripts\build_vgc_team_pool.py --format gen9vgc2026regi --target-team-count 150
```

Or build a Regulation F pool:

```powershell
& .\.venv\Scripts\python.exe .\scripts\build_vgc_team_pool.py --format gen9vgc2026regf --target-team-count 150
```

Output is written under `data/team_pools/<format>/` with:

- `curated/`: validated online source teams only
- `generated/`: additional legal remixes only
- `pool/`: the full usable pool, with curated teams first and generated teams appended after them
- `manifest.json`: summary stats, source links, and file inventory

If `--target-team-count` is set, the script keeps every curated team and only
generates enough extra remixes to reach that total when possible.
