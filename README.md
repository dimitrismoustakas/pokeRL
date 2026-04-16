# pokeRL

Minimal local setup and run notes.

## Environment

This repo uses `uv` and Python `3.13.7`.

```powershell
uv venv --python 3.13.7
uv sync --dev
```

The project package is installed editable into `.venv`.

## Local Showdown server

Start the local Pokemon Showdown server before smoke tests or training:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_showdown_server.ps1
```

It serves on `localhost:8000` by default.

To run additional local instances on other ports:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_showdown_server.ps1 -Port 8001
```

## Smoke battle

```powershell
& .\.venv\Scripts\python.exe .\scripts\smoke_battle.py --policy max-power --opponent random --max-turns 0 --json
```

## Small training run

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --device cpu --num-envs 2 --rollout-steps 8 --total-timesteps 32 --num-epochs 1 --num-minibatches 1 --log-interval 1
```

Training logs include rollout time, PPO update time, completed episodes, average completed battle duration, and average completed battle turns for the current update.

To train on a fixed Regulation F pool with team preview (bring 4 of 6):

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --battle-format gen9vgc2026regf --team-pool-format gen9vgc2026regf --num-envs 8 --server-urls localhost:8000,localhost:8001
```

To shard envs across multiple local Showdown servers:

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --device cpu --num-envs 16 --server-urls localhost:8000,localhost:8001
```

For a short env/server scaling benchmark over two rollout->PPO rounds:

```powershell
& .\.venv\Scripts\python.exe .\scripts\benchmark_parallel_training.py --battle-format gen9vgc2026regf --team-pool-format gen9vgc2026regf --env-counts 4,8,16 --server-counts 1,2,4 --rollout-steps 16 --num-updates 2
```

The benchmark writes per-run logs and a JSON summary under `benchmarks/<timestamp>/`.

## Building a VGC team pool

The repository includes a helper to fetch public VGC teams, validate them
against the local Pokemon Showdown checkout, and optionally generate extra
legal remix teams from the curated pool.

First make sure the local Showdown checkout is installed and up to date:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start_showdown_server.ps1 -InstallOnly -Update
```

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