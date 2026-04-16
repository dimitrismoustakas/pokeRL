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

## Smoke battle

```powershell
& .\.venv\Scripts\python.exe .\scripts\smoke_battle.py --policy max-power --opponent random --max-turns 0 --json
```

## Small training run

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --device cpu --num-envs 2 --rollout-steps 8 --total-timesteps 32 --num-epochs 1 --num-minibatches 1 --log-interval 1
```

Training logs include rollout time, PPO update time, completed episodes, average completed battle duration, and average completed battle turns for the current update.