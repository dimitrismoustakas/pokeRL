# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

- Uses `uv` package manager with Python 3.13
- Install: `uv venv --python 3.13` then `uv sync --dev`

### Local Pokemon Showdown Server

Required before running any training or smoke tests:

```powershell
# Start server (blocks until port is reachable)
powershell -ExecutionPolicy Bypass -File .\scripts\start_showdown_server.ps1 -Port 8000 -Detached

# Additional instances for parallel training
powershell -ExecutionPolicy Bypass -File .\scripts\start_showdown_server.ps1 -Port 8001 -Detached
```

## Common Commands

```bash
# Tests
pytest
pytest tests/test_observation_builder.py
pytest tests/test_observation_builder.py::test_pokemon_encoder

# Lint / format
ruff check pokerl/ scripts/ tests/
ruff format pokerl/ scripts/ tests/
```

```powershell
# Smoke battle (quick sanity check)
& .\.venv\Scripts\python.exe .\scripts\smoke_battle.py --policy max-power --opponent random --max-turns 0 --json

# Small CPU training run (~2 minutes)
& .\.venv\Scripts\python.exe .\scripts\train.py --device cpu --num-envs 2 --rollout-steps 8 --total-timesteps 32 --num-epochs 1 --num-minibatches 1 --log-interval 1

# Standard VGC training
& .\.venv\Scripts\python.exe .\scripts\train.py --battle-format gen9vgc2026regf --team-pool-format gen9vgc2026regf --num-envs 8 --server-urls localhost:8000,localhost:8001

# Build VGC team pool
& .\.venv\Scripts\python.exe .\scripts\build_vgc_team_pool.py --format gen9vgc2026regf --target-team-count 150

# Evaluate a checkpoint
& .\.venv\Scripts\python.exe .\scripts\evaluate_fixed_bot.py --checkpoint checkpoints/step_100000.pt

# Parallel training benchmark (output → benchmarks/<timestamp>/)
& .\.venv\Scripts\python.exe .\scripts\benchmark_parallel_training.py --battle-format gen9vgc2026regf --team-pool-format gen9vgc2026regf --env-counts 4,8,16 --server-counts 1,2,4
```

## Architecture

### Training Loop

1. **Rollout** (`pokerl/training/rollout.py`) — parallel envs step synchronously; agent produces actions + value estimates into a pre-allocated `RolloutBatch` with shape `(num_steps, num_envs, ...)`
2. **PPO update** (`pokerl/training/ppo.py`) — clipped surrogate loss with action masking; minibatches slice *envs* (not timesteps) to maintain LSTM gradient flow across episode boundaries
3. **League self-play** (`pokerl/training/league.py`, optional) — pool of frozen checkpoints as opponents; `_LeagueInferenceService` batches inference requests per checkpoint

### Environment (`pokerl/env/`)

- `showdown_env.py` — Gymnasium wrapper around poke-env; async event loop + thread pool for concurrent doubles battles
- `observation.py` — `ObservationBuilder` encodes full game state (pokemon stats/moves/items, field conditions, HP history)
- `actions.py` — dual-slot action space for doubles; separate masking per slot (moves + switches + tera)
- `team_preview.py` — lead selection + full team ordering at battle start
- `rewards.py` — sparse reward (win=+1 / loss=-1 only)

### Model (`pokerl/model/`)

`PokeAgent` (`agent.py`) composes three sub-modules:

- **Spatial Encoder** (`encoders.py`): `PokemonEncoder` (shared weights for all pokemon) → `TeamEncoder` (order-invariant multi-head attention over own team / opponent team separately) → `FieldEncoder` (active conditions)
- **Memory** (`memory.py`): pluggable LSTM or identity; hidden state zeroed on episode `done`; supports truncated BPTT via `bptt_chunk_steps`
- **Heads** (`heads.py`): autoregressive action head (slot 1 → slot 2), value head, team preview heads (selection + lead)

### Key Design Decisions

**LSTM-aware minibatching**: minibatches are env slices so the full rollout sequence can be unrolled with correct gradient flow. Slicing timesteps would break episode boundary handling.

**Action masking shape**: `(num_steps, num_envs, num_actions+1, num_actions)` — dim 2 encodes "how many actions are available per slot" as an implicit count.

**Stat-based embedding init** (`config.py → init_embeddings_from_stats`): species/move embeddings initialized from a linear projection of normalized base stats; improves cold-start performance considerably.

**Opponent pokemon encoding**: continuous features (PP remaining, etc.) are zeroed for opponent pokemon. This asymmetry is intentional — the model learns this as an implicit "is own" signal. Giving opponents their own `continuous_dim` is a known improvement not yet implemented.

**Item vocabulary**: named items from Showdown pokedex; unknown items fall back to hash-bucketed embeddings. Most competitive held items currently land in hash buckets, not named slots (see `KNOWN_ISSUES.md`).

**Team pool layout** (`data/team_pools/<format>/`):
- `curated/` — validated teams from public sources
- `generated/` — legal remix variants
- `pool/` — full usable pool (curated first, generated appended)
- `manifest.json` — metadata and inventory

### Configuration

All hyperparameters live in `pokerl/config.py` as dataclasses (`ModelConfig`, `EnvConfig`, `TrainingConfig`, `LeagueConfig`). Notable knobs: embedding dims, attention heads/layers, BPTT chunk size, PPO clipping ε, GAE λ, league pool size and refresh frequency.

## Development Principles

1. **Fail loudly** — no silent fallbacks; make failures visible
2. **Separate concerns** — keep code for different responsibilities in separate files
3. **No workarounds** — fix issues at their root, not by bypassing them
4. **Keep CLIs simple** — avoid flags that encourage hacks or hide configuration
5. **Avoid clutter** — fix suboptimal code directly instead of layering workarounds
