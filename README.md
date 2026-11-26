# pokeRL

pokeRL trains a recurrent policy to play Pokémon Regulation F doubles. Pikapro runs the battles in Rust. The policy learns from wins and losses on 170 published level-50 teams.

## Install

Use Python 3.13, uv and Rust. Windows also requires the MSVC C++ Build Tools.

```powershell
rustup toolchain install 1.94.1 --profile minimal
$env:RUSTUP_TOOLCHAIN = "1.94.1"
uv sync --locked --python 3.13.7
```

## Train

This example collects two small PPO updates on CPU.

```powershell
uv run python -m scripts.train --device cpu --num-envs 2 --rollout-steps 8 `
  --num-epochs 1 --num-minibatches 1 --num-updates 2 `
  --tera-exploration-decay-updates 2 --checkpoint-dir checkpoints/example
```

On the local 4 GB Quadro T2000, use `--device cuda --num-epochs 4 --opponent-groups 2`. The 576-environment configurations favor throughput. Use 144 environments to collect 64 consecutive steps per environment at 9,216 decisions per update.

| `--num-envs` | `--rollout-steps` | Decisions per update | `--num-minibatches` | Frozen inference batch | Cache size |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 576 | 32 | 18,432 | 8 | 72 | 5 |
| 576 | 16 | 9,216 | 4 | 72 | 8 |
| 144 | 64 | 9,216 | 4 | 36 | 4 |

Set the frozen inference batch with `--league-cuda-graph-batch-size`. Set both `--league-size` and `--league-cuda-graph-cache-size` to the cache size. `--league-sampler` accepts `random` or `balanced`.

A fresh run needs its own checkpoint directory. To continue, pass `--resume checkpoints/example/final.pt` with the same rollout settings and the number of additional updates. To train against one fixed policy, pass `--opponent-checkpoint PATH`. Use `--init-from PATH` to initialize a fresh learner from saved weights.

## Evaluate

Each matchup plays both seats with the same teams and simulator seed. The default evaluation pool contains 15 held-out teams.

```powershell
uv run python -m scripts.evaluate_match checkpoints/run-a/final.pt checkpoints/run-b/final.pt --pairs 210
```

Team imports use a local Pokémon Showdown checkout for Regulation F legality. Run `python -m scripts.import_published_regf_teams --showdown-dir PATH` to preview an import.

[Model, simulator and training architecture](docs/ARCHITECTURE.md)
