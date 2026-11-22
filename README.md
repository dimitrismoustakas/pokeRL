# pokeRL

pokeRL trains a recurrent actor-critic to play Regulation F doubles. Training
and reciprocal policy matches use Pikapro's Rust vector environment. The
model reads each player's visible battle state and legal actions directly.
Games use level 50 with the original IVs, EVs and natures. Wins, losses and
draws supply the entire reward.

## Documentation

- [Architecture and training semantics](docs/ARCHITECTURE.md)
- [Independent review of current evidence and open decisions](docs/INDEPENDENT_REVIEW.md)

## Setup

Install Python 3.13, uv and Rust. Windows builds also require MSVC C++ Build
Tools. The lockfile uses source distributions of Pikapro and poke-env in
`vendor/`. The first install compiles the Rust extension for the local
platform. Subsequent installs can reuse uv's wheel cache.

```powershell
rustup toolchain install 1.94.1 --profile minimal
$env:RUSTUP_TOOLCHAIN = "1.94.1"
uv sync --locked --python 3.13.7
```

The two patches in `vendor/` describe the changes to the upstream sources.
To rebuild an archive, check out its base revision, apply the corresponding
patch with `git apply`, then run the build command from that checkout.

| Package | Base revision | Source build |
| --- | --- | --- |
| Pikapro | [spedr/pikapro-engine at 0bfff72](https://github.com/spedr/pikapro-engine/commit/0bfff72d3b6a15ce56a226655b8fce45a15d3016) | `uvx --from maturin==1.16.0 maturin sdist --manifest-path crates/pikapro-py/Cargo.toml` |
| poke-env | [dimitrismoustakas/poke-env at 21292fd](https://github.com/dimitrismoustakas/poke-env/commit/21292fd82d317ec7e5a0b0b59822067095477c5f) | `uv build --sdist` |

Showdown supplies the format-specific team validator and an independent
simulator reference. For a reference battle, pass `--simulator showdown`
and set `--showdown-dir` or the environment variable below.

```powershell
$env:POKERL_SHOWDOWN_DIR = "C:\path\to\pokemon-showdown"
```

## Checks

```powershell
& .\.venv\Scripts\python.exe -m pytest
& .\.venv\Scripts\ruff.exe check pokerl scripts tests
& .\.venv\Scripts\ruff.exe format --check pokerl scripts tests
```

A simulator sanity check:

```powershell
& .\.venv\Scripts\python.exe .\scripts\smoke_battle.py `
  --policy max-power `
  --opponent random `
  --max-turns 0 `
  --json
```

## Training

A small CPU smoke run:

```powershell
& .\.venv\Scripts\python.exe -m scripts.train `
  --device cpu `
  --num-envs 2 `
  --rollout-steps 8 `
  --num-updates 2 `
  --tera-exploration-decay-updates 2 `
  --num-epochs 1 `
  --num-minibatches 1 `
  --log-interval 1 `
  --checkpoint-dir checkpoints\smoke
```

The measured local CUDA configuration collects 9,216 decisions per update
across 144 battles. Four frozen policies can be resident at once. Every
archived opponent remains eligible for sampling.

```powershell
& .\.venv\Scripts\python.exe -m scripts.train `
  --device cuda `
  --battle-format gen9vgc2026regf `
  --team-pool-format gen9vgc2026regf `
  --team-pool-subset pool `
  --num-envs 144 `
  --opponent-groups 2 `
  --league-size 4 `
  --league-cuda-graph-batch-size 36 `
  --rollout-steps 64 `
  --optimizer adamw --lr 0.0002 --weight-decay 0.05 `
  --num-updates 10 `
  --tera-exploration-decay-updates 113 `
  --num-epochs 4 `
  --num-minibatches 4 `
  --log-interval 1 `
  --save-interval 10 `
  --checkpoint-dir checkpoints\run-name
```

Resume only from a versioned trainer checkpoint and its paired state file:

```powershell
& .\.venv\Scripts\python.exe -m scripts.train `
  --resume checkpoints\run-name\final.pt `
  --num-envs 144 --rollout-steps 64 `
  --league-size 4 --league-cuda-graph-batch-size 36 `
  --num-updates 10 `
  --checkpoint-dir checkpoints\run-name-resumed
```

Resume restores the saved optimizer family, learning rate, weight decay and
exploration schedule. The model, tensor schema, vocabulary, team pool,
completed update, RNG state and paired league state must agree. Rollout
geometry and GPU cache capacity are runtime settings. Live battles restart
from the saved sampling boundary.

## Evaluation

For playing strength, run sampled games on the held-out teams. Each checkpoint
plays both sides of each matchup with matching battle seeds. The report
includes a 95% interval obtained by resampling whole roster pairs.

```powershell
& .\.venv\Scripts\python.exe -m scripts.evaluate_match `
  checkpoints\candidate.pt checkpoints\anchor.pt `
  --pairs 1680 --seed 551733 --summary-path logs\heldout-match.json
```

Population admission uses its own mirrored evaluation with stable policy IDs
and detailed action diagnostics. It retains poke-env's player protocol.

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_anchor.py `
  --checkpoint checkpoints\candidate.pt `
  --checkpoint-policy-id candidate `
  --anchor checkpoints\anchor.pt `
  --anchor-policy-id anchor `
  --games 64 `
  --action-mode sample `
  --seed 0 `
  --summary-path logs\candidate-v-anchor-s0.json
```

`--games 64` means 64 games in each direction, or 128 total. Sampled policy
evaluation is the strength estimate. `--action-mode argmax` is a separate
deterministic diagnostic: it always chooses the highest-probability legal
action and does not sample from the policy.

Inspect a fixed roster against diagnostic bots:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_fixed_bot.py `
  --checkpoint checkpoints\candidate.pt `
  --opponent simple-heuristics `
  --games 84 `
  --action-mode sample `
  --preview-mode learned `
  --summary-path logs\fixed-bot.json
```

Enumerate the complete preview distribution:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_preview_policy.py `
  --checkpoint checkpoints\candidate.pt `
  --summary-path logs\preview.json
```

Preview reports distinguish sampled choices from the
**most-probable (`argmax`) choice**. An argmax bring or lead is simply the
single choice with the largest model probability for that opposing roster; it
does not mean a hand-written heuristic or a strategically optimal choice.

Use a new summary path for each evaluation. Population summaries also pin
checkpoint and team-pool hashes for admission.

## Self-play and exploiters

Ordinary training is main-policy self-play against a permanent archive sampled
with PFSP. Exploiter training is a separate, explicit best-response workflow;
it is not launched automatically by a normal training command.

The workflow is:

1. Bootstrap a population revision from a frozen main checkpoint.
2. Plan a main- or league-exploiter phase.
3. Train with `scripts.train --phase-spec ...`.
4. Evaluate the new frozen policy against every population member.
5. Build a complete child population revision from that evidence.
6. Explicitly admit a selected checkpoint to a future main league.

See [the population section](docs/ARCHITECTURE.md#population-and-exploiters)
for the exact boundary between implemented components and manual orchestration.
There is currently no production controller or supported CLI for step 6.

## Team pools

| Use | Teams | Manifest |
| --- | ---: | --- |
| Training | 170 | [Regulation F training teams](data/team_pools/gen9vgc2026regf/manifest.json) |
| Development evaluation | 15 | [Original evaluation panel](data/team_pools/gen9vgc2026regf_heldout/manifest.json) |
| Independent roster evaluation | 30 | [Additional evaluation panel](data/team_pools/gen9vgc2026regf_heldout30/manifest.json) |

Every team plays at level 50 with its published spread. The 15-team panel
shares two visible six-species combinations with training. The 30-team panel
excludes all visible combinations in the 170 training teams and the
development panel. Both evaluation manifests
reject training use. Pass `--team-pool-manifest` to select a panel.
