# pokeRL

pokeRL trains a recurrent actor-critic to play VGC doubles through the real
Pokemon Showdown simulator. Training uses sparse terminal rewards and a local
poke-env BattleStream backend; no websocket server is required.

## Documentation

- [Architecture and training semantics](docs/ARCHITECTURE.md)
- [Current measured status](docs/STATUS.md)
- [Archived rollout-efficiency experiment](docs/experiments/2026-07-17-lazy-orders-target-coalescing.md)

`docs/STATUS.md` describes only the latest accepted evidence. Historical run
details belong in immutable JSON summaries and Git history, not an
ever-growing Markdown diary.

## Setup

The project uses Python 3.13 and `uv`.

```powershell
uv venv --python 3.13.7
uv sync --dev
uv pip install -e ..\poke-env
```

The editable poke-env checkout must provide
`LocalBattleStreamConfiguration`.

Put Pokemon Showdown at
`%LOCALAPPDATA%\pokeRL\pokemon-showdown`, or set:

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
  --num-epochs 1 `
  --num-minibatches 1 `
  --log-interval 1 `
  --checkpoint-dir checkpoints\smoke
```

The measured local CUDA configuration uses 72 environments, two frozen
opponent groups, 24 BattleStream workers, three battles per worker, two
runtime loops, and 128 rollout steps:

```powershell
& .\.venv\Scripts\python.exe -m scripts.train `
  --device cuda `
  --battle-format gen9vgc2026regf `
  --team-pool-format gen9vgc2026regf `
  --team-pool-subset pool `
  --num-envs 72 `
  --opponent-groups 2 `
  --worker-count 24 `
  --battles-per-worker 3 `
  --runtime-loop-count 2 `
  --rollout-steps 128 `
  --num-updates 10 `
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
  --num-updates 10 `
  --checkpoint-dir checkpoints\run-name-resumed
```

Resume validation pins the model, tensor schema, vocabulary, team pool,
optimizer, completed-update boundary, RNG state, and league state. A frozen
league snapshot is not a resumable trainer checkpoint.

## Evaluation

Compare two compatible checkpoints on the same sampled team schedule and
mirror player roles:

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

Use a new summary path for every evaluation. Summaries pin checkpoint and
team-pool hashes and are intended to be immutable evidence.

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

The committed Regulation F training manifest is:

```text
data/team_pools/gen9vgc2026regf/manifest.json
```

The disjoint development panel is:

```text
data/team_pools/gen9vgc2026regf_heldout/manifest.json
```

Both contain published, locally validated teams. The training loader rejects
the held-out manifest. Once its results influence development, it is a
development-validation panel rather than a permanently untouched test set.
