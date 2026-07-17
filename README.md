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
payoff coverage, and conditional emitted/branch/route/interaction/CANT/field
calibration. Those consequence labels come only from exact public protocol
transitions; they do not change the sparse terminal win/loss reward.

`--resume` accepts only versioned trainer checkpoints produced by the exact same model class, configuration, categorical vocabulary, and tensor schema. Trainer checkpoints and their uniquely paired league-state files are atomically replaced and cross-checked; together they restore Adam, global step, the exact completed-update boundary, Python/NumPy/PyTorch RNG, league sampling, and team-pool sampling. The paired state also pins the battle format, team-pool signature, vocabulary signature, and nullable evaluated-population cursor, so a later population workflow cannot silently reinterpret the run. Lightweight league snapshots are evaluation/opponent artifacts and cannot resume training.

To train on a fixed Regulation F pool with team preview (bring 4 of 6):

```powershell
& .\.venv\Scripts\python.exe .\scripts\train.py --battle-format gen9vgc2026regf --team-pool-format gen9vgc2026regf --num-envs 8 --opponent-groups 2
```

To evaluate whether a checkpoint is improving against a frozen past self,
sample both trained policies on a pre-planned schedule and mirror every team
pair across the two player roles:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_anchor.py --checkpoint checkpoints\current.pt --checkpoint-policy-id main-12 --anchor checkpoints\anchor.pt --anchor-policy-id main-11 --games 64 --action-mode sample --summary-path logs\main-12-v-main-11.json
```

`--games` is the number per role, so this command runs 128 battles. The planner
uses deterministic cyclic blocks: every complete block places every roster
exactly once on each side, and it exhausts directed pairings before repeating
one. Use
`--action-mode argmax` only as a separate deterministic diagnostic; the same
mode is applied to both checkpoints. Results separate wins/losses/draws, include
95% intervals, retain per-game teams and preview choices, and report Protect,
Fake Out, Trick Room, switching, and Tera diagnostics independently for each
external policy direction. The summary also
pins both checkpoint SHA-256 digests, battle format, team-pool and categorical
vocabulary signatures, and a hash of the mirrored role/team schedule. Summary
files are immutable: choose a new path rather than overwriting completed
evidence.

To measure player-role bias without pretending a checkpoint played against
itself is a strength comparison, pass the same checkpoint for both policies and
add `--same-policy-role-bias`. This requires mirrored play, omits population
evidence, swaps every team pair, and reports the external-player score advantage
over 50%.

For a multi-seed estimate, aggregate complete mirrored summaries by paired team
matchup rather than treating their two role games as independent samples:

```powershell
& .\.venv\Scripts\python.exe .\scripts\aggregate_anchor_evaluations.py --summary logs\candidate-v-anchor-s0.json --summary logs\candidate-v-anchor-s1.json --summary logs\candidate-v-anchor-s2.json --output logs\candidate-v-anchor-aggregate.json
```

The aggregate validates checkpoint, model, pool, format, and action-mode
identity and rejects duplicate evidence. Its hierarchical bootstrap resamples
seed strata with replacement, then resamples whole mirrored matchup pairs
within each selected seed, yielding an unconditional multi-seed interval while
preserving the pairing and reporting between-seed variation. Python, NumPy,
Torch, policy sampling, and team schedules are seeded. The current poke-env
local BattleStream API does not expose Showdown's battle seed, so summaries
explicitly retain
`simulator_rng_seeded: false`; evaluator code does not monkey-patch that private
session path.

Start a population from a versioned main checkpoint, then plan a genuine
best-response phase against either that frozen main or a solved population
mixture:

```powershell
& .\.venv\Scripts\python.exe .\scripts\bootstrap_population.py --checkpoint checkpoints\main-0.pt --team-pool-manifest data\team_pools\gen9vgc2026regf\manifest.json --policy-id main-0 --lineage-id main --output populations\revision-0.json
& .\.venv\Scripts\python.exe .\scripts\plan_exploiter_phase.py --population populations\revision-0.json --role main --phase-id main-exploiter-1 --learner-policy-id main-exploiter-1 --initializer-policy-id main-0 --target-main-policy-id main-0 --output phases\main-exploiter-1.json
```

Use `--role league` once the population contains a complete payoff matrix; the
planner solves and certifies the target mixture instead of accepting a manually
chosen opponent distribution. Both commands pin checkpoint bytes, model and
team-pool signatures, and immutable lineage. The resulting phase is consumed
by `train.py --phase-spec ...`.

Sampled mirrored summaries can close an empirical population row. Write each
new member as the exact JSON returned by `PolicyMember.to_dict()`, with its
checkpoint path interpreted relative to that member file, then build a child:

```powershell
& .\.venv\Scripts\python.exe .\scripts\build_population_revision.py --parent populations\revision-3.json --member-spec populations\main-exploiter-4.member.json --evidence logs\main-exploiter-4-v-main-1.json --evidence logs\main-exploiter-4-v-main-2.json --evidence logs\main-exploiter-4-v-main-3.json --output populations\revision-4.json
```

The builder verifies the summary bytes, checkpoint bytes, both role schedules,
W/L/D aggregation, and all signatures. It inherits already-verified parent
pairs and writes nothing unless the child has exactly one evidence artifact for
every unordered policy pair. Greedy (`argmax`) diagnostics are rejected for
training meta-games. There is deliberately no win-rate admission threshold or
reward shaping: a trained oracle enters the finite game only after its complete
frozen cross-play row exists.

The training loader independently revalidates exploiter semantics even for a
correctly hashed phase file. A main exploiter must target exactly one policy
whose role is `main`; a league exploiter's declared mixture must have exact
empirical-game exploitability at most `0.005` on a complete payoff matrix.

After an external controller has verified that child revision and selected the
member to expose to the adaptive main learner, `League.add_admitted_checkpoint`
installs its exact checkpoint bytes as a permanent, neutral-prior PFSP member.
The league then updates its difficulty estimate from ordinary on-policy results
against the changing main policy. This last installation is deliberately an
explicit update-boundary operation; the repository does not silently admit a
checkpoint merely because a best-response training phase finished.

For an absolute fixed-bot diagnostic with action, Tera, and team-preview
telemetry:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_fixed_bot.py --checkpoint checkpoints\current.pt --opponent simple-heuristics --games 76 --action-mode sample --summary-path logs\fixed_bot_eval.json
```

Fixed-bot and exact-preview evaluation can use independent own-roster and
opponent-panel manifests through `--model-team-pool-manifest` and
`--opponent-team-pool-manifest`. This is useful for holding the evaluated
roster fixed while sweeping a larger or genuinely held-out opponent panel.
Their summaries pin both pool signatures, the exact model-team hash, and the
checkpoint hash, and refuse to overwrite a different existing artifact.

To measure whether preview probabilities actually change with the opposing
roster, exactly enumerate all 180 legal bring/lead choices:

```powershell
& .\.venv\Scripts\python.exe .\scripts\evaluate_preview_policy.py --checkpoint checkpoints\current.pt --summary-path logs\preview_eval.json
```

The default Regulation F manifest contains only teams used for training. A
fifteen-team tournament-derived panel is checked in separately at
`data/team_pools/gen9vgc2026regf_heldout/manifest.json`; it covers hyper
offense, balance, control, and several forms of Trick Room. Pass that manifest
with `--team-pool-manifest` to any evaluation command. Its metadata marks it
evaluation-only, and the training loader rejects it rather than silently
leaking held-out teams into rollout or PFSP. It is a development validation
panel once results have informed design choices, not a permanently sealed test
set.

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

The retired eager-order and target-aware batching experiment is preserved in
[the experiment archive](docs/experiments/2026-07-17-lazy-orders-target-coalescing.md).
It records the design, pinned revisions, results, and a detached-worktree
recipe for revisiting the ideas without leaving experimental gates in the
training code.

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
