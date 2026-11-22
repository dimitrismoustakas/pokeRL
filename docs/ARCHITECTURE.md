# Architecture and training semantics

This document records stable design boundaries. The code is authoritative for
tensor dimensions and default values.

## System boundary

Regulation F training and reciprocal policy matches run inside Pikapro's Rust
VecEnv. Rust constructs a view for each seat, including its own exact stats,
the opposing HP percentage, public events and the current legal menu. Python
maps those arrays to the policy's tensors. Fixed bots and population action
diagnostics still use poke-env's player protocol. Showdown is a separate
reference simulator for those commands and the team validator.

```text
public battle state
  -> observation encoder
  -> recurrent policy and value model
  -> legal joint action
  -> battle simulator
  -> sparse terminal result
```

The policy sees its own team and private request, plus information publicly
revealed by the opponent. The trainer uses unrevealed opponent attributes,
the opponent's selected actions and complete transition labels as masked
auxiliary targets.

The reward is `+1` for a win, `-1` for a loss, and the terminal draw outcome.
There is no HP, damage, knockout, board-position, or bot-imitation reward
shaping.

## Observation and representation

The native player views and the protocol observation builder share these inputs.

- both six-Pokémon rosters, masks, active positions, revealed information,
  stats, HP, status, boosts, moves, items, abilities, types, and Tera state;
- field, side, weather, terrain, Trick Room, and turn state;
- recent exact public protocol events;
- active-slot mechanics such as first active turn, consecutive protection
  state, active-turn count, the exact public Perish Song countdown, and the
  previous move's public resolution.

`pokerl/model/encoders.py` applies a shared Pokémon encoder and separate
set-attention team encoders. Field, recent-event, and active-mechanics
encoders provide structured context.

The representation has two temporal paths:

1. an LSTM carries information across the battle; and
2. a normalized residual projects the current spatial state directly around
   the LSTM.

The residual prevents current public facts from being compressed exclusively
through recurrent memory. The LSTM remains responsible for history that is not
fully represented in the current observation. Hidden state resets at episode
boundaries.

## Team preview

At preview, the policy sees both complete public rosters and produces one
exact distribution over the 180 legal ordered four-Pokémon choices. The head
factorizes that decision into:

1. one of the 15 four-of-six bring subsets; then
2. one of the 12 ordered lead pairs from that subset.

Subset scoring is symmetric over the selected members, so permuting roster
slots only permutes the semantic action probabilities. Live inference scores
all subsets in one vectorized pass and only the selected subset's lead orders.
Training, exact diagnostics, and deterministic argmax selection expose the
complete joint distribution without repeating the observation 180 times.

The preview head shares the spatial encoder and recurrent module with the
battle policy. It has its own value baseline and auxiliary opponent-preview
heads.

The live sampler records the chosen action, selected-action behavior log
probability, value, and policy version at episode start. It does not enumerate
all 180 actions on the reset path. After collection and before optimization,
one batched no-gradient pass freezes the complete old preview distribution for
exact KL diagnostics.

Preview is the zero-reward first transition into the battle. Its direct
truncated lambda-1 return stops at the first terminal result inside the rollout
or bootstraps once from the battle value at the rollout boundary. Starts at the
final boundary are valid. A long battle's preview is therefore trained once
when it occurs rather than being discarded or replayed at terminal.

## Battle policy

The battle action head is autoregressive:

```text
slot-one action
  -> slot-two action conditioned on slot one
  -> Tera use/slot conditioned on both
```

Every selection is masked to the simulator's current legal request. Candidate move
representations include move identity and public mechanics such as protection,
switching, healing, field creation, Fake Out, recharge, and charge behavior.
Move metadata describes mechanics, not strategic value.

Move action slots follow Showdown's one-based request indices rather than the
compacted `available_moves` list. The observation separately carries the exact
live move ID in each request slot, so disabled moves retain their holes while
one-entry locked and Struggle requests keep their true semantics. Forced
pseudo-moves absent from the vocabulary remain unknown rather than inheriting
an unrelated roster move.

The value head shares the observation, belief-conditioned representations, and
recurrent context with the policy.

## Belief and public-consequence auxiliaries

The hidden-information head predicts masked opponent item, ability, Tera type,
and move-set targets. A compact belief residual is added to public opponent
representations consumed by the action and value heads.

The opponent-intent head predicts semantic action kind, global move identity,
target, switch species, and factorized Tera intent. It currently supplies only
a dense trainer-only auxiliary loss. Its predictions are not consumed by the
deployed action or value heads, and exact opponent decisions remain private
labels.

The public-consequence model predicts what a selected action publicly resolves
into. Field outcomes are decomposed into:

- persistence when nothing changes;
- a change gate; and
- the alternative value when a change occurs.

Other branches distinguish emitted/non-emitted actions, route, target
interaction, miss, immunity, protection, failure, `CANT`, and related public
resolution classes. This decomposition does not assert that a move must
connect. Misses, immunities, failed actions, and probabilistic outcomes have
explicit labels.

Detached compact predictions are fed back into the policy. Full decoder logits
and exact targets exist only during training.

## PPO

`scripts/train.py` collects fixed-length sequences from parallel environments.
The rollout buffer computes generalized advantage estimates.
`pokerl/training/ppo.py` optimizes the policy with

- clipped policy ratios;
- clipped value changes;
- entropy regularization;
- gradient-norm clipping; and
- masked auxiliary losses.

Minibatches slice environments while retaining their full rollout sequences,
so recurrent state and episode boundaries remain valid.

One optimizer owns the shared model. It is currently selectable between Adam
and AdamW; AdamW applies decay only to non-embedding matrix parameters. Preview
and battle advantages are normalized together, and their
policy/value/entropy sums are weighted by the complete rollout's number of
decisions. Preview samples are independently shuffled and paired with the
full-environment recurrent battle minibatches. Each pair produces one combined
loss, one backward pass, one global gradient clip, and one optimizer step.
Four epochs by four minibatches therefore means 16 optimizer steps.

The complete old preview distribution is frozen before those steps and used
for exact `KL(old || current)`. The action-time selected log probability remains
the PPO behavior reference. This preserves one behavior-policy version for
both preview and battle data.

Production uses `gamma=1` for undiscounted win probability. Battle GAE retains
lambda `0.95`; preview uses the direct truncated lambda-1 return described
above.

Tera has separate use-now and eligible-slot decisions. A temporary behavior
epsilon is annealed over the configured opening fraction of a run; policy
log-probabilities retain the behavior-policy correction. Separate entropy
terms remain after epsilon reaches zero.

## Main self-play

Normal training follows one main policy trajectory:

- snapshots enter a permanent on-disk archive;
- bounded LRU caches limit resident frozen models, not archive history;
- hard-opponent PFSP uses discounted current-policy outcomes;
- uniform coverage prevents the archive from becoming unreachable; and
- opponent groups receive distinct snapshots when possible.

This guards against forgetting known historical policies. It cannot create a
counter-strategy that the single main trajectory never discovers.

## Population and exploiters

The repository implements frozen-population best-response components, but
does not contain an automatic background PSRO loop.

### Main exploiter

A main exploiter starts from a specified frozen policy's weights with a fresh
optimizer and trains ordinary sparse-reward PPO against one immutable main
policy.

### League exploiter

A league exploiter trains against a solver-certified mixture from a complete
sampled-policy payoff matrix. The phase validator rejects incomplete,
incompatible, or deterministic-argmax meta-games.

### Artifact lifecycle

The implemented artifact workflow is:

```text
trainer checkpoint
  -> population revision
  -> immutable phase specification
  -> best-response training checkpoint
  -> mirrored sampled cross-play evidence
  -> complete child population revision
  -> explicit admission to a later main league
```

The relevant entry points are:

- `scripts/bootstrap_population.py`
- `scripts/plan_exploiter_phase.py`
- `scripts/train.py --phase-spec ...`
- `scripts/evaluate_anchor.py`
- `scripts/build_population_revision.py`
- `League.add_admitted_checkpoint`

A completed best-response phase does **not** silently modify a running main
league. Evaluation, child-revision construction, selection, and admission are
explicit orchestration steps. A normal `main_self_play` command never launches
an exploiter. `League.add_admitted_checkpoint` is implemented and tested, but
there is currently no production controller or supported CLI that applies a
verified child population to a resumable main league.

## Checkpoints and resume

Trainer checkpoints contain model and optimizer state, counters, schema metadata,
and RNG state. A paired league or phase-state file records the exact opponent
sampler boundary and assignments.

Resume fails if the model signature, tensor schema, vocabulary, team pool,
battle format, completed update, or paired state disagrees. Frozen league
snapshots intentionally omit trainer state and cannot resume learning.

Best-response initialization is different from resume: it loads compatible
weights but creates a fresh optimizer, counters, and RNG streams.

## Runtime architecture

Pikapro advances many battles in one call on up to four Rust threads. Each
lane stops at a real learner choice. Opponent-only requests, including an
Eject Pack switch before turn one, resolve with the assigned frozen policy.
Each new game samples new teams and keeps its assigned opponent through the
terminal result. A strict resume restores saved sampling state and starts
fresh battles from a seeded per-lane, per-episode stream.

Frozen decisions from one vector step enter inference together, grouped by
checkpoint. CUDA graph replay reuses fixed buffers. Team preview reuses an
encoding only for identical player-visible input under the same weights.
The learner discards cached encodings on a policy update. Evicting a frozen
model releases its cache. The Showdown reference uses local BattleStream
workers and runtime loops.

The training summary reports rollout and PPO wall time separately.
`--collect-timing` also records native stepping, player-view projection,
previews and frozen inference. Some of these timers nest. Compare seconds per
complete update at a fixed decision budget when measuring throughput.
