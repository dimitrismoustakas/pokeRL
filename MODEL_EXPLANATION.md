# pokeRL Model Architecture Reference

This document describes the current model and training stack implemented in:

- `pokerl/model/agent.py`
- `pokerl/model/encoders.py`
- `pokerl/model/memory.py`
- `pokerl/model/heads.py`
- `pokerl/env/observation.py`
- `pokerl/env/actions.py`
- `pokerl/training/rollout.py`
- `pokerl/training/ppo.py`
- `pokerl/training/league.py`
- `scripts/train.py`

If this document and the code ever disagree, the code is authoritative.

## 1. High-Level Overview

`PokeAgent` is a recurrent actor-critic for doubles battles. The policy is trained with PPO against opponents sampled from a self-play league of frozen checkpoints. Team preview is trained separately with a preview-specific value baseline plus opponent-model auxiliary losses.

Battle-turn pipeline:

```text
observation
  -> spatial encoder
  -> memory (LSTM or identity)
  -> action head
       slot 1 action
       -> slot 2 action conditioned on slot 1
       -> tera choice conditioned on both slot actions
  -> value head
```

Preview pipeline:

```text
synthetic preview observation
  -> same spatial encoder
  -> same memory module
  -> team preview head
       canonical bring set
       -> ordered lead pair
  -> opponent preview head (auxiliary only)
  -> preview value head
```

Core properties:

- Doubles action selection is autoregressive across the two active slots.
- Terastallization is a separate third decision, not folded into the per-slot action IDs.
- The critic is recurrent: it consumes the memory output, not the raw spatial context.
- Team preview shares the same spatial encoder and memory stack as battle turns.
- Opponent selection at team preview is supervised with auxiliary heads for "bring" and "lead pair" prediction.

## 2. Observation Structure

Observation building lives in `pokerl/env/observation.py`.

### 2.1 Per-Pokemon Inputs

Each side contributes up to 6 Pokemon. For each Pokemon, the encoder receives:

Categorical tensors:

| Feature | Shape | Notes |
|---|---|---|
| `species` | `(6,)` | species vocab index |
| `moves` | `(6, 4)` | revealed/known moves |
| `last_move` | `(6,)` | most recent revealed or used move |
| `ability` | `(6,)` | ability vocab index |
| `item` | `(6,)` | named item or hash-bucketed fallback |
| `types` | `(6, 3)` | type 1, type 2, tera type |

Continuous features: `44` floats per Pokemon.

| Group | Dims | Details |
|---|---|---|
| HP fraction | 1 | current HP / max HP |
| Status one-hot | 7 | none, BRN, FRZ, PAR, PSN, SLP, TOX |
| Boost stages | 7 | atk, def, spa, spd, spe, accuracy, evasion; divided by 6 |
| Volatiles | 16 | confusion, substitute, leech seed, taunt, encore, torment, disable, yawn, perish song, protect, endure, focus energy, attract, curse, nightmare, flash fire |
| Move PP fractions | 4 | own side only; opponent entries stay zero |
| Extra features | 9 | protect counter, status counter, first turn, revealed, known move count, ability known, item known, tera known, is terastallized |

Important asymmetries:

- Opponent move PP is not tracked; those 4 values remain zero.
- Unknown categorical values are mapped to `UNKNOWN_IDX`.
- Own preview observations are built from the packed team; opponent preview observations expose visible species only.

### 2.2 Field Features

`FieldEncoder.INPUT_DIM = 36`.

| Group | Dims | Details |
|---|---|---|
| Weather | 10 | none flag, 8 weather types, normalized weather age |
| Terrain | 6 | none flag, 4 terrain types, normalized terrain age |
| Own side conditions | 8 | stealth rock, spikes, toxic spikes, sticky web, reflect age, light screen age, aurora veil age, tailwind age |
| Opp side conditions | 8 | same layout as own side |
| Global | 4 | trick room age, own tera used, opp tera used, normalized turn |

### 2.3 Team Metadata

Additional tensors passed outside the per-Pokemon feature blocks:

- `own_roles`, `opp_roles`: `(6,)`, values in `{ACTIVE_1, ACTIVE_2, BENCH}`
- `own_mask`, `opp_mask`: `(6,)`, padding mask (`True` means unused slot)
- `own_active_indices`, `opp_active_indices`: `(2,)`, indices of the currently active Pokemon within the ordered team list
- `own_active_present`, `opp_active_present`: `(2,)`, whether each active slot exists

The team ordering puts active Pokemon first and keeps stable team-slot identity for switches. Missing actives are gathered and then zeroed by `SpatialEncoder._gather_slots`.

### 2.4 Item Vocabulary

Item indexing is handled in `pokerl/data.py`:

- A small named-item tier is built from required-form items in the Showdown Pokedex.
- Everything else falls into hash buckets via `item_index(...)`.

That means many competitive items still collide in the embedding table.

## 3. Model Architecture

Entry point: `pokerl/model/agent.py`.

`PokeAgent` owns:

- `spatial`: `SpatialEncoder`
- `memory`: `LSTMMemory` or `IdentityMemory`
- `action_head`: `ActionHead`
- `team_preview_head`: `TeamPreviewHead`
- `opponent_preview_head`: `OpponentPreviewHead`
- `value_head`: `ValueHead`
- `preview_value_head`: `ValueHead`
- `hidden_info_head`: `HiddenInfoHead` — auxiliary; predicts end-of-episode opponent item / ability / tera type / move set per pokemon
- `opponent_action_head`: `OpponentActionHead` — auxiliary; predicts the opponent's chosen (a1, a2, tera) this turn

If `ModelConfig.init_embeddings_from_stats` is enabled, `GameData` initializes species and move embeddings from stats before training.

### 3.1 PokemonEncoder

Implemented in `pokerl/model/encoders.py`.

Per-Pokemon input width:

- species embedding: `64`
- 4 move embeddings: `4 * 32 = 128`
- last-move embedding: `32`
- ability embedding: `32`
- item embedding: `32`
- 3 type embeddings: `3 * 16 = 48`
- continuous features: `44`

Total input width: `380`

Network:

```text
380 -> Linear(256) -> ReLU -> Linear(128) -> ReLU
```

Output: `(batch, 128)` per Pokemon.

The same encoder is reused for all own and opponent Pokemon.

### 3.2 TeamEncoder

Also in `pokerl/model/encoders.py`.

There are two separate instances:

- `team_own`
- `team_opp`

Each one:

- adds a learned role embedding with 3 roles and width `128`
- runs a `TransformerEncoder` over the 6-Pokemon set

Default layer shape:

```text
d_model = 128
nhead = 4
num_layers = 2
dim_feedforward = 256
dropout = 0.0
norm_first = True
```

The module is permutation-equivariant with respect to team order. The model gets role information, but there is no slot-position embedding.

### 3.3 FieldEncoder

`FieldEncoder` is a one-layer MLP:

```text
36 -> Linear(64) -> ReLU
```

Output: `(batch, 64)`.

### 3.4 SpatialEncoder

`SpatialEncoder` combines the Pokemon, team, and field encoders.

Outputs:

| Name | Shape | Meaning |
|---|---|---|
| `context` | `(batch, 320)` | mean-pooled own team `(128)` + mean-pooled opp team `(128)` + field `(64)` |
| `active_reps` | `(batch, 2, 128)` | own active pair |
| `own_reps` | `(batch, 6, 128)` | full own team after attention |
| `opp_reps` | `(batch, 6, 128)` | full opponent team after attention |
| `opp_active_reps` | `(batch, 2, 128)` | opponent active pair |

Important details:

- Team means are mask-aware; padded slots are ignored.
- Active reps are gathered by stable team index, then multiplied by `*_active_present`.
- The global `context` is purely spatial. Temporal state is added later by `memory`.

### 3.5 Memory

Implemented in `pokerl/model/memory.py`.

Two options exist:

`IdentityMemory`

- single linear projection `input_dim -> hidden_dim`
- no recurrent state

`LSTMMemory`

- `nn.LSTM(input_dim=320, hidden_dim=256, batch_first=True)`
- accepts either `(batch, 320)` or `(batch, seq, 320)`
- returns the LSTM output plus `(h, c)`

Important implementation detail:

- The memory module does not reset hidden state on `done` and does not perform truncated BPTT by itself.
- Hidden-state zeroing and optional detaching are handled by the rollout loop and PPO trainer.

### 3.6 ActionHead

Implemented in `pokerl/model/heads.py`.

The battle policy makes three sequential decisions:

1. slot-1 action
2. slot-2 action conditioned on slot 1
3. tera choice conditioned on both slot actions

#### Per-slot action universe

`ACTIONS_PER_SLOT = 27`:

- `0..19`: move index `0..3` crossed with target index `0..4`
- `20..25`: switch to stable own team slot `0..5`
- `26`: pass

Tera is separate:

- `0`: no tera
- `1`: tera slot 0
- `2`: tera slot 1

#### Static action features

For each of the 27 per-slot actions, the head builds a static embedding by summing:

- action ID embedding
- action kind embedding (`move`, `switch`, `pass`)
- move-slot embedding
- target embedding
- switch-slot embedding

All of these live in the `action_hidden` width (`256` by default).

#### Dynamic action entity features

Each candidate action also gets a dynamic entity representation:

- move actions gather a target-side entity from `[opp0, opp1, partner, self, zero, zero]`
- switch actions gather the destination team slot from `own_team_reps`
- pass resolves to a zero vector

That entity is projected through:

```text
128 -> Linear(256) -> ReLU
```

#### Explicit matchup features

The action head now also scores each candidate with hand-aligned move matchup features computed from the live observation plus move/type metadata from `GameData`.

Examples include:

- move base power, accuracy, and priority
- move category (`status`, `physical`, `special`)
- current STAB and potential tera STAB gain
- current target effectiveness
- best/mean foe effectiveness across the two active opponents
- current expected power proxy and tera power gain proxy
- spread / ally-hit flags for targetless moves

#### Slot 1

Base query:

```text
[memory_out, own_active_0, own_active_1, opp_active_0, opp_active_1]
-> Linear(256) -> ReLU
```

Opponent reserve summary:

- `MaskedSetAttention(query1, opp_team_reps, opp_bench_mask)`
- the mask excludes padded slots and non-bench roles, so this attends over opponent bench only

Fused query:

```text
[base_query1, reserve_summary] -> Linear(256) -> ReLU
```

Scoring:

```text
[query, static_action_features, dynamic_entity_features, explicit_matchup_features]
-> Linear(256) -> ReLU -> Linear(1)
```

Illegal actions from `masks["joint"][0]` are set to `-1e8` before the categorical distribution is formed.

#### Slot 2

Slot 2 repeats the same scoring pipeline, but its query also includes the chosen slot-1 action embedding:

```text
[memory_out, own_active_1, own_active_0, opp_active_0, opp_active_1, a1_emb]
-> Linear(256) -> ReLU
```

Legality comes from row `a1 + 1` of the joint mask.

#### Tera exploration during rollout

Turn-level tera sampling supports optional uniform-over-valid exploration,
controlled by `ActionHead.forward(..., tera_exploration_prob=epsilon)`. When
`epsilon > 0` and there are at least two valid tera options for a batch item,
the policy's tera sample is replaced with a uniform draw with probability
`epsilon`. This intentionally breaks the "tera always on turn 1" attractor:
once the policy tera's turn 1, `can_tera` is False for every later turn in
the episode, so without exploration the policy never observes the
counterfactual trajectory where it saved tera.

The stored log-prob for the taken action is always computed under the
policy distribution (not the mixed behavior distribution), so PPO's
importance ratio is a slightly off-policy approximation. For small
`epsilon` this is well within the clip window. Exploration is only active
when `actions is None` (rollout path) — the PPO update path passes stored
actions and is unaffected.

The rollout loop in `scripts/train.py` linearly decays epsilon from
`tera_exploration_start` to `tera_exploration_end` across
`tera_exploration_decay_updates` PPO updates.

#### Tera decision

After `a1` and `a2` are chosen, the model makes a third categorical choice:

```text
[memory_out, own_active_0, own_active_1, opp_active_0, opp_active_1, a1_emb, a2_emb, a1_matchup_features, a2_matchup_features]
-> Linear(256) -> ReLU
-> Linear(3)
```

The tera mask is derived inside `ActionHead.forward(...)` from:

- `masks["can_tera"]`
- whether the chosen slot action is a move

So the policy does not consume a precomputed `(a1, a2, tera)` legality tensor during normal inference.

#### ActionHead outputs

`ActionHead.forward(...)` returns:

- `actions`: `(batch, 3)` with `[a1, a2, tera_choice]`
- `log_prob`: `(batch,)`, sum of the three categorical log-prob terms
- `entropy`: `(batch,)`, sum of the three entropies

### 3.7 ValueHead

The critic is implemented in `pokerl/model/heads.py` and is recurrent in the current code.

Inputs:

- `context`: the output of `memory`, not the raw `SpatialEncoder` context
- `own_team_reps`
- `opp_team_reps`
- `own_bench_mask`
- `opp_bench_mask`

Structure:

1. `MaskedSetAttention(context, own_team_reps, own_bench_mask)` -> own bench summary
2. `MaskedSetAttention(context, opp_team_reps, opp_bench_mask)` -> opponent bench summary
3. MLP over `[memory_out, own_bench_summary, opp_bench_summary]`

MLP:

```text
(256 + 128 + 128) -> Linear(256) -> ReLU -> Linear(1)
```

If a side has no valid bench targets, the corresponding attention summary collapses to zeros.

### 3.8 TeamPreviewHead

`TeamPreviewHead` now separates bring selection from lead ordering.

Bring stage:

1. accumulate the sum of already-picked own representations
2. project `[memory_out, picked_sum] -> context_dim`
3. attend over opponent reps with `MaskedSetAttention`
4. score each still-available own Pokemon from `[pokemon_rep, projected_context, opponent_summary]`
5. pick `pick_size` unique Pokemon in canonical increasing team-slot order

Lead stage:

1. summarize the chosen bring set
2. project `[memory_out, bring_summary] -> context_dim`
3. score all ordered pairs from `[left_rep, right_rep, bring_summary, opponent_summary, projected_context]`
4. mask out self-pairs, unchosen Pokemon, and padding
5. choose an ordered lead pair

Final Showdown preview order:

- picks `0` and `1` are the chosen ordered lead pair
- the remaining brought Pokemon are appended in canonical team-slot order

Outputs:

- `bring_picks`: `(batch, pick_size)` canonical bring set
- `lead_pair`: `(batch, 2)` ordered leads
- `picks`: `(batch, pick_size)`
- `log_prob`: bring-set log-prob plus lead-pair log-prob
- `entropy`: bring entropy plus lead-pair entropy

### 3.8b HiddenInfoHead (auxiliary)

Auxiliary head that runs only during the PPO update, not during rollout. For
each opponent pokemon it emits four heads fed from the same `opp_reps` tensor
used by the value and action heads:

- `item_logits: (batch, 6, num_items)` — cross-entropy label is the opponent's
  item at end-of-episode, masked by `ITEM_KNOWN_OFFSET` at end-of-episode.
- `ability_logits: (batch, 6, num_abilities)` — analogous to item.
- `tera_type_logits: (batch, 6, num_types)` — `opp_types[..., 2]` at
  end-of-episode, masked by `TERA_KNOWN_OFFSET`.
- `move_logits: (batch, 6, num_moves)` — multi-label BCE, target is the
  multi-hot of moves revealed by end-of-episode (slot 0 aka "no move" is
  explicitly excluded). Per-pokemon mean across the vocab.

Labels come from the rollout buffer's retroactive fill: when an env reports
`done` at step `t`, the buffer copies obs[t]'s opp_* features onto every
timestep from the episode's start through `t`, and marks
`aux_label_valid=True` for that slice.

### 3.8c OpponentActionHead (auxiliary)

Auxiliary head predicting the opponent's chosen action this turn from
`[memory_out, opp_active_0, opp_active_1, own_active_0, own_active_1]`.
Factorized into three independent categorical outputs:

- `opp_a1_logits: (batch, ACTIONS_PER_SLOT)` — CE against the observed `a1`
- `opp_a2_logits: (batch, ACTIONS_PER_SLOT)` — CE against observed `a2`
- `opp_tera_logits: (batch, NUM_TERA_CHOICES)` — CE against observed tera

Labels come from the env: `LeaguePlayer` exposes its chosen `(a1, a2, tera)`
array as `_last_actions` after each `choose_move`, and `BattleEnv.step`
propagates it on the per-step `info` dict under the key `opp_actions`. The
rollout buffer stores these with an `opp_actions_valid` flag; non-league
opponents leave the flag False and the loss masks them out.

Only the CE signal is used for now; commit 5 (Option B) will reuse the
distribution over opponent actions as an input to the policy.

### 3.9 OpponentPreviewHead

This head is only used for auxiliary training signal during team preview.

It predicts:

- `opponent_bring_logits`: `(batch, team_size)` independent bring scores
- `opponent_lead_pair_logits`: `(batch, team_size * team_size)` ordered lead-pair scores

Details:

- self-pairs are masked out
- padded slots are masked out
- logits are flattened after masking

### 3.10 Forward Paths

Battle turn:

```text
obs
  -> spatial(obs)
  -> memory(context, hidden)
  -> action_head(...)
  -> value_head(memory_out, own_reps, opp_reps, own_bench_mask, opp_bench_mask)
```

Team preview:

```text
obs
  -> spatial(obs)
  -> memory(context, hidden=None at preview time in current training flow)
  -> team_preview_head(...)
  -> opponent_preview_head(...)
```

Value-only bootstrap:

```text
obs
  -> spatial(obs)
  -> memory(context, hidden)
  -> value_head(...)
```

## 4. Action Masking and Legality

Action helpers live in `pokerl/env/actions.py`.

The policy consumes:

- `joint`: shape `(ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT) = (28, 27)`
- `can_tera`: shape `(2,)`

Interpretation of `joint`:

- row `0`: legal slot-1 actions
- row `1 + a1`: legal slot-2 actions conditioned on chosen `a1`

This structure enforces constraints such as:

- both Pokemon cannot switch into the same reserve slot
- illegal move targets are removed
- pass-only states are still representable

There is also a helper `build_tera_action_mask(...)`, but the model currently derives tera legality internally from `can_tera` plus the selected slot actions.

## 5. Training Semantics

### 5.1 Rollout Collection

Rollout collection is orchestrated in `scripts/train.py` and stored in `pokerl/training/rollout.py`.

Default high-level settings:

- `EnvConfig.num_envs = 16`
- `TrainingConfig.rollout_steps = 128`

For each environment step:

1. observations and action masks are copied to the training device
2. `agent(obs_t, hidden, action_masks=masks_t)` runs one policy step
3. actions, log-probs, and values are stored in `RolloutBuffer`
4. environments are stepped
5. rewards and dones are written back to the buffer
6. recurrent state for finished envs is zeroed in the rollout loop

`RolloutBuffer` stores:

- sequence-layout observations and masks
- actions shaped `(T, E, 3)`
- log-probs, values, rewards, dones
- advantages and returns
- the recurrent hidden state at the start of the rollout

GAE is computed after the rollout using a bootstrap value from:

```text
last_value = agent.get_value(final_obs, hidden)
```

### 5.2 PPO Update

Implemented in `pokerl/training/ppo.py`.

Important detail: minibatches split over environments, not timesteps. Each minibatch keeps full time sequences so the recurrent state stays valid.

Per minibatch:

1. slice a subset of environments
2. flatten the spatial encoder work over `T * E_mb`
3. reshape back to `(T, E_mb, ...)`
4. unroll memory and heads one timestep at a time
5. zero hidden state at episode boundaries
6. detach hidden state every `bptt_chunk_steps` steps if chunking is enabled
7. compute PPO losses and update once

Losses:

- clipped policy loss
- clipped value loss
- entropy bonus
- `aux_hidden_info_coef *` (item + ability + tera + moves losses) — auxiliary
- `aux_opp_action_coef *` (a1 + a2 + tera CE) — auxiliary

Defaults:

- `clip_eps = 0.2`
- `value_coef = 0.5`
- `entropy_coef = 0.01`
- `max_grad_norm = 0.5`
- `bptt_chunk_steps = None`

Full BPTT across the rollout is the default; truncation via `bptt_chunk_steps` is an OOM fallback only.

### 5.3 Team Preview Update

Also implemented in `pokerl/training/ppo.py` via `PPOTrainer.update_team_preview(...)`.

This update uses end-of-battle preview examples captured during rollout:

```text
(preview_obs, own_picks, final_reward, opponent_picks)
```

Preview samples are accumulated across `team_preview_update_interval` PPO updates before a gradient step runs. Completed episodes per update are few (tens), so batching several updates together gives the preview REINFORCE loss a usable sample size. The accumulator is capped at `team_preview_max_samples`; oldest samples are dropped first on overflow.

The preview actor uses a dedicated preview value head as its baseline:

```text
advantage = final_reward - preview_value.detach()
policy_loss = -(log_prob * advantage).mean()
value_loss = 0.5 * (preview_value - final_reward)^2
```

The preview loss combines:

- preview policy loss
- preview value loss
- opponent bring BCE auxiliary loss
- opponent ordered lead-pair cross-entropy auxiliary loss

Auxiliary losses:

- bring prediction BCE over opponent chosen 4
- ordered lead-pair cross-entropy over opponent picks `0` and `1`

Those losses update the shared encoder stack and the opponent-preview head through the same optimizer step.

### 5.4 Inference Batchers

There are two batching helpers:

- `TeamPreviewBatcher` batches live policy preview inference for rollout collection
- `_LeagueInferenceService` batches frozen league checkpoint inference for both battle turns and preview

Both exist to avoid many tiny forward passes from separate environment threads.

## 6. League Self-Play

League code lives in `pokerl/training/league.py`.

Current behavior:

- the league stores up to `league_size` frozen checkpoints
- `add_checkpoint(...)` snapshots the current policy to `checkpoint_dir/league/step_<global_step>.pt`
- opponents are sampled with a `70%` bias to the recent half and `30%` to the older half
- `_LeagueInferenceService` groups requests by `(checkpoint_path, kind)` before running batched inference

`LeaguePlayer` specifics:

- keeps its own recurrent hidden state across battle turns
- stages hidden updates until the battle observation advances
- prepares preview inference ahead of time when possible
- converts sampled actions into legal Showdown orders through `action_to_order(...)`

## 7. Default Configuration Reference

Source: `pokerl/config.py`.

### 7.1 ModelConfig

| Parameter | Default |
|---|---|
| `species_dim` | `64` |
| `move_dim` | `32` |
| `ability_dim` | `32` |
| `item_dim` | `32` |
| `type_dim` | `16` |
| `pokemon_hidden` | `256` |
| `pokemon_out` | `128` |
| `team_attention_heads` | `4` |
| `team_attention_layers` | `2` |
| `field_out` | `64` |
| `memory_type` | `"lstm"` |
| `memory_hidden` | `256` |
| `action_hidden` | `256` |
| `value_hidden` | `256` |
| `pick_size` | `4` |
| `init_embeddings_from_stats` | `True` |

### 7.2 EnvConfig

| Parameter | Default |
|---|---|
| `battle_format` | `"gen9randomdoublesbattle"` |
| `server_url` | `"localhost:8000"` |
| `num_envs` | `16` |
| `team_size` | `6` |
| `pick_size` | `4` |

### 7.3 TrainingConfig

| Parameter | Default |
|---|---|
| `lr` | `2.5e-4` |
| `gamma` | `0.99` |
| `gae_lambda` | `0.95` |
| `clip_eps` | `0.2` |
| `entropy_coef` | `0.01` |
| `value_coef` | `0.5` |
| `max_grad_norm` | `0.5` |
| `team_preview_bring_coef` | `0.1` |
| `team_preview_lead_coef` | `0.05` |
| `rollout_steps` | `128` |
| `num_epochs` | `4` |
| `num_minibatches` | `4` |
| `bptt_chunk_steps` | `None` |
| `total_timesteps` | `10_000_000` |
| `log_interval` | `10` |
| `save_interval` | `100` |
| `checkpoint_dir` | `"checkpoints"` |
| `league_size` | `12` |
| `league_update_interval` | `10` |
| `league_coalesce_ms` | `10.0` |

## 8. Current Limitations

These are real limitations of the current implementation, not leftovers from an older architecture.

### 8.1 Item embeddings are still mostly hash-bucketed

Most competitive items do not get their own dedicated embedding entry.

### 8.2 Opponent PP is still unobserved

Opponent PP fractions remain all-zero in the continuous feature block.

### 8.3 Reward is purely terminal

The battle reward function is sparse: win `+1`, loss `-1`, otherwise `0`.

### 8.4 Team preview still assumes a canonical bench tail

Bring choice and lead ordering are now separated, but the two non-lead brought Pokemon are still emitted in canonical team-slot order rather than from a separate bench-order policy.

### 8.5 Opponent belief modeling is supervised only, not yet consumed

The `HiddenInfoHead` and `OpponentActionHead` train the shared encoder to
predict unrevealed opponent attributes and the opponent's next action, but
the policy does not yet consume those distributions. A future change (Option
B / commit 5) will feed the opponent-action distribution into the action
head so the policy best-responds to the *distribution*, not the argmax.

### 8.6 Long-range gradients are bounded by rollout length

The default is full BPTT across the rollout (`bptt_chunk_steps = None`), so gradients flow across an entire rollout chunk. Battles that span multiple rollouts still see a detach at the rollout boundary, and `bptt_chunk_steps` remains available as an OOM fallback.

### 8.7 Simulator latency remains a throughput bottleneck

Even with batching and multiple environments, Showdown battle latency dominates wall-clock training speed.

### 8.8 League diversity is bounded by a small rolling window

With the default `league_size = 12`, the opponent distribution can still become narrow if successive checkpoints are too similar.

## 9. One-Screen Summary

```text
Battle turn:
  obs
    -> SpatialEncoder
    -> Memory
    -> ActionHead:
         a1
         -> a2 | a1
         -> tera | a1, a2
    -> ValueHead(memory_out, own_bench, opp_bench)

Team preview:
  synthetic obs
    -> SpatialEncoder
    -> Memory
    -> TeamPreviewHead (bring set -> lead pair)
    -> OpponentPreviewHead (auxiliary)
    -> PreviewValueHead

Training:
  rollout collection
    -> GAE
    -> PPO update on battle turns
    -> REINFORCE update on preview decisions
    -> periodic league checkpoint snapshotting
```
