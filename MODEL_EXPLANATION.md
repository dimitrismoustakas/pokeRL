# pokeRL: Model Architecture & Training Semantics

A detailed reference for how the model works, what decisions were made, and where the system has room to improve. Written both as an onboarding guide for someone new to the codebase and as an analytical reference for ongoing development.

The authoritative sources are:

- [pokerl/model/agent.py](pokerl/model/agent.py)
- [pokerl/model/encoders.py](pokerl/model/encoders.py)
- [pokerl/model/memory.py](pokerl/model/memory.py)
- [pokerl/model/heads.py](pokerl/model/heads.py)
- [pokerl/env/observation.py](pokerl/env/observation.py)
- [pokerl/env/actions.py](pokerl/env/actions.py)
- [pokerl/env/rewards.py](pokerl/env/rewards.py)
- [pokerl/training/rollout.py](pokerl/training/rollout.py)
- [pokerl/training/ppo.py](pokerl/training/ppo.py)
- [pokerl/training/preview.py](pokerl/training/preview.py)
- [pokerl/training/league.py](pokerl/training/league.py)
- [pokerl/config.py](pokerl/config.py)

If this document and the code ever disagree, the code is authoritative.

---

## Table of Contents

1. [High-Level Overview](#1-high-level-overview)
2. [State Representation: What the Model Sees](#2-state-representation-what-the-model-sees)
3. [Model Architecture](#3-model-architecture)
4. [Action Space](#4-action-space)
5. [Training Algorithm](#5-training-algorithm)
6. [Reward Structure](#6-reward-structure)
7. [Team Preview](#7-team-preview)
8. [League Self-Play](#8-league-self-play)
9. [Configuration Reference](#9-configuration-reference)
10. [Known Limitations & Improvement Opportunities](#10-known-limitations--improvement-opportunities)
11. [The Training Loop in One Picture](#11-the-training-loop-in-one-picture)

---

## 1. High-Level Overview

The agent plays **VGC doubles battles** (and other doubles formats) via **Proximal Policy Optimization (PPO)** against opponents sampled from a **league of past checkpoints**. The environment is a real [Pokemon Showdown](https://pokemonshowdown.com/) server running locally, bridged into Gymnasium through [poke-env](https://github.com/hsahovic/poke-env).

The core loop:

```
Reset → Observe → Spatial Encoder → Memory → Action + Value → Step → Reward → PPO update
```

**Key properties:**

- **Doubles battle**: two active pokemon per side simultaneously.
- **Sparse reward**: only `+1` win / `-1` loss at battle end; no intermediate shaping.
- **Recurrent actor-critic**: both the policy and the value function see an LSTM-carried battle history.
- **League self-play**: trained against a rotating pool of frozen past checkpoints.
- **Three autoregressive decisions per turn**: slot-1 action → slot-2 action conditioned on slot 1 → terastallization choice conditioned on both.
- **Separate team preview stage**: picks the 4-of-6 bring set first, then picks an ordered lead pair from those 4. Trained with REINFORCE plus opponent-model auxiliaries.

### Battle-turn pipeline

```text
observation
  → SpatialEncoder (pokemon → teams → field)
  → Memory (LSTM or identity)
  → ActionHead
       slot-1 action
       → slot-2 action | slot-1 action
       → tera choice  | slot-1, slot-2
  → ValueHead
```

### Team preview pipeline

```text
synthetic preview observation
  → SpatialEncoder (same weights as battle)
  → Memory (hidden=None at preview time)
  → TeamPreviewHead
       canonical bring set (4 of 6)
       → ordered lead pair (2 of the 4)
  → OpponentPreviewHead (auxiliary bring + lead prediction)
  → PreviewValueHead (baseline for REINFORCE)
```

The spatial encoder, memory module, and their weights are shared between battle turns and team preview — only the heads differ.

During PPO updates, two additional auxiliary heads provide dense supervised signal without changing rewards or inference-time action selection: `HiddenInfoHead` predicts eventually revealed opponent item / ability / tera type / move set from `opp_reps`, and `OpponentActionHead` predicts the opponent's observed `(a1, a2, tera)` from `memory_out` plus both sides' active reps.

---

## 2. State Representation: What the Model Sees

Observation building lives in [pokerl/env/observation.py](pokerl/env/observation.py). Each call to `ObservationBuilder.build(battle)` produces a dict of numpy arrays that get converted to tensors on the training device.

### 2.1 Pokemon Features

Each side contributes up to 6 pokemon. For each pokemon the encoder receives two kinds of features:

#### Categorical features (looked up as learned embeddings)

| Feature | Shape | Notes |
|---|---|---|
| `species` | `(6,)` | species vocab index |
| `moves` | `(6, 4)` | revealed move slots; unknown → `UNKNOWN_IDX` |
| `last_move` | `(6,)` | most recent revealed or used move |
| `ability` | `(6,)` | ability vocab index |
| `item` | `(6,)` | named-item vocab + hash-bucketed fallback |
| `types` | `(6, 3)` | `[type_1, type_2, tera_type]` |

#### Continuous features — 44 dimensions per pokemon

Defined by `CONTINUOUS_DIM` in [observation.py:85](pokerl/env/observation.py#L85).

| Group | Dims | Details |
|---|---|---|
| HP fraction | 1 | `current_hp / max_hp` |
| Status one-hot | 7 | `{none, BRN, FRZ, PAR, PSN, SLP, TOX}` |
| Stat boosts | 7 | `{atk, def, spa, spd, spe, accuracy, evasion}` divided by 6 |
| Volatile effects | 16 | confusion, substitute, leech seed, taunt, encore, torment, disable, yawn, perish song, protect, endure, focus energy, attract, curse, nightmare, flash fire |
| Move PP fractions | 4 | `current_pp / max_pp` per slot — **own side only**; opponent entries stay zero |
| Extra | 9 | protect counter, status counter, first turn, revealed, known move count, ability known, item known, tera known, is terastallized |

**Important asymmetry:** PP is populated for own pokemon but zero for opponent pokemon. The model learns this as an implicit "is own" signal rather than having a separate flag. The cost is that it cannot track opponent PP depletion (e.g. knowing a Fake Out can't be used again). See [§10.2](#102-opponent-pp-features-always-zero).

### 2.2 Field Features — 36 dimensions

`FieldEncoder.INPUT_DIM = 36`, defined in [encoders.py:112-122](pokerl/model/encoders.py#L112-L122).

| Group | Dims | Details |
|---|---|---|
| Weather | 10 | none flag + 8 weather types + normalized weather age |
| Terrain | 6 | none flag + 4 terrain types + normalized terrain age |
| Own side conditions | 8 | stealth rock, spikes (0–3), toxic spikes (0–2), sticky web, reflect age, light screen age, aurora veil age, tailwind age |
| Opp side conditions | 8 | same layout as own side |
| Global | 4 | trick room age, own tera used, opp tera used, normalized turn |

### 2.3 Team Metadata (passed separately)

These tensors are passed alongside the per-pokemon blocks rather than inside them, because they describe *positions* rather than individual pokemon:

- `own_roles`, `opp_roles`: `(6,)`, values in `{ACTIVE_1, ACTIVE_2, BENCH}`.
- `own_mask`, `opp_mask`: `(6,)`, padding mask where `True` = unused slot.
- `own_active_indices`, `opp_active_indices`: `(2,)`, indices of each active pokemon within the ordered team list.
- `own_active_present`, `opp_active_present`: `(2,)`, whether each active slot actually exists (handles faint / forced switch gaps).

Team ordering places active pokemon first but keeps stable team-slot identity so switches resolve correctly. Missing actives are gathered and then zeroed out by `SpatialEncoder._gather_slots`.

### 2.4 Item Vocabulary

Item indexing lives in [pokerl/data.py](pokerl/data.py). It is a two-tier system:

1. **Named tier** — items harvested from the Showdown Pokedex `requiredItem` / `requiredItems` fields (form-locked items like Giratina-Origin's Griseous Orb, Arceus types, etc.).
2. **Hash bucket tier** — everything else lands in `blake2b(item_id) % num_hash_buckets`.

Most competitive held items (Choice Band, Assault Vest, Leftovers, Life Orb, Safety Goggles…) fall into hash buckets today. Multiple items can collide into one embedding, and there is no stat-based init for items. See [§10.1](#101-item-vocabulary-is-mostly-hash-bucketed).

---

## 3. Model Architecture

Entry point: [pokerl/model/agent.py](pokerl/model/agent.py). `PokeAgent` composes:

- `spatial` — `SpatialEncoder` (pokemon → teams → field → single context + per-pokemon reps)
- `memory` — `LSTMMemory` or `IdentityMemory`
- `action_head` — `ActionHead` (autoregressive slot-1, slot-2, tera)
- `team_preview_head` — `TeamPreviewHead`
- `opponent_preview_head` — `OpponentPreviewHead` (auxiliary only)
- `hidden_info_head` — `HiddenInfoHead` (auxiliary hidden-info prediction only)
- `opponent_action_head` — `OpponentActionHead` (auxiliary opponent-action prediction only)
- `value_head` — `ValueHead` for battle turns
- `preview_value_head` — separate `ValueHead` used as the REINFORCE baseline at preview

If `ModelConfig.init_embeddings_from_stats` is true (the default), `GameData` initializes species and move embeddings from normalized base stats before the first forward pass — see [§3.11](#311-stat-based-embedding-initialization).

### 3.1 PokemonEncoder

[pokerl/model/encoders.py:7-75](pokerl/model/encoders.py#L7-L75).

Shared weights encode each individual pokemon into a fixed-size vector. The same encoder is applied to all 12 pokemon (own 6 + opponent 6).

**Input per pokemon — 380 dims:**

| Component | Source | Dims |
|---|---|---|
| Species embedding | `species_to_idx` lookup | 64 |
| Move embeddings (×4) | `move_to_idx` lookup, concatenated | 128 |
| Last-move embedding | `move_to_idx` lookup | 32 |
| Ability embedding | `ability_to_idx` lookup | 32 |
| Item embedding | named + hash bucket lookup | 32 |
| Type embeddings (×3) | `type_to_idx` lookup, concatenated | 48 |
| Continuous features | raw float vector | 44 |
| **Total input** | | **380** |

**MLP:**

```
380 → Linear(256) → ReLU → Linear(128) → ReLU   →   (batch, 128)
```

### 3.2 TeamEncoder

[pokerl/model/encoders.py:78-109](pokerl/model/encoders.py#L78-L109).

Two separate instances — `team_own` and `team_opp` — each apply order-invariant self-attention over the 6-pokemon set.

**Input:**

- Per-pokemon reps: `(batch, 6, 128)` from `PokemonEncoder`.
- Role indices: `(batch, 6)` — role embedding (`ACTIVE_1`, `ACTIVE_2`, `BENCH`) added to each rep. Role embedding width matches `pokemon_dim = 128`.

**Architecture:**

```
(pokemon_rep + role_emb) → TransformerEncoder(
    d_model=128, nhead=4, num_layers=2,
    dim_feedforward=256, dropout=0.0, norm_first=True
) → (batch, 6, 128)
```

Attention is **permutation-equivariant** with respect to team order — the model sees roles but has no slot-position embedding. Padding slots are masked out. This matters because VGC team order is partially arbitrary: once the policy has learned to read role, nothing should depend on whether Incineroar is team slot 3 or 5.

### 3.3 FieldEncoder

[pokerl/model/encoders.py:112-133](pokerl/model/encoders.py#L112-L133).

```
36 → Linear(64) → ReLU   →   (batch, 64)
```

### 3.4 SpatialEncoder

[pokerl/model/encoders.py:136-253](pokerl/model/encoders.py#L136-L253). Combines the three encoders above and exposes five tensors:

| Output | Shape | Description |
|---|---|---|
| `context` | `(batch, 320)` | `mean(own_reps) ∥ mean(opp_reps) ∥ field` |
| `active_reps` | `(batch, 2, 128)` | own two active pokemon |
| `own_reps` | `(batch, 6, 128)` | full own team after attention |
| `opp_reps` | `(batch, 6, 128)` | full opponent team after attention |
| `opp_active_reps` | `(batch, 2, 128)` | opponent two active pokemon |

Notes:

- Team means are **mask-aware** — padded slots are excluded from both the sum and the denominator.
- Active reps are gathered by stable team index, then multiplied by `*_active_present` so missing actives zero out.
- `context` is purely spatial: no temporal state has been injected yet.

### 3.5 Memory Module

[pokerl/model/memory.py](pokerl/model/memory.py).

Two interchangeable implementations selected by `ModelConfig.memory_type`:

**`IdentityMemory`** — feedforward baseline for ablation. Single `Linear(320, 256)` projection, no hidden state, each turn independent.

**`LSTMMemory`** (default):

- `nn.LSTM(input_dim=320, hidden_dim=256, batch_first=True)`.
- Accepts either `(batch, 320)` single-step or `(batch, seq, 320)` sequence input.
- Returns `(memory_out, (h, c))`.
- **Does not** reset hidden state on `done` and **does not** detach for BPTT by itself — both are handled by the rollout loop and PPO trainer. This keeps `LSTMMemory` a pure recurrent cell, with episode-boundary semantics owned by the caller that knows when episodes end.

The LSTM is what gives the policy (and critic — see below) their memory of the battle: HP trajectories, revealed moves, field-control flow. Without it the model would need to infer all of that purely from the current frame.

### 3.6 MaskedSetAttention (building block)

[pokerl/model/heads.py:21-62](pokerl/model/heads.py#L21-L62). A learned attention-pooling primitive reused inside `ActionHead`, `ValueHead`, and `TeamPreviewHead`.

```
query (batch, q_dim) × items (batch, N, d) [mask (batch, N)] → (batch, d)
```

- Separate `query_proj`, `key_proj`, and `value_proj` linear layers.
- Scaled dot-product scores over the items.
- Mask applied before softmax (`-1e8` fill), then weights are **renormalized** after masking to handle rows where *all* items are invalid (in that case weights stay zero and the output collapses to zero rather than NaN).

### 3.7 ActionHead

[pokerl/model/heads.py:65-750](pokerl/model/heads.py#L65-L750). This is the most intricate component — it makes three conditioned decisions per turn.

#### Per-slot action universe

`ACTIONS_PER_SLOT = 27`, defined in [actions.py:16-44](pokerl/env/actions.py#L16-L44):

```
 0..19  move_i * 5 + target_j       (4 moves × 5 targets)
20..25  switch to own_team_slot_k   (stable team slot 0..5)
26      pass                        (slot has no action this turn)
```

Target encoding:

```
0: opposing slot 1    3: self
1: opposing slot 2    4: no explicit target (field / all-target moves)
2: ally (partner)
```

Tera is a **separate third categorical**:

```
0: no tera
1: tera slot 0
2: tera slot 1
```

This is the major structural change from earlier versions of the agent, where tera was folded into the move action IDs (doubling the space to 47 per slot). Splitting it out means fewer total actions, and the tera decision gets its own conditioning on both chosen slot actions.

#### Static action features

For each of the 27 per-slot actions the head builds a static `action_hidden`-dim embedding by summing:

- action ID embedding
- action kind embedding (`move` / `switch` / `pass`)
- move-slot embedding
- target embedding
- switch-slot embedding

These are pre-computed buffers indexed per action — they don't depend on the current state.

#### Dynamic action entity features

Each candidate action also gets a *dynamic* 128-dim entity representation from the current battle:

- move actions gather a target-side entity from `[opp0, opp1, partner, self, zero, zero]`
- switch actions gather the destination team slot from `own_team_reps`
- pass resolves to a zero vector

Projected through `Linear(128) → ReLU → (hidden_dim)`.

#### Explicit matchup features — 22 dims per candidate

[heads.py:418-605](pokerl/model/heads.py#L418-L605). For each of the 27 candidate actions the head computes hand-aligned numerical features from live observation + move/type metadata stored in `GameData`:

- action-kind one-hots (`is_move`, `is_switch`, `is_pass`)
- move base power, accuracy, priority
- move category one-hots (status / physical / special)
- current STAB multiplier, and post-tera STAB gain
- current target effectiveness (0 / 0–1 / 1+ bucket one-hots, plus scaled value)
- best and mean foe effectiveness across the two live opponents
- current expected power proxy (`power * stab * eff / 4`) and tera power gain proxy
- number of live foes (normalized)
- spread-hit and ally-hit flags for targetless moves

These features give the policy a shortcut to "is this move super-effective into this target right now?" without having to recover that from raw embeddings. They are especially valuable with sparse rewards, where learning these interactions from scratch is slow.

#### Slot 1

```
base_query1 = Linear(context ⊕ rep1 ⊕ rep2 ⊕ opp1 ⊕ opp2, 256) → ReLU
reserve_summary1 = MaskedSetAttention(base_query1, opp_team_reps, opp_bench_mask)
query1 = Linear(base_query1 ⊕ reserve_summary1, 256) → ReLU
```

Note: `context` here is the **LSTM output** (`memory_out`, 256 dims), not the raw spatial context. The reserve mask excludes padded slots and non-bench roles, so this attends only over the opponent's benched pokemon — a summary of "what else might they switch in."

Scoring each of the 27 candidates:

```
score = MLP(query1 ⊕ static_action_features ⊕ dynamic_entity_features ⊕ matchup_emb)
      = Linear(1024) → ReLU → Linear(1)
```

Illegal actions from `masks["joint"][0]` are set to `-1e8` before the softmax, so they become exact zero probability. This is a hard constraint, not a soft one.

#### Slot 2

Same pipeline, but the query is augmented with slot 1's chosen action embedding:

```
base_query2 = Linear(context ⊕ rep2 ⊕ rep1 ⊕ opp1 ⊕ opp2 ⊕ a1_emb, 256) → ReLU
reserve_summary2 = MaskedSetAttention(base_query2, opp_team_reps, opp_bench_mask)
query2 = Linear(base_query2 ⊕ reserve_summary2, 256) → ReLU
```

Legality comes from row `a1 + 1` of the joint mask (see [§4.2](#42-action-masking)). Conditioning on `a1` lets slot 2 coordinate with slot 1 — e.g. don't protect with both, don't both target the same spread-vulnerable opponent.

#### Tera decision

After both slot actions are chosen, a third categorical picks `{no tera, tera slot 0, tera slot 1}`:

```
tera_query = Linear(
    context ⊕ rep1 ⊕ rep2 ⊕ opp1 ⊕ opp2 ⊕ a1_emb ⊕ a2_emb
        ⊕ chosen_matchup_1 ⊕ chosen_matchup_2,
    256
) → ReLU
tera_logits = Linear(256) → ReLU → Linear(3)
```

The tera mask is **derived internally** inside `ActionHead.forward(...)` from `masks["can_tera"]` and whether each slot action is a move — so the policy does not consume a precomputed `(a1, a2, tera)` legality tensor during normal inference. (The `build_tera_action_mask` helper in `actions.py` still exists for supervised analyses.)

#### Tera exploration during rollout

[heads.py](pokerl/model/heads.py) and [config.py](pokerl/config.py). The `tera_exploration_prob` parameter enables uniform-over-valid exploration during rollout collection. With probability `epsilon`, on any batch item that has ≥ 2 valid tera options, the policy's sampled tera is replaced with a uniform draw over the valid set.

**Why:** once the policy tera's on turn 1, `can_tera` is false for every later turn in the episode. Without exploration the policy never observes the counterfactual trajectory where it saved tera, which traps it in a local optimum where tera fires immediately regardless of situation.

Uniform-over-valid exploration is complemented by targeted early-turn delay. If `tera_force_delay_prob > 0`, the rollout path can convert a sampled tera action into `no tera` through `tera_force_delay_max_turn`. This does not change the model architecture or reward; it directly creates delayed-tera trajectories so the value function and policy can compare "tera now" against "save tera".

The stored log-prob for the taken action is always computed under the **policy** distribution (not the mixed behavior distribution), so PPO's importance ratio is slightly off-policy. For small, decayed probabilities this is kept within the clip window. Exploration is only active on the rollout path (`actions is None`); the PPO update path passes stored actions and is unaffected.

Epsilon decays linearly from `tera_exploration_start = 0.15` to `tera_exploration_end = 0.02` across `tera_exploration_decay_updates = 2000` PPO updates.

Forced delay decays linearly from `tera_force_delay_start = 0.40` to `tera_force_delay_end = 0.03` across `tera_force_delay_decay_updates = 3000` PPO updates, and applies only through `tera_force_delay_max_turn = 3`.

#### ActionHead outputs

```
{
  "actions":  (batch, 3),   # [a1, a2, tera_choice]
  "log_prob": (batch,),     # log π(a1) + log π(a2 | a1) + log π(tera | a1, a2)
  "entropy":  (batch,),     # sum of the three entropies
}
```

### 3.8 ValueHead

[pokerl/model/heads.py:939-962](pokerl/model/heads.py#L939-L962). Estimates `V(s)` for GAE.

The critic is **recurrent**: `PokeAgent.forward` overwrites `context` with `memory_out` before passing it to the value head (see [agent.py:127-143](pokerl/model/agent.py#L127-L143)). So the value function sees the same temporal summary the policy does — this is a recent change; in an earlier version the critic received the raw spatial context instead.

**Structure:**

1. `MaskedSetAttention(memory_out, own_team_reps, own_bench_mask)` → own bench summary `(batch, 128)`.
2. `MaskedSetAttention(memory_out, opp_team_reps, opp_bench_mask)` → opponent bench summary `(batch, 128)`.
3. MLP over `[memory_out (256) ⊕ own_bench (128) ⊕ opp_bench (128)]`:

```
512 → Linear(256) → ReLU → Linear(1)
```

The bench summaries let the critic consider team *depth* — knowing the opponent still has three healthy benched sweepers versus one weakened one matters for win probability in doubles.

If a side has no valid bench targets, the corresponding attention summary collapses to zero (see the mask-renormalization note in [§3.6](#36-maskedsetattention-building-block)).

### 3.9 TeamPreviewHead

[pokerl/model/heads.py:753-894](pokerl/model/heads.py#L753-L894). This head separates **bring selection** from **lead ordering** — an important split that the previous version of the architecture did not make.

#### Stage 1 — bring set (canonical, autoregressive)

Picks 4 of the 6 own pokemon in **canonical increasing team-slot order**:

1. accumulate the sum of already-picked reps: `picked_sum`
2. project `[memory_out ⊕ picked_sum] → context_dim`
3. attend over opponent reps with `MaskedSetAttention`
4. score each still-available own pokemon from `[pokemon_rep ⊕ projected_context ⊕ opponent_summary]`
5. mask: only positions past the last pick are legal, and only if enough later positions remain to finish the bring set
6. sample / argmax → next pick

Repeat for 4 picks. Because the mask enforces increasing indices, the head learns to output a *set* rather than an order — the order at this stage is not meaningful.

#### Stage 2 — ordered lead pair

1. summarize the chosen bring set: `bring_summary = picked_sum / pick_size`
2. project `[memory_out ⊕ bring_summary] → context_dim`
3. attend over opponent reps
4. score all `team_size × team_size` ordered pairs from `[left_rep ⊕ right_rep ⊕ bring_summary ⊕ opponent_summary ⊕ projected_context]`
5. mask out self-pairs, unbrought pokemon, and padding
6. sample / argmax over the flattened `team_size²` logits

#### Final Showdown preview order

- Picks 0 and 1 are the chosen ordered lead pair.
- The remaining brought pokemon are appended in canonical team-slot order. This is a structural limitation — there is no separate "bench order" policy. See [§10.6](#106-bench-order-still-canonical).

**Outputs:**

```
bring_picks:     (batch, 4)   canonical bring set
lead_pair:       (batch, 2)   ordered lead pair
picks:           (batch, 4)   final Showdown order = lead_pair ⊕ bench tail
bring_log_prob:  (batch,)
lead_log_prob:   (batch,)
log_prob:        (batch,)     bring + lead
entropy:         (batch,)     bring + lead
```

### 3.10 OpponentPreviewHead

[pokerl/model/heads.py:897-936](pokerl/model/heads.py#L897-L936). Auxiliary head — used **only for training signal**, never consumed at inference. It predicts what the opponent will bring and who they will lead with.

- `opponent_bring_logits`: `(batch, 6)` independent bring scores (sigmoid-trained via BCE).
- `opponent_lead_pair_logits`: `(batch, 6 × 6)` ordered lead-pair scores (cross-entropy trained). Self-pairs and padded slots are masked.

The gradients from the auxiliary losses flow back through the shared encoder stack, shaping `opp_reps` to encode information relevant to opponent behavior. That helps the downstream action head as well.

Two battle-turn auxiliary heads also live in [pokerl/model/heads.py](pokerl/model/heads.py), but they do not get separate architecture sections because they are training-only heads rather than action-path components:

- `HiddenInfoHead` maps `opp_reps` to item, ability, tera-type, and move-set logits for each opponent pokemon. Labels come from the post-step terminal observation when an episode completes and are copied backward across that episode's timesteps in the rollout buffer.
- `OpponentActionHead` maps `[memory_out, opp_active_0, opp_active_1, own_active_0, own_active_1]` to three categorical predictions for the opponent's observed `a1`, `a2`, and tera choice. Non-league opponents have no labels and are masked out.

### 3.11 Stat-based embedding initialization

When `init_embeddings_from_stats=True` (the default), [agent.py:99-101](pokerl/model/agent.py#L99-L101) asks `GameData` to initialize two embedding tables from base stats rather than random:

- **Species**: `(HP, Atk, Def, SpA, SpD, Spe)` normalized by ÷255 → linear projection → embedding.
- **Moves**: `(power/250, accuracy/100, pp/64)` → linear projection → embedding.

This gives the model a warm start that encodes known structure — fast pokemon begin with high-Spe embeddings, strong attackers begin with high-Atk embeddings, and so on. In practice it noticeably improves early training.

### 3.12 Complete forward pass

Battle turn ([agent.py:103-149](pokerl/model/agent.py#L103-L149)):

```
obs dict
  → PokemonEncoder  (all 12 pokemon, shared weights)
  → TeamEncoder     (own team) + TeamEncoder (opponent team)
  → FieldEncoder
  → SpatialEncoder output:
      { context (320), active_reps (2×128),
        own_reps (6×128), opp_reps (6×128), opp_active_reps (2×128) }
  → LSTMMemory:     context (320) → memory_out (256), (h, c)
  → ActionHead:     (memory_out, active_reps, own_reps, opp_reps, opp_bench_mask,
                     opp_active_reps, masks, battle_obs,
                     tera_exploration_prob, tera_force_delay_prob)
                    → { actions (3), log_prob, entropy }
  → ValueHead:      (memory_out, own_reps, opp_reps, own_bench_mask, opp_bench_mask)
                    → value
```

Team preview ([agent.py:151-189](pokerl/model/agent.py#L151-L189)):

```
obs dict
  → SpatialEncoder
  → LSTMMemory:     context → memory_out, (h, c=None at preview time)
  → TeamPreviewHead      → bring_picks, lead_pair, picks, log_prob, entropy
  → OpponentPreviewHead  → bring_logits, lead_pair_logits   (aux only)
  → PreviewValueHead     → preview_value                    (baseline)
```

---

## 4. Action Space

### 4.1 Action encoding

Already covered in [§3.7](#37-actionhead). To recap, per active slot there are 27 actions — 20 `move × target` combinations, 6 switches, one pass — and terastallization is a separate 3-way joint decision made after both slot actions. See [actions.py](pokerl/env/actions.py) for the canonical encoding.

### 4.2 Action masking

The policy consumes two mask tensors at every turn (see `build_policy_action_masks` in [actions.py:149-159](pokerl/env/actions.py#L149-L159)):

- `joint`: shape `(ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT) = (28, 27)`
- `can_tera`: shape `(2,)`

Interpretation of `joint`:

- Row `0` (`FIRST_SLOT_MASK_ROW`): legal slot-1 actions.
- Row `1 + a1`: legal slot-2 actions **conditioned on** slot-1 choice `a1`.

This structure enforces constraints that a per-slot mask cannot, such as:

- Both pokemon cannot switch into the same reserve slot.
- If slot 1 picks an ally-target move, slot 2's ally-target space updates to reflect the new situation.
- Pass-only states (e.g. single forced switch) are still representable.

The joint table is built by feeding poke-env's `DoubleBattleOrder.join_orders` back through the action encoding — so legality is *delegated* to poke-env rather than re-derived, which avoids subtle bugs around disabled moves, trapped pokemon, and forced switches.

During inference, illegal actions are filled with `-1e8` before the softmax.

---

## 5. Training Algorithm

Implemented in [pokerl/training/ppo.py](pokerl/training/ppo.py) with rollout storage in [pokerl/training/rollout.py](pokerl/training/rollout.py).

### 5.1 Rollout Collection

`RolloutBuffer` pre-allocates tensors of shape `(rollout_steps=128, num_envs=16, ...)` on the trainer device. At each timestep:

1. All environments step in parallel (poke-env async loop + thread pool).
2. Agent batches all observations → forward pass → `(num_envs, 3)` actions.
3. Actions are sent back to environments; environments step and return `(reward, done)`.
4. `(obs, actions, masks, log_prob, value, reward, done)` are copied into the buffer.
5. Recurrent hidden state for any `done` env is zeroed in the rollout loop (not inside `LSTMMemory` itself).

The buffer also stores the initial hidden state from the start of the rollout so PPO can rebuild the same temporal sequence during update.

### 5.2 GAE

After `rollout_steps` steps, we compute a bootstrap value from `get_value(final_obs, hidden)` and then Generalized Advantage Estimation:

```python
for t in reversed(range(T)):
    next_value = last_value if t == T - 1 else values[t + 1]
    next_non_terminal = (~dones[t]).float()
    delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
    gae = delta + gamma * gae_lambda * next_non_terminal * gae
    advantages[t] = gae
returns = advantages + values
```

With `gamma=0.99` and `gae_lambda=0.95`, this biases the advantage estimate toward actual returns while reducing variance.

### 5.3 PPO Update

`PPOTrainer.update()` runs `num_epochs=4` passes over the rollout, each pass splitting `num_envs` into `num_minibatches=4` env subsets.

**Critical design: minibatches slice *environments*, not timesteps.** Each minibatch contains a subset of envs but *all* `T` timesteps for those envs. This preserves the full temporal sequence so the LSTM can be unrolled correctly across episode boundaries. Slicing timesteps instead would invalidate hidden state at minibatch boundaries and break gradient flow across episode boundaries.

**Per-minibatch update:**

1. **Normalize advantages** across the full rollout once, *before* the update loop (not per minibatch):

   ```python
   adv = (adv - adv.mean()) / (adv.std() + 1e-8)
   ```

2. **Batched spatial encoder pass** over all `T × E_mb` timesteps at once. The heavy attention layers only run once per minibatch — a significant speedup since the spatial encoder dominates forward-pass cost.

3. **Sequential LSTM + head pass**, one timestep at a time:
   - Zero hidden at episode boundaries: `hidden = hidden * (~done).float()`.
   - Optional truncated BPTT: every `bptt_chunk_steps` steps, detach hidden. Default is `None` (full BPTT across the rollout). Truncation is an OOM fallback only — it trades gradient quality for memory.

4. **Losses:**

   **Policy loss** (clipped surrogate):
   ```python
   log_ratio = new_log_prob - old_log_prob
   ratio = exp(log_ratio.clamp(-20, 20))   # prevent float32 overflow in pathological cases
   surr1 = ratio * adv
   surr2 = clamp(ratio, 1 - eps, 1 + eps) * adv
   policy_loss = -min(surr1, surr2).mean()
   ```
   With `clip_eps=0.2`, ratios clamp to `[0.8, 1.2]`.

   **Value loss** (clipped):
   ```python
   v_clipped = old_value + clamp(new_value - old_value, -eps, eps)
   value_loss = 0.5 * max((new_value - returns)**2, (v_clipped - returns)**2).mean()
   ```

   **Entropy bonus:**
   ```python
   entropy_loss = -entropy.mean()
   ```

   **Battle-turn auxiliary losses:**
   - `hidden_info_loss`: masked CE/BCE for opponent item, ability, tera type, and revealed move set using completed-episode labels.
   - `opp_action_loss`: masked CE for the league opponent's observed `(a1, a2, tera)` this turn.

   **Total:** `loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss + aux_hidden_info_coef * hidden_info_loss + aux_opp_action_coef * opp_action_loss`, with `value_coef=0.5`, `entropy_coef=0.01`, and both auxiliary coefficients defaulting to `0.25`.

5. **Gradient update:**
   ```python
   loss.backward()
   clip_grad_norm(params, max_norm=0.5)
   optimizer.step()
   ```
   Adam with `lr=2.5e-4`, `eps=1e-5`.

**Non-finite-gradient guard:** `_apply_update_step` in [ppo.py:35-52](pokerl/training/ppo.py#L35-L52) zeros gradients and skips the `optimizer.step()` if the loss or any gradient is non-finite, or if the clipped grad-norm is not finite. The minibatch is counted but does not poison Adam's momentum / variance state. Rare, but robust against pathological policy shifts.

**Diagnostic:** approximate KL divergence `((ratio - 1) - log_ratio).mean()` — useful for checking whether the policy is updating too aggressively.

### 5.4 Team Preview Update

`PPOTrainer.update_team_preview()` runs REINFORCE-with-baseline for preview decisions, because they happen once per battle (not per turn) and receive the final sparse reward much later than any individual action.

```python
advantage = episode_reward - preview_value.detach()
policy_loss = -(log_prob * advantage).mean()
value_loss = 0.5 * ((preview_value - episode_reward) ** 2).mean()
```

The **preview value head** is a dedicated `ValueHead` instance used only as the REINFORCE baseline. Using the same battle-turn value head would be awkward because it is trained on per-turn state-value, whereas the preview head needs to predict the final return from a synthetic preview state.

**Accumulation:** preview samples are collected across `team_preview_update_interval = 8` PPO updates before a gradient step runs. Completed episodes per update are few (tens), so batching several updates together gives the REINFORCE loss a usable sample size. The accumulator is capped at `team_preview_max_samples = 1024`; oldest samples are dropped on overflow.

**Auxiliary losses:**

- `bring_loss`: binary cross-entropy predicting which 4 of 6 opponent pokemon were actually brought. Masked to valid opponent slots only.
- `lead_loss`: cross-entropy predicting the opponent's ordered lead pair over the flattened `6 × 6` logits.

**Combined loss:**

```
loss = policy_loss
     - entropy_coef * entropy_bonus
     + team_preview_value_coef * value_loss
     + team_preview_bring_coef * bring_loss
     + team_preview_lead_coef * lead_loss
```

With `team_preview_value_coef=0.5`, `team_preview_bring_coef=0.1`, `team_preview_lead_coef=0.05`.

Those losses update the shared encoder stack (and hence `opp_reps`) as well as the opponent-preview head, through the same optimizer step as preview policy + value.

### 5.5 Inference Batchers

Two helpers reduce the cost of per-env tiny forward passes:

- `TeamPreviewBatcher` ([preview.py](pokerl/training/preview.py)) batches live-policy preview inference during rollout collection.
- `_LeagueInferenceService` ([league.py](pokerl/training/league.py)) batches frozen league-checkpoint inference for both battle turns and preview, grouping requests by `(checkpoint_path, kind)`.

Both run in background threads with a small coalesce window so multiple concurrent environment threads can amortize the forward pass.

---

## 6. Reward Structure

[pokerl/env/rewards.py](pokerl/env/rewards.py):

```python
def sparse_reward(battle):
    if battle.won:  return  1.0
    if battle.lost: return -1.0
    return 0.0
```

**Purely sparse terminal rewards.** No intermediate shaping for KOs, damage dealt, HP preserved, or effective moves.

Implications:

- The value function must learn to predict win probability from state alone.
- The LSTM must learn to track relevant dynamics without dense signal.
- Credit assignment is hard: a turn-3 mistake that loses on turn 15 has to be attributed.

This is intentional. Reward shaping introduces bias (a bot that maximizes damage dealt but ignores winning) and the sparse setup forces the agent to actually learn to win. The cost is slower learning — see [§10.4](#104-sparse-reward-is-hard).

---

## 7. Team Preview

Observation side: [pokerl/env/team_preview.py](pokerl/env/team_preview.py) and `build_team_preview_observation` in [observation.py:448-466](pokerl/env/observation.py#L448-L466). Head side: [§3.9](#39-teampreviewhead).

### 7.1 What happens at team preview

At the start of each VGC battle both players see each other's full 6-pokemon roster — species only; moves, items, and abilities are hidden. Each player then:

1. Chooses which **4 of 6** pokemon to bring.
2. Chooses the **order** they appear in — picks 0 and 1 are the leads, picks 2 and 3 are the bench.

### 7.2 The synthetic preview observation

A synthetic `_SyntheticTeamPreviewBattle` is constructed with:

- own pokemon at full HP with known moves / items / abilities
- opponent pokemon at HP = 0 (unrevealed side) and with only species visible
- no active pokemon, no field conditions, turn = 0

This is run through the same `ObservationBuilder` as live battles, so the spatial encoder sees a coherent "battle at turn 0" state.

### 7.3 Why separate bring and lead

The old architecture picked 4 pokemon autoregressively and assumed "pick 1 = lead 1, pick 2 = lead 2, rest = bench." That required the model to implicitly learn that pick order mattered — it couldn't share representation across permutations of the same bring set.

The current head splits the two concerns:

- The **bring stage** outputs a canonical (sorted) set of 4 pokemon — its decision is *which 4*, not in what order.
- The **lead stage** then scores all ordered pairs from those 4 and picks one — its decision is *who leads* given the set.

Gradient-wise this shares structure across equivalent bring sets and lets the lead scorer specialize on matchup reasoning.

---

## 8. League Self-Play

[pokerl/training/league.py](pokerl/training/league.py).

### 8.1 Why self-play

Training against a fixed opponent (random, max-power greedy) produces a policy that exploits that specific opponent's weaknesses. League self-play trains against a distribution of past selves, creating pressure to generalize rather than fit a single opponent policy.

### 8.2 League structure

A rolling pool of up to `league_size = 12` frozen past checkpoints. Every `league_update_interval = 10` PPO updates, `add_checkpoint(...)` snapshots the current policy to `checkpoint_dir/league/step_<global_step>.pt` and appends it to the pool. When the pool is full, the oldest checkpoint is evicted (`deque(maxlen=league_size)`).

**Sampling policy** ([league.py:549-562](pokerl/training/league.py#L549-L562)): 70% chance to sample from the *recent half*, 30% from the *older half*. Balances:

- **Exploitation** — playing against strong, recent opponents improves measured skill.
- **Exploration** — playing against diverse, older opponents avoids exploitation holes where the current policy overfits to the most recent training partner.

### 8.3 Inference architecture

`_LeagueInferenceService` runs in a background thread:

- Multiple parallel `LeaguePlayer` instances (one per concurrent battle) submit inference requests via `Future`.
- Requests are buffered over a `coalesce_window_ms = 10ms` window, then grouped by `(checkpoint_path, kind)` — so all "checkpoint X, battle turn" requests across envs run in a single batched forward pass.
- Responses are dispatched back to the waiting `LeaguePlayer`s.

Without this, each `LeaguePlayer` would drive a per-env forward pass — extremely slow given Showdown's per-battle latency.

### 8.4 LeaguePlayer

Wraps a frozen checkpoint behind a poke-env `Player` interface:

- Maintains its own LSTM hidden state across battle turns, committed only when the live battle state advances (so "did Showdown ask twice for the same turn" doesn't corrupt hidden state).
- Supports **mid-battle checkpoint staging** (`stage_checkpoint`) — the next battle rollout uses the new checkpoint without interrupting battles in flight.
- Pre-computes the team preview inference ahead of time when the opponent's packed team is known, and verifies that the live preview observation matches the prepared one before consuming the result.
- Converts sampled actions into legal Showdown orders through `action_to_order(...)`, which also applies terastallization if the tera choice selected this slot.

---

## 9. Configuration Reference

All hyperparameters live in [pokerl/config.py](pokerl/config.py) as dataclasses.

### 9.1 ModelConfig (defaults)

| Parameter | Default | Effect |
|---|---|---|
| `species_dim` | 64 | species embedding size |
| `move_dim` | 32 | move embedding size |
| `ability_dim` | 32 | ability embedding size |
| `item_dim` | 32 | item embedding size |
| `type_dim` | 16 | type embedding size |
| `pokemon_hidden` | 256 | `PokemonEncoder` MLP hidden size |
| `pokemon_out` | 128 | `PokemonEncoder` output size |
| `team_attention_heads` | 4 | team-encoder attention heads |
| `team_attention_layers` | 2 | team-encoder Transformer depth |
| `field_out` | 64 | `FieldEncoder` output size |
| `memory_type` | `"lstm"` | `"lstm"` or `"identity"` |
| `memory_hidden` | 256 | LSTM hidden size |
| `action_hidden` | 256 | `ActionHead` MLP hidden size |
| `value_hidden` | 256 | `ValueHead` MLP hidden size |
| `pick_size` | 4 | preview: bring N of 6 |
| `init_embeddings_from_stats` | `True` | warm-start species/move embeddings |

### 9.2 EnvConfig (defaults)

| Parameter | Default |
|---|---|
| `battle_format` | `"gen9randomdoublesbattle"` |
| `num_envs` | 16 |
| `team_size` | 6 |
| `pick_size` | 4 |
| `showdown_dir` | `None` |
| `worker_count` | `None` |
| `battles_per_worker` | `None` |
| `runtime_loop_count` | `None` |
| `battle_loop_count` | `None` |
| `reset_concurrency` | `None` |
| `opponent_group` | 0 |

### 9.3 TrainingConfig (defaults)

| Parameter | Default | Effect |
|---|---|---|
| `lr` | 2.5e-4 | Adam learning rate |
| `gamma` | 0.99 | discount factor |
| `gae_lambda` | 0.95 | GAE smoothing parameter |
| `clip_eps` | 0.2 | PPO clipping range |
| `entropy_coef` | 0.01 | entropy bonus weight |
| `value_coef` | 0.5 | value loss weight |
| `max_grad_norm` | 0.5 | gradient clipping |
| `team_preview_value_coef` | 0.5 | preview value-loss weight |
| `team_preview_bring_coef` | 0.1 | bring-prediction aux weight |
| `team_preview_lead_coef` | 0.05 | lead-prediction aux weight |
| `team_preview_update_interval` | 8 | flush preview accumulator every N PPO updates |
| `team_preview_max_samples` | 1024 | accumulator cap (oldest dropped first) |
| `aux_hidden_info_coef` | 0.25 | hidden-info prediction aux weight |
| `aux_opp_action_coef` | 0.25 | opponent-action prediction aux weight |
| `rollout_steps` | 128 | steps per env before update |
| `num_epochs` | 4 | PPO epochs per rollout |
| `num_minibatches` | 4 | env minibatches per epoch |
| `bptt_chunk_steps` | `None` | truncated BPTT chunk size (`None` = full) |
| `tera_exploration_start` | 0.15 | initial tera exploration ε |
| `tera_exploration_end` | 0.02 | final tera exploration ε |
| `tera_exploration_decay_updates` | 2000 | linear decay horizon (updates) |
| `tera_force_delay_start` | 0.40 | initial probability of converting early sampled tera to no-tera |
| `tera_force_delay_end` | 0.03 | final early-turn force-delay probability |
| `tera_force_delay_decay_updates` | 3000 | linear force-delay decay horizon (updates) |
| `tera_force_delay_max_turn` | 3 | latest battle turn eligible for forced tera delay |
| `total_timesteps` | 10M | training budget |
| `log_interval` | 10 | console log every N updates |
| `save_interval` | 100 | save checkpoint every N updates |
| `checkpoint_dir` | `"checkpoints"` | where to write checkpoints |
| `league_size` | 12 | max past checkpoints |
| `league_update_interval` | 10 | save to league every N updates |
| `league_coalesce_ms` | 10.0 | inference coalescing window (ms) |

---

## 10. Known Limitations & Improvement Opportunities

These are real limitations of the current implementation, not leftovers from an older architecture.

### 10.1 Item vocabulary is mostly hash-bucketed

The named-item tier (`GameData._known_item_ids`) only captures items from the Showdown Pokedex's `requiredItem` / `requiredItems` fields — essentially form-locked items like Griseous Orb or Adamant Orb. Competitive items (Choice Band, Assault Vest, Leftovers, Life Orb, Safety Goggles, Covert Cloak, Clear Amulet…) fall into hash buckets.

Hash collisions mean multiple items share an embedding. There is also no stat-based init for items.

**Fix direction:** build a competitive item vocabulary from VGC usage data (or the full Showdown items DB), and add stat-like features for items (power-boost %, residual %, speed mod, type-immunity) to enable a stat-based init similar to species / moves.

### 10.2 Opponent PP features always zero

The 4 PP dimensions in the continuous feature vector are zero for all opponent pokemon. The model learns "zero PP = opponent," but it can't track opponent PP depletion (e.g. knowing that Fake Out can't fire again, or that a Choice Band user is one Struggle away).

**Fix direction:** give opponent pokemon their own `continuous_dim` without PP, and separate the encodings — the model then cannot confuse "no PP info" with "full PP." Alternatively, expose a PP-inferred signal derived from move use counts in the battle log.

### 10.3 Sparse reward is hard

+1/-1 terminal rewards with no intermediate shaping means:

- Slow credit assignment across 15–20 turn battles.
- The value function learns slowly — no dense signal.
- Team preview decisions are especially hard to learn (very delayed reward).

**Fix direction:** a learned **win-probability shaper**. Train a separate win-prob predictor on game states (supervised on final outcomes), and use its per-turn delta as an additional reward component. Self-supervised and much more neutral to reward hacking than hand-crafted shaping.

### 10.4 Long-range gradients are bounded by rollout length

Default is full BPTT across the rollout (`bptt_chunk_steps=None`), so gradients flow across an entire 128-step chunk. Battles that span multiple rollouts still see a detach at the rollout boundary. `bptt_chunk_steps` remains available as an OOM fallback if the rollout length or model size grows.

### 10.5 Opponent belief modeling is supervised only, not yet consumed

The network sees revealed moves / items / abilities as they appear. `HiddenInfoHead` and `OpponentActionHead` now train the shared encoder to predict eventually revealed opponent attributes and the opponent's current action, but the policy does not yet consume those distributions directly.

**Fix direction:** feed the opponent-action and hidden-info distributions back into the action head so the policy best-responds to the distribution, not just to the raw recurrent state.

### 10.6 Bench order still canonical

Bring-choice and lead ordering are now separated, but the two non-lead brought pokemon are still emitted in canonical team-slot order rather than from a separate bench-order policy. Bench order rarely matters, but it is not zero-impact — it determines which pokemon faces an opponent's lead when a lead faints.

### 10.7 Simulator latency remains a throughput bottleneck

Even with batching and multiple environments, Showdown battle latency dominates wall-clock training speed. Per-battle latency is on the order of minutes, so with sparse rewards the number of reward signals per wall-clock hour is limited.

**Mitigations already in place:**

- Batched inference across env threads (`TeamPreviewBatcher`, `_LeagueInferenceService`).
- Multiple Showdown server instances (supported via `--server-urls`).
- Parallel `BattleEnv` instances sharing inference.

**Still open:**

- A faster Python-native battle simulator would be a multi-month engineering project but is the only way to get orders-of-magnitude throughput.

### 10.8 League diversity is bounded by a small rolling window

With `league_size=12` and `league_update_interval=10`, the pool spans at most ~120 updates back. If the policy converges slowly, the pool can contain policies that are all very similar — no diversity benefit. There is also no diagnostic in place to detect strategic cycling (the current agent silently forgetting how to beat older policies).

**Fix direction:** larger `league_size`, or a priority-replay scheme that maintains diverse checkpoints (e.g. keep a sample across different ELO tiers, or different seeds). See [LEAGUE_IMPROVEMENTS.md](LEAGUE_IMPROVEMENTS.md) for a fuller design-space overview — evaluation harness, PFSP sampling, archiving, KL anchors, exploiter agents.

### 10.9 No explicit hazard / weather strategy

The agent must discover hazard, weather, and terrain strategies entirely from scratch via sparse reward. It's learnable but slow. A curriculum — starting with simpler formats and increasing complexity — could accelerate this substantially, and would compose with the win-probability shaper idea from §10.3.

---

## 11. The Training Loop in One Picture

```
┌───────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                               │
│                                                                       │
│  ┌─────────────┐     128 steps     ┌──────────────────────────────┐   │
│  │ 16 parallel │ ─────────────────→│     RolloutBuffer            │   │
│  │ Showdown    │  obs/action/rew   │     (T=128, E=16, ...)       │   │
│  │ battles     │ ←─────────────────│     → compute GAE            │   │
│  └─────────────┘   agent actions   └──────────────────────────────┘   │
│         ↑                                         │                    │
│         │  league opponents                       │ RolloutBatch       │
│  ┌──────────────┐                                 ↓                    │
│  │  League      │              ┌──────────────────────────────────┐    │
│  │  (≤12 frozen │              │     PPOTrainer.update()          │    │
│  │  checkpoints)│              │  4 epochs × 4 env-minibatches    │    │
│  └──────────────┘              │  → spatial batched, LSTM looped  │    │
│         ↑                      │  → clipped policy + value losses │    │
│         │ every 10 updates     │  → entropy bonus                 │    │
│  ┌──────────────┐              │  → Adam + grad clip              │    │
│  │  PokeAgent   │←─────────────└──────────────────────────────────┘    │
│  │  (trained)   │                                                      │
│  └──────────────┘                                                      │
│         │                                                              │
│         │ every 8 updates (accumulated)                                │
│         ↓                                                              │
│  ┌──────────────────────────────────────────────────────────────┐      │
│  │  PPOTrainer.update_team_preview()                            │      │
│  │  REINFORCE + preview value baseline + opponent-model aux     │      │
│  └──────────────────────────────────────────────────────────────┘      │
└───────────────────────────────────────────────────────────────────────┘

Battle turn:
  obs
    → SpatialEncoder
    → Memory (LSTM)
    → ActionHead:
         a1
         → a2 | a1
         → tera | a1, a2
    → ValueHead(memory_out, own_bench, opp_bench)

Team preview:
  synthetic obs
    → SpatialEncoder
    → Memory
    → TeamPreviewHead (canonical bring set → ordered lead pair)
    → OpponentPreviewHead (auxiliary)
    → PreviewValueHead (REINFORCE baseline)
```
