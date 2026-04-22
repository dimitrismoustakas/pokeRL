# pokeRL: Model Architecture & Training Semantics

This document is a detailed reference for how the model works, what design decisions were made, and where the system has room to improve. It is intended both as an onboarding guide for people unfamiliar with the codebase and as an analytical reference for future development.

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

---

## 1. High-Level Overview

The agent is trained to play **VGC doubles battles** (and random doubles) via **Proximal Policy Optimization (PPO)** against opponents sampled from a **league of past checkpoints**. The environment is a real [Pokemon Showdown](https://pokemonshowdown.com/) server running locally, bridged into Gymnasium via the [poke-env](https://github.com/hsahovic/poke-env) library.

The core loop is:

```
Reset → Observe Battle State → Encode State → LSTM Memory → Choose Actions → Step → Collect Reward → Update Policy
```

**Key properties:**
- **Doubles battle**: two active pokemon per side simultaneously
- **Sparse reward**: only +1 win / -1 loss at battle end; no intermediate shaping
- **Recurrent policy**: LSTM carries information across turns
- **League self-play**: trained against a rotating pool of frozen past checkpoints
- **Autoregressive actions**: slot 1 is chosen first, slot 2 is conditioned on slot 1
- **Team preview**: separate policy for picking and ordering which 4 of 6 pokemon to bring

---

## 2. State Representation: What the Model Sees

Implemented in [pokerl/env/observation.py](pokerl/env/observation.py).

### 2.1 Pokemon Features

Each pokemon (own team 6 + opponent team 6) is encoded with two types of features:

#### Categorical features (looked up as learned embeddings)

| Feature | Shape | Notes |
|---|---|---|
| `species` | `(6,)` int | Index into species vocabulary |
| `moves` | `(6, 4)` int | 4 move slots; unknown moves → UNKNOWN index |
| `last_move` | `(6,)` int | Last revealed or used move |
| `ability` | `(6,)` int | Ability index |
| `item` | `(6,)` int | Named item vocab + hash-bucketed fallback |
| `types` | `(6, 3)` int | `[type_1, type_2, tera_type]` |

#### Continuous features (44 dimensions per pokemon)

| Feature | Dims | Notes |
|---|---|---|
| HP fraction | 1 | `current_hp / max_hp` |
| Status condition | 7 | One-hot: {none, BRN, FRZ, PAR, PSN, SLP, TOX} |
| Stat boosts | 7 | {ATK, DEF, SPA, SPD, SPE, ACC, EVA} normalized by ÷6 |
| Volatile effects | 16 | Binary: confusion, substitute, leech seed, taunt, encore, torment, disable, yawn, perish song, protect, endure, focus energy, attract, curse, nightmare, flash fire |
| Move PP remaining | 4 | `current_pp / max_pp` per move slot; **always zero for opponent pokemon** |
| Protect counter | 1 | Normalized to [0, 4] |
| Status counter | 1 | Sleep turns / toxic turns; normalized to [0, 8] |
| First turn | 1 | True if pokemon just switched in |
| Revealed | 1 | True if opponent pokemon has been seen in battle |
| Known move count | 1 | `num_known_moves / 4` |
| Ability known | 1 | True if revealed |
| Item known | 1 | True if revealed |
| Tera type known | 1 | True if revealed |
| Terastallized | 1 | True if currently in tera form |

**Important asymmetry:** PP features are populated for own pokemon but zero for opponent pokemon. This is intentional: the model learns this as an implicit "is own" signal. See [Section 10](#10-known-limitations--improvement-opportunities) for the planned fix.

### 2.2 Field Features (36 dimensions)

| Feature | Dims | Notes |
|---|---|---|
| Weather type | 9 | One-hot over 8 weather types + none |
| Weather age | 1 | Normalized to [0, 8] |
| Terrain type | 5 | One-hot over 4 terrain types + none |
| Terrain age | 1 | Normalized |
| Own side conditions | 8 | Stealth Rock (bool), Spikes (0-3), Toxic Spikes (0-2), Sticky Web (bool), Reflect age, Light Screen age, Aurora Veil age, Tailwind age |
| Opponent side conditions | 8 | Same as own |
| Trick Room age | 1 | Normalized |
| Tera used (own) | 1 | Bool: own tera already consumed this battle |
| Tera used (opponent) | 1 | Bool |
| Turn counter | 1 | Normalized to [0, 20] |

### 2.3 Battle Metadata (passed separately as tensors)

- `own_roles / opp_roles`: `(6,)` int — assigns each team slot a role: `ACTIVE_SLOT_1`, `ACTIVE_SLOT_2`, or `BENCH`
- `own_mask / opp_mask`: `(6,)` bool — padding mask for unused slots
- `own_active_indices / opp_active_indices`: `(2,)` int — which team slots are currently active
- `own_active_present / opp_active_present`: `(2,)` bool — whether each active slot is actually present (handles faints)

### 2.4 Item Vocabulary

The item vocabulary uses a two-tier system implemented in [pokerl/data.py](pokerl/data.py):

1. **Named tier**: items harvested from the Showdown Pokedex `requiredItem`/`requiredItems` fields — forms with held item requirements like Giratina-Origin, Arceus types, etc.
2. **Hash bucket tier**: all other items are mapped via `blake2b(item_id) % num_hash_buckets` into a fixed pool. Most competitive items (Choice Band, Life Orb, Leftovers, etc.) currently land here.

This means most battle-relevant items share embedding buckets and cannot be individually learned. This is a known limitation described in [Section 10](#10-known-limitations--improvement-opportunities).

---

## 3. Model Architecture

Entry point: [pokerl/model/agent.py](pokerl/model/agent.py). The `PokeAgent` composes four sub-modules in sequence.

### 3.1 PokemonEncoder

**File:** [pokerl/model/encoders.py](pokerl/model/encoders.py)

Shared weights encode each individual pokemon into a fixed-size vector. The same `PokemonEncoder` is applied to all 12 pokemon (own team + opponent team).

**Input per pokemon:**

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
380 → Linear(256) → ReLU → Linear(128) → ReLU → output: 128
```

**Output:** `(batch, pokemon_out=128)` per pokemon.

The species and move embeddings can optionally be initialized from base stats (see [Section 3.6](#36-stat-based-embedding-initialization)).

### 3.2 TeamEncoder

**File:** [pokerl/model/encoders.py](pokerl/model/encoders.py)

Two separate `TeamEncoder` instances (one for own team, one for opponent team) apply order-invariant attention over the 6 per-pokemon representations.

**Input:**
- Per-pokemon reps: `(batch, 6, 128)` from `PokemonEncoder`
- Role indices: `(batch, 6)` — role embedding added to each pokemon representation

**Role embedding:** 3 roles (active slot 1, active slot 2, bench) → `role_dim=128`

**Architecture:**
```
(pokemon_rep + role_emb) → TransformerEncoder(nhead=4, num_layers=2, dim_feedforward=256)
```

The Transformer operates on the 6-pokemon sequence with no positional bias — the attention is **order-invariant** (only role matters, not team slot order). Padding mask is applied for unused team slots.

**Output:** `(batch, 6, 128)` attended pokemon representations.

### 3.3 FieldEncoder

**File:** [pokerl/model/encoders.py](pokerl/model/encoders.py)

A simple MLP over the 36-dimensional field feature vector:

```
36 → Linear(64) → ReLU → output: 64
```

### 3.4 SpatialEncoder (Combines the Above)

**File:** [pokerl/model/encoders.py](pokerl/model/encoders.py)

Runs `PokemonEncoder` on all pokemon, passes them through `TeamEncoder` for each team, runs `FieldEncoder`, and produces:

| Output | Shape | Description |
|---|---|---|
| `context` | `(batch, 320)` | Global battle context: `mean(own_reps) ∥ mean(opp_reps) ∥ field` |
| `active_reps` | `(batch, 2, 128)` | Own two active pokemon representations |
| `own_reps` | `(batch, 6, 128)` | Full own team (post-attention) |
| `opp_reps` | `(batch, 6, 128)` | Full opponent team (post-attention) |
| `opp_active_reps` | `(batch, 2, 128)` | Opponent's two active pokemon |

The `context` vector is the "global summary" used by the LSTM and action head. It concatenates the mean-pooled own team, mean-pooled opponent team, and field encoding — so it collapses team order entirely.

### 3.5 Memory Module

**File:** [pokerl/model/memory.py](pokerl/model/memory.py)

Two interchangeable implementations:

**IdentityMemory** (feedforward baseline):
- Projects context through a linear layer: `input_dim → hidden_dim`
- No temporal state; each turn is independent
- Useful for ablation

**LSTMMemory** (default):
- Standard `nn.LSTM(input_dim=320, hidden_dim=256, batch_first=True)`
- Maintains `(h, c)` hidden state across turns
- At episode boundaries (`done=True`), hidden state is zeroed: `h = h * not_done`
- Supports **truncated BPTT**: optionally detaches hidden state every `bptt_chunk_steps` timesteps to reduce memory during training

**Output:** `(batch, 256)` memory output, updated hidden state.

The LSTM gives the policy its "memory" of the battle — how HP has changed, what moves have been revealed, what the flow of the battle has been. Without it the model would need to infer all of this purely from current observation features.

### 3.6 Stat-Based Embedding Initialization

**File:** [pokerl/config.py](pokerl/config.py), [pokerl/data.py](pokerl/data.py)

When `init_embeddings_from_stats=True` (the default), species and move embeddings are initialized as linear projections of base stats rather than random:

- **Species**: `(HP, Atk, Def, SpA, SpD, Spe)` normalized by ÷255 → linear projection → embedding
- **Moves**: `(power/250, accuracy/100, pp/64)` → linear projection → embedding

This gives the model a warm start that encodes known structure: e.g., fast pokemon start with high Spe embeddings, strong attackers start with high Atk embeddings. In practice this improves early training significantly.

### 3.7 Action Head

**File:** [pokerl/model/heads.py](pokerl/model/heads.py)

**Autoregressive doubles selection.** Slot 1 is chosen first; slot 2 is chosen conditioned on what slot 1 chose.

#### MaskedSetAttention (building block)

A learned attention pooling operation used inside the action head:

```
Query (batch, q_dim) × Items (batch, N, d) → (batch, d)
```

Computes scaled dot-product attention weights over the items given the query, applies the mask (to ignore illegal/unavailable items), and returns a weighted sum. Used to summarize the opponent bench conditioned on the current action context.

#### Slot 1 Selection

1. **Query construction:**
   - Concatenate: `context (320) ∥ active[0] (128) ∥ active[1] (128) ∥ opp_active[0] (128) ∥ opp_active[1] (128)` → 832 dims
   - MLP: `832 → Linear(256) → ReLU` → `query1 (256)`

2. **Opponent bench summary:**
   - `MaskedSetAttention(query1, opp_bench_reps, opp_bench_mask)` → `opp_bench_summary (128)`
   - Fuse: `[query1 (256) ∥ opp_bench_summary (128)] → Linear(256) → ReLU` → `query1_fused (256)`

3. **Action scoring:**
   Each of the 47 possible actions gets a scalar score from three components:
   - **Static features**: per-action embeddings for move slot, target, switch slot, action kind, tera flag, action ID → projected to 256 dims
   - **Dynamic entity features**: the pokemon representation most relevant to this action (own pokemon for moves, switch target for switches) → projected to 256 dims
   - **Score network**: `[query (256) ∥ static (256) ∥ entity (256)] → Linear(256) → ReLU → Linear(1)`

4. Apply action mask (illegal actions → -inf), softmax, sample (or argmax in greedy mode) → `a1`

#### Slot 2 Selection

Same pipeline, but query is augmented with the slot 1 action embedding:

```
context ∥ active[0] ∥ active[1] ∥ opp_active[0] ∥ opp_active[1] ∥ a1_emb
```

This conditioning allows slot 2 to reason about what slot 1 will do — e.g., don't protect with both pokemon, don't use two moves that both require the same target.

**Output of ActionHead:**
- `actions: (batch, 2)` — chosen slot 1 and slot 2 action indices
- `log_prob: (batch,)` — `log π(a1) + log π(a2)` (sum of both log probs)
- `entropy: (batch,)` — `H(π1) + H(π2)` (sum of both entropies)

### 3.8 Value Head

**File:** [pokerl/model/heads.py](pokerl/model/heads.py)

Estimates V(s) for Generalized Advantage Estimation (GAE):

1. **Bench attention:** `MaskedSetAttention(context, own_bench, bench_mask)` → `(batch, 128)`
2. **Opponent bench attention:** `MaskedSetAttention(context, opp_bench, bench_mask)` → `(batch, 128)`
3. **MLP:** `[context (320) ∥ own_bench_summary (128) ∥ opp_bench_summary (128)] → Linear(256) → ReLU → Linear(1)`

The bench summaries allow the value function to consider the depth of both teams (not just what's active), which matters a lot for predicting win probability in doubles.

### 3.9 Complete Forward Pass

For a single battle turn:

```
obs dict
  → PokemonEncoder (all 12 pokemon, shared weights)
  → TeamEncoder (own team) + TeamEncoder (opponent team)
  → FieldEncoder
  → SpatialEncoder output: {context (320), active_reps, own_reps, opp_reps, opp_active_reps}
  → LSTMMemory: context → (memory_out (256), new_hidden)
  → ActionHead: (memory_out, active_reps, opp_reps) → {actions (2,), log_prob, entropy}
  → ValueHead: (context, own_reps, opp_reps) → value (scalar)
```

Note: The LSTM receives the `context` (spatial summary) and produces `memory_out`. Action head uses `memory_out` as its query foundation; value head uses raw `context`. This means the value function is not recurrent — it sees the current frame's spatial summary directly, while the policy's decisions are informed by the LSTM's accumulated history.

---

## 4. Action Space

**File:** [pokerl/env/actions.py](pokerl/env/actions.py)

47 actions per slot (`ACTIONS_PER_SLOT`):

```
0–19:   Move 0–3 × Target 0–4 (no tera)
20–39:  Move 0–3 × Target 0–4 (with terastallization)
40–45:  Switch to team slot 0–5
46:     Pass (no action for this slot)
```

**Target encoding:**
```
0: Opponent slot 1
1: Opponent slot 2
2: Ally (partner)
3: Self
4: No target (field/all-target moves)
```

### 4.1 Action Masking

The action mask has shape `(ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT) = (48, 47)`:

- **Row 0**: legal slot-1 actions
- **Row 1 + a1**: legal slot-2 actions **conditioned on** slot-1 choice `a1`

This joint masking prevents physically impossible combinations (e.g., both pokemon switching to the same slot, or illegal move targets given the opponent's active lineup).

During inference, illegal actions are masked to `-inf` before softmax, making them probability 0. This is a hard constraint, not a soft one.

---

## 5. Training Algorithm

**File:** [pokerl/training/ppo.py](pokerl/training/ppo.py), [pokerl/training/rollout.py](pokerl/training/rollout.py)

### 5.1 Rollout Collection

`RolloutBuffer` pre-allocates tensors of shape `(rollout_steps=128, num_envs=8, ...)` on the GPU.

At each timestep:
1. All environments step in parallel (via poke-env async loop)
2. Agent batches all observations: `(num_envs, ...)` → forward pass → `(num_envs, 2)` actions
3. Actions sent back to environments
4. Collect `(obs, action, action_mask, log_prob, value, reward, done)` tuples into buffer

After `rollout_steps` steps:
- Call `get_value()` on the final observation to get the bootstrap value `V(s_T)`
- Compute **GAE (Generalized Advantage Estimation)**:

```python
for t in reversed(range(T)):
    next_value = last_value if t == T-1 else values[t+1]
    next_non_terminal = (~dones[t]).float()
    delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
    gae = delta + gamma * gae_lambda * next_non_terminal * gae
    advantages[t] = gae
returns = advantages + values
```

With `gamma=0.99` and `gae_lambda=0.95`, this biases the advantage estimate towards actual returns while reducing variance.

### 5.2 PPO Update

Called once per rollout. Runs `num_epochs=4` passes over the data.

**Critical design: minibatches slice environments, not timesteps.** With `num_minibatches=4` and `num_envs=8`, each minibatch contains 2 environments × all 128 timesteps. This preserves the full temporal sequence for each environment so the LSTM can be unrolled correctly.

If minibatches sliced timesteps instead, LSTM hidden states would be incorrect at minibatch boundaries, breaking the gradient flow through episode boundaries.

**Per-minibatch update:**

1. **Normalize advantages** (across the full batch, once, before epochs):
   ```python
   adv = (adv - adv.mean()) / (adv.std() + 1e-8)
   ```

2. **Spatial encoder pass** (batched over all T×E_mb timesteps at once for efficiency)

3. **Sequential LSTM pass** (loop over timesteps to maintain hidden state):
   - Zero hidden at `done` boundaries: `h = h * (~done).float()`
   - Optional truncated BPTT: detach hidden every `bptt_chunk_steps` steps

4. **Losses:**

   **Policy loss (clipped surrogate):**
   ```python
   ratio = exp(new_log_prob - old_log_prob)
   policy_loss = -min(ratio * adv, clamp(ratio, 1-eps, 1+eps) * adv).mean()
   ```
   With `clip_eps=0.2`, ratios are clamped to [0.8, 1.2].

   **Value loss (clipped):**
   ```python
   v_clipped = old_value + clamp(new_value - old_value, -eps, eps)
   value_loss = max((new_value - returns)^2, (v_clipped - returns)^2).mean()
   ```

   **Entropy bonus:**
   ```python
   entropy_loss = -entropy.mean()
   ```
   With `entropy_coef=0.01`, encourages exploration.

   **Total:**
   ```python
   loss = policy_loss + value_coef * value_loss + entropy_coef * entropy_loss
   ```
   With `value_coef=0.5`.

5. **Gradient update:**
   ```python
   loss.backward()
   clip_grad_norm(parameters, max_norm=0.5)
   optimizer.step()
   ```
   Adam with `lr=2.5e-4`, `eps=1e-5`.

**Diagnostic tracked:** Approximate KL divergence `((ratio - 1) - log_ratio).mean()` — useful for monitoring if the policy is updating too aggressively.

### 5.3 Team Preview Update (REINFORCE)

**File:** [pokerl/training/preview.py](pokerl/training/preview.py)

Team preview decisions cannot be handled with PPO directly because:
- They happen once per battle (not once per turn)
- The reward signal is the final battle outcome (+1/-1), received much later

Instead, team preview uses **REINFORCE with a baseline**:

```python
advantage = episode_reward - value_baseline
policy_loss = -log_prob * advantage
```

Where `value_baseline` is the value estimate at the start of the battle (before any turns are played).

**Auxiliary losses for opponent modeling:**

The `OpponentPreviewHead` predicts what the opponent will bring and who they'll lead:

- `bring_loss`: Binary cross-entropy predicting which 4 of 6 opponent pokemon were actually brought
- `lead_loss`: Cross-entropy predicting the opponent's lead pair

These don't affect the team selection policy but improve the representations in `opp_reps`, helping downstream reasoning.

```python
total = -policy_loss - entropy_bonus + bring_coef * bring_loss + lead_coef * lead_loss
```

With `bring_coef=0.1` and `lead_coef=0.05`.

---

## 6. Reward Structure

**File:** [pokerl/env/rewards.py](pokerl/env/rewards.py)

```python
def sparse_reward(battle):
    if battle.won:   return +1.0
    if battle.lost:  return -1.0
    return 0.0
```

**Purely sparse terminal rewards.** The agent receives no signal during the battle — only at the very end.

This means:
- The value function must learn to predict win probability from state alone
- The LSTM must learn to track relevant battle dynamics (HP, KOs, field control) without any shaping
- Credit assignment is hard: a mistake on turn 3 that causes a loss on turn 15 must still be attributed

There are **no intermediate rewards** for: knocking out opponent pokemon, dealing damage, preserving own pokemon, or executing type-effective moves.

This is intentional: reward shaping can introduce unintended biases (e.g., a bot that maximizes damage dealt but ignores winning). The sparse setup forces the agent to actually learn to win. The cost is slower learning.

---

## 7. Team Preview

**Files:** [pokerl/model/heads.py](pokerl/model/heads.py), [pokerl/env/team_preview.py](pokerl/env/team_preview.py)

### 7.1 What happens at team preview

At the start of each VGC battle, both players see each other's **full 6-pokemon roster** (species only — moves/items/abilities are hidden). Each player must:

1. Choose which **4 of 6** pokemon to bring
2. Choose the **order** they'll appear in (leading with 2, bench of 2)

### 7.2 TeamPreviewHead

**Autoregressive 4-of-6 selection:**

For each pick position `i` in {1, 2, 3, 4}:
1. Build context from: global context + accumulated representation of already-picked pokemon
2. Attend over opponent team to get opponent summary
3. Score each remaining (not yet picked) own pokemon: `[context ∥ opp_summary ∥ pokemon_rep] → score`
4. Mask already-picked, softmax, sample
5. Add picked pokemon rep to accumulator

This is autoregressive: pick 2 is conditioned on pick 1, pick 3 on picks 1 and 2, etc. The model can learn "if I'm bringing Incineroar, I should also bring a pokemon that covers its weaknesses."

### 7.3 OpponentPreviewHead (auxiliary)

Predicts opponent behavior:
- `bring_score`: per-opponent-pokemon sigmoid score for whether it'll be brought
- `lead_pair_score`: score for each pair of opponent pokemon as potential leads

These predictions are **not used at inference** — they're purely for training signal to improve the learned `opp_reps`.

### 7.4 Team preview observation

A synthetic battle state is constructed with all pokemon at full HP, no status, no field conditions. This gives the agent a clean "initial snapshot" before the battle begins.

---

## 8. League Self-Play

**File:** [pokerl/training/league.py](pokerl/training/league.py)

### 8.1 Why self-play?

Training against a fixed opponent (random, greedy max-power) produces a policy that exploits that specific opponent's weaknesses. League self-play trains against a distribution of past selves, creating pressure to generalize.

### 8.2 League Structure

A pool of up to `league_size=12` frozen past checkpoints. Every `league_update_interval=10` PPO updates, the current agent is saved and added to the pool. When the pool is full, the oldest checkpoint is evicted.

**Sampling policy:** 70% chance to sample from the most recent half of the pool, 30% from the older half. This balances:
- **Exploitation**: playing against strong, recent opponents (improves skill)
- **Exploration**: playing against diverse, older opponents (avoids exploitation holes)

### 8.3 Inference Architecture

`_LeagueInferenceService` runs in a background thread and batches inference requests:

- Multiple parallel `LeaguePlayer` instances (one per concurrent battle) submit inference requests
- Requests are batched per checkpoint over a `coalesce_window_ms=10ms` window
- Batched forward passes run on the service device, responses returned via `Future`

This is necessary because each `LeaguePlayer` drives a separate poke-env async player, and naively running each separately would be extremely slow.

### 8.4 LeaguePlayer

Wraps a frozen checkpoint into a poke-env `Player` interface:
- Maintains its own LSTM hidden state across battle turns
- Supports mid-battle checkpoint staging (for updating opponents without interrupting battles)
- Produces legal poke-env orders from raw action indices

---

## 9. Configuration Reference

**File:** [pokerl/config.py](pokerl/config.py)

### ModelConfig (default values)

| Parameter | Default | Effect |
|---|---|---|
| `species_dim` | 64 | Species embedding size |
| `move_dim` | 32 | Move embedding size |
| `ability_dim` | 32 | Ability embedding size |
| `item_dim` | 32 | Item embedding size |
| `type_dim` | 16 | Type embedding size |
| `pokemon_hidden` | 256 | PokemonEncoder MLP hidden size |
| `pokemon_out` | 128 | PokemonEncoder output size |
| `team_attention_heads` | 4 | TeamEncoder attention heads |
| `team_attention_layers` | 2 | TeamEncoder Transformer depth |
| `field_out` | 64 | FieldEncoder output size |
| `memory_type` | `"lstm"` | `"lstm"` or `"identity"` |
| `memory_hidden` | 256 | LSTM hidden size |
| `action_hidden` | 256 | ActionHead MLP hidden size |
| `value_hidden` | 256 | ValueHead MLP hidden size |
| `pick_size` | 4 | Team preview: bring N of 6 |
| `init_embeddings_from_stats` | `True` | Warm-start embeddings from base stats |

### TrainingConfig (default values)

| Parameter | Default | Effect |
|---|---|---|
| `lr` | 2.5e-4 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `gae_lambda` | 0.95 | GAE smoothing parameter |
| `clip_eps` | 0.2 | PPO clipping range |
| `entropy_coef` | 0.01 | Entropy bonus weight |
| `value_coef` | 0.5 | Value loss weight |
| `max_grad_norm` | 0.5 | Gradient clipping |
| `rollout_steps` | 128 | Steps per env before update |
| `num_epochs` | 4 | PPO epochs per rollout |
| `num_minibatches` | 4 | Env minibatches per epoch |
| `bptt_chunk_steps` | `None` | Truncated BPTT chunk size (None = full) |
| `total_timesteps` | 10M | Training budget |
| `save_interval` | 100 | Save checkpoint every N updates |
| `league_size` | 12 | Max past checkpoints |
| `league_update_interval` | 10 | Save to league every N updates |
| `team_preview_bring_coef` | 0.1 | Bring-prediction auxiliary loss weight |
| `team_preview_lead_coef` | 0.05 | Lead-prediction auxiliary loss weight |

---

## 10. Known Limitations & Improvement Opportunities

### 10.1 Item Vocabulary — Most Items Are Hash-Bucketed

**Problem:** The named item vocabulary (`GameData._known_item_ids`) only captures items from the Showdown Pokedex's `requiredItem`/`requiredItems` fields — essentially form-locked items like Giratina-Origin's Griseous Orb. Virtually all competitive items (Choice Band, Assault Vest, Leftovers, Life Orb, Safety Goggles, etc.) fall into hash buckets.

Hash buckets collide: multiple items share the same embedding, so the model cannot distinguish Life Orb from Choice Specs if they hash to the same bucket. There is also no stat-based initialization for items.

**Fix:** Build a proper competitive item vocabulary from actual VGC usage data (or the full Showdown items database), not just the Pokedex. Add stat-like features for items (power boost amount, residual %, speed mod, etc.) to enable stat-based init.

### 10.2 Opponent PP Features Always Zero

**Problem:** The 4 PP dimensions in the continuous feature vector are set to zero for all opponent pokemon. The model can learn that "zero PP = opponent" but cannot learn to track opponent PP depletion (e.g., knowing an opponent's Fake Out can't be used again).

**Fix:** Give opponent pokemon their own `continuous_dim` without PP features, and explicitly separate the encoding: the model then can't confuse "no PP info" with "full PP." Alternatively, expose a "PP remaining" signal derived from move use counts (inferred from battle log).

### 10.3 Value Function Is Not Recurrent

**Problem:** The value head receives `context` (the current-frame spatial summary) rather than `memory_out` (the LSTM output). This means the value estimate doesn't benefit from the LSTM's accumulated history of the battle.

This is a subtle asymmetry: the policy uses memory to decide actions, but the value function is essentially Markovian. In practice, the value function must infer battle history from the current observation (e.g., HP changes, known moves), but this is a harder task.

**Fix:** Pass `memory_out` into the value head as an additional input, or use a separate LSTM for the value function (asymmetric actor-critic). This should improve value estimates and therefore advantage quality.

### 10.4 Sparse Reward Is Hard

**Problem:** +1/-1 terminal rewards with no intermediate shaping means:
- Slow credit assignment across long battles (15–20 turns)
- The value function learns slowly (no dense signal)
- Team preview decisions are especially hard to learn (very delayed reward)

**Potential improvements:**
- **Win probability shaping**: train a separate win-probability predictor on game states and use its delta as reward. Self-supervised and neutral to reward hacking.

### 10.5 Full BPTT Over Long Rollouts Is Memory-Intensive

**Problem:** When `bptt_chunk_steps=None` (default), the entire 128-step LSTM unroll is kept in the computation graph for backprop. With larger rollouts or bigger models, this can exhaust GPU memory.

**Fix:** `bptt_chunk_steps=32` is already implemented but not the default. Setting it truncates backprop every 32 steps, trading some gradient quality for memory. For very long rollouts, this becomes necessary.

### 10.6 Lead Ordering Not Explicitly Learned

**Problem:** The `TeamPreviewHead` picks 4 pokemon but the **order** of picks determines the lead (picks 1 and 2 are the lead). The model is not explicitly trained to optimize lead selection separately from bring selection — it must implicitly learn that "pick first = lead slot 1."

**Fix:** A dedicated `LeadHead` that selects the lead pair from the 4 already-chosen pokemon as a separate decision, with its own reasoning over opponent team composition.

### 10.7 No Opponent Move/Item Inference

**Problem:** When an opponent uses a move, the model observes that move in `last_move` but there is no mechanism to update a probabilistic belief over the opponent's full moveset. Similarly, opponent items are unknown.

**Fix:** An auxiliary prediction head for opponent moves (multi-label classification), trained on revealed moves as supervision. The learned beliefs could be fed as additional features into the action head. This is a significant architectural addition.

### 10.8 Single Battle Latency Limits Throughput

**Problem:** Even with many parallel environments, the per-battle latency through the Showdown server (async, network overhead) is the bottleneck. Individual battles can take minutes. With sparse rewards this means few reward signals per wall-clock hour.

**Potential improvements:**
- Faster Showdown server (raw node.js, not electron)
- Multiple Showdown instances (already supported via `--server-urls`)
- Batching inference for all envs together (already done via `ParallelBattleEnv`)
- A faster Python-native battle simulator (significant engineering work)

### 10.9 No Explicit Hazard/Weather Strategy

The agent must discover hazard, weather, and terrain strategies entirely from scratch via sparse rewards. This is learnable but very slow. Curriculum approaches — starting with simpler battle formats and increasing complexity — could accelerate this.

### 10.10 League Pool May Not Be Diverse Enough

With `league_size=12` and `league_update_interval=10`, the pool spans at most 120 updates back. If the policy is converging slowly (as expected in a hard env), the pool might contain policies that are all very similar — no diversity benefit.

**Fix:** Larger `league_size`, or a priority replay that maintains diverse checkpoints (e.g., at different ELO levels or trained on different opponent distributions).

---

## Summary: The Full Training Loop in One Picture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                             │
│                                                                     │
│  ┌─────────────┐     128 steps     ┌──────────────────────────┐   │
│  │ 8 parallel  │ ──────────────→   │     RolloutBuffer        │   │
│  │ Showdown    │  obs/action/rew   │  (T=128, E=8, ...)       │   │
│  │ battles     │ ←──────────────   │  → compute GAE           │   │
│  └─────────────┘   agent actions   └──────────────────────────┘   │
│         ↑                                        │                  │
│         │  league opponents                      │ RolloutBatch     │
│  ┌──────────────┐                               ↓                  │
│  │  League      │              ┌─────────────────────────────────┐ │
│  │  (12 frozen  │              │     PPOTrainer.update()         │ │
│  │  checkpoints)│              │  4 epochs × 4 env-minibatches   │ │
│  └──────────────┘              │  → policy loss (clipped ratio)  │ │
│         ↑                      │  → value loss (clipped)         │ │
│         │ every 10 updates     │  → entropy bonus                │ │
│  ┌──────────────┐              │  → Adam step + grad clip        │ │
│  │  PokeAgent   │←─────────────└─────────────────────────────────┘ │
│  │  (trained)   │                                                   │
│  └──────────────┘                                                   │
└─────────────────────────────────────────────────────────────────────┘
```

The agent sees battle state → encodes it spatially → passes through LSTM → chooses two actions autoregressively → receives sparse reward at battle end → PPO updates policy and value function → repeat.