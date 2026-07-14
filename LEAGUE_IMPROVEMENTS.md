# League Self-Play: Improvement Options

A design-space overview for improving the current permanent-archive self-play
league. This is a reference for *what the options are* and *what the tradeoffs
look like*, not a committed plan. See
[pokerl/training/league.py](pokerl/training/league.py) and
[§8 of MODEL_EXPLANATION.md](MODEL_EXPLANATION.md#8-league-self-play) for the
current implementation.

---

## 1. The problem we're worried about

Pure self-play is famously prone to **non-transitive dynamics** (a.k.a. **strategic cycling**). The policy optimizes against the distribution it currently sees, so a weakness that was patched 300k steps ago can silently re-emerge once the opponent distribution no longer contains the counter-strategy that forced the patch. Pokemon is especially non-transitive: threat coverage is roughly rock-paper-scissors between archetypes, so cycling is a plausible failure mode rather than a purely theoretical one.

Concretely, the questions we want to answer are:

- Is the agent getting **globally** better, or just chasing whatever the current meta in its own pool happens to be?
- Would the agent at step 5M still beat the agent from step 1M? Or has it forgotten how?
- Are old checkpoints in the pool actually applying pressure, or do they all share the same blind spots as the current agent?

Heuristic opponents (random, max-power, scripted) are deliberately excluded here — the goal is to understand whether *self-play itself* is producing monotone improvement, not to prop it up with external anchors.

---

## 2. Current state

- Permanent checkpoint archive, snapshot every `league_update_interval = 10` PPO updates. `league_size = 12` bounds the LRU inference cache rather than archive history. The default two opponent groups are resampled every update and receive distinct members whenever possible.
- Sampling: implemented hard-opponent PFSP over exponentially discounted, update-clustered live-policy outcomes, mixed with 10% uniform archive coverage. Later group draws are conditional on earlier distinct selections.
- Training summaries record wins, losses, and draws by opponent checkpoint. The evaluation harness now runs sampled or greedy checkpoint cross-play on pre-planned team pairs, mirrors player/team roles, and retains the raw payoff matrix.

The first 0/50/100/125/150 matrix found strong improvement through update 100 and then a plateau, with no statistically supported cycle. The immediate gap is therefore **absolute and roster-specific skill**, not demonstrated league cycling: the same final checkpoint ranged from 14.5% to 94.7% against SimpleHeuristics depending on its own roster. See [EVALUATION.md](EVALUATION.md).

---

## 3. The literature, in order of relevance

1. **AlphaStar (Vinyals et al., 2019)** — the most directly relevant reference. Addresses cycling explicitly with:
   - **Prioritized Fictitious Self-Play (PFSP)**: sample opponents weighted by `f(win_rate)` — focus training on checkpoints we *aren't* beating, not the uniform pool. This is the single highest-leverage idea for our setup.
   - **Main agents + league exploiters + main exploiters**: parallel agents with different training objectives. Exploiters specialize in finding weaknesses and get added to the league once they find one.
   - **Permanent archive**: checkpoints are never evicted. Old checkpoints re-surface via PFSP if the current agent has drifted away from beating them.

2. **OpenAI Five (Dota 2)** — simpler: 80% current / 20% past self. Shows that rolling-pool self-play can work at scale, but doesn't directly address cycling — it assumes compute makes the problem small.

3. **AlphaGo Zero / AlphaZero** — tournament gating: new checkpoints join the pool only if they beat the current best >55% over many games. Portable idea: *measure before trusting*.

4. **PSRO / Double Oracle (Lanctot et al., 2017)** — meta-game framework: compute a Nash meta-strategy over the current population, then add best-responders to that distribution. Theoretically clean, heavy machinery.

5. **Neural Fictitious Self-Play (Heinrich & Silver, 2016)** — explicitly tracks an average-policy network alongside the best-response, and plays against the average. Has convergence guarantees in two-player zero-sum games.

---

## 4. Menu of options

Organized by effort vs. expected payoff.

### Tier 1 — cheap, high value

#### A. Evaluation harness first

**Implemented offline.** Keep a small **anchor set** of frozen checkpoints permanently on disk, outside the rolling pool. Periodically run current agent versus each anchor on the mirrored schedule and retain the complete payoff matrix.

- If win rate vs an old anchor flatlines or *decreases*, cycling is real.
- If it keeps rising, the current setup is working and further changes may be premature.

The remaining work is orchestration: invoke the panel periodically during long runs and add held-out team populations.

#### B. Prioritized Fictitious Self-Play (PFSP) — implemented

The trainer now replaces the 70/30 recent/old rule with weighted sampling over the full archive. Each checkpoint/update bucket contributes one outcome-proportion evidence unit, evidence has a configurable half-life, and hard-PFSP is mixed with uniform coverage:

- `(1 − p)^α` — train against what we're losing to (AlphaStar's "hard" variant).
- `p · (1 − p)` — train against coin-flip matchups (variance-focused, more conservative).

If an old checkpoint starts beating the current agent, it gets sampled more, and the agent is forced to re-learn its counter. Small code change, directly addresses cycling.

#### C. Don't evict — archive — implemented

Checkpoint files remain permanently on disk and in paired league state. Each inference service uses a bounded LRU model cache (`league_size`, default 12); selecting an evicted member reloads it lazily. PFSP (B) decides which archived policy to play without unbounded model residency.

### Tier 2 — moderate effort, real payoff

#### D. KL anchor to a past snapshot

Add a small KL penalty in the PPO loss against a slowly-updated reference policy (e.g. a checkpoint from 100k steps ago, or an EMA of past policies). Directly penalizes catastrophic drift. One extra term in the loss; standard technique. Hyperparameter-sensitive — too strong and the agent can't improve.

#### E. Behavior fingerprints per checkpoint

Cheap summary stats per league member: tera-turn distribution, switch rate, lead species distribution, average battle length, opening-move distribution. Surface in logs. If the current agent's fingerprint is cycling around in fingerprint-space over training time, we can see it even without head-to-head eval. Also enables diversity-aware retention ("don't evict a checkpoint whose style no one else in the pool has").

#### F. ELO / TrueSkill matchmaking

Maintain a rating per checkpoint (including current). Sample opponents within a band around current rating — a curriculum effect where the agent always plays appropriately-matched opponents. Loses the "hardest matchups" focus of PFSP but gains smoother learning signal. Can combine with PFSP: ELO-gate to a band, then PFSP-sample within the band.

### Tier 3 — larger engineering lift

#### G. One exploiter agent

A second training run whose single objective is to beat the current main agent. When it wins consistently over some window, snapshot it into the main league and reset it to continue searching. This is the core idea from AlphaStar, scoped to one exploiter instead of a population. Roughly doubles compute but systematically surfaces the main agent's blind spots. Biggest effort, biggest payoff.

#### H. Opponent-team-conditioned sampling

Currently opponent *checkpoint* and opponent *team* are sampled independently. Instead, maintain `(checkpoint, team)` pairs, and sample teams the opponent checkpoint was historically strongest with. Makes each league member an actual adversary rather than a generic opponent forced to play whatever team pool draws.

#### I. Targeted best-response fine-tuning

Every N updates, identify the one historical checkpoint the current agent has the worst win rate against. Run a short targeted PPO warmup against only that opponent before returning to normal league sampling. Explicit anti-forgetting, more aggressive than (D).

---

## 5. Guardrails

- **Do not train against weak scripted bots.** That would optimize for their
  exploitable quirks and does not scale with the game's strategy space. Keep
  them as fixed, out-of-training diagnostic probes only. New strategic pressure
  should come from self-play history or learned best responses.

---

## 6. Suggested order

Given the current evidence and the single-GPU constraint:

1. **Automate A.** Keep the matrix, fixed bots, preview conditioning, and
   roster-stratified metrics as quality gates. Add a legal held-out team
   population once one can be supplied without randomly assembling invalid or
   incoherent teams.
2. **Fix the measured Tera, preview, and archetype failures.** These dominate current quality more clearly than league sampling does.
3. **Implemented: B + C.** Retain every snapshot, bound resident models with an LRU cache, and sample distinct groups with discounted hard-PFSP plus uniform coverage.
4. **Measure the new league.** Compare online payoff support with mirrored frozen-checkpoint matrices; PFSP cannot create strategies absent from the main trajectory.
5. Revisit **D** and Tier 3 only if simpler measured corrections leave a league-specific failure.

The evaluation harness now tells the truth: the current policy is improving, then plateauing, with serious shared strategic blind spots.
