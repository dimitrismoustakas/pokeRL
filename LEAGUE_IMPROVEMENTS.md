# League Self-Play: Improvement Options

A design-space overview for improving the self-play league beyond the current rolling-window setup. This is a reference for *what the options are* and *what the tradeoffs look like*, not a committed plan. See [pokerl/training/league.py](pokerl/training/league.py) and [§8 of MODEL_EXPLANATION.md](MODEL_EXPLANATION.md#8-league-self-play) for the current implementation.

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

- Rolling pool of `league_size = 12` checkpoints, snapshot every `league_update_interval = 10` PPO updates. Pool spans at most ~120 updates of history; oldest evicted on overflow.
- Sampling: 70% from the recent half, 30% from the older half. Uniform within each half. No per-opponent tracking.
- No evaluation harness that could detect cycling if it were happening.

So the primary gap is **diagnostic**: we don't currently have the data to tell whether the naive setup is working.

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

Before changing the training, diagnose the actual problem. Keep a small **anchor set** of frozen checkpoints (e.g. snapshots at 100k, 500k, 1M, 2M steps) permanently on disk, *outside* the rolling pool. Periodically run N games of current agent vs each anchor, log win rates over training time.

- If win rate vs an old anchor flatlines or *decreases*, cycling is real.
- If it keeps rising, the current setup is working and further changes may be premature.

Roughly a day of work. Tells us whether anything else in this document is worth implementing.

#### B. Prioritized Fictitious Self-Play (PFSP)

Track per-opponent win rate with an EMA. Replace the 70/30 recent/old rule with weighted sampling over the full pool:

- `(1 − p)^α` — train against what we're losing to (AlphaStar's "hard" variant).
- `p · (1 − p)` — train against coin-flip matchups (variance-focused, more conservative).

If an old checkpoint starts beating the current agent, it gets sampled more, and the agent is forced to re-learn its counter. Small code change, directly addresses cycling.

#### C. Don't evict — archive

Checkpoint files are small. Keep them all on disk, but tier what's *resident* in the inference service: all of the last ~20, every 5th of the next ~50, every 25th beyond that. PFSP (B) handles "which to actually play." This preserves strategic diversity without unbounded memory. Pair well with (B): PFSP is only as useful as the pool it samples from.

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

## 5. Explicitly not pursued

- **Heuristic bots as training partners.** Adding random/max-power/scripted opponents to *training* would bias the policy toward beating specific weak strategies rather than generalizing. These are fine as additional rows in the eval harness (A) — as sanity checks on absolute skill — but not in the opponent distribution.

---

## 6. Suggested order

Given the constraints (single developer, single GPU, sparse rewards making every training hour expensive):

1. **A — Evaluation harness.** Diagnostic prerequisite. Without it we're guessing whether any of the below is solving a real problem.
2. **B — PFSP sampling.** Where the literature most clearly points. Directly targets cycling.
3. **C — Archive beyond the rolling window.** Makes (B) materially better.
4. **D — KL anchor.** Revisit once (A) gives us a measured picture of drift.
5. Tier 3 is an "if I still see cycling after 1–4" question.

The honest starting move is (A): it's not the most exciting, but it's the one that tells the truth.
