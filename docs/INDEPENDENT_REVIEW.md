# Independent review brief

Updated: 2026-07-18.

This file is the current decision brief for the training system. It replaces
the earlier review state; it is not a chronological diary. Code, tests, and
immutable run summaries remain authoritative.

## Decision summary

The 440-update reference run learned useful transferable strength, but it did
not justify continuing the same training procedure unchanged. Two preview
problems were real and have now been corrected:

1. The old preview action construction was index-biased rather than an exact
   permutation-equivariant policy over semantic team choices.
2. Preview PPO updated the shared model before battle PPO consumed behavior
   collected by the old model. Long episodes also lost preview training because
   samples were owned only when a terminal result arrived.

The current code now has:

- an exact policy over all 180 legal bring-four and ordered-lead choices;
- a fast live sampler that does not enumerate all conditional lead choices;
- `gamma=1`, matching undiscounted win probability;
- action-time ownership of every preview decision;
- direct truncated lambda-1 preview returns, with exact terminal outcomes or
  one battle-value bootstrap at the rollout boundary;
- one combined preview-and-battle PPO transaction with 16 optimizer steps per
  update, rather than 16 preview steps followed by 16 battle steps;
- exact preview `KL(old || current)` from one frozen pre-update behavior
  distribution;
- trainer-private selected-team labels installed at episode start and retained
  across rollout boundaries; and
- no migration, stale-replay, sample-cap, or separate-preview-update path.

The uncompromised joint update fits and runs on the 4 GiB reference GPU. Step 2
is therefore complete. The next work is not another long main run: first prove
that a standalone exploiter learns a useful counter-policy, then strengthen
league evidence, and only later run controlled optimizer experiments.

## Objective and non-negotiable boundaries

pokeRL trains a recurrent actor-critic for Generation 9 VGC doubles through
the real Pokemon Showdown simulator.

- Reward remains the sparse terminal result: win `+1`, loss `-1`, and the draw
  result. HP, damage, knockouts, board heuristics, and bot imitation are not
  reward.
- Inference receives only information legitimately available to a player.
  Exact packed teams and simulator decisions may be trainer-only targets, never
  policy observations.
- Fixed bots are evaluation probes, not a training curriculum.
- Evaluation, best-response training, population construction, and admission
  remain separate explicit stages.
- Active research has one production path. Old checkpoints are evidence, not a
  reason to retain conversion or compatibility branches.

## Current policy and training semantics

### Observation and representation

The policy observes both public rosters, revealed set information, active
Pokemon, HP, status, boosts, field state, legal actions, recent public protocol
events, and explicit public mechanics such as first active turn, consecutive
Protect state, active-turn count, and preceding public move resolution.

The model combines Pokemon and move embeddings, set-attention team encoders,
structured field/event/mechanics encoders, an LSTM, and a normalized residual
path around the LSTM. The residual carries the current public board directly;
the recurrent state is responsible for history not fully present in the
current observation.

### Team preview

Preview factorizes one exact 180-action distribution into:

1. one of 15 four-of-six bring subsets; and
2. one of 12 ordered lead pairs conditional on that subset.

Subset scoring is symmetric in selected members, so roster-slot permutations
only permute semantic action probabilities. Live stochastic inference scores
all bring subsets and only the sampled subset's 12 leads. It returns the picks,
selected-action log probability, value, and policy version.

After rollout collection, before any optimizer step, the trainer reevaluates
all collected preview observations together under the unchanged parameters.
That one no-gradient pass freezes the complete old 180-action distributions
for exact KL. This keeps exhaustive work off the reset critical path.

### Preview return

Preview is the zero-reward first transition into the first battle state.
Production uses `gamma=1`.

- If the episode terminates inside the rollout, the preview target is the exact
  sparse terminal outcome.
- If it crosses the rollout boundary, the target contains all observed rewards
  and one bootstrap from the boundary battle value.
- Preview uses this direct truncated lambda-1 target. It does not inherit the
  battle GAE lambda of `0.95`.
- A preview created by the reset after the final rollout action is a valid
  start at `k=T` and bootstraps from that reset state.

Each preview action is trained once in the rollout where it was taken. It is
not replayed when a long battle later terminates.

### Joint PPO transaction

Battle and preview advantages are normalized together. Policy, value, and
entropy terms are summed and scaled by the complete rollout's number of
decisions, so one preview decision and one battle decision have the same base
weight. Preview opponent-bring and opponent-lead losses keep their explicit
auxiliary coefficients.

Each epoch independently shuffles preview samples while battle minibatches
retain complete environment sequences for recurrent unrolling. Every paired
minibatch produces:

1. one combined loss;
2. one backward pass;
3. one global gradient-norm clip; and
4. one Adam step.

With four epochs and four minibatches this is 16 optimizer steps. Preview can
no longer move shared parameters before the first battle PPO evaluation.

### Auxiliary heads

- The hidden-information head predicts exact selected-team item, ability, Tera
  type, and moves under masks. A learned belief residual is consumed by policy
  and value heads.
- The public-consequence model predicts the next public resolution. Compact
  predictions are detached before entering the policy.
- The opponent-intent head predicts semantic opponent decisions but remains
  auxiliary-only.

Keeping opponent intent auxiliary-only is deliberate for now. A policy adapter
would be legal if it consumed only predictions computed from public inputs, but
it would be a new architectural claim. It should be tested against an
auxiliary-only control after the present trainer is stable. Exact intent labels
must never enter inference.

## Evidence

### The 440-update reference run

Reference artifacts:

```text
checkpoints/selfplay_public_state_v3_main_440_20260717/final.pt
logs/selfplay_public_state_v3_main_440_20260717.summary.json
logs/evaluation/selfplay_public_state_v3_main_440_20260717/
```

Verified run totals:

- 440 updates and 4,055,040 battle-policy decisions;
- 653,915 completed battles;
- 14 h 54 min wall time;
- 75.59 end-to-end steps/s;
- no retained-battle growth, CUDA-memory leak, non-finite update report,
  traceback, or incomplete checkpoint.

Mirrored sampled-policy evaluation showed substantial strength over
initialization on both training and held-out teams. Comparisons against updates
110, 220, and 340 did not establish sustained improvement late in the run.
This supports a local plateau, not a claim that all learning stopped.

The old preview policy showed useful behavior but excessive concentration and
limited opponent-dependent bring-four diversity. More importantly, the old
trainer measured substantial preview clipping and nonzero battle-policy drift
before battle PPO began. That was a policy-version error in the update design,
not merely an aggressive coefficient.

The old checkpoint is still a valuable evaluation and population artifact. It
is not resumable by the corrected architecture/trainer, and no converter is
provided. The next main model will be trained from scratch.

### Step 1 correctness and speed

Tests cover exact uniformity, normalization, legal masking, semantic roster
permutations, sampled and supplied-action likelihoods, entropy, global argmax,
extreme logits, and first-step learnability.

On the reference GPU:

- live preview-head inference at batch 72 improved from about 8.6 ms to
  4.4 ms;
- full-agent preview inference at batch 72 is about 23 ms; and
- exact evaluation obtains all 180 probabilities in one forward rather than
  repeating an observation 180 times.

The fast path's selected log probability was directly compared with exhaustive
reevaluation for masked rosters and matched exactly.

### Step 2 real update

The corrected trainer completed one full 72-environment, 128-step CUDA update:

| Measurement | Joint Step 2 |
| --- | ---: |
| Rollout | 154.72 s |
| PPO section | 22.56 s |
| Update wall time | 177.95 s |
| Preview decisions | 877 |
| Terminal preview targets | 805 |
| Boundary-bootstrapped targets | 72 |
| Joint optimizer steps | 16 |
| CUDA allocated after update | 0.307 GB |
| CUDA reserved after update | 3.618 GB |

The action-time versus pre-update selected preview log-probability difference
was `1.28e-7` mean and `9.54e-7` maximum. Exact preview KL remained finite.
There was no non-finite skip or out-of-memory fallback.

A controlled run of the exact pre-Step-2 commit with the same command measured
157.84 s rollout, 18.98 s PPO, and 177.22 s update wall time. The joint PPO
section was 3.58 s slower in this single comparison, while complete update wall
time differed by 0.73 s (0.41%). The joint run also trained the 72 in-flight
preview decisions that the terminal-only baseline could not own.

This establishes feasibility and exposes the local cost. It is not a powered
throughput comparison, and it does not justify claiming that Step 2 is faster.
It also gives no reason to introduce a second memory-saving update path: the
single combined backward fits, and simulator time still dominates.

## League and exploiter assessment

### League

The current league is a permanent checkpoint archive with a bounded resident
model cache. Opponent groups share frozen checkpoints for inference batching.
Sampling combines discounted payoff evidence, a PFSP hardness transform, and a
uniform coverage component.

The implementation is coherent, but the evidence is not yet sufficient to
claim that two opponent groups, the current evidence half-life, exponent, or
uniform fraction are optimal. Required next checks include:

- replaying saved assignments/outcomes under alternative estimators;
- exact inclusion marginals and newcomer exposure;
- effective support and uncertainty;
- controlled tests with more opponent groups; and
- mirrored evaluation separated from live sampling decisions.

### Exploiters

The repository can define immutable populations and best-response phases, but
a written checkpoint is not evidence of exploitation.

Before admitting or scheduling exploiters in a main run:

1. derive an immutable target population from the reference league;
2. train a fresh main exploiter outside main self-play;
3. evaluate it in mirrored roles on locked and genuinely held-out schedules;
4. require credible improvement over the frozen target; and
5. repeat with a league exploiter against a complete 3--5 member cross-play
   population.

Only after this standalone proof should orchestration frequency and admission
rules be decided.

## Optimizer and hyperparameter work

The current optimizer is fixed-learning-rate Adam with epsilon `1e-5`. Neither
the optimizer family nor its important hyperparameters were tuned. There is no
basis for calling this choice optimal.

The later controlled study should compare:

1. current Adam as the reproducibility baseline;
2. tuned AdamW with explicit decay and parameter groups;
3. a named hybrid using Muon only on appropriate matrix parameters and AdamW
   on embeddings, vectors, normalization, and unsupported parameters;
4. learning-rate schedules;
5. entropy, clipping, epochs, and minibatches; and
6. repeated seeds or balanced repeated runs with equal simulator budgets.

`gamma=1` is now the principled production default because the stated objective
is undiscounted win probability. Optimizer experiments should not silently
change it at the same time.

Training loss alone is not a selection criterion. Compare mirrored strength,
held-out robustness, preview diversity, mechanic-specific behavior, stability,
wall time, and memory.

Primary references:

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [AlphaStar league and PFSP](https://doi.org/10.1038/s41586-019-1724-z)

## Remaining correctness and evaluation work

- Decide whether the public-only opponent-intent prediction merits a policy
  adapter through a controlled auxiliary-only versus adapter experiment.
- Distinguish legitimate same-turn forced-switch requests from pathological
  repeated requests in diagnostics.
- Expand published, locally validated teams and split them before use into
  training, development, and locked test manifests.
- Report per-team and per-archetype results, mirrored roles, confidence
  intervals, preview diversity, and mechanic-specific behavior.
- Audit production fallbacks and observation saturation. Remove silent neutral
  substitutions where invalid state should fail.
- Run the existing opt-in shared-gradient diagnostic on a short update before
  changing auxiliary routing or coefficients.

## Next order of work

1. Review and commit the completed Step 2 implementation.
2. Produce the standalone main-exploiter proof against the frozen reference
   league.
3. Produce the league-exploiter proof against a complete small population.
4. Replay and stress-test league/PFSP evidence and opponent-group choices.
5. Decide whether opponent intent should remain auxiliary-only.
6. Run controlled optimizer and hyperparameter experiments.
7. Start a new long main run only after those decisions are explicit.

## Source map

- `pokerl/config.py`: PPO, exploration, auxiliary, and league defaults
- `pokerl/model/heads.py`: exact preview, battle, value, and auxiliary heads
- `pokerl/training/preview.py`: fast live preview batching and action-time data
- `pokerl/training/rollout.py`: episode-start ownership, returns, and labels
- `pokerl/training/ppo.py`: the joint PPO transaction
- `pokerl/training/league.py`: archive, payoff evidence, PFSP, inference cache
- `pokerl/training/exploiter.py`: best-response phase definitions
- `pokerl/training/population.py`: immutable populations and meta-solver
- `pokerl/env/showdown_env.py`: simulator lifecycle and preview records
- `scripts/train.py`: collection, update, version advancement, and refresh order
