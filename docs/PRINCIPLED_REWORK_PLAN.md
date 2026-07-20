# Principled training rework

Updated: 2026-07-21.

The production path now targets architecture version 5 and starts from random
initialization. Earlier checkpoints remain diagnostic evidence; they are not a
compatibility target and are not resumed.

## Completed foundation

### Exact team preview

- one exact permutation-equivariant distribution over all 180 legal choices;
- a vectorized live sampler that avoids exhaustive lead scoring;
- exact entropy, deterministic argmax, and full-distribution evaluation;
- direct tests for normalization, permutation behavior, masking, sampled log
  probabilities, and learnability.

Measured live CUDA preview-head inference at batch 72 improved from about
8.6 ms to 4.4 ms. The exact training path fits comfortably on the 4 GiB
reference GPU.

### Joint preview and battle PPO

- `gamma=1` for undiscounted win probability;
- every preview owned at action time and trained once;
- exact terminal or single boundary-bootstrap preview targets;
- one combined preview-and-battle loss, backward pass, clip, and optimizer step
  per paired minibatch;
- 16 optimizer steps for four epochs by four minibatches;
- one frozen pre-update preview distribution for exact KL;
- no stale replay, separate preview updater, sample cap, or migration path.

A real 72-environment CUDA update and a fresh 10-update run established that
this path fits, trains boundary-crossing previews, and preserves one behavior
policy version. The 10-update run is diagnostic evidence only and is not a
checkpoint to resume.

### Resume-invariant exploration

Tera exploration decays over an explicit absolute number of learner updates.
Its start, end, and decay length are stored in the trainer checkpoint and
remain unchanged across resumes. The planned 450-update trajectory uses 113
decay updates.

### Optimizer support and measurement

- selectable Adam and AdamW;
- explicit AdamW decay/no-decay grouping;
- complete optimizer and PPO settings in run summaries;
- pre-clip global gradient norm and clipped-step fraction;
- opt-in per-objective memory-gradient norms and cosines.

The version-4 experiments did not prove AdamW better than Adam at update 80.
AdamW `2e-4`, weight decay `0.05` is the working setting for the corrected run,
not a final optimizer conclusion.

### Live request semantics

- actions use Showdown `request_index` rather than compact list position;
- observations expose the exact live move ID in each request slot;
- disabled slots remain holes;
- locked moves, transformed moves, recharge, and Struggle keep their live
  semantics;
- invalid and duplicate request indices fail loudly;
- the exact public Perish Song countdown is retained in the existing scalar.

This changes same-shaped model inputs, so architecture version 5 intentionally
rejects all earlier checkpoints. No converter is planned.

## Immediate run sequence

### 1. Completed ten-update diagnostic

The fresh AdamW diagnostic completed cleanly: all joint PPO updates were
accounted for, preview behavior-policy likelihoods agreed, gradients were
finite, and memory and throughput were practical. It used a horizon-derived
Tera exploration decay of three updates, so it is a standalone diagnostic and
must not be resumed.

Its measurements established that:

- every update completes with 16 joint optimizer steps and no non-finite skip;
- preview action-time and pre-update likelihoods agree;
- preview exact KL, sampled KL, and clipping remain finite;
- per-objective gradient norms and pairwise cosines are finite and identify any
  material opposition on shared memory parameters;
- pre-clip global norms explain how often and by how much the `0.5` cap binds;
- rollout/PPO time, CUDA memory, and simulator throughput remain practical.

### 2. Start the 450-update trajectory and inspect at update 80

Start fresh with AdamW and an explicit 113-update Tera exploration decay. Stop
first at update 80 and evaluate against initialization and intermediate
snapshots with mirrored sampled schedules. Also inspect exact preview
distributions and mechanic-specific action behavior. Do not treat training loss
or self-play win rate as sufficient evidence.

### 3. Continue to update 450

If update 80 learns normally, resume the same run for 370 more updates. The
checkpointed 113-update exploration schedule remains fixed. Compare updates
100, 250, 350, and 450 on held-out teams to locate any plateau. Retain the
resulting league for PFSP and exploiter work.

## Work after the corrected main run

### League evidence

- replay saved assignments and outcomes under alternative payoff estimators;
- measure inclusion marginals, newcomer exposure, uncertainty, and effective
  support;
- benchmark additional opponent groups before changing live sampling;
- keep evaluation and admission separate from training.

### Standalone exploiters

- derive an immutable target population from the corrected league;
- train a fresh main exploiter outside the main run;
- evaluate it in mirrored roles on training and genuinely held-out schedules;
- require credible improvement over the frozen target;
- then train a league exploiter against a complete three-to-five-member frozen
  cross-play population.

Only after those tests should exploiter frequency or admission be added to a
main run.

### Optimizer and hyperparameter study

- implement a named Muon/AdamW parameter partition and compare it from scratch;
- use AdamW for embeddings, vectors, normalization, and unsupported parameters;
- decide the gradient cap from measured norms and objective interactions;
- tune learning rate, decay, entropy, epochs, minibatches, and schedules in
  controlled stages;
- confirm final choices with balanced evaluation or repeated runs.

Opponent intent remains auxiliary-only until a stable corrected baseline makes
a controlled policy-adapter comparison meaningful.

## Artifact policy

The findings from retired runs are consolidated in `INDEPENDENT_REVIEW.md`.
Their ignored checkpoints, summaries, console logs, and local evaluation
artifacts have been removed. Keep the next production run's summary and
evaluation outputs until the optimizer, plateau, and exploiter decisions are
complete. Obsolete population and phase specifications remain removed so they
cannot be mistaken for current inputs.
