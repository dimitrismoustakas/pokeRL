# Principled training rework

This is the implementation plan following the independent review of the
440-update run. Each step must be reported before work proceeds to the next
one. The original trainer checkpoint, paired league state, evaluation outputs,
and archived league checkpoints remain immutable reference artifacts.

Current status: Step 1 is complete after implementation, real-checkpoint proof,
performance measurement, adversarial review, and the full repository suite.
Step 2 has not been implemented.

## Step 1: exact team-preview policy

Scope:

- replace the index-biased increasing-index sampler with an exact distribution
  over every legal bring subset and ordered lead pair;
- make the distribution permutation-equivariant with respect to roster order;
- vectorize scoring and sampling, with a fast path for live sampled inference;
- compute exact preview entropy and expose the exact joint distribution needed
  for later KL diagnostics;
- preserve the useful parameterization where possible, but bump the model
  architecture signature because policy semantics change;
- add explicit migration tooling before any version-3 checkpoint is reused.

Correctness checks:

- uniform distribution under zero scoring logits;
- normalization over all legal actions;
- semantic roster permutations produce correspondingly permuted probabilities;
- sampled and externally supplied actions are legal and have exact log
  probabilities;
- deterministic inference returns the global joint argmax;
- masks and non-default team/pick sizes remain supported;
- old checkpoints cannot silently resume under the new semantics.

Performance checks:

- benchmark current and corrected heads at live inference batch sizes on CPU and
  CUDA when available;
- live sampled inference must avoid enumerating all lead choices and should be
  at least as fast as the retired sequential implementation after optimization;
- benchmark the exact-statistics training path separately and report its share
  of end-to-end update time rather than hiding extra work.

Out of scope for this step: PPO scheduling, rollout ownership, exploiters,
league sampling, optimizer selection, and admission.

Measured Step 1 result:

- exact uniformity, normalization, mask legality, full-agent roster
  equivariance, opponent-roster invariance, fast/full scorer equivalence,
  supplied-action likelihood, entropy, global argmax, extreme-logit legality,
  and first-step learnability tests pass;
- live CUDA head inference at batch 72 improved from about 8.6 ms to 4.4 ms;
- full-agent CUDA preview inference is about 23 ms at batch 72;
- a batch-365 full-agent preview forward/backward takes about 139 ms and peaks
  at about 421 MiB allocated on the reference GPU;
- the complete current preview-PPO routine on 1,356 real-format samples, four
  epochs, and 16 optimizer steps took a 1.996 s synchronized median and peaked
  at 431 MiB allocated on the 4 GiB reference GPU. For scale only, the
  historical run's medians were 14.989 s in its combined old-preview/battle
  PPO section and 128.149 s per update, so the isolated exact-preview routine
  is about 13.3% and 1.6% of those respective times. This is not a controlled
  old-versus-new PPO comparison;
- exact evaluation now obtains all 180 probabilities in one model forward per
  roster batch instead of repeating each observation 180 times;
- architecture version 4 rejects version-3 checkpoints. The version-3
  conversion preserves compatible battle, auxiliary, and preview feature
  tensors; zeros only the two preview-policy output layers; resets the stale
  preview value head; starts fresh optimizer/counters; and records compact
  source provenance in the checkpoint;
- the actual 440-update trainer was converted and strictly verified in a
  temporary artifact: 229 tensors copied, 20 reset/zeroed, exact entropy
  `ln(180)`, and maximum probability `1/180`. The original trainer and its
  45-member version-3 league were not modified or converted;
- a production `--initialize-from` path can seed a fresh main run from the
  converted checkpoint. It requires an explicit empty checkpoint directory,
  creates a new step-zero league, and refuses on-disk league checkpoint
  collisions;
- final validation: 574 tests pass and Ruff lint passes. The repository-wide
  Ruff format check still reports 21 pre-existing files; files that were
  formatted before this step remain formatted, and the new migration files
  are formatted.

## Step 2: one preview-and-battle PPO transaction

The exact design will be presented before implementation. It must:

- treat preview as an action-time transition with behavior log probability,
  value, policy version, and recurrent boundary recorded when the action is
  taken;
- retain episodes that cross a rollout/update boundary instead of discarding
  them when their terminal reward arrives;
- optimize preview and battle objectives from the same behavior snapshot in one
  explicitly weighted transaction;
- prevent preview-only gradients from silently consuming the battle clipping
  budget;
- report exact preview KL, battle pre-update KL, per-objective shared-gradient
  norms/cosines, clipping incidence, and optimizer-step exposure.

Step 2 is not satisfied by merely moving the existing preview update after
battle PPO or by suppressing stale-sample warnings.

## Step 3: standalone exploiter proof

- use a frozen, explicitly versioned target population derived from the
  reference run;
- train fresh main-exploiter learners outside a main self-play run;
- evaluate saved checkpoints in mirrored roles on locked and genuinely held-out
  team schedules;
- require statistically credible improvement over the frozen target before
  attempting admission;
- then run one real league-exploiter phase against a complete 3--5 member
  frozen cross-play population.

## Step 4: league evidence and PFSP experiments

- replay the saved assignment/outcome history under alternative estimators;
- compare exact group-inclusion marginals, newcomer exposure, uncertainty, and
  effective support;
- benchmark more opponent groups before changing live sampling;
- keep training, frozen evaluation, and admission as separate fail-closed
  stages.

## Step 5: optimizer and hyperparameter study

- serialize and validate the complete training/optimizer protocol first;
- compare strict Adam continuation, fresh Adam, tuned AdamW, and later a named
  hybrid Muon/AdamW partition from identical weights and simulator budgets;
- tune learning rate, decay, schedule, entropy, clipping/update epochs, and
  `gamma` in controlled stages rather than changing them together;
- confirm any final-seeded winner from an earlier checkpoint or initialization.

## Secondary correctness work

- correct the opponent-intent documentation and decide auxiliary-only versus a
  public-prediction policy adapter from controlled evidence;
- distinguish legitimate same-turn forced-switch requests from pathological
  repeated requests;
- make evaluation provenance, confidence, and held-out status explicit.
