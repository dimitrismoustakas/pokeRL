# Principled training rework

This is the implementation plan following the independent review of the
440-update run. Each step must be reported before work proceeds to the next
one. The corrected trainer will be trained from scratch; the earlier run is
evidence for diagnosis and comparison, not a compatibility target.

Current status: Steps 1 and 2 are complete. Step 2 passed an integrated
correctness review, the full repository suite, and a real 72-environment CUDA
update on the 4 GiB reference GPU. No work on Step 3 has started.

## Step 1: exact team-preview policy

Scope:

- replace the index-biased increasing-index sampler with an exact distribution
  over every legal bring subset and ordered lead pair;
- make the distribution permutation-equivariant with respect to roster order;
- vectorize scoring and sampling, with a fast path for live sampled inference;
- compute exact preview entropy and expose the exact joint distribution needed
  for later KL diagnostics;
- bump the model architecture signature because policy semantics change.

Correctness checks:

- uniform distribution under zero scoring logits;
- normalization over all legal actions;
- semantic roster permutations produce correspondingly permuted probabilities;
- sampled and externally supplied actions are legal and have exact log
  probabilities;
- deterministic inference returns the global joint argmax;
- masks and non-default team/pick sizes remain supported;
- incompatible checkpoints fail normal strict loading.

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
- final validation after removing the now-unneeded migration path: 571 tests
  pass and Ruff lint passes. The repository-wide Ruff format check still
  reports 21 pre-existing files.

## Step 2: one preview-and-battle PPO transaction

Implemented semantics:

- `gamma=1` is the production objective, matching undiscounted win
  probability. Battle advantages retain GAE lambda `0.95`.
- Preview is the zero-reward first transition of an episode. Its observation,
  chosen action, selected-action behavior log probability, value, and policy
  version are owned when the action is taken.
- Every preview beginning in a rollout is trained once. A terminal inside the
  rollout supplies the exact sparse outcome; otherwise a direct truncated
  lambda-1 return bootstraps once from the battle value at the rollout
  boundary. This includes starts at the final boundary `k=T`.
- Long battles are no longer excluded or replayed at terminal. Their preview
  action is trained in the rollout where it occurred, while exact
  trainer-private selected-team labels persist across later rollout
  boundaries.
- Battle and preview advantages are normalized together. Their policy, value,
  and entropy sums are weighted by the number of decisions in the complete
  rollout.
- Each paired recurrent-battle and independently shuffled preview minibatch
  produces one loss, one backward pass, one global gradient clip, and one Adam
  step. Four epochs by four minibatches therefore means 16 optimizer steps,
  not 16 preview steps followed by 16 battle steps.
- Live preview keeps the fast factorized sampler and computes only the sampled
  action log probability and value. After collection and before the first
  optimizer step, one batched no-gradient evaluation freezes the complete
  180-action old distribution for exact `KL(old || current)`.
- The terminal-only preview updater, stale-sample filter, optional sample cap,
  separate preview minibatch setting, and update-order wrapper were removed.
  There is one production update path and no checkpoint conversion path.
- Exact preview KL, sampled approximate KL, clipping, value clipping,
  boundary/terminal counts, return horizon, and joint optimizer-step exposure
  are always logged. The existing expensive per-objective gradient diagnostic
  remains opt-in for short diagnostic runs rather than burdening every update.

Correctness evidence:

- focused lifecycle, return, joint-objective, policy-version, exact-KL, and
  fast-sampler suites pass;
- the fast sampled log probability equals exhaustive reevaluation exactly in a
  direct masked-roster test;
- the first real update trained 877 preview decisions: 805 terminal and 72
  boundary-bootstrapped;
- action-time versus pre-update preview log-probability disagreement was
  `1.28e-7` mean and `9.54e-7` maximum, consistent with floating-point noise;
- the update made exactly 16 joint optimizer steps and completed without a
  non-finite skip.

Feasibility and timing:

- the full 72-environment, 128-step CUDA update completed on the 4 GiB Quadro
  T2000: rollout 154.72 s, PPO 22.56 s, and update wall time 177.95 s;
- CUDA allocator state after the update was 0.307 GB allocated and 3.618 GB
  reserved, so the uncompromised combined graph fit without an out-of-memory
  fallback;
- a controlled run of the exact pre-Step-2 `HEAD` under the same command took
  157.84 s rollout, 18.98 s PPO, and 177.22 s update wall time. The joint PPO
  section was 3.58 s slower in this one comparison, while total update time
  differed by 0.73 s (0.41%) and the joint run trained the 72 in-flight
  previews that the terminal-only baseline omitted. This is a feasibility A/B,
  not a statistically powered throughput claim.

The chosen implementation remains the single combined-backward reference.
There is no evidence from the end-to-end run that a memory-motivated
two-backward accumulation path is necessary.

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
- tune learning rate, decay, schedule, entropy, clipping, and update epochs in
  controlled stages rather than changing them together;
- keep `gamma=1` fixed unless an explicit experiment changes the objective away
  from undiscounted win probability;
- confirm any final-seeded winner from an earlier checkpoint or initialization.

## Secondary correctness work

- correct the opponent-intent documentation and decide auxiliary-only versus a
  public-prediction policy adapter from controlled evidence;
- distinguish legitimate same-turn forced-switch requests from pathological
  repeated requests;
- make evaluation provenance, confidence, and held-out status explicit.
