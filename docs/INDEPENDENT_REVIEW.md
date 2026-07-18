# Independent review brief

Updated: 2026-07-18.

This is a self-contained decision brief for a reviewer who has not seen the
development discussion. It replaces the previous status file rather than
adding another chronological project log.

The brief distinguishes four kinds of statements:

- **Verified** means directly supported by current code, tests, or saved run
  artifacts.
- **Interpretation** means a reasoned reading of the evidence, not a proven
  causal result.
- **Proposal** means an idea that has not been accepted or implemented.
- **Unknown** means the existing evidence is insufficient.

No source or training behavior was changed while preparing this review.

## Review questions

The independent reviewer is asked to challenge, in particular:

1. Is the current opponent-sampling design a sound implementation of
   prioritized fictitious self-play for this local training regime?
2. How should a newly archived policy acquire enough evidence to be sampled
   intelligently without an arbitrary recency bonus?
3. Should team preview and battle learning become one coordinated policy
   update, and what is the cleanest objective?
4. Are the auxiliary tasks helping the deployed policy or dominating/conflicting
   with sparse-reward policy learning?
5. Which optimizer, discount, exploration, and update-schedule experiments are
   scientifically justified?
6. Which silent fallbacks or unobserved safety mechanisms should be removed,
   made fail-loud, or instrumented before another long run?
7. What evidence should a dedicated exploiter have to produce before it is
   allowed to affect main-policy training?

## Project objective and constraints

pokeRL trains a policy for Generation 9 VGC doubles through the real Pokemon
Showdown simulator. The long-term objective is a generally strong self-play
agent, not a policy specialized to the diagnostic bots in this repository.

The intended design constraints are:

- Use sparse terminal outcomes: win `+1`, loss `-1`, and the terminal draw
  result. Do not shape reward with HP, damage, knockouts, board heuristics, or
  bot imitation.
- Train primarily through scalable self-play. Fixed bots are evaluation probes,
  not a curriculum.
- Give inference only public information available to a real player. Exact
  private teams and simulator decisions may be trainer-only labels, never
  policy inputs.
- Improve throughput without reducing model capacity, policy quality, legal
  action coverage, or game fidelity.
- Prefer principled architecture and sampling changes over special cases.
- Do not preserve backward compatibility during active research when it would
  complicate the one production path.
- Do not admit an exploiter because its checkpoint was written successfully.
  Demonstrate that it learned a useful counter-policy first.
- Every mechanism that clips, skips, masks, truncates, substitutes, or falls
  back should either be an explicit part of the algorithm or be observable.
  Silent compatibility behavior is suspect.

## Current system

### Policy and observation

The deployed model is a recurrent actor-critic. It receives both public
rosters, revealed set information, active Pokemon, HP, status, boosts, field
state, legal actions, recent public protocol events, and explicit public
mechanics such as first active turn, consecutive Protect state, active-turn
count, and the preceding public move resolution.

The representation has:

1. learned Pokemon and move embeddings;
2. set-attention encoders for each team;
3. structured field, event, and active-mechanics encoders;
4. an LSTM for temporal information; and
5. a normalized residual path that sends the current encoded public state
   around the LSTM.

The residual is important: the LSTM is not required to memorize the complete
current board. It carries history that is not fully present in the current
observation. There is no current evidence that replacing the LSTM alone is the
highest-value change. There is evidence that the inputs and losses around the
shared representation need closer examination.

### Team preview

At team preview, the policy sees both public six-Pokemon rosters and defines a
distribution over all 180 legal ordered choices:

- choose four of six Pokemon; and
- choose an ordered lead pair from those four.

Evaluation samples from this distribution. Exact diagnostics may also report
the **most-probable choice**, meaning only the single action with the largest
policy probability. It is not a heuristic choice and is not claimed to be
optimal.

The preview head shares the spatial encoder and LSTM with the battle policy. It
also has a preview value head and opponent bring/lead prediction losses.

### Battle policy and auxiliary tasks

The battle head chooses:

1. the first active slot's action;
2. the second slot's action conditioned on the first; and
3. whether to Terastallize and which eligible slot, conditioned on both.

Legal masks prevent impossible joint actions.

Three trainer-only auxiliary families are present:

- hidden opponent item, ability, Tera type, and move-set prediction;
- semantic opponent action-intent prediction; and
- prediction of the public consequence of selected actions.

Compact belief predictions are consumed by the policy. Public-consequence
predictions are detached before being fed back to the actor so that the sparse
policy objective cannot make the predictor invent convenient beliefs. Exact
private labels remain outside the inference path. These losses do not alter
reward.

### Optimization

The current defaults are:

| Setting | Value |
| --- | ---: |
| Optimizer | Adam |
| Learning rate | `2.5e-4`, fixed |
| Adam epsilon | `1e-5` |
| Discount `gamma` | `0.99` |
| GAE lambda | `0.95` |
| PPO/value clip | `0.2` |
| Entropy coefficient | `0.01` |
| Tera gate/slot entropy | `0.01` each |
| Value coefficient | `0.5` |
| Global gradient-norm limit | `0.5` |
| Epochs / minibatches | `4 / 4` |
| Rollout | 128 steps x 72 environments |

One optimizer owns the complete shared model. The current update order is:

1. four epochs of preview PPO and preview auxiliary learning; then
2. four epochs of battle PPO and battle auxiliary learning.

This ordering matters because the preview update changes the shared policy
before battle PPO evaluates the behavior samples collected by the old policy.

## Current reference evidence

### Artifacts

The current reference checkpoint is:

```text
checkpoints/selfplay_public_state_v3_main_440_20260717/final.pt
```

Its immutable training summary is:

```text
logs/selfplay_public_state_v3_main_440_20260717.summary.json
```

Evaluation artifacts are under:

```text
logs/evaluation/selfplay_public_state_v3_main_440_20260717/
```

### Training run

**Verified:**

- 440 updates;
- 4,055,040 policy decisions;
- 653,915 completed battles;
- 14 h 54 min wall time;
- 75.59 end-to-end steps/s and 71.92 median update steps/s;
- approximately 87% of wall time in simulation and 13% in optimization;
- no retained-battle growth, CUDA-memory leak, non-finite update report, stall,
  traceback, or incomplete checkpoint.

The speed difference from the preceding run was not a controlled experiment.
Available laptop CPU varied, and running Codex or other applications changes
the simulator's available CPU. No throughput improvement should be claimed
from these aggregate runs alone.

### Strength trajectory

Mirrored sampled-policy evaluation gave:

| Matchup | Final-policy score |
| --- | ---: |
| Initialization, training teams | 84.4% (108-20) |
| Initialization, 15 held-out teams | 81.3% (104-24) |
| Update 110 | 55.5% (71-57) |
| Update 220 | 51.6% (66-62) |
| Update 340 | 55.5% (71-57) |
| Identical-policy role control | 47.7% (61-67) |

**Interpretation:** the model learned substantial transferable strength over
initialization. These panels do not establish meaningful improvement after
update 110. Online results show some continued separation from much older
policies, so the evidence supports a local plateau more strongly than a claim
that all learning stopped.

### Preview and public mechanics

For the exact historical Portland roster, exact enumeration found:

| Metric | Previous model | Current model |
| --- | ---: | ---: |
| Entropy | 2.542 nats | 1.598 nats |
| Opponent-conditioned information | 0.629 bits | 0.555 bits |
| Distinct most-probable bring sets | 2 | 1 |
| Distinct most-probable lead pairs | 6 | 5 |

Against `SimpleHeuristics` on Portland, learned preview scored 44.0%,
max-base-stat preview 39.3%, and random preview 31.0%.

**Interpretation:** preview learning is useful but too concentrated. The policy
mostly changes probabilities and leads; it has not demonstrated robust
opponent-dependent four-Pokemon selection.

Other verified diagnostic findings:

- Trick Room is now selected on both training and held-out Trick Room teams.
- Repeat Protect fell on the historical Portland panel.
- Fake Out is rarely selected while legal on the first active turn and is
  sometimes selected after its effective window.
- A held-out Hard Trick Room team set Trick Room in 39 of 84 opportunity
  battles but won only 17.6%.
- Tera use is near-universal, but some teams remain heavily biased to turn one.
- Held-out targeting is legal and safe, but super-effective target selection is
  lower outside the training pool.
- Unknown-only in-pool item, ability, and Tera prediction approached 99%, and
  move recall approached 90%. Repeated exposure to the 85-team training pool
  means this is not yet a held-out belief benchmark.

### Team coverage and evaluation validity

The training manifest contains 85 published, locally validated teams. A
separate manifest contains 15 published, locally validated held-out teams, and
the training loader rejects that held-out manifest.

The separation prevents accidental training use, but the 15-team panel has
already influenced architecture and development decisions. It is now a
validation set, not a pristine final test. Reusing 85 exact rosters also means
high hidden-set prediction scores may combine transferable inference with
roster memorization.

The desired expansion is not random legal team generation. It is additional
published teams with traceable sources and local Showdown validation, chosen
to widen:

- species, items, abilities, moves, and Tera types;
- speed-control, weather, terrain, setup, denial, redirection, and switching
  archetypes;
- positional and targeting interactions; and
- matchup combinations absent from the present pool.

New sourced teams should be split before they affect decisions into training,
development, and locked test manifests. Evaluation should report per-team and
per-archetype results, mirrored roles, uncertainty intervals, preview
diversity, and mechanic-specific behavior rather than only aggregate bot win
rate.

## PPO and the preview problem

### What happened

**Verified:** battle PPO was numerically stable, but preview updates caused
material behavior-policy drift.

| Metric | All updates | Last 50 |
| --- | ---: | ---: |
| Battle approximate KL | 0.0144 | 0.0186 |
| Battle clip fraction | 16.4% | 19.2% |
| Battle value clip fraction | 12.3% | 6.3% |
| Preview approximate KL | 0.0302 | 0.0398 |
| Preview clip fraction | 26.0% | 30.8% |
| Pre-battle-PPO clip fraction | 10.3% | 13.6% |

The largest preview minibatch KL was 0.480. The final-50 mean maximum preview
KL was 0.0837.

`pre-battle-PPO` is measured after preview learning but before the first battle
optimizer step. Its clipping therefore cannot be caused by battle PPO in the
same update. The shared preview update made already-collected battle data
off-policy.

There is **no target-KL early stop in the current code**. It was discussed as a
possible emergency mechanism but was not implemented. It should not be
described as a current fix.

### Principled proposal

**Proposal:** treat preview as the first decision of the same episode and make
preview and battle optimization one coordinated behavior-policy transaction.

A concrete design to review is:

1. Record preview observation, chosen one-of-180 action, behavior log
   probability, behavior value, policy version, and terminal return with the
   rollout.
2. Freeze all old behavior metadata before any shared optimizer step.
3. Form one update schedule in which preview policy, battle policy, values, and
   auxiliaries all refer to the same behavior-policy version.
4. Keep task-specific heads and sampling weights, but update shared parameters
   under a coordinated objective rather than completing all preview steps
   first.
5. Because preview has exactly 180 legal actions, compute its exact categorical
   KL rather than relying only on sampled approximate KL.
6. Measure gradient norms and cosine similarity for preview actor, battle
   actor, both values, and each auxiliary objective on the shared trunk.
7. If an explicit KL constraint is wanted, test an adaptive constrained
   objective against ordinary clipped PPO. Do not silently stop epochs.

This proposal is not proven better. Its purpose is to fix the policy-version
semantics and make interference measurable rather than hiding preview
instability by lowering one coefficient.

## Opponent sampling audit

### What the current implementation does

The reference run used two opponent groups. Each group shared one frozen
checkpoint across 36 of the 72 environments so frozen inference could batch
efficiently.

The league:

- begins with an initialization checkpoint;
- snapshots the main policy every 10 updates;
- keeps every snapshot in a permanent 45-member archive;
- refreshes the two group assignments after every non-final update;
- samples two distinct members without replacement when at least two exist;
  and
- stages new opponents at episode boundaries rather than changing an opponent
  during a battle.

For archive member `i`, the current learner's estimated win probability is:

```text
p_i = (1 + discounted_wins_i)
      / (2 + discounted_wins_i + discounted_losses_i)
```

All games against one member in one learner update contribute one normalized
evidence unit. Evidence has a 20-update half-life. An unseen or fully stale
member therefore has the symmetric Beta(1,1) prior estimate `p_i = 0.5`.

The first-draw probability is:

```text
hard_weight_i = (1 - p_i)^2
draw_probability_i
    = 0.9 * hard_weight_i / sum(hard_weights)
    + 0.1 / archive_size
```

After a rollout, the code trains the live policy, records the completed
rollout's outcomes in PFSP, adds a due snapshot of the newly trained policy,
and then samples from the expanded archive. Tests cover the
record/add-before-draw boundary, distinct weighted draws, exact RNG restoration
on resume, and restoration of existing group assignments without an extra
draw.

### What `Opp seen` means

`Opp seen` is the count of distinct checkpoint paths attached to battles that
**finished during that update**. It is not the number of fresh PFSP draws and
not the number of teams encountered.

Only two fresh checkpoint paths are drawn per refresh. Opponent replacement is
staged separately for each environment when its current battle finishes.
Consequently, an update commonly contains completed battles against:

- two previous assignments whose battles were still draining; and
- two newly active assignments.

That is why the reference run reported:

| Distinct completed-battle opponents | Updates |
| ---: | ---: |
| 1 | 10 |
| 2 | 23 |
| 3 | 94 |
| 4 | 313 |

Four is not an absolute code-level cap: a sufficiently long battle could span
more than one refresh boundary. It was the operational maximum in this run.
After the single-member bootstrap, each of the remaining 430 rollout updates
used two distinct nominal assignments; after each non-final update, another
pair was drawn for the next rollout.

Two groups are a throughput/diversity choice, not a demonstrated learning
optimum. More groups would expose more snapshots per update but create smaller
inference batches, more resident models, and more cache churn. That tradeoff
needs a controlled benchmark and a learning argument.

### The 43-of-45 result

The two archive members never assigned or observed were:

- `step_3870720.pt`, created at update 420; and
- `step_4055040.pt`, created at update 440.

The update-440 snapshot had no possible next rollout because the run was
complete. It should not be counted as a generic sampling failure.

The update-420 snapshot was immediately eligible but received no newcomer or
recency priority. Its creation-time first-draw probability was 2.505%; with two
distinct group draws, its inclusion probability per refresh was 5.006%. Over
the 20 remaining refreshes:

- expected assignments: 0.986; and
- probability of never being selected: 36.37%.

Its absence is therefore unsurprising.

Across the members that were eventually observed, median first-use delay was
5 updates, mean delay 9.35, and maximum delay 50. The 50-update case was the
snapshot created at update 280 and first observed at update 330. New snapshots
arrive every 10 updates.

At the end, the distribution was nearly uniform:

- effective support: 44.74 of 45 members; and
- first-draw probabilities: 1.45% to 2.49%;
- estimated current-policy win probabilities: 48.5% to 62.0%; and
- median discounted evidence: 2.60 update-equivalents, with a maximum of 5.94.

### Assessment and alternatives

**Verified:** PFSP is operating as coded. Outcome attribution, add-before-draw
ordering, distinct group draws, and resume state appear correct.

**Interpretation:** the implemented curriculum has a meaningful evidence and
recency weakness. A neutral prior is defensible, but with two draws and a
growing archive it may take longer to evaluate a new policy than to create the
next one. Near-uniform final probabilities also mean the hardness estimates
were not producing much prioritization.

The console label should be renamed from `Opp seen` to something like
`Completed-battle checkpoint count`; the existing phrase invites exactly the
misinterpretation discussed above.

**Unknown:** it has not been established whether near-uniformity reflects a
genuinely even population, evidence decay that is too fast, per-update evidence
compression that is too aggressive, or insufficient distinct opponent draws.
The outcomes used for a refresh were also generated by the pre-update policy,
while the next rollout uses the just-updated policy. The 20-update decay is
intended to track this non-stationarity, but the adequacy of that estimator has
not been tested.

Options for independent review, none implemented:

1. **Uncertainty-aware PFSP.** Maintain a posterior over current-policy win
   rate and sample or prioritize using both estimated hardness and uncertainty.
   This gives an unseen member a reason to be measured without a hand-written
   age bonus.
2. **Explicit calibration allocation.** Give each new member a small,
   separately logged evidence budget before ordinary PFSP. This guarantees
   measurement but spends simulations on a quota and must not be hidden as
   ordinary hardness sampling.
3. **Current-self-play plus historical PFSP.** Reserve an explicit mixture
   component for a recent main checkpoint and use PFSP for history. This is
   literature-grounded but is still a chosen mixture coefficient.
4. **More opponent groups.** Increase distinct draws only if a controlled
   throughput test shows the simulator/inference cost is acceptable.
5. **Change evidence decay or compression.** Only after replaying the saved
   outcome stream to show how alternative estimators change support and
   time-to-first-use.

Any selected design should report newcomer time-to-first-draw, evidence amount,
posterior uncertainty, distribution entropy/effective support, and the exact
reason each opponent was sampled.

## Exploiter status

An exploiter is a dedicated best-response learner. A main exploiter targets one
frozen main policy; a league exploiter targets a certified mixture of a
population. It is allowed to become strategically narrow because its purpose
is to expose a blind spot, not replace the robust main policy.

### What is verified

The repository can:

1. bootstrap an immutable population from a trainer checkpoint;
2. plan a main-exploiter phase against one frozen main;
3. train ordinary sparse-reward PPO without adapting or archiving the target;
4. save a resumable exploiter checkpoint and paired phase state;
5. evaluate a complete cross-play row;
6. build a verified child population; and
7. plan a league exploiter against a solver-certified mixture.

A current-code preflight initialized from the reference model and ran one
main-exploiter update:

- 9,216 decisions;
- 1,428 battles;
- 701 wins and 727 losses (49.1%);
- valid `final.pt` and phase state;
- frozen target; and
- no automatic admission.

This proves plumbing and checkpoint semantics. It does **not** prove the
exploiter learned an exploit. A league-exploiter phase has been constructed
and validated but has not been trained.

The completed 440-update main run did not launch an exploiter. It had no phase
specification, and all 45 opponents were ordinary self-play snapshots.

### What is missing

- No controller schedules exploiter phases around main training.
- No production command automatically turns a completed exploiter into a
  member specification, complete cross-play evidence, and an admitted main
  opponent.
- `League.add_admitted_checkpoint` exists and is tested, but normal training
  never calls it.
- There is no evidence yet that a trained exploiter discovers a useful,
  repeatable counter-policy.

It would be misleading to say exploiters "failed to land" in the reference
run. They were not run there.

### Correct next experiment

**Proposal:** prove exploiter learning before implementing automatic
integration.

1. Freeze the current main checkpoint and team pool.
2. Train a main exploiter from the specified initialization with a fresh
   optimizer.
3. At fixed intervals, run mirrored sampled evaluation against the frozen
   target and held-out teams.
4. Require statistically credible improvement over 50%, evidence across more
   than one team, stable PPO, and a behavioral diagnosis of what the exploiter
   changed.
5. Evaluate the candidate against historical policies to measure brittleness.
6. Only then construct its complete payoff row and test how admitting it changes
   main training.
7. Train a league exploiter only after the population and payoff matrix are
   meaningful.

Checkpoint creation is not an admission criterion.

## Auxiliary-loss and optimizer questions

### Auxiliary objectives

Current coefficients are:

- hidden information: `0.25`;
- opponent intent: `0.25`;
- public consequence: `0.10`;
- preview opponent bring: `0.10`; and
- preview opponent lead: `0.05`.

Loss magnitudes alone do not reveal their influence on shared parameters.
Battle PPO clipping constrains the sampled policy term, not the value or
auxiliary gradients that also update the trunk.

An optional diagnostic currently measures objective gradient norms/cosines on
the LSTM and opponent-team encoder. It was disabled in the reference run. It
does not include preview objectives, the entire shared trunk, or the actual
pre-clipping global gradient norm.

**Proposal:** first measure per-objective norms, pairwise cosine similarities,
combined cancellation, and actual gradient-clipping incidence across each
shared module. Depending on that evidence, test:

- task-specific adapters with a smaller, explicit shared path;
- separate optimizer parameter groups or update frequencies;
- controlled auxiliary-gradient scaling; or
- a gradient-conflict method such as PCGrad only where conflict is measured.

Do not remove useful beliefs merely because their scalar losses are large, and
do not assume a good auxiliary accuracy means the actor benefits.

### Optimizer and discount

The current Adam configuration is stable but not validated as optimal.

AdamW should not be dismissed. With zero weight decay it is effectively the
current Adam update; a meaningful experiment needs an explicit decay value and
parameter grouping so biases, normalization parameters, embeddings, and
possibly recurrent parameters are treated intentionally.

Muon is also a legitimate research candidate, but evidence from large
feed-forward language models does not establish that it is superior for a
recurrent on-policy actor-critic. A fair test would likely be a hybrid: Muon on
appropriate two-dimensional matrix parameters and AdamW on embeddings,
vectors, normalization, and other unsupported parameters.

Suggested isolated comparisons are:

1. current Adam baseline;
2. AdamW with documented parameter groups and decay;
3. hybrid Muon/AdamW;
4. preview-specific learning rate/epoch schedule after fixing policy-version
   semantics;
5. a learning-rate schedule versus the current fixed rate; and
6. `gamma=1.0` versus `0.99`.

`gamma=0.99` discounts a terminal outcome by battle length and therefore
slightly prefers faster wins and delayed losses. `gamma=1.0` is the cleaner
objective for undiscounted win probability, but it changes return variance and
should be tested separately.

Optimizer claims require multiple seeds or at least balanced repeated runs,
equal simulation data, the same opponent/team streams where possible, and
strength/robustness measures rather than training loss alone.

Primary references:

- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982)
- [AlphaStar league and PFSP](https://doi.org/10.1038/s41586-019-1724-z)

## Guardrail, masking, and fallback inventory

This table includes material mechanisms in the training and inference path.
It intentionally separates core algorithm choices from compatibility
fallbacks. "Not observed" means the summary cannot prove whether it activated.

| Mechanism | Current behavior | Reference-run evidence | Assessment |
| --- | --- | --- | --- |
| PPO policy clipping | Ratio objective clipped to `[0.8, 1.2]` | Battle and preview clip fractions logged every update | Core PPO; highly active, especially preview |
| Value clipping | Value delta clipped to `+-0.2` | Fractions logged; maximum 74.4% early, 6.3% late mean | Core heuristic; visible |
| Global gradient clipping | Every Adam step clipped to norm `0.5` | Pre-clip norm and activation fraction not logged | Material observability gap |
| Extreme log-ratio clamp | Log ratio clamped to `[-20, 20]` before exponentiation | No activation counter; pre-battle maximum was below it, later battle/preview maxima incomplete | Numerical safety should be counted |
| Non-finite update skip | Battle skips individual bad minibatches and fails only if all skip; preview aborts that preview update | Activation-only keys absent in reference run | Zero observed, but always log explicit zero/count |
| Preview-before-battle order | Preview updates shared model before battle PPO | Pre-battle KL/clip logged and high | Training-design problem, not a guardrail |
| Stale preview exclusion | Drops previews generated by an older policy version | 31,608 dropped; 622,307 current previews used | Correct on-policy protection, but expensive and motivates unified metadata |
| Preview sample cap | Optional uniform cap | Disabled (`None`); all current samples used | No hidden cap in reference run |
| Tera epsilon exploration | Gate/slot epsilon `0.15 -> 0` over updates 1-110 with exact behavior likelihood | Value logged each update | Explicit exploration heuristic, not reward shaping |
| PFSP prior/decay/uniform floor | Beta(1,1), 20-update half-life, 10% uniform mixture | Full final distribution and member evidence logged | Algorithm choices; coverage is not guaranteed |
| PFSP evidence compression | One normalized evidence unit per member per learner update | Raw lifetime results and discounted equivalents logged | Auditable heuristic; sensitivity unknown |
| Opponent groups | Two distinct paths drawn, shared across 36 environments each | Assignments and completed-battle paths logged | Throughput/diversity choice, not validated optimum |
| Legal-action masks | Invalid actions receive masked logits; joint conflicts and Tera eligibility masked | No illegal-sample counter | Necessary game semantics; production should fail on corrupted masks |
| Default-only/all-pass bypass | Opponent inference can be skipped when no meaningful choice exists; an all-pass order uses poke-env's default move | Counts only available in detailed timing, disabled in reference run | Likely semantically harmless, but activation should be visible |
| Move hydration compatibility path | Missing request index/target is reconstructed from the request | No counter | Silent production activation would be suspect |
| Neutral public-field fallback | Missing battle field input substitutes a neutral field | No counter | Remove or fail-loud in production |
| Missing mechanics/lookup fallback | Missing explicit mechanics may disable matchup context or substitute zero lookup buffers | No counter | Remove or fail-loud in production |
| Recent-event cap | Keeps first 12 public events and records capped overflow metadata | Overflow mean was tiny; maximum update fraction 0.0258% | Visible at aggregate level; raw tail loss not reported |
| Partial/overflow label masking | Ambiguous public-consequence segments receive no exact consequence label | Partial fraction about 24.3% late; per-head support logged, reason breakdown absent | Correct not to invent labels; exclusion reasons need counts |
| Invalid consequence schema | Missing/wrong-shaped arrays may return empty label masks instead of failing | No invalid-schema counter | Can silently erase supervision; high-priority audit |
| Hidden-label identity matching | Retrofits labels only for a unique terminal Pokemon identity | Overall coverage above 97%; no ambiguity/untracked breakdown | Correct no-guess rule; improve diagnostics |
| Field-change initialization prior | Change gate initialized with zero weights and bias `-12` | No direct activation metric; rare-change recall remained low | Strong deliberate prior; review whether it impedes learning |
| Detached consequence feedback | Actor consumes detached predictor output | Adapter norm and task metrics logged; gradient flow not logged by default | Principled one-way coupling, but benefit unknown |
| Scalar observation caps | Protect 4, status 8, active turns 6, field timers by mechanic, battle turn 20, hit count 10 | Saturation not logged | States above a cap alias; add saturation metrics |
| Optional BPTT truncation | Disabled in reference run (`None`) | Config logged | Not an active hidden limit |
| CUDA graph fallback | On exception restore RNG, record error/fallback, disable graphs, continue eager | Zero fallbacks/errors; captures/replays/evictions logged | Performance-only and observable |
| Resident model cache | Permanent archive, 12-model LRU resident cache | 99.9% hit rate and 329 evictions logged | Performance-only |
| Checkpoint/resume validation | Exact metadata, optimizer/RNG, paired league or phase state; incompatibility fails | Completed cleanly | Fail-loud, no weights-only resume fallback |
| BattleStream timeout | 30 s timeout; raises instead of retrying/restarting | No activation in completed run | Fail-loud |
| Proposed target-KL stop | Not implemented | Not applicable | Must remain labelled as a proposal |

The highest-priority observability fixes before another long run are:

1. pre-clip gradient norms and fraction/count clipped, by objective/module;
2. log-ratio clamp counts;
3. explicit zero-valued non-finite skip counters;
4. production fallback counters followed by removal or fail-loud validation;
5. observation saturation counts;
6. reason-specific public-label exclusions; and
7. newcomer sampling delay and uncertainty.

The empty-event team-preview encoder shortcut was also audited. Preview always
precedes public battle events, the ordinary event encoder would return the same
zero representation, and an equivalence test covers it. It is a
behavior-preserving performance path, not an unreported semantic shortcut.

## Proposed order of work

This is a review proposal, not an approved implementation plan:

1. Add observability for the existing mechanisms without changing training.
2. Run a controlled exploiter-learning proof; do not admit it automatically.
3. Replay the saved PFSP outcome/assignment history under alternative evidence
   estimators and report coverage/hardness changes.
4. Design and unit-test coordinated preview/battle behavior metadata and exact
   preview KL.
5. Run short balanced experiments for preview update semantics and opponent
   group/sampling choices.
6. Enable expanded gradient diagnostics for a short run, then decide whether
   auxiliary routing or coefficients need change.
7. Run isolated optimizer and `gamma` experiments only after the update
   semantics are stable.
8. Benchmark simulator changes with Codex and unrelated work closed, alternating
   baseline/candidate order and preserving model behavior.
9. Start another long main run only after the independent review and the
   selected short experiments.

More training with the present checkpoint may still improve some skills, but a
long continuation would not answer the measured preview-interference,
opponent-evidence, exploiter, or auxiliary-gradient questions.

## Source map

Key implementation areas:

- `pokerl/config.py`: model, PPO, exploration, and league defaults
- `pokerl/model/`: encoders, recurrent model, action/preview/auxiliary heads
- `pokerl/training/ppo.py`: battle and preview optimization
- `pokerl/training/preview.py`: preview versioning and update ordering
- `pokerl/training/league.py`: payoff evidence, PFSP, archive, inference cache
- `pokerl/training/exploiter.py`: best-response phase definitions
- `pokerl/training/population.py`: immutable populations and meta-solver
- `pokerl/env/observation.py`: public observation and representation caps
- `pokerl/env/actions.py`: legal joint-action construction
- `scripts/train.py`: rollout, update, snapshot, and opponent-refresh order
- `scripts/evaluate_anchor.py`: mirrored sampled-policy evaluation

The code and immutable JSON artifacts are authoritative when this brief and the
repository disagree.
