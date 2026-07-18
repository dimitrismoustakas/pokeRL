# Current project status

Updated: 2026-07-18.

This file is a replace-in-place snapshot of accepted evidence. It is not a
chronological development log.

## Current reference run

The current model is:

```text
checkpoints/selfplay_public_state_v3_main_440_20260717/final.pt
```

The immutable training summary is:

```text
logs/selfplay_public_state_v3_main_440_20260717.summary.json
```

The evaluation directory is:

```text
logs/evaluation/selfplay_public_state_v3_main_440_20260717/
```

The run completed 440 updates, 4,055,040 decisions, and 653,915 battles in
14h54m. End-to-end throughput was 75.59 SPS; median update throughput was
71.92 SPS. Simulation used about 87% of wall time and PPO about 13%. There was
no retained-battle growth, CUDA memory leak, non-finite update, or trainer
failure.

The aggregate speed increase over the previous run is not a controlled
efficiency result: mature median throughput was effectively unchanged and the
laptop's available CPU varied during both runs.

## Strength trajectory

Mirrored sampled-policy evaluation:

| Matchup | Final-policy score |
| --- | ---: |
| Initialization, training teams | 84.4% (108-20) |
| Initialization, 15 held-out teams | 81.3% (104-24) |
| Update 110 | 55.5% (71-57) |
| Update 220 | 51.6% (66-62) |
| Update 340 | 55.5% (71-57) |
| Identical-policy role control | 47.7% (61-67) |

The policy clearly learned transferable strength over initialization. The
available panels do not establish improvement after update 110. Online league
results still show slow separation from much older policies, so this is a
local plateau rather than proof that learning stopped completely.

## PPO

Battle PPO remained numerically stable:

- overall clip fraction: 16.4%;
- last-50 clip fraction: 19.2%;
- last-50 approximate KL: 0.0186;
- last-50 value explained variance: about 0.70; and
- no non-finite or skipped minibatches.

Preview PPO is the measured trust-region concern. In the last 50 updates its
clip fraction was about 30.6% and mean KL about 0.0418, with isolated much
larger minibatch spikes.

Preview optimization runs before battle PPO on the same optimizer and shared
model. Consequently, about 13.6 percentage points of late battle clipping
already existed before the battle update began. This is shared-update
interference, not evidence that the PPO implementation numerically diverged.

### Hyperparameter evidence

The reference run used one fixed-rate Adam optimizer (`lr=2.5e-4`,
`eps=1e-5`), four epochs, four minibatches, PPO/value clipping `0.2`,
gradient-norm clipping `0.5`, `gamma=0.99`, GAE `0.95`, and entropy coefficient
`0.01`. Preview used the same optimizer, learning rate, clip range, entropy
coefficient, and four epochs.

These settings are stable, not validated as optimal. Current evidence does not
support changing optimizer or globally lowering the battle learning rate.
It does support a controlled preview-specific update experiment and
shared-gradient diagnostics.

`gamma=0.99` discounts a terminal result by battle length and therefore
slightly prefers faster wins and delayed losses. A separate `gamma=1.0` test is
the clean objective-level comparison for pure win probability; it should not
be bundled with preview changes.

## Preview

For the exact historical Portland roster, exact enumeration over all 180
legal choices found:

| Metric | Previous model | Current model |
| --- | ---: | ---: |
| Entropy | 2.542 nats | 1.598 nats |
| Opponent-conditioned information | 0.629 bits | 0.555 bits |
| Distinct most-probable bring sets | 2 | 1 |
| Distinct most-probable lead pairs | 6 | 5 |

“Most-probable” means the single highest-probability (`argmax`) policy choice
for each opposing roster. It does not mean a heuristic or an optimal choice.

Against SimpleHeuristics on Portland, learned preview scored 44.0%, max-BST
preview 39.3%, and random preview 31.0%. The learned preview is useful but
narrow and has not learned robust matchup-dependent four-Pokémon selection.

## Public mechanics

Measured improvements:

- Trick Room setup is now selected on both training and held-out Trick Room
  rosters.
- Repeat Protect use fell substantially on the historical Portland panel.
- Field persistence/change/alternative prediction has positive skill over
  persistence, unlike the previous architecture.
- Public route, Protect, miss, and immunity classifications improved.

Measured failures:

- Fake Out is rarely selected while live and is still sometimes selected
  after its first-active-turn window.
- A held-out Hard Trick Room roster set Trick Room in 39 of 84
  opportunity-battles but won only 17.6%; setup knowledge has not become
  complete archetype execution.
- Rare-event gates under-detect changes. Late field-change recall was about
  8.5% at 2.4% prevalence.
- Tera use is appropriately near-universal, but several rosters remain heavily
  biased toward turn one.
- Held-out targeting remains safe, while super-effective target selection is
  materially lower than in-pool.

## Beliefs and generalization

Unknown-only in-pool item, ability, and Tera accuracy approached 99%, and move
recall approached 90%. The policy adapters are active. These numbers can still
include memorization of the recurring 85-team roster and are not a held-out
belief benchmark.

Team-specific performance remains highly variable. Utrecht was strong against
diagnostic bots, Hatterene was weak, and held-out roster execution was weaker
still. Fixed bots are diagnostic probes, not training opponents or an absolute
competitive ladder.

## Population status

The reference run was plain `main_self_play`. It did not attempt to run or
admit an exploiter. Every archived opponent was a normal self-play snapshot.

The repository has explicit main-exploiter and league-exploiter phase,
population, evidence, and admission components. A current-HEAD isolated
main-exploiter preflight initialized from the reference final policy, trained
one update (9,216 decisions and 1,428 battles) against exactly one frozen main,
and wrote a valid trainer checkpoint plus paired phase state.

That proves best-response training, not the closed population loop. Scheduling
best responses, evaluating the complete payoff row, choosing admission,
admitting the checkpoint, and launching the next main phase remain manual
orchestration. In particular, no production command currently installs a
verified population member into a resumable main league.

## Open measured questions

- Can preview and battle learning share parameters without the preview update
  pushing battle samples out of PPO's trust region?
- Can preview learn multiple matchup-dependent bring sets rather than mainly
  changing leads and probabilities?
- Does explicit first-active-turn action binding make Fake Out learnable under
  sparse self-play, or does its training signal need a different auxiliary
  formulation?
- Do best-response exploiters supply useful adversarial pressure once the full
  population lifecycle is actually scheduled?
- Which hyperparameter changes improve these outcomes in controlled,
  multi-seed comparisons rather than merely reducing a loss?
- Can simulator throughput improve without changing the sampled policy or
  model capacity?
