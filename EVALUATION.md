# Evaluation and quality plan

This document records the evaluation methodology and the main findings from the
150-update run completed on 2026-07-12. The machine-readable summaries live
under `logs/evaluation_150_20260712/` and are intentionally ignored by Git.

## Methodology

The primary checkpoint comparison uses the policy that PPO trained: both
players sample from their policy with training-only Tera exploration disabled.
Every scheduled team pair is played in both directions, swapping player and
team roles. Greedy-vs-greedy results are a secondary deterministic diagnostic.

The evaluator now:

- uses the normal matchup-aware `PokeAgent.forward` path on both sides;
- applies the same sample or argmax mode to both policies;
- rejects partial or architecture-mismatched checkpoints;
- pre-plans identical team pairs and mirrors player/team roles;
- records per-game teams, preview choices, turns, and outcomes;
- separates wins, losses, and draws and reports 95% intervals;
- evaluates Random, MaxBasePower, and SimpleHeuristics bots behind the same
  deterministic max-BST team-preview rule;
- reports targeting, friendly-fire, Protect, switching, focus-fire, Tera, and
  preview diagnostics; and
- exactly enumerates all 180 legal 6-to-4 preview actions to measure how much
  the preview distribution actually depends on the opposing roster.

Random, MaxBasePower, SimpleHeuristics, and any future scripted policies are
evaluation baselines and deterministic probes only. They are not training
opponents. Training remains scalable self-play against the frozen-checkpoint
league; this avoids teaching the policy to exploit hand-written bot quirks and
keeps the training distribution extensible without maintaining a parallel
scripted curriculum.

Python, NumPy, Torch, bot, and team-schedule randomness is seeded. Pokemon
Showdown's simulator RNG is not yet seed-controlled, so damage rolls and other
in-battle randomness are not bitwise reproducible. The present team panel is
also in-distribution: all 77 teams were used during training.

## What 150 updates learned

### Checkpoint cross-play

Each non-final matrix cell contains 64 mirrored sampled games. Final cells
contain 128. Percentages are the later checkpoint's win rate.

| Later checkpoint | Initial | Update 50 | Update 100 | Update 125 |
|---|---:|---:|---:|---:|
| Update 50 | 64.1% | - | - | - |
| Update 100 | 81.2% | 62.5% | - | - |
| Update 125 | 71.9% | 67.2% | 48.4% | - |
| Update 150 | 83.6% | 60.9% | 53.9% | 52.3% |

Final-vs-final was 45.3% over 128 games, with a 95% interval of 37.0-53.9%.
The evaluator therefore detected no clear player-side bias. There is strong,
mostly transitive improvement through about update 100, followed by a plateau.
No statistically supported non-transitive cycle appeared in this matrix.

### Fixed bots

The first panel fixes the model's roster to the Portland team and varies all 76
other in-pool opponent rosters. These are sampled-policy results.

| Opponent | Initial | Final |
|---|---:|---:|
| Random | 27.6% | 63.2% |
| MaxBasePower | 1.3% | 53.9% |
| SimpleHeuristics | 2.6% | 40.8% |

This competence is not roster-general. With the final checkpoint:

| Model roster | MaxBasePower | SimpleHeuristics |
|---|---:|---:|
| Portland balance | 53.9% | 40.8% |
| Hatterene Trick Room/sun | 23.7% | 14.5% |
| Utrecht balance | 90.8% | 94.7% |

On the Hatterene roster, representative greedy traces repeatedly led Hatterene
plus Indeedee but selected two attacks instead of Trick Room plus redirection,
then lost in three to five turns. The policy has learned effective offense on
some teams, not the strategic modes required by every team it was trained on.

### Mechanics and tactics

On the deterministic Portland panel, training reduced damaging friendly-fire
from about 30% to zero and immune targeting from about 12% to approximately
zero. Super-effective targeting rose from roughly 20-24% to about 45%. The
final model also chose lower-base-power moves more often while increasing
super-effective targeting, evidence that it learned matchup-aware move choice
rather than only the largest displayed power.

The main tactical failures are equally clear:

- the final policy Terastallizes on turn one in essentially every game;
- the initial policy also has a strong immediate-Tera prior, and training makes
  the learned no-Tera probability still smaller;
- Protect use is below 1% on the main sampled panel; and
- the Trick Room team does not reliably set Trick Room, redirect, Fake Out, or
  otherwise execute its intended speed-control plan.

### Team preview

For the fixed Portland roster, exact enumeration across 76 opposing rosters
shows:

| Metric | Initial | Final |
|---|---:|---:|
| Exact policy entropy | 4.657 nats | 2.497 nats |
| Mean top-action probability | 2.81% | 19.40% |
| Distinct joint MAP choices | 1 | 1 |
| Opponent/preview mutual information | ~0 bits | 0.00089 bits |
| Mean TV distance from mean policy | 0.007% | 1.345% |

The Hatterene roster gives the same conclusion (0.00045 bits at the final
checkpoint). The preview policy learned a sharper team-specific default, but
almost no opponent-conditioned response.

That default is useful. Holding the final battle policy fixed, learned preview
scored 53.9%/40.8% against MaxBasePower/SimpleHeuristics, compared with
51.3%/34.2% for max-BST preview and 44.7%/32.9% for random preview. Thus the
model learned global roster synergy, not matchup selection.

The auxiliary opponent-bring predictor improved from 7.9% exact to 23.7%
exact (random 4-of-6 chance is 6.7%). Ordered opponent-lead accuracy remained
1.3%, below the 3.3% uniform baseline. Useful opponent-roster information is
present in the representation, but the own preview policy barely uses it.

## Accepted direction after the 150-update evaluation

### Implemented for the next run

1. **Tera is now a factorized decision.** The deployment policy represents a
   binary use-now gate followed by a conditional eligible-slot choice, while
   preserving the exact external `{none, slot 1, slot 2}` distribution. The
   gate starts with a waiting-biased prior and gate/slot entropy are measured
   and weighted separately. The timing-specific force-delay override has been
   removed. Rollout exploration is now the standard exact mixture on the gate
   and conditional slot policy, and its linear schedule is expressed relative
   to the planned update horizon instead of fixed update counts. PPO evaluates
   the same behavior distribution that generated each rollout.

2. **Every current-policy preview sample is used by default.** The former 1,024
   cap discarded roughly one quarter of eligible samples in the 150-update
   run. Preview training now accumulates globally normalized gradients over
   bounded chunks and performs one optimizer step, so activation memory stays
   bounded without changing the full-sample objective. An explicit cap remains
   available for constrained experiments, but is not the default.

3. **The policy now receives causal recent history explicitly.** `poke-env`
   exposes a bounded immutable sequence of up to 12 perspective-public
   MOVE/SWITCH/DRAG/CANT/FAINT/TERA events, with order, actor/target positions,
   move, incoming species, Tera type, partial-turn state, and explicit overflow
   metadata. pokeRL also carries the agent's own previous semantic intent
   (move, target, switch, pass, and Tera), resolved before request-local action
   slots can change. A dedicated recent-event encoder is concatenated into the
   spatial context before the LSTM. Opponent-private action labels, packed
   teams, and terminal auxiliary labels never enter the policy observation.

4. **The stable PPO path remains in place.** The sampled action likelihood,
   recurrent unroll, terminal resets, clipped objective, and pre-PPO KL/clip
   diagnostics remain the source of truth. The 150-update run showed stable
   pre-update ratios and bounded CUDA memory, and the factorized Tera path keeps
   rollout and PPO likelihoods exact rather than introducing an off-policy
   timing correction.

### Representation-learning boundary

The hidden-information, opponent-action, and opponent-preview heads currently
shape the shared representation only through auxiliary losses. Their predicted
beliefs are not fed back into the policy, so improvements in auxiliary accuracy
must not be described as direct strategic use. Before promoting any belief
output into policy context, evaluation must separate genuinely unknown labels
from already revealed facts, report calibration on that unknown-only subset,
and define semantic targets whose accuracy is actionable rather than merely
easy to memorize. Until then these heads remain indirect representation
regularizers.

### Remaining priorities

1. **Create a real validation population.** Expand the legal team population,
   reserve an archetype-stratified set that training never samples, and report
   preview, belief, checkpoint cross-play, and fixed-probe metrics on both train
   and held-out teams. The current hidden-info result can otherwise be fixed-set
   memorization.

2. **Improve coverage inside self-play.** Balance league rollout sampling by
   roster archetype and measured per-roster weakness while retaining frozen
   checkpoints as the scalable opponent source. Trick Room, redirection, Fake
   Out, Protect, speed control, immunity avoidance, and Tera timing should have
   deterministic evaluation probes, not scripted training opponents.

3. **Keep preview evidence-based.** Continue to report exact opponent/preview
   mutual information and preview ablations. Add stratification, clipping, or
   an entropy floor only when short A/B runs show better held-out outcomes; do
   not reward a preview for merely being different across opponents.

4. **Keep the LSTM through this iteration.** Explicit public events and own
   intent repair a real information omission while recurrence preserves longer
   history. Measure full-history versus reset-hidden inference before replacing
   memory.

5. **Do not deploy PFSP merely because it is fashionable.** The current matrix
   shows a plateau but no cycle. Keep permanent evaluation anchors and the raw
   payoff matrix. Add PFSP or worst-opponent sampling only if a larger held-out
   matrix exposes forgetting or a recurrent worst matchup.

## Efficiency implications

After the fifth update, the 150-update run spent 92.4% of wall time collecting
rollouts and 7.4% in PPO. Preview refresh, checkpoint, league maintenance, and
unattributed overhead together were well below 1%. Replacing the LSTM or
micro-optimizing the optimizer therefore cannot materially change end-to-end
speed.

The safe efficiency work is:

- profile and optimize the local BattleStream/Showdown request path;
- retain two opponent groups because they provide useful diversity while
  preserving inference batches;
- benchmark 0-1 ms league coalescing on long, warmed runs rather than trusting
  startup-heavy microbenchmarks; and
- improve learning per battle through roster balancing, better preview use,
  and better state information. A run that reaches the same strength in fewer
  battles is the largest remaining speedup that does not trade away quality.

The synchronous batched collector remains the production path. A prior
asynchronous collector experiment reduced effective batching and regressed
throughput, so asynchronous collection is postponed rather than carried as a
second complex training path. It should be revisited only with a design that
preserves inference batch size and wins a warmed, multi-update A/B benchmark.

Before a larger run, use short multi-seed A/B runs to require: no throughput
regression, no friendly-fire/immunity regression, delayed and context-dependent
Tera, non-trivial preview mutual information, improved Hatterene-roster bot
results, and at least the existing checkpoint/fixed-bot strength trajectory.
