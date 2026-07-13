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

## Interrupted event-aware run at update 140

The event-aware run was planned for 400 updates but completed 141 updates
(1,299,456 decisions and 197,752 battles) in 6.51 hours before an environment
reset timed out. The strongest immutable checkpoint before the failure is the
league snapshot at update 140 (`league/step_1290240.pt`), which is the model
evaluated below.

The timeout was not a PPO, CUDA, or simulator crash. Concurrent resets could
select the same apparently available local runtime shard before either caller
reserved it. One shard could therefore be overbooked while another still had a
free slot; the extra reset then waited on the full shard and the synchronous
collector waited behind that reset. poke-env now makes shard selection and
reservation one atomic operation and uses a global capacity wait/reselection
path. Cancellation and acquisition failures also return reservations. Focused
cross-thread tests reproduce the former race deterministically.

Training itself remained healthy through update 141: all 72 player and opponent
objects retained exactly one live battle, CUDA allocation stayed bounded,
pre-PPO ratios remained close to one, and the final ten updates averaged
0.00046 pre-PPO KL, 0.0138 post-update KL, 0.149 clip fraction, and 0.703 value
explained variance. There is no evidence here for changing PPO.

### Frozen self-play anchors

Every comparison used 64 planned team pairs in each player direction, sampled
actions, and identical team-pair roles after mirroring (128 games total).

| Update 140 opponent | Wins | Losses | Win rate | 95% interval |
|---|---:|---:|---:|---:|
| Initialization | 99 | 29 | 77.3% | 69.4-83.7% |
| Update 50 | 79 | 49 | 61.7% | 53.1-69.7% |
| Update 100 | 60 | 68 | 46.9% | 38.4-55.5% |
| Update 125 | 69 | 59 | 53.9% | 45.3-62.3% |
| Itself | 71 | 57 | 55.5% | 46.8-63.8% |

The model clearly learned beyond initialization and update 50, then plateaued.
The self-match result and interval show that the apparent 55.5% is sampling
noise rather than a meaningful first-player advantage. Update 140 is not
statistically distinguishable from updates 100 or 125.

### Common fixed-bot yardstick

The table compares the previous 150-update checkpoint with update 140 of the
harder event-aware model. Each cell contains 76 sampled-policy games against 76
varying opponent rosters. Fixed bots remain evaluation probes only; they are not
training opponents.

| Model roster / opponent | Previous update 150 | Event-aware update 140 |
|---|---:|---:|
| Portland / Random | 63.2% | 68.4% |
| Portland / MaxBasePower | 53.9% | 51.3% |
| Portland / SimpleHeuristics | 40.8% | 40.8% |
| Hatterene / MaxBasePower | 23.7% | 18.4% |
| Hatterene / SimpleHeuristics | 14.5% | 10.5% |
| Utrecht / MaxBasePower | 90.8% | 89.5% |
| Utrecht / SimpleHeuristics | 94.7% | 89.5% |

An independent Hatterene/SimpleHeuristics repeat scored 13.2%. At this sample
size none of the differences supports a general regression claim, especially
with ten fewer updates and a harder representation-learning problem. The
important conclusion is instead that the same roster dependence remains:
balanced/offensive rosters work, while the Trick Room/sun roster does not.

Opportunity-normalized diagnostics rule out the explanation that the required
actions simply never became legal. Against SimpleHeuristics, the Hatterene
policy selected Trick Room on only 1.9% of decisions where it was available,
redirection on 4.5%, Fake Out on 11.2%, and a Protect-family move on 11.5%.
Portland selected Protect-family moves on 2.9% of available opportunities and
redirection on 0.7%. At the same time, damaging friendly fire remained zero,
immune targeting stayed below 0.4% on Portland/Hatterene, and 42-49% of typed
opponent targets were super-effective. Basic mechanics are learned; coordinated
strategic modes are not.

### Preview and auxiliary information

Exact enumeration covers all 180 legal bring/lead choices against every other
roster in the 77-team pool.

| Fixed own roster | Entropy | Mean top probability | Distinct joint MAPs | Opponent/choice MI | Mean TV from mean policy |
|---|---:|---:|---:|---:|---:|
| Portland | 2.181 nats | 33.6% | 1 | 0.00038 bits | 0.82% |
| Hatterene | 3.800 nats | 8.6% | 1 | 0.00054 bits | 1.00% |

The preview policy sharpened roster-specific defaults but still barely reacts
to the opposing six. This is a structural learning bottleneck, not just a lack
of preview samples: all current-policy preview samples were trained in this run.
Meanwhile, the Hatterene fixed evaluation predicted the opponent's unordered
bring set exactly in 28.9% of games and its ordered leads in 3.9%. Useful roster
information exists in the representation but the own-choice head does not make
effective candidate-specific use of it.

The unknown-only hidden-information metrics became extremely high in-pool
(roughly 99% item/ability/Tera top-1 and 71% move recall@4). Those results are
not evidence of a general poker-like posterior: with only 77 recurring teams,
species and roster identity can memorize most labels. Full vocabulary logits
therefore should not be injected into the deployment policy. A compact shared
belief residual can be policy-consumed while its expensive semantic decoders
remain trainer-only, but it needs held-out-team and context-ablation evaluation.
The old opponent-action auxiliary also uses request-local action indices and
counts forced/default choices; it should be replaced by semantic, component-
masked intent targets before any of its predictions influence the actor.

### Throughput interpretation

The apparent SPS decline is partly a real success of the learned policy rather
than only a slower computer or larger encoder. Update 1 contained 784 completed
battles averaging 9.43 turns and took 150.8 seconds. Update 141 contained 1,618
battles averaging 4.46 turns and took 166.4 seconds. Both updates contain the
same 9,216 policy decisions, but the later policy incurs roughly twice as many
terminal resets and team previews and supplies roughly twice as many terminal
learning signals. Decision SPS therefore fell while battles per second rose.

That does not explain the full throughput regression. The previous architecture
averaged 70.98 wall SPS overall and 73.46 over its final 50 updates, with 1,585
battles/update at 4.56 turns. The event-aware run averaged 55.65 SPS overall and
55.56 over its final 50, with an almost identical 1,581 battles/update at 4.63
turns. Isolated construction of the new history costs only about 31 microseconds
per observation, and live-policy profiling attributes roughly 2.7 seconds per
update to the broader event/belief architecture. Frozen-opponent inference and
queue amplification account for more of the remaining gap than Python event
construction itself.

The clean efficiency target is the now-dominant reset/preview path. Team preview
has no prior public events or submitted intent, so its recent-event vector is
provably zero and the padded 12-step event GRU can be bypassed exactly on preview
and preview-value calls. Timing instrumentation remains optional and should be
disabled for production runs.
