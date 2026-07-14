# Evaluation and quality plan

This document records the evaluation methodology and the main findings from the
150-update run, the interrupted event-aware run, and the completed 400-update
matchup-belief run. The machine-readable summaries live under `logs/` and are
intentionally ignored by Git.

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

1. **Make the self-play population permanent and payoff-aware.** Replace the
   rolling FIFO window with an on-disk permanent archive and discounted PFSP.
   The two inference groups should receive distinct historical opponents when
   possible, while a uniform component preserves broad coverage. Dedicated
   exploiter policies are deferred until the payoff matrix demonstrates a
   cycle or persistent blind spot that PFSP alone does not repair.

2. **Fix preview optimization before changing its objective.** Reuse each
   current-policy preview sample with clipped on-policy PPO, including value
   clipping and direct KL/clip diagnostics. Only if this still collapses to an
   opponent-insensitive default should a candidate-value/Q preview objective
   be trialled. Continue to report exact opponent/preview mutual information
   and learned/max-BST/random preview ablations; do not reward variation for
   its own sake.

3. **Measure strategic moves with semantic denominators.** Trick Room must be
   reported per selectable slot-decision and per battle. Fake Out must separate
   live first-active-turn choices from dead later choices. Protect must separate
   fresh use, same-Pokemon consecutive use, and simultaneous ally use. These
   remain evaluation probes, not scripted rewards or training opponents.

4. **Keep the LSTM through this iteration.** Explicit public events and own
   intent repair a real information omission while recurrence preserves longer
   history. Measure full-history versus reset-hidden inference before replacing
   memory.

5. **Keep held-out teams in the backlog.** A legal archetype-stratified team
   population remains necessary to distinguish general beliefs from memorized
   roster identity, but it should not be approximated with randomly assembled
   invalid or strategically incoherent teams. Until a sound source exists,
   report the 77-team panel explicitly as in-distribution.

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

## Completed matchup-belief run at update 400

The replacement run completed all 400 updates: 3,686,400 policy decisions and
628,523 battles in 68,420.8 seconds (19 hours 0 minutes). Mean/median wall-update
throughput was 54.90/56.45 SPS, with a 22.92-70.44 range. Excluding the first
five updates gives 54.87/56.44 SPS and the same range. Updates 301-350 averaged
56.92 SPS; the final 50 fell to 46.72 while the host was also doing other work,
so the overall mean is not a clean isolated-machine ceiling.

All 72 persistent learner players and all 72 persistent opponent players
retained exactly one live battle at every update. CUDA allocation stayed within
230-560 MB and reservation within 3.04-3.41 GB. The previous battle-retention
leak and reset-reservation race did not recur.

### Learning trajectory and fixed probes

Each frozen-anchor comparison contains 128 mirrored sampled-policy games.

| Final checkpoint opponent | Wins-losses | Win rate | 95% interval |
|---|---:|---:|---:|
| Initialization | 102-26 | 79.7% | 71.9-85.7% |
| Update 100 | 65-63 | 50.8% | 42.2-59.3% |
| Update 200 | 64-64 | 50.0% | 41.5-58.5% |
| Update 300 | 68-60 | 53.1% | 44.5-61.6% |
| Itself | 56-72 | 43.8% | 35.5-52.4% |

The final policy is decisively stronger than initialization but is not
distinguishable from updates 100, 200, or 300. The self-match interval includes
50%, so its asymmetric point estimate is sampling and simulator noise, not
evidence that the final policy loses to itself. The central result remains a
large early gain followed by a long learning plateau.

Each fixed-bot cell below contains 76 games against all other in-pool rosters.
Bots remain deterministic evaluation probes only.

| Fixed model roster | Random | MaxBasePower | SimpleHeuristics |
|---|---:|---:|---:|
| Portland balance | 69.7% | 48.7% | 32.9% |
| Hatterene Trick Room/sun | - | 23.7% | 14.5% |
| Utrecht balance | - | 93.4% | 90.8% |

This is not a generally strong policy. Utrecht is already easy for its learned
offensive mode, Portland is mixed, and the Hatterene roster still fails to
execute its defining strategy.

### Mechanics diagnostics

The semantic evaluator was rerun on four 76-game sampled-policy panels. Tera
was used in 71/76 Hatterene games against each bot, 72/76 Utrecht games, and
75/76 Portland games. Conditional turn-one use was 60.6%, 56.9%, and 78.7%
respectively, and both active slots were selected often. Every non-Tera game
ended on turn 2-4; Utrecht won all four of its non-Tera games. Near-universal
use remains the expectation, but evaluation should not encode an unconditional
rule: declining a type change in an already decisive short win can be rational.

Trick Room is a clear failure under either denominator. Hatterene selected it
in 3/119 eligible setup slot-decisions spanning 3/71 opportunity-games against
SimpleHeuristics, and 2/112 decisions spanning 2/72 games against MaxBasePower.
The greedy action mode selected it in 0/72 opportunity-games against
SimpleHeuristics when rerun with the checkpoint's original observation
semantics. The setter was present and able to act; this is not primarily a
preview-denominator artifact.

Fake Out timing was not learned. Against SimpleHeuristics, Utrecht selected it
6/140 times while the user was on its live first active turn, but 16/169 times
after that condition had expired and the move was guaranteed to fail. The
Hatterene panel was similarly inverted (3/77 live versus 4/47 late). Greedy
inference also chose late Fake Out more often than live Fake Out, so the sampled
tail is not the explanation.

Protect behavior is roster-dependent but includes obvious over-chaining. The
sampled Hatterene policy repeated a Protect-family move after a successful
Protect on 19/44 available decisions against SimpleHeuristics and 27/65 against
MaxBasePower, succeeding only 6 and 8 times. It attempted a fourth consecutive
Protect once. Greedy Hatterene repeated on 37/59 opportunities; greedy Utrecht
repeated on 9/11. By contrast, sampled Utrecht repeated on 0/8 and Portland on
6/37, so the policy has some contextual variation but no reliable general rule.
Simultaneous ally Protect remained rare (4/438 and 6/453 Hatterene joint
decisions) and is a separate behavior.

These mechanics are structurally learnable: the observation exposes
`first_turn`, `protect_counter`, `last_move`, and timed field state to the action
head. One information defect was nevertheless repaired for the next run. Timed
conditions previously encoded both absence and their activation turn as zero;
absence is now zero while active Trick Room, screens, and Tailwind are positive
from the turn they begin. The remaining failure is chiefly sparse action-credit
learning and exploration, not an LSTM capacity bottleneck.

Basic target selection remains much healthier: damaging friendly fire was zero
on all Portland panels and 0.1-0.2% on Hatterene, while immune targeting was at
or below 0.8% there. That competence does not compensate for the missing
multi-turn strategies.

### Preview, beliefs, and event history

Exact enumeration still finds a collapsed preview policy despite a measurable
opponent signal.

| Roster / checkpoint | Entropy | Joint MAPs | Opponent/choice MI |
|---|---:|---:|---:|
| Portland / initialization | 4.661 nats | 1 | ~0 bits |
| Portland / update 200 | 2.290 nats | 2 | 0.0181 bits |
| Portland / final | 0.317 nats | 1 | 0.0135 bits |
| Hatterene / initialization | 4.653 nats | 1 | ~0 bits |
| Hatterene / update 200 | 3.504 nats | 1 | 0.0107 bits |
| Hatterene / final | 1.195 nats | 2 | 0.0104 bits |

For Hatterene, the two final joint MAPs differ only in lead order; the greedy
bring set is still unique across all 76 opponents. Against SimpleHeuristics,
learned/max-BST/random preview scored 32.9/34.2/22.4% on Portland and
14.5/10.5/10.5% on Hatterene. Those small, noisy panels show that roster synergy
matters, but they do not demonstrate reliable opponent-conditioned selection.

Unknown-only in-pool item, ability, and Tera accuracy finished near 100%, with
move recall@4 at 91.1%. This is still compatible with memorizing 77 recurring
teams. In direct 128-game mirrored ablations, the full policy scored 43.0%
(34.7-51.6%) against a policy with the belief adapter zeroed and 47.7%
(39.2-56.3%) against one with the recent-event encoder zeroed. Both overlap the
final self-match result, so neither module has demonstrated a deployment-strength
benefit yet. That is a reason to improve learning and diagnostics, not to remove
causal information from the model after one in-distribution run.

### PPO assessment and accepted next iteration

Battle PPO remained numerically healthy. Over the final 50 updates, mean
pre-update KL was 0.00060, post-update KL 0.01836, clip fraction 0.166, and
value explained variance 0.734. CUDA memory was bounded and no non-finite update
was reported. Action entropy nevertheless fell from a mean 2.78 over updates
1-50 to 0.92 over updates 351-400 while strength had already plateaued. The
problem is learning coverage and sample use, not an unstable PPO implementation.

The accepted next iteration has now been implemented as follows:

1. Every frozen checkpoint is kept in a permanent archive. The league maintains
   discounted current-policy payoff evidence per opponent and PFSP-samples
   opponents that remain difficult, with a uniform coverage component. It
   assigns distinct draws to the two inference groups whenever the archive
   permits it. Do not add
   dedicated exploiters yet; first determine whether this simpler population
   fixes the plateau and exposes stable payoff cycles.
2. Preview's one-use terminal update is now clipped on-policy PPO, so
   each current-policy sample receives bounded multi-epoch reuse and logs its
   own KL/clip behavior. Run this diagnostic before introducing a candidate-Q
   objective, whose unchosen-action estimates would add extrapolation risk.
3. The fixed panels were rerun with live/dead Fake Out, fresh/chained Protect,
   simultaneous ally Protect, and per-slot and per-battle Trick Room metrics.
   They exposed concrete failures rather than a denominator problem. The next
   model therefore learns selected-action public consequences (execution,
   failure/miss/CANT, target CANT, and next field) and feeds detached learned
   public-transition predictions into the policy. This remains sparse
   terminal-reward learning;
   it does not add heuristics, shaped rewards, or simulator-private outcomes.
4. Keep periodic frozen-anchor and small fixed-probe evaluation cheap and
   separate from training. Keep dedicated exploiters and a legal held-out team
   population in the backlog rather than substituting bots or randomly
   assembled teams.
5. Frozen-opponent CUDA graphs are enabled for production rollout inference.
   A two-update same-seed A/B preserved action/RNG parity, used one capture and
   1,299 replays with zero fallbacks, reduced aggregate rollout time by 6.27%,
   and improved wall throughput by 6.13% without changing PPO work or reserved
   CUDA memory. This is an efficiency change only; terminal win/loss and the
   learned policy are unchanged.
