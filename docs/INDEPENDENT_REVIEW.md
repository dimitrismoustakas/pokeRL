# Independent review brief

Updated: 2026-07-21.

This is the current decision brief for pokeRL training. It replaces earlier
review state rather than recording a chronological diary. Code, run summaries,
and evaluation artifacts remain the detailed evidence.

## Current decision

The next useful model must be trained from scratch with architecture version 5.
Version 5 combines four corrections:

- an exact, permutation-equivariant team-preview policy;
- one joint preview-and-battle PPO update from a single behavior policy;
- battle actions keyed by Showdown request slots and the exact live request
  move identities; and
- the exact public Perish Song countdown rather than a binary present flag.

The request-slot defect is the reason not to resume or evaluate version-4
checkpoints as current baselines. When Showdown omitted disabled moves, the old
adapter compacted the remaining list. The command stayed legal, but the model
scored it using a different roster move's identity and mechanics. Choice locks,
Taunt, Encore, Disable, PP exhaustion, recharge, and Struggle could all trigger
wrong candidate semantics. Both the websocket and local BattleStream backends
shared this pokeRL adapter, so the defect was not specific to poke-env's local
backend.

AdamW with learning rate `2e-4` and weight decay `0.05` is the working optimizer
for the first version-5 run. It is not declared optimal. The immediate run is a
10-update correctness and gradient diagnostic; if it is clean, the same run is
continued to update 80 and evaluated before spending a 450-update budget.

## Non-negotiable boundaries

- Reward is the sparse terminal result: win `+1`, loss `-1`, and the draw
  outcome. No HP, damage, knockout, board, or imitation shaping is added.
- Inference receives only information legitimately available to a player.
  Exact packed teams and simulator decisions may be masked trainer-only labels,
  never policy observations.
- Evaluation, best-response training, population construction, and admission
  remain separate explicit stages.
- Active research has one production path. There is no checkpoint converter or
  runtime compatibility branch for retired architectures.
- Opponent intent remains an auxiliary target. A public-prediction policy
  adapter requires a later controlled comparison.

## Corrected training semantics

### Team preview and PPO

Preview defines one exact distribution over all 180 legal bring-four and
ordered-lead choices. Live inference uses a factorized fast sampler; exact
training statistics and deterministic evaluation expose the complete
distribution without repeating an observation 180 times.

Preview is the zero-reward first transition of a battle. Production uses
`gamma=1`, matching undiscounted win probability. A preview target ends at an
observed terminal result or bootstraps once from the battle value at the rollout
boundary. Long battles therefore keep their preview sample rather than being
excluded.

Battle and preview data are optimized together. Four epochs by four minibatches
produce 16 combined backward passes and optimizer steps. The full old preview
distribution is frozen before the first step for exact `KL(old || current)`;
the selected action-time log probability remains PPO's behavior reference.

### Live move requests

Move action slots now follow each move's one-based `request_index`. Disabled
moves leave holes instead of shifting later actions left. The observation also
carries the exact move ID currently offered in each request slot, so locked
single-move requests, copied or transformed moves, recharge, and Struggle do
not inherit unrelated roster-move semantics. Missing, out-of-range, or duplicate
request indices fail loudly.

### Public timed state

The existing Perish Song feature now distinguishes all public states without
changing the observation width: absent `0`, then `PERISH0/1/2/3` as
`0.25/0.5/0.75/1.0`. Showdown emits successive countdown effects and poke-env
can retain earlier ones, so the lowest present count is the current count.

### Optimizers and diagnostics

The trainer exposes Adam and AdamW. AdamW applies decoupled weight decay only to
non-embedding matrix parameters; embeddings, vectors, and normalization
parameters remain in the no-decay group. Run summaries record the complete
optimizer and PPO settings.

Global gradient clipping remains at `0.5` for the next diagnostic so only the
architecture changes. The trainer now records the pre-clip mean, maximum, and
clipped-step fraction. The opt-in shared-gradient diagnostic measures each
objective's memory-path norm and pairwise cosine before any decision about the
clip threshold or auxiliary routing.

## Historical evidence

### Version-3 and corrected-preview runs

The 440-update version-3 run learned substantial strength but plateaued late.
Its team-preview policy and PPO scheduling had the defects described above. A
fresh 10-update run verified the corrected preview lifecycle: every preview
start was trained, boundary bootstraps were present, PPO used 16 joint steps,
and behavior-policy disagreement stayed at floating-point noise.

These artifacts remain useful for diagnosis, but their checkpoints are not
inputs to version-5 training.

### Version-4 optimizer study

Fresh 80-update Adam and AdamW runs established that optimizer choice was not
settled by training curves alone. In the direct 256-game mirrored argmax
comparison, AdamW `2e-4`, weight decay `0.05` scored 117-139 against the Adam
baseline: `45.7%`, 95% interval `39.7%-51.8%`. That does not establish AdamW
superiority. AdamW remains the working setting because it was stable and was
used for the longer study, not because the search is complete.

Muon still warrants a controlled experiment after the corrected architecture
has a stable baseline. It must use an explicit Muon/AdamW parameter partition;
it should not be added before the action semantics and gradient diagnostic are
validated.

### Version-4 450-update main run

The AdamW main run learned real strength: on the held-out mirrored sampled
schedule, update 450 scored 220-36 against initialization (`85.9%`). It also
beat update 100 by 117-75 (`60.9%`). Later progress was not established:
update 450 scored 134-122 against update 250 and 108-84 against update 350,
with both 95% intervals crossing 50%. This is evidence of a late plateau, not
proof that learning stopped completely.

The request-slot defect means this run cannot answer how a correctly aligned
policy would learn disabled or forced move states. Its logs and evaluations are
historical evidence only.

### Version-4 standalone main exploiter

The 80-update exploiter was mechanically trained against the frozen update-450
main. It scored 145-111 on the training pool (`56.6%`, interval
`50.5%-62.6%`) and 138-118 held out (`53.9%`, interval `47.8%-59.9%`). This is
at most a weak nearby best response, not credible broad exploitation.

The optimizer was not numerically frozen: parameters moved and finite gradients
were present, although global clipping occurred on about 97% of optimizer
steps. Raising the clip threshold without measuring objective norms and
interference would not be principled. More importantly, this exploiter inherited
the version-4 move semantics and a concentrated main policy, so it is not a
valid test of the corrected path.

The old main did learn some multi-turn behavior, including Trick Room and
Tailwind use. Perish Song was a weak learnability probe because only four of 85
training teams contained it, all on the same set, with no trapping support and
no exact countdown observation. A future Perish test needs a small roster set
that actually supports the strategy while retaining terminal-only reward.

## Next evidence required

1. Run 10 fresh version-5 updates with AdamW `2e-4`, weight decay `0.05`, and
   the shared-gradient diagnostic enabled.
2. Check request semantics, preview PPO/KL, gradient norms and cosines, clipping,
   memory, and throughput before changing any training hyperparameter.
3. If clean, resume the same run to update 80 and compare learned strength,
   preview behavior, and mechanic-specific actions.
4. If the update-80 evidence is sound, resume to update 450 and build a new
   league.
5. Only then repeat the standalone main exploiter and test a league exploiter
   against a complete frozen population.
6. Continue optimizer work with Muon and later tune clipping, entropy, learning
   rate, decay, and update geometry in controlled experiments.

## Evidence locations

- `logs/selfplay_public_state_adam_baseline_80_20260718.summary.json`
- `logs/selfplay_public_state_adamw_lr2e4_wd5e2_80_20260719.summary.json`
- `logs/selfplay_public_state_adamw_lr2e4_wd5e2_450_20260719.summary.json`
- `logs/main_exploiter_u450_s1_80_20260720.summary.json`
- `logs/evaluation/optimizer_adamw_lr2e4_vs_adam_80_20260719/`
- `logs/evaluation/selfplay_public_state_adamw_lr2e4_wd5e2_450_20260719/`
- `logs/evaluation/main_exploiter_u450_s1_80_20260720/`
