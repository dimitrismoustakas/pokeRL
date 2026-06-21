**On Improvements**

Highest ROI

Re-factor Tera into its own decision instead of baking it into every move action. Right now the flat action space creates a tera and non-tera twin for many moves, and the entropy bonus can unintentionally push exploration toward early tera. This is the first thing I would investigate for the “worse early tera” behavior.
Add explicit last-turn event features. The cheap version is: previous own action ids, observed opponent action ids, move order, switches, KOs, protect success/failure, and maybe whether damage was taken before moving. These are observation features, not reward terms. They should help strategic timing more than making the model bigger.
Improve the team preview training algorithm, not just the preview head. The current preview decision is still a sparse terminal-reward problem, so collapse is not surprising. A preview-specific critic/baseline, or at least a more stable preview update path, is probably higher leverage than more preview capacity alone.
Add hidden-information auxiliaries. Predict opponent item, ability, unrevealed moves, and tera type from legal history, using eventual reveals as labels. That usually helps strategic play without changing the reward objective.
Increase opponent diversity in training. A larger and more stratified self-play league, plus a few fixed heuristic opponents, is very feasible on one machine and often matters more than architecture tweaks.
Very plausible training-dynamics issue
The tera problem may be partly training dynamics, not just policy quality. In a flat action space, entropy is paid over duplicated tera/non-tera branches. That can bias the policy toward using tera too freely even if the value function is not strong enough to justify it. So I would prioritize “better action factorization” over “more search”.

Good medium-priority options

Make the battle action head hierarchical: choose move/switch first, then target, then tera gate. That is a cleaner credit-assignment problem than one big categorical over all composite actions.
Give memory better turn-level summaries. Even with the new opponent-reserve access, the recurrent state still mostly reasons from compressed turn snapshots. A compact turn-summary token per step would help long-horizon play.
Strengthen preview supervision slightly if needed: bring marginals are good; if preview is still brittle, softer lead targets or lead marginals may generalize better than only exact ordered leads.
MCTS
Full online MCTS for doubles battle play is probably overkill on your machine right now. The branching factor is large, the game is partially observed, and belief-aware search would crush training throughput.

Where search is not overkill:
Team preview only.
Opening turn only.
Offline label generation on a small subset of states.
Top-k guided lookahead rather than full tree search.

So if I were prioritizing the next steps, I’d do:
Tera/action factorization or entropy fix.
Preview-specific critic/stabilization.
Last-turn event/action features.
Hidden-info auxiliaries.
More diverse league/curriculum.


**On efficiency**
The two biggest remaining cheap wins are still profiling overhead, not model math.

Gate the train-loop microtimers in train.py (line 518) and train.py (line 578).
Right now all those perf_counter() measurements run every step even when summary_path is not used. That is pure overhead on the hot path.

Make env timing a no-op unless explicitly enabled.
In showdown_env.py (line 51) and throughout the file, the env still does a lot of perf_counter() calls plus locked counter updates on every reset/step/emit. That can definitely hurt SPS and also add jitter.

After that, the next tier would be:

Reduce league_coalesce_ms.
The default in config.py (line 67) is 10.0 ms, and league.py (line 166) really does wait for that batching window. On a local T2000 setup, that can absolutely cause occasional rollout stalls. I’d try 0, 1, or 2 ms and benchmark.

Remove hot-path dict churn around masks.
Now that the mask payload is smaller, the next cleanup is mostly Python overhead: passing joint/can_tera as separate tensors instead of dicts through action head, PPO, eval, and league inference.

Small model-side cleanup.
Compute static action features once per ActionHead.forward() instead of recomputing them for both slot scorers in heads.py (line 156) and heads.py (line 185). This is a smaller win than the timing changes.
