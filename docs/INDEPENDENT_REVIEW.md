# Regulation F training review

22 November 2026

## Playing strength

Version 6 reached 2,027,520 learner decisions in 220 updates. On the original 15 held-out teams it continued to gain against earlier policies beyond the historical update-100 plateau. Each native comparison uses sampled policies. They exchange seats and rosters with the same battle seed. The 95% intervals resample unordered roster pairs together, and scalar replay verifies every winner.

| Candidate | Opponent | Wins / games | Score | 95% interval |
| --- | --- | ---: | ---: | ---: |
| v6 update 150 | v6 update 100 | 930 / 1,680 | 55.4% | 52.9–58.0% |
| v6 update 200 | v6 update 150 | 1,782 / 3,360 | 53.0% | 51.1–55.0% |
| v6 update 220 | v6 update 200 | 1,749 / 3,360 | 52.1% | 50.3–53.9% |
| v6 update 220 | v6 update 150 on 30 additional teams | 983 / 1,740 | 56.5% | 54.3–58.7% |

In the historical run v5 update 400 won 104/192 against its own update 100. On the historical Showdown simulator at level 50, v6 update 220 beat v5 update 400 in 126/192 games on the original panel and 507/870 on the 30 additional teams (58.3%, 55.0–61.5%). Each checkpoint used its own observation and action code. A fixed-bot diagnostic still exposes substantial matchup differences. Update 220 won 58/64 on Glimmora Sun, 27/64 on ICE QUEEN and 22/64 on Hard Trick Room against SimpleHeuristicsPlayer. The historical update 400 won 54, 33 and 9 respectively. These are greedy diagnostics with a fixed friendly roster. All learning used terminal wins, losses and draws.

Team preview became more sensitive to the opposing roster from update 100 to 220. With Hard Trick Room as the friendly team, the mutual information between the opposing roster and the policy's choice rose from 0.03 to 0.33 bits across the 30 additional teams. Its most likely four Pokémon stayed the same across all 30, while its preferred lead varied. The same measure rose from 0.07 to 0.81 bits on ICE QUEEN and from 0.11 to 0.46 on Glimmora Sun.

## Simulator and observations

The original simulator took imported levels literally on 42/85 training teams. The loader now sets all Regulation F members to level 50 before their stats are calculated. All six owner stats across the 600 original training and evaluation Pokémon matched Showdown. The policy sees its own stats and the four Pokémon brought to battle. Its switches use the same slots as its observations. Ability lookup accepts canonical IDs, restoring 258 previously unrecognized abilities in 510 training sets while preserving checkpoint indices. Exact opponent spreads, unrevealed abilities, moves and items are private training targets. [Official Regulation F rules](https://sv-news.pokemon.co.jp/en/page/166.html)

Pikapro's Rust VecEnv supplies the player views, event history and legal actions for training and routine policy matches. One observed difference from the April Showdown revision was a pre-switch speed tie that consumed a random draw in November. A separate Eject Pack case showed November reserves a queued switch target. The collector follows the live request, including replacement choices before turn one. Against the previous tracker both legal masks matched at 5,257 replayed decisions. Eight greedy actions changed. Substituting the direct features for fainted Pokémon accounted for all eight. Public move history still records the action. The reviewed source pins are in the [setup instructions](../README.md#setup).

## Training throughput

Measurements below include simulation, both players' inference and all PPO optimizer steps on the 4 GB Quadro T2000. Each row's control and candidate started from the same checkpoint. Rows from different checkpoints have different opponent populations.

| Change | Control seconds/update | Candidate seconds/update |
| --- | ---: | ---: |
| Build policy inputs directly from Pikapro, bypassing poke-env tracking | 123.52 | 70.13 |
| Submit ready opponent batches together and reuse public preview encodings | 67.45 | 57.28 |
| Retain four frozen models in GPU memory, 72 battles × 128 decisions | 65.21 | 62.69 |
| Run 144 battles × 64 decisions with four frozen models | 62.69 | 40.63 |

The frozen opponent uses a 36-row CUDA graph. Adding a second 16-row graph reduced padding from 9,308 to 4,132 rows per update in the earlier tracking collector, while full updates rose from 123.52 to 130.35 seconds. A graph for the learner saved 3.28 seconds in collection and produced essentially the same full update time as ordinary PyTorch, 118.01 versus 117.29 seconds.

The two layouts each collect 9,216 decisions and run 16 optimizer steps per update. All league opponents remain eligible with four resident models. In the 144-battle trial, collection took 22.02 seconds and PPO took 18.36 seconds, about 45% of the update. Frozen battle inference took 8.68 seconds, learner inference 4.83 and both previews 3.41. Rust battle steps and status checks took 0.24 seconds. Shorter trajectories brought the fraction of preview decisions reaching a terminal result inside the rollout from 94.7% to 89.4%. Other previews bootstrap from the battle value at the boundary.

Over 20 updates from checkpoint 200, total run time was 15m30s for the wider layout and 23m22s for 72 battles. The updates themselves took 15m20s and 23m11s. The wider policy won 1,732/3,360 at equal decisions (51.5%, 50.0–53.2%) and 1,712/3,360 at nearly equal training time (51.0%, 49.2–52.8%). These are paired comparisons of one training seed.

A separate trial against one frozen opponent compared 18,432 decisions and 32 optimizer steps after equal warmup. Each configuration ran twice in reverse order. With 144 battles and 64 decisions per rollout, the integrated Rust player-input build took 66.27 seconds versus 68.66 with the earlier extension. Gathering the frozen opponent's inputs by lane index brought this to 65.07 seconds (283 decisions/s). All 49 packed fields matched across 210 comparisons, including padded batches.

The integrated lane gather reproduced all 249 model tensors and 747 optimizer tensors after two matched CPU updates. A later GPU verification during archive checks took 84.54 and 88.22 seconds per 9,216-decision update. The driver reported thermal limiting on 6 of 16 samples, and available system memory was 0.8 GB at one reading. The relative speed of the two implementations under that desktop workload remains unmeasured.

Doubling rollout length to 128 took 94.01 seconds at the same decision and optimizer-step budget. Collection fell from 35.96 to 34.88 seconds, but PPO rose from 29.78 to 58.73. After 36,864 training decisions the longer-rollout learner won 826/1,740 against the shorter one on the 30-team panel (47.5%, 45.3–49.5%). The local configuration remains 144 × 64. A separate CUDA profile of one warmed 9,216-decision PPO update measured 21.48 seconds. The battle action head took 7.25 seconds of forward work and backpropagation across the whole network took 9.87 seconds. Gradient clipping and AdamW together took 0.13 seconds. The action head is the next forward pass to investigate.

## Training roster

Two learners started from update 220 with fresh AdamW optimizers and the same frozen update-220 opponent. Each received 460,800 battle decisions in 50 updates. The control used the original 85 rosters. The expanded pool added 85 owner-published teams with complete stat spreads and natures, extending species coverage from 88 to 116. The independent evaluation panel excludes all 185 training and development rosters by visible six-species combination. [Published team source](https://docs.google.com/spreadsheets/d/1nHEcS0r55JFVlC6EBX5zLgjwdnXTC9KOJnxIkduAs6Y/view)

Both learners beat their frozen training opponent on the original evaluation panel. The 85-team learner won 990/1,680 and the 170-team learner won 1,020/1,680. Head-to-head, the expanded learner won 1,685/3,360 on the original panel (50.1%, 47.9–52.4%) and 1,742/3,480 on the independent panel (50.1%, 48.4–51.7%). Against the earlier update-150 policy on the independent panel, the 85-team learner won 1,030/1,740 and the 170-team learner won 1,048/1,740. Their paired difference was 1.0 percentage point (95% interval −1.7 to +3.7). The common parent won 983/1,740. The original 85 remain the training default. The expanded pool is available by manifest.

On the same 90 openings from the 30-team panel, the expanded learner's recall of unrevealed moves among its four predictions rose from 57.8% to 63.6%. Its top ability prediction improved from 70.9% to 79.1%. Item accuracy moved from 48.1% to 50.8%, while item negative log likelihood fell from 6.03 to 4.20. Tera negative log likelihood fell from 5.75 to 3.56. The larger pool improved the probabilities assigned to the true hidden attributes. Even with 170 teams it inferred only about half the unrevealed items correctly.

## Move order

Two models trained from scratch for 50 updates on the original 85 teams with seed 8719. They started with the same 246 shared tensors and received 460,800 decisions each. The experimental model encodes each move with its PP, pools the available moves and shares action weights across move slots. When the four move slots were independently permuted and legal actions relabeled, its greedy action stayed the same on all 508 sampled decisions. Its largest joint probability difference was below 0.000001. The ordinary model changed its choice on 137 of the same decisions.

The move-set model scored 52.3% at update 25 (1,756/3,360, 50.7–53.9%). At update 50 it scored 50.7% on the original panel (1,704/3,360, 49.0–52.4%) and 50.5% on the additional panel (1,758/3,480, 49.0–52.0%). Against the main update-220 policy the two fresh models won 667 and 666 of 1,740 games. The experiment establishes move-order invariance. A longer run with another seed would test whether it improves playing strength. The default trainer uses the update-220 architecture.

The research archive `pokeRL-research-archive/2026-11-21` contains the per-game results, timing traces, checkpoints and experimental source. Its `RESTORE.md` identifies the files for each comparison.
