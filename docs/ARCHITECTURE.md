# Architecture

## Battle simulation and observations

[Pikapro’s Rust vector environment](../vendor/pikapro-player-inputs.patch) runs Regulation F doubles in process. Each lane holds one battle. Both players bring four Pokémon from six at level 50. The collector records each seat’s own exact stats, public opposing HP, revealed moves and items, field state, recent events and legal actions. The player sees when an item was lost and which stat Protosynthesis or Quark Drive currently boosts. Private opponent attributes are available separately as training labels.

Our Rust changes are in the [Python extension binding](../vendor/pikapro-player-inputs.patch#L27), [public action tracker](../vendor/pikapro-player-inputs.patch#L840) and [player-view updates](../vendor/pikapro-player-inputs.patch#L1458). The build uses the complete patched Rust sources in [the Pikapro source distribution](../vendor/pikapro-0.2.0+pokerl.1.tar.gz). On Windows the result is an importable Python extension (`.pyd`). [The Python collector](../pokerl/env/vector.py) feeds the arrays to PyTorch, batches opponents by checkpoint, and resets finished lanes. Pikapro also runs paired policy evaluation. Pokémon Showdown checks the legality of external teams on import.

## Policy

[The encoder](../pokerl/model/encoders.py) applies the same network to each known move and its remaining PP, then sums the resulting vectors for each Pokémon. Padded move slots contribute zero. Reordering a moveset preserves its representation. Attention encodes the two teams. Field conditions, recent events and active-slot mechanics join the team representations. An LSTM carries history across turns, and a residual connection makes the current board available directly to the heads.

At team preview the policy chooses one of 15 four-member subsets, then one of 12 ordered lead pairs within it. The resulting 180 choices share the spatial encoder and have a dedicated value estimate. The preview cache reuses the parts of a team’s encoding that are constant while the policy weights are fixed.

[The battle head](../pokerl/model/heads.py) chooses the first active Pokémon’s action, the second Pokémon’s action conditional on the first, then whether and where to Terastallize. A move candidate carries its identity, target and mechanics. All four menu positions share the scoring weights. Permuting the live menu permutes the corresponding scores and legal masks. The selected indices are mapped back to simulator orders.

The policy also predicts hidden attributes, opponent actions and public consequences. The action head can use predictions derived from the player’s observations. Exact supervision stays in the rollout and loss computation. A positive `--aux-damage-coef` trains the move-and-target head to predict the fraction of visible HP lost to a selected attack. The reward is the battle result.

## Learning and evaluation

Each battle returns +1 for a win, −1 for a loss and 0 for a draw or an intermediate step. [PPO](../pokerl/training/ppo.py) trains the battle and preview heads together. Battle returns use generalized advantage estimation with γ = 1 and λ = 0.95. Preview returns use the battle outcome when available in the rollout and bootstrap from the battle value at its boundary. Minibatches preserve recurrent sequences and reset memory at battle boundaries.

Self-play periodically freezes the learner. The [league](../pokerl/training/league.py) samples archived opponents using the current learner’s recent win rates, with uniform sampling mixed in for coverage. The default draws at random. With `--league-sampler balanced`, each refresh favors checkpoints with the largest deficit against their expected number of assignments. A fixed opponent plays every new battle. A trainer save contains model weights, optimizer, learner counters, exploration schedule, random states, opponent assignments and league membership. The frozen policies themselves live on disk. On resume the vector environment starts new battles.

[Paired evaluation](../scripts/evaluate_match.py) plays both seats against another saved policy. It reports wins, draws and losses, and estimates a score interval by resampling roster pairs.
