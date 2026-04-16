# Known Issues

## Performance

- **Single-battle latency**: localhost is the intended setup, and `ParallelBattleEnv` now steps envs concurrently. Individual random doubles battles are still relatively slow to finish end to end, so long smoke battles can exceed short terminal timeouts even when the battle loop is progressing correctly.

- **LSTM BPTT over the full rollout**: The PPO update unrolls the LSTM through all `rollout_steps` (default 128) per minibatch sequence. This retains the full computation graph in memory for the duration of the LSTM loop. For much longer rollouts or significantly larger models the memory cost could become an issue; truncated BPTT (processing the sequence in fixed-size chunks and detaching between them) would be the fix.

## Observation / Feature gaps

- **Item vocabulary is mostly hash-based**: `GameData._known_item_ids` only collects items from the pokedex's `requiredItem`/`requiredItems` fields (Arceus plates, Genesect drives, etc.). Common competitive held items (Choice Band, Leftovers, Life Orb, …) are not in the named vocab and fall back to hash-bucket indices. There is no stat-based initialisation for items either. Item representations for the vast majority of held items will start as colliding hash indices rather than learned embeddings tied to meaningful identifiers.

- **Opponent PP dimensions are always zero**: The four PP features in `CONTINUOUS_DIM` are populated only for own pokemon (`full_info=True`). Opponent pokemon always have zeros there. Fixing this properly would require separate `continuous_dim` values for own vs. opponent pokemon in `PokemonEncoder`, which adds architectural complexity. For now these four dimensions serve as an implicit "is own pokemon" signal, which the model can learn.

- **`_attach_choice_timer` timing is unreliable for async opponents**: The closure wraps `choose_move` synchronously. For `LeaguePlayer` and `MaxBasePowerPlayer` (both synchronous) timing is accurate. If an opponent's `choose_move` is a coroutine the wrapper will measure ~0 ms (the coroutine object is returned immediately without being awaited). This only affects profiling output, not correctness.
