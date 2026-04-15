# Known Issues

## Will likely need fixing on first run

- **poke-env API names**: `battle.can_tera` (might be `can_terastallize`), `pokemon.effects` (might be `_effects` or different name for volatile statuses), `move.target` attribute. All guarded with `hasattr` or will throw clear errors -- fix by checking poke-env source or error messages.

- **`side_conditions` type**: code uses both `in` and `.get()` on `battle.side_conditions`. If poke-env returns a set (not dict), `.get()` will fail. Check actual type and adjust.

## Performance

- **Sequential env stepping**: `ParallelBattleEnv.step()` steps envs one at a time. Each blocks until its turn resolves. With 16 envs this is ~16x slower than true parallel stepping. Fix with `ThreadPoolExecutor` or async stepping if throughput becomes the bottleneck.

- **LSTM truncated BPTT**: per-step hidden states are stored and restored, so transitions can be shuffled freely in PPO. The tradeoff is gradients don't flow back through time. Standard simplification, fine for v1.

## Not yet implemented

- **Learned team preview**: team preview uses a fixed heuristic (`/team 1234`). The `TeamPreviewHead` exists in the model but isn't wired into training. `ShowdownPlayer.set_team_preview_fn()` is the hook point.