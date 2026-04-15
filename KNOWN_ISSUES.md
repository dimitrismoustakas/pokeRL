# Known Issues

## Training

- **Multi-env reset**: single-env training smoke runs complete, but more than one `BattleEnv` in the same process currently hangs during reset before rollout collection begins. The default `num_envs=16` training path is therefore not ready for a real run yet.

## Performance

- **Single-battle latency**: localhost is the intended setup, and `ParallelBattleEnv` now steps envs concurrently. Individual random doubles battles are still relatively slow to finish end to end, so long smoke battles can exceed short terminal timeouts even when the battle loop is progressing correctly.

## Not yet implemented

- **Learned team preview**: team preview uses a fixed heuristic (`/team 1234`). The `TeamPreviewHead` exists in the model but isn't wired into training. `ShowdownPlayer.set_team_preview_fn()` is the hook point.