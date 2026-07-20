# Selected-order construction and target-aware coalescing

Status: archived on 2026-07-17.

The experiment was useful, but neither runtime switch belongs in production.
pokeRL now has one action-order path and one league-coalescing policy:

- legal actions retain request-local `Move`, `Pokemon`, and pass templates;
  poke-env orders are constructed only for the two actions selected by the
  policy;
- frozen-opponent inference collects requests until the configured coalescing
  deadline, then processes the frozen snapshot in checkpoint/action-kind
  groups.

The alternate eager-order and early-wakeup implementations were removed rather
than retained as compatibility or “just in case” branches. This note preserves
enough information to repeat the test on a temporary branch.

## Hypotheses

Selected-order construction avoids allocating every legal
`SingleBattleOrder`, `PassBattleOrder`, and `SkippedBattleOrder` before the
policy chooses two actions.

Target-aware coalescing ended the normal 5 ms inference wait as soon as one
compatible frozen-checkpoint battle queue reached the fixed 16-row CUDA-graph
capacity. The hypothesis was that a full graph replay could no longer become
more efficient by waiting.

## Historical implementation

- `a418c1848bdcb0451bbbc8efa138f8097dc5c6b2` introduced selected-order
  construction.
- `4e33699ad11b4027b8574fe47680bc216180a092` introduced the target-aware
  wakeup. That commit also contains valuable checkpoint validation and
  snapshot metadata hardening; do not revert or transplant it wholesale.
- `01a9eab3f421b88d0731c2e7eecdd734e4469c7a` added exact eager/deadline
  controls.
- `fdaee814d881064d7e2b9c9fb4a102ea6ae205d3` added the restart-safe
  factorial harness.
- `8fb5ff93107e3985728367a533fcb0ab1a498d3d` is the exact tested pokeRL tree.

## Design

The production experiment was a 2x2 factorial:

| Cell | Order construction | Coalescing |
| --- | --- | --- |
| Q | eager | deadline |
| L | selected only | deadline |
| C | eager | target aware |
| B | selected only | target aware |

All cells resumed from one update of the B configuration, giving them the same
weights, optimizer, RNG, league, and team-sampler boundary. Four Williams
blocks used the rows `QLBC`, `LCQB`, `CBLQ`, and `BQCL`. Each cell ran 10
updates; the first 3 were discarded as invocation warm-up, with a 30 second
cooldown between cells.

The common production configuration used 72 environments, 2 opponent groups,
24 BattleStream workers, 3 battles per worker, 2 runtime loops, 128 rollout
steps, 4 PPO epochs, and 4 minibatches. Detailed timing was disabled.

A separate simulator test ran six alternating Q/L pairs for 128 vector steps
per cell.

## Pinned identity

- pokeRL: `8fb5ff93107e3985728367a533fcb0ab1a498d3d`
- poke-env: `529b92ca8f6433d8131b46ba689cb152a8bb4fd4`
- Pokemon Showdown: `2f66b8495560d5f786db1914173c981fa85f2ff8`
- team manifest SHA-256:
  `e2395b772d17df56cfd650389c0c96c411b54224d851f2e9dda4c06ae82438e6`
- normalized training-config fingerprint:
  `a82bbd81e94c3e4354e4fef1639c229c229d13f54638fd4498eaa2995d71218b`
- Python 3.13.7; PyTorch 2.11.0+cu130; Node 22.17.0
- NVIDIA Quadro T2000, driver 595.97

## Results

For end-to-end wall SPS:

- selected-order construction: -1.37% (95% CI -2.54% to -0.18%); 3 of 8
  paired constituents were positive;
- target-aware coalescing: -2.10% (95% CI -5.35% to +1.27%); 3 of 8 paired
  constituents were positive;
- interaction: -1.45% (95% CI -10.96% to +9.08%).

The isolated simulator test found a +5.63% selected-order effect (95% CI
+1.58% to +9.84%), positive in all 6 pairs. The local allocation reduction is
real, but it was too small relative to complete training to establish an
end-to-end gain on this machine.

The target-aware code activated 11,790 times. It changed CUDA-graph padding
only from about 29.61% to 29.35%, and produced no graph fallbacks or errors.
All production cells retained 72/72 player and opponent battles.

The laptop slowed during the sequence: block means were approximately 73.27,
69.71, 69.12, and 69.38 wall SPS. The balanced design limits ordinary order
bias, but this nonlinear drift means the negative point estimates should not
be treated as precise intrinsic slowdowns. The defensible conclusion is that
neither factor established a production benefit. In the final, relatively
stable block both effects were close to neutral.

The ignored raw artifacts were written under
`benchmarks/efficiency_factorial_20260716/` (`report.md`, `aggregate.json`,
`aggregate.csv`, and `experiment.json`) and were deleted during the subsequent
workspace cleanup. The identities, protocol, effects, and uncertainty needed
to reproduce or interpret the experiment are retained in this document.

## Historical rerun

The removed harness can be run without restoring gates to current production
code by checking out the exact experiment in a detached worktree:

```powershell
git worktree add --detach ..\pokeRL-efficiency-retest 8fb5ff93107e3985728367a533fcb0ab1a498d3d
Set-Location ..\pokeRL-efficiency-retest
uv sync --dev
& .\.venv\Scripts\python.exe .\scripts\benchmark_efficiency_factorial.py --output-dir .\benchmarks\efficiency_factorial_20260716
```

Before starting the harness, point that dedicated environment at the recorded
poke-env revision and the configured Showdown checkout at its recorded
revision; the harness verifies both identities. For a test against future main,
create a disposable experiment branch, port the small factor seams from
`01a9eab` and the harness from `fdaee81`, and remove them again after
measurement.

A future retest should use at least eight balanced blocks, a longer preheat,
recorded randomized or reversed Williams-row order, and lightweight CPU/GPU
temperature, clock, and power telemetry. Keep the end-to-end SPS and
retained-battle checks as the decision criteria.
