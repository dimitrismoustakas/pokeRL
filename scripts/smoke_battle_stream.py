"""Quick smoke test for BattleStreamEnv.

Runs one full battle (agent=random, opponent=random) via the subprocess
battle stream and reports how long it took vs the WebSocket approach.

Usage:
    python scripts/smoke_battle_stream.py
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from pokerl.data import GameData
from pokerl.env.battle_stream_env import BattleStreamEnv
from pokerl.env.observation import ObservationBuilder
from pokerl.team_pool import load_team_pool, default_team_pool_manifest_path


def main() -> None:
    gd = GameData(gen=9)
    pool = load_team_pool(default_team_pool_manifest_path(pathlib.Path("."), "gen9vgc2026regf"))
    obs_builder = ObservationBuilder(gd)

    env = BattleStreamEnv(
        battle_format="gen9vgc2026regf",
        team_pool=pool,
        obs_builder=obs_builder,
    )

    rng = np.random.default_rng(42)

    t0 = time.perf_counter()
    obs, mask = env.reset()
    print(f"reset() done in {(time.perf_counter()-t0)*1000:.1f}ms")
    print(f"obs keys: {list(obs.keys())}")
    print(f"mask shape: {mask.shape}, legal slot-0 actions: {np.flatnonzero(mask[0]).size}")

    turn = 0
    done = False
    step_times = []

    while not done:
        a0 = int(rng.choice(np.flatnonzero(mask[0])))
        legal1 = np.flatnonzero(mask[a0 + 1])
        a1 = int(rng.choice(legal1)) if legal1.size else a0

        ts = time.perf_counter()
        obs, mask, reward, done, _ = env.step(np.array([a0, a1], dtype=np.int64))
        step_times.append(time.perf_counter() - ts)
        turn += 1

    env.close()

    total = time.perf_counter() - t0
    avg_ms = sum(step_times) / len(step_times) * 1000 if step_times else 0
    print(f"\nBattle finished in {turn} turns, reward={reward:.0f}")
    print(f"Total wall time : {total*1000:.0f}ms")
    print(f"Avg step time   : {avg_ms:.1f}ms  ({1000/avg_ms:.0f} steps/s if single-env)")
    print(f"Steps/s @ 64envs: ~{64000/avg_ms:.0f} (theoretical, ignoring parallelism limits)")


if __name__ == "__main__":
    main()
