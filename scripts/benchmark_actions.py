"""Micro-benchmark for action masking hot path using a live Showdown battle.

Wraps build_legal_action_state() so that every real call during a battle
is timed N extra times immediately on the live battle state (before any state
mutation), then reports per-call statistics.

Usage:
    python scripts/benchmark_actions.py [--server localhost:8000] [--repeat N]

--repeat N  : extra iterations per live state for a stable per-state estimate
              (default: 200).  The real call still happens once normally.
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from poke_env import ServerConfiguration

import pokerl.env.actions as actions_module
import pokerl.env.showdown_env as showdown_env_module
from pokerl.config import EnvConfig
from pokerl.data import GameData
from pokerl.env.showdown_env import BattleEnv, make_local_account_configuration
from poke_env.player import RandomPlayer


def _server_config(server_url: str) -> ServerConfiguration:
    host, port = server_url.split(":")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php",
    )


def run_benchmark(server_url: str, repeat: int) -> None:
    print("Loading GameData ...", flush=True)
    gd = GameData(gen=9)
    env_config = EnvConfig(num_envs=1, server_url=server_url)
    srv_cfg = _server_config(server_url)

    original_blas = actions_module.build_legal_action_state

    # Per-state timing: time `repeat` extra calls immediately while the battle
    # state is live (moves/switches fully populated), then do one real call.
    per_state_times: list[float] = []

    def timed_blas(battle):
        t0 = time.perf_counter()
        for _ in range(repeat):
            original_blas(battle)
        per_state_times.append((time.perf_counter() - t0) / repeat)
        return original_blas(battle)

    actions_module.build_legal_action_state = timed_blas
    showdown_env_module.build_legal_action_state = timed_blas

    def opponent_factory(loop):
        return RandomPlayer(
            account_configuration=make_local_account_configuration("benchopp"),
            battle_format=env_config.battle_format,
            server_configuration=srv_cfg,
            loop=loop,
        )

    env = BattleEnv(
        env_config,
        gd,
        opponent_factory,
        player_kwargs={"account_configuration": make_local_account_configuration("benchagent")},
    )

    rng = np.random.default_rng(42)
    print(f"Running battle vs random opponent on {server_url} ...", flush=True)
    try:
        obs, masks = env.reset()
        done = False
        while not done:
            # masks is the joint mask: row 0 = legal slot-0 actions,
            # row a0+1 = legal slot-1 actions conditioned on slot-0 choosing a0.
            a0 = int(rng.choice(np.flatnonzero(masks[0])))
            a1_legal = np.flatnonzero(masks[a0 + 1])
            a1 = int(rng.choice(a1_legal)) if a1_legal.size else a0
            obs, masks, _reward, done, _info = env.step(np.array([a0, a1], dtype=np.int64))
    finally:
        env.close()
        actions_module.build_legal_action_state = original_blas
        showdown_env_module.build_legal_action_state = original_blas

    if not per_state_times:
        print("No action-masking calls were captured — did the battle start?")
        return

    n = len(per_state_times)
    avg_us = sum(per_state_times) / n * 1e6
    min_us = min(per_state_times) * 1e6
    max_us = max(per_state_times) * 1e6

    print(f"\n--- build_legal_action_state  ({n} states × {repeat} iters each = {n*repeat:,} total calls) ---")
    print(f"  avg : {avg_us:.1f} µs/call")
    print(f"  min : {min_us:.1f} µs")
    print(f"  max : {max_us:.1f} µs")
    print(f"  rate: {1e6/avg_us:,.0f} calls/s")
    print(f"  @ 8 envs: {1e6/avg_us/8:,.0f} steps/s (action mask only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="localhost:8000")
    parser.add_argument("--repeat", type=int, default=200)
    args = parser.parse_args()
    run_benchmark(args.server, args.repeat)
