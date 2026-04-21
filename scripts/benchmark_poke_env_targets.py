"""Profile get_possible_showdown_targets() and Move() construction cost.

Patches both functions during a live battle to measure their per-call cost
using the same approach as benchmark_actions.py.

Usage:
    python scripts/benchmark_poke_env_targets.py [--server localhost:8000] [--repeat N]

--repeat N : extra timing iterations per live call (default: 500)
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from poke_env import ServerConfiguration
from poke_env.battle.move import Move

import pokerl.env.actions as actions_module
import pokerl.env.showdown_env as showdown_env_module
from pokerl.config import EnvConfig
from pokerl.data import GameData
from pokerl.env.showdown_env import BattleEnv, make_local_account_configuration
from poke_env.player import RandomPlayer

import poke_env.battle.double_battle as double_battle_module


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

    # --- Patch get_possible_showdown_targets ---
    original_gpst = double_battle_module.DoubleBattle.get_possible_showdown_targets
    gpst_times: list[float] = []

    def timed_gpst(self, move, pokemon, dynamax=False):
        t0 = time.perf_counter()
        for _ in range(repeat):
            original_gpst(self, move, pokemon, dynamax)
        gpst_times.append((time.perf_counter() - t0) / repeat)
        return original_gpst(self, move, pokemon, dynamax)

    double_battle_module.DoubleBattle.get_possible_showdown_targets = timed_gpst

    # --- Patch Move.__init__ ---
    original_move_init = Move.__init__
    move_init_times: list[float] = []

    def timed_move_init(self, move_id: str, gen: int, raw_id=None, from_mimic=False, from_transform=False):
        t0 = time.perf_counter()
        for _ in range(repeat):
            original_move_init(self, move_id, gen, raw_id, from_mimic, from_transform)
        move_init_times.append((time.perf_counter() - t0) / repeat)
        original_move_init(self, move_id, gen, raw_id, from_mimic, from_transform)

    Move.__init__ = timed_move_init

    def opponent_factory(loop):
        return RandomPlayer(
            account_configuration=make_local_account_configuration("benchopp2"),
            battle_format=env_config.battle_format,
            server_configuration=srv_cfg,
            loop=loop,
        )

    env = BattleEnv(
        env_config,
        gd,
        opponent_factory,
        player_kwargs={"account_configuration": make_local_account_configuration("benchagent2")},
    )

    rng = np.random.default_rng(42)
    print(f"Running battle vs random opponent on {server_url} ...", flush=True)
    try:
        obs, masks = env.reset()
        done = False
        while not done:
            a0 = int(rng.choice(np.flatnonzero(masks[0])))
            a1_legal = np.flatnonzero(masks[a0 + 1])
            a1 = int(rng.choice(a1_legal)) if a1_legal.size else a0
            obs, masks, _reward, done, _info = env.step(np.array([a0, a1], dtype=np.int64))
    finally:
        env.close()
        double_battle_module.DoubleBattle.get_possible_showdown_targets = original_gpst
        Move.__init__ = original_move_init

    def _report(label: str, times: list[float], calls_per_step: int) -> None:
        if not times:
            print(f"\n{label}: no calls captured")
            return
        n = len(times)
        avg_us = sum(times) / n * 1e6
        min_us = min(times) * 1e6
        max_us = max(times) * 1e6
        print(f"\n--- {label}  ({n} calls × {repeat} iters each = {n*repeat:,} total) ---")
        print(f"  avg : {avg_us:.1f} µs/call")
        print(f"  min : {min_us:.1f} µs")
        print(f"  max : {max_us:.1f} µs")
        print(f"  rate: {1e6/avg_us:,.0f} calls/s")
        print(f"  @ {calls_per_step} calls/step, 8 envs: ~{avg_us * calls_per_step:.0f} µs/step overhead")

    # get_possible_showdown_targets is called once per available move per slot per step.
    # In a typical doubles step that's ~4 moves * 2 slots = ~8 calls/step.
    _report("get_possible_showdown_targets()", gpst_times, calls_per_step=8)

    # Move.__init__ is called once per available move per slot (for cloning in _request_aware_move_targets).
    # Same ~8 calls/step.
    _report("Move.__init__() [for request_target cloning]", move_init_times, calls_per_step=8)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="localhost:8000")
    parser.add_argument("--repeat", type=int, default=500)
    args = parser.parse_args()
    run_benchmark(args.server, args.repeat)
