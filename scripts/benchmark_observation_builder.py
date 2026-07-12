"""Repeatable CPU microbenchmark for live-style ObservationBuilder states."""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
from poke_env.battle import Field, Pokemon, SideCondition, Weather
from poke_env.teambuilder.teambuilder import Teambuilder

from pokerl.data import GameData
from pokerl.env.observation import (
    ObservationBuilder,
    _SyntheticTeamPreviewBattle,
    _init_team_preview_pokemon,
)
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=10_000)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_benchmark_state(seed: int = 0):
    """Build a stable 6v6 state from two checked-in VGC teams."""
    game_data = GameData(gen=9)
    team_pool = load_team_pool(
        default_team_pool_manifest_path(REPO_ROOT, "gen9vgc2026regf"),
        subset="curated",
        seed=seed,
    )
    own_packed, opponent_packed = team_pool.sample_packed_pair(distinct=True)
    own_team = [
        _init_team_preview_pokemon(
            Pokemon(gen=game_data.gen, teambuilder=entry),
            own_team=True,
        )
        for entry in Teambuilder.parse_packed_team(own_packed)[:6]
    ]
    opponent_team = [
        _init_team_preview_pokemon(
            Pokemon(gen=game_data.gen, teambuilder=entry),
            own_team=False,
        )
        for entry in Teambuilder.parse_packed_team(opponent_packed)[:6]
    ]
    battle = _SyntheticTeamPreviewBattle(own_team, opponent_team)
    battle.active_pokemon = own_team[:2]
    battle.opponent_active_pokemon = opponent_team[:2]
    battle.weather = {Weather.RAINDANCE: 1}
    battle.fields = {Field.TRICK_ROOM: 2}
    battle.side_conditions = {SideCondition.TAILWIND: 2}
    battle.opponent_side_conditions = {SideCondition.STEALTH_ROCK: 1}
    battle.turn = 5
    return ObservationBuilder(game_data, team_size=6), battle


def benchmark(iterations: int, repeats: int, warmup: int, seed: int) -> dict:
    if iterations <= 0 or repeats <= 0 or warmup < 0:
        raise ValueError("iterations/repeats must be positive and warmup non-negative.")
    builder, battle = build_benchmark_state(seed)
    for _ in range(warmup):
        builder.build(battle)

    timings = []
    checksum = 0
    gc_enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(repeats):
            started = time.perf_counter()
            result = None
            for _ in range(iterations):
                result = builder.build(battle)
            timings.append(time.perf_counter() - started)
            assert result is not None
            checksum += int(sum(np.asarray(value).sum() for value in result.values()))
    finally:
        if gc_enabled:
            gc.enable()

    best = min(timings)
    return {
        "iterations": iterations,
        "repeats": repeats,
        "warmup": warmup,
        "seconds": timings,
        "best_us_per_build": best * 1_000_000.0 / iterations,
        "median_us_per_build": float(np.median(timings)) * 1_000_000.0 / iterations,
        "checksum": checksum,
    }


def main() -> None:
    args = parse_args()
    print(
        json.dumps(
            benchmark(args.iterations, args.repeats, args.warmup, args.seed),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
