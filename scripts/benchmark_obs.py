"""Micro-benchmark for observation building hot path.

Usage:
    python scripts/benchmark_obs.py [--iters N]

Builds synthetic battle states that are realistic in complexity (2 active
pokemon with effects/boosts, 4 bench pokemon) and times ObservationBuilder.build()
across many iterations.  Run on the branch before and after the optimisation
commit to compare throughput numbers.
"""

import argparse
import time
from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from poke_env.battle import Effect, Field, SideCondition, Status, Weather

from pokerl.data import GameData
from pokerl.env.observation import ObservationBuilder


def _move(move_id: str, current_pp: int = 16, max_pp: int = 16) -> SimpleNamespace:
    return SimpleNamespace(id=move_id, current_pp=current_pp, max_pp=max_pp)


def _pokemon(
    *,
    species: str,
    moves: dict | None = None,
    ability: str | None = None,
    item: str | None = None,
    type_1: str = "normal",
    type_2: str | None = None,
    tera_type: str | None = None,
    current_hp: int = 100,
    max_hp: int = 100,
    status=None,
    boosts: dict | None = None,
    effects: dict | None = None,
    protect_counter: int = 0,
    status_counter: int = 0,
    first_turn: bool = False,
    revealed: bool = True,
    terastallized: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        moves=moves or {},
        last_move=None,
        ability=ability,
        item=item,
        type_1=SimpleNamespace(name=type_1.upper()),
        type_2=SimpleNamespace(name=type_2.upper()) if type_2 else None,
        tera_type=SimpleNamespace(name=tera_type.upper()) if tera_type else None,
        current_hp=current_hp,
        max_hp=max_hp,
        current_hp_fraction=current_hp / max_hp if max_hp else None,
        status=status,
        boosts=boosts or {},
        effects=effects or {},
        protect_counter=protect_counter,
        status_counter=status_counter,
        first_turn=first_turn,
        revealed=revealed,
        is_terastallized=terastallized,
    )


def make_battle() -> SimpleNamespace:
    """Realistic doubles battle with effects, boosts, and bench pokemon."""
    own_active_1 = _pokemon(
        species="incineroar",
        moves=OrderedDict([
            ("fakeout", _move("fakeout", 16, 16)),
            ("partingshot", _move("partingshot", 18, 32)),
            ("flareblitz", _move("flareblitz", 12, 16)),
            ("knockoff", _move("knockoff", 20, 32)),
        ]),
        ability="intimidate",
        item="sitrusberry",
        type_1="fire", type_2="dark", tera_type="grass",
        current_hp=150, max_hp=200,
        status=Status.BRN,
        boosts={"atk": -1, "spa": 2, "spe": 1},
        effects={Effect.CONFUSION: 1, Effect.PROTECT: 1},
        protect_counter=1, status_counter=3,
    )
    own_active_2 = _pokemon(
        species="amoonguss",
        moves=OrderedDict([
            ("spore", _move("spore", 12, 24)),
            ("ragepowder", _move("ragepowder", 15, 24)),
            ("pollenpuff", _move("pollenpuff", 15, 24)),
            ("clearsmog", _move("clearsmog", 30, 40)),
        ]),
        ability="regenerator", item="rockyhelmet",
        type_1="grass", type_2="poison", tera_type="water",
        current_hp=190, max_hp=220,
        effects={Effect.YAWN: 1},
    )
    bench = [
        _pokemon(species="pelipper", ability="drizzle", item="focussash",
                 type_1="water", type_2="flying", current_hp=100, max_hp=160),
        _pokemon(species="urshifu", ability="unseenfist", item="choicescarf",
                 type_1="fighting", type_2="dark", current_hp=180, max_hp=180),
        _pokemon(species="rillaboom", ability="grassy surge", item="assaultvest",
                 type_1="grass", current_hp=200, max_hp=200),
        _pokemon(species="grimmsnarl", ability="prankster", item="lightclay",
                 type_1="dark", type_2="fairy", current_hp=170, max_hp=190),
    ]

    opp_active_1 = _pokemon(
        species="flutter mane",
        moves=OrderedDict([("moonblast", _move("moonblast")), ("shadowball", _move("shadowball"))]),
        type_1="ghost", type_2="fairy",
        current_hp=100, max_hp=100,
        boosts={"spa": 2},
        effects={Effect.SUBSTITUTE: 1},
    )
    opp_active_2 = _pokemon(
        species="landorus-therian",
        moves=OrderedDict([("earthquake", _move("earthquake"))]),
        type_1="ground", type_2="flying",
        current_hp=80, max_hp=100,
    )
    opp_bench = [
        _pokemon(species="tornadus-therian", type_1="flying", current_hp=100, max_hp=100),
        _pokemon(species="archaludon", type_1="steel", type_2="dragon", current_hp=100, max_hp=100),
    ]

    own_team = OrderedDict(
        {f"p1:{i}": p for i, p in enumerate([own_active_1, own_active_2] + bench, 1)}
    )
    opp_team = OrderedDict(
        {f"p2:{i}": p for i, p in enumerate([opp_active_1, opp_active_2] + opp_bench, 1)}
    )

    return SimpleNamespace(
        team=own_team,
        opponent_team=opp_team,
        active_pokemon=[own_active_1, own_active_2],
        opponent_active_pokemon=[opp_active_1, opp_active_2],
        weather={Weather.RAINDANCE: 1},
        fields={Field.TRICK_ROOM: 1},
        side_conditions={SideCondition.TAILWIND: 1},
        opponent_side_conditions={SideCondition.STEALTH_ROCK: 1},
        turn=8,
        used_tera=False,
        opponent_used_tera=True,
    )


def run_benchmark(iters: int) -> None:
    print(f"Loading GameData ...", flush=True)
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)
    battle = make_battle()

    # Warmup
    for _ in range(200):
        builder.build(battle)

    start = time.perf_counter()
    for _ in range(iters):
        builder.build(battle)
    elapsed = time.perf_counter() - start

    per_call_us = elapsed / iters * 1e6
    calls_per_sec = iters / elapsed
    print(f"ObservationBuilder.build()  iters={iters:,}")
    print(f"  total   : {elapsed*1000:.1f} ms")
    print(f"  per call: {per_call_us:.1f} µs")
    print(f"  rate    : {calls_per_sec:,.0f} calls/s")
    print(f"  @ 8 envs: {calls_per_sec/8:,.0f} steps/s (obs only)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=50_000)
    args = parser.parse_args()
    run_benchmark(args.iters)
