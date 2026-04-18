from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.battle import Effect, Field, SideCondition, Status, Weather

from pokerl.data import GameData
from pokerl.env.observation import ObservationBuilder, ROLE_ACTIVE_1, ROLE_ACTIVE_2, ROLE_BENCH


def _move(move_id: str, current_pp: int, max_pp: int) -> SimpleNamespace:
    return SimpleNamespace(id=move_id, current_pp=current_pp, max_pp=max_pp)


def _pokemon(
    *,
    species: str,
    moves: OrderedDict[str, SimpleNamespace] | None,
    ability: str | None,
    item: str | None,
    type_1: str | None,
    type_2: str | None,
    tera_type: str | None,
    current_hp: int,
    max_hp: int,
    status=None,
    boosts: dict | None = None,
    effects: dict | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        moves=moves,
        ability=ability,
        item=item,
        type_1=SimpleNamespace(name=type_1.upper()) if type_1 else None,
        type_2=SimpleNamespace(name=type_2.upper()) if type_2 else None,
        tera_type=SimpleNamespace(name=tera_type.upper()) if tera_type else None,
        current_hp=current_hp,
        max_hp=max_hp,
        current_hp_fraction=current_hp / max_hp if max_hp else None,
        status=status,
        boosts=boosts or {},
        effects=effects or {},
    )


def test_observation_builder_reorders_actives_and_preserves_feature_values() -> None:
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)

    own_active_1 = _pokemon(
        species="incineroar",
        moves=OrderedDict(
            [
                ("fakeout", _move("fakeout", 16, 16)),
                ("partingshot", _move("partingshot", 18, 32)),
            ]
        ),
        ability="intimidate",
        item="sitrusberry",
        type_1="fire",
        type_2="dark",
        tera_type="grass",
        current_hp=150,
        max_hp=200,
        status=Status.BRN,
        boosts={"atk": -1, "spa": 2},
        effects={Effect.CONFUSION: 1, Effect.PROTECT: 1},
    )
    own_active_2 = _pokemon(
        species="amoonguss",
        moves=OrderedDict([("spore", _move("spore", 12, 24))]),
        ability="regenerator",
        item="rockyhelmet",
        type_1="grass",
        type_2="poison",
        tera_type="water",
        current_hp=190,
        max_hp=220,
        status=None,
        effects={Effect.YAWN: 1},
    )
    own_bench = _pokemon(
        species="pelipper",
        moves=OrderedDict([("hurricane", _move("hurricane", 8, 16))]),
        ability="drizzle",
        item="focussash",
        type_1="water",
        type_2="flying",
        tera_type="ghost",
        current_hp=100,
        max_hp=160,
    )
    opp_active = _pokemon(
        species="archaludon",
        moves=OrderedDict([("electroshot", _move("electroshot", 8, 16))]),
        ability=None,
        item=None,
        type_1="steel",
        type_2="dragon",
        tera_type=None,
        current_hp=100,
        max_hp=100,
    )
    opp_bench = _pokemon(
        species="urshifu",
        moves=None,
        ability=None,
        item=None,
        type_1="fighting",
        type_2="dark",
        tera_type=None,
        current_hp=100,
        max_hp=100,
    )

    battle = SimpleNamespace(
        team=OrderedDict(
            [
                ("p1:3", own_bench),
                ("p1:1", own_active_1),
                ("p1:2", own_active_2),
            ]
        ),
        opponent_team=OrderedDict([("p2:1", opp_active), ("p2:2", opp_bench)]),
        active_pokemon=[own_active_1, own_active_2],
        opponent_active_pokemon=[opp_active, None],
        weather={Weather.RAINDANCE: 1},
        fields={Field.TRICK_ROOM: 1},
        side_conditions={SideCondition.TAILWIND: 1},
        opponent_side_conditions={SideCondition.STEALTH_ROCK: 1},
    )

    obs = builder.build(battle)

    assert obs["own_active_indices"].tolist() == [0, 1]
    assert obs["own_roles"].tolist() == [ROLE_ACTIVE_1, ROLE_ACTIVE_2, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH]
    assert obs["opp_roles"].tolist() == [ROLE_ACTIVE_1, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH]
    assert obs["own_mask"].tolist() == [False, False, False, True, True, True]
    assert obs["opp_mask"].tolist() == [False, False, True, True, True, True]

    assert obs["own_species"][0] == gd.species_to_idx["incineroar"]
    assert obs["own_species"][2] == gd.species_to_idx["pelipper"]
    assert obs["own_moves"][0, 0] == gd.move_to_idx["fakeout"]
    assert obs["own_moves"][0, 1] == gd.move_to_idx["partingshot"]
    assert obs["opp_moves"][0, 0] == gd.move_to_idx["electroshot"]
    assert np.all(obs["opp_continuous"][0, -4:] == 0.0)

    assert obs["own_ability"][0] == gd.abilities_to_idx["intimidate"]
    assert obs["own_item"][0] == gd.item_index("sitrusberry")
    assert obs["own_types"][0].tolist() == [
        gd.type_to_idx["fire"],
        gd.type_to_idx["dark"],
        gd.type_to_idx["grass"],
    ]

    own_cont = obs["own_continuous"][0]
    assert own_cont[0] == pytest.approx(0.75)
    assert own_cont[1 + 1] == 1.0  # burned
    assert own_cont[8] == pytest.approx(-1 / 6)
    assert own_cont[10] == pytest.approx(2 / 6)
    assert own_cont[-4] == pytest.approx(1.0)
    assert own_cont[-3] == pytest.approx(18 / 32)

    field = obs["field"]
    assert field[0] == 0.0  # weather is active
    assert field[3] == 1.0  # rain slot
    assert field[-1] == 1.0  # trick room


def test_observation_builder_raises_when_active_pokemon_is_missing_from_team() -> None:
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)

    missing_active = _pokemon(
        species="incineroar",
        moves=OrderedDict(),
        ability="intimidate",
        item="sitrusberry",
        type_1="fire",
        type_2="dark",
        tera_type="grass",
        current_hp=100,
        max_hp=100,
    )
    bench = _pokemon(
        species="pelipper",
        moves=OrderedDict(),
        ability="drizzle",
        item="focussash",
        type_1="water",
        type_2="flying",
        tera_type="ghost",
        current_hp=100,
        max_hp=100,
    )

    battle = SimpleNamespace(
        team=OrderedDict([("p1:2", bench)]),
        opponent_team=OrderedDict(),
        active_pokemon=[missing_active, None],
        opponent_active_pokemon=[None, None],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
    )

    with pytest.raises(ValueError, match="not found in the ordered team list"):
        builder.build(battle)
