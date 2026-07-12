from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import pytest
from poke_env.battle import Effect, Field, Pokemon, SideCondition, Status, Weather
from poke_env.data import GenData
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.teambuilder.teambuilder import Teambuilder

from pokerl.data import GameData, UNKNOWN_IDX
from pokerl.env.observation import (
    ABILITY_KNOWN_OFFSET,
    FIELD_OPPONENT_USED_TERA_OFFSET,
    FIELD_TRICK_ROOM_OFFSET,
    FIELD_TURN_OFFSET,
    FIELD_USED_TERA_OFFSET,
    FIRST_TURN_OFFSET,
    ITEM_KNOWN_OFFSET,
    KNOWN_MOVE_COUNT_OFFSET,
    MOVE_PP_OFFSET,
    ObservationBuilder,
    PROTECT_COUNTER_OFFSET,
    REVEALED_OFFSET,
    ROLE_ACTIVE_1,
    ROLE_ACTIVE_2,
    ROLE_BENCH,
    STATUS_COUNTER_OFFSET,
    TERA_KNOWN_OFFSET,
    TERASTALLIZED_OFFSET,
    _init_team_preview_pokemon,
    build_packed_team_aux_labels,
)


def _move(move_id: str, current_pp: int, max_pp: int) -> SimpleNamespace:
    return SimpleNamespace(id=move_id, current_pp=current_pp, max_pp=max_pp)


def test_packed_team_aux_labels_are_exact_private_targets_for_selected_team() -> None:
    game_data = GameData(gen=9)
    packed_team = ConstantTeambuilder(
        """Flutter Mane @ Choice Specs
Ability: Protosynthesis
Tera Type: Fairy
- Dazzling Gleam
- Shadow Ball
- Moonblast
- Protect

Incineroar @ Safety Goggles
Ability: Intimidate
Tera Type: Grass
- Fake Out
- Flare Blitz
- Knock Off
- Parting Shot
"""
    ).yield_team()

    labels = build_packed_team_aux_labels(
        game_data,
        team_size=6,
        packed_team=packed_team,
        selected_indices=[1],
    )

    assert labels["opp_source_index"].tolist() == [0, 1, -1, -1, -1, -1]
    assert labels["opp_mask"].tolist() == [True, False, True, True, True, True]
    assert labels["opp_item"][1] == game_data.item_index("safetygoggles")
    assert labels["opp_ability"][1] == game_data.abilities_to_idx["intimidate"]
    assert labels["opp_types"][1, 2] == game_data.type_to_idx["grass"]
    assert set(labels["opp_moves"][1]) == {
        game_data.move_to_idx[move]
        for move in ("fakeout", "flareblitz", "knockoff", "partingshot")
    }
    assert labels["opp_continuous"][1, ITEM_KNOWN_OFFSET] == 1.0
    assert labels["opp_continuous"][1, ABILITY_KNOWN_OFFSET] == 1.0
    assert labels["opp_continuous"][1, TERA_KNOWN_OFFSET] == 1.0


def test_packed_team_aux_labels_mask_exactly_the_unselected_two() -> None:
    game_data = GameData(gen=9)
    packed_team = ConstantTeambuilder(
        "\n\n".join(
            """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
- Thunderbolt
- Protect

""".strip()
            for _ in range(6)
        )
    ).yield_team()

    labels = build_packed_team_aux_labels(
        game_data,
        team_size=6,
        packed_team=packed_team,
        selected_indices=[0, 2, 4, 5],
    )

    assert labels["opp_source_index"].tolist() == [0, 1, 2, 3, 4, 5]
    assert labels["opp_mask"].tolist() == [False, True, False, True, False, False]


def test_packed_team_aux_labels_reject_invalid_selected_positions() -> None:
    game_data = GameData(gen=9)
    packed_team = ConstantTeambuilder(
        """Pikachu @ Light Ball
Ability: Static
Tera Type: Electric
- Thunderbolt
"""
    ).yield_team()

    with pytest.raises(ValueError, match="unique packed-team positions"):
        build_packed_team_aux_labels(
            game_data,
            team_size=6,
            packed_team=packed_team,
            selected_indices=[0, 0],
        )


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
    last_move: SimpleNamespace | None = None,
    protect_counter: int = 0,
    status_counter: int = 0,
    first_turn: bool = False,
    revealed: bool = True,
    terastallized: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        base_species=species,
        moves=moves,
        last_move=last_move,
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
        protect_counter=protect_counter,
        status_counter=status_counter,
        first_turn=first_turn,
        revealed=revealed,
        is_terastallized=terastallized,
    )


def _opponent_pokemon(
    species: str,
    *,
    revealed: bool,
    item: str | None = None,
) -> SimpleNamespace:
    return _pokemon(
        species=species,
        moves=OrderedDict(),
        ability=None,
        item=item if revealed else GenData.UNKNOWN_ITEM,
        type_1="normal",
        type_2=None,
        tera_type=None,
        current_hp=100 if revealed else 0,
        max_hp=100 if revealed else 0,
        revealed=revealed,
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
        last_move=_move("partingshot", 18, 32),
        protect_counter=2,
        status_counter=3,
        first_turn=True,
        terastallized=True,
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
        last_move=_move("spore", 12, 24),
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
        last_move=_move("electroshot", 8, 16),
        revealed=True,
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
        turn=4,
        used_tera=True,
        opponent_used_tera=False,
    )

    obs = builder.build(battle)

    assert obs["own_active_indices"].tolist() == [0, 1]
    assert obs["own_active_present"].tolist() == [True, True]
    assert obs["opp_active_indices"].tolist() == [0, 0]
    assert obs["opp_active_present"].tolist() == [True, False]
    assert obs["opp_source_index"].tolist() == [0, 1, -1, -1, -1, -1]
    assert obs["own_roles"].tolist() == [
        ROLE_ACTIVE_1,
        ROLE_ACTIVE_2,
        ROLE_BENCH,
        ROLE_BENCH,
        ROLE_BENCH,
        ROLE_BENCH,
    ]
    assert obs["opp_roles"].tolist() == [
        ROLE_ACTIVE_1,
        ROLE_BENCH,
        ROLE_BENCH,
        ROLE_BENCH,
        ROLE_BENCH,
        ROLE_BENCH,
    ]
    assert obs["own_mask"].tolist() == [False, False, False, True, True, True]
    assert obs["opp_mask"].tolist() == [False, False, True, True, True, True]

    assert obs["own_species"][0] == gd.species_to_idx["incineroar"]
    assert obs["own_species"][2] == gd.species_to_idx["pelipper"]
    assert obs["own_moves"][0, 0] == gd.move_to_idx["fakeout"]
    assert obs["own_moves"][0, 1] == gd.move_to_idx["partingshot"]
    assert obs["own_last_move"][0] == gd.move_to_idx["partingshot"]
    assert obs["opp_last_move"][0] == gd.move_to_idx["electroshot"]
    assert obs["opp_moves"][0, 0] == gd.move_to_idx["electroshot"]
    assert np.all(obs["opp_continuous"][0, MOVE_PP_OFFSET : MOVE_PP_OFFSET + 4] == 0.0)

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
    assert own_cont[MOVE_PP_OFFSET] == pytest.approx(1.0)
    assert own_cont[MOVE_PP_OFFSET + 1] == pytest.approx(18 / 32)
    assert own_cont[PROTECT_COUNTER_OFFSET] == pytest.approx(0.5)
    assert own_cont[STATUS_COUNTER_OFFSET] == pytest.approx(3 / 8)
    assert own_cont[FIRST_TURN_OFFSET] == 1.0
    assert own_cont[REVEALED_OFFSET] == 1.0
    assert own_cont[KNOWN_MOVE_COUNT_OFFSET] == pytest.approx(0.5)
    assert own_cont[ABILITY_KNOWN_OFFSET] == 1.0
    assert own_cont[ITEM_KNOWN_OFFSET] == 1.0
    assert own_cont[TERA_KNOWN_OFFSET] == 1.0
    assert own_cont[TERASTALLIZED_OFFSET] == 1.0

    field = obs["field"]
    assert field[0] == 0.0  # weather is active
    assert field[3] == 1.0  # rain slot
    assert field[9] == pytest.approx(3 / 8)  # weather age
    assert field[23] == pytest.approx(3 / 4)  # own tailwind age
    assert field[FIELD_TRICK_ROOM_OFFSET] == pytest.approx(3 / 5)
    assert field[FIELD_USED_TERA_OFFSET] == 1.0
    assert field[FIELD_OPPONENT_USED_TERA_OFFSET] == 0.0
    assert field[FIELD_TURN_OFFSET] == pytest.approx(4 / 20)


def test_opponent_source_indices_remain_stable_when_actives_switch() -> None:
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)
    opponents = [
        _pokemon(
            species=species,
            moves=OrderedDict(),
            ability=None,
            item=None,
            type_1="normal",
            type_2=None,
            tera_type=None,
            current_hp=100,
            max_hp=100,
        )
        for species in ("incineroar", "pelipper", "archaludon")
    ]
    battle = SimpleNamespace(
        team=OrderedDict(),
        opponent_team=OrderedDict(
            (f"p2:{index}", pokemon) for index, pokemon in enumerate(opponents)
        ),
        active_pokemon=[None, None],
        opponent_active_pokemon=[opponents[0], opponents[1]],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=1,
        used_tera=False,
        opponent_used_tera=False,
    )

    before_switch = builder.build(battle)
    battle.opponent_active_pokemon = [opponents[2], opponents[1]]
    after_switch = builder.build(battle)

    assert before_switch["opp_source_index"].tolist() == [0, 1, 2, -1, -1, -1]
    assert after_switch["opp_source_index"].tolist() == [1, 2, 0, -1, -1, -1]
    expected_identity_by_species = {
        gd.species_to_idx[pokemon.species]: index
        for index, pokemon in enumerate(opponents)
    }
    for obs in (before_switch, after_switch):
        identity_by_species = {
            int(species): int(source_index)
            for species, source_index, padded in zip(
                obs["opp_species"],
                obs["opp_source_index"],
                obs["opp_mask"],
                strict=True,
            )
            if not padded
        }
        assert identity_by_species == expected_identity_by_species


def test_public_preview_roster_persists_as_live_opponents_are_revealed() -> None:
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)
    species = (
        "incineroar",
        "pelipper",
        "amoonguss",
        "archaludon",
        "urshifu",
        "tornadus",
    )
    preview = [
        _opponent_pokemon(species_name, revealed=False) for species_name in species
    ]
    live = {
        index: _opponent_pokemon(species[index], revealed=True) for index in (0, 1, 3)
    }
    battle = SimpleNamespace(
        team=OrderedDict(),
        opponent_team=OrderedDict((f"p2:{index}", live[index]) for index in (0, 1)),
        teampreview_opponent_team=preview,
        active_pokemon=[None, None],
        opponent_active_pokemon=[live[0], live[1]],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=1,
        used_tera=False,
        opponent_used_tera=False,
    )

    leads_only = builder.build(battle)

    assert leads_only["opp_source_index"].tolist() == [0, 1, 2, 3, 4, 5]
    assert not leads_only["opp_mask"].any()
    assert set(leads_only["opp_species"].tolist()) == {
        gd.species_to_idx[name] for name in species
    }
    revealed_by_source = {
        int(source): float(row[REVEALED_OFFSET])
        for source, row in zip(
            leads_only["opp_source_index"],
            leads_only["opp_continuous"],
            strict=True,
        )
    }
    assert revealed_by_source == {0: 1.0, 1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

    battle.opponent_team["p2:3"] = live[3]
    battle.opponent_active_pokemon = [live[3], live[1]]
    after_switch = builder.build(battle)

    assert after_switch["opp_source_index"].tolist() == [1, 3, 0, 2, 4, 5]
    assert not after_switch["opp_mask"].any()
    assert set(after_switch["opp_species"].tolist()) == {
        gd.species_to_idx[name] for name in species
    }
    revealed_by_source = {
        int(source): float(row[REVEALED_OFFSET])
        for source, row in zip(
            after_switch["opp_source_index"],
            after_switch["opp_continuous"],
            strict=True,
        )
    }
    assert revealed_by_source == {0: 1.0, 1: 1.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}


def test_illusion_roster_keeps_display_but_withholds_ambiguous_source() -> None:
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)
    species = ("gengar", "zoroark", "amoonguss", "pelipper", "urshifu", "tornadus")
    preview = [
        _opponent_pokemon(species_name, revealed=False) for species_name in species
    ]
    apparent_gengar = _opponent_pokemon("gengar", revealed=True)
    battle = SimpleNamespace(
        team=OrderedDict(),
        opponent_team=OrderedDict([("p2:gengar", apparent_gengar)]),
        teampreview_opponent_team=preview,
        active_pokemon=[None, None],
        opponent_active_pokemon=[apparent_gengar, None],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=1,
        used_tera=False,
        opponent_used_tera=False,
    )

    ambiguous = builder.build(battle)

    assert ambiguous["opp_species"][0] == gd.species_to_idx["gengar"]
    assert ambiguous["opp_source_index"][0] == -1

    revealed_zoroark = _opponent_pokemon("zoroark", revealed=True)
    battle.opponent_team["p2:zoroark"] = revealed_zoroark
    battle.opponent_active_pokemon = [revealed_zoroark, None]
    confirmed = builder.build(battle)

    assert confirmed["opp_species"][0] == gd.species_to_idx["zoroark"]
    assert confirmed["opp_source_index"][0] == 1
    source_by_species = {
        int(species_index): int(source)
        for species_index, source in zip(
            confirmed["opp_species"],
            confirmed["opp_source_index"],
            strict=True,
        )
    }
    assert source_by_species[gd.species_to_idx["gengar"]] == 0


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


def test_observation_builder_keeps_unmatched_visible_opponent_active() -> None:
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)

    visible_active = _pokemon(
        species="gengar",
        moves=OrderedDict([("shadowball", _move("shadowball", 24, 24))]),
        ability=None,
        item=None,
        type_1="ghost",
        type_2="poison",
        tera_type="ghost",
        current_hp=100,
        max_hp=100,
        revealed=True,
    )
    tracked_opponent = _pokemon(
        species="zoroark",
        moves=OrderedDict(),
        ability=None,
        item=None,
        type_1="dark",
        type_2=None,
        tera_type="poison",
        current_hp=100,
        max_hp=100,
    )

    battle = SimpleNamespace(
        team=OrderedDict(),
        opponent_team=OrderedDict([("p2:1", tracked_opponent)]),
        active_pokemon=[None, None],
        opponent_active_pokemon=[visible_active, None],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=1,
        used_tera=False,
        opponent_used_tera=False,
    )

    obs = builder.build(battle)

    assert obs["opp_active_indices"].tolist() == [0, 0]
    assert obs["opp_active_present"].tolist() == [True, False]
    assert obs["opp_roles"].tolist()[:2] == [ROLE_ACTIVE_1, ROLE_BENCH]
    assert obs["opp_species"][0] == gd.species_to_idx["gengar"]
    assert obs["opp_species"][1] == gd.species_to_idx["zoroark"]
    assert obs["opp_source_index"].tolist() == [-1, 0, -1, -1, -1, -1]
    assert obs["opp_moves"][0, 0] == gd.move_to_idx["shadowball"]


def test_team_preview_pokemon_init_keeps_preview_revealed_flag_unset() -> None:
    gd = GameData(gen=9)
    packed_team = ConstantTeambuilder(
        """Incineroar @ Safety Goggles
Ability: Intimidate
Tera Type: Grass
- Fake Out
"""
    ).yield_team()
    teambuilder_pokemon = Teambuilder.parse_packed_team(packed_team)[0]

    own_preview = _init_team_preview_pokemon(
        Pokemon(gen=gd.gen, teambuilder=teambuilder_pokemon),
        own_team=True,
    )
    opp_preview = _init_team_preview_pokemon(
        Pokemon(gen=gd.gen, species="amoonguss"),
        own_team=False,
    )

    assert own_preview.revealed is False
    assert own_preview.tera_type.name.lower() == "grass"
    assert own_preview.is_terastallized is False
    assert opp_preview.revealed is False


def test_unrevealed_item_sentinel_is_encoded_as_unknown() -> None:
    gd = GameData(gen=9)
    builder = ObservationBuilder(gd, team_size=6)
    opponent = Pokemon(gen=gd.gen, species="amoonguss")
    battle = SimpleNamespace(
        team=OrderedDict(),
        opponent_team=OrderedDict([("amoonguss", opponent)]),
        active_pokemon=[None, None],
        opponent_active_pokemon=[None, None],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=0,
        used_tera=False,
        opponent_used_tera=False,
    )

    obs = builder.build(battle)

    assert opponent.item == GenData.UNKNOWN_ITEM
    assert obs["opp_item"][0] == UNKNOWN_IDX
    assert obs["opp_continuous"][0, ITEM_KNOWN_OFFSET] == 0.0
