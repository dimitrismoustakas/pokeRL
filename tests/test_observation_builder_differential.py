"""Differential coverage for the optimized observation hot path."""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
from poke_env.battle import Effect, Field, SideCondition, Status, Weather

from pokerl.data import GameData, UNKNOWN_IDX
from pokerl.env import observation as observation_module
from pokerl.env.observation import ObservationBuilder


class _ReferenceObservationBuilder(ObservationBuilder):
    """The pre-optimization team encoder, retained only as a test oracle."""

    def _encode_team(
        self,
        pokemon_list: list,
        roles: np.ndarray,
        obs: dict,
        prefix: str,
        full_info: bool,
    ) -> None:
        team_size = self.team_size
        species = np.zeros(team_size, dtype=np.int64)
        moves = np.zeros((team_size, 4), dtype=np.int64)
        last_move = np.zeros(team_size, dtype=np.int64)
        ability = np.zeros(team_size, dtype=np.int64)
        item = np.zeros(team_size, dtype=np.int64)
        types = np.zeros((team_size, 3), dtype=np.int64)
        continuous = np.zeros(
            (team_size, observation_module.CONTINUOUS_DIM),
            dtype=np.float32,
        )
        mask = np.ones(team_size, dtype=bool)

        for index, pokemon in enumerate(pokemon_list):
            if pokemon is None or index >= team_size:
                break
            mask[index] = False
            species[index] = self._species_idx(pokemon)
            ability[index] = self._reference_ability_idx(pokemon)
            item[index] = self._reference_item_idx(pokemon)
            last_move[index] = self._move_idx(
                pokemon.last_move.id if pokemon.last_move is not None else None
            )
            self._fill_type_indices(types[index], pokemon)
            self._reference_fill_continuous_features(continuous[index], pokemon)
            if prefix == "own":
                stats = [pokemon.max_hp] + [
                    pokemon.stats[key] for key in ("atk", "def", "spa", "spd", "spe")
                ]
                continuous[index, observation_module.OWN_STATS_OFFSET :] = (
                    np.array(stats) / 256
                )
            self._reference_fill_move_features(
                moves[index],
                continuous[
                    index,
                    observation_module.MOVE_PP_OFFSET : observation_module.MOVE_PP_OFFSET
                    + observation_module.NUM_MOVE_SLOTS,
                ],
                pokemon,
                full_info,
            )

        obs[f"{prefix}_species"] = species
        obs[f"{prefix}_moves"] = moves
        obs[f"{prefix}_last_move"] = last_move
        obs[f"{prefix}_ability"] = ability
        obs[f"{prefix}_item"] = item
        obs[f"{prefix}_types"] = types
        obs[f"{prefix}_continuous"] = continuous
        obs[f"{prefix}_roles"] = roles
        obs[f"{prefix}_mask"] = mask

    def _reference_ability_idx(self, pokemon: SimpleNamespace) -> int:
        ability = pokemon.ability
        if ability is None:
            return UNKNOWN_IDX
        return self.gd.abilities_to_idx.get(ability.lower(), UNKNOWN_IDX)

    def _reference_item_idx(self, pokemon: SimpleNamespace) -> int:
        item = pokemon.item
        if item is None or item == "":
            return UNKNOWN_IDX
        return self.gd.item_index(item)

    def _reference_fill_move_features(
        self,
        move_indices: np.ndarray,
        pp_features: np.ndarray,
        pokemon: SimpleNamespace,
        full_info: bool,
    ) -> None:
        if not pokemon.moves:
            return
        for move_index, move in enumerate(pokemon.moves.values()):
            if move_index >= observation_module.NUM_MOVE_SLOTS:
                break
            move_indices[move_index] = self._move_idx(move.id)
            if full_info and move.max_pp and move.max_pp > 0:
                pp_features[move_index] = move.current_pp / move.max_pp

    def _reference_fill_continuous_features(
        self,
        feats: np.ndarray,
        pokemon: SimpleNamespace,
    ) -> None:
        offset = 0
        if pokemon.max_hp and pokemon.max_hp > 0:
            feats[offset] = pokemon.current_hp / pokemon.max_hp
        elif pokemon.current_hp_fraction is not None:
            feats[offset] = pokemon.current_hp_fraction
        offset += 1

        for index, status in enumerate(observation_module.STATUS_LIST):
            if pokemon.status == status:
                feats[offset + index] = 1.0
                break
        offset += observation_module.NUM_STATUS

        boosts = pokemon.boosts if pokemon.boosts else {}
        boost_names = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        for index, stat in enumerate(boost_names):
            feats[offset + index] = boosts.get(stat, 0) / 6.0
        offset += observation_module.NUM_BOOSTS

        effects = pokemon.effects
        for index, volatile in enumerate(observation_module.VOLATILES):
            if volatile == observation_module.PERISH_EFFECTS:
                counts = [
                    count
                    for effect, count in (
                        (Effect.PERISH0, 0),
                        (Effect.PERISH1, 1),
                        (Effect.PERISH2, 2),
                        (Effect.PERISH3, 3),
                    )
                    if effect in effects
                ]
                if counts:
                    feats[offset + index] = (min(counts) + 1) / 4.0
            elif isinstance(volatile, frozenset):
                feats[offset + index] = float(
                    any(effect in effects for effect in volatile)
                )
            else:
                feats[offset + index] = float(volatile in effects)

        feats[observation_module.PROTECT_COUNTER_OFFSET] = self._normalized_counter(
            pokemon.protect_counter,
            observation_module.PROTECT_COUNTER_CAP,
        )
        feats[observation_module.STATUS_COUNTER_OFFSET] = self._normalized_counter(
            pokemon.status_counter,
            observation_module.STATUS_COUNTER_CAP,
        )
        feats[observation_module.FIRST_TURN_OFFSET] = float(pokemon.first_turn)
        feats[observation_module.REVEALED_OFFSET] = float(pokemon.revealed)
        known_moves = pokemon.moves or {}
        feats[observation_module.KNOWN_MOVE_COUNT_OFFSET] = min(
            len(known_moves),
            observation_module.NUM_MOVE_SLOTS,
        ) / float(observation_module.NUM_MOVE_SLOTS)
        feats[observation_module.ABILITY_KNOWN_OFFSET] = float(
            pokemon.ability is not None
        )
        feats[observation_module.ITEM_KNOWN_OFFSET] = float(
            pokemon.item not in (None, "")
        )
        feats[observation_module.TERA_KNOWN_OFFSET] = float(
            pokemon.tera_type is not None
        )
        feats[observation_module.TERASTALLIZED_OFFSET] = float(pokemon.is_terastallized)
        feats[observation_module.ACTIVE_TURNS_OFFSET] = self._normalized_counter(
            getattr(pokemon, "active_turns", int(pokemon.first_turn)),
            observation_module.ACTIVE_TURNS_CAP,
        )


def _move(move_id: str, current_pp: int, max_pp: int) -> SimpleNamespace:
    return SimpleNamespace(id=move_id, current_pp=current_pp, max_pp=max_pp)


def _type(name: str | None) -> SimpleNamespace | None:
    return SimpleNamespace(name=name.upper()) if name else None


def _pokemon(
    species: str,
    *,
    moves: OrderedDict | None = None,
    ability: str | None = None,
    item: str | None = None,
    type_1: str = "normal",
    type_2: str | None = None,
    tera_type: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        species=species,
        moves=moves,
        last_move=None,
        ability=ability,
        item=item,
        base_types=[_type(t) for t in (type_1, type_2) if t],
        stats=dict.fromkeys(("atk", "def", "spa", "spd", "spe"), 100),
        tera_type=_type(tera_type),
        current_hp=100,
        max_hp=100,
        current_hp_fraction=1.0,
        status=None,
        boosts={},
        effects={},
        protect_counter=0,
        status_counter=0,
        first_turn=True,
        revealed=False,
        is_terastallized=False,
    )


def _assert_byte_identical(
    optimized: ObservationBuilder,
    reference: ObservationBuilder,
    battle: SimpleNamespace,
) -> None:
    actual = optimized.build(battle)
    expected = reference.build(battle)
    assert actual.keys() == expected.keys()
    for key in actual:
        assert actual[key].dtype == expected[key].dtype, key
        assert actual[key].shape == expected[key].shape, key
        assert actual[key].tobytes() == expected[key].tobytes(), key


def test_optimized_encoder_is_byte_identical_as_battle_knowledge_changes() -> None:
    game_data = GameData(gen=9)
    optimized = ObservationBuilder(game_data, team_size=6)
    reference = _ReferenceObservationBuilder(game_data, team_size=6)

    own_active = _pokemon(
        "incineroar",
        moves=OrderedDict(
            [
                ("fakeout", _move("fakeout", 16, 16)),
                ("partingshot", _move("partingshot", 32, 32)),
                ("protect", _move("protect", 16, 16)),
                ("flareblitz", _move("flareblitz", 24, 24)),
            ]
        ),
        ability="intimidate",
        item="sitrusberry",
        type_1="fire",
        type_2="dark",
        tera_type="grass",
    )
    own_bench = _pokemon(
        "amoonguss",
        moves=OrderedDict(
            [
                ("spore", _move("spore", 24, 24)),
                ("ragepowder", _move("ragepowder", 32, 32)),
            ]
        ),
        ability="regenerator",
        item="rockyhelmet",
        type_1="grass",
        type_2="poison",
        tera_type="water",
    )
    opponent_active = _pokemon(
        "gengar",
        moves=OrderedDict(),
        type_1="ghost",
        type_2="poison",
    )
    opponent_bench = _pokemon(
        "dragonite",
        moves=None,
        type_1="dragon",
        type_2="flying",
    )
    battle = SimpleNamespace(
        team=OrderedDict([("p1:2", own_bench), ("p1:1", own_active)]),
        opponent_team=OrderedDict(
            [("p2:1", opponent_active), ("p2:2", opponent_bench)]
        ),
        active_pokemon=[own_active, None],
        available_moves=[[], []],
        opponent_active_pokemon=[opponent_active, None],
        weather={},
        fields={},
        side_conditions={},
        opponent_side_conditions={},
        turn=0,
        used_tera=False,
        opponent_used_tera=False,
    )

    _assert_byte_identical(optimized, reference, battle)

    own_active.current_hp = 37
    own_active.status = Status.BRN
    own_active.boosts = {"atk": -2, "spa": 3, "evasion": 1}
    own_active.effects = {
        Effect.CONFUSION: 1,
        Effect.PERISH3: 1,
        Effect.PROTECT: 1,
    }
    own_active.protect_counter = 3
    own_active.status_counter = 5
    own_active.first_turn = False
    own_active.revealed = True
    own_active.moves["partingshot"].current_pp = 17
    own_active.last_move = own_active.moves["partingshot"]
    battle.weather = {Weather.RAINDANCE: 2}
    battle.fields = {Field.TRICK_ROOM: 3}
    battle.side_conditions = {SideCondition.TAILWIND: 3}
    battle.opponent_side_conditions = {SideCondition.STEALTH_ROCK: 1}
    battle.turn = 5

    _assert_byte_identical(optimized, reference, battle)

    opponent_active.moves["shadowball"] = _move("shadowball", 23, 24)
    opponent_active.moves["protect"] = _move("protect", 15, 16)
    opponent_active.last_move = opponent_active.moves["shadowball"]
    opponent_active.ability = "cursedbody"
    opponent_active.item = "focussash"
    opponent_active.tera_type = _type("dark")
    opponent_active.revealed = True
    opponent_active.is_terastallized = True
    opponent_active.effects = {Effect.PERISH0: 1, Effect.SUBSTITUTE: 1}
    opponent_active.status = Status.PAR
    opponent_active.current_hp = 1
    battle.opponent_used_tera = True

    _assert_byte_identical(optimized, reference, battle)

    own_bench.current_hp = 0
    own_bench.status = Status.SLP
    own_bench.status_counter = 8
    own_bench.moves["spore"].current_pp = 0
    own_bench.moves["ragepowder"].max_pp = 0
    battle.active_pokemon = [own_bench, own_active]
    battle.opponent_active_pokemon = [opponent_bench, opponent_active]
    opponent_bench.moves = OrderedDict([("extremespeed", _move("extremespeed", 7, 8))])
    opponent_bench.revealed = True
    battle.turn = 12
    battle.used_tera = True

    _assert_byte_identical(optimized, reference, battle)

    for status in [*observation_module.STATUS_LIST, Status.FNT]:
        opponent_active.status = status
        _assert_byte_identical(optimized, reference, battle)

    tracked_effects = {
        effect
        for volatile in observation_module.VOLATILES
        for effect in (volatile if isinstance(volatile, frozenset) else (volatile,))
    }
    for effect in tracked_effects:
        opponent_active.effects = {effect: 1}
        _assert_byte_identical(optimized, reference, battle)

    opponent_active.effects = {Effect.DYNAMAX: 1}
    _assert_byte_identical(optimized, reference, battle)
