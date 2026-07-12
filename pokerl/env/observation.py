"""Converts a poke-env DoubleBattle into the tensor dict the model expects."""

from collections import OrderedDict
from functools import lru_cache

import numpy as np
from poke_env.battle import (
    DoubleBattle,
    Effect,
    Field,
    Pokemon,
    SideCondition,
    Status,
    Weather,
)
from poke_env.data import GenData
from poke_env.data.normalize import to_id_str
from poke_env.teambuilder.teambuilder import Teambuilder

from ..data import GameData, UNKNOWN_IDX


NUM_STATUS = 7
NUM_BOOSTS = 7
NUM_MOVE_SLOTS = 4
NUM_EXTRA_POKEMON_FEATURES = 9

WEATHER_TYPES = [
    Weather.SUNNYDAY,
    Weather.DESOLATELAND,
    Weather.RAINDANCE,
    Weather.PRIMORDIALSEA,
    Weather.SANDSTORM,
    Weather.HAIL,
    Weather.SNOW,
    Weather.DELTASTREAM,
]
TERRAIN_TYPES = [
    Field.ELECTRIC_TERRAIN,
    Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN,
    Field.PSYCHIC_TERRAIN,
]
FIELD_TIMER_CAPS = {
    "weather": 8,
    "terrain": 8,
    "screen": 8,
    "tailwind": 4,
    "trick_room": 5,
    "turn": 20,
}

PROTECT_COUNTER_CAP = 4
STATUS_COUNTER_CAP = 8

# Volatile effects to track as binary flags.
# Each entry is either a single Effect or a frozenset of Effects (any match = True).
# Perish song is tracked as any of PERISH0/1/2/3 (countdown from 3 to 0).
VOLATILES: list = [
    Effect.CONFUSION,
    Effect.SUBSTITUTE,
    Effect.LEECH_SEED,
    Effect.TAUNT,
    Effect.ENCORE,
    Effect.TORMENT,
    Effect.DISABLE,
    Effect.YAWN,
    frozenset({Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3}),
    Effect.PROTECT,
    Effect.ENDURE,
    Effect.FOCUS_ENERGY,
    Effect.ATTRACT,
    Effect.CURSE,
    Effect.NIGHTMARE,
    Effect.FLASH_FIRE,
]
NUM_VOLATILES = len(VOLATILES)

# Status conditions for one-hot encoding
STATUS_LIST = [
    None,
    Status.BRN,
    Status.FRZ,
    Status.PAR,
    Status.PSN,
    Status.SLP,
    Status.TOX,
]
_STATUS_INDICES = {status: index for index, status in enumerate(STATUS_LIST)}
_BOOST_NAMES = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
_VOLATILE_INDICES = {
    effect: index
    for index, volatile in enumerate(VOLATILES)
    for effect in (volatile if isinstance(volatile, frozenset) else (volatile,))
}

HP_OFFSET = 0
STATUS_OFFSET = HP_OFFSET + 1
BOOSTS_OFFSET = STATUS_OFFSET + NUM_STATUS
VOLATILES_OFFSET = BOOSTS_OFFSET + NUM_BOOSTS
MOVE_PP_OFFSET = VOLATILES_OFFSET + NUM_VOLATILES
EXTRA_FEATURES_OFFSET = MOVE_PP_OFFSET + NUM_MOVE_SLOTS

PROTECT_COUNTER_OFFSET = EXTRA_FEATURES_OFFSET + 0
STATUS_COUNTER_OFFSET = EXTRA_FEATURES_OFFSET + 1
FIRST_TURN_OFFSET = EXTRA_FEATURES_OFFSET + 2
REVEALED_OFFSET = EXTRA_FEATURES_OFFSET + 3
KNOWN_MOVE_COUNT_OFFSET = EXTRA_FEATURES_OFFSET + 4
ABILITY_KNOWN_OFFSET = EXTRA_FEATURES_OFFSET + 5
ITEM_KNOWN_OFFSET = EXTRA_FEATURES_OFFSET + 6
TERA_KNOWN_OFFSET = EXTRA_FEATURES_OFFSET + 7
TERASTALLIZED_OFFSET = EXTRA_FEATURES_OFFSET + 8

# Continuous features per pokemon:
# hp(1) + status_one_hot(7) + stat_stages(7) + volatiles(16) + move_pp(4)
# + explicit legal state/history features(9) = 44 total.
# Must match PokemonEncoder continuous_dim.
CONTINUOUS_DIM = EXTRA_FEATURES_OFFSET + NUM_EXTRA_POKEMON_FEATURES

FIELD_WEATHER_OFFSET = 0
FIELD_TERRAIN_OFFSET = 10
FIELD_OWN_SIDE_OFFSET = 16
FIELD_OPP_SIDE_OFFSET = 24
FIELD_TRICK_ROOM_OFFSET = 32
FIELD_USED_TERA_OFFSET = 33
FIELD_OPPONENT_USED_TERA_OFFSET = 34
FIELD_TURN_OFFSET = 35

# Role indices
ROLE_ACTIVE_1 = 0
ROLE_ACTIVE_2 = 1
ROLE_BENCH = 2
_ILLUSION_SPECIES = frozenset(("zoroark", "zoroarkhisui"))

assert (
    1
    + NUM_STATUS
    + NUM_BOOSTS
    + NUM_VOLATILES
    + NUM_MOVE_SLOTS
    + NUM_EXTRA_POKEMON_FEATURES
    == CONTINUOUS_DIM
), (
    f"CONTINUOUS_DIM={CONTINUOUS_DIM} does not match the actual feature count "
    "(1 + "
    f"{NUM_STATUS} + {NUM_BOOSTS} + {NUM_VOLATILES} + {NUM_MOVE_SLOTS} + {NUM_EXTRA_POKEMON_FEATURES}"
    f" = {1 + NUM_STATUS + NUM_BOOSTS + NUM_VOLATILES + NUM_MOVE_SLOTS + NUM_EXTRA_POKEMON_FEATURES}). "
    "Update CONTINUOUS_DIM or the feature list."
)


class ObservationBuilder:
    """Builds observation tensors from a DoubleBattle."""

    def __init__(self, game_data: GameData, team_size: int = 6):
        self.gd = game_data
        self.team_size = team_size

    def build(self, battle: DoubleBattle) -> dict[str, np.ndarray]:
        """Convert a battle state to a dict of numpy arrays."""
        obs = {}

        own_pokemon, own_roles, own_active_indices, own_active_present = (
            self._ordered_team_with_roles(battle, own=True)
        )
        self._encode_team(own_pokemon, own_roles, obs, prefix="own", full_info=True)

        opponent_team, opponent_source_lookup = self._public_opponent_team(battle)
        opp_pokemon, opp_roles, opp_active_indices, opp_active_present = (
            self._ordered_team_with_roles(
                battle,
                own=False,
                team_override=opponent_team,
            )
        )
        self._encode_team(opp_pokemon, opp_roles, obs, prefix="opp", full_info=False)
        obs["opp_source_index"] = np.asarray(
            [
                opponent_source_lookup.get(id(pokemon), -1)
                if pokemon is not None
                else -1
                for pokemon in opp_pokemon
            ],
            dtype=np.int64,
        )

        obs["field"] = self._encode_field(battle)
        obs["own_active_indices"] = own_active_indices
        obs["own_active_present"] = own_active_present
        obs["opp_active_indices"] = opp_active_indices
        obs["opp_active_present"] = opp_active_present
        return obs

    def _public_opponent_team(
        self,
        battle: DoubleBattle,
    ) -> tuple[list[Pokemon], dict[int, int]]:
        """Overlay public live state onto the immutable team-preview roster."""
        live_team = list(battle.opponent_team.values())
        preview_team = list(getattr(battle, "teampreview_opponent_team", ()) or ())[
            : self.team_size
        ]
        if not preview_team:
            return live_team, {
                id(pokemon): index for index, pokemon in enumerate(live_team)
            }

        roster = list(preview_team)
        source_lookup = {
            id(pokemon): index for index, pokemon in enumerate(preview_team)
        }
        preview_object_ids = set(source_lookup)
        preview_slots_by_species: dict[str, list[int]] = {}
        for index, pokemon in enumerate(preview_team):
            preview_slots_by_species.setdefault(
                self._base_species_id(pokemon),
                [],
            ).append(index)

        public_live_team = [
            pokemon for pokemon in live_team if id(pokemon) not in preview_object_ids
        ]
        illusion_is_confirmed = any(
            self._base_species_id(pokemon) in _ILLUSION_SPECIES
            for pokemon in public_live_team
        )
        illusion_is_ambiguous = (
            any(species in preview_slots_by_species for species in _ILLUSION_SPECIES)
            and not illusion_is_confirmed
        )

        for pokemon in public_live_team:
            species = self._base_species_id(pokemon)
            candidate_slots = preview_slots_by_species.get(species, ())
            if len(candidate_slots) != 1:
                source_lookup[id(pokemon)] = -1
                continue
            source_index = candidate_slots[0]
            roster[source_index] = pokemon
            source_lookup[id(pokemon)] = (
                -1
                if illusion_is_ambiguous and species not in _ILLUSION_SPECIES
                else source_index
            )

        return roster, source_lookup

    @staticmethod
    def _base_species_id(pokemon: Pokemon) -> str:
        return to_id_str(getattr(pokemon, "base_species", pokemon.species))

    def _encode_team(
        self,
        pokemon_list: list[Pokemon | None],
        roles: np.ndarray,
        obs: dict,
        prefix: str,
        full_info: bool,
    ) -> None:
        """Encode a team into arrays."""
        ts = self.team_size
        species = np.zeros(ts, dtype=np.int64)
        moves = np.zeros((ts, 4), dtype=np.int64)
        last_move = np.zeros(ts, dtype=np.int64)
        ability = np.zeros(ts, dtype=np.int64)
        item = np.zeros(ts, dtype=np.int64)
        types = np.zeros((ts, 3), dtype=np.int64)
        continuous = np.zeros((ts, CONTINUOUS_DIM), dtype=np.float32)
        mask = np.ones(ts, dtype=bool)

        for index, pokemon in enumerate(pokemon_list):
            if pokemon is None or index >= ts:
                break
            pokemon_moves = pokemon.moves or {}
            ability_value = pokemon.ability
            item_value = pokemon.item
            last_move_value = pokemon.last_move
            mask[index] = False
            species[index] = self._species_idx(pokemon)
            ability[index] = self._ability_idx(ability_value)
            item[index] = self._item_idx(item_value)
            last_move[index] = self._move_idx(
                last_move_value.id if last_move_value is not None else None
            )
            self._fill_type_indices(types[index], pokemon)
            self._fill_continuous_features(
                continuous[index],
                pokemon,
                pokemon_moves,
                ability_value,
                item_value,
            )
            self._fill_move_features(
                moves[index],
                continuous[index, MOVE_PP_OFFSET : MOVE_PP_OFFSET + NUM_MOVE_SLOTS],
                pokemon_moves,
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

    def _species_idx(self, pokemon: Pokemon) -> int:
        key = pokemon.species.lower() if pokemon.species else "<unk>"
        return self.gd.species_to_idx.get(key, UNKNOWN_IDX)

    def _move_idx(self, move_id: str | None) -> int:
        key = move_id.lower() if move_id else "<unk>"
        return self.gd.move_to_idx.get(key, UNKNOWN_IDX)

    def _ability_idx(self, ability: str | None) -> int:
        if ability is None:
            return UNKNOWN_IDX
        return self.gd.abilities_to_idx.get(ability.lower(), UNKNOWN_IDX)

    def _item_idx(self, item: str | None) -> int:
        if item in (None, "", GenData.UNKNOWN_ITEM):
            return UNKNOWN_IDX
        return self.gd.item_index(item)

    def _fill_type_indices(self, indices: np.ndarray, pokemon: Pokemon) -> None:
        if pokemon.type_1:
            indices[0] = self.gd.type_to_idx.get(
                pokemon.type_1.name.lower(), UNKNOWN_IDX
            )
        if pokemon.type_2:
            indices[1] = self.gd.type_to_idx.get(
                pokemon.type_2.name.lower(), UNKNOWN_IDX
            )
        if pokemon.tera_type:
            indices[2] = self.gd.type_to_idx.get(
                pokemon.tera_type.name.lower(), UNKNOWN_IDX
            )

    def _fill_move_features(
        self,
        move_indices: np.ndarray,
        pp_features: np.ndarray,
        pokemon_moves: dict,
        full_info: bool,
    ) -> None:
        # poke-env tracks only revealed moves for the opponent; zeros = unknown.
        for move_index, move in enumerate(pokemon_moves.values()):
            if move_index >= 4:
                break
            move_indices[move_index] = self._move_idx(move.id)
            if full_info:
                max_pp = move.max_pp
                if max_pp > 0:
                    pp_features[move_index] = move.current_pp / max_pp

    def _fill_continuous_features(
        self,
        feats: np.ndarray,
        pokemon: Pokemon,
        pokemon_moves: dict,
        ability: str | None,
        item: str | None,
    ) -> None:

        if pokemon.max_hp and pokemon.max_hp > 0:
            feats[HP_OFFSET] = pokemon.current_hp / pokemon.max_hp
        elif pokemon.current_hp_fraction is not None:
            feats[HP_OFFSET] = pokemon.current_hp_fraction

        status_index = _STATUS_INDICES.get(pokemon.status)
        if status_index is not None:
            feats[STATUS_OFFSET + status_index] = 1.0

        boosts = pokemon.boosts if pokemon.boosts else {}
        for index, stat in enumerate(_BOOST_NAMES):
            feats[BOOSTS_OFFSET + index] = boosts.get(stat, 0) / 6.0

        for effect in pokemon.effects:
            volatile_index = _VOLATILE_INDICES.get(effect)
            if volatile_index is not None:
                feats[VOLATILES_OFFSET + volatile_index] = 1.0

        feats[PROTECT_COUNTER_OFFSET] = self._normalized_counter(
            pokemon.protect_counter, PROTECT_COUNTER_CAP
        )
        feats[STATUS_COUNTER_OFFSET] = self._normalized_counter(
            pokemon.status_counter, STATUS_COUNTER_CAP
        )
        feats[FIRST_TURN_OFFSET] = float(pokemon.first_turn)
        feats[REVEALED_OFFSET] = float(pokemon.revealed)
        feats[KNOWN_MOVE_COUNT_OFFSET] = min(
            len(pokemon_moves), NUM_MOVE_SLOTS
        ) / float(NUM_MOVE_SLOTS)
        feats[ABILITY_KNOWN_OFFSET] = float(ability is not None)
        feats[ITEM_KNOWN_OFFSET] = float(item not in (None, "", GenData.UNKNOWN_ITEM))
        feats[TERA_KNOWN_OFFSET] = float(pokemon.tera_type is not None)
        feats[TERASTALLIZED_OFFSET] = float(pokemon.is_terastallized)

    def _ordered_team_with_roles(
        self,
        battle: DoubleBattle,
        own: bool,
        team_override: list[Pokemon] | None = None,
    ) -> tuple[list[Pokemon | None], np.ndarray, np.ndarray, np.ndarray]:
        """Return ordered team members plus role and active-index metadata."""
        team = (
            list(team_override)
            if team_override is not None
            else (
                list(battle.team.values())
                if own
                else list(battle.opponent_team.values())
            )
        )
        active_list = battle.active_pokemon if own else battle.opponent_active_pokemon
        active_lookup = {
            id(pokemon): slot
            for slot, pokemon in enumerate(active_list)
            if pokemon is not None
        }

        ordered: list[Pokemon | None] = []
        roles = np.full(self.team_size, ROLE_BENCH, dtype=np.int64)
        active_indices = np.zeros(2, dtype=np.int64)
        active_present = np.zeros(2, dtype=bool)
        ordered_ids: set[int] = set()

        def append_active(pokemon: Pokemon, slot: int) -> None:
            if len(ordered) >= self.team_size:
                return
            ordered.append(pokemon)
            ordered_ids.add(id(pokemon))
            ordered_index = len(ordered) - 1
            roles[ordered_index] = ROLE_ACTIVE_1 if slot == 0 else ROLE_ACTIVE_2
            active_indices[slot] = ordered_index
            active_present[slot] = True

        for pokemon in team:
            slot = active_lookup.get(id(pokemon))
            if slot is None:
                continue
            append_active(pokemon, slot)

        if not own:
            for slot, pokemon in enumerate(active_list):
                if pokemon is not None and not active_present[slot]:
                    append_active(pokemon, slot)

        for pokemon in team:
            if id(pokemon) in active_lookup or id(pokemon) in ordered_ids:
                continue
            if len(ordered) >= self.team_size:
                break
            ordered.append(pokemon)
            ordered_ids.add(id(pokemon))

        for slot, pokemon in enumerate(active_list):
            if pokemon is not None and not active_present[slot]:
                raise ValueError(
                    f"Active pokemon {pokemon.species!r} was not found in the ordered team list. "
                    "This indicates a mismatch between battle.active_pokemon and battle.team - "
                    "check poke-env battle state tracking."
                )

        while len(ordered) < self.team_size:
            ordered.append(None)
        return ordered, roles, active_indices, active_present

    @staticmethod
    def _encode_field(battle: DoubleBattle) -> np.ndarray:
        """Encode field conditions into a flat feature vector."""
        feats = np.zeros(FIELD_TURN_OFFSET + 1, dtype=np.float32)
        offset = 0
        current_turn = getattr(battle, "turn", 0)

        if not battle.weather:
            feats[offset] = 1.0
        else:
            for index, weather in enumerate(WEATHER_TYPES):
                started_turn = battle.weather.get(weather)
                if started_turn is not None:
                    feats[offset + 1 + index] = 1.0
                    feats[offset + 9] = ObservationBuilder._normalized_timer_age(
                        current_turn,
                        started_turn,
                        FIELD_TIMER_CAPS["weather"],
                    )
                    break
        offset += 10

        terrain_active = False
        for index, terrain in enumerate(TERRAIN_TYPES):
            started_turn = battle.fields.get(terrain)
            if started_turn is not None:
                feats[offset + 1 + index] = 1.0
                feats[offset + 5] = ObservationBuilder._normalized_timer_age(
                    current_turn,
                    started_turn,
                    FIELD_TIMER_CAPS["terrain"],
                )
                terrain_active = True
                break
        if not terrain_active:
            feats[offset] = 1.0
        offset += 6

        for side_conditions in [
            battle.side_conditions,
            battle.opponent_side_conditions,
        ]:
            feats[offset] = float(SideCondition.STEALTH_ROCK in side_conditions)
            offset += 1
            feats[offset] = side_conditions.get(SideCondition.SPIKES, 0) / 3.0
            offset += 1
            feats[offset] = side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0
            offset += 1
            feats[offset] = float(SideCondition.STICKY_WEB in side_conditions)
            offset += 1
            feats[offset] = ObservationBuilder._condition_age_feature(
                side_conditions,
                SideCondition.REFLECT,
                current_turn,
                FIELD_TIMER_CAPS["screen"],
            )
            offset += 1
            feats[offset] = ObservationBuilder._condition_age_feature(
                side_conditions,
                SideCondition.LIGHT_SCREEN,
                current_turn,
                FIELD_TIMER_CAPS["screen"],
            )
            offset += 1
            feats[offset] = ObservationBuilder._condition_age_feature(
                side_conditions,
                SideCondition.AURORA_VEIL,
                current_turn,
                FIELD_TIMER_CAPS["screen"],
            )
            offset += 1
            feats[offset] = ObservationBuilder._condition_age_feature(
                side_conditions,
                SideCondition.TAILWIND,
                current_turn,
                FIELD_TIMER_CAPS["tailwind"],
            )
            offset += 1

        feats[offset] = ObservationBuilder._condition_age_feature(
            battle.fields,
            Field.TRICK_ROOM,
            current_turn,
            FIELD_TIMER_CAPS["trick_room"],
        )
        offset += 1

        feats[offset] = float(getattr(battle, "used_tera", False))
        offset += 1
        feats[offset] = float(getattr(battle, "opponent_used_tera", False))
        offset += 1
        feats[offset] = ObservationBuilder._normalized_timer_age(
            current_turn,
            0,
            FIELD_TIMER_CAPS["turn"],
        )
        return feats

    @staticmethod
    def _condition_age_feature(
        conditions: dict,
        condition,
        current_turn: int,
        timer_cap: int,
    ) -> float:
        started_turn = conditions.get(condition)
        if started_turn is None or not isinstance(started_turn, int):
            return 0.0
        return ObservationBuilder._normalized_timer_age(
            current_turn, started_turn, timer_cap
        )

    @staticmethod
    def _normalized_timer_age(
        current_turn: int, started_turn: int, timer_cap: int
    ) -> float:
        turns_active = max(current_turn - started_turn, 0)
        return min(turns_active, timer_cap) / float(timer_cap)

    @staticmethod
    def _normalized_counter(value: int, cap: int) -> float:
        return min(max(int(value), 0), cap) / float(cap)


class _SyntheticTeamPreviewBattle:
    def __init__(self, own_team: list[Pokemon], opponent_team: list[Pokemon]):
        self.team = OrderedDict(
            (f"p1:{index}", pokemon) for index, pokemon in enumerate(own_team, start=1)
        )
        self.opponent_team = OrderedDict(
            (f"p2:{index}", pokemon)
            for index, pokemon in enumerate(opponent_team, start=1)
        )
        self.active_pokemon = [None, None]
        self.opponent_active_pokemon = [None, None]
        self.fields = {}
        self.weather = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.turn = 0
        self.used_tera = False
        self.opponent_used_tera = False


def build_team_preview_observation(
    game_data: GameData,
    team_size: int,
    own_packed_team: str,
    opponent_packed_team: str,
) -> dict[str, np.ndarray]:
    own_team = [
        _init_team_preview_pokemon(
            Pokemon(gen=game_data.gen, teambuilder=tb), own_team=True
        )
        for tb in Teambuilder.parse_packed_team(own_packed_team)[:team_size]
    ]
    opponent_team = [
        _init_team_preview_pokemon(
            Pokemon(
                gen=game_data.gen, species=_preview_visible_species(game_data.gen, tb)
            ),
            own_team=False,
        )
        for tb in Teambuilder.parse_packed_team(opponent_packed_team)[:team_size]
    ]
    battle = _SyntheticTeamPreviewBattle(own_team, opponent_team)
    return ObservationBuilder(game_data, team_size).build(battle)


def summarize_observation_mismatch(
    actual: dict[str, np.ndarray],
    prepared: dict[str, np.ndarray],
) -> str:
    """Describe a failed prepared-observation equality check compactly."""
    if actual.keys() != prepared.keys():
        missing = sorted(prepared.keys() - actual.keys())
        extra = sorted(actual.keys() - prepared.keys())
        return f"missing keys={missing}, extra keys={extra}"

    summaries: list[str] = []
    for key in actual:
        left = np.asarray(actual[key])
        right = np.asarray(prepared[key])
        if np.array_equal(left, right):
            continue
        if left.shape != right.shape:
            summaries.append(f"{key} shape live={left.shape} prepared={right.shape}")
            continue
        differing = np.flatnonzero(left.reshape(-1) != right.reshape(-1))
        samples = [
            (
                int(index),
                left.reshape(-1)[index].item(),
                right.reshape(-1)[index].item(),
            )
            for index in differing[:4]
        ]
        summaries.append(
            f"{key} differs at {len(differing)} value(s) "
            f"(flat index, live, prepared)={samples}"
        )
    return "; ".join(summaries) or "unknown difference"


@lru_cache(maxsize=512)
def _cached_packed_team_aux_labels(
    game_data: GameData,
    team_size: int,
    packed_team: str,
) -> dict[str, np.ndarray]:
    """Encode a packed team once for trainer-private auxiliary targets."""
    pokemon = [
        Pokemon(gen=game_data.gen, teambuilder=entry)
        for entry in Teambuilder.parse_packed_team(packed_team)[:team_size]
    ]
    padded_team: list[Pokemon | None] = [*pokemon]
    padded_team.extend([None] * (team_size - len(padded_team)))
    labels: dict[str, np.ndarray] = {}
    ObservationBuilder(game_data, team_size)._encode_team(
        padded_team,
        np.full(team_size, ROLE_BENCH, dtype=np.int64),
        labels,
        prefix="opp",
        full_info=True,
    )
    source_indices = np.full(team_size, -1, dtype=np.int64)
    source_indices[: len(pokemon)] = np.arange(len(pokemon), dtype=np.int64)
    labels["opp_source_index"] = source_indices
    return labels


def build_packed_team_aux_labels(
    game_data: GameData,
    team_size: int,
    packed_team: str,
    *,
    selected_indices: np.ndarray | list[int] | tuple[int, ...] | None = None,
) -> dict[str, np.ndarray]:
    """Return exact, trainer-only opponent labels in team-preview order.

    The packed team is already assigned to the simulator, but none of these
    arrays are included in the policy observation.  They are attached only to
    terminal training info so the hidden-information auxiliary head can learn
    from complete item, ability, tera-type, and move-set targets.
    """
    labels = {
        key: value.copy()
        for key, value in _cached_packed_team_aux_labels(
            game_data,
            team_size,
            packed_team,
        ).items()
    }
    if selected_indices is None:
        return labels

    selected = np.asarray(selected_indices, dtype=np.int64)
    if selected.ndim != 1:
        raise ValueError("selected_indices must be one-dimensional.")
    available = int((~labels["opp_mask"]).sum())
    if (
        len(np.unique(selected)) != len(selected)
        or np.any(selected < 0)
        or np.any(selected >= available)
    ):
        raise ValueError("selected_indices must contain unique packed-team positions.")
    selected_mask = np.ones(team_size, dtype=bool)
    selected_mask[selected] = False
    labels["opp_mask"] |= selected_mask
    return labels


def _preview_visible_species(gen: int, tb) -> str:
    species = Pokemon(gen=gen, teambuilder=tb).species
    if species is None:
        raise ValueError(
            f"Could not infer canonical species from teambuilder pokemon: {tb}"
        )
    if species == "urshifurapidstrike":
        return "urshifu"
    return species


def _init_team_preview_pokemon(pokemon: Pokemon, *, own_team: bool) -> Pokemon:
    if own_team:
        max_hp = pokemon.stats.get("hp") if pokemon.stats else None
        if max_hp is None or max_hp <= 0:
            max_hp = 100
        pokemon._max_hp = max_hp
        pokemon._current_hp = max_hp
    else:
        pokemon._max_hp = 0
        pokemon._current_hp = 0
    return pokemon
