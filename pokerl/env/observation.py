"""Converts a poke-env DoubleBattle into the tensor dict the model expects."""

from collections import OrderedDict

import numpy as np
from poke_env.battle import DoubleBattle, Effect, Field, Pokemon, SideCondition, Status, Weather
from poke_env.teambuilder.teambuilder import Teambuilder

from ..data import GameData, UNKNOWN_IDX


# Continuous features per pokemon: hp(1) + status_one_hot(7) + stat_stages(7)
# + volatiles(16) + move_pp(4) = 35 total. Must match PokemonEncoder continuous_dim.
CONTINUOUS_DIM = 35

WEATHER_TYPES = [
    Weather.SUNNYDAY, Weather.DESOLATELAND, Weather.RAINDANCE,
    Weather.PRIMORDIALSEA, Weather.SANDSTORM, Weather.HAIL,
    Weather.SNOW, Weather.DELTASTREAM,
]
TERRAIN_TYPES = [
    Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
    Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
]
FIELD_TIMER_CAPS = {
    "weather": 8,
    "terrain": 8,
    "screen": 8,
    "tailwind": 4,
    "trick_room": 5,
}

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
STATUS_LIST = [None, Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOX]

# Role indices
ROLE_ACTIVE_1 = 0
ROLE_ACTIVE_2 = 1
ROLE_BENCH = 2

assert 1 + 7 + 7 + NUM_VOLATILES + 4 == CONTINUOUS_DIM, (
    f"CONTINUOUS_DIM={CONTINUOUS_DIM} does not match the actual feature count "
    f"(1 + 7 + 7 + {NUM_VOLATILES} + 4 = {1 + 7 + 7 + NUM_VOLATILES + 4}). "
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

        own_pokemon, own_roles, own_active_indices, own_active_present = self._ordered_team_with_roles(battle, own=True)
        self._encode_team(own_pokemon, own_roles, obs, prefix="own", full_info=True)

        opp_pokemon, opp_roles, opp_active_indices, opp_active_present = self._ordered_team_with_roles(battle, own=False)
        self._encode_team(opp_pokemon, opp_roles, obs, prefix="opp", full_info=False)

        obs["field"] = self._encode_field(battle)
        obs["own_active_indices"] = own_active_indices
        obs["own_active_present"] = own_active_present
        obs["opp_active_indices"] = opp_active_indices
        obs["opp_active_present"] = opp_active_present
        return obs

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
        ability = np.zeros(ts, dtype=np.int64)
        item = np.zeros(ts, dtype=np.int64)
        types = np.zeros((ts, 3), dtype=np.int64)
        continuous = np.zeros((ts, CONTINUOUS_DIM), dtype=np.float32)
        mask = np.ones(ts, dtype=bool)

        for index, pokemon in enumerate(pokemon_list):
            if pokemon is None or index >= ts:
                break
            mask[index] = False
            species[index] = self._species_idx(pokemon)
            ability[index] = self._ability_idx(pokemon)
            item[index] = self._item_idx(pokemon)
            self._fill_type_indices(types[index], pokemon)
            self._fill_continuous_features(continuous[index], pokemon)
            self._fill_move_features(moves[index], continuous[index, -4:], pokemon, full_info)

        obs[f"{prefix}_species"] = species
        obs[f"{prefix}_moves"] = moves
        obs[f"{prefix}_ability"] = ability
        obs[f"{prefix}_item"] = item
        obs[f"{prefix}_types"] = types
        obs[f"{prefix}_continuous"] = continuous
        obs[f"{prefix}_roles"] = roles
        obs[f"{prefix}_mask"] = mask

    def _species_idx(self, pokemon: Pokemon) -> int:
        key = pokemon.species.lower() if pokemon.species else "<unk>"
        return self.gd.species_to_idx.get(key, UNKNOWN_IDX)

    def _ability_idx(self, pokemon: Pokemon) -> int:
        ab = pokemon.ability
        if ab is None:
            return UNKNOWN_IDX
        return self.gd.abilities_to_idx.get(ab.lower(), UNKNOWN_IDX)

    def _item_idx(self, pokemon: Pokemon) -> int:
        it = pokemon.item
        if it is None or it == "":
            return UNKNOWN_IDX
        return self.gd.item_index(it)

    def _fill_type_indices(self, indices: np.ndarray, pokemon: Pokemon) -> None:
        if pokemon.type_1:
            indices[0] = self.gd.type_to_idx.get(pokemon.type_1.name.lower(), UNKNOWN_IDX)
        if pokemon.type_2:
            indices[1] = self.gd.type_to_idx.get(pokemon.type_2.name.lower(), UNKNOWN_IDX)
        if pokemon.tera_type:
            indices[2] = self.gd.type_to_idx.get(pokemon.tera_type.name.lower(), UNKNOWN_IDX)

    def _fill_move_features(
        self,
        move_indices: np.ndarray,
        pp_features: np.ndarray,
        pokemon: Pokemon,
        full_info: bool,
    ) -> None:
        # poke-env tracks only revealed moves for the opponent; zeros = unknown.
        if not pokemon.moves:
            return
        for move_index, move in enumerate(pokemon.moves.values()):
            if move_index >= 4:
                break
            key = move.id.lower() if move.id else "<unk>"
            move_indices[move_index] = self.gd.move_to_idx.get(key, UNKNOWN_IDX)
            if full_info and move.max_pp and move.max_pp > 0:
                pp_features[move_index] = move.current_pp / move.max_pp

    def _fill_continuous_features(self, feats: np.ndarray, pokemon: Pokemon) -> None:
        offset = 0

        if pokemon.max_hp and pokemon.max_hp > 0:
            feats[offset] = pokemon.current_hp / pokemon.max_hp
        elif pokemon.current_hp_fraction is not None:
            feats[offset] = pokemon.current_hp_fraction
        offset += 1

        for index, status in enumerate(STATUS_LIST):
            if pokemon.status == status:
                feats[offset + index] = 1.0
                break
        offset += 7

        boosts = pokemon.boosts if pokemon.boosts else {}
        for index, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
            feats[offset + index] = boosts.get(stat, 0) / 6.0
        offset += 7

        effects = pokemon.effects
        for index, volatile in enumerate(VOLATILES):
            if isinstance(volatile, frozenset):
                feats[offset + index] = float(any(effect in effects for effect in volatile))
            else:
                feats[offset + index] = float(volatile in effects)

    def _ordered_team_with_roles(
        self,
        battle: DoubleBattle,
        own: bool,
    ) -> tuple[list[Pokemon | None], np.ndarray, np.ndarray, np.ndarray]:
        """Return ordered team members plus role and active-index metadata."""
        team = list(battle.team.values()) if own else list(battle.opponent_team.values())
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

        for pokemon in team:
            slot = active_lookup.get(id(pokemon))
            if slot is None:
                continue
            if len(ordered) >= self.team_size:
                break
            ordered.append(pokemon)
            ordered_index = len(ordered) - 1
            roles[ordered_index] = ROLE_ACTIVE_1 if slot == 0 else ROLE_ACTIVE_2
            active_indices[slot] = ordered_index
            active_present[slot] = True

        for pokemon in team:
            if id(pokemon) in active_lookup:
                continue
            if len(ordered) >= self.team_size:
                break
            ordered.append(pokemon)

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
        feats = np.zeros(33, dtype=np.float32)
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

        for side_conditions in [battle.side_conditions, battle.opponent_side_conditions]:
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
        return ObservationBuilder._normalized_timer_age(current_turn, started_turn, timer_cap)

    @staticmethod
    def _normalized_timer_age(current_turn: int, started_turn: int, timer_cap: int) -> float:
        turns_active = max(current_turn - started_turn, 0)
        return min(turns_active, timer_cap) / float(timer_cap)


class _SyntheticTeamPreviewBattle:
    def __init__(self, own_team: list[Pokemon], opponent_team: list[Pokemon]):
        self.team = OrderedDict((f"p1:{index}", pokemon) for index, pokemon in enumerate(own_team, start=1))
        self.opponent_team = OrderedDict(
            (f"p2:{index}", pokemon) for index, pokemon in enumerate(opponent_team, start=1)
        )
        self.active_pokemon = [None, None]
        self.opponent_active_pokemon = [None, None]
        self.fields = {}
        self.weather = {}
        self.side_conditions = {}
        self.opponent_side_conditions = {}
        self.turn = 0


def build_team_preview_observation(
    game_data: GameData,
    team_size: int,
    own_packed_team: str,
    opponent_packed_team: str,
) -> dict[str, np.ndarray]:
    own_team = [
        _init_team_preview_pokemon(Pokemon(gen=game_data.gen, teambuilder=tb), own_team=True)
        for tb in Teambuilder.parse_packed_team(own_packed_team)[:team_size]
    ]
    opponent_team = [
        _init_team_preview_pokemon(
            Pokemon(gen=game_data.gen, species=_preview_visible_species(game_data.gen, tb)),
            own_team=False,
        )
        for tb in Teambuilder.parse_packed_team(opponent_packed_team)[:team_size]
    ]
    battle = _SyntheticTeamPreviewBattle(own_team, opponent_team)
    return ObservationBuilder(game_data, team_size).build(battle)


def _preview_visible_species(gen: int, tb) -> str:
    species = Pokemon(gen=gen, teambuilder=tb).species
    if species is None:
        raise ValueError(f"Could not infer canonical species from teambuilder pokemon: {tb}")
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
        pokemon._terastallized_type = None
    else:
        pokemon._max_hp = 0
        pokemon._current_hp = 0
    return pokemon
