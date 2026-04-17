"""Converts a poke-env DoubleBattle into the tensor dict the model expects."""

from collections import OrderedDict

import numpy as np
from poke_env.battle import DoubleBattle, Effect, Field, Pokemon, SideCondition, Status, Weather
from poke_env.teambuilder.teambuilder import Teambuilder

from ..data import GameData, UNKNOWN_IDX


# Continuous features per pokemon: hp(1) + status_one_hot(7) + stat_stages(7)
# + volatiles(16) + move_pp(4) = 35 total. Must match PokemonEncoder continuous_dim.
CONTINUOUS_DIM = 35

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
        """Convert a battle state to a dict of numpy arrays.

        All arrays have no batch dimension (added by the env wrapper / collation).
        """
        obs = {}

        # Own team
        own_pokemon = self._ordered_team(battle, own=True)
        own_active_slots = self._active_slot_map(battle, own=True)
        self._encode_team(own_pokemon, own_active_slots, obs, prefix="own", full_info=True)

        # Opponent team
        opp_pokemon = self._ordered_team(battle, own=False)
        opp_active_slots = self._active_slot_map(battle, own=False)
        self._encode_team(opp_pokemon, opp_active_slots, obs, prefix="opp", full_info=False)

        # Field conditions
        obs["field"] = self._encode_field(battle)

        # Which own slots are active (indices into the team array)
        obs["own_active_indices"] = np.array(
            [self._find_active_index(own_pokemon, battle.active_pokemon[s]) for s in range(2)],
            dtype=np.int64,
        )

        return obs

    def _encode_team(
        self,
        pokemon_list: list[Pokemon | None],
        active_map: dict,
        obs: dict,
        prefix: str,
        full_info: bool,
    ):
        """Encode a team into arrays."""
        ts = self.team_size
        species = np.zeros(ts, dtype=np.int64)
        moves = np.zeros((ts, 4), dtype=np.int64)
        ability = np.zeros(ts, dtype=np.int64)
        item = np.zeros(ts, dtype=np.int64)
        types = np.zeros((ts, 3), dtype=np.int64)  # type_1, type_2, tera_type
        continuous = np.zeros((ts, CONTINUOUS_DIM), dtype=np.float32)
        roles = np.full(ts, ROLE_BENCH, dtype=np.int64)
        mask = np.ones(ts, dtype=bool)  # True = padding

        for i, pokemon in enumerate(pokemon_list):
            if pokemon is None or i >= ts:
                break
            mask[i] = False
            species[i] = self._species_idx(pokemon)
            moves[i] = self._move_indices(pokemon)
            ability[i] = self._ability_idx(pokemon)
            item[i] = self._item_idx(pokemon)
            types[i] = self._type_indices(pokemon)
            continuous[i] = self._continuous_features(pokemon, full_info)

            # Role
            if pokemon in active_map:
                roles[i] = active_map[pokemon]

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

    def _move_indices(self, pokemon: Pokemon) -> np.ndarray:
        # poke-env tracks only revealed moves for the opponent; zeros = unknown
        indices = np.zeros(4, dtype=np.int64)
        known_moves = list(pokemon.moves.values()) if pokemon.moves else []
        for j, move in enumerate(known_moves[:4]):
            key = move.id.lower() if move.id else "<unk>"
            indices[j] = self.gd.move_to_idx.get(key, UNKNOWN_IDX)
        return indices

    def _ability_idx(self, pokemon: Pokemon) -> int:
        # poke-env fills in the ability once it is revealed; None = still unknown
        ab = pokemon.ability
        if ab is None:
            return UNKNOWN_IDX
        return self.gd.abilities_to_idx.get(ab.lower(), UNKNOWN_IDX)

    def _item_idx(self, pokemon: Pokemon) -> int:
        # poke-env fills in the item once it is revealed; None/empty = still unknown
        it = pokemon.item
        if it is None or it == "":
            return UNKNOWN_IDX
        return self.gd.item_index(it)

    def _type_indices(self, pokemon: Pokemon) -> np.ndarray:
        indices = np.zeros(3, dtype=np.int64)
        if pokemon.type_1:
            indices[0] = self.gd.type_to_idx.get(pokemon.type_1.name.lower(), UNKNOWN_IDX)
        if pokemon.type_2:
            indices[1] = self.gd.type_to_idx.get(pokemon.type_2.name.lower(), UNKNOWN_IDX)
        if pokemon.tera_type:
            indices[2] = self.gd.type_to_idx.get(pokemon.tera_type.name.lower(), UNKNOWN_IDX)
        return indices

    def _continuous_features(self, pokemon: Pokemon, full_info: bool) -> np.ndarray:
        feats = np.zeros(CONTINUOUS_DIM, dtype=np.float32)
        offset = 0

        # HP (normalized)
        if pokemon.max_hp and pokemon.max_hp > 0:
            feats[offset] = pokemon.current_hp / pokemon.max_hp
        elif pokemon.current_hp_fraction is not None:
            feats[offset] = pokemon.current_hp_fraction
        offset += 1

        # Status one-hot (7)
        for j, status in enumerate(STATUS_LIST):
            if pokemon.status == status:
                feats[offset + j] = 1.0
                break
        offset += 7

        # Stat stages (7): atk, def, spa, spd, spe, accuracy, evasion, normalized to [-1, 1]
        boosts = pokemon.boosts if pokemon.boosts else {}
        for j, stat in enumerate(["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]):
            feats[offset + j] = boosts.get(stat, 0) / 6.0
        offset += 7

        # Volatile statuses (NUM_VOLATILES binary flags)
        effects = pokemon.effects  # Dict[Effect, int]
        for j, vol in enumerate(VOLATILES):
            if isinstance(vol, frozenset):
                feats[offset + j] = float(any(e in effects for e in vol))
            else:
                feats[offset + j] = float(vol in effects)
        offset += NUM_VOLATILES

        # Move PP (4), normalized by max PP, only for own pokemon (full_info)
        if full_info and pokemon.moves:
            for j, move in enumerate(list(pokemon.moves.values())[:4]):
                if move.max_pp and move.max_pp > 0:
                    feats[offset + j] = move.current_pp / move.max_pp
        offset += 4

        return feats

    def _ordered_team(self, battle: DoubleBattle, own: bool) -> list[Pokemon | None]:
        """Get team as a list, active pokemon first, then bench."""
        if own:
            team = list(battle.team.values())
        else:
            team = list(battle.opponent_team.values())

        # Sort: active first, then bench, then pad with None
        active_set = set()
        for p in (battle.active_pokemon if own else battle.opponent_active_pokemon):
            if p is not None:
                active_set.add(p)

        active = [p for p in team if p in active_set]
        bench = [p for p in team if p not in active_set]
        ordered = active + bench

        # Pad to team_size
        while len(ordered) < self.team_size:
            ordered.append(None)
        return ordered[:self.team_size]

    def _active_slot_map(self, battle: DoubleBattle, own: bool) -> dict:
        """Map active pokemon objects to their role indices."""
        active_list = battle.active_pokemon if own else battle.opponent_active_pokemon
        mapping = {}
        for slot, pokemon in enumerate(active_list):
            if pokemon is not None:
                mapping[pokemon] = ROLE_ACTIVE_1 if slot == 0 else ROLE_ACTIVE_2
        return mapping

    @staticmethod
    def _find_active_index(team_list: list, pokemon) -> int:
        """Find where a specific pokemon is in the ordered team list.

        Returns 0 if the slot is empty (fainted/not yet sent out) — the action
        mask will force pass or switch for that slot so the rep value doesn't matter.
        Raises if a non-None pokemon cannot be found, which indicates a bug.
        """
        if pokemon is None:
            return 0
        for i, p in enumerate(team_list):
            if p is pokemon:
                return i
        raise ValueError(
            f"Active pokemon {pokemon.species!r} was not found in the ordered team list. "
            "This indicates a mismatch between battle.active_pokemon and battle.team — "
            "check poke-env battle state tracking."
        )

    @staticmethod
    def _encode_field(battle: DoubleBattle) -> np.ndarray:
        """Encode field conditions into a flat feature vector."""
        feats = np.zeros(31, dtype=np.float32)
        offset = 0

        # Weather (9: 8 types + none). Use first slot as "no weather"
        weather_types = [
            Weather.SUNNYDAY, Weather.DESOLATELAND, Weather.RAINDANCE,
            Weather.PRIMORDIALSEA, Weather.SANDSTORM, Weather.HAIL,
            Weather.SNOW, Weather.DELTASTREAM,
        ]
        if not battle.weather:
            feats[offset] = 1.0
        else:
            for j, w in enumerate(weather_types):
                if w in battle.weather:
                    feats[offset + 1 + j] = 1.0
                    break
        offset += 9

        # Terrain (5: 4 types + none).
        # Note: battle.fields also contains TRICK_ROOM and other non-terrain effects,
        # so we must check for each terrain type explicitly rather than using `not battle.fields`.
        terrain_types = [
            Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
            Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
        ]
        terrain_active = False
        for j, t in enumerate(terrain_types):
            if battle.fields and t in battle.fields:
                feats[offset + 1 + j] = 1.0
                terrain_active = True
                break
        if not terrain_active:
            feats[offset] = 1.0  # no terrain
        offset += 5

        # Side conditions, per side (own then opp)
        for side_conditions in [battle.side_conditions, battle.opponent_side_conditions]:
            # Stealth Rock (1)
            feats[offset] = float(SideCondition.STEALTH_ROCK in side_conditions)
            offset += 1
            # Spikes (0-3)
            feats[offset] = side_conditions.get(SideCondition.SPIKES, 0) / 3.0
            offset += 1
            # Toxic Spikes (0-2)
            feats[offset] = side_conditions.get(SideCondition.TOXIC_SPIKES, 0) / 2.0
            offset += 1
            # Sticky Web (1)
            feats[offset] = float(SideCondition.STICKY_WEB in side_conditions)
            offset += 1
            # Screens
            feats[offset] = float(SideCondition.REFLECT in side_conditions)
            offset += 1
            feats[offset] = float(SideCondition.LIGHT_SCREEN in side_conditions)
            offset += 1
            feats[offset] = float(SideCondition.AURORA_VEIL in side_conditions)
            offset += 1
            # Tailwind
            feats[offset] = float(SideCondition.TAILWIND in side_conditions)
            offset += 1

        # Trick Room (global)
        feats[offset] = float(Field.TRICK_ROOM in battle.fields)
        offset += 1

        return feats


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
    # Showdown preview hides Rapid-Strike style and exposes generic Urshifu instead.
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
