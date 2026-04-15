"""Converts a poke-env DoubleBattle into the tensor dict the model expects."""

import numpy as np
from poke_env.environment import DoubleBattle, Pokemon, Move, Status, SideCondition, Weather, Field

from ..data import GameData, UNKNOWN_IDX


# Volatile statuses we track (binary flags)
VOLATILES = [
    "confusion", "substitute", "leechseed", "taunt", "encore", "torment",
    "disable", "yawn", "perishsong", "protect", "endure", "focusenergy",
    "attract", "curse", "nightmare", "flashfire",
]
NUM_VOLATILES = len(VOLATILES)

# Status conditions for one-hot encoding
STATUS_LIST = [None, Status.BRN, Status.FRZ, Status.PAR, Status.PSN, Status.SLP, Status.TOK]

# Role indices
ROLE_ACTIVE_1 = 0
ROLE_ACTIVE_2 = 1
ROLE_BENCH = 2


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
        types = np.zeros((ts, 2), dtype=np.int64)
        continuous = np.zeros((ts, 35), dtype=np.float32)
        roles = np.full(ts, ROLE_BENCH, dtype=np.int64)
        mask = np.ones(ts, dtype=bool)  # True = padding

        for i, pokemon in enumerate(pokemon_list):
            if pokemon is None or i >= ts:
                break
            mask[i] = False
            species[i] = self._species_idx(pokemon)
            moves[i] = self._move_indices(pokemon, full_info)
            ability[i] = self._ability_idx(pokemon, full_info)
            item[i] = self._item_idx(pokemon, full_info)
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

    def _move_indices(self, pokemon: Pokemon, full_info: bool) -> np.ndarray:
        indices = np.zeros(4, dtype=np.int64)
        known_moves = list(pokemon.moves.values()) if pokemon.moves else []
        for j, move in enumerate(known_moves[:4]):
            key = move.id.lower() if move.id else "<unk>"
            indices[j] = self.gd.move_to_idx.get(key, UNKNOWN_IDX)
        return indices

    def _ability_idx(self, pokemon: Pokemon, full_info: bool) -> int:
        ab = pokemon.ability
        if ab is None:
            return UNKNOWN_IDX
        return self.gd.abilities_to_idx.get(ab.lower(), UNKNOWN_IDX)

    def _item_idx(self, pokemon: Pokemon, full_info: bool) -> int:
        it = pokemon.item
        if it is None or it == "":
            return UNKNOWN_IDX
        return self.gd.item_to_idx.get(it.lower(), UNKNOWN_IDX)

    def _type_indices(self, pokemon: Pokemon) -> np.ndarray:
        indices = np.zeros(2, dtype=np.int64)
        if pokemon.type_1:
            indices[0] = self.gd.type_to_idx.get(pokemon.type_1.name.lower(), UNKNOWN_IDX)
        if pokemon.type_2:
            indices[1] = self.gd.type_to_idx.get(pokemon.type_2.name.lower(), UNKNOWN_IDX)
        return indices

    def _continuous_features(self, pokemon: Pokemon, full_info: bool) -> np.ndarray:
        feats = np.zeros(35, dtype=np.float32)
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

        # Volatile statuses (16 binary)
        effects = pokemon.effects if hasattr(pokemon, "effects") else {}
        for j, vol in enumerate(VOLATILES):
            if vol in effects:
                feats[offset + j] = 1.0
        offset += NUM_VOLATILES

        # Move PP (4), normalized by max PP, only for full info
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
        """Find where a specific pokemon is in the team list."""
        if pokemon is None:
            return 0
        for i, p in enumerate(team_list):
            if p is pokemon:
                return i
        return 0

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

        # Terrain (5: 4 types + none)
        terrain_types = [
            Field.ELECTRIC_TERRAIN, Field.GRASSY_TERRAIN,
            Field.MISTY_TERRAIN, Field.PSYCHIC_TERRAIN,
        ]
        if not battle.fields:
            feats[offset] = 1.0
        else:
            for j, t in enumerate(terrain_types):
                if t in battle.fields:
                    feats[offset + 1 + j] = 1.0
                    break
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
        feats[offset] = float(Field.TRICK_ROOM in battle.fields) if battle.fields else 0.0
        offset += 1

        return feats
