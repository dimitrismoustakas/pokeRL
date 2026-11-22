"""Game data: vocab lookups and optional stat-based embedding initialization."""

import json
from importlib.resources import files

import torch
import numpy as np
from pikapro import to_id


UNKNOWN_IDX = 0  # reserved for unknown / unobserved

MOVE_SCOPE_UNKNOWN = 0
MOVE_SCOPE_SCOPED = 1
MOVE_SCOPE_TARGETED = 2
NUM_MOVE_SCOPES = 3
MOVE_MECHANICS_DIM = 18

_SCOPED_MOVE_TARGETS = frozenset(
    {"self", "allyside", "allies", "allyteam", "foeside", "all", "scripted"}
)
_TARGETED_MOVE_TARGETS = frozenset(
    {
        "normal",
        "adjacentfoe",
        "any",
        "randomnormal",
        "adjacentally",
        "adjacentallyorself",
        "alladjacentfoes",
        "alladjacent",
    }
)


class GameData:
    """Gen 9 species, moves and mechanics used by the policy."""

    def __init__(self, gen: int = 9, item_vocab_size: int = 2048):
        self.gen = gen
        data = json.loads(
            files("pokerl").joinpath(f"gen{gen}_dex.json").read_text(encoding="utf-8")
        )

        # Build vocabs: string id -> integer index (0 = unknown)
        self.species_to_idx = self._build_vocab(data["species"].keys())
        self.move_to_idx = self._build_vocab(data["moves"].keys())
        # Checkpoints number abilities by dex display name. Battle requests use
        # Showdown IDs, so index those same rows by the ID the tracker returns.
        ability_labels = self._build_vocab(
            a
            for pokemon in data["species"].values()
            for a in (pokemon.get("abilities") or {}).values()
        )
        self.abilities_to_idx = {
            to_id(name) if idx != UNKNOWN_IDX else name: idx
            for name, idx in ability_labels.items()
        }
        self.item_to_idx = self._build_vocab(data["items"])
        self.type_to_idx = self._build_vocab(data["type_chart"].keys())
        if gen >= 9:
            # Stellar is a Tera type, absent from the ordinary defensive type chart.
            self.type_to_idx.setdefault("stellar", len(self.type_to_idx))
        self._item_vocab_size = max(item_vocab_size, len(self.item_to_idx))

        # Store raw data for stat init
        self._pokedex = data["species"]
        self._moves = data["moves"]
        self._type_chart = data["type_chart"]

    @property
    def num_species(self) -> int:
        return len(self.species_to_idx)

    @property
    def num_moves(self) -> int:
        return len(self.move_to_idx)

    @property
    def num_abilities(self) -> int:
        return len(self.abilities_to_idx)

    @property
    def num_items(self) -> int:
        return self._item_vocab_size

    @property
    def num_types(self) -> int:
        return len(self.type_to_idx)

    def species_stats_matrix(self) -> np.ndarray:
        """Base stats matrix: (num_species, 6) with [HP, Atk, Def, SpA, SpD, Spe].

        Row 0 (UNKNOWN) is zeros. Stats are normalized to [0, 1] by dividing by 255.
        """
        mat = np.zeros((self.num_species, 6), dtype=np.float32)
        stat_keys = ["hp", "atk", "def", "spa", "spd", "spe"]
        for species_id, idx in self.species_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            entry = self._pokedex.get(species_id, {})
            base = entry.get("baseStats", {})
            for i, key in enumerate(stat_keys):
                mat[idx, i] = base.get(key, 0) / 255.0
        return mat

    def move_stats_matrix(self) -> np.ndarray:
        """Move stats matrix: (num_moves, 3) with [base_power/250, accuracy/100, pp/64]."""
        mat = np.zeros((self.num_moves, 3), dtype=np.float32)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            entry = self._moves.get(move_id, {})
            mat[idx, 0] = (entry.get("basePower", 0) or 0) / 250.0
            # Showdown represents moves which bypass the accuracy check with
            # the boolean ``true``.  Python's ``bool`` is an ``int`` subclass,
            # so dividing it by 100 silently encoded always-hit moves as 0.01.
            accuracy = entry.get("accuracy", 100)
            mat[idx, 1] = 1.0 if accuracy is True else (accuracy or 100) / 100.0
            mat[idx, 2] = (entry.get("pp", 0) or 0) / 64.0
        return mat

    def move_type_indices(self) -> np.ndarray:
        """Move type index lookup aligned with the move vocabulary."""
        indices = np.zeros(self.num_moves, dtype=np.int64)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            move_type = str(self._moves.get(move_id, {}).get("type", "")).lower()
            indices[idx] = self.type_to_idx.get(move_type, UNKNOWN_IDX)
        return indices

    def move_category_indices(self) -> np.ndarray:
        """Move category lookup: 0=status, 1=physical, 2=special."""
        category_to_idx = {
            "status": 0,
            "physical": 1,
            "special": 2,
        }
        indices = np.zeros(self.num_moves, dtype=np.int64)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            category = str(
                self._moves.get(move_id, {}).get("category", "Status")
            ).lower()
            indices[idx] = category_to_idx.get(category, 0)
        return indices

    def move_priority_vector(self) -> np.ndarray:
        """Move priority lookup normalized into roughly [-1, 1]."""
        values = np.zeros(self.num_moves, dtype=np.float32)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            priority = float(self._moves.get(move_id, {}).get("priority", 0) or 0)
            values[idx] = np.clip(priority / 7.0, -1.0, 1.0)
        return values

    def move_bypasses_accuracy_vector(self) -> np.ndarray:
        """Return whether each move bypasses accuracy/evasion checks.

        Showdown represents this mechanic with boolean ``true``. Numeric 100
        remains subject to accuracy and evasion stages, so the two cases need
        distinct model features even though both have base accuracy 1.0.
        """

        values = np.zeros(self.num_moves, dtype=np.float32)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            values[idx] = float(self._moves.get(move_id, {}).get("accuracy") is True)
        return values

    def move_ignores_evasion_vector(self) -> np.ndarray:
        """Return whether each move ignores only the target's evasion stage."""

        values = np.zeros(self.num_moves, dtype=np.float32)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            values[idx] = float(
                bool(self._moves.get(move_id, {}).get("ignoreEvasion", False))
            )
        return values

    def move_target_scope_vector(self) -> np.ndarray:
        """Classify whether a move resolves at public targets or at a scope."""

        values = np.full(self.num_moves, MOVE_SCOPE_UNKNOWN, dtype=np.int64)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            target = str(self._moves.get(move_id, {}).get("target", "")).lower()
            if target in _SCOPED_MOVE_TARGETS:
                values[idx] = MOVE_SCOPE_SCOPED
            elif target in _TARGETED_MOVE_TARGETS:
                values[idx] = MOVE_SCOPE_TARGETED
        return values

    def move_multihit_vector(self) -> np.ndarray:
        """Return whether each move can emit a public multi-hit count."""

        values = np.zeros(self.num_moves, dtype=np.float32)
        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            values[idx] = float(bool(self._moves.get(move_id, {}).get("multihit")))
        return values

    def move_mechanics_matrix(self) -> np.ndarray:
        """Return generic public move-effect metadata aligned with move ids.

        These features describe simulator mechanics, not strategic value.  In
        particular they let the action scorer distinguish effects such as a
        protection/stalling move, field creation, switching, healing, and
        secondary effects without asking a compressed four-move Pokemon vector
        to preserve every bit.  Move identity remains available separately via
        the shared learned move embedding.

        Columns are:
        stalling/protection move, self switch, forced target switch, healing,
        drain fraction, recoil fraction, target boosts, self boosts, status,
        volatile status, side condition, slot condition, pseudo-weather,
        weather, terrain, secondary-effect probability, two-turn move, recharge.
        """

        values = np.zeros((self.num_moves, MOVE_MECHANICS_DIM), dtype=np.float32)

        def fraction(raw: object) -> float:
            if not isinstance(raw, (list, tuple)) or len(raw) != 2:
                return 0.0
            numerator, denominator = raw
            if not isinstance(numerator, (int, float)) or not isinstance(
                denominator, (int, float)
            ):
                return 0.0
            if denominator == 0:
                return 0.0
            return float(np.clip(float(numerator) / float(denominator), 0.0, 1.0))

        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            move = self._moves.get(move_id, {})
            secondary = move.get("secondary")
            secondaries = move.get("secondaries")
            secondary_chances: list[float] = []
            if isinstance(secondary, dict):
                secondary_chances.append(float(secondary.get("chance", 100) or 0))
            if isinstance(secondaries, list):
                secondary_chances.extend(
                    float(effect.get("chance", 100) or 0)
                    for effect in secondaries
                    if isinstance(effect, dict)
                )
            self_effect = move.get("self")
            values[idx] = np.asarray(
                [
                    bool(move.get("stallingMove")),
                    bool(move.get("selfSwitch")),
                    bool(move.get("forceSwitch")),
                    bool(move.get("heal")),
                    fraction(move.get("drain")),
                    fraction(move.get("recoil")),
                    bool(move.get("boosts")),
                    bool(isinstance(self_effect, dict) and self_effect.get("boosts")),
                    bool(move.get("status")),
                    bool(move.get("volatileStatus")),
                    bool(move.get("sideCondition")),
                    bool(move.get("slotCondition")),
                    bool(move.get("pseudoWeather")),
                    bool(move.get("weather")),
                    bool(move.get("terrain")),
                    max(secondary_chances, default=0.0) / 100.0,
                    bool(move.get("flags", {}).get("charge")),
                    bool(move.get("flags", {}).get("recharge"))
                    or bool(
                        isinstance(self_effect, dict)
                        and self_effect.get("volatileStatus") == "mustrecharge"
                    ),
                ],
                dtype=np.float32,
            )
        return values

    def move_target_flags(self) -> np.ndarray:
        """Move target class flags aligned with the move vocabulary.

        Columns are:
        0. no explicit target hits foes
        1. no explicit target hits all active pokemon
        2. no explicit target hits allies / own side
        3. no explicit target hits self
        4. can damage allies
        5. can hit multiple foes
        """
        flags = np.zeros((self.num_moves, 6), dtype=np.float32)
        foe_only_targets = {"alladjacentfoes", "foeside"}
        all_active_targets = {"alladjacent", "all"}
        ally_targets = {"allies", "allyside", "allyteam"}
        self_targets = {"self"}

        for move_id, idx in self.move_to_idx.items():
            if idx == UNKNOWN_IDX:
                continue
            target = str(self._moves.get(move_id, {}).get("target", "")).lower()
            if target in foe_only_targets:
                flags[idx, 0] = 1.0
                flags[idx, 5] = 1.0
            elif target in all_active_targets:
                flags[idx, 1] = 1.0
                flags[idx, 4] = 1.0
                flags[idx, 5] = 1.0
            elif target in ally_targets:
                flags[idx, 2] = 1.0
            elif target in self_targets:
                flags[idx, 3] = 1.0
        return flags

    def type_effectiveness_matrix(self) -> np.ndarray:
        """Attack-type x defend-type effectiveness lookup.

        Unknown rows/columns stay neutral at 1.0.
        """
        matrix = np.ones((self.num_types, self.num_types), dtype=np.float32)
        for defend_type, attack_map in self._type_chart.items():
            defend_idx = self.type_to_idx.get(str(defend_type).lower(), UNKNOWN_IDX)
            if defend_idx == UNKNOWN_IDX:
                continue
            for attack_type, multiplier in attack_map.items():
                attack_idx = self.type_to_idx.get(str(attack_type).lower(), UNKNOWN_IDX)
                if attack_idx == UNKNOWN_IDX:
                    continue
                matrix[attack_idx, defend_idx] = float(multiplier)
        return matrix

    def init_species_embedding(self, embedding: torch.nn.Embedding):
        """Initialize species embedding with a linear projection of base stats."""
        stats = torch.from_numpy(self.species_stats_matrix())
        proj = torch.nn.Linear(6, embedding.embedding_dim, bias=False)
        with torch.no_grad():
            embedding.weight.copy_(proj(stats))

    def init_move_embedding(self, embedding: torch.nn.Embedding):
        """Initialize move embedding with a linear projection of move stats."""
        stats = torch.from_numpy(self.move_stats_matrix())
        proj = torch.nn.Linear(3, embedding.embedding_dim, bias=False)
        with torch.no_grad():
            embedding.weight.copy_(proj(stats))

    @staticmethod
    def _build_vocab(keys) -> dict[str, int]:
        vocab = {"<unk>": UNKNOWN_IDX}
        for key in sorted(set(str(k).lower() for k in keys)):
            if key not in vocab:
                vocab[key] = len(vocab)
        return vocab
