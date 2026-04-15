"""Game data: vocab lookups and optional stat-based embedding initialization."""

import torch
import numpy as np
from poke_env.data import GenData


UNKNOWN_IDX = 0  # reserved for unknown / unobserved


class GameData:
    """Builds vocab mappings and stat tables from poke-env's gen data."""

    def __init__(self, gen: int = 9):
        data = GenData.from_gen(gen)

        # Build vocabs: string id -> integer index (0 = unknown)
        self.species_to_idx = self._build_vocab(data.pokedex.keys())
        self.move_to_idx = self._build_vocab(data.moves.keys())
        self.abilities_to_idx = self._build_vocab(
            a for pokemon in data.pokedex.values()
            for a in (pokemon.get("abilities") or {}).values()
        )
        self.item_to_idx = self._build_vocab(
            item for item in data.items.keys() if item  # skip empty
        )
        self.type_to_idx = self._build_vocab(data.type_chart.keys())

        # Store raw data for stat init
        self._pokedex = data.pokedex
        self._moves = data.moves

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
        return len(self.item_to_idx)

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
            mat[idx, 1] = (entry.get("accuracy", 100) or 100) / 100.0
            mat[idx, 2] = (entry.get("pp", 0) or 0) / 64.0
        return mat

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
