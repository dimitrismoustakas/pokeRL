"""Game data: vocab lookups and optional stat-based embedding initialization."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import torch
import numpy as np
from poke_env.data import GenData


UNKNOWN_IDX = 0  # reserved for unknown / unobserved


class GameData:
    """Builds vocab mappings and stat tables from poke-env's gen data."""

    def __init__(self, gen: int = 9, item_vocab_size: int = 2048):
        self.gen = gen
        data = GenData.from_gen(gen)

        # Build vocabs: string id -> integer index (0 = unknown)
        self.species_to_idx = self._build_vocab(data.pokedex.keys())
        self.move_to_idx = self._build_vocab(data.moves.keys())
        self.abilities_to_idx = self._build_vocab(
            a for pokemon in data.pokedex.values()
            for a in (pokemon.get("abilities") or {}).values()
        )
        self.item_to_idx = self._build_vocab(self._known_item_ids(data.pokedex.values(), gen))
        self.type_to_idx = self._build_vocab(data.type_chart.keys())
        self._item_vocab_size = max(item_vocab_size, len(self.item_to_idx))
        self._item_hash_offset = len(self.item_to_idx)
        self._item_hash_buckets = self._item_vocab_size - self._item_hash_offset

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

    def item_index(self, item: str | None) -> int:
        if not item:
            return UNKNOWN_IDX

        item_id = str(item).lower()
        if item_id in self.item_to_idx:
            return self.item_to_idx[item_id]
        if self._item_hash_buckets <= 0:
            return UNKNOWN_IDX

        digest = hashlib.blake2b(item_id.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest, byteorder="big") % self._item_hash_buckets
        return self._item_hash_offset + bucket

    @staticmethod
    def _build_vocab(keys) -> dict[str, int]:
        vocab = {"<unk>": UNKNOWN_IDX}
        for key in sorted(set(str(k).lower() for k in keys)):
            if key not in vocab:
                vocab[key] = len(vocab)
        return vocab

    @classmethod
    def _known_item_ids(cls, pokedex_entries, gen: int) -> list[str]:
        items = {GenData.UNKNOWN_ITEM}
        for pokemon in pokedex_entries:
            required_item = pokemon.get("requiredItem")
            if required_item:
                items.add(required_item)

            required_items = pokemon.get("requiredItems") or []
            items.update(required_items)

        items.update(cls._load_showdown_item_ids(gen))
        return sorted(str(item).lower() for item in items if item)

    @classmethod
    def _load_showdown_item_ids(cls, gen: int) -> list[str]:
        item_export_path = cls._showdown_item_export_path(gen)
        node_script = (
            "const path = require('path');"
            "const exportPath = path.resolve(process.argv[1]);"
            "const mod = require(exportPath);"
            "const items = mod.Items || {};"
            "const ids = Object.keys(items).filter((id) => id && id !== 'noitem');"
            "process.stdout.write(JSON.stringify(ids));"
        )
        try:
            completed = subprocess.run(
                ["node", "-e", node_script, str(item_export_path)],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "node is required to build the item vocabulary from the local Pokemon Showdown install."
            ) from exc
        except subprocess.SubprocessError as exc:
            raise RuntimeError(
                f"Failed to load item ids from {item_export_path}. "
                "The local Pokemon Showdown install is required for named item embeddings."
            ) from exc

        try:
            item_ids = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Could not decode item ids from {item_export_path}. "
                "The local Pokemon Showdown install may be incomplete or out of date."
            ) from exc
        return [str(item_id).lower() for item_id in item_ids]

    @staticmethod
    def _showdown_item_export_path(gen: int) -> Path:
        env_override = os.environ.get("POKERL_SHOWDOWN_ITEM_EXPORT")
        if env_override:
            path = Path(env_override)
            if not path.exists():
                raise RuntimeError(f"POKERL_SHOWDOWN_ITEM_EXPORT points to a missing file: {path}")
            return path

        local_appdata = os.environ.get("LOCALAPPDATA")
        if not local_appdata:
            raise RuntimeError(
                "LOCALAPPDATA is not set, so the local Pokemon Showdown item export cannot be located."
            )

        base_path = Path(local_appdata) / "pokeRL" / "pokemon-showdown"
        candidates = [
            base_path / "dist" / "data" / "mods" / f"gen{gen}" / "items.js",
            base_path / "dist" / "data" / "items.js",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        raise RuntimeError(
            "Could not find Pokemon Showdown item data under the local install. "
            f"Checked: {', '.join(str(path) for path in candidates)}"
        )
