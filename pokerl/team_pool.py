"""Utilities for loading and sampling fixed team pools."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder


VALID_TEAM_POOL_SUBSETS = {"curated", "generated", "pool"}


@dataclass(frozen=True)
class TeamPoolEntry:
    title: str
    file: str
    packed_team: str


class TeamPool:
    def __init__(
        self,
        *,
        format_id: str,
        subset: str,
        entries: list[TeamPoolEntry],
        seed: int | None = None,
    ):
        if not entries:
            raise ValueError("TeamPool requires at least one team.")

        self.format_id = format_id
        self.subset = subset
        self.entries = tuple(entries)
        self._rng = random.Random(seed)
        self._lock = Lock()

    @property
    def size(self) -> int:
        return len(self.entries)

    def describe(self) -> dict[str, str | int]:
        return {
            "format": self.format_id,
            "subset": self.subset,
            "size": self.size,
        }

    def sample_pair(self, distinct: bool = True) -> tuple[TeamPoolEntry, TeamPoolEntry]:
        with self._lock:
            if distinct and len(self.entries) > 1:
                first, second = self._rng.sample(self.entries, k=2)
            else:
                first = self._rng.choice(self.entries)
                second = self._rng.choice(self.entries)
        return first, second

    def sample_packed_pair(self, distinct: bool = True) -> tuple[str, str]:
        first, second = self.sample_pair(distinct=distinct)
        return first.packed_team, second.packed_team


def default_team_pool_manifest_path(repo_root: Path, format_id: str) -> Path:
    return repo_root / "data" / "team_pools" / format_id / "manifest.json"


def load_team_pool(
    manifest_path: str | Path,
    *,
    subset: str = "pool",
    seed: int | None = None,
) -> TeamPool:
    if subset not in VALID_TEAM_POOL_SUBSETS:
        raise ValueError(
            f"Unsupported team pool subset {subset!r}; expected one of {sorted(VALID_TEAM_POOL_SUBSETS)}."
        )

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _load_manifest_json(manifest_path)
    repo_root = _find_repo_root(manifest_path.parent)
    raw_entries = manifest.get(subset)
    if not isinstance(raw_entries, list):
        raise ValueError(f"Manifest {manifest_path} is missing a list-valued {subset!r} section.")

    entries: list[TeamPoolEntry] = []
    for raw_entry in raw_entries:
        file_value = raw_entry.get("file")
        if not isinstance(file_value, str) or not file_value:
            raise ValueError(f"Manifest entry is missing a valid file path: {raw_entry!r}")

        team_path = Path(file_value)
        if not team_path.is_absolute():
            team_path = repo_root / team_path
        team_text = team_path.read_text(encoding="utf-8")
        packed_team = ConstantTeambuilder(team_text).yield_team()
        entries.append(
            TeamPoolEntry(
                title=str(raw_entry.get("title") or team_path.stem),
                file=file_value,
                packed_team=packed_team,
            )
        )

    format_id = str(manifest.get("format") or manifest_path.parent.name)
    return TeamPool(format_id=format_id, subset=subset, entries=entries, seed=seed)


def _load_manifest_json(path: Path) -> dict:
    def reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        duplicates = []
        for key, value in pairs:
            if key in result:
                duplicates.append(key)
            result[key] = value
        if duplicates:
            duplicate_list = ", ".join(sorted(set(duplicates)))
            raise ValueError(f"Manifest {path} contains duplicate JSON key(s): {duplicate_list}")
        return result

    loaded = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_keys)
    if not isinstance(loaded, dict):
        raise ValueError(f"Manifest {path} must contain a JSON object.")
    return loaded


def _find_repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root above {start}")
