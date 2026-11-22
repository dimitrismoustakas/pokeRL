"""Utilities for loading and sampling fixed team pools."""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from pikapro import pack, to_id


VALID_TEAM_POOL_SUBSETS = {"curated", "generated", "pool"}


@dataclass(frozen=True)
class TeamPoolEntry:
    title: str
    file: str
    packed_team: str

    @property
    def species(self) -> tuple[str, ...]:
        return tuple(
            to_id(mon[1] or mon[0])
            for mon in (slot.split("|") for slot in self.packed_team.split("]"))
        )


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

    def rng_state(self) -> tuple:
        """Return the private sampling state for an exact training resume."""
        with self._lock:
            return self._rng.getstate()

    def restore_rng_state(self, state: tuple) -> None:
        """Restore a previously captured private sampling state."""
        with self._lock:
            self._rng.setstate(state)


def default_team_pool_manifest_path(repo_root: Path, format_id: str) -> Path:
    return repo_root / "data" / "team_pools" / format_id / "manifest.json"


def pack_team(team_text: str) -> str:
    """Pack a Regulation F roster with all battle stats calculated at level 50."""
    slots = []
    for slot in pack(team_text, "gen9vgc").split("]"):
        fields = slot.split("|")
        fields[10] = "50"
        slots.append("|".join(fields))
    return "]".join(slots)


def load_team_pool(
    manifest_path: str | Path,
    *,
    subset: str = "pool",
    seed: int | None = None,
    usage: str = "training",
) -> TeamPool:
    if subset not in VALID_TEAM_POOL_SUBSETS:
        raise ValueError(
            f"Unsupported team pool subset {subset!r}; expected one of {sorted(VALID_TEAM_POOL_SUBSETS)}."
        )

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if usage not in {"training", "evaluation"}:
        raise ValueError(
            f"Unsupported team-pool usage {usage!r}; expected 'training' or 'evaluation'."
        )
    if usage == "training" and manifest["purpose"] != "training":
        raise ValueError(
            f"Manifest {manifest_path} is evaluation-only and cannot be used for training."
        )
    root = repo_root(manifest_path.parent)
    format_id = manifest["format"]
    entries: list[TeamPoolEntry] = []
    for raw_entry in manifest[subset]:
        file_value = raw_entry["file"]
        team_path = root / file_value
        team_text = team_path.read_text(encoding="utf-8")
        entries.append(
            TeamPoolEntry(
                title=str(raw_entry.get("title") or team_path.stem),
                file=file_value,
                packed_team=pack_team(team_text),
            )
        )

    return TeamPool(format_id=format_id, subset=subset, entries=entries, seed=seed)


def repo_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"Could not find repo root above {start}")
