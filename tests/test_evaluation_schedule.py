from collections import Counter

import pytest

from pokerl.eval.schedule import (
    balanced_distinct_team_pairs,
    team_pair_schedule_summary,
)
from pokerl.team_pool import TeamPoolEntry


def _entries(count: int) -> list[TeamPoolEntry]:
    return [
        TeamPoolEntry(
            title=f"team-{index}", file=f"{index}.txt", packed_team=str(index)
        )
        for index in range(count)
    ]


def test_balanced_schedule_is_deterministic_distinct_and_role_balanced() -> None:
    entries = _entries(7)
    pairs = balanced_distinct_team_pairs(entries, games=31, seed=17)

    assert pairs == balanced_distinct_team_pairs(entries, games=31, seed=17)
    assert pairs != balanced_distinct_team_pairs(entries, games=31, seed=18)
    assert all(left != right for left, right in pairs)
    left_counts = Counter(left.file for left, _ in pairs)
    right_counts = Counter(right.file for _, right in pairs)
    assert max(left_counts.values()) - min(left_counts.values()) <= 1
    assert max(right_counts.values()) - min(right_counts.values()) <= 1


def test_balanced_schedule_covers_all_directed_pairs_before_repeating() -> None:
    entries = _entries(5)
    pairs = balanced_distinct_team_pairs(entries, games=20, seed=3)

    assert len({(left.file, right.file) for left, right in pairs}) == 20
    summary = team_pair_schedule_summary(pairs)
    assert summary["distinct_directed_pairs"] == 20
    assert summary["max_directed_pair_repetitions"] == 1
    assert summary["left_role_min_count"] == 4
    assert summary["left_role_max_count"] == 4
    assert len(summary["schedule_sha256"]) == 64


def test_balanced_schedule_validates_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        balanced_distinct_team_pairs(_entries(2), games=-1, seed=0)
    with pytest.raises(ValueError, match="at least two"):
        balanced_distinct_team_pairs(_entries(1), games=1, seed=0)
    assert balanced_distinct_team_pairs(_entries(1), games=0, seed=0) == []
