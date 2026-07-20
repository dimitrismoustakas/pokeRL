"""Deterministic, role-balanced schedules for evaluation team pools."""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from collections.abc import Sequence

from ..team_pool import TeamPoolEntry


def balanced_distinct_team_pairs(
    entries: Sequence[TeamPoolEntry],
    *,
    games: int,
    seed: int,
) -> list[tuple[TeamPoolEntry, TeamPoolEntry]]:
    """Plan deterministic distinct pairs with balanced left/right exposure.

    A block contains one cyclic offset of a shuffled roster. Every team occurs
    exactly once on each side of a complete block, and every directed pairing
    occurs exactly once before a new epoch begins. A partial final block still
    uses every team at most once on either side, so role-count imbalance is at
    most one.
    """
    if games < 0:
        raise ValueError("games must be non-negative.")
    roster = list(entries)
    if games == 0:
        return []
    if len(roster) < 2:
        raise ValueError("Balanced distinct schedules require at least two teams.")

    rng = random.Random(seed)
    schedule: list[tuple[TeamPoolEntry, TeamPoolEntry]] = []
    while len(schedule) < games:
        order = list(roster)
        rng.shuffle(order)
        offsets = list(range(1, len(order)))
        rng.shuffle(offsets)
        rotation = rng.randrange(len(order))
        order = order[rotation:] + order[:rotation]
        for offset in offsets:
            block = [
                (entry, order[(index + offset) % len(order)])
                for index, entry in enumerate(order)
            ]
            remaining = games - len(schedule)
            schedule.extend(block[:remaining])
            if len(schedule) == games:
                break
    return schedule


def team_pair_schedule_summary(
    pairs: Sequence[tuple[TeamPoolEntry, TeamPoolEntry]],
) -> dict[str, object]:
    """Describe and fingerprint an exact ordered team-pair schedule."""
    left_counts = Counter(left.file for left, _ in pairs)
    right_counts = Counter(right.file for _, right in pairs)
    directed_counts = Counter((left.file, right.file) for left, right in pairs)
    payload = [
        {"left_team_file": left.file, "right_team_file": right.file}
        for left, right in pairs
    ]
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return {
        "schedule_sha256": hashlib.sha256(serialized).hexdigest(),
        "games": len(pairs),
        "distinct_directed_pairs": len(directed_counts),
        "max_directed_pair_repetitions": max(directed_counts.values(), default=0),
        "left_role_min_count": min(left_counts.values(), default=0),
        "left_role_max_count": max(left_counts.values(), default=0),
        "right_role_min_count": min(right_counts.values(), default=0),
        "right_role_max_count": max(right_counts.values(), default=0),
    }
