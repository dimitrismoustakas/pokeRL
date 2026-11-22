"""Deterministic, role-balanced schedules for evaluation team pools."""

from __future__ import annotations

import random
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
