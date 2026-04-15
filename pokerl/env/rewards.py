"""Reward functions for Pokemon battles."""

from poke_env.environment import DoubleBattle


def sparse_reward(battle: DoubleBattle) -> float:
    """Pure win/loss reward. +1 for win, -1 for loss, 0 during battle."""
    if battle.won:
        return 1.0
    elif battle.lost:
        return -1.0
    return 0.0
