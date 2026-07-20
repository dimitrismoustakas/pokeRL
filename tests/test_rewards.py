from types import SimpleNamespace

from pokerl.env.rewards import sparse_reward


def test_sparse_reward_is_win_loss_only() -> None:
    assert sparse_reward(SimpleNamespace(won=True, lost=False)) == 1.0
    assert sparse_reward(SimpleNamespace(won=False, lost=True)) == -1.0
    assert sparse_reward(SimpleNamespace(won=False, lost=False)) == 0.0
