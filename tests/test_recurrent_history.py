import pytest
import torch
import torch.nn as nn

from pokerl.model.memory import IdentityMemory, LSTMMemory
from pokerl.training.recurrent_history import RecurrentObservationHistory


class _Spatial(nn.Module):
    def forward(self, obs):
        context = obs["context"].float()
        batch = context.shape[0]
        unused = context.new_zeros((batch, 1, 1))
        return context, unused, unused, unused, unused


class _Agent(nn.Module):
    def __init__(self, memory: nn.Module):
        super().__init__()
        self.spatial = _Spatial()
        self.memory = memory

    def initial_hidden(self, batch_size: int):
        return self.memory.initial_hidden(batch_size)


def _reference_hidden(agent: _Agent, sequences: list[torch.Tensor]):
    hidden = agent.initial_hidden(len(sequences))
    if hidden is None:
        return None
    max_length = max((sequence.shape[0] for sequence in sequences), default=0)
    if max_length == 0:
        return hidden
    padding = torch.zeros_like(
        next(sequence[0] for sequence in sequences if len(sequence))
    )
    for step in range(max_length):
        context = torch.stack(
            [
                sequence[step] if step < len(sequence) else padding
                for sequence in sequences
            ]
        )
        _, next_hidden = agent.memory(context, hidden)
        active = torch.tensor(
            [step < len(sequence) for sequence in sequences],
            dtype=torch.bool,
        ).view(1, len(sequences), 1)
        hidden = tuple(
            torch.where(active, next_state, state)
            for next_state, state in zip(next_hidden, hidden, strict=True)
        )
    return hidden


def test_history_keeps_only_each_ongoing_episode_tail_across_rollouts() -> None:
    history = RecurrentObservationHistory.empty(2)
    first_obs = {
        "context": torch.tensor(
            [
                [[1.0], [10.0]],
                [[2.0], [20.0]],
                [[3.0], [30.0]],
            ]
        )
    }
    first_dones = torch.tensor([[False, False], [True, False], [False, False]])

    history = history.extend(first_obs, first_dones)

    assert history.lengths.tolist() == [1, 3]
    assert history.obs["context"][:1, 0, 0].tolist() == [3.0]
    assert history.obs["context"][:3, 1, 0].tolist() == [10.0, 20.0, 30.0]

    second_obs = {
        "context": torch.tensor(
            [
                [[4.0], [40.0]],
                [[5.0], [50.0]],
            ]
        )
    }
    second_dones = torch.tensor([[False, False], [False, True]])

    history = history.extend(second_obs, second_dones)

    assert history.lengths.tolist() == [3, 0]
    assert history.obs["context"][:3, 0, 0].tolist() == [3.0, 4.0, 5.0]
    assert history.mean_length == 1.5
    assert history.max_length == 3


def test_recomputed_hidden_matches_full_tail_replay_after_weight_change() -> None:
    torch.manual_seed(4)
    agent = _Agent(LSTMMemory(input_dim=2, hidden_dim=3))
    history = RecurrentObservationHistory.empty(2).extend(
        {
            "context": torch.tensor(
                [
                    [[1.0, 0.5], [2.0, -0.5]],
                    [[1.5, 1.0], [3.0, 0.25]],
                    [[2.0, 1.5], [4.0, 0.75]],
                ]
            )
        },
        torch.tensor([[True, False], [False, False], [False, False]]),
    )
    with torch.no_grad():
        for parameter in agent.memory.parameters():
            parameter.add_(0.1)

    actual = history.recompute_hidden(agent)
    expected = _reference_hidden(
        agent,
        [
            torch.tensor([[1.5, 1.0], [2.0, 1.5]]),
            torch.tensor([[2.0, -0.5], [3.0, 0.25], [4.0, 0.75]]),
        ],
    )

    for actual_state, expected_state in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_state, expected_state)


def test_recomputed_hidden_handles_an_env_done_on_final_rollout_step() -> None:
    torch.manual_seed(8)
    agent = _Agent(LSTMMemory(input_dim=2, hidden_dim=3))
    history = RecurrentObservationHistory.empty(2).extend(
        {
            "context": torch.tensor(
                [
                    [[1.0, 0.5], [2.0, -0.5]],
                    [[1.5, 1.0], [3.0, 0.25]],
                ]
            )
        },
        torch.tensor([[False, False], [True, False]]),
    )

    actual = history.recompute_hidden(agent)
    expected = _reference_hidden(
        agent,
        [
            torch.empty((0, 2)),
            torch.tensor([[2.0, -0.5], [3.0, 0.25]]),
        ],
    )

    assert history.lengths.tolist() == [0, 2]
    for actual_state, expected_state in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_state, expected_state)


def test_identity_memory_needs_no_recurrent_history_replay() -> None:
    agent = _Agent(IdentityMemory(input_dim=2, hidden_dim=3))
    history = RecurrentObservationHistory.empty(1).extend(
        {"context": torch.ones((2, 1, 2))},
        torch.zeros((2, 1), dtype=torch.bool),
    )

    assert history.recompute_hidden(agent) is None


def test_history_rejects_environment_count_mismatch() -> None:
    history = RecurrentObservationHistory.empty(2)

    with pytest.raises(ValueError, match="History has 2 envs"):
        history.extend(
            {"context": torch.ones((2, 1, 2))},
            torch.zeros((2, 1), dtype=torch.bool),
        )
