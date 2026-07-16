from contextlib import contextmanager

import numpy as np
import pytest
import torch

import scripts.train as train_module
from pokerl.data import MOVE_SCOPE_SCOPED
from pokerl.env.actions import INTENT_MOVE
from pokerl.env.observation import (
    EVENT_KIND_TO_INDEX,
    FIELD_TURN_OFFSET,
    NUM_PUBLIC_EVENT_FLAGS,
    NUM_PUBLIC_TARGET_POSITIONS,
    NUM_TARGET_OUTCOME_FEATURES,
    POSITION_IMPLICIT,
    POSITION_OWN_1,
)
from pokerl.public_consequence import (
    MOVE_BRANCH_SCOPED,
    NONMOVE_REASON_CANT,
)
from pokerl.training.rollout import RolloutBuffer
from poke_env.battle import PublicBattleEventKind


def _batched_obs(states: list[int], *, aux: bool) -> dict[str, np.ndarray]:
    obs: dict[str, np.ndarray] = {
        "state": np.asarray(states, dtype=np.float32).reshape(-1, 1),
    }
    if not aux:
        return obs

    for key in RolloutBuffer.AUX_LABEL_KEYS:
        if key == "opp_continuous":
            dtype = np.float32
        elif key == "opp_mask":
            dtype = np.bool_
        else:
            dtype = np.int64
        if key == "opp_mask":
            obs[key] = np.zeros((len(states), 1), dtype=dtype)
        else:
            obs[key] = np.asarray(states, dtype=dtype).reshape(-1, 1)
    obs[RolloutBuffer.AUX_IDENTITY_KEY] = np.ones(
        (len(states), 1),
        dtype=np.int64,
    )
    return obs


def _single_obs(state: int, *, aux_value: int | None = None) -> dict[str, np.ndarray]:
    obs = {"state": np.asarray([state], dtype=np.float32)}
    if aux_value is None:
        return obs

    for key in RolloutBuffer.AUX_LABEL_KEYS:
        if key == "opp_continuous":
            value = np.asarray([aux_value], dtype=np.float32)
        elif key == "opp_mask":
            value = np.zeros(1, dtype=np.bool_)
        else:
            value = np.asarray([aux_value], dtype=np.int64)
        obs[key] = value
    obs[RolloutBuffer.AUX_IDENTITY_KEY] = np.ones(1, dtype=np.int64)
    return obs


def _batched_masks(num_envs: int) -> dict[str, np.ndarray]:
    return {"legal": np.ones((num_envs, 2), dtype=np.bool_)}


class _DeterministicAgent:
    def __init__(self) -> None:
        self.hidden_inputs: dict[tuple[int, ...], list[float] | None] = {}

    def __call__(
        self,
        obs,
        hidden,
        *,
        action_masks,
        tera_exploration_prob,
    ):
        del (
            action_masks,
            tera_exploration_prob,
        )
        states = obs["state"].reshape(-1)
        state_key = tuple(int(value) for value in states.detach().cpu().tolist())
        self.hidden_inputs[state_key] = (
            None if hidden is None else hidden[0].reshape(-1).detach().cpu().tolist()
        )
        actions = torch.stack(
            (
                states.long(),
                states.long() + 1,
                torch.zeros_like(states, dtype=torch.long),
            ),
            dim=1,
        )
        next_hidden = None if hidden is None else tuple(state + 1 for state in hidden)
        return {
            "actions": actions,
            "log_prob": states.float() / 10.0,
            "value": states.float(),
            "hidden": next_hidden,
        }


class _ScriptedParallelEnv:
    num_envs = 4

    def __init__(self) -> None:
        self.states = [0, 10, 20, 30]
        self.steps = [0] * self.num_envs

    def step(self, actions):
        np.testing.assert_array_equal(
            actions[:, 0],
            np.asarray(self.states, dtype=np.int64),
        )
        next_observations = []
        infos = []
        rewards = np.empty(self.num_envs, dtype=np.float32)
        dones = np.empty(self.num_envs, dtype=bool)
        for env_index in range(self.num_envs):
            logical_step = self.steps[env_index]
            self.steps[env_index] += 1
            current_state = self.states[env_index]
            done = (env_index, logical_step) in {(0, 0), (3, 1)}
            info = {"env_index": env_index, "logical_step": logical_step}
            if done:
                label = 900 if env_index == 0 else 933
                reset_state = 100 if env_index == 0 else 300
                info["final_observation"] = _single_obs(
                    current_state + 1,
                    aux_value=label,
                )
                if env_index == 0:
                    info["opponent_aux_labels"] = _single_obs(
                        current_state + 1,
                        aux_value=990,
                    )
                info["episode_reward"] = float(label)
                next_state = reset_state
            else:
                next_state = current_state + 1

            self.states[env_index] = next_state
            next_observations.append(_single_obs(next_state, aux_value=next_state))
            infos.append(info)
            rewards[env_index] = float(100 * logical_step + env_index)
            dones[env_index] = done

        batched_obs = {
            key: np.stack([observation[key] for observation in next_observations])
            for key in next_observations[0]
        }
        return batched_obs, _batched_masks(self.num_envs), rewards, dones, infos


def test_collect_synchronous_rollout_prefers_private_exact_labels_and_preserves_sequences(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _ScriptedParallelEnv()
    obs = _batched_obs(env.states, aux=True)
    masks = _batched_masks(env.num_envs)
    buffer = RolloutBuffer(2, env.num_envs, obs, masks, torch.device("cpu"))
    hidden = (
        torch.tensor([[[10.0], [11.0], [12.0], [13.0]]]),
        torch.tensor([[[110.0], [111.0], [112.0], [113.0]]]),
    )
    buffer.reset(initial_hidden=hidden)
    agent = _DeterministicAgent()
    guard_entries = 0

    @contextmanager
    def recording_guard(device):
        nonlocal guard_entries
        assert device.type == "cpu"
        guard_entries += 1
        yield

    monkeypatch.setattr(train_module, "cuda_model_execution", recording_guard)

    final_obs, _final_masks, final_hidden, infos, profile = (
        train_module.collect_synchronous_rollout(
            parallel_env=env,
            agent=agent,
            buffer=buffer,
            obs=obs,
            action_masks=masks,
            hidden=hidden,
            device=torch.device("cpu"),
            num_steps=2,
            tera_exploration_prob=0.2,
            collect_timing=True,
        )
    )

    batch = buffer.compute_gae(
        torch.zeros(env.num_envs),
        gamma=0.0,
        gae_lambda=0.0,
    )
    assert torch.equal(
        batch.obs["state"].squeeze(-1),
        torch.tensor([[0.0, 10.0, 20.0, 30.0], [100.0, 11.0, 21.0, 31.0]]),
    )
    assert torch.equal(batch.actions[:, :, 0], batch.obs["state"].squeeze(-1).long())
    assert batch.dones.tolist() == [
        [True, False, False, False],
        [False, False, False, True],
    ]
    assert batch.aux_label_valid[:, 0].any(dim=-1).tolist() == [True, False]
    assert batch.aux_labels["opp_item"][0, 0, 0].item() == 990
    assert batch.aux_label_valid[:, 3].any(dim=-1).tolist() == [True, True]
    assert batch.aux_labels["opp_item"][:, 3, 0].tolist() == [933, 933]
    assert batch.tera_exploration_prob == pytest.approx(0.2)

    assert agent.hidden_inputs[(0, 10, 20, 30)] == [10.0, 11.0, 12.0, 13.0]
    assert agent.hidden_inputs[(100, 11, 21, 31)] == [0.0, 12.0, 13.0, 14.0]
    assert final_hidden is not None
    assert final_hidden[0].reshape(-1).tolist() == [1.0, 13.0, 14.0, 0.0]
    assert final_hidden[1].reshape(-1).tolist() == [1.0, 113.0, 114.0, 0.0]
    assert final_obs["state"].reshape(-1).tolist() == [101.0, 12.0, 22.0, 300.0]
    assert len(infos) == 8
    assert all("final_observation" not in info for info in infos)
    assert all("opponent_aux_labels" not in info for info in infos)
    assert profile["exact_aux_label_episodes"] == 1.0
    assert profile["env_step_sec"] >= 0.0
    assert guard_entries == 2


class _MissingFinalObservationEnv:
    num_envs = 1

    def step(self, actions):
        del actions
        return (
            _batched_obs([100], aux=True),
            _batched_masks(1),
            np.asarray([1.0], dtype=np.float32),
            np.asarray([True]),
            [{}],
        )


def test_collect_synchronous_rollout_requires_terminal_final_observation() -> None:
    obs = _batched_obs([0], aux=True)
    masks = _batched_masks(1)
    buffer = RolloutBuffer(1, 1, obs, masks, torch.device("cpu"))
    buffer.reset()

    with pytest.raises(RuntimeError, match="omitted final_observation"):
        train_module.collect_synchronous_rollout(
            parallel_env=_MissingFinalObservationEnv(),
            agent=_DeterministicAgent(),
            buffer=buffer,
            obs=obs,
            action_masks=masks,
            hidden=None,
            device=torch.device("cpu"),
            num_steps=1,
            tera_exploration_prob=0.0,
            collect_timing=False,
        )

    assert buffer.step == 0


def _consequence_obs(
    state: int,
    *,
    field_value: float,
    cant: bool,
) -> dict[str, np.ndarray]:
    event_kind = EVENT_KIND_TO_INDEX[
        PublicBattleEventKind.CANT if cant else PublicBattleEventKind.MOVE
    ]
    return {
        "state": np.asarray([state], dtype=np.float32),
        "field": np.full(FIELD_TURN_OFFSET + 1, field_value, dtype=np.float32),
        "recent_event_meta": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "recent_event_kind": np.asarray([event_kind], dtype=np.int64),
        "recent_event_actor": np.asarray([POSITION_OWN_1], dtype=np.int64),
        "recent_event_move": np.asarray([0 if cant else 7], dtype=np.int64),
        "recent_event_source_move": np.zeros(1, dtype=np.int64),
        "recent_event_move_scope": np.asarray(
            [0 if cant else MOVE_SCOPE_SCOPED], dtype=np.int64
        ),
        "recent_event_flags": np.zeros((1, NUM_PUBLIC_EVENT_FLAGS), dtype=np.float32),
        "recent_event_target_outcomes": np.zeros(
            (1, NUM_PUBLIC_TARGET_POSITIONS, NUM_TARGET_OUTCOME_FEATURES),
            dtype=np.float32,
        ),
        "recent_event_target_activation_move": np.zeros(
            (1, NUM_PUBLIC_TARGET_POSITIONS), dtype=np.int64
        ),
        "recent_event_mask": np.asarray([False]),
        "prev_own_valid": np.asarray([True, False]),
        "prev_own_kind": np.asarray([INTENT_MOVE, 0], dtype=np.int64),
        "prev_own_move": np.asarray([7, 0], dtype=np.int64),
        "prev_own_target": np.asarray([POSITION_IMPLICIT, 0], dtype=np.int64),
    }


def _batch_single_observation(
    observation: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    return {key: value[None, ...] for key, value in observation.items()}


class _ConsequenceAlignmentEnv:
    num_envs = 1

    def __init__(self) -> None:
        self.step_index = 0

    def step(self, actions):
        del actions
        if self.step_index == 0:
            observation = _consequence_obs(1, field_value=10.0, cant=False)
            done = False
            info = {}
        else:
            # Auto-reset returns this unrelated observation.  The transition
            # label must instead use the terminal public observation below.
            observation = _consequence_obs(100, field_value=99.0, cant=False)
            done = True
            info = {
                "final_observation": _consequence_obs(
                    2,
                    field_value=20.0,
                    cant=True,
                )
            }
        self.step_index += 1
        return (
            _batch_single_observation(observation),
            _batched_masks(1),
            np.asarray([0.0], dtype=np.float32),
            np.asarray([done]),
            [info],
        )


def test_public_consequences_align_next_and_terminal_observations_through_last_step() -> (
    None
):
    initial = _batch_single_observation(
        _consequence_obs(0, field_value=0.0, cant=False)
    )
    masks = _batched_masks(1)
    buffer = RolloutBuffer(2, 1, initial, masks, torch.device("cpu"))
    buffer.reset()

    train_module.collect_synchronous_rollout(
        parallel_env=_ConsequenceAlignmentEnv(),
        agent=_DeterministicAgent(),
        buffer=buffer,
        obs=initial,
        action_masks=masks,
        hidden=None,
        device=torch.device("cpu"),
        num_steps=2,
        tera_exploration_prob=0.0,
        collect_timing=False,
    )
    batch = buffer.compute_gae(torch.zeros(1))

    assert batch.public_consequence_move_emitted_targets[:, 0, 0].tolist() == [
        1.0,
        0.0,
    ]
    assert batch.public_consequence_move_emitted_masks[:, 0, 0].tolist() == [
        True,
        True,
    ]
    assert batch.public_consequence_move_branch_labels[0, 0, 0] == MOVE_BRANCH_SCOPED
    assert batch.public_consequence_nonmove_reason_labels[1, 0, 0] == (
        NONMOVE_REASON_CANT
    )
    assert batch.public_consequence_field_targets[:, 0, 0, 0].tolist() == [
        10.0,
        20.0,
    ]
    # Terminal Showdown observations do not advance the turn counter, so their
    # field state is not a valid target for the one-turn persistence baseline.
    assert batch.public_consequence_field_masks[:, 0, 0].tolist() == [True, False]


class _FailingParallelEnv:
    num_envs = 1

    def step(self, actions):
        del actions
        raise RuntimeError("environment step failed")


def test_collect_synchronous_rollout_propagates_step_failure() -> None:
    obs = _batched_obs([0], aux=False)
    masks = _batched_masks(1)
    buffer = RolloutBuffer(1, 1, obs, masks, torch.device("cpu"))
    buffer.reset()

    with pytest.raises(RuntimeError, match="environment step failed"):
        train_module.collect_synchronous_rollout(
            parallel_env=_FailingParallelEnv(),
            agent=_DeterministicAgent(),
            buffer=buffer,
            obs=obs,
            action_masks=masks,
            hidden=None,
            device=torch.device("cpu"),
            num_steps=1,
            tera_exploration_prob=0.0,
            collect_timing=False,
        )

    assert buffer.step == 0
