import asyncio
import copy
from concurrent.futures import Future
from threading import Thread
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

import pokerl.env.actions as actions_module
import pokerl.env.showdown_env as showdown_env_module
import pokerl.training.league as league_module
from pokerl.config import TrainingConfig
from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    PASS_ACTION,
    SWITCH_OFFSET,
    TERA_NONE,
    TERA_SLOT_1,
    TERA_SLOT_2,
    TARGET_NONE,
    TARGET_SELF,
    action_to_order,
    build_action_mask,
    build_legal_action_state,
    build_joint_action_mask,
    build_tera_action_mask,
    is_default_only_action_state,
)
from pokerl.env.showdown_env import BattleEnv, ParallelBattleEnv, ShowdownPlayer
from pokerl.env.observation import (
    CONTINUOUS_DIM,
    FIELD_TIMER_CAPS,
    FIELD_TURN_OFFSET,
    ROLE_ACTIVE_1,
    ROLE_ACTIVE_2,
    ROLE_BENCH,
)
from pokerl.model.heads import (
    ActionHead,
    MaskedSetAttention,
    TeamPreviewHead,
    ValueHead,
)
from pokerl.model.memory import IdentityMemory
from pokerl.training.league import LeaguePlayer
from pokerl.training.ppo import PPOTrainer
from pokerl.training.preview import TeamPreviewResult
from pokerl.training.rollout import RolloutBuffer
from poke_env.battle import Move, Pokemon
from poke_env.battle.move import SPECIAL_MOVES
from poke_env.battle.target import Target
from poke_env.player import PassBattleOrder


def _obs(step: int, num_envs: int) -> dict[str, np.ndarray]:
    return {
        "float_obs": np.full((num_envs, 3), float(step), dtype=np.float32),
        "index_obs": np.full((num_envs, 2), step, dtype=np.int64),
    }


def _mask(num_envs: int) -> dict[str, np.ndarray]:
    joint = np.zeros((num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)
    joint[:, FIRST_SLOT_MASK_ROW, 0] = True
    joint[:, 1, 0] = True
    return {
        "joint": joint,
        "can_tera": np.zeros((num_envs, 2), dtype=bool),
    }


@pytest.mark.asyncio
async def test_async_queue_sync_get_records_item_timing() -> None:
    timing = showdown_env_module._TimingStats(enabled=True)
    queue = showdown_env_module._AsyncQueue(
        asyncio.get_running_loop(),
        sync_get=True,
        timing=timing,
        key_prefix="test_queue",
    )

    payload = object()
    queue.put(payload)

    assert queue.get(timeout=0.1) is payload
    snapshot = timing.snapshot_and_reset()
    assert snapshot["test_queue_put_calls"] == 1.0
    assert snapshot["test_queue_get_calls"] == 1.0
    assert snapshot["test_queue_get_wait_sec"] >= 0.0
    assert snapshot["test_queue_item_age_sec"] >= 0.0


@pytest.mark.asyncio
async def test_async_queue_threadsafe_put_records_loop_delay() -> None:
    timing = showdown_env_module._TimingStats(enabled=True)
    queue = showdown_env_module._AsyncQueue(
        asyncio.get_running_loop(),
        timing=timing,
        key_prefix="test_queue",
    )

    thread = Thread(target=queue.put, args=("action",))
    thread.start()
    thread.join(timeout=1.0)
    assert not thread.is_alive()

    assert await asyncio.wait_for(queue.async_get(), timeout=1.0) == "action"
    snapshot = timing.snapshot_and_reset()
    assert snapshot["test_queue_threadsafe_put_calls"] == 1.0
    assert snapshot["test_queue_threadsafe_put_completed_calls"] == 1.0
    assert snapshot["test_queue_put_calls"] == 1.0
    assert snapshot["test_queue_get_calls"] == 1.0
    assert snapshot["test_queue_loop_put_delay_sec"] >= 0.0
    assert snapshot["test_queue_item_age_sec"] >= 0.0


def _obs_to_tensor(
    obs: dict[str, np.ndarray], device: torch.device
) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value).to(device) for key, value in obs.items()}


def _aux_obs(num_envs: int, item_idx: int) -> dict[str, np.ndarray]:
    return {
        # Deliberately duplicate species: auxiliary alignment must use the
        # episode-local source identity, not species.
        "opp_species": np.full((num_envs, 6), 42, dtype=np.int64),
        "opp_source_index": np.broadcast_to(
            np.arange(6, dtype=np.int64),
            (num_envs, 6),
        ).copy(),
        "opp_item": np.full((num_envs, 6), item_idx, dtype=np.int64),
        "opp_ability": np.full((num_envs, 6), 2, dtype=np.int64),
        "opp_types": np.full((num_envs, 6, 3), 3, dtype=np.int64),
        "opp_moves": np.full((num_envs, 6, 4), 4, dtype=np.int64),
        "opp_continuous": np.zeros((num_envs, 6, CONTINUOUS_DIM), dtype=np.float32),
        "opp_mask": np.zeros((num_envs, 6), dtype=bool),
    }


def test_rollout_buffer_keeps_sequence_tensors_on_device_and_clones_initial_hidden() -> (
    None
):
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    buffer = RolloutBuffer(
        num_steps, num_envs, _obs(0, num_envs), _mask(num_envs), device
    )

    hidden_h = torch.arange(8, dtype=torch.float32, device=device).reshape(
        1, num_envs, 4
    )
    hidden_c = hidden_h + 100
    expected_h = hidden_h.squeeze(0).clone()
    expected_c = hidden_c.squeeze(0).clone()

    buffer.reset(initial_hidden=(hidden_h, hidden_c))
    hidden_h.zero_()
    hidden_c.zero_()

    for step in range(num_steps):
        obs_t = _obs_to_tensor(_obs(step, num_envs), device)
        actions_t = torch.tensor(
            [[step, step + 1, TERA_NONE], [step + 2, step + 3, TERA_NONE]],
            dtype=torch.long,
            device=device,
        )
        masks_t = {
            key: torch.from_numpy(value).to(device)
            for key, value in _mask(num_envs).items()
        }
        log_probs_t = torch.full(
            (num_envs,), float(step), dtype=torch.float32, device=device
        )
        values_t = torch.full(
            (num_envs,), float(step) + 0.5, dtype=torch.float32, device=device
        )
        rewards_t = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        dones_t = torch.tensor(
            [False, step == num_steps - 1], dtype=torch.bool, device=device
        )
        buffer.add(obs_t, actions_t, masks_t, log_probs_t, values_t, rewards_t, dones_t)

    batch = buffer.compute_gae(
        torch.zeros(num_envs, dtype=torch.float32, device=device)
    )

    assert batch.obs["float_obs"].shape == (num_steps, num_envs, 3)
    assert batch.obs["index_obs"].shape == (num_steps, num_envs, 2)
    assert batch.actions.shape == (num_steps, num_envs, 3)
    assert batch.action_masks["joint"].shape == (
        num_steps,
        num_envs,
        ACTIONS_PER_SLOT + 1,
        ACTIONS_PER_SLOT,
    )
    assert batch.action_masks["can_tera"].shape == (num_steps, num_envs, 2)
    assert batch.advantages.shape == (num_steps, num_envs)
    assert batch.obs["float_obs"].device == device
    assert torch.equal(batch.initial_hidden_h, expected_h)
    assert torch.equal(batch.initial_hidden_c, expected_c)
    assert batch.aux_labels == {}
    assert not batch.aux_label_valid.any()
    assert not batch.opp_actions_valid.any()


def test_rollout_buffer_uses_terminal_aux_obs_for_completed_episode_labels() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 1
    buffer = RolloutBuffer(
        num_steps, num_envs, _aux_obs(num_envs, 1), _mask(num_envs), device
    )
    buffer.reset()

    actions_t = torch.zeros((num_envs, 3), dtype=torch.long, device=device)
    masks_t = {
        key: torch.from_numpy(value).to(device)
        for key, value in _mask(num_envs).items()
    }
    log_probs_t = torch.zeros(num_envs, dtype=torch.float32, device=device)
    values_t = torch.zeros(num_envs, dtype=torch.float32, device=device)
    rewards_t = torch.zeros(num_envs, dtype=torch.float32, device=device)

    buffer.add(
        _obs_to_tensor(_aux_obs(num_envs, 1), device),
        actions_t,
        masks_t,
        log_probs_t,
        values_t,
        rewards_t,
        torch.tensor([True], dtype=torch.bool, device=device),
        aux_label_obs=_obs_to_tensor(_aux_obs(num_envs, 9), device),
    )
    buffer.add(
        _obs_to_tensor(_aux_obs(num_envs, 3), device),
        actions_t,
        masks_t,
        log_probs_t,
        values_t,
        rewards_t,
        torch.tensor([False], dtype=torch.bool, device=device),
    )

    batch = buffer.compute_gae(
        torch.zeros(num_envs, dtype=torch.float32, device=device)
    )

    assert batch.aux_label_valid[:, 0].any(dim=-1).tolist() == [True, False]
    assert batch.aux_labels["opp_item"][0, 0, 0].item() == 9


def test_rollout_buffer_aligns_terminal_aux_labels_by_source_identity() -> None:
    device = torch.device("cpu")
    buffer = RolloutBuffer(2, 1, _aux_obs(1, 1), _mask(1), device)
    buffer.reset()

    actions = torch.zeros((1, 3), dtype=torch.long)
    masks = _obs_to_tensor(_mask(1), device)
    zeros = torch.zeros(1)

    first = _aux_obs(1, 1)
    first["opp_source_index"][0] = [0, 1, -1, -1, -1, -1]
    first["opp_mask"][0] = [False, False, True, True, True, True]
    buffer.add(
        _obs_to_tensor(first, device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([False]),
    )

    switched = _aux_obs(1, 1)
    switched["opp_source_index"][0] = [1, 0, 2, -1, -1, -1]
    switched["opp_mask"][0] = [False, False, False, True, True, True]
    terminal = _aux_obs(1, 1)
    terminal["opp_source_index"][0] = [1, 0, 2, -1, -1, -1]
    terminal["opp_item"][0] = [200, 100, 300, 0, 0, 0]
    terminal["opp_mask"][0] = [False, False, False, True, True, True]
    buffer.add(
        _obs_to_tensor(switched, device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([True]),
        aux_label_obs=_obs_to_tensor(terminal, device),
    )

    batch = buffer.compute_gae(torch.zeros(1))

    assert batch.aux_labels["opp_item"][0, 0].tolist() == [100, 200, 0, 0, 0, 0]
    assert batch.aux_labels["opp_item"][1, 0].tolist() == [200, 100, 300, 0, 0, 0]
    assert batch.aux_label_valid[0, 0].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
    ]
    assert batch.aux_label_valid[1, 0].tolist() == [
        True,
        True,
        True,
        False,
        False,
        False,
    ]


def test_rollout_buffer_batches_aux_alignment_without_overwriting_prior_episode() -> (
    None
):
    device = torch.device("cpu")
    buffer = RolloutBuffer(3, 2, _aux_obs(2, 1), _mask(2), device)
    buffer.reset()
    actions = torch.zeros((2, 3), dtype=torch.long)
    masks = _obs_to_tensor(_mask(2), device)
    zeros = torch.zeros(2)

    first_terminal = _aux_obs(2, 0)
    first_terminal["opp_item"][0].fill(10)
    buffer.add(
        _obs_to_tensor(_aux_obs(2, 1), device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([True, False]),
        aux_label_obs=_obs_to_tensor(first_terminal, device),
        done_env_indices=[0],
    )
    buffer.add(
        _obs_to_tensor(_aux_obs(2, 2), device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([False, False]),
    )

    final_terminal = _aux_obs(2, 0)
    final_terminal["opp_item"][0].fill(20)
    final_terminal["opp_item"][1].fill(30)
    buffer.add(
        _obs_to_tensor(_aux_obs(2, 3), device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([True, True]),
        aux_label_obs=_obs_to_tensor(final_terminal, device),
        done_env_indices=[0, 1],
    )

    batch = buffer.compute_gae(torch.zeros(2))

    assert batch.aux_labels["opp_item"][:, 0, 0].tolist() == [10, 20, 20]
    assert batch.aux_labels["opp_item"][:, 1, 0].tolist() == [30, 30, 30]
    assert batch.aux_label_valid[:, 0, 0].tolist() == [True, True, True]
    assert batch.aux_label_valid[:, 1, 0].tolist() == [True, True, True]


def test_rollout_buffer_requires_a_complete_rollout_before_gae() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    buffer = RolloutBuffer(
        num_steps, num_envs, _obs(0, num_envs), _mask(num_envs), device
    )
    buffer.reset()

    buffer.add(
        _obs_to_tensor(_obs(0, num_envs), device),
        torch.zeros((num_envs, 3), dtype=torch.long, device=device),
        {
            key: torch.from_numpy(value).to(device)
            for key, value in _mask(num_envs).items()
        },
        torch.zeros(num_envs, dtype=torch.float32, device=device),
        torch.zeros(num_envs, dtype=torch.float32, device=device),
        torch.zeros(num_envs, dtype=torch.float32, device=device),
        torch.zeros(num_envs, dtype=torch.bool, device=device),
    )

    with pytest.raises(RuntimeError, match="incomplete rollout"):
        buffer.compute_gae(torch.zeros(num_envs, dtype=torch.float32, device=device))


class _DummySpatial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_proj = nn.Linear(3, 3)
        self.active_proj = nn.Linear(2, 2)
        self.team_proj = nn.Linear(2, 2)
        self.opp_proj = nn.Linear(2, 2)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_proj(obs["context"].float())
        active = self.active_proj(obs["active"].float())
        team = self.team_proj(obs["team"].float())
        opp_team = self.opp_proj(obs["opp_team"].float())
        opp_active = self.opp_proj(obs["opp_active"].float())
        return context, active, team, opp_team, opp_active


class _DummyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = _DummySpatial()
        self.memory = IdentityMemory(input_dim=3, hidden_dim=3)
        self.action_head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
        self.value_head = ValueHead(context_dim=3, pokemon_dim=2, hidden_dim=8)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden=None,
        actions: torch.Tensor | None = None,
        action_masks: dict[str, torch.Tensor] | None = None,
    ) -> dict[str, torch.Tensor]:
        context, active_reps, own_reps, opp_reps, opp_active_reps = self.spatial(obs)
        context, hidden = self.memory(context, hidden)
        own_bench_mask = (obs["own_roles"] != ROLE_BENCH) | obs["own_mask"]
        opp_bench_mask = (obs["opp_roles"] != ROLE_BENCH) | obs["opp_mask"]
        action_out = self.action_head(
            context,
            active_reps,
            own_reps,
            opp_reps,
            opp_bench_mask,
            opp_active_reps,
            action_masks,
            actions,
        )
        return {
            **action_out,
            "value": self.value_head(
                context, own_reps, opp_reps, own_bench_mask, opp_bench_mask
            ),
            "hidden": hidden,
        }

    def initial_hidden(self, batch_size: int):
        return self.memory.initial_hidden(batch_size)


def _ppo_obs(step: int, num_envs: int) -> dict[str, np.ndarray]:
    own_roles = np.tile(
        np.array(
            [
                ROLE_ACTIVE_1,
                ROLE_ACTIVE_2,
                ROLE_BENCH,
                ROLE_BENCH,
                ROLE_BENCH,
                ROLE_BENCH,
            ],
            dtype=np.int64,
        ),
        (num_envs, 1),
    )
    opp_roles = np.tile(
        np.array(
            [
                ROLE_ACTIVE_1,
                ROLE_ACTIVE_2,
                ROLE_BENCH,
                ROLE_BENCH,
                ROLE_BENCH,
                ROLE_BENCH,
            ],
            dtype=np.int64,
        ),
        (num_envs, 1),
    )
    mask = np.zeros((num_envs, 6), dtype=bool)
    return {
        "context": np.full((num_envs, 3), float(step), dtype=np.float32),
        "active": np.full((num_envs, 2, 2), float(step + 1), dtype=np.float32),
        "team": np.full((num_envs, 6, 2), float(step + 2), dtype=np.float32),
        "opp_team": np.full((num_envs, 6, 2), float(step + 2.5), dtype=np.float32),
        "opp_active": np.full((num_envs, 2, 2), float(step + 3), dtype=np.float32),
        "own_roles": own_roles,
        "opp_roles": opp_roles,
        "own_mask": mask.copy(),
        "opp_mask": mask.copy(),
    }


def test_ppo_trainer_updates_from_device_resident_sequence_batch() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    action_mask = {
        "joint": np.ones(
            (num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool
        ),
        "can_tera": np.ones((num_envs, 2), dtype=bool),
    }
    agent = _DummyAgent().to(device)
    config = TrainingConfig(num_epochs=1, num_minibatches=2)
    trainer = PPOTrainer(agent, config, device)
    buffer = RolloutBuffer(
        num_steps, num_envs, _ppo_obs(0, num_envs), action_mask, device
    )
    buffer.reset(initial_hidden=agent.initial_hidden(num_envs))

    for step in range(num_steps):
        obs_t = _obs_to_tensor(_ppo_obs(step, num_envs), device)
        masks_t = {
            key: torch.from_numpy(value).to(device)
            for key, value in action_mask.items()
        }
        with torch.no_grad():
            out = agent(obs_t, action_masks=masks_t)

        rewards_t = torch.tensor([1.0, -0.5], dtype=torch.float32, device=device)
        dones_t = torch.tensor(
            [False, step == num_steps - 1], dtype=torch.bool, device=device
        )
        buffer.add(
            obs_t,
            out["actions"],
            masks_t,
            out["log_prob"],
            out["value"],
            rewards_t,
            dones_t,
        )

    batch = buffer.compute_gae(
        torch.zeros(num_envs, dtype=torch.float32, device=device)
    )
    metrics = trainer.update(batch)

    assert trainer.global_step == num_steps * num_envs
    assert {"policy_loss", "value_loss", "entropy", "approx_kl"} <= set(metrics)
    assert metrics["pre_ppo_clip_fraction"] >= 0.0
    assert metrics["base_tera_real_choice_evaluations"] >= 0.0


def test_ppo_trainer_clamps_extreme_log_ratios_and_keeps_parameters_finite() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    action_mask = {
        "joint": np.ones(
            (num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool
        ),
        "can_tera": np.ones((num_envs, 2), dtype=bool),
    }
    agent = _DummyAgent().to(device)
    config = TrainingConfig(num_epochs=1, num_minibatches=2)
    trainer = PPOTrainer(agent, config, device)
    buffer = RolloutBuffer(
        num_steps, num_envs, _ppo_obs(0, num_envs), action_mask, device
    )
    buffer.reset(initial_hidden=agent.initial_hidden(num_envs))

    for step in range(num_steps):
        obs_t = _obs_to_tensor(_ppo_obs(step, num_envs), device)
        masks_t = {
            key: torch.from_numpy(value).to(device)
            for key, value in action_mask.items()
        }
        with torch.no_grad():
            out = agent(obs_t, action_masks=masks_t)

        rewards_t = torch.tensor([1.0, -0.5], dtype=torch.float32, device=device)
        dones_t = torch.tensor(
            [False, step == num_steps - 1], dtype=torch.bool, device=device
        )
        buffer.add(
            obs_t,
            out["actions"],
            masks_t,
            out["log_prob"],
            out["value"],
            rewards_t,
            dones_t,
        )

    batch = buffer.compute_gae(
        torch.zeros(num_envs, dtype=torch.float32, device=device)
    )
    batch.log_probs.fill_(-1.0e9)
    batch.advantages.copy_(
        torch.tensor(
            [[-1.0, 1.0], [-1.0, 1.0]],
            dtype=torch.float32,
            device=device,
        )
    )

    metrics = trainer.update(batch)

    assert all(np.isfinite(float(value)) for value in metrics.values())
    assert all(torch.isfinite(param).all() for param in agent.parameters())


def test_ppo_head_chunking_matches_stepwise_head_evaluation() -> None:
    device = torch.device("cpu")
    num_steps = 3
    num_envs = 2
    action_mask = {
        "joint": np.ones(
            (num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool
        ),
        "can_tera": np.ones((num_envs, 2), dtype=bool),
    }

    torch.manual_seed(42)
    base_agent = _DummyAgent().to(device)
    initial_state = copy.deepcopy(base_agent.state_dict())

    buffer = RolloutBuffer(
        num_steps, num_envs, _ppo_obs(0, num_envs), action_mask, device
    )
    buffer.reset(initial_hidden=base_agent.initial_hidden(num_envs))
    for step in range(num_steps):
        obs_t = _obs_to_tensor(_ppo_obs(step, num_envs), device)
        masks_t = {
            key: torch.from_numpy(value).to(device)
            for key, value in action_mask.items()
        }
        with torch.no_grad():
            out = base_agent(obs_t, action_masks=masks_t)

        rewards_t = torch.tensor([1.0, -0.5], dtype=torch.float32, device=device)
        dones_t = torch.tensor(
            [False, step == num_steps - 1], dtype=torch.bool, device=device
        )
        buffer.add(
            obs_t,
            out["actions"],
            masks_t,
            out["log_prob"],
            out["value"],
            rewards_t,
            dones_t,
        )

    batch = buffer.compute_gae(
        torch.zeros(num_envs, dtype=torch.float32, device=device)
    )
    config = TrainingConfig(num_epochs=1, num_minibatches=1, lr=1e-4)

    step_agent = _DummyAgent().to(device)
    step_agent.load_state_dict(initial_state)
    step_trainer = PPOTrainer(step_agent, config, device)
    step_trainer._HEAD_BATCH_TARGET = 1
    torch.manual_seed(777)
    step_metrics = step_trainer.update(batch)

    chunk_agent = _DummyAgent().to(device)
    chunk_agent.load_state_dict(initial_state)
    chunk_trainer = PPOTrainer(chunk_agent, config, device)
    chunk_trainer._HEAD_BATCH_TARGET = 2048
    torch.manual_seed(777)
    chunk_metrics = chunk_trainer.update(batch)

    for key in step_metrics:
        assert chunk_metrics[key] == pytest.approx(step_metrics[key], abs=1e-6)
    for key, step_value in step_agent.state_dict().items():
        chunk_value = chunk_agent.state_dict()[key]
        if step_value.dtype.is_floating_point:
            assert torch.allclose(chunk_value, step_value, atol=1e-6, rtol=1e-6)
        else:
            assert torch.equal(chunk_value, step_value)


def _minimal_action_head_inputs():
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    own_team_reps = torch.zeros((1, 6, 2))
    opp_team_reps = torch.zeros((1, 6, 2))
    opp_reserve_mask = torch.tensor(
        [[True, True, False, False, False, False]],
        dtype=torch.bool,
    )
    opp_active_reps = torch.zeros((1, 2, 2))
    masks = {
        "joint": torch.zeros(
            (1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
            dtype=torch.bool,
        ),
        "can_tera": torch.zeros((1, 2), dtype=torch.bool),
    }
    masks["joint"][0, FIRST_SLOT_MASK_ROW, 0] = True
    masks["joint"][0, 1, 0] = True
    return (
        head,
        context,
        active_reps,
        own_team_reps,
        opp_team_reps,
        opp_reserve_mask,
        opp_active_reps,
        masks,
    )


def test_action_head_uses_joint_and_tera_masks_to_coordinate_actions() -> None:
    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    own_team_reps = torch.zeros((1, 6, 2))
    opp_team_reps = torch.zeros((1, 6, 2))
    opp_reserve_mask = torch.tensor(
        [[True, True, False, False, False, False]], dtype=torch.bool
    )
    opp_active_reps = torch.zeros((1, 2, 2))
    masks = {
        "joint": torch.zeros(
            (1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=torch.bool
        ),
        "can_tera": torch.zeros((1, 2), dtype=torch.bool),
    }
    masks["joint"][0, FIRST_SLOT_MASK_ROW, 0] = True
    masks["joint"][0, 1, 0] = True
    masks["can_tera"][0, 0] = True

    chosen_actions = torch.tensor([[0, 0, TERA_SLOT_1]], dtype=torch.long)
    out = head(
        context,
        active_reps,
        own_team_reps,
        opp_team_reps,
        opp_reserve_mask,
        opp_active_reps,
        masks,
        actions=chosen_actions,
    )

    assert out["actions"][0, 0].item() == 0
    assert out["actions"][0, 1].item() == 0
    assert out["actions"][0, 2].item() == TERA_SLOT_1


def test_action_head_can_skip_policy_stats_for_frozen_opponent_inference() -> None:
    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    own_team_reps = torch.zeros((1, 6, 2))
    opp_team_reps = torch.zeros((1, 6, 2))
    opp_reserve_mask = torch.tensor(
        [[True, True, False, False, False, False]], dtype=torch.bool
    )
    opp_active_reps = torch.zeros((1, 2, 2))
    masks = {
        "joint": torch.zeros(
            (1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=torch.bool
        ),
        "can_tera": torch.zeros((1, 2), dtype=torch.bool),
    }
    masks["joint"][0, FIRST_SLOT_MASK_ROW, 0] = True
    masks["joint"][0, 1, 0] = True
    chosen_actions = torch.tensor([[0, 0, TERA_NONE]], dtype=torch.long)

    out = head(
        context,
        active_reps,
        own_team_reps,
        opp_team_reps,
        opp_reserve_mask,
        opp_active_reps,
        masks,
        actions=chosen_actions,
        include_policy_stats=False,
    )

    assert torch.equal(out["actions"], chosen_actions)
    assert "log_prob" not in out
    assert "entropy" not in out


def test_action_head_sampled_actions_skip_host_legality_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _minimal_action_head_inputs()

    def fail_all(*args, **kwargs):
        del args, kwargs
        raise AssertionError("sampled actions must not run Tensor.all legality checks")

    monkeypatch.setattr(torch.Tensor, "all", fail_all)

    out = inputs[0](
        *inputs[1:-1],
        inputs[-1],
        greedy=True,
        include_policy_stats=False,
    )

    assert torch.equal(
        out["actions"],
        torch.tensor([[0, 0, TERA_NONE]], dtype=torch.long),
    )


def test_action_head_validates_externally_supplied_actions_by_default() -> None:
    inputs = _minimal_action_head_inputs()
    illegal_actions = torch.tensor([[1, 0, TERA_NONE]], dtype=torch.long)

    with pytest.raises(RuntimeError, match="illegal first-slot action"):
        inputs[0](
            *inputs[1:-1],
            inputs[-1],
            actions=illegal_actions,
            include_policy_stats=False,
        )


def test_action_head_allows_trusted_supplied_actions_to_skip_validation() -> None:
    inputs = _minimal_action_head_inputs()
    trusted_actions = torch.tensor([[1, 0, TERA_NONE]], dtype=torch.long)

    out = inputs[0](
        *inputs[1:-1],
        inputs[-1],
        actions=trusted_actions,
        validate_actions=False,
        include_policy_stats=False,
    )

    assert torch.equal(out["actions"], trusted_actions)


def test_action_head_uses_conditional_joint_mask_to_block_duplicate_switch_target() -> (
    None
):
    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    own_team_reps = torch.zeros((1, 6, 2))
    opp_team_reps = torch.zeros((1, 6, 2))
    opp_reserve_mask = torch.tensor(
        [[True, True, False, False, False, False]], dtype=torch.bool
    )
    opp_active_reps = torch.zeros((1, 2, 2))
    masks = {
        "joint": torch.zeros(
            (1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=torch.bool
        ),
        "can_tera": torch.zeros((1, 2), dtype=torch.bool),
    }
    shared_switch = SWITCH_OFFSET + 5
    masks["joint"][0, FIRST_SLOT_MASK_ROW, shared_switch] = True
    masks["joint"][0, shared_switch + 1, 0] = True

    out = head(
        context,
        active_reps,
        own_team_reps,
        opp_team_reps,
        opp_reserve_mask,
        opp_active_reps,
        masks,
    )

    assert out["actions"][0, 0].item() == shared_switch
    assert out["actions"][0, 1].item() == 0
    assert out["actions"][0, 2].item() == TERA_NONE


def test_action_head_matchup_features_distinguish_super_effective_and_resisted_targets() -> (
    None
):
    head = ActionHead(
        context_dim=3,
        pokemon_dim=2,
        hidden_dim=8,
        move_type_lookup=torch.tensor([0, 1], dtype=torch.long),
        move_base_power_lookup=torch.tensor([0.0, 1.0], dtype=torch.float32),
        move_accuracy_lookup=torch.tensor([0.0, 1.0], dtype=torch.float32),
        move_priority_lookup=torch.tensor([0.0, 0.0], dtype=torch.float32),
        move_category_lookup=torch.tensor([0, 1], dtype=torch.long),
        move_target_flags=torch.zeros((2, 6), dtype=torch.float32),
        type_effectiveness=torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 2.0, 0.5],
                [1.0, 0.5, 1.0, 2.0],
                [1.0, 2.0, 0.5, 1.0],
            ],
            dtype=torch.float32,
        ),
    )
    battle_obs = {
        "own_moves": torch.tensor([[[1, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.long),
        "own_types": torch.tensor([[[1, 0, 0], [2, 0, 0]]], dtype=torch.long),
        "own_continuous": torch.zeros((1, 2, 44), dtype=torch.float32),
        "own_active_indices": torch.tensor([[0, 1]], dtype=torch.long),
        "own_active_present": torch.tensor([[True, True]], dtype=torch.bool),
        "opp_types": torch.tensor([[[2, 0, 0], [3, 0, 0]]], dtype=torch.long),
        "opp_continuous": torch.zeros((1, 2, 44), dtype=torch.float32),
        "opp_active_indices": torch.tensor([[0, 1]], dtype=torch.long),
        "opp_active_present": torch.tensor([[True, True]], dtype=torch.bool),
    }

    features = head._candidate_action_features(
        battle_obs=battle_obs,
        actor_slot=0,
        can_tera_slot=torch.tensor([True], dtype=torch.bool),
        batch_size=1,
        device=torch.device("cpu"),
    )

    target_eff_feature = 14
    assert features[0, 0, target_eff_feature] > features[0, 1, target_eff_feature]


def test_action_head_categorical_choice_matches_torch_distribution_for_evaluation() -> (
    None
):
    logits = torch.tensor(
        [
            [1.5, -2.0, 0.25],
            [-0.75, 0.5, 2.25],
        ],
        dtype=torch.float32,
    )
    actions = torch.tensor([0, 2], dtype=torch.long)

    selected, log_prob, entropy = ActionHead._categorical_choice(
        logits,
        actions=actions,
        greedy=False,
    )
    dist = torch.distributions.Categorical(logits=logits)

    assert torch.equal(selected, actions)
    assert torch.allclose(log_prob, dist.log_prob(actions))
    assert torch.allclose(entropy, dist.entropy())


def test_action_head_categorical_choice_samples_without_torch_multinomial(
    monkeypatch,
) -> None:
    def fail_multinomial(*args, **kwargs):
        raise AssertionError("torch.multinomial should not be used for policy sampling")

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    torch.manual_seed(0)
    logits = torch.tensor(
        [
            [1.5, -2.0, 0.25],
            [-0.75, 0.5, 2.25],
        ],
        dtype=torch.float32,
    )

    selected, log_prob, entropy = ActionHead._categorical_choice(
        logits,
        actions=None,
        greedy=False,
    )
    selected_no_stats, log_prob_no_stats, entropy_no_stats = (
        ActionHead._categorical_choice(
            logits,
            actions=None,
            greedy=False,
            include_stats=False,
        )
    )

    assert selected.shape == (2,)
    assert selected.device == logits.device
    assert torch.all((selected >= 0) & (selected < logits.shape[1]))
    assert log_prob.shape == (2,)
    assert entropy.shape == (2,)
    assert selected_no_stats.shape == (2,)
    assert torch.all((selected_no_stats >= 0) & (selected_no_stats < logits.shape[1]))
    assert log_prob_no_stats is None
    assert entropy_no_stats is None


def test_action_head_force_delay_turns_early_sampled_tera_into_no_tera() -> None:
    class FixedTeraScore(nn.Module):
        def forward(self, query: torch.Tensor) -> torch.Tensor:
            logits = torch.full(
                (query.shape[0], 3), -1e8, dtype=query.dtype, device=query.device
            )
            logits[:, TERA_SLOT_1] = 1e8
            return logits

    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    head.tera_score = FixedTeraScore()
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    own_team_reps = torch.zeros((1, 6, 2))
    opp_team_reps = torch.zeros((1, 6, 2))
    opp_reserve_mask = torch.tensor(
        [[True, True, False, False, False, False]], dtype=torch.bool
    )
    opp_active_reps = torch.zeros((1, 2, 2))
    masks = {
        "joint": torch.zeros(
            (1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=torch.bool
        ),
        "can_tera": torch.tensor([[True, False]], dtype=torch.bool),
    }
    masks["joint"][0, FIRST_SLOT_MASK_ROW, 0] = True
    masks["joint"][0, 1, 0] = True
    battle_obs = {
        "field": torch.zeros((1, FIELD_TURN_OFFSET + 1), dtype=torch.float32),
    }
    battle_obs["field"][0, FIELD_TURN_OFFSET] = 1.0 / float(FIELD_TIMER_CAPS["turn"])

    out = head(
        context,
        active_reps,
        own_team_reps,
        opp_team_reps,
        opp_reserve_mask,
        opp_active_reps,
        masks,
        battle_obs=battle_obs,
        tera_force_delay_prob=1.0,
        tera_force_delay_max_turn=3,
    )

    assert out["actions"][0, 2].item() == TERA_NONE

    battle_obs["field"][0, FIELD_TURN_OFFSET] = 4.0 / float(FIELD_TIMER_CAPS["turn"])
    out_after_window = head(
        context,
        active_reps,
        own_team_reps,
        opp_team_reps,
        opp_reserve_mask,
        opp_active_reps,
        masks,
        battle_obs=battle_obs,
        tera_force_delay_prob=1.0,
        tera_force_delay_max_turn=3,
    )

    assert out_after_window["actions"][0, 2].item() == TERA_SLOT_1


def test_masked_set_attention_returns_zero_when_every_item_is_masked() -> None:
    attention = MaskedSetAttention(query_dim=3, item_dim=2, hidden_dim=4)
    query = torch.randn(2, 3)
    items = torch.randn(2, 4, 2)
    mask = torch.ones((2, 4), dtype=torch.bool)

    summary = attention(query, items, mask)

    assert torch.equal(summary, torch.zeros_like(summary))


def test_team_preview_head_accepts_opponent_structure_and_greedy_mode() -> None:
    torch.manual_seed(0)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=3)
    own_reps = torch.randn(2, 6, 4)
    opp_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)
    own_mask = torch.tensor(
        [
            [False, False, False, False, False, True],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    opponent_mask = torch.tensor(
        [
            [False, False, False, False, True, True],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    )

    out = head(
        own_reps,
        opp_reps,
        context,
        team_mask=own_mask,
        opponent_mask=opponent_mask,
        greedy=True,
    )

    assert out["picks"].shape == (2, 3)
    assert out["log_prob"].shape == (2,)
    assert out["entropy"].shape == (2,)
    assert torch.all(out["picks"][0] != 5)
    assert torch.all(out["bring_picks"][:, 1:] > out["bring_picks"][:, :-1])
    assert torch.all(out["picks"][:, 2:] == out["picks"][:, 2:].sort(dim=1).values)


def test_team_preview_head_samples_without_torch_multinomial(monkeypatch) -> None:
    def fail_multinomial(*args, **kwargs):
        raise AssertionError(
            "torch.multinomial should not be used for team-preview sampling"
        )

    monkeypatch.setattr(torch, "multinomial", fail_multinomial)
    torch.manual_seed(0)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=3)
    own_reps = torch.randn(2, 6, 4)
    opp_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)

    out = head(own_reps, opp_reps, context, greedy=False)

    assert out["picks"].shape == (2, 3)
    assert out["log_prob"].shape == (2,)
    assert out["entropy"].shape == (2,)


def test_team_preview_head_sampled_picks_skip_host_legality_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_all(*args, **kwargs):
        del args, kwargs
        raise AssertionError("sampled picks must not run Tensor.all legality checks")

    monkeypatch.setattr(torch.Tensor, "all", fail_all)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=3)
    out = head(
        torch.randn(2, 6, 4),
        torch.randn(2, 6, 4),
        torch.randn(2, 6),
        greedy=True,
        include_policy_stats=False,
    )

    assert out["picks"].shape == (2, 3)


def test_team_preview_head_validates_externally_supplied_picks() -> None:
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=3)

    with pytest.raises(RuntimeError, match="invalid canonical bring target"):
        head(
            torch.randn(1, 6, 4),
            torch.randn(1, 6, 4),
            torch.randn(1, 6),
            picks=torch.tensor([[0, 0, 1]], dtype=torch.long),
        )


def test_team_preview_head_can_return_picks_without_policy_stats() -> None:
    torch.manual_seed(0)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=3)
    own_reps = torch.randn(2, 6, 4)
    opp_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)

    out = head(own_reps, opp_reps, context, greedy=False, include_policy_stats=False)

    assert out["picks"].shape == (2, 3)
    assert out["lead_pair"].shape == (2, 2)
    assert out["bring_picks"].shape == (2, 3)
    assert "log_prob" not in out
    assert "entropy" not in out


def test_team_preview_head_preserves_lead_order_while_canonicalizing_bench_tail() -> (
    None
):
    torch.manual_seed(0)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=4)
    own_reps = torch.randn(1, 6, 4)
    opp_reps = torch.randn(1, 6, 4)
    context = torch.randn(1, 6)
    provided_picks = torch.tensor([[4, 1, 5, 0]], dtype=torch.long)

    out = head(own_reps, opp_reps, context, picks=provided_picks)

    assert torch.equal(out["lead_pair"], torch.tensor([[4, 1]], dtype=torch.long))
    assert torch.equal(
        out["bring_picks"], torch.tensor([[0, 1, 4, 5]], dtype=torch.long)
    )
    assert torch.equal(out["picks"], torch.tensor([[4, 1, 0, 5]], dtype=torch.long))


class _FakePokemon:
    def __init__(self, species: str):
        self.species = species
        self.name = species
        self.fainted = False


class _FakeBattle:
    POKEMON_1_POSITION = -1
    POKEMON_2_POSITION = -2
    OPPONENT_1_POSITION = 1
    OPPONENT_2_POSITION = 2
    EMPTY_TARGET_POSITION = 0

    def __init__(self) -> None:
        self.force_switch = [True, False]
        self.can_tera = [False, False]
        self.active_pokemon = [_FakePokemon("lead-one"), _FakePokemon("lead-two")]
        bench_one = _FakePokemon("bench-one")
        bench_two = _FakePokemon("bench-two")
        # Intentionally place bench-two before active mons in team order to verify stable remapping.
        self.team = {
            "p1:3": bench_two,
            "p1:1": self.active_pokemon[0],
            "p1:2": self.active_pokemon[1],
            "p1:4": bench_one,
        }
        self.available_moves = [[], []]
        self.available_switches = [[bench_one], []]
        self.last_request = {"active": [{"moves": []}, {"moves": []}]}


class _DefaultOnlyBattle:
    POKEMON_1_POSITION = -1
    POKEMON_2_POSITION = -2
    OPPONENT_1_POSITION = 1
    OPPONENT_2_POSITION = 2
    EMPTY_TARGET_POSITION = 0

    def __init__(self) -> None:
        self.finished = False
        self.battle_tag = "battle-default-only"
        self.turn = 3
        self.force_switch = [False, False]
        self.can_tera = [False, False]
        self.trapped = [False, False]
        self.active_pokemon = [None, None]
        self.opponent_active_pokemon = [None, None]
        self.available_moves = [[], []]
        self.available_switches = [[], []]
        self.team = {}
        self.last_request = {"active": [{"moves": []}, {"moves": []}]}


def test_switch_actions_use_stable_team_slots() -> None:
    battle = _FakeBattle()

    mask = build_action_mask(battle, slot=0)
    expected_team_slot = 3  # [lead-one, lead-two, bench-two, bench-one, None, None]
    switch_action = SWITCH_OFFSET + expected_team_slot

    assert mask[switch_action]
    assert mask[SWITCH_OFFSET:PASS_ACTION].sum() == 1

    order = action_to_order(switch_action, battle, slot=0)
    assert order.order is battle.available_switches[0][0]


def test_legal_action_state_reuses_slot_orders(monkeypatch: pytest.MonkeyPatch) -> None:
    battle = _FakeBattle()
    original = actions_module._build_slot_action_state
    calls = 0

    def counted_slot_action_state(current_battle, slot):
        nonlocal calls
        calls += 1
        return original(current_battle, slot)

    monkeypatch.setattr(
        actions_module, "_build_slot_action_state", counted_slot_action_state
    )

    action_state = build_legal_action_state(battle)
    expected_team_slot = 3
    switch_action = SWITCH_OFFSET + expected_team_slot

    assert build_action_mask(battle, slot=0, action_state=action_state)[switch_action]
    assert build_joint_action_mask(battle, action_state=action_state)[
        FIRST_SLOT_MASK_ROW, switch_action
    ]
    assert (
        action_to_order(switch_action, battle, slot=0, action_state=action_state).order
        is battle.available_switches[0][0]
    )
    assert calls == 2


def test_default_only_action_state_detects_pass_only_request() -> None:
    action_state = build_legal_action_state(_DefaultOnlyBattle())

    assert is_default_only_action_state(action_state)


@pytest.mark.asyncio
async def test_showdown_player_default_only_request_skips_observation() -> None:
    player = ShowdownPlayer.__new__(ShowdownPlayer)
    player._closing = False
    player.choose_default_move = lambda: PassBattleOrder()
    player._begin_action_request = lambda: (_ for _ in ()).throw(
        AssertionError("default-only request should not ask the trainer for an action")
    )
    player._emit_battle_update = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("default-only request should not emit an observation")
    )

    order = await ShowdownPlayer._choose_move(player, _DefaultOnlyBattle())

    assert isinstance(order, PassBattleOrder)


class _FakeTargetBattle:
    EMPTY_TARGET_POSITION = 0
    POKEMON_1_POSITION = -1
    POKEMON_2_POSITION = -2
    OPPONENT_1_POSITION = 1
    OPPONENT_2_POSITION = 2

    def __init__(self, move: Move) -> None:
        self.force_switch = [False, False]
        self.can_tera = [False, False]
        self.active_pokemon = [_FakePokemon("attacker"), _FakePokemon("ally")]
        self.opponent_active_pokemon = [
            _FakePokemon("foe-one"),
            _FakePokemon("foe-two"),
        ]
        self.available_moves = [[move], []]
        self.available_switches = [[], []]
        self.team = {
            "p1:1": self.active_pokemon[0],
            "p1:2": self.active_pokemon[1],
        }
        self.last_request = {
            "active": [
                {
                    "moves": [
                        {
                            "id": "icespinner",
                            "target": "adjacentFoe",
                            "disabled": False,
                        }
                    ]
                },
                {"moves": []},
            ]
        }
        self.trapped = [False, False]

    def get_possible_showdown_targets(
        self, move: Move, pokemon, dynamax: bool = False
    ) -> list[int]:
        del dynamax
        if pokemon == self.active_pokemon[0] and move.id in [
            m.id for m in self.available_moves[0]
        ]:
            self_position = self.POKEMON_1_POSITION
            ally_position = self.POKEMON_2_POSITION
        elif pokemon == self.active_pokemon[1] and move.id in [
            m.id for m in self.available_moves[1]
        ]:
            self_position = self.POKEMON_2_POSITION
            ally_position = self.POKEMON_1_POSITION
        else:
            raise AssertionError("move does not belong to an active ally")

        if move.id in SPECIAL_MOVES:
            targets = [self.EMPTY_TARGET_POSITION]
        else:
            targets = {
                Target.ADJACENT_ALLY: [ally_position],
                Target.ADJACENT_ALLY_OR_SELF: [ally_position, self_position],
                Target.ADJACENT_FOE: [
                    self.OPPONENT_1_POSITION,
                    self.OPPONENT_2_POSITION,
                ],
                Target.ALL: [self.EMPTY_TARGET_POSITION],
                Target.ALL_ADJACENT: [self.EMPTY_TARGET_POSITION],
                Target.ALL_ADJACENT_FOES: [self.EMPTY_TARGET_POSITION],
                Target.ALLIES: [self.EMPTY_TARGET_POSITION],
                Target.ALLY_SIDE: [self.EMPTY_TARGET_POSITION],
                Target.ALLY_TEAM: [self.EMPTY_TARGET_POSITION],
                Target.ANY: [
                    ally_position,
                    self.OPPONENT_1_POSITION,
                    self.OPPONENT_2_POSITION,
                ],
                Target.FOE_SIDE: [self.EMPTY_TARGET_POSITION],
                Target.NORMAL: [
                    ally_position,
                    self.OPPONENT_1_POSITION,
                    self.OPPONENT_2_POSITION,
                ],
                Target.RANDOM_NORMAL: [self.EMPTY_TARGET_POSITION],
                Target.SCRIPTED: [self.EMPTY_TARGET_POSITION],
                Target.SELF: [self.EMPTY_TARGET_POSITION],
                None: [self.OPPONENT_1_POSITION, self.OPPONENT_2_POSITION],
            }[move.deduced_target]

        occupied_targets = {self.EMPTY_TARGET_POSITION}
        for position, target_pokemon in {
            self.POKEMON_1_POSITION: self.active_pokemon[0],
            self.POKEMON_2_POSITION: self.active_pokemon[1],
            self.OPPONENT_1_POSITION: self.opponent_active_pokemon[0],
            self.OPPONENT_2_POSITION: self.opponent_active_pokemon[1],
        }.items():
            if target_pokemon is not None and not target_pokemon.fainted:
                occupied_targets.add(position)
        return [target for target in targets if target in occupied_targets]


def test_request_target_restricts_foe_only_moves_in_action_mask() -> None:
    move = Move("icespinner", gen=9)
    move.request_index = 1
    battle = _FakeTargetBattle(move)

    mask = build_action_mask(battle, slot=0)

    assert mask[0]
    assert mask[1]
    assert not mask[2]
    assert action_to_order(0, battle, slot=0).message == "/choose move 1 1"

    with pytest.raises(RuntimeError, match="not legal"):
        action_to_order(2, battle, slot=0)


def test_targetless_orders_are_representable_in_the_action_space() -> None:
    battle = _FakeTargetBattle(Move("dazzlinggleam", gen=9))
    battle.last_request = {
        "active": [
            {
                "moves": [
                    {
                        "id": "dazzlinggleam",
                        "target": "allAdjacentFoes",
                        "disabled": False,
                    }
                ]
            },
            {"moves": []},
        ]
    }

    mask = build_action_mask(battle, slot=0)
    no_target_action = TARGET_NONE

    assert mask[no_target_action]

    order = action_to_order(no_target_action, battle, slot=0)
    assert order.move_target == battle.EMPTY_TARGET_POSITION


def test_self_target_orders_are_representable_in_the_action_space() -> None:
    battle = _FakeTargetBattle(Move("acupressure", gen=9))
    battle.last_request = {
        "active": [
            {
                "moves": [
                    {
                        "id": "acupressure",
                        "target": "adjacentAllyOrSelf",
                        "disabled": False,
                    }
                ]
            },
            {"moves": []},
        ]
    }

    mask = build_action_mask(battle, slot=0)
    self_target_action = TARGET_SELF

    assert mask[self_target_action]

    order = action_to_order(self_target_action, battle, slot=0)
    assert order.move_target == battle.POKEMON_1_POSITION


def test_struggle_does_not_expose_explicit_targets() -> None:
    battle = _FakeTargetBattle(Move("struggle", gen=9))
    battle.last_request = {
        "active": [
            {
                "moves": [
                    {
                        "id": "struggle",
                        "target": "randomNormal",
                        "disabled": False,
                    }
                ]
            },
            {"moves": []},
        ]
    }

    mask = build_action_mask(battle, slot=0)

    assert mask[TARGET_NONE]
    assert not mask[0]
    assert not mask[1]
    assert not mask[2]
    assert not mask[TARGET_SELF]

    order = action_to_order(TARGET_NONE, battle, slot=0)
    assert order.move_target == battle.EMPTY_TARGET_POSITION


@pytest.mark.parametrize(
    "move_request",
    [
        {"id": "outrage", "target": None, "disabled": False},
        {"id": "outrage", "disabled": False},
    ],
)
def test_random_normal_move_without_explicit_request_target_uses_request_slot(
    move_request: dict[str, object],
) -> None:
    move = Move("outrage", gen=9)
    move.request_index = 1
    battle = _FakeTargetBattle(move)
    battle.last_request = {
        "active": [
            {"moves": [move_request]},
            {"moves": []},
        ]
    }

    mask = build_action_mask(battle, slot=0)

    assert mask[TARGET_NONE]
    assert not mask[0]
    assert not mask[1]
    assert action_to_order(TARGET_NONE, battle, slot=0).message == "/choose move 1"

    with pytest.raises(RuntimeError, match="not legal"):
        action_to_order(0, battle, slot=0)


def test_action_mask_filters_missing_target_slots() -> None:
    battle = _FakeTargetBattle(Move("icespinner", gen=9))
    battle.opponent_active_pokemon[0] = None

    mask = build_action_mask(battle, slot=0)

    assert not mask[0]
    assert mask[1]


def test_parallel_battle_env_collate_reuses_batch_buffers() -> None:
    env = object.__new__(ParallelBattleEnv)
    env.num_envs = 2
    env._obs_batch_buffers = None
    env._mask_batch_buffers = None

    first_obs = [
        {
            "a": np.array([1, 2], dtype=np.int64),
            "b": np.array([[1.0], [2.0]], dtype=np.float32),
        },
        {
            "a": np.array([3, 4], dtype=np.int64),
            "b": np.array([[3.0], [4.0]], dtype=np.float32),
        },
    ]
    first_masks = [
        {
            "joint": np.array([[True, False], [False, True]], dtype=bool),
            "can_tera": np.array([True, False], dtype=bool),
        },
        {
            "joint": np.array([[False, True], [True, False]], dtype=bool),
            "can_tera": np.array([False, True], dtype=bool),
        },
    ]
    batch_obs_1, batch_mask_1 = ParallelBattleEnv._collate(env, first_obs, first_masks)

    second_obs = [
        {
            "a": np.array([5, 6], dtype=np.int64),
            "b": np.array([[5.0], [6.0]], dtype=np.float32),
        },
        {
            "a": np.array([7, 8], dtype=np.int64),
            "b": np.array([[7.0], [8.0]], dtype=np.float32),
        },
    ]
    second_masks = [
        {
            "joint": np.array([[False, False], [True, True]], dtype=bool),
            "can_tera": np.array([False, False], dtype=bool),
        },
        {
            "joint": np.array([[True, True], [False, False]], dtype=bool),
            "can_tera": np.array([True, True], dtype=bool),
        },
    ]
    batch_obs_2, batch_mask_2 = ParallelBattleEnv._collate(
        env, second_obs, second_masks
    )

    assert batch_obs_1["a"] is batch_obs_2["a"]
    assert batch_obs_1["b"] is batch_obs_2["b"]
    assert batch_mask_1["joint"] is batch_mask_2["joint"]
    assert batch_mask_1["can_tera"] is batch_mask_2["can_tera"]
    np.testing.assert_array_equal(
        batch_obs_2["a"], np.array([[5, 6], [7, 8]], dtype=np.int64)
    )
    np.testing.assert_array_equal(
        batch_obs_2["b"],
        np.array([[[5.0], [6.0]], [[7.0], [8.0]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        batch_mask_2["joint"],
        np.array(
            [
                [[False, False], [True, True]],
                [[True, True], [False, False]],
            ],
            dtype=bool,
        ),
    )


def test_joint_action_mask_comes_from_joined_valid_orders() -> None:
    first_move = Move("icespinner", gen=9)
    second_move = Move("suckerpunch", gen=9)
    lead_one = Pokemon(gen=9, species="Indeedee")
    lead_two = Pokemon(gen=9, species="Amoonguss")
    shared_switch = Pokemon(gen=9, species="Farigiraf")

    battle = SimpleNamespace(
        force_switch=[False, False],
        active_pokemon=[lead_one, lead_two],
        available_moves=[[first_move], [second_move]],
        available_switches=[[shared_switch], [shared_switch]],
        can_tera=[True, True],
        trapped=[False, False],
        team={
            "p1:1": lead_one,
            "p1:2": lead_two,
            "p1:3": shared_switch,
        },
        POKEMON_1_POSITION=-1,
        POKEMON_2_POSITION=-2,
        OPPONENT_1_POSITION=1,
        OPPONENT_2_POSITION=2,
        EMPTY_TARGET_POSITION=0,
        get_possible_showdown_targets=lambda move, pokemon, dynamax=False: [1, 2],
        last_request={
            "active": [
                {
                    "moves": [
                        {
                            "id": "icespinner",
                            "target": "adjacentFoe",
                            "disabled": False,
                        },
                    ]
                },
                {
                    "moves": [
                        {
                            "id": "suckerpunch",
                            "target": "adjacentFoe",
                            "disabled": False,
                        },
                    ]
                },
            ]
        },
    )

    joint_mask = build_joint_action_mask(battle)
    tera_mask = build_tera_action_mask(battle)
    shared_switch_action = SWITCH_OFFSET + 2

    assert joint_mask[FIRST_SLOT_MASK_ROW, 0]
    assert joint_mask[1, 0]
    assert joint_mask[FIRST_SLOT_MASK_ROW, shared_switch_action]
    assert joint_mask[shared_switch_action + 1, 0]
    assert not joint_mask[shared_switch_action + 1, shared_switch_action]
    assert tera_mask[0, 0, TERA_NONE]
    assert tera_mask[0, 0, TERA_SLOT_1]
    assert tera_mask[0, 0, TERA_SLOT_2]
    assert tera_mask[shared_switch_action, 0, TERA_NONE]
    assert not tera_mask[shared_switch_action, 0, TERA_SLOT_1]


def test_battle_env_team_preview_propagates_precompute_failure() -> None:
    obs = {"context": np.array([1, 2, 3], dtype=np.int64)}
    future: Future = Future()
    future.set_exception(RuntimeError("preview batch failed"))
    timing = SimpleNamespace(
        increment=lambda *args, **kwargs: None, add=lambda *args, **kwargs: None
    )
    fake_env = SimpleNamespace(
        _active_preview_obs=obs,
        _active_preview_future=future,
        _timing=timing,
        _ext_team_preview_fn=lambda current_obs: (_ for _ in ()).throw(
            AssertionError("fallback should not run")
        ),
        _obs_equal=lambda left, right: BattleEnv._obs_equal(left, right),
        _record_preview_obs_mismatch=lambda actual, prepared: (_ for _ in ()).throw(
            AssertionError("mismatch handler should not run")
        ),
    )

    with pytest.raises(RuntimeError, match="preview batch failed"):
        BattleEnv._choose_team_preview_picks(fake_env, obs)


@pytest.mark.asyncio
async def test_battle_env_async_team_preview_awaits_precomputed_future() -> None:
    obs = {"context": np.array([1, 2, 3], dtype=np.int64)}
    future: Future = Future()
    timing = SimpleNamespace(
        increment=lambda *args, **kwargs: None, add=lambda *args, **kwargs: None
    )
    fake_env = SimpleNamespace(
        _active_preview_obs=obs,
        _active_preview_future=future,
        _team_preview_batcher=None,
        _timing=timing,
        _ext_team_preview_fn=lambda current_obs: (_ for _ in ()).throw(
            AssertionError("fallback should not run")
        ),
        _obs_equal=lambda left, right: BattleEnv._obs_equal(left, right),
        _record_preview_obs_mismatch=lambda actual, prepared: (_ for _ in ()).throw(
            AssertionError("mismatch handler should not run")
        ),
    )

    task = asyncio.create_task(
        BattleEnv._choose_team_preview_picks_async(fake_env, obs)
    )
    await asyncio.sleep(0)
    assert not task.done()

    expected = TeamPreviewResult(
        picks=np.array([0, 1, 2, 3], dtype=np.int64),
        policy_version=12,
    )
    future.set_result(expected)

    result = await asyncio.wait_for(task, timeout=0.2)
    np.testing.assert_array_equal(result.picks, expected.picks)
    assert result.policy_version == 12


@pytest.mark.asyncio
async def test_battle_env_team_preview_records_policy_version() -> None:
    obs = {"context": np.array([1, 2, 3], dtype=np.int64)}
    expected = TeamPreviewResult(
        picks=np.array([0, 1, 2, 3], dtype=np.int64),
        policy_version=21,
    )
    fake_env = SimpleNamespace(
        _ext_team_preview_fn=object(),
        player=SimpleNamespace(obs_builder=SimpleNamespace(build=lambda battle: obs)),
        _choose_team_preview_picks_async=lambda current_obs: asyncio.sleep(
            0,
            result=expected,
        ),
        _episode_team_preview=None,
    )

    team_order = await BattleEnv._do_team_preview(fake_env, battle=object())

    assert team_order == "/team 1234"
    assert fake_env._episode_team_preview == (obs, expected.picks, 21)


def test_finished_battle_update_skips_legality_rebuild(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = []
    timing = SimpleNamespace(
        increment=lambda *args, **kwargs: None, add=lambda *args, **kwargs: None
    )
    fake_player = SimpleNamespace(
        last_battle=None,
        _last_finished_token=None,
        obs_builder=SimpleNamespace(
            build=lambda battle: {"context": np.array([1], dtype=np.int64)}
        ),
        timing=timing,
        obs_queue=SimpleNamespace(put=lambda item: queued.append(item)),
        _pending_action_state="stale",
        _pending_action_battle=object(),
    )
    battle = SimpleNamespace(
        battle_tag="battle-1", turn=7, finished=True, won=True, lost=False
    )

    monkeypatch.setattr(
        showdown_env_module,
        "build_legal_action_state",
        lambda battle: (_ for _ in ()).throw(
            AssertionError("finished battles should not rebuild legality")
        ),
    )

    ShowdownPlayer._emit_battle_update(fake_player, battle)

    assert fake_player._pending_action_state is None
    assert fake_player._pending_action_battle is None
    _, mask, _, done, queued_battle = queued[0]
    assert done is True
    assert queued_battle is battle
    assert mask["joint"][FIRST_SLOT_MASK_ROW, PASS_ACTION]
    assert mask["joint"][PASS_ACTION + 1, PASS_ACTION]
    np.testing.assert_array_equal(
        mask["can_tera"], np.array([False, False], dtype=bool)
    )


def test_finished_battle_callback_defers_terminal_observation() -> None:
    queued = []
    fake_player = SimpleNamespace(
        _pending_finished_battle=None,
        obs_queue=SimpleNamespace(put=lambda item: queued.append(item)),
    )
    battle = SimpleNamespace(battle_tag="battle-1", turn=7, finished=True)

    ShowdownPlayer._battle_finished_callback(fake_player, battle)

    assert queued == []
    assert fake_player._pending_finished_battle is battle


def test_emit_pending_finished_battle_enqueues_terminal_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    queued = []
    timing = SimpleNamespace(
        increment=lambda *args, **kwargs: None,
        add=lambda *args, **kwargs: None,
    )
    fake_player = SimpleNamespace(
        last_battle=None,
        _last_finished_token=None,
        _pending_finished_battle=None,
        obs_builder=SimpleNamespace(
            build=lambda battle: {"context": np.array([1], dtype=np.int64)}
        ),
        timing=timing,
        obs_queue=SimpleNamespace(put=lambda item: queued.append(item)),
        _pending_action_state=None,
        _pending_action_battle=None,
    )
    fake_player._emit_battle_update = lambda battle, request_id=None: (
        ShowdownPlayer._emit_battle_update(
            fake_player,
            battle,
            request_id=request_id,
        )
    )
    battle = SimpleNamespace(
        battle_tag="battle-1", turn=7, finished=True, won=True, lost=False
    )
    fake_player._pending_finished_battle = battle

    monkeypatch.setattr(
        showdown_env_module,
        "build_legal_action_state",
        lambda battle: (_ for _ in ()).throw(
            AssertionError("finished battles should not rebuild legality")
        ),
    )

    emitted = ShowdownPlayer.emit_pending_finished_battle(fake_player)

    assert emitted is True
    assert fake_player._pending_finished_battle is None
    _, _mask, reward, done, queued_battle = queued[0]
    assert reward == 1.0
    assert done is True
    assert queued_battle is battle


def test_league_player_finished_battle_cancels_pending_inference() -> None:
    battle = SimpleNamespace(finished=True)
    future: Future = Future()
    fake_player = SimpleNamespace(
        _pending_action_future=future,
        _pending_action_battle=battle,
    )

    LeaguePlayer._battle_finished_callback(fake_player, battle)

    assert future.cancelled()


@pytest.mark.asyncio
async def test_league_player_team_preview_propagates_precompute_failure() -> None:
    obs = {"context": np.array([1, 2, 3], dtype=np.int64)}
    future: Future = Future()
    future.set_exception(RuntimeError("opponent preview batch failed"))
    fake_player = SimpleNamespace(
        obs_builder=SimpleNamespace(build=lambda battle: obs),
        _prepared_preview_obs=obs,
        _prepared_preview_future=future,
        _increment_timing=lambda *args, **kwargs: None,
        _add_timing=lambda *args, **kwargs: None,
        _obs_equal=lambda left, right: LeaguePlayer._obs_equal(left, right),
        _record_preview_obs_mismatch=lambda actual, prepared: (_ for _ in ()).throw(
            AssertionError("mismatch handler should not run")
        ),
        inference_service=SimpleNamespace(
            infer_team_preview=lambda checkpoint_path, current_obs: (
                _ for _ in ()
            ).throw(AssertionError("fallback should not run"))
        ),
        checkpoint_path="checkpoint.pt",
    )

    with pytest.raises(RuntimeError, match="opponent preview batch failed"):
        await LeaguePlayer._teampreview(fake_player, battle=object())


@pytest.mark.asyncio
async def test_league_player_keeps_recurrent_state_on_same_state_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeInferenceService:
        def __init__(self) -> None:
            self.hidden_inputs: list[torch.Tensor | None] = []
            self.step = 0

        def infer_action(self, checkpoint_path, obs, masks, hidden, timing=None):
            return self.submit_action(
                checkpoint_path, obs, masks, hidden, timing=timing
            ).result()

        def submit_action(self, checkpoint_path, obs, masks, hidden, timing=None):
            self.step += 1
            self.hidden_inputs.append(None if hidden is None else hidden[0].clone())
            next_value = float(10 + self.step)
            next_hidden = (
                torch.full((1, 1, 1), next_value),
                torch.full((1, 1, 1), next_value + 100),
            )
            future: Future = Future()
            future.set_result(
                (
                    np.array([PASS_ACTION, PASS_ACTION, TERA_NONE], dtype=np.int64),
                    next_hidden,
                )
            )
            return future

    fake_service = _FakeInferenceService()
    build_calls = []

    def _build_obs(battle):
        build_calls.append(battle.state)
        return {"context": np.array([battle.state], dtype=np.int64)}

    player = LeaguePlayer.__new__(LeaguePlayer)
    player.checkpoint_path = "checkpoint.pt"
    player.inference_service = fake_service
    player.obs_builder = SimpleNamespace(build=_build_obs)
    player._hidden = (
        torch.full((1, 1, 1), 1.0),
        torch.full((1, 1, 1), 101.0),
    )
    player._pending_hidden = None
    player._pending_decision_key = None
    player._inflight_decision_key = None
    player._pending_order = None
    player._pending_action_future = None
    player._pending_action_battle = None
    player._pending_action_started_at = None
    player._last_actions = None
    player.choose_default_move = lambda: PassBattleOrder()
    monkeypatch.setattr(
        league_module,
        "build_legal_action_state",
        lambda battle: SimpleNamespace(
            joint_mask=np.pad(
                np.array([[True] + [False] * (ACTIONS_PER_SLOT - 1)], dtype=bool),
                ((0, ACTIONS_PER_SLOT), (0, 0)),
                constant_values=False,
            ),
            can_tera=np.array([False, False], dtype=bool),
            slot_action_orders=({}, {}),
            slot_masks=(
                np.zeros(ACTIONS_PER_SLOT, dtype=bool),
                np.zeros(ACTIONS_PER_SLOT, dtype=bool),
            ),
        ),
    )
    monkeypatch.setattr(
        league_module,
        "action_to_order",
        lambda action_idx, battle, slot, terastallize=False, action_state=None: (
            PassBattleOrder()
        ),
    )

    battle_kwargs = dict(
        finished=False,
        battle_tag="battle-hidden",
        active_pokemon=[SimpleNamespace(species="a"), SimpleNamespace(species="b")],
        opponent_active_pokemon=[
            SimpleNamespace(species="c"),
            SimpleNamespace(species="d"),
        ],
        force_switch=[False, False],
        can_tera=[False, False],
        trapped=[False, False],
        last_request={"active": [{"moves": []}, {"moves": []}]},
    )
    retry_battle = SimpleNamespace(state=1, turn=1, **battle_kwargs)
    next_battle = SimpleNamespace(state=2, turn=2, **battle_kwargs)

    await LeaguePlayer.choose_move(player, retry_battle)
    await LeaguePlayer.choose_move(player, retry_battle)
    await LeaguePlayer.choose_move(player, next_battle)

    assert fake_service.step == 2
    assert build_calls == [1, 2]
    assert torch.equal(fake_service.hidden_inputs[0], torch.full((1, 1, 1), 1.0))
    assert torch.equal(fake_service.hidden_inputs[1], torch.full((1, 1, 1), 11.0))


@pytest.mark.asyncio
async def test_league_player_reuses_inflight_same_state_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeInferenceService:
        def __init__(self) -> None:
            self.future: Future = Future()
            self.submit_calls = 0
            self.hidden_inputs: list[torch.Tensor | None] = []

        def submit_action(self, checkpoint_path, obs, masks, hidden, timing=None):
            self.submit_calls += 1
            self.hidden_inputs.append(None if hidden is None else hidden[0].clone())
            return self.future

    fake_service = _FakeInferenceService()
    build_calls = []

    def _build_obs(battle):
        build_calls.append(battle.state)
        return {"context": np.array([battle.state], dtype=np.int64)}

    player = LeaguePlayer.__new__(LeaguePlayer)
    player.checkpoint_path = "checkpoint.pt"
    player.inference_service = fake_service
    player.obs_builder = SimpleNamespace(build=_build_obs)
    player._hidden = (
        torch.full((1, 1, 1), 1.0),
        torch.full((1, 1, 1), 101.0),
    )
    player._pending_hidden = None
    player._pending_decision_key = None
    player._inflight_decision_key = None
    player._pending_order = None
    player._pending_action_future = None
    player._pending_action_battle = None
    player._pending_action_started_at = None
    player._last_actions = None
    player._timing = None
    player.choose_default_move = lambda: PassBattleOrder()
    monkeypatch.setattr(
        league_module,
        "build_legal_action_state",
        lambda battle: SimpleNamespace(
            joint_mask=np.pad(
                np.array([[True] + [False] * (ACTIONS_PER_SLOT - 1)], dtype=bool),
                ((0, ACTIONS_PER_SLOT), (0, 0)),
                constant_values=False,
            ),
            can_tera=np.array([False, False], dtype=bool),
            slot_action_orders=({}, {}),
            slot_masks=(
                np.zeros(ACTIONS_PER_SLOT, dtype=bool),
                np.zeros(ACTIONS_PER_SLOT, dtype=bool),
            ),
        ),
    )
    monkeypatch.setattr(
        league_module,
        "action_to_order",
        lambda action_idx, battle, slot, terastallize=False, action_state=None: (
            PassBattleOrder()
        ),
    )

    battle = SimpleNamespace(
        state=1,
        finished=False,
        battle_tag="battle-inflight",
        turn=1,
        active_pokemon=[SimpleNamespace(species="a"), SimpleNamespace(species="b")],
        opponent_active_pokemon=[
            SimpleNamespace(species="c"),
            SimpleNamespace(species="d"),
        ],
        force_switch=[False, False],
        can_tera=[False, False],
        trapped=[False, False],
        last_request={"active": [{"moves": []}, {"moves": []}]},
    )

    first = asyncio.create_task(LeaguePlayer.choose_move(player, battle))
    await asyncio.sleep(0)
    second = asyncio.create_task(LeaguePlayer.choose_move(player, battle))
    await asyncio.sleep(0)

    assert fake_service.submit_calls == 1
    assert build_calls == [1]

    next_hidden = (
        torch.full((1, 1, 1), 11.0),
        torch.full((1, 1, 1), 111.0),
    )
    fake_service.future.set_result(
        (np.array([PASS_ACTION, PASS_ACTION, TERA_NONE], dtype=np.int64), next_hidden)
    )
    order1, order2 = await asyncio.gather(first, second)

    assert isinstance(order1, PassBattleOrder)
    assert order2 is order1
    assert fake_service.submit_calls == 1
    assert torch.equal(fake_service.hidden_inputs[0], torch.full((1, 1, 1), 1.0))
    assert torch.equal(player._pending_hidden[0], torch.full((1, 1, 1), 11.0))


@pytest.mark.asyncio
async def test_league_player_default_only_request_skips_inference() -> None:
    class _FakeInferenceService:
        def submit_action(self, *args, **kwargs):
            raise AssertionError(
                "default-only request should not submit league inference"
            )

    player = LeaguePlayer.__new__(LeaguePlayer)
    player.checkpoint_path = "checkpoint.pt"
    player.inference_service = _FakeInferenceService()
    player.obs_builder = SimpleNamespace(
        build=lambda battle: (_ for _ in ()).throw(
            AssertionError("default-only request should not build neural observations")
        )
    )
    player._hidden = (
        torch.full((1, 1, 1), 3.0),
        torch.full((1, 1, 1), 103.0),
    )
    player._pending_hidden = None
    player._pending_decision_key = None
    player._inflight_decision_key = None
    player._pending_order = None
    player._pending_action_future = None
    player._pending_action_battle = None
    player._pending_action_started_at = None
    player._last_actions = None
    player._timing = None
    player.choose_default_move = lambda: PassBattleOrder()

    order = await LeaguePlayer.choose_move(player, _DefaultOnlyBattle())

    assert isinstance(order, PassBattleOrder)
    assert player._pending_order is order
    np.testing.assert_array_equal(
        player._last_actions,
        np.array([PASS_ACTION, PASS_ACTION, TERA_NONE], dtype=np.int64),
    )
    assert torch.equal(player._pending_hidden[0], player._hidden[0])


def test_league_player_decision_signature_tracks_action_request_fields() -> None:
    request = {
        "forceSwitch": [False, False],
        "side": {"pokemon": [{"ident": "p1: A", "condition": "100/100"}]},
        "active": [
            {
                "canTerastallize": True,
                "moves": [
                    {"id": "protect", "target": "self", "disabled": False},
                    {"id": "earthquake", "target": "allAdjacent", "disabled": False},
                ],
            },
            {
                "moves": [
                    {"id": "spore", "target": "normal", "disabled": True},
                ],
            },
        ],
    }

    base_signature = LeaguePlayer._request_decision_signature(request)

    unrelated_side_update = copy.deepcopy(request)
    unrelated_side_update["side"]["pokemon"][0]["condition"] = "1/100"
    assert (
        LeaguePlayer._request_decision_signature(unrelated_side_update)
        == base_signature
    )

    fainted_side_update = copy.deepcopy(request)
    fainted_side_update["side"]["pokemon"][0]["condition"] = "0 fnt"
    assert (
        LeaguePlayer._request_decision_signature(fainted_side_update) != base_signature
    )

    changed_target = copy.deepcopy(request)
    changed_target["active"][0]["moves"][1]["target"] = "adjacentFoe"
    assert LeaguePlayer._request_decision_signature(changed_target) != base_signature

    changed_disabled = copy.deepcopy(request)
    changed_disabled["active"][1]["moves"][0]["disabled"] = False
    assert LeaguePlayer._request_decision_signature(changed_disabled) != base_signature

    changed_force_switch = copy.deepcopy(request)
    changed_force_switch["forceSwitch"] = [True, False]
    assert (
        LeaguePlayer._request_decision_signature(changed_force_switch) != base_signature
    )
