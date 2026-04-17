import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    PASS_ACTION,
    SWITCH_OFFSET,
    TERA_OFFSET,
    action_to_order,
    build_action_mask,
)
from pokerl.model.heads import ActionHead, ValueHead
from pokerl.model.memory import IdentityMemory
from pokerl.training.ppo import PPOTrainer
from pokerl.training.rollout import RolloutBuffer


def _obs(step: int, num_envs: int) -> dict[str, np.ndarray]:
    return {
        "float_obs": np.full((num_envs, 3), float(step), dtype=np.float32),
        "index_obs": np.full((num_envs, 2), step, dtype=np.int64),
    }


def _mask(num_envs: int) -> np.ndarray:
    mask = np.zeros((num_envs, 2, 4), dtype=bool)
    mask[..., 0] = True
    mask[..., 1] = True
    return mask


def _obs_to_tensor(obs: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.from_numpy(value).to(device) for key, value in obs.items()}


def test_rollout_buffer_keeps_sequence_tensors_on_device_and_clones_initial_hidden() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    buffer = RolloutBuffer(num_steps, num_envs, _obs(0, num_envs), _mask(num_envs), device)

    hidden_h = torch.arange(8, dtype=torch.float32, device=device).reshape(1, num_envs, 4)
    hidden_c = hidden_h + 100
    expected_h = hidden_h.squeeze(0).clone()
    expected_c = hidden_c.squeeze(0).clone()

    buffer.reset(initial_hidden=(hidden_h, hidden_c))
    hidden_h.zero_()
    hidden_c.zero_()

    for step in range(num_steps):
        obs_t = _obs_to_tensor(_obs(step, num_envs), device)
        actions_t = torch.tensor([[step, step + 1], [step + 2, step + 3]], dtype=torch.long, device=device)
        masks_t = torch.from_numpy(_mask(num_envs)).to(device)
        log_probs_t = torch.full((num_envs,), float(step), dtype=torch.float32, device=device)
        values_t = torch.full((num_envs,), float(step) + 0.5, dtype=torch.float32, device=device)
        rewards_t = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        dones_t = torch.tensor([False, step == num_steps - 1], dtype=torch.bool, device=device)
        buffer.add(obs_t, actions_t, masks_t, log_probs_t, values_t, rewards_t, dones_t)

    batch = buffer.compute_gae(torch.zeros(num_envs, dtype=torch.float32, device=device))

    assert batch.obs["float_obs"].shape == (num_steps, num_envs, 3)
    assert batch.obs["index_obs"].shape == (num_steps, num_envs, 2)
    assert batch.actions.shape == (num_steps, num_envs, 2)
    assert batch.action_masks.shape == (num_steps, num_envs, 2, 4)
    assert batch.advantages.shape == (num_steps, num_envs)
    assert batch.obs["float_obs"].device == device
    assert torch.equal(batch.initial_hidden_h, expected_h)
    assert torch.equal(batch.initial_hidden_c, expected_c)


def test_rollout_buffer_requires_a_complete_rollout_before_gae() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    buffer = RolloutBuffer(num_steps, num_envs, _obs(0, num_envs), _mask(num_envs), device)
    buffer.reset()

    buffer.add(
        _obs_to_tensor(_obs(0, num_envs), device),
        torch.zeros((num_envs, 2), dtype=torch.long, device=device),
        torch.from_numpy(_mask(num_envs)).to(device),
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

    def forward(
        self,
        obs: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        context = self.context_proj(obs["context"].float())
        active = self.active_proj(obs["active"].float())
        return context, active, active


class _DummyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = _DummySpatial()
        self.memory = IdentityMemory(input_dim=3, hidden_dim=3)
        self.action_head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
        self.value_head = ValueHead(context_dim=3, hidden_dim=8)

    def forward(
        self,
        obs: dict[str, torch.Tensor],
        hidden=None,
        actions: torch.Tensor | None = None,
        action_masks: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        context, active_reps, _ = self.spatial(obs)
        context, hidden = self.memory(context, hidden)
        action_out = self.action_head(context, active_reps, action_masks, actions)
        return {
            **action_out,
            "value": self.value_head(context),
            "hidden": hidden,
        }

    def initial_hidden(self, batch_size: int):
        return self.memory.initial_hidden(batch_size)


def _ppo_obs(step: int, num_envs: int) -> dict[str, np.ndarray]:
    return {
        "context": np.full((num_envs, 3), float(step), dtype=np.float32),
        "active": np.full((num_envs, 2, 2), float(step + 1), dtype=np.float32),
    }


def test_ppo_trainer_updates_from_device_resident_sequence_batch() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    action_mask = np.ones((num_envs, 2, ACTIONS_PER_SLOT), dtype=bool)
    agent = _DummyAgent().to(device)
    config = TrainingConfig(num_epochs=1, num_minibatches=2)
    trainer = PPOTrainer(agent, config, device)
    buffer = RolloutBuffer(num_steps, num_envs, _ppo_obs(0, num_envs), action_mask, device)
    buffer.reset(initial_hidden=agent.initial_hidden(num_envs))

    for step in range(num_steps):
        obs_t = _obs_to_tensor(_ppo_obs(step, num_envs), device)
        masks_t = torch.from_numpy(action_mask).to(device)
        with torch.no_grad():
            out = agent(obs_t, action_masks=masks_t)

        rewards_t = torch.tensor([1.0, -0.5], dtype=torch.float32, device=device)
        dones_t = torch.tensor([False, step == num_steps - 1], dtype=torch.bool, device=device)
        buffer.add(obs_t, out["actions"], masks_t, out["log_prob"], out["value"], rewards_t, dones_t)

    batch = buffer.compute_gae(torch.zeros(num_envs, dtype=torch.float32, device=device))
    metrics = trainer.update(batch)

    assert trainer.global_step == num_steps * num_envs
    assert set(metrics) == {"policy_loss", "value_loss", "entropy", "approx_kl"}


def test_action_head_masks_double_tera_for_second_slot() -> None:
    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    masks = torch.zeros((1, 2, ACTIONS_PER_SLOT), dtype=torch.bool)
    masks[0, 0, TERA_OFFSET] = True
    masks[0, 1, 0] = True
    masks[0, 1, TERA_OFFSET] = True

    out = head(context, active_reps, masks)

    assert out["actions"][0, 0].item() == TERA_OFFSET
    assert out["actions"][0, 1].item() == 0


def test_action_head_masks_duplicate_switch_target_for_second_slot() -> None:
    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    masks = torch.zeros((1, 2, ACTIONS_PER_SLOT), dtype=torch.bool)
    shared_switch = SWITCH_OFFSET + 5
    masks[0, 0, shared_switch] = True
    masks[0, 1, shared_switch] = True
    masks[0, 1, 0] = True

    out = head(context, active_reps, masks)

    assert out["actions"][0, 0].item() == shared_switch
    assert out["actions"][0, 1].item() == 0


class _FakePokemon:
    def __init__(self, species: str):
        self.species = species


class _FakeBattle:
    def __init__(self) -> None:
        self.force_switch = [True, False]
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
        self.available_switches = [[bench_one], []]


def test_switch_actions_use_stable_team_slots() -> None:
    battle = _FakeBattle()

    mask = build_action_mask(battle, slot=0)
    expected_team_slot = 3  # [lead-one, lead-two, bench-two, bench-one, None, None]
    switch_action = SWITCH_OFFSET + expected_team_slot

    assert mask[switch_action]
    assert mask[SWITCH_OFFSET:PASS_ACTION].sum() == 1

    order = action_to_order(switch_action, battle, slot=0)
    assert order.order is battle.available_switches[0][0]
