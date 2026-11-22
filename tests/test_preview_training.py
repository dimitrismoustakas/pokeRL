from __future__ import annotations

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.training.ppo import PPOTrainer
from pokerl.training.preview import TeamPreviewEpisodeStart
from pokerl.training.rollout import RolloutBuffer, TeamPreviewRolloutBatch
from tests.test_rollout_buffer import (
    _DummyAgent,
    _mask,
    _obs_to_tensor,
    _ppo_obs,
)


class _JointAgent(_DummyAgent):
    def __init__(self) -> None:
        super().__init__()
        self.preview_policy = nn.Linear(3, 180)
        self.preview_value = nn.Linear(3, 1)
        self.preview_bring = nn.Linear(3, 6)
        self.preview_lead = nn.Linear(3, 36)
        self.preview_batch_sizes: list[int] = []
        self.preview_calls: list[tuple[int, bool, dict[str, object]]] = []
        self.preview_support_size: int | None = None

    def team_preview(
        self,
        obs: dict[str, torch.Tensor],
        *,
        picks: torch.Tensor,
        **_kwargs,
    ) -> dict[str, torch.Tensor]:
        self.preview_batch_sizes.append(int(picks.shape[0]))
        self.preview_calls.append((int(picks.shape[0]), self.training, dict(_kwargs)))
        context, *_ = self.spatial(obs)
        context, _ = self.memory(context)
        joint_logits = self.preview_policy(context)
        if self.preview_support_size is not None:
            action_indices = torch.arange(
                joint_logits.shape[1],
                device=joint_logits.device,
            )
            joint_logits = joint_logits.masked_fill(
                action_indices.unsqueeze(0) >= self.preview_support_size,
                -torch.inf,
            )
        joint_log_probs = joint_logits.log_softmax(dim=1)
        probabilities = joint_log_probs.exp()
        # Test fixtures use the first canonical preview action.
        selected_log_prob = joint_log_probs[:, 0]
        finite_joint_log_probs = joint_log_probs.masked_fill(
            ~torch.isfinite(joint_log_probs),
            0.0,
        )
        entropy = -(probabilities * finite_joint_log_probs).sum(dim=1)
        return {
            "log_prob": selected_log_prob,
            "entropy": entropy,
            "preview_value": self.preview_value(context).squeeze(1),
            "joint_log_probs": joint_log_probs,
            "opponent_bring_logits": self.preview_bring(context),
            "opponent_lead_pair_logits": self.preview_lead(context),
        }


def _joint_rollout(
    agent: _JointAgent,
    *,
    num_steps: int = 2,
    num_envs: int = 3,
    preview_count: int = 5,
) -> object:
    device = torch.device("cpu")
    buffer = RolloutBuffer(
        num_steps,
        num_envs,
        _ppo_obs(0, num_envs),
        _mask(num_envs),
        device,
    )
    buffer.reset(initial_hidden=agent.initial_hidden(num_envs))
    for step in range(num_steps):
        obs = _obs_to_tensor(_ppo_obs(step, num_envs), device)
        masks = _obs_to_tensor(_mask(num_envs), device)
        with torch.no_grad():
            output = agent(obs, action_masks=masks)
        buffer.add(
            obs,
            output["actions"],
            masks,
            output["log_prob"],
            output["value"],
            torch.linspace(-1.0, 1.0, num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
        )

    batch = buffer.compute_gae(
        torch.zeros(num_envs),
        gamma=1.0,
        gae_lambda=0.95,
    )
    if preview_count == 0:
        return batch

    preview_obs = _obs_to_tensor(_ppo_obs(7, preview_count), device)
    picks = torch.tensor([[0, 1, 2, 3]]).expand(preview_count, -1).clone()
    opponent_picks = picks.clone()
    agent.eval()
    with torch.no_grad():
        behavior = agent.team_preview(preview_obs, picks=picks)
    agent.train()
    returns = torch.linspace(-1.0, 1.0, preview_count)
    values = behavior["preview_value"].detach()
    batch.team_preview = TeamPreviewRolloutBatch(
        obs=preview_obs,
        picks=picks,
        log_probs=behavior["log_prob"].detach(),
        values=values,
        opponent_picks=opponent_picks,
        advantages=returns - values,
        returns=returns,
        terminal=torch.arange(preview_count).remainder(2) == 0,
        horizons=torch.arange(preview_count, dtype=torch.long) + 1,
    )
    agent.preview_batch_sizes.clear()
    agent.preview_calls.clear()
    return batch


def test_joint_update_pairs_preview_shards_with_battle_optimizer_steps() -> None:
    torch.manual_seed(13)
    agent = _JointAgent()
    trainer = PPOTrainer(
        agent,
        TrainingConfig(
            lr=3e-4,
            max_grad_norm=0.05,
            num_epochs=2,
            num_minibatches=2,
        ),
        torch.device("cpu"),
    )
    batch = _joint_rollout(agent, preview_count=5)

    metrics = trainer.update(batch)

    assert agent.preview_batch_sizes == [5, 3, 2, 3, 2]
    assert agent.preview_calls[0] == (
        5,
        False,
        {
            "include_value": False,
            "include_policy_stats": True,
            "include_opponent_preview": False,
        },
    )
    assert all(training for _size, training, _kwargs in agent.preview_calls[1:])
    assert metrics["ppo_optimizer_steps"] == 4.0
    assert metrics["team_preview_joint_optimizer_steps"] == 4.0
    assert np.isfinite(metrics["global_grad_norm_pre_clip_mean"])
    assert (
        metrics["global_grad_norm_pre_clip_max"]
        >= metrics["global_grad_norm_pre_clip_mean"]
    )
    assert 0.0 <= metrics["global_grad_clip_fraction"] <= 1.0
    assert metrics["team_preview_samples"] == 5.0
    assert metrics["team_preview_pre_ppo_approx_kl"] == pytest.approx(
        0.0,
        abs=1e-7,
    )
    assert np.isfinite(metrics["team_preview_exact_kl"])
    optimizer_steps = {
        int(state["step"].item()) for state in trainer.optimizer.state.values()
    }
    assert optimizer_steps == {4}


def test_joint_loss_gives_every_battle_and_preview_decision_equal_weight() -> None:
    torch.manual_seed(15)
    agent = _JointAgent()
    trainer = PPOTrainer(
        agent,
        TrainingConfig(
            num_epochs=1,
            num_minibatches=2,
            value_coef=0.0,
            team_preview_value_coef=0.0,
            entropy_coef=0.0,
            tera_gate_entropy_coef=0.0,
            tera_slot_entropy_coef=0.0,
            team_preview_bring_coef=0.0,
            team_preview_lead_coef=0.0,
            aux_hidden_info_coef=0.0,
            aux_opponent_intent_coef=0.0,
        ),
        torch.device("cpu"),
    )
    batch = _joint_rollout(
        agent,
        num_steps=1,
        num_envs=3,
        preview_count=2,
    )
    captured_losses: list[float] = []

    def capture_step(loss: torch.Tensor) -> bool:
        captured_losses.append(float(loss.detach()))
        return True

    trainer._apply_update_step = capture_step
    trainer.update(batch)

    # With an unchanged policy the actor loss is -advantage. Joint
    # normalization makes the sum over all five decisions zero. The common
    # M / (B + P) scale therefore makes the two unequal 3/2 shards sum to zero;
    # averaging each shard independently would overweight the smaller shard.
    assert len(captured_losses) == 2
    assert sum(captured_losses) == pytest.approx(0.0, abs=1e-6)


def test_exact_preview_kl_handles_masked_behavior_actions() -> None:
    torch.manual_seed(17)
    agent = _JointAgent()
    trainer = PPOTrainer(
        agent,
        TrainingConfig(num_epochs=2, num_minibatches=1),
        torch.device("cpu"),
    )
    agent.preview_support_size = 3
    batch = _joint_rollout(agent, num_envs=1, preview_count=2)

    metrics = trainer.update(batch)

    assert np.isfinite(metrics["team_preview_exact_kl"])
    assert metrics["team_preview_exact_kl"] > 0.0


def test_preview_and_battle_value_clipping_are_reported_separately() -> None:
    torch.manual_seed(23)
    agent = _JointAgent()
    trainer = PPOTrainer(
        agent,
        TrainingConfig(num_epochs=1, num_minibatches=1),
        torch.device("cpu"),
    )
    batch = _joint_rollout(agent, num_envs=1, preview_count=2)
    preview = batch.team_preview
    assert preview is not None
    preview.values = preview.values - 1.0

    metrics = trainer.update(batch)

    assert metrics["value_clip_fraction"] == 0.0
    assert metrics["team_preview_value_clip_fraction"] == 1.0


def test_nonfinite_preview_loss_cannot_apply_a_battle_only_step() -> None:
    torch.manual_seed(29)
    agent = _JointAgent()
    trainer = PPOTrainer(
        agent,
        TrainingConfig(num_epochs=1, num_minibatches=1),
        torch.device("cpu"),
    )
    batch = _joint_rollout(agent, num_envs=1, preview_count=2)
    preview = batch.team_preview
    assert preview is not None
    preview.returns[0] = torch.nan
    initial_parameters = copy.deepcopy(agent.state_dict())

    with pytest.raises(RuntimeError, match="skipped every minibatch"):
        trainer.update(batch)

    assert trainer.optimizer.state == {}
    for name, parameter in agent.state_dict().items():
        assert torch.equal(parameter, initial_parameters[name])


def test_team_preview_episode_start_owns_copied_payload() -> None:
    obs_value = np.asarray([4, 5], dtype=np.int64)
    picks = np.asarray([0, 1, 2, 3], dtype=np.int64)
    record = TeamPreviewEpisodeStart(
        obs={"context": obs_value},
        picks=picks,
        log_prob=-1.25,
        value=0.5,
        policy_version=3,
        opponent_picks=np.asarray([3, 2, 1, 0], dtype=np.int64),
        opponent_aux_labels={"opp_item": np.asarray([9], dtype=np.int64)},
    )

    obs_value[0] = 99
    picks[0] = 5
    assert record.obs["context"].tolist() == [4, 5]
    assert record.picks.tolist() == [0, 1, 2, 3]
