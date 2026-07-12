import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    TERA_NONE,
    TERA_SLOT_1,
)
from pokerl.env.observation import (
    FIELD_TIMER_CAPS,
    FIELD_TURN_OFFSET,
    ROLE_ACTIVE_1,
    ROLE_ACTIVE_2,
    ROLE_BENCH,
)
from pokerl.model.heads import (
    ActionHead,
    ValueHead,
    _sample_categorical_logits,
)
from pokerl.model.memory import IdentityMemory
from pokerl.training.ppo import PPOTrainer
from pokerl.training.rollout import RolloutBuffer


def test_tera_behavior_distribution_matches_exact_two_stage_marginal() -> None:
    logits = torch.tensor(
        [
            [2.0, 0.5, -1.0],
            [-0.5, 1.0, 0.25],
            [3.0, -4.0, 8.0],
        ],
        dtype=torch.double,
        requires_grad=True,
    )
    valid = torch.tensor(
        [
            [True, True, True],
            [True, True, True],
            [True, False, False],
        ]
    )
    early = torch.tensor([True, False, True])
    epsilon = 0.30
    delay = 0.40

    log_probs, probs = ActionHead._tera_behavior_distribution(
        logits,
        valid,
        early_turn=early,
        exploration_prob=epsilon,
        force_delay_prob=delay,
    )

    base = torch.softmax(logits[:2], dim=-1)
    mixed = (1.0 - epsilon) * base + epsilon / 3.0
    expected_early = torch.cat(
        (
            (mixed[0, TERA_NONE] + delay * (1.0 - mixed[0, TERA_NONE])).view(1),
            mixed[0, 1:] * (1.0 - delay),
        )
    )
    expected = torch.stack(
        (
            expected_early,
            mixed[1],
            torch.tensor([1.0, 0.0, 0.0], dtype=torch.double),
        )
    )

    torch.testing.assert_close(probs, expected)
    torch.testing.assert_close(log_probs[:2].exp(), expected[:2])
    assert log_probs[2, TERA_NONE].item() == 0.0
    assert torch.isneginf(log_probs[2, 1:]).all()
    torch.testing.assert_close(
        ActionHead._entropy_from_probs(probs),
        torch.distributions.Categorical(probs=expected).entropy(),
    )

    loss = -log_probs[0, TERA_SLOT_1] - log_probs[1, TERA_NONE]
    loss = loss + ActionHead._entropy_from_probs(probs).sum()
    loss.backward()
    assert torch.isfinite(logits.grad).all()


def test_tera_behavior_sampler_matches_transformed_distribution() -> None:
    logits = torch.tensor([[1.5, 0.25, -0.75]], dtype=torch.float32)
    valid = torch.ones_like(logits, dtype=torch.bool)
    log_probs, expected_probs = ActionHead._tera_behavior_distribution(
        logits,
        valid,
        early_turn=torch.tensor([True]),
        exploration_prob=0.35,
        force_delay_prob=0.45,
    )

    torch.manual_seed(17)
    num_samples = 50_000
    samples = _sample_categorical_logits(log_probs.expand(num_samples, -1))
    empirical = torch.bincount(samples, minlength=3).float() / num_samples

    torch.testing.assert_close(empirical, expected_probs[0], atol=0.01, rtol=0.0)


class _FixedTeraScore(nn.Module):
    def __init__(self, logits: tuple[float, float, float]):
        super().__init__()
        self.register_buffer("logits", torch.tensor(logits, dtype=torch.float32))

    def forward(self, query: torch.Tensor) -> torch.Tensor:
        return self.logits.to(dtype=query.dtype).expand(query.shape[0], -1)


def _minimal_head_inputs(batch_size: int = 1):
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    head.tera_score = _FixedTeraScore((-0.5, 1.5, 0.25))
    context = torch.zeros((batch_size, 3))
    active_reps = torch.zeros((batch_size, 2, 2))
    own_team_reps = torch.zeros((batch_size, 6, 2))
    opp_team_reps = torch.zeros((batch_size, 6, 2))
    opp_reserve_mask = torch.tensor(
        [[True, True, False, False, False, False]], dtype=torch.bool
    ).expand(batch_size, -1)
    opp_active_reps = torch.zeros((batch_size, 2, 2))
    masks = {
        "joint": torch.zeros(
            (batch_size, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
            dtype=torch.bool,
        ),
        "can_tera": torch.ones((batch_size, 2), dtype=torch.bool),
    }
    masks["joint"][:, FIRST_SLOT_MASK_ROW, 0] = True
    masks["joint"][:, 1, 0] = True
    battle_obs = {
        "field": torch.zeros((batch_size, FIELD_TURN_OFFSET + 1), dtype=torch.float32)
    }
    battle_obs["field"][:, FIELD_TURN_OFFSET] = 1.0 / float(FIELD_TIMER_CAPS["turn"])
    return (
        head,
        context,
        active_reps,
        own_team_reps,
        opp_team_reps,
        opp_reserve_mask,
        opp_active_reps,
        masks,
        battle_obs,
    )


def test_rollout_and_supplied_action_use_identical_tera_behavior_stats() -> None:
    inputs = _minimal_head_inputs()
    head = inputs[0]
    torch.manual_seed(23)
    rollout = head(
        *inputs[1:8],
        battle_obs=inputs[8],
        tera_exploration_prob=0.25,
        tera_force_delay_prob=0.60,
        tera_force_delay_max_turn=3,
    )
    evaluated = head(
        *inputs[1:8],
        actions=rollout["actions"],
        battle_obs=inputs[8],
        tera_exploration_prob=0.25,
        tera_force_delay_prob=0.60,
        tera_force_delay_max_turn=3,
    )

    torch.testing.assert_close(evaluated["log_prob"], rollout["log_prob"])
    torch.testing.assert_close(evaluated["entropy"], rollout["entropy"])
    expected_base = torch.distributions.Categorical(logits=head.tera_score.logits)
    torch.testing.assert_close(
        rollout["base_tera_entropy"],
        expected_base.entropy().expand_as(rollout["base_tera_entropy"]),
    )
    torch.testing.assert_close(
        rollout["base_tera_none_prob"],
        expected_base.probs[TERA_NONE].expand_as(rollout["base_tera_none_prob"]),
    )
    assert rollout["base_tera_has_real_choice"].all()


def test_greedy_action_ignores_training_exploration_transform() -> None:
    inputs = _minimal_head_inputs()
    out = inputs[0](
        *inputs[1:8],
        battle_obs=inputs[8],
        greedy=True,
        tera_exploration_prob=1.0,
        tera_force_delay_prob=1.0,
        tera_force_delay_max_turn=3,
    )

    assert out["actions"][0, 2].item() == TERA_SLOT_1


def test_zero_force_delay_turn_limit_disables_delay_even_at_turn_zero() -> None:
    inputs = _minimal_head_inputs()
    inputs[8]["field"][:, FIELD_TURN_OFFSET] = 0.0

    early = ActionHead._early_turn_mask(
        battle_obs=inputs[8],
        batch_size=1,
        device=torch.device("cpu"),
        max_turn=0,
    )

    assert not early.any()


def test_rollout_buffer_carries_fixed_tera_behavior_policy() -> None:
    obs = {"x": np.zeros((2, 1), dtype=np.float32)}
    masks = {"legal": np.ones((2, 1), dtype=np.bool_)}
    buffer = RolloutBuffer(1, 2, obs, masks, torch.device("cpu"))
    buffer.reset()
    buffer.set_tera_behavior_policy(
        exploration_prob=0.2,
        force_delay_prob=0.3,
        force_delay_max_turn=4,
    )
    zeros = torch.zeros(2)
    buffer.add(
        {"x": torch.zeros((2, 1))},
        torch.zeros((2, 3), dtype=torch.long),
        {"legal": torch.ones((2, 1), dtype=torch.bool)},
        zeros,
        zeros,
        zeros,
        torch.zeros(2, dtype=torch.bool),
    )
    batch = buffer.compute_gae(zeros)

    assert batch.tera_exploration_prob == pytest.approx(0.2)
    assert batch.tera_force_delay_prob == pytest.approx(0.3)
    assert batch.tera_force_delay_max_turn == 4
    with pytest.raises(RuntimeError, match="cannot change"):
        buffer.set_tera_behavior_policy(
            exploration_prob=0.1,
            force_delay_prob=0.1,
            force_delay_max_turn=1,
        )


class _TinySpatial(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_proj = nn.Linear(3, 3)
        self.pokemon_proj = nn.Linear(2, 2)

    def forward(self, obs):
        context = self.context_proj(obs["context"].float())
        active = self.pokemon_proj(obs["active"].float())
        own = self.pokemon_proj(obs["own"].float())
        opp = self.pokemon_proj(obs["opp"].float())
        opp_active = self.pokemon_proj(obs["opp_active"].float())
        return context, active, own, opp, opp_active


class _TinyAgent(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial = _TinySpatial()
        self.memory = IdentityMemory(input_dim=3, hidden_dim=3)
        self.action_head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
        self.action_head.tera_score = _FixedTeraScore((-0.5, 1.5, 0.25))
        self.value_head = ValueHead(context_dim=3, pokemon_dim=2, hidden_dim=8)

    def forward(
        self,
        obs,
        hidden=None,
        *,
        action_masks,
        tera_exploration_prob=0.0,
        tera_force_delay_prob=0.0,
        tera_force_delay_max_turn=0,
    ):
        context, active, own, opp, opp_active = self.spatial(obs)
        context, hidden = self.memory(context, hidden)
        own_bench_mask = (obs["own_roles"] != ROLE_BENCH) | obs["own_mask"]
        opp_bench_mask = (obs["opp_roles"] != ROLE_BENCH) | obs["opp_mask"]
        action_out = self.action_head(
            context,
            active,
            own,
            opp,
            opp_bench_mask,
            opp_active,
            action_masks,
            battle_obs=obs,
            tera_exploration_prob=tera_exploration_prob,
            tera_force_delay_prob=tera_force_delay_prob,
            tera_force_delay_max_turn=tera_force_delay_max_turn,
        )
        return {
            **action_out,
            "value": self.value_head(
                context,
                own,
                opp,
                own_bench_mask,
                opp_bench_mask,
            ),
            "hidden": hidden,
        }

    def initial_hidden(self, batch_size: int):
        return self.memory.initial_hidden(batch_size)


def _ppo_obs(num_envs: int) -> dict[str, np.ndarray]:
    roles = np.broadcast_to(
        np.asarray(
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
        (num_envs, 6),
    ).copy()
    field = np.zeros((num_envs, FIELD_TURN_OFFSET + 1), dtype=np.float32)
    field[:, FIELD_TURN_OFFSET] = 1.0 / float(FIELD_TIMER_CAPS["turn"])
    return {
        "context": np.zeros((num_envs, 3), dtype=np.float32),
        "active": np.zeros((num_envs, 2, 2), dtype=np.float32),
        "own": np.zeros((num_envs, 6, 2), dtype=np.float32),
        "opp": np.zeros((num_envs, 6, 2), dtype=np.float32),
        "opp_active": np.zeros((num_envs, 2, 2), dtype=np.float32),
        "own_roles": roles.copy(),
        "opp_roles": roles.copy(),
        "own_mask": np.zeros((num_envs, 6), dtype=np.bool_),
        "opp_mask": np.zeros((num_envs, 6), dtype=np.bool_),
        "field": field,
    }


def _ppo_masks(num_envs: int) -> dict[str, np.ndarray]:
    joint = np.zeros((num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=np.bool_)
    joint[:, FIRST_SLOT_MASK_ROW, 0] = True
    joint[:, 1, 0] = True
    return {
        "joint": joint,
        "can_tera": np.ones((num_envs, 2), dtype=np.bool_),
    }


def test_ppo_initial_ratio_is_one_under_rollout_behavior_policy() -> None:
    torch.manual_seed(31)
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    obs_np = _ppo_obs(num_envs)
    masks_np = _ppo_masks(num_envs)
    agent = _TinyAgent()
    buffer = RolloutBuffer(num_steps, num_envs, obs_np, masks_np, device)
    buffer.reset()
    buffer.set_tera_behavior_policy(
        exploration_prob=0.25,
        force_delay_prob=0.60,
        force_delay_max_turn=3,
    )
    obs = {key: torch.from_numpy(value) for key, value in obs_np.items()}
    masks = {key: torch.from_numpy(value) for key, value in masks_np.items()}
    for step in range(num_steps):
        with torch.no_grad():
            out = agent(
                obs,
                action_masks=masks,
                tera_exploration_prob=0.25,
                tera_force_delay_prob=0.60,
                tera_force_delay_max_turn=3,
            )
        rewards = torch.tensor([1.0, -1.0]) if step else torch.zeros(num_envs)
        dones = (
            torch.tensor([True, True])
            if step
            else torch.zeros(num_envs, dtype=torch.bool)
        )
        buffer.add(
            obs,
            out["actions"],
            masks,
            out["log_prob"],
            out["value"],
            rewards,
            dones,
        )

    batch = buffer.compute_gae(torch.zeros(num_envs))
    trainer = PPOTrainer(
        agent,
        TrainingConfig(
            lr=0.0,
            num_epochs=1,
            num_minibatches=1,
            aux_hidden_info_coef=0.0,
            aux_opp_action_coef=0.0,
        ),
        device,
    )
    metrics = trainer.update(batch)

    assert metrics["approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["pre_ppo_approx_kl"] == pytest.approx(0.0, abs=1e-7)
    assert metrics["pre_ppo_clip_fraction"] == 0.0
    assert metrics["base_tera_real_choice_evaluations"] > 0
    assert 0.0 <= metrics["base_tera_entropy"] <= np.log(3)
    assert 0.0 <= metrics["base_tera_none_prob"] <= 1.0


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_action_head_rejects_invalid_exploration_probabilities(value: float) -> None:
    inputs = _minimal_head_inputs()
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        inputs[0](
            *inputs[1:8],
            battle_obs=inputs[8],
            tera_exploration_prob=value,
        )


@pytest.mark.parametrize(
    "override",
    [
        {"tera_exploration_start": -0.01},
        {"tera_exploration_end": 1.01},
        {"tera_force_delay_start": float("nan")},
        {"tera_force_delay_end": float("inf")},
    ],
)
def test_training_config_rejects_invalid_tera_probabilities(override) -> None:
    with pytest.raises(ValueError, match="within \\[0, 1\\]"):
        TrainingConfig(**override)
