import copy

import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import TrainingConfig
from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    SWITCH_OFFSET,
    TERA_NONE,
    TERA_SLOT_1,
)
from pokerl.env.observation import (
    CONTINUOUS_DIM,
    ROLE_ACTIVE_1,
    ROLE_ACTIVE_2,
    ROLE_BENCH,
)
from pokerl.model.heads import (
    ActionHead,
    HiddenInfoHead,
    MaskedSetAttention,
    TeamPreviewHead,
    ValueHead,
)
from pokerl.model.agent import PokeAgent
from pokerl.model.memory import IdentityMemory
from pokerl.training.ppo import PPOTrainer
from pokerl.training.preview import TeamPreviewEpisodeStart
from pokerl.training.rollout import RolloutBuffer


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


def _unbatched_aux(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        key: np.array(value[0], copy=True)
        for key, value in obs.items()
        if key in RolloutBuffer.AUX_OBS_KEYS
    }


def _preview_start(
    *,
    policy_version: int = 3,
    value: float = 0.0,
    marker: int = 0,
    opponent_aux_labels: dict[str, np.ndarray] | None = None,
) -> TeamPreviewEpisodeStart:
    return TeamPreviewEpisodeStart(
        obs={"preview": np.asarray([marker], dtype=np.int64)},
        picks=np.asarray([0, 1, 2, 3], dtype=np.int64),
        log_prob=-0.5,
        value=value,
        policy_version=policy_version,
        opponent_picks=np.asarray([0, 1, 2, 3], dtype=np.int64),
        opponent_aux_labels=opponent_aux_labels,
    )


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
    assert not batch.opponent_intent_slot_masks.any()
    assert not batch.opponent_intent_tera_masks.any()
    assert torch.all(batch.opponent_intent_slot_labels == -1)
    assert torch.all(batch.opponent_intent_tera_labels == -1)
    assert not batch.public_consequence_move_emitted_masks.any()
    assert not batch.public_consequence_nonmove_reason_masks.any()
    assert not batch.public_consequence_move_branch_masks.any()
    assert not batch.public_consequence_route_masks.any()
    assert not batch.public_consequence_interaction_masks.any()
    assert not batch.public_consequence_interaction_reason_masks.any()
    assert not batch.public_consequence_hit_count_masks.any()
    assert not batch.public_consequence_cant_masks.any()
    assert not batch.public_consequence_field_masks.any()
    assert torch.all(batch.public_consequence_nonmove_reason_labels == -1)
    assert torch.all(batch.public_consequence_move_branch_labels == -1)
    assert torch.all(batch.public_consequence_interaction_reason_labels == -1)
    assert torch.all(batch.public_consequence_hit_count_labels == -1)
    assert not batch.public_consequence_cant_targets.any()
    assert not batch.public_consequence_field_targets.any()


def test_rollout_buffer_reset_clears_public_consequence_labels() -> None:
    device = torch.device("cpu")
    obs = _obs(0, 1)
    masks = _mask(1)
    buffer = RolloutBuffer(1, 1, obs, masks, device)
    common = (
        _obs_to_tensor(obs, device),
        torch.zeros((1, 3), dtype=torch.long),
        _obs_to_tensor(masks, device),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
    )
    buffer.reset()
    buffer.add(
        *common,
        public_consequence_move_emitted_targets=torch.ones((1, 2)),
        public_consequence_move_emitted_masks=torch.ones((1, 2), dtype=torch.bool),
        public_consequence_move_branch_labels=torch.tensor([[2, 3]]),
        public_consequence_move_branch_masks=torch.ones((1, 2), dtype=torch.bool),
        public_consequence_cant_targets=torch.ones((1, 2, 2)),
        public_consequence_cant_masks=torch.ones((1, 2, 2), dtype=torch.bool),
        public_consequence_field_targets=torch.ones((1, 2, 33)),
        public_consequence_field_masks=torch.ones((1, 2), dtype=torch.bool),
    )

    buffer.reset()
    buffer.add(*common)
    batch = buffer.compute_gae(torch.zeros(1))

    assert not batch.public_consequence_move_emitted_targets.any()
    assert not batch.public_consequence_move_emitted_masks.any()
    assert torch.all(batch.public_consequence_move_branch_labels == -1)
    assert not batch.public_consequence_move_branch_masks.any()
    assert not batch.public_consequence_cant_targets.any()
    assert not batch.public_consequence_cant_masks.any()
    assert not batch.public_consequence_field_targets.any()
    assert not batch.public_consequence_field_masks.any()


def test_rollout_buffer_uses_episode_start_aux_labels_online() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 1
    buffer = RolloutBuffer(
        num_steps, num_envs, _aux_obs(num_envs, 1), _mask(num_envs), device
    )
    buffer.reset()
    buffer.register_team_preview_starts(
        [
            (
                0,
                _preview_start(
                    opponent_aux_labels=_unbatched_aux(_aux_obs(num_envs, 9))
                ),
            )
        ],
        policy_version=3,
    )

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

    exact = _aux_obs(1, 1)
    exact["opp_source_index"][0] = [0, 1, 2, -1, -1, -1]
    exact["opp_item"][0] = [100, 200, 300, 0, 0, 0]
    exact["opp_mask"][0] = [False, False, False, True, True, True]
    buffer.register_team_preview_starts(
        [(0, _preview_start(opponent_aux_labels=_unbatched_aux(exact)))],
        policy_version=3,
    )

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
    buffer.add(
        _obs_to_tensor(switched, device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([True]),
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


def test_rollout_buffer_switches_online_aux_labels_at_episode_start() -> None:
    device = torch.device("cpu")
    buffer = RolloutBuffer(3, 2, _aux_obs(2, 1), _mask(2), device)
    buffer.reset()
    actions = torch.zeros((2, 3), dtype=torch.long)
    masks = _obs_to_tensor(_mask(2), device)
    zeros = torch.zeros(2)

    first_env = _aux_obs(1, 10)
    second_env = _aux_obs(1, 30)
    buffer.register_team_preview_starts(
        [
            (
                0,
                _preview_start(marker=0, opponent_aux_labels=_unbatched_aux(first_env)),
            ),
            (
                1,
                _preview_start(
                    marker=1,
                    opponent_aux_labels=_unbatched_aux(second_env),
                ),
            ),
        ],
        policy_version=3,
    )
    buffer.add(
        _obs_to_tensor(_aux_obs(2, 1), device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([True, False]),
    )
    next_first_env = _aux_obs(1, 20)
    buffer.register_team_preview_starts(
        [
            (
                0,
                _preview_start(
                    marker=2,
                    opponent_aux_labels=_unbatched_aux(next_first_env),
                ),
            )
        ],
        policy_version=3,
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

    buffer.add(
        _obs_to_tensor(_aux_obs(2, 3), device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([True, True]),
    )

    batch = buffer.compute_gae(torch.zeros(2))

    assert batch.aux_labels["opp_item"][:, 0, 0].tolist() == [10, 20, 20]
    assert batch.aux_labels["opp_item"][:, 1, 0].tolist() == [30, 30, 30]
    assert batch.aux_label_valid[:, 0, 0].tolist() == [True, True, True]
    assert batch.aux_label_valid[:, 1, 0].tolist() == [True, True, True]


def test_rollout_buffer_preserves_active_aux_labels_across_rollout_reset() -> None:
    device = torch.device("cpu")
    buffer = RolloutBuffer(1, 1, _aux_obs(1, 1), _mask(1), device)
    buffer.reset()
    buffer.register_team_preview_starts(
        [
            (
                0,
                _preview_start(opponent_aux_labels=_unbatched_aux(_aux_obs(1, 77))),
            )
        ],
        policy_version=3,
    )
    actions = torch.zeros((1, 3), dtype=torch.long)
    masks = _obs_to_tensor(_mask(1), device)
    zeros = torch.zeros(1)

    buffer.add(
        _obs_to_tensor(_aux_obs(1, 1), device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([False]),
    )
    first = buffer.compute_gae(torch.zeros(1))
    assert first.aux_label_valid[0, 0, 0]
    assert first.aux_labels["opp_item"][0, 0, 0].item() == 77

    buffer.reset()
    buffer.add(
        _obs_to_tensor(_aux_obs(1, 2), device),
        actions,
        masks,
        zeros,
        zeros,
        zeros,
        torch.tensor([False]),
    )
    second = buffer.compute_gae(torch.zeros(1))
    assert second.aux_label_valid[0, 0, 0]
    assert second.aux_labels["opp_item"][0, 0, 0].item() == 77


def test_rollout_buffer_preview_returns_stop_at_terminal_and_include_final_reset() -> (
    None
):
    device = torch.device("cpu")
    buffer = RolloutBuffer(4, 1, _obs(0, 1), _mask(1), device)
    buffer.reset()
    buffer.register_team_preview_starts(
        [(0, _preview_start(value=0.25, marker=0))],
        policy_version=3,
    )
    actions = torch.zeros((1, 3), dtype=torch.long)
    masks = _obs_to_tensor(_mask(1), device)
    zeros = torch.zeros(1)
    rewards = [0.0, 1.0, 0.0, -1.0]
    dones = [False, True, False, True]

    for step, (reward, done) in enumerate(zip(rewards, dones, strict=True)):
        buffer.add(
            _obs_to_tensor(_obs(step, 1), device),
            actions,
            masks,
            zeros,
            zeros,
            torch.tensor([reward]),
            torch.tensor([done]),
        )
        if step == 1:
            buffer.register_team_preview_starts(
                [(0, _preview_start(value=-0.25, marker=1))],
                policy_version=3,
            )
        if step == 3:
            buffer.register_team_preview_starts(
                [(0, _preview_start(value=0.1, marker=2))],
                policy_version=3,
            )

    batch = buffer.compute_gae(torch.tensor([0.4]), gamma=1.0, gae_lambda=0.95)
    preview = batch.team_preview
    assert preview is not None
    assert preview.returns.tolist() == pytest.approx([1.0, -1.0, 0.4])
    assert preview.advantages.tolist() == pytest.approx([0.75, -0.75, 0.3])
    assert preview.terminal.tolist() == [True, True, False]
    assert preview.horizons.tolist() == [2, 2, 0]
    assert preview.obs["preview"].reshape(-1).tolist() == [0, 1, 2]


def test_rollout_buffer_preview_boundary_return_excludes_previous_episode_reward() -> (
    None
):
    device = torch.device("cpu")
    buffer = RolloutBuffer(3, 1, _obs(0, 1), _mask(1), device)
    buffer.reset()
    actions = torch.zeros((1, 3), dtype=torch.long)
    masks = _obs_to_tensor(_mask(1), device)
    zeros = torch.zeros(1)

    buffer.add(
        _obs_to_tensor(_obs(0, 1), device),
        actions,
        masks,
        zeros,
        zeros,
        torch.tensor([5.0]),
        torch.tensor([True]),
    )
    buffer.register_team_preview_starts(
        [(0, _preview_start(value=0.125))],
        policy_version=3,
    )
    for step, reward in ((1, 2.0), (2, 3.0)):
        buffer.add(
            _obs_to_tensor(_obs(step, 1), device),
            actions,
            masks,
            zeros,
            zeros,
            torch.tensor([reward]),
            torch.tensor([False]),
        )

    batch = buffer.compute_gae(torch.tensor([7.0]), gamma=0.5, gae_lambda=0.95)
    preview = batch.team_preview
    assert preview is not None
    # Preview reward is zero, so the preview-to-first-battle transition adds
    # one gamma: .5 * (2 + .5 * 3 + .5**2 * 7) = 2.625.
    assert preview.returns.tolist() == pytest.approx([2.625])
    assert preview.advantages.tolist() == pytest.approx([2.5])
    assert preview.terminal.tolist() == [False]
    assert preview.horizons.tolist() == [2]


def test_rollout_buffer_rejects_preview_from_different_policy_version() -> None:
    buffer = RolloutBuffer(1, 1, _obs(0, 1), _mask(1), torch.device("cpu"))
    buffer.reset()

    with pytest.raises(RuntimeError, match="wrong policy version"):
        buffer.register_team_preview_starts(
            [(0, _preview_start(policy_version=2))],
            policy_version=3,
        )

    buffer.add(
        _obs_to_tensor(_obs(0, 1), torch.device("cpu")),
        torch.zeros((1, 3), dtype=torch.long),
        _obs_to_tensor(_mask(1), torch.device("cpu")),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1),
        torch.zeros(1, dtype=torch.bool),
    )
    assert buffer.compute_gae(torch.zeros(1)).team_preview is None


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


class _DummyAgent(PokeAgent):
    def __init__(self) -> None:
        nn.Module.__init__(self)
        self.spatial = _DummySpatial()
        self.memory = IdentityMemory(input_dim=3, hidden_dim=3)
        self.hidden_info_head = HiddenInfoHead(
            pokemon_dim=2,
            context_dim=3,
            num_items=3,
            num_abilities=3,
            num_types=3,
            num_moves=3,
            hidden_dim=4,
        )
        self.action_head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
        self.value_head = ValueHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
        self.belief_condition_calls = 0

    def condition_opponent_reps(
        self,
        opp_reps: torch.Tensor,
        opp_active_reps: torch.Tensor,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.belief_condition_calls += 1
        return super().condition_opponent_reps(opp_reps, opp_active_reps, context)


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
    with torch.no_grad():
        agent.hidden_info_head.policy_belief_adapter.weight.fill_(0.05)
        agent.hidden_info_head.policy_belief_adapter.bias.fill_(-0.02)
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
    agent.belief_condition_calls = 0
    metrics = trainer.update(batch)

    assert trainer.global_step == num_steps * num_envs
    assert {
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "max_approx_kl",
        "clip_fraction",
        "value_clip_fraction",
        "value_explained_variance",
    } <= set(metrics)
    assert metrics["pre_ppo_clip_fraction"] >= 0.0
    assert metrics["pre_ppo_max_abs_log_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["base_tera_real_choice_evaluations"] >= 0.0
    assert metrics["policy_belief_adapter_weight_norm"] == pytest.approx(
        agent.hidden_info_head.policy_belief_adapter.weight.detach().norm().item()
    )
    assert agent.belief_condition_calls == 2


def test_ppo_trainer_updates_public_consequence_model_from_rollout_labels() -> None:
    device = torch.device("cpu")
    num_steps = 2
    num_envs = 2
    action_mask = {
        "joint": np.ones(
            (num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool
        ),
        "can_tera": np.ones((num_envs, 2), dtype=bool),
    }

    def consequence_obs(step: int) -> dict[str, np.ndarray]:
        obs = _ppo_obs(step, num_envs)
        obs["field"] = np.full((num_envs, 36), step / 10.0, dtype=np.float32)
        obs["field"][:, :9] = 0.0
        obs["field"][:, 0] = 1.0
        obs["field"][:, 10:15] = 0.0
        obs["field"][:, 10] = 1.0
        return obs

    agent = _DummyAgent().to(device)
    initial_consequence_weight = (
        agent.action_head.public_consequence_action_head.weight.detach().clone()
    )
    config = TrainingConfig(
        num_epochs=1,
        num_minibatches=1,
        aux_hidden_info_coef=0.0,
        aux_opponent_intent_coef=0.0,
    )
    trainer = PPOTrainer(agent, config, device)
    buffer = RolloutBuffer(
        num_steps,
        num_envs,
        consequence_obs(0),
        action_mask,
        device,
    )
    buffer.reset(initial_hidden=agent.initial_hidden(num_envs))

    for step in range(num_steps):
        obs_t = _obs_to_tensor(consequence_obs(step), device)
        field_targets = torch.full((num_envs, 2, 33), 0.5 + step / 10.0)
        field_targets[..., :9] = 0.0
        field_targets[..., 0] = 1.0
        field_targets[..., 10:15] = 0.0
        field_targets[..., 10] = 1.0
        masks_t = {
            key: torch.from_numpy(value).to(device)
            for key, value in action_mask.items()
        }
        with torch.no_grad():
            out = agent(obs_t, action_masks=masks_t)
        buffer.add(
            obs_t,
            out["actions"],
            masks_t,
            out["log_prob"],
            out["value"],
            torch.tensor([1.0, -1.0]),
            torch.tensor([False, step == num_steps - 1]),
            public_consequence_move_emitted_targets=torch.ones(num_envs, 2),
            public_consequence_move_emitted_masks=torch.ones(
                num_envs, 2, dtype=torch.bool
            ),
            public_consequence_move_branch_labels=torch.tensor(
                [[0, 1], [2, 3]], dtype=torch.long
            ),
            public_consequence_move_branch_masks=torch.ones(
                num_envs, 2, dtype=torch.bool
            ),
            public_consequence_route_targets=torch.randint(
                0, 2, (num_envs, 2, 4), dtype=torch.int64
            ).float(),
            public_consequence_route_masks=torch.ones(num_envs, 2, 4, dtype=torch.bool),
            public_consequence_interaction_targets=torch.randint(
                0, 2, (num_envs, 2, 4), dtype=torch.int64
            ).float(),
            public_consequence_interaction_masks=torch.ones(
                num_envs, 2, 4, dtype=torch.bool
            ),
            public_consequence_cant_targets=torch.tensor(
                [[[0.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [0.0, 0.0]]]
            ),
            public_consequence_cant_masks=torch.ones(num_envs, 2, 2, dtype=torch.bool),
            public_consequence_field_targets=field_targets,
            public_consequence_field_masks=torch.ones(num_envs, 2, dtype=torch.bool),
        )

    metrics = trainer.update(buffer.compute_gae(torch.zeros(num_envs)))

    assert np.isfinite(metrics["aux_public_consequence_loss"])
    assert metrics["aux_public_consequence_emitted_count"] == 8.0
    assert metrics["aux_public_consequence_move_branch_count"] == 8.0
    assert metrics["aux_public_consequence_route_count"] == 32.0
    assert metrics["aux_public_consequence_cant_count"] == 16.0
    assert metrics["aux_public_consequence_field_count"] == 8.0
    assert np.isfinite(metrics["aux_public_consequence_field_mse"])
    assert np.isfinite(metrics["aux_public_consequence_field_persistence_mse"])
    assert metrics["public_consequence_adapter_weight_norm"] >= 0.0
    assert not torch.equal(
        agent.action_head.public_consequence_action_head.weight,
        initial_consequence_weight,
    )


def test_actor_and_value_objectives_reach_belief_adapter() -> None:
    device = torch.device("cpu")
    num_envs = 2
    agent = _DummyAgent().to(device)
    obs = _obs_to_tensor(_ppo_obs(0, num_envs), device)
    masks = {
        "joint": torch.ones(
            (num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
            dtype=torch.bool,
        ),
        "can_tera": torch.ones((num_envs, 2), dtype=torch.bool),
    }

    actor_out = agent(obs, action_masks=masks)
    actor_out["log_prob"].sum().backward()
    actor_grad = agent.hidden_info_head.policy_belief_adapter.weight.grad
    assert actor_grad is not None
    assert torch.count_nonzero(actor_grad) > 0

    agent.zero_grad(set_to_none=True)
    value_out = agent(obs, action_masks=masks)
    value_out["value"].sum().backward()
    value_grad = agent.hidden_info_head.policy_belief_adapter.weight.grad
    assert value_grad is not None
    assert torch.count_nonzero(value_grad) > 0

    agent.zero_grad(set_to_none=True)
    agent.get_value(obs).sum().backward()
    bootstrap_value_grad = agent.hidden_info_head.policy_belief_adapter.weight.grad
    assert bootstrap_value_grad is not None
    assert torch.count_nonzero(bootstrap_value_grad) > 0


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
        move_base_power_lookup=torch.tensor([0.0, 1.0], dtype=torch.float32),
        move_accuracy_lookup=torch.tensor([0.0, 1.0], dtype=torch.float32),
        move_bypasses_accuracy_lookup=torch.zeros(2, dtype=torch.float32),
        move_ignores_evasion_lookup=torch.zeros(2, dtype=torch.float32),
        move_target_scope_lookup=torch.tensor([0, 2], dtype=torch.long),
        move_multihit_lookup=torch.zeros(2, dtype=torch.float32),
        move_priority_lookup=torch.tensor([0.0, 0.0], dtype=torch.float32),
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
        "own_request_move_metadata": torch.tensor(
            [[[[1, 1], [0, 0], [0, 0], [0, 0]], [[0, 0]] * 4]]
        ),
        "own_request_moves": torch.tensor(
            [[[1, 0, 0, 0], [0, 0, 0, 0]]], dtype=torch.long
        ),
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

    target_eff_feature = 19
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


def test_team_preview_head_fast_sample_returns_selected_log_prob_only() -> None:
    torch.manual_seed(0)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=3)
    own_reps = torch.randn(2, 6, 4)
    opp_reps = torch.randn(2, 6, 4)
    context = torch.randn(2, 6)

    out = head(own_reps, opp_reps, context, greedy=False, include_policy_stats=False)

    assert out["picks"].shape == (2, 3)
    assert out["lead_pair"].shape == (2, 2)
    assert out["bring_picks"].shape == (2, 3)
    assert out["log_prob"].shape == (2,)
    assert torch.isfinite(out["log_prob"]).all()
    assert "joint_log_probs" not in out
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
