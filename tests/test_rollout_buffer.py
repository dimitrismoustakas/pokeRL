from concurrent.futures import Future
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
    NUM_TERA_CHOICES,
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
)
from pokerl.env.showdown_env import BattleEnv, ParallelBattleEnv, ShowdownPlayer
from pokerl.env.observation import ROLE_ACTIVE_1, ROLE_ACTIVE_2, ROLE_BENCH
from pokerl.model.heads import ActionHead, MaskedSetAttention, TeamPreviewHead, ValueHead
from pokerl.model.memory import IdentityMemory
from pokerl.training.league import LeaguePlayer
from pokerl.training.ppo import PPOTrainer
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
        actions_t = torch.tensor(
            [[step, step + 1, TERA_NONE], [step + 2, step + 3, TERA_NONE]],
            dtype=torch.long,
            device=device,
        )
        masks_t = {
            key: torch.from_numpy(value).to(device)
            for key, value in _mask(num_envs).items()
        }
        log_probs_t = torch.full((num_envs,), float(step), dtype=torch.float32, device=device)
        values_t = torch.full((num_envs,), float(step) + 0.5, dtype=torch.float32, device=device)
        rewards_t = torch.tensor([1.0, -1.0], dtype=torch.float32, device=device)
        dones_t = torch.tensor([False, step == num_steps - 1], dtype=torch.bool, device=device)
        buffer.add(obs_t, actions_t, masks_t, log_probs_t, values_t, rewards_t, dones_t)

    batch = buffer.compute_gae(torch.zeros(num_envs, dtype=torch.float32, device=device))

    assert batch.obs["float_obs"].shape == (num_steps, num_envs, 3)
    assert batch.obs["index_obs"].shape == (num_steps, num_envs, 2)
    assert batch.actions.shape == (num_steps, num_envs, 3)
    assert batch.action_masks["joint"].shape == (num_steps, num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT)
    assert batch.action_masks["can_tera"].shape == (num_steps, num_envs, 2)
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
            "value": self.value_head(context, own_reps, opp_reps, own_bench_mask, opp_bench_mask),
            "hidden": hidden,
        }

    def initial_hidden(self, batch_size: int):
        return self.memory.initial_hidden(batch_size)


def _ppo_obs(step: int, num_envs: int) -> dict[str, np.ndarray]:
    own_roles = np.tile(
        np.array([ROLE_ACTIVE_1, ROLE_ACTIVE_2, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH], dtype=np.int64),
        (num_envs, 1),
    )
    opp_roles = np.tile(
        np.array([ROLE_ACTIVE_1, ROLE_ACTIVE_2, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH, ROLE_BENCH], dtype=np.int64),
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
        "joint": np.ones((num_envs, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool),
        "can_tera": np.ones((num_envs, 2), dtype=bool),
    }
    agent = _DummyAgent().to(device)
    config = TrainingConfig(num_epochs=1, num_minibatches=2)
    trainer = PPOTrainer(agent, config, device)
    buffer = RolloutBuffer(num_steps, num_envs, _ppo_obs(0, num_envs), action_mask, device)
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
        dones_t = torch.tensor([False, step == num_steps - 1], dtype=torch.bool, device=device)
        buffer.add(obs_t, out["actions"], masks_t, out["log_prob"], out["value"], rewards_t, dones_t)

    batch = buffer.compute_gae(torch.zeros(num_envs, dtype=torch.float32, device=device))
    metrics = trainer.update(batch)

    assert trainer.global_step == num_steps * num_envs
    assert set(metrics) == {"policy_loss", "value_loss", "entropy", "approx_kl"}


def test_action_head_uses_joint_and_tera_masks_to_coordinate_actions() -> None:
    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    own_team_reps = torch.zeros((1, 6, 2))
    opp_team_reps = torch.zeros((1, 6, 2))
    opp_reserve_mask = torch.tensor([[True, True, False, False, False, False]], dtype=torch.bool)
    opp_active_reps = torch.zeros((1, 2, 2))
    masks = {
        "joint": torch.zeros((1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=torch.bool),
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


def test_action_head_uses_conditional_joint_mask_to_block_duplicate_switch_target() -> None:
    torch.manual_seed(0)
    head = ActionHead(context_dim=3, pokemon_dim=2, hidden_dim=8)
    context = torch.zeros((1, 3))
    active_reps = torch.zeros((1, 2, 2))
    own_team_reps = torch.zeros((1, 6, 2))
    opp_team_reps = torch.zeros((1, 6, 2))
    opp_reserve_mask = torch.tensor([[True, True, False, False, False, False]], dtype=torch.bool)
    opp_active_reps = torch.zeros((1, 2, 2))
    masks = {
        "joint": torch.zeros((1, ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=torch.bool),
        "can_tera": torch.zeros((1, 2), dtype=torch.bool),
    }
    shared_switch = SWITCH_OFFSET + 5
    masks["joint"][0, FIRST_SLOT_MASK_ROW, shared_switch] = True
    masks["joint"][0, shared_switch + 1, 0] = True

    out = head(context, active_reps, own_team_reps, opp_team_reps, opp_reserve_mask, opp_active_reps, masks)

    assert out["actions"][0, 0].item() == shared_switch
    assert out["actions"][0, 1].item() == 0
    assert out["actions"][0, 2].item() == TERA_NONE


def test_action_head_matchup_features_distinguish_super_effective_and_resisted_targets() -> None:
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

    out = head(own_reps, opp_reps, context, team_mask=own_mask, opponent_mask=opponent_mask, greedy=True)

    assert out["picks"].shape == (2, 3)
    assert out["log_prob"].shape == (2,)
    assert out["entropy"].shape == (2,)
    assert torch.all(out["picks"][0] != 5)
    assert torch.all(out["bring_picks"][:, 1:] > out["bring_picks"][:, :-1])
    assert torch.all(out["picks"][:, 2:] == out["picks"][:, 2:].sort(dim=1).values)


def test_team_preview_head_preserves_lead_order_while_canonicalizing_bench_tail() -> None:
    torch.manual_seed(0)
    head = TeamPreviewHead(pokemon_dim=4, context_dim=6, pick_size=4)
    own_reps = torch.randn(1, 6, 4)
    opp_reps = torch.randn(1, 6, 4)
    context = torch.randn(1, 6)
    provided_picks = torch.tensor([[4, 1, 5, 0]], dtype=torch.long)

    out = head(own_reps, opp_reps, context, picks=provided_picks)

    assert torch.equal(out["lead_pair"], torch.tensor([[4, 1]], dtype=torch.long))
    assert torch.equal(out["bring_picks"], torch.tensor([[0, 1, 4, 5]], dtype=torch.long))
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

    monkeypatch.setattr(actions_module, "_build_slot_action_state", counted_slot_action_state)

    action_state = build_legal_action_state(battle)
    expected_team_slot = 3
    switch_action = SWITCH_OFFSET + expected_team_slot

    assert build_action_mask(battle, slot=0, action_state=action_state)[switch_action]
    assert build_joint_action_mask(battle, action_state=action_state)[FIRST_SLOT_MASK_ROW, switch_action]
    assert action_to_order(switch_action, battle, slot=0, action_state=action_state).order is battle.available_switches[0][0]
    assert calls == 2


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
        self.opponent_active_pokemon = [_FakePokemon("foe-one"), _FakePokemon("foe-two")]
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

    def get_possible_showdown_targets(self, move: Move, pokemon, dynamax: bool = False) -> list[int]:
        del dynamax
        if pokemon == self.active_pokemon[0] and move.id in [m.id for m in self.available_moves[0]]:
            self_position = self.POKEMON_1_POSITION
            ally_position = self.POKEMON_2_POSITION
        elif pokemon == self.active_pokemon[1] and move.id in [m.id for m in self.available_moves[1]]:
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
                Target.ADJACENT_FOE: [self.OPPONENT_1_POSITION, self.OPPONENT_2_POSITION],
                Target.ALL: [self.EMPTY_TARGET_POSITION],
                Target.ALL_ADJACENT: [self.EMPTY_TARGET_POSITION],
                Target.ALL_ADJACENT_FOES: [self.EMPTY_TARGET_POSITION],
                Target.ALLIES: [self.EMPTY_TARGET_POSITION],
                Target.ALLY_SIDE: [self.EMPTY_TARGET_POSITION],
                Target.ALLY_TEAM: [self.EMPTY_TARGET_POSITION],
                Target.ANY: [ally_position, self.OPPONENT_1_POSITION, self.OPPONENT_2_POSITION],
                Target.FOE_SIDE: [self.EMPTY_TARGET_POSITION],
                Target.NORMAL: [ally_position, self.OPPONENT_1_POSITION, self.OPPONENT_2_POSITION],
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
    battle = _FakeTargetBattle(Move("icespinner", gen=9))

    mask = build_action_mask(battle, slot=0)

    assert mask[0]
    assert mask[1]
    assert not mask[2]

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
def test_random_normal_move_without_explicit_request_target_uses_foe_targets(
    move_request: dict[str, object],
) -> None:
    battle = _FakeTargetBattle(Move("outrage", gen=9))
    battle.last_request = {
        "active": [
            {
                "moves": [move_request]
            },
            {"moves": []},
        ]
    }

    mask = build_action_mask(battle, slot=0)

    assert mask[0]
    assert mask[1]
    assert not mask[TARGET_NONE]

    with pytest.raises(RuntimeError, match="not legal"):
        action_to_order(TARGET_NONE, battle, slot=0)


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
    batch_obs_2, batch_mask_2 = ParallelBattleEnv._collate(env, second_obs, second_masks)

    assert batch_obs_1["a"] is batch_obs_2["a"]
    assert batch_obs_1["b"] is batch_obs_2["b"]
    assert batch_mask_1["joint"] is batch_mask_2["joint"]
    assert batch_mask_1["can_tera"] is batch_mask_2["can_tera"]
    np.testing.assert_array_equal(batch_obs_2["a"], np.array([[5, 6], [7, 8]], dtype=np.int64))
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
                        {"id": "icespinner", "target": "adjacentFoe", "disabled": False},
                    ]
                },
                {
                    "moves": [
                        {"id": "suckerpunch", "target": "adjacentFoe", "disabled": False},
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
    timing = SimpleNamespace(increment=lambda *args, **kwargs: None, add=lambda *args, **kwargs: None)
    fake_env = SimpleNamespace(
        _active_preview_obs=obs,
        _active_preview_future=future,
        _timing=timing,
        _ext_team_preview_fn=lambda current_obs: (_ for _ in ()).throw(AssertionError("fallback should not run")),
        _obs_equal=lambda left, right: BattleEnv._obs_equal(left, right),
        _record_preview_obs_mismatch=lambda actual, prepared: (_ for _ in ()).throw(
            AssertionError("mismatch handler should not run")
        ),
    )

    with pytest.raises(RuntimeError, match="preview batch failed"):
        BattleEnv._choose_team_preview_picks(fake_env, obs)


def test_finished_battle_update_skips_legality_rebuild(monkeypatch: pytest.MonkeyPatch) -> None:
    queued = []
    timing = SimpleNamespace(increment=lambda *args, **kwargs: None, add=lambda *args, **kwargs: None)
    fake_player = SimpleNamespace(
        last_battle=None,
        _last_finished_token=None,
        obs_builder=SimpleNamespace(build=lambda battle: {"context": np.array([1], dtype=np.int64)}),
        timing=timing,
        obs_queue=SimpleNamespace(put=lambda item: queued.append(item)),
        _pending_action_state="stale",
        _pending_action_battle=object(),
    )
    battle = SimpleNamespace(battle_tag="battle-1", turn=7, finished=True, won=True, lost=False)

    monkeypatch.setattr(
        showdown_env_module,
        "build_legal_action_state",
        lambda battle: (_ for _ in ()).throw(AssertionError("finished battles should not rebuild legality")),
    )

    ShowdownPlayer._emit_battle_update(fake_player, battle)

    assert fake_player._pending_action_state is None
    assert fake_player._pending_action_battle is None
    _, mask, _, done, queued_battle = queued[0]
    assert done is True
    assert queued_battle is battle
    assert mask["joint"][FIRST_SLOT_MASK_ROW, PASS_ACTION]
    assert mask["joint"][PASS_ACTION + 1, PASS_ACTION]
    np.testing.assert_array_equal(mask["can_tera"], np.array([False, False], dtype=bool))


def test_league_player_team_preview_propagates_precompute_failure() -> None:
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
            infer_team_preview=lambda checkpoint_path, current_obs: (_ for _ in ()).throw(
                AssertionError("fallback should not run")
            )
        ),
        checkpoint_path="checkpoint.pt",
    )

    with pytest.raises(RuntimeError, match="opponent preview batch failed"):
        LeaguePlayer.teampreview(fake_player, battle=object())


def test_league_player_keeps_recurrent_state_on_same_state_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeInferenceService:
        def __init__(self) -> None:
            self.hidden_inputs: list[torch.Tensor | None] = []
            self.step = 0

        def infer_action(self, checkpoint_path, obs, masks, hidden):
            self.step += 1
            self.hidden_inputs.append(None if hidden is None else hidden[0].clone())
            next_value = float(10 + self.step)
            next_hidden = (
                torch.full((1, 1, 1), next_value),
                torch.full((1, 1, 1), next_value + 100),
            )
            return np.array([PASS_ACTION, PASS_ACTION, TERA_NONE], dtype=np.int64), next_hidden

    fake_service = _FakeInferenceService()
    player = LeaguePlayer.__new__(LeaguePlayer)
    player.checkpoint_path = "checkpoint.pt"
    player.inference_service = fake_service
    player.obs_builder = SimpleNamespace(
        build=lambda battle: {"context": np.array([battle.state], dtype=np.int64)}
    )
    player._hidden = (
        torch.full((1, 1, 1), 1.0),
        torch.full((1, 1, 1), 101.0),
    )
    player._pending_hidden = None
    player._pending_battle_obs = None
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
            slot_masks=(np.zeros(ACTIONS_PER_SLOT, dtype=bool), np.zeros(ACTIONS_PER_SLOT, dtype=bool)),
        ),
    )
    monkeypatch.setattr(
        league_module,
        "action_to_order",
        lambda action_idx, battle, slot, terastallize=False, action_state=None: PassBattleOrder(),
    )

    battle_kwargs = dict(
        finished=False,
        battle_tag="battle-hidden",
        active_pokemon=[SimpleNamespace(species="a"), SimpleNamespace(species="b")],
        opponent_active_pokemon=[SimpleNamespace(species="c"), SimpleNamespace(species="d")],
        force_switch=[False, False],
        can_tera=[False, False],
        trapped=[False, False],
        last_request={"active": [{"moves": []}, {"moves": []}]},
    )
    retry_battle = SimpleNamespace(state=1, turn=1, **battle_kwargs)
    next_battle = SimpleNamespace(state=2, turn=2, **battle_kwargs)

    LeaguePlayer.choose_move(player, retry_battle)
    LeaguePlayer.choose_move(player, retry_battle)
    LeaguePlayer.choose_move(player, next_battle)

    assert torch.equal(fake_service.hidden_inputs[0], torch.full((1, 1, 1), 1.0))
    assert torch.equal(fake_service.hidden_inputs[1], torch.full((1, 1, 1), 1.0))
    assert torch.equal(fake_service.hidden_inputs[2], torch.full((1, 1, 1), 12.0))
