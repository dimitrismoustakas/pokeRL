from pokerl.env.vector_inputs import preview_observations
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn as nn

from pokerl.config import ModelConfig
from pokerl.data import GameData
from pokerl.env.actions import ACTIONS_PER_SLOT
from pokerl.env.observation import (
    ROLE_ACTIVE_1,
    ROLE_ACTIVE_2,
    ROLE_BENCH,
)
from pokerl.model.agent import PokeAgent
from pokerl.model.preview_cache import TeamPreviewCache
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool
from pokerl.training.league import _InferenceRow, _LeagueInferenceService
from pokerl.training.rollout_policy import RolloutPolicy
from pokerl.training.inference_batch import InputTransfer


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable"
)


class _CaptureAgent(nn.Module):
    """Small recurrent stochastic policy that is safe to capture on CUDA."""

    def __init__(self) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.linspace(-0.4, 0.4, 8))
        self.action_head = SimpleNamespace(num_actions=8)

    def initial_hidden(self, batch_size: int):
        device = self.logit_scale.device
        return (
            torch.zeros(1, batch_size, 1, device=device),
            torch.zeros(1, batch_size, 1, device=device),
        )

    def forward(
        self,
        obs,
        hidden=None,
        *,
        action_masks,
        greedy=False,
        **_kwargs,
    ):
        logits = obs["x"] * self.logit_scale
        logits = logits.masked_fill(~action_masks["legal"], -1e8)
        if greedy:
            selected = logits.argmax(dim=1)
        else:
            sampling_exponentials = _kwargs.get("sampling_exponentials")
            exponential = (
                torch.empty_like(logits).exponential_()
                if sampling_exponentials is None
                else sampling_exponentials[0]
            )
            selected = (logits - exponential.log()).argmax(dim=1)
        actions = torch.stack(
            (selected, selected, torch.zeros_like(selected)),
            dim=1,
        )
        if hidden is None:
            hidden = self.initial_hidden(obs["x"].shape[0])
        increment = obs["x"][:, :1].reshape(1, obs["x"].shape[0], 1)
        return {
            "actions": actions,
            "hidden": (hidden[0] + increment, hidden[1] + 1.0),
        }


def _service(
    *,
    greedy: bool = False,
    cuda_graph_batch_size: int = 16,
    cuda_graph_cache_size: int = 4,
):
    agent = _CaptureAgent().cuda().eval().requires_grad_(False)
    service = _LeagueInferenceService(
        agent,
        torch.device("cuda"),
        max_models=4,
        greedy=greedy,
        enable_cuda_graphs=True,
        cuda_graph_batch_size=cuda_graph_batch_size,
        cuda_graph_cache_size=cuda_graph_cache_size,
    )
    service._models = OrderedDict((path, agent) for path in ("a.pt", "b.pt", "c.pt"))
    return service


def _requests(
    batch_size: int,
    *,
    with_hidden: bool = True,
) -> list[_InferenceRow]:
    requests = []
    for index in range(batch_size):
        legal = np.zeros(8, dtype=np.bool_)
        legal[index % 8] = True
        legal[(index + 3) % 8] = True
        hidden = None
        if with_hidden:
            hidden = (
                torch.full((1, 1, 1), float(index), device="cuda"),
                torch.full((1, 1, 1), float(2 * index), device="cuda"),
            )
        requests.append(
            _InferenceRow(
                obs={"x": np.full(8, 0.01 * (index + 1), dtype=np.float32)},
                masks={"legal": legal},
                hidden=hidden,
            )
        )
    return requests


def _run(service, requests, path="a.pt"):
    # Production hidden states are published only after the inference stream
    # synchronizes. Mirror that ownership contract for tensors built directly
    # by this test on the default stream.
    torch.cuda.synchronize()
    return service._run_battle_batch(path, requests)


def test_cuda_graph_handles_padded_split_batches_legality_and_hidden() -> None:
    service = _service(greedy=True)
    try:
        for batch_size in (1, 23, 36, 72):
            requests = _requests(batch_size)
            results = _run(service, requests)
            for index, ((actions, hidden), request) in enumerate(
                zip(results, requests, strict=True)
            ):
                assert request.masks is not None
                assert request.masks["legal"][actions[0]]
                assert actions[0] == np.flatnonzero(request.masks["legal"])[-1]
                assert actions[1] == actions[0]
                assert actions[2] == 0
                assert hidden is not None
                expected_increment = 0.01 * (index + 1)
                assert hidden[0].item() == pytest.approx(
                    index + expected_increment,
                    abs=1e-5,
                )
                assert hidden[1].item() == pytest.approx(2 * index + 1.0)

        info = service.cache_info()
        assert info["cuda_graph_captures"] == 1
        assert info["cuda_graph_batch_size"] == 16
        assert info["cuda_graph_replays"] == 11
        assert info["cuda_graph_real_rows"] == 132
        assert info["cuda_graph_padded_rows"] == 44
        assert info["resident_cuda_graphs"] == 1
        # B=72 is split into five fixed B=16 replays.
        assert info["cuda_graph_hits"] >= 10
    finally:
        service.close()


def test_cuda_graph_stochastic_replay_and_hidden_outputs_do_not_alias() -> None:
    service = _service()
    try:
        first_requests = _requests(23, with_hidden=False)
        first = _run(service, first_requests)
        first_actions = np.stack([result[0] for result in first])
        retained_hidden = first[0][1]
        assert retained_hidden is not None
        retained_copy = (retained_hidden[0].clone(), retained_hidden[1].clone())

        second = _run(service, _requests(23, with_hidden=False))
        second_actions = np.stack([result[0] for result in second])
        assert not np.array_equal(first_actions, second_actions)
        assert torch.equal(retained_hidden[0], retained_copy[0])
        assert torch.equal(retained_hidden[1], retained_copy[1])
    finally:
        service.close()


def test_cuda_graph_recapture_preserves_saved_rng_sequence() -> None:
    service = _service()
    try:
        saved_rng = torch.cuda.get_rng_state().clone()
        first = _run(service, _requests(23, with_hidden=False))
        first_actions = np.stack([result[0] for result in first])

        with torch.no_grad():
            service._evict_cuda_graph("a.pt")
        torch.cuda.set_rng_state(saved_rng)
        second = _run(service, _requests(23, with_hidden=False))
        second_actions = np.stack([result[0] for result in second])

        np.testing.assert_array_equal(second_actions, first_actions)
    finally:
        service.close()


def test_cuda_graph_cache_is_lru_bounded_to_two_checkpoints() -> None:
    service = _service(greedy=True, cuda_graph_cache_size=2)
    try:
        _run(service, _requests(1), path="a.pt")
        _run(service, _requests(1), path="b.pt")
        _run(service, _requests(1), path="a.pt")
        _run(service, _requests(1), path="c.pt")

        assert list(service._cuda_graphs) == ["a.pt", "c.pt"]
        info = service.cache_info()
        assert info["resident_cuda_graphs"] == 2
        assert info["cuda_graph_evictions"] >= 1
    finally:
        service.close()


@pytest.mark.parametrize(
    "failing_method",
    ("_prepare_cuda_graph_executor", "_capture_cuda_graph"),
)
def test_cuda_graph_failure_restores_rng_and_falls_back_to_eager(
    monkeypatch: pytest.MonkeyPatch,
    failing_method: str,
) -> None:
    service = _service()
    try:
        saved_rng = torch.cuda.get_rng_state().clone()

        def fail(*_args, **_kwargs):
            raise RuntimeError(f"injected {failing_method} failure")

        monkeypatch.setattr(service, failing_method, fail)
        fallback = _run(service, _requests(23, with_hidden=False))
        fallback_actions = np.stack([result[0] for result in fallback])
        fallback_next_rng = torch.cuda.get_rng_state().clone()

        # The optional graph path disables itself after a fault. Replaying the
        # same logical batch from the same RNG state therefore exercises the
        # direct eager path and must be policy-identical.
        torch.cuda.set_rng_state(saved_rng)
        eager = _run(service, _requests(23, with_hidden=False))
        eager_actions = np.stack([result[0] for result in eager])

        np.testing.assert_array_equal(fallback_actions, eager_actions)
        assert torch.equal(torch.cuda.get_rng_state(), fallback_next_rng)
        info = service.cache_info()
        assert info["cuda_graphs_enabled"] is False
        assert info["resident_cuda_graphs"] == 0
        assert info["cuda_graph_fallbacks"] == 1
        assert failing_method in str(info["cuda_graph_last_error"])
    finally:
        service.close()


def _full_agent_inputs():
    game_data = GameData(gen=9)
    team_pool = load_team_pool(
        default_team_pool_manifest_path(Path.cwd(), "gen9vgc2026regf"),
        subset="curated",
        seed=0,
    )
    own_team, opponent_team = team_pool.sample_packed_pair()
    one_obs = {
        key: rows[0]
        for key, rows in preview_observations(
            [(own_team, opponent_team)], game_data
        ).items()
    }
    roles = np.asarray(
        [
            ROLE_ACTIVE_1,
            ROLE_ACTIVE_2,
            ROLE_BENCH,
            ROLE_BENCH,
            ROLE_BENCH,
            ROLE_BENCH,
        ],
        dtype=np.int64,
    )
    for side in ("own", "opp"):
        one_obs[f"{side}_roles"][:] = roles
        one_obs[f"{side}_active_indices"][:] = [0, 1]
        one_obs[f"{side}_active_present"][:] = True
    one_masks = {
        "joint": np.ones(
            (ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT),
            dtype=np.bool_,
        ),
        "can_tera": np.ones(2, dtype=np.bool_),
    }
    return game_data, one_obs, one_masks


def test_feedback_buffer_reused_with_default_cuda_device() -> None:
    transfer = InputTransfer({"labels": np.array([1, 2])}, torch.device("cuda"))
    previous = transfer.stage({"labels": np.array([1, 2])})["labels"]
    np.testing.assert_array_equal(previous.cpu().numpy(), [1, 2])
    transfer.host.arrays["labels"][:] = [3, 4]
    updated = transfer.upload()["labels"]
    assert updated is previous
    np.testing.assert_array_equal(updated.cpu().numpy(), [3, 4])


@torch.no_grad()
def test_learner_graph_matches_eager_on_changing_inputs_and_hidden_state() -> None:
    data, row, mask = _full_agent_inputs()
    device = torch.device("cuda")
    agent = PokeAgent(ModelConfig(), data).to(device).eval()
    obs = {key: np.repeat(value[None], 2, axis=0) for key, value in row.items()}
    masks = {key: np.repeat(value[None], 2, axis=0) for key, value in mask.items()}
    policy = RolloutPolicy(
        agent, obs, masks, device, 0.13, capture_stream=torch.cuda.Stream(device=device)
    )
    hidden = agent.initial_hidden(2)
    for move in (0, 5):
        masks["joint"].fill(False)
        masks["joint"][:, 0, move : move + 2] = True
        masks["joint"][:, 1 + move : 3 + move, move : move + 2] = True
        obs["own_continuous"][:, 0, 0] = move / 5
        inputs = policy.observations.stage(obs)
        legal = policy.masks.stage(masks)
        before = torch.cuda.get_rng_state()
        kwargs = dict(action_masks=legal, tera_exploration_prob=0.13)
        eager = agent(inputs, hidden, **kwargs)
        after = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before)
        replay = policy.sample(hidden)
        assert torch.equal(torch.cuda.get_rng_state(), after)
        for key in ("actions", "log_prob", "value"):
            torch.testing.assert_close(replay[key], eager[key], rtol=0, atol=0)
        for actual, expected in zip(replay["hidden"], eager["hidden"], strict=True):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        hidden = eager["hidden"]


def test_full_agent_graph_matches_eager_across_batches_and_weight_swaps(
    tmp_path,
) -> None:
    game_data, one_obs, one_masks = _full_agent_inputs()
    torch.manual_seed(77)
    torch.cuda.manual_seed_all(77)
    agent = PokeAgent(ModelConfig(), game_data).cuda().eval().requires_grad_(False)
    first, second = (str(tmp_path / f"{name}.pt") for name in ("first", "second"))
    torch.save({"agent": agent.state_dict()}, first)
    other = PokeAgent(ModelConfig(), game_data)
    torch.save({"agent": other.state_dict()}, second)
    service = _LeagueInferenceService(
        agent,
        torch.device("cuda"),
        max_models=1,
        enable_cuda_graphs=True,
    )
    try:
        for batch_size, path in ((1, first), (23, second), (36, first), (72, second)):
            agent.load_state_dict(torch.load(path, map_location="cpu")["agent"])
            obs_np = {
                key: np.repeat(value[None, ...], batch_size, axis=0)
                for key, value in one_obs.items()
            }
            masks_np = {
                key: np.repeat(value[None, ...], batch_size, axis=0)
                for key, value in one_masks.items()
            }
            obs = {key: torch.from_numpy(value).cuda() for key, value in obs_np.items()}
            masks = {
                key: torch.from_numpy(value).cuda() for key, value in masks_np.items()
            }
            hidden = agent.initial_hidden(batch_size)
            torch.cuda.synchronize()
            saved_rng = torch.cuda.get_rng_state().clone()

            with torch.inference_mode():
                eager = agent(
                    obs,
                    hidden,
                    action_masks=masks,
                    include_value=False,
                    include_policy_stats=False,
                )
            torch.cuda.synchronize()
            eager_actions = eager["actions"].cpu().numpy()
            eager_hidden = (
                eager["hidden"][0].cpu(),
                eager["hidden"][1].cpu(),
            )
            eager_next_rng = torch.cuda.get_rng_state().clone()

            torch.cuda.set_rng_state(saved_rng)
            requests = [
                _InferenceRow(
                    obs={key: value[index].copy() for key, value in obs_np.items()},
                    masks={key: value[index].copy() for key, value in masks_np.items()},
                    hidden=(
                        hidden[0][:, index : index + 1],
                        hidden[1][:, index : index + 1],
                    ),
                )
                for index in range(batch_size)
            ]
            graph_results = _run(service, requests, path=path)
            graph_actions = np.stack([result[0] for result in graph_results])
            graph_hidden = (
                torch.cat([result[1][0].cpu() for result in graph_results], dim=1),
                torch.cat([result[1][1].cpu() for result in graph_results], dim=1),
            )

            np.testing.assert_array_equal(graph_actions, eager_actions)
            assert torch.equal(torch.cuda.get_rng_state(), eager_next_rng)
            # Padded fixed-shape graph GEMMs and eager variable-shape GEMMs can
            # differ by a few float32 ULPs while producing identical actions.
            assert torch.allclose(graph_hidden[0], eager_hidden[0], atol=5e-7, rtol=0)
            cell_max_abs_error = (graph_hidden[1] - eager_hidden[1]).abs().max()
            assert torch.allclose(
                graph_hidden[1], eager_hidden[1], atol=5e-7, rtol=0
            ), f"cell-state max abs error: {cell_max_abs_error.item():.3g}"
            preview_rng = torch.cuda.get_rng_state()
            lanes = np.arange(batch_size)
            expected_picks = (
                TeamPreviewCache()
                .predict(
                    agent, torch.device("cuda"), obs_np, lanes, include_value=False
                )["picks"]
                .cpu()
                .numpy()
            )
            torch.cuda.set_rng_state(preview_rng)
            actual_picks = service._worker.submit(
                service._run_preview_batch, path, obs_np, lanes
            ).result()
            np.testing.assert_array_equal(actual_picks, expected_picks)
        info = service.cache_info()
        assert info["cuda_graph_captures"] == 1
        assert info["cuda_graph_fallbacks"] == 0
        assert info["cuda_graph_last_error"] is None
    finally:
        service.close()
