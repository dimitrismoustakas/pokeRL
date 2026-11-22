from contextlib import contextmanager
from collections import OrderedDict

import pytest
import numpy as np
import torch

import pokerl.training.league as league_module
from pokerl.config import ModelConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.env.vector_inputs import VectorInputRow
from pokerl.model.agent import PokeAgent
from pokerl.training.league import League
from pokerl.training.league import (
    _InferenceRow,
    _LeagueInferenceService,
)


def _service_stub():
    return _LeagueInferenceService(
        torch.nn.Identity(), torch.device("cpu"), max_models=12
    )


def test_league_inference_sliced_hidden_reuses_one_bounded_batch_storage() -> None:
    h = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    c = torch.arange(100, 112, dtype=torch.float32).reshape(1, 3, 4)

    hidden_views = [
        _LeagueInferenceService._slice_hidden((h, c), index) for index in range(3)
    ]
    sliced_h, sliced_c = hidden_views[1]

    assert sliced_h.shape == (1, 1, 4)
    assert sliced_c.shape == (1, 1, 4)
    assert sliced_h._base is None
    assert sliced_c._base is None
    assert {view[0].untyped_storage().data_ptr() for view in hidden_views} == {
        h.untyped_storage().data_ptr()
    }
    assert {view[1].untyped_storage().data_ptr() for view in hidden_views} == {
        c.untyped_storage().data_ptr()
    }
    assert sliced_h.untyped_storage().nbytes() == h.untyped_storage().nbytes()
    assert sliced_c.untyped_storage().nbytes() == c.untyped_storage().nbytes()


def test_league_battle_inference_synchronizes_before_publishing_device_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    @contextmanager
    def fake_cuda_model_execution(device: torch.device):
        events.append(("enter", str(device)))
        try:
            yield
        finally:
            events.append(("exit", str(device)))

    class _FakeAgent:
        def __init__(self) -> None:
            self.kwargs = None

        def initial_hidden(self, batch_size: int):
            del batch_size
            return None

        def __call__(self, obs_batch, hidden_batch, *, action_masks, **kwargs):
            del hidden_batch, action_masks
            self.kwargs = kwargs
            batch_size = next(iter(obs_batch.values())).shape[0]
            h = torch.arange(batch_size * 4, dtype=torch.float32).reshape(
                1,
                batch_size,
                4,
            )
            return {
                "actions": torch.zeros((batch_size, 3), dtype=torch.long),
                "hidden": (h, h + 100),
            }

    monkeypatch.setattr(
        league_module,
        "cuda_model_execution",
        fake_cuda_model_execution,
    )

    service = _service_stub()
    service._device = torch.device("cpu")
    service._cuda_stream = None
    agent = _FakeAgent()
    service._get_agent = lambda _checkpoint_path: agent
    service._synchronize_device_execution = lambda: events.append(("sync", "cpu"))
    slice_hidden = service._slice_hidden
    service._slice_hidden = lambda hidden, index: (
        events.append(("slice", str(index))),
        slice_hidden(hidden, index),
    )[1]

    requests = [
        _InferenceRow(
            obs={"x": np.full((2,), index, dtype=np.float32)},
            masks={
                "joint": np.ones((3, 4), dtype=bool),
                "can_tera": np.ones((2,), dtype=bool),
            },
            hidden=None,
        )
        for index in range(2)
    ]

    results = service._run_battle_batch("checkpoint.pt", requests)

    assert events == [
        ("enter", "cpu"),
        ("sync", "cpu"),
        ("slice", "0"),
        ("slice", "1"),
        ("exit", "cpu"),
    ]
    assert agent.kwargs == {
        "greedy": False,
        "include_value": False,
        "include_policy_stats": False,
    }
    for index, (actions, hidden) in enumerate(results):
        np.testing.assert_array_equal(actions, np.zeros(3, dtype=np.int64))
        assert hidden is not None
        assert torch.equal(
            hidden[0],
            torch.arange(index * 4, (index + 1) * 4, dtype=torch.float32).reshape(
                1,
                1,
                4,
            ),
        )


class _RecordingBatchAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []

    def initial_hidden(self, batch_size: int):
        del batch_size
        return None

    def __call__(self, obs_batch, hidden_batch, *, action_masks, **kwargs):
        del hidden_batch, kwargs
        self.calls.append(
            (
                {key: value.detach().cpu().clone() for key, value in obs_batch.items()},
                {
                    key: value.detach().cpu().clone()
                    for key, value in action_masks.items()
                },
            )
        )
        batch_size = obs_batch["x"].shape[0]
        mask_count = sum(
            value.reshape(batch_size, -1).long().sum(dim=1)
            for value in action_masks.values()
        )
        actions = torch.stack(
            (
                obs_batch["x"][:, 0].long(),
                mask_count,
                obs_batch["ids"][:, 0].long(),
            ),
            dim=1,
        )
        hidden = obs_batch["x"][:, :1].float().reshape(1, batch_size, 1)
        return {"actions": actions, "hidden": (hidden, hidden + 100)}


def _staging_requests(
    batch_size: int,
    *,
    offset: int,
    wide: bool = False,
) -> list[_InferenceRow]:
    requests = []
    for index in range(batch_size):
        value = offset + index
        if not wide:
            obs = {
                "x": np.asarray([value + 0.25, value + 0.75], dtype=np.float32),
                "ids": np.asarray([value], dtype=np.int64),
            }
            masks = {
                "joint": np.asarray(
                    [
                        [True, index % 2 == 0, False],
                        [False, True, index % 3 == 0],
                    ],
                    dtype=np.bool_,
                ),
                "can_tera": np.asarray([index % 2 == 0, True], dtype=np.bool_),
            }
        else:
            obs = {
                "x": np.asarray(
                    [value + 0.125, value + 0.5, value + 0.875],
                    dtype=np.float64,
                ),
                "ids": np.asarray([value, value + 1], dtype=np.int32),
                "flag": np.asarray([index % 2], dtype=np.uint8),
            }
            masks = {
                "joint": np.asarray(
                    [
                        [True, False],
                        [index % 2 == 0, True],
                        [False, True],
                    ],
                    dtype=np.bool_,
                ),
                "can_tera": np.asarray([True, index % 2 == 1], dtype=np.bool_),
                "forced": np.asarray([index % 3 == 0], dtype=np.bool_),
            }
        requests.append(
            _InferenceRow(
                obs=obs,
                masks=masks,
                hidden=None,
            )
        )
    return requests


def _reference_stacked_batch(requests: list[_InferenceRow]):
    obs = {
        key: torch.from_numpy(np.stack([request.obs[key] for request in requests]))
        for key in requests[0].obs
    }
    assert requests[0].masks is not None
    masks = {
        key: torch.from_numpy(
            np.stack([request.masks[key] for request in requests])
        ).bool()
        for key in requests[0].masks
    }
    return obs, masks


def _assert_recorded_batch_matches_reference(
    agent: _RecordingBatchAgent,
    requests: list[_InferenceRow],
    results,
) -> None:
    expected_obs, expected_masks = _reference_stacked_batch(requests)
    actual_obs, actual_masks = agent.calls[-1]
    assert actual_obs.keys() == expected_obs.keys()
    assert actual_masks.keys() == expected_masks.keys()
    for key in expected_obs:
        assert actual_obs[key].dtype == expected_obs[key].dtype
        assert torch.equal(actual_obs[key], expected_obs[key])
    for key in expected_masks:
        assert actual_masks[key].dtype == torch.bool
        assert torch.equal(actual_masks[key], expected_masks[key])

    batch_size = len(requests)
    mask_count = sum(
        value.reshape(batch_size, -1).long().sum(dim=1)
        for value in expected_masks.values()
    )
    expected_actions = torch.stack(
        (
            expected_obs["x"][:, 0].long(),
            mask_count,
            expected_obs["ids"][:, 0].long(),
        ),
        dim=1,
    )
    for index, (actions, hidden) in enumerate(results):
        np.testing.assert_array_equal(actions, expected_actions[index].numpy())
        assert hidden is not None
        assert hidden[0].item() == pytest.approx(float(expected_obs["x"][index, 0]))


def test_vector_lane_gather_preserves_request_order_and_padding() -> None:
    service = _service_stub()
    columns = {
        "field": np.arange(32, dtype=np.float32).reshape(8, 4)[1::2],
        "legal": np.arange(8)[1::2] > 3,
    }
    lanes = [2, 0, 3]
    rows = [VectorInputRow(columns, lane) for lane in lanes]
    schema = service._batch_schema(rows)
    buffers = service._allocate_host_batch_buffers(schema, capacity=5)
    service._pack_input_rows(buffers, rows, size=5)
    for key, column in columns.items():
        np.testing.assert_array_equal(buffers.arrays[key], column[[2, 0, 3, 3, 3]])

    # A batch may also contain independently constructed observations.
    rows[1] = {key: value[1] for key, value in columns.items()}
    assert service._batch_schema(rows) == schema
    service._pack_input_rows(buffers, rows, size=3)
    for key, column in columns.items():
        np.testing.assert_array_equal(buffers.arrays[key][:3], column[[2, 1, 3]])


def test_league_staging_reuses_capacity_without_exposing_stale_rows() -> None:
    service = _service_stub()
    agent = _RecordingBatchAgent()
    service._get_agent = lambda _checkpoint_path: agent

    first = _staging_requests(3, offset=10)
    results = service._run_battle_batch("checkpoint.pt", first)
    _assert_recorded_batch_matches_reference(agent, first, results)
    first_host = service._host_batch_buffers["battle_obs"]
    first_pointers = {
        key: value.__array_interface__["data"][0]
        for key, value in first_host.arrays.items()
    }
    assert first_host.capacity == 3

    smaller = _staging_requests(1, offset=100)
    results = service._run_battle_batch("checkpoint.pt", smaller)
    _assert_recorded_batch_matches_reference(agent, smaller, results)
    reused_host = service._host_batch_buffers["battle_obs"]
    assert reused_host is first_host
    assert {
        key: value.__array_interface__["data"][0]
        for key, value in reused_host.arrays.items()
    } == first_pointers
    assert agent.calls[-1][0]["x"].shape == (1, 2)

    larger = _staging_requests(5, offset=200)
    results = service._run_battle_batch("checkpoint.pt", larger)
    _assert_recorded_batch_matches_reference(agent, larger, results)
    grown_host = service._host_batch_buffers["battle_obs"]
    assert grown_host is not first_host
    assert grown_host.capacity >= 5

    changed_schema = _staging_requests(2, offset=300, wide=True)
    results = service._run_battle_batch("checkpoint.pt", changed_schema)
    _assert_recorded_batch_matches_reference(agent, changed_schema, results)
    rebuilt_host = service._host_batch_buffers["battle_obs"]
    assert rebuilt_host is not grown_host
    assert rebuilt_host.schema != grown_host.schema
    assert rebuilt_host.capacity == 2


def test_action_batch_size_profile_accumulates_once_per_batch() -> None:
    service = _service_stub()
    service._get_agent = lambda path: _RecordingBatchAgent()
    service._run_battle_batch("checkpoint.pt", _staging_requests(4, offset=0))

    assert service.cache_info()["action_batch_size_histogram"] == {
        "1": 0,
        "2": 0,
        "3-4": 1,
        "5-8": 0,
        "9-12": 0,
        "13-16": 0,
        "17-24": 0,
        "25-36": 0,
        ">36": 0,
    }


def test_league_inference_model_cache_is_lru_bounded() -> None:
    service = _service_stub()
    service._models = OrderedDict()
    service._max_models = 2
    loaded: dict[str, object] = {}

    def load_agent(path: str):
        return loaded.setdefault(path, object())

    service._load_agent = load_agent

    first = service._get_agent("first.pt")
    service._get_agent("second.pt")
    assert service._get_agent("first.pt") is first
    service._get_agent("third.pt")

    assert list(service._models) == ["first.pt", "third.pt"]
    assert service.cache_info() == {
        "capacity": 2,
        "resident_models": 2,
        "hits": 1,
        "misses": 3,
        "loads": 3,
        "evictions": 1,
        "action_batch_size_histogram": {
            "1": 0,
            "2": 0,
            "3-4": 0,
            "5-8": 0,
            "9-12": 0,
            "13-16": 0,
            "17-24": 0,
            "25-36": 0,
            ">36": 0,
        },
        "cuda_graphs_enabled": False,
        "cuda_graph_batch_size": 16,
        "cuda_graph_capacity": 4,
        "resident_cuda_graphs": 0,
        "cuda_graph_hits": 0,
        "cuda_graph_misses": 0,
        "cuda_graph_captures": 0,
        "cuda_graph_replays": 0,
        "cuda_graph_evictions": 0,
        "cuda_graph_fallbacks": 0,
        "cuda_graph_real_rows": 0,
        "cuda_graph_padded_rows": 0,
        "cuda_graph_last_error": None,
    }


def test_league_staging_packs_cpu_before_entering_cuda_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    service = _service_stub()
    agent = _RecordingBatchAgent()
    service._get_agent = lambda _checkpoint_path: agent
    original_pack = service._pack_cpu_batch

    def recording_pack(cache_key, mappings):
        events.append(f"pack:{cache_key}")
        return original_pack(cache_key, mappings)

    @contextmanager
    def recording_guard(device):
        del device
        events.append("guard:enter")
        try:
            yield
        finally:
            events.append("guard:exit")

    service._pack_cpu_batch = recording_pack
    monkeypatch.setattr(league_module, "cuda_model_execution", recording_guard)
    service._run_battle_batch(
        "checkpoint.pt",
        _staging_requests(2, offset=0),
    )

    assert events[:3] == ["pack:battle_obs", "pack:battle_masks", "guard:enter"]
    assert events[-1] == "guard:exit"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_league_staging_reuses_cuda_buffers_for_smaller_active_batch() -> None:
    service = _service_stub()
    service._device = torch.device("cuda")
    agent = _RecordingBatchAgent()
    service._get_agent = lambda _checkpoint_path: agent

    first = _staging_requests(3, offset=10)
    results = service._run_battle_batch("checkpoint.pt", first)
    _assert_recorded_batch_matches_reference(agent, first, results)
    obs_buffers = service._device_batch_buffers["battle_obs"]
    mask_buffers = service._device_batch_buffers["battle_masks"]
    obs_pointers = {
        key: value.untyped_storage().data_ptr()
        for key, value in obs_buffers.tensors.items()
    }
    mask_pointers = {
        key: value.untyped_storage().data_ptr()
        for key, value in mask_buffers.tensors.items()
    }

    smaller = _staging_requests(1, offset=100)
    results = service._run_battle_batch("checkpoint.pt", smaller)
    _assert_recorded_batch_matches_reference(agent, smaller, results)
    assert service._device_batch_buffers["battle_obs"] is obs_buffers
    assert service._device_batch_buffers["battle_masks"] is mask_buffers
    assert {
        key: value.untyped_storage().data_ptr()
        for key, value in obs_buffers.tensors.items()
    } == obs_pointers
    assert {
        key: value.untyped_storage().data_ptr()
        for key, value in mask_buffers.tensors.items()
    } == mask_pointers
    assert agent.calls[-1][0]["x"].shape == (1, 2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_league_uses_same_process_cuda_inference(tmp_path) -> None:
    game_data = GameData(gen=9)
    agent = PokeAgent(ModelConfig(), game_data)
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league = League(config, agent, torch.device("cuda"))
    try:
        assert league.device.type == "cuda"
        assert league.inference_device.type == "cuda"
        assert isinstance(league._inference, _LeagueInferenceService)
        assert league.preview_inference_device.type == "cuda"
        assert league._preview_inference is league._inference
    finally:
        league.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_league_can_use_separate_cpu_preview_inference(tmp_path) -> None:
    game_data = GameData(gen=9)
    agent = PokeAgent(ModelConfig(), game_data)
    config = TrainingConfig(
        checkpoint_dir=str(tmp_path),
        league_preview_device="cpu",
    )
    league = League(config, agent, torch.device("cuda"))
    try:
        assert league.inference_device.type == "cuda"
        assert league.preview_inference_device.type == "cpu"
        assert league._preview_inference is not league._inference
    finally:
        league.close()


def test_league_reuses_action_service_for_cpu_preview_inference(tmp_path) -> None:
    game_data = GameData(gen=9)
    agent = PokeAgent(ModelConfig(), game_data)
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league = League(config, agent, torch.device("cpu"))
    try:
        assert league.inference_device.type == "cpu"
        assert league.preview_inference_device.type == "cpu"
        assert league._preview_inference is league._inference
    finally:
        league.close()
