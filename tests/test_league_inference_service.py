from concurrent.futures import Future
from contextlib import contextmanager
from collections import OrderedDict, defaultdict
from threading import Condition
import time

import pytest
import numpy as np
import torch

import pokerl.training.league as league_module
from pokerl.config import EnvConfig, ModelConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.model.agent import PokeAgent
from pokerl.training.league import League, LeaguePlayer
from pokerl.training.league import (
    _InferenceRequest,
    _LeagueInferenceService,
)
from pokerl.training.preview import TeamPreviewBatcher, TeamPreviewResult


class _AliveThread:
    def is_alive(self) -> bool:
        return True


class _StoppedThread:
    def is_alive(self) -> bool:
        return False

    def join(self, timeout: float | None = None) -> None:
        return None


class _RetryableCloseThread:
    def __init__(self) -> None:
        self.alive = True
        self.join_calls = 0

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float | None = None) -> None:
        del timeout
        self.join_calls += 1
        if self.join_calls >= 2:
            self.alive = False


class _CancelBeforeSetExceptionFuture(Future):
    """Deterministically model cancellation between a state check and publish."""

    def set_exception(self, exception: BaseException) -> None:
        self.cancel()
        super().set_exception(exception)


class _CancelBeforeSetResultFuture(Future):
    """Deterministically model cancellation racing result publication."""

    def set_result(self, result: object) -> None:
        self.cancel()
        super().set_result(result)


def _request(
    future: Future,
    *,
    checkpoint_path: str = "checkpoint.pt",
    kind: str = "battle",
    submitted_at: float = 0.0,
) -> _InferenceRequest:
    return _InferenceRequest(
        checkpoint_path=checkpoint_path,
        kind=kind,
        obs={},
        masks={},
        hidden=None,
        future=future,
        submitted_at=submitted_at,
    )


def _service_stub() -> _LeagueInferenceService:
    service = object.__new__(_LeagueInferenceService)
    service._condition = Condition()
    service._closed = False
    service._pending = []
    service._active_requests = []
    service._active_key = None
    service._active_started_at = None
    service._active_phase = None
    service._active_phase_started_at = None
    service._fatal_error = None
    service._submitted_requests = 0
    service._completed_requests = 0
    service._failed_requests = 0
    service._last_progress_at = time.perf_counter()
    service._thread = _AliveThread()
    service._models = {"checkpoint.pt": object()}
    service._max_models = 12
    service._cache_hits = 0
    service._cache_misses = 0
    service._cache_loads = 0
    service._cache_evictions = 0
    service._host_batch_buffers = {}
    service._device_batch_buffers = {}
    service._prefetch_paths = set()
    service._coalesce_window_sec = 0.0
    service._coalesce_target_batch_size = None
    service._coalesce_target_wakeups = 0
    service._device = torch.device("cpu")
    service._cuda_stream = None
    service._cuda_graph_batch_size = 16
    service._cuda_graph_cache_size = 4
    service._action_batch_size_histogram = (
        league_module._empty_action_batch_size_histogram()
    )
    return service


def test_league_inference_close_ignores_cancelled_pending_future() -> None:
    cancelled: Future = Future()
    cancelled.cancel()
    pending: Future = Future()

    service = _service_stub()
    service._pending = [
        _request(cancelled),
        _request(pending),
    ]
    service._thread = _StoppedThread()

    service.close()

    assert cancelled.cancelled()
    assert pending.done()
    assert isinstance(pending.exception(), RuntimeError)
    assert service._models == {}
    assert service._completed_requests == 0
    assert service._failed_requests == 1


def test_league_inference_close_finishes_cleanup_after_cancellation_race() -> None:
    racing = _CancelBeforeSetExceptionFuture()
    trailing: Future = Future()

    service = _service_stub()
    service._pending = [_request(racing), _request(trailing)]
    service._thread = _StoppedThread()

    service.close()

    assert racing.cancelled()
    assert trailing.done()
    assert isinstance(trailing.exception(), RuntimeError)
    assert service._completed_requests == 0
    assert service._failed_requests == 1


def test_team_preview_batcher_close_can_be_retried_after_timeout() -> None:
    batcher = object.__new__(TeamPreviewBatcher)
    batcher._condition = Condition()
    batcher._closed = False
    batcher._pending = []
    batcher._thread = _RetryableCloseThread()
    batcher._CLOSE_TIMEOUT_SEC = 0.0

    with pytest.raises(RuntimeError, match="did not stop"):
        batcher.close()

    batcher.close()

    assert batcher._thread.join_calls == 2


def test_league_close_attempts_every_inference_service() -> None:
    calls: list[str] = []

    class _Service:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def close(self) -> None:
            calls.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

    league = object.__new__(League)
    league._inference = _Service("action", fail=True)
    league._preview_inference = _Service("preview")

    with pytest.raises(ExceptionGroup, match="inference services"):
        league.close()

    assert calls == ["action", "preview"]


def test_league_inference_submit_rejects_stopped_worker() -> None:
    service = _service_stub()
    service._thread = _StoppedThread()
    future: Future = Future()

    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        service._submit(_request(future))

    assert service._closed is True
    assert isinstance(service._fatal_error, RuntimeError)
    assert service._pending == []
    assert not future.done()


def test_league_inference_failure_resolves_pending_and_active_futures() -> None:
    service = _service_stub()
    pending: Future = Future()
    active: Future = Future()
    cancelled: Future = Future()
    cancelled.cancel()
    service._pending = [_request(pending), _request(cancelled)]
    service._active_requests = [_request(active)]

    error = RuntimeError("worker crashed")
    service._fail_service(error)

    assert service._closed is True
    assert service._fatal_error is error
    assert pending.done()
    assert isinstance(pending.exception(), RuntimeError)
    assert active.done()
    assert isinstance(active.exception(), RuntimeError)
    assert cancelled.cancelled()
    assert service._pending == []
    assert service._failed_requests == 2


def test_league_inference_describe_reports_pending_and_active_state() -> None:
    service = _service_stub()
    now = time.perf_counter()
    pending: Future = Future()
    active: Future = Future()
    service._pending = [_request(pending, submitted_at=now - 2.0)]
    service._active_requests = [_request(active, submitted_at=now - 1.0)]
    service._active_key = ("C:/tmp/step_1.pt", "battle")
    service._active_started_at = now - 1.0
    service._active_phase = "forward"
    service._active_phase_started_at = now - 0.5
    service._submitted_requests = 4
    service._completed_requests = 2

    description = service.describe()

    assert "thread_alive=True" in description
    assert "pending=1" in description
    assert "active=1" in description
    assert "active_key=step_1.pt:battle" in description
    assert "active_phase=forward" in description
    assert "submitted=4" in description
    assert "completed=2" in description


def test_league_inference_prioritizes_battle_requests() -> None:
    preview = _request(Future(), submitted_at=1.0)
    preview.kind = "preview"
    battle = _request(Future(), submitted_at=2.0)
    load = _request(Future(), submitted_at=0.5)
    load.kind = "load"

    ordered = sorted(
        [
            (("checkpoint.pt", "preview"), [preview]),
            (("checkpoint.pt", "load"), [load]),
            (("checkpoint.pt", "battle"), [battle]),
        ],
        key=_LeagueInferenceService._group_priority,
    )

    assert [key[1] for key, _ in ordered] == ["battle", "preview", "load"]


def test_league_inference_coalesce_target_is_checkpoint_and_kind_specific() -> None:
    service = _service_stub()
    service._coalesce_target_batch_size = 3
    cancelled: Future = Future()
    cancelled.cancel()

    service._pending = [
        _request(Future(), checkpoint_path="a.pt"),
        _request(Future(), checkpoint_path="b.pt"),
        _request(Future(), checkpoint_path="a.pt", kind="preview"),
        _request(cancelled, checkpoint_path="a.pt"),
    ]
    assert not service._coalesce_target_reached_locked()

    service._pending.extend(
        [
            _request(Future(), checkpoint_path="a.pt"),
            _request(Future(), checkpoint_path="a.pt"),
        ]
    )
    assert service._coalesce_target_reached_locked()


def test_league_inference_coalesce_target_can_be_disabled() -> None:
    service = _service_stub()
    service._coalesce_target_batch_size = None
    service._pending = [_request(Future()) for _ in range(32)]

    assert not service._coalesce_target_reached_locked()


def test_league_inference_target_wake_preserves_snapshot_membership() -> None:
    service = _service_stub()
    service._coalesce_window_sec = 60.0
    service._coalesce_target_batch_size = 2
    first = [_request(Future(), checkpoint_path="a.pt") for _ in range(2)]
    late = _request(Future(), checkpoint_path="a.pt")
    service._pending = list(first)
    calls: list[list[_InferenceRequest]] = []

    def run_battle_batch(
        checkpoint_path: str,
        requests: list[_InferenceRequest],
    ) -> None:
        assert checkpoint_path == "a.pt"
        calls.append(list(requests))
        for request in requests:
            request.future.set_result(None)
        with service._condition:
            if len(calls) == 1:
                service._pending.append(late)
                # The second cycle is intentionally deadline-free: this test
                # exercises target-triggered snapshot ownership, not waiting.
                service._coalesce_window_sec = 0.0
            else:
                service._closed = True

    service._run_battle_batch = run_battle_batch
    service._run_loop()

    assert calls == [first, [late]]
    assert service._coalesce_target_wakeups == 1


def test_league_inference_snapshot_does_not_late_merge_later_key() -> None:
    service = _service_stub()
    first_a = _request(Future(), checkpoint_path="a.pt", submitted_at=1.0)
    first_b = _request(Future(), checkpoint_path="b.pt", submitted_at=2.0)
    late_a = _request(Future(), checkpoint_path="a.pt", submitted_at=3.0)
    late_b = _request(Future(), checkpoint_path="b.pt", submitted_at=4.0)
    service._pending = [first_a, first_b]
    calls: list[tuple[str, list[_InferenceRequest]]] = []

    def run_battle_batch(
        checkpoint_path: str,
        requests: list[_InferenceRequest],
    ) -> None:
        calls.append((checkpoint_path, list(requests)))
        for request in requests:
            request.future.set_result(None)
        if len(calls) == 1:
            with service._condition:
                service._pending.extend((late_a, late_b))
        elif len(calls) == 4:
            with service._condition:
                service._closed = True

    service._run_battle_batch = run_battle_batch

    service._run_loop()

    assert calls == [
        ("a.pt", [first_a]),
        ("b.pt", [first_b]),
        ("a.pt", [late_a]),
        ("b.pt", [late_b]),
    ]


def test_league_inference_fatal_error_resolves_later_snapshot_group() -> None:
    service = _service_stub()
    active = _request(Future(), checkpoint_path="a.pt", submitted_at=1.0)
    future_key = _request(Future(), checkpoint_path="b.pt", submitted_at=2.0)
    service._pending = [active, future_key]

    def fail_batch(
        checkpoint_path: str,
        requests: list[_InferenceRequest],
    ) -> None:
        del checkpoint_path, requests
        raise KeyboardInterrupt("fatal worker failure")

    service._run_battle_batch = fail_batch

    service._run()

    assert isinstance(service._fatal_error, KeyboardInterrupt)
    assert service._closed is True
    assert service._pending == []
    assert service._active_requests == [active]
    assert service._failed_requests == 2
    for request in (active, future_key):
        assert request.future.done()
        assert isinstance(request.future.exception(), RuntimeError)


def test_league_inference_close_resolves_later_snapshot_group() -> None:
    service = _service_stub()
    service._thread = _StoppedThread()
    active = _request(Future(), checkpoint_path="a.pt", submitted_at=1.0)
    future_key = _request(Future(), checkpoint_path="b.pt", submitted_at=2.0)
    service._pending = [active, future_key]
    calls: list[str] = []

    def close_during_batch(
        checkpoint_path: str,
        requests: list[_InferenceRequest],
    ) -> None:
        del requests
        calls.append(checkpoint_path)
        service.close()

    service._run_battle_batch = close_during_batch

    service._run_loop()

    assert calls == ["a.pt"]
    assert service._pending == []
    for request in (active, future_key):
        assert request.future.done()
        assert isinstance(request.future.exception(), RuntimeError)
    assert service._completed_requests == 0
    assert service._failed_requests == 2


def test_league_inference_active_cancellation_does_not_fail_batch_peer() -> None:
    service = _service_stub()
    racing = _request(
        _CancelBeforeSetResultFuture(),
        checkpoint_path="a.pt",
        kind="load",
        submitted_at=1.0,
    )
    healthy = _request(
        Future(),
        checkpoint_path="a.pt",
        kind="load",
        submitted_at=2.0,
    )
    service._pending = [racing, healthy]
    service._get_agent = lambda _checkpoint_path: object()
    run_load_batch = service._run_load_batch

    def run_load_and_stop(
        checkpoint_path: str,
        requests: list[_InferenceRequest],
    ) -> None:
        run_load_batch(checkpoint_path, requests)
        with service._condition:
            service._closed = True

    service._run_load_batch = run_load_and_stop

    service._run_loop()

    assert racing.future.cancelled()
    assert healthy.future.result() is None
    assert service._completed_requests == 1
    assert service._failed_requests == 0


def test_league_inference_ordinary_failure_does_not_abort_snapshot() -> None:
    service = _service_stub()
    failing = _request(
        Future(),
        checkpoint_path="a.pt",
        submitted_at=1.0,
    )
    healthy = _request(
        Future(),
        checkpoint_path="b.pt",
        submitted_at=2.0,
    )
    service._pending = [failing, healthy]
    calls: list[str] = []

    def run_battle_batch(
        checkpoint_path: str,
        requests: list[_InferenceRequest],
    ) -> None:
        calls.append(checkpoint_path)
        if checkpoint_path == "a.pt":
            raise RuntimeError("isolated batch failure")
        service._try_set_future_result(requests[0].future, None)
        with service._condition:
            service._closed = True

    service._run_battle_batch = run_battle_batch

    service._run_loop()

    assert calls == ["a.pt", "b.pt"]
    assert isinstance(failing.future.exception(), RuntimeError)
    assert healthy.future.result() is None
    assert service._completed_requests == 1
    assert service._failed_requests == 1


def test_league_inference_prunes_model_references_inside_cuda_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    guard_active = False
    released_inside_guard: list[bool] = []

    @contextmanager
    def fake_cuda_model_execution(device: torch.device):
        nonlocal guard_active
        del device
        guard_active = True
        try:
            yield
        finally:
            guard_active = False

    class _TrackedModel:
        def __del__(self) -> None:
            released_inside_guard.append(guard_active)

    monkeypatch.setattr(
        league_module,
        "cuda_model_execution",
        fake_cuda_model_execution,
    )
    service = _service_stub()
    service._models = {"stale.pt": _TrackedModel()}
    future: Future = Future()
    future.set_running_or_notify_cancel()
    request = _InferenceRequest(
        checkpoint_path="",
        kind="prune",
        obs={},
        masks=None,
        hidden=None,
        future=future,
        keep_paths=set(),
    )

    service._run_prune_batch([request])

    assert service._models == {}
    assert released_inside_guard == [True]
    assert future.result() is None


def test_league_inference_prune_submission_does_not_wait_for_worker() -> None:
    service = _service_stub()

    service.prune_checkpoints(set())

    assert len(service._pending) == 1
    request = service._pending[0]
    assert request.kind == "prune"
    assert request.keep_paths == set()
    assert not request.future.done()


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

    futures = [Future(), Future()]
    for index, future in enumerate(futures):
        future.add_done_callback(
            lambda _future, index=index: events.append(("publish", str(index)))
        )
    requests = [
        _InferenceRequest(
            checkpoint_path="checkpoint.pt",
            kind="battle",
            obs={"x": np.full((2,), index, dtype=np.float32)},
            masks={
                "joint": np.ones((3, 4), dtype=bool),
                "can_tera": np.ones((2,), dtype=bool),
            },
            hidden=None,
            future=future,
        )
        for index, future in enumerate(futures)
    ]

    service._run_battle_batch("checkpoint.pt", requests)

    assert events == [
        ("enter", "cpu"),
        ("sync", "cpu"),
        ("slice", "0"),
        ("slice", "1"),
        ("exit", "cpu"),
        ("publish", "0"),
        ("publish", "1"),
    ]
    assert agent.kwargs == {
        "greedy": False,
        "include_value": False,
        "include_policy_stats": False,
    }
    for index, future in enumerate(futures):
        actions, hidden = future.result()
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
    schema_version: int = 1,
) -> list[_InferenceRequest]:
    requests = []
    for index in range(batch_size):
        value = offset + index
        if schema_version == 1:
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
            _InferenceRequest(
                checkpoint_path="checkpoint.pt",
                kind="battle",
                obs=obs,
                masks=masks,
                hidden=None,
                future=Future(),
            )
        )
    return requests


def _reference_stacked_batch(requests: list[_InferenceRequest]):
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
    requests: list[_InferenceRequest],
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
    for index, request in enumerate(requests):
        actions, hidden = request.future.result()
        np.testing.assert_array_equal(actions, expected_actions[index].numpy())
        assert hidden is not None
        assert hidden[0].item() == pytest.approx(float(expected_obs["x"][index, 0]))


def test_league_staging_reuses_capacity_without_exposing_stale_rows() -> None:
    service = _service_stub()
    agent = _RecordingBatchAgent()
    service._get_agent = lambda _checkpoint_path: agent

    first = _staging_requests(3, offset=10)
    service._run_battle_batch("checkpoint.pt", first)
    _assert_recorded_batch_matches_reference(agent, first)
    first_host = service._host_batch_buffers["battle_obs"]
    first_pointers = {
        key: value.__array_interface__["data"][0]
        for key, value in first_host.arrays.items()
    }
    assert first_host.capacity == 3

    smaller = _staging_requests(1, offset=100)
    service._run_battle_batch("checkpoint.pt", smaller)
    _assert_recorded_batch_matches_reference(agent, smaller)
    reused_host = service._host_batch_buffers["battle_obs"]
    assert reused_host is first_host
    assert {
        key: value.__array_interface__["data"][0]
        for key, value in reused_host.arrays.items()
    } == first_pointers
    assert agent.calls[-1][0]["x"].shape == (1, 2)

    larger = _staging_requests(5, offset=200)
    service._run_battle_batch("checkpoint.pt", larger)
    _assert_recorded_batch_matches_reference(agent, larger)
    grown_host = service._host_batch_buffers["battle_obs"]
    assert grown_host is not first_host
    assert grown_host.capacity >= 5

    changed_schema = _staging_requests(2, offset=300, schema_version=2)
    service._run_battle_batch("checkpoint.pt", changed_schema)
    _assert_recorded_batch_matches_reference(agent, changed_schema)
    rebuilt_host = service._host_batch_buffers["battle_obs"]
    assert rebuilt_host is not grown_host
    assert rebuilt_host.schema != grown_host.schema
    assert rebuilt_host.capacity == 2


def test_action_batch_size_profile_accumulates_once_per_batch() -> None:
    class _Timing:
        def __init__(self) -> None:
            self.values = defaultdict(float)

        def add(self, key: str, value: float) -> None:
            self.values[key] += value

        def increment(self, key: str, delta: int = 1) -> None:
            self.values[key] += delta

    service = _service_stub()
    service._get_agent = lambda _checkpoint_path: _RecordingBatchAgent()
    timing = _Timing()
    requests = _staging_requests(4, offset=0)
    for request in requests:
        request.timing = timing

    service._run_battle_batch("checkpoint.pt", requests)

    assert timing.values["opponent_infer_action_requests"] == 4
    assert timing.values["opponent_infer_action_batches"] == pytest.approx(1.0)
    assert timing.values["opponent_infer_action_batch_size_sum"] == pytest.approx(4.0)
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
        "coalesce_target_batch_size": None,
        "coalesce_target_wakeups": 0,
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
    service._run_battle_batch("checkpoint.pt", first)
    _assert_recorded_batch_matches_reference(agent, first)
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
    service._run_battle_batch("checkpoint.pt", smaller)
    _assert_recorded_batch_matches_reference(agent, smaller)
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


def test_league_uses_same_process_cuda_inference(tmp_path) -> None:
    game_data = GameData(gen=9)
    agent = PokeAgent(ModelConfig(), game_data)
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league = League(config, EnvConfig(), game_data, agent, torch.device("cuda"))
    try:
        assert league.device.type == "cuda"
        assert league.inference_device.type == "cuda"
        assert isinstance(league._inference, _LeagueInferenceService)
        assert league._inference._coalesce_target_batch_size == 16
        assert league.preview_inference_device.type == "cuda"
        assert league._preview_inference is league._inference
    finally:
        league.close()


def test_league_can_use_separate_cpu_preview_inference(tmp_path) -> None:
    game_data = GameData(gen=9)
    agent = PokeAgent(ModelConfig(), game_data)
    config = TrainingConfig(
        checkpoint_dir=str(tmp_path),
        league_preview_device="cpu",
    )
    league = League(config, EnvConfig(), game_data, agent, torch.device("cuda"))
    try:
        assert league.inference_device.type == "cuda"
        assert league.preview_inference_device.type == "cpu"
        assert league._preview_inference is not league._inference
        assert league._inference._coalesce_target_batch_size == 16
        assert league._preview_inference._coalesce_target_batch_size is None
    finally:
        league.close()


def test_league_reuses_action_service_for_cpu_preview_inference(tmp_path) -> None:
    game_data = GameData(gen=9)
    agent = PokeAgent(ModelConfig(), game_data)
    config = TrainingConfig(checkpoint_dir=str(tmp_path))
    league = League(config, EnvConfig(), game_data, agent, torch.device("cpu"))
    try:
        assert league.inference_device.type == "cpu"
        assert league.preview_inference_device.type == "cpu"
        assert league._preview_inference is league._inference
        assert league._inference._coalesce_target_batch_size is None
    finally:
        league.close()


def test_league_player_submits_previews_to_preview_service(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    obs = {"context": np.array([1, 2, 3], dtype=np.int64)}
    preview_future: Future = Future()

    class _ActionService:
        def submit_team_preview(self, *args, **kwargs):
            raise AssertionError("preview should not use the action inference service")

    class _PreviewService:
        def __init__(self) -> None:
            self.calls = []

        def submit_team_preview(self, checkpoint_path, request_obs, timing=None):
            self.calls.append((checkpoint_path, request_obs, timing))
            return preview_future

    preview_service = _PreviewService()
    monkeypatch.setattr(
        league_module,
        "build_team_preview_observation",
        lambda gd, team_size, own_packed_team, opponent_packed_team: obs,
    )

    player = LeaguePlayer.__new__(LeaguePlayer)
    player.checkpoint_path = "checkpoint.pt"
    player.inference_service = _ActionService()
    player.preview_inference_service = preview_service
    player.obs_builder = type("_ObsBuilder", (), {"gd": object(), "team_size": 6})()
    player._pending_checkpoint_path = None
    player._prepared_preview_obs = None
    player._prepared_preview_future = None
    player._prepared_preview_started_at = None
    player._timing = None

    LeaguePlayer.prepare_team_preview(player, "own", "opp")

    assert player._prepared_preview_obs is obs
    assert player._prepared_preview_future is preview_future
    assert len(preview_service.calls) == 1
    assert preview_service.calls[0] == ("checkpoint.pt", obs, None)


def test_team_preview_batcher_requests_pick_only_inference() -> None:
    class _FakeAgent:
        def __init__(self) -> None:
            self.kwargs = None

        def team_preview(self, obs_batch, **kwargs):
            self.kwargs = kwargs
            batch_size = next(iter(obs_batch.values())).shape[0]
            return {"picks": torch.zeros((batch_size, 4), dtype=torch.long)}

    agent = _FakeAgent()
    batcher = TeamPreviewBatcher(agent, torch.device("cpu"))
    try:
        result = batcher.choose({"context": np.array([1, 2, 3], dtype=np.int64)})
    finally:
        batcher.close()

    np.testing.assert_array_equal(result.picks, np.zeros(4, dtype=np.int64))
    assert result.policy_version == 0
    assert agent.kwargs == {
        "include_value": False,
        "include_policy_stats": False,
        "include_opponent_preview": False,
    }


def test_team_preview_batcher_rejects_training_mode_without_dying() -> None:
    class _FakeAgent:
        training = True

        def team_preview(self, obs_batch, **kwargs):
            del kwargs
            batch_size = next(iter(obs_batch.values())).shape[0]
            return {"picks": torch.zeros((batch_size, 4), dtype=torch.long)}

    agent = _FakeAgent()
    batcher = TeamPreviewBatcher(agent, torch.device("cpu"))
    try:
        with pytest.raises(RuntimeError, match="training mode"):
            batcher.choose({"context": np.array([1, 2, 3], dtype=np.int64)})

        agent.training = False
        result = batcher.choose({"context": np.array([1, 2, 3], dtype=np.int64)})
    finally:
        batcher.close()

    np.testing.assert_array_equal(result.picks, np.zeros(4, dtype=np.int64))
    assert result.policy_version == 0


def test_team_preview_batcher_ignores_cancelled_request(monkeypatch) -> None:
    class _FakeAgent:
        training = False

        def team_preview(self, obs_batch, **kwargs):
            del kwargs
            batch_size = next(iter(obs_batch.values())).shape[0]
            return {"picks": torch.zeros((batch_size, 4), dtype=torch.long)}

    monkeypatch.setattr(TeamPreviewBatcher, "_COALESCE_WINDOW_SEC", 0.05)
    batcher = TeamPreviewBatcher(_FakeAgent(), torch.device("cpu"))
    try:
        cancelled = batcher.submit({"context": np.array([1, 2, 3], dtype=np.int64)})
        assert cancelled.cancel()
        healthy = batcher.submit({"context": np.array([4, 5, 6], dtype=np.int64)})
        result = healthy.result(timeout=1.0)
    finally:
        batcher.close()

    assert cancelled.cancelled()
    np.testing.assert_array_equal(result.picks, np.zeros(4, dtype=np.int64))
    assert result.policy_version == 0


def test_team_preview_batcher_tags_results_across_policy_versions() -> None:
    class _FakeAgent:
        training = False

        def team_preview(self, obs_batch, **kwargs):
            del kwargs
            batch_size = next(iter(obs_batch.values())).shape[0]
            return {"picks": torch.zeros((batch_size, 4), dtype=torch.long)}

    batcher = TeamPreviewBatcher(
        _FakeAgent(),
        torch.device("cpu"),
        policy_version=8,
    )
    try:
        old_result = batcher.choose({"context": np.array([1, 2, 3], dtype=np.int64)})
        assert batcher.advance_policy_version() == 9
        new_result = batcher.choose({"context": np.array([4, 5, 6], dtype=np.int64)})
    finally:
        batcher.close()

    assert isinstance(old_result, TeamPreviewResult)
    assert old_result.policy_version == 8
    assert new_result.policy_version == 9


def test_team_preview_batcher_refuses_version_advance_with_pending_work(
    monkeypatch,
) -> None:
    class _FakeAgent:
        training = False

        def team_preview(self, obs_batch, **kwargs):
            del kwargs
            batch_size = next(iter(obs_batch.values())).shape[0]
            return {"picks": torch.zeros((batch_size, 4), dtype=torch.long)}

    monkeypatch.setattr(TeamPreviewBatcher, "_COALESCE_WINDOW_SEC", 0.05)
    batcher = TeamPreviewBatcher(_FakeAgent(), torch.device("cpu"))
    try:
        pending = batcher.submit({"context": np.array([1, 2, 3], dtype=np.int64)})
        with pytest.raises(RuntimeError, match="pending or active"):
            batcher.advance_policy_version()
        result = pending.result(timeout=1.0)
        assert batcher.advance_policy_version() == 1
    finally:
        batcher.close()

    assert result.policy_version == 0
