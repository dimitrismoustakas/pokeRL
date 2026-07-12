"""Self-play league: a pool of past agent checkpoints as opponents."""

import asyncio
import copy
import json
import random
import time
from collections import deque
from contextlib import nullcontext
from concurrent.futures import CancelledError, Future, InvalidStateError
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread

import numpy as np
import torch
from poke_env.battle import DoubleBattle
from poke_env.player import DoubleBattleOrder, PassBattleOrder, Player

from ..model.agent import PokeAgent, load_state_dict_compatible
from ..data import GameData
from ..config import TrainingConfig, EnvConfig
from ..env.observation import ObservationBuilder, build_team_preview_observation
from ..env.actions import (
    PASS_ACTION,
    TERA_NONE,
    action_to_order,
    build_legal_action_state,
    build_policy_action_masks,
    decode_tera_choice,
    is_default_only_action_state,
)
from ..env.showdown_env import (
    _NoopBattleOrder,
    make_battlestream_configuration,
    make_local_account_configuration,
    showdown_log_level,
)
from .cuda_execution import cuda_model_execution


@dataclass
class _InferenceRequest:
    checkpoint_path: str
    kind: str
    obs: dict[str, np.ndarray]
    masks: dict[str, np.ndarray] | None
    hidden: tuple[torch.Tensor, torch.Tensor] | None
    future: Future
    timing: object | None = None
    submitted_at: float = 0.0
    keep_paths: set[str] | None = None


_BatchSchema = tuple[tuple[str, tuple[int, ...], str], ...]


@dataclass
class _HostBatchBuffers:
    schema: _BatchSchema
    capacity: int
    arrays: dict[str, np.ndarray]
    tensors: dict[str, torch.Tensor]
    group_arrays: dict[str, np.ndarray]
    group_tensors: dict[str, torch.Tensor]


@dataclass
class _DeviceBatchBuffers:
    schema: _BatchSchema
    capacity: int
    force_bool: bool
    tensors: dict[str, torch.Tensor]
    group_tensors: dict[str, torch.Tensor]


def _same_device(left: torch.device, right: torch.device) -> bool:
    return left.type == right.type and (
        right.index is None or left.index == right.index
    )


class _LeagueInferenceService:
    """Shared batched inference for league opponents.

    A single worker thread batches requests per checkpoint and runs them on the
    configured device. This preserves the opponent policy while avoiding one
    tiny GPU forward per environment thread.
    """

    _CLOSE_TIMEOUT_SEC = 60.0

    def __init__(
        self,
        agent_template: PokeAgent,
        device: torch.device,
        coalesce_window_sec: float,
    ):
        self._agent_template = copy.deepcopy(agent_template).to(torch.device("cpu"))
        self._agent_template.eval()
        self._agent_template.requires_grad_(False)
        self._device = device
        self._cuda_stream: torch.cuda.Stream | None = None
        self._coalesce_window_sec = max(0.0, coalesce_window_sec)
        self._models: dict[str, PokeAgent] = {}
        self._host_batch_buffers: dict[str, _HostBatchBuffers] = {}
        self._device_batch_buffers: dict[str, _DeviceBatchBuffers] = {}
        self._prefetch_paths: set[str] = set()
        self._pending: list[_InferenceRequest] = []
        self._active_requests: list[_InferenceRequest] = []
        self._active_key: tuple[str, str] | None = None
        self._active_started_at: float | None = None
        self._active_phase: str | None = None
        self._active_phase_started_at: float | None = None
        self._fatal_error: BaseException | None = None
        self._submitted_requests = 0
        self._completed_requests = 0
        self._failed_requests = 0
        self._last_progress_at = time.perf_counter()
        self._closed = False
        self._condition = Condition()
        self._thread = Thread(target=self._run, name="league-inference", daemon=True)
        self._thread.start()

    def ensure_checkpoint(self, checkpoint_path: str) -> None:
        with self._condition:
            self._raise_if_unavailable_locked()
            if checkpoint_path in self._models:
                return

        agent_copy = self._load_agent(checkpoint_path)

        with self._condition:
            self._raise_if_unavailable_locked()
            self._models.setdefault(checkpoint_path, agent_copy)

    def prefetch_checkpoint(self, checkpoint_path: str) -> None:
        future: Future = Future()
        with self._condition:
            self._raise_if_unavailable_locked()
            if (
                checkpoint_path in self._models
                or checkpoint_path in self._prefetch_paths
            ):
                return
            self._prefetch_paths.add(checkpoint_path)
            future.set_running_or_notify_cancel()
            request = _InferenceRequest(
                checkpoint_path=checkpoint_path,
                kind="load",
                obs={},
                masks=None,
                hidden=None,
                future=future,
                timing=None,
                submitted_at=time.perf_counter(),
            )
            self._pending.append(request)
            self._submitted_requests += 1
            self._condition.notify()

    def prune_checkpoints(self, keep_paths: set[str]) -> None:
        future: Future = Future()
        with self._condition:
            self._raise_if_unavailable_locked()
            future.set_running_or_notify_cancel()
            self._pending.append(
                _InferenceRequest(
                    checkpoint_path="",
                    kind="prune",
                    obs={},
                    masks=None,
                    hidden=None,
                    future=future,
                    keep_paths=set(keep_paths),
                    submitted_at=time.perf_counter(),
                )
            )
            self._submitted_requests += 1
            self._condition.notify()

    def infer_action(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        timing: object | None = None,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor] | None]:
        return self.submit_action(
            checkpoint_path,
            obs,
            masks,
            hidden,
            timing=timing,
        ).result()

    def submit_action(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
        masks: dict[str, np.ndarray],
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        timing: object | None = None,
    ) -> Future:
        future: Future = Future()
        self._submit(
            _InferenceRequest(
                checkpoint_path=checkpoint_path,
                kind="battle",
                obs=obs,
                masks=masks,
                hidden=hidden,
                future=future,
                timing=timing,
            )
        )
        return future

    def infer_team_preview(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
        timing: object | None = None,
    ) -> np.ndarray:
        return self.submit_team_preview(checkpoint_path, obs, timing=timing).result()

    def submit_team_preview(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
        timing: object | None = None,
    ) -> Future:
        future: Future = Future()
        self._submit(
            _InferenceRequest(
                checkpoint_path=checkpoint_path,
                kind="preview",
                obs=obs,
                masks=None,
                hidden=None,
                future=future,
                timing=timing,
            )
        )
        return future

    def close(self) -> None:
        with self._condition:
            already_closed = self._closed
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            active = list(self._active_requests)
            self._condition.notify_all()

        if not already_closed:
            failed = 0
            for request in pending + active:
                failed += self._try_set_future_exception(
                    request.future,
                    RuntimeError("League inference service closed."),
                )
            self._record_failed(failed)

        if self._thread.is_alive():
            self._thread.join(timeout=self._CLOSE_TIMEOUT_SEC)
        if self._thread.is_alive():
            raise RuntimeError(
                "League inference worker did not stop within "
                f"{self._CLOSE_TIMEOUT_SEC:g} seconds."
            )

        # Releasing the final references to CUDA models can enqueue allocator
        # work. Keep that destruction in the same execution domain as model
        # inference and learner work.
        with cuda_model_execution(self._device):
            with self._condition:
                self._models.clear()
            getattr(self, "_device_batch_buffers", {}).clear()
        getattr(self, "_host_batch_buffers", {}).clear()

    def _submit(self, request: _InferenceRequest) -> None:
        with self._condition:
            self._raise_if_unavailable_locked()
            request.submitted_at = time.perf_counter()
            self._pending.append(request)
            self._submitted_requests += 1
            self._condition.notify()

    def _run(self) -> None:
        try:
            self._run_loop()
        except BaseException as exc:
            self._fail_service(exc)

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                if self._coalesce_window_sec > 0:
                    deadline = time.perf_counter() + self._coalesce_window_sec
                    while not self._closed:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                batch = [
                    request
                    for request in self._pending
                    if not request.future.cancelled()
                ]
                # Keep snapshot members visible to close() and _fail_service()
                # until each group atomically becomes active. The separate
                # list freezes membership so later arrivals wait for the next
                # coalescing cycle, matching the legacy batching policy.
                self._pending = list(batch)

            grouped: dict[tuple[str, str], list[_InferenceRequest]] = {}
            for request in batch:
                key = (request.checkpoint_path, request.kind)
                grouped.setdefault(key, []).append(request)

            for (checkpoint_path, kind), eligible_requests in sorted(
                grouped.items(), key=self._group_priority
            ):
                key = (checkpoint_path, kind)
                with self._condition:
                    if self._closed:
                        return
                    requests = self._activate_snapshot_group_locked(
                        key,
                        eligible_requests,
                    )
                    if not requests:
                        continue

                clear_active = False
                try:
                    try:
                        if kind == "battle":
                            self._run_battle_batch(checkpoint_path, requests)
                        elif kind == "preview":
                            self._run_preview_batch(checkpoint_path, requests)
                        elif kind == "load":
                            self._run_load_batch(checkpoint_path, requests)
                        elif kind == "prune":
                            self._run_prune_batch(requests)
                        else:
                            raise ValueError(
                                f"Unknown inference request kind: {kind}"
                            )
                    except Exception as exc:
                        failed = 0
                        for request in requests:
                            failed += self._try_set_future_exception(
                                request.future,
                                exc,
                            )
                        self._record_failed(failed)
                    else:
                        self._record_completed(requests)
                    clear_active = True
                finally:
                    if kind == "load":
                        with self._condition:
                            self._prefetch_paths.discard(checkpoint_path)
                    # A BaseException is handled by _run(), which needs the
                    # active batch to remain discoverable for failure delivery.
                    if clear_active:
                        self._clear_active(key, requests)

    def _activate_snapshot_group_locked(
        self,
        key: tuple[str, str],
        eligible_requests: list[_InferenceRequest],
    ) -> list[_InferenceRequest]:
        eligible_ids = {id(request) for request in eligible_requests}
        requests: list[_InferenceRequest] = []
        remaining: list[_InferenceRequest] = []
        for request in self._pending:
            if id(request) in eligible_ids:
                if not request.future.cancelled():
                    requests.append(request)
            else:
                remaining.append(request)
        if requests:
            # Mark the batch active before removing it from pending so every
            # request stays discoverable throughout the ownership transfer.
            self._mark_active_locked(key, requests)
        self._pending = remaining
        return requests

    def describe(self) -> str:
        now = time.perf_counter()
        with self._condition:
            thread_alive = self._thread.is_alive()
            pending_count = len(self._pending)
            active_count = len(self._active_requests)
            active_key = self._active_key
            active_phase = self._active_phase
            active_age = (
                None
                if self._active_started_at is None
                else max(0.0, now - self._active_started_at)
            )
            active_phase_age = (
                None
                if self._active_phase_started_at is None
                else max(0.0, now - self._active_phase_started_at)
            )
            oldest_pending_age = None
            if self._pending:
                oldest_pending_age = max(
                    0.0,
                    now - min(request.submitted_at for request in self._pending),
                )
            fatal_error = (
                repr(self._fatal_error) if self._fatal_error is not None else None
            )
            last_progress_age = max(0.0, now - self._last_progress_at)
            submitted = self._submitted_requests
            completed = self._completed_requests
            failed = self._failed_requests
            closed = self._closed
            model_count = len(self._models)

        active_key_text = None
        if active_key is not None:
            checkpoint_path, kind = active_key
            active_key_text = f"{Path(checkpoint_path).name}:{kind}"
        return (
            f"thread_alive={thread_alive} closed={closed} fatal_error={fatal_error} "
            f"pending={pending_count} active={active_count} active_key={active_key_text} "
            f"active_phase={active_phase} active_age_sec={active_age} "
            f"active_phase_age_sec={active_phase_age} "
            f"oldest_pending_age_sec={oldest_pending_age} "
            f"submitted={submitted} completed={completed} failed={failed} "
            f"model_count={model_count} last_progress_age_sec={last_progress_age}"
        )

    def _raise_if_unavailable_locked(self) -> None:
        if self._fatal_error is not None:
            raise RuntimeError(
                "League inference service failed."
            ) from self._fatal_error
        if self._closed:
            raise RuntimeError("League inference service is closed.")
        if not self._thread.is_alive():
            self._fatal_error = RuntimeError(
                "League inference service worker stopped unexpectedly."
            )
            self._closed = True
            raise RuntimeError("League inference service worker stopped unexpectedly.")

    def _fail_service(self, error: BaseException) -> None:
        with self._condition:
            if self._fatal_error is None:
                self._fatal_error = error
            self._closed = True
            pending = list(self._pending)
            active = list(self._active_requests)
            self._pending.clear()
            self._condition.notify_all()
        failed = 0
        for request in pending + active:
            failed += self._try_set_future_exception(
                request.future,
                RuntimeError("League inference service worker failed."),
            )
        self._record_failed(failed)

    def _mark_active_locked(
        self,
        key: tuple[str, str],
        requests: list[_InferenceRequest],
    ) -> None:
        self._active_key = key
        self._active_requests = list(requests)
        self._active_started_at = time.perf_counter()
        self._active_phase = "starting"
        self._active_phase_started_at = self._active_started_at

    def _clear_active(
        self,
        key: tuple[str, str],
        requests: list[_InferenceRequest],
    ) -> None:
        with self._condition:
            if self._active_key == key and self._active_requests == requests:
                self._active_key = None
                self._active_requests = []
                self._active_started_at = None
                self._active_phase = None
                self._active_phase_started_at = None
            self._last_progress_at = time.perf_counter()

    def _record_completed(self, requests: list[_InferenceRequest]) -> None:
        completed = sum(
            1
            for request in requests
            if self._future_completed_successfully(request.future)
        )
        with self._condition:
            self._completed_requests += completed
            self._last_progress_at = time.perf_counter()

    def _record_failed(self, count: int) -> None:
        with self._condition:
            self._failed_requests += count
            self._last_progress_at = time.perf_counter()

    @staticmethod
    def _try_set_future_result(future: Future, result: object) -> bool:
        """Publish a result unless cancellation or another owner won the race."""
        try:
            future.set_result(result)
        except InvalidStateError:
            return False
        return True

    @staticmethod
    def _try_set_future_exception(future: Future, error: BaseException) -> bool:
        """Publish an error unless cancellation or another owner won the race."""
        try:
            future.set_exception(error)
        except InvalidStateError:
            return False
        return True

    @staticmethod
    def _future_completed_successfully(future: Future) -> bool:
        if not future.done() or future.cancelled():
            return False
        try:
            return future.exception() is None
        except CancelledError:
            return False

    def _set_active_phase(self, phase: str) -> None:
        with self._condition:
            self._active_phase = phase
            self._active_phase_started_at = time.perf_counter()

    @staticmethod
    def _group_priority(
        item: tuple[tuple[str, str], list[_InferenceRequest]],
    ) -> tuple[int, float]:
        (_, kind), requests = item
        priority = {"battle": 0, "preview": 1, "load": 2, "prune": 3}.get(kind, 99)
        oldest = min((request.submitted_at for request in requests), default=0.0)
        return priority, oldest

    @staticmethod
    def _batch_schema(
        mappings: list[dict[str, np.ndarray]],
        *,
        cache_key: str,
    ) -> _BatchSchema:
        if not mappings:
            raise ValueError(f"Cannot pack an empty {cache_key} batch.")
        keys = tuple(mappings[0])
        if not keys:
            raise ValueError(f"Cannot pack {cache_key} without tensor fields.")
        schema = tuple(
            (key, tuple(mappings[0][key].shape), mappings[0][key].dtype.str)
            for key in keys
        )
        for batch_index, mapping in enumerate(mappings[1:], start=1):
            if tuple(mapping) != keys:
                raise ValueError(
                    f"{cache_key} request {batch_index} has keys {tuple(mapping)}, "
                    f"expected {keys}."
                )
            for key, expected_shape, expected_dtype in schema:
                value = mapping[key]
                if tuple(value.shape) != expected_shape or value.dtype.str != expected_dtype:
                    raise ValueError(
                        f"{cache_key}[{key!r}] request {batch_index} has "
                        f"shape={tuple(value.shape)} dtype={value.dtype}, expected "
                        f"shape={expected_shape} dtype={mappings[0][key].dtype}."
                    )
        return schema

    def _pack_cpu_batch(
        self,
        cache_key: str,
        mappings: list[dict[str, np.ndarray]],
    ) -> tuple[_HostBatchBuffers, int]:
        """Pack one request batch into reusable worker-private CPU storage."""
        schema = self._batch_schema(mappings, cache_key=cache_key)
        batch_size = len(mappings)
        caches = getattr(self, "_host_batch_buffers", None)
        if caches is None:
            caches = {}
            self._host_batch_buffers = caches
        buffers = caches.get(cache_key)
        if buffers is None or buffers.schema != schema:
            buffers = self._allocate_host_batch_buffers(schema, batch_size)
            caches[cache_key] = buffers
        elif batch_size > buffers.capacity:
            capacity = max(batch_size, 2 * buffers.capacity)
            buffers = self._allocate_host_batch_buffers(schema, capacity)
            caches[cache_key] = buffers

        for key, _, _ in schema:
            np.stack(
                [mapping[key] for mapping in mappings],
                axis=0,
                out=buffers.arrays[key][:batch_size],
                casting="no",
            )
        return buffers, batch_size

    @staticmethod
    def _allocate_host_batch_buffers(
        schema: _BatchSchema,
        capacity: int,
    ) -> _HostBatchBuffers:
        group_sizes: dict[str, int] = {}
        for _, shape, dtype in schema:
            row_size = int(np.prod(shape, dtype=np.int64))
            group_sizes[dtype] = group_sizes.get(dtype, 0) + capacity * row_size
        group_arrays = {
            dtype: np.empty(size, dtype=np.dtype(dtype))
            for dtype, size in group_sizes.items()
        }
        group_tensors = {
            dtype: torch.from_numpy(value) for dtype, value in group_arrays.items()
        }
        group_offsets = dict.fromkeys(group_sizes, 0)
        arrays: dict[str, np.ndarray] = {}
        tensors: dict[str, torch.Tensor] = {}
        for key, shape, dtype in schema:
            row_size = int(np.prod(shape, dtype=np.int64))
            field_size = capacity * row_size
            offset = group_offsets[dtype]
            arrays[key] = group_arrays[dtype][offset : offset + field_size].reshape(
                capacity,
                *shape,
            )
            tensors[key] = group_tensors[dtype][offset : offset + field_size].view(
                capacity,
                *shape,
            )
            group_offsets[dtype] = offset + field_size
        return _HostBatchBuffers(
            schema=schema,
            capacity=capacity,
            arrays=arrays,
            tensors=tensors,
            group_arrays=group_arrays,
            group_tensors=group_tensors,
        )

    def _copy_staged_batch_to_device(
        self,
        cache_key: str,
        host_buffers: _HostBatchBuffers,
        batch_size: int,
        *,
        force_bool: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Copy active rows to reusable device buffers with blocking semantics."""
        active_host = {
            key: host_buffers.tensors[key][:batch_size]
            for key, _, _ in host_buffers.schema
        }
        if self._device.type == "cpu" and (
            not force_bool
            or all(value.dtype == torch.bool for value in active_host.values())
        ):
            return active_host

        caches = getattr(self, "_device_batch_buffers", None)
        if caches is None:
            caches = {}
            self._device_batch_buffers = caches
        buffers = caches.get(cache_key)
        if (
            buffers is None
            or buffers.schema != host_buffers.schema
            or buffers.capacity < host_buffers.capacity
            or buffers.force_bool != force_bool
        ):
            buffers = self._allocate_device_batch_buffers(
                host_buffers,
                force_bool=force_bool,
            )
            caches[cache_key] = buffers

        if not force_bool:
            for group_key, source in host_buffers.group_tensors.items():
                buffers.group_tensors[group_key].copy_(source)
        elif all(value.dtype == torch.bool for value in host_buffers.tensors.values()):
            source = next(iter(host_buffers.group_tensors.values()))
            next(iter(buffers.group_tensors.values())).copy_(source)
        else:
            for key, source in host_buffers.tensors.items():
                buffers.tensors[key].copy_(source)
        return {
            key: buffers.tensors[key][:batch_size]
            for key, _, _ in host_buffers.schema
        }

    def _allocate_device_batch_buffers(
        self,
        host_buffers: _HostBatchBuffers,
        *,
        force_bool: bool,
    ) -> _DeviceBatchBuffers:
        group_sizes: dict[str, int] = {}
        group_dtypes: dict[str, torch.dtype] = {}
        field_groups: dict[str, str] = {}
        for key, shape, dtype in host_buffers.schema:
            group_key = "bool" if force_bool else dtype
            row_size = int(np.prod(shape, dtype=np.int64))
            group_sizes[group_key] = (
                group_sizes.get(group_key, 0) + host_buffers.capacity * row_size
            )
            group_dtypes[group_key] = (
                torch.bool if force_bool else host_buffers.tensors[key].dtype
            )
            field_groups[key] = group_key
        group_tensors = {
            group_key: torch.empty(
                size,
                dtype=group_dtypes[group_key],
                device=self._device,
            )
            for group_key, size in group_sizes.items()
        }
        group_offsets = dict.fromkeys(group_sizes, 0)
        tensors: dict[str, torch.Tensor] = {}
        for key, shape, _ in host_buffers.schema:
            group_key = field_groups[key]
            row_size = int(np.prod(shape, dtype=np.int64))
            field_size = host_buffers.capacity * row_size
            offset = group_offsets[group_key]
            tensors[key] = group_tensors[group_key][
                offset : offset + field_size
            ].view(host_buffers.capacity, *shape)
            group_offsets[group_key] = offset + field_size
        return _DeviceBatchBuffers(
            schema=host_buffers.schema,
            capacity=host_buffers.capacity,
            force_bool=force_bool,
            tensors=tensors,
            group_tensors=group_tensors,
        )

    def _device_execution_context(self):
        if self._device.type != "cuda":
            return nullcontext()
        if self._cuda_stream is None:
            self._cuda_stream = torch.cuda.Stream(device=self._device)
        return torch.cuda.stream(self._cuda_stream)

    def _synchronize_device_execution(self) -> None:
        if self._device.type == "cuda" and self._cuda_stream is not None:
            self._cuda_stream.synchronize()

    def _run_battle_batch(
        self, checkpoint_path: str, requests: list[_InferenceRequest]
    ) -> None:
        self._set_active_phase("get_agent")
        agent = self._get_agent(checkpoint_path)
        batch_started = time.perf_counter()
        batch_size = len(requests)
        for request in requests:
            self._increment_timing(request.timing, "opponent_infer_action_requests")
            self._add_timing(
                request.timing, "opponent_infer_action_batches", 1.0 / batch_size
            )
            self._add_timing(
                request.timing,
                "opponent_infer_action_batch_size_sum",
                float(batch_size),
            )
            self._increment_timing(
                request.timing, "opponent_infer_action_queue_wait_calls"
            )
            self._add_timing(
                request.timing,
                "opponent_infer_action_queue_wait_sec",
                max(0.0, batch_started - request.submitted_at),
            )

        if requests[0].masks is None:
            raise RuntimeError("Battle inference request was missing action masks.")
        request_masks = []
        for request in requests:
            if request.masks is None:
                raise RuntimeError("Battle inference request was missing action masks.")
            request_masks.append(request.masks)

        self._set_active_phase("cpu_pack")
        cpu_pack_started = time.perf_counter()
        obs_host, obs_batch_size = self._pack_cpu_batch(
            "battle_obs",
            [request.obs for request in requests],
        )
        masks_host, masks_batch_size = self._pack_cpu_batch(
            "battle_masks",
            request_masks,
        )
        if obs_batch_size != batch_size or masks_batch_size != batch_size:
            raise RuntimeError("Battle inference staging produced a mismatched batch size.")
        cpu_pack_sec = time.perf_counter() - cpu_pack_started
        self._add_batch_timing(
            requests,
            "opponent_infer_action_cpu_pack_sec",
            cpu_pack_sec,
        )

        self._set_active_phase("cuda_lock_wait")
        with cuda_model_execution(self._device):
            self._set_active_phase("device_copy")
            device_copy_started = time.perf_counter()
            obs_batch = self._copy_staged_batch_to_device(
                "battle_obs",
                obs_host,
                batch_size,
            )
            masks_batch = self._copy_staged_batch_to_device(
                "battle_masks",
                masks_host,
                batch_size,
                force_bool=True,
            )
            device_copy_sec = time.perf_counter() - device_copy_started
            self._add_batch_timing(
                requests,
                "opponent_infer_action_device_copy_sec",
                device_copy_sec,
            )
            self._add_batch_timing(
                requests,
                "opponent_infer_action_stack_to_device_sec",
                cpu_pack_sec + device_copy_sec,
            )

            self._set_active_phase("hidden_batch")
            hidden_started = time.perf_counter()
            hidden_batch = self._build_hidden_batch(agent, requests)
            self._add_batch_timing(
                requests,
                "opponent_infer_action_hidden_batch_sec",
                time.perf_counter() - hidden_started,
            )

            self._set_active_phase("forward")
            forward_started = time.perf_counter()
            with torch.inference_mode(), self._device_execution_context():
                out = agent(
                    obs_batch,
                    hidden_batch,
                    action_masks=masks_batch,
                    include_value=False,
                    include_policy_stats=False,
                )
            self._add_batch_timing(
                requests,
                "opponent_infer_action_forward_launch_sec",
                time.perf_counter() - forward_started,
            )

            self._set_active_phase("output_sync")
            output_started = time.perf_counter()
            self._synchronize_device_execution()
            actions = out["actions"].cpu().numpy()
            new_hidden = out["hidden"]
            hidden_views = [
                self._slice_hidden(new_hidden, index) for index in range(batch_size)
            ]
            self._add_batch_timing(
                requests,
                "opponent_infer_action_output_sync_sec",
                time.perf_counter() - output_started,
            )

        self._set_active_phase("set_result")
        result_started = time.perf_counter()
        for index, request in enumerate(requests):
            self._try_set_future_result(
                request.future,
                (actions[index], hidden_views[index]),
            )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_set_result_sec",
            time.perf_counter() - result_started,
        )

    def _run_preview_batch(
        self, checkpoint_path: str, requests: list[_InferenceRequest]
    ) -> None:
        self._set_active_phase("get_agent")
        agent = self._get_agent(checkpoint_path)
        batch_started = time.perf_counter()
        batch_size = len(requests)
        for request in requests:
            self._increment_timing(request.timing, "opponent_infer_preview_requests")
            self._add_timing(
                request.timing, "opponent_infer_preview_batches", 1.0 / batch_size
            )
            self._add_timing(
                request.timing,
                "opponent_infer_preview_batch_size_sum",
                float(batch_size),
            )
            self._increment_timing(
                request.timing, "opponent_infer_preview_queue_wait_calls"
            )
            self._add_timing(
                request.timing,
                "opponent_infer_preview_queue_wait_sec",
                max(0.0, batch_started - request.submitted_at),
            )

        self._set_active_phase("cpu_pack")
        cpu_pack_started = time.perf_counter()
        obs_host, obs_batch_size = self._pack_cpu_batch(
            "preview_obs",
            [request.obs for request in requests],
        )
        if obs_batch_size != batch_size:
            raise RuntimeError("Preview inference staging produced a mismatched batch size.")
        cpu_pack_sec = time.perf_counter() - cpu_pack_started
        self._add_batch_timing(
            requests,
            "opponent_infer_preview_cpu_pack_sec",
            cpu_pack_sec,
        )

        self._set_active_phase("cuda_lock_wait")
        with cuda_model_execution(self._device):
            self._set_active_phase("device_copy")
            device_copy_started = time.perf_counter()
            obs_batch = self._copy_staged_batch_to_device(
                "preview_obs",
                obs_host,
                batch_size,
            )
            device_copy_sec = time.perf_counter() - device_copy_started
            self._add_batch_timing(
                requests,
                "opponent_infer_preview_device_copy_sec",
                device_copy_sec,
            )
            self._add_batch_timing(
                requests,
                "opponent_infer_preview_stack_to_device_sec",
                cpu_pack_sec + device_copy_sec,
            )

            self._set_active_phase("forward")
            forward_started = time.perf_counter()
            with torch.inference_mode(), self._device_execution_context():
                picks_tensor = agent.team_preview(
                    obs_batch,
                    include_value=False,
                    include_policy_stats=False,
                    include_opponent_preview=False,
                )["picks"]
            self._add_batch_timing(
                requests,
                "opponent_infer_preview_forward_sync_sec",
                time.perf_counter() - forward_started,
            )

            self._set_active_phase("output_sync")
            output_started = time.perf_counter()
            self._synchronize_device_execution()
            picks = picks_tensor.cpu().numpy()
            self._add_batch_timing(
                requests,
                "opponent_infer_preview_output_sync_sec",
                time.perf_counter() - output_started,
            )

        self._set_active_phase("set_result")
        result_started = time.perf_counter()
        for index, request in enumerate(requests):
            self._try_set_future_result(request.future, picks[index])
        self._add_batch_timing(
            requests,
            "opponent_infer_preview_set_result_sec",
            time.perf_counter() - result_started,
        )

    def _run_load_batch(
        self, checkpoint_path: str, requests: list[_InferenceRequest]
    ) -> None:
        self._set_active_phase("load_agent")
        self._get_agent(checkpoint_path)
        for request in requests:
            self._try_set_future_result(request.future, None)
        with self._condition:
            self._prefetch_paths.discard(checkpoint_path)

    def _run_prune_batch(self, requests: list[_InferenceRequest]) -> None:
        keep_paths = requests[-1].keep_paths
        if keep_paths is None:
            raise RuntimeError("League inference prune request was missing keep paths.")

        # The worker owns cached-model retirement so no concurrent inference
        # frame can hold the final CUDA reference after the execution guard is
        # released. Multiple queued prune requests coalesce to the newest set.
        with cuda_model_execution(self._device):
            with self._condition:
                stale_models = [
                    self._models.pop(path)
                    for path in list(self._models)
                    if path not in keep_paths
                ]
                self._prefetch_paths.intersection_update(keep_paths)
            stale_models.clear()

        for request in requests:
            self._try_set_future_result(request.future, None)

    def _get_agent(self, checkpoint_path: str) -> PokeAgent:
        with self._condition:
            agent = self._models.get(checkpoint_path)
        if agent is None:
            loaded_agent = self._load_agent(checkpoint_path)
            with self._condition:
                agent = self._models.setdefault(checkpoint_path, loaded_agent)
                self._prefetch_paths.discard(checkpoint_path)
        return agent

    def _load_agent(self, checkpoint_path: str) -> PokeAgent:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent_copy = copy.deepcopy(self._agent_template)
        load_state_dict_compatible(agent_copy, checkpoint["agent"])
        with cuda_model_execution(self._device):
            agent_copy.to(self._device)
        agent_copy.eval()
        agent_copy.requires_grad_(False)
        return agent_copy

    def _build_hidden_batch(
        self,
        agent: PokeAgent,
        requests: list[_InferenceRequest],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if all(request.hidden is not None for request in requests):
            return (
                torch.cat(
                    [
                        self._to_inference_device(request.hidden[0])
                        for request in requests
                    ],
                    dim=1,
                ),
                torch.cat(
                    [
                        self._to_inference_device(request.hidden[1])
                        for request in requests
                    ],
                    dim=1,
                ),
            )

        hidden_batch = agent.initial_hidden(len(requests))
        if hidden_batch is None:
            return None

        h_batch, c_batch = hidden_batch
        for index, request in enumerate(requests):
            if request.hidden is None:
                continue
            h_req, c_req = request.hidden
            h_batch[:, index : index + 1].copy_(self._to_inference_device(h_req))
            c_batch[:, index : index + 1].copy_(self._to_inference_device(c_req))
        return h_batch, c_batch

    def _to_inference_device(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.device.type == self._device.type and (
            self._device.index is None or tensor.device.index == self._device.index
        ):
            return tensor
        return tensor.to(self._device, non_blocking=True)

    def _add_batch_timing(
        self,
        requests: list[_InferenceRequest],
        key: str,
        value: float,
    ) -> None:
        per_request = value / len(requests)
        for request in requests:
            self._add_timing(request.timing, key, per_request)

    @staticmethod
    def _add_timing(timing, key: str, value: float) -> None:
        if timing is not None:
            timing.add(key, value)

    @staticmethod
    def _increment_timing(timing, key: str, delta: int = 1) -> None:
        if timing is not None:
            timing.increment(key, delta)

    @staticmethod
    def _slice_hidden(
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return a detached, read-only-by-convention view for one request.

        The inference stream is synchronized before these views are created,
        and the service never mutates a published hidden batch. Sharing the
        backing batch avoids two CUDA clones per request. Each live
        ``LeaguePlayer`` retains at most its current and pending hidden state,
        so retained backing batches are bounded by the number of live players
        rather than by the number of completed decisions.
        """
        if hidden is None:
            return None
        h, c = hidden
        return (
            h[:, index : index + 1].detach(),
            c[:, index : index + 1].detach(),
        )


class LeaguePlayer(Player):
    """A poke-env player that uses a frozen agent checkpoint to choose moves.

    This is the opponent in self-play: it loads a past version of the agent
    and plays battles autonomously (no queue bridging needed).
    """

    def __init__(
        self,
        checkpoint_path: str,
        inference_service: _LeagueInferenceService,
        obs_builder: ObservationBuilder,
        preview_inference_service: _LeagueInferenceService | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.inference_service = inference_service
        self.preview_inference_service = preview_inference_service or inference_service
        self.obs_builder = obs_builder
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pending_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pending_decision_key: tuple[object, ...] | None = None
        self._inflight_decision_key: tuple[object, ...] | None = None
        self._pending_order = None
        self._pending_action_future: Future | None = None
        self._pending_action_battle: DoubleBattle | None = None
        self._pending_action_started_at: float | None = None
        self._pending_checkpoint_path: str | None = None
        self._prepared_preview_obs: dict[str, np.ndarray] | None = None
        self._prepared_preview_future: Future | None = None
        self._prepared_preview_started_at: float | None = None
        self._timing = None
        # Most-recent (a1, a2, tera) chosen by this opponent. Read by the env
        # after each turn resolves so the aux opponent-action head has labels.
        self._last_actions: np.ndarray | None = None

    def stage_checkpoint(self, checkpoint_path: str) -> None:
        self._pending_checkpoint_path = checkpoint_path
        self._prepared_preview_obs = None
        self._prepared_preview_future = None

    def apply_staged_checkpoint(self) -> bool:
        if self._pending_checkpoint_path is None:
            return False
        self.checkpoint_path = self._pending_checkpoint_path
        self._pending_checkpoint_path = None
        return True

    def set_timing(self, timing) -> None:
        self._timing = timing

    def prepare_team_preview(
        self, own_packed_team: str, opponent_packed_team: str
    ) -> None:
        timing_enabled = LeaguePlayer._timing_enabled(self)
        if timing_enabled:
            started = time.perf_counter()
        obs = build_team_preview_observation(
            self.obs_builder.gd,
            self.obs_builder.team_size,
            own_packed_team,
            opponent_packed_team,
        )
        if timing_enabled:
            self._add_timing(
                "opponent_precompute_preview_obs_sec", time.perf_counter() - started
            )

        if timing_enabled:
            submit_started = time.perf_counter()
        self._prepared_preview_obs = obs
        checkpoint_path = self._pending_checkpoint_path or self.checkpoint_path
        self._prepared_preview_future = self._preview_service().submit_team_preview(
            checkpoint_path,
            obs,
            timing=self._timing if timing_enabled else None,
        )
        self._prepared_preview_started_at = time.perf_counter()
        if timing_enabled:
            self._add_timing(
                "opponent_precompute_preview_submit_sec",
                time.perf_counter() - submit_started,
            )
            self._increment_timing("opponent_precompute_preview_submit_calls")

    def choose_move(self, battle: DoubleBattle):
        return self._choose_move(battle)

    async def _choose_move(self, battle: DoubleBattle):
        timing_enabled = LeaguePlayer._timing_enabled(self)
        if timing_enabled:
            choice_started = time.perf_counter()
            key_started = time.perf_counter()
        decision_key = self._decision_key(battle)
        if timing_enabled:
            self._add_timing(
                "opponent_decision_key_sec", time.perf_counter() - key_started
            )
        if (
            self._pending_decision_key == decision_key
            and self._pending_order is not None
        ):
            if timing_enabled:
                self._increment_timing("opponent_reused_pending_order")
                self._add_timing(
                    "opponent_choose_move_internal_sec",
                    time.perf_counter() - choice_started,
                )
                self._increment_timing("opponent_choose_move_internal_calls")
            return self._pending_order

        action_state = None
        action_future = self._pending_action_future
        if self._inflight_decision_key == decision_key and action_future is not None:
            if timing_enabled:
                self._increment_timing("opponent_reused_inflight_action")
                infer_started = time.perf_counter()
        else:
            if timing_enabled:
                hidden_started = time.perf_counter()
            self._commit_hidden_if_battle_advanced(decision_key)
            if timing_enabled:
                self._add_timing(
                    "opponent_hidden_commit_sec", time.perf_counter() - hidden_started
                )
                masks_started = time.perf_counter()
            action_state = build_legal_action_state(battle)
            if is_default_only_action_state(action_state):
                result = self.choose_default_move()
                self._pending_decision_key = decision_key
                self._pending_hidden = self._hidden
                self._pending_order = result
                self._last_actions = np.asarray(
                    [PASS_ACTION, PASS_ACTION, TERA_NONE],
                    dtype=np.int64,
                )
                if timing_enabled:
                    self._increment_timing("opponent_default_only_requests")
                    self._add_timing(
                        "opponent_mask_build_sec", time.perf_counter() - masks_started
                    )
                    self._add_timing(
                        "opponent_choose_move_internal_sec",
                        time.perf_counter() - choice_started,
                    )
                    self._increment_timing("opponent_choose_move_internal_calls")
                return result
            masks = build_policy_action_masks(battle, action_state=action_state)
            if timing_enabled:
                self._add_timing(
                    "opponent_mask_build_sec", time.perf_counter() - masks_started
                )
                obs_started = time.perf_counter()
            obs = self.obs_builder.build(battle)
            if timing_enabled:
                self._add_timing(
                    "opponent_obs_build_sec", time.perf_counter() - obs_started
                )
                infer_started = time.perf_counter()
            action_future = self.inference_service.submit_action(
                self.checkpoint_path,
                obs,
                masks,
                self._hidden,
                timing=self._timing if timing_enabled else None,
            )
            self._pending_action_future = action_future
            self._pending_action_battle = battle
            self._pending_action_started_at = time.perf_counter()
            self._inflight_decision_key = decision_key

        try:
            actions, next_hidden = await asyncio.wrap_future(action_future)
        except asyncio.CancelledError:
            if battle.finished:
                return _NoopBattleOrder()
            raise
        finally:
            if self._pending_action_future is action_future and action_future.done():
                self._pending_action_future = None
                self._pending_action_battle = None
                self._pending_action_started_at = None
                if self._inflight_decision_key == decision_key:
                    self._inflight_decision_key = None
        if battle.finished:
            return _NoopBattleOrder()
        if (
            self._pending_decision_key == decision_key
            and self._pending_order is not None
        ):
            return self._pending_order
        if action_state is None:
            action_state = build_legal_action_state(battle)
        if timing_enabled:
            self._add_timing(
                "opponent_infer_action_wait_sec", time.perf_counter() - infer_started
            )
        self._pending_decision_key = decision_key
        self._pending_hidden = next_hidden
        self._last_actions = np.asarray(actions, dtype=np.int64).copy()

        if timing_enabled:
            order_started = time.perf_counter()
        orders = []
        tera_slot_1, tera_slot_2 = decode_tera_choice(int(actions[2]))
        for slot in range(2):
            orders.append(
                action_to_order(
                    int(actions[slot]),
                    battle,
                    slot,
                    terastallize=tera_slot_1 if slot == 0 else tera_slot_2,
                    action_state=action_state,
                )
            )
        if all(isinstance(order, PassBattleOrder) for order in orders):
            result = self.choose_default_move()
        else:
            result = DoubleBattleOrder(orders[0], orders[1])
        self._pending_order = result
        if timing_enabled:
            self._add_timing(
                "opponent_order_build_sec", time.perf_counter() - order_started
            )
            self._add_timing(
                "opponent_choose_move_internal_sec",
                time.perf_counter() - choice_started,
            )
            self._increment_timing("opponent_choose_move_internal_calls")
        return result

    def _battle_finished_callback(self, battle: DoubleBattle) -> None:
        pending = self._pending_action_future
        if pending is None or pending.done():
            return
        if self._pending_action_battle is battle:
            pending.cancel()

    def teampreview(self, battle: DoubleBattle):
        return self._teampreview(battle)

    async def _teampreview(self, battle: DoubleBattle) -> str:
        obs = self.obs_builder.build(battle)
        prepared_obs = self._prepared_preview_obs
        prepared_future = self._prepared_preview_future
        self._prepared_preview_obs = None
        self._prepared_preview_future = None
        self._prepared_preview_started_at = None
        timing_enabled = LeaguePlayer._timing_enabled(self)

        if prepared_obs is None and prepared_future is None:
            if timing_enabled:
                infer_started = time.perf_counter()
            preview_future = self._preview_service().submit_team_preview(
                self.checkpoint_path,
                obs,
                timing=self._timing if timing_enabled else None,
            )
            picks = await asyncio.wrap_future(preview_future)
            if timing_enabled:
                self._add_timing(
                    "opponent_teampreview_infer_sec",
                    time.perf_counter() - infer_started,
                )
            return "/team " + "".join(str(int(p) + 1) for p in picks)

        if prepared_obs is None or prepared_future is None:
            self._increment_timing("opponent_precompute_preview_missing_state")
            raise RuntimeError(
                "Prepared opponent team preview state was partially missing."
            )

        if self._obs_equal(obs, prepared_obs):
            if timing_enabled:
                self._increment_timing("opponent_precompute_preview_hits")
                wait_started = time.perf_counter()
            picks = await asyncio.wrap_future(prepared_future)
            if timing_enabled:
                self._add_timing(
                    "opponent_precompute_preview_hit_wait_sec",
                    time.perf_counter() - wait_started,
                )
            return "/team " + "".join(str(int(p) + 1) for p in picks)

        if timing_enabled:
            self._increment_timing("opponent_precompute_preview_misses")
        self._record_preview_obs_mismatch(obs, prepared_obs)
        raise RuntimeError(
            "Prepared opponent team preview observation did not match the live battle state."
        )

    def reset_hidden(self):
        """Clear LSTM state between battles. Called by BattleEnv.reset()."""
        self._hidden = None
        self._pending_hidden = None
        self._pending_decision_key = None
        self._inflight_decision_key = None
        self._pending_order = None
        self._pending_action_future = None
        self._pending_action_battle = None
        self._pending_action_started_at = None
        self._last_actions = None

    def describe_pending_inference(self) -> str:
        now = time.perf_counter()
        pending = self._pending_action_future
        preview = self._prepared_preview_future
        pending_age = (
            None
            if pending is None or self._pending_action_started_at is None
            else max(0.0, now - self._pending_action_started_at)
        )
        preview_age = (
            None
            if preview is None or self._prepared_preview_started_at is None
            else max(0.0, now - self._prepared_preview_started_at)
        )
        battle = self._pending_action_battle
        service_describe = getattr(self.inference_service, "describe", None)
        if callable(service_describe):
            try:
                service_description = service_describe()
            except Exception as exc:
                service_description = f"describe_failed={exc!r}"
        else:
            service_description = "describe_unavailable"
        preview_service = self._preview_service()
        if preview_service is self.inference_service:
            preview_service_description = "same_as_action_service"
        else:
            preview_service_describe = getattr(preview_service, "describe", None)
            if callable(preview_service_describe):
                try:
                    preview_service_description = preview_service_describe()
                except Exception as exc:
                    preview_service_description = f"describe_failed={exc!r}"
            else:
                preview_service_description = "describe_unavailable"
        return (
            f"checkpoint={Path(self.checkpoint_path).name} "
            f"pending_action={pending is not None} "
            f"pending_done={None if pending is None else pending.done()} "
            f"pending_cancelled={None if pending is None else pending.cancelled()} "
            f"inflight_decision_key={self._inflight_decision_key is not None} "
            f"pending_age_sec={pending_age} "
            f"pending_battle={getattr(battle, 'battle_tag', None)} "
            f"pending_turn={getattr(battle, 'turn', None)} "
            f"prepared_preview={preview is not None} "
            f"prepared_preview_done={None if preview is None else preview.done()} "
            f"prepared_preview_age_sec={preview_age} "
            f"service=({service_description}) "
            f"preview_service=({preview_service_description})"
        )

    def _preview_service(self) -> _LeagueInferenceService:
        return getattr(self, "preview_inference_service", self.inference_service)

    @staticmethod
    def _obs_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
        if left.keys() != right.keys():
            return False
        return all(np.array_equal(left[key], right[key]) for key in left)

    def _commit_hidden_if_battle_advanced(
        self, decision_key: tuple[object, ...]
    ) -> None:
        if self._pending_decision_key is None:
            return
        if decision_key == self._pending_decision_key:
            return
        self._hidden = self._pending_hidden
        self._pending_hidden = None
        self._pending_decision_key = None
        self._pending_order = None

    @classmethod
    def _decision_key(cls, battle: DoubleBattle) -> tuple[object, ...]:
        return (
            getattr(battle, "battle_tag", ""),
            int(getattr(battle, "turn", 0)),
            cls._request_decision_signature(getattr(battle, "last_request", None)),
            cls._simple_decision_tuple(getattr(battle, "force_switch", None)),
            cls._simple_decision_tuple(getattr(battle, "can_tera", None)),
            cls._simple_decision_tuple(getattr(battle, "trapped", None)),
        )

    @classmethod
    def _request_decision_signature(cls, request) -> tuple[object, ...]:
        """Compact key for the request fields that determine a battle decision.

        poke-env assigns ``battle.last_request`` from the parsed Showdown request.
        The previous implementation hashed the entire request JSON on every
        opponent choice. For hidden-state commits we only need to distinguish
        distinct decision requests, so keep the action-relevant fields instead
        of serializing side/team payloads every time.
        """
        if not isinstance(request, dict):
            return ()

        active_requests = request.get("active") or ()
        active_signature = []
        for active_request in active_requests:
            if not isinstance(active_request, dict):
                active_signature.append(repr(active_request))
                continue
            moves = active_request.get("moves") or ()
            move_signature = []
            for move_request in moves:
                if not isinstance(move_request, dict):
                    move_signature.append(repr(move_request))
                    continue
                move_signature.append(
                    (
                        move_request.get("id"),
                        move_request.get("target"),
                        bool(move_request.get("disabled", False)),
                    )
                )
            active_signature.append(
                (
                    bool(active_request.get("trapped", False)),
                    bool(active_request.get("maybeTrapped", False)),
                    bool(active_request.get("canTerastallize", False)),
                    tuple(move_signature),
                )
            )

        return (
            bool(request.get("wait", False)),
            bool(request.get("teamPreview", False)),
            cls._simple_decision_tuple(request.get("forceSwitch")),
            tuple(active_signature),
            cls._side_switch_signature(request.get("side")),
        )

    @staticmethod
    def _simple_decision_tuple(value) -> tuple:
        if value is None:
            return ()
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    @classmethod
    def _side_switch_signature(cls, side) -> tuple[object, ...]:
        if not isinstance(side, dict):
            return ()
        pokemon = side.get("pokemon") or ()
        signature = []
        for pokemon_request in pokemon:
            if not isinstance(pokemon_request, dict):
                signature.append(repr(pokemon_request))
                continue
            condition = str(pokemon_request.get("condition", ""))
            signature.append(
                (
                    pokemon_request.get("ident"),
                    bool(pokemon_request.get("active", False)),
                    condition.endswith(" fnt"),
                )
            )
        return tuple(signature)

    def _record_preview_obs_mismatch(
        self,
        actual: dict[str, np.ndarray],
        prepared: dict[str, np.ndarray],
    ) -> None:
        if actual.keys() != prepared.keys():
            self._increment_timing("opponent_precompute_preview_keyset_mismatch")
            return
        for key in actual:
            if not np.array_equal(actual[key], prepared[key]):
                self._increment_timing(f"opponent_precompute_preview_mismatch_{key}")

    def _add_timing(self, key: str, value: float) -> None:
        timing = getattr(self, "_timing", None)
        if timing is not None:
            timing.add(key, value)

    def _increment_timing(self, key: str, delta: int = 1) -> None:
        timing = getattr(self, "_timing", None)
        if timing is not None:
            timing.increment(key, delta)

    def _timing_enabled(self) -> bool:
        timing = getattr(self, "_timing", None)
        return timing is not None and bool(getattr(timing, "enabled", True))


class League:
    """Manages a pool of opponent checkpoints for self-play training.

    Opponents are sampled from a rolling window of past checkpoints, with a
    bias towards recent ones (70%) over older ones (30%) to track current skill
    while retaining some diversity.
    """

    def __init__(
        self,
        config: TrainingConfig,
        env_config: EnvConfig,
        game_data: GameData,
        agent_template: PokeAgent,
        device: torch.device,
    ):
        self.config = config
        self.env_config = env_config
        self.game_data = game_data
        self.agent_template = agent_template
        self.device = device
        self.inference_device = device
        self.preview_inference_device = (
            torch.device("cpu") if device.type == "cuda" else self.inference_device
        )
        self.checkpoints: deque[str] = deque(maxlen=config.league_size)
        self._inference = _LeagueInferenceService(
            agent_template,
            self.inference_device,
            coalesce_window_sec=config.league_coalesce_ms / 1000.0,
        )
        if self.preview_inference_device == self.inference_device:
            self._preview_inference = self._inference
        else:
            self._preview_inference = _LeagueInferenceService(
                agent_template,
                self.preview_inference_device,
                coalesce_window_sec=config.league_coalesce_ms / 1000.0,
            )

    def add_checkpoint(self, agent: PokeAgent, step: int):
        """Snapshot the current agent as a new league member."""
        path = f"{self.config.checkpoint_dir}/league/step_{step}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint_type": "league", "agent": agent.state_dict()}, path)
        self.checkpoints.append(path)
        keep_paths = set(self.checkpoints)
        for service in self._inference_services():
            service.prune_checkpoints(keep_paths)
            service.prefetch_checkpoint(path)

    def make_opponent(
        self,
        loop: asyncio.AbstractEventLoop,
        checkpoint_path: str | None = None,
    ) -> "LeaguePlayer | None":
        """Create a LeaguePlayer from a sampled checkpoint.

        Args:
            loop: The asyncio event loop the opponent will run on.

        Returns None if the league is empty.
        """
        path = (
            checkpoint_path
            if checkpoint_path is not None
            else self.sample_checkpoint_path()
        )
        if path is None:
            return None
        for service in self._inference_services():
            service.prefetch_checkpoint(path)

        battlestream_config = make_battlestream_configuration(self.env_config)

        obs_builder = ObservationBuilder(self.game_data, self.env_config.team_size)
        return LeaguePlayer(
            path,
            self._inference,
            obs_builder,
            preview_inference_service=self._preview_inference,
            account_configuration=make_local_account_configuration("league"),
            battle_format=self.env_config.battle_format,
            server_configuration=battlestream_config,
            log_level=showdown_log_level(),
            loop=loop,
        )

    def sample_checkpoint_path(self) -> str | None:
        path = self._sample_checkpoint()
        if path is None:
            return None
        for service in self._inference_services():
            service.prefetch_checkpoint(path)
        return path

    def save_state(self, path: str) -> None:
        """Persist the ordered list of checkpoint paths to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(list(self.checkpoints), f)

    def load_state(self, path: str) -> None:
        """Restore checkpoint paths from a saved state file.

        Skips any paths that no longer exist on disk and warns if any are missing.
        """
        with open(path) as f:
            paths: list[str] = json.load(f)

        existing = [p for p in paths if Path(p).exists()]
        missing = len(paths) - len(existing)
        if missing:
            print(
                f"Warning: {missing} league checkpoint(s) no longer on disk, skipping."
            )

        for p in existing[-self.config.league_size :]:
            self.checkpoints.append(p)
        keep_paths = set(self.checkpoints)
        for service in self._inference_services():
            service.prune_checkpoints(keep_paths)

    def close(self) -> None:
        errors: list[Exception] = []
        for service in self._inference_services():
            try:
                service.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise ExceptionGroup(
                "One or more league inference services failed to close.", errors
            )

    def _inference_services(self) -> tuple[_LeagueInferenceService, ...]:
        preview_inference = getattr(self, "_preview_inference", self._inference)
        if preview_inference is self._inference:
            return (self._inference,)
        return (self._inference, preview_inference)

    def _sample_checkpoint(self) -> str | None:
        """Sample a checkpoint path. 70% recent half, 30% older half."""
        checkpoints = list(self.checkpoints)
        if not checkpoints:
            return None

        mid = len(checkpoints) // 2

        if len(checkpoints) < 2 or random.random() < 0.7:
            recent = checkpoints[mid:] if mid < len(checkpoints) else checkpoints
            return random.choice(recent)
        else:
            older = checkpoints[:mid] if mid > 0 else checkpoints
            return random.choice(older)
