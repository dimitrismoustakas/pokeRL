"""Self-play league: a pool of past agent checkpoints as opponents."""

import asyncio
import copy
import hashlib
import json
import math
import random
import time
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from concurrent.futures import CancelledError, Future, InvalidStateError
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Lock, Thread

import numpy as np
import torch
from poke_env.battle import DoubleBattle
from poke_env.player import DoubleBattleOrder, PassBattleOrder, Player

from ..model.agent import PokeAgent
from ..model.checkpoint import (
    LEAGUE_MODEL_CHECKPOINT_FORMAT_VERSION,
    model_checkpoint_metadata,
    validate_model_checkpoint_metadata,
)
from ..data import GameData
from ..config import TrainingConfig, EnvConfig
from ..env.observation import (
    ObservationBuilder,
    build_team_preview_observation,
    summarize_observation_mismatch,
)
from ..env.actions import (
    NUM_TERA_CHOICES,
    SemanticActionIntent,
    SemanticActionSupervision,
    action_to_order,
    build_legal_action_state,
    build_policy_action_masks,
    decode_tera_choice,
    is_default_only_action_state,
    semantic_action_request_key,
    semantic_action_supervision,
)
from ..env.showdown_env import (
    _NoopBattleOrder,
    make_battlestream_configuration,
    make_local_account_configuration,
    showdown_log_level,
)
from .cuda_execution import cuda_model_execution
from .ppo import TRAINER_CHECKPOINT_FORMAT_VERSION


_LEAGUE_STATE_FORMAT_VERSION = 4
_LEAGUE_STATE_CONFIG_FIELDS = (
    "league_size",
    "league_update_interval",
    "league_refresh_interval",
    "league_payoff_half_life_updates",
    "league_pfsp_exponent",
    "league_pfsp_uniform_fraction",
)
_LEAGUE_MEMBER_STATE_FIELDS = (
    "path",
    "checkpoint_sha256",
    "archive_source",
    "step",
    "created_update",
    "discounted_wins",
    "discounted_losses",
    "last_payoff_update",
    "lifetime_wins",
    "lifetime_losses",
    "lifetime_draws",
)
_LEAGUE_MEMBER_SOURCES = frozenset({"self_play_snapshot", "admitted_external"})
_SHA256_HEX_LENGTH = 64


@dataclass
class LeagueMember:
    """One permanent archive member and its live-policy payoff evidence."""

    path: str
    checkpoint_sha256: str
    archive_source: str
    step: int
    created_update: int
    discounted_wins: float = 0.0
    discounted_losses: float = 0.0
    last_payoff_update: int | None = None
    lifetime_wins: int = 0
    lifetime_losses: int = 0
    lifetime_draws: int = 0

    def payoff_evidence(
        self,
        update: int,
        half_life_updates: float,
    ) -> tuple[float, float]:
        if self.last_payoff_update is None:
            return 0.0, 0.0
        if update < self.last_payoff_update:
            raise ValueError(
                "League payoff update cannot precede recorded evidence: "
                f"update={update}, last={self.last_payoff_update}."
            )
        decay = 2.0 ** (-(update - self.last_payoff_update) / half_life_updates)
        return self.discounted_wins * decay, self.discounted_losses * decay

    def payoff_summary(
        self,
        update: int,
        half_life_updates: float,
    ) -> tuple[float, float]:
        wins, losses = self.payoff_evidence(update, half_life_updates)
        effective_games = wins + losses
        # A symmetric Beta(1, 1) prior makes unseen and fully stale members
        # neutral rather than spuriously easy or hard.
        win_probability = (1.0 + wins) / (2.0 + effective_games)
        return win_probability, effective_games

    def record_outcomes(
        self,
        *,
        update: int,
        wins: int,
        losses: int,
        draws: int,
        half_life_updates: float,
    ) -> None:
        games = wins + losses + draws
        if games <= 0:
            raise ValueError("League payoff update requires at least one outcome.")
        discounted_wins, discounted_losses = self.payoff_evidence(
            update,
            half_life_updates,
        )
        # All games in a bucket share one learner version and assignment
        # schedule, so they are clustered evidence rather than hundreds of
        # independent policy observations. Each nonempty update contributes
        # exactly one effective evidence unit.
        self.discounted_wins = discounted_wins + (wins + 0.5 * draws) / games
        self.discounted_losses = discounted_losses + (losses + 0.5 * draws) / games
        self.last_payoff_update = update
        self.lifetime_wins += wins
        self.lifetime_losses += losses
        self.lifetime_draws += draws


def _tuples_to_lists(value):
    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, dict):
        return {key: _tuples_to_lists(item) for key, item in value.items()}
    return value


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _checkpoint_file_sha256(
    path: str | Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> str:
    """Hash exact checkpoint bytes without retaining them in memory."""
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"League checkpoint does not exist: {source!s}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _lists_to_tuples(value):
    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    if isinstance(value, dict):
        return {key: _lists_to_tuples(item) for key, item in value.items()}
    return value


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


@dataclass
class _CudaGraphExecutor:
    """One fixed-shape CUDA graph tied to a frozen checkpoint's parameters."""

    checkpoint_path: str
    obs_schema: _BatchSchema
    masks_schema: _BatchSchema
    obs_host: _HostBatchBuffers
    masks_host: _HostBatchBuffers
    obs_device: _DeviceBatchBuffers | None
    masks_device: _DeviceBatchBuffers | None
    hidden: tuple[torch.Tensor, torch.Tensor] | None
    sampling_exponentials: (
        tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None
    )
    graph: torch.cuda.CUDAGraph | None
    output: dict[str, object]


def _same_device(left: torch.device, right: torch.device) -> bool:
    return left.type == right.type and (
        right.index is None or left.index == right.index
    )


_ACTION_BATCH_SIZE_BUCKETS = (
    (1, "1"),
    (2, "2"),
    (4, "3-4"),
    (8, "5-8"),
    (12, "9-12"),
    (16, "13-16"),
    (24, "17-24"),
    (36, "25-36"),
)


def _action_batch_size_bucket(batch_size: int) -> str:
    for upper_bound, label in _ACTION_BATCH_SIZE_BUCKETS:
        if batch_size <= upper_bound:
            return label
    return ">36"


def _empty_action_batch_size_histogram() -> dict[str, int]:
    return {label: 0 for _, label in _ACTION_BATCH_SIZE_BUCKETS} | {">36": 0}


class _LeagueInferenceService:
    """Shared batched inference for league opponents.

    A single worker thread batches requests per checkpoint and runs them on the
    configured device. This preserves the opponent policy while avoiding one
    tiny GPU forward per environment thread.
    """

    _CLOSE_TIMEOUT_SEC = 60.0
    _DEFAULT_CUDA_GRAPH_BATCH_SIZE = 16
    _DEFAULT_CUDA_GRAPH_CACHE_SIZE = 4
    _CUDA_GRAPH_WARMUP_CALLS = 3

    def __init__(
        self,
        agent_template: PokeAgent,
        device: torch.device,
        coalesce_window_sec: float,
        *,
        max_models: int,
        greedy: bool = False,
        enable_cuda_graphs: bool = False,
        cuda_graph_batch_size: int = _DEFAULT_CUDA_GRAPH_BATCH_SIZE,
        cuda_graph_cache_size: int = _DEFAULT_CUDA_GRAPH_CACHE_SIZE,
    ):
        if max_models <= 0:
            raise ValueError(f"max_models must be positive (got {max_models}).")
        for name, value in (
            ("cuda_graph_batch_size", cuda_graph_batch_size),
            ("cuda_graph_cache_size", cuda_graph_cache_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer (got {value}).")
        self._agent_template = copy.deepcopy(agent_template).to(torch.device("cpu"))
        self._agent_template.eval()
        self._agent_template.requires_grad_(False)
        self._device = device
        self._cuda_stream: torch.cuda.Stream | None = None
        self._coalesce_window_sec = max(0.0, coalesce_window_sec)
        self._greedy = greedy
        self._max_models = max_models
        self._models: OrderedDict[str, PokeAgent] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_loads = 0
        self._cache_evictions = 0
        self._host_batch_buffers: dict[str, _HostBatchBuffers] = {}
        self._device_batch_buffers: dict[str, _DeviceBatchBuffers] = {}
        self._cuda_graphs_enabled = bool(
            enable_cuda_graphs
            and device.type == "cuda"
            and hasattr(torch.cuda, "CUDAGraph")
        )
        self._cuda_graph_batch_size = cuda_graph_batch_size
        self._cuda_graph_cache_size = cuda_graph_cache_size
        self._cuda_graphs: OrderedDict[str, _CudaGraphExecutor] = OrderedDict()
        self._cuda_graph_hits = 0
        self._cuda_graph_misses = 0
        self._cuda_graph_captures = 0
        self._cuda_graph_replays = 0
        self._cuda_graph_evictions = 0
        self._cuda_graph_fallbacks = 0
        self._cuda_graph_real_rows = 0
        self._cuda_graph_padded_rows = 0
        self._cuda_graph_last_error: str | None = None
        self._action_batch_size_histogram = _empty_action_batch_size_histogram()
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
        self._get_agent(checkpoint_path)

    def prefetch_checkpoint(self, checkpoint_path: str) -> None:
        future: Future = Future()
        with self._condition:
            self._raise_if_unavailable_locked()
            if (
                checkpoint_path in self._models
                or checkpoint_path in self._prefetch_paths
            ):
                if checkpoint_path in self._models:
                    self._touch_model_locked(checkpoint_path)
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

    def cache_info(self) -> dict[str, object]:
        """Return a consistent snapshot of bounded model-cache diagnostics."""
        with self._condition:
            return {
                "capacity": self._max_models,
                "resident_models": len(self._models),
                "hits": self._cache_hits,
                "misses": self._cache_misses,
                "loads": self._cache_loads,
                "evictions": self._cache_evictions,
                "action_batch_size_histogram": dict(
                    getattr(
                        self,
                        "_action_batch_size_histogram",
                        _empty_action_batch_size_histogram(),
                    )
                ),
                "cuda_graphs_enabled": getattr(self, "_cuda_graphs_enabled", False),
                "cuda_graph_batch_size": self._cuda_graph_batch_size,
                "cuda_graph_capacity": self._cuda_graph_cache_size,
                "resident_cuda_graphs": len(getattr(self, "_cuda_graphs", {})),
                "cuda_graph_hits": getattr(self, "_cuda_graph_hits", 0),
                "cuda_graph_misses": getattr(self, "_cuda_graph_misses", 0),
                "cuda_graph_captures": getattr(self, "_cuda_graph_captures", 0),
                "cuda_graph_replays": getattr(self, "_cuda_graph_replays", 0),
                "cuda_graph_evictions": getattr(self, "_cuda_graph_evictions", 0),
                "cuda_graph_fallbacks": getattr(self, "_cuda_graph_fallbacks", 0),
                "cuda_graph_real_rows": getattr(self, "_cuda_graph_real_rows", 0),
                "cuda_graph_padded_rows": getattr(self, "_cuda_graph_padded_rows", 0),
                "cuda_graph_last_error": getattr(self, "_cuda_graph_last_error", None),
            }

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
            self._clear_cuda_graphs()
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
                            raise ValueError(f"Unknown inference request kind: {kind}")
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
            cache_capacity = self._max_models
            cache_hits = self._cache_hits
            cache_misses = self._cache_misses
            cache_loads = self._cache_loads
            cache_evictions = self._cache_evictions

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
            f"model_count={model_count}/{cache_capacity} cache_hits={cache_hits} "
            f"cache_misses={cache_misses} cache_loads={cache_loads} "
            f"cache_evictions={cache_evictions} "
            f"last_progress_age_sec={last_progress_age}"
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
                if (
                    tuple(value.shape) != expected_shape
                    or value.dtype.str != expected_dtype
                ):
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

    def _allocate_host_batch_buffers(
        self,
        schema: _BatchSchema,
        capacity: int,
        *,
        pin_memory: bool = False,
    ) -> _HostBatchBuffers:
        group_sizes: dict[str, int] = {}
        for _, shape, dtype in schema:
            row_size = int(np.prod(shape, dtype=np.int64))
            group_sizes[dtype] = group_sizes.get(dtype, 0) + capacity * row_size
        if pin_memory:
            group_tensors = {
                dtype: torch.empty(
                    size,
                    dtype=torch.from_numpy(np.empty((), dtype=np.dtype(dtype))).dtype,
                    pin_memory=True,
                )
                for dtype, size in group_sizes.items()
            }
            group_arrays = {
                dtype: value.numpy() for dtype, value in group_tensors.items()
            }
        else:
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
            key: buffers.tensors[key][:batch_size] for key, _, _ in host_buffers.schema
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
            tensors[key] = group_tensors[group_key][offset : offset + field_size].view(
                host_buffers.capacity, *shape
            )
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

    def _run_cuda_graph_battle_batch(
        self,
        checkpoint_path: str,
        agent: PokeAgent,
        requests: list[_InferenceRequest],
        request_masks: list[dict[str, np.ndarray]],
    ) -> bool:
        """Run one logical request batch through bounded fixed-shape graphs."""
        chunk_size = self._cuda_graph_batch_size
        chunk_results: list[
            tuple[np.ndarray, list[tuple[torch.Tensor, torch.Tensor] | None]]
        ] = []
        cpu_pack_sec = 0.0
        replay_sec = 0.0
        capture_sec = 0.0
        replay_count = 0
        lock_wait_started = time.perf_counter()
        saved_rng: torch.Tensor | None = None
        try:
            self._set_active_phase("cuda_graph_lock_wait")
            with cuda_model_execution(self._device):
                lock_wait_sec = time.perf_counter() - lock_wait_started
                saved_rng = torch.cuda.get_rng_state(self._device).clone()
                sampling_exponentials = self._make_sampling_exponentials(
                    agent,
                    len(requests),
                )
                for start in range(0, len(requests), chunk_size):
                    stop = min(start + chunk_size, len(requests))
                    chunk = requests[start:stop]
                    chunk_masks = request_masks[start:stop]
                    pack_started = time.perf_counter()
                    self._set_active_phase("cuda_graph_cpu_pack")
                    executor = self._prepare_cuda_graph_executor(
                        checkpoint_path,
                        [request.obs for request in chunk],
                        chunk_masks,
                    )
                    cpu_pack_sec += time.perf_counter() - pack_started

                    if executor.graph is None:
                        capture_started = time.perf_counter()
                        self._set_active_phase("cuda_graph_capture")
                        self._capture_cuda_graph(executor, agent, chunk)
                        capture_sec += time.perf_counter() - capture_started
                    replay_started = time.perf_counter()
                    self._set_active_phase("cuda_graph_replay")
                    actions, hidden_views = self._replay_cuda_graph(
                        executor,
                        chunk,
                        sampling_exponentials=(
                            None
                            if sampling_exponentials is None
                            else tuple(
                                values[start:stop] for values in sampling_exponentials
                            )
                        ),
                    )
                    replay_sec += time.perf_counter() - replay_started
                    replay_count += 1
                    chunk_results.append((actions, hidden_views))
        except Exception as exc:
            # Capture is an optional execution optimization. Restore the exact
            # pre-attempt RNG state so eager fallback samples the same policy it
            # would have sampled if graphs had never been attempted.
            with cuda_model_execution(self._device):
                if saved_rng is not None:
                    torch.cuda.set_rng_state(saved_rng, self._device)
                self._cuda_graph_fallbacks += 1
                self._cuda_graph_last_error = f"{type(exc).__name__}: {exc}"
                self._cuda_graphs_enabled = False
                self._clear_cuda_graphs()
            return False

        self._cuda_graph_replays += replay_count
        self._cuda_graph_real_rows += len(requests)
        self._cuda_graph_padded_rows += replay_count * chunk_size - len(requests)

        self._add_batch_timing(
            requests,
            "opponent_infer_action_cpu_pack_sec",
            cpu_pack_sec,
        )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_cuda_graph_replay_sec",
            replay_sec,
        )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_cuda_graph_capture_sec",
            capture_sec,
        )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_cuda_graph_lock_wait_sec",
            lock_wait_sec,
        )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_cuda_graph_replays",
            float(len(chunk_results)),
        )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_cuda_graph_real_rows",
            float(len(requests)),
        )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_cuda_graph_padded_rows",
            float(len(chunk_results) * chunk_size - len(requests)),
        )

        result_index = 0
        for actions, hidden_views in chunk_results:
            for chunk_index in range(len(actions)):
                request = requests[result_index]
                self._try_set_future_result(
                    request.future,
                    (actions[chunk_index], hidden_views[chunk_index]),
                )
                result_index += 1
        return True

    def _make_sampling_exponentials(
        self,
        agent: PokeAgent,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if getattr(self, "_greedy", False):
            return None
        if self._cuda_stream is None:
            self._cuda_stream = torch.cuda.Stream(device=self._device)
        num_actions = agent.action_head.num_actions
        with torch.cuda.stream(self._cuda_stream):
            return (
                torch.empty(
                    (batch_size, num_actions),
                    device=self._device,
                ).exponential_(),
                torch.empty(
                    (batch_size, num_actions),
                    device=self._device,
                ).exponential_(),
                torch.empty(
                    (batch_size, NUM_TERA_CHOICES),
                    device=self._device,
                ).exponential_(),
            )

    def _prepare_cuda_graph_executor(
        self,
        checkpoint_path: str,
        observations: list[dict[str, np.ndarray]],
        masks: list[dict[str, np.ndarray]],
    ) -> _CudaGraphExecutor:
        obs_schema = self._batch_schema(observations, cache_key="battle_obs")
        masks_schema = self._batch_schema(masks, cache_key="battle_masks")
        graphs = self._cuda_graphs
        executor = graphs.get(checkpoint_path)
        if executor is not None and (
            executor.obs_schema != obs_schema or executor.masks_schema != masks_schema
        ):
            self._evict_cuda_graph(checkpoint_path)
            executor = None

        if executor is None:
            self._cuda_graph_misses += 1
            while len(graphs) >= self._cuda_graph_cache_size:
                stale_path = next(iter(graphs))
                self._evict_cuda_graph(stale_path)
            obs_host = self._allocate_host_batch_buffers(
                obs_schema,
                self._cuda_graph_batch_size,
                pin_memory=True,
            )
            masks_host = self._allocate_host_batch_buffers(
                masks_schema,
                self._cuda_graph_batch_size,
                pin_memory=True,
            )
            executor = _CudaGraphExecutor(
                checkpoint_path=checkpoint_path,
                obs_schema=obs_schema,
                masks_schema=masks_schema,
                obs_host=obs_host,
                masks_host=masks_host,
                obs_device=None,
                masks_device=None,
                hidden=None,
                sampling_exponentials=None,
                graph=None,
                output={},
            )
            graphs[checkpoint_path] = executor
        else:
            self._cuda_graph_hits += 1
            graphs.move_to_end(checkpoint_path)

        self._pack_padded_cuda_graph_input(
            executor.obs_host,
            observations,
        )
        self._pack_padded_cuda_graph_input(
            executor.masks_host,
            masks,
        )
        return executor

    @staticmethod
    def _pack_padded_cuda_graph_input(
        buffers: _HostBatchBuffers,
        mappings: list[dict[str, np.ndarray]],
    ) -> None:
        batch_size = len(mappings)
        if not 0 < batch_size <= buffers.capacity:
            raise ValueError(
                "CUDA graph input batch must be within its fixed capacity."
            )
        for key, _, _ in buffers.schema:
            np.stack(
                [mapping[key] for mapping in mappings],
                axis=0,
                out=buffers.arrays[key][:batch_size],
                casting="no",
            )
            if batch_size < buffers.capacity:
                buffers.arrays[key][batch_size:] = buffers.arrays[key][batch_size - 1]

    def _capture_cuda_graph(
        self,
        executor: _CudaGraphExecutor,
        agent: PokeAgent,
        requests: list[_InferenceRequest],
    ) -> None:
        if self._cuda_stream is None:
            self._cuda_stream = torch.cuda.Stream(device=self._device)
        obs_buffers = self._allocate_device_batch_buffers(
            executor.obs_host,
            force_bool=False,
        )
        masks_buffers = self._allocate_device_batch_buffers(
            executor.masks_host,
            force_bool=True,
        )
        if (
            len(executor.masks_host.group_tensors) != 1
            or next(iter(executor.masks_host.group_tensors.values())).dtype
            != torch.bool
        ):
            raise RuntimeError("CUDA graph action masks must be boolean tensors.")
        executor.obs_device = obs_buffers
        executor.masks_device = masks_buffers
        with torch.inference_mode():
            executor.hidden = agent.initial_hidden(self._cuda_graph_batch_size)
            executor.sampling_exponentials = (
                torch.ones(
                    (self._cuda_graph_batch_size, agent.action_head.num_actions),
                    device=self._device,
                ),
                torch.ones(
                    (self._cuda_graph_batch_size, agent.action_head.num_actions),
                    device=self._device,
                ),
                torch.ones(
                    (self._cuda_graph_batch_size, NUM_TERA_CHOICES),
                    device=self._device,
                ),
            )

        saved_rng = torch.cuda.get_rng_state(self._device).clone()
        try:
            with torch.cuda.stream(self._cuda_stream), torch.inference_mode():
                self._copy_cuda_graph_inputs(executor)
                self._stage_cuda_graph_hidden(executor, requests)
                for _ in range(self._CUDA_GRAPH_WARMUP_CALLS):
                    agent(
                        executor.obs_device.tensors,
                        executor.hidden,
                        action_masks=executor.masks_device.tensors,
                        greedy=getattr(self, "_greedy", False),
                        include_value=False,
                        include_policy_stats=False,
                        sampling_exponentials=executor.sampling_exponentials,
                    )
            self._cuda_stream.synchronize()

            graph = torch.cuda.CUDAGraph()
            with (
                torch.cuda.graph(graph, stream=self._cuda_stream),
                torch.inference_mode(),
            ):
                output = agent(
                    executor.obs_device.tensors,
                    executor.hidden,
                    action_masks=executor.masks_device.tensors,
                    greedy=getattr(self, "_greedy", False),
                    include_value=False,
                    include_policy_stats=False,
                    sampling_exponentials=executor.sampling_exponentials,
                )
            self._cuda_stream.synchronize()
            executor.graph = graph
            executor.output = output
            self._cuda_graph_captures += 1
        finally:
            torch.cuda.set_rng_state(saved_rng, self._device)
            torch.cuda.synchronize(self._device)

    def _copy_cuda_graph_inputs(self, executor: _CudaGraphExecutor) -> None:
        if executor.obs_device is None or executor.masks_device is None:
            raise RuntimeError("CUDA graph device inputs were not allocated.")
        for group_key, source in executor.obs_host.group_tensors.items():
            destination = executor.obs_device.group_tensors[group_key]
            destination.copy_(source, non_blocking=True)
        # Action masks are normalized to one packed bool allocation.
        mask_destination = next(iter(executor.masks_device.group_tensors.values()))
        mask_source = next(iter(executor.masks_host.group_tensors.values()))
        mask_destination.copy_(mask_source, non_blocking=True)

    def _stage_cuda_graph_hidden(
        self,
        executor: _CudaGraphExecutor,
        requests: list[_InferenceRequest],
    ) -> None:
        if executor.hidden is None:
            return
        static_h, static_c = executor.hidden
        static_h.zero_()
        static_c.zero_()
        batch_size = len(requests)
        if all(request.hidden is not None for request in requests):
            h = torch.cat(
                [self._to_inference_device(request.hidden[0]) for request in requests],
                dim=1,
            )
            c = torch.cat(
                [self._to_inference_device(request.hidden[1]) for request in requests],
                dim=1,
            )
            static_h[:, :batch_size].copy_(h)
            static_c[:, :batch_size].copy_(c)
            return
        for index, request in enumerate(requests):
            if request.hidden is None:
                continue
            static_h[:, index : index + 1].copy_(
                self._to_inference_device(request.hidden[0])
            )
            static_c[:, index : index + 1].copy_(
                self._to_inference_device(request.hidden[1])
            )

    def _replay_cuda_graph(
        self,
        executor: _CudaGraphExecutor,
        requests: list[_InferenceRequest],
        sampling_exponentials: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]
        | None,
    ) -> tuple[np.ndarray, list[tuple[torch.Tensor, torch.Tensor] | None]]:
        if self._cuda_stream is None or executor.graph is None:
            raise RuntimeError("CUDA graph executor was not captured.")
        batch_size = len(requests)
        with torch.cuda.stream(self._cuda_stream), torch.inference_mode():
            self._copy_cuda_graph_inputs(executor)
            self._stage_cuda_graph_hidden(executor, requests)
            if executor.sampling_exponentials is None:
                raise RuntimeError("CUDA graph sampling inputs were not allocated.")
            for static_values in executor.sampling_exponentials:
                static_values.fill_(1.0)
            if sampling_exponentials is not None:
                for static_values, source_values in zip(
                    executor.sampling_exponentials,
                    sampling_exponentials,
                    strict=True,
                ):
                    static_values[:batch_size].copy_(source_values)
            executor.graph.replay()
            output_hidden = executor.output.get("hidden")
            if output_hidden is None:
                stable_hidden = None
            else:
                stable_hidden = (
                    output_hidden[0][:, :batch_size].clone(),
                    output_hidden[1][:, :batch_size].clone(),
                )
        self._cuda_stream.synchronize()
        actions_tensor = executor.output.get("actions")
        if not isinstance(actions_tensor, torch.Tensor):
            raise RuntimeError("CUDA graph output omitted actions.")
        actions = actions_tensor[:batch_size].cpu().numpy()
        hidden_views = [
            self._slice_hidden(stable_hidden, index) for index in range(batch_size)
        ]
        return actions, hidden_views

    def _evict_cuda_graph(self, checkpoint_path: str) -> None:
        executor = getattr(self, "_cuda_graphs", {}).pop(checkpoint_path, None)
        if executor is not None:
            self._cuda_graph_evictions += 1

    def _clear_cuda_graphs(self) -> None:
        graphs = getattr(self, "_cuda_graphs", None)
        if graphs is None:
            return
        self._cuda_graph_evictions += len(graphs)
        graphs.clear()

    def _run_battle_batch(
        self, checkpoint_path: str, requests: list[_InferenceRequest]
    ) -> None:
        histogram = self._action_batch_size_histogram
        bucket = _action_batch_size_bucket(len(requests))
        histogram[bucket] += 1
        self._set_active_phase("get_agent")
        agent = self._get_agent(checkpoint_path)
        batch_started = time.perf_counter()
        batch_size = len(requests)
        self._add_batch_timing(
            requests,
            "opponent_infer_action_batch_size_sum",
            float(batch_size),
        )
        for request in requests:
            self._increment_timing(request.timing, "opponent_infer_action_requests")
            self._add_timing(
                request.timing, "opponent_infer_action_batches", 1.0 / batch_size
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

        if getattr(self, "_cuda_graphs_enabled", False):
            if self._run_cuda_graph_battle_batch(
                checkpoint_path,
                agent,
                requests,
                request_masks,
            ):
                return

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
            raise RuntimeError(
                "Battle inference staging produced a mismatched batch size."
            )
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
                    greedy=getattr(self, "_greedy", False),
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
        self._add_batch_timing(
            requests,
            "opponent_infer_preview_batch_size_sum",
            float(batch_size),
        )
        for request in requests:
            self._increment_timing(request.timing, "opponent_infer_preview_requests")
            self._add_timing(
                request.timing, "opponent_infer_preview_batches", 1.0 / batch_size
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
            raise RuntimeError(
                "Preview inference staging produced a mismatched batch size."
            )
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
                    greedy=getattr(self, "_greedy", False),
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
            stale_paths = [
                path for path in list(self._models) if path not in keep_paths
            ]
            for path in stale_paths:
                self._evict_cuda_graph(path)
            with self._condition:
                stale_models = [self._models.pop(path) for path in stale_paths]
                self._cache_evictions += len(stale_models)
                self._prefetch_paths.intersection_update(keep_paths)
            stale_models.clear()

        for request in requests:
            self._try_set_future_result(request.future, None)

    def _get_agent(self, checkpoint_path: str) -> PokeAgent:
        with self._condition:
            agent = self._models.get(checkpoint_path)
            if agent is not None:
                self._cache_hits += 1
                self._touch_model_locked(checkpoint_path)
                return agent
            self._cache_misses += 1

        loaded_agent = self._load_agent(checkpoint_path)
        stale_models: list[PokeAgent] = []
        stale_paths: list[str] = []
        with self._condition:
            self._raise_if_unavailable_locked()
            agent = self._models.get(checkpoint_path)
            if agent is None:
                agent = loaded_agent
                self._models[checkpoint_path] = agent
                self._cache_loads += 1
            else:
                # Another caller completed the same load while this copy was
                # being constructed. Retire the redundant copy below.
                stale_models.append(loaded_agent)
            self._touch_model_locked(checkpoint_path)
            self._prefetch_paths.discard(checkpoint_path)
            while len(self._models) > self._max_models:
                stale_path, stale = self._models.popitem(last=False)
                stale_paths.append(stale_path)
                stale_models.append(stale)
                self._cache_evictions += 1

        # CUDA model destruction can enqueue allocator work. Keep final
        # reference release in the model execution domain, just like pruning.
        if stale_models:
            with cuda_model_execution(self._device):
                for path in stale_paths:
                    self._evict_cuda_graph(path)
                stale_models.clear()
        return agent

    def _touch_model_locked(self, checkpoint_path: str) -> None:
        move_to_end = getattr(self._models, "move_to_end", None)
        if callable(move_to_end):
            move_to_end(checkpoint_path)

    def _load_agent(self, checkpoint_path: str) -> PokeAgent:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if not isinstance(checkpoint, Mapping):
            raise ValueError(
                f"League inference checkpoint {checkpoint_path!r} must be a mapping."
            )
        checkpoint_type = checkpoint.get("checkpoint_type")
        if checkpoint_type == "league":
            expected_format = LEAGUE_MODEL_CHECKPOINT_FORMAT_VERSION
        elif checkpoint_type == "trainer":
            expected_format = TRAINER_CHECKPOINT_FORMAT_VERSION
        else:
            raise ValueError(
                f"League inference checkpoint {checkpoint_path!r} has unsupported "
                f"checkpoint_type={checkpoint_type!r}."
            )
        if checkpoint.get("format_version") != expected_format:
            raise ValueError(
                f"League inference checkpoint {checkpoint_path!r} has unsupported "
                f"format_version={checkpoint.get('format_version')!r}; expected "
                f"{expected_format}."
            )
        missing = sorted({"agent", "model_metadata"}.difference(checkpoint))
        if missing:
            raise ValueError(
                f"League inference checkpoint {checkpoint_path!r} is missing "
                "required key(s): " + ", ".join(missing)
            )
        agent_copy = copy.deepcopy(self._agent_template)
        checkpoint_state = validate_model_checkpoint_metadata(
            path=checkpoint_path,
            checkpoint_state=checkpoint["agent"],
            checkpoint_metadata=checkpoint["model_metadata"],
            current_agent=agent_copy,
        )
        try:
            agent_copy.load_state_dict(checkpoint_state, strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"League checkpoint {checkpoint_path!r} does not exactly match the "
                f"current agent architecture: {exc}"
            ) from exc
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
        # Trainer-only semantic decisions are keyed by poke-env's normalized
        # battle-local request generation so the synchronous env cannot race a
        # concurrently prepared next request. This works for both direct
        # BattleStream and websocket battles without server-only rqids.
        self._intent_supervision_lock = Lock()
        self._intent_supervision_by_request: dict[
            tuple[str, int], SemanticActionSupervision | None
        ] = {}
        self._consumed_intent_requests: set[tuple[str, int]] = set()
        self._committed_intent: SemanticActionIntent | None = None
        self._pending_intent: SemanticActionIntent | None = None

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
        intent_request_key = semantic_action_request_key(battle)
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
                self._pending_intent = None
                self._pending_order = result
                self._record_intent_supervision(intent_request_key, None)
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
            obs = self.obs_builder.build(
                battle,
                previous_intent=self._committed_intent,
            )
            self._committed_intent = None
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
        supervision = semantic_action_supervision(actions, action_state)
        self._pending_intent = supervision.intent
        self._record_intent_supervision(intent_request_key, supervision)

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
        mismatch = self._record_preview_obs_mismatch(obs, prepared_obs)
        raise RuntimeError(
            "Prepared opponent team preview observation did not match the live battle "
            f"state: {mismatch}."
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
        with self._intent_supervision_lock:
            self._intent_supervision_by_request.clear()
            self._consumed_intent_requests.clear()
        self._committed_intent = None
        self._pending_intent = None

    def consume_intent_supervision(
        self,
        request_key: tuple[str, int],
    ) -> SemanticActionSupervision | None:
        """Consume exactly one trainer-only label for a Showdown request.

        A missing entry is an explicit no-decision result: poke-env bypasses
        ``choose_move`` for ``wait`` requests. Consumed state identities are
        retained until the episode reset so accidental double consumption or
        a late producer is surfaced instead of falling back to newer state.
        """
        with self._intent_supervision_lock:
            if request_key in self._consumed_intent_requests:
                raise RuntimeError(
                    "Opponent intent supervision was consumed twice for "
                    f"request {request_key!r}."
                )
            self._consumed_intent_requests.add(request_key)
            return self._intent_supervision_by_request.pop(request_key, None)

    def _record_intent_supervision(
        self,
        request_key: tuple[str, int],
        supervision: SemanticActionSupervision | None,
    ) -> None:
        with self._intent_supervision_lock:
            if request_key in self._consumed_intent_requests:
                raise RuntimeError(
                    "Opponent intent supervision arrived after request "
                    f"{request_key!r} was consumed."
                )
            # Hidden-trap and invalid-choice retries can request a replacement
            # decision without changing public state. Keep the final attempted
            # decision until that state is resolved and consumed.
            self._intent_supervision_by_request[request_key] = supervision

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
        self._committed_intent = self._pending_intent
        self._pending_hidden = None
        self._pending_intent = None
        self._pending_decision_key = None
        self._pending_order = None

    @classmethod
    def _decision_key(cls, battle: DoubleBattle) -> tuple[object, ...]:
        return (
            getattr(battle, "battle_tag", ""),
            int(getattr(battle, "request_generation", 0)),
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
        Hash only action-relevant fields, rather than serializing side/team
        payloads, while still distinguishing request updates whose legal action
        changed without advancing ``request_generation``.
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
    ) -> str:
        if actual.keys() != prepared.keys():
            self._increment_timing("opponent_precompute_preview_keyset_mismatch")
        else:
            for key in actual:
                if not np.array_equal(actual[key], prepared[key]):
                    self._increment_timing(
                        f"opponent_precompute_preview_mismatch_{key}"
                    )
        return summarize_observation_mismatch(actual, prepared)

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
    """Permanent self-play archive with payoff-aware opponent sampling.

    Every snapshot remains eligible on disk. ``TrainingConfig.league_size``
    bounds only each inference service's resident model cache, preventing the
    permanent archive from causing unbounded CPU or CUDA memory growth.
    """

    def __init__(
        self,
        config: TrainingConfig,
        env_config: EnvConfig,
        game_data: GameData,
        agent_template: PokeAgent,
        device: torch.device,
        *,
        rng: random.Random | None = None,
        inference_greedy: bool = False,
    ):
        self.config = config
        self.env_config = env_config
        self.game_data = game_data
        self.agent_template = agent_template
        self.device = device
        # League membership sampling has its own RNG so team-pool draws and
        # unrelated Python randomness cannot perturb the opponent schedule.
        self._rng = rng if rng is not None else random.Random(config.seed)
        self.inference_device = device
        self.preview_inference_device = (
            self.inference_device
            if config.league_preview_device == "action"
            else torch.device("cpu")
        )
        self.checkpoints: list[str] = []
        self._members: dict[str, LeagueMember] = {}
        self._current_update = 0
        self._frozen_population = False
        self._inference = _LeagueInferenceService(
            agent_template,
            self.inference_device,
            coalesce_window_sec=config.league_coalesce_ms / 1000.0,
            max_models=config.league_size,
            greedy=inference_greedy,
            enable_cuda_graphs=config.league_cuda_graphs,
            cuda_graph_batch_size=config.league_cuda_graph_batch_size,
            cuda_graph_cache_size=config.league_cuda_graph_cache_size,
        )
        if self.preview_inference_device == self.inference_device:
            self._preview_inference = self._inference
        else:
            self._preview_inference = _LeagueInferenceService(
                agent_template,
                self.preview_inference_device,
                coalesce_window_sec=config.league_coalesce_ms / 1000.0,
                max_models=config.league_size,
                greedy=inference_greedy,
                enable_cuda_graphs=False,
                cuda_graph_batch_size=config.league_cuda_graph_batch_size,
                cuda_graph_cache_size=config.league_cuda_graph_cache_size,
            )

    @property
    def current_update(self) -> int:
        """Last completed update represented by the payoff/archive state."""
        return self._current_update

    def install_frozen_population(self, checkpoint_paths: list[str]) -> None:
        """Install immutable external checkpoints for inference-only phases.

        A best-response phase owns its sampling distribution outside ``League``.
        This mode reuses the mature batched inference and ``LeaguePlayer`` path
        while defensively disabling every adaptive-archive operation.
        """
        if self.checkpoints or self._members or self._current_update != 0:
            raise RuntimeError(
                "Frozen population must be installed into an empty League."
            )
        if not checkpoint_paths:
            raise ValueError("Frozen population requires at least one checkpoint.")
        if len(set(checkpoint_paths)) != len(checkpoint_paths):
            raise ValueError("Frozen population contains duplicate checkpoint paths.")
        resolved_paths: list[str] = []
        for raw_path in checkpoint_paths:
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    "Frozen population checkpoint paths must be non-empty strings."
                )
            resolved = Path(raw_path).expanduser().resolve()
            if not resolved.is_file():
                raise FileNotFoundError(
                    f"Frozen population checkpoint does not exist: {resolved!s}"
                )
            resolved_paths.append(str(resolved))
        self.checkpoints = resolved_paths
        self._frozen_population = True

    def _require_adaptive_archive(self, operation: str) -> None:
        if getattr(self, "_frozen_population", False):
            raise RuntimeError(
                f"Cannot {operation} an inference-only frozen population."
            )

    def add_checkpoint(
        self,
        agent: PokeAgent,
        step: int,
        *,
        update: int | None = None,
    ) -> None:
        """Snapshot the current agent as a permanent archive member."""
        self._require_adaptive_archive("add a checkpoint to")
        if step < 0:
            raise ValueError(f"League checkpoint step must be non-negative: {step}.")
        member_update = self._current_update if update is None else update
        if member_update < self._current_update:
            raise ValueError(
                "League checkpoint update cannot move backwards: "
                f"current={self._current_update}, requested={member_update}."
            )
        path = f"{self.config.checkpoint_dir}/league/step_{step}.pt"
        if path in self._members:
            raise ValueError(f"League archive already contains {path!r}.")
        target = Path(path)
        if target.exists():
            raise FileExistsError(
                f"League checkpoint path already exists on disk: {target!s}"
            )
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "checkpoint_type": "league",
                "format_version": LEAGUE_MODEL_CHECKPOINT_FORMAT_VERSION,
                "model_metadata": model_checkpoint_metadata(agent),
                "agent": agent.state_dict(),
            },
            target,
        )
        checkpoint_sha256 = _checkpoint_file_sha256(path)
        self.checkpoints.append(path)
        self._members[path] = LeagueMember(
            path=path,
            checkpoint_sha256=checkpoint_sha256,
            archive_source="self_play_snapshot",
            step=step,
            created_update=member_update,
        )
        self._current_update = member_update

    def add_admitted_checkpoint(
        self,
        checkpoint_path: str | Path,
        checkpoint_sha256: str,
        *,
        step: int,
        update: int | None = None,
    ) -> str:
        """Add an already-admitted frozen policy to the adaptive main archive.

        The decision to admit a policy deliberately lives outside ``League``.
        This method only installs a checkpoint with a verified immutable byte
        identity.  It starts with the same neutral payoff prior as an ordinary
        self-play snapshot, after which the main policy's online outcomes feed
        the existing PFSP evidence path.

        Returns the canonical checkpoint path used as the league member key.
        """
        self._require_adaptive_archive("add an admitted checkpoint to")
        if type(step) is not int or step < 0:
            raise ValueError(
                f"Admitted checkpoint step must be a non-negative integer: {step!r}."
            )
        if not _is_sha256(checkpoint_sha256):
            raise ValueError(
                "Admitted checkpoint sha256 must be 64 lowercase hexadecimal digits."
            )
        member_update = self._current_update if update is None else update
        if type(member_update) is not int or member_update < 0:
            raise ValueError(
                "Admitted checkpoint update must be a non-negative integer: "
                f"{member_update!r}."
            )
        if member_update < self._current_update:
            raise ValueError(
                "Admitted checkpoint update cannot move backwards: "
                f"current={self._current_update}, requested={member_update}."
            )

        if not isinstance(checkpoint_path, (str, Path)) or not str(checkpoint_path):
            raise ValueError("Admitted checkpoint path must be non-empty.")
        resolved = Path(checkpoint_path).expanduser().resolve()
        actual_sha256 = _checkpoint_file_sha256(resolved)
        if actual_sha256 != checkpoint_sha256:
            raise ValueError(
                "Admitted checkpoint hash mismatch for "
                f"{resolved!s}: expected={checkpoint_sha256}, "
                f"actual={actual_sha256}."
            )
        canonical_path = str(resolved)
        existing_resolved_paths = {
            str(Path(path).expanduser().resolve()) for path in self.checkpoints
        }
        if canonical_path in existing_resolved_paths:
            raise ValueError(
                "League archive already contains admitted checkpoint path "
                f"{canonical_path!r}."
            )
        duplicate_identity = next(
            (
                member.path
                for member in self._members.values()
                if member.checkpoint_sha256 == checkpoint_sha256
            ),
            None,
        )
        if duplicate_identity is not None:
            raise ValueError(
                "League archive already contains checkpoint bytes "
                f"{checkpoint_sha256} at {duplicate_identity!r}."
            )

        self.checkpoints.append(canonical_path)
        self._members[canonical_path] = LeagueMember(
            path=canonical_path,
            checkpoint_sha256=checkpoint_sha256,
            archive_source="admitted_external",
            step=step,
            created_update=member_update,
        )
        self._current_update = member_update
        return canonical_path

    def record_outcomes(
        self,
        counts: Mapping[str | None, Mapping[str, int]],
        *,
        update: int,
    ) -> None:
        """Update discounted current-policy payoffs before the next PFSP draw."""
        self._require_adaptive_archive("record outcomes in")
        if update < self._current_update:
            raise ValueError(
                "League outcome update cannot move backwards: "
                f"current={self._current_update}, requested={update}."
            )
        for checkpoint_path, bucket in counts.items():
            if checkpoint_path is None:
                continue
            member = self._members.get(checkpoint_path)
            if member is None:
                raise KeyError(
                    "Outcome was attributed to a checkpoint outside the permanent "
                    f"league archive: {checkpoint_path!r}."
                )
            if set(bucket) != {"wins", "losses", "draws"}:
                raise ValueError(
                    f"League outcome bucket has invalid keys for {checkpoint_path!r}."
                )
            values = tuple(bucket[name] for name in ("wins", "losses", "draws"))
            if any(type(value) is not int or value < 0 for value in values):
                raise ValueError(
                    "League outcome counts must be non-negative integers for "
                    f"{checkpoint_path!r}: {values}."
                )
            wins, losses, draws = values
            if wins + losses + draws == 0:
                continue
            member.record_outcomes(
                update=update,
                wins=wins,
                losses=losses,
                draws=draws,
                half_life_updates=self.config.league_payoff_half_life_updates,
            )
        self._current_update = update

    def sample_checkpoint_paths(
        self,
        count: int,
        *,
        update: int | None = None,
    ) -> list[str]:
        """PFSP-sample distinct archive members whenever enough exist."""
        self._require_adaptive_archive("PFSP-sample from")
        if count <= 0:
            raise ValueError(f"Opponent sample count must be positive (got {count}).")
        sample_update = self._resolve_sampling_update(update)
        paths = self._draw_checkpoint_paths(count, sample_update)
        for path in set(paths):
            for service in self._inference_services():
                service.prefetch_checkpoint(path)
        return paths

    def sampling_distribution(
        self,
        *,
        update: int | None = None,
    ) -> list[dict[str, object]]:
        """Return the exact auditable PFSP distribution in archive order."""
        self._require_adaptive_archive("compute a PFSP distribution for")
        sample_update = self._resolve_sampling_update(update)
        if not self.checkpoints:
            return []

        rows: list[dict[str, object]] = []
        hard_weights: list[float] = []
        for path in self.checkpoints:
            member = self._members[path]
            win_probability, effective_games = member.payoff_summary(
                sample_update,
                self.config.league_payoff_half_life_updates,
            )
            hard_weight = (1.0 - win_probability) ** self.config.league_pfsp_exponent
            hard_weights.append(hard_weight)
            rows.append(
                {
                    "checkpoint_path": path,
                    "checkpoint_sha256": member.checkpoint_sha256,
                    "archive_source": member.archive_source,
                    "step": member.step,
                    "created_update": member.created_update,
                    "current_policy_win_probability": win_probability,
                    "discounted_effective_update_equivalents": effective_games,
                    "last_payoff_update": member.last_payoff_update,
                    "payoff_staleness_updates": (
                        None
                        if member.last_payoff_update is None
                        else sample_update - member.last_payoff_update
                    ),
                    "lifetime_wins": member.lifetime_wins,
                    "lifetime_losses": member.lifetime_losses,
                    "lifetime_draws": member.lifetime_draws,
                }
            )

        hard_total = math.fsum(hard_weights)
        member_count = len(rows)
        uniform_probability = 1.0 / member_count
        uniform_fraction = self.config.league_pfsp_uniform_fraction
        for row, hard_weight in zip(rows, hard_weights, strict=True):
            hard_probability = (
                hard_weight / hard_total if hard_total > 0.0 else uniform_probability
            )
            # This is the unconditional first draw. Later opponent-group draws
            # are renormalized after removing earlier members.
            row["first_draw_probability"] = (
                1.0 - uniform_fraction
            ) * hard_probability + uniform_fraction * uniform_probability
        return rows

    def describe(
        self,
        *,
        update: int | None = None,
        include_members: bool = True,
    ) -> dict[str, object]:
        if getattr(self, "_frozen_population", False):
            services = self._inference_services()
            cache = {"action": services[0].cache_info()}
            if len(services) > 1:
                cache["preview"] = services[1].cache_info()
            result: dict[str, object] = {
                "mode": "frozen_population_inference",
                "archive_size": len(self.checkpoints),
                "resident_cache_capacity": self.config.league_size,
                "inference_cache": cache,
            }
            if include_members:
                result["checkpoint_paths"] = list(self.checkpoints)
            return result
        sample_update = self._resolve_sampling_update(update)
        services = self._inference_services()
        cache = {"action": services[0].cache_info()}
        if len(services) > 1:
            cache["preview"] = services[1].cache_info()
        rows = self.sampling_distribution(update=sample_update)
        probabilities = [float(row["first_draw_probability"]) for row in rows]
        entropy = -math.fsum(
            probability * math.log(probability)
            for probability in probabilities
            if probability > 0.0
        )
        hardest = (
            min(
                rows,
                key=lambda row: float(row["current_policy_win_probability"]),
            )
            if rows
            else None
        )
        result: dict[str, object] = {
            "archive_size": len(self.checkpoints),
            "resident_cache_capacity": self.config.league_size,
            "sampling_update": sample_update,
            "payoff_half_life_updates": self.config.league_payoff_half_life_updates,
            "pfsp_exponent": self.config.league_pfsp_exponent,
            "pfsp_uniform_fraction": self.config.league_pfsp_uniform_fraction,
            "first_draw_entropy": entropy,
            "first_draw_effective_support": math.exp(entropy) if rows else 0.0,
            "first_draw_probability_min": min(probabilities, default=0.0),
            "first_draw_probability_max": max(probabilities, default=0.0),
            "hardest_checkpoint_path": (
                None if hardest is None else hardest["checkpoint_path"]
            ),
            "hardest_current_policy_win_probability": (
                None if hardest is None else hardest["current_policy_win_probability"]
            ),
            "inference_cache": cache,
        }
        if include_members:
            result["members"] = rows
        return result

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
        if getattr(self, "_frozen_population", False) and path not in self.checkpoints:
            raise ValueError(
                "Frozen-population opponent path is outside the pinned population: "
                f"{path!r}."
            )
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
        if not self.checkpoints:
            return None
        return self.sample_checkpoint_paths(1)[0]

    def save_state(
        self,
        path: str,
        *,
        extra_state: dict[str, object] | None = None,
    ) -> None:
        """Atomically persist league membership and private sampling state."""
        self._require_adaptive_archive("save adaptive league state for")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        league_config = {
            field: getattr(self.config, field) for field in _LEAGUE_STATE_CONFIG_FIELDS
        }
        members = [
            {
                field: getattr(self._members[checkpoint_path], field)
                for field in _LEAGUE_MEMBER_STATE_FIELDS
            }
            for checkpoint_path in self.checkpoints
        ]
        payload = {
            "format_version": _LEAGUE_STATE_FORMAT_VERSION,
            "league_config": league_config,
            "current_update": self._current_update,
            "members": members,
            "rng_state": _tuples_to_lists(self._rng.getstate()),
            "extra_state": _tuples_to_lists(extra_state or {}),
        }
        try:
            temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    def load_state(self, path: str) -> dict[str, object]:
        """Strictly restore checkpoint paths and the league sampling state."""
        self._require_adaptive_archive("load adaptive league state into")
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("League state must contain a versioned JSON object.")
        if payload.get("format_version") != _LEAGUE_STATE_FORMAT_VERSION:
            raise ValueError(
                "Unsupported league-state format version: "
                f"{payload.get('format_version')!r}."
            )
        saved_league_config = payload.get("league_config")
        if not isinstance(saved_league_config, dict) or set(saved_league_config) != set(
            _LEAGUE_STATE_CONFIG_FIELDS
        ):
            raise ValueError("League state has an invalid league_config.")
        positive_integer_fields = (
            "league_size",
            "league_update_interval",
            "league_refresh_interval",
        )
        if any(
            type(saved_league_config[field]) is not int
            or saved_league_config[field] <= 0
            for field in positive_integer_fields
        ):
            raise ValueError("League state schedule values must be positive integers.")
        for field in (
            "league_payoff_half_life_updates",
            "league_pfsp_exponent",
            "league_pfsp_uniform_fraction",
        ):
            value = saved_league_config[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"League state has invalid numeric {field!r}.")
        if saved_league_config["league_payoff_half_life_updates"] <= 0.0:
            raise ValueError("League payoff half-life must be positive.")
        if saved_league_config["league_pfsp_exponent"] <= 0.0:
            raise ValueError("League PFSP exponent must be positive.")
        if not 0.0 <= saved_league_config["league_pfsp_uniform_fraction"] <= 1.0:
            raise ValueError("League PFSP uniform fraction must be within [0, 1].")
        expected_league_config = {
            field: getattr(self.config, field) for field in _LEAGUE_STATE_CONFIG_FIELDS
        }
        if saved_league_config != expected_league_config:
            raise ValueError(
                "Configured league schedule does not match the resume state: "
                f"configured={expected_league_config}, saved={saved_league_config}."
            )
        current_update = payload.get("current_update")
        if type(current_update) is not int or current_update < 0:
            raise ValueError("League state current_update must be non-negative.")
        saved_members = payload.get("members")
        if not isinstance(saved_members, list) or not saved_members:
            raise ValueError("League state members must be a non-empty list.")

        members: list[LeagueMember] = []
        for raw_member in saved_members:
            if not isinstance(raw_member, dict) or set(raw_member) != set(
                _LEAGUE_MEMBER_STATE_FIELDS
            ):
                raise ValueError("League state contains an invalid member record.")
            path_value = raw_member["path"]
            if not isinstance(path_value, str) or not path_value:
                raise ValueError("League member path must be a non-empty string.")
            checkpoint_sha256 = raw_member["checkpoint_sha256"]
            if not _is_sha256(checkpoint_sha256):
                raise ValueError(
                    "League member checkpoint_sha256 must be 64 lowercase "
                    "hexadecimal digits."
                )
            archive_source = raw_member["archive_source"]
            if archive_source not in _LEAGUE_MEMBER_SOURCES:
                raise ValueError(
                    "League member archive_source must be one of "
                    f"{sorted(_LEAGUE_MEMBER_SOURCES)!r}."
                )
            for field in (
                "step",
                "created_update",
                "lifetime_wins",
                "lifetime_losses",
                "lifetime_draws",
            ):
                value = raw_member[field]
                if type(value) is not int or value < 0:
                    raise ValueError(
                        f"League member {field!r} must be a non-negative integer."
                    )
            if raw_member["created_update"] > current_update:
                raise ValueError("League member was created after the saved update.")
            last_payoff_update = raw_member["last_payoff_update"]
            if last_payoff_update is not None and (
                type(last_payoff_update) is not int or last_payoff_update < 0
            ):
                raise ValueError(
                    "League member last_payoff_update must be null or non-negative."
                )
            if last_payoff_update is not None and last_payoff_update > current_update:
                raise ValueError("League member payoff evidence is from the future.")
            for field in ("discounted_wins", "discounted_losses"):
                value = raw_member[field]
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or value < 0.0
                ):
                    raise ValueError(
                        f"League member {field!r} must be finite and non-negative."
                    )
            if last_payoff_update is None and (
                raw_member["discounted_wins"] != 0.0
                or raw_member["discounted_losses"] != 0.0
            ):
                raise ValueError(
                    "League member without a payoff update cannot contain evidence."
                )
            members.append(
                LeagueMember(
                    path=path_value,
                    checkpoint_sha256=checkpoint_sha256,
                    archive_source=archive_source,
                    step=raw_member["step"],
                    created_update=raw_member["created_update"],
                    discounted_wins=float(raw_member["discounted_wins"]),
                    discounted_losses=float(raw_member["discounted_losses"]),
                    last_payoff_update=last_payoff_update,
                    lifetime_wins=raw_member["lifetime_wins"],
                    lifetime_losses=raw_member["lifetime_losses"],
                    lifetime_draws=raw_member["lifetime_draws"],
                )
            )
        paths = [member.path for member in members]
        if len(set(paths)) != len(paths):
            raise ValueError("League state contains duplicate archive paths.")
        missing_paths = [
            checkpoint_path
            for checkpoint_path in paths
            if not Path(checkpoint_path).is_file()
        ]
        if missing_paths:
            raise FileNotFoundError(
                "Strict resume requires every saved league checkpoint; missing: "
                + ", ".join(missing_paths)
            )
        hash_mismatches: list[str] = []
        for member in members:
            actual_sha256 = _checkpoint_file_sha256(member.path)
            if actual_sha256 != member.checkpoint_sha256:
                hash_mismatches.append(
                    f"{member.path} (expected={member.checkpoint_sha256}, "
                    f"actual={actual_sha256})"
                )
        if hash_mismatches:
            raise ValueError(
                "Strict resume requires the exact saved league checkpoint bytes; "
                "hash mismatch: " + ", ".join(hash_mismatches)
            )
        rng_state = _lists_to_tuples(payload.get("rng_state"))
        try:
            validated_rng = random.Random()
            validated_rng.setstate(rng_state)
        except (TypeError, ValueError) as exc:
            raise ValueError("League state contains an invalid RNG state.") from exc
        extra_state = _lists_to_tuples(payload.get("extra_state", {}))
        if not isinstance(extra_state, dict):
            raise ValueError("League state extra_state must be an object.")

        self._rng.setstate(rng_state)
        self.checkpoints = paths
        self._members = {member.path: member for member in members}
        self._current_update = current_update
        keep_paths = set(self.checkpoints)
        for service in self._inference_services():
            service.prune_checkpoints(keep_paths)
        return extra_state

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
        """Internal single PFSP draw without inference prefetch."""
        if not self.checkpoints:
            return None
        return self._draw_checkpoint_paths(1, self._current_update)[0]

    def _resolve_sampling_update(self, update: int | None) -> int:
        sample_update = self._current_update if update is None else update
        if type(sample_update) is not int or sample_update < 0:
            raise ValueError(
                f"League sampling update must be non-negative (got {sample_update})."
            )
        if sample_update < self._current_update:
            raise ValueError(
                "League sampling update cannot move backwards: "
                f"current={self._current_update}, requested={sample_update}."
            )
        return sample_update

    def _draw_checkpoint_paths(self, count: int, update: int) -> list[str]:
        rows = self.sampling_distribution(update=update)
        if not rows:
            return []

        available_paths = [str(row["checkpoint_path"]) for row in rows]
        available_weights = [float(row["first_draw_probability"]) for row in rows]
        sampled: list[str] = []

        # First exhaust a weighted-without-replacement draw, guaranteeing
        # distinct opponent groups whenever the archive is large enough.
        unique_draws = min(count, len(available_paths))
        for _ in range(unique_draws):
            index = self._weighted_index(available_weights)
            sampled.append(available_paths.pop(index))
            available_weights.pop(index)

        # During initial seeding a single member may need to serve multiple
        # groups. Once enough members exist, this branch is never reached.
        if len(sampled) < count:
            full_paths = [str(row["checkpoint_path"]) for row in rows]
            full_weights = [float(row["first_draw_probability"]) for row in rows]
            for _ in range(count - len(sampled)):
                sampled.append(full_paths[self._weighted_index(full_weights)])
        return sampled

    def _weighted_index(self, weights: list[float]) -> int:
        total = math.fsum(weights)
        if total <= 0.0 or not math.isfinite(total):
            raise RuntimeError(f"Invalid PFSP sampling weights: {weights!r}.")
        threshold = self._rng.random() * total
        cumulative = 0.0
        for index, weight in enumerate(weights):
            cumulative += weight
            if threshold < cumulative:
                return index
        # Floating-point roundoff can place the threshold at the endpoint.
        return len(weights) - 1
