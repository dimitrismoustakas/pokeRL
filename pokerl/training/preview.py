"""Batched team preview inference for the live training policy."""

from __future__ import annotations

import time
from collections.abc import Mapping
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Condition, Thread

import numpy as np
import torch

from ..model.agent import PokeAgent
from .cuda_execution import cuda_model_execution


@dataclass(frozen=True)
class TeamPreviewResult:
    """One preview decision returned by a callback or the live-policy batcher."""

    picks: np.ndarray
    policy_version: int | None
    log_prob: float | None = None
    value: float | None = None


def _copy_array_mapping(
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    return {key: np.array(value, copy=True) for key, value in values.items()}


@dataclass(frozen=True)
class TeamPreviewEpisodeStart:
    """Immutable trainer record for a preview action at episode start."""

    obs: Mapping[str, np.ndarray]
    picks: np.ndarray
    log_prob: float
    value: float
    policy_version: int
    opponent_picks: np.ndarray
    opponent_aux_labels: Mapping[str, np.ndarray] | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "obs", _copy_array_mapping(self.obs))
        object.__setattr__(
            self,
            "picks",
            np.array(self.picks, dtype=np.int64, copy=True),
        )
        object.__setattr__(self, "log_prob", float(self.log_prob))
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(
            self,
            "opponent_picks",
            np.array(self.opponent_picks, dtype=np.int64, copy=True),
        )
        if self.opponent_aux_labels is not None:
            object.__setattr__(
                self,
                "opponent_aux_labels",
                _copy_array_mapping(self.opponent_aux_labels),
            )


@dataclass
class _PreviewRequest:
    obs: dict[str, np.ndarray]
    future: Future
    policy_version: int


class TeamPreviewBatcher:
    """Batches concurrent team preview requests from environment threads."""

    _COALESCE_WINDOW_SEC = 0.005
    _TARGET_BATCH_SIZE = 8
    _CLOSE_TIMEOUT_SEC = 60.0

    def __init__(
        self,
        agent: PokeAgent,
        device: torch.device,
        *,
        policy_version: int = 0,
    ):
        if (
            isinstance(policy_version, bool)
            or not isinstance(policy_version, int)
            or policy_version < 0
        ):
            raise ValueError(
                f"policy_version must be non-negative (got {policy_version})."
            )
        self._agent = agent
        self._device = device
        self._policy_version = policy_version
        self._pending: list[_PreviewRequest] = []
        self._active_request_count = 0
        self._closed = False
        self._fatal_error: BaseException | None = None
        self._condition = Condition()
        self._thread = Thread(
            target=self._run, name="team-preview-batcher", daemon=True
        )
        self._thread.start()

    @property
    def policy_version(self) -> int:
        with self._condition:
            return self._policy_version

    def advance_policy_version(self) -> int:
        """Advance after an optimizer cycle once old-version work is drained."""
        with self._condition:
            if self._closed:
                raise RuntimeError(
                    "Cannot advance the policy version of a closed preview batcher."
                )
            if self._pending or self._active_request_count:
                raise RuntimeError(
                    "Cannot advance team-preview policy version while inference "
                    "requests are pending or active."
                )
            self._policy_version += 1
            return self._policy_version

    def choose(self, obs: dict[str, np.ndarray]) -> TeamPreviewResult:
        return self.submit(obs).result()

    def submit(self, obs: dict[str, np.ndarray]) -> Future:
        future: Future = Future()
        with self._condition:
            if self._fatal_error is not None:
                raise RuntimeError(
                    "TeamPreviewBatcher worker failed."
                ) from self._fatal_error
            if self._closed:
                raise RuntimeError("TeamPreviewBatcher is closed.")
            if not self._thread.is_alive():
                raise RuntimeError("TeamPreviewBatcher worker stopped unexpectedly.")
            self._pending.append(
                _PreviewRequest(
                    obs=obs,
                    future=future,
                    policy_version=self._policy_version,
                )
            )
            self._condition.notify()
        return future

    def close(self) -> None:
        with self._condition:
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()

        for request in pending:
            if not request.future.done():
                request.future.set_exception(RuntimeError("TeamPreviewBatcher closed."))

        if self._thread.is_alive():
            self._thread.join(timeout=self._CLOSE_TIMEOUT_SEC)
        if self._thread.is_alive():
            raise RuntimeError(
                "TeamPreviewBatcher worker did not stop within "
                f"{self._CLOSE_TIMEOUT_SEC:g} seconds."
            )

    def _run(self) -> None:
        try:
            self._run_loop()
        except BaseException as exc:
            with self._condition:
                self._fatal_error = exc
                self._closed = True
                pending = list(self._pending)
                self._pending.clear()
                self._condition.notify_all()
            for request in pending:
                if not request.future.done():
                    request.future.set_exception(
                        RuntimeError("TeamPreviewBatcher worker failed.")
                    )

    def _run_loop(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                if self._COALESCE_WINDOW_SEC > 0:
                    deadline = time.perf_counter() + self._COALESCE_WINDOW_SEC
                    while (
                        not self._closed
                        and len(self._pending) < self._TARGET_BATCH_SIZE
                    ):
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                batch = self._pending
                self._pending = []
                # Claim the snapshot before releasing the condition so version
                # advancement cannot slip between pending removal and inference.
                self._active_request_count = len(batch)

            batch = [
                request
                for request in batch
                if request.future.set_running_or_notify_cancel()
            ]
            if not batch:
                with self._condition:
                    self._active_request_count = 0
                    self._condition.notify_all()
                continue

            try:
                with cuda_model_execution(self._device):
                    if getattr(self._agent, "training", False):
                        raise RuntimeError(
                            "TeamPreviewBatcher cannot run while the live policy is in training mode."
                        )
                    obs_batch = {
                        key: torch.from_numpy(
                            np.stack([request.obs[key] for request in batch])
                        ).to(self._device)
                        for key in batch[0].obs
                    }
                    with torch.inference_mode():
                        preview = self._agent.team_preview(
                            obs_batch,
                            include_value=True,
                            include_policy_stats=False,
                            include_opponent_preview=False,
                        )
                        picks = preview["picks"].cpu().numpy()
                        log_probs = preview["log_prob"].cpu().numpy()
                        values = preview["preview_value"].cpu().numpy()
            except Exception as exc:
                with self._condition:
                    self._active_request_count = 0
                    self._condition.notify_all()
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(exc)
            else:
                # Inference no longer reads model weights. Mark the batch idle
                # before publishing results so the optimizer may advance the
                # version without racing an old-version model invocation.
                with self._condition:
                    self._active_request_count = 0
                    self._condition.notify_all()
                for index, request in enumerate(batch):
                    if not request.future.done():
                        request.future.set_result(
                            TeamPreviewResult(
                                picks=picks[index],
                                policy_version=request.policy_version,
                                log_prob=float(log_probs[index]),
                                value=float(values[index]),
                            )
                        )
