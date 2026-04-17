"""Batched team preview inference for the live training policy."""

from __future__ import annotations

import time
from concurrent.futures import Future
from dataclasses import dataclass
from threading import Condition, Thread

import numpy as np
import torch

from ..model.agent import PokeAgent


@dataclass
class _PreviewRequest:
    obs: dict[str, np.ndarray]
    future: Future


class TeamPreviewBatcher:
    """Batches concurrent team preview requests from environment threads."""

    _COALESCE_WINDOW_SEC = 0.005
    _TARGET_BATCH_SIZE = 8

    def __init__(self, agent: PokeAgent, device: torch.device):
        self._agent = agent
        self._device = device
        self._pending: list[_PreviewRequest] = []
        self._closed = False
        self._condition = Condition()
        self._thread = Thread(target=self._run, name="team-preview-batcher", daemon=True)
        self._thread.start()

    def choose(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        return self.submit(obs).result()

    def submit(self, obs: dict[str, np.ndarray]) -> Future:
        future: Future = Future()
        with self._condition:
            if self._closed:
                raise RuntimeError("TeamPreviewBatcher is closed.")
            self._pending.append(_PreviewRequest(obs=obs, future=future))
            self._condition.notify()
        return future

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()

        for request in pending:
            request.future.set_exception(RuntimeError("TeamPreviewBatcher closed."))

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                if self._COALESCE_WINDOW_SEC > 0:
                    deadline = time.perf_counter() + self._COALESCE_WINDOW_SEC
                    while not self._closed and len(self._pending) < self._TARGET_BATCH_SIZE:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                batch = self._pending
                self._pending = []

            try:
                obs_batch = {
                    key: torch.from_numpy(np.stack([request.obs[key] for request in batch])).to(self._device)
                    for key in batch[0].obs
                }
                with torch.inference_mode():
                    picks = self._agent.team_preview(obs_batch)["picks"].cpu().numpy()
                for index, request in enumerate(batch):
                    request.future.set_result(picks[index])
            except Exception as exc:
                for request in batch:
                    request.future.set_exception(exc)
