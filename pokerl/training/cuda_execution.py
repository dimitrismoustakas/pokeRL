"""Shared CUDA execution domain for model work across training threads."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import RLock, local

import torch


_CUDA_MODEL_EXECUTION_LOCK = RLock()
_CUDA_MODEL_EXECUTION_STATE = local()


def cuda_model_execution(device: torch.device):
    """Serialize CUDA work and finish it before handing execution to another thread.

    pokeRL runs the learner and frozen league opponents in different Python
    threads. A host-side lock alone is insufficient because CUDA launches are
    asynchronous: work from one holder could still overlap work launched by the
    next holder on a different per-thread stream. The outermost guarded section
    therefore synchronizes its device before releasing the process-wide lock.
    Nested sections are synchronized only once. CPU execution needs no guard.
    """

    if device.type != "cuda":
        return nullcontext()
    return _synchronized_cuda_execution(device)


@contextmanager
def _synchronized_cuda_execution(device: torch.device):
    with _CUDA_MODEL_EXECUTION_LOCK:
        depth = getattr(_CUDA_MODEL_EXECUTION_STATE, "depth", 0)
        _CUDA_MODEL_EXECUTION_STATE.depth = depth + 1
        failed = False
        try:
            yield
        except BaseException:
            failed = True
            raise
        finally:
            _CUDA_MODEL_EXECUTION_STATE.depth = depth
            if depth == 0:
                if failed:
                    # Do not hide the original exception with a secondary sync
                    # failure, but make a best effort to drain queued work.
                    try:
                        torch.cuda.synchronize(device)
                    except Exception:
                        pass
                else:
                    torch.cuda.synchronize(device)
