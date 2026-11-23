"""Self-play league: a pool of past agent checkpoints as opponents."""

import copy
import math
import random
from collections import OrderedDict
from collections.abc import Mapping
from contextlib import nullcontext
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from weakref import WeakKeyDictionary

import numpy as np
import torch

from ..model.agent import PokeAgent, policy_weights
from ..model.preview_cache import TeamPreviewCache
from .inference_batch import BatchSchema, PackedInputs
from ..config import TrainingConfig
from ..env.vector_inputs import VectorInputRow
from ..env.actions import (
    NUM_TERA_CHOICES,
)
from .cuda_execution import cuda_model_execution


@dataclass
class LeagueMember:
    """One permanent archive member and its live-policy payoff evidence."""

    path: str
    step: int
    created_update: int
    discounted_wins: float = 0.0
    discounted_losses: float = 0.0
    last_payoff_update: int | None = None
    lifetime_wins: int = 0
    lifetime_losses: int = 0
    lifetime_draws: int = 0
    assignment_balance: float = 0.0

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


@dataclass
class _InferenceRow:
    obs: Mapping[str, np.ndarray]
    masks: Mapping[str, np.ndarray] | None
    hidden: tuple[torch.Tensor, torch.Tensor] | None


def _shared_vector_columns(rows):
    first = rows[0]
    if isinstance(first, VectorInputRow) and all(
        isinstance(row, VectorInputRow) and row.columns is first.columns for row in rows
    ):
        return first.columns
    return None


@dataclass
class _CudaGraphExecutor:
    """One fixed-shape CUDA graph tied to a frozen checkpoint's parameters."""

    checkpoint_path: str
    obs_schema: BatchSchema
    masks_schema: BatchSchema
    obs_host: PackedInputs
    masks_host: PackedInputs
    obs_device: PackedInputs | None
    masks_device: PackedInputs | None
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
    """Run vector decisions by frozen checkpoint and prefetch opponents on the same worker."""

    _DEFAULT_CUDA_GRAPH_BATCH_SIZE = 16
    _DEFAULT_CUDA_GRAPH_CACHE_SIZE = 4
    _CUDA_GRAPH_WARMUP_CALLS = 3

    def __init__(
        self,
        agent_template: PokeAgent,
        device: torch.device,
        *,
        max_models: int,
        greedy: bool = False,
        enable_cuda_graphs: bool = False,
        cuda_graph_batch_size: int = _DEFAULT_CUDA_GRAPH_BATCH_SIZE,
        cuda_graph_cache_size: int = _DEFAULT_CUDA_GRAPH_CACHE_SIZE,
    ):
        self._agent_template = copy.deepcopy(agent_template).to(torch.device("cpu"))
        self._agent_template.eval()
        self._agent_template.requires_grad_(False)
        self._device = device
        self._cuda_stream: torch.cuda.Stream | None = None
        self._preview_encodings = WeakKeyDictionary()
        self._greedy = greedy
        self._max_models = max_models
        self._models: OrderedDict[str, PokeAgent] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._cache_loads = 0
        self._cache_evictions = 0
        self._host_batch_buffers: dict[str, PackedInputs] = {}
        self._device_batch_buffers: dict[str, PackedInputs] = {}
        self._cuda_graphs_enabled = bool(enable_cuda_graphs and device.type == "cuda")
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
        self._worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="league")
        self._prefetches: dict[str, Future] = {}

    def prefetch_checkpoint(self, path: str) -> None:
        if path not in self._prefetches:
            self._prefetches[path] = self._worker.submit(self._prefetch, path)

    def _prefetch(self, path: str) -> None:
        self._get_agent(path)

    def _finish_prefetches(self) -> None:
        pending, self._prefetches = self._prefetches, {}
        for future in pending.values():
            future.result()

    def cache_info(self) -> dict[str, object]:
        """Return a consistent snapshot of bounded model-cache diagnostics."""
        return self._worker.submit(self._cache_info).result()

    def _cache_info(self) -> dict[str, object]:
        return {
            "capacity": self._max_models,
            "resident_models": len(self._models),
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "loads": self._cache_loads,
            "evictions": self._cache_evictions,
            "action_batch_size_histogram": dict(self._action_batch_size_histogram),
            "cuda_graphs_enabled": self._cuda_graphs_enabled,
            "cuda_graph_batch_size": self._cuda_graph_batch_size,
            "cuda_graph_capacity": self._cuda_graph_cache_size,
            "resident_cuda_graphs": len(self._cuda_graphs),
            "cuda_graph_hits": self._cuda_graph_hits,
            "cuda_graph_misses": self._cuda_graph_misses,
            "cuda_graph_captures": self._cuda_graph_captures,
            "cuda_graph_replays": self._cuda_graph_replays,
            "cuda_graph_evictions": self._cuda_graph_evictions,
            "cuda_graph_fallbacks": self._cuda_graph_fallbacks,
            "cuda_graph_real_rows": self._cuda_graph_real_rows,
            "cuda_graph_padded_rows": self._cuda_graph_padded_rows,
            "cuda_graph_last_error": self._cuda_graph_last_error,
        }

    def run_grouped(self, paths, forward):
        """Run each checkpoint's batch on the inference worker and restore lane order."""
        self._finish_prefetches()

        def run():
            groups = {}
            for lane, path in enumerate(paths):
                groups.setdefault(path, []).append(lane)
            results = [None] * len(paths)
            for path, lanes in groups.items():
                for lane, result in zip(lanes, forward(path, lanes), strict=True):
                    results[lane] = result
            return results

        return self._worker.submit(run).result()

    def close(self) -> None:
        self._worker.shutdown(wait=True)
        try:
            self._finish_prefetches()
        finally:
            with cuda_model_execution(self._device):
                self._clear_cuda_graphs()
                self._models.clear()
                self._device_batch_buffers.clear()
            self._host_batch_buffers.clear()

    @staticmethod
    def _batch_schema(
        mappings: list[Mapping[str, np.ndarray]],
    ) -> BatchSchema:
        columns = _shared_vector_columns(mappings)
        if columns is not None:
            return tuple(
                (key, array.shape[1:], array.dtype.str)
                for key, array in columns.items()
            )
        return tuple(
            (key, value.shape, value.dtype.str) for key, value in mappings[0].items()
        )

    def _pack_cpu_batch(
        self,
        cache_key: str,
        mappings: list[Mapping[str, np.ndarray]],
    ) -> tuple[PackedInputs, int]:
        """Pack one request batch into reusable worker-private CPU storage."""
        schema = self._batch_schema(mappings)
        batch_size = len(mappings)
        caches = self._host_batch_buffers
        buffers = caches.get(cache_key)
        if buffers is None or buffers.schema != schema:
            buffers = PackedInputs(schema, batch_size, torch.device("cpu"))
            caches[cache_key] = buffers
        elif batch_size > buffers.capacity:
            capacity = max(batch_size, 2 * buffers.capacity)
            buffers = PackedInputs(schema, capacity, torch.device("cpu"))
            caches[cache_key] = buffers

        self._pack_input_rows(buffers, mappings, size=batch_size)
        return buffers, batch_size

    def _copy_staged_batch_to_device(
        self,
        cache_key: str,
        host_buffers: PackedInputs,
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

        caches = self._device_batch_buffers
        buffers = caches.get(cache_key)
        if (
            buffers is None
            or buffers.schema != host_buffers.schema
            or buffers.capacity < host_buffers.capacity
            or buffers.force_bool != force_bool
        ):
            buffers = PackedInputs(
                host_buffers.schema,
                host_buffers.capacity,
                self._device,
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
        requests: list[_InferenceRow],
        request_masks: list[Mapping[str, np.ndarray]],
    ):
        """Run one logical request batch through bounded fixed-shape graphs."""
        chunk_size = self._cuda_graph_batch_size
        chunk_results: list[
            tuple[np.ndarray, list[tuple[torch.Tensor, torch.Tensor] | None]]
        ] = []

        saved_rng: torch.Tensor | None = None
        try:
            with cuda_model_execution(self._device):
                saved_rng = torch.cuda.get_rng_state(self._device).clone()
                sampling_exponentials = self._make_sampling_exponentials(
                    agent,
                    len(requests),
                )
                for start in range(0, len(requests), chunk_size):
                    stop = min(start + chunk_size, len(requests))
                    chunk = requests[start:stop]
                    chunk_masks = request_masks[start:stop]

                    executor = self._prepare_cuda_graph_executor(
                        checkpoint_path,
                        [request.obs for request in chunk],
                        chunk_masks,
                    )

                    if executor.graph is None:
                        self._capture_cuda_graph(executor, agent, chunk)

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
            return None

        self._cuda_graph_replays += len(chunk_results)
        self._cuda_graph_real_rows += len(requests)
        self._cuda_graph_padded_rows += len(chunk_results) * chunk_size - len(requests)

        return [
            pair
            for actions, hidden in chunk_results
            for pair in zip(actions, hidden, strict=True)
        ]

    def _make_sampling_exponentials(
        self,
        agent: PokeAgent,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if self._greedy:
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
        observations: list[Mapping[str, np.ndarray]],
        masks: list[Mapping[str, np.ndarray]],
    ) -> _CudaGraphExecutor:
        obs_schema = self._batch_schema(observations)
        masks_schema = self._batch_schema(masks)
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
            obs_host = PackedInputs(
                obs_schema,
                self._cuda_graph_batch_size,
                torch.device("cpu"),
                pin_memory=True,
            )
            masks_host = PackedInputs(
                masks_schema,
                self._cuda_graph_batch_size,
                torch.device("cpu"),
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

        self._pack_input_rows(
            executor.obs_host, observations, size=executor.obs_host.capacity
        )
        self._pack_input_rows(
            executor.masks_host, masks, size=executor.masks_host.capacity
        )
        return executor

    @staticmethod
    def _pack_input_rows(
        buffers: PackedInputs,
        mappings: list[Mapping[str, np.ndarray]],
        *,
        size: int,
    ) -> None:
        batch_size = len(mappings)
        if not 0 < batch_size <= size <= buffers.capacity:
            raise ValueError("Inference batch exceeds its input buffer capacity.")
        columns = _shared_vector_columns(mappings)
        if columns is not None:
            indices = np.empty(size, dtype=np.intp)
            indices[:batch_size] = [row.lane for row in mappings]
            indices[batch_size:] = indices[batch_size - 1]
            for key, array in columns.items():
                buffers.arrays[key][:size] = array[indices]
            return
        for key, _, _ in buffers.schema:
            np.stack(
                [mapping[key] for mapping in mappings],
                axis=0,
                out=buffers.arrays[key][:batch_size],
                casting="no",
            )
            if batch_size < size:
                buffers.arrays[key][batch_size:size] = buffers.arrays[key][
                    batch_size - 1
                ]

    def _capture_cuda_graph(
        self,
        executor: _CudaGraphExecutor,
        agent: PokeAgent,
        requests: list[_InferenceRow],
    ) -> None:
        if self._cuda_stream is None:
            self._cuda_stream = torch.cuda.Stream(device=self._device)
        obs_buffers = PackedInputs(
            executor.obs_host.schema,
            executor.obs_host.capacity,
            self._device,
            force_bool=False,
        )
        masks_buffers = PackedInputs(
            executor.masks_host.schema,
            executor.masks_host.capacity,
            self._device,
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
                        greedy=self._greedy,
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
                    greedy=self._greedy,
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
        requests: list[_InferenceRow],
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
        requests: list[_InferenceRow],
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
        executor = self._cuda_graphs.pop(checkpoint_path, None)
        if executor is not None:
            self._cuda_graph_evictions += 1

    def _clear_cuda_graphs(self) -> None:
        graphs = self._cuda_graphs
        if graphs is None:
            return
        self._cuda_graph_evictions += len(graphs)
        graphs.clear()

    def _run_battle_batch(self, checkpoint_path: str, requests: list[_InferenceRow]):
        histogram = self._action_batch_size_histogram
        bucket = _action_batch_size_bucket(len(requests))
        histogram[bucket] += 1
        agent = self._get_agent(checkpoint_path)
        masks = [row.masks for row in requests]
        if self._cuda_graphs_enabled:
            results = self._run_cuda_graph_battle_batch(
                checkpoint_path, agent, requests, masks
            )
            if results is not None:
                return results

        obs_host, count = self._pack_cpu_batch(
            "battle_obs", [row.obs for row in requests]
        )
        masks_host, _ = self._pack_cpu_batch("battle_masks", masks)
        with cuda_model_execution(self._device):
            obs = self._copy_staged_batch_to_device("battle_obs", obs_host, count)
            legal = self._copy_staged_batch_to_device(
                "battle_masks", masks_host, count, force_bool=True
            )
            hidden = self._build_hidden_batch(agent, requests)
            with torch.inference_mode(), self._device_execution_context():
                out = agent(
                    obs,
                    hidden,
                    action_masks=legal,
                    greedy=self._greedy,
                    include_value=False,
                    include_policy_stats=False,
                )
            self._synchronize_device_execution()
            actions = out["actions"].cpu().numpy()
            return [
                (action, self._slice_hidden(out["hidden"], index))
                for index, action in enumerate(actions)
            ]

    def _run_preview_batch(self, checkpoint_path, observations, indices):
        agent = self._get_agent(checkpoint_path)
        cache = self._preview_encodings.get(agent)
        if cache is None:
            cache = self._preview_encodings[agent] = TeamPreviewCache()
        with cuda_model_execution(self._device), self._device_execution_context():
            result = cache.predict(
                agent,
                self._device,
                observations,
                indices,
                include_value=False,
                greedy=self._greedy,
            )
            return result["picks"].cpu().numpy()

    def _get_agent(self, checkpoint_path: str) -> PokeAgent:
        if checkpoint_path in self._models:
            self._cache_hits += 1
            self._models.move_to_end(checkpoint_path)
            return self._models[checkpoint_path]
        self._cache_misses += 1
        agent = self._load_agent(checkpoint_path)
        self._models[checkpoint_path] = agent
        self._cache_loads += 1
        if len(self._models) > self._max_models:
            with cuda_model_execution(self._device):
                path, expired = self._models.popitem(last=False)
                self._evict_cuda_graph(path)
                del expired
                self._cache_evictions += 1
        return agent

    def _load_agent(self, checkpoint_path: str) -> PokeAgent:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        agent_copy = copy.deepcopy(self._agent_template)
        agent_copy.load_state_dict(checkpoint["agent"])
        with cuda_model_execution(self._device):
            agent_copy.to(self._device)
        agent_copy.eval()
        agent_copy.requires_grad_(False)
        return agent_copy

    def _build_hidden_batch(
        self,
        agent: PokeAgent,
        requests: list[_InferenceRow],
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

    @staticmethod
    def _slice_hidden(
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return a detached, read-only-by-convention view for one request.

        The inference stream is synchronized before these views are created,
        and the service never mutates a published hidden batch. Sharing the
        backing batch avoids two CUDA clones per request. Each lane retains its current opponent state until the next decision.
        """
        if hidden is None:
            return None
        h, c = hidden
        return (
            h[:, index : index + 1].detach(),
            c[:, index : index + 1].detach(),
        )


class League:
    """Permanent self-play archive with payoff-aware opponent sampling.

    Every snapshot remains eligible on disk. ``TrainingConfig.league_size``
    bounds only each inference service's resident model cache, preventing the
    permanent archive from causing unbounded CPU or CUDA memory growth.
    """

    def __init__(
        self,
        config: TrainingConfig,
        agent_template: PokeAgent,
        device: torch.device,
        *,
        rng: random.Random | None = None,
        inference_greedy: bool = False,
    ):
        self.config = config
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
        self._inference = _LeagueInferenceService(
            agent_template,
            self.inference_device,
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

    def vector_actions(self, paths, observations, masks, hidden):
        def forward(path, lanes):
            requests = [
                _InferenceRow(observations[i], masks[i], hidden[i]) for i in lanes
            ]
            return self._inference._run_battle_batch(path, requests)

        return self._inference.run_grouped(paths, forward)

    def vector_previews(self, paths, observations, indices):
        def forward(path, lanes):
            return self._preview_inference._run_preview_batch(
                path, observations, indices[lanes]
            )

        return np.stack(self._preview_inference.run_grouped(paths, forward))

    def add_checkpoint(
        self,
        agent: PokeAgent,
        step: int,
        *,
        update: int | None = None,
    ) -> None:
        """Snapshot the current agent as a permanent archive member."""
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
        torch.save(policy_weights(agent), target)
        self.checkpoints.append(path)
        self._members[path] = LeagueMember(
            path=path,
            step=step,
            created_update=member_update,
        )
        self._current_update = member_update

    def record_outcomes(
        self,
        counts: Mapping[str | None, Mapping[str, int]],
        *,
        update: int,
    ) -> None:
        """Update discounted current-policy payoffs before the next PFSP draw."""
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

    def state_dict(self) -> dict:
        return {
            "members": [asdict(self._members[path]) for path in self.checkpoints],
            "current_update": self._current_update,
            "rng_state": self._rng.getstate(),
            "sampling_config": {
                name: getattr(self.config, name)
                for name in (
                    "league_update_interval",
                    "league_refresh_interval",
                    "league_payoff_half_life_updates",
                    "league_sampler",
                    "league_pfsp_exponent",
                    "league_pfsp_uniform_fraction",
                )
            },
        }

    def load_state_dict(self, state: dict) -> None:
        members = [LeagueMember(**row) for row in state["members"]]
        for member in members:
            if not Path(member.path).is_file():
                raise FileNotFoundError(member.path)
        self._members = {member.path: member for member in members}
        self.checkpoints = list(self._members)
        self._current_update = state["current_update"]
        self._rng.setstate(state["rng_state"])
        for name, value in state["sampling_config"].items():
            setattr(self.config, name, value)

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
        if self.config.league_sampler == "balanced":
            members = [self._members[path] for path in available_paths]
            for member, expected in zip(
                members, _expected_group_counts(available_weights, count), strict=True
            ):
                member.assignment_balance += expected
            selected = sorted(
                members, key=lambda m: m.assignment_balance, reverse=True
            )[:count]
            for member in selected:
                member.assignment_balance -= 1
            while len(selected) < count:
                member = max(members, key=lambda m: m.assignment_balance)
                member.assignment_balance -= 1
                selected.append(member)
            self._rng.shuffle(selected)
            return [member.path for member in selected]

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


def _expected_group_counts(weights: list[float], count: int) -> list[float]:
    """Marginal counts for the league's successive weighted draws."""
    total = math.fsum(weights)
    probabilities = [weight / total for weight in weights]
    if count >= len(weights):
        return [1 + (count - len(weights)) * p for p in probabilities]
    if count == 1:
        return probabilities
    if count == 2:
        # Sum the weights left after the likeliest opponent explicitly. Its
        # probability can round to one while other opponents still have weight.
        peak = max(probabilities)
        others = probabilities.copy()
        others.remove(peak)
        remaining = math.fsum(others)
        base = 1 + math.fsum(p / (1 - p) for p in others)
        return [
            peak * base
            if p == peak
            else p * (base - p / (1 - p)) + peak * (p / remaining)
            for p in probabilities
        ]

    # A weighted draw orders independent exponential clocks. An opponent gets
    # a slot when at most count-1 of the other clocks have rung at its arrival.
    # Integrate on log time from the fastest clock's lower tail to the slowest's upper tail.
    rates = np.asarray(probabilities)
    positive = rates[rates > 0]
    eps = np.finfo(float).eps
    start = math.log(eps) - math.log(positive.max())
    stop = math.log(-math.log(eps)) - math.log(positive.min())
    edges = np.linspace(start, stop, math.ceil(stop - start) + 1)
    nodes, weights_dt = np.polynomial.legendre.leggauss(12)
    half = (edges[1] - edges[0]) / 2
    log_time = (edges[:-1, None] + half * (1 + nodes)).ravel()
    log_rates = np.log(rates, out=np.full_like(rates, -np.inf), where=rates > 0)
    log_hazard = log_rates[:, None] + log_time
    with np.errstate(over="ignore"):
        hazard = np.exp(log_hazard)
    early = -np.expm1(-hazard)
    late = np.exp(-hazard)
    density = np.exp(log_hazard - hazard)

    # Combine counts on either side of each candidate, excluding its own clock.
    before = np.zeros((len(rates) + 1, count, len(log_time)))
    before[0, 0] = 1
    for i in range(len(rates)):
        before[i + 1] = before[i] * late[i]
        before[i + 1, 1:] += before[i, :-1] * early[i]
    after = np.ones((count, len(log_time)))  # cumulative counts in the suffix
    expected = np.empty(len(rates))
    dt = np.tile(half * weights_dt, len(edges) - 1)
    for i in reversed(range(len(rates))):
        available = (before[i] * after[::-1]).sum(axis=0)
        expected[i] = (dt * density[i] * available).sum()
        after[1:] = after[1:] * late[i] + after[:-1] * early[i]
        after[0] *= late[i]
    return expected.tolist()
