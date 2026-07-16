"""Gymnasium-compatible wrapper around poke-env for doubles battles."""

from __future__ import annotations

import asyncio
import itertools
import math
import os
import sys
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from concurrent.futures import (
    CancelledError,
    Future,
    ThreadPoolExecutor,
    TimeoutError,
    as_completed,
)
from dataclasses import dataclass
from queue import Empty, Queue as ThreadQueue
from threading import BoundedSemaphore, Lock, Thread
from typing import Generic, TypeVar

import numpy as np

from poke_env import AccountConfiguration, LocalBattleStreamConfiguration
from poke_env.battle import DoubleBattle
from poke_env.concurrency import create_in_poke_loop
from poke_env.data.normalize import to_id_str
from poke_env.player import (
    BattleOrder,
    DoubleBattleOrder,
    ForfeitBattleOrder,
    PassBattleOrder,
    Player,
)

from ..config import EnvConfig
from ..data import GameData
from ..training.preview import TeamPreviewBatcher, TeamPreviewResult
from .observation import (
    ObservationBuilder,
    build_packed_team_aux_labels,
    build_team_preview_observation,
    summarize_observation_mismatch,
)
from .team_preview import parse_team_preview_order
from .actions import (
    INTENT_MOVE,
    INTENT_SWITCH,
    NUM_SLOT_INTENT_COMPONENTS,
    NUM_TERA_INTENT_COMPONENTS,
    SLOT_INTENT_KIND,
    SLOT_INTENT_MOVE,
    SLOT_INTENT_SWITCH_SPECIES,
    SLOT_INTENT_TARGET,
    SemanticActionIntent,
    SemanticActionSupervision,
    TERA_INTENT_SLOT,
    TERA_INTENT_USE,
    TERA_NONE,
    action_to_order,
    build_legal_action_state,
    build_policy_action_masks,
    decode_tera_choice,
    empty_action_masks,
    is_default_only_action_state,
    semantic_action_request_key,
    semantic_action_intent,
)
from .rewards import sparse_reward


_ACCOUNT_LOCK = Lock()
_ACCOUNT_COUNTER = itertools.count()
_ACCOUNT_RUN_TAG = uuid.uuid4().hex[:4]
QueueItemT = TypeVar("QueueItemT")
_OBSERVATION_POLL_SECONDS = 1.0
_DEFAULT_BATTLE_LOOP_COUNT = 4
_DEFAULT_RUNTIME_LOOP_COUNT = 2
_DEFAULT_SHARED_BATTLES_PER_WORKER = 3
_LOCAL_DISCONNECT_TIMEOUT_SECONDS = 30.0
OpponentFactory = Callable[[asyncio.AbstractEventLoop], Player]
TeamPairProvider = Callable[[], tuple[str | None, str | None]]


def _encode_opponent_intent_supervision(
    game_data: GameData,
    supervision: SemanticActionSupervision,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Encode trainer-only semantic intent labels into global vocab indices."""
    slot_labels = np.full(
        (2, NUM_SLOT_INTENT_COMPONENTS),
        fill_value=-1,
        dtype=np.int64,
    )
    slot_masks = np.zeros((2, NUM_SLOT_INTENT_COMPONENTS), dtype=bool)

    for slot, (intent, decision) in enumerate(
        zip(
            supervision.intent.slots,
            supervision.decision_mask.slots,
            strict=True,
        )
    ):
        slot_labels[slot, SLOT_INTENT_KIND] = intent.kind
        slot_masks[slot, SLOT_INTENT_KIND] = decision.kind
        if intent.kind == INTENT_MOVE:
            move_id = to_id_str(intent.move_id or "")
            move_index = game_data.move_to_idx.get(move_id)
            if move_index is None:
                raise RuntimeError(
                    f"Opponent intent move {move_id!r} is absent from GameData."
                )
            slot_labels[slot, SLOT_INTENT_MOVE] = move_index
            slot_labels[slot, SLOT_INTENT_TARGET] = intent.target
            slot_masks[slot, SLOT_INTENT_MOVE] = decision.move
            slot_masks[slot, SLOT_INTENT_TARGET] = decision.target
        elif intent.kind == INTENT_SWITCH:
            species = to_id_str(intent.switch_species or "")
            species_index = game_data.species_to_idx.get(species)
            if species_index is None:
                raise RuntimeError(
                    f"Opponent switch species {species!r} is absent from GameData."
                )
            slot_labels[slot, SLOT_INTENT_SWITCH_SPECIES] = species_index
            slot_masks[slot, SLOT_INTENT_SWITCH_SPECIES] = decision.switch_species

    tera_used_slots = [
        slot
        for slot, intent in enumerate(supervision.intent.slots)
        if intent.terastallize
    ]
    if len(tera_used_slots) > 1:
        raise RuntimeError("A semantic opponent intent terastallized both slots.")
    tera_labels = np.full(NUM_TERA_INTENT_COMPONENTS, -1, dtype=np.int64)
    tera_labels[TERA_INTENT_USE] = int(bool(tera_used_slots))
    if tera_used_slots:
        tera_labels[TERA_INTENT_SLOT] = tera_used_slots[0]
    tera_masks = np.array(
        [
            supervision.decision_mask.tera_use,
            supervision.decision_mask.tera_slot,
        ],
        dtype=bool,
    )
    return slot_labels, slot_masks, tera_labels, tera_masks


def showdown_log_level() -> int | None:
    value = os.environ.get("POKERL_SHOWDOWN_LOG_LEVEL")
    if value is None or value == "":
        return None
    return int(value)


def default_showdown_dir() -> str | None:
    env_path = os.environ.get("POKERL_SHOWDOWN_DIR")
    if env_path:
        return env_path
    local_app_data = os.environ.get("LOCALAPPDATA")
    if not local_app_data:
        return None
    candidate = os.path.join(local_app_data, "pokeRL", "pokemon-showdown")
    return candidate if os.path.exists(candidate) else None


class _TimingStats:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._lock = Lock()
        self._values: dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def increment(self, key: str, delta: int = 1) -> None:
        if not self.enabled:
            return
        self.add(key, float(delta))

    def snapshot_and_reset(self) -> dict[str, float]:
        if not self.enabled:
            return {}
        with self._lock:
            snapshot = dict(self._values)
            self._values.clear()
        return snapshot

    def reset(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._values.clear()


def _timing_enabled(timing) -> bool:
    return bool(getattr(timing, "enabled", True))


class LoopThread:
    def __init__(self, loop: asyncio.AbstractEventLoop, thread: Thread):
        self.loop = loop
        self.thread = thread
        self._closed = False

    @classmethod
    def start(cls) -> "LoopThread":
        def run_loop(loop: asyncio.AbstractEventLoop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.new_event_loop()
        thread = Thread(target=run_loop, args=(loop,), daemon=True)
        thread.start()
        return cls(loop, thread)

    def close(self) -> None:
        if self._closed:
            return

        async def _cancel_pending() -> None:
            current_task = asyncio.current_task()
            pending_tasks = [
                task for task in asyncio.all_tasks() if task is not current_task
            ]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await asyncio.get_running_loop().shutdown_asyncgens()

        try:
            asyncio.run_coroutine_threadsafe(_cancel_pending(), self.loop).result(
                timeout=2.0
            )
        except (CancelledError, TimeoutError):
            pass

        self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if not self.loop.is_running():
            self.loop.close()
        self._closed = True


class _AsyncQueue(Generic[QueueItemT]):
    def __init__(
        self,
        loop: asyncio.AbstractEventLoop,
        maxsize: int = 0,
        *,
        sync_get: bool = False,
        timing: _TimingStats | None = None,
        key_prefix: str | None = None,
    ):
        self._record_timing = (
            timing is not None
            and key_prefix is not None
            and bool(getattr(timing, "enabled", True))
        )
        self._sync_queue: ThreadQueue[QueueItemT | tuple[float, QueueItemT]] | None = (
            ThreadQueue(maxsize=maxsize) if sync_get else None
        )
        self.queue: asyncio.Queue[QueueItemT | tuple[float, QueueItemT]] | None = (
            None if sync_get else create_in_poke_loop(asyncio.Queue, loop, maxsize)
        )
        self._loop = loop
        self._timing = timing
        self._key_prefix = key_prefix

    async def async_get(self) -> QueueItemT:
        if self.queue is None:
            raise RuntimeError("This queue is configured for synchronous reads only.")
        if not self._record_timing:
            return await self.queue.get()

        started = time.perf_counter()
        try:
            queued_at, item = await self.queue.get()
        except asyncio.CancelledError:
            self._record_get_cancelled(started)
            raise
        self._record_get_success(started, queued_at)
        return item

    def get(self, timeout: float | None = None) -> QueueItemT:
        started = 0.0
        if self._record_timing:
            started = time.perf_counter()
        if self._sync_queue is not None:
            try:
                payload = self._sync_queue.get(timeout=timeout)
            except Empty as exc:
                if self._record_timing:
                    self._record_get_timeout(started)
                raise TimeoutError from exc
            if not self._record_timing:
                return payload
            queued_at, item = payload
            self._record_get_success(started, queued_at)
            return item
        coro = (
            self.async_get()
            if timeout is None
            else asyncio.wait_for(self.async_get(), timeout)
        )
        result = asyncio.run_coroutine_threadsafe(coro, self._loop)
        if timeout is None:
            return result.result()
        try:
            return result.result(timeout=timeout + 0.5)
        except TimeoutError:
            result.cancel()
            if self._record_timing:
                self._record_get_timeout(started)
            raise

    async def async_put(self, item: QueueItemT) -> None:
        if self.queue is None:
            raise RuntimeError("This queue is configured for synchronous reads only.")
        if not self._record_timing:
            await self.queue.put(item)
            return

        started = time.perf_counter()
        await self.queue.put((started, item))
        self._record_put(started, from_thread=False)

    def qsize(self) -> int:
        if self._sync_queue is not None:
            return self._sync_queue.qsize()
        assert self.queue is not None
        return self.queue.qsize()

    def put(self, item: QueueItemT) -> None:
        if not self._record_timing:
            if self._sync_queue is not None:
                self._sync_queue.put_nowait(item)
                return
            try:
                if asyncio.get_running_loop() is self._loop:
                    assert self.queue is not None
                    self.queue.put_nowait(item)
                    return
            except RuntimeError:
                pass
            assert self.queue is not None
            self._loop.call_soon_threadsafe(self.queue.put_nowait, item)
            return

        started = time.perf_counter()
        if self._sync_queue is not None:
            self._sync_queue.put_nowait((started, item))
            self._record_put(started, from_thread=False)
            return
        try:
            if asyncio.get_running_loop() is self._loop:
                self._put_nowait_from_loop(started, item, from_thread=False)
                return
        except RuntimeError:
            pass
        assert self.queue is not None
        self._loop.call_soon_threadsafe(
            self._put_nowait_from_loop,
            started,
            item,
            True,
        )
        self._record_increment("threadsafe_put_calls")

    def clear(self) -> None:
        if self._sync_queue is not None:
            while True:
                try:
                    self._sync_queue.get_nowait()
                except Empty:
                    return

        async def _clear() -> None:
            assert self.queue is not None
            while True:
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

        asyncio.run_coroutine_threadsafe(_clear(), self._loop).result()

    def _put_nowait_from_loop(
        self,
        started: float,
        item: QueueItemT,
        from_thread: bool,
    ) -> None:
        assert self.queue is not None
        self.queue.put_nowait((started, item))
        self._record_put(started, from_thread=from_thread)
        if from_thread:
            self._record_add("loop_put_delay_sec", time.perf_counter() - started)

    def _record_put(self, started: float, *, from_thread: bool) -> None:
        self._record_increment("put_calls")
        self._record_add("put_sec", time.perf_counter() - started)
        if from_thread:
            self._record_increment("threadsafe_put_completed_calls")

    def _record_get_success(self, started: float, queued_at: float) -> None:
        now = time.perf_counter()
        self._record_increment("get_calls")
        self._record_add("get_wait_sec", now - started)
        self._record_add("item_age_sec", now - queued_at)

    def _record_get_timeout(self, started: float) -> None:
        self._record_increment("get_timeouts")
        self._record_add("get_timeout_wait_sec", time.perf_counter() - started)

    def _record_get_cancelled(self, started: float) -> None:
        self._record_increment("get_cancelled")
        self._record_add("get_cancelled_wait_sec", time.perf_counter() - started)

    def _record_increment(self, suffix: str, delta: int = 1) -> None:
        if self._timing is None or self._key_prefix is None:
            return
        self._timing.increment(f"{self._key_prefix}_{suffix}", delta)

    def _record_add(self, suffix: str, value: float) -> None:
        if self._timing is None or self._key_prefix is None:
            return
        self._timing.add(f"{self._key_prefix}_{suffix}", value)


@dataclass(frozen=True)
class _PlayerObservation:
    request_id: int | None
    obs: dict[str, np.ndarray]
    mask: dict[str, np.ndarray]
    reward: float
    done: bool
    battle: DoubleBattle | None

    def __iter__(self):
        yield self.obs
        yield self.mask
        yield self.reward
        yield self.done
        yield self.battle


class _NoopBattleOrder(BattleOrder):
    @property
    def message(self) -> str:
        return ""


def make_local_account_configuration(prefix: str) -> AccountConfiguration:
    """Create a short, unique local showdown username for this process."""
    clean_prefix = "".join(ch for ch in prefix.lower() if ch.isalnum())[:10] or "bot"
    with _ACCOUNT_LOCK:
        suffix = f"{_ACCOUNT_RUN_TAG}{next(_ACCOUNT_COUNTER):04x}"
    return AccountConfiguration(f"{clean_prefix}{suffix}"[:18], None)


def _worker_count(config: EnvConfig) -> int:
    if config.worker_count is not None:
        return config.worker_count
    return max(1, math.ceil(config.num_envs / _DEFAULT_SHARED_BATTLES_PER_WORKER))


def _battles_per_worker(config: EnvConfig, worker_count: int) -> int:
    if config.battles_per_worker is not None:
        return config.battles_per_worker
    if config.worker_count is None:
        return _DEFAULT_SHARED_BATTLES_PER_WORKER
    return max(1, math.ceil(config.num_envs / worker_count))


def _runtime_loop_count(config: EnvConfig, worker_count: int) -> int:
    if config.runtime_loop_count is not None:
        return config.runtime_loop_count
    if worker_count <= 1:
        return 0
    return min(_DEFAULT_RUNTIME_LOOP_COUNT, worker_count)


def resolve_battle_loop_count(configured: int | None, num_envs: int) -> int:
    """Resolve the number of event-loop threads hosting battle sessions."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if configured is None:
        return min(num_envs, _DEFAULT_BATTLE_LOOP_COUNT)
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured <= 0
    ):
        raise ValueError("battle_loop_count must be a positive integer.")
    return configured


def resolve_reset_concurrency(configured: int | None, num_envs: int) -> int:
    """Resolve the maximum number of concurrent environment resets."""
    if num_envs <= 0:
        raise ValueError("num_envs must be positive.")
    if configured is None:
        return num_envs
    if (
        isinstance(configured, bool)
        or not isinstance(configured, int)
        or configured <= 0
    ):
        raise ValueError("reset_concurrency must be a positive integer.")
    return min(configured, num_envs)


def battlestream_summary(config: EnvConfig) -> dict[str, object]:
    worker_count = _worker_count(config)
    return {
        "showdown_dir": config.showdown_dir,
        "worker_count": worker_count,
        "max_battles_per_worker": _battles_per_worker(config, worker_count),
        "runtime_loop_count": _runtime_loop_count(config, worker_count),
        "battle_loop_count": resolve_battle_loop_count(
            config.battle_loop_count,
            config.num_envs,
        ),
        "startup_timeout": config.battlestream_startup_timeout,
        "reset_concurrency": resolve_reset_concurrency(
            config.reset_concurrency,
            config.num_envs,
        ),
        "lazy_action_orders": config.lazy_action_orders,
    }


def make_battlestream_configuration(
    config: EnvConfig,
) -> LocalBattleStreamConfiguration:
    if not config.showdown_dir:
        raise ValueError(
            "EnvConfig.showdown_dir must be set for local BattleStream training."
        )

    worker_count = _worker_count(config)
    return LocalBattleStreamConfiguration(
        showdown_dir=config.showdown_dir,
        worker_count=worker_count,
        max_battles_per_worker=_battles_per_worker(config, worker_count),
        runtime_loop_count=_runtime_loop_count(config, worker_count),
        startup_timeout=config.battlestream_startup_timeout,
    )


class ShowdownPlayer(Player):
    """A poke-env player that bridges to our RL agent via queues.

    Instead of choosing moves itself, it sends observations to a queue
    and waits for actions from another queue. This lets us use poke-env's
    async battle loop from a synchronous Gymnasium-style step() function.
    """

    def __init__(
        self,
        obs_builder: ObservationBuilder,
        timing: _TimingStats,
        *,
        lazy_action_orders: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.obs_builder = obs_builder
        self.timing = timing
        self.lazy_action_orders = lazy_action_orders
        self.obs_queue: _AsyncQueue[_PlayerObservation] = _AsyncQueue(
            self.ps_client.loop,
            sync_get=True,
            timing=timing,
            key_prefix="player_obs_queue",
        )
        self.action_queue: _AsyncQueue[np.ndarray | None] = _AsyncQueue(
            self.ps_client.loop,
            timing=timing,
            key_prefix="player_action_queue",
        )
        self.last_battle: DoubleBattle | None = None
        self._last_finished_token: tuple[str, int] | None = None
        self._closing = False
        self._team_preview_fn = None
        self._pending_action_state = None
        self._pending_action_battle: DoubleBattle | None = None
        self._pending_action_requested_at: float | None = None
        self._next_action_request_id = 1
        self._pending_action_request_id: int | None = None
        self._pending_action_future: asyncio.Future[np.ndarray | None] | None = None
        self._pending_finished_battle: DoubleBattle | None = None
        self._previous_intent: SemanticActionIntent | None = None

    def choose_move(self, battle: DoubleBattle) -> Awaitable[BattleOrder]:
        return self._choose_move(battle)

    async def _choose_move(self, battle: DoubleBattle) -> BattleOrder:
        if battle.finished:
            self._pending_finished_battle = battle
            return self.choose_default_move()

        if self._closing:
            return ForfeitBattleOrder()

        action_state = (
            build_legal_action_state(battle)
            if getattr(self, "lazy_action_orders", True)
            else build_legal_action_state(battle, lazy_action_orders=False)
        )
        if is_default_only_action_state(action_state):
            return self.choose_default_move()

        request_id, action_future = self._begin_action_request()
        self._emit_battle_update(
            battle, request_id=request_id, action_state=action_state
        )

        try:
            action = await action_future
        finally:
            if self._pending_action_future is action_future:
                self._pending_action_future = None
                self._pending_action_request_id = None

        if action is None:
            if battle.finished:
                return _NoopBattleOrder()
            if self._closing:
                return ForfeitBattleOrder()
            raise RuntimeError(
                "ShowdownPlayer received an empty action outside env.close()."
            )
        return self._make_order(action, battle)

    def _begin_action_request(self) -> tuple[int, asyncio.Future[np.ndarray | None]]:
        pending = self._pending_action_future
        if pending is not None and not pending.done():
            raise RuntimeError(
                "ShowdownPlayer received a new action request while request "
                f"{self._pending_action_request_id} is still waiting for an action."
            )

        request_id = self._next_action_request_id
        self._next_action_request_id += 1
        future: asyncio.Future[np.ndarray | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_action_request_id = request_id
        self._pending_action_future = future
        return request_id, future

    def submit_action(
        self,
        request_id: int | None,
        action: np.ndarray,
        *,
        timeout: float | None = None,
    ) -> None:
        if request_id is None:
            raise RuntimeError(
                "BattleEnv tried to submit an action without a pending request id."
            )

        future: Future = Future()
        action_copy = np.array(action, copy=True)

        def submit_on_loop() -> None:
            try:
                self._submit_action_nowait(request_id, action_copy)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(None)

        self.ps_client.loop.call_soon_threadsafe(submit_on_loop)
        future.result(timeout=timeout)

    async def _submit_action(self, request_id: int, action: np.ndarray) -> None:
        ShowdownPlayer._submit_action_nowait(self, request_id, action)

    def _submit_action_nowait(self, request_id: int, action: np.ndarray) -> None:
        pending = self._pending_action_future
        pending_id = self._pending_action_request_id
        if pending is None or pending.done():
            raise RuntimeError(
                "BattleEnv submitted an action for request "
                f"{request_id}, but ShowdownPlayer has no pending action request."
            )
        if pending_id != request_id:
            raise RuntimeError(
                "BattleEnv submitted an action for request "
                f"{request_id}, but ShowdownPlayer is waiting for request {pending_id}."
            )
        pending.set_result(action)

    def _battle_finished_callback(self, battle):
        self._pending_finished_battle = battle
        ShowdownPlayer._resolve_pending_action_for_finished_battle(self, battle)

    def _resolve_pending_action_for_finished_battle(self, battle: DoubleBattle) -> None:
        pending = getattr(self, "_pending_action_future", None)
        if pending is None or pending.done():
            return
        if getattr(self, "_pending_action_battle", None) is not battle:
            return
        pending.set_result(None)

    def emit_pending_finished_battle(self) -> bool:
        battle = self._pending_finished_battle
        if (
            battle is None
            and self.last_battle is not None
            and self.last_battle.finished
        ):
            battle = self.last_battle
        if battle is None:
            return False
        self._pending_finished_battle = None
        self._emit_battle_update(battle, request_id=None)
        return True

    def teampreview(self, battle: DoubleBattle) -> Awaitable[str]:
        if self._team_preview_fn is None:
            raise RuntimeError(
                "ShowdownPlayer.teampreview() called but no team_preview_fn was set. "
                "Pass team_preview_fn to BattleEnv so the agent learns team selection."
            )
        return self._teampreview(battle)

    async def _teampreview(self, battle: DoubleBattle) -> str:
        assert self._team_preview_fn is not None
        result = self._team_preview_fn(battle)
        if isinstance(result, Awaitable):
            return await result
        return result

    def set_team_preview_fn(self, fn):
        self._team_preview_fn = fn

    def request_close(self) -> None:
        self._closing = True
        pending = self._pending_action_future
        if pending is None or pending.done():
            return

        def resolve_pending_close() -> None:
            if self._pending_action_future is pending and not pending.done():
                pending.set_result(None)

        try:
            self.ps_client.loop.call_soon_threadsafe(resolve_pending_close)
        except RuntimeError:
            return

    def reset_transition_history(self) -> None:
        """Clear semantic action history at an episode boundary."""
        self._previous_intent = None

    def _make_order(self, action: np.ndarray, battle: DoubleBattle) -> BattleOrder:
        """Convert a (3,) action array to a DoubleBattleOrder."""
        timing_enabled = _timing_enabled(self.timing)
        if timing_enabled:
            started = time.perf_counter()
        action_state = (
            self._pending_action_state
            if self._pending_action_battle is battle
            else None
        )
        self._pending_action_state = None
        self._pending_action_battle = None
        self._pending_action_requested_at = None
        if action_state is None:
            action_state = (
                build_legal_action_state(battle)
                if getattr(self, "lazy_action_orders", True)
                else build_legal_action_state(battle, lazy_action_orders=False)
            )
        resolved_intent = semantic_action_intent(action, action_state)
        orders = []
        tera_slot_1, tera_slot_2 = decode_tera_choice(int(action[2]))
        for slot in range(2):
            order = action_to_order(
                int(action[slot]),
                battle,
                slot,
                terastallize=tera_slot_1 if slot == 0 else tera_slot_2,
                action_state=action_state,
            )
            orders.append(order)
        self._previous_intent = resolved_intent

        if timing_enabled:
            self.timing.increment("player_make_order_calls")
            self.timing.add("player_make_order_sec", time.perf_counter() - started)
        if all(isinstance(order, PassBattleOrder) for order in orders):
            return self.choose_default_move()
        return DoubleBattleOrder(orders[0], orders[1])

    def _emit_battle_update(
        self,
        battle: DoubleBattle,
        *,
        request_id: int | None = None,
        action_state=None,
    ) -> None:
        self.last_battle = battle
        finished_token = (battle.battle_tag, battle.turn)
        if battle.finished and finished_token == self._last_finished_token:
            return

        timing_enabled = _timing_enabled(self.timing)
        if timing_enabled:
            emit_started = time.perf_counter()
        obs = self.obs_builder.build(
            battle,
            previous_intent=getattr(self, "_previous_intent", None),
        )
        self._previous_intent = None
        if timing_enabled:
            self.timing.add("player_emit_obs_sec", time.perf_counter() - emit_started)

        if timing_enabled:
            mask_started = time.perf_counter()
        if battle.finished:
            mask = empty_action_masks()
            self._pending_action_state = None
            self._pending_action_battle = None
            self._pending_action_requested_at = None
        else:
            if request_id is None:
                raise RuntimeError(
                    "ShowdownPlayer emitted a non-terminal observation without an action request id."
                )
            if action_state is None:
                action_state = build_legal_action_state(battle)
            mask = build_policy_action_masks(battle, action_state=action_state)
            self._pending_action_state = action_state
            self._pending_action_battle = battle
        if timing_enabled:
            self.timing.add("player_emit_mask_sec", time.perf_counter() - mask_started)

        if timing_enabled:
            reward_started = time.perf_counter()
        reward = sparse_reward(battle)
        if timing_enabled:
            self.timing.add(
                "player_emit_reward_sec", time.perf_counter() - reward_started
            )
        done = battle.finished

        if timing_enabled:
            queue_put_started = time.perf_counter()
        self.obs_queue.put(
            _PlayerObservation(
                request_id=request_id,
                obs=obs,
                mask=mask,
                reward=reward,
                done=done,
                battle=battle,
            )
        )
        if not done:
            self._pending_action_requested_at = time.perf_counter()
        if timing_enabled:
            self.timing.add(
                "player_emit_queue_put_sec", time.perf_counter() - queue_put_started
            )
            self.timing.increment("player_emit_calls")
            self.timing.add("player_emit_total_sec", time.perf_counter() - emit_started)
        if done:
            self._last_finished_token = finished_token


class BattleEnv:
    """Synchronous environment wrapping one concurrent battle.

    Usage:
        env = BattleEnv(config, game_data, opponent_factory)
        obs, mask = env.reset()
        while not done:
            obs, mask, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        config: EnvConfig,
        game_data: GameData,
        opponent_factory: OpponentFactory,
        team_preview_fn: Callable[[dict], tuple[np.ndarray | TeamPreviewResult, str]]
        | None = None,
        team_preview_batcher: TeamPreviewBatcher | None = None,
        team_pair_provider: TeamPairProvider | None = None,
        player_kwargs: dict | None = None,
        loop_thread: LoopThread | None = None,
        collect_timing: bool = False,
    ):
        self.config = config
        self.game_data = game_data
        self._battle_task = None
        self._owns_loop_thread = loop_thread is None
        self._loop_thread = (
            loop_thread if loop_thread is not None else LoopThread.start()
        )
        self._loop = self._loop_thread.loop
        self._pending_opponent: Player | None = None
        self._closed = False
        self._battle_started_at: float | None = None
        self._timing = _TimingStats(enabled=collect_timing)
        self._ext_team_preview_fn = team_preview_fn
        self._team_preview_batcher = team_preview_batcher
        self._team_pair_provider = team_pair_provider
        self._prepared_player_team: str | None = None
        self._prepared_opponent_team: str | None = None
        self._prepared_preview_obs: dict[str, np.ndarray] | None = None
        self._prepared_preview_future: Future | None = None
        self._active_preview_obs: dict[str, np.ndarray] | None = None
        self._active_preview_future: Future | None = None
        # Set at team preview time, read when the episode ends to attach to info.
        self._episode_team_preview: tuple[dict, np.ndarray, int | None] | None = None
        self._episode_opponent_preview_picks: np.ndarray | None = None
        self._episode_opponent_packed_team: str | None = None
        self._episode_opponent_choice_calls = 0
        self._episode_opponent_turn_calls: dict[tuple[str, int], int] = defaultdict(int)
        self._episode_opponent_checkpoint_path: str | None = None
        self._episode_first_tera_turn: int | None = None
        self._last_player_action: np.ndarray | None = None
        self._pending_player_request_id: int | None = None

        try:
            player_options = dict(player_kwargs or {})
            reserved_player_keys = {"battle_format", "loop", "server_configuration"}
            unexpected_keys = reserved_player_keys.intersection(player_options)
            if unexpected_keys:
                bad_keys = ", ".join(sorted(unexpected_keys))
                raise ValueError(
                    f"BattleEnv manages {bad_keys}; do not pass them via player_kwargs."
                )
            obs_builder = ObservationBuilder(game_data, config.team_size)
            player_options.setdefault(
                "account_configuration",
                make_local_account_configuration("agent"),
            )
            log_level = showdown_log_level()
            if log_level is not None:
                player_options.setdefault("log_level", log_level)
            self.player = ShowdownPlayer(
                obs_builder,
                timing=self._timing,
                lazy_action_orders=config.lazy_action_orders,
                battle_format=config.battle_format,
                server_configuration=make_battlestream_configuration(config),
                loop=self._loop,
                **player_options,
            )
            self._assert_player_loop(self.player, "player")
            if team_preview_fn is not None:
                self.player.set_team_preview_fn(self._do_team_preview)
            self.opponent = self._build_opponent(opponent_factory)
            self._prepare_next_battle()
        except BaseException as construction_error:
            for role in ("player", "opponent"):
                participant = getattr(self, role, None)
                if participant is not None:
                    self._disconnect_player(participant, self._loop)
            if self._owns_loop_thread:
                try:
                    self._loop_thread.close()
                except BaseException as cleanup_error:
                    construction_error.add_note(
                        f"BattleEnv loop cleanup also failed: {cleanup_error!r}"
                    )
            self._closed = True
            raise

    def reset(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Start a new battle and return the first observation + mask."""
        if self._closed:
            raise RuntimeError(
                "BattleEnv.reset() called after the environment was closed."
            )

        timing_enabled = self._timing.enabled
        if timing_enabled:
            reset_started = time.perf_counter()
        self._wait_for_battle_task_completion("before BattleEnv reset")
        self._pending_player_request_id = None
        self.player.reset_transition_history()
        self._episode_team_preview = None
        self._episode_opponent_preview_picks = None
        self._episode_opponent_packed_team = None
        self._episode_opponent_choice_calls = 0
        self._episode_opponent_turn_calls.clear()
        self._episode_first_tera_turn = None
        self._episode_opponent_checkpoint_path = getattr(
            self.opponent, "checkpoint_path", None
        )
        if timing_enabled:
            apply_pending_started = time.perf_counter()
        self._apply_pending_opponent()
        if timing_enabled:
            self._timing.add(
                "env_reset_apply_pending_sec",
                time.perf_counter() - apply_pending_started,
            )
        self._episode_opponent_checkpoint_path = getattr(
            self.opponent, "checkpoint_path", None
        )

        if timing_enabled:
            assign_teams_started = time.perf_counter()
        self._assign_teams_for_next_battle()
        if timing_enabled:
            self._timing.add(
                "env_reset_team_assign_sec", time.perf_counter() - assign_teams_started
            )

        # Give stateful opponents (e.g. LeaguePlayer) a clean slate for the new battle.
        if timing_enabled:
            hidden_reset_started = time.perf_counter()
        if hasattr(self.opponent, "reset_hidden"):
            self.opponent.reset_hidden()
        if timing_enabled:
            self._timing.add(
                "env_reset_hidden_reset_sec", time.perf_counter() - hidden_reset_started
            )

        if timing_enabled:
            battle_start_started = time.perf_counter()
        self._battle_task = asyncio.run_coroutine_threadsafe(
            self._run_battle(), self._loop
        )
        if timing_enabled:
            self._timing.add(
                "env_reset_battle_start_sec", time.perf_counter() - battle_start_started
            )
        self._battle_started_at = time.perf_counter()

        if timing_enabled:
            wait_started = time.perf_counter()
        player_observation = self._get_player_observation(
            "initial observation",
            max_wait_sec=self.config.battlestream_startup_timeout,
        )
        self._pending_player_request_id = player_observation.request_id
        if timing_enabled:
            self._timing.add(
                "env_reset_wait_first_obs_sec", time.perf_counter() - wait_started
            )
        self._prepare_next_battle()
        if timing_enabled:
            self._timing.increment("env_reset_calls")
            self._timing.add("env_reset_total_sec", time.perf_counter() - reset_started)
        return player_observation.obs, player_observation.mask

    async def _run_battle(self) -> None:
        await self.player.battle_against(self.opponent, n_battles=1)
        self.player.emit_pending_finished_battle()

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, dict[str, np.ndarray], float, bool, dict]:
        """Send an action and get the next observation.

        Args:
            action: (3,) int array of [slot_1_action, slot_2_action, tera_choice].

        Returns:
            obs, mask, reward, done, info
        """
        if self._closed:
            raise RuntimeError(
                "BattleEnv.step() called after the environment was closed."
            )

        timing_enabled = self._timing.enabled
        if timing_enabled:
            step_started = time.perf_counter()
            enqueue_started = step_started
        self._last_player_action = np.array(action, copy=True)
        if self._episode_first_tera_turn is None and int(action[2]) != TERA_NONE:
            self._episode_first_tera_turn = int(
                getattr(getattr(self.player, "last_battle", None), "turn", 0)
            )
        consume_opponent_intent = getattr(
            self.opponent,
            "consume_intent_supervision",
            None,
        )
        opponent_intent_request_key = None
        if callable(consume_opponent_intent):
            pending_battle = getattr(self.player, "_pending_action_battle", None)
            if pending_battle is None:
                raise RuntimeError(
                    "BattleEnv has no pending battle for opponent intent alignment."
                )
            opponent_intent_request_key = semantic_action_request_key(pending_battle)
        action_requested_at = getattr(self.player, "_pending_action_requested_at", None)
        if timing_enabled and action_requested_at is not None:
            self._timing.increment("env_agent_action_latency_calls")
            self._timing.add(
                "env_agent_action_latency_sec",
                time.perf_counter() - action_requested_at,
            )
        self._submit_player_action(action)
        if timing_enabled:
            self._timing.add(
                "env_action_enqueue_sec",
                time.perf_counter() - enqueue_started,
            )

        if timing_enabled:
            wait_started = time.perf_counter()
        player_observation = self._get_player_observation(
            "step observation",
            max_wait_sec=self.config.battlestream_startup_timeout,
        )
        self._pending_player_request_id = player_observation.request_id
        if timing_enabled:
            self._timing.add("env_queue_wait_sec", time.perf_counter() - wait_started)
            self._timing.increment("env_step_calls")
            self._timing.add(
                "env_step_total_sec",
                time.perf_counter() - step_started,
            )
        obs = player_observation.obs
        mask = player_observation.mask
        reward = player_observation.reward
        done = player_observation.done
        battle = player_observation.battle
        info = {
            "battle_tag": battle.battle_tag if battle else "",
            "battle_turn": getattr(battle, "turn", 0),
        }

        # Semantic opponent intent is trainer-only supervision. It is attached
        # to transition info after resolution and never enters observations or
        # league inference. Non-league opponents leave it unavailable.
        opponent_intent = None
        if callable(consume_opponent_intent):
            if opponent_intent_request_key is None:
                raise RuntimeError(
                    "BattleEnv omitted the opponent intent request identity."
                )
            request_resolved = done
            if not request_resolved:
                if battle is None:
                    raise RuntimeError(
                        "BattleEnv received a non-terminal observation without a battle."
                    )
                request_resolved = (
                    semantic_action_request_key(battle) != opponent_intent_request_key
                )
            if request_resolved:
                opponent_intent = consume_opponent_intent(opponent_intent_request_key)
        if opponent_intent is not None:
            (
                info["opponent_intent_slot_labels"],
                info["opponent_intent_slot_masks"],
                info["opponent_intent_tera_labels"],
                info["opponent_intent_tera_masks"],
            ) = _encode_opponent_intent_supervision(
                self.game_data,
                opponent_intent,
            )
        if done:
            if self._battle_started_at is not None:
                info["episode_sec"] = time.perf_counter() - self._battle_started_at
            if self._episode_team_preview is not None:
                obs_tp, picks_tp, policy_version = self._episode_team_preview
                info["team_preview_obs"] = obs_tp
                info["team_preview_picks"] = picks_tp
                info["team_preview_policy_version"] = policy_version
            if self._episode_opponent_preview_picks is not None:
                info["opponent_team_preview_picks"] = (
                    self._episode_opponent_preview_picks
                )
                opponent_packed_team = getattr(
                    self,
                    "_episode_opponent_packed_team",
                    None,
                )
                if opponent_packed_team is not None:
                    info["opponent_aux_labels"] = build_packed_team_aux_labels(
                        self.game_data,
                        self.config.team_size,
                        opponent_packed_team,
                        selected_indices=self._episode_opponent_preview_picks,
                    )
            info["episode_opponent_choice_calls"] = self._episode_opponent_choice_calls
            info["episode_opponent_repeated_turn_requests"] = sum(
                count - 1
                for count in self._episode_opponent_turn_calls.values()
                if count > 1
            )
            info["episode_opponent_max_calls_same_turn"] = max(
                self._episode_opponent_turn_calls.values(),
                default=0,
            )
            info["episode_first_tera_turn"] = self._episode_first_tera_turn
            if self._episode_opponent_checkpoint_path is not None:
                info["episode_opponent_checkpoint_path"] = (
                    self._episode_opponent_checkpoint_path
                )
            self._wait_for_battle_task_completion(
                "after terminal BattleEnv observation"
            )
        return obs, mask, reward, done, info

    def _submit_player_action(self, action: np.ndarray) -> None:
        submit_action = getattr(self.player, "submit_action", None)
        if callable(submit_action):
            request_id = self._pending_player_request_id
            try:
                submit_action(
                    request_id,
                    action,
                    timeout=self.config.battlestream_startup_timeout,
                )
            except TimeoutError as exc:
                diagnostics = self._describe_pending_battle_tasks()
                raise TimeoutError(
                    "Timed out waiting "
                    f"{self.config.battlestream_startup_timeout:.1f}s for "
                    "BattleEnv agent action submission to reach the battle loop. "
                    f"request_id={request_id} "
                    f"action={np.asarray(action).tolist()} | {diagnostics}"
                ) from exc
            self._pending_player_request_id = None
            return

        self.player.action_queue.put(action)
        self._pending_player_request_id = None

    def _get_player_observation(
        self,
        context: str,
        *,
        max_wait_sec: float | None = None,
    ) -> _PlayerObservation:
        started = time.perf_counter()
        while True:
            try:
                return self._coerce_player_observation(
                    self.player.obs_queue.get(timeout=_OBSERVATION_POLL_SECONDS)
                )
            except TimeoutError as exc:
                self._raise_battle_task_failure(context, exc)
                if (
                    max_wait_sec is not None
                    and time.perf_counter() - started >= max_wait_sec
                ):
                    diagnostics = self._describe_pending_battle_tasks()
                    raise TimeoutError(
                        f"Timed out waiting {max_wait_sec:.1f}s for BattleEnv "
                        f"{context} while the battle task was still pending. "
                        f"{diagnostics}"
                    ) from exc

    @staticmethod
    def _coerce_player_observation(item) -> _PlayerObservation:
        if isinstance(item, _PlayerObservation):
            return item
        obs, mask, reward, done, battle = item
        return _PlayerObservation(
            request_id=None,
            obs=obs,
            mask=mask,
            reward=reward,
            done=done,
            battle=battle,
        )

    def _raise_battle_task_failure(
        self,
        context: str,
        timeout_error: TimeoutError,
    ) -> None:
        task = self._battle_task
        if task is None or not task.done():
            return
        try:
            task.result()
        except Exception as exc:
            raise RuntimeError(
                f"Battle task failed while waiting for BattleEnv {context}."
            ) from exc
        raise RuntimeError(
            f"Battle task completed without producing BattleEnv {context}."
        ) from timeout_error

    def _describe_current_battle_state(self) -> str:
        battle = getattr(getattr(self, "player", None), "last_battle", None)
        if battle is None:
            return "battle=None"

        def species_list(values) -> list[str | None]:
            return [
                pokemon.species if pokemon is not None else None for pokemon in values
            ]

        def move_request_summary() -> list[list[dict[str, object]]]:
            last_request = getattr(battle, "last_request", {}) or {}
            active_requests = last_request.get("active") or []
            summary: list[list[dict[str, object]]] = []
            for active_request in active_requests:
                moves = []
                for move_request in active_request.get("moves", []):
                    moves.append(
                        {
                            "id": move_request.get("id"),
                            "target": move_request.get("target"),
                            "disabled": move_request.get("disabled", False),
                        }
                    )
                summary.append(moves)
            return summary

        def valid_order_summary() -> list[list[str]]:
            valid_orders = getattr(battle, "valid_orders", [])
            result: list[list[str]] = []
            for slot_orders in valid_orders[:2]:
                result.append([str(order) for order in slot_orders[:12]])
            return result

        last_player_action = (
            self._last_player_action.tolist()
            if self._last_player_action is not None
            else None
        )
        return (
            f"battle_tag={getattr(battle, 'battle_tag', None)} "
            f"turn={getattr(battle, 'turn', None)} "
            f"wait={getattr(battle, 'wait', None)} "
            f"force_switch={getattr(battle, 'force_switch', None)} "
            f"active={species_list(getattr(battle, 'active_pokemon', []))} "
            f"opponent_active={species_list(getattr(battle, 'opponent_active_pokemon', []))} "
            f"last_player_action={last_player_action} "
            f"request_moves={move_request_summary()} "
            f"valid_orders={valid_order_summary()}"
        )

    def _describe_pending_battle_tasks(self) -> str:
        def _format_thread_stack(thread: Thread | None, *, limit: int = 16) -> str:
            if thread is None:
                return "thread=None"
            ident = thread.ident
            if ident is None:
                return (
                    f"thread_name={thread.name} thread_alive={thread.is_alive()} "
                    "thread_ident=None"
                )
            frame = sys._current_frames().get(ident)
            if frame is None:
                return (
                    f"thread_name={thread.name} thread_alive={thread.is_alive()} "
                    f"thread_ident={ident} stack=unavailable"
                )
            stack = []
            current_frame = frame
            for _ in range(limit):
                stack.append(current_frame)
                current_frame = current_frame.f_back
                if current_frame is None:
                    break
            stack.reverse()
            stack_bits = [
                f"{entry.f_code.co_name}@{os.path.basename(entry.f_code.co_filename)}:{entry.f_lineno}"
                for entry in stack
            ]
            return (
                f"thread_name={thread.name} thread_alive={thread.is_alive()} "
                f"thread_ident={ident} stack={' -> '.join(stack_bits)}"
            )

        def _describe_loop_thread_state() -> str:
            loop = getattr(self, "_loop", None)
            loop_thread = getattr(getattr(self, "_loop_thread", None), "thread", None)
            loop_running = None if loop is None else loop.is_running()
            loop_closed = None if loop is None else loop.is_closed()
            return (
                f"loop_running={loop_running} loop_closed={loop_closed} "
                + _format_thread_stack(loop_thread)
            )

        def _describe_await_chain(coro, *, limit: int = 12) -> str:
            bits: list[str] = []
            current = coro
            for _ in range(limit):
                frame = getattr(current, "cr_frame", None) or getattr(
                    current, "gi_frame", None
                )
                code = getattr(current, "cr_code", None) or getattr(
                    current, "gi_code", None
                )
                if frame is not None:
                    bits.append(
                        f"{frame.f_code.co_name}@{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
                    )
                elif code is not None:
                    bits.append(f"{code.co_name}@{os.path.basename(code.co_filename)}")
                else:
                    bits.append(repr(current))

                next_coro = getattr(current, "cr_await", None) or getattr(
                    current, "gi_yieldfrom", None
                )
                if next_coro is None:
                    break
                current = next_coro
            return " -> ".join(bits)

        loop_thread_state = _describe_loop_thread_state()

        async def _describe() -> str:
            descriptions: list[str] = []
            descriptions.append("BattleEnv loop: " + loop_thread_state)
            obs_queue = getattr(getattr(self, "player", None), "obs_queue", None)
            action_queue = getattr(getattr(self, "player", None), "action_queue", None)
            queue_bits = []
            for name, queue in (("obs", obs_queue), ("action", action_queue)):
                if hasattr(queue, "qsize"):
                    queue_bits.append(f"{name}_qsize={queue.qsize()}")
            if queue_bits:
                descriptions.append("BattleEnv queues: " + ", ".join(queue_bits))
            pending_action_future = getattr(
                getattr(self, "player", None),
                "_pending_action_future",
                None,
            )
            pending_action_done = (
                None if pending_action_future is None else pending_action_future.done()
            )
            descriptions.append(
                "BattleEnv action request: "
                f"env_pending_request_id={getattr(self, '_pending_player_request_id', None)} "
                f"player_pending_request_id={getattr(getattr(self, 'player', None), '_pending_action_request_id', None)} "
                f"player_pending_done={pending_action_done}"
            )
            descriptions.append(
                "BattleEnv current battle: " + self._describe_current_battle_state()
            )
            describe_opponent_inference = getattr(
                getattr(self, "opponent", None),
                "describe_pending_inference",
                None,
            )
            if callable(describe_opponent_inference):
                try:
                    descriptions.append(
                        "League opponent inference: " + describe_opponent_inference()
                    )
                except Exception as error:
                    descriptions.append(
                        f"League opponent inference: describe_failed={error!r}"
                    )
            ps_client = getattr(getattr(self, "player", None), "ps_client", None)
            describe_active_sessions = getattr(
                ps_client, "describe_active_sessions", None
            )
            if callable(describe_active_sessions):
                try:
                    descriptions.append(
                        "LocalBattleStream sessions: "
                        + await asyncio.wait_for(
                            describe_active_sessions(),
                            timeout=0.75,
                        )
                    )
                except Exception as error:
                    descriptions.append(
                        f"LocalBattleStream sessions: describe_failed={error!r}"
                    )
            current = asyncio.current_task()
            for task in asyncio.all_tasks():
                if task is current or task.done():
                    continue
                coro = task.get_coro()
                descriptions.append(f"{coro!r}: {_describe_await_chain(coro)}")
            return "Pending event-loop tasks: " + (
                " | ".join(descriptions) if descriptions else "none"
            )

        try:
            return asyncio.run_coroutine_threadsafe(_describe(), self._loop).result(
                timeout=1.0
            )
        except Exception as exc:
            return (
                f"Could not describe pending event-loop tasks: {exc!r} | "
                "BattleEnv loop: " + loop_thread_state
            )

    def _wait_for_battle_task_completion(self, context: str) -> None:
        task = self._battle_task
        if task is None:
            return

        timeout = self.config.battlestream_startup_timeout
        timing_enabled = self._timing.enabled
        if timing_enabled:
            wait_started = time.perf_counter()
        try:
            task.result(timeout=timeout)
        except TimeoutError as exc:
            raise TimeoutError(
                f"Timed out waiting {timeout:.1f}s for the BattleEnv battle task "
                f"to finish {context}; a new local BattleStream battle would "
                "overlap the previous battle."
            ) from exc
        except Exception as exc:
            raise RuntimeError(f"Battle task failed {context}.") from exc
        else:
            self._battle_task = None
        finally:
            if timing_enabled:
                self._timing.add(
                    "env_battle_task_finish_wait_sec",
                    time.perf_counter() - wait_started,
                )
                self._timing.increment("env_battle_task_finish_wait_calls")

    async def _do_team_preview(self, battle: DoubleBattle) -> str:
        """Build the team preview response using the agent callback.

        Called from ShowdownPlayer.teampreview() at the start of
        each battle. Stores (obs, picks) so they can be returned in the info dict
        when this episode ends, enabling the REINFORCE team preview gradient.
        """
        if self._ext_team_preview_fn is None:
            raise RuntimeError(
                "BattleEnv team preview requested without a team_preview_fn."
            )
        obs = self.player.obs_builder.build(battle)
        result = await self._choose_team_preview_picks_async(obs)
        picks = result.picks
        team_str = "/team " + "".join(str(int(pick) + 1) for pick in picks)
        self._episode_team_preview = (obs, picks, result.policy_version)
        return team_str

    def _assign_teams_for_next_battle(self) -> None:
        if self._team_pair_provider is None:
            return

        self._prepare_next_battle()
        player_team = self._prepared_player_team
        opponent_team = self._prepared_opponent_team
        self._active_preview_obs = self._prepared_preview_obs
        self._active_preview_future = self._prepared_preview_future
        self._prepared_player_team = None
        self._prepared_opponent_team = None
        self._prepared_preview_obs = None
        self._prepared_preview_future = None
        if player_team is not None:
            self.player.update_team(player_team)
        if opponent_team is not None:
            self.opponent.update_team(opponent_team)
            self._episode_opponent_packed_team = opponent_team

    @property
    def last_battle(self) -> DoubleBattle | None:
        return self.player.last_battle

    def set_next_opponent(self, opponent_factory: OpponentFactory):
        timing_enabled = self._timing.enabled
        if timing_enabled:
            prebuild_started = time.perf_counter()
        next_opponent = self._build_opponent(opponent_factory)
        if timing_enabled:
            self._timing.add(
                "env_prebuild_opponent_sec", time.perf_counter() - prebuild_started
            )
        if self._pending_opponent is not None:
            self._disconnect_player(self._pending_opponent, self._loop)
        self._pending_opponent = next_opponent

    def stage_next_opponent_checkpoint(self, checkpoint_path: str) -> bool:
        if self._pending_opponent is not None or not hasattr(
            self.opponent, "stage_checkpoint"
        ):
            return False
        timing_enabled = self._timing.enabled
        if timing_enabled:
            stage_started = time.perf_counter()
        self.opponent.stage_checkpoint(checkpoint_path)
        if timing_enabled:
            self._timing.add(
                "env_stage_opponent_checkpoint_sec",
                time.perf_counter() - stage_started,
            )
            self._timing.increment("env_stage_opponent_checkpoint_calls")
        return True

    def _apply_pending_opponent(self):
        if self._pending_opponent is None:
            if hasattr(self.opponent, "apply_staged_checkpoint"):
                timing_enabled = self._timing.enabled
                if timing_enabled:
                    swap_started = time.perf_counter()
                if self.opponent.apply_staged_checkpoint():
                    if timing_enabled:
                        self._timing.add(
                            "env_reset_opponent_checkpoint_swap_sec",
                            time.perf_counter() - swap_started,
                        )
            return

        timing_enabled = self._timing.enabled
        if timing_enabled:
            disconnect_started = time.perf_counter()
        self._disconnect_player(self.opponent, self._loop)
        if timing_enabled:
            self._timing.add(
                "env_reset_opponent_disconnect_sec",
                time.perf_counter() - disconnect_started,
            )
        self.opponent = self._pending_opponent
        self._pending_opponent = None

    def _build_opponent(self, opponent_factory: OpponentFactory) -> Player:
        opponent = opponent_factory(self._loop)
        self._attach_choice_timer(opponent)
        if hasattr(opponent, "set_timing"):
            opponent.set_timing(self._timing)
        self._assert_player_loop(opponent, "opponent")
        return opponent

    def _prepare_next_battle(self, recompute_preview: bool = False) -> None:
        if self._team_pair_provider is None:
            return

        if self._prepared_player_team is None or self._prepared_opponent_team is None:
            self._prepared_player_team, self._prepared_opponent_team = (
                self._team_pair_provider()
            )
            recompute_preview = True

        if (
            not recompute_preview
            or self._team_preview_batcher is None
            or self._ext_team_preview_fn is None
        ):
            return

        timing_enabled = self._timing.enabled
        if timing_enabled:
            build_started = time.perf_counter()
        self._prepared_preview_obs = build_team_preview_observation(
            self.game_data,
            self.config.team_size,
            self._prepared_player_team,
            self._prepared_opponent_team,
        )
        if timing_enabled:
            self._timing.add(
                "env_precompute_preview_obs_sec", time.perf_counter() - build_started
            )

        if timing_enabled:
            submit_started = time.perf_counter()
        self._prepared_preview_future = self._team_preview_batcher.submit(
            self._prepared_preview_obs
        )
        if timing_enabled:
            self._timing.add(
                "env_precompute_preview_submit_sec",
                time.perf_counter() - submit_started,
            )
            self._timing.increment("env_precompute_preview_submit_calls")

        next_opponent = (
            self._pending_opponent
            if self._pending_opponent is not None
            else self.opponent
        )
        if hasattr(next_opponent, "prepare_team_preview"):
            if timing_enabled:
                opp_started = time.perf_counter()
            next_opponent.prepare_team_preview(
                own_packed_team=self._prepared_opponent_team,
                opponent_packed_team=self._prepared_player_team,
            )
            if timing_enabled:
                self._timing.add(
                    "env_precompute_opponent_preview_prepare_sec",
                    time.perf_counter() - opp_started,
                )

    def refresh_prepared_team_preview(self) -> None:
        if self._prepared_player_team is None or self._prepared_opponent_team is None:
            return
        self._prepare_next_battle(recompute_preview=True)

    def wait_for_prepared_team_preview(self) -> None:
        if self._prepared_preview_future is None:
            return
        timing_enabled = self._timing.enabled
        if timing_enabled:
            wait_started = time.perf_counter()
        self._prepared_preview_future.result()
        if timing_enabled:
            self._timing.add(
                "env_precompute_preview_wait_sec", time.perf_counter() - wait_started
            )

    def _choose_team_preview_picks(
        self, obs: dict[str, np.ndarray]
    ) -> TeamPreviewResult:
        prepared_obs = self._active_preview_obs
        prepared_future = self._active_preview_future
        self._active_preview_obs = None
        self._active_preview_future = None

        if prepared_obs is None and prepared_future is None:
            result, _ = self._ext_team_preview_fn(obs)
            return BattleEnv._coerce_team_preview_result(result)

        if prepared_obs is None or prepared_future is None:
            if _timing_enabled(self._timing):
                self._timing.increment("env_precompute_preview_missing_state")
            raise RuntimeError("Prepared team preview state was partially missing.")

        if self._obs_equal(obs, prepared_obs):
            timing_enabled = _timing_enabled(self._timing)
            if timing_enabled:
                self._timing.increment("env_precompute_preview_hits")
            if timing_enabled:
                wait_started = time.perf_counter()
            result = prepared_future.result()
            if timing_enabled:
                self._timing.add(
                    "env_precompute_preview_hit_wait_sec",
                    time.perf_counter() - wait_started,
                )
            return BattleEnv._coerce_team_preview_result(result)

        if _timing_enabled(self._timing):
            self._timing.increment("env_precompute_preview_misses")
        mismatch = self._record_preview_obs_mismatch(obs, prepared_obs)
        raise RuntimeError(
            "Prepared team preview observation did not match the live battle state: "
            f"{mismatch}."
        )

    async def _choose_team_preview_picks_async(
        self, obs: dict[str, np.ndarray]
    ) -> TeamPreviewResult:
        prepared_obs = self._active_preview_obs
        prepared_future = self._active_preview_future
        self._active_preview_obs = None
        self._active_preview_future = None

        if prepared_obs is None and prepared_future is None:
            if self._team_preview_batcher is not None:
                return await asyncio.wrap_future(self._team_preview_batcher.submit(obs))
            result = self._ext_team_preview_fn(obs)
            if isinstance(result, Awaitable):
                result = await result
            picks, _ = result
            return BattleEnv._coerce_team_preview_result(picks)

        if prepared_obs is None or prepared_future is None:
            if _timing_enabled(self._timing):
                self._timing.increment("env_precompute_preview_missing_state")
            raise RuntimeError("Prepared team preview state was partially missing.")

        if self._obs_equal(obs, prepared_obs):
            timing_enabled = _timing_enabled(self._timing)
            if timing_enabled:
                self._timing.increment("env_precompute_preview_hits")
            if timing_enabled:
                wait_started = time.perf_counter()
            result = await asyncio.wrap_future(prepared_future)
            if timing_enabled:
                self._timing.add(
                    "env_precompute_preview_hit_wait_sec",
                    time.perf_counter() - wait_started,
                )
            return BattleEnv._coerce_team_preview_result(result)

        if _timing_enabled(self._timing):
            self._timing.increment("env_precompute_preview_misses")
        mismatch = self._record_preview_obs_mismatch(obs, prepared_obs)
        raise RuntimeError(
            "Prepared team preview observation did not match the live battle state: "
            f"{mismatch}."
        )

    @staticmethod
    def _coerce_team_preview_result(
        result: np.ndarray | TeamPreviewResult,
    ) -> TeamPreviewResult:
        if isinstance(result, TeamPreviewResult):
            return result
        return TeamPreviewResult(
            picks=np.asarray(result, dtype=np.int64),
            policy_version=None,
        )

    @staticmethod
    def _obs_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
        if left.keys() != right.keys():
            return False
        return all(np.array_equal(left[key], right[key]) for key in left)

    def _record_preview_obs_mismatch(
        self,
        actual: dict[str, np.ndarray],
        prepared: dict[str, np.ndarray],
    ) -> str:
        if actual.keys() != prepared.keys():
            self._timing.increment("env_precompute_preview_keyset_mismatch")
        else:
            for key in actual:
                if not np.array_equal(actual[key], prepared[key]):
                    self._timing.increment(f"env_precompute_preview_mismatch_{key}")
        return summarize_observation_mismatch(actual, prepared)

    def _attach_choice_timer(self, opponent: Player) -> None:
        original_choose_move = opponent.choose_move

        def timed_choose_move(battle):
            timing_enabled = self._timing.enabled
            started = 0.0
            if timing_enabled:
                started = time.perf_counter()
            result = original_choose_move(battle)
            if isinstance(result, Awaitable):

                async def finish_async_choice():
                    return finish_timed_choose_move(
                        await result, battle, started, timing_enabled
                    )

                return finish_async_choice()

            return finish_timed_choose_move(result, battle, started, timing_enabled)

        def finish_timed_choose_move(
            result, battle, started: float, timing_enabled: bool
        ):
            if timing_enabled:
                self._timing.increment("opponent_choice_calls")
                self._timing.add("opponent_choice_sec", time.perf_counter() - started)
            battle_tag = getattr(battle, "battle_tag", "")
            turn = int(getattr(battle, "turn", 0))
            self._episode_opponent_choice_calls += 1
            key = (battle_tag, turn)
            self._episode_opponent_turn_calls[key] += 1
            repeated_calls = self._episode_opponent_turn_calls[key]
            if repeated_calls in (10, 100, 1000):
                own_active = [
                    pokemon.species if pokemon is not None else None
                    for pokemon in getattr(battle, "active_pokemon", [])
                ]
                opp_active = [
                    pokemon.species if pokemon is not None else None
                    for pokemon in getattr(battle, "opponent_active_pokemon", [])
                ]
                checkpoint_path = getattr(self.opponent, "checkpoint_path", None)
                request_moves = []
                last_request = getattr(battle, "last_request", {})
                for active_request in last_request.get("active", []):
                    request_moves.append(
                        [
                            {
                                "id": move_request.get("id"),
                                "target": move_request.get("target"),
                                "disabled": move_request.get("disabled", False),
                            }
                            for move_request in active_request.get("moves", [])
                        ]
                    )
                print(
                    "Warning: repeated opponent choice requests "
                    f"battle={battle_tag} turn={turn} count={repeated_calls} "
                    f"checkpoint={checkpoint_path} own_active={own_active} opp_active={opp_active} "
                    f"force_switch={getattr(battle, 'force_switch', None)} "
                    f"last_player_action={self._last_player_action.tolist() if self._last_player_action is not None else None} "
                    f"request_moves={request_moves} "
                    f"opponent_order={getattr(result, 'message', repr(result))}"
                )
            return result

        opponent.choose_move = timed_choose_move

        original_teampreview = opponent.teampreview

        def timed_teampreview(battle):
            timing_enabled = self._timing.enabled
            started = 0.0
            if timing_enabled:
                started = time.perf_counter()
            result = original_teampreview(battle)
            if isinstance(result, Awaitable):

                async def finish_async_teampreview():
                    return finish_timed_teampreview(
                        await result,
                        started,
                        timing_enabled,
                    )

                return finish_async_teampreview()

            return finish_timed_teampreview(result, started, timing_enabled)

        def finish_timed_teampreview(result: str, started: float, timing_enabled: bool):
            if timing_enabled:
                self._timing.increment("opponent_teampreview_calls")
                self._timing.add(
                    "opponent_teampreview_sec", time.perf_counter() - started
                )
            self._episode_opponent_preview_picks = parse_team_preview_order(
                result,
                team_size=self.config.team_size,
                pick_size=self.config.pick_size,
            )
            return result

        opponent.teampreview = timed_teampreview

    def snapshot_timing(self) -> dict[str, float]:
        return self._timing.snapshot_and_reset()

    def reset_timing(self) -> None:
        self._timing.reset()

    def _assert_player_loop(self, participant: Player, role: str) -> None:
        participant_loop = getattr(
            getattr(participant, "ps_client", None), "loop", None
        )
        if participant_loop is not self._loop:
            raise ValueError(
                f"BattleEnv {role} must be created on the environment event loop."
            )

    def close(self):
        if self._closed:
            return

        self.player.request_close()

        if self._battle_task is not None and not self._battle_task.done():
            try:
                self._battle_task.result(timeout=2.0)
            except TimeoutError:
                self._battle_task.cancel()
                try:
                    self._battle_task.result(timeout=1.0)
                except (CancelledError, TimeoutError):
                    pass
            except Exception:
                pass

        for participant in (self.player, self.opponent):
            self._disconnect_player(participant, self._loop)
        if self._pending_opponent is not None:
            self._disconnect_player(self._pending_opponent, self._loop)

        self.player.obs_queue.clear()
        self.player.action_queue.clear()
        if self._owns_loop_thread:
            self._loop_thread.close()

        self._battle_task = None
        self._pending_opponent = None
        self._prepared_player_team = None
        self._prepared_opponent_team = None
        self._prepared_preview_obs = None
        self._prepared_preview_future = None
        self._active_preview_obs = None
        self._active_preview_future = None
        self._pending_player_request_id = None
        self._closed = True

    @staticmethod
    def _disconnect_player(player: Player, loop: asyncio.AbstractEventLoop) -> None:
        ps_client = getattr(player, "ps_client", None)
        if ps_client is None:
            return
        stop_listening = getattr(ps_client, "stop_listening", None)
        if not callable(stop_listening):
            return
        is_local = hasattr(ps_client, "local_server_configuration")
        if not (hasattr(ps_client, "websocket") or is_local):
            return

        try:
            stop_coro = stop_listening()
            if not is_local:
                stop_coro = asyncio.wait_for(stop_coro, timeout=1.0)
            future = asyncio.run_coroutine_threadsafe(stop_coro, loop)
            future.result(
                timeout=_LOCAL_DISCONNECT_TIMEOUT_SECONDS if is_local else 2.0
            )
        except Exception as exc:
            print(
                f"Warning: disconnect failed for {getattr(player, 'username', '?')!r}: {exc}"
            )


class ParallelBattleEnv:
    """Runs multiple BattleEnvs and collects observations in batch.

    Each env runs its own battle. step() sends actions to all envs
    and collects results. Environments that finish are auto-reset.
    """

    def __init__(
        self,
        envs: list[BattleEnv],
        collect_timing: bool = False,
        reset_concurrency: int | None = None,
        loop_threads: list[LoopThread] | tuple[LoopThread, ...] | None = None,
    ):
        self.envs = envs
        self.num_envs = len(envs)
        self._loop_threads = tuple(loop_threads or ())
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)
        self._reset_concurrency = self._resolve_reset_concurrency(reset_concurrency)
        self._reset_semaphore = BoundedSemaphore(max(1, self._reset_concurrency))
        self._timing = _TimingStats(enabled=collect_timing)
        self._obs_batch_buffers: dict[str, np.ndarray] | None = None
        self._mask_batch_buffers: dict[str, np.ndarray] | None = None
        self._reward_buffer = np.empty(self.num_envs, dtype=np.float32)
        self._done_buffer = np.empty(self.num_envs, dtype=bool)

    def reset(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Reset all environments and return batched obs + masks."""
        timing_enabled = self._timing.enabled
        if timing_enabled:
            started = time.perf_counter()
        results = self._reset_envs()
        if timing_enabled:
            self._timing.increment("parallel_reset_calls")
            self._timing.add(
                "parallel_reset_concurrency", float(self._reset_concurrency)
            )
            self._timing.add("parallel_reset_wall_sec", time.perf_counter() - started)
        return self._collate([r[0] for r in results], [r[1] for r in results])

    def _resolve_reset_concurrency(self, reset_concurrency: int | None) -> int:
        configured = reset_concurrency
        if configured is None and self.envs:
            config = getattr(self.envs[0], "config", None)
            configured = getattr(config, "reset_concurrency", None)
        return resolve_reset_concurrency(configured, self.num_envs)

    def _reset_envs(self) -> list[tuple[dict, dict[str, np.ndarray]]]:
        if self._reset_concurrency >= self.num_envs:
            return list(self._executor.map(self._reset_env, self.envs))

        results: list[tuple[dict, dict[str, np.ndarray]] | None] = [
            None
        ] * self.num_envs
        next_index = 0
        futures: dict[Future, int] = {}

        while next_index < self.num_envs and len(futures) < self._reset_concurrency:
            futures[self._executor.submit(self._reset_env, self.envs[next_index])] = (
                next_index
            )
            next_index += 1

        while futures:
            for future in as_completed(tuple(futures)):
                env_index = futures.pop(future)
                results[env_index] = future.result()
                if next_index < self.num_envs:
                    futures[
                        self._executor.submit(self._reset_env, self.envs[next_index])
                    ] = next_index
                    next_index += 1
                break

        if any(result is None for result in results):
            raise RuntimeError(
                "ParallelBattleEnv.reset() completed with missing environment results."
            )
        return [result for result in results if result is not None]

    def _reset_env(self, env: BattleEnv) -> tuple[dict, dict[str, np.ndarray]]:
        with self._reset_semaphore:
            return env.reset()

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict, dict[str, np.ndarray], np.ndarray, np.ndarray, list[dict]]:
        """Step all environments.

        Args:
            actions: (num_envs, 3) action array.

        Returns:
            obs_batch, mask_batch, rewards, dones, infos
        """
        timing_enabled = self._timing.enabled
        if timing_enabled:
            step_started = time.perf_counter()
            futures = {
                self._executor.submit(self._step_env, env, action): index
                for index, (env, action) in enumerate(
                    zip(self.envs, actions, strict=True)
                )
            }
            results = [None] * self.num_envs
            completion_times: list[float] = []
            for future in as_completed(futures):
                completion_times.append(time.perf_counter() - step_started)
                results[futures[future]] = future.result()
            step_wait_sec = time.perf_counter() - step_started
        else:
            results = list(self._executor.map(self._step_env, self.envs, actions))
            completion_times = []
            step_wait_sec = 0.0
        if any(result is None for result in results):
            raise RuntimeError(
                "ParallelBattleEnv.step() completed with missing environment results."
            )
        obs_list = [result[0] for result in results]
        mask_list = [result[1] for result in results]
        infos = [result[4] for result in results]
        for index, result in enumerate(results):
            self._reward_buffer[index] = result[2]
            self._done_buffer[index] = result[3]

        if timing_enabled:
            collate_started = time.perf_counter()
        batched_obs, batched_mask = self._collate(obs_list, mask_list)
        if timing_enabled:
            self._timing.increment("parallel_step_calls")
            self._timing.add("parallel_step_wait_sec", step_wait_sec)
            if completion_times:
                sorted_completions = sorted(completion_times)
                p50_index = min(
                    len(sorted_completions) - 1, len(sorted_completions) // 2
                )
                p90_index = min(
                    len(sorted_completions) - 1,
                    math.ceil(len(sorted_completions) * 0.9) - 1,
                )
                self._timing.add(
                    "parallel_step_first_completion_sec", sorted_completions[0]
                )
                self._timing.add(
                    "parallel_step_median_completion_sec", sorted_completions[p50_index]
                )
                self._timing.add(
                    "parallel_step_p90_completion_sec", sorted_completions[p90_index]
                )
                self._timing.add(
                    "parallel_step_completion_span_sec",
                    sorted_completions[-1] - sorted_completions[0],
                )
                self._timing.add(
                    "parallel_step_completed_envs", float(len(sorted_completions))
                )
            self._timing.add(
                "parallel_collate_sec", time.perf_counter() - collate_started
            )
            self._timing.add(
                "parallel_step_wall_sec", time.perf_counter() - step_started
            )
        return (
            batched_obs,
            batched_mask,
            self._reward_buffer,
            self._done_buffer,
            infos,
        )

    def snapshot_timing(self) -> dict[str, float]:
        snapshot = self._timing.snapshot_and_reset()
        for env in self.envs:
            env_snapshot = env.snapshot_timing()
            for key, value in env_snapshot.items():
                snapshot[key] = snapshot.get(key, 0.0) + value
        return snapshot

    def reset_timing(self) -> None:
        self._timing.reset()
        for env in self.envs:
            env.reset_timing()

    def close(self):
        if self._closed:
            return

        errors: list[Exception] = []
        # A worker may be blocked inside BattleEnv.step() waiting for its battle
        # loop. Close environments first so their pending requests/tasks are
        # resolved before waiting for executor workers to exit.
        for env in self.envs:
            try:
                env.close()
            except Exception as exc:
                errors.append(exc)

        try:
            self._executor.shutdown(wait=True, cancel_futures=True)
        except Exception as exc:
            errors.append(exc)

        # Borrowed BattleEnv loops remain alive until every environment has had
        # a chance to stop its battle task and disconnect both clients, and all
        # synchronous workers have unwound from those loops.
        for loop_thread in self._loop_threads:
            try:
                loop_thread.close()
            except Exception as exc:
                errors.append(exc)

        self._closed = not errors
        if errors:
            raise ExceptionGroup("ParallelBattleEnv cleanup failed.", errors)

    def wait_for_prepared_team_previews(self) -> None:
        list(
            self._executor.map(
                lambda env: env.wait_for_prepared_team_preview(), self.envs
            )
        )

    def refresh_prepared_team_previews(self) -> None:
        list(
            self._executor.map(
                lambda env: env.refresh_prepared_team_preview(), self.envs
            )
        )

    def _collate(
        self,
        obs_list: list[dict[str, np.ndarray]],
        mask_list: list[dict[str, np.ndarray]],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """Copy observations into reusable batch buffers."""
        if self._obs_batch_buffers is None or self._mask_batch_buffers is None:
            self._init_collate_buffers(obs_list[0], mask_list[0])
        else:
            self._ensure_collate_buffers(obs_list[0], mask_list[0])

        assert self._obs_batch_buffers is not None
        assert self._mask_batch_buffers is not None
        for env_index, obs in enumerate(obs_list):
            for key, value in obs.items():
                self._obs_batch_buffers[key][env_index] = value
            for key, value in mask_list[env_index].items():
                self._mask_batch_buffers[key][env_index] = value
        return self._obs_batch_buffers, self._mask_batch_buffers

    def _init_collate_buffers(
        self, obs: dict[str, np.ndarray], mask: dict[str, np.ndarray]
    ) -> None:
        self._obs_batch_buffers = {
            key: np.empty((self.num_envs, *value.shape), dtype=value.dtype)
            for key, value in obs.items()
        }
        self._mask_batch_buffers = {
            key: np.empty((self.num_envs, *value.shape), dtype=value.dtype)
            for key, value in mask.items()
        }

    def _ensure_collate_buffers(
        self, obs: dict[str, np.ndarray], mask: dict[str, np.ndarray]
    ) -> None:
        assert self._obs_batch_buffers is not None
        assert self._mask_batch_buffers is not None

        if self._obs_batch_buffers.keys() != obs.keys():
            self._obs_batch_buffers = None
            self._mask_batch_buffers = None
            self._init_collate_buffers(obs, mask)
            return

        for key, value in obs.items():
            buffer = self._obs_batch_buffers.get(key)
            expected_shape = (self.num_envs, *value.shape)
            if (
                buffer is None
                or buffer.shape != expected_shape
                or buffer.dtype != value.dtype
            ):
                self._obs_batch_buffers = None
                self._mask_batch_buffers = None
                self._init_collate_buffers(obs, mask)
                return

        if self._mask_batch_buffers.keys() != mask.keys():
            self._obs_batch_buffers = None
            self._mask_batch_buffers = None
            self._init_collate_buffers(obs, mask)
            return

        for key, value in mask.items():
            buffer = self._mask_batch_buffers.get(key)
            expected_shape = (self.num_envs, *value.shape)
            if (
                buffer is None
                or buffer.shape != expected_shape
                or buffer.dtype != value.dtype
            ):
                self._obs_batch_buffers = None
                self._mask_batch_buffers = None
                self._init_collate_buffers(obs, mask)
                return

    def _step_env(
        self, env: BattleEnv, action: np.ndarray
    ) -> tuple[dict, dict[str, np.ndarray], float, bool, dict]:
        obs, mask, reward, done, info = env.step(action)
        if done:
            info["final_observation"] = obs
            obs, mask = self._reset_env(env)
            info["episode_reward"] = reward
        return obs, mask, reward, done, info
