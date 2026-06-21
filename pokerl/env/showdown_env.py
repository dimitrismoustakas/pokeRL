"""Gymnasium-compatible wrapper around poke-env for doubles battles."""

from __future__ import annotations

import asyncio
import itertools
import math
import os
import time
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from threading import Lock, Thread
from typing import Generic, TypeVar

import numpy as np

from poke_env import AccountConfiguration
try:
    from poke_env import LocalBattleStreamConfiguration
except ImportError as exc:  # pragma: no cover - exercised by monkeypatch tests.
    LocalBattleStreamConfiguration = None
    _LOCAL_BATTLESTREAM_IMPORT_ERROR = exc
else:
    _LOCAL_BATTLESTREAM_IMPORT_ERROR = None
from poke_env.battle import DoubleBattle
from poke_env.concurrency import create_in_poke_loop
from poke_env.player import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder, PassBattleOrder, Player

from ..config import EnvConfig
from ..data import GameData
from ..training.preview import TeamPreviewBatcher
from .observation import ObservationBuilder, build_team_preview_observation
from .team_preview import parse_team_preview_order
from .actions import (
    action_to_order,
    build_legal_action_state,
    build_policy_action_masks,
    decode_tera_choice,
    empty_action_masks,
)
from .rewards import sparse_reward


_ACCOUNT_LOCK = Lock()
_ACCOUNT_COUNTER = itertools.count()
_ACCOUNT_RUN_TAG = uuid.uuid4().hex[:4]
QueueItemT = TypeVar("QueueItemT")
_OBSERVATION_POLL_SECONDS = 1.0
OpponentFactory = Callable[[asyncio.AbstractEventLoop], Player]
TeamPairProvider = Callable[[], tuple[str | None, str | None]]


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
            pending_tasks = [task for task in asyncio.all_tasks() if task is not current_task]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await asyncio.get_running_loop().shutdown_asyncgens()

        try:
            asyncio.run_coroutine_threadsafe(_cancel_pending(), self.loop).result(timeout=2.0)
        except (CancelledError, TimeoutError):
            pass

        self.loop.call_soon_threadsafe(self.loop.stop)
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if not self.loop.is_running():
            self.loop.close()
        self._closed = True


class _AsyncQueue(Generic[QueueItemT]):
    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int = 0):
        self.queue = create_in_poke_loop(asyncio.Queue, loop, maxsize)
        self._loop = loop

    async def async_get(self) -> QueueItemT:
        return await self.queue.get()

    def get(self, timeout: float | None = None) -> QueueItemT:
        coro = self.async_get() if timeout is None else asyncio.wait_for(self.async_get(), timeout)
        result = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return result.result()

    async def async_put(self, item: QueueItemT) -> None:
        await self.queue.put(item)

    def put(self, item: QueueItemT) -> None:
        try:
            if asyncio.get_running_loop() is self._loop:
                self.queue.put_nowait(item)
                return
        except RuntimeError:
            pass
        self._loop.call_soon_threadsafe(self.queue.put_nowait, item)

    def clear(self) -> None:
        async def _clear() -> None:
            while True:
                try:
                    self.queue.get_nowait()
                except asyncio.QueueEmpty:
                    return

        asyncio.run_coroutine_threadsafe(_clear(), self._loop).result()


def make_local_account_configuration(prefix: str) -> AccountConfiguration:
    """Create a short, unique local showdown username for this process."""
    clean_prefix = "".join(ch for ch in prefix.lower() if ch.isalnum())[:10] or "bot"
    with _ACCOUNT_LOCK:
        suffix = f"{_ACCOUNT_RUN_TAG}{next(_ACCOUNT_COUNTER):04x}"
    return AccountConfiguration(f"{clean_prefix}{suffix}"[:18], None)


def _worker_count(config: EnvConfig) -> int:
    if config.worker_count is not None:
        return config.worker_count
    cpu_count = os.cpu_count() or 1
    return max(1, min(config.num_envs, max(16, cpu_count // 2 or 1)))


def _battles_per_worker(config: EnvConfig, worker_count: int) -> int:
    if config.battles_per_worker is not None:
        return config.battles_per_worker
    return max(1, math.ceil(config.num_envs / worker_count * 2.0))


def _runtime_loop_count(config: EnvConfig, worker_count: int) -> int:
    if config.runtime_loop_count is not None:
        return config.runtime_loop_count
    if worker_count <= 1:
        return 0
    return min(4, worker_count)


def battlestream_summary(config: EnvConfig) -> dict[str, object]:
    worker_count = _worker_count(config)
    return {
        "showdown_dir": config.showdown_dir,
        "worker_count": worker_count,
        "max_battles_per_worker": _battles_per_worker(config, worker_count),
        "pool_mode": config.pool_mode,
        "runtime_loop_count": _runtime_loop_count(config, worker_count),
        "startup_timeout": config.battlestream_startup_timeout,
    }


def make_battlestream_configuration(
    config: EnvConfig,
) -> LocalBattleStreamConfiguration:
    if not config.showdown_dir:
        raise ValueError("EnvConfig.showdown_dir must be set for local BattleStream training.")

    battlestream_configuration_cls = _require_local_battlestream_configuration()
    worker_count = _worker_count(config)
    return battlestream_configuration_cls(
        showdown_dir=config.showdown_dir,
        worker_count=worker_count,
        max_battles_per_worker=_battles_per_worker(config, worker_count),
        pool_mode=config.pool_mode,
        runtime_loop_count=_runtime_loop_count(config, worker_count),
        startup_timeout=config.battlestream_startup_timeout,
    )


def _require_local_battlestream_configuration():
    if LocalBattleStreamConfiguration is None:
        raise RuntimeError(
            "pokeRL local no-server training requires a poke-env build exposing "
            "LocalBattleStreamConfiguration. Install the local poke-env fork/editable "
            "checkout before running smoke tests or training."
        ) from _LOCAL_BATTLESTREAM_IMPORT_ERROR
    return LocalBattleStreamConfiguration


class ShowdownPlayer(Player):
    """A poke-env player that bridges to our RL agent via queues.

    Instead of choosing moves itself, it sends observations to a queue
    and waits for actions from another queue. This lets us use poke-env's
    async battle loop from a synchronous Gymnasium-style step() function.
    """

    def __init__(self, obs_builder: ObservationBuilder, timing: _TimingStats, **kwargs):
        super().__init__(**kwargs)
        self.obs_builder = obs_builder
        self.timing = timing
        self.obs_queue: _AsyncQueue[tuple[dict, dict[str, np.ndarray], float, bool, DoubleBattle]] = _AsyncQueue(self.ps_client.loop)
        self.action_queue: _AsyncQueue[np.ndarray | None] = _AsyncQueue(self.ps_client.loop)
        self.last_battle: DoubleBattle | None = None
        self._last_finished_token: tuple[str, int] | None = None
        self._closing = False
        self._team_preview_fn = None
        self._pending_action_state = None
        self._pending_action_battle: DoubleBattle | None = None

    def choose_move(self, battle: DoubleBattle) -> Awaitable[BattleOrder]:
        return self._choose_move(battle)

    async def _choose_move(self, battle: DoubleBattle) -> BattleOrder:
        self._emit_battle_update(battle)

        if battle.finished:
            return self.choose_default_move()

        action = await self.action_queue.async_get()
        if action is None:
            if not self._closing:
                raise RuntimeError("ShowdownPlayer received a shutdown signal outside env.close().")
            return ForfeitBattleOrder()
        return self._make_order(action, battle)

    def _battle_finished_callback(self, battle):
        self._emit_battle_update(battle)

    def teampreview(self, battle: DoubleBattle) -> str:
        if self._team_preview_fn is None:
            raise RuntimeError(
                "ShowdownPlayer.teampreview() called but no team_preview_fn was set. "
                "Pass team_preview_fn to BattleEnv so the agent learns team selection."
            )
        return self._team_preview_fn(battle)

    def set_team_preview_fn(self, fn):
        self._team_preview_fn = fn

    def request_close(self) -> None:
        self._closing = True
        self.action_queue.put(None)

    def _make_order(self, action: np.ndarray, battle: DoubleBattle) -> BattleOrder:
        """Convert a (3,) action array to a DoubleBattleOrder."""
        timing_enabled = _timing_enabled(self.timing)
        if timing_enabled:
            started = time.perf_counter()
        action_state = self._pending_action_state if self._pending_action_battle is battle else None
        self._pending_action_state = None
        self._pending_action_battle = None
        if action_state is None:
            action_state = build_legal_action_state(battle)
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

        if timing_enabled:
            self.timing.increment("player_make_order_calls")
            self.timing.add("player_make_order_sec", time.perf_counter() - started)
        if all(isinstance(order, PassBattleOrder) for order in orders):
            return self.choose_default_move()
        return DoubleBattleOrder(orders[0], orders[1])

    def _emit_battle_update(self, battle: DoubleBattle) -> None:
        self.last_battle = battle
        finished_token = (battle.battle_tag, battle.turn)
        if battle.finished and finished_token == self._last_finished_token:
            return

        timing_enabled = _timing_enabled(self.timing)
        if timing_enabled:
            emit_started = time.perf_counter()
        obs = self.obs_builder.build(battle)
        if timing_enabled:
            self.timing.add("player_emit_obs_sec", time.perf_counter() - emit_started)

        if timing_enabled:
            mask_started = time.perf_counter()
        if battle.finished:
            mask = empty_action_masks()
            self._pending_action_state = None
            self._pending_action_battle = None
        else:
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
            self.timing.add("player_emit_reward_sec", time.perf_counter() - reward_started)
        done = battle.finished

        if timing_enabled:
            queue_put_started = time.perf_counter()
        self.obs_queue.put((obs, mask, reward, done, battle))
        if timing_enabled:
            self.timing.add("player_emit_queue_put_sec", time.perf_counter() - queue_put_started)
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
        team_preview_fn: Callable[[dict], tuple[np.ndarray, str]] | None = None,
        team_preview_batcher: TeamPreviewBatcher | None = None,
        team_pair_provider: TeamPairProvider | None = None,
        player_kwargs: dict | None = None,
        collect_timing: bool = False,
    ):
        self.config = config
        self.game_data = game_data
        self._battle_task = None
        self._loop_thread = LoopThread.start()
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
        self._episode_team_preview: tuple[dict, np.ndarray] | None = None
        self._episode_opponent_preview_picks: np.ndarray | None = None
        self._episode_opponent_choice_calls = 0
        self._episode_opponent_turn_calls: dict[tuple[str, int], int] = defaultdict(int)
        self._episode_opponent_checkpoint_path: str | None = None
        self._last_player_action: np.ndarray | None = None

        obs_builder = ObservationBuilder(game_data, config.team_size)
        player_options = dict(player_kwargs or {})
        reserved_player_keys = {"battle_format", "loop", "server_configuration"}
        unexpected_keys = reserved_player_keys.intersection(player_options)
        if unexpected_keys:
            bad_keys = ", ".join(sorted(unexpected_keys))
            raise ValueError(f"BattleEnv manages {bad_keys}; do not pass them via player_kwargs.")
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

    def reset(self) -> tuple[dict, dict[str, np.ndarray]]:
        """Start a new battle and return the first observation + mask."""
        if self._closed:
            raise RuntimeError("BattleEnv.reset() called after the environment was closed.")

        timing_enabled = self._timing.enabled
        if timing_enabled:
            reset_started = time.perf_counter()
        self._wait_for_battle_task_completion("before BattleEnv reset")
        self._episode_team_preview = None
        self._episode_opponent_preview_picks = None
        self._episode_opponent_choice_calls = 0
        self._episode_opponent_turn_calls.clear()
        self._episode_opponent_checkpoint_path = getattr(self.opponent, "checkpoint_path", None)
        if timing_enabled:
            apply_pending_started = time.perf_counter()
        self._apply_pending_opponent()
        if timing_enabled:
            self._timing.add("env_reset_apply_pending_sec", time.perf_counter() - apply_pending_started)
        self._episode_opponent_checkpoint_path = getattr(self.opponent, "checkpoint_path", None)

        if timing_enabled:
            assign_teams_started = time.perf_counter()
        self._assign_teams_for_next_battle()
        if timing_enabled:
            self._timing.add("env_reset_team_assign_sec", time.perf_counter() - assign_teams_started)

        # Give stateful opponents (e.g. LeaguePlayer) a clean slate for the new battle.
        if timing_enabled:
            hidden_reset_started = time.perf_counter()
        if hasattr(self.opponent, "reset_hidden"):
            self.opponent.reset_hidden()
        if timing_enabled:
            self._timing.add("env_reset_hidden_reset_sec", time.perf_counter() - hidden_reset_started)

        if timing_enabled:
            battle_start_started = time.perf_counter()
        self._battle_task = asyncio.run_coroutine_threadsafe(
            self.player.battle_against(self.opponent, n_battles=1),
            self._loop,
        )
        if timing_enabled:
            self._timing.add("env_reset_battle_start_sec", time.perf_counter() - battle_start_started)
        self._battle_started_at = time.perf_counter()

        if timing_enabled:
            wait_started = time.perf_counter()
        obs, mask, _, _, _ = self._get_player_observation("initial observation")
        if timing_enabled:
            self._timing.add("env_reset_wait_first_obs_sec", time.perf_counter() - wait_started)
        self._prepare_next_battle()
        if timing_enabled:
            self._timing.increment("env_reset_calls")
            self._timing.add("env_reset_total_sec", time.perf_counter() - reset_started)
        return obs, mask

    def step(self, action: np.ndarray) -> tuple[dict, dict[str, np.ndarray], float, bool, dict]:
        """Send an action and get the next observation.

        Args:
            action: (3,) int array of [slot_1_action, slot_2_action, tera_choice].

        Returns:
            obs, mask, reward, done, info
        """
        if self._closed:
            raise RuntimeError("BattleEnv.step() called after the environment was closed.")

        timing_enabled = self._timing.enabled
        if timing_enabled:
            step_started = time.perf_counter()
            enqueue_started = step_started
        self._last_player_action = np.array(action, copy=True)
        if hasattr(self.opponent, "_last_actions"):
            self.opponent._last_actions = None
        self.player.action_queue.put(action)
        if timing_enabled:
            self._timing.add("env_action_enqueue_sec", time.perf_counter() - enqueue_started)

        if timing_enabled:
            wait_started = time.perf_counter()
        obs, mask, reward, done, battle = self._get_player_observation("step observation")
        if timing_enabled:
            self._timing.add("env_queue_wait_sec", time.perf_counter() - wait_started)
            self._timing.increment("env_step_calls")
            self._timing.add("env_step_total_sec", time.perf_counter() - step_started)
        info = {
            "battle_tag": battle.battle_tag if battle else "",
            "battle_turn": getattr(battle, "turn", 0),
        }

        # Capture the opponent's most recent (a1, a2, tera) for the aux
        # opponent-action head. Only LeaguePlayer exposes this; other opponents
        # leave it None and the PPO update masks the sample.
        opp_actions = getattr(self.opponent, "_last_actions", None)
        if opp_actions is not None:
            info["opp_actions"] = np.asarray(opp_actions, dtype=np.int64).copy()
        if done:
            if self._battle_started_at is not None:
                info["episode_sec"] = time.perf_counter() - self._battle_started_at
            if self._episode_team_preview is not None:
                obs_tp, picks_tp = self._episode_team_preview
                info["team_preview_obs"] = obs_tp
                info["team_preview_picks"] = picks_tp
            if self._episode_opponent_preview_picks is not None:
                info["opponent_team_preview_picks"] = self._episode_opponent_preview_picks
            info["episode_opponent_choice_calls"] = self._episode_opponent_choice_calls
            info["episode_opponent_repeated_turn_requests"] = sum(
                count - 1 for count in self._episode_opponent_turn_calls.values() if count > 1
            )
            info["episode_opponent_max_calls_same_turn"] = max(
                self._episode_opponent_turn_calls.values(),
                default=0,
            )
            if self._episode_opponent_checkpoint_path is not None:
                info["episode_opponent_checkpoint_path"] = self._episode_opponent_checkpoint_path
            self._wait_for_battle_task_completion("after terminal BattleEnv observation")
        return obs, mask, reward, done, info

    def _get_player_observation(
        self,
        context: str,
    ) -> tuple[dict, dict[str, np.ndarray], float, bool, DoubleBattle]:
        while True:
            try:
                return self.player.obs_queue.get(timeout=_OBSERVATION_POLL_SECONDS)
            except TimeoutError as exc:
                self._raise_battle_task_failure(context, exc)

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

    def _do_team_preview(self, battle: DoubleBattle) -> str:
        """Build the team preview response using the agent callback.

        Called synchronously from ShowdownPlayer.teampreview() at the start of
        each battle. Stores (obs, picks) so they can be returned in the info dict
        when this episode ends, enabling the REINFORCE team preview gradient.
        """
        if self._ext_team_preview_fn is None:
            raise RuntimeError("BattleEnv team preview requested without a team_preview_fn.")
        obs = self.player.obs_builder.build(battle)
        picks = self._choose_team_preview_picks(obs)
        team_str = "/team " + "".join(str(int(pick) + 1) for pick in picks)
        self._episode_team_preview = (obs, picks)
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

    @property
    def last_battle(self) -> DoubleBattle | None:
        return self.player.last_battle

    def set_next_opponent(self, opponent_factory: OpponentFactory):
        timing_enabled = self._timing.enabled
        if timing_enabled:
            prebuild_started = time.perf_counter()
        next_opponent = self._build_opponent(opponent_factory)
        if timing_enabled:
            self._timing.add("env_prebuild_opponent_sec", time.perf_counter() - prebuild_started)
        if self._pending_opponent is not None:
            self._disconnect_player(self._pending_opponent, self._loop)
        self._pending_opponent = next_opponent

    def stage_next_opponent_checkpoint(self, checkpoint_path: str) -> bool:
        if self._pending_opponent is not None or not hasattr(self.opponent, "stage_checkpoint"):
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

        if self._pending_opponent is None:
            return

        timing_enabled = self._timing.enabled
        if timing_enabled:
            disconnect_started = time.perf_counter()
        self._disconnect_player(self.opponent, self._loop)
        if timing_enabled:
            self._timing.add("env_reset_opponent_disconnect_sec", time.perf_counter() - disconnect_started)
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
            self._prepared_player_team, self._prepared_opponent_team = self._team_pair_provider()
            recompute_preview = True

        if not recompute_preview or self._team_preview_batcher is None or self._ext_team_preview_fn is None:
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
            self._timing.add("env_precompute_preview_obs_sec", time.perf_counter() - build_started)

        if timing_enabled:
            submit_started = time.perf_counter()
        self._prepared_preview_future = self._team_preview_batcher.submit(self._prepared_preview_obs)
        if timing_enabled:
            self._timing.add("env_precompute_preview_submit_sec", time.perf_counter() - submit_started)
            self._timing.increment("env_precompute_preview_submit_calls")

        next_opponent = self._pending_opponent if self._pending_opponent is not None else self.opponent
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
            self._timing.add("env_precompute_preview_wait_sec", time.perf_counter() - wait_started)

    def _choose_team_preview_picks(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        prepared_obs = self._active_preview_obs
        prepared_future = self._active_preview_future
        self._active_preview_obs = None
        self._active_preview_future = None

        if prepared_obs is None and prepared_future is None:
            picks, _ = self._ext_team_preview_fn(obs)
            return picks

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
            picks = prepared_future.result()
            if timing_enabled:
                self._timing.add("env_precompute_preview_hit_wait_sec", time.perf_counter() - wait_started)
            return picks

        if _timing_enabled(self._timing):
            self._timing.increment("env_precompute_preview_misses")
        self._record_preview_obs_mismatch(obs, prepared_obs)
        raise RuntimeError("Prepared team preview observation did not match the live battle state.")

    @staticmethod
    def _obs_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
        if left.keys() != right.keys():
            return False
        return all(np.array_equal(left[key], right[key]) for key in left)

    def _record_preview_obs_mismatch(
        self,
        actual: dict[str, np.ndarray],
        prepared: dict[str, np.ndarray],
    ) -> None:
        if actual.keys() != prepared.keys():
            self._timing.increment("env_precompute_preview_keyset_mismatch")
            return
        for key in actual:
            if not np.array_equal(actual[key], prepared[key]):
                self._timing.increment(f"env_precompute_preview_mismatch_{key}")

    def _attach_choice_timer(self, opponent: Player) -> None:
        original_choose_move = opponent.choose_move

        def timed_choose_move(battle):
            timing_enabled = self._timing.enabled
            if timing_enabled:
                started = time.perf_counter()
            result = original_choose_move(battle)
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
            if timing_enabled:
                started = time.perf_counter()
            result = original_teampreview(battle)
            if timing_enabled:
                self._timing.increment("opponent_teampreview_calls")
                self._timing.add("opponent_teampreview_sec", time.perf_counter() - started)
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
        participant_loop = getattr(getattr(participant, "ps_client", None), "loop", None)
        if participant_loop is not self._loop:
            raise ValueError(f"BattleEnv {role} must be created on the environment event loop.")

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
        self._loop_thread.close()

        self._battle_task = None
        self._pending_opponent = None
        self._prepared_player_team = None
        self._prepared_opponent_team = None
        self._prepared_preview_obs = None
        self._prepared_preview_future = None
        self._active_preview_obs = None
        self._active_preview_future = None
        self._closed = True

    @staticmethod
    def _disconnect_player(player: Player, loop: asyncio.AbstractEventLoop) -> None:
        ps_client = getattr(player, "ps_client", None)
        if ps_client is None or not hasattr(ps_client, "websocket"):
            return

        try:
            asyncio.run_coroutine_threadsafe(
                asyncio.wait_for(ps_client.stop_listening(), timeout=1.0),
                loop,
            ).result(timeout=2.0)
        except Exception as exc:
            print(f"Warning: disconnect failed for {getattr(player, 'username', '?')!r}: {exc}")


class ParallelBattleEnv:
    """Runs multiple BattleEnvs and collects observations in batch.

    Each env runs its own battle. step() sends actions to all envs
    and collects results. Environments that finish are auto-reset.
    """

    def __init__(self, envs: list[BattleEnv], collect_timing: bool = False):
        self.envs = envs
        self.num_envs = len(envs)
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)
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
        results = list(self._executor.map(lambda env: env.reset(), self.envs))
        if timing_enabled:
            self._timing.increment("parallel_reset_calls")
            self._timing.add("parallel_reset_wall_sec", time.perf_counter() - started)
        return self._collate([r[0] for r in results], [r[1] for r in results])

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
        results = list(self._executor.map(self._step_env, self.envs, actions))
        if timing_enabled:
            step_wait_sec = time.perf_counter() - step_started
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
            self._timing.add("parallel_collate_sec", time.perf_counter() - collate_started)
            self._timing.add("parallel_step_wall_sec", time.perf_counter() - step_started)
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
        for env in self.envs:
            env.close()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def wait_for_prepared_team_previews(self) -> None:
        list(self._executor.map(lambda env: env.wait_for_prepared_team_preview(), self.envs))

    def refresh_prepared_team_previews(self) -> None:
        list(self._executor.map(lambda env: env.refresh_prepared_team_preview(), self.envs))

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

    def _init_collate_buffers(self, obs: dict[str, np.ndarray], mask: dict[str, np.ndarray]) -> None:
        self._obs_batch_buffers = {
            key: np.empty((self.num_envs, *value.shape), dtype=value.dtype)
            for key, value in obs.items()
        }
        self._mask_batch_buffers = {
            key: np.empty((self.num_envs, *value.shape), dtype=value.dtype)
            for key, value in mask.items()
        }

    def _ensure_collate_buffers(self, obs: dict[str, np.ndarray], mask: dict[str, np.ndarray]) -> None:
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
            if buffer is None or buffer.shape != expected_shape or buffer.dtype != value.dtype:
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
            if buffer is None or buffer.shape != expected_shape or buffer.dtype != value.dtype:
                self._obs_batch_buffers = None
                self._mask_batch_buffers = None
                self._init_collate_buffers(obs, mask)
                return

    @staticmethod
    def _step_env(
        env: BattleEnv, action: np.ndarray
    ) -> tuple[dict, dict[str, np.ndarray], float, bool, dict]:
        obs, mask, reward, done, info = env.step(action)
        if done:
            obs, mask = env.reset()
            info["episode_reward"] = reward
        return obs, mask, reward, done, info
