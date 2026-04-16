"""Gymnasium-compatible wrapper around poke-env for doubles battles."""

import asyncio
import itertools
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError
from threading import Lock, Thread
from typing import Generic, TypeVar

import numpy as np

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import DoubleBattle
from poke_env.concurrency import create_in_poke_loop
from poke_env.player import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder, PassBattleOrder, Player

from ..config import EnvConfig
from ..data import GameData
from ..teamgen import maybe_build_teambuilder
from .observation import ObservationBuilder
from .actions import (
    build_action_mask, action_to_order,
)
from .rewards import sparse_reward


_ACCOUNT_LOCK = Lock()
_ACCOUNT_COUNTER = itertools.count()
_ACCOUNT_RUN_TAG = uuid.uuid4().hex[:4]
QueueItemT = TypeVar("QueueItemT")
OpponentFactory = Callable[[asyncio.AbstractEventLoop], Player]


class _TimingStats:
    def __init__(self):
        self._lock = Lock()
        self._values: dict[str, float] = {}

    def add(self, key: str, value: float) -> None:
        with self._lock:
            self._values[key] = self._values.get(key, 0.0) + value

    def increment(self, key: str, delta: int = 1) -> None:
        self.add(key, float(delta))

    def snapshot_and_reset(self) -> dict[str, float]:
        with self._lock:
            snapshot = dict(self._values)
            self._values.clear()
        return snapshot

    def reset(self) -> None:
        with self._lock:
            self._values.clear()


class _AsyncQueue(Generic[QueueItemT]):
    def __init__(self, loop: asyncio.AbstractEventLoop, maxsize: int = 0):
        self.queue = create_in_poke_loop(asyncio.Queue, loop, maxsize)
        self._loop = loop

    async def async_get(self) -> QueueItemT:
        return await self.queue.get()

    def get(self, timeout: float | None = None) -> QueueItemT:
        result = asyncio.run_coroutine_threadsafe(
            asyncio.wait_for(self.async_get(), timeout),
            self._loop,
        )
        return result.result()

    async def async_put(self, item: QueueItemT) -> None:
        await self.queue.put(item)

    def put(self, item: QueueItemT) -> None:
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


def make_server_configuration(config: EnvConfig) -> ServerConfiguration:
    host, port = config.server_url.split(":")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php",
    )


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
        self.obs_queue: _AsyncQueue[tuple[dict, np.ndarray, float, bool, DoubleBattle]] = _AsyncQueue(self.ps_client.loop)
        self.action_queue: _AsyncQueue[np.ndarray | None] = _AsyncQueue(self.ps_client.loop)
        self.last_battle: DoubleBattle | None = None
        self._last_finished_token: tuple[str, int] | None = None
        self._closing = False
        self._team_preview_fn = None

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
        """Convert a (2,) action array to a DoubleBattleOrder."""
        started = time.perf_counter()
        orders = []
        for slot in range(2):
            order = action_to_order(int(action[slot]), battle, slot)
            orders.append(order)

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

        emit_started = time.perf_counter()
        obs = self.obs_builder.build(battle)
        self.timing.add("player_emit_obs_sec", time.perf_counter() - emit_started)

        mask_started = time.perf_counter()
        mask = np.stack([build_action_mask(battle, s) for s in range(2)])
        self.timing.add("player_emit_mask_sec", time.perf_counter() - mask_started)

        reward_started = time.perf_counter()
        reward = sparse_reward(battle)
        self.timing.add("player_emit_reward_sec", time.perf_counter() - reward_started)
        done = battle.finished

        queue_put_started = time.perf_counter()
        self.obs_queue.put((obs, mask, reward, done, battle))
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
        team_preview_fn: Callable[[dict], tuple[np.ndarray, str]],
        player_kwargs: dict | None = None,
    ):
        self.config = config
        self.game_data = game_data
        self._battle_task = None
        self._loop, self._thread = self._start_event_loop()
        self._pending_opponent_factory: OpponentFactory | None = None
        self._closed = False
        self._battle_started_at: float | None = None
        self._timing = _TimingStats()
        self._ext_team_preview_fn = team_preview_fn
        # Set at team preview time, read when the episode ends to attach to info.
        self._episode_team_preview: tuple[dict, np.ndarray] | None = None

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
        if "team" not in player_options:
            team_builder = maybe_build_teambuilder(config.battle_format, config.team_pool_size)
            if team_builder is not None:
                player_options["team"] = team_builder
        self.player = ShowdownPlayer(
            obs_builder,
            timing=self._timing,
            battle_format=config.battle_format,
            server_configuration=make_server_configuration(config),
            loop=self._loop,
            **player_options,
        )
        self._assert_player_loop(self.player, "player")
        self.player.set_team_preview_fn(self._do_team_preview)
        self.opponent = self._build_opponent(opponent_factory)

    def reset(self) -> tuple[dict, np.ndarray]:
        """Start a new battle and return the first observation + mask."""
        if self._closed:
            raise RuntimeError("BattleEnv.reset() called after the environment was closed.")

        reset_started = time.perf_counter()
        self._episode_team_preview = None
        self._apply_pending_opponent()
        # Give stateful opponents (e.g. LeaguePlayer) a clean slate for the new battle.
        if hasattr(self.opponent, "reset_hidden"):
            self.opponent.reset_hidden()
        self._battle_task = asyncio.run_coroutine_threadsafe(
            self.player.battle_against(self.opponent, n_battles=1),
            self._loop,
        )
        self._battle_started_at = time.perf_counter()
        obs, mask, _, _, _ = self.player.obs_queue.get()
        self._timing.increment("env_reset_calls")
        self._timing.add("env_reset_total_sec", time.perf_counter() - reset_started)
        return obs, mask

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray, float, bool, dict]:
        """Send an action and get the next observation.

        Args:
            action: (2,) int array of action indices for each active slot.

        Returns:
            obs, mask, reward, done, info
        """
        if self._closed:
            raise RuntimeError("BattleEnv.step() called after the environment was closed.")

        step_started = time.perf_counter()
        enqueue_started = time.perf_counter()
        self.player.action_queue.put(action)
        self._timing.add("env_action_enqueue_sec", time.perf_counter() - enqueue_started)

        wait_started = time.perf_counter()
        obs, mask, reward, done, battle = self.player.obs_queue.get()
        self._timing.add("env_queue_wait_sec", time.perf_counter() - wait_started)
        self._timing.increment("env_step_calls")
        self._timing.add("env_step_total_sec", time.perf_counter() - step_started)
        info = {
            "battle_tag": battle.battle_tag if battle else "",
            "battle_turn": getattr(battle, "turn", 0),
        }
        if done:
            if self._battle_started_at is not None:
                info["episode_sec"] = time.perf_counter() - self._battle_started_at
            if self._episode_team_preview is not None:
                obs_tp, picks_tp = self._episode_team_preview
                info["team_preview_obs"] = obs_tp
                info["team_preview_picks"] = picks_tp
        return obs, mask, reward, done, info

    def _do_team_preview(self, battle: DoubleBattle) -> str:
        """Build the team preview response using the agent callback.

        Called synchronously from ShowdownPlayer.teampreview() at the start of
        each battle. Stores (obs, picks) so they can be returned in the info dict
        when this episode ends, enabling the REINFORCE team preview gradient.
        """
        obs = self.player.obs_builder.build(battle)
        picks, team_str = self._ext_team_preview_fn(obs)
        self._episode_team_preview = (obs, picks)
        return team_str

    @property
    def last_battle(self) -> DoubleBattle | None:
        return self.player.last_battle

    @staticmethod
    def _start_event_loop() -> tuple[asyncio.AbstractEventLoop, Thread]:
        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        loop = asyncio.new_event_loop()
        thread = Thread(target=run_loop, args=(loop,), daemon=True)
        thread.start()
        return loop, thread

    def set_next_opponent(self, opponent_factory: OpponentFactory):
        self._pending_opponent_factory = opponent_factory

    def _apply_pending_opponent(self):
        if self._pending_opponent_factory is None:
            return
        self._disconnect_player(self.opponent, self._loop)
        self.opponent = self._build_opponent(self._pending_opponent_factory)
        self._pending_opponent_factory = None

    def _build_opponent(self, opponent_factory: OpponentFactory) -> Player:
        opponent = opponent_factory(self._loop)
        self._attach_choice_timer(opponent)
        self._assert_player_loop(opponent, "opponent")
        return opponent

    def _attach_choice_timer(self, opponent: Player) -> None:
        original_choose_move = opponent.choose_move

        def timed_choose_move(battle):
            started = time.perf_counter()
            result = original_choose_move(battle)
            self._timing.increment("opponent_choice_calls")
            self._timing.add("opponent_choice_sec", time.perf_counter() - started)
            return result

        opponent.choose_move = timed_choose_move

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

        for participant in (self.player, self.opponent):
            self._disconnect_player(participant, self._loop)

        self._cancel_pending_tasks()
        self.player.obs_queue.clear()
        self.player.action_queue.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if not self._loop.is_running():
            self._loop.close()

        self._battle_task = None
        self._pending_opponent_factory = None
        self._closed = True

    def _cancel_pending_tasks(self) -> None:
        async def _cancel_pending() -> None:
            current_task = asyncio.current_task()
            pending_tasks = [task for task in asyncio.all_tasks() if task is not current_task]
            for task in pending_tasks:
                task.cancel()
            if pending_tasks:
                await asyncio.gather(*pending_tasks, return_exceptions=True)
            await asyncio.get_running_loop().shutdown_asyncgens()

        try:
            asyncio.run_coroutine_threadsafe(_cancel_pending(), self._loop).result(timeout=2.0)
        except (CancelledError, TimeoutError):
            pass

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

    def __init__(self, envs: list[BattleEnv]):
        self.envs = envs
        self.num_envs = len(envs)
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)
        self._timing = _TimingStats()

    def reset(self) -> tuple[dict, np.ndarray]:
        """Reset all environments and return batched obs + masks."""
        started = time.perf_counter()
        results = list(self._executor.map(lambda env: env.reset(), self.envs))
        self._timing.increment("parallel_reset_calls")
        self._timing.add("parallel_reset_wall_sec", time.perf_counter() - started)
        return self._collate([r[0] for r in results], [r[1] for r in results])

    def step(
        self, actions: np.ndarray
    ) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
        """Step all environments.

        Args:
            actions: (num_envs, 2) action array.

        Returns:
            obs_batch, mask_batch, rewards, dones, infos
        """
        step_started = time.perf_counter()
        results = list(
            self._executor.map(
                self._step_env,
                self.envs,
                actions,
            )
        )
        step_wait_sec = time.perf_counter() - step_started
        obs_list = [result[0] for result in results]
        mask_list = [result[1] for result in results]
        rewards = [result[2] for result in results]
        dones = [result[3] for result in results]
        infos = [result[4] for result in results]

        collate_started = time.perf_counter()
        batched_obs, batched_mask = self._collate(obs_list, mask_list)
        self._timing.increment("parallel_step_calls")
        self._timing.add("parallel_step_wait_sec", step_wait_sec)
        self._timing.add("parallel_collate_sec", time.perf_counter() - collate_started)
        self._timing.add("parallel_step_wall_sec", time.perf_counter() - step_started)
        return (
            batched_obs,
            batched_mask,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
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

    @staticmethod
    def _collate(
        obs_list: list[dict], mask_list: list[np.ndarray]
    ) -> tuple[dict, np.ndarray]:
        """Stack observations into batched tensors."""
        keys = obs_list[0].keys()
        batched = {k: np.stack([o[k] for o in obs_list]) for k in keys}
        masks = np.stack(mask_list)
        return batched, masks

    @staticmethod
    def _step_env(
        env: BattleEnv, action: np.ndarray
    ) -> tuple[dict, np.ndarray, float, bool, dict]:
        obs, mask, reward, done, info = env.step(action)
        if done:
            obs, mask = env.reset()
            info["episode_reward"] = reward
        return obs, mask, reward, done, info
