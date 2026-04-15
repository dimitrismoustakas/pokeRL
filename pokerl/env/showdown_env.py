"""Gymnasium-compatible wrapper around poke-env for doubles battles."""

import asyncio
import itertools
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

    def __init__(self, obs_builder: ObservationBuilder, **kwargs):
        super().__init__(**kwargs)
        self.obs_builder = obs_builder
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
        """Pick 4 pokemon from the team preview.

        Handled internally with a heuristic -- not routed through the
        training loop. Learned team preview can be added later by
        passing a callback via set_team_preview_fn().
        """
        if self._team_preview_fn is not None:
            return self._team_preview_fn(battle)
        # Default: bring first 4 in order
        return "/team 1234"

    def set_team_preview_fn(self, fn):
        """Optional: override team preview with a custom function."""
        self._team_preview_fn = fn

    def request_close(self) -> None:
        self._closing = True
        self.action_queue.put(None)

    def _make_order(self, action: np.ndarray, battle: DoubleBattle) -> BattleOrder:
        """Convert a (2,) action array to a DoubleBattleOrder."""
        orders = []
        for slot in range(2):
            order = action_to_order(int(action[slot]), battle, slot)
            orders.append(order)

        if all(isinstance(order, PassBattleOrder) for order in orders):
            return self.choose_default_move()
        return DoubleBattleOrder(orders[0], orders[1])

    def _emit_battle_update(self, battle: DoubleBattle) -> None:
        self.last_battle = battle
        finished_token = (battle.battle_tag, battle.turn)
        if battle.finished and finished_token == self._last_finished_token:
            return

        obs = self.obs_builder.build(battle)
        mask = np.stack([build_action_mask(battle, s) for s in range(2)])
        reward = sparse_reward(battle)
        done = battle.finished

        self.obs_queue.put((obs, mask, reward, done, battle))
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
        player_kwargs: dict | None = None,
    ):
        self.config = config
        self.game_data = game_data
        self._battle_task = None
        self._loop, self._thread = self._start_event_loop()
        self._pending_opponent_factory: OpponentFactory | None = None
        self._closed = False

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
        self.player = ShowdownPlayer(
            obs_builder,
            battle_format=config.battle_format,
            server_configuration=make_server_configuration(config),
            loop=self._loop,
            **player_options,
        )
        self._assert_player_loop(self.player, "player")
        self.opponent = self._build_opponent(opponent_factory)

    def reset(self) -> tuple[dict, np.ndarray]:
        """Start a new battle and return the first observation + mask."""
        if self._closed:
            raise RuntimeError("BattleEnv.reset() called after the environment was closed.")

        self._apply_pending_opponent()
        self._battle_task = asyncio.run_coroutine_threadsafe(
            self.player.battle_against(self.opponent, n_battles=1),
            self._loop,
        )
        obs, mask, _, _, _ = self.player.obs_queue.get()
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

        self.player.action_queue.put(action)

        obs, mask, reward, done, battle = self.player.obs_queue.get()
        info = {"battle_tag": battle.battle_tag if battle else ""}
        return obs, mask, reward, done, info

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
        self._disconnect_player(self.opponent)
        self.opponent = self._build_opponent(self._pending_opponent_factory)
        self._pending_opponent_factory = None

    def _build_opponent(self, opponent_factory: OpponentFactory) -> Player:
        opponent = opponent_factory(self._loop)
        self._assert_player_loop(opponent, "opponent")
        return opponent

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
            self._disconnect_player(participant)

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
    def _disconnect_player(player: Player):
        ps_client = getattr(player, "ps_client", None)
        if ps_client is None or not hasattr(ps_client, "websocket"):
            return

        try:
            asyncio.run(asyncio.wait_for(ps_client.stop_listening(), timeout=1.0))
        except Exception:
            pass


class ParallelBattleEnv:
    """Runs multiple BattleEnvs and collects observations in batch.

    Each env runs its own battle. step() sends actions to all envs
    and collects results. Environments that finish are auto-reset.
    """

    def __init__(self, envs: list[BattleEnv]):
        self.envs = envs
        self.num_envs = len(envs)
        self._executor = ThreadPoolExecutor(max_workers=self.num_envs)

    def reset(self) -> tuple[dict, np.ndarray]:
        """Reset all environments and return batched obs + masks."""
        results = list(self._executor.map(lambda env: env.reset(), self.envs))
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
        results = list(
            self._executor.map(
                self._step_env,
                self.envs,
                actions,
            )
        )
        obs_list = [result[0] for result in results]
        mask_list = [result[1] for result in results]
        rewards = [result[2] for result in results]
        dones = [result[3] for result in results]
        infos = [result[4] for result in results]

        batched_obs, batched_mask = self._collate(obs_list, mask_list)
        return (
            batched_obs,
            batched_mask,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

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
