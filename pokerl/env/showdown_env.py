"""Gymnasium-compatible wrapper around poke-env for doubles battles."""

import asyncio
import itertools
import numpy as np
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Thread
from threading import Lock
from queue import Queue

from poke_env import AccountConfiguration
from poke_env.battle import DoubleBattle
from poke_env.player import BattleOrder, DoubleBattleOrder, PassBattleOrder, Player

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


def make_local_account_configuration(prefix: str) -> AccountConfiguration:
    """Create a short, unique local showdown username for this process."""
    clean_prefix = "".join(ch for ch in prefix.lower() if ch.isalnum())[:10] or "bot"
    with _ACCOUNT_LOCK:
        suffix = f"{_ACCOUNT_RUN_TAG}{next(_ACCOUNT_COUNTER):04x}"
    return AccountConfiguration(f"{clean_prefix}{suffix}"[:18], None)


class ShowdownPlayer(Player):
    """A poke-env player that bridges to our RL agent via queues.

    Instead of choosing moves itself, it sends observations to a queue
    and waits for actions from another queue. This lets us use poke-env's
    async battle loop from a synchronous Gymnasium-style step() function.
    """

    def __init__(self, obs_builder: ObservationBuilder, **kwargs):
        super().__init__(**kwargs)
        self.obs_builder = obs_builder
        self.obs_queue: Queue = Queue()
        self.action_queue: Queue = Queue()
        self.last_battle: DoubleBattle | None = None
        self._last_finished_token: tuple[str, int] | None = None
        self._team_preview_fn = None

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        self._emit_battle_update(battle)

        if battle.finished:
            return self.choose_default_move()

        # Block until the RL agent sends an action
        action = self.action_queue.get()
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
        env = BattleEnv(config, game_data, opponent)
        obs, mask = env.reset()
        while not done:
            obs, mask, reward, done, info = env.step(action)
    """

    def __init__(
        self,
        config: EnvConfig,
        game_data: GameData,
        opponent: Player,
        player_kwargs: dict | None = None,
    ):
        self.config = config
        self.game_data = game_data
        self.opponent = opponent
        self._pending_opponent: Player | None = None

        obs_builder = ObservationBuilder(game_data, config.team_size)
        player_options = dict(player_kwargs or {})
        player_options.setdefault(
            "account_configuration",
            make_local_account_configuration("agent"),
        )
        self.player = ShowdownPlayer(
            obs_builder,
            battle_format=config.battle_format,
            server_configuration=self._server_config(),
            **player_options,
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None
        self._battle_task = None

    def reset(self) -> tuple[dict, np.ndarray]:
        """Start a new battle and return the first observation + mask."""
        self._apply_pending_opponent()
        self._ensure_event_loop()
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
        self.player.action_queue.put(action)

        obs, mask, reward, done, battle = self.player.obs_queue.get()
        info = {"battle_tag": battle.battle_tag if battle else ""}
        return obs, mask, reward, done, info

    @property
    def last_battle(self) -> DoubleBattle | None:
        return self.player.last_battle

    def _ensure_event_loop(self):
        if self._loop is not None:
            return

        def run_loop(loop):
            asyncio.set_event_loop(loop)
            loop.run_forever()

        self._loop = asyncio.new_event_loop()
        self._thread = Thread(target=run_loop, args=(self._loop,), daemon=True)
        self._thread.start()

    def _server_config(self):
        from poke_env import ServerConfiguration
        host, port = self.config.server_url.split(":")
        return ServerConfiguration(
            f"ws://{host}:{port}/showdown/websocket",
            f"http://{host}:{port}/action.php",
        )

    def set_next_opponent(self, opponent: Player):
        if self._pending_opponent is not None:
            self._disconnect_player(self._pending_opponent)
        self._pending_opponent = opponent

    def _apply_pending_opponent(self):
        if self._pending_opponent is None:
            return
        self._disconnect_player(self.opponent)
        self.opponent = self._pending_opponent
        self._pending_opponent = None

    def close(self):
        if self._battle_task is not None and not self._battle_task.done():
            self._battle_task.cancel()

        for participant in (self.player, self.opponent):
            self._disconnect_player(participant)
        if self._pending_opponent is not None:
            self._disconnect_player(self._pending_opponent)

        while not self.player.obs_queue.empty():
            self.player.obs_queue.get_nowait()
        while not self.player.action_queue.empty():
            self.player.action_queue.get_nowait()

        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread is not None and self._thread.is_alive():
                self._thread.join(timeout=1.0)
            if not self._loop.is_running():
                self._loop.close()

        self._battle_task = None
        self._loop = None
        self._thread = None
        self._pending_opponent = None

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
        self._executor.shutdown(wait=False, cancel_futures=True)

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
