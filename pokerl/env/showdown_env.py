"""Gymnasium-compatible wrapper around poke-env for doubles battles."""

import asyncio
import numpy as np
import torch
from threading import Thread
from queue import Queue, Empty

from poke_env.player import Player, BattleOrder
from poke_env.environment import DoubleBattle

from ..config import EnvConfig
from ..data import GameData
from .observation import ObservationBuilder
from .actions import (
    ACTIONS_PER_SLOT, PASS_ACTION,
    build_action_mask, action_to_order,
)
from .rewards import sparse_reward


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

    def choose_move(self, battle: DoubleBattle) -> BattleOrder:
        obs = self.obs_builder.build(battle)
        mask = np.stack([build_action_mask(battle, s) for s in range(2)])
        reward = sparse_reward(battle)
        done = battle.finished

        self.obs_queue.put((obs, mask, reward, done, battle))

        if done:
            return self.choose_default_move()

        # Block until the RL agent sends an action
        action = self.action_queue.get()
        return self._make_order(action, battle)

    def teampreview(self, battle: DoubleBattle) -> str:
        """For now, return the team order from the action queue.

        The RL agent can send a team order as a list of indices [0-5].
        """
        obs = self.obs_builder.build(battle)
        mask = np.ones((2, ACTIONS_PER_SLOT), dtype=bool)  # dummy mask for preview
        self.obs_queue.put((obs, mask, 0.0, False, battle))

        team_order = self.action_queue.get()
        if isinstance(team_order, (list, np.ndarray)):
            return "/team " + "".join(str(i + 1) for i in team_order)
        return "/team 1234"

    def _make_order(self, action: np.ndarray, battle: DoubleBattle) -> BattleOrder:
        """Convert a (2,) action array to a DoubleBattleOrder."""
        from poke_env.player import DoubleBattleOrder

        orders = []
        for slot in range(2):
            order = action_to_order(int(action[slot]), battle, slot)
            orders.append(order)

        # DoubleBattleOrder expects (first_order, second_order)
        if orders[0] is None and orders[1] is None:
            return self.choose_default_move()
        return DoubleBattleOrder(orders[0], orders[1])


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

        obs_builder = ObservationBuilder(game_data, config.team_size)
        self.player = ShowdownPlayer(
            obs_builder,
            battle_format=config.battle_format,
            server_configuration=self._server_config(),
            **(player_kwargs or {}),
        )

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: Thread | None = None

    def reset(self) -> tuple[dict, np.ndarray]:
        """Start a new battle and return the first observation + mask."""
        self._ensure_event_loop()
        asyncio.run_coroutine_threadsafe(
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


class ParallelBattleEnv:
    """Runs multiple BattleEnvs and collects observations in batch.

    Each env runs its own battle. step() sends actions to all envs
    and collects results. Environments that finish are auto-reset.
    """

    def __init__(self, envs: list[BattleEnv]):
        self.envs = envs
        self.num_envs = len(envs)

    def reset(self) -> tuple[dict, np.ndarray]:
        """Reset all environments and return batched obs + masks."""
        results = [env.reset() for env in self.envs]
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
        obs_list, mask_list, rewards, dones, infos = [], [], [], [], []

        for i, env in enumerate(self.envs):
            obs, mask, reward, done, info = env.step(actions[i])

            if done:
                # Auto-reset
                obs, mask = env.reset()
                info["episode_reward"] = reward

            obs_list.append(obs)
            mask_list.append(mask)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)

        batched_obs, batched_mask = self._collate(obs_list, mask_list)
        return (
            batched_obs,
            batched_mask,
            np.array(rewards, dtype=np.float32),
            np.array(dones, dtype=bool),
            infos,
        )

    @staticmethod
    def _collate(
        obs_list: list[dict], mask_list: list[np.ndarray]
    ) -> tuple[dict, np.ndarray]:
        """Stack observations into batched tensors."""
        keys = obs_list[0].keys()
        batched = {k: np.stack([o[k] for o in obs_list]) for k in keys}
        masks = np.stack(mask_list)
        return batched, masks
