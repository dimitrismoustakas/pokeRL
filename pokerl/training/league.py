"""Self-play league: a pool of past agent checkpoints as opponents."""

import asyncio
import copy
import json
import random
import time
from collections import defaultdict, deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread

import numpy as np
import torch
from poke_env.battle import DoubleBattle
from poke_env.player import DoubleBattleOrder, PassBattleOrder, Player

from ..model.agent import PokeAgent
from ..data import GameData
from ..config import TrainingConfig, EnvConfig
from ..env.observation import ObservationBuilder
from ..env.actions import build_action_mask, action_to_order
from ..env.showdown_env import make_local_account_configuration


@dataclass
class _InferenceRequest:
    checkpoint_path: str
    kind: str
    obs: dict[str, np.ndarray]
    masks: np.ndarray | None
    hidden: tuple[torch.Tensor, torch.Tensor] | None
    future: Future


class _LeagueInferenceService:
    """Shared batched inference for league opponents.

    A single worker thread batches requests per checkpoint and runs them on the
    configured device. This preserves the opponent policy while avoiding one
    tiny GPU forward per environment thread.
    """

    _COALESCE_WINDOW_SEC = 0.001

    def __init__(self, agent_template: PokeAgent, device: torch.device):
        self._agent_template = agent_template
        self._device = device
        self._models: dict[str, PokeAgent] = {}
        self._pending: list[_InferenceRequest] = []
        self._closed = False
        self._condition = Condition()
        self._thread = Thread(target=self._run, name="league-inference", daemon=True)
        self._thread.start()

    def ensure_checkpoint(self, checkpoint_path: str) -> None:
        with self._condition:
            if checkpoint_path in self._models:
                return

        checkpoint = torch.load(checkpoint_path, map_location=self._device, weights_only=False)
        agent_copy = copy.deepcopy(self._agent_template)
        agent_copy.load_state_dict(checkpoint["agent"])
        agent_copy.to(self._device)
        agent_copy.eval()
        agent_copy.requires_grad_(False)

        with self._condition:
            self._models.setdefault(checkpoint_path, agent_copy)

    def prune_checkpoints(self, keep_paths: set[str]) -> None:
        with self._condition:
            stale_paths = [path for path in self._models if path not in keep_paths]
            for path in stale_paths:
                del self._models[path]

    def infer_action(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
        masks: np.ndarray,
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor] | None]:
        future: Future = Future()
        self._submit(
            _InferenceRequest(
                checkpoint_path=checkpoint_path,
                kind="battle",
                obs=obs,
                masks=masks,
                hidden=hidden,
                future=future,
            )
        )
        return future.result()

    def infer_team_preview(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        future: Future = Future()
        self._submit(
            _InferenceRequest(
                checkpoint_path=checkpoint_path,
                kind="preview",
                obs=obs,
                masks=None,
                hidden=None,
                future=future,
            )
        )
        return future.result()

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closed = True
            pending = list(self._pending)
            self._pending.clear()
            self._condition.notify_all()

        for request in pending:
            request.future.set_exception(RuntimeError("League inference service closed."))

        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

        with self._condition:
            self._models.clear()

    def _submit(self, request: _InferenceRequest) -> None:
        self.ensure_checkpoint(request.checkpoint_path)
        with self._condition:
            if self._closed:
                raise RuntimeError("League inference service is closed.")
            self._pending.append(request)
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while not self._pending and not self._closed:
                    self._condition.wait()
                if self._closed and not self._pending:
                    return
                if self._COALESCE_WINDOW_SEC > 0:
                    deadline = time.perf_counter() + self._COALESCE_WINDOW_SEC
                    while not self._closed:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                batch = self._pending
                self._pending = []

            grouped: dict[tuple[str, str], list[_InferenceRequest]] = defaultdict(list)
            for request in batch:
                grouped[(request.checkpoint_path, request.kind)].append(request)

            for (checkpoint_path, kind), requests in grouped.items():
                try:
                    if kind == "battle":
                        self._run_battle_batch(checkpoint_path, requests)
                    elif kind == "preview":
                        self._run_preview_batch(checkpoint_path, requests)
                    else:
                        raise ValueError(f"Unknown inference request kind: {kind}")
                except Exception as exc:
                    for request in requests:
                        request.future.set_exception(exc)

    def _run_battle_batch(self, checkpoint_path: str, requests: list[_InferenceRequest]) -> None:
        agent = self._get_agent(checkpoint_path)
        obs_batch = {
            key: torch.from_numpy(np.stack([request.obs[key] for request in requests])).to(self._device)
            for key in requests[0].obs
        }
        masks_batch = torch.from_numpy(np.stack([request.masks for request in requests])).to(self._device).bool()
        hidden_batch = self._build_hidden_batch(agent, requests)

        with torch.inference_mode():
            out = agent(obs_batch, hidden_batch, action_masks=masks_batch)

        actions = out["actions"].cpu().numpy()
        new_hidden = out["hidden"]
        for index, request in enumerate(requests):
            request.future.set_result((actions[index], self._slice_hidden(new_hidden, index)))

    def _run_preview_batch(self, checkpoint_path: str, requests: list[_InferenceRequest]) -> None:
        agent = self._get_agent(checkpoint_path)
        obs_batch = {
            key: torch.from_numpy(np.stack([request.obs[key] for request in requests])).to(self._device)
            for key in requests[0].obs
        }

        with torch.inference_mode():
            picks = agent.team_preview(obs_batch)["picks"].cpu().numpy()

        for index, request in enumerate(requests):
            request.future.set_result(picks[index])

    def _get_agent(self, checkpoint_path: str) -> PokeAgent:
        with self._condition:
            agent = self._models.get(checkpoint_path)
        if agent is None:
            self.ensure_checkpoint(checkpoint_path)
            with self._condition:
                agent = self._models[checkpoint_path]
        return agent

    def _build_hidden_batch(
        self,
        agent: PokeAgent,
        requests: list[_InferenceRequest],
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        hidden_batch = agent.initial_hidden(len(requests))
        if hidden_batch is None:
            return None

        h_batch, c_batch = hidden_batch
        for index, request in enumerate(requests):
            if request.hidden is None:
                continue
            h_req, c_req = request.hidden
            h_batch[:, index : index + 1].copy_(h_req.to(self._device))
            c_batch[:, index : index + 1].copy_(c_req.to(self._device))
        return h_batch, c_batch

    @staticmethod
    def _slice_hidden(
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if hidden is None:
            return None
        h, c = hidden
        return (
            h[:, index : index + 1].detach().cpu(),
            c[:, index : index + 1].detach().cpu(),
        )


class LeaguePlayer(Player):
    """A poke-env player that uses a frozen agent checkpoint to choose moves.

    This is the opponent in self-play: it loads a past version of the agent
    and plays battles autonomously (no queue bridging needed).
    """

    def __init__(
        self,
        checkpoint_path: str,
        inference_service: _LeagueInferenceService,
        obs_builder: ObservationBuilder,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.checkpoint_path = checkpoint_path
        self.inference_service = inference_service
        self.obs_builder = obs_builder
        self._hidden: tuple[torch.Tensor, torch.Tensor] | None = None

    def choose_move(self, battle: DoubleBattle):
        obs = self.obs_builder.build(battle)
        masks = np.stack([build_action_mask(battle, s) for s in range(2)])
        actions, self._hidden = self.inference_service.infer_action(
            self.checkpoint_path,
            obs,
            masks,
            self._hidden,
        )

        orders = []
        for slot in range(2):
            orders.append(action_to_order(int(actions[slot]), battle, slot))

        if all(isinstance(order, PassBattleOrder) for order in orders):
            return self.choose_default_move()
        return DoubleBattleOrder(orders[0], orders[1])

    def teampreview(self, battle: DoubleBattle) -> str:
        obs = self.obs_builder.build(battle)
        picks = self.inference_service.infer_team_preview(self.checkpoint_path, obs)
        return "/team " + "".join(str(int(p) + 1) for p in picks)

    def reset_hidden(self):
        """Clear LSTM state between battles. Called by BattleEnv.reset()."""
        self._hidden = None


class League:
    """Manages a pool of opponent checkpoints for self-play training.

    Opponents are sampled from a rolling window of past checkpoints, with a
    bias towards recent ones (70%) over older ones (30%) to track current skill
    while retaining some diversity.
    """

    def __init__(
        self,
        config: TrainingConfig,
        env_config: EnvConfig,
        game_data: GameData,
        agent_template: PokeAgent,
        device: torch.device,
    ):
        self.config = config
        self.env_config = env_config
        self.game_data = game_data
        self.agent_template = agent_template
        self.device = device
        self.checkpoints: deque[str] = deque(maxlen=config.league_size)
        self._inference = _LeagueInferenceService(agent_template, device)

    def add_checkpoint(self, agent: PokeAgent, step: int):
        """Snapshot the current agent as a new league member."""
        path = f"{self.config.checkpoint_dir}/league/step_{step}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"agent": agent.state_dict()}, path)
        self.checkpoints.append(path)
        self._inference.prune_checkpoints(set(self.checkpoints))

    def make_opponent(
        self,
        loop: asyncio.AbstractEventLoop,
        server_url: str | None = None,
    ) -> "LeaguePlayer | None":
        """Create a LeaguePlayer from a sampled checkpoint.

        Args:
            loop: The asyncio event loop the opponent will run on.
            server_url: The Showdown server URL to connect to. Must match the URL
                used by the paired player so both are on the same server. Defaults
                to the league's own env_config.server_url if not provided.

        Returns None if the league is empty.
        """
        path = self._sample_checkpoint()
        if path is None:
            return None
        self._inference.ensure_checkpoint(path)

        url = server_url if server_url is not None else self.env_config.server_url
        host, port = url.split(":")
        from poke_env import ServerConfiguration
        server_config = ServerConfiguration(
            f"ws://{host}:{port}/showdown/websocket",
            f"http://{host}:{port}/action.php",
        )

        obs_builder = ObservationBuilder(self.game_data, self.env_config.team_size)
        return LeaguePlayer(
            path,
            self._inference,
            obs_builder,
            account_configuration=make_local_account_configuration("league"),
            battle_format=self.env_config.battle_format,
            server_configuration=server_config,
            loop=loop,
        )

    def save_state(self, path: str) -> None:
        """Persist the ordered list of checkpoint paths to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(list(self.checkpoints), f)

    def load_state(self, path: str) -> None:
        """Restore checkpoint paths from a saved state file.

        Skips any paths that no longer exist on disk and warns if any are missing.
        """
        with open(path) as f:
            paths: list[str] = json.load(f)

        existing = [p for p in paths if Path(p).exists()]
        missing = len(paths) - len(existing)
        if missing:
            print(f"Warning: {missing} league checkpoint(s) no longer on disk, skipping.")

        for p in existing[-self.config.league_size:]:
            self.checkpoints.append(p)
        self._inference.prune_checkpoints(set(self.checkpoints))

    def close(self) -> None:
        self._inference.close()

    def _sample_checkpoint(self) -> str | None:
        """Sample a checkpoint path. 70% recent half, 30% older half."""
        if not self.checkpoints:
            return None

        checkpoints = list(self.checkpoints)
        mid = len(checkpoints) // 2

        if len(checkpoints) < 2 or random.random() < 0.7:
            recent = checkpoints[mid:] if mid < len(checkpoints) else checkpoints
            return random.choice(recent)
        else:
            older = checkpoints[:mid] if mid > 0 else checkpoints
            return random.choice(older)
