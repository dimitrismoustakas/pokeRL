"""Self-play league: a pool of past agent checkpoints as opponents."""

import asyncio
import copy
import json
import random
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from threading import Condition, Thread

import numpy as np
import torch
from poke_env.battle import DoubleBattle
from poke_env.player import DoubleBattleOrder, PassBattleOrder, Player

from ..model.agent import PokeAgent, load_state_dict_compatible
from ..data import GameData
from ..config import TrainingConfig, EnvConfig
from ..env.observation import ObservationBuilder, build_team_preview_observation
from ..env.actions import (
    action_to_order,
    build_legal_action_state,
    build_policy_action_masks,
    decode_tera_choice,
)
from ..env.showdown_env import (
    make_battlestream_configuration,
    make_local_account_configuration,
    showdown_log_level,
)


@dataclass
class _InferenceRequest:
    checkpoint_path: str
    kind: str
    obs: dict[str, np.ndarray]
    masks: dict[str, np.ndarray] | None
    hidden: tuple[torch.Tensor, torch.Tensor] | None
    future: Future
    timing: object | None = None


class _LeagueInferenceService:
    """Shared batched inference for league opponents.

    A single worker thread batches requests per checkpoint and runs them on the
    configured device. This preserves the opponent policy while avoiding one
    tiny GPU forward per environment thread.
    """

    def __init__(
        self,
        agent_template: PokeAgent,
        device: torch.device,
        coalesce_window_sec: float,
    ):
        self._agent_template = agent_template
        self._device = device
        self._coalesce_window_sec = max(0.0, coalesce_window_sec)
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
        agent_copy.to(self._device)
        load_state_dict_compatible(agent_copy, checkpoint["agent"])
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
        masks: dict[str, np.ndarray],
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        timing: object | None = None,
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
                timing=timing,
            )
        )
        return future.result()

    def infer_team_preview(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
    ) -> np.ndarray:
        return self.submit_team_preview(checkpoint_path, obs).result()

    def submit_team_preview(
        self,
        checkpoint_path: str,
        obs: dict[str, np.ndarray],
    ) -> Future:
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
                if self._coalesce_window_sec > 0:
                    deadline = time.perf_counter() + self._coalesce_window_sec
                    while not self._closed:
                        remaining = deadline - time.perf_counter()
                        if remaining <= 0:
                            break
                        self._condition.wait(timeout=remaining)
                batch = self._pending
                self._pending = []

            grouped: dict[tuple[str, str], list[_InferenceRequest]] = {}
            for request in batch:
                key = (request.checkpoint_path, request.kind)
                grouped.setdefault(key, []).append(request)

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
        batch_size = len(requests)
        for request in requests:
            self._increment_timing(request.timing, "opponent_infer_action_requests")
            self._add_timing(request.timing, "opponent_infer_action_batches", 1.0 / batch_size)
            self._add_timing(request.timing, "opponent_infer_action_batch_size_sum", float(batch_size))

        stack_started = time.perf_counter()
        obs_batch = {
            key: torch.from_numpy(np.stack([request.obs[key] for request in requests])).to(self._device)
            for key in requests[0].obs
        }
        if requests[0].masks is None:
            raise RuntimeError("Battle inference request was missing action masks.")
        masks_batch = {
            key: torch.from_numpy(np.stack([request.masks[key] for request in requests])).to(self._device).bool()
            for key in requests[0].masks
        }
        self._add_batch_timing(
            requests,
            "opponent_infer_action_stack_to_device_sec",
            time.perf_counter() - stack_started,
        )

        hidden_started = time.perf_counter()
        hidden_batch = self._build_hidden_batch(agent, requests)
        self._add_batch_timing(
            requests,
            "opponent_infer_action_hidden_batch_sec",
            time.perf_counter() - hidden_started,
        )

        forward_started = time.perf_counter()
        with torch.inference_mode():
            out = agent(
                obs_batch,
                hidden_batch,
                action_masks=masks_batch,
                include_value=False,
                include_policy_stats=False,
            )
        self._add_batch_timing(
            requests,
            "opponent_infer_action_forward_launch_sec",
            time.perf_counter() - forward_started,
        )

        output_started = time.perf_counter()
        actions = out["actions"].cpu().numpy()
        new_hidden = out["hidden"]
        self._add_batch_timing(
            requests,
            "opponent_infer_action_output_sync_sec",
            time.perf_counter() - output_started,
        )

        result_started = time.perf_counter()
        for index, request in enumerate(requests):
            request.future.set_result((actions[index], self._slice_hidden(new_hidden, index)))
        self._add_batch_timing(
            requests,
            "opponent_infer_action_set_result_sec",
            time.perf_counter() - result_started,
        )

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
        if all(request.hidden is not None for request in requests):
            return (
                torch.cat(
                    [self._to_inference_device(request.hidden[0]) for request in requests],
                    dim=1,
                ),
                torch.cat(
                    [self._to_inference_device(request.hidden[1]) for request in requests],
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

    def _add_batch_timing(
        self,
        requests: list[_InferenceRequest],
        key: str,
        value: float,
    ) -> None:
        per_request = value / len(requests)
        for request in requests:
            self._add_timing(request.timing, key, per_request)

    @staticmethod
    def _add_timing(timing, key: str, value: float) -> None:
        if timing is not None:
            timing.add(key, value)

    @staticmethod
    def _increment_timing(timing, key: str, delta: int = 1) -> None:
        if timing is not None:
            timing.increment(key, delta)

    @staticmethod
    def _slice_hidden(
        hidden: tuple[torch.Tensor, torch.Tensor] | None,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        if hidden is None:
            return None
        h, c = hidden
        return (
            h[:, index : index + 1].detach(),
            c[:, index : index + 1].detach(),
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
        self._pending_hidden: tuple[torch.Tensor, torch.Tensor] | None = None
        self._pending_decision_key: tuple[object, ...] | None = None
        self._pending_checkpoint_path: str | None = None
        self._prepared_preview_obs: dict[str, np.ndarray] | None = None
        self._prepared_preview_future: Future | None = None
        self._timing = None
        # Most-recent (a1, a2, tera) chosen by this opponent. Read by the env
        # after each turn resolves so the aux opponent-action head has labels.
        self._last_actions: np.ndarray | None = None

    def stage_checkpoint(self, checkpoint_path: str) -> None:
        self._pending_checkpoint_path = checkpoint_path
        self._prepared_preview_obs = None
        self._prepared_preview_future = None

    def apply_staged_checkpoint(self) -> bool:
        if self._pending_checkpoint_path is None:
            return False
        self.checkpoint_path = self._pending_checkpoint_path
        self._pending_checkpoint_path = None
        return True

    def set_timing(self, timing) -> None:
        self._timing = timing

    def prepare_team_preview(self, own_packed_team: str, opponent_packed_team: str) -> None:
        timing_enabled = LeaguePlayer._timing_enabled(self)
        if timing_enabled:
            started = time.perf_counter()
        obs = build_team_preview_observation(
            self.obs_builder.gd,
            self.obs_builder.team_size,
            own_packed_team,
            opponent_packed_team,
        )
        if timing_enabled:
            self._add_timing("opponent_precompute_preview_obs_sec", time.perf_counter() - started)

        if timing_enabled:
            submit_started = time.perf_counter()
        self._prepared_preview_obs = obs
        checkpoint_path = self._pending_checkpoint_path or self.checkpoint_path
        self._prepared_preview_future = self.inference_service.submit_team_preview(
            checkpoint_path,
            obs,
        )
        if timing_enabled:
            self._add_timing("opponent_precompute_preview_submit_sec", time.perf_counter() - submit_started)
            self._increment_timing("opponent_precompute_preview_submit_calls")

    def choose_move(self, battle: DoubleBattle):
        timing_enabled = LeaguePlayer._timing_enabled(self)
        if timing_enabled:
            choice_started = time.perf_counter()
            key_started = time.perf_counter()
        decision_key = self._decision_key(battle)
        if timing_enabled:
            self._add_timing("opponent_decision_key_sec", time.perf_counter() - key_started)

        if timing_enabled:
            hidden_started = time.perf_counter()
        self._commit_hidden_if_battle_advanced(decision_key)
        if timing_enabled:
            self._add_timing("opponent_hidden_commit_sec", time.perf_counter() - hidden_started)
            obs_started = time.perf_counter()
        obs = self.obs_builder.build(battle)
        if timing_enabled:
            self._add_timing("opponent_obs_build_sec", time.perf_counter() - obs_started)
            masks_started = time.perf_counter()
        action_state = build_legal_action_state(battle)
        masks = build_policy_action_masks(battle, action_state=action_state)
        if timing_enabled:
            self._add_timing("opponent_mask_build_sec", time.perf_counter() - masks_started)
            infer_started = time.perf_counter()
        actions, next_hidden = self.inference_service.infer_action(
            self.checkpoint_path,
            obs,
            masks,
            self._hidden,
            timing=self._timing if timing_enabled else None,
        )
        if timing_enabled:
            self._add_timing("opponent_infer_action_wait_sec", time.perf_counter() - infer_started)
        self._pending_decision_key = decision_key
        self._pending_hidden = next_hidden
        self._last_actions = np.asarray(actions, dtype=np.int64).copy()

        if timing_enabled:
            order_started = time.perf_counter()
        orders = []
        tera_slot_1, tera_slot_2 = decode_tera_choice(int(actions[2]))
        for slot in range(2):
            orders.append(
                action_to_order(
                    int(actions[slot]),
                    battle,
                    slot,
                    terastallize=tera_slot_1 if slot == 0 else tera_slot_2,
                    action_state=action_state,
                )
            )
        if all(isinstance(order, PassBattleOrder) for order in orders):
            result = self.choose_default_move()
        else:
            result = DoubleBattleOrder(orders[0], orders[1])
        if timing_enabled:
            self._add_timing("opponent_order_build_sec", time.perf_counter() - order_started)
            self._add_timing("opponent_choose_move_internal_sec", time.perf_counter() - choice_started)
            self._increment_timing("opponent_choose_move_internal_calls")
        return result

    def teampreview(self, battle: DoubleBattle) -> str:
        obs = self.obs_builder.build(battle)
        prepared_obs = self._prepared_preview_obs
        prepared_future = self._prepared_preview_future
        self._prepared_preview_obs = None
        self._prepared_preview_future = None
        timing_enabled = LeaguePlayer._timing_enabled(self)

        if prepared_obs is None and prepared_future is None:
            if timing_enabled:
                infer_started = time.perf_counter()
            picks = self.inference_service.infer_team_preview(self.checkpoint_path, obs)
            if timing_enabled:
                self._add_timing("opponent_teampreview_infer_sec", time.perf_counter() - infer_started)
            return "/team " + "".join(str(int(p) + 1) for p in picks)

        if prepared_obs is None or prepared_future is None:
            self._increment_timing("opponent_precompute_preview_missing_state")
            raise RuntimeError("Prepared opponent team preview state was partially missing.")

        if self._obs_equal(obs, prepared_obs):
            if timing_enabled:
                self._increment_timing("opponent_precompute_preview_hits")
                wait_started = time.perf_counter()
            picks = prepared_future.result()
            if timing_enabled:
                self._add_timing(
                    "opponent_precompute_preview_hit_wait_sec",
                    time.perf_counter() - wait_started,
                )
            return "/team " + "".join(str(int(p) + 1) for p in picks)

        if timing_enabled:
            self._increment_timing("opponent_precompute_preview_misses")
        self._record_preview_obs_mismatch(obs, prepared_obs)
        raise RuntimeError("Prepared opponent team preview observation did not match the live battle state.")

    def reset_hidden(self):
        """Clear LSTM state between battles. Called by BattleEnv.reset()."""
        self._hidden = None
        self._pending_hidden = None
        self._pending_decision_key = None
        self._last_actions = None

    @staticmethod
    def _obs_equal(left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> bool:
        if left.keys() != right.keys():
            return False
        return all(np.array_equal(left[key], right[key]) for key in left)

    def _commit_hidden_if_battle_advanced(self, decision_key: tuple[object, ...]) -> None:
        if self._pending_decision_key is None:
            return
        if decision_key == self._pending_decision_key:
            return
        self._hidden = self._pending_hidden
        self._pending_hidden = None
        self._pending_decision_key = None

    @classmethod
    def _decision_key(cls, battle: DoubleBattle) -> tuple[object, ...]:
        return (
            getattr(battle, "battle_tag", ""),
            int(getattr(battle, "turn", 0)),
            cls._request_decision_signature(getattr(battle, "last_request", None)),
            cls._simple_decision_tuple(getattr(battle, "force_switch", None)),
            cls._simple_decision_tuple(getattr(battle, "can_tera", None)),
            cls._simple_decision_tuple(getattr(battle, "trapped", None)),
        )

    @classmethod
    def _request_decision_signature(cls, request) -> tuple[object, ...]:
        """Compact key for the request fields that determine a battle decision.

        poke-env assigns ``battle.last_request`` from the parsed Showdown request.
        The previous implementation hashed the entire request JSON on every
        opponent choice. For hidden-state commits we only need to distinguish
        distinct decision requests, so keep the action-relevant fields instead
        of serializing side/team payloads every time.
        """
        if not isinstance(request, dict):
            return ()

        active_requests = request.get("active") or ()
        active_signature = []
        for active_request in active_requests:
            if not isinstance(active_request, dict):
                active_signature.append(repr(active_request))
                continue
            moves = active_request.get("moves") or ()
            move_signature = []
            for move_request in moves:
                if not isinstance(move_request, dict):
                    move_signature.append(repr(move_request))
                    continue
                move_signature.append(
                    (
                        move_request.get("id"),
                        move_request.get("target"),
                        bool(move_request.get("disabled", False)),
                    )
                )
            active_signature.append(
                (
                    bool(active_request.get("trapped", False)),
                    bool(active_request.get("maybeTrapped", False)),
                    bool(active_request.get("canTerastallize", False)),
                    tuple(move_signature),
                )
            )

        return (
            bool(request.get("wait", False)),
            bool(request.get("teamPreview", False)),
            cls._simple_decision_tuple(request.get("forceSwitch")),
            tuple(active_signature),
            cls._side_switch_signature(request.get("side")),
        )

    @staticmethod
    def _simple_decision_tuple(value) -> tuple:
        if value is None:
            return ()
        if isinstance(value, np.ndarray):
            return tuple(value.tolist())
        if isinstance(value, (list, tuple)):
            return tuple(value)
        return (value,)

    @classmethod
    def _side_switch_signature(cls, side) -> tuple[object, ...]:
        if not isinstance(side, dict):
            return ()
        pokemon = side.get("pokemon") or ()
        signature = []
        for pokemon_request in pokemon:
            if not isinstance(pokemon_request, dict):
                signature.append(repr(pokemon_request))
                continue
            condition = str(pokemon_request.get("condition", ""))
            signature.append(
                (
                    pokemon_request.get("ident"),
                    bool(pokemon_request.get("active", False)),
                    condition.endswith(" fnt"),
                )
            )
        return tuple(signature)

    def _record_preview_obs_mismatch(
        self,
        actual: dict[str, np.ndarray],
        prepared: dict[str, np.ndarray],
    ) -> None:
        if actual.keys() != prepared.keys():
            self._increment_timing("opponent_precompute_preview_keyset_mismatch")
            return
        for key in actual:
            if not np.array_equal(actual[key], prepared[key]):
                self._increment_timing(f"opponent_precompute_preview_mismatch_{key}")

    def _add_timing(self, key: str, value: float) -> None:
        timing = getattr(self, "_timing", None)
        if timing is not None:
            timing.add(key, value)

    def _increment_timing(self, key: str, delta: int = 1) -> None:
        timing = getattr(self, "_timing", None)
        if timing is not None:
            timing.increment(key, delta)

    def _timing_enabled(self) -> bool:
        timing = getattr(self, "_timing", None)
        return timing is not None and bool(getattr(timing, "enabled", True))


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
        self._inference = _LeagueInferenceService(
            agent_template,
            device,
            coalesce_window_sec=config.league_coalesce_ms / 1000.0,
        )

    def add_checkpoint(self, agent: PokeAgent, step: int):
        """Snapshot the current agent as a new league member."""
        path = f"{self.config.checkpoint_dir}/league/step_{step}.pt"
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"checkpoint_type": "league", "agent": agent.state_dict()}, path)
        self.checkpoints.append(path)
        self._inference.prune_checkpoints(set(self.checkpoints))

    def make_opponent(
        self,
        loop: asyncio.AbstractEventLoop,
        checkpoint_path: str | None = None,
    ) -> "LeaguePlayer | None":
        """Create a LeaguePlayer from a sampled checkpoint.

        Args:
            loop: The asyncio event loop the opponent will run on.

        Returns None if the league is empty.
        """
        path = checkpoint_path if checkpoint_path is not None else self.sample_checkpoint_path()
        if path is None:
            return None
        self._inference.ensure_checkpoint(path)

        battlestream_config = make_battlestream_configuration(self.env_config)

        obs_builder = ObservationBuilder(self.game_data, self.env_config.team_size)
        return LeaguePlayer(
            path,
            self._inference,
            obs_builder,
            account_configuration=make_local_account_configuration("league"),
            battle_format=self.env_config.battle_format,
            server_configuration=battlestream_config,
            log_level=showdown_log_level(),
            loop=loop,
        )

    def sample_checkpoint_path(self) -> str | None:
        path = self._sample_checkpoint()
        if path is None:
            return None
        self._inference.ensure_checkpoint(path)
        return path

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
        checkpoints = list(self.checkpoints)
        if not checkpoints:
            return None

        mid = len(checkpoints) // 2

        if len(checkpoints) < 2 or random.random() < 0.7:
            recent = checkpoints[mid:] if mid < len(checkpoints) else checkpoints
            return random.choice(recent)
        else:
            older = checkpoints[:mid] if mid > 0 else checkpoints
            return random.choice(older)
