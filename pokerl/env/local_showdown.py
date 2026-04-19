"""Local Showdown transport using persistent BattleStream workers."""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import time
import uuid
from pathlib import Path
from threading import Lock, Thread
from typing import Callable

import orjson
import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.data import GenData
from poke_env.player import BattleOrder, DoubleBattleOrder, PassBattleOrder
from poke_env.teambuilder.teambuilder import Teambuilder

from ..config import EnvConfig
from ..data import GameData
from ..training.league import FrozenOpponentPolicy
from .actions import (
    ACTIONS_PER_SLOT,
    FIRST_SLOT_MASK_ROW,
    PASS_ACTION,
    action_to_order,
    build_legal_action_state,
)
from .observation import ObservationBuilder
from .rewards import sparse_reward
from .showdown_env import _TimingStats


LocalOpponentFactory = Callable[[], FrozenOpponentPolicy]
TeamPairProvider = Callable[[], tuple[str | None, str | None]]
_WORKER_EVENT_TIMEOUT_SEC = 60.0


def default_showdown_install_dir() -> str:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        raise RuntimeError("LOCALAPPDATA is not set; cannot infer a default Showdown install.")
    return str(Path(local_appdata) / "pokeRL" / "pokemon-showdown-srv8600")


def _normalize_choice_message(message: str) -> str:
    choice = message.strip()
    if choice.startswith("/choose "):
        return choice[len("/choose ") :]
    if choice.startswith("/team "):
        return "team " + choice[len("/team ") :]
    if choice.startswith("/"):
        return choice[1:]
    return choice


class LocalShowdownWorker:
    def __init__(self, install_dir: str, worker_name: str):
        self.install_dir = install_dir
        self.worker_name = worker_name
        self._queues: dict[str, queue.Queue[dict]] = {}
        self._lock = Lock()
        self._closed = False
        self._stderr_lines: list[str] = []
        self._writer_error: RuntimeError | None = None
        self._writer_sentinel = object()
        self._outbound: queue.SimpleQueue[str | object] = queue.SimpleQueue()
        self._process = subprocess.Popen(
            [
                "node",
                str(Path(__file__).resolve().parents[2] / "scripts" / "showdown_stream_worker.js"),
                "--install-dir",
                install_dir,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
            cwd=Path(__file__).resolve().parents[2],
        )
        if self._process.stdin is None or self._process.stdout is None or self._process.stderr is None:
            raise RuntimeError("Failed to create worker pipes.")
        self._reader = Thread(target=self._read_stdout, name=f"{worker_name}-stdout", daemon=True)
        self._stderr_reader = Thread(
            target=self._read_stderr,
            name=f"{worker_name}-stderr",
            daemon=True,
        )
        self._writer = Thread(
            target=self._write_stdin,
            name=f"{worker_name}-stdin",
            daemon=True,
        )
        self._writer.start()
        self._reader.start()
        self._stderr_reader.start()

    def create_battle(
        self,
        battle_id: str,
        *,
        format_id: str,
        p1_name: str,
        p2_name: str,
        p1_team: str,
        p2_team: str,
    ) -> None:
        with self._lock:
            if battle_id in self._queues:
                raise RuntimeError(f"Battle {battle_id} already exists on worker {self.worker_name}.")
            self._queues[battle_id] = queue.Queue()
        self._send(
            {
                "type": "create",
                "battle_id": battle_id,
                "format": format_id,
                "p1_name": p1_name,
                "p2_name": p2_name,
                "p1_team": p1_team,
                "p2_team": p2_team,
            }
        )

    def send_choice(self, battle_id: str, side: str, choice: str) -> None:
        self._send(
            {
                "type": "choice",
                "battle_id": battle_id,
                "side": side,
                "choice": choice,
            }
        )

    def get_event(self, battle_id: str, timeout: float | None = None) -> dict:
        worker_queue = self._get_queue(battle_id)
        try:
            return worker_queue.get(timeout=timeout)
        except queue.Empty as exc:
            raise TimeoutError(
                f"Timed out waiting for local worker event on {self.worker_name} battle {battle_id}."
            ) from exc

    def close_battle(self, battle_id: str) -> None:
        try:
            self._send({"type": "close_battle", "battle_id": battle_id})
        except RuntimeError:
            pass
        with self._lock:
            self._queues.pop(battle_id, None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._send({"type": "shutdown"})
        except RuntimeError:
            pass
        self._outbound.put(self._writer_sentinel)
        if self._process.stdin is not None:
            try:
                self._writer.join(timeout=2.0)
                self._process.stdin.close()
            except OSError:
                pass
        try:
            self._process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait(timeout=5.0)
        self._reader.join(timeout=1.0)
        self._stderr_reader.join(timeout=1.0)
        self._broadcast_error(RuntimeError(f"Worker {self.worker_name} was closed."))

    def _get_queue(self, battle_id: str) -> queue.Queue[dict]:
        with self._lock:
            worker_queue = self._queues.get(battle_id)
        if worker_queue is None:
            raise RuntimeError(f"Unknown battle {battle_id} on worker {self.worker_name}.")
        return worker_queue

    def _send(self, payload: dict) -> None:
        if self._closed:
            raise RuntimeError(f"Worker {self.worker_name} is closed.")
        if self._writer_error is not None:
            raise self._writer_error
        line = orjson.dumps(payload).decode("utf-8") + "\n"
        self._outbound.put(line)

    def _read_stdout(self) -> None:
        assert self._process.stdout is not None
        try:
            for raw_line in self._process.stdout:
                line = raw_line.strip()
                if not line:
                    continue
                event = json.loads(line)
                battle_id = event.get("battle_id")
                if battle_id is None:
                    continue
                with self._lock:
                    worker_queue = self._queues.get(battle_id)
                if worker_queue is not None:
                    worker_queue.put(event)
        except Exception as exc:
            self._broadcast_error(RuntimeError(self._worker_failure_message(f"stdout reader failed: {exc}")))
            return

        if not self._closed:
            self._broadcast_error(RuntimeError(self._worker_failure_message("stdout closed unexpectedly")))

    def _read_stderr(self) -> None:
        assert self._process.stderr is not None
        for raw_line in self._process.stderr:
            line = raw_line.rstrip()
            if line:
                self._stderr_lines.append(line)
        self._stderr_lines[:] = self._stderr_lines[-20:]

    def _write_stdin(self) -> None:
        assert self._process.stdin is not None
        try:
            while True:
                item = self._outbound.get()
                if item is self._writer_sentinel:
                    return
                batch = [item]
                while True:
                    try:
                        pending = self._outbound.get_nowait()
                    except queue.Empty:
                        break
                    if pending is self._writer_sentinel:
                        self._outbound.put(self._writer_sentinel)
                        break
                    batch.append(pending)
                self._process.stdin.write("".join(batch))
                self._process.stdin.flush()
        except OSError:
            error = RuntimeError(self._worker_failure_message("send failed"))
            self._writer_error = error
            self._broadcast_error(error)
        except Exception as exc:
            error = RuntimeError(self._worker_failure_message(f"stdin writer failed: {exc}"))
            self._writer_error = error
            self._broadcast_error(error)

    def _broadcast_error(self, error: RuntimeError) -> None:
        with self._lock:
            queues = list(self._queues.values())
        for worker_queue in queues:
            worker_queue.put({"type": "worker_error", "error": str(error)})

    def _worker_failure_message(self, prefix: str) -> str:
        stderr_tail = " | ".join(self._stderr_lines[-5:])
        if stderr_tail:
            return f"{prefix} ({self.worker_name}): {stderr_tail}"
        return f"{prefix} ({self.worker_name})"


class LocalShowdownWorkerPool:
    def __init__(self, install_dir: str, worker_count: int):
        if worker_count <= 0:
            raise ValueError("worker_count must be positive.")
        self.workers = [
            LocalShowdownWorker(install_dir=install_dir, worker_name=f"local-worker-{index}")
            for index in range(worker_count)
        ]

    def worker_for_env(self, env_index: int) -> tuple[LocalShowdownWorker, int]:
        worker_index = env_index % len(self.workers)
        return self.workers[worker_index], worker_index

    def close(self) -> None:
        for worker in self.workers:
            worker.close()


class _LocalBattleParser:
    _IGNORED_COMMANDS = {"t:", "expire", "uhtmlchange", "uhtml"}
    _RETRYABLE_ERROR_PREFIXES = ("[Unavailable choice]", "[Invalid choice]")

    def __init__(
        self,
        *,
        battle_tag: str,
        username: str,
        format_id: str,
        packed_team: str | None,
    ):
        self.battle_tag = battle_tag
        self.username = username
        gen = GenData.from_format(format_id).gen
        self.battle = DoubleBattle(
            battle_tag=battle_tag,
            username=username,
            logger=logging.getLogger(f"pokerl.local.{battle_tag}.{username}"),
            save_replays=False,
            gen=gen,
        )
        if packed_team is not None:
            self.battle._teambuilder_team = Teambuilder.parse_packed_team(packed_team)
        self.request_pending = False
        self.finished = False

    def apply_chunk(self, chunk: str) -> None:
        for line in chunk.split("\n"):
            if not line:
                continue
            split_message = line.split("|")
            if len(split_message) <= 1:
                continue
            command = split_message[1]
            if command == "request":
                request = orjson.loads(split_message[2]) if len(split_message) > 2 and split_message[2] else {}
                self.battle.parse_request(request, False)
                self.request_pending = not self.battle.wait
            elif command == "showteam":
                role = split_message[2]
                if role != self.battle.player_role:
                    teambuilder_team = Teambuilder.parse_packed_team("|".join(split_message[3:]))
                    self.battle.apply_teambuilder_team(
                        role,
                        teambuilder_team,
                        self.battle.teampreview_opponent_team,
                    )
            elif command == "win":
                self.battle.won_by(split_message[2])
                self.finished = True
                self.request_pending = False
            elif command == "tie":
                self.battle.tied()
                self.finished = True
                self.request_pending = False
            elif command == "error":
                error_message = "|".join(split_message[2:])
                if error_message.startswith(self._RETRYABLE_ERROR_PREFIXES):
                    continue
                raise RuntimeError(f"Local battle error for {self.username}: {error_message}")
            elif command == "bigerror":
                raise RuntimeError(f"Local battle bigerror for {self.username}: {'|'.join(split_message[2:])}")
            elif command in self._IGNORED_COMMANDS:
                continue
            else:
                self.battle.parse_message(split_message)


class LocalBattleEnv:
    def __init__(
        self,
        config: EnvConfig,
        game_data: GameData,
        opponent_factory: LocalOpponentFactory,
        worker: LocalShowdownWorker,
        worker_index: int,
        env_index: int,
        opponent_shard_key: str,
        team_preview_fn: Callable[[dict], tuple[np.ndarray, str]] | None = None,
        team_pair_provider: TeamPairProvider | None = None,
    ):
        self.config = config
        self.game_data = game_data
        self._opponent_factory = opponent_factory
        self._worker = worker
        self.opponent_shard_key = opponent_shard_key
        self._env_index = env_index
        self._team_preview_fn = team_preview_fn
        self._team_pair_provider = team_pair_provider
        self._timing = _TimingStats()
        self._battle_started_at: float | None = None
        self._battle_id: str | None = None
        self._player_parser: _LocalBattleParser | None = None
        self._opponent_parser: _LocalBattleParser | None = None
        self._pending_opponent: FrozenOpponentPolicy | None = None
        self._closed = False
        self._prepared_player_team: str | None = None
        self._prepared_opponent_team: str | None = None
        self._episode_team_preview: tuple[dict, np.ndarray] | None = None
        self._episode_opponent_choice_calls = 0
        self._episode_opponent_turn_calls: dict[tuple[str, int], int] = {}
        self._episode_opponent_checkpoint_path: str | None = None
        self._last_player_action: np.ndarray | None = None

        self.obs_builder = ObservationBuilder(game_data, config.team_size)
        self.opponent = opponent_factory()
        self._prepare_next_battle()

    @property
    def last_battle(self) -> DoubleBattle | None:
        if self._player_parser is None:
            return None
        return self._player_parser.battle

    def reset(self) -> tuple[dict, np.ndarray]:
        if self._closed:
            raise RuntimeError("LocalBattleEnv.reset() called after close().")

        reset_started = time.perf_counter()
        self._episode_team_preview = None
        self._episode_opponent_choice_calls = 0
        self._episode_opponent_turn_calls.clear()
        self._apply_pending_opponent()
        self._assign_teams_for_next_battle()
        self.opponent.reset_hidden()
        self._start_local_battle()
        obs, mask, _, _, _ = self._advance_until_agent_actionable()
        self._prepare_next_battle()
        self._timing.increment("env_reset_calls")
        self._timing.add("env_reset_total_sec", time.perf_counter() - reset_started)
        return obs, mask

    def step(self, action: np.ndarray) -> tuple[dict, np.ndarray, float, bool, dict]:
        if self._closed:
            raise RuntimeError("LocalBattleEnv.step() called after close().")
        if self._player_parser is None:
            raise RuntimeError("LocalBattleEnv.step() called before reset().")

        step_started = time.perf_counter()
        self._last_player_action = np.array(action, copy=True)
        action_state = build_legal_action_state(self._player_parser.battle)
        orders = []
        for slot in range(2):
            orders.append(
                action_to_order(
                    int(action[slot]),
                    self._player_parser.battle,
                    slot,
                    action_state=action_state,
                )
            )
        if all(isinstance(order, PassBattleOrder) for order in orders):
            choice = "default"
        else:
            choice = _normalize_choice_message(DoubleBattleOrder(orders[0], orders[1]).message)

        send_started = time.perf_counter()
        self._worker.send_choice(self._battle_id, "p1", choice)
        self._timing.add("local_worker_player_choice_send_sec", time.perf_counter() - send_started)

        obs, mask, reward, done, info = self._advance_until_agent_actionable()
        self._timing.increment("env_step_calls")
        self._timing.add("env_step_total_sec", time.perf_counter() - step_started)
        return obs, mask, reward, done, info

    def set_next_opponent(self, opponent_factory: LocalOpponentFactory):
        self._pending_opponent = opponent_factory()

    def stage_next_opponent_checkpoint(self, checkpoint_path: str) -> bool:
        if self._pending_opponent is not None:
            return False
        self.opponent.stage_checkpoint(checkpoint_path)
        return True

    def snapshot_timing(self) -> dict[str, float]:
        return self._timing.snapshot_and_reset()

    def reset_timing(self) -> None:
        self._timing.reset()

    def wait_for_prepared_team_preview(self) -> None:
        return

    def refresh_prepared_team_preview(self) -> None:
        return

    def close(self):
        if self._closed:
            return
        if self._battle_id is not None:
            self._worker.close_battle(self._battle_id)
            self._battle_id = None
        self._closed = True

    def _advance_until_agent_actionable(
        self,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, float, bool, dict]:
        while True:
            wait_started = time.perf_counter()
            event = self._worker.get_event(self._battle_id, timeout=_WORKER_EVENT_TIMEOUT_SEC)
            self._timing.add("local_worker_wait_sec", time.perf_counter() - wait_started)
            if event["type"] == "worker_error":
                raise RuntimeError(event["error"])
            if event["type"] != "chunk":
                continue

            process_started = time.perf_counter()
            side = event["side"]
            parser = self._player_parser if side == "p1" else self._opponent_parser
            assert parser is not None
            parser.apply_chunk(event["chunk"])
            self._timing.add("local_worker_chunk_process_sec", time.perf_counter() - process_started)

            if parser.finished and side == "p1":
                obs = self.obs_builder.build(parser.battle)
                mask = np.zeros((ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)
                mask[FIRST_SLOT_MASK_ROW, PASS_ACTION] = True
                mask[PASS_ACTION + 1, PASS_ACTION] = True
                reward = sparse_reward(parser.battle)
                info = {
                    "battle_tag": parser.battle.battle_tag,
                    "battle_turn": getattr(parser.battle, "turn", 0),
                    "episode_opponent_choice_calls": self._episode_opponent_choice_calls,
                    "episode_opponent_repeated_turn_requests": sum(
                        count - 1 for count in self._episode_opponent_turn_calls.values() if count > 1
                    ),
                    "episode_opponent_max_calls_same_turn": max(
                        self._episode_opponent_turn_calls.values(),
                        default=0,
                    ),
                }
                if self._battle_started_at is not None:
                    info["episode_sec"] = time.perf_counter() - self._battle_started_at
                if self._episode_team_preview is not None:
                    preview_obs, preview_picks = self._episode_team_preview
                    info["team_preview_obs"] = preview_obs
                    info["team_preview_picks"] = preview_picks
                if self._episode_opponent_checkpoint_path is not None:
                    info["episode_opponent_checkpoint_path"] = self._episode_opponent_checkpoint_path
                return obs, mask, reward, True, info

            if not parser.request_pending:
                continue

            if side == "p2":
                self._handle_opponent_request(parser.battle)
                parser.request_pending = False
                continue

            if parser.battle.teampreview:
                team_choice = self._do_team_preview(parser.battle)
                send_started = time.perf_counter()
                self._worker.send_choice(self._battle_id, "p1", _normalize_choice_message(team_choice))
                self._timing.add("local_worker_teampreview_send_sec", time.perf_counter() - send_started)
                parser.request_pending = False
                continue

            emit_started = time.perf_counter()
            obs = self.obs_builder.build(parser.battle)
            self._timing.add("player_emit_obs_sec", time.perf_counter() - emit_started)
            mask_started = time.perf_counter()
            mask = build_legal_action_state(parser.battle).joint_mask
            self._timing.add("player_emit_mask_sec", time.perf_counter() - mask_started)
            reward_started = time.perf_counter()
            reward = sparse_reward(parser.battle)
            self._timing.add("player_emit_reward_sec", time.perf_counter() - reward_started)
            self._timing.increment("player_emit_calls")
            self._timing.add("player_emit_total_sec", time.perf_counter() - emit_started)
            parser.request_pending = False
            info = {
                "battle_tag": parser.battle.battle_tag,
                "battle_turn": getattr(parser.battle, "turn", 0),
            }
            return obs, mask, reward, False, info

    def _handle_opponent_request(self, battle: DoubleBattle) -> None:
        self._episode_opponent_choice_calls += 1
        key = (battle.battle_tag, int(getattr(battle, "turn", 0)))
        self._episode_opponent_turn_calls[key] = self._episode_opponent_turn_calls.get(key, 0) + 1

        choice_started = time.perf_counter()
        if battle.teampreview:
            choice = _normalize_choice_message(self.opponent.teampreview(battle))
            self._timing.increment("opponent_teampreview_calls")
            self._timing.add("opponent_teampreview_sec", time.perf_counter() - choice_started)
        else:
            order = self.opponent.choose_move(battle)
            if isinstance(order, PassBattleOrder):
                choice = "default"
            else:
                assert isinstance(order, BattleOrder)
                choice = _normalize_choice_message(order.message)
            self._timing.increment("opponent_choice_calls")
            self._timing.add("opponent_choice_sec", time.perf_counter() - choice_started)

        send_started = time.perf_counter()
        self._worker.send_choice(self._battle_id, "p2", choice)
        self._timing.add("local_worker_opponent_choice_send_sec", time.perf_counter() - send_started)

    def _do_team_preview(self, battle: DoubleBattle) -> str:
        if self._team_preview_fn is None:
            raise RuntimeError("LocalBattleEnv team preview requested without a callback.")
        obs = self.obs_builder.build(battle)
        picks, team_str = self._team_preview_fn(obs)
        self._episode_team_preview = (obs, picks)
        return team_str

    def _start_local_battle(self) -> None:
        if self._battle_id is not None:
            self._worker.close_battle(self._battle_id)
        battle_id = f"local-{self._env_index}-{uuid.uuid4().hex[:12]}"
        player_team = self._prepared_player_team
        opponent_team = self._prepared_opponent_team
        self._player_parser = _LocalBattleParser(
            battle_tag=battle_id,
            username="agent",
            format_id=self.config.battle_format,
            packed_team=player_team,
        )
        self._opponent_parser = _LocalBattleParser(
            battle_tag=battle_id,
            username="league",
            format_id=self.config.battle_format,
            packed_team=opponent_team,
        )
        self._battle_id = battle_id
        self._battle_started_at = time.perf_counter()
        self._episode_opponent_checkpoint_path = self.opponent.checkpoint_path
        self._worker.create_battle(
            battle_id,
            format_id=self.config.battle_format,
            p1_name="agent",
            p2_name="league",
            p1_team=player_team,
            p2_team=opponent_team,
        )

    def _assign_teams_for_next_battle(self) -> None:
        self._prepare_next_battle()

    def _prepare_next_battle(self) -> None:
        if self._team_pair_provider is None:
            self._prepared_player_team = None
            self._prepared_opponent_team = None
        else:
            self._prepared_player_team, self._prepared_opponent_team = self._team_pair_provider()

    def _apply_pending_opponent(self) -> None:
        if self._pending_opponent is not None:
            self.opponent = self._pending_opponent
            self._pending_opponent = None
            return
        self.opponent.apply_staged_checkpoint()
