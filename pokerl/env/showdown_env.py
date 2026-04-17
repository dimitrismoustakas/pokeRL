"""Gymnasium-compatible wrapper around poke-env for doubles battles."""

import asyncio
import itertools
import time
import uuid
from collections.abc import Awaitable, Callable
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, TimeoutError
from threading import Lock, Thread
from typing import Generic, TypeVar

import numpy as np

from poke_env import AccountConfiguration, ServerConfiguration
from poke_env.battle import DoubleBattle
from poke_env.concurrency import create_in_poke_loop
from poke_env.player import BattleOrder, DoubleBattleOrder, ForfeitBattleOrder, PassBattleOrder, Player

from ..config import EnvConfig
from ..data import GameData
from ..training.preview import TeamPreviewBatcher
from .observation import ObservationBuilder, build_team_preview_observation
from .actions import (
    build_action_mask, action_to_order,
)
from .rewards import sparse_reward


_ACCOUNT_LOCK = Lock()
_ACCOUNT_COUNTER = itertools.count()
_ACCOUNT_RUN_TAG = uuid.uuid4().hex[:4]
QueueItemT = TypeVar("QueueItemT")
OpponentFactory = Callable[[asyncio.AbstractEventLoop], Player]
TeamPairProvider = Callable[[], tuple[str | None, str | None]]


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


class _BattleStartTrace:
    def __init__(self):
        self.started_at = time.perf_counter()
        self._lock = Lock()
        self._marks: dict[str, float] = {}

    def mark_once(self, key: str) -> None:
        with self._lock:
            self._marks.setdefault(key, time.perf_counter())

    def has_mark(self, key: str) -> bool:
        with self._lock:
            return key in self._marks

    def snapshot(self) -> dict[str, float]:
        with self._lock:
            return dict(self._marks)


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
        self._battle_start_trace_getter: Callable[[], _BattleStartTrace | None] | None = None

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

    def set_battle_start_trace_getter(
        self,
        getter: Callable[[], _BattleStartTrace | None],
    ) -> None:
        self._battle_start_trace_getter = getter

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

        trace = self._battle_start_trace_getter() if self._battle_start_trace_getter is not None else None
        if trace is not None:
            trace.mark_once("player_first_emit_started")

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
        if trace is not None:
            trace.mark_once("player_first_emit_finished")
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
    ):
        self.config = config
        self.game_data = game_data
        self._battle_task = None
        self._loop, self._thread = self._start_event_loop()
        self._pending_opponent: Player | None = None
        self._battle_start_trace: _BattleStartTrace | None = None
        self._closed = False
        self._battle_started_at: float | None = None
        self._timing = _TimingStats()
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
            timing=self._timing,
            battle_format=config.battle_format,
            server_configuration=make_server_configuration(config),
            loop=self._loop,
            **player_options,
        )
        self._assert_player_loop(self.player, "player")
        if team_preview_fn is not None:
            self.player.set_team_preview_fn(self._do_team_preview)
        self.player.set_battle_start_trace_getter(self._get_battle_start_trace)
        self._install_battle_start_hooks(self.player, role="player")
        self.opponent = self._build_opponent(opponent_factory)
        self._prepare_next_battle()

    def reset(self) -> tuple[dict, np.ndarray]:
        """Start a new battle and return the first observation + mask."""
        if self._closed:
            raise RuntimeError("BattleEnv.reset() called after the environment was closed.")

        reset_started = time.perf_counter()
        self._episode_team_preview = None
        apply_pending_started = time.perf_counter()
        self._apply_pending_opponent()
        self._timing.add("env_reset_apply_pending_sec", time.perf_counter() - apply_pending_started)

        assign_teams_started = time.perf_counter()
        self._assign_teams_for_next_battle()
        self._timing.add("env_reset_team_assign_sec", time.perf_counter() - assign_teams_started)

        # Give stateful opponents (e.g. LeaguePlayer) a clean slate for the new battle.
        hidden_reset_started = time.perf_counter()
        if hasattr(self.opponent, "reset_hidden"):
            self.opponent.reset_hidden()
        self._timing.add("env_reset_hidden_reset_sec", time.perf_counter() - hidden_reset_started)

        self._battle_start_trace = _BattleStartTrace()
        battle_start_started = time.perf_counter()
        self._battle_task = asyncio.run_coroutine_threadsafe(
            self.player.battle_against(self.opponent, n_battles=1),
            self._loop,
        )
        self._timing.add("env_reset_battle_start_sec", time.perf_counter() - battle_start_started)
        self._battle_started_at = time.perf_counter()

        wait_started = time.perf_counter()
        obs, mask, _, _, _ = self.player.obs_queue.get()
        self._timing.add("env_reset_wait_first_obs_sec", time.perf_counter() - wait_started)
        self._record_battle_start_trace()
        self._prepare_next_battle()
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
        if self._ext_team_preview_fn is None:
            raise RuntimeError("BattleEnv team preview requested without a team_preview_fn.")
        if self._battle_start_trace is not None:
            self._battle_start_trace.mark_once("player_teampreview_started")
        obs = self.player.obs_builder.build(battle)
        picks = self._choose_team_preview_picks(obs)
        team_str = "/team " + "".join(str(int(pick) + 1) for pick in picks)
        self._episode_team_preview = (obs, picks)
        if self._battle_start_trace is not None:
            self._battle_start_trace.mark_once("player_teampreview_finished")
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
        prebuild_started = time.perf_counter()
        next_opponent = self._build_opponent(opponent_factory)
        self._timing.add("env_prebuild_opponent_sec", time.perf_counter() - prebuild_started)
        if self._pending_opponent is not None:
            self._disconnect_player(self._pending_opponent, self._loop)
        self._pending_opponent = next_opponent

    def stage_next_opponent_checkpoint(self, checkpoint_path: str) -> bool:
        if self._pending_opponent is not None or not hasattr(self.opponent, "stage_checkpoint"):
            return False
        stage_started = time.perf_counter()
        self.opponent.stage_checkpoint(checkpoint_path)
        self._timing.add(
            "env_stage_opponent_checkpoint_sec",
            time.perf_counter() - stage_started,
        )
        self._timing.increment("env_stage_opponent_checkpoint_calls")
        return True

    def _apply_pending_opponent(self):
        if self._pending_opponent is None:
            if hasattr(self.opponent, "apply_staged_checkpoint"):
                swap_started = time.perf_counter()
                if self.opponent.apply_staged_checkpoint():
                    self._timing.add(
                        "env_reset_opponent_checkpoint_swap_sec",
                        time.perf_counter() - swap_started,
                    )
            return

        if self._pending_opponent is None:
            return

        disconnect_started = time.perf_counter()
        self._disconnect_player(self.opponent, self._loop)
        self._timing.add("env_reset_opponent_disconnect_sec", time.perf_counter() - disconnect_started)
        self.opponent = self._pending_opponent
        self._pending_opponent = None

    def _build_opponent(self, opponent_factory: OpponentFactory) -> Player:
        opponent = opponent_factory(self._loop)
        self._attach_choice_timer(opponent)
        if hasattr(opponent, "set_timing"):
            opponent.set_timing(self._timing)
        self._assert_player_loop(opponent, "opponent")
        self._install_battle_start_hooks(opponent, role="opponent")
        return opponent

    def _prepare_next_battle(self, recompute_preview: bool = False) -> None:
        if self._team_pair_provider is None:
            return

        if self._prepared_player_team is None or self._prepared_opponent_team is None:
            self._prepared_player_team, self._prepared_opponent_team = self._team_pair_provider()
            recompute_preview = True

        if not recompute_preview or self._team_preview_batcher is None or self._ext_team_preview_fn is None:
            return

        build_started = time.perf_counter()
        self._prepared_preview_obs = build_team_preview_observation(
            self.game_data,
            self.config.team_size,
            self._prepared_player_team,
            self._prepared_opponent_team,
        )
        self._timing.add("env_precompute_preview_obs_sec", time.perf_counter() - build_started)

        submit_started = time.perf_counter()
        self._prepared_preview_future = self._team_preview_batcher.submit(self._prepared_preview_obs)
        self._timing.add("env_precompute_preview_submit_sec", time.perf_counter() - submit_started)
        self._timing.increment("env_precompute_preview_submit_calls")

        next_opponent = self._pending_opponent if self._pending_opponent is not None else self.opponent
        if hasattr(next_opponent, "prepare_team_preview"):
            opp_started = time.perf_counter()
            next_opponent.prepare_team_preview(
                own_packed_team=self._prepared_opponent_team,
                opponent_packed_team=self._prepared_player_team,
            )
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
        wait_started = time.perf_counter()
        self._prepared_preview_future.result()
        self._timing.add("env_precompute_preview_wait_sec", time.perf_counter() - wait_started)

    def _choose_team_preview_picks(self, obs: dict[str, np.ndarray]) -> np.ndarray:
        prepared_obs = self._active_preview_obs
        prepared_future = self._active_preview_future
        self._active_preview_obs = None
        self._active_preview_future = None

        if prepared_obs is not None and prepared_future is not None and self._obs_equal(obs, prepared_obs):
            self._timing.increment("env_precompute_preview_hits")
            try:
                wait_started = time.perf_counter()
                picks = prepared_future.result()
                self._timing.add("env_precompute_preview_hit_wait_sec", time.perf_counter() - wait_started)
                return picks
            except Exception:
                self._timing.increment("env_precompute_preview_fallbacks")
        elif prepared_obs is not None or prepared_future is not None:
            self._timing.increment("env_precompute_preview_misses")
            if prepared_obs is not None:
                self._record_preview_obs_mismatch(obs, prepared_obs)

        picks, _ = self._ext_team_preview_fn(obs)
        return picks

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

    def _get_battle_start_trace(self) -> _BattleStartTrace | None:
        return self._battle_start_trace

    def _install_battle_start_hooks(self, participant: Player, role: str) -> None:
        if getattr(participant, "_pokerl_battle_start_hooks_installed", False):
            return
        participant._pokerl_battle_start_hooks_installed = True

        if role == "player":
            original_challenge = participant.ps_client.challenge
            original_send_message = participant.ps_client.send_message
            original_battle_callback = participant.ps_client._on_battle_message

            async def challenge(username: str, format_: str, packed_team: str | None):
                trace = self._battle_start_trace
                if trace is not None:
                    trace.mark_once("player_challenge_send_started")
                result = await original_challenge(username, format_, packed_team)
                if trace is not None:
                    trace.mark_once("player_challenge_send_finished")
                return result

            participant.ps_client.challenge = challenge

            async def send_message(message: str, room: str = "", message_2: str | None = None):
                trace = self._battle_start_trace
                if trace is not None and room and message.startswith("/team "):
                    trace.mark_once("player_teampreview_send_started")
                result = await original_send_message(message, room, message_2)
                if trace is not None and room and message.startswith("/team "):
                    trace.mark_once("player_teampreview_send_finished")
                return result

            participant.ps_client.send_message = send_message

            async def battle_callback(split_messages: list[list[str]]) -> None:
                trace = self._battle_start_trace
                if (
                    trace is not None
                    and trace.has_mark("opponent_teampreview_send_finished")
                    and not trace.has_mark("player_post_preview_battle_message_started")
                ):
                    trace.mark_once("player_post_preview_battle_message_started")
                result = await original_battle_callback(split_messages)
                if (
                    trace is not None
                    and trace.has_mark("player_post_preview_battle_message_started")
                ):
                    trace.mark_once("player_post_preview_battle_message_finished")
                return result

            participant.ps_client._on_battle_message = battle_callback

            original_create_battle = participant._create_battle

            async def create_battle(split_message):
                trace = self._battle_start_trace
                if trace is not None:
                    trace.mark_once("player_create_battle_started")
                battle = await original_create_battle(split_message)
                if trace is not None:
                    trace.mark_once("player_create_battle_finished")
                return battle

            participant._create_battle = create_battle

            original_handle_battle_request = participant._handle_battle_request

            async def handle_battle_request(battle, maybe_default_order: bool = False):
                trace = self._battle_start_trace
                if trace is not None:
                    if trace.has_mark("player_teampreview_send_finished"):
                        trace.mark_once("player_post_preview_request_started")
                    else:
                        trace.mark_once("player_first_request_started")
                result = await original_handle_battle_request(battle, maybe_default_order)
                if trace is not None:
                    if trace.has_mark("player_post_preview_request_started"):
                        trace.mark_once("player_post_preview_request_finished")
                    else:
                        trace.mark_once("player_first_request_finished")
                return result

            participant._handle_battle_request = handle_battle_request

        elif role == "opponent":
            original_accept_challenge = participant.ps_client.accept_challenge
            original_send_message = participant.ps_client.send_message
            original_teampreview = participant.teampreview

            async def accept_challenge(username: str, packed_team: str | None):
                trace = self._battle_start_trace
                if trace is not None:
                    trace.mark_once("opponent_accept_send_started")
                result = await original_accept_challenge(username, packed_team)
                if trace is not None:
                    trace.mark_once("opponent_accept_send_finished")
                return result

            participant.ps_client.accept_challenge = accept_challenge
            def teampreview(battle: DoubleBattle) -> str:
                trace = self._battle_start_trace
                if trace is not None:
                    trace.mark_once("opponent_teampreview_started")
                result = original_teampreview(battle)
                if trace is not None:
                    trace.mark_once("opponent_teampreview_finished")
                return result

            participant.teampreview = teampreview
            async def send_message(message: str, room: str = "", message_2: str | None = None):
                trace = self._battle_start_trace
                if trace is not None and room and message.startswith("/team "):
                    trace.mark_once("opponent_teampreview_send_started")
                result = await original_send_message(message, room, message_2)
                if trace is not None and room and message.startswith("/team "):
                    trace.mark_once("opponent_teampreview_send_finished")
                return result

            participant.ps_client.send_message = send_message

    def _record_battle_start_trace(self) -> None:
        trace = self._battle_start_trace
        self._battle_start_trace = None
        if trace is None:
            return

        marks = trace.snapshot()

        def add_interval(key: str, start: float | None, end: float | None) -> None:
            if start is None or end is None or end < start:
                return
            self._timing.add(key, end - start)

        started_at = trace.started_at
        challenge_started = marks.get("player_challenge_send_started")
        challenge_finished = marks.get("player_challenge_send_finished")
        accept_started = marks.get("opponent_accept_send_started")
        accept_finished = marks.get("opponent_accept_send_finished")
        create_started = marks.get("player_create_battle_started")
        create_finished = marks.get("player_create_battle_finished")
        request_started = marks.get("player_first_request_started")
        request_finished = marks.get("player_first_request_finished")
        teampreview_started = marks.get("player_teampreview_started")
        teampreview_finished = marks.get("player_teampreview_finished")
        player_teampreview_send_started = marks.get("player_teampreview_send_started")
        player_teampreview_send_finished = marks.get("player_teampreview_send_finished")
        emit_started = marks.get("player_first_emit_started")
        emit_finished = marks.get("player_first_emit_finished")
        opponent_teampreview_started = marks.get("opponent_teampreview_started")
        opponent_teampreview_finished = marks.get("opponent_teampreview_finished")
        opponent_teampreview_send_started = marks.get("opponent_teampreview_send_started")
        opponent_teampreview_send_finished = marks.get("opponent_teampreview_send_finished")
        post_preview_battle_message_started = marks.get("player_post_preview_battle_message_started")
        post_preview_battle_message_finished = marks.get("player_post_preview_battle_message_finished")
        post_preview_request_started = marks.get("player_post_preview_request_started")
        post_preview_request_finished = marks.get("player_post_preview_request_finished")

        add_interval("env_reset_to_challenge_send_sec", started_at, challenge_started)
        add_interval("env_reset_challenge_send_sec", challenge_started, challenge_finished)
        add_interval("env_reset_challenge_to_accept_sec", challenge_finished, accept_started)
        add_interval("env_reset_accept_send_sec", accept_started, accept_finished)
        add_interval("env_reset_accept_to_create_sec", accept_finished, create_started)
        add_interval("env_reset_create_battle_sec", create_started, create_finished)
        add_interval("env_reset_create_to_first_request_sec", create_finished, request_started)
        add_interval("env_reset_first_request_total_sec", request_started, request_finished)
        add_interval("env_reset_first_request_to_teampreview_sec", request_started, teampreview_started)
        add_interval("env_reset_teampreview_compute_sec", teampreview_started, teampreview_finished)
        add_interval(
            "env_reset_player_preview_to_opponent_preview_sec",
            teampreview_finished,
            opponent_teampreview_started,
        )
        add_interval(
            "env_reset_opponent_teampreview_compute_sec",
            opponent_teampreview_started,
            opponent_teampreview_finished,
        )
        add_interval(
            "env_reset_opponent_teampreview_sec",
            opponent_teampreview_started,
            opponent_teampreview_finished,
        )
        add_interval(
            "env_reset_teampreview_send_sec",
            player_teampreview_send_started,
            player_teampreview_send_finished,
        )
        add_interval(
            "env_reset_opponent_preview_to_send_sec",
            opponent_teampreview_finished,
            opponent_teampreview_send_started,
        )
        add_interval(
            "env_reset_opponent_teampreview_send_sec",
            opponent_teampreview_send_started,
            opponent_teampreview_send_finished,
        )
        add_interval(
            "env_reset_opponent_preview_send_to_message_sec",
            opponent_teampreview_send_finished,
            post_preview_battle_message_started,
        )
        add_interval(
            "env_reset_post_preview_battle_message_sec",
            post_preview_battle_message_started,
            post_preview_battle_message_finished,
        )
        add_interval(
            "env_reset_post_preview_message_to_request_sec",
            post_preview_battle_message_started,
            post_preview_request_started,
        )
        add_interval(
            "env_reset_post_preview_request_total_sec",
            post_preview_request_started,
            post_preview_request_finished,
        )
        add_interval(
            "env_reset_post_preview_request_to_emit_sec",
            post_preview_request_started,
            emit_started,
        )
        add_interval(
            "env_reset_opponent_preview_to_first_emit_sec",
            opponent_teampreview_finished,
            emit_started,
        )
        add_interval("env_reset_teampreview_to_first_emit_sec", teampreview_finished, emit_started)
        add_interval("env_reset_request_to_first_emit_sec", request_finished, emit_started)
        add_interval("env_reset_first_emit_total_sec", emit_started, emit_finished)

    def _attach_choice_timer(self, opponent: Player) -> None:
        original_choose_move = opponent.choose_move

        def timed_choose_move(battle):
            started = time.perf_counter()
            result = original_choose_move(battle)
            self._timing.increment("opponent_choice_calls")
            self._timing.add("opponent_choice_sec", time.perf_counter() - started)
            return result

        opponent.choose_move = timed_choose_move

        original_teampreview = opponent.teampreview

        def timed_teampreview(battle):
            started = time.perf_counter()
            result = original_teampreview(battle)
            self._timing.increment("opponent_teampreview_calls")
            self._timing.add("opponent_teampreview_sec", time.perf_counter() - started)
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

        for participant in (self.player, self.opponent):
            self._disconnect_player(participant, self._loop)
        if self._pending_opponent is not None:
            self._disconnect_player(self._pending_opponent, self._loop)

        self._cancel_pending_tasks()
        self.player.obs_queue.clear()
        self.player.action_queue.clear()
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if not self._loop.is_running():
            self._loop.close()

        self._battle_task = None
        self._pending_opponent = None
        self._prepared_player_team = None
        self._prepared_opponent_team = None
        self._prepared_preview_obs = None
        self._prepared_preview_future = None
        self._active_preview_obs = None
        self._active_preview_future = None
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

    def wait_for_prepared_team_previews(self) -> None:
        list(self._executor.map(lambda env: env.wait_for_prepared_team_preview(), self.envs))

    def refresh_prepared_team_previews(self) -> None:
        list(self._executor.map(lambda env: env.refresh_prepared_team_preview(), self.envs))

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
