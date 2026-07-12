import asyncio
from concurrent.futures import Future, TimeoutError
from types import SimpleNamespace

import numpy as np
import pytest

from pokerl.env.showdown_env import (
    BattleEnv,
    ShowdownPlayer,
    _OBSERVATION_POLL_SECONDS,
    _PlayerObservation,
)


class _ObservationQueue:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.timeouts = []
        self.cleared = False

    def get(self, timeout=None):
        self.timeouts.append(timeout)
        if not self.outcomes:
            raise AssertionError("Observation queue was read too many times.")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    def clear(self):
        self.cleared = True


def _make_env(queue: _ObservationQueue, battle_task: Future):
    env = object.__new__(BattleEnv)
    env.player = SimpleNamespace(obs_queue=queue)
    env._battle_task = battle_task
    return env


class _ActionQueue:
    def __init__(self):
        self.items = []
        self.cleared = False

    def put(self, item):
        self.items.append(item)

    def clear(self):
        self.cleared = True


def _make_step_env(
    queue: _ObservationQueue, battle_task: Future, *, timeout: float = 30.0
):
    env = object.__new__(BattleEnv)
    env._closed = False
    env._timing = SimpleNamespace(enabled=False)
    env.config = SimpleNamespace(battlestream_startup_timeout=timeout)
    env.player = SimpleNamespace(obs_queue=queue, action_queue=_ActionQueue())
    env.opponent = SimpleNamespace()
    env._battle_task = battle_task
    env._last_player_action = None
    env._battle_started_at = None
    env._episode_team_preview = None
    env._episode_opponent_preview_picks = None
    env._episode_opponent_choice_calls = 0
    env._episode_opponent_turn_calls = {}
    env._episode_opponent_checkpoint_path = None
    return env


def test_get_player_observation_polls_until_observation_while_task_runs() -> None:
    item = ({}, {}, 0.0, False, None)
    queue = _ObservationQueue(TimeoutError(), item)
    env = _make_env(queue, Future())

    observation = env._get_player_observation("step observation")

    assert observation.request_id is None
    assert tuple(observation) == item
    assert queue.timeouts == [_OBSERVATION_POLL_SECONDS, _OBSERVATION_POLL_SECONDS]


def test_get_player_observation_surfaces_failed_battle_task() -> None:
    task = Future()
    task.set_exception(RuntimeError("simulator failed"))
    env = _make_env(_ObservationQueue(TimeoutError()), task)

    with pytest.raises(RuntimeError, match="Battle task failed") as exc_info:
        env._get_player_observation("initial observation")

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "simulator failed"


def test_get_player_observation_reports_task_completed_without_observation() -> None:
    task = Future()
    task.set_result(None)
    env = _make_env(_ObservationQueue(TimeoutError()), task)

    with pytest.raises(RuntimeError, match="completed without producing"):
        env._get_player_observation("step observation")


def test_get_player_observation_times_out_pending_battle_task() -> None:
    env = _make_env(_ObservationQueue(TimeoutError()), Future())
    env._describe_pending_battle_tasks = lambda: "diagnostic task stack"

    with pytest.raises(TimeoutError, match="diagnostic task stack"):
        env._get_player_observation("initial observation", max_wait_sec=0.0)


def test_step_waits_for_terminal_battle_task_before_returning_done() -> None:
    task = Future()
    task.set_result(None)
    battle = SimpleNamespace(battle_tag="battle-test", turn=7)
    queue = _ObservationQueue(({}, {}, 1.0, True, battle))
    env = _make_step_env(queue, task)
    preview_obs = {"context": np.asarray([1], dtype=np.int64)}
    preview_picks = np.asarray([0, 1, 2, 3], dtype=np.int64)
    env._episode_team_preview = (preview_obs, preview_picks, 14)

    _obs, _mask, reward, done, info = env.step(np.asarray([0, 0, 0], dtype=np.int64))

    assert reward == 1.0
    assert done is True
    assert info["battle_tag"] == "battle-test"
    assert info["team_preview_obs"] is preview_obs
    np.testing.assert_array_equal(info["team_preview_picks"], preview_picks)
    assert info["team_preview_policy_version"] == 14
    assert env._battle_task is None
    assert len(env.player.action_queue.items) == 1


def test_step_submits_action_to_matching_player_request() -> None:
    task = Future()
    battle = SimpleNamespace(battle_tag="battle-test", turn=2)
    next_observation = _PlayerObservation(
        request_id=43,
        obs={},
        mask={},
        reward=0.0,
        done=False,
        battle=battle,
    )
    queue = _ObservationQueue(next_observation)
    env = _make_step_env(queue, task)
    env._pending_player_request_id = 42
    submitted = []

    def submit_action(request_id, action, *, timeout=None):
        submitted.append((request_id, action.copy(), timeout))

    env.player.submit_action = submit_action

    action = np.asarray([4, 1, 0], dtype=np.int64)
    _obs, _mask, reward, done, info = env.step(action)

    assert reward == 0.0
    assert done is False
    assert info["battle_tag"] == "battle-test"
    assert env._pending_player_request_id == 43
    assert len(env.player.action_queue.items) == 0
    assert submitted[0][0] == 42
    np.testing.assert_array_equal(submitted[0][1], action)
    assert submitted[0][2] == 30.0


def test_step_action_submit_timeout_includes_diagnostics() -> None:
    task = Future()
    queue = _ObservationQueue()
    env = _make_step_env(queue, task, timeout=12.0)
    env._pending_player_request_id = 42
    env._describe_pending_battle_tasks = lambda: "loop diagnostic stack"

    def submit_action(request_id, action, *, timeout=None):
        assert timeout == 12.0
        raise TimeoutError()

    env.player.submit_action = submit_action

    with pytest.raises(TimeoutError) as exc_info:
        env.step(np.asarray([4, 1, 0], dtype=np.int64))

    message = str(exc_info.value)
    assert "BattleEnv agent action submission" in message
    assert "request_id=42" in message
    assert "action=[4, 1, 0]" in message
    assert "loop diagnostic stack" in message
    assert env._pending_player_request_id == 42


@pytest.mark.asyncio
async def test_showdown_player_submit_action_resolves_matching_request() -> None:
    action = np.asarray([4, 1, 0], dtype=np.int64)
    fake_player = SimpleNamespace(
        _pending_action_request_id=7,
        _pending_action_future=asyncio.get_running_loop().create_future(),
    )

    await ShowdownPlayer._submit_action(fake_player, 7, action)

    np.testing.assert_array_equal(fake_player._pending_action_future.result(), action)


@pytest.mark.asyncio
async def test_showdown_player_submit_action_rejects_wrong_request() -> None:
    fake_player = SimpleNamespace(
        _pending_action_request_id=7,
        _pending_action_future=asyncio.get_running_loop().create_future(),
    )

    with pytest.raises(RuntimeError, match="waiting for request 7"):
        await ShowdownPlayer._submit_action(
            fake_player,
            8,
            np.asarray([4, 1, 0], dtype=np.int64),
        )

    assert not fake_player._pending_action_future.done()


@pytest.mark.asyncio
async def test_showdown_player_finished_battle_resolves_pending_action() -> None:
    battle = SimpleNamespace(finished=True)
    future = asyncio.get_running_loop().create_future()
    fake_player = SimpleNamespace(
        _pending_finished_battle=None,
        _pending_action_future=future,
        _pending_action_battle=battle,
    )

    ShowdownPlayer._battle_finished_callback(fake_player, battle)

    assert fake_player._pending_finished_battle is battle
    assert future.done()
    assert future.result() is None


def test_step_surfaces_terminal_battle_task_failure() -> None:
    task = Future()
    task.set_exception(RuntimeError("finalizer failed"))
    battle = SimpleNamespace(battle_tag="battle-test", turn=7)
    queue = _ObservationQueue(({}, {}, -1.0, True, battle))
    env = _make_step_env(queue, task)

    with pytest.raises(
        RuntimeError, match="Battle task failed after terminal"
    ) as exc_info:
        env.step(np.asarray([0, 0, 0], dtype=np.int64))

    assert isinstance(exc_info.value.__cause__, RuntimeError)
    assert str(exc_info.value.__cause__) == "finalizer failed"


def test_step_rejects_terminal_task_that_does_not_finish() -> None:
    battle = SimpleNamespace(battle_tag="battle-test", turn=7)
    queue = _ObservationQueue(({}, {}, -1.0, True, battle))
    env = _make_step_env(queue, Future(), timeout=0.0)

    with pytest.raises(TimeoutError, match="overlap the previous battle"):
        env.step(np.asarray([0, 0, 0], dtype=np.int64))


def test_run_battle_emits_terminal_after_battle_coroutine_returns() -> None:
    events = []

    class Player:
        async def battle_against(self, opponent, n_battles: int = 1):
            assert n_battles == 1
            assert opponent == "opponent"
            events.append("battle-returned")

        def emit_pending_finished_battle(self):
            events.append("terminal-emitted")

    env = object.__new__(BattleEnv)
    env.player = Player()
    env.opponent = "opponent"

    asyncio.run(env._run_battle())

    assert events == ["battle-returned", "terminal-emitted"]


def test_close_does_not_mask_existing_training_exception() -> None:
    class FailingTask:
        def done(self):
            return False

        def result(self, timeout=None):
            raise RuntimeError("battle task failed during cleanup")

        def cancel(self):
            raise AssertionError("cancel should not be needed after a task failure")

    class Clearable:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    loop_thread = SimpleNamespace(closed=False)
    loop_thread.close = lambda: setattr(loop_thread, "closed", True)
    obs_queue = Clearable()
    action_queue = Clearable()

    env = object.__new__(BattleEnv)
    env._closed = False
    env.player = SimpleNamespace(
        request_close=lambda: None,
        obs_queue=obs_queue,
        action_queue=action_queue,
    )
    env.opponent = SimpleNamespace()
    env._pending_opponent = None
    env._battle_task = FailingTask()
    env._loop = None
    env._loop_thread = loop_thread
    env._owns_loop_thread = True

    env.close()

    assert env._closed is True
    assert obs_queue.cleared is True
    assert action_queue.cleared is True
    assert loop_thread.closed is True
