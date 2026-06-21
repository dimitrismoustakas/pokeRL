import asyncio
from concurrent.futures import Future, TimeoutError
from types import SimpleNamespace

import numpy as np
import pytest

from pokerl.env.showdown_env import BattleEnv, _OBSERVATION_POLL_SECONDS


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


def _make_step_env(queue: _ObservationQueue, battle_task: Future, *, timeout: float = 30.0):
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

    assert env._get_player_observation("step observation") is item
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


def test_step_waits_for_terminal_battle_task_before_returning_done() -> None:
    task = Future()
    task.set_result(None)
    battle = SimpleNamespace(battle_tag="battle-test", turn=7)
    queue = _ObservationQueue(({}, {}, 1.0, True, battle))
    env = _make_step_env(queue, task)

    _obs, _mask, reward, done, info = env.step(np.asarray([0, 0, 0], dtype=np.int64))

    assert reward == 1.0
    assert done is True
    assert info["battle_tag"] == "battle-test"
    assert env._battle_task is None
    assert len(env.player.action_queue.items) == 1


def test_step_surfaces_terminal_battle_task_failure() -> None:
    task = Future()
    task.set_exception(RuntimeError("finalizer failed"))
    battle = SimpleNamespace(battle_tag="battle-test", turn=7)
    queue = _ObservationQueue(({}, {}, -1.0, True, battle))
    env = _make_step_env(queue, task)

    with pytest.raises(RuntimeError, match="Battle task failed after terminal") as exc_info:
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

    env.close()

    assert env._closed is True
    assert obs_queue.cleared is True
    assert action_queue.cleared is True
    assert loop_thread.closed is True
