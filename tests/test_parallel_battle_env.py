import time
from types import SimpleNamespace
from threading import Event, Lock, Thread

import numpy as np
import pytest

from pokerl.env.showdown_env import BattleEnv, LoopThread, ParallelBattleEnv


class _ResetEnv:
    def __init__(self, index: int, state: dict[str, int], lock: Lock):
        self.index = index
        self.config = SimpleNamespace(showdown_dir="C:/showdown", worker_count=2)
        self._state = state
        self._lock = lock

    def reset(self):
        with self._lock:
            self._state["active"] += 1
            self._state["max_active"] = max(
                self._state["max_active"], self._state["active"]
            )
        try:
            time.sleep(0.02)
            return (
                {"obs": np.asarray([self.index], dtype=np.int64)},
                {"mask": np.asarray([True])},
            )
        finally:
            with self._lock:
                self._state["active"] -= 1

    def step(self, action):
        return (
            {"obs": np.asarray([self.index], dtype=np.int64)},
            {"mask": np.asarray([True])},
            float(self.index),
            True,
            {},
        )

    def snapshot_timing(self) -> dict[str, float]:
        return {}

    def reset_timing(self) -> None:
        return None

    def close(self) -> None:
        return None


def test_parallel_reset_limits_concurrency_and_preserves_order() -> None:
    state = {"active": 0, "max_active": 0}
    lock = Lock()
    env = ParallelBattleEnv(
        [_ResetEnv(index, state, lock) for index in range(5)],
        reset_concurrency=2,
    )

    try:
        obs, masks = env.reset()
    finally:
        env.close()

    assert state["max_active"] == 2
    assert obs["obs"].reshape(-1).tolist() == [0, 1, 2, 3, 4]
    assert masks["mask"].reshape(-1).tolist() == [True] * 5


def test_parallel_autoreset_uses_shared_reset_concurrency_limit() -> None:
    state = {"active": 0, "max_active": 0}
    lock = Lock()
    env = ParallelBattleEnv(
        [_ResetEnv(index, state, lock) for index in range(5)],
        reset_concurrency=2,
    )

    try:
        obs, masks, rewards, dones, infos = env.step(np.zeros((5, 3), dtype=np.int64))
    finally:
        env.close()

    assert state["max_active"] == 2
    assert obs["obs"].reshape(-1).tolist() == [0, 1, 2, 3, 4]
    assert masks["mask"].reshape(-1).tolist() == [True] * 5
    assert rewards.tolist() == [0.0, 1.0, 2.0, 3.0, 4.0]
    assert dones.tolist() == [True] * 5
    assert [info["episode_reward"] for info in infos] == rewards.tolist()


@pytest.mark.parametrize(("owns_loop", "expected_closes"), [(False, 0), (True, 1)])
def test_battle_env_closes_only_owned_loop_thread(
    owns_loop: bool,
    expected_closes: int,
) -> None:
    loop_closes: list[None] = []
    queue = SimpleNamespace(clear=lambda: None)
    player = SimpleNamespace(
        ps_client=None,
        request_close=lambda: None,
        obs_queue=queue,
        action_queue=queue,
    )
    env = object.__new__(BattleEnv)
    env._closed = False
    env._battle_task = None
    env._loop = object()
    env._loop_thread = SimpleNamespace(close=lambda: loop_closes.append(None))
    env._owns_loop_thread = owns_loop
    env.player = player
    env.opponent = SimpleNamespace(ps_client=None)
    env._pending_opponent = None

    env.close()

    assert env._closed
    assert len(loop_closes) == expected_closes


def test_battle_env_constructor_cleans_owned_but_not_borrowed_loop_on_failure(
    monkeypatch,
) -> None:
    owned_closes: list[None] = []
    borrowed_closes: list[None] = []
    owned_loop = SimpleNamespace(
        loop=object(),
        close=lambda: owned_closes.append(None),
    )
    borrowed_loop = SimpleNamespace(
        loop=object(),
        close=lambda: borrowed_closes.append(None),
    )
    monkeypatch.setattr(LoopThread, "start", staticmethod(lambda: owned_loop))

    with pytest.raises(ValueError, match="manages loop"):
        BattleEnv(
            SimpleNamespace(team_size=6),
            game_data=object(),
            opponent_factory=lambda loop: None,
            player_kwargs={"loop": object()},
        )
    with pytest.raises(ValueError, match="manages loop"):
        BattleEnv(
            SimpleNamespace(team_size=6),
            game_data=object(),
            opponent_factory=lambda loop: None,
            player_kwargs={"loop": object()},
            loop_thread=borrowed_loop,
        )

    assert len(owned_closes) == 1
    assert borrowed_closes == []


def test_parallel_close_closes_all_envs_before_owned_loop_threads() -> None:
    events: list[str] = []
    envs = [
        SimpleNamespace(close=lambda index=index: events.append(f"env-{index}"))
        for index in range(2)
    ]
    loop_threads = [
        SimpleNamespace(close=lambda index=index: events.append(f"loop-{index}"))
        for index in range(2)
    ]
    env = ParallelBattleEnv(envs, loop_threads=loop_threads)

    env.close()
    env.close()

    assert events == ["env-0", "env-1", "loop-0", "loop-1"]


def test_parallel_close_attempts_every_env_and_loop_after_failure() -> None:
    events: list[str] = []

    def fail_first_env() -> None:
        events.append("env-0")
        raise RuntimeError("injected close failure")

    envs = [
        SimpleNamespace(close=fail_first_env),
        SimpleNamespace(close=lambda: events.append("env-1")),
    ]
    loop_threads = [
        SimpleNamespace(close=lambda index=index: events.append(f"loop-{index}"))
        for index in range(2)
    ]
    env = ParallelBattleEnv(envs, loop_threads=loop_threads)

    with pytest.raises(ExceptionGroup, match="cleanup failed"):
        env.close()

    assert events == ["env-0", "env-1", "loop-0", "loop-1"]


def test_parallel_close_unblocks_running_step_before_executor_shutdown() -> None:
    step_started = Event()
    release_step = Event()
    step_finished = Event()
    close_finished = Event()
    close_errors: list[BaseException] = []

    def step(_action: np.ndarray):
        step_started.set()
        assert release_step.wait(timeout=2.0)
        step_finished.set()
        return {"obs": np.asarray([0])}, {"mask": np.asarray([True])}, 0.0, False, {}

    battle_env = SimpleNamespace(
        step=step,
        close=release_step.set,
    )
    loop_closed_after_step: list[bool] = []
    loop_thread = SimpleNamespace(
        close=lambda: loop_closed_after_step.append(step_finished.is_set())
    )
    env = ParallelBattleEnv([battle_env], loop_threads=[loop_thread])
    future = env._executor.submit(
        env._step_env,
        battle_env,
        np.zeros(3, dtype=np.int64),
    )
    assert step_started.wait(timeout=1.0)

    def close() -> None:
        try:
            env.close()
        except BaseException as exc:
            close_errors.append(exc)
        finally:
            close_finished.set()

    close_thread = Thread(target=close)
    close_thread.start()
    try:
        assert close_finished.wait(timeout=1.0)
    finally:
        release_step.set()
        close_thread.join(timeout=1.0)

    assert not close_thread.is_alive()
    assert close_errors == []
    future.result(timeout=1.0)
    assert loop_closed_after_step == [True]


def test_parallel_autoreset_exposes_terminal_observation_separately() -> None:
    terminal_obs = {"obs": np.asarray([42], dtype=np.int64)}
    reset_obs = {"obs": np.asarray([-1], dtype=np.int64)}

    def step(_action: np.ndarray):
        return terminal_obs, {"mask": np.asarray([False])}, 2.5, True, {}

    def reset():
        return reset_obs, {"mask": np.asarray([True])}

    battle_env = SimpleNamespace(
        step=step,
        reset=reset,
        close=lambda: None,
    )
    env = ParallelBattleEnv([battle_env])
    try:
        obs, masks, rewards, dones, infos = env.step(np.zeros((1, 3), dtype=np.int64))
    finally:
        env.close()

    assert obs["obs"].reshape(-1).tolist() == [-1]
    assert masks["mask"].reshape(-1).tolist() == [True]
    assert rewards.tolist() == [2.5]
    assert dones.tolist() == [True]
    assert infos[0]["final_observation"] is terminal_obs
    assert infos[0]["final_observation"]["obs"].tolist() == [42]


def test_disconnect_player_stops_local_client_without_websocket() -> None:
    class _LocalClient:
        local_server_configuration = object()

        def __init__(self):
            self.stopped = False

        async def stop_listening(self):
            self.stopped = True

    loop_thread = LoopThread.start()
    client = _LocalClient()
    player = SimpleNamespace(ps_client=client, username="local-test")
    try:
        BattleEnv._disconnect_player(player, loop_thread.loop)
    finally:
        loop_thread.close()

    assert client.stopped
