"""Battle-stream based environment for local VGC doubles training.

Replaces the WebSocket+Showdown-server approach with direct subprocess I/O,
using Pokemon Showdown's built-in simulate-battle mode. Each battle gets its
own node process — no TCP, no WebSocket, no server-side queue contention.

Protocol is format-agnostic. The implementation covers VGC doubles (two active
slots per side) and handles team preview, force switches, and the pass action.

Draft status: covers the happy path. Edge cases to harden before production:
  - Reconnect / subprocess crash recovery
  - Timeout handling on stdout reads
  - Full VecEnv / multi-worker support (currently single-env only)
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.player import DoubleBattleOrder, PassBattleOrder

from pokerl.env.actions import (
    ACTIONS_PER_SLOT,
    PASS_ACTION,
    action_to_order,
    build_legal_action_state,
)
from pokerl.env.observation import ObservationBuilder
from pokerl.team_pool import TeamPool

_SHOWDOWN_DIR = Path(os.environ.get("LOCALAPPDATA", "")) / "pokeRL" / "pokemon-showdown"
_RUNNER_SCRIPT = Path(__file__).parent.parent.parent / "scripts" / "battle_stream_runner.js"
_LOG = logging.getLogger(__name__)

# Prefix length of "/choose " that poke-env prepends to all order messages.
_CHOOSE_PREFIX = len("/choose ")


def _order_to_sim_choice(order: DoubleBattleOrder) -> str:
    """Convert a DoubleBattleOrder to a simulate-battle choice string.

    DoubleBattleOrder.message is "/choose move 1 -2, switch Incineroar".
    simulate-battle expects                "move 1 -2, switch Incineroar".
    """
    return order.message[_CHOOSE_PREFIX:]


class BattleStreamProcess:
    """One pokemon-showdown simulate-battle subprocess per battle.

    Drives both players (p1 = agent, p2 = random) from a single Node.js
    process communicating over stdin/stdout pipes.
    """

    def __init__(
        self,
        formatid: str,
        p1_team: str,
        p2_team: str,
        gen: int = 9,
        showdown_dir: Path = _SHOWDOWN_DIR,
    ) -> None:
        self._gen = gen
        env = {**os.environ, "SHOWDOWN_DIR": str(showdown_dir)}
        self._proc = subprocess.Popen(
            ["node", str(_RUNNER_SCRIPT)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            cwd=showdown_dir,
            env=env,
        )
        import io
        self._stdin = io.TextIOWrapper(self._proc.stdin, encoding="utf-8", line_buffering=True)

        # A background reader thread continuously drains stdout into a queue so
        # the OS pipe buffer never fills up and blocks Node.js (the classic
        # subprocess deadlock).  The main thread reads from the queue.
        self._line_queue: queue.Queue[str | None] = queue.Queue()
        self._reader_thread = threading.Thread(
            target=self._reader_loop,
            args=(self._proc.stdout, self._line_queue),
            daemon=True,
            name="bsp-stdout-reader",
        )
        self._reader_thread.start()

        logger = _LOG
        tag = f"battle-{formatid}-stream"
        self.battle_p1: DoubleBattle = DoubleBattle(tag, "p1", logger, gen=gen)
        self.battle_p2: DoubleBattle = DoubleBattle(tag, "p2", logger, gen=gen)

        self.done: bool = False
        self.winner: str | None = None  # "p1" or "p2"
        self._prev_hp_p1: float = 1.0
        self._prev_hp_p2: float = 1.0

        self._line_buf: str | None = None  # one-line lookahead buffer

        self._write(f'>start {json.dumps({"formatid": formatid})}')
        self._write(f'>player p1 {json.dumps({"name": "p1", "team": p1_team})}')
        self._write(f'>player p2 {json.dumps({"name": "p2", "team": p2_team})}')

    @staticmethod
    def _reader_loop(stdout_pipe, line_queue: queue.Queue) -> None:
        """Background thread: drain subprocess stdout into line_queue.

        Runs as a daemon thread so the main thread never blocks on readline.
        Sends None as the EOF sentinel when the pipe closes.
        """
        try:
            for raw in stdout_pipe:
                line_queue.put(raw.decode("utf-8", errors="replace").rstrip("\n"))
        except Exception:
            pass
        finally:
            line_queue.put(None)  # EOF sentinel

    # ------------------------------------------------------------------
    # Low-level I/O
    # ------------------------------------------------------------------

    def _write(self, line: str) -> None:
        self._stdin.write(line + "\n")
        self._stdin.flush()

    def _readline(self) -> str | None:
        """Return the next line (no newline), None on EOF, '' on blank line.

        Reads from the background-thread queue so the main thread never blocks
        on the OS pipe directly; the reader thread drains the pipe continuously,
        preventing Node.js from stalling on back-pressure.
        """
        if self._line_buf is not None:
            line, self._line_buf = self._line_buf, None
            return line
        line = self._line_queue.get()  # blocks until reader thread delivers
        return line  # None == EOF sentinel from reader thread

    def _read_chunk(self) -> tuple[str, str | None, list[str]]:
        """Read one output chunk from stdout.

        The runner script terminates every chunk with a '.' sentinel line so
        Python knows the chunk is complete without blocking on the next header.

        Returns (chunk_type, player_or_none, message_lines).
        """
        # Advance to the next chunk-header line, skipping sentinels and blanks.
        while True:
            line = self._readline()
            if line is None:  # EOF — process exited
                self.done = True
                return "update", None, []
            if line in ("update", "sideupdate"):
                break
            # skip '.' sentinels and blank lines between chunks

        chunk_type = line
        player: str | None = None
        lines: list[str] = []

        if chunk_type == "sideupdate":
            player = self._readline()  # next non-blank line is the player id
            while player == "":        # skip any blank lines before player id
                player = self._readline()

        while True:
            line = self._readline()
            if line is None:  # EOF
                self.done = True
                break
            if line == ".":  # explicit end-of-chunk sentinel
                break
            if line in ("update", "sideupdate"):
                self._line_buf = line  # shouldn't happen but handle gracefully
                break
            if line:  # skip blank lines within chunks (e.g. between |split| sections)
                lines.append(line)

        return chunk_type, player, lines

    # ------------------------------------------------------------------
    # Message dispatch
    # ------------------------------------------------------------------

    def _feed_line(self, battle: DoubleBattle, line: str) -> None:
        if not line.startswith("|"):
            return
        try:
            battle.parse_message(line.split("|"))
        except Exception:
            pass

    def _process_chunk(
        self, chunk_type: str, player: str | None, lines: list[str]
    ) -> None:
        if chunk_type == "sideupdate":
            battle = self.battle_p1 if player == "p1" else self.battle_p2
            for line in lines:
                if line.startswith("|request|"):
                    request = json.loads(line[len("|request|"):])
                    battle.parse_request(request)
                else:
                    self._feed_line(battle, line)
            return

        # update chunk — shared messages with optional |split|pX sections
        target = "both"
        for line in lines:
            if line == "|split|p1":
                target = "p1"
                continue
            if line == "|split|p2":
                target = "p2"
                continue

            if line.startswith("|win|"):
                self.done = True
                parts = line.split("|")
                self.winner = parts[2] if len(parts) > 2 else None
            elif line == "|tie|":
                self.done = True

            if target in ("both", "p1"):
                self._feed_line(self.battle_p1, line)
            if target in ("both", "p2"):
                self._feed_line(self.battle_p2, line)

    # ------------------------------------------------------------------
    # Read until both sides have a request (or battle ends)
    # ------------------------------------------------------------------

    def read_until_requests(self, *, need_p1: bool = True, need_p2: bool = True) -> None:
        """Consume stdout until the requested sides have a fresh request, or done.

        need_p1 / need_p2 control which sides must receive a sideupdate before
        returning.  Pass need_p1=False when p1 is in a Showdown "wait" state
        (already submitted its choice) so only p2's response is awaited — this
        also handles Showdown error-retry chunks where only one side gets a
        fresh sideupdate.

        When the battle ends we immediately terminate the subprocess rather than
        trying to drain remaining post-win output.
        """
        p1_ready = not need_p1
        p2_ready = not need_p2

        while True:
            chunk_type, player, lines = self._read_chunk()

            # EOF: subprocess already exited.
            if self.done and not lines:
                break

            self._process_chunk(chunk_type, player, lines)

            if self.done:
                # Battle over — kill the subprocess so it stops writing.
                try:
                    self._proc.terminate()
                except Exception:
                    pass
                break

            if chunk_type == "sideupdate":
                if player == "p1":
                    p1_ready = True
                elif player == "p2":
                    p2_ready = True

            if p1_ready and p2_ready:
                break

    # ------------------------------------------------------------------
    # High-level commands
    # ------------------------------------------------------------------

    def send_team_preview(self, p1_order: str, p2_order: str) -> None:
        """Send team order strings, e.g. '1234' for both."""
        self._write(f">p1 team {p1_order}")
        self._write(f">p2 team {p2_order}")

    def send_choices(self, p1_choice: str, p2_choice: str) -> None:
        self._write(f">p1 {p1_choice}")
        self._write(f">p2 {p2_choice}")

    def close(self) -> None:
        try:
            self._stdin.close()
        except Exception:
            pass
        try:
            self._proc.terminate()
            self._proc.wait(timeout=2.0)
        except Exception:
            pass


class BattleStreamEnv:
    """Gymnasium-style env backed by a BattleStream subprocess.

    The agent is p1. The opponent (p2) is a simple uniform-random policy
    driven inline — no separate process needed.

    Returns the same (obs_dict, joint_mask, reward, done, info) interface
    as BattleEnv so the training loop needs no changes.
    """

    def __init__(
        self,
        battle_format: str,
        team_pool: TeamPool,
        obs_builder: ObservationBuilder,
        gen: int = 9,
        showdown_dir: Path = _SHOWDOWN_DIR,
    ) -> None:
        self._format = battle_format
        self._pool = team_pool
        self._obs_builder = obs_builder
        self._gen = gen
        self._showdown_dir = showdown_dir
        self._stream: BattleStreamProcess | None = None
        self._rng = np.random.default_rng()

    # ------------------------------------------------------------------
    # Team preview
    # ------------------------------------------------------------------

    def _preview_order(self, battle: DoubleBattle) -> str:
        """Return a simple team order string: first 4 pokemon, no reordering."""
        n = min(battle._max_team_size, 4)
        return "".join(str(i) for i in range(1, n + 1))

    # ------------------------------------------------------------------
    # Random policy for p2
    # ------------------------------------------------------------------

    def _random_order_p2(self) -> str:
        battle = self._stream.battle_p2
        try:
            action_state = build_legal_action_state(battle)
        except Exception:
            return "default"

        legal0 = np.flatnonzero(action_state.joint_mask[0])
        if not legal0.size:
            return "default"
        a0 = int(self._rng.choice(legal0))
        legal1 = np.flatnonzero(action_state.joint_mask[a0 + 1])
        a1 = int(self._rng.choice(legal1)) if legal1.size else a0

        o0 = action_to_order(a0, battle, 0, action_state)
        o1 = action_to_order(a1, battle, 1, action_state)
        return _order_to_sim_choice(DoubleBattleOrder(o0, o1))

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    def reset(self) -> tuple[dict, np.ndarray]:
        if self._stream is not None:
            self._stream.close()

        p1_team, p2_team = self._pool.sample_packed_pair()
        self._stream = BattleStreamProcess(
            self._format, p1_team, p2_team,
            gen=self._gen, showdown_dir=self._showdown_dir,
        )
        self._stream.read_until_requests()

        if self._stream.battle_p1.teampreview:
            p1_order = self._preview_order(self._stream.battle_p1)
            p2_order = self._preview_order(self._stream.battle_p2)
            self._stream.send_team_preview(p1_order, p2_order)
            self._stream.read_until_requests()

        battle = self._stream.battle_p1
        obs = self._obs_builder.build(battle)
        action_state = build_legal_action_state(battle)
        return obs, action_state.joint_mask

    def step(
        self, action: np.ndarray
    ) -> tuple[dict, np.ndarray, float, bool, dict]:
        stream = self._stream
        battle = stream.battle_p1

        action_state = build_legal_action_state(battle)
        a0, a1 = int(action[0]), int(action[1])
        o0 = action_to_order(a0, battle, 0, action_state)
        o1 = action_to_order(a1, battle, 1, action_state)
        p1_choice = _order_to_sim_choice(DoubleBattleOrder(o0, o1))

        if stream.battle_p2._wait:
            stream._write(f">p1 {p1_choice}")
        else:
            p2_choice = self._random_order_p2()
            stream.send_choices(p1_choice, p2_choice)

        stream.read_until_requests()

        # p2 has a forceswitch while p1 waits — resolve before returning to caller.
        # Use need_p1=False: p1 won't get a new sideupdate until p2 acts, so
        # read_until_requests breaks as soon as p2's response (or error-retry) arrives.
        while not stream.done and stream.battle_p1._wait:
            p2_choice = self._random_order_p2()
            stream._write(f">p2 {p2_choice}")
            stream.read_until_requests(need_p1=False)

        done = stream.done
        reward = 0.0
        if done:
            reward = 1.0 if stream.winner == "p1" else -1.0

        battle = stream.battle_p1
        obs = self._obs_builder.build(battle)
        if done:
            mask = np.zeros((ACTIONS_PER_SLOT + 1, ACTIONS_PER_SLOT), dtype=bool)
        else:
            mask = build_legal_action_state(battle).joint_mask

        return obs, mask, reward, done, {}

    def close(self) -> None:
        if self._stream:
            self._stream.close()
            self._stream = None
