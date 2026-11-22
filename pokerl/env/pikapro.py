"""In-process Regulation F battles on the poke-env player protocol."""

import asyncio
import json
import os
import re

import pikapro

from pokerl.config import REGULATION_F

_TIER = "[Gen 9] VGC 2026 Reg F"
_HP_FIELDS = {
    "switch": ((2, 4),),
    "drag": ((2, 4),),
    "replace": ((2, 4),),
    "-damage": ((2, 3),),
    "-heal": ((2, 3),),
    "-sethp": ((2, 3), (4, 5)),
}
_EXACT_HP = re.compile(r"(\d+)/(\d+)( .*)?")


def _player_message(line: str, opponent: str) -> list[str]:
    fields = line.split("|")
    if len(fields) < 2:
        return fields
    if fields[1] == "tier":
        fields[2] = _TIER
    for actor, hp in _HP_FIELDS.get(fields[1], ()):
        if hp >= len(fields) or not fields[actor].startswith(opponent):
            continue
        match = _EXACT_HP.fullmatch(fields[hp])
        if match:
            current, maximum = int(match[1]), int(match[2])
            percent = (100 * current + maximum - 1) // maximum
            if current < maximum and percent == 100:
                percent = 99
            fields[hp] = f"{percent}/100{match[3] or ''}"
    return fields


class PikaproBattle:
    """A battle owned by one local player-session loop."""

    def __init__(self):
        self._inbox: asyncio.Queue[dict] = asyncio.Queue()
        self._battle: pikapro.Battle | None = None
        self._cursor = 0

    async def start_battle(self, lines: list[str]) -> None:
        start, p1, p2 = lines
        options = json.loads(start.removeprefix(">start "))
        if options["formatid"] != REGULATION_F:
            raise ValueError(f"Pikapro supports {REGULATION_F} in pokeRL.")
        players = [
            json.loads(p1.removeprefix(">player p1 ")),
            json.loads(p2.removeprefix(">player p2 ")),
        ]
        self._battle = pikapro.Battle(
            "gen9vgc",
            [p["team"] for p in players],
            seed=options["seed"] if "seed" in options else os.urandom(32),
            players=[p["name"] for p in players],
        )
        self._publish(opening=True)

    def _publish(self, *, opening: bool = False) -> None:
        battle = self._battle
        assert battle is not None
        header = battle.header() if opening else []
        updates = []
        for seat, perspective, opponent in ((0, "p1", "p2"), (1, "p2", "p1")):
            lines = header + battle.log(perspective, self._cursor)
            messages = [_player_message(line, opponent) for line in lines]
            if not battle.over:
                messages.append(["", "request", battle.request_json(seat)])
            updates.append(messages)
        self._cursor = battle.event_count
        self._enqueue(updates[0], updates[1])

    def _enqueue(self, p1: list[list[str]], p2: list[list[str]]) -> None:
        self._inbox.put_nowait(
            {
                "type": "protocol-batch",
                "messages": [
                    {"type": "split-chunk", "p1_messages": p1, "p2_messages": p2}
                ],
            }
        )

    async def send_battle_line(self, line: str) -> None:
        await self.send_battle_lines([line])

    async def send_battle_lines(self, lines: list[str]) -> None:
        battle = self._battle
        assert battle is not None
        for line in lines:
            player, command = line.split(" ", 1)
            if player == ">forcelose":
                battle.forfeit({"p1": 0, "p2": 1}[command])
                self._publish()
                continue
            seat = {">p1": 0, ">p2": 1}[player]
            try:
                resolved = battle.submit(seat, command)
            except ValueError as exc:
                if not str(exc).startswith("[Unavailable choice]"):
                    raise
                # Imprison and similar hidden restrictions revise only this seat.
                revised = [
                    ["", "error", str(exc)],
                    ["", "request", battle.request_json(seat)],
                ]
                self._enqueue(
                    revised if seat == 0 else [], revised if seat == 1 else []
                )
            else:
                if resolved:
                    self._publish()

    async def read_protocol_message(self, timeout: float | None) -> dict:
        if not self._inbox.empty():
            return self._inbox.get_nowait()
        return await asyncio.wait_for(self._inbox.get(), timeout)

    async def close_battle(self) -> None:
        self._battle = None

    async def describe(self) -> str:
        return f"Pikapro turn {self._battle.turn}" if self._battle else "Pikapro idle"
