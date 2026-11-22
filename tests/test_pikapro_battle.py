import asyncio
import json
from pathlib import Path

import pytest

from pokerl.env.pikapro import PikaproBattle, _player_message
from pokerl.team_pool import load_team_pool


@pytest.mark.parametrize(
    ("line", "public"),
    [
        ("|switch|p2a: Rillaboom|Rillaboom, L50|132/206", "65/100"),
        ("|-damage|p2b: Flutter Mane|143/144 brn", "99/100 brn"),
        ("|-heal|p2a: Rillaboom|206/206", "100/100"),
        ("|-damage|p2b: Flutter Mane|0 fnt", "0 fnt"),
    ],
)
def test_health_is_projected_per_player_in_each_event(line: str, public: str) -> None:
    assert _player_message(line, "p2")[-1] == public
    assert _player_message(line, "p1") == line.split("|")


def test_sethp_preserves_own_and_rounds_opponent_in_same_message() -> None:
    line = "|-sethp|p1a: One|114/155|p2a: Two|115/224"
    assert _player_message(line, "p2")[3::2] == ["114/155", "52/100"]
    assert _player_message(line, "p1")[3::2] == ["74/100", "115/224"]


@pytest.mark.parametrize("affected", [0, 1])
def test_imprison_private_replacement_reaches_next_turn(affected: int) -> None:
    pool = load_team_pool(
        Path(__file__).resolve().parents[1]
        / "data/team_pools/gen9vgc2026regf/manifest.json"
    )
    teams = [
        next(e.packed_team for e in pool.entries if name in e.file)
        for name in ("008_arch-team", "039_orlando-17th")
    ]
    if affected == 1:
        teams.reverse()
    order = (affected + 1, 2 - affected)

    async def play():
        battle = PikaproBattle()
        start = {"formatid": "gen9vgc2026regf", "seed": [638, 42, 83, 71]}
        await battle.start_battle(
            [">start " + json.dumps(start)]
            + [
                f">player p{slot} " + json.dumps({"name": f"seat{slot}", "team": team})
                for slot, team in enumerate(teams, 1)
            ]
        )
        await battle.read_protocol_message(None)
        for moves in (
            ["team 1234", "team 1234"],
            ["switch 3, move followme", "switch 3, move imprison"],
            ["switch 4, move followme", "switch 4, move followme"],
        ):
            await battle.send_battle_lines(
                [f">p{p} {m}" for p, m in sorted(zip(order, moves, strict=True))]
            )
            packet = await battle.read_protocol_message(None)
        chunk = packet["messages"][0]
        assert chunk[f"p{order[1]}_messages"] == []
        error, request = chunk[f"p{order[0]}_messages"]
        assert error[1] == "error" and error[2].startswith("[Unavailable choice]")
        assert json.loads(request[2])["active"][1]["moves"][0]["disabled"]
        await battle.send_battle_line(f">p{order[0]} switch 4, move thunderwave 1")
        next_chunk = (await battle.read_protocol_message(None))["messages"][0]
        for key in ("p1_messages", "p2_messages"):
            assert ["", "turn", "3"] in next_chunk[key]
        await battle.close_battle()

    asyncio.run(play())
