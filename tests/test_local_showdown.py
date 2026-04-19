import json

from pokerl.env.local_showdown import _LocalBattleParser, _normalize_choice_message


def test_normalize_choice_message_matches_battlestream_input() -> None:
    assert _normalize_choice_message("/choose move protect") == "move protect"
    assert _normalize_choice_message("/choose move protect terastallize") == "move protect terastallize"
    assert _normalize_choice_message("/choose default") == "default"
    assert _normalize_choice_message("/team 1234") == "team 1234"


def test_local_battle_parser_ignores_timestamp_lines() -> None:
    parser = _LocalBattleParser(
        battle_tag="local-test",
        username="agent",
        format_id="gen9vgc2026regf",
        packed_team=None,
    )
    parser.apply_chunk(
        "\n".join(
            [
                "|player|p1|agent||",
                "|player|p2|league||",
                "|t:|1776566371",
                "|win|agent",
            ]
        )
    )

    assert parser.finished is True
    assert parser.battle.won is True


def test_local_battle_parser_marks_request_pending_from_player_stream_request() -> None:
    parser = _LocalBattleParser(
        battle_tag="local-request",
        username="agent",
        format_id="gen9vgc2026regf",
        packed_team=None,
    )
    request = {
        "teamPreview": True,
        "maxChosenTeamSize": 4,
        "side": {
            "name": "agent",
            "id": "p1",
            "pokemon": [
                {
                    "ident": "p1: Pikachu",
                    "details": "Pikachu, L50, M",
                    "condition": "100/100",
                    "active": True,
                    "stats": {"atk": 1, "def": 1, "spa": 1, "spd": 1, "spe": 1},
                    "moves": ["thunderbolt"],
                    "baseAbility": "static",
                    "item": "lightball",
                    "pokeball": "pokeball",
                    "ability": "static",
                    "commanding": False,
                    "reviving": False,
                    "teraType": "Electric",
                    "terastallized": "",
                }
            ],
        },
    }
    parser.apply_chunk(
        "\n".join(
            [
                "|player|p1|agent||",
                "|player|p2|league||",
                f"|request|{json.dumps(request, separators=(',', ':'))}",
            ]
        )
    )

    assert parser.request_pending is True
    assert parser.battle.teampreview is True


def test_local_battle_parser_ignores_retryable_choice_errors() -> None:
    parser = _LocalBattleParser(
        battle_tag="local-error",
        username="league",
        format_id="gen9vgc2026regf",
        packed_team=None,
    )

    parser.apply_chunk("|error|[Unavailable choice] Can't move: Follow Me is disabled")
    parser.apply_chunk("|error|[Invalid choice] Can't switch: slot empty")

    assert parser.finished is False
    assert parser.request_pending is False
