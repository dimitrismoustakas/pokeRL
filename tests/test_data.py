from pathlib import Path

import pytest

from pokerl.data import GameData


def test_game_data_uses_named_showdown_items(monkeypatch) -> None:
    monkeypatch.setattr(
        GameData,
        "_showdown_item_export_path",
        staticmethod(lambda gen: Path("C:/fake/items.js")),
    )
    monkeypatch.setattr(
        GameData,
        "_load_showdown_item_ids",
        classmethod(lambda cls, gen: ["leftovers", "assaultvest", "safetygoggles"]),
    )

    gd = GameData(gen=9, item_vocab_size=16)

    assert gd.item_index("leftovers") == gd.item_to_idx["leftovers"]
    assert gd.item_index("assaultvest") == gd.item_to_idx["assaultvest"]
    assert gd.item_index("safetygoggles") == gd.item_to_idx["safetygoggles"]


def test_move_stats_encode_always_hit_accuracy_as_one() -> None:
    gd = GameData(gen=9)

    stats = gd.move_stats_matrix()

    assert gd._moves["aerialace"]["accuracy"] is True
    assert stats[gd.move_to_idx["aerialace"], 1] == pytest.approx(1.0)
    assert stats[gd.move_to_idx["hydropump"], 1] == pytest.approx(0.8)
    bypasses = gd.move_bypasses_accuracy_vector()
    assert bypasses[gd.move_to_idx["aerialace"]] == 1.0
    assert bypasses[gd.move_to_idx["tackle"]] == 0.0
    ignores_evasion = gd.move_ignores_evasion_vector()
    assert gd._moves["sacredsword"]["ignoreEvasion"] is True
    assert ignores_evasion[gd.move_to_idx["sacredsword"]] == 1.0
    assert ignores_evasion[gd.move_to_idx["tackle"]] == 0.0
