from pathlib import Path

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
