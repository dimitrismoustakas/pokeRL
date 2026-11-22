import pytest

from pokerl.data import GameData


def test_game_data_preserves_checkpoint_item_indices() -> None:
    gd = GameData(gen=9)
    assert len(gd.item_to_idx) == 661
    assert gd.item_index("leftovers") == 281
    assert gd.item_index("vilevial") == 636
    assert gd.item_index("widelens") == 652
    assert gd.item_index("example-previously-unobserved-item") == 1714


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


def test_move_mechanics_are_generic_dex_facts() -> None:
    gd = GameData(gen=9)

    mechanics = gd.move_mechanics_matrix()

    assert mechanics[gd.move_to_idx["protect"], 0] == 1.0
    assert mechanics[gd.move_to_idx["uturn"], 1] == 1.0
    assert mechanics[gd.move_to_idx["whirlwind"], 2] == 1.0
    assert mechanics[gd.move_to_idx["recover"], 3] == 1.0
    assert mechanics[gd.move_to_idx["gigadrain"], 4] == pytest.approx(0.5)
    assert mechanics[gd.move_to_idx["doubleedge"], 5] > 0.0
    assert mechanics[gd.move_to_idx["trickroom"], 12] == 1.0
    assert mechanics[gd.move_to_idx["electricterrain"], 14] == 1.0
    assert mechanics[gd.move_to_idx["fakeout"], 15] == 1.0
    assert mechanics[gd.move_to_idx["solarbeam"], 16] == 1.0
    assert mechanics[gd.move_to_idx["hyperbeam"], 17] == 1.0


def test_battle_ability_ids_use_the_existing_checkpoint_rows() -> None:
    gd = GameData(gen=9)
    assert gd.abilities_to_idx["unseenfist"] == 299
    assert gd.abilities_to_idx["waterabsorb"] == 305
    assert gd.abilities_to_idx["intimidate"] == 120
    assert gd.num_abilities == 319
    assert gd.vocab_signature() == (
        "d5980c827cc17cc3e4b422ac9e4a7393ccf9e1e6f0bde200b04c71fc2a1d0821"
    )
