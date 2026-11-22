from pathlib import Path

import pytest
from poke_env.teambuilder.teambuilder import Teambuilder

from pokerl.team_pool import TeamPool, TeamPoolEntry, load_team_pool


def test_load_team_pool_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        """
        {
          "format": "test-format",
          "pool": [
            {
              "title": "duplicate",
              "file": "first.txt",
              "file": "second.txt"
            }
          ]
        }
        """,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_team_pool(manifest_path)


def test_checked_in_team_pool_manifest_loads() -> None:
    manifest_path = Path("data/team_pools/gen9vgc2026regf/manifest.json")

    pool = load_team_pool(manifest_path, subset="pool", seed=0)

    assert pool.size == 85
    assert {
        pokemon.level
        for entry in pool.entries
        for pokemon in Teambuilder.parse_packed_team(entry.packed_team)
    } == {50}


def test_heldout_team_pool_is_evaluation_only() -> None:
    manifest_path = Path("data/team_pools/gen9vgc2026regf_heldout/manifest.json")

    with pytest.raises(ValueError, match="evaluation-only"):
        load_team_pool(manifest_path, subset="pool", seed=0)

    heldout = load_team_pool(
        manifest_path,
        subset="pool",
        seed=0,
        usage="evaluation",
    )
    assert heldout.size == 15
    assert {
        pokemon.level
        for entry in heldout.entries
        for pokemon in Teambuilder.parse_packed_team(entry.packed_team)
    } == {50}


def test_team_pool_training_metadata_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"format":"test","purpose":"evaluation_only","pool":[]}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="boolean allow_training"):
        load_team_pool(manifest_path)


def test_team_pool_rng_state_and_population_signature_roundtrip() -> None:
    entries = [
        TeamPoolEntry(title="A", file="a", packed_team="a-packed"),
        TeamPoolEntry(title="B", file="b", packed_team="b-packed"),
        TeamPoolEntry(title="C", file="c", packed_team="c-packed"),
    ]
    source = TeamPool(format_id="test", subset="pool", entries=entries, seed=17)
    state = source.rng_state()
    expected = [source.sample_packed_pair() for _ in range(10)]

    restored = TeamPool(format_id="test", subset="pool", entries=entries, seed=999)
    restored.restore_rng_state(state)

    assert [restored.sample_packed_pair() for _ in range(10)] == expected
    assert restored.signature() == source.signature()
    reordered = TeamPool(
        format_id="test",
        subset="pool",
        entries=list(reversed(entries)),
        seed=17,
    )
    assert reordered.signature() != source.signature()
