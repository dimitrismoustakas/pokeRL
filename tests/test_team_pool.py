from pathlib import Path

import pytest

from pokerl.team_pool import (
    TeamPool,
    TeamPoolEntry,
    default_team_pool_manifest_path,
    load_team_pool,
)


def test_checked_in_team_pool_manifest_loads() -> None:
    manifest_path = default_team_pool_manifest_path(Path.cwd(), "gen9vgc2026regf")

    pool = load_team_pool(manifest_path, subset="pool", seed=0)

    assert pool.size == len({entry.packed_team for entry in pool.entries}) == 170
    assert {
        int(pokemon.split("|")[10])
        for entry in pool.entries
        for pokemon in entry.packed_team.split("]")
    } == {50}


@pytest.mark.parametrize(("panel", "size"), [("heldout", 15), ("heldout30", 30)])
def test_heldout_team_pool_is_evaluation_only(panel: str, size: int) -> None:
    manifest_path = Path(f"data/team_pools/gen9vgc2026regf_{panel}/manifest.json")

    with pytest.raises(ValueError, match="evaluation-only"):
        load_team_pool(manifest_path, subset="pool", seed=0)

    heldout = load_team_pool(
        manifest_path,
        subset="pool",
        seed=0,
        usage="evaluation",
    )
    assert heldout.size == size
    assert {
        int(pokemon.split("|")[10])
        for entry in heldout.entries
        for pokemon in entry.packed_team.split("]")
    } == {50}


def test_restoring_team_pool_rng_repeats_team_pairs() -> None:
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
