from pathlib import Path

import numpy as np
from poke_env.battle import Pokemon
from poke_env.teambuilder.constant_teambuilder import ConstantTeambuilder
from poke_env.teambuilder.teambuilder import Teambuilder

from pokerl.data import GameData
from pokerl.env.observation import (
    OWN_STATS_OFFSET,
    ObservationBuilder,
    _SyntheticTeamPreviewBattle,
    _cached_empty_team_preview_observation,
    _cached_team_preview_side,
    _init_team_preview_pokemon,
    _preview_visible_species,
    build_packed_team_aux_labels,
    build_team_preview_observation,
)
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool


REPO_ROOT = Path(__file__).resolve().parents[1]


def _packed_teams(count: int = 3) -> list[str]:
    pool = load_team_pool(
        default_team_pool_manifest_path(REPO_ROOT, "gen9vgc2026regf"),
        subset="pool",
        seed=0,
    )
    return [entry.packed_team for entry in pool.entries[:count]]


def _uncached_team_preview_observation(
    game_data: GameData,
    team_size: int,
    own_packed_team: str,
    opponent_packed_team: str,
) -> dict[str, np.ndarray]:
    own_team = [
        _init_team_preview_pokemon(
            Pokemon(gen=game_data.gen, teambuilder=entry),
            own_team=True,
        )
        for entry in Teambuilder.parse_packed_team(own_packed_team)[:team_size]
    ]
    opponent_team = [
        _init_team_preview_pokemon(
            Pokemon(
                gen=game_data.gen,
                species=_preview_visible_species(game_data.gen, entry),
            ),
            own_team=False,
        )
        for entry in Teambuilder.parse_packed_team(opponent_packed_team)[:team_size]
    ]
    battle = _SyntheticTeamPreviewBattle(own_team, opponent_team)
    return ObservationBuilder(game_data, team_size).build(battle)


def test_cached_team_preview_observation_matches_uncached_builder_exactly() -> None:
    game_data = GameData(gen=9)
    teams = _packed_teams()

    for own_team, opponent_team in (
        (teams[0], teams[1]),
        (teams[1], teams[0]),
        (teams[0], teams[2]),
    ):
        expected = _uncached_team_preview_observation(
            game_data,
            6,
            own_team,
            opponent_team,
        )
        actual = build_team_preview_observation(
            game_data,
            6,
            own_team,
            opponent_team,
        )

        assert actual.keys() == expected.keys()
        for key in expected:
            assert actual[key].dtype == expected[key].dtype
            assert actual[key].shape == expected[key].shape
            assert np.array_equal(actual[key], expected[key]), key


def test_preview_separates_level_50_spreads_and_keeps_opposing_stats_private() -> None:
    game_data = GameData()
    slow = ConstantTeambuilder(
        "Torkoal\nLevel: 50\nEVs: 252 HP\nQuiet Nature\nIVs: 0 Spe\n- Protect"
    ).yield_team()
    fast = ConstantTeambuilder(
        "Torkoal\nLevel: 50\nEVs: 252 HP / 252 Spe\nTimid Nature\n- Protect"
    ).yield_team()

    slow_side = build_team_preview_observation(game_data, 6, slow, fast)
    fast_side = build_team_preview_observation(game_data, 6, fast, slow)
    for obs, speed in ((slow_side, 22), (fast_side, 79)):
        stats = obs["own_continuous"][0, OWN_STATS_OFFSET:] * 256
        assert stats[0] == 177
        assert stats[-1] == speed
    np.testing.assert_array_equal(
        slow_side["opp_continuous"], fast_side["opp_continuous"]
    )


def test_omitted_tera_type_uses_primary_type_for_owner_and_private_labels() -> None:
    game_data = GameData(gen=9)
    team = ConstantTeambuilder(
        "Urshifu-Rapid-Strike\nLevel: 50\nAbility: Unseen Fist\n- Surging Strikes"
    ).yield_team()
    obs = build_team_preview_observation(game_data, 6, team, team)
    truth = build_packed_team_aux_labels(game_data, 6, team)

    fighting = game_data.type_to_idx["fighting"]
    assert obs["own_types"][0, 2] == truth["opp_types"][0, 2] == fighting
    assert obs["opp_types"][0, 2] == 0


def test_team_preview_cache_reuses_each_roster_across_different_matchups() -> None:
    game_data = GameData(gen=9)
    teams = _packed_teams()
    _cached_team_preview_side.cache_clear()
    _cached_empty_team_preview_observation.cache_clear()

    build_team_preview_observation(game_data, 6, teams[0], teams[1])
    first = _cached_team_preview_side.cache_info()
    build_team_preview_observation(game_data, 6, teams[0], teams[2])
    second = _cached_team_preview_side.cache_info()

    assert first.misses == 2
    assert second.hits == first.hits + 1
    assert second.misses == first.misses + 1


def test_team_preview_observations_do_not_alias_cached_or_sibling_arrays() -> None:
    game_data = GameData(gen=9)
    own_team, opponent_team = _packed_teams(count=2)
    first = build_team_preview_observation(
        game_data,
        6,
        own_team,
        opponent_team,
    )
    second = build_team_preview_observation(
        game_data,
        6,
        own_team,
        opponent_team,
    )

    original_species = second["own_species"].copy()
    first["own_species"].fill(-1)

    assert first["own_species"].flags.writeable
    assert second["own_species"].flags.writeable
    assert not np.shares_memory(first["own_species"], second["own_species"])
    assert np.array_equal(second["own_species"], original_species)

    cached_own = dict(_cached_team_preview_side(game_data, 6, own_team, True))
    cached_neutral = dict(_cached_empty_team_preview_observation(game_data, 6))
    assert not cached_own["own_species"].flags.writeable
    assert not cached_neutral["field"].flags.writeable
