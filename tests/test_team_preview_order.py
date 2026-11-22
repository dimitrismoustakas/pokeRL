import numpy as np
import pytest

from pokerl.env.team_preview import parse_team_preview_order


def test_parse_team_preview_order_returns_zero_based_indices() -> None:
    picks = parse_team_preview_order("/team 1526", team_size=6, pick_size=4)
    np.testing.assert_array_equal(picks, np.array([0, 4, 1, 5], dtype=np.int64))


@pytest.mark.parametrize(
    ("team_order", "message"),
    [
        ("1526", "starting with"),
        ("/team 152", "Expected 4 team preview picks"),
        ("/team 1556", "must not repeat"),
        ("/team 1576", "out of range"),
        ("/team 15a6", "only slot digits"),
    ],
)
def test_parse_team_preview_order_rejects_invalid_commands(
    team_order: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_team_preview_order(team_order, team_size=6, pick_size=4)
