import pytest
import torch

from pokerl.field_transition import (
    FIELD_CHANGE_COMPONENT_DIM,
    SCALAR_FIELD_INDICES,
    TERRAIN_AGE_INDEX,
    TERRAIN_CLASS_SLICE,
    WEATHER_AGE_INDEX,
    WEATHER_CLASS_SLICE,
    advanced_public_field_persistence,
    decode_public_field_transition,
    public_field_change_targets,
)


def _field(batch: int = 1) -> torch.Tensor:
    field = torch.zeros(batch, 33)
    field[..., WEATHER_CLASS_SLICE.start] = 1.0
    field[..., TERRAIN_CLASS_SLICE.start] = 1.0
    return field


def test_advanced_persistence_advances_only_active_timers() -> None:
    field = _field()
    field[..., WEATHER_CLASS_SLICE] = 0.0
    field[..., WEATHER_CLASS_SLICE.start + 2] = 1.0
    field[..., WEATHER_AGE_INDEX] = 0.25
    field[..., TERRAIN_AGE_INDEX] = 0.75  # no terrain: must reset to zero
    field[..., 20] = 0.25  # own Reflect
    field[..., 23] = 0.50  # own Tailwind
    field[..., 32] = 0.20  # Trick Room

    advanced = advanced_public_field_persistence(field)

    assert advanced[..., WEATHER_AGE_INDEX] == pytest.approx(0.375)
    assert advanced[..., TERRAIN_AGE_INDEX] == 0.0
    assert advanced[..., 20] == pytest.approx(0.375)
    assert advanced[..., 23] == pytest.approx(0.75)
    assert advanced[..., 32] == pytest.approx(0.40)
    assert advanced[..., 21] == 0.0


def test_near_zero_change_gate_matches_advanced_persistence() -> None:
    persistence = advanced_public_field_persistence(_field(batch=3))
    change_logits = torch.full((3, FIELD_CHANGE_COMPONENT_DIM), -12.0)
    alternative_logits = torch.randn(3, 33)

    decoded = decode_public_field_transition(
        persistence=persistence,
        change_logits=change_logits,
        alternative_logits=alternative_logits,
    )

    torch.testing.assert_close(decoded.probabilities, persistence, atol=7e-6, rtol=0)


def test_categorical_transition_groups_remain_valid_probabilities() -> None:
    persistence = _field(batch=5)
    decoded = decode_public_field_transition(
        persistence=persistence,
        change_logits=torch.randn(5, FIELD_CHANGE_COMPONENT_DIM),
        alternative_logits=torch.randn(5, 33),
    )

    torch.testing.assert_close(
        decoded.probabilities[..., WEATHER_CLASS_SLICE].sum(dim=-1),
        torch.ones(5),
    )
    torch.testing.assert_close(
        decoded.probabilities[..., TERRAIN_CLASS_SLICE].sum(dim=-1),
        torch.ones(5),
    )
    assert torch.all(
        (decoded.probabilities[..., SCALAR_FIELD_INDICES] >= 0.0)
        & (decoded.probabilities[..., SCALAR_FIELD_INDICES] <= 1.0)
    )


def test_change_targets_group_categories_and_scalar_values() -> None:
    persistence = _field()
    targets = persistence.clone()
    targets[..., WEATHER_CLASS_SLICE] = 0.0
    targets[..., WEATHER_CLASS_SLICE.start + 1] = 1.0
    targets[..., 16] = 1.0

    changes = public_field_change_targets(targets, persistence)

    assert changes.shape == (1, FIELD_CHANGE_COMPONENT_DIM)
    assert changes[..., 0] == 1.0
    assert changes[..., 1] == 0.0
    scalar_component = 2 + SCALAR_FIELD_INDICES.index(16)
    assert changes[..., scalar_component] == 1.0
