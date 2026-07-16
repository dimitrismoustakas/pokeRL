"""Structured one-turn transitions for public battle-field features."""

from __future__ import annotations

from typing import NamedTuple

import torch

from .env.observation import (
    FIELD_OPP_SIDE_OFFSET,
    FIELD_OWN_SIDE_OFFSET,
    FIELD_TERRAIN_OFFSET,
    FIELD_TRICK_ROOM_OFFSET,
    FIELD_WEATHER_OFFSET,
)


PUBLIC_FIELD_DIM = FIELD_TRICK_ROOM_OFFSET + 1
WEATHER_CLASS_SLICE = slice(FIELD_WEATHER_OFFSET, FIELD_WEATHER_OFFSET + 9)
WEATHER_AGE_INDEX = FIELD_WEATHER_OFFSET + 9
TERRAIN_CLASS_SLICE = slice(FIELD_TERRAIN_OFFSET, FIELD_TERRAIN_OFFSET + 5)
TERRAIN_AGE_INDEX = FIELD_TERRAIN_OFFSET + 5

_OWN_SCREEN_AGE_INDICES = tuple(FIELD_OWN_SIDE_OFFSET + offset for offset in (4, 5, 6))
_OPP_SCREEN_AGE_INDICES = tuple(FIELD_OPP_SIDE_OFFSET + offset for offset in (4, 5, 6))
_TAILWIND_AGE_INDICES = (
    FIELD_OWN_SIDE_OFFSET + 7,
    FIELD_OPP_SIDE_OFFSET + 7,
)
TIMED_AGE_STEPS = {
    WEATHER_AGE_INDEX: 1.0 / 8.0,
    TERRAIN_AGE_INDEX: 1.0 / 8.0,
    **{index: 1.0 / 8.0 for index in _OWN_SCREEN_AGE_INDICES},
    **{index: 1.0 / 8.0 for index in _OPP_SCREEN_AGE_INDICES},
    **{index: 1.0 / 4.0 for index in _TAILWIND_AGE_INDICES},
    FIELD_TRICK_ROOM_OFFSET: 1.0 / 5.0,
}

CATEGORICAL_FIELD_GROUPS = (WEATHER_CLASS_SLICE, TERRAIN_CLASS_SLICE)
SCALAR_FIELD_SLICES = (
    slice(WEATHER_AGE_INDEX, WEATHER_AGE_INDEX + 1),
    slice(TERRAIN_AGE_INDEX, PUBLIC_FIELD_DIM),
)
SCALAR_FIELD_INDICES = tuple(
    index
    for index in range(PUBLIC_FIELD_DIM)
    if index not in range(WEATHER_CLASS_SLICE.start, WEATHER_CLASS_SLICE.stop)
    and index not in range(TERRAIN_CLASS_SLICE.start, TERRAIN_CLASS_SLICE.stop)
)
FIELD_CHANGE_COMPONENT_DIM = len(CATEGORICAL_FIELD_GROUPS) + len(SCALAR_FIELD_INDICES)


class FieldTransitionOutput(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    change_probabilities: torch.Tensor
    alternative_probabilities: torch.Tensor
    persistence: torch.Tensor


def _require_public_field(tensor: torch.Tensor, name: str) -> None:
    if tensor.ndim < 1 or tensor.shape[-1] != PUBLIC_FIELD_DIM:
        raise ValueError(
            f"{name} must end with {PUBLIC_FIELD_DIM} public field features."
        )
    if not tensor.is_floating_point():
        raise ValueError(f"{name} must use a floating-point dtype.")


def advanced_public_field_persistence(current: torch.Tensor) -> torch.Tensor:
    """Advance public timers one turn while preserving every other component.

    This is the deterministic no-event transition. Weather and terrain ages
    advance only when their non-``none`` class is active. Screens, Tailwind,
    and Trick Room have zero-as-absence encodings, so their ages advance only
    from positive values. Expiry, replacement, setup, and hazard changes are
    learned deviations from this baseline.
    """
    _require_public_field(current, "current field")
    baseline = current.clone()
    weather_active = current[..., WEATHER_CLASS_SLICE][..., 1:].sum(dim=-1) > 0.5
    terrain_active = current[..., TERRAIN_CLASS_SLICE][..., 1:].sum(dim=-1) > 0.5
    explicitly_active = {
        WEATHER_AGE_INDEX: weather_active,
        TERRAIN_AGE_INDEX: terrain_active,
    }
    for index, step in TIMED_AGE_STEPS.items():
        active = explicitly_active.get(index, current[..., index] > 0.0)
        advanced = (current[..., index] + step).clamp(max=1.0)
        baseline[..., index] = torch.where(
            active,
            advanced,
            torch.zeros_like(advanced),
        )
    return baseline


def public_field_change_targets(
    targets: torch.Tensor,
    persistence: torch.Tensor,
    *,
    tolerance: float = 1e-6,
) -> torch.Tensor:
    """Return one change label per categorical group and scalar component."""
    _require_public_field(targets, "field targets")
    _require_public_field(persistence, "field persistence")
    if targets.shape != persistence.shape:
        raise ValueError("Field targets and persistence must have identical shapes.")
    categorical_changes = [
        (
            targets[..., group].argmax(dim=-1) != persistence[..., group].argmax(dim=-1)
        ).to(targets.dtype)
        for group in CATEGORICAL_FIELD_GROUPS
    ]
    scalar_changes = torch.cat(
        [
            (targets[..., group] - persistence[..., group]).abs() > tolerance
            for group in SCALAR_FIELD_SLICES
        ],
        dim=-1,
    )
    return torch.cat(
        [
            torch.stack(categorical_changes, dim=-1),
            scalar_changes.to(targets.dtype),
        ],
        dim=-1,
    )


def decode_public_field_transition(
    *,
    persistence: torch.Tensor,
    change_logits: torch.Tensor,
    alternative_logits: torch.Tensor,
    probability_epsilon: float = 1e-6,
) -> FieldTransitionOutput:
    """Mix typed alternatives with persistence through component change gates."""
    _require_public_field(persistence, "field persistence")
    _require_public_field(alternative_logits, "field alternative logits")
    if persistence.shape != alternative_logits.shape:
        raise ValueError(
            "Field persistence and alternative logits must have identical shapes."
        )
    if change_logits.shape != (*persistence.shape[:-1], FIELD_CHANGE_COMPONENT_DIM):
        raise ValueError(
            "Field change logits must align with persistence and end with "
            f"{FIELD_CHANGE_COMPONENT_DIM} components."
        )
    if not 0.0 < probability_epsilon < 0.5:
        raise ValueError("probability_epsilon must be in (0, 0.5).")

    change_probabilities = change_logits.sigmoid()
    alternative_probabilities = torch.empty_like(alternative_logits)
    alternative_probabilities[..., WEATHER_CLASS_SLICE] = alternative_logits[
        ..., WEATHER_CLASS_SLICE
    ].softmax(dim=-1)
    alternative_probabilities[..., TERRAIN_CLASS_SLICE] = alternative_logits[
        ..., TERRAIN_CLASS_SLICE
    ].softmax(dim=-1)
    for group in SCALAR_FIELD_SLICES:
        alternative_probabilities[..., group] = alternative_logits[..., group].sigmoid()

    probabilities = torch.empty_like(persistence)
    for component, group in enumerate(CATEGORICAL_FIELD_GROUPS):
        gate = change_probabilities[..., component].unsqueeze(-1)
        probabilities[..., group] = (1.0 - gate) * persistence[
            ..., group
        ] + gate * alternative_probabilities[..., group]
    component_offset = len(CATEGORICAL_FIELD_GROUPS)
    for group in SCALAR_FIELD_SLICES:
        group_width = group.stop - group.start
        scalar_gates = change_probabilities[
            ..., component_offset : component_offset + group_width
        ]
        probabilities[..., group] = (1.0 - scalar_gates) * persistence[
            ..., group
        ] + scalar_gates * alternative_probabilities[..., group]
        component_offset += group_width
    probabilities = probabilities.clamp(0.0, 1.0)
    logits = torch.logit(
        probabilities.clamp(probability_epsilon, 1.0 - probability_epsilon)
    )
    return FieldTransitionOutput(
        logits=logits,
        probabilities=probabilities,
        change_probabilities=change_probabilities,
        alternative_probabilities=alternative_probabilities,
        persistence=persistence,
    )
