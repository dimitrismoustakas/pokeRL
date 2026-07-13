"""Pure additive diagnostics for trainer-only opponent belief heads."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..env.actions import (
    NUM_SLOT_INTENT_COMPONENTS,
    NUM_TERA_INTENT_COMPONENTS,
    SLOT_INTENT_KIND,
    SLOT_INTENT_MOVE,
    SLOT_INTENT_SWITCH_SPECIES,
    SLOT_INTENT_TARGET,
    TERA_INTENT_SLOT,
    TERA_INTENT_USE,
)
from ..env.observation import (
    ABILITY_KNOWN_OFFSET,
    ITEM_KNOWN_OFFSET,
    TERA_KNOWN_OFFSET,
)


_CLASSIFICATION_TASKS = (
    ("item", "item_logits", "opp_item", ITEM_KNOWN_OFFSET),
    ("ability", "ability_logits", "opp_ability", ABILITY_KNOWN_OFFSET),
    ("tera", "tera_type_logits", None, TERA_KNOWN_OFFSET),
)


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")


def _classification_additive_sums(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    selected: torch.Tensor,
    prefix: str,
) -> dict[str, float]:
    class_count = logits.shape[-1]
    flat_logits = logits.reshape(-1, class_count)
    flat_targets = targets.reshape(-1)
    flat_selected = selected.reshape(-1)
    selected_logits = flat_logits[flat_selected]
    selected_targets = flat_targets[flat_selected]
    count = int(selected_targets.numel())
    if count == 0:
        return {
            f"{prefix}_top1_correct": 0.0,
            f"{prefix}_count": 0.0,
            f"{prefix}_nll_sum": 0.0,
            f"{prefix}_brier_sum": 0.0,
        }
    if ((selected_targets < 0) | (selected_targets >= class_count)).any():
        raise ValueError(f"{prefix} labels contain an out-of-range class index.")

    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = log_probs.exp()
    target_probs = probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1)
    nll = -log_probs.gather(1, selected_targets.unsqueeze(1)).squeeze(1)
    # Multiclass Brier score: sum_c (p_c - 1[c=y])^2.
    brier = probs.square().sum(dim=-1) - 2.0 * target_probs + 1.0
    correct = (selected_logits.argmax(dim=-1) == selected_targets).sum()
    return {
        f"{prefix}_top1_correct": float(correct.item()),
        f"{prefix}_count": float(count),
        f"{prefix}_nll_sum": float(nll.sum().item()),
        f"{prefix}_brier_sum": float(brier.sum().item()),
    }


@torch.no_grad()
def hidden_info_belief_additive_metrics(
    logits: Mapping[str, torch.Tensor],
    terminal_labels: Mapping[str, torch.Tensor],
    label_valid: torch.Tensor,
    public_obs: Mapping[str, torch.Tensor],
) -> dict[str, float]:
    """Return additive unknown-only belief metrics.

    Terminal labels are used only as evaluation targets. Item, ability, and
    Tera metrics include a slot only when that attribute is exact in the
    terminal label and still unknown in the current public observation. Move
    recall excludes already revealed moves from both its target set and top-4
    candidate ranking. Returned values are Python sums/counts that can be
    accumulated across chunks and converted to rates or means afterwards.
    """
    required_logits = {
        "item_logits",
        "ability_logits",
        "tera_type_logits",
        "move_logits",
    }
    required_labels = {
        "opp_mask",
        "opp_continuous",
        "opp_types",
        "opp_item",
        "opp_ability",
        "opp_moves",
    }
    required_public = {"opp_mask", "opp_continuous", "opp_moves"}
    for name, required, values in (
        ("logits", required_logits, logits),
        ("terminal_labels", required_labels, terminal_labels),
        ("public_obs", required_public, public_obs),
    ):
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(f"{name} is missing key(s): {', '.join(missing)}.")

    base_shape = tuple(label_valid.shape)
    if label_valid.ndim < 1:
        raise ValueError("label_valid must include a team-slot dimension.")
    _require_shape(
        "terminal_labels['opp_mask']", terminal_labels["opp_mask"], base_shape
    )
    _require_shape("public_obs['opp_mask']", public_obs["opp_mask"], base_shape)
    _require_shape(
        "terminal_labels['opp_item']", terminal_labels["opp_item"], base_shape
    )
    _require_shape(
        "terminal_labels['opp_ability']", terminal_labels["opp_ability"], base_shape
    )
    _require_shape(
        "terminal_labels['opp_types']",
        terminal_labels["opp_types"],
        (*base_shape, 3),
    )
    terminal_continuous = terminal_labels["opp_continuous"]
    public_continuous = public_obs["opp_continuous"]
    if terminal_continuous.shape[:-1] != base_shape:
        raise ValueError(
            "terminal opponent continuous features do not match label_valid."
        )
    if public_continuous.shape != terminal_continuous.shape:
        raise ValueError(
            "public and terminal opponent continuous features must have the same shape."
        )
    required_continuous_width = (
        max(
            ITEM_KNOWN_OFFSET,
            ABILITY_KNOWN_OFFSET,
            TERA_KNOWN_OFFSET,
        )
        + 1
    )
    if terminal_continuous.shape[-1] < required_continuous_width:
        raise ValueError("opponent continuous features omit hidden-info known flags.")

    terminal_moves = terminal_labels["opp_moves"]
    public_moves = public_obs["opp_moves"]
    if (
        terminal_moves.shape[:-1] != base_shape
        or public_moves.shape != terminal_moves.shape
    ):
        raise ValueError(
            "public and terminal opponent moves must share label_valid's leading shape."
        )

    base_mask = (
        label_valid.to(torch.bool)
        & ~terminal_labels["opp_mask"].to(torch.bool)
        & ~public_obs["opp_mask"].to(torch.bool)
    )
    result: dict[str, float] = {}
    for task, logits_key, label_key, known_offset in _CLASSIFICATION_TASKS:
        task_logits = logits[logits_key]
        if task_logits.shape[:-1] != base_shape:
            raise ValueError(f"{logits_key} leading dimensions must match label_valid.")
        terminal_known = terminal_continuous[..., known_offset] > 0.5
        public_known = public_continuous[..., known_offset] > 0.5
        selected = base_mask & terminal_known & ~public_known
        if task == "tera":
            targets = terminal_labels["opp_types"][..., 2]
            selected = selected & (targets > 0)
        else:
            targets = terminal_labels[label_key]
        result.update(
            _classification_additive_sums(
                logits=task_logits,
                targets=targets,
                selected=selected,
                prefix=f"aux_{task}_unknown",
            )
        )

    move_logits = logits["move_logits"]
    if move_logits.shape[:-1] != base_shape:
        raise ValueError("move_logits leading dimensions must match label_valid.")
    num_moves = move_logits.shape[-1]
    if num_moves <= 1:
        raise ValueError(
            "move_logits must include padding plus at least one move class."
        )
    terminal_move_valid = terminal_moves > 0
    public_move_valid = public_moves > 0
    if (terminal_moves[terminal_move_valid] >= num_moves).any() or (
        public_moves[public_move_valid] >= num_moves
    ).any():
        raise ValueError("opponent move labels contain an out-of-range move index.")

    exact_multihot = torch.zeros_like(move_logits, dtype=torch.bool)
    exact_multihot.scatter_(
        -1, terminal_moves.clamp(0, num_moves - 1), terminal_move_valid
    )
    revealed_multihot = torch.zeros_like(move_logits, dtype=torch.bool)
    revealed_multihot.scatter_(
        -1, public_moves.clamp(0, num_moves - 1), public_move_valid
    )
    exact_multihot[..., 0] = False
    revealed_multihot[..., 0] = True
    unknown_targets = exact_multihot & ~revealed_multihot & base_mask.unsqueeze(-1)

    candidate_logits = move_logits.masked_fill(revealed_multihot, -torch.inf)
    top_k = min(4, num_moves - 1)
    top_indices = candidate_logits.topk(top_k, dim=-1).indices
    predicted_multihot = torch.zeros_like(move_logits, dtype=torch.bool)
    predicted_multihot.scatter_(-1, top_indices, True)
    move_correct = (predicted_multihot & unknown_targets).sum()
    move_count = unknown_targets.sum()
    result["aux_move_unknown_recall_at_4_correct"] = float(move_correct.item())
    result["aux_move_unknown_recall_at_4_count"] = float(move_count.item())
    return result


@torch.no_grad()
def opponent_intent_additive_metrics(
    logits: Mapping[str, torch.Tensor],
    slot_labels: torch.Tensor,
    slot_masks: torch.Tensor,
    tera_labels: torch.Tensor,
    tera_masks: torch.Tensor,
) -> dict[str, float]:
    """Return additive diagnostics for non-forced semantic intent choices."""
    required = {
        "opp_intent_kind_logits",
        "opp_intent_move_logits",
        "opp_intent_target_logits",
        "opp_intent_switch_species_logits",
        "opp_intent_tera_use_logits",
        "opp_intent_tera_slot_logits",
    }
    missing = sorted(required.difference(logits))
    if missing:
        raise ValueError(f"logits is missing key(s): {', '.join(missing)}.")
    if slot_labels.shape != slot_masks.shape or slot_labels.shape[-2:] != (
        2,
        NUM_SLOT_INTENT_COMPONENTS,
    ):
        raise ValueError(
            "opponent slot intent labels/masks must share shape (..., 2, 4)."
        )
    base_shape = tuple(slot_labels.shape[:-2])
    expected_tera_shape = (*base_shape, NUM_TERA_INTENT_COMPONENTS)
    if (
        tera_labels.shape != expected_tera_shape
        or tera_masks.shape != expected_tera_shape
    ):
        raise ValueError("opponent Tera intent labels/masks must have shape (..., 2).")

    result: dict[str, float] = {}
    for name, logits_key, component in (
        ("kind", "opp_intent_kind_logits", SLOT_INTENT_KIND),
        ("move", "opp_intent_move_logits", SLOT_INTENT_MOVE),
        ("target", "opp_intent_target_logits", SLOT_INTENT_TARGET),
        (
            "switch_species",
            "opp_intent_switch_species_logits",
            SLOT_INTENT_SWITCH_SPECIES,
        ),
    ):
        component_logits = logits[logits_key]
        if component_logits.shape[:-1] != (*base_shape, 2):
            raise ValueError(f"{logits_key} leading dimensions must match slot labels.")
        targets = slot_labels[..., component]
        selected = slot_masks[..., component].to(torch.bool)
        result.update(
            _classification_additive_sums(
                logits=component_logits,
                targets=targets,
                selected=selected,
                prefix=f"opp_intent_{name}",
            )
        )

    for name, logits_key, component in (
        ("tera_use", "opp_intent_tera_use_logits", TERA_INTENT_USE),
        ("tera_slot", "opp_intent_tera_slot_logits", TERA_INTENT_SLOT),
    ):
        component_logits = logits[logits_key]
        if component_logits.shape[:-1] != base_shape:
            raise ValueError(f"{logits_key} leading dimensions must match Tera labels.")
        result.update(
            _classification_additive_sums(
                logits=component_logits,
                targets=tera_labels[..., component],
                selected=tera_masks[..., component].to(torch.bool),
                prefix=f"opp_intent_{name}",
            )
        )
    return result
