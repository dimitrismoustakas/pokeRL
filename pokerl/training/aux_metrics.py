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
    FIELD_TRICK_ROOM_OFFSET,
    ITEM_KNOWN_OFFSET,
    NUM_PUBLIC_TARGET_POSITIONS,
    TERA_KNOWN_OFFSET,
)
from ..public_consequence import (
    MAX_PUBLIC_HIT_COUNT,
    MOVE_BRANCH_ACTION_FAILED,
    MOVE_BRANCH_NO_TARGET,
    MOVE_BRANCH_SCOPED,
    MOVE_BRANCH_TARGET_PHASE,
    NONMOVE_REASON_CANT,
    NONMOVE_REASON_FAINTED,
    NUM_MOVE_BRANCHES,
    NUM_NONMOVE_REASONS,
    NUM_TARGET_INTERACTION_REASONS,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
    TARGET_INTERACTION_BLOCKED,
    TARGET_INTERACTION_FAILED,
    TARGET_INTERACTION_IMMUNE,
    TARGET_INTERACTION_MISS,
    TARGET_INTERACTION_PROTECTED,
)


_CLASSIFICATION_TASKS = (
    ("item", "item_logits", "opp_item", ITEM_KNOWN_OFFSET),
    ("ability", "ability_logits", "opp_ability", ABILITY_KNOWN_OFFSET),
    ("tera", "tera_type_logits", None, TERA_KNOWN_OFFSET),
)
_PUBLIC_CONSEQUENCE_NONMOVE_REASON_CLASSES = (
    (NONMOVE_REASON_CANT, "cant"),
    (NONMOVE_REASON_FAINTED, "fainted"),
)
_PUBLIC_CONSEQUENCE_MOVE_BRANCH_CLASSES = (
    (MOVE_BRANCH_ACTION_FAILED, "action_failed"),
    (MOVE_BRANCH_NO_TARGET, "no_target"),
    (MOVE_BRANCH_SCOPED, "scoped"),
    (MOVE_BRANCH_TARGET_PHASE, "target_phase"),
)
_PUBLIC_CONSEQUENCE_INTERACTION_REASON_CLASSES = (
    (TARGET_INTERACTION_MISS, "miss"),
    (TARGET_INTERACTION_IMMUNE, "immune"),
    (TARGET_INTERACTION_PROTECTED, "protected"),
    (TARGET_INTERACTION_BLOCKED, "blocked"),
    (TARGET_INTERACTION_FAILED, "failed"),
)
_PUBLIC_CONSEQUENCE_HIT_COUNT_CLASSES = tuple(
    (hit_count - 1, f"hit_{hit_count}")
    for hit_count in range(1, MAX_PUBLIC_HIT_COUNT + 1)
)


def _require_shape(name: str, tensor: torch.Tensor, shape: tuple[int, ...]) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}.")


def _classification_additive_tensors(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    selected: torch.Tensor,
    prefix: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    class_count = logits.shape[-1]
    flat_logits = logits.reshape(-1, class_count)
    flat_targets = targets.reshape(-1)
    flat_selected = selected.reshape(-1)
    selected_logits = flat_logits[flat_selected]
    selected_targets = flat_targets[flat_selected]
    invalid_targets = (selected_targets < 0) | (selected_targets >= class_count)
    # Invalid selected targets are rejected after the public helper's single
    # packed device-to-host transfer.  Clamping keeps gather non-synchronizing.
    safe_targets = selected_targets.clamp(0, class_count - 1)

    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = log_probs.exp()
    target_probs = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    # Multiclass Brier score: sum_c (p_c - 1[c=y])^2.
    brier = probs.square().sum(dim=-1) - 2.0 * target_probs + 1.0
    correct = (selected_logits.argmax(dim=-1) == safe_targets).sum()
    return (
        {
            f"{prefix}_top1_correct": correct,
            f"{prefix}_count": flat_selected.sum(),
            f"{prefix}_nll_sum": nll.sum(),
            f"{prefix}_brier_sum": brier.sum(),
        },
        invalid_targets.any(),
    )


def _binary_additive_tensors(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    selected: torch.Tensor,
    prefix: str,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return additive diagnostics for one conditionally observed Bernoulli."""

    flat_logits = logits.reshape(-1)
    flat_targets = targets.reshape(-1).to(flat_logits.dtype)
    flat_selected = selected.reshape(-1).to(torch.bool)
    selected_logits = flat_logits[flat_selected]
    selected_targets = flat_targets[flat_selected]
    invalid_targets = (
        ~torch.isfinite(selected_targets)
        | (selected_targets < 0.0)
        | (selected_targets > 1.0)
    )
    probs = selected_logits.sigmoid()
    nll = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_logits,
        selected_targets,
        reduction="sum",
    )
    brier = (probs - selected_targets).square().sum()
    predicted_positive = selected_logits >= 0.0
    positive = selected_targets >= 0.5
    negative = ~positive
    return (
        {
            f"{prefix}_correct": (predicted_positive == positive).sum(),
            f"{prefix}_count": flat_selected.sum(),
            f"{prefix}_positive": positive.sum(),
            f"{prefix}_negative": negative.sum(),
            f"{prefix}_predicted_positive": predicted_positive.sum(),
            f"{prefix}_true_positive": (predicted_positive & positive).sum(),
            f"{prefix}_true_negative": ((~predicted_positive) & negative).sum(),
            f"{prefix}_nll_sum": nll,
            f"{prefix}_brier_sum": brier,
        },
        invalid_targets.any(),
    )


def _named_classification_additive_tensors(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    selected: torch.Tensor,
    prefix: str,
    classes: tuple[tuple[int, str], ...],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return additive diagnostics for a conditional named categorical task."""

    class_count = logits.shape[-1]
    if tuple(class_index for class_index, _name in classes) != tuple(
        range(class_count)
    ):
        raise ValueError(
            f"{prefix} class names must cover indices 0..{class_count - 1}."
        )
    flat_logits = logits.reshape(-1, class_count)
    flat_targets = targets.reshape(-1)
    flat_selected = selected.reshape(-1).to(torch.bool)
    selected_logits = flat_logits[flat_selected]
    selected_targets = flat_targets[flat_selected]
    invalid_targets = (selected_targets < 0) | (selected_targets >= class_count)
    safe_targets = selected_targets.clamp(0, class_count - 1)

    log_probs = torch.log_softmax(selected_logits, dim=-1)
    probs = log_probs.exp()
    target_probs = probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    nll = -log_probs.gather(1, safe_targets.unsqueeze(1)).squeeze(1)
    brier = probs.square().sum(dim=-1) - 2.0 * target_probs + 1.0
    predictions = selected_logits.argmax(dim=-1)
    metrics = {
        f"{prefix}_top1_correct": (predictions == safe_targets).sum(),
        f"{prefix}_count": flat_selected.sum(),
        f"{prefix}_nll_sum": nll.sum(),
        f"{prefix}_brier_sum": brier.sum(),
    }
    for class_index, class_name in classes:
        class_selected = selected_targets == class_index
        class_prefix = f"{prefix}_{class_name}"
        metrics.update(
            {
                f"{class_prefix}_count": class_selected.sum(),
                f"{class_prefix}_predicted_count": (predictions == class_index).sum(),
                f"{class_prefix}_correct": (
                    predictions[class_selected] == class_index
                ).sum(),
                f"{class_prefix}_nll_sum": nll[class_selected].sum(),
                f"{class_prefix}_brier_sum": brier[class_selected].sum(),
            }
        )
    return metrics, invalid_targets.any()


def _pack_additive_metrics(
    metric_tensors: Mapping[str, torch.Tensor],
    validation_tensors: Mapping[str, torch.Tensor],
) -> tuple[dict[str, float], dict[str, float]]:
    """Materialize device scalar dictionaries with one packed host transfer."""
    metric_names = tuple(metric_tensors)
    validation_names = tuple(validation_tensors)
    packed = torch.stack(
        [
            *(metric_tensors[name] for name in metric_names),
            *(validation_tensors[name] for name in validation_names),
        ]
    )
    packed_values = packed.to(device="cpu", dtype=torch.float64).tolist()
    metric_values = packed_values[: len(metric_names)]
    validation_values = packed_values[len(metric_names) :]
    return (
        dict(zip(metric_names, metric_values, strict=True)),
        dict(zip(validation_names, validation_values, strict=True)),
    )


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
    metric_tensors: dict[str, torch.Tensor] = {}
    validation_tensors: dict[str, torch.Tensor] = {}
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
        prefix = f"aux_{task}_unknown"
        task_metrics, invalid_targets = _classification_additive_tensors(
            logits=task_logits,
            targets=targets,
            selected=selected,
            prefix=prefix,
        )
        metric_tensors.update(task_metrics)
        validation_tensors[prefix] = invalid_targets

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
    validation_tensors["moves"] = (
        (terminal_moves >= num_moves) & terminal_move_valid
    ).any() | ((public_moves >= num_moves) & public_move_valid).any()

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
    metric_tensors["aux_move_unknown_recall_at_4_correct"] = move_correct
    metric_tensors["aux_move_unknown_recall_at_4_count"] = move_count
    result, invalid = _pack_additive_metrics(metric_tensors, validation_tensors)
    for task, _logits_key, _label_key, _known_offset in _CLASSIFICATION_TASKS:
        prefix = f"aux_{task}_unknown"
        if invalid[prefix]:
            raise ValueError(f"{prefix} labels contain an out-of-range class index.")
    if invalid["moves"]:
        raise ValueError("opponent move labels contain an out-of-range move index.")
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

    slot_tasks = (
        ("kind", "opp_intent_kind_logits", SLOT_INTENT_KIND),
        ("move", "opp_intent_move_logits", SLOT_INTENT_MOVE),
        ("target", "opp_intent_target_logits", SLOT_INTENT_TARGET),
        (
            "switch_species",
            "opp_intent_switch_species_logits",
            SLOT_INTENT_SWITCH_SPECIES,
        ),
    )
    tera_tasks = (
        ("tera_use", "opp_intent_tera_use_logits", TERA_INTENT_USE),
        ("tera_slot", "opp_intent_tera_slot_logits", TERA_INTENT_SLOT),
    )
    metric_tensors: dict[str, torch.Tensor] = {}
    validation_tensors: dict[str, torch.Tensor] = {}
    for name, logits_key, component in slot_tasks:
        component_logits = logits[logits_key]
        if component_logits.shape[:-1] != (*base_shape, 2):
            raise ValueError(f"{logits_key} leading dimensions must match slot labels.")
        targets = slot_labels[..., component]
        selected = slot_masks[..., component].to(torch.bool)
        prefix = f"opp_intent_{name}"
        task_metrics, invalid_targets = _classification_additive_tensors(
            logits=component_logits,
            targets=targets,
            selected=selected,
            prefix=prefix,
        )
        metric_tensors.update(task_metrics)
        validation_tensors[prefix] = invalid_targets

    for name, logits_key, component in tera_tasks:
        component_logits = logits[logits_key]
        if component_logits.shape[:-1] != base_shape:
            raise ValueError(f"{logits_key} leading dimensions must match Tera labels.")
        prefix = f"opp_intent_{name}"
        task_metrics, invalid_targets = _classification_additive_tensors(
            logits=component_logits,
            targets=tera_labels[..., component],
            selected=tera_masks[..., component].to(torch.bool),
            prefix=prefix,
        )
        metric_tensors.update(task_metrics)
        validation_tensors[prefix] = invalid_targets
    result, invalid = _pack_additive_metrics(metric_tensors, validation_tensors)
    for name, _logits_key, _component in (*slot_tasks, *tera_tasks):
        prefix = f"opp_intent_{name}"
        if invalid[prefix]:
            raise ValueError(f"{prefix} labels contain an out-of-range class index.")
    return result


@torch.no_grad()
def public_consequence_additive_metrics(
    logits: Mapping[str, torch.Tensor],
    move_emitted_targets: torch.Tensor,
    move_emitted_masks: torch.Tensor,
    nonmove_reason_labels: torch.Tensor,
    nonmove_reason_masks: torch.Tensor,
    move_branch_labels: torch.Tensor,
    move_branch_masks: torch.Tensor,
    route_targets: torch.Tensor,
    route_masks: torch.Tensor,
    interaction_targets: torch.Tensor,
    interaction_masks: torch.Tensor,
    interaction_reason_labels: torch.Tensor,
    interaction_reason_masks: torch.Tensor,
    hit_count_labels: torch.Tensor,
    hit_count_masks: torch.Tensor,
    cant_targets: torch.Tensor,
    cant_masks: torch.Tensor,
    field_targets: torch.Tensor,
    field_masks: torch.Tensor,
    persistence_fields: torch.Tensor,
) -> dict[str, float]:
    """Return additive diagnostics for the conditional consequence factorization.

    Every task is evaluated only where its corresponding exact protocol label is
    available. Binary and categorical sums therefore remain meaningful when
    the hierarchy exposes very different amounts of support at each stage.
    Field errors retain the per-labeled-action normalization used by the loss,
    while feature-level confusion counts expose the underlying support.
    """

    required = {
        "public_consequence_move_emitted_logits",
        "public_consequence_nonmove_reason_logits",
        "public_consequence_move_branch_logits",
        "public_consequence_route_logits",
        "public_consequence_interaction_logits",
        "public_consequence_interaction_reason_logits",
        "public_consequence_hit_count_logits",
        "public_consequence_cant_logits",
        "public_consequence_field_logits",
    }
    missing = sorted(required.difference(logits))
    if missing:
        raise ValueError(f"logits is missing key(s): {', '.join(missing)}.")

    if move_emitted_targets.shape != move_emitted_masks.shape:
        raise ValueError(
            "public consequence move-emitted targets/masks must have the same shape."
        )
    if move_emitted_targets.ndim < 1 or move_emitted_targets.shape[-1] != 2:
        raise ValueError(
            "public consequence move-emitted targets must end with two own slots."
        )
    base_shape = tuple(move_emitted_targets.shape)
    target_shape = (*base_shape, NUM_PUBLIC_TARGET_POSITIONS)
    for name, tensor, expected_shape in (
        (
            "move-emitted logits",
            logits["public_consequence_move_emitted_logits"],
            base_shape,
        ),
        ("nonmove-reason labels", nonmove_reason_labels, base_shape),
        ("nonmove-reason masks", nonmove_reason_masks, base_shape),
        (
            "nonmove-reason logits",
            logits["public_consequence_nonmove_reason_logits"],
            (*base_shape, NUM_NONMOVE_REASONS),
        ),
        ("move-branch labels", move_branch_labels, base_shape),
        ("move-branch masks", move_branch_masks, base_shape),
        (
            "move-branch logits",
            logits["public_consequence_move_branch_logits"],
            (*base_shape, NUM_MOVE_BRANCHES),
        ),
        ("route targets", route_targets, target_shape),
        ("route masks", route_masks, target_shape),
        ("route logits", logits["public_consequence_route_logits"], target_shape),
        ("interaction targets", interaction_targets, target_shape),
        ("interaction masks", interaction_masks, target_shape),
        (
            "interaction logits",
            logits["public_consequence_interaction_logits"],
            target_shape,
        ),
        ("interaction-reason labels", interaction_reason_labels, target_shape),
        ("interaction-reason masks", interaction_reason_masks, target_shape),
        (
            "interaction-reason logits",
            logits["public_consequence_interaction_reason_logits"],
            (*target_shape, NUM_TARGET_INTERACTION_REASONS),
        ),
        ("hit-count labels", hit_count_labels, target_shape),
        ("hit-count masks", hit_count_masks, target_shape),
        (
            "hit-count logits",
            logits["public_consequence_hit_count_logits"],
            (*target_shape, MAX_PUBLIC_HIT_COUNT),
        ),
        ("CANT targets", cant_targets, (*base_shape, 2)),
        ("CANT masks", cant_masks, (*base_shape, 2)),
        (
            "CANT logits",
            logits["public_consequence_cant_logits"],
            (*base_shape, 2),
        ),
        ("field masks", field_masks, base_shape),
        (
            "field targets",
            field_targets,
            (*base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM),
        ),
        (
            "persistence fields",
            persistence_fields,
            (*base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM),
        ),
        (
            "field logits",
            logits["public_consequence_field_logits"],
            (*base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM),
        ),
    ):
        _require_shape(f"public consequence {name}", tensor, expected_shape)

    metric_tensors: dict[str, torch.Tensor] = {}
    validation_tensors: dict[str, torch.Tensor] = {}

    binary_tasks = (
        (
            "emitted",
            logits["public_consequence_move_emitted_logits"],
            move_emitted_targets,
            move_emitted_masks,
        ),
        (
            "route",
            logits["public_consequence_route_logits"],
            route_targets,
            route_masks,
        ),
        (
            "interaction",
            logits["public_consequence_interaction_logits"],
            interaction_targets,
            interaction_masks,
        ),
        (
            "cant",
            logits["public_consequence_cant_logits"],
            cant_targets,
            cant_masks,
        ),
    )
    for name, task_logits, task_targets, task_masks in binary_tasks:
        prefix_name = f"aux_public_consequence_{name}"
        task_metrics, invalid_targets = _binary_additive_tensors(
            logits=task_logits,
            targets=task_targets,
            selected=task_masks,
            prefix=prefix_name,
        )
        metric_tensors.update(task_metrics)
        validation_tensors[name] = invalid_targets

    categorical_tasks = (
        (
            "nonmove_reason",
            logits["public_consequence_nonmove_reason_logits"],
            nonmove_reason_labels,
            nonmove_reason_masks,
            _PUBLIC_CONSEQUENCE_NONMOVE_REASON_CLASSES,
        ),
        (
            "move_branch",
            logits["public_consequence_move_branch_logits"],
            move_branch_labels,
            move_branch_masks,
            _PUBLIC_CONSEQUENCE_MOVE_BRANCH_CLASSES,
        ),
        (
            "interaction_reason",
            logits["public_consequence_interaction_reason_logits"],
            interaction_reason_labels,
            interaction_reason_masks,
            _PUBLIC_CONSEQUENCE_INTERACTION_REASON_CLASSES,
        ),
        (
            "hit_count",
            logits["public_consequence_hit_count_logits"],
            hit_count_labels,
            hit_count_masks,
            _PUBLIC_CONSEQUENCE_HIT_COUNT_CLASSES,
        ),
    )
    for name, task_logits, task_labels, task_masks, classes in categorical_tasks:
        prefix_name = f"aux_public_consequence_{name}"
        task_metrics, invalid_targets = _named_classification_additive_tensors(
            logits=task_logits,
            targets=task_labels,
            selected=task_masks,
            prefix=prefix_name,
            classes=classes,
        )
        metric_tensors.update(task_metrics)
        validation_tensors[name] = invalid_targets

    selected_fields = field_masks.reshape(-1).to(torch.bool)
    flat_field_logits = logits["public_consequence_field_logits"].reshape(
        -1, PUBLIC_CONSEQUENCE_FIELD_DIM
    )
    flat_field_targets = field_targets.reshape(-1, PUBLIC_CONSEQUENCE_FIELD_DIM)
    flat_persistence = persistence_fields.reshape(-1, PUBLIC_CONSEQUENCE_FIELD_DIM)
    selected_field_logits = flat_field_logits[selected_fields]
    selected_field_targets = flat_field_targets[selected_fields]
    selected_field_persistence = flat_persistence[selected_fields]
    validation_tensors["field_targets"] = (
        ~torch.isfinite(selected_field_targets)
        | (selected_field_targets < 0.0)
        | (selected_field_targets > 1.0)
    ).any()
    validation_tensors["field_persistence"] = (
        ~torch.isfinite(selected_field_persistence)
        | (selected_field_persistence < 0.0)
        | (selected_field_persistence > 1.0)
    ).any()

    selected_field_probs = selected_field_logits.sigmoid()
    selected_field_nll = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_field_logits,
        selected_field_targets,
        reduction="none",
    )
    selected_field_squared_errors = (
        selected_field_probs - selected_field_targets
    ).square()
    field_nll = selected_field_nll.mean(dim=-1).sum()
    field_brier = selected_field_squared_errors.mean(dim=-1).sum()
    field_predictions = selected_field_logits >= 0.0
    field_positive = selected_field_targets >= 0.5
    field_negative = ~field_positive

    squared_field_errors = (flat_field_logits.sigmoid() - flat_field_targets).square()
    squared_persistence_errors = (flat_persistence - flat_field_targets).square()
    persistence_mse = squared_persistence_errors[selected_fields].mean(dim=-1).sum()
    selected_feature_mask = selected_fields.unsqueeze(-1).expand_as(
        squared_field_errors
    )
    changed_feature_mask = selected_feature_mask & (
        (flat_field_targets - flat_persistence).abs() > 1e-6
    )
    changed_feature_mse = squared_field_errors[changed_feature_mask].sum()
    changed_feature_persistence_mse = squared_persistence_errors[
        changed_feature_mask
    ].sum()

    trick_room_errors = squared_field_errors[:, FIELD_TRICK_ROOM_OFFSET]
    trick_room_persistence_errors = squared_persistence_errors[
        :, FIELD_TRICK_ROOM_OFFSET
    ]
    trick_room_mse = trick_room_errors[selected_fields].sum()
    trick_room_persistence_mse = trick_room_persistence_errors[selected_fields].sum()
    trick_room_value_changed = selected_fields & (
        (
            flat_field_targets[:, FIELD_TRICK_ROOM_OFFSET]
            - flat_persistence[:, FIELD_TRICK_ROOM_OFFSET]
        ).abs()
        > 1e-6
    )
    trick_room_value_changed_mse = trick_room_errors[trick_room_value_changed].sum()
    trick_room_value_changed_persistence_mse = trick_room_persistence_errors[
        trick_room_value_changed
    ].sum()
    trick_room_presence_changed = selected_fields & (
        (flat_field_targets[:, FIELD_TRICK_ROOM_OFFSET] > 1e-6)
        != (flat_persistence[:, FIELD_TRICK_ROOM_OFFSET] > 1e-6)
    )
    trick_room_presence_changed_mse = trick_room_errors[
        trick_room_presence_changed
    ].sum()
    trick_room_presence_changed_persistence_mse = trick_room_persistence_errors[
        trick_room_presence_changed
    ].sum()

    metric_tensors.update(
        {
            "aux_public_consequence_field_count": selected_fields.sum(),
            "aux_public_consequence_field_nll_sum": field_nll,
            "aux_public_consequence_field_brier_sum": field_brier,
            "aux_public_consequence_field_mse_sum": field_brier,
            "aux_public_consequence_field_persistence_mse_sum": persistence_mse,
            "aux_public_consequence_field_feature_count": selected_feature_mask.sum(),
            "aux_public_consequence_field_feature_correct": (
                field_predictions == field_positive
            ).sum(),
            "aux_public_consequence_field_feature_positive": field_positive.sum(),
            "aux_public_consequence_field_feature_negative": field_negative.sum(),
            "aux_public_consequence_field_feature_predicted_positive": (
                field_predictions.sum()
            ),
            "aux_public_consequence_field_feature_true_positive": (
                field_predictions & field_positive
            ).sum(),
            "aux_public_consequence_field_feature_true_negative": (
                (~field_predictions) & field_negative
            ).sum(),
            "aux_public_consequence_field_changed_count": changed_feature_mask.sum(),
            "aux_public_consequence_field_changed_mse_sum": changed_feature_mse,
            "aux_public_consequence_field_changed_persistence_mse_sum": (
                changed_feature_persistence_mse
            ),
            "aux_public_consequence_field_trick_room_count": selected_fields.sum(),
            "aux_public_consequence_field_trick_room_mse_sum": trick_room_mse,
            "aux_public_consequence_field_trick_room_persistence_mse_sum": (
                trick_room_persistence_mse
            ),
            "aux_public_consequence_field_trick_room_value_changed_count": (
                trick_room_value_changed.sum()
            ),
            "aux_public_consequence_field_trick_room_value_changed_mse_sum": (
                trick_room_value_changed_mse
            ),
            "aux_public_consequence_field_trick_room_value_changed_persistence_mse_sum": (
                trick_room_value_changed_persistence_mse
            ),
            "aux_public_consequence_field_trick_room_presence_changed_count": (
                trick_room_presence_changed.sum()
            ),
            "aux_public_consequence_field_trick_room_presence_changed_mse_sum": (
                trick_room_presence_changed_mse
            ),
            "aux_public_consequence_field_trick_room_presence_changed_persistence_mse_sum": (
                trick_room_presence_changed_persistence_mse
            ),
        }
    )

    result, invalid = _pack_additive_metrics(metric_tensors, validation_tensors)
    for name in ("emitted", "route", "interaction", "cant"):
        if invalid[name]:
            raise ValueError(
                f"public consequence {name} targets must be finite and within [0, 1]."
            )
    for name in (
        "nonmove_reason",
        "move_branch",
        "interaction_reason",
        "hit_count",
    ):
        if invalid[name]:
            raise ValueError(
                f"public consequence {name} labels contain an out-of-range class index."
            )
    if invalid["field_targets"]:
        raise ValueError("public consequence field targets must be within [0, 1].")
    if invalid["field_persistence"]:
        raise ValueError("public consequence persistence fields must be within [0, 1].")
    return result
