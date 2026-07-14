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
    TERA_KNOWN_OFFSET,
)
from ..public_consequence import (
    NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES,
    PUBLIC_CONSEQUENCE_FIELD_DIM,
    PUBLIC_CONSEQUENCE_CANT,
    PUBLIC_CONSEQUENCE_EXECUTED,
    PUBLIC_CONSEQUENCE_FAILED,
    PUBLIC_CONSEQUENCE_MISSED,
)


_CLASSIFICATION_TASKS = (
    ("item", "item_logits", "opp_item", ITEM_KNOWN_OFFSET),
    ("ability", "ability_logits", "opp_ability", ABILITY_KNOWN_OFFSET),
    ("tera", "tera_type_logits", None, TERA_KNOWN_OFFSET),
)
_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES = (
    (PUBLIC_CONSEQUENCE_EXECUTED, "executed"),
    (PUBLIC_CONSEQUENCE_FAILED, "failed"),
    (PUBLIC_CONSEQUENCE_MISSED, "missed"),
    (PUBLIC_CONSEQUENCE_CANT, "cant"),
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
    resolution_labels: torch.Tensor,
    resolution_masks: torch.Tensor,
    cant_targets: torch.Tensor,
    cant_masks: torch.Tensor,
    field_targets: torch.Tensor,
    field_masks: torch.Tensor,
    persistence_fields: torch.Tensor,
) -> dict[str, float]:
    """Return additive diagnostics for selected actions' public outcomes.

    The two selected own-action slots form the final axis of ``resolution`` and
    ``field_masks``.  Each action predicts two opponent CANT indicators and the
    same 33-dimensional next-public-field view.  Field error is averaged over
    dimensions before it is accumulated so one labeled action is one unit of
    support, just as it is in the normalized training objective.
    """
    required = {
        "public_consequence_resolution_logits",
        "public_consequence_cant_logits",
        "public_consequence_field_logits",
    }
    missing = sorted(required.difference(logits))
    if missing:
        raise ValueError(f"logits is missing key(s): {', '.join(missing)}.")
    if resolution_labels.shape != resolution_masks.shape:
        raise ValueError(
            "public consequence resolution labels/masks must have the same shape."
        )
    if resolution_labels.ndim < 1 or resolution_labels.shape[-1] != 2:
        raise ValueError(
            "public consequence resolution labels must end with two own action slots."
        )
    base_shape = tuple(resolution_labels.shape)
    if field_masks.shape != resolution_labels.shape:
        raise ValueError(
            "public consequence field masks must match resolution-label shape."
        )
    _require_shape(
        "public consequence resolution logits",
        logits["public_consequence_resolution_logits"],
        (*base_shape, NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES),
    )
    _require_shape("public consequence CANT targets", cant_targets, (*base_shape, 2))
    _require_shape("public consequence CANT masks", cant_masks, (*base_shape, 2))
    _require_shape(
        "public consequence CANT logits",
        logits["public_consequence_cant_logits"],
        (*base_shape, 2),
    )
    expected_field_shape = (*base_shape, PUBLIC_CONSEQUENCE_FIELD_DIM)
    _require_shape(
        "public consequence field targets", field_targets, expected_field_shape
    )
    _require_shape(
        "public consequence persistence fields",
        persistence_fields,
        expected_field_shape,
    )
    _require_shape(
        "public consequence field logits",
        logits["public_consequence_field_logits"],
        expected_field_shape,
    )

    # Keep every additive diagnostic on-device until the end of this function.
    # Calling ``.item()`` for each scalar serialized the CUDA stream dozens of
    # times per PPO minibatch.  Packing the scalars (including validation flags)
    # lets us preserve the Python-float API with one device-to-host transfer.
    metric_tensors: dict[str, torch.Tensor] = {}
    validation_tensors: dict[str, torch.Tensor] = {}

    flat_resolution_logits = logits["public_consequence_resolution_logits"].reshape(
        -1,
        NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES,
    )
    flat_resolution_targets = resolution_labels.reshape(-1)
    flat_resolution_masks = resolution_masks.reshape(-1).to(torch.bool)
    selected_resolution_logits = flat_resolution_logits[flat_resolution_masks]
    selected_resolution_targets = flat_resolution_targets[flat_resolution_masks]
    invalid_resolution_targets = (selected_resolution_targets < 0) | (
        selected_resolution_targets >= NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES
    )
    validation_tensors["resolution"] = invalid_resolution_targets.any()
    # Clamp only to make the packed validation non-synchronizing: invalid labels
    # are still rejected after the single host transfer below.
    safe_resolution_targets = selected_resolution_targets.clamp(
        0,
        NUM_PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES - 1,
    )
    selected_resolution_log_probs = torch.log_softmax(
        selected_resolution_logits,
        dim=-1,
    )
    selected_resolution_probs = selected_resolution_log_probs.exp()
    selected_resolution_target_probs = selected_resolution_probs.gather(
        1,
        safe_resolution_targets.unsqueeze(1),
    ).squeeze(1)
    selected_resolution_nll = -selected_resolution_log_probs.gather(
        1,
        safe_resolution_targets.unsqueeze(1),
    ).squeeze(1)
    selected_resolution_brier = (
        selected_resolution_probs.square().sum(dim=-1)
        - 2.0 * selected_resolution_target_probs
        + 1.0
    )
    selected_resolution_predictions = selected_resolution_logits.argmax(dim=-1)
    resolution_prefix = "aux_public_consequence_resolution"
    metric_tensors.update(
        {
            f"{resolution_prefix}_top1_correct": (
                selected_resolution_predictions == safe_resolution_targets
            ).sum(),
            f"{resolution_prefix}_count": flat_resolution_masks.sum(),
            f"{resolution_prefix}_nll_sum": selected_resolution_nll.sum(),
            f"{resolution_prefix}_brier_sum": selected_resolution_brier.sum(),
        }
    )
    for class_index, class_name in _PUBLIC_CONSEQUENCE_RESOLUTION_CLASSES:
        class_selected = selected_resolution_targets == class_index
        class_prefix = f"aux_public_consequence_resolution_{class_name}"
        metric_tensors.update(
            {
                f"{class_prefix}_correct": (
                    selected_resolution_predictions[class_selected] == class_index
                ).sum(),
                f"{class_prefix}_count": class_selected.sum(),
                f"{class_prefix}_nll_sum": selected_resolution_nll[
                    class_selected
                ].sum(),
            }
        )

    flat_cant_logits = logits["public_consequence_cant_logits"].reshape(-1)
    flat_cant_targets = cant_targets.reshape(-1).to(flat_cant_logits.dtype)
    flat_cant_masks = cant_masks.reshape(-1).to(torch.bool)
    selected_cant_logits = flat_cant_logits[flat_cant_masks]
    selected_cant_targets = flat_cant_targets[flat_cant_masks]
    validation_tensors["cant"] = (
        ~torch.isfinite(selected_cant_targets)
        | (selected_cant_targets < 0.0)
        | (selected_cant_targets > 1.0)
    ).any()
    cant_probs = selected_cant_logits.sigmoid()
    cant_nll = torch.nn.functional.binary_cross_entropy_with_logits(
        selected_cant_logits,
        selected_cant_targets,
        reduction="sum",
    )
    cant_brier = (cant_probs - selected_cant_targets).square().sum()
    cant_predicted_positive = selected_cant_logits >= 0.0
    cant_positive_mask = selected_cant_targets >= 0.5
    cant_negative_mask = ~cant_positive_mask
    metric_tensors.update(
        {
            "aux_public_consequence_cant_correct": (
                cant_predicted_positive == cant_positive_mask
            ).sum(),
            "aux_public_consequence_cant_count": flat_cant_masks.sum(),
            "aux_public_consequence_cant_positive": cant_positive_mask.sum(),
            "aux_public_consequence_cant_negative": cant_negative_mask.sum(),
            "aux_public_consequence_cant_predicted_positive": (
                cant_predicted_positive.sum()
            ),
            "aux_public_consequence_cant_true_positive": (
                cant_predicted_positive & cant_positive_mask
            ).sum(),
            "aux_public_consequence_cant_true_negative": (
                (~cant_predicted_positive) & cant_negative_mask
            ).sum(),
            "aux_public_consequence_cant_nll_sum": cant_nll,
            "aux_public_consequence_cant_brier_sum": cant_brier,
        }
    )

    selected_fields = field_masks.reshape(-1).to(torch.bool)
    flat_field_targets = field_targets.reshape(-1, PUBLIC_CONSEQUENCE_FIELD_DIM)
    flat_persistence = persistence_fields.reshape(-1, PUBLIC_CONSEQUENCE_FIELD_DIM)
    field_probs = (
        logits["public_consequence_field_logits"]
        .sigmoid()
        .reshape(-1, PUBLIC_CONSEQUENCE_FIELD_DIM)
    )
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
    squared_field_errors = (field_probs - flat_field_targets).square()
    squared_persistence_errors = (flat_persistence - flat_field_targets).square()
    field_mse = squared_field_errors[selected_fields].mean(dim=-1).sum()
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
            "aux_public_consequence_field_mse_sum": field_mse,
            "aux_public_consequence_field_persistence_mse_sum": persistence_mse,
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
    if invalid["resolution"]:
        raise ValueError(
            "aux_public_consequence_resolution labels contain an out-of-range "
            "class index."
        )
    if invalid["cant"]:
        raise ValueError("public consequence CANT targets must be within [0, 1].")
    if invalid["field_targets"]:
        raise ValueError("public consequence field targets must be within [0, 1].")
    if invalid["field_persistence"]:
        raise ValueError("public consequence persistence fields must be within [0, 1].")
    return result
