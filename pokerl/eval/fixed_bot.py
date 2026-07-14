"""Evaluate a policy against a MaxBasePower bot with real team preview battles."""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from poke_env.battle import DoubleBattle, Field, Pokemon
from poke_env.data import GenData
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer
from poke_env.teambuilder.teambuilder import Teambuilder

from ..config import EnvConfig, ModelConfig
from ..data import GameData, UNKNOWN_IDX
from ..env.actions import (
    TARGET_ALLY,
    TARGET_NONE,
    TARGET_OPPONENT_1,
    TARGET_OPPONENT_2,
    TARGET_SELF,
    decode_action,
    decode_tera_choice,
)
from ..env.showdown_env import (
    BattleEnv,
    make_battlestream_configuration,
    make_local_account_configuration,
)
from ..team_pool import TeamPool, TeamPoolEntry
from .metrics import outcome_summary
from .replay import save_replay, save_replay_json


def normalize_species_id(species: str) -> str:
    return "".join(ch for ch in str(species).lower() if ch.isalnum())


def build_species_bst_lookup(gen: int = 9) -> dict[str, int]:
    data = GenData.from_gen(gen)
    lookup: dict[str, int] = {}
    for species_id, entry in data.pokedex.items():
        base_stats = entry.get("baseStats", {})
        lookup[species_id] = int(sum(int(value) for value in base_stats.values()))
    return lookup


def packed_team_species(packed_team: str, *, gen: int = 9) -> list[str]:
    data = GenData.from_gen(gen)
    species_names: list[str] = []
    for teambuilder_pokemon in Teambuilder.parse_packed_team(packed_team):
        pokemon = Pokemon(gen=gen, teambuilder=teambuilder_pokemon)
        species_id = pokemon.species
        if species_id is None:
            raise RuntimeError(
                f"Could not infer species from packed team member {teambuilder_pokemon!r}."
            )
        species_names.append(data.pokedex.get(species_id, {}).get("name", species_id))
    return species_names


def packed_team_segments(packed_team: str) -> list[str]:
    return [segment for segment in packed_team.split("]") if segment]


def _species_idx_display_lookup(game_data: GameData) -> dict[int, str]:
    gen_data = GenData.from_gen(game_data.gen)
    lookup = {UNKNOWN_IDX: "<unk>"}
    for species_id, idx in game_data.species_to_idx.items():
        if idx == UNKNOWN_IDX:
            continue
        lookup[idx] = gen_data.pokedex.get(species_id, {}).get("name", species_id)
    return lookup


def _species_names_from_obs(
    obs_species: np.ndarray, idx_lookup: dict[int, str]
) -> list[str]:
    return [
        idx_lookup.get(int(species_idx), f"<species:{int(species_idx)}>")
        for species_idx in obs_species
        if int(species_idx) != UNKNOWN_IDX
    ]


def _counter_rows(counter: Counter[str], total: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "label": label,
                "count": int(count),
                "rate": _safe_rate(int(count), total),
            }
        )
    return rows


@dataclass(frozen=True)
class FixedTeamSpec:
    title: str
    file: str
    packed_team: str
    max_bst_packed_team: str
    species: tuple[str, ...]
    species_bst: tuple[int, ...]
    max_bst_pick_indices: tuple[int, ...]
    max_bst_pick_species: tuple[str, ...]
    max_bst_pick_bst: tuple[int, ...]
    max_bst_pick_total: int

    @property
    def team_total_bst(self) -> int:
        return int(sum(self.species_bst))

    def to_summary(self) -> dict[str, object]:
        return {
            "title": self.title,
            "file": self.file,
            "species": list(self.species),
            "species_bst": list(self.species_bst),
            "team_total_bst": self.team_total_bst,
            "pick_size": len(self.max_bst_pick_species),
            "max_bst_pick_indices": list(self.max_bst_pick_indices),
            "max_bst_pick_species": list(self.max_bst_pick_species),
            "max_bst_pick_bst": list(self.max_bst_pick_bst),
            "max_bst_pick_total": self.max_bst_pick_total,
        }


def fixed_team_spec_from_entry(
    entry: TeamPoolEntry,
    *,
    species_bst_lookup: dict[str, int],
    pick_size: int,
) -> FixedTeamSpec:
    species = tuple(packed_team_species(entry.packed_team))
    packed_segments = packed_team_segments(entry.packed_team)
    species_bst = []
    for name in species:
        species_id = normalize_species_id(name)
        if species_id not in species_bst_lookup:
            raise RuntimeError(
                f"No BST entry found for species {name!r} ({species_id!r})."
            )
        species_bst.append(species_bst_lookup[species_id])

    ranked = sorted(
        enumerate(species_bst),
        key=lambda item: (-item[1], item[0]),
    )[:pick_size]
    pick_indices = tuple(sorted(index for index, _ in ranked))
    pick_species = tuple(species[index] for index in pick_indices)
    pick_bst = tuple(species_bst[index] for index in pick_indices)
    max_bst_packed_team = "]".join(packed_segments[index] for index in pick_indices)
    return FixedTeamSpec(
        title=entry.title,
        file=entry.file,
        packed_team=entry.packed_team,
        max_bst_packed_team=max_bst_packed_team,
        species=species,
        species_bst=tuple(species_bst),
        max_bst_pick_indices=pick_indices,
        max_bst_pick_species=pick_species,
        max_bst_pick_bst=pick_bst,
        max_bst_pick_total=int(sum(pick_bst)),
    )


def select_max_bst_team_spec(
    pool: TeamPool,
    *,
    species_bst_lookup: dict[str, int],
    pick_size: int,
) -> FixedTeamSpec:
    candidates = [
        fixed_team_spec_from_entry(
            entry,
            species_bst_lookup=species_bst_lookup,
            pick_size=pick_size,
        )
        for entry in pool.entries
    ]
    candidates.sort(
        key=lambda candidate: (
            -candidate.max_bst_pick_total,
            -candidate.team_total_bst,
            candidate.title,
        )
    )
    return candidates[0]


def select_team_spec_by_title(
    pool: TeamPool,
    *,
    title: str,
    species_bst_lookup: dict[str, int],
    pick_size: int,
) -> FixedTeamSpec:
    normalized_title = " ".join(title.split()).casefold()
    matches = [
        entry
        for entry in pool.entries
        if " ".join(entry.title.split()).casefold() == normalized_title
    ]
    if not matches:
        raise RuntimeError(
            f"Could not find a team titled {title!r} in the requested pool."
        )
    if len(matches) > 1:
        raise RuntimeError(f"Team title {title!r} was ambiguous in the requested pool.")
    return fixed_team_spec_from_entry(
        matches[0],
        species_bst_lookup=species_bst_lookup,
        pick_size=pick_size,
    )


def plan_opponent_team_specs(
    pool: TeamPool,
    *,
    model_team_spec: FixedTeamSpec,
    species_bst_lookup: dict[str, int],
    pick_size: int,
    num_games: int,
    seed: int,
) -> list[FixedTeamSpec]:
    if num_games <= 0:
        return []

    candidate_entries = [
        entry for entry in pool.entries if entry.file != model_team_spec.file
    ]
    if not candidate_entries:
        candidate_entries = list(pool.entries)
    if not candidate_entries:
        raise RuntimeError(
            "Opponent team planning requires at least one team in the pool."
        )

    rng = random.Random(seed)
    planned_entries: list[TeamPoolEntry] = []
    while len(planned_entries) < num_games:
        batch = list(candidate_entries)
        rng.shuffle(batch)
        planned_entries.extend(batch)

    return [
        fixed_team_spec_from_entry(
            entry,
            species_bst_lookup=species_bst_lookup,
            pick_size=pick_size,
        )
        for entry in planned_entries[:num_games]
    ]


class PlannedMatchups:
    def __init__(
        self, model_team_spec: FixedTeamSpec, opponent_specs: list[FixedTeamSpec]
    ):
        if not opponent_specs:
            raise ValueError("PlannedMatchups requires at least one opponent team.")
        self.model_team_spec = model_team_spec
        self._prepared_specs = list(opponent_specs) + [opponent_specs[-1]]
        self._prepared_index = 0
        self._active_specs: list[FixedTeamSpec] = []

    def team_pair_provider(self) -> tuple[str, str]:
        if self._prepared_index >= len(self._prepared_specs):
            raise RuntimeError(
                "PlannedMatchups exhausted unexpectedly during evaluation."
            )
        opponent_spec = self._prepared_specs[self._prepared_index]
        self._prepared_index += 1
        self._active_specs.append(opponent_spec)
        return self.model_team_spec.packed_team, opponent_spec.packed_team

    def consume_active_opponent(self) -> FixedTeamSpec:
        if not self._active_specs:
            raise RuntimeError(
                "No active opponent was prepared for the current evaluation battle."
            )
        return self._active_specs.pop(0)


class _DeterministicPreviewMixin:
    def __init__(self, *, species_bst_lookup: dict[str, int], pick_size: int, **kwargs):
        super().__init__(**kwargs)
        self._species_bst_lookup = species_bst_lookup
        self._pick_size = pick_size

    def teampreview(self, battle) -> str:
        members = list(battle.team.values())
        ranked = []
        for index, pokemon in enumerate(members):
            species_id = normalize_species_id(getattr(pokemon, "species", "") or "")
            bst = self._species_bst_lookup.get(species_id)
            if bst is None:
                raise RuntimeError(
                    f"No BST entry found for teampreview species {pokemon.species!r} ({species_id!r})."
                )
            ranked.append((index, bst))

        picks = sorted(
            index
            for index, _ in sorted(ranked, key=lambda item: (-item[1], item[0]))[
                : self._pick_size
            ]
        )
        for pick in picks:
            members[pick]._selected_in_teampreview = True
        return "/team " + "".join(str(pick + 1) for pick in picks)


class DeterministicPreviewRandomPlayer(_DeterministicPreviewMixin, RandomPlayer):
    """Random battle policy with the same deterministic preview rule as the other bots."""


class DeterministicPreviewMaxBasePowerPlayer(
    _DeterministicPreviewMixin,
    MaxBasePowerPlayer,
):
    """Max-base-power battle policy with deterministic max-BST preview."""


class DeterministicPreviewSimpleHeuristicsPlayer(
    _DeterministicPreviewMixin,
    SimpleHeuristicsPlayer,
):
    """Simple heuristic battle policy with deterministic max-BST preview."""


_OPPONENT_PLAYER_TYPES = {
    "random": DeterministicPreviewRandomPlayer,
    "max-base-power": DeterministicPreviewMaxBasePowerPlayer,
    "simple-heuristics": DeterministicPreviewSimpleHeuristicsPlayer,
}


class ModelPolicyController:
    def __init__(self, agent, *, device: torch.device, action_mode: str):
        self.agent = agent
        self.device = device
        self.action_mode = action_mode
        self.hidden = agent.initial_hidden(1)

    def reset_battle(self) -> None:
        self.hidden = self.agent.initial_hidden(1)

    def choose_team_preview(self, obs: dict[str, np.ndarray]) -> dict[str, object]:
        obs_t = {
            key: torch.from_numpy(value).unsqueeze(0).to(self.device)
            for key, value in obs.items()
        }

        with torch.inference_mode():
            preview_out = self.agent.team_preview(
                obs_t,
                greedy=self.action_mode == "argmax",
            )

        return {
            "picks": preview_out["picks"].cpu().numpy()[0],
            "entropy": float(preview_out["entropy"].item()),
            "log_prob": float(preview_out["log_prob"].item()),
            "opponent_bring_logits": preview_out["opponent_bring_logits"]
            .cpu()
            .numpy()[0],
            "opponent_lead_pair_logits": preview_out["opponent_lead_pair_logits"]
            .cpu()
            .numpy()[0],
        }

    def choose_action(
        self, obs: dict[str, np.ndarray], masks: dict[str, np.ndarray]
    ) -> dict[str, object]:
        obs_t = {
            key: torch.from_numpy(value).unsqueeze(0).to(self.device)
            for key, value in obs.items()
        }
        masks_t = {
            key: torch.from_numpy(value)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.bool)
            for key, value in masks.items()
        }

        with torch.inference_mode():
            action_out = self.agent(
                obs_t,
                self.hidden,
                action_masks=masks_t,
                greedy=self.action_mode == "argmax",
                include_value=False,
            )
            new_hidden = action_out["hidden"]
            self.hidden = (
                None
                if new_hidden is None
                else tuple(state.detach() for state in new_hidden)
            )
            entropy = action_out["entropy"].item()
            log_prob = action_out["log_prob"].item()
            actions = action_out["actions"].cpu().numpy()[0]

        return {
            "actions": actions,
            "entropy": float(entropy),
            "log_prob": float(log_prob),
        }


def _max_legal_damaging_base_power(battle: DoubleBattle, slot: int) -> int:
    if slot >= len(battle.available_moves):
        return 0
    return max(
        (int(move.base_power or 0) for move in battle.available_moves[slot]), default=0
    )


_MOVE_OPPORTUNITY_GROUPS = {
    "protect_like": frozenset(
        {
            "banefulbunker",
            "burningbulwark",
            "detect",
            "endure",
            "kingsshield",
            "maxguard",
            "obstruct",
            "protect",
            "silktrap",
            "spikyshield",
        }
    ),
    "trick_room": frozenset({"trickroom"}),
    "redirection": frozenset({"followme", "ragepowder"}),
    "fake_out": frozenset({"fakeout"}),
    "tailwind": frozenset({"tailwind"}),
}

_CONTEXTUAL_MOVE_GROUPS = (
    "protect_like_fresh",
    "protect_like_repeat",
    "trick_room_setup",
    "trick_room_cancel",
    "fake_out_first_turn",
    "fake_out_late",
)


def _active_pokemon_for_slot(battle: DoubleBattle, slot: int) -> Pokemon | None:
    active_pokemon = getattr(battle, "active_pokemon", ())
    if slot >= len(active_pokemon):
        return None
    return active_pokemon[slot]


def _trick_room_is_active(battle: DoubleBattle) -> bool:
    return Field.TRICK_ROOM in getattr(battle, "fields", {})


def _contextual_move_group(
    battle: DoubleBattle,
    slot: int,
    move_id: str,
) -> str | None:
    """Classify mechanics whose useful opportunity depends on live state."""
    active_pokemon = _active_pokemon_for_slot(battle, slot)
    if move_id in _MOVE_OPPORTUNITY_GROUPS["protect_like"]:
        protect_counter = int(getattr(active_pokemon, "protect_counter", 0))
        return "protect_like_repeat" if protect_counter > 0 else "protect_like_fresh"
    if move_id == "trickroom":
        return "trick_room_cancel" if _trick_room_is_active(battle) else "trick_room_setup"
    if move_id == "fakeout":
        return (
            "fake_out_first_turn"
            if bool(getattr(active_pokemon, "first_turn", False))
            else "fake_out_late"
        )
    return None


def _record_action_opportunities(
    battle: DoubleBattle,
    slot: int,
) -> Counter[str]:
    """Count strategic actions that were legal for this slot decision."""
    stats: Counter[str] = Counter()
    moves = battle.available_moves[slot] if slot < len(battle.available_moves) else []
    move_ids = {str(getattr(move, "id", "")) for move in moves}
    for label, diagnostic_moves in _MOVE_OPPORTUNITY_GROUPS.items():
        if move_ids & diagnostic_moves:
            stats[f"{label}_opportunities"] += 1

    contextual_groups = {
        contextual_group
        for move_id in move_ids
        if (contextual_group := _contextual_move_group(battle, slot, move_id))
        is not None
    }
    for contextual_group in contextual_groups:
        stats[f"{contextual_group}_opportunities"] += 1
    if move_ids & _MOVE_OPPORTUNITY_GROUPS["protect_like"]:
        active_pokemon = _active_pokemon_for_slot(battle, slot)
        prior_successes = int(getattr(active_pokemon, "protect_counter", 0))
        stats[f"protect_like_prior_{min(prior_successes, 4)}_opportunities"] += 1

    switches = (
        battle.available_switches[slot] if slot < len(battle.available_switches) else []
    )
    if switches:
        stats["switch_opportunities"] += 1
    return stats


def _classify_slot_action(
    battle: DoubleBattle,
    slot: int,
    action_idx: int,
    tera_choice: int,
) -> Counter[str]:
    kind, move_idx, target_idx = decode_action(int(action_idx))
    tera_slot_1, tera_slot_2 = decode_tera_choice(int(tera_choice))
    tera = tera_slot_1 if slot == 0 else tera_slot_2
    stats = _record_action_opportunities(battle, slot)
    stats["slot_decisions"] += 1

    if kind == "pass":
        stats["pass_actions"] += 1
        if _max_legal_damaging_base_power(battle, slot) > 0:
            stats["pass_while_damage_available_actions"] += 1
        return stats

    if kind == "switch":
        stats["switch_actions"] += 1
        if _max_legal_damaging_base_power(battle, slot) > 0:
            stats["switch_while_damage_available_actions"] += 1
        return stats

    if slot >= len(battle.available_moves) or move_idx >= len(
        battle.available_moves[slot]
    ):
        raise RuntimeError(
            f"Action {action_idx} decoded to move index {move_idx}, but slot {slot} only has "
            f"{len(battle.available_moves[slot]) if slot < len(battle.available_moves) else 0} legal moves."
        )

    move = battle.available_moves[slot][move_idx]
    base_power = int(move.base_power or 0)
    best_base_power = _max_legal_damaging_base_power(battle, slot)
    move_target = getattr(getattr(move, "target", None), "name", "")

    stats["move_actions"] += 1
    if tera:
        stats["tera_move_actions"] += 1
    else:
        stats["regular_move_actions"] += 1

    if base_power > 0:
        stats["damaging_move_actions"] += 1
    else:
        stats["status_move_actions"] += 1

    stats["chosen_base_power_total"] += base_power
    stats["best_legal_base_power_total"] += best_base_power
    if best_base_power > 0:
        stats["power_comparable_actions"] += 1

    if target_idx == TARGET_ALLY:
        stats["ally_target_actions"] += 1
    elif target_idx == TARGET_SELF:
        stats["self_target_actions"] += 1
    elif target_idx in (TARGET_OPPONENT_1, TARGET_OPPONENT_2):
        stats["opponent_target_actions"] += 1
    elif target_idx == TARGET_NONE:
        stats["no_explicit_target_actions"] += 1

    if base_power > 0 and target_idx == TARGET_ALLY:
        stats["damaging_ally_target_actions"] += 1
        stats["own_team_hitting_move_actions"] += 1

    if base_power > 0 and target_idx == TARGET_SELF:
        stats["damaging_self_target_actions"] += 1
        stats["own_team_hitting_move_actions"] += 1

    if base_power > 0 and move_target == "ALL_ADJACENT":
        stats["ally_spread_damage_actions"] += 1
        stats["own_team_hitting_move_actions"] += 1

    if base_power > 0 and target_idx in (TARGET_OPPONENT_1, TARGET_OPPONENT_2):
        opponent_slot = 0 if target_idx == TARGET_OPPONENT_1 else 1
        opponents = list(battle.opponent_active_pokemon)
        target = opponents[opponent_slot] if opponent_slot < len(opponents) else None
        if target is not None:
            multiplier = float(target.damage_multiplier(move))
            stats["typed_opponent_damage_actions"] += 1
            if multiplier == 0.0:
                stats["immune_target_damage_actions"] += 1
            elif multiplier < 1.0:
                stats["resisted_target_damage_actions"] += 1
            elif multiplier > 1.0:
                stats["super_effective_target_damage_actions"] += 1
            else:
                stats["neutral_target_damage_actions"] += 1

    if getattr(move, "id", "") == "protect":
        stats["protect_actions"] += 1
    move_id = str(getattr(move, "id", ""))
    for label, diagnostic_moves in _MOVE_OPPORTUNITY_GROUPS.items():
        if move_id in diagnostic_moves:
            stats[f"{label}_actions"] += 1
    contextual_group = _contextual_move_group(battle, slot, move_id)
    if contextual_group is not None:
        stats[f"{contextual_group}_actions"] += 1
    if move_id in _MOVE_OPPORTUNITY_GROUPS["protect_like"]:
        active_pokemon = _active_pokemon_for_slot(battle, slot)
        prior_successes = int(getattr(active_pokemon, "protect_counter", 0))
        stats[f"protect_like_prior_{min(prior_successes, 4)}_actions"] += 1

    if base_power > 0 and best_base_power > 0:
        if base_power == best_base_power:
            stats["max_legal_base_power_choices"] += 1
        else:
            stats["non_max_legal_base_power_choices"] += 1
    elif base_power == 0 and best_base_power > 0:
        stats["non_damaging_while_damage_available_actions"] += 1

    return stats


def _describe_slot_action(
    battle: DoubleBattle,
    slot: int,
    action_idx: int,
    tera_choice: int,
) -> dict[str, object]:
    kind, move_idx, target_idx = decode_action(int(action_idx))
    tera_slot_1, tera_slot_2 = decode_tera_choice(int(tera_choice))
    tera = tera_slot_1 if slot == 0 else tera_slot_2
    description: dict[str, object] = {
        "slot": slot,
        "action_idx": int(action_idx),
        "kind": kind,
        "tera": bool(tera),
        "target": int(target_idx),
    }
    if kind != "move":
        return description

    move = battle.available_moves[slot][move_idx]
    description.update(
        {
            "move": move.id,
            "base_power": int(move.base_power or 0),
            "category": getattr(getattr(move, "category", None), "name", None),
            "move_target": getattr(getattr(move, "target", None), "name", None),
            "max_legal_base_power": _max_legal_damaging_base_power(battle, slot),
        }
    )
    return description


def _safe_rate(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _summarize_action_stats(action_stats: Counter[str]) -> dict[str, object]:
    slot_decisions = int(action_stats.get("slot_decisions", 0))
    move_actions = int(action_stats.get("move_actions", 0))
    power_comparable = int(action_stats.get("power_comparable_actions", 0))
    chosen_power_total = int(action_stats.get("chosen_base_power_total", 0))
    best_power_total = int(action_stats.get("best_legal_base_power_total", 0))
    typed_opponent_damage = int(action_stats.get("typed_opponent_damage_actions", 0))
    joint_decisions = int(action_stats.get("joint_decisions", 0))

    counts = {
        key: int(value)
        for key, value in action_stats.items()
        if key not in {"chosen_base_power_total", "best_legal_base_power_total"}
    }
    rates = {
        "move_rate": _safe_rate(counts.get("move_actions", 0), slot_decisions),
        "tera_move_rate": _safe_rate(
            counts.get("tera_move_actions", 0), slot_decisions
        ),
        "switch_rate": _safe_rate(counts.get("switch_actions", 0), slot_decisions),
        "pass_rate": _safe_rate(counts.get("pass_actions", 0), slot_decisions),
        "damaging_move_rate": _safe_rate(
            counts.get("damaging_move_actions", 0), slot_decisions
        ),
        "status_move_rate": _safe_rate(
            counts.get("status_move_actions", 0), slot_decisions
        ),
        "own_team_hitting_move_rate": _safe_rate(
            counts.get("own_team_hitting_move_actions", 0), slot_decisions
        ),
        "non_damaging_while_damage_available_rate": _safe_rate(
            counts.get("non_damaging_while_damage_available_actions", 0),
            slot_decisions,
        ),
        "switch_while_damage_available_rate": _safe_rate(
            counts.get("switch_while_damage_available_actions", 0),
            slot_decisions,
        ),
        "pass_while_damage_available_rate": _safe_rate(
            counts.get("pass_while_damage_available_actions", 0),
            slot_decisions,
        ),
        "max_legal_base_power_choice_rate": _safe_rate(
            counts.get("max_legal_base_power_choices", 0),
            power_comparable,
        ),
        "non_max_legal_base_power_choice_rate": _safe_rate(
            counts.get("non_max_legal_base_power_choices", 0),
            power_comparable,
        ),
        "protect_rate": _safe_rate(counts.get("protect_actions", 0), slot_decisions),
        **{
            f"{label}_choice_rate_when_available": _safe_rate(
                counts.get(f"{label}_actions", 0),
                counts.get(f"{label}_opportunities", 0),
            )
            for label in _MOVE_OPPORTUNITY_GROUPS
        },
        **{
            f"{label}_choice_rate_when_available": _safe_rate(
                counts.get(f"{label}_actions", 0),
                counts.get(f"{label}_opportunities", 0),
            )
            for label in _CONTEXTUAL_MOVE_GROUPS
        },
        "switch_choice_rate_when_available": _safe_rate(
            counts.get("switch_actions", 0),
            counts.get("switch_opportunities", 0),
        ),
        "immune_target_rate": _safe_rate(
            counts.get("immune_target_damage_actions", 0),
            typed_opponent_damage,
        ),
        "resisted_target_rate": _safe_rate(
            counts.get("resisted_target_damage_actions", 0),
            typed_opponent_damage,
        ),
        "neutral_target_rate": _safe_rate(
            counts.get("neutral_target_damage_actions", 0),
            typed_opponent_damage,
        ),
        "super_effective_target_rate": _safe_rate(
            counts.get("super_effective_target_damage_actions", 0),
            typed_opponent_damage,
        ),
        "focus_fire_turn_rate": _safe_rate(
            counts.get("focus_fire_turns", 0),
            joint_decisions,
        ),
        "double_protect_turn_rate": _safe_rate(
            counts.get("double_protect_turns", 0),
            joint_decisions,
        ),
        "double_protect_like_turn_rate": _safe_rate(
            counts.get("double_protect_like_turns", 0),
            joint_decisions,
        ),
        "protect_like_success_rate": _safe_rate(
            counts.get("protect_like_successes", 0),
            counts.get("protect_like_attempts", 0),
        ),
        "protect_like_fresh_success_rate": _safe_rate(
            counts.get("protect_like_fresh_successes", 0),
            counts.get("protect_like_fresh_attempts", 0),
        ),
        "protect_like_repeat_success_rate": _safe_rate(
            counts.get("protect_like_repeat_successes", 0),
            counts.get("protect_like_repeat_attempts", 0),
        ),
        **{
            f"{label}_battle_choice_rate_when_available": _safe_rate(
                counts.get(f"{label}_action_battles", 0),
                counts.get(f"{label}_opportunity_battles", 0),
            )
            for label in _CONTEXTUAL_MOVE_GROUPS
        },
    }

    power = {
        "avg_chosen_base_power_per_move_action": float(
            chosen_power_total / move_actions
        )
        if move_actions
        else 0.0,
        "avg_best_legal_base_power_per_slot_decision": float(
            best_power_total / slot_decisions
        )
        if slot_decisions
        else 0.0,
        "chosen_vs_best_base_power_ratio": float(chosen_power_total / best_power_total)
        if best_power_total
        else 0.0,
    }
    return {
        "counts": counts,
        "rates": rates,
        "power": power,
        "protect_chain": [
            {
                "prior_successes": prior_successes if prior_successes < 4 else "4+",
                "opportunities": counts.get(
                    f"protect_like_prior_{prior_successes}_opportunities", 0
                ),
                "actions": counts.get(
                    f"protect_like_prior_{prior_successes}_actions", 0
                ),
                "successes": counts.get(
                    f"protect_like_prior_{prior_successes}_successes", 0
                ),
                "choice_rate_when_available": _safe_rate(
                    counts.get(f"protect_like_prior_{prior_successes}_actions", 0),
                    counts.get(
                        f"protect_like_prior_{prior_successes}_opportunities", 0
                    ),
                ),
                "success_rate_when_chosen": _safe_rate(
                    counts.get(
                        f"protect_like_prior_{prior_successes}_successes", 0
                    ),
                    counts.get(f"protect_like_prior_{prior_successes}_attempts", 0),
                ),
            }
            for prior_successes in range(5)
        ],
    }


def _preview_pick_key(species: list[str]) -> str:
    return " / ".join(species)


def _summarize_preview_results(
    preview_results: list[dict[str, object]],
) -> dict[str, object]:
    ordered_picks = Counter()
    unordered_picks = Counter()
    species_counts = Counter()
    lead_counts = Counter()
    entropies: list[float] = []
    preview_modes = {
        str(result.get("preview_mode", "learned")) for result in preview_results
    }

    for result in preview_results:
        picked_species = [str(species) for species in result["picked_species_ordered"]]
        ordered_picks[_preview_pick_key(picked_species)] += 1
        unordered_picks[_preview_pick_key(sorted(picked_species))] += 1
        for species in picked_species:
            species_counts[species] += 1
        for species in picked_species[:2]:
            lead_counts[species] += 1
        entropies.append(float(result["entropy"]))

    total_games = len(preview_results)
    total_slots = sum(
        len(result["picked_species_ordered"]) for result in preview_results
    )
    total_leads = sum(
        min(2, len(result["picked_species_ordered"])) for result in preview_results
    )
    modal_unordered_count = max(unordered_picks.values(), default=0)
    lead_pairs = Counter(
        _preview_pick_key(
            [str(species) for species in result["picked_species_ordered"][:2]]
        )
        for result in preview_results
    )
    modal_lead_count = max(lead_pairs.values(), default=0)
    preview_mode = next(iter(preview_modes)) if len(preview_modes) == 1 else "mixed"
    return {
        "games": total_games,
        "preview_mode": preview_mode,
        "distinct_ordered_picks": len(ordered_picks),
        "distinct_unordered_picks": len(unordered_picks),
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_std": float(np.std(entropies)) if entropies else 0.0,
        "ordered_pick_sets": _counter_rows(ordered_picks, total_games),
        "unordered_pick_sets": _counter_rows(unordered_picks, total_games),
        "species_pick_counts": _counter_rows(species_counts, total_slots),
        "lead_species_counts": _counter_rows(lead_counts, total_leads),
        "opponent_conditioning": {
            "bring_change_from_modal_rate": _safe_rate(
                total_games - modal_unordered_count,
                total_games,
            ),
            "ordered_lead_change_from_modal_rate": _safe_rate(
                total_games - modal_lead_count,
                total_games,
            ),
            "note": (
                "For learned greedy preview with a fixed own roster, changes are caused by "
                "the opposing roster. Learned sampled and random-preview changes also include "
                "policy randomness; max-BST preview is opponent-independent by construction."
            ),
        },
    }


def _classify_joint_action(
    action_descriptions: list[dict[str, object]],
) -> Counter[str]:
    stats: Counter[str] = Counter(joint_decisions=1)
    if len(action_descriptions) != 2:
        return stats

    damaging_targets = [
        int(action["target"])
        for action in action_descriptions
        if action.get("kind") == "move"
        and int(action.get("base_power", 0)) > 0
        and int(action["target"]) in (TARGET_OPPONENT_1, TARGET_OPPONENT_2)
    ]
    if len(damaging_targets) == 2 and len(set(damaging_targets)) == 1:
        stats["focus_fire_turns"] += 1

    if all(
        action.get("kind") == "move" and action.get("move") == "protect"
        for action in action_descriptions
    ):
        stats["double_protect_turns"] += 1
    if all(
        action.get("kind") == "move"
        and action.get("move") in _MOVE_OPPORTUNITY_GROUPS["protect_like"]
        for action in action_descriptions
    ):
        stats["double_protect_like_turns"] += 1
    return stats


def _classify_battle_mechanics(
    battle_action_stats: Counter[str],
) -> Counter[str]:
    """Collapse opportunity/action counts to one observation per battle."""
    stats: Counter[str] = Counter(evaluated_battles=1)
    for label in _CONTEXTUAL_MOVE_GROUPS:
        if battle_action_stats.get(f"{label}_opportunities", 0) > 0:
            stats[f"{label}_opportunity_battles"] += 1
        if battle_action_stats.get(f"{label}_actions", 0) > 0:
            stats[f"{label}_action_battles"] += 1
    return stats


def _protect_attempt_contexts(
    battle: DoubleBattle,
    action_descriptions: list[dict[str, object]],
) -> list[tuple[Pokemon, int]]:
    """Capture selected Protect-family users before the simulator mutates state."""
    contexts: list[tuple[Pokemon, int]] = []
    for action in action_descriptions:
        if (
            action.get("kind") != "move"
            or action.get("move") not in _MOVE_OPPORTUNITY_GROUPS["protect_like"]
        ):
            continue
        pokemon = _active_pokemon_for_slot(battle, int(action["slot"]))
        if pokemon is not None:
            contexts.append((pokemon, int(pokemon.protect_counter)))
    return contexts


def _classify_protect_outcomes(
    contexts: list[tuple[Pokemon, int]],
) -> Counter[str]:
    """Use poke-env's post-turn Protect counter to score attempt outcomes."""
    stats: Counter[str] = Counter()
    for pokemon, prior_successes in contexts:
        bucket = min(prior_successes, 4)
        repeated = prior_successes > 0
        succeeded = int(pokemon.protect_counter) > prior_successes
        stats["protect_like_attempts"] += 1
        stats[f"protect_like_prior_{bucket}_attempts"] += 1
        stats[
            "protect_like_repeat_attempts" if repeated else "protect_like_fresh_attempts"
        ] += 1
        if succeeded:
            stats["protect_like_successes"] += 1
            stats[f"protect_like_prior_{bucket}_successes"] += 1
            stats[
                "protect_like_repeat_successes"
                if repeated
                else "protect_like_fresh_successes"
            ] += 1
    return stats


def _summarize_opponent_preview_predictions(
    preview_results: list[dict[str, object]],
) -> dict[str, object]:
    bring_overlaps: list[float] = []
    lead_overlaps: list[float] = []
    bring_exact = 0
    ordered_lead_exact = 0

    for result in preview_results:
        predicted_bring = {
            int(index) for index in result["predicted_opponent_bring_indices"]
        }
        actual_bring = {int(index) for index in result["actual_opponent_bring_indices"]}
        predicted_lead = tuple(
            int(index) for index in result["predicted_opponent_lead_indices"]
        )
        actual_lead = tuple(
            int(index) for index in result["actual_opponent_lead_indices"]
        )
        bring_denominator = max(len(actual_bring), 1)
        lead_denominator = max(len(actual_lead), 1)
        bring_overlaps.append(len(predicted_bring & actual_bring) / bring_denominator)
        lead_overlaps.append(
            len(set(predicted_lead) & set(actual_lead)) / lead_denominator
        )
        bring_exact += int(predicted_bring == actual_bring)
        ordered_lead_exact += int(predicted_lead == actual_lead)

    games = len(preview_results)
    return {
        "games": games,
        "bring_overlap_rate_mean": (
            float(np.mean(bring_overlaps)) if bring_overlaps else 0.0
        ),
        "bring_exact_rate": _safe_rate(bring_exact, games),
        "unordered_lead_overlap_rate_mean": (
            float(np.mean(lead_overlaps)) if lead_overlaps else 0.0
        ),
        "ordered_lead_exact_rate": _safe_rate(ordered_lead_exact, games),
        "six_slot_chance_baselines": {
            "bring_overlap_rate_mean": 2.0 / 3.0,
            "bring_exact_rate": 1.0 / 15.0,
            "ordered_lead_exact_rate": 1.0 / 30.0,
        },
        "target_policy": "deterministic max-BST preview used by the fixed opponent",
    }


def _summarize_battle_behavior(
    battle_results: list[dict[str, object]],
) -> dict[str, object]:
    tera_used_games = [result for result in battle_results if result.get("tera_used")]
    tera_unused_games = [
        result for result in battle_results if not result.get("tera_used")
    ]
    tera_turns = [
        int(result["first_tera_turn"])
        for result in tera_used_games
        if result.get("first_tera_turn")
    ]
    tera_turn_counts = Counter(tera_turns)
    tera_slot_counts = Counter(
        int(result["first_tera_slot"])
        for result in tera_used_games
        if result.get("first_tera_slot") is not None
    )
    turn_one_tera_games = sum(turn == 1 for turn in tera_turns)
    switch_counts = [int(result.get("switch_actions", 0)) for result in battle_results]

    return {
        "games_with_tera": len(tera_used_games),
        "games_without_tera": len(tera_unused_games),
        "tera_game_rate": _safe_rate(len(tera_used_games), len(battle_results)),
        "turn_one_tera_rate_all_games": _safe_rate(
            turn_one_tera_games, len(battle_results)
        ),
        "turn_one_tera_rate_when_used": _safe_rate(
            turn_one_tera_games, len(tera_used_games)
        ),
        "win_rate_when_tera_used": _safe_rate(
            sum(1 for result in tera_used_games if result.get("won")),
            len(tera_used_games),
        ),
        "win_rate_when_tera_unused": _safe_rate(
            sum(1 for result in tera_unused_games if result.get("won")),
            len(tera_unused_games),
        ),
        "avg_first_tera_turn": float(np.mean(tera_turns)) if tera_turns else 0.0,
        "first_tera_turn_distribution": [
            {
                "turn": int(turn),
                "count": int(count),
                "rate": _safe_rate(int(count), len(tera_used_games)),
            }
            for turn, count in sorted(tera_turn_counts.items())
        ],
        "first_tera_slot_distribution": [
            {
                "slot": int(slot),
                "count": int(count),
                "rate": _safe_rate(int(count), len(tera_used_games)),
            }
            for slot, count in sorted(tera_slot_counts.items())
        ],
        "avg_switch_actions_per_game": float(np.mean(switch_counts))
        if switch_counts
        else 0.0,
    }


def _battle_tag_slug(battle: DoubleBattle) -> str:
    return battle.battle_tag.replace("[", "").replace("]", "").replace(" ", "_")


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def evaluate_policy_vs_fixed_bot(
    *,
    agent,
    device: torch.device,
    env_config: EnvConfig,
    model_team_spec: FixedTeamSpec,
    opponent_team_pool: TeamPool,
    species_bst_lookup: dict[str, int],
    num_games: int,
    action_mode: str,
    seed: int,
    max_turns: int,
    opponent_kind: str = "max-base-power",
    preview_mode: str = "learned",
    replay_dir: str | None = None,
    replay_count: int = 1,
) -> dict[str, object]:
    if env_config.team_size <= env_config.pick_size:
        raise ValueError(
            "Fixed-bot evaluation now expects real preview semantics: team_size must be greater than pick_size."
        )
    if opponent_kind not in _OPPONENT_PLAYER_TYPES:
        choices = ", ".join(sorted(_OPPONENT_PLAYER_TYPES))
        raise ValueError(
            f"Unknown opponent_kind={opponent_kind!r}; choose from {choices}."
        )
    if preview_mode not in {"learned", "max-bst", "random"}:
        raise ValueError(
            f"Unknown preview_mode={preview_mode!r}; choose from learned, max-bst, random."
        )

    random.seed(seed)
    preview_rng = random.Random(seed + 1_000_003)

    policy = ModelPolicyController(agent, device=device, action_mode=action_mode)
    game_data = GameData(gen=9)
    species_idx_lookup = _species_idx_display_lookup(game_data)
    opponent_specs = plan_opponent_team_specs(
        opponent_team_pool,
        model_team_spec=model_team_spec,
        species_bst_lookup=species_bst_lookup,
        pick_size=env_config.pick_size,
        num_games=num_games,
        seed=seed,
    )
    matchups = PlannedMatchups(model_team_spec, opponent_specs)
    preview_records: list[dict[str, object]] = []

    def live_team_preview(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
        preview_out = policy.choose_team_preview(obs)
        learned_picks = np.asarray(preview_out["picks"], dtype=np.int64)
        if preview_mode == "learned":
            picks = learned_picks
        elif preview_mode == "max-bst":
            picks = np.asarray(model_team_spec.max_bst_pick_indices, dtype=np.int64)
        else:
            own_mask = np.asarray(
                obs.get("own_mask", np.zeros(env_config.team_size, dtype=bool)),
                dtype=bool,
            )
            available = [int(index) for index in np.flatnonzero(~own_mask).tolist()]
            picks = np.asarray(
                preview_rng.sample(available, k=env_config.pick_size),
                dtype=np.int64,
            )
        preview_records.append(
            {
                "picks": picks.copy(),
                "learned_picks": learned_picks.copy(),
                "preview_mode": preview_mode,
                "entropy": float(preview_out["entropy"]),
                "log_prob": float(preview_out["log_prob"]),
                "opponent_bring_logits": np.asarray(
                    preview_out["opponent_bring_logits"],
                    dtype=np.float32,
                ),
                "opponent_lead_pair_logits": np.asarray(
                    preview_out["opponent_lead_pair_logits"],
                    dtype=np.float32,
                ),
            }
        )
        return picks, "/team " + "".join(str(int(pick) + 1) for pick in picks)

    def team_pair_provider() -> tuple[str, str]:
        return matchups.team_pair_provider()

    def opponent_factory(loop):
        opponent_type = _OPPONENT_PLAYER_TYPES[opponent_kind]
        return opponent_type(
            account_configuration=make_local_account_configuration("fixedbot"),
            battle_format=env_config.battle_format,
            server_configuration=make_battlestream_configuration(env_config),
            loop=loop,
            team=model_team_spec.packed_team,
            species_bst_lookup=species_bst_lookup,
            pick_size=env_config.pick_size,
        )

    env = BattleEnv(
        env_config,
        game_data,
        opponent_factory,
        team_preview_fn=live_team_preview,
        team_pair_provider=team_pair_provider,
        player_kwargs={
            "account_configuration": make_local_account_configuration("evalagent"),
        },
    )

    battle_turns: list[int] = []
    decision_steps: list[int] = []
    entropies: list[float] = []
    action_stats: Counter[str] = Counter()
    preview_results: list[dict[str, object]] = []
    battle_results: list[dict[str, object]] = []
    replay_artifacts: list[dict[str, str]] = []

    replay_path_obj = None if replay_dir is None else Path(replay_dir)

    try:
        for game_index in range(num_games):
            policy.reset_battle()
            obs, masks = env.reset()
            opponent_spec = matchups.consume_active_opponent()
            if not preview_records:
                raise RuntimeError(
                    "Model team preview did not run before the first observation arrived."
                )
            preview_record = preview_records.pop(0)
            done = False
            reward = 0.0
            step_count = 0
            trace: list[dict[str, object]] = []
            battle_action_stats: Counter[str] = Counter()
            first_tera_turn: int | None = None
            first_tera_step: int | None = None
            first_tera_slot: int | None = None

            while not done:
                if max_turns > 0 and step_count >= max_turns:
                    raise RuntimeError(
                        f"Evaluation exceeded max_turns={max_turns} on game {game_index + 1}."
                    )
                battle = env.last_battle
                if battle is None:
                    raise RuntimeError(
                        "BattleEnv returned no active battle during evaluation."
                    )

                action_out = policy.choose_action(obs, masks)
                actions = np.asarray(action_out["actions"], dtype=np.int64)
                tera_choice = int(actions[2])
                entropies.append(float(action_out["entropy"]))
                for slot, action_idx in enumerate(actions[:2]):
                    slot_stats = _classify_slot_action(
                        battle, slot, int(action_idx), tera_choice
                    )
                    action_stats.update(slot_stats)
                    battle_action_stats.update(slot_stats)

                action_descriptions = [
                    _describe_slot_action(battle, 0, int(actions[0]), tera_choice),
                    _describe_slot_action(battle, 1, int(actions[1]), tera_choice),
                ]
                joint_stats = _classify_joint_action(action_descriptions)
                action_stats.update(joint_stats)
                battle_action_stats.update(joint_stats)
                if first_tera_turn is None and any(
                    description["tera"] for description in action_descriptions
                ):
                    first_tera_turn = int(getattr(battle, "turn", 0))
                    first_tera_step = step_count + 1
                    first_tera_slot = next(
                        int(description["slot"])
                        for description in action_descriptions
                        if description["tera"]
                    )
                protect_contexts = _protect_attempt_contexts(
                    battle, action_descriptions
                )
                trace.append(
                    {
                        "decision_step": step_count + 1,
                        "battle_turn": int(getattr(battle, "turn", 0)),
                        "entropy": float(action_out["entropy"]),
                        "actions": action_descriptions,
                    }
                )

                obs, masks, reward, done, info = env.step(actions)
                protect_outcomes = _classify_protect_outcomes(protect_contexts)
                action_stats.update(protect_outcomes)
                battle_action_stats.update(protect_outcomes)
                step_count += 1

            won = reward > 0
            drawn = reward == 0
            battle_turn = int(
                info.get("battle_turn", getattr(env.last_battle, "turn", 0))
            )
            battle_turns.append(battle_turn)
            decision_steps.append(step_count)
            action_stats.update(_classify_battle_mechanics(battle_action_stats))

            if "team_preview_obs" not in info or "team_preview_picks" not in info:
                raise RuntimeError(
                    "Evaluation battle finished without team preview metadata."
                )
            preview_obs = info["team_preview_obs"]
            preview_picks = np.asarray(info["team_preview_picks"], dtype=np.int64)
            if not np.array_equal(preview_picks, preview_record["picks"]):
                raise RuntimeError(
                    "Recorded model team preview picks disagreed with the episode summary."
                )

            own_preview_species = _species_names_from_obs(
                preview_obs["own_species"], species_idx_lookup
            )
            opponent_preview_species = _species_names_from_obs(
                preview_obs["opp_species"], species_idx_lookup
            )
            picked_species_ordered = [
                own_preview_species[int(index)] for index in preview_picks
            ]
            predicted_opponent_bring = sorted(
                int(index)
                for index in np.argsort(
                    -preview_record["opponent_bring_logits"],
                    kind="stable",
                )[: env_config.pick_size]
            )
            predicted_opponent_lead_flat = int(
                np.argmax(preview_record["opponent_lead_pair_logits"])
            )
            predicted_opponent_lead = [
                predicted_opponent_lead_flat // env_config.team_size,
                predicted_opponent_lead_flat % env_config.team_size,
            ]
            actual_opponent_bring = list(opponent_spec.max_bst_pick_indices)
            actual_opponent_lead = actual_opponent_bring[:2]
            preview_result = {
                "pick_indices": [int(index) for index in preview_picks.tolist()],
                "learned_pick_indices": [
                    int(index) for index in preview_record["learned_picks"].tolist()
                ],
                "preview_mode": preview_mode,
                "picked_species_ordered": picked_species_ordered,
                "picked_species_sorted": sorted(picked_species_ordered),
                "entropy": float(preview_record["entropy"]),
                "log_prob": float(preview_record["log_prob"]),
                "opponent_preview_species": opponent_preview_species,
                "predicted_opponent_bring_indices": predicted_opponent_bring,
                "actual_opponent_bring_indices": actual_opponent_bring,
                "predicted_opponent_lead_indices": predicted_opponent_lead,
                "actual_opponent_lead_indices": actual_opponent_lead,
            }
            preview_results.append(preview_result)

            battle_result = {
                "game": game_index + 1,
                "won": bool(won),
                "drawn": bool(drawn),
                "outcome": "win" if won else ("draw" if drawn else "loss"),
                "reward": float(reward),
                "battle_turn": battle_turn,
                "decision_steps": step_count,
                "battle_tag": info.get("battle_tag", ""),
                "opponent_title": opponent_spec.title,
                "opponent_file": opponent_spec.file,
                "opponent_species": list(opponent_spec.species),
                "preview": preview_result,
                "tera_used": first_tera_turn is not None,
                "first_tera_turn": first_tera_turn,
                "first_tera_decision_step": first_tera_step,
                "first_tera_slot": first_tera_slot,
                "switch_actions": int(battle_action_stats.get("switch_actions", 0)),
            }
            battle_results.append(battle_result)

            if replay_path_obj is not None and len(replay_artifacts) < replay_count:
                battle = env.last_battle
                if battle is None:
                    raise RuntimeError(
                        "Expected a completed battle when saving a replay artifact."
                    )
                replay_html = save_replay(battle, str(replay_path_obj))
                replay_json = save_replay_json(battle, str(replay_path_obj))
                trace_path = replay_path_obj / f"{_battle_tag_slug(battle)}_trace.json"
                _write_json(
                    trace_path,
                    {
                        "model_team": model_team_spec.to_summary(),
                        "opponent_team": opponent_spec.to_summary(),
                        "preview": preview_result,
                        "result": battle_result,
                        "trace": trace,
                    },
                )
                replay_artifacts.append(
                    {
                        "html": replay_html or "",
                        "json": replay_json or "",
                        "trace": str(trace_path),
                    }
                )

        outcomes = outcome_summary(
            [float(result["reward"]) for result in battle_results]
        )
        return {
            "games": num_games,
            "opponent_kind": opponent_kind,
            "preview_mode": preview_mode,
            "team_size": env_config.team_size,
            "pick_size": env_config.pick_size,
            "distinct_opponent_teams": len(
                {result["opponent_title"] for result in battle_results}
            ),
            **outcomes,
            "avg_battle_turns": float(np.mean(battle_turns)) if battle_turns else 0.0,
            "avg_decision_steps": float(np.mean(decision_steps))
            if decision_steps
            else 0.0,
            "action_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
            "action_entropy_std": float(np.std(entropies)) if entropies else 0.0,
            "preview_summary": _summarize_preview_results(preview_results),
            "opponent_preview_prediction": _summarize_opponent_preview_predictions(
                preview_results
            ),
            "action_breakdown": _summarize_action_stats(action_stats),
            "battle_behavior": _summarize_battle_behavior(battle_results),
            "battle_results": battle_results,
            "replays": replay_artifacts,
        }
    finally:
        env.close()


def load_agent_for_evaluation(
    *,
    checkpoint_path: str | None,
    device: torch.device,
    model_config: ModelConfig,
    game_data: GameData,
):
    from ..model.agent import PokeAgent

    agent = PokeAgent(model_config, game_data).to(device)
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        if not isinstance(checkpoint, Mapping) or not isinstance(
            checkpoint.get("agent"), Mapping
        ):
            raise ValueError(
                f"Evaluation checkpoint {checkpoint_path!r} must contain an agent state mapping."
            )

        metadata = checkpoint.get("model_metadata")
        if metadata is not None:
            if not isinstance(metadata, Mapping):
                raise ValueError(
                    f"Evaluation checkpoint {checkpoint_path!r} has invalid model metadata."
                )
            checkpoint_config = metadata.get("model_config")
            expected_config = {
                "type": f"{type(model_config).__module__}.{type(model_config).__qualname__}",
                "values": asdict(model_config),
            }
            if checkpoint_config != expected_config:
                raise ValueError(
                    f"Evaluation checkpoint {checkpoint_path!r} model config does not match "
                    "the evaluator architecture."
                )
            if metadata.get("game_data_signature") != agent.game_data_signature:
                raise ValueError(
                    f"Evaluation checkpoint {checkpoint_path!r} game-data vocabulary does "
                    "not match the evaluator."
                )

        try:
            agent.load_state_dict(checkpoint["agent"], strict=True)
        except RuntimeError as exc:
            raise ValueError(
                f"Evaluation checkpoint {checkpoint_path!r} does not exactly match the "
                f"evaluator model: {exc}"
            ) from exc
    agent.eval()
    agent.requires_grad_(False)
    return agent
