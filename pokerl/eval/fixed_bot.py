"""Evaluate a policy against a MaxBasePower bot with real team preview battles."""

from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from poke_env.battle import DoubleBattle, Pokemon
from poke_env.data import GenData
from poke_env.player import MaxBasePowerPlayer
from poke_env.teambuilder.teambuilder import Teambuilder

from ..config import EnvConfig, ModelConfig
from ..data import GameData, UNKNOWN_IDX
from ..model.agent import load_state_dict_compatible
from ..env.actions import (
    TARGET_ALLY,
    TARGET_NONE,
    TARGET_OPPONENT_1,
    TARGET_OPPONENT_2,
    TARGET_SELF,
    decode_action,
)
from ..env.showdown_env import BattleEnv, make_local_account_configuration, make_server_configuration
from ..team_pool import TeamPool, TeamPoolEntry
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
            raise RuntimeError(f"Could not infer species from packed team member {teambuilder_pokemon!r}.")
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


def _species_names_from_obs(obs_species: np.ndarray, idx_lookup: dict[int, str]) -> list[str]:
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
            raise RuntimeError(f"No BST entry found for species {name!r} ({species_id!r}).")
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
        raise RuntimeError(f"Could not find a team titled {title!r} in the requested pool.")
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

    candidate_entries = [entry for entry in pool.entries if entry.file != model_team_spec.file]
    if not candidate_entries:
        candidate_entries = list(pool.entries)
    if not candidate_entries:
        raise RuntimeError("Opponent team planning requires at least one team in the pool.")

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
    def __init__(self, model_team_spec: FixedTeamSpec, opponent_specs: list[FixedTeamSpec]):
        if not opponent_specs:
            raise ValueError("PlannedMatchups requires at least one opponent team.")
        self.model_team_spec = model_team_spec
        self._prepared_specs = list(opponent_specs) + [opponent_specs[-1]]
        self._prepared_index = 0
        self._active_specs: list[FixedTeamSpec] = []

    def team_pair_provider(self) -> tuple[str, str]:
        if self._prepared_index >= len(self._prepared_specs):
            raise RuntimeError("PlannedMatchups exhausted unexpectedly during evaluation.")
        opponent_spec = self._prepared_specs[self._prepared_index]
        self._prepared_index += 1
        self._active_specs.append(opponent_spec)
        return self.model_team_spec.packed_team, opponent_spec.packed_team

    def consume_active_opponent(self) -> FixedTeamSpec:
        if not self._active_specs:
            raise RuntimeError("No active opponent was prepared for the current evaluation battle.")
        return self._active_specs.pop(0)


class DeterministicPreviewMaxBasePowerPlayer(MaxBasePowerPlayer):
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
            index for index, _ in sorted(ranked, key=lambda item: (-item[1], item[0]))[: self._pick_size]
        )
        for pick in picks:
            members[pick]._selected_in_teampreview = True
        return "/team " + "".join(str(pick + 1) for pick in picks)


class ModelPolicyController:
    def __init__(self, agent, *, device: torch.device, action_mode: str):
        self.agent = agent
        self.device = device
        self.action_mode = action_mode
        self.hidden = agent.initial_hidden(1)

    def reset_battle(self) -> None:
        self.hidden = self.agent.initial_hidden(1)

    def choose_team_preview(self, obs: dict[str, np.ndarray]) -> dict[str, object]:
        obs_t = {key: torch.from_numpy(value).unsqueeze(0).to(self.device) for key, value in obs.items()}

        with torch.inference_mode():
            preview_out = self.agent.team_preview(
                obs_t,
                greedy=self.action_mode == "argmax",
            )

        return {
            "picks": preview_out["picks"].cpu().numpy()[0],
            "entropy": float(preview_out["entropy"].item()),
            "log_prob": float(preview_out["log_prob"].item()),
        }

    def choose_action(self, obs: dict[str, np.ndarray], masks: np.ndarray) -> dict[str, object]:
        obs_t = {key: torch.from_numpy(value).unsqueeze(0).to(self.device) for key, value in obs.items()}
        masks_t = torch.from_numpy(masks).unsqueeze(0).to(device=self.device, dtype=torch.bool)

        with torch.inference_mode():
            context, active_reps, own_reps, opp_reps, opp_active_reps = self.agent.spatial(obs_t)
            context, new_hidden = self.agent.memory(context, self.hidden)
            opp_bench_mask = self.agent._bench_mask(obs_t.get("opp_mask"), obs_t["opp_roles"])
            action_out = self.agent.action_head(
                context,
                active_reps,
                own_reps,
                opp_reps,
                opp_bench_mask,
                opp_active_reps,
                masks_t,
                greedy=self.action_mode == "argmax",
            )
            self.hidden = None if new_hidden is None else tuple(state.detach() for state in new_hidden)
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
    return max((int(move.base_power or 0) for move in battle.available_moves[slot]), default=0)


def _classify_slot_action(battle: DoubleBattle, slot: int, action_idx: int) -> Counter[str]:
    kind, move_idx, target_idx, tera = decode_action(int(action_idx))
    stats: Counter[str] = Counter()
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

    if slot >= len(battle.available_moves) or move_idx >= len(battle.available_moves[slot]):
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

    if base_power > 0 and best_base_power > 0:
        if base_power == best_base_power:
            stats["max_legal_base_power_choices"] += 1
        else:
            stats["non_max_legal_base_power_choices"] += 1
    elif base_power == 0 and best_base_power > 0:
        stats["non_damaging_while_damage_available_actions"] += 1

    return stats


def _describe_slot_action(battle: DoubleBattle, slot: int, action_idx: int) -> dict[str, object]:
    kind, move_idx, target_idx, tera = decode_action(int(action_idx))
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

    counts = {
        key: int(value)
        for key, value in action_stats.items()
        if key not in {"chosen_base_power_total", "best_legal_base_power_total"}
    }
    rates = {
        "move_rate": _safe_rate(counts.get("move_actions", 0), slot_decisions),
        "tera_move_rate": _safe_rate(counts.get("tera_move_actions", 0), slot_decisions),
        "switch_rate": _safe_rate(counts.get("switch_actions", 0), slot_decisions),
        "pass_rate": _safe_rate(counts.get("pass_actions", 0), slot_decisions),
        "damaging_move_rate": _safe_rate(counts.get("damaging_move_actions", 0), slot_decisions),
        "status_move_rate": _safe_rate(counts.get("status_move_actions", 0), slot_decisions),
        "own_team_hitting_move_rate": _safe_rate(counts.get("own_team_hitting_move_actions", 0), slot_decisions),
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
    }

    power = {
        "avg_chosen_base_power_per_move_action": float(chosen_power_total / move_actions) if move_actions else 0.0,
        "avg_best_legal_base_power_per_slot_decision": float(best_power_total / slot_decisions) if slot_decisions else 0.0,
        "chosen_vs_best_base_power_ratio": float(chosen_power_total / best_power_total) if best_power_total else 0.0,
    }
    return {
        "counts": counts,
        "rates": rates,
        "power": power,
    }


def _preview_pick_key(species: list[str]) -> str:
    return " / ".join(species)


def _summarize_preview_results(preview_results: list[dict[str, object]]) -> dict[str, object]:
    ordered_picks = Counter()
    unordered_picks = Counter()
    species_counts = Counter()
    lead_counts = Counter()
    entropies: list[float] = []

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
    total_slots = sum(len(result["picked_species_ordered"]) for result in preview_results)
    total_leads = sum(min(2, len(result["picked_species_ordered"])) for result in preview_results)
    return {
        "games": total_games,
        "distinct_ordered_picks": len(ordered_picks),
        "distinct_unordered_picks": len(unordered_picks),
        "entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "entropy_std": float(np.std(entropies)) if entropies else 0.0,
        "ordered_pick_sets": _counter_rows(ordered_picks, total_games),
        "unordered_pick_sets": _counter_rows(unordered_picks, total_games),
        "species_pick_counts": _counter_rows(species_counts, total_slots),
        "lead_species_counts": _counter_rows(lead_counts, total_leads),
    }


def _summarize_battle_behavior(battle_results: list[dict[str, object]]) -> dict[str, object]:
    tera_used_games = [result for result in battle_results if result.get("tera_used")]
    tera_unused_games = [result for result in battle_results if not result.get("tera_used")]
    tera_turns = [float(result["first_tera_turn"]) for result in tera_used_games if result.get("first_tera_turn")]
    switch_counts = [int(result.get("switch_actions", 0)) for result in battle_results]

    return {
        "games_with_tera": len(tera_used_games),
        "tera_game_rate": _safe_rate(len(tera_used_games), len(battle_results)),
        "win_rate_when_tera_used": _safe_rate(
            sum(1 for result in tera_used_games if result.get("won")),
            len(tera_used_games),
        ),
        "win_rate_when_tera_unused": _safe_rate(
            sum(1 for result in tera_unused_games if result.get("won")),
            len(tera_unused_games),
        ),
        "avg_first_tera_turn": float(np.mean(tera_turns)) if tera_turns else 0.0,
        "avg_switch_actions_per_game": float(np.mean(switch_counts)) if switch_counts else 0.0,
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
    replay_dir: str | None = None,
    replay_count: int = 1,
) -> dict[str, object]:
    if env_config.team_size <= env_config.pick_size:
        raise ValueError(
            "Fixed-bot evaluation now expects real preview semantics: team_size must be greater than pick_size."
        )

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
        picks = np.asarray(preview_out["picks"], dtype=np.int64)
        preview_records.append(
            {
                "picks": picks.copy(),
                "entropy": float(preview_out["entropy"]),
                "log_prob": float(preview_out["log_prob"]),
            }
        )
        return picks, "/team " + "".join(str(int(pick) + 1) for pick in picks)

    def team_pair_provider() -> tuple[str, str]:
        return matchups.team_pair_provider()

    def opponent_factory(loop):
        return DeterministicPreviewMaxBasePowerPlayer(
            account_configuration=make_local_account_configuration("fixedbot"),
            battle_format=env_config.battle_format,
            server_configuration=make_server_configuration(env_config),
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

    wins = 0
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
                raise RuntimeError("Model team preview did not run before the first observation arrived.")
            preview_record = preview_records.pop(0)
            done = False
            reward = 0.0
            step_count = 0
            trace: list[dict[str, object]] = []
            battle_action_stats: Counter[str] = Counter()
            first_tera_turn: int | None = None
            first_tera_step: int | None = None

            while not done:
                if max_turns > 0 and step_count >= max_turns:
                    raise RuntimeError(
                        f"Evaluation exceeded max_turns={max_turns} on game {game_index + 1}."
                    )
                battle = env.last_battle
                if battle is None:
                    raise RuntimeError("BattleEnv returned no active battle during evaluation.")

                action_out = policy.choose_action(obs, masks)
                actions = np.asarray(action_out["actions"], dtype=np.int64)
                entropies.append(float(action_out["entropy"]))
                for slot, action_idx in enumerate(actions):
                    slot_stats = _classify_slot_action(battle, slot, int(action_idx))
                    action_stats.update(slot_stats)
                    battle_action_stats.update(slot_stats)

                action_descriptions = [
                    _describe_slot_action(battle, 0, int(actions[0])),
                    _describe_slot_action(battle, 1, int(actions[1])),
                ]
                if first_tera_turn is None and any(description["tera"] for description in action_descriptions):
                    first_tera_turn = int(getattr(battle, "turn", 0))
                    first_tera_step = step_count + 1
                trace.append(
                    {
                        "decision_step": step_count + 1,
                        "battle_turn": int(getattr(battle, "turn", 0)),
                        "entropy": float(action_out["entropy"]),
                        "actions": action_descriptions,
                    }
                )

                obs, masks, reward, done, info = env.step(actions)
                step_count += 1

            won = reward > 0
            wins += int(won)
            battle_turn = int(info.get("battle_turn", getattr(env.last_battle, "turn", 0)))
            battle_turns.append(battle_turn)
            decision_steps.append(step_count)

            if "team_preview_obs" not in info or "team_preview_picks" not in info:
                raise RuntimeError("Evaluation battle finished without team preview metadata.")
            preview_obs = info["team_preview_obs"]
            preview_picks = np.asarray(info["team_preview_picks"], dtype=np.int64)
            if not np.array_equal(preview_picks, preview_record["picks"]):
                raise RuntimeError("Recorded model team preview picks disagreed with the episode summary.")

            own_preview_species = _species_names_from_obs(preview_obs["own_species"], species_idx_lookup)
            opponent_preview_species = _species_names_from_obs(preview_obs["opp_species"], species_idx_lookup)
            picked_species_ordered = [own_preview_species[int(index)] for index in preview_picks]
            preview_result = {
                "pick_indices": [int(index) for index in preview_picks.tolist()],
                "picked_species_ordered": picked_species_ordered,
                "picked_species_sorted": sorted(picked_species_ordered),
                "entropy": float(preview_record["entropy"]),
                "log_prob": float(preview_record["log_prob"]),
                "opponent_preview_species": opponent_preview_species,
            }
            preview_results.append(preview_result)

            battle_result = {
                "game": game_index + 1,
                "won": bool(won),
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
                "switch_actions": int(battle_action_stats.get("switch_actions", 0)),
            }
            battle_results.append(battle_result)

            if replay_path_obj is not None and len(replay_artifacts) < replay_count:
                battle = env.last_battle
                if battle is None:
                    raise RuntimeError("Expected a completed battle when saving a replay artifact.")
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

        losses = num_games - wins
        return {
            "games": num_games,
            "opponent_kind": "max-base-power",
            "team_size": env_config.team_size,
            "pick_size": env_config.pick_size,
            "distinct_opponent_teams": len({result["opponent_title"] for result in battle_results}),
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / num_games) if num_games else 0.0,
            "avg_battle_turns": float(np.mean(battle_turns)) if battle_turns else 0.0,
            "avg_decision_steps": float(np.mean(decision_steps)) if decision_steps else 0.0,
            "action_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
            "action_entropy_std": float(np.std(entropies)) if entropies else 0.0,
            "preview_summary": _summarize_preview_results(preview_results),
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
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        load_state_dict_compatible(agent, checkpoint["agent"])
    agent.eval()
    agent.requires_grad_(False)
    return agent
