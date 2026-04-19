"""Evaluate a policy against a fixed max-BST, max-base-power bot."""

from __future__ import annotations

import json
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
from ..data import GameData
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
            "fixed_team_size": len(self.max_bst_pick_species),
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


class ModelPolicyController:
    def __init__(self, agent, *, device: torch.device, action_mode: str):
        self.agent = agent
        self.device = device
        self.action_mode = action_mode
        self.hidden = agent.initial_hidden(1)

    def reset_battle(self) -> None:
        self.hidden = self.agent.initial_hidden(1)

    def choose_action(self, obs: dict[str, np.ndarray], masks: np.ndarray) -> dict[str, object]:
        obs_t = {key: torch.from_numpy(value).unsqueeze(0).to(self.device) for key, value in obs.items()}
        masks_t = torch.from_numpy(masks).unsqueeze(0).to(device=self.device, dtype=torch.bool)

        with torch.inference_mode():
            context, active_reps, own_reps, opp_active_reps = self.agent.spatial(obs_t)
            context, new_hidden = self.agent.memory(context, self.hidden)
            action_out = self.agent.action_head(
                context,
                active_reps,
                own_reps,
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
    team_spec: FixedTeamSpec,
    num_games: int,
    action_mode: str,
    max_turns: int,
    replay_dir: str | None = None,
    replay_count: int = 1,
) -> dict[str, object]:
    policy = ModelPolicyController(agent, device=device, action_mode=action_mode)

    def fixed_team_preview(_obs: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
        return np.arange(env_config.team_size, dtype=np.int64), ""

    def team_pair_provider() -> tuple[str, str]:
        return team_spec.max_bst_packed_team, team_spec.max_bst_packed_team

    def opponent_factory(loop):
        return MaxBasePowerPlayer(
            account_configuration=make_local_account_configuration("fixedbot"),
            battle_format=env_config.battle_format,
            server_configuration=make_server_configuration(env_config),
            loop=loop,
            team=team_spec.max_bst_packed_team,
        )

    env = BattleEnv(
        env_config,
        GameData(gen=9),
        opponent_factory,
        team_preview_fn=fixed_team_preview,
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
    battle_results: list[dict[str, object]] = []
    replay_artifacts: list[dict[str, str]] = []

    replay_path_obj = None if replay_dir is None else Path(replay_dir)

    try:
        for game_index in range(num_games):
            policy.reset_battle()
            obs, masks = env.reset()
            done = False
            reward = 0.0
            step_count = 0
            trace: list[dict[str, object]] = []

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
                    action_stats.update(_classify_slot_action(battle, slot, int(action_idx)))

                trace.append(
                    {
                        "decision_step": step_count + 1,
                        "battle_turn": int(getattr(battle, "turn", 0)),
                        "entropy": float(action_out["entropy"]),
                        "actions": [
                            _describe_slot_action(battle, 0, int(actions[0])),
                            _describe_slot_action(battle, 1, int(actions[1])),
                        ],
                    }
                )

                obs, masks, reward, done, info = env.step(actions)
                step_count += 1

            won = reward > 0
            wins += int(won)
            battle_turn = int(info.get("battle_turn", getattr(env.last_battle, "turn", 0)))
            battle_turns.append(battle_turn)
            decision_steps.append(step_count)
            battle_result = {
                "game": game_index + 1,
                "won": bool(won),
                "reward": float(reward),
                "battle_turn": battle_turn,
                "decision_steps": step_count,
                "battle_tag": info.get("battle_tag", ""),
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
                        "team": team_spec.to_summary(),
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
            "fixed_team_size": env_config.team_size,
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / num_games) if num_games else 0.0,
            "avg_battle_turns": float(np.mean(battle_turns)) if battle_turns else 0.0,
            "avg_decision_steps": float(np.mean(decision_steps)) if decision_steps else 0.0,
            "action_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
            "action_entropy_std": float(np.std(entropies)) if entropies else 0.0,
            "action_breakdown": _summarize_action_stats(action_stats),
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
