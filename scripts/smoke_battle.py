"""Run one local smoke battle against a configurable baseline opponent."""

from __future__ import annotations

import argparse
import json
import time
from types import MethodType

import numpy as np
import torch
from poke_env import ServerConfiguration
from poke_env.battle import DoubleBattle
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from pokerl.config import EnvConfig, ModelConfig
from pokerl.data import GameData
from pokerl.env.actions import FIRST_SLOT_MASK_ROW, NUM_TARGETS, PASS_ACTION, SWITCH_OFFSET
from pokerl.env.showdown_env import BattleEnv, make_local_account_configuration
from pokerl.model.agent import PokeAgent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one local pokeRL smoke battle.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Maximum decision steps. Use 0 for no decision-step cap.",
    )
    parser.add_argument("--policy", choices=["max-power", "random-legal", "legal-first", "model"], default="max-power")
    parser.add_argument(
        "--opponent",
        choices=["max-base-power", "random", "simple-heuristic"],
        default="max-base-power",
        help="Built-in poke-env baseline opponent.",
    )
    parser.add_argument("--trace", action="store_true", help="Print per-decision timing and turn progress.")
    parser.add_argument("--trace-limit", type=int, default=20, help="Maximum traced decision steps.")
    parser.add_argument("--json", action="store_true", help="Emit the final summary as JSON.")
    return parser.parse_args()


def server_config(env_config: EnvConfig) -> ServerConfiguration:
    host, port = env_config.server_url.split(":")
    return ServerConfiguration(
        f"ws://{host}:{port}/showdown/websocket",
        f"http://{host}:{port}/action.php",
    )


def choose_action(
    policy: str,
    obs: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    agent: PokeAgent,
    hidden,
    device: torch.device,
    rng: np.random.Generator,
    battle: DoubleBattle | None,
) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor] | None]:
    joint_masks = masks["joint"]
    can_tera = masks["can_tera"]

    if policy == "max-power":
        if battle is None:
            raise RuntimeError("Smoke battle policy requires an active battle state.")
        action_1 = _max_power_action(battle, 0, joint_masks[FIRST_SLOT_MASK_ROW], rng)
        action_2 = _max_power_action(battle, 1, joint_masks[action_1 + 1], rng)
        action = np.array([action_1, action_2, _first_legal_tera_choice(can_tera, action_1, action_2)], dtype=np.int64)
        return action, hidden

    if policy == "random-legal":
        action_1 = _preferred_legal_action(joint_masks[FIRST_SLOT_MASK_ROW], rng)
        action_2 = _preferred_legal_action(joint_masks[action_1 + 1], rng)
        action = np.array([action_1, action_2, _random_legal_tera_choice(can_tera, action_1, action_2, rng)], dtype=np.int64)
        return action, hidden

    if policy == "legal-first":
        action_1 = int(joint_masks[FIRST_SLOT_MASK_ROW].argmax())
        action_2 = int(joint_masks[action_1 + 1].argmax())
        action = np.array([action_1, action_2, _first_legal_tera_choice(can_tera, action_1, action_2)], dtype=np.int64)
        return action, hidden

    obs_t = {key: torch.from_numpy(value).unsqueeze(0).to(device) for key, value in obs.items()}
    masks_t = {
        key: torch.from_numpy(value).unsqueeze(0).to(device).bool()
        for key, value in masks.items()
    }
    with torch.no_grad():
        out = agent(obs_t, hidden, action_masks=masks_t)
    return out["actions"].cpu().numpy()[0], out["hidden"]


def _preferred_legal_action(slot_mask: np.ndarray, rng: np.random.Generator) -> int:
    legal_actions = np.flatnonzero(slot_mask)
    preference_buckets = (
        legal_actions[legal_actions < SWITCH_OFFSET],
        legal_actions[(legal_actions >= SWITCH_OFFSET) & (legal_actions < PASS_ACTION)],
        legal_actions[legal_actions == PASS_ACTION],
    )
    for bucket in preference_buckets:
        if bucket.size:
            return int(rng.choice(bucket))
    raise RuntimeError("Smoke battle found no legal action for an active slot.")


def _max_power_action(
    battle: DoubleBattle,
    slot: int,
    slot_mask: np.ndarray,
    rng: np.random.Generator,
) -> int:
    legal_actions = np.flatnonzero(slot_mask)
    regular_moves = legal_actions[legal_actions < SWITCH_OFFSET]
    if regular_moves.size:
        def move_power(action: int) -> int:
            return battle.available_moves[slot][action // NUM_TARGETS].base_power or 0

        best_power = max(move_power(action) for action in regular_moves)
        strongest = [int(action) for action in regular_moves if move_power(action) == best_power]
        return int(rng.choice(strongest))
    return _preferred_legal_action(slot_mask, rng)


def _legal_tera_choices(can_tera: np.ndarray, action_1: int, action_2: int) -> np.ndarray:
    legal = [0]
    if can_tera[0] and action_1 < SWITCH_OFFSET:
        legal.append(1)
    if can_tera[1] and action_2 < SWITCH_OFFSET:
        legal.append(2)
    return np.array(legal, dtype=np.int64)


def _first_legal_tera_choice(can_tera: np.ndarray, action_1: int, action_2: int) -> int:
    legal = _legal_tera_choices(can_tera, action_1, action_2)
    if legal.size == 0:
        raise RuntimeError("Smoke battle found no legal tera choice for the selected joint action.")
    return int(legal[0])


def _random_legal_tera_choice(
    can_tera: np.ndarray,
    action_1: int,
    action_2: int,
    rng: np.random.Generator,
) -> int:
    legal = _legal_tera_choices(can_tera, action_1, action_2)
    if legal.size == 0:
        raise RuntimeError("Smoke battle found no legal tera choice for the selected joint action.")
    return int(rng.choice(legal))


def _attach_choice_timer(player, timing: dict[str, float | int], key_prefix: str) -> None:
    original_choose_move = player.choose_move

    def timed_choose_move(self, battle):
        started = time.perf_counter()
        result = original_choose_move(battle)
        timing[f"{key_prefix}_choice_calls"] += 1
        timing[f"{key_prefix}_choice_sec"] += time.perf_counter() - started
        return result

    player.choose_move = MethodType(timed_choose_move, player)


def _empty_timing() -> dict[str, float | int]:
    return {
        "policy_calls": 0,
        "policy_sec": 0.0,
        "step_calls": 0,
        "step_sec": 0.0,
        "opponent_choice_calls": 0,
        "opponent_choice_sec": 0.0,
    }


def _timing_summary(timing: dict[str, float | int]) -> dict[str, float | int]:
    policy_calls = int(timing["policy_calls"])
    step_calls = int(timing["step_calls"])
    opponent_calls = int(timing["opponent_choice_calls"])
    policy_sec = float(timing["policy_sec"])
    step_sec = float(timing["step_sec"])
    opponent_sec = float(timing["opponent_choice_sec"])
    return {
        "policy_calls": policy_calls,
        "policy_total_ms": round(policy_sec * 1000, 2),
        "policy_avg_ms": round((policy_sec / policy_calls) * 1000, 2) if policy_calls else 0.0,
        "step_calls": step_calls,
        "step_total_ms": round(step_sec * 1000, 2),
        "step_avg_ms": round((step_sec / step_calls) * 1000, 2) if step_calls else 0.0,
        "opponent_choice_calls": opponent_calls,
        "opponent_choice_total_ms": round(opponent_sec * 1000, 2),
        "opponent_choice_avg_ms": round((opponent_sec / opponent_calls) * 1000, 2) if opponent_calls else 0.0,
        "step_minus_opponent_total_ms": round(max(step_sec - opponent_sec, 0.0) * 1000, 2),
    }


def _emit(payload: dict, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload), flush=True)
    else:
        print(payload, flush=True)


def _build_opponent_factory(name: str, env_config: EnvConfig, timing: dict[str, float | int]):
    opponent_classes = {
        "max-base-power": MaxBasePowerPlayer,
        "random": RandomPlayer,
        "simple-heuristic": SimpleHeuristicsPlayer,
    }

    def factory(loop):
        opponent = opponent_classes[name](
            account_configuration=make_local_account_configuration("smokeopp"),
            battle_format=env_config.battle_format,
            server_configuration=server_config(env_config),
            loop=loop,
        )
        _attach_choice_timer(opponent, timing, "opponent")
        return opponent

    return factory


def main() -> None:
    args = parse_args()
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    started_at = time.perf_counter()

    game_data = GameData(gen=9)
    env_config = EnvConfig(num_envs=1)
    agent = PokeAgent(ModelConfig(), game_data).to(device)
    timing = _empty_timing()
    opponent_factory = _build_opponent_factory(args.opponent, env_config, timing)
    env = BattleEnv(
        env_config,
        game_data,
        opponent_factory,
        player_kwargs={
            "account_configuration": make_local_account_configuration("smokeagent"),
        },
    )
    battle_started_at = time.perf_counter()

    hidden = agent.initial_hidden(1)
    rng = np.random.default_rng(0)
    done = False
    decision_steps = 0
    reward = 0.0
    battle_tag = ""

    try:
        reset_started = time.perf_counter()
        obs, masks = env.reset()
        reset_sec = time.perf_counter() - reset_started
        unlimited_steps = args.max_turns <= 0

        while not done and (unlimited_steps or decision_steps < args.max_turns):
            battle_turn_before = getattr(env.last_battle, "turn", None)

            policy_started = time.perf_counter()
            action, hidden = choose_action(
                args.policy,
                obs,
                masks,
                agent,
                hidden,
                device,
                rng,
                env.last_battle,
            )
            timing["policy_calls"] += 1
            timing["policy_sec"] += time.perf_counter() - policy_started

            step_started = time.perf_counter()
            obs, masks, reward, done, info = env.step(action)
            timing["step_calls"] += 1
            timing["step_sec"] += time.perf_counter() - step_started

            battle_tag = info.get("battle_tag", battle_tag)
            decision_steps += 1

            if args.trace and decision_steps <= args.trace_limit:
                _emit(
                    {
                        "event": "decision",
                        "decision_step": decision_steps,
                        "battle_turn_before": battle_turn_before,
                        "battle_turn_after": getattr(env.last_battle, "turn", None),
                        "policy_avg_ms": round(float(timing["policy_sec"]) * 1000 / int(timing["policy_calls"]), 2),
                        "step_avg_ms": round(float(timing["step_sec"]) * 1000 / int(timing["step_calls"]), 2),
                        "opponent_choice_avg_ms": round(
                            float(timing["opponent_choice_sec"]) * 1000 / int(timing["opponent_choice_calls"]),
                            2,
                        ) if int(timing["opponent_choice_calls"]) else 0.0,
                        "done": done,
                    },
                    args.json,
                )

        summary = {
            "status": "completed" if done else "incomplete",
            "device": str(device),
            "policy": args.policy,
            "opponent": args.opponent,
            "decision_steps": decision_steps,
            "battle_turn": getattr(env.last_battle, "turn", None),
            "reward": float(reward),
            "won": reward > 0,
            "battle_tag": battle_tag,
            "total_elapsed_sec": round(time.perf_counter() - started_at, 2),
            "setup_elapsed_sec": round(battle_started_at - started_at, 2),
            "battle_elapsed_sec": round(time.perf_counter() - battle_started_at, 2),
            "reset_ms": round(reset_sec * 1000, 2),
            **_timing_summary(timing),
        }
        _emit(summary, args.json)

        if not done and not unlimited_steps:
            raise RuntimeError(
                f"Smoke battle exceeded max turns ({args.max_turns}); "
                f"battle_turn={summary['battle_turn']}, battle_tag={battle_tag}, opponent={args.opponent}, "
                f"battle_elapsed_sec={summary['battle_elapsed_sec']:.2f}."
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
