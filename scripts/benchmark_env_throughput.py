"""Benchmark local BattleStream env throughput without PPO or league inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from poke_env.battle import DoubleBattle
from poke_env.player import MaxBasePowerPlayer, RandomPlayer, SimpleHeuristicsPlayer

from pokerl.config import EnvConfig
from pokerl.data import GameData
from pokerl.env.actions import FIRST_SLOT_MASK_ROW, NUM_TARGETS, PASS_ACTION, SWITCH_OFFSET
from pokerl.env.showdown_env import (
    BattleEnv,
    ParallelBattleEnv,
    battlestream_summary,
    default_showdown_dir,
    make_battlestream_configuration,
    make_local_account_configuration,
)
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark pokeRL env/local BattleStream throughput without model training."
    )
    parser.add_argument("--battle-format", default="gen9vgc2026regf", help="Showdown battle format.")
    parser.add_argument("--showdown-dir", default=None, help="Pokemon Showdown checkout for local BattleStream.")
    parser.add_argument("--num-envs", type=int, default=72, help="Number of parallel BattleEnv instances.")
    parser.add_argument("--worker-count", type=int, default=16, help="Local BattleStream worker count.")
    parser.add_argument("--battles-per-worker", type=int, default=9, help="Max battles per local worker.")
    parser.add_argument("--runtime-loop-count", type=int, default=4, help="Local BattleStream runtime loop count.")
    parser.add_argument("--pool-mode", choices=("single", "shared"), default="shared", help="BattleStream pool mode.")
    parser.add_argument(
        "--battlestream-startup-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for a local BattleStream battle to produce protocol output.",
    )
    parser.add_argument("--steps", type=int, default=256, help="Vector env steps to run after reset.")
    parser.add_argument(
        "--policy",
        choices=("legal-first", "random-legal", "max-power"),
        default="legal-first",
        help="Policy used by the pokeRL-controlled side.",
    )
    parser.add_argument(
        "--opponent",
        choices=("max-base-power", "random", "simple-heuristic"),
        default="max-base-power",
        help="Built-in poke-env opponent.",
    )
    parser.add_argument(
        "--team-pool-format",
        default="gen9vgc2026regf",
        help="Load teams from data/team_pools/<format>/manifest.json.",
    )
    parser.add_argument(
        "--team-pool-subset",
        choices=("curated", "generated", "pool"),
        default="pool",
        help="Which team pool subset to sample from.",
    )
    parser.add_argument("--team-pool-seed", type=int, default=0, help="Seed for team-pool sampling.")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed for random-legal policy.")
    parser.add_argument("--collect-timing", action="store_true", help="Collect BattleEnv timing counters.")
    parser.add_argument("--summary-path", default=None, help="Optional JSON summary path.")
    return parser.parse_args()


def _write_json(path: str, payload: dict) -> Path:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _team_preview_first_four(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
    team_size = int(obs["own_species"].shape[0])
    picks = np.arange(min(4, team_size), dtype=np.int64)
    return picks, "/team " + "".join(str(int(pick) + 1) for pick in picks)


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
    raise RuntimeError("No legal action for an active slot.")


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
    return np.asarray(legal, dtype=np.int64)


def _choose_actions(
    *,
    policy: str,
    masks: dict[str, np.ndarray],
    envs: list[BattleEnv],
    rng: np.random.Generator,
) -> np.ndarray:
    actions = np.empty((len(envs), 3), dtype=np.int64)
    joint_masks = masks["joint"]
    can_tera = masks["can_tera"]

    for env_index, env in enumerate(envs):
        if policy == "legal-first":
            action_1 = int(joint_masks[env_index, FIRST_SLOT_MASK_ROW].argmax())
            action_2 = int(joint_masks[env_index, action_1 + 1].argmax())
            tera = 0
        elif policy == "random-legal":
            action_1 = _preferred_legal_action(joint_masks[env_index, FIRST_SLOT_MASK_ROW], rng)
            action_2 = _preferred_legal_action(joint_masks[env_index, action_1 + 1], rng)
            tera = int(rng.choice(_legal_tera_choices(can_tera[env_index], action_1, action_2)))
        elif policy == "max-power":
            battle = env.last_battle
            if battle is None:
                raise RuntimeError("max-power policy requires an active battle.")
            action_1 = _max_power_action(battle, 0, joint_masks[env_index, FIRST_SLOT_MASK_ROW], rng)
            action_2 = _max_power_action(battle, 1, joint_masks[env_index, action_1 + 1], rng)
            tera = 0
        else:
            raise ValueError(f"Unsupported policy: {policy}")
        actions[env_index] = (action_1, action_2, tera)

    return actions


def _build_opponent_factory(env_config: EnvConfig, opponent_name: str):
    opponent_classes = {
        "max-base-power": MaxBasePowerPlayer,
        "random": RandomPlayer,
        "simple-heuristic": SimpleHeuristicsPlayer,
    }
    opponent_class = opponent_classes[opponent_name]

    def factory(loop):
        return opponent_class(
            account_configuration=make_local_account_configuration("envbenchopp"),
            battle_format=env_config.battle_format,
            server_configuration=make_battlestream_configuration(env_config),
            loop=loop,
        )

    return factory


def run_benchmark(args: argparse.Namespace) -> dict:
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive.")
    if args.steps <= 0:
        raise ValueError("--steps must be positive.")

    showdown_dir = args.showdown_dir or default_showdown_dir()
    if showdown_dir is None:
        raise SystemExit("--showdown-dir is required, or set POKERL_SHOWDOWN_DIR.")
    if not Path(showdown_dir).exists():
        raise SystemExit(f"Pokemon Showdown checkout not found: {showdown_dir}")

    env_config = EnvConfig(
        battle_format=args.battle_format,
        num_envs=args.num_envs,
        team_size=6,
        pick_size=4,
        showdown_dir=showdown_dir,
        worker_count=args.worker_count,
        battles_per_worker=args.battles_per_worker,
        runtime_loop_count=args.runtime_loop_count,
        pool_mode=args.pool_mode,
        battlestream_startup_timeout=args.battlestream_startup_timeout,
    )
    game_data = GameData(gen=9)
    team_pool = load_team_pool(
        default_team_pool_manifest_path(REPO_ROOT, args.team_pool_format),
        subset=args.team_pool_subset,
        seed=args.team_pool_seed,
    )

    def team_pair_provider() -> tuple[str, str]:
        return team_pool.sample_packed_pair(distinct=True)

    opponent_factory = _build_opponent_factory(env_config, args.opponent)
    envs = [
        BattleEnv(
            env_config,
            game_data,
            opponent_factory,
            team_preview_fn=_team_preview_first_four,
            team_pair_provider=team_pair_provider,
            collect_timing=args.collect_timing,
        )
        for _ in range(args.num_envs)
    ]
    parallel_env = ParallelBattleEnv(envs, collect_timing=args.collect_timing)
    rng = np.random.default_rng(args.seed)

    completed_battles = 0
    episode_secs: list[float] = []
    episode_turns: list[int] = []
    reset_started = time.perf_counter()
    try:
        obs, masks = parallel_env.reset()
        reset_sec = time.perf_counter() - reset_started
        run_started = time.perf_counter()
        for _ in range(args.steps):
            actions = _choose_actions(
                policy=args.policy,
                masks=masks,
                envs=parallel_env.envs,
                rng=rng,
            )
            obs, masks, _rewards, _dones, infos = parallel_env.step(actions)
            for info in infos:
                if "episode_reward" not in info:
                    continue
                completed_battles += 1
                if "episode_sec" in info:
                    episode_secs.append(float(info["episode_sec"]))
                if "battle_turn" in info:
                    episode_turns.append(int(info["battle_turn"]))
        run_sec = time.perf_counter() - run_started
        timing = parallel_env.snapshot_timing() if args.collect_timing else {}
    finally:
        parallel_env.close()

    total_env_steps = args.steps * args.num_envs
    return {
        "config": {
            "battle_format": env_config.battle_format,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "policy": args.policy,
            "opponent": args.opponent,
            "team_pool_format": args.team_pool_format,
            "team_pool_subset": args.team_pool_subset,
            "team_pool_size": team_pool.size,
            "collect_timing": args.collect_timing,
            "battlestream": battlestream_summary(env_config),
        },
        "results": {
            "reset_sec": reset_sec,
            "run_sec": run_sec,
            "total_env_steps": total_env_steps,
            "env_steps_per_sec": total_env_steps / run_sec if run_sec > 0 else 0.0,
            "vector_steps_per_sec": args.steps / run_sec if run_sec > 0 else 0.0,
            "completed_battles": completed_battles,
            "avg_episode_sec": float(np.mean(episode_secs)) if episode_secs else 0.0,
            "avg_episode_turns": float(np.mean(episode_turns)) if episode_turns else 0.0,
            "timing": timing,
        },
    }


def main() -> None:
    args = parse_args()
    summary = run_benchmark(args)
    if args.summary_path:
        out_path = _write_json(args.summary_path, summary)
        print(f"Wrote summary to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
