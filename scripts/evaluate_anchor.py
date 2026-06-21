"""Evaluate one checkpoint against a frozen anchor checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from pokerl.config import EnvConfig, ModelConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.env.showdown_env import BattleEnv, default_showdown_dir
from pokerl.eval.fixed_bot import ModelPolicyController, load_agent_for_evaluation
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool
from pokerl.training.league import League


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate checkpoint against a frozen self-play anchor."
    )
    parser.add_argument("--checkpoint", required=True, help="Candidate trainer or league checkpoint.")
    parser.add_argument("--anchor", required=True, help="Frozen anchor trainer or league checkpoint.")
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument("--battle-format", default="gen9vgc2026regf", help="Showdown battle format.")
    parser.add_argument("--showdown-dir", default=None, help="Pokemon Showdown checkout for local BattleStream.")
    parser.add_argument("--worker-count", type=int, default=None, help="Local BattleStream worker count.")
    parser.add_argument("--battles-per-worker", type=int, default=None, help="Max concurrent local battles per worker.")
    parser.add_argument("--runtime-loop-count", type=int, default=None, help="Local BattleStream runtime loop count.")
    parser.add_argument(
        "--battlestream-startup-timeout",
        type=float,
        default=None,
        help="Seconds to wait for a local BattleStream battle to produce protocol output.",
    )
    parser.add_argument("--pool-mode", choices=("single", "shared"), default=None, help="Local BattleStream pool mode.")
    parser.add_argument("--league-coalesce-ms", type=float, default=0.0, help="Anchor inference coalescing window.")
    parser.add_argument("--games", type=int, default=32, help="Number of evaluation battles.")
    parser.add_argument(
        "--action-mode",
        choices=["sample", "argmax"],
        default="argmax",
        help="Whether the candidate policy samples or plays greedily.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch/numpy/team-pool seed.")
    parser.add_argument("--max-turns", type=int, default=200, help="Fail if a battle exceeds this many decisions.")
    parser.add_argument(
        "--team-pool-format",
        default="gen9vgc2026regf",
        help="Load teams from data/team_pools/<format>/manifest.json.",
    )
    parser.add_argument(
        "--team-pool-subset",
        choices=["curated", "generated", "pool"],
        default="pool",
        help="Which manifest subset to sample teams from.",
    )
    parser.add_argument("--summary-path", default=None, help="Optional path for JSON results.")
    return parser.parse_args()


def _write_json(path: str, payload: dict) -> Path:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def evaluate_anchor_matchup(args: argparse.Namespace) -> dict[str, object]:
    if args.games <= 0:
        raise ValueError("--games must be positive.")

    showdown_dir = args.showdown_dir or default_showdown_dir()
    if showdown_dir is None:
        raise SystemExit("--showdown-dir is required, or set POKERL_SHOWDOWN_DIR.")
    if not Path(showdown_dir).exists():
        raise SystemExit(f"Pokemon Showdown checkout not found: {showdown_dir}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    env_config = EnvConfig(
        battle_format=args.battle_format,
        num_envs=1,
        team_size=6,
        pick_size=4,
        showdown_dir=showdown_dir,
        worker_count=args.worker_count,
        battles_per_worker=args.battles_per_worker,
        runtime_loop_count=args.runtime_loop_count,
        battlestream_startup_timeout=(
            args.battlestream_startup_timeout
            if args.battlestream_startup_timeout is not None
            else EnvConfig.battlestream_startup_timeout
        ),
        pool_mode=args.pool_mode or EnvConfig.pool_mode,
    )
    model_config = ModelConfig(pick_size=env_config.pick_size)
    game_data = GameData(gen=9)
    team_pool = load_team_pool(
        default_team_pool_manifest_path(REPO_ROOT, args.team_pool_format),
        subset=args.team_pool_subset,
        seed=args.seed,
    )

    candidate = load_agent_for_evaluation(
        checkpoint_path=args.checkpoint,
        device=device,
        model_config=model_config,
        game_data=game_data,
    )
    policy = ModelPolicyController(candidate, device=device, action_mode=args.action_mode)

    league_config = TrainingConfig(league_coalesce_ms=args.league_coalesce_ms)
    anchor_league = League(
        league_config,
        env_config,
        game_data,
        candidate,
        device,
    )

    def team_preview_fn(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
        preview_out = policy.choose_team_preview(obs)
        picks = np.asarray(preview_out["picks"], dtype=np.int64)
        return picks, "/team " + "".join(str(int(pick) + 1) for pick in picks)

    def team_pair_provider() -> tuple[str, str]:
        return team_pool.sample_packed_pair(distinct=True)

    def opponent_factory(loop):
        opponent = anchor_league.make_opponent(loop, checkpoint_path=args.anchor)
        if opponent is None:
            raise RuntimeError("Anchor opponent could not be constructed.")
        return opponent

    env = BattleEnv(
        env_config,
        game_data,
        opponent_factory,
        team_preview_fn=team_preview_fn,
        team_pair_provider=team_pair_provider,
    )

    wins = 0
    losses = 0
    battle_turns: list[int] = []
    decision_steps: list[int] = []
    try:
        for game_index in range(args.games):
            policy.reset_battle()
            obs, masks = env.reset()
            done = False
            reward = 0.0
            step_count = 0

            while not done:
                if args.max_turns > 0 and step_count >= args.max_turns:
                    raise RuntimeError(
                        f"Anchor evaluation exceeded max_turns={args.max_turns} on game {game_index + 1}."
                    )
                action_out = policy.choose_action(obs, masks)
                obs, masks, reward, done, info = env.step(np.asarray(action_out["actions"], dtype=np.int64))
                step_count += 1

            wins += int(reward > 0)
            losses += int(reward <= 0)
            battle_turns.append(int(info.get("battle_turn", 0)))
            decision_steps.append(step_count)
    finally:
        env.close()
        anchor_league.close()

    return {
        "config": {
            "checkpoint": args.checkpoint,
            "anchor": args.anchor,
            "device": str(device),
            "battle_format": env_config.battle_format,
            "games": args.games,
            "action_mode": args.action_mode,
            "seed": args.seed,
            "team_pool_format": args.team_pool_format,
            "team_pool_subset": args.team_pool_subset,
            "team_pool_size": team_pool.size,
            "league_coalesce_ms": args.league_coalesce_ms,
            "battlestream_startup_timeout": env_config.battlestream_startup_timeout,
        },
        "results": {
            "wins": wins,
            "losses": losses,
            "win_rate": float(wins / args.games),
            "avg_battle_turns": float(np.mean(battle_turns)) if battle_turns else 0.0,
            "avg_decision_steps": float(np.mean(decision_steps)) if decision_steps else 0.0,
        },
    }


def main() -> None:
    args = parse_args()
    summary = evaluate_anchor_matchup(args)
    if args.summary_path:
        out_path = _write_json(args.summary_path, summary)
        print(f"Wrote summary to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
