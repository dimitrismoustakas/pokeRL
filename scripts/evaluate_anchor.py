"""Evaluate one checkpoint against a frozen anchor checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from pokerl.config import EnvConfig, ModelConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.env.showdown_env import BattleEnv, default_showdown_dir
from pokerl.eval.fixed_bot import ModelPolicyController, load_agent_for_evaluation
from pokerl.eval.metrics import mirrored_candidate_summary, outcome_summary
from pokerl.team_pool import (
    TeamPoolEntry,
    default_team_pool_manifest_path,
    load_team_pool,
)
from pokerl.training.league import League


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate checkpoint against a frozen self-play anchor."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Candidate trainer or league checkpoint."
    )
    parser.add_argument(
        "--anchor", required=True, help="Frozen anchor trainer or league checkpoint."
    )
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument(
        "--battle-format", default="gen9vgc2026regf", help="Showdown battle format."
    )
    parser.add_argument(
        "--showdown-dir",
        default=None,
        help="Pokemon Showdown checkout for local BattleStream.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=None,
        help="Local BattleStream worker count.",
    )
    parser.add_argument(
        "--battles-per-worker",
        type=int,
        default=None,
        help="Max concurrent local battles per worker.",
    )
    parser.add_argument(
        "--runtime-loop-count",
        type=int,
        default=None,
        help="Local BattleStream runtime loop count.",
    )
    parser.add_argument(
        "--battlestream-startup-timeout",
        type=float,
        default=None,
        help="Seconds to wait for a local BattleStream battle to produce protocol output.",
    )
    parser.add_argument(
        "--league-coalesce-ms",
        type=float,
        default=0.0,
        help="Anchor inference coalescing window.",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=64,
        help="Battles per player role (twice this total when --mirrored is enabled).",
    )
    parser.add_argument(
        "--action-mode",
        choices=["sample", "argmax"],
        default="sample",
        help="Shared action mode for both candidate and anchor policies.",
    )
    parser.add_argument(
        "--mirrored",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Repeat the same team pairs with checkpoints and player/team roles reversed.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Python/Torch/numpy/team-pool seed. Showdown battle RNG is not seed-controlled.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Fail if a battle exceeds this many decisions.",
    )
    parser.add_argument(
        "--team-pool-manifest",
        default=None,
        help="Optional explicit team-pool manifest; overrides --team-pool-format.",
    )
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
    parser.add_argument(
        "--summary-path", default=None, help="Optional path for JSON results."
    )
    return parser.parse_args()


def _write_json(path: str, payload: dict) -> Path:
    out_path = Path(path)
    if not out_path.is_absolute():
        out_path = REPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


class _PlannedAnchorMatchups:
    def __init__(
        self,
        pairs: list[tuple[TeamPoolEntry, TeamPoolEntry]],
        *,
        swap_teams: bool,
    ):
        if not pairs:
            raise ValueError(
                "Anchor evaluation requires at least one planned team pair."
            )
        self._pairs = list(pairs) + [pairs[-1]]
        self._swap_teams = swap_teams
        self._prepared_index = 0
        self._active: list[dict[str, str]] = []

    def provide(self) -> tuple[str, str]:
        if self._prepared_index >= len(self._pairs):
            raise RuntimeError("Planned anchor team pairs were exhausted unexpectedly.")
        left, right = self._pairs[self._prepared_index]
        self._prepared_index += 1
        external, opponent = (right, left) if self._swap_teams else (left, right)
        self._active.append(
            {
                "external_team_title": external.title,
                "external_team_file": external.file,
                "opponent_team_title": opponent.title,
                "opponent_team_file": opponent.file,
            }
        )
        return external.packed_team, opponent.packed_team

    def consume(self) -> dict[str, str]:
        if not self._active:
            raise RuntimeError("No prepared anchor matchup was available after reset.")
        return self._active.pop(0)


def _run_anchor_direction(
    *,
    args: argparse.Namespace,
    env_config: EnvConfig,
    model_config: ModelConfig,
    game_data: GameData,
    device: torch.device,
    external_checkpoint: str,
    opponent_checkpoint: str,
    planned_pairs: list[tuple[TeamPoolEntry, TeamPoolEntry]],
    swap_teams: bool,
    label: str,
) -> dict[str, object]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    external_agent = load_agent_for_evaluation(
        checkpoint_path=external_checkpoint,
        device=device,
        model_config=model_config,
        game_data=game_data,
    )
    # Validate the frozen opponent independently before League constructs it.
    load_agent_for_evaluation(
        checkpoint_path=opponent_checkpoint,
        device=torch.device("cpu"),
        model_config=model_config,
        game_data=game_data,
    )
    policy = ModelPolicyController(
        external_agent,
        device=device,
        action_mode=args.action_mode,
    )

    league_config = TrainingConfig(league_coalesce_ms=args.league_coalesce_ms)
    opponent_league = League(
        league_config,
        env_config,
        game_data,
        external_agent,
        device,
        inference_greedy=args.action_mode == "argmax",
    )
    matchups = _PlannedAnchorMatchups(planned_pairs, swap_teams=swap_teams)
    preview_records: list[dict[str, object]] = []

    def team_preview_fn(obs: dict[str, np.ndarray]) -> tuple[np.ndarray, str]:
        preview_out = policy.choose_team_preview(obs)
        picks = np.asarray(preview_out["picks"], dtype=np.int64)
        preview_records.append(
            {
                "pick_indices": [int(index) for index in picks],
                "entropy": float(preview_out["entropy"]),
            }
        )
        return picks, "/team " + "".join(str(int(pick) + 1) for pick in picks)

    def opponent_factory(loop):
        opponent = opponent_league.make_opponent(
            loop,
            checkpoint_path=opponent_checkpoint,
        )
        if opponent is None:
            raise RuntimeError("Anchor opponent could not be constructed.")
        return opponent

    env = BattleEnv(
        env_config,
        game_data,
        opponent_factory,
        team_preview_fn=team_preview_fn,
        team_pair_provider=matchups.provide,
    )

    rewards: list[float] = []
    battle_turns: list[int] = []
    decision_steps: list[int] = []
    game_results: list[dict[str, object]] = []
    try:
        for game_index in range(args.games):
            policy.reset_battle()
            obs, masks = env.reset()
            matchup = matchups.consume()
            if not preview_records:
                raise RuntimeError(
                    "External policy team preview was not recorded during reset."
                )
            preview = preview_records.pop(0)
            done = False
            reward = 0.0
            step_count = 0

            while not done:
                if args.max_turns > 0 and step_count >= args.max_turns:
                    raise RuntimeError(
                        f"Anchor evaluation exceeded max_turns={args.max_turns} "
                        f"on game {game_index + 1}."
                    )
                action_out = policy.choose_action(obs, masks)
                obs, masks, reward, done, info = env.step(
                    np.asarray(action_out["actions"], dtype=np.int64)
                )
                step_count += 1

            battle_turn = int(info.get("battle_turn", 0))
            rewards.append(float(reward))
            battle_turns.append(battle_turn)
            decision_steps.append(step_count)
            game_results.append(
                {
                    "game": game_index + 1,
                    "reward": float(reward),
                    "outcome": "win"
                    if reward > 0
                    else ("draw" if reward == 0 else "loss"),
                    "battle_turn": battle_turn,
                    "decision_steps": step_count,
                    "battle_tag": info.get("battle_tag", ""),
                    "preview": preview,
                    **matchup,
                }
            )
    finally:
        env.close()
        opponent_league.close()

    return {
        "label": label,
        "external_checkpoint": external_checkpoint,
        "opponent_checkpoint": opponent_checkpoint,
        "teams_swapped_from_forward_schedule": swap_teams,
        **outcome_summary(rewards),
        "avg_battle_turns": float(np.mean(battle_turns)) if battle_turns else 0.0,
        "median_battle_turns": float(np.median(battle_turns)) if battle_turns else 0.0,
        "avg_decision_steps": float(np.mean(decision_steps)) if decision_steps else 0.0,
        "games_detail": game_results,
    }


def evaluate_anchor_matchup(args: argparse.Namespace) -> dict[str, object]:
    if args.games <= 0:
        raise ValueError("--games must be positive.")

    showdown_dir = args.showdown_dir or default_showdown_dir()
    if showdown_dir is None:
        raise SystemExit("--showdown-dir is required, or set POKERL_SHOWDOWN_DIR.")
    if not Path(showdown_dir).exists():
        raise SystemExit(f"Pokemon Showdown checkout not found: {showdown_dir}")

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
    )
    model_config = ModelConfig(pick_size=env_config.pick_size)
    game_data = GameData(gen=9)
    manifest_path = (
        Path(args.team_pool_manifest)
        if args.team_pool_manifest
        else default_team_pool_manifest_path(REPO_ROOT, args.team_pool_format)
    )
    team_pool = load_team_pool(
        manifest_path,
        subset=args.team_pool_subset,
        seed=args.seed,
    )
    planned_pairs = [team_pool.sample_pair(distinct=True) for _ in range(args.games)]

    forward = _run_anchor_direction(
        args=args,
        env_config=env_config,
        model_config=model_config,
        game_data=game_data,
        device=device,
        external_checkpoint=args.checkpoint,
        opponent_checkpoint=args.anchor,
        planned_pairs=planned_pairs,
        swap_teams=False,
        label="candidate_external",
    )
    directions = [forward]
    if args.mirrored:
        directions.append(
            _run_anchor_direction(
                args=args,
                env_config=env_config,
                model_config=model_config,
                game_data=game_data,
                device=device,
                external_checkpoint=args.anchor,
                opponent_checkpoint=args.checkpoint,
                planned_pairs=planned_pairs,
                swap_teams=True,
                label="anchor_external_candidate_opponent",
            )
        )

    candidate_results = (
        mirrored_candidate_summary(forward, directions[1])
        if args.mirrored
        else {
            key: forward[key]
            for key in (
                "games",
                "wins",
                "losses",
                "draws",
                "win_rate",
                "score_rate",
                "win_rate_ci95",
                "score_rate_ci95",
            )
        }
    )
    return {
        "config": {
            "checkpoint": args.checkpoint,
            "anchor": args.anchor,
            "device": str(device),
            "battle_format": env_config.battle_format,
            "games_per_role": args.games,
            "mirrored": args.mirrored,
            "action_mode": args.action_mode,
            "seed": args.seed,
            "simulator_rng_seeded": False,
            "team_pool_manifest": str(manifest_path),
            "team_pool_format": args.team_pool_format,
            "team_pool_subset": args.team_pool_subset,
            "team_pool_size": team_pool.size,
            "team_pool_signature": team_pool.signature(),
            "league_coalesce_ms": args.league_coalesce_ms,
            "battlestream_startup_timeout": env_config.battlestream_startup_timeout,
        },
        "results": {
            "candidate_perspective": candidate_results,
            "directions": directions,
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
