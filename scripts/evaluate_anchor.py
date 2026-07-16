"""Evaluate one checkpoint against a frozen anchor checkpoint."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from pokerl.config import EnvConfig, ModelConfig, TrainingConfig
from pokerl.data import GameData
from pokerl.env.showdown_env import BattleEnv, default_showdown_dir
from pokerl.eval.fixed_bot import (
    ModelPolicyController,
    _classify_battle_mechanics,
    _classify_joint_action,
    _classify_protect_outcomes,
    _classify_slot_action,
    _describe_slot_action,
    _protect_attempt_contexts,
    _summarize_action_stats,
    _summarize_battle_behavior,
    load_agent_for_evaluation,
)
from pokerl.eval.metrics import mirrored_candidate_summary, outcome_summary
from pokerl.eval.schedule import (
    balanced_distinct_team_pairs,
    team_pair_schedule_summary,
)
from pokerl.model.signature import model_signature as build_model_signature
from pokerl.team_pool import (
    TeamPoolEntry,
    default_team_pool_manifest_path,
    load_team_pool,
)
from pokerl.training.league import League
from pokerl.training.population import CheckpointRef
from pokerl.training.population_evidence import build_pairwise_evidence_block


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a candidate checkpoint against a frozen self-play anchor."
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Candidate trainer or league checkpoint."
    )
    parser.add_argument(
        "--checkpoint-policy-id",
        required=True,
        help="Stable population policy id for the candidate checkpoint.",
    )
    parser.add_argument(
        "--anchor", required=True, help="Frozen anchor trainer or league checkpoint."
    )
    parser.add_argument(
        "--anchor-policy-id",
        required=True,
        help="Stable population policy id for the frozen anchor checkpoint.",
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
        "--same-policy-role-bias",
        action="store_true",
        help=(
            "Require identical checkpoint bytes and report external-player role bias "
            "instead of candidate strength. Requires mirrored evaluation."
        ),
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
    serialized = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if out_path.exists():
        if out_path.read_text(encoding="utf-8") == serialized:
            return out_path.resolve()
        raise FileExistsError(
            f"Refusing to overwrite immutable evaluation summary: {out_path!s}"
        )
    temporary = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        temporary.write_text(serialized, encoding="utf-8")
        temporary.replace(out_path)
    finally:
        temporary.unlink(missing_ok=True)
    return out_path.resolve()


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
    entropies: list[float] = []
    action_stats: Counter[str] = Counter()
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
            battle_action_stats: Counter[str] = Counter()
            first_tera_turn: int | None = None
            first_tera_step: int | None = None
            first_tera_slot: int | None = None

            while not done:
                if args.max_turns > 0 and step_count >= args.max_turns:
                    raise RuntimeError(
                        f"Anchor evaluation exceeded max_turns={args.max_turns} "
                        f"on game {game_index + 1}."
                    )
                battle = env.last_battle
                if battle is None:
                    raise RuntimeError(
                        "BattleEnv returned no active battle during anchor evaluation."
                    )
                action_out = policy.choose_action(obs, masks)
                actions = np.asarray(action_out["actions"], dtype=np.int64)
                tera_choice = int(actions[2])
                entropies.append(float(action_out["entropy"]))
                for slot, action_idx in enumerate(actions[:2]):
                    slot_stats = _classify_slot_action(
                        battle,
                        slot,
                        int(action_idx),
                        tera_choice,
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
                    battle,
                    action_descriptions,
                )
                obs, masks, reward, done, info = env.step(actions)
                protect_outcomes = _classify_protect_outcomes(protect_contexts)
                action_stats.update(protect_outcomes)
                battle_action_stats.update(protect_outcomes)
                step_count += 1

            battle_turn = int(info.get("battle_turn", 0))
            rewards.append(float(reward))
            battle_turns.append(battle_turn)
            decision_steps.append(step_count)
            action_stats.update(_classify_battle_mechanics(battle_action_stats))
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
                    "won": bool(reward > 0),
                    "preview": preview,
                    "tera_used": first_tera_turn is not None,
                    "first_tera_turn": first_tera_turn,
                    "first_tera_decision_step": first_tera_step,
                    "first_tera_slot": first_tera_slot,
                    "switch_actions": int(
                        battle_action_stats.get("switch_actions", 0)
                    ),
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
        "action_entropy_mean": float(np.mean(entropies)) if entropies else 0.0,
        "action_entropy_std": float(np.std(entropies)) if entropies else 0.0,
        "action_breakdown": _summarize_action_stats(action_stats),
        "battle_behavior": _summarize_battle_behavior(game_results),
        "games_detail": game_results,
    }


def evaluate_anchor_matchup(args: argparse.Namespace) -> dict[str, object]:
    if args.games <= 0:
        raise ValueError("--games must be positive.")

    checkpoint_ref = CheckpointRef.from_file(args.checkpoint)
    anchor_ref = CheckpointRef.from_file(args.anchor)
    identical_checkpoints = checkpoint_ref.sha256 == anchor_ref.sha256
    if args.same_policy_role_bias:
        if not args.mirrored:
            raise ValueError("Same-policy role-bias evaluation requires --mirrored.")
        if not identical_checkpoints:
            raise ValueError(
                "Same-policy role-bias evaluation requires identical checkpoint bytes."
            )
    else:
        if args.checkpoint_policy_id == args.anchor_policy_id:
            raise ValueError("Candidate and anchor policy ids must be distinct.")
        if identical_checkpoints:
            raise ValueError("Candidate and anchor checkpoints contain identical bytes.")

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
        usage="evaluation",
    )
    planned_pairs = balanced_distinct_team_pairs(
        team_pool.entries,
        games=args.games,
        seed=args.seed,
    )
    schedule_summary = team_pair_schedule_summary(planned_pairs)

    forward = _run_anchor_direction(
        args=args,
        env_config=env_config,
        model_config=model_config,
        game_data=game_data,
        device=device,
        external_checkpoint=str(checkpoint_ref.resolve(REPO_ROOT)),
        opponent_checkpoint=str(anchor_ref.resolve(REPO_ROOT)),
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
                external_checkpoint=str(anchor_ref.resolve(REPO_ROOT)),
                opponent_checkpoint=str(checkpoint_ref.resolve(REPO_ROOT)),
                planned_pairs=planned_pairs,
                swap_teams=True,
                label="anchor_external_candidate_opponent",
            )
        )

    candidate_results = None
    role_bias_results = None
    if args.same_policy_role_bias:
        role_bias_results = outcome_summary(
            wins=sum(int(direction["wins"]) for direction in directions),
            losses=sum(int(direction["losses"]) for direction in directions),
            draws=sum(int(direction["draws"]) for direction in directions),
        )
        role_bias_results["external_role_score_advantage"] = (
            float(role_bias_results["score_rate"]) - 0.5
        )
    else:
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
    # Reject a checkpoint that changed while games were being played.  The
    # summary then always describes the exact bytes it names.
    checkpoint_ref.verify(REPO_ROOT)
    anchor_ref.verify(REPO_ROOT)
    model_signature = build_model_signature(
        model_config,
        game_data.vocab_signature(),
    )
    summary: dict[str, object] = {
        "config": {
            "checkpoint_policy_id": args.checkpoint_policy_id,
            "anchor_policy_id": args.anchor_policy_id,
            "checkpoint": checkpoint_ref.to_dict(),
            "anchor": anchor_ref.to_dict(),
            "device": str(device),
            "battle_format": env_config.battle_format,
            "games_per_role": args.games,
            "mirrored": args.mirrored,
            "evaluation_mode": (
                "same_policy_role_bias"
                if args.same_policy_role_bias
                else "checkpoint_comparison"
            ),
            "action_mode": args.action_mode,
            "seed": args.seed,
            "simulator_rng_seeded": False,
            "team_pool_manifest": str(manifest_path),
            "team_pool_format": args.team_pool_format,
            "team_pool_subset": args.team_pool_subset,
            "team_pool_size": team_pool.size,
            "team_pool_signature": team_pool.signature(),
            "team_pair_schedule": schedule_summary,
            "league_coalesce_ms": args.league_coalesce_ms,
            "battlestream_startup_timeout": env_config.battlestream_startup_timeout,
            "model_signature": model_signature,
        },
        "results": {
            "directions": directions,
        },
    }
    results = summary["results"]
    assert isinstance(results, dict)
    if role_bias_results is not None:
        results["role_bias"] = role_bias_results
    else:
        assert candidate_results is not None
        results["candidate_perspective"] = candidate_results
    if args.mirrored and not args.same_policy_role_bias:
        assert candidate_results is not None
        summary["pairwise_evidence"] = build_pairwise_evidence_block(
            checkpoint_policy_id=args.checkpoint_policy_id,
            anchor_policy_id=args.anchor_policy_id,
            checkpoint=checkpoint_ref,
            anchor=anchor_ref,
            battle_format=env_config.battle_format,
            team_pool_signature=team_pool.signature(),
            model_signature=model_signature,
            action_mode=args.action_mode,
            games_per_role=args.games,
            directions=directions,
            candidate_perspective=candidate_results,
        )
    return summary


def main() -> None:
    args = parse_args()
    summary = evaluate_anchor_matchup(args)
    if args.summary_path:
        out_path = _write_json(args.summary_path, summary)
        print(f"Wrote summary to {out_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
