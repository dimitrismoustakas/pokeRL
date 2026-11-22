"""Evaluate a random or trained policy against a reproducible fixed-bot panel."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from pokerl.config import EnvConfig, ModelConfig
from pokerl.data import GameData
from pokerl.env.showdown_env import resolve_showdown_dir
from pokerl.eval.fixed_bot import (
    build_species_bst_lookup,
    evaluate_policy_vs_fixed_bot,
    load_agent_for_evaluation,
    select_max_bst_team_spec,
    select_team_spec_by_title,
)
from pokerl.model.signature import model_signature as build_model_signature
from pokerl.team_pool import default_team_pool_manifest_path, load_team_pool
from pokerl.training.population import CheckpointRef


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pokeRL against a fixed bot using real 6-to-4 team preview battles."
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional checkpoint path; omit for random initialization.",
    )
    parser.add_argument("--device", default=None, help="cpu or cuda; defaults to auto")
    parser.add_argument(
        "--battle-format", default="gen9vgc2026regf", help="Showdown battle format."
    )
    parser.add_argument(
        "--simulator",
        choices=("pikapro", "showdown"),
        default="pikapro",
        help="Battle simulator. Pikapro runs Regulation F in process.",
    )
    parser.add_argument(
        "--showdown-dir",
        default=None,
        help="Checkout used with --simulator showdown.",
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
        "--opponent",
        choices=["random", "max-base-power", "simple-heuristics"],
        default="max-base-power",
        help="Fixed battle policy. All choices use the same deterministic max-BST preview.",
    )
    parser.add_argument(
        "--team-pool-format",
        default="gen9vgc2026regf",
        help="Load teams from data/team_pools/<format>/manifest.json.",
    )
    parser.add_argument(
        "--team-pool-manifest",
        default=None,
        help="Shared manifest fallback for both model and opponent rosters.",
    )
    parser.add_argument(
        "--team-pool-subset",
        choices=["curated", "generated", "pool"],
        default="pool",
        help="Shared subset fallback for both model and opponent rosters.",
    )
    parser.add_argument(
        "--team-pool-seed",
        type=int,
        default=0,
        help="Shared pool-order seed fallback.",
    )
    parser.add_argument(
        "--model-team-pool-manifest",
        default=None,
        help="Independent manifest for the evaluated policy's fixed own roster.",
    )
    parser.add_argument(
        "--opponent-team-pool-manifest",
        default=None,
        help="Independent manifest for the fixed-bot opponent panel.",
    )
    parser.add_argument(
        "--model-team-pool-subset",
        choices=["curated", "generated", "pool"],
        default=None,
        help="Own-roster subset; defaults to --team-pool-subset.",
    )
    parser.add_argument(
        "--opponent-team-pool-subset",
        choices=["curated", "generated", "pool"],
        default=None,
        help="Opponent-panel subset; defaults to --team-pool-subset.",
    )
    parser.add_argument(
        "--model-team-pool-seed",
        type=int,
        default=None,
        help="Own-roster pool seed; defaults to --team-pool-seed.",
    )
    parser.add_argument(
        "--opponent-team-pool-seed",
        type=int,
        default=None,
        help="Opponent-panel pool seed; defaults to --team-pool-seed.",
    )
    parser.add_argument(
        "--model-team-title",
        default=None,
        help="Optional exact team title to use as the model's fixed 6-mon roster. Defaults to the strongest pool team.",
    )
    parser.add_argument(
        "--games", type=int, default=32, help="Number of evaluation battles."
    )
    parser.add_argument(
        "--action-mode",
        choices=["sample", "argmax"],
        default="argmax",
        help="Whether to sample from the policy or play greedily.",
    )
    parser.add_argument(
        "--preview-mode",
        choices=["learned", "max-bst", "random"],
        default="learned",
        help="Team-preview policy; battle actions always come from the checkpoint.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Python/Torch/numpy seed. Showdown battle RNG is not yet seed-controlled.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=200,
        help="Fail if a battle exceeds this many decisions.",
    )
    parser.add_argument(
        "--replay-dir",
        default=None,
        help="Optional output directory for representative replay HTML/JSON/trace artifacts.",
    )
    parser.add_argument(
        "--replay-count",
        type=int,
        default=1,
        help="How many completed battles to save as replay artifacts.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Optional path for the JSON evaluation summary.",
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


def main() -> None:
    args = parse_args()
    showdown_dir = resolve_showdown_dir(args.simulator, args.showdown_dir)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    env_config = EnvConfig(
        battle_format=args.battle_format,
        num_envs=1,
        team_size=6,
        pick_size=4,
        showdown_dir=showdown_dir,
        simulator=args.simulator,
        worker_count=args.worker_count,
        battles_per_worker=args.battles_per_worker,
        runtime_loop_count=args.runtime_loop_count,
    )
    model_config = ModelConfig(
        team_size=env_config.team_size,
        pick_size=env_config.pick_size,
    )
    game_data = GameData(gen=9)
    species_bst_lookup = build_species_bst_lookup(gen=game_data.gen)

    shared_manifest_path = (
        Path(args.team_pool_manifest)
        if args.team_pool_manifest
        else default_team_pool_manifest_path(REPO_ROOT, args.team_pool_format)
    )
    model_manifest_path = (
        Path(args.model_team_pool_manifest)
        if args.model_team_pool_manifest
        else shared_manifest_path
    )
    opponent_manifest_path = (
        Path(args.opponent_team_pool_manifest)
        if args.opponent_team_pool_manifest
        else shared_manifest_path
    )
    model_subset = args.model_team_pool_subset or args.team_pool_subset
    opponent_subset = args.opponent_team_pool_subset or args.team_pool_subset
    model_pool_seed = (
        args.team_pool_seed
        if args.model_team_pool_seed is None
        else args.model_team_pool_seed
    )
    opponent_pool_seed = (
        args.team_pool_seed
        if args.opponent_team_pool_seed is None
        else args.opponent_team_pool_seed
    )
    model_team_pool = load_team_pool(
        model_manifest_path,
        subset=model_subset,
        seed=model_pool_seed,
        usage="evaluation",
    )
    opponent_team_pool = load_team_pool(
        opponent_manifest_path,
        subset=opponent_subset,
        seed=opponent_pool_seed,
        usage="evaluation",
    )
    if args.model_team_title:
        model_team_spec = select_team_spec_by_title(
            model_team_pool,
            title=args.model_team_title,
            species_bst_lookup=species_bst_lookup,
            pick_size=env_config.pick_size,
        )
    else:
        model_team_spec = select_max_bst_team_spec(
            model_team_pool,
            species_bst_lookup=species_bst_lookup,
            pick_size=env_config.pick_size,
        )
    checkpoint_ref = (
        None if args.checkpoint is None else CheckpointRef.from_file(args.checkpoint)
    )
    checkpoint_path = (
        None if checkpoint_ref is None else str(checkpoint_ref.resolve(REPO_ROOT))
    )
    agent = load_agent_for_evaluation(
        checkpoint_path=checkpoint_path,
        device=device,
        model_config=model_config,
        game_data=game_data,
    )

    results = evaluate_policy_vs_fixed_bot(
        agent=agent,
        device=device,
        env_config=env_config,
        model_team_spec=model_team_spec,
        opponent_team_pool=opponent_team_pool,
        species_bst_lookup=species_bst_lookup,
        num_games=args.games,
        action_mode=args.action_mode,
        seed=args.seed,
        max_turns=args.max_turns,
        opponent_kind=args.opponent,
        preview_mode=args.preview_mode,
        replay_dir=args.replay_dir,
        replay_count=args.replay_count,
    )
    if checkpoint_ref is not None:
        checkpoint_ref.verify(REPO_ROOT)
    summary = {
        "config": {
            "checkpoint": None if checkpoint_ref is None else checkpoint_ref.to_dict(),
            "device": str(device),
            "battle_format": env_config.battle_format,
            "simulator": env_config.simulator,
            "showdown_dir": env_config.showdown_dir,
            "games": args.games,
            "action_mode": args.action_mode,
            "preview_mode": args.preview_mode,
            "opponent": args.opponent,
            "team_size": env_config.team_size,
            "pick_size": env_config.pick_size,
            "seed": args.seed,
            "simulator_rng_seeded": False,
            "team_pool_format": args.team_pool_format,
            "model_team_pool_manifest": str(model_manifest_path.resolve()),
            "model_team_pool_subset": model_subset,
            "model_team_pool_size": model_team_pool.size,
            "model_team_pool_signature": model_team_pool.signature(),
            "opponent_team_pool_manifest": str(opponent_manifest_path.resolve()),
            "opponent_team_pool_subset": opponent_subset,
            "opponent_team_pool_size": opponent_team_pool.size,
            "opponent_team_pool_signature": opponent_team_pool.signature(),
            "model_team_title": model_team_spec.title,
            "model_signature": build_model_signature(
                model_config,
                game_data.vocab_signature(),
            ),
        },
        "model_team": model_team_spec.to_summary(),
        "results": results,
    }

    if args.summary_path:
        out_path = _write_json(args.summary_path, summary)
        print(f"Wrote summary to {out_path}")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
